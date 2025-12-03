# ================================================================
# eval_mix_8.py — YOLOv1 mix_8 Evaluation (VOC mAP)
# ================================================================
import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
from nets.nn import resnet50


VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor'
]


# ---------------------- IoU --------------------------
def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h

    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    union = area_a + area_b - inter + 1e-6
    return inter / union


# ---------------------- NMS --------------------------
def nms(boxes, scores, iou_th=0.5):
    idx = np.argsort(scores)[::-1]
    keep = []
    while len(idx) > 0:
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break
        rest = idx[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        idx = rest[ious < iou_th]
    return keep


# ---------------------- Predict --------------------------
def predict_mix_8(model, img_name, root_path="./Dataset/Images"):
    path = os.path.join(root_path, img_name)
    img = cv2.imread(path)
    if img is None:
        return []

    H, W = img.shape[:2]

    resized = cv2.resize(img, (448, 448))[:, :, ::-1].copy()
    inp = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    inp = inp.to(next(model.parameters()).device)

    with torch.no_grad():
        pred = model(inp)[0].cpu().numpy().reshape(14, 14, 30)

    S = 14
    boxes, scores, labels = [], [], []

    for gy in range(S):
        for gx in range(S):

            tx, ty, sw, sh, conf = pred[gy, gx, 0:5]

            if conf < 0.01:
                continue

            cx = (tx + gx) / S
            cy = (ty + gy) / S
            w = sw ** 2
            h = sh ** 2

            # convert back to original image scale
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)

            cls_scores = pred[gy, gx, 10:]
            cls = np.argmax(cls_scores)
            cls_prob = cls_scores[cls]

            final_score = conf * cls_prob
            if final_score < 0.01:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(final_score)
            labels.append(cls)

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    keep = nms(boxes, scores)
    results = []

    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cls = labels[i]
        conf = scores[i]
        results.append([img_name, cls, conf, x1, y1, x2, y2])

    return results


# ---------------------- AP --------------------------
def voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
    return ap


# ---------------------- Evaluate --------------------------
def evaluate(preds, gts, iou_th=0.5):
    APs = []
    print("CLASS".ljust(15), " AP")

    for ci, cls_name in enumerate(VOC_CLASSES):
        dets = preds[cls_name]

        gt_for_cls = {}
        gt_hit = {}
        npos = 0

        for (img, c), bboxes in gts.items():
            if c != cls_name:
                continue
            gt_for_cls.setdefault(img, [])
            gt_for_cls[img].extend(bboxes)
            gt_hit[img] = [False] * len(bboxes)
            npos += len(bboxes)

        if len(dets) == 0 or npos == 0:
            print(cls_name.ljust(15), "0.00")
            APs.append(0.0)
            continue

        dets.sort(key=lambda x: -x[2])
        TP, FP = [], []

        for img, cls_idx, score, x1, y1, x2, y2 in dets:
            box = [x1, y1, x2, y2]

            if img in gt_for_cls:
                gts_img = gt_for_cls[img]
                ious = np.array([iou_xyxy(box, g) for g in gts_img])
                j = np.argmax(ious)
                iou_max = ious[j]

                if iou_max >= iou_th and not gt_hit[img][j]:
                    TP.append(1)
                    FP.append(0)
                    gt_hit[img][j] = True
                else:
                    TP.append(0)
                    FP.append(1)
            else:
                TP.append(0)
                FP.append(1)

        TP = np.cumsum(TP)
        FP = np.cumsum(FP)

        rec = TP / (npos + 1e-6)
        prec = TP / np.maximum(TP + FP, 1e-6)

        ap = voc_ap(rec, prec)
        APs.append(ap)
        print(cls_name.ljust(15), f"{ap:.4f}")

    return np.mean(APs)


# ---------------------- MAIN --------------------------
if __name__ == "__main__":
    data_dir = "./Dataset"
    weight_path = "./weights/yolov1_mix_8_final.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- GT ----------------
    gts = defaultdict(list)
    img_list = []
    with open(os.path.join(data_dir, "test.txt")) as f:
        stems = [l.strip() for l in f.readlines()]

    for s in stems:
        img_name = f"{s}.jpg"
        img_list.append(img_name)
        with open(os.path.join(data_dir, "Labels", f"{s}.txt")) as lf:
            for line in lf:
                c, x1, y1, x2, y2 = map(int, line.split())
                gts[(img_name, VOC_CLASSES[c])].append([x1,y1,x2,y2])

    # ---------------- Model ----------------
    model = resnet50().to(device)
    sd = torch.load(weight_path, map_location=device)
    sd = sd['state_dict'] if 'state_dict' in sd else sd
    model.load_state_dict(sd)
    model.eval()

    # ---------------- Predict ----------------
    preds = defaultdict(list)
    print("START TESTING...")

    with torch.no_grad():
        for img_name in tqdm(img_list):
            dets = predict_mix_8(model, img_name, root_path=os.path.join(data_dir, "Images"))
            for item in dets:
                preds[VOC_CLASSES[item[1]]].append(item)

    # ---------------- Evaluate ----------------
    print("\nSTART EVALUATION...")
    mAP = evaluate(preds, gts)
    print("\n=> mAP:", mAP)
    print("DONE.")
