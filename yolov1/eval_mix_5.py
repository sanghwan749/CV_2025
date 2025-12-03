# ================================================================
# eval_mix_5.py  (YOLO mix_5 loss 구조에 정확히 맞춘 평가 코드)
# ================================================================
import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
from nets.nn import resnet50


VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
    'cow','diningtable','dog','horse','motorbike','person','pottedplant',
    'sheep','sofa','train','tvmonitor'
]


# --------------------------------------------------------
# IoU (xyxy)
# --------------------------------------------------------
def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h

    area_a = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
    area_b = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))

    union = area_a + area_b - inter + 1e-6
    return inter / union


# --------------------------------------------------------
# NMS
# --------------------------------------------------------
def nms(boxes, scores, th=0.5):
    idx = np.argsort(scores)[::-1]
    keep = []

    while len(idx) > 0:
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break

        rest = idx[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        idx = rest[ious < th]

    return keep


# --------------------------------------------------------
# predict_mix5(): loss_mix_5 구조에 정확히 맞춘 디코딩
# --------------------------------------------------------
def predict_mix5(model, image_name, root_path="./Dataset/Images/"):
    img_path = os.path.join(root_path, image_name)
    img = cv2.imread(img_path)
    if img is None:
        return []

    H, W = img.shape[:2]
    resized = cv2.resize(img, (448, 448))
    rgb = resized[:, :, ::-1].copy()

    inp = torch.from_numpy(rgb.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    inp = inp.to(next(model.parameters()).device)

    with torch.no_grad():
        out = model(inp)      # (1, 14,14,30)
    out = out[0]

    S = 14
    out = out.view(S, S, 30)

    box = out[..., :10].view(S, S, 2, 5)
    cls_logit = out[..., 10:]       # (14,14,20)

    cls_prob = torch.sigmoid(cls_logit).cpu().numpy()
    box = box.cpu().numpy()

    boxes = []
    scores = []
    labels = []

    for gy in range(S):
        for gx in range(S):
            for b in range(2):
                tx, ty, tw, th, conf_raw = box[gy, gx, b]

                # confidence(raw) → sigmoid
                conf = 1 / (1 + np.exp(-conf_raw))

                # threshold 매우 낮게 조정
                if conf < 0.0001:
                    continue

                cx = (1/(1+np.exp(-tx)) + gx) / S
                cy = (1/(1+np.exp(-ty)) + gy) / S
                w = max(tw, 1e-6)
                h = max(th, 1e-6)

                x1 = (cx - w/2) * W
                y1 = (cy - h/2) * H
                x2 = (cx + w/2) * W
                y2 = (cy + h/2) * H

                c = np.argmax(cls_prob[gy, gx])
                cls_conf = cls_prob[gy, gx, c]

                final = conf * cls_conf
                if final < 0.0001:
                    continue

                boxes.append([x1, y1, x2, y2])
                scores.append(final)
                labels.append(VOC_CLASSES[c])

    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    keep = nms(boxes, scores, 0.5)

    results = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cls = labels[i]
        conf = scores[i]
        results.append([(x1,y1),(x2,y2),cls,image_name,conf])

    return results


# --------------------------------------------------------
# AP 계산
# --------------------------------------------------------
def voc_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i+1]-mrec[i]) * mpre[i+1])
    return ap


def iou_xyxy_np(a, b):
    xx1 = np.maximum(a[0], b[:, 0])
    yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2])
    yy2 = np.minimum(a[3], b[:, 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h

    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])

    union = area_a + area_b - inter + 1e-6
    return inter / union


# --------------------------------------------------------
# Evaluation
# --------------------------------------------------------
def evaluate_MAP(preds, gts, iou_th=0.5):
    APs = []
    print("CLASS".ljust(20), "AP")

    for cls in VOC_CLASSES:
        dets = preds[cls]

        gt_for_cls = {}
        gt_det = {}
        npos = 0

        for (img, label), arr in gts.items():
            if label != cls:
                continue
            gt_for_cls[img] = arr
            gt_det[img] = np.zeros(len(arr))
            npos += len(arr)

        if len(dets) == 0:
            print(cls.ljust(20), "0.00")
            APs.append(0.0)
            continue

        dets.sort(key=lambda x: -x[1])
        tp, fp = [], []

        for (img_id, conf, x1, y1, x2, y2) in dets:
            bb = np.array([x1, y1, x2, y2])

            if img_id in gt_for_cls:
                gts_arr = gt_for_cls[img_id]
                ious = iou_xyxy_np(bb, gts_arr)
                j = np.argmax(ious)
                iou_max = ious[j]

                if iou_max >= iou_th and not gt_det[img_id][j]:
                    tp.append(1)
                    fp.append(0)
                    gt_det[img_id][j] = 1
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                tp.append(0)
                fp.append(1)

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recall = tp / max(npos, 1)
        precision = tp / np.maximum(tp+fp, 1e-6)

        ap = voc_ap(recall, precision)
        APs.append(ap)
        print(cls.ljust(20), f"{ap:.2f}")

    return np.mean(APs)



# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    data_dir = "./Dataset"
    weight_path = "./weights/yolov1_mix_5_final.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GT
    gts = defaultdict(list)
    img_list = []

    with open(os.path.join(data_dir,"test.txt")) as f:
        stems = [x.strip() for x in f.readlines()]

    for s in stems:
        imgname = f"{s}.jpg"
        img_list.append(imgname)
        with open(os.path.join(data_dir,"Labels",f"{s}.txt")) as lf:
            for line in lf:
                c,x1,y1,x2,y2 = map(int, line.split())
                gts[(imgname, VOC_CLASSES[c])].append([x1,y1,x2,y2])

    gts = {k:np.array(v,float) for k,v in gts.items()}

    # Load model
    model = resnet50().to(device)
    sd = torch.load(weight_path, map_location=device)
    model.load_state_dict(sd["state_dict"])
    model.eval()

    # Predict
    preds = defaultdict(list)
    print("START TESTING...")
    with torch.no_grad():
        for img in tqdm(img_list):
            dets = predict_mix5(model, img, root_path=os.path.join(data_dir,"Images"))
            for (x1,y1),(x2,y2),cls,imgid,conf in dets:
                preds[cls].append([imgid, conf, x1, y1, x2, y2])

    print("\nSTART EVALUATION...")
    mAP = evaluate_MAP(preds, gts)
    print("\n=> mAP:", mAP)
