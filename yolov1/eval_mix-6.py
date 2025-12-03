import os
import torch
import torch.nn as nn
import numpy as np
import tqdm
import argparse
import torchvision
from torchvision.ops import nms

from nets.nn import resnet50                    # 너의 YOLOv1 모델
from utils.dataloader import YOLOv1Dataset      # 같은 DataLoader 사용


# ==========================================================
# YOLOv1 Output → Bounding Boxes 변환
# ==========================================================
def decode_yolo_output(pred, S=7, B=2, C=20, conf_thresh=0.1):
    """
    pred: (7,7,30)
    return: [ [x1,y1,x2,y2,score,class] , ... ]
    """
    boxes = []

    cell_size = 1.0 / S

    for i in range(S):
        for j in range(S):
            cell = pred[i, j]

            cls_probs = cell[5:]
            class_id = torch.argmax(cls_probs).item()
            class_prob = cls_probs[class_id].item()

            # B=2 boxes
            for b in range(B):
                bx = cell[b*5 + 0]
                by = cell[b*5 + 1]
                bw = cell[b*5 + 2]
                bh = cell[b*5 + 3]
                conf = cell[b*5 + 4]

                score = conf.item() * class_prob
                if score < conf_thresh:
                    continue

                cx = (j + bx.item()) * cell_size
                cy = (i + by.item()) * cell_size
                w = bw.item()
                h = bh.item()

                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2

                boxes.append([x1, y1, x2, y2, score, class_id])

    return boxes


# ==========================================================
# IoU 계산
# ==========================================================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


# ==========================================================
# mAP 계산 (VOC 2007 방식)
# ==========================================================
def compute_map(pred_boxes, gt_boxes, iou_threshold=0.5, num_classes=20):
    aps = []

    for cls in range(num_classes):
        cls_preds = [p for p in pred_boxes if p[5] == cls]
        cls_gts   = [g for g in gt_boxes if g[5] == cls]

        if len(cls_gts) == 0:
            continue

        cls_preds = sorted(cls_preds, key=lambda x: x[4], reverse=True)

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        matched = []

        for i, pred in enumerate(cls_preds):
            max_iou = 0
            max_gt = None

            for gt in cls_gts:
                iou = compute_iou(pred, gt)
                if iou > max_iou:
                    max_iou = iou
                    max_gt = gt

            if max_iou >= iou_threshold and max_gt not in matched:
                tp[i] = 1
                matched.append(max_gt)
            else:
                fp[i] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recalls = tp / (len(cls_gts) + 1e-6)
        precisions = tp / (tp + fp + 1e-6)

        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        aps.append(ap)

    return np.mean(aps) if aps else 0


# ==========================================================
# Default Weight Auto Loader (best → last)
# ==========================================================
def load_default_weights(model, device):
    weight_list = ["best_mix_6.pth", "last_mix_6.pth"]

    for w in weight_list:
        if os.path.exists(w):
            print(f"✔ Default weight found: {w}")
            model.load_state_dict(torch.load(w, map_location=device))
            return w

    raise FileNotFoundError(
        "\n[ERROR] No weight file found!\n"
        "Expected one of:\n"
        "  best_mix_6.pth\n"
        "  last_mix_6.pth\n"
    )


# ==========================================================
# Main
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--S", type=int, default=7)
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--C", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    # ==========================================================
    # Load Model
    # ==========================================================
    model = resnet50().to(device)

    if args.weights is None:
        loaded = load_default_weights(model, device)
        print(f"Loaded (auto): {loaded}")
    else:
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"[ERROR] Weight file not found: {args.weights}")
        print(f"Loaded (manual): {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.eval()

    # ==========================================================
    # Dataset
    # ==========================================================
    dataset = YOLOv1Dataset(
        img_dir="./Dataset/Images",
        label_dir="./Dataset/Labels",
        list_file="./Dataset/test.txt",
        S=args.S, B=args.B, C=args.C,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((448, 448)),
            torchvision.transforms.ToTensor()
        ])
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    pred_boxes = []
    gt_boxes = []

    # ==========================================================
    # Evaluation Loop
    # ==========================================================
    print("\nEvaluating...")
    for imgs, targets in tqdm.tqdm(loader):
        imgs = imgs.to(device)

        with torch.no_grad():
            preds = model(imgs)   # (1,7,7,30)

        preds = preds.squeeze(0).cpu()
        decoded = decode_yolo_output(
            preds,
            S=args.S, B=args.B, C=args.C,
            conf_thresh=args.conf
        )
        pred_boxes.extend(decoded)

        # GT 변환
        ts = targets.squeeze(0)
        S = args.S

        for i in range(S):
            for j in range(S):
                if ts[i, j, 4] == 1:
                    cx = (j + ts[i, j, 0].item()) / S
                    cy = (i + ts[i, j, 1].item()) / S
                    w  = ts[i, j, 2].item()
                    h  = ts[i, j, 3].item()
                    cls = torch.argmax(ts[i, j, 5:]).item()

                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2

                    gt_boxes.append([x1, y1, x2, y2, 1.0, cls])

    # ==========================================================
    # Compute mAP
    # ==========================================================
    mAP = compute_map(pred_boxes, gt_boxes, iou_threshold=0.5, num_classes=args.C)

    print("\n==========================")
    print(f" mAP (IoU=0.5): {mAP:.4f}")
    print("==========================\n")


if __name__ == "__main__":
    main()
