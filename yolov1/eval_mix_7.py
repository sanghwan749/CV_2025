# eval_mix_7.py
import os
import cv2
import numpy as np
import torch
from collections import defaultdict
from utils.util import predict


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]


# =============================================================
# IoU 계산 함수 (xyxy)
# =============================================================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-7
    return inter / union


# =============================================================
# 평가 클래스
# =============================================================
class Evaluation:
    def __init__(self, predictions, targets, iou_thresh=0.5):
        self.predictions = predictions   # {class: [[img,score,x1,y1,x2,y2], ...]}
        self.targets = targets           # {(image,class): [boxes]}
        self.iou_thresh = iou_thresh

    def compute_ap(self, rec, prec):
        rec = np.concatenate(([0.], rec, [1.]))
        prec = np.concatenate(([0.], prec, [0.]))

        # precision 누적 내림 처리
        for i in range(len(prec) - 1, 0, -1):
            prec[i - 1] = max(prec[i - 1], prec[i])

        ap = 0.0
        for i in range(len(prec) - 1):
            ap += (rec[i + 1] - rec[i]) * prec[i + 1]
        return ap

    def evaluate(self):
        aps = {}

        for cls in VOC_CLASSES:
            preds = self.predictions[cls]      # list
            gts = [(k, v) for k, v in self.targets.items() if k[1] == cls]

            npos = sum([len(v) for _, v in gts])
            if npos == 0:
                aps[cls] = -1
                continue

            # GT matched mask
            gt_used = {k: np.zeros(len(v)) for k, v in gts}

            # prediction sorting
            preds_sorted = sorted(preds, key=lambda x: -x[1])

            tp = []
            fp = []

            for (img, conf, x1, y1, x2, y2) in preds_sorted:
                if (img, cls) in self.targets:
                    gt_boxes = self.targets[(img, cls)]
                    ious = [compute_iou([x1, y1, x2, y2], gt) for gt in gt_boxes]

                    max_iou = max(ious)
                    idx = np.argmax(ious)

                    if max_iou >= self.iou_thresh:
                        if gt_used[(img, cls)][idx] == 0:
                            tp.append(1)
                            fp.append(0)
                            gt_used[(img, cls)][idx] = 1
                        else:
                            tp.append(0)
                            fp.append(1)
                    else:
                        tp.append(0)
                        fp.append(1)
                else:
                    tp.append(0)
                    fp.append(1)

            tp = np.cumsum(tp)
            fp = np.cumsum(fp)

            rec = tp / max(npos, 1e-6)
            prec = tp / np.maximum(tp + fp, 1e-7)

            ap = self.compute_ap(rec, prec)
            aps[cls] = ap

        return aps


# =============================================================
# Main Evaluation
# =============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    targets = defaultdict(list)
    predictions = defaultdict(list)

    # GT 읽기
    with open("./Dataset/test.txt") as f:
        lines = f.readlines()

    image_list = []

    for line in lines:
        name = line.strip()
        img_name = f"{name}.jpg"
        image_list.append(img_name)

        with open(f"./Dataset/Labels/{name}.txt") as f2:
            for obj in f2:
                c, x1, y1, x2, y2 = map(int, obj.split())
                cls_name = VOC_CLASSES[c]
                targets[(img_name, cls_name)].append([x1, y1, x2, y2])

    print("Loading model...")

    from nets.nn import resnet50
    model = resnet50().to(device)
    model.load_state_dict(torch.load("./weights/yolov1_mix_7_final.pth")["state_dict"])
    model.eval()

    print("Running inference...")

    with torch.no_grad():
        for img in image_list:
            result = predict(model, img, root_path="./Dataset/Images/")
            for (pt1, pt2, cls, image_name, conf) in result:
                x1, y1 = pt1
                x2, y2 = pt2
                predictions[cls].append([image_name, conf, x1, y1, x2, y2])

    print("Evaluating...")

    evaluator = Evaluation(predictions, targets)
    aps = evaluator.evaluate()

    for cls in VOC_CLASSES:
        print(f"{cls:15s}: {aps[cls]:.4f}")

    valid_aps = [v for v in aps.values() if v >= 0]
    print(f"\n mAP: {np.mean(valid_aps):.4f}")
