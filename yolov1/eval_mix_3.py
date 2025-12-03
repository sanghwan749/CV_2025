import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import defaultdict
from tqdm import tqdm

from main_mix_3 import YOLO_ResNet_CBAM, GRID, IMG_SIZE, B, C

VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
    'cow','diningtable','dog','horse','motorbike','person','pottedplant',
    'sheep','sofa','train','tvmonitor'
]

# ======================================================
# YOLOv1 decoder for mix_3 model
# ======================================================
def decode_yolo(pred, conf_thres=0.1, nms_iou=0.4):
    """
    pred shape: (7,7,30)
    """

    S = GRID
    BBOXES = B
    CLASSES = C

    pred = pred.cpu()

    class_logits = pred[..., :CLASSES]
    pred_boxes = pred[..., CLASSES:].view(S, S, BBOXES, 5)

    # Activation
    cls_prob = torch.softmax(class_logits, dim=-1)      # (7,7,20)
    xy = torch.sigmoid(pred_boxes[..., 0:2])            # (7,7,2,2)
    wh = pred_boxes[..., 2:4] ** 2                      # square(w,h)
    conf = torch.sigmoid(pred_boxes[..., 4])            # (7,7,2)

    boxes = []
    scores = []
    cls_idxs = []

    for i in range(S):
        for j in range(S):
            for b in range(BBOXES):
                x, y = xy[i, j, b]
                w, h = wh[i, j, b]
                c = conf[i, j, b]

                cx = (j + x.item()) / S
                cy = (i + y.item()) / S

                bw = w.item()
                bh = h.item()

                x1 = (cx - bw / 2) * IMG_SIZE
                y1 = (cy - bh / 2) * IMG_SIZE
                x2 = (cx + bw / 2) * IMG_SIZE
                y2 = (cy + bh / 2) * IMG_SIZE

                # class 선택
                cls_id = torch.argmax(cls_prob[i, j]).item()
                cls_score = cls_prob[i, j, cls_id].item()

                final_conf = cls_score * c.item()
                if final_conf < conf_thres:
                    continue

                boxes.append([x1, y1, x2, y2])
                scores.append(final_conf)
                cls_idxs.append(cls_id)

    if len(boxes) == 0:
        return []

    # NMS
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    cls_idxs = torch.tensor(cls_idxs)

    keep_idx = torchvision.ops.nms(boxes, scores, nms_iou)

    result = []
    for idx in keep_idx:
        x1, y1, x2, y2 = boxes[idx]
        cls_id = cls_idxs[idx]
        conf = scores[idx]

        result.append([
            (float(x1), float(y1)),
            (float(x2), float(y2)),
            VOC_CLASSES[int(cls_id)],
            float(conf)
        ])

    return result


# ======================================================
# AP evaluation
# ======================================================
class Evaluation:
    def __init__(self, preds, targets, threshold=0.5):
        self.preds = preds
        self.targets = targets
        self.threshold = threshold

    @staticmethod
    def compute_ap(rec, prec):
        rec = np.concatenate(([0.], rec, [1.]))
        prec = np.concatenate(([0.], prec, [0.]))

        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])

        ap = 0.0
        for i in range(len(rec) - 1):
            ap += (rec[i + 1] - rec[i]) * prec[i + 1]

        return ap

    def evaluate(self):
        aps = []
        print("CLASS".ljust(25), "AP")

        for cls in VOC_CLASSES:
            cls_preds = self.preds[cls]

            if len(cls_preds) == 0:
                print(f"---class {cls} ap -1---")
                aps.append(-1)
                continue

            img_ids = [p[0] for p in cls_preds]
            conf = np.array([p[1] for p in cls_preds])
            BB = np.array([p[2:] for p in cls_preds])

            # sort by conf desc
            sidx = np.argsort(-conf)
            conf = conf[sidx]
            BB = BB[sidx]
            img_ids = [img_ids[i] for i in sidx]

            npos = sum([
                len(self.targets.get((img, cls), []))
                for img in set(img_ids)
            ])

            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))

            for i, img in enumerate(img_ids):
                bb = BB[i]

                if (img, cls) in self.targets:
                    matched = False
                    for gt in list(self.targets[(img, cls)]):

                        x1 = max(bb[0], gt[0])
                        y1 = max(bb[1], gt[1])
                        x2 = min(bb[2], gt[2])
                        y2 = min(bb[3], gt[3])

                        w = max(x2 - x1, 0)
                        h = max(y2 - y1, 0)
                        inter = w * h

                        union = (
                            (bb[2] - bb[0]) * (bb[3] - bb[1]) +
                            (gt[2] - gt[0]) * (gt[3] - gt[1]) -
                            inter
                        )

                        iou = inter / (union + 1e-6)

                        if iou >= self.threshold:
                            tp[i] = 1
                            matched = True
                            self.targets[(img, cls)].remove(gt)
                            break

                    if not matched:
                        fp[i] = 1
                else:
                    fp[i] = 1

            rec = np.cumsum(tp) / (npos + 1e-6)
            prec = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp) + 1e-6)

            ap = self.compute_ap(rec, prec)

            print(cls.ljust(25), f"{ap:.2f}")
            aps.append(ap)

        return aps


# ======================================================
# EVALUATION MAIN
# ======================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("DATA PREPARING...")
    with open("./Dataset/test.txt") as f:
        test_list = [x.strip() for x in f]

    targets = defaultdict(list)

    # load GT
    for name in test_list:
        img = name + ".jpg"
        with open(f"./Dataset/Labels/{name}.txt") as f:
            for line in f:
                c, x1, y1, x2, y2 = map(int, line.split())
                cls = VOC_CLASSES[c]
                targets[(img, cls)].append([x1, y1, x2, y2])

    print("DONE.")
    print("START TESTING...")

    model = YOLO_ResNet_CBAM().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load("./weights/yolov1_mix3_final.pth")["state_dict"])
    model.eval()

    from torchvision import transforms

    predictions = defaultdict(list)

    with torch.no_grad():
        for name in tqdm(test_list):
            img_name = name + ".jpg"
            path = "./Dataset/Images/" + img_name

            img_bgr = cv2.imread(path)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img = img.to(device)

            pred = model(img)[0]  # (7,7,30)
            decoded = decode_yolo(pred)

            for (x1, y1), (x2, y2), cls, conf in decoded:
                predictions[cls].append([img_name, conf, x1, y1, x2, y2])

    print("\nSTART EVALUATION...")
    aps = Evaluation(predictions, targets).evaluate()

    print(f"\nmAP: {np.mean(aps):.2f}")
    print("\nDONE.")
