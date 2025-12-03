import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from main_mix_1 import ResNet101_CBAM   # ← main_mix_1 전용 모델 사용
from utils.loss_mix_1 import YoloLoss   # loss만 불러오기

VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
    'cow','diningtable','dog','horse','motorbike','person','pottedplant',
    'sheep','sofa','train','tvmonitor'
]

# ======================================================
# YOLOv1 Decoder (local version)
# ======================================================
def decoder(pred, conf_thres=0.05):
    """
    pred: (7,7,30) raw linear output from main_mix_1
    A-style stabilization decoder
    """
    S = 7
    B = 2
    C = 20

    boxes = []
    cls_indexes = []
    confidences = []

    for gy in range(S):
        for gx in range(S):
            cell = pred[gy, gx]   # (30,)

            class_raw = cell[:C]
            box_raw   = cell[C:].view(B,5)

            # --- class score: softmax ---
            cls_prob = torch.softmax(class_raw, dim=-1)
            cls_id = torch.argmax(cls_prob).item()
            cls_score = cls_prob[cls_id].item()

            for b in range(B):
                px, py, pw, ph, conf = box_raw[b]

                # --- x,y,w,h,conf stabilization ---
                px = torch.sigmoid(px)          # 0~1
                py = torch.sigmoid(py)          # 0~1

                pw = torch.sigmoid(pw) * 1.5    # YOLOv1-style scale
                ph = torch.sigmoid(ph) * 1.5

                conf = torch.sigmoid(conf).item()

                final_conf = conf * cls_score

                if final_conf < conf_thres:
                    continue

                # --- Convert to absolute values ---
                cx = (gx + px.item()) / S
                cy = (gy + py.item()) / S

                w  = pw.item() / S
                h  = ph.item() / S

                x1 = (cx - w/2) * 448
                y1 = (cy - h/2) * 448
                x2 = (cx + w/2) * 448
                y2 = (cy + h/2) * 448

                boxes.append([(x1, y1), (x2, y2)])
                cls_indexes.append(cls_id)
                confidences.append(final_conf)

    # --- fallback: if no boxes, relax threshold ---
    if len(boxes) == 0 and conf_thres > 0.001:
        return decoder(pred, conf_thres=conf_thres * 0.5)

    return boxes, cls_indexes, confidences

# ======================================================
# predict() for mix_1
# ======================================================
def predict_mix(model, image_name, root_path='./Dataset/Images/'):
    img_path = os.path.join(root_path, image_name)
    img = cv2.imread(img_path)
    if img is None:
        return []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (448,448))
    img = img.astype(np.float32)/255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda()

    pred = model(img)[0]   # shape = (1470,)
    pred = pred.view(7,7,30)

    boxes, cls_idxs, confs = decoder(pred)

    results = []
    for box, c, conf in zip(boxes, cls_idxs, confs):
        (x1,y1), (x2,y2) = box

        results.append([
            (int(x1),int(y1)),
            (int(x2),int(y2)),
            VOC_CLASSES[c],
            image_name,
            float(conf)
        ])

    return results


# ======================================================
# Evaluation (same printing format as eval.py)
# ======================================================
class Evaluation:
    def __init__(self, preds, targets, threshold=0.5):
        self.preds = preds
        self.targets = targets
        self.threshold = threshold

    @staticmethod
    def compute_ap(recall, precision):
        recall = np.concatenate(([0.], recall, [1.]))
        precision = np.concatenate(([0.], precision,[0.]))

        for i in range(len(precision)-2,-1,-1):
            precision[i] = max(precision[i], precision[i+1])

        ap = 0.0
        for i in range(len(recall)-1):
            ap += (recall[i+1]-recall[i]) * precision[i+1]
        return ap

    def evaluate(self):
        aps = []
        print("CLASS".ljust(25), "AP")

        for cls in VOC_CLASSES:
            cls_preds = self.preds[cls]

            if len(cls_preds)==0:
                print(f"---class {cls} ap -1---")
                aps.append(-1)
                continue

            img_ids = [x[0] for x in cls_preds]
            conf    = np.array([x[1] for x in cls_preds])
            BB      = np.array([x[2:] for x in cls_preds])

            sidx = np.argsort(-conf)
            conf = conf[sidx]
            BB   = BB[sidx]
            img_ids = [img_ids[i] for i in sidx]

            # Count GTs
            npos = sum([
                len(self.targets[(img,cls)]) for (img,c) in self.targets if c==cls
            ])

            tp = np.zeros(len(img_ids))
            fp = np.zeros(len(img_ids))

            for i, img_id in enumerate(img_ids):
                bb = BB[i]

                if (img_id,cls) in self.targets:
                    gts = self.targets[(img_id,cls)]
                    hit=False
                    for gt in list(gts):
                        x1=max(gt[0],bb[0])
                        y1=max(gt[1],bb[1])
                        x2=min(gt[2],bb[2])
                        y2=min(gt[3],bb[3])

                        iw = max(x2-x1+1,0)
                        ih = max(y2-y1+1,0)
                        inter = iw*ih
                        union = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) + \
                                (gt[2]-gt[0]+1)*(gt[3]-gt[1]+1) - inter

                        iou = inter/union

                        if iou > self.threshold:
                            tp[i]=1
                            gts.remove(gt)
                            hit=True
                            break

                    if not hit:
                        fp[i]=1
                else:
                    fp[i]=1

            recall = np.cumsum(tp)/max(npos,1)
            precision = np.cumsum(tp)/(np.cumsum(tp)+np.cumsum(fp)+1e-6)

            ap = self.compute_ap(recall,precision)

            print(cls.ljust(25), f"{ap:.2f}")
            aps.append(ap)

        return aps


# ======================================================
# Main Eval
# ======================================================
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GT load
    targets = defaultdict(list)
    preds   = defaultdict(list)
    image_list = []

    print("DATA PREPARING...")
    with open("./Dataset/test.txt") as f:
        lines = f.readlines()

    for line in lines:
        name = line.strip()
        img = name + ".jpg"
        image_list.append(img)

        with open(f"./Dataset/Labels/{name}.txt") as f:
            for obj in f:
                c,x1,y1,x2,y2 = map(int,obj.split())
                cls = VOC_CLASSES[c]
                targets[(img,cls)].append([x1,y1,x2,y2])

    print("DONE.\n")
    print("START TESTING...")

    model = ResNet101_CBAM().to(device)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load("./weights/yolov1_mix_1_final.pth")["state_dict"])
    model.eval()

    with torch.no_grad():
        for img_name in tqdm(image_list):
            result = predict_mix(model,img_name)

            for (x1,y1),(x2,y2),cls,img,conf in result:
                preds[cls].append([img,conf,x1,y1,x2,y2])

    print("\nSTART EVALUATION...")

    aps = Evaluation(preds, targets, 0.5).evaluate()

    print(f"\nmAP: {np.mean(aps):.2f}")
    print("\nDONE.")
