import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import defaultdict
from tqdm import tqdm

from main_mix_2 import Model, IMG_SIZE, GRID, CLASSES   # main_mix_2 구조 그대로 사용

VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
    'cow','diningtable','dog','horse','motorbike','person','pottedplant',
    'sheep','sofa','train','tvmonitor'
]

# ======================================================
# Modern YOLOv1 Decoder (14x14 grid)
# ======================================================
def decode_prediction(pred, conf_thres=0.1):
    S = GRID
    C = CLASSES
    B = 2

    boxes = []
    cls_idxs = []
    confs = []

    for gy in range(S):
        for gx in range(S):
            cell = pred[gy, gx]             # shape = (30,)
            box_data = cell[:10].view(2,5)  # each = (x,y,w,h,conf)
            cls_logits = cell[10:]

            # softmax class prob
            cls_prob = torch.softmax(cls_logits, dim=-1)
            cls_id = torch.argmax(cls_prob).item()
            cls_score = cls_prob[cls_id].item()

            for b in range(B):
                px, py, pw, ph, conf = box_data[b]

                conf = torch.sigmoid(conf).item()
                final_conf = conf * cls_score
                if final_conf < conf_thres:
                    continue

                cx = (gx + px.item()) / S
                cy = (gy + py.item()) / S
                w  = pw.item()
                h  = ph.item()

                x1 = (cx - w/2) * IMG_SIZE
                y1 = (cy - h/2) * IMG_SIZE
                x2 = (cx + w/2) * IMG_SIZE
                y2 = (cy + h/2) * IMG_SIZE

                boxes.append([(x1,y1),(x2,y2)])
                cls_idxs.append(cls_id)
                confs.append(final_conf)

    return boxes, cls_idxs, confs


# ======================================================
# Prediction Wrapper
# ======================================================
def predict_mix2(model, image_name, root="./Dataset/Images/"):
    path = os.path.join(root, image_name)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return []

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)/255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda()

    pred = model(img)[0]     # (14,14,30)
    boxes, cls_idxs, confs = decode_prediction(pred)

    results = []
    for box, c, conf in zip(boxes, cls_idxs, confs):
        (x1,y1),(x2,y2) = box
        results.append([
            (int(x1),int(y1)),
            (int(x2),int(y2)),
            VOC_CLASSES[c],
            image_name,
            float(conf)
        ])
    return results


# ======================================================
# Evaluation (eval.py 스타일)
# ======================================================
class Evaluation:
    def __init__(self, predictions, targets, threshold=0.5):
        self.preds = predictions
        self.targets = targets
        self.threshold = threshold

    @staticmethod
    def compute_ap(recall, precision):
        recall = np.concatenate(([0.],recall,[1.]))
        precision = np.concatenate(([0.],precision,[0.]))

        for i in range(len(precision)-2, -1, -1):
            precision[i] = max(precision[i], precision[i+1])

        ap = 0.0
        for i in range(len(recall)-1):
            ap += (recall[i+1] - recall[i]) * precision[i+1]
        return ap

    def evaluate(self):
        aps = []
        print("CLASS".ljust(25), "AP")

        for cls in VOC_CLASSES:
            preds = self.preds[cls]

            if len(preds)==0:
                print(f"---class {cls} ap -1---")
                aps.append(-1)
                continue

            img_ids = [p[0] for p in preds]
            conf    = np.array([p[1] for p in preds])
            BB      = np.array([p[2:] for p in preds])

            order = np.argsort(-conf)
            conf = conf[order]
            BB   = BB[order]
            img_ids = [img_ids[i] for i in order]

            # count GT
            npos = 0
            for (img,c),box_list in self.targets.items():
                if c == cls:
                    npos += len(box_list)

            tp = np.zeros(len(img_ids))
            fp = np.zeros(len(img_ids))

            for i, imgid in enumerate(img_ids):
                bb = BB[i]

                if (imgid, cls) in self.targets:
                    gt_list = self.targets[(imgid, cls)]
                    hit=False
                    for gt in list(gt_list):
                        x1 = max(bb[0], gt[0])
                        y1 = max(bb[1], gt[1])
                        x2 = min(bb[2], gt[2])
                        y2 = min(bb[3], gt[3])

                        iw = max(x2-x1+1,0)
                        ih = max(y2-y1+1,0)
                        inter = iw*ih

                        union = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) + \
                                (gt[2]-gt[0]+1)*(gt[3]-gt[1]+1) - inter

                        iou = inter / union

                        if iou > self.threshold:
                            tp[i]=1
                            gt_list.remove(gt)
                            hit=True
                            break

                    if not hit:
                        fp[i]=1
                else:
                    fp[i]=1

            recall = np.cumsum(tp) / max(npos,1)
            precision = np.cumsum(tp) / (np.cumsum(tp)+np.cumsum(fp)+1e-9)

            ap = self.compute_ap(recall, precision)
            print(cls.ljust(25), f"{ap:.2f}")
            aps.append(ap)

        return aps


# ======================================================
# MAIN EVAL SCRIPT
# ======================================================
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("DATA PREPARING...")
    with open("./Dataset/test.txt") as f:
        test_list = [x.strip() for x in f]

    targets = defaultdict(list)
    predictions = defaultdict(list)

    # Load GT
    for name in test_list:
        img = name + ".jpg"
        with open(f"./Dataset/Labels/{name}.txt") as f:
            for line in f:
                c,x1,y1,x2,y2 = map(int,line.split())
                cls = VOC_CLASSES[c]
                targets[(img,cls)].append([x1,y1,x2,y2])

    print("DONE.\n")
    print("START TESTING...")

    model = Model().to(device)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load("./weights/yolov1_mix_2_final.pth")["state_dict"])
    model.eval()

    with torch.no_grad():
        for name in tqdm(test_list):
            imgname = name + ".jpg"
            result = predict_mix2(model, imgname)

            for (x1,y1),(x2,y2),cls,img,conf in result:
                predictions[cls].append([img, conf, x1,y1,x2,y2])

    print("\nSTART EVALUATION...")
    aps = Evaluation(predictions, targets, threshold=0.5).evaluate()
    print(f"\nmAP: {np.mean(aps):.2f}")
    print("\nDONE.")
