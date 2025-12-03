# eval_mix_5.py
import os
import cv2
import numpy as np
import argparse
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn

from nets.nn import resnet50            # ← 너의 DetNet+YOLO head
from utils.util import predict          # ← 기존 decode+NMS 사용

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']


def iou_xyxy(a, b):
    """
    a: [4] (x1,y1,x2,y2)
    b: [N,4]
    returns IoU with each row of b
    """
    xx1 = np.maximum(a[0], b[:, 0])
    yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2])
    yy2 = np.minimum(a[3], b[:, 3])

    w = np.maximum(0., xx2 - xx1 + 1.)
    h = np.maximum(0., yy2 - yy1 + 1.)
    inter = w * h

    area_a = (a[2] - a[0] + 1.) * (a[3] - a[1] + 1.)
    area_b = (b[:, 2] - b[:, 0] + 1.) * (b[:, 3] - b[:, 1] + 1.)
    union = area_a + area_b - inter
    iou = inter / np.maximum(union, 1e-12)
    return iou


def voc_ap(recall, precision, use_07_metric=False):
    """
    Compute AP given precision and recall.
    If use_07_metric=True, uses the VOC 2007 11-point method.
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.
        return ap

    # integral metric (VOC2012+)
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # make precision monotonically decreasing
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # area under PR curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def evaluate_map(predictions, gt_dict, iou_thresh=0.5, use_07_metric=False):
    """
    predictions: dict[class_name] = list([image_id, conf, x1,y1,x2,y2], ...)
    gt_dict: dict[(image_id, class_name)] = np.array([[x1,y1,x2,y2], ...])
    """
    aps = []
    per_class_stats = []

    print('CLASS'.ljust(25), 'AP')

    for cls in VOC_CLASSES:
        preds = predictions[cls]  # [[img_id, conf, x1,y1,x2,y2], ...]
        # 구성: 모든 GT 수(npos), 이미지별 GT박스/매칭플래그 준비
        # npos 계산
        npos = 0
        gt_for_cls = {}      # image_id -> array of boxes
        gt_detected = {}     # image_id -> matched flags (False init)

        for (img_id, c) in gt_dict.keys():
            if c != cls:
                continue
            boxes = gt_dict[(img_id, c)]
            if len(boxes) == 0:
                continue
            gt_for_cls[img_id] = boxes
            gt_detected[img_id] = np.zeros(len(boxes), dtype=bool)
            npos += len(boxes)

        # GT가 전혀 없는 클래스라면 AP 계산 생략(평균에서 제외)
        if npos == 0 and len(preds) == 0:
            # 출력상 표기는 0.00으로 하고 평균에는 반영하지 않음
            print(f'{cls}'.ljust(25), f'{0.00:.2f} (no GT & no pred)')
            continue

        if len(preds) == 0:
            # 예측이 한 개도 없으면 AP=0 (GT는 있음)
            print(f'{cls}'.ljust(25), f'{0.00:.2f}')
            aps.append(0.0)
            per_class_stats.append((cls, 0.0, npos, 0, 0))
            continue

        # confidence로 내림차순 정렬
        confs = np.array([float(x[1]) for x in preds])
        sort_ind = np.argsort(-confs)
        preds = [preds[i] for i in sort_ind]

        # TP/FP
        nd = len(preds)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d, det in enumerate(preds):
            img_id, conf, x1, y1, x2, y2 = det
            bb_det = np.array([x1, y1, x2, y2], dtype=np.float32)

            if img_id in gt_for_cls:
                bb_gt = gt_for_cls[img_id]  # [N,4]
                ious = iou_xyxy(bb_det, bb_gt)
                jmax = np.argmax(ious)
                iou_max = ious[jmax] if len(ious) > 0 else 0.0

                if iou_max >= iou_thresh and not gt_detected[img_id][jmax]:
                    tp[d] = 1.0
                    gt_detected[img_id][jmax] = True
                else:
                    fp[d] = 1.0
            else:
                # 해당 이미지에 해당 클래스 GT 없음
                fp[d] = 1.0

        # 누적
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recall = tp / float(max(npos, 1))
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = voc_ap(recall, precision, use_07_metric=use_07_metric)
        print(f'{cls}'.ljust(25), f'{ap:.2f}')
        aps.append(ap)
        per_class_stats.append((cls, ap, npos, int(tp[-1] if len(tp) else 0), int(fp[-1] if len(fp) else 0)))

    # mAP: GT가 전혀 없던 클래스는 제외(aps는 이미 제외되어 있음)
    mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0
    return mAP, per_class_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./Dataset', help='Dataset root (Images/, Labels/, train.txt, test.txt)')
    parser.add_argument('--weights', type=str, default='./weights/yolov1_mix_5_final.pth', help='model weights path')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for TP')
    parser.add_argument('--voc07', action='store_true', help='use VOC2007 11-point AP (default: integral AP)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id (default: auto)')
    parser.add_argument('--show', action='store_true', help='visualize predictions (slow)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.gpu is None \
        else torch.device(f'cuda:{args.gpu}')

    # ----------------- Load GT -----------------
    gt = defaultdict(list)  # (image_name, class_name) -> list of boxes
    image_list = []

    with open(os.path.join(args.data_dir, 'test.txt')) as f:
        lines = [x.strip() for x in f.readlines()]

    for stem in lines:
        img_name = f'{stem}.jpg'
        image_list.append(img_name)
        with open(os.path.join(args.data_dir, 'Labels', f'{stem}.txt')) as lf:
            for line in lf:
                c, x1, y1, x2, y2 = map(int, line.strip().split())
                cls_name = VOC_CLASSES[c]
                gt[(img_name, cls_name)].append([x1, y1, x2, y2])

    # numpy 배열로 변환
    gt_np = {}
    for k, v in gt.items():
        gt_np[k] = np.array(v, dtype=np.float32)

    # ----------------- Load Model -----------------
    model = resnet50().to(device)
    if torch.cuda.device_count() > 1 and args.gpu is None:
        model = nn.DataParallel(model)
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(sd['state_dict'])
    model.eval()

    # ----------------- Inference -----------------
    predictions = defaultdict(list)  # class -> [img_id, conf, x1,y1,x2,y2]
    print('START TESTING...')
    with torch.no_grad():
        for img_name in tqdm(image_list):
            # utils.util.predict: returns list of ((x1,y1),(x2,y2), class_name, image_name, conf)
            dets = predict(model, img_name, root_path=os.path.join(args.data_dir, 'Images') + '/')
            for (x1, y1), (x2, y2), cls_name, image_id, conf in dets:
                predictions[cls_name].append([image_id, float(conf), float(x1), float(y1), float(x2), float(y2)])

    # ----------------- Evaluation -----------------
    print('\nSTART EVALUATION...')
    mAP, stats = evaluate_map(predictions, gt_np, iou_thresh=args.iou, use_07_metric=args.voc07)
    print(f'\n=> mAP: {mAP:.3f}  (IoU={args.iou}, metric={"VOC07-11pt" if args.voc07 else "VOC12-integral"})')

    # 간단한 요약
    detected = sum(tp for _, _, _, tp, _ in stats)
    total_gt = sum(npos for _, _, npos, _, _ in stats)
    print(f'Detected GT (TP sum): {detected} / {total_gt}')
    print('DONE.')


if __name__ == '__main__':
    main()
