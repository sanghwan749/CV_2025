# utils/loss_mix_1.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# Focal Loss (for classification)
# ======================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        # logits: (C,), target: scalar (long)
        ce = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0), reduction="none")
        pt = torch.exp(-ce)  # = softmax probability of the target class
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ======================================================
# IoU / Alpha-DIoU (boxes in cx, cy, w, h; normalized)
# ======================================================
def _xyxy_from_cxcywh(box):
    x1 = box[..., 0] - box[..., 2] / 2
    y1 = box[..., 1] - box[..., 3] / 2
    x2 = box[..., 0] + box[..., 2] / 2
    y2 = box[..., 1] + box[..., 3] / 2
    return x1, y1, x2, y2


def bbox_iou(box1, box2):
    """
    box1: (..., 4)  cx,cy,w,h in [0,1]
    box2: (..., 4)  cx,cy,w,h in [0,1]
    returns IoU in [0,1]
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = _xyxy_from_cxcywh(box1)
    b2_x1, b2_y1, b2_x2, b2_y2 = _xyxy_from_cxcywh(box2)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)

    union = (area1 + area2 - inter).clamp(min=1e-6)
    return inter / union


def alpha_diou_loss(pred_box, gt_box, alpha=1.0):
    """
    pred_box, gt_box: (N, 4) cx,cy,w,h
    """
    iou = bbox_iou(pred_box, gt_box)
    return (1.0 - (iou ** alpha)).mean()


# ======================================================
# YOLOv1 head + YOLOv5-style assign + Focal + Alpha-DIoU
# ======================================================
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.focal = FocalLoss(alpha=1.0, gamma=2.0)

    def forward(self, pred, labels):
        """
        pred: (B, S*S*(C + 5B))  — YOLOv1 output (linear)
        labels: per-sample packed GT tensor (cls, x1, y1, x2, y2) in pixel coords
                DataLoader collate에 따라 (B, L) 또는 (L,) 가능
        """
        device = pred.device
        BATCH = pred.size(0)

        # reshape prediction
        pred = pred.view(BATCH, self.S, self.S, self.C + self.B * 5)
        pred_cls  = pred[..., :self.C]                                     # (B, S, S, C)
        pred_box5 = pred[..., self.C:].view(BATCH, self.S, self.S, self.B, 5)  # (B, S, S, B, 5)
        pred_xywh = pred_box5[..., :4]
        pred_conf = pred_box5[..., 4]

        # 안전 가드: w,h는 최소값 보장(학습 초반 NaN 방지)
        # x,y도 0~1로 제한해 gradient 폭주 방지
        pred_xywh = torch.stack([
            pred_xywh[..., 0].clamp(0.0, 1.0),
            pred_xywh[..., 1].clamp(0.0, 1.0),
            pred_xywh[..., 2].clamp(min=1e-4, max=1.0),
            pred_xywh[..., 3].clamp(min=1e-4, max=1.0),
        ], dim=-1)

        # totals (must be tensors on device to keep graph)
        total_cls_loss  = torch.zeros((), device=device)
        total_box_loss  = torch.zeros((), device=device)
        total_conf_loss = torch.zeros((), device=device)

        # labels shape normalize: (B, L) or (L,) -> ensure batch-dim
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)  # (1, L)

        for b in range(BATCH):
            gt_raw = labels[b]
            if gt_raw.numel() == 0:
                # no GT: only noobj confidence
                obj_mask   = torch.zeros(self.S, self.S, self.B, device=device, dtype=torch.bool)
                noobj_mask = ~obj_mask
                noobj_conf = pred_conf[b][noobj_mask]
                total_conf_loss = total_conf_loss + F.mse_loss(noobj_conf, torch.zeros_like(noobj_conf)) * 0.5
                continue

            # truncate to multiple of 5 and reshape
            L = (gt_raw.numel() // 5) * 5
            gt_boxes = gt_raw[:L].view(-1, 5)  # (N, 5) : cls, x1, y1, x2, y2

            # filter invalid boxes (x1<x2, y1<y2)
            valid = (gt_boxes[:, 1] < gt_boxes[:, 3]) & (gt_boxes[:, 2] < gt_boxes[:, 4])
            gt_boxes = gt_boxes[valid]
            if gt_boxes.numel() == 0:
                obj_mask   = torch.zeros(self.S, self.S, self.B, device=device, dtype=torch.bool)
                noobj_mask = ~obj_mask
                noobj_conf = pred_conf[b][noobj_mask]
                total_conf_loss = total_conf_loss + F.mse_loss(noobj_conf, torch.zeros_like(noobj_conf)) * 0.5
                continue

            # init masks
            obj_mask   = torch.zeros(self.S, self.S, self.B, device=device, dtype=torch.bool)
            noobj_mask = torch.ones(self.S, self.S, self.B, device=device, dtype=torch.bool)

            # process each GT
            for gt in gt_boxes:
                cls, x1, y1, x2, y2 = gt  # all tensors

                # normalize to 448x448 space (이미 transform에서 448로 resize됨)
                cx = ((x1 + x2) * 0.5) / 448.0
                cy = ((y1 + y2) * 0.5) / 448.0
                w  = ((x2 - x1) / 448.0).clamp(min=1e-4, max=1.0)
                h  = ((y2 - y1) / 448.0).clamp(min=1e-4, max=1.0)

                # build gt box (1,4) — keep tensor ops to avoid breaking graph on pred path
                gt_box = torch.stack([cx, cy, w, h], dim=0).unsqueeze(0)  # (1,4)

                # grid cell index (not part of graph; int cast is fine)
                gx = (cx * self.S).clamp(0, self.S - 1)
                gy = (cy * self.S).clamp(0, self.S - 1)
                cell_x = int(gx.item())
                cell_y = int(gy.item())

                # choose best of B boxes (IoU to GT)
                pb0 = pred_xywh[b, cell_y, cell_x, 0].unsqueeze(0)  # (1,4)
                pb1 = pred_xywh[b, cell_y, cell_x, 1].unsqueeze(0)  # (1,4)
                iou0 = bbox_iou(pb0, gt_box)
                iou1 = bbox_iou(pb1, gt_box)
                best_b = 0 if (iou0 >= iou1).item() else 1

                obj_mask[cell_y, cell_x, best_b] = True
                noobj_mask[cell_y, cell_x, best_b] = False

                # box reg loss (Alpha-DIoU)
                pred_b = pred_xywh[b, cell_y, cell_x, best_b].unsqueeze(0)  # (1,4)
                total_box_loss = total_box_loss + alpha_diou_loss(pred_b, gt_box, alpha=1.0)

                # class focal loss (logits vs class index)
                total_cls_loss = total_cls_loss + self.focal(
                    pred_cls[b, cell_y, cell_x],
                    cls.long()
                )

            # confidence loss (YOLOv1 style weights)
            obj_conf   = pred_conf[b][obj_mask]
            noobj_conf = pred_conf[b][noobj_mask]
            if obj_conf.numel() > 0:
                total_conf_loss = total_conf_loss + F.mse_loss(obj_conf, torch.ones_like(obj_conf)) * 5.0
            if noobj_conf.numel() > 0:
                total_conf_loss = total_conf_loss + F.mse_loss(noobj_conf, torch.zeros_like(noobj_conf)) * 0.5

        # sum up (requires_grad=True thanks to pred path)
        return total_cls_loss + total_box_loss + total_conf_loss
