# utils/loss_mix_7.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CIoU (center distance + aspect ratio + IoU 기반)
# pred/tgt format: (N,4) → (cx,cy,w,h)
# ============================================================
def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    tx, ty, tw, th = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

    # xyxy 변환
    p_x1 = px - pw / 2
    p_y1 = py - ph / 2
    p_x2 = px + pw / 2
    p_y2 = py + ph / 2

    t_x1 = tx - tw / 2
    t_y1 = ty - th / 2
    t_x2 = tx + tw / 2
    t_y2 = ty + th / 2

    # IoU
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_t = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)

    union = area_p + area_t - inter_area + eps
    iou = inter_area / union

    # center distance
    rho2 = (px - tx) ** 2 + (py - ty) ** 2

    # enclosing box
    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)

    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # aspect ratio penalty
    v = (4 / (3.14159265 ** 2)) * ((torch.atan(tw / th) - torch.atan(pw / ph)) ** 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = 1 - iou + (rho2 / c2) + alpha * v
    return ciou


# ============================================================
# LOSS MIX_7 : CIoU + SmoothL1 안정 조합
# pred, target shapes: (B,S,S,30)
# ============================================================
class YoloLoss(nn.Module):
    def __init__(self, S=14, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, pred, target):
        B, S, _, _ = pred.shape
        device = pred.device

        pred = pred.view(B, S, S, 30)
        target = target.view(B, S, S, 30)

        # -------------------------------
        # prediction split
        # -------------------------------
        pred_boxes = pred[..., :10].view(B, S, S, 2, 5)
        pred_cls   = pred[..., 10:]

        tgt_boxes  = target[..., :10].view(B, S, S, 2, 5)
        tgt_cls    = target[..., 10:]

        # object mask: (B,S,S)
        obj_mask = (target[..., 4] > 0) | (target[..., 9] > 0)
        obj_mask_box = obj_mask.unsqueeze(-1).expand(-1, -1, -1, 2)

        # ============================================================
        # 1) BBox Regression (CIoU + SmoothL1)
        # ============================================================
        if obj_mask_box.any():
            # xy → sigmoid, wh → clamp positive
            pred_xy = torch.sigmoid(pred_boxes[..., 0:2])
            pred_wh = torch.clamp(pred_boxes[..., 2:4], min=1e-4)
            pred_box_all = torch.cat([pred_xy, pred_wh], dim=-1)  # (B,S,S,2,4)

            tgt_xy = tgt_boxes[..., 0:2]
            tgt_wh = torch.clamp(tgt_boxes[..., 2:4], min=1e-4)
            tgt_box_all = torch.cat([tgt_xy, tgt_wh], dim=-1)

            pb = pred_box_all[obj_mask_box]
            tb = tgt_box_all[obj_mask_box]

            ciou = ciou_loss(pb, tb)
            smooth = F.smooth_l1_loss(pb, tb, reduction="none").mean(dim=1)

            bbox_loss = (0.7 * ciou + 0.3 * smooth).mean()
        else:
            bbox_loss = torch.zeros((), device=device)

        # ============================================================
        # 2) Objectness Loss
        # ============================================================
        pred_obj = pred_boxes[..., 4]
        tgt_obj  = tgt_boxes[..., 4]

        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, tgt_obj, reduction="mean")

        # ============================================================
        # 3) Classification Loss
        # ============================================================
        if obj_mask.any():
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_cls[obj_mask], tgt_cls[obj_mask], reduction="mean"
            )
        else:
            cls_loss = torch.zeros((), device=device)

        # ============================================================
        # total
        # ============================================================
        loss = self.lambda_coord * bbox_loss + obj_loss + cls_loss
        return loss
