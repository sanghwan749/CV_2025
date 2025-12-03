# utils/loss_mix_5.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# EIoU (cx,cy,w,h) 입력
# =========================
def eiou_loss(pred_boxes, target_boxes):
    eps = 1e-7
    px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    tx, ty, tw, th = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

    # to xyxy
    p_x1 = px - pw * 0.5
    p_y1 = py - ph * 0.5
    p_x2 = px + pw * 0.5
    p_y2 = py + ph * 0.5

    t_x1 = tx - tw * 0.5
    t_y1 = ty - th * 0.5
    t_x2 = tx + tw * 0.5
    t_y2 = ty + th * 0.5

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    p_area = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    t_area = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)
    union = p_area + t_area - inter_area + eps
    iou = inter_area / union

    # center distance
    center_dist = (px - tx) ** 2 + (py - ty) ** 2

    # outer box
    x1 = torch.min(p_x1, t_x1)
    y1 = torch.min(p_y1, t_y1)
    x2 = torch.max(p_x2, t_x2)
    y2 = torch.max(p_y2, t_y2)
    C = (x2 - x1) ** 2 + (y2 - y1) ** 2 + eps

    # width/height penalty
    w_diff = (pw - tw) ** 2
    h_diff = (ph - th) ** 2
    W = (x2 - x1) ** 2 + eps
    H = (y2 - y1) ** 2 + eps

    eiou = 1 - iou + center_dist / C + w_diff / W + h_diff / H
    return eiou


class YoloLoss(nn.Module):
    def __init__(self, S=14, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5  # (현재 사용 X, 필요 시 obj/noobj 분리 가중치에 활용)

    def forward(self, pred, target):
        """
        pred:   (B, S, S, 30)  ─ raw logits (xy,wh,obj,obj,cls)
        target: (B, S, S, 30)  ─ encoded (offset_xy∈[0,1], wh∈[0,1], obj∈{0,1}, one-hot cls)
        """
        B, S, _, _ = pred.shape
        device = pred.device

        # ----- split -----
        # boxes -> (B,S,S,2,5) ; each box: [x,y,w,h,conf_logit]
        pred_ = pred.view(B, S, S, 30)
        pred_boxes = pred_[..., :10].view(B, S, S, 2, 5)
        pred_cls = pred_[..., 10:]                     # (B,S,S,C)

        tgt_ = target.view(B, S, S, 30)
        tgt_boxes = tgt_[..., :10].view(B, S, S, 2, 5)
        tgt_cls = tgt_[..., 10:]                       # (B,S,S,C)

        # object mask per cell (target has both [4] and [9] set to 1 for object cells)
        obj_mask_cell = (tgt_[..., 4] > 0) | (tgt_[..., 9] > 0)  # (B,S,S)
        # expand to per-box mask (B,S,S,2)
        obj_mask_box = obj_mask_cell.unsqueeze(-1).expand(-1, -1, -1, 2)

        # =======================
        # 1) BBox regression (EIoU)
        # =======================
        # select only object cells (both boxes per cell)
        if obj_mask_box.any():
            # pred xy are offsets (0~1) -> pass through sigmoid to keep in [0,1]
            pred_xy = torch.sigmoid(pred_boxes[..., 0:2])
            # pred wh keep positive
            pred_wh = torch.clamp(pred_boxes[..., 2:4], min=1e-4)
            pred_box_all = torch.cat([pred_xy, pred_wh], dim=-1)   # (B,S,S,2,4)

            tgt_xy = tgt_boxes[..., 0:2]
            tgt_wh = torch.clamp(tgt_boxes[..., 2:4], min=1e-4)
            tgt_box_all = torch.cat([tgt_xy, tgt_wh], dim=-1)      # (B,S,S,2,4)

            # mask index
            pred_box_sel = pred_box_all[obj_mask_box]              # (Npos*2, 4) or (K,4)
            tgt_box_sel = tgt_box_all[obj_mask_box]                # (K,4)

            bbox_loss = eiou_loss(pred_box_sel, tgt_box_sel).mean()
        else:
            bbox_loss = torch.zeros((), device=device)

        # =======================
        # 2) Objectness (both boxes)
        # =======================
        pred_obj = pred_boxes[..., 4]   # (B,S,S,2) logits
        tgt_obj = tgt_boxes[..., 4]     # (B,S,S,2) 0/1
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, tgt_obj, reduction="mean")

        # =======================
        # 3) Classification (per object cell)
        # =======================
        if obj_mask_cell.any():
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_cls[obj_mask_cell], tgt_cls[obj_mask_cell], reduction="mean"
            )
        else:
            cls_loss = torch.zeros((), device=device)

        # =======================
        # total
        # =======================
        loss = self.lambda_coord * bbox_loss + obj_loss + cls_loss
        return loss
