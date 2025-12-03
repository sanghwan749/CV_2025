import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# CIoU Loss
# ==========================================================
def ciou_loss(pred_boxes, target_boxes):
    px, py, pw, ph = pred_boxes.unbind(-1)
    tx, ty, tw, th = target_boxes.unbind(-1)

    # Convert to corners
    pred_xmin = px - pw / 2
    pred_ymin = py - ph / 2
    pred_xmax = px + pw / 2
    pred_ymax = py + ph / 2

    tgt_xmin = tx - tw / 2
    tgt_ymin = ty - th / 2
    tgt_xmax = tx + tw / 2
    tgt_ymax = ty + th / 2

    # Intersection
    inter_xmin = torch.max(pred_xmin, tgt_xmin)
    inter_ymin = torch.max(pred_ymin, tgt_ymin)
    inter_xmax = torch.min(pred_xmax, tgt_xmax)
    inter_ymax = torch.min(pred_ymax, tgt_ymax)

    inter_area = (inter_xmax - inter_xmin).clamp(min=0) * (inter_ymax - inter_ymin).clamp(min=0)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    tgt_area = (tgt_xmax - tgt_xmin) * (tgt_ymax - tgt_ymin)

    union = pred_area + tgt_area - inter_area + 1e-7
    iou = inter_area / union

    # Enclosing box
    cxmin = torch.min(pred_xmin, tgt_xmin)
    cymin = torch.min(pred_ymin, tgt_ymin)
    cxmax = torch.max(pred_xmax, tgt_xmax)
    cymax = torch.max(pred_ymax, tgt_ymax)

    c2 = (cxmax - cxmin)**2 + (cymax - cymin)**2 + 1e-7
    rho2 = (px - tx)**2 + (py - ty)**2

    # Aspect ratio
    v = (4 / (3.141592653589793**2)) * (torch.atan(tw / th) - torch.atan(pw / ph))**2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - rho2 / c2 - alpha * v
    return 1 - ciou


# ==========================================================
# Focal BCE (objectness)
# ==========================================================
def focal_bce_with_logits(logit, target, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p = torch.sigmoid(logit)
    pt = torch.where(target == 1, p, 1 - p)
    return ((1 - pt) ** gamma * bce).mean()


# ==========================================================
# Final YOLOv1-style loss
# ==========================================================
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

    def forward(self, pred, target):
        # pred,target shape = (N, S, S, 5B + C)

        obj_mask = target[..., 4] == 1
        noobj_mask = target[..., 4] == 0

        # === Box loss (CIoU) ===
        pred_box = pred[..., :4]
        tgt_box = target[..., :4]
        loss_box = ciou_loss(pred_box[obj_mask], tgt_box[obj_mask]).mean()

        # === Objectness (Focal BCE) ===
        loss_obj = focal_bce_with_logits(pred[..., 4], target[..., 4])

        # === No-object loss ===
        loss_noobj = 0.5 * F.mse_loss(pred[..., 4][noobj_mask], target[..., 4][noobj_mask])

        # === Class loss ===
        pred_cls = pred[..., 5:]
        tgt_cls = target[..., 5:]
        loss_cls = F.binary_cross_entropy_with_logits(pred_cls[obj_mask], tgt_cls[obj_mask])

        # total
        total_loss = loss_box + loss_obj + loss_noobj + loss_cls
        return total_loss
