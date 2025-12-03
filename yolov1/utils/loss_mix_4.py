import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
#  CIoU 계산
# ============================
def bbox_ciou(pred_boxes, target_boxes):
    # pred_boxes, target_boxes: [N, 4] tensor only (cx,cy,w,h)
    eps = 1e-7

    # convert to x1y1x2y2
    p_xy1 = pred_boxes[:, :2] - pred_boxes[:, 2:] * 0.5
    p_xy2 = pred_boxes[:, :2] + pred_boxes[:, 2:] * 0.5
    t_xy1 = target_boxes[:, :2] - target_boxes[:, 2:] * 0.5
    t_xy2 = target_boxes[:, :2] + target_boxes[:, 2:] * 0.5

    # intersection
    inter_xy1 = torch.max(p_xy1, t_xy1)
    inter_xy2 = torch.min(p_xy2, t_xy2)
    inter_wh = torch.clamp(inter_xy2 - inter_xy1, min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    # union
    p_area = (p_xy2[:, 0]-p_xy1[:, 0]) * (p_xy2[:, 1]-p_xy1[:, 1])
    t_area = (t_xy2[:, 0]-t_xy1[:, 0]) * (t_xy2[:, 1]-t_xy1[:, 1])
    union = p_area + t_area - inter_area + eps

    iou = inter_area / union

    # center distance
    center_dist = torch.sum((pred_boxes[:, :2] - target_boxes[:, :2])**2, dim=1)

    # outer box
    c_xy1 = torch.min(p_xy1, t_xy1)
    c_xy2 = torch.max(p_xy2, t_xy2)
    c_wh = c_xy2 - c_xy1
    c_diagonal = torch.sum(c_wh**2, dim=1) + eps

    # v & alpha
    v = (4 / (3.14159265**2)) * torch.pow(
        torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + eps))
        - torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + eps)), 2)

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist / c_diagonal) - alpha * v
    return ciou


# ============================
#  Focal Loss
# ============================
def focal_loss(pred, target, alpha=1.0, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


# ============================
#  YOLOv1 + CIoU + Focal Loss
# ============================
class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.S = 14
        self.B = 2
        self.C = 20

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, pred, target):
        """
        pred, target: [B, 14, 14, 30]
        """
        B = pred.size(0)

        pred_boxes = pred[..., :10].view(B, self.S, self.S, self.B, 5)
        target_boxes = target[..., :10].view(B, self.S, self.S, self.B, 5)

        pred_cls = pred[..., 10:]
        target_cls = target[..., 10:]

        # object mask
        obj_mask = target_boxes[..., 4] > 0
        noobj_mask = ~obj_mask

        # ============================
        # 1) No-object loss (Focal)
        # ============================
        noobj_loss = focal_loss(pred_boxes[..., 4][noobj_mask],
                                target_boxes[..., 4][noobj_mask])

        # ============================
        # 2) Object Confidence Loss (Focal)
        # ============================
        obj_conf_loss = focal_loss(pred_boxes[..., 4][obj_mask],
                                   target_boxes[..., 4][obj_mask])

        # ============================
        # 3) Classification Loss (Focal)
        # ============================
        class_loss = focal_loss(pred_cls[obj_mask[..., 0]],
                                target_cls[obj_mask[..., 0]])

        # ============================
        # 4) CIoU BBox Loss
        # ============================
        p = pred_boxes[obj_mask][..., :4]
        t = target_boxes[obj_mask][..., :4]
        ciou = bbox_ciou(p, t)
        ciou_loss = (1 - ciou).mean()

        # ============================
        # Final loss
        # ============================
        loss = (
            obj_conf_loss +
            self.lambda_noobj * noobj_loss +
            self.lambda_coord * ciou_loss +
            class_loss
        )

        return loss
