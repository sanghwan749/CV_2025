import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
#   Simple IoU in (cx,cy,w,h) coordinate (normalized)
#   SAFE for your dataset & main.py implementation
# ======================================================
def iou_xywh(box1, box2, eps=1e-7):
    # box = (cx,cy,w,h)
    cx1, cy1, w1, h1 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    cx2, cy2, w2, h2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    x1_min = cx1 - w1/2
    y1_min = cy1 - h1/2
    x1_max = cx1 + w1/2
    y1_max = cy1 + h1/2

    x2_min = cx2 - w2/2
    y2_min = cy2 - h2/2
    x2_max = cx2 + w2/2
    y2_max = cy2 + h2/2

    inter_x1 = torch.max(x1_min, x2_min)
    inter_y1 = torch.max(y1_min, y2_min)
    inter_x2 = torch.min(x1_max, x2_max)
    inter_y2 = torch.min(y1_max, y2_max)

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter + eps

    return inter / union


# ======================================================
#   Alpha IoU Loss
# ======================================================
class AlphaIoULoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred_xywh, tgt_xywh):
        iou = iou_xywh(pred_xywh, tgt_xywh).clamp(min=1e-7)
        loss = 1 - iou.pow(self.alpha)
        return loss.mean()


# ======================================================
#   YOLOv1 Loss (Alpha-IoU version)
# ======================================================
class yoloLoss(nn.Module):
    def __init__(self, num_class=20):
        super().__init__()
        self.S = 14
        self.B = 2
        self.C = num_class
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

        self.alpha_iou = AlphaIoULoss(alpha=0.5)

    def forward(self, pred, target):
        N = pred.size(0)

        # reshape
        pred_boxes = pred[..., :10].view(N, self.S, self.S, self.B, 5)
        tgt_boxes  = target[..., :10].view(N, self.S, self.S, self.B, 5)

        pred_cls = pred[..., 10:]
        tgt_cls  = target[..., 10:]

        pred_xywh = pred_boxes[..., :4]
        pred_conf = pred_boxes[..., 4]

        tgt_xywh  = tgt_boxes[..., :4]
        tgt_conf  = tgt_boxes[..., 4]

        # -------------------------
        # responsible box selection
        # -------------------------
        # IoU between 2 predicted boxes and GT box (tgt_xywh[...,0] is GT)
        ious = torch.zeros((N, self.S, self.S, self.B), device=pred.device)

        for b in range(self.B):
            ious[..., b] = iou_xywh(pred_xywh[..., b, :], tgt_xywh[..., 0, :])

        # best box selection
        best_box = torch.argmax(ious, dim=-1, keepdim=True)  # (N,S,S,1)

        obj_mask = (tgt_conf[..., 0] > 0).unsqueeze(-1).expand_as(best_box)  # (N,S,S,1)
        resp_mask = torch.zeros_like(ious).bool()
        resp_mask.scatter_(-1, best_box, True)
        resp_mask = resp_mask & obj_mask  # only in object cells

        noobj_mask = ~resp_mask

        # -------------------------
        # 1) Confidence Loss
        # -------------------------
        loss_obj = F.mse_loss(pred_conf[resp_mask], tgt_conf[resp_mask], reduction='sum') if resp_mask.any() else 0
        loss_noobj = F.mse_loss(pred_conf[noobj_mask], tgt_conf[noobj_mask], reduction='sum') if noobj_mask.any() else 0

        # -------------------------
        # 2) BBox Regression (Alpha-IoU)
        # -------------------------
        if resp_mask.any():
            pred_box_resp = pred_xywh[resp_mask]
            tgt_box_resp  = tgt_xywh[resp_mask]
            loss_coord = self.alpha_iou(pred_box_resp, tgt_box_resp) * pred_box_resp.size(0)
        else:
            loss_coord = torch.tensor(0., device=pred.device)

        # -------------------------
        # 3) Class Loss (MSE)
        # -------------------------
        cell_obj = (tgt_conf[..., 0] > 0)
        if cell_obj.any():
            loss_cls = F.mse_loss(pred_cls[cell_obj], tgt_cls[cell_obj], reduction='sum')
        else:
            loss_cls = torch.tensor(0., device=pred.device)

        # -------------------------
        # final
        # -------------------------
        total = (
            self.lambda_coord * loss_coord +
            loss_obj +
            self.lambda_noobj * loss_noobj +
            loss_cls
        )

        return total / N
