import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *


# ============================================================
#                    IoU 계산 함수 (공용)
# ============================================================

def bbox_iou(box1, box2, eps=1e-7):
    """IoU 계산 (x1,y1,x2,y2 형식 가정)"""

    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    area1 = (box1[..., 2] - box1[..., 0]).clamp(0) * (box1[..., 3] - box1[..., 1]).clamp(0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(0) * (box2[..., 3] - box2[..., 1]).clamp(0)

    union = area1 + area2 - inter + eps
    return inter / union


def xywh_to_xyxy(box):
    """ cx,cy,w,h → x1,y1,x2,y2 """
    cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)



# ============================================================
#                       Focal Loss (Class용)
# ============================================================

class FocalLoss(nn.Module):
    """ YOLOv1의 class MSE → Focal-MSE로 강화 """
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce = F.mse_loss(pred, target, reduction='none')
        pt = torch.exp(-ce).clamp(min=1e-7, max=1.0)
        loss = self.alpha * (1 - pt)**self.gamma * ce
        return loss.mean()



# ============================================================
#                       Alpha-DIoU Loss
# ============================================================

class AlphaDIoULoss(nn.Module):
    """
    Alpha-DIoU = (1 - IoU^α) + DIoU center penalty
    """
    def __init__(self, alpha=0.5):
        super(AlphaDIoULoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_xyxy = xywh_to_xyxy(pred[..., :4])
        tgt_xyxy = xywh_to_xyxy(target[..., :4])

        iou = bbox_iou(pred_xyxy, tgt_xyxy).clamp(min=1e-7)

        # DIoU 중심 거리 페널티
        px = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) / 2
        py = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) / 2
        tx = (tgt_xyxy[..., 0] + tgt_xyxy[..., 2]) / 2
        ty = (tgt_xyxy[..., 1] + tgt_xyxy[..., 3]) / 2

        center_dist = (px - tx)**2 + (py - ty)**2

        x1 = torch.min(pred_xyxy[..., 0], tgt_xyxy[..., 0])
        y1 = torch.min(pred_xyxy[..., 1], tgt_xyxy[..., 1])
        x2 = torch.max(pred_xyxy[..., 2], tgt_xyxy[..., 2])
        y2 = torch.max(pred_xyxy[..., 3], tgt_xyxy[..., 3])
        diag = (x2 - x1)**2 + (y2 - y1)**2 + 1e-7

        diou_penalty = center_dist / diag

        # Alpha-IoU term
        alpha_iou_term = 1 - iou**self.alpha

        return (alpha_iou_term + diou_penalty).mean()



# ============================================================
#               YOLO Loss (Focal + Alpha-DIoU 안정화)
# ============================================================

class yoloLoss(Module):

    def __init__(self, num_class=20):
        super(yoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = 14
        self.B = 2
        self.C = num_class
        self.step = 1.0 / 14

        # 강화된 loss 적용
        self.focal = FocalLoss(alpha=1.0, gamma=2.0)
        self.alpha_diou = AlphaDIoULoss(alpha=0.5)

    # ------------ (버그 수정된 IoU responsible 선택용 함수) ----------------
    def compute_iou(self, box1, box2, index):
        box1 = torch.clone(box1)
        box2 = torch.clone(box2)

        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)

        x1, y1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        x2, y2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_w = (w1 + w2) - (torch.max(x1 + w1, x2 + w2) - torch.min(x1, x2))
        inter_h = (h1 + h2) - (torch.max(y1 + h1, y2 + h2) - torch.min(y1, y2))

        inter = torch.clamp(inter_w, 0) * torch.clamp(inter_h, 0)
        union = (w1 * h1) + (w2 * h2) - inter + 1e-7   # ⭐ 버그 수정됨

        return inter / union

    def conver_box(self, box, index):
        i, j = index
        box[:, 0], box[:, 1] = [(box[:, 0] + i) * self.step - box[:, 2] / 2,
                                (box[:, 1] + j) * self.step - box[:, 3] / 2]
        return torch.clamp(box, 0)

    # -------------------------------------------------------------

    def forward(self, pred, target):
        batch_size = pred.size(0)

        target_boxes = target[:, :, :, :10].reshape((-1, self.S, self.S, 2, 5))
        pred_boxes = pred[:, :, :, :10].reshape((-1, self.S, self.S, 2, 5))

        target_cls = target[:, :, :, 10:]
        pred_cls = pred[:, :, :, 10:]

        obj_mask = (target_boxes[..., 4] > 0)
        sig_mask = obj_mask[..., 1].bool()

        index = torch.where(sig_mask == True)

        # responsible bbox 선택
        for img_i, y, x in zip(*index):
            img_i, y, x = img_i.item(), y.item(), x.item()
            pbox = pred_boxes[img_i, y, x][:, :4]
            tbox = target_boxes[img_i, y, x][:, :4]

            ious = self.compute_iou(pbox, tbox, [x, y])
            _, max_i = ious.max(0)
            obj_mask[img_i, y, x, 1 - max_i] = False

        noobj_mask = ~obj_mask

        # confidence loss (그대로 유지)
        obj_loss = F.mse_loss(pred_boxes[obj_mask][:, 4],
                              target_boxes[obj_mask][:, 4],
                              reduction="sum")

        noobj_loss = F.mse_loss(pred_boxes[noobj_mask][:, 4],
                                target_boxes[noobj_mask][:, 4],
                                reduction="sum")

        # -------------------------
        # bbox regression (Alpha-DIoU)
        # -------------------------
        if pred_boxes[obj_mask].shape[0] > 0:
            bbox_loss = self.alpha_diou(pred_boxes[obj_mask],
                                        target_boxes[obj_mask])
        else:
            bbox_loss = torch.tensor(0.0, device=pred.device)

        # -------------------------
        # class focal loss
        # -------------------------
        class_loss = self.focal(pred_cls[sig_mask], target_cls[sig_mask])

        total_loss = (
            obj_loss +
            self.lambda_noobj * noobj_loss +
            self.lambda_coord * bbox_loss +
            class_loss
        )

        return total_loss / batch_size
