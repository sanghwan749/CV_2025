import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------
# Focal BCE (Objectness)
# ---------------------------------------------
def focal_bce_with_logits(logits, targets, gamma=2.0, reduction="mean"):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)  # correctness prob
    loss = ((1 - pt) ** gamma) * bce

    return loss.mean() if reduction == "mean" else loss.sum()


# ---------------------------------------------
# Focal CE (Classification)
# ---------------------------------------------
def focal_ce(logits, targets, gamma=2.0, reduction="mean"):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)

    loss = ((1 - pt) ** gamma) * ce
    return loss.mean() if reduction == "mean" else loss.sum()


# ---------------------------------------------
# YOLOv1용 Focal Loss 조합
# 좌표/크기 회귀는 기존 yoloLoss 유지
# ---------------------------------------------
class focal_yoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = 7
        self.B = 2
        self.C = 20

        # 기존 YOLO 손실 가중치 그대로 사용
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, pred, target):
        """
        pred: (N, 7, 7, 30)
        target: same size
        """

        # -----------------------------
        # 좌표/크기/IoU/Confidence 계산 (기존 방식 그대로)
        # -----------------------------
        exist_obj = target[..., 4].unsqueeze(-1)

        # 좌표 손실
        coord_loss = self.lambda_coord * torch.sum(
            exist_obj * (pred[..., 0:2] - target[..., 0:2]) ** 2
        )
        size_loss = self.lambda_coord * torch.sum(
            exist_obj * (torch.sqrt(pred[..., 2:4] + 1e-6) - torch.sqrt(target[..., 2:4] + 1e-6)) ** 2
        )

        # -----------------------------
        # Focal Loss 기반 Confidence 손실
        # -----------------------------
        pred_conf = pred[..., 4]
        target_conf = target[..., 4]

        conf_loss_obj = focal_bce_with_logits(
            pred_conf[target_conf == 1],
            target_conf[target_conf == 1],
            gamma=2.0
        )

        conf_loss_noobj = focal_bce_with_logits(
            pred_conf[target_conf == 0],
            target_conf[target_conf == 0],
            gamma=2.0
        ) * self.lambda_noobj

        # -----------------------------
        # Focal Loss 기반 클래스 손실
        # -----------------------------
        pred_cls = pred[..., 10:30]
        target_cls = target[..., 10:30]  # one-hot

        cls_mask = target[..., 4] == 1
        cls_pred_flat = pred_cls[cls_mask]           # (N_cls, 20)
        cls_target_flat = target_cls[cls_mask].argmax(dim=1)  # class id

        if cls_pred_flat.numel() > 0:
            class_loss = focal_ce(cls_pred_flat, cls_target_flat, gamma=2.0)
        else:
            class_loss = 0.0

        return coord_loss + size_loss + conf_loss_obj + conf_loss_noobj + class_loss
