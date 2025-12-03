import torch
import torch.nn as nn
import torch.nn.functional as F


class YoloLoss(nn.Module):
    def __init__(self, S=14, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()

        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    # ============================================================
    # Decode Box (YOLOv1 공식 Grid 방식으로 완전한 패치 버전)
    # ============================================================
    def decode_box(self, box):
        # box: [B, S, S, 5]
        B = box.size(0)
        S = self.S
        device = box.device

        tx = box[..., 0]
        ty = box[..., 1]
        tw = box[..., 2]
        th = box[..., 3]

        # ---- grid offsets (정확한 YOLOv1 방식) ----
        grid_x = torch.arange(S, device=device).repeat(S, 1)       # [S,S]
        grid_y = torch.arange(S, device=device).repeat(S, 1).t()   # [S,S]

        grid_x = grid_x.unsqueeze(0)    # [1,S,S]
        grid_y = grid_y.unsqueeze(0)    # [1,S,S]

        # ---- Apply decode formula (broadcast 정상 동작) ----
        bx = (tx + grid_x) / S
        by = (ty + grid_y) / S
        bw = (tw ** 2)
        bh = (th ** 2)

        # ---- Convert to corner coordinates ----
        x1 = bx - bw * 0.5
        y1 = by - bh * 0.5
        x2 = bx + bw * 0.5
        y2 = by + bh * 0.5

        return torch.stack([x1, y1, x2, y2], dim=-1)

    # ============================================================
    # IoU 계산
    # ============================================================
    def bbox_iou(self, box1, box2):
        # box1, box2: [..., 4]
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])

        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        area1 = (box1[..., 2] - box1[..., 0]).clamp(0) * (box1[..., 3] - box1[..., 1]).clamp(0)
        area2 = (box2[..., 2] - box2[..., 0]).clamp(0) * (box2[..., 3] - box2[..., 1]).clamp(0)

        union = area1 + area2 - inter + 1e-6

        return inter / union

    # ============================================================
    # Forward (전체 Loss)
    # ============================================================
    def forward(self, pred, target):
        """
        pred, target shape: [B, S, S, 30]
        """

        B = pred.size(0)
        device = pred.device

        # --------------------------------------------------------
        # 분리
        # pred: [x, y, w, h, conf] * 2 + [cls_probs(20)]
        # --------------------------------------------------------
        pred = pred.view(B, self.S, self.S, self.B * 5 + self.C)

        pred_box1 = pred[..., 0:5]
        pred_box2 = pred[..., 5:10]
        pred_cls  = pred[..., 10:10 + self.C]

        tgt_box   = target[..., 0:5]
        tgt_cls   = target[..., 10:10 + self.C]

        obj_mask = target[..., 4] > 0   # object cell mask: [B,S,S]
        noobj_mask = ~obj_mask

        # --------------------------------------------------------
        # 1) Box decode (YOLO 방식)
        # --------------------------------------------------------
        pred1_xyxy = self.decode_box(pred_box1)
        pred2_xyxy = self.decode_box(pred_box2)
        tgt_xyxy   = self.decode_box(tgt_box)

        # --------------------------------------------------------
        # 2) IoU 계산 (Object cell만)
        # --------------------------------------------------------
        iou1 = self.bbox_iou(pred1_xyxy, tgt_xyxy)
        iou2 = self.bbox_iou(pred2_xyxy, tgt_xyxy)

        # --------------------------------------------------------
        # 3) Best box 선택
        # --------------------------------------------------------
        best_box_mask = (iou1 > iou2).float()[..., None]  # [B,S,S,1]

        chosen_pred_box = pred1_xyxy * best_box_mask + pred2_xyxy * (1 - best_box_mask)
        chosen_pred_conf = pred_box1[..., 4:5] * best_box_mask + pred_box2[..., 4:5] * (1 - best_box_mask)

        # --------------------------------------------------------
        # 4) Loss 계산
        # --------------------------------------------------------

        # ---- Coord Loss ----
        coord_loss = F.mse_loss(chosen_pred_box[obj_mask], tgt_xyxy[obj_mask], reduction='sum')

        # ---- Object confidence loss ----
        obj_conf_loss = F.mse_loss(chosen_pred_conf[obj_mask], iou1[obj_mask].unsqueeze(-1), reduction='sum')

        # ---- No-object confidence loss ----
        noobj_loss = (
            F.mse_loss(pred_box1[..., 4][noobj_mask], torch.zeros_like(pred_box1[..., 4][noobj_mask]), reduction='sum') +
            F.mse_loss(pred_box2[..., 4][noobj_mask], torch.zeros_like(pred_box2[..., 4][noobj_mask]), reduction='sum')
        )

        # ---- Class probability loss ----
        class_loss = F.mse_loss(pred_cls[obj_mask], tgt_cls[obj_mask], reduction='sum')

        total_loss = (
            self.lambda_coord * coord_loss +
            obj_conf_loss +
            self.lambda_noobj * noobj_loss +
            class_loss
        )

        return total_loss
