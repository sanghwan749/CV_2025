import torch
import torch.nn as nn

class YoloLoss_mix_3(nn.Module):
    """
    YOLOv1 Loss for mix_3 model
    Output shape: (batch, 7, 7, 30)
      - 20 class scores
      - 2 boxes × (x,y,w,h,conf)
    """
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss_mix_3, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def compute_iou(self, box1, box2):
        """
        box shape: (batch, S, S, B, 4)
        x,y,w,h are absolute format
        """

        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2

        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        union = area1 + area2 - inter_area + 1e-6
        return inter_area / union

    def forward(self, pred, target):
        """
        pred: (batch, 7, 7, 30) raw output
        target: (batch, 7, 7, 30)
        """

        batch = pred.size(0)

        # pred tensors
        pred_cls = pred[..., :self.C]                       # (B,S,S,20)
        pred_boxes = pred[..., self.C:self.C + self.B*5]    # (B,S,S,10)
        pred_boxes = pred_boxes.view(batch, self.S, self.S, self.B, 5)

        pred_xy = pred_boxes[..., 0:2]    # (x,y)
        pred_wh = pred_boxes[..., 2:4]    # (w,h)
        pred_conf = pred_boxes[..., 4]    # conf

        # target tensors
        tgt_cls = target[..., :self.C]
        tgt_boxes = target[..., self.C:self.C + self.B*5]
        tgt_boxes = tgt_boxes.view(batch, self.S, self.S, self.B, 5)

        tgt_xy = tgt_boxes[..., 0:2]
        tgt_wh = tgt_boxes[..., 2:4]
        tgt_conf = tgt_boxes[..., 4]   # object mask (1 or 0)

        # Which cell has object? (sum over B)
        has_obj = tgt_conf.max(dim=-1)[0]  # shape (B,S,S)

        # IOU for box responsibility
        ious = self.compute_iou(
            torch.cat([pred_xy, pred_wh], dim=-1),
            torch.cat([tgt_xy, tgt_wh], dim=-1)
        )  # shape: (batch,S,S,B)

        best_box = ious.argmax(dim=-1)  # (batch,S,S)

        # Extract predictions for responsible box
        obj_mask = torch.zeros_like(tgt_conf)
        for b in range(self.B):
            obj_mask[..., b] = (best_box == b).float() * has_obj

        obj_mask = obj_mask.unsqueeze(-1)

        # xy loss
        xy_loss = self.lambda_coord * torch.sum(
            obj_mask * (pred_xy - tgt_xy) ** 2
        )

        # wh loss (sqrt)
        pred_wh_sqrt = torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh) + 1e-6)
        tgt_wh_sqrt = torch.sign(tgt_wh) * torch.sqrt(torch.abs(tgt_wh) + 1e-6)

        wh_loss = self.lambda_coord * torch.sum(
            obj_mask * (pred_wh_sqrt - tgt_wh_sqrt) ** 2
        )

        # confidence loss
        conf_loss_obj = torch.sum(
            obj_mask.squeeze(-1) * (pred_conf - tgt_conf) ** 2
        )

        conf_loss_noobj = self.lambda_noobj * torch.sum(
            (1 - has_obj) * (pred_conf[..., 0] ** 2 + pred_conf[..., 1] ** 2)
        )

        # class loss
        class_loss = torch.sum(
            has_obj.unsqueeze(-1) * (pred_cls - tgt_cls) ** 2
        )

        total = (
            xy_loss +
            wh_loss +
            conf_loss_obj +
            conf_loss_noobj +
            class_loss
        ) / batch

        return total
