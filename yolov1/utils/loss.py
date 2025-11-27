import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import *


class yoloLoss(Module):
    def __init__(self, num_class=20):
        super(yoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = 14
        self.B = 2
        self.C = num_class
        self.step = 1.0 / 14

    def compute_iou(self, box1, box2, index):
        box1 = torch.clone(box1)
        box2 = torch.clone(box2)
        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)
        x1, y1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        x2, y2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        # 
        inter_w = (w1 + w2) - (torch.max(x1 + w1, x2 + w2) - torch.min(x1, x2))
        inter_h = (h1 + h2) - (torch.max(y1 + h1, y2 + h2) - torch.min(y1, y2))
        inter_h = torch.clamp(inter_h, 0)
        inter_w = torch.clamp(inter_w, 0)
        # 
        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter
        return inter / union

    def conver_box(self, box, index):
        i, j = index
        box[:, 0], box[:, 1] = [(box[:, 0] + i) * self.step - box[:, 2] / 2,
                                (box[:, 1] + j) * self.step - box[:, 3] / 2]
        box = torch.clamp(box, 0)
        return box

    def forward(self, pred, target):
        batch_size = pred.size(0)
        #target tensor = [batch_size,14,14,30]
        #pred tensor = [batch_size,14,14,30]       
        
        #target matirx tensor로부터 bbox 정보만 분리, [batch_size,14,14,10]
        #분리된 bbox tensor=[batch_size,14,14,10] --> [batch_size,14,14,2,5]로 변경
        #이후 예측 tensor에 관해 동일 반복
        target_boxes = target[:, :, :, :10].contiguous().reshape(
            (-1, self.S, self.S, 2, 5))
        pred_boxes = pred[:, :, :, :10].contiguous().reshape(
            (-1, self.S, self.S, 2, 5))
        
        #target matirx tensor로부터 grid cell의 class probability 분리, [batch_size,14,14,20]
        #예측 tensor에 관해 동일 반복
        target_cls = target[:, :, :, 10:]
        pred_cls = pred[:, :, :, 10:]
        
        #target tensor로부터 object가 위치하는 batch image 위치와 
        # grid cell에서 좌표를 계산
        #여기서, Obj_mask는 target tensor로부터 물체가 위치하는 
        # grid cell 위치에 true값을 갖는 mask tensor임
        #index tensor는 물체가 위치하는 batch image 위치와 grid cell에서
        # 좌표값의 index를 갖음
        obj_mask = (target_boxes[..., 4] > 0).byte()  #obj_mask=[-1,14,14,2]
        sig_mask = obj_mask[..., 1].bool() #sig_mask=[-1,14,14]
        index = torch.where(sig_mask == True) #
        #object가 위치하는 grid cell마다 2개 bbox와 ground truth bbox(이하 GT)와
        # IOU값을 계산하고 예측된 2 bbox중 IOU가 최대인 bbox를 찾아 GT bbox를 예측한 
        # bbox로 선정함.
        # 그리고 obj_mask에 나머지 bbox의 confidence score 위치에 0으로 reset함
        for img_i, y, x in zip(*index):
            img_i, y, x = img_i.item(), y.item(), x.item()
            pbox = pred_boxes[img_i, y, x]
            target_box = target_boxes[img_i, y, x]
            ious = self.compute_iou(pbox[:, :4], target_box[:, :4], [x, y])
            iou, max_i = ious.max(0)
            #pred_boxes[img_i, y, x, max_i, 4] = iou.item()
            #pred_boxes[img_i, y, x, 1 - max_i, 4] = 0
            obj_mask[img_i, y, x, 1 - max_i] = 0
        
        #obj_mask를 반전시켜 물체가 위치하지 않은 mask tensor를 구성
        noobj_mask = ~obj_mask
         
        #물체가 존재하지 않은 bbox의 confidence score 오차의 loss값 계산 
        noobj_loss = F.mse_loss(pred_boxes[noobj_mask][:, 4],
                                target_boxes[noobj_mask][:, 4],
                                reduction="sum")
        #물체가 존재하는 bbox의 confidence score 오차의 loss값 계산
        obj_loss = F.mse_loss(pred_boxes[obj_mask][:, 4],
                              target_boxes[obj_mask][:, 4],
                              reduction="sum")
        #물체가 존재하는 bbox의 중심점의 오차에 loss값 계산
        xy_loss = F.mse_loss(pred_boxes[obj_mask][:, :2],
                             target_boxes[obj_mask][:, :2],
                             reduction="sum")
        
        #물체가 존재하는 bbox의 width와 height의 오차에 loss값 계산
        wh_loss = F.mse_loss(torch.sqrt(target_boxes[obj_mask][:, 2:4]),
                             torch.sqrt(pred_boxes[obj_mask][:, 2:4]),
                             reduction="sum")
        
        #물체가 존재하는 grid cell의 class probability의 오차에 loss값 계산
        class_loss = F.mse_loss(pred_cls[sig_mask],
                                target_cls[sig_mask],
                                reduction="sum")

        #각 loss값의 합산
        loss = obj_loss + self.lambda_noobj * noobj_loss \
                    + self.lambda_coord * xy_loss + self.lambda_coord * wh_loss \
                    + class_loss
        
        return loss/batch_size
    
    

# ============================================================
#                    IoU 계산 함수 (공용)
# ============================================================

def bbox_iou(box1, box2, eps=1e-7):
    """
    IoU = area(intersection) / area(union)
    box = [x1, y1, x2, y2]
    """
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
    """
    (center x, center y, w, h) → (x1, y1, x2, y2)
    """
    cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ============================================================
#                          DIoU Loss
# ============================================================

class DIoULoss(nn.Module):
    """
    기존 YOLO box regression → DIoU 기반으로 수정
    """
    def __init__(self):
        super(DIoULoss, self).__init__()

    def forward(self, pred, target):
        pred_xyxy = xywh_to_xyxy(pred[..., :4])
        tgt_xyxy = xywh_to_xyxy(target[..., :4])

        iou = bbox_iou(pred_xyxy, tgt_xyxy)

        # distance penalty
        px = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) / 2
        py = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) / 2
        tx = (tgt_xyxy[..., 0] + tgt_xyxy[..., 2]) / 2
        ty = (tgt_xyxy[..., 1] + tgt_xyxy[..., 3]) / 2

        center_dist = (px - tx) ** 2 + (py - ty) ** 2

        # diagonal length of smallest enclosing box
        x1 = torch.min(pred_xyxy[..., 0], tgt_xyxy[..., 0])
        y1 = torch.min(pred_xyxy[..., 1], tgt_xyxy[..., 1])
        x2 = torch.max(pred_xyxy[..., 2], tgt_xyxy[..., 2])
        y2 = torch.max(pred_xyxy[..., 3], tgt_xyxy[..., 3])

        diag = (x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-7

        diou = iou - (center_dist / diag)
        return (1 - diou).mean()



# ============================================================
#                          CIoU Loss
# ============================================================

class CIoULoss(nn.Module):
    """
    DIoU + aspect ratio penalty
    """
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, pred, target):
        pred_xyxy = xywh_to_xyxy(pred[..., :4])
        tgt_xyxy = xywh_to_xyxy(target[..., :4])

        iou = bbox_iou(pred_xyxy, tgt_xyxy)

        # center distance
        px = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) / 2
        py = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) / 2
        tx = (tgt_xyxy[..., 0] + tgt_xyxy[..., 2]) / 2
        ty = (tgt_xyxy[..., 1] + tgt_xyxy[..., 3]) / 2

        center_dist = (px - tx)**2 + (py - ty)**2

        # diagonal length of enclosing box
        x1 = torch.min(pred_xyxy[..., 0], tgt_xyxy[..., 0])
        y1 = torch.min(pred_xyxy[..., 1], tgt_xyxy[..., 1])
        x2 = torch.max(pred_xyxy[..., 2], tgt_xyxy[..., 2])
        y2 = torch.max(pred_xyxy[..., 3], tgt_xyxy[..., 3])

        diag = (x2 - x1)**2 + (y2 - y1)**2 + 1e-7

        # aspect ratio
        pred_w = pred_xyxy[..., 2] - pred_xyxy[..., 0]
        pred_h = pred_xyxy[..., 3] - pred_xyxy[..., 1]
        tgt_w = tgt_xyxy[..., 2] - tgt_xyxy[..., 0]
        tgt_h = tgt_xyxy[..., 3] - tgt_xyxy[..., 1]

        v = (4 / (3.141592653589793**2)) * torch.pow(torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h), 2)
        with torch.no_grad():
            S = 1 - iou
            alpha = v / (S + v + 1e-7)

        ciou = iou - (center_dist / diag) - alpha * v
        return (1 - ciou).mean()



# ============================================================
#                          Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """
    CrossEntropy 기반 Focal Loss
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt)**self.gamma * ce
        return focal.mean()



# ============================================================
#                       Alpha-IoU Loss
# ============================================================

class AlphaIoULoss(nn.Module):
    """
    Alpha IoU Loss = 1 - IoU^α
    α < 1 → small object penalty 증가
    """
    def __init__(self, alpha=0.5):
        super(AlphaIoULoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_xyxy = xywh_to_xyxy(pred[..., :4])
        tgt_xyxy = xywh_to_xyxy(target[..., :4])

        iou = bbox_iou(pred_xyxy, tgt_xyxy)
        return (1 - iou.clamp(min=1e-7)**self.alpha).mean()