import torch
import numpy as np

def predict_mix8(model, img_tensor, device, S=14, C=20):
    model.eval()
    
    with torch.no_grad():
        out = model(img_tensor.to(device))
        out = out.view(1, S, S, 30)

    box1 = out[..., 0:5]
    box2 = out[..., 5:10]
    cls  = out[..., 10:]

    conf1 = box1[..., 4]
    conf2 = box2[..., 4]

    # choose higher conf
    choose1 = conf1 > conf2
    chosen_box = torch.where(
        choose1.unsqueeze(-1),
        box1[..., 0:4],
        box2[..., 0:4]
    )
    chosen_conf = torch.where(choose1, conf1, conf2)

    tx = chosen_box[..., 0]
    ty = chosen_box[..., 1]
    tw = chosen_box[..., 2] ** 2
    th = chosen_box[..., 3] ** 2

    cx = torch.arange(S).view(1, 1, S).expand(1, S, S).float().to(device)
    cy = torch.arange(S).view(1, S, 1).expand(1, S, S).float().to(device)

    bx = (tx + cx) / S
    by = (ty + cy) / S

    x1 = (bx - tw / 2) * 448
    y1 = (by - th / 2) * 448
    x2 = (bx + tw / 2) * 448
    y2 = (by + th / 2) * 448

    cls_idx = cls.argmax(-1)

    boxes = []
    for gy in range(S):
        for gx in range(S):
            if chosen_conf[0, gy, gx] < 0.05:
                continue

            boxes.append([
                float(x1[0, gy, gx]),
                float(y1[0, gy, gx]),
                float(x2[0, gy, gx]),
                float(y2[0, gy, gx]),
                int(cls_idx[0, gy, gx]),
                float(chosen_conf[0, gy, gx]),
            ])
    return boxes
