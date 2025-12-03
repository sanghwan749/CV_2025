# utils/ema_mix_4.py
import copy
import torch

class ModelEMA:
    def __init__(self, model, decay=0.9998):
        # 학습 모델과 완전히 분리된 복제본
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        d = self.decay

        for k, v in esd.items():
            if v.dtype.is_floating_point:
                esd[k].mul_(d).add_(msd[k], alpha=1.0 - d)
            else:
                esd[k].copy_(msd[k])
