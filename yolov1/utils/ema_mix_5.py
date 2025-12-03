import copy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.9998):
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
