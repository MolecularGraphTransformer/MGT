import torch
import numpy as np
from typing import Optional
from torch import nn, Tensor


class MLPLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(MLPLayer, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.SiLU = nn.SiLU(inplace=True)        

    def forward(self, x: Tensor):
        x = self.linear(x)
        x = self.norm(x)
        return self.SiLU(x)


class RBFExpansion(nn.Module):
    def __init__(self, vmin: float = 0, vmax: float = 8, bins: int = 40, lenghtscale: Optional[float] = None):
        super(RBFExpansion, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer('centers', torch.linspace(vmin, vmax, bins))

        if lenghtscale is None:
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lenghtscale
            self.gamma = 1 / (lenghtscale ** 2)

    def forward(self, x: Tensor):
        return torch.exp(-self.gamma * (x.unsqueeze(1) - self.centers) ** 2)
