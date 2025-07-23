

import torch
from mambapy.M2 import Mamba2, Mamba2Config


config = Mamba2Config(d_model=16, n_layers=2, d_head=4)
model = Mamba2(config)

B, L, D = 2, 64, 16
x = torch.randn(B, L, D)
y = model(x)

assert y.shape == x.shape