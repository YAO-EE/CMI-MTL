import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pscan import pscan

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


@dataclass
class MambaConfig:
    d_model: int  #  D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  #  N in paper/comments
    expand_factor: int = 2  #  E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  #  "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128  # width=d_model

    pscan: bool = True  #  use parallel scan mode or sequential mode when training
    use_cuda: bool = False  # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward_unimodal(self, x):
        #  x : (B, L, D)

        #  y : (B, L, D)
        for layer in self.layers:
            x = layer.forward_unimodal(x)

        return x

    def forward(self, x_v, x_q):
        #  x : (B, L, D)

        #  y : (B, L, D)

        for layer in self.layers:
            x_v, x_q = layer(x_v, x_q)

        return x_v, x_q

    def step(self, x, caches):
        #  x : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        #  y : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)
        self.layernorm = LayerNorm(config.d_model, config.rms_norm_eps, config.mup)


    def forward_unimodal(self, x):
        #  x : (B, L, D)

        #  output : (B, L, D)
        output = self.mixer.forward_unimodal(self.norm.forward_unimodal(x))
        output = output + x
        return output

    def forward(self, x_v, x_q):
        #  x : (B, L, D)

        #  output : (B, L, D)
        #  对输入进行规范化
        # output_v, output_q = self.norm(x_v, x_q)
        output_v = self.layernorm(x_v)
        output_q = self.layernorm(x_q)

        output_v, output_q = self.mixer(output_v, output_q)
        output_v = output_v + x_v
        output_q = output_q + x_q
        return output_v, output_q

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs: (B, ED, d_conv-1)

        #  output : (B, D)
        #  cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        #  projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        #  projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        #  projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward_unimodal(self, x):
        #  x : (B, L, D)

        # y : (B, L, D)
        # 获取输入特征的长度
        _, L, _ = x.shape
        # 对x进行线性变换
        xz = self.in_proj(x)  # (B, L, 2*ED)
        # 将xz沿最后一个维度分割为两部分
        x, z = xz.chunk(2, dim=-1)  #  (B, L, ED), (B, L, ED)

        #  x branch
        x = x.transpose(1, 2)  #  (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  #  (B, L, ED)

        # 激活函数
        x = F.silu(x)
        # 状态传播
        y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y)  # (B, L, D)
            return output  # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        #  z branch
        # 激活函数
        # z = F.silu(z)

        # 将y和z逐元素进行相乘，得到最终的输出
        # output = y * z
        output = y
        # 输出进行线性变化改变维度
        # output = self.out_proj(output)  #  (B, L, D)

        return output

    def forward(self, x_v, x_concat):
        # x_v 和 x_q 分别是两种输入 (B, L, D)
        # y : (B, L, D)
        # 获取x_v输入特征的长度
        _, L_v, D_v = x_v.shape
        # 获取x_q输入特征的长度
        # _, L_q, D_q = x_q.shape

        # Visual 模态流处理
        xz_v = self.in_proj(x_v)  # (B, L, 2*ED)
        x_v, z_v = xz_v.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        x_v = x_v.transpose(1, 2)  # (B, ED, L)
        x_v = self.conv1d(x_v)[:, :, :L_v]
        x_v = x_v.transpose(1, 2)  # (B, L, ED)

        x_v = F.silu(x_v)
        y_v = self.ssm(x_v, z_v)

        # z_v = F.silu(z_v)

        # Query 模态流处理
        # xz_q = self.in_proj(x_q)  # (B, L, 2*ED)
        # x_q, z_q = xz_q.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)
        #
        # x_q = x_q.transpose(1, 2)  # (B, ED, L)
        # x_q = self.conv1d(x_q)[:, :, :L_q]
        # x_q = x_q.transpose(1, 2)  # (B, L, ED)
        #
        # x_q = F.silu(x_q)
        # y_q = self.ssm(x_q, z_q)
        #
        # z_q = F.silu(z_q)

        # 单模态融合
        # output = y_v * z_v
        # yz_q = y_q * z_q

        output_concat = self.downsample(x_concat, L_v)
        # 跨模态交互
        # z_q = z_q.transpose(1, 2)
        # z_q = self.convtranspose1d(z_q)[:, :, :L_q]
        # z_q = z_q.transpose(1, 2)
        # z_q = self.downsample(z_q)
        # yz_vq = y_v * z_q

        # z_v = z_v.transpose(1, 2)
        # z_v = self.conv1d_v2q(z_v)[:, :, :L_v]
        # z_v = z_v.transpose(1, 2)
        # z_v = self.upsample(z_v)
        # yz_qv = y_q * z_v

        # y_v = yz_v * yz_vq
        # y_q = yz_q * yz_qv
        # output = output * output_concat
        output = y_v * output_concat

        output = self.out_proj(output)  # (B, L, D)
        # out_q = self.out_proj(y_q)  # (B, L, D)
        # out_v = self.out_proj(yz_v)  # (B, L, D)
        # out_q = self.out_proj(yz_q)  # (B, L, D)

        return output

    def downsample(self, x, output_size):
        return F.adaptive_avg_pool1d(x.permute(0, 2, 1), output_size=output_size).permute(0, 2, 1)

    def upsample(self, x):
        return F.interpolate(x.permute(0, 2, 1), size=577, mode='linear', align_corners=True).permute(0, 2, 1)

    def ssm(self, x, z):
        #  x : (B, L, ED)

        #  y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  #  (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True,
                                         delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2)  # (B, L, ED)

        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  #  (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    #  -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs : (B, ED, d_conv-1)

        #  y : (B, D)
        #  cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  #  (B, ED), (B, ED)

        #  x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  #  (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  #  (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        #  x : (B, ED)
        #  h : (B, ED, N)

        #  y : (B, ED)
        #  h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x)  #  (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))  #  (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  #  (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)

        h = deltaA * h + BX  #  (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h


# 对输入张量进行归一化处理
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        #  https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward_unimodal(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output

    def forward(self, x_v, x_q):
        output_v = x_v * torch.rsqrt(x_v.pow(2).mean(-1, keepdim=True) + self.eps)
        output_q = x_q * torch.rsqrt(x_q.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output_v * self.weight, output_q * self.weight
        else:
            return output_v, output_q

class LayerNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, use_mup: bool = False):
        super().__init__()

        self.eps = eps
        self.use_mup = use_mup

        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        if not self.use_mup:
            return norm_x * self.weight
        else:
            return norm_x
