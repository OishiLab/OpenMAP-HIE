"""
ControlNet-style conditioning for the ANDi DDPM U-Net.

- Frozen pretrained U-Net; trainable control encoder + zero-initialized convs.
- CFG (Classifier-Free Guidance): training-time control dropout + inference
  noise_uncond + scale * (noise_cond - noise_uncond) as in the ControlNet paper.

Control input channel layout / which CSV columns map to which channels is left
to config via ``control_column_indices`` (placeholder until wired to data).
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from modules import DoubleConv, Down, SelfAttention, UNet


def zero_module(module: nn.Module) -> nn.Module:
    """Zero all parameters and buffers; used for ControlNet output convs."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class DownControl(nn.Module):
    """Downsample path without time embedding — zero control stays zero."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class ControlNetUNet(nn.Module):
    """
    U-Net noise predictor with ControlNet side branch.

    Base U-Net is frozen; only control stem, encoder copy, and zero convs train.

    Forward
    -------
    Training: pass ``control`` and optional ``cfg_dropout_prob`` (e.g. 0.1).
    Inference with CFG: pass ``control`` and ``cfg_scale`` > 1.0 (set model.eval()).
    """

    def __init__(
        self,
        base_unet: UNet,
        control_channels: int,
        img_size: int = 128,
    ):
        super().__init__()
        self.control_channels = int(control_channels)
        self.img_size = int(img_size)
        self.time_dim = base_unet.time_dim

        self.base = copy.deepcopy(base_unet)
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

        c = self.control_channels
        iz = self.img_size
        self.zero_in = zero_module(nn.Conv2d(c, 32, kernel_size=1))
        self.c_inc = DoubleConv(32, 32)
        self.c_inc2 = DoubleConv(32, 32, residual=True)
        self.z1 = zero_module(nn.Conv2d(32, 32, kernel_size=1))

        self.c_down1 = DownControl(32, 64)
        self.z2 = zero_module(nn.Conv2d(64, 64, kernel_size=1))

        self.c_down2 = DownControl(64, 128)
        self.c_sa1 = SelfAttention(128, int(iz / 4))
        self.z3 = zero_module(nn.Conv2d(128, 128, kernel_size=1))

        self.c_down3 = DownControl(128, 256)
        self.c_sa2 = SelfAttention(256, int(iz / 8))
        self.z4 = zero_module(nn.Conv2d(256, 256, kernel_size=1))

        self.c_down4 = DownControl(256, 256)
        self.c_sa3 = SelfAttention(256, int(iz / 16))
        self.z5 = zero_module(nn.Conv2d(256, 256, kernel_size=1))

        # Mid / bottleneck injection (after base.bot1, which outputs 512 channels)
        self.c_mid = DoubleConv(256, 512)
        self.z_mid = zero_module(nn.Conv2d(512, 512, kernel_size=1))

    def train(self, mode: bool = True):
        """Keep frozen base in eval (BN/GN stats); train only control branch."""
        super().train(mode)
        self.base.eval()
        return self

    def load_base_from_checkpoint(
        self,
        path: str,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> nn.Module:
        """Load pretrained DDPM U-Net weights into the frozen base (and sync copy)."""
        state = torch.load(path, map_location=map_location or "cpu")
        self.base.load_state_dict(state, strict=strict)
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False
        return self

    def _encode_control(self, control: torch.Tensor) -> List[torch.Tensor]:
        """Control branch features at each injection scale."""
        h = self.zero_in(control)
        h = self.c_inc(h)
        h = self.c_inc2(h)
        r1 = self.z1(h)

        h = self.c_down1(h)
        r2 = self.z2(h)

        h = self.c_down2(h)
        h = self.c_sa1(h)
        r3 = self.z3(h)

        h = self.c_down3(h)
        h = self.c_sa2(h)
        r4 = self.z4(h)

        h = self.c_down4(h)
        h = self.c_sa3(h)
        r5 = self.z5(h)

        h_mid = self.c_mid(h)
        r_mid = self.z_mid(h_mid)
        return [r1, r2, r3, r4, r5, r_mid]

    def _forward_core(
        self, x: torch.Tensor, t_emb: torch.Tensor, residuals: List[torch.Tensor]
    ) -> torch.Tensor:
        r1, r2, r3, r4, r5, r_mid = residuals

        # Resolution weighting (ControlNet paper): w_i = 64 / h_i
        # where h_i is spatial size of the i-th block feature map.
        def w(res: torch.Tensor) -> torch.Tensor:
            return res.new_tensor(64.0 / float(res.shape[-1]))

        x1 = self.base.inc(x)
        x1 = self.base.inc2(x1) + w(r1) * r1

        x2 = self.base.down1(x1, t_emb) + w(r2) * r2

        x3 = self.base.down2(x2, t_emb)
        x3 = self.base.sa1(x3) + w(r3) * r3

        x4 = self.base.down3(x3, t_emb)
        x4 = self.base.sa2(x4) + w(r4) * r4

        x5 = self.base.down4(x4, t_emb)
        x5 = self.base.sa3(x5) + w(r5) * r5

        x5 = self.base.bot1(x5) + w(r_mid) * r_mid
        x5 = self.base.bot3(x5)

        x = self.base.sa4(x5)
        x = self.base.up1(x, x4, t_emb)
        x = self.base.sa5(x)
        x = self.base.up2(x, x3, t_emb)
        x = self.base.sa6(x)
        x = self.base.up3(x, x2, t_emb)
        x = self.base.up4(x, x1, t_emb)
        return self.base.outc(x)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        cfg_dropout_prob: float = 0.0,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Predict noise epsilon.

        Parameters
        ----------
        x, t : same as base UNet
        control : (B, control_channels, H, W). If None, zeros (unconditional).
        cfg_dropout_prob : during training, replace control with zeros with this prob.
        cfg_scale : if not None and != 1, run cond + uncond and combine (inference).
        """
        t_emb = self.base.pos_encoding(
            t.unsqueeze(-1).type(x.dtype), self.base.time_dim
        )
        B, _, H, W = x.shape
        device, dtype = x.device, x.dtype

        if control is None:
            control = torch.zeros(
                B, self.control_channels, H, W, device=device, dtype=dtype
            )
        else:
            control = control.to(device=device, dtype=dtype)
            if control.shape[0] != B:
                raise ValueError(
                    f"control batch {control.shape[0]} != x batch {B}"
                )

        if self.training and cfg_dropout_prob > 0.0:
            if torch.rand(1, device=device).item() < cfg_dropout_prob:
                control = torch.zeros_like(control)

        if cfg_scale is not None and cfg_scale != 1.0 and not self.training:
            res_c = self._encode_control(control)
            res_u = self._encode_control(torch.zeros_like(control))
            eps_c = self._forward_core(x, t_emb, res_c)
            eps_u = self._forward_core(x, t_emb, res_u)
            return eps_u + cfg_scale * (eps_c - eps_u)

        residuals = self._encode_control(control)
        return self._forward_core(x, t_emb, residuals)

    def predict_eps_cond_uncond(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        control: Optional[torch.Tensor],
    ) -> tuple:
        """Inference-only: ``(eps_cond, eps_uncond)`` from one cond + one uncond forward.

        Uncond uses a zero control map (same as CFG dropout). Both predictions
        share ``x`` and ``t``.
        """
        if self.training:
            raise RuntimeError(
                "predict_eps_cond_uncond is inference-only (model.eval())."
            )
        t_emb = self.base.pos_encoding(
            t.unsqueeze(-1).type(x.dtype), self.base.time_dim
        )
        B, _, H, W = x.shape
        device, dtype = x.device, x.dtype
        if control is None:
            control = torch.zeros(
                B, self.control_channels, H, W, device=device, dtype=dtype
            )
        else:
            control = control.to(device=device, dtype=dtype)
            if control.shape[0] != B:
                raise ValueError(
                    f"control batch {control.shape[0]} != x batch {B}"
                )
        res_c = self._encode_control(control)
        res_u = self._encode_control(torch.zeros_like(control))
        eps_c = self._forward_core(x, t_emb, res_c)
        eps_u = self._forward_core(x, t_emb, res_u)
        return eps_c, eps_u

    def predict_eps_cfg_pair(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        control: Optional[torch.Tensor],
        cfg_scale_small: float,
        cfg_scale_large: float,
    ) -> tuple:
        """Inference-only: one cond + uncond forward, two CFG-combined epsilons.

        ``eps = eps_uncond + s * (eps_cond - eps_uncond)`` for each scale.
        """
        eps_c, eps_u = self.predict_eps_cond_uncond(x, t, control)
        delta = eps_c - eps_u
        return (
            eps_u + float(cfg_scale_small) * delta,
            eps_u + float(cfg_scale_large) * delta,
        )

    def trainable_parameters(self):
        """Parameters to optimize (everything except frozen base)."""
        base_ids = {id(p) for p in self.base.parameters()}
        for p in self.parameters():
            if id(p) not in base_ids and p.requires_grad:
                yield p

    def state_dict_trainable(self):
        """Checkpoint only trainable ControlNet weights (smaller saves)."""
        d = self.state_dict()
        base_keys = {k for k in d if k.startswith("base.")}
        return {k: v for k, v in d.items() if k not in base_keys}

    def load_state_dict_trainable(self, state: dict, strict: bool = True):
        full = self.state_dict()
        for k, v in state.items():
            if k in full:
                full[k] = v
        self.load_state_dict(full, strict=False)
