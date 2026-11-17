"""Lightweight Shifted-Window Transformer (SwinIR) generator for 2D/2.5D XCT super-resolution.

Implements Residual Swin Transformer Blocks (RSTB) with shifted-window 2D self-attention,
learnable relative position biases, automatic reflect padding for arbitrary slice dimensions,
central-slice global bicubic residual learning, and ``ArbitraryScaleUpsample`` supporting both
power-of-two (``2x, 4x, 8x``) and fractional synchrotron magnification ratios (``2.5x, 5x``).

Reference:
    Liang, J. et al., "SwinIR: Image Restoration Using Swin Transformer", ICCV Workshops (2021).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..losses import custom_sigmoid
from .rrdb import ArbitraryScaleUpsample
from .submodels import OutputActivation


def _window_partition(x: Tensor, window_size: int) -> Tensor:
    """Partition ``(B, H, W, C)`` into ``(B * nW, window_size, window_size, C)``."""
    b, h, w, c = x.shape
    ws = window_size
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)


def _window_reverse(windows: Tensor, window_size: int, h: int, w: int) -> Tensor:
    """Reverse ``(B * nW, window_size, window_size, C)`` back into ``(B, H, W, C)``."""
    ws = window_size
    n_win_per_img = (h // ws) * (w // ws)
    b = int(windows.shape[0] // n_win_per_img)
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention2D(nn.Module):
    """Window-based 2D multi-head self-attention with learnable 2D relative position bias."""

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        ws = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flat = torch.flatten(coords, 1)  # (2, ws*ws)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        rel_coords[:, :, 0] += ws - 1
        rel_coords[:, :, 1] += ws - 1
        rel_coords[:, :, 0] *= 2 * ws - 1
        rel_pos_index = rel_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", rel_pos_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b_win, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b_win, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        rel_idx = self.get_buffer("relative_position_index").view(-1)
        bias = self.relative_position_bias_table[rel_idx].view(n, n, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            n_win = mask.shape[0]
            attn = attn.view(b_win // n_win, n_win, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_win, n, c)
        return self.proj(out)


class SwinTransformerLayer(nn.Module):
    """Single Swin Transformer layer with optional cyclic window shift and pre-LayerNorm MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        hidden_dim = int(round(dim * mlp_ratio))

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention2D(dim=dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    @staticmethod
    def _build_shift_mask(
        h: int, w: int, window_size: int, shift_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        img_mask = torch.zeros((1, h, w, 1), device=device, dtype=dtype)
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = _window_partition(img_mask, window_size).view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward on ``(B, H, W, C)`` where ``H, W`` are multiples of ``window_size``."""
        _, h, w, c = x.shape
        shortcut = x
        x_norm = self.norm1(x)

        eff_shift = self.shift_size if min(h, w) > self.window_size else 0
        if eff_shift > 0:
            shifted = torch.roll(x_norm, shifts=(-eff_shift, -eff_shift), dims=(1, 2))
            attn_mask: Tensor | None = self._build_shift_mask(
                h, w, self.window_size, eff_shift, device=x.device, dtype=x.dtype
            )
        else:
            shifted = x_norm
            attn_mask = None

        x_windows = _window_partition(shifted, self.window_size).view(
            -1, self.window_size * self.window_size, c
        )
        attn_windows = self.attn(x_windows, mask=attn_mask).view(
            -1, self.window_size, self.window_size, c
        )
        merged = _window_reverse(attn_windows, self.window_size, h, w)

        if eff_shift > 0:
            merged = torch.roll(merged, shifts=(eff_shift, eff_shift), dims=(1, 2))

        x = shortcut + merged
        return x + self.mlp(self.norm2(x))


class ResidualSwinBlock(nn.Module):
    """Residual Swin Transformer Block (RSTB) combining alternating W-MSA/SW-MSA with a 3x3 Conv."""

    def __init__(
        self,
        dim: int = 48,
        depth: int = 2,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.res_scale = float(res_scale)
        self.layers = nn.ModuleList(
            [
                SwinTransformerLayer(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else (window_size // 2),
                    mlp_ratio=mlp_ratio,
                )
                for i in range(depth)
            ]
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, padding_mode="reflect")

    def forward(self, x: Tensor) -> Tensor:
        """Forward on ``(B, C, H, W)`` with automatic padding to ``window_size``."""
        _, _, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        work = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h > 0 or pad_w > 0) else x

        tokens = work.permute(0, 2, 3, 1).contiguous()
        for layer in self.layers:
            tokens = layer(tokens)
        feat = tokens.permute(0, 3, 1, 2).contiguous()
        if pad_h > 0 or pad_w > 0:
            feat = feat[:, :, :h, :w]
        return x + self.res_scale * self.conv(feat)


class SwinIRGenerator(nn.Module):
    """SwinIR Generator for 2D and 2.5D synchrotron XCT super-resolution."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 48,
        num_blocks: int = 4,
        depth_per_block: int = 2,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        super_res_factor: float | int = 2,
        sigmoid_k: float = 1.0,
        output_activation: OutputActivation = "clamp",
        out_img_channels: int = 1,
    ) -> None:
        super().__init__()
        if output_activation not in ("custom_sigmoid", "sigmoid", "clamp", "identity"):
            raise ValueError(
                f"Unsupported output_activation={output_activation!r}. "
                "Expected 'custom_sigmoid', 'sigmoid', 'clamp', or 'identity'."
            )
        # Ensure embed_dim is divisible by num_heads
        eff_heads = int(num_heads)
        if out_channels % eff_heads != 0:
            eff_heads = max(1, math.gcd(int(out_channels), eff_heads))

        self.in_channels = int(in_channels)
        self.out_img_channels = int(out_img_channels)
        self.out_channels = int(out_channels)
        self.num_blocks = int(num_blocks)
        self.depth_per_block = int(depth_per_block)
        self.num_heads = eff_heads
        self.window_size = int(window_size)
        self.super_res_factor = float(super_res_factor)
        self.sigmoid_k = float(sigmoid_k)
        self.output_activation: OutputActivation = output_activation

        self.conv_first = nn.Conv2d(
            self.in_channels, self.out_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.trunk = nn.Sequential(
            *[
                ResidualSwinBlock(
                    dim=self.out_channels,
                    depth=self.depth_per_block,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.trunk_conv = nn.Conv2d(
            self.out_channels, self.out_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.upsample = ArbitraryScaleUpsample(
            channels=self.out_channels, scale_factor=self.super_res_factor
        )
        final_conv = nn.Conv2d(
            self.out_channels, self.out_img_channels, 3, 1, 1, padding_mode="reflect"
        )
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)
        final_conv.weight.data.mul_(0.1)
        self.hr_head = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            final_conv,
        )

    def _apply_output_activation(self, x: Tensor) -> Tensor:
        if self.output_activation == "custom_sigmoid":
            return custom_sigmoid(x, k=self.sigmoid_k)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "clamp":
            return torch.clamp(x, 0.0, 1.0)
        return x

    def forward(self, x: Tensor) -> Tensor:
        target_h = int(round(x.shape[-2] * self.super_res_factor))
        target_w = int(round(x.shape[-1] * self.super_res_factor))
        if x.shape[1] == self.out_img_channels:
            center = x
        else:
            c_mid = x.shape[1] // 2
            center = x[:, c_mid : c_mid + self.out_img_channels, :, :]
        base = F.interpolate(center, size=(target_h, target_w), mode="bicubic", align_corners=False)

        feat = self.conv_first(x)
        feat = feat + self.trunk_conv(self.trunk(feat))
        up = self.upsample(feat)
        residual = self.hr_head(up)
        if self.output_activation in ("clamp", "identity"):
            return self._apply_output_activation(base + residual)
        return self._apply_output_activation(residual)
