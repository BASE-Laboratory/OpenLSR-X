"""Unit tests for neural network blocks, normalizations, activations, and VGGLoss."""

from __future__ import annotations

import pytest
import torch

from openlsrx.losses import PixelNorm, VGGLoss, custom_sigmoid
from openlsrx.models.blocks import ConvBlock, ResidualBlock, UpsampleBlock


@pytest.mark.parametrize("norm_type", ["pixel", "batch", "instance", "none"])
def test_conv_block_norms_and_shapes(norm_type: str) -> None:
    block = ConvBlock(
        in_channels=1,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        padding_mode="reflect",
        norm_type=norm_type,  # type: ignore[arg-type]
    )
    x = torch.randn(2, 1, 32, 32)
    out = block(x)
    assert out.shape == (2, 16, 32, 32)
    assert torch.isfinite(out).all()


def test_conv_block_invalid_norm_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported norm_type"):
        ConvBlock(1, 8, kernel_size=3, padding=1, norm_type="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("scale_factor", [2, 4])
def test_upsample_block(scale_factor: int) -> None:
    block = UpsampleBlock(in_channels=16, scale_factor=scale_factor)
    x = torch.randn(2, 16, 12, 12)
    out = block(x)
    assert out.shape == (2, 16, 12 * scale_factor, 12 * scale_factor)


def test_residual_block_preserves_shape() -> None:
    block = ResidualBlock(in_channels=16, norm_type="pixel")
    x = torch.randn(2, 16, 24, 24)
    out = block(x)
    assert out.shape == x.shape


def test_custom_sigmoid_fp16_stability() -> None:
    # Large negative values overflow naive exp(-k*x) in float16, but torch.sigmoid(k*x) stays finite
    x = torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float16)
    y = custom_sigmoid(x, k=2.0)
    assert torch.isfinite(y).all()
    assert float(y[1]) == pytest.approx(0.5, abs=1e-3)


def test_pixel_norm_unit_rms() -> None:
    norm = PixelNorm()
    x = torch.randn(2, 32, 8, 8) * 5.0
    y = norm(x)
    rms = torch.sqrt(torch.mean(y.square(), dim=1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_vgg_loss_offline_and_clean_state_dict() -> None:
    loss_fn = VGGLoss(vgg_layers=8, pretrained=False)
    gen = torch.rand(2, 1, 32, 32, requires_grad=True)
    tgt = torch.rand(2, 1, 32, 32)
    val = loss_fn(gen, tgt)
    val.backward()
    assert val.ndim == 0 and float(val.item()) >= 0.0
    assert gen.grad is not None
    # Frozen VGG19 weights must NOT pollute model checkpoints
    assert loss_fn.state_dict() == {}
