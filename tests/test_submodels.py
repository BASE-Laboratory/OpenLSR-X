"""Unit tests for Generator and Discriminator submodels."""

from __future__ import annotations

import pytest
import torch

from openlsrx.models.submodels import Discriminator, Generator


@pytest.mark.parametrize("scale_factor", [1, 2, 4])
@pytest.mark.parametrize("activation", ["custom_sigmoid", "sigmoid", "clamp", "identity"])
def test_generator_scales_and_activations(scale_factor: int, activation: str) -> None:
    gen = Generator(
        in_channels=1,
        out_channels=16,
        num_blocks=2,
        super_res_factor=scale_factor,
        output_activation=activation,  # type: ignore[arg-type]
    )
    x = torch.randn(2, 1, 16, 16)
    out = gen(x)
    assert out.shape == (2, 1, 16 * scale_factor, 16 * scale_factor)
    if activation in ("custom_sigmoid", "sigmoid", "clamp"):
        assert float(out.detach().min()) >= 0.0
        assert float(out.detach().max()) <= 1.0


def test_generator_rejects_non_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of 2"):
        Generator(super_res_factor=3)


def test_discriminator_logits_shape() -> None:
    disc = Discriminator(in_channels=1, features=[8, 16, 32], adaptive_size=4)
    x = torch.randn(3, 1, 32, 32)
    out = disc(x)
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()
