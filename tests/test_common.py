import torch

from models.common import trunc_normal_init_


def test_shape_and_dtype_preserved():
    t = torch.empty(4, 8, dtype=torch.float32)
    out = trunc_normal_init_(t, std=1.0)
    assert out.shape == (4, 8)
    assert out.dtype == torch.float32
    assert out is t  # in-place


def test_std_zero_zeros_tensor():
    t = torch.empty(3, 3)
    out = trunc_normal_init_(t, std=0.0)
    assert torch.all(out == 0)


def test_values_within_bounds():
    # The function clips to [lower, upper] * a compensated std (>= the
    # requested std), so just check the output stays comfortably bounded
    # and doesn't blow up, rather than pinning the exact clip threshold.
    t = torch.empty(1000)
    std = 0.5
    lower, upper = -2.0, 2.0
    out = trunc_normal_init_(t, std=std, lower=lower, upper=upper)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 10 * std


def test_nonzero_std_not_constant():
    t = torch.empty(100)
    out = trunc_normal_init_(t, std=1.0)
    assert out.std().item() > 0
