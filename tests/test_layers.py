import math

import pytest
import torch

# models/layers.py unconditionally imports flash_attn(_interface) at module
# scope (even though most classes here never call it), so importing this
# module at all requires flash_attn to be installed. Skip the whole file
# cleanly when it isn't, rather than failing collection.
layers = pytest.importorskip("models.layers", reason="models.layers requires flash_attn to import")

rotate_half = layers.rotate_half
apply_rotary_pos_emb = layers.apply_rotary_pos_emb
CastedLinear = layers.CastedLinear
CastedEmbedding = layers.CastedEmbedding
RotaryEmbedding = layers.RotaryEmbedding
TropicalLinear = layers.TropicalLinear
DeepSet = layers.DeepSet
TropicalAttention = layers.TropicalAttention
SwiGLU = layers.SwiGLU
ConvSwiGLU = layers.ConvSwiGLU
FullyLinearGLU = layers.FullyLinearGLU
LinearGLU = layers.LinearGLU
SiLU = layers.SiLU
LinearSwish = layers.LinearSwish
ReLU = layers.ReLU
rms_norm = layers.rms_norm


def test_rotate_half_shape_and_values():
    x = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8)
    out = rotate_half(x)
    assert out.shape == x.shape
    x1, x2 = x[..., :4], x[..., 4:]
    assert torch.allclose(out, torch.cat((-x2, x1), dim=-1))


def test_apply_rotary_pos_emb_zero_angle_is_identity():
    bs, seq_len, num_heads, head_dim = 2, 4, 2, 8
    q = torch.randn(bs, seq_len, num_heads, head_dim)
    k = torch.randn(bs, seq_len, num_heads, head_dim)
    cos = torch.ones(seq_len, head_dim)
    sin = torch.zeros(seq_len, head_dim)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q_out, q, atol=1e-5)
    assert torch.allclose(k_out, k, atol=1e-5)


def test_casted_linear_shape():
    lin = CastedLinear(in_features=8, out_features=4, bias=True)
    x = torch.randn(2, 8)
    out = lin(x)
    assert out.shape == (2, 4)


def test_casted_linear_no_bias():
    lin = CastedLinear(in_features=8, out_features=4, bias=False)
    assert lin.bias is None


def test_casted_embedding_cast_to():
    emb = CastedEmbedding(num_embeddings=10, embedding_dim=6, init_std=0.1, cast_to=torch.float32)
    idx = torch.tensor([0, 1, 2])
    out = emb(idx)
    assert out.shape == (3, 6)
    assert out.dtype == torch.float32


def test_rotary_embedding_shapes():
    dim, max_pos = 8, 16
    rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_pos, base=10000.0)
    cos, sin = rope()
    assert cos.shape == (max_pos, dim)
    assert sin.shape == (max_pos, dim)


def test_tropical_linear_near_identity_diag():
    tl = TropicalLinear(4, 4, diag_init=0.0, offdiag_init=-9.0, init_jitter_std=0.0)
    diag = torch.diagonal(tl.W)
    off_diag_mask = ~torch.eye(4, dtype=torch.bool)
    assert torch.allclose(diag, torch.zeros(4))
    assert torch.all(tl.W[off_diag_mask] == -9.0)


def test_tropical_linear_forward_shape():
    tl = TropicalLinear(4, 6)
    x = torch.randn(2, 5, 4)
    out = tl(x)
    assert out.shape == (2, 5, 6)


def test_deepset_permutation_invariance():
    ds = DeepSet(dim=8)
    x = torch.randn(2, 5, 8)
    perm = torch.randperm(5)
    out1 = ds(x)
    out2 = ds(x[:, perm, :])
    assert out1.shape == (2, 1, 8)
    assert torch.allclose(out1, out2, atol=1e-5)


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("tropical_norm", ["none", "max", "learnable"])
@pytest.mark.parametrize("tropical_proj,tropical_qkv_proj", [(True, False), (False, True), (False, False)])
def test_tropical_attention_forward_shapes(symmetric, tropical_norm, tropical_proj, tropical_qkv_proj):
    hidden_size, num_heads = 16, 2
    attn = TropicalAttention(
        hidden_size=hidden_size,
        head_dim=hidden_size // num_heads,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        symmetric=symmetric,
        tropical_norm=tropical_norm,
        tropical_proj=tropical_proj,
        tropical_qkv_proj=tropical_qkv_proj,
    )
    x = torch.randn(2, 5, hidden_size)
    out = attn(cos_sin=None, hidden_states=x)
    assert out.shape == (2, 5, hidden_size)

    out2, scores = attn.forward_with_scores(x)
    assert out2.shape == (2, 5, hidden_size)
    assert scores.shape[0] == 2 * num_heads


def test_tropical_attention_invalid_norm_raises():
    with pytest.raises(ValueError):
        TropicalAttention(hidden_size=16, head_dim=8, num_heads=2, num_key_value_heads=2, tropical_norm="bogus")


def test_tropical_attention_head_dim_mismatch_raises():
    with pytest.raises(ValueError):
        TropicalAttention(hidden_size=16, head_dim=4, num_heads=2, num_key_value_heads=2)


@pytest.mark.parametrize("cls,kwargs", [
    (SwiGLU, dict(hidden_size=16, expansion=2.0)),
    (ConvSwiGLU, dict(hidden_size=16, expansion=2.0)),
    (FullyLinearGLU, dict(hidden_size=16, expansion=2.0)),
    (LinearGLU, dict(hidden_size=16, expansion=2.0)),
    (SiLU, dict(hidden_size=16, expansion=2.0)),
    (ReLU, dict(hidden_size=16, expansion=2.0)),
])
def test_mlp_blocks_preserve_shape(cls, kwargs):
    hidden_size = kwargs["hidden_size"]
    module = cls(**kwargs)
    x = torch.randn(2, 5, hidden_size)
    out = module(x)
    assert out.shape == (2, 5, hidden_size)


@pytest.mark.parametrize("reverse", [True, False])
def test_linear_swish_shape(reverse):
    module = LinearSwish(hidden_size=16, reverse=reverse)
    x = torch.randn(2, 5, 16)
    out = module(x)
    assert out.shape == (2, 5, 16)


def test_rms_norm_unit_variance():
    x = torch.randn(2, 5, 32) * 10 + 3
    out = rms_norm(x, variance_epsilon=1e-5)
    variance = out.float().square().mean(-1)
    assert torch.allclose(variance, torch.ones_like(variance), atol=1e-2)


def test_attention_class_forward():
    Attention = layers.Attention
    hidden_size, num_heads = 16, 2
    attn = Attention(hidden_size=hidden_size, head_dim=hidden_size // num_heads, num_heads=num_heads, num_key_value_heads=num_heads)
    x = torch.randn(2, 5, hidden_size)
    out = attn(cos_sin=None, hidden_states=x)
    assert out.shape == (2, 5, hidden_size)
