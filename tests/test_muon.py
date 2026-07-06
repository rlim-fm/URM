import pytest
import torch

from models.muon import msign, Muon

# msign is @torch.compile-decorated; first invocation pays JIT compilation
# cost, making this whole module noticeably slower than the rest of the suite.
pytestmark = pytest.mark.slow


def test_msign_shape_preserved_square():
    G = torch.randn(6, 6)
    out = msign(G, steps=5)
    assert out.shape == G.shape


def test_msign_shape_preserved_rectangular():
    G = torch.randn(4, 8)
    out = msign(G, steps=5)
    assert out.shape == G.shape

    G2 = torch.randn(8, 4)
    out2 = msign(G2, steps=5)
    assert out2.shape == G2.shape


def test_msign_approximately_orthogonalizes():
    # For a well-conditioned square matrix, msign should push it toward
    # having singular values close to 1 (X @ X^T ~ scaled identity-ish).
    torch.manual_seed(0)
    G = torch.eye(5) + 0.01 * torch.randn(5, 5)
    out = msign(G, steps=8)
    gram = out @ out.mT
    diag = torch.diagonal(gram)
    assert torch.allclose(diag, torch.ones_like(diag), atol=0.5)


def test_msign_rejects_1d_input():
    with pytest.raises(ValueError):
        msign(torch.randn(5), steps=1)


def test_muon_step_updates_2d_param():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(4, 4))
    param.grad = torch.randn(4, 4)

    opt = Muon([{"params": [param], "use_muon": True}], lr=0.02, ns_steps=3)
    before = param.data.clone()
    opt.step()
    assert not torch.equal(before, param.data)


def test_muon_step_adamw_fallback_for_non_muon_group():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(4))
    param.grad = torch.randn(4)

    opt = Muon([{"params": [param], "use_muon": False}], lr=0.01)
    before = param.data.clone()
    opt.step()
    assert not torch.equal(before, param.data)
    assert opt.param_groups[0]["step"] == 1
