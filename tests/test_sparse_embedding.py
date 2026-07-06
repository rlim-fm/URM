import pytest
import torch

from models.sparse_embedding import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
    _sparse_emb_signsgd_dist,
)


def test_casted_sparse_embedding_eval_mode_no_grad():
    emb = CastedSparseEmbedding(num_embeddings=8, embedding_dim=4, batch_size=3, init_std=0.1, cast_to=torch.float32)
    emb.eval()
    ids = torch.tensor([0, 2, 5])
    out = emb(ids)
    assert out.shape == (3, 4)
    assert not out.requires_grad


def test_casted_sparse_embedding_train_mode_fills_local_state():
    emb = CastedSparseEmbedding(num_embeddings=8, embedding_dim=4, batch_size=3, init_std=0.1, cast_to=torch.float32)
    emb.train()
    ids = torch.tensor([1, 3, 4])
    out = emb(ids)
    assert out.shape == (3, 4)
    assert torch.equal(emb.local_ids, ids)
    assert torch.allclose(emb.local_weights, emb.weights[ids])


def test_casted_sparse_embedding_out_of_range_raises():
    emb = CastedSparseEmbedding(num_embeddings=8, embedding_dim=4, batch_size=3, init_std=0.1, cast_to=torch.float32)
    with pytest.raises(ValueError):
        emb(torch.tensor([0, 8, 2]))
    with pytest.raises(ValueError):
        emb(torch.tensor([-1, 0, 2]))


def test_sparse_emb_signsgd_dist_sign_update_direction():
    num_embeddings, dim = 8, 4
    weights = torch.zeros(num_embeddings, dim)
    local_ids = torch.tensor([1, 3], dtype=torch.int64)
    grad = torch.tensor([[1.0, -1.0, 2.0, -3.0], [0.5, 0.5, -0.5, -0.5]])

    lr = 0.1
    _sparse_emb_signsgd_dist(grad, local_ids, weights, lr=lr, weight_decay=0.0, world_size=1)

    expected = -lr * torch.sign(grad)
    assert torch.allclose(weights[1], expected[0])
    assert torch.allclose(weights[3], expected[1])
    assert torch.all(weights[[0, 2, 4, 5, 6, 7]] == 0)


def test_sparse_emb_signsgd_dist_weight_decay():
    weights = torch.ones(4, 2) * 2.0
    local_ids = torch.tensor([0], dtype=torch.int64)
    grad = torch.zeros(1, 2)

    _sparse_emb_signsgd_dist(grad, local_ids, weights, lr=0.5, weight_decay=0.1, world_size=1)
    # sign(0) == 0, so only weight decay applies: p *= (1 - lr*wd)
    assert torch.allclose(weights[0], torch.full((2,), 2.0 * (1 - 0.5 * 0.1)))


def test_sparse_emb_signsgd_dist_out_of_range_ids_raises():
    weights = torch.zeros(4, 2)
    local_ids = torch.tensor([0, 4], dtype=torch.int64)
    grad = torch.zeros(2, 2)
    with pytest.raises(ValueError):
        _sparse_emb_signsgd_dist(grad, local_ids, weights, lr=0.1, weight_decay=0.0, world_size=1)


def test_casted_sparse_embedding_sign_sgd_optimizer_step():
    emb = CastedSparseEmbedding(num_embeddings=8, embedding_dim=4, batch_size=2, init_std=0.0, cast_to=torch.float32)
    emb.train()
    ids = torch.tensor([1, 5])
    out = emb(ids)
    loss = out.sum()
    loss.backward()

    assert emb.local_weights.grad is not None

    optimizer = CastedSparseEmbeddingSignSGD_Distributed(
        [emb.local_weights, emb.local_ids, emb.weights], world_size=1, lr=0.1, weight_decay=0.0,
    )
    before = emb.weights.clone()
    optimizer.step()
    after = emb.weights

    assert not torch.equal(before[ids], after[ids])
    untouched = torch.tensor([i for i in range(8) if i not in ids.tolist()])
    assert torch.equal(before[untouched], after[untouched])


def test_optimizer_rejects_invalid_lr_or_weight_decay():
    dummy_param = torch.zeros(1, requires_grad=True)
    with pytest.raises(ValueError):
        CastedSparseEmbeddingSignSGD_Distributed([dummy_param], world_size=1, lr=-0.1)
    with pytest.raises(ValueError):
        CastedSparseEmbeddingSignSGD_Distributed([dummy_param], world_size=1, weight_decay=-0.1)
