import torch
from torch import nn

from models.losses import (
    IGNORE_LABEL_ID,
    stablemax_cross_entropy,
    softmax_cross_entropy,
    ACTLossHead,
)


def _labels_with_ignore(shape, vocab_size, ignore_frac=0.3):
    labels = torch.randint(0, vocab_size, shape)
    mask = torch.rand(shape) < ignore_frac
    labels[mask] = IGNORE_LABEL_ID
    return labels


def test_stablemax_cross_entropy_shape_and_finite():
    logits = torch.randn(2, 5, 10)
    labels = _labels_with_ignore((2, 5), 10)
    loss = stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)
    assert loss.shape == (2, 5)
    assert torch.isfinite(loss).all()
    assert (loss >= 0).all()


def test_stablemax_cross_entropy_ignored_positions_are_zero():
    logits = torch.randn(2, 5, 10)
    labels = torch.full((2, 5), IGNORE_LABEL_ID)
    loss = stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)
    assert torch.all(loss == 0)


def test_softmax_cross_entropy_shape_and_finite():
    logits = torch.randn(2, 5, 10)
    labels = _labels_with_ignore((2, 5), 10)
    loss = softmax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)
    assert loss.shape == (2, 5)
    assert torch.isfinite(loss).all()


class _FakeCarry:
    def __init__(self, current_data, halted, steps):
        self.current_data = current_data
        self.halted = halted
        self.steps = steps


class _FakeInnerModel(nn.Module):
    """Minimal stand-in for a real recurrent model's forward contract, so
    ACTLossHead's metric/loss computation can be tested without flash_attn.
    """

    def __init__(self, vocab_size=10, seq_len=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.linear = nn.Linear(1, 1)  # so the module has a parameter

    def initial_carry(self, batch):
        bs = batch["labels"].shape[0]
        return _FakeCarry(
            current_data=batch,
            halted=torch.ones(bs, dtype=torch.bool),
            steps=torch.zeros(bs, dtype=torch.int32),
        )

    def forward(self, carry, batch, **kwargs):
        labels = batch["labels"]
        bs, seq_len = labels.shape
        logits = torch.randn(bs, seq_len, self.vocab_size, requires_grad=True)
        q_halt_logits = torch.randn(bs, requires_grad=True)
        q_continue_logits = torch.randn(bs, requires_grad=True)
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }
        return carry, outputs


def _make_batch(bs=4, seq_len=5, vocab_size=10):
    return {
        "labels": _labels_with_ignore((bs, seq_len), vocab_size),
    }


def test_act_loss_head_metrics_and_loss():
    inner = _FakeInnerModel()
    head = ACTLossHead(inner, loss_type="stablemax_cross_entropy")
    batch = _make_batch()
    carry = head.initial_carry(batch)

    new_carry, total_loss, metrics, returned_outputs, all_halted = head(
        return_keys={"preds"}, carry=carry, batch=batch,
    )

    assert torch.isfinite(total_loss)
    for key in ("count", "accuracy", "exact_accuracy", "q_halt_accuracy", "steps", "lm_loss", "q_halt_loss"):
        assert key in metrics
    assert "preds" in returned_outputs
    assert bool(all_halted) is True


def test_act_loss_head_return_raw_outputs():
    inner = _FakeInnerModel()
    head = ACTLossHead(inner, loss_type="softmax_cross_entropy")
    batch = _make_batch()
    carry = head.initial_carry(batch)

    _, _, _, returned_outputs, _ = head(
        return_keys=set(), carry=carry, batch=batch, return_raw_outputs=True,
    )
    assert "raw_outputs" in returned_outputs
    assert "logits" in returned_outputs["raw_outputs"]


def test_act_loss_head_aux_and_router_metrics_passthrough():
    class _InnerWithAux(_FakeInnerModel):
        def forward(self, carry, batch, **kwargs):
            carry, outputs = super().forward(carry, batch, **kwargs)
            outputs["moe_aux_loss"] = torch.tensor(0.1, requires_grad=True)
            outputs["router_metrics"] = {"entropy": torch.tensor(0.5)}
            return carry, outputs

    inner = _InnerWithAux()
    head = ACTLossHead(inner, loss_type="stablemax_cross_entropy")
    batch = _make_batch()
    carry = head.initial_carry(batch)

    _, total_loss, metrics, _, _ = head(return_keys=set(), carry=carry, batch=batch)
    assert "moe_aux_loss" in metrics
    assert "router/entropy" in metrics
    assert torch.isfinite(total_loss)
