"""Unit tests for the pure-logic pieces of pretrain.py.

pretrain.py imports hydra/omegaconf/wandb/coolname/adam_atan2_pytorch at
module scope for its CLI entrypoint, so importing it at all requires that
full environment. In a minimal dev environment this whole file is skipped;
in a full environment (matching requirements.txt) it exercises EMAHelper,
the LR schedule, unwrap(), and the pydantic config models without needing a
real (torchrun/distributed) training launch.
"""
import math

import pytest
import torch
from torch import nn

pretrain = pytest.importorskip("pretrain", reason="pretrain.py requires hydra/wandb/adam_atan2_pytorch to import")


def test_unwrap_plain_module():
    m = nn.Linear(2, 2)
    assert pretrain.unwrap(m) is m


def test_unwrap_ddp_like_wrapper():
    inner = nn.Linear(2, 2)

    class _FakeDDP(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    wrapped = _FakeDDP(inner)
    assert pretrain.unwrap(wrapped) is inner


def test_ema_helper_register_update_ema_round_trip():
    model = nn.Linear(4, 4)
    ema = pretrain.EMAHelper(mu=0.5)
    ema.register(model)

    original_weight = model.weight.data.clone()
    with torch.no_grad():
        model.weight.add_(1.0)

    ema.update(model)
    # shadow should now be an average of original and updated weights
    expected = 0.5 * (original_weight + 1.0) + 0.5 * original_weight
    assert torch.allclose(ema.shadow["weight"], expected, atol=1e-5)

    ema.ema(model)
    assert torch.allclose(model.weight.data, expected, atol=1e-5)


def test_ema_helper_state_dict_round_trip():
    model = nn.Linear(3, 3)
    ema = pretrain.EMAHelper()
    ema.register(model)
    state = ema.state_dict()

    ema2 = pretrain.EMAHelper()
    ema2.load_state_dict(state)
    assert set(ema2.shadow.keys()) == set(state.keys())


def test_ema_copy_does_not_mutate_original():
    model = nn.Linear(3, 3)
    ema = pretrain.EMAHelper(mu=0.9)
    ema.register(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)

    original_weight = model.weight.data.clone()
    copy = ema.ema_copy(model)
    assert not torch.allclose(copy.weight.data, original_weight)
    assert torch.allclose(model.weight.data, original_weight)


def test_cosine_schedule_warmup_ramps_linearly():
    lr_at_0 = pretrain.cosine_schedule_with_warmup_lr_lambda(
        0, base_lr=1.0, num_warmup_steps=10, num_training_steps=100,
    )
    lr_at_half_warmup = pretrain.cosine_schedule_with_warmup_lr_lambda(
        5, base_lr=1.0, num_warmup_steps=10, num_training_steps=100,
    )
    assert lr_at_0 == 0.0
    assert math.isclose(lr_at_half_warmup, 0.5, rel_tol=1e-6)


def test_cosine_schedule_decays_after_warmup():
    lr_end_of_warmup = pretrain.cosine_schedule_with_warmup_lr_lambda(
        10, base_lr=1.0, num_warmup_steps=10, num_training_steps=100,
    )
    lr_mid = pretrain.cosine_schedule_with_warmup_lr_lambda(
        55, base_lr=1.0, num_warmup_steps=10, num_training_steps=100,
    )
    lr_end = pretrain.cosine_schedule_with_warmup_lr_lambda(
        100, base_lr=1.0, num_warmup_steps=10, num_training_steps=100,
    )
    assert lr_end_of_warmup >= lr_mid >= lr_end
    assert lr_end >= 0.0


def test_cosine_schedule_respects_min_ratio():
    lr_end = pretrain.cosine_schedule_with_warmup_lr_lambda(
        100, base_lr=2.0, num_warmup_steps=0, num_training_steps=100, min_ratio=0.1,
    )
    assert lr_end >= 2.0 * 0.1 - 1e-6


def _minimal_pretrain_config_kwargs():
    return dict(
        arch=pretrain.ArchConfig(name="urm.urm@URM", loss=pretrain.LossConfig(name="losses@ACTLossHead", loss_type="stablemax_cross_entropy")),
        data_path="data/fake",
        global_batch_size=4,
        epochs=1,
        lr=1e-4,
        lr_min_ratio=1.0,
        lr_warmup_steps=10,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        target_q_update_every=4,
        puzzle_emb_lr=1e-2,
        puzzle_emb_weight_decay=0.1,
    )


def test_pretrain_config_valid_construction():
    config = pretrain.PretrainConfig(**_minimal_pretrain_config_kwargs())
    assert config.global_batch_size == 4
    assert config.ema is False
    assert config.evaluators == []


def test_arch_config_allows_extra_fields():
    arch = pretrain.ArchConfig(
        name="urm.urm@URM",
        loss=pretrain.LossConfig(name="losses@ACTLossHead", loss_type="stablemax_cross_entropy"),
        hidden_size=32,
        num_heads=2,
    )
    assert arch.hidden_size == 32
    assert arch.num_heads == 2
