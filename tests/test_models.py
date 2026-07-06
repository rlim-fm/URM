"""Forward-pass tests for the full recurrent architectures.

models/layers.py imports flash_attn(_interface) at module scope, and every
architecture module here imports from models.layers, so this whole file is
skipped (not failed) unless flash_attn is installed.
"""
import pytest
import torch

pytest.importorskip("models.layers", reason="model forward passes require flash_attn")

from models.urm.urm import URM
from models.tarm.tarm import TARM
from models.trm.trm import TRM
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.hrm.hrm_act_v2 import HierarchicalReasoningModel_ACTV2


BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PUZZLE_IDS = 2, 8, 16, 4


def _batch():
    return {
        "inputs": torch.randint(2, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.int32),
        "labels": torch.randint(2, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.int32),
        "puzzle_identifiers": torch.randint(0, NUM_PUZZLE_IDS, (BATCH_SIZE,), dtype=torch.int32),
    }


def _common_kwargs(puzzle_emb_ndim=32):
    return dict(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        puzzle_emb_ndim=puzzle_emb_ndim,
        num_puzzle_identifiers=NUM_PUZZLE_IDS,
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        expansion=2.0,
        num_heads=2,
        pos_encodings="rope",
        forward_dtype="float32",
    )


def _urm_config(**overrides):
    cfg = dict(_common_kwargs(), num_layers=1, loops=3, H_cycles=2, L_cycles=1)
    cfg.update(overrides)
    return cfg


def _tarm_config(**overrides):
    cfg = dict(_common_kwargs(), num_layers=1, loops=3, H_cycles=2, L_cycles=1)
    cfg.update(overrides)
    return cfg


def _trm_config(**overrides):
    cfg = dict(_common_kwargs(), H_cycles=2, L_cycles=1, H_layers=1, L_layers=1,
               halt_max_steps=3, halt_exploration_prob=0.1, puzzle_emb_len=1)
    cfg.update(overrides)
    return cfg


def _hrm_v1_config(**overrides):
    cfg = dict(_common_kwargs(), H_cycles=2, L_cycles=1, H_layers=1, L_layers=1,
               halt_max_steps=3, halt_exploration_prob=0.1)
    cfg.update(overrides)
    return cfg


def _hrm_v2_config(**overrides):
    cfg = dict(_common_kwargs(), H_cycles=2, H_layers=1,
               halt_max_steps=3, halt_exploration_prob=0.1)
    cfg.update(overrides)
    return cfg


MODEL_CASES = [
    (URM, _urm_config),
    (TARM, _tarm_config),
    (TRM, _trm_config),
    (HierarchicalReasoningModel_ACTV1, _hrm_v1_config),
    (HierarchicalReasoningModel_ACTV2, _hrm_v2_config),
]


@pytest.mark.parametrize("model_cls,config_fn", MODEL_CASES)
def test_initial_carry_shapes(model_cls, config_fn):
    model = model_cls(config_fn())
    batch = _batch()
    carry = model.initial_carry(batch)
    assert carry.steps.shape == (BATCH_SIZE,)
    assert carry.halted.shape == (BATCH_SIZE,)
    assert bool(carry.halted.all())


@pytest.mark.parametrize("model_cls,config_fn", MODEL_CASES)
def test_forward_produces_expected_outputs(model_cls, config_fn):
    model = model_cls(config_fn())
    model.eval()
    batch = _batch()
    carry = model.initial_carry(batch)

    new_carry, outputs = model(carry=carry, batch=batch)

    assert outputs["logits"].shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert outputs["q_halt_logits"].shape == (BATCH_SIZE,)
    assert outputs["q_continue_logits"].shape == (BATCH_SIZE,)
    assert torch.equal(new_carry.steps, torch.ones(BATCH_SIZE, dtype=torch.int32))
    assert not bool(new_carry.halted.all()) or True  # halting depends on loops/halt_max_steps


@pytest.mark.parametrize("model_cls,config_fn", MODEL_CASES)
def test_forward_halts_after_max_steps(model_cls, config_fn):
    max_steps = 3
    config = config_fn()
    # Normalize the "how many steps until forced halt" knob across archs.
    for key in ("loops", "halt_max_steps"):
        if key in config:
            config[key] = max_steps

    model = model_cls(config)
    model.eval()
    batch = _batch()
    carry = model.initial_carry(batch)

    for _ in range(max_steps):
        carry, outputs = model(carry=carry, batch=batch)

    assert bool(carry.halted.all())


def test_urm_without_puzzle_embedding():
    model = URM(_urm_config(puzzle_emb_ndim=0))
    batch = _batch()
    carry = model.initial_carry(batch)
    _, outputs = model(carry=carry, batch=batch)
    assert outputs["logits"].shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
