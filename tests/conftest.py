"""Shared fixtures for the URM test suite.

All fixtures run on CPU with tiny configurations so the suite stays fast and
deterministic. Tests that need optional heavyweight deps (flash_attn, numba,
hydra) are gated with the ``requires_*`` skip markers below so the suite
collects and passes in a minimal environment, and automatically exercises
those code paths once the full ``requirements.txt`` is installed.
"""
import importlib.util
import json
import os

import numpy as np
import pytest
import torch


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


requires_flash_attn = pytest.mark.skipif(
    not _has_module("flash_attn") and not _has_module("flash_attn_interface"),
    reason="flash_attn not installed",
)
requires_numba = pytest.mark.skipif(
    not _has_module("numba"), reason="numba not installed",
)
requires_hydra = pytest.mark.skipif(
    not (_has_module("hydra") and _has_module("omegaconf")),
    reason="hydra/omegaconf not installed",
)


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    np.random.seed(0)
    yield


def _tiny_config(**overrides):
    base = dict(
        batch_size=2,
        seq_len=8,
        puzzle_emb_ndim=32,
        num_puzzle_identifiers=4,
        vocab_size=16,
        num_layers=1,
        hidden_size=32,
        expansion=2.0,
        num_heads=2,
        pos_encodings="rope",
        loops=2,
        L_cycles=1,
        H_cycles=2,
        forward_dtype="float32",
    )
    base.update(overrides)
    return base


@pytest.fixture
def tiny_urm_config():
    return _tiny_config()


@pytest.fixture
def tiny_tarm_config():
    return _tiny_config()


@pytest.fixture
def tiny_trm_config():
    return _tiny_config()


@pytest.fixture
def tiny_hrm_config():
    return _tiny_config()


def _write_puzzle_dataset_split(root, split, num_groups=3, puzzles_per_group=2,
                                 examples_per_puzzle=2, seq_len=8, vocab_size=16,
                                 num_puzzle_identifiers=4, pad_id=0, ignore_label_id=-1,
                                 blank_identifier_id=0, set_name="all"):
    split_dir = os.path.join(root, split)
    os.makedirs(split_dir, exist_ok=True)

    num_puzzles = num_groups * puzzles_per_group
    num_examples = num_puzzles * examples_per_puzzle

    rng = np.random.default_rng(0)
    inputs = rng.integers(2, vocab_size, size=(num_examples, seq_len)).astype(np.int32)
    labels = rng.integers(2, vocab_size, size=(num_examples, seq_len)).astype(np.int32)
    puzzle_identifiers = (np.arange(num_puzzles) % (num_puzzle_identifiers - 1) + 1).astype(np.int32)

    puzzle_indices = np.arange(0, num_examples + 1, examples_per_puzzle).astype(np.int64)
    group_indices = np.arange(0, num_puzzles + 1, puzzles_per_group).astype(np.int64)

    np.save(os.path.join(split_dir, f"{set_name}__inputs.npy"), inputs)
    np.save(os.path.join(split_dir, f"{set_name}__labels.npy"), labels)
    np.save(os.path.join(split_dir, f"{set_name}__puzzle_identifiers.npy"), puzzle_identifiers)
    np.save(os.path.join(split_dir, f"{set_name}__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(split_dir, f"{set_name}__group_indices.npy"), group_indices)

    metadata = dict(
        pad_id=pad_id,
        ignore_label_id=ignore_label_id,
        blank_identifier_id=blank_identifier_id,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers,
        total_groups=num_groups,
        mean_puzzle_examples=float(examples_per_puzzle),
        sets=[set_name],
    )
    with open(os.path.join(split_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)

    return metadata


@pytest.fixture
def tmp_puzzle_dataset(tmp_path):
    """Writes a minimal on-disk PuzzleDataset (train + test splits) and
    returns (dataset_path, metadata_dict, set_name).
    """
    root = str(tmp_path / "puzzle_dataset")
    metadata = _write_puzzle_dataset_split(root, "train")
    _write_puzzle_dataset_split(root, "test")
    return root, metadata, "all"
