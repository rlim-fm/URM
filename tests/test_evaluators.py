"""Tests for evaluators/arc.py.

Importing evaluators.arc transitively requires numba (used directly via
@njit) and argdantic (pulled in through data.build_arc_dataset). Skip the
whole module cleanly if either is missing, rather than failing collection.
"""
import json

import numpy as np
import pytest
import torch

arc_module = pytest.importorskip("evaluators.arc", reason="evaluators.arc requires numba + argdantic")

_crop = arc_module._crop
ARC = arc_module.ARC


def _make_grid_with_content(rows, cols, fill=3):
    """A 30x30 grid (flattened) where the top-left rows x cols block holds
    values in [2, 11] (valid content) and everything else is >= 12 (EOS-like,
    outside the [2, 11] "content" range `_crop` scans for).
    """
    grid = np.full((30, 30), 12, dtype=np.uint8)
    grid[:rows, :cols] = fill
    return grid.reshape(-1)


def test_crop_finds_content_rectangle():
    grid = _make_grid_with_content(5, 7, fill=3)
    cropped = _crop(grid)
    assert cropped.shape == (5, 7)
    assert np.all(cropped == 3 - 2)


def test_crop_full_grid_content():
    grid = np.full((30, 30), 5, dtype=np.uint8).reshape(-1)
    cropped = _crop(grid)
    assert cropped.shape == (30, 30)


def test_crop_empty_content_returns_empty():
    grid = np.full((30, 30), 12, dtype=np.uint8).reshape(-1)
    cropped = _crop(grid)
    assert cropped.shape == (0, 0)


class _FakeEvalMetadata:
    def __init__(self, blank_identifier_id=0):
        self.blank_identifier_id = blank_identifier_id


def _write_arc_fixture(tmp_path, identifiers, test_puzzles):
    data_path = tmp_path / "arc_eval"
    data_path.mkdir()
    with open(data_path / "identifiers.json", "w") as f:
        json.dump(identifiers, f)
    with open(data_path / "test_puzzles.json", "w") as f:
        json.dump(test_puzzles, f)
    return str(data_path)


def test_arc_update_batch_populates_local_predictions(tmp_path):
    identifiers = [None, "puzzle_a"]  # index 0 is blank/reserved
    test_puzzles = {
        "puzzle_a": {
            "test": [{"input": [[1, 1], [1, 1]], "output": [[1, 1], [1, 1]]}],
        }
    }
    data_path = _write_arc_fixture(tmp_path, identifiers, test_puzzles)

    evaluator = ARC(data_path=data_path, eval_metadata=_FakeEvalMetadata(blank_identifier_id=0))
    evaluator.begin_eval()

    seq_len = 900  # 30x30 flattened
    content = np.full(seq_len, 12, dtype=np.int64)
    content[:4] = 3  # a 2x2 block of "color 1" content (value 3 = 1 + 2 offset)

    batch = {
        "puzzle_identifiers": torch.tensor([1], dtype=torch.int64),
        "inputs": torch.tensor(content, dtype=torch.int64).unsqueeze(0),
    }
    preds = {
        "q_halt_logits": torch.tensor([2.0]),
        "preds": torch.tensor(content, dtype=torch.int64).unsqueeze(0),
    }

    evaluator.update_batch(batch, preds)

    assert len(evaluator._local_preds) == 1
    assert "puzzle_a" in evaluator._local_preds


def test_arc_update_batch_filters_padding(tmp_path):
    identifiers = [None, "puzzle_a"]
    test_puzzles = {"puzzle_a": {"test": [{"input": [[1]], "output": [[1]]}]}}
    data_path = _write_arc_fixture(tmp_path, identifiers, test_puzzles)

    evaluator = ARC(data_path=data_path, eval_metadata=_FakeEvalMetadata(blank_identifier_id=0))
    evaluator.begin_eval()

    seq_len = 900
    content = np.full(seq_len, 12, dtype=np.int64)
    content[:1] = 3

    # Two rows: one real (id=1), one padding (id=0, the blank_identifier_id).
    batch = {
        "puzzle_identifiers": torch.tensor([0, 1], dtype=torch.int64),
        "inputs": torch.tensor(np.stack([content, content]), dtype=torch.int64),
    }
    preds = {
        "q_halt_logits": torch.tensor([1.0, 1.0]),
        "preds": torch.tensor(np.stack([content, content]), dtype=torch.int64),
    }

    evaluator.update_batch(batch, preds)
    # Only the non-padding row should have contributed a prediction.
    assert len(evaluator._local_preds) == 1
