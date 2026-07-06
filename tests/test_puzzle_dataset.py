import numpy as np
import pytest
import torch

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, _sample_batch
from models.losses import IGNORE_LABEL_ID


def _make_config(dataset_path, global_batch_size=4, test_set_mode=False, rank=0, num_replicas=1, seed=0):
    return PuzzleDatasetConfig(
        seed=seed,
        dataset_path=dataset_path,
        global_batch_size=global_batch_size,
        test_set_mode=test_set_mode,
        epochs_per_iter=1,
        rank=rank,
        num_replicas=num_replicas,
    )


def test_missing_split_raises_file_not_found(tmp_path):
    config = _make_config(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        PuzzleDataset(config, split="train")


def test_global_batch_size_must_divide_num_replicas(tmp_puzzle_dataset):
    root, _, _ = tmp_puzzle_dataset
    config = _make_config(root, global_batch_size=5, num_replicas=2)
    with pytest.raises(AssertionError):
        PuzzleDataset(config, split="train")


def test_sample_batch_packs_to_global_batch_size():
    rng = np.random.default_rng(0)
    # 2 groups, 2 puzzles each, 2 examples per puzzle -> 4 puzzles, 8 examples
    group_indices = np.array([0, 2, 4], dtype=np.int64)
    puzzle_indices = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    group_order = np.concatenate([rng.permutation(2) for _ in range(2)])

    start_index, batch_indices, batch_puzzle_indices = _sample_batch(
        rng, group_order=group_order, puzzle_indices=puzzle_indices,
        group_indices=group_indices, start_index=0, global_batch_size=4,
    )
    assert batch_indices.shape[0] == 4
    assert batch_puzzle_indices.shape[0] == 4
    assert start_index <= group_order.size


def test_train_iteration_yields_expected_batch_shapes(tmp_puzzle_dataset):
    root, metadata, set_name = tmp_puzzle_dataset
    config = _make_config(root, global_batch_size=4, test_set_mode=False)
    dataset = PuzzleDataset(config, split="train")

    batches = list(dataset)
    assert len(batches) > 0
    for name, batch, effective_size in batches:
        assert name == set_name
        assert batch["inputs"].shape == (4, metadata["seq_len"])
        assert batch["labels"].shape == (4, metadata["seq_len"])
        assert batch["puzzle_identifiers"].shape == (4,)
        assert isinstance(batch["inputs"], torch.Tensor)
        assert effective_size <= 4


def test_train_iteration_drops_last_partial_batch(tmp_puzzle_dataset):
    root, metadata, set_name = tmp_puzzle_dataset
    # global_batch_size larger than total examples forces every batch to be
    # "partial" and therefore dropped -> no batches yielded.
    config = _make_config(root, global_batch_size=1000, test_set_mode=False)
    dataset = PuzzleDataset(config, split="train")
    batches = list(dataset)
    assert batches == []


def test_test_mode_pads_last_batch_and_remaps_ignore_label(tmp_puzzle_dataset):
    root, metadata, set_name = tmp_puzzle_dataset
    config = _make_config(root, global_batch_size=1000, test_set_mode=True)
    dataset = PuzzleDataset(config, split="test")

    batches = list(dataset)
    assert len(batches) == 1
    name, batch, total_examples = batches[0]
    assert name == set_name
    assert batch["inputs"].shape[0] == config.global_batch_size  # padded to local_batch_size
    assert total_examples < config.global_batch_size

    # Padded tail should use pad_id / blank_identifier_id.
    assert (batch["inputs"][total_examples:] == metadata["pad_id"]).all()
    assert (batch["puzzle_identifiers"][total_examples:] == metadata["blank_identifier_id"]).all()


def test_test_mode_puzzle_identifiers_align_with_puzzle_indices(tmp_puzzle_dataset):
    root, metadata, set_name = tmp_puzzle_dataset
    config = _make_config(root, global_batch_size=1000, test_set_mode=True)
    dataset = PuzzleDataset(config, split="test")
    _, batch, total_examples = next(iter(dataset))

    raw = np.load(f"{root}/test/{set_name}__puzzle_identifiers.npy")
    puzzle_indices = np.load(f"{root}/test/{set_name}__puzzle_indices.npy")

    expected = []
    puzzle_index = 0
    for i in range(total_examples):
        while puzzle_index + 1 < len(puzzle_indices) and i >= puzzle_indices[puzzle_index + 1]:
            puzzle_index += 1
        expected.append(raw[puzzle_index])

    assert list(batch["puzzle_identifiers"][:total_examples].numpy()) == expected
