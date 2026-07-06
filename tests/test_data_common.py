import numpy as np
import pydantic
import pytest

from data.common import dihedral_transform, inverse_dihedral_transform, PuzzleDatasetMetadata


@pytest.mark.parametrize("tid", range(8))
def test_dihedral_transform_round_trip(tid):
    arr = np.arange(9).reshape(3, 3)
    transformed = dihedral_transform(arr, tid)
    restored = inverse_dihedral_transform(transformed, tid)
    assert np.array_equal(restored, arr)


def test_dihedral_transform_identity():
    arr = np.arange(6).reshape(2, 3)
    assert np.array_equal(dihedral_transform(arr, 0), arr)


def test_dihedral_transform_unknown_tid_returns_input():
    arr = np.arange(4).reshape(2, 2)
    assert np.array_equal(dihedral_transform(arr, 99), arr)


def test_puzzle_dataset_metadata_valid_construction():
    meta = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=-1,
        blank_identifier_id=0,
        vocab_size=16,
        seq_len=8,
        num_puzzle_identifiers=4,
        total_groups=3,
        mean_puzzle_examples=2.0,
        sets=["all"],
    )
    assert meta.sets == ["all"]


def test_puzzle_dataset_metadata_ignore_label_id_optional():
    meta = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=None,
        blank_identifier_id=0,
        vocab_size=16,
        seq_len=8,
        num_puzzle_identifiers=4,
        total_groups=3,
        mean_puzzle_examples=2.0,
        sets=["all"],
    )
    assert meta.ignore_label_id is None


def test_puzzle_dataset_metadata_missing_required_field_raises():
    with pytest.raises(pydantic.ValidationError):
        PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=-1,
            blank_identifier_id=0,
            vocab_size=16,
            seq_len=8,
            num_puzzle_identifiers=4,
            total_groups=3,
            sets=["all"],
        )
