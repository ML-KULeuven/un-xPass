"""Tests for the unxpass.datasets module."""
from pathlib import Path

from unxpass.databases import SQLiteDatabase
from unxpass.datasets import CompletedPassesDataset, FailedPassesDataset, PassesDataset
from unxpass.features import endlocation, get_features, startlocation
from unxpass.labels import get_labels, scores


def test_passes_dataset(tmp_path: Path, db: SQLiteDatabase) -> None:
    """It should create a dataset with passes and compute the requested features and labels."""
    dataset = PassesDataset(
        tmp_path,
        xfns={startlocation: ["start_x_a0", "start_y_a0"], endlocation: ["end_x_a0", "end_y_a0"]},
        yfns=[scores],
    )
    dataset.create(db, filters=[{"game_id": 3795107}])
    assert len(dataset) > 0
    assert dataset.features is not None
    assert len(dataset.features) == len(dataset)
    assert "start_x_a0" in dataset.features.columns
    assert (tmp_path / "x_startlocation.parquet").is_file()
    assert "end_x_a0" in dataset.features.columns
    assert (tmp_path / "x_endlocation.parquet").is_file()
    assert dataset.labels is not None
    assert len(dataset.labels) == len(dataset)
    assert "scores" in dataset.labels.columns
    assert (tmp_path / "y_scores.parquet").is_file()


def test_passes_dataset_from_store(db: SQLiteDatabase, tmp_path: Path) -> None:
    """It should create a dataset with passes and load the requested features and labels."""
    # First, create a dataset with features and labels
    game_id = 3795107
    store_fp = tmp_path / "store"
    store_fp.mkdir(parents=True, exist_ok=True)
    df_startlocation = get_features(
        db,
        game_id=game_id,
        xfns=[startlocation],
    )
    df_startlocation.to_parquet(store_fp / "x_startlocation.parquet")
    df_endlocation = get_features(
        db,
        game_id=game_id,
        xfns=[endlocation],
    )
    df_endlocation.to_parquet(store_fp / "x_endlocation.parquet")
    df_scores = get_labels(
        db,
        game_id=game_id,
        yfns=[scores],
    )
    df_scores.to_parquet(store_fp / "y_scores.parquet")

    dataset = PassesDataset(
        tmp_path,
        xfns={startlocation: ["start_x_a0", "start_y_a0"], endlocation: ["end_x_a0", "end_y_a0"]},
        yfns=[scores],
    )

    # Then, load the dataset
    dataset = PassesDataset(
        store_fp,
        xfns={startlocation: ["start_x_a0", "start_y_a0"], endlocation: ["end_x_a0", "end_y_a0"]},
        yfns=[scores],
    )
    assert len(dataset) > 0
    assert dataset.features is not None
    assert len(dataset.features) == len(dataset)
    assert "start_x_a0" in dataset.features.columns
    assert "end_x_a0" in dataset.features.columns
    assert dataset.labels is not None
    assert len(dataset.labels) == len(dataset)
    assert "scores" in dataset.labels.columns


def test_completed_passes_dataset(tmp_path: Path, db: SQLiteDatabase) -> None:
    """It should only return completed passes."""
    dataset = CompletedPassesDataset(
        tmp_path,
        xfns={startlocation: ["start_x_a0", "start_y_a0"], endlocation: ["end_x_a0", "end_y_a0"]},
        yfns=[scores],
    )
    dataset.create(db, filters=[{"game_id": 3795107}])
    assert 0 in dataset.features.index.get_level_values("action_id")
    assert 1 not in dataset.features.index.get_level_values("action_id")
    assert 2 in dataset.features.index.get_level_values("action_id")


def test_failed_passes_dataset(tmp_path: Path, db: SQLiteDatabase) -> None:
    """It should only return failed passes."""
    dataset = FailedPassesDataset(
        tmp_path,
        xfns={startlocation: ["start_x_a0", "start_y_a0"], endlocation: ["end_x_a0", "end_y_a0"]},
        yfns=[scores],
    )
    dataset.create(db, filters=[{"game_id": 3795107}])
    assert 0 not in dataset.features.index.get_level_values("action_id")
    assert 2 not in dataset.features.index.get_level_values("action_id")
    assert 156 in dataset.features.index.get_level_values("action_id")
