"""Tests for the publication train/validation/locked-test protocol."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pandas as pd
import torch

from src.train_set_ratio_mlp import (
    apply_elevation_filters,
    choose_set_ratio_split,
    clone_model_state,
)


class SetRatioProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-05-05T05:00:00"] * 6
                ),
                "station": [f"S{i}" for i in range(6)],
                "prn": [1, 2, 3, 4, 5, 6],
                "elevation": [35.0, 40.0, 45.0, 50.0, 55.0, 60.0],
            }
        )

    @staticmethod
    def args(validation_path: Path, test_path: Path | None):
        return SimpleNamespace(
            split="random",
            validation_keys_from=str(validation_path),
            validation_key_columns="timestamp,station,prn",
            test_keys_from=None if test_path is None else str(test_path),
            test_key_columns="timestamp,station,prn",
            min_train_elevation_deg=30.0,
            min_eval_elevation_deg=30.0,
        )

    def write_keys(self, path: Path, rows: list[int]) -> None:
        self.df.loc[
            rows, ["timestamp", "station", "prn"]
        ].to_csv(path, index=False)

    def test_three_way_key_split_is_disjoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation_path = root / "validation.csv"
            test_path = root / "test.csv"
            self.write_keys(validation_path, [1, 2])
            self.write_keys(test_path, [4])

            train, validation, test, metadata = choose_set_ratio_split(
                self.df, self.args(validation_path, test_path)
            )

        self.assertEqual(set(train.index), {0, 3, 5})
        self.assertEqual(set(validation.index), {1, 2})
        self.assertEqual(set(test.index), {4})
        self.assertEqual(
            metadata["protocol"], "train_validation_locked_test"
        )
        self.assertFalse(metadata["validation_test_overlap"])

    def test_validation_only_protocol_leaves_test_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            validation_path = Path(tmp) / "validation.csv"
            self.write_keys(validation_path, [1, 2])
            train, validation, test, metadata = choose_set_ratio_split(
                self.df, self.args(validation_path, None)
            )

        self.assertEqual(len(train), 4)
        self.assertEqual(len(validation), 2)
        self.assertTrue(test.empty)
        self.assertEqual(metadata["protocol"], "train_validation")

    def test_overlapping_validation_and_test_keys_are_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation_path = root / "validation.csv"
            test_path = root / "test.csv"
            self.write_keys(validation_path, [1, 2])
            self.write_keys(test_path, [2, 3])

            with self.assertRaisesRegex(ValueError, "overlap"):
                choose_set_ratio_split(
                    self.df, self.args(validation_path, test_path)
                )

    def test_elevation_filter_tracks_all_three_roles(self) -> None:
        args = SimpleNamespace(
            min_train_elevation_deg=40.0,
            min_eval_elevation_deg=50.0,
        )
        train, validation, test, metadata = apply_elevation_filters(
            self.df.iloc[:4],
            self.df.iloc[3:5],
            self.df.iloc[4:],
            args,
        )
        self.assertEqual(len(train), 3)
        self.assertEqual(len(validation), 2)
        self.assertEqual(len(test), 2)
        self.assertEqual(metadata["validation_rows_after"], 2)

    def test_cloned_best_state_is_independent(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        best_state = clone_model_state(model)
        with torch.no_grad():
            model.weight.fill_(2.0)
        model.load_state_dict(best_state)
        self.assertEqual(float(model.weight.item()), 1.0)


if __name__ == "__main__":
    unittest.main()
