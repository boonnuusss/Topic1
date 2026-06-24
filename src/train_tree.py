"""Train Ridge and RandomForest direct-prediction STEC baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .features import build_feature_matrix
    from .metrics import elevation_bin_metrics, regression_metrics
    from .models.tree_models import build_random_forest, build_ridge
    from .split import prn_holdout_split, random_split, station_holdout_split, time_split
except ImportError:
    from features import build_feature_matrix
    from metrics import elevation_bin_metrics, regression_metrics
    from models.tree_models import build_random_forest, build_ridge
    from split import prn_holdout_split, random_split, station_holdout_split, time_split


DEFAULT_INPUT = "datasets/processed/raystec_china_20240505.parquet"
DEFAULT_TABLE_DIR = "results/tables"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a parquet or CSV table."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a table as parquet or CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return path
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return path
    raise ValueError(f"Unsupported output format: {path}")


def choose_split(
    df: pd.DataFrame,
    split_name: str,
    *,
    test_size: float,
    seed: int,
    train_end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the requested train/test split and return split metadata."""

    if split_name == "random":
        train_df, test_df = random_split(df, test_size=test_size, seed=seed)
        metadata = {"type": "random", "test_size": test_size, "seed": seed}
    elif split_name == "time":
        if train_end is None:
            raise ValueError("--train-end is required for --split time.")
        train_df, test_df = time_split(df, train_end=train_end)
        metadata = {"type": "time", "train_end": train_end}
    elif split_name == "station":
        train_df, test_df = station_holdout_split(df, test_size=test_size, seed=seed)
        metadata = {"type": "station", "test_size": test_size, "seed": seed}
        metadata["test_stations"] = sorted(test_df["station"].dropna().unique().tolist())
    elif split_name == "prn":
        train_df, test_df = prn_holdout_split(df, test_size=test_size, seed=seed)
        metadata = {"type": "prn", "test_size": test_size, "seed": seed}
        metadata["test_prn"] = sorted(int(v) for v in test_df["prn"].dropna().unique())
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Split {split_name!r} produced train={len(train_df)} and "
            f"test={len(test_df)} rows."
        )
    metadata["n_train"] = int(len(train_df))
    metadata["n_test"] = int(len(test_df))
    return train_df, test_df, metadata


def prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], int, int]:
    """Create aligned direct-prediction feature matrices."""

    if target not in train_df.columns or target not in test_df.columns:
        raise KeyError(f"Target column missing from split data: {target}")

    x_train, y_train = build_feature_matrix(train_df, target=target)
    x_test, y_test = build_feature_matrix(test_df, target=target)
    feature_cols = [col for col in x_train.columns if col in x_test.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns found in the input table.")

    x_train = x_train.loc[:, feature_cols]
    x_test = x_test.loc[:, feature_cols]
    train_valid = y_train.notna()
    test_valid = y_test.notna()
    dropped_train = int((~train_valid).sum())
    dropped_test = int((~test_valid).sum())
    return (
        x_train.loc[train_valid],
        y_train.loc[train_valid],
        x_test.loc[test_valid],
        y_test.loc[test_valid],
        feature_cols,
        dropped_train,
        dropped_test,
    )


def build_model(args: argparse.Namespace):
    """Create the requested direct-prediction model."""

    if args.model == "ridge":
        return build_ridge(alpha=args.ridge_alpha, random_state=args.seed)
    if args.model == "random_forest":
        return build_random_forest(
            n_estimators=args.rf_n_estimators,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_depth=args.rf_max_depth,
            random_state=args.seed,
            n_jobs=args.rf_n_jobs,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def save_checkpoint(
    model,
    path: Path,
    metadata: dict[str, Any],
) -> Path:
    """Save a fitted sklearn model and metadata with joblib."""

    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)
    return path


def write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def train_direct_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Train and evaluate a Ridge or RandomForest direct STEC baseline."""

    input_path = Path(args.input)
    table_dir = Path(args.table_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = args.run_name or f"direct_{args.model}_{args.split}"
    pred_col = f"pred_{args.model}_direct"

    df = read_table(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    train_df, test_df, split_metadata = choose_split(
        df,
        args.split,
        test_size=args.test_size,
        seed=args.seed,
        train_end=args.train_end,
    )
    (
        x_train,
        y_train,
        x_test,
        y_test,
        feature_cols,
        dropped_train,
        dropped_test,
    ) = prepare_xy(train_df, test_df, target=args.target)

    model = build_model(args)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_metrics = regression_metrics(y_train, train_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    predictions = test_df.loc[y_test.index].copy()
    predictions[pred_col] = test_pred
    prediction_path = save_table(
        predictions,
        table_dir / f"{run_name}_predictions.{args.output_format}",
    )
    elevation_metrics = elevation_bin_metrics(predictions, pred_col=pred_col)
    elevation_metrics_path = save_table(
        elevation_metrics,
        table_dir / f"{run_name}_elevation_metrics.csv",
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "direct_prediction",
        "model": args.model,
        "run_name": run_name,
        "input_file": str(input_path),
        "target": args.target,
        "prediction_column": pred_col,
        "feature_columns": feature_cols,
        "n_features": int(len(feature_cols)),
        "split": split_metadata,
        "row_counts": {
            "input": int(len(df)),
            "train_after_target_filter": int(len(x_train)),
            "test_after_target_filter": int(len(x_test)),
            "dropped_train_missing_target": dropped_train,
            "dropped_test_missing_target": dropped_test,
        },
        "hyperparameters": {
            "ridge_alpha": args.ridge_alpha if args.model == "ridge" else None,
            "rf_n_estimators": (
                args.rf_n_estimators if args.model == "random_forest" else None
            ),
            "rf_min_samples_leaf": (
                args.rf_min_samples_leaf if args.model == "random_forest" else None
            ),
            "rf_max_depth": args.rf_max_depth if args.model == "random_forest" else None,
            "seed": args.seed,
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
        "outputs": {
            "predictions": str(prediction_path),
            "elevation_metrics": str(elevation_metrics_path),
        },
    }

    checkpoint_path = checkpoint_dir / f"{run_name}.joblib"
    metadata["outputs"]["checkpoint"] = str(
        save_checkpoint(model, checkpoint_path, metadata)
    )
    metrics_json_path = write_json(metadata, table_dir / f"{run_name}_metrics.json")
    metrics_csv = pd.DataFrame(
        [
            {"subset": "train", **train_metrics},
            {"subset": "test", **test_metrics},
        ]
    )
    metrics_csv_path = save_table(metrics_csv, table_dir / f"{run_name}_metrics.csv")
    metadata["outputs"]["metrics_json"] = str(metrics_json_path)
    metadata["outputs"]["metrics_csv"] = str(metrics_csv_path)
    write_json(metadata, metrics_json_path)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input parquet/CSV table.")
    parser.add_argument("--target", default="stec_obs")
    parser.add_argument("--model", choices=["ridge", "random_forest"], required=True)
    parser.add_argument(
        "--split",
        choices=["random", "time", "station", "prn"],
        default="random",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-end", help="Required for --split time.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rf-max-depth", type=int)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--run-name")
    parser.add_argument("--max-rows", type=int, help="Optional smoke-test row cap.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = train_direct_baseline(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
