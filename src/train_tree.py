"""Legacy/off-mainline Ridge and RandomForest STEC baselines.

These entry points predict observed STEC directly or learn residuals relative
to legacy ``stec_slm``. The current Topic1 mainline evaluates models by their
effect on STEC-to-VTEC conversion error.
"""

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


def ensure_residual_target(
    df: pd.DataFrame,
    *,
    observed_column: str,
    baseline_column: str,
    residual_column: str,
) -> pd.DataFrame:
    """Ensure `residual_column = observed_column - baseline_column` exists."""

    for column in [observed_column, baseline_column]:
        if column not in df.columns:
            raise KeyError(f"Residual learning requires column: {column}")
    out = df.copy()
    out[residual_column] = (
        out[observed_column].astype(float) - out[baseline_column].astype(float)
    )
    return out


def prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str,
    include_physical_baseline: bool = False,
    physical_baseline_column: str = "stec_slm",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], int, int]:
    """Create aligned feature matrices for a train/test split."""

    if target not in train_df.columns or target not in test_df.columns:
        raise KeyError(f"Target column missing from split data: {target}")

    x_train, y_train = build_feature_matrix(
        train_df,
        target=target,
        include_physical_baseline=include_physical_baseline,
        physical_baseline_column=physical_baseline_column,
    )
    x_test, y_test = build_feature_matrix(
        test_df,
        target=target,
        include_physical_baseline=include_physical_baseline,
        physical_baseline_column=physical_baseline_column,
    )
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
    """Create the requested tabular model."""

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


def _metrics_rows(
    mode: str,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    residual_train_metrics: dict[str, float] | None = None,
    residual_test_metrics: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = [
        {"subset": "train", "metric_target": "stec_obs", **train_metrics},
        {"subset": "test", "metric_target": "stec_obs", **test_metrics},
    ]
    if mode == "residual" and residual_train_metrics and residual_test_metrics:
        rows.extend(
            [
                {
                    "subset": "train",
                    "metric_target": "stec_residual_slm",
                    **residual_train_metrics,
                },
                {
                    "subset": "test",
                    "metric_target": "stec_residual_slm",
                    **residual_test_metrics,
                },
            ]
        )
    return pd.DataFrame(rows)


def train_tabular_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Train and evaluate a direct or SLM-residual Ridge/RF baseline."""

    input_path = Path(args.input)
    table_dir = Path(args.table_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = args.run_name or f"{args.mode}_{args.model}_{args.split}"
    residual_pred_col = f"pred_{args.model}_residual"
    final_pred_col = f"pred_{args.model}_residual_stec"
    direct_pred_col = f"pred_{args.model}_direct"

    df = read_table(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    if args.mode == "residual":
        df = ensure_residual_target(
            df,
            observed_column=args.target,
            baseline_column=args.baseline_column,
            residual_column=args.residual_target,
        )
        model_target = args.residual_target
        pred_col = final_pred_col
    else:
        model_target = args.target
        pred_col = direct_pred_col

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
    ) = prepare_xy(
        train_df,
        test_df,
        target=model_target,
        include_physical_baseline=args.include_baseline_feature,
        physical_baseline_column=args.baseline_column,
    )

    model = build_model(args)
    model.fit(x_train, y_train)
    train_model_pred = model.predict(x_train)
    test_model_pred = model.predict(x_test)

    if args.mode == "residual":
        train_eval_df = train_df.loc[y_train.index].copy()
        test_eval_df = test_df.loc[y_test.index].copy()
        train_final_pred = (
            train_eval_df[args.baseline_column].to_numpy(float) + train_model_pred
        )
        test_final_pred = (
            test_eval_df[args.baseline_column].to_numpy(float) + test_model_pred
        )
        train_metrics = regression_metrics(train_eval_df[args.target], train_final_pred)
        test_metrics = regression_metrics(test_eval_df[args.target], test_final_pred)
        residual_train_metrics = regression_metrics(y_train, train_model_pred)
        residual_test_metrics = regression_metrics(y_test, test_model_pred)

        predictions = test_eval_df.copy()
        predictions[residual_pred_col] = test_model_pred
        predictions[final_pred_col] = test_final_pred
        model_prediction_column = residual_pred_col
        final_prediction_column = final_pred_col
    else:
        train_metrics = regression_metrics(y_train, train_model_pred)
        test_metrics = regression_metrics(y_test, test_model_pred)
        residual_train_metrics = None
        residual_test_metrics = None

        predictions = test_df.loc[y_test.index].copy()
        predictions[direct_pred_col] = test_model_pred
        model_prediction_column = direct_pred_col
        final_prediction_column = direct_pred_col

    prediction_path = save_table(
        predictions,
        table_dir / f"{run_name}_predictions.{args.output_format}",
    )
    elevation_metrics = elevation_bin_metrics(
        predictions,
        pred_col=final_prediction_column,
        target_col=args.target,
    )
    elevation_metrics_path = save_table(
        elevation_metrics,
        table_dir / f"{run_name}_elevation_metrics.csv",
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "direct_prediction" if args.mode == "direct" else "slm_residual",
        "model": args.model,
        "run_name": run_name,
        "input_file": str(input_path),
        "observed_target": args.target,
        "model_target": model_target,
        "baseline_column": args.baseline_column if args.mode == "residual" else None,
        "model_prediction_column": model_prediction_column,
        "final_prediction_column": final_prediction_column,
        "include_baseline_feature": bool(args.include_baseline_feature),
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
    if args.mode == "residual":
        metadata["residual_metrics"] = {
            "train": residual_train_metrics,
            "test": residual_test_metrics,
        }

    checkpoint_path = checkpoint_dir / f"{run_name}.joblib"
    metadata["outputs"]["checkpoint"] = str(
        save_checkpoint(model, checkpoint_path, metadata)
    )
    metrics_json_path = write_json(metadata, table_dir / f"{run_name}_metrics.json")
    metrics_csv = _metrics_rows(
        args.mode,
        train_metrics,
        test_metrics,
        residual_train_metrics,
        residual_test_metrics,
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
    parser.add_argument("--mode", choices=["direct", "residual"], default="direct")
    parser.add_argument("--baseline-column", default="stec_slm")
    parser.add_argument("--residual-target", default="stec_residual_slm")
    parser.add_argument(
        "--include-baseline-feature",
        action="store_true",
        help="Include the physical baseline column as a model feature.",
    )
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
    metadata = train_tabular_baseline(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
