"""Legacy/off-mainline RayMLP residual learning on top of SLM STEC.

The current Topic1 mainline should adapt RayMLP to the IRI-backed
STEC-to-VTEC conversion experiment instead of treating this residual STEC
prediction entry point as the final scientific route.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .metrics import elevation_bin_metrics, regression_metrics
    from .models.mlp import build_ray_mlp
    from .train_tree import (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_TABLE_DIR,
        choose_split,
        ensure_residual_target,
        prepare_xy,
        read_table,
        save_table,
        write_json,
    )
except ImportError:
    from metrics import elevation_bin_metrics, regression_metrics
    from models.mlp import build_ray_mlp
    from train_tree import (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_TABLE_DIR,
        choose_split,
        ensure_residual_target,
        prepare_xy,
        read_table,
        save_table,
        write_json,
    )


DEFAULT_INPUT = "datasets/processed/raystec_china_20240505_slm.parquet"


def set_random_seed(seed: int) -> None:
    """Set NumPy and PyTorch seeds."""

    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str):
    """Return the requested torch device."""

    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def standardize_xy(x_train, y_train, x_test):
    """Fit imputers/scalers on training data and transform X/y."""

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    x_imputer = SimpleImputer(strategy="median")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_np = x_train.to_numpy(dtype=float)
    x_test_np = x_test.to_numpy(dtype=float)
    y_train_np = y_train.to_numpy(dtype=float).reshape(-1, 1)

    x_train_scaled = x_scaler.fit_transform(x_imputer.fit_transform(x_train_np))
    x_test_scaled = x_scaler.transform(x_imputer.transform(x_test_np))
    y_train_scaled = y_scaler.fit_transform(y_train_np).ravel()
    return x_train_scaled, y_train_scaled, x_test_scaled, x_imputer, x_scaler, y_scaler


def make_loader(x_train, y_train, batch_size: int, seed: int):
    """Create a deterministic PyTorch DataLoader for tabular arrays."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if len(x_train) < 2:
        raise ValueError("RayMLP training requires at least two training samples.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
    )
    drop_last = len(dataset) > batch_size and len(dataset) % batch_size == 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
    )


def train_loop(
    model,
    loader,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device,
) -> list[dict[str, float | int]]:
    """Train a RayMLP and return per-epoch loss history."""

    import torch
    import torch.nn as nn

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_size = int(len(xb))
            total_loss += float(loss.detach().cpu()) * batch_size
            total_count += batch_size
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_count, 1),
            }
        )
    return history


def predict_scaled(model, x, device) -> np.ndarray:
    """Predict scaled residuals."""

    import torch

    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32).to(device)
        pred = model(tensor).detach().cpu().numpy().reshape(-1)
    return pred


def save_checkpoint(
    path: Path,
    *,
    model,
    x_imputer,
    x_scaler,
    y_scaler,
    metadata: dict[str, Any],
) -> Path:
    """Save a RayMLP checkpoint and preprocessing objects."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_imputer": x_imputer,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "metadata": metadata,
        },
        path,
    )
    return path


def train_ray_mlp_residual(args: argparse.Namespace) -> dict[str, Any]:
    """Train and evaluate a RayMLP residual model."""

    set_random_seed(args.seed)
    input_path = Path(args.input)
    table_dir = Path(args.table_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = args.run_name or f"residual_ray_mlp_{args.split}"
    residual_pred_col = "pred_ray_mlp_residual"
    final_pred_col = "pred_ray_mlp_residual_stec"

    df = read_table(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    df = ensure_residual_target(
        df,
        observed_column=args.target,
        baseline_column=args.baseline_column,
        residual_column=args.residual_target,
    )
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
        target=args.residual_target,
        include_physical_baseline=args.include_baseline_feature,
        physical_baseline_column=args.baseline_column,
    )

    x_train_scaled, y_train_scaled, x_test_scaled, x_imputer, x_scaler, y_scaler = (
        standardize_xy(x_train, y_train, x_test)
    )
    device = choose_device(args.device)
    loader = make_loader(x_train_scaled, y_train_scaled, args.batch_size, args.seed)
    model = build_ray_mlp(input_dim=len(feature_cols), dropout=args.dropout)
    history = train_loop(
        model,
        loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    train_residual_scaled = predict_scaled(model, x_train_scaled, device)
    test_residual_scaled = predict_scaled(model, x_test_scaled, device)
    train_residual_pred = y_scaler.inverse_transform(
        train_residual_scaled.reshape(-1, 1)
    ).ravel()
    test_residual_pred = y_scaler.inverse_transform(
        test_residual_scaled.reshape(-1, 1)
    ).ravel()

    train_eval_df = train_df.loc[y_train.index].copy()
    test_eval_df = test_df.loc[y_test.index].copy()
    train_final_pred = (
        train_eval_df[args.baseline_column].to_numpy(float) + train_residual_pred
    )
    test_final_pred = (
        test_eval_df[args.baseline_column].to_numpy(float) + test_residual_pred
    )

    train_metrics = regression_metrics(train_eval_df[args.target], train_final_pred)
    test_metrics = regression_metrics(test_eval_df[args.target], test_final_pred)
    residual_train_metrics = regression_metrics(y_train, train_residual_pred)
    residual_test_metrics = regression_metrics(y_test, test_residual_pred)

    predictions = test_eval_df.copy()
    predictions[residual_pred_col] = test_residual_pred
    predictions[final_pred_col] = test_final_pred
    prediction_path = save_table(
        predictions,
        table_dir / f"{run_name}_predictions.{args.output_format}",
    )
    elevation_metrics = elevation_bin_metrics(
        predictions,
        pred_col=final_pred_col,
        target_col=args.target,
    )
    elevation_metrics_path = save_table(
        elevation_metrics,
        table_dir / f"{run_name}_elevation_metrics.csv",
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "slm_residual",
        "model": "ray_mlp",
        "run_name": run_name,
        "input_file": str(input_path),
        "observed_target": args.target,
        "model_target": args.residual_target,
        "baseline_column": args.baseline_column,
        "model_prediction_column": residual_pred_col,
        "final_prediction_column": final_pred_col,
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
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": str(device),
        },
        "history": history,
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
        "residual_metrics": {
            "train": residual_train_metrics,
            "test": residual_test_metrics,
        },
        "outputs": {
            "predictions": str(prediction_path),
            "elevation_metrics": str(elevation_metrics_path),
        },
    }

    checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    metadata["outputs"]["checkpoint"] = str(
        save_checkpoint(
            checkpoint_path,
            model=model,
            x_imputer=x_imputer,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            metadata=metadata,
        )
    )
    metrics_json_path = write_json(metadata, table_dir / f"{run_name}_metrics.json")
    metrics_csv = pd.DataFrame(
        [
            {"subset": "train", "metric_target": "stec_obs", **train_metrics},
            {"subset": "test", "metric_target": "stec_obs", **test_metrics},
            {
                "subset": "train",
                "metric_target": args.residual_target,
                **residual_train_metrics,
            },
            {
                "subset": "test",
                "metric_target": args.residual_target,
                **residual_test_metrics,
            },
        ]
    )
    metrics_csv_path = save_table(metrics_csv, table_dir / f"{run_name}_metrics.csv")
    metadata["outputs"]["metrics_json"] = str(metrics_json_path)
    metadata["outputs"]["metrics_csv"] = str(metrics_csv_path)
    write_json(metadata, metrics_json_path)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input SLM parquet/CSV.")
    parser.add_argument("--target", default="stec_obs")
    parser.add_argument("--baseline-column", default="stec_slm")
    parser.add_argument("--residual-target", default="stec_residual_slm")
    parser.add_argument(
        "--include-baseline-feature",
        action="store_true",
        help="Include the physical baseline column as a model feature.",
    )
    parser.add_argument(
        "--split",
        choices=["random", "time", "station", "prn"],
        default="random",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-end", help="Required for --split time.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--run-name")
    parser.add_argument("--max-rows", type=int, help="Optional smoke-test row cap.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = train_ray_mlp_residual(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
