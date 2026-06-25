"""Train the legacy ML_sTEC-style FNN on IRI simulation rays.

The FNN is a direct ray-geometry-to-STEC model inspired by
``00_iri/mycode/ML_sTEC/FNN_sTEC_test.ipynb``.  It is trained only on simulated
slant-ray STEC labels.  For STEC-to-VTEC conversion, the trained model is
queried for both the original slant ray and a vertical special ray at the same
IPP, then the model-implied ratio is applied to the source STEC.
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
    from .evaluate import build_vtec_metrics
    from .geometry import geodetic_to_ecef
    from .metrics import regression_metrics, vtec_error_metrics
    from .models.mlp import build_fnn_stec
    from .train_mlp import (
        choose_device,
        make_loader,
        predict_scaled,
        save_checkpoint,
        set_random_seed,
        standardize_xy,
        train_loop,
    )
    from .train_ray_mlp_simulation import (
        add_mapping_aliases,
        choose_simulation_split,
        require_columns,
    )
    from .train_tree import (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_TABLE_DIR,
        read_table,
        save_table,
        write_json,
    )
except ImportError:
    from evaluate import build_vtec_metrics
    from geometry import geodetic_to_ecef
    from metrics import regression_metrics, vtec_error_metrics
    from models.mlp import build_fnn_stec
    from train_mlp import (
        choose_device,
        make_loader,
        predict_scaled,
        save_checkpoint,
        set_random_seed,
        standardize_xy,
        train_loop,
    )
    from train_ray_mlp_simulation import (
        add_mapping_aliases,
        choose_simulation_split,
        require_columns,
    )
    from train_tree import (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_TABLE_DIR,
        read_table,
        save_table,
        write_json,
    )


DEFAULT_INPUT = "datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.parquet"
ECEF6_COLUMNS = ["rec_x", "rec_y", "rec_z", "sat_x", "sat_y", "sat_z"]
BLH6_COLUMNS = ["rec_lon", "rec_lat", "rec_h", "sat_lon", "sat_lat", "sat_h"]
BL4_COLUMNS = ["rec_lon", "rec_lat", "sat_lon", "sat_lat"]


def fnn_feature_columns(feature_set: str) -> list[str]:
    """Return the FNN feature columns for the requested feature set."""

    if feature_set == "ecef6":
        return list(ECEF6_COLUMNS)
    if feature_set == "blh6":
        return list(BLH6_COLUMNS)
    if feature_set == "bl4":
        return list(BL4_COLUMNS)
    raise ValueError(f"Unsupported FNN feature set: {feature_set}")


def build_vertical_fnn_features(
    df: pd.DataFrame,
    *,
    feature_set: str,
    vertical_top_km: float,
) -> pd.DataFrame:
    """Build FNN features for the vertical VTEC special ray."""

    lon = df["ipp_lon"].to_numpy(float)
    lat = df["ipp_lat"].to_numpy(float)
    rec_x, rec_y, rec_z = geodetic_to_ecef(lon, lat, np.zeros(len(df)))
    sat_x, sat_y, sat_z = geodetic_to_ecef(
        lon,
        lat,
        np.full(len(df), vertical_top_km * 1000.0),
    )

    out = pd.DataFrame(index=df.index)
    if feature_set == "ecef6":
        out["rec_x"] = rec_x
        out["rec_y"] = rec_y
        out["rec_z"] = rec_z
        out["sat_x"] = sat_x
        out["sat_y"] = sat_y
        out["sat_z"] = sat_z
        return out
    if feature_set == "blh6":
        out["rec_lon"] = lon
        out["rec_lat"] = lat
        out["rec_h"] = 0.0
        out["sat_lon"] = lon
        out["sat_lat"] = lat
        out["sat_h"] = float(vertical_top_km)
        return out
    if feature_set == "bl4":
        out["rec_lon"] = lon
        out["rec_lat"] = lat
        out["sat_lon"] = lon
        out["sat_lat"] = lat
        return out
    raise ValueError(f"Unsupported FNN feature set: {feature_set}")


def predict_fnn(
    *,
    model,
    x: pd.DataFrame,
    x_imputer,
    x_scaler,
    y_scaler,
    device,
) -> np.ndarray:
    """Predict TECU values from an FNN feature table."""

    x_scaled = x_scaler.transform(x_imputer.transform(x.to_numpy(dtype=float)))
    pred_scaled = predict_scaled(model, x_scaled, device)
    return y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


def train_fnn_simulation(args: argparse.Namespace) -> dict[str, Any]:
    """Train FNN on slant-ray STEC and evaluate VTEC conversion."""

    set_random_seed(args.seed)
    input_path = Path(args.input)
    table_dir = Path(args.table_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = args.run_name or f"fnn_sim_{args.feature_set}_{args.split}"

    df = read_table(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    df = add_mapping_aliases(df, mapping_column=args.mapping_column)

    feature_cols = fnn_feature_columns(args.feature_set)
    required = feature_cols + [
        "timestamp",
        "station",
        "prn",
        "ipp_lon",
        "ipp_lat",
        args.source_stec_column,
        args.slant_target_column,
        args.vtec_target_column,
        args.baseline_vtec_column,
        args.mapping_column,
    ]
    require_columns(df, required)

    train_df, test_df, split_metadata = choose_simulation_split(df, args)
    y_train = pd.to_numeric(train_df[args.slant_target_column], errors="coerce")
    train_valid = y_train.notna()
    train_df = train_df.loc[train_valid].copy()
    y_train = y_train.loc[train_valid]
    if len(train_df) < 2:
        raise ValueError("FNN training requires at least two finite training rows.")

    x_train = train_df.loc[:, feature_cols]
    x_train_scaled, y_train_scaled, _, x_imputer, x_scaler, y_scaler = standardize_xy(
        x_train,
        y_train,
        x_train,
    )
    device = choose_device(args.device)
    loader = make_loader(x_train_scaled, y_train_scaled, args.batch_size, args.seed)
    model = build_fnn_stec(input_dim=len(feature_cols), dropout=args.dropout)
    history = train_loop(
        model,
        loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    train_pred = y_scaler.inverse_transform(
        predict_scaled(model, x_train_scaled, device).reshape(-1, 1)
    ).ravel()
    train_stec_metrics = regression_metrics(y_train, train_pred)

    test_slant_x = test_df.loc[:, feature_cols]
    test_vertical_x = build_vertical_fnn_features(
        test_df,
        feature_set=args.feature_set,
        vertical_top_km=args.vertical_top_km,
    ).loc[:, feature_cols]
    slant_pred = predict_fnn(
        model=model,
        x=test_slant_x,
        x_imputer=x_imputer,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
    )
    vertical_pred = predict_fnn(
        model=model,
        x=test_vertical_x,
        x_imputer=x_imputer,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
    )

    # Physical constraint: STEC must be non-negative.
    slant_pred = np.maximum(slant_pred, 0.0)
    vertical_pred = np.maximum(vertical_pred, 0.0)

    predictions = test_df.copy()
    predictions["stec_fnn"] = slant_pred
    predictions["vtec_fnn_direct"] = vertical_pred
    denominator = predictions["stec_fnn"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            predictions["vtec_fnn_direct"].to_numpy(float),
            denominator,
            out=np.full(len(predictions), np.nan, dtype=float),
            where=np.abs(denominator) > 1e-12,
        )
    predictions["fnn_conversion_ratio"] = ratio
    predictions["vtec_fnn"] = (
        predictions[args.source_stec_column].to_numpy(float) * ratio
    )
    predictions["vtec_fnn_error_iri"] = (
        predictions["vtec_fnn"] - predictions[args.vtec_target_column]
    )
    predictions["vtec_fnn_abs_error_iri"] = predictions["vtec_fnn_error_iri"].abs()
    predictions["vtec_fnn_direct_error_iri"] = (
        predictions["vtec_fnn_direct"] - predictions[args.vtec_target_column]
    )
    predictions["vtec_fnn_direct_abs_error_iri"] = predictions[
        "vtec_fnn_direct_error_iri"
    ].abs()

    prediction_path = save_table(
        predictions,
        table_dir / f"{run_name}_predictions.{args.output_format}",
    )
    vtec_metadata = None
    if not args.skip_vtec_evaluation:
        vtec_metadata = build_vtec_metrics(
            input_path=prediction_path,
            table_dir=table_dir,
            run_name=f"{run_name}_test",
            target_column=args.vtec_target_column,
            baseline_column=args.baseline_vtec_column,
            vtec_prediction_columns=["vtec_fnn", "vtec_fnn_direct"],
            stec_prediction_columns=[],
            mapping_column=args.mapping_column,
            elevation_bins=[
                float(item.strip())
                for item in args.elevation_bins.split(",")
                if item.strip()
            ],
            station_min_count=args.station_min_count,
            max_rows=None,
        )

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "simulation_fnn_stec_to_vtec",
        "model": "fnn_stec",
        "run_name": run_name,
        "input_file": str(input_path),
        "legacy_reference": "../00_iri/mycode/ML_sTEC/FNN_sTEC_test.ipynb",
        "feature_set": args.feature_set,
        "feature_columns": feature_cols,
        "n_features": int(len(feature_cols)),
        "source_stec_column": args.source_stec_column,
        "slant_target_column": args.slant_target_column,
        "vtec_target_column": args.vtec_target_column,
        "baseline_vtec_column": args.baseline_vtec_column,
        "stec_prediction_column": "stec_fnn",
        "vtec_prediction_column": "vtec_fnn",
        "vtec_direct_query_column": "vtec_fnn_direct",
        "conversion_ratio_column": "fnn_conversion_ratio",
        "conversion_interpretation": (
            "The FNN maps ray geometry directly to STEC. VTEC is estimated by "
            "querying the same model for the slant ray and the vertical special "
            "ray, then applying VTEC = source_STEC * vertical_query / slant_query."
        ),
        "vtec_reference_used_for_training": False,
        "vertical_top_km": float(args.vertical_top_km),
        "split": split_metadata,
        "row_counts": {
            "input": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_after_target_filter": int(len(train_df)),
        },
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "seed": int(args.seed),
            "device": str(device),
        },
        "history": history,
        "metrics": {
            "train_slant_stec": train_stec_metrics,
            "test_slant_stec": regression_metrics(
                predictions[args.slant_target_column],
                predictions["stec_fnn"],
            ),
            "test_vtec": vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions["vtec_fnn"],
            ),
            "test_vtec_direct_vertical_query": vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions["vtec_fnn_direct"],
            ),
            "test_slm_vtec": vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions[args.baseline_vtec_column],
            ),
        },
        "outputs": {"predictions": str(prediction_path)},
    }
    if vtec_metadata is not None:
        metadata["outputs"]["vtec_metrics"] = vtec_metadata["outputs"]

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
    summary_rows = [
        {"subset": "train", "metric_target": args.slant_target_column, **train_stec_metrics},
        {
            "subset": "test",
            "metric_target": args.slant_target_column,
            **regression_metrics(
                predictions[args.slant_target_column],
                predictions["stec_fnn"],
            ),
        },
        {
            "subset": "test",
            "metric_target": args.vtec_target_column,
            **vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions["vtec_fnn"],
            ),
        },
        {
            "subset": "test",
            "metric_target": f"{args.vtec_target_column}_direct_vertical_query",
            **vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions["vtec_fnn_direct"],
            ),
        },
        {
            "subset": "test",
            "metric_target": f"slm_{args.vtec_target_column}",
            **vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions[args.baseline_vtec_column],
            ),
        },
    ]
    metrics_csv_path = save_table(
        pd.DataFrame(summary_rows),
        table_dir / f"{run_name}_metrics.csv",
    )
    metadata["outputs"]["metrics_json"] = str(metrics_json_path)
    metadata["outputs"]["metrics_csv"] = str(metrics_csv_path)
    write_json(metadata, metrics_json_path)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--source-stec-column", default="stec_iri_observable")
    parser.add_argument("--slant-target-column", default="stec_iri_ray")
    parser.add_argument("--vtec-target-column", default="vtec_iri_ref")
    parser.add_argument("--baseline-vtec-column", default="vtec_slm_from_iri_stec")
    parser.add_argument("--mapping-column", default="slm_mapping_factor_iri")
    parser.add_argument("--vertical-top-km", type=float, default=2000.0)
    parser.add_argument(
        "--feature-set",
        choices=["ecef6", "blh6", "bl4"],
        default="ecef6",
        help="FNN inputs. The notebook default is ecef6.",
    )
    parser.add_argument(
        "--split",
        choices=["random", "time", "station", "prn"],
        default="random",
    )
    parser.add_argument("--test-keys-from")
    parser.add_argument("--test-key-columns", default="timestamp,station,prn")
    parser.add_argument("--test-timestamp")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-end")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--elevation-bins", default="10,20,30,40,50,60,70,80,90")
    parser.add_argument("--station-min-count", type=int, default=1)
    parser.add_argument("--skip-vtec-evaluation", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = train_fnn_simulation(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
