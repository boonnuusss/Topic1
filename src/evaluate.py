"""Evaluate SLM and ML-assisted STEC-to-VTEC conversion in VTEC space."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .metrics import vtec_error_metrics
except ImportError:
    from metrics import vtec_error_metrics


DEFAULT_INPUT = "datasets/processed/iri_simulation_rayvtec_20240505.parquet"
DEFAULT_TABLE_DIR = "results/tables"


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
    """Save a table as CSV or parquet."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return path
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return path
    raise ValueError(f"Unsupported output format: {path}")


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write JSON metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def parse_csv_list(value: str | None) -> list[str]:
    """Parse a comma-separated CLI list."""

    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    """Parse comma-separated bin edges."""

    edges = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(edges) < 2:
        raise ValueError("At least two bin edges are required.")
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError("Bin edges must be strictly increasing.")
    return edges


def prediction_label(column: str, *, baseline_column: str, source: str) -> str:
    """Return a stable model label for a prediction column."""

    if column == baseline_column:
        return "slm"
    if source == "stec_prediction":
        return f"ml_assisted_from_{column}"
    return column


def prepare_predictions(
    df: pd.DataFrame,
    *,
    target_column: str,
    baseline_column: str,
    vtec_prediction_columns: list[str],
    stec_prediction_columns: list[str],
    mapping_column: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Validate and assemble VTEC prediction columns.

    ``vtec_prediction_columns`` are interpreted as already converted VTEC.
    ``stec_prediction_columns`` are divided by ``mapping_column`` to create
    VTEC estimates before evaluation.
    """

    required = [target_column, baseline_column]
    if stec_prediction_columns:
        required.append(mapping_column)
    required.extend(vtec_prediction_columns)
    required.extend(stec_prediction_columns)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Evaluation input is missing required columns: {missing}")

    out = df.copy()
    predictions: list[dict[str, str]] = [
        {
            "model": "slm",
            "prediction_column": baseline_column,
            "source": "vtec_prediction",
        }
    ]
    seen = {baseline_column}

    for column in vtec_prediction_columns:
        if column in seen:
            continue
        predictions.append(
            {
                "model": prediction_label(
                    column,
                    baseline_column=baseline_column,
                    source="vtec_prediction",
                ),
                "prediction_column": column,
                "source": "vtec_prediction",
            }
        )
        seen.add(column)

    if stec_prediction_columns:
        mapping = pd.to_numeric(out[mapping_column], errors="coerce")
        if (mapping <= 0).any():
            raise ValueError(f"{mapping_column} must be positive for conversion.")
        for column in stec_prediction_columns:
            derived_column = f"vtec_from_{column}"
            out[derived_column] = pd.to_numeric(out[column], errors="coerce") / mapping
            predictions.append(
                {
                    "model": prediction_label(
                        column,
                        baseline_column=baseline_column,
                        source="stec_prediction",
                    ),
                    "prediction_column": derived_column,
                    "source": "stec_prediction",
                    "source_column": column,
                }
            )

    return out, predictions


def metric_row(
    df: pd.DataFrame,
    *,
    target_column: str,
    prediction: dict[str, str],
    scope: str,
    group: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute one long-format metric row."""

    row: dict[str, Any] = {
        "scope": scope,
        "model": prediction["model"],
        "prediction_column": prediction["prediction_column"],
        "prediction_source": prediction["source"],
        "source_column": prediction.get("source_column", ""),
        "target_column": target_column,
    }
    if group:
        row.update(group)
    row.update(
        vtec_error_metrics(
            df[target_column],
            df[prediction["prediction_column"]],
        )
    )
    return row


def build_metric_rows(
    df: pd.DataFrame,
    *,
    target_column: str,
    predictions: list[dict[str, str]],
    elevation_bins: list[float],
    station_min_count: int,
) -> pd.DataFrame:
    """Compute overall and grouped VTEC-space metrics."""

    rows: list[dict[str, Any]] = []

    for prediction in predictions:
        rows.append(
            metric_row(
                df,
                target_column=target_column,
                prediction=prediction,
                scope="overall",
            )
        )

    if "elevation" in df.columns:
        labels = [
            f"[{elevation_bins[i]:g},{elevation_bins[i + 1]:g})"
            for i in range(len(elevation_bins) - 1)
        ]
        grouped = df.copy()
        grouped["elevation_bin_deg"] = pd.cut(
            grouped["elevation"],
            bins=elevation_bins,
            labels=labels,
            right=False,
            include_lowest=True,
        )
        for bin_label, group_df in grouped.groupby("elevation_bin_deg", observed=True):
            for prediction in predictions:
                rows.append(
                    metric_row(
                        group_df,
                        target_column=target_column,
                        prediction=prediction,
                        scope="elevation_bin",
                        group={"elevation_bin_deg": str(bin_label)},
                    )
                )

    if "station" in df.columns:
        for station, group_df in df.groupby("station", observed=True):
            if len(group_df) < station_min_count:
                continue
            for prediction in predictions:
                rows.append(
                    metric_row(
                        group_df,
                        target_column=target_column,
                        prediction=prediction,
                        scope="station",
                        group={"station": str(station)},
                    )
                )

    if "timestamp" in df.columns:
        grouped = df.copy()
        grouped["timestamp"] = pd.to_datetime(grouped["timestamp"])
        grouped["utc_hour"] = grouped["timestamp"].dt.floor("h").dt.strftime(
            "%Y-%m-%dT%H:00:00"
        )
        for utc_hour, group_df in grouped.groupby("utc_hour", observed=True):
            for prediction in predictions:
                rows.append(
                    metric_row(
                        group_df,
                        target_column=target_column,
                        prediction=prediction,
                        scope="utc_hour",
                        group={"utc_hour": str(utc_hour)},
                    )
                )

    return pd.DataFrame(rows)


def add_baseline_comparison(
    metrics: pd.DataFrame,
    *,
    baseline_model: str = "slm",
) -> pd.DataFrame:
    """Add metric deltas relative to the SLM baseline within each scope/group."""

    if metrics.empty:
        return metrics

    group_columns = [
        column
        for column in ["scope", "elevation_bin_deg", "station", "utc_hour"]
        if column in metrics.columns
    ]
    baseline = metrics.loc[metrics["model"] == baseline_model].copy()
    keep = group_columns + ["rmse_tecu", "mae_tecu", "bias_tecu"]
    baseline = baseline.loc[:, keep].rename(
        columns={
            "rmse_tecu": "baseline_rmse_tecu",
            "mae_tecu": "baseline_mae_tecu",
            "bias_tecu": "baseline_bias_tecu",
        }
    )
    out = metrics.merge(baseline, on=group_columns, how="left")
    out["delta_rmse_vs_slm_tecu"] = out["rmse_tecu"] - out["baseline_rmse_tecu"]
    out["delta_mae_vs_slm_tecu"] = out["mae_tecu"] - out["baseline_mae_tecu"]
    out["delta_abs_bias_vs_slm_tecu"] = (
        out["bias_tecu"].abs() - out["baseline_bias_tecu"].abs()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rmse_reduction_vs_slm_pct"] = (
            (out["baseline_rmse_tecu"] - out["rmse_tecu"])
            / out["baseline_rmse_tecu"]
            * 100.0
        )
        out["mae_reduction_vs_slm_pct"] = (
            (out["baseline_mae_tecu"] - out["mae_tecu"])
            / out["baseline_mae_tecu"]
            * 100.0
        )
    return out


def build_vtec_metrics(
    *,
    input_path: str | Path,
    table_dir: str | Path,
    run_name: str,
    target_column: str,
    baseline_column: str,
    vtec_prediction_columns: list[str],
    stec_prediction_columns: list[str],
    mapping_column: str,
    elevation_bins: list[float],
    station_min_count: int,
    max_rows: int | None,
) -> dict[str, Any]:
    """Build VTEC-space metrics tables and metadata."""

    input_path = Path(input_path)
    table_dir = Path(table_dir)
    df_in = read_table(input_path)
    df = df_in.head(max_rows).copy() if max_rows else df_in.copy()
    df, predictions = prepare_predictions(
        df,
        target_column=target_column,
        baseline_column=baseline_column,
        vtec_prediction_columns=vtec_prediction_columns,
        stec_prediction_columns=stec_prediction_columns,
        mapping_column=mapping_column,
    )
    metrics = build_metric_rows(
        df,
        target_column=target_column,
        predictions=predictions,
        elevation_bins=elevation_bins,
        station_min_count=station_min_count,
    )
    metrics = add_baseline_comparison(metrics)

    all_path = save_table(metrics, table_dir / f"{run_name}_vtec_metrics.csv")
    output_paths = {"all_metrics": str(all_path)}
    for scope in ["overall", "elevation_bin", "station", "utc_hour"]:
        scope_df = metrics.loc[metrics["scope"] == scope].copy()
        if scope_df.empty:
            continue
        scope_path = save_table(
            scope_df,
            table_dir / f"{run_name}_vtec_{scope}_metrics.csv",
        )
        output_paths[f"{scope}_metrics"] = str(scope_path)

    metadata_path = table_dir / f"{run_name}_vtec_metrics.metadata.json"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "row_counts": {
            "input": int(len(df_in)),
            "evaluated": int(len(df)),
        },
        "target_column": target_column,
        "baseline_column": baseline_column,
        "mapping_column": mapping_column,
        "metric_convention": "error = prediction - reference",
        "primary_metrics": [
            "rmse_tecu",
            "mae_tecu",
            "bias_tecu",
            "median_error_tecu",
            "p05_error_tecu",
            "p95_error_tecu",
        ],
        "comparison_convention": (
            "Negative delta_rmse_vs_slm_tecu or positive "
            "rmse_reduction_vs_slm_pct means improvement over SLM."
        ),
        "elevation_bins": elevation_bins,
        "station_min_count": int(station_min_count),
        "predictions": predictions,
        "outputs": output_paths,
    }
    metadata["outputs"]["metadata"] = str(metadata_path)
    write_json(metadata, metadata_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--run-name", default="vtec_conversion")
    parser.add_argument("--target-column", default="vtec_iri_ref")
    parser.add_argument("--baseline-column", default="vtec_slm_from_iri_stec")
    parser.add_argument(
        "--vtec-prediction-columns",
        default="",
        help="Comma-separated VTEC estimate columns, excluding the SLM baseline.",
    )
    parser.add_argument(
        "--stec-prediction-columns",
        default="",
        help=(
            "Comma-separated STEC estimate columns to convert to VTEC using "
            "--mapping-column before evaluation."
        ),
    )
    parser.add_argument("--mapping-column", default="slm_mapping_factor_iri")
    parser.add_argument(
        "--elevation-bins",
        default="10,20,30,40,50,60,70,80,90",
        help="Comma-separated elevation bin edges in degrees.",
    )
    parser.add_argument("--station-min-count", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_vtec_metrics(
        input_path=args.input,
        table_dir=args.table_dir,
        run_name=args.run_name,
        target_column=args.target_column,
        baseline_column=args.baseline_column,
        vtec_prediction_columns=parse_csv_list(args.vtec_prediction_columns),
        stec_prediction_columns=parse_csv_list(args.stec_prediction_columns),
        mapping_column=args.mapping_column,
        elevation_bins=parse_float_list(args.elevation_bins),
        station_min_count=args.station_min_count,
        max_rows=args.max_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
