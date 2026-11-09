"""Evaluate physical STEC-to-VTEC mapping baselines on IRI simulation data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import regression_metrics
from .physical_mapping import (
    APMFModel,
    build_high_low_cpp_pairs,
    chapman_plasmasphere_weights,
    code_mslm_slant_factor,
    multilayer_slant_factor,
    slm_slant_factor,
)


DEFAULT_INPUT = (
    "datasets/interim/"
    "iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet"
)
DEFAULT_EVALUATION_KEYS = (
    "results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet"
)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")
    return path


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_key_columns(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def select_cpp_apmf_ridge(
    pairs: pd.DataFrame,
    *,
    candidates: list[float],
    folds: int,
    timestamp_column: str,
    elevation_column: str,
    azimuth_column: str,
    local_time_column: str,
    latitude_column: str,
    order: int,
    latitude_band_width_deg: float,
    min_band_samples: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Select CPP APMF regularization using STEC-derived fractions only."""

    if folds < 2:
        raise ValueError("CPP ridge cross-validation requires at least two folds.")
    if not candidates or any(value < 0 for value in candidates):
        raise ValueError("CPP ridge candidates must be nonnegative and nonempty.")
    timestamps = pd.to_datetime(pairs[timestamp_column])
    unique_epochs = np.array(sorted(timestamps.unique()))
    if len(unique_epochs) < folds:
        raise ValueError("Too few CPP epochs for the requested cross-validation.")
    epoch_folds = {
        pd.Timestamp(epoch): index % folds
        for index, epoch in enumerate(unique_epochs)
    }
    fold_ids = timestamps.map(epoch_folds).to_numpy(int)
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        fold_rmse: list[float] = []
        for fold in range(folds):
            train = pairs.loc[fold_ids != fold]
            validation = pairs.loc[fold_ids == fold]
            model = APMFModel.fit_vertical_fraction(
                train,
                fraction_column="cpp_vertical_fraction",
                elevation_column=elevation_column,
                azimuth_column=azimuth_column,
                local_time_column=local_time_column,
                latitude_column=latitude_column,
                order=order,
                latitude_band_width_deg=latitude_band_width_deg,
                ridge_alpha=candidate,
                min_band_samples=min_band_samples,
            )
            predicted, _ = model.predict_vertical_fraction(
                validation,
                elevation_column=elevation_column,
                azimuth_column=azimuth_column,
                local_time_column=local_time_column,
                latitude_column=latitude_column,
            )
            error = predicted - validation["cpp_vertical_fraction"].to_numpy(float)
            fold_rmse.append(float(np.sqrt(np.mean(error**2))))
        records.append(
            {
                "ridge_alpha": float(candidate),
                "mean_cpp_fraction_rmse": float(np.mean(fold_rmse)),
                "std_cpp_fraction_rmse": float(np.std(fold_rmse, ddof=1)),
                "fold_rmse": fold_rmse,
            }
        )
    selected = min(
        records,
        key=lambda item: (item["mean_cpp_fraction_rmse"], item["ridge_alpha"]),
    )
    return float(selected["ridge_alpha"]), records


def key_mask(
    df: pd.DataFrame,
    keys: pd.DataFrame,
    key_columns: list[str],
) -> pd.Series:
    """Return rows in df matching a unique key table."""

    for column in key_columns:
        if column not in df.columns or column not in keys.columns:
            raise KeyError(f"Missing evaluation key column: {column}")
    left = df.loc[:, key_columns].copy()
    right = keys.loc[:, key_columns].drop_duplicates().copy()
    for column in key_columns:
        if "time" in column.lower():
            left[column] = pd.to_datetime(left[column])
            right[column] = pd.to_datetime(right[column])
    right["__matched"] = True
    indexed = left.reset_index().merge(right, how="left", on=key_columns)
    return indexed.set_index("index")["__matched"].fillna(False).astype(bool)


def add_physical_predictions(
    evaluation_df: pd.DataFrame,
    training_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Attach all physical mapping factors and VTEC estimates."""

    out = evaluation_df.copy()
    reference_model = args.reference_model
    elevation = out[args.elevation_column].to_numpy(float)
    stec = out[args.stec_column].to_numpy(float)
    baseline_columns: dict[str, str] = {}

    for height in args.slm_heights:
        label = f"slm_h{height:g}".replace(".", "p")
        mapping_column = f"mapping_factor_{label}"
        vtec_column = f"vtec_{label}"
        out[mapping_column] = slm_slant_factor(
            elevation, shell_height_km=height
        )
        out[vtec_column] = stec / out[mapping_column].to_numpy(float)
        baseline_columns[label] = vtec_column

    out["mapping_factor_code_mslm"] = code_mslm_slant_factor(elevation)
    out["vtec_code_mslm"] = stec / out["mapping_factor_code_mslm"]
    baseline_columns["code_mslm"] = "vtec_code_mslm"

    heights = np.arange(
        args.multilayer_alt_min_km,
        args.multilayer_alt_max_km + 0.5 * args.multilayer_alt_step_km,
        args.multilayer_alt_step_km,
    )
    weights = chapman_plasmasphere_weights(
        heights,
        peak_height_km=args.multilayer_peak_height_km,
        chapman_scale_height_km=args.multilayer_scale_height_km,
        plasmasphere_density_ratio=args.multilayer_plasmasphere_ratio,
        plasmasphere_scale_height_km=args.multilayer_plasmasphere_scale_km,
    )
    out["mapping_factor_multilayer_chapman"] = multilayer_slant_factor(
        elevation,
        shell_heights_km=heights,
        shell_weights=weights,
    )
    out["vtec_multilayer_chapman"] = (
        stec / out["mapping_factor_multilayer_chapman"]
    )
    baseline_columns["multilayer_chapman"] = "vtec_multilayer_chapman"

    apmf = APMFModel.fit(
        training_df,
        stec_column=args.stec_column,
        vtec_column=args.truth_column,
        elevation_column=args.elevation_column,
        azimuth_column=args.azimuth_column,
        local_time_column=args.local_time_column,
        latitude_column=args.latitude_column,
        order=args.apmf_order,
        latitude_band_width_deg=args.apmf_lat_band_width_deg,
        ridge_alpha=args.apmf_ridge_alpha,
        min_band_samples=args.apmf_min_band_samples,
    )
    apmf_fraction, apmf_fallback = apmf.predict_vertical_fraction(
        out,
        elevation_column=args.elevation_column,
        azimuth_column=args.azimuth_column,
        local_time_column=args.local_time_column,
        latitude_column=args.latitude_column,
    )
    apmf_label = f"apmf_{reference_model}_calibrated"
    apmf_fraction_column = f"vertical_fraction_{apmf_label}"
    apmf_mapping_column = f"mapping_factor_{apmf_label}"
    apmf_vtec_column = f"vtec_{apmf_label}"
    out[apmf_fraction_column] = apmf_fraction
    out[apmf_mapping_column] = 1.0 / apmf_fraction
    out[apmf_vtec_column] = stec * apmf_fraction
    out["apmf_global_fallback"] = apmf_fallback
    baseline_columns[apmf_label] = apmf_vtec_column

    cpp_training_df = training_df
    if args.cpp_train_window_start or args.cpp_train_window_end:
        if not args.cpp_train_window_start or not args.cpp_train_window_end:
            raise ValueError(
                "Both CPP train-window boundaries must be supplied together."
            )
        cpp_timestamps = pd.to_datetime(cpp_training_df[args.timestamp_column])
        cpp_start = pd.Timestamp(args.cpp_train_window_start)
        cpp_end = pd.Timestamp(args.cpp_train_window_end)
        if cpp_end <= cpp_start:
            raise ValueError("CPP train-window end must be later than its start.")
        cpp_training_df = cpp_training_df.loc[
            (cpp_timestamps >= cpp_start) & (cpp_timestamps < cpp_end)
        ].copy()
    cpp_pairs = build_high_low_cpp_pairs(
        cpp_training_df,
        stec_column=args.stec_column,
        timestamp_column=args.timestamp_column,
        elevation_column=args.elevation_column,
        azimuth_column=args.azimuth_column,
        local_time_column=args.local_time_column,
        latitude_column=args.latitude_column,
        longitude_column=args.longitude_column,
        high_elevation_deg=args.cpp_high_elevation_deg,
        low_elevation_min_deg=args.cpp_low_elevation_min_deg,
        low_elevation_max_deg=args.cpp_low_elevation_max_deg,
        latitude_tolerance_deg=args.cpp_latitude_tolerance_deg,
        scaled_longitude_tolerance_deg=(
            args.cpp_scaled_longitude_tolerance_deg
        ),
        pair_policy=args.cpp_pair_policy,
    )
    cpp_ridge_alpha, cpp_ridge_cv = select_cpp_apmf_ridge(
        cpp_pairs,
        candidates=args.cpp_apmf_ridge_candidates,
        folds=args.cpp_apmf_cv_folds,
        timestamp_column=args.timestamp_column,
        elevation_column=args.elevation_column,
        azimuth_column=args.azimuth_column,
        local_time_column=args.local_time_column,
        latitude_column=args.latitude_column,
        order=args.apmf_order,
        latitude_band_width_deg=args.apmf_lat_band_width_deg,
        min_band_samples=args.apmf_min_band_samples,
    )
    cpp_apmf = APMFModel.fit_vertical_fraction(
        cpp_pairs,
        fraction_column="cpp_vertical_fraction",
        elevation_column=args.elevation_column,
        azimuth_column=args.azimuth_column,
        local_time_column=args.local_time_column,
        latitude_column=args.latitude_column,
        order=args.apmf_order,
        latitude_band_width_deg=args.apmf_lat_band_width_deg,
        ridge_alpha=cpp_ridge_alpha,
        min_band_samples=args.apmf_min_band_samples,
    )
    cpp_fraction, cpp_fallback = cpp_apmf.predict_vertical_fraction(
        out,
        elevation_column=args.elevation_column,
        azimuth_column=args.azimuth_column,
        local_time_column=args.local_time_column,
        latitude_column=args.latitude_column,
    )
    cpp_label = "apmf_cpp_stec_only"
    out[f"vertical_fraction_{cpp_label}"] = cpp_fraction
    out[f"mapping_factor_{cpp_label}"] = 1.0 / cpp_fraction
    out[f"vtec_{cpp_label}"] = stec * cpp_fraction
    out["apmf_cpp_global_fallback"] = cpp_fallback
    baseline_columns[cpp_label] = f"vtec_{cpp_label}"

    truth = out[args.truth_column].to_numpy(float)
    ideal_mapping = np.divide(
        stec,
        truth,
        out=np.full_like(stec, np.nan),
        where=truth > 0,
    )
    oracle_label = f"{reference_model}_ideal_oracle"
    oracle_mapping_column = f"mapping_factor_{oracle_label}"
    oracle_vtec_column = f"vtec_{oracle_label}"
    out[oracle_mapping_column] = ideal_mapping
    out[oracle_vtec_column] = stec / ideal_mapping
    baseline_columns[oracle_label] = oracle_vtec_column

    metadata = {
        "baseline_columns": baseline_columns,
        "slm_shell_heights_km": args.slm_heights,
        "code_mslm": {
            "shell_height_km": 506.7,
            "elevation_scale": 0.9782,
        },
        "multilayer_chapman": {
            "shell_count": int(len(heights)),
            "altitude_min_km": float(heights.min()),
            "altitude_max_km": float(heights.max()),
            "altitude_step_km": float(args.multilayer_alt_step_km),
            "peak_height_km": float(args.multilayer_peak_height_km),
            "chapman_scale_height_km": float(
                args.multilayer_scale_height_km
            ),
            "plasmasphere_density_ratio": float(
                args.multilayer_plasmasphere_ratio
            ),
            "plasmasphere_scale_height_km": float(
                args.multilayer_plasmasphere_scale_km
            ),
            "weighted_mean_shell_height_km": float(np.sum(heights * weights)),
        },
        "apmf": {
            "calibration_source": (
                f"{reference_model} training partition STEC/VTEC"
            ),
            "order": int(apmf.order),
            "latitude_band_width_deg": float(apmf.latitude_band_width_deg),
            "ridge_alpha": float(apmf.ridge_alpha),
            "min_band_samples": int(apmf.min_band_samples),
            "band_sample_counts": {
                str(key): value for key, value in apmf.band_sample_counts.items()
            },
            "fitted_band_ids": sorted(apmf.band_coefficients),
            "global_coefficients": apmf.global_coefficients.tolist(),
            "band_coefficients": {
                str(key): value.tolist()
                for key, value in apmf.band_coefficients.items()
            },
            "evaluation_global_fallback_rows": int(apmf_fallback.sum()),
            "evaluation_fraction_clipped_low": int(
                (apmf_fraction <= apmf.vertical_fraction_min).sum()
            ),
            "evaluation_fraction_clipped_high": int(
                (apmf_fraction >= apmf.vertical_fraction_max).sum()
            ),
            "vertical_fraction_clip": [
                apmf.vertical_fraction_min,
                apmf.vertical_fraction_max,
            ],
        },
        "apmf_cpp_stec_only": {
            "calibration_source": (
                "same-epoch high-low CPP pairs using STEC and CODE MSLM only"
            ),
            "vtec_reference_used_for_fit": False,
            "training_rows_before_window": int(len(training_df)),
            "training_rows_after_window": int(len(cpp_training_df)),
            "train_window_start": args.cpp_train_window_start,
            "train_window_end": args.cpp_train_window_end,
            "paper": "Huo et al. (2024), doi:10.1007/s00190-023-01819-w",
            "high_elevation_deg_exclusive": float(
                args.cpp_high_elevation_deg
            ),
            "low_elevation_range_deg": [
                float(args.cpp_low_elevation_min_deg),
                float(args.cpp_low_elevation_max_deg),
            ],
            "latitude_tolerance_deg": float(args.cpp_latitude_tolerance_deg),
            "scaled_longitude_tolerance_deg": float(
                args.cpp_scaled_longitude_tolerance_deg
            ),
            "pair_policy": args.cpp_pair_policy,
            "n_pairs": int(len(cpp_pairs)),
            "n_unique_low_rays": int(
                cpp_pairs["cpp_source_row_index"].nunique()
            ),
            "n_unique_high_rays": int(
                cpp_pairs["cpp_high_source_row_index"].nunique()
            ),
            "n_pair_epochs": int(
                pd.to_datetime(cpp_pairs[args.timestamp_column]).nunique()
            ),
            "vertical_fraction_stats": {
                "mean": float(cpp_pairs["cpp_vertical_fraction"].mean()),
                "min": float(cpp_pairs["cpp_vertical_fraction"].min()),
                "max": float(cpp_pairs["cpp_vertical_fraction"].max()),
            },
            "order": int(cpp_apmf.order),
            "latitude_band_width_deg": float(
                cpp_apmf.latitude_band_width_deg
            ),
            "ridge_alpha": float(cpp_apmf.ridge_alpha),
            "ridge_selection": {
                "uses_vtec_reference": False,
                "criterion": "same-epoch-fold CPP vertical-fraction RMSE",
                "folds": int(args.cpp_apmf_cv_folds),
                "candidates": args.cpp_apmf_ridge_candidates,
                "results": cpp_ridge_cv,
            },
            "min_band_samples": int(cpp_apmf.min_band_samples),
            "band_sample_counts": {
                str(key): value
                for key, value in cpp_apmf.band_sample_counts.items()
            },
            "fitted_band_ids": sorted(cpp_apmf.band_coefficients),
            "evaluation_global_fallback_rows": int(cpp_fallback.sum()),
            "evaluation_fraction_clipped_low": int(
                (cpp_fraction <= cpp_apmf.vertical_fraction_min).sum()
            ),
            "evaluation_fraction_clipped_high": int(
                (cpp_fraction >= cpp_apmf.vertical_fraction_max).sum()
            ),
        },
        f"{reference_model}_ideal_oracle": {
            "formula": f"{args.stec_column} / {args.truth_column}",
            "role": "simulation-only oracle upper bound",
        },
        "references": {
            "code_mslm": "Schaer (1999); H=506.7 km, alpha=0.9782",
            "apmf": "Huo et al. (2024), doi:10.1007/s00190-023-01819-w",
            "multilayer": (
                "Hoque et al. (2025), doi:10.1186/s43020-025-00182-9"
            ),
        },
    }
    return out, metadata, cpp_pairs


def build_metric_rows(
    df: pd.DataFrame,
    *,
    truth_column: str,
    elevation_column: str,
    baseline_columns: dict[str, str],
    report_min_elevations: list[float],
    elevation_bins: list[float],
) -> pd.DataFrame:
    """Build overall/high-elevation and elevation-bin metrics."""

    rows: list[dict[str, Any]] = []

    def append_metrics(
        frame: pd.DataFrame,
        baseline: str,
        prediction_column: str,
        scope: str,
        label: str,
    ) -> None:
        valid = frame[[truth_column, prediction_column]].dropna()
        metrics = regression_metrics(
            valid[truth_column].to_numpy(float),
            valid[prediction_column].to_numpy(float),
        )
        rows.append(
            {
                "baseline": baseline,
                "prediction_column": prediction_column,
                "scope": scope,
                "scope_label": label,
                "n": int(len(valid)),
                **metrics,
            }
        )

    for baseline, prediction_column in baseline_columns.items():
        for threshold in report_min_elevations:
            selected = df.loc[df[elevation_column] >= threshold]
            append_metrics(
                selected,
                baseline,
                prediction_column,
                "minimum_elevation",
                f">={threshold:g}",
            )
        binned = pd.cut(
            df[elevation_column],
            bins=elevation_bins,
            right=False,
        )
        for label in binned.dropna().unique().sort_values():
            append_metrics(
                df.loc[binned == label],
                baseline,
                prediction_column,
                "elevation_bin",
                str(label),
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    df = read_table(input_path)
    required = [
        args.stec_column,
        args.truth_column,
        args.elevation_column,
        args.azimuth_column,
        args.local_time_column,
        args.latitude_column,
        args.longitude_column,
        args.timestamp_column,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Input is missing required columns: {missing}")

    if args.evaluation_keys_from:
        keys = read_table(args.evaluation_keys_from)
        mask = key_mask(df, keys, args.key_columns)
        training_df = df.loc[~mask].copy()
        evaluation_df = df.loc[mask].copy()
    else:
        training_df = df.copy()
        evaluation_df = df.copy()
    if evaluation_df.empty:
        raise ValueError("No rows matched the requested evaluation keys.")
    training_df = training_df.loc[
        training_df[args.elevation_column] >= args.min_fit_elevation_deg
    ].copy()
    evaluation_df = evaluation_df.loc[
        evaluation_df[args.elevation_column] >= args.min_output_elevation_deg
    ].copy()

    predictions, method_metadata, cpp_pairs = add_physical_predictions(
        evaluation_df,
        training_df,
        args=args,
    )
    metrics = build_metric_rows(
        predictions,
        truth_column=args.truth_column,
        elevation_column=args.elevation_column,
        baseline_columns=method_metadata["baseline_columns"],
        report_min_elevations=args.report_min_elevations,
        elevation_bins=args.elevation_bins,
    )
    metrics.insert(0, "reference_model", args.reference_model)
    save_table(predictions, args.output)
    save_table(metrics, args.metrics_output)
    if args.cpp_pairs_output:
        save_table(cpp_pairs, args.cpp_pairs_output)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(Path(args.output)),
        "metrics_output": str(Path(args.metrics_output)),
        "evaluation_keys_from": args.evaluation_keys_from,
        "key_columns": args.key_columns,
        "training_rows": int(len(training_df)),
        "evaluation_rows": int(len(predictions)),
        "stec_column": args.stec_column,
        "truth_column": args.truth_column,
        "reference_model": args.reference_model,
        "truth_usage": (
            "NeQuick-truth APMF calibration and evaluation scoring; "
            "CPP APMF calibration uses STEC and geometry only; "
            f"{args.reference_model} oracle construction on evaluation partition"
        ),
        "method_metadata": method_metadata,
        "cpp_pairs_output": args.cpp_pairs_output,
    }
    metadata_path = Path(args.metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--reference-model",
        choices=["iri", "nequickg"],
        default="iri",
    )
    parser.add_argument(
        "--output",
        default="results/tables/physical_mapping_baseline_predictions.parquet",
    )
    parser.add_argument(
        "--metrics-output",
        default="results/tables/physical_mapping_baseline_metrics.csv",
    )
    parser.add_argument(
        "--metadata-output",
        default="results/tables/physical_mapping_baseline_metadata.json",
    )
    parser.add_argument("--evaluation-keys-from", default=DEFAULT_EVALUATION_KEYS)
    parser.add_argument("--key-columns", default="timestamp,station,prn")
    parser.add_argument("--stec-column", default="stec_iri_observable")
    parser.add_argument("--truth-column", default="vtec_iri_ref")
    parser.add_argument("--elevation-column", default="elevation")
    parser.add_argument("--azimuth-column", default="azimuth")
    parser.add_argument("--local-time-column", default="local_time_ipp")
    parser.add_argument("--latitude-column", default="ipp_lat")
    parser.add_argument("--longitude-column", default="ipp_lon")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--slm-heights", default="300,350,400,450,500,550,600")
    parser.add_argument("--report-min-elevations", default="10,30")
    parser.add_argument("--elevation-bins", default="10,20,30,40,50,60,70,80,90.0001")
    parser.add_argument("--min-fit-elevation-deg", type=float, default=10.0)
    parser.add_argument("--min-output-elevation-deg", type=float, default=10.0)
    parser.add_argument("--apmf-order", type=int, default=12)
    parser.add_argument("--apmf-lat-band-width-deg", type=float, default=5.0)
    parser.add_argument("--apmf-ridge-alpha", type=float, default=1e-4)
    parser.add_argument("--apmf-min-band-samples", type=int, default=100)
    parser.add_argument("--cpp-high-elevation-deg", type=float, default=70.0)
    parser.add_argument("--cpp-low-elevation-min-deg", type=float, default=10.0)
    parser.add_argument("--cpp-low-elevation-max-deg", type=float, default=30.0)
    parser.add_argument("--cpp-latitude-tolerance-deg", type=float, default=0.4)
    parser.add_argument(
        "--cpp-scaled-longitude-tolerance-deg", type=float, default=0.4
    )
    parser.add_argument(
        "--cpp-pair-policy", choices=["all", "nearest"], default="all"
    )
    parser.add_argument("--cpp-pairs-output")
    parser.add_argument(
        "--cpp-apmf-ridge-candidates",
        default="0.0001,0.001,0.01,0.1,1,10,100",
    )
    parser.add_argument("--cpp-apmf-cv-folds", type=int, default=5)
    parser.add_argument("--cpp-train-window-start")
    parser.add_argument("--cpp-train-window-end")
    parser.add_argument("--multilayer-alt-min-km", type=float, default=90.0)
    parser.add_argument("--multilayer-alt-max-km", type=float, default=2000.0)
    parser.add_argument("--multilayer-alt-step-km", type=float, default=10.0)
    parser.add_argument("--multilayer-peak-height-km", type=float, default=350.0)
    parser.add_argument("--multilayer-scale-height-km", type=float, default=70.0)
    parser.add_argument("--multilayer-plasmasphere-ratio", type=float, default=100.0)
    parser.add_argument(
        "--multilayer-plasmasphere-scale-km", type=float, default=10000.0
    )
    args = parser.parse_args(argv)
    args.key_columns = parse_key_columns(args.key_columns)
    args.slm_heights = parse_float_list(args.slm_heights)
    args.report_min_elevations = parse_float_list(args.report_min_elevations)
    args.elevation_bins = parse_float_list(args.elevation_bins)
    args.cpp_apmf_ridge_candidates = parse_float_list(
        args.cpp_apmf_ridge_candidates
    )
    return args


def main(argv: list[str] | None = None) -> int:
    metadata = run(parse_args(argv))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
