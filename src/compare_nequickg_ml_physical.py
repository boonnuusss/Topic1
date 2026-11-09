"""Compare NeQuick-G set-ratio validation metrics with physical baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _metric_row(label: str, family: str, row: pd.Series) -> dict[str, object]:
    return {
        "method": label,
        "family": family,
        "n": int(row["n"]),
        "rmse_tecu": float(row["rmse"]),
        "mae_tecu": float(row["mae"]),
        "bias_tecu": float(row["bias"]),
        "r2": float(row["r2"]),
    }


def build_comparison(
    epoch_metrics: pd.DataFrame,
    physical: pd.DataFrame,
    matched_cpp_physical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_epoch = {"epoch", "n_eval", "rmse_tecu", "mae_tecu", "bias_tecu", "r2"}
    missing_epoch = required_epoch.difference(epoch_metrics.columns)
    if missing_epoch:
        raise KeyError(f"Epoch metrics missing columns: {sorted(missing_epoch)}")

    high = physical.loc[
        (physical["scope"] == "minimum_elevation")
        & (physical["scope_label"] == ">=30")
    ].copy()
    if high.empty:
        raise ValueError("No NeQuick-G physical metrics found for elevation >=30 deg.")

    best_epoch = epoch_metrics.loc[epoch_metrics["rmse_tecu"].astype(float).idxmin()]
    final_epoch = epoch_metrics.loc[epoch_metrics["epoch"].astype(int).idxmax()]
    rows: list[dict[str, object]] = [
        {
            "method": f"set_ratio_mlp_best_e{int(best_epoch['epoch'])}",
            "family": "ml_validation_selected",
            "n": int(best_epoch["n_eval"]),
            "rmse_tecu": float(best_epoch["rmse_tecu"]),
            "mae_tecu": float(best_epoch["mae_tecu"]),
            "bias_tecu": float(best_epoch["bias_tecu"]),
            "r2": float(best_epoch["r2"]),
        },
        {
            "method": f"set_ratio_mlp_final_e{int(final_epoch['epoch'])}",
            "family": "ml_final_epoch",
            "n": int(final_epoch["n_eval"]),
            "rmse_tecu": float(final_epoch["rmse_tecu"]),
            "mae_tecu": float(final_epoch["mae_tecu"]),
            "bias_tecu": float(final_epoch["bias_tecu"]),
            "r2": float(final_epoch["r2"]),
        },
    ]

    fixed = high.loc[high["baseline"].str.match(r"slm_h\d")].copy()
    best_fixed = fixed.loc[fixed["rmse"].astype(float).idxmin()]
    physical_labels = [
        ("slm_h450", "slm_450km"),
        (str(best_fixed["baseline"]), "best_fixed_slm"),
        ("code_mslm", "code_mslm"),
        ("multilayer_chapman", "multilayer_chapman"),
        ("apmf_cpp_stec_only", "high_low_cpp_apmf_stec_only"),
        ("apmf_nequickg_calibrated", "nequickg_calibrated_apmf"),
        ("nequickg_ideal_oracle", "nequickg_ideal_oracle"),
    ]
    for baseline, label in physical_labels:
        selected = high.loc[high["baseline"] == baseline]
        if selected.empty:
            raise ValueError(f"Missing physical baseline: {baseline}")
        rows.append(_metric_row(label, "physical", selected.iloc[0]))

    if matched_cpp_physical is not None:
        matched_high = matched_cpp_physical.loc[
            (matched_cpp_physical["scope"] == "minimum_elevation")
            & (matched_cpp_physical["scope_label"] == ">=30")
            & (matched_cpp_physical["baseline"] == "apmf_cpp_stec_only")
        ]
        if len(matched_high) != 1:
            raise ValueError("Expected one matched-window CPP APMF metric row.")
        rows.append(
            _metric_row(
                "high_low_cpp_apmf_matched_ut0408",
                "physical_stec_only_matched_window",
                matched_high.iloc[0],
            )
        )

    comparison = pd.DataFrame(rows)
    slm_rmse = float(high.loc[high["baseline"] == "slm_h450", "rmse"].iloc[0])
    apmf_rmse = float(
        high.loc[high["baseline"] == "apmf_nequickg_calibrated", "rmse"].iloc[0]
    )
    cpp_rmse = float(
        high.loc[high["baseline"] == "apmf_cpp_stec_only", "rmse"].iloc[0]
    )
    comparison["delta_rmse_vs_slm450_tecu"] = comparison["rmse_tecu"] - slm_rmse
    comparison["rmse_reduction_vs_slm450_pct"] = (
        (slm_rmse - comparison["rmse_tecu"]) / slm_rmse * 100.0
    )
    comparison["delta_rmse_vs_apmf_tecu"] = comparison["rmse_tecu"] - apmf_rmse
    comparison["delta_rmse_vs_cpp_apmf_tecu"] = (
        comparison["rmse_tecu"] - cpp_rmse
    )
    if matched_cpp_physical is not None:
        matched_cpp_rmse = float(matched_high.iloc[0]["rmse"])
        comparison["delta_rmse_vs_matched_cpp_apmf_tecu"] = (
            comparison["rmse_tecu"] - matched_cpp_rmse
        )
    return comparison.sort_values("rmse_tecu", kind="stable").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-metrics", required=True)
    parser.add_argument(
        "--physical-metrics",
        default=(
            "results/tables/"
            "nequickg_physical_mapping_baselines_ut05_validation_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "results/tables/"
            "nequickg_set_ratio_static_log_p192k_e25_vs_physical_"
            "ut05_validation.csv"
        ),
    )
    parser.add_argument("--matched-cpp-physical-metrics")
    args = parser.parse_args()

    matched = (
        pd.read_csv(args.matched_cpp_physical_metrics)
        if args.matched_cpp_physical_metrics
        else None
    )
    comparison = build_comparison(
        pd.read_csv(args.epoch_metrics),
        pd.read_csv(args.physical_metrics),
        matched,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    print(comparison.to_string(index=False))
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
