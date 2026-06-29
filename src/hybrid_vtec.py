"""Build hybrid VTEC estimates from ML and SLM conversion columns.

The first use case is a physically motivated low-elevation hybrid: use an ML
STEC-to-VTEC conversion at low elevation, where the single-layer mapping model
has its largest approximation error, and keep SLM at higher elevation, where the
traditional mapping is already strong.
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
    from .train_tree import DEFAULT_TABLE_DIR, read_table, save_table, write_json
except ImportError:
    from evaluate import build_vtec_metrics
    from train_tree import DEFAULT_TABLE_DIR, read_table, save_table, write_json


DEFAULT_INPUT = (
    "results/tables/"
    "ray_mlp_sim_0500_nearby200_e100_source_ratio_predictions.parquet"
)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Hybrid VTEC input is missing columns: {missing}")


def build_hybrid_vtec(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    table_dir = Path(args.table_dir)
    run_name = args.run_name or output_path.stem

    df = read_table(input_path)
    ml_constellations = [
        item.strip()
        for item in str(args.ml_constellations).split(",")
        if item.strip()
    ]
    required = [
        args.ml_vtec_column,
        args.slm_vtec_column,
        args.elevation_column,
        args.target_column,
        args.baseline_column,
    ]
    if ml_constellations:
        required.append(args.constellation_column)
    require_columns(df, required)

    out = df.copy()
    elevation = pd.to_numeric(out[args.elevation_column], errors="coerce")
    ml_values = pd.to_numeric(out[args.ml_vtec_column], errors="coerce")
    slm_values = pd.to_numeric(out[args.slm_vtec_column], errors="coerce")

    use_ml = elevation < float(args.elevation_threshold_deg)
    if ml_constellations:
        allowed = {item.lower() for item in ml_constellations}
        constellation = out[args.constellation_column].astype(str).str.lower()
        use_ml = use_ml & constellation.isin(allowed)
    if args.ml_must_lower_slm:
        use_ml = use_ml & (ml_values < slm_values)

    ml_source_label = f"{args.ml_vtec_column}_elevation_lt_{args.elevation_threshold_deg:g}"
    if ml_constellations:
        ml_source_label += "_constellation_in_" + "-".join(ml_constellations)
    if args.ml_must_lower_slm:
        ml_source_label += "_ml_lt_slm"
    slm_source_label = f"{args.slm_vtec_column}_fallback"

    out[args.output_column] = np.where(use_ml, ml_values, slm_values)
    out[f"{args.output_column}_use_ml"] = use_ml
    out[f"{args.output_column}_source"] = np.where(
        use_ml,
        ml_source_label,
        slm_source_label,
    )
    out[f"{args.output_column}_error_iri"] = (
        out[args.output_column] - out[args.target_column]
    )
    out[f"{args.output_column}_abs_error_iri"] = out[
        f"{args.output_column}_error_iri"
    ].abs()

    pred_path = save_table(out, output_path)

    metric_meta = build_vtec_metrics(
        input_path=pred_path,
        table_dir=table_dir,
        run_name=run_name,
        target_column=args.target_column,
        baseline_column=args.baseline_column,
        vtec_prediction_columns=[args.output_column, args.ml_vtec_column],
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

    selection_rule = (
        f"Use {args.ml_vtec_column} when {args.elevation_column} < "
        f"{args.elevation_threshold_deg:g} deg"
    )
    if ml_constellations:
        selection_rule += (
            f" and {args.constellation_column} in "
            f"{', '.join(ml_constellations)}"
        )
    if args.ml_must_lower_slm:
        selection_rule += f" and {args.ml_vtec_column} < {args.slm_vtec_column}"
    selection_rule += f"; otherwise use {args.slm_vtec_column}."

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "hybrid_vtec_low_elevation_ml",
        "run_name": run_name,
        "input_file": str(input_path),
        "output_file": str(pred_path),
        "ml_vtec_column": args.ml_vtec_column,
        "slm_vtec_column": args.slm_vtec_column,
        "output_column": args.output_column,
        "target_column": args.target_column,
        "baseline_column": args.baseline_column,
        "elevation_column": args.elevation_column,
        "elevation_threshold_deg": float(args.elevation_threshold_deg),
        "constellation_column": args.constellation_column,
        "ml_constellations": ml_constellations,
        "ml_must_lower_slm": bool(args.ml_must_lower_slm),
        "selection_rule": selection_rule,
        "vtec_reference_used_for_selection": False,
        "row_counts": {
            "input": int(len(df)),
            "use_ml": int(use_ml.sum()),
            "use_slm": int((~use_ml).sum()),
        },
        "outputs": {
            "predictions": str(pred_path),
            "vtec_metrics": metric_meta["outputs"],
        },
    }
    metadata_path = write_json(metadata, table_dir / f"{run_name}_hybrid.metadata.json")
    metadata["outputs"]["metadata"] = str(metadata_path)
    write_json(metadata, metadata_path)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        default="results/tables/ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid20.parquet",
    )
    parser.add_argument("--ml-vtec-column", default="vtec_ray_mlp")
    parser.add_argument("--slm-vtec-column", default="vtec_slm_from_iri_stec")
    parser.add_argument("--output-column", default="vtec_hybrid_lowele_ml")
    parser.add_argument("--elevation-column", default="elevation")
    parser.add_argument("--elevation-threshold-deg", type=float, default=20.0)
    parser.add_argument("--constellation-column", default="constellation")
    parser.add_argument(
        "--ml-constellations",
        default="",
        help="Comma-separated constellation names allowed to use ML. Empty means all.",
    )
    parser.add_argument(
        "--ml-must-lower-slm",
        action="store_true",
        help="Use ML only where its VTEC estimate is lower than the SLM estimate.",
    )
    parser.add_argument("--target-column", default="vtec_iri_ref")
    parser.add_argument("--baseline-column", default="vtec_slm_from_iri_stec")
    parser.add_argument("--mapping-column", default="slm_mapping_factor_iri")
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--run-name")
    parser.add_argument("--elevation-bins", default="10,20,30,40,50,60,70,80,90")
    parser.add_argument("--station-min-count", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = build_hybrid_vtec(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())