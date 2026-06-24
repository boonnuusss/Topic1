"""Build a single-layer ionosphere mapping baseline for ray-wise STEC tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .baselines import add_single_layer_baseline, add_slm_residual
    from .metrics import regression_metrics
except ImportError:
    from baselines import add_single_layer_baseline, add_slm_residual
    from metrics import regression_metrics


DEFAULT_MERGE_KEYS = ["timestamp", "station", "prn"]


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
    """Save a table as parquet or CSV based on the output suffix."""

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


def _coerce_merge_keys(
    df: pd.DataFrame,
    vtec_df: pd.DataFrame,
    merge_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make merge-key dtypes compatible where this is unambiguous."""

    out = df.copy()
    vtec = vtec_df.copy()
    for key in merge_keys:
        if key not in out.columns:
            raise KeyError(f"Input table is missing merge key: {key}")
        if key not in vtec.columns:
            raise KeyError(f"VTEC table is missing merge key: {key}")
        if key == "timestamp":
            out[key] = pd.to_datetime(out[key])
            vtec[key] = pd.to_datetime(vtec[key])
    return out, vtec


def attach_vtec_from_table(
    df: pd.DataFrame,
    vtec_table: str | Path,
    merge_keys: list[str],
    vtec_table_column: str,
    vtec_column: str,
) -> pd.DataFrame:
    """Attach row-wise VTEC values from an external table by exact keys."""

    vtec_df = read_table(vtec_table)
    df, vtec_df = _coerce_merge_keys(df, vtec_df, merge_keys)
    if vtec_table_column not in vtec_df.columns:
        raise KeyError(f"VTEC table is missing column: {vtec_table_column}")

    duplicated = vtec_df.duplicated(subset=merge_keys, keep=False)
    if duplicated.any():
        n_dup = int(duplicated.sum())
        raise ValueError(
            f"VTEC table has {n_dup} duplicate rows for merge keys {merge_keys}."
        )

    if vtec_column in df.columns:
        df = df.drop(columns=[vtec_column])

    columns = merge_keys + [vtec_table_column]
    vtec_df = vtec_df.loc[:, columns].rename(columns={vtec_table_column: vtec_column})
    return df.merge(vtec_df, on=merge_keys, how="left", validate="many_to_one")


def attach_vtec_source(
    df: pd.DataFrame,
    vtec_column: str,
    vtec_source: str | None,
    vtec_table: str | Path | None,
    vtec_table_column: str,
    merge_keys: list[str],
    vtec_constant: float | None,
    allow_diagnostic_constant: bool,
) -> tuple[pd.DataFrame, str, str]:
    """Attach or validate the VTEC source used by the SLM baseline."""

    if vtec_table and vtec_constant is not None:
        raise ValueError("Use either --vtec-table or --vtec-constant, not both.")

    if vtec_constant is not None:
        if not allow_diagnostic_constant:
            raise ValueError(
                "Constant VTEC is diagnostic only. Re-run with "
                "--allow-diagnostic-constant if this is intentional."
            )
        out = df.copy()
        out[vtec_column] = float(vtec_constant)
        source = vtec_source or f"diagnostic_constant_{vtec_constant:g}_TECU"
        return out, source, "diagnostic_constant"

    if vtec_table:
        out = attach_vtec_from_table(
            df=df,
            vtec_table=vtec_table,
            merge_keys=merge_keys,
            vtec_table_column=vtec_table_column,
            vtec_column=vtec_column,
        )
        source = vtec_source or f"table:{Path(vtec_table).name}"
        return out, source, "table"

    if vtec_column not in df.columns:
        raise KeyError(
            f"Input table has no {vtec_column!r}. Provide --vtec-table or add "
            "an independent VTEC column before building the SLM baseline."
        )

    source = vtec_source or f"column:{vtec_column}"
    return df.copy(), source, "input_column"


def _numeric_stats(values: pd.Series) -> dict[str, float | int | None]:
    """Return compact JSON-safe stats for a numeric series."""

    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    stats: dict[str, float | int | None] = {
        "count": int(valid.size),
        "missing": int(numeric.isna().sum()),
    }
    if valid.empty:
        stats.update({"min": None, "median": None, "max": None, "mean": None})
        return stats
    stats.update(
        {
            "min": float(valid.min()),
            "median": float(valid.median()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
        }
    )
    return stats


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place SLM baseline columns near the observed STEC target."""

    baseline_columns = [
        "vtec_baseline",
        "vtec_source",
        "shell_height_km",
        "slm_mapping_factor",
        "stec_slm",
        "stec_residual_slm",
    ]
    existing_baseline = [col for col in baseline_columns if col in df.columns]
    if "stec_obs" not in df.columns:
        first = [col for col in df.columns if col not in existing_baseline]
        return df.loc[:, first + existing_baseline]

    before = []
    after = []
    seen_stec = False
    for col in df.columns:
        if col in existing_baseline:
            continue
        if col == "stec_obs":
            seen_stec = True
            continue
        if seen_stec:
            after.append(col)
        else:
            before.append(col)
    return df.loc[:, before + ["stec_obs"] + existing_baseline + after]


def build_metadata(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    *,
    vtec_column: str,
    vtec_source: str,
    vtec_mode: str,
    shell_height_km: float,
    mapping_column: str,
    baseline_column: str,
    residual_column: str,
    dropped_missing_vtec: int,
) -> dict[str, Any]:
    """Build a JSON-serializable metadata record."""

    metrics = None
    if {"stec_obs", baseline_column}.issubset(df_out.columns):
        mask = df_out["stec_obs"].notna() & df_out[baseline_column].notna()
        if mask.any():
            metrics = regression_metrics(
                df_out.loc[mask, "stec_obs"],
                df_out.loc[mask, baseline_column],
            )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "metadata_file": str(metadata_path),
        "row_counts": {
            "input": int(len(df_in)),
            "output": int(len(df_out)),
            "dropped_missing_vtec": int(dropped_missing_vtec),
        },
        "single_layer_model": {
            "formula": f"{baseline_column} = {mapping_column} * {vtec_column}",
            "mapping_function": "1 / sqrt(1 - (Re / (Re + H) * cos(E))^2)",
            "earth_radius_km": 6371.0,
            "shell_height_km": float(shell_height_km),
            "vtec_column": vtec_column,
            "vtec_source": vtec_source,
            "vtec_mode": vtec_mode,
            "baseline_column": baseline_column,
            "residual_column": residual_column,
        },
        "stats": {
            vtec_column: _numeric_stats(df_out[vtec_column]),
            mapping_column: _numeric_stats(df_out[mapping_column]),
            baseline_column: _numeric_stats(df_out[baseline_column]),
            residual_column: _numeric_stats(df_out[residual_column]),
        },
        "metrics_vs_stec_obs": metrics,
    }


def build_slm_baseline(
    input_path: str | Path,
    output_path: str | Path,
    *,
    vtec_column: str = "vtec_baseline",
    vtec_source: str | None = None,
    vtec_table: str | Path | None = None,
    vtec_table_column: str = "vtec_baseline",
    merge_keys: list[str] | None = None,
    vtec_constant: float | None = None,
    allow_diagnostic_constant: bool = False,
    drop_missing_vtec: bool = False,
    shell_height_km: float = 450.0,
    mapping_column: str = "slm_mapping_factor",
    baseline_column: str = "stec_slm",
    residual_column: str = "stec_residual_slm",
    metadata_path: str | Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Read a ray-wise table, attach VTEC, and write SLM baseline columns."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path) if metadata_path else output_path.with_suffix(
        ".metadata.json"
    )
    merge_keys = merge_keys or DEFAULT_MERGE_KEYS

    df_in = read_table(input_path)
    df = df_in.head(max_rows).copy() if max_rows else df_in.copy()
    df, source, vtec_mode = attach_vtec_source(
        df=df,
        vtec_column=vtec_column,
        vtec_source=vtec_source,
        vtec_table=vtec_table,
        vtec_table_column=vtec_table_column,
        merge_keys=merge_keys,
        vtec_constant=vtec_constant,
        allow_diagnostic_constant=allow_diagnostic_constant,
    )
    df["vtec_source"] = source

    missing_vtec = df[vtec_column].isna()
    n_missing = int(missing_vtec.sum())
    if n_missing:
        if not drop_missing_vtec:
            raise ValueError(
                f"{n_missing} rows are missing {vtec_column!r}. Use "
                "--drop-missing-vtec to write only matched rows."
            )
        df = df.loc[~missing_vtec].copy()

    if df.empty:
        raise ValueError("No rows remain after VTEC filtering.")

    df = add_single_layer_baseline(
        df,
        vtec_column=vtec_column,
        output_column=baseline_column,
        mapping_column=mapping_column,
        shell_height_km=shell_height_km,
        use_existing_mapping=False,
    )
    df = add_slm_residual(
        df,
        baseline_column=baseline_column,
        output_column=residual_column,
    )
    df = _order_columns(df)
    saved_path = save_table(df, output_path)

    metadata = build_metadata(
        input_path=input_path,
        output_path=saved_path,
        metadata_path=metadata_path,
        df_in=df_in,
        df_out=df,
        vtec_column=vtec_column,
        vtec_source=source,
        vtec_mode=vtec_mode,
        shell_height_km=shell_height_km,
        mapping_column=mapping_column,
        baseline_column=baseline_column,
        residual_column=residual_column,
        dropped_missing_vtec=n_missing,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet/CSV table.")
    parser.add_argument("--output", required=True, help="Output parquet/CSV table.")
    parser.add_argument(
        "--metadata-output",
        help="Output metadata JSON. Defaults to <output>.metadata.json.",
    )
    parser.add_argument("--vtec-column", default="vtec_baseline")
    parser.add_argument("--vtec-source", help="Independent VTEC source label.")
    parser.add_argument("--vtec-table", help="Optional row-wise VTEC table.")
    parser.add_argument("--vtec-table-column", default="vtec_baseline")
    parser.add_argument(
        "--vtec-merge-keys",
        nargs="+",
        default=DEFAULT_MERGE_KEYS,
        help="Exact merge keys for --vtec-table.",
    )
    parser.add_argument("--vtec-constant", type=float)
    parser.add_argument(
        "--allow-diagnostic-constant",
        action="store_true",
        help="Allow constant VTEC for smoke tests only.",
    )
    parser.add_argument("--drop-missing-vtec", action="store_true")
    parser.add_argument("--shell-height-km", type=float, default=450.0)
    parser.add_argument("--mapping-column", default="slm_mapping_factor")
    parser.add_argument("--baseline-column", default="stec_slm")
    parser.add_argument("--residual-column", default="stec_residual_slm")
    parser.add_argument("--max-rows", type=int, help="Optional smoke-test row cap.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = build_slm_baseline(
        input_path=args.input,
        output_path=args.output,
        metadata_path=args.metadata_output,
        vtec_column=args.vtec_column,
        vtec_source=args.vtec_source,
        vtec_table=args.vtec_table,
        vtec_table_column=args.vtec_table_column,
        merge_keys=args.vtec_merge_keys,
        vtec_constant=args.vtec_constant,
        allow_diagnostic_constant=args.allow_diagnostic_constant,
        drop_missing_vtec=args.drop_missing_vtec,
        shell_height_km=args.shell_height_km,
        mapping_column=args.mapping_column,
        baseline_column=args.baseline_column,
        residual_column=args.residual_column,
        max_rows=args.max_rows,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
