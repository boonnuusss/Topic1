"""Build a traditional SLM VTEC product from observed ray-wise STEC."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .baselines import add_slm_vtec_from_stec
except ImportError:
    from baselines import add_slm_vtec_from_stec


DEFAULT_INPUT = "datasets/processed/raystec_china_20240505.parquet"
DEFAULT_OUTPUT = "datasets/processed/rayvtec_china_20240505_slm.parquet"


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


def _order_columns(
    df: pd.DataFrame,
    *,
    stec_column: str,
    vtec_column: str,
    mapping_column: str,
) -> pd.DataFrame:
    """Place SLM VTEC columns near the observed STEC column."""

    slm_columns = [
        "shell_height_km",
        mapping_column,
        vtec_column,
        "vtec_slm_source",
    ]
    existing_slm = [col for col in slm_columns if col in df.columns]
    if stec_column not in df.columns:
        first = [col for col in df.columns if col not in existing_slm]
        return df.loc[:, first + existing_slm]

    before = []
    after = []
    seen_stec = False
    for col in df.columns:
        if col in existing_slm:
            continue
        if col == stec_column:
            seen_stec = True
            continue
        if seen_stec:
            after.append(col)
        else:
            before.append(col)
    return df.loc[:, before + [stec_column] + existing_slm + after]


def build_metadata(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    *,
    stec_column: str,
    vtec_column: str,
    mapping_column: str,
    shell_height_km: float,
    use_existing_mapping: bool,
    dropped_missing: int,
) -> dict[str, Any]:
    """Build a JSON-serializable metadata record."""

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "metadata_file": str(metadata_path),
        "row_counts": {
            "input": int(len(df_in)),
            "output": int(len(df_out)),
            "dropped_missing_inputs": int(dropped_missing),
        },
        "single_layer_conversion": {
            "formula": f"{vtec_column} = {stec_column} / {mapping_column}",
            "mapping_function": "1 / sqrt(1 - (Re / (Re + H) * cos(E))^2)",
            "earth_radius_km": 6371.0,
            "shell_height_km": float(shell_height_km),
            "stec_column": stec_column,
            "vtec_column": vtec_column,
            "mapping_column": mapping_column,
            "use_existing_mapping": bool(use_existing_mapping),
            "source_column": "vtec_slm_source",
        },
        "stats": {
            stec_column: _numeric_stats(df_out[stec_column]),
            mapping_column: _numeric_stats(df_out[mapping_column]),
            vtec_column: _numeric_stats(df_out[vtec_column]),
        },
    }


def build_slm_vtec(
    input_path: str | Path,
    output_path: str | Path,
    *,
    stec_column: str = "stec_obs",
    vtec_column: str = "vtec_slm",
    mapping_column: str = "slm_mapping_factor",
    elevation_column: str = "elevation",
    shell_height_km: float = 450.0,
    use_existing_mapping: bool = False,
    drop_missing: bool = False,
    metadata_path: str | Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Read ray-wise STEC, compute SLM VTEC, and write table plus metadata."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path) if metadata_path else output_path.with_suffix(
        ".metadata.json"
    )

    df_in = read_table(input_path)
    df = df_in.head(max_rows).copy() if max_rows else df_in.copy()

    required = [stec_column]
    if not use_existing_mapping or mapping_column not in df.columns:
        required.append(elevation_column)
    else:
        required.append(mapping_column)

    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Input table is missing required columns: {missing_columns}")

    missing_inputs = df[required].isna().any(axis=1)
    n_missing = int(missing_inputs.sum())
    if n_missing:
        if not drop_missing:
            raise ValueError(
                f"{n_missing} rows have missing SLM conversion inputs. Use "
                "--drop-missing to write only valid rows."
            )
        df = df.loc[~missing_inputs].copy()

    if df.empty:
        raise ValueError("No rows remain after SLM conversion input filtering.")

    df = add_slm_vtec_from_stec(
        df,
        stec_column=stec_column,
        output_column=vtec_column,
        mapping_column=mapping_column,
        elevation_column=elevation_column,
        shell_height_km=shell_height_km,
        use_existing_mapping=use_existing_mapping,
    )
    df["vtec_slm_source"] = "observed_stec_single_layer_mapping"
    df = _order_columns(
        df,
        stec_column=stec_column,
        vtec_column=vtec_column,
        mapping_column=mapping_column,
    )
    saved_path = save_table(df, output_path)

    metadata = build_metadata(
        input_path=input_path,
        output_path=saved_path,
        metadata_path=metadata_path,
        df_in=df_in,
        df_out=df,
        stec_column=stec_column,
        vtec_column=vtec_column,
        mapping_column=mapping_column,
        shell_height_km=shell_height_km,
        use_existing_mapping=use_existing_mapping,
        dropped_missing=n_missing,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input parquet/CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output parquet/CSV.")
    parser.add_argument(
        "--metadata-output",
        help="Output metadata JSON. Defaults to <output>.metadata.json.",
    )
    parser.add_argument("--stec-column", default="stec_obs")
    parser.add_argument("--vtec-column", default="vtec_slm")
    parser.add_argument("--mapping-column", default="slm_mapping_factor")
    parser.add_argument("--elevation-column", default="elevation")
    parser.add_argument("--shell-height-km", type=float, default=450.0)
    parser.add_argument(
        "--use-existing-mapping",
        action="store_true",
        help="Use --mapping-column if it is already present in the input table.",
    )
    parser.add_argument("--drop-missing", action="store_true")
    parser.add_argument("--max-rows", type=int, help="Optional smoke-test row cap.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = build_slm_vtec(
        input_path=args.input,
        output_path=args.output,
        metadata_path=args.metadata_output,
        stec_column=args.stec_column,
        vtec_column=args.vtec_column,
        mapping_column=args.mapping_column,
        elevation_column=args.elevation_column,
        shell_height_km=args.shell_height_km,
        use_existing_mapping=args.use_existing_mapping,
        drop_missing=args.drop_missing,
        max_rows=args.max_rows,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
