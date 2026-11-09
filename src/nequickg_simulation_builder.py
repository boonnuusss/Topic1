"""Build a ray-wise NeQuick-G STEC/VTEC simulation table."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer

from .geometry import compute_slant_factor


DEFAULT_INPUT = (
    "datasets/interim/"
    "iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet"
)
DEFAULT_OUTPUT = (
    "datasets/interim/"
    "nequickg_simulation_20240505_0000_2400_5min.parquet"
)
DEFAULT_NAVIGATION = (
    "datasets/raw/navigation/BRDC00IGS_R_20241260000_01D_MN.rnx.gz"
)
DEFAULT_NAVIGATION_URL = (
    "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2024/126/"
    "BRDC00IGS_R_20241260000_01D_MN.rnx.gz"
)


@dataclass(frozen=True)
class GalileoIonoCoefficients:
    a0: float
    a1: float
    a2: float
    raw_header_line: str


REFERENCE_COLUMNS = {
    "simulation_id",
    "simulation_role",
    "stec_iri_ray",
    "stec_iri_observable",
    "vtec_iri_ref",
    "vtec_iri_ref_method",
    "stec_iri_method",
    "slm_shell_height_km",
    "slm_mapping_factor_iri",
    "vtec_slm_from_iri_stec",
    "vtec_slm_error_iri",
    "vtec_slm_abs_error_iri",
    "bias_mode",
    "receiver_bias_tecu",
    "satellite_bias_tecu",
    "noise_tecu",
    "ray_integral_n_samples",
    "ray_integral_min_alt_km",
    "ray_integral_max_alt_km",
    "ray_integral_path_length_km",
    "ray_integral_outside_grid_fraction",
}


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_galileo_iono_coefficients(
    path: str | Path,
) -> GalileoIonoCoefficients:
    """Parse the unique Galileo ai0/ai1/ai2 record from a RINEX header."""

    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    records: list[GalileoIonoCoefficients] = []
    float_pattern = re.compile(r"[-+]?\d*\.\d+(?:[DEde][-+]?\d+)?")
    with opener(path, "rt", encoding="ascii", errors="strict") as handle:
        for line in handle:
            if "END OF HEADER" in line:
                break
            if not (line.startswith("GAL") and "IONOSPHERIC CORR" in line):
                continue
            values = [
                float(value.replace("D", "E").replace("d", "e"))
                for value in float_pattern.findall(line[:60])
            ]
            if len(values) < 3:
                raise ValueError(f"Malformed Galileo ionosphere header: {line!r}")
            records.append(
                GalileoIonoCoefficients(
                    a0=values[0],
                    a1=values[1],
                    a2=values[2],
                    raw_header_line=line.rstrip("\r\n"),
                )
            )
    unique = {(item.a0, item.a1, item.a2) for item in records}
    if len(unique) != 1:
        raise ValueError(
            "Expected one unique GAL IONOSPHERIC CORR coefficient set, "
            f"found {sorted(unique)}."
        )
    return records[0]


def strip_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove IRI target columns while preserving identical ray geometry."""

    columns = [column for column in df.columns if column not in REFERENCE_COLUMNS]
    return df.loc[:, columns].copy()


def _altitude_at_alpha(
    rec_xyz: np.ndarray,
    direction: np.ndarray,
    alpha: np.ndarray,
    transformer: Transformer,
) -> np.ndarray:
    points = rec_xyz + alpha[:, None] * direction
    if len(points) == 1:
        _, _, altitude_m = transformer.transform(
            float(points[0, 0]), float(points[0, 1]), float(points[0, 2])
        )
        return np.asarray([altitude_m], dtype=np.float64)
    _, _, altitude_m = transformer.transform(
        points[:, 0], points[:, 1], points[:, 2]
    )
    return np.asarray(altitude_m, dtype=np.float64)


def ray_points_at_altitudes(
    rec_xyz: np.ndarray,
    sat_xyz: np.ndarray,
    target_altitudes_m: tuple[float, float],
    *,
    iterations: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect visible receiver-satellite rays with geodetic altitude shells."""

    rec_xyz = np.asarray(rec_xyz, dtype=np.float64)
    sat_xyz = np.asarray(sat_xyz, dtype=np.float64)
    if rec_xyz.shape != sat_xyz.shape or rec_xyz.ndim != 2 or rec_xyz.shape[1] != 3:
        raise ValueError("rec_xyz and sat_xyz must both have shape (n, 3).")
    direction = sat_xyz - rec_xyz
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    outputs: list[np.ndarray] = []
    n_rows = len(rec_xyz)
    for target in target_altitudes_m:
        lower = np.zeros(n_rows, dtype=np.float64)
        upper = np.ones(n_rows, dtype=np.float64)
        alt_lower = _altitude_at_alpha(rec_xyz, direction, lower, transformer)
        alt_upper = _altitude_at_alpha(rec_xyz, direction, upper, transformer)
        if np.any(alt_lower > target) or np.any(alt_upper < target):
            raise ValueError(
                f"Ray endpoints do not bracket target altitude {target} m."
            )
        for _ in range(iterations):
            middle = 0.5 * (lower + upper)
            altitude = _altitude_at_alpha(rec_xyz, direction, middle, transformer)
            below = altitude < target
            lower = np.where(below, middle, lower)
            upper = np.where(below, upper, middle)
        alpha = 0.5 * (lower + upper)
        points = rec_xyz + alpha[:, None] * direction
        if len(points) == 1:
            lon_value, lat_value, altitude_value = transformer.transform(
                float(points[0, 0]),
                float(points[0, 1]),
                float(points[0, 2]),
            )
            lon = np.asarray([lon_value])
            lat = np.asarray([lat_value])
            altitude = np.asarray([altitude_value])
        else:
            lon, lat, altitude = transformer.transform(
                points[:, 0], points[:, 1], points[:, 2]
            )
        outputs.append(np.column_stack((lon, lat, altitude)))
    return outputs[0], outputs[1]


def _part_path(parts_dir: Path, timestamp: pd.Timestamp) -> Path:
    return parts_dir / f"epoch_{timestamp:%Y%m%dT%H%M%S}.parquet"


def _part_is_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        return len(pd.read_parquet(path, columns=["timestamp"])) == expected_rows
    except Exception:
        return False


def _process_epoch_task(task: dict[str, Any]) -> dict[str, Any]:
    """Compute and atomically save one epoch; designed for spawned workers."""

    from nequick import NeQuick

    frame: pd.DataFrame = task["frame"]
    timestamp = pd.Timestamp(task["timestamp"])
    epoch = timestamp.to_pydatetime().replace(tzinfo=None)
    lower_altitude_m = float(task["lower_altitude_m"])
    upper_altitude_m = float(task["upper_altitude_m"])
    coefficients = task["coefficients"]
    model = NeQuick(
        coefficients["a0"], coefficients["a1"], coefficients["a2"]
    )

    rec_xyz = frame[["rec_x", "rec_y", "rec_z"]].to_numpy(float)
    sat_xyz = frame[["sat_x", "sat_y", "sat_z"]].to_numpy(float)
    lower, upper = ray_points_at_altitudes(
        rec_xyz,
        sat_xyz,
        (lower_altitude_m, upper_altitude_m),
    )
    stec = np.empty(len(frame), dtype=np.float64)
    vtec = np.empty(len(frame), dtype=np.float64)
    ipp_lon = frame["ipp_lon"].to_numpy(float)
    ipp_lat = frame["ipp_lat"].to_numpy(float)
    for index in range(len(frame)):
        # The PyPI extension uses longitude before latitude despite ambiguous docs.
        stec[index] = model.compute_stec(
            epoch,
            float(lower[index, 0]),
            float(lower[index, 1]),
            lower_altitude_m,
            float(upper[index, 0]),
            float(upper[index, 1]),
            upper_altitude_m,
        )
        vtec[index] = model.compute_stec(
            epoch,
            float(ipp_lon[index]),
            float(ipp_lat[index]),
            lower_altitude_m,
            float(ipp_lon[index]),
            float(ipp_lat[index]),
            upper_altitude_m,
        )

    out = strip_reference_columns(frame)
    out.insert(1, "simulation_id", task["simulation_id"])
    out.insert(2, "simulation_role", "nequickg_reference")
    out.insert(4, "stec_nequickg_ray", stec)
    out.insert(5, "stec_nequickg_observable", stec)
    out.insert(6, "vtec_nequickg_ref", vtec)
    out["stec_nequickg_method"] = "nequickg_gauss_kronrod_clipped_ray"
    out["vtec_nequickg_ref_method"] = "nequickg_gauss_kronrod_ipp_vertical"
    out["nequickg_lower_altitude_km"] = lower_altitude_m / 1000.0
    out["nequickg_upper_altitude_km"] = upper_altitude_m / 1000.0
    out["slm_shell_height_km"] = float(task["slm_shell_height_km"])
    out["slm_mapping_factor_nequickg"] = compute_slant_factor(
        out["elevation"].to_numpy(float),
        shell_height_km=float(task["slm_shell_height_km"]),
    )
    out["vtec_slm_from_nequickg_stec"] = (
        out["stec_nequickg_observable"] / out["slm_mapping_factor_nequickg"]
    )
    out["vtec_slm_error_nequickg"] = (
        out["vtec_slm_from_nequickg_stec"] - out["vtec_nequickg_ref"]
    )
    out["vtec_slm_abs_error_nequickg"] = out[
        "vtec_slm_error_nequickg"
    ].abs()
    out["bias_mode"] = "none"
    out["receiver_bias_tecu"] = 0.0
    out["satellite_bias_tecu"] = 0.0
    out["noise_tecu"] = 0.0

    part_path = Path(task["part_path"])
    part_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = part_path.with_suffix(".tmp.parquet")
    out.to_parquet(temporary_path, index=False)
    temporary_path.replace(part_path)
    return {
        "timestamp": timestamp.isoformat(),
        "rows": int(len(out)),
        "part_path": str(part_path),
        "stec_mean_tecu": float(np.mean(stec)),
        "vtec_mean_tecu": float(np.mean(vtec)),
    }


def _json_safe_stats(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric[np.isfinite(numeric)]
    return {
        "count": int(len(valid)),
        "missing": int(len(numeric) - len(valid)),
        "min": float(valid.min()),
        "mean": float(valid.mean()),
        "max": float(valid.max()),
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    navigation_path = Path(args.navigation_file)
    metadata_path = Path(args.metadata_output) if args.metadata_output else (
        output_path.with_suffix(".metadata.json")
    )
    parts_dir = Path(args.parts_dir) if args.parts_dir else (
        output_path.parent / f"{output_path.stem}_parts"
    )
    coefficients = parse_galileo_iono_coefficients(navigation_path)

    frame = read_table(input_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if args.start_time:
        frame = frame.loc[frame["timestamp"] >= pd.Timestamp(args.start_time)]
    if args.end_time:
        frame = frame.loc[frame["timestamp"] < pd.Timestamp(args.end_time)]
    if args.max_rows:
        frame = frame.head(args.max_rows)
    epochs = sorted(frame["timestamp"].dropna().unique())
    if args.max_epochs:
        epochs = epochs[: args.max_epochs]
        frame = frame.loc[frame["timestamp"].isin(epochs)]
    if frame.empty:
        raise ValueError("No input rays remain after filtering.")

    required = [
        "timestamp",
        "station",
        "prn",
        "rec_x",
        "rec_y",
        "rec_z",
        "sat_x",
        "sat_y",
        "sat_z",
        "ipp_lon",
        "ipp_lat",
        "elevation",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Input is missing required ray columns: {missing}")

    workers = args.workers or max(1, (os.cpu_count() or 2) // 2)
    tasks = []
    completed_stats: list[dict[str, Any]] = []
    for epoch_value in epochs:
        timestamp = pd.Timestamp(epoch_value)
        epoch_frame = frame.loc[frame["timestamp"] == timestamp].copy()
        part_path = _part_path(parts_dir, timestamp)
        if not args.overwrite_parts and _part_is_complete(
            part_path, len(epoch_frame)
        ):
            completed_stats.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "rows": int(len(epoch_frame)),
                    "part_path": str(part_path),
                    "status": "reused",
                }
            )
            continue
        tasks.append(
            {
                "frame": epoch_frame,
                "timestamp": timestamp.isoformat(),
                "part_path": str(part_path),
                "coefficients": asdict(coefficients),
                "lower_altitude_m": args.lower_altitude_km * 1000.0,
                "upper_altitude_m": args.upper_altitude_km * 1000.0,
                "slm_shell_height_km": args.slm_shell_height_km,
                "simulation_id": args.simulation_id,
            }
        )

    if tasks and workers == 1:
        for index, task in enumerate(tasks, start=1):
            completed_stats.append(_process_epoch_task(task))
            print(f"NeQuick-G epoch {index}/{len(tasks)} complete", flush=True)
    elif tasks:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_epoch_task, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                completed_stats.append(future.result())
                print(f"NeQuick-G epoch {index}/{len(tasks)} complete", flush=True)

    selected_parts = [_part_path(parts_dir, pd.Timestamp(epoch)) for epoch in epochs]
    missing_parts = [str(path) for path in selected_parts if not path.exists()]
    if missing_parts:
        raise RuntimeError(f"Missing completed epoch parts: {missing_parts[:5]}")
    output = pd.concat(
        [pd.read_parquet(path) for path in selected_parts], ignore_index=True
    )
    output = output.sort_values(["timestamp", "station", "prn"]).reset_index(
        drop=True
    )
    save_table(output, output_path)

    import nequick
    import nequick._c_ext
    import numpy

    metadata = {
        "simulation_id": args.simulation_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "parts_dir": str(parts_dir),
        "rows": int(len(output)),
        "epochs": int(len(epochs)),
        "timestamp_min": pd.Timestamp(output["timestamp"].min()).isoformat(),
        "timestamp_max": pd.Timestamp(output["timestamp"].max()).isoformat(),
        "workers": int(workers),
        "nequickg": {
            "package": "nequick",
            "package_version": nequick.__version__,
            "extension_path": str(Path(nequick._c_ext.__file__).resolve()),
            "python_version": sys.version,
            "numpy_version": numpy.__version__,
            "model": "JRC NeQuick-G",
            "integration": "built-in Gauss-Kronrod",
            "coordinate_argument_order": "longitude, latitude, altitude_m",
            "coefficients": asdict(coefficients),
        },
        "navigation": {
            "file": str(navigation_path),
            "url": args.navigation_url,
            "sha256": sha256_file(navigation_path),
        },
        "altitude_bounds_km": [
            float(args.lower_altitude_km),
            float(args.upper_altitude_km),
        ],
        "slm_shell_height_km": float(args.slm_shell_height_km),
        "epoch_stats": sorted(completed_stats, key=lambda item: item["timestamp"]),
        "stats": {
            column: _json_safe_stats(output[column])
            for column in [
                "stec_nequickg_ray",
                "vtec_nequickg_ref",
                "vtec_slm_from_nequickg_stec",
                "vtec_slm_error_nequickg",
            ]
        },
        "scientific_note": (
            "Ray keys and altitude bounds match the IRI experiment, but "
            "NeQuick-G uses its built-in Gauss-Kronrod integration rather "
            "than the IRI regular-grid trapezoidal integration."
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output")
    parser.add_argument("--parts-dir")
    parser.add_argument("--navigation-file", default=DEFAULT_NAVIGATION)
    parser.add_argument("--navigation-url", default=DEFAULT_NAVIGATION_URL)
    parser.add_argument(
        "--simulation-id", default="nequickg_20240505_full_day_5min_v001"
    )
    parser.add_argument("--lower-altitude-km", type=float, default=90.0)
    parser.add_argument("--upper-altitude-km", type=float, default=2000.0)
    parser.add_argument("--slm-shell-height-km", type=float, default=450.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--overwrite-parts", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    metadata = build_dataset(parse_args(argv))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
