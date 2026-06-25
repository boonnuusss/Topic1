"""Build an IRI-backed STEC-to-VTEC simulation table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

try:
    from .geometry import compute_slant_factor
except ImportError:
    from geometry import compute_slant_factor


DEFAULT_INPUT = "datasets/processed/raystec_china_20240505.parquet"
DEFAULT_OUTPUT = "datasets/processed/iri_simulation_rayvtec_20240505.parquet"
TECU_SCALE = 1.0e16


@dataclass(frozen=True)
class IRIGridConfig:
    """IRI electron-density grid settings."""

    lon_min: float = 60.0
    lon_max: float = 160.0
    dlon: float = 1.0
    lat_min: float = -20.0
    lat_max: float = 80.0
    dlat: float = 0.5
    alt_min_km: float = 90.0
    alt_max_km: float = 2000.0
    dalt_km: float = 5.0

    def altitude_array(self) -> np.ndarray:
        return np.arange(
            self.alt_min_km,
            self.alt_max_km + 0.5 * self.dalt_km,
            self.dalt_km,
            dtype=float,
        )

    def longitude_array(self) -> np.ndarray:
        return np.arange(
            self.lon_min,
            self.lon_max + 0.5 * self.dlon,
            self.dlon,
            dtype=float,
        )

    def latitude_array(self) -> np.ndarray:
        return np.arange(
            self.lat_min,
            self.lat_max + 0.5 * self.dlat,
            self.dlat,
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "dlon": self.dlon,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "dlat": self.dlat,
            "alt_min_km": self.alt_min_km,
            "alt_max_km": self.alt_max_km,
            "dalt_km": self.dalt_km,
        }


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


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write JSON metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


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


def resolve_pyiri_path(pyiri_path: str | Path | None) -> Path:
    """Resolve the vendored PyIRI package path."""

    if pyiri_path:
        path = Path(pyiri_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path
    return Path(__file__).resolve().parents[2] / "00_iri" / "PyIRI"


def load_pyiri(pyiri_path: str | Path | None = None):
    """Import vendored PyIRI and return package plus density module."""

    path = resolve_pyiri_path(pyiri_path)
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    try:
        import PyIRI
        import PyIRI.edp_update as iri_density
    except ImportError as exc:
        raise RuntimeError(
            "PyIRI is required. Provide --pyiri-path pointing to the vendored "
            "00_iri/PyIRI directory or install PyIRI in the environment."
        ) from exc
    return PyIRI, iri_density, path


def validate_input(
    df: pd.DataFrame,
    *,
    min_elevation_deg: float,
    drop_invalid_geometry: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate required columns and filter invalid rows."""

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
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input table is missing required columns: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    numeric_columns = [
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
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    finite_geometry = np.isfinite(out[numeric_columns].to_numpy(float)).all(axis=1)
    elevation_ok = out["elevation"].to_numpy(float) >= min_elevation_deg
    valid = finite_geometry & elevation_ok
    counts = {
        "dropped_missing_geometry": int((~finite_geometry).sum()),
        "dropped_below_min_elevation": int((finite_geometry & ~elevation_ok).sum()),
    }
    if not valid.all():
        n_invalid = int((~valid).sum())
        if not drop_invalid_geometry:
            raise ValueError(
                f"{n_invalid} rows have invalid geometry or elevation below "
                f"{min_elevation_deg:g} deg. Use --drop-invalid-geometry to "
                "write only valid rows."
            )
        out = out.loc[valid].copy()

    if out.empty:
        raise ValueError("No rows remain after IRI simulation input filtering.")
    return out, counts


def decimal_utc_hour(ts: pd.Timestamp) -> float:
    """Return decimal UTC hour for a timestamp."""

    return (
        ts.hour
        + ts.minute / 60.0
        + ts.second / 3600.0
        + ts.microsecond / 3.6e9
    )


def sample_ray_ecef(
    rec_xyz: np.ndarray,
    sat_xyz: np.ndarray,
    *,
    sample_spacing_km: float,
    sample_alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample an ECEF receiver-satellite segment."""

    rel = sat_xyz.astype(float) - rec_xyz.astype(float)
    path_length_m = float(np.linalg.norm(rel))
    path_length_km = path_length_m / 1000.0
    if path_length_km <= 0:
        raise ValueError("Receiver-satellite path length must be positive.")
    n_segments = max(1, int(np.ceil(path_length_km / sample_spacing_km)))
    u = np.linspace(0.0, 1.0, n_segments + 1)
    if sample_alpha != 1.0:
        u = u**float(sample_alpha)
        u[0] = 0.0
        u[-1] = 1.0
    ecef = rec_xyz.astype(float)[None, :] + rel[None, :] * u[:, None]
    seg_lengths_m = np.linalg.norm(np.diff(ecef, axis=0), axis=1)
    return ecef, seg_lengths_m, path_length_km


def ecef_to_geodetic(ecef: np.ndarray, transformer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF sample points to lon/lat/height km."""

    lon, lat, h_m = transformer.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    return np.asarray(lon), np.asarray(lat), np.asarray(h_m) / 1000.0


def interpolate_vertical_profiles(
    profiles: np.ndarray,
    alt_grid_km: np.ndarray,
    sample_alt_km: np.ndarray,
) -> np.ndarray:
    """Interpolate vertical electron-density profiles at sample altitudes."""

    if profiles.shape[1] != sample_alt_km.size:
        raise ValueError("Profile count and altitude count do not match.")
    idx = np.searchsorted(alt_grid_km, sample_alt_km, side="right") - 1
    idx = np.clip(idx, 0, len(alt_grid_km) - 2)
    alt0 = alt_grid_km[idx]
    alt1 = alt_grid_km[idx + 1]
    weight = (sample_alt_km - alt0) / np.maximum(alt1 - alt0, 1e-12)
    cols = np.arange(sample_alt_km.size)
    ne0 = profiles[idx, cols]
    ne1 = profiles[idx + 1, cols]
    return ne0 * (1.0 - weight) + ne1 * weight


def trapezoid_vtec(profiles: np.ndarray, alt_grid_km: np.ndarray) -> np.ndarray:
    """Integrate vertical electron-density profiles to VTEC."""

    try:
        integral = np.trapezoid(profiles, x=alt_grid_km * 1000.0, axis=0)
    except AttributeError:
        integral = np.trapz(profiles, x=alt_grid_km * 1000.0, axis=0)
    return integral / TECU_SCALE


def run_pyiri_density(
    iri_density,
    PyIRI,
    *,
    timestamp: pd.Timestamp,
    lon: np.ndarray,
    lat: np.ndarray,
    alt_grid_km: np.ndarray,
    f107: float,
    ccir_or_ursi: int,
):
    """Run PyIRI for one timestamp and a set of horizontal points."""

    ahr = np.array([decimal_utc_hour(timestamp)], dtype=float)
    _, _, _, _, _, _, edp = iri_density.IRI_density_1day(
        int(timestamp.year),
        int(timestamp.month),
        int(timestamp.day),
        ahr,
        lon.astype(float),
        lat.astype(float),
        alt_grid_km.astype(float),
        float(f107),
        PyIRI.coeff_dir,
        int(ccir_or_ursi),
    )
    return edp[0]


def build_epoch_simulation(
    epoch_df: pd.DataFrame,
    *,
    timestamp: pd.Timestamp,
    grid: IRIGridConfig,
    iri_density,
    PyIRI,
    ecef2geo,
    f107: float,
    ccir_or_ursi: int,
    ray_sample_spacing_km: float,
    sample_alpha: float,
    max_iri_points_per_epoch: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build simulation columns for one timestamp."""

    alt_grid_km = grid.altitude_array()
    n_rows = len(epoch_df)
    rec_xyz = epoch_df[["rec_x", "rec_y", "rec_z"]].to_numpy(float)
    sat_xyz = epoch_df[["sat_x", "sat_y", "sat_z"]].to_numpy(float)

    ray_ecef_list = []
    ray_seg_list = []
    ray_lon_list = []
    ray_lat_list = []
    ray_alt_list = []
    inside_masks = []
    path_lengths_km = np.zeros(n_rows, dtype=float)
    min_alt_km = np.zeros(n_rows, dtype=float)
    max_alt_km = np.zeros(n_rows, dtype=float)
    outside_fraction = np.zeros(n_rows, dtype=float)
    n_samples = np.zeros(n_rows, dtype=int)

    query_lon_parts = [epoch_df["ipp_lon"].to_numpy(float)]
    query_lat_parts = [epoch_df["ipp_lat"].to_numpy(float)]

    for i in range(n_rows):
        ecef, seg_lengths_m, path_length_km = sample_ray_ecef(
            rec_xyz[i],
            sat_xyz[i],
            sample_spacing_km=ray_sample_spacing_km,
            sample_alpha=sample_alpha,
        )
        lon, lat, alt_km = ecef_to_geodetic(ecef, ecef2geo)
        inside = (
            (alt_km >= grid.alt_min_km)
            & (alt_km <= grid.alt_max_km)
            & (lon >= grid.lon_min)
            & (lon <= grid.lon_max)
            & (lat >= grid.lat_min)
            & (lat <= grid.lat_max)
        )

        ray_ecef_list.append(ecef)
        ray_seg_list.append(seg_lengths_m)
        ray_lon_list.append(lon)
        ray_lat_list.append(lat)
        ray_alt_list.append(alt_km)
        inside_masks.append(inside)
        path_lengths_km[i] = path_length_km
        min_alt_km[i] = float(np.nanmin(alt_km))
        max_alt_km[i] = float(np.nanmax(alt_km))
        outside_fraction[i] = 1.0 - float(inside.sum()) / float(len(inside))
        n_samples[i] = int(len(alt_km))

        if inside.any():
            query_lon_parts.append(lon[inside])
            query_lat_parts.append(lat[inside])

    query_lon = np.concatenate(query_lon_parts)
    query_lat = np.concatenate(query_lat_parts)
    if query_lon.size > max_iri_points_per_epoch:
        raise ValueError(
            f"Epoch {timestamp} requires {query_lon.size} IRI horizontal points, "
            f"above --max-iri-points-per-epoch={max_iri_points_per_epoch}. "
            "Use fewer rows/epochs, increase --ray-sample-spacing-km, or raise "
            "the limit intentionally."
        )

    edp_profiles = run_pyiri_density(
        iri_density,
        PyIRI,
        timestamp=timestamp,
        lon=query_lon,
        lat=query_lat,
        alt_grid_km=alt_grid_km,
        f107=f107,
        ccir_or_ursi=ccir_or_ursi,
    )
    vtec_ref = trapezoid_vtec(edp_profiles[:, :n_rows], alt_grid_km)

    stec_iri = np.zeros(n_rows, dtype=float)
    cursor = n_rows
    for i in range(n_rows):
        ne_all = np.zeros(n_samples[i], dtype=float)
        inside = inside_masks[i]
        n_inside = int(inside.sum())
        if n_inside:
            profiles = edp_profiles[:, cursor : cursor + n_inside]
            ne_inside = interpolate_vertical_profiles(
                profiles,
                alt_grid_km,
                ray_alt_list[i][inside],
            )
            ne_all[inside] = ne_inside
            cursor += n_inside
        integral_el_m2 = np.sum(
            0.5 * (ne_all[:-1] + ne_all[1:]) * ray_seg_list[i]
        )
        stec_iri[i] = integral_el_m2 / TECU_SCALE

    out = epoch_df.copy()
    out["stec_iri_ray"] = stec_iri
    out["vtec_iri_ref"] = vtec_ref
    out["vtec_iri_ref_method"] = "vertical_ipp_trapezoid"
    out["stec_iri_method"] = "ecef_ray_trapezoid_point_profile"
    out["ray_integral_n_samples"] = n_samples
    out["ray_integral_min_alt_km"] = min_alt_km
    out["ray_integral_max_alt_km"] = max_alt_km
    out["ray_integral_path_length_km"] = path_lengths_km
    out["ray_integral_outside_grid_fraction"] = outside_fraction

    epoch_metadata = {
        "timestamp": timestamp.isoformat(),
        "rows": int(n_rows),
        "iri_horizontal_points": int(query_lon.size),
        "ray_sample_points": int(n_samples.sum()),
        "inside_grid_sample_points": int(query_lon.size - n_rows),
    }
    return out, epoch_metadata


def build_epoch_simulation_regular_grid(
    epoch_df: pd.DataFrame,
    *,
    timestamp: pd.Timestamp,
    grid: IRIGridConfig,
    iri_density,
    PyIRI,
    ecef2geo,
    f107: float,
    ccir_or_ursi: int,
    ray_sample_spacing_km: float,
    sample_alpha: float,
    max_iri_points_per_epoch: int,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one timestamp by interpolating a regular IRI grid.

    This mirrors the legacy notebooks that use scipy RegularGridInterpolator.
    It is much more memory-friendly for dense ray sampling because PyIRI is
    evaluated once on a regional lon-lat-alt grid rather than once per ray
    sample point.
    """

    from scipy.interpolate import RegularGridInterpolator

    alt_grid_km = grid.altitude_array()
    lon_vals = grid.longitude_array()
    lat_vals = grid.latitude_array()
    lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals, indexing="ij")
    query_lon = lon_2d.reshape(-1)
    query_lat = lat_2d.reshape(-1)
    if query_lon.size > max_iri_points_per_epoch:
        raise ValueError(
            f"Regular grid for epoch {timestamp} requires {query_lon.size} "
            "IRI horizontal points, above "
            f"--max-iri-points-per-epoch={max_iri_points_per_epoch}."
        )

    edp_profiles = run_pyiri_density(
        iri_density,
        PyIRI,
        timestamp=timestamp,
        lon=query_lon,
        lat=query_lat,
        alt_grid_km=alt_grid_km,
        f107=f107,
        ccir_or_ursi=ccir_or_ursi,
    )
    edp_3d = edp_profiles.reshape(
        len(alt_grid_km),
        len(lon_vals),
        len(lat_vals),
    )
    density_interp = RegularGridInterpolator(
        (alt_grid_km, lon_vals, lat_vals),
        edp_3d,
        bounds_error=False,
        fill_value=0.0,
    )
    try:
        vtec_grid = (
            np.trapezoid(edp_3d, x=alt_grid_km * 1000.0, axis=0)
            / TECU_SCALE
        )
    except AttributeError:
        vtec_grid = np.trapz(edp_3d, x=alt_grid_km * 1000.0, axis=0) / TECU_SCALE
    vtec_interp = RegularGridInterpolator(
        (lon_vals, lat_vals),
        vtec_grid,
        bounds_error=False,
        fill_value=np.nan,
    )

    n_rows = len(epoch_df)
    rec_xyz = epoch_df[["rec_x", "rec_y", "rec_z"]].to_numpy(float)
    sat_xyz = epoch_df[["sat_x", "sat_y", "sat_z"]].to_numpy(float)
    ipp_points = epoch_df[["ipp_lon", "ipp_lat"]].to_numpy(float)

    stec_iri = np.zeros(n_rows, dtype=float)
    path_lengths_km = np.zeros(n_rows, dtype=float)
    min_alt_km = np.zeros(n_rows, dtype=float)
    max_alt_km = np.zeros(n_rows, dtype=float)
    outside_fraction = np.zeros(n_rows, dtype=float)
    n_samples = np.zeros(n_rows, dtype=int)
    inside_grid_samples = 0

    for i in range(n_rows):
        ecef, seg_lengths_m, path_length_km = sample_ray_ecef(
            rec_xyz[i],
            sat_xyz[i],
            sample_spacing_km=ray_sample_spacing_km,
            sample_alpha=sample_alpha,
        )
        lon, lat, alt_km = ecef_to_geodetic(ecef, ecef2geo)
        inside = (
            (alt_km >= grid.alt_min_km)
            & (alt_km <= grid.alt_max_km)
            & (lon >= grid.lon_min)
            & (lon <= grid.lon_max)
            & (lat >= grid.lat_min)
            & (lat <= grid.lat_max)
        )
        ne_all = density_interp(np.column_stack((alt_km, lon, lat)))
        integral_el_m2 = np.sum(
            0.5 * (ne_all[:-1] + ne_all[1:]) * seg_lengths_m
        )
        stec_iri[i] = integral_el_m2 / TECU_SCALE
        path_lengths_km[i] = path_length_km
        min_alt_km[i] = float(np.nanmin(alt_km))
        max_alt_km[i] = float(np.nanmax(alt_km))
        outside_fraction[i] = 1.0 - float(inside.sum()) / float(len(inside))
        n_samples[i] = int(len(alt_km))
        inside_grid_samples += int(inside.sum())

        if progress_every and (i + 1) % progress_every == 0:
            print(
                f"[{timestamp}] regular-grid ray {i + 1}/{n_rows} complete",
                flush=True,
            )

    out = epoch_df.copy()
    out["stec_iri_ray"] = stec_iri
    out["vtec_iri_ref"] = vtec_interp(ipp_points)
    out["vtec_iri_ref_method"] = "vertical_ipp_trapezoid_regular_grid_interp"
    out["stec_iri_method"] = "ecef_ray_trapezoid_regular_grid_interp"
    out["ray_integral_n_samples"] = n_samples
    out["ray_integral_min_alt_km"] = min_alt_km
    out["ray_integral_max_alt_km"] = max_alt_km
    out["ray_integral_path_length_km"] = path_lengths_km
    out["ray_integral_outside_grid_fraction"] = outside_fraction

    epoch_metadata = {
        "timestamp": timestamp.isoformat(),
        "rows": int(n_rows),
        "iri_horizontal_points": int(query_lon.size),
        "regular_grid_shape": [
            int(len(alt_grid_km)),
            int(len(lon_vals)),
            int(len(lat_vals)),
        ],
        "ray_sample_points": int(n_samples.sum()),
        "inside_grid_sample_points": int(inside_grid_samples),
    }
    return out, epoch_metadata


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place IRI simulation columns near observed STEC and VTEC references."""

    iri_columns = [
        "source_row_id",
        "simulation_id",
        "simulation_role",
        "stec_obs",
        "stec_iri_ray",
        "stec_iri_observable",
        "vtec_iri_ref",
        "slm_shell_height_km",
        "slm_mapping_factor_iri",
        "vtec_slm_from_iri_stec",
        "vtec_slm_error_iri",
        "vtec_slm_abs_error_iri",
        "vtec_iri_ref_method",
        "stec_iri_method",
        "bias_mode",
        "receiver_bias_tecu",
        "satellite_bias_tecu",
        "noise_tecu",
        "ray_integral_n_samples",
        "ray_integral_min_alt_km",
        "ray_integral_max_alt_km",
        "ray_integral_path_length_km",
        "ray_integral_outside_grid_fraction",
    ]
    first = [col for col in iri_columns if col in df.columns]
    rest = [col for col in df.columns if col not in first]
    return df.loc[:, first + rest]


def build_metadata(
    *,
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    pyiri_path: Path,
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    grid: IRIGridConfig,
    row_counts: dict[str, int],
    epoch_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build JSON metadata for the simulation run."""

    stats_columns = [
        "stec_iri_ray",
        "stec_iri_observable",
        "vtec_iri_ref",
        "vtec_slm_from_iri_stec",
        "vtec_slm_error_iri",
        "vtec_slm_abs_error_iri",
        "slm_mapping_factor_iri",
        "ray_integral_outside_grid_fraction",
    ]
    return {
        "simulation_id": args.simulation_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "metadata_file": str(metadata_path),
        "row_counts": {
            "input": int(len(df_in)),
            "output": int(len(df_out)),
            **row_counts,
        },
        "iri": {
            "engine": "PyIRI",
            "pyiri_path": str(pyiri_path),
            "call": "PyIRI.edp_update.IRI_density_1day",
            "ccir_or_ursi": int(args.ccir_or_ursi),
            "f107_source": "cli_config",
            "f107": float(args.f107),
            "grid_strategy": args.grid_strategy,
            "grid": grid.as_dict(),
        },
        "integration": {
            "ray_method": (
                "ecef_straight_line_trapezoid_regular_grid_interp"
                if args.grid_strategy == "regular_grid_interpolator"
                else "ecef_straight_line_trapezoid_point_profile"
            ),
            "vertical_method": (
                "ipp_vertical_trapezoid_regular_grid_interp"
                if args.grid_strategy == "regular_grid_interpolator"
                else "ipp_vertical_trapezoid"
            ),
            "ray_sample_spacing_km": float(args.ray_sample_spacing_km),
            "sample_alpha": float(args.sample_alpha),
            "outside_grid_fill_value": 0.0,
            "tecu_scale": TECU_SCALE,
            "max_iri_points_per_epoch": int(args.max_iri_points_per_epoch),
            "progress_every": int(args.progress_every),
        },
        "slm_baseline": {
            "formula": (
                "vtec_slm_from_iri_stec = stec_iri_observable / "
                "slm_mapping_factor_iri"
            ),
            "shell_height_km": float(args.slm_shell_height_km),
            "mapping_function": "1 / sqrt(1 - (Re / (Re + H) * cos(E))^2)",
            "earth_radius_km": 6371.0,
        },
        "bias": {
            "bias_mode": "none",
            "receiver_bias_tecu": 0.0,
            "satellite_bias_tecu": 0.0,
            "noise_tecu": 0.0,
            "random_seed": None,
        },
        "epoch_stats": epoch_stats,
        "stats": {
            column: _numeric_stats(df_out[column])
            for column in stats_columns
            if column in df_out.columns
        },
        "legacy_references": [
            "../00_iri/mycode/ML_sTEC/generate_sTEC_obs_iri_regional.ipynb",
            "../00_iri/mycode/ML_sTEC/gen_iri_val_dataset.ipynb",
        ],
    }


def build_iri_simulation(
    input_path: str | Path,
    output_path: str | Path,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build an IRI-backed ray-wise simulation table."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    metadata_path = Path(args.metadata_output) if args.metadata_output else (
        output_path.with_suffix(".metadata.json")
    )
    grid = IRIGridConfig(
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        dlon=args.dlon,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        dlat=args.dlat,
        alt_min_km=args.alt_min_km,
        alt_max_km=args.alt_max_km,
        dalt_km=args.dalt_km,
    )
    PyIRI, iri_density, pyiri_path = load_pyiri(args.pyiri_path)

    df_in = read_table(input_path)
    df = df_in.copy()
    df["source_row_id"] = np.arange(len(df), dtype=int)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    selected_after_row_cap = int(len(df))
    df, validation_counts = validate_input(
        df,
        min_elevation_deg=args.min_elevation_deg,
        drop_invalid_geometry=args.drop_invalid_geometry,
    )

    unique_epochs = sorted(pd.to_datetime(df["timestamp"]).dropna().unique())
    if args.max_epochs:
        unique_epochs = unique_epochs[: args.max_epochs]
        df = df.loc[df["timestamp"].isin(unique_epochs)].copy()
    selected_after_epoch_cap = int(len(df))
    if not unique_epochs:
        raise ValueError("No epochs remain for IRI simulation.")

    from pyproj import Transformer

    ecef2geo = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    epoch_outputs = []
    epoch_stats = []
    for epoch in unique_epochs:
        timestamp = pd.Timestamp(epoch)
        epoch_df = df.loc[df["timestamp"] == timestamp].copy()
        if args.grid_strategy == "regular_grid_interpolator":
            out_epoch, meta_epoch = build_epoch_simulation_regular_grid(
                epoch_df,
                timestamp=timestamp,
                grid=grid,
                iri_density=iri_density,
                PyIRI=PyIRI,
                ecef2geo=ecef2geo,
                f107=args.f107,
                ccir_or_ursi=args.ccir_or_ursi,
                ray_sample_spacing_km=args.ray_sample_spacing_km,
                sample_alpha=args.sample_alpha,
                max_iri_points_per_epoch=args.max_iri_points_per_epoch,
                progress_every=args.progress_every,
            )
        else:
            out_epoch, meta_epoch = build_epoch_simulation(
                epoch_df,
                timestamp=timestamp,
                grid=grid,
                iri_density=iri_density,
                PyIRI=PyIRI,
                ecef2geo=ecef2geo,
                f107=args.f107,
                ccir_or_ursi=args.ccir_or_ursi,
                ray_sample_spacing_km=args.ray_sample_spacing_km,
                sample_alpha=args.sample_alpha,
                max_iri_points_per_epoch=args.max_iri_points_per_epoch,
            )
        epoch_outputs.append(out_epoch)
        epoch_stats.append(meta_epoch)

    out = pd.concat(epoch_outputs, ignore_index=True)
    out["simulation_id"] = args.simulation_id
    out["simulation_role"] = "iri_reference"
    out["slm_shell_height_km"] = float(args.slm_shell_height_km)
    out["slm_mapping_factor_iri"] = compute_slant_factor(
        out["elevation"].to_numpy(float),
        shell_height_km=float(args.slm_shell_height_km),
    )
    out["bias_mode"] = "none"
    out["receiver_bias_tecu"] = 0.0
    out["satellite_bias_tecu"] = 0.0
    out["noise_tecu"] = 0.0
    out["stec_iri_observable"] = out["stec_iri_ray"]
    out["vtec_slm_from_iri_stec"] = (
        out["stec_iri_observable"] / out["slm_mapping_factor_iri"]
    )
    out["vtec_slm_error_iri"] = (
        out["vtec_slm_from_iri_stec"] - out["vtec_iri_ref"]
    )
    out["vtec_slm_abs_error_iri"] = out["vtec_slm_error_iri"].abs()
    out = _order_columns(out)

    saved_path = save_table(out, output_path)
    metadata = build_metadata(
        args=args,
        input_path=input_path,
        output_path=saved_path,
        metadata_path=metadata_path,
        pyiri_path=pyiri_path,
        df_in=df_in,
        df_out=out,
        grid=grid,
        row_counts={
            **validation_counts,
            "dropped_by_max_rows": int(len(df_in) - selected_after_row_cap),
            "dropped_by_max_epochs": int(
                selected_after_row_cap - selected_after_epoch_cap
            ),
        },
        epoch_stats=epoch_stats,
    )
    write_json(metadata, metadata_path)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input parquet/CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output parquet/CSV.")
    parser.add_argument(
        "--metadata-output",
        help="Output metadata JSON. Defaults to <output>.metadata.json.",
    )
    parser.add_argument("--simulation-id", default="iri_rayvtec_20240505_v001")
    parser.add_argument("--pyiri-path", help="Path to vendored 00_iri/PyIRI.")
    parser.add_argument("--f107", type=float, default=204.0)
    parser.add_argument("--ccir-or-ursi", type=int, choices=[0, 1], default=0)
    parser.add_argument("--lon-min", type=float, default=60.0)
    parser.add_argument("--lon-max", type=float, default=160.0)
    parser.add_argument("--dlon", type=float, default=1.0)
    parser.add_argument("--lat-min", type=float, default=-20.0)
    parser.add_argument("--lat-max", type=float, default=80.0)
    parser.add_argument("--dlat", type=float, default=0.5)
    parser.add_argument("--alt-min-km", type=float, default=90.0)
    parser.add_argument("--alt-max-km", type=float, default=2000.0)
    parser.add_argument("--dalt-km", type=float, default=5.0)
    parser.add_argument("--ray-sample-spacing-km", type=float, default=2.5)
    parser.add_argument("--sample-alpha", type=float, default=2.0)
    parser.add_argument(
        "--grid-strategy",
        choices=["row_adaptive_point_profiles", "regular_grid_interpolator"],
        default="row_adaptive_point_profiles",
        help=(
            "IRI query strategy. Use regular_grid_interpolator for dense "
            "ray sampling such as 2.5 km spacing."
        ),
    )
    parser.add_argument("--slm-shell-height-km", type=float, default=450.0)
    parser.add_argument("--min-elevation-deg", type=float, default=10.0)
    parser.add_argument("--drop-invalid-geometry", action="store_true")
    parser.add_argument("--max-rows", type=int, help="Optional smoke-test row cap.")
    parser.add_argument("--max-epochs", type=int, help="Optional epoch cap.")
    parser.add_argument(
        "--max-iri-points-per-epoch",
        type=int,
        default=200000,
        help="Safety limit for PyIRI horizontal query points per epoch.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N rays for long regular-grid runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = build_iri_simulation(
        input_path=args.input,
        output_path=args.output,
        args=args,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
