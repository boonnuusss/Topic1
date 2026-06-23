"""Build standard ray-wise STEC datasets from GNSS TEC and SP3 files."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .config import DatasetBuildConfig, iter_dates, iter_epochs, load_dataset_config
    from .features import add_constellation_onehot, add_time_features
    from .geometry import add_geometry_features
    from .io_mat import read_station_mat, select_epoch_observations, scan_station_files
    from .io_sp3 import attach_satellite_position, combine_sp3_files
except ImportError:
    from config import DatasetBuildConfig, iter_dates, iter_epochs, load_dataset_config
    from features import add_constellation_onehot, add_time_features
    from geometry import add_geometry_features
    from io_mat import read_station_mat, select_epoch_observations, scan_station_files
    from io_sp3 import attach_satellite_position, combine_sp3_files


STANDARD_COLUMNS = [
    "year",
    "doy",
    "timestamp",
    "utc_hour",
    "station",
    "prn",
    "constellation",
    "rec_lon",
    "rec_lat",
    "rec_h",
    "rec_x",
    "rec_y",
    "rec_z",
    "sat_lon",
    "sat_lat",
    "sat_h",
    "sat_x",
    "sat_y",
    "sat_z",
    "ipp_lon",
    "ipp_lat",
    "elevation",
    "azimuth",
    "slant_factor",
    "ray_length",
    "stec_obs",
    "stec_iri",
    "valid_flag",
]


def _empty_epoch_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def quality_filter(
    df: pd.DataFrame,
    cfg: DatasetBuildConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply basic observation quality control and return removal counts."""

    counts: dict[str, int] = {"input": int(len(df))}
    mask = pd.Series(True, index=df.index)

    stec_ok = df["stec_obs"].notna()
    counts["drop_nan_stec"] = int((~stec_ok).sum())
    mask &= stec_ok

    elevation_ok = df["elevation"].astype(float) >= cfg.min_elevation_deg
    counts["drop_low_elevation"] = int((mask & ~elevation_ok).sum())
    mask &= elevation_ok

    lon_min, lon_max, lat_min, lat_max = cfg.region.as_tuple()
    region_ok = (
        df["ipp_lon"].astype(float).between(lon_min, lon_max)
        & df["ipp_lat"].astype(float).between(lat_min, lat_max)
    )
    counts["drop_outside_region"] = int((mask & ~region_ok).sum())
    mask &= region_ok

    range_ok = df["stec_obs"].astype(float).between(cfg.stec_min, cfg.stec_max)
    counts["drop_outside_stec_range"] = int((mask & ~range_ok).sum())
    mask &= range_ok

    sat_ok = df["sat_position_matched"].fillna(False)
    counts["drop_unmatched_satellite"] = int((mask & ~sat_ok).sum())
    mask &= sat_ok

    out = df.loc[mask].copy()
    out["valid_flag"] = True
    counts["output"] = int(len(out))
    return out.reset_index(drop=True), counts


def build_epoch_dataset(
    cfg: DatasetBuildConfig,
    sp3_ds,
    timestamp: datetime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a standard ray-wise table for one epoch."""

    year = timestamp.year
    doy = int(timestamp.strftime("%j"))
    station_files = scan_station_files(
        cfg.tec_data_dir,
        year=year,
        doy=doy,
        stations=cfg.stations,
        station_glob=cfg.station_glob,
    )
    station_frames, read_errors = load_station_day_frames(station_files, year, doy)
    return build_epoch_dataset_from_station_frames(
        cfg=cfg,
        sp3_ds=sp3_ds,
        timestamp=timestamp,
        station_frames=station_frames,
        station_file_count=len(station_files),
        read_errors=read_errors,
    )


def load_station_day_frames(
    station_files: list[Path],
    year: int,
    doy: int,
) -> tuple[list[pd.DataFrame], dict[str, int]]:
    """Read all station files for one day once."""

    station_frames = []
    errors = Counter()
    for path in station_files:
        try:
            frame = read_station_mat(path, year=year, doy=doy)
        except Exception as exc:
            errors[type(exc).__name__] += 1
            continue
        if not frame.empty:
            station_frames.append(frame)
    return station_frames, dict(errors)


def build_epoch_dataset_from_station_frames(
    cfg: DatasetBuildConfig,
    sp3_ds,
    timestamp: datetime,
    station_frames: list[pd.DataFrame],
    station_file_count: int,
    read_errors: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one epoch from preloaded daily station DataFrames."""

    year = timestamp.year
    doy = int(timestamp.strftime("%j"))
    stats: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "year": year,
        "doy": doy,
        "station_files": station_file_count,
        "read_errors": read_errors or {},
    }

    frames = []
    for station_df in station_frames:
        frame = select_epoch_observations(
                station_df,
                target_time=timestamp,
                window_seconds=cfg.select_window_seconds,
            )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        stats["quality"] = {"input": 0, "output": 0}
        return _empty_epoch_frame(), stats

    obs = pd.concat(frames, ignore_index=True)
    obs = attach_satellite_position(
        obs,
        sp3_ds,
        timestamp=timestamp,
        tolerance_seconds=cfg.sp3_tolerance_seconds,
    )
    obs = add_time_features(obs)
    obs = add_geometry_features(obs)
    obs = add_constellation_onehot(obs)
    obs["stec_iri"] = pd.NA

    filtered, quality = quality_filter(obs, cfg)
    stats["quality"] = quality
    return order_standard_columns(filtered), stats


def order_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place standard columns first while preserving extension columns."""

    columns = [col for col in STANDARD_COLUMNS if col in df.columns]
    columns += [col for col in df.columns if col not in columns]
    return df.loc[:, columns]


def save_table(df: pd.DataFrame, path: Path, output_format: str) -> Path:
    """Save a dataset table as parquet when possible, otherwise CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    output_format = output_format.lower()
    if output_format == "parquet":
        try:
            df.to_parquet(path.with_suffix(".parquet"), index=False)
            return path.with_suffix(".parquet")
        except Exception:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return csv_path
    if output_format == "csv":
        df.to_csv(path.with_suffix(".csv"), index=False)
        return path.with_suffix(".csv")
    raise ValueError(f"Unsupported output format: {output_format}")


def build_day_dataset(
    cfg: DatasetBuildConfig,
    sp3_ds,
    day: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and save one day of ray-wise observations."""

    year = day.year
    doy = int(day.strftime("%j"))
    station_files = scan_station_files(
        cfg.tec_data_dir,
        year=year,
        doy=doy,
        stations=cfg.stations,
        station_glob=cfg.station_glob,
    )
    station_frames, read_errors = load_station_day_frames(station_files, year, doy)

    frames = []
    epoch_stats = []
    for timestamp in iter_epochs(day, cfg.epoch_interval_seconds):
        frame, stats = build_epoch_dataset_from_station_frames(
            cfg=cfg,
            sp3_ds=sp3_ds,
            timestamp=timestamp,
            station_frames=station_frames,
            station_file_count=len(station_files),
            read_errors=read_errors,
        )
        epoch_stats.append(stats)
        if not frame.empty:
            frames.append(frame)

    day_df = pd.concat(frames, ignore_index=True) if frames else _empty_epoch_frame()
    metadata = {
        "date": day.isoformat(),
        "n_samples": int(len(day_df)),
        "n_stations": int(day_df["station"].nunique()) if "station" in day_df else 0,
        "n_prn": int(day_df["prn"].nunique()) if "prn" in day_df else 0,
        "station_files": len(station_files),
        "station_read_errors": read_errors,
        "epoch_stats": epoch_stats,
    }
    return day_df, metadata


def build_dataset(cfg: DatasetBuildConfig) -> dict[str, Any]:
    """Build datasets for all configured dates."""

    if not cfg.tec_data_dir.exists():
        raise FileNotFoundError(
            f"TEC data directory does not exist: {cfg.tec_data_dir}"
        )
    missing_sp3 = [path for path in cfg.sp3_files if not path.exists()]
    if missing_sp3:
        missing_text = ", ".join(str(path) for path in missing_sp3)
        raise FileNotFoundError(f"SP3/orbit file does not exist: {missing_text}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    sp3_ds = combine_sp3_files(cfg.sp3_files)

    all_metadata: dict[str, Any] = {
        "project_name": cfg.project_name,
        "start_date": cfg.start_date.isoformat(),
        "end_date": cfg.end_date.isoformat(),
        "region": cfg.region.__dict__,
        "min_elevation_deg": cfg.min_elevation_deg,
        "select_window_seconds": cfg.select_window_seconds,
        "epoch_interval_seconds": cfg.epoch_interval_seconds,
        "sp3_files": [str(path) for path in cfg.sp3_files],
        "tec_data_dir": str(cfg.tec_data_dir),
        "outputs": [],
        "days": [],
    }

    for day in iter_dates(cfg.start_date, cfg.end_date):
        day_df, metadata = build_day_dataset(cfg, sp3_ds, day)
        stem = f"raystec_china_{day.strftime('%Y%m%d')}"
        saved_path = save_table(day_df, cfg.output_dir / stem, cfg.output_format)
        metadata["output_file"] = str(saved_path)
        all_metadata["outputs"].append(str(saved_path))
        all_metadata["days"].append(metadata)

    metadata_path = cfg.output_dir / f"metadata_{cfg.project_name}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    all_metadata["metadata_file"] = str(metadata_path)
    return all_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    args = parser.parse_args(argv)
    cfg = load_dataset_config(args.config)
    metadata = build_dataset(cfg)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
