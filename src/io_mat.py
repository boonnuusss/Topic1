"""Read GNSS TEC MATLAB files into standard ray-wise observation tables."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_MAT_FIELDS = (
    "StationAZIMUTH",
    "StationELEVATION",
    "UTh",
    "PRN",
    "slantTEC",
    "ippLON",
    "ippLAT",
    "StationLON",
    "StationLAT",
)


def scan_station_files(
    tec_data_dir: str | Path,
    year: int,
    doy: int,
    stations: Iterable[str] | None = None,
    station_glob: str = "*.mat",
) -> list[Path]:
    """Find station `.mat` files under `tec_data_dir/year/doy`."""

    day_dir = Path(tec_data_dir) / str(year) / str(doy)
    if stations:
        files: list[Path] = []
        for station in stations:
            files.extend(sorted(day_dir.glob(f"{station.lower()}*.mat")))
            files.extend(sorted(day_dir.glob(f"{station.upper()}*.mat")))
        return sorted(set(files))
    return sorted(day_dir.glob(station_glob))


def station_from_path(path: str | Path) -> str:
    """Infer a four-character station code from a TEC filename."""

    return Path(path).stem[:4].upper()


def _loadmat(path: str | Path) -> dict:
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise RuntimeError("scipy is required to read MATLAB `.mat` files.") from exc
    return sio.loadmat(path)


def _field(data: dict, name: str) -> np.ndarray:
    if name not in data:
        raise KeyError(f"Missing required MATLAB field: {name}")
    return np.asarray(data[name]).ravel()


def read_station_mat(path: str | Path, year: int, doy: int) -> pd.DataFrame:
    """Read one station TEC `.mat` file as a row-wise observation DataFrame."""

    mat = _loadmat(path)
    for name in REQUIRED_MAT_FIELDS:
        _field(mat, name)

    station = station_from_path(path)
    base_date = datetime(int(year), 1, 1) + timedelta(days=int(doy) - 1)
    uth = _field(mat, "UTh").astype(float)
    timestamps = [base_date + timedelta(hours=float(h)) for h in uth]

    n_rows = len(uth)
    station_lon = _field(mat, "StationLON").astype(float)
    station_lat = _field(mat, "StationLAT").astype(float)
    rec_lon = float(station_lon[0]) if len(station_lon) else np.nan
    rec_lat = float(station_lat[0]) if len(station_lat) else np.nan

    return pd.DataFrame(
        {
            "year": int(year),
            "doy": int(doy),
            "timestamp": pd.to_datetime(timestamps),
            "utc_hour": uth,
            "station": station,
            "prn": _field(mat, "PRN").astype(int),
            "rec_lon": rec_lon,
            "rec_lat": rec_lat,
            "rec_h": 0.0,
            "ipp_lon": _field(mat, "ippLON").astype(float),
            "ipp_lat": _field(mat, "ippLAT").astype(float),
            "elevation": _field(mat, "StationELEVATION").astype(float),
            "azimuth": _field(mat, "StationAZIMUTH").astype(float),
            "stec_obs": _field(mat, "slantTEC").astype(float),
            "source_file": str(Path(path)),
        },
        index=np.arange(n_rows),
    )


def select_epoch_observations(
    obs_df: pd.DataFrame,
    target_time: datetime,
    window_seconds: int = 30,
) -> pd.DataFrame:
    """Select observations within `target_time +/- window_seconds / 2`."""

    half_window = timedelta(seconds=window_seconds / 2)
    start = pd.Timestamp(target_time - half_window)
    end = pd.Timestamp(target_time + half_window)
    mask = (obs_df["timestamp"] >= start) & (obs_df["timestamp"] <= end)
    selected = obs_df.loc[mask].copy()
    selected["timestamp"] = pd.Timestamp(target_time)
    selected["utc_hour"] = (
        target_time.hour
        + target_time.minute / 60.0
        + target_time.second / 3600.0
    )
    return selected.reset_index(drop=True)


def read_one_station_epoch(
    path: str | Path,
    year: int,
    doy: int,
    target_time: datetime,
    window_seconds: int = 30,
) -> pd.DataFrame:
    """Read one station file and return observations for one epoch."""

    return select_epoch_observations(
        read_station_mat(path, year, doy),
        target_time=target_time,
        window_seconds=window_seconds,
    )

