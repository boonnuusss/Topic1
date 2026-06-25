"""SP3 precise orbit readers and satellite-position attachment utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def map_sp3_prn(system: str, prn_number: int) -> int:
    """Map constellation-specific SP3 PRN to the project-wide PRN convention."""

    offsets = {
        "G": 0,
        "R": 500,
        "E": 70,
        "C": 700,
        "S": 119,
        "J": 180,
    }
    return int(prn_number) + offsets.get(system.upper(), 0)


def constellation_from_prn(mapped_prn: int) -> str:
    """Infer constellation name from the mapped PRN convention."""

    prn = int(mapped_prn)
    if prn >= 700:
        return "BDS"
    if prn >= 500:
        return "GLONASS"
    if prn >= 180:
        return "QZSS"
    if prn >= 119:
        return "SBAS"
    if prn >= 70:
        return "Galileo"
    return "GPS"


_geo_transformer = None


def _get_geo_transformer():
    """Return a cached pyproj Transformer for ECEF→geodetic conversion."""
    global _geo_transformer
    if _geo_transformer is None:
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise RuntimeError(
                "pyproj is required to convert SP3 ECEF to BLH."
            ) from exc
        _geo_transformer = Transformer.from_crs(
            "EPSG:4978", "EPSG:4326", always_xy=True
        )
    return _geo_transformer


def _ecef_to_geodetic(x, y, z):
    lon, lat, h_m = _get_geo_transformer().transform(x, y, z)
    return lon, lat, np.asarray(h_m) / 1000.0


def read_sp3_to_xarray(sp3_path: str | Path) -> xr.Dataset:
    """Read an ASCII SP3 file into an xarray Dataset.

    Coordinates:
        time: precise-orbit epochs.
        prn: project-wide mapped PRN.

    Data variables:
        X, Y, Z: satellite ECEF coordinates in meters.
        longitude, latitude: geodetic satellite longitude/latitude in degrees.
        height: satellite height above ellipsoid in kilometers.
    """

    epochs: list[datetime] = []
    epoch_positions: list[dict[int, tuple[float, float, float]]] = []
    prn_set: set[int] = set()

    with Path(sp3_path).open("r", encoding="utf-8", errors="ignore") as f:
        current_positions: dict[int, tuple[float, float, float]] | None = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("*"):
                parts = line.split()
                try:
                    second = float(parts[6])
                    whole_second = int(second)
                    microsecond = int(round((second - whole_second) * 1e6))
                    epoch = datetime(
                        int(parts[1]),
                        int(parts[2]),
                        int(parts[3]),
                        int(parts[4]),
                        int(parts[5]),
                        whole_second,
                        microsecond,
                    )
                except (IndexError, ValueError):
                    current_positions = None
                    continue
                epochs.append(epoch)
                current_positions = {}
                epoch_positions.append(current_positions)
                continue

            if not line.startswith("P") or current_positions is None:
                continue

            try:
                system = line[1]
                prn_number = int(line[2:4].strip())
                fields = line[4:].strip().split()
                x_km, y_km, z_km = map(float, fields[:3])
            except (IndexError, ValueError):
                continue

            mapped_prn = map_sp3_prn(system, prn_number)
            current_positions[mapped_prn] = (x_km * 1000, y_km * 1000, z_km * 1000)
            prn_set.add(mapped_prn)

    if not epochs:
        raise ValueError(f"No SP3 epochs found in {sp3_path}")

    prn_list = sorted(prn_set)
    prn_index = {prn: idx for idx, prn in enumerate(prn_list)}
    shape = (len(epochs), len(prn_list))
    x = np.full(shape, np.nan)
    y = np.full(shape, np.nan)
    z = np.full(shape, np.nan)

    for time_idx, positions in enumerate(epoch_positions):
        for prn, xyz in positions.items():
            prn_idx = prn_index[prn]
            x[time_idx, prn_idx] = xyz[0]
            y[time_idx, prn_idx] = xyz[1]
            z[time_idx, prn_idx] = xyz[2]

    lon, lat, height_km = _ecef_to_geodetic(x, y, z)
    return xr.Dataset(
        data_vars={
            "X": (("time", "prn"), x),
            "Y": (("time", "prn"), y),
            "Z": (("time", "prn"), z),
            "longitude": (("time", "prn"), lon),
            "latitude": (("time", "prn"), lat),
            "height": (("time", "prn"), height_km),
        },
        coords={
            "time": np.array(epochs, dtype="datetime64[ns]"),
            "prn": np.array(prn_list, dtype=int),
        },
        attrs={"source_file": str(sp3_path)},
    )


def combine_sp3_files(sp3_files: list[str | Path]) -> xr.Dataset:
    """Read and concatenate one or more SP3 files."""

    datasets = []
    for path in sp3_files:
        path = Path(path)
        if path.suffix.lower() in {".nc", ".netcdf"}:
            datasets.append(xr.open_dataset(path))
        else:
            datasets.append(read_sp3_to_xarray(path))
    if not datasets:
        raise ValueError("At least one SP3 file is required.")
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="time").sortby("time")


def satellite_frame_at_epoch(
    sp3_ds: xr.Dataset,
    timestamp: datetime | pd.Timestamp,
    tolerance_seconds: int = 900,
) -> pd.DataFrame:
    """Return all satellite positions nearest to one epoch."""

    target = np.datetime64(pd.Timestamp(timestamp).to_datetime64())
    selected = sp3_ds.sel(
        time=target,
        method="nearest",
        tolerance=np.timedelta64(int(tolerance_seconds), "s"),
    )
    return pd.DataFrame(
        {
            "prn": selected["prn"].values.astype(int),
            "sat_x": selected["X"].values.astype(float),
            "sat_y": selected["Y"].values.astype(float),
            "sat_z": selected["Z"].values.astype(float),
            "sat_lon": selected["longitude"].values.astype(float),
            "sat_lat": selected["latitude"].values.astype(float),
            "sat_h": selected["height"].values.astype(float),
            "satellite_timestamp": pd.Timestamp(selected["time"].values),
        }
    )


def attach_satellite_position(
    obs_df: pd.DataFrame,
    sp3_ds: xr.Dataset,
    timestamp: datetime | pd.Timestamp,
    tolerance_seconds: int = 900,
) -> pd.DataFrame:
    """Attach nearest SP3 satellite position to each observation row."""

    sat_df = satellite_frame_at_epoch(sp3_ds, timestamp, tolerance_seconds)
    out = obs_df.merge(sat_df, on="prn", how="left")
    out["constellation"] = out["prn"].map(constellation_from_prn)
    out["sat_position_matched"] = out[["sat_x", "sat_y", "sat_z"]].notna().all(axis=1)
    return out
