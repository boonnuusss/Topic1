"""Geometry and coordinate utilities for receiver-satellite ray features."""

from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0

_ecef_transformer = None


def _get_ecef_transformer():
    """Return a cached pyproj Transformer for geodetic→ECEF conversion."""
    global _ecef_transformer
    if _ecef_transformer is None:
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise RuntimeError(
                "pyproj is required for geodetic/ECEF conversion."
            ) from exc
        _ecef_transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:4978", always_xy=True
        )
    return _ecef_transformer


def geodetic_to_ecef(lon_deg, lat_deg, height_m=0.0):
    """Convert geodetic lon/lat/height to ECEF meters."""
    return _get_ecef_transformer().transform(lon_deg, lat_deg, height_m)


def add_receiver_ecef(df: pd.DataFrame) -> pd.DataFrame:
    """Add receiver ECEF coordinates in meters from receiver BLH columns."""

    out = df.copy()
    x, y, z = geodetic_to_ecef(
        out["rec_lon"].to_numpy(float),
        out["rec_lat"].to_numpy(float),
        out.get("rec_h", 0.0),
    )
    out["rec_x"] = x
    out["rec_y"] = y
    out["rec_z"] = z
    return out


def compute_ray_length(rec_xyz: np.ndarray, sat_xyz: np.ndarray) -> np.ndarray:
    """Compute straight-line receiver-satellite distance in meters."""

    return np.linalg.norm(sat_xyz - rec_xyz, axis=1)


def compute_slant_factor(elevation_deg, shell_height_km: float = 450.0):
    """Thin-shell mapping factor using elevation angle in degrees."""

    elevation_rad = np.deg2rad(elevation_deg)
    ratio = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + shell_height_km)
    inside = 1.0 - (ratio * np.cos(elevation_rad)) ** 2
    return 1.0 / np.sqrt(np.maximum(inside, 1e-12))


def add_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ray-length, line-of-sight, slant-factor, and angular encodings."""

    out = add_receiver_ecef(df)
    rec = out[["rec_x", "rec_y", "rec_z"]].to_numpy(float)
    sat = out[["sat_x", "sat_y", "sat_z"]].to_numpy(float)
    rel = sat - rec
    ray_length = compute_ray_length(rec, sat)
    out["ray_length"] = ray_length
    out["los_x"] = rel[:, 0] / ray_length
    out["los_y"] = rel[:, 1] / ray_length
    out["los_z"] = rel[:, 2] / ray_length
    out["slant_factor"] = compute_slant_factor(out["elevation"].to_numpy(float))

    az_rad = np.deg2rad(out["azimuth"].to_numpy(float))
    el_rad = np.deg2rad(out["elevation"].to_numpy(float))
    out["azimuth_sin"] = np.sin(az_rad)
    out["azimuth_cos"] = np.cos(az_rad)
    out["elevation_sin"] = np.sin(el_rad)
    out["elevation_cos"] = np.cos(el_rad)
    return out

