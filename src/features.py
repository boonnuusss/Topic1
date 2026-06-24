"""Feature engineering for standard ray-wise STEC tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
    "rec_lon",
    "rec_lat",
    "rec_h",
    "sat_lon",
    "sat_lat",
    "sat_h",
    "ipp_lon",
    "ipp_lat",
    "elevation",
    "azimuth_sin",
    "azimuth_cos",
    "slant_factor",
    "ray_length",
    "los_x",
    "los_y",
    "los_z",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "local_time_ipp_sin",
    "local_time_ipp_cos",
]


CONSTELLATION_FEATURE_COLUMNS = [
    "constellation_gps",
    "constellation_glonass",
    "constellation_galileo",
    "constellation_bds",
    "constellation_sbas",
    "constellation_qzss",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add UTC, DOY, and IPP local-time cyclical encodings."""

    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    utc_hour = (
        ts.dt.hour
        + ts.dt.minute / 60.0
        + ts.dt.second / 3600.0
        + ts.dt.microsecond / 3.6e9
    )
    out["utc_hour"] = utc_hour
    out["month"] = ts.dt.month
    out["doy"] = ts.dt.dayofyear
    out["local_time_ipp"] = (utc_hour + out["ipp_lon"].astype(float) / 15.0) % 24.0

    out["hour_sin"] = np.sin(2 * np.pi * out["utc_hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["utc_hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    out["local_time_ipp_sin"] = np.sin(2 * np.pi * out["local_time_ipp"] / 24.0)
    out["local_time_ipp_cos"] = np.cos(2 * np.pi * out["local_time_ipp"] / 24.0)
    return out


def add_constellation_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple one-hot constellation indicators."""

    out = df.copy()
    if "constellation" not in out:
        return out
    for name in ["GPS", "GLONASS", "Galileo", "BDS", "SBAS", "QZSS"]:
        out[f"constellation_{name.lower()}"] = (out["constellation"] == name).astype(int)
    return out


def feature_columns(include_iri: bool = False) -> list[str]:
    """Return default model feature columns."""

    columns = list(BASE_FEATURE_COLUMNS) + list(CONSTELLATION_FEATURE_COLUMNS)
    if include_iri:
        columns.append("stec_iri")
    return columns


def feature_columns_with_physical_baseline(
    baseline_column: str = "stec_slm",
) -> list[str]:
    """Return default features plus a physical STEC baseline column."""

    return feature_columns(include_iri=False) + [baseline_column]


def build_feature_matrix(
    df: pd.DataFrame,
    include_iri: bool = False,
    include_physical_baseline: bool = False,
    physical_baseline_column: str = "stec_slm",
    target: str = "stec_obs",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X/y tables for baseline modeling."""

    if include_physical_baseline:
        requested = feature_columns_with_physical_baseline(
            baseline_column=physical_baseline_column
        )
    else:
        requested = feature_columns(include_iri=include_iri)
    columns = [c for c in requested if c in df.columns]
    return df[columns].copy(), df[target].copy()
