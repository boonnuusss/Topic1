"""Physical baseline helpers for ray-wise STEC-to-VTEC modeling."""

from __future__ import annotations

import pandas as pd

try:
    from .geometry import compute_slant_factor
except ImportError:
    from geometry import compute_slant_factor


def add_single_layer_baseline(
    df: pd.DataFrame,
    vtec_column: str = "vtec_baseline",
    output_column: str = "stec_slm",
    mapping_column: str = "slm_mapping_factor",
    elevation_column: str = "elevation",
    shell_height_km: float = 450.0,
    use_existing_mapping: bool = False,
) -> pd.DataFrame:
    """Add a thin-shell single-layer STEC baseline.

    The model is:

    ``STEC_slm = M(elevation) * VTEC(IPP, time)``

    where ``M`` is the thin-shell mapping factor. The VTEC column must come
    from an independent source such as IRI/PyIRI or an interpolated GIM product.
    """

    if vtec_column not in df.columns:
        raise KeyError(f"Missing VTEC baseline column: {vtec_column}")

    out = df.copy()
    if use_existing_mapping and mapping_column in out.columns:
        mapping = out[mapping_column].astype(float)
    else:
        if elevation_column not in out.columns:
            raise KeyError(f"Missing required elevation column: {elevation_column}")
        mapping = compute_slant_factor(
            out[elevation_column].astype(float),
            shell_height_km=shell_height_km,
        )
        out[mapping_column] = mapping

    out[output_column] = out[vtec_column].astype(float) * mapping
    out["shell_height_km"] = shell_height_km
    return out


def add_slm_vtec_from_stec(
    df: pd.DataFrame,
    stec_column: str = "stec_obs",
    output_column: str = "vtec_slm",
    mapping_column: str = "slm_mapping_factor",
    elevation_column: str = "elevation",
    shell_height_km: float = 450.0,
    use_existing_mapping: bool = False,
) -> pd.DataFrame:
    """Add the classical thin-shell VTEC estimate from observed STEC.

    The current Topic1 mainline uses the single-layer mapping model as the
    traditional STEC-to-VTEC conversion baseline:

    ``VTEC_slm = STEC_obs / M(elevation)``

    where ``M`` is the thin-shell mapping factor. This is intentionally the
    inverse direction of the legacy STEC-prediction helper above.
    """

    if stec_column not in df.columns:
        raise KeyError(f"Missing observed STEC column: {stec_column}")

    out = df.copy()
    if use_existing_mapping and mapping_column in out.columns:
        mapping = out[mapping_column].astype(float)
    else:
        if elevation_column not in out.columns:
            raise KeyError(f"Missing required elevation column: {elevation_column}")
        mapping = compute_slant_factor(
            out[elevation_column].astype(float),
            shell_height_km=shell_height_km,
        )
        out[mapping_column] = mapping

    if (mapping <= 0).any():
        raise ValueError(f"{mapping_column} must be positive for VTEC conversion.")

    out[output_column] = out[stec_column].astype(float) / mapping
    out["shell_height_km"] = shell_height_km
    return out


def add_slm_residual(
    df: pd.DataFrame,
    observation_column: str = "stec_obs",
    baseline_column: str = "stec_slm",
    output_column: str = "stec_residual_slm",
) -> pd.DataFrame:
    """Add the residual target left by the single-layer baseline."""

    for column in [observation_column, baseline_column]:
        if column not in df.columns:
            raise KeyError(f"Missing required column: {column}")

    out = df.copy()
    out[output_column] = (
        out[observation_column].astype(float) - out[baseline_column].astype(float)
    )
    return out
