"""Physical baseline helpers for ray-wise STEC modeling."""

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
