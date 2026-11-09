"""Physical mapping functions for STEC-to-VTEC conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge


EARTH_RADIUS_KM = 6371.0


def slm_slant_factor(
    elevation_deg,
    *,
    shell_height_km: float,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Return the classical thin-shell STEC/VTEC factor."""

    elevation = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    argument = (
        earth_radius_km / (earth_radius_km + shell_height_km)
    ) * np.cos(elevation)
    return 1.0 / np.sqrt(np.maximum(1.0 - argument**2, 1e-12))


def code_mslm_slant_factor(
    elevation_deg,
    *,
    shell_height_km: float = 506.7,
    elevation_scale: float = 0.9782,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Return the CODE modified single-layer STEC/VTEC factor."""

    elevation = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    zenith = np.pi / 2.0 - elevation
    argument = (
        earth_radius_km / (earth_radius_km + shell_height_km)
    ) * np.sin(elevation_scale * zenith)
    return 1.0 / np.sqrt(np.maximum(1.0 - argument**2, 1e-12))


def chapman_plasmasphere_weights(
    shell_heights_km,
    *,
    peak_height_km: float = 350.0,
    chapman_scale_height_km: float = 70.0,
    plasmasphere_density_ratio: float = 100.0,
    plasmasphere_scale_height_km: float = 10000.0,
) -> np.ndarray:
    """Return normalized vertical-TEC weights for a fixed multi-layer profile."""

    heights = np.asarray(shell_heights_km, dtype=np.float64)
    if heights.ndim != 1 or heights.size < 2 or np.any(np.diff(heights) <= 0):
        raise ValueError("shell_heights_km must be a strictly increasing 1-D array.")
    if chapman_scale_height_km <= 0 or plasmasphere_scale_height_km <= 0:
        raise ValueError("Profile scale heights must be positive.")
    if plasmasphere_density_ratio <= 0:
        raise ValueError("plasmasphere_density_ratio must be positive.")

    z = (heights - peak_height_km) / chapman_scale_height_km
    chapman = np.exp(0.5 * (1.0 - z - np.exp(-z)))
    plasmasphere = (
        np.exp(-heights / plasmasphere_scale_height_km)
        / plasmasphere_density_ratio
    )
    density = chapman + plasmasphere

    widths = np.empty_like(heights)
    widths[1:-1] = 0.5 * (heights[2:] - heights[:-2])
    widths[0] = 0.5 * (heights[1] - heights[0])
    widths[-1] = 0.5 * (heights[-1] - heights[-2])
    vertical_content = density * widths
    return vertical_content / vertical_content.sum()


def multilayer_slant_factor(
    elevation_deg,
    *,
    shell_heights_km,
    shell_weights,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Return a vertical-content-weighted sum of thin-shell factors."""

    elevations = np.asarray(elevation_deg, dtype=np.float64)
    heights = np.asarray(shell_heights_km, dtype=np.float64)
    weights = np.asarray(shell_weights, dtype=np.float64)
    if heights.shape != weights.shape:
        raise ValueError("shell heights and weights must have the same shape.")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("shell weights must be nonnegative and sum to one.")
    factors = np.stack(
        [
            slm_slant_factor(
                elevations,
                shell_height_km=float(height),
                earth_radius_km=earth_radius_km,
            )
            for height in heights
        ],
        axis=1,
    )
    return factors @ weights


def _apmf_design(
    elevation_deg,
    azimuth_deg,
    local_time_hour,
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Eq. (3) APMF base and linear coefficient design."""

    if order < 1:
        raise ValueError("APMF order must be at least one.")
    azimuth = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    local_time = (
        2.0 * np.pi * np.asarray(local_time_hour, dtype=np.float64) / 24.0
    )
    vertical_fraction = 1.0 / code_mslm_slant_factor(elevation_deg)
    eprime = np.arcsin(np.clip(vertical_fraction, -1.0, 1.0))
    multiplier = np.cos(eprime) * np.cos(azimuth)

    terms = [np.ones_like(azimuth)]
    for harmonic in range(1, order + 1):
        terms.extend(
            [np.cos(harmonic * azimuth), np.sin(harmonic * azimuth)]
        )
    for harmonic in range(1, order + 1):
        terms.extend(
            [np.cos(harmonic * local_time), np.sin(harmonic * local_time)]
        )
    design = multiplier[:, None] * np.column_stack(terms)
    return vertical_fraction, design


@dataclass
class APMFModel:
    """Latitude-banded additional-azimuth-parameter mapping function."""

    order: int
    latitude_band_width_deg: float
    ridge_alpha: float
    min_band_samples: int
    global_coefficients: np.ndarray
    band_coefficients: dict[int, np.ndarray]
    band_sample_counts: dict[int, int]
    vertical_fraction_min: float = 0.05
    vertical_fraction_max: float = 1.0

    @staticmethod
    def band_ids(latitude_deg, width_deg: float) -> np.ndarray:
        return np.floor(
            np.asarray(latitude_deg, dtype=np.float64) / width_deg
        ).astype(int)

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        *,
        stec_column: str,
        vtec_column: str,
        elevation_column: str,
        azimuth_column: str,
        local_time_column: str,
        latitude_column: str,
        order: int = 12,
        latitude_band_width_deg: float = 5.0,
        ridge_alpha: float = 1e-4,
        min_band_samples: int = 100,
    ) -> "APMFModel":
        """Fit APMF coefficients from model-derived STEC/VTEC training rows."""

        required = [
            stec_column,
            vtec_column,
            elevation_column,
            azimuth_column,
            local_time_column,
            latitude_column,
        ]
        clean = df.dropna(subset=required).copy()
        clean = clean.loc[
            (clean[stec_column].astype(float) > 0)
            & (clean[vtec_column].astype(float) > 0)
        ]
        if len(clean) < max(min_band_samples, 4 * order + 1):
            raise ValueError("Too few valid rows to fit APMF coefficients.")

        fraction_column = "__apmf_vertical_fraction"
        clean[fraction_column] = (
            clean[vtec_column].to_numpy(float)
            / clean[stec_column].to_numpy(float)
        )
        return cls.fit_vertical_fraction(
            clean,
            fraction_column=fraction_column,
            elevation_column=elevation_column,
            azimuth_column=azimuth_column,
            local_time_column=local_time_column,
            latitude_column=latitude_column,
            order=order,
            latitude_band_width_deg=latitude_band_width_deg,
            ridge_alpha=ridge_alpha,
            min_band_samples=min_band_samples,
        )

    @classmethod
    def fit_vertical_fraction(
        cls,
        df: pd.DataFrame,
        *,
        fraction_column: str,
        elevation_column: str,
        azimuth_column: str,
        local_time_column: str,
        latitude_column: str,
        order: int = 12,
        latitude_band_width_deg: float = 5.0,
        ridge_alpha: float = 1e-4,
        min_band_samples: int = 100,
    ) -> "APMFModel":
        """Fit APMF coefficients from precomputed VTEC/STEC fractions."""

        required = [
            fraction_column,
            elevation_column,
            azimuth_column,
            local_time_column,
            latitude_column,
        ]
        clean = df.dropna(subset=required).copy()
        clean = clean.loc[clean[fraction_column].astype(float) > 0]
        if len(clean) < max(min_band_samples, 4 * order + 1):
            raise ValueError("Too few valid rows to fit APMF coefficients.")

        base, design = _apmf_design(
            clean[elevation_column],
            clean[azimuth_column],
            clean[local_time_column],
            order=order,
        )
        target = clean[fraction_column].to_numpy(float) - base
        global_model = Ridge(alpha=ridge_alpha, fit_intercept=False)
        global_model.fit(design, target)

        band_ids = cls.band_ids(clean[latitude_column], latitude_band_width_deg)
        coefficients: dict[int, np.ndarray] = {}
        sample_counts: dict[int, int] = {}
        for band_id in np.unique(band_ids):
            selected = band_ids == band_id
            sample_counts[int(band_id)] = int(selected.sum())
            if selected.sum() < min_band_samples:
                continue
            model = Ridge(alpha=ridge_alpha, fit_intercept=False)
            model.fit(design[selected], target[selected])
            coefficients[int(band_id)] = model.coef_.astype(np.float64)

        return cls(
            order=order,
            latitude_band_width_deg=latitude_band_width_deg,
            ridge_alpha=ridge_alpha,
            min_band_samples=min_band_samples,
            global_coefficients=global_model.coef_.astype(np.float64),
            band_coefficients=coefficients,
            band_sample_counts=sample_counts,
        )

    def predict_vertical_fraction(
        self,
        df: pd.DataFrame,
        *,
        elevation_column: str,
        azimuth_column: str,
        local_time_column: str,
        latitude_column: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict VTEC/STEC and indicate rows using the global fallback."""

        base, design = _apmf_design(
            df[elevation_column],
            df[azimuth_column],
            df[local_time_column],
            order=self.order,
        )
        bands = self.band_ids(
            df[latitude_column], self.latitude_band_width_deg
        )
        predicted = np.empty(len(df), dtype=np.float64)
        fallback = np.zeros(len(df), dtype=bool)
        for band_id in np.unique(bands):
            selected = bands == band_id
            coefficients = self.band_coefficients.get(int(band_id))
            if coefficients is None:
                coefficients = self.global_coefficients
                fallback[selected] = True
            predicted[selected] = base[selected] + design[selected] @ coefficients
        return np.clip(
            predicted,
            self.vertical_fraction_min,
            self.vertical_fraction_max,
        ), fallback


def build_high_low_cpp_pairs(
    df: pd.DataFrame,
    *,
    stec_column: str,
    timestamp_column: str,
    elevation_column: str,
    azimuth_column: str,
    local_time_column: str,
    latitude_column: str,
    longitude_column: str,
    high_elevation_deg: float = 70.0,
    low_elevation_min_deg: float = 10.0,
    low_elevation_max_deg: float = 30.0,
    latitude_tolerance_deg: float = 0.4,
    scaled_longitude_tolerance_deg: float = 0.4,
    pair_policy: str = "all",
) -> pd.DataFrame:
    """Build same-epoch high-low coinciding-pierce-point STEC pairs.

    The proximity rule follows Eq. (4) of Huo et al. (2024). The high-ray
    VTEC proxy uses CODE MSLM and therefore requires no VTEC reference column.
    """

    if pair_policy not in {"all", "nearest"}:
        raise ValueError("pair_policy must be 'all' or 'nearest'.")
    if latitude_tolerance_deg <= 0 or scaled_longitude_tolerance_deg <= 0:
        raise ValueError("CPP coordinate tolerances must be positive.")
    if not low_elevation_min_deg < low_elevation_max_deg < high_elevation_deg:
        raise ValueError(
            "Expected low_min < low_max < high_elevation for CPP pairing."
        )

    required = [
        stec_column,
        timestamp_column,
        elevation_column,
        azimuth_column,
        local_time_column,
        latitude_column,
        longitude_column,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"CPP input is missing required columns: {missing}")

    clean = df.loc[:, required].dropna(subset=required).copy().reset_index(drop=False)
    clean = clean.rename(columns={"index": "cpp_source_row_index"})
    clean = clean.loc[clean[stec_column].astype(float) > 0].reset_index(drop=True)
    low_positions: list[int] = []
    high_positions: list[int] = []
    pair_scores: list[float] = []

    for _, group in clean.groupby(timestamp_column, sort=False):
        high = group.loc[group[elevation_column].astype(float) > high_elevation_deg]
        low = group.loc[
            (group[elevation_column].astype(float) >= low_elevation_min_deg)
            & (group[elevation_column].astype(float) < low_elevation_max_deg)
        ]
        if high.empty or low.empty:
            continue

        high_lat = high[latitude_column].to_numpy(float)
        low_lat = low[latitude_column].to_numpy(float)
        high_scaled_lon = high[longitude_column].to_numpy(float) / np.cos(
            np.deg2rad(high_lat)
        )
        low_scaled_lon = low[longitude_column].to_numpy(float) / np.cos(
            np.deg2rad(low_lat)
        )
        high_points = np.column_stack(
            [
                high_lat / latitude_tolerance_deg,
                high_scaled_lon / scaled_longitude_tolerance_deg,
            ]
        )
        low_points = np.column_stack(
            [
                low_lat / latitude_tolerance_deg,
                low_scaled_lon / scaled_longitude_tolerance_deg,
            ]
        )
        tree = cKDTree(high_points)
        matches = tree.query_ball_point(low_points, r=1.0, p=np.inf)
        high_index = high.index.to_numpy(int)
        low_index = low.index.to_numpy(int)
        for low_offset, candidates in enumerate(matches):
            if not candidates:
                continue
            deltas = np.abs(high_points[candidates] - low_points[low_offset])
            scores = np.max(deltas, axis=1)
            if pair_policy == "nearest":
                selected_offsets = [int(candidates[int(np.argmin(scores))])]
                selected_scores = [float(np.min(scores))]
            else:
                selected_offsets = [int(value) for value in candidates]
                selected_scores = [float(value) for value in scores]
            low_positions.extend([int(low_index[low_offset])] * len(selected_offsets))
            high_positions.extend(high_index[selected_offsets].astype(int).tolist())
            pair_scores.extend(selected_scores)

    if not low_positions:
        raise ValueError("No high-low CPP pairs satisfy the requested criteria.")

    low_rows = clean.iloc[low_positions].reset_index(drop=True)
    high_rows = clean.iloc[high_positions].reset_index(drop=True)
    pairs = low_rows.copy()
    high_stec = high_rows[stec_column].to_numpy(float)
    high_elevation = high_rows[elevation_column].to_numpy(float)
    vtec_proxy = high_stec / code_mslm_slant_factor(high_elevation)
    pairs["cpp_high_source_row_index"] = high_rows[
        "cpp_source_row_index"
    ].to_numpy()
    pairs["cpp_high_elevation_deg"] = high_elevation
    pairs["cpp_high_stec"] = high_stec
    pairs["cpp_vtec_proxy"] = vtec_proxy
    pairs["cpp_vertical_fraction"] = (
        vtec_proxy / low_rows[stec_column].to_numpy(float)
    )
    pairs["cpp_latitude_delta_deg"] = np.abs(
        high_rows[latitude_column].to_numpy(float)
        - low_rows[latitude_column].to_numpy(float)
    )
    pairs["cpp_scaled_longitude_delta_deg"] = np.abs(
        high_rows[longitude_column].to_numpy(float)
        / np.cos(np.deg2rad(high_rows[latitude_column].to_numpy(float)))
        - low_rows[longitude_column].to_numpy(float)
        / np.cos(np.deg2rad(low_rows[latitude_column].to_numpy(float)))
    )
    pairs["cpp_normalized_distance"] = np.asarray(pair_scores)
    return pairs
