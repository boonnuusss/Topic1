"""Tests for physical STEC-to-VTEC mapping functions."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.physical_mapping import (
    APMFModel,
    build_high_low_cpp_pairs,
    chapman_plasmasphere_weights,
    code_mslm_slant_factor,
    multilayer_slant_factor,
    slm_slant_factor,
)


class PhysicalMappingTests(unittest.TestCase):
    def test_all_geometric_factors_are_one_at_zenith(self) -> None:
        self.assertAlmostEqual(float(slm_slant_factor([90], shell_height_km=450)[0]), 1.0)
        self.assertAlmostEqual(float(code_mslm_slant_factor([90])[0]), 1.0)
        heights = np.array([300.0, 450.0, 600.0])
        weights = np.array([0.2, 0.6, 0.2])
        self.assertAlmostEqual(
            float(multilayer_slant_factor([90], shell_heights_km=heights, shell_weights=weights)[0]),
            1.0,
        )

    def test_slm_factor_decreases_with_elevation(self) -> None:
        factors = slm_slant_factor([10, 30, 60, 90], shell_height_km=450)
        self.assertTrue(np.all(np.diff(factors) < 0))

    def test_chapman_weights_are_normalized(self) -> None:
        weights = chapman_plasmasphere_weights(np.arange(90.0, 2000.1, 10.0))
        self.assertTrue(np.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_apmf_recovers_synthetic_harmonic(self) -> None:
        rng = np.random.default_rng(42)
        n = 1000
        frame = pd.DataFrame(
            {
                "elevation": rng.uniform(15, 85, n),
                "azimuth": rng.uniform(0, 360, n),
                "local_time_ipp": rng.uniform(0, 24, n),
                "ipp_lat": rng.uniform(25, 29.9, n),
            }
        )
        base = 1.0 / code_mslm_slant_factor(frame["elevation"])
        eprime = np.arcsin(base)
        azimuth = np.deg2rad(frame["azimuth"].to_numpy())
        correction = 0.03 * np.cos(eprime) * np.cos(azimuth)
        fraction = base + correction
        frame["stec"] = 50.0
        frame["vtec"] = frame["stec"] * fraction
        model = APMFModel.fit(
            frame,
            stec_column="stec",
            vtec_column="vtec",
            elevation_column="elevation",
            azimuth_column="azimuth",
            local_time_column="local_time_ipp",
            latitude_column="ipp_lat",
            order=2,
            ridge_alpha=1e-10,
            min_band_samples=100,
        )
        predicted, fallback = model.predict_vertical_fraction(
            frame,
            elevation_column="elevation",
            azimuth_column="azimuth",
            local_time_column="local_time_ipp",
            latitude_column="ipp_lat",
        )
        self.assertLess(float(np.sqrt(np.mean((predicted - fraction) ** 2))), 1e-6)
        self.assertFalse(fallback.any())

    def test_high_low_cpp_pairs_use_stec_without_vtec(self) -> None:
        rows = []
        for epoch in range(60):
            timestamp = pd.Timestamp("2024-05-05") + pd.Timedelta(minutes=epoch)
            latitude = 25.0 + 0.01 * (epoch % 20)
            longitude = 110.0 + 0.01 * (epoch % 10)
            vtec = 40.0 + 0.1 * epoch
            high_elevation = 80.0
            high_stec = vtec * code_mslm_slant_factor([high_elevation])[0]
            low_elevation = 20.0
            low_fraction = 1.0 / code_mslm_slant_factor([low_elevation])[0]
            rows.extend(
                [
                    {
                        "timestamp": timestamp,
                        "stec": high_stec,
                        "elevation": high_elevation,
                        "azimuth": 0.0,
                        "local_time_ipp": 12.0,
                        "ipp_lat": latitude,
                        "ipp_lon": longitude,
                    },
                    {
                        "timestamp": timestamp,
                        "stec": vtec / low_fraction,
                        "elevation": low_elevation,
                        "azimuth": 180.0,
                        "local_time_ipp": 12.0,
                        "ipp_lat": latitude,
                        "ipp_lon": longitude,
                    },
                ]
            )
        frame = pd.DataFrame(rows)
        frame["vtec_forbidden"] = 9999.0
        pairs = build_high_low_cpp_pairs(
            frame,
            stec_column="stec",
            timestamp_column="timestamp",
            elevation_column="elevation",
            azimuth_column="azimuth",
            local_time_column="local_time_ipp",
            latitude_column="ipp_lat",
            longitude_column="ipp_lon",
        )
        self.assertEqual(len(pairs), 60)
        self.assertNotIn("vtec_forbidden", pairs.columns)
        self.assertTrue(
            np.allclose(pairs["cpp_vertical_fraction"], low_fraction)
        )
        self.assertTrue((pairs["cpp_latitude_delta_deg"] == 0).all())
        self.assertTrue(
            (pairs["cpp_scaled_longitude_delta_deg"] == 0).all()
        )

    def test_cpp_pairing_never_crosses_epochs(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "timestamp": "2024-05-05T00:00:00",
                    "stec": 40.0,
                    "elevation": 80.0,
                    "azimuth": 0.0,
                    "local_time_ipp": 8.0,
                    "ipp_lat": 30.0,
                    "ipp_lon": 110.0,
                },
                {
                    "timestamp": "2024-05-05T00:05:00",
                    "stec": 80.0,
                    "elevation": 20.0,
                    "azimuth": 180.0,
                    "local_time_ipp": 8.0,
                    "ipp_lat": 30.0,
                    "ipp_lon": 110.0,
                },
            ]
        )
        with self.assertRaisesRegex(ValueError, "No high-low CPP pairs"):
            build_high_low_cpp_pairs(
                frame,
                stec_column="stec",
                timestamp_column="timestamp",
                elevation_column="elevation",
                azimuth_column="azimuth",
                local_time_column="local_time_ipp",
                latitude_column="ipp_lat",
                longitude_column="ipp_lon",
            )


if __name__ == "__main__":
    unittest.main()
