"""Tests for NeQuick-G navigation parsing and ray clipping."""

from __future__ import annotations

import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.geometry import geodetic_to_ecef
from src.nequickg_simulation_builder import (
    parse_galileo_iono_coefficients,
    ray_points_at_altitudes,
)


class NeQuickGSimulationTests(unittest.TestCase):
    def test_parse_unique_galileo_coefficients(self) -> None:
        content = (
            "     3.04           N: GNSS NAV DATA    M: MIXED            "
            "RINEX VERSION / TYPE\n"
            "GAL    1.4200E+02  1.5234E-01  1.0101E-02                   "
            "IONOSPHERIC CORR\n"
            "                                                            "
            "END OF HEADER\n"
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "nav.rnx.gz"
            with gzip.open(path, "wt", encoding="ascii") as handle:
                handle.write(content)
            coefficients = parse_galileo_iono_coefficients(path)
        self.assertEqual(coefficients.a0, 142.0)
        self.assertAlmostEqual(coefficients.a1, 0.15234)
        self.assertAlmostEqual(coefficients.a2, 0.010101)

    def test_ray_clipping_reaches_requested_geodetic_altitudes(self) -> None:
        rec = np.asarray(
            [geodetic_to_ecef(110.0, 30.0, 0.0)], dtype=np.float64
        )
        sat = np.asarray(
            [geodetic_to_ecef(115.0, 35.0, 20_000_000.0)],
            dtype=np.float64,
        )
        lower, upper = ray_points_at_altitudes(rec, sat, (90_000.0, 2_000_000.0))
        self.assertAlmostEqual(float(lower[0, 2]), 90_000.0, places=3)
        self.assertAlmostEqual(float(upper[0, 2]), 2_000_000.0, places=3)

    def test_pypi_nequick_matches_published_package_benchmark(self) -> None:
        try:
            from datetime import datetime
            from nequick import NeQuick
        except ImportError:
            self.skipTest("PyPI nequick is installed only in .venv-nequick")
        model = NeQuick(236.831641, -0.39362878, 0.00402826613)
        stec = model.compute_stec(
            datetime(2025, 4, 21, 0),
            297.66,
            82.49,
            78.11,
            8.23,
            54.29,
            20_281_546.18,
        )
        self.assertAlmostEqual(stec, 20.40224, delta=0.2)


if __name__ == "__main__":
    unittest.main()
