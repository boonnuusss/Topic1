"""Combine IRI and NeQuick-G physical mapping metric tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iri",
        default=(
            "results/tables/"
            "physical_mapping_baselines_ut05_validation_metrics.csv"
        ),
    )
    parser.add_argument(
        "--nequickg",
        default=(
            "results/tables/"
            "nequickg_physical_mapping_baselines_ut05_validation_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/tables/iri_nequickg_physical_baseline_comparison.csv",
    )
    args = parser.parse_args(argv)

    tables = []
    for model, path_value in [("iri", args.iri), ("nequickg", args.nequickg)]:
        table = pd.read_csv(path_value)
        if "reference_model" not in table.columns:
            table.insert(0, "reference_model", model)
        tables.append(table)
    comparison = pd.concat(tables, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
