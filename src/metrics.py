"""Evaluation metrics for STEC regression experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_pred - y_true
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_samples": int(len(y_true)),
    }


def elevation_bin_metrics(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = "stec_obs",
) -> pd.DataFrame:
    bins = [10, 20, 40, 60, 90]
    labels = ["10-20", "20-40", "40-60", "60-90"]
    out = df.copy()
    out["elevation_bin"] = pd.cut(out["elevation"], bins=bins, labels=labels)
    rows = []
    for name, group in out.groupby("elevation_bin", observed=True):
        metrics = regression_metrics(group[target_col], group[pred_col])
        metrics["elevation_bin"] = str(name)
        rows.append(metrics)
    return pd.DataFrame(rows)

