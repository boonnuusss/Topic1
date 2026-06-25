"""Evaluation metrics for STEC and VTEC conversion experiments."""

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


def vtec_error_metrics(y_true, y_pred) -> dict[str, float | int]:
    """Return VTEC-space error metrics in TECU.

    The metric convention is always ``error = prediction - reference``.
    Positive bias therefore means the conversion overestimates reference VTEC.
    """

    target = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if target.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    finite = np.isfinite(target) & np.isfinite(prediction)
    n_total = int(target.size)
    n_valid = int(finite.sum())
    if n_valid == 0:
        return {
            "n_samples": 0,
            "n_missing": n_total,
            "target_mean_tecu": float("nan"),
            "prediction_mean_tecu": float("nan"),
            "bias_tecu": float("nan"),
            "median_error_tecu": float("nan"),
            "std_error_tecu": float("nan"),
            "mae_tecu": float("nan"),
            "rmse_tecu": float("nan"),
            "p05_error_tecu": float("nan"),
            "p95_error_tecu": float("nan"),
            "min_error_tecu": float("nan"),
            "max_error_tecu": float("nan"),
            "max_abs_error_tecu": float("nan"),
            "r2": float("nan"),
        }

    target = target[finite]
    prediction = prediction[finite]
    error = prediction - target
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "n_samples": n_valid,
        "n_missing": int(n_total - n_valid),
        "target_mean_tecu": float(np.mean(target)),
        "prediction_mean_tecu": float(np.mean(prediction)),
        "bias_tecu": float(np.mean(error)),
        "median_error_tecu": float(np.median(error)),
        "std_error_tecu": float(np.std(error, ddof=1)) if n_valid > 1 else 0.0,
        "mae_tecu": float(np.mean(np.abs(error))),
        "rmse_tecu": float(np.sqrt(np.mean(error**2))),
        "p05_error_tecu": float(np.quantile(error, 0.05)),
        "p95_error_tecu": float(np.quantile(error, 0.95)),
        "min_error_tecu": float(np.min(error)),
        "max_error_tecu": float(np.max(error)),
        "max_abs_error_tecu": float(np.max(np.abs(error))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def vtec_group_metrics(
    df: pd.DataFrame,
    *,
    pred_col: str,
    target_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Compute VTEC-space metrics for each group in ``group_col``."""

    rows = []
    for name, group in df.groupby(group_col, observed=True):
        metrics = vtec_error_metrics(group[target_col], group[pred_col])
        metrics[group_col] = str(name)
        rows.append(metrics)
    return pd.DataFrame(rows)


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
