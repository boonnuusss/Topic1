"""Dataset split helpers for strict STEC generalization tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) >= test_size
    return df.loc[mask].copy(), df.loc[~mask].copy()


def time_split(df: pd.DataFrame, train_end, valid_end=None):
    ts = pd.to_datetime(df["timestamp"])
    train = df.loc[ts <= pd.Timestamp(train_end)].copy()
    if valid_end is None:
        test = df.loc[ts > pd.Timestamp(train_end)].copy()
        return train, test
    valid = df.loc[(ts > pd.Timestamp(train_end)) & (ts <= pd.Timestamp(valid_end))].copy()
    test = df.loc[ts > pd.Timestamp(valid_end)].copy()
    return train, valid, test


def group_holdout_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float = 0.2,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(df[group_col].dropna().unique()))
    n_test = max(1, int(round(len(groups) * test_size)))
    test_groups = set(rng.choice(groups, size=n_test, replace=False))
    test_mask = df[group_col].isin(test_groups)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def station_holdout_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    return group_holdout_split(df, "station", test_size=test_size, seed=seed)


def prn_holdout_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    return group_holdout_split(df, "prn", test_size=test_size, seed=seed)

