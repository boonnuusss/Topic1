"""Tree-model baseline factories."""

from __future__ import annotations


def build_random_forest(**kwargs):
    from sklearn.ensemble import RandomForestRegressor

    defaults = {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "min_samples_leaf": 2,
    }
    defaults.update(kwargs)
    return RandomForestRegressor(**defaults)

