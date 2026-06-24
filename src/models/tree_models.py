"""Tabular baseline factories for ray-wise STEC regression."""

from __future__ import annotations


def build_ridge(alpha: float = 1.0, **kwargs):
    """Build a Ridge pipeline with median imputation and feature scaling."""

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, **kwargs)),
        ]
    )


def build_random_forest(**kwargs):
    """Build a RandomForest pipeline with median imputation."""

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    defaults = {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "min_samples_leaf": 2,
    }
    defaults.update(kwargs)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(**defaults)),
        ]
    )
