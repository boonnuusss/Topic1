"""Train RayMLP as a simulation-backed STEC ray model.

Training uses only simulated slant-ray STEC labels.  VTEC is evaluated after
training by querying the STEC model with a vertical target-ray geometry; the IRI
VTEC reference is never used as a training label.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .evaluate import build_vtec_metrics
    from .geometry import compute_slant_factor, geodetic_to_ecef
    from .metrics import regression_metrics, vtec_error_metrics
    from .models.mlp import build_ray_mlp
    from .train_mlp import (
        choose_device,
        make_loader,
        predict_scaled,
        save_checkpoint,
        set_random_seed,
        standardize_xy,
        train_loop,
    )
    from .train_tree import (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_TABLE_DIR,
        choose_split,
        read_table,
        save_table,
        write_json,
    )
except ImportError:
    from evaluate import build_vtec_metrics
    from geometry import compute_slant_factor, geodetic_to_ecef
    from metrics import regression_metrics, vtec_error_metrics
    from models.mlp import build_ray_mlp
    from train_mlp import (
        choose_device,
        make_loader,
        predict_scaled,
        save_checkpoint,
        set_random_seed,
        standardize_xy,
        train_loop,
    )
    from train_tree import (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_TABLE_DIR,
        choose_split,
        read_table,
        save_table,
        write_json,
    )


DEFAULT_INPUT = "datasets/processed/iri_simulation_rayvtec_20240505.parquet"
SOURCE_GEOMETRY_COLUMNS = [
    "rec_lon",
    "rec_lat",
    "rec_h",
    "sat_lon",
    "sat_lat",
    "sat_h",
    "ipp_lon",
    "ipp_lat",
    "elevation",
    "azimuth_sin",
    "azimuth_cos",
    "slant_factor",
    "ray_length",
    "los_x",
    "los_y",
    "los_z",
]
TIME_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "local_time_ipp_sin",
    "local_time_ipp_cos",
]
CONSTELLATION_COLUMNS = [
    "constellation_gps",
    "constellation_glonass",
    "constellation_galileo",
    "constellation_bds",
    "constellation_sbas",
    "constellation_qzss",
]
TARGET_GEOMETRY_COLUMNS = [
    "target_rec_lon",
    "target_rec_lat",
    "target_rec_h",
    "target_sat_lon",
    "target_sat_lat",
    "target_sat_h",
    "target_ipp_lon",
    "target_ipp_lat",
    "target_elevation",
    "target_azimuth_sin",
    "target_azimuth_cos",
    "target_slant_factor",
    "target_ray_length",
    "target_los_x",
    "target_los_y",
    "target_los_z",
]


def _haversine_km(lon1, lat1, lon2, lat2):
    """Vectorized great-circle IPP distance in km.

    All inputs are numpy arrays in degrees.  Returns km.
    """
    R = 6371.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arctan2(np.sqrt(np.maximum(a, 0)), np.sqrt(np.maximum(1.0 - a, 1e-15)))


def build_nearby_pairs(
    df: pd.DataFrame,
    *,
    source_stec_column: str,
    slant_target_column: str,
    max_ipp_distance_km: float = 200.0,
    max_neighbors: int = 20,
    include_identity: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build training pairs from rays whose IPPs lie within *max_ipp_distance_km*.

    Ionospheric electron density has a spatial correlation length of roughly
    200–500 km.  Pairs with IPPs farther apart are essentially uncorrelated
    and would contribute noise rather than signal.  This keeps pair count at
    *O(N × avg_neighbors)* instead of *O(N²)*.

    Returns
    -------
    X : np.ndarray, shape (approx N × k, 45), dtype float64
    y : np.ndarray, shape (approx N × k,), dtype float64
    feature_cols : list[str]
    """
    n = len(df)
    if n < 2:
        raise ValueError("Nearby-pairs training requires at least 2 rays.")

    ipp_lon = df["ipp_lon"].to_numpy(dtype=np.float64)
    ipp_lat = df["ipp_lat"].to_numpy(dtype=np.float64)
    source_geom = df[SOURCE_GEOMETRY_COLUMNS].to_numpy(dtype=np.float64)
    time_arr = df[TIME_COLUMNS].to_numpy(dtype=np.float64)
    const_arr = df[CONSTELLATION_COLUMNS].to_numpy(dtype=np.float64)
    source_stec_vals = df[source_stec_column].to_numpy(dtype=np.float64)
    target_stec_vals = df[slant_target_column].to_numpy(dtype=np.float64)

    src_parts = []
    tgt_parts = []

    for i in range(n):
        dists = _haversine_km(ipp_lon[i], ipp_lat[i], ipp_lon, ipp_lat)
        mask = dists <= max_ipp_distance_km
        if include_identity and not mask[i]:
            mask[i] = True
        candidates = np.where(mask)[0]
        if len(candidates) > max_neighbors:
            order = np.argsort(dists[candidates])
            candidates = candidates[order[:max_neighbors]]
        src_parts.append(np.full(len(candidates), i, dtype=np.intp))
        tgt_parts.append(candidates.astype(np.intp))

    src_idx = np.concatenate(src_parts)
    tgt_idx = np.concatenate(tgt_parts)

    X_source = source_geom[src_idx]
    X_time = time_arr[src_idx]
    X_const = const_arr[src_idx]
    X_target = source_geom[tgt_idx]
    X_stec = source_stec_vals[src_idx].reshape(-1, 1)

    X = np.column_stack([X_source, X_time, X_const, X_target, X_stec])
    y = target_stec_vals[tgt_idx]

    feature_cols = (
        [f"source_{c}" for c in SOURCE_GEOMETRY_COLUMNS]
        + list(TIME_COLUMNS)
        + list(CONSTELLATION_COLUMNS)
        + list(TARGET_GEOMETRY_COLUMNS)
        + ["source_stec"]
    )

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[valid], y[valid], feature_cols


def build_all_to_all_pairs(
    df: pd.DataFrame,
    *,
    source_stec_column: str,
    slant_target_column: str,
    chunk_size: int = 300,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build all-to-all source-to-target ray pairs for a single epoch.

    For *N* training rays this produces *N²* training pairs.  Each pair maps::

        source geometry + source time + source constellation
        + target geometry + source STEC  →  target STEC

    The source context tells the model what was observed from one ray.
    Changing the target geometry while keeping the same source context
    forces the model to learn the geometrical relationship between
    different rays, which is the core of STEC-to-VTEC conversion.

    Returns
    -------
    X : np.ndarray, shape (N², 45), dtype float64
    y : np.ndarray, shape (N²,), dtype float64
    feature_cols : list[str]
        Column names in the same order as the existing pair convention.
    """
    n = len(df)
    if n < 2:
        raise ValueError("All-pairs training requires at least 2 training rays.")

    source_geom = df[SOURCE_GEOMETRY_COLUMNS].to_numpy(dtype=np.float64)
    time_arr = df[TIME_COLUMNS].to_numpy(dtype=np.float64)
    const_arr = df[CONSTELLATION_COLUMNS].to_numpy(dtype=np.float64)
    source_stec = df[source_stec_column].to_numpy(dtype=np.float64)
    target_stec = df[slant_target_column].to_numpy(dtype=np.float64)

    feature_cols = (
        [f"source_{c}" for c in SOURCE_GEOMETRY_COLUMNS]
        + list(TIME_COLUMNS)
        + list(CONSTELLATION_COLUMNS)
        + list(TARGET_GEOMETRY_COLUMNS)
        + ["source_stec"]
    )

    n_src = len(SOURCE_GEOMETRY_COLUMNS)
    n_tm = len(TIME_COLUMNS)
    n_cn = len(CONSTELLATION_COLUMNS)
    n_tgt = len(TARGET_GEOMETRY_COLUMNS)
    n_feat = n_src + n_tm + n_cn + n_tgt + 1

    total = n * n
    X = np.empty((total, n_feat), dtype=np.float64)
    y = np.empty(total, dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        m = end - start
        i0 = start * n
        i1 = end * n

        X[i0:i1, :n_src] = np.repeat(source_geom[start:end], n, axis=0)
        X[i0:i1, n_src : n_src + n_tm] = np.repeat(time_arr[start:end], n, axis=0)
        off = n_src + n_tm
        X[i0:i1, off : off + n_cn] = np.repeat(const_arr[start:end], n, axis=0)
        off += n_cn
        X[i0:i1, off : off + n_tgt] = np.tile(source_geom, (m, 1))
        off += n_tgt
        X[i0:i1, off] = np.repeat(source_stec[start:end], n)
        y[i0:i1] = np.tile(target_stec, m)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not valid.all():
        X = X[valid]
        y = y[valid]
    return X, y, feature_cols


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Simulation RayMLP input is missing columns: {missing}")


def add_missing_constellation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure optional constellation one-hot columns exist."""

    out = df.copy()
    for column in CONSTELLATION_COLUMNS:
        if column not in out.columns:
            out[column] = 0
    return out


def add_mapping_aliases(df: pd.DataFrame, *, mapping_column: str) -> pd.DataFrame:
    """Make feature names compatible with the standard ray-wise table."""

    out = df.copy()
    if "slant_factor" not in out.columns and mapping_column in out.columns:
        out["slant_factor"] = out[mapping_column]
    return out


def build_vertical_target_geometry(
    df: pd.DataFrame,
    *,
    vertical_top_km: float,
) -> pd.DataFrame:
    """Build target-ray geometry for the vertical VTEC special ray."""

    lon = df["ipp_lon"].to_numpy(float)
    lat = df["ipp_lat"].to_numpy(float)
    rec_x, rec_y, rec_z = geodetic_to_ecef(lon, lat, np.zeros(len(df)))
    sat_x, sat_y, sat_z = geodetic_to_ecef(
        lon,
        lat,
        np.full(len(df), vertical_top_km * 1000.0),
    )
    rec = np.column_stack((rec_x, rec_y, rec_z))
    sat = np.column_stack((sat_x, sat_y, sat_z))
    rel = sat - rec
    ray_length = np.linalg.norm(rel, axis=1)
    los = rel / ray_length[:, None]

    target = pd.DataFrame(index=df.index)
    target["target_rec_lon"] = df["ipp_lon"].to_numpy(float)
    target["target_rec_lat"] = df["ipp_lat"].to_numpy(float)
    target["target_rec_h"] = 0.0
    target["target_sat_lon"] = df["ipp_lon"].to_numpy(float)
    target["target_sat_lat"] = df["ipp_lat"].to_numpy(float)
    target["target_sat_h"] = float(vertical_top_km)
    target["target_ipp_lon"] = df["ipp_lon"].to_numpy(float)
    target["target_ipp_lat"] = df["ipp_lat"].to_numpy(float)
    target["target_elevation"] = 90.0
    target["target_azimuth_sin"] = 0.0
    target["target_azimuth_cos"] = 1.0
    target["target_slant_factor"] = compute_slant_factor(np.full(len(df), 90.0))
    target["target_ray_length"] = ray_length
    target["target_los_x"] = los[:, 0]
    target["target_los_y"] = los[:, 1]
    target["target_los_z"] = los[:, 2]
    return target


def build_slant_target_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Build target geometry equal to the source slant ray."""

    mapping = {
        "rec_lon": "target_rec_lon",
        "rec_lat": "target_rec_lat",
        "rec_h": "target_rec_h",
        "sat_lon": "target_sat_lon",
        "sat_lat": "target_sat_lat",
        "sat_h": "target_sat_h",
        "ipp_lon": "target_ipp_lon",
        "ipp_lat": "target_ipp_lat",
        "elevation": "target_elevation",
        "azimuth_sin": "target_azimuth_sin",
        "azimuth_cos": "target_azimuth_cos",
        "slant_factor": "target_slant_factor",
        "ray_length": "target_ray_length",
        "los_x": "target_los_x",
        "los_y": "target_los_y",
        "los_z": "target_los_z",
    }
    target = pd.DataFrame(index=df.index)
    for source, output in mapping.items():
        target[output] = df[source].to_numpy(float)
    return target


def build_pair_examples(
    df: pd.DataFrame,
    *,
    source_stec_column: str,
    target_column: str,
    pair_role: str,
    target_geometry: pd.DataFrame,
) -> pd.DataFrame:
    """Build source-ray to target-ray STEC examples."""

    parts = [
        df[SOURCE_GEOMETRY_COLUMNS].add_prefix("source_"),
        df[TIME_COLUMNS],
        df[CONSTELLATION_COLUMNS],
        target_geometry[TARGET_GEOMETRY_COLUMNS],
    ]
    out = pd.concat(parts, axis=1)
    out["source_stec"] = pd.to_numeric(df[source_stec_column], errors="coerce")
    out["target_stec"] = pd.to_numeric(df[target_column], errors="coerce")
    out["pair_role"] = pair_role
    out["source_row_id"] = df.get("source_row_id", pd.Series(df.index, index=df.index))
    return out


def build_training_pairs(
    df: pd.DataFrame,
    *,
    source_stec_column: str,
    slant_target_column: str,
) -> pd.DataFrame:
    """Create training pairs only for source slant to original slant rays."""

    pairs = build_pair_examples(
        df,
        source_stec_column=source_stec_column,
        target_column=slant_target_column,
        pair_role="source_slant_to_same_slant",
        target_geometry=build_slant_target_geometry(df),
    )
    numeric_columns = [column for column in pairs.columns if column != "pair_role"]
    finite = np.isfinite(pairs[numeric_columns].to_numpy(float)).all(axis=1)
    return pairs.loc[finite].copy()


def feature_columns_from_pairs(pairs: pd.DataFrame) -> list[str]:
    """Return model feature columns from a pair table."""

    excluded = {"target_stec", "pair_role", "source_row_id"}
    return [column for column in pairs.columns if column not in excluded]


def predict_pairs(
    *,
    model,
    pairs: pd.DataFrame,
    feature_cols: list[str],
    x_imputer,
    x_scaler,
    y_scaler,
    device,
) -> np.ndarray:
    """Predict target STEC for pair examples."""

    x = pairs.loc[:, feature_cols].to_numpy(dtype=float)
    x_scaled = x_scaler.transform(x_imputer.transform(x))
    pred_scaled = predict_scaled(model, x_scaled, device)
    return y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


def choose_simulation_split(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Choose the train/test split for simulation RayMLP experiments."""

    if args.test_keys_from and args.test_timestamp:
        raise ValueError("--test-keys-from and --test-timestamp are mutually exclusive.")

    if args.test_keys_from:
        key_path = Path(args.test_keys_from)
        key_df = read_table(key_path)
        key_columns = [
            item.strip() for item in args.test_key_columns.split(",") if item.strip()
        ]
        if not key_columns:
            raise ValueError("--test-key-columns must contain at least one column.")
        require_columns(df, key_columns)
        require_columns(key_df, key_columns)

        left = df.copy()
        right = key_df.loc[:, key_columns].drop_duplicates().copy()
        if "timestamp" in key_columns:
            left["timestamp"] = pd.to_datetime(left["timestamp"])
            right["timestamp"] = pd.to_datetime(right["timestamp"])

        marker = "_ray_mlp_test_key"
        merged = left.reset_index().merge(
            right.assign(**{marker: True}),
            on=key_columns,
            how="left",
            sort=False,
        )
        test_mask = merged[marker].fillna(False).to_numpy(bool)
        train_df = df.loc[~test_mask].copy()
        test_df = df.loc[test_mask].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Key holdout produced train={len(train_df)} and "
                f"test={len(test_df)} rows."
            )
        return train_df, test_df, {
            "type": "key_holdout",
            "test_keys_from": str(key_path),
            "key_columns": key_columns,
            "n_unique_test_keys": int(len(right)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        }

    if args.test_timestamp:
        timestamp = pd.Timestamp(args.test_timestamp)
        ts = pd.to_datetime(df["timestamp"])
        test_mask = ts == timestamp
        train_df = df.loc[~test_mask].copy()
        test_df = df.loc[test_mask].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Timestamp holdout {timestamp} produced train={len(train_df)} "
                f"and test={len(test_df)} rows."
            )
        return train_df, test_df, {
            "type": "timestamp_holdout",
            "test_timestamp": timestamp.isoformat(),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        }

    return choose_split(
        df,
        args.split,
        test_size=args.test_size,
        seed=args.seed,
        train_end=args.train_end,
    )


def train_ray_mlp_simulation(args: argparse.Namespace) -> dict[str, Any]:
    """Train RayMLP on IRI simulation pairs and write conversion outputs."""

    set_random_seed(args.seed)
    input_path = Path(args.input)
    table_dir = Path(args.table_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = args.run_name or f"ray_mlp_sim_{args.split}"

    df = read_table(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    df = add_mapping_aliases(df, mapping_column=args.mapping_column)
    df = add_missing_constellation_columns(df)
    required = (
        SOURCE_GEOMETRY_COLUMNS
        + TIME_COLUMNS
        + [
            args.source_stec_column,
            args.slant_target_column,
            args.vtec_target_column,
            args.baseline_vtec_column,
            args.mapping_column,
        ]
    )
    require_columns(df, required)

    train_df, test_df, split_metadata = choose_simulation_split(df, args)

    # ── training pair construction ──────────────────────────────────
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    use_identity_pairs = not args.all_pairs and args.nearby_km <= 0
    use_numpy_pairs = args.all_pairs or args.nearby_km > 0

    if args.all_pairs:
        X_train, y_train, feature_cols = build_all_to_all_pairs(
            train_df,
            source_stec_column=args.source_stec_column,
            slant_target_column=args.slant_target_column,
            chunk_size=args.all_pairs_chunk_size,
        )
        pair_kind = "all_to_all"
    elif args.nearby_km > 0:
        X_train, y_train, feature_cols = build_nearby_pairs(
            train_df,
            source_stec_column=args.source_stec_column,
            slant_target_column=args.slant_target_column,
            max_ipp_distance_km=args.nearby_km,
            max_neighbors=args.nearby_max_neighbors,
            include_identity=not args.nearby_no_identity,
        )
        pair_kind = "ipp_nearby"
    else:
        train_pairs = build_training_pairs(
            train_df,
            source_stec_column=args.source_stec_column,
            slant_target_column=args.slant_target_column,
        )
        if train_pairs.empty:
            raise ValueError(
                "No finite RayMLP simulation training pairs were created."
            )
        feature_cols = feature_columns_from_pairs(train_pairs)
        x_train = train_pairs.loc[:, feature_cols]
        y_train = train_pairs["target_stec"]
        if args.target_mode == "source_ratio":
            source_for_pairs = train_pairs["source_stec"].to_numpy(float)
            target_for_pairs = train_pairs["target_stec"].to_numpy(float)
            valid_ratio = (
                np.isfinite(source_for_pairs)
                & np.isfinite(target_for_pairs)
                & (np.abs(source_for_pairs) > 1e-12)
            )
            train_pairs = train_pairs.loc[valid_ratio].copy()
            train_pairs["target_stec"] = (
                target_for_pairs[valid_ratio] / source_for_pairs[valid_ratio]
            )
            if train_pairs.empty:
                raise ValueError("No finite source-ratio training pairs were created.")
            x_train = train_pairs.loc[:, feature_cols]
            y_train = train_pairs["target_stec"]
        (
            x_train_scaled,
            y_train_scaled,
            _,
            x_imputer,
            x_scaler,
            y_scaler,
        ) = standardize_xy(x_train, y_train, x_train)
        pair_kind = "source_slant_to_same_slant"

    if use_numpy_pairs and args.target_mode == "source_ratio":
        source_for_pairs = X_train[:, -1].astype(np.float64)
        valid_ratio = (
            np.isfinite(source_for_pairs)
            & np.isfinite(y_train)
            & (np.abs(source_for_pairs) > 1e-12)
        )
        X_train = X_train[valid_ratio]
        y_train = y_train[valid_ratio] / source_for_pairs[valid_ratio]

    if use_numpy_pairs:
        if len(X_train) < 2:
            raise ValueError("Training requires at least two valid pairs.")
        x_imputer = SimpleImputer(strategy="median")
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_imp = x_imputer.fit_transform(X_train)
        x_train_scaled = x_scaler.fit_transform(X_imp)
        y_train_scaled = y_scaler.fit_transform(
            y_train.reshape(-1, 1)
        ).ravel()
        n_train_pairs = int(len(X_train))
    else:
        n_train_pairs = int(len(train_pairs))

    # ── training ────────────────────────────────────────────────────
    device = choose_device(args.device)
    loader = make_loader(
        x_train_scaled, y_train_scaled, args.batch_size, args.seed
    )
    model = build_ray_mlp(input_dim=len(feature_cols), dropout=args.dropout)
    history = train_loop(
        model,
        loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    # ── train metrics (subset for large pair sets to keep it fast) ──
    if use_numpy_pairs:
        rng = np.random.default_rng(args.seed)
        n_metric = min(int(args.all_pairs_train_metric_samples), len(X_train))
        metric_idx = rng.choice(len(X_train), size=n_metric, replace=False)
        x_metric = X_train[metric_idx]
        y_metric = y_train[metric_idx]
        x_metric_scaled = x_scaler.transform(x_imputer.transform(x_metric))
        train_pred = y_scaler.inverse_transform(
            predict_scaled(model, x_metric_scaled, device).reshape(-1, 1)
        ).ravel()
        train_stec_metrics = regression_metrics(y_metric, train_pred)
        train_stec_metrics["metric_subset_size"] = n_metric
        train_stec_metrics["metric_subset_of_total_pairs"] = int(len(X_train))
    else:
        train_pred = y_scaler.inverse_transform(
            predict_scaled(model, x_train_scaled, device).reshape(-1, 1)
        ).ravel()
        train_stec_metrics = regression_metrics(y_train, train_pred)

    # ── test predictions ───────────────────────────────────────────
    test_vertical_pairs = build_pair_examples(
        test_df,
        source_stec_column=args.source_stec_column,
        target_column=args.source_stec_column,
        pair_role="source_slant_to_vertical",
        target_geometry=build_vertical_target_geometry(
            test_df,
            vertical_top_km=args.vertical_top_km,
        ),
    )
    test_slant_pairs = build_pair_examples(
        test_df,
        source_stec_column=args.source_stec_column,
        target_column=args.slant_target_column,
        pair_role="source_slant_to_same_slant",
        target_geometry=build_slant_target_geometry(test_df),
    )

    vtec_pred = predict_pairs(
        model=model,
        pairs=test_vertical_pairs,
        feature_cols=feature_cols,
        x_imputer=x_imputer,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
    )
    slant_pred = predict_pairs(
        model=model,
        pairs=test_slant_pairs,
        feature_cols=feature_cols,
        x_imputer=x_imputer,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
    )

    predictions = test_df.copy()
    source_stec_eval = predictions[args.source_stec_column].to_numpy(float)

    if args.target_mode == "source_ratio":
        vertical_ratio = np.maximum(vtec_pred, 0.0)
        slant_ratio = np.maximum(slant_pred, 0.0)
        predictions["stec_ray_mlp"] = source_stec_eval * slant_ratio
        predictions["ray_mlp_conversion_ratio"] = vertical_ratio
        predictions["vtec_ray_mlp_direct"] = source_stec_eval * vertical_ratio
        predictions["vtec_ray_mlp"] = predictions["vtec_ray_mlp_direct"]
    else:
        # Physical constraint: STEC must be non-negative.
        vtec_pred = np.maximum(vtec_pred, 0.0)
        slant_pred = np.maximum(slant_pred, 0.0)
        predictions["stec_ray_mlp"] = slant_pred
        predictions["vtec_ray_mlp_direct"] = vtec_pred
        denominator = predictions["stec_ray_mlp"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(
                predictions["vtec_ray_mlp_direct"].to_numpy(float),
                denominator,
                out=np.full(len(predictions), np.nan, dtype=float),
                where=np.abs(denominator) > 1e-12,
            )
        predictions["ray_mlp_conversion_ratio"] = ratio
        predictions["vtec_ray_mlp"] = source_stec_eval * ratio
    predictions["vtec_ray_mlp_error_iri"] = (
        predictions["vtec_ray_mlp"] - predictions[args.vtec_target_column]
    )
    predictions["vtec_ray_mlp_abs_error_iri"] = predictions[
        "vtec_ray_mlp_error_iri"
    ].abs()
    predictions["vtec_ray_mlp_direct_error_iri"] = (
        predictions["vtec_ray_mlp_direct"] - predictions[args.vtec_target_column]
    )
    predictions["vtec_ray_mlp_direct_abs_error_iri"] = predictions[
        "vtec_ray_mlp_direct_error_iri"
    ].abs()

    prediction_path = save_table(
        predictions,
        table_dir / f"{run_name}_predictions.{args.output_format}",
    )
    vtec_metadata = None
    if not args.skip_vtec_evaluation:
        vtec_metadata = build_vtec_metrics(
            input_path=prediction_path,
            table_dir=table_dir,
            run_name=f"{run_name}_test",
            target_column=args.vtec_target_column,
            baseline_column=args.baseline_vtec_column,
            vtec_prediction_columns=["vtec_ray_mlp", "vtec_ray_mlp_direct"],
            stec_prediction_columns=[],
            mapping_column=args.mapping_column,
            elevation_bins=[
                float(item.strip())
                for item in args.elevation_bins.split(",")
                if item.strip()
            ],
            station_min_count=args.station_min_count,
            max_rows=None,
        )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": (
            "simulation_stec_all_to_all" if args.all_pairs
            else "simulation_stec_ipp_nearby" if args.nearby_km > 0
            else "simulation_stec_ray_transform"
        ),
        "model": "ray_mlp",
        "run_name": run_name,
        "input_file": str(input_path),
        "source_stec_column": args.source_stec_column,
        "slant_target_column": args.slant_target_column,
        "vtec_target_column": args.vtec_target_column,
        "baseline_vtec_column": args.baseline_vtec_column,
        "stec_prediction_column": "stec_ray_mlp",
        "vtec_prediction_column": "vtec_ray_mlp",
        "vtec_direct_query_column": "vtec_ray_mlp_direct",
        "conversion_ratio_column": "ray_mlp_conversion_ratio",
        "conversion_interpretation": (
            "IPP-nearby source→target ray pairs used for training: "
            "source_STEC + source_geometry + target_geometry → target_STEC. "
            "Pairs are created only for rays with IPPs within "
            f"{args.nearby_km:.0f} km. VTEC is estimated after training "
            "by querying both the original slant target ray and a vertical "
            "target ray, using VTEC = source_STEC × vertical_query / slant_query."
        )
        if args.nearby_km > 0
        else (
            "All-to-all source→target ray pairs used for training: "
            "source_STEC + source_geometry + target_geometry → target_STEC. "
            "VTEC is estimated after training by querying both the original "
            "slant target ray and a vertical target ray, then applying "
            "VTEC = source_STEC × vertical_query / slant_query."
        )
        if args.all_pairs
        else (
            "The model is trained only on slant-ray STEC labels. VTEC is "
            "estimated after training by querying both the original slant "
            "target ray and a vertical target ray, then applying "
            "VTEC = source_STEC * vertical_query / slant_query."
        ),
        "vtec_reference_used_for_training": False,
        "vtec_reference_used_for_inference_features": False,
        "vertical_top_km": float(args.vertical_top_km),
        "training_pair_role": pair_kind,
        "target_mode": args.target_mode,
        "feature_columns": feature_cols,
        "n_features": int(len(feature_cols)),
        "split": split_metadata,
        "row_counts": {
            "input": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_pairs": n_train_pairs,
        },
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "seed": int(args.seed),
            "device": str(device),
            "all_pairs": bool(args.all_pairs),
            "nearby_km": float(args.nearby_km),
            "target_mode": args.target_mode,
        },
        "history": history,
        "metrics": {
            "train_pair_stec": train_stec_metrics,
            "test_slant_stec": regression_metrics(
                predictions[args.slant_target_column],
                predictions["stec_ray_mlp"],
            ),
            "test_vtec": vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions["vtec_ray_mlp"],
            ),
            "test_vtec_direct_vertical_query": vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions["vtec_ray_mlp_direct"],
            ),
            "test_slm_vtec": vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions[args.baseline_vtec_column],
            ),
        },
        "outputs": {
            "predictions": str(prediction_path),
        },
    }
    if args.target_mode == "source_ratio":
        metadata["conversion_interpretation"] = (
            "Model is trained on target_STEC/source_STEC ratios from "
            "IPP-nearby slant-ray pairs. For STEC-to-VTEC conversion, "
            "the model is queried with vertical target geometry and "
            "VTEC = source_STEC * predicted_vertical_ratio."
        )

    if vtec_metadata is not None:
        metadata["outputs"]["vtec_metrics"] = vtec_metadata["outputs"]

    checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    metadata["outputs"]["checkpoint"] = str(
        save_checkpoint(
            checkpoint_path,
            model=model,
            x_imputer=x_imputer,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            metadata=metadata,
        )
    )
    metrics_json_path = write_json(metadata, table_dir / f"{run_name}_metrics.json")
    summary_rows = [
        {"subset": "train", "metric_target": args.target_mode, **train_stec_metrics},
        {
            "subset": "test",
            "metric_target": args.slant_target_column,
            **regression_metrics(
                predictions[args.slant_target_column],
                predictions["stec_ray_mlp"],
            ),
        },
        {
            "subset": "test",
            "metric_target": args.vtec_target_column,
            **vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions["vtec_ray_mlp"],
            ),
        },
        {
            "subset": "test",
            "metric_target": f"{args.vtec_target_column}_direct_vertical_query",
            **vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions["vtec_ray_mlp_direct"],
            ),
        },
        {
            "subset": "test",
            "metric_target": f"slm_{args.vtec_target_column}",
            **vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions[args.baseline_vtec_column],
            ),
        },
    ]
    metrics_csv_path = save_table(
        pd.DataFrame(summary_rows),
        table_dir / f"{run_name}_metrics.csv",
    )
    metadata["outputs"]["metrics_json"] = str(metrics_json_path)
    metadata["outputs"]["metrics_csv"] = str(metrics_csv_path)
    write_json(metadata, metrics_json_path)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--source-stec-column", default="stec_iri_observable")
    parser.add_argument("--target-mode", choices=["absolute", "source_ratio"], default="absolute", help="Training target: absolute target STEC or target/source STEC ratio.")
    parser.add_argument("--slant-target-column", default="stec_iri_ray")
    parser.add_argument("--vtec-target-column", default="vtec_iri_ref")
    parser.add_argument("--baseline-vtec-column", default="vtec_slm_from_iri_stec")
    parser.add_argument("--mapping-column", default="slm_mapping_factor_iri")
    parser.add_argument("--vertical-top-km", type=float, default=2000.0)
    parser.add_argument(
        "--split",
        choices=["random", "time", "station", "prn"],
        default="random",
    )
    parser.add_argument(
        "--test-keys-from",
        help=(
            "Optional parquet/CSV table containing rows to hold out as the test "
            "set. Useful for reusing a previous test split."
        ),
    )
    parser.add_argument(
        "--test-key-columns",
        default="timestamp,station,prn",
        help="Comma-separated key columns used with --test-keys-from.",
    )
    parser.add_argument(
        "--test-timestamp",
        help="Optional UTC timestamp to hold out as the test set.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-end", help="Required for --split time.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--elevation-bins", default="10,20,30,40,50,60,70,80,90")
    parser.add_argument("--station-min-count", type=int, default=1)
    parser.add_argument("--skip-vtec-evaluation", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--nearby-km",
        type=float,
        default=0.0,
        help=(
            "Max IPP great-circle distance in km for training pairs. "
            "Only source→target pairs whose IPPs are within this distance "
            "are created. 200–500 km is a reasonable range (ionospheric "
            "correlation length).  0 (default) disables nearby-pair mode."
        ),
    )
    parser.add_argument(
        "--nearby-max-neighbors",
        type=int,
        default=30,
        help="Cap on target rays per source ray for nearby-pair mode.",
    )
    parser.add_argument(
        "--nearby-no-identity",
        action="store_true",
        help="Exclude identity (i→i) pairs from nearby-pair training.",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help=(
            "Build all-to-all source→target ray pairs for training "
            "(N² pairs for N training rays).  This lets the model learn the "
            "geometrical relationship between different rays, which is "
            "essential for STEC-to-VTEC conversion."
        ),
    )
    parser.add_argument(
        "--all-pairs-chunk-size",
        type=int,
        default=300,
        help="Source rays processed per chunk when building all-to-all pairs.",
    )
    parser.add_argument(
        "--all-pairs-train-metric-samples",
        type=int,
        default=10000,
        help="Number of training pairs sampled for metric reporting.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = train_ray_mlp_simulation(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
