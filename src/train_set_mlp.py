"""Train a set-to-query RayMLP model.

Each training example is a query ray + a set of *K* nearby observed rays.
The model learns to predict query-ray STEC from the collective context of
surrounding observations, rather than from a single source ray.

VTEC is obtained by querying the trained model with a vertical target ray
at each test IPP, using the training-ray set as context.
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
    from .models.set_mlp import SetToQueryModel
    from .train_mlp import (
        choose_device,
        save_checkpoint,
        set_random_seed,
    )
    from .train_ray_mlp_simulation import (
        TARGET_GEOMETRY_COLUMNS,
        build_vertical_target_geometry,
        require_columns,
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
    from models.set_mlp import SetToQueryModel
    from train_mlp import (
        choose_device,
        save_checkpoint,
        set_random_seed,
    )
    from train_ray_mlp_simulation import (
        TARGET_GEOMETRY_COLUMNS,
        build_vertical_target_geometry,
        require_columns,
    )
    from train_tree import (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_TABLE_DIR,
        choose_split,
        read_table,
        save_table,
        write_json,
    )


DEFAULT_INPUT = "datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.parquet"

QUERY_GEOM_COLS = [
    "rec_lon", "rec_lat", "rec_h",
    "sat_lon", "sat_lat", "sat_h",
    "ipp_lon", "ipp_lat",
    "elevation", "azimuth_sin", "azimuth_cos",
    "slant_factor", "ray_length",
    "los_x", "los_y", "los_z",
]
CONSTELLATION_COLS = [
    "constellation_gps", "constellation_glonass",
    "constellation_galileo", "constellation_bds",
    "constellation_sbas", "constellation_qzss",
]
TIME_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
QUERY_FEATURE_COLS = QUERY_GEOM_COLS + TIME_COLS
OBS_GEO_CONST_COLS = QUERY_GEOM_COLS + TIME_COLS + CONSTELLATION_COLS
# "source_stec" appended dynamically from args


def _haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(np.maximum(a, 0)), np.sqrt(np.maximum(1 - a, 1e-15)))


def _ensure_constellation(df):
    out = df.copy()
    for col in CONSTELLATION_COLS:
        if col not in out.columns:
            out[col] = 0
    return out


# ── set-example construction ─────────────────────────────────────────────


def build_set_examples(
    df: pd.DataFrame,
    *,
    source_stec_col: str = "stec_iri_observable",
    target_stec_col: str = "stec_iri_ray",
    max_obs: int = 40,
    ipp_radius_km: float = 200.0,
    rng: np.random.Generator | None = None,
    include_self: bool = False,
    obs_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Build leave-one-out set examples for one epoch.

    Each row i: query = target geometry, obs = nearby rays j≠i, label = absolute STEC.
    """
    if obs_cols is None:
        obs_cols = OBS_GEO_CONST_COLS + [source_stec_col]
    n_obs_cols = len(obs_cols)
    n = len(df)
    ipp_lon = df["ipp_lon"].to_numpy(dtype=np.float64)
    ipp_lat = df["ipp_lat"].to_numpy(dtype=np.float64)
    q_geom = df[QUERY_FEATURE_COLS].to_numpy(dtype=np.float64)
    o_feat = df[obs_cols].to_numpy(dtype=np.float64)
    stec = df[target_stec_col].to_numpy(dtype=np.float64)

    QG = np.zeros((n, len(QUERY_FEATURE_COLS)), dtype=np.float64)
    OF = np.zeros((n, max_obs, n_obs_cols), dtype=np.float64)
    OM = np.zeros((n, max_obs), dtype=bool)
    OD = np.zeros((n, max_obs), dtype=np.float64) + np.inf
    Y  = np.zeros(n, dtype=np.float64)
    n_neighbors = np.zeros(n, dtype=int)

    for i in range(n):
        dists = _haversine_km(ipp_lon[i], ipp_lat[i], ipp_lon, ipp_lat)
        if not include_self:
            dists[i] = np.inf
        candidates = np.where(dists <= ipp_radius_km)[0]
        if len(candidates) > max_obs:
            order = np.argsort(dists[candidates])
            candidates = candidates[order[:max_obs]]
        k = len(candidates)
        n_neighbors[i] = k
        QG[i] = q_geom[i]
        Y[i]  = stec[i]
        if k > 0:
            OF[i, :k] = o_feat[candidates]
            OM[i, :k] = True
            OD[i, :k] = dists[candidates]

    valid = np.isfinite(Y) & (n_neighbors > 0)
    extra = {
        "mean_neighbors": float(np.mean(n_neighbors[valid])),
        "min_neighbors": int(np.min(n_neighbors[valid])),
        "max_neighbors": int(np.max(n_neighbors[valid])),
        "n_total": int(n),
        "n_valid": int(valid.sum()),
    }
    QG, OF, OM, OD, Y = QG[valid], OF[valid], OM[valid], OD[valid], Y[valid]
    return QG, OF, OM, OD, Y, extra


def build_observation_sets(
    query_df: pd.DataFrame,
    context_df: pd.DataFrame,
    *,
    obs_cols: list[str],
    max_obs: int,
    ipp_radius_km: float,
    include_query_observation: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build nearby observed-ray sets for query rows.

    ``context_df`` contains observable STEC inputs. It may be the training split
    only, or all currently observed rays. When ``include_query_observation`` is
    false, rows with the same dataframe index as the query are excluded.
    """
    n_query = len(query_df)
    n_obs_cols = len(obs_cols)
    context_ipp_lon = context_df["ipp_lon"].to_numpy(dtype=np.float64)
    context_ipp_lat = context_df["ipp_lat"].to_numpy(dtype=np.float64)
    context_index = context_df.index.to_numpy()
    query_ipp_lon = query_df["ipp_lon"].to_numpy(dtype=np.float64)
    query_ipp_lat = query_df["ipp_lat"].to_numpy(dtype=np.float64)
    query_index = query_df.index.to_numpy()
    context_obs_feat = context_df[obs_cols].to_numpy(dtype=np.float64)

    obs_feat = np.zeros((n_query, max_obs, n_obs_cols), dtype=np.float64)
    obs_mask = np.zeros((n_query, max_obs), dtype=bool)
    obs_dist = np.zeros((n_query, max_obs), dtype=np.float64) + np.inf
    n_neighbors = np.zeros(n_query, dtype=int)

    for i in range(n_query):
        dists = _haversine_km(
            query_ipp_lon[i], query_ipp_lat[i], context_ipp_lon, context_ipp_lat
        )
        candidate_mask = dists <= ipp_radius_km
        if not include_query_observation:
            candidate_mask &= context_index != query_index[i]
        candidates = np.where(candidate_mask)[0]
        if len(candidates) > max_obs:
            order = np.argsort(dists[candidates])
            candidates = candidates[order[:max_obs]]
        k = len(candidates)
        n_neighbors[i] = k
        if k > 0:
            obs_feat[i, :k] = context_obs_feat[candidates]
            obs_mask[i, :k] = True
            obs_dist[i, :k] = dists[candidates]

    valid = n_neighbors > 0
    stats = {
        "mean_neighbors": float(np.mean(n_neighbors[valid])) if valid.any() else 0.0,
        "min_neighbors": int(np.min(n_neighbors[valid])) if valid.any() else 0,
        "max_neighbors": int(np.max(n_neighbors[valid])) if valid.any() else 0,
        "n_total": int(n_query),
        "n_valid": int(valid.sum()),
    }
    return obs_feat, obs_mask, obs_dist, stats


# ── training utilities ────────────────────────────────────────────────────


import torch
from torch.utils.data import Dataset, DataLoader


class _SetDataset(Dataset):
    def __init__(self, q_geom, obs_feat, obs_mask, obs_dists, labels):
        self._q = q_geom.copy()
        self._o = obs_feat.copy()
        self._m = obs_mask.copy()
        self._d = obs_dists.copy()
        self._y = labels.copy()
    def __len__(self): return len(self._q)
    def __getitem__(self, idx): return (self._q[idx], self._o[idx], self._m[idx], self._d[idx], self._y[idx])


def _collate(batch):
    qg, of, om, od, y = zip(*batch)
    return (
        torch.tensor(np.stack(qg), dtype=torch.float32),
        torch.tensor(np.stack(of), dtype=torch.float32),
        torch.tensor(np.stack(om), dtype=torch.bool),
        torch.tensor(np.stack(od), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1),
    )


def _make_loader(ds, batch_size, seed, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, collate_fn=_collate, generator=g)


def _predict_set(model, device, q_geom, obs_feat, obs_mask, obs_dists, batch_size=512):
    preds = []
    model.eval()
    for start in range(0, len(q_geom), batch_size):
        end = min(start + batch_size, len(q_geom))
        qg = torch.tensor(q_geom[start:end], dtype=torch.float32, device=device)
        of_ = torch.tensor(obs_feat[start:end], dtype=torch.float32, device=device)
        om = torch.tensor(obs_mask[start:end], dtype=torch.bool, device=device)
        od = torch.tensor(obs_dists[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            p = model(qg, of_, om, od)
        preds.append(p.cpu().numpy())
    return np.concatenate(preds, axis=0).ravel()


# ── main training function ───────────────────────────────────────────────


def train_set_mlp(args: argparse.Namespace) -> dict[str, Any]:
    set_random_seed(args.seed)
    input_path = Path(args.input)
    table_dir = Path(args.table_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = args.run_name or f"set_mlp_{args.split}"

    df = read_table(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    df = _ensure_constellation(df)

    obs_feature_cols = OBS_GEO_CONST_COLS + [args.source_stec_column]
    n_obs_cols = len(obs_feature_cols)

    required = (QUERY_FEATURE_COLS + obs_feature_cols
                + [args.source_stec_column, args.slant_target_column,
                   args.vtec_target_column, args.baseline_vtec_column,
                   args.mapping_column])
    require_columns(df, required)

    train_df, test_df, split_metadata = choose_simulation_split(df, args)

    rng = np.random.default_rng(args.seed)
    QG, OF, OM, OD, Y, set_extra = build_set_examples(
        train_df,
        source_stec_col=args.source_stec_column,
        target_stec_col=args.slant_target_column,
        max_obs=args.max_obs,
        ipp_radius_km=args.nearby_km if args.nearby_km > 0 else 200.0,
        rng=rng,
        include_self=args.include_query_observation,
        obs_cols=obs_feature_cols,
    )
    if len(Y) < 2:
        raise ValueError("Set training requires at least 2 valid examples.")

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    obs_flat = OF.reshape(-1, OF.shape[-1])
    valid_rows = OM.ravel()
    obs_valid = obs_flat[valid_rows]

    q_imp = SimpleImputer(strategy="median")
    q_scaler = StandardScaler()
    o_imp = SimpleImputer(strategy="median")
    o_scaler = StandardScaler()
    y_scaler = StandardScaler()

    QG_imp = q_imp.fit_transform(QG)
    QG_scaled = q_scaler.fit_transform(QG_imp).astype(np.float64)

    o_imp.fit(obs_valid)
    o_scaler.fit(o_imp.transform(obs_valid))
    OF_imp = np.zeros_like(OF)
    for i in range(len(OF)):
        nv = int(OM[i].sum())
        if nv > 0:
            OF_imp[i, :nv] = o_imp.transform(OF[i, :nv])
    OF_scaled = np.zeros_like(OF)
    for i in range(len(OF)):
        nv = int(OM[i].sum())
        if nv > 0:
            OF_scaled[i, :nv] = o_scaler.transform(OF_imp[i, :nv])

    Y_scaled = y_scaler.fit_transform(Y.reshape(-1, 1)).ravel().astype(np.float64)

    device = choose_device(args.device)
    train_ds = _SetDataset(QG_scaled, OF_scaled, OM, OD, Y_scaled)
    loader = _make_loader(train_ds, args.batch_size, args.seed, shuffle=True)

    model = SetToQueryModel(
        dropout=args.dropout,
        spatial_length_scale_km=args.spatial_sigma_km,
        n_query_features=len(QUERY_FEATURE_COLS),
        n_obs_features=n_obs_cols,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    criterion = torch.nn.MSELoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for qg, of_, om, od, yb in loader:
            qg = qg.to(device)
            of_ = of_.to(device)
            om = om.to(device)
            od = od.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(qg, of_, om, od)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            bs = len(qg)
            total_loss += float(loss.detach().cpu()) * bs
            total_count += bs
        history.append({"epoch": epoch, "train_loss": total_loss / max(total_count, 1)})

    train_pred_scaled = _predict_set(model, device, QG_scaled, OF_scaled, OM, OD, batch_size=args.batch_size)
    train_pred = y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
    train_stec_metrics = regression_metrics(Y, train_pred)

    # ── test predictions ─────────────────────────────────────────────
    test_slant_geom = test_df[QUERY_FEATURE_COLS].to_numpy(dtype=np.float64)
    n_test = len(test_df)
    context_df = df if args.inference_context == "all_observed" else train_df
    radius_km = args.nearby_km if args.nearby_km > 0 else 200.0
    test_OF, test_OM, test_OD, test_context_stats = build_observation_sets(
        test_df,
        context_df,
        obs_cols=obs_feature_cols,
        max_obs=args.max_obs,
        ipp_radius_km=radius_km,
        include_query_observation=args.include_query_observation,
    )

    test_OG_imp = q_imp.transform(test_slant_geom)
    test_OG_scaled = q_scaler.transform(test_OG_imp).astype(np.float64)
    test_OF_imp = np.zeros_like(test_OF)
    for i in range(n_test):
        nv = int(test_OM[i].sum())
        if nv > 0:
            test_OF_imp[i, :nv] = o_imp.transform(test_OF[i, :nv])
    test_OF_scaled = np.zeros_like(test_OF)
    for i in range(n_test):
        nv = int(test_OM[i].sum())
        if nv > 0:
            test_OF_scaled[i, :nv] = o_scaler.transform(test_OF_imp[i, :nv])

    slant_pred_scaled = _predict_set(model, device, test_OG_scaled, test_OF_scaled, test_OM, test_OD, batch_size=args.batch_size)
    slant_pred = y_scaler.inverse_transform(slant_pred_scaled.reshape(-1, 1)).ravel()

    # Vertical query
    test_vert_base = build_vertical_target_geometry(test_df, vertical_top_km=args.vertical_top_km)[TARGET_GEOMETRY_COLUMNS].to_numpy(dtype=np.float64)
    test_vert_time = test_df[TIME_COLS].to_numpy(dtype=np.float64)
    test_vert_geom = np.column_stack([test_vert_base, test_vert_time])
    tv_n = len(test_df)
    tv_OF = test_OF
    tv_OM = test_OM
    tv_OD = test_OD

    tv_OG_imp = q_imp.transform(test_vert_geom)
    tv_OG_scaled = q_scaler.transform(tv_OG_imp).astype(np.float64)
    tv_OF_imp = np.zeros_like(tv_OF)
    for i in range(tv_n):
        nv = int(tv_OM[i].sum())
        if nv > 0:
            tv_OF_imp[i, :nv] = o_imp.transform(tv_OF[i, :nv])
    tv_OF_scaled = np.zeros_like(tv_OF)
    for i in range(tv_n):
        nv = int(tv_OM[i].sum())
        if nv > 0:
            tv_OF_scaled[i, :nv] = o_scaler.transform(tv_OF_imp[i, :nv])

    vtec_pred_scaled = _predict_set(model, device, tv_OG_scaled, tv_OF_scaled, tv_OM, tv_OD, batch_size=args.batch_size)
    vtec_pred = y_scaler.inverse_transform(vtec_pred_scaled.reshape(-1, 1)).ravel()

    # clamp
    slant_pred = np.maximum(slant_pred, 0.0)
    vtec_pred = np.maximum(vtec_pred, 0.0)

    # ── predictions table ────────────────────────────────────────────
    predictions = test_df.copy()
    predictions["stec_set_mlp"] = slant_pred
    predictions["vtec_set_mlp_direct"] = vtec_pred
    denom = predictions["stec_set_mlp"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(vtec_pred, denom, out=np.full(len(predictions), np.nan, dtype=float), where=np.abs(denom) > 1e-12)
    predictions["set_mlp_conversion_ratio"] = ratio
    predictions["vtec_set_mlp"] = predictions[args.source_stec_column].to_numpy(float) * ratio
    predictions["vtec_set_mlp_error_iri"] = predictions["vtec_set_mlp"] - predictions[args.vtec_target_column]
    predictions["vtec_set_mlp_abs_error_iri"] = predictions["vtec_set_mlp_error_iri"].abs()
    predictions["vtec_set_mlp_direct_error_iri"] = predictions["vtec_set_mlp_direct"] - predictions[args.vtec_target_column]
    predictions["vtec_set_mlp_direct_abs_error_iri"] = predictions["vtec_set_mlp_direct_error_iri"].abs()

    pred_path = save_table(predictions, table_dir / f"{run_name}_predictions.{args.output_format}")

    vtec_meta = None
    if not args.skip_vtec_evaluation:
        vtec_meta = build_vtec_metrics(
            input_path=pred_path, table_dir=table_dir, run_name=f"{run_name}_test",
            target_column=args.vtec_target_column, baseline_column=args.baseline_vtec_column,
            vtec_prediction_columns=["vtec_set_mlp", "vtec_set_mlp_direct"],
            stec_prediction_columns=[], mapping_column=args.mapping_column,
            elevation_bins=[float(s.strip()) for s in args.elevation_bins.split(",") if s.strip()],
            station_min_count=args.station_min_count, max_rows=None,
        )

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "simulation_set_to_query", "model": "set_ray_mlp", "run_name": run_name,
        "input_file": str(input_path),
        "source_stec_column": args.source_stec_column,
        "slant_target_column": args.slant_target_column,
        "vtec_target_column": args.vtec_target_column,
        "baseline_vtec_column": args.baseline_vtec_column,
        "stec_prediction_column": "stec_set_mlp",
        "vtec_prediction_column": "vtec_set_mlp",
        "vtec_direct_query_column": "vtec_set_mlp_direct",
        "conversion_ratio_column": "set_mlp_conversion_ratio",
        "conversion_interpretation": (
            "Set-to-query model: encodes a set of nearby observed rays "
            "(STEC + geometry) and predicts query-ray STEC via cross-attention "
            "and pooling. VTEC = source_STEC × vertical_query / slant_query."
        ),
        "vtec_reference_used_for_training": False,
        "vertical_top_km": float(args.vertical_top_km),
        "training_pair_type": "leave_one_out_set",
        "query_geom_columns": QUERY_FEATURE_COLS,
        "obs_feature_columns": obs_feature_cols,
        "max_obs": int(args.max_obs),
        "ipp_radius_km": float(args.nearby_km),
        "inference_context": args.inference_context,
        "include_query_observation": bool(args.include_query_observation),
        "test_context_stats": test_context_stats,
        "spatial_sigma_km": float(args.spatial_sigma_km),
        "n_query_features": len(QUERY_FEATURE_COLS),
        "n_obs_features": n_obs_cols,
        "set_stats": set_extra,
        "split": split_metadata,
        "row_counts": {"input": int(len(df)), "train_rows": int(len(train_df)), "test_rows": int(len(test_df)), "train_examples": int(len(Y))},
        "hyperparameters": {
            "epochs": int(args.epochs), "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate), "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout), "seed": int(args.seed),
            "device": str(device), "spatial_sigma_km": float(args.spatial_sigma_km),
        },
        "history": history,
        "metrics": {
            "train_pair_stec": train_stec_metrics,
            "test_slant_stec": regression_metrics(predictions[args.slant_target_column], predictions["stec_set_mlp"]),
            "test_vtec": vtec_error_metrics(predictions[args.vtec_target_column], predictions["vtec_set_mlp"]),
            "test_vtec_direct": vtec_error_metrics(predictions[args.vtec_target_column], predictions["vtec_set_mlp_direct"]),
            "test_slm_vtec": vtec_error_metrics(predictions[args.vtec_target_column], predictions[args.baseline_vtec_column]),
        },
        "outputs": {"predictions": str(pred_path)},
    }
    if vtec_meta is not None:
        metadata["outputs"]["vtec_metrics"] = vtec_meta["outputs"]

    ckpt_path = checkpoint_dir / f"{run_name}.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "q_imputer": q_imp, "q_scaler": q_scaler, "o_imputer": o_imp, "o_scaler": o_scaler, "y_scaler": y_scaler, "metadata": metadata}, ckpt_path)
    metadata["outputs"]["checkpoint"] = str(ckpt_path)
    metadata["outputs"]["metrics_json"] = str(write_json(metadata, table_dir / f"{run_name}_metrics.json"))
    return metadata


def choose_simulation_split(df, args):
    if args.test_keys_from:
        key_path = Path(args.test_keys_from)
        key_df = read_table(key_path)
        key_columns = [c.strip() for c in args.test_key_columns.split(",") if c.strip()]
        require_columns(df, key_columns)
        require_columns(key_df, key_columns)
        left = df.copy()
        right = key_df.loc[:, key_columns].drop_duplicates().copy()
        if "timestamp" in key_columns:
            left["timestamp"] = pd.to_datetime(left["timestamp"])
            right["timestamp"] = pd.to_datetime(right["timestamp"])
        marker = "_set_mlp_test_key"
        merged = left.reset_index().merge(right.assign(**{marker: True}), on=key_columns, how="left", sort=False)
        test_mask = merged[marker].fillna(False).to_numpy(bool)
        train_df = df.loc[~test_mask].copy()
        test_df = df.loc[test_mask].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(f"Key holdout: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df, {"type": "key_holdout", "test_keys_from": str(key_path), "key_columns": key_columns, "n_train": int(len(train_df)), "n_test": int(len(test_df))}
    return choose_split(df, args.split, test_size=args.test_size, seed=args.seed, train_end=args.train_end)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--source-stec-column", default="stec_iri_observable")
    p.add_argument("--slant-target-column", default="stec_iri_ray")
    p.add_argument("--vtec-target-column", default="vtec_iri_ref")
    p.add_argument("--baseline-vtec-column", default="vtec_slm_from_iri_stec")
    p.add_argument("--mapping-column", default="slm_mapping_factor_iri")
    p.add_argument("--vertical-top-km", type=float, default=2000.0)
    p.add_argument("--split", choices=["random","time","station","prn"], default="random")
    p.add_argument("--test-keys-from")
    p.add_argument("--test-key-columns", default="timestamp,station,prn")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--train-end")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    p.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--output-format", choices=["parquet","csv"], default="parquet")
    p.add_argument("--elevation-bins", default="10,20,30,40,50,60,70,80,90")
    p.add_argument("--station-min-count", type=int, default=1)
    p.add_argument("--skip-vtec-evaluation", action="store_true")
    p.add_argument("--run-name")
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-obs", type=int, default=40, help="Max observed rays per set example.")
    p.add_argument("--inference-context", choices=["train", "all_observed"], default="train", help="Observed-ray pool used at inference time.")
    p.add_argument("--include-query-observation", action="store_true", help="Allow a query row's own observable STEC to appear in its observation set.")
    p.add_argument("--nearby-km", type=float, default=200.0, help="IPP radius (km) for observation set construction.")
    p.add_argument("--spatial-sigma-km", type=float, default=100.0, help="Spatial attention length-scale σ (km).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metadata = train_set_mlp(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
