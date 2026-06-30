"""Train a set/context source-ratio STEC-to-VTEC model.

This entry point extends the successful RayMLP source-ratio idea with a local
set context.  Each training sample contains:

    source-ray observation set + target-ray query geometry -> target/source STEC

The model never uses ``vtec_iri_ref`` for training.  Inference queries a vertical
special ray and converts the observed source STEC as::

    vtec_set_ratio_mlp = source_STEC * predicted_vertical_ratio
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .evaluate import build_vtec_metrics
    from .metrics import regression_metrics, vtec_error_metrics
    from .models.set_mlp import SetToQueryModel
    from .train_mlp import choose_device, set_random_seed
    from .train_ray_mlp_simulation import (
        TARGET_GEOMETRY_COLUMNS,
        build_vertical_target_geometry,
        require_columns,
    )
    from .train_set_mlp import (
        CONSTELLATION_COLS,
        OBS_GEO_CONST_COLS,
        QUERY_FEATURE_COLS,
        TIME_COLS,
        _SetDataset,
        _ensure_constellation,
        _haversine_km,
        _make_loader,
        _predict_set,
        choose_simulation_split,
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
    from metrics import regression_metrics, vtec_error_metrics
    from models.set_mlp import SetToQueryModel
    from train_mlp import choose_device, set_random_seed
    from train_ray_mlp_simulation import (
        TARGET_GEOMETRY_COLUMNS,
        build_vertical_target_geometry,
        require_columns,
    )
    from train_set_mlp import (
        CONSTELLATION_COLS,
        OBS_GEO_CONST_COLS,
        QUERY_FEATURE_COLS,
        TIME_COLS,
        _SetDataset,
        _ensure_constellation,
        _haversine_km,
        _make_loader,
        _predict_set,
        choose_simulation_split,
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
ANCHOR_FLAG_COLUMN = "is_source_anchor"


def _context_indices_for_source(
    source_pos: int,
    *,
    source_ipp_lon: float,
    source_ipp_lat: float,
    context_ipp_lon: np.ndarray,
    context_ipp_lat: np.ndarray,
    context_index: np.ndarray,
    source_index_value: Any,
    radius_km: float,
    max_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    dists = _haversine_km(
        source_ipp_lon,
        source_ipp_lat,
        context_ipp_lon,
        context_ipp_lat,
    )
    mask = dists <= radius_km
    self_matches = np.where(context_index == source_index_value)[0]
    if len(self_matches):
        mask[self_matches[0]] = True
    candidates = np.where(mask)[0]
    if len(candidates) > max_obs:
        order = np.argsort(dists[candidates])
        candidates = candidates[order[:max_obs]]
    return candidates.astype(np.intp), dists[candidates]


def _fill_context_tensor(
    *,
    out_features: np.ndarray,
    out_mask: np.ndarray,
    out_dists: np.ndarray,
    row: int,
    candidates: np.ndarray,
    source_index_value: Any,
    context_index: np.ndarray,
    context_base_features: np.ndarray,
    query_ipp_lon: float,
    query_ipp_lat: float,
    context_ipp_lon: np.ndarray,
    context_ipp_lat: np.ndarray,
) -> None:
    if len(candidates) == 0:
        return
    base = context_base_features[candidates]
    anchor = (context_index[candidates] == source_index_value).astype(np.float64)
    out_features[row, : len(candidates)] = np.column_stack([base, anchor])
    out_mask[row, : len(candidates)] = True
    out_dists[row, : len(candidates)] = _haversine_km(
        query_ipp_lon,
        query_ipp_lat,
        context_ipp_lon[candidates],
        context_ipp_lat[candidates],
    )


def build_pair_set_ratio_examples(
    df: pd.DataFrame,
    *,
    source_stec_col: str,
    target_stec_col: str,
    obs_cols: list[str],
    max_obs: int,
    context_radius_km: float,
    pair_radius_km: float,
    max_target_neighbors: int,
    include_identity: bool,
    max_pair_examples: int | None,
    seed: int,
    mapping_col: str | None = None,
    target_mode: str = "source_ratio",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build source-context/target-query examples with ratio-like labels."""

    n = len(df)
    if n < 2:
        raise ValueError("Set-ratio training requires at least 2 rows.")

    ipp_lon = df["ipp_lon"].to_numpy(dtype=np.float64)
    ipp_lat = df["ipp_lat"].to_numpy(dtype=np.float64)
    index_values = df.index.to_numpy()
    query_geom = df[QUERY_FEATURE_COLS].to_numpy(dtype=np.float64)
    context_base = df[obs_cols].to_numpy(dtype=np.float64)
    source_stec = df[source_stec_col].to_numpy(dtype=np.float64)
    target_stec = df[target_stec_col].to_numpy(dtype=np.float64)
    mapping = None
    if target_mode == "slm_correction":
        if mapping_col is None:
            raise ValueError("mapping_col is required for target_mode=slm_correction.")
        if mapping_col not in df.columns:
            raise KeyError(f"Training input is missing mapping column: {mapping_col}")
        mapping = df[mapping_col].to_numpy(dtype=np.float64)
    elif target_mode != "source_ratio":
        raise ValueError(f"Unsupported set-ratio target mode: {target_mode}")

    src_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    for i in range(n):
        dists = _haversine_km(ipp_lon[i], ipp_lat[i], ipp_lon, ipp_lat)
        mask = dists <= pair_radius_km
        if include_identity:
            mask[i] = True
        else:
            mask[i] = False
        candidates = np.where(mask)[0]
        if len(candidates) > max_target_neighbors:
            order = np.argsort(dists[candidates])
            candidates = candidates[order[:max_target_neighbors]]
        if len(candidates) == 0:
            continue
        src_parts.append(np.full(len(candidates), i, dtype=np.intp))
        tgt_parts.append(candidates.astype(np.intp))

    if not src_parts:
        raise ValueError("No source-ratio set pairs were created.")

    src_idx = np.concatenate(src_parts)
    tgt_idx = np.concatenate(tgt_parts)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = np.divide(
            target_stec[tgt_idx],
            source_stec[src_idx],
            out=np.full(len(src_idx), np.nan, dtype=np.float64),
            where=np.abs(source_stec[src_idx]) > 1e-12,
        )
    slm_pair_ratio = None
    if target_mode == "source_ratio":
        labels = raw_ratio
    else:
        assert mapping is not None
        with np.errstate(divide="ignore", invalid="ignore"):
            slm_pair_ratio = np.divide(
                mapping[tgt_idx],
                mapping[src_idx],
                out=np.full(len(src_idx), np.nan, dtype=np.float64),
                where=mapping[src_idx] > 0,
            )
            labels = np.divide(
                raw_ratio,
                slm_pair_ratio,
                out=np.full(len(src_idx), np.nan, dtype=np.float64),
                where=np.abs(slm_pair_ratio) > 1e-12,
            )
    finite = (
        np.isfinite(labels)
        & np.isfinite(raw_ratio)
        & np.isfinite(query_geom[tgt_idx]).all(axis=1)
    )
    if slm_pair_ratio is not None:
        finite &= np.isfinite(slm_pair_ratio) & (slm_pair_ratio > 0)
    src_idx = src_idx[finite]
    tgt_idx = tgt_idx[finite]
    labels = labels[finite]
    raw_ratio = raw_ratio[finite]
    if slm_pair_ratio is not None:
        slm_pair_ratio = slm_pair_ratio[finite]

    if max_pair_examples is not None and len(labels) > max_pair_examples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(labels), size=int(max_pair_examples), replace=False)
        src_idx = src_idx[keep]
        tgt_idx = tgt_idx[keep]
        labels = labels[keep]
        raw_ratio = raw_ratio[keep]
        if slm_pair_ratio is not None:
            slm_pair_ratio = slm_pair_ratio[keep]

    n_examples = len(labels)
    n_obs_features = len(obs_cols) + 1
    QG = np.zeros((n_examples, len(QUERY_FEATURE_COLS)), dtype=np.float64)
    OF = np.zeros((n_examples, max_obs, n_obs_features), dtype=np.float64)
    OM = np.zeros((n_examples, max_obs), dtype=bool)
    OD = np.full((n_examples, max_obs), np.inf, dtype=np.float64)
    n_context = np.zeros(n_examples, dtype=int)

    context_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row, (src, tgt) in enumerate(zip(src_idx, tgt_idx)):
        QG[row] = query_geom[tgt]
        if int(src) not in context_cache:
            context_cache[int(src)] = _context_indices_for_source(
                int(src),
                source_ipp_lon=float(ipp_lon[src]),
                source_ipp_lat=float(ipp_lat[src]),
                context_ipp_lon=ipp_lon,
                context_ipp_lat=ipp_lat,
                context_index=index_values,
                source_index_value=index_values[src],
                radius_km=context_radius_km,
                max_obs=max_obs,
            )
        candidates, _ = context_cache[int(src)]
        n_context[row] = len(candidates)
        _fill_context_tensor(
            out_features=OF,
            out_mask=OM,
            out_dists=OD,
            row=row,
            candidates=candidates,
            source_index_value=index_values[src],
            context_index=index_values,
            context_base_features=context_base,
            query_ipp_lon=float(ipp_lon[tgt]),
            query_ipp_lat=float(ipp_lat[tgt]),
            context_ipp_lon=ipp_lon,
            context_ipp_lat=ipp_lat,
        )

    valid = OM.any(axis=1) & np.isfinite(labels)
    identity = src_idx == tgt_idx
    stats = {
        "target_mode": target_mode,
        "training_label_stats": _finite_stats(labels[valid]),
        "raw_source_ratio_stats": _finite_stats(raw_ratio[valid]),
        "slm_pair_ratio_stats": _finite_stats(slm_pair_ratio[valid])
        if slm_pair_ratio is not None else None,
        "n_pairs_before_context_filter": int(n_examples),
        "n_pairs_after_context_filter": int(valid.sum()),
        "n_identity_pairs": int(np.count_nonzero(identity[valid])),
        "identity_pair_fraction": float(np.count_nonzero(identity[valid]) / max(valid.sum(), 1)),
        "mean_context_neighbors": float(np.mean(n_context[valid])) if valid.any() else 0.0,
        "min_context_neighbors": int(np.min(n_context[valid])) if valid.any() else 0,
        "max_context_neighbors": int(np.max(n_context[valid])) if valid.any() else 0,
        "max_pair_examples": int(max_pair_examples) if max_pair_examples is not None else None,
    }
    return QG[valid], OF[valid], OM[valid], OD[valid], labels[valid], stats


def build_inference_context_sets(
    query_df: pd.DataFrame,
    context_df: pd.DataFrame,
    *,
    target_geom: np.ndarray,
    obs_cols: list[str],
    max_obs: int,
    context_radius_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build source-centered contexts for test rows."""

    n_query = len(query_df)
    n_obs_features = len(obs_cols) + 1
    source_ipp_lon = query_df["ipp_lon"].to_numpy(dtype=np.float64)
    source_ipp_lat = query_df["ipp_lat"].to_numpy(dtype=np.float64)
    source_index = query_df.index.to_numpy()
    target_ipp_lon = target_geom[:, QUERY_FEATURE_COLS.index("ipp_lon")]
    target_ipp_lat = target_geom[:, QUERY_FEATURE_COLS.index("ipp_lat")]

    context_ipp_lon = context_df["ipp_lon"].to_numpy(dtype=np.float64)
    context_ipp_lat = context_df["ipp_lat"].to_numpy(dtype=np.float64)
    context_index = context_df.index.to_numpy()
    context_base = context_df[obs_cols].to_numpy(dtype=np.float64)

    OF = np.zeros((n_query, max_obs, n_obs_features), dtype=np.float64)
    OM = np.zeros((n_query, max_obs), dtype=bool)
    OD = np.full((n_query, max_obs), np.inf, dtype=np.float64)
    n_context = np.zeros(n_query, dtype=int)
    has_anchor = np.zeros(n_query, dtype=bool)

    for i in range(n_query):
        candidates, _ = _context_indices_for_source(
            i,
            source_ipp_lon=float(source_ipp_lon[i]),
            source_ipp_lat=float(source_ipp_lat[i]),
            context_ipp_lon=context_ipp_lon,
            context_ipp_lat=context_ipp_lat,
            context_index=context_index,
            source_index_value=source_index[i],
            radius_km=context_radius_km,
            max_obs=max_obs,
        )
        n_context[i] = len(candidates)
        has_anchor[i] = np.any(context_index[candidates] == source_index[i])
        _fill_context_tensor(
            out_features=OF,
            out_mask=OM,
            out_dists=OD,
            row=i,
            candidates=candidates,
            source_index_value=source_index[i],
            context_index=context_index,
            context_base_features=context_base,
            query_ipp_lon=float(target_ipp_lon[i]),
            query_ipp_lat=float(target_ipp_lat[i]),
            context_ipp_lon=context_ipp_lon,
            context_ipp_lat=context_ipp_lat,
        )

    valid = OM.any(axis=1)
    stats = {
        "mean_context_neighbors": float(np.mean(n_context[valid])) if valid.any() else 0.0,
        "min_context_neighbors": int(np.min(n_context[valid])) if valid.any() else 0,
        "max_context_neighbors": int(np.max(n_context[valid])) if valid.any() else 0,
        "n_total": int(n_query),
        "n_valid": int(valid.sum()),
        "n_with_source_anchor": int(np.count_nonzero(has_anchor)),
    }
    return OF, OM, OD, stats


def _scale_obs(OF: np.ndarray, OM: np.ndarray, o_imp: Any, o_scaler: Any) -> np.ndarray:
    out_imp = np.zeros_like(OF)
    out = np.zeros_like(OF)
    for i in range(len(OF)):
        nv = int(OM[i].sum())
        if nv > 0:
            out_imp[i, :nv] = o_imp.transform(OF[i, :nv])
            out[i, :nv] = o_scaler.transform(out_imp[i, :nv])
    return out


def _anchor_source_geometry(
    obs_feat: np.ndarray,
    obs_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the source-anchor geometry from an unscaled observation tensor."""

    n = len(obs_feat)
    source_geom = np.zeros((n, len(QUERY_FEATURE_COLS)), dtype=np.float64)
    has_anchor = np.zeros(n, dtype=bool)
    has_valid = obs_mask.any(axis=1)
    for i in range(n):
        if not has_valid[i]:
            continue
        anchor_mask = obs_mask[i] & (obs_feat[i, :, -1] > 0.5)
        if anchor_mask.any():
            j = int(np.where(anchor_mask)[0][0])
            has_anchor[i] = True
        else:
            j = int(np.where(obs_mask[i])[0][0])
        source_geom[i] = obs_feat[i, j, : len(QUERY_FEATURE_COLS)]
    return source_geom, has_anchor


def compute_identity_delta(
    query_geom: np.ndarray,
    obs_feat: np.ndarray,
    obs_mask: np.ndarray,
    *,
    ipp_scale_km: float = 200.0,
) -> np.ndarray:
    """Geometry distance used in ratio = 1 + delta * residual.

    The value is exactly zero when the query ray geometry is identical to the
    source-anchor ray geometry.  It grows with IPP separation, elevation change,
    slant-factor change, ray-length change, and LOS direction change.
    """

    source_geom, _ = _anchor_source_geometry(obs_feat, obs_mask)
    q = np.asarray(query_geom, dtype=np.float64)
    s = source_geom

    ipp_lon_i = QUERY_FEATURE_COLS.index("ipp_lon")
    ipp_lat_i = QUERY_FEATURE_COLS.index("ipp_lat")
    elev_i = QUERY_FEATURE_COLS.index("elevation")
    slant_i = QUERY_FEATURE_COLS.index("slant_factor")
    ray_i = QUERY_FEATURE_COLS.index("ray_length")
    los_i = [
        QUERY_FEATURE_COLS.index("los_x"),
        QUERY_FEATURE_COLS.index("los_y"),
        QUERY_FEATURE_COLS.index("los_z"),
    ]

    ipp_delta = _haversine_km(
        s[:, ipp_lon_i],
        s[:, ipp_lat_i],
        q[:, ipp_lon_i],
        q[:, ipp_lat_i],
    ) / max(float(ipp_scale_km), 1e-6)
    elev_delta = np.abs(q[:, elev_i] - s[:, elev_i]) / 90.0
    slant_delta = np.abs(q[:, slant_i] - s[:, slant_i]) / np.maximum(
        np.abs(s[:, slant_i]), 1e-6
    )
    ray_delta = np.abs(q[:, ray_i] - s[:, ray_i]) / np.maximum(
        np.abs(s[:, ray_i]), 1e-6
    )
    los_delta = np.linalg.norm(q[:, los_i] - s[:, los_i], axis=1) / 2.0
    delta = np.sqrt(
        ipp_delta**2 + elev_delta**2 + slant_delta**2 + ray_delta**2 + los_delta**2
    )
    delta[~obs_mask.any(axis=1)] = 1.0
    delta[np.abs(delta) < 1e-12] = 0.0
    return delta.astype(np.float64)


def _finite_stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _ratio_identity_stats(ratio: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    ratio = np.asarray(ratio, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    identity = np.isfinite(delta) & (np.abs(delta) <= 1e-12)
    if not identity.any():
        return {"n_identity": 0, "ratio": _finite_stats(np.array([], dtype=np.float64))}
    return {
        "n_identity": int(np.count_nonzero(identity)),
        "ratio": _finite_stats(ratio[identity]),
        "abs_error_to_one": _finite_stats(np.abs(ratio[identity] - 1.0)),
    }


def elevation_shrinkage_strength(
    elevation_deg: np.ndarray,
    *,
    enabled: bool,
    start_deg: float,
    end_deg: float,
    power: float,
    min_strength: float,
) -> np.ndarray:
    """Return how much of the ML correction to keep at each source elevation."""

    elevation = np.asarray(elevation_deg, dtype=np.float64)
    if not enabled:
        return np.ones_like(elevation, dtype=np.float64)
    if end_deg <= start_deg:
        raise ValueError("--shrinkage-end-deg must be larger than --shrinkage-start-deg.")
    if power <= 0:
        raise ValueError("--shrinkage-power must be positive.")
    clipped_min = float(np.clip(min_strength, 0.0, 1.0))
    base = (float(end_deg) - elevation) / (float(end_deg) - float(start_deg))
    strength = np.clip(base, 0.0, 1.0) ** float(power)
    return np.maximum(strength, clipped_min).astype(np.float64)


def apply_correction_shrinkage(
    correction: np.ndarray,
    strength: np.ndarray,
) -> np.ndarray:
    """Shrink a correction factor toward 1 using a 0..1 strength."""

    correction = np.asarray(correction, dtype=np.float64)
    strength = np.asarray(strength, dtype=np.float64)
    return 1.0 + strength * (correction - 1.0)


class _SetDeltaDataset(Dataset):
    def __init__(self, q_geom, obs_feat, obs_mask, obs_dists, labels, deltas):
        self._q = q_geom.copy()
        self._o = obs_feat.copy()
        self._m = obs_mask.copy()
        self._d = obs_dists.copy()
        self._y = labels.copy()
        self._g = deltas.copy()

    def __len__(self):
        return len(self._q)

    def __getitem__(self, idx):
        return (
            self._q[idx],
            self._o[idx],
            self._m[idx],
            self._d[idx],
            self._y[idx],
            self._g[idx],
        )


def _collate_delta(batch):
    qg, of, om, od, y, gd = zip(*batch)
    return (
        torch.tensor(np.stack(qg), dtype=torch.float32),
        torch.tensor(np.stack(of), dtype=torch.float32),
        torch.tensor(np.stack(om), dtype=torch.bool),
        torch.tensor(np.stack(od), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1),
        torch.tensor(np.array(gd), dtype=torch.float32).view(-1, 1),
    )


def _make_delta_loader(ds, batch_size, seed, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=_collate_delta,
        generator=g,
    )


def _predict_hard_identity_ratio(
    model,
    device,
    q_geom,
    obs_feat,
    obs_mask,
    obs_dists,
    deltas,
    batch_size=512,
) -> np.ndarray:
    preds = []
    model.eval()
    for start in range(0, len(q_geom), batch_size):
        end = min(start + batch_size, len(q_geom))
        qg = torch.tensor(q_geom[start:end], dtype=torch.float32, device=device)
        of_ = torch.tensor(obs_feat[start:end], dtype=torch.float32, device=device)
        om = torch.tensor(obs_mask[start:end], dtype=torch.bool, device=device)
        od = torch.tensor(obs_dists[start:end], dtype=torch.float32, device=device)
        gd = torch.tensor(deltas[start:end], dtype=torch.float32, device=device).view(-1, 1)
        with torch.no_grad():
            residual = model(qg, of_, om, od)
            ratio = 1.0 + gd * residual
        preds.append(ratio.cpu().numpy())
    return np.concatenate(preds, axis=0).ravel()


def choose_set_ratio_split(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Choose train/evaluation rows for set-ratio simulation experiments."""

    if args.split == "all":
        if args.test_keys_from:
            raise ValueError("--test-keys-from cannot be used with --split all.")
        full_df = df.copy()
        return full_df, full_df.copy(), {
            "type": "all",
            "n_train": int(len(full_df)),
            "n_test": int(len(full_df)),
            "train_test_overlap": True,
            "interpretation": (
                "Diagnostic full-data fit/evaluation: all rays are used for "
                "training and all rays are evaluated. VTEC truth remains "
                "evaluation-only and is not used as a training label."
            ),
        }
    return choose_simulation_split(df, args)


def apply_elevation_filters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Optionally restrict train/evaluation rows by source-ray elevation."""

    train_before = int(len(train_df))
    test_before = int(len(test_df))
    train_min = args.min_train_elevation_deg
    eval_min = args.min_eval_elevation_deg

    if train_min is not None:
        train_df = train_df.loc[
            pd.to_numeric(train_df["elevation"], errors="coerce") >= float(train_min)
        ].copy()
    if eval_min is not None:
        test_df = test_df.loc[
            pd.to_numeric(test_df["elevation"], errors="coerce") >= float(eval_min)
        ].copy()

    if len(train_df) < 2:
        raise ValueError(
            "Elevation filtering left fewer than 2 training rays; "
            "relax --min-train-elevation-deg."
        )
    if len(test_df) == 0:
        raise ValueError(
            "Elevation filtering left no evaluation rays; "
            "relax --min-eval-elevation-deg."
        )

    return train_df, test_df, {
        "min_train_elevation_deg": None if train_min is None else float(train_min),
        "min_eval_elevation_deg": None if eval_min is None else float(eval_min),
        "train_rows_before": train_before,
        "train_rows_after": int(len(train_df)),
        "test_rows_before": test_before,
        "test_rows_after": int(len(test_df)),
    }

def train_set_ratio_mlp(args: argparse.Namespace) -> dict[str, Any]:
    set_random_seed(args.seed)
    input_path = Path(args.input)
    table_dir = Path(args.table_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = args.run_name or "set_ratio_mlp"

    df = read_table(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    df = _ensure_constellation(df)

    obs_cols = OBS_GEO_CONST_COLS + [args.source_stec_column]
    model_obs_cols = obs_cols + [ANCHOR_FLAG_COLUMN]
    required = (
        QUERY_FEATURE_COLS
        + obs_cols
        + [
            args.source_stec_column,
            args.slant_target_column,
            args.vtec_target_column,
            args.baseline_vtec_column,
            args.mapping_column,
        ]
    )
    require_columns(df, required)

    train_df, test_df, split_metadata = choose_set_ratio_split(df, args)
    train_df, test_df, elevation_filter_metadata = apply_elevation_filters(
        train_df, test_df, args
    )
    split_metadata["elevation_filter"] = elevation_filter_metadata
    QG, OF, OM, OD, Y, pair_stats = build_pair_set_ratio_examples(
        train_df,
        source_stec_col=args.source_stec_column,
        target_stec_col=args.slant_target_column,
        obs_cols=obs_cols,
        max_obs=args.max_obs,
        context_radius_km=args.context_km,
        pair_radius_km=args.nearby_km,
        max_target_neighbors=args.nearby_max_neighbors,
        include_identity=not args.no_identity,
        max_pair_examples=args.max_pair_examples,
        seed=args.seed,
        mapping_col=args.mapping_column,
        target_mode=args.target_mode,
    )
    if len(Y) < 2:
        raise ValueError("Set-ratio training requires at least 2 valid examples.")

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    q_imp = SimpleImputer(strategy="median")
    q_scaler = StandardScaler()
    o_imp = SimpleImputer(strategy="median")
    o_scaler = StandardScaler()
    y_scaler = None if args.hard_identity_ratio else StandardScaler()

    QG_scaled = q_scaler.fit_transform(q_imp.fit_transform(QG)).astype(np.float64)
    obs_valid = OF.reshape(-1, OF.shape[-1])[OM.ravel()]
    o_imp.fit(obs_valid)
    o_scaler.fit(o_imp.transform(obs_valid))
    OF_scaled = _scale_obs(OF, OM, o_imp, o_scaler)
    identity_delta = compute_identity_delta(
        QG,
        OF,
        OM,
        ipp_scale_km=args.identity_delta_ipp_scale_km,
    )

    device = choose_device(args.device)
    if args.hard_identity_ratio:
        loader = _make_delta_loader(
            _SetDeltaDataset(QG_scaled, OF_scaled, OM, OD, Y.astype(np.float64), identity_delta),
            args.batch_size,
            args.seed,
            shuffle=True,
        )
    else:
        Y_scaled = y_scaler.fit_transform(Y.reshape(-1, 1)).ravel().astype(np.float64)
        loader = _make_loader(
            _SetDataset(QG_scaled, OF_scaled, OM, OD, Y_scaled),
            args.batch_size,
            args.seed,
            shuffle=True,
        )
    model = SetToQueryModel(
        dropout=args.dropout,
        spatial_length_scale_km=args.spatial_sigma_km,
        n_query_features=len(QUERY_FEATURE_COLS),
        n_obs_features=len(model_obs_cols),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = torch.nn.MSELoss()
    correction_mean_tensor = None
    correction_scale_tensor = None
    if (
        args.target_mode == "slm_correction"
        and args.correction_regularization > 0
        and not args.hard_identity_ratio
    ):
        correction_mean_tensor = torch.tensor(
            float(y_scaler.mean_[0]), dtype=torch.float32, device=device
        )
        correction_scale_tensor = torch.tensor(
            float(y_scaler.scale_[0]), dtype=torch.float32, device=device
        )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            if args.hard_identity_ratio:
                qg, of_, om, od, yb, gd = batch
                gd = gd.to(device)
            else:
                qg, of_, om, od, yb = batch
                gd = None
            qg = qg.to(device)
            of_ = of_.to(device)
            om = om.to(device)
            od = od.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw = model(qg, of_, om, od)
            pred = 1.0 + gd * raw if args.hard_identity_ratio else raw
            loss = criterion(pred, yb)
            if args.target_mode == "slm_correction" and args.correction_regularization > 0:
                if args.hard_identity_ratio:
                    pred_label = pred
                else:
                    pred_label = pred * correction_scale_tensor + correction_mean_tensor
                loss = loss + float(args.correction_regularization) * torch.mean(
                    (pred_label - 1.0) ** 2
                )
            loss.backward()
            optimizer.step()
            bs = len(qg)
            total_loss += float(loss.detach().cpu()) * bs
            total_count += bs
        history.append({"epoch": epoch, "train_loss": total_loss / max(total_count, 1)})

    if args.hard_identity_ratio:
        train_pred_label = _predict_hard_identity_ratio(
            model,
            device,
            QG_scaled,
            OF_scaled,
            OM,
            OD,
            identity_delta,
            batch_size=args.batch_size,
        )
    else:
        train_pred_scaled = _predict_set(
            model, device, QG_scaled, OF_scaled, OM, OD, batch_size=args.batch_size
        )
        train_pred_label = y_scaler.inverse_transform(
            train_pred_scaled.reshape(-1, 1)
        ).ravel()

    context_df = df if args.inference_context == "all_observed" else train_df
    test_slant_geom = test_df[QUERY_FEATURE_COLS].to_numpy(dtype=np.float64)
    test_vert_base = build_vertical_target_geometry(
        test_df, vertical_top_km=args.vertical_top_km
    )[TARGET_GEOMETRY_COLUMNS].to_numpy(dtype=np.float64)
    test_vert_geom = np.column_stack(
        [test_vert_base, test_df[TIME_COLS].to_numpy(dtype=np.float64)]
    )

    slant_OF, slant_OM, slant_OD, slant_context_stats = build_inference_context_sets(
        test_df,
        context_df,
        target_geom=test_slant_geom,
        obs_cols=obs_cols,
        max_obs=args.max_obs,
        context_radius_km=args.context_km,
    )
    vert_OF, vert_OM, vert_OD, vert_context_stats = build_inference_context_sets(
        test_df,
        context_df,
        target_geom=test_vert_geom,
        obs_cols=obs_cols,
        max_obs=args.max_obs,
        context_radius_km=args.context_km,
    )

    slant_QG_scaled = q_scaler.transform(q_imp.transform(test_slant_geom)).astype(np.float64)
    vert_QG_scaled = q_scaler.transform(q_imp.transform(test_vert_geom)).astype(np.float64)
    slant_OF_scaled = _scale_obs(slant_OF, slant_OM, o_imp, o_scaler)
    vert_OF_scaled = _scale_obs(vert_OF, vert_OM, o_imp, o_scaler)
    slant_identity_delta = compute_identity_delta(
        test_slant_geom,
        slant_OF,
        slant_OM,
        ipp_scale_km=args.identity_delta_ipp_scale_km,
    )
    vertical_identity_delta = compute_identity_delta(
        test_vert_geom,
        vert_OF,
        vert_OM,
        ipp_scale_km=args.identity_delta_ipp_scale_km,
    )

    if args.hard_identity_ratio:
        slant_model_output = _predict_hard_identity_ratio(
            model,
            device,
            slant_QG_scaled,
            slant_OF_scaled,
            slant_OM,
            slant_OD,
            slant_identity_delta,
            batch_size=args.batch_size,
        )
        vertical_model_output = _predict_hard_identity_ratio(
            model,
            device,
            vert_QG_scaled,
            vert_OF_scaled,
            vert_OM,
            vert_OD,
            vertical_identity_delta,
            batch_size=args.batch_size,
        )
    else:
        slant_model_output = y_scaler.inverse_transform(
            _predict_set(
                model,
                device,
                slant_QG_scaled,
                slant_OF_scaled,
                slant_OM,
                slant_OD,
                batch_size=args.batch_size,
            ).reshape(-1, 1)
        ).ravel()
        vertical_model_output = y_scaler.inverse_transform(
            _predict_set(
                model,
                device,
                vert_QG_scaled,
                vert_OF_scaled,
                vert_OM,
                vert_OD,
                batch_size=args.batch_size,
            ).reshape(-1, 1)
        ).ravel()

    slant_model_output = np.maximum(slant_model_output, 0.0)
    vertical_model_output = np.maximum(vertical_model_output, 0.0)
    mapping_i = QUERY_FEATURE_COLS.index("slant_factor")
    source_mapping_eval = test_df[args.mapping_column].to_numpy(dtype=np.float64)
    slant_target_mapping = test_slant_geom[:, mapping_i]
    vertical_target_mapping = test_vert_geom[:, mapping_i]
    with np.errstate(divide="ignore", invalid="ignore"):
        slm_slant_ratio = np.divide(
            slant_target_mapping,
            source_mapping_eval,
            out=np.full(len(test_df), np.nan, dtype=np.float64),
            where=source_mapping_eval > 0,
        )
        slm_vertical_ratio = np.divide(
            vertical_target_mapping,
            source_mapping_eval,
            out=np.full(len(test_df), np.nan, dtype=np.float64),
            where=source_mapping_eval > 0,
        )
    source_elevation_eval = test_df["elevation"].to_numpy(dtype=np.float64)
    shrinkage_enabled = bool(args.elevation_shrinkage and args.target_mode == "slm_correction")
    shrinkage_strength = elevation_shrinkage_strength(
        source_elevation_eval,
        enabled=shrinkage_enabled,
        start_deg=args.shrinkage_start_deg,
        end_deg=args.shrinkage_end_deg,
        power=args.shrinkage_power,
        min_strength=args.shrinkage_min_strength,
    )
    if args.target_mode == "source_ratio":
        slant_ratio = slant_model_output
        vertical_ratio = vertical_model_output
        with np.errstate(divide="ignore", invalid="ignore"):
            slant_correction_raw = np.divide(slant_ratio, slm_slant_ratio)
            vertical_correction_raw = np.divide(vertical_ratio, slm_vertical_ratio)
        slant_correction = slant_correction_raw
        vertical_correction = vertical_correction_raw
    else:
        slant_correction_raw = slant_model_output
        vertical_correction_raw = vertical_model_output
        slant_correction = apply_correction_shrinkage(
            slant_correction_raw, shrinkage_strength
        )
        vertical_correction = apply_correction_shrinkage(
            vertical_correction_raw, shrinkage_strength
        )
        slant_ratio = slant_correction * slm_slant_ratio
        vertical_ratio = vertical_correction * slm_vertical_ratio
    slant_ratio = np.maximum(slant_ratio, 0.0)
    vertical_ratio = np.maximum(vertical_ratio, 0.0)
    source_stec_eval = test_df[args.source_stec_column].to_numpy(dtype=np.float64)

    predictions = test_df.copy()
    predictions["set_ratio_slant_identity_delta"] = slant_identity_delta
    predictions["set_ratio_vertical_identity_delta"] = vertical_identity_delta
    predictions["set_ratio_slant_slm_ratio"] = slm_slant_ratio
    predictions["set_ratio_vertical_slm_ratio"] = slm_vertical_ratio
    predictions["set_ratio_slant_model_output"] = slant_model_output
    predictions["set_ratio_vertical_model_output"] = vertical_model_output
    predictions["set_ratio_elevation_shrinkage_strength"] = shrinkage_strength
    predictions["set_ratio_slant_correction_raw"] = slant_correction_raw
    predictions["set_ratio_vertical_correction_raw"] = vertical_correction_raw
    predictions["set_ratio_slant_correction"] = slant_correction
    predictions["set_ratio_vertical_correction"] = vertical_correction
    predictions["set_ratio_slant_ratio"] = slant_ratio
    predictions["set_ratio_vertical_ratio"] = vertical_ratio
    predictions["stec_set_ratio_mlp"] = source_stec_eval * slant_ratio
    predictions["vtec_set_ratio_mlp"] = source_stec_eval * vertical_ratio
    predictions["vtec_set_ratio_mlp_direct"] = predictions["vtec_set_ratio_mlp"]
    predictions["vtec_set_ratio_mlp_error_iri"] = (
        predictions["vtec_set_ratio_mlp"] - predictions[args.vtec_target_column]
    )
    predictions["vtec_set_ratio_mlp_abs_error_iri"] = predictions[
        "vtec_set_ratio_mlp_error_iri"
    ].abs()

    pred_path = save_table(
        predictions, table_dir / f"{run_name}_predictions.{args.output_format}"
    )

    vtec_meta = None
    if not args.skip_vtec_evaluation:
        vtec_meta = build_vtec_metrics(
            input_path=pred_path,
            table_dir=table_dir,
            run_name=f"{run_name}_test",
            target_column=args.vtec_target_column,
            baseline_column=args.baseline_vtec_column,
            vtec_prediction_columns=["vtec_set_ratio_mlp", "vtec_set_ratio_mlp_direct"],
            stec_prediction_columns=[],
            mapping_column=args.mapping_column,
            elevation_bins=[
                float(s.strip()) for s in args.elevation_bins.split(",") if s.strip()
            ],
            station_min_count=args.station_min_count,
            max_rows=None,
        )

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": f"simulation_set_context_{args.target_mode}"
        + ("_hard_identity" if args.hard_identity_ratio else ""),
        "model": "set_ratio_mlp",
        "target_mode": args.target_mode,
        "run_name": run_name,
        "input_file": str(input_path),
        "source_stec_column": args.source_stec_column,
        "slant_target_column": args.slant_target_column,
        "vtec_target_column": args.vtec_target_column,
        "baseline_vtec_column": args.baseline_vtec_column,
        "stec_prediction_column": "stec_set_ratio_mlp",
        "vtec_prediction_column": "vtec_set_ratio_mlp",
        "conversion_ratio_column": "set_ratio_vertical_ratio",
        "conversion_interpretation": (
            "Set/context model: context observed rays include an is_source_anchor "
            "flag. In source_ratio mode, the training target is "
            "target_STEC/source_STEC. In slm_correction mode, the training "
            "target is (target_STEC/source_STEC) divided by the SLM geometric "
            "ratio, and inference multiplies the predicted correction by the "
            "SLM source-to-query ratio. vtec_iri_ref is evaluation-only."
        ),
        "vtec_reference_used_for_training": False,
        "vertical_top_km": float(args.vertical_top_km),
        "query_geom_columns": QUERY_FEATURE_COLS,
        "obs_feature_columns": model_obs_cols,
        "pair_stats": pair_stats,
        "slant_context_stats": slant_context_stats,
        "vertical_context_stats": vert_context_stats,
        "split": split_metadata,
        "row_counts": {
            "input": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_pair_examples": int(len(Y)),
        },
        "hyperparameters": {
            "target_mode": args.target_mode,
            "correction_regularization": float(args.correction_regularization),
            "elevation_shrinkage": bool(args.elevation_shrinkage),
            "shrinkage_start_deg": float(args.shrinkage_start_deg),
            "shrinkage_end_deg": float(args.shrinkage_end_deg),
            "shrinkage_power": float(args.shrinkage_power),
            "shrinkage_min_strength": float(args.shrinkage_min_strength),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "seed": int(args.seed),
            "device": str(device),
            "nearby_km": float(args.nearby_km),
            "context_km": float(args.context_km),
            "nearby_max_neighbors": int(args.nearby_max_neighbors),
            "max_obs": int(args.max_obs),
            "max_pair_examples": int(args.max_pair_examples)
            if args.max_pair_examples is not None else None,
            "spatial_sigma_km": float(args.spatial_sigma_km),
            "include_identity": bool(not args.no_identity),
            "hard_identity_ratio": bool(args.hard_identity_ratio),
            "identity_delta_ipp_scale_km": float(args.identity_delta_ipp_scale_km),
            "inference_context": args.inference_context,
            "min_train_elevation_deg": None
            if args.min_train_elevation_deg is None
            else float(args.min_train_elevation_deg),
            "min_eval_elevation_deg": None
            if args.min_eval_elevation_deg is None
            else float(args.min_eval_elevation_deg),
        },
        "history": history,
        "identity_delta_stats": {
            "train": _finite_stats(identity_delta),
            "test_slant": _finite_stats(slant_identity_delta),
            "test_vertical": _finite_stats(vertical_identity_delta),
            "test_slant_identity_ratio": _ratio_identity_stats(
                slant_ratio, slant_identity_delta
            ),
        },
        "elevation_shrinkage_stats": {
            "enabled": shrinkage_enabled,
            "strength": _finite_stats(shrinkage_strength),
            "raw_vertical_correction": _finite_stats(vertical_correction_raw),
            "shrunk_vertical_correction": _finite_stats(vertical_correction),
        },
        "metrics": {
            "train_pair_training_target": regression_metrics(Y, train_pred_label),
            (
                "train_pair_ratio"
                if args.target_mode == "source_ratio"
                else "train_pair_slm_correction"
            ): regression_metrics(Y, train_pred_label),
            "test_slant_stec": regression_metrics(
                predictions[args.slant_target_column],
                predictions["stec_set_ratio_mlp"],
            ),
            "test_vtec": vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions["vtec_set_ratio_mlp"],
            ),
            "test_slm_vtec": vtec_error_metrics(
                predictions[args.vtec_target_column],
                predictions[args.baseline_vtec_column],
            ),
        },
        "outputs": {"predictions": str(pred_path)},
    }
    if vtec_meta is not None:
        metadata["outputs"]["vtec_metrics"] = vtec_meta["outputs"]

    ckpt_path = checkpoint_dir / f"{run_name}.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "q_imputer": q_imp,
            "q_scaler": q_scaler,
            "o_imputer": o_imp,
            "o_scaler": o_scaler,
            "y_scaler": y_scaler,
            "metadata": metadata,
        },
        ckpt_path,
    )
    metadata["outputs"]["checkpoint"] = str(ckpt_path)
    metadata["outputs"]["metrics_json"] = str(
        write_json(metadata, table_dir / f"{run_name}_metrics.json")
    )
    return metadata


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--source-stec-column", default="stec_iri_observable")
    p.add_argument("--slant-target-column", default="stec_iri_ray")
    p.add_argument("--vtec-target-column", default="vtec_iri_ref")
    p.add_argument("--baseline-vtec-column", default="vtec_slm_from_iri_stec")
    p.add_argument("--mapping-column", default="slm_mapping_factor_iri")
    p.add_argument("--vertical-top-km", type=float, default=2000.0)
    p.add_argument(
        "--target-mode",
        choices=["source_ratio", "slm_correction"],
        default="source_ratio",
        help="Train either direct source-to-query ratio or correction to SLM geometry.",
    )
    p.add_argument(
        "--correction-regularization",
        type=float,
        default=0.0,
        help="For target-mode=slm_correction, penalize deviations from correction=1.",
    )
    p.add_argument(
        "--elevation-shrinkage",
        action="store_true",
        help="Shrink SLM-correction outputs toward 1 as source elevation increases.",
    )
    p.add_argument("--shrinkage-start-deg", type=float, default=30.0)
    p.add_argument("--shrinkage-end-deg", type=float, default=90.0)
    p.add_argument("--shrinkage-power", type=float, default=1.0)
    p.add_argument("--shrinkage-min-strength", type=float, default=0.0)
    p.add_argument("--min-train-elevation-deg", type=float)
    p.add_argument("--min-eval-elevation-deg", type=float)
    p.add_argument("--split", choices=["random", "time", "station", "prn", "all"], default="random")
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
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    p.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--elevation-bins", default="10,20,30,40,50,60,70,80,90")
    p.add_argument("--station-min-count", type=int, default=1)
    p.add_argument("--skip-vtec-evaluation", action="store_true")
    p.add_argument("--run-name")
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-obs", type=int, default=40)
    p.add_argument("--nearby-km", type=float, default=200.0)
    p.add_argument("--context-km", type=float, default=200.0)
    p.add_argument("--nearby-max-neighbors", type=int, default=30)
    p.add_argument("--max-pair-examples", type=int)
    p.add_argument("--no-identity", action="store_true")
    p.add_argument("--hard-identity-ratio", action="store_true")
    p.add_argument("--identity-delta-ipp-scale-km", type=float, default=200.0)
    p.add_argument("--spatial-sigma-km", type=float, default=100.0)
    p.add_argument("--inference-context", choices=["train", "all_observed"], default="all_observed")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metadata = train_set_ratio_mlp(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())