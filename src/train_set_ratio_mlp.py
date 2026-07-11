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

def transform_training_labels(
    labels: np.ndarray,
    *,
    target_mode: str,
    target_transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed training labels and their finite-valid mask."""

    if target_transform == "linear":
        return labels, np.isfinite(labels)
    if target_transform == "log":
        if target_mode != "slm_correction":
            raise ValueError("--target-transform log requires --target-mode slm_correction.")
        with np.errstate(divide="ignore", invalid="ignore"):
            transformed = np.log(labels)
        return transformed, np.isfinite(transformed) & (labels > 0)
    raise ValueError(f"Unsupported target transform: {target_transform}")
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
    min_target_elevation_deg: float | None = None,
    target_elevation_keep_identity: bool = False,
    mapping_col: str | None = None,
    target_mode: str = "source_ratio",
    target_transform: str = "linear",
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
    target_elevation = df["elevation"].to_numpy(dtype=np.float64)
    target_elevation_threshold = (
        None if min_target_elevation_deg is None
        else float(min_target_elevation_deg)
    )
    target_elevation_mask = np.ones(n, dtype=bool)
    if target_elevation_threshold is not None:
        target_elevation_mask = target_elevation >= target_elevation_threshold
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
        if target_elevation_threshold is not None:
            mask &= target_elevation_mask
        if include_identity:
            keep_source_identity = (
                target_elevation_threshold is None
                or target_elevation_keep_identity
                or bool(target_elevation_mask[i])
            )
            mask[i] = keep_source_identity
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
    n_pairs_after_target_elevation_filter = int(len(src_idx))
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = np.divide(
            target_stec[tgt_idx],
            source_stec[src_idx],
            out=np.full(len(src_idx), np.nan, dtype=np.float64),
            where=np.abs(source_stec[src_idx]) > 1e-12,
        )
    slm_pair_ratio = None
    if target_mode == "source_ratio":
        linear_labels = raw_ratio
    else:
        assert mapping is not None
        with np.errstate(divide="ignore", invalid="ignore"):
            slm_pair_ratio = np.divide(
                mapping[tgt_idx],
                mapping[src_idx],
                out=np.full(len(src_idx), np.nan, dtype=np.float64),
                where=mapping[src_idx] > 0,
            )
            linear_labels = np.divide(
                raw_ratio,
                slm_pair_ratio,
                out=np.full(len(src_idx), np.nan, dtype=np.float64),
                where=np.abs(slm_pair_ratio) > 1e-12,
            )
    labels, transform_finite = transform_training_labels(
        linear_labels,
        target_mode=target_mode,
        target_transform=target_transform,
    )
    finite = (
        transform_finite
        & np.isfinite(linear_labels)
        & np.isfinite(raw_ratio)
        & np.isfinite(query_geom[tgt_idx]).all(axis=1)
    )
    if slm_pair_ratio is not None:
        finite &= np.isfinite(slm_pair_ratio) & (slm_pair_ratio > 0)
    src_idx = src_idx[finite]
    tgt_idx = tgt_idx[finite]
    labels = labels[finite]
    linear_labels = linear_labels[finite]
    raw_ratio = raw_ratio[finite]
    target_pair_elevation = target_elevation[tgt_idx]
    if slm_pair_ratio is not None:
        slm_pair_ratio = slm_pair_ratio[finite]

    if max_pair_examples is not None and len(labels) > max_pair_examples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(labels), size=int(max_pair_examples), replace=False)
        src_idx = src_idx[keep]
        tgt_idx = tgt_idx[keep]
        labels = labels[keep]
        linear_labels = linear_labels[keep]
        raw_ratio = raw_ratio[keep]
        target_pair_elevation = target_pair_elevation[keep]
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
    identity_below_target_elevation = np.zeros_like(identity, dtype=bool)
    if target_elevation_threshold is not None:
        identity_below_target_elevation = identity & (
            target_pair_elevation < target_elevation_threshold
        )
    stats = {
        "target_mode": target_mode,
        "target_transform": target_transform,
        "min_target_elevation_deg": target_elevation_threshold,
        "target_elevation_keep_identity": bool(target_elevation_keep_identity),
        "training_label_stats": _finite_stats(labels[valid]),
        "linear_training_label_stats": _finite_stats(linear_labels[valid]),
        "raw_source_ratio_stats": _finite_stats(raw_ratio[valid]),
        "slm_pair_ratio_stats": _finite_stats(slm_pair_ratio[valid])
        if slm_pair_ratio is not None else None,
        "target_elevation_stats": _finite_stats(target_pair_elevation[valid]),
        "n_pairs_after_target_elevation_filter": n_pairs_after_target_elevation_filter,
        "n_pairs_before_context_filter": int(n_examples),
        "n_pairs_after_context_filter": int(valid.sum()),
        "n_identity_pairs": int(np.count_nonzero(identity[valid])),
        "n_identity_below_target_elevation": int(
            np.count_nonzero(identity_below_target_elevation[valid])
        ),
        "identity_pair_fraction": float(np.count_nonzero(identity[valid]) / max(valid.sum(), 1)),
        "mean_context_neighbors": float(np.mean(n_context[valid])) if valid.any() else 0.0,
        "min_context_neighbors": int(np.min(n_context[valid])) if valid.any() else 0,
        "max_context_neighbors": int(np.max(n_context[valid])) if valid.any() else 0,
        "max_pair_examples": int(max_pair_examples) if max_pair_examples is not None else None,
        "same_epoch_pairs": False,
        "_identity_pair_mask": identity[valid],
        "_identity_below_target_elevation_mask": identity_below_target_elevation[valid],
        "_target_elevation_values": target_pair_elevation[valid],
    }
    return QG[valid], OF[valid], OM[valid], OD[valid], labels[valid], stats


def build_pair_set_ratio_examples_same_epoch(
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
    min_target_elevation_deg: float | None = None,
    target_elevation_keep_identity: bool = False,
    mapping_col: str | None = None,
    target_mode: str = "source_ratio",
    target_transform: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build set-ratio examples independently inside each timestamp."""

    if "timestamp" not in df.columns:
        raise KeyError("--same-epoch-pairs requires a timestamp column.")
    groups = [
        (pd.Timestamp(ts), group.copy())
        for ts, group in df.groupby(pd.to_datetime(df["timestamp"]), sort=True)
        if len(group) >= 2
    ]
    if not groups:
        raise ValueError("No timestamp group contains at least two training rays.")

    per_epoch_cap = None
    if max_pair_examples is not None:
        per_epoch_cap = max(1, int(np.ceil(int(max_pair_examples) / len(groups))))

    QG_parts: list[np.ndarray] = []
    OF_parts: list[np.ndarray] = []
    OM_parts: list[np.ndarray] = []
    OD_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []
    identity_parts: list[np.ndarray] = []
    identity_below_target_parts: list[np.ndarray] = []
    target_elevation_parts: list[np.ndarray] = []
    epoch_stats: list[dict[str, Any]] = []
    total_after_target_elevation_filter = 0
    total_before_context = 0
    total_after_context = 0

    for idx, (timestamp, group) in enumerate(groups):
        QG, OF, OM, OD, Y, stats = build_pair_set_ratio_examples(
            group,
            source_stec_col=source_stec_col,
            target_stec_col=target_stec_col,
            obs_cols=obs_cols,
            max_obs=max_obs,
            context_radius_km=context_radius_km,
            pair_radius_km=pair_radius_km,
            max_target_neighbors=max_target_neighbors,
            include_identity=include_identity,
            max_pair_examples=per_epoch_cap,
            seed=seed + idx,
            min_target_elevation_deg=min_target_elevation_deg,
            target_elevation_keep_identity=target_elevation_keep_identity,
            mapping_col=mapping_col,
            target_mode=target_mode,
            target_transform=target_transform,
        )
        QG_parts.append(QG)
        OF_parts.append(OF)
        OM_parts.append(OM)
        OD_parts.append(OD)
        Y_parts.append(Y)
        identity_parts.append(stats["_identity_pair_mask"])
        identity_below_target_parts.append(
            stats["_identity_below_target_elevation_mask"]
        )
        target_elevation_parts.append(stats["_target_elevation_values"])
        total_after_target_elevation_filter += int(
            stats["n_pairs_after_target_elevation_filter"]
        )
        total_before_context += int(stats["n_pairs_before_context_filter"])
        total_after_context += int(stats["n_pairs_after_context_filter"])
        epoch_stats.append(
            {
                "timestamp": timestamp.isoformat(),
                "rows": int(len(group)),
                "n_pairs_before_context_filter": int(
                    stats["n_pairs_before_context_filter"]
                ),
                "n_pairs_after_target_elevation_filter": int(
                    stats["n_pairs_after_target_elevation_filter"]
                ),
                "n_pairs_after_context_filter": int(
                    stats["n_pairs_after_context_filter"]
                ),
                "target_elevation_stats": stats["target_elevation_stats"],
                "n_identity_pairs": int(stats["n_identity_pairs"]),
                "n_identity_below_target_elevation": int(
                    stats["n_identity_below_target_elevation"]
                ),
                "mean_context_neighbors": float(stats["mean_context_neighbors"]),
            }
        )

    QG = np.concatenate(QG_parts, axis=0)
    OF = np.concatenate(OF_parts, axis=0)
    OM = np.concatenate(OM_parts, axis=0)
    OD = np.concatenate(OD_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    identity = np.concatenate(identity_parts, axis=0)
    identity_below_target_elevation = np.concatenate(identity_below_target_parts, axis=0)
    target_elevation_values = np.concatenate(target_elevation_parts, axis=0)

    if max_pair_examples is not None and len(Y) > int(max_pair_examples):
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(Y), size=int(max_pair_examples), replace=False)
        QG = QG[keep]
        OF = OF[keep]
        OM = OM[keep]
        OD = OD[keep]
        Y = Y[keep]
        identity = identity[keep]
        identity_below_target_elevation = identity_below_target_elevation[keep]
        target_elevation_values = target_elevation_values[keep]

    n_context = OM.sum(axis=1)
    stats = {
        "target_mode": target_mode,
        "target_transform": target_transform,
        "same_epoch_pairs": True,
        "n_training_epochs": int(len(groups)),
        "n_cross_epoch_pairs": 0,
        "same_epoch_per_epoch_cap": per_epoch_cap,
        "min_target_elevation_deg": None
        if min_target_elevation_deg is None else float(min_target_elevation_deg),
        "target_elevation_keep_identity": bool(target_elevation_keep_identity),
        "training_label_stats": _finite_stats(Y),
        "raw_source_ratio_stats": None,
        "slm_pair_ratio_stats": None,
        "target_elevation_stats": _finite_stats(target_elevation_values),
        "n_pairs_after_target_elevation_filter": total_after_target_elevation_filter,
        "n_pairs_before_context_filter": total_before_context,
        "n_pairs_after_context_filter": int(len(Y)),
        "n_pairs_after_epoch_cap_pre_global_cap": total_after_context,
        "n_identity_pairs": int(np.count_nonzero(identity)),
        "n_identity_below_target_elevation": int(
            np.count_nonzero(identity_below_target_elevation)
        ),
        "identity_pair_fraction": float(np.count_nonzero(identity) / max(len(Y), 1)),
        "mean_context_neighbors": float(np.mean(n_context)) if len(n_context) else 0.0,
        "min_context_neighbors": int(np.min(n_context)) if len(n_context) else 0,
        "max_context_neighbors": int(np.max(n_context)) if len(n_context) else 0,
        "max_pair_examples": int(max_pair_examples) if max_pair_examples is not None else None,
        "epoch_pair_stats": epoch_stats,
        "_identity_pair_mask": identity,
        "_identity_below_target_elevation_mask": identity_below_target_elevation,
        "_target_elevation_values": target_elevation_values,
    }
    return QG, OF, OM, OD, Y, stats


def build_dynamic_pair_pool_same_epoch(
    df: pd.DataFrame,
    *,
    source_stec_col: str,
    target_stec_col: str,
    obs_cols: list[str],
    pair_radius_km: float,
    max_target_neighbors: int,
    include_identity: bool,
    min_target_elevation_deg: float | None = None,
    target_elevation_keep_identity: bool = False,
    mapping_col: str | None = None,
    target_mode: str = "source_ratio",
    target_transform: str = "linear",
) -> dict[str, Any]:
    """Build an index-only same-epoch pair pool for dynamic resampling."""

    if "timestamp" not in df.columns:
        raise KeyError("--dynamic-pair-resampling requires --same-epoch-pairs and timestamp.")
    n = len(df)
    if n < 2:
        raise ValueError("Dynamic set-ratio training requires at least 2 rows.")

    timestamp_values = pd.to_datetime(df["timestamp"]).to_numpy()
    epoch_series = pd.Series(timestamp_values)
    groups = [
        (pd.Timestamp(ts), np.asarray(pos, dtype=np.intp))
        for ts, pos in sorted(epoch_series.groupby(epoch_series).indices.items())
        if len(pos) >= 2
    ]
    if not groups:
        raise ValueError("No timestamp group contains at least two training rays.")

    ipp_lon = df["ipp_lon"].to_numpy(dtype=np.float64)
    ipp_lat = df["ipp_lat"].to_numpy(dtype=np.float64)
    query_geom = df[QUERY_FEATURE_COLS].to_numpy(dtype=np.float64)
    context_base = df[obs_cols].to_numpy(dtype=np.float64)
    index_values = df.index.to_numpy()
    source_stec = df[source_stec_col].to_numpy(dtype=np.float64)
    target_stec = df[target_stec_col].to_numpy(dtype=np.float64)
    target_elevation = df["elevation"].to_numpy(dtype=np.float64)
    target_elevation_threshold = (
        None if min_target_elevation_deg is None else float(min_target_elevation_deg)
    )
    target_elevation_mask = np.ones(n, dtype=bool)
    if target_elevation_threshold is not None:
        target_elevation_mask = target_elevation >= target_elevation_threshold

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
    epoch_parts: list[np.ndarray] = []
    epoch_pre_counts: list[int] = []

    for epoch_id, (_, locs) in enumerate(groups):
        pre_count = 0
        epoch_ipp_lon = ipp_lon[locs]
        epoch_ipp_lat = ipp_lat[locs]
        for local_src, src in enumerate(locs):
            dists = _haversine_km(
                float(ipp_lon[src]),
                float(ipp_lat[src]),
                epoch_ipp_lon,
                epoch_ipp_lat,
            )
            mask = dists <= pair_radius_km
            if target_elevation_threshold is not None:
                mask &= target_elevation_mask[locs]
            if include_identity:
                keep_source_identity = (
                    target_elevation_threshold is None
                    or target_elevation_keep_identity
                    or bool(target_elevation_mask[src])
                )
                mask[local_src] = keep_source_identity
            else:
                mask[local_src] = False
            candidates_local = np.where(mask)[0]
            if len(candidates_local) > max_target_neighbors:
                order = np.argsort(dists[candidates_local])
                candidates_local = candidates_local[order[:max_target_neighbors]]
            if len(candidates_local) == 0:
                continue
            targets = locs[candidates_local].astype(np.intp)
            pre_count += int(len(targets))
            src_parts.append(np.full(len(targets), int(src), dtype=np.intp))
            tgt_parts.append(targets)
            epoch_parts.append(np.full(len(targets), epoch_id, dtype=np.intp))
        epoch_pre_counts.append(pre_count)

    if not src_parts:
        raise ValueError("No dynamic pair pool examples were created.")

    src_idx = np.concatenate(src_parts)
    tgt_idx = np.concatenate(tgt_parts)
    epoch_id = np.concatenate(epoch_parts)
    n_pairs_after_target_elevation_filter = int(len(src_idx))

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = np.divide(
            target_stec[tgt_idx],
            source_stec[src_idx],
            out=np.full(len(src_idx), np.nan, dtype=np.float64),
            where=np.abs(source_stec[src_idx]) > 1e-12,
        )
    slm_pair_ratio = None
    if target_mode == "source_ratio":
        linear_labels = raw_ratio
    else:
        assert mapping is not None
        with np.errstate(divide="ignore", invalid="ignore"):
            slm_pair_ratio = np.divide(
                mapping[tgt_idx],
                mapping[src_idx],
                out=np.full(len(src_idx), np.nan, dtype=np.float64),
                where=mapping[src_idx] > 0,
            )
            linear_labels = np.divide(
                raw_ratio,
                slm_pair_ratio,
                out=np.full(len(src_idx), np.nan, dtype=np.float64),
                where=np.abs(slm_pair_ratio) > 1e-12,
            )
    labels, transform_finite = transform_training_labels(
        linear_labels,
        target_mode=target_mode,
        target_transform=target_transform,
    )
    finite = (
        transform_finite
        & np.isfinite(linear_labels)
        & np.isfinite(raw_ratio)
        & np.isfinite(query_geom[tgt_idx]).all(axis=1)
        & np.isfinite(context_base[src_idx]).all(axis=1)
    )
    if slm_pair_ratio is not None:
        finite &= np.isfinite(slm_pair_ratio) & (slm_pair_ratio > 0)

    src_idx = src_idx[finite]
    tgt_idx = tgt_idx[finite]
    epoch_id = epoch_id[finite]
    labels = labels[finite]
    linear_labels = linear_labels[finite]
    raw_ratio = raw_ratio[finite]
    target_pair_elevation = target_elevation[tgt_idx]
    if slm_pair_ratio is not None:
        slm_pair_ratio = slm_pair_ratio[finite]

    identity = src_idx == tgt_idx
    identity_below_target_elevation = np.zeros_like(identity, dtype=bool)
    if target_elevation_threshold is not None:
        identity_below_target_elevation = identity & (
            target_pair_elevation < target_elevation_threshold
        )

    epoch_pair_indices: list[np.ndarray] = []
    epoch_stats: list[dict[str, Any]] = []
    for epoch_idx, (timestamp, locs) in enumerate(groups):
        pair_locs = np.flatnonzero(epoch_id == epoch_idx)
        epoch_pair_indices.append(pair_locs.astype(np.intp))
        epoch_stats.append(
            {
                "timestamp": timestamp.isoformat(),
                "rows": int(len(locs)),
                "n_pairs_after_target_elevation_filter": int(
                    epoch_pre_counts[epoch_idx]
                ),
                "n_pairs_after_context_filter": int(len(pair_locs)),
                "n_identity_pairs": int(np.count_nonzero(identity[pair_locs])),
                "n_identity_below_target_elevation": int(
                    np.count_nonzero(identity_below_target_elevation[pair_locs])
                ),
            }
        )

    stats = {
        "target_mode": target_mode,
        "target_transform": target_transform,
        "same_epoch_pairs": True,
        "dynamic_pair_resampling": True,
        "n_training_epochs": int(len(groups)),
        "n_cross_epoch_pairs": 0,
        "same_epoch_per_epoch_cap": None,
        "min_target_elevation_deg": target_elevation_threshold,
        "target_elevation_keep_identity": bool(target_elevation_keep_identity),
        "training_label_stats": _finite_stats(labels),
        "linear_training_label_stats": _finite_stats(linear_labels),
        "raw_source_ratio_stats": _finite_stats(raw_ratio),
        "slm_pair_ratio_stats": _finite_stats(slm_pair_ratio)
        if slm_pair_ratio is not None else None,
        "target_elevation_stats": _finite_stats(target_pair_elevation),
        "n_pairs_after_target_elevation_filter": n_pairs_after_target_elevation_filter,
        "n_pairs_before_context_filter": int(len(labels)),
        "n_pairs_after_context_filter": int(len(labels)),
        "n_pairs_after_epoch_cap_pre_global_cap": int(len(labels)),
        "n_identity_pairs": int(np.count_nonzero(identity)),
        "n_identity_below_target_elevation": int(
            np.count_nonzero(identity_below_target_elevation)
        ),
        "identity_pair_fraction": float(np.count_nonzero(identity) / max(len(labels), 1)),
        "mean_context_neighbors": 1.0,
        "min_context_neighbors": 1 if len(labels) else 0,
        "max_context_neighbors": 1 if len(labels) else 0,
        "max_pair_examples": None,
        "pair_pool_examples": int(len(labels)),
        "epoch_pair_stats": epoch_stats,
    }
    return {
        "src_idx": src_idx.astype(np.intp),
        "tgt_idx": tgt_idx.astype(np.intp),
        "epoch_id": epoch_id.astype(np.intp),
        "labels": labels.astype(np.float64),
        "linear_labels": linear_labels.astype(np.float64),
        "raw_ratio": raw_ratio.astype(np.float64),
        "slm_pair_ratio": None if slm_pair_ratio is None else slm_pair_ratio.astype(np.float64),
        "identity": identity,
        "identity_below_target_elevation": identity_below_target_elevation,
        "target_pair_elevation": target_pair_elevation.astype(np.float64),
        "epoch_pair_indices": epoch_pair_indices,
        "query_geom": query_geom,
        "context_base": context_base,
        "ipp_lon": ipp_lon,
        "ipp_lat": ipp_lat,
        "index_values": index_values,
        "obs_cols": obs_cols,
        "stats": stats,
    }


def sample_dynamic_pair_indices(
    pool: dict[str, Any],
    *,
    n_examples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample pair-pool rows approximately evenly across timestamps."""

    if n_examples <= 0:
        raise ValueError("--dynamic-pair-examples must be positive.")
    epoch_pair_indices = [
        np.asarray(indices, dtype=np.intp)
        for indices in pool["epoch_pair_indices"]
        if len(indices) > 0
    ]
    if not epoch_pair_indices:
        raise ValueError("Dynamic pair pool is empty after filtering.")
    rng = np.random.default_rng(seed)
    per_epoch_cap = max(1, int(np.ceil(int(n_examples) / len(epoch_pair_indices))))
    sampled_parts: list[np.ndarray] = []
    sampled_counts: list[int] = []
    for indices in epoch_pair_indices:
        k = min(len(indices), per_epoch_cap)
        if k < len(indices):
            chosen = rng.choice(indices, size=k, replace=False)
        else:
            chosen = indices.copy()
        sampled_parts.append(chosen.astype(np.intp))
        sampled_counts.append(int(len(chosen)))
    sampled = np.concatenate(sampled_parts)
    if len(sampled) > int(n_examples):
        sampled = rng.choice(sampled, size=int(n_examples), replace=False)
    rng.shuffle(sampled)
    return sampled.astype(np.intp), {
        "requested_pairs": int(n_examples),
        "actual_pairs": int(len(sampled)),
        "active_epochs": int(len(epoch_pair_indices)),
        "per_epoch_cap": int(per_epoch_cap),
        "min_pairs_per_epoch_before_global_cap": int(np.min(sampled_counts)),
        "max_pairs_per_epoch_before_global_cap": int(np.max(sampled_counts)),
        "mean_pairs_per_epoch_before_global_cap": float(np.mean(sampled_counts)),
        "seed": int(seed),
    }


def materialize_pair_pool_examples(
    pool: dict[str, Any],
    pair_indices: np.ndarray,
    *,
    max_obs: int,
    context_radius_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn sampled pair-pool rows into set-model tensors."""

    pair_indices = np.asarray(pair_indices, dtype=np.intp)
    src_idx = pool["src_idx"][pair_indices]
    tgt_idx = pool["tgt_idx"][pair_indices]
    query_geom = pool["query_geom"]
    context_base = pool["context_base"]
    ipp_lon = pool["ipp_lon"]
    ipp_lat = pool["ipp_lat"]
    index_values = pool["index_values"]
    labels = pool["labels"][pair_indices]

    n_examples = int(len(pair_indices))
    n_obs_features = int(context_base.shape[1]) + 1
    QG = query_geom[tgt_idx].astype(np.float64, copy=True)
    OF = np.zeros((n_examples, max_obs, n_obs_features), dtype=np.float64)
    OM = np.zeros((n_examples, max_obs), dtype=bool)
    OD = np.full((n_examples, max_obs), np.inf, dtype=np.float64)

    if max_obs == 1 and context_radius_km <= 0:
        OF[:, 0, :-1] = context_base[src_idx]
        OF[:, 0, -1] = 1.0
        OM[:, 0] = True
        OD[:, 0] = _haversine_km(
            query_geom[tgt_idx, QUERY_FEATURE_COLS.index("ipp_lon")],
            query_geom[tgt_idx, QUERY_FEATURE_COLS.index("ipp_lat")],
            ipp_lon[src_idx],
            ipp_lat[src_idx],
        )
        return QG, OF, OM, OD, labels.astype(np.float64, copy=True)

    context_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row, (src, tgt) in enumerate(zip(src_idx, tgt_idx)):
        src_int = int(src)
        if src_int not in context_cache:
            context_cache[src_int] = _context_indices_for_source(
                src_int,
                source_ipp_lon=float(ipp_lon[src]),
                source_ipp_lat=float(ipp_lat[src]),
                context_ipp_lon=ipp_lon,
                context_ipp_lat=ipp_lat,
                context_index=index_values,
                source_index_value=index_values[src],
                radius_km=context_radius_km,
                max_obs=max_obs,
            )
        candidates, _ = context_cache[src_int]
        _fill_context_tensor(
            out_features=OF,
            out_mask=OM,
            out_dists=OD,
            row=row,
            candidates=candidates,
            source_index_value=index_values[src],
            context_index=index_values,
            context_base_features=context_base,
            query_ipp_lon=float(query_geom[tgt, QUERY_FEATURE_COLS.index("ipp_lon")]),
            query_ipp_lat=float(query_geom[tgt, QUERY_FEATURE_COLS.index("ipp_lat")]),
            context_ipp_lon=ipp_lon,
            context_ipp_lat=ipp_lat,
        )
    valid = OM.any(axis=1)
    if not valid.all():
        QG = QG[valid]
        OF = OF[valid]
        OM = OM[valid]
        OD = OD[valid]
        labels = labels[valid]
    return QG, OF, OM, OD, labels.astype(np.float64, copy=True)
def build_inference_context_sets(
    query_df: pd.DataFrame,
    context_df: pd.DataFrame,
    *,
    target_geom: np.ndarray,
    obs_cols: list[str],
    max_obs: int,
    context_radius_km: float,
    same_epoch_context: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build source-centered contexts for evaluation rows."""

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

    query_ts = None
    context_ts = None
    if same_epoch_context:
        if "timestamp" not in query_df.columns or "timestamp" not in context_df.columns:
            raise KeyError("--same-epoch-pairs requires timestamp for inference context.")
        query_ts = pd.to_datetime(query_df["timestamp"]).to_numpy()
        context_ts = pd.to_datetime(context_df["timestamp"]).to_numpy()

    OF = np.zeros((n_query, max_obs, n_obs_features), dtype=np.float64)
    OM = np.zeros((n_query, max_obs), dtype=bool)
    OD = np.full((n_query, max_obs), np.inf, dtype=np.float64)
    n_context = np.zeros(n_query, dtype=int)
    has_anchor = np.zeros(n_query, dtype=bool)
    missing_epoch_context = 0

    for i in range(n_query):
        if same_epoch_context:
            assert query_ts is not None and context_ts is not None
            epoch_locs = np.flatnonzero(context_ts == query_ts[i])
            if len(epoch_locs) == 0:
                missing_epoch_context += 1
                continue
            local_candidates, _ = _context_indices_for_source(
                i,
                source_ipp_lon=float(source_ipp_lon[i]),
                source_ipp_lat=float(source_ipp_lat[i]),
                context_ipp_lon=context_ipp_lon[epoch_locs],
                context_ipp_lat=context_ipp_lat[epoch_locs],
                context_index=context_index[epoch_locs],
                source_index_value=source_index[i],
                radius_km=context_radius_km,
                max_obs=max_obs,
            )
            candidates = epoch_locs[local_candidates]
        else:
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
        "same_epoch_context": bool(same_epoch_context),
        "n_missing_epoch_context": int(missing_epoch_context),
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



def scale_obs_tensor(OF: np.ndarray, OM: np.ndarray, o_imp: Any, o_scaler: Any) -> np.ndarray:
    """Scale observation tensors, vectorized for the common max_obs=1 case."""

    if OF.shape[1] == 1:
        out = np.zeros_like(OF)
        valid = OM[:, 0]
        if valid.any():
            out[valid, 0] = o_scaler.transform(o_imp.transform(OF[valid, 0]))
        return out
    return _scale_obs(OF, OM, o_imp, o_scaler)
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

def parse_epoch_metrics_epochs(spec: str | None, max_epoch: int) -> set[int]:
    """Parse a comma-separated list of epochs for checkpoint diagnostics."""

    if not spec:
        return set()
    epochs: set[int] = set()
    for raw in str(spec).split(","):
        item = raw.strip()
        if not item:
            continue
        epoch = int(item)
        if epoch < 1 or epoch > max_epoch:
            raise ValueError(
                f"--epoch-metrics-epochs contains {epoch}, but valid range is "
                f"1..{max_epoch}."
            )
        epochs.add(epoch)
    return epochs


def should_record_epoch_metrics(
    epoch: int,
    *,
    enabled: bool,
    every: int,
    selected_epochs: set[int],
    final_epoch: int,
) -> bool:
    """Return whether the current epoch should run VTEC diagnostics."""

    if not enabled:
        return False
    if selected_epochs:
        return epoch in selected_epochs
    if every < 1:
        raise ValueError("--epoch-metrics-every must be >= 1.")
    return epoch % every == 0 or epoch == final_epoch


def predict_model_output_label(
    *,
    model,
    device,
    q_geom_scaled: np.ndarray,
    obs_feat_scaled: np.ndarray,
    obs_mask: np.ndarray,
    obs_dists: np.ndarray,
    identity_delta: np.ndarray,
    y_scaler,
    hard_identity_ratio: bool,
    batch_size: int,
) -> np.ndarray:
    """Predict either hard-identity labels or inverse-scaled model outputs."""

    if hard_identity_ratio:
        return _predict_hard_identity_ratio(
            model,
            device,
            q_geom_scaled,
            obs_feat_scaled,
            obs_mask,
            obs_dists,
            identity_delta,
            batch_size=batch_size,
        )
    pred_scaled = _predict_set(
        model,
        device,
        q_geom_scaled,
        obs_feat_scaled,
        obs_mask,
        obs_dists,
        batch_size=batch_size,
    )
    return y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


def predict_set_ratio_conversion(
    *,
    model,
    device,
    y_scaler,
    hard_identity_ratio: bool,
    target_mode: str,
    target_transform: str,
    batch_size: int,
    slant_QG_scaled: np.ndarray,
    slant_OF_scaled: np.ndarray,
    slant_OM: np.ndarray,
    slant_OD: np.ndarray,
    slant_identity_delta: np.ndarray,
    vert_QG_scaled: np.ndarray,
    vert_OF_scaled: np.ndarray,
    vert_OM: np.ndarray,
    vert_OD: np.ndarray,
    vertical_identity_delta: np.ndarray,
    slm_slant_ratio: np.ndarray,
    slm_vertical_ratio: np.ndarray,
    shrinkage_strength: np.ndarray,
    source_stec_eval: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run slant and vertical queries and convert source STEC to VTEC."""

    slant_model_output = predict_model_output_label(
        model=model,
        device=device,
        q_geom_scaled=slant_QG_scaled,
        obs_feat_scaled=slant_OF_scaled,
        obs_mask=slant_OM,
        obs_dists=slant_OD,
        identity_delta=slant_identity_delta,
        y_scaler=y_scaler,
        hard_identity_ratio=hard_identity_ratio,
        batch_size=batch_size,
    )
    vertical_model_output = predict_model_output_label(
        model=model,
        device=device,
        q_geom_scaled=vert_QG_scaled,
        obs_feat_scaled=vert_OF_scaled,
        obs_mask=vert_OM,
        obs_dists=vert_OD,
        identity_delta=vertical_identity_delta,
        y_scaler=y_scaler,
        hard_identity_ratio=hard_identity_ratio,
        batch_size=batch_size,
    )

    if target_mode == "source_ratio":
        slant_model_output = np.maximum(slant_model_output, 0.0)
        vertical_model_output = np.maximum(vertical_model_output, 0.0)
        slant_ratio = slant_model_output
        vertical_ratio = vertical_model_output
        with np.errstate(divide="ignore", invalid="ignore"):
            slant_correction_raw = np.divide(slant_ratio, slm_slant_ratio)
            vertical_correction_raw = np.divide(vertical_ratio, slm_vertical_ratio)
        slant_correction = slant_correction_raw
        vertical_correction = vertical_correction_raw
    else:
        if target_transform == "log":
            slant_correction_raw = np.exp(slant_model_output)
            vertical_correction_raw = np.exp(vertical_model_output)
        elif target_transform == "linear":
            slant_model_output = np.maximum(slant_model_output, 0.0)
            vertical_model_output = np.maximum(vertical_model_output, 0.0)
            slant_correction_raw = slant_model_output
            vertical_correction_raw = vertical_model_output
        else:
            raise ValueError(f"Unsupported target transform: {target_transform}")
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
    return {
        "slant_model_output": slant_model_output,
        "vertical_model_output": vertical_model_output,
        "slant_correction_raw": slant_correction_raw,
        "vertical_correction_raw": vertical_correction_raw,
        "slant_correction": slant_correction,
        "vertical_correction": vertical_correction,
        "slant_ratio": slant_ratio,
        "vertical_ratio": vertical_ratio,
        "stec_set_ratio_mlp": source_stec_eval * slant_ratio,
        "vtec_set_ratio_mlp": source_stec_eval * vertical_ratio,
    }


def build_epoch_metric_row(
    *,
    epoch: int,
    train_loss: float,
    target_vtec: pd.Series | np.ndarray,
    predicted_vtec: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Create one compact row for epoch-level VTEC diagnostics."""

    metrics = vtec_error_metrics(target_vtec, predicted_vtec)
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "rmse_tecu": float(metrics["rmse_tecu"]),
        "mae_tecu": float(metrics["mae_tecu"]),
        "bias_tecu": float(metrics["bias_tecu"]),
        "r2": float(metrics["r2"]),
        "n_eval": int(metrics["n_samples"]),
    }


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



def _make_tensor_loader(q_geom, obs_feat, obs_mask, obs_dists, labels, batch_size, seed, shuffle=True):
    """Build a DataLoader from tensors to avoid Python per-sample collation."""

    g = torch.Generator()
    g.manual_seed(seed)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(q_geom, dtype=torch.float32),
        torch.tensor(obs_feat, dtype=torch.float32),
        torch.tensor(obs_mask, dtype=torch.bool),
        torch.tensor(obs_dists, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32).view(-1, 1),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Choose train, validation, and locked-test rows.

    Key-based validation is the publication protocol. The older two-way split
    remains available so historical commands keep working, but it is marked as
    legacy in metadata and cannot drive best-checkpoint selection.
    """

    validation_keys_from = getattr(args, "validation_keys_from", None)
    validation_key_columns = getattr(
        args, "validation_key_columns", "timestamp,station,prn"
    )

    def key_mask(
        key_path_raw: str,
        key_columns_raw: str,
        *,
        marker: str,
    ) -> tuple[np.ndarray, list[str]]:
        key_df = read_table(Path(key_path_raw))
        key_columns = [
            column.strip()
            for column in key_columns_raw.split(",")
            if column.strip()
        ]
        if not key_columns:
            raise ValueError("Key holdout requires at least one key column.")
        require_columns(df, key_columns)
        require_columns(key_df, key_columns)
        left = df.loc[:, key_columns].copy()
        right = key_df.loc[:, key_columns].drop_duplicates().copy()
        if "timestamp" in key_columns:
            left["timestamp"] = pd.to_datetime(left["timestamp"])
            right["timestamp"] = pd.to_datetime(right["timestamp"])
        matched = left.reset_index(drop=True).merge(
            right.assign(**{marker: True}),
            on=key_columns,
            how="left",
            sort=False,
        )
        return matched[marker].fillna(False).to_numpy(bool), key_columns

    if validation_keys_from:
        if args.split == "all":
            raise ValueError(
                "--validation-keys-from cannot be used with --split all."
            )
        validation_mask, validation_columns = key_mask(
            validation_keys_from,
            validation_key_columns,
            marker="_set_ratio_validation_key",
        )
        test_mask = np.zeros(len(df), dtype=bool)
        test_columns: list[str] | None = None
        if args.test_keys_from:
            test_mask, test_columns = key_mask(
                args.test_keys_from,
                args.test_key_columns,
                marker="_set_ratio_test_key",
            )
        overlap = validation_mask & test_mask
        if np.any(overlap):
            raise ValueError(
                "Validation and locked-test key files overlap: "
                f"{int(np.count_nonzero(overlap))} input rows match both."
            )
        train_mask = ~(validation_mask | test_mask)
        train_df = df.loc[train_mask].copy()
        validation_df = df.loc[validation_mask].copy()
        test_df = df.loc[test_mask].copy()
        if train_df.empty or validation_df.empty:
            raise ValueError(
                "Three-way key protocol requires non-empty train and "
                f"validation rows; train={len(train_df)}, "
                f"validation={len(validation_df)}."
            )
        if args.test_keys_from and test_df.empty:
            raise ValueError("Locked-test key file matched no input rows.")
        protocol = (
            "train_validation_locked_test"
            if args.test_keys_from
            else "train_validation"
        )
        return train_df, validation_df, test_df, {
            "protocol": protocol,
            "type": "key_holdout",
            "validation_keys_from": str(validation_keys_from),
            "validation_key_columns": validation_columns,
            "test_keys_from": (
                None if not args.test_keys_from else str(args.test_keys_from)
            ),
            "test_key_columns": test_columns,
            "n_train": int(len(train_df)),
            "n_validation": int(len(validation_df)),
            "n_test": int(len(test_df)),
            "train_validation_overlap": False,
            "train_test_overlap": False,
            "validation_test_overlap": False,
            "test_role": (
                "locked_final_evaluation" if len(test_df) else "not_configured"
            ),
        }

    if args.split == "all":
        if args.test_keys_from:
            raise ValueError("--test-keys-from cannot be used with --split all.")
        full_df = df.copy()
        return full_df, df.iloc[0:0].copy(), full_df.copy(), {
            "protocol": "legacy_all_data_diagnostic",
            "type": "all",
            "n_train": int(len(full_df)),
            "n_validation": 0,
            "n_test": int(len(full_df)),
            "train_test_overlap": True,
            "interpretation": (
                "Diagnostic full-data fit/evaluation: all rays are used for "
                "training and all rays are evaluated. VTEC truth remains "
                "evaluation-only and is not used as a training label."
            ),
        }
    train_df, test_df, metadata = choose_simulation_split(df, args)
    metadata = dict(metadata)
    metadata.update(
        {
            "protocol": "legacy_two_way",
            "n_validation": 0,
            "interpretation": (
                "Backward-compatible two-way split. This test/evaluation set "
                "must not be used for publication checkpoint selection."
            ),
        }
    )
    return train_df, df.iloc[0:0].copy(), test_df, metadata


def apply_train_window_filter(
    train_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Optionally restrict training rows to a half-open timestamp window."""

    start_raw = args.train_window_start
    end_raw = args.train_window_end
    before = int(len(train_df))
    if not start_raw and not end_raw:
        timestamps = pd.to_datetime(train_df["timestamp"]) if "timestamp" in train_df.columns else None
        return train_df, {
            "enabled": False,
            "train_window_start": None,
            "train_window_end": None,
            "train_rows_before_window": before,
            "train_rows_after_window": before,
            "n_train_epochs_after_window": int(timestamps.nunique()) if timestamps is not None else None,
            "train_timestamp_min": None if timestamps is None or timestamps.empty else pd.Timestamp(timestamps.min()).isoformat(),
            "train_timestamp_max": None if timestamps is None or timestamps.empty else pd.Timestamp(timestamps.max()).isoformat(),
        }
    if not start_raw or not end_raw:
        raise ValueError("Both --train-window-start and --train-window-end are required.")
    if "timestamp" not in train_df.columns:
        raise KeyError("--train-window-start/end requires a timestamp column.")

    start = pd.Timestamp(start_raw)
    end = pd.Timestamp(end_raw)
    if end <= start:
        raise ValueError("--train-window-end must be later than --train-window-start.")

    timestamps = pd.to_datetime(train_df["timestamp"])
    mask = (timestamps >= start) & (timestamps < end)
    filtered = train_df.loc[mask].copy()
    filtered_ts = pd.to_datetime(filtered["timestamp"]) if not filtered.empty else pd.Series(dtype="datetime64[ns]")
    if len(filtered) < 2:
        raise ValueError(
            "Training time-window filtering left fewer than 2 rows; "
            "widen --train-window-start/end."
        )
    return filtered, {
        "enabled": True,
        "train_window_start": start.isoformat(),
        "train_window_end": end.isoformat(),
        "train_rows_before_window": before,
        "train_rows_after_window": int(len(filtered)),
        "n_train_epochs_after_window": int(filtered_ts.nunique()),
        "train_timestamp_min": None if filtered_ts.empty else pd.Timestamp(filtered_ts.min()).isoformat(),
        "train_timestamp_max": None if filtered_ts.empty else pd.Timestamp(filtered_ts.max()).isoformat(),

    }
def apply_elevation_filters(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Optionally restrict train/validation/test rows by source elevation."""

    train_before = int(len(train_df))
    validation_before = int(len(validation_df))
    test_before = int(len(test_df))
    train_min = args.min_train_elevation_deg
    eval_min = args.min_eval_elevation_deg

    if train_min is not None:
        train_df = train_df.loc[
            pd.to_numeric(train_df["elevation"], errors="coerce") >= float(train_min)
        ].copy()
    if eval_min is not None and not validation_df.empty:
        validation_df = validation_df.loc[
            pd.to_numeric(validation_df["elevation"], errors="coerce")
            >= float(eval_min)
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
    if validation_before and len(validation_df) == 0:
        raise ValueError(
            "Elevation filtering left no validation rays; "
            "relax --min-eval-elevation-deg."
        )
    if test_before and len(test_df) == 0:
        raise ValueError(
            "Elevation filtering left no locked-test rays; "
            "relax --min-eval-elevation-deg."
        )

    return train_df, validation_df, test_df, {
        "min_train_elevation_deg": None if train_min is None else float(train_min),
        "min_eval_elevation_deg": None if eval_min is None else float(eval_min),
        "train_rows_before": train_before,
        "train_rows_after": int(len(train_df)),
        "validation_rows_before": validation_before,
        "validation_rows_after": int(len(validation_df)),
        "test_rows_before": test_before,
        "test_rows_after": int(len(test_df)),
    }

def build_evaluation_context(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    *,
    inference_context: str,
) -> pd.DataFrame:
    """Return context rows without exposing another held-out partition."""

    if inference_context == "train":
        return train_df
    combined = pd.concat([train_df, evaluation_df], axis=0)
    return combined.loc[~combined.index.duplicated(keep="first")].copy()


def prepare_set_ratio_evaluation(
    evaluation_df: pd.DataFrame,
    context_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    obs_cols: list[str],
    q_imp: Any,
    q_scaler: Any,
    o_imp: Any,
    o_scaler: Any,
) -> dict[str, Any]:
    """Prepare scaled geometry and context arrays for one held-out role."""

    slant_geom = evaluation_df[QUERY_FEATURE_COLS].to_numpy(dtype=np.float64)
    vertical_base = build_vertical_target_geometry(
        evaluation_df, vertical_top_km=args.vertical_top_km
    )[TARGET_GEOMETRY_COLUMNS].to_numpy(dtype=np.float64)
    vertical_geom = np.column_stack(
        [vertical_base, evaluation_df[TIME_COLS].to_numpy(dtype=np.float64)]
    )

    slant_obs, slant_mask, slant_dist, slant_context_stats = (
        build_inference_context_sets(
            evaluation_df,
            context_df,
            target_geom=slant_geom,
            obs_cols=obs_cols,
            max_obs=args.max_obs,
            context_radius_km=args.context_km,
            same_epoch_context=args.same_epoch_pairs,
        )
    )
    vertical_obs, vertical_mask, vertical_dist, vertical_context_stats = (
        build_inference_context_sets(
            evaluation_df,
            context_df,
            target_geom=vertical_geom,
            obs_cols=obs_cols,
            max_obs=args.max_obs,
            context_radius_km=args.context_km,
            same_epoch_context=args.same_epoch_pairs,
        )
    )

    slant_geom_scaled = q_scaler.transform(
        q_imp.transform(slant_geom)
    ).astype(np.float64)
    vertical_geom_scaled = q_scaler.transform(
        q_imp.transform(vertical_geom)
    ).astype(np.float64)
    slant_obs_scaled = scale_obs_tensor(
        slant_obs, slant_mask, o_imp, o_scaler
    )
    vertical_obs_scaled = scale_obs_tensor(
        vertical_obs, vertical_mask, o_imp, o_scaler
    )
    slant_identity_delta = compute_identity_delta(
        slant_geom,
        slant_obs,
        slant_mask,
        ipp_scale_km=args.identity_delta_ipp_scale_km,
    )
    vertical_identity_delta = compute_identity_delta(
        vertical_geom,
        vertical_obs,
        vertical_mask,
        ipp_scale_km=args.identity_delta_ipp_scale_km,
    )

    mapping_i = QUERY_FEATURE_COLS.index("slant_factor")
    source_mapping = evaluation_df[args.mapping_column].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        slm_slant_ratio = np.divide(
            slant_geom[:, mapping_i],
            source_mapping,
            out=np.full(len(evaluation_df), np.nan, dtype=np.float64),
            where=source_mapping > 0,
        )
        slm_vertical_ratio = np.divide(
            vertical_geom[:, mapping_i],
            source_mapping,
            out=np.full(len(evaluation_df), np.nan, dtype=np.float64),
            where=source_mapping > 0,
        )
    source_elevation = evaluation_df["elevation"].to_numpy(dtype=np.float64)
    shrinkage_enabled = bool(
        args.elevation_shrinkage and args.target_mode == "slm_correction"
    )
    shrinkage_strength = elevation_shrinkage_strength(
        source_elevation,
        enabled=shrinkage_enabled,
        start_deg=args.shrinkage_start_deg,
        end_deg=args.shrinkage_end_deg,
        power=args.shrinkage_power,
        min_strength=args.shrinkage_min_strength,
    )
    return {
        "df": evaluation_df,
        "slant_QG_scaled": slant_geom_scaled,
        "slant_OF_scaled": slant_obs_scaled,
        "slant_OM": slant_mask,
        "slant_OD": slant_dist,
        "slant_identity_delta": slant_identity_delta,
        "vert_QG_scaled": vertical_geom_scaled,
        "vert_OF_scaled": vertical_obs_scaled,
        "vert_OM": vertical_mask,
        "vert_OD": vertical_dist,
        "vertical_identity_delta": vertical_identity_delta,
        "slm_slant_ratio": slm_slant_ratio,
        "slm_vertical_ratio": slm_vertical_ratio,
        "shrinkage_strength": shrinkage_strength,
        "source_stec_eval": evaluation_df[
            args.source_stec_column
        ].to_numpy(dtype=np.float64),
        "target_vtec_eval": evaluation_df[
            args.vtec_target_column
        ].to_numpy(dtype=np.float64),
        "slant_context_stats": slant_context_stats,
        "vertical_context_stats": vertical_context_stats,
    }


def predict_set_ratio_evaluation(
    *,
    model: Any,
    device: Any,
    y_scaler: Any,
    args: argparse.Namespace,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Predict one validation or locked-test partition."""

    outputs = predict_set_ratio_conversion(
        model=model,
        device=device,
        y_scaler=y_scaler,
        hard_identity_ratio=args.hard_identity_ratio,
        target_mode=args.target_mode,
        target_transform=args.target_transform,
        batch_size=args.batch_size,
        slant_QG_scaled=bundle["slant_QG_scaled"],
        slant_OF_scaled=bundle["slant_OF_scaled"],
        slant_OM=bundle["slant_OM"],
        slant_OD=bundle["slant_OD"],
        slant_identity_delta=bundle["slant_identity_delta"],
        vert_QG_scaled=bundle["vert_QG_scaled"],
        vert_OF_scaled=bundle["vert_OF_scaled"],
        vert_OM=bundle["vert_OM"],
        vert_OD=bundle["vert_OD"],
        vertical_identity_delta=bundle["vertical_identity_delta"],
        slm_slant_ratio=bundle["slm_slant_ratio"],
        slm_vertical_ratio=bundle["slm_vertical_ratio"],
        shrinkage_strength=bundle["shrinkage_strength"],
        source_stec_eval=bundle["source_stec_eval"],
    )
    predictions = bundle["df"].copy()
    predictions["set_ratio_slant_identity_delta"] = bundle[
        "slant_identity_delta"
    ]
    predictions["set_ratio_vertical_identity_delta"] = bundle[
        "vertical_identity_delta"
    ]
    predictions["set_ratio_slant_slm_ratio"] = bundle["slm_slant_ratio"]
    predictions["set_ratio_vertical_slm_ratio"] = bundle["slm_vertical_ratio"]
    predictions["set_ratio_slant_model_output"] = outputs["slant_model_output"]
    predictions["set_ratio_vertical_model_output"] = outputs[
        "vertical_model_output"
    ]
    predictions["set_ratio_elevation_shrinkage_strength"] = bundle[
        "shrinkage_strength"
    ]
    for name in (
        "slant_correction_raw",
        "vertical_correction_raw",
        "slant_correction",
        "vertical_correction",
        "slant_ratio",
        "vertical_ratio",
    ):
        predictions[f"set_ratio_{name}"] = outputs[name]
    predictions["stec_set_ratio_mlp"] = outputs["stec_set_ratio_mlp"]
    predictions["vtec_set_ratio_mlp"] = outputs["vtec_set_ratio_mlp"]
    predictions["vtec_set_ratio_mlp_direct"] = outputs["vtec_set_ratio_mlp"]
    predictions["vtec_set_ratio_mlp_error_iri"] = (
        predictions["vtec_set_ratio_mlp"]
        - predictions[args.vtec_target_column]
    )
    predictions["vtec_set_ratio_mlp_abs_error_iri"] = predictions[
        "vtec_set_ratio_mlp_error_iri"
    ].abs()
    return predictions, outputs


def clone_model_state(model: Any) -> dict[str, torch.Tensor]:
    """Copy model parameters to CPU for deterministic best-state restore."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
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

    train_df, validation_df, test_df, split_metadata = choose_set_ratio_split(
        df, args
    )
    train_df, train_window_metadata = apply_train_window_filter(train_df, args)
    split_metadata["train_window"] = train_window_metadata
    train_df, validation_df, test_df, elevation_filter_metadata = (
        apply_elevation_filters(train_df, validation_df, test_df, args)
    )
    split_metadata["elevation_filter"] = elevation_filter_metadata
    if args.target_transform == "log" and args.target_mode != "slm_correction":
        raise ValueError("--target-transform log requires --target-mode slm_correction.")
    if args.dynamic_pair_resampling:
        if not args.same_epoch_pairs:
            raise ValueError("--dynamic-pair-resampling currently requires --same-epoch-pairs.")
        if args.hard_identity_ratio:
            raise ValueError("--dynamic-pair-resampling does not support --hard-identity-ratio.")
        if args.dynamic_pair_examples is None or args.dynamic_pair_examples <= 0:
            raise ValueError("--dynamic-pair-examples must be positive in dynamic mode.")
        dynamic_pair_pool = build_dynamic_pair_pool_same_epoch(
            train_df,
            source_stec_col=args.source_stec_column,
            target_stec_col=args.slant_target_column,
            obs_cols=obs_cols,
            pair_radius_km=args.nearby_km,
            max_target_neighbors=args.nearby_max_neighbors,
            include_identity=not args.no_identity,
            min_target_elevation_deg=args.min_target_elevation_deg,
            target_elevation_keep_identity=args.target_elevation_keep_identity,
            mapping_col=args.mapping_column,
            target_mode=args.target_mode,
            target_transform=args.target_transform,
        )
        scaler_pair_count = int(args.dynamic_scaler_pair_examples)
        scaler_indices, scaler_sample_stats = sample_dynamic_pair_indices(
            dynamic_pair_pool,
            n_examples=scaler_pair_count,
            seed=args.seed + 100_000,
        )
        QG, OF, OM, OD, Y = materialize_pair_pool_examples(
            dynamic_pair_pool,
            scaler_indices,
            max_obs=args.max_obs,
            context_radius_km=args.context_km,
        )
        pair_stats = dict(dynamic_pair_pool["stats"])
        pair_stats["dynamic_pair_examples"] = int(args.dynamic_pair_examples)
        pair_stats["dynamic_scaler_pair_examples"] = scaler_pair_count
        pair_stats["dynamic_scaler_sample"] = scaler_sample_stats
        pair_stats["max_pair_examples"] = None
    else:
        dynamic_pair_pool = None
        pair_builder = (
            build_pair_set_ratio_examples_same_epoch
            if args.same_epoch_pairs else build_pair_set_ratio_examples
        )
        QG, OF, OM, OD, Y, pair_stats = pair_builder(
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
            min_target_elevation_deg=args.min_target_elevation_deg,
            target_elevation_keep_identity=args.target_elevation_keep_identity,
            mapping_col=args.mapping_column,
            target_mode=args.target_mode,
            target_transform=args.target_transform,
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
    OF_scaled = scale_obs_tensor(OF, OM, o_imp, o_scaler)
    identity_delta = compute_identity_delta(
        QG,
        OF,
        OM,
        ipp_scale_km=args.identity_delta_ipp_scale_km,
    )

    has_validation = not validation_df.empty
    selection_df = validation_df if has_validation else test_df
    selection_role = "validation" if has_validation else "legacy_test"
    selection_context_df = build_evaluation_context(
        train_df,
        selection_df,
        inference_context=args.inference_context,
    )
    selection_bundle = prepare_set_ratio_evaluation(
        selection_df,
        selection_context_df,
        args=args,
        obs_cols=obs_cols,
        q_imp=q_imp,
        q_scaler=q_scaler,
        o_imp=o_imp,
        o_scaler=o_scaler,
    )
    slant_context_stats = selection_bundle["slant_context_stats"]
    vert_context_stats = selection_bundle["vertical_context_stats"]
    slant_identity_delta = selection_bundle["slant_identity_delta"]
    vertical_identity_delta = selection_bundle["vertical_identity_delta"]
    shrinkage_strength = selection_bundle["shrinkage_strength"]
    target_vtec_eval = selection_bundle["target_vtec_eval"]
    shrinkage_enabled = bool(
        args.elevation_shrinkage and args.target_mode == "slm_correction"
    )
    epoch_metrics_enabled = bool(args.epoch_metrics or args.epoch_metrics_epochs)
    epoch_metrics_epochs = parse_epoch_metrics_epochs(
        args.epoch_metrics_epochs, args.epochs
    )
    epoch_metrics_rows: list[dict[str, float | int]] = []
    best_validation_state: dict[str, torch.Tensor] | None = None
    best_validation_metric: dict[str, float | int] | None = None

    device = choose_device(args.device)
    if args.dynamic_pair_resampling:
        Y_scaled = y_scaler.fit_transform(Y.reshape(-1, 1)).ravel().astype(np.float64)
        loader = None
    elif args.hard_identity_ratio:
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
    correction_regularization_center = 0.0 if args.target_transform == "log" else 1.0
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
        epoch_sample_stats = None
        if args.dynamic_pair_resampling:
            assert dynamic_pair_pool is not None
            epoch_indices, epoch_sample_stats = sample_dynamic_pair_indices(
                dynamic_pair_pool,
                n_examples=int(args.dynamic_pair_examples),
                seed=args.seed + epoch,
            )
            QG_epoch, OF_epoch, OM_epoch, OD_epoch, Y_epoch = materialize_pair_pool_examples(
                dynamic_pair_pool,
                epoch_indices,
                max_obs=args.max_obs,
                context_radius_km=args.context_km,
            )
            QG_epoch_scaled = q_scaler.transform(
                q_imp.transform(QG_epoch)
            ).astype(np.float64)
            OF_epoch_scaled = scale_obs_tensor(OF_epoch, OM_epoch, o_imp, o_scaler)
            Y_epoch_scaled = y_scaler.transform(
                Y_epoch.reshape(-1, 1)
            ).ravel().astype(np.float64)
            loader = _make_tensor_loader(
                QG_epoch_scaled,
                OF_epoch_scaled,
                OM_epoch,
                OD_epoch,
                Y_epoch_scaled,
                args.batch_size,
                args.seed + epoch,
                shuffle=True,
            )
        assert loader is not None
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
                    (pred_label - correction_regularization_center) ** 2
                )
            loss.backward()
            optimizer.step()
            bs = len(qg)
            total_loss += float(loss.detach().cpu()) * bs
            total_count += bs
        train_loss = total_loss / max(total_count, 1)
        history_row = {"epoch": epoch, "train_loss": train_loss}
        if epoch_sample_stats is not None:
            history_row["dynamic_sample"] = epoch_sample_stats
        history.append(history_row)
        if should_record_epoch_metrics(
            epoch,
            enabled=epoch_metrics_enabled or has_validation,
            every=args.epoch_metrics_every,
            selected_epochs=epoch_metrics_epochs,
            final_epoch=args.epochs,
        ):
            _, epoch_outputs = predict_set_ratio_evaluation(
                model=model,
                device=device,
                y_scaler=y_scaler,
                args=args,
                bundle=selection_bundle,
            )
            metric_row = build_epoch_metric_row(
                epoch=epoch,
                train_loss=train_loss,
                target_vtec=target_vtec_eval,
                predicted_vtec=epoch_outputs["vtec_set_ratio_mlp"],
            )
            if epoch_sample_stats is not None:
                metric_row["dynamic_sample_pairs"] = int(epoch_sample_stats["actual_pairs"])
                metric_row["dynamic_per_epoch_cap"] = int(epoch_sample_stats["per_epoch_cap"])
            epoch_metrics_rows.append(metric_row)
            if has_validation and (
                best_validation_metric is None
                or float(metric_row["rmse_tecu"])
                < float(best_validation_metric["rmse_tecu"])
            ):
                best_validation_metric = dict(metric_row)
                best_validation_state = clone_model_state(model)
            print(
                f"{selection_role}_metrics "
                f"epoch={metric_row['epoch']} "
                f"train_loss={metric_row['train_loss']:.6g} "
                f"rmse={metric_row['rmse_tecu']:.4f} "
                f"mae={metric_row['mae_tecu']:.4f} "
                f"bias={metric_row['bias_tecu']:+.4f} "
                f"r2={metric_row['r2']:.4f}"
            )

    if has_validation:
        if best_validation_state is None or best_validation_metric is None:
            raise RuntimeError(
                "Validation was configured but no checkpoint metric was recorded."
            )
        model.load_state_dict(best_validation_state)
        selected_epoch = int(best_validation_metric["epoch"])
        selected_model_source = "best_validation_rmse"
    else:
        selected_epoch = int(args.epochs)
        selected_model_source = "final_epoch_legacy_protocol"

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

    selection_predictions, selection_outputs = predict_set_ratio_evaluation(
        model=model,
        device=device,
        y_scaler=y_scaler,
        args=args,
        bundle=selection_bundle,
    )
    slant_correction_raw = selection_outputs["slant_correction_raw"]
    vertical_correction_raw = selection_outputs["vertical_correction_raw"]
    slant_correction = selection_outputs["slant_correction"]
    vertical_correction = selection_outputs["vertical_correction"]
    slant_ratio = selection_outputs["slant_ratio"]

    validation_predictions = selection_predictions if has_validation else None
    test_predictions = None if has_validation else selection_predictions
    test_bundle = None if has_validation else selection_bundle
    validation_pred_path = None
    test_pred_path = None
    if has_validation:
        validation_pred_path = save_table(
            selection_predictions,
            table_dir
            / f"{run_name}_validation_predictions.{args.output_format}",
        )
        if not test_df.empty:
            locked_test_context = build_evaluation_context(
                train_df,
                test_df,
                inference_context=args.inference_context,
            )
            test_bundle = prepare_set_ratio_evaluation(
                test_df,
                locked_test_context,
                args=args,
                obs_cols=obs_cols,
                q_imp=q_imp,
                q_scaler=q_scaler,
                o_imp=o_imp,
                o_scaler=o_scaler,
            )
            test_predictions, _ = predict_set_ratio_evaluation(
                model=model,
                device=device,
                y_scaler=y_scaler,
                args=args,
                bundle=test_bundle,
            )
            test_pred_path = save_table(
                test_predictions,
                table_dir / f"{run_name}_test_predictions.{args.output_format}",
            )
    else:
        test_pred_path = save_table(
            selection_predictions,
            table_dir / f"{run_name}_predictions.{args.output_format}",
        )

    epoch_metrics_path = None
    if epoch_metrics_rows:
        metrics_role = "validation_" if has_validation else ""
        epoch_metrics_path = (
            table_dir / f"{run_name}_{metrics_role}epoch_metrics.csv"
        )
        epoch_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(epoch_metrics_rows).to_csv(epoch_metrics_path, index=False)

    validation_vtec_meta = None
    test_vtec_meta = None
    if not args.skip_vtec_evaluation:
        evaluation_kwargs = {
            "table_dir": table_dir,
            "target_column": args.vtec_target_column,
            "baseline_column": args.baseline_vtec_column,
            "vtec_prediction_columns": [
                "vtec_set_ratio_mlp",
                "vtec_set_ratio_mlp_direct",
            ],
            "stec_prediction_columns": [],
            "mapping_column": args.mapping_column,
            "elevation_bins": [
                float(item.strip())
                for item in args.elevation_bins.split(",")
                if item.strip()
            ],
            "station_min_count": args.station_min_count,
            "max_rows": None,
        }
        if validation_pred_path is not None:
            validation_vtec_meta = build_vtec_metrics(
                input_path=validation_pred_path,
                run_name=f"{run_name}_validation",
                **evaluation_kwargs,
            )
        if test_pred_path is not None:
            test_vtec_meta = build_vtec_metrics(
                input_path=test_pred_path,
                run_name=f"{run_name}_test",
                **evaluation_kwargs,
            )

    train_pair_metric_name = (
        "train_pair_ratio"
        if args.target_mode == "source_ratio"
        else (
            "train_pair_log_slm_correction"
            if args.target_transform == "log"
            else "train_pair_slm_correction"
        )
    )

    metric_summary: dict[str, Any] = {
        "train_pair_training_target": regression_metrics(Y, train_pred_label),
        train_pair_metric_name: regression_metrics(Y, train_pred_label),
    }
    if validation_predictions is not None:
        metric_summary.update(
            {
                "validation_slant_stec": regression_metrics(
                    validation_predictions[args.slant_target_column],
                    validation_predictions["stec_set_ratio_mlp"],
                ),
                "validation_vtec": vtec_error_metrics(
                    validation_predictions[args.vtec_target_column],
                    validation_predictions["vtec_set_ratio_mlp"],
                ),
                "validation_slm_vtec": vtec_error_metrics(
                    validation_predictions[args.vtec_target_column],
                    validation_predictions[args.baseline_vtec_column],
                ),
            }
        )
    if test_predictions is not None:
        metric_summary.update(
            {
                "test_slant_stec": regression_metrics(
                    test_predictions[args.slant_target_column],
                    test_predictions["stec_set_ratio_mlp"],
                ),
                "test_vtec": vtec_error_metrics(
                    test_predictions[args.vtec_target_column],
                    test_predictions["vtec_set_ratio_mlp"],
                ),
                "test_slm_vtec": vtec_error_metrics(
                    test_predictions[args.vtec_target_column],
                    test_predictions[args.baseline_vtec_column],
                ),
            }
        )

    output_files: dict[str, Any] = {}
    if validation_pred_path is not None:
        output_files["validation_predictions"] = str(validation_pred_path)
    if test_pred_path is not None:
        output_files["test_predictions"] = str(test_pred_path)
        if not has_validation:
            output_files["predictions"] = str(test_pred_path)
    if validation_vtec_meta is not None:
        output_files["validation_vtec_metrics"] = validation_vtec_meta["outputs"]
    if test_vtec_meta is not None:
        output_files["test_vtec_metrics"] = test_vtec_meta["outputs"]

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": f"simulation_set_context_{args.target_mode}"
        + ("_hard_identity" if args.hard_identity_ratio else ""),
        "model": "set_ratio_mlp",
        "target_mode": args.target_mode,
        "target_transform": args.target_transform,
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
        "model_selection": {
            "enabled": bool(has_validation),
            "selection_role": selection_role,
            "criterion": "validation_vtec_rmse",
            "selected_epoch": selected_epoch,
            "selected_model_source": selected_model_source,
            "best_validation_metric": best_validation_metric,
            "locked_test_used_during_selection": False,
            "evaluation_every": int(args.epoch_metrics_every),
            "evaluation_epochs": sorted(epoch_metrics_epochs),
        },
        "vertical_top_km": float(args.vertical_top_km),
        "query_geom_columns": QUERY_FEATURE_COLS,
        "obs_feature_columns": model_obs_cols,
        "pair_stats": {
            key: value for key, value in pair_stats.items()
            if not key.startswith("_")
        },
        "selection_context_role": selection_role,
        "selection_slant_context_stats": slant_context_stats,
        "selection_vertical_context_stats": vert_context_stats,
        "test_slant_context_stats": (
            None if test_bundle is None
            else test_bundle["slant_context_stats"]
        ),
        "test_vertical_context_stats": (
            None if test_bundle is None
            else test_bundle["vertical_context_stats"]
        ),
        "split": split_metadata,
        "row_counts": {
            "input": int(len(df)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "n_train_epochs": int(pd.to_datetime(train_df["timestamp"]).nunique())
            if "timestamp" in train_df.columns else None,
            "train_pair_examples": int(len(Y)),
            "train_pair_pool_examples": int(pair_stats.get("pair_pool_examples", len(Y))),
        },
        "hyperparameters": {
            "target_mode": args.target_mode,
            "target_transform": args.target_transform,
            "correction_regularization": float(args.correction_regularization),
            "elevation_shrinkage": bool(args.elevation_shrinkage),
            "shrinkage_start_deg": float(args.shrinkage_start_deg),
            "shrinkage_end_deg": float(args.shrinkage_end_deg),
            "shrinkage_power": float(args.shrinkage_power),
            "shrinkage_min_strength": float(args.shrinkage_min_strength),
            "epochs": int(args.epochs),
            "epoch_metrics": bool(epoch_metrics_enabled or has_validation),
            "epoch_metrics_every": int(args.epoch_metrics_every),
            "epoch_metrics_epochs": sorted(epoch_metrics_epochs),
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
            "dynamic_pair_resampling": bool(args.dynamic_pair_resampling),
            "dynamic_pair_examples": None
            if args.dynamic_pair_examples is None else int(args.dynamic_pair_examples),
            "dynamic_scaler_pair_examples": int(args.dynamic_scaler_pair_examples),
            "spatial_sigma_km": float(args.spatial_sigma_km),
            "include_identity": bool(not args.no_identity),
            "min_target_elevation_deg": None
            if args.min_target_elevation_deg is None
            else float(args.min_target_elevation_deg),
            "target_elevation_keep_identity": bool(args.target_elevation_keep_identity),
            "hard_identity_ratio": bool(args.hard_identity_ratio),
            "identity_delta_ipp_scale_km": float(args.identity_delta_ipp_scale_km),
            "inference_context": args.inference_context,
            "same_epoch_pairs": bool(args.same_epoch_pairs),
            "validation_keys_from": args.validation_keys_from,
            "test_keys_from": args.test_keys_from,
            "train_window_start": args.train_window_start,
            "train_window_end": args.train_window_end,
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
            f"{selection_role}_slant": _finite_stats(slant_identity_delta),
            f"{selection_role}_vertical": _finite_stats(vertical_identity_delta),
            f"{selection_role}_slant_identity_ratio": _ratio_identity_stats(
                slant_ratio, slant_identity_delta
            ),
        },
        "elevation_shrinkage_stats": {
            "enabled": shrinkage_enabled,
            "strength": _finite_stats(shrinkage_strength),
            "raw_vertical_correction": _finite_stats(vertical_correction_raw),
            "shrunk_vertical_correction": _finite_stats(vertical_correction),
        },
        "metrics": metric_summary,
        "outputs": output_files,
    }
    if epoch_metrics_path is not None:
        metadata["outputs"]["epoch_metrics"] = str(epoch_metrics_path)
        metadata["epoch_metrics"] = {
            "n_rows": int(len(epoch_metrics_rows)),
            "columns": list(pd.DataFrame(epoch_metrics_rows).columns),
        }
    ckpt_path = checkpoint_dir / f"{run_name}.pth"
    metrics_json_path = table_dir / f"{run_name}_metrics.json"
    metadata["outputs"]["checkpoint"] = str(ckpt_path)
    metadata["outputs"]["metrics_json"] = str(metrics_json_path)
    metadata["checkpoint_epoch"] = selected_epoch
    metadata["checkpoint_selection_source"] = selected_model_source
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selected_epoch": selected_epoch,
            "selection_metric": best_validation_metric,
            "q_imputer": q_imp,
            "q_scaler": q_scaler,
            "o_imputer": o_imp,
            "o_scaler": o_scaler,
            "y_scaler": y_scaler,
            "metadata": metadata,
        },
        ckpt_path,
    )
    write_json(metadata, metrics_json_path)
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
        "--target-transform",
        choices=["linear", "log"],
        default="linear",
        help="Transform the training label; log is only valid for slm_correction.",
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
    p.add_argument(
        "--min-target-elevation-deg",
        type=float,
        help="Restrict training pair target rays to this minimum elevation.",
    )
    p.add_argument(
        "--target-elevation-keep-identity",
        action="store_true",
        help=(
            "With --min-target-elevation-deg, keep source self-pairs i->i even "
            "when the source elevation is below the target threshold. Ignored "
            "when --no-identity is set."
        ),
    )
    p.add_argument("--split", choices=["random", "time", "station", "prn", "all"], default="random")
    p.add_argument(
        "--validation-keys-from",
        help=(
            "Rows matching these keys form the validation set used for epoch "
            "selection and best-checkpoint saving."
        ),
    )
    p.add_argument(
        "--validation-key-columns",
        default="timestamp,station,prn",
    )
    p.add_argument(
        "--test-keys-from",
        help=(
            "Rows matching these keys form the locked test set. With "
            "--validation-keys-from, test truth is evaluated only after model "
            "selection is complete."
        ),
    )
    p.add_argument("--test-key-columns", default="timestamp,station,prn")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--train-end")
    p.add_argument("--train-window-start")
    p.add_argument("--train-window-end")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--epoch-metrics",
        action="store_true",
        help=(
            "After selected epochs, evaluate VTEC conversion metrics on the "
            "validation rows and save an epoch_metrics CSV. Legacy two-way "
            "commands still evaluate their historical test/evaluation rows."
        ),
    )
    p.add_argument(
        "--epoch-metrics-every",
        type=int,
        default=1,
        help=(
            "Evaluate validation metrics every N epochs. Validation also "
            "enables best-checkpoint selection automatically."
        ),
    )
    p.add_argument(
        "--epoch-metrics-epochs",
        default="",
        help=(
            "Comma-separated epochs to evaluate, e.g. 30,40,50,60. "
            "Providing this also enables epoch metrics."
        ),
    )
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
    p.add_argument("--dynamic-pair-resampling", action="store_true")
    p.add_argument("--dynamic-pair-examples", type=int)
    p.add_argument("--dynamic-scaler-pair-examples", type=int, default=200000)
    p.add_argument("--same-epoch-pairs", action="store_true", help="Build source-target training pairs and inference contexts only within each timestamp.")
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
