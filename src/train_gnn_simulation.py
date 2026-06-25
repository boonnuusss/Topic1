"""Train a notebook-style GNN on IRI simulation rays.

This entry point adapts ``00_iri/mycode/ML_sTEC/GNN_sTEC_improve.ipynb`` to the
Topic1 simulation setting.  The model is trained only on simulated slant-ray
STEC labels.  VTEC is evaluated after training by querying the same STEC model
with a vertical special ray at the IPP location.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .evaluate import build_vtec_metrics
    from .geometry import geodetic_to_ecef
    from .metrics import regression_metrics, vtec_error_metrics
    from .models.gnn import build_graph_sage_edge_regressor
    from .train_mlp import choose_device, set_random_seed
    from .train_ray_mlp_simulation import (
        add_mapping_aliases,
        choose_simulation_split,
        require_columns,
    )
    from .train_tree import (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_TABLE_DIR,
        read_table,
        save_table,
        write_json,
    )
except ImportError:
    from evaluate import build_vtec_metrics
    from geometry import geodetic_to_ecef
    from metrics import regression_metrics, vtec_error_metrics
    from models.gnn import build_graph_sage_edge_regressor
    from train_mlp import choose_device, set_random_seed
    from train_ray_mlp_simulation import (
        add_mapping_aliases,
        choose_simulation_split,
        require_columns,
    )
    from train_tree import (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_TABLE_DIR,
        read_table,
        save_table,
        write_json,
    )


DEFAULT_INPUT = "datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.parquet"
NODE_COLUMNS = ["rec_x", "rec_y", "rec_z", "sat_x", "sat_y", "sat_z"]
EDGE_COLUMNS = ["elevation", "azimuth"]
KEY_COLUMNS = ["timestamp", "station", "prn"]


@dataclass
class GraphBundle:
    """Container for one receiver-satellite graph."""

    node_xyz_raw: np.ndarray
    rec_xyz_unique: np.ndarray
    sat_xyz_unique: np.ndarray
    x_scaled: np.ndarray
    ray_edge_index: np.ndarray
    ray_edge_attr_raw: np.ndarray
    edge_index: np.ndarray
    edge_attr_scaled: np.ndarray
    y_scaled_full: np.ndarray
    train_edge_indices: np.ndarray
    train_forward_edge_indices: np.ndarray
    test_forward_edge_indices: np.ndarray
    node_scaler: Any
    edge_scaler: Any
    y_scaler: Any
    metadata: dict[str, Any]


def finite_numeric_rows(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return a row mask where all requested numeric columns are finite."""

    numeric = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(numeric.to_numpy(float)).all(axis=1), index=df.index)


def ray_edge_features(df: pd.DataFrame) -> np.ndarray:
    """Build edge features matching the reference notebook convention."""

    elevation_rad = np.deg2rad(df["elevation"].to_numpy(float))
    # The reference notebook computes azimuth with atan2, therefore its range
    # is [-pi, pi].  The ray-wise parquet stores azimuth as [0, 360) degrees;
    # wrap it before scaling to avoid a false discontinuity at north.
    azimuth_deg = ((df["azimuth"].to_numpy(float) + 180.0) % 360.0) - 180.0
    azimuth_rad = np.deg2rad(azimuth_deg)
    return np.column_stack((elevation_rad, azimuth_rad))


def edge_index_from_pairs(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return a PyTorch-style edge index array."""

    return np.vstack((src.astype(np.int64), dst.astype(np.int64)))


def build_receiver_knn_edges(
    receiver_xyz: np.ndarray,
    receiver_node_ids: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Build directed receiver-receiver KNN edges."""

    if k <= 0 or len(receiver_node_ids) <= 1:
        return np.empty((2, 0), dtype=np.int64)

    from sklearn.neighbors import NearestNeighbors

    n_neighbors = min(int(k) + 1, len(receiver_node_ids))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(receiver_xyz)
    _, indices = nn.kneighbors(receiver_xyz)

    src_nodes: list[int] = []
    dst_nodes: list[int] = []
    for src_pos, neighbors in enumerate(indices):
        src_node = int(receiver_node_ids[src_pos])
        for dst_pos in neighbors:
            if int(dst_pos) == src_pos:
                continue
            src_nodes.append(src_node)
            dst_nodes.append(int(receiver_node_ids[int(dst_pos)]))

    if not src_nodes:
        return np.empty((2, 0), dtype=np.int64)
    return edge_index_from_pairs(np.asarray(src_nodes), np.asarray(dst_nodes))


def build_graph(
    df: pd.DataFrame,
    *,
    train_row_mask: np.ndarray,
    test_row_mask: np.ndarray,
    slant_target_column: str,
    knn_k: int,
) -> GraphBundle:
    """Build the transductive training graph."""

    from sklearn.preprocessing import StandardScaler

    n_rows = len(df)
    rec_xyz = df[["rec_x", "rec_y", "rec_z"]].to_numpy(dtype=np.float64)
    sat_xyz = df[["sat_x", "sat_y", "sat_z"]].to_numpy(dtype=np.float64)

    rec_unique, rec_inverse = np.unique(rec_xyz, axis=0, return_inverse=True)
    sat_unique, sat_inverse = np.unique(sat_xyz, axis=0, return_inverse=True)
    n_rec = len(rec_unique)
    n_sat = len(sat_unique)
    sat_nodes = n_rec + sat_inverse

    forward_edges = edge_index_from_pairs(rec_inverse, sat_nodes)
    reverse_edges = edge_index_from_pairs(sat_nodes, rec_inverse)
    ray_edge_index = np.hstack((forward_edges, reverse_edges))

    edge_attr_one_way = ray_edge_features(df)
    ray_edge_attr_raw = np.vstack((edge_attr_one_way, edge_attr_one_way))
    y_ray_raw = np.concatenate(
        (
            df[slant_target_column].to_numpy(dtype=np.float64),
            df[slant_target_column].to_numpy(dtype=np.float64),
        )
    )

    finite_y = np.isfinite(y_ray_raw)
    train_edge_mask = np.concatenate((train_row_mask, train_row_mask)) & finite_y
    train_edge_indices = np.flatnonzero(train_edge_mask)
    if len(train_edge_indices) < 2:
        raise ValueError("GNN training requires at least two finite train edges.")

    train_forward_edge_indices = np.flatnonzero(
        train_row_mask & np.isfinite(df[slant_target_column].to_numpy(float))
    )
    test_forward_edge_indices = np.flatnonzero(
        test_row_mask & np.isfinite(df[slant_target_column].to_numpy(float))
    )
    if len(test_forward_edge_indices) == 0:
        raise ValueError("The selected split contains no finite test edges.")

    knn_edges = build_receiver_knn_edges(
        rec_unique,
        np.arange(n_rec, dtype=np.int64),
        k=knn_k,
    )
    knn_edge_attr = np.zeros((knn_edges.shape[1], ray_edge_attr_raw.shape[1]))

    edge_index = np.hstack((ray_edge_index, knn_edges))
    edge_attr_raw = np.vstack((ray_edge_attr_raw, knn_edge_attr))

    node_xyz_raw = np.vstack((rec_unique, sat_unique))
    node_scaler = StandardScaler()
    edge_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaled = node_scaler.fit_transform(node_xyz_raw)
    edge_scaler.fit(ray_edge_attr_raw[train_edge_indices])
    edge_attr_scaled = edge_scaler.transform(edge_attr_raw)
    y_scaled_ray = np.full(len(ray_edge_attr_raw), np.nan, dtype=np.float64)
    y_scaler.fit(y_ray_raw[train_edge_indices].reshape(-1, 1))
    y_scaled_ray[train_edge_indices] = y_scaler.transform(
        y_ray_raw[train_edge_indices].reshape(-1, 1)
    ).ravel()
    y_scaled_full = np.full(edge_index.shape[1], np.nan, dtype=np.float64)
    y_scaled_full[: len(y_scaled_ray)] = y_scaled_ray

    metadata = {
        "n_rows": int(n_rows),
        "n_receiver_nodes": int(n_rec),
        "n_satellite_nodes": int(n_sat),
        "n_nodes": int(len(node_xyz_raw)),
        "n_ray_edges": int(ray_edge_index.shape[1]),
        "n_receiver_knn_edges": int(knn_edges.shape[1]),
        "n_message_edges": int(edge_index.shape[1]),
        "edge_features": ["elevation_rad", "azimuth_rad"],
        "knn_k": int(knn_k),
        "transductive_geometry": True,
    }
    return GraphBundle(
        node_xyz_raw=node_xyz_raw,
        rec_xyz_unique=rec_unique,
        sat_xyz_unique=sat_unique,
        x_scaled=x_scaled,
        ray_edge_index=ray_edge_index,
        ray_edge_attr_raw=ray_edge_attr_raw,
        edge_index=edge_index,
        edge_attr_scaled=edge_attr_scaled,
        y_scaled_full=y_scaled_full,
        train_edge_indices=train_edge_indices,
        train_forward_edge_indices=train_forward_edge_indices,
        test_forward_edge_indices=test_forward_edge_indices,
        node_scaler=node_scaler,
        edge_scaler=edge_scaler,
        y_scaler=y_scaler,
        metadata=metadata,
    )


def make_tensors(graph: GraphBundle, device):
    """Convert graph arrays to torch tensors."""

    import torch

    return {
        "x": torch.tensor(graph.x_scaled, dtype=torch.float32, device=device),
        "edge_index": torch.tensor(graph.edge_index, dtype=torch.long, device=device),
        "edge_attr": torch.tensor(
            graph.edge_attr_scaled, dtype=torch.float32, device=device
        ),
        "y": torch.tensor(graph.y_scaled_full, dtype=torch.float32, device=device),
        "train_edges": torch.tensor(
            graph.train_edge_indices, dtype=torch.long, device=device
        ),
    }


def train_gnn_loop(
    model,
    graph: GraphBundle,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device,
) -> list[dict[str, float | int]]:
    """Train the GNN with full-batch graph optimization."""

    import torch
    import torch.nn as nn

    model.to(device)
    tensors = make_tensors(graph, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(
            tensors["x"],
            tensors["edge_index"],
            tensors["edge_attr"],
        ).reshape(-1)
        loss = criterion(
            pred[tensors["train_edges"]],
            tensors["y"][tensors["train_edges"]],
        )
        loss.backward()
        optimizer.step()
        history.append({"epoch": int(epoch), "train_loss": float(loss.detach().cpu())})
    return history


def predict_graph_edges(model, graph: GraphBundle, device) -> np.ndarray:
    """Predict unscaled STEC for all base graph edges."""

    import torch

    model.eval()
    tensors = make_tensors(graph, device)
    with torch.no_grad():
        pred_scaled = (
            model(
                tensors["x"],
                tensors["edge_index"],
                tensors["edge_attr"],
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    return graph.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


def build_vertical_query_graph(
    graph: GraphBundle,
    test_df: pd.DataFrame,
    *,
    vertical_top_km: float,
    knn_k: int,
    connect_query_knn: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an augmented graph and vertical query edge list."""

    lon = test_df["ipp_lon"].to_numpy(float)
    lat = test_df["ipp_lat"].to_numpy(float)
    rec_x, rec_y, rec_z = geodetic_to_ecef(lon, lat, np.zeros(len(test_df)))
    sat_x, sat_y, sat_z = geodetic_to_ecef(
        lon,
        lat,
        np.full(len(test_df), vertical_top_km * 1000.0),
    )
    vertical_rec = np.column_stack((rec_x, rec_y, rec_z))
    vertical_sat = np.column_stack((sat_x, sat_y, sat_z))

    n_base_nodes = len(graph.node_xyz_raw)
    query_rec_nodes = n_base_nodes + np.arange(len(test_df), dtype=np.int64)
    query_sat_nodes = (
        n_base_nodes + len(test_df) + np.arange(len(test_df), dtype=np.int64)
    )
    query_forward_edges = edge_index_from_pairs(query_rec_nodes, query_sat_nodes)
    query_reverse_edges = edge_index_from_pairs(query_sat_nodes, query_rec_nodes)
    query_ray_edges = np.hstack((query_forward_edges, query_reverse_edges))
    query_attr_one_way = np.column_stack(
        (
            np.full(len(test_df), np.pi / 2.0),
            np.zeros(len(test_df), dtype=np.float64),
        )
    )
    query_ray_attr = np.vstack((query_attr_one_way, query_attr_one_way))

    node_xyz_raw = np.vstack((graph.node_xyz_raw, vertical_rec, vertical_sat))
    if connect_query_knn:
        receiver_xyz = np.vstack((graph.rec_xyz_unique, vertical_rec))
        receiver_ids = np.concatenate(
            (
                np.arange(len(graph.rec_xyz_unique), dtype=np.int64),
                query_rec_nodes,
            )
        )
    else:
        receiver_xyz = graph.rec_xyz_unique
        receiver_ids = np.arange(len(graph.rec_xyz_unique), dtype=np.int64)
    knn_edges = build_receiver_knn_edges(receiver_xyz, receiver_ids, k=knn_k)
    knn_edge_attr = np.zeros((knn_edges.shape[1], graph.ray_edge_attr_raw.shape[1]))

    message_edge_index = np.hstack((graph.ray_edge_index, query_ray_edges, knn_edges))
    message_edge_attr_raw = np.vstack(
        (graph.ray_edge_attr_raw, query_ray_attr, knn_edge_attr)
    )
    x_scaled = graph.node_scaler.transform(node_xyz_raw)
    edge_attr_scaled = graph.edge_scaler.transform(message_edge_attr_raw)
    query_decode_attr_scaled = graph.edge_scaler.transform(query_attr_one_way)
    return (
        x_scaled,
        message_edge_index,
        edge_attr_scaled,
        query_forward_edges,
        query_decode_attr_scaled,
    )


def predict_vertical_queries(
    model,
    graph: GraphBundle,
    test_df: pd.DataFrame,
    *,
    vertical_top_km: float,
    knn_k: int,
    connect_query_knn: bool,
    device,
) -> np.ndarray:
    """Predict STEC for vertical special rays."""

    import torch

    (
        x_scaled,
        message_edge_index,
        message_edge_attr,
        query_forward_edges,
        query_decode_attr,
    ) = build_vertical_query_graph(
        graph,
        test_df,
        vertical_top_km=vertical_top_km,
        knn_k=knn_k,
        connect_query_knn=connect_query_knn,
    )
    model.eval()
    with torch.no_grad():
        pred_scaled = (
            model(
                torch.tensor(x_scaled, dtype=torch.float32, device=device),
                torch.tensor(message_edge_index, dtype=torch.long, device=device),
                torch.tensor(message_edge_attr, dtype=torch.float32, device=device),
                decode_edge_index=torch.tensor(
                    query_forward_edges, dtype=torch.long, device=device
                ),
                decode_edge_attr=torch.tensor(
                    query_decode_attr, dtype=torch.float32, device=device
                ),
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    return graph.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


def save_gnn_checkpoint(
    path: Path,
    *,
    model,
    graph: GraphBundle,
    metadata: dict[str, Any],
) -> Path:
    """Save model state and graph preprocessing objects."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_scaler": graph.node_scaler,
            "edge_scaler": graph.edge_scaler,
            "y_scaler": graph.y_scaler,
            "metadata": metadata,
        },
        path,
    )
    return path


def train_gnn_simulation(args: argparse.Namespace) -> dict[str, Any]:
    """Train GNN on slant-ray STEC and evaluate VTEC conversion."""

    set_random_seed(args.seed)
    input_path = Path(args.input)
    table_dir = Path(args.table_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = args.run_name or f"gnn_sim_{args.split}"

    df = read_table(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    df = add_mapping_aliases(df, mapping_column=args.mapping_column)
    required = (
        KEY_COLUMNS
        + NODE_COLUMNS
        + EDGE_COLUMNS
        + [
            "ipp_lon",
            "ipp_lat",
            args.source_stec_column,
            args.slant_target_column,
            args.vtec_target_column,
            args.baseline_vtec_column,
            args.mapping_column,
        ]
    )
    require_columns(df, required)

    numeric_required = [
        *NODE_COLUMNS,
        *EDGE_COLUMNS,
        "ipp_lon",
        "ipp_lat",
        args.source_stec_column,
        args.slant_target_column,
        args.vtec_target_column,
        args.baseline_vtec_column,
        args.mapping_column,
    ]
    finite_mask = finite_numeric_rows(df, numeric_required)
    n_dropped_nonfinite = int((~finite_mask).sum())
    df = df.loc[finite_mask].reset_index(drop=True)
    df["_gnn_row_id"] = np.arange(len(df), dtype=np.int64)

    train_df, test_df, split_metadata = choose_simulation_split(df, args)
    train_row_mask = np.zeros(len(df), dtype=bool)
    test_row_mask = np.zeros(len(df), dtype=bool)
    train_row_mask[train_df["_gnn_row_id"].to_numpy(np.int64)] = True
    test_row_mask[test_df["_gnn_row_id"].to_numpy(np.int64)] = True

    graph = build_graph(
        df,
        train_row_mask=train_row_mask,
        test_row_mask=test_row_mask,
        slant_target_column=args.slant_target_column,
        knn_k=args.knn_k,
    )
    device = choose_device(args.device)
    model = build_graph_sage_edge_regressor(
        node_dim=3,
        edge_dim=2,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )
    history = train_gnn_loop(
        model,
        graph,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    edge_pred = predict_graph_edges(model, graph, device)
    train_pred = edge_pred[graph.train_forward_edge_indices]
    test_pred = edge_pred[graph.test_forward_edge_indices]
    train_eval_df = df.iloc[graph.train_forward_edge_indices].copy()
    test_eval_df = df.iloc[graph.test_forward_edge_indices].copy()

    vertical_pred = predict_vertical_queries(
        model,
        graph,
        test_eval_df,
        vertical_top_km=args.vertical_top_km,
        knn_k=args.knn_k,
        connect_query_knn=not args.disable_query_knn,
        device=device,
    )

    # Physical constraint: STEC/VTEC ray integrals should be non-negative.
    train_pred = np.maximum(train_pred, 0.0)
    test_pred = np.maximum(test_pred, 0.0)
    vertical_pred = np.maximum(vertical_pred, 0.0)

    predictions = test_eval_df.drop(columns=["_gnn_row_id"], errors="ignore").copy()
    predictions["stec_gnn"] = test_pred
    predictions["vtec_gnn_direct"] = vertical_pred
    denominator = predictions["stec_gnn"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            predictions["vtec_gnn_direct"].to_numpy(float),
            denominator,
            out=np.full(len(predictions), np.nan, dtype=float),
            where=np.abs(denominator) > 1e-12,
        )
    predictions["gnn_conversion_ratio"] = ratio
    predictions["vtec_gnn"] = (
        predictions[args.source_stec_column].to_numpy(float) * ratio
    )
    predictions["vtec_gnn_error_iri"] = (
        predictions["vtec_gnn"] - predictions[args.vtec_target_column]
    )
    predictions["vtec_gnn_abs_error_iri"] = predictions["vtec_gnn_error_iri"].abs()
    predictions["vtec_gnn_direct_error_iri"] = (
        predictions["vtec_gnn_direct"] - predictions[args.vtec_target_column]
    )
    predictions["vtec_gnn_direct_abs_error_iri"] = predictions[
        "vtec_gnn_direct_error_iri"
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
            vtec_prediction_columns=["vtec_gnn", "vtec_gnn_direct"],
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

    train_stec_metrics = regression_metrics(
        train_eval_df[args.slant_target_column],
        train_pred,
    )
    test_stec_metrics = regression_metrics(
        predictions[args.slant_target_column],
        predictions["stec_gnn"],
    )
    test_vtec_metrics = vtec_error_metrics(
        predictions[args.vtec_target_column],
        predictions["vtec_gnn"],
    )
    direct_vtec_metrics = vtec_error_metrics(
        predictions[args.vtec_target_column],
        predictions["vtec_gnn_direct"],
    )
    slm_vtec_metrics = vtec_error_metrics(
        predictions[args.vtec_target_column],
        predictions[args.baseline_vtec_column],
    )

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "simulation_gnn_stec_to_vtec",
        "model": "graph_sage_edge_regressor",
        "run_name": run_name,
        "input_file": str(input_path),
        "legacy_reference": "../00_iri/mycode/ML_sTEC/GNN_sTEC_improve.ipynb",
        "source_stec_column": args.source_stec_column,
        "slant_target_column": args.slant_target_column,
        "vtec_target_column": args.vtec_target_column,
        "baseline_vtec_column": args.baseline_vtec_column,
        "stec_prediction_column": "stec_gnn",
        "vtec_prediction_column": "vtec_gnn",
        "vtec_direct_query_column": "vtec_gnn_direct",
        "conversion_ratio_column": "gnn_conversion_ratio",
        "conversion_interpretation": (
            "The GNN predicts STEC for graph edges. VTEC is estimated by "
            "querying the model for the original slant edge and a vertical "
            "special edge, then applying VTEC = source_STEC * vertical/slant."
        ),
        "vtec_reference_used_for_training": False,
        "training_label": args.slant_target_column,
        "vertical_top_km": float(args.vertical_top_km),
        "split": split_metadata,
        "row_counts": {
            "input": int(len(df) + n_dropped_nonfinite),
            "dropped_nonfinite": int(n_dropped_nonfinite),
            "usable": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "graph": graph.metadata
        | {"query_knn_enabled": bool(not args.disable_query_knn)},
        "hyperparameters": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "hidden_dim": int(args.hidden_dim),
            "embedding_dim": int(args.embedding_dim),
            "seed": int(args.seed),
            "device": str(device),
        },
        "history": history,
        "metrics": {
            "train_slant_stec": train_stec_metrics,
            "test_slant_stec": test_stec_metrics,
            "test_vtec": test_vtec_metrics,
            "test_vtec_direct_vertical_query": direct_vtec_metrics,
            "test_slm_vtec": slm_vtec_metrics,
        },
        "outputs": {"predictions": str(prediction_path)},
    }
    if vtec_metadata is not None:
        metadata["outputs"]["vtec_metrics"] = vtec_metadata["outputs"]

    checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    metadata["outputs"]["checkpoint"] = str(
        save_gnn_checkpoint(
            checkpoint_path,
            model=model,
            graph=graph,
            metadata=metadata,
        )
    )
    metrics_json_path = write_json(metadata, table_dir / f"{run_name}_metrics.json")
    summary_rows = [
        {
            "subset": "train",
            "metric_target": args.slant_target_column,
            **train_stec_metrics,
        },
        {
            "subset": "test",
            "metric_target": args.slant_target_column,
            **test_stec_metrics,
        },
        {
            "subset": "test",
            "metric_target": args.vtec_target_column,
            **test_vtec_metrics,
        },
        {
            "subset": "test",
            "metric_target": f"{args.vtec_target_column}_direct_vertical_query",
            **direct_vtec_metrics,
        },
        {
            "subset": "test",
            "metric_target": f"slm_{args.vtec_target_column}",
            **slm_vtec_metrics,
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
    parser.add_argument("--test-keys-from")
    parser.add_argument("--test-key-columns", default="timestamp,station,prn")
    parser.add_argument("--test-timestamp")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-end")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--knn-k", type=int, default=4)
    parser.add_argument("--disable-query-knn", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--elevation-bins", default="10,20,30,40,50,60,70,80,90")
    parser.add_argument("--station-min-count", type=int, default=1)
    parser.add_argument("--skip-vtec-evaluation", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = train_gnn_simulation(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
