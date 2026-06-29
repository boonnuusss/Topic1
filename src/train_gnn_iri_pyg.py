"""Train ImprovedSTEC_GNN (SAGEConv) on IRI simulation data and evaluate VTEC conversion.

Combines the notebook's PyG model architecture with the IRI simulation parquet
pipeline. Training uses simulated slant-ray STEC labels only; vtec_iri_ref is
reserved for post-training VTEC evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# ---------------------------------------------------------------------------
# Model (exact copy of ImprovedSTEC_GNN from notebook repro)
# ---------------------------------------------------------------------------


class ImprovedSTEC_GNN(nn.Module):
    def __init__(self, num_node_features: int, num_edge_features: int):
        super().__init__()
        self.lin_node = nn.Linear(num_node_features, 64)
        self.conv1 = SAGEConv(64, 64)
        self.conv2 = SAGEConv(64, 32)
        decoder_input_dim = 32 * 2 + num_edge_features
        self.edge_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.lin_node(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        edge_input = torch.cat([src, dst, edge_attr], dim=1)
        return self.edge_decoder(edge_input)


# ---------------------------------------------------------------------------
# Early stopping (matching notebook)
# ---------------------------------------------------------------------------


class RobustEarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 1e-5, window_size: int = 5):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.loss_history: list[float] = []
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int):
        self.loss_history.append(val_loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        if len(self.loss_history) < self.window_size:
            return
        current_avg_loss = float(np.mean(self.loss_history))
        if self.best_loss is None:
            self.best_loss = current_avg_loss
            self.best_epoch = epoch
        elif current_avg_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_avg_loss
            self.best_epoch = epoch
            self.counter = 0


# ---------------------------------------------------------------------------
# Graph construction from IRI simulation parquet
# ---------------------------------------------------------------------------


def wrap_azimuth_deg(az_deg: np.ndarray) -> np.ndarray:
    """Wrap azimuth from [0, 360) to [-180, 180) for atan2 convention."""
    out = np.array(az_deg, dtype=float, copy=True)
    out[out >= 180.0] -= 360.0
    return out


def build_graph_from_parquet(
    df: pd.DataFrame,
    *,
    knn_k: int = 4,
    seed: int = 42,
    split: str = "random",
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Build a PyG Data graph from IRI simulation parquet rows."""

    n_rays = len(df)

    # --- Extract arrays ---
    rec_xyz = df[["rec_x", "rec_y", "rec_z"]].to_numpy(float)
    sat_xyz = df[["sat_x", "sat_y", "sat_z"]].to_numpy(float)
    elev_deg = df["elevation"].to_numpy(float)
    azim_deg = df["azimuth"].to_numpy(float)

    # Wrap azimuth and convert to radians
    azim_wrapped = wrap_azimuth_deg(azim_deg)
    elev_rad = np.deg2rad(elev_deg)
    azim_rad = np.deg2rad(azim_wrapped)

    # Targets
    stec_label = df["stec_iri_ray"].to_numpy(float)

    # --- Unique nodes ---
    unique_rec, rec_indices = np.unique(rec_xyz, axis=0, return_inverse=True)
    unique_sat, sat_indices = np.unique(sat_xyz, axis=0, return_inverse=True)
    n_rec = len(unique_rec)
    n_sat = len(unique_sat)
    sat_indices_shifted = sat_indices + n_rec

    # --- Ray edge features ---
    ray_edge_attr = np.stack([elev_rad, azim_rad], axis=1)  # (n_rays, 2)

    # --- Bidirectional ray edges ---
    ray_edge_index = np.stack([rec_indices, sat_indices_shifted], axis=0)
    ray_edge_index = np.concatenate([ray_edge_index, ray_edge_index[::-1]], axis=1)
    ray_edge_attr = np.concatenate([ray_edge_attr, ray_edge_attr], axis=0)
    y_ray = np.concatenate([stec_label, stec_label], axis=0)
    num_ray_edges = len(y_ray)

    # --- Receiver KNN spatial edges ---
    nbrs = NearestNeighbors(n_neighbors=knn_k + 1).fit(unique_rec)
    _, knn_indices = nbrs.kneighbors(unique_rec)
    rec_rec_src, rec_rec_dst = [], []
    for i in range(n_rec):
        for ni in knn_indices[i]:
            if i != ni:
                rec_rec_src.append(i)
                rec_rec_dst.append(ni)
    spatial_edge_index = np.stack([rec_rec_src, rec_rec_dst], axis=0)
    spatial_edge_attr = np.zeros((spatial_edge_index.shape[1], 2))
    y_all = np.concatenate([y_ray, np.full(spatial_edge_index.shape[1], np.nan)], axis=0)

    # --- Merge edges ---
    final_edge_index = np.concatenate([ray_edge_index, spatial_edge_index], axis=1)
    final_edge_attr = np.concatenate([ray_edge_attr, spatial_edge_attr], axis=0)

    # --- Node features (ECEF XYZ) ---
    x_raw = np.concatenate([unique_rec, unique_sat], axis=0)

    # --- Train/test split on RAY EDGES ---
    train_idx, test_idx = _split_ray_edges(
        num_ray_edges, df, split=split, test_size=test_size, seed=seed,
    )

    # --- Scale using training ray edges only ---
    scaler_node = StandardScaler()
    scaler_node.fit(x_raw)
    x_scaled = scaler_node.transform(x_raw)

    scaler_edge = StandardScaler()
    # Fit edge scaler on training ray edges only (forward direction)
    train_forward = train_idx[train_idx < n_rays]
    scaler_edge.fit(ray_edge_attr[train_forward])
    edge_attr_scaled = scaler_edge.transform(final_edge_attr)

    # Fit y scaler on training ray edges only
    scaler_y = StandardScaler()
    scaler_y.fit(y_ray[train_idx].reshape(-1, 1))

    # --- Create PyG Data ---
    data = Data(
        x=torch.FloatTensor(x_scaled),
        edge_index=torch.LongTensor(final_edge_index),
        edge_attr=torch.FloatTensor(edge_attr_scaled),
    )
    # Store scaled y separately (PyG Data.y only for ray edges + NaN for KNN)
    y_scaled_full = np.full(len(y_all), np.nan, dtype=np.float64)
    y_scaled_full[:num_ray_edges] = scaler_y.transform(y_ray.reshape(-1, 1)).flatten()
    data.y = torch.FloatTensor(y_scaled_full).unsqueeze(1)

    train_mask = torch.zeros(final_edge_index.shape[1], dtype=torch.bool)
    test_mask = torch.zeros(final_edge_index.shape[1], dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.test_mask = test_mask

    stats = {
        "n_nodes": int(x_scaled.shape[0]),
        "n_edges_total": int(final_edge_index.shape[1]),
        "n_ray_edges": int(num_ray_edges),
        "n_train_edges": int(train_mask.sum()),
        "n_test_edges": int(test_mask.sum()),
        "n_rec": int(n_rec),
        "n_sat": int(n_sat),
        "n_rays_original": int(n_rays),
    }

    return {
        "data": data,
        "scaler_node": scaler_node,
        "scaler_edge": scaler_edge,
        "scaler_y": scaler_y,
        "unique_rec": unique_rec,
        "unique_sat": unique_sat,
        "rec_indices": rec_indices,
        "sat_indices_shifted": sat_indices_shifted,
        "n_rec": n_rec,
        "n_rays": n_rays,
        "stats": stats,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def _split_ray_edges(
    num_ray_edges: int,
    df: pd.DataFrame,
    *,
    split: str = "random",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ray edges into train/test. Only supports random split for now."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_ray_edges)
    n_train = int((1.0 - test_size) * num_ray_edges)
    return indices[:n_train], indices[n_train:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    data: Data,
    device: torch.device,
    *,
    epochs: int = 15000,
    lr: float = 0.001,
    patience: int = 500,
    window_size: int = 20,
    min_delta: float = 1e-6,
    log_interval: int | None = None,
) -> dict[str, Any]:
    """Train ImprovedSTEC_GNN with notebook-style early stopping."""

    data = data.to(device)
    model = ImprovedSTEC_GNN(
        num_node_features=data.x.shape[1],
        num_edge_features=data.edge_attr.shape[1],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = int(data.train_mask.sum())
    print(f"Model parameters: {n_params}")
    print(f"Param / train ratio: {n_params / n_train:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = RobustEarlyStopping(
        patience=patience, min_delta=min_delta, window_size=window_size,
    )

    if log_interval is None:
        log_interval = max(1, epochs // 100)

    train_losses: list[float] = []
    val_losses: list[float] = []

    print(f"Starting training ({epochs} epochs max)...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = criterion(val_out[data.test_mask], data.y[data.test_mask]).item()
        val_losses.append(val_loss)
        model.train()

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}, best val at epoch {early_stopping.best_epoch}")
            break

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stopped_epoch": epoch + 1,
        "best_epoch": early_stopping.best_epoch,
        "best_val_loss": early_stopping.best_loss,
        "n_params": n_params,
    }


# ---------------------------------------------------------------------------
# Vertical query & VTEC evaluation
# ---------------------------------------------------------------------------


def _geodetic_to_ecef(lon_deg: np.ndarray, lat_deg: np.ndarray, height_m: np.ndarray):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    x, y, z = t.transform(lon_deg, lat_deg, height_m)
    return np.column_stack((x, y, z))


def evaluate_vtec(
    graph: dict,
    result: dict,
    df: pd.DataFrame,
    device: torch.device,
    vertical_top_km: float = 2000.0,
) -> dict[str, Any]:
    """Build vertical query graph, run inference, compute VTEC metrics vs IRI ref."""

    model = result["model"]
    data = graph["data"]
    scaler_y = graph["scaler_y"]
    scaler_edge = graph["scaler_edge"]
    scaler_node = graph["scaler_node"]
    unique_rec = graph["unique_rec"]
    n_rec = graph["n_rec"]
    n_rays = graph["n_rays"]
    test_idx = graph["test_idx"]

    # Test rays (forward-direction only, original indices)
    test_ray_forward = test_idx[test_idx < n_rays]
    n_test = len(test_ray_forward)
    print(f"Test rays (forward): {n_test}")

    # --- Get test ray data ---
    df_test = df.iloc[test_ray_forward].reset_index(drop=True)
    ipp_lon = df_test["ipp_lon"].to_numpy(float)
    ipp_lat = df_test["ipp_lat"].to_numpy(float)
    stec_source = df_test["stec_iri_observable"].to_numpy(float)
    vtec_iri_ref = df_test["vtec_iri_ref"].to_numpy(float)
    vtec_slm = df_test["vtec_slm_from_iri_stec"].to_numpy(float)
    mapping_factor = df_test["slm_mapping_factor_iri"].to_numpy(float)
    slant_label = df_test["stec_iri_ray"].to_numpy(float)

    # --- Get slant STEC predictions for test rays ---
    all_nodes = data.x.shape[0]
    with torch.no_grad():
        data_dev = data.to(device)
        all_preds = model(data_dev).cpu().numpy().flatten()

    # Slant preds for test edges (forward + reverse)
    slant_pred_scaled = all_preds[test_idx]  # includes both directions
    # Forward direction only
    fwd_mask = test_idx < n_rays
    test_fwd_indices = test_idx[fwd_mask]
    slant_pred_fwd_scaled = all_preds[test_fwd_indices]
    slant_pred_fwd = np.maximum(
        scaler_y.inverse_transform(slant_pred_fwd_scaled.reshape(-1, 1)).flatten(), 0.0,
    )

    # --- Build vertical query ---
    rec_ecef = _geodetic_to_ecef(ipp_lon, ipp_lat, np.zeros(n_test))
    sat_ecef = _geodetic_to_ecef(ipp_lon, ipp_lat, np.full(n_test, vertical_top_km * 1000.0))

    # Scale virtual nodes
    all_nodes_raw = np.concatenate([unique_rec, graph["unique_sat"]], axis=0)
    all_nodes_ext = np.concatenate([all_nodes_raw, rec_ecef, sat_ecef], axis=0)
    x_ext = torch.FloatTensor(scaler_node.transform(all_nodes_ext)).to(device)

    # Vertical edges
    v_rec_start = all_nodes
    v_sat_start = all_nodes + n_test
    v_forward = np.stack([
        np.arange(v_rec_start, v_sat_start),
        np.arange(v_sat_start, v_sat_start + n_test),
    ], axis=0)
    v_reverse = np.stack([v_forward[1], v_forward[0]], axis=0)
    v_edges = np.concatenate([v_forward, v_reverse], axis=1)

    v_attr_one = np.column_stack((np.full(n_test, np.pi / 2.0), np.zeros(n_test)))
    v_attr = np.concatenate([v_attr_one, v_attr_one], axis=0)

    # Extend edge index and attributes
    ei_base = data.edge_index.numpy()
    ea_base_scaled = all_preds  # not needed, reconstruct
    ea_base = scaler_edge.inverse_transform(data.edge_attr.numpy())
    ei_ext = torch.LongTensor(np.concatenate([ei_base, v_edges], axis=1)).to(device)
    ea_ext_raw = np.concatenate([ea_base, v_attr], axis=0)
    ea_ext = torch.FloatTensor(scaler_edge.transform(ea_ext_raw)).to(device)

    # --- Run inference on augmented graph ---
    data_ext = Data(x=x_ext, edge_index=ei_ext, edge_attr=ea_ext)
    with torch.no_grad():
        all_preds_ext = model(data_ext).cpu().numpy().flatten()

    v_start_idx = ei_base.shape[1]
    v_forward_end = v_start_idx + n_test
    vtec_gnn_direct_scaled = all_preds_ext[v_start_idx:v_forward_end]
    vtec_gnn_direct = np.maximum(
        scaler_y.inverse_transform(vtec_gnn_direct_scaled.reshape(-1, 1)).flatten(), 0.0,
    )

    # --- VTEC conversion ---
    gnn_ratio = np.where(
        np.abs(slant_pred_fwd) > 1e-6,
        vtec_gnn_direct / slant_pred_fwd,
        1.0 / mapping_factor,
    )
    vtec_gnn = stec_source * gnn_ratio

    # --- Metrics ---
    def _m(pred: np.ndarray, ref: np.ndarray, label: str) -> dict:
        finite = np.isfinite(pred) & np.isfinite(ref)
        p, r = pred[finite], ref[finite]
        if len(p) == 0:
            return {"model": label, "n": 0}
        e = p - r
        ss_res = float(np.sum(e ** 2))
        ss_tot = float(np.sum((r - np.mean(r)) ** 2))
        return {
            "model": label,
            "n": int(len(p)),
            "rmse_tecu": float(np.sqrt(np.mean(e ** 2))),
            "mae_tecu": float(np.mean(np.abs(e))),
            "bias_tecu": float(np.mean(e)),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "pred_mean": float(np.mean(p)),
            "ref_mean": float(np.mean(r)),
        }

    ref = vtec_iri_ref
    m_slm = _m(vtec_slm, ref, "slm")
    m_gnn = _m(vtec_gnn, ref, "gnn_ratio")
    m_gnn_dir = _m(vtec_gnn_direct, ref, "gnn_direct")

    # STEC metrics
    stec_rmse = float(np.sqrt(np.mean((slant_pred_fwd - slant_label) ** 2)))
    stec_r2 = float(r2_score(slant_label, slant_pred_fwd))

    print(f"\n{'='*60}")
    print("VTEC Conversion Results (vs vtec_iri_ref)")
    print(f"{'='*60}")
    for m in [m_slm, m_gnn, m_gnn_dir]:
        print(f"\n{m['model']}:")
        print(f"  RMSE: {m['rmse_tecu']:.4f} TECU")
        print(f"  MAE:  {m['mae_tecu']:.4f} TECU")
        print(f"  Bias: {m['bias_tecu']:.4f} TECU")
        print(f"  R2:   {m['r2']:.4f}")
        print(f"  N:    {m['n']}")

    delta_rmse = m_gnn["rmse_tecu"] - m_slm["rmse_tecu"]
    pct = -delta_rmse / m_slm["rmse_tecu"] * 100.0
    print(f"\nGNN vs SLM: dRMSE = {delta_rmse:+.4f} TECU ({pct:+.1f}%)")
    print(f"  (negative = GNN better)")

    print(f"\nGNN Slant STEC: RMSE={stec_rmse:.4f} TECU, R2={stec_r2:.4f}")

    return {
        "vtec_metrics": {"slm": m_slm, "gnn_ratio": m_gnn, "gnn_direct": m_gnn_dir},
        "stec_metrics": {"rmse_tecu": stec_rmse, "r2": stec_r2},
        "n_test_rays": n_test,
        "vtec_predictions": {
            "vtec_slm": vtec_slm.tolist(),
            "vtec_gnn": vtec_gnn.tolist(),
            "vtec_gnn_direct": vtec_gnn_direct.tolist(),
            "vtec_iri_ref": vtec_iri_ref.tolist(),
            "slant_pred_fwd": slant_pred_fwd.tolist(),
            "slant_label": slant_label.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ImprovedSTEC_GNN (SAGEConv) on IRI simulation data"
    )
    p.add_argument("--input",
                   default="datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.parquet")
    p.add_argument("--run-name", default=None)
    p.add_argument("--split", default="random", choices=["random"])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--epochs", type=int, default=15000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=500)
    p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--knn-k", type=int, default=4)
    p.add_argument("--vertical-top-km", type=float, default=2000.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--table-dir", default="results/tables")
    p.add_argument("--checkpoint-dir", default="results/checkpoints")
    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if args.run_name is None:
        args.run_name = f"gnn_iri_pyg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(f"Run: {args.run_name}")

    os.makedirs(args.table_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Load data ---
    print(f"\nLoading: {args.input}")
    df = pd.read_parquet(args.input)
    if args.max_rows is not None:
        df = df.head(args.max_rows)
    print(f"  Rows: {len(df)}")

    # --- Build graph ---
    print("\nBuilding graph...")
    graph = build_graph_from_parquet(
        df, knn_k=args.knn_k, seed=args.seed,
        split=args.split, test_size=args.test_size,
    )
    stats = graph["stats"]
    print(f"  Nodes: {stats['n_nodes']} (rec={stats['n_rec']}, sat={stats['n_sat']})")
    print(f"  Ray edges: {stats['n_ray_edges']}, Total edges: {stats['n_edges_total']}")
    print(f"  Train: {stats['n_train_edges']}, Test: {stats['n_test_edges']}")

    # --- Train ---
    print("\n--- Training ---")
    result = train_model(
        graph["data"], device,
        epochs=args.epochs, lr=args.lr,
        patience=args.patience, window_size=args.window_size,
        min_delta=args.min_delta,
    )
    print(f"Training finished: stopped at epoch {result['stopped_epoch']}, "
          f"best val at epoch {result['best_epoch']}")

    # --- VTEC evaluation ---
    print("\n--- VTEC Evaluation ---")
    eval_result = evaluate_vtec(
        graph, result, df, device, vertical_top_km=args.vertical_top_km,
    )

    # --- Save model ---
    model_path = os.path.join(args.checkpoint_dir, f"{args.run_name}.pth")
    torch.save(result["model"].state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    # --- Save scalers ---
    scaler_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_scalers.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "node_scaler": graph["scaler_node"],
            "edge_scaler": graph["scaler_edge"],
            "y_scaler": graph["scaler_y"],
        }, f)
    print(f"Scalers saved to: {scaler_path}")

    # --- Save metrics ---
    output = {
        "run_name": args.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "graph_stats": stats,
        "training": {
            "epochs_max": args.epochs,
            "stopped_epoch": result["stopped_epoch"],
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "lr": args.lr,
            "patience": args.patience,
            "window_size": args.window_size,
            "n_params": result["n_params"],
        },
        "metrics": eval_result["vtec_metrics"],
        "stec_metrics": eval_result["stec_metrics"],
    }
    metrics_path = os.path.join(args.table_dir, f"{args.run_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
