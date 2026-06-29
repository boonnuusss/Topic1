"""Reproduce the GNN_sTEC_improve.ipynb experiment faithfully.

This entry point mirrors the legacy notebook exactly:
- Data: obs_sTEC_2024128_1000_regional_big.npz (observed GNSS STEC)
- Model: ImprovedSTEC_GNN with torch_geometric.nn.SAGEConv
- Training: 15000 epochs, Adam (lr=1e-3), MSE, early stopping (patience=500, window=20)
- Evaluation: STEC-space RMSE and R^2 on random 80/20 split
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# ============================================================
# 1. Geometry helpers (exact copies from the notebook)
# ============================================================

def xyz2llh(xyz: np.ndarray) -> np.ndarray:
    """ECEF XYZ -> geodetic lat/lon (radians)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a, e2 = 6378137.0, 0.00669437999014
    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * a, p * np.sqrt(a ** 2 - z ** 2 * e2))
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + (e2 * a * (np.sin(theta) ** 3)),
        p - (e2 * a * (np.cos(theta) ** 3)),
    )
    return np.column_stack((lat, lon))


def ecef2enu(rec_xyz: np.ndarray, sat_xyz: np.ndarray):
    """Compute elevation (rad) and azimuth (rad) from ECEF positions."""
    rec_llh = xyz2llh(rec_xyz)
    lat, lon = rec_llh[:, 0], rec_llh[:, 1]
    rx = sat_xyz[:, 0] - rec_xyz[:, 0]
    ry = sat_xyz[:, 1] - rec_xyz[:, 1]
    rz = sat_xyz[:, 2] - rec_xyz[:, 2]
    e = -np.sin(lon) * rx + np.cos(lon) * ry
    n = -np.sin(lat) * np.cos(lon) * rx - np.sin(lat) * np.sin(lon) * ry + np.cos(lat) * rz
    u = np.cos(lat) * np.cos(lon) * rx + np.cos(lat) * np.sin(lon) * ry + np.sin(lat) * rz
    elev = np.arctan2(u, np.sqrt(e ** 2 + n ** 2))
    azim = np.arctan2(e, n)
    return elev, azim


# ============================================================
# 2. Model (exact copy of ImprovedSTEC_GNN from the notebook)
# ============================================================

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


# ============================================================
# 3. Early stopping (exact copy of RobustEarlyStopping)
# ============================================================

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

        current_avg_loss = np.mean(self.loss_history)
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


# ============================================================
# 4. Graph construction (identical logic to notebook cells 1-2)
# ============================================================

def build_graph_from_npz(data_path: str, knn_k: int = 4) -> dict:
    """Load .npz data and build the PyG graph exactly as the notebook does.

    Returns a dict with:
        data: PyG Data object
        scaler_y: StandardScaler fitted on training targets
        scaler_node: StandardScaler fitted on node features
        scaler_edge: StandardScaler fitted on ray edge features
        train_mask, test_mask: boolean edge masks
        num_rays: number of directed ray edges (before bidirectional)
        metrics: dict of graph statistics
    """

    # --- Load data ---
    sTEC_data = np.load(data_path, allow_pickle=True)
    slantTEC = sTEC_data["slantTEC"].astype(np.float64)
    Rec_XYZ = sTEC_data["Rec_XYZ"].astype(np.float64)
    Sat_XYZ = sTEC_data["Sat_XYZ"].astype(np.float64)
    Rec_BLH = sTEC_data["Rec_BLH"].astype(np.float64)
    Sat_BLH = sTEC_data["Sat_BLH"].astype(np.float64)

    # Shape fix (matches notebook: if first dim is 3, transpose)
    if Rec_XYZ.shape[0] == 3:
        Rec_XYZ = Rec_XYZ.T
    if Sat_XYZ.shape[0] == 3:
        Sat_XYZ = Sat_XYZ.T
    if Rec_BLH.shape[0] == 3:
        Rec_BLH = Rec_BLH.T
    if Sat_BLH.shape[0] == 3:
        Sat_BLH = Sat_BLH.T

    # --- Unique nodes ---
    unique_rec, rec_indices = np.unique(Rec_XYZ, axis=0, return_inverse=True)
    unique_sat, sat_indices = np.unique(Sat_XYZ, axis=0, return_inverse=True)
    n_rec = len(unique_rec)
    n_sat = len(unique_sat)
    sat_indices_shifted = sat_indices + n_rec

    # --- Ray edge features (elevation, azimuth) ---
    elev, azim = ecef2enu(Rec_XYZ, Sat_XYZ)
    ray_edge_attr = np.stack([elev, azim], axis=1)  # (N_rays, 2)

    # --- Bidirectional ray edges ---
    ray_edge_index = np.stack([rec_indices, sat_indices_shifted], axis=0)
    ray_edge_index = np.concatenate([ray_edge_index, ray_edge_index[::-1]], axis=1)
    ray_edge_attr = np.concatenate([ray_edge_attr, ray_edge_attr], axis=0)
    y = np.concatenate([slantTEC, slantTEC], axis=0)
    ray_mask_len = len(y)

    # --- Receiver KNN spatial edges ---
    nbrs = NearestNeighbors(n_neighbors=knn_k + 1).fit(unique_rec)
    distances, indices = nbrs.kneighbors(unique_rec)
    rec_rec_src = []
    rec_rec_dst = []
    for i in range(n_rec):
        for neighbor_idx in indices[i]:
            if i != neighbor_idx:
                rec_rec_src.append(i)
                rec_rec_dst.append(neighbor_idx)
    spatial_edge_index = np.stack([rec_rec_src, rec_rec_dst], axis=0)
    spatial_edge_attr = np.zeros((spatial_edge_index.shape[1], 2))
    spatial_y = np.zeros(spatial_edge_index.shape[1])
    y = np.concatenate([y, spatial_y], axis=0)

    # --- Merge edges ---
    final_edge_index = np.concatenate([ray_edge_index, spatial_edge_index], axis=1)
    final_edge_attr = np.concatenate([ray_edge_attr, spatial_edge_attr], axis=0)

    # --- Node features ---
    x = np.concatenate([unique_rec, unique_sat], axis=0)
    scaler_node = StandardScaler()
    x = scaler_node.fit_transform(x)
    x = torch.FloatTensor(x)

    # --- Edge features (fit scaler on ray edges only) ---
    scaler_edge = StandardScaler()
    scaler_edge.fit(ray_edge_attr)
    edge_attr = scaler_edge.transform(final_edge_attr)
    edge_attr = torch.FloatTensor(edge_attr)

    edge_index = torch.LongTensor(final_edge_index)

    # --- Target scaling ---
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    y_tensor = torch.FloatTensor(y_scaled).unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)

    # --- Train/test masks on ray edges only ---
    num_rays = ray_mask_len
    rng = np.random.default_rng(42)
    indices_arr = rng.permutation(num_rays)
    train_size = int(0.8 * num_rays)
    train_ray = indices_arr[:train_size]
    test_ray = indices_arr[train_size:]

    train_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    test_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    train_mask[train_ray] = True
    test_mask[test_ray] = True
    data.train_mask = train_mask
    data.test_mask = test_mask

    stats = {
        "n_nodes": int(data.x.shape[0]),
        "n_edges_total": int(data.edge_index.shape[1]),
        "n_ray_edges": int(num_rays),
        "n_train_edges": int(train_mask.sum().item()),
        "n_test_edges": int(test_mask.sum().item()),
        "n_rec": int(n_rec),
        "n_sat": int(n_sat),
        "edge_feature_dim": int(data.edge_attr.shape[1]),
    }

    return {
        "data": data,
        "scaler_y": scaler_y,
        "scaler_node": scaler_node,
        "scaler_edge": scaler_edge,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "num_rays": num_rays,
        "stats": stats,
    }


# ============================================================
# 5. Training loop (matches notebook cell 2)
# ============================================================

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
) -> dict:
    """Train ImprovedSTEC_GNN with notebook-style early stopping."""

    data = data.to(device)
    model = ImprovedSTEC_GNN(
        num_node_features=data.x.shape[1],
        num_edge_features=data.edge_attr.shape[1],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = int(data.train_mask.sum())
    param_ratio = n_params / n_train if n_train > 0 else 0
    print(f"Model parameters: {n_params}")
    print(f"Parameter / training sample ratio: {param_ratio:.2f} (ideal < 1.0)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = RobustEarlyStopping(
        patience=patience, min_delta=min_delta, window_size=window_size
    )

    if log_interval is None:
        log_interval = max(1, epochs // 100)

    train_losses: list[float] = []
    val_losses: list[float] = []

    print(f"Starting GNN training ({epochs} epochs max)...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
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
            print(
                f"Early stopping at epoch {epoch+1}, "
                f"best val loss at epoch {early_stopping.best_epoch}"
            )
            break

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stopped_epoch": epoch + 1,
        "best_epoch": early_stopping.best_epoch,
        "best_val_loss": early_stopping.best_loss,
        "n_params": n_params,
        "param_ratio": param_ratio,
    }


# ============================================================
# 6. Evaluation (matches notebook cell 3)
# ============================================================

def evaluate_model(
    model: nn.Module,
    data: Data,
    scaler_y: StandardScaler,
    device: torch.device,
) -> dict:
    """Evaluate STEC prediction on test edges."""

    model.eval()
    with torch.no_grad():
        pred = model(data.to(device))
        pred_test = pred[data.test_mask].cpu().numpy()
        y_test = data.y[data.test_mask].cpu().numpy()

        pred_test_inv = scaler_y.inverse_transform(pred_test).flatten()
        y_test_inv = scaler_y.inverse_transform(y_test).flatten()

        # Non-negativity clamp
        pred_test_inv = np.maximum(pred_test_inv, 0.0)

        rmse = float(np.sqrt(mean_squared_error(y_test_inv, pred_test_inv)))
        r2 = float(r2_score(y_test_inv, pred_test_inv))
        mae = float(np.mean(np.abs(pred_test_inv - y_test_inv)))
        bias = float(np.mean(pred_test_inv - y_test_inv))

    return {
        "rmse_tecu": rmse,
        "r2": r2,
        "mae_tecu": mae,
        "bias_tecu": bias,
        "pred_mean_tecu": float(np.mean(pred_test_inv)),
        "target_mean_tecu": float(np.mean(y_test_inv)),
        "n_test": int(len(y_test_inv)),
    }


# ============================================================
# 7. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the GNN_sTEC_improve.ipynb experiment"
    )
    parser.add_argument(
        "--data-path",
        default="../00_iri/mycode/ML_sTEC/Mydata/obs_sTEC_2024128_1000_regional_big.npz",
        help="Path to the legacy .npz data file",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name for output files (default: gnn_notebook_repro_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15000,
        help="Maximum training epochs (default: 15000, matching notebook)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Adam learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=500, help="Early stopping patience"
    )
    parser.add_argument(
        "--window-size", type=int, default=20, help="Early stopping window size"
    )
    parser.add_argument(
        "--knn-k", type=int, default=4, help="K for receiver KNN edges"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, or cuda",
    )
    parser.add_argument(
        "--table-dir",
        default="results/tables",
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="results/checkpoints",
        help="Output directory for model and scalers",
    )
    parser.add_argument(
        "--figure-dir",
        default="results/figures",
        help="Output directory for scatter plot",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip scatter plot generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Resolve device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Resolve paths ---
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.normpath(os.path.join(os.getcwd(), data_path))
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Data: {data_path}")

    if args.run_name is None:
        args.run_name = f"gnn_notebook_repro_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(f"Run: {args.run_name}")

    os.makedirs(args.table_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    # --- Build graph ---
    print("\n--- Building graph ---")
    graph = build_graph_from_npz(data_path, knn_k=args.knn_k)
    data = graph["data"]
    stats = graph["stats"]
    print(f"Graph: nodes={stats['n_nodes']}, total edges={stats['n_edges_total']}, "
          f"ray edges={stats['n_ray_edges']}")
    print(f"Train edges: {stats['n_train_edges']}, Test edges: {stats['n_test_edges']}")

    # --- Train ---
    print("\n--- Training ---")
    result = train_model(
        data,
        device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        window_size=args.window_size,
    )
    model = result["model"]
    print(f"Training finished: stopped at epoch {result['stopped_epoch']}, "
          f"best val at epoch {result['best_epoch']}")

    # --- Evaluate ---
    print("\n--- Evaluation ---")
    metrics = evaluate_model(model, data, graph["scaler_y"], device)
    print(f"STEC Test RMSE: {metrics['rmse_tecu']:.4f} TECU")
    print(f"STEC Test MAE:  {metrics['mae_tecu']:.4f} TECU")
    print(f"STEC Test Bias: {metrics['bias_tecu']:.4f} TECU")
    print(f"STEC Test R2:   {metrics['r2']:.4f}")
    print(f"Test samples:   {metrics['n_test']}")

    # --- Save model ---
    model_path = os.path.join(args.checkpoint_dir, f"{args.run_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    # --- Save scalers ---
    scaler_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_scalers.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(
            {
                "node_scaler": graph["scaler_node"],
                "edge_scaler": graph["scaler_edge"],
                "y_scaler": graph["scaler_y"],
            },
            f,
        )
    print(f"Scalers saved to: {scaler_path}")

    # --- Save metrics ---
    output = {
        "run_name": args.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path,
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
            "param_ratio": result["param_ratio"],
        },
        "metrics": metrics,
    }
    metrics_path = os.path.join(args.table_dir, f"{args.run_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"Metrics saved to: {metrics_path}")

    # --- Scatter plot ---
    if not args.no_plot:
        print("\n--- Generating scatter plot ---")
        _save_scatter_plot(
            model, data, graph["scaler_y"], device, args.run_name, args.figure_dir
        )


def _save_scatter_plot(
    model: nn.Module,
    data: Data,
    scaler_y: StandardScaler,
    device: torch.device,
    run_name: str,
    figure_dir: str,
) -> None:
    """Save predicted-vs-true scatter plot (matches notebook output)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        pred = model(data.to(device))
        pred_test = pred[data.test_mask].cpu().numpy()
        y_test = data.y[data.test_mask].cpu().numpy()
        pred_test_inv = np.maximum(
            scaler_y.inverse_transform(pred_test).flatten(), 0.0
        )
        y_test_inv = scaler_y.inverse_transform(y_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test_inv, pred_test_inv))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test_inv, pred_test_inv, alpha=0.6, s=20, color="blue", label="GNN")
    ax.plot(
        [y_test_inv.min(), y_test_inv.max()],
        [y_test_inv.min(), y_test_inv.max()],
        "r--",
        lw=2,
    )
    ax.set_xlabel("True STEC (TECU)")
    ax.set_ylabel("Predicted STEC (TECU)")
    ax.set_title(f"GNN Notebook Reproduction (RMSE={rmse:.2f} TECU)")
    ax.legend()
    ax.grid(True)

    fig_path = os.path.join(figure_dir, f"{run_name}_scatter.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved to: {fig_path}")


if __name__ == "__main__":
    main()
