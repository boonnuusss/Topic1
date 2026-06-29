"""Evaluate the notebook-trained GNN on STEC-to-VTEC conversion.

Loads the GNN checkpoint from Experiment 029, computes IRI VTEC reference at each
test-ray IPP, builds vertical query edges, and compares GNN ratio-based VTEC
against the SLM baseline and IRI VTEC reference.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TECU_SCALE = 1.0e16
EARTH_RADIUS_KM = 6371.0


# ============================================================
# 1. Geometry helpers (from notebook + src/geometry.py)
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


def geodetic_to_ecef(lon_deg: np.ndarray, lat_deg: np.ndarray, height_m: np.ndarray):
    """Convert geodetic lon/lat/height to ECEF meters using pyproj."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    x, y, z = transformer.transform(lon_deg, lat_deg, height_m)
    return np.column_stack((x, y, z))


def compute_slant_factor(elevation_deg: np.ndarray, shell_height_km: float = 450.0) -> np.ndarray:
    """Thin-shell mapping factor M(elevation)."""
    elevation_rad = np.deg2rad(elevation_deg)
    ratio = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + shell_height_km)
    inside = 1.0 - (ratio * np.cos(elevation_rad)) ** 2
    return 1.0 / np.sqrt(np.maximum(inside, 1e-12))


# ============================================================
# 2. Model (exact copy from train_gnn_notebook_repro.py)
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
# 3. IRI VTEC computation
# ============================================================

def _get_pyiri():
    """Import vendored PyIRI (mirrors src/iri_simulation_builder.py)."""
    from pathlib import Path
    pyiri_path = Path(__file__).resolve().parents[2] / "00_iri" / "PyIRI"
    path_str = str(pyiri_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    import PyIRI
    import PyIRI.edp_update as iri_density
    return PyIRI, iri_density


def compute_iri_vtec_at_ipps(
    ipp_lon: np.ndarray,
    ipp_lat: np.ndarray,
    timestamp,
    *,
    f107: float = 204.0,
    ccir_or_ursi: int = 0,
    alt_min_km: float = 90.0,
    alt_max_km: float = 2000.0,
    dalt_km: float = 5.0,
) -> np.ndarray:
    """Compute IRI VTEC (TECU) at each IPP by vertical integration.

    Parameters
    ----------
    ipp_lon, ipp_lat : 1D arrays of length N (degrees)
    timestamp : datetime-like with year/month/day/hour/minute
    f107 : solar radio flux index
    ccir_or_ursi : 0 for CCIR, 1 for URSI

    Returns
    -------
    vtec : 1D array of length N in TECU
    """
    PyIRI, iri_density = _get_pyiri()

    alt_grid_km = np.arange(alt_min_km, alt_max_km + dalt_km / 2, dalt_km)
    n_points = len(ipp_lon)
    ahr = np.full(n_points, float(timestamp.hour) + float(timestamp.minute) / 60.0)

    _, _, _, _, _, _, edp = iri_density.IRI_density_1day(
        int(timestamp.year),
        int(timestamp.month),
        int(timestamp.day),
        ahr,
        ipp_lon.astype(float),
        ipp_lat.astype(float),
        alt_grid_km.astype(float),
        float(f107),
        PyIRI.coeff_dir,
        int(ccir_or_ursi),
    )

    edp_profiles = edp[0]  # (n_alt, n_points)

    try:
        integral = np.trapezoid(edp_profiles, x=alt_grid_km * 1000.0, axis=0)
    except AttributeError:
        integral = np.trapz(edp_profiles, x=alt_grid_km * 1000.0, axis=0)

    return integral / TECU_SCALE


# ============================================================
# 4. Main evaluation logic
# ============================================================

def run_evaluation(
    data_path: str,
    model_path: str,
    scaler_path: str,
    *,
    device: torch.device,
    knn_k: int = 4,
    shell_height_km: float = 450.0,
    vertical_top_km: float = 2000.0,
    f107: float = 204.0,
    ccir_or_ursi: int = 0,
) -> dict[str, Any]:
    """Load model, compute IRI VTEC, run vertical query, evaluate VTEC metrics."""

    # ---- 3a. Load .npz data ----
    print("Loading .npz data...")
    sTEC_data = np.load(data_path, allow_pickle=True)
    slantTEC_all = sTEC_data["slantTEC"].astype(np.float64)
    Rec_XYZ_all = sTEC_data["Rec_XYZ"].astype(np.float64)
    Sat_XYZ_all = sTEC_data["Sat_XYZ"].astype(np.float64)
    ipp_lon_all = sTEC_data["ipp_lon"].astype(np.float64)
    ipp_lat_all = sTEC_data["ipp_lat"].astype(np.float64)
    timestamp = sTEC_data["target_time"].item()

    # Shape fix
    if Rec_XYZ_all.shape[0] == 3:
        Rec_XYZ_all = Rec_XYZ_all.T
    if Sat_XYZ_all.shape[0] == 3:
        Sat_XYZ_all = Sat_XYZ_all.T

    print(f"  Rays: {len(slantTEC_all)}, Epoch: {timestamp}")

    # ---- 3b. Build graph (same as training) ----
    print("Building graph...")
    unique_rec, rec_indices = np.unique(Rec_XYZ_all, axis=0, return_inverse=True)
    unique_sat, sat_indices = np.unique(Sat_XYZ_all, axis=0, return_inverse=True)
    n_rec = len(unique_rec)
    n_sat = len(unique_sat)
    sat_indices_shifted = sat_indices + n_rec

    elev_rad, azim_rad = ecef2enu(Rec_XYZ_all, Sat_XYZ_all)
    elevation_deg = np.rad2deg(elev_rad)
    ray_edge_attr = np.stack([elev_rad, azim_rad], axis=1)

    # Bidirectional ray edges
    ray_edge_index = np.stack([rec_indices, sat_indices_shifted], axis=0)
    ray_edge_index = np.concatenate([ray_edge_index, ray_edge_index[::-1]], axis=1)
    ray_edge_attr = np.concatenate([ray_edge_attr, ray_edge_attr], axis=0)
    y_all = np.concatenate([slantTEC_all, slantTEC_all], axis=0)
    num_ray_edges = len(y_all)

    # KNN spatial edges
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
    y_all = np.concatenate([y_all, np.zeros(spatial_edge_index.shape[1])], axis=0)

    final_edge_index = np.concatenate([ray_edge_index, spatial_edge_index], axis=1)
    final_edge_attr = np.concatenate([ray_edge_attr, spatial_edge_attr], axis=0)

    # Node features
    all_nodes_xyz = np.concatenate([unique_rec, unique_sat], axis=0)
    n_base_nodes = len(all_nodes_xyz)
    print(f"  Nodes: {n_base_nodes}, Ray edges: {num_ray_edges}, Total edges: {final_edge_index.shape[1]}")

    # ---- 3c. Load scalers ----
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    scaler_node: StandardScaler = scalers["node_scaler"]
    scaler_edge: StandardScaler = scalers["edge_scaler"]
    scaler_y: StandardScaler = scalers["y_scaler"]

    # ---- 3d. Train/test split (seed=42, matching training) ----
    rng = np.random.default_rng(42)
    indices_arr = rng.permutation(num_ray_edges)
    train_size = int(0.8 * num_ray_edges)
    test_ray = indices_arr[train_size:]
    # Map test ray indices back to original (pre-bidirectional) indices
    test_original = test_ray[test_ray < len(slantTEC_all)]
    n_test = len(test_original)
    print(f"  Test rays (original): {n_test}")

    # ---- 3e. Load model ----
    model = ImprovedSTEC_GNN(num_node_features=3, num_edge_features=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Model loaded.")

    # ---- 3f. Get slant STEC predictions for test rays ----
    print("Computing slant STEC predictions...")
    x_base = torch.FloatTensor(scaler_node.transform(all_nodes_xyz)).to(device)
    ei_base = torch.LongTensor(final_edge_index).to(device)
    ea_base = torch.FloatTensor(scaler_edge.transform(final_edge_attr)).to(device)

    with torch.no_grad():
        # Create a temporary graph with just the base edges for slant prediction
        data_slant = Data(x=x_base, edge_index=ei_base, edge_attr=ea_base)
        all_preds_scaled = model(data_slant).cpu().numpy().flatten()
    slant_pred_scaled = all_preds_scaled[test_ray]  # bidirectional test edges
    slant_pred = np.maximum(scaler_y.inverse_transform(slant_pred_scaled.reshape(-1, 1)).flatten(), 0.0)

    # Map to original order: the test_ray indices include both forward and reverse edges.
    # For original-forward test rays: test_ray indices < 4049 are forward; >= 4049 are reverse.
    # Use only the forward-direction test edges
    test_ray_forward = test_ray[test_ray < len(slantTEC_all)]
    slant_pred_forward = all_preds_scaled[test_ray_forward]
    slant_pred_forward_tcu = np.maximum(
        scaler_y.inverse_transform(slant_pred_forward.reshape(-1, 1)).flatten(), 0.0
    )
    # Same test rays, forward-direction ground truth
    stec_obs_test = slantTEC_all[test_ray_forward]
    elev_test_deg = elevation_deg[test_ray_forward]
    ipp_lon_test = ipp_lon_all[test_ray_forward]
    ipp_lat_test = ipp_lat_all[test_ray_forward]

    # ---- 3g. Compute IRI VTEC at test ray IPPs ----
    print(f"Computing IRI VTEC at {n_test} IPP locations...")
    # Deduplicate IPPs for efficiency
    ipp_rounded = np.column_stack((
        np.round(ipp_lon_test, 4),
        np.round(ipp_lat_test, 4),
    ))
    _, unique_idx, inverse_idx = np.unique(ipp_rounded, axis=0, return_index=True, return_inverse=True)
    unique_lon = ipp_lon_test[unique_idx]
    unique_lat = ipp_lat_test[unique_idx]
    print(f"  Unique IPP locations: {len(unique_idx)}")

    # Compute IRI VTEC in batches to avoid OOM (792 points x 383 alt = too large)
    batch_size = 80
    vtec_iri_unique = np.empty(len(unique_idx), dtype=np.float64)
    for b_start in range(0, len(unique_idx), batch_size):
        b_end = min(b_start + batch_size, len(unique_idx))
        print(f"  IRI batch {b_start // batch_size + 1}/{(len(unique_idx) + batch_size - 1) // batch_size} "
              f"({b_start}-{b_end})")
        vtec_iri_unique[b_start:b_end] = compute_iri_vtec_at_ipps(
            unique_lon[b_start:b_end], unique_lat[b_start:b_end], timestamp,
            f107=f107, ccir_or_ursi=ccir_or_ursi,
        )
    vtec_iri_ref = vtec_iri_unique[inverse_idx]
    print(f"  IRI VTEC range: [{vtec_iri_ref.min():.2f}, {vtec_iri_ref.max():.2f}] TECU")

    # ---- 3h. Build vertical query graph ----
    print("Building vertical query graph...")
    # Virtual receiver at IPP ground, virtual satellite at IPP top
    rec_ecef = geodetic_to_ecef(ipp_lon_test, ipp_lat_test, np.zeros(n_test))
    sat_ecef = geodetic_to_ecef(ipp_lon_test, ipp_lat_test, np.full(n_test, vertical_top_km * 1000.0))

    # Extend node features
    all_nodes_ext = np.concatenate([all_nodes_xyz, rec_ecef, sat_ecef], axis=0)
    x_ext = torch.FloatTensor(scaler_node.transform(all_nodes_ext)).to(device)

    # Vertical edge indices
    v_rec_start = n_base_nodes
    v_sat_start = n_base_nodes + n_test
    v_forward = np.stack([
        np.arange(v_rec_start, v_sat_start),
        np.arange(v_sat_start, v_sat_start + n_test),
    ], axis=0)
    v_reverse = np.stack([v_forward[1], v_forward[0]], axis=0)
    v_edges = np.concatenate([v_forward, v_reverse], axis=1)

    # Vertical edge features: elevation=pi/2, azimuth=0
    v_attr_one = np.column_stack((
        np.full(n_test, np.pi / 2.0),
        np.zeros(n_test),
    ))
    v_attr = np.concatenate([v_attr_one, v_attr_one], axis=0)

    # Extend edges
    ei_ext = torch.LongTensor(np.concatenate([final_edge_index, v_edges], axis=1)).to(device)
    ea_ext = torch.FloatTensor(
        scaler_edge.transform(np.concatenate([final_edge_attr, v_attr], axis=0))
    ).to(device)

    # ---- 3i. Run GNN on augmented graph ----
    print("Running GNN inference on augmented graph...")
    data_ext = Data(x=x_ext, edge_index=ei_ext, edge_attr=ea_ext)
    with torch.no_grad():
        all_preds_ext = model(data_ext).cpu().numpy().flatten()

    # Extract vertical forward-query predictions
    v_start_idx = final_edge_index.shape[1]  # where vertical edges begin
    v_forward_end = v_start_idx + n_test
    vtec_gnn_direct_scaled = all_preds_ext[v_start_idx:v_forward_end]
    vtec_gnn_direct = np.maximum(
        scaler_y.inverse_transform(vtec_gnn_direct_scaled.reshape(-1, 1)).flatten(), 0.0
    )

    # ---- 3j. VTEC conversion ----
    # SLM
    slm_factor = compute_slant_factor(elev_test_deg, shell_height_km=shell_height_km)
    vtec_slm = stec_obs_test / slm_factor

    # GNN ratio-based
    gnn_ratio = np.where(
        np.abs(slant_pred_forward_tcu) > 1e-6,
        vtec_gnn_direct / slant_pred_forward_tcu,
        1.0 / slm_factor,
    )
    vtec_gnn = stec_obs_test * gnn_ratio

    # ---- 3k. Metrics ----
    def _vtc_metrics(pred: np.ndarray, ref: np.ndarray, label: str) -> dict:
        finite = np.isfinite(pred) & np.isfinite(ref)
        p, r = pred[finite], ref[finite]
        if len(p) == 0:
            return {"model": label, "n_samples": 0}
        err = p - r
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((r - np.mean(r)) ** 2))
        return {
            "model": label,
            "n_samples": int(len(p)),
            "rmse_tecu": float(np.sqrt(np.mean(err ** 2))),
            "mae_tecu": float(np.mean(np.abs(err))),
            "bias_tecu": float(np.mean(err)),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "pred_mean_tecu": float(np.mean(p)),
            "ref_mean_tecu": float(np.mean(r)),
            "pred_std_tecu": float(np.std(p)),
            "ref_std_tecu": float(np.std(r)),
        }

    ref = vtec_iri_ref
    metrics_slm = _vtc_metrics(vtec_slm, ref, "slm")
    metrics_gnn = _vtc_metrics(vtec_gnn, ref, "gnn_ratio")
    metrics_gnn_direct = _vtc_metrics(vtec_gnn_direct, ref, "gnn_direct_vertical")

    print("\n" + "=" * 60)
    print("VTEC Conversion Results (vs IRI reference)")
    print("=" * 60)
    for m in [metrics_slm, metrics_gnn, metrics_gnn_direct]:
        print(f"\n{m['model']}:")
        print(f"  RMSE: {m['rmse_tecu']:.4f} TECU")
        print(f"  MAE:  {m['mae_tecu']:.4f} TECU")
        print(f"  Bias: {m['bias_tecu']:.4f} TECU")
        print(f"  R2:   {m['r2']:.4f}")
        print(f"  N:    {m['n_samples']}")

    # Comparison
    if metrics_slm["n_samples"] > 0:
        delta_rmse = metrics_gnn["rmse_tecu"] - metrics_slm["rmse_tecu"]
        rmse_pct = -delta_rmse / metrics_slm["rmse_tecu"] * 100.0
        print(f"\nGNN vs SLM: dRMSE = {delta_rmse:+.4f} TECU ({rmse_pct:+.2f}%)")
        print(f"  (negative dRMSE = GNN better; positive = SLM better)")

    # ---- 3l. Additional STEC metrics ----
    stec_rmse = float(np.sqrt(np.mean((slant_pred_forward_tcu - stec_obs_test) ** 2)))
    stec_r2 = float(r2_score(stec_obs_test, slant_pred_forward_tcu))
    print(f"\nGNN Slant STEC test: RMSE={stec_rmse:.4f} TECU, R2={stec_r2:.4f}")

    return {
        "stec_metrics": {"rmse_tecu": stec_rmse, "r2": stec_r2},
        "vtec_metrics": {
            "slm": metrics_slm,
            "gnn_ratio": metrics_gnn,
            "gnn_direct_vertical": metrics_gnn_direct,
        },
        "config": {
            "shell_height_km": shell_height_km,
            "vertical_top_km": vertical_top_km,
            "f107": f107,
            "ccir_or_ursi": ccir_or_ursi,
        },
        "data_info": {
            "n_test_rays": n_test,
            "iri_vtec_mean": float(np.mean(vtec_iri_ref)),
            "iri_vtec_std": float(np.std(vtec_iri_ref)),
        },
    }


# ============================================================
# 5. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate notebook-trained GNN on STEC-to-VTEC conversion"
    )
    p.add_argument("--data-path",
                   default="../00_iri/mycode/ML_sTEC/Mydata/obs_sTEC_2024128_1000_regional_big.npz")
    p.add_argument("--model-path",
                   default="results/checkpoints/gnn_notebook_repro_r00.pth")
    p.add_argument("--scaler-path",
                   default="results/checkpoints/gnn_notebook_repro_r00_scalers.pkl")
    p.add_argument("--run-name", default=None,
                   help="Run name (default: gnn_vtec_eval_YYYYMMDD_HHMMSS)")
    p.add_argument("--shell-height-km", type=float, default=450.0)
    p.add_argument("--vertical-top-km", type=float, default=2000.0)
    p.add_argument("--f107", type=float, default=204.0)
    p.add_argument("--ccir-or-ursi", type=int, default=0)
    p.add_argument("--knn-k", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--table-dir", default="results/tables")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--figure-dir", default="results/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if args.run_name is None:
        args.run_name = f"gnn_vtec_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(f"Run: {args.run_name}")

    os.makedirs(args.table_dir, exist_ok=True)

    # Resolve paths
    data_path = os.path.normpath(os.path.join(os.getcwd(), args.data_path))
    model_path = os.path.normpath(os.path.join(os.getcwd(), args.model_path))
    scaler_path = os.path.normpath(os.path.join(os.getcwd(), args.scaler_path))

    for label, p in [("data", data_path), ("model", model_path), ("scaler", scaler_path)]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    result = run_evaluation(
        data_path=data_path,
        model_path=model_path,
        scaler_path=scaler_path,
        device=device,
        knn_k=args.knn_k,
        shell_height_km=args.shell_height_km,
        vertical_top_km=args.vertical_top_km,
        f107=args.f107,
        ccir_or_ursi=args.ccir_or_ursi,
    )

    # Save metrics
    output = {
        "run_name": args.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path,
        "model_path": model_path,
    }
    output.update(result)
    metrics_path = os.path.join(args.table_dir, f"{args.run_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nMetrics saved to: {metrics_path}")

    # Optional scatter plot
    if not args.no_plot:
        _save_vtec_scatter(result, args.run_name, args.figure_dir)


def _save_vtec_scatter(result: dict, run_name: str, figure_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figure_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # We stored raw data arrays as part of result
    for ax, models, title in [
        (axes[0], ["slm", "gnn_ratio"], "SLM vs GNN (ratio-based)"),
        (axes[1], ["gnn_direct_vertical"], "GNN Direct Vertical Query"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("IRI VTEC Reference (TECU)")
        ax.set_ylabel("Estimated VTEC (TECU)")
        ax.grid(True)
    # Simplified: save the metrics figure
    fig.suptitle(f"GNN VTEC Evaluation - {run_name}")
    fig.savefig(os.path.join(figure_dir, f"{run_name}_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {figure_dir}/{run_name}_summary.png")


if __name__ == "__main__":
    main()
