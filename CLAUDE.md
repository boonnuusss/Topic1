# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Reproducible research pipeline for using ray-wise GNSS STEC modeling to improve STEC-to-VTEC conversion. The central comparison is: can ML-assisted conversion beat the classical single-layer mapping (SLM) `VTEC_slm = STEC_obs / M(elevation)`?

Current benchmark: single-epoch UT05:00 from 2024-05-05, 80E-140E, 0N-60N, ~2041 rays, IRI-backed simulation with known reference VTEC.

## Python Environment

Always use the project `.venv` interpreter explicitly — the Codex sandbox cannot launch the user's local `C:\Users\12928\AppData\Local\Programs\Python\Python312\python.exe`:

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe -m src.<module> ...
```

Validate with: `E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe --version` → Python 3.12.3.

## Core Architecture

### Data flow

```
.mat TEC files + SP3 NetCDF orbits
  → src.dataset_builder → datasets/processed/raystec_china_20240505.parquet
  → src.slm_vtec → datasets/processed/rayvtec_china_20240505_slm.parquet
  → src.iri_simulation_builder → datasets/interim/iri_simulation_*.parquet
  → src.train_ray_mlp_simulation / src.train_set_mlp / ... → results/tables/*_predictions.parquet
  → src.evaluate → results/tables/*_vtec_*_metrics.csv
```

### Module roles

| Module | Role |
|--------|------|
| `src/baselines.py` | SLM helpers: `add_slm_vtec_from_stec` (mainline) and `add_single_layer_baseline` (legacy direction). Always check which direction you need. |
| `src/geometry.py` | Geodetic↔ECEF transforms (cached pyproj), ray length, slant factor `M(elevation)`, azimuth/elevation trig encodings. |
| `src/features.py` | Time cyclicals (hour, DOY, IPP local time), constellation one-hot, `BASE_FEATURE_COLUMNS` list, `build_feature_matrix()`. |
| `src/metrics.py` | VTEC-space metrics: `vtec_error_metrics()` (convention: `error = prediction - reference`), `vtec_group_metrics()`, `elevation_bin_metrics()`. |
| `src/split.py` | `random_split`, `time_split`, `station_holdout_split`, `prn_holdout_split`. Random split is smoke-test only — scientific claims need time/station/PRN holdouts. |
| `src/models/mlp.py` | `build_ray_mlp()` (128→128→64→1) and `build_fnn_stec()` (64→32→1). **No Softplus on output** — incompatible with StandardScaler zero-mean targets. Use `np.maximum(pred, 0)` clamp after inverse transform. |
| `src/models/set_mlp.py` | Set-to-query: QueryEncoder, ObsEncoder, SpatialCrossAttention (4-head + IPP distance bias `-D/σ`), OutputHead. Permutation-invariant. |
| `src/models/gnn.py` | Pure-PyTorch GraphSAGE edge regressor. Receiver/satellite nodes, ray edges with elevation/azimuth, receiver-receiver KNN. No `torch_geometric` dependency. |
| `src/models/tree_models.py` | sklearn Ridge (median impute + StandardScaler) and RandomForest (median impute only). |

### Current mainline entry points

| Entry point | Purpose |
|-------------|---------|
| `src.dataset_builder` | Build ray-wise STEC parquet from .mat + SP3 |
| `src.slm_vtec` | Classical `VTEC_slm = STEC_obs / M(elevation)` |
| `src.iri_simulation_builder` | IRI-backed simulation: `stec_iri_ray`, `vtec_iri_ref`, `vtec_slm_from_iri_stec` |
| `src.evaluate` | Compare VTEC predictions against reference; outputs overall, elevation-bin, station, UTC-hour metrics |
| `src.train_ray_mlp_simulation` | **Leading ML route**: RayMLP pairs (source→target STEC), supports `--nearby-km`, `--target-mode source_ratio` |
| `src.train_set_mlp` | Set-to-query with nearby observation sets and spatial cross-attention |
| `src.hybrid_vtec` | Low-elevation ML + high-elevation SLM hybrid from source-ratio prediction table |
| `src.train_fnn_simulation` | Compact FNN from legacy notebook (ecef6/blh6/bl4) |
| `src.train_gnn_simulation` | GraphSAGE edge regressor from legacy notebook |
| `src.plot_vtec_diagnostics` | Cartopy spatial maps, elevation/station/time diagnostics |

### Legacy off-mainline (do not treat as current route)

- `src.slm_baseline` — produces `stec_slm = M(elevation) * VTEC` (wrong direction for Topic1)
- `src.train_tree` — direct/residual Ridge/RF STEC prediction
- `src.train_mlp` — RayMLP residual STEC prediction

## Critical Design Rules

1. **`vtec_iri_ref` is evaluation-only.** It must never be used as a training label or for model/hybrid selection. Training labels are always simulated slant-ray STEC.

2. **STEC-to-VTEC conversion via ratio.** All models predict slant STEC; conversion uses the model-implied ratio:
   ```
   conversion_ratio = vtec_<model>_direct / stec_<model>
   vtec_<model> = source_STEC * conversion_ratio
   ```

3. **Identity-only pair training is a known failure.** The model must see non-identity source→target ray pairs to learn cross-geometry conversion. IPP-nearby 200 km filtering is the current standard.

4. **No Softplus in model output.** StandardScaler normalizes targets to zero-mean; `softplus(x) > 0` prevents predicting values below the mean. Clamp with `np.maximum(pred, 0)` after inverse standardization instead.

5. **Evaluation convention:** `error = prediction - reference`. Positive bias = overestimation. Negative `delta_rmse_vs_slm_tecu` or positive `rmse_reduction_vs_slm_pct` = improvement over SLM.

6. **Azimuth convention:** Parquet columns use `[0, 360)`; GNN wraps to `[-180, 180)` before radians to match legacy notebook's `atan2` convention.

## Key Hard-coded Paths

- TEC data: `E:/Scientific_Research/data/gnssTECmatFiles`
- Orbit files: `../00_iri/mycode/ML_sTEC/Mydata/WUM2024(126)0505.nc`
- PyIRI: vendored under `../00_iri/PyIRI`
- Reference notebooks: `../00_iri/mycode/ML_sTEC/`

## PRN Mapping

```
GPS     → PRN
Galileo → PRN + 70
SBAS    → PRN + 119
QZSS    → PRN + 180
GLONASS → PRN + 500
BeiDou  → PRN + 700
```

## Current Benchmark Status (UT05:00 random split)

| Route | VTEC RMSE | R² |
|-------|----------:|----:|
| SLM baseline | 2.44 | 0.961 |
| **Hybrid source-ratio <30° + SLM** | **1.99** | **0.974** |
| RayMLP source-ratio nearby200 e100 | 2.13 | 0.970 |
| Set-to-query nearby200 σ100 e100 | 2.40 | 0.962 |
| RayMLP identity-only | 24.08 | −2.80 |

The hybrid is the current best result. Next scientific gate: time/station/PRN holdout splits.

## Reproducibility Rules

- Never overwrite raw TEC `.mat`, SP3, NetCDF, PyIRI, or model artifact files.
- Run small one-day tests before multi-day processing.
- Record every experiment in `EXPERIMENT_LOG.md`; update `TODO.md` on status changes.
- Distinguish observation, inference, and speculation. Mark uncertain statements `[UNCERTAIN]`.
- Prefer reusable modules (`src/`) and thin scripts (`scripts/`) over large notebooks.
- Keep scientific assumptions explicit: units, coordinate systems, shell height, mapping formula, time basis.
- Consult `FAILED_ATTEMPTS.md` before implementing something that may have been tried before.
