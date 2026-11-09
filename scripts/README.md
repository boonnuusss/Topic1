## Paper Evaluation Protocol

Use `train_set_ratio_paper_protocol.ps1` for the current set-ratio paper
protocol. The historical UT05 key set is passed as validation, not test:

```powershell
.\scripts\train_set_ratio_paper_protocol.ps1
```

When an independent locked-test key file is available, pass it explicitly:

```powershell
.\scripts\train_set_ratio_paper_protocol.ps1 -InputPath datasets\interim\multiday_iri.parquet -LockedTestKeysFrom configs\splits\independent_day_test_keys.csv
```

Validation VTEC RMSE selects and saves the best checkpoint. Locked-test truth is
evaluated once only after that checkpoint has been restored. Runs without
`--validation-keys-from` retain the historical two-way behavior and are marked
`legacy_two_way` in metadata.

# Scripts

This directory contains thin wrappers around reproducible command-line
pipelines. Keep scientific logic in `src/`; keep scripts small and explicit.

## Quick Dataset Build

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\build_quick_dataset.ps1
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m src.dataset_builder --config configs/quick_test.yaml
```

When running from Codex, prefer the absolute interpreter path and request
escalated execution if the sandbox cannot launch the local base Python:

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe -m src.dataset_builder --config configs/quick_test.yaml
```

## Mainline SLM VTEC Conversion

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\build_slm_vtec.ps1
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m src.slm_vtec `
  --input datasets\processed\raystec_china_20240505.parquet `
  --output datasets\processed\rayvtec_china_20240505_slm.parquet `
  --shell-height-km 450
```

This produces the traditional STEC-to-VTEC conversion baseline:

```text
vtec_slm = stec_obs / slm_mapping_factor
```

Use `-MaxRows 1000` for a smoke test.

## Extended Physical Mapping Baselines

Evaluate the physical baseline family on the fixed UT05 validation keys:

```powershell
.\scripts\evaluate_physical_mapping_baselines.ps1
```

The wrapper uses the full-day 5 min IRI simulation for APMF fitting, excludes
the validation keys from that fit, and writes predictions, metrics, and full
method/coefficient metadata under `results/tables/`. SLM, CODE MSLM, and the
fixed multi-layer profile are truth-free. APMF is IRI-calibrated on training
data; the IRI ideal ratio is an oracle upper bound.

For NeQuick-G, the same wrapper also writes a STEC-only high-low CPP pair table
and evaluates its APMF. Match CPP calibration to the Set-ratio training window
with:

```powershell
.\scripts\evaluate_nequickg_physical_baselines.ps1 `
  -RunName nequickg_physical_mapping_baselines_ut05_validation_cpp_tw0400_0800 `
  -CppTrainWindowStart 2024-05-05T04:00:00 `
  -CppTrainWindowEnd 2024-05-05T08:00:00
```

## VTEC Diagnostic Plots

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\plot_vtec_diagnostics.ps1
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m src.plot_vtec_diagnostics `
  --input datasets\processed\rayvtec_china_20240505_slm.parquet `
  --figure-dir results\figures `
  --table-dir results\tables `
  --date-label 20240505
```

This creates map, elevation-bin, station-bin, and time-evolution diagnostics for
`vtec_slm`. If `-MapTime` is omitted, the map uses the data-rich epoch with the
highest median `vtec_slm`.

## IRI Simulation Builder

Start with a small smoke test:

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\build_iri_simulation.ps1 `
  -OutputPath datasets\interim\iri_simulation_smoke_3row.parquet `
  -MaxRows 3 `
  -MaxEpochs 1 `
  -RaySampleSpacingKm 50
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m src.iri_simulation_builder `
  --input datasets\processed\raystec_china_20240505.parquet `
  --output datasets\interim\iri_simulation_smoke_3row.parquet `
  --simulation-id iri_rayvtec_20240505_smoke_3row `
  --max-rows 3 `
  --max-epochs 1 `
  --ray-sample-spacing-km 50
```

The builder writes `stec_iri_ray`, `vtec_iri_ref`,
`vtec_slm_from_iri_stec`, and VTEC-space SLM error columns. It calls the
vendored PyIRI package under `../00_iri/PyIRI` and records all IRI/grid/ray
integration settings in metadata.

## NeQuick-G Simulation Builder

Install and verify the isolated PyPI environment, download the day-126 Galileo
navigation coefficients, then build the matching full-day simulation:

```powershell
.\scripts\install_nequick.ps1
.\scripts\verify_nequick_install.ps1
.\scripts\download_nequickg_navigation.ps1
.\scripts\build_nequickg_simulation.ps1
```

The builder writes one parquet part per 5 min epoch and reuses valid parts after
an interruption. It extracts only ray geometry from the IRI table and computes
new NeQuick-G STEC/VTEC targets between 90 and 2000 km.

Repeat the physical baseline matrix and create the cross-model table with:

```powershell
.\scripts\evaluate_nequickg_physical_baselines.ps1
.\.venv-nequick\Scripts\python.exe -m src.compare_physical_mapping_baselines
```

## VTEC-Space Metrics

Evaluate the SLM conversion baseline and optional ML-assisted columns against
the IRI VTEC reference:

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\evaluate_vtec_metrics.ps1 `
  -InputPath datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  -RunName vtec_eval_20240505_0500_spacing2p5
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --table-dir results\tables `
  --run-name vtec_eval_20240505_0500_spacing2p5
```

For future ML outputs, provide VTEC estimates with `-VtecPredictionColumns` or
STEC estimates with `-StecPredictionColumns`; STEC estimates are converted to
VTEC with `slm_mapping_factor_iri` before scoring.

## RayMLP Simulation Training

Train RayMLP as a STEC ray-transformation model on IRI simulation output:

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\train_ray_mlp_simulation.ps1 `
  -InputPath datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  -RunName ray_mlp_sim_20240505_0500_smoke `
  -Epochs 5 `
  -MaxRows 1000
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m src.train_ray_mlp_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name ray_mlp_sim_20240505_0500_smoke `
  --epochs 5 `
  --max-rows 1000
```

The model is trained only with simulated slant-ray STEC labels. VTEC is not used
as a training label; it is used after training to evaluate the model queried with
slant and vertical target rays. Main output columns are `stec_ray_mlp`,
`vtec_ray_mlp_direct`, `ray_mlp_conversion_ratio`, and the final converted
`vtec_ray_mlp`.

To reuse a previous test split, pass a prediction table containing the held-out
row keys:

```powershell
.\scripts\train_ray_mlp_simulation.ps1 `
  -InputPath datasets\interim\iri_simulation_20240505_0400_0600_spacing2p5_regular.parquet `
  -RunName ray_mlp_sim_0400_0600_e100_ut05_fixedtest `
  -TestKeysFrom results\tables\ray_mlp_sim_0500_random_e100_ratiofix_predictions.parquet `
  -TestKeyColumns timestamp,station,prn `
  -Epochs 100 `
  -BatchSize 128 `
  -Device cpu
```

For the current pair-learning route, use IPP-nearby ray pairs:

```powershell
.\scripts\run_nearby_pairs.ps1 `
  -RunName ray_mlp_sim_0500_nearby200_e100 `
  -NearbyKm 200 `
  -Epochs 100 `
  -Device cpu
```

Use `run_nearby_sweep.ps1` for the 100/200/400 km radius sweep. For the current best RayMLP route, pass `--target-mode source_ratio` to the Python entry point.

## Hybrid Low-Elevation VTEC

Build the current low-elevation ML / high-elevation SLM hybrid from a RayMLP
source-ratio prediction table:

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\build_hybrid_vtec.ps1 `
  -InputPath results\tables\ray_mlp_sim_0500_nearby200_e100_source_ratio_predictions.parquet `
  -OutputPath results\tables\ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid20_ml_lt_slm.parquet `
  -RunName ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid20_ml_lt_slm `
  -ElevationThresholdDeg 20 `
  -MlMustLowerSlm
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m src.hybrid_vtec `
  --input results\tables\ray_mlp_sim_0500_nearby200_e100_source_ratio_predictions.parquet `
  --output results\tables\ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid20_ml_lt_slm.parquet `
  --run-name ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid20_ml_lt_slm `
  --elevation-threshold-deg 20 `
  --ml-must-lower-slm
```

The default rule is conservative: use ML only below 20 deg, where the
low-elevation SLM path-mapping error is largest, and keep SLM elsewhere.
Add `-MlMustLowerSlm` or `--ml-must-lower-slm` for the current
recommended robustness gate. `-MlConstellations GPS` remains available for
constellation-filter diagnostics.

## Set-to-Query Simulation Training

Train the set-to-query model with nearby observable rays:

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\run_set_mlp.ps1 `
  -RunName set_mlp_0500_nearby200_s100_e100 `
  -NearbyKm 200 `
  -SpatialSigmaKm 100 `
  -Epochs 100 `
  -Device cpu
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m src.train_set_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name set_mlp_0500_nearby200_s100_e100 `
  --nearby-km 200 `
  --spatial-sigma-km 100 `
  --epochs 100 `
  --device cpu
```

Main output columns are `stec_set_mlp`, `vtec_set_mlp_direct`,
`set_mlp_conversion_ratio`, and final converted `vtec_set_mlp`. Use
`run_set_mlp_sweep.ps1` for the nearby radius and spatial-sigma sweep.

## FNN Simulation Training

Train the compact FNN adapted from
`../00_iri/mycode/ML_sTEC/FNN_sTEC_test.ipynb`:

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\train_fnn_simulation.ps1 `
  -InputPath datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  -RunName fnn_sim_0500_ecef6_e500_fixedtest `
  -FeatureSet ecef6 `
  -TestKeysFrom results\tables\ray_mlp_sim_0500_random_e100_ratiofix_predictions.parquet `
  -TestKeyColumns timestamp,station,prn `
  -Epochs 500 `
  -BatchSize 64 `
  -Device cpu
```

Supported feature sets:

- `ecef6`: `rec_x, rec_y, rec_z, sat_x, sat_y, sat_z`; notebook default.
- `blh6`: receiver/satellite lon, lat, height.
- `bl4`: receiver/satellite lon and lat only.

Main output columns are `stec_fnn`, `vtec_fnn_direct`,
`fnn_conversion_ratio`, and final converted `vtec_fnn`.

## GNN Simulation Training

Train the receiver/satellite graph edge regressor adapted from
`../00_iri/mycode/ML_sTEC/GNN_sTEC_improve.ipynb`:

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\train_gnn_simulation.ps1 `
  -InputPath datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  -RunName gnn_sim_0500_e500_fixedtest `
  -TestKeysFrom results\tables\ray_mlp_sim_0500_random_e100_ratiofix_predictions.parquet `
  -TestKeyColumns timestamp,station,prn `
  -Epochs 500 `
  -Device cpu
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m src.train_gnn_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name gnn_sim_0500_e500_fixedtest `
  --test-keys-from results\tables\ray_mlp_sim_0500_random_e100_ratiofix_predictions.parquet `
  --test-key-columns timestamp,station,prn `
  --epochs 500 `
  --device cpu
```

Main output columns are `stec_gnn`, `vtec_gnn_direct`,
`gnn_conversion_ratio`, and final converted `vtec_gnn`. The implementation is a
pure-PyTorch GraphSAGE-style model, so it does not require `torch_geometric`.

## Legacy Off-mainline SLM STEC Baseline

The older wrapper below predicts STEC from an external VTEC source:

```powershell
.\scripts\build_slm_baseline.ps1 `
  -VtecTable path\to\independent_vtec.parquet `
  -VtecSource IRI
```

It calls `python -m src.slm_baseline` and writes
`datasets/processed/raystec_china_20240505_slm.parquet` by default. This is not
the current mainline. A diagnostic constant VTEC smoke test is available with:

```powershell
.\scripts\build_slm_baseline.ps1 `
  -OutputPath datasets\interim\slm_smoke.parquet `
  -VtecConstant 20 `
  -AllowDiagnosticConstant
```

## Legacy Off-mainline Direct ML Baselines

Train Ridge:

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\train_direct_baselines.ps1 `
  -Model ridge `
  -Split random `
  -RunName direct_ridge_random
```

Train RandomForest:

```powershell
.\scripts\train_direct_baselines.ps1 `
  -Model random_forest `
  -Split random `
  -RunName direct_rf_random
```

Use `-MaxRows 5000` for quick smoke tests. These scripts predict `stec_obs`
directly and should not be treated as the current scientific endpoint.

## Legacy Off-mainline Residual ML Baselines

Train Ridge residual learning on a legacy SLM STEC table:

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\train_residual_baselines.ps1 `
  -Model ridge `
  -InputPath datasets\processed\raystec_china_20240505_slm.parquet `
  -Split random `
  -RunName residual_ridge_random
```

Train RandomForest residual learning:

```powershell
.\scripts\train_residual_baselines.ps1 `
  -Model random_forest `
  -InputPath datasets\processed\raystec_china_20240505_slm.parquet `
  -Split random `
  -RunName residual_rf_random
```

Train RayMLP residual learning:

```powershell
.\scripts\train_ray_mlp_residual.ps1 `
  -InputPath datasets\processed\raystec_china_20240505_slm.parquet `
  -Split random `
  -Epochs 30 `
  -RunName residual_ray_mlp_random
```

## NeQuick-G Set-Ratio Validation

Run the static density-matched NeQuick-G protocol with 192,000 fixed pairs and
25 training epochs:

```powershell
.\scripts\train_set_ratio_nequickg_paper_protocol.ps1
```

The wrapper uses UT04:00-08:00 for training, evaluates the fixed UT05
validation keys every five epochs, and explicitly selects the NeQuick-G STEC,
VTEC, SLM VTEC, and mapping-factor columns. It does not configure a locked test
partition, so its metrics are validation results.
