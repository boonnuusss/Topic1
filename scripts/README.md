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
