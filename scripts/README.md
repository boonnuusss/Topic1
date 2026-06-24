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

## SLM Baseline

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\build_slm_baseline.ps1 `
  -VtecTable path\to\independent_vtec.parquet `
  -VtecSource IRI
```

The wrapper calls `python -m src.slm_baseline` and writes
`datasets/processed/raystec_china_20240505_slm.parquet` by default. A diagnostic
constant VTEC smoke test is available with:

```powershell
.\scripts\build_slm_baseline.ps1 `
  -OutputPath datasets\interim\slm_smoke.parquet `
  -VtecConstant 20 `
  -AllowDiagnosticConstant
```

## Direct ML Baselines

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
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\train_direct_baselines.ps1 `
  -Model random_forest `
  -Split random `
  -RunName direct_rf_random
```

Use `-MaxRows 5000` for quick smoke tests. Results are written under
`results/tables/` and `results/checkpoints/`.

## Residual ML Baselines

Train Ridge residual learning on an SLM table:

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
