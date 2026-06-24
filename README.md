# Topic1 Ray-wise STEC Pipeline

`Topic1` is the reproducible research pipeline for ray-wise GNSS STEC
modeling. Each sample is one receiver-satellite ray at one epoch:

```text
station + PRN + timestamp -> observed STEC
```

The current goal is to build a clean, auditable dataset and then compare a
traditional single-layer ionosphere baseline, direct machine-learning models,
and residual-learning models for regional STEC prediction.

## Modeling Route

The modeling route is intentionally ordered from physical reference to more
flexible statistical models:

1. Build a traditional single-layer mapping baseline:
   `STEC_slm = M(elevation) * VTEC(IPP, time)`.
2. Evaluate direct machine-learning models, such as Ridge and RandomForest, for
   predicting `stec_obs` from ray-wise geometry, IPP, and time features.
3. Evaluate residual-learning models, where ML predicts
   `stec_obs - stec_slm` and the final prediction is
   `stec_slm + residual_pred`.
4. Add RayMLP after the physical baseline and tabular baselines are stable.

This order keeps the scientific comparison anchored to the classical
thin-shell ionospheric projection model before introducing more flexible ML
models.

## Agent Project Brain

New coding agents should read these files before making changes:

- `AGENT.md`: long-term rules for scientific coding agents.
- `RESEARCH_PLAN.md`: scientific question, hypotheses, datasets, and risks.
- `DATA_DICTIONARY.md`: processed table schema, units, and provenance.
- `TODO.md`: current milestone and executable task list.
- `EXPERIMENT_LOG.md`: reproducibility log for data builds and figures.
- `FAILED_ATTEMPTS.md`: known pitfalls and fixes.

## Layout

- `src/`: reusable Python modules and command-line entry points.
- `configs/`: experiment and dataset-building configuration files.
- `scripts/`: small command-line wrappers for reproducible runs.
- `datasets/interim/`: intermediate files produced during construction.
- `datasets/processed/`: standard ray-wise tables and metadata.
- `results/tables/`: metric tables.
- `results/figures/`: paper-ready and diagnostic figures.
- `results/checkpoints/`: trained models and scalers.
- `notebooks/`: lightweight exploration and visualization notebooks.
- `papers/`: paper PDFs, notes, and citation material.
- `manuscript/`: manuscript drafts, outlines, and response material.

## Environment

Activate the project environment:

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\activate_topic1.ps1
```

Or call it directly:

```powershell
.\.venv\Scripts\python.exe -m src.dataset_builder --config configs/quick_test.yaml
```

For Codex agents, prefer the absolute interpreter path when running validation
commands:

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe
```

If Codex reports that it cannot start the local base interpreter under
`C:\Users\12928\AppData\Local\Programs\Python\Python312`, request escalated
execution for the project `.venv` interpreter. This is a sandbox permission
issue, not evidence that the virtual environment is broken.

## Dataset Builder

The core entry point is:

```powershell
.\.venv\Scripts\python.exe -m src.dataset_builder --config configs/quick_test.yaml
```

The current quick dataset is:

```text
datasets/processed/raystec_china_20240505.parquet
datasets/processed/metadata_raystec_china_20240505_quick.json
```

It contains 199,087 valid ray-wise samples from 2024-05-05 over
80E-140E, 0N-60N, after basic quality control.

## SLM Baseline

The single-layer mapping entry point is:

```powershell
.\.venv\Scripts\python.exe -m src.slm_baseline `
  --input datasets/processed/raystec_china_20240505.parquet `
  --output datasets/processed/raystec_china_20240505_slm.parquet `
  --vtec-table path\to\independent_vtec.parquet `
  --vtec-table-column vtec_baseline `
  --vtec-source IRI `
  --shell-height-km 450
```

The command writes `vtec_baseline`, `vtec_source`, `shell_height_km`,
`slm_mapping_factor`, `stec_slm`, and `stec_residual_slm`, plus a metadata JSON
next to the output table. The VTEC table is expected to provide row-wise
independent VTEC values, merged by `timestamp station prn` unless
`--vtec-merge-keys` is supplied.

For smoke tests only, a constant VTEC can be used with
`--vtec-constant <TECU> --allow-diagnostic-constant`. Do not use that diagnostic
mode for scientific baseline results.

## Direct ML Baselines

Ridge and RandomForest direct-prediction baselines are trained with:

```powershell
.\.venv\Scripts\python.exe -m src.train_tree `
  --input datasets/processed/raystec_china_20240505.parquet `
  --model ridge `
  --split random `
  --run-name direct_ridge_random
```

```powershell
.\.venv\Scripts\python.exe -m src.train_tree `
  --input datasets/processed/raystec_china_20240505.parquet `
  --model random_forest `
  --split random `
  --rf-n-estimators 300 `
  --run-name direct_rf_random
```

Supported split modes are `random`, `time`, `station`, and `prn`. The entry
point writes metrics, elevation-bin metrics, test-set predictions, and a
checkpoint under `results/`.

## Residual ML Baselines

After an SLM table exists, Ridge/RandomForest can learn the SLM residual:

```powershell
.\.venv\Scripts\python.exe -m src.train_tree `
  --mode residual `
  --input datasets/processed/raystec_china_20240505_slm.parquet `
  --model ridge `
  --split random `
  --run-name residual_ridge_random
```

The residual model target is:

```text
stec_residual_slm = stec_obs - stec_slm
```

The final prediction evaluated against `stec_obs` is:

```text
predicted STEC = stec_slm + predicted residual
```

RayMLP residual learning uses:

```powershell
.\.venv\Scripts\python.exe -m src.train_mlp `
  --input datasets/processed/raystec_china_20240505_slm.parquet `
  --split random `
  --epochs 30 `
  --run-name residual_ray_mlp_random
```

## Reproducibility Rules

- Do not overwrite raw TEC, SP3, IRI, or model artifact files.
- Run small one-day tests before any larger multi-day regional processing.
- Record every experiment in `EXPERIMENT_LOG.md`.
- Update `TODO.md` when task status changes.
- Distinguish observation, inference, and speculation in scientific notes.
