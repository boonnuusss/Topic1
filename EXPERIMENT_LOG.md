# Experiment Log

## 2026-06-12 Experiment 001: One-day Ray-wise Dataset Build

### Goal

Generate a standard ray-wise STEC dataset for a quick one-day test.

### Input Data

- TEC root: `E:/Scientific_Research/data/gnssTECmatFiles`
- Date: `2024-05-05`, DOY `126`
- Orbit file: `../00_iri/mycode/ML_sTEC/Mydata/WUM2024(126)0505.nc`

### Parameters

- Region: `80E-140E`, `0N-60N`
- Epoch interval: `900 s`
- Selection window: `30 s`
- Minimum elevation: `10 deg`
- STEC range: `0-600 TECU`

### Output

- `datasets/processed/raystec_china_20240505.parquet`
- `datasets/processed/metadata_raystec_china_20240505_quick.json`

### Observations

- Valid samples: `199087`
- Columns: `52`
- Effective stations: `112`
- PRNs: `100`
- Scanned station files: `133`

### Notes

- `station_files` is larger than `n_stations` because some station files do not
  contribute valid rows after region, elevation, STEC, and satellite-matching QC.

## 2026-06-12 Experiment 002: Dataset Diagnostic Figures

### Goal

Check whether processed IPP locations and observed STEC values are spatially
reasonable.

### Input Data

- `datasets/processed/raystec_china_20240505.parquet`

### Method

- Read processed parquet with pandas.
- Plot single-epoch IPP longitude/latitude on a Cartopy basemap.
- Color points by `stec_obs`; use robust percentile color limits.
- Plot `stec_obs` histogram for distribution sanity check.

### Output

- `results/figures/stec_ipp_scatter_20240505_1200.png`
- `results/figures/stec_ipp_scatter_20240505_1200.svg`
- `results/figures/stec_obs_distribution.png`
- `results/figures/stec_obs_distribution.svg`

### Notes

- Earlier `bbox_inches="tight"` with Cartopy clipped the map to the colorbar.
  Saving without tight bbox fixed the issue.

## 2026-06-23 Experiment 003: Agent-facing Project Organization

### Goal

Organize `Topic1` according to `agent_use_method.md` so future agents can
resume the project without relying on chat history.

### Changes

- Added long-term project brain files:
  `AGENT.md`, `RESEARCH_PLAN.md`, `DATA_DICTIONARY.md`, `TODO.md`,
  `FAILED_ATTEMPTS.md`, and this log.
- Added reproducible script wrapper documentation under `scripts/`.
- Added `papers/` and `manuscript/` holding areas.

### Remaining Concerns

- Training/evaluation scripts are still placeholders.
- IRI baseline is not yet integrated.
- Additional ordinary-day SP3/NetCDF files are not configured.

## 2026-06-23 Experiment 004: Narrow Project Scope

### Goal

Temporarily narrow `Topic1` to ordinary regional ray-wise STEC baseline
modeling and keep the project plan focused on baseline methodology.

### Changes

- Removed the former multi-day config file that no longer matched the current
  scope.
- Cleaned scope language in `README.md`, `RESEARCH_PLAN.md`, and `TODO.md`.
- Kept the existing one-day dataset and diagnostic figures unchanged.

### Notes

- The next scientific step remains baseline training and strict split
  evaluation on the existing one-day dataset.
- Multi-day processing can be reintroduced later as ordinary regional
  generalization, after the baseline system is stable.

## 2026-06-23 Experiment 005: Reprioritize Modeling Route

### Goal

Anchor `Topic1` to the traditional single-layer ionospheric projection model
before introducing direct ML or neural-network baselines.

### Changes

- Updated `README.md`, `RESEARCH_PLAN.md`, `TODO.md`, `DATA_DICTIONARY.md`,
  and `AGENT.md` to make SLM the first baseline.
- Defined the intended comparison order:
  SLM baseline, direct Ridge/RandomForest, SLM residual learning, then RayMLP.
- Added a reusable SLM helper module under `src/baselines.py`.

### Notes

- Existing processed data and figures were not regenerated.
- The next implementation step is an SLM baseline entry point that attaches an
  independent VTEC source and writes `stec_slm`.

## 2026-06-23 Experiment 006: Record Python Execution Convention

### Goal

Prevent future Codex sessions from misdiagnosing the project `.venv` as broken
when the sandbox cannot launch the local base Python interpreter.

### Changes

- Added a `Python Execution Convention` section to `AGENT.md`.
- Updated `README.md` to prefer the project `.venv` interpreter over a bare
  `python` command.
- Added a sandbox-specific pitfall and fix to `FAILED_ATTEMPTS.md`.

### Notes

- The successful smoke-test command was:
  `E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe --version`
- The result was `Python 3.12.3`.

## 2026-06-23 Experiment 007: SLM Baseline Entry Point

### Goal

Implement a reproducible command-line entry point that attaches independent
VTEC and produces the traditional single-layer STEC baseline.

### Changes

- Added `src.slm_baseline` for reading ray-wise tables, attaching VTEC,
  computing `stec_slm`, computing `stec_residual_slm`, and writing metadata.
- Updated `src.baselines` so SLM uses an explicit `slm_mapping_factor` computed
  for the requested shell height.
- Added `scripts/build_slm_baseline.ps1` as a PowerShell wrapper.
- Updated `README.md`, `DATA_DICTIONARY.md`, `TODO.md`, and `scripts/README.md`
  with the new workflow.

### Notes

- A real scientific run still needs an independent VTEC source, such as IRI or
  GIM-derived VTEC at each row's IPP/time.
- Constant VTEC mode exists only for CLI smoke testing and is explicitly marked
  diagnostic.
- Validation completed with Codex runtime Python:
  `python -m py_compile src/baselines.py src/slm_baseline.py src/features.py`,
  `python -m src.slm_baseline --help`, and a two-row CSV smoke test using
  diagnostic constant VTEC.
- Full parquet diagnostic smoke test later completed with the project `.venv`:
  `python -m src.slm_baseline --input datasets/processed/raystec_china_20240505.parquet --output datasets/interim/slm_full_smoke.parquet --metadata-output datasets/interim/slm_full_smoke.metadata.json --vtec-constant 20 --vtec-source diagnostic_constant_20_TECU --allow-diagnostic-constant`.
  It read and wrote all `199087` rows and produced all SLM columns. Because it
  used constant VTEC, the metrics are diagnostic only: RMSE `53.4299` TECU,
  MAE `37.7904` TECU, bias `-34.0537` TECU, R2 `-0.5164`.

## 2026-06-24 Experiment 008: Direct Ridge/RF Entry Points

### Goal

Implement direct-prediction tabular ML baselines for `stec_obs` before residual
learning.

### Changes

- Added `src.train_tree` with `ridge` and `random_forest` model choices.
- Added random, time, station, and PRN split support to the training CLI.
- Added metric, elevation-bin metric, prediction, and checkpoint outputs.
- Updated `src.models.tree_models` with sklearn pipelines using median
  imputation; Ridge also uses standard scaling.
- Added `scripts/train_direct_baselines.ps1` wrapper.

### Notes

- These are direct predictors of `stec_obs`; they do not use `stec_slm`.
- Strict split experiments still need to be run and compared systematically.
- Validation:
  - `python -m py_compile src/features.py src/models/tree_models.py src/train_tree.py`
    passed.
  - `python -m src.train_tree --help` passed.
  - Ridge smoke test on the first `5000` rows with random split passed:
    test RMSE `9.8038` TECU, MAE `5.9134` TECU, R2 `0.6293`.
  - RandomForest smoke test on the first `3000` rows with `10` trees passed:
    test RMSE `4.4243` TECU, MAE `2.6906` TECU, R2 `0.9092`.
  - Smoke-test metrics are not final scientific results because they use row
    caps and random splits.
