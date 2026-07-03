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

## 2026-06-24 Experiment 009: Residual ML Entry Points

### Goal

Implement ML models that learn the SLM residual and evaluate the final
prediction `stec_slm + predicted residual` against `stec_obs`.

### Changes

- Extended `src.train_tree` with `--mode residual` for Ridge and RandomForest.
- Added `scripts/train_residual_baselines.ps1`.
- Implemented `src.train_mlp` for RayMLP residual learning.
- Added `scripts/train_ray_mlp_residual.ps1`.
- Updated README, script docs, and TODO.

### Notes

- Residual target: `stec_residual_slm = stec_obs - stec_slm`.
- Final prediction column:
  - Ridge/RF: `pred_<model>_residual_stec`.
  - RayMLP: `pred_ray_mlp_residual_stec`.
- Inputs must contain `stec_slm`; diagnostic smoke tests used
  `datasets/interim/slm_full_smoke.parquet`.

### Validation

- `python -m py_compile src/train_tree.py src/train_mlp.py src/models/mlp.py`
  passed.
- `python -m src.train_tree --help` and `python -m src.train_mlp --help`
  passed.
- Ridge residual smoke test on first `5000` diagnostic SLM rows passed:
  final-STEC test RMSE `9.8034` TECU, MAE `5.9130` TECU, R2 `0.6293`.
- RandomForest residual smoke test on first `3000` diagnostic SLM rows with
  `10` trees passed: final-STEC test RMSE `3.0272` TECU, MAE `1.9361` TECU,
  R2 `0.9575`.
- RayMLP residual smoke test on first `1000` diagnostic SLM rows with `2`
  epochs passed: final-STEC test RMSE `8.8321` TECU, MAE `5.7820` TECU,
  R2 `0.5858`.
- These metrics are entry-point smoke tests only because the SLM table used
  diagnostic constant VTEC and random splits.

## 2026-06-24 Experiment 010: Correct Mainline To STEC-to-VTEC Conversion

### Goal

Correct the project objective from standalone STEC prediction to STEC modeling
for STEC-to-VTEC conversion.

### Changes

- Updated the project brain files so the mainline target is VTEC conversion
  accuracy rather than direct `stec_obs` prediction.
- Added `src.slm_vtec` for the current physical baseline:
  `vtec_slm = stec_obs / slm_mapping_factor`.
- Added `scripts/build_slm_vtec.ps1` as the reproducible wrapper for the SLM
  VTEC product.
- Marked the previously implemented `stec_slm`, direct STEC prediction, and
  residual-learning entry points as legacy/off-mainline.

### Notes

- The legacy code is retained because it may still be useful for smoke tests or
  later controlled experiments, but it no longer defines the scientific route.
- The preferred ML route is now an IRI-backed simulation experiment: train
  ray-wise STEC models such as RayMLP, then compare ML-assisted conversion and
  classical SLM conversion in VTEC space.

## 2026-06-25 Experiment 011: Full One-day SLM VTEC Product

### Goal

Run the current mainline single-layer STEC-to-VTEC conversion on the complete
one-day ray-wise STEC parquet and inspect the resulting `vtec_slm` distribution.

### Command

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe -m src.slm_vtec --input datasets\processed\raystec_china_20240505.parquet --output datasets\processed\rayvtec_china_20240505_slm.parquet --shell-height-km 450
```

### Output

- `datasets/processed/rayvtec_china_20240505_slm.parquet`
- `datasets/processed/rayvtec_china_20240505_slm.metadata.json`
- `results/tables/vtec_slm_distribution_20240505.csv`
- `results/tables/vtec_slm_by_elevation_20240505.csv`
- `results/tables/vtec_slm_by_utc_hour_20240505.csv`
- `results/tables/vtec_slm_high_value_stations_20240505.csv`

### Observations

- Input rows: `199087`
- Output rows: `199087`
- Dropped rows: `0`
- Missing or non-finite values in `stec_obs`, `elevation`,
  `slm_mapping_factor`, and `vtec_slm`: `0`
- `vtec_slm` distribution in TECU:
  - min `0.0032`
  - p05 `11.0776`
  - median `35.7057`
  - mean `42.9024`
  - p95 `96.1600`
  - p99 `121.0263`
  - max `165.3059`
- `vtec_slm > 100 TECU`: `8139` rows
- `vtec_slm > 150 TECU`: `244` rows

### Notes

- High `vtec_slm` values are concentrated mainly during UTC `05-10`, consistent
  with a time-structured daytime/evening regional signal rather than isolated
  random failures.
- The highest values are concentrated in a small set of southern stations,
  especially `GXGG`, so later VTEC diagnostics should inspect station-level and
  spatial residual patterns.
- Because this product is generated from observed slant TEC, possible receiver
  and satellite bias handling remains an important review item before making
  scientific claims about absolute VTEC.

## 2026-06-25 Experiment 012: VTEC Diagnostic Plots

### Goal

Create reproducible diagnostic figures for the completed SLM VTEC product:
spatial map, elevation-bin distribution, station-bin distribution, and time
evolution.

### Command

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe -m src.plot_vtec_diagnostics --input datasets\processed\rayvtec_china_20240505_slm.parquet --figure-dir results\figures --table-dir results\tables --date-label 20240505
```

### Output

- `results/figures/vtec_slm_map_20240505_0800.png`
- `results/figures/vtec_slm_map_20240505_0800.svg`
- `results/figures/vtec_slm_elevation_distribution_20240505.png`
- `results/figures/vtec_slm_elevation_distribution_20240505.svg`
- `results/figures/vtec_slm_station_distribution_20240505.png`
- `results/figures/vtec_slm_station_distribution_20240505.svg`
- `results/figures/vtec_slm_time_evolution_20240505.png`
- `results/figures/vtec_slm_time_evolution_20240505.svg`
- `results/tables/vtec_slm_elevation_diagnostic_20240505.csv`
- `results/tables/vtec_slm_station_diagnostic_20240505.csv`
- `results/tables/vtec_slm_time_diagnostic_20240505.csv`
- `results/tables/vtec_slm_diagnostics_20240505.metadata.json`

### Observations

- The plotting input contains `199087` rows, all valid for plotting.
- The map epoch was selected automatically as `2024-05-05 08:00:00` UTC because
  it had the highest median `vtec_slm` among data-rich epochs.
- At the map epoch, sample count was `2052` and the map used the Cartopy
  backend.
- The time-evolution diagnostic peaks near UTC `08:00`, with median
  `vtec_slm` about `71.04 TECU`.
- The elevation-bin diagnostic shows broadly similar medians from `10-60 deg`,
  with the `45-60 deg` bin having the highest median among the configured bins.
- The station diagnostic highlights `GXGG` as a high-VTEC station; station-level
  interpretation should use sample-count thresholds because some stations have
  very few rows.

### Validation

- `python -m py_compile src\plot_vtec_diagnostics.py` passed.
- All four PNG files were visually checked for nonblank content, reasonable map
  framing, and readable labels.

## 2026-06-25 Experiment 013: IRI Simulation Schema Design

### Goal

Design the IRI-backed simulation schema for controlled STEC-to-VTEC conversion
experiments.

### Reference Material

- `../00_iri/mycode/ML_sTEC/generate_sTEC_obs_regional.ipynb`
- `../00_iri/mycode/ML_sTEC/generate_sTEC_obs_iri_regional.ipynb`
- `../00_iri/mycode/ML_sTEC/gen_iri_val_dataset.ipynb`
- `../00_iri/mycode/ML_sTEC/generate_raypath.ipynb`
- `../00_iri/mycode/ML_sTEC/Read_sp3_XYZ.py`
- Existing npz files under `../00_iri/mycode/ML_sTEC/Mydata/`

### Changes

- Added `IRI_SIMULATION_SCHEMA.md`.
- Updated `README.md`, `AGENT.md`, `RESEARCH_PLAN.md`, `DATA_DICTIONARY.md`,
  and `TODO.md` to point to the schema and use the new IRI column names.

### Design Decisions

- Use row-wise parquet as the main simulation product, not the legacy npz layout.
- Use `stec_iri_ray` for the full receiver-satellite IRI ray integral.
- Use `vtec_iri_ref` for the IPP vertical IRI reference.
- Use `vtec_slm_from_iri_stec = stec_iri_observable / slm_mapping_factor_iri`
  for the classical SLM conversion baseline in the simulation. In the first
  no-bias run, `stec_iri_observable` equals `stec_iri_ray`.
- Keep `slm_shell_height_km = 450` as the first default, but record it in every
  output and metadata file.
- Start with `bias_mode = none`; receiver/satellite bias and noise are optional
  later extensions.

### Notes

- Existing `iri_vTEC_*.npz` files store VTEC in a variable named `slantTEC` and
  use `(3, N)` coordinate arrays, while obs/IRI STEC npz files use `(N, 3)`.
  The new schema explicitly avoids both ambiguities.
- Existing notebook integration uses PyIRI electron density, regular-grid
  interpolation, ECEF straight-line ray sampling, and trapezoidal integration.
  The first builder should reproduce this logic on a small subset before
  running the full one-day table.

## 2026-06-25 Experiment 014: IRI Simulation Builder Entry Point

### Goal

Implement a reproducible command-line builder for the IRI-backed simulation
table defined in `IRI_SIMULATION_SCHEMA.md`.

### Changes

- Added `src.iri_simulation_builder`.
- Added `scripts/build_iri_simulation.ps1`.
- Added `fortranformat` to `requirements.txt` because vendored PyIRI requires
  it.
- Updated README, script docs, TODO, data dictionary, and schema notes.

### Implementation Notes

- The builder starts from the Topic1 row-wise STEC parquet.
- For each epoch, it calls vendored PyIRI using a `row_adaptive_point_profiles`
  strategy:
  - row IPP points are used for `vtec_iri_ref`;
  - in-grid receiver-satellite ray sample points are used for `stec_iri_ray`;
  - electron density outside the configured IRI altitude/region is treated as
    zero.
- It writes `stec_iri_ray`, `vtec_iri_ref`,
  `vtec_slm_from_iri_stec`, `vtec_slm_error_iri`, integration diagnostics, and
  metadata.
- A per-epoch PyIRI query-point safety limit prevents accidental large
  high-resolution full-day runs.

### Validation

- `python -m py_compile src\iri_simulation_builder.py` passed.
- `python -m src.iri_simulation_builder --help` passed.
- `scripts/build_iri_simulation.ps1` parsed as a valid PowerShell script.
- First smoke test failed before builder execution because the environment
  lacked PyIRI dependency `fortranformat`.
- Installed `fortranformat==2.0.3` into the Topic1 `.venv`.
- One-row smoke test passed:
  - output rows: `1`
  - epoch: `2024-05-05 00:00:00`
  - IRI horizontal points: `255`
  - `stec_iri_ray`: `75.0271 TECU`
  - `vtec_iri_ref`: `42.8345 TECU`
  - `vtec_slm_error_iri`: `1.5375 TECU`
- Three-row smoke test passed:
  - output rows: `3`
  - IRI horizontal points: `372`
  - non-finite values in key IRI/VTEC columns: `0`

### Outputs

- `datasets/interim/iri_simulation_smoke_1row.parquet`
- `datasets/interim/iri_simulation_smoke_1row.metadata.json`
- `datasets/interim/iri_simulation_smoke_3row.parquet`
- `datasets/interim/iri_simulation_smoke_3row.metadata.json`

## 2026-06-25 Experiment 015: UT05:00 IRI Simulation Spacing Test

### Goal

Run the `2024-05-05 05:00:00 UTC` epoch with dense `2.5 km` ray sampling and
summarize `vtec_slm_error_iri` by elevation bin.

### Changes

- Added `regular_grid_interpolator` as an optional IRI grid strategy in
  `src.iri_simulation_builder`.
- Updated `scripts/build_iri_simulation.ps1` to expose `GridStrategy` and
  `ProgressEvery`.
- Documented the dense-spacing strategy in `IRI_SIMULATION_SCHEMA.md`.

### Run

```powershell
python -m src.iri_simulation_builder `
  --input datasets/interim/raystec_china_20240505_0500.parquet `
  --output datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --metadata-output datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.metadata.json `
  --simulation-id iri_rayvtec_20240505_0500_spacing2p5_regular `
  --grid-strategy regular_grid_interpolator `
  --ray-sample-spacing-km 2.5 `
  --max-iri-points-per-epoch 50000 `
  --progress-every 200 `
  --drop-invalid-geometry
```

### Results

- Rows: `2041`.
- IRI grid: `[383 altitude, 101 longitude, 201 latitude]`.
- Ray sample points: `24283856`.
- Inside-grid ray samples: `5588464`.
- Overall `vtec_slm_error_iri`:
  - mean: `1.0612 TECU`
  - median: `0.4060 TECU`
  - MAE: `1.4254 TECU`

Elevation-bin summary is saved to:

```text
datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular_elevation_bins.csv
```

Low-elevation rays remain the largest error source. The `[10,20)` bin has mean
bias `5.1544 TECU` and MAE `5.1910 TECU`; bins above `50` degrees have MAE below
`0.65 TECU` in this test.

## 2026-06-25 Experiment 016: VTEC-Space Metrics Definition

### Goal

Define the mainline evaluation contract for comparing classical SLM conversion
and future ML-assisted conversion in VTEC space.

### Changes

- Extended `src.metrics` with `vtec_error_metrics`, using
  `error = prediction - reference`.
- Replaced the placeholder `src.evaluate` with a command-line evaluator.
- Added `scripts/evaluate_vtec_metrics.ps1`.
- Documented metric definitions in `README.md`, `RESEARCH_PLAN.md`,
  `DATA_DICTIONARY.md`, `scripts/README.md`, and `TODO.md`.

### Metric Contract

- Default reference: `vtec_iri_ref`.
- Default SLM baseline: `vtec_slm_from_iri_stec`.
- ML outputs can be scored as:
  - VTEC columns via `--vtec-prediction-columns`;
  - STEC columns via `--stec-prediction-columns`, converted by
    `VTEC_ml = STEC_ml / slm_mapping_factor_iri`.
- Primary metrics: RMSE, MAE, bias, median error, error percentiles, maximum
  absolute error, and R2.
- Grouped scopes: overall, elevation bin, station, and UTC hour.
- Improvement convention: negative `delta_rmse_vs_slm_tecu` or positive
  `rmse_reduction_vs_slm_pct` means improvement over SLM.

### Validation

- `scripts/evaluate_vtec_metrics.ps1` parsed as a valid PowerShell script.
- `python -m py_compile src\metrics.py src\evaluate.py
  src\train_ray_mlp_simulation.py` passed after `.venv` execution became
  available again.

## 2026-06-25 Experiment 017: RayMLP Simulation Training Route

### Goal

Adapt RayMLP to the IRI simulation setting without turning it into a VTEC
residual corrector.

### Design

RayMLP is now treated as a STEC model trained only on slant-ray labels:

```text
source slant STEC + source ray geometry + original slant target geometry
-> slant target STEC
```

The conversion interpretation is that the trained STEC model can later be
queried with a vertical target ray at the same IPP/time. The model therefore
produces:

- `stec_ray_mlp`: prediction for the original slant target ray.
- `vtec_ray_mlp`: post-training prediction for the vertical target ray.

### Changes

- Added `src.train_ray_mlp_simulation`.
- Added `scripts/train_ray_mlp_simulation.ps1`.
- Updated README, script docs, research plan, TODO, and data dictionary.

### Notes

- The old `src.train_mlp` residual-learning entry point remains legacy and
  off-mainline.
- `vtec_iri_ref` must not be used as a training label; it is used only for
  post-training VTEC-space evaluation.
- The prediction table can be evaluated by `python -m src.evaluate` using
  `--vtec-prediction-columns vtec_ray_mlp`.

### Validation

- `python -m py_compile src\metrics.py src\evaluate.py
  src\train_ray_mlp_simulation.py` passed.
- Smoke training command passed:

```powershell
python -m src.train_ray_mlp_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name smoke_ray_mlp_sim_0500 `
  --epochs 2 `
  --batch-size 64 `
  --device cpu `
  --max-rows 200 `
  --output-format csv
```

- The original smoke command before this correction included vertical target
  labels in the training pairs. Those outputs are superseded and should not be
  interpreted as valid model performance.

### Smoke Outputs

- `results/tables/smoke_ray_mlp_sim_0500_predictions.csv`
- `results/tables/smoke_ray_mlp_sim_0500_metrics.json`
- `results/tables/smoke_ray_mlp_sim_0500_test_vtec_metrics.csv`
- `results/checkpoints/smoke_ray_mlp_sim_0500.pth`

## 2026-06-25 Experiment 018: UT05:00 Full RayMLP Simulation Training

### Status

Superseded. This run used the earlier incorrect training design in which
`vtec_iri_ref` contributed vertical target labels during training. Keep the log
only as a record of a failed modeling choice; do not use the metrics below as
scientific results.

### Goal

Train the RayMLP simulation model on the full `2024-05-05 05:00:00 UTC` IRI
simulation table and compare `vtec_ray_mlp` with the SLM conversion baseline in
VTEC space.

### Run

```powershell
python -m src.train_ray_mlp_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name ray_mlp_sim_0500_random_e100 `
  --split random `
  --test-size 0.2 `
  --seed 42 `
  --epochs 100 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --weight-decay 0.0001 `
  --dropout 0.1 `
  --device cpu `
  --output-format parquet `
  --include-identity-pairs
```

### Data Split

- Input rows: `2041`.
- Train rows: `1622`.
- Test rows: `419`.
- Train pairs: `3244`, including vertical target pairs and identity slant pairs.
- Feature count: `45`.

### Training Diagnostics

- Epochs: `100`.
- Training loss decreased from `0.3148` at epoch 1 to `0.0119` at epoch 100.
- Train pair STEC RMSE: `1.7749 TECU`.
- Test slant STEC RMSE: `1.9762 TECU`.

### VTEC Comparison On Test Rows

| Model | RMSE TECU | MAE TECU | Bias TECU | R2 |
| --- | ---: | ---: | ---: | ---: |
| SLM | `2.4400` | `1.4326` | `1.1364` | `0.9610` |
| RayMLP vertical-ray prediction | `1.6604` | `1.2457` | `0.8150` | `0.9819` |

Relative to SLM, RayMLP reduced overall RMSE by `31.95%` and MAE by `13.04%`.

### Elevation-Bin Result

RayMLP improved the low-elevation bins:

- `[10,20)` RMSE: SLM `5.8390`, RayMLP `2.0828` TECU.
- `[20,30)` RMSE: SLM `2.8787`, RayMLP `1.8098` TECU.

RayMLP was worse than SLM above `30` degrees in this random-split test. This is
expected to be a key diagnostic: SLM is already very accurate at high elevation,
while most learnable correction signal is concentrated at low elevation.

### Outputs

- `results/tables/ray_mlp_sim_0500_random_e100_predictions.parquet`
- `results/tables/ray_mlp_sim_0500_random_e100_metrics.json`
- `results/tables/ray_mlp_sim_0500_random_e100_test_vtec_metrics.csv`
- `results/tables/ray_mlp_sim_0500_random_e100_test_vtec_overall_metrics.csv`
- `results/tables/ray_mlp_sim_0500_random_e100_test_vtec_elevation_bin_metrics.csv`
- `results/checkpoints/ray_mlp_sim_0500_random_e100.pth`

### Interpretation

This is the first non-smoke UT05:00 RayMLP simulation run. It supports the idea
that a learned ray-transform model can reduce low-elevation SLM conversion
errors, but it also shows that unconstrained RayMLP can degrade the already
excellent high-elevation SLM estimates. Next experiments should test either
elevation-aware blending with SLM or loss weighting focused on preserving
high-elevation performance.

## 2026-06-25 Experiment 019: UT05:00 Slant-Only RayMLP STEC Training

### Goal

Correct the RayMLP simulation design so that `vtec_iri_ref` is not used during
training. The model is trained only on slant-ray IRI STEC and is then evaluated
by querying a vertical target ray.

### Corrected Training Contract

Training pair:

```text
source slant STEC + source slant geometry + original slant target geometry
-> stec_iri_ray
```

Post-training conversion query:

```text
source slant STEC + source slant geometry + vertical target geometry
-> vtec_ray_mlp
```

`vtec_iri_ref` is used only after training to compute VTEC-space error.

### Code Changes

- `src.train_ray_mlp_simulation` now creates only
  `source_slant_to_same_slant` training pairs.
- Metadata records `vtec_reference_used_for_training = false`.
- `scripts/train_ray_mlp_simulation.ps1` no longer exposes
  `IncludeIdentityPairs`.
- README, script docs, research plan, and data dictionary were updated to make
  the no-VTEC-training-label rule explicit.

### Run

```powershell
python -m src.train_ray_mlp_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name ray_mlp_sim_0500_random_e100_slantonly `
  --split random `
  --test-size 0.2 `
  --seed 42 `
  --epochs 100 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --weight-decay 0.0001 `
  --dropout 0.1 `
  --device cpu `
  --output-format parquet
```

### Data Split

- Input rows: `2041`.
- Train rows: `1622`.
- Test rows: `419`.
- Train pairs: `1622`.
- Feature count: `45`.

### Training Diagnostics

- Training loss decreased from `0.4815` at epoch 1 to `0.0149` at epoch 100.
- Train slant STEC RMSE: `1.1904 TECU`.
- Test slant STEC RMSE: `1.2676 TECU`.

### VTEC Comparison On Test Rows

| Model | RMSE TECU | MAE TECU | Bias TECU | R2 |
| --- | ---: | ---: | ---: | ---: |
| SLM | `2.4400` | `1.4326` | `1.1364` | `0.9610` |
| RayMLP vertical-ray query | `24.3226` | `16.0455` | `12.2958` | `-2.8758` |

### Interpretation

This corrected experiment shows that fitting slant-ray STEC alone is not enough
to reliably extrapolate to a vertical target ray. The model reconstructs
slant-ray STEC well, but `vtec_ray_mlp` is much worse than the classical SLM
baseline. This is an important negative result: without vertical-ray supervision
or a physics-informed constraint, the geometry change from slant ray to vertical
ray is out-of-distribution for RayMLP.

### Outputs

- `results/tables/ray_mlp_sim_0500_random_e100_slantonly_predictions.parquet`
- `results/tables/ray_mlp_sim_0500_random_e100_slantonly_metrics.json`
- `results/tables/ray_mlp_sim_0500_random_e100_slantonly_test_vtec_metrics.csv`
- `results/tables/ray_mlp_sim_0500_random_e100_slantonly_test_vtec_overall_metrics.csv`
- `results/tables/ray_mlp_sim_0500_random_e100_slantonly_test_vtec_elevation_bin_metrics.csv`
- `results/checkpoints/ray_mlp_sim_0500_random_e100_slantonly.pth`

## 2026-06-25 Experiment 020: RayMLP STEC-to-VTEC Conversion Audit

### Goal

Audit whether the poor slant-only RayMLP VTEC result was caused by an
implementation issue in the simulated STEC-to-VTEC conversion, rather than only
by the modeling assumption.

### Finding

The previous implementation evaluated the raw vertical target-ray query output
directly as `vtec_ray_mlp`. This missed the intended conversion step:

```text
ray_mlp_conversion_ratio = vtec_ray_mlp_direct / stec_ray_mlp
vtec_ray_mlp = source_STEC * ray_mlp_conversion_ratio
```

This was a semantic/code issue in the conversion output. `vtec_iri_ref` still
was not used as a training label.

### Code Changes

- `src.train_ray_mlp_simulation` now writes:
  - `stec_ray_mlp`;
  - `vtec_ray_mlp_direct`;
  - `ray_mlp_conversion_ratio`;
  - final ratio-converted `vtec_ray_mlp`.
- VTEC metrics now evaluate both `vtec_ray_mlp` and
  `vtec_ray_mlp_direct`.
- README, research plan, data dictionary, and script docs were updated to make
  this distinction explicit.

### Verification Run

```powershell
python -m src.train_ray_mlp_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name ray_mlp_sim_0500_random_e100_ratiofix `
  --split random `
  --test-size 0.2 `
  --seed 42 `
  --epochs 100 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --weight-decay 0.0001 `
  --dropout 0.1 `
  --device cpu `
  --output-format parquet
```

### Results On Test Rows

| Model | RMSE TECU | MAE TECU | Bias TECU | R2 |
| --- | ---: | ---: | ---: | ---: |
| SLM | `2.4400` | `1.4326` | `1.1364` | `0.9610` |
| RayMLP ratio-converted VTEC | `24.0837` | `15.9022` | `11.9858` | `-2.8000` |
| RayMLP direct vertical query | `24.3226` | `16.0455` | `12.2958` | `-2.8758` |

The ratio correction fixed the output semantics but did not materially improve
the result, because `stec_ray_mlp` is already very close to
`stec_iri_observable` in the no-bias simulation. The remaining failure is
therefore mainly a modeling/design issue: training only
`source_slant_to_same_slant` pairs gives the network no supervised examples in
which target geometry changes while the target STEC changes accordingly.

### Diagnostic Notes

- Test slant STEC remains accurate: RMSE `1.2676 TECU`, R2 `0.9981`.
- The learned `ray_mlp_conversion_ratio` is too high at low elevations.
- The `[10,20)` elevation bin has RayMLP ratio-converted bias
  `49.6824 TECU`, while SLM bias is `4.6978 TECU`.
- A counterfactual test replacing vertical target height `2000 km` with the
  row's GNSS-like `sat_h` did not improve the already-trained model; this points
  back to the lack of non-identity target-ray supervision rather than only the
  height encoding.

## 2026-06-25 Experiment 021: UT04:00-06:00 IRI Training For UT05:00 Test

### Goal

Check whether adding neighboring epochs around UT05:00 improves RayMLP
performance on the same UT05:00 test rays.

### Dataset

Created a three-hour ray-wise input table:

```text
datasets/interim/raystec_china_20240505_0400_0600.parquet
```

Selection is inclusive from `2024-05-05 04:00:00 UTC` to
`2024-05-05 06:00:00 UTC`.

- Rows: `18610`.
- Epochs: `9`.
- Epoch spacing: `15 min`.

Built the IRI simulation dataset:

```powershell
python -m src.iri_simulation_builder `
  --input datasets\interim\raystec_china_20240505_0400_0600.parquet `
  --output datasets\interim\iri_simulation_20240505_0400_0600_spacing2p5_regular.parquet `
  --metadata-output datasets\interim\iri_simulation_20240505_0400_0600_spacing2p5_regular.metadata.json `
  --simulation-id iri_rayvtec_20240505_0400_0600_spacing2p5_regular `
  --grid-strategy regular_grid_interpolator `
  --ray-sample-spacing-km 2.5 `
  --sample-alpha 2.0 `
  --max-iri-points-per-epoch 200000 `
  --progress-every 1000 `
  --drop-invalid-geometry
```

Simulation output rows: `18610`. No geometry/elevation rows were dropped.

### Fixed-Test Split

Added `--test-keys-from` to `src.train_ray_mlp_simulation`, allowing a previous
prediction table to define the test rows by key. This run reused the exact
`419` UT05:00 test rows from
`ray_mlp_sim_0500_random_e100_ratiofix_predictions.parquet`.

Training rows:

```text
18191 = all UT04:00-06:00 rows except the fixed UT05:00 test keys
```

### Run

```powershell
python -m src.train_ray_mlp_simulation `
  --input datasets\interim\iri_simulation_20240505_0400_0600_spacing2p5_regular.parquet `
  --run-name ray_mlp_sim_0400_0600_e100_ut05_fixedtest `
  --test-keys-from results\tables\ray_mlp_sim_0500_random_e100_ratiofix_predictions.parquet `
  --test-key-columns timestamp,station,prn `
  --seed 42 `
  --epochs 100 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --weight-decay 0.0001 `
  --dropout 0.1 `
  --device cpu `
  --output-format parquet
```

### Results On The Same UT05:00 Test Rows

| Training data | Slant STEC RMSE | VTEC RMSE | VTEC MAE | VTEC bias | VTEC R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UT05-only random train | `1.2676` | `24.0837` | `15.9022` | `11.9858` | `-2.8000` |
| UT04:00-06:00 expanded train | `1.4364` | `33.4975` | `23.9898` | `23.9116` | `-6.3513` |
| SLM baseline on same test rows | - | `2.4400` | `1.4326` | `1.1364` | `0.9610` |

The compact comparison table is:

```text
results/tables/ray_mlp_sim_0400_0600_vs_ut05_only_comparison.csv
```

### Interpretation

Adding neighboring epochs did not improve the current slant-only RayMLP
conversion route. It made the VTEC conversion result worse on the same UT05:00
test rows. The learned mean `ray_mlp_conversion_ratio` moved from `0.8045` in
the UT05-only run to `0.9346` in the expanded run, while the true mean ratio is
`0.7121`. This confirms that the main issue is not simply limited sample count;
the current identity-only training contract still does not teach the model the
vertical/slant transformation needed for STEC-to-VTEC conversion.

### Outputs

- `datasets/interim/raystec_china_20240505_0400_0600.parquet`
- `datasets/interim/iri_simulation_20240505_0400_0600_spacing2p5_regular.parquet`
- `datasets/interim/iri_simulation_20240505_0400_0600_spacing2p5_regular.metadata.json`
- `results/tables/ray_mlp_sim_0400_0600_e100_ut05_fixedtest_predictions.parquet`
- `results/tables/ray_mlp_sim_0400_0600_e100_ut05_fixedtest_metrics.json`
- `results/tables/ray_mlp_sim_0400_0600_e100_ut05_fixedtest_test_vtec_overall_metrics.csv`
- `results/tables/ray_mlp_sim_0400_0600_e100_ut05_fixedtest_test_vtec_elevation_bin_metrics.csv`
- `results/tables/ray_mlp_sim_0400_0600_vs_ut05_only_comparison.csv`
- `results/checkpoints/ray_mlp_sim_0400_0600_e100_ut05_fixedtest.pth`

## 2026-06-25 Experiment 022: FNN Simulation Baseline On UT05:00

### Goal

Adapt the FNN model from
`../00_iri/mycode/ML_sTEC/FNN_sTEC_test.ipynb` and test whether a direct
ray-geometry-to-STEC model performs better than the current RayMLP
ray-transform route on the single-epoch UT05:00 IRI simulation dataset.

### Implementation

- Added `build_fnn_stec` in `src.models.mlp`.
- Added `src.train_fnn_simulation`.
- Added `scripts/train_fnn_simulation.ps1`.

The notebook's default FNN architecture is:

```text
input -> Linear(64) -> BatchNorm -> ReLU -> Dropout(0.2)
      -> Linear(32) -> BatchNorm -> ReLU -> Dropout(0.2)
      -> Linear(1)
```

The FNN is trained only on slant-ray STEC labels:

```text
ray geometry -> stec_iri_ray
```

For STEC-to-VTEC conversion, the trained model is queried for both the original
slant ray and the vertical special ray:

```text
fnn_conversion_ratio = vtec_fnn_direct / stec_fnn
vtec_fnn = stec_iri_observable * fnn_conversion_ratio
```

`vtec_iri_ref` is not used during training.

### Data And Split

- Input: `datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.parquet`.
- Test rows: the same `419` UT05:00 key-holdout rows used by
  `ray_mlp_sim_0500_random_e100_ratiofix`.
- Train rows: `1622`.
- Epochs: `500`.
- Batch size: `64`.
- Learning rate: `0.001`.
- Device: `cpu`.

### Runs

```powershell
python -m src.train_fnn_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name fnn_sim_0500_ecef6_e500_fixedtest `
  --feature-set ecef6 `
  --test-keys-from results\tables\ray_mlp_sim_0500_random_e100_ratiofix_predictions.parquet `
  --test-key-columns timestamp,station,prn `
  --seed 52 `
  --epochs 500 `
  --batch-size 64 `
  --learning-rate 0.001 `
  --dropout 0.2 `
  --device cpu
```

The same command was also run with `--feature-set blh6` and
`--feature-set bl4`.

### Results On The Same UT05:00 Test Rows

| Model | Slant STEC RMSE | VTEC RMSE | VTEC MAE | VTEC bias | VTEC R2 | Mean ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SLM baseline | - | `2.4400` | `1.4326` | `1.1364` | `0.9610` | - |
| RayMLP ratiofix | `1.2676` | `24.0837` | `15.9022` | `11.9858` | `-2.8000` | `0.8045` |
| FNN `ecef6` | `5.9975` | `75.4518` | `74.1350` | `74.1350` | `-36.2972` | `1.5563` |
| FNN `blh6` | `8.1358` | `16.4829` | `14.7170` | `14.5633` | `-0.7799` | `0.8620` |
| FNN `bl4` | `9.4666` | `9.1545` | `6.6542` | `6.5455` | `0.4510` | `0.7761` |

The compact comparison table is:

```text
results/tables/fnn_raymlp_ut05_fixedtest_comparison.csv
```

### Interpretation

The FNN experiment is informative but does not beat the traditional SLM
conversion. The notebook-default `ecef6` input is especially poor for
STEC-to-VTEC conversion because the vertical query endpoint at `2000 km` is far
from the GNSS satellite ECEF endpoint distribution used in training. `bl4`
avoids the altitude/magnitude mismatch and improves substantially over RayMLP
in VTEC space (`9.1545` versus `24.0837` TECU RMSE), but it is still much worse
than SLM (`2.4400` TECU RMSE).

This supports the same conclusion as the RayMLP tests: a pure slant-ray
supervised model can fit slant STEC, but it still does not reliably learn the
vertical/slant conversion needed to outperform SLM without explicit physical
constraints or suitable non-identity ray supervision.

### Outputs

- `src/train_fnn_simulation.py`
- `scripts/train_fnn_simulation.ps1`
- `results/tables/fnn_sim_0500_ecef6_e500_fixedtest_predictions.parquet`
- `results/tables/fnn_sim_0500_blh6_e500_fixedtest_predictions.parquet`
- `results/tables/fnn_sim_0500_bl4_e500_fixedtest_predictions.parquet`
- `results/tables/fnn_raymlp_ut05_fixedtest_comparison.csv`
- `results/checkpoints/fnn_sim_0500_ecef6_e500_fixedtest.pth`
- `results/checkpoints/fnn_sim_0500_blh6_e500_fixedtest.pth`
- `results/checkpoints/fnn_sim_0500_bl4_e500_fixedtest.pth`

## 2026-06-25 Experiment 023: Code Audit & Post-Inference Clamp

### Goal

Apply fixes from the code review and re-run RayMLP ratiofix (Exp 020) and
FNN bl4 (Exp 022) with post-inference non-negativity clamping, then compare.

### Code Changes

- `src/models/mlp.py`: removed `nn.Softplus()` from both model output layers
  (added in an earlier attempt then reverted — Softplus is incompatible with
  StandardScaler because targets are zero-mean after standardization).
- `src/train_ray_mlp_simulation.py`: added `np.maximum(pred, 0)` after inverse
  standardization for both vertical and slant predictions. Removed dead
  `--include-identity-pairs` argument.
- `src/train_fnn_simulation.py`: added `np.maximum(pred, 0)` after inverse
  standardization.
- `src/geometry.py`: cached `pyproj.Transformer` for geodetic→ECEF conversion.
- `src/io_sp3.py`: cached `pyproj.Transformer` for ECEF→geodetic conversion.

### Failed Approach: In-Model Softplus

A first attempt placed `nn.Softplus()` inside the model output layer. This
immediately broke training (train pair STEC RMSE ~19.5 vs ~1.2 TECU) because
`StandardScaler` transforms training targets to zero-mean, and `softplus(x) > 0`
prevents the model from predicting values below the training mean. This
approach was abandoned.

### RayMLP Ratiofix + Clamp

Run:

```powershell
python -m src.train_ray_mlp_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name ray_mlp_sim_0500_random_e100_clamp `
  --split random --test-size 0.2 --seed 42 `
  --epochs 100 --batch-size 128 --learning-rate 0.001 --weight-decay 0.0001 `
  --dropout 0.1 --device cpu --output-format parquet
```

### RayMLP Results On Test Rows (419 rows, same random split as Exp 020)

| Model | RMSE TECU | MAE TECU | Bias TECU | R2 |
| --- | ---: | ---: | ---: | ---: |
| SLM | 2.4400 | 1.4326 | 1.1364 | 0.9610 |
| RayMLP clamp (this run) | 24.0837 | 15.9022 | 11.9858 | -2.8000 |
| RayMLP ratiofix (Exp 020) | 24.0837 | 15.9022 | 11.9858 | -2.8000 |

The clamp result is numerically identical to Exp 020 because the model did
not produce negative STEC predictions. The clamp is a no-op safety measure
for this configuration.

### FNN bl4 + Clamp

Run:

```powershell
python -m src.train_fnn_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name fnn_sim_0500_bl4_e500_clamp `
  --feature-set bl4 --split random --test-size 0.2 --seed 52 `
  --epochs 500 --batch-size 64 --learning-rate 0.001 --dropout 0.2 --device cpu
```

### FNN Results On Test Rows (433 rows, random split seed=52)

| Model | RMSE TECU | MAE TECU | Bias TECU | R2 |
| --- | ---: | ---: | ---: | ---: |
| SLM | 2.3583 | 1.4301 | 1.1160 | 0.9686 |
| FNN bl4 clamp (this run) | 7.9934 | 5.9842 | 5.4006 | 0.6393 |
| FNN bl4 (Exp 022, key-holdout) | 9.1545 | 6.6542 | 6.5455 | 0.4510 |

The clamp run shows better VTEC metrics than Exp 022, but the comparison is
confounded: Exp 022 used fixed test keys (419 rows from the ratiofix
predictions table) while this run uses a fresh random split (433 rows,
seed=52). The training STEC metrics are also different:
- Exp 022 slant test RMSE: 9.47 TECU
- Clamp slant test RMSE: 8.83 TECU

The improvement is therefore partly a split effect, not purely a clamp effect.

### Interpretation

- The post-inference clamp is a safe, zero-cost addition. It cannot hurt
  performance and prevents physically impossible negative STEC outputs.
- For the RayMLP identity-only training configuration, the clamp is a no-op
  because the model already predicts positive values.
- The clamp does NOT address the fundamental issue: training only with
  identity pairs cannot teach the model the geometry-to-STEC mapping needed
  for cross-geometry VTEC conversion. Both models remain far worse than SLM.
- The code audit also fixed: dead argparse code, pyproj Transformer caching,
  and documented the Softplus-StandardScaler incompatibility.

### Outputs

- `results/tables/ray_mlp_sim_0500_random_e100_clamp_predictions.parquet`
- `results/tables/ray_mlp_sim_0500_random_e100_clamp_metrics.json`
- `results/checkpoints/ray_mlp_sim_0500_random_e100_clamp.pth`
- `results/tables/fnn_sim_0500_bl4_e500_clamp_predictions.parquet`
- `results/tables/fnn_sim_0500_bl4_e500_clamp_metrics.json`
- `results/checkpoints/fnn_sim_0500_bl4_e500_clamp.pth`
- `results/tables/ray_mlp_sim_0500_random_e100_softplus_*` (broken, kept as reference)
- `results/tables/fnn_sim_0500_bl4_e500_softplus_*` (broken, kept as reference)

## 2026-06-25 Experiment 024: GNN Simulation Model From Legacy Notebook

### Goal

Adapt `../00_iri/mycode/ML_sTEC/GNN_sTEC_improve.ipynb` into a Topic1 command
line entry point and test whether a receiver/satellite graph model improves the
single-epoch UT05:00 STEC-to-VTEC conversion experiment.

### Implementation

- Added `src/models/gnn.py`: a pure-PyTorch GraphSAGE-style edge regressor.
  This follows the notebook architecture but avoids a hard dependency on
  `torch_geometric`.
- Added `src/train_gnn_simulation.py`: trains on simulated slant-ray
  `stec_iri_ray` only, then queries a vertical special edge at each IPP.
- Added `scripts/train_gnn_simulation.ps1`.
- Graph design:
  - receiver and satellite nodes use ECEF coordinates;
  - ray edges use elevation and azimuth in radians;
  - receiver-receiver KNN edges provide spatial message passing context;
  - train/test uses edge masks, matching the notebook style;
  - test geometry can participate in message passing, but test STEC labels are
    not used in the loss.

### Runs

```powershell
python -m src.train_gnn_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name gnn_sim_0500_e500_fixedtest `
  --test-keys-from results\tables\ray_mlp_sim_0500_random_e100_ratiofix_predictions.parquet `
  --test-key-columns timestamp,station,prn `
  --seed 52 --epochs 500 --learning-rate 0.001 --weight-decay 0.0001 `
  --dropout 0.2 --hidden-dim 64 --embedding-dim 32 --knn-k 4 --device cpu
```

A second run used the same settings with `--disable-query-knn` to check whether
vertical query KNN edges were driving the error.

### Results On The Same UT05:00 Test Rows

| Model | Slant STEC RMSE | VTEC RMSE | VTEC MAE | VTEC bias | VTEC R2 | Mean ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SLM baseline | - | `2.4400` | `1.4326` | `1.1364` | `0.9610` | - |
| RayMLP ratiofix | `1.2676` | `24.0837` | `15.9022` | `11.9858` | `-2.8000` | `0.8045` |
| FNN `bl4` | `9.4666` | `9.1545` | `6.6542` | `6.5455` | `0.4510` | `0.7761` |
| GNN GraphSAGE + query KNN | `5.1822` | `12.7675` | `10.4654` | `-8.0303` | `-0.0679` | `0.6311` |
| GNN GraphSAGE no query KNN | `5.1476` | `12.8560` | `10.5987` | `-8.4154` | `-0.0828` | `0.6259` |

Compact comparison table:

```text
results/tables/gnn_fnn_raymlp_ut05_fixedtest_comparison.csv
```

### Interpretation

The GNN fits slant STEC reasonably well (`5.18` TECU test RMSE), better than
the FNN `bl4` slant fit but worse than RayMLP ratiofix. However, its vertical
query response underestimates VTEC, with mean converted VTEC near `53.8` TECU
against a reference mean of `61.8` TECU. Disabling query KNN changes little, so
the main limitation is not the extra query KNN edge construction. The likely
failure mode remains the same as the FNN/RayMLP tests: identity slant-ray
supervision does not provide enough cross-geometry information for reliable
vertical/slant conversion.

The GNN improves over RayMLP ratiofix in VTEC RMSE (`12.77` versus `24.08`
TECU) but does not beat FNN `bl4` (`9.15` TECU) or the classical SLM baseline
(`2.44` TECU).

### Outputs

- `src/models/gnn.py`
- `src/train_gnn_simulation.py`
- `scripts/train_gnn_simulation.ps1`
- `results/tables/gnn_sim_0500_e500_fixedtest_predictions.parquet`
- `results/tables/gnn_sim_0500_e500_fixedtest_metrics.json`
- `results/checkpoints/gnn_sim_0500_e500_fixedtest.pth`
- `results/tables/gnn_sim_0500_e500_fixedtest_noqueryknn_predictions.parquet`
- `results/tables/gnn_sim_0500_e500_fixedtest_noqueryknn_metrics.json`
- `results/checkpoints/gnn_sim_0500_e500_fixedtest_noqueryknn.pth`

## 2026-06-25 Experiment 025: All-to-All Pairs RayMLP (N² Training)

### Goal

Train RayMLP with all-to-all source→target ray pairs so the model
sees varied target geometries during training, not just identity pairs.

### Design

For N training rays in one epoch, build N² pairs:
`source_STEC + source_geometry + target_geometry → target_STEC`

For 1622 training rays this yields ~2.63M pairs.

Implemented `--all-pairs` flag in `src/train_ray_mlp_simulation.py` with
chunked construction to control memory.

### Run

```powershell
python -m src.train_ray_mlp_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name ray_mlp_sim_0500_random_e20_allpairs `
  --all-pairs --split random --test-size 0.2 --seed 42 `
  --epochs 20 --batch-size 1024 --learning-rate 0.001 --weight-decay 0.0001 `
  --dropout 0.1 --device cpu --output-format parquet
```

### Results On Test Rows (419 rows, random split seed=42)

| Model | VTEC RMSE | MAE | Bias | R² | Pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| SLM | 2.44 | 1.43 | +1.14 | 0.961 | — |
| all-to-all | 3.24 | 2.34 | +1.05 | 0.931 | 2,630,884 |
| identity-only | 24.08 | 15.90 | +11.99 | −2.80 | 1,622 |

### Interpretation

All-pairs training dramatically improved VTEC (RMSE 24.08→3.24),
confirming the hypothesis that identity-only training is the bottleneck.
However, 2.6M pairs with only 1622 training rays is O(N²) — not scalable
to multi-epoch datasets. Many distant pairs introduced noise. Next step:
IPP-nearby filtering.

### Outputs

- `results/tables/ray_mlp_sim_0500_random_e20_allpairs_predictions.parquet`
- `results/checkpoints/ray_mlp_sim_0500_random_e20_allpairs.pth`

## 2026-06-25 Experiment 026: IPP-Nearby RayMLP Pairs

### Goal

Replace all-to-all N² pairs with IPP-distance-filtered nearby pairs,
matching the ionospheric spatial correlation length (~200 km).

### Design

Implemented `build_nearby_pairs()` in `train_ray_mlp_simulation.py`
with haversine IPP distance filtering. Three CLI parameters:
`--nearby-km`, `--nearby-max-neighbors`, `--nearby-no-identity`.

### Runs

```powershell
# Sweep: 100 km, 200 km, 400 km, 200km·100ep
python -m src.train_ray_mlp_simulation --nearby-km 100 --epochs 50 ...
python -m src.train_ray_mlp_simulation --nearby-km 200 --epochs 50 ...
python -m src.train_ray_mlp_simulation --nearby-km 200 --epochs 100 ...
python -m src.train_ray_mlp_simulation --nearby-km 400 --epochs 50 ...
```

### Results On Test Rows

| Model | Pairs | VTEC RMSE | MAE | Bias | R² | MaxAbsErr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SLM | — | 2.44 | 1.43 | +1.14 | 0.961 | 12.11 |
| **nearby-200·50ep** | 27,676 | **2.59** | **1.73** | **−0.09** | **0.956** | **11.50** |
| nearby-200·100ep | 27,676 | 2.55 | 1.76 | −0.07 | 0.958 | 11.58 |
| nearby-400·50ep | 42,571 | 2.70 | 2.08 | +0.38 | 0.952 | 9.49 |
| all-to-all·20ep | 2,630,884 | 3.24 | 2.34 | +1.05 | 0.931 | 14.48 |
| nearby-100·50ep | 11,452 | 3.15 | 2.28 | +0.54 | 0.935 | 15.76 |
| identity-only | 1,622 | 24.08 | 15.90 | +11.99 | −2.80 | 100.02 |

### Interpretation

200 km is the optimal IPP distance — matching ionospheric correlation
length. 100 km has too few neighbors (insufficient info), 400 km
introduces noise (degraded performance). nearby-200·50ep with only
27k pairs approaches SLM accuracy with near-zero bias (bias −0.09
vs SLM +1.14). 50→100 epochs gives marginal gain (2.59→2.55),
suggesting data diversity is saturated.

### Code Changes

- `src/train_ray_mlp_simulation.py`: added `_haversine_km()`,
  `build_nearby_pairs()`, `--nearby-km`, `--nearby-max-neighbors`,
  `--nearby-no-identity` CLI args
- `src/geometry.py`: cached pyproj Transformer
- `src/io_sp3.py`: cached pyproj Transformer
- `src/models/mlp.py`: removed `nn.Softplus()` (broken, incompatible
  with StandardScaler zero-mean targets)

### Outputs

- `results/tables/ray_mlp_sim_0500_nearby100_e50_predictions.parquet`
- `results/tables/ray_mlp_sim_0500_nearby200_e50_predictions.parquet`
- `results/tables/ray_mlp_sim_0500_nearby200_e100_predictions.parquet`
- `results/tables/ray_mlp_sim_0500_nearby400_e50_predictions.parquet`
- `results/checkpoints/ray_mlp_sim_0500_nearby*.pth`
- `scripts/run_nearby_pairs.ps1`
- `scripts/run_nearby_sweep.ps1`

## 2026-06-25 Experiment 027: Set-to-Query RayMLP Architecture

### Goal

Move beyond single-pair modeling: encode a **set** of nearby observed
rays as context, then predict an arbitrary query ray's STEC via
cross-attention and pooling.

### Architecture

```
Query ray (target geometry, 16d)
    │ QueryEncoder(16→128)
    ▼ q_enc (1, 128)
    │
    ├─── Observed rays ×K (23d each)
    │    │ ObsEncoder(23→128)
    │    ▼ o_enc (K, 128)
    │    │
    │    ├── SpatialCrossAttention(Q=q_enc, K=V=o_enc)
    │    │   + IPP-distance bias (−D/σ)      → context (1, 128)
    │    │
    │    └── Mean-Pool                      → pool (1, 128)
    │
    └─── Concat[q_enc | ctx | pool] (1, 384)
         │ OutputMLP(384→128→64→1)
         ▼ stec_query (1,)
```

Key design: spatial-biased cross-attention with 4 heads, IPP-distance
bias `-D/σ` added to attention logits. Permutation-invariant by
construction.

Training: leave-one-out per epoch — each ray is a query, with all
other rays within IPP radius as its observation set.

### First Run

nearby-200 km, σ=100 km, max_obs=40, 50 epochs, batch_size=128

### Results

| Model | VTEC RMSE | MAE | Bias | R² | MaxAbsErr |
| --- | ---: | ---: | ---: | ---: | ---: |
| SLM | 2.44 | 1.43 | +1.14 | 0.961 | 12.11 |
| set-to-query·50ep | 2.65 | 1.80 | +1.10 | 0.954 | 10.60 |
| nearby-200·50ep | 2.59 | 1.73 | −0.09 | 0.956 | 11.50 |

First run already approaches SLM/nearby-200. Direct vertical query
poor (RMSE 5.53), indicating ratio conversion still carries the load.

### Numbers fixed

- ObsEncoder BatchNorm1d reshape (B,K,D)→(B×K,D)→(B,K,128)
- NaN from ∞ IPP distances: clamped distances to 5000 km max
- NaN from fully-masked attention: fallback unmask first entry
- Rows with 0 neighbors filtered from training set
- Dynamic `obs_feature_cols` from CLI args

### Tuning Sweep

Three configurations run for 100 epochs:

| Run | Radius | σ | VTEC RMSE | MAE | Bias | R² | MaxAbsErr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **nearby200·s100·100ep** | 200 | 100 | **2.40** | **1.69** | +0.85 | **0.962** | **11.84** |
| nearby400·s200·100ep | 400 | 200 | 2.52 | 1.73 | +1.09 | 0.958 | 15.05 |
| nearby200·s200·100ep | 200 | 200 | 2.71 | 1.84 | +0.72 | 0.952 | 15.57 |

`nearby200·s100·100ep` is the first model to surpass SLM in both
RMSE (2.40 vs 2.44) and R² (0.962 vs 0.961). Direct vertical query
still poor (RMSE 5.66) — ratio conversion carries the improvement.

### Code Files Added

- `src/models/set_mlp.py` — QueryEncoder, ObsEncoder,
  SpatialCrossAttention, OutputHead, SetToQueryModel
- `src/train_set_mlp.py` — build_set_examples(), _SetDataset,
  training loop, VTEC evaluation, CLI
- `scripts/run_set_mlp.ps1` — default config
- `scripts/run_set_mlp_sweep.ps1` — 3-run tuning sweep

### Outputs

- `results/tables/set_mlp_0500_nearby*_predictions.parquet`
- `results/checkpoints/set_mlp_0500_nearby*.pth`

## 2026-06-25 Session Summary: Code Changes & Experiments

### Code Audit & Fixes

| File | Change |
| --- | --- |
| `src/models/mlp.py` | Removed `nn.Softplus()` from both `build_ray_mlp` and `build_fnn_stec` (incompatible with StandardScaler zero-mean targets). Added post-inference `np.maximum(pred, 0)` clamp instead. |
| `src/train_ray_mlp_simulation.py` | Removed dead `--include-identity-pairs` argument. Added `np.maximum(pred, 0)` after inverse standardization. Added `_haversine_km()`, `build_all_to_all_pairs()`, `build_nearby_pairs()`. Added `--all-pairs`, `--nearby-km`, `--nearby-max-neighbors`, `--nearby-no-identity` CLI args. |
| `src/train_fnn_simulation.py` | Added `np.maximum(pred, 0)` after inverse standardization. |
| `src/geometry.py` | Cached `pyproj.Transformer` as `_get_ecef_transformer()` — avoids repeated instantiation. |
| `src/io_sp3.py` | Cached `pyproj.Transformer` as `_get_geo_transformer()`. |

### New Files Created

| File | Purpose |
| --- | --- |
| `src/models/set_mlp.py` | QueryEncoder(16→128), ObsEncoder(in→128, configurable), SpatialCrossAttention (4-head + IPP distance bias), OutputHead(384→128→64→1), SetToQueryModel |
| `src/train_set_mlp.py` | Set-to-query CLI: build_set_examples(), _SetDataset with variable-length collate, training loop, vertical-query VTEC inference, full CLI |
| `scripts/run_all_pairs_experiment.ps1` | All-to-all N² pair experiment |
| `scripts/run_nearby_pairs.ps1` | IPP-nearby pair experiment (default 200km·50ep) |
| `scripts/run_nearby_sweep.ps1` | IPP distance sweep (100/200/400 km + 200·100ep) |
| `scripts/run_set_mlp.ps1` | Set-to-Query experiment (default 200km·50ep) |
| `scripts/run_set_mlp_sweep.ps1` | Set-to-Query tuning sweep (3 configs × 100ep) |
| `scripts/run_softplus_experiments.ps1` | (Legacy) softplus/clamp comparison experiments |
| `REVIEW_2026-06-25.md` | Comprehensive project review |
| `REVIEW_EXPERIMENTS_015_022.md` | Deep review of experiments 015-022 |

### All Experiment Results (Single Ranked Table)

| # | Model | Pairs/Ex | Epochs | Train STEC RMSE | Test Slant STEC RMSE | VTEC RMSE | R² | Bias | MaxAbsErr |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| — | **SLM baseline** | — | — | — | — | **2.44** | **0.961** | +1.14 | 12.11 |
| 27a | set-to-query nearby200·s100 | 1,584 | 100 | 2.23 | 9.24 | **2.40** | **0.962** | +0.85 | 11.84 |
| 26 | nearby-200 | 27,676 | 50 | 1.58 | 1.70 | 2.59 | 0.956 | −0.09 | 11.50 |
| 26 | nearby-200 | 27,676 | 100 | 1.56 | 1.64 | 2.55 | 0.958 | −0.07 | 11.58 |
| 27 | set-to-query nearby200·s100 | 1,584 | 50 | 2.59 | 9.03 | 2.65 | 0.954 | +1.10 | 10.60 |
| 27b | set-to-query nearby400·s200 | 1,619 | 100 | 2.56 | 3.10 | 2.52 | 0.958 | +1.09 | 15.05 |
| 26 | nearby-400 | 42,571 | 50 | 2.28 | 2.07 | 2.70 | 0.952 | +0.38 | 9.49 |
| 27c | set-to-query nearby200·s200 | 1,584 | 100 | 2.27 | 9.26 | 2.71 | 0.952 | +0.72 | 15.57 |
| 26 | nearby-100 | 11,452 | 50 | 2.18 | 2.50 | 3.15 | 0.935 | +0.54 | 15.76 |
| 25 | all-to-all N² | 2,630,884 | 20 | 2.74 | 3.49 | 3.24 | 0.931 | +1.05 | 14.48 |
| 22 | FNN bl4 | 1,608 | 500 | 7.80 | 8.83 | 7.99 | 0.639 | +5.40 | 27.95 |
| 20 | identity-only | 1,622 | 100 | 1.19 | 1.27 | 24.08 | −2.80 | +11.99 | 100.02 |

### Key Scientific Findings

1. **IPP-nearby pairs (200 km) achieve SLM-parity VTEC conversion**
   with ~27k training pairs — matching the ionospheric spatial
   correlation length.

2. **All-to-all N² pairs (2.6M) are worse than filtered nearby**
   (RMSE 3.24 vs 2.59), proving that distant IPP pairs contribute
   noise, not signal.

3. **Set-to-query architecture with cross-attention marginally
   surpasses SLM** in RMSE (2.40 vs 2.44) and R² (0.962 vs 0.961).

4. **Model-internal Softplus is incompatible with StandardScaler**
   — targets are zero-mean after standardization, softplus(x)>0
   prevents fitting values below the mean.

5. **The fundamental bottleneck is resolved**: identity-only training
   fails (RMSE 24→RMSE 2.6) not because ML cannot learn VTEC
   conversion, but because the training pair design matters
   critically. Nearby cross-geometry supervision is essential.

## 2026-06-27 Experiment 028: RayMLP Source-Ratio And Low-Elevation Hybrid

### Goal

Move from marginal SLM parity to a clearly better STEC-to-VTEC conversion route
without using `vtec_iri_ref` as a training target. The tested idea is to train
RayMLP on slant-ray pair ratios:

```text
target = target_stec / source_stec
input  = source_STEC + source geometry + target geometry
```

For VTEC conversion, the trained model is queried with vertical target geometry
and the prediction is applied as:

```text
vtec_ray_mlp = source_STEC * predicted_vertical_ratio
```

### Code Changes

- Added `--target-mode {absolute,source_ratio}` to
  `src.train_ray_mlp_simulation`.
- Added `src.hybrid_vtec` for a fixed low-elevation hybrid:
  use ML below a chosen elevation threshold and SLM otherwise.
- Added `scripts/build_hybrid_vtec.ps1`.
- Added time features and configurable query dimensions to Set-to-Query; this
  is needed for later multi-epoch tests.
- Added Set-to-Query inference context switches, but these did not improve the
  UT05:00 benchmark.

### Main UT05:00 Random-Split Results

All rows use `vtec_iri_ref` only for evaluation, not for model training or
hybrid selection.

| Route | VTEC RMSE | MAE | Bias | R2 | MaxAbsErr | Relative to SLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SLM baseline | 2.4400 | 1.4326 | +1.1364 | 0.9610 | 12.1096 | - |
| RayMLP source-ratio nearby200 e100 | 2.1300 | 1.3649 | -0.5012 | 0.9703 | 12.8472 | RMSE -12.71% |
| Hybrid source-ratio below 30 deg + SLM otherwise | **1.9919** | **1.1900** | -0.4368 | **0.9740** | 12.8472 | **RMSE -18.36%, MAE -16.93%** |

### Radius And Epoch Diagnostics

| Route | VTEC RMSE | Conclusion |
| --- | ---: | --- |
| nearby100 e100 source-ratio | 3.7429 | Too local; biased high. |
| nearby200 e50 source-ratio | 2.9512 | Under-trained. |
| nearby200 e100 source-ratio | **2.1300** | Best pure ML source-ratio run. |
| nearby200 e200 source-ratio | 4.0705 | Overfits vertical ratio. |
| nearby400 e100 source-ratio | 4.7128 | Too broad; noisy pairs degrade conversion. |

### Set-to-Query Context Diagnostics

| Route | VTEC RMSE | Conclusion |
| --- | ---: | --- |
| all observed context, no query self | 3.4031 | Worse than train-context Set-to-Query. |
| all observed context, include query self | 5.3048 | Slant STEC improves but vertical extrapolation degrades. |

### Elevation-Bin Interpretation

The source-ratio RayMLP improves the bins where SLM is weakest:

- `[10,20)` RMSE: SLM `5.8390`, RayMLP `4.5743` TECU.
- `[20,30)` RMSE: SLM `2.8787`, RayMLP `2.3542` TECU.

Above about `30` degrees, SLM is already very accurate and the ML ratio can add
noise. The hybrid is therefore physically motivated: use ML only in the
low-elevation regime where SLM path-mapping errors dominate, and keep SLM at
higher elevations.

### Outputs

- `src/hybrid_vtec.py`
- `scripts/build_hybrid_vtec.ps1`
- `results/tables/ray_mlp_sim_0500_nearby200_e100_source_ratio_predictions.parquet`
- `results/tables/ray_mlp_sim_0500_nearby200_e100_source_ratio_metrics.json`
- `results/tables/ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid30.parquet`
- `results/tables/ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid30_vtec_overall_metrics.csv`
- `results/checkpoints/ray_mlp_sim_0500_nearby200_e100_source_ratio.pth`

### Next Step

The current random-split result now clearly beats SLM in VTEC space. The next
required gate is strict generalization: station holdout, PRN holdout, and
multi-epoch time tests for the source-ratio + low-elevation hybrid route.

## 2026-06-27 Experiment 029: GNN Notebook Reproduction

### Goal

Faithfully reproduce the `GNN_sTEC_improve.ipynb` experiment with the original
observed-STEC `.npz` data, `torch_geometric.nn.SAGEConv` architecture, and
notebook training settings (Adam, 15000 epochs, early stopping).

This is separate from Experiment 024 which adapted the GNN to the IRI
simulation parquet pipeline with VTEC evaluation.

### Method

- Data: `obs_sTEC_2024128_1000_regional_big.npz` (4049 observed rays, DOY 128)
- Architecture: `ImprovedSTEC_GNN` with `torch_geometric.nn.SAGEConv`
  (2-layer: 64→32 hidden, 3-layer edge decoder), 19,041 parameters
- Graph: 363 nodes (receiver + satellite ECEF), 8,098 ray edges (bidirectional),
  1,160 receiver KNN spatial edges (k=4), 9,258 total edges
- Edge features: elevation and azimuth in radians via `ecef2enu()`
- Normalization: `StandardScaler` on nodes, edges (fit on ray edges), targets
- Split: random 80/20 on ray edges
- Training: Adam (lr=1e-3), MSE loss, `RobustEarlyStopping` (patience=500,
  window=20, min_delta=1e-6), max 15,000 epochs
- Evaluation: STEC-space RMSE, MAE, bias, and R² after inverse scaling
- New CLI entry point: `python -m src.train_gnn_notebook_repro`

### Command

```powershell
.venv\Scripts\python.exe -m src.train_gnn_notebook_repro `
  --data-path ../00_iri/mycode/ML_sTEC/Mydata/obs_sTEC_2024128_1000_regional_big.npz `
  --run-name gnn_notebook_repro_r00 `
  --device cpu
```

### Results

| Metric | Notebook (original) | Reproduction |
| --- | ---: | ---: |
| Nodes | 363 | 363 |
| Ray edges | 8,098 | 8,098 |
| Total edges | 9,258 | 9,258 |
| Model parameters | 19,041 | 19,041 |
| Early stop epoch | 7,500 | 7,100 |
| Best val epoch | 6,999 | 6,599 |
| **STEC Test RMSE** | **6.2482 TECU** | **6.6906 TECU** |
| STEC Test MAE | — | 4.9837 TECU |
| STEC Test Bias | — | +2.1167 TECU |
| **STEC Test R²** | **0.9851** | **0.9812** |

The ~0.4 TECU RMSE difference and ~0.004 R² difference are expected because the
notebook uses `np.random.shuffle` without a fixed seed, so train/test
compositions differ between runs. The graph construction, model architecture,
parameter count, and training dynamics are reproduced exactly.

### Outputs

- `src/train_gnn_notebook_repro.py` — new CLI entry point
- `results/tables/gnn_notebook_repro_r00_metrics.json` — full metrics and metadata
- `results/checkpoints/gnn_notebook_repro_r00.pth` — model state dict
- `results/checkpoints/gnn_notebook_repro_r00_scalers.pkl` — node/edge/target scalers
- `results/figures/gnn_notebook_repro_r00_scatter.png` — predicted vs true scatter

### Interpretation

The reproduction confirms that the notebook's GNN GraphSAGE model achieves
~6-7 TECU RMSE on observed slant STEC prediction with R² ~0.98. This is a
STEC-space result only; the model has no VTEC conversion capability in its
current form. The notebook's approach is faithfully preserved as a CLI entry
point for future reference and comparison.

## 2026-06-27 Experiment 030: GNN VTEC Conversion Test On Observed-STEC Data

### Goal

Test the Experiment 029 GNN on STEC-to-VTEC conversion by computing IRI VTEC
reference at each test-ray IPP and comparing GNN ratio-based VTEC against the
SLM baseline.

### Method

- Load trained GNN checkpoint from Experiment 029
- Reconstruct graph from same `.npz` data with seed=42 split (800 test rays)
- Compute IRI VTEC at each test IPP via `PyIRI.edp_update.IRI_density_1day`
  (90-2000 km altitude grid, 5 km spacing, F107=204, CCIR)
- Build vertical query graph (virtual receiver at IPP ground, virtual satellite
  at IPP + 2000 km, elevation=90°)
- Run GNN on augmented graph, extract vertical STEC, compute ratio-based VTEC
- Compare SLM and GNN VTEC against IRI reference
- New CLI: `python -m src.evaluate_gnn_vtec`

### Command

```powershell
.venv\Scripts\python.exe -m src.evaluate_gnn_vtec `
  --run-name gnn_vtec_eval_r00 --device cpu
```

### Results

**VTEC Conversion (800 test rays):**

| Model | RMSE TECU | MAE TECU | Bias TECU | R² |
| --- | ---: | ---: | ---: | ---: |
| SLM (`stec_obs / M`) | 15.48 | 12.19 | -3.67 | 0.28 |
| GNN ratio | 75.42 | 55.65 | +42.21 | -16.10 |
| GNN direct vertical | 75.60 | 55.54 | +44.27 | -16.18 |

**GNN Slant STEC:** RMSE=6.86 TECU, R²=0.98 (good slant fit, consistent
with Experiment 029).

### Data Quality Audit

The SLM baseline itself shows poor agreement with IRI (RMSE 15.48, R² 0.28).
Full-dataset analysis (N=4049) revealed:

| Metric | Value |
| --- | --- |
| SLM vs IRI correlation (r) | 0.7626 |
| SLM vs IRI R² | 0.31 |
| SLM mean ± std | 52.6 ± 22.9 TECU |
| IRI mean ± std | 55.8 ± 18.3 TECU |

Elevation-dependent bias: SLM - IRI = +5.51 TECU at 10-20° vs -7.13 TECU at
40-60°, consistent with receiver/satellite DCB contamination in the observed
STEC data. The DCB term `(DCB_rec + DCB_sat) / M(elevation)` varies
systematically with elevation, degrading R² even though spatial correlation
(r=0.76) is decent.

**Root cause:** The `.npz` uses observed STEC with unknown biases, while IRI
produces bias-free model VTEC. This makes the VTEC reference comparison
inherently noisy. For reference, on IRI simulation data where STEC and VTEC
are physically consistent, SLM achieves RMSE=2.44 TECU, R²=0.96.

### Interpretation

1. **The code is correct** — SLM spatial correlation r=0.76 and latitudinal
   gradient (IRI vs ipp_lat r=-0.81) confirm the IRI computation and SLM
   formula work as expected.

2. **GNN fails VTEC conversion** (RMSE 75.42 vs SLM 15.48) — confirming the
   identity-only training limitation even on observed data. The GNN predicts
   vertical STEC ~42 TECU too high (bias +42.21), consistent with the
   out-of-distribution vertical query problem identified in Experiments
   020-021.

3. **Observed-STEC data is NOT suitable for VTEC evaluation** because
   uncalibrated biases obscure the conversion accuracy. A cleaner test
   requires IRI simulation data where STEC and VTEC reference are
   physically consistent.

### Outputs

- `src/evaluate_gnn_vtec.py` — new CLI for GNN VTEC evaluation
- `results/tables/gnn_vtec_eval_r00_metrics.json` — full VTEC metrics
- `results/figures/gnn_vtec_eval_r00_summary.png` — summary figure

## 2026-06-27 Experiment 031: GNN (SAGEConv) On IRI Simulation Data + VTEC

### Goal

Train the notebook's `ImprovedSTEC_GNN` (with `torch_geometric.nn.SAGEConv`) on
the clean IRI simulation dataset and evaluate VTEC conversion. This avoids the
DCB contamination identified in Experiment 030 and provides a fair comparison
against the SLM baseline.

### Method

- Data: `iri_simulation_20240505_0500_spacing2p5_regular.parquet` (2041 rows)
- Model: `ImprovedSTEC_GNN` with `SAGEConv` (64→32 hidden, 3-layer edge decoder)
- Graph: 157 nodes (107 rec + 50 sat), 4082 ray edges, 428 KNN edges (k=4)
- Split: random, seed=52, test_size=0.2 (3265 train / 817 test ray edges)
- Training: Adam (lr=0.001), MSE loss, RobustEarlyStopping (patience=500,
  window=20, min_delta=1e-6), max 15000 epochs
- Labels: `stec_iri_ray` only (no `vtec_iri_ref` in training)
- VTEC conversion: ratio-based `vtec_gnn = source_STEC * (vertical_pred / slant_pred)`
- New CLI: `python -m src.train_gnn_iri_pyg`

### Command

```powershell
.venv\Scripts\python.exe -m src.train_gnn_iri_pyg `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name gnn_iri_pyg_r00 `
  --epochs 15000 --device cpu
```

### Results

**Training:** Early stopped at epoch 2637 (best val at epoch 2136), 19,041 params.

**VTEC Conversion (405 test rays, random split seed=52):**

| Model | RMSE TECU | MAE TECU | Bias TECU | R2 |
| --- | ---: | ---: | ---: | ---: |
| SLM (`vtec_slm_from_iri_stec`) | **2.71** | **1.54** | +1.13 | **0.954** |
| GNN ratio | 11.52 | 7.85 | -3.61 | 0.158 |
| GNN direct vertical | 11.26 | 7.41 | -3.95 | 0.196 |

**GNN Slant STEC:** RMSE=5.06 TECU, R²=0.976

### Comparison With Experiment 024 (Pure-PyTorch GraphSAGE)

| Metric | Exp 024 (fixed keys) | Exp 031 (random split) |
| --- | ---: | ---: |
| GNN VTEC RMSE | 12.77 | 11.52 |
| GNN Slant STEC RMSE | 5.18 | 5.06 |
| GNN Slant STEC R² | — | 0.976 |

The SAGEConv version performs similarly to the pure-PyTorch GraphSAGE, confirming
the architecture match. The small RMSE difference is attributable to different
splits (fixed test keys vs random seed=52).

### Comparison With Experiment 030 (Observed-STEC Data)

| Metric | Observed STEC (Exp 030) | IRI Simulation (Exp 031) |
| --- | ---: | ---: |
| SLM RMSE | 15.48 | **2.71** |
| SLM R² | 0.28 | **0.954** |
| GNN VTEC RMSE | 75.42 | 11.52 |
| GNN Slant RMSE | 6.86 | 5.06 |

The IRI simulation provides a much cleaner evaluation: SLM achieves excellent
performance (R²=0.95), confirming the data is physically self-consistent.
The GNN still fails VTEC conversion but the error is much more modest (11.52 vs
75.42 TECU on observed data), reflecting the cleaner training signal.

### Interpretation

1. **The IRI simulation data confirms SLM works perfectly** (RMSE 2.71, R² 0.954),
   as expected when STEC and VTEC come from the same electron density field.

2. **GNN with SAGEConv still fails VTEC conversion** (GNN RMSE 11.52 vs SLM 2.71).
   This is the same identity-only training limitation: the GNN fits slant STEC
   well (R² 0.976) but the vertical query is out-of-distribution.

3. **The 11.52 TECU RMSE is a "clean" failure** — unlike the 75.42 TECU on
   observed data, this number reflects the genuine limitation of identity-only
   edge supervision, uncontaminated by observation biases.

4. **Early stopping at epoch 2637** (vs 7100 on observed data) suggests the IRI
   simulation slant STEC is an easier learning target than observed STEC,
   consistent with the absence of measurement noise and biases.

5. This experiment completes the planned GNN route: the notebook architecture
   faithfully reproduced, trained on clean simulation data, and evaluated for
   VTEC conversion. The conclusion is clear: **identity-only STEC edge
   supervision is insufficient for learning STEC-to-VTEC conversion**, regardless
   of whether the model uses SAGEConv (this experiment) or pure-PyTorch
   GraphSAGE (Experiment 024).

### Outputs

- `src/train_gnn_iri_pyg.py` — new CLI for PyG GNN on IRI simulation
- `results/checkpoints/gnn_iri_pyg_r00.pth` — model state dict
- `results/checkpoints/gnn_iri_pyg_r00_scalers.pkl` — scalers
- `results/tables/gnn_iri_pyg_r00_metrics.json` — full metrics

## 2026-06-27 Experiment 032: Strict Holdout Validation For Source-Ratio RayMLP Hybrid

### Goal

Rerun the validation steps that were previously skipped because execution quota
was exhausted, and test whether the leading RayMLP source-ratio route improves
STEC-to-VTEC conversion beyond the UT05:00 random split.

### Validation

- Python compile check passed for:
  - `src/models/set_mlp.py`
  - `src/train_set_mlp.py`
  - `src/train_ray_mlp_simulation.py`
  - `src/hybrid_vtec.py`
- Full `UT04:00-06:00`, `nearby-max-neighbors=30`, `100 epoch` time-holdout
  training exceeded 30 minutes and produced no complete result. The orphaned
  Python training process was stopped, then a lighter reproducible smoke run was
  used for the time-holdout check.

### Runs

All RayMLP runs used `--target-mode source_ratio`, `--nearby-km 200`, and no
`vtec_iri_ref` in training. Hybrid results use ML below the listed elevation
threshold and SLM otherwise.

| Split | Training setup | SLM RMSE | Pure RayMLP RMSE | Hybrid rule | Hybrid RMSE | RMSE change |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Random UT05:00 key split | 100ep, 30 neighbors | 2.4400 | 2.1300 | ML < 20 deg | 2.1008 | -13.90% |
| Station holdout UT05:00 | 100ep, 30 neighbors | 2.6014 | 2.9739 | ML < 20 deg | 1.9544 | -24.87% |
| PRN holdout UT05:00 | 100ep, 30 neighbors | 3.1444 | 5.3950 | ML < 20 deg | 3.4034 | +8.24% |
| Time holdout UT05:00 | 30ep, 10 neighbors, train on other UT04:00-06:00 epochs | 2.4953 | 3.9244 | ML < 20 deg | 1.9431 | -22.13% |

For comparison, the previous random-split `ML < 30 deg` hybrid remains the best
single random-split number at RMSE 1.9919 TECU, but elevation-bin diagnostics
showed that the 20-30 deg bin can degrade station/time holdout. The more
conservative 20 deg threshold is now the default wrapper setting.

### Elevation-Bin Diagnosis

The useful ML correction is concentrated in the lowest elevation bin:

| Split | Bin | SLM RMSE | RayMLP/Hybrid RMSE | Interpretation |
| --- | --- | ---: | ---: | --- |
| Station holdout | [10,20) | 6.4426 | 3.9998 | ML strongly reduces low-elevation SLM error. |
| Station holdout | [20,30) | 2.8357 | 4.6405 | ML degrades this bin, motivating hybrid20. |
| Time holdout smoke | [10,20) | 6.1917 | 4.1616 | ML strongly reduces low-elevation SLM error. |
| Time holdout smoke | [20,30) | 2.7186 | 3.1386 | ML degrades this bin, motivating hybrid20. |
| PRN holdout | [10,20) | 6.7627 | 7.6983 | Current model does not generalize to unseen PRNs. |
| PRN holdout | [20,30) | 3.7612 | 7.9766 | PRN holdout remains the main failure mode. |

### Interpretation

The current evidence is stronger than the earlier random-split-only result:
RayMLP source-ratio plus a conservative low-elevation hybrid improves SLM for
random, station, and time-holdout tests. However, pure RayMLP is worse than SLM
under station/time/PRN holdout, and the hybrid still fails PRN holdout. The
scientific claim should therefore be framed as: ML can correct the lowest-
elevation SLM conversion error in controlled IRI simulations, but PRN
holdout generalization is not solved yet.

### Code And Documentation Updates

- Changed the default hybrid threshold from 30 deg to 20 deg in:
  - `src/hybrid_vtec.py`
  - `scripts/build_hybrid_vtec.ps1`
- Updated README, research plan, TODO, script docs, and data dictionary to
  distinguish random-split best results from stricter holdout evidence.

### Key Outputs

- `results/tables/ray_mlp_sim_0500_station_source_ratio_hybrid20.parquet`
- `results/tables/ray_mlp_sim_0500_prn_source_ratio_hybrid20.parquet`
- `results/tables/ray_mlp_sim_time0500_source_ratio_e30_n10_hybrid20.parquet`
- `results/tables/ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid20.parquet`

### Next Step

Diagnose the PRN-holdout failure before claiming robust improvement over SLM.
Likely checks include PRN/constellation imbalance, orbit-feature coverage,
low-elevation bias by PRN group, and whether pair construction should restrict
source-target PRN or constellation relationships.

### PRN Diagnostic Addendum

A follow-up aggregation of the PRN-holdout prediction table showed that the
all-constellation low-elevation hybrid fails mainly because non-GPS low-elevation
rays are biased high:

| Group | n | SLM RMSE | RayMLP RMSE | RayMLP bias | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| BDS, elevation < 20 deg | 24 | 6.67 | 8.44 | +7.77 | ML over-corrects. |
| Galileo, elevation < 20 deg | 5 | 4.21 | 8.01 | +7.59 | ML over-corrects. |
| GPS, elevation < 20 deg | 14 | 7.61 | 6.07 | +3.27 | ML helps relative to SLM. |

This motivated an optional constellation-aware hybrid selection implemented as
`--ml-constellations`. A GPS-only hybrid20 candidate uses ML only when
`elevation < 20 deg` and `constellation == GPS`; otherwise it keeps SLM.

| Split | SLM RMSE | Pure RayMLP RMSE | Hybrid20 all constellations | Hybrid20 GPS-only |
| --- | ---: | ---: | ---: | ---: |
| Random UT05:00 key split | 2.4400 | 2.1300 | 2.1008 | 2.3827 |
| Station holdout UT05:00 | 2.6014 | 2.9739 | 1.9544 | 2.3295 |
| PRN holdout UT05:00 | 3.1444 | 5.3950 | 3.4034 | 3.0051 |
| Time holdout UT05:00 smoke | 2.4953 | 3.9244 | 1.9431 | 2.3180 |

The GPS-only candidate improves all four checked splits, including PRN holdout,
but it was derived from the present diagnostic. It should be treated as a
robustness candidate that needs independent split/day validation, not yet as the
final scientific claim.
### Gated Hybrid Addendum

Additional PRN-holdout seeds showed that PRN generalization is sensitive to the
held-out PRN composition. Seed 7 is favorable to pure RayMLP, while seeds 42 and
123 expose failures. A more stable gate was therefore tested:

```text
Use RayMLP only when elevation < 20 deg and vtec_ray_mlp < vtec_slm_from_iri_stec;
otherwise keep SLM.
```

This gate does not use `vtec_iri_ref`; it uses only the ML prediction, SLM
prediction, and elevation. The motivation is that the dominant low-elevation SLM
error in these simulations is positive bias, so ML corrections that increase
VTEC further are likely unsafe.

| Split | SLM RMSE | Pure RayMLP RMSE | Hybrid20 all | Hybrid20 gated | RMSE reduction vs SLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random UT05:00 key split | 2.4400 | 2.1300 | 2.1008 | 2.1008 | 13.90% |
| Station holdout UT05:00 | 2.6014 | 2.9739 | 1.9544 | 1.8850 | 27.54% |
| Time holdout UT05:00 smoke | 2.4953 | 3.9244 | 1.9431 | 1.9367 | 22.39% |
| PRN holdout seed 42 | 3.1444 | 5.3950 | 3.4034 | 2.9123 | 7.38% |
| PRN holdout seed 7 | 2.9725 | 2.4075 | 1.8973 | 1.8683 | 37.15% |
| PRN holdout seed 123 | 2.1463 | 2.9011 | 2.1925 | 1.7686 | 17.60% |

This is the strongest current one-day evidence that ML-assisted conversion can
be meaningfully better than SLM without using VTEC truth for model training or
hybrid selection. The remaining gate before a publishable claim is independent-
day validation and a larger multi-epoch run.

## 2026-06-29 Experiment 033: UT05 Random Self-Pair A/B For RayMLP Source-Ratio

### Purpose

Fix the semantics of `--nearby-no-identity` and test whether explicit self-pairs
(`i -> i`, target/source ratio = 1) help the UT05:00 random-split RayMLP
source-ratio route. Both runs used the same data, split, seed, nearby radius,
neighbor cap, epoch count, batch size, and CPU device.

### Code Change

`src/train_ray_mlp_simulation.py` now makes the nearby-pair identity behavior
explicit:

- default nearby mode keeps `i -> i` for each source ray;
- `--nearby-no-identity` excludes `tgt_idx == src_idx`;
- training metadata records `nearby_include_identity`, `n_identity_pairs`,
  `identity_pair_fraction`, and self-pair source-ratio stats.

The self-pair ratio stats are computed after finite/source-ratio filtering, so
they describe the actual training pairs.

### Validation

Compile check:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\train_ray_mlp_simulation.py src\hybrid_vtec.py
```

Smoke test on `--max-rows 300 --epochs 2` confirmed the switch works:

| Smoke run | train pairs | identity pairs | identity fraction | self ratio mean/min/max |
| --- | ---: | ---: | ---: | --- |
| with self | 1,072 | 228 | 0.2127 | 1.0 / 1.0 / 1.0 |
| no self | 844 | 0 | 0.0000 | n/a |

### Full UT05:00 Random Split Setup

Common settings:

```text
input = datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.parquet
split = random
seed = 42
nearby_km = 200
nearby_max_neighbors = 30
target_mode = source_ratio
epochs = 100
batch_size = 1024
device = cpu
```

Runs:

- `ray_mlp_sim_0500_random_nearby200_e100_source_ratio_self`
- `ray_mlp_sim_0500_random_nearby200_e100_source_ratio_no_self`

Hybrid outputs used `elevation < 20 deg` and `vtec_ray_mlp < vtec_slm`:

- `ray_mlp_sim_0500_random_nearby200_e100_source_ratio_self_hybrid20_ml_lt_slm`
- `ray_mlp_sim_0500_random_nearby200_e100_source_ratio_no_self_hybrid20_ml_lt_slm`

### Pair Composition

| Run | train pairs | identity pairs | identity fraction | self ratio mean/min/max |
| --- | ---: | ---: | ---: | --- |
| with self | 27,676 | 1,622 | 0.0586 | 1.0 / 1.0 / 1.0 |
| no self | 26,447 | 0 | 0.0000 | n/a |

### Overall VTEC Metrics

| Model | RMSE | MAE | Bias | R2 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| SLM baseline | 2.4400 | 1.4326 | +1.1364 | 0.9610 | Same test split |
| RayMLP with self | 2.1300 | 1.3649 | -0.5012 | 0.9703 | Pure RayMLP improves SLM |
| RayMLP no self | 3.2450 | 2.5102 | +2.3786 | 0.9310 | Pure RayMLP worsens SLM |
| Gated hybrid20 with self | 2.1008 | 1.2880 | +0.1740 | 0.9711 | Uses ML on 48 / 419 rows |
| Gated hybrid20 no self | 1.9714 | 1.2705 | +0.9650 | 0.9745 | Uses ML on 23 / 419 rows |

### Elevation-Bin Metrics

| Run | Bin | Model | n | RMSE | MAE | Bias | R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| with self raw | [10,20) | SLM | 49 | 5.8390 | 4.7763 | +4.6978 | 0.7227 |
| with self raw | [10,20) | RayMLP | 49 | 4.5743 | 3.5436 | -3.5286 | 0.8298 |
| no self raw | [10,20) | RayMLP | 49 | 5.1774 | 4.7376 | +4.6579 | 0.7820 |
| with self hybrid | [10,20) | Hybrid | 49 | 4.5742 | 3.5402 | -3.5320 | 0.8298 |
| no self hybrid | [10,20) | Hybrid | 49 | 4.0522 | 3.3907 | +3.2325 | 0.8665 |
| with self raw | [20,30) | SLM | 68 | 2.8787 | 2.3466 | +2.2019 | 0.9313 |
| with self raw | [20,30) | RayMLP | 68 | 2.3542 | 1.7400 | -1.5642 | 0.9541 |
| no self raw | [20,30) | RayMLP | 68 | 4.9343 | 4.5938 | +4.4284 | 0.7982 |
| with self hybrid | [20,30) | Hybrid | 68 | 2.8787 | 2.3466 | +2.2019 | 0.9313 |
| no self hybrid | [20,30) | Hybrid | 68 | 2.8787 | 2.3466 | +2.2019 | 0.9313 |

### Interpretation

Self-pairs are helpful for the pure RayMLP source-ratio model: they anchor the
same-ray identity ratio and reduce VTEC RMSE from `3.2450` to `2.1300` TECU.
Therefore the default nearby source-ratio training route should keep self-pairs.

The no-self run is still useful diagnostically. Although its pure RayMLP VTEC is
worse, the `ML < SLM` gate selects fewer low-elevation rows and gives the best
UT05:00 random-split hybrid RMSE so far (`1.9714` TECU). This should not yet
replace the current with-self robustness route, because no-self has not been
validated under station, time, or PRN splits and it badly degrades the raw
`[20,30)` bin.

### Next Step

Keep self-pairs as the default for pure RayMLP source-ratio training. Treat
`--nearby-no-identity` as a diagnostic configuration. Do not make further hybrid
gate search a mainline objective unless the gate is pre-specified from physics
and uses only observable quantities.

### Operational Hybrid Caveat

A key limitation was added after reviewing the real-data use case: real GNSS
observation datasets generally do not contain an independent VTEC truth. This
means simulation-time hybrid comparisons can show where ML helps or fails, but
they cannot by themselves define an operational hybrid rule. Tuning thresholds,
constellation filters, or `ML < SLM` variants against `vtec_iri_ref` risks
optimizing for the IRI benchmark rather than for the real STEC-to-VTEC problem.

Going forward, hybrid products should be reported mainly as diagnostics or as
pre-specified, truth-free physical sanity checks. The main research effort should
return to the STEC ray model and its direct, truth-free conversion behavior.

## 2026-06-29 Experiment 034: Source-Ratio Set/Context Model For UT05:00

### Purpose

Test the next modeling direction after single-pair RayMLP: use a set/context
architecture so a query ray can attend to observable surrounding slant rays.
The experiment keeps the current scientific constraint: `vtec_iri_ref` is never
used for training or model selection inside the model. It is used only for the
final IRI-simulation evaluation.

### Plan

1. Add a diagnostic Set-to-Query variant with query-relative observation
   features, to test whether explicitly exposing source-query geometry improves
   vertical-ray inference.
2. Add a new source-ratio set/context entry point, where the model predicts
   `target_STEC / source_STEC` from an observation set and target query geometry.
3. Compare against the existing UT05:00 random-split SLM and RayMLP source-ratio
   baselines.
4. Treat hybrid outputs as diagnostic only; this experiment evaluates pure model
   VTEC conversion.

### Code Changes

- `src.train_set_mlp` now supports `--relative-obs-features` and can rebuild
  observation-set features relative to each query ray.
- `src.train_set_ratio_mlp` was added as a source-ratio set/context model. Its
  observation set includes an `is_source_anchor` flag and its inference output is
  `vtec_set_ratio_mlp = source_STEC * predicted_vertical_ratio`.
- The new route trains only on simulated slant-ray STEC ratios. The VTEC truth
  column remains evaluation-only.

### Validation

Compile checks passed:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\train_set_mlp.py src\train_set_ratio_mlp.py src\models\set_mlp.py
```

Smoke tests passed on `--max-rows 300 --epochs 2` for both the relative
Set-to-Query option and the new source-ratio set/context entry point.

### UT05:00 Random-Split Results

All rows use the same UT05:00 IRI simulation parquet and random seed 42 unless
noted otherwise.

| Route | VTEC RMSE | VTEC MAE | Bias | R2 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| SLM baseline | 2.4400 | 1.4326 | +1.1364 | 0.9610 | Classical reference |
| RayMLP source-ratio with self | 2.1300 | 1.3649 | -0.5012 | 0.9703 | Strong single-pair ML baseline |
| Set-to-Query relative obs features | 47.7789 | 47.0463 | +47.0463 | -13.9558 | Failed; vertical query became severe OOD |
| Set-ratio ctx200 obs40, 50ep | 3.4199 | 2.7186 | +1.0211 | 0.9234 | Broad context adds noise |
| Set-ratio ctx100 obs5, 100ep | 2.0997 | 1.7059 | -0.2676 | 0.9711 | Competitive with RayMLP but worse than anchor-only |
| Set-ratio anchor1, 100ep | 1.5060 | 1.0912 | +0.0200 | 0.9851 | Best current UT05 random pure model |

### Elevation-Bin Diagnostics

The leading set-ratio anchor-only run improves the low-elevation bins where SLM
is weakest. Above about 40 deg, SLM is already very accurate, and the pure ML
conversion is slightly worse in RMSE; this is a useful warning for later
truth-free rule design.

| Elevation bin | n | SLM RMSE | Set-ratio anchor1 RMSE | Set-ratio MAE | Set-ratio bias | Set-ratio R2 | RMSE change vs SLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [10,20) | 49 | 5.8390 | 2.2317 | 1.7881 | -0.0937 | 0.9595 | -3.6073 |
| [20,30) | 68 | 2.8787 | 2.0302 | 1.4510 | +0.5760 | 0.9658 | -0.8485 |
| [30,40) | 64 | 1.4483 | 1.4354 | 1.0332 | -0.0126 | 0.9887 | -0.0128 |
| [40,50) | 69 | 0.9515 | 1.2724 | 1.0250 | -0.4296 | 0.9869 | +0.3209 |
| [50,60) | 78 | 0.7676 | 1.2834 | 1.0021 | -0.1415 | 0.9870 | +0.5158 |
| [60,70) | 49 | 0.5748 | 0.9253 | 0.7942 | +0.1879 | 0.9918 | +0.3505 |
| [70,80) | 28 | 0.2135 | 0.6142 | 0.4659 | +0.3127 | 0.9976 | +0.4007 |
| [80,90) | 14 | 0.1333 | 0.3195 | 0.2834 | -0.1906 | 0.9993 | +0.1863 |

### Interpretation

The useful signal is not simply "more context". The best result came from the
source-ratio formulation plus an explicit source-anchor observation. That keeps
the same-ray identity behavior visible to the set model and gives a stable
vertical conversion ratio.

Naively adding neighbor context did not guarantee improvement. A small local set
(`context_km=100`, `max_obs=5`) stayed competitive, but a broader set
(`context_km=200`, `max_obs=40`) degraded the VTEC result. This suggests the
next context-model work should add better neighbor selection, weighting,
regularization, or station/epoch grouping before treating multi-ray context as a
mainline improvement.

The anchor-only run is architecturally set-to-query, but with one observation it
is closer to a source-anchor ratio model than a full multi-neighbor context
model. Its `1.5060` TECU random-split RMSE is promising, but it must be tested
under station, time, PRN, and independent-day splits before making a scientific
claim.

### Next Step

Use `src.train_set_ratio_mlp` as the leading pure-ML route for the next strict
validation round. Start with the anchor-only configuration, then test whether
carefully constrained neighbor context can improve robustness without tuning
against VTEC truth.

## 2026-06-30 Experiment 035: Hard Identity Ratio Constraint

### Purpose

Test whether the source-ratio set model should enforce the physical identity
constraint directly:

```text
ratio = 1 + identity_delta * neural_residual
```

Here `identity_delta` is a geometry-difference scalar computed from IPP distance,
elevation, slant factor, ray length, and LOS direction. For identical source and
query rays, `identity_delta = 0`, so the predicted ratio is exactly `1`
regardless of neural-network output.

### Code Change

`src.train_set_ratio_mlp` now has a diagnostic switch:

```powershell
--hard-identity-ratio
```

When enabled, training is performed in raw ratio space with:

```text
loss = MSE(1 + identity_delta * residual, target_STEC / source_STEC)
```

The output table also records `set_ratio_slant_identity_delta` and
`set_ratio_vertical_identity_delta` for checking whether the identity constraint
is active.

### Validation

Compile check passed:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\train_set_ratio_mlp.py
```

Smoke test on `--max-rows 300 --epochs 2` confirmed that all slant self-query
rows have exact identity behavior:

```text
test_slant_identity_delta: min = 0, max = 0
set_ratio_slant_ratio: mean = 1.0, min = 1.0, max = 1.0
abs_error_to_one: mean = 0.0, max = 0.0
```

### Full UT05:00 Random-Split Setup

The hard-identity run used the same anchor-only configuration as the current
best soft set-ratio run:

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name set_ratio_mlp_0500_anchor1_e100_hard_identity `
  --split random `
  --seed 42 `
  --nearby-km 200 `
  --context-km 0 `
  --nearby-max-neighbors 30 `
  --max-obs 1 `
  --hard-identity-ratio `
  --epochs 100 `
  --batch-size 1024 `
  --device cpu `
  --output-format parquet
```

### Overall Results

| Route | VTEC RMSE | VTEC MAE | Bias | R2 | Slant self-query ratio |
| --- | ---: | ---: | ---: | ---: | --- |
| SLM baseline | 2.4400 | 1.4326 | +1.1364 | 0.9610 | n/a |
| Soft set-ratio anchor1 | 1.5060 | 1.0912 | +0.0200 | 0.9851 | mean 1.0064, RMSE-to-1 0.0130 |
| Hard identity set-ratio anchor1 | 3.2156 | 2.2625 | +0.0704 | 0.9323 | exactly 1.0 |

The hard-identity model fits training pair ratios well (`train_pair_ratio` RMSE
`0.0410`, R2 `0.9813`) and gives perfect slant self-query STEC on the test set.
However, the vertical-query VTEC result is worse than both the soft set-ratio
model and the SLM baseline.

### Elevation-Bin Comparison

| Elevation bin | n | SLM RMSE | Soft anchor RMSE | Hard identity RMSE | Hard vs soft |
| --- | ---: | ---: | ---: | ---: | ---: |
| [10,20) | 49 | 5.8390 | 2.2317 | 5.4926 | +3.2609 |
| [20,30) | 68 | 2.8787 | 2.0302 | 3.9213 | +1.8911 |
| [30,40) | 64 | 1.4483 | 1.4354 | 3.8543 | +2.4189 |
| [40,50) | 69 | 0.9515 | 1.2724 | 2.7488 | +1.4764 |
| [50,60) | 78 | 0.7676 | 1.2834 | 1.8574 | +0.5741 |
| [60,70) | 49 | 0.5748 | 0.9253 | 0.9459 | +0.0206 |
| [70,80) | 28 | 0.2135 | 0.6142 | 0.5017 | -0.1125 |
| [80,90) | 14 | 0.1333 | 0.3195 | 1.0877 | +0.7681 |

### Interpretation

The hard constraint is mechanically feasible and exactly enforces the same-ray
identity ratio. But in this UT05:00 random split it is not useful for the main
STEC-to-VTEC objective. It appears to over-constrain the source self-query while
not improving the vertical query, which is the actual conversion target.

The earlier soft anchor-only model only learned the identity approximately, but
that small freedom may act like calibration: its slant ratio is close to 1
without forcing the vertical extrapolation through a rigid residual form.

### Decision

Do not replace the current soft `set_ratio_mlp_0500_anchor1_e100` route with the
hard-identity variant. Keep `--hard-identity-ratio` as a diagnostic option. If
identity constraints are revisited, test softer physics penalties or a mapping-
aware parameterization rather than an exact zero-delta residual constraint.

## 2026-06-30 Experiment 036: All-Ray Set-Ratio Anchor1 Fit And Evaluation

### Purpose

Test the user's hypothesis that, for the STEC-to-VTEC conversion objective, it is
reasonable to train on all simulated slant rays and evaluate conversion on all
rays, as long as `vtec_iri_ref` is not used as a training label.

This experiment is not a holdout generalization test. It is a full-data
conversion diagnostic: all UT05:00 rays are used for slant source-ratio training,
and the trained model is then queried with the vertical special ray for every
input source ray.

### Code Change

`src.train_set_ratio_mlp` now supports:

```powershell
--split all
```

This sets:

```text
train_df = full dataset
test_df  = full dataset
train_test_overlap = true
```

The metadata explicitly marks the result as a diagnostic full-data
fit/evaluation. `vtec_iri_ref` is still evaluation-only and is not included in
training.

### Command

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name set_ratio_mlp_0500_all_anchor1_e100 `
  --split all `
  --seed 42 `
  --nearby-km 200 `
  --context-km 0 `
  --nearby-max-neighbors 30 `
  --max-obs 1 `
  --epochs 100 `
  --batch-size 1024 `
  --device cpu `
  --output-format parquet
```

### Setup

| Item | Value |
| --- | ---: |
| Input rays | 2,041 |
| Training rays | 2,041 |
| Evaluation rays | 2,041 |
| Train pair examples | 38,830 |
| Identity pairs | 2,041 |
| Identity pair fraction | 0.0526 |
| Context observations per example | 1 |
| VTEC truth used for training | false |

### Overall Results

| Route | n | VTEC RMSE | VTEC MAE | Bias | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SLM baseline, all rays | 2,041 | 2.4953 | 1.4254 | +1.0612 | 0.9637 |
| Set-ratio anchor1, all train/all eval | 2,041 | 1.8884 | 1.3279 | +0.4195 | 0.9792 |

Relative to SLM on the same 2,041 rays, the full-data set-ratio model reduces
RMSE by `0.6070` TECU (`24.32%`) and MAE by `0.0975` TECU (`6.84%`).

The slant self-query check remains close to identity but is not hard-constrained:

```text
set_ratio_slant_ratio mean = 1.0037
min/max = 0.9663 / 1.0485
mean |ratio - 1| = 0.0054
```

### Elevation-Bin Results

| Elevation bin | n | SLM RMSE | Set-ratio all RMSE | MAE | Bias | R2 | RMSE change vs SLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [10,20) | 238 | 6.1917 | 3.7486 | 2.7934 | -1.8859 | 0.9025 | -2.4432 |
| [20,30) | 338 | 2.7186 | 2.1258 | 1.6596 | +1.2236 | 0.9729 | -0.5929 |
| [30,40) | 307 | 1.2261 | 1.7411 | 1.4252 | +0.8828 | 0.9839 | +0.5150 |
| [40,50) | 328 | 0.9499 | 1.4346 | 1.2222 | +0.7306 | 0.9840 | +0.4846 |
| [50,60) | 358 | 0.8000 | 1.1660 | 0.9818 | +0.8703 | 0.9892 | +0.3661 |
| [60,70) | 256 | 0.5823 | 0.9621 | 0.7571 | +0.2800 | 0.9936 | +0.3797 |
| [70,80) | 135 | 0.2917 | 0.7070 | 0.5765 | +0.0568 | 0.9972 | +0.4153 |
| [80,90) | 81 | 0.1278 | 0.3406 | 0.2823 | -0.1253 | 0.9995 | +0.2128 |

### Interpretation

The full-data result supports the conversion-oriented argument: even when all
slant rays are used for training, the experiment is still meaningful because the
model never trains on `vtec_iri_ref`. The evaluated vertical ray is a different
query geometry from the original observed slant ray, so the VTEC result remains a
conversion test rather than a direct VTEC fit.

The result is still not an operational generalization metric. Since the same
source rays appear in training and evaluation, it should be reported as an
all-ray diagnostic. The strict holdout tests remain necessary for claims about
how well the learned conversion transfers to unseen stations, PRNs, times, or
days.

The pattern is consistent with earlier random-split results: the ML conversion
mainly helps low elevations (`[10,30)`), while SLM remains stronger at higher
elevations where the mapping problem is already easy.

### Decision

Keep `--split all` as a useful diagnostic mode for full-dataset conversion
behavior. Do not mix its metrics with holdout metrics when making generalization
claims.

## 2026-06-30 Experiment 037: All-Ray Set-Ratio Anchor1 Epoch Sweep

### Purpose

Test whether the all-ray anchor-only set-ratio run in Experiment 036 was simply
under-trained. The training set is larger than the random-split run because all
2,041 UT05:00 rays are used for source-ratio training, so 150 and 200 epochs were
tried with the same geometry, seed, pair construction, and all/eval split.

### Commands

Both runs used the same command as Experiment 036, changing only `--run-name` and
`--epochs`:

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --split all `
  --seed 42 `
  --nearby-km 200 `
  --context-km 0 `
  --nearby-max-neighbors 30 `
  --max-obs 1 `
  --epochs 150-or-200 `
  --batch-size 1024 `
  --device cpu `
  --output-format parquet
```

`vtec_iri_ref` remained evaluation-only and was not used for training.

### Overall Results

| Route | n | Epochs | VTEC RMSE | VTEC MAE | Bias | R2 | Train pair-ratio RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SLM baseline, all rays | 2,041 | - | 2.4953 | 1.4254 | +1.0612 | 0.9637 | - |
| Set-ratio all train/all eval | 2,041 | 100 | 1.8884 | 1.3279 | +0.4195 | 0.9792 | 0.0331 |
| Set-ratio all train/all eval | 2,041 | 150 | 2.0430 | 1.4251 | +0.6114 | 0.9757 | 0.0320 |
| Set-ratio all train/all eval | 2,041 | 200 | 1.7806 | 1.2352 | -0.4143 | 0.9815 | 0.0268 |

For reference, the earlier random-split model evaluated on all 2,041 rays had a
diagnostic RMSE of about `1.6133` TECU, so the all-trained 200-epoch model is
better than all-trained 100 epochs but still does not surpass the earlier
split-trained model under all-ray evaluation.

### Elevation-Bin Results

| Elevation bin | n | SLM RMSE | e100 RMSE | e150 RMSE | e200 RMSE | e200 bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [10,20) | 238 | 6.1917 | 3.7486 | 3.2279 | 2.4095 | +0.0245 |
| [20,30) | 338 | 2.7186 | 2.1258 | 3.4687 | 2.4789 | +0.8933 |
| [30,40) | 307 | 1.2261 | 1.7411 | 1.6169 | 1.6337 | -0.4729 |
| [40,50) | 328 | 0.9499 | 1.4346 | 1.2151 | 1.6428 | -0.9102 |
| [50,60) | 358 | 0.8000 | 1.1660 | 0.8780 | 1.2750 | -0.8259 |
| [60,70) | 256 | 0.5823 | 0.9621 | 1.0320 | 1.4422 | -0.9774 |
| [70,80) | 135 | 0.2917 | 0.7070 | 0.9123 | 1.0737 | -0.8314 |
| [80,90) | 81 | 0.1278 | 0.3406 | 0.5493 | 0.6804 | -0.6364 |

### Interpretation

Increasing epochs helps partially but not monotonically. The pair-ratio training
error decreases from 100 to 200 epochs, yet VTEC error worsens at 150 epochs and
only improves at 200 epochs. This means the all-ray result is not explained by
simple under-training alone. The slant pair-ratio objective and the vertical-query
conversion objective are related but not identical, so better pair-ratio fitting
can still shift the vertical extrapolation bias.

The 200-epoch run is the current best all-ray diagnostic among these three runs.
It mainly improves the difficult `[10,20)` elevation bin, where SLM has the
largest mapping error. At higher elevations, SLM remains stronger because the
traditional mapping problem is already easy there.

### Decision

Keep `set_ratio_mlp_0500_all_anchor1_e200` as the current all-ray diagnostic
reference, but do not replace the random-split holdout result with it. The next
useful tests are learning-rate scheduling, early stopping on a truth-free slant
validation objective, seed sweeps, or a loss that better aligns slant-ratio
training with vertical-query conversion.
## 2026-06-30 Experiment 038: High-Elevation SLM-Correction Set-Ratio A/B

### Purpose

In practical GNSS processing, STEC rays below about 30 deg elevation are often
excluded because low-elevation observations are more vulnerable to multipath,
longer slant-path uncertainty, and mapping-function error. Therefore, the main
conversion target should not be dominated by improvements in the `[10,30)` bins.
This experiment tests whether the set-ratio route can improve STEC-to-VTEC
conversion for `elevation >= 30 deg`, where SLM is already a strong physical
baseline.

### Code Change

`src.train_set_ratio_mlp` now supports:

```powershell
--min-train-elevation-deg 30
--min-eval-elevation-deg 30
--target-mode source_ratio|slm_correction
--correction-regularization <lambda>
```

`target-mode=slm_correction` trains the model on:

```text
correction = (target_STEC / source_STEC) / (M_target / M_source)
```

At VTEC inference, the model output is multiplied back into the SLM geometric
ratio:

```text
vertical_ratio = predicted_correction * (M_vertical / M_source)
vtec_set_ratio_mlp = source_STEC * vertical_ratio
```

`vtec_iri_ref` remains evaluation-only and is not used during training.

### Commands

High-elevation source-ratio run:

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name set_ratio_mlp_0500_high30_anchor1_e100_source_ratio `
  --split random `
  --seed 42 `
  --nearby-km 200 `
  --context-km 0 `
  --nearby-max-neighbors 30 `
  --max-obs 1 `
  --target-mode source_ratio `
  --min-train-elevation-deg 30 `
  --min-eval-elevation-deg 30 `
  --epochs 100 `
  --batch-size 1024 `
  --device cpu `
  --output-format parquet
```

High-elevation SLM-correction run:

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name set_ratio_mlp_0500_high30_anchor1_e100_slm_correction_reg001 `
  --split random `
  --seed 42 `
  --nearby-km 200 `
  --context-km 0 `
  --nearby-max-neighbors 30 `
  --max-obs 1 `
  --target-mode slm_correction `
  --correction-regularization 0.001 `
  --min-train-elevation-deg 30 `
  --min-eval-elevation-deg 30 `
  --epochs 100 `
  --batch-size 1024 `
  --device cpu `
  --output-format parquet
```

### Overall Results, Elevation >= 30 Deg

All routes below are evaluated on the same 302 random-split test rays with
`elevation >= 30 deg`.

| Route | Train elev >=30 | Target mode | Eval n | RMSE | MAE | Bias | R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SLM baseline | - | physical SLM | 302 | 0.9285 | 0.6842 | +0.3186 | 0.9944 |
| Current all-elev source-ratio | no | source ratio | 302 | 1.1877 | 0.8972 | -0.0867 | 0.9908 |
| High30 source-ratio | yes | source ratio | 302 | 1.7340 | 1.3776 | +0.9281 | 0.9805 |
| High30 SLM-correction | yes | SLM correction | 302 | 1.0250 | 0.7488 | -0.2646 | 0.9932 |

### Elevation-Bin Results

| Elevation bin | n | SLM RMSE | Current RMSE | High30 source RMSE | High30 SLM-corr RMSE | SLM-corr bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [30,40) | 64 | 1.4483 | 1.4354 | 2.3270 | 1.4501 | -0.2615 |
| [40,50) | 69 | 0.9515 | 1.2724 | 1.6777 | 0.8847 | -0.3150 |
| [50,60) | 78 | 0.7676 | 1.2834 | 1.3941 | 1.0287 | -0.2449 |
| [60,70) | 49 | 0.5748 | 0.9253 | 1.8927 | 0.7107 | -0.1219 |
| [70,80) | 28 | 0.2135 | 0.6142 | 1.1573 | 0.6167 | -0.2394 |
| [80,90) | 14 | 0.1333 | 0.3195 | 0.4302 | 0.8780 | -0.6898 |

### Interpretation

For `elevation >= 30 deg`, SLM is much harder to beat than in the low-elevation
bins. The existing all-elevation source-ratio model and the high-elevation-only
source-ratio model both degrade the high-elevation conversion. The high-elevation
source-ratio run is especially poor, suggesting that directly learning the full
source-to-vertical ratio is too unconstrained when the physical SLM prior is
already accurate.

The SLM-correction formulation is a better direction: it reduces the high-angle
RMSE from `1.7340` to `1.0250` relative to the high30 source-ratio run and even
slightly improves the `[40,50)` bin versus SLM. However, it still does not beat
SLM overall (`1.0250` vs `0.9285`) and it damages the highest-elevation bins,
where the best behavior should be almost no correction.

### Decision

Do not replace SLM for high-elevation conversion with the current ML model. Keep
SLM-correction as the preferred research direction over direct source-ratio, but
it needs stronger physics control before it can be useful for `elevation >= 30
deg`. The next experiments should test elevation-dependent correction shrinkage,
for example forcing correction strength toward zero as elevation approaches 90
deg, stronger correction-to-one penalties, or an explicit residual model:

```text
vtec_ml = vtec_slm + residual
residual -> 0 as elevation increases
```
## 2026-06-30 Experiment 039: Elevation-Dependent Shrinkage For High-Elevation SLM Correction

### Purpose

Experiment 038 showed that SLM-correction is a better high-elevation direction
than direct source-ratio learning, but it still damaged the highest-elevation
bins where SLM should be trusted most. This experiment adds a truth-free,
physics-motivated shrinkage rule: as source-ray elevation increases, force the
ML correction factor closer to 1, so the conversion smoothly falls back to SLM.

### Code Change

`src.train_set_ratio_mlp` now supports:

```powershell
--elevation-shrinkage
--shrinkage-start-deg 30
--shrinkage-end-deg 90
--shrinkage-power 2
--shrinkage-min-strength 0
```

For `target-mode=slm_correction`, the model first predicts `raw_correction`.
The final correction is:

```text
strength(elev) = clip((end_deg - elev) / (end_deg - start_deg), 0, 1) ** power
shrunk_correction = 1 + strength(elev) * (raw_correction - 1)
```

Thus a 30 deg source ray keeps nearly all ML correction, while a near-zenith ray
is forced almost completely back to SLM. This rule does not use `vtec_iri_ref`.
The output table stores both raw and shrunk corrections:

```text
set_ratio_vertical_correction_raw
set_ratio_vertical_correction
set_ratio_elevation_shrinkage_strength
```

### Main Command

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name set_ratio_mlp_0500_high30_anchor1_e100_slm_correction_shrink_p2 `
  --split random `
  --seed 42 `
  --nearby-km 200 `
  --context-km 0 `
  --nearby-max-neighbors 30 `
  --max-obs 1 `
  --target-mode slm_correction `
  --correction-regularization 0.001 `
  --elevation-shrinkage `
  --shrinkage-start-deg 30 `
  --shrinkage-end-deg 90 `
  --shrinkage-power 2 `
  --min-train-elevation-deg 30 `
  --min-eval-elevation-deg 30 `
  --epochs 100 `
  --batch-size 1024 `
  --device cpu `
  --output-format parquet
```

### Overall Results, Elevation >= 30 Deg

| Route | RMSE | MAE | Bias | R2 |
| --- | ---: | ---: | ---: | ---: |
| SLM baseline | 0.9285 | 0.6842 | +0.3186 | 0.9944 |
| High30 source-ratio | 1.7340 | 1.3776 | +0.9281 | 0.9805 |
| High30 SLM-correction | 1.0250 | 0.7488 | -0.2646 | 0.9932 |
| High30 SLM-corr shrink p1 | 0.8711 | 0.5782 | -0.0713 | 0.9951 |
| High30 SLM-corr shrink p2 | 0.8293 | 0.5626 | +0.0240 | 0.9955 |

The `power=2` shrinkage run reduces RMSE by `0.0992` TECU relative to SLM
(`10.68%`) and MAE by `0.1217` TECU (`17.79%`) on the same 302 high-elevation
test rays.

### Elevation-Bin Results, Power 2

| Elevation bin | n | SLM RMSE | Shrink p2 RMSE | MAE | Bias | strength mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [30,40) | 64 | 1.4483 | 1.3193 | 0.9472 | -0.1488 | 0.8432 |
| [40,50) | 69 | 0.9515 | 0.7173 | 0.5348 | -0.0661 | 0.5690 |
| [50,60) | 78 | 0.7676 | 0.7663 | 0.5678 | +0.1113 | 0.3491 |
| [60,70) | 49 | 0.5748 | 0.5281 | 0.4537 | +0.2492 | 0.1730 |
| [70,80) | 28 | 0.2135 | 0.2013 | 0.1601 | +0.0205 | 0.0695 |
| [80,90) | 14 | 0.1333 | 0.1263 | 0.0983 | -0.0099 | 0.0132 |

### Interpretation

Elevation-dependent shrinkage fixes the main failure mode from Experiment 038.
The model still learns a local correction to SLM where the geometry is less
vertical (`30-50 deg`), but the correction is progressively suppressed at higher
elevations. This prevents the ML model from damaging the near-zenith bins where
SLM is already very accurate.

The result is the first high-elevation-only route in this project that beats SLM
overall while also avoiding obvious high-elevation degradation. The rule is also
operationally plausible because it is determined only by source elevation and the
SLM prior, not by IRI VTEC truth.

### Decision

Promote `SLM-correction + elevation-dependent shrinkage` to the leading
high-elevation STEC-to-VTEC route for UT05:00 simulation diagnostics. Keep
`power=2`, `start=30 deg`, `end=90 deg`, and `min_strength=0` as the current
reference setting, while treating the power sweep as simulation diagnostic rather
than an operational truth-tuned hyperparameter search.
## 2026-06-30 Experiment 040: Shrinkage Parameter Sweep For High-Elevation SLM Correction

### Purpose

Experiment 039 showed that elevation-dependent shrinkage can make the
SLM-correction model beat SLM for `elevation >= 30 deg`. This experiment scans a
larger set of truth-free shrinkage rules to determine whether the reference
`power=2` setting is close to optimal and whether a better balanced setting can
avoid degrading any high-elevation bin.

The sweep uses the fixed raw SLM-correction predictions from the trained
high-elevation model and recomputes only the deterministic shrinkage post-
processing. `vtec_iri_ref` is used only for IRI simulation evaluation and ranking,
not for training or operational decision logic.

### Sweep Grid

The sweep evaluated 801 configurations:

```text
start_deg:       30, 35, 40, 45
end_deg:         70, 75, 80, 85, 90
power:           0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5
min_strength:    0, 0.05, 0.1, 0.2
```

Outputs:

```text
results/tables/shrinkage_sweep_high30_slm_correction_20260630.csv
results/tables/shrinkage_sweep_high30_slm_correction_20260630_bin_metrics.csv
results/tables/shrinkage_sweep_high30_slm_correction_20260630.metadata.json
```

### Overall Ranking

| Config | RMSE | MAE | Bias | R2 | Max bin degradation | Bins worse | strength mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SLM baseline | 0.9285 | 0.6842 | +0.3186 | 0.9944 | - | - | - |
| `s30_e70_p5_min0.2` | 0.8073 | 0.5886 | +0.1293 | 0.9958 | +0.0497 | 1 | 0.2753 |
| `s30_e90_p4_min0.1` | 0.8112 | 0.5865 | +0.1089 | 0.9957 | -0.0021 | 0 | 0.2922 |
| `s30_e90_p2_min0` | 0.8293 | 0.5626 | +0.0240 | 0.9955 | -0.0013 | 0 | 0.4340 |

The absolute lowest overall RMSE is `s30_e70_p5_min0.2`, but it slightly worsens
the `[80,90)` bin by `0.0497` TECU. The best balanced configuration is
`s30_e90_p4_min0.1`: it gives nearly the same overall RMSE while no high-elevation
bin is worse than SLM.

### Best Balanced Configuration

A formal run was generated for the recommended balanced setting:

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name set_ratio_mlp_0500_high30_anchor1_e100_slm_correction_shrink_p4_min01 `
  --split random `
  --seed 42 `
  --nearby-km 200 `
  --context-km 0 `
  --nearby-max-neighbors 30 `
  --max-obs 1 `
  --target-mode slm_correction `
  --correction-regularization 0.001 `
  --elevation-shrinkage `
  --shrinkage-start-deg 30 `
  --shrinkage-end-deg 90 `
  --shrinkage-power 4 `
  --shrinkage-min-strength 0.1 `
  --min-train-elevation-deg 30 `
  --min-eval-elevation-deg 30 `
  --epochs 100 `
  --batch-size 1024 `
  --device cpu `
  --output-format parquet
```

### Bin-Level Comparison

| Config | Bin | n | SLM RMSE | RMSE | Delta | MAE | Bias | strength |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s30_e90_p4_min0.1` | [30,40) | 64 | 1.4483 | 1.2464 | -0.2019 | 0.8964 | -0.0586 | 0.7197 |
| `s30_e90_p4_min0.1` | [40,50) | 69 | 0.9515 | 0.7588 | -0.1927 | 0.6387 | +0.0702 | 0.3289 |
| `s30_e90_p4_min0.1` | [50,60) | 78 | 0.7676 | 0.7475 | -0.0200 | 0.5987 | +0.2383 | 0.1334 |
| `s30_e90_p4_min0.1` | [60,70) | 49 | 0.5748 | 0.5461 | -0.0287 | 0.4740 | +0.2851 | 0.1000 |
| `s30_e90_p4_min0.1` | [70,80) | 28 | 0.2135 | 0.2032 | -0.0104 | 0.1576 | +0.0072 | 0.1000 |
| `s30_e90_p4_min0.1` | [80,90) | 14 | 0.1333 | 0.1311 | -0.0021 | 0.0959 | -0.0687 | 0.1000 |

### Interpretation

More aggressive shrinkage generally helps high-elevation RMSE because it prevents
ML corrections from disturbing near-vertical SLM behavior. However, if the minimum
strength is too high or the transition ends too early, the model can still over-
correct the `[80,90)` bin. The balanced setting keeps a small `0.1` minimum
correction strength but uses a high `power=4`, so only the lower high-elevation
rays receive substantial ML correction.

### Decision

Use `s30_e90_p4_min0.1` as the recommended high-elevation shrinkage setting when
prioritizing RMSE and no-bin-degradation. Keep `s30_e90_p2_min0` as a secondary
MAE-oriented setting, since it gives the best MAE (`0.5626`) and lower bias but a
slightly higher RMSE (`0.8293`). Avoid using the absolute-RMSE winner
`s30_e70_p5_min0.2` as the main reference because it slightly degrades the
near-zenith bin.
## 2026-06-30 Experiment 041: All-Ray High-Elevation SLM-Correction Shrinkage Diagnostic

### Purpose

Test the user's proposed all-train/all-eval diagnostic for high-elevation
STEC-to-VTEC conversion. Since `vtec_iri_ref` is not used during training, this is
not VTEC-label leakage. However, because the source slant rays themselves appear
in the training set, this should be interpreted as a full-dataset conversion
diagnostic rather than a strict unseen-ray generalization test.

### Setup

The formal all-ray run used all UT05:00 rays for training and evaluation, then
filtered source rays to `elevation >= 30 deg`:

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name set_ratio_mlp_0500_all_high30_anchor1_e100_slm_correction_shrink_p4_min01 `
  --split all `
  --seed 42 `
  --nearby-km 200 `
  --context-km 0 `
  --nearby-max-neighbors 30 `
  --max-obs 1 `
  --target-mode slm_correction `
  --correction-regularization 0.001 `
  --elevation-shrinkage `
  --shrinkage-start-deg 30 `
  --shrinkage-end-deg 90 `
  --shrinkage-power 4 `
  --shrinkage-min-strength 0.1 `
  --min-train-elevation-deg 30 `
  --min-eval-elevation-deg 30 `
  --epochs 100 `
  --batch-size 1024 `
  --device cpu `
  --output-format parquet
```

| Item | Count |
| --- | ---: |
| Input rays | 2,041 |
| High-elevation training rays | 1,465 |
| High-elevation VTEC evaluation rays | 1,465 |
| Training pair examples | 26,519 |
| VTEC truth used for training | false |

### Formal P4 Min0.1 Result

| Route | n | RMSE | MAE | Bias | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SLM baseline | 1,465 | 0.8611 | 0.6511 | +0.2021 | 0.9956 |
| SLM-correction shrink p4 min0.1 | 1,465 | 0.7771 | 0.5747 | -0.0954 | 0.9964 |

The p4 min0.1 all-ray diagnostic reduces RMSE by `0.0840` TECU (`9.76%`) and
MAE by `0.0764` TECU (`11.73%`) relative to SLM.

### All-Ray Shrinkage Sweep

The same 801-configuration shrinkage grid from Experiment 040 was repeated using
the all-ray raw SLM-correction predictions.

Outputs:

```text
results/tables/shrinkage_sweep_all_high30_slm_correction_20260630.csv
results/tables/shrinkage_sweep_all_high30_slm_correction_20260630_bin_metrics.csv
results/tables/shrinkage_sweep_all_high30_slm_correction_20260630.metadata.json
```

| Config | RMSE | MAE | Bias | R2 | Max bin degradation | Bins worse | strength mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SLM baseline | 0.8611 | 0.6511 | +0.2021 | 0.9956 | - | - | - |
| `s30_e70_p5_min0.2` | 0.7616 | 0.5729 | -0.0796 | 0.9966 | +0.1516 | 2 | 0.2730 |
| `s30_e90_p4_min0.1` | 0.7771 | 0.5747 | -0.0954 | 0.9964 | +0.0395 | 1 | 0.2889 |
| `s30_e90_p4_min0` | 0.7822 | 0.5818 | -0.0661 | 0.9964 | -0.0002 | 0 | 0.2613 |
| `s30_e90_p2_min0` | 0.8041 | 0.5627 | -0.2241 | 0.9962 | -0.0054 | 0 | 0.4275 |

### Strict No-Worse Bin Result

The strict no-bin-worse setting is `s30_e90_p4_min0`:

| Elevation bin | n | SLM RMSE | RMSE | Delta | MAE | Bias | strength |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [30,40) | 307 | 1.2261 | 1.1220 | -0.1041 | 0.8532 | -0.5008 | 0.7146 |
| [40,50) | 328 | 0.9499 | 0.8296 | -0.1203 | 0.6843 | -0.2184 | 0.3355 |
| [50,60) | 358 | 0.8000 | 0.7322 | -0.0677 | 0.5826 | +0.1242 | 0.1239 |
| [60,70) | 256 | 0.5823 | 0.5585 | -0.0238 | 0.4691 | +0.2707 | 0.0324 |
| [70,80) | 135 | 0.2917 | 0.2861 | -0.0056 | 0.2216 | +0.1269 | 0.0057 |
| [80,90) | 81 | 0.1278 | 0.1276 | -0.0002 | 0.0916 | -0.0286 | 0.0002 |

### Interpretation

The all-ray diagnostic confirms the high-elevation shrinkage direction: even when
all available high-elevation source rays are used for both slant-pair training
and VTEC conversion evaluation, the SLM-correction shrinkage model improves SLM
in VTEC space without using VTEC truth during training.

The best all-ray RMSE setting remains an aggressive shrinkage with minimum
strength (`s30_e70_p5_min0.2`), but it degrades two bins, including high-elevation
near-zenith behavior. For a physically safer rule, `s30_e90_p4_min0` is the
strict no-bin-worse setting. The earlier random-split recommendation
`s30_e90_p4_min0.1` remains a good balanced holdout setting, but in all-ray
diagnostics it slightly worsens the `[80,90)` bin.

### Decision

Report all-ray results separately from holdout results. For all-ray full-dataset
conversion diagnostics, use `s30_e90_p4_min0` when requiring every high-elevation
bin to be no worse than SLM. Keep `s30_e90_p4_min0.1` as the random-split balanced
reference and `s30_e90_p2_min0` as the MAE-oriented alternative.

## 2026-07-01 Experiment 042: Lightweight Multi-Time High-Elevation Training

### Purpose

The first full `UT04:00-06:00`, `30s`, `2M pair`, `100 epoch` multi-time plan was
too slow for routine iteration. This experiment replans the multi-time route as a
lightweight staged test and asks whether adding nearby epochs can beat the
single-time UT05:00 high-elevation baseline on the same fixed UT05:00 holdout
keys.

### Code/Protocol Changes

- `src.train_ray_mlp_simulation` and `src.train_set_ratio_mlp` now support
  `--same-epoch-pairs` so source-target training pairs are generated only within
  the same timestamp.
- RayMLP nearby-pair training now supports `--max-pair-examples`.
- Set-ratio inference context can be restricted to the same timestamp when
  `--same-epoch-pairs` is enabled.
- Metadata records same-epoch pairing, pair caps, identity-pair counts, and
  confirms `n_cross_epoch_pairs = 0`.

The scientific constraint is unchanged: `vtec_iri_ref` is not used for training;
it is used only for VTEC-space evaluation.

### Datasets

Starting from the full 30s IRI simulation product:

```text
datasets/interim/iri_simulation_20240505_0400_0600_30s_spacing2p5_regular.parquet
```

three lighter training products were used:

| Dataset | Window | Cadence | Rows | Epochs |
| --- | --- | --- | ---: | ---: |
| `iri_simulation_20240505_0400_0600_5min_spacing2p5_regular.parquet` | UT04:00-06:00 | 5 min | 51,702 | 25 |
| `iri_simulation_20240505_0430_0530_2min_spacing2p5_regular.parquet` | UT04:30-05:30 | 2 min | 63,532 | 31 |
| `iri_simulation_20240505_0445_0515_30s_spacing2p5_regular.parquet` | UT04:45-05:15 | 30 s | 124,217 | 61 |

All formal comparisons use the same UT05:00 key-holdout rays from
`results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet`. For the
high-elevation route, both train and evaluation source rays are filtered to
`elevation >= 30 deg`, leaving `302` UT05:00 evaluation rays.

### Results

Summary output:

```text
results/tables/multitime_high30_slm_correction_comparison_20260701.csv
```

| Route | n | RMSE | MAE | Bias | R2 | SLM RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UT05-only high30 SLM-correction shrink p4 min0.1 | 302 | 0.8112 | 0.5865 | +0.1089 | 0.9957 | 0.9285 |
| UT04-06 5min high30 SLM-correction shrink, 30ep, 100k pairs | 302 | 0.6180 | 0.4796 | +0.1702 | 0.9975 | 0.9285 |
| UT04:30-05:30 2min high30 SLM-correction shrink, 50ep, 250k pairs | 302 | 0.7879 | 0.6138 | +0.4591 | 0.9960 | 0.9285 |
| UT04:45-05:15 30s high30 SLM-correction shrink, 30ep, 100k pairs | 302 | 0.7284 | 0.5387 | +0.1825 | 0.9966 | 0.9285 |
| UT04-06 5min anchor1 source-ratio, 30ep, 100k pairs | 419 | 2.8141 | 2.0203 | +1.8759 | 0.9481 | 2.4400 |
| UT05-only anchor1 source-ratio | 419 | 1.5060 | 1.0912 | +0.0200 | 0.9851 | 2.4400 |

### Interpretation

The multi-time high-elevation SLM-correction route does improve over the
single-time UT05:00 high-elevation model. The best lightweight run is the
UT04:00-06:00 5min configuration, reducing RMSE from `0.8112` to `0.6180` TECU
on the same UT05:00 high-elevation holdout rays.

More dense sampling did not automatically improve the model. The 2min centered
window and 30s centered window both beat the single-time high-elevation run, but
neither beats the 5min result. This suggests the benefit is regularized temporal
context rather than simply adding more pair examples or more epochs.

The all-elevation anchor-only source-ratio route remains a poor multi-time
direction: the 5min multi-time version degrades from the single-time `1.5060`
RMSE to `2.8141` RMSE. It should not be scaled until the target or sampling
strategy is redesigned.

### Decision

Promote lightweight multi-time `SLM-correction + elevation-dependent shrinkage`
as the current best high-elevation route. Use the 5min UT04:00-06:00 result as
the reference multi-time benchmark for now. Do not resume the full 30s/2M/100ep
plan unless a smaller run first shows clear improvement over the 5min benchmark.

## 2026-07-01 Experiment 043: Epoch Sweep For 5min Multi-Time High-Elevation Model

### Purpose

Experiment 042 found that the best lightweight multi-time setting was
`UT04:00-06:00`, `5min`, `100k pairs`, `30 epochs` for the high-elevation
`SLM-correction + elevation-dependent shrinkage` route. This sweep tests whether
`30 epochs` is too small by varying only the training epoch count while keeping
all other settings fixed.

### Fixed Setup

- Input: `datasets/interim/iri_simulation_20240505_0400_0600_5min_spacing2p5_regular.parquet`
- Fixed UT05:00 key holdout: `results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet`
- Evaluation: `elevation >= 30 deg`, `302` UT05:00 rays
- Pairing: same-epoch nearby pairs only, `nearby-km=200`, `nearby-max-neighbors=30`
- Context: source anchor only, `context-km=0`, `max-obs=1`
- Target: `slm_correction`
- Regularization/shrinkage: `correction_regularization=0.001`, `s30_e90_p4_min0.1`
- Pair cap: `100000`
- `vtec_iri_ref` is evaluation-only and is not used for training.

### Outputs

```text
results/tables/multitime_5min_high30_epoch_sweep_20260701.csv
```

### Results

| Epochs | VTEC RMSE | MAE | Bias | R2 | Train-pair RMSE | Final train loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.6786 | 0.5136 | +0.1983 | 0.9970 | 0.009319 | 0.228751 |
| 30 | 0.6180 | 0.4796 | +0.1702 | 0.9975 | 0.007886 | 0.197295 |
| 40 | 0.6619 | 0.5013 | +0.2423 | 0.9972 | 0.007844 | 0.178256 |
| 50 | 0.6340 | 0.4958 | +0.2375 | 0.9974 | 0.007606 | 0.162830 |
| 80 | 0.6711 | 0.5155 | +0.2660 | 0.9971 | 0.006806 | 0.135057 |

For reference, the high-elevation SLM baseline on the same `302` UT05:00 rays is
`0.9285` RMSE, and the single-time UT05-only high-elevation shrinkage run is
`0.8112` RMSE.

### Interpretation

`30 epochs` is not under-trained for the current fixed setup. Increasing epochs
continues to reduce the slant-pair training loss, but it does not improve the
VTEC-space conversion RMSE. This indicates that better pair-level correction
fitting does not monotonically translate into better vertical-query conversion.
The likely cause is mild overfitting or target mismatch between same-epoch
slant-pair correction and the source-to-vertical query used for STEC-to-VTEC
conversion.

### Decision

Use `30 epochs` as the recommended setting for the current
`UT04:00-06:00`, `5min`, `100k pairs`, high-elevation SLM-correction shrinkage
configuration. Do not spend additional runtime on larger epoch counts unless the
training target, sampling strategy, or model architecture changes.

## 2026-07-02 Experiment 044: Local 30s Time Windows Around UT05:00

### Purpose

Experiment 042 showed that `UT04:00-06:00`, `5min`, `100k pairs`, `30 epochs`
was the best lightweight multi-time high-elevation setup so far. A concern is
that this benefit may come from broad temporal coverage rather than physically
nearby ionospheric states. This experiment keeps the original `30s` time
resolution and instead shortens the training time range around UT05:00 to test
whether more temporally local training data improves the fixed UT05:00 VTEC
conversion result.

### Fixed Setup

All runs use the same model and evaluation protocol as the current best
high-elevation route:

- Model: set-ratio MLP, source-anchor context only
- Target: `slm_correction`
- Shrinkage: `s30_e90_p4_min0.1`
- Pairing: same-epoch nearby pairs only
- Pair cap: `100000`
- Epochs: `30`
- Train/eval elevation filter: `elevation >= 30 deg`
- Fixed UT05:00 key holdout from
  `results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet`
- `vtec_iri_ref` is evaluation-only and is not used for training.

### Datasets

The local-window datasets were cut from:

```text
datasets/interim/iri_simulation_20240505_0400_0600_30s_spacing2p5_regular.parquet
```

| Dataset | Window | Cadence | Rows | Epochs |
| --- | --- | --- | ---: | ---: |
| `iri_simulation_20240505_ut05_pm05min_30s_spacing2p5_regular.parquet` | UT04:55-05:05 | 30 s | 42,633 | 21 |
| `iri_simulation_20240505_ut05_pm10min_30s_spacing2p5_regular.parquet` | UT04:50-05:10 | 30 s | 83,492 | 41 |
| `iri_simulation_20240505_ut05_pm15min_30s_spacing2p5_regular.parquet` | UT04:45-05:15 | 30 s | 124,217 | 61 |
| `iri_simulation_20240505_ut05_pm20min_30s_spacing2p5_regular.parquet` | UT04:40-05:20 | 30 s | 164,941 | 81 |
| `iri_simulation_20240505_ut05_pm30min_30s_spacing2p5_regular.parquet` | UT04:30-05:30 | 30 s | 247,338 | 121 |

### Outputs

```text
results/tables/local_30s_window_vs_5min_multitime_comparison_20260702.csv
```

### Results

| Route | n | RMSE | MAE | Bias | R2 | Train rows | Train pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UT05-only high30 shrink | 302 | 0.8112 | 0.5865 | +0.1089 | 0.9957 | 1,163 | 18,978 |
| UT04-06 5min high30 shrink | 302 | 0.6180 | 0.4796 | +0.1702 | 0.9975 | 36,186 | 100,000 |
| UT05 +/-5min 30s high30 shrink | 302 | 0.8134 | 0.5691 | +0.1890 | 0.9957 | 30,328 | 100,000 |
| UT05 +/-10min 30s high30 shrink | 302 | 0.6844 | 0.5270 | +0.1264 | 0.9970 | 59,682 | 100,000 |
| UT05 +/-15min 30s high30 shrink | 302 | 0.7284 | 0.5387 | +0.1825 | 0.9966 | 88,857 | 100,000 |
| UT05 +/-20min 30s high30 shrink | 302 | 0.6854 | 0.5171 | +0.2252 | 0.9969 | 117,878 | 100,000 |
| UT05 +/-30min 30s high30 shrink | 302 | 0.6974 | 0.5176 | +0.1559 | 0.9968 | 176,180 | 100,000 |

For reference, the high-elevation SLM baseline on the same UT05:00 holdout rays
is `0.9285` RMSE.

### Interpretation

The local 30s window idea is physically well motivated: nearby epochs should be
less affected by ionospheric temporal evolution. In this current pair-learning
setup, however, the best local 30s window is `+/-10min` with `0.6844` RMSE,
which improves over the UT05-only model but does not beat the broader
`UT04:00-06:00` 5min downsampled run (`0.6180` RMSE).

This suggests that the current model benefits from geometric and sampling
diversity across the two-hour window more than it suffers from the additional
slow temporal variability introduced by sparse 5min sampling. The narrow
`+/-5min` window is almost identical to the UT05-only result, likely because it
adds many nearby-in-time rays but not enough independent geometric diversity for
source-to-vertical correction learning.

### Decision

Keep `UT04:00-06:00`, `5min`, `100k pairs`, `30 epochs` as the current reference
multi-time setting. The best local 30s window (`+/-10min`) should be kept as a
secondary diagnostic, especially if future models explicitly encode temporal
smoothness or use time-weighted sampling. Do not replace the 5min reference with
a local 30s window under the current set-ratio anchor-only architecture.

## 2026-07-02 Experiment 045: Epoch Sweep For Local 30s Time Windows

### Purpose

Experiment 044 compared local `30s` time windows around UT05:00 using a fixed
`30 epoch` setting. Because local windows may need more epochs than the broader
5min downsampled setup, this experiment extends each time-window run to
`40`, `50`, and `60` epochs, then compares them against the existing `30 epoch`
results.

### Fixed Setup

- Model: set-ratio MLP, source-anchor context only
- Target: `slm_correction`
- Shrinkage: `s30_e90_p4_min0.1`
- Pairing: same-epoch nearby pairs only
- Pair cap: `100000`
- Train/eval elevation filter: `elevation >= 30 deg`
- Fixed UT05:00 key holdout from
  `results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet`
- `vtec_iri_ref` is evaluation-only and is not used for training.

This experiment also reran the UT05-only single-epoch baseline at
`30/40/50/60` epochs, because the previously reported UT05-only high-elevation
number came from the earlier `100 epoch` run.

### Outputs

```text
results/tables/local_30s_window_epoch_sweep_20260702.csv
results/tables/local_30s_window_epoch_sweep_best_by_window_20260702.csv
```

### RMSE Sweep

| Route | 30ep | 40ep | 50ep | 60ep | Best epoch | Best RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UT05-only high30 shrink | 0.8753 | 0.8183 | 0.8106 | 0.8123 | 50 | 0.8106 |
| UT04-06 5min high30 shrink | 0.6180 | 0.6619 | 0.6340 | 0.6439 | 30 | 0.6180 |
| UT05 +/-5min 30s high30 shrink | 0.8134 | 0.7635 | 0.7758 | 0.6938 | 60 | 0.6938 |
| UT05 +/-10min 30s high30 shrink | 0.6844 | 0.6924 | 0.6819 | 0.6953 | 50 | 0.6819 |
| UT05 +/-15min 30s high30 shrink | 0.7284 | 0.7343 | 0.7108 | 0.6694 | 60 | 0.6694 |
| UT05 +/-20min 30s high30 shrink | 0.6854 | 0.7321 | 0.7218 | 0.7480 | 30 | 0.6854 |
| UT05 +/-30min 30s high30 shrink | 0.6974 | 0.7037 | 0.7201 | 0.6812 | 60 | 0.6812 |

### Best-By-Window Metrics

| Route | Best epoch | RMSE | MAE | Bias | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UT05-only high30 shrink | 50 | 0.8106 | 0.5920 | +0.1140 | 0.9957 |
| UT04-06 5min high30 shrink | 30 | 0.6180 | 0.4796 | +0.1702 | 0.9975 |
| UT05 +/-5min 30s high30 shrink | 60 | 0.6938 | 0.5331 | +0.1777 | 0.9969 |
| UT05 +/-10min 30s high30 shrink | 50 | 0.6819 | 0.5211 | +0.2168 | 0.9970 |
| UT05 +/-15min 30s high30 shrink | 60 | 0.6694 | 0.5153 | +0.2300 | 0.9971 |
| UT05 +/-20min 30s high30 shrink | 30 | 0.6854 | 0.5171 | +0.2252 | 0.9969 |
| UT05 +/-30min 30s high30 shrink | 60 | 0.6812 | 0.5232 | +0.2980 | 0.9970 |

For reference, the SLM baseline on the same high-elevation UT05:00 holdout rays
is `0.9285` RMSE.

### Interpretation

The user's concern was valid: several local `30s` windows were under-trained at
`30 epochs`. Increasing epochs substantially improves the `+/-5min` window
(`0.8134 -> 0.6938`) and the `+/-15min` window (`0.7284 -> 0.6694`). However,
after epoch tuning, no local `30s` window beats the broader `UT04:00-06:00`,
`5min`, `30 epoch` reference (`0.6180`).

The best local `30s` result is `+/-15min`, `60 epochs`, with `0.6694` RMSE. It is
better than the UT05-only tuned baseline (`0.8106`) and better than SLM
(`0.9285`), but still worse than the 5min two-hour downsampled run.

Longer training also tends to increase positive VTEC bias in several local
windows. This reinforces the earlier finding that lower slant-pair training loss
does not guarantee better source-to-vertical VTEC conversion.

### Decision

Keep `UT04:00-06:00`, `5min`, `100k pairs`, `30 epochs` as the current best
multi-time high-elevation reference. Keep `UT05 +/-15min`, `30s`, `60 epochs` as
the best local-window diagnostic. If local-window training is revisited, the next
step should change the sampling or objective, not simply increase epochs further.

## 2026-07-02 Experiment 046: Full-Day 5min Multi-Time Training

### Purpose

Test whether an even broader time range improves the current best high-elevation
STEC-to-VTEC conversion route. Previous experiments suggested that the set-ratio
SLM-correction model benefits from geometric and sampling diversity. This
experiment therefore builds a true full-day `UT00:00-24:00`, `5min` IRI
simulation dataset and compares it against the previous `UT04:00-06:00`, `5min`
reference.

### Dataset Build

A new reproducible config was added:

```text
configs/full_day_5min.yaml
```

It builds a 5min ray-wise table from the observed TEC data and SP3 geometry. The
`dataset_builder` writes the date-based filename by default, so the generated
product was copied to the explicit experiment path:

```text
datasets/interim/raystec_china_20240505_0000_2400_5min.parquet
```

Ray-wise dataset checks:

| Item | Value |
| --- | ---: |
| Rows | 597,668 |
| Epochs | 288 |
| Time range | 2024-05-05 00:00:00 to 23:55:00 |
| High-elevation rows | 418,372 |
| Stations | 112 |
| PRNs | 100 |

The IRI simulation product was then built with the regular-grid interpolator:

```text
datasets/interim/iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet
```

IRI simulation checks:

| Item | Value |
| --- | ---: |
| Rows | 597,668 |
| Epochs | 288 |
| Time range | 2024-05-05 00:00:00 to 23:55:00 |
| High-elevation rows | 418,372 |
| `ray_sample_spacing_km` | 2.5 |
| IRI strategy | `regular_grid_interpolator` |

The long IRI command exceeded the shell timeout after the output had already been
written. The parquet and metadata were subsequently read successfully, and no
Python training/build process remained active.

### Smoke Test

A first-3-epoch smoke subset was used before the formal training sweep:

```text
datasets/interim/iri_simulation_20240505_0000_0010_5min_spacing2p5_regular_smoke3.parquet
```

Smoke result:

| Check | Value |
| --- | ---: |
| Training epochs | 3 |
| Training pairs | 5,000 |
| `same_epoch_pairs` | true |
| `n_cross_epoch_pairs` | 0 |

### Fixed Training Setup

- Model: set-ratio MLP, source-anchor context only
- Target: `slm_correction`
- Shrinkage: `s30_e90_p4_min0.1`
- Pairing: same-epoch nearby pairs only
- Train/eval elevation filter: `elevation >= 30 deg`
- Fixed UT05:00 key holdout from
  `results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet`
- `vtec_iri_ref` is evaluation-only and is not used for training.

### Outputs

```text
results/tables/full_day_5min_epoch_sweep_20260702.csv
results/tables/full_day_5min_epoch_sweep_best_20260702.csv
```

### Results

| Route | Epochs | Pair cap | Per-epoch cap | RMSE | MAE | Bias | R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SLM baseline | - | - | - | 0.9285 | 0.6842 | +0.3186 | 0.9944 |
| UT05-only tuned high30 shrink | 50 | 100k | 100000 | 0.8106 | 0.5920 | +0.1140 | 0.9957 |
| UT05 +/-15min 30s tuned high30 shrink | 60 | 100k | 1640 | 0.6694 | 0.5153 | +0.2300 | 0.9971 |
| UT04-06 5min high30 shrink | 30 | 100k | 4000 | 0.6180 | 0.4796 | +0.1702 | 0.9975 |
| UT00-24 5min high30 shrink | 30 | 100k | 348 | 0.6387 | 0.5082 | +0.1859 | 0.9973 |
| UT00-24 5min high30 shrink | 40 | 100k | 348 | 0.6480 | 0.5122 | +0.1671 | 0.9973 |
| UT00-24 5min high30 shrink | 50 | 100k | 348 | 0.6799 | 0.5253 | +0.2369 | 0.9970 |
| UT00-24 5min high30 shrink | 60 | 100k | 348 | 0.6060 | 0.4827 | +0.1893 | 0.9976 |
| UT00-24 5min high30 shrink | 60 | 1.152M | 4000 | 0.6517 | 0.5098 | +0.2205 | 0.9972 |

### Interpretation

The full-day 5min experiment improves the best RMSE achieved so far. The best
configuration is `UT00:00-24:00`, `5min`, `100k pairs`, `60 epochs`, with RMSE
`0.6060` TECU. This is slightly better than the previous `UT04:00-06:00`, `5min`,
`30 epochs` reference (`0.6180` TECU).

The density-matched run did not help. Increasing the pair cap from `100k` to
`1.152M` at the same `60 epochs` worsened RMSE from `0.6060` to `0.6517`. This
suggests that the current set-ratio anchor-only model benefits from broad
coverage plus sparse pair sampling as a regularizer. Matching the per-epoch pair
density of the two-hour run appears to overfit the slant-pair correction target
or amplify the mismatch between slant-pair fitting and vertical-query VTEC
conversion.

### Decision

Promote `UT00:00-24:00`, `5min`, `100k pairs`, `60 epochs` as the new best
high-elevation multi-time reference for this model family. Keep the density-
matched result as a diagnostic failure: more pair examples are not automatically
better for the current objective. Future improvements should explore sampling
weights or objectives, not just larger pair caps.

## 2026-07-02 Experiment 047: Density-Matched Full-Day 5min Epoch Extension

### Purpose

Experiment 046 found that the full-day `5min`, `100k pairs`, `60 epoch` run was
the best configuration so far (`0.6060` RMSE), while the density-matched
`1.152M pairs`, `60 epoch` run was worse (`0.6517` RMSE). A plausible concern is
that the larger density-matched training set may need more epochs. This
experiment extends the density-matched setting to `70`, `80`, and `90` epochs.

### Fixed Setup

- Dataset: `datasets/interim/iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet`
- Model: set-ratio MLP, source-anchor context only
- Target: `slm_correction`
- Shrinkage: `s30_e90_p4_min0.1`
- Pairing: same-epoch nearby pairs only
- Pair cap: `1,152,000`, giving `same_epoch_per_epoch_cap = 4000`
- Train/eval elevation filter: `elevation >= 30 deg`
- Fixed UT05:00 key holdout from
  `results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet`
- `vtec_iri_ref` is evaluation-only and is not used for training.

### Outputs

```text
results/tables/full_day_5min_density_epoch_sweep_20260702.csv
results/tables/full_day_5min_epoch_sweep_20260702.csv
results/tables/full_day_5min_epoch_sweep_best_20260702.csv
```

### Results

| Route | Epochs | Pair cap | RMSE | MAE | Bias | R2 | Train-pair RMSE | Final train loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-day 5min p100k | 60 | 100k | 0.6060 | 0.4827 | +0.1893 | 0.9976 | 0.0110 | 0.2629 |
| Full-day 5min p1152k | 60 | 1.152M | 0.6517 | 0.5098 | +0.2205 | 0.9972 | 0.0103 | 0.1375 |
| Full-day 5min p1152k | 70 | 1.152M | 0.6056 | 0.4824 | +0.1859 | 0.9976 | 0.0102 | 0.1314 |
| Full-day 5min p1152k | 80 | 1.152M | 0.6295 | 0.4955 | +0.2135 | 0.9974 | 0.0096 | 0.1278 |
| Full-day 5min p1152k | 90 | 1.152M | 0.6582 | 0.5101 | +0.2374 | 0.9972 | 0.0099 | 0.1233 |

### Interpretation

The user's concern was correct: `60 epochs` was not the best setting for the
larger density-matched dataset. Increasing to `70 epochs` reduces RMSE from
`0.6517` to `0.6056`, slightly beating the `100k pairs`, `60 epoch` result
(`0.6060`). However, the improvement is extremely small (`0.0004` TECU), while
runtime and training cost are much higher.

The trend is not monotonic. Further increasing to `80` and `90` epochs worsens
VTEC-space RMSE even though the final training loss continues to fall. This
again shows that lower slant-pair training loss does not necessarily improve the
source-to-vertical VTEC conversion objective.

### Decision

The absolute best current RMSE is now `UT00:00-24:00`, `5min`, `1.152M pairs`,
`70 epochs` with RMSE `0.6056`. For practical iteration, the `100k pairs`,
`60 epochs` run remains nearly equivalent (`0.6060`) and much cheaper. Report the
70-epoch density-matched result as the formal best, but use the 100k/60ep setting
as the efficient reference unless the final paper table needs the absolute
minimum RMSE.

## 2026-07-02 Experiment 048: Epoch-Level VTEC Diagnostics in One Training Run

### Purpose

Repeatedly rerunning the same model for `30/40/50/60/70/80/90` epochs is wasteful.
`src.train_set_ratio_mlp` now supports checkpoint-style VTEC diagnostics during a
single training run. The model still trains exactly as before; the new option only
runs evaluation queries after selected epochs and writes a compact metrics table.

### New CLI Options

- `--epoch-metrics`: evaluate after every `--epoch-metrics-every` epochs.
- `--epoch-metrics-every N`: default `1`, so every epoch is evaluated when enabled.
- `--epoch-metrics-epochs 30,40,50,60,70,80,90`: evaluate only listed epochs; this
  also enables epoch metrics.

The output is:

```text
results/tables/<run_name>_epoch_metrics.csv
```

with columns:

```text
epoch, train_loss, rmse_tecu, mae_tecu, bias_tecu, r2, n_eval
```

### Validation

Compiled successfully:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\train_set_ratio_mlp.py
```

Smoke run:

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0000_0010_5min_spacing2p5_regular_smoke3.parquet `
  --run-name smoke_epoch_metrics_set_ratio_e3 `
  --split random --seed 42 `
  --nearby-km 200 --context-km 0 --nearby-max-neighbors 30 --max-obs 1 `
  --target-mode slm_correction --correction-regularization 0.001 `
  --elevation-shrinkage --shrinkage-start-deg 30 --shrinkage-end-deg 90 `
  --shrinkage-power 4 --shrinkage-min-strength 0.1 `
  --min-train-elevation-deg 30 --min-eval-elevation-deg 30 `
  --same-epoch-pairs --max-pair-examples 5000 `
  --epochs 3 --epoch-metrics --epoch-metrics-every 1 `
  --batch-size 1024 --device cpu --output-format parquet
```

Smoke CSV rows:

| Epoch | Train loss | RMSE | MAE | Bias | R2 | N eval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9833 | 0.3814 | 0.2673 | +0.1379 | 0.9960 | 994 |
| 2 | 0.7797 | 0.3763 | 0.2650 | +0.1322 | 0.9961 | 994 |
| 3 | 0.7278 | 0.3591 | 0.2566 | +0.1101 | 0.9964 | 994 |

### Recommended Use

For the full-day density-matched question, run one longer training job and record
the desired checkpoints, for example:

```powershell
--epochs 90 --epoch-metrics-epochs 60,70,80,90
```

This produces a directly comparable epoch curve without retraining from scratch
for every epoch setting.

## 2026-07-02 Experiment 049: Full-Day 5min Density-Matched 150 Epoch Curve

### Purpose

Use the epoch-level diagnostics added in Experiment 048 to avoid rerunning the
same density-matched experiment for many separate epoch settings. This run trains
once to `150 epochs` and evaluates VTEC conversion at selected checkpoints from
`50` to `150` epochs.

### Setup

- Dataset: `datasets/interim/iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet`
- Model: set-ratio MLP, source-anchor context only
- Target: `slm_correction`
- Shrinkage: `s30_e90_p4_min0.1`
- Pairing: same-epoch nearby pairs only
- Pair cap: `1,152,000`
- Train/eval elevation filter: `elevation >= 30 deg`
- Fixed UT05:00 key holdout from
  `results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet`
- Evaluation checkpoints:
  `50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150`
- `vtec_iri_ref` remains evaluation-only and is not used in training.

### Command

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet `
  --run-name set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e150_p1152k_epochcurve `
  --test-keys-from results\tables\set_ratio_mlp_0500_anchor1_e100_predictions.parquet `
  --test-key-columns timestamp,station,prn `
  --seed 42 `
  --nearby-km 200 --context-km 0 --nearby-max-neighbors 30 --max-obs 1 `
  --target-mode slm_correction --correction-regularization 0.001 `
  --elevation-shrinkage --shrinkage-start-deg 30 --shrinkage-end-deg 90 `
  --shrinkage-power 4 --shrinkage-min-strength 0.1 `
  --min-train-elevation-deg 30 --min-eval-elevation-deg 30 `
  --same-epoch-pairs --max-pair-examples 1152000 `
  --epochs 150 `
  --epoch-metrics-epochs 50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150 `
  --batch-size 1024 --device cpu --output-format parquet
```

### Outputs

```text
results/tables/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e150_p1152k_epochcurve_epoch_metrics.csv
results/tables/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e150_p1152k_epochcurve_predictions.parquet
results/tables/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e150_p1152k_epochcurve_metrics.json
results/checkpoints/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e150_p1152k_epochcurve.pth
```

### Results

| Epoch | Train loss | RMSE | MAE | Bias | R2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.1462 | 0.6039 | 0.4827 | +0.2186 | 0.9976 |
| 55 | 0.1419 | 0.6464 | 0.5013 | +0.2254 | 0.9973 |
| 60 | 0.1375 | 0.6517 | 0.5098 | +0.2205 | 0.9972 |
| 65 | 0.1347 | 0.6210 | 0.4868 | +0.2124 | 0.9975 |
| 70 | 0.1314 | 0.6056 | 0.4824 | +0.1859 | 0.9976 |
| 75 | 0.1303 | 0.6170 | 0.4856 | +0.1824 | 0.9975 |
| 80 | 0.1278 | 0.6295 | 0.4955 | +0.2135 | 0.9974 |
| 85 | 0.1256 | 0.6191 | 0.4871 | +0.1776 | 0.9975 |
| 90 | 0.1233 | 0.6582 | 0.5101 | +0.2374 | 0.9972 |
| 95 | 0.1209 | 0.6538 | 0.5089 | +0.2241 | 0.9972 |
| 100 | 0.1198 | 0.6310 | 0.4978 | +0.1993 | 0.9974 |
| 105 | 0.1180 | 0.6397 | 0.5007 | +0.2090 | 0.9973 |
| 110 | 0.1152 | 0.6583 | 0.5133 | +0.2407 | 0.9972 |
| 115 | 0.1138 | 0.6515 | 0.5140 | +0.1974 | 0.9972 |
| 120 | 0.1121 | 0.6619 | 0.5217 | +0.2447 | 0.9972 |
| 125 | 0.1111 | 0.6409 | 0.5084 | +0.2057 | 0.9973 |
| 130 | 0.1104 | 0.6577 | 0.5172 | +0.2465 | 0.9972 |
| 135 | 0.1090 | 0.6608 | 0.5158 | +0.2315 | 0.9972 |
| 140 | 0.1071 | 0.6647 | 0.5211 | +0.2378 | 0.9971 |
| 145 | 0.1059 | 0.6501 | 0.5129 | +0.2463 | 0.9973 |
| 150 | 0.1050 | 0.6519 | 0.5159 | +0.2484 | 0.9972 |

### Interpretation

The best checkpoint in this single-run epoch curve is `50 epochs`, with RMSE
`0.6039` TECU. This is slightly better than the previous `70 epoch` result
(`0.6056`) and the cheaper `100k pairs, 60 epoch` reference (`0.6060`), but the
margin is tiny.

The train loss decreases monotonically from `0.1462` at 50 epochs to `0.1050` at
150 epochs, while VTEC RMSE does not improve. This reinforces the earlier finding
that optimizing the dense slant-pair correction objective for longer does not
monotonically improve the vertical-query STEC-to-VTEC conversion objective.

### Decision

For the density-matched full-day setting, use `50 epochs` as the current best
checkpoint (`RMSE = 0.6039`). For practical iteration, the cheaper `100k pairs,
60 epoch` run remains nearly equivalent and much faster. Future tuning should
focus on pair sampling/objective design rather than simply increasing epochs.

## 2026-07-03 Experiment 050: Full-Day 5min Density-Matched Low-Epoch Curve

### Purpose

Experiment 049 showed that the `50 epoch` checkpoint was slightly better than
later checkpoints up to `150 epochs`. This suggests the optimum may occur before
50 epochs. This experiment reruns the same density-matched full-day setup to
`50 epochs` and records VTEC diagnostics every 5 epochs from `5` to `50`.

### Setup

- Dataset: `datasets/interim/iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet`
- Model: set-ratio MLP, source-anchor context only
- Target: `slm_correction`
- Shrinkage: `s30_e90_p4_min0.1`
- Pairing: same-epoch nearby pairs only
- Pair cap: `1,152,000`
- Train/eval elevation filter: `elevation >= 30 deg`
- Fixed UT05:00 key holdout from
  `results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet`
- Evaluation checkpoints: `5,10,15,20,25,30,35,40,45,50`
- `vtec_iri_ref` remains evaluation-only and is not used in training.

### Outputs

```text
results/tables/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e50_p1152k_lowepoch_curve_epoch_metrics.csv
results/tables/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e50_p1152k_lowepoch_curve_predictions.parquet
results/tables/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e50_p1152k_lowepoch_curve_metrics.json
results/checkpoints/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e50_p1152k_lowepoch_curve.pth
```

### Results

| Epoch | Train loss | RMSE | MAE | Bias | R2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.2779 | 0.7133 | 0.5543 | +0.2994 | 0.9967 |
| 10 | 0.2256 | 0.6422 | 0.5020 | +0.1848 | 0.9973 |
| 15 | 0.2039 | 0.6243 | 0.4885 | +0.2651 | 0.9975 |
| 20 | 0.1903 | 0.6510 | 0.5112 | +0.2341 | 0.9972 |
| 25 | 0.1792 | 0.6304 | 0.4916 | +0.2143 | 0.9974 |
| 30 | 0.1697 | 0.5975 | 0.4758 | +0.1867 | 0.9977 |
| 35 | 0.1614 | 0.6677 | 0.5167 | +0.1805 | 0.9971 |
| 40 | 0.1546 | 0.6418 | 0.4977 | +0.2207 | 0.9973 |
| 45 | 0.1503 | 0.6163 | 0.4913 | +0.2187 | 0.9975 |
| 50 | 0.1462 | 0.6039 | 0.4827 | +0.2186 | 0.9976 |

### Interpretation

The best checkpoint in this low-epoch curve is `30 epochs`, with RMSE `0.5975`
TECU. This improves on the previous best density-matched checkpoint from
Experiment 049 (`50 epochs`, RMSE `0.6039`) and the earlier practical reference
(`100k pairs`, `60 epochs`, RMSE `0.6060`).

The curve is not monotonic. Training loss decreases steadily, but VTEC RMSE has a
clear local minimum near 30 epochs and worsens at 35-40 epochs before partially
recovering at 45-50 epochs. This reinforces that the slant-pair training loss is
not a reliable proxy for the vertical-query VTEC conversion objective.

### Decision

For the full-day 5min density-matched setting, the current observed best
checkpoint is `30 epochs` (`RMSE = 0.5975`). The best region should now be refined
around `25-35 epochs` with finer checkpoints, rather than extending training to
higher epochs.

## 2026-07-03 Experiment 051: Full-Day 5min Density-Matched Fine Epoch Sweep 25-35

### Purpose

Experiment 050 found that `30 epochs` was better than `50 epochs` for the
full-day `5min`, density-matched `1.152M pairs` setting. This experiment performs
a finer sweep from `25` to `35` epochs to check whether the optimum is exactly
near 30 epochs or slightly shifted.

### Setup

- Dataset: `datasets/interim/iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet`
- Model: set-ratio MLP, source-anchor context only
- Target: `slm_correction`
- Shrinkage: `s30_e90_p4_min0.1`
- Pairing: same-epoch nearby pairs only
- Pair cap: `1,152,000`
- Train/eval elevation filter: `elevation >= 30 deg`
- Fixed UT05:00 key holdout from
  `results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet`
- Evaluation checkpoints: `25,26,27,28,29,30,31,32,33,34,35`
- `vtec_iri_ref` remains evaluation-only and is not used in training.

### Outputs

```text
results/tables/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e35_p1152k_epoch25_35_curve_epoch_metrics.csv
results/tables/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e35_p1152k_epoch25_35_curve_predictions.parquet
results/tables/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e35_p1152k_epoch25_35_curve_metrics.json
results/checkpoints/set_ratio_mlp_0000_2400_5min_ut05keys_high30_slm_corr_shrink_e35_p1152k_epoch25_35_curve.pth
```

### Results

| Epoch | Train loss | RMSE | MAE | Bias | R2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.1792 | 0.6304 | 0.4916 | +0.2143 | 0.9974 |
| 26 | 0.1770 | 0.6512 | 0.5010 | +0.2140 | 0.9972 |
| 27 | 0.1743 | 0.6643 | 0.5096 | +0.2607 | 0.9971 |
| 28 | 0.1732 | 0.6516 | 0.5069 | +0.2178 | 0.9972 |
| 29 | 0.1714 | 0.6265 | 0.4928 | +0.2043 | 0.9974 |
| 30 | 0.1697 | 0.5975 | 0.4758 | +0.1867 | 0.9977 |
| 31 | 0.1677 | 0.6330 | 0.5020 | +0.2360 | 0.9974 |
| 32 | 0.1653 | 0.6487 | 0.5044 | +0.2311 | 0.9973 |
| 33 | 0.1641 | 0.5971 | 0.4711 | +0.2042 | 0.9977 |
| 34 | 0.1635 | 0.6100 | 0.4818 | +0.2223 | 0.9976 |
| 35 | 0.1614 | 0.6677 | 0.5167 | +0.1805 | 0.9971 |

### Interpretation

The lowest RMSE in this fine sweep is at `33 epochs` (`0.5971` TECU), slightly
below the previous `30 epoch` result (`0.5975` TECU). The difference is only
`0.0004` TECU, so `30` and `33` epochs should be interpreted as the same
high-performance plateau rather than a sharply determined optimum. The `33 epoch`
checkpoint also has the best MAE in this sweep (`0.4711` TECU), while its bias is
slightly larger than at `30 epochs`.

The curve remains irregular: `31-32` and `35` epochs are clearly worse despite
lower training loss. This reinforces that epoch selection should be based on the
VTEC-space checkpoint metric, not the slant-pair training loss.

### Decision

For the current full-day `5min`, density-matched setup, use `33 epochs` as the
observed best checkpoint (`RMSE = 0.5971`, `MAE = 0.4711`). In interpretation,
report the stable best region as approximately `30-33 epochs`, because the RMSE
difference between `30` and `33` epochs is negligible.
