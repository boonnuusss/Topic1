# Topic1 Ray-wise STEC-to-VTEC Pipeline

`Topic1` is the reproducible research pipeline for using ray-wise GNSS STEC
modeling to improve STEC-to-VTEC conversion. Each basic observation is one
receiver-satellite ray at one epoch:

```text
station + PRN + timestamp -> observed STEC
```

The main scientific target is not standalone STEC prediction. The target is to
test whether ray-wise STEC models, including RayMLP, can reduce VTEC conversion
error compared with the traditional single-layer mapping model.

## Modeling Route

The current route is:

1. Build a clean ray-wise STEC observation table.
2. Build the traditional single-layer mapping conversion baseline:
   `VTEC_slm = STEC_obs / M(elevation)`.
3. Use IRI-generated background ionosphere fields to run controlled simulation
   experiments with known reference VTEC/STEC relationships.
4. Train ray-wise ML STEC models, starting with RayMLP/FNN/GNN variants after
   the simulation labels and SLM baseline are stable.
5. Compare SLM and ML-assisted STEC-to-VTEC conversion in VTEC space, using
   metrics such as VTEC RMSE, MAE, bias, elevation-bin error, station-bin error,
   PRN-bin error, and spatial residual maps.

This order keeps the comparison anchored to the classical thin-shell
ionospheric projection model while making the evaluation target clear:
conversion accuracy from STEC to VTEC.

## Legacy Off-mainline Code

The following entry points were implemented during an earlier STEC-prediction
route and are retained for reference, smoke tests, or later reuse:

- `src.slm_baseline`: computes `STEC_slm = M(elevation) * VTEC`.
- `src.train_tree --mode direct`: direct Ridge/RandomForest prediction of
  `stec_obs`.
- `src.train_tree --mode residual`: residual learning of
  `stec_obs - stec_slm`.
- `src.train_mlp`: RayMLP residual learning on top of `stec_slm`.

They are not the current mainline unless a later experiment explicitly brings
them back into the STEC-to-VTEC conversion design.

## Agent Project Brain

New coding agents should read these files before making changes:

- `AGENT.md`: long-term rules for scientific coding agents.
- `RESEARCH_PLAN.md`: scientific question, hypotheses, datasets, and risks.
- `DATA_DICTIONARY.md`: processed table schema, units, and provenance.
- `IRI_SIMULATION_SCHEMA.md`: IRI-backed simulation targets, integration
  assumptions, SLM comparison columns, and metadata contract.
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

## SLM STEC-to-VTEC Baseline

The current physical baseline entry point is:

```powershell
.\.venv\Scripts\python.exe -m src.slm_vtec `
  --input datasets/processed/raystec_china_20240505.parquet `
  --output datasets/processed/rayvtec_china_20240505_slm.parquet `
  --shell-height-km 450
```

Equivalent wrapper:

```powershell
.\scripts\build_slm_vtec.ps1
```

The command writes:

```text
shell_height_km
slm_mapping_factor
vtec_slm
vtec_slm_source
```

The formula is:

```text
VTEC_slm = STEC_obs / M(elevation)
```

This is the traditional SLM conversion baseline. It does not predict STEC; it
maps observed STEC to a vertical TEC estimate.

## IRI Simulation Target

The next mainline step is to build an IRI-backed simulation experiment:

1. Generate or sample IRI background electron density/VTEC for IPP/time
   conditions.
2. Produce ray-wise simulated STEC or reference VTEC labels with known
   provenance.
3. Apply the SLM conversion baseline to estimate VTEC from STEC.
4. Train RayMLP-style STEC models under the same simulation setup.
5. Compare SLM and ML-assisted conversion errors against the IRI reference VTEC.

The detailed schema and metadata contract are defined in
`IRI_SIMULATION_SCHEMA.md`. It is based on the existing notebooks under
`../00_iri/mycode/ML_sTEC/`, especially `generate_sTEC_obs_iri_regional.ipynb`
and `gen_iri_val_dataset.ipynb`, but uses explicit row-wise parquet columns
instead of legacy npz field names.

The simulation builder entry point is:

```powershell
.\.venv\Scripts\python.exe -m src.iri_simulation_builder `
  --input datasets/processed/raystec_china_20240505.parquet `
  --output datasets/interim/iri_simulation_smoke.parquet `
  --max-rows 3 `
  --max-epochs 1 `
  --ray-sample-spacing-km 50
```

Start with small smoke tests. Full-day high-resolution ray integration can be
expensive, so the builder includes a per-epoch IRI query-point safety limit.

## VTEC-Space Evaluation

The current comparison endpoint is `python -m src.evaluate`. It treats
`vtec_iri_ref` as the default reference and `vtec_slm_from_iri_stec` as the SLM
baseline:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate `
  --input datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --table-dir results/tables `
  --run-name vtec_eval_20240505_0500_spacing2p5
```

The evaluator writes overall, elevation-bin, station-bin, UTC-hour, and merged
metric tables. The error convention is:

```text
error = VTEC_prediction - VTEC_reference
```

Future ML-assisted conversion results can be evaluated by passing either VTEC
prediction columns:

```powershell
--vtec-prediction-columns vtec_ray_mlp
```

or other STEC prediction columns, if that model is meant to be converted through
the same mapping factor:

```powershell
--stec-prediction-columns stec_other_model --mapping-column slm_mapping_factor_iri
```

## Current ML Simulation Entrypoints

- `python -m src.train_ray_mlp_simulation`: RayMLP ray-transformation STEC
  model.
- `python -m src.train_fnn_simulation`: compact FNN adapted from the local
  `FNN_sTEC_test.ipynb`.
- `python -m src.train_gnn_simulation`: pure-PyTorch GraphSAGE-style edge
  regressor adapted from `GNN_sTEC_improve.ipynb`.

All three train on simulated slant-ray `stec_iri_ray` labels only. They query a
vertical special ray after training and are scored against `vtec_iri_ref` only
in the evaluation step.

For the RayMLP simulation route below, evaluate `vtec_ray_mlp` as the final
ratio-converted VTEC prediction column. The raw vertical target-ray model output
is retained separately as `vtec_ray_mlp_direct` for diagnostics.

## RayMLP Simulation Training

The current mainline RayMLP entry point is:

```powershell
.\.venv\Scripts\python.exe -m src.train_ray_mlp_simulation `
  --input datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name ray_mlp_sim_20240505_0500_smoke `
  --epochs 5 `
  --max-rows 1000
```

Equivalent wrapper:

```powershell
.\scripts\train_ray_mlp_simulation.ps1 `
  -InputPath datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  -RunName ray_mlp_sim_20240505_0500_smoke `
  -Epochs 5 `
  -MaxRows 1000
```

This trains RayMLP only on simulated slant-ray STEC labels:

```text
source slant STEC + source geometry + target geometry -> target STEC
```

For training, the target geometry is the original slant ray and the label is
`stec_iri_ray`. For conversion/evaluation, the same trained STEC model is queried
with both the original slant target ray and a vertical target ray at the same
IPP/time. The output table includes `stec_ray_mlp`, `vtec_ray_mlp_direct`,
`ray_mlp_conversion_ratio`, and the final ratio-converted `vtec_ray_mlp`, which
is then evaluated by `src.evaluate`.

## FNN Simulation Training

The notebook-style FNN entry point is:

```powershell
.\.venv\Scripts\python.exe -m src.train_fnn_simulation `
  --input datasets/interim/iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --feature-set ecef6 `
  --epochs 500
```

It adapts `../00_iri/mycode/ML_sTEC/FNN_sTEC_test.ipynb` to the Topic1 IRI
simulation table. Supported feature sets are `ecef6`, `blh6`, and `bl4`.

## Reproducibility Rules

- Do not overwrite raw TEC, SP3, IRI, or model artifact files.
- Run small one-day tests before any larger multi-day regional processing.
- Record every experiment in `EXPERIMENT_LOG.md`.
- Update `TODO.md` when task status changes.
- Distinguish observation, inference, and speculation in scientific notes.
