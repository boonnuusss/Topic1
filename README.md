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
4. Train ray-wise ML STEC models after the simulation labels and SLM baseline
   are stable.
5. Prioritize non-identity ML routes, especially IPP-nearby ray pairs and
   set-to-query attention models. Identity-only slant supervision is retained
   only as a diagnostic because it does not reliably learn STEC-to-VTEC
   conversion.
6. Compare SLM and ML-assisted STEC-to-VTEC conversion in VTEC space, using
   metrics such as VTEC RMSE, MAE, bias, elevation-bin error, station-bin error,
   PRN-bin error, and spatial residual maps.

This order keeps the comparison anchored to the classical thin-shell
ionospheric projection model while making the evaluation target clear:
conversion accuracy from STEC to VTEC.

## Paper Evaluation Protocol

Publication-oriented set-ratio experiments use three distinct roles:

1. `train`: fits pair examples, scalers, and model parameters.
2. `validation`: selects the epoch with minimum VTEC RMSE and saves that
   model as the main checkpoint.
3. `locked test`: is evaluated once after best-checkpoint selection; it must
   not influence epoch or hyperparameter choices.

The historical UT05 key set
`results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet` is now a
validation set. Results such as RMSE `0.5586 TECU` are validation results, not
final independent-test claims.

`src.train_set_ratio_mlp` implements the protocol through:

```text
--validation-keys-from <validation keys>
--test-keys-from <locked independent-test keys>
```

When validation is configured, epoch diagnostics run on validation only, the
minimum-validation-RMSE state is restored, and locked-test predictions are
created afterward. Historical commands without `--validation-keys-from`
remain executable but are labeled `legacy_two_way` in metadata.

The reproducible wrapper is
`scripts/train_set_ratio_paper_protocol.ps1`. The next scientific requirement
is an independent-day locked-test key manifest.

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
- `NEW_COMPUTER_HANDOFF.md`: clone, environment, ignored-asset restoration,
  current results, and exact continuation instructions for another computer.
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

NeQuick-G uses an isolated environment because `nequick==1.0.0` requires
NumPy 1.x while the main ML environment uses NumPy 2.x:

```powershell
.\scripts\install_nequick.ps1
.\activate_nequick.ps1
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

## Extended Physical Mapping Baselines

The IRI simulation benchmark now compares more than the default 450 km SLM:

```powershell
.\scripts\evaluate_physical_mapping_baselines.ps1
```

`src.physical_mapping_baselines` evaluates SLM shell heights 300-600 km, CODE
MSLM (`H=506.7 km`, `alpha=0.9782`), a fixed Chapman-plus-plasmasphere
multi-layer MF, an IRI-calibrated APMF, and the exact IRI mapping ratio. The
historical 419-ray UT05 key set is treated as validation and is excluded from
APMF coefficient fitting.

These methods have different scientific roles. SLM, CODE MSLM, and the fixed
multi-layer profile are truth-free geometric/physical baselines. APMF uses IRI
VTEC/STEC on the training partition to estimate latitude-banded azimuth/local-
time coefficients, so it is an empirical simulation reference rather than a
deployable truth-free baseline. `mapping_factor_iri_ideal_oracle =
stec_iri_observable / vtec_iri_ref` uses evaluation truth directly and is only
the simulation upper bound.

The same benchmark is implemented for the PyPI JRC NeQuick-G model. It uses the
2024-05-05 Galileo broadcast coefficients from the IGS/BKG RINEX navigation
file and integrates both slant and IPP-vertical paths between 90 and 2000 km:

```powershell
.\scripts\download_nequickg_navigation.ps1
.\scripts\build_nequickg_simulation.ps1
.\scripts\evaluate_nequickg_physical_baselines.ps1
```

The NeQuick-G benchmark also includes a STEC-only high-low CPP APMF adapted
from Huo et al. (2024). Same-epoch rays above 70 degrees and from 10-30 degrees
are paired with the published 0.4-degree CPP rule. The high ray is converted to
a VTEC proxy with CODE MSLM; no simulated VTEC column is available to pair
construction, coefficient fitting, or ridge selection. Five-fold epoch-grouped
CPP cross-validation selects regularization. A UT04:00-08:00 calibration run is
reported beside the full-day calibration so its data coverage matches the
leading Set-ratio experiment.

IRI and NeQuick-G use identical 597,668 ray keys, but different numerical
integration implementations: IRI uses the project's regular-grid trapezoidal
route, whereas NeQuick-G uses its built-in Gauss-Kronrod integration.

## IRI Simulation Target

The next mainline step is to build an IRI-backed simulation experiment:

1. Generate or sample IRI background electron density/VTEC for IPP/time
   conditions.
2. Produce ray-wise simulated STEC or reference VTEC labels with known
   provenance.
3. Apply the SLM conversion baseline to estimate VTEC from STEC.
4. Train ray-wise ML STEC models under the same simulation setup.
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
  model, including identity-only diagnostics, all-to-all ray pairs, and
  IPP-nearby ray pairs.
- `python -m src.train_set_mlp`: set-to-query STEC model that encodes nearby
  observed rays and predicts a query ray.
- `python -m src.train_set_ratio_mlp`: source-ratio set/context model for direct
  STEC-to-VTEC conversion via predicted target/source STEC ratios.
- `python -m src.train_fnn_simulation`: compact FNN adapted from the local
  `FNN_sTEC_test.ipynb`.
- `python -m src.train_gnn_simulation`: pure-PyTorch GraphSAGE-style edge
  regressor adapted from `GNN_sTEC_improve.ipynb`.

All current ML routes train on simulated slant-ray `stec_iri_ray` labels only.
They query a vertical special ray after training and are scored against
`vtec_iri_ref` only in the evaluation step.

Current UT05:00 random-split benchmark status:

| Route | VTEC RMSE | R2 | Notes |
| --- | ---: | ---: | --- |
| SLM baseline | 2.44 | 0.961 | Traditional reference |
| Set-ratio MLP, source anchor only, 100ep | 1.51 | 0.985 | Best current UT05 random pure model; source-ratio set architecture with one anchor observation |
| Set-ratio MLP, hard identity anchor1, 100ep | 3.22 | 0.932 | Diagnostic; enforces same-ray ratio exactly 1 but worsens vertical-query VTEC |
| Set-ratio MLP, ctx100 obs5, 100ep | 2.10 | 0.971 | Small context helps SLM but is worse than anchor-only |
| RayMLP source-ratio nearby200, 100ep, with self-pairs | 2.13 | 0.970 | Default nearby behavior; self-pair ratio is exactly 1 in this simulation |
| RayMLP source-ratio nearby200, 100ep, no self-pairs | 3.25 | 0.931 | Diagnostic; pure VTEC conversion worsens |
| Hybrid low-elevation ML, threshold 30 deg | 1.99 | 0.974 | Earlier random-split diagnostic, but less conservative under stricter splits |
| Hybrid gated20 with self-pairs | 2.10 | 0.971 | Use ML below 20 deg only when ML VTEC is lower than SLM |
| Hybrid gated20 no-self diagnostic | 1.97 | 0.975 | Best random-split diagnostic so far; strict splits still untested |
| Set-to-query nearby200, sigma100, 100ep | 2.40 | 0.962 | Slightly better than SLM in random split |
| RayMLP nearby-200 absolute STEC, 100ep | 2.55 | 0.958 | Near SLM with filtered ray pairs |
| RayMLP all-to-all, 20ep | 3.24 | 0.931 | Worse than nearby filtering |
| RayMLP identity-only, 100ep | 24.08 | -2.80 | Old direction, diagnostic only |

Strict holdout checks currently favor a conservative gated hybrid:
use RayMLP only when `elevation < 20 deg` and `vtec_ray_mlp < vtec_slm`.
This gate uses only model/SLM predictions, not `vtec_iri_ref`. However, hybrid
search should be treated as an IRI-simulation diagnostic, not as the main
operational target. In real GNSS observations there is usually no independent
VTEC truth available to decide which hybrid gate is actually best, so tuning
thresholds or gate logic against `vtec_iri_ref` can overfit the simulation
benchmark. Further hybrid work should use only pre-specified, physically
motivated, truth-free rules.

| Split | SLM RMSE | Pure RayMLP RMSE | Hybrid20 all | Hybrid20 gated | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Random UT05:00 key split | 2.44 | 2.13 | 2.10 | 2.10 | Improves SLM |
| Station holdout UT05:00 | 2.60 | 2.97 | 1.95 | 1.89 | Improves SLM |
| Time holdout UT05:00 from UT04:00-06:00 smoke | 2.50 | 3.92 | 1.94 | 1.94 | Improves SLM |
| PRN holdout UT05:00, seed 42 | 3.14 | 5.39 | 3.40 | 2.91 | Improves SLM |
| PRN holdout UT05:00, seed 7 | 2.97 | 2.41 | 1.90 | 1.87 | Improves SLM |
| PRN holdout UT05:00, seed 123 | 2.15 | 2.90 | 2.19 | 1.77 | Improves SLM |

The present best random-split evidence is that source-ratio STEC models can beat
SLM without using VTEC truth for training or model selection. The new
set-ratio anchor-only model is the strongest UT05:00 random result so far, but
it still needs strict station/time/PRN and independent-day validation. Hybrid
results remain useful for diagnosing where ML helps, but the main scientific
route should prioritize the STEC ray model and its truth-free STEC-to-VTEC
conversion behavior, because real observational data do not provide VTEC truth
for selecting a hybrid rule.

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

For the current pair-learning route, prefer IPP-nearby pairs over all-to-all
pairs. The best current pure RayMLP route trains a target/source STEC ratio and
keeps the default self-pairs, so each source ray also teaches the identity ratio
`i -> i = 1`:

```powershell
.\.venv\Scripts\python.exe -m src.train_ray_mlp_simulation `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name ray_mlp_sim_0500_nearby200_e100_source_ratio `
  --nearby-km 200 `
  --nearby-max-neighbors 30 `
  --target-mode source_ratio `
  --epochs 100
```

Use `--nearby-no-identity` only as a diagnostic switch. After the UT05:00 random
self-pair A/B test, excluding self-pairs worsened pure RayMLP VTEC RMSE
(`3.25` versus `2.13` TECU), although it produced a more conservative gated20
random-split diagnostic (`1.97` TECU) that still needs strict-split validation.

A physically constrained hybrid keeps the ML correction only at the lowest
elevation angles. This is retained as a simulation diagnostic and sanity check,
not as a rule to keep optimizing against IRI truth. The current truth-free gate
also requires the ML VTEC estimate to be lower than the SLM estimate:

```powershell
.\scripts\build_hybrid_vtec.ps1 `
  -InputPath results\tables\ray_mlp_sim_0500_nearby200_e100_source_ratio_predictions.parquet `
  -OutputPath results\tables\ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid20_ml_lt_slm.parquet `
  -RunName ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid20_ml_lt_slm `
  -ElevationThresholdDeg 20 `
  -MlMustLowerSlm
```

## Set-Ratio Context Training

The current best UT05:00 random pure-model result uses a source-ratio
set/context entry point:

```powershell
.\.venv\Scripts\python.exe -m src.train_set_ratio_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name set_ratio_mlp_0500_anchor1_e100 `
  --nearby-km 200 `
  --context-km 0 `
  --nearby-max-neighbors 30 `
  --max-obs 1 `
  --epochs 100 `
  --batch-size 1024 `
  --device cpu
```

This model trains on nearby slant-ray pairs with target
`target_STEC / source_STEC`. The observation set contains the source ray and an
`is_source_anchor` flag, so the model learns the same source-ratio task as
RayMLP while using the set-to-query architecture. VTEC inference is simply:

```text
vtec_set_ratio_mlp = source_STEC * predicted_vertical_ratio
```

The anchor-only version reached `1.5060` TECU RMSE on the UT05:00 random split,
versus `2.1300` for RayMLP source-ratio and `2.4400` for SLM. Adding broader
context was not automatically better: `context_km=100, max_obs=5` gave `2.0997`,
and `context_km=200, max_obs=40` gave `3.4199`. Treat neighbor context as a
research variable, not an assumed improvement.

For full-dataset conversion diagnostics, use `--split all`. This trains on all
slant rays and evaluates vertical-query VTEC on all rays, while still keeping
`vtec_iri_ref` out of training. On UT05:00, increasing the all-ray anchor-only
run from 100 to 200 epochs improved RMSE from `1.8884` to `1.7806` TECU on all
`2,041` rays, versus `2.4953` for SLM. Report this separately from holdout
metrics because training and evaluation rays overlap, and the improvement with
epochs is not monotonic.
For high-elevation operational relevance, the training entry point also supports
`--min-train-elevation-deg`, `--min-eval-elevation-deg`, and
`--target-mode slm_correction`. Add `--elevation-shrinkage` to force the learned
SLM correction toward 1 as source elevation increases:

```text
correction = 1 + strength(elevation) * (raw_correction - 1)
```

On the UT05:00 random split with `elevation >= 30`, a sweep over 801 shrinkage
settings found `start=30`, `end=90`, `power=4`, `min_strength=0.1` as the best
balanced RMSE setting: RMSE `0.8112` TECU and MAE `0.5865`, versus SLM RMSE
`0.9285` and MAE `0.6842`, with no high-elevation bin worse than SLM. The earlier
`power=2`, `min_strength=0` setting remains a MAE-oriented alternative (`0.5626`
TECU MAE, `0.8293` TECU RMSE).

For all-train/all-eval high-elevation diagnostics, `vtec_iri_ref` is still not
used for training, but the same source rays appear in slant-pair training and
VTEC conversion evaluation. Treat this as full-dataset conversion behavior, not a
strict unseen-ray generalization metric. On all `1,465` high-elevation rays, SLM
has RMSE `0.8611`; `power=4`, `min_strength=0.1` gives RMSE `0.7771`, while the
strict no-bin-worse all-ray setting is `power=4`, `min_strength=0` with RMSE
`0.7822`.

## Set-to-Query Simulation Training

The current best UT05:00 simulation route is:

```powershell
.\scripts\run_set_mlp.ps1 `
  -RunName set_mlp_0500_nearby200_s100_e100 `
  -NearbyKm 200 `
  -SpatialSigmaKm 100 `
  -Epochs 100
```

Equivalent Python entry point:

```powershell
.\.venv\Scripts\python.exe -m src.train_set_mlp `
  --input datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet `
  --run-name set_mlp_0500_nearby200_s100_e100 `
  --nearby-km 200 `
  --spatial-sigma-km 100 `
  --epochs 100
```

This route builds one query example per held-out ray, encodes nearby observable
slant rays, applies spatially biased cross-attention, and predicts the target
query STEC. The output table includes `stec_set_mlp`,
`vtec_set_mlp_direct`, `set_mlp_conversion_ratio`, and `vtec_set_mlp`.

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
