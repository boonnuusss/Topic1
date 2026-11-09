# TODO

## Current Milestone

Turn the one-day ray-wise STEC dataset into a STEC-to-VTEC conversion benchmark:
first with the traditional single-layer mapping baseline, then with an
IRI-backed simulation experiment for ML-assisted conversion.

## Task List

- [x] Create `Topic1` reproducible project structure.
- [x] Configure dedicated Python environment in `.venv`.
- [x] Build one-day 2024-05-05 ray-wise STEC dataset.
- [x] Add Cartopy-based IPP/STEC diagnostic figure notebook.
- [x] Add agent-facing project brain files.
- [x] Reframe the project from standalone STEC prediction to STEC modeling for
  STEC-to-VTEC conversion.
- [x] Implement mainline SLM STEC-to-VTEC conversion entry point:
  `vtec_slm = stec_obs / slm_mapping_factor`.
- [x] Keep previously implemented `stec_slm`, direct STEC prediction, and
  residual-learning code as legacy/off-mainline code.
- [x] Run `src.slm_vtec` on the full one-day parquet and inspect the resulting
  `vtec_slm` distribution.
- [x] Add VTEC diagnostic plots: map of `vtec_slm`, elevation-bin distribution,
  station-bin distribution, and time evolution.
- [x] Design the IRI-backed simulation schema: reference VTEC, simulated STEC,
  shell height, integration assumptions, and metadata.
- [x] Implement the IRI simulation dataset builder.
- [x] Define VTEC-space metrics for SLM versus ML-assisted conversion.
- [x] Adapt RayMLP training to the simulation setting.
- [x] Adapt FNN and GNN notebook-style models to the UT05:00 IRI simulation
  setting for slant STEC training and vertical-query VTEC conversion tests.
- [x] Implement and test all-to-all RayMLP pair training.
- [x] Implement and test IPP-nearby RayMLP pair training.
- [x] Implement and test RayMLP source-ratio training target for STEC-to-VTEC
  conversion.
- [x] Implement and test low-elevation ML / high-elevation SLM hybrid VTEC
  conversion.
- [x] Implement and test Set-to-Query training with nearby observation sets.
- [x] Fix GNN azimuth feature convention to match the legacy notebook.
- [x] Validate IPP-nearby RayMLP source-ratio plus low-elevation hybrid under
  random, time, station, and PRN splits.
- [x] Fix `--nearby-no-identity` semantics and run UT05:00 random self-pair A/B
  for source-ratio RayMLP.
- [x] Record the limitation that real observation data lack VTEC truth, so
  hybrid-gate exploration is diagnostic rather than a mainline objective.
- [x] Implement and test a source-ratio set/context model with a source-anchor
  observation flag.
- [x] Implement train/validation/locked-test key protocol for the leading
  set-ratio route.
- [x] Save and restore the minimum-validation-RMSE checkpoint before final
  predictions.
- [x] Reclassify the repeatedly reused UT05 key holdout as validation.
- [x] Install PyPI JRC NeQuick-G in an isolated NumPy 1.x environment.
- [x] Build the matching full-day NeQuick-G dataset from actual 2024-05-05
  Galileo broadcast coefficients and repeat all physical mapping baselines.
- [x] Add extended physical mapping benchmarks: SLM shell-height sweep, CODE
  MSLM, fixed-profile multi-layer MF, training-partition IRI-calibrated APMF,
  and the IRI ideal mapping-ratio oracle.
- [x] Add a high-low CPP APMF whose pairing, fitting, and regularization
  selection use STEC and geometry only; compare full-day and UT04-08 matched
  calibration with the NeQuick-G Set-ratio model.
- [ ] Validate the leading set-ratio anchor-only model under station, time, PRN,
  and independent-day splits.
- [ ] Validate core RayMLP / Set-to-Query STEC-to-VTEC conversion on independent
  days without relying on hybrid selection tuned by `vtec_iri_ref`.
- [ ] Validate leading Set-to-Query configurations under time, station, and PRN
  holdout splits.
- [ ] Add paper-quality result figures and tables.
- [ ] Draft methods section after the SLM and IRI simulation results are stable.
- [ ] After the one-day benchmark is stable, choose ordinary additional days for
  multi-day regional generalization tests.

## Legacy Completed Work

These items were completed during the earlier STEC-prediction route. They are
preserved for reference but are not the current mainline:

- [x] Implement `src.slm_baseline` producing `stec_slm`.
- [x] Record VTEC source, shell height, and mapping formula for legacy
  `stec_slm`.
- [x] Add direct SLM evaluation metrics against `stec_obs`.
- [x] Implement Ridge and RandomForest direct-prediction entry points.
- [x] Implement Ridge and RandomForest residual-learning entry points:
  `stec_obs - stec_slm`.
- [x] Implement RayMLP residual-learning entry point on top of `stec_slm`.

## Highest Priority Next

Create an independent-day locked-test key manifest and evaluate the frozen
set-ratio paper protocol exactly once after validation checkpoint selection.
The current UT05 key set is validation because it has already guided extensive
model development. Then validate station, time, PRN, and constellation
generalization with multiple seeds. Hybrid gates remain simulation diagnostics;
they should not drive the main research direction because real observations do
not provide VTEC truth for gate selection.

Freeze the physical baseline configuration at the same time. Do not select a
shell height or refit APMF using locked-test truth.

## Blocked / Needs User Decision

- Need decision on whether to keep `rec_h = 0.0` for all stations or parse
  receiver height from another source.
- Need decision on whether to keep UT05:00 as the primary single-epoch benchmark
  or move the leading routes to the UT04:00-06:00 multi-epoch simulation next.
