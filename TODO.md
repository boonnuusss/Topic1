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

Validate the core STEC-to-VTEC modeling route on independent days. Hybrid gates
remain useful for IRI simulation diagnostics, but they should not drive the main
research direction because real observation data do not provide VTEC truth for
choosing the best gate. Prioritize truth-free model behavior, physically
pre-specified conversion rules, and larger multi-epoch validation.

## Blocked / Needs User Decision

- Need decision on whether to keep `rec_h = 0.0` for all stations or parse
  receiver height from another source.
- Need decision on whether to keep UT05:00 as the primary single-epoch benchmark
  or move the leading routes to the UT04:00-06:00 multi-epoch simulation next.
