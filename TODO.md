# TODO

## Current Milestone

Turn the one-day ray-wise STEC dataset builder into a publishable baseline
system with a traditional single-layer physical baseline and strict
generalization tests.

## Task List

- [x] Create `Topic1` reproducible project structure.
- [x] Configure dedicated Python environment in `.venv`.
- [x] Build one-day 2024-05-05 ray-wise STEC dataset.
- [x] Add Cartopy-based IPP/STEC diagnostic figure notebook.
- [x] Add agent-facing project brain files.
- [x] Implement single-layer mapping baseline entry point producing
  `stec_slm`.
- [x] Record the VTEC source, shell height, and mapping formula used by the SLM
  baseline.
- [x] Add direct SLM evaluation metrics against `stec_obs`.
- [x] Implement Ridge and RandomForest direct-prediction entry points.
- [x] Implement Ridge and RandomForest residual-learning entry points:
  `stec_obs - stec_slm`.
- [x] Implement RayMLP residual-learning entry point after SLM and tabular
  baselines are stable.
- [ ] Implement random, time, station, and PRN split experiments.
- [ ] Add elevation-bin, station-bin, PRN-bin, and time-bin metrics.
- [ ] Add paper-quality result figures and tables.
- [ ] Draft methods section after baseline results are stable.
- [ ] After baseline evaluation is stable, choose ordinary additional days for
  multi-day regional generalization tests.

## Highest Priority Next

Provide an independent VTEC source for the existing one-day dataset, then run
`src.slm_baseline` to create the first science-useful `stec_slm` product.
Next, replace diagnostic constant-VTEC smoke data with an independent VTEC
source, then run SLM, direct ML, and residual ML baselines across strict splits.

## Blocked / Needs User Decision

- Need decision on whether to keep `rec_h = 0.0` for all stations or parse
  receiver height from another source.
