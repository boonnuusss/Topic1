# Research Plan

## Central Scientific Question

Can ray-wise GNSS STEC modeling improve STEC-to-VTEC conversion relative to the
traditional single-layer mapping model, when the comparison is evaluated in a
controlled IRI-backed simulation experiment?

The project should not be framed as standalone STEC prediction. STEC modeling is
the means; improved VTEC conversion is the scientific endpoint.

## Working Hypotheses

- H1: The traditional single-layer mapping model provides the necessary
  physical baseline for converting observed STEC to VTEC:
  `VTEC_slm = STEC_obs / M(elevation)`.
- H2: Ray-wise geometry, IPP location, time encoding, satellite identity, and
  receiver context contain information that can help ML models correct
  conversion errors left by the single-layer assumption.
- H3: IRI-generated background ionosphere fields can provide controlled
  reference VTEC/STEC relationships for simulation experiments before using
  noisier external products.
- H4: ML methods such as RayMLP, GNN, and set-to-query models should be
  evaluated by their VTEC conversion error, not only by their STEC prediction
  error.
- H5: Random sample splits can overestimate performance; time, station, and PRN
  holdout splits are required to test generalization.
- H6: Local IPP context matters. IPP-nearby ray pairs and set-to-query models
  should be preferred over identity-only slant training when the goal is
  STEC-to-VTEC conversion.
- H7: ML gains must be compared with a stronger physical benchmark family,
  including shell-height sensitivity, CODE MSLM, and a multi-layer profile.
  IRI-calibrated APMF and the exact IRI ratio are simulation references, not
  truth-free operational baselines.

## Required Datasets

- GNSS TEC `.mat` files with `slantTEC`, PRN, station position, IPP position,
  azimuth, elevation, and UTh.
- SP3 precise orbit files or previously converted NetCDF orbit files.
- IRI/PyIRI background ionosphere outputs used to generate controlled VTEC and
  ray-wise STEC simulation labels.
- Optional later data: GIM or another independent VTEC product for external
  validation after the IRI simulation route is stable.
- JRC NeQuick-G simulation driven by Galileo broadcast `ai0/ai1/ai2`, used as
  an independent empirical background-model sensitivity experiment.

## Key Observables

- Observed ray-wise input: `stec_obs` in TECU.
- Traditional conversion baseline: `vtec_slm` in TECU.
- Future simulation reference: `vtec_ref_iri` or equivalent, in TECU.
- Future simulated ray target: `stec_sim_iri` or equivalent, in TECU.
- Main geometry: receiver ECEF/BLH, satellite ECEF/BLH, IPP lon/lat,
  elevation, azimuth, ray length, LOS vector, slant factor.
- Time: UTC hour, DOY, month, IPP local time, cyclical encodings.
- Grouping variables: station, PRN, constellation, timestamp.

## Minimum Publishable Result

1. Multi-epoch ray-wise STEC dataset with explicit quality control.
2. Physical conversion baseline: single-layer mapping
   `VTEC_slm = STEC_obs / M(elevation)`, with shell height and mapping formula
   recorded.
   The benchmark family must also report the shell-height sweep, CODE MSLM,
   and a fixed-profile multi-layer MF.
3. IRI-backed simulation dataset with documented reference VTEC and simulated
   ray-wise STEC labels.
4. Ray-wise ML STEC model trained in the simulation setting, with IPP-nearby
   and set-to-query routes prioritized.
5. Main comparison in VTEC space: SLM conversion error versus ML-assisted
   STEC-to-VTEC conversion error.
6. Strict evaluation: random split, time split, station holdout, PRN holdout,
   elevation-bin metrics.
7. Diagnostic figures: IPP coverage, VTEC spatial map, SLM conversion error
   map, ML conversion error map, predicted/reference scatter, elevation-bin
   RMSE.

## Publication Evaluation Protocol

The main set-ratio route now uses explicit train, validation, and locked-test
roles. Validation VTEC truth may select the epoch in controlled simulation, but
locked-test VTEC truth must be used exactly once after all model and
hyperparameter decisions are frozen.

The former fixed UT05 key holdout has been reused across Experiments 042-057 for
time-window, pair-budget, target-transform, shrinkage, and epoch choices. It is
therefore formally classified as validation. Its best observed RMSE values are
model-development evidence, not unbiased test performance.

Required publication claims must use:

1. A validation set for checkpoint and hyperparameter selection.
2. A locked independent-day test set excluded from training, scaler fitting,
   pair construction, and epoch selection.
3. Additional station, PRN, constellation, and time holdouts.
4. Multiple random seeds and uncertainty intervals.

`src.train_set_ratio_mlp` saves the minimum-validation-RMSE checkpoint and
evaluates a configured locked test only after restoring that checkpoint.
Runs using the historical two-way protocol remain diagnostic and are not
sufficient for final paper claims.

The same partition rule applies to empirical physical functions. APMF
coefficients are fitted only on the training partition; validation/test VTEC is
used only for scoring. The exact IRI STEC/VTEC ratio is labeled an oracle and
must never be presented as an available real-observation method.

For a truth-free empirical comparison, the NeQuick-G benchmark additionally
fits APMF from same-epoch high-low coinciding-pierce-point pairs. The fit uses
only STEC, elevation, azimuth, IPP coordinates, and local time. A high-elevation
CODE-MSLM conversion supplies the vertical proxy, and regularization is chosen
only from held-out CPP proxy ratios. Both full-day and UT04:00-08:00 matched-
window calibration must be shown; the latter is the fair comparison with the
current Set-ratio run.

The physical benchmark is now repeated with NeQuick-G on the same ray keys and
90-2000 km integration bounds. This tests whether shell-height and APMF
conclusions are specific to IRI. NeQuick-G results remain validation evidence
until an independent locked day is evaluated.

## Current Scope

The implemented quick test builds one day:

```text
Date: 2024-05-05
Region: 80E-140E, 0N-60N
Resolution: 15 min
Output: datasets/processed/raystec_china_20240505.parquet
```

The SLM baseline output is:

```text
datasets/processed/rayvtec_china_20240505_slm.parquet
```

generated by `python -m src.slm_vtec`.

The IRI-backed simulation schema is now defined in:

```text
IRI_SIMULATION_SCHEMA.md
```

It specifies `stec_iri_ray` as the full receiver-satellite ray integral,
`vtec_iri_ref` as the IPP vertical reference, and
`vtec_slm_from_iri_stec` as the classical SLM conversion of simulated STEC.

Current single-epoch UT05:00 modeling status:

| Route | Status | Main conclusion |
| --- | --- | --- |
| Identity-only RayMLP | Diagnostic only | Fits slant STEC but fails VTEC conversion because the vertical query is out of distribution. |
| All-to-all RayMLP pairs | Tested | Better than identity-only but worse than local filtering. |
| IPP-nearby RayMLP pairs | Active baseline | 200 km filtering reaches near-SLM VTEC accuracy with far fewer pairs. |
| Set-ratio source-anchor model | Current leading UT05 random pure-ML route | Source-ratio set architecture with only the source anchor reaches 1.51 TECU RMSE without VTEC truth in training. |
| RayMLP source-ratio pairs with self-pairs | Strong pair-ML baseline | Nearby200, 100ep trains target/source STEC ratio and reduces UT05:00 random-split VTEC RMSE from 2.44 to 2.13 TECU. |
| RayMLP source-ratio no-self pairs | Diagnostic | Excluding `i -> i` worsens pure VTEC conversion to 3.25 TECU on the same random split. |
| Low-elevation gated hybrid20 | Simulation diagnostic / sanity check | Uses a truth-free rule below 20 deg, but hybrid selection is not a main operational target because real observations lack VTEC truth. |
| Low-elevation gated hybrid20 no-self | Random-split diagnostic only | Shows gate sensitivity in IRI simulation; not worth optimizing further without a truth-free physical rationale. |
| Low-elevation hybrid30 | Random-split diagnostic | Best earlier UT05:00 random-split RMSE was 1.99 TECU, but the 20-30 deg bin can degrade stricter splits. |
| Set-to-query absolute model | Secondary/diagnostic route | Absolute-STEC set-to-query slightly surpasses SLM, but source-ratio formulation is now stronger. |
| Neighbor context in set-ratio model | Diagnostic | Small context (`100 km`, `5` obs) is competitive, but larger context degraded VTEC; context aggregation needs more design. |
| Hard identity set-ratio model | Diagnostic / not mainline | Enforces same-ray ratio exactly 1, but worsens UT05:00 random VTEC RMSE to 3.22 TECU; keep as a failed hard-constraint test. |
| All-ray set-ratio fit/evaluation | Diagnostic / full-data conversion check | Trains on all UT05:00 slant rays and evaluates all vertical queries without using VTEC truth for training; 200 epochs improve all-ray SLM RMSE from 2.50 to 1.78 TECU, but this is not a holdout generalization metric. |
| High-elevation SLM-correction with shrinkage | Leading high-elevation route | Random split recommends start=30/end=90/power=4/min=0.1 (RMSE 0.81 vs SLM 0.93); all-ray diagnostics prefer min=0 for strict no-bin-worse behavior (RMSE 0.78 vs SLM 0.86). |
| FNN/GNN notebook ports | Diagnostics | Useful implementation checks, but not current leading routes. |

Current strict split summary for RayMLP source-ratio plus gated hybrid20 uses the with-self nearby default. These hybrid numbers are retained mainly to diagnose low-elevation behavior in IRI simulation. Because real observations do not include VTEC truth, selecting among hybrid gates by `vtec_iri_ref` performance is not a transferable operational strategy:

| Split | SLM RMSE | Pure RayMLP RMSE | Hybrid20 all | Hybrid20 gated | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Random UT05:00 key split | 2.44 | 2.13 | 2.10 | 2.10 | Improves SLM. |
| Station holdout UT05:00 | 2.60 | 2.97 | 1.95 | 1.89 | Improves SLM despite pure RayMLP degradation. |
| Time holdout UT05:00 smoke | 2.50 | 3.92 | 1.94 | 1.94 | Improves SLM despite pure RayMLP degradation. |
| PRN holdout UT05:00 seed 42 | 3.14 | 5.39 | 3.40 | 2.91 | Gating fixes the all-constellation failure. |
| PRN holdout UT05:00 seed 7 | 2.97 | 2.41 | 1.90 | 1.87 | Gating preserves the gain. |
| PRN holdout UT05:00 seed 123 | 2.15 | 2.90 | 2.19 | 1.77 | Gating beats a strong SLM split. |


## VTEC-Space Metrics

All mainline evaluation is in VTEC space with the convention:

```text
error = VTEC_prediction - VTEC_reference
```

For the IRI simulation, the default reference is `vtec_iri_ref` and the SLM
baseline prediction is `vtec_slm_from_iri_stec`. Primary metrics are RMSE, MAE,
bias, median error, 5th/95th percentile error, maximum absolute error, and R2.
Metrics are reported overall and by elevation bin, station, and UTC hour.

Hybrid metrics must be interpreted cautiously. They are meaningful for IRI
simulation diagnostics, where `vtec_iri_ref` exists, but real observation data
usually lack an independent VTEC truth. Therefore, the project should avoid
spending effort on empirical hybrid-gate searches unless the rule is defined in
advance from observable quantities and physical reasoning.

ML-assisted conversion can be evaluated in either of three table forms:

- a model-produced VTEC column, compared directly with `vtec_iri_ref`;
- a model-produced STEC column, converted to VTEC with
  `VTEC_ml = STEC_ml / slm_mapping_factor_iri`;
- a ray-wise STEC model queried with vertical special-ray geometry, then used to
  form a STEC-to-VTEC conversion ratio.

Improvements are reported as deltas and percentage reductions relative to the
SLM baseline; negative `delta_rmse_vs_slm_tecu` or positive
`rmse_reduction_vs_slm_pct` means improvement.

## ML Simulation Routes

The active ML training entry points are:

```text
python -m src.train_ray_mlp_simulation
python -m src.train_set_mlp
```

It does not learn a VTEC residual, and `vtec_iri_ref` must not be used as a
training label. Instead, it learns a STEC model from simulated slant rays:

```text
source slant STEC ray + source ray geometry + original slant target geometry
-> slant target STEC
```

For STEC-to-VTEC conversion, the trained STEC model is queried with both the
original slant target ray and a vertical target ray at the same IPP/time. The
final converted VTEC is:

```text
vtec_ray_mlp = source_STEC * vtec_ray_mlp_direct / stec_ray_mlp
```

`vtec_iri_ref` is used only after training to evaluate that converted
prediction.

The identity-only RayMLP setup is now treated as a failed diagnostic route. The
active routes are IPP-nearby pair learning and set-to-query learning, because
both expose the model to non-identity source-to-target ray relationships.

## Legacy Off-mainline Work

The previously implemented `stec_slm`, direct STEC prediction, and
SLM-residual learning entry points are kept as legacy/off-mainline code. They
may be useful later, but they do not define the current research question.

## Review Risks

- Confusing the SLM direction: current baseline is `STEC_obs -> VTEC_slm`, not
  `VTEC -> STEC_slm`.
- Evaluating ML in STEC space and then overstating VTEC conversion value.
- Data leakage from random sample split across same stations/times/PRNs.
- Confusing geographic coordinate systems.
- Confusing UTC and local time.
- Hidden hard-coded paths.
- IRI simulation assumptions that are not explicitly documented.
- Hidden assumptions in the single-layer shell height or mapping function.
- Feature convention mismatches, especially azimuth range differences between
  legacy notebooks and parquet columns.
- Overstating causality from regression performance.
