# Research Plan

## Central Scientific Question

Can observation-level, ray-wise modeling improve regional GNSS STEC prediction
beyond a traditional single-layer ionospheric projection baseline by using
receiver-satellite geometry, IPP location, time features, and physical model
priors directly at the ray level?

## Working Hypotheses

- H1: Receiver and satellite geometry contains strong information for STEC
  prediction.
- H2: Adding IPP position, elevation/azimuth, and local-time/DOY encoding should
  improve generalization over geometry-only features.
- H3: A traditional single-layer mapping baseline should provide the first
  physical reference for STEC skill.
- H4: Machine-learning models may be most scientifically useful when they learn
  the residual left by the single-layer baseline, rather than replacing it from
  the start.
- H5: Random sample splits can overestimate performance; time, station, and PRN
  holdout splits are required to test generalization.

## Required Datasets

- GNSS TEC `.mat` files with `slantTEC`, PRN, station position, IPP position,
  azimuth, elevation, and UTh.
- SP3 precise orbit files or previously converted NetCDF orbit files.
- Optional future data: independent VTEC source for the single-layer baseline,
  such as IRI/PyIRI-derived VTEC or GIM-interpolated VTEC, plus historical STEC
  lag features.

## Key Observables

- Target: `stec_obs` in TECU.
- Main geometry: receiver ECEF/BLH, satellite ECEF/BLH, IPP lon/lat,
  elevation, azimuth, ray length, LOS vector, slant factor.
- Time: UTC hour, DOY, month, IPP local time, cyclical encodings.
- Grouping variables: station, PRN, constellation, timestamp.

## Minimum Publishable Result

1. Multi-epoch ray-wise STEC dataset with explicit quality control.
2. Physical baseline: single-layer model
   `STEC_slm = M(elevation) * VTEC(IPP, time)`, with the VTEC source recorded.
3. Direct ML baselines: Ridge and RandomForest trained on ray-wise features.
4. Residual ML baselines: Ridge/RandomForest/RayMLP trained on
   `stec_obs - stec_slm`.
5. Strict evaluation: random split, time split, station holdout, PRN holdout,
   elevation-bin metrics.
6. Diagnostic figures: IPP coverage, STEC spatial map, predicted vs observed,
   residual map, elevation-bin RMSE.

## Current Scope

The current implemented quick test builds one day:

```text
Date: 2024-05-05
Region: 80E-140E, 0N-60N
Resolution: 15 min
Output: datasets/processed/raystec_china_20240505.parquet
```

## Review Risks

- Data leakage from random sample split across same stations/times/PRNs.
- Confusing geographic coordinate systems.
- Confusing UTC and local time.
- Hidden hard-coded paths.
- Baseline VTEC products that are not independent of evaluation data.
- Hidden assumptions in the single-layer shell height or mapping function.
- Overstating causality from regression performance.
