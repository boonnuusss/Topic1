# Data Dictionary

## Processed Dataset

Current ray-wise STEC file:

```text
datasets/processed/raystec_china_20240505.parquet
```

Each row is one receiver-satellite ray at one epoch.

Current SLM VTEC conversion target file:

```text
datasets/processed/rayvtec_china_20240505_slm.parquet
```

This file is produced by `python -m src.slm_vtec` and adds the traditional
single-layer STEC-to-VTEC conversion baseline.

## Provenance

- TEC observations: `E:/Scientific_Research/data/gnssTECmatFiles/2024/126/*.mat`
- Orbit data: `../00_iri/mycode/ML_sTEC/Mydata/WUM2024(126)0505.nc`
- Builder: `python -m src.dataset_builder --config configs/quick_test.yaml`
- SLM VTEC conversion: `python -m src.slm_vtec`
- IRI simulation builder: `python -m src.iri_simulation_builder`

## Core Columns

| Column | Meaning | Unit / Type |
| --- | --- | --- |
| `year` | Calendar year | integer |
| `doy` | Day of year | integer |
| `timestamp` | UTC epoch assigned by builder | datetime |
| `utc_hour` | Decimal UTC hour | hour |
| `station` | Receiver station code | string |
| `prn` | Mapped satellite PRN | integer |
| `constellation` | GNSS constellation inferred from mapped PRN | string |
| `stec_obs` | Observed slant TEC input | TECU |
| `stec_iri` | Legacy/reserved IRI-derived STEC column | TECU / currently empty |
| `valid_flag` | Passed quality control | boolean |

## Receiver Position

| Column | Meaning | Unit |
| --- | --- | --- |
| `rec_lon` | Receiver longitude | degree |
| `rec_lat` | Receiver latitude | degree |
| `rec_h` | Receiver height, currently set from source/default | meter |
| `rec_x`, `rec_y`, `rec_z` | Receiver ECEF position | meter |

## Satellite Position

| Column | Meaning | Unit |
| --- | --- | --- |
| `sat_lon` | Satellite longitude | degree |
| `sat_lat` | Satellite latitude | degree |
| `sat_h` | Satellite ellipsoidal height | km |
| `sat_x`, `sat_y`, `sat_z` | Satellite ECEF position | meter |
| `satellite_timestamp` | Nearest orbit epoch matched to observation epoch | datetime |
| `sat_position_matched` | Whether SP3/NetCDF position was matched | boolean |

## IPP And Geometry

| Column | Meaning | Unit |
| --- | --- | --- |
| `ipp_lon` | Ionospheric pierce point longitude | degree |
| `ipp_lat` | Ionospheric pierce point latitude | degree |
| `elevation` | Receiver-satellite elevation angle | degree |
| `azimuth` | Receiver-satellite azimuth angle | degree |
| `slant_factor` | Thin-shell mapping factor from the dataset builder | unitless |
| `ray_length` | Receiver-satellite straight-line distance | meter |
| `los_x`, `los_y`, `los_z` | Unit line-of-sight vector | unitless |

## Encoded Features

| Column | Meaning |
| --- | --- |
| `month` | Month of timestamp |
| `local_time_ipp` | Approximate IPP local time |
| `hour_sin`, `hour_cos` | UTC hour cyclical encoding |
| `doy_sin`, `doy_cos` | DOY cyclical encoding |
| `local_time_ipp_sin`, `local_time_ipp_cos` | IPP local-time cyclical encoding |
| `azimuth_sin`, `azimuth_cos` | Azimuth cyclical encoding |
| `elevation_sin`, `elevation_cos` | Elevation trigonometric encoding |
| `constellation_*` | One-hot constellation indicators |

## Mainline SLM VTEC Conversion Columns

These columns are produced by `python -m src.slm_vtec`.

| Column | Meaning | Unit / Type |
| --- | --- | --- |
| `shell_height_km` | Thin-shell height used by the mapping function | km |
| `slm_mapping_factor` | SLM mapping factor computed for `shell_height_km` | unitless |
| `vtec_slm` | Traditional SLM conversion result, `stec_obs / slm_mapping_factor` | TECU |
| `vtec_slm_source` | Provenance label for `vtec_slm` | string |

The current mainline formula is:

```text
VTEC_slm = STEC_obs / M(elevation)
```

## IRI Simulation Columns

The detailed design is in `IRI_SIMULATION_SCHEMA.md`. The intended processed
simulation file is:

```text
datasets/processed/iri_simulation_rayvtec_YYYYMMDD.parquet
```

Each row remains one receiver-satellite ray at one UTC epoch. The IRI-backed
simulation adds controlled reference targets and SLM conversion diagnostics.

| Column | Meaning | Unit / Type |
| --- | --- | --- |
| `source_row_id` | Original row number in the input ray-wise parquet | integer |
| `simulation_id` | Stable label for the IRI simulation run | string |
| `stec_iri_ray` | Full receiver-satellite ray integral through the IRI electron-density field | TECU |
| `vtec_iri_ref` | Reference VTEC at IPP lon/lat from IRI vertical integration | TECU |
| `vtec_iri_ref_method` | Method label, initially `vertical_ipp_trapezoid` | string |
| `stec_iri_method` | Method label, initially `ecef_ray_integral` | string |
| `slm_shell_height_km` | Thin-shell height used for SLM conversion | km |
| `slm_mapping_factor_iri` | Mapping factor computed from `elevation` and `slm_shell_height_km` | unitless |
| `vtec_slm_from_iri_stec` | SLM conversion of simulated observable STEC, `stec_iri_observable / slm_mapping_factor_iri` | TECU |
| `vtec_slm_error_iri` | SLM VTEC error, `vtec_slm_from_iri_stec - vtec_iri_ref` | TECU |
| `vtec_slm_abs_error_iri` | Absolute SLM VTEC error | TECU |
| `ray_integral_n_samples` | Number of samples used along the receiver-satellite ray | integer |
| `ray_integral_min_alt_km` | Minimum sampled ray altitude | km |
| `ray_integral_max_alt_km` | Maximum sampled ray altitude | km |
| `ray_integral_path_length_km` | Full sampled ray path length | km |
| `ray_integral_outside_grid_fraction` | Fraction of ray samples outside the IRI grid | unitless |
| `bias_mode` | Bias/noise mode, initially `none` | string |
| `receiver_bias_tecu` | Simulated receiver bias, initially `0` | TECU |
| `satellite_bias_tecu` | Simulated satellite bias, initially `0` | TECU |
| `noise_tecu` | Simulated random noise, initially `0` | TECU |
| `stec_iri_observable` | Simulated observable STEC after optional bias/noise | TECU |

For the first no-bias simulation:

```text
stec_iri_observable = stec_iri_ray
vtec_slm_from_iri_stec = stec_iri_observable / slm_mapping_factor_iri
```

## Legacy Off-mainline Columns

These columns are produced by `python -m src.slm_baseline` and were introduced
for an earlier STEC-prediction route. They are not the current mainline target.

| Column | Meaning | Unit / Type |
| --- | --- | --- |
| `vtec_baseline` | External VTEC value used to predict STEC | TECU |
| `vtec_source` | External VTEC source label, for example `IRI` or `GIM` | string |
| `stec_slm` | Legacy STEC prediction, `M(elevation) * vtec_baseline` | TECU |
| `stec_residual_slm` | Legacy residual target, `stec_obs - stec_slm` | TECU |

## VTEC-Space Metric Tables

These tables are produced by `python -m src.evaluate` and compare SLM or
ML-assisted VTEC estimates against a reference VTEC column, usually
`vtec_iri_ref`.

Default outputs:

```text
results/tables/<run_name>_vtec_metrics.csv
results/tables/<run_name>_vtec_overall_metrics.csv
results/tables/<run_name>_vtec_elevation_bin_metrics.csv
results/tables/<run_name>_vtec_station_metrics.csv
results/tables/<run_name>_vtec_utc_hour_metrics.csv
results/tables/<run_name>_vtec_metrics.metadata.json
```

Metric convention:

```text
error = prediction - reference
```

Core columns:

| Column | Meaning | Unit / Type |
| --- | --- | --- |
| `scope` | Metric scope: `overall`, `elevation_bin`, `station`, or `utc_hour` | string |
| `model` | Model label, for example `slm` or `ml_assisted_from_stec_ray_mlp` | string |
| `prediction_column` | VTEC column evaluated after any STEC-to-VTEC conversion | string |
| `prediction_source` | `vtec_prediction` or `stec_prediction` | string |
| `source_column` | Original STEC column if converted to VTEC | string |
| `target_column` | Reference VTEC column, usually `vtec_iri_ref` | string |
| `elevation_bin_deg` | Elevation bin label when `scope=elevation_bin` | degree bin |
| `station` | Station label when `scope=station` | string |
| `utc_hour` | UTC-hour label when `scope=utc_hour` | string |
| `n_samples` | Number of finite prediction/reference pairs | integer |
| `n_missing` | Number of non-finite pairs excluded from metrics | integer |
| `bias_tecu` | Mean VTEC error | TECU |
| `median_error_tecu` | Median VTEC error | TECU |
| `std_error_tecu` | Standard deviation of VTEC error | TECU |
| `mae_tecu` | Mean absolute VTEC error | TECU |
| `rmse_tecu` | Root-mean-square VTEC error | TECU |
| `p05_error_tecu`, `p95_error_tecu` | 5th and 95th percentile VTEC error | TECU |
| `max_abs_error_tecu` | Maximum absolute VTEC error | TECU |
| `r2` | R2 in VTEC space | unitless |
| `delta_rmse_vs_slm_tecu` | RMSE difference relative to the SLM baseline | TECU |
| `delta_mae_vs_slm_tecu` | MAE difference relative to the SLM baseline | TECU |
| `rmse_reduction_vs_slm_pct` | RMSE reduction relative to SLM; positive means improvement | percent |
| `mae_reduction_vs_slm_pct` | MAE reduction relative to SLM; positive means improvement | percent |

## RayMLP Simulation Prediction Columns

These columns are produced by `python -m src.train_ray_mlp_simulation`. The
model is trained as a STEC ray model, not as a direct VTEC residual model.
`vtec_iri_ref` is not used during training; it is used only for post-training
evaluation.

Model input concept:

```text
source slant STEC + source ray geometry + target ray geometry -> target STEC
```

During training, the target ray is the original slant ray and the label is
`stec_iri_ray`. For STEC-to-VTEC conversion, the trained model is queried twice:
once with the original slant target ray and once with a vertical target ray at
the same IPP/time. The final conversion uses the model-implied ratio between the
vertical and slant target responses.

```text
ray_mlp_conversion_ratio = vtec_ray_mlp_direct / stec_ray_mlp
vtec_ray_mlp = stec_iri_observable * ray_mlp_conversion_ratio
```

| Column | Meaning | Unit / Type |
| --- | --- | --- |
| `stec_ray_mlp` | RayMLP prediction when the target ray is the original slant ray | TECU |
| `vtec_ray_mlp_direct` | Raw RayMLP prediction for the vertical target ray before ratio conversion | TECU |
| `ray_mlp_conversion_ratio` | Model-implied vertical/slant response ratio, `vtec_ray_mlp_direct / stec_ray_mlp` | unitless |
| `vtec_ray_mlp` | Final RayMLP-assisted converted VTEC estimate | TECU |
| `vtec_ray_mlp_error_iri` | `vtec_ray_mlp - vtec_iri_ref` | TECU |
| `vtec_ray_mlp_abs_error_iri` | Absolute RayMLP VTEC error | TECU |
| `vtec_ray_mlp_direct_error_iri` | Direct vertical-query diagnostic error, `vtec_ray_mlp_direct - vtec_iri_ref` | TECU |
| `vtec_ray_mlp_direct_abs_error_iri` | Absolute direct vertical-query diagnostic error | TECU |

## FNN Simulation Prediction Columns

These columns are produced by `python -m src.train_fnn_simulation`. The FNN is
adapted from `../00_iri/mycode/ML_sTEC/FNN_sTEC_test.ipynb` and maps ray
geometry directly to STEC.

| Column | Meaning | Unit / Type |
| --- | --- | --- |
| `stec_fnn` | FNN prediction for the original slant ray | TECU |
| `vtec_fnn_direct` | Raw FNN prediction for the vertical target ray before ratio conversion | TECU |
| `fnn_conversion_ratio` | Model-implied vertical/slant response ratio, `vtec_fnn_direct / stec_fnn` | unitless |
| `vtec_fnn` | Final FNN-assisted converted VTEC estimate | TECU |
| `vtec_fnn_error_iri` | `vtec_fnn - vtec_iri_ref` | TECU |
| `vtec_fnn_abs_error_iri` | Absolute FNN VTEC error | TECU |
| `vtec_fnn_direct_error_iri` | Direct vertical-query diagnostic error, `vtec_fnn_direct - vtec_iri_ref` | TECU |
| `vtec_fnn_direct_abs_error_iri` | Absolute direct vertical-query diagnostic error | TECU |

## GNN Simulation Prediction Columns

These columns are produced by `python -m src.train_gnn_simulation`. The GNN is
adapted from `../00_iri/mycode/ML_sTEC/GNN_sTEC_improve.ipynb` as a
receiver/satellite graph edge regressor. Nodes are receiver and satellite ECEF
positions; ray edges carry elevation and azimuth features. Receiver-receiver
KNN edges provide spatial message-passing context but do not have STEC labels.

Training uses only simulated slant-ray `stec_iri_ray` labels. `vtec_iri_ref`
is used only for post-training VTEC-space evaluation.

| Column | Meaning | Unit / Type |
| --- | --- | --- |
| `stec_gnn` | GNN prediction for the original slant receiver-satellite edge | TECU |
| `vtec_gnn_direct` | Raw GNN prediction for the vertical special edge before ratio conversion | TECU |
| `gnn_conversion_ratio` | Model-implied vertical/slant response ratio, `vtec_gnn_direct / stec_gnn` | unitless |
| `vtec_gnn` | Final GNN-assisted converted VTEC estimate | TECU |
| `vtec_gnn_error_iri` | `vtec_gnn - vtec_iri_ref` | TECU |
| `vtec_gnn_abs_error_iri` | Absolute GNN VTEC error | TECU |
| `vtec_gnn_direct_error_iri` | Direct vertical-query diagnostic error, `vtec_gnn_direct - vtec_iri_ref` | TECU |
| `vtec_gnn_direct_abs_error_iri` | Absolute direct vertical-query diagnostic error | TECU |

## Current Quality Control

- `stec_obs` is not NaN.
- `elevation >= 10 deg`.
- `ipp_lon` and `ipp_lat` lie within `80E-140E`, `0N-60N`.
- `0 <= stec_obs <= 600 TECU`.
- Satellite position successfully matched.
