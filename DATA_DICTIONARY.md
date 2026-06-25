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

## Future IRI Simulation Columns

Names may be refined when the IRI simulation builder is implemented, but the
intended meanings are:

| Column | Meaning | Unit / Type |
| --- | --- | --- |
| `vtec_ref_iri` | Reference VTEC generated from IRI background ionosphere | TECU |
| `stec_sim_iri` | Simulated ray-wise STEC derived from the IRI background | TECU |
| `iri_model_version` | IRI/PyIRI version or configuration label | string |
| `iri_assumption_notes` | Compact label for simulation assumptions | string |

## Legacy Off-mainline Columns

These columns are produced by `python -m src.slm_baseline` and were introduced
for an earlier STEC-prediction route. They are not the current mainline target.

| Column | Meaning | Unit / Type |
| --- | --- | --- |
| `vtec_baseline` | External VTEC value used to predict STEC | TECU |
| `vtec_source` | External VTEC source label, for example `IRI` or `GIM` | string |
| `stec_slm` | Legacy STEC prediction, `M(elevation) * vtec_baseline` | TECU |
| `stec_residual_slm` | Legacy residual target, `stec_obs - stec_slm` | TECU |

## Current Quality Control

- `stec_obs` is not NaN.
- `elevation >= 10 deg`.
- `ipp_lon` and `ipp_lat` lie within `80E-140E`, `0N-60N`.
- `0 <= stec_obs <= 600 TECU`.
- Satellite position successfully matched.
