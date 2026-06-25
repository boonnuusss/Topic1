# IRI-backed Simulation Schema

## Purpose

This schema defines the controlled IRI simulation dataset used to compare
classical single-layer STEC-to-VTEC conversion with ML-assisted conversion.

The scientific target is VTEC-space error:

```text
SLM baseline: vtec_slm_from_iri_stec = stec_iri_observable / M(elevation)
Reference:    vtec_iri_ref
Error:        vtec_slm_from_iri_stec - vtec_iri_ref
```

Machine-learning models should be judged by whether their STEC modeling reduces
the final VTEC conversion error relative to this SLM baseline.

## References From Existing ML_sTEC Work

The schema is based on these existing notebooks and helper scripts:

| Existing file | Reused idea | Schema correction |
| --- | --- | --- |
| `../00_iri/mycode/ML_sTEC/generate_sTEC_obs_regional.ipynb` | Regional ray sample fields: `slantTEC`, receiver/satellite coordinates, IPP coordinates, PRN, target time. | Use row-wise parquet with explicit column names instead of loose npz arrays. |
| `../00_iri/mycode/ML_sTEC/generate_sTEC_obs_iri_regional.ipynb` | PyIRI electron-density grid and full ray integration with `RegularGridInterpolator` and trapezoidal integration. | Store integration settings and diagnostics in metadata/columns. |
| `../00_iri/mycode/ML_sTEC/gen_iri_val_dataset.ipynb` | IRI VTEC reference at IPP by vertical integration. | Rename output to `vtec_iri_ref`; do not store VTEC under a `slantTEC` key. |
| `../00_iri/mycode/ML_sTEC/Read_sp3_XYZ.py` | SP3 reading, ECEF conversion, and cross-constellation PRN mapping. | Keep the same PRN offsets as Topic1. |

Existing npz files are useful for validation, but they should not define the new
storage schema. In particular, `iri_vTEC_*.npz` stores VTEC in a variable named
`slantTEC` and uses `(3, N)` coordinate arrays, while obs/IRI STEC npz files use
`(N, 3)`. Topic1 should use explicit table columns and `(N rows) x (columns)`.

## Output Files

Recommended first implementation output:

```text
datasets/processed/iri_simulation_rayvtec_YYYYMMDD.parquet
datasets/processed/iri_simulation_rayvtec_YYYYMMDD.metadata.json
```

Optional heavy diagnostic output, only for small smoke tests:

```text
datasets/interim/iri_simulation_ray_diagnostics_YYYYMMDD.parquet
```

Do not store full electron-density profiles or all ray samples in the main
processed table. Store them only in interim diagnostics when needed.

## Builder Entry Point

Implemented entry point:

```powershell
python -m src.iri_simulation_builder
```

Recommended smoke-test command:

```powershell
python -m src.iri_simulation_builder `
  --input datasets/processed/raystec_china_20240505.parquet `
  --output datasets/interim/iri_simulation_smoke_3row.parquet `
  --simulation-id iri_rayvtec_20240505_smoke_3row `
  --max-rows 3 `
  --max-epochs 1 `
  --ray-sample-spacing-km 50
```

The default implementation uses `row_adaptive_point_profiles`: for each epoch,
it queries PyIRI at the row IPP locations and at the in-grid ray sample
locations. This avoids building a full regional 3-D grid for smoke tests while
retaining the same ECEF ray-integration logic.

For dense ray sampling, use `--grid-strategy regular_grid_interpolator`. This
matches the legacy notebook pattern under `00_iri/mycode/ML_sTEC`: PyIRI is run
once on a regional lon-lat-alt grid, then scipy `RegularGridInterpolator` is
used for receiver-satellite ray integration and IPP VTEC interpolation. This is
the preferred strategy for tests such as `--ray-sample-spacing-km 2.5`.

## Row Definition

Each row is one receiver-satellite ray at one UTC epoch, inherited from the
Topic1 ray-wise table.

Primary key:

```text
timestamp + station + prn
```

If duplicate rays exist for the same key, add `source_row_id` from the input
table and treat it as part of the unique key.

## Required Input Columns

The builder should start from the current Topic1 ray-wise parquet:

```text
datasets/processed/raystec_china_YYYYMMDD.parquet
```

Required columns:

| Column | Unit | Use |
| --- | --- | --- |
| `timestamp` | datetime UTC | IRI time and row key |
| `station` | string | row key and grouping |
| `prn` | integer | row key and grouping |
| `constellation` | string | grouping |
| `rec_x`, `rec_y`, `rec_z` | meter | ray start for IRI STEC integration |
| `rec_lon`, `rec_lat`, `rec_h` | degree, degree, meter | receiver metadata and fallback coordinate source |
| `sat_x`, `sat_y`, `sat_z` | meter | ray end for IRI STEC integration |
| `sat_lon`, `sat_lat`, `sat_h` | degree, degree, km | satellite metadata |
| `ipp_lon`, `ipp_lat` | degree | vertical VTEC reference location |
| `elevation` | degree | SLM mapping factor |
| `azimuth` | degree | ML feature and QC |
| `stec_obs` | TECU | optional observed comparison, not simulation truth |

## Main Output Columns

### Identity And Geometry

Keep the existing ray-wise identity and geometry columns from the input table.
Add:

| Column | Unit / Type | Meaning |
| --- | --- | --- |
| `source_row_id` | integer | Original row number in the input parquet. |
| `simulation_id` | string | Stable label for the IRI simulation run. |
| `simulation_role` | string | Usually `iri_reference`. |

### IRI Reference Targets

| Column | Unit / Type | Meaning |
| --- | --- | --- |
| `stec_iri_ray` | TECU | Full receiver-satellite ray integral through the IRI electron-density field. |
| `vtec_iri_ref` | TECU | Vertical TEC reference at the IPP longitude/latitude and epoch. |
| `vtec_iri_ref_method` | string | `vertical_ipp_trapezoid`. |
| `stec_iri_method` | string | `ecef_ray_integral`. |

### SLM Conversion Baseline On Simulated STEC

| Column | Unit / Type | Meaning |
| --- | --- | --- |
| `slm_shell_height_km` | km | Thin-shell height for SLM conversion, default `450`. |
| `slm_mapping_factor_iri` | unitless | `M(elevation)` computed using `slm_shell_height_km`. |
| `vtec_slm_from_iri_stec` | TECU | `stec_iri_observable / slm_mapping_factor_iri`; equal to `stec_iri_ray / slm_mapping_factor_iri` for the first no-bias run. |
| `vtec_slm_error_iri` | TECU | `vtec_slm_from_iri_stec - vtec_iri_ref`. |
| `vtec_slm_abs_error_iri` | TECU | Absolute value of `vtec_slm_error_iri`. |

### Optional Bias And Noise Columns

The first simulation should use `bias_mode = none`. Add bias/noise later only
when the no-bias pipeline is stable.

| Column | Unit / Type | Meaning |
| --- | --- | --- |
| `bias_mode` | string | `none`, `receiver_satellite_constant`, or future mode. |
| `receiver_bias_tecu` | TECU | Simulated receiver bias added to STEC; default `0`. |
| `satellite_bias_tecu` | TECU | Simulated satellite bias added to STEC; default `0`. |
| `noise_tecu` | TECU | Simulated random noise added to STEC; default `0`. |
| `stec_iri_observable` | TECU | `stec_iri_ray + receiver_bias_tecu + satellite_bias_tecu + noise_tecu`. |

For the first no-bias run:

```text
stec_iri_observable = stec_iri_ray
```

## Integration Assumptions

### IRI Electron-density Grid

Initial legacy-compatible settings, matching the existing notebooks:

| Setting | Value |
| --- | --- |
| longitude range | `60E` to `160E` |
| longitude spacing | `1 deg` |
| latitude range | `-20N` to `80N` |
| latitude spacing | `0.5 deg` |
| altitude range | `90 km` to `2000 km` |
| altitude spacing | `5 km` |
| temporal samples | unique UTC epochs from the ray-wise table, or a configured subset |
| IRI call | `PyIRI.edp_update.IRI_density_1day` |
| coefficient set | `ccir_or_ursi = 0` unless configured otherwise |
| electron-density unit | `m^-3` |

The builder must record `f107`, coefficient directory, PyIRI version if
available, grid ranges, and grid spacings in metadata.

### Full Ray STEC Integral

Use the existing notebook logic as the first implementation reference:

1. Convert each receiver-satellite ECEF line segment to geodetic sample points.
2. Query PyIRI electron-density profiles at the in-grid sample lon/lat points.
3. Interpolate each vertical profile to the corresponding sample altitude.
4. Fill below/above the configured IRI altitude range with `0.0`.
5. Integrate with the trapezoidal rule:

```text
stec_iri_ray = integral(Ne ds) / 1e16
```

Initial settings:

| Setting | Value |
| --- | --- |
| ray start | receiver ECEF coordinates |
| ray end | satellite ECEF coordinates |
| nominal sample spacing | CLI `--ray-sample-spacing-km`, default `2.5 km`; smoke tests may use coarser spacing such as `50 km` |
| `dalt_km` | `5` for the PyIRI vertical grid |
| path sampling | `sample_alpha = 2.0` for legacy compatibility, with a future sensitivity test using uniform sampling |
| out-of-grid electron density | `0.0` |
| integral method | trapezoidal over ECEF segment lengths |

The main processed table should store only compact diagnostics:

| Column | Unit / Type | Meaning |
| --- | --- | --- |
| `ray_integral_n_samples` | integer | Number of integration samples. |
| `ray_integral_min_alt_km` | km | Minimum sampled altitude. |
| `ray_integral_max_alt_km` | km | Maximum sampled altitude. |
| `ray_integral_path_length_km` | km | Full receiver-satellite path length used for sampling. |
| `ray_integral_outside_grid_fraction` | unitless | Fraction of samples outside the IRI grid. |

### Vertical VTEC Reference

For each row's IPP:

1. Use `(ipp_lon, ipp_lat)` as the vertical column location.
2. Integrate IRI electron density over the same altitude grid used for ray
   integration.
3. Convert to TECU:

```text
vtec_iri_ref = integral(Ne dh) / 1e16
```

Recommended initial vertical integral:

| Setting | Value |
| --- | --- |
| vertical bottom | `90 km` |
| vertical top | `2000 km` |
| vertical step | `5 km` |
| horizontal location | IPP lon/lat |
| out-of-grid electron density | `0.0` |

The old notebook integrates from IPP altitude `0 km` toward `20200 km`, with
electron density outside the IRI grid set to zero. The new implementation should
prefer explicit `90-2000 km` limits and record them.

## Metadata JSON

The metadata file must include:

```json
{
  "simulation_id": "iri_rayvtec_20240505_v001",
  "created_at_utc": "...",
  "input_file": "datasets/processed/raystec_china_20240505.parquet",
  "output_file": "datasets/processed/iri_simulation_rayvtec_20240505.parquet",
  "row_counts": {
    "input": 199087,
    "output": 199087,
    "dropped_missing_geometry": 0,
    "dropped_iri_failure": 0
  },
  "iri": {
    "engine": "PyIRI",
    "call": "PyIRI.edp_update.IRI_density_1day",
    "ccir_or_ursi": 0,
    "f107_source": "config",
    "f107": 204.0,
    "coeff_dir": "...",
    "grid": {
      "lon_min": 60.0,
      "lon_max": 160.0,
      "dlon": 1.0,
      "lat_min": -20.0,
      "lat_max": 80.0,
      "dlat": 0.5,
      "alt_min_km": 90.0,
      "alt_max_km": 2000.0,
      "dalt_km": 5.0
    }
  },
  "integration": {
    "ray_method": "ecef_straight_line_trapezoid",
    "vertical_method": "ipp_vertical_trapezoid",
    "sample_alpha": 2.0,
    "outside_grid_fill_value": 0.0,
    "tecu_scale": 1e16
  },
  "slm_baseline": {
    "formula": "vtec_slm_from_iri_stec = stec_iri_observable / slm_mapping_factor_iri",
    "shell_height_km": 450.0,
    "mapping_function": "1 / sqrt(1 - (Re / (Re + H) * cos(E))^2)",
    "earth_radius_km": 6371.0
  },
  "bias": {
    "bias_mode": "none",
    "receiver_bias_tecu": 0.0,
    "satellite_bias_tecu": 0.0,
    "noise_tecu": 0.0,
    "random_seed": null
  },
  "legacy_references": [
    "../00_iri/mycode/ML_sTEC/generate_sTEC_obs_iri_regional.ipynb",
    "../00_iri/mycode/ML_sTEC/gen_iri_val_dataset.ipynb"
  ]
}
```

## Quality Control

Required QC before writing the processed simulation table:

- Required geometry columns are finite.
- `elevation >= configured_min_elevation_deg`.
- IRI grid covers all IPP lon/lat with a configurable margin.
- `stec_iri_ray >= 0`.
- `vtec_iri_ref >= 0`.
- `slm_mapping_factor_iri > 0`.
- Rows with failed IRI interpolation/integration are counted in metadata.

Recommended diagnostic summaries:

- Distribution of `stec_iri_ray`, `vtec_iri_ref`, and
  `vtec_slm_from_iri_stec`.
- Distribution of `vtec_slm_error_iri`.
- Error by elevation bin.
- Error by station and PRN.
- Spatial map of `vtec_slm_error_iri`.

## First Implementation Milestones

1. Implement a small-row smoke test using the existing 2024-05-05 ray-wise
   parquet.
2. Reproduce the old npz field logic on a tiny subset for sanity:
   `stec_iri_ray` should correspond to old `iri_sTEC_*.npz/slantTEC`, while
   `vtec_iri_ref` should correspond to old `iri_vTEC_*.npz/slantTEC`.
3. Write the row-wise parquet and metadata JSON.
4. Add VTEC-space SLM diagnostic metrics.
5. Only after this is stable, add RayMLP experiments for ML-assisted conversion.
