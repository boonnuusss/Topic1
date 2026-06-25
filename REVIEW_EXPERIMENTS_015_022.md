# Deep Review: Experiments 015-022 and Related Code

**Date:** 2026-06-25  
**Scope:** IRI simulation builder, RayMLP simulation training, FNN simulation training, VTEC evaluator, and all shared utilities  
**Reviewer:** Claude (Cowork agent)

---

## 1. Overall Architecture

The data flow is clean and well-separated:

```
raw .mat + SP3 → dataset_builder → raystec_china.parquet (199k rows)
                                          │
                            ┌─────────────┼──────────────┐
                            ▼             ▼              ▼
                      slm_vtec.py   iri_simulation    (legacy slm/
                                  _builder.py         residual code)
                            │             │
                            ▼             ▼
                      vtec_slm.     iri_simulation
                      parquet       .parquet
                                       │
                            ┌──────────┼──────────┐
                            ▼                     ▼
                 train_ray_mlp_           train_fnn_
                 simulation.py            simulation.py
                            │                     │
                            ▼                     ▼
                       predictions           predictions
                       .parquet              .parquet
                            │                     │
                            └──────────┬──────────┘
                                       ▼
                                  evaluate.py
                                       │
                                       ▼
                               vtec_metrics.csv
```

Each stage is independently runnable, produces metadata JSON, and records provenance. This is solid engineering.

---

## 2. Scientific Problem: The Conversion Gap

Experiments 019-022 converge on the same finding: training a model exclusively on identity pairs (`source_slant_to_same_slant`) does not teach it to generalize to non-identity target rays. This is not a code bug — the experiment log explicitly diagnoses it.

The mechanism:

- During training, every pair has `source_geometry == target_geometry` and the label is `stec_iri_ray`. The model learns: "given source STEC and source geometry, reproduce the same STEC for an identical target ray." This is effectively learning a consistency check, not a geometric transformation.
- During VTEC inference, the target ray is vertical (elevation=90°, sat_h=2000 km) while the source is still the GNSS slant ray. The model has never seen any example where the target geometry differs from the source geometry, so it cannot learn the STEC scaling that accompanies a geometry change.
- The result: `stec_ray_mlp` (slant-to-slant) is accurate (RMSE 1.27 TECU), but `vtec_ray_mlp` (slant-to-vertical) is terrible (RMSE 24.08 TECU vs SLM 2.44 TECU).

This is correctly identified as a *modeling/design* issue rather than an implementation bug. However, several implementation details amplify the problem, as described below.

---

## 3. Critical Code Issues

### 3.1 No Non-Negativity Constraint on STEC Outputs

**Files:** `src/train_ray_mlp_simulation.py` (line 441-448), `src/train_fnn_simulation.py` (line 218-233)  
**Severity:** High — directly affects VTEC conversion quality

Both RayMLP (`build_ray_mlp`) and FNN (`build_fnn_stec`) use a final `nn.Linear(N, 1)` with no activation function. The output can be any real number. During training on slant-ray STEC (always positive, typically 20-150 TECU), the model likely stays in the positive range. But when queried with the vertical target geometry — which has `target_sat_h=2000` instead of the training distribution's ~20,000 km — the model can produce negative or near-zero predictions.

The conversion ratio logic (lines 462-469 of `train_ray_mlp_simulation.py`):

```python
ratio = np.divide(vtec_pred, slant_pred, ...)
vtec_ray_mlp = source_stec * ratio
```

If `stec_ray_mlp` (the slant prediction) is small or negative, the ratio explodes or flips sign. A negative ratio multiplied by positive `source_stec` produces a negative VTEC, which is physically impossible.

**Fix:** Apply `torch.nn.functional.softplus` (or at minimum `torch.clamp(min=0)`) on the final layer output. Since the training labels are positive STEC, a softplus activation would be physically appropriate and would prevent sign-flip errors in the ratio.

The same issue affects the FNN pipeline (lines 238-249 of `train_fnn_simulation.py`).

### 3.2 Dead `--include-identity-pairs` Argument

**File:** `src/train_ray_mlp_simulation.py` (lines 646-650)  
**Severity:** Low — dead code, but misleading

```python
parser.add_argument(
    "--include-identity-pairs",
    action="store_true",
    help=argparse.SUPPRESS,
)
```

This argument is parsed but never referenced in `train_ray_mlp_simulation()`. It is a vestige of experiment 018 (the superseded design where vertical target labels contributed to training). Since it's suppressed from help output, users won't discover it, but code readers will wonder why it exists.

**Fix:** Remove it.

### 3.3 pyproj.Transformer Repeated Instantiation

**File:** `src/geometry.py` (line 19), `src/io_sp3.py` (line 49), `src/iri_simulation_builder.py` (line 787)  
**Severity:** Medium — performance, O(n) overhead where n is rows

Every call to `geodetic_to_ecef()` creates a new `Transformer.from_crs(...)`, which reads PROJ data files from disk:

```python
def geodetic_to_ecef(lon_deg, lat_deg, height_m=0.0):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    return transformer.transform(lon_deg, lat_deg, height_m)
```

This is fine for one-off use, but `build_vertical_target_geometry()` in `train_ray_mlp_simulation.py` calls it twice for every test row (once for receiver, once for "satellite" at 2000 km). For 419 test rows, that's ~838 Transformer instantiations. For larger test sets (e.g., 18,610 rows from experiment 021), this becomes noticeable.

The same pattern exists in `build_vertical_fnn_features()` in `train_fnn_simulation.py`.

The IRI simulation builder does this correctly — it creates one `ecef2geo` Transformer at line 787 and reuses it across all epochs. All other files should follow this pattern.

### 3.4 Fragile Import Fallback Pattern

**Files:** Nearly every module that uses `try/except ImportError`  
**Severity:** Medium — correctness risk

The dual import pattern:

```python
try:
    from .geometry import compute_slant_factor
except ImportError:
    from geometry import compute_slant_factor
```

appears in `baselines.py`, `train_ray_mlp_simulation.py`, `train_fnn_simulation.py`, `evaluate.py`, `iri_simulation_builder.py`, `train_mlp.py`, and `train_tree.py`. It exists because the modules can be run either as `python -m src.<module>` (relative imports) or `python src/<module>.py` (absolute imports).

The risk: if a different `geometry.py` exists elsewhere on `sys.path`, the absolute import silently picks it up. Given that the user's environment includes `../00_iri/mycode/ML_sTEC/` which might contain similarly-named modules, this is not theoretical.

**Fix:** Standardize on `python -m src.<module>` as the only supported invocation method, remove the fallback imports, and document this in `AGENT.md`. Add a `if __name__ == "__main__" and __package__ is None` check at the top of entry-point files to fail with a clear message if someone tries to run them directly.

---

## 4. Design Analysis Per Experiment

### 4.1 Experiment 015: UT05:00 IRI Simulation (Regular Grid)

**Code:** `src/iri_simulation_builder.py`, `build_epoch_simulation_regular_grid()`

The implementation is solid. Key observation: in the regular grid strategy, `vtec_iri_ref` is obtained by interpolating the VTEC grid at IPP locations using `RegularGridInterpolator` (lines 533-538):

```python
vtec_interp = RegularGridInterpolator(
    (lon_vals, lat_vals),
    vtec_grid,
    bounds_error=False,
    fill_value=np.nan,
)
out["vtec_iri_ref"] = vtec_interp(ipp_points)
```

One subtlety: `vtec_iri_ref` is obtained by 2D interpolation of pre-integrated VTEC, while `stec_iri_ray` is obtained by 3D interpolation of electron density along the full ray. These use the same underlying IRI electron density grid, so they are physically consistent. However, the 2D VTEC interpolation can smooth out small-scale horizontal gradients compared to directly integrating at each IPP. For IPP locations between grid points, the interpolated VTEC is a weighted average of neighboring column integrals, which is slightly different from the column integral that would be obtained if IRI were queried directly at the IPP. This is an acceptable approximation given the 1° × 0.5° grid resolution.

The experiment log correctly notes that low-elevation rays ([10,20) deg bin) have the largest SLM VTEC error (MAE 5.19 TECU), confirming that the thin-shell assumption breaks down most at low elevation.

### 4.2 Experiments 017-021: RayMLP Simulation

**Code:** `src/train_ray_mlp_simulation.py`

The feature engineering is sophisticated. Each training pair combines:
- 16 source geometry features (receiver, satellite, IPP, elevation, azimuth, slant factor, ray length, LOS vector)
- 6 time features (cyclical UTC hour, DOY, IPP local time)
- 6 constellation one-hot features
- 1 source STEC value
- 16 target geometry features

Total: 45 features.

**Design question:** Including `slant_factor` (the SLM mapping factor) as a feature and `source_stec` as an input means the model receives a strong hint about the SLM conversion. However, since the model is trained only on identity pairs, this doesn't hurt or help the conversion — the model can just learn to reproduce the target STEC from the source + geometry features.

One significant observation: `target_sat_h` is 2000 km for the vertical ray but ~20,000 km for GNSS rays. Since the training data only contains GNSS-ray values for `target_sat_h`, the vertical ray's `target_sat_h=2000` is far outside the training distribution. With StandardScaler normalization, this value maps to roughly (2000 - mean_sat_h) / std_sat_h standard deviations, likely around -3 to -5 sigma. The model has no training examples in this region of feature space.

### 4.3 Experiment 022: FNN Simulation

**Code:** `src/train_fnn_simulation.py`

Three feature sets were tested:

**`ecef6`**: `[rec_x, rec_y, rec_z, sat_x, sat_y, sat_z]`

This is the worst-performing for VTEC (RMSE 75.45 TECU). The reason is geometric: GNSS satellite ECEF coordinates have magnitudes of ~26,000 km (satellite orbit radius), while the vertical "satellite" at 2000 km altitude has an ECEF magnitude of ~8,371 km. The model is trained on features at the ~26,000 km scale and queried at ~8,371 km — this is an extreme distribution shift.

**`blh6`**: `[rec_lon, rec_lat, rec_h, sat_lon, sat_lat, sat_h]`

Better than `ecef6` (RMSE 16.48 TECU) because lon/lat are bounded and `sat_h` is the only strongly-shifted feature. The model still struggles because training `sat_h ~20,000` vs inference `sat_h=2000` is a 10x difference.

**`bl4`**: `[rec_lon, rec_lat, sat_lon, sat_lat]`

Best of the three (RMSE 9.15 TECU), but still much worse than SLM (2.44 TECU). By removing all height information, the model cannot distinguish between a GNSS ray and a vertical ray at the same horizontal location. Since `rec_lon==sat_lon` and `rec_lat==sat_lat` for the vertical ray, the model receives identical features for what should be very different STEC values. The FNN essentially learns a location-dependent STEC baseline, which is why its performance is better than models that get confused by the height mismatch — but it's still far from useful.

---

## 5. The Ratio Conversion: Mathematical Analysis

Both RayMLP and FNN use the same VTEC conversion formula:

```
vtec_ml = source_stec * (vtec_direct / stec_ml)
```

where `vtec_direct` is the model's prediction for the vertical target ray and `stec_ml` is the prediction for the original slant ray.

In the ideal case where the model perfectly predicts STEC for any target geometry:
- `stec_ml = stec_iri_ray` (true slant STEC)
- `vtec_direct = vtec_iri_ref` (true vertical STEC)
- `vtec_ml = stec_iri_observable * (vtec_iri_ref / stec_iri_ray)`

But since `stec_iri_ray / slm_mapping_factor_iri` is the classical SLM estimate, the ideal ratio `vtec_iri_ref / stec_iri_ray` is approximately `1 / slm_mapping_factor_iri` (modulo the thin-shell approximation error). So the conversion ratio approach is mathematically equivalent to the SLM when the model is perfect. The model's job is to estimate this ratio better than `1 / M(elevation)`.

The experiment log shows the learned mean ratio is 0.8045 vs the true mean ratio of 0.7121 (experiment 021). This overestimation means the model systematically predicts too little STEC attenuation from slant to vertical, resulting in VTEC overestimation (positive bias of 11.99 TECU).

---

## 6. Shared Utility Code Issues

### 6.1 `standardize_xy` Return Signature

**File:** `src/train_mlp.py` (lines 71-88)

```python
def standardize_xy(x_train, y_train, x_test):
    ...
    return x_train_scaled, y_train_scaled, x_test_scaled, x_imputer, x_scaler, y_scaler
```

Returns 6 values with no type hints beyond what can be inferred. The callers unpack these consistently, so it works, but adding a `TypedDict` or named tuple return would improve readability.

### 6.2 `make_loader` BatchNorm Safety

**File:** `src/train_mlp.py` (lines 91-112)

```python
drop_last = len(dataset) > batch_size and len(dataset) % batch_size == 1
```

This correctly drops the last batch when it would contain exactly 1 sample, preventing a BatchNorm1d error. However, the condition `len(dataset) % batch_size == 1` doesn't cover the case where `len(dataset) < 2` — but that's already caught by the `len(x_train) < 2` check earlier in both callers.

### 6.3 `choose_simulation_split` vs `choose_split`

**Files:** `src/train_ray_mlp_simulation.py` (lines 284-358), `src/train_tree.py` (lines 61-97)

The simulation split function extends the tree split function with two additional modes:
- **key holdout** (`--test-keys-from`): holds out specific rows by key columns (timestamp, station, PRN)
- **timestamp holdout** (`--test-timestamp`): holds out all rows from a specific timestamp

The key holdout implementation (lines 293-332) uses a merge-based approach:

```python
merged = left.reset_index().merge(
    right.assign(**{marker: True}),
    on=key_columns,
    how="left",
    sort=False,
)
test_mask = merged[marker].fillna(False).to_numpy(bool)
```

This is robust. One consideration: when `timestamp` is a key column, both `left["timestamp"]` and `right["timestamp"]` are converted to `pd.Timestamp` via `pd.to_datetime()`. The merge should work correctly on datetime columns. But if the key columns include floating-point columns (not the case here, as keys are `timestamp,station,prn`), floating-point merge could miss some matches due to precision.

---

## 7. VTEC Evaluator Analysis

**File:** `src/evaluate.py`

### 7.1 Elevation Binning (Corrected from Previous Review)

In `build_metric_rows()` (lines 215-238), the elevation binning uses:

```python
pd.cut(
    grouped["elevation"],
    bins=elevation_bins,
    labels=labels,
    right=False,
    include_lowest=True,
)
```

With `right=False` and `include_lowest=True`, the intervals are `[a, b)`. This is different from the `metrics.py` `elevation_bin_metrics` function (used by legacy code) which uses `right=True` (the default). The evaluate.py version is actually more correct for scientific interpretation — elevation 20.0 goes in the `[20, 30)` bin, etc.

But the default bin edges are `10,20,30,40,50,60,70,80,90` — so the highest bin is `[80, 90)` which excludes elevation=90.0 exactly. Since the minimum elevation filter is 10°, and elevation can't practically be exactly 90.0 for GNSS rays, this is unlikely to matter.

### 7.2 Baseline Comparison Logic

```python
def add_baseline_comparison(metrics, *, baseline_model="slm"):
    ...
    out["rmse_reduction_vs_slm_pct"] = (
        (out["baseline_rmse_tecu"] - out["rmse_tecu"])
        / out["baseline_rmse_tecu"]
        * 100.0
    )
```

This is correct: positive means the model's RMSE is lower than SLM's (improvement). The experiment 018 log shows RayMLP with +31.95% RMSE reduction, which is consistent with this formula.

However, the comparison is computed by merging on group columns (scope, elevation_bin_deg, station, utc_hour). If a model's metrics row doesn't have a corresponding SLM row for the same group (e.g., if SLM was not evaluated in that group), the merge produces NaN for the baseline columns and the delta columns become NaN. This is handled correctly by numpy's `divide="ignore"`.

### 7.3 STEC Prediction Column Handling

The evaluator supports STEC prediction columns via `--stec-prediction-columns`. It converts them to VTEC by dividing by `slm_mapping_factor_iri`:

```python
for column in stec_prediction_columns:
    derived_column = f"vtec_from_{column}"
    out[derived_column] = pd.to_numeric(out[column], errors="coerce") / mapping
```

This is a useful feature, but the RayMLP and FNN experiments pass their output as `--vtec-prediction-columns` (not `--stec-prediction-columns`) because the training scripts already apply the ratio conversion. This is the correct usage.

---

## 8. Summary of Issues by Severity

### Must Fix (affects correctness or results)

1. **No non-negativity constraint on STEC model outputs.** The model can predict negative STEC for OOD vertical rays, corrupting the ratio conversion. Add `softplus` or `ReLU` on the output layer. *(train_ray_mlp_simulation.py, train_fnn_simulation.py, models/mlp.py)*

### Should Fix (improves robustness)

2. **Dead `--include-identity-pairs` argument.** Remove it to avoid confusion. *(train_ray_mlp_simulation.py)*

3. **pyproj.Transformer repeated instantiation.** Cache the Transformer at module level in `geometry.py` — impacts `build_vertical_target_geometry()` and `build_vertical_fnn_features()` which are called on every test row. *(geometry.py)*

4. **Fragile import fallbacks.** Remove the `except ImportError` absolute-import fallbacks in all files. Standardize on `python -m src.<module>`. *(baselines.py, train_ray_mlp_simulation.py, train_fnn_simulation.py, evaluate.py, iri_simulation_builder.py, train_mlp.py, train_tree.py)*

### Consider Fixing (design improvements)

5. **The training pair design.** The current approach of training only with identity pairs fundamentally cannot teach the model the geometry-to-STEC mapping needed for VTEC conversion. The next iteration needs: (a) training pairs with varied target geometries (not just identity), using `stec_iri_ray` computed for those target geometries; or (b) a physics-informed architecture that analytically enforces the elevation-dependent mapping factor; or (c) direct VTEC supervision on the vertical query, acknowledging that this uses `vtec_iri_ref` during training (the current prohibition on using `vtec_iri_ref` as a training label was introduced to prevent the model from "cheating," but the alternative — identity-only training — has proven worse than SLM).

6. **Vertical top parameter.** The default of 2000 km matches the IRI grid top, but the SLM uses a 450 km shell. The vertical target ray from 0 to 2000 km integrates more electron content than the SLM shell height implies. While the ratio conversion should compensate, the mismatch between the physical model (thin shell at 450 km) and the ML model (ray to 2000 km) is a conceptual inconsistency that should be explicitly discussed.

### Observations (no action needed)

7. The **regular grid VTEC interpolation** in the IRI simulation builder is a reasonable approximation at 1° × 0.5° resolution, but could smooth small-scale VTEC gradients. Document this as a known limitation.

8. The **FNN `bl4` feature set** produces the best FNN VTEC result because it avoids the height-magnitude OOD problem entirely, but it does so by discarding all geometry information — making it essentially a location-only STEC baseline. This is a useful diagnostic but not a viable path forward.

9. The **experiment log documentation is excellent** — the honest recording of negative results in experiments 019-021 is a model for scientific computing.
