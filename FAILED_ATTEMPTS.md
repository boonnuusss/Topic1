# Failed Attempts And Pitfalls

## Attempt 001: Wrong TEC Root Path

### Problem

The first quick build returned zero samples because `tec_data_dir` pointed to:

```text
D:/Scientific_Research/data/gnssTECmatFiles
```

### Cause

The actual local data directory was:

```text
E:/Scientific_Research/data/gnssTECmatFiles
```

### Fix

Updated `configs/quick_test.yaml`. The former multi-day config was later
removed when the project scope was narrowed to ordinary regional STEC baseline
modeling.
Added path existence checks in `src/dataset_builder.py`.

## Attempt 002: Re-reading Every Station File For Every Epoch

### Problem

The first implementation re-read all station `.mat` files for every epoch and
the one-day quick build timed out.

### Cause

The loop structure loaded daily station files inside the epoch loop.

### Fix

`build_day_dataset()` now reads each station file once per day, then slices the
preloaded DataFrames by epoch.

## Attempt 003: Cartopy Map Saved As Colorbar Only

### Problem

The Cartopy diagnostic figure was saved with only the colorbar visible.

### Cause

`bbox_inches="tight"` interacted poorly with Cartopy GeoAxes and gridlines.

### Fix

Use `fig.add_subplot(..., projection=ccrs.PlateCarree())`, `fig.colorbar(...)`,
and save without `bbox_inches="tight"`.

## Pitfall: Random Split Leakage

Random sample splits can place the same station, PRN, and nearby timestamps in
both train and test sets. Use random split only as a quick smoke test; report
time, station, and PRN holdout results for scientific claims.

## Pitfall: ML Baselines Before Physical Baselines

Starting with Ridge, RandomForest, or RayMLP alone makes it difficult to explain
whether the model improves on classical ionospheric practice. The current route
uses the single-layer model as the first baseline, then evaluates direct ML and
SLM residual learning against that reference.

## Pitfall: Confusing The SLM Direction

### Problem

The project briefly treated the single-layer model as a STEC prediction
baseline:

```text
STEC_slm = M(elevation) * VTEC
```

and then built direct/residual STEC prediction entry points around that target.

### Cause

That route answers a different question: "How well can the pipeline predict
observed STEC?" The current Topic1 question is instead: "Can STEC modeling
improve STEC-to-VTEC conversion?"

### Fix

The current mainline SLM baseline must use the inverse conversion:

```text
VTEC_slm = STEC_obs / M(elevation)
```

Use `python -m src.slm_vtec` for the mainline SLM VTEC product. Keep
`src.slm_baseline`, direct Ridge/RandomForest, residual Ridge/RandomForest, and
RayMLP residual learning as legacy/off-mainline code unless a task explicitly
reactivates them.

## Pitfall: Codex Sandbox Cannot Launch Local Python

### Problem

Codex reported that `Topic1/.venv/Scripts/python.exe` could not create a
process using the base interpreter:

```text
C:\Users\12928\AppData\Local\Programs\Python\Python312\python.exe
```

### Cause

The virtual environment was not broken. The default Codex sandbox could not
launch the user's local Python installation directly.

### Fix

Use the project `.venv` interpreter explicitly and request escalated execution
when Codex needs to run it:

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe --version
```

This succeeded and returned `Python 3.12.3`. Future commands should use the
same interpreter path instead of a bare `python`.
