# AGENT.md

## Project Goal

Build a reproducible ray-wise GNSS STEC dataset and modeling pipeline for
regional ionosphere research. The central task is to predict observed slant TEC
directly at the receiver-satellite ray level, rather than first mapping
observations to a regular VTEC grid.

## Scientific Background

Each sample represents one GNSS receiver, one satellite PRN, and one UTC epoch.
The current scientific hypothesis is that explicit ray geometry, IPP location,
time encoding, and physical model priors can improve STEC prediction compared
with a traditional single-layer ionospheric projection baseline.

Modeling should proceed in this order:

1. Traditional single-layer model baseline:
   `STEC_slm = M(elevation) * VTEC(IPP, time)`.
2. Direct tabular ML baselines, starting with Ridge and RandomForest.
3. Residual learning relative to the SLM baseline.
4. RayMLP only after the physical and tabular baselines are stable.

## Required Reading Before Work

Before editing or running experiments, read:

1. `RESEARCH_PLAN.md`
2. `DATA_DICTIONARY.md`
3. `TODO.md`
4. `EXPERIMENT_LOG.md`
5. `FAILED_ATTEMPTS.md`

## Important Rules

1. Do not overwrite raw TEC `.mat`, SP3, NetCDF, PyIRI, coefficient, model, or
   source data files.
2. Keep scientific assumptions explicit: units, coordinate systems, PRN mapping,
   time basis, selection window, region, and quality thresholds.
3. Prefer reusable Python modules and command-line entry points over large
   notebook-only workflows.
4. Run a small test before full multi-day processing.
5. Update `EXPERIMENT_LOG.md` after any data build, model run, or figure
   generation.
6. Update `TODO.md` when tasks are completed, blocked, or reprioritized.
7. Do not invent references, numerical results, or physical interpretations.
8. Separate observation, inference, and speculation. Mark uncertain statements as
   `[UNCERTAIN]`.
9. Watch for data leakage, unit errors, coordinate errors, time conversion
   errors, hard-coded paths, and inefficient loops.
10. Treat generated datasets, figures, and checkpoints as artifacts. Do not
    delete or regenerate them unless the task explicitly asks for it.

## Python Execution Convention

Use the project virtual environment explicitly instead of relying on a bare
`python` command:

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe
```

For example:

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe -m src.dataset_builder --config configs/quick_test.yaml
```

Inside Codex, the default sandbox may refuse to launch the user's local Python
installation under `C:\Users\12928\AppData\Local\Programs\Python\Python312`.
If that happens, do not assume the environment is broken. Request escalated
execution for the project `.venv` interpreter and rerun the same command. A
previous approved smoke test was:

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe --version
```

which returned `Python 3.12.3`.

## Current Data Convention

- TEC data root: `E:/Scientific_Research/data/gnssTECmatFiles`
- Current processed day: `2024-05-05`, DOY `126`
- Region: `80E-140E`, `0N-60N`
- Epoch interval: `900 s`
- Selection window: `30 s`
- Minimum elevation: `10 deg`
- STEC range: `0-600 TECU`

## PRN Mapping

Keep this mapping consistent across TEC observations and SP3 data:

```text
GPS     -> PRN
Galileo -> PRN + 70
SBAS    -> PRN + 119
QZSS    -> PRN + 180
GLONASS -> PRN + 500
BeiDou  -> PRN + 700
```

## Preferred Output

- Clean Python code under `src/`
- Reproducible commands under `scripts/`
- Metadata-rich datasets under `datasets/processed/`
- Diagnostic or paper-quality figures under `results/figures/`
- Markdown experiment logs and concise explanations

## After Each Task

Report:

1. changed files
2. why each file changed
3. command used for validation
4. remaining risks or assumptions
