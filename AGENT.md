# AGENT.md

## Project Goal

Build a reproducible ray-wise GNSS STEC dataset and modeling pipeline for
STEC-to-VTEC conversion research. The central task is to use STEC modeling to
improve or evaluate VTEC retrieval from slant observations, not to treat
standalone STEC prediction as the final scientific goal.

## Scientific Background

Each sample represents one GNSS receiver, one satellite PRN, and one UTC epoch.
The current scientific hypothesis is that explicit ray geometry, IPP location,
time encoding, and physical model priors can help reduce VTEC conversion error
relative to the traditional single-layer mapping model.

The current mainline should proceed in this order:

1. Traditional single-layer STEC-to-VTEC conversion baseline:
   `VTEC_slm = STEC_obs / M(elevation)`.
2. IRI-backed simulation experiment with documented reference VTEC and
   simulated ray-wise STEC labels.
3. Ray-wise ML STEC modeling, with RayMLP as the preferred neural entry point
   after simulation labels and SLM metrics are stable.
4. Main evaluation in VTEC space: SLM conversion error versus ML-assisted
   conversion error.

Do not frame the mainline as direct prediction of observed `stec_obs`. The
previous `stec_slm`, direct STEC prediction, and residual-learning entry points
are legacy/off-mainline unless a task explicitly reactivates them.

## Required Reading Before Work

Before editing or running experiments, read:

1. `RESEARCH_PLAN.md`
2. `NEW_COMPUTER_HANDOFF.md`
3. `DATA_DICTIONARY.md`
4. `IRI_SIMULATION_SCHEMA.md`
5. `TODO.md`
6. `EXPERIMENT_LOG.md`
7. `FAILED_ATTEMPTS.md`

## Important Rules

1. Do not overwrite raw TEC `.mat`, SP3, NetCDF, PyIRI, coefficient, model, or
   source data files.
2. Keep scientific assumptions explicit: units, coordinate systems, PRN mapping,
   time basis, selection window, region, quality thresholds, shell height, and
   mapping formula.
3. Prefer reusable Python modules and command-line entry points over large
   notebook-only workflows.
4. Run a small test before full multi-day processing.
5. Update `EXPERIMENT_LOG.md` after any data build, model run, or figure
   generation.
6. Update `TODO.md` when tasks are completed, blocked, or reprioritized.
7. Do not invent references, numerical results, or physical interpretations.
8. Separate observation, inference, and speculation. Mark uncertain statements as
   `[UNCERTAIN]`.
9. Watch for SLM direction errors, data leakage, unit errors, coordinate errors,
   time conversion errors, hard-coded paths, and inefficient loops.
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
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe -m src.slm_vtec --help
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

NeQuick-G commands are the exception. Use the isolated interpreter:

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv-nequick\Scripts\python.exe
```

It intentionally pins NumPy 1.26.4 because PyPI `nequick==1.0.0` is
incompatible with the main NumPy 2.x environment. Do not install NeQuick into
`.venv` or silently upgrade `.venv-nequick` to NumPy 2.x.

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
