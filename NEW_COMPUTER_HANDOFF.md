# New Computer And Codex Handoff

This document is the operational checkpoint for moving Topic1 to another
Windows computer and continuing the project with a new Codex chat. The GitHub
repository contains source code, configuration, tests, experiment history, and
research decisions. Generated datasets, checkpoints, figures, and result tables
are intentionally not stored in Git.

## 1. Canonical Repository

```text
Repository: git@github.com:boonnuusss/Topic1.git
Branch: main
Git author: boonnuusss <liubowei23@mails.ucas.ac.cn>
```

Clone and verify:

```powershell
git clone git@github.com:boonnuusss/Topic1.git
cd Topic1
git status
git log -5 --oneline
```

If SSH authentication is not configured on the new computer, add its public
key to the `boonnuusss` GitHub account before cloning or pushing. Repository
visibility does not affect pushing as long as that account/key has write
permission.

## 2. Required Reading For A New Codex Chat

Read these files before editing code or running experiments:

1. `AGENT.md`: project rules, scientific goal, and Python execution convention.
2. `RESEARCH_PLAN.md`: hypotheses, publication protocol, and research risks.
3. `NEW_COMPUTER_HANDOFF.md`: environment and asset restoration instructions.
4. `DATA_DICTIONARY.md`: table columns, units, and provenance.
5. `IRI_SIMULATION_SCHEMA.md`: IRI ray-integration contract.
6. `TODO.md`: current priority and unfinished work.
7. `EXPERIMENT_LOG.md`: reproducible history through Experiment 062.
8. `FAILED_ATTEMPTS.md`: known implementation and modeling pitfalls.

Suggested first prompt on the new computer:

```text
Read AGENT.md, NEW_COMPUTER_HANDOFF.md, RESEARCH_PLAN.md, TODO.md,
EXPERIMENT_LOG.md, and FAILED_ATTEMPTS.md. Then inspect git status and the latest
commit. Do not rerun long experiments or change scientific protocols until you
have summarized the current state and confirmed which ignored data artifacts
are available locally.
```

## 3. Software Prerequisites

- Windows 10/11 and PowerShell 7 or Windows PowerShell 5.1.
- Git and OpenSSH.
- 64-bit Python 3.12.
- Enough disk space for external source data, generated parquet files, and
  checkpoints. These assets are larger than the Git repository.
- A C/C++ compiler may be needed because PyPI `nequick==1.0.0` is distributed
  as source. Use an existing compatible MSVC or MinGW toolchain; do not install
  a large compiler stack silently from an automated agent.

Allow activation scripts for the current PowerShell process if required:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 4. Recreate The Main Python Environment

Virtual environments are machine-specific and are excluded from Git. Recreate
the main environment rather than copying `.venv`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe --version
```

Activate it with:

```powershell
.\activate_topic1.ps1
```

Agents should call `.\.venv\Scripts\python.exe` explicitly. If Codex cannot
launch the interpreter because of sandbox permissions, request permission for
that project interpreter; do not conclude that the environment is corrupt.

## 5. Recreate The Isolated NeQuick-G Environment

NeQuick-G must remain isolated because `nequick==1.0.0` uses NumPy 1.x while
the main ML environment may use NumPy 2.x:

```powershell
.\scripts\install_nequick.ps1
.\scripts\verify_nequick_install.ps1
.\activate_nequick.ps1
```

Expected core versions are recorded in `requirements-nequick.txt`, including
`numpy==1.26.4` and `nequick==1.0.0`. Never install NeQuick into `.venv` and
never upgrade `.venv-nequick` to NumPy 2.x without a compatibility test.

## 6. Local Paths That Must Be Reviewed

The committed YAML files contain paths from the original computer:

```text
configs/quick_test.yaml
configs/full_day_5min.yaml
```

Update `tec_data_dir` to the new local TEC root. Confirm that `sp3_files` points
to the shared `00_iri/mycode/ML_sTEC/Mydata/WUM2024(126)0505.nc` file. The
relative SP3 path assumes this layout:

```text
STEC_ML/
  00_iri/
  Topic1/
```

Do not move or commit source data merely to make a path work. Prefer changing a
local configuration value while preserving its scientific meaning.

## 7. Assets Not Stored In GitHub

`.gitignore` excludes `.venv*`, `datasets/raw`, `datasets/interim`,
`datasets/processed`, `results/tables`, `results/figures`,
`results/checkpoints`, `results/logs`, and `manuscript`.

For the fastest exact continuation, transfer these assets separately from the
old computer or a research storage location:

| Priority | Asset | Purpose |
| --- | --- | --- |
| Required for raw rebuild | GNSS TEC `.mat` tree for 2024/126 | Ray-wise observation builder |
| Required for raw rebuild | `00_iri/.../WUM2024(126)0505.nc` | Satellite geometry/SP3 input |
| Recommended | `datasets/interim/raystec_china_20240505_0000_2400_5min.parquet` | Full-day 5 min ray table |
| Recommended | `datasets/interim/iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet` | IRI simulation experiments |
| Recommended | `datasets/interim/nequickg_simulation_20240505_0000_2400_5min.parquet` | NeQuick-G experiments |
| Required by current validation wrappers | `results/tables/set_ratio_mlp_0500_anchor1_e100_predictions.parquet` | Historical UT05 validation keys |
| Useful for exact audit | `results/tables/` | Existing metrics and validation predictions |
| Useful for resuming models | `results/checkpoints/` | Saved model/scaler states |
| Private, optional | `manuscript/` | Local manuscript drafts; intentionally never pushed |

Copy artifacts only after cloning so the Git working tree remains clean. Do not
use `git add -f` to upload them. Before trusting copied parquet/checkpoint files,
compare file sizes and preferably SHA-256 hashes between computers.

## 8. Rebuild Order When Artifacts Are Not Copied

1. Edit the two YAML data roots.
2. Build the full-day 5 min ray table:

   ```powershell
   .\.venv\Scripts\python.exe -m src.dataset_builder --config configs/full_day_5min.yaml
   ```

3. Build the IRI simulation using the assumptions documented in
   `IRI_SIMULATION_SCHEMA.md` and the command patterns in `scripts/README.md`.
4. Download and verify Galileo broadcast coefficients:

   ```powershell
   .\scripts\download_nequickg_navigation.ps1
   ```

5. Build the NeQuick-G simulation:

   ```powershell
   .\scripts\build_nequickg_simulation.ps1
   ```

6. Run physical baselines and only then run ML training. Start with smoke tests;
   full-day ray integration and CPU training are long jobs.

## 9. Current Scientific State

The project goal is STEC modeling for STEC-to-VTEC conversion, not standalone
STEC prediction. Training must not use `vtec_iri_ref` or
`vtec_nequickg_ref` as a label. Simulated VTEC is evaluation-only.

The repeatedly reused UT05 key set is validation, not an independent test. It
has guided model, time-window, pair-budget, target-transform, shrinkage, and
epoch choices. Publication claims require a new independent-day locked test.

The latest fair NeQuick-G validation comparison uses elevation >= 30 degrees,
302 rays, and reports:

| Method | VTEC RMSE (TECU) | Role |
| --- | ---: | --- |
| NeQuick ideal ratio | approximately 0 | Oracle only |
| NeQuick-truth calibrated APMF | 0.5547 | Privileged simulation comparator |
| High-low CPP APMF, full-day | 1.0205 | STEC-only, broader calibration coverage |
| Set-ratio, UT04-08, 192k pairs, best checkpoint | 1.2918 | STEC-only ML validation |
| High-low CPP APMF, matched UT04-08 | 1.3259 | Fair time-window STEC-only comparator |
| SLM, 450 km | 1.9224 | Traditional physical baseline |

The matched-window comparison is the fairest current A/B: Set-ratio improves
RMSE over CPP APMF by about 2.57%, while CPP APMF has lower MAE and Set-ratio
has much smaller absolute bias. These are validation findings, not locked-test
results.

Experiment 062 implements the truth-free high-low coinciding-pierce-point APMF.
Its regularization is selected from held-out CPP proxy ratios only. The older
truth-calibrated APMF remains in tables as a privileged simulation reference and
must not be presented as deployable on real observations.

## 10. Exact Next Research Step

Create an independent-day dataset and locked-test key manifest, freeze all
current model and physical-baseline choices, and evaluate the selected
checkpoint exactly once. The locked day must be excluded from training,
scaler fitting, pair construction, ridge selection, shell-height selection,
epoch selection, and all hyperparameter decisions.

After that, run station, PRN, constellation, and time holdouts with multiple
random seeds and uncertainty intervals. Do not optimize further against the
historical UT05 validation set before the locked test is defined.

## 11. Minimal Health Check After Migration

```powershell
git status --short
.\.venv\Scripts\python.exe -m py_compile src\train_set_ratio_mlp.py src\physical_mapping.py src\physical_mapping_baselines.py
.\.venv\Scripts\python.exe -m unittest tests.test_set_ratio_protocol
.\.venv-nequick\Scripts\python.exe -m pytest tests\test_physical_mapping.py tests\test_nequickg_simulation.py -q
git diff --check
```

Expected test state at handoff: five set-ratio protocol tests pass and nine
NeQuick/physical-mapping tests pass. A clean clone has no generated results, so
tests that require transferred datasets should be run only after those assets
are present.
