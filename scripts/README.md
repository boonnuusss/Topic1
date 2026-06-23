# Scripts

This directory contains thin wrappers around reproducible command-line
pipelines. Keep scientific logic in `src/`; keep scripts small and explicit.

## Quick Dataset Build

```powershell
cd E:\Scientific_Research\work_2026\STEC_ML\Topic1
.\scripts\build_quick_dataset.ps1
```

Equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m src.dataset_builder --config configs/quick_test.yaml
```

When running from Codex, prefer the absolute interpreter path and request
escalated execution if the sandbox cannot launch the local base Python:

```powershell
E:\Scientific_Research\work_2026\STEC_ML\Topic1\.venv\Scripts\python.exe -m src.dataset_builder --config configs/quick_test.yaml
```
