$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BasePython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$NeQuickPython = Join-Path $ProjectRoot ".venv-nequick\Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements-nequick.txt"

if (-not (Test-Path $BasePython)) {
    throw "Topic1 base environment is missing: $BasePython"
}

if (-not (Test-Path $NeQuickPython)) {
    & $BasePython -m venv (Join-Path $ProjectRoot ".venv-nequick")
}

& $NeQuickPython -m pip install --upgrade pip setuptools wheel
& $NeQuickPython -m pip install --requirement $Requirements
& $NeQuickPython -m pip check
& $NeQuickPython -c "import numpy, nequick; print('numpy', numpy.__version__); print('nequick', nequick.__version__); print(nequick.NeQuick)"
