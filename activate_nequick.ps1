$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectDir ".venv-nequick\Scripts\python.exe"
$ActivateScript = Join-Path $ProjectDir ".venv-nequick\Scripts\Activate.ps1"

if (-not (Test-Path $VenvPython)) {
    throw "NeQuick environment not found. Run scripts\install_nequick.ps1 first."
}

. $ActivateScript
Write-Host "Activated Topic1 NeQuick-G environment:" -ForegroundColor Green
python -c "import sys, numpy, nequick; print(sys.executable); print('numpy', numpy.__version__); print('nequick', nequick.__version__)"
