$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$ActivateScript = Join-Path $ProjectDir ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $VenvPython)) {
    throw "Topic1 virtual environment was not found. Create it with: python -m venv .venv"
}

. $ActivateScript
Write-Host "Activated Topic1 Python environment:" -ForegroundColor Green
python -c "import sys; print(sys.executable)"

