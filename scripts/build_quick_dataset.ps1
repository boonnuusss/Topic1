$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectDir "configs\quick_test.yaml"

if (-not (Test-Path $Python)) {
    throw "Topic1 virtual environment not found: $Python"
}

if (-not (Test-Path $Config)) {
    throw "Config file not found: $Config"
}

Push-Location $ProjectDir
try {
    & $Python -m src.dataset_builder --config $Config
}
finally {
    Pop-Location
}

