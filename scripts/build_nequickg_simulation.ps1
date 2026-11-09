param(
    [string]$InputPath = "datasets\interim\iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet",
    [string]$OutputPath = "datasets\interim\nequickg_simulation_20240505_0000_2400_5min.parquet",
    [int]$Workers = 0,
    [string]$StartTime = "",
    [string]$EndTime = "",
    [int]$MaxRows = 0,
    [int]$MaxEpochs = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-nequick\Scripts\python.exe"

$CliArgs = @(
    "-m", "src.nequickg_simulation_builder",
    "--input", $InputPath,
    "--output", $OutputPath,
    "--workers", "$Workers"
)
if ($StartTime) { $CliArgs += @("--start-time", $StartTime) }
if ($EndTime) { $CliArgs += @("--end-time", $EndTime) }
if ($MaxRows -gt 0) { $CliArgs += @("--max-rows", "$MaxRows") }
if ($MaxEpochs -gt 0) { $CliArgs += @("--max-epochs", "$MaxEpochs") }

Push-Location $ProjectRoot
try {
    & $Python @CliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "NeQuick-G dataset build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
