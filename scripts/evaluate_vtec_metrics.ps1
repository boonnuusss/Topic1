param(
    [string]$InputPath = "",
    [string]$TableDir = "",
    [string]$RunName = "vtec_conversion",
    [string]$TargetColumn = "vtec_iri_ref",
    [string]$BaselineColumn = "vtec_slm_from_iri_stec",
    [string]$VtecPredictionColumns = "",
    [string]$StecPredictionColumns = "",
    [string]$MappingColumn = "slm_mapping_factor_iri",
    [string]$ElevationBins = "10,20,30,40,50,60,70,80,90",
    [int]$StationMinCount = 1,
    [int]$MaxRows = 0
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if ($InputPath -eq "") {
    $InputPath = Join-Path $ProjectDir "datasets\processed\iri_simulation_rayvtec_20240505.parquet"
}

if ($TableDir -eq "") {
    $TableDir = Join-Path $ProjectDir "results\tables"
}

if (-not (Test-Path $Python)) {
    throw "Topic1 virtual environment not found: $Python"
}

if (-not (Test-Path $InputPath)) {
    throw "Input table not found: $InputPath"
}

$CliArgs = @(
    "-m", "src.evaluate",
    "--input", $InputPath,
    "--table-dir", $TableDir,
    "--run-name", $RunName,
    "--target-column", $TargetColumn,
    "--baseline-column", $BaselineColumn,
    "--mapping-column", $MappingColumn,
    "--elevation-bins", $ElevationBins,
    "--station-min-count", "$StationMinCount"
)

if ($VtecPredictionColumns -ne "") {
    $CliArgs += @("--vtec-prediction-columns", $VtecPredictionColumns)
}

if ($StecPredictionColumns -ne "") {
    $CliArgs += @("--stec-prediction-columns", $StecPredictionColumns)
}

if ($MaxRows -gt 0) {
    $CliArgs += @("--max-rows", "$MaxRows")
}

Push-Location $ProjectDir
try {
    & $Python @CliArgs
}
finally {
    Pop-Location
}
