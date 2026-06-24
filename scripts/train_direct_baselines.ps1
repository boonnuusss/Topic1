param(
    [ValidateSet("ridge", "random_forest")]
    [string]$Model = "ridge",
    [ValidateSet("random", "time", "station", "prn")]
    [string]$Split = "random",
    [string]$InputPath = "",
    [string]$RunName = "",
    [double]$TestSize = 0.2,
    [string]$TrainEnd = "",
    [int]$Seed = 42,
    [double]$RidgeAlpha = 1.0,
    [int]$RfNEstimators = 300,
    [int]$RfMinSamplesLeaf = 2,
    [string]$RfMaxDepth = "",
    [int]$RfNJobs = -1,
    [string]$OutputFormat = "parquet",
    [string]$MaxRows = ""
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if ($InputPath -eq "") {
    $InputPath = Join-Path $ProjectDir "datasets\processed\raystec_china_20240505.parquet"
}

if (-not (Test-Path $Python)) {
    throw "Topic1 virtual environment not found: $Python"
}

if (-not (Test-Path $InputPath)) {
    throw "Input table not found: $InputPath"
}

$CliArgs = @(
    "-m", "src.train_tree",
    "--input", $InputPath,
    "--model", $Model,
    "--split", $Split,
    "--test-size", "$TestSize",
    "--seed", "$Seed",
    "--ridge-alpha", "$RidgeAlpha",
    "--rf-n-estimators", "$RfNEstimators",
    "--rf-min-samples-leaf", "$RfMinSamplesLeaf",
    "--rf-n-jobs", "$RfNJobs",
    "--output-format", $OutputFormat
)

if ($RunName -ne "") {
    $CliArgs += @("--run-name", $RunName)
}

if ($TrainEnd -ne "") {
    $CliArgs += @("--train-end", $TrainEnd)
}

if ($RfMaxDepth -ne "") {
    $CliArgs += @("--rf-max-depth", $RfMaxDepth)
}

if ($MaxRows -ne "") {
    $CliArgs += @("--max-rows", $MaxRows)
}

Push-Location $ProjectDir
try {
    & $Python @CliArgs
}
finally {
    Pop-Location
}
