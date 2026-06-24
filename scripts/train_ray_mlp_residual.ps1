param(
    [ValidateSet("random", "time", "station", "prn")]
    [string]$Split = "random",
    [string]$InputPath = "",
    [string]$RunName = "",
    [double]$TestSize = 0.2,
    [string]$TrainEnd = "",
    [int]$Seed = 42,
    [int]$Epochs = 30,
    [int]$BatchSize = 1024,
    [double]$LearningRate = 0.001,
    [double]$WeightDecay = 0.0001,
    [double]$Dropout = 0.1,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [switch]$IncludeBaselineFeature,
    [string]$OutputFormat = "parquet",
    [string]$MaxRows = ""
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if ($InputPath -eq "") {
    $InputPath = Join-Path $ProjectDir "datasets\processed\raystec_china_20240505_slm.parquet"
}

if (-not (Test-Path $Python)) {
    throw "Topic1 virtual environment not found: $Python"
}

if (-not (Test-Path $InputPath)) {
    throw "Input table not found: $InputPath"
}

$CliArgs = @(
    "-m", "src.train_mlp",
    "--input", $InputPath,
    "--split", $Split,
    "--test-size", "$TestSize",
    "--seed", "$Seed",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--weight-decay", "$WeightDecay",
    "--dropout", "$Dropout",
    "--device", $Device,
    "--output-format", $OutputFormat
)

if ($RunName -ne "") {
    $CliArgs += @("--run-name", $RunName)
}

if ($TrainEnd -ne "") {
    $CliArgs += @("--train-end", $TrainEnd)
}

if ($IncludeBaselineFeature) {
    $CliArgs += "--include-baseline-feature"
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
