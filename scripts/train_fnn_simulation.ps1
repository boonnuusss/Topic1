param(
    [ValidateSet("random", "time", "station", "prn")]
    [string]$Split = "random",
    [ValidateSet("ecef6", "blh6", "bl4")]
    [string]$FeatureSet = "ecef6",
    [string]$InputPath = "",
    [string]$RunName = "",
    [string]$SourceStecColumn = "stec_iri_observable",
    [string]$SlantTargetColumn = "stec_iri_ray",
    [string]$VtecTargetColumn = "vtec_iri_ref",
    [string]$BaselineVtecColumn = "vtec_slm_from_iri_stec",
    [string]$MappingColumn = "slm_mapping_factor_iri",
    [double]$VerticalTopKm = 2000.0,
    [double]$TestSize = 0.2,
    [string]$TrainEnd = "",
    [string]$TestKeysFrom = "",
    [string]$TestKeyColumns = "timestamp,station,prn",
    [string]$TestTimestamp = "",
    [int]$Seed = 52,
    [int]$Epochs = 500,
    [int]$BatchSize = 64,
    [double]$LearningRate = 0.001,
    [double]$WeightDecay = 0.0,
    [double]$Dropout = 0.2,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [switch]$SkipVtecEvaluation,
    [string]$OutputFormat = "parquet",
    [string]$ElevationBins = "10,20,30,40,50,60,70,80,90",
    [int]$StationMinCount = 1,
    [string]$MaxRows = ""
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if ($InputPath -eq "") {
    $InputPath = Join-Path $ProjectDir "datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet"
}

if (-not (Test-Path $Python)) {
    throw "Topic1 virtual environment not found: $Python"
}

if (-not (Test-Path $InputPath)) {
    throw "Input table not found: $InputPath"
}

$CliArgs = @(
    "-m", "src.train_fnn_simulation",
    "--input", $InputPath,
    "--split", $Split,
    "--feature-set", $FeatureSet,
    "--source-stec-column", $SourceStecColumn,
    "--slant-target-column", $SlantTargetColumn,
    "--vtec-target-column", $VtecTargetColumn,
    "--baseline-vtec-column", $BaselineVtecColumn,
    "--mapping-column", $MappingColumn,
    "--vertical-top-km", "$VerticalTopKm",
    "--test-size", "$TestSize",
    "--seed", "$Seed",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--weight-decay", "$WeightDecay",
    "--dropout", "$Dropout",
    "--device", $Device,
    "--output-format", $OutputFormat,
    "--elevation-bins", $ElevationBins,
    "--station-min-count", "$StationMinCount"
)

if ($RunName -ne "") {
    $CliArgs += @("--run-name", $RunName)
}

if ($TrainEnd -ne "") {
    $CliArgs += @("--train-end", $TrainEnd)
}

if ($TestKeysFrom -ne "") {
    $CliArgs += @("--test-keys-from", $TestKeysFrom)
    $CliArgs += @("--test-key-columns", $TestKeyColumns)
}

if ($TestTimestamp -ne "") {
    $CliArgs += @("--test-timestamp", $TestTimestamp)
}

if ($SkipVtecEvaluation) {
    $CliArgs += "--skip-vtec-evaluation"
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
