param(
    [string]$InputPath = "",
    [string]$OutputPath = "",
    [string]$SimulationId = "iri_rayvtec_20240505_v001",
    [string]$PyiriPath = "",
    [double]$F107 = 204.0,
    [int]$CcirOrUrsi = 0,
    [double]$RaySampleSpacingKm = 2.5,
    [double]$SampleAlpha = 2.0,
    [ValidateSet("row_adaptive_point_profiles", "regular_grid_interpolator")]
    [string]$GridStrategy = "row_adaptive_point_profiles",
    [double]$SlmShellHeightKm = 450.0,
    [double]$MinElevationDeg = 10.0,
    [int]$MaxRows = 0,
    [int]$MaxEpochs = 0,
    [int]$MaxIriPointsPerEpoch = 200000,
    [int]$ProgressEvery = 0,
    [switch]$DropInvalidGeometry
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if ($InputPath -eq "") {
    $InputPath = Join-Path $ProjectDir "datasets\processed\raystec_china_20240505.parquet"
}

if ($OutputPath -eq "") {
    $OutputPath = Join-Path $ProjectDir "datasets\processed\iri_simulation_rayvtec_20240505.parquet"
}

if (-not (Test-Path $Python)) {
    throw "Topic1 virtual environment not found: $Python"
}

if (-not (Test-Path $InputPath)) {
    throw "Input table not found: $InputPath"
}

$CliArgs = @(
    "-m", "src.iri_simulation_builder",
    "--input", $InputPath,
    "--output", $OutputPath,
    "--simulation-id", $SimulationId,
    "--f107", "$F107",
    "--ccir-or-ursi", "$CcirOrUrsi",
    "--ray-sample-spacing-km", "$RaySampleSpacingKm",
    "--sample-alpha", "$SampleAlpha",
    "--grid-strategy", "$GridStrategy",
    "--slm-shell-height-km", "$SlmShellHeightKm",
    "--min-elevation-deg", "$MinElevationDeg",
    "--max-iri-points-per-epoch", "$MaxIriPointsPerEpoch",
    "--progress-every", "$ProgressEvery"
)

if ($PyiriPath -ne "") {
    $CliArgs += @("--pyiri-path", $PyiriPath)
}

if ($MaxRows -gt 0) {
    $CliArgs += @("--max-rows", "$MaxRows")
}

if ($MaxEpochs -gt 0) {
    $CliArgs += @("--max-epochs", "$MaxEpochs")
}

if ($DropInvalidGeometry) {
    $CliArgs += "--drop-invalid-geometry"
}

Push-Location $ProjectDir
try {
    & $Python @CliArgs
}
finally {
    Pop-Location
}
