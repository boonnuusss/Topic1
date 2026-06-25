param(
    [string]$InputPath = "",
    [string]$OutputPath = "",
    [string]$StecColumn = "stec_obs",
    [string]$VtecColumn = "vtec_slm",
    [string]$MappingColumn = "slm_mapping_factor",
    [string]$ElevationColumn = "elevation",
    [double]$ShellHeightKm = 450.0,
    [switch]$UseExistingMapping,
    [switch]$DropMissing,
    [int]$MaxRows = 0
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if ($InputPath -eq "") {
    $InputPath = Join-Path $ProjectDir "datasets\processed\raystec_china_20240505.parquet"
}

if ($OutputPath -eq "") {
    $OutputPath = Join-Path $ProjectDir "datasets\processed\rayvtec_china_20240505_slm.parquet"
}

if (-not (Test-Path $Python)) {
    throw "Topic1 virtual environment not found: $Python"
}

if (-not (Test-Path $InputPath)) {
    throw "Input table not found: $InputPath"
}

$CliArgs = @(
    "-m", "src.slm_vtec",
    "--input", $InputPath,
    "--output", $OutputPath,
    "--stec-column", $StecColumn,
    "--vtec-column", $VtecColumn,
    "--mapping-column", $MappingColumn,
    "--elevation-column", $ElevationColumn,
    "--shell-height-km", "$ShellHeightKm"
)

if ($UseExistingMapping) {
    $CliArgs += "--use-existing-mapping"
}

if ($DropMissing) {
    $CliArgs += "--drop-missing"
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
