param(
    [string]$InputPath = "",
    [string]$OutputPath = "",
    [string]$VtecColumn = "vtec_baseline",
    [string]$VtecSource = "",
    [string]$VtecTable = "",
    [string]$VtecTableColumn = "vtec_baseline",
    [string[]]$VtecMergeKeys = @("timestamp", "station", "prn"),
    [string]$VtecConstant = "",
    [switch]$AllowDiagnosticConstant,
    [switch]$DropMissingVtec,
    [double]$ShellHeightKm = 450.0
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if ($InputPath -eq "") {
    $InputPath = Join-Path $ProjectDir "datasets\processed\raystec_china_20240505.parquet"
}

if ($OutputPath -eq "") {
    $OutputPath = Join-Path $ProjectDir "datasets\processed\raystec_china_20240505_slm.parquet"
}

if (-not (Test-Path $Python)) {
    throw "Topic1 virtual environment not found: $Python"
}

if (-not (Test-Path $InputPath)) {
    throw "Input table not found: $InputPath"
}

$CliArgs = @(
    "-m", "src.slm_baseline",
    "--input", $InputPath,
    "--output", $OutputPath,
    "--vtec-column", $VtecColumn,
    "--shell-height-km", "$ShellHeightKm"
)

if ($VtecSource -ne "") {
    $CliArgs += @("--vtec-source", $VtecSource)
}

if ($VtecTable -ne "") {
    $CliArgs += @("--vtec-table", $VtecTable)
    $CliArgs += @("--vtec-table-column", $VtecTableColumn)
    $CliArgs += @("--vtec-merge-keys")
    $CliArgs += $VtecMergeKeys
}

if ($VtecConstant -ne "") {
    $CliArgs += @("--vtec-constant", $VtecConstant)
}

if ($AllowDiagnosticConstant) {
    $CliArgs += "--allow-diagnostic-constant"
}

if ($DropMissingVtec) {
    $CliArgs += "--drop-missing-vtec"
}

Push-Location $ProjectDir
try {
    & $Python @CliArgs
}
finally {
    Pop-Location
}
