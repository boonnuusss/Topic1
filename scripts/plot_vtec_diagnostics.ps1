param(
    [string]$InputPath = "",
    [string]$FigureDir = "",
    [string]$TableDir = "",
    [string]$VtecColumn = "vtec_slm",
    [string]$TimestampColumn = "timestamp",
    [string]$DateLabel = "",
    [string]$MapTime = "",
    [int]$MinMapPoints = 100,
    [int]$TopStations = 35,
    [double[]]$Extent = @(60.0, 160.0, -20.0, 80.0)
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if ($InputPath -eq "") {
    $InputPath = Join-Path $ProjectDir "datasets\processed\rayvtec_china_20240505_slm.parquet"
}

if ($FigureDir -eq "") {
    $FigureDir = Join-Path $ProjectDir "results\figures"
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
    "-m", "src.plot_vtec_diagnostics",
    "--input", $InputPath,
    "--figure-dir", $FigureDir,
    "--table-dir", $TableDir,
    "--vtec-column", $VtecColumn,
    "--timestamp-column", $TimestampColumn,
    "--min-map-points", "$MinMapPoints",
    "--top-stations", "$TopStations",
    "--extent"
)

foreach ($Value in $Extent) {
    $CliArgs += "$Value"
}

if ($DateLabel -ne "") {
    $CliArgs += @("--date-label", $DateLabel)
}

if ($MapTime -ne "") {
    $CliArgs += @("--map-time", $MapTime)
}

Push-Location $ProjectDir
try {
    & $Python @CliArgs
}
finally {
    Pop-Location
}
