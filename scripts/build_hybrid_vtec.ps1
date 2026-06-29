param(
    [string]$InputPath = "results\tables\ray_mlp_sim_0500_nearby200_e100_source_ratio_predictions.parquet",
    [string]$OutputPath = "results\tables\ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid20.parquet",
    [string]$RunName = "ray_mlp_sim_0500_nearby200_e100_source_ratio_hybrid20",
    [double]$ElevationThresholdDeg = 20.0,
    [string]$MlVtecColumn = "vtec_ray_mlp",
    [string]$SlmVtecColumn = "vtec_slm_from_iri_stec",
    [string]$OutputColumn = "vtec_hybrid_lowele_ml",
    [string]$MlConstellations = "",
    [switch]$MlMustLowerSlm
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) { throw "Virtual env not found: $Python" }

Push-Location $ProjectDir

$HybridArgs = @(
    "-m", "src.hybrid_vtec",
    "--input", $InputPath,
    "--output", $OutputPath,
    "--run-name", $RunName,
    "--elevation-threshold-deg", $ElevationThresholdDeg,
    "--ml-vtec-column", $MlVtecColumn,
    "--slm-vtec-column", $SlmVtecColumn,
    "--output-column", $OutputColumn,
    "--ml-constellations", $MlConstellations
)
if ($MlMustLowerSlm) { $HybridArgs += "--ml-must-lower-slm" }

& $Python @HybridArgs

if ($LASTEXITCODE -ne 0) { Write-Error "Hybrid VTEC build failed"; exit 1 }

Write-Host "Hybrid VTEC complete." -ForegroundColor Green
Write-Host "  Output   : $OutputPath"
Write-Host "  Run name : $RunName"
Write-Host "  Rule     : use $MlVtecColumn below $ElevationThresholdDeg deg, else $SlmVtecColumn"
if ($MlConstellations) { Write-Host "  ML constellations: $MlConstellations" }
if ($MlMustLowerSlm) { Write-Host "  ML must lower SLM: true" }

Pop-Location