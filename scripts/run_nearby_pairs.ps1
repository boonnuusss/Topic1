param(
    [string]$InputPath = "",
    [string]$RunName = "ray_mlp_sim_0500_nearby200_e50",
    [float]$NearbyKm = 200.0,
    [int]$MaxNeighbors = 30,
    [int]$Epochs = 50,
    [int]$BatchSize = 1024,
    [string]$Device = "cpu"
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
    throw "IRI simulation input not found: $InputPath"
}

Push-Location $ProjectDir

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "RayMLP IPP-nearby pairs experiment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IPP distance threshold : ${NearbyKm} km"
Write-Host "Max neighbors per ray  : ${MaxNeighbors}"
Write-Host "Epochs                 : ${Epochs}"
Write-Host "Batch size             : ${BatchSize}"
Write-Host ""
Write-Host "With 1622 training rays and ~200 km radius, expect ~20k-40k pairs."
Write-Host "This is ~100x fewer than all-to-all (2.6M pairs)."
Write-Host ""

& $Python -m src.train_ray_mlp_simulation `
    --input $InputPath `
    --run-name $RunName `
    --nearby-km $NearbyKm `
    --nearby-max-neighbors $MaxNeighbors `
    --split random `
    --test-size 0.2 `
    --seed 42 `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --learning-rate 0.001 `
    --weight-decay 0.0001 `
    --dropout 0.1 `
    --device $Device `
    --output-format parquet

if ($LASTEXITCODE -ne 0) {
    Write-Error "RayMLP nearby-pairs training failed"
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Experiment complete." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Comparison baselines:" -ForegroundColor Yellow
Write-Host "  SLM:                      VTEC RMSE 2.44, R2 0.961"
Write-Host "  RayMLP identity-only:     VTEC RMSE 24.08, R2 -2.80"
Write-Host "  RayMLP all-to-all (20ep): VTEC RMSE 3.24,  R2 0.931"
Write-Host ""

Pop-Location
