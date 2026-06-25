param(
    [string]$InputPath = "",
    [string]$RunName = "set_mlp_0500_nearby200_e50",
    [float]$NearbyKm = 200.0,
    [int]$MaxObs = 40,
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [float]$SpatialSigmaKm = 100.0,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if ($InputPath -eq "") {
    $InputPath = Join-Path $ProjectDir "datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet"
}

if (-not (Test-Path $Python)) { throw "Virtual env not found: $Python" }
if (-not (Test-Path $InputPath)) { throw "Input not found: $InputPath" }

Push-Location $ProjectDir

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Set-to-Query RayMLP experiment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  IPP radius        : $NearbyKm km"
Write-Host "  Max obs per query : $MaxObs"
Write-Host "  Spatial sigma     : $SpatialSigmaKm km"
Write-Host "  Epochs            : $Epochs"
Write-Host "  Batch size        : $BatchSize"
Write-Host ""
Write-Host "Expected: ~1622 training examples, each with $MaxObs nearby observations."
Write-Host "Architecture: QueryEncoder + ObsEncoder + SpatialCrossAttention + OutputMLP"
Write-Host ""

& $Python -m src.train_set_mlp `
    --input $InputPath `
    --run-name $RunName `
    --nearby-km $NearbyKm `
    --max-obs $MaxObs `
    --spatial-sigma-km $SpatialSigmaKm `
    --split random --test-size 0.2 --seed 42 `
    --epochs $Epochs --batch-size $BatchSize `
    --learning-rate 0.001 --weight-decay 0.0001 `
    --dropout 0.1 --device $Device `
    --output-format parquet

if ($LASTEXITCODE -ne 0) { Write-Error "Set-MLP training failed"; exit 1 }

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Set-to-Query experiment complete." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Comparison baselines:" -ForegroundColor Yellow
Write-Host "  SLM:                         VTEC RMSE 2.44, R2 0.961"
Write-Host "  RayMLP nearby-200·50ep:      VTEC RMSE 2.59, R2 0.956"
Write-Host "  RayMLP identity-only:        VTEC RMSE 24.08, R2 -2.80"

Pop-Location
