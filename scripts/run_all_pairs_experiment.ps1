param(
    [string]$InputPath = "",
    [string]$RunName = "ray_mlp_sim_0500_random_e20_allpairs",
    [int]$Epochs = 20,
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
Write-Host "RayMLP all-to-all pairs experiment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This builds N^2 training pairs from ~1622 training rays."
Write-Host "Expected: ~2.63 million training pairs."
Write-Host "Memory: ~1.5-2 GB during training."
Write-Host ""

& $Python -m src.train_ray_mlp_simulation `
    --input $InputPath `
    --run-name $RunName `
    --all-pairs `
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
    Write-Error "RayMLP all-pairs training failed"
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Experiment complete." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Comparison baselines:" -ForegroundColor Yellow
Write-Host "  SLM:                     VTEC RMSE 2.44 TECU, R2 0.961"
Write-Host "  RayMLP identity-only:    VTEC RMSE 24.08 TECU, R2 -2.80"
Write-Host "  FNN bl4:                 VTEC RMSE 7.99 TECU,  R2 0.64"
Write-Host ""
Write-Host "Outputs:" -ForegroundColor Green
Write-Host "  results/tables/${RunName}_predictions.parquet"
Write-Host "  results/tables/${RunName}_metrics.json"
Write-Host "  results/checkpoints/${RunName}.pth"

Pop-Location
