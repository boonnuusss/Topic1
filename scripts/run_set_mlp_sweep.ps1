param(
    [string]$Device = "cpu"
)
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$IRIInput = Join-Path $ProjectDir "datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet"

Push-Location $ProjectDir

$experiments = @(
    @{Name="set_mlp_0500_nearby400_s200_e100"; Km=400; Sigma=200; Ep=100},
    @{Name="set_mlp_0500_nearby200_s200_e100"; Km=200; Sigma=200; Ep=100},
    @{Name="set_mlp_0500_nearby200_s100_e100"; Km=200; Sigma=100; Ep=100}
)

foreach ($exp in $experiments) {
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "$($exp.Name)" -ForegroundColor Cyan
    Write-Host "  radius=$($exp.Km) km, sigma=$($exp.Sigma) km, epochs=$($exp.Ep)" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan

    & $Python -m src.train_set_mlp `
        --input $IRIInput `
        --run-name $exp.Name `
        --nearby-km $exp.Km `
        --max-obs 60 `
        --spatial-sigma-km $exp.Sigma `
        --split random --test-size 0.2 --seed 42 `
        --epochs $exp.Ep --batch-size 128 `
        --learning-rate 0.001 --weight-decay 0.0001 `
        --dropout 0.1 --device $Device `
        --output-format parquet

    if ($LASTEXITCODE -ne 0) { Write-Error "$($exp.Name) failed"; exit 1 }
    Write-Host ""
}

Write-Host "============================================" -ForegroundColor Green
Write-Host "All 3 set-to-query experiments complete." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Pop-Location
