param(
    [string]$Device = "cpu"
)
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$IRIInput = Join-Path $ProjectDir "datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet"

Push-Location $ProjectDir

$experiments = @(
    @{Name="ray_mlp_sim_0500_nearby100_e50";  Km=100;  Ep=50},
    @{Name="ray_mlp_sim_0500_nearby400_e50";  Km=400;  Ep=50},
    @{Name="ray_mlp_sim_0500_nearby200_e100"; Km=200;  Ep=100}
)

foreach ($exp in $experiments) {
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "$($exp.Name)" -ForegroundColor Cyan
    Write-Host "  IPP radius = $($exp.Km) km, epochs = $($exp.Ep)" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan

    & $Python -m src.train_ray_mlp_simulation `
        --input $IRIInput `
        --run-name $exp.Name `
        --nearby-km $exp.Km `
        --nearby-max-neighbors 30 `
        --split random --test-size 0.2 --seed 42 `
        --epochs $exp.Ep --batch-size 1024 `
        --learning-rate 0.001 --weight-decay 0.0001 `
        --dropout 0.1 --device $Device `
        --output-format parquet

    if ($LASTEXITCODE -ne 0) {
        Write-Error "$($exp.Name) failed"
        exit 1
    }
    Write-Host ""
}

Write-Host "============================================" -ForegroundColor Green
Write-Host "All 3 experiments complete." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

Pop-Location
