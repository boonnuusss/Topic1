param(
    [switch]$SkipRayMLP,
    [switch]$SkipFNN
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$IRIInput = Join-Path $ProjectDir "datasets\interim\iri_simulation_20240505_0500_spacing2p5_regular.parquet"

if (-not (Test-Path $Python)) {
    throw "Topic1 virtual environment not found: $Python"
}
if (-not (Test-Path $IRIInput)) {
    throw "IRI simulation input not found: $IRIInput"
}

Push-Location $ProjectDir

# ──────────────────────────────────────────────────────────────────
# Fix: post-inference clamping replaces in-model softplus.
# Softplus in the model broke training because StandardScaler
# transforms targets to zero-mean, which softplus cannot output.
# The correct approach: train with unconstrained Linear output,
# then clamp predictions to >= 0 after inverse-transform.
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────
# Experiment A: RayMLP ratiofix + post-inference clamp
# ──────────────────────────────────────────────────
if (-not $SkipRayMLP) {
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "Experiment A: RayMLP ratiofix + post-inference clamp" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""

    $RayMLPRunName = "ray_mlp_sim_0500_random_e100_clamp"

    & $Python -m src.train_ray_mlp_simulation `
        --input $IRIInput `
        --run-name $RayMLPRunName `
        --split random `
        --test-size 0.2 `
        --seed 42 `
        --epochs 100 `
        --batch-size 128 `
        --learning-rate 0.001 `
        --weight-decay 0.0001 `
        --dropout 0.1 `
        --device cpu `
        --output-format parquet

    if ($LASTEXITCODE -ne 0) {
        Write-Error "RayMLP training failed"
        exit 1
    }
    Write-Host ""
    Write-Host "RayMLP outputs:" -ForegroundColor Green
    Write-Host "  results/tables/${RayMLPRunName}_predictions.parquet"
    Write-Host "  results/tables/${RayMLPRunName}_metrics.json"
    Write-Host "  results/checkpoints/${RayMLPRunName}.pth"
    Write-Host ""
}

# ──────────────────────────────────────────────────
# Experiment B: FNN bl4 + post-inference clamp
# ──────────────────────────────────────────────────
if (-not $SkipFNN) {
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "Experiment B: FNN bl4 + post-inference clamp" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""

    $FNNRunName = "fnn_sim_0500_bl4_e500_clamp"

    & $Python -m src.train_fnn_simulation `
        --input $IRIInput `
        --run-name $FNNRunName `
        --feature-set bl4 `
        --split random `
        --test-size 0.2 `
        --seed 52 `
        --epochs 500 `
        --batch-size 64 `
        --learning-rate 0.001 `
        --dropout 0.2 `
        --device cpu

    if ($LASTEXITCODE -ne 0) {
        Write-Error "FNN training failed"
        exit 1
    }
    Write-Host ""
    Write-Host "FNN outputs:" -ForegroundColor Green
    Write-Host "  results/tables/${FNNRunName}_predictions.parquet"
    Write-Host "  results/tables/${FNNRunName}_metrics.json"
    Write-Host "  results/checkpoints/${FNNRunName}.pth"
    Write-Host ""
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Both experiments complete." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Previous (no-softplus, no-clamp) baselines from EXPERIMENT_LOG.md:" -ForegroundColor Yellow
Write-Host "  Experiment 020 (RayMLP ratiofix): VTEC RMSE 24.08 TECU, SLM 2.44 TECU"
Write-Host "  Experiment 022 (FNN bl4):         VTEC RMSE 9.15 TECU,  SLM 2.44 TECU"
Write-Host ""
Write-Host "Broken softplus runs (archived as negative result):" -ForegroundColor DarkYellow
Write-Host "  ray_mlp_sim_0500_random_e100_softplus: VTEC RMSE 29.64 TECU"
Write-Host "  fnn_sim_0500_bl4_e500_softplus:        VTEC RMSE 24.57 TECU"

Pop-Location
