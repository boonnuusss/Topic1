param(
    [string]$InputPath = "datasets\interim\iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet",
    [string]$EvaluationKeys = "results\tables\set_ratio_mlp_0500_anchor1_e100_predictions.parquet",
    [string]$RunName = "physical_mapping_baselines_ut05_validation"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Push-Location $ProjectRoot
try {
    $CliArgs = @(
        "-m", "src.physical_mapping_baselines",
        "--input", $InputPath,
        "--evaluation-keys-from", $EvaluationKeys,
        "--output", "results\tables\${RunName}_predictions.parquet",
        "--metrics-output", "results\tables\${RunName}_metrics.csv",
        "--metadata-output", "results\tables\${RunName}_metadata.json"
    )
    & $Python @CliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Physical mapping baseline evaluation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
