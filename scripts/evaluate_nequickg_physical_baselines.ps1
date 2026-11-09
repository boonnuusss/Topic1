param(
    [string]$InputPath = "datasets\interim\nequickg_simulation_20240505_0000_2400_5min.parquet",
    [string]$EvaluationKeys = "results\tables\set_ratio_mlp_0500_anchor1_e100_predictions.parquet",
    [string]$RunName = "nequickg_physical_mapping_baselines_ut05_validation",
    [string]$CppTrainWindowStart = "",
    [string]$CppTrainWindowEnd = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-nequick\Scripts\python.exe"
$CliArgs = @(
    "-m", "src.physical_mapping_baselines",
    "--input", $InputPath,
    "--reference-model", "nequickg",
    "--stec-column", "stec_nequickg_observable",
    "--truth-column", "vtec_nequickg_ref",
    "--evaluation-keys-from", $EvaluationKeys,
    "--output", "results\tables\${RunName}_predictions.parquet",
    "--metrics-output", "results\tables\${RunName}_metrics.csv",
    "--metadata-output", "results\tables\${RunName}_metadata.json",
    "--cpp-pairs-output", "results\tables\${RunName}_cpp_pairs.parquet"
)

if ($CppTrainWindowStart -or $CppTrainWindowEnd) {
    if (-not $CppTrainWindowStart -or -not $CppTrainWindowEnd) {
        throw "Both CPP train-window boundaries are required."
    }
    $CliArgs += @(
        "--cpp-train-window-start", $CppTrainWindowStart,
        "--cpp-train-window-end", $CppTrainWindowEnd
    )
}

Push-Location $ProjectRoot
try {
    & $Python @CliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "NeQuick-G physical baseline evaluation failed: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
