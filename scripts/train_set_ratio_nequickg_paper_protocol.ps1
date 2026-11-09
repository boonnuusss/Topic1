param(
    [string]$InputPath =
        "datasets\interim\nequickg_simulation_20240505_0000_2400_5min.parquet",
    [string]$ValidationKeysFrom =
        "results\tables\set_ratio_mlp_0500_anchor1_e100_predictions.parquet",
    [string]$RunName =
        "set_ratio_mlp_nequickg_tw0400_0800_static_log_p192k_e25",
    [int]$Epochs = 25,
    [int]$EpochMetricsEvery = 5
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

$CliArgs = @(
    "-m", "src.train_set_ratio_mlp",
    "--input", $InputPath,
    "--validation-keys-from", $ValidationKeysFrom,
    "--validation-key-columns", "timestamp,station,prn",
    "--run-name", $RunName,
    "--source-stec-column", "stec_nequickg_observable",
    "--slant-target-column", "stec_nequickg_ray",
    "--vtec-target-column", "vtec_nequickg_ref",
    "--baseline-vtec-column", "vtec_slm_from_nequickg_stec",
    "--mapping-column", "slm_mapping_factor_nequickg",
    "--seed", "42",
    "--nearby-km", "200",
    "--context-km", "0",
    "--nearby-max-neighbors", "30",
    "--max-obs", "1",
    "--target-mode", "slm_correction",
    "--target-transform", "log",
    "--correction-regularization", "0.001",
    "--elevation-shrinkage",
    "--shrinkage-start-deg", "30",
    "--shrinkage-end-deg", "90",
    "--shrinkage-power", "4",
    "--shrinkage-min-strength", "0.1",
    "--min-train-elevation-deg", "30",
    "--min-eval-elevation-deg", "30",
    "--same-epoch-pairs",
    "--train-window-start", "2024-05-05T04:00:00",
    "--train-window-end", "2024-05-05T08:00:00",
    "--max-pair-examples", "192000",
    "--epochs", $Epochs.ToString(),
    "--epoch-metrics",
    "--epoch-metrics-every", $EpochMetricsEvery.ToString(),
    "--batch-size", "1024",
    "--device", "cpu",
    "--output-format", "parquet"
)

Push-Location $ProjectRoot
try {
    & $Python @CliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "NeQuick-G training failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
