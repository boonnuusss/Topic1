param(
    [string]$ArchivePath = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TemporaryArchive = $null

$ExpectedSha256 = `
    "623509581E8758DFE07BF4AAF921E3F850BD552BD718B6BDD1BDBA60F63320EB"

if (-not $ArchivePath) {
    $FullArchive = Join-Path $ProjectRoot `
        "artifacts\Topic1_transfer_20261109.zip"
    if (Test-Path -LiteralPath $FullArchive) {
        $ArchivePath = $FullArchive
    }
    else {
        $PartsDirectory = Join-Path $ProjectRoot "artifacts\parts"
        $Parts = @(Get-ChildItem -LiteralPath $PartsDirectory `
            -Filter "Topic1_transfer_20261109.zip.part???" -File |
            Sort-Object Name)
        if ($Parts.Count -ne 5) {
            throw "Expected 5 migration chunks in $PartsDirectory, found " +
                "$($Parts.Count). Fetch and switch to " +
                "'artifact-transfer-20261109' first."
        }

        $TemporaryArchive = Join-Path $env:TEMP `
            "Topic1_transfer_20261109_$PID.zip"
        $Output = [System.IO.File]::Create($TemporaryArchive)
        try {
            foreach ($Part in $Parts) {
                $Input = [System.IO.File]::OpenRead($Part.FullName)
                try {
                    $Input.CopyTo($Output)
                }
                finally {
                    $Input.Dispose()
                }
            }
        }
        finally {
            $Output.Dispose()
        }
        $ArchivePath = $TemporaryArchive
    }
}

if (-not (Test-Path -LiteralPath $ArchivePath)) {
    throw "Migration archive not found: $ArchivePath"
}

$Archive = Get-Item -LiteralPath $ArchivePath
try {
    $ActualSha256 = (Get-FileHash -LiteralPath $Archive.FullName `
        -Algorithm SHA256).Hash
    if ($ActualSha256 -ne $ExpectedSha256) {
        throw "Migration archive SHA-256 mismatch. Expected $ExpectedSha256, " +
            "got $ActualSha256."
    }

    & tar.exe -xf $Archive.FullName -C $ProjectRoot datasets results
    if ($LASTEXITCODE -ne 0) {
        throw "Archive extraction failed with exit code $LASTEXITCODE."
    }
}
finally {
    if ($TemporaryArchive -and (Test-Path -LiteralPath $TemporaryArchive)) {
        Remove-Item -LiteralPath $TemporaryArchive -Force
    }
}

$RequiredFiles = @(
    "datasets\interim\raystec_china_20240505_0000_2400_5min.parquet",
    "datasets\interim\iri_simulation_20240505_0000_2400_5min_spacing2p5_regular.parquet",
    "datasets\interim\nequickg_simulation_20240505_0000_2400_5min.parquet",
    "results\tables\set_ratio_mlp_0500_anchor1_e100_predictions.parquet"
)

foreach ($RelativePath in $RequiredFiles) {
    $Path = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required restored file is missing: $Path"
    }
}

$TableCount = (Get-ChildItem -LiteralPath `
    (Join-Path $ProjectRoot "results\tables") -File -Recurse).Count
$CheckpointCount = (Get-ChildItem -LiteralPath `
    (Join-Path $ProjectRoot "results\checkpoints") -File -Recurse).Count

Write-Host "Topic1 research artifacts restored successfully." `
    -ForegroundColor Green
Write-Host "Archive SHA-256: $ActualSha256"
Write-Host "Result table files: $TableCount"
Write-Host "Checkpoint files: $CheckpointCount"
