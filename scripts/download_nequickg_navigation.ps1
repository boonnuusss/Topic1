param(
    [string]$OutputPath = "datasets\raw\navigation\BRDC00IGS_R_20241260000_01D_MN.rnx.gz"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-nequick\Scripts\python.exe"
$Url = "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2024/126/BRDC00IGS_R_20241260000_01D_MN.rnx.gz"
$Destination = Join-Path $ProjectRoot $OutputPath

New-Item -ItemType Directory -Force (Split-Path -Parent $Destination) | Out-Null
& curl.exe -L --fail --retry 5 --retry-delay 2 --connect-timeout 30 `
    --max-time 300 -o $Destination $Url
if ($LASTEXITCODE -ne 0) {
    throw "Navigation download failed with exit code $LASTEXITCODE"
}

Push-Location $ProjectRoot
try {
    & $Python -c "from src.nequickg_simulation_builder import parse_galileo_iono_coefficients,sha256_file; p=r'$OutputPath'; print(parse_galileo_iono_coefficients(p)); print('sha256',sha256_file(p))"
}
finally {
    Pop-Location
}
