$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-nequick\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "NeQuick environment not found. Run scripts\install_nequick.ps1 first."
}

Push-Location $ProjectRoot
try {
    & $Python -c "from datetime import datetime; from nequick import NeQuick; m=NeQuick(236.831641,-0.39362878,0.00402826613); x=m.compute_stec(datetime(2025,3,21,12),40.0,-3.0,0.0,45.0,-2.0,20000000.0); assert x > 0; print('documented STEC example:', x)"
    & $Python -m pip check
}
finally {
    Pop-Location
}
