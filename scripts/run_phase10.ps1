# Phase 10 feedback loop (800 personas x 3 scenarios x up to 6 iterations).
# grid 10x10, per-link capacity 5 veh/h: calibrated so congestion emerges at
# development population scale (documented in the Phase 10 report).
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    & .venv\Scripts\python.exe -u scripts\run_phase10_loop.py `
        --checkpoint outputs/student_v0_3_c/checkpoints/best.pt `
        --num-personas 800 --trips-per-persona 1 `
        --scenarios baseline,rain,fare_surge `
        --max-iterations 6 --eps 0.02 `
        --grid-n 10 --link-capacity 5 `
        --output data/phase10_loop
    Write-Host "phase10 exit=$LASTEXITCODE"
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
