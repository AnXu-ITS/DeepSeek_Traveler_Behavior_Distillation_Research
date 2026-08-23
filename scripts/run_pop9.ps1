# Phase 9 population-scale experiment (800 personas x 4 scenarios).
# 800 personas: validated scale. 1000-persona runs hit a MATSim 2026.0 /
# Java 25 cleanup NPE (AbstractQNetsimEngine.afterMobsim, runners null) --
# documented in the Phase 9 report; bisection: 200/500/800 OK, 1000 fails.
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    & .venv\Scripts\python.exe -u scripts\run_population_experiment.py `
        --checkpoint outputs/student_v0_3_c/checkpoints/best.pt `
        --num-personas 1000 --trips-per-persona 2 `
        --scenarios baseline,rain,fare_surge,combined `
        --output data/population_experiment
    Write-Host "population experiment exit=$LASTEXITCODE"
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
