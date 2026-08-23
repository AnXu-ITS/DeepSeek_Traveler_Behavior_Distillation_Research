# v0.3 real dataset generation against the OFFICIAL DeepSeek API (direct).
# --no-cache-bypass: LiteLLM cache metadata must not be sent to the official API.
# --resume: skips already-completed states (safe to restart).
# NOTE: no non-ASCII literals in this file (PowerShell reads BOM-less .ps1 as
# ANSI, which corrupts Chinese characters).
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    & .venv\Scripts\python.exe -u scripts\generate_aggregated_teacher_dataset.py `
        --num-personas 20 --num-trips 2 `
        --axes weather_intensity,fare_multiplier `
        --workers 6 --no-cache-bypass --resume `
        --output data/student_v0_3
    Write-Host "generator exit=$LASTEXITCODE"
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
