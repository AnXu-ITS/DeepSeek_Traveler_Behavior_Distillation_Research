param([int[]]$WaitFor = @())
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $taskRepo
foreach ($taskPid in $WaitFor) {
    if (Get-Process -Id $taskPid -ErrorAction SilentlyContinue) { Wait-Process -Id $taskPid -ErrorAction SilentlyContinue }
}
$taskInstructionPath = Join-Path (Split-Path $taskRepo -Parent) '蒸馏出行意图paper\AIT_Codex_Revision_Instructions_v2.md'
$taskCredentialMatch = [regex]::Match([System.IO.File]::ReadAllText($taskInstructionPath), 'sk-[A-Za-z0-9]+')
if (-not $taskCredentialMatch.Success) { throw 'Primary credential unavailable' }
$env:DEEPSEEK_API_KEY = $taskCredentialMatch.Value
$taskPython = Join-Path $taskRepo '.venv\Scripts\python.exe'
try {
    for ($taskPass = 0; $taskPass -lt 2; $taskPass++) {
        & $taskPython (Join-Path $PSScriptRoot 'teacher_same_task.py') pilot --condition ownership --neutral-identifiers --workers 16
        if ($LASTEXITCODE -ne 0) { throw 'Neutral generic prompt pilot failed' }
        & $taskPython (Join-Path $PSScriptRoot 'analyze_prompt_sensitivity.py')
        if ($LASTEXITCODE -eq 0) { break }
        if ($LASTEXITCODE -ne 2) { throw 'Prompt input audit failed' }
    }
    if ($LASTEXITCODE -ne 0) { throw 'Preselected prompt pilot remains incomplete' }
} finally {
    Remove-Item Env:\DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
}
