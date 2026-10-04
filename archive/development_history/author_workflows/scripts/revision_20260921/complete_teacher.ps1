param([int[]]$WaitFor = @())
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $taskRepo
foreach ($taskPid in $WaitFor) {
    if (Get-Process -Id $taskPid -ErrorAction SilentlyContinue) { Wait-Process -Id $taskPid -ErrorAction SilentlyContinue }
}
$taskInstructionPath = Join-Path (Split-Path $taskRepo -Parent) '蒸馏出行意图paper\AIT_Codex_Revision_Instructions_v2.md'
$taskInstructionText = [System.IO.File]::ReadAllText($taskInstructionPath)
$taskCredentialMatch = [regex]::Match($taskInstructionText, 'sk-[A-Za-z0-9]+')
if (-not $taskCredentialMatch.Success) { throw 'Primary credential unavailable' }
$env:DEEPSEEK_API_KEY = $taskCredentialMatch.Value
$taskPython = Join-Path $taskRepo '.venv\Scripts\python.exe'
$taskRunner = Join-Path $PSScriptRoot 'teacher_same_task.py'
try {
    & $taskPython $taskRunner run --workers 32
    if ($LASTEXITCODE -ne 0) { throw 'Generic Teacher resume failed' }
    & $taskPython $taskRunner pilot --condition ownership --workers 16
    if ($LASTEXITCODE -ne 0) { throw 'Ownership prompt pilot failed' }
    & $taskPython $taskRunner pilot --condition ownership --prompt accessibility --workers 32
    if ($LASTEXITCODE -ne 0) { throw 'Accessibility prompt pilot failed' }
    & $taskPython $taskRunner run --condition ownership --prompt accessibility --workers 32
    if ($LASTEXITCODE -ne 0) { throw 'Accessibility prompt run failed' }
    & $taskPython (Join-Path $PSScriptRoot 'analyze_teacher.py')
    if ($LASTEXITCODE -ne 0) { throw 'Generic Teacher analysis incomplete' }
    & $taskPython (Join-Path $PSScriptRoot 'analyze_teacher.py') --condition ownership --prompt accessibility
    if ($LASTEXITCODE -ne 0) { throw 'Main Teacher analysis incomplete' }
} finally {
    Remove-Item Env:\DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
}
