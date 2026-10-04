param([int[]]$WaitFor = @())
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $taskRepo
foreach ($taskPid in $WaitFor) {
    if (Get-Process -Id $taskPid -ErrorAction SilentlyContinue) { Wait-Process -Id $taskPid -ErrorAction SilentlyContinue }
}
# Use a caller-supplied credential; never read credentials from documents.
if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
    throw 'Set DEEPSEEK_API_KEY in the environment before running this script.'
}
$taskPython = Join-Path $taskRepo '.venv\Scripts\python.exe'
$taskRunner = Join-Path $PSScriptRoot 'teacher_same_task.py'
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
