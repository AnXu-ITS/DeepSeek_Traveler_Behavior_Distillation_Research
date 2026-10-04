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
for ($taskPass = 0; $taskPass -lt 2; $taskPass++) {
    & $taskPython (Join-Path $PSScriptRoot 'teacher_same_task.py') pilot --condition ownership --neutral-identifiers --workers 16
    if ($LASTEXITCODE -ne 0) { throw 'Neutral generic prompt pilot failed' }
    & $taskPython (Join-Path $PSScriptRoot 'analyze_prompt_sensitivity.py')
    if ($LASTEXITCODE -eq 0) { break }
    if ($LASTEXITCODE -ne 2) { throw 'Prompt input audit failed' }
}
if ($LASTEXITCODE -ne 0) { throw 'Preselected prompt pilot remains incomplete' }
