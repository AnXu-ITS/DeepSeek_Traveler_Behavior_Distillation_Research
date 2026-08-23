# Run MATSim (equil sanity test or student scenario) with the preloaded-scenario
# launcher. All classpath entries are ASCII (Temp junction) to avoid the
# PowerShell->java argv encoding bug on the Chinese workspace path.
param([string]$ConfigDir, [string]$LogName = "run.log")
$tmp = 'C:\Users\xuan1\AppData\Local\Temp\matsim_run'
$rel = 'C:\Users\xuan1\AppData\Local\Temp\matsim_rel'
$cp = "$tmp;$rel\matsim-2026.0.jar;$rel\libs\*"
Push-Location $ConfigDir
try {
    java -Xmx2g -cp $cp RunMatsimPreloaded config.xml *> $LogName
    $code = $LASTEXITCODE
    Write-Host "matsim exit=$code"
    if (Test-Path .\output) {
        Write-Host 'output dir OK'
        Get-ChildItem .\output | Select-Object -First 10 Name | ForEach-Object { Write-Host "  $_" }
    } else {
        Write-Host 'NO output dir'
    }
    exit $code
} finally {
    Pop-Location
}
