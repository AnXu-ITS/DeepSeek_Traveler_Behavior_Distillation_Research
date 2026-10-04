# Run MATSim with the preloaded-scenario launcher and explicit runtime paths.
# Build the launcher with build_matsim_launcher.ps1 using the same BuildDir.
param(
    [Parameter(Mandatory = $true)][string]$ConfigDir,
    [string]$LogName = 'run.log',
    [string]$BuildDir = $env:MATSIM_BUILD_DIR,
    [string]$ReleaseDir = $env:MATSIM_RELEASE_DIR
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($BuildDir) -or [string]::IsNullOrWhiteSpace($ReleaseDir)) {
    throw 'Provide -BuildDir and -ReleaseDir, or set MATSIM_BUILD_DIR and MATSIM_RELEASE_DIR.'
}
$tmp = (Resolve-Path -LiteralPath $BuildDir).Path
$rel = (Resolve-Path -LiteralPath $ReleaseDir).Path
$jar = Join-Path $rel 'matsim-2026.0.jar'
$libs = Join-Path $rel 'libs'
if (-not (Test-Path -LiteralPath (Join-Path $tmp 'RunMatsimPreloaded.class') -PathType Leaf)) {
    throw 'BuildDir has no RunMatsimPreloaded.class. Run build_matsim_launcher.ps1 first.'
}
if (-not (Test-Path -LiteralPath $jar -PathType Leaf) -or -not (Test-Path -LiteralPath $libs -PathType Container)) {
    throw 'ReleaseDir must contain matsim-2026.0.jar and the libs directory.'
}
$cp = @($tmp, $jar, (Join-Path $libs '*')) -join [System.IO.Path]::PathSeparator
Push-Location -LiteralPath $ConfigDir
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
