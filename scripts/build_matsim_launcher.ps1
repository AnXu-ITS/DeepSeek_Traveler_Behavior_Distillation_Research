# Build the preloaded-scenario launcher using caller-selected runtime paths.
# Use ASCII-only build/release paths if the local Java setup requires them.
param(
    [string]$BuildDir = $env:MATSIM_BUILD_DIR,
    [string]$ReleaseDir = $env:MATSIM_RELEASE_DIR
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($BuildDir) -or [string]::IsNullOrWhiteSpace($ReleaseDir)) {
    throw 'Provide -BuildDir and -ReleaseDir, or set MATSIM_BUILD_DIR and MATSIM_RELEASE_DIR.'
}
$rel = (Resolve-Path -LiteralPath $ReleaseDir).Path
$jar = Join-Path $rel 'matsim-2026.0.jar'
$libs = Join-Path $rel 'libs'
if (-not (Test-Path -LiteralPath $jar -PathType Leaf) -or -not (Test-Path -LiteralPath $libs -PathType Container)) {
    throw 'ReleaseDir must contain matsim-2026.0.jar and the libs directory.'
}
$null = New-Item -ItemType Directory -Path $BuildDir -Force
$tmp = (Resolve-Path -LiteralPath $BuildDir).Path
$repo = Split-Path $PSScriptRoot -Parent
$source = Join-Path $repo 'tools/java/RunMatsimPreloaded.java'
# Stage the source as well as the classpath for Windows argv-encoding safety.
$stagedSource = Join-Path $tmp 'RunMatsimPreloaded.java'
if ([System.IO.Path]::GetFullPath($source) -ne [System.IO.Path]::GetFullPath($stagedSource)) {
    Copy-Item -LiteralPath $source -Destination $stagedSource
}
$cp = "$jar$([System.IO.Path]::PathSeparator)$(Join-Path $libs '*')"
Push-Location -LiteralPath $tmp
try {
    javac -cp $cp -d $tmp RunMatsimPreloaded.java
    if ($LASTEXITCODE -ne 0) { throw "javac failed with exit code $LASTEXITCODE" }
    Write-Host "compiled: $(Join-Path $tmp 'RunMatsimPreloaded.class')"
} finally {
    Pop-Location
}
