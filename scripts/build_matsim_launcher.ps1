# Build RunMatsimPreloaded.class (ASCII paths only: javac argv chokes on the
# Chinese workspace path, so everything lives under the Temp junction).
$ErrorActionPreference = 'Stop'
$tmp = 'C:\Users\xuan1\AppData\Local\Temp\matsim_run'
$rel = 'C:\Users\xuan1\AppData\Local\Temp\matsim_rel'
$cp = "$rel\matsim-2026.0.jar;$rel\libs\*"
Push-Location $tmp
try {
    javac -cp $cp -d $tmp RunMatsimPreloaded.java
    Write-Host "compiled: $tmp\RunMatsimPreloaded.class"
} finally {
    Pop-Location
}
