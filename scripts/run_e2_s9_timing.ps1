# E2 S9 timing runner: waits for the merged 100k pool, then runs
# 6 time-s9 passes (threads=1 x3, threads=default x3), saves G4 decision
# files from two threads=1 passes, and writes the G2 CLI check result.
# make_e2_report.py is run separately after the DeepSeek job settles.

$ErrorActionPreference = 'Stop'
$pool = 'outputs\e2_efficiency\states_100000.jsonl'
$out = 'outputs\e2_efficiency'

Write-Output "[runner] waiting for $pool ..."
while (-not (Test-Path $pool)) {
    Start-Sleep -Seconds 30
}
# guard: file must be stable for 60s (merge completed)
$size1 = (Get-Item $pool).Length
Start-Sleep -Seconds 60
$size2 = (Get-Item $pool).Length
while ($size1 -ne $size2) {
    $size1 = $size2
    Start-Sleep -Seconds 60
    $size2 = (Get-Item $pool).Length
}
Write-Output "[runner] pool ready ($size2 bytes). Starting timing runs."

$py = '.venv\Scripts\python.exe'
$script = 'scripts\bench_e2_efficiency.py'

& $py $script time-s9 --pool $pool --threads 1 --repeat 0 --decisions-out "$out\g4_decisions_r0.json"
if ($LASTEXITCODE -ne 0) { throw "time-s9 t1 r0 failed" }
& $py $script time-s9 --pool $pool --threads 1 --repeat 1 --decisions-out "$out\g4_decisions_r1.json"
if ($LASTEXITCODE -ne 0) { throw "time-s9 t1 r1 failed" }
& $py $script time-s9 --pool $pool --threads 1 --repeat 2
if ($LASTEXITCODE -ne 0) { throw "time-s9 t1 r2 failed" }
& $py $script time-s9 --pool $pool --threads default --repeat 0
if ($LASTEXITCODE -ne 0) { throw "time-s9 tdef r0 failed" }
& $py $script time-s9 --pool $pool --threads default --repeat 1
if ($LASTEXITCODE -ne 0) { throw "time-s9 tdef r1 failed" }
& $py $script time-s9 --pool $pool --threads default --repeat 2
if ($LASTEXITCODE -ne 0) { throw "time-s9 tdef r2 failed" }

& $py $script g2-check --decisions "$out\decisions_100000.json" --n 10000 |
    Out-File -FilePath "$out\g2_check.json" -Encoding utf8

Write-Output "[runner] all timing runs + G2 done."
