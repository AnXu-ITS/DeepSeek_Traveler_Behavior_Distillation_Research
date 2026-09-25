$ErrorActionPreference = 'Stop'
$figureRoot = Split-Path $PSScriptRoot -Parent
$pptApp = $null
$pptDeck = $null
try {
  $pptApp = New-Object -ComObject PowerPoint.Application
  $pptDeck = $pptApp.Presentations.Open((Join-Path $figureRoot 'figures/narrative/figure2_method_template.pptx'),0,0,0)
  $normalizedPptx = Join-Path ([System.IO.Path]::GetTempPath()) ('figure2_' + [guid]::NewGuid().ToString() + '.pptx')
  $pptDeck.SaveAs($normalizedPptx,24)
  $pptDeck.SaveAs((Join-Path $figureRoot 'figures/narrative/figure2_method.pdf'),32)
} finally {
  if ($pptDeck) { $pptDeck.Close() }
  if ($pptApp) { $pptApp.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pptApp) }
}

if ($normalizedPptx -and (Test-Path -LiteralPath $normalizedPptx)) {
  Copy-Item -LiteralPath $normalizedPptx -Destination (Join-Path $figureRoot 'figures/narrative/figure2_method_template.pptx') -Force
  Remove-Item -LiteralPath $normalizedPptx
}
