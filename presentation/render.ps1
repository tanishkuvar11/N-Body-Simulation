# Render every slide of the deck to PNG through PowerPoint (real fonts), optionally re-save
# the deck with embedded TrueType fonts.   powershell -File render.ps1 [-Embed]
param([switch]$Embed)
$ErrorActionPreference = "Stop"
$deck = Join-Path $PSScriptRoot "PCAP_NBody_Adaptive_Scheduling.pptx"
$out = Join-Path $PSScriptRoot "render"
New-Item -ItemType Directory -Force $out | Out-Null
Get-ChildItem $out -Filter *.png | Remove-Item -Force
$app = New-Object -ComObject PowerPoint.Application
try {
    # Open(FileName, ReadOnly, Untitled, WithWindow)
    $p = $app.Presentations.Open($deck, 0, 0, 0)
    for ($i = 1; $i -le $p.Slides.Count; $i++) {
        $p.Slides.Item($i).Export((Join-Path $out ("slide-{0:D2}.png" -f $i)), "PNG", 1920, 1080)
    }
    "rendered $($p.Slides.Count) slides"
    $missing = @()
    foreach ($f in $p.Fonts) { if (-not ($f.Embeddable)) { $missing += $f.Name } }
    "fonts in deck: " + (($p.Fonts | ForEach-Object { $_.Name }) -join ", ")
    if ($Embed) {
        $tmp = Join-Path $PSScriptRoot "embedded_tmp.pptx"
        $p.SaveAs($tmp, 24, -1)          # ppSaveAsOpenXMLPresentation, EmbedTrueTypeFonts = msoTrue
        $p.Close()
        Move-Item -Force $tmp $deck
        "saved with embedded fonts"
    } else { $p.Close() }
} finally { $app.Quit() }
