# Smoke tests for llama-mm-embed (Windows / PowerShell).
#
#   .\run-tests.ps1 -ModelDir D:\models\EmbeddingGemma-2\Unsloth [-Exe path\to\llama-mm-embed.exe]
#
# 1. synthetic shapes      : generated images vs. matching + decoy captions
# 2. real photos           : Windows wallpapers, 256-dim Matryoshka embeddings
# 3. capped image tokens   : same photos with --image-max-tokens 280 + --all-pairs
# 4. text only             : no --mmproj

param(
    [Parameter(Mandatory = $true)][string] $ModelDir,
    [string] $Exe    = "$PSScriptRoot\..\..\build\bin\Release\llama-mm-embed.exe",
    [string] $Model  = "embeddinggemma-2-Q8_0.gguf",
    [string] $MmProj = "mmproj-Q8_0.gguf",
    [string] $Only   = ""   # run a single test: shapes | photos | capped | text
)

$ErrorActionPreference = "Stop"
$m  = Join-Path $ModelDir $Model
$mp = Join-Path $ModelDir $MmProj
foreach ($f in $Exe, $m, $mp) { if (-not (Test-Path $f)) { throw "not found: $f" } }

function Run([string] $title, [string[]] $a) {
    Write-Host "`n===== $title =====" -ForegroundColor Cyan
    & $Exe -m $m @a 2>&1 | Where-Object { $_ -notmatch '^(load|init_audio|\s+https)' }
    if ($LASTEXITCODE -ne 0) { throw "$title failed (exit $LASTEXITCODE)" }
}

# synthetic test images
Add-Type -AssemblyName System.Drawing
$imgDir = Join-Path $env:TEMP "mmembed"
New-Item -ItemType Directory $imgDir -Force | Out-Null
function MakeImage([string] $name, [scriptblock] $draw) {
    $b = New-Object System.Drawing.Bitmap 384, 384
    $g = [System.Drawing.Graphics]::FromImage($b)
    $g.SmoothingMode = 'AntiAlias'
    $g.Clear([System.Drawing.Color]::White)
    & $draw $g
    $b.Save((Join-Path $imgDir "$name.png"))
    $g.Dispose(); $b.Dispose()
}
MakeImage red_circle     { param($g) $g.FillEllipse([System.Drawing.Brushes]::Red, 60, 60, 264, 264) }
MakeImage blue_square    { param($g) $g.FillRectangle([System.Drawing.Brushes]::Blue, 70, 70, 244, 244) }
MakeImage green_triangle { param($g)
    $pts = [System.Drawing.Point[]]@((New-Object System.Drawing.Point 192, 40),
                                     (New-Object System.Drawing.Point 344, 340),
                                     (New-Object System.Drawing.Point 40, 340))
    $g.FillPolygon([System.Drawing.Brushes]::Green, $pts) }
MakeImage stop_text      { param($g)
    $font = New-Object System.Drawing.Font('Arial', 72, [System.Drawing.FontStyle]::Bold)
    $g.DrawString('STOP', $font, [System.Drawing.Brushes]::Black, 40, 130) }

# Windows wallpapers (content as of Windows 11)
$s = 'C:\Windows\Web\Screen'
$photos = @('img100.jpg', 'img101.jpg', 'img102.jpg', 'img103.jpg') | ForEach-Object { "--image"; Join-Path $s $_ }
$photoTexts = @(
    '--text', 'sunset over a calm lake with snowy mountains',  # img102
    '--text', 'a glowing blue and purple sphere on a dark background',  # img101
    '--text', 'flowing purple and orange ribbons',  # img103
    '--text', 'abstract blue curved shapes')  # img100

if (-not $Only -or $Only -eq 'shapes') {
    Run "1. synthetic shapes" (@('--mmproj', $mp, '--image-dir', $imgDir,
        '--text', 'a red circle', '--text', 'a blue square', '--text', 'a green triangle',
        '--text', 'a sign with the word STOP', '--text', 'a photo of a cat'))
}
if (-not $Only -or $Only -eq 'photos') {
    Run "2. real photos, 256 dims" (@('--mmproj', $mp) + $photos + @('--image', (Join-Path $imgDir 'red_circle.png')) +
        $photoTexts + @('--text', 'a red circle', '--dim', '256', '--top-k', '2'))
}
if (-not $Only -or $Only -eq 'capped') {
    Run "3. real photos, --image-max-tokens 280" (@('--mmproj', $mp) + $photos + $photoTexts +
        @('--image-max-tokens', '280', '--all-pairs'))
}
if (-not $Only -or $Only -eq 'text') {
    Run "4. text only" @('--text', 'how do I bake bread', '--text', 'recipe for sourdough loaf',
        '--text', 'car engine repair', '--text', 'fixing a broken alternator', '--top-k', '1')
}

Write-Host "`nAll tests ran. Expected: every image's '*' falls on its matching caption." -ForegroundColor Green
