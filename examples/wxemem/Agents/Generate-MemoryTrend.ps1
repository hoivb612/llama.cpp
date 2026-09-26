[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$DailyRoot,
    [ValidateNotNullOrEmpty()]
    [string]$OutputDirectory,
    [switch]$DataOnly,
    [string]$BuildList
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 may not populate PSScriptRoot during parameter binding.
if (-not $PSBoundParameters.ContainsKey('DailyRoot')) {
    $DailyRoot = Join-Path $PSScriptRoot '..\daily'
}
if (-not $PSBoundParameters.ContainsKey('OutputDirectory')) {
    $OutputDirectory = Join-Path $PSScriptRoot 'output'
}
$GeneratorVersion = '1.4.1'
$BaselineStagePatterns = @('beforepnp', 'prepnp*', 'nopnp')
$PostPnpStagePatterns = @('afterpnp', 'postpnp*', 'pnp')
$SignInStagePatterns = @('xboxappsignin', 'signin', 'postsignin')

function Read-BuildList([string]$Path, [string]$Root) {
    $builds = [System.Collections.Generic.List[string]]::new()
    $lineNumber = 0
    foreach ($line in Get-Content -LiteralPath $Path) {
        $lineNumber++
        $name = $line.Trim()
        if ($name.Length -eq 0 -or $name.StartsWith('#')) { continue }
        if ($name -notmatch '^\d+\.\d+\.\d{6}-\d{4}$') {
            throw "Invalid build ID at line $lineNumber in '$Path': '$name'."
        }
        if ($builds.Contains($name)) {
            throw "Duplicate build ID at line $lineNumber in '$Path': '$name'."
        }
        if (-not (Test-Path -LiteralPath (Join-Path $Root $name) -PathType Container)) {
            throw "Listed build '$name' does not exist under '$Root'."
        }
        $builds.Add($name)
    }
    if ($builds.Count -eq 0) { throw "Build list '$Path' contains no build IDs." }
    return $builds.ToArray()
}

function Read-Capture([string]$Path) {
    try {
        $data = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    } catch {
        throw "Cannot parse capture '$Path': $($_.Exception.Message)"
    }
    $physicalProperty = $data.PSObject.Properties['physical']
    if ($null -eq $physicalProperty -or $null -eq $physicalProperty.Value) {
        throw "Missing physical object in '$Path'."
    }
    $physical = $physicalProperty.Value
    foreach ($field in @('used_bytes', 'total_bytes')) {
        $property = $physical.PSObject.Properties[$field]
        if ($null -eq $property -or
            $property.Value -isnot [ValueType] -or $property.Value -is [bool]) {
            throw "Missing or nonnumeric physical.$field in '$Path'."
        }
        $number = [double]$property.Value
        if ([double]::IsNaN($number) -or [double]::IsInfinity($number) -or
            $number -lt 0 -or $number -gt [long]::MaxValue -or
            $number -ne [math]::Truncate($number)) {
            throw "Invalid physical.$field in '$Path'."
        }
    }
    if ($physical.total_bytes -le 0 -or $physical.used_bytes -gt $physical.total_bytes) {
        throw "Inconsistent physical used/total bytes in '$Path'."
    }
    $version = $data.PSObject.Properties['wxemem_version']
    [pscustomobject]@{
        Used = [long]$physical.used_bytes
        Total = [long]$physical.total_bytes
        Version = $(if ($null -ne $version) { [string]$version.Value } else { 'unversioned' })
    }
}

function Get-StageData([string]$Wxe, [string]$Build, [string[]]$Patterns, [string]$Label) {
    $result = [ordered]@{
        status = 'missing Kennan\WXE'
        used_bytes = $null
        used_gib = $null
        total_bytes = $null
        capture_date = ''
        wxemem_version = ''
        source = ''
    }
    $candidates = @()
    if (Test-Path -LiteralPath $Wxe -PathType Container) {
        $stages = @(Get-ChildItem -LiteralPath $Wxe -Directory |
            Where-Object {
                $stage = $_
                $Patterns.Where({ $stage.Name -like $_ }).Count -gt 0
            })
        $result.status = "no $Label stage"
        if ($stages.Count -gt 0) {
            $result.status = "no direct $Label JSON (nested folders ignored)"
            $candidates = @(foreach ($stage in $stages) {
                Get-ChildItem -LiteralPath $stage.FullName -File |
                    Where-Object {
                        $_.Name -like 'wxemem*' -and
                        ($_.Extension -eq '.json' -or
                         ($_.Extension -eq '.txt' -and $_.Name -match 'json'))
                    }
            })
        }
    }
    if ($candidates.Count -gt 1) {
        throw "Ambiguous $Label captures for $Build; select one capture before continuing:`n$($candidates.FullName -join "`n")"
    }
    if ($candidates.Count -eq 1) {
        $file = $candidates[0]
        $capture = Read-Capture $file.FullName
        $result.status = 'included'
        $result.used_bytes = $capture.Used
        $result.used_gib = $capture.Used / 1GB
        $result.total_bytes = $capture.Total
        $result.wxemem_version = $capture.Version
        $result.source = $file.FullName
        if ($file.Name -match '^wxemem_(20\d{6})_') {
            $result.capture_date = [datetime]::ParseExact(
                $Matches[1], 'yyyyMMdd',
                [Globalization.CultureInfo]::InvariantCulture).ToString('yyyy-MM-dd')
        }
    }
    return $result
}

function Get-TrendData([string]$Root, [string[]]$SelectedBuilds) {
    $rows = [System.Collections.Generic.List[object]]::new()
    $excluded = [System.Collections.Generic.List[string]]::new()
    foreach ($build in Get-ChildItem -LiteralPath $Root -Directory) {
        if ($build.Name -notmatch '^\d+\.\d+\.(\d{6}-\d{4})$') {
            $excluded.Add($build.Name)
            continue
        }
        if ($SelectedBuilds.Count -gt 0 -and $build.Name -notin $SelectedBuilds) {
            continue
        }
        $stamp = [datetime]::ParseExact(
            "20$($Matches[1])", 'yyyyMMdd-HHmm',
            [Globalization.CultureInfo]::InvariantCulture)
        $wxe = Join-Path $build.FullName 'Kennan\WXE'
        $pre = Get-StageData $wxe $build.Name $BaselineStagePatterns 'before-PnP'
        $post = Get-StageData $wxe $build.Name $PostPnpStagePatterns 'post-PnP'
        $signin = Get-StageData $wxe $build.Name $SignInStagePatterns 'sign-in'
        $row = [ordered]@{
            build = $build.Name
            build_date = $stamp.ToString('yyyy-MM-dd')
            build_timestamp = $stamp.ToString('yyyy-MM-ddTHH:mm:ss')
        }
        foreach ($key in $pre.Keys) { $row[$key] = $pre[$key] }
        foreach ($key in $post.Keys) { $row["postpnp_$key"] = $post[$key] }
        foreach ($key in $signin.Keys) { $row["signin_$key"] = $signin[$key] }
        $rows.Add([pscustomobject]$row)
    }
    $sorted = @($rows | Sort-Object build_timestamp, build)
    if (@($sorted | Where-Object {
        $_.status -eq 'included' -or $_.postpnp_status -eq 'included' -or
        $_.signin_status -eq 'included'
    }).Count -eq 0) {
        throw "No eligible before-PnP, post-PnP, or sign-in JSON captures found under '$Root'."
    }
    [pscustomobject]@{ Rows = $sorted; Excluded = @($excluded.ToArray() | Sort-Object) }
}

function Add-Text($Slide, [string]$Text, $X, $Y, $W, $H, $Size, $Color) {
    $shape = $Slide.Shapes.AddTextbox(1, $X, $Y, $W, $H)
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = 'Aptos'
    $shape.TextFrame.TextRange.Font.Size = $Size
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color
    return $shape
}

function Get-SeriesSummary($Rows, [string]$Prefix, [string]$Label) {
    $available = @($Rows | Where-Object { $_."${Prefix}status" -eq 'included' })
    if ($available.Count -eq 0) { return "${Label}: no eligible captures" }
    $first = $available[0]
    $last = $available[-1]
    $peak = $available | Sort-Object "${Prefix}used_bytes" -Descending | Select-Object -First 1
    $delta = ($last."${Prefix}used_bytes" - $first."${Prefix}used_bytes") / 1GB
    return '{0}: latest {1:N2} GiB ({2}) | first {3:N2} GiB | change {4:+0.00;-0.00;0.00} GiB | peak {5:N2} GiB' -f
        $Label, $last."${Prefix}used_gib", $last.build_date,
        $first."${Prefix}used_gib", $delta, $peak."${Prefix}used_gib"
}

function Set-SeriesStyle($Series, $Color, $Marker) {
    $Series.Format.Line.ForeColor.RGB = $Color
    $Series.Format.Line.Weight = 2.5
    $Series.MarkerStyle = $Marker
    $Series.MarkerSize = 5
    $Series.MarkerForegroundColor = $Color
    $Series.MarkerBackgroundColor = $Color
}

function Write-Presentation($Rows, [string]$Path) {
    $included = @($Rows | Where-Object status -eq 'included')
    $missing = @($Rows | Where-Object status -ne 'included')
    $postIncluded = @($Rows | Where-Object postpnp_status -eq 'included')
    $postMissing = @($Rows | Where-Object postpnp_status -ne 'included')
    $signinIncluded = @($Rows | Where-Object signin_status -eq 'included')
    $signinMissing = @($Rows | Where-Object signin_status -ne 'included')
    $navy = 0x382418
    $blue = 0xC47D18
    $orange = 0x297FE6
    $green = 0x579B2B
    $gray = 0x706050
    $app = $null
    $deck = $null
    $workbook = $null
    try {
        $app = New-Object -ComObject PowerPoint.Application
        $deck = $app.Presentations.Add(-1)
        $deck.PageSetup.SlideWidth = 1152
        $deck.PageSetup.SlideHeight = 648
        $slide = $deck.Slides.Add(1, 12)
        $null = Add-Text $slide 'Trending memory usage' 36 18 1080 52 32 $navy
        $null = Add-Text $slide 'Kennan | WXE | Before-PnP / post-PnP / sign-in | physical.used_bytes' 38 72 1080 32 17 $gray
        $null = Add-Text $slide (Get-SeriesSummary $Rows '' 'Before-PnP') 38 109 1080 24 14 $blue
        $null = Add-Text $slide (Get-SeriesSummary $Rows 'postpnp_' 'Post-PnP') 38 132 1080 24 14 $orange
        $null = Add-Text $slide (Get-SeriesSummary $Rows 'signin_' 'Sign-in') 38 155 1080 24 14 $green

        # Line with markers; source values remain editable in its embedded workbook.
        $chartShape = $slide.Shapes.AddChart(65, 32, 190, 1088, 374)
        $chart = $chartShape.Chart
        $chart.ChartData.Activate()
        $workbook = $chart.ChartData.Workbook
        $sheet = $workbook.Worksheets.Item(1)
        $sheet.Cells.Clear() | Out-Null
        $sheet.Cells.Item(1, 1).Value2 = 'Build date'
        $sheet.Cells.Item(1, 2).Value2 = 'Before-PnP'
        $sheet.Cells.Item(1, 3).Value2 = 'Post-PnP'
        $sheet.Cells.Item(1, 4).Value2 = 'Sign-in'
        $sheet.Cells.Item(1, 5).Value2 = 'Build'
        $sheet.Cells.Item(1, 6).Value2 = 'Before-PnP source'
        $sheet.Cells.Item(1, 7).Value2 = 'Before-PnP status'
        $sheet.Cells.Item(1, 8).Value2 = 'Post-PnP source'
        $sheet.Cells.Item(1, 9).Value2 = 'Post-PnP status'
        $sheet.Cells.Item(1, 10).Value2 = 'Sign-in source'
        $sheet.Cells.Item(1, 11).Value2 = 'Sign-in status'
        $sheet.Range("A2:A$($Rows.Count + 1)").NumberFormat = '@'
        $years = @($Rows.build_date | ForEach-Object { $_.Substring(0, 4) } | Select-Object -Unique)
        for ($i = 0; $i -lt $Rows.Count; $i++) {
            $r = $Rows[$i]
            $n = $i + 2
            $label = $r.build_date.Substring(5).Replace('-', '/')
            if ($years.Count -gt 1) { $label = $r.build_date }
            $sheet.Cells.Item($n, 1).Value2 = $label
            if ($r.status -eq 'included') {
                $sheet.Cells.Item($n, 2).Value2 = [double]$r.used_gib
            }
            if ($r.postpnp_status -eq 'included') {
                $sheet.Cells.Item($n, 3).Value2 = [double]$r.postpnp_used_gib
            }
            if ($r.signin_status -eq 'included') {
                $sheet.Cells.Item($n, 4).Value2 = [double]$r.signin_used_gib
            }
            $sheet.Cells.Item($n, 5).Value2 = $r.build
            $sheet.Cells.Item($n, 6).Value2 = $r.source
            $sheet.Cells.Item($n, 7).Value2 = $r.status
            $sheet.Cells.Item($n, 8).Value2 = $r.postpnp_source
            $sheet.Cells.Item($n, 9).Value2 = $r.postpnp_status
            $sheet.Cells.Item($n, 10).Value2 = $r.signin_source
            $sheet.Cells.Item($n, 11).Value2 = $r.signin_status
        }
        $sheet.Calculate()
        $chart.SetSourceData("'$($sheet.Name)'!`$A`$1:`$D`$$($Rows.Count + 1)", 2)
        $chart.HasTitle = $false
        $chart.HasLegend = $true
        $chart.Legend.Position = -4160
        $chart.Legend.Font.Size = 12
        $chart.DisplayBlanksAs = 1
        $valueAxis = $chart.Axes(2)
        $valueAxis.MinimumScale = 0
        $valueAxis.HasTitle = $true
        $valueAxis.AxisTitle.Text = 'Physical memory used (GiB)'
        $valueAxis.TickLabels.NumberFormat = '0.0'
        $valueAxis.TickLabels.Font.Size = 11
        $categoryAxis = $chart.Axes(1)
        $categoryAxis.CategoryType = 2
        $categoryAxis.TickLabelSpacing = 1
        $categoryAxis.TickLabels.Font.Size = 10
        $categoryAxis.HasTitle = $true
        $categoryAxis.AxisTitle.Text = "Build date ($($years -join ', ')); one category per build"
        Set-SeriesStyle ($chart.SeriesCollection(1)) $blue 8
        Set-SeriesStyle ($chart.SeriesCollection(2)) $orange 1
        Set-SeriesStyle ($chart.SeriesCollection(3)) $green 3
        $chart.Refresh()
        $coverage = 'Captures: before-PnP {0}/{3} | post-PnP {1}/{3} | sign-in {2}/{3} | {4} to {5}' -f
            $included.Count, $postIncluded.Count, $signinIncluded.Count, $Rows.Count, $Rows[0].build_date, $Rows[-1].build_date
        $null = Add-Text $slide $coverage 38 568 1080 27 14 $navy
        $null = Add-Text $slide 'Sign-in: xboxappsignin / signin / postsignin. Direct JSON only; missing captures are gaps, not zero.' 38 600 1080 24 11 $gray
        $notes = @(
            "Generator version: $GeneratorVersion"
            'Metric: physical.used_bytes / 1073741824. Equal category spacing, not elapsed time. Dates are build dates, not capture dates.'
            'See memory-trend.csv for all build IDs, exact bytes, versions, and source paths.'
            'Each series summary uses its own eligible captures; these may cover different build dates.'
            "Before-PnP stages: $($BaselineStagePatterns -join ', '). Post-PnP stages: $($PostPnpStagePatterns -join ', ')."
            "Sign-in stages: $($SignInStagePatterns -join ', '). presignin and Steam sign-in stages excluded."
            'Missing before-PnP captures:'
            ($missing | ForEach-Object { "$($_.build): $($_.status)" })
            'Missing post-PnP captures:'
            ($postMissing | ForEach-Object { "$($_.build): $($_.postpnp_status)" })
            'Missing sign-in captures:'
            ($signinMissing | ForEach-Object { "$($_.build): $($_.signin_status)" })
        ) -join "`r`n"
        $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $notes
        $deck.SaveAs($Path, 24)
        $slide.Export([IO.Path]::ChangeExtension($Path, '.png'), 'PNG', 1600, 900)
        $workbook.Close($true)
        $workbook = $null
        $deck.Save()
    } finally {
        if ($null -ne $workbook) { $workbook.Close($false) }
        if ($null -ne $deck) { $deck.Close() }
        # Do not quit applications: an existing Office session may belong to the user.
        foreach ($obj in @($workbook, $deck, $app)) {
            if ($null -ne $obj -and [Runtime.InteropServices.Marshal]::IsComObject($obj)) {
                $null = [Runtime.InteropServices.Marshal]::ReleaseComObject($obj)
            }
        }
    }
}

$root = (Resolve-Path -LiteralPath $DailyRoot).Path
$selectedBuilds = @()
$buildListPath = $null
if ($PSBoundParameters.ContainsKey('BuildList')) {
    if ([string]::IsNullOrWhiteSpace($BuildList)) { throw 'BuildList must name a text file.' }
    $buildListPath = (Resolve-Path -LiteralPath $BuildList).Path
    $selectedBuilds = @(Read-BuildList $buildListPath $root)
}
$data = Get-TrendData $root $selectedBuilds
$null = New-Item -ItemType Directory -Path $OutputDirectory -Force
$out = (Resolve-Path -LiteralPath $OutputDirectory).Path
$data.Rows | Export-Csv -LiteralPath (Join-Path $out 'memory-trend.csv') -NoTypeInformation -Encoding UTF8
$missing = @($data.Rows | Where-Object status -ne 'included')
$postMissing = @($data.Rows | Where-Object postpnp_status -ne 'included')
$signinMissing = @($data.Rows | Where-Object signin_status -ne 'included')
$coverage = [ordered]@{
    generator_version = $GeneratorVersion
    generated_at = [datetime]::Now.ToString('o')
    daily_root = $root
    build_list_file = $buildListPath
    selected_builds = $selectedBuilds
    metric = 'physical.used_bytes'
    stage_patterns = $BaselineStagePatterns
    postpnp_stage_patterns = $PostPnpStagePatterns
    signin_stage_patterns = $SignInStagePatterns
    capture_search = 'direct stage files only; nested folders (including Houseman and Omni) ignored'
    date_basis = 'build directory timestamp, not capture date'
    total_builds = $data.Rows.Count
    included_builds = $data.Rows.Count - $missing.Count
    missing_builds = $missing
    postpnp_included_builds = $data.Rows.Count - $postMissing.Count
    postpnp_missing_builds = $postMissing
    signin_included_builds = $data.Rows.Count - $signinMissing.Count
    signin_missing_builds = $signinMissing
    excluded_nonbuild_directories = $data.Excluded
}
$coverage | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $out 'memory-trend.coverage.json') -Encoding UTF8
foreach ($row in $missing) { Write-Warning "Before-PnP $($row.build): $($row.status)" }
foreach ($row in $postMissing) { Write-Warning "Post-PnP $($row.build): $($row.postpnp_status)" }
foreach ($row in $signinMissing) { Write-Warning "Sign-in $($row.build): $($row.signin_status)" }
if (-not $DataOnly) {
    $pptx = Join-Path $out 'memory-trend.pptx'
    Write-Presentation $data.Rows $pptx
    Write-Output "PowerPoint: $pptx"
}
Write-Output "Captures: before-PnP $($coverage.included_builds)/$($coverage.total_builds), post-PnP $($coverage.postpnp_included_builds)/$($coverage.total_builds), sign-in $($coverage.signin_included_builds)/$($coverage.total_builds). Data: $out"
