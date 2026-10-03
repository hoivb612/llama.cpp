param(
    [Parameter(Mandatory=$true)]
    [string]$ResultsDir,
    [string]$CompareDir = "",
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

function Median($values) {
    $sorted = @($values | ForEach-Object { [double]$_ } | Sort-Object)
    if ($sorted.Count -eq 0) {
        return $null
    }
    $middle = [int]($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) {
        return $sorted[$middle]
    }
    return ($sorted[$middle - 1] + $sorted[$middle]) / 2.0
}

function Escape-Cell($value) {
    if ($null -eq $value) {
        return ""
    }
    return ([string]$value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Add-Table($lines, $headers, $rows) {
    $lines.Add("| " + (($headers | ForEach-Object { Escape-Cell $_ }) -join " | ") + " |")
    $lines.Add("| " + (($headers | ForEach-Object { "---" }) -join " | ") + " |")
    foreach ($row in $rows) {
        $lines.Add("| " + (($row | ForEach-Object { Escape-Cell $_ }) -join " | ") + " |")
    }
    $lines.Add("")
}

function Scope-Name($scope) {
    switch ([int]$scope) {
        0 { return "thread" }
        1 { return "wave" }
        2 { return "threadgroup" }
        default { return [string]$scope }
    }
}

function Load-Name($load) {
    switch ([int]$load) {
        0 { return "direct" }
        1 { return "scalar LDS" }
        2 { return "vector LDS" }
        3 { return "prefetch LDS" }
        4 { return "direct + transpose" }
        default { return [string]$load }
    }
}

function Layout-Name($layout) {
    if ([int]$layout -eq 0) { return "row" }
    return "column"
}

function Type-Name($type) {
    switch ([int]$type) {
        0 { return "F16" }
        1 { return "F32" }
        2 { return "BF16" }
        default { return [string]$type }
    }
}

function Case-Key($row) {
    $fields = @(
        "scope","m","n","k","k_steps","inner","wave","waves","threads",
        "a_load","b_load","a_layout","b_layout","c_layout",
        "a_source_layout","b_source_layout","a_type","b_type",
        "a_offset","b_offset","c_offset",
        "a_source_pad","b_source_pad","a_lds_pad","b_lds_pad",
        "vector_width","acc_tiles","tile_order",
        "align","epilogue","groups"
    )
    return (($fields | ForEach-Object { [string]$row.$_ }) -join ":")
}

function Is-Correct($row) {
    return $row.status -eq "ok" -or $row.status -eq "correct"
}

$csvPath = Join-Path $ResultsDir "results.csv"
if (-not (Test-Path $csvPath)) {
    throw "Missing results.csv: $csvPath"
}
if ($OutputPath -eq "") {
    $OutputPath = Join-Path $ResultsDir "report.md"
}

$data = @(Import-Csv $csvPath)
$correct = @($data | Where-Object { Is-Correct $_ })
$metadata = $data | Where-Object { -not [string]::IsNullOrWhiteSpace($_.adapter_name) } |
    Select-Object -First 1
if ($null -eq $metadata) {
    $metadata = $data | Where-Object { -not [string]::IsNullOrWhiteSpace($_.vendor_id) } |
        Select-Object -First 1
}
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# DX12 LinAlg characterization report")
$lines.Add("")
$lines.Add("- Results: ``$ResultsDir``")
if ($null -ne $metadata) {
    if (-not [string]::IsNullOrWhiteSpace($metadata.adapter_name)) {
        $lines.Add("- Adapter: $($metadata.adapter_name)")
    }
    $lines.Add("- Vendor/device: $($metadata.vendor_id)/$($metadata.device_id)")
    $lines.Add("- Driver: $($metadata.driver_version)")
    $lines.Add("- LinAlg tier: $($metadata.linalg_tier)")
}
$lines.Add("- Cases: $($data.Count)")
$lines.Add("")

$lines.Add("## Status")
$lines.Add("")
$statusRows = [System.Collections.Generic.List[object]]::new()
foreach ($group in ($data | Group-Object status | Sort-Object Name)) {
    $statusRows.Add(@($group.Name, $group.Count))
}
Add-Table $lines @("Status", "Count") $statusRows

$lines.Add("## Capability by scope and wave")
$lines.Add("")
$capRows = [System.Collections.Generic.List[object]]::new()
foreach ($group in ($data | Group-Object scope,wave | Sort-Object Name)) {
    $first = $group.Group[0]
    $counts = @{}
    foreach ($status in ($group.Group | Group-Object status)) {
        $counts[$status.Name] = $status.Count
    }
    $capRows.Add(@(
        (Scope-Name $first.scope), $first.wave, $group.Count,
        ([int]$counts["ok"] + [int]$counts["correct"]), [int]$counts["mismatch"],
        ([int]$counts["unsupported"] + [int]$counts["unsupported_group_size"]),
        [int]$counts["pso_failed"], [int]$counts["compile_failed"]
    ))
}
Add-Table $lines @(
    "Scope", "Wave", "Cases", "Correct", "Mismatch", "Unsupported",
    "PSO failed", "Compile failed"
) $capRows

$tgQueries = @($data | Where-Object {
    [int]$_.scope -eq 2 -and
    -not [string]::IsNullOrWhiteSpace($_.tg_support_flags)
})
if ($tgQueries.Count -gt 0) {
    $lines.Add("## Threadgroup runtime capabilities")
    $lines.Add("")
    $tgRows = [System.Collections.Generic.List[object]]::new()
    foreach ($group in ($tgQueries |
        Group-Object m,k,n,tg_support_flags,tg_min_threads,tg_max_threads,tg_preferred_threads |
        Sort-Object { [int]$_.Group[0].m },
                    { [int]$_.Group[0].n })) {
        $first = $group.Group[0]
        $valid = @($group.Group | Where-Object { [int]$_.tg_group_size_valid -ne 0 })
        $tgRows.Add(@(
            "$($first.m)x$($first.k)x$($first.n)",
            (([int]$first.tg_support_flags -band 1) -ne 0),
            $first.tg_min_threads,
            $first.tg_max_threads,
            $first.tg_preferred_threads,
            $valid.Count
        ))
    }
    Add-Table $lines @(
        "M x K x N", "Supported", "Min threads", "Max threads",
        "Preferred", "Tested counts in range"
    ) $tgRows
}

$addressing = @($data | Where-Object {
    $_.tag -like "wave_descriptor_offset_*" -or
    $_.tag -eq "wave_odd_stride" -or
    $_.tag -eq "wave_shape_boundary"
})
if ($addressing.Count -gt 0) {
    $lines.Add("## Addressing and shape boundaries")
    $lines.Add("")
    $addressRows = [System.Collections.Generic.List[object]]::new()
    foreach ($row in $addressing) {
        $addressRows.Add(@(
            $row.tag,
            "$($row.m)x$($row.k)x$($row.n)",
            "$($row.a_offset)/$($row.b_offset)/$($row.c_offset)",
            "$($row.a_source_pad)/$($row.b_source_pad)",
            "$(Layout-Name $row.a_layout)/$(Layout-Name $row.b_layout)",
            $row.status
        ))
    }
    Add-Table $lines @(
        "Family", "M x K x N", "A/B/C byte offsets",
        "A/B source pad", "A/B layout", "Status"
    ) $addressRows
}

$waveCorrect = @($correct | Where-Object { [int]$_.scope -eq 1 })
if ($waveCorrect.Count -gt 0) {
    $lines.Add("## Wave-scope load paths")
    $lines.Add("")
    $loadRows = [System.Collections.Generic.List[object]]::new()
    foreach ($group in ($waveCorrect | Group-Object a_load,b_load | Sort-Object Name)) {
        $first = $group.Group[0]
        $median = Median ($group.Group | ForEach-Object tflops)
        $best = ($group.Group | Measure-Object tflops -Maximum).Maximum
        $loadRows.Add(@(
            (Load-Name $first.a_load), (Load-Name $first.b_load),
            $group.Count, ("{0:F3}" -f $median), ("{0:F3}" -f [double]$best)
        ))
    }
    Add-Table $lines @("A load", "B load", "Correct", "Median TFLOP/s", "Best TFLOP/s") $loadRows

    $lines.Add("## Wave-scope staged source types")
    $lines.Add("")
    $typeRows = [System.Collections.Generic.List[object]]::new()
    $staged = @($waveCorrect | Where-Object {
        [int]$_.a_load -ne 0 -and [int]$_.a_load -eq [int]$_.b_load
    })
    foreach ($group in ($staged | Group-Object a_load,a_type,b_type | Sort-Object Name)) {
        $first = $group.Group[0]
        $typeRows.Add(@(
            (Load-Name $first.a_load), (Type-Name $first.a_type),
            (Type-Name $first.b_type), $group.Count,
            ("{0:F3}" -f (Median ($group.Group | ForEach-Object tflops)))
        ))
    }
    Add-Table $lines @("Load", "A source", "B source", "Correct", "Median TFLOP/s") $typeRows

    $lines.Add("## Wave-scope direct-load layouts")
    $lines.Add("")
    $layoutRows = [System.Collections.Generic.List[object]]::new()
    $direct = @($waveCorrect | Where-Object {
        [int]$_.a_load -eq 0 -and [int]$_.b_load -eq 0 -and
        $_.tag -eq "wave_direct"
    })
    foreach ($group in ($direct | Group-Object a_layout,b_layout | Sort-Object Name)) {
        $first = $group.Group[0]
        $layoutRows.Add(@(
            (Layout-Name $first.a_layout), (Layout-Name $first.b_layout),
            $group.Count, ("{0:F3}" -f (Median ($group.Group | ForEach-Object tflops)))
        ))
    }
    Add-Table $lines @("A layout", "B layout", "Correct", "Median TFLOP/s") $layoutRows

    $depth = @($waveCorrect | Where-Object tag -like "wave_kdepth*")
    if ($depth.Count -gt 0) {
        $lines.Add("## Wave-scope K depth")
        $lines.Add("")
        $depthRows = [System.Collections.Generic.List[object]]::new()
        foreach ($group in ($depth | Group-Object k_steps,a_load,b_load |
            Sort-Object { [int]$_.Group[0].k_steps },
                        { [int]$_.Group[0].a_load })) {
            $first = $group.Group[0]
            $depthRows.Add(@(
                $first.k_steps, (Load-Name $first.a_load),
                (Load-Name $first.b_load),
                ("{0:F3}" -f (Median ($group.Group | ForEach-Object tflops)))
            ))
        }
        Add-Table $lines @("K steps", "A load", "B load", "Median TFLOP/s") $depthRows
    }

    $geometry = @($waveCorrect | Where-Object tag -eq "wave_geometry")
    if ($geometry.Count -gt 0) {
        $lines.Add("## Wave-scope group geometry")
        $lines.Add("")
        $geometryRows = [System.Collections.Generic.List[object]]::new()
        foreach ($group in ($geometry | Group-Object waves,a_load,b_load |
            Sort-Object { [int]$_.Group[0].waves },
                        { [int]$_.Group[0].a_load })) {
            $first = $group.Group[0]
            $geometryRows.Add(@(
                $first.waves, ([int]$first.wave * [int]$first.waves),
                (Load-Name $first.a_load), (Load-Name $first.b_load),
                ("{0:F3}" -f (Median ($group.Group | ForEach-Object tflops)))
            ))
        }
        Add-Table $lines @("Waves/group", "Threads", "A load", "B load", "Median TFLOP/s") $geometryRows
    }

    $reuse = @($waveCorrect | Where-Object tag -eq "wave_compute_reuse")
    if ($reuse.Count -gt 0) {
        $lines.Add("## Wave-scope accumulator reuse")
        $lines.Add("")
        $reuseRows = [System.Collections.Generic.List[object]]::new()
        foreach ($group in ($reuse | Group-Object inner |
            Sort-Object { [int]$_.Group[0].inner })) {
            $first = $group.Group[0]
            $reuseRows.Add(@(
                $first.inner,
                ("{0:F3}" -f (Median ($group.Group | ForEach-Object tflops)))
            ))
        }
        Add-Table $lines @("Inner repeats", "Median TFLOP/s") $reuseRows
    }

    $alignment = @($waveCorrect | Where-Object tag -eq "wave_direct")
    if ($alignment.Count -gt 0) {
        $lines.Add("## Wave-scope descriptor alignment promises")
        $lines.Add("")
        $alignmentRows = [System.Collections.Generic.List[object]]::new()
        foreach ($group in ($alignment | Group-Object align |
            Sort-Object { [int]$_.Group[0].align })) {
            $first = $group.Group[0]
            $alignmentRows.Add(@(
                $first.align, $group.Count,
                ("{0:F3}" -f (Median ($group.Group | ForEach-Object tflops)))
            ))
        }
        Add-Table $lines @("Alignment", "Correct", "Median TFLOP/s") $alignmentRows
    }
}

$padding = @($data | Where-Object {
    $_.tag -eq "wave_kcontig_pad_direct" -or
    $_.tag -eq "wave_kcontig_pad_lds"
})
if ($padding.Count -gt 0) {
    $lines.Add("## K-contiguity and source padding")
    $lines.Add("")
    $paddingRows = [System.Collections.Generic.List[object]]::new()
    foreach ($group in ($padding |
        Group-Object tag,a_layout,b_layout,a_source_pad,b_source_pad |
        Sort-Object { [int]$_.Group[0].a_source_pad },
                    { [int]$_.Group[0].a_layout },
                    { [int]$_.Group[0].b_layout })) {
        $first = $group.Group[0]
        $ok = @($group.Group | Where-Object { Is-Correct $_ })
        $median = ""
        if ($ok.Count -gt 0) {
            $median = "{0:F3}" -f (Median ($ok | ForEach-Object tflops))
        }
        $paddingRows.Add(@(
            $(if ($first.tag -eq "wave_kcontig_pad_direct") {
                "direct"
            } else {
                "vector LDS"
            }),
            (Layout-Name $first.a_layout), (Layout-Name $first.b_layout),
            $first.a_source_pad, $ok.Count, $median
        ))
    }
    Add-Table $lines @("Path", "A layout", "B layout", "Pad", "Correct", "Median TFLOP/s") $paddingRows
}

$vectors = @($data | Where-Object {
    $_.tag -eq "wave_vector_width" -or
    ($_.tag -like "wave_lds_bank_*" -and
     [int]$_.a_lds_pad -eq 0 -and [int]$_.b_lds_pad -eq 0 -and
     [int]$_.a_type -eq 0 -and [int]$_.b_type -eq 0)
})
if ($vectors.Count -gt 0) {
    $lines.Add("## Global vector load widths")
    $lines.Add("")
    $vectorRows = [System.Collections.Generic.List[object]]::new()
    foreach ($group in ($vectors | Group-Object a_load,a_type,vector_width |
        Sort-Object { [int]$_.Group[0].a_load },
                    { [int]$_.Group[0].a_type },
                    { [int]$_.Group[0].vector_width })) {
        $first = $group.Group[0]
        $ok = @($group.Group | Where-Object { Is-Correct $_ })
        $median = ""
        if ($ok.Count -gt 0) {
            $median = "{0:F3}" -f (Median ($ok | ForEach-Object tflops))
        }
        $vectorRows.Add(@(
            (Load-Name $first.a_load), (Type-Name $first.a_type),
            $first.vector_width, $ok.Count, $median
        ))
    }
    Add-Table $lines @("Path", "Source", "Elements/load", "Correct", "Median TFLOP/s") $vectorRows
}

$banks = @($data | Where-Object { $_.tag -like "wave_lds_bank_*" })
if ($banks.Count -gt 0) {
    $lines.Add("## LDS stride and bank patterns")
    $lines.Add("")
    $bankRows = [System.Collections.Generic.List[object]]::new()
    foreach ($group in ($banks | Group-Object a_load,a_lds_pad,b_lds_pad |
        Sort-Object { [int]$_.Group[0].a_load },
                    { [int]$_.Group[0].a_lds_pad },
                    { [int]$_.Group[0].b_lds_pad })) {
        $first = $group.Group[0]
        $ok = @($group.Group | Where-Object { Is-Correct $_ })
        $mismatch = @($group.Group | Where-Object status -eq "mismatch")
        $unsupported = @($group.Group | Where-Object {
            $_.status -eq "unsupported" -or
            $_.status -eq "unsupported_group_size"
        })
        $median = ""
        if ($ok.Count -gt 0) {
            $median = "{0:F3}" -f (Median ($ok | ForEach-Object tflops))
        }
        $bankRows.Add(@(
            (Load-Name $first.a_load), $first.a_lds_pad, $first.b_lds_pad,
            $ok.Count, $mismatch.Count, $median
        ))
    }
    Add-Table $lines @("Path", "A LDS pad", "B LDS pad", "Correct", "Mismatch", "Median TFLOP/s") $bankRows
}

$multiWave = @($data | Where-Object tag -eq "wave_multi_acc")
if ($multiWave.Count -gt 0) {
    $lines.Add("## Multiple wave accumulator tiles")
    $lines.Add("")
    $multiRows = [System.Collections.Generic.List[object]]::new()
    foreach ($group in ($multiWave |
        Group-Object acc_tiles,waves,tile_order,epilogue |
        Sort-Object { [int]$_.Group[0].acc_tiles },
                    { [int]$_.Group[0].waves },
                    { [int]$_.Group[0].tile_order },
                    { [int]$_.Group[0].epilogue })) {
        $first = $group.Group[0]
        $ok = @($group.Group | Where-Object { Is-Correct $_ })
        $median = ""
        if ($ok.Count -gt 0) {
            $median = "{0:F3}" -f (Median ($ok | ForEach-Object tflops))
        }
        $multiRows.Add(@(
            $first.acc_tiles, $first.waves, $first.tile_order,
            $first.epilogue, $ok.Count, $median
        ))
    }
    Add-Table $lines @("Accumulator tiles", "Waves/group", "Tile order", "Epilogue", "Correct", "Median TFLOP/s") $multiRows
}

$multiTg = @($data | Where-Object tag -eq "threadgroup_multi_acc")
if ($multiTg.Count -gt 0) {
    $lines.Add("## Multiple threadgroup accumulator tiles")
    $lines.Add("")
    $multiTgRows = [System.Collections.Generic.List[object]]::new()
    foreach ($group in ($multiTg |
        Group-Object m,n,acc_tiles,tile_order,epilogue |
        Sort-Object { [int]$_.Group[0].m },
                    { [int]$_.Group[0].n },
                    { [int]$_.Group[0].acc_tiles })) {
        $first = $group.Group[0]
        $ok = @($group.Group | Where-Object { Is-Correct $_ })
        $mismatch = @($group.Group | Where-Object status -eq "mismatch")
        $unsupported = @($group.Group | Where-Object {
            $_.status -eq "unsupported" -or
            $_.status -eq "unsupported_group_size"
        })
        $pso = @($group.Group | Where-Object status -eq "pso_failed")
        $multiTgRows.Add(@(
            "$($first.m)x$($first.n)", $first.acc_tiles,
            $first.tile_order, $first.epilogue,
            $ok.Count, $mismatch.Count, $unsupported.Count, $pso.Count
        ))
    }
    Add-Table $lines @(
        "Shape", "Accumulator tiles", "Tile order", "Epilogue",
        "Correct", "Mismatch", "Unsupported", "PSO failed"
    ) $multiTgRows
}

$tgDepth = @($data | Where-Object tag -like "threadgroup_kdepth*")
if ($tgDepth.Count -gt 0) {
    $lines.Add("## Threadgroup K depth")
    $lines.Add("")
    $tgDepthRows = [System.Collections.Generic.List[object]]::new()
    foreach ($group in ($tgDepth |
        Group-Object m,n,wave,k_steps,a_load,b_load,b_layout,epilogue |
        Sort-Object { [int]$_.Group[0].m },
                    { [int]$_.Group[0].n },
                    { [int]$_.Group[0].wave },
                    { [int]$_.Group[0].k_steps },
                    { [int]$_.Group[0].b_layout },
                    { [int]$_.Group[0].epilogue })) {
        $first = $group.Group[0]
        $ok = @($group.Group | Where-Object { Is-Correct $_ })
        $mismatch = @($group.Group | Where-Object status -eq "mismatch")
        $unsupported = @($group.Group | Where-Object {
            $_.status -eq "unsupported" -or
            $_.status -eq "unsupported_group_size"
        })
        $median = ""
        if ($ok.Count -gt 0) {
            $median = "{0:F3}" -f (Median ($ok | ForEach-Object tflops))
        }
        $tgDepthRows.Add(@(
            "$($first.m)x$($first.n)", $first.wave,
            ([int]$first.k_steps * [int]$first.k),
            (Load-Name $first.a_load), (Load-Name $first.b_load),
            (Layout-Name $first.b_layout), $first.epilogue,
            $ok.Count, $mismatch.Count, $unsupported.Count, $median
        ))
    }
    Add-Table $lines @(
        "Shape", "Wave", "Total K", "A load", "B load", "B layout",
        "Epilogue", "Correct", "Mismatch", "Unsupported", "Median TFLOP/s"
    ) $tgDepthRows
}

$tg = @($data | Where-Object { [int]$_.scope -eq 2 })
if ($tg.Count -gt 0) {
    $lines.Add("## Threadgroup shapes")
    $lines.Add("")
    $shapeRows = [System.Collections.Generic.List[object]]::new()
    foreach ($group in ($tg | Group-Object m,n,wave | Sort-Object Name)) {
        $first = $group.Group[0]
        $ok = @($group.Group | Where-Object { Is-Correct $_ })
        $mismatch = @($group.Group | Where-Object status -eq "mismatch")
        $unsupported = @($group.Group | Where-Object {
            $_.status -eq "unsupported" -or
            $_.status -eq "unsupported_group_size"
        })
        $pso = @($group.Group | Where-Object status -eq "pso_failed")
        $compile = @($group.Group | Where-Object status -eq "compile_failed")
        $best = ""
        if ($ok.Count -gt 0) {
            $best = "{0:F3}" -f [double](($ok | Measure-Object tflops -Maximum).Maximum)
        }
        $shapeRows.Add(@(
            "$($first.m)x$($first.n)", $first.wave, $ok.Count,
            $mismatch.Count, $unsupported.Count, $pso.Count, $compile.Count, $best
        ))
    }
    Add-Table $lines @(
        "Shape", "Wave", "Correct", "Mismatch", "Unsupported",
        "PSO failed", "Compile failed", "Best TFLOP/s"
    ) $shapeRows
}

if ($correct.Count -gt 0) {
    $lines.Add("## Fastest correct cases")
    $lines.Add("")
    $bestRows = [System.Collections.Generic.List[object]]::new()
    foreach ($row in ($correct | Sort-Object { [double]$_.tflops } -Descending |
        Select-Object -First 20)) {
        $bestRows.Add(@(
            $row.case_id, $row.tag, (Scope-Name $row.scope),
            "$($row.m)x$($row.n)x$($row.k)", $row.k_steps, $row.inner,
            $row.wave, $row.waves, (Load-Name $row.a_load),
            (Load-Name $row.b_load), ("{0:F3}" -f [double]$row.tflops)
        ))
    }
    Add-Table $lines @("Case", "Tag", "Scope", "MxNxK", "K steps", "Inner", "Wave", "Waves", "A load", "B load", "TFLOP/s") $bestRows
}

$problemRows = [System.Collections.Generic.List[object]]::new()
foreach ($group in ($data | Where-Object { -not (Is-Correct $_) } |
    Group-Object status,tag | Sort-Object Name)) {
    $first = $group.Group[0]
    $problemRows.Add(@($first.status, $first.tag, $group.Count))
}
if ($problemRows.Count -gt 0) {
    $lines.Add("## Rejections and mismatches")
    $lines.Add("")
    Add-Table $lines @("Status", "Tag", "Count") $problemRows
}

if ($CompareDir -ne "") {
    $comparePath = Join-Path $CompareDir "results.csv"
    if (-not (Test-Path $comparePath)) {
        throw "Missing comparison results.csv: $comparePath"
    }
    $other = @(Import-Csv $comparePath | Where-Object { Is-Correct $_ })
    $otherByKey = @{}
    foreach ($row in $other) {
        $otherByKey[(Case-Key $row)] = $row
    }
    $pairs = [System.Collections.Generic.List[object]]::new()
    foreach ($row in $correct) {
        $key = Case-Key $row
        if ($otherByKey.ContainsKey($key)) {
            $rhs = $otherByKey[$key]
            $pairs.Add([pscustomobject]@{
                Left=$row
                Right=$rhs
                Ratio=([double]$rhs.tflops / [double]$row.tflops)
            })
        }
    }
    $lines.Add("## Cross-run comparison")
    $lines.Add("")
    $lines.Add("- Comparison results: ``$CompareDir``")
    $lines.Add("- Common correct cases: $($pairs.Count)")
    $lines.Add("- Ratio: comparison TFLOP/s / primary TFLOP/s")
    $lines.Add("")
    if ($pairs.Count -gt 0) {
        $ratioRows = [System.Collections.Generic.List[object]]::new()
        foreach ($pair in ($pairs | Sort-Object Ratio -Descending |
            Select-Object -First 20)) {
            $ratioRows.Add(@(
                $pair.Left.case_id, $pair.Left.tag,
                (Scope-Name $pair.Left.scope),
                "$($pair.Left.m)x$($pair.Left.n)x$($pair.Left.k)",
                ("{0:F3}" -f [double]$pair.Left.tflops),
                ("{0:F3}" -f [double]$pair.Right.tflops),
                ("{0:F3}" -f $pair.Ratio)
            ))
        }
        Add-Table $lines @("Case", "Tag", "Scope", "MxNxK", "Primary", "Comparison", "Ratio") $ratioRows
    }
}

$lines | Set-Content -Encoding ascii $OutputPath
Write-Host "Report: $OutputPath"
