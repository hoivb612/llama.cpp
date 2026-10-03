param(
    [ValidateSet("quick", "full", "extended-quick", "extended", "exhaustive")]
    [string]$Profile = "full",
    [string]$Dxc = "C:\dxc\1.10.2605.37\bin\x64\dxc.exe",
    [string]$DxcInclude = "C:\dxc\1.10.2605.37\inc\hlsl",
    [string]$AgilityRoot = "C:\AgilitySDK\1.721.3-preview",
    [int]$SdkVersion = 721,
    [int]$Adapter = 0,
    [string]$OutputDir = "",
    [string]$TagFilter = "",
    [int]$TimedDispatches = 40,
    [int]$WarmupDispatches = 8
)

$ErrorActionPreference = "Continue"
$isQuick = $Profile -eq "quick" -or $Profile -eq "extended-quick"
$isExtended = $Profile -eq "extended" -or $Profile -eq "extended-quick"
$root = $PSScriptRoot
$shader = Join-Path $root "linalg_bench.hlsl"
$hostSource = Join-Path $root "linalg_bench_host.cpp"

if ($OutputDir -eq "") {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $root "results\$env:COMPUTERNAME-$stamp"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$shaderDir = Join-Path $OutputDir "shaders"
New-Item -ItemType Directory -Force -Path $shaderDir | Out-Null
$csvPath = Join-Path $OutputDir "results.csv"
$jsonlPath = Join-Path $OutputDir "results.jsonl"
$compileLog = Join-Path $OutputDir "compile.log"

foreach ($path in @($Dxc, $DxcInclude, $shader, $hostSource)) {
    if (-not (Test-Path $path)) {
        throw "Missing required path: $path"
    }
}

$agilityBinCandidates = @(
    (Join-Path $AgilityRoot "build\native\bin\x64"),
    (Join-Path $AgilityRoot "bin\x64"),
    $AgilityRoot
)
$agilityIncludeCandidates = @(
    (Join-Path $AgilityRoot "build\native\include"),
    (Join-Path $AgilityRoot "include")
)
$agilityBin = $agilityBinCandidates |
    Where-Object { Test-Path (Join-Path $_ "D3D12Core.dll") } |
    Select-Object -First 1
$agilityInclude = $agilityIncludeCandidates |
    Where-Object { Test-Path (Join-Path $_ "d3d12.h") } |
    Select-Object -First 1
if (-not $agilityBin -or -not $agilityInclude) {
    throw "Could not locate Agility SDK bin/include under $AgilityRoot"
}

$hostExe = Join-Path $OutputDir "linalg_bench_host.exe"
$hostObj = Join-Path $OutputDir "linalg_bench_host.obj"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "Visual Studio vswhere.exe not found"
}
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {
    throw "Visual Studio C++ tools not found"
}
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$vsInstaller = Split-Path $vswhere
$compileHost = 'set "PATH={0};%PATH%" && call "{1}" >nul && cl /nologo /std:c++17 /EHsc /O2 "{2}" /Fo:"{3}" /Fe:"{4}" /I"{5}" /link d3d12.lib dxgi.lib dxguid.lib' -f `
    $vsInstaller, $vcvars, $hostSource, $hostObj, $hostExe, $agilityInclude
& $env:ComSpec /c $compileHost
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $hostExe)) {
    throw "Failed to compile linalg_bench_host.cpp"
}

$cases = [System.Collections.Generic.List[object]]::new()
function Add-Case {
    param(
        [string]$Tag,
        [int]$Scope = 1,
        [int]$M = 16,
        [int]$N = 16,
        [int]$K = 16,
        [int]$KSteps = 16,
        [int]$Inner = 1,
        [int]$Wave = 32,
        [int]$Waves = 1,
        [int]$Threads = 32,
        [int]$ALoad = 0,
        [int]$BLoad = 0,
        [int]$ALayout = 0,
        [int]$BLayout = 0,
        [int]$CLayout = 0,
        [int]$ASourceLayout = -1,
        [int]$BSourceLayout = -1,
        [int]$AType = 0,
        [int]$BType = 0,
        [int]$AOffset = 0,
        [int]$BOffset = 0,
        [int]$COffset = 0,
        [int]$ASourcePad = 0,
        [int]$BSourcePad = 0,
        [int]$ALdsPad = 0,
        [int]$BLdsPad = 0,
        [int]$VectorWidth = 4,
        [int]$AccTiles = 1,
        [int]$TileOrder = 0,
        [int]$Align = 32,
        [int]$Epilogue = 0,
        [int[]]$Groups = @(256)
    )
    $resolvedASource = if ($ASourceLayout -ge 0) {
        $ASourceLayout
    } elseif ($ALoad -eq 0 -or $Scope -eq 0) {
        $ALayout
    } else {
        0
    }
    $resolvedBSource = if ($BSourceLayout -ge 0) {
        $BSourceLayout
    } elseif ($BLoad -eq 0) {
        $BLayout
    } else {
        0
    }
    $cases.Add([pscustomobject]@{
        Tag=$Tag; Scope=$Scope; M=$M; N=$N; K=$K; KSteps=$KSteps;
        Inner=$Inner; Wave=$Wave; Waves=$Waves; Threads=$Threads;
        ALoad=$ALoad; BLoad=$BLoad; ALayout=$ALayout; BLayout=$BLayout;
        CLayout=$CLayout; ASourceLayout=$resolvedASource;
        BSourceLayout=$resolvedBSource; AType=$AType; BType=$BType; Align=$Align;
        AOffset=$AOffset; BOffset=$BOffset; COffset=$COffset;
        ASourcePad=$ASourcePad; BSourcePad=$BSourcePad;
        ALdsPad=$ALdsPad; BLdsPad=$BLdsPad; VectorWidth=$VectorWidth;
        AccTiles=$AccTiles; TileOrder=$TileOrder;
        Epilogue=$Epilogue; Groups=$Groups
    })
}

$groupSet = if ($Profile -eq "exhaustive") { @(1, 64, 512, 4096) } else { @(4096) }

# Descriptor layouts, alignment promises, transpose cast, and epilogues.
$aligns = if ($isQuick) { @(32, 128) } else { @(0, 16, 32, 64, 128) }
$layouts = if ($isQuick) {
    @(
        [pscustomobject]@{ A=0; B=0 },
        [pscustomobject]@{ A=0; B=1 },
        [pscustomobject]@{ A=1; B=0 }
    )
} else {
    @(
        [pscustomobject]@{ A=0; B=0 },
        [pscustomobject]@{ A=0; B=1 },
        [pscustomobject]@{ A=1; B=0 },
        [pscustomobject]@{ A=1; B=1 }
    )
}
$epilogues = if ($isQuick) { @(0, 2) } else { @(0, 1, 2) }
foreach ($layout in $layouts) {
    foreach ($align in $aligns) {
        foreach ($epi in $epilogues) {
            Add-Case -Tag "wave_direct" -ALayout $layout.A -BLayout $layout.B `
                -Align $align -Epilogue $epi -Groups $groupSet
        }
    }
}
foreach ($align in $aligns) {
    foreach ($epi in $epilogues) {
        Add-Case -Tag "wave_qk_cast" -BLoad 4 -Align $align `
            -Epilogue $epi -Groups $groupSet
    }
}

# Actual byte offsets, odd leading dimensions, and native shape boundaries.
$offsets = if ($isQuick) { @(0, 4, 28) } else { @(0, 2, 4, 12, 28, 32) }
foreach ($offset in $offsets) {
    Add-Case -Tag "wave_descriptor_offset_a" -AOffset $offset -Align 0 `
        -KSteps 1 -Groups @(64)
    Add-Case -Tag "wave_descriptor_offset_b" -BOffset $offset -Align 0 `
        -KSteps 1 -Groups @(64)
    Add-Case -Tag "wave_descriptor_offset_c" -COffset $offset -Align 0 `
        -KSteps 1 -Groups @(64)
}
foreach ($pad in $(if ($isQuick) { @(1, 3) } else { @(1, 3, 5, 7, 15) })) {
    foreach ($layout in $layouts) {
        Add-Case -Tag "wave_odd_stride" -ALayout $layout.A -BLayout $layout.B `
            -ASourcePad $pad -BSourcePad $pad -Align 0 `
            -KSteps 1 -Groups @(64)
    }
}
foreach ($shape in @(
    [pscustomobject]@{ M=16; N=8; K=8 },
    [pscustomobject]@{ M=16; N=16; K=8 },
    [pscustomobject]@{ M=16; N=16; K=16 }
)) {
    Add-Case -Tag "wave_shape_boundary" -M $shape.M -N $shape.N -K $shape.K `
        -KSteps 1 -Align 0 -Groups @(64)
}

# LDS scalar, wide, and register-prefetched staging with every matrix layout.
$stageLoads = if ($isQuick) { @(1, 2, 3) } else { @(1, 2, 3) }
$sourceTypes = if ($isQuick) { @(0) } else { @(0, 1, 2) }
foreach ($load in $stageLoads) {
    foreach ($layout in $layouts) {
        foreach ($type in $sourceTypes) {
            foreach ($epi in $epilogues) {
                Add-Case -Tag "wave_lds" -ALoad $load -BLoad $load `
                    -ALayout $layout.A -BLayout $layout.B `
                    -AType $type -BType $type -Epilogue $epi -Groups $groupSet
            }
        }
    }
}

# Mixed descriptor/LDS paths isolate one operand at a time.
if (-not $isQuick) {
    foreach ($load in @(1, 2)) {
        foreach ($layout in $layouts) {
            Add-Case -Tag "wave_a_direct_b_lds" -ALoad 0 -BLoad $load `
                -ALayout $layout.A -BLayout $layout.B -Groups $groupSet
            Add-Case -Tag "wave_a_lds_b_direct" -ALoad $load -BLoad 0 `
                -ALayout $layout.A -BLayout $layout.B -Groups $groupSet
        }
    }
    foreach ($load in @(1, 2, 3)) {
        foreach ($matrixLayout in $layouts) {
            foreach ($sourceLayout in $layouts) {
                Add-Case -Tag "wave_lds_source_layout" -ALoad $load -BLoad $load `
                    -ALayout $matrixLayout.A -BLayout $matrixLayout.B `
                    -ASourceLayout $sourceLayout.A -BSourceLayout $sourceLayout.B `
                    -Groups $groupSet
            }
        }
    }
}

# Wave width, waves per group, K depth, and compute reuse.
$waveSizes = if ($isQuick) { @(32) } else { @(16, 32, 64) }
$waveCounts = if ($isQuick) { @(1, 4) } else { @(1, 2, 4, 8) }
foreach ($wave in $waveSizes) {
    foreach ($waves in $waveCounts) {
        foreach ($load in @(0, 2, 3)) {
            Add-Case -Tag "wave_geometry" -Wave $wave -Waves $waves `
                -Threads ($wave * $waves) -ALoad $load -BLoad $load `
                -Groups $groupSet
        }
    }
}
$depths = if ($isQuick) { @(1, 16) } else { @(1, 4, 16, 64) }
foreach ($depth in $depths) {
    foreach ($load in @(0, 2, 3)) {
        Add-Case -Tag "wave_kdepth" -KSteps $depth -ALoad $load -BLoad $load `
            -Groups $groupSet
    }
}
foreach ($inner in @(1, 4, 16)) {
    Add-Case -Tag "wave_compute_reuse" -Inner $inner -Groups $groupSet
}

if ($isExtended) {
    foreach ($load in @(0, 2, 3)) {
        Add-Case -Tag "wave_kdepth_mid" -KSteps 8 `
            -ALoad $load -BLoad $load -Align 0 -Groups $groupSet
    }

    # K-contiguity and physical source padding.
    $extendedLayouts = if ($isQuick) {
        @(
            [pscustomobject]@{ A=0; B=1 },
            [pscustomobject]@{ A=1; B=0 }
        )
    } else {
        @(
            [pscustomobject]@{ A=0; B=0 },
            [pscustomobject]@{ A=0; B=1 },
            [pscustomobject]@{ A=1; B=0 },
            [pscustomobject]@{ A=1; B=1 }
        )
    }
    $extendedPads = if ($isQuick) { @(0, 2, 8) } else { @(0, 2, 4, 8, 16) }
    foreach ($layout in $extendedLayouts) {
        foreach ($pad in $extendedPads) {
                Add-Case -Tag "wave_kcontig_pad_direct" `
                    -ALayout $layout.A -BLayout $layout.B `
                    -ASourcePad $pad -BSourcePad $pad -Align 0 `
                    -Groups $groupSet
                Add-Case -Tag "wave_kcontig_pad_lds" `
                    -ALoad 2 -BLoad 2 -ALayout $layout.A -BLayout $layout.B `
                    -ASourceLayout $layout.A -BSourceLayout $layout.B `
                    -ASourcePad $pad -BSourcePad $pad -Align 0 `
                    -Groups $groupSet
        }
    }

    # Native ByteAddressBuffer vector widths.
    foreach ($width in @(2, 4, 8)) {
        foreach ($load in @(2, 3)) {
            foreach ($type in $(if ($isQuick) { @(0) } else { @(0, 1, 2) })) {
                Add-Case -Tag "wave_vector_width" `
                    -ALoad $load -BLoad $load -AType $type -BType $type `
                    -VectorWidth $width -Align 0 -Groups $groupSet
            }
        }
    }

    # Padded LDS strides deliberately shift rows across memory banks.
    foreach ($pad in $(if ($isQuick) { @(0, 1, 8) } else { @(0, 1, 2, 4, 8, 16) })) {
        foreach ($load in @(2, 3)) {
            Add-Case -Tag "wave_lds_bank_a" -ALoad $load -BLoad $load `
                -ALdsPad $pad -Align 0 -Groups $groupSet
            Add-Case -Tag "wave_lds_bank_b" -ALoad $load -BLoad $load `
                -BLdsPad $pad -Align 0 -Groups $groupSet
            Add-Case -Tag "wave_lds_bank_ab" -ALoad $load -BLoad $load `
                -ALdsPad $pad -BLdsPad $pad -Align 0 -Groups $groupSet
        }
    }

    # Multiple live accumulator tiles and wave-to-tile ownership order.
    foreach ($tiles in $(if ($isQuick) { @(1, 2) } else { @(1, 2, 4) })) {
        foreach ($waves in @(1, 4)) {
            foreach ($order in @(0, 1)) {
                foreach ($epi in @(0, 2)) {
                    Add-Case -Tag "wave_multi_acc" -AccTiles $tiles `
                        -Waves $waves -Threads (32 * $waves) `
                        -TileOrder $order -Epilogue $epi -Align 0 `
                        -Groups $groupSet
                }
            }
        }
    }
    $multiShapes = if ($isQuick) {
        @([pscustomobject]@{ M=64; N=128 })
    } else {
        @(
        [pscustomobject]@{ M=32; N=32 },
        [pscustomobject]@{ M=64; N=128 },
        [pscustomobject]@{ M=128; N=128 }
        )
    }
    foreach ($shape in $multiShapes) {
        foreach ($tiles in $(if ($isQuick) { @(1, 2) } else { @(1, 2, 4) })) {
            foreach ($order in @(0, 1)) {
                foreach ($epi in @(0, 2)) {
                    Add-Case -Tag "threadgroup_multi_acc" -Scope 2 `
                        -M $shape.M -N $shape.N -KSteps 4 `
                        -Wave 32 -Waves 8 -Threads 256 `
                        -ALoad 2 -BLoad 2 -BLayout 1 -AccTiles $tiles `
                        -TileOrder $order -Epilogue $epi -Align 0 -Groups @(64)
                }
            }
        }

        foreach ($depth in $(if ($isQuick) { @(1, 4, 8, 16) } else { @(1, 4, 8, 16, 64) })) {
            foreach ($bLayout in @(0, 1)) {
                foreach ($epi in @(0, 2)) {
                    Add-Case -Tag "threadgroup_kdepth" -Scope 2 `
                        -M 64 -N 128 -KSteps $depth -Wave 32 -Waves 8 `
                        -Threads 256 -ALoad 2 -BLoad 2 -BLayout $bLayout `
                        -Epilogue $epi -Groups @(64)
                }
            }
        }
    }
}

# Threadgroup-scope support and performance surface. Unsupported shapes are
# useful results and are retained as PSO failures.
$tgShapes = if ($isQuick) {
    @(
        [pscustomobject]@{ M=16; N=16 },
        [pscustomobject]@{ M=64; N=128 },
        [pscustomobject]@{ M=128; N=256 }
    )
} else {
    @(
        [pscustomobject]@{ M=16; N=16 },
        [pscustomobject]@{ M=32; N=32 },
        [pscustomobject]@{ M=64; N=64 },
        [pscustomobject]@{ M=64; N=128 },
        [pscustomobject]@{ M=128; N=64 },
        [pscustomobject]@{ M=128; N=128 },
        [pscustomobject]@{ M=128; N=256 }
    )
}
foreach ($shape in $tgShapes) {
    foreach ($wave in $waveSizes) {
        foreach ($load in @(0, 2)) {
            foreach ($bLayout in @(0, 1)) {
                foreach ($epi in $epilogues) {
                    Add-Case -Tag "threadgroup" -Scope 2 -M $shape.M -N $shape.N `
                        -Wave $wave -Waves ([math]::Max(1, 256 / $wave)) `
                        -Threads 256 -ALoad $load -BLoad $load `
                        -BLayout $bLayout -Epilogue $epi `
                        -KSteps 4 -Groups $(if ($Profile -eq "exhaustive") { @(1,16,64) } else { @(64) })
                }
            }
        }
        Add-Case -Tag "threadgroup_a_lds_b_direct" -Scope 2 `
            -M $shape.M -N $shape.N -Wave $wave `
            -Waves ([math]::Max(1, 256 / $wave)) -Threads 256 `
            -ALoad 2 -BLoad 0 -BLayout 1 -KSteps 4 -Groups @(64)
        Add-Case -Tag "threadgroup_a_direct_b_lds" -Scope 2 `
            -M $shape.M -N $shape.N -Wave $wave `
            -Waves ([math]::Max(1, 256 / $wave)) -Threads 256 `
            -ALoad 0 -BLoad 2 -BLayout 1 -KSteps 4 -Groups @(64)
    }
}
foreach ($threads in @(32, 64, 128, 256)) {
    Add-Case -Tag "threadgroup_thread_range_native" -Scope 2 `
        -M 16 -N 16 -K 16 -KSteps 1 -Wave 32 `
        -Waves ([math]::Max(1, $threads / 32)) -Threads $threads `
        -ALoad 0 -BLoad 0 -Align 0 -Groups @(64)
    Add-Case -Tag "threadgroup_thread_range_large" -Scope 2 `
        -M 64 -N 128 -K 16 -KSteps 1 -Wave 32 `
        -Waves ([math]::Max(1, $threads / 32)) -Threads $threads `
        -ALoad 2 -BLoad 2 -BLayout 1 -Align 0 -Groups @(64)
}

if (-not $isQuick) {
    foreach ($shape in @(
        [pscustomobject]@{ M=128; N=64 },
        [pscustomobject]@{ M=128; N=256 }
    )) {
        foreach ($wave in @(32, 64)) {
            foreach ($depth in @(1, 4, 8, 16, 64)) {
                Add-Case -Tag "threadgroup_kdepth_a_lds_b_direct" -Scope 2 `
                    -M $shape.M -N $shape.N -Wave $wave `
                    -Waves ([math]::Max(1, 256 / $wave)) -Threads 256 `
                    -ALoad 2 -BLoad 0 -BLayout 1 -KSteps $depth -Groups @(64)
            }
        }
    }
}

# Thread-scope matrix-vector surface.
$threadShapes = if ($isQuick) {
    @([pscustomobject]@{ M=16; K=16 })
} else {
    @(
        [pscustomobject]@{ M=8; K=8 },
        [pscustomobject]@{ M=16; K=16 },
        [pscustomobject]@{ M=16; K=32 },
        [pscustomobject]@{ M=32; K=16 },
        [pscustomobject]@{ M=32; K=32 }
    )
}
foreach ($shape in $threadShapes) {
    foreach ($wave in $waveSizes) {
        foreach ($layout in @(0, 1)) {
            Add-Case -Tag "thread_matvec" -Scope 0 -M $shape.M -N 1 -K $shape.K `
                -Wave $wave -Waves 1 -Threads $wave -ALayout $layout `
                -KSteps 16 -Groups $(if ($Profile -eq "exhaustive") { @(1,64,256) } else { @(256) })
        }
    }
}

# Exhaustive adds the full wave-scope Cartesian product.
if ($Profile -eq "exhaustive") {
    foreach ($aLoad in @(0,1,2,3)) {
        foreach ($bLoad in @(0,1,2,3,4)) {
            foreach ($layout in $layouts) {
                foreach ($wave in @(16,32,64)) {
                    foreach ($waves in @(1,2,4,8)) {
                        foreach ($depth in @(1,4,16,64)) {
                            foreach ($epi in @(0,1,2)) {
                                Add-Case -Tag "wave_exhaustive" -ALoad $aLoad -BLoad $bLoad `
                                    -ALayout $layout.A -BLayout $layout.B `
                                    -Wave $wave -Waves $waves -Threads ($wave * $waves) `
                                    -KSteps $depth -Epilogue $epi -Groups $groupSet
                            }
                        }
                    }
                }
            }
        }
    }
}

if ($TagFilter -ne "") {
    $cases = @($cases | Where-Object { $_.Tag -like $TagFilter })
}
$cases = $cases | Sort-Object Scope,M,N,K,KSteps,Inner,Wave,Waves,Threads,ALoad,BLoad,ALayout,BLayout,CLayout,ASourceLayout,BSourceLayout,AType,BType,AOffset,BOffset,COffset,ASourcePad,BSourcePad,ALdsPad,BLdsPad,VectorWidth,AccTiles,TileOrder,Align,Epilogue -Unique

"case_id,tag,scope,m,n,k,k_steps,inner,wave,waves,threads,a_load,b_load,a_layout,b_layout,c_layout,a_source_layout,b_source_layout,a_type,b_type,a_offset,b_offset,c_offset,a_source_pad,b_source_pad,a_lds_pad,b_lds_pad,vector_width,acc_tiles,tile_order,align,epilogue,groups,status,gpu_us,tflops,mismatches,max_abs,max_rel,first_mismatch,first_actual,first_expected,vendor_id,device_id,linalg_tier,driver_version,timestamp_frequency,tg_support_flags,tg_min_threads,tg_max_threads,tg_preferred_threads,tg_group_size_valid,wave_support_flags,wave_native_shapes,wave_shape_supported,adapter_name,hr,error" |
    Set-Content -Encoding ascii $csvPath
Remove-Item $jsonlPath, $compileLog -ErrorAction SilentlyContinue

function Csv-Quote([object]$value) {
    $text = if ($null -eq $value) { "" } else { [string]$value }
    return '"' + $text.Replace('"', '""') + '"'
}

$index = 0
foreach ($case in $cases) {
    ++$index
    $id = "{0:D5}" -f $index
    $name = "$id-s$($case.Scope)-m$($case.M)n$($case.N)k$($case.K)-ks$($case.KSteps)-i$($case.Inner)-w$($case.Wave)x$($case.Waves)-l$($case.ALoad)$($case.BLoad)-xy$($case.ALayout)$($case.BLayout)-src$($case.ASourceLayout)$($case.BSourceLayout)-t$($case.AType)$($case.BType)-off$($case.AOffset)-$($case.BOffset)-$($case.COffset)-p$($case.ASourcePad)$($case.BSourcePad)-lp$($case.ALdsPad)$($case.BLdsPad)-v$($case.VectorWidth)-ac$($case.AccTiles)o$($case.TileOrder)-a$($case.Align)-e$($case.Epilogue)"
    $cso = Join-Path $shaderDir "$name.cso"
    $defines = @(
        "BENCH_SCOPE=$($case.Scope)", "BENCH_M=$($case.M)",
        "BENCH_N=$($case.N)", "BENCH_K=$($case.K)",
        "K_STEPS=$($case.KSteps)", "INNER_REPEATS=$($case.Inner)",
        "WAVE_SIZE=$($case.Wave)", "NUM_WAVES=$($case.Waves)",
        "GROUP_THREADS=$($case.Threads)", "A_LOAD=$($case.ALoad)",
        "B_LOAD=$($case.BLoad)", "A_LAYOUT=$($case.ALayout)",
        "B_LAYOUT=$($case.BLayout)", "C_LAYOUT=$($case.CLayout)",
        "A_SOURCE_LAYOUT=$($case.ASourceLayout)",
        "B_SOURCE_LAYOUT=$($case.BSourceLayout)",
        "A_SOURCE_TYPE=$($case.AType)", "B_SOURCE_TYPE=$($case.BType)",
        "A_BASE_OFFSET=$($case.AOffset)", "B_BASE_OFFSET=$($case.BOffset)",
        "C_BASE_OFFSET=$($case.COffset)",
        "A_SOURCE_PAD=$($case.ASourcePad)",
        "B_SOURCE_PAD=$($case.BSourcePad)",
        "A_LDS_PAD=$($case.ALdsPad)", "B_LDS_PAD=$($case.BLdsPad)",
        "LOAD_VECTOR_WIDTH=$($case.VectorWidth)",
        "ACC_TILES=$($case.AccTiles)", "TILE_ORDER=$($case.TileOrder)",
        "DESCRIPTOR_ALIGN=$($case.Align)", "EPILOGUE=$($case.Epilogue)"
    )
    $dxcArgs = @("-T","cs_6_10","-E","main","-Fo",$cso,
        "-I",$DxcInclude,"-enable-16bit-types","-O3","-Qstrip_debug","-Qstrip_reflect")
    foreach ($define in $defines) {
        $dxcArgs += @("-D", $define)
    }
    $dxcArgs += $shader
    $compileOutput = & $Dxc @dxcArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        $errorText = ($compileOutput -join " ") -replace '\s+', ' '
        if ([string]::IsNullOrWhiteSpace($errorText)) {
            $errorText = "DXC failed; see compile.log"
        }
        "[$name] $errorText" | Add-Content -Encoding utf8 $compileLog
        foreach ($groups in $case.Groups) {
            [pscustomobject]@{
                case_id=$id; tag=$case.Tag; scope=$case.Scope; m=$case.M;
                n=$case.N; k=$case.K; k_steps=$case.KSteps; inner=$case.Inner;
                wave=$case.Wave; waves=$case.Waves; threads=$case.Threads;
                a_load=$case.ALoad; b_load=$case.BLoad;
                a_layout=$case.ALayout; b_layout=$case.BLayout;
                c_layout=$case.CLayout; a_source_layout=$case.ASourceLayout;
                b_source_layout=$case.BSourceLayout; a_type=$case.AType;
                b_type=$case.BType; a_offset=$case.AOffset;
                b_offset=$case.BOffset; c_offset=$case.COffset;
                a_source_pad=$case.ASourcePad;
                b_source_pad=$case.BSourcePad; a_lds_pad=$case.ALdsPad;
                b_lds_pad=$case.BLdsPad; vector_width=$case.VectorWidth;
                acc_tiles=$case.AccTiles; tile_order=$case.TileOrder;
                align=$case.Align;
                epilogue=$case.Epilogue; groups=$groups;
                result=[pscustomobject]@{
                    status="compile_failed"; error=$errorText
                }
            } | ConvertTo-Json -Compress -Depth 4 |
                Add-Content -Encoding utf8 $jsonlPath
            $row = @($id,$case.Tag,$case.Scope,$case.M,$case.N,$case.K,
                $case.KSteps,$case.Inner,$case.Wave,$case.Waves,$case.Threads,
                $case.ALoad,$case.BLoad,$case.ALayout,$case.BLayout,$case.CLayout,
                $case.ASourceLayout,$case.BSourceLayout,
                $case.AType,$case.BType,$case.AOffset,$case.BOffset,
                $case.COffset,$case.ASourcePad,$case.BSourcePad,
                $case.ALdsPad,$case.BLdsPad,$case.VectorWidth,
                $case.AccTiles,$case.TileOrder,
                $case.Align,$case.Epilogue,$groups,
                "compile_failed") + (@("") * 23) + @($errorText)
            $row = $row | ForEach-Object { Csv-Quote $_ }
            ($row -join ",") | Add-Content -Encoding ascii $csvPath
        }
        continue
    }

    foreach ($groups in $case.Groups) {
        $bCast = [int]($case.BLoad -eq 4)
        $hostArgs = @($cso,$agilityBin,$SdkVersion,$Adapter,$case.Scope,
            $case.M,$case.N,$case.K,$case.KSteps,$case.Inner,$case.Wave,
            $case.Waves,$case.Threads,$groups,$TimedDispatches,$WarmupDispatches,
            $case.ALayout,$case.BLayout,$case.CLayout,
            $case.ASourceLayout,$case.BSourceLayout,$bCast,$case.AType,
            $case.BType,$case.AOffset,$case.BOffset,$case.COffset,
            $case.ASourcePad,$case.BSourcePad,
            $case.ALdsPad,$case.BLdsPad,$case.VectorWidth,
            $case.AccTiles,$case.TileOrder,0)
        $raw = & $hostExe @hostArgs 2>&1
        $jsonText = ($raw | Select-Object -Last 1)
        try {
            $result = $jsonText | ConvertFrom-Json
        } catch {
            $result = [pscustomobject]@{
                status="host_error"; error=($raw -join " "); hr="";
                gpu_us=""; tflops=""; mismatches=""; max_abs=""; max_rel="";
                first_mismatch=""; first_actual=""; first_expected=""
            }
        }
        [pscustomobject]@{
            case_id=$id; tag=$case.Tag; scope=$case.Scope; m=$case.M; n=$case.N;
            k=$case.K; k_steps=$case.KSteps; inner=$case.Inner; wave=$case.Wave;
            waves=$case.Waves; threads=$case.Threads; a_load=$case.ALoad;
            b_load=$case.BLoad; a_layout=$case.ALayout; b_layout=$case.BLayout;
            c_layout=$case.CLayout; a_source_layout=$case.ASourceLayout;
            b_source_layout=$case.BSourceLayout;
            a_type=$case.AType; b_type=$case.BType;
            a_offset=$case.AOffset; b_offset=$case.BOffset;
            c_offset=$case.COffset;
            a_source_pad=$case.ASourcePad; b_source_pad=$case.BSourcePad;
            a_lds_pad=$case.ALdsPad; b_lds_pad=$case.BLdsPad;
            vector_width=$case.VectorWidth; acc_tiles=$case.AccTiles;
            tile_order=$case.TileOrder;
            align=$case.Align; epilogue=$case.Epilogue; groups=$groups;
            result=$result
        } | ConvertTo-Json -Compress -Depth 4 | Add-Content -Encoding utf8 $jsonlPath
        $row = @($id,$case.Tag,$case.Scope,$case.M,$case.N,$case.K,
            $case.KSteps,$case.Inner,$case.Wave,$case.Waves,$case.Threads,
            $case.ALoad,$case.BLoad,$case.ALayout,$case.BLayout,$case.CLayout,
            $case.ASourceLayout,$case.BSourceLayout,
            $case.AType,$case.BType,$case.AOffset,$case.BOffset,
            $case.COffset,$case.ASourcePad,$case.BSourcePad,
            $case.ALdsPad,$case.BLdsPad,$case.VectorWidth,
            $case.AccTiles,$case.TileOrder,
            $case.Align,$case.Epilogue,$groups,
            $result.status,$result.gpu_us,$result.tflops,$result.mismatches,
            $result.max_abs,$result.max_rel,$result.first_mismatch,
            $result.first_actual,$result.first_expected,
            $result.vendor_id,$result.device_id,
            $result.linalg_tier,$result.driver_version,$result.timestamp_frequency,
            $result.tg_support_flags,$result.tg_min_threads,
            $result.tg_max_threads,$result.tg_preferred_threads,
            $result.tg_group_size_valid,$result.wave_support_flags,
            $result.wave_native_shapes,$result.wave_shape_supported,
            $result.adapter,
            $result.hr,$result.error) |
            ForEach-Object { Csv-Quote $_ }
        ($row -join ",") | Add-Content -Encoding ascii $csvPath
        Write-Host ("[{0}/{1}] {2} g={3}: {4} {5} TFLOP/s" -f
            $index,$cases.Count,$name,$groups,$result.status,$result.tflops)
    }
}

$summary = Import-Csv $csvPath | Group-Object status | Sort-Object Name |
    Select-Object Name, Count
$summary | Format-Table | Out-String | Set-Content -Encoding ascii (Join-Path $OutputDir "summary.txt")
$summary | Format-Table
$analyzer = Join-Path $root "analyze_linalg_bench.ps1"
if (Test-Path $analyzer) {
    & $analyzer -ResultsDir $OutputDir
}
Write-Host "Results: $OutputDir"
