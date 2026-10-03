param(
    [string]$Dxc = "C:\dxc\1.10.2605.37\bin\x64\dxc.exe",
    [string]$DxcInclude = "C:\dxc\1.10.2605.37\inc\hlsl",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path (Split-Path $PSScriptRoot)
$shaderDir = Join-Path $root "shaders"
if ($OutputDir -eq "") {
    $OutputDir = Join-Path $PSScriptRoot "results\dxil-inspection"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$cases = @(
    [pscustomobject]@{
        Name = "gemm-128x128-nv-w32"
        Source = "mul_mat_linalg_f16.hlsl"
        Defines = @(
            "WAVE_SIZE=32", "NATIVE_FP16=1", "LA_NWAVE=8", "LA_NT=4",
            "LA_MT=2", "LA_WN=2", "LA_BK=16", "LA_REG_EPILOGUE=1"
        )
    },
    [pscustomobject]@{
        Name = "attention-d64-qkt-w32"
        Source = "flash_attn_linalg.hlsl"
        Defines = @(
            "WAVE_SIZE=32", "NATIVE_FP16=1", "FA_QK_TRANSPOSED=1",
            "FA_D=64", "FA_BR=32", "FA_TPW=1"
        )
    },
    [pscustomobject]@{
        Name = "attention-d96-qkt-w32"
        Source = "flash_attn_linalg.hlsl"
        Defines = @(
            "WAVE_SIZE=32", "NATIVE_FP16=1", "FA_QK_TRANSPOSED=1",
            "FA_D=96", "FA_BR=32", "FA_TPW=1"
        )
    }
)
foreach ($threads in @(64, 128, 256)) {
    $cases += [pscustomobject]@{
        Name = "threadgroup-64x128-t$threads"
        Source = "mul_mat_linalg_tg_f16.hlsl"
        Defines = @(
            "NATIVE_FP16=1", "TG_BM=64", "TG_BN=128",
            "THREADS=$threads"
        )
    }
}

$results = foreach ($case in $cases) {
    $source = Join-Path $shaderDir $case.Source
    $dxil = Join-Path $OutputDir "$($case.Name).dxil"
    $listing = Join-Path $OutputDir "$($case.Name).ll"
    $args = @(
        "-T", "cs_6_10", "-E", "main", "-Fo", $dxil, "-Fc", $listing,
        "-I", $shaderDir, "-I", $DxcInclude, "-enable-16bit-types", "-O3"
    )
    foreach ($define in $case.Defines) {
        $args += @("-D", $define)
    }
    $args += $source
    & $Dxc @args
    if ($LASTEXITCODE -ne 0) {
        throw "DXC failed for $($case.Name)"
    }

    $text = Get-Content $listing -Raw
    [pscustomobject]@{
        shader = $case.Name
        alloca_count = ([regex]::Matches($text, "\balloca\b")).Count
        linalg_reference_count = ([regex]::Matches(
            $text, "linAlg", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
        listing = $listing
    }
}

$csv = Join-Path $OutputDir "dxil-inspection.csv"
$results | Export-Csv -NoTypeInformation -Encoding ascii $csv
$results | Format-Table | Out-String |
    Set-Content -Encoding ascii (Join-Path $OutputDir "dxil-inspection.txt")
$results | Format-Table

$failures = @($results | Where-Object alloca_count -ne 0)
if ($failures.Count -ne 0) {
    throw "Optimized LinAlg DXIL contains allocas; see $csv"
}
Write-Host "No allocas found in optimized LinAlg DXIL."
