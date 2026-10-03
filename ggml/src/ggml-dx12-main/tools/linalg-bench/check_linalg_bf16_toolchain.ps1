param(
    [string]$Dxc = "C:\dxc\1.10.2605.37\bin\x64\dxc.exe",
    [string]$DxcInclude = "C:\dxc\1.10.2605.37\inc\hlsl",
    [string]$AgilityInclude = "C:\AgilitySDK\1.721.3-preview\build\native\include",
    [switch]$RequireReady
)

$ErrorActionPreference = "Stop"
$linalgHeader = Join-Path $DxcInclude "dx\linalg.h"
$d3d12Header = Join-Path $AgilityInclude "d3d12.h"
foreach ($path in @($Dxc, $linalgHeader, $d3d12Header)) {
    if (-not (Test-Path $path)) {
        throw "Missing required path: $path"
    }
}

$dxcVersion = (& $Dxc --version 2>&1) -join " "
$linalgText = Get-Content $linalgHeader -Raw
$d3d12Text = Get-Content $d3d12Header -Raw
$compilerReady = $linalgText -match "ComponentType::BFloat16|__COMPONENT_TYPE\(BFloat16\)"
$runtimeReady = $d3d12Text -match "D3D12_LINEAR_ALGEBRA_DATATYPE_(BFloat16|BFLOAT16)"
$ready = $compilerReady -and $runtimeReady

$result = [pscustomobject]@{
    ready = $ready
    compiler_bfloat16 = $compilerReady
    runtime_bfloat16 = $runtimeReady
    dxc = $Dxc
    dxc_version = $dxcVersion
    linalg_header = $linalgHeader
    d3d12_header = $d3d12Header
}
$result | Format-List
$result | ConvertTo-Json -Compress

if ($RequireReady -and -not $ready) {
    throw "Native LinAlg BF16 requires matching compiler and runtime headers."
}
