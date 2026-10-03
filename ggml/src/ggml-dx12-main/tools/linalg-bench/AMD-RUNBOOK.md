# AMD LinAlg characterization runbook

This procedure collects a DX12 LinAlg data set on an AMD Radeon RX 9070 XT
that can be compared directly with a run from another adapter or driver. The
suite is standalone: it does not change production shader routing or require a
model.

Keep the complete result directory. Compiler and PSO failures, numerical
mismatches, and successful timings are all useful capability data.

## 1. Use the same source revision

Record the commit before running:

```powershell
git rev-parse HEAD
git status --short
```

Run the NVIDIA and AMD scans from the same commit. Local files outside
`ggml\src\ggml-dx12\tools\linalg-bench` do not affect this suite, but record a
dirty worktree so the source state is unambiguous.

## 2. Prerequisites

Install or locate:

- Windows 11 with the intended AMD display driver
- Visual Studio 2022 with the Desktop development with C++ workload
- a recent DXC package with `dxc.exe` and the HLSL include directory
- a D3D12 Agility SDK package with `D3D12Core.dll` and headers
- PowerShell 5.1 or newer

The runner finds Visual Studio through `vswhere.exe`. DXC and Agility SDK paths
are explicit command-line parameters, so their versions do not need to match
the example paths.

Before a timed scan:

- use the same driver and power profile for every repetition
- close GPU-accelerated applications and overlays
- avoid display mode changes while the scan is running
- do not run another GPU benchmark concurrently
- let the adapter return to its normal idle temperature between comparative
  runs

## 3. Set machine paths

Open PowerShell in the repository root and set paths for this machine:

```powershell
$Dxc = "C:\dxc\bin\x64\dxc.exe"
$DxcInclude = "C:\dxc\inc\hlsl"
$AgilityRoot = "C:\AgilitySDK"
$SdkVersion = 721
$Runner = ".\ggml\src\ggml-dx12\tools\linalg-bench\run_linalg_bench.ps1"
$Analyzer = ".\ggml\src\ggml-dx12\tools\linalg-bench\analyze_linalg_bench.ps1"
$OutputRoot = "C:\bench\linalg-rx9070xt"
```

`AgilityRoot` may be either a NuGet-style package root containing
`build\native\bin\x64` and `build\native\include`, or a root with `bin\x64` and
`include`.

## 4. Identify the DXGI adapter index

The adapter parameter is the zero-based order returned by
`EnumAdapterByGpuPreference(..., HIGH_PERFORMANCE, ...)`. On an AMD-only
benchmark machine it is normally adapter 0.

Run the quick profile and inspect the adapter line in `report.md`:

```powershell
powershell -ExecutionPolicy Bypass -File $Runner `
  -Profile quick `
  -Dxc $Dxc -DxcInclude $DxcInclude `
  -AgilityRoot $AgilityRoot -SdkVersion $SdkVersion `
  -Adapter 0 -OutputDir "$OutputRoot\adapter-check"
```

If the report does not name the RX 9070 XT, repeat with `-Adapter 1`, then
continue with the index that selects the intended GPU.

Do not compare timings from the adapter-check run. It is only a capability and
configuration smoke test.

## 5. Run the complete scan

Use the `extended` profile for the primary cross-vendor data set:

```powershell
$Adapter = 0
$RunDir = "$OutputRoot\extended"

powershell -ExecutionPolicy Bypass -File $Runner `
  -Profile extended `
  -Dxc $Dxc -DxcInclude $DxcInclude `
  -AgilityRoot $AgilityRoot -SdkVersion $SdkVersion `
  -Adapter $Adapter -OutputDir $RunDir `
  -WarmupDispatches 8 -TimedDispatches 40
```

The extended profile includes the baseline matrix plus K-contiguity, physical
source padding, 2/4/8-element load batches, LDS stride patterns, accumulator
counts, ownership orders, and the K=128 depth point.

The scan compiles one shader per case. Compile or PSO rejection is expected for
unsupported scopes, shapes, wave sizes, or LDS requirements. Do not stop the
run because individual cases fail.

## 6. Repeat timing-sensitive families

The complete scan provides coverage. Repeat important families into separate
directories to distinguish stable performance from clock or driver variance:

```powershell
$Families = @(
  "wave_descriptor_offset*",
  "wave_odd_stride",
  "wave_shape_boundary",
  "wave_kdepth*",
  "wave_vector_width",
  "wave_kcontig_pad*",
  "wave_lds_bank*",
  "wave_multi_acc",
  "threadgroup_thread_range*",
  "threadgroup_kdepth*",
  "threadgroup_multi_acc"
)

foreach ($Family in $Families) {
  $SafeName = $Family.Replace("*", "all")
  powershell -ExecutionPolicy Bypass -File $Runner `
    -Profile extended -TagFilter $Family `
    -Dxc $Dxc -DxcInclude $DxcInclude `
    -AgilityRoot $AgilityRoot -SdkVersion $SdkVersion `
    -Adapter $Adapter -OutputDir "$OutputRoot\focused-$SafeName" `
    -WarmupDispatches 16 -TimedDispatches 80
}
```

Correctness gates every performance result. Treat a high-throughput mismatch
as a driver or shader correctness finding, not as an optimization candidate.
If a case changes between `ok` and `mismatch` across repetitions, retain every
run and flag it as nondeterministic.

## 7. Review the result

Each result directory contains:

- `results.csv`: flat table used by the analyzer
- `results.jsonl`: complete structured records
- `compile.log`: DXC rejection output
- `summary.txt`: status counts
- `report.md`: generated capability and performance report
- `shaders`: the exact DXIL supplied to the driver

Regenerate the report if needed:

```powershell
powershell -ExecutionPolicy Bypass -File $Analyzer `
  -ResultsDir $RunDir
```

Confirm that the report records:

- `AMD Radeon RX 9070 XT` as the adapter
- the intended vendor/device IDs and driver version
- a nonzero LinAlg tier
- the expected case count for the selected profile

Do not assume NVIDIA capability results apply to AMD. In particular, compare:

- accepted matrix scopes and wave sizes
- exact threadgroup shape support and the reported minimum, maximum, and
  preferred threadgroup sizes
- PSO creation versus numerical correctness
- direct descriptor loads versus LDS staging
- row-major and column-major layouts
- actual descriptor offsets and odd leading dimensions
- K=16, 64, 128, 256, and 1024 behavior
- source and LDS padding requirements
- F16, F32, and BF16 source behavior
- accumulator count and tile ownership sensitivity
- exact threadgroup capability ranges versus correct execution

Run the toolchain checks once on the AMD machine:

```powershell
powershell -ExecutionPolicy Bypass -File `
  "$Repo\ggml\src\ggml-dx12\tools\linalg-bench\inspect_linalg_dxil.ps1" `
  -Dxc $Dxc -DxcInclude $DxcInclude `
  -OutputDir "$OutputRoot\dxil-inspection"

powershell -ExecutionPolicy Bypass -File `
  "$Repo\ggml\src\ggml-dx12\tools\linalg-bench\check_linalg_bf16_toolchain.ps1" `
  -Dxc $Dxc -DxcInclude $DxcInclude `
  -AgilityInclude "$AgilityRoot\build\native\include"
```

Threadgroup cases reported as `unsupported` were rejected by the exact
`D3D12_FEATURE_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT` query and were not
executed. A driver accepting a PSO is not a substitute for this capability
result.

## 8. Compare with another adapter

Copy the complete NVIDIA result directory to the AMD machine, or copy the AMD
directory back to the analysis machine. Then generate a common-case report:

```powershell
powershell -ExecutionPolicy Bypass -File $Analyzer `
  -ResultsDir "C:\bench\linalg-rx9070xt\extended" `
  -CompareDir "C:\bench\linalg-rtx5070\extended" `
  -OutputPath "C:\bench\linalg-amd-vs-nvidia.md"
```

The comparison includes only matching cases that were correct in both runs.
Review the individual reports alongside it because capability failures and
mismatches are intentionally excluded from timing ratios.

## 9. Package the data

Record the source revision and machine details next to the result directory:

```powershell
@"
commit: $(git rev-parse HEAD)
machine: $env:COMPUTERNAME
adapter: AMD Radeon RX 9070 XT
driver: <fill from report.md>
dxc: $Dxc
agility_root: $AgilityRoot
agility_sdk_version: $SdkVersion
"@ | Set-Content "$OutputRoot\run-metadata.txt"

Compress-Archive -Path "$OutputRoot\*" `
  -DestinationPath "$OutputRoot.zip"
```

Return the archive without deleting failed shaders, mismatch records, or
compile logs. Those artifacts are required to reproduce driver-specific
behavior.
