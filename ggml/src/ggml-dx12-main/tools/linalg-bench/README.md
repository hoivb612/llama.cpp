# DX12 LinAlg characterization

This standalone suite maps the D3D12 LinAlg implementation without changing
production shader routing. It compiles each case separately, creates the PSO on
the selected adapter, compares the result with a CPU reference, and measures
the dispatch with GPU timestamp queries.

Threadgroup cases first query the exact type, shape, wave size, and threadgroup
size through `D3D12_FEATURE_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT`.
Unsupported configurations are recorded without executing the shader.
Wave cases likewise query F16 x F16 -> F32 support and require the requested
shape to be an integer multiple of a reported native multiply shape.

See [AMD-RUNBOOK.md](AMD-RUNBOOK.md) for the reproducible scan procedure used
to collect a comparable data set on another Windows machine.

The matrix includes:

- thread, wave, and threadgroup matrix scopes
- descriptor matrix loads and LDS scalar, vector, and prefetched staging
- row-major and column-major A, B, and output layouts
- direct row-major load plus transpose cast for attention K
- descriptor alignment promises from 0 through 128 bytes, including deliberate
  over-promises that can expose drivers which trust the hint
- actual A, B, and output byte offsets, including non-power-of-two alignments
- odd physical leading dimensions and native matrix-shape boundaries
- wave16, wave32, and wave64 with 1, 2, 4, and 8 waves per group
- K depth and repeated matrix use
- threadgroup K depth with both LDS-staged operands and the production-like
  LDS A plus direct column-major B path
- F16, F32, and BF16 sources staged to F16
- tight and padded K-contiguous/non-contiguous source records
- 2-, 4-, and 8-element global vector-load ownership
- padded LDS strides that exercise different bank and alignment patterns
- one, two, and four live accumulator tiles with alternate tile ownership
- direct descriptor store, LDS drain, and `GetCoordinate()` drain
- supported and unsupported threadgroup matrix shapes
- exact threadgroup capability ranges and 32-, 64-, 128-, and 256-thread cases

Compiler failures, PSO failures, numerical mismatches, and successful timings
are all retained. Unsupported combinations are useful capability data.
Every descriptor matrix record starts at a 128-byte boundary; matrix row or
column strides remain the independently varied values under test.

Run from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\ggml\src\ggml-dx12\tools\linalg-bench\run_linalg_bench.ps1 `
  -Profile full
```

Profiles:

- `quick`: smoke test of every major path
- `full`: orthogonal characterization matrix
- `extended-quick`: baseline smoke plus the additional padding, vector-width,
  LDS-stride, ownership, and accumulator tests
- `extended`: the full baseline plus the complete additional matrix
- `exhaustive`: full Cartesian wave-scope sweep plus the full matrix

Use `-TagFilter` to run one experiment family without rerunning the complete
profile. Wildcards use PowerShell matching rules:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\ggml\src\ggml-dx12\tools\linalg-bench\run_linalg_bench.ps1 `
  -Profile extended -TagFilter "threadgroup_multi_acc"
```

Results are written below `results\<machine>-<timestamp>\`:

- `results.csv`: flat analysis table
- `results.jsonl`: complete structured records
- `compile.log`: compiler rejection details
- `summary.txt`: status counts
- `report.md`: capability and performance summary
- `shaders\`: the exact DXIL used for each case

The selected adapter is the zero-based DXGI adapter index. For example, run
the same full matrix on a second GPU with:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\ggml\src\ggml-dx12\tools\linalg-bench\run_linalg_bench.ps1 `
  -Profile full -Adapter 1 -OutputDir C:\bench\linalg-amd
```

Generate or regenerate a report without rerunning the shaders:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\ggml\src\ggml-dx12\tools\linalg-bench\analyze_linalg_bench.ps1 `
  -ResultsDir C:\bench\linalg-nvidia
```

Inspect optimized production-style DXIL for matrix allocas introduced by
unrolling or matrix-array lowering:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\ggml\src\ggml-dx12\tools\linalg-bench\inspect_linalg_dxil.ps1
```

Check whether a DXC and Agility SDK pair exposes native LinAlg BF16 in both
the compiler and runtime capability API:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\ggml\src\ggml-dx12\tools\linalg-bench\check_linalg_bf16_toolchain.ps1
```

Compare common correct cases from two runs:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\ggml\src\ggml-dx12\tools\linalg-bench\analyze_linalg_bench.ps1 `
  -ResultsDir C:\bench\linalg-nvidia `
  -CompareDir C:\bench\linalg-amd `
  -OutputPath C:\bench\linalg-comparison.md
```

Load path values in the result files:

| value | path |
|---:|---|
| 0 | direct descriptor `Matrix::Load` |
| 1 | scalar global load to LDS |
| 2 | vector global load to LDS |
| 3 | vector register prefetch to double-buffered LDS |
| 4 | direct row-major load plus transpose cast (B only) |

Layout values are 0 for row-major and 1 for column-major. Source types are
0 for F16, 1 for F32, and 2 for BF16. Epilogues are 0 for direct descriptor
store, 1 for LDS store plus scalar drain, and 2 for `GetCoordinate()` drain.
Source and LDS padding values are elements added to the physical leading
dimension. `vector_width` is the number of consecutive source elements in
each per-thread load batch. `acc_tiles` is the number of live accumulator
matrices per wave or threadgroup, and `tile_order` selects owner-major or
tile-major buffer ordering. Threadgroup records also include the exact
capability query result and its minimum, maximum, and preferred thread counts.

Do not discard mismatches or rejected PSOs when comparing drivers. In
particular, direct threadgroup descriptor loads are intentionally retained:
they can create a PSO yet return incorrect results on some driver and shape
combinations.
