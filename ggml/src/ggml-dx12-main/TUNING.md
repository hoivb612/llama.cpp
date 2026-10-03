# ggml-dx12 cross-vendor tuning guide

How to measure, gate and validate a performance change in the DX12 backend
across AMD, NVIDIA and Intel parts.

Read [GOTCHAS.md](GOTCHAS.md) first - it lists the correctness traps that
will silently corrupt a "win" if you step on them.

Section numbers are not unique. This file is the union of two long-running
lines of work (the mainline backend and the LinAlg branch) that numbered
their sections independently; the merge kept both rather than renumbering
~700 lines of prose. Search by title, not by number.

---

## 1. Architecture detection

The dispatcher classifies every device into an `dx12_arch_family` at init
(`ggml-dx12.cpp`, ~L498):

| family | covers |
| --- | --- |
| `DX12_ARCH_NV_LEGACY` | pre-Pascal NVIDIA, no dp4a |
| `DX12_ARCH_NV_PASCAL_PLUS` | Pascal and newer |
| `DX12_ARCH_AMD_WAVE64` | GCN, Vega, CDNA |
| `DX12_ARCH_AMD_RDNA` | RDNA1-4, wave32 consumer |
| `DX12_ARCH_INTEL_UHD` | Gen9, Xe-LP - wave8 integrated |
| `DX12_ARCH_INTEL_XE_HPG_PLUS` | Arc A/B, Xe2, Xe3 - wave >= 16 |
| `DX12_ARCH_QUALCOMM`, `DX12_ARCH_APPLE`, `DX12_ARCH_MICROSOFT_WARP`, `DX12_ARCH_OTHER` | everything else |

AMD is refined further into a `dx12_arch_subfamily` (GCN / CDNA / RDNA1_2 /
RDNA3_X / RDNA4_PLUS) via the `AMD_DID_TABLE` DeviceId ranges.

**Why a DeviceId table:** D3D12 exposes no capability that separates RDNA1/2
from RDNA3+ on Windows. The SM 6.9 WaveMMA tier was a Microsoft preview that
was deprecated and never shipped, so genuine WMMA32 hardware advertises no
matrix tier. The DeviceId is the only authoritative signal. When a new AMD
chip ships, add its ID to the table - the source is `amdgpu.ids` in libdrm.

Prefer `dx12_subarch_is_rdna3_plus()` over testing a single sub-family, so
tuning validated on RDNA3 automatically applies to later parts.

---

## 2. Measurement methodology

Every performance claim in this backend must come from an A/B against a
**same-session** baseline. Numbers from different days are not comparable -
driver state, clocks and thermals move.

```powershell
# ALWAYS clear stale gates first; they persist across a shell.
Remove-Item Env:DX12_* -EA 0

.\build-linalg\bin\llama-bench.exe -hf <repo>:<quant> `
    -p 2048 -ngl 99 -fa on -r 4 --delay 2 -dev DX120
```

Rules that have repeatedly caught us out:

- **Fresh process per measurement.** Each `llama-bench` invocation is a new
  process, so environment variables must be re-set inside every loop
  iteration - setting them once before a loop does nothing for later runs.
- **`-r 4` minimum**, and compare against the reported standard deviation.
  Anything inside 1.5 sigma is noise. Run-to-run spread on a warm dGPU is
  around 0.5%, so a "2% win" measured once is not a win.
- **Rebuild the agility stage.** Agility staging is now a POST_BUILD step of
  `ggml-dx12` (the old `ggml-dx12-agility-stage` target is gone), so building
  `ggml-dx12` restores the `D3D12\` folder.
- **Check the build actually succeeded.** Grep for
  `FAILED|error C|error X|ninja: build stopped`. A failed shader compile
  leaves the previous blob in place and looks like a no-op result.
- Piping an exe straight into `Select-String` can drop output when several
  exes run in one script. Redirect to a temp file, then match on the file.

### Profiling

```powershell
$env:DX12_PROFILE = "1"
$env:DX12_PROFILE_PROMPT = "1"
```

Prints per-dispatch `s0=<ggml type> fl=<flags> K/N/M grp=` lines. Use it to
find which op dominates before optimising anything - the backend has enough
shaders that intuition is usually wrong.

`grp=1` on a large output is the signature of a missing dispatch-grid entry
(see section 5).

Other diagnostics:

| variable | effect |
| --- | --- |
| `DX12_DEBUG=1` | enable the D3D12 debug layer |
| `DX12_DEBUG_GBV=1` | GPU-based validation (very slow, catches OOB) |
| `DX12_DRED=1` | DRED breadcrumbs, for triaging a TDR / `DEVICE_HUNG` |
| `DX12_LINALG_CAPS=1` | dump the detected LinAlg matrix capabilities |
| `DX12_LINALG_TG_STRICT_CAPS=1` | require the exact advertised threadgroup tuple; disables the validated NVIDIA fallback |
| `DX12_PHASE_PROFILE=1` | per-phase timing (generation graphs only) |
| `DX12_PHASE_PROFILE=2` | per-phase timing, including prompt/vision graphs |
| `DX12_PSO_TRACE=1` | per-PSO creation cost, and the prewarm summary |
| `DX12_NO_PSO_PREWARM=1` | disable background PSO prewarm |
| `DX12_LOG_UNSUPPORTED_OPS=1` | show ops falling back to CPU |
| `DX12_VISIBLE_DEVICES=<i>` | restrict adapter enumeration |
| `DX12_WAVE_BLOB=16\|32\|64` | force a wave-width blob variant |

### Measuring power, not just time

On a power-limited part (any iGPU) a kernel can be at peak achieved
bandwidth and still be wasteful: redundant loads that hit L1 cost watts but
never appear in a GB/s figure, and the watts they burn come out of the clock
budget. When DX12 and Vulkan move the same bytes at the same rate but only
one of them slows down as the part heats, the difference is energy, and it
is directly measurable.

Windows exposes Intel RAPL through the `Energy Meter` counter set:

| instance | rail |
| --- | --- |
| `rapl_package0_pkg` | whole package |
| `rapl_package0_pp0` | CPU cores |
| `rapl_package0_pp1` | **iGPU** |
| `rapl_package0_dram` | DRAM |

```powershell
(Get-Counter "\Energy Meter(*)\Power").CounterSamples |
    ForEach-Object { "$($_.InstanceName) = $($_.CookedValue)" }   # mW
```

Sample it in a loop while a benchmark runs and divide by t/s to get joules
per token. Watch the GPU and DRAM rails *together*: a change that raises
DRAM power while lowering GPU power is converting wasted core energy into
real memory traffic, which is what a good fix looks like. See section 7g for
a case where this found a ~9% win that the bandwidth numbers said was not
there.

Remote Desktop note: the same adapter can enumerate twice, and the duplicate
fails `CreateCommandQueue` with `0x887A0005`. The
`ggml-dx12: Skipping ... CreateCommandQueue failed` line is benign as long as
a `Device 0:` line also appears; check `Backend 1/2: DX120` in
`test-backend-ops` output to confirm the GPU actually ran the tests.

---

### Orphaned GPU processes will silently corrupt every number

Piping a long-running benchmark into anything that short-circuits the
pipeline - `Select-Object -First N`, `Select-String ... | Select-Object`, and
similar - returns to the prompt while the process keeps running and keeps
using the GPU. Two orphaned `test-backend-ops` runs made DX12 llama-bench
decay 3280 -> 2587 -> 1567 on Qwen3-4B across successive invocations while
Vulkan, measured in between, looked fine. The results are plausible, not
obviously broken, and they poisoned an entire comparison before anyone
noticed.

Redirect long runs to a file (`> out.txt 2>&1`) and parse afterwards, and
check `Get-Process | Where-Object ProcessName -match 'llama|test-backend'`
is empty before trusting a measurement.

## 2.1 Runtime and offline autotuning

The runtime tuner is intentionally narrow. Cache schema v12 measures only
device-level Q4_K/Q5_K DP4A thread counts and separate F16/BF16/F32 matvec
K crossovers. It uses legal layouts, production dispatch geometry, route
gates, interleaved measurements, and median selection.

Cache files are per physical adapter and include driver, shader build stamp,
wave blob, and adapter LUID:

```text
%LOCALAPPDATA%\.ggml_dx12_tune_<VID>_<DID>_<LUIDHI>_<LUIDLO>.txt
```

| variable | effect |
| --- | --- |
| `DX12_TUNE_REFRESH=1` | ignore a valid cache and regenerate it |
| `DX12_NO_AUTOTUNE=1` | load a valid cache, but do not benchmark a miss |
| `DX12_TUNE_FORCE_Q4K_DP4A_32=0\|1` | override Q4_K device choice |
| `DX12_TUNE_FORCE_Q5K_DP4A_32=0\|1` | override Q5_K device choice |
| `DX12_TUNE_FORCE_F16_MR_K_THRESH=N` | override F16 crossover |
| `DX12_TUNE_FORCE_BF16_MR_K_THRESH=N` | override BF16 crossover |
| `DX12_TUNE_FORCE_F32_MR_K_THRESH=N` | override F32 crossover |

Use `llama-mmv-tune` for model-shape-dependent choices:

```powershell
.\build-linalg\bin\llama-mmv-tune.exe sweep `
    --in tuples.json -dev DX120 --preset linalg-groups `
    --changed-only --reps 20 --out linalg-groups.json
```

Presets cover Q4_K/Q5_K DP4A, Q5_K M thresholds, float-family matvec,
LinAlg group/K gates, MMQ M/K/N gates, and MoE token thresholds. See
`tools/llama-mmv-tune/README.md`.

Flash attention and standalone norms are not MMV operations. Tune them with
fresh-process `llama-bench` A/B runs:

| family | vary | required workload coverage |
| --- | --- | --- |
| FA tile shape | `DX12_FA_TILED_BR=8\|16` | head dimensions 64-256; prompt 1, 16, 64, 512 |
| FA tiled gate | `DX12_FA_TILED_MINQ`, `DX12_FA_TILED_MIND` | decode and prefill; short and long KV |
| FA split-KV | `DX12_FA_MIN_KV`, `DX12_FA_SPLIT_GROUPS` | KV 512, 2048, 8192+ |
| wide norm | `DX12_NORM_WIDE_1024`, `DX12_NORM_WIDE_1024_MIN` | widths around 1024/2048 and model-native widths |
| fused wide RMS | `DX12_RMS_NORM_MUL_1024`, `DX12_RMS_NORM_MUL_1024_MIN` | decode and prefill |

NVIDIA limits the Q4_K gate/up+SwiGLU RMS fold to K<=1024. At larger K,
recomputing the RMS reduction in every output-row group costs more than the
standalone RMS dispatch. Other vendors retain the previous unrestricted
behavior. `DX12_Q4K_GLU_FOLD_K_MAX` overrides the limit.

NVIDIA Q6_K DP4A matvec uses scalar reconstructed loads. The wider
reconstructed loads remain active on other vendors, but reduced RTX 5070
Qwen3-0.6B output-projection throughput from about 489 to 391 GB/s.
`DX12_NV_Q6K_WIDE_LOADS=1` restores the wider form for A/B testing.

Do not turn correctness gates into tuner candidates. Numerical-drift,
alignment, TDR, and known-bad driver gates remain static.

## 3. Validation gates

A change is not landable until both pass:

1. **`test-backend-ops -b DX120`** - the full suite, not a filtered subset.
   A filtered run will not catch a broadcast or permuted-shape regression in
   an op you did not think you touched.
2. **Perplexity against CPU.** Run `llama-perplexity` on the same model with
   `-dev DX120` and with `-ngl 0`, and compare. A few thousandths is
   expected from fp16 accumulation; a few tenths is a bug.

`test-backend-ops` alone is not sufficient - it uses random data, which
misses errors that only appear on real weight distributions. Perplexity
alone is not sufficient either - it is remarkably tolerant of a wrong result
on a rare shape.

---

## 4. The flag-number namespace

Blob `case` numbers in `select_mul_mat_blob` and friends are a **single flat
namespace** shared by every gate in `ggml-dx12.cpp`.

- A silent collision routes a gate to the wrong shader. It usually still
  produces plausible-looking output.
- Before adding a flag, enumerate what is taken:
  `Select-String -Path ggml-dx12.cpp -Pattern 'case (\d+):'`
- When merging two branches that both added flags, **re-check for
  collisions after the merge** - git will not flag them. The LinAlg GEMM
  gates own MUL_MAT flags 107-126 and 130-156, and derive the tile shape
  from `(flags - 107) % 4` / `(flags - 130) % 4`, so those ranges must stay
  contiguous; flags added on other branches inside them get renumbered
  above 156 when merged.

`use_dp4a_matvec` and `is_matvec_dispatch` leak across gates. Any new
MUL_MAT / MUL_MAT_ID gate that overrides `key.flags` must also clear them,
or a later dispatch inherits the wrong path.

---

## 5. Adding a quant type: five synchronised edits

Miss any one of these and the type either fails to build or, worse,
silently produces wrong answers on a subset of shapes.

1. `LA_FETCH_B` plus the `LA_B_PT` / `LA_B_STEP` guard in
   `shaders/mul_mat_linalg_f16.hlsl`
2. The CMake `LA_KQ` (dense) and `LA_MQ` (MoE) quant loops
3. The blob `case`s in the five `select_*` lambdas
4. The LinAlg dispatch group-count flag range
5. **The flat-quant dispatch grid type list** (`ggml-dx12.cpp`, ~L10240) -
   the `fl=0` path that computes
   `total_groups = ceil(ne0*ne1*ne2*ne3/256)`

Omitting (5) gives `grp=1` and wrong answers on broadcast and permuted
shapes, while every contiguous shape passes. This is the single most
expensive mistake available in this codebase.

Adding one `MMID_<TYPE>` block to `shaders/quant_dequant.hlsli` yields
mul_mat, both mul_mat_id variants **and** get_rows, since
`get_rows_quant.hlsli` is driven by the same `mmid_dequant()` contract.

---

## 6. Known per-vendor pitfalls

**All vendors - root SRVs are not bounds checked.** Buffers bound with
`SetComputeRootShaderResourceView` get no bounds clamping, unlike
descriptor-table SRVs. DXC speculates a `Load` guarded by an `if` *or by an
early `return`*, so a trailing funnel-shift word can walk past the
allocation and fault the device. Make the address unconditionally in
bounds instead of guarding the load. See GOTCHAS.md.

**NVIDIA - load alignment.** Misaligned `ByteAddressBuffer` loads are far
more expensive than on AMD, and in some cases incorrect. Keep quant row
offsets dword-aligned.

**AMD - wave32 vs wave64 blobs.** RDNA runs wave32 by default but several
shaders have wave64 variants that win on dense GEMM. The blob split is a
real fork in the shader tree, not a runtime switch; both must be kept
building. `DX12_WAVE_BLOB` forces one for A/B purposes.

**AMD RDNA4 - LinAlg driver defects.** See `AMD_LinAlg_Driver_Bug.md`. A
hoisted, arithmetically-equivalent nibble fetch in the MXFP4 path fails
deterministically at one specific shape. Where a comment says a fetch
sequence must not be simplified, believe it - the "obvious" simplification
has already been tried and measured.

**Intel UHD - wave8.** Tile sizes tuned for wave32/64 are badly wrong here;
the `_tiled_64` gates exist specifically for this family.

### RDNA1/2 UMA defaults

The RDNA1/2 UMA path uses several defaults established on an RDNA2 iGPU.
Discrete RDNA1/2 devices retain their previous routes. Each new default
retains an environment override for A/B testing:

| route | default | override | measured impact |
| --- | --- | --- | --- |
| Large-K Q8_0 decode | skip `mr256v`, then use wave64 rows2 when fusion lands or DP4A otherwise | `DX12_Q8_MR256V=1` restores `mr256v` | Qwen3 +44%, Phi-3 +42% |
| D=64 prefill FA | BR=32 wide shader | `DX12_FA_PF_WIDE=0` | pp512 +29%, pp2048 +56%, pp6144 +71% |
| Prefill FA tile values | use FP16 Q/K LDS tiles | `DX12_FA_PF_FP16=0` | SmolLM2 +4.7-5.2%, Phi-3 +6.5%, Qwen3.5 +2.8% |
| Q5_0 prefill GEMM | 128x64 register-blocked MMQ with paired-lane weight sharing | `DX12_Q50_MMQ=0`; `DX12_Q50_MMQ_WAVE_SHARE=0` | SmolLM2 Q4_K_M pp6144 +17.7%; decode route unchanged |
| Intel D=64 prefill FA | mask prescan blob | `DX12_FA_PF_PRESCAN=0` | pp6144 +13% granite, +46-52% Smol |
| Intel D=96/128 prefill FA | mask prescan blob | `DX12_FA_PF_PRESCAN=0` | pp4096 +5.8% Phi-3, +5.3% Qwen3-4B |
| NVIDIA Pascal+ D=96/128 prefill FA | mask prescan blob | `DX12_FA_PF_PRESCAN=0` | pp6144 +5.8% Qwen3-4B, +6.3% Phi-3, +4.5% Qwen3-0.6B |
| Quantized MoE GEMM | tall BM=128 tile when >=128 pairs/expert | `DX12_MOE_GEMM_TALL=0` | granite Q4_K_M pp512 +17%, pp6144 +15% |
| Quantized KV cache | q8_0 + legacy-quant `SET_ROWS` enabled | `DX12_SET_ROWS_Q8_0=0`, `DX12_SET_ROWS_LEGACY_QUANT=0` | `-ctk/-ctv q8_0`/`q4_0` work; SET_ROWS 135/135 |
| Small-query FA | use the tiled/prefill shader from two queries | `DX12_FA_TILED_MINQ=64` restores the old threshold | SmolLM2 pp32 +13%, Phi-3 pp32 +8% |
| Small-M Q8_0 GEMM | use MMQ for M=2-16, retain the old path at M=17-63 | explicit `DX12_MMQ_MIN_M` overrides the split gate | SmolLM2 pp4/8/16 +94%/+89%/+86% |
| Q8_0 MMID | 64-thread DP4A workgroup | `DX12_MOE_Q8_G64=0` | Granite pp512 +34% |
| F16 decode | wave64 matvec | `DX12_F16_WAVE64=0` | SmolLM2 decode +28% |
| K<=1024 Q8_0 SwiGLU | retain the DP4A GLU kernel instead of folding RMS norm into a scalar kernel | `DX12_Q80_GLU_RMS_FOLD=1` restores the fold | Smol models decode +8% |

Strix Point UMA (device ID `0x150E`, including Radeon 880M/890M) also uses
the large-K Q8_0 and D=64 prefill FA defaults above. On Radeon 880M, skipping
`mr256v` improved Qwen3 Q8_0 decode by 22% and Phi-3 Q8_0 decode by 16%;
the wide FA shader improved F16 pp512 by 12%, pp2048 by 25%, and pp6144 by
54%. The other RDNA1/2 UMA defaults remain unchanged for this device.

### RDNA4 Q8_0 block-vectorized SwiGLU

RDNA4 wave64 uses a block-vectorized variant of the fused Q8_0 DP4A
gate/up/SwiGLU kernel. One lane consumes a complete 32-value Q8 block using
two 128-bit loads per weight matrix, instead of assigning eight lanes four
values each through scalar misaligned loads. `DX12_Q80_GLU_BLOCK64=0`
restores the scalar flag-62 shader; `=1` forces the vectorized shader on
other wave64 devices for testing.

On RX 9070 XT Q8_0 decode, the vectorized route measured Falcon-H1-7B
59.40 -> 60.14 t/s, Qwen3-4B 118.67 -> 120.15 t/s, SmolLM2
1004.12 -> 1045.10 t/s, and SmolVLM2 1013.18 -> 1044.69 t/s. Phi-3 was
neutral at 129.74 -> 129.73 t/s. The Falcon kernel itself improved from
about 578 to 595 GB/s.

### AMD projection fusion defaults

BF16 gate/up/SwiGLU fusion is default-on for AMD RDNA1/2 and RDNA4. It
improved Qwen3-0.6B decode by about 1% on RDNA4 and 3-5% on RDNA2. RDNA3
remains unchanged pending measurements. `DX12_MMV_GLU_FUSION_BF16=0`
disables it.

Standalone single-row RMS_NORM, NORM, and L2_NORM use 1024 threads at widths
of at least 2048 on RDNA1/2 and RDNA4. At width 8192 this reduced kernel time
from 12.44 to 5.22 us for NORM and from 9.68 to 4.37 us for RMS_NORM on
RDNA4. RDNA1/2 measured 14.17 to 5.72 us and 10.45 to 4.29 us respectively.
Model-level Phi-3 decode was neutral because its hot norms are fused.
`DX12_NORM_WIDE_1024=0` disables the route and
`DX12_NORM_WIDE_1024_MIN` changes the minimum width.

Q/K/V resource partitioning is enabled for F16, BF16, and Q8_0 on RDNA4.
The measured Qwen3-0.6B decode gains were 1.3-1.4%, 3.6%, and 2.8%
respectively. RDNA1/2 remains disabled after neutral F16/BF16 results and
a 5.6% Q8_0 regression. `DX12_QKV_RESOURCE_PARTITION=0` disables the route.

Q2_K prefill uses the register-blocked Q8_1 MMQ on AMD RDNA when N and M
reach the normal MMQ thresholds. TinyLlama Q2_K improved from 352.49 to
383.39 t/s at pp512 on RDNA4 and from 5.98 to 6.64 t/s at pp128 on RDNA2.
Decode keeps the existing matvec route. `DX12_Q2K_MMQ=0` disables the MMQ.

Wide-head F16 flash attention uses tiled shaders for D/D_v shapes 192/128,
192/192, 256/256, 320/256, 512/512, and 576/512 on RDNA4. At 512 queries
and 4096 KV tokens, the six shapes improved by 2.2x to 4.7x. The route
requires at least 64 queries and no attention sinks. RDNA1/2 retains the
generic path because its wide tiled route did not pass the full mask and
sink correctness matrix. `DX12_FA_TILED_LARGE=0` disables it,
`DX12_FA_TILED_LARGE=1` permits explicit testing on other devices, and
`DX12_FA_TILED_LARGE_MINQ` changes the query threshold.

---

## 7. Rejected experiments

Recorded so they are not retried. Full context and measurements are in
`docs/linalg_how_to_setup.md`.

Tile and layout: 128x128 / 256x64 / 64x64 mmq tiles; 256x64 and 64x32
LinAlg tiles; `MMQ_KSTEP` 2 and 4; `LA_KT=2`; strided output mapping;
super-tile swizzle; Vulkan's `BM=128,BN=128,BK_STEP=4`.

Memory and data flow: single-buffered LDS; A-side fragment hoist; direct
`MatAcc::Store`; an f32->f16 activation pre-pass; a separate MoE row-map
prepare pass.

Dispatch and routing: any single global `linalg_mm_target`; routing
576-wide ops to mmq via `DX12_MMQ_MIN_N`; larger `-ub`;
`DX12_FA_SPLIT_GROUPS` other than 512.

Flash attention: prefill-shaped FA on RDNA4; the GQA fold for wave-matrix
FA; `FA_BC=64` / `FA_BR=32`.

Other: wave32 blobs for the LinAlg GEMM; the int8 wave-matrix GEMM; the
uniform-nibble-half batched MXFP4 fetch; replacing the MXFP4 kvalues table
with direct float bit-construction (DXC already handles the constant array
well - measured ~1% slower).

Barriers: narrowing barrier scope. Barriers cost ~23% of a multimodal
prefill (332 per graph at ~30us each), but turning every global barrier
into a scoped one is worth only +1.5% - a UAV barrier drains the pipeline
whether it names one resource or all of them. Enhanced Barriers are
already in use with the tightest scope the spec permits (`Offset` must be
0 and `Size` 0 or `UINT64_MAX`, so byte-range buffer barriers do not
exist). 82% of the barriers are `raw-src`, i.e. genuine read-after-write
dependencies. Only reducing node count can reduce them further.

Q8_0 `mr256` (fl=44) on Intel Xe-HPG+ (wave16, Arc B390): -30% on
SmolLM2-135M (K=576, tg128 322 -> 224 t/s) and -22% on SmolLM2-360M.
The 256-thread scalar group trades away dp4a, which is too strong on this
family to give up for occupancy. The `wave_size == 32` gate stays.
`DX12_Q8_MR256=1` now reaches any wave size if it needs re-testing.

Q6_K `mr4` (fl=107) on Intel Xe-HPG+ (wave16, Arc B390): 4.38 vs 4.44 t/s
on Qwen3.8-27B Q6_K tg256. The 2-row shader re-reads the whole Q8_1
activation vector once per group, so at N=17408 the activations look like a
third of the traffic on paper - but they are already L2-resident, so halving
the re-reads buys nothing while four unrolled Q6_K decodes plus four
accumulators cost enough registers to lose occupancy. (Before the `load4`
fix below it was much worse, 3.91 vs 4.13; fixing the loads narrowed the gap
but did not close it.) Same conclusion as Q4_K `mr4` (fl=46), which is
likewise gated off for small waves. Kept as `DX12_Q6K_DP4A_MR4=1` for
large-wave vendors.

---

## 7g. Qwen3.8-27B Q6_K decode: misaligned loads cost power, and power costs clocks

Profiled because DX12 trailed Vulkan on `-p 0 -n 128` (4.07 vs 4.67 t/s).
Almost every intuition about where the time goes was wrong, and the fix was
not where the profiler pointed, so the whole chain is recorded here.

The model is a hybrid: 64 layers, 48 `GATED_DELTA_NET` (recurrent) and 16
full-attention (D=256, nh=24/4). Per token: 1603 dispatches, ~216 ms,
**94% MUL_MAT**, and GPU idle of **0.3%** - dispatch and barrier overhead
are not a factor, despite the dispatch count.

The trap: the big matvecs already ran at **104-111 GB/s**, ~85% of
theoretical LPDDR5x peak on this part, matching Vulkan's *end-to-end* rate.
By every bandwidth measure the kernel was done. It was not.

### The signature

The gap only appears as the run gets longer:

| | 64 tokens | 256 tokens |
|---|---|---|
| DX12 | 4.71 | 4.10 (-13%) |
| Vulkan | 4.82 | 4.61 (-4%) |

Late in a run *every* matvec loses ~19% of achieved bandwidth uniformly
(~110 -> ~90 GB/s). Uniformity across unrelated kernels rules out any single
shader and points at clocks.

Falsified, each by measurement: token index / context growth (`-n 64,256,64`
gives 4.53 / 3.92 / **3.96** - the trailing short run does not recover);
process state (a *fresh* process on a hot GPU also starts slow, 4.10 vs
4.53-4.71 cold); CPU contention (DX12 uses *less* CPU); leaks (working set
and handles flat over 80 s); footprint (Vulkan identical at 42.2 GB);
paging (`--no-mmap` halves the footprint and changes nothing);
`DX12_MMV_GROUP_SIZE` (ABBA showed no effect).

### Measuring it instead of guessing

Windows exposes Intel RAPL through the `Energy Meter` counter set, which
settles this directly - `rapl_package0_pp1` is the iGPU rail:

```powershell
(Get-Counter "\Energy Meter(*)\Power").CounterSamples |
    Where-Object { $_.InstanceName -eq 'rapl_package0_pp1' }
```

Sampled across a tg256 run (mW, averaged over the run):

| | t/s | GPU W | DRAM W | J/token (GPU) |
|---|---|---|---|---|
| DX12 before | 3.94-4.10 | 17.3-18.5 | 2.2-2.3 | 4.38-4.51 |
| Vulkan | 4.72 | 14.4-14.7 | 3.0 | 3.05-3.12 |

DX12 was drawing **~20% more GPU power to go 15% slower** - 44% worse
energy per token. Note the DRAM rail moves the *other* way: Vulkan burns
more there because it is genuinely pushing more DRAM traffic. So the excess
was in the GPU core domain (ALU, L1/L2, issue), not in memory.

### The cause

`mul_mat_vec_q6k_dp4a.hlsl` fetched its 16 bytes of `ql` and 16 of `qh` as
four independent `load_u32_u` calls each. Q6_K blocks are 210 bytes, so
`block_off` is 2-byte misaligned for every other block, and on that path
each call issued **two** loads - 8 loads per 16-byte fetch, re-reading every
boundary word twice. The redundant halves hit L1, so they never showed up as
DRAM bandwidth; they showed up as watts.

The fix is the `load4_u_q6k` helper that
`mul_mat_vec_q6k_mr_blocked.hlsl` already had: one `Load4` when aligned,
five word loads when not. That is 8 -> 5 loads on the misaligned path and
8 -> 1 instruction when aligned.

| | t/s | GPU W | DRAM W | J/token (GPU) |
|---|---|---|---|---|
| DX12 before | 3.94-4.10 | 17.3-18.5 | 2.2-2.3 | 4.38-4.51 |
| DX12 after | **4.38-4.48** | **14.9-15.8** | 2.8 | **3.40-3.56** |
| Vulkan | 4.72-4.77 | 14.4-16.0 | 3.0 | 3.05-3.36 |

**+9% throughput and -14% GPU power**, closing roughly two thirds of the
gap to Vulkan. DRAM power *rises* toward Vulkan's, which is the tell that
the freed power budget went into real memory traffic.

The same pattern was applied to `mul_mat_vec_q3k_dp4a.hlsl` (Q3_K blocks are
110 bytes, same misalignment) and to the `_nc2` / `_mr4` Q6_K variants.
Phi-3-mini Q3_K_M tg128 went 28.23 -> 28.81 t/s; small models are not power
limited, so the win there is only the instruction count.

### Lessons

- **Achieved DRAM bandwidth can look optimal while the kernel is wasteful.**
  Redundant loads that hit L1 are invisible to a GB/s number and to the
  profiler's per-op table. Watts caught what bandwidth could not.
- On a power-limited part, *energy per byte* is a first-class performance
  metric: excess core power steals clock from the memory path.
- Any quant whose block size is not a multiple of 4 (Q6_K 210, Q3_K 110)
  will have a misaligned fetch path. Fetch 16 bytes at a time, never four
  separate words.

### Which other quants have this bug: none

Q4_0 (18 bytes), Q5_0 (22) and Q8_0 (34) are all misaligned block sizes,
so they look like the same bug, but they are not affected. Their matvecs
give each *thread* a single 4-byte word of quant data (`qs4`, one chunk
per lane), so there is no 16-byte contiguous fetch to coalesce and no
overlapping re-read between neighbouring calls. Q4_1 (20), Q5_1 (24),
Q2_K (84), Q4_K (144) and Q5_K (176) are all 4-byte multiples and use
plain `Load`/`Load4` already.

The one residual cost in those shaders is that `read_u32_*` issues its
second `Load` unconditionally:

```hlsl
uint lo = buf.Load(aligned);
uint hi = buf.Load(aligned + (shift == 0u ? 0u : 4u));   // same address when aligned
```

Roughly half of all blocks are 4-byte aligned, so half the calls load the
same word twice. Rewriting all 38 sites (36 shaders) to return early and
skip the second load measured **neutral** on Phi-3-mini Q5_0 and Q8_0 and
was reverted: the branchless form lets both loads issue back to back so
their latency overlaps, and the duplicate always hits L1. Instruction
count is not the constraint here - unlike Q6_K, where the problem was 8
loads to fetch 16 bytes.

The rule is therefore narrower than it first looks: **widen fetches, do
not merely count them.**

### Still open

- The `K=5120 N=6144` matvec runs at 81.5 GB/s against 104-111 for
  same-shader siblings with the same K (7.2% of the graph). Thread count
  rules out occupancy starvation (3072 groups x 256 threads); the likelier
  explanation is ramp-up/drain that cannot overlap, since these sit between
  the serial GDN ops. `K=5120 N=1024` is similarly low but only 0.8%.
- `--no-mmap` is a free ~20 GB saving on UMA at zero throughput cost, since
  the file mapping and the host-shared D3D12 buffers are otherwise both
  resident. Worth considering as a UMA default.
- The scalar MoE dequant path (`quant_dequant.hlsli`, `mmid_dequant` for
  MMID_Q6_K / MMID_Q3_K) reads one *byte* at a time via `mmid_read_byte`,
  which is a full 32-bit `Load` per element. That is a much coarser
  version of the same disease, but it only runs where the tiled
  `mul_mat_id_gemm` path does not, so it was not measured here.

### A warning about baselines

The first pass at the experiment above looked like a clean 3% regression
(Q8_0 24.95 -> 24.13 t/s) and was nearly reported as one. It was not: the
24.95 baseline was measured on a cold GPU, and reverting the change
reproduced 24.19, not 24.95. Every DX12 decode baseline on this part must
be taken in the same thermal state as the comparison, or the drift will
be attributed to the code.

Note that the profiler only dumped generation graphs 3-5, which hid the
decay entirely - the op table looked identical at `-n 32` and `-n 200`
because it was always early tokens. Use `DX12_PROFILE_GEN_LO` /
`DX12_PROFILE_GEN_HI` to profile a late window.

---

## 7a. Quant MoE prefill is load-issue bound

Q4_K_M MoE prefill trailed Q8_0 by 25% while reading barely half the bytes
(144 vs 272 per 256 elements), so the deficit was instruction issue, not
bandwidth, and not a GEMM or driver problem. `mul_mat_id_q4k_block` was
spending 288 `ByteAddressBuffer.Load` calls per 256-element block: one per
`qs` word plus two per activation element.

`Load4` only requires 4-byte alignment, not 16, which these offsets already
satisfy. Folding the `qs` fetch into 2x `Load4` and the activations into
`Load4` when they are contiguous F32 drops that to 72 loads and leaves the
accumulation order bit-identical. On Arc B390 (wave16), granite-3.0-1b-a400m
Q4_K_M: pp2048 357 -> 429 t/s, pp6144 337 -> 399 t/s, tg128 127 -> 140 t/s.

The residual ~10% is structural: Q8_0 has a dp4a MMID kernel and Q4_K has
none. Before assuming a quant gap is a GEMM deficit, count the load
instructions in the matvec inner loop.

---

## 7b. BF16 is easy to leave out of a type gate

Several MUL_MAT/MUL_MAT_ID fast paths were written as `t == F16 || t == F32`
(or `F16 || Q6_K`) and so silently excluded BF16, even though the kernels
behind them read through the type-generic `load_auto`/`esize` path and
handle BF16 already. The result was that BF16 models quietly kept the slow
route on every vendor, not just one.

Opened up so far, on Arc B390 (wave16):

- fl=105 `mul_mat_wmma64` (64x64 tile): Falcon-H1-7B BF16 pp2048 78 -> 113
  t/s and pp6144 72 -> 101, Qwen3-0.6B BF16 pp512 1412 -> 1843, BF16 mmproj
  vision encode 1948 -> 1719 ms. The gain holds as context grows, so it is
  not a short-prompt artifact.
- fl=53 `mul_mat_id_coop_wide` (NUM_ROWS=16): granite-3.0-1b-a400m
  converted to BF16, pp2048 254 -> 288 t/s.

Still excluded, deliberately, for want of a measurement:

- fl=54 `mul_mat_wmma_kfull` (K<=64). Same omission and the shader is just
  as generic, but no BF16 model on hand reaches it - vision towers checked
  use K=128 - and the op suite has no BF16 K<=64 case. Left alone rather
  than shipped unmeasured.

The LinAlg GEMM also treats BF16 as an unquantized input. Its staging path
must test the BF16 sentinel (`esize == 3`) separately from the physical
two-byte stride: packed BF16 values are loaded eight bytes at a time and
shifted into F32 before conversion to the F16 matrix operand. This applies to
both dense GEMM and grouped-expert MMID; BF16 MMID reuses the F16 LinAlg blob
because the runtime element size controls the load conversion.

On RX 9070 XT, Granite 3.0 1B A400M converted to BF16 measured:

- grouped-expert LinAlg off (`fl=53`): pp512 2646, pp2048 2618 t/s;
- grouped-expert LinAlg on (`fl=202`): pp512 19157, pp2048 19208 t/s,
  about 7.3x faster;
- a prompt profile showed the BF16 expert GEMMs on `fl=202`; tiny residual
  expert calls remained on the cooperative routes;
- decode was unchanged: tg128 302.6 off versus 302.0 on.

Qwen3 0.6B BF16 with dense LinAlg disabled/enabled measured pp512
8993 -> 26446 and pp2048 8484 -> 22994 t/s on the same device.

GLU/UNARY `supports_op` also test `F32 || F16`, but those ops run on F32
activations even under BF16 weights, so BF16 there is moot.

When adding a type to a kernel, grep for the flag number and check every
gate that mentions it.

---

## 7c. Q4_K MoE decode: dp4a, and why one thread must own a whole sub-block

Q8_0 had a dp4a MMID kernel (fl=17) and Q4_K had none. The Q8_1 activation
pre-pass is already type-agnostic, so the only missing piece was the kernel:
`shaders/mul_mat_id_q4k_dp4a.hlsl`, fl=159, env `DX12_MOE_Q4K_DP4A`.

Q4_K sub-blocks are 32 elements and `QK8_1` is 32, so sub-block `j` maps 1:1
onto Q8_1 block `j` and one `qs` word feeds two dp4a lanes (low and high
nibbles). That 1:1 alignment is what makes the kernel cheap to write.

Two results worth keeping:

- **Decode only.** The kernel is matvec-shaped (two output rows per group), so
  at prefill it re-reads the weights once per token and loses badly to the
  block decoder: granite-a400m pp2048 406 -> 334 (-18%) when it was allowed to
  take prefill too. Gated on `src[2]->ne[1] <= 8`; prefill measured unchanged
  after gating (427/427 at pp2048, 339/336 at d6144).

- **Thread granularity dominates.** The first version gave each thread one
  `qs` word: 2 dp4a for ~9 scalar loads (dm, three scale words, qs, plus the
  activation words) and two extra dp4a to rebuild the activation sum. Measured
  inside noise vs the block kernel. Giving each thread a whole 32-byte group
  (two complete sub-blocks) amortises the scale decode over 16 dp4a, turns
  both `qs` and the activations into `Load4`, and - because the thread now
  spans the full sub-block - lets the min term read the Q8_1 `s` field
  (`d * sum(q)`) directly instead of reconstructing it. tg256 126.5 -> 133.7
  (+5.7%), with all four ABBA runs of each arm cleanly separated.

Lane layout is `il = tid%4`, `row_sel = (tid/4)%NUM_ROWS`, `slot = tid/8`.
Splitting the two output rows *across* lanes rather than having every lane do
both matters at small K: granite gate/up is K=1024, i.e. only 4 superblocks,
which would otherwise idle most of the group.

---

## 7d. MoE prefill re-reads every expert once per token

Every one of the 51 `mul_mat_id_*` shaders is a matvec: the dispatch is
`groups_y = n_expert_used`, `groups_z = n_tokens`, so one threadgroup owns a
single (token, expert) pair. At decode that is correct - `n_tokens` is 1 and
the weight read is unavoidable. At prefill it means each expert matrix is
re-read once per token routed to it, and the kernel becomes bound by weight
traffic instead of arithmetic.

granite-3.0-1b-a400m (32 experts, 8 used, 24 layers, n_embd 1024, n_ff 512),
F16 pp6144 on Arc B390, per-dispatch GPU timestamps:

| op | ms | share |
| --- | ---: | ---: |
| MUL_MAT_ID | 36543 | 87.8% |
| FLASH_ATTN_EXT | 3456 | 8.3% |
| MUL_MAT | 1038 | 2.5% |
| everything else | 573 | 1.4% |

Per 512-token ubatch a single MMID node covers 512*8 = 4096 (token, expert)
pairs and each reads a 1 MiB expert matrix: ~4 GiB of weight traffic for
4.295 GFLOP of work. Measured 21.7 ms, i.e. ~198 GFLOP/s against the ~1500
GFLOP/s the dense F16 `mul_mat` reaches on the same device. Grouping the
tokens by expert first would read 32 MiB instead of 4 GiB - a 128x traffic
reduction - and leave the node compute-bound.

The signature to look for is prefill throughput tracking *weight size*
rather than compute. Same model, same prompt:

| quant | pp6144 | weight bytes |
| --- | ---: | ---: |
| F16 | 275 t/s | 1.00x |
| Q8_0 | 465 t/s | 0.50x |
| Q4_K_M | 405 t/s | 0.28x |

Q8_0 prefill has no business being 1.7x faster than F16; on a GEMM path
quantising weights barely moves prefill at all. That it does here is the
tell. (Q4_K_M falls back below Q8_0 because dequant cost starts to outweigh
the smaller read - consistent with 7a.)

This is the bulk of the remaining Vulkan prefill gap on MoE models: Vulkan
sorts tokens into per-expert lists and runs a real tiled GEMM per expert.
The arithmetic reconciles - grouping is worth ~7.6x on the MMID nodes, so
~4.2x overall by Amdahl, and the residual ~2.7x is `KHR_coopmat`, which
together account for the measured 11.4x (275 vs 3161 t/s).

Fixing it needs two new pieces: a bucketing pass that builds per-expert
(token, slot) lists from the router ids, and a tiled GEMM MMID that consumes
them. Both are now implemented, for dense and quantized weights alike -
see 7e.

---

## 7e. Tiled MoE GEMM (`mul_mat_id_gemm`)

`moe_expert_bucket.hlsl` reads the ids tensor in one threadgroup and writes,
into a device scratch buffer, `n_expert+1` exclusive prefix sums followed by
the flat pair indices (`token * n_expert_used + slot`) grouped by expert.
`mul_mat_id_gemm.hlsl` then dispatches `(ceil(ne0/64), ceil(n_tokens/BM),
n_expert)` and each group covers a `BM`x64 tile of one expert, reading that
expert's weights once per tile instead of once per routed token. It is the
dense `mul_mat_wmma_fp16` tile shape - BN=64, BK=16, 4x4 register
blocking, half LDS tiles, fp32 accumulate.

`n_tokens` is a tight bound on the pairs any single expert can own (a token
cannot select the same expert twice), so `groups_y` needs no readback; the
group exits immediately when its tile is past the expert's count.

The slot/token decode and the activation row offset are resolved once per
row into groupshared arrays before the K-loop, so the inner loop costs no
integer divides.

gate/up/down within a layer route through the same ids tensor, so the
bucketing dispatch is cached on (ids tensor, offset, pair count) exactly
like the Q8_1 quantize pre-pass - 72 bucket dispatches per graph become 24.

Arc B390, granite-3.0-1b-a400m F16, `DX12_MOE_GEMM` off vs on:

| test | off | on | |
| --- | ---: | ---: | ---: |
| pp16 | 264 | 337 | +28% |
| pp32 | 316 | 620 | +96% |
| pp64 | 337 | 990 | +2.9x |
| pp128 | 359 | 1505 | +4.2x |
| pp512 | 380 | 2295 | +6.0x |
| pp6144 | 257 | 1465 | +5.7x |
| tg64 | 91.4 | 90.9 | neutral |

pp6144 sits below pp512 because attention, not MUL_MAT_ID, dominates once
the context is long - which is the point: MMID is no longer the bottleneck.

### Dequant-to-LDS

Quantized weights take the same route. `quant_dequant.hlsli` already exposes
a per-element `mmid_dequant(buf, row_off, k)` for every type, so the tile
loader only needs to call it instead of `load_auto`; one wrapper per type
defines `MMID_<TYPE>` plus `MMID_QUANT` and includes the shared body.
Covers Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, IQ4_NL, IQ4_XS
and MXFP4.

Both tile loads stride by `THREADS` rather than taking four consecutive
elements per thread. That keeps the loads coalesced *and* gives each thread
`BK/4` k-values inside a single weight row, so for a quantized tile every
one of them lands in the same block and the block scale folds out of the
unrolled loop. Dequant cost is what makes this matter: the group decodes
`K * BN` weight elements per tile, and re-reading the superblock header per
element would dominate.

Arc B390, `DX12_MOE_GEMM` off vs on (best of the two ABBA passes for off):

| model / quant | test | off | on | |
| --- | --- | ---: | ---: | ---: |
| granite-a400m Q4_K_M | pp512 | 683 | 1380 | +2.0x |
| granite-a400m Q4_K_M | pp6144 | 408 | 886 | +2.2x |
| granite-a400m Q8_0 | pp512 | 594 | 2112 | +3.6x |
| granite-a400m Q8_0 | pp6144 | 459 | 1157 | +2.5x |
| Qwen3.6-35B-A3B Q4_K_M | pp512 | 65 | 148 | +2.3x |

Decode is unchanged on all of them. The quantized gain is smaller than F16
because dequant ALU replaces part of the traffic that was saved, which is
the expected shape: the kernel has moved from bandwidth-bound to
compute-bound.

Note the ordering constraint: the GEMM gate is evaluated *after* the Q8_0
dp4a (fl=17), Q4_K block (fl=51) and Q4_K dp4a (fl=117) gates so it wins
over them, and it clears `use_dp4a_matvec` (the GEMM reads the weights
directly, not through the Q8_1 scratch). The MMID weighted-sum fusion is
skipped when the GEMM is selected.

`BM` is 64 by default; wrappers that set `MMID_BM 128` build a parallel
"tall" blob set (`mul_mat_id_gemm_tall_*`, flag 122 instead of 119) for
every quantized type. A taller tile halves how often an expert's weight
tile is re-read, but a group still computes all `BM` rows, so it only pays
off once a model routes at least that many pairs to one expert.
`dx12_mmid_gemm_use_tall()` estimates that as `n_tokens * n_used /
n_expert` and takes the tall blob at >= 128. B390, r=3-5, pp512/pp6144:

| model / weights                  | BM=64       | BM=128      |
| -------------------------------- | ----------- | ----------- |
| granite-3.0-1b-a400m Q4_K_M      | 1373 /  858 | 1614 /  978 |
| granite-3.0-1b-a400m F16         | 2210 / 1195 | 1966 / 1122 |
| Qwen3.6-35B-A3B Q4_K_M           |  144 /   89 |  136 /   89 |

Granite is 32 experts top-8, so pp512 routes exactly 128 pairs per expert
and the tall tile is fully packed. Qwen3.6-35B-A3B is 128 experts top-8,
so pp512 routes 32 and the tall tile wastes three quarters of its rows.
f16 has no decode to hide and never takes the tall path. Override with
`DX12_MOE_GEMM_TALL=0|1`.

Three sites are coupled and must move together: `MMID_BM` in the wrapper,
`groups_y` in the MMID dispatch, and the 65535 group-limit check in the
flag gate - all three now go through `dx12_mmid_gemm_bm()`.

An earlier attempt recorded `BM = 128` as failing MUL_MAT_ID at 770/790
and blamed register pressure from `acc[8][4]` plus `tacc[8][4]`. That was
wrong. `SHADER_INCLUDE_DEPS` in `CMakeLists.txt` is a hand-maintained list
and did not name `mul_mat_id_gemm.hlsli`, so editing the header recompiled
nothing: the experiment ran BM=128 host geometry against BM=64 blobs, and
half the tile rows were never computed. That explains every symptom,
including why the shader read as dimensionally correct. The header is now
tracked (along with three others that were also missing) and BM=128 passes
790/790. When touching any `.hlsli`, confirm it is in `SHADER_INCLUDE_DEPS`
or the build will silently serve stale blobs.

Gated on the types above, F32 contiguous activations, `ne[3] == 1`,
`n_expert <= 512`, native fp16, and `n_tokens >= 16`. The token floor keeps
decode and small speculative batches on the fused matvec path (the MMID
weighted-sum fusion is itself gated at 8 tokens) and costs nothing: at 8
tokens the two paths measure the same. Overrides: `DX12_MOE_GEMM=0`,
`DX12_MOE_GEMM_MINTOK=<n>`.

Type coverage is now every type `quant_dequant.hlsli` can decode: the 13
above plus NVFP4, Q1_0, Q2_0, TQ1_0, TQ2_0, IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, IQ3_S, IQ1_S and IQ1_M. Each is a three-line wrapper.

---

## 7f. Tiled dense GEMM for codebook quants (`mul_mat_gemm_quant`)

The same re-read pathology existed in *dense* MUL_MAT, and much worse.
The IQ/TQ types have no wmma or dp4a batch variant, so batched MUL_MAT
fell back to `mul_mat_quant.hlsli`: one thread per output element, each
walking the whole of K decoding a codebook as it went. Nothing is reused -
a weight row is re-decoded once per token and an activation row re-read
once per output column - so the cost is `ne0 * ne1 * K` dequants. That is
slow enough that the dispatch had to be split by output row just to stay
under the Windows TDR (see the `key.flags == 43` chunking).

`mul_mat_gemm_quant.hlsli` is the `mul_mat_id_gemm` tile without the
expert bucketing: a 64(token) x 64(output) tile, BK=16, dequant into LDS,
fp32 accumulate. A weight element is decoded once per 64 tokens.

Arc B390, `DX12_MM_GEMM` off vs on, pp512:

| model / quant | off | on | |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B IQ3_XXS | 32.0 | 737 | +23x |
| Qwen3.5-0.8B IQ2_XXS | 23.7 | 587 | +25x |
| Llama-3.2-1B IQ2_M | 17.7 | 500 | +28x |
| SmolLM2-360M IQ4_XS | 162 | 252 | +1.6x |
| SmolLM2-135M IQ4_XS | 478 | 801 | +1.7x |

MXFP4 and NVFP4 reach the same per-element template through the src0_type
fallback rather than flag 43, so they are picked up by type. No MXFP4
model was cached to run end to end, but the op benchmark shows the same
shape of win (`m=4096,n=512,k=14336`):

| type | off | on | |
| --- | ---: | ---: | ---: |
| MXFP4 | 20.1 GFLOPS | 1.35 TFLOPS | +67x |
| IQ3_XXS | - | 766 GFLOPS | |

That single MXFP4 dispatch took 2.99 s before and 44 ms after, which is
why the per-element path needed row-chunking to stay under the TDR.

Decode is unchanged (checked ABBA on SmolLM2-135M IQ4_XS: 223 vs 223 -
a single interleaved run reads low because the preceding pp512 leaves the
GPU in a different state, not because of this path).

For scale: Qwen3.5-0.8B in Q4_K_M reaches 2259 pp512, so IQ3_XXS was 68x
slower than the same model on a tiled path. The gap that remains after
this change is codebook dequant ALU, which is expected.

Gated on `n_tokens >= 16`, F32 contiguous activations, native fp16, and
the dispatch limits. It is evaluated after the per-element gates (43) and
the IQ1_S matvec gate (130) so it wins over both; below the token floor
those still handle decode. Overrides: `DX12_MM_GEMM=0`,
`DX12_MM_GEMM_MINTOK=<n>`.

Two other shaders include `quant_dequant.hlsli`. `get_rows_quant.hlsli`
is one thread per output element but each element is read exactly once,
so there is no reuse to exploit. `flash_attn.hlsl` stages only scores in
LDS (`s_scores`, `s_reduce`), and its quantized K/V reads have no
intra-group redundancy either: a group owns one query, and each thread
decodes a distinct `(kv, d)` element. The reuse there is *across* queries,
so exploiting it would mean tiling several queries per group, not adding
an LDS stage. In practice that path is rarely reached: the KV cache
defaults to F16 regardless of the model's weight quantization
(`llama_context_default_params`, `common/common.h`), so it takes an
explicit `-ctk`/`-ctv` to get there at all.

### Quantized KV cache: the f16 scale bug

`SET_ROWS` is implemented for quantized dst (Q8_0 plus the legacy
Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL shaders) and is now **on by default**.  Kill
switches: `DX12_SET_ROWS_Q8_0=0`, `DX12_SET_ROWS_LEGACY_QUANT=0`.

Getting there took finding a real bug.  Four q8_0 cases failed at ~1.3e-7
NMSE (thresholds 1.3e-8 to 9.1e-8) and two theories were tested and
falsified first - marking `d`/`id` `precise` (codegen changed, error did
not), and matching the CPU one-step `127/amax` reciprocal
(`ggml-cpu/arch/x86/quants.c:333`) against the shader two-step
`1/(amax/127)`.  It was also not broadcast-specific: `nr23=[2,3]` appeared
among both passing and failing cases.

The magnitude was the tell.  NMSE 1.3e-7 is an RMS relative error of
~3.6e-4, which is essentially f16 precision (~4.9e-4) - a 1 ulp error in
the block scale, not in `qs`.  The shader stored `d` with `f32tof16()`,
the **legacy** D3D conversion, which does not round to nearest even.  The
CPU `GGML_FP32_TO_FP16` (`_cvtss_sh(x, 0)` under F16C) does.  Casting to
the native type instead emits an IEEE `fptrunc` that matches:

    // before
    dst.Store(off, f32tof16(d));
    // after
    dst.Store(off, asuint16((float16_t)d));

Applied to `set_rows_q8_0.hlsl` and the five legacy quant shaders (7
sites, including the `vmin` stores).  `SET_ROWS` went 71/75 -> 75/75 for
q8_0 and 135/135 with the legacy quants enabled; the full suite is
15211/15211.

**Any shader writing a stored f16 quantization scale via `f32tof16` has
this bug.**  Prefer `asuint16((float16_t)x)`.

A sweep found 6 more sites, all Q8_1 activation scales on the dp4a path
(`quantize_q8_1.hlsl`, `rms_norm_mul_quantize_q8_1.hlsl`).  Left alone
deliberately: those shaders are in the plain `DX12_SHADERS` list, so they
compile without `-enable-16bit-types` and cannot use `float16_t` without
an integer RNE helper, and the resulting truncation bias (~2.4e-4 mean,
toward zero) sits roughly 10x below Q8_1's own ~4e-3 quantization noise.
Worth revisiting only alongside other dp4a accuracy work, with a
re-benchmark.

KV cache tensors are pre-allocated on the backend buffer, so a declined
`SET_ROWS` cannot fall back to CPU and the scheduler fail-fasts:

    ggml-backend.cpp:898: pre-allocated tensor (cache_k_l0 (view)) in a
    buffer (DX12) that cannot run the operation (SET_ROWS)

`llama_kv_cache` now probes the device with a dummy `SET_ROWS` before
allocating (same `buft_supported` pattern as `llama-model.cpp`) and throws
a normal error instead.  Only quantized types are probed, so the F16/F32
default path is untouched.  This still matters for KV types the backend
genuinely declines (e.g. row widths that are not a multiple of 32).

Verified on Arc B390 with no env vars set: Phi-3-mini Q4_K_M is coherent
under f16, q8_0 and q4_0 KV.

Note the process still exits non-zero (0xC0000005) after *any* failed
context creation, including CPU-only and the pre-existing "failed to
allocate buffer for kv cache" path.  That teardown bug is unrelated and
not backend-specific.

---

## 7d. Command-allocator ring

`DX12_CMD_RING` (default 16, max `CMD_RING_MAX` = 16) sets how far the CPU
may run ahead of the GPU before `ensure_cmd_list_open` blocks recycling an
allocator. Costs 2 MiB of upload buffer per slot.

The default was 4 and that was sized for a decode token. A multimodal
prefill graph issues far more submissions, so the ring wrapped inside a
single graph: on SmolVLM2-256M the first prompt graph spent 31.9 ms of its
32.4 ms dispatch-record phase blocked there. Depth 16 cuts it to 1.9 ms,
worth ~12% end-to-end, and is neutral on pure-LLM prefill and decode.

If you see a GPU idle bubble that no shader or barrier change moves, check
`alloc_wait` in `DX12_PHASE_PROFILE=2` before anything else.

---

## 8. MUL_MAT routing knobs

Two kernels compete for the same quantised MUL_MAT nodes: the wave-matrix
(LinAlg) tiles and the int-dot `mmq` kernel. LinAlg selects a tile first;
`mmq` may then take the node off it if the shape gates allow.

| var | default | effect |
|---|---|---|
| `DX12_MMQ_MIN_N` | 1024 (Q8_0) / 4096 (Q4_K) when LinAlg claimed the node, else 256 | minimum output width before mmq may take a node |
| `DX12_MMQ_MIN_M` | 64 | minimum batch before mmq may take a node |
| `DX12_MMQ_MIN_K` | 8192 | minimum K before mmq may take a node **that LinAlg already claimed** |
| `DX12_MMQ_NARROW64` | on for NVIDIA N<=2560 | select the 64x64 Q8_0/Q4_K/Q5_K/Q6_K MMQ tile |
| `DX12_Q50_INTDOT` | on for NVIDIA LinAlg N<=2560 | override Q5_0 LinAlg with the packed 32x32 int-dot tile |
| `DX12_LINALG_TILE` | unset | force a tile slot (0..3), bypassing selection |
| `DX12_LINALG_TILE_MIN_K` | 2048 | minimum K before the 2D-grid 128x128 tile may be auto-selected |
| `DX12_LINALG_Q8_NC16` | 1 on RDNA4 | use a 16x16 Q8_0 LinAlg tile for M=2..31 and K>=2048 |
| `DX12_LINALG_NV_128X64_OFF` | unset | restore the original four-wave 128x64 blob on NVIDIA for A/B testing |

`DX12_MMQ_MIN_K` exists because mmq's win comes from amortising its Q8_1
setup over a long K, while the N gate could not see K at all. On wide but
shallow ops LinAlg is well ahead (K=576 N=1536: 42.4 vs 35.7 TFLOP/s), and
only past K ~ 8192 does int-dot retake the lead (K=9728 N=2560: 39.1 vs
41.0). Setting `DX12_MMQ_MIN_K=0` restores the old K-blind behaviour and
costs 5-10% of prefill on Qwen3-4B and Phi-3. See
`docs/linalg_how_to_setup.md` section 19.11.

`DX12_LINALG_TILE` is for A/B-ing tile shapes; slot 1 (128x128) is the only
2D warp grid shape and needs >= 64 groups to be selected automatically, so
small models never reach it without forcing.

RDNA4 Q8_0 NC batches use a separate 16x16 tile when M is 2..31 and K is at
least 2048. Against the previous 32x16 tile, `llama-bench` opt-out A/B tests
improved Phi-3 pp8/16/31 by 84%/44%/8% and Qwen3-4B by 55%/32%/15%.
SmolLM2 has K=576 and remains on the old tile. The replay cache retains the
full `uint32_t` pipeline flag because existing specializations already use
values above 255.

Barriers: narrowing barrier scope (see TUNING.md section 7).

GEMM split-K for narrow N. The wide tile really is occupancy-starved at
N=576/M=512 (36 groups on 64 CUs), so split-K would help it in isolation, but
it would not close the Vulkan gap: raising the ubatch gives the wide tile full
occupancy for free and both backends speed up by the same factor, leaving the
ratio at ~1.4x. The gap is uniform kernel efficiency, not tile geometry.
Measured at docs/linalg_how_to_setup.md section 19.16.

## 9. Batch and ubatch

The GEMM M dimension is the ubatch, not the prompt length, so `-p` alone will
not change it - use `-b N -ub N`. Raising the ubatch to 1024 lets the wide
128-row tile fill the GPU on narrow models.

Measuring this in llama-bench needs care. llama-bench sets
`n_ctx = n_prompt + n_gen + n_depth`, and llama_context then clamps
`n_batch` to `n_ctx` and `n_ubatch` to `n_batch`. So at `-p 512` a `-ub 1024`
is silently clamped back to 512 and changes nothing - the prompt length has
to be raised too, and then the ubatch must be varied at a *fixed* `-p` or the
two columns are different workloads. Check the `n_batch` / `n_ubatch` columns
of the llama-bench output, or `-v`, to confirm what actually took effect.

Measured at `-p 1024 -b 1024`, r=8, varying only the ubatch:

| backend | ub=512 | ub=1024 |       |
|---------|--------|---------|-------|
| DX12    | 50559  | 65730   | +30%  |
| Vulkan  | 78459  | 88381   | +13%  |

Both backends gain, so this is not a DX12 quirk - it is the generic
"more rows, more groups, better occupancy" effect, and it does not change the
relative standing of the two backends. Costs more memory for activations, so
it is a caller-side choice rather than a backend default. Prefill only -
token generation runs at M=1 whatever the ubatch is.

Defaults are identical for every backend (`n_batch` 2048, `n_ubatch` 512, set
in `llama_context_default_params`); no backend reads or overrides either, so
a default-flag DX12-vs-Vulkan comparison is already apples to apples.

## 9a. LinAlg staging depth (BK)

Per-shape, the sixth field of the shape string in
ggml/src/ggml-dx12/CMakeLists.txt. Narrow tiles use 32 so a Q8_0/Q4_K
sub-block is covered by one staged step; wide tiles must stay at 16 (128x128
needs 36 KB of groupshared at BK 32, over the limit). Worth ~10% on the mtmd
vision prefill, flat on every llama-bench pp512. See
docs/linalg_how_to_setup.md section 19.18.

## 10. Rejected LinAlg experiments (RDNA4)

LinAlg tile rebalancing for narrow N. 32x32 built as 1:2:2:1 (4 MACs per 4
staging loads) instead of 2:2:1:1 (2 per 3) keeps identical BM/BN but halves
the wave count, and loses 22% - pp512 39691 vs 50877. Wave count governs the
narrow tiles; see docs/linalg_how_to_setup.md section 19.17.

Register-resident GEMM epilogue. Blocked by the driver: only 128 of a 16x16
accumulator's 256 cells can be read back through Get(), for F32 accumulators
as well as integer ones, so Store() to groupshared is the only complete path.
See AMD_LinAlg_Driver_Bug.md.

Copying Vulkan tile selection. Vulkan runs 128x128 for SmolVLM2's prefill
shapes; DX12 is 40% slower there than with 32x32, because LinAlg throughput
falls as a wave holds more accumulators (2 acc 14.1 TF, 8 acc 13.0, 16 acc
10.0). The group-count heuristic is compensating for that and should be left
alone.

64x32 LinAlg tile (BM=64, 4 accumulators per wave). Built and measured via the
repurposed fallback slot: 74.5 us against 32x32's 64.3 on K=1536 N=576 M=512.
Refuted; see section 19.18.

64x64 LinAlg tile (BM=64, BN=64, 4 accumulators per wave). Selected for
SmolVLM2's K=576 N=576 shape via la_order {1,0,3,2}: 72 groups, so unlike
128x128 it does fill 64 CUs, and it is still 17% slower per dispatch than
32x32 (1.917 ms vs 1.645). Occupancy is therefore not the explanation - with
group count controlled, the accumulators-per-wave effect stands on its own.
Correct (1145/1145) but reverted.

BK 64 on the 32x32 tile. Halves the K-step count at K=576, so it should
amortise the barriers and the per-block quant setup. Loses 24% instead -
SmolVLM2 pp512 40005 against 52332. BK 32 is the optimum, not a way-point.
F16 accumulation and a half-width epilogue tile passed 1216/1216 `MUL_MAT`
tests, but reduced SmolLM2 Q8_0 pp6144 by about 1% and was within noise on
Phi-3 and Qwen3-4B. F32 accumulation remains the default.
Fixing GetCoordinate would not close the Vulkan gap. Build with
DX12_REG_EPILOGUE set to compile the register-drain epilogue a working driver
would allow: the GEMM moves ~0.5%. The 18% in AMD_LinAlg_Driver_Bug.md is the
integer probe's number, not the GEMM's. See section 19.19.
## 10a. Isolated GEMM throughput vs Vulkan (large shape)

`test-backend-ops perf -o MUL_MAT -p "n=512,k=14336"`, m=4096. This is the
one measurement that separates the GEMM kernel from dispatch overhead,
attention and everything else in a model run.

| type   | DX12 (TF) | Vulkan (TF) | ratio |
|--------|-----------|-------------|-------|
| f16    |     50.51 |       43.99 | 1.15x |
| q4_0   |     48.15 |       60.06 | 0.80x |
| q8_0   |     44.40 |       54.98 | 0.81x |
| q4_K   |     38.79 |       46.43 | 0.84x |
| q6_K   |     33.74 |       36.36 | 0.93x |
| mxfp4  |     47.32 |       52.67 | 0.90x |
| f32    |     25.26 |       37.90 | 0.67x |
| bf16   |      6.49 |       32.72 | 0.20x |

At a large shape the LinAlg GEMM is competitive - ahead on f16, 0.8-0.93x on
the quants. There is no global throughput ceiling, so the 1.5x model-level
gap on narrow models is a small-shape problem, not a kernel-wide one.

bf16 was a genuine outlier and the clearest single defect in the table: 5x
off Vulkan and 7x off our own f16. Fixed - see below. Nothing in llama.cpp's
common paths exercises bf16, which is why it went unnoticed for so long.

### bf16 routing (fixed)

bf16 was named in the fl=4 wmma list but excluded from both widenings that
follow it: the fl=105 64x64 gate and the LinAlg float gate each tested only
F16/F32. So bf16 always ran the oldest 32x32 / 2x2 tile. Both shaders read
src0 through load_auto(src0_esize), which has handled esize 3 all along, so
the exclusions were oversights rather than limitations.

| stage                        | TFLOP/s |
|------------------------------|---------|
| fl=4 32x32 (before)          |    6.49 |
| fl=105 64x64                 |    9.84 |
| LinAlg                       |   30.42 |
| Vulkan, same shape           |   32.72 |

4.7x, from 0.20x of Vulkan to 0.93x - in line with the other quantised types.

## 10b. Where SmolVLM2 prefill time actually goes

`DX12_PROFILE=1 DX12_PROFILE_PROMPT=1`, pp512, totals over the run:

| op                    |    ms |    % |
|-----------------------|-------|------|
| MUL_MAT (all shapes)  | 5.884 | 72.3 |
| FLASH_ATTN_EXT        | 1.536 | 18.9 |
| ADD / GLU / ROPE      | 0.683 |  8.4 |

Attention is a fifth of the prefill and has never been tuned on this part;
the non-GEMM elementwise ops are another 8%. Even a perfect GEMM leaves 27%
untouched, so the GEMM is not the only thing standing between us and Vulkan.

Per-shape efficiency inside MUL_MAT, same run:

| shape                  | tile    | TFLOP/s |
|------------------------|---------|---------|
| K=576  N=1536 M=512    | 128x128 |    34.5 |
| K=576  N=576  M=512    | 32x32   |    12.4 |

The wide tile reaches 34.5 TF at short K. The narrow tile manages 12.4. The
tile is picked by group count, so the shapes that cannot fill the GPU with a
wide tile are exactly the ones that get the inefficient one.

### NVIDIA D=64 LinAlg flash attention

The original D=64 wave-matrix shader uses 64 query rows and two tiles per wave.
On an RTX 5070 the driver rejects that PSO with `E_OUTOFMEMORY`, so dispatch
falls back to the scalar PF shader. A separate NVIDIA variant with `FA_BR=32`
and `FA_TPW=1` reduces register pressure enough to create the PSO while leaving
the AMD path unchanged.

SmolLM2-135M F16 `pp6144` on RTX 5070:

| path | t/s |
|---|---:|
| LinAlg before (PF fallback) | 8572 |
| PF wide | 9574 |
| LinAlg D=64, BR=64, TPW=1 | 14670 |
| LinAlg D=64, BR=32, TPW=1 | 17080 |
| Vulkan | 22537 |

The final prompt chunk's 30 flash-attention calls dropped from about 82.8 ms
on PF to about 25.6 ms on the BR=32 LinAlg path. `BR=16` regressed to 15666
t/s; `BR=32, BC=64` was also rejected during PSO creation.

### NVIDIA Qwen3 LinAlg GEMM

Qwen3 prefill is dominated by the 128x64 and 128x128 LinAlg GEMM tiles. The
original 128x128 2x2 wave grid uses four waves with 16 accumulators per wave.
On RTX 5070 an 8-wave grid with 8 accumulators per wave is substantially faster
because it reduces register pressure. The NVIDIA 128x64 tile likewise moves
from four waves and eight accumulators per wave to eight waves and four
accumulators per wave. AMD retains the original blobs.

The lower-pressure 128x64 tile reversed the old K=1024 result at pp512 when it
was introduced. After the later attention and pipeline changes, a new RTX 5070
sweep found that Qwen3-0.6B BF16 now prefers 128x128 at both pp512 and pp6144.
NVIDIA float routing therefore allows the wide tile at K>=1024 and accepts 32
output groups; quantized routing and other vendors retain K>=2048 and 64 groups.

NVIDIA also drains matrix accumulators directly with `GetCoordinate()` and
`Get()`, avoiding the LDS store/reload epilogue. The RTX 5070 runtime probe
matched `Accumulator::Store()` exactly. This path must remain NVIDIA-only:
AMD's coordinate access is covered by `AMD_LinAlg_Driver_Bug.md`.

Qwen3 `pp512` on RTX 5070:

| model | reported before | first 128x128 tuning | final |
|---|---:|---:|---:|
| 0.6B BF16 | 2918 | 9207 | 10894 |
| 0.6B Q8_0 | 2670 | 9338 | 10714 |
| 0.6B Q4_K_M | 2931 | 9356 | 10610 |
| 4B BF16 | 692 | 2439 | 2593 |
| 4B Q8_0 | 957 | 2522 | 2670 |
| 4B Q4_K_M | 899 | 2551 | 2553 |

A 16-wave, 4-accumulator layout was slightly slower than the 8-wave layout.
Forcing 128x64 during the shape sweep improved the new eight-wave blob over
the old four-wave blob by 33-67% on the tested Qwen shapes.

The later pp6144 routing sweep measured Qwen3-0.6B BF16 at 7033 t/s with the
old K>=2048 and 64-group gates, 7216 t/s with K>=1024 alone, 7105 t/s with the
32-group threshold alone, and 7333 t/s with both. The production defaults
reproduced 7323 t/s, +4.1% over the original 7059 t/s baseline. Q8_0 and Q4_K_M
were unchanged because they retain the quantized routing thresholds; Phi-3 F16
was also unchanged. Raising the NVIDIA 128x128 staging depth from BK=16 to
BK=32 regressed Qwen3-0.6B BF16 to 7207 t/s and was removed.

### NVIDIA long-prompt LinAlg routing and attention

The SM 6.10 header in DXC 1.10.2605.37 has no BF16 `ComponentType`. Native
BF16 matrix operations cannot be implemented with this preview API even when
the device reports general BF16 shader support; BF16 weights must continue to
convert to F16 during staging.

This remains the newest official SM 6.10 preview compiler as of 2026-08-14.
DXC PR 8734 added `ComponentType::BFloat16` on 2026-08-05, but that change is
not present in the 1.10.2605.37 package. The matching Agility SDK
1.721.3-preview header also has no BF16 LinAlg runtime datatype, so there is no
official compiler/runtime pair that can be used for a native capability query
and execution test yet. Do not mix a main-branch DXC header with the older
runtime. `check_linalg_bf16_toolchain.ps1` reports when both surfaces become
available.

I8 is exposed by the API. An I8 x I8 -> I32 16x16 wave-matrix probe compiled,
created a PSO and executed successfully on RTX 5070. A production quantized
GEMM still needs Q8_1 activation input, per-32-element scale application and
floating accumulation between scaled integer blocks. The existing Q8/Q4
LinAlg shaders therefore continue to dequantize into F16 LDS until that
dataflow is implemented and measured.

The legacy F16 WMMA gate used to run before LinAlg on NVIDIA. This was hidden
by BF16 testing but routed true-F16 Qwen3-4B and Phi-3 through `fl=53`.
Allowing LinAlg to take those nodes raised pp6144 from 700 to 1088 t/s on
Qwen3-4B and from 778 to 1376 t/s on Phi-3 before attention changes.

The original D=96 and D=128 LinAlg flash-attention PSOs are rejected by the
NVIDIA driver with `E_OUTOFMEMORY`. D=96 accepts the same lower-pressure
`BR=32, TPW=1` shape as D=64. D=128 still fails after reducing BR, removing K/V
prefetch arrays, moving Q or output accumulators to LDS, and combining those
reductions. `BR=32, BC=64` also exceeds the 32 KB groupshared limit.

D=128 now splits the output dimension across two workgroups. Each group owns
half the V/output tiles, cutting persistent output registers in half; QK and
softmax are recomputed. This is less efficient than a monolithic kernel but
creates a valid PSO and is substantially faster than PF. The full
FLASH_ATTN_EXT suite passes 5097/5097.

RTX 5070 pp6144 after true-F16 routing and NVIDIA D=96/D=128 attention:

| model | before | after | Vulkan |
|---|---:|---:|---:|
| Qwen3-0.6B BF16 | 3289 | 4703 | 10823 |
| Qwen3-0.6B Q8_0 | 3269 | 4671 | 11550 |
| Qwen3-0.6B Q4_K_M | 3261 | 4671 | 11472 |
| Qwen3-4B F16 | 700 | 1541 | 3836 |
| Qwen3-4B Q8_0 | 1104 | 1550 | 3678 |
| Qwen3-4B Q4_K_M | 1084 | 1514 | 3585 |
| Phi-3 F16 | 778 | 2045 | 3999 |
| Phi-3 Q8_0 | 1386 | 2045 | 3872 |
| Phi-3 Q4_K_M | 1354 | 1982 | 3782 |

The remaining pp6144 gap is not attention alone. In the final true-F16 prompt
chunk, Qwen3-4B DX12 spends about 322 ms in attention and 152 ms in its five
main GEMMs, versus about 141 ms and 46 ms respectively on Vulkan. Phi-3 spends
about 202 ms in attention and 134 ms in its four GEMMs, versus about 137 ms and
39 ms on Vulkan.

### NVIDIA quantized MMQ routing

The Q8_0 and Q4_K LinAlg kernels dequantize weights into F16 LDS before each
matrix step. DX12 already has a better native-integer path for NVIDIA:
`mul_mat_q8_0_q8_1_mmq` and `mul_mat_q4k_q8_1_mmq` quantize activations to
Q8_1, keep the packed weights, apply scales in registers, and use
`dot4add_i8packed`. This is the production form of the native-I8 experiment;
it also avoids adding a second scale/min dataflow to the preview LinAlg API.

The old LinAlg-to-MMQ crossover was too conservative on RTX 5070. It kept most
short-K Qwen and Phi shapes on F16 dequant staging even though the fused MMQ
path won end-to-end at both pp512 and pp6144. NVIDIA LinAlg devices now prefer
MMQ for eligible Q8_0 and Q4_K batch GEMMs. AMD and Intel keep their existing
thresholds, and `DX12_MMQ_MIN_N` remains the override.

The Q4_K `K=4096, N=2560` projection is the measured exception on RTX 5070.
At M=512 its 128x128 LinAlg tile took about 18.1 ms per final graph versus
20.6 ms for narrow MMQ. Keeping that shape on LinAlg raised Qwen3-4B Q4_K_M
pp6144 from an A/B/A MMQ average of 2074.1 to 2081.7 t/s (+0.36%). The
equivalent Q8_0 route produced no measurable end-to-end change and remains on
MMQ.

RTX 5070 results:

| model | type | pp512 before | pp512 after | pp6144 before | pp6144 after |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | Q8_0 | 10714 | 15125 | 4671 | 5045 |
| Qwen3-0.6B | Q4_K_M | 10610 | 14114 | 4671 | 4897 |
| Qwen3-4B | Q8_0 | 2670 | 3175 | 1550 | 1640 |
| Qwen3-4B | Q4_K_M | 2553 | 2876 | 1514 | 1552 |
| Phi-3 | Q8_0 | - | 3468 | 2045 | 2126 |
| Phi-3 | Q4_K_M | - | 3029 | 1982 | 1999 |

A 64x64 MMQ variant halved per-thread accumulator pressure and added 1-2% on
Qwen, but regressed Phi-3 by up to 1.3%. It was removed rather than adding a
second shader family for a mixed, marginal result.

### RTX 5070 follow-up: narrow MMQ, small batches, and Falcon H1

Reintroducing the 64x64 MMQ shape only for output widths at or below 2560
avoids the wide-shape regression. On RTX 5070 pp6144:

| model | type | 128x64 | narrow 64x64 |
|---|---|---:|---:|
| Qwen3-4B | Q8_0 | 1807 | 1838 |
| Qwen3-4B | Q4_K_M | 1698 | 1712 |
| SmolLM2-135M | Q8_0 | 21414 | 22075 |
| SmolLM2-135M | Q4_K_M | 18138 | 18184 |

NVIDIA therefore defaults to the 64x64 MMQ shader for N<=2560.
`DX12_MMQ_NARROW64=0/1` remains available for A/B testing.

The unfinished small-batch MMQ gate also has a stable discriminator. On
NVIDIA, K>=2048 uses a minimum M of 4 while smaller K retains 64. Qwen3-4B
Q8_0 pp16 improved from 334 to 412 t/s. The K gate keeps the narrow models
on their existing path. An explicit `DX12_MMQ_MIN_M` still overrides this.

Falcon H1 exposed a separate SSM convolution gap. Vulkan uses a vector dot
for the common width-4 convolution while DX12 issued scalar loads. Matching
that specialization reduced 44 SSM_CONV dispatches from about 5.8 ms to
2.17 ms at pp6144. Broadcasting the SSM_SCAN uniform values explicitly with
wave intrinsics regressed the scan from about 27 ms to 33 ms and was removed.

Granite MoE confirmed the existing LinAlg MMID routing. The 128x64 tile
measured about 7.8k t/s for Q8_0 and 7.6k for Q4_K_M at pp6144; forcing the
32x32 tile lost 7-10%. Thresholds from 1 through 128 were otherwise noise, so
the default crossover remains unchanged.

`DX12_FUSION_AUDIT` on Falcon, Qwen, and Granite found that most frequent
pairs are views or are already consumed by dispatch-time fusions. Granite's
ADD->ADD residual chain is the only repeated unfused arithmetic candidate,
but it requires graph-level ternary-add plumbing and is not justified without
broader model coverage.

Nsight Graphics GPU Trace now works with a single visible device and a
submit-independent time trigger. A Qwen3-4B F16 pp6144 capture reported:

- tensor-pipe active: 28.6%
- active compute warps: 27.2% of peak
- register file allocation: 67.3%
- active CTAs: 8.9% of peak
- long L1 scoreboard stalls: 11.4%
- barrier stalls: 2.4%
- L2 hit rate: 92.8%

The trace confirms that the retained threadgroup GEMM and D=128 attention are
limited primarily by occupancy/register allocation and dependent load latency,
not DRAM bandwidth or barrier stalls. The exported dispatch timings also show
D=128 attention at about 4.34 ms per layer versus 0.38-1.16 ms for the main
threadgroup GEMMs in the captured pp512 chunk.

### RTX 5070 follow-up: multimodal fusion, D=128, and additional quants

A SmolVLM2 text pp11 graph contains 422 dispatches and 332 barriers, with about
5.88 ms of summed GPU work over a 6.06 ms span. Disabling fusion reduced
throughput from about 1832 to 1793 t/s. Disabling all barriers reached about
4043 t/s but produced incorrect output, so the barrier count is not removable
serialization. An experimental GEMM plus residual-ADD epilogue removed those
ADD dispatches but displaced the existing `ADD + RMS_NORM + MUL` fusion and
exposed standalone RMS dispatches. A/B/A was neutral at 1802/1808/1806 t/s,
so it was removed. A useful multimodal fusion needs to combine a larger chain,
such as Q/K RoPE, QKV, or the dual FFN projections.

The successful pp6144 Nsight trace contains fourteen D=128 attention
dispatches totaling 60.77 ms, or about 4.34 ms each. Duration-weighted metrics
were 23.1% SM throughput, 15.6% tensor-pipe activity, 15.9% active compute
warps, 68.7% register allocation, 8.0% active CTAs, 6.2% long L1 scoreboard
stalls, 0.6% barrier stalls, 0.6% DRAM throughput, and a 98.7% L2 hit rate.
This is a register/residency and serialized matrix-pipeline limit, not a DRAM
or barrier limit. Splitting the output four ways to reduce live accumulators
compiled but produced NaNs in the D=128 tests and was removed.

Q5_K and Q6_K can reuse the register-blocked MMQ shader with the same 64x64,
4x4-thread output shape used by narrow Q8_0/Q4_K. Q5_0 has no register-blocked
MMQ shader; its existing packed 32x32 int-dot tile is faster than LinAlg, while
the packed 64x64 tile is slower. RTX 5070 pp6144 results:

| model | route before | candidate | before | after |
|---|---|---|---:|---:|
| SmolLM2 Q5_K_M | LinAlg | Q5_K MMQ 64x64 | 17200-17295 | 19024 |
| SmolLM2 Q6_K | LinAlg | Q6_K MMQ 64x64 | 19590-19599 | 22021 |
| SmolVLM2 Q6_K | LinAlg | Q6_K MMQ 64x64 | 19514-19570 | 21871 |
| SmolLM2 Q4_K_M | Q5_0 LinAlg | Q5_0 packed 32x32 | 18120-18214 | 19422 |

The combined Q4_K_M default, which also moves its narrow Q6_K tensors to MMQ,
reached 20352 t/s. The same routes improved SmolLM2 pp512 from 27872 to 32929
t/s for Q5_K_M, from 35036 to 42906 t/s for Q6_K, and from 31181 to 34917 t/s
when only Q5_0 changed. The width gate is important: forcing the Q5_K 64x64
tile at Phi-3's N=3072 reduced pp6144 from 1943 to 1884 t/s. NVIDIA therefore
uses these additional narrow routes only at N<=2560. Real SmolLM graph-shape
tests passed 14/14 for Q5_K_M, 10/10 for Q6_K, and 14/14 for Q4_K_M; the full
MUL_MAT suite passed 1146/1146.

### NVIDIA threadgroup LinAlg GEMM

The original LinAlg implementation was built from the RDNA4 driver surface,
which only made the 16x16 wave-matrix path useful. That limitation does not
apply to the NVIDIA driver. Standalone probes on RTX 5070 compiled, created a
PSO, and executed threadgroup-scope matrices at 64x128, 128x128, and 128x256.
The last shape matches Vulkan coopmat2's large output tile.

`mul_mat_linalg_tg_f16` uses one 64x128 threadgroup accumulator. F32
activations are converted into a 64x16 F16 groupshared tile, while aligned F16
weights load directly from the tensor buffer. The result stores directly to an
aligned F32 destination. This removes the weight global-to-LDS round trip and
replaces a grid of independent 16x16 wave fragments with one cooperative
threadgroup operation. Ragged and unaligned shapes retain the wave path.

The tile sweep confirms that larger is not automatically better on this
driver:

| tile | Qwen3-4B F16 pp512 |
|---|---:|
| wave baseline | 2832 |
| 64x64 | 3390 |
| 64x128 | 3580 |
| 128x64 | 2386 |
| 128x128 | 2681 |
| 128x256 | 2109 |

The shader uses the exact F16/F16/F32 64x16x128 capability and reported thread
range when available. Current NVIDIA drivers do not advertise that application
shape despite executing it correctly, so the correctness-tested NVIDIA route
retains a 64-thread fallback. `DX12_LINALG_TG=0` remains an opt-out, and
`DX12_LINALG_TG_STRICT_CAPS=1` requires the exact advertised tuple.

| model | wave pp6144 | threadgroup pp6144 |
|---|---:|---:|
| Qwen3-4B F16 | 1546 | 1753 |
| Phi-3 F16 | 2057 | 2390 |

Full `MUL_MAT` correctness passes 1146/1146. This closes part of the F16 gap,
but Vulkan still has capabilities the current D3D12 API path does not use:
tensor-layout decode callbacks for quantized weights, native BF16 matrix
inputs, flexible edge-tile loads, and coopmat reductions in attention.

### NVIDIA profiler comparison with current Vulkan

The old `build-vulkan` directory was stale at build `2a656e959 (9604)`. A clean
Vulkan build from the same `e57c414ce (10591)` source more than doubled
Qwen3-4B pp6144 performance because it uses the current coopmat2 flash-attention
path:

| type | current Vulkan | DX12 LinAlg |
|---|---:|---:|
| F16 | 7195 | 1755 |
| Q8_0 | 6682 | 1638 |
| Q4_K_M | 6365 | 1551 |

GPU timestamp profiling of the final 512-token chunk at `nkv=6144` isolates
the gap:

| work | Vulkan | DX12 | ratio |
|---|---:|---:|---:|
| F16 flash attention | 28.63 ms | 323.54 ms | 11.3x |
| F16 static GEMMs | 46.17 ms | 111.90 ms | 2.4x |
| Q8_0 static GEMMs | 52.13 ms | 129.69 ms | 2.5x |
| Q4_K_M static GEMMs | 56.41 ms | 147.54 ms | 2.6x |

This is not submission overhead. DX12 dispatch idle is under 0.7 ms in a
446 ms graph. Nsight Systems also shows that DX12 keeps more compute warps
resident while doing less useful work. During active pp512 samples, DX12
averaged 9.6% SM issue and 23.6% tensor utilization with 50.6% compute warps
in flight. Vulkan averaged 20.4%, 43.6%, and 31.1%, respectively.

The implementation difference explains both groups of numbers:

- Vulkan flash attention uses workgroup-scope coopmat2, direct tensor-layout
  K/V loads, cooperative-matrix element operations and reductions, and one
  full D=128 output pass.
- DX12 uses wave matrices, repeatedly stages through LDS, stores matrix results
  to LDS for scalar access, and uses two D=128 output groups. Each output group
  recomputes QK and online softmax because groups cannot share the score tile.
- A single DX12 D=128 output group avoids the duplicate work but increases the
  live output state enough to regress Qwen3-4B pp6144 to 1199 t/s.
- Vulkan quantized GEMMs use coopmat2 decode callbacks to feed packed weights
  directly into tensor-core matrix loads. LinAlg has no equivalent decode
  callback. The DX12 choices are F16 LDS dequantization or DP4A, and both top
  out at roughly half Vulkan's effective matrix throughput.

The remaining gap therefore requires a different API or driver execution path,
not another wave-count or tile-size sweep. The required capabilities are
productive threadgroup matrices with accumulator element access/reductions and
tensor-layout decode callbacks for packed weights.

## 10c. FLASH_ATTN_EXT: quantised K/V cache

`test-backend-ops perf -o FLASH_ATTN_EXT` against Vulkan exposed a defect far
larger than any GEMM gap. Both the `flash_attn_pf_<D>` and `flash_attn_tiled`
gates test K/V through `fa_tiled_type()`, which only accepts F32/F16/BF16, so a
quantised cache fell through to the decode-shaped kernel. That kernel is built
for one query row and collapses at prefill widths:

| case (D=64, nh=8, nr23=[8,1], kv=7680, nb=512) | before | after | Vulkan |
|---|---|---|---|
| K/V q8_0 | 96233 us | 4239 us (22.7x) | 5248 us |
| K/V q4_0 | 78349 us | 4097 us (19.1x) | 5259 us |

End-to-end, Qwen3-4B pp2048 with `-ctk q8_0 -ctv q8_0`: **425 -> 3961 t/s**.

The fix reuses the wave-matrix kernel rather than adding a new one. It already
stages K and V through LDS as `float16_t`, so only the staging step needs to
know the block layout; the matrix core is untouched. `mmid_dequant` is a
compile-time specialisation, so `flash_attn_linalg.hlsl` gains `FA_KV_Q8_0` /
`FA_KV_Q4_0` variants (flags 169-171 / 172-174), mirroring how `flash_attn_cd`
specialises its decode kernel. Head dims are multiples of 32, so a row never
splits a quant block.

Note the quantised path deliberately forces `kv_quad = false`: that fast path
assumes a contiguous F16 cache it can pull 4 elements at a time from.

**Trap that cost an hour.** The blob switch has an outer range guard. Adding a
case inside the switch without widening that guard compiles clean but leaves
the case unreachable, so the pipeline gets no blob and the dispatch produces
NaN. A constant-value dequant still NaN'd, which is what proved the fault was
structural rather than arithmetic. **When adding a flag, widen the outer range
as well as the switch.**

Failures also appeared at `hsk=72`, which cannot reach this path at all -
collateral corruption from the broken dispatch, not a second bug.

### Still open

- **hsk=72 is on no fast path** (992 us vs Vulkan 68 us, 14.6x). `head_dim` is
  gated to {64, 96, 128} in both the pf and wave-matrix paths, and
  `flash_attn_linalg` requires a 16-aligned `FA_D`. Needs either padding 72
  into the 96 kernel with lane masking, or a dedicated variant.
- **q8_0 `SET_ROWS` is still opt-in** (`DX12_SET_ROWS_Q8_0=1`), so a quantised
  KV cache is unreachable end-to-end without it; without the env var
  llama-bench aborts on `cache_k_l0`. q4_0 dst is unsupported, so the q4_0 FA
  variants are currently only reachable through test-backend-ops.

## 10d. Small-shape tile efficiency: no lever left in routing

SmolVLM2's vision shapes run the 32x32 tile at 12.4 TFLOP/s while the same
kernel reaches 34.5 at 128x128 on wider outputs (see 10b), so the obvious
theory was that the selector is too eager to maximise group count: it picks
32x32 (288 groups) over 128x64 (36 groups) for K=576 N=576 M=512, even though
128x128 hits 34.5 TFLOP/s at only 48 groups.

Both halves of that theory were tested and both are wrong.

Sweeping the group-count target (`DX12_LINALG_MM_GROUPS`), SmolVLM2 pp512:

| target | 64 (default) | 48 | 32 | 24 | 16 |
|---|---|---|---|---|---|
| t/s | 55401 | 55669 | 52902 | 52593 | 53323 |

Nothing above noise at 48, and a real loss below it.

Forcing a single tile for the whole model (`DX12_LINALG_TILE`), same workload:

| | heuristic | 128x64 | 128x128 | 32x32 | 32x16 |
|---|---|---|---|---|---|
| t/s | **55152** | 50871 | 38616 | 46957 | 41696 |

The heuristic beats every fixed tile, including 32x32 on its own (46957 vs
55152). Per-shape selection is already extracting more than any single tile
can, so the 12.4 TFLOP/s figure is a property of the 32x32 tile at that shape,
not evidence of a mis-route.

Combined with the earlier refutations - 64x64 (fills the GPU at 72 groups and
is still 17% slower than 32x32), BM=64, BK=64, split-K, and copying Vulkan's
selection table - every routing and geometry lever over the current tile set is
now exhausted. Closing this gap needs a structurally different kernel (e.g.
persistent workgroups, or fusing the small GEMMs), not another tile shape or
threshold.

Note SmolVLM2 pp512 carries about +/-6% run-to-run noise, so treat anything
under ~10% here as nothing.

## 10e. Small-batch float GEMM routing (the LinAlg min-M rule)

LinAlg eligibility used to have a small-batch rule for quantised weights only:
`quant_take` admits anything with `ne[0] >= 128 && ne[1] >= linalg_mm_min_m`,
while floats could only qualify through `key.flags == 105`, which needs
`ne[1] >= 64`. Float GEMMs with M between the two therefore fell back to the
fl=4 32x32 wmma tile. The argument in the code for admitting narrow quant
shapes - the padded wave-matrix tile still streams the weights exactly once -
applies to floats verbatim, so `float_take` now mirrors `quant_take`.

`linalg_mm_min_m` then dropped from 4 to 2. Not to 1: M=1 is decode and wants
the mat-vec kernels. At M=2 the 32x32 tile computes 32 rows to use 2.

This only moves shapes no benchmark exercises - `pp6144` runs at M=512 (already
fl=105) and decode at M=1 - which is why it went unnoticed. It matters for
mtmd, speculative decoding, and small-batch serving.

Measured on SmolVLM2-256M F16 mtmd, per-graph GPU totals (DX12_PROFILE=1
DX12_PROFILE_PROMPT=1):

| text prefill graph | before | after |
|---|---|---|
| M=7 chunk (fl=4 -> fl=110) | 18.08 ms | 4.14 ms |
| M=2 chunk (fl=4 -> fl=110) | 7.10 ms  | 3.95 ms |

The M=2 figure is reproducible to +/-0.05 ms across runs, which matters because
the end-to-end number is not - see below. Gates held: test-backend-ops
15138/15138 on DX120 and DX121, SmolLM2 pp6144 42159 / tg512 973, Qwen3-4B
pp6144 4220.

## 10f. Where the SmolVLM2 mtmd gap actually is

End-to-end mtmd prefill carries +/-25% run-to-run noise (observed spread
1282-1793 t/s over nine runs of an identical command). Never A/B this workload
end-to-end; measure the affected graph's GPU sum instead, which is stable to
under 1%.

`llama-mtmd-cli` logs `mtmd batch encoding done in N ms`, which splits the run
cleanly. That split is stable to +/-1 ms and is the right instrument:

| phase | DX12 | Vulkan |
|---|---|---|
| vision encode | 37 ms | 17 ms |
| text prefill (remainder) | 13 ms | 12 ms |

Text prefill is at parity after 10e. The entire remaining gap is the vision
encoder, so work aimed at the language model cannot move this number.

Inside the encode, GPU work is 12-20 ms while the encode is 32-43 ms. The
vision graph is one-shot - a vision encoder runs exactly once per image - so it
never benefits from command-list replay, and pays full CPU recording every
time. `DX12_GRAPH_WALL=1` shows the split (the 935-node text graphs replay and
cost 0.18-2.4 ms of CPU; the 396-node vision graph costs 4.6-26 ms).

Two things that look like the cause and are not:

- PSO compiles. The prewarm log covers them (28/28 keys). Disabling it with
  DX12_NO_PSO_PREWARM=1 costs only ~6 ms of the vision graph, so the rest of
  the recording cost is real work, not compilation. The 8.9 ms of GPU idle
  visible in graph #1 of a fresh build is the log being invalidated by the
  build stamp - it disappears on the second run. Always discard the first run
  after a build.
- Submission chopping. `flush_threshold` is 24 during prompt, which splits the
  vision graph across many submissions. Sweeping DX12_FLUSH_THRESHOLD over
  24/48/96/192/384/100000 moved the encode 37 -> 32 ms, at the edge of noise.
  Left at 24 (it is TDR insurance).

Worth noting for whoever picks this up: a model with a vision encoder creates
two DX12 devices on one adapter, and both prewarm the same merged key list, so
the 28 keys get created twice (~27 ms of duplicated work). It runs on a
background thread during model load, so it is off the critical path.

## 10g. Flash attention at D=128: the tile shape is already optimal

Qwen3-4B F16 pp6144 is *entirely* an FA problem. Per-op, last ubatch (nkv=6144),
DX12 vs Vulkan (GGML_VK_PERF_LOGGER=1):

| op                                   | DX12      | Vulkan    | ratio          |
|--------------------------------------|-----------|-----------|----------------|
| FLASH_ATTN_EXT fl=168 D=128 nq=512   | 79.28 ms  | 43.04 ms  | 1.84x slower   |
| all 5 MUL_MATs combined              | 65.45 ms  | 66.61 ms  | parity         |
| graph total                          | ~153.5 ms | ~120.1 ms | 1.28x          |

The GEMMs are done - DX12 even wins two of the five shapes. Everything left is FA.
Note the contrast with SmolLM2 (D=64), where FA is only 1.19x off and the narrow
GEMMs are the problem instead. The two models fail for opposite reasons.

D=128 runs at FA_BR=32/FA_BC=32/FA_TPW=1, i.e. half the query rows of D<=96.
That looked like the obvious lever. It is not - all three ways of reshaping the
tile were measured and all three are worse (Qwen3-4B F16 pp6144, baseline 4244):

- **FA_BR 32 -> 64** (double the query rows): **4011**, -5.5%. It does fit in the
  32 KB budget - the LDS comment implies otherwise, but DXC compiles it. The
  group just gets too big: 512 threads and ~30 KB of LDS drop the card to one
  group per CU, and the lost occupancy costs more than the halved K/V staging.
- **FA_BC 32 -> 64 with FA_TPW 2** (double the keys, hold wave count at 4/256
  threads so the PV staging slots still fit): **3882**, -8.5%. This is the
  trade the tile-shape comment recommends ("FA_BC wants to be as wide as the
  32 KB allows") and it does not pay at D=128.
- **Stage K natural [c][d] and load the B fragment ColMajor** instead of doing a
  transposing scattered LDS write: **3846**, -9.4%. Correct (5097/5097) and it
  uses *less* LDS, but `MatrixLayout::ColMajor` is evidently not a first-class
  path in the driver - it loses more than the scattered stores it removes.
  (The enum is `ColMajor`, not `ColumnMajor`.)

So the D=128 shape is a local optimum in all three directions. The remaining
1.84x is inside the driver's wave-matrix throughput or its LDS behaviour, not in
anything the tile geometry can reach. Do not re-run these three.

## 10h. Where each model loses: FA scales with head_dim, GEMMs with output width

Completing the picture from 10g with Phi-3 (D=96) and the Q8_0 quant path.
All figures are the last prefill graph of pp6144, DX12 vs Vulkan per-op.

### Flash attention is a fixed ~22 TFLOP/s ceiling, not a shape problem

| model     | D   | DX12 FA  | Vulkan FA | ratio | DX12 TFLOP/s | VK TFLOP/s |
|-----------|-----|----------|-----------|-------|--------------|------------|
| SmolLM2   | 64  | 9.68 ms  | 8.14 ms   | 1.19x | -            | -          |
| Phi-3-mini| 96  | 52.94 ms | 29.21 ms  | 1.81x | ~21.4        | 38.8       |
| Qwen3-4B  | 128 | 79.28 ms | 43.04 ms  | 1.84x | ~23.4        | 43.1       |

The important observation is the last two columns: **DX12 FA delivers ~21-23
TFLOP/s no matter what the head dim is, while Vulkan scales up to 38-43.** The
deficit is not that the D=128 tile is misshapen - 10g proved the geometry is a
local optimum in all three directions, and D=96 is 1.81x off while already
running the "good" FA_BR=64/FA_TPW=2 config. DX12 is simply pinned at a fixed
throughput that the per-KV-tile serial work (11 group barriers, the softmax row
scan, and the P round trip through LDS) sets, and which more D cannot amortise
because FA_BC stays at 32.

That makes the only remaining FA lever a structural one - fewer barriers and
less LDS traffic per KV tile - not another tile-size sweep. D=64 looks healthy
only because the kernel is short enough there that the ceiling barely binds.

### F16 GEMMs are fine, and on Phi-3 they are better than Vulkan

Phi-3 F16, four MUL_MAT shapes, DX12 vs Vulkan:

| shape (M=512)     | DX12     | Vulkan   |                |
|-------------------|----------|----------|----------------|
| K=8192  N=3072    | 27.11 ms | 27.90 ms | parity         |
| K=3072  N=16384   | 23.24 ms | 50.23 ms | DX12 2.16x faster |
| K=3072  N=9216    | 14.93 ms | 16.35 ms | DX12 faster    |
| K=3072  N=3072    | 5.55 ms  | 5.88 ms  | DX12 faster    |
| all four          | 70.83 ms | 100.36 ms| **DX12 1.42x faster** |

DX12 wins Phi-3 GEMMs outright, which is why DX12 beats Vulkan end to end on
Phi-3 F16 (4566 vs 4077) despite giving up 24 ms on FA. Vulkan's 31.7 TFLOP/s on
K=3072 N=16384 is its own outlier - that shape is what drags its F16 number
below its Q8_0 number.

### The quant gap is the narrow-output GEMM deficit, not a dequant problem

Qwen3-4B Q8_0, five MUL_MAT shapes:

| shape (M=512)  | DX12     | Vulkan   | ratio             |
|----------------|----------|----------|-------------------|
| K=2560 N=9728  | 30.06 ms | 28.77 ms | 1.04x             |
| K=9728 N=2560  | 22.29 ms | 15.17 ms | 1.47x             |
| K=4096 N=2560  |  9.19 ms |  8.48 ms | 1.08x             |
| K=2560 N=4096  |  5.87 ms |  6.49 ms | DX12 faster       |
| K=2560 N=1024  |  5.32 ms |  2.30 ms | **2.31x**         |
| all five       | 72.74 ms | 61.21 ms | 1.19x             |

Compare F16 on the same model, where the same five shapes are at parity
(65.45 vs 66.61). The whole Q8_0 GEMM regression sits in the two narrow-output
shapes, N=1024 and N=2560 - the identical signature to the N=576 shapes on
SmolLM2 F16, i.e. the `GetCoordinate` driver bug in AMD_LinAlg_Driver_Bug.md.
There is no separate quant/dequant problem to chase. FA is 1.84x off on Q8_0
exactly as it is on F16, so the quant path does not touch it.

**Refuted:** routing K=9728 off the int-dot kernel via `DX12_MMQ_MIN_K=100000`
(so LinAlg takes it) - Qwen3-4B Q8_0 pp6144 3986.6 vs 3988.4 default. The two
kernels are tied at 39-41 TFLOP/s just as the mmq_min_k comment records, and
both lose to Vulkan's 58.8. Leave mmq_min_k at 8192.

## 10i. Vulkan dataflow audit and the remaining FA floor

A second implementation-level comparison with Vulkan CM1 found two material
differences in the live shaders:

- Vulkan can load aligned F16 K and V cooperative matrices directly from
  global memory. DX12 staged both through LDS.
- Every DX12 PV output tile used a full workgroup rendezvous even though its
  staging slot is wave-private.

The direct-load API is only partly usable on the tested AMD driver:

- Direct `MatB` descriptor loads of K with `ColMajor` are numerically broken.
  `Align=32` produced about 0.20 relative error. Restricting the experiment to
  the 128-byte-aligned K fragments still produced about 0.05 error. This is a
  driver defect, not an optimization path.
- Direct row-major descriptor loads of V are correct: 5097/5097 FA tests pass.
  Full aligned F16 tiles now bypass the V prefetch, LDS write, LDS reload and
  staging barrier. Quantized and partial tiles retain the old path.
- The PV staging slot is written and read only by one wave. Replacing its
  `GroupMemoryBarrierWithGroupSync()` with `GroupMemoryBarrier()` is correct;
  the group barrier at the loop tail still protects reuse as score storage.

The NVIDIA D=128 variant later added a different direct-K path: load K
row-major and transpose it with a matrix cast. That operation is also broken on
the tested AMD driver. When the path was compiled into the generic D=128
shader, it caused 271 F16-KV failures with 0.14-0.24 relative error. Direct K
is therefore compile-time enabled only for the NVIDIA D=128 variant; AMD keeps
the existing LDS-transposed K path while retaining direct V. This fixes the
full FA suite (5166/5166) without falling back to the scalar shader. Qwen3-4B
Q8_0 pp6144 measured 4374 t/s, versus about 3906 with the incorrect direct-K
path and about 1890 with D=128 LinAlg disabled.

Qwen3-4B F16 at pp6144:

| implementation                  | FA at nkv=6144 | pp6144 |
|---------------------------------|----------------|--------|
| previous                         | 79.28 ms       | 4244   |
| direct V                         | 76.64 ms       | 4352   |
| direct V + wave-private PV fence | 75.85 ms       | 4372   |

The final 3-model results (same command as section 10h) were:

| model             | F16  | Q8_0 | Q4_K_M |
|-------------------|-------|------|--------|
| SmolLM2-135M      | 42663 | 41449 | 38980 |
| Qwen3-4B          | 4367  | 4037 | 3784  |
| Phi-3-mini        | 4658  | 4620 | 4175  |

SmolLM2 remains dominated by its narrow GEMMs, so its FA change is hidden by
run variance. The larger models show a repeatable 1-3% end-to-end improvement.

SmolVLM2-256M mtmd was repeated four times per format after the final rebuild,
discarding the cold first run. The median prompt result and mean image-encoding
time of the remaining runs were:

| format | prompt t/s | image encode |
|--------|-----------:|-------------:|
| F16    | 1505       | 43 ms        |
| Q8_0   | 1365       | 38 ms        |
| Q4_K_M | 1504       | 35 ms        |

The prompt results still span 1241-1700 t/s, consistent with the known mtmd
variance. Direct V affects the language FA graph, not the vision GEMMs, and no
mtmd end-to-end gain can be distinguished from that noise.

The audit also put hard upper bounds on the remaining structural ideas:

- Replacing the LDS partial max/sum reductions with segmented wave reductions
  was correct (5097/5097) but flat to slightly slower: Qwen 4231 vs 4244 and
  Phi-3 4540 vs 4566.
- Removing the entire online-softmax calculation and two group barriers, while
  retaining QK, PV and their stores, reduced D=128 FA only from 79.28 to
  68.57 ms. A perfect softmax rewrite can recover at most 13.5%; the result is
  still 1.59x Vulkan's 43.04 ms.
- Keeping the four PV accumulators live across KV tiles regressed FA from
  75.85 to 104.54 ms (+38%). Persistent accumulation is blocked by the same
  accumulator/VGPR scaling defect as the GEMM experiments.
- Folding the scale into Q was correct but performance-neutral.
- A dedicated direct-V shader variant was slower (77.35 ms) than the guarded
  runtime path (75.85 ms).
- GQA packing cannot reduce prefill traffic at fixed `FA_BR`: packing `g`
  heads leaves `n_heads * N_queries / FA_BR` groups unchanged. Vulkan only
  enables its GQA fold for `N <= 8`; it is not active in the pp6144 comparison.

This narrows the remaining gap to the LinAlg matrix/LDS pipeline itself:
descriptor `ColMajor` is broken, `GetCoordinate()` is broken, PV accumulators
cannot remain live, and even a softmax-free kernel is 1.59x behind Vulkan.
Further shader tuning or a large softmax rewrite is not justified on this
driver. Material progress now requires driver fixes for descriptor ColMajor,
accumulator coordinate access, or accumulator throughput.

## 10j. Direct LinAlg convolution and OUT_PROD

`CONV_2D`, `CONV_TRANSPOSE_2D`, and `CONV_3D` previously ran one scalar dot
product per output element. They now use adaptive 32x32, 64x32, and 128x32
LinAlg tiles. The wider tiles add waves rather than accumulators, keeping every
wave at two accumulator matrices while reusing the expensive gathered input
tile across more output channels. Each group gathers its convolution window
directly into a 32-deep F16 LDS tile, so there is no materialized `IM2COL`
tensor or additional dispatch. The 2D variants also use host-precomputed
multiply-high fast-div constants for output and kernel coordinates, matching
Vulkan's address-generation strategy. Shapes with fewer than 16 output
channels, output positions, or reduction elements retain the scalar shader.
`DX12_LINALG_CONV=0` forces the old path.

Representative RX 9070 XT results:

| operation and shape | scalar DX12 | LinAlg DX12 | speedup | Vulkan |
|---|---:|---:|---:|---:|
| CONV_2D, 4x4x8 -> 128, F32 | 144.98 us | 18.55 us | 7.82x | 7.33 us |
| CONV_2D, 4x4x256 -> 4096, F32 | 218.95 ms | 13.73 ms | 15.95x | 3.38 ms |
| CONV_TRANSPOSE_2D, 3x3, F32 | 6.22 ms | 2.24 ms | 2.78x | 1.08 ms |
| CONV_TRANSPOSE_2D, 3x3, F16 | 7.56 ms | 2.23 ms | 3.40x | 1.07 ms |

Vulkan remains faster because its cooperative-matrix convolution uses larger
tiles and a more efficient matrix/LDS pipeline. The DX12 tile deliberately
stays at two accumulators per wave: the accumulator scaling experiments in
10i show that increasing register-resident matrix count is counterproductive
on this driver. The largest legal tile reaches 10.01 TFLOP/s on the large
shape, versus Vulkan's 40.69 TFLOP/s.

`OUT_PROD` now has an F32/F32 LinAlg implementation. It computes the transposed
view `B[N,K] * A^T[K,M]` so the physical `[M,N]` destination is written
contiguously. Batch repetition in dimensions 2 and 3 and transposed `src1`
strides are handled without intermediate tensors.

`CONV_TRANSPOSE_1D` remains on its scalar F32 shader. Its tests require 1e-7
NMSE, while converting operands to the only matrix input type exposed by this
driver (F16) produces 4.66e-7. Enabling a faster but less accurate path is not
acceptable.

Validation: 15209/15209 tests on DX120 and 15138/15138 on DX121. Repeated
SmolVLM2 Q8_0 image encoding remained in the existing 33-42 ms range, so these
direct-convolution gains do not produce a distinguishable mtmd end-to-end
change in that model.

## 10k. LinAlg GEMM bias epilogue

LinAlg `MUL_MAT` can absorb an immediately following broadcast `ADD` when the
other operand is one contiguous F32 bias value per output channel. The bias is
read while scalar lanes drain the accumulator tile from LDS, immediately before
the final global store. This removes the standalone ADD dispatch and avoids
writing and rereading the unfused GEMM output.

The fusion requires an F32 GEMM and ADD destination with identical shape and
strides, an exclusive GEMM consumer within the graph, and a bias tensor whose
element count equals the GEMM output width. Residual adds, shared GEMM outputs,
and non-LinAlg GEMMs keep the standalone ADD. `MUL_MAT_ID` uses the same shader
source but compiles the epilogue out because its `op_params` and `src2` bindings
carry expert-routing metadata.

`DX12_NO_FUSE_GEMM_BIAS=1` disables this fusion. Set
`DX12_FUSE_GEMM_BIAS_LOG=1` to print each absorbed GEMM shape and LinAlg route.

SmolVLM2 vision encoding absorbs 72 GEMM bias adds, reducing the graph from
300 to 228 dispatches. RX 9070 XT prompt-profile results:

| model type | separate ADD | fused epilogue | GPU reduction |
|---|---:|---:|---:|
| F16 | 12.522 ms | 10.003 ms | 20.1% |
| Q8_0 | 13.116 ms | 11.476 ms | 12.5% |
| Q4_K_M | 12.346 ms | 11.026 ms | 10.7% |

The one-shot `mtmd batch encoding` wall time remains noisy at roughly 30-40 ms,
so the GPU improvement does not translate into a stable end-to-end percentage.
The F16 enabled and disabled runs generated identical text with the same seed.
Validation: 1145/1145 targeted `MUL_MAT` tests and 15209/15209 full tests on
DX120, plus 15138/15138 full tests on the non-LinAlg iGPU.

## 10l. SmolVLM2 vision startup and PSO prewarm

A fresh RX 9070 XT comparison separated vision graph GPU time from one-shot
encoding wall time. Four unprofiled fresh-process F16 runs averaged 36.75 ms
for DX12 and 16.00 ms for Vulkan, with enough run-to-run noise that these wall
figures are diagnostic rather than suitable for small A/B decisions.

Concurrent timestamp samples put the identical F16 vision graph at 11.840 ms
on DX12 and 10.068 ms on Vulkan. Q8_0 and Q4_K_M model files showed the same
direction, with DX12 GPU sums 1.8-1.9 ms higher, but the vision tower remains
F16 and the per-run clocks varied substantially. These are repeated samples of
the same graph, not quantization-specific vision results.

The F16 timestamp profile localizes the slower DX12 dispatches to the large
LinAlg GEMMs and D=64 flash attention. Vulkan executes those cooperative-matrix
groups faster. DX12 recovers much of the gross difference through its 72 fused
bias adds and a faster GELU group, leaving the smaller timestamped gap above.

Direct instrumentation of `clip_image_batch_encode` found that graph building,
input preparation, and output readback are not the large difference:

| stage, eight alternating runs | Vulkan | DX12 |
|---|---:|---:|
| graph build and allocation | 0.27 ms | 0.14 ms |
| input preparation and upload enqueue | 0.77 ms | 1.01 ms |
| scheduler compute and synchronize | 13.52 ms | 30.10 ms |
| output readback | 0.32 ms | 0.86 ms |

The vision scheduler contains one GPU split on both backends. Light
instrumentation measured DX12 command recording at about 4.8 ms, and its
dispatch timestamps span about 11.8 ms, so roughly 13 ms of the first DX12
compute remains outside the timestamped dispatch interval. Skipping the 3 MiB
image upload and forcing stable 1740 MHz GPU clocks did not remove it.

Running the exact same encode twice in one process is decisive: the second
DX12 encode usually takes 11-13 ms, while the second Vulkan encode takes
8-10 ms. Thus steady execution is close to the timestamp result; the large
first-image gap is cold D3D12 work before the first timestamp, consistent with
driver resource residency or first-use command processing rather than shader
throughput. A model-load dummy graph only improved the later first image by
about 2 ms after the intervening startup work, so that workaround was not kept.

PSO creation can still amplify cold-start latency. With prewarm disabled, one
run recorded a 22.387 ms `NORM` creation outlier inside the vision graph, but
the same PSO normally created in less than 0.7 ms. Controlled prewarm-disabled
measurements put the reproducible first-use pipeline overhead closer to 5-7 ms.
This does not explain the full default-path wall gap.

The previous prewarm started workers for every enumerated logical DX12 device,
including an unused duplicate RX 9070 XT entry. Prewarm now starts on the first
buffer allocation for a selected device, early enough to overlap model weight
loading, with backend initialization as a fallback. This preserves the disk
log's first-use ordering and avoids duplicate unused-device compilation. In a
27-key trace the selected worker completed in 14.6 ms and no PSO was created
inside the vision graph; lightly instrumented graph recording was 4.757 ms.

A D3D12 pipeline library could reduce background model-load CPU cost for large
logs, but it cannot address the cold pre-dispatch interval and does not help the
first run after a build or driver-cache invalidation. PSO serialization and
moving the current PSO prewarm earlier are therefore not expected to close the
SmolVLM2 gap. The remaining actionable shader work is the smaller steady-state
GEMM and D=64 FA difference; the larger first-image penalty needs driver-level
residency and queue analysis.

Validation: 15209/15209 tests on DX120.

## 10m. RDNA4 SQTT follow-up

RGP capture on this multi-adapter process needs both
`DX12_VISIBLE_DEVICES=0` and `DX12_RGP_CAPTURE=1`. The latter skips the
temporary D3D12 capability-probe device so Radeon Developer Panel attaches to
the real compute device instead of the short-lived probe. Default adapter
validation is unchanged when the variable is absent.

A full 228-dispatch vision capture and instruction traces for the 768x768
GEMM, D=64 FA, and 768x3072 GEMM resolved 100% of sampled ISA addresses. All
three kernels spend substantially more traced time in `s_wait_loadcnt`,
`s_wait_dscnt`, LDS dependencies, and `s_barrier_wait` than in matrix
instructions. This confirms that the remaining steady deficit is the
load/LDS pipeline, not insufficient matrix instruction issue rate.

Two structural changes were tested from that evidence:

- Raising the 128x64 GEMM tile from BK=16 to BK=32 halves its K-step barrier
  count but increases LDS from about 16 KB to 28 KB. The three main vision
  GEMMs regressed 26-29%, and the graph GPU sum rose from about 11.8 ms to
  16.8 ms. The barrier waits are an occupancy symptom; deeper staging removes
  too many resident groups.
- Full aligned F16 FA now prefetches the next K tile into registers after QK
  and overlaps that load with the current tile's softmax and PV work. The next
  iteration only unpacks the completed load into LDS. A same-build A/B reduced
  vision D=64 FA from 3.078 ms to 2.958 ms (-3.9%) and the complete vision GPU
  graph by about 1.1%. Masked attention also uses the pipeline and invalidates
  prefetched data when the mask skips a tile. At pp6144 this reduced final-tile
  FA from 9.27 to 8.73 ms on SmolLM2 D=64 and from 54.79 to 52.61 ms on Phi-3
  D=96, improving end-to-end throughput by about 2.0% and 1.1%. Quantized KV
  paths retain the original loading sequence.

The GPU profiler now has an optional low-overhead graph-boundary mode,
`DX12_GRAPH_GPU_PROFILE=1`, and `DX12_SYNC_WALL=1` reports individual backend
synchronize calls. Boundary timestamps proved there is no hidden command work
between graph start and the first dispatch: that interval is about 0.001 ms.
Normal-path synchronize tracing instead found one first-vision fence wait of
22-23 ms in the typical 30-32 ms encode. On noisier runs the same one-time cost
moves into a D3D12 recording or submission call and the later fence wait
shrinks. This is deferred AMD first-use work, not an omitted backend barrier.

Creating all logged PSOs early already removes lazy PSO creation from the
vision graph. Executing zero-work bindings for those PSOs during model load
did not change the 22-23 ms first fence wait or the 30-31 ms median encode, so
it was rejected. Disk PSO caching, earlier PSO creation, and zero-dispatch
warming cannot close the cold gap on this driver.

Validation after the retained FA pipeline and profiling support:
15209/15209 tests on DX120.

## 10n. Vision LayerNorm and affine fusion

SmolVLM2 repeats `ADD -> NORM -> MUL(weight) -> ADD(bias)` around each
transformer sublayer. A fused shader now preserves the residual ADD output
while computing the LayerNorm and affine result in one dispatch. The initial
standalone `NORM -> MUL -> ADD` chain uses the same shader without the
residual input.

Fusion is limited to F32 tensors with identical output shapes, exclusive NORM
and MUL intermediates, and contiguous F32 weight and bias vectors matching the
row width. The residual and affine outputs must share one UAV resource so both
remain visible to later graph nodes. `DX12_NO_FUSE_NORM_AFFINE=1` restores the
separate dispatches for A/B testing.

The shader caches the three 768-wide row values owned by each thread, performs
the same two-pass mean and variance reductions as `norm.hlsl`, and uses
`precise` intermediates for normalize, scale, and bias. The explicit
intermediates are required: allowing arithmetic contraction changed the final
vision embedding, while the retained version is bit-identical to the unfused
graph.

All 25 SmolVLM2 affine LayerNorm chains qualify. The first three-dispatch chain
becomes one dispatch, and 24 four-dispatch residual chains become one dispatch,
reducing the vision graph from 228 to 154 dispatches. Alternating same-build
F16 runs on RX 9070 XT measured:

| path | complete vision GPU graph |
|---|---:|
| separate ADD/NORM/MUL/ADD | 15.89 ms |
| fused | 13.80 ms |

This is a 2.09 ms, or 13.2%, reduction in steady vision GPU time. Four-image
prompt runs improved from about 2088 to 2235 tokens/s in the same alternating
sample, although the GPU graph timestamp is the more stable comparison.

Validation: bit-identical final 36,864-element vision embeddings in fused and
unfused runs, repeated-image replay coverage, and 15209/15209 tests on DX120.

## 10o. Rejected row-major GEMM and FA K layouts

Two row-major input experiments were rejected on the RX 9070 XT driver.

For GEMM weights, a GPU transpose produced the intended padded F16 row-major
data. Reading that data through scalar loads and the existing LDS staging path
was bit-identical, proving the packing step. Direct descriptor-backed
`MatB::Load` remained numerically incorrect after varying root slots, alignment
hints, byte versus element strides, and load sequencing. Staging the packed
layout has no performance rationale because it replaces contiguous source
reads with scattered reads, so the complete experiment was removed.

For D=64 vision FA, the existing `PERMUTE -> CPY` was replaced experimentally
with a physical `[D][tokens]` K layout and a matching FA variant. A control
variant using the original physical layout and the new FA staging code was
bit-identical. With the alternate physical layout, both direct descriptor
loads and scalar reload into LDS produced the same incorrect final embedding:
all 36,864 values differed, with 12.38 RMSE and 77.15 maximum absolute error.
The path was removed rather than weakening the correctness gate. The retained
FA continues to transpose K into LDS and to load row-major V directly.

## 10p. MoE decode routing and MTP gate fusion

Granite MoE decode exposed three avoidable routing and aggregation costs:

- The down-projection MMID epilogue now applies the selected router weight
  directly. A single `moe_sum` dispatch then reduces the selected expert
  outputs instead of replaying a chain of expert-view `ADD` nodes.
- Descending `ARGSORT` with source width at most 256 and K at most 16 uses
  repeated wave maxima instead of padding to a 1024-entry bitonic sort.
- `GET_ROWS -> RESHAPE -> SUM_ROWS -> CLAMP -> DIV` is fused into one selected
  router-weight normalization dispatch.

The first implementation of expert aggregation serialized every expert in one
MMID dispatch and regressed Granite decode by 24%. Combining independent gate
and up MMIDs with SwiGLU similarly lost 1.5%. Both were removed; retaining
projection and expert parallelism is more important than reducing dispatch
count.

On Granite 3.0 1B A400M Q4_K_M, RX 9070 XT, sequential same-session tg64
runs measured:

| path | tokens/s |
|---|---:|
| all three MoE fusions disabled | 296.37 +/- 2.44 |
| all three MoE fusions enabled | 341.56 +/- 2.68 |

This is a 15.2% combined decode gain. Perplexity was identical at 3.6711 on
the same 512-token corpus. The individual retained gains measured during
development were 5.4% for weighted aggregation, 4.0% for small-K selection,
and 5.0% for router normalization. A/B controls are
`DX12_NO_FUSE_MOE_SUM`, `DX12_NO_SMALL_TOPK`, and
`DX12_NO_FUSE_MOE_WEIGHT_NORM`.

The small top-K shader sizes its group scratch for 32 waves so a 256-thread
group is safe on Intel UHD wave8 hardware.

The MMID epilogue weighted-sum fusion is default-off on Intel UHD. Granite
Q4_K_M tg128 improved from 38.3 to 43.4 tokens/s when the standalone
`moe_sum` reduction was retained instead. `DX12_MOE_WEIGHTED_SUM=1` forces
the epilogue fusion on; `=0` disables it on other devices.

Q8_0 MoE prefill must also use the expert-aware DP4A MMID path. It was
previously limited to generation-sized token counts, while Q8_0 was omitted
from the cooperative fallback. Larger graphs therefore reached the scalar
one-output-per-thread shader: Granite pp512 spent about 3.62 seconds in MMID
and ran at 139.16 tokens/s. Removing the token limit routes the same graph
through Q8_1 activation quantization plus Q8_0 DP4A, reducing MMID to about
152 ms and improving pp512 to 3082.28 tokens/s. pp6144 reaches 2963.48
tokens/s. `DX12_MOE_Q8_DP4A=0` restores the scalar route for A/B testing.
Optimized and scalar perplexity were 3.6651 and 3.6665 respectively, a 0.04%
difference and far below the reported uncertainty.

The DP4A MMID shader uses shared memory for cross-wave reduction and is also
enabled on Intel UHD wave8 hardware. Granite Q8_0 improved from about 15
tokens/s for both pp512 and tg128 on the scalar path to 78.4 and 42.7
tokens/s respectively. `DX12_MOE_Q8_DP4A=0` restores the scalar route.

RDNA1/2 Q8_0 MoE prefill uses a grouped 128x64 integer-dot GEMM once the
prompt reaches 32 tokens. The group builds the expert row map in-shader and
reuses each Q8_0 weight tile across up to 128 routed rows; smaller graphs keep
the DP4A matvec. On the RDNA2 iGPU, Granite pp512 improved from 68.48 to
405.20 tokens/s (5.92x), while pp32 improved from 63.74 to 83.87 tokens/s.
Forcing the GEMM at pp16 regressed 51.58 to 42.39 tokens/s, which established
the default threshold. Decode remains on the existing matvec route.
`DX12_MOE_Q8_MMQ=0` disables the grouped route and
`DX12_MOE_Q8_MMQ_MIN_TOK` changes the threshold.

Wave64 adapters need a wider cooperative MMID prefill tile. A 32-thread,
four-row group leaves half of a wave64 idle and reloads each activation for
too few output rows. The prefill-only shader uses one native wave and 16 rows
for F16 and Q6_K; decode retains the original kernel because the wide tile
regresses small-token graphs. Granite F16 pp512 improved from 1690.86 to
2496.05 tokens/s, while tg64 remained 309.36 versus the 310.00 baseline.
`DX12_MOE_WIDE=0` disables the prefill-only route.
The Pascal+ Tegra iGPU (GB20B, wave32) also profits despite the 16-row group
spanning a single wave: Granite F16 pp512 improved from 350 to 429 tokens/s
(+22%), pp2048 +19%, and Q4_K_M pp512 +5% from its Q6_K tensors, with decode
unchanged. Discrete NVIDIA keeps the per-element route.
One-chunk perplexity was 13.3006 versus 13.3004 with the route disabled.

Q4_K prefill on wave64 uses the block-level decoder, which unpacks each
256-element block's scales and mins once instead of repeating that work per
element. Together with the wide Q6_K path used by mixed Q4_K_M files,
Granite pp512 improved from 694.07 to 1292.45 tokens/s. tg64 remains on the
cooperative kernel and measured 339.26 versus the 339.51 baseline. One-chunk
perplexity was 13.2038 versus 13.2263 with both optimizations disabled.
`DX12_MMID_Q4K_BLOCK=0` restores the cooperative Q4_K route.

NVIDIA also uses the block-level decoder for prefill only. On RTX 6000 Ada,
Granite Q4_K_M pp512 improved from 1410 to 2035 tokens/s; decode remains on
the cooperative kernel because forcing the block decoder there reduced
tg128 from 423 to 319 tokens/s. The Pascal+ Tegra iGPU (GB20B) behaves like
Intel Xe-HPG+ rather than discrete Ada and takes the block decoder for both
phases: Granite Q4_K_M tg128 improved from 194 to 223 tokens/s (+15%) with
prefill unchanged. `DX12_MMID_Q4K_BLOCK=0` restores the cooperative decode.

`ADD_ID` now uses aligned four-float loads and stores when both row strides
permit it, with scalar fallback and tail handling. Decode-sized rows improved
from about 2.8 us to 1.8 us; 512-token cases improved by roughly 4-7%.

Qwen3.5 MTP repeats `CONT(mtp_gate) -> SIGMOID -> MUL(attention)`. The fused
shader reads the original interleaved gate view, reconstructs its
`[head_dim, heads, tokens]` coordinates, applies sigmoid, and multiplies the
attention output. Restrict matching to tensors named `mtp_gate`: the initial
shape-only matcher was too broad. Direct dumps at 8,192, 45,056, and 16,384
elements were bit-identical, two dispatches were removed per graph, and steady
MTP graph time improved by about 0.4-0.6%. Set
`DX12_NO_FUSE_MTP_GATE=1` for the unfused path.

Serial all-expert aggregation regressed Granite decode by 24%, and combining
independent gate and up MMIDs with SwiGLU lost 1.5%; both designs were
removed. Retaining expert and projection parallelism matters more than
reducing dispatch count.

## 9. Non-LLM workloads expose gates that llama.cpp never touches

`ggml-dx12` is not an llama.cpp-only backend. Every tuning decision in this
file was originally made against a transformer decoder, and that workload
mix is narrow in a way that hides whole classes of regression. Diffusion
transformers, vision encoders and image or 3D generators run the same ops
at wildly different shapes, so a gate that is "obviously fine" for LLM
inference can be leaving a large factor on the floor elsewhere.

The concrete case: `trellis.cpp` (image -> 3D, `pwilkin/trellis.cpp`) runs
a DiT whose sparse-structure flow is 4,096 dense tokens, `d_model` 1,536,
30 blocks, 12 heads, `head_dim` 128, with BF16 K/V and F32 Q. That is
about 13.7 TFLOP per forward and 22 forwards per generation.

### Dispatch count is not time

`DX12_SHADER_AUDIT=1` reported 65,935 dispatches for one stage, 88.5% of
them "generic" ops - CONT 10,949, ARANGE 2,684, SET_ROWS 2,640, RMS_NORM
2,640. The obvious conclusion is that the backend is drowning in
memory-movement dispatches and needs fusion or batching.

GPU timestamps said otherwise. Those 88.5% of dispatches were about 9% of
GPU time; `FLASH_ATTN_EXT` and `MUL_MAT` were 90%. Note also that the
audit's `MISSED-specialization` column only considers MUL_MAT, MUL_MAT_ID
and FLASH_ATTN_EXT (`op_expects_specialization`), so "generic" there means
"this op ships only a generic shader", not "this op is slow".

Get timestamps before optimising. `DX12_TUNE_PROFILE_JSON=<path>` forces
per-dispatch profiling on every graph and appends JSON lines keyed by op,
src0 type, shader flag and K/N/M. `DX12_PROFILE=1` prints a table for
gen-graphs 3-5 only.

Two cautions when reading that output:

- Absolute times are trustworthy, but verify that for your workload. Enabling
  `DX12_TUNE_PROFILE_JSON` sets `cr_eligible = false`, disabling the
  command-list replay cache, and wraps every dispatch in timestamp queries,
  so it *can* inflate and serialise. Measured on trellis it does not: the
  four flows took 244.9/88.3/709.5/425.3 s profiled against 245.9/89.6/717.3
  unprofiled, an inflation of 1.0%. Check by comparing one profiled wall
  time against its unprofiled counterpart before trusting absolute numbers.
- The `[GPU_SPAN] dispatches=N sum=X span=Y idle=Z` line answers "CPU or
  GPU bound?" directly. For this workload idle was 15-30 ms out of about
  30,000 ms, so no amount of submission batching would have helped.

### Profile every stage, not the first one

The sparse-structure flow is the obvious thing to profile: it runs first,
it is a fixed 4096 dense tokens regardless of input, and it is the only
stage that is comparable across machines. It is also not where the time
goes. The pipeline has four flows, and the two largest were the last to be
looked at:

| flow | tokens | time | FA share | MUL_MAT share |
| --- | --- | --- | --- | --- |
| sparse-structure | 4096 | 244.9 s | 37.8% | 49.3% |
| shape SLAT LR | 2048 | 88.3 s | 27.1% | 55.6% |
| shape SLAT HR | 8192 | 709.5 s | 57.6% | 33.7% |
| texture SLAT | 8192 | 425.3 s | (as above) | (as above) |

The SLAT flows reuse the same DiT with N = active voxels instead of dense
tokens. Attention is quadratic in N while the projections are linear, so
the op mix shifts with N: at 2048 tokens MUL_MAT is dominant, at 8192 it
is FA by a wide margin. Tuning against the 4096-token stage alone gives a
materially wrong priority order.

Over the whole pipeline (1572 s of dispatch time):

| op | time | share |
| --- | --- | --- |
| FLASH_ATTN_EXT | 768.0 s | 48.8% |
| MUL_MAT | 620.3 s | 39.5% |
| everything else | 184 s | 11.7% |

A single shape, `D=128 nq=8192 nkv=8192`, is 27.4% of all GPU time.

### FA runs at about half the throughput of the GEMM path

Deriving FLOP/s from the profile (`4 * nq * nkv * D * nh` for FA,
`2 * K * N * M` for MUL_MAT) separates "big because there is a lot of it"
from "big because it is slow":

| op | shape | GFLOP/s |
| --- | --- | --- |
| FLASH_ATTN_EXT | nq=nkv=8192, D=128, nh=12 | 919 |
| FLASH_ATTN_EXT | nq=8192 nkv=4352 | 949 |
| FLASH_ATTN_EXT | nq=nkv=4096 | 958 |
| MUL_MAT fl=53 | K=8192 N=1536 M=8192 | 1397 |
| MUL_MAT fl=53 | K=1536 N=8192 M=8192 | 1886 |
| MUL_MAT fl=53 | K=1536 N=8192 M=4096 | 2000 |

FA sits at roughly 950 GFLOP/s across every shape while the GEMM path
reaches 1400-2000. That gap is remarkably flat in nq and nkv, which points
at per-tile overhead rather than anything shape-dependent - the online
softmax (exp, max/sum reductions, the rescale of every accumulator), the
`GroupMemoryBarrierWithGroupSync` pairs around each KV tile, and an
occupancy limit from 32516 B of LDS at D=128.

This is the largest remaining opportunity in the backend for
attention-heavy workloads: FA is 48.8% of pipeline time at about half the
achievable rate, so closing the gap is worth roughly 24% of total GPU
time. Note the fp16 QK blob already collected part of it, and that the PV
register-tile reshape below did not, so the next attempt should start by
establishing what the kernel is actually bound by. Candidates worth
measuring, in rough order of expected value: staging V as half to cut LDS
by 8 KB and raise occupancy, amortising the softmax over a larger BC, and
reducing the barrier count per tile.

### The FA prefill fp16 QK blob was gated to Intel UHD alone

`wblob_fa_pf16_pick` defaulted the fp16 QK blob on only for
`DX12_ARCH_INTEL_UHD`. The blob changes just the QK pass: Q/K stage as
`half4` and the dot product folds into `dot2add` with an f32 accumulator.
The f32 path reads 3 LDS floats per 2 FMAs; the half4 path reads 3 vec4
per 8 FMAs. V staging, PV, the online softmax and the mask/scale/softcap/
sink math are untouched, and `FA_PF_BR`/`BC` are identical, so the two
blobs are interchangeable at the same `pf_var.br`.

Xe-HPG+ has `fp16_supported` and `dot2add` just like UHD and was simply
never measured, because llama.cpp prefill spends little time in FA - the
quadratic term only dominates once `N_q` and `N_kv` are both large. At
`nq = nkv = 4096` it dominates completely.

Results on a B390 (Xe-HPG+, wave 16, UMA):

| measurement | f32 blob | fp16 blob |
| --- | --- | --- |
| trellis SS flow, end to end | 268.8 s | 240.3 s (-10.6%) |
| FA share of SS-flow GPU time | 44% | 37.8% |

`test-backend-ops test -o FLASH_ATTN_EXT`: 5097/5097.

Because FA is a larger share of the SLAT flows than of the SS flow (see
the per-stage table below), the pipeline-wide saving is larger than the
-10.6% measured on SS alone.

### Interleave A/B runs, and check for a stale process first

The first measurement of the above put the flow at 323.7s -> 242.8s, a
-25% win. The real figure is -10.6%. The A arm had been contaminated by a
leftover `trellis-cli` process from an earlier aborted run still holding
the GPU, which inflated the baseline by about 20%.

Two habits catch this. Interleave the arms as ABBA rather than running one
arm then the other, so drift and contamination show up as within-arm
spread; the corrected numbers were A 265.8/271.7 and B 240.1/240.5, tight
enough to trust. And confirm no stale process holds the device before
starting - on Windows a `Copy-Item` of `ggml-dx12.dll` failing with "being
used by another process" is the giveaway, but a run that merely *ends*
without the handle being released gives no such signal.

Corroboration matters too. The -10.6% flow gain and the 1.109x geomean
from the perf sweep agree closely; the original -25% agreed with nothing,
which should have been the tell.

### Reading `test-backend-ops perf` when the noise floor is +/-25%

A single perf sweep showed apparent 0.75-0.81x regressions and a geomean
of 0.986x, which looks like a clear reject. It was entirely noise.

The trick is that the `flash_attn_pf_*` blob is only reachable when
`nb >= fa_tiled_min_q` (64), `hsk == hsv`, and `head_dim` is in
{64, 96, 128}. Every other case - notably all `nb=1` decode cases and
everything at `hsk=72` - runs a **byte-identical shader in both arms**.
Those cases are a free, built-in control group that reads out the harness
noise floor directly.

Splitting the sweep that way (median of 2 reps per arm):

| slice | n | geomean | min | max |
| --- | --- | --- | --- | --- |
| affected (fp16 blob in play) | 6 | 1.109x | 1.026x | 1.218x |
| control (identical shader) | 18 | 1.015x | 0.878x | 1.245x |

The control spread brackets the affected range, so the raw geomean was
meaningless. Every affected shape improved, minimum 1.026x. Always
partition a perf sweep into shapes the change can reach and shapes it
cannot, and report the geomean of each separately.

The affected slice also settled the gate's shape. The win holds for F16
K/V (1.026-1.218x) and for q4_0/q8_0 K/V (1.075-1.163x), not only the
BF16 K/V that trellis uses, so the gate keys on architecture rather than
on K/V type. llama.cpp prefill shapes gain too: `hsk=128, kv=512, nb=512`
at 1.218x and `hsk=96, nh=32, nb=512` at 1.026x.

`DX12_FA_PF_FP16=0/1` still overrides the architecture default.

### Mask prescan on Xe-HPG+

`flash_attn_pf_64_wide_prescan_relaxed_maskclass.hlsl` (flag 113) was
written for Intel UHD and gated to `DX12_ARCH_INTEL_UHD` alone. It has the
same `FA_PF_BR`/`BC` as the plain wide blob, so the two are interchangeable
at the same `pf_var.br`; the only differences are that it prescans the mask
once per query group to bound the KV range and classify each tile (fully
masked tiles skip K/V staging, finite all-zero tiles skip per-score mask
loads) and lets the QK/PV accumulators reassociate.

Xe-HPG+ was never measured, and the reason it matters here is that the
profile puts FA at 50.9% of granite-3.0-1b-a400m F16 pp6144 (227 ms of 445
ms across 24 dispatches) - MoE MUL_MAT_ID is only 35%. B390, ABBA
`llama-bench`, pp6144:

| model | prescan off | prescan on | delta |
| --- | --- | --- | --- |
| granite-3.0-1b-a400m F16 | 1118-1190 | 1252-1349 | +13% |
| SmolVLM2-256M Q4_K_M     | 3056      | 4653      | +52% |
| SmolLM2-135M Q8_0        | 3179      | 4638      | +46% |

pp512 is unchanged (2309 vs 2329 on granite) - at short KV almost every
tile is live, so there is nothing to skip. The gain scales with how much of
the mask is dead, which is exactly the long-context case. 15138/15138 and
FLASH_ATTN_EXT 5097/5097 pass either way.

Nothing in the prescan or mask-class code depends on the head dim (only the
`FA_PF_BC` default does), so `flash_attn_pf_96_prescan.hlsl` (flag 114) and
`flash_attn_pf_128_prescan.hlsl` (flag 115) are the stock 96/128 wrappers
plus the three defines, keeping `FA_PF_BR` at 16. ABBA at pp4096:

| model | head_dim | prescan off | prescan on | delta |
| --- | --- | --- | --- | --- |
| Phi-3-mini-4k Q4_K_M         |  96 | 403.2 / 405.1 | 428.8 / 426.6 | +5.8% |
| Qwen3-4B-Instruct-2507 Q4_K_M| 128 | 362.1 / 361.7 | 378.8 / 383.3 | +5.3% |

The smaller gain at 96/128 is expected: those blobs run BR=16 rather than
32, so a query group spans half as many rows and a smaller share of the KV
range is dead for the whole group.

`DX12_FA_PF_PRESCAN=0` remains the kill switch.

The NVIDIA Pascal+ iGPU (GB20B, wave 32, UMA) behaves like Xe-HPG+ here, so
the D=96/128 prescan is on for it too. Alternating-order `llama-bench`
pp6144, 5 rounds of `-r 5` each (the D=64 path is unreachable there because
`fa_pf_wide` is off at wave 32):

| model | head_dim | prescan off | prescan on | delta |
| --- | --- | --- | --- | --- |
| Qwen3-4B-Instruct-2507 Q4_K_M | 128 | 631.9 +/- 1.0 | 668.4 +/- 1.0 | +5.8% |
| Phi-3-mini-4k Q4_K_M          |  96 | 723.6 +/- 34.1 | 768.9 +/- 18.9 | +6.3% |
| Qwen3-0.6B Q4_K_M             | 128 | 2026.1 +/- 38.5 | 2116.2 +/- 7.0 | +4.5% |

Qwen3-4B won all 5 rounds with no overlap between the arms. The gain tracks
prompt length, as the dead-mask argument predicts: +5.8% at pp6144, +3.4%
at pp3072, and neutral at pp1024 (+0.5% mean, -1.6% median - noise). Prescan
also lowered run-to-run spread on every model. Perplexity is unchanged
(Phi-3 5.8975 vs 5.9005, Qwen3-4B 9.0188 vs 9.0155) and FLASH_ATTN_EXT is
5166/5166 with prescan forced on, including `DX12_FA_PF_WIDE=1` to cover the
D=64 blob that the shared arch predicate also unlocks.

AMD and discrete NVIDIA are still excluded - the blob has not been measured
there.

### Confirmed against real models

The synthetic sweep predicts, but llama.cpp prefill is what must not
regress. ABBA-interleaved `llama-bench` on a B390, discarding the first
run of each model (a cold run reads 15-20% high and will otherwise be
mistaken for a win):

| model | head_dim | test | f32 | fp16 | delta |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B-Instruct-2507 Q4_K_M | 128 | pp512 | 691.0 | 730.5 | +5.7% |
| Qwen3-4B-Instruct-2507 Q4_K_M | 128 | pp2048 | 443.9 | 471.8 | +6.3% |
| Phi-3.1-mini-4k Q4_K_M | 96 | pp512 | 755.6 | 766.3 | +1.4% |
| Phi-3.1-mini-4k Q4_K_M | 96 | pp2048 | 515.9 | 516.9 | +0.2% |
| Falcon-H1-7B-Instruct BF16 | 128 | pp6144 | 107.4 | 108.9 | +1.4% |

No regression anywhere, and the per-model gains track the head_dim
predictions from the sweep. Falcon-H1 is a hybrid Mamba model, so FA is a
smaller share of its prefill and the gain is correspondingly smaller.

Models live in `%USERPROFILE%\.cache\huggingface\hub\models--*\snapshots\*`
and can be passed to `llama-bench -m` directly, which matters because `-hf`
needs network access that may not be available.

### Reproducing

`test-backend-ops` needs the subcommand before the filter
(`test-backend-ops perf -o FLASH_ATTN_EXT`); `-b DX12` alone silently
matches nothing, because the device is named `DX120`.

Trellis vendors a patched ggml (`pwilkin/ggml`, branch `trellis-patches`)
with a larger `GGML_MAX_NAME`, so its `ggml-base.dll` and `ggml-cpu.dll`
must be left alone - dropping in a stock build fails GGUF loading with
"tensor name ... is too long". Swapping only `ggml-dx12.dll` is safe:
`name` is second to last in `ggml_tensor`, followed only by `extra` and
`padding`, so every field the backend reads sits at an identical offset,
and the backend never touches `extra`.

### Staging V as half: keep, but for the LDS not the speed

The fp16 blob originally staged only Q and K as half and left V as f32, so
at D=128 the V tile alone was 16512 B of a 32768 B budget and the whole
group needed 32516 B - i.e. 99.2% of the limit, and with a 64 KB SLM that
caps residency at one group per core. Staging V as half too (`s_vh`, flat
`float16_t` so the PV read stays a plain index) drops the group to 24324 B.

Correctness is unaffected: 5097/5097, and f32 accumulators everywhere that
matters. On the FA perf sweep it is a clear win - 1.103x geomean with every
affected shape improving (min 1.025x) against a 0.991x control.

That win does not reach any real workload. Trellis end to end went 1456.5s
-> 1461.3s and llama-bench pp512/pp2048 on Qwen3-4B and Phi-3.1-mini moved
within +/-1.3% in both directions. Every shape the sweep rewards has a
small KV (512-4096) and is cache-resident; the workloads that actually
spend time in FA do not look like that.

So it is kept for the LDS headroom, not for a speedup: the fp16 path at
D=128 had ~250 B of slack for any future FA state and now has ~8 KB. Do
not cite the 1.103x as an end-to-end number.

### Rejected: BR=32 at D=128, and the memory-traffic theory behind it

`FA_PF_BR` sets how many query rows a group owns, so it also sets how many
times each K/V element is re-read from memory: at BR=16 every element is
fetched once per 16 rows. On a UMA part that looked like the obvious bound,
and it explained the two things the profile showed - FA pinned near 950
GFLOP/s regardless of shape, and half-V (which cuts LDS but not one byte of
global traffic) doing nothing for real workloads. The freed LDS made BR=32
affordable at D=128 for the first time (~30.7 KB).

It was wrong. 5097/5097 passed, but the trellis flows went 238.0 -> 244.0s
(SS, +2.5%) and 708.0 -> 711.5s (shape-HR, +0.5%) - and shape-HR is the
one with twice the sequence length and therefore twice the traffic BR=32
was supposed to halve. The L2 is evidently already absorbing the K/V reuse
across neighbouring groups, so BR buys nothing and the wider tile just adds
register pressure (ACC goes 8 -> 16). Reverted.

Two things to carry forward. FA at these shapes is not bound by LDS size,
LDS throughput, or K/V memory traffic - three hypotheses now tested and
disproved, which leaves the online softmax itself (transcendentals, the
per-tile max/sum reductions, rescaling every accumulator) and the barrier
count as the untested candidates. And note the pattern: the FA perf sweep
has now twice predicted a win that did not appear in any real workload.
Treat it as a correctness-preserving smoke test, not as evidence; the
per-flow trellis numbers and llama-bench are the ones that decide.


The PV pass assigns each thread one output dim and `FA_PF_ACC` query rows,
so at D=128 the inner loop reads 1 V value plus 8 scores from LDS to issue
8 FMAs. Giving each thread 2 output dims and 4 rows keeps the accumulator
count identical (the budget is fixed at `FA_PF_BR * FA_D / FA_PF_THREADS`)
while cutting LDS reads per 8 FMAs from 9 to 6, with the two dims placed
`FA_D / FA_PF_DCOL` apart so each read stays a contiguous run.

It made no difference: 5097/5097 still passed, the affected perf shapes
came in at 1.008x geomean inside a +/-4% control band, and the trellis SS
flow moved 240.3s -> 239.7s. The PV pass is not LDS-throughput bound, so
the change was reverted rather than kept as unpaid-for complexity. If PV
is revisited, measure what it *is* bound by first - the QK pass, which the
fp16 blob does help, has a much worse read-to-FMA ratio.

### Whole-pipeline accounting: where the non-transformer time goes

The four DiT flows are not the whole pipeline. Timestamping every stdout
line of a full run (1812.0 s, profiled; profiling costs ~1%) accounts for
99.9% of wall time:

| stage | wall | share |
| --- | ---: | ---: |
| SLAT HR flow | 709.6 s | 39.2% |
| texture SLAT flow | 428.8 s | 23.7% |
| sparse-structure flow | 239.0 s | 13.2% |
| **decimate_qem_vk** | 111.5 s | 6.2% |
| SLAT LR flow | 88.3 s | 4.9% |
| FlexiDualGrid decode + mesh | 66.7 s | 3.7% |
| PBR decode | 60.1 s | 3.3% |
| uv_bake | 25.5 s | 1.4% |
| BiRefNet bg removal | 22.3 s | 1.2% |
| remesh_dc | 22.0 s | 1.2% |
| winding clean | 12.3 s | 0.7% |
| upsample / DINOv3 / weld / GLB / floaters | 20.5 s | 1.1% |

Rolled up: DiT flows 80.9%, mesh extraction and export 9.9%, VAE/voxel
decode 7.2%, preprocess and conditioning 1.5%.

Three things follow, and they matter for any non-LLM workload:

**About 10% of the pipeline is not ggml at all.** `decimate_qem_vk` is
trellis's own self-contained Vulkan compute port (`src/decimate_qem_vk.cpp`
plus `src/decimate_qem.comp`), and `remesh_dc` / `uv_bake` are CPU (xatlas).
None of it routes through a ggml backend, so swapping or tuning ggml-dx12
cannot move it. It is the single largest block outside the transformer and
it belongs to the application.

**GPU dispatch is 1570.8 s of the 1812.0 s wall (86.7%), and within any one
graph the GPU is saturated** - `[GPU_SPAN]` reports idle at 0.02-0.1% of
span across every submission. So there is no submission-batching or
pipelining win available. The remaining ~13% is application CPU work
between `graph_compute` calls, not backend overhead. Note this is why
BiRefNet costs 22.3 s of wall for only ~1.4 s of GPU.

**The VAE/voxel decode is the only non-transformer stage that is ours**, and
it does not look like an LLM: 63.4 s of GPU across ~2500 dispatches of a
sparse-convolution pattern (GET_ROWS gather -> MUL_MAT -> ADD accumulate).
MUL_MAT is 58% of it, ADD 26.5% and GET_ROWS 10.5%. The elementwise half is
already at the memory roof - `ADD K=256 M=246264` moves ~756 MB in 7.8 ms,
about 97 GB/s on a UMA part - so the only way to win there is to *remove*
round trips by fusing the gather/accumulate into the matmul, not to make
the kernels faster. Ceiling on that work is ~16 s, under 1% of the
pipeline, which is why it has not been done.

So the honest summary is: matrix-core access is the only *large* lever
inside ggml-dx12 for this workload, but it is not the only lever, and the
biggest single non-flow cost is not in ggml-dx12's hands at all.

### Rejected: forcing the wave-32 shader blobs

Vulkan reports `warp size: 32` on this same B390 while DXC compiles our
shaders at wave 16, which looked like an obvious untried lever - the w32
blobs are already built, so `DX12_WAVE_BLOB=32` tests it with no rebuild.

It is a clear loss. ABBA on Qwen3-4B Q4_K_M: pp512 718.8 -> 650.8 (-9.5%),
tg64 37.35 -> 36.45 (-2.4%), and pp2048 pinned at 426-429 against 470+ for
wave 16. The device reports wave 16, the blobs are tuned for the wave the
device reports, and Vulkan's warp-32 figure reflects how its own kernels
are organised rather than a mode we should be copying. Do not retry this
without a specific reason; `DX12_WAVE_BLOB` remains available for probing
a device whose reported wave is wrong.

### Where the FA gap actually comes from

Four hypotheses have now been tested and disproved: LDS throughput (the PV
reshape), LDS size/occupancy (half-V), K/V memory traffic (BR=32) and wave
width. FA sits near 950 GFLOP/s against 1400-2000 for `mul_mat_wmma_fp16`,
flat across shapes, and nothing about the kernel's memory behaviour moves
it.

The likeliest reading is that there is no large win left here on this
class of device. Our FA is a vector-ALU kernel; Vulkan's advantage on the
same GPU is `KHR_coopmat`, and that applies to its attention path as much
as to its GEMMs. Roughly half of GEMM rate is about what a hand-written
vector FA gets against a well-tuned tiled GEMM, so the ~950 GFLOP/s is
probably close to the practical ceiling without matrix-core access rather
than a symptom of a specific defect.

That makes the LinAlg/Cooperative-Vector preview path the thing that
actually moves FA on Intel, not further shader tuning. If someone does
revisit the kernel, the two untested candidates are the online softmax's
transcendentals and per-tile reductions, and the three-barrier-per-KV-tile
structure - but measure the bound before writing code, because the four
attempts above each cost a full build-and-validate cycle to disprove.

### The remaining gap to Vulkan is mostly matrix cores

Same GPU, same model, same flow: Vulkan runs the SS flow in 132.9s against
240.3s for DX12. Nearly all of it is one structural difference.

`GGML_VK_VERBOSE=1` reports for this device:

```
0 = Intel(R) Arc(TM) B390 GPU | uma: 1 | fp16: 1 | bf16: 0 |
    warp size: 32 | shared memory: 49152 | int dot: 1 |
    matrix cores: KHR_coopmat
```

`VK_KHR_cooperative_matrix` puts Vulkan's GEMMs on the XMX systolic
arrays. D3D12 on this driver exposes no equivalent - WaveMMA, Cooperative
Vector and the SM 6.10 LinAlg matrix feature all probe as unavailable - so
`mul_mat_wmma_fp16` runs on the vector ALUs at 1400-2000 GFLOP/s.

Do not read this as "the whole gap is MUL_MAT", which is what the SS flow
alone suggested when MUL_MAT looked like 48% of the time. Pipeline-wide
MUL_MAT is 39.5% and FA is 48.8%. But FA is not a separate, reachable win:
the four experiments above failed to move it, and Vulkan puts coopmat
behind its attention path as well as its GEMMs. Both halves of the deficit
lead back to matrix-core access, which is the LinAlg preview work rather
than anything tunable in the shaders today.

One further note: Vulkan reports `bf16: 0` here while DX12 reports
`bf16: yes`, so the two backends are not converting trellis's BF16 K/V the
same way - account for that in any future cross-backend comparison on a
BF16 model.

## 8. Coalescing the GEMM tile loads (SmolVLM2 vision encode, F16 prefill)

### Symptom

SmolVLM2 image encode was 2.9x slower on DX12 than Vulkan (137 ms vs 46 ms),
and DX12 was flat across f16/Q8_0/Q4_K_M while Vulkan scaled. The mmproj is
f16 in all three cases, so the vision tower weights are byte-identical - a
bottleneck that ignores weight format is not in the quantized matmuls.

### How much of the gap is reachable

Measured GEMM throughput on the vision shapes:

| shape | DX12 | Vulkan vector | Vulkan coopmat |
|---|---|---|---|
| m=3072 n=1024 k=768 | 2.01 | 3.35 | 17.4 TFLOP/s |
| m=768 n=1024 k=3072 | 2.44 | 2.82 | 17.8 |
| m=768 n=1024 k=768  | 2.07 | 2.68 | 15.6 |
| FLASH_ATTN nq=1024  | 1.23 | 1.42 | 2.15 |

The Vulkan vector column comes from `GGML_VK_DISABLE_COOPMAT=1
GGML_VK_DISABLE_COOPMAT2=1`. Note that run reports a huge wall-clock encode
(pipeline recompilation) - use the per-op GPU timestamps, not the total.

So only ~1.3x was ever reachable in the shaders; the other ~6x is XMX via
`VK_KHR_cooperative_matrix`. This also explains why other models show a
smaller gap: coopmat's lead scales with GEMM size, and typical LLM prefill
here is n=64 where Vulkan only reaches 1.3-2.5 TFLOP/s.

### The actual defect: tile_b walked the wrong axis

`mul_mat_wmma_fp16.hlsl` and `mul_mat_wmma64.hlsl` filled the B tile with

```hlsl
uint idx = flat_id * 4 + e;
uint k = idx / BN;
uint n = idx % BN;      // fast axis is N
off = global_k*nb00 + global_n*nb01;
```

Consecutive `e` walks N, which strides by `nb01` - a whole weight row. A
16-thread wave therefore touched 64 different rows to fetch 64 halves. The A
tile already walked K (stride `nb00`, contiguous) and was fine.

Swapping the mapping so each thread covers 4 consecutive K makes each group
of `BK/4` threads cover one weight row back to back. On top of that, 4
consecutive halves are 8 contiguous bytes, so the 4 scalar dword loads
collapse into one `Load2`, and `asfloat16()` replaces the f16->f32->f16 round
trip that `load_auto()` forced on a device with native 16-bit types.

Both shaders keep the original scalar loop as a fallback, guarded on
in-range, unit stride (`nb00 == esize`) and 4-byte alignment, so BF16
(esize sentinel 3), permuted src0 and K-tails are unaffected.

### Results (paired, interleaved, min of 2 per cell)

| workload | before | after | delta | paired wins |
|---|---|---|---|---|
| SmolVLM2 f16 image encode | 143.0 ms | 125.0 ms | -12.6% | 8/8 |
| SmolVLM2 encode (both shaders) | 138.7 ms | 124.0 ms | -10.6% | 6/6 |
| Phi-3 f16 pp512 | 375.6 t/s | 462.4 t/s | +23.1% | 4/4 |
| Phi-3 f16 pp512 (repeat) | 351.8 t/s | 438.7 t/s | +24.7% | 4/4 |
| Phi-3 Q4_K_M pp512 | 759.9 t/s | 760.8 t/s | +0.1% | 2/4 |

Q4_K_M is untouched because Q4_K prefill dispatches fl=127/128/129
(`mul_mat_q4k_q8_1_mmq` and friends), not these shaders. `test-backend-ops
test` stays at 15211/15211.

### Rejected: packing the B tile as float16_t4 in LDS

Storing `tile_b` as `groupshared float16_t4[BK][BN/4]` and running a packed
inner loop (one vector LDS read + 4 packed FMAs per k, mirroring Vulkan's
`dot_product` form in `mul_mm.comp`) measured **neutral to slightly worse**
once measured properly. The shader was never LDS- or ALU-bound; it was
bound on the uncoalesced global loads above. Fix the memory access before
reaching for packed math.

Retried after the coalescing fix landed, on the theory that the earlier
null result was only masked by the memory bottleneck - this time keeping
the LDS layout and packing just the accumulators (`float16_t4 tacc[TM]`,
one packed FMA per k per row). Still neutral: 446.8 vs 449.6 t/s on Phi-3
f16 pp512, splitting 2/2 across paired rounds. Source-level fp16 packing
does not buy anything on this wave-16 part; do not try it a third time.

### Why the win shrinks on long prompts

Phi-3 f16 gains ~23% at pp512 but only ~5% at pp6144 (204 -> 215 t/s).
That is Amdahl, not a defect: attention is O(n^2) in prompt length while
these GEMMs are O(n). Share of prefill time, profiled per chunk:

| op | pp512 (chunk 1) | pp6144 (chunk 26) |
|---|---|---|
| FLASH_ATTN_EXT fl=114 | 4.1% | 51.6% |
| fl=53 GEMMs | 89.7% | 45.6% |

So on a long prompt the fix applies to under half the work.

> **Retested 2026-08-18: the +23% does not reproduce.** Rebuilt both
> sides from git (`581a11755` vs its parent) with forced shader
> recompiles and confirmed-distinct DLL hashes, then measured the fl=53
> GEMM total with the profiler, ABBA, six samples per side:
> A_fixed 975.7 970.2 971.8 976.7 968.9 976.1, B_base 972.2 975.1 977.5
> 977.8 978.7 976.3 - **-0.3%, neutral**. The op harness agrees (-1.1% on
> `MUL_MAT f16 m=4096,n=512,k=14336`); model wall clock is useless here
> (293-468 t/s spread on a 7.6 GB f16 model on a UMA part).
>
> The change is kept: it is neutral, and it also turns four scalar dword
> loads into one Load2 and drops an f16->f32->f16 round trip, so the code
> is simpler. But the number above is wrong and the Amdahl story built on
> it explains a win that is not there.
>
> Two further cautions this exposed. First, the initial one-shot readings
> were A 1096 / 1017 against B 977 / 966 and looked like a 9% regression;
> those were cold-start artifacts, and everything settled to ~973 once
> warm. **Discard the first two runs of any measurement session.** Second,
> this claim was produced by the same `Copy-Item` A/B pattern that
> manufactured the phantom MoE +21% in section 13.

### Where the long-prompt headroom actually is

At pp6144 DX12 is 215 t/s against Vulkan's 567.9. Disabling coopmat puts
Vulkan's vector path at 291.9, so ~1.36x is reachable and ~1.95x is XMX.
Per-op, against the Vulkan vector path on the same shapes:

| op (per chunk) | DX12 | Vulkan vector | ratio |
|---|---|---|---|
| MUL_MAT m=16384 k=3072 | 583.9 ms | 407.5 ms | 1.43x |
| MUL_MAT m=9216 k=3072  | 342.9 ms | 233.3 ms | 1.47x |
| MUL_MAT m=3072 k=8192  | 290.8 ms | 233.8 ms | 1.25x |
| FLASH_ATTN_EXT nkv=6144 | 1507.6 ms | 1371.2 ms | **1.10x** |

FA is essentially at parity with Vulkan's vector attention, which is the
real reason the four earlier FA experiments went nowhere - there was never
much there to win. Of the ~520 ms of reachable gap, ~369 ms is still
MUL_MAT and only ~136 ms is FA. Vulkan's vector GEMM reaches 3.4-4.0
TFLOP/s here against DX12's 2.4-2.8, and it gets there with larger tiles
plus warp-level subtiling (WM/WN/WMITER/WNITER in `mul_mm.comp`), not with
packed math. That, rather than the inner loop, is the next thing to try -
and it runs straight into the unresolved BM=128 correctness failure.

### Methodology: only interleaved DLL-swap A/B is trustworthy here

This is the most important finding in this section and it invalidated two
earlier conclusions before they shipped.

Measuring variant A, rebuilding, then measuring variant B produces false
deltas of ~17% on this part, and **A/B/A ordering does not save you** - the
drift over a session is monotonic (thermal/DVFS), not alternating. The
float16_t4 experiment "won" by 17% sequentially and lost when interleaved.

The tell is a control op: `FLASH_ATTN_EXT` appeared to improve 22% from a
change that only touched a MUL_MAT shader. Under interleaving it was
identical (31.25 vs 31.17 ms).

The reliable procedure:

1. Build each variant and save `bin/Release/ggml-dx12.dll` to a side
   directory. The shader blobs are linked into that DLL, so swapping the
   file switches variants with no rebuild.
2. Alternate variants **within** each round, and pair the comparison per
   round. Report paired win counts, not pooled means.
3. Discard the first run of a variant - pipeline compilation shows up as a
   150-300 ms outlier.

Second trap: `DX12_PROFILE=1` inserts per-dispatch timestamps that serialize
dispatches. The coalescing fix showed only -2% on the profiled GEMM total
and 0% on profiled TOTAL, while unprofiled wall-clock encode improved 4.6%
consistently and 12.6% once both shaders were fixed. Use the profiler for
attribution, and unprofiled interleaved wall clock for verdicts.

### Still open

- `mul_mat_wmma.hlsl`, `mul_mat_wmma_kfull.hlsl` and the `mul_mat_*_wmma`
  quant GEMMs share the same `idx % BN` fast-axis-on-N pattern. They were
  not touched here because the live prefill paths on this part are fl=53,
  fl=105 and fl=127/128/129. Check dispatch flags before optimizing one.
- The quant WMMA GEMMs additionally re-decode the block scale per element
  (`dequant_q4k` called 4x per thread against 4 different rows). Walking K
  within a thread would let one block header serve 4 outputs.
- FLASH_ATTN_EXT fl=110 is 22.8% of the vision graph at 1.23 vs Vulkan
  vector's 1.42 TFLOP/s. Four earlier experiments failed to move it.

## 10. The F16 GEMM is ALU-issue bound, and DXIL cannot express packed fp16

Section 8 closed by naming larger tiles plus warp-level subtiling as the
next lever, on the grounds that Vulkan's vector GEMM reaches 3.4-4.0
TFLOP/s against our 2.4-2.8. Four experiments on `mul_mat_wmma_fp16.hlsl`
(fl=53) say that lever does not exist on B390. All numbers are Phi-3 f16
pp512, interleaved DLL swap, first round discarded, Phi-3 Q4_K_M pp512 as
a control op the change cannot touch:

| variant                                  | delta  | paired wins |
| ---------------------------------------- | ------ | ----------- |
| BM=128, TM=8 (32 accumulators)           | -16.4% | 0/4         |
| BM=128 at 512 threads (16 accumulators)  |  -3.0% | 0/4         |
| K-paired `uint` LDS (half the LDS loads) | -19.2% | 0/4         |
| BK=32 (half the group barriers)          | -19.4% | 0/4         |
| null test: baseline against itself       |  +0.8% | 1/4         |

The null test matters. Three unrelated changes all landing near -19% looked
like an ordering artifact, because the harness always ran the variant second
within a round. Running the baseline against a copy of itself through the
same harness returned +0.8%, so the regressions are real. Run that null test
whenever a batch of results clusters suspiciously.

Read together the four results bracket the problem:

- Growing TM/TN costs more in registers than the tile reuse returns. The
  same `acc[8][4]` shape that wins in the MoE GEMM loses 16% here.
- The same 128x64 tile at a fixed 16 accumulators is only -3%, so the
  register pressure explains the collapse - but a 2x larger tile still
  bought nothing, so global memory traffic is not the limiter either. The
  coalescing fix in section 8 already took that.
- Halving LDS load instructions made it worse, so LDS issue is not the
  limiter: the shift/mask needed to unpack two halves out of a dword costs
  more than the load it saves.
- Halving the barrier count made it worse too.

What is left is the multiply itself. The unrolled K-tile is 256 scalar
`fmul half` plus 256 `fadd half` for 128 LDS loads, and every attempt to
change the ratio around those 256 multiplies loses. The kernel is issuing
close to as many fp16 MADs per cycle as this part will retire.

### DXIL scalarizes vectors until SM 6.9

Packed fp16 would halve those 256 multiplies. It was tried twice before and
was neutral both times. The reason is not that it fails to help - it is that
it never reached the GPU. Compile the same HLSL with the same DXC and change
only the backend:

| target        | inner loop of the repro                       |
| ------------- | --------------------------------------------- |
| DXIL cs_6_6   | 2 scalar `fmul half`                          |
| DXIL cs_6_8   | 2 scalar `fmul half`                          |
| DXIL cs_6_9   | 1 packed `fmul <2 x half>`                    |
| SPIR-V cs_6_6 | 1 packed `OpFMul %v2half`                     |

DXIL was a scalar IL by design and discarded vector semantics, leaving
drivers to re-vectorize; SPIR-V keeps them at every version. SM 6.9 /
DXIL 1.9 adds native vectors and fixes it - see HLSL proposal 0030
"DXIL Vectors", whose stated motivation is exactly this. Groupshared
`float16_t2` is scalarized the same way, which is why a packed LDS layout
also cannot work below 6.9: DXC lowers a `float16_t2` groupshared load back
into two 16-bit loads.

On the real shader the payoff is visible in the IL: written on
`float16_t2`, the K-tile is 256 scalar multiplies at cs_6_6 and 128 packed
`<2 x half>` multiplies at cs_6_9.

So packed fp16 math looked worth one more attempt at cs_6_9. It was tried,
and it lost - see below. Do not retry it, and do not retry a packed LDS
layout either.

### The cs_6_9 lever is dead - measured, not assumed

The obvious follow-up was to build the GEMM at cs_6_9 and get packed fp16.
That was measured end to end on B390 and it does not work.

First, the shader model cap is a *runtime* limit, not a driver one. The
stock OS D3D12 runtime reports a highest shader model of 6.8, but loading
the preview Agility runtime (1.721.3, in `build_linalg\bin\Release\D3D12`)
reports 6.9 on the same driver. So cs_6_9 blobs do load and do create
pipelines here.

Second, and decisively, packed fp16 is *slower* than scalar fp16 on this
part. An FMA-only ALU microbenchmark (`fp16pack-repro/alu_bench.*`), where
both variants retire the same 16 fp16 elements per iteration and the DXIL
was verified to contain 16 scalar vs 8 packed FMAs:

| groups | scalar `half` | packed `float16_t2` | speedup |
|---|---|---|---|
| 128  |  8.5 TFLOP/s |  6.8 TFLOP/s | 0.79x |
| 512  | 13.7 TFLOP/s |  9.6 TFLOP/s | 0.70x |
| 2048 | 14.6 TFLOP/s | 10.1 TFLOP/s | 0.69x |

The control explains why: scalar fp32 FMA on the same harness reaches
4.0 TFLOP/s, so scalar `half` at 14.6 is already running ~3.6x fp32. The
Intel driver is evidently already extracting packed (and better) rate out
of scalar 16-bit DXIL, so DXIL's scalarization costs nothing here, and
writing explicit vectors only constrains the driver into a worse schedule.

Conclusion: do not pursue cs_6_9 for packed math on this hardware. The
DXIL-vs-SPIR-V difference above is real at the IL level, but on this
driver it has no performance consequence. It may still matter on a vendor
whose compiler does not re-vectorize.

### What this says about the GEMM

fp16 FMA peak on this part is ~14.6 TFLOP/s, and the GEMM sustains
2.4-2.8. It is therefore nowhere near a hardware math limit - the limiter
is the surrounding instruction mix (LDS issue, address arithmetic,
barriers, and wave-16 occupancy under register pressure), not fp16
throughput. Note that every structural change tried above moved work
between those categories and lost, which points at a balanced issue mix
rather than one dominant stall. Any further attempt should start from a
measured issue-slot breakdown, not from another tile-shape guess.

### Not the BM=128 failure

The "unresolved BM=128 correctness failure" referred to elsewhere is the
MoE one, already root-caused to a missing `SHADER_INCLUDE_DEPS` entry (see
section on `mul_mat_id_gemm`). The dense F16 GEMM at BM=128 is correct:
greedy output was byte identical to baseline on both a >=256 token Phi-3
prefill and the SmolVLM2 encode. It is simply slower.

Note that `test-backend-ops` cannot cover fl=53 at all - the shader is
gated on `ne[0] >= 256 && ne[1] >= 256` and the MUL_MAT test shapes top out
far below that. Verify this shader by diffing greedy model output against
the baseline DLL, using a prompt long enough to reach the gate.

## 11. SM 6.9 is reachable on B390, and what it is worth

The shader-model cap is a *runtime* limit, not a driver one. Probed with
`fp16pack-repro/caps_probe.cpp` against the preview Agility runtime
(1.721.3, in `build_linalg\bin\Release\D3D12`):

    === Intel(R) Arc(TM) B390 GPU ===
      shader model            : 6.9      (6.8 on the stock OS runtime)
      LinearAlgebra (CoopVec) : tier 0   <- not supported
      WaveMMA tier            : 0        <- not supported
      wave lanes              : 16..32 (total 3072)
      int64 shader ops        : yes
      native 16-bit ops       : yes

Two things follow.

**Cooperative Vector / LinAlg is genuinely unavailable here.** The banner's
"CV: no" was never a device query - `cooperative_vector_supported` is
hardcoded false outside the LinAlg preview build - so it proved nothing.
Queried properly, `D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT` returns tier 0
even on SM 6.9, and WaveMMA is tier 0 too. There is no matrix-engine path
for this part through D3D12 today, at any shader model. Re-run the probe
on new drivers before assuming that is still true.

**Nothing else in SM 6.9 is a lever here.** Native 16-bit and wave ops are
already supported and already used. Int64 is irrelevant to these kernels.
SER is ray tracing. Long vectors are a coding convenience that lowers to
the same scalar ops (see section 10) and, on the one path where explicit
vectors do survive to the IL, they measured *slower*. That leaves
Cooperative Vector as the only SM 6.9 feature that would have mattered,
and it is tier 0.

### Forcing 32 lanes on the F16 GEMM: no gain

`WaveLaneCountMax` is 32, the GEMM has no wave intrinsics, and it carries
no `[WaveSize]`, so the driver chooses the SIMD width - and chooses 16.
Since section 10 showed the kernel is limited by instruction issue rather
than fp16 math, 32 lanes should retire the same work in half the
instructions. It does not help:

| measurement | result |
|---|---|
| Phi-3 f16 pp512 wall clock, run 1 | +7.0% median, 6/8 wins |
| Phi-3 f16 pp512 wall clock, run 2 | -4.7% median, 0/8 wins |
| SmolVLM2 f16 pp2048 + control     | target +1.0%, control +3.4% |
| `DX12_PROFILE` fl=53 dispatch time | +1.4% slower, 1/4 wins |

Greedy output was byte identical, so the attribute is correct - just not
faster. Note how badly the first two disagree: Phi-3 f16 pp512 swings
320-460 t/s on this part (a 7 GiB working set on a UMA iGPU), which is far
more than the effect being measured. The third run is the tell - the
control moved *more* than the target, which can only be drift.

### Use the profiler, not wall clock, for single-shader changes

`DX12_PROFILE=1` with `DX12_PROFILE_PROMPT=1` prints per-dispatch ms per
op. Summing the fl=53 rows gives a direct measurement of just this shader,
with an 8% run-to-run spread against ~40% for end-to-end wall clock.
Serialization inflates the absolute numbers, but the A/B ratio is sound
and it is the right tool for any change scoped to one kernel.

### Where the prefill time actually goes

From the same profile (SmolVLM2 f16 pp2048, 86.3 ms/graph):

| op | ms | share |
|---|---|---|
| FLASH_ATTN_EXT fl=113 | 42.5 | 49.3% |
| MUL_MAT fl=53 (3 shapes) | 32.5 | 37.7% |
| MUL_MAT fl=105 | 5.2 | 6.0% |
| everything else | 6.1 | 7.0% |

At long context FA dominates, and it grows with nkv while the GEMMs do
not. FA was previously deprioritized on the basis of being only 1.10x off
Vulkan at short prompts; at pp2048 it is the single largest consumer and
is the better target than further GEMM tuning.

## 12. FA prefill: what the PV pass is and is not bound by

Section 11 identified FLASH_ATTN_EXT as 49.3% of a pp2048 prefill graph.
The op mix for fl=113 explains where that time is not going: QK already
runs on `dot2add` (packed fp16 MAC, f32 accumulate), but the PV pass was
scalar f32, and it is the larger of the two.

Per KV tile, per thread, at D=64 BR=32 BC=32:

| pass | MACs | LDS loads | pipe |
| --- | --- | --- | --- |
| QK | 128 (64 dot2add) | 80 | fp16 |
| PV | 256 | 288 | f32 |

PV is 2x the MACs of QK at a 1:0.9 MAC-to-LDS-load ratio, so it looked
bound by either the f32 pipe or the LDS traffic. It is bound by neither.

### Rejected: probabilities in fp16 so PV can use dot2add

Staging the post-softmax probabilities in a `float16_t` LDS tile lets PV
pair columns onto `dot2add`, halving the MAC count. Measured **12.6%
slower** on FA, 0/4 wins, control flat.

An isolation build kept the fp16 probability tile but left the MACs
scalar f32, so the only difference from base is the width of the LDS
access. That measured **13.2% slower** - i.e. the entire regression is
the 16-bit groupshared access, and `dot2add` was mildly positive
underneath it.

**Scalar 16-bit LDS loads are substantially more expensive than 32-bit
ones on this part.** This is the same direction as the packed-fp16 ALU
result in section 10 and explains why half-width LDS experiments keep
losing. It does not apply to `float16_t4` tiles (s_qh/s_kh), which are
64-bit elements and are fine.

### Rejected: two output dims per PV thread (register tiling)

Giving each PV thread FA_PF_DV=2 output dims amortizes each s_scores read
over 2 MACs, cutting LDS loads per tile from 288 to 192 for the same 256
MACs, with no numerical change. Measured **4.0% slower**, 0/4 wins.

So PV is not LDS-load bound either. Combined with the BR=64 result
(section 11) this is the third tile/layout change to regress: the FA
kernel is not limited by tile shape, LDS width, or LDS traffic.

### Accepted: reverse the query-group order (-1.9%, 4/4 wins)

Under a causal mask the KV range a query group covers grows with q_start,
so with `q_start = gid.x * BR` the heaviest group is dispatched last and
leaves a long tail. At nq=512 BR=32 that is 16 groups whose work ranges
from 1 to 64 KV tiles. Walking the groups backwards launches the heavy
ones first.

Purely a scheduling change - each group computes the same rows, so output
is bit-identical. The gain is bounded by how much of the dispatch is tail
rather than steady state.

### Accepted: half-precision exp in the softmax (-1.7%, 4/4 wins)

`exp()` is not free relative to the MACs: each thread does QK_PER=4 exps
per KV tile against 512 MACs, and exp2 is a multi-instruction sequence.
Computing it as `(float)exp((float16_t)(sc - max))` lowers to a native
f16 exp2 (4 of the 11 exp2 ops in the shader, the hot-loop ones) while
the value stays in a register and is still stored to the f32 s_scores
tile - so this gets the fp16 ALU win without paying the 16-bit LDS cost
above.

p is in [0,1] and is accumulated in f32, so the added error is ~5e-4
relative. FLASH_ATTN_EXT passes 5097/5097 and greedy output is unchanged.

### Combined

FA op time **-4.5%, 4/4 wins** vs HEAD (profiler, SmolVLM2 f16 pp2048).
Full suite 15211/15211.

End-to-end wall clock could not resolve this: pp6144 spread 3058-4200 t/s
run to run. FA is ~half the graph, so -4.5% on FA is ~-2% overall, well
under that noise floor. Use the profiler harness, not llama-bench, for
changes of this size.

## 13. MoE GEMM B-tile coalescing: no effect, and a phantom +21%

Granite was the last model far off Vulkan. granite-3.0-1b-a400m Q4_K_M
pp512 on B390:

| build | pp512 | vs DX12 |
| --- | --- | --- |
| DX12 | 1631 | - |
| Vulkan, coopmat off | 4870 | 3.0x |
| Vulkan, coopmat on | 7127 | 4.4x |

Only 1.46x of that is coopmat/XMX. A 3.0x gap on the vector path is far
worse than the ~1.3x dense models show. Profiling pp512 put MUL_MAT_ID
fl=122 at 80% of the graph running ~1.15 TFLOP/s, while the dense Q4_K
GEMM (fl=127) in the same graph ran ~4.4 TFLOP/s.

### The hypothesis

`mul_mat_id_gemm.hlsli` fills the B tile with the fast axis on N:

    const uint b_n = flat_id % BN;                    // 64 different rows
    uint k = e2 * (THREADS / BN) + flat_id / BN;      // and k strides by 4

Consecutive threads read consecutive output features, which are separate
weight rows `nb01` apart, so a 16-wide wave touches 16 unrelated rows per
load. This looks like the bug fixed in the dense GEMMs by "dx12 : walk K
when filling the GEMM B tile" (claimed +23% on Phi-3 f16 - but see the
retest in section 9, which shows that change is neutral too), which was
applied to
`mul_mat_wmma_fp16` and `mul_mat_wmma64` only and never reached the MMID
shader.

The obvious fix is B_TPR = BK / B_PER_THREAD threads per weight row, each
walking B_PER_THREAD contiguous k:

    const uint b_n  = flat_id / B_TPR;
    const uint b_k0 = (flat_id % B_TPR) * B_PER_THREAD;

### Measured: nothing

Built from git on both sides with forced shader recompiles, DLL hashes
confirmed different, interleaved:

| metric | fixed | base | delta |
| --- | --- | --- | --- |
| pp512 wall clock, 4 interleaved rounds | 1549.7 | 1567.3 | -1.1% |
| profiler fl=122, 3 runs each | 193.7 ms | 187.5 ms | +3.3% |

Both inside the noise band. The change was reverted; a neutral rewrite of
a working load path is churn.

Why it does nothing here needs no special explanation any more: the dense
"walk K" fix it was modelled on was retested on 2026-08-18 and is neutral
as well (-0.3%, six ABBA samples; see section 9). Neither B-tile rewrite
changes throughput on this part. The coalescing story that motivated both
was never validated - what the numbers actually support is that these
GEMMs are not bound by weight-load transactions at all.

### The real lesson: how a +21% appeared that was never there

The first pass at this reported +20.9% on Q4_K_M, and it was written up
here as a win. It was not real. The sequence:

1. Edit the hlsli, build, save `fix.dll`.
2. `Copy-Item` the edited hlsli aside to `_ab\`.
3. `git checkout --` the hlsli, build, save `base.dll`.
4. `Copy-Item` the saved hlsli back over the source.

Step 4 is the trap. `Copy-Item` carries the *source* file's timestamp, so
the restored hlsli was older than the shader blobs produced by the step-3
build. Every later build considered those blobs current and never
recompiled. The tree, `git show HEAD`, and the commit all contained the
fix while the DLL contained base code - and the two labelled DLLs in the
A/B could not be trusted either.

What made it survive review: MUL_MAT_ID stays 790/790 and greedy output
stays byte identical, because only the thread-to-element mapping changes.
Correctness checks cannot catch this class of change at all, in either
direction.

Rules that follow, for any shader A/B:

- Never restore a shader with `Copy-Item`. Use `git checkout <rev> -- <path>`,
  which writes a fresh timestamp.
- Force the timestamp before every build: set `LastWriteTime` to now.
- Grep the build log for the `Compiling HLSL shader: <name>` line of the
  exact variant under test and confirm it appears. For Granite that is
  `mul_mat_id_gemm_tall_q4k`, not `mul_mat_id_gemm`.
- Hash the two DLLs and confirm they differ.
- Interleave A/B/A/B and never compare across separate sessions.

Also worth pinning: single-shot `DX12_PROFILE=1` op timings on this part
are far noisier than they look. Across three runs of the identical
binary, fl=122 ranged 129.9 - 199.2 ms. Any profiler claim needs repeats
in both orders; one number per side is how a 30% swing gets mistaken for
a result.

### What is left on Granite

fl=122 is still ~80% of the graph at ~1.15 TFLOP/s against the dense
GEMM's 4.4, and the gap is still unexplained. The A-tile reuse idea (with
BN=64 and N=512 each activation tile is re-read 8 times) collides with
the register wall from section 10: BN=128 gives TN=8, and with TM=8 at
the tall BM=128 that is 64 accumulators per thread, measured at -16.4% on
the dense GEMM. Note the ~2 MB activation matrix likely fits in LLC, so
this is L2 rather than DRAM pressure and the bandwidth-bound story is not
confirmed. Start from a measured issue-slot breakdown, not another
tile-shape guess.

## 14. MoE GEMM routing: gate on pairs per expert, not token count

Section 13 chased the MoE GEMM's low throughput inside the shader and
found nothing. The problem is not in the shader - it is that the shader
is being selected for shapes it cannot win.

`test-backend-ops perf -o MUL_MAT_ID -p type_a=q4_K` turns out to be a
far better harness than a model benchmark here: no model load, no DVFS
drift between sides, and it sweeps token counts directly. It shows a
cliff that a model bench hides completely (n_mats=128, n_used=8, m=768,
k=2048):

| n | time | GFLOPS |
| --- | --- | --- |
| 8 | 0.48 ms | 417 |
| 32 | 23.98 ms | 33.6 |
| 128 | 27.9 ms | 115 |
| 512 | 31.5 ms | 410 |

50x the time for 4x the work at n=32, then near-constant time out to
n=512. Constant time against growing work means a fixed cost, and n=16
is exactly `DX12_MOE_GEMM_MINTOK` - the cliff is the GEMM route
switching on.

### Cause

The dispatch is sized by the worst case:

    groups_y = ceil_div(n_tokens, BM);   // a token could pick any expert
    groups_z = n_expert;

Every expert is launched deep enough to hold *all* tokens, but the work
an expert actually receives is `n_tokens * n_used / n_expert`. With 128
experts and top-8 that is a 16x oversized launch, and the tiles that get
no pairs still cost a dispatch and a pair scan. The old gate
(`n_tokens >= 16`) never looked at `n_expert` or `n_used` at all.

Forcing the matvec route with `DX12_MOE_GEMM=0` confirms it:

| n_mats / n_used | n | pairs/expert | GEMM | matvec |
| --- | --- | --- | --- | --- |
| 128 / 8 | 32 | 2 | 21.2 ms | 2.2 ms |
| 128 / 8 | 128 | 8 | 28.1 ms | 8.8 ms |
| 128 / 8 | 256 | 16 | 28.3 ms | 17.1 ms |
| 128 / 8 | 512 | 32 | 30.5 ms | 33.5 ms |
| 32 / 4 | 32 | 4 | 14.1 ms | 2.5 ms |
| 32 / 4 | 256 | 32 | 19.1 ms | 17.9 ms |
| 32 / 4 | 512 | 64 | 25.5 ms | 35.5 ms |

Two independent shapes cross over at the same place: **32 pairs per
expert**. Below it the matvec route wins, by 9.6x at the extreme; above
it the GEMM route wins, by 1.4x at 64 pairs.

### Fix

Add the missing term to the gate, with `DX12_MOE_GEMM_MINPAIRS`
(default 32) to retune without a rebuild:

    pairs_per_expert = (n_tokens * n_used) / n_expert;
    ... && pairs_per_expert >= gemm_min_pairs

### Measured

Op level, routing now picking the better kernel at every point:

| case | before | after | gain |
| --- | --- | --- | --- |
| 128 experts, n=32 | 21.2 ms | 2.24 ms | 9.5x |
| 128 experts, n=64 | 29.2 ms | 4.45 ms | 6.6x |
| 128 experts, n=128 | 28.1 ms | 8.54 ms | 3.3x |
| 128 experts, n=256 | 28.3 ms | 16.8 ms | 1.69x |
| 128 experts, n=512 | 30.5 ms | 28.3 ms | GEMM kept |
| 32 experts, n=32 | 14.1 ms | 2.40 ms | 5.9x |
| 32 experts, n=512 | 25.5 ms | 25.4 ms | GEMM kept |

End to end on granite-3.0-1b-a400m Q4_K_M (32 experts, top-8), A/B in one
binary via the env knob so there is no rebuild and no stale-blob risk:

| | old gate | new gate | delta |
| --- | --- | --- | --- |
| pp32 (8 pairs/expert) | 270.2 | 560.2 | +107% |
| pp64 (16) | 506.2 | 640.0 | +26.5% |
| pp128 (32) | 830.8 | 830.0 | unchanged |
| pp512 (128) | 1636.1 | 1630.1 | unchanged |

The two unchanged rows are the point: the gate drops the GEMM only where
it was losing. test-backend-ops 15211/15211. Generation stays coherent;
wording can diverge from the GEMM route after a few tokens because the
two kernels accumulate in a different order.

Note this leaves pp512 on Granite where it was - section 13's gap is
still open. The launch is still sized by n_tokens rather than by pairs
per expert, but that costs less than first assumed: the shader already
reads a per-expert prefix sum from `temp` and returns immediately when
`row_base >= cnt`, so surplus tiles are launch overhead only, not work.
The real waste is *inside* a tile - a group always computes all BM rows,
so an expert holding `cnt` pairs still pays BM. That is what the numbers
show: at n=32 with 128 experts and BM=64 the executed work is
n_expert * BM * m * k = 25.8 GFLOP against 0.8 GFLOP of real work, which
at 21 ms is 1.23 TFLOP/s - the ~1.15 TFLOP/s actually measured. So the
lever is a smaller BM, not indirect dispatch. With the gate above the
surviving GEMM regime is 32 or more pairs per expert, where the worst
padding is 2x (32 pairs into BM=64); a BM=32 variant would close it.

### Method note

A model benchmark would never have found this. Granite at pp512 is
completely unaffected, and pp32/pp64 are not shapes anyone benches. The
op harness sweeps the parameter that mattered (`pairs_per_expert`) while
a model pins it to one value per model.

### 14a. BM=32 for the surviving GEMM band: tested, slower

The padding model above says an expert holding 32 pairs pays for 64 rows,
so a BM=32 tile should halve the executed work in the 32..63 band. Built
one (q4_K, flag 123) and measured it at exactly that point - q4_K,
128 experts, top-8, n=512, which is 32 pairs per expert:

| | BM=64 | BM=32 |
| --- | --- | --- |
| run 1 | 28.73 ms | 32.22 ms |
| run 2 | 28.52 ms | 30.62 ms |

About 10% slower, consistently, in both directions of an ABBA. Every
other n was unchanged, as expected, since only this shape lands in the
band.

So the arithmetic model is right about the wasted FLOPs and wrong about
what limits the kernel. Halving BM doubles the group count over the same
N, and each group re-reads the whole B tile, so the saved padding is
bought back in weight traffic - and with half the rows there is less
work per group to hide the load latency behind. 790/790 either way.

Reverted. Combined with sections 9 and 13, that is now three separate
attempts to make this GEMM faster by reasoning about memory traffic or
wasted FLOPs, all neutral or negative. The evidence says the kernel is
not limited by either, and the next attempt should start by finding out
what it *is* limited by rather than by another tile-shape guess.

## 15. The n=2..8 hole: NUM_COLS matvecs existed and were switched off

Sweeping `test-backend-ops perf -o MUL_MAT` across batch sizes, rather
than benchmarking a model, exposes a gap between the n=1 matvec and the
n>=16 GEMM. At m=4096, k=14336:

| n | q4_K | GFLOPS |
| --- | --- | --- |
| 1 | 302 us | 389 |
| 2 | 3734 us | 45.7 |
| 8 | 3756 us | 147 |
| 512 | 13785 us | 4360 |

Two rows cost 12x one row, and n=2 through n=8 is a flat plateau - the
same constant-time signature as section 14. Running the n=1 matvec twice
would take ~600 us, so the generic path is 6x worse than doing the work
twice.

Lowering `DX12_MM_GEMM_MINTOK` to 2 makes it worse (q4_K n=2 3688 ->
4587 us), so the GEMM is not the answer and the existing gate is right.

### Cause

NUM_COLS=2/4/8 dp4a matvec shaders (flags 47-52) already exist for Q4_K,
Q5_K, Q6_K and Q8_0. They were written for speculative decoding, gated
behind `DX12_Q4K_DP4A_NC2` and friends as opt-in, and never turned on.
Each gate matches an exact `ne[1]` (2, 4 or 8), so nothing outside those
batch sizes can be affected.

Switched to the same tri-state arch default the GEMM routes use: on for
Intel Xe-HPG+, `=0` opts out, other vendors keep the opt-in until someone
benchmarks them.

### Measured

Op level, m=4096, k=14336:

| type | n=2 before | after | n=4 before | after | n=8 before | after |
| --- | --- | --- | --- | --- | --- | --- |
| q4_K | 3734 us | 299 us | 3752 us | 505 us | 3756 us | 846 us |
| q5_K | 3987 us | 387 us | - | - | - | - |
| q6_K | 3154 us | 461 us | - | - | - | - |
| q8_0 | 3151 us | 603 us | - | - | - | - |

Q4_K at n=2 now costs what n=1 costs (299 vs 300 us) - two tokens for the
price of one.

### 15a. Extending NC4/NC8 to Q8_0

Only Q4_K had NC4/NC8 blobs, so Q8_0/Q5_K/Q6_K still fell off the cliff
at n=4 and n=8. Q4_K got its extra widths cheaply because its shader
loops over NUM_COLS; the other three hand-unroll their two columns, so
the width is welded in.

Lifted the Q8_0 body into `mul_mat_vec_q8_0_dp4a_nc.hlsli` with the
column loop restored and NUM_COLS supplied by a wrapper, matching the
wrapper pattern the MMID GEMM variants already use, then added nc4 and
nc8 wrappers (flags 124/125). At m=4096, k=14336, ABBA:

| n | off | on | off | on |
| --- | --- | --- | --- | --- |
| 4 | 3159 us | 627 us | 3781 us | 644 us |
| 8 | 3261 us | 749 us | 3523 us | 728 us |

5.3x and 4.5x, and n=8 now runs at 1.50 TFLOP/s against 199 GFLOP/s for
the n=1 matvec - eight columns for about the cost of one, which is what
reusing the decoded weights across columns should buy. n=1, 2, 3, 5 and
512 are unchanged, as expected. 1145/1145.

### 15b. Q5_K and Q6_K

Same refactor applied to Q5_K and Q6_K (flags 126/132 and 133/134):

| n | q5_K off | q5_K on | q6_K off | q6_K on |
| --- | --- | --- | --- | --- |
| 4 | 5264 / 6076 us | 645 / 679 us | 4965 / 4022 us | 500 / 566 us |
| 8 | 6073 / 6122 us | 945 / 981 us | 3599 / 5252 us | 624 / 756 us |

About 8.5x at n=4 and 6.4x at n=8 for both, ABBA. n=1 is unchanged, and
the existing NC2 path is unaffected by the refactor - q5_K 394 us against
387 before, q6_K 451 against 461. 1145/1145.

All four dp4a types now cover n=1, 2, 4 and 8.

### 15c. n=3, 5, 6, 7: round up to the next width

The widths are 2, 4 and 8, so odd batches still took the generic path -
q8_0 was 592 us at n=4 but 3621 us at n=3. Building three more widths per
type is the wrong answer; routing n=3 to the width-4 blob is the right
one, and the only thing preventing it was that the shaders read and write
every column they compute, which would run off the end of both the
quantized activation buffer and the destination.

`ne11` is already a shader constant, so the fix is two `break`s - one in
the column loop and one in the store loop. Columns past `ne11` are
skipped rather than clamped, so a short batch does not even pay for them.
The four per-type gates collapsed into one that rounds `ne[1]` up to the
next available width.

| | n=3 off | n=3 on | n=5 off | n=5 on |
| --- | --- | --- | --- | --- |
| q4_K | 3780 / 6194 us | 440 / 443 us | 5085 / 6078 us | 685 / 689 us |
| q8_0 | 3721 / 3683 us | 760 / 787 us | 3451 / 3848 us | 954 / 810 us |

Gemma-4 E2B Q4_K_M end to end:

| | off | on | delta |
| --- | --- | --- | --- |
| pp3 | 22.57 t/s | 82.26 t/s | 3.6x |
| pp5 | 37.40 t/s | 107.99 t/s | 2.9x |
| pp6 | 45.80 t/s | 119.64 t/s | 2.6x |
| pp7 | 53.03 t/s | 137.21 t/s | 2.6x |

The eval suite already covered these widths (25-26 cases each for n=3, 5,
6 and 7), which is why the routing change could be made with confidence.
15211/15211.

Every batch from 1 to 8 now has a fast path for all four dp4a types. A
short batch costs its rounded-up width in launch geometry but not in
work, so n=5 lands between n=4 and n=8 as expected rather than at n=8.

End to end, Phi-3-mini-4k Q4_K_M, A/B in one binary via the env knobs:

| | off | on | delta |
| --- | --- | --- | --- |
| pp2 | 9.52 t/s | 63.35 t/s | 6.7x |
| pp4 | 18.87 t/s | 37.17 t/s | 2.0x |
| pp8 | 37.22 t/s | 65.26 t/s | 1.8x |

test-backend-ops 15211/15211.

These are exactly the shapes speculative decoding verification and
multi-sequence batch decode run at, and nothing in the usual pp512/tg128
benchmark set touches them - which is why a 12x hole sat in the routing
table unnoticed. Worth sweeping batch size on every op that has more than
one kernel behind it.

### 15d. NUM_COLS width 16 closes the n=9..15 hole; width 32 is model-dependent

Closing n=3,5,6,7 exposed the next cliff immediately above it. Gemma-4
E2B Q4_K_M, llama-bench, NC ladder capped at 8:

| | pp8 | pp9 | pp12 | pp15 | pp16 |
| --- | --- | --- | --- | --- | --- |
| t/s | 145.6 | 65.3 | 88.5 | 109.3 | 114.0 |

Nine tokens took 2.2x longer than eight. Converting to total time shows
what is really happening: 0.055 s at n=8, 0.138 s at n=9, 0.140 s at
n=16. The whole 9..15 band was already paying the full fixed cost of the
n>=16 GEMM. Lowering DX12_MM_GEMM_MINTOK to 9 changed nothing (65.6 vs
65.3), confirming the band was on that path already rather than being
kept off it by the threshold.

Adding width 16 to the ladder (flags 135/136/137/138) fixes it. ABBA in
one binary via the env knobs:

| model / type | n | off | on | delta |
| --- | --- | --- | --- | --- |
| Gemma-4 E2B Q4_K | 9 | 66.3 | 87.4 | +32% |
| Gemma-4 E2B Q4_K | 12 | 87.9 | 103.8 | +18% |
| Phi-3 Q8_0 | 9 | 53.4 | 158.7 | 3.0x |
| Phi-3 Q8_0 | 12 | 70.8 | 180.7 | 2.6x |
| Phi-3 Q8_0 | 16 | 93.2 | 185.7 | 2.0x |
| SmolLM2-135M Q6_K | 9 | 1090 | 1433 | +31% |
| SmolLM2-135M Q6_K | 12 | 1393 | 1705 | +22% |
| SmolLM2-135M Q5_K | 12 | 620 | 680 | +10% |

pp20 was unchanged (115.3 vs 115.9), confirming the change is isolated to
n <= 16. No regression was found on any model or type. Default on.

Width 32 is a different story and is **opt-in only**. At n=16 the width-16
matvec beat the GEMM 2.0x, which suggested pushing the ladder into the
GEMM's own territory. On Phi-3 that works:

| Phi-3 | n=20 | n=24 | n=25 | n=28 | n=30 | n=32 |
| --- | --- | --- | --- | --- | --- | --- |
| Q8_0 GEMM | 115.6 | 137.4 | 143.1 | 157.8 | 168.8 | 473.2 |
| Q8_0 NC32 | 191.9 | 201.1 | 201.0 | 207.0 | 209.6 | 209.2 |

n=32 is the one loss, and the reason is visible in the table: the GEMM
tile is 32 wide, so n=32 fills it exactly and jumps to 473 t/s while
every non-multiple below it wastes most of the tile. Hence the ladder is
capped at n<=31 and n>=32 goes to the GEMM.

But the same experiment on Gemma-4 E2B Q4_K inverts:

| Gemma-4 E2B Q4_K | n=20 | n=24 | n=31 |
| --- | --- | --- | --- |
| GEMM | 151.6 | 176.8 | 210.0 |
| NC32 | 111.4 | 122.1 | 125.8 |

-27% to -40%. The obvious hypothesis was register pressure from 32
accumulators making the heavier K-quant decode spill, i.e. a per-type
effect. That is wrong: Q4_K on *Phi-3* gains from NC32 (98.9 -> 113.4 at
n=20, +15%). Same type, same shader, opposite sign - so the deciding
factor is the model's shapes, not the quant type. Whichever of the two
kernels happens to suit a given model's N and K wins, and neither
dominates.

Shipping width 32 on by default would therefore mean a 40% regression on
Gemma-class models to buy 15-66% on Phi-3-class ones. It is gated behind
an explicit DX12_<type>_DP4A_NC32=1 instead; widths 2..16 keep the normal
tri-state default-on. Verified that unset and =0 measure identically
(138.0/166.8 vs 138.4/166.7) so the opt-in gate does not perturb the
default path.

Two process notes. A default-run measurement came in 8% below a baseline
taken earlier the same session (138 vs 151); an immediate A/B of unset vs
=0 showed them identical, so it was thermal drift, not routing - the
"never compare across sessions" rule earned its keep again. And the
reduction loop now breaks at ne11 as well as the column and store loops,
so a short batch on a wide blob does not pay for the columns it skips.

test-backend-ops 15211/15211. Eval covers the new band at n=9 (25 cases),
n=12 and n=16 (313).

## 16. Upstream merge, and the M=1 invariant the NC matvecs quietly broke

Merged 792 upstream commits (merge base a646006f0). Three groups of the
new upstream tests failed, and only one of them was upstream's fault.

**MUL_MAT_VEC_FUSION (18 cases).** These were ours. The matvec post-op
fusions - bias ADD, GLU, and the Q/K/V projection bundle - all gate on
`is_matvec_dispatch`, which historically meant "M == 1" because that was
the only way a matvec shader got selected. The NUM_COLS work broke that
equivalence: NC sets `is_matvec_dispatch` for M = 2..31. The bias path
then skipped the ADD node (`skip_count += 1`) on the promise that the
shader would apply the bias, but the NC shaders never read a bias at all,
so it was dropped for every column. Errors were only 0.01-0.05, small
enough that no existing test caught it and no benchmark looked wrong.

Four of the seven fusion gates already tested `ne[1] == 1` explicitly;
three did not, and those three are exactly the three that failed. Fixed
by restoring the invariant at the gate rather than teaching the NC
shaders about bias - the fusion only ever made sense for a single column.
Cost is ~1-2% at n=9..12 on models that were silently getting a wrong
fused bias; correctness is not optional.

Lesson: `is_matvec_dispatch` is a *routing* predicate, not a shape
predicate. Any new gate that depends on the output being one column must
say `node->ne[1] == 1` out loud. Widening a shader's shape coverage
silently widens every predicate downstream of it.

**ROPE (18 cases).** Upstream added `n_offs` (the channel where the
rotated window starts) at `op_params[15]` - a slot our backend had
claimed for `has_ff`, on the since-invalidated comment that "op_params[15]
is unused by ggml ROPE". The shader cbuffer is fixed at op0..op15, so a
new slot would touch every shader; instead slot 15 now packs `n_offs` in
the low bits and `has_ff` in bit 31, mirroring what the matvec RoPE
fusion already does with head_dim/has_ff in op10. `rope.hlsl` and
`rope_multi.hlsl` implement the shifted window (both the rotation indices
and the passthrough index mapping, which must now skip a hole in the
middle rather than a suffix). The five fused rope paths do not implement
it and decline when `n_offs != 0`.

**SSM_SCAN (3 cases).** New `K > 1` rollback scan wants K intermediate
state slots in the output tail; our shader only writes the final state.
Declined in `supports_op`.

test-backend-ops 17200/17200. ABBA vs the pre-merge build across Phi-3
Q4_K_M / Q8_0 and Gemma-4 E2B Q4_K_M: all metrics flat (pp512 +0.46%,
tg128 +0.07% on the tightest run).

One measurement note, since it nearly produced a false bug report. An
early sweep showed Phi-3 Q4_K_M at pp512 -14% and tg128 -9.6%, with the
post-merge arm erratic (662/719) while the pre-merge arm was tight
(803/806). That asymmetry looked like a real instability. It was the RDP
session tearing down mid-run. Re-measured on a quiesced machine, both
arms sat inside 769-777 and the delta vanished. Asymmetric variance
between arms is not automatically evidence of an asymmetric cause - the
arm that happens to occupy the disturbed slots absorbs the disturbance.


## 17. Closing the op coverage gaps the upstream merge opened

The 792-commit sync brought in new ops and widened the test matrix, so
`test-backend-ops support` grew a long tail of unsupported cases. Most
of that tail is training-only (OUT_PROD, the `*_BACK` family,
CROSS_ENTROPY, OPT_STEP) or belongs to models we do not target
(LIGHTNING_INDEXER, DSV4_HC_*), and was deliberately left alone. What
follows is the inference-relevant work and the traps it turned up.

Landed: SWIGLU_CLAMP, EXPM1, PAD_REFLECT_1D, COL2IM_1D, F16/BF16-source
SET_ROWS, dequantizing CPY/DUP across 25 quant types, byte-level
CONCAT/REPEAT for I16/I64/quant, mixed K/V-type FLASH_ATTN_EXT, and
GATED_LINEAR_ATTN. Full suite after: 17856/17856.

### The BF16 esize sentinel

`load_auto`/`store_auto` take an `elem_stride` where 2 = F16,
**3 = BF16**, 4 = F32. Three is a type tag, not a byte count - BF16's
physical stride is 2. Any shader that computes its own byte offsets as
`idx * esize` must map 3 to 2 first. Shaders that go through
`offset_4d` with the `nbXX` byte strides never see this. COL2IM_1D
hit it and produced garbage only on BF16.

### Never derive blocks-per-row from nb01 / nb00

The obvious way to get blocks per row in a block-copy shader is
`nb01 / nb00`. That is wrong the moment src0 is a **view**: the row
stride is padded and the ratio overshoots. Derive the block size from the
(contiguous) destination instead:

`
blck  = ne0 / (nb1 / nb0);
bpr_0 = ne00 / blck;
`

This was the entire cause of the `v=1` / `v=3` CONCAT failures.

### uint16_t shaders must live in exactly one CMake list

`uint16_t` needs `-enable-16bit-types`, so `concat_block` and
`repeat_block` belong in `DX12_SHADERS_FP16_ONLY`. Putting them there
*and* in the general shader list compiles them twice and fails the build.
The host must also gate dispatch on `dx12_device::fp16_supported`.

### The cheapest way to add a per-element dequant op

`shaders/quant_dequant.hlsli` exposes
`float mmid_dequant(ByteAddressBuffer, uint row_off, uint k)`, selected
by defining exactly one `MMID_<TYPE>` macro plus `MMID_QK` and
`MMID_BLOCK_SIZE` - 25 types available. `get_rows_quant.hlsli` and the
new `cpy_quant_f32.hlsli` are thin generic bodies over it, so each
per-type wrapper is two lines. Reach for this before writing a new
decoder.

### Mixed K/V flash attention: runtime dispatch beats variant explosion

The FA quant path compiled one macro-selected `mmid_dequant` and used it
for **both** K and V, so supporting `kt != vt` naively meant a 6x6
variant blowup. Instead `shaders/quant_dequant_kv.hlsli` provides
`kvq_dequant(buf, row_off, k, type_id)` with a runtime switch over ids
1..8 (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL/Q1_0/Q2_0); id 0 is the float path
via `load_auto`. One extra shader (`flash_attn_kvmix.hlsl`) covers the
whole matrix.

All 16 FA `op_params` slots were already spoken for. The free space is
in `op_params[8]`: bit 0 is has_mask, bits 8-15 mask nb0, bits 16-23
mask esize, bit 24 has_sinks - leaving bits 1-7 and 25-31. K's type id
went in bits 1-5, V's in 25-29.

### FA pipeline-override ordering

FA dispatch picks the quant wrapper first, then a **small-D override**
(`flash_attn_64` / `flash_attn_128`) can replace it whenever
`!kv_is_quant_fa && head_dim <= 128`. `kv_is_quant_fa` originally
inspected only `src[1]` against the six legacy quants, so an F16-K /
Q4_0-V mix - or any Q1_0/Q2_0 cache - would have been silently routed to a
float-only shader. **Any new KV type has to be reflected in both the
wrapper selection and `kv_is_quant_fa`.** `fa_pf`/`fa_tiled` were
already safe (gated on `fa_tiled_type`), as was `fa_coop` (checks both
src1 and src2).

### GATED_LINEAR_ATTN is wkv6 with a transposed state

`gla.hlsl` is structurally `wkv6.hlsl`: one workgroup per
`(batch, head)`, `groups_x = H * B`, 64 threads, each thread holding
its own state column in a `float state[64]` register array, with k/q/g
staged through groupshared per token and barriers on both sides.

The one thing to get right is the index convention. In the CPU reference
`i` indexes k/q/g and `j` indexes v/dst, and state is `[i][j]` - so
thread `tid` owns `j` and the inner loop reads
`state[i * head_size + tid]`. That is the transpose of how wkv6 is
arranged. Per token:

`
s = state[i] * g[i] + k[i] * v_val;   y += s * q[i];   state[i] = s;
`

with `scale` folded into `q` at load. Note `y` uses the
**post-update** state, unlike wkv6. Because each workgroup owns a whole
sequence, loading `src4` once at the top and keeping the state in
registers reproduces the CPU's "state_prev is src4 only at the first token
of the sequence" rule for free.

Host-side, GLA needs adding to four separate hand-maintained lists: the
op-params list, `gdn_or_ssm` (baked-in src2+ VA offsets), `needs_src4`,
and the dispatch group-count switch. Missing any one of them fails
silently or page-faults rather than erroring at build time.

### NUM_ROWS is now a knob on the NC matvecs

The NC (NUM_COLS) matvecs hardcoded two output rows per group, with the
two rows written out as unrolled `row0`/`row1` locals. Upstream Vulkan
tunes rows as a function of column count, so Q4_K's
`mul_mat_vec_q4k_dp4a_nc.hlsli` was rewritten to loop over `NUM_ROWS`
and the define moved behind `#ifndef`, letting each wrapper pick its own
value. Default stays 2, which is bit-identical to the old code
(MUL_MAT 1215/1215 across the refactor).

NUM_ROWS has nothing to do with cooperative vector / LinAlg. These are
scalar dp4a matvecs on the decode path and never touch a matrix unit;
the knob trades registers and groupshared against activation reuse. The
only connection is that the machines worth re-measuring it on happen to
be the same discrete parts that also have LinAlg hardware.

Two things to know before turning the knob. Row clamping matters: with
2 rows the shader read `row0 + 1` unconditionally and got away with it
on buffer padding; the generic version clamps to `ne0 - 1`. And the host
must agree - `dx12_matvec_rows4()` gates the row-group arithmetic in
**two** places (the `matvec_row_groups` computation and the
`rows_per_group` chunking ternary). Disagreement there silently skips or
double-writes output rows.

Groupshared cost is `NUM_ROWS * NUM_COLS * 64` floats, so raising rows is
only viable at the narrow widths - NC32 at 4 rows would want 32 KB of LDS.
The theory for why narrow widths should benefit is that the q8 activation
loads are shared across rows while the weight decode is not, so at
NUM_COLS=2 the activation traffic per row-pair is at its worst.
Sweeping it on B390 did not pay off. ABBA (warm-up discarded) on
`test-backend-ops perf -o MUL_MAT`, q4_K m=4096 k=14336, us/run:

| n   | shader | rows=2       | rows=4       |
| --- | ------ | ------------ | ------------ |
| 2   | NC2    | 308.0, 345.4 | 334.9, 351.3 |
| 3   | NC4    | 437.3, 393.3 | 388.5, 381.6 |
| 4   | NC4    | 422.5, 347.6 | 409.2, 407.1 |

n=3 separates cleanly in favour of 4 rows (both samples below both
rows=2 samples, -7%), but n=4 runs the *same shader* and goes the other
way. Within-arm spread reaches 20% - larger than any between-arm
difference - so there is no defensible win here and the default stays at
2. Worth noting the rows=4 arm is consistently the *tighter* of the two
(407-409 vs 348-423 at n=4); the extra rows appear to cost occupancy in a
way that damps scheduling variance without improving throughput.

The knob is left in place because it is free at the default and the
answer is likely architecture-dependent - a discrete GPU with more
registers and real VRAM bandwidth is the obvious place to re-run this.
To make that re-run cheap, the 4-row variants ship as their own blobs
(flags 143/144, `mul_mat_vec_q4k_dp4a_nc{2,4}_r4.hlsl`) behind an
opt-in environment variable rather than requiring a rebuild:

`
DX12_Q4K_NC_ROWS4=1
`

Unset (the default) routes Q4_K n=2..4 to the 2-row NC2/NC4 blobs exactly
as before. Set, it routes them to the 4-row blobs. Full suite is
17856/17856 either way.

Verifying a knob like this needs care. The obvious negative control -
deliberately mismatching the host row-group count against the shader -
proves nothing here, because the shader clamps out-of-range rows and the
host dispatching *too many* groups just recomputes correct values
redundantly. Both arms pass. The test that actually discriminates is a
temporary `fprintf` in the routing branch: zero hits with the variable
unset, hits reporting flag=143/144 with it set.

Testing status: N1X (Arm/Tegra) validates the branch with no regressions.
RDNA4 and RTX 5080 are the interesting targets for this knob - both have
working LinAlg, more registers, and real VRAM bandwidth, which is exactly
the regime where trading occupancy for activation reuse should behave
differently than it does on an Xe3 iGPU.
The dual-normalization and concat MTP prefix costs only about 7-8 us per graph;
a four-resource dual-reduction fusion was deferred because its measured
ceiling is below 1%. Q2_K and Q3_K LinAlg MMID were also deferred: both require
new packed scale and data decoding, not just shader registration.

Validation: `MUL_MAT_ID_FUSION` 13/13, `ARGSORT` 98/98, `ADD_ID` 36/36,
`GET_ROWS` 111/111, and 15209/15209 complete DX120 backend tests. Ordinary
`GET_ROWS` must continue to dispatch `ceil(nelements / 256)` groups; a
one-group default caused 107 failures before it was corrected.

## 10q. LinAlg specification audit and direct K transpose

The version 0.9 runtime specification and HLSL proposal 0035 expose more than
the original backend assumed. Wave and threadgroup matrices support row-major
and column-major loads, element access, casts with optional transpose, and
accumulator stores. The API still has no tensor-layout decode callback, native
row reduction, masked matrix load, or asynchronous matrix-load primitive.

The useful missed operation is `Matrix::Cast<..., Transpose=true>()`. For a
full aligned D=128 F16 KV tile, flash attention now loads K row-major directly
from the tensor buffer as an A matrix, casts and transposes it to a B matrix,
and feeds it to QK. This removes the global-to-LDS transpose without relying
on a direct column-major descriptor load. The numerical column-major failure
described in section 10i was measured on AMD; this load pattern was not tested
directly on RTX 5070. D=96 remains on the staged path because the direct cast
was slightly slower there. Partial and quantized KV cache tiles also retain
staging.

The direct column-major descriptor load was subsequently tested on RTX 5070.
It is correct (5097/5097 tests), but measured 1926.79 +/- 5.19 tokens/s versus
1928.72 +/- 2.65 for row-major load plus transpose cast. The cast path is
retained because the two are effectively tied and it is the slightly faster
same-session result.

On the cached Qwen3-4B-Instruct-2507 F16 model at pp6144, the change improved
1920.11 tokens/s from a roughly 1755 tokens/s baseline. Final measurement after
the D=96 gate was 1921.50 +/- 2.23 tokens/s. Q8_0 and Q4_K_M model files use
the same default F16 KV cache and improved to 1797.14 and 1690.52 tokens/s.
`FLASH_ATTN_EXT` passes 5097/5097.

The `DX12_LINALG_CAPS=1` diagnostic now matches the installed Agility 1.721.3
`_1` query ABI. It uses datatype values SINT32=4 and UINT32=5, queries wave
support with WaveSize=0, probes exact threadgroup shapes, and reports
construction, outer-product, and atomic-accumulate support. RTX 5070 reports:

- F16 wave multiply to F16 or F32 at 16x16x16, 16x16x8, and 16x8x8.
- S8 wave multiply to S32 at 16x32x16 and 16x32x8.
- Threadgroup F16 multiply only at 16x16x16, with 32-64 threads and no
  preferred group size.
- F16 to F32 outer product and F32 atomic accumulate to buffer or groupshared
  memory.

The accepted 64x128 threadgroup shader is therefore not a native large
threadgroup operation according to this driver query. It is composed from the
same 16x16 operation, which explains why matching Vulkan's large coopmat2
source shape did not provide Vulkan-like throughput. The published `_2`
specification says larger integer-multiple application shapes should report
supported, but the installed `_1` driver exposes only the exact native shape.

Production routing now consumes this query rather than treating it as a
diagnostic. The backend compiles 64-, 128-, and 256-thread variants and
prefers the reported thread count when available, otherwise the largest
compiled count in range. RTX 5070 uses a validated 64-thread
NVIDIA fallback because its driver does not advertise 64x16x128. Strict
capability mode disables that fallback. A 32-thread variant was rejected
because optimized DXIL materialized its 64x128 threadgroup accumulator with
`alloca`; the other production variants, the 128x128 NVIDIA wave GEMM, and
D=64/D=96 attention have no allocas.

An alternating pp6144 sweep on Qwen3-4B F16 measured 1975.7 tokens/s with 64
threads, 1923.3 with 256, and 1623.5 with 128. The 64-thread fallback is 2.72%
faster than 256 for this workload. `DX12_LINALG_TG_THREADS=64|128|256` can
override only the selected compiled variant for device characterization; it
does not bypass the capability or strict-capability route gates.

The characterization suite records the exact threadgroup query beside every
case. On RTX 5070, the advertised 16x16x16 tuple accepts 32-64 threads but
direct descriptor loads produce numerical mismatches. Conversely, the
unadvertised 64x16x128 LDS-staged cases execute correctly at 32-256 threads.
This is why successful PSO creation is retained as evidence but is no longer
sufficient for production routing.

## 10r. Transposed QK orientation

The Vulkan cooperative-matrix attention path computes `K * Q^T` and stores the
result transposed. The DX12 D=64 and D=96 NVIDIA variants now use the same
orientation. Q is loaded once as a column-major B fragment, each K tile is
loaded directly in its canonical row-major cache layout as an A fragment, and
the accumulator is stored column-major. The resulting LDS score layout is
unchanged, so masking, online softmax, and PV reuse the existing code.

This removes the per-KV-tile K transpose through LDS without requiring a
second cache layout. Alternating pp6144 measurements on RTX 5070 were:

| model and head dimension | existing QK | transposed QK | change |
|---|---:|---:|---:|
| SmolLM2 F16, D=64 | 19581 | 20840 | +6.4% |
| Phi-3 Q8_0, D=96 | 2134 | 2285 | +7.1% |
| Qwen3-4B Q8_0, D=128 | 1836 | 1829 | -0.4% |

D=128 retains its row-major load plus transpose-cast path. Set
`DX12_FA_QK_TRANSPOSED=0` to restore the previous D=64/D=96 orientation.
Validation: 5097/5097 `FLASH_ATTN_EXT` tests on DX120.

A persistent transposed K cache was rejected. The canonical cache is written
by SET_ROWS and several fused RoPE paths, consumed by flash and non-flash
attention, shifted in place, and serialized. Maintaining a second layout would
duplicate K-cache memory and synchronize every update and maintenance path.
The transposed QK orientation already obtains direct matrix loads from the
canonical cache, leaving no data-path benefit to offset that complexity.

Aligned eight-element F16 loads were also tested in the LinAlg GEMM staging
path. They passed 1146/1146 `MUL_MAT` tests, but pp6144 changed by less than
0.1% on SmolLM2, Phi-3, and Qwen3 F16. The existing register prefetch hides the
load-width difference, so the four-element loads were retained.

## 10s. D=128 eight-wave output ownership

The original NVIDIA D=128 path split the output dimension across two
workgroups. This kept 16 output values per lane but duplicated QK, mask, and
online-softmax work. A four-wave single-group path avoided the duplication but
needed 32 output values per lane and the driver rejected or severely
underperformed it.

The retained variant uses eight waves in one workgroup. Four extra waves
repeat the legal QK matrix operations into unused LDS slots, then all eight
waves divide the complete output while keeping 16 output values per lane.
This avoids divergent control flow around LinAlg operations and computes the
mask and softmax once.

The decisive follow-up was reducing the wave-private PV staging from two LDS
slots per wave to one. The store and read are ordered within one wave, and the
loop-tail group barrier still protects reuse as score storage. This reduces
the D=128 temporary score/PV allocation from 16 KB to 8 KB.

Qwen3-4B F16 pp6144, alternating measurements on RTX 5070:

| variant | pp6144 |
|---|---:|
| two output groups, two PV slots | 1985.7 |
| eight waves, two PV slots | 2031.3 |
| two output groups, one PV slot | 2283.8 |
| eight waves, one PV slot | 2451.4 |

The final-tile D=128 attention time fell from 268.97 to 174.83 ms. The
four-wave single-group, one-slot variant still failed PSO creation with
`E_OUTOFMEMORY`, confirming that its output-register pressure is independent
of the LDS reduction.

Branching the eight-wave shader so only the four score-owning waves execute
QK would remove the duplicated matrix work, but the RTX 5070 driver also
rejects that PSO with `E_OUTOFMEMORY`. The uniform eight-wave QK sequence is
therefore required for this driver, not just a conservative control-flow
choice.

The eight-wave one-slot variant also improved every tested prompt size:

| prompt | previous | eight-wave one-slot | change |
|---|---:|---:|---:|
| pp128 | 1832 | 1884 | +2.8% |
| pp256 | 2923 | 2969 | +1.6% |
| pp512 | 3822 | 3961 | +3.6% |
| pp1024 | 3538 | 3786 | +7.0% |
| pp2048 | 3082 | 3440 | +11.6% |
| pp4096 | 2417 | 2857 | +18.2% |
| pp6144 | 1980 | 2441 | +23.3% |

## 10t. D=64 and D=96 one-slot PV staging

The NVIDIA transposed-QK D=64 and D=96 variants also benefit from one wave-private PV staging slot. The store and read remain ordered within the wave, and the loop-tail group barrier protects reuse as score storage. This reduces the temporary score/PV allocation without changing output ownership or duplicating matrix work.

RTX 5070 F16 pp6144 and final-tile attention results:

| model | head dimension | previous pp6144 | one-slot pp6144 | change | attention before | attention after |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | 64 | 20969 | 23476 | +12.0% | 22.45 ms | 17.97 ms |
| Phi-3 mini | 96 | 2651 | 3068 | +15.7% | 170.39 ms | 119.67 ms |

The final production build measured 23396 t/s for SmolLM2 and 3064 t/s for Phi-3 at pp6144. Decode controls were 875.8 t/s and 75.8 t/s, respectively. The complete prompt sweep was:

| prompt | SmolLM2 | Phi-3 |
|---|---:|---:|
| pp128 | 19840 | 2410 |
| pp256 | 29368 | 3575 |
| pp512 | 36294 | 4612 |
| pp1024 | 34450 | 4404 |
| pp2048 | 31851 | 4103 |
| pp4096 | 27346 | 3511 |
| pp6144 | 23396 | 3064 |

Three follow-ups were rejected:

- Eight D=64 waves regressed pp6144 from about 23400 to 21200 t/s because the extra waves duplicated QK work without eliminating an output split.
- Eight D=96 waves required uneven output ownership and failed 28 `FLASH_ATTN_EXT` cases. Exact output-tile divisibility and V-prefetch divisibility remain shader invariants.
- SmolLM2 split-KV targets from 256 through 1536 groups were swept. The existing 512-group target remained best at about 23475 t/s.

The production change only compiles the existing NVIDIA D=64/D=96 transposed-QK blobs with `FA_PV_SLOTS=1`; no new flags or runtime gates are required. Validation passed 5166/5166 `FLASH_ATTN_EXT` tests and 18106/18106 complete DX12 backend tests. Both shaders use 128 threads, contain no `alloca`, and require 11140 bytes of LDS for D=64 and 13188 bytes for D=96.

---

## 18. Merging the LinAlg branch into the mainline backend

`dx12-linalg-phase0` and the mainline backend both grew for ~140 commits
without a shared base. Merging them was mostly mechanical, but two classes
of conflict are worth recording because they will recur on the next merge.

### The flag id space is the real conflict, and git cannot see it

`key.flags` is a flat integer namespace shared by every shader-selection
site. Both branches allocated from it independently, so the two sides
collided on 107-126 and 130-153: the mainline had NC matvecs and the tiled
quant GEMM there, the LinAlg branch had its wave-matrix GEMM tiles.

Git merged most of those sites cleanly because they live in different
functions - the duplicate `case` labels only surfaced at compile time, and
the range predicates (`key.flags >= 107 && key.flags <= 126`) would not
have surfaced at all. A silently wrong range predicate routes a correct
kernel through the wrong group-count arithmetic, which shows up as a
partially written output, not as a crash.

Resolution: the LinAlg GEMM ids were shifted by +100 (107-126 -> 207-226,
130-153 -> 230-253, 161 -> 261). The offset is deliberate - the selection
code derives the tile shape with `(flags - base) % 4`, so a uniform shift
preserves the arithmetic and the mapping stays checkable by eye:

    207 f16   211 q8_0  215 q4_K  219 q5_K  223 q6_K
    230 q5_0  234 q4_0  238 q4_1  242 q5_1  246 iq4_nl  250 mxfp4

Sites that had to move together: the blob switch, `la_base`, the two
`linalg_took` range checks, `linalg_gemm` (bias fusion), the
`linalg_gemm_dispatch` root-constant predicate, and the GEMM group-count
branch. If a future merge shifts these again, grep for all six.

**That audit was incomplete, and the follow-up commits say how.** The +100
shift covered the GEMM ids but missed two other LinAlg allocations: the FA
tiles at 34-42, which overlap the mainline FA range 20-34 at 34, and the
NVIDIA FA flags 114-118, which overlap the mainline 107-115. Both were
later moved to 166-179 behind named constants
(`DX12_LINALG_FA_F16_BASE` and friends) rather than raw literals, which is
the better habit - a named base makes the allocation greppable and the
`+0/+1/+2` offsets self-documenting.

Shifting ids upward has a second-order cost that is easy to miss: flags
above 255 no longer fit a `uint8_t`. The replay cache stored `key_flags`
in one, so 261 and 262 truncated. It is now `uint32_t`, matching
`key.flags` itself. **When renumbering, grep for every field the flag is
stored in, not just every site that compares it.**

### Two MoE prefill GEMMs, and neither one wins everywhere

Both branches independently concluded that MoE prefill is bound by
re-reading each expert once per routed token, and both wrote a tiled GEMM
to fix it - the mainline a dequant-to-LDS tile (flags 119/122), the LinAlg
branch a wave-matrix tile (200/202). They are not redundant: the LinAlg
route is gated on `linalg_matrix_supported`, so it is unreachable on parts
without a matrix unit, where the mainline route is the only option.

Both were kept, with the LinAlg route decided last so it wins on hardware
that has the unit. That ordering is a guess, not a measurement - nothing in
the lab can run both. Re-check it on RDNA4/Blackwell before trusting it.

### Device selection: upstream superseded a fork-local fix

The fork carried `common_model_select_device()` so mtmd would follow the
model's device on multi-adapter boxes; upstream has since added an explicit
`--mmproj-device`. The merge keeps both, explicit first:

    mparams.device = params.mmproj_device
                       ? params.mmproj_device
                       : common_model_select_device(...);

Dropping the fallback would silently put the vision encoder on adapter 0
whenever the flag is not passed, which is the exact bug the fork-local fix
was written for.

### What this merge does and does not verify

Gate: `test-backend-ops test` = 17876/17876, 2/2 backends, both with and
without `DX12_Q4K_NC_ROWS4=1`. The count is +20 over the pre-merge 17856
because the LinAlg branch added `NORM`/`RMS_NORM`/`L2_NORM` cases at 8192.
`test-dx12-autotune`, `test-dx12-q6k-hang`, `test-col2im-1d`, `test-rope`
and `test-alloc` also pass.

That gate does not cover the LinAlg kernels at all. B390 reports `CV: no`
and SM 6.8, so every LinAlg path is dormant, and the default build compiles
the `GGML_DX12_LINALG_PREVIEW` blocks out entirely. To at least prove the
renumbering compiles, configure a second tree with the preview enabled and
build the `ggml-dx12` target.

Note that a preview build currently fails at DXIL validation on this
machine:

    error : Declaration '%dx.types.LinAlgMatrixC9M16N16U2S1 = type { i8* }'
            uses a reserved prefix.
    error : Pointers to pointers, or pointers in structures are not allowed.

That is the DXC version, not the code - the LinAlg GEMM shaders need a
newer validator than the DXC 1.10.2605.2 staged here (CMakeLists already
documents 1.10.2605.24 as the floor). Passing `-Vd` on the `cs_6_10` rules
compiles the tree far enough to type-check the host code, which is how the
+100 renumbering above was verified; it is not a substitute for running the
kernels on hardware that has a matrix unit.
### Re-merging after either side moves

Both source branches kept advancing after the initial merge. Pulling them
in again produced 18 conflicts in `ggml-dx12.cpp` and 6 in `TUNING.md`,
every one of them spurious: the LinAlg branch had merged `origin/main`
into itself, moving the merge base, so git re-presented phase0's original
LinAlg flag numbering against the +100 renumbering on this side.

The check that settles it is cheap - confirm the other branch adds no
content this side is missing:

    git log --oneline HEAD..origin/main            # 0 commits
    git merge-base --is-ancestor <old-phase0-tip> HEAD
    git show --stat --name-only <the one real new commit>

When the only genuinely new commit touches an unrelated directory (here,
`tools/linalg-bench/`, which merged cleanly on its own), resolving the
backend source and TUNING.md with `--ours` is correct, and a diff of the
new directory against the source branch confirms nothing was dropped.
Hand-merging those 18 hunks would have risked reintroducing the flag-id
collision this section exists to warn about.

The upstream side contributed `GGML_OP_LIGHTNING_INDEXER`, whose only
conflict was the `needs_src2` / `needs_src3` binding predicates - a plain
union with phase0's `fused_norm_bias_node` term. Those 156 cases moved
from "not supported" to passing, taking the gate from 17876 to 18032.

## 19. Flat-index shaders must survive the 65535 group limit

A D3D12 dispatch dimension caps at 65535 groups, so any 1D elementwise
shader whose element count exceeds `65535 * 256` has to spill into
`groups_y`. Reading `tid.x` alone then silently reprocesses the first
65535 groups' worth of elements and never touches the rest - wrong
output, no error, and invisible unless a test actually crosses the
boundary.

The fix is `flat_idx_2d_256()` in `ggml_common.hlsli`, which folds the y
dimension back into a flat index. Every flat-index shader in the tree now
uses it; `expm1`, `pad_reflect_1d` and `cpy_quant_f32` were the last three
still on bare `tid.x`. The audit is a one-liner worth re-running whenever
a shader is added:

    Select-String -Path *.hlsl,*.hlsli -Pattern "^\s*uint\s+idx\s*=\s*tid\.x\s*;"

It should return nothing. `test-backend-ops` now pins the boundary with
EXPM1, CPY q4_0 and PAD_REFLECT_1D cases at ne=[256, 65535..65536].
Note the general shape of this bug: correctness tests that stay under the
limit pass forever while the defect sits in the tree.

## 20. First hardware LinAlg run: the tier is not a capability

The Intel LinAlg driver (32.0.101.8974) plus DXC 1.10.2605.24 and Agility
1.721.3-preview made the wave-matrix path executable on real hardware for the
first time. It executed and was wrong: 439 failures out of 18106.

    FLASH_ATTN_EXT   403 failures, ERR ~0.05
    MUL_MAT_ID        36 failures, ERR ~0.93, all at n=129

The failure shape was misleading. FA failures clustered hard on nb=75 (392 of
403) and included hsk=72, a head dim that cannot even enter the LinAlg gate,
which pointed at cross-test corruption. It was not corruption. Chasing the
shape was the wrong move; the capability query answered it in one step.

### Root cause

`linalg_matrix_supported` was set from `LinearAlgebraTier != 0` alone. The
tier says the feature exists, not that any particular operation, component
type, or shape is implemented. Every wave-scope shader we ship
(`flash_attn_linalg`, `mul_mat_linalg_f16`, `conv_linalg`, `out_prod_linalg`)
asks for f16 x f16 -> **f32** accumulator at **16x16x16**. This part reports:

    wave f16xf16 -> f16   flags=0x1  shapes: 8x16x16
    wave s8xs8   -> s32   flags=0x1  shapes: 8x32x16
    threadgroup matrix multiply: (nothing)

An f16 accumulator at 8x16x16 - neither the accumulator type nor the shape we
compile. The PSO still builds and the dispatch still runs. It just returns
wrong numbers. `tools/linalg-bench/README.md` warned about exactly this:
a case "can create a PSO yet return incorrect results on some driver and
shape combinations".

### The fix

`dx12_query_linalg_wave_shape()` asks operation 1 (WaveMatrixMultiply) for the
exact triple we compile and checks the returned shape list for 16x16x16.
`linalg_wave_f16_16x16_supported` gates the five wave-scope dispatch sites.
The banner reports it, so a device that advertises the tier without the shape
now says so instead of failing silently:

    LinAlg: yes tier=16 no-wave16x16 ... CV: no

Result on B390: 18035/18035 with LinAlg fully enabled, no kill switches.

`nv_linalg_mmq` and `wmma_fp16_auto` still read `linalg_matrix_supported`.
Both are NVIDIA-gated threshold tweaks rather than wave-scope dispatches, and
NVIDIA reports 16x16x16 f32, so they were left alone.

### Method notes

Two env kill switches settled the biggest question in one run. With
`DX12_FA_LINALG=0 DX12_LINALG_MMID=0` the preview tree passed 18106/18106,
which proved the new DXC was not miscompiling ordinary shaders and confined
the fault to LinAlg. Discriminate before diagnosing.

A reboot after the driver install changed nothing - the mismatch was
bit-identical (126976 mismatches, max_rel 4.75, same first element). Cheap to
rule out, worth ruling out.

`tools/linalg-bench` is the right instrument here because it checks every
case against a CPU reference, separating a driver bug from a shader bug. Two
traps: its default `-Dxc` points at a version that may not be installed, and
the `quick` profile hardcodes wave 32, so on a wave-16 part nearly everything
reports `unsupported` for the wrong reason. Use `-Profile full`, and read the
capability dump (`DX12_LINALG_CAPS=1`) before drawing conclusions from it.

Generalised: a tier, a version, or a successful PSO creation are all weaker
claims than "this operation at this shape with this accumulator". Query the
thing you are about to dispatch.

## 21. What the Intel Xe3 matrix engine can actually do

> CORRECTION: the groupshared-load finding below is WRONG - the probe was
> miscoded. See section 35. The descriptor-side findings still hold.

Section 20 stopped at "the shapes we ship are not the shapes this device
implements". This section is the follow-up measurement: given that the device
only offers `s8 x s8 -> s32` at 8x32x16 and `f16 x f16 -> f16` at 8x16x16, what
can be built on it? The answer is more restrictive than the capability dump
suggests, and the restriction is not in the multiply.

Everything below is from four probes that run one wave and check against a
hand-computed reference. They live next to the other probes in `shaders/` and
run under `tools/linalg_repro_host.cpp`:

- `linalg_repro_i8_coord_intel.hlsl` - does GetCoordinate agree with Store
- `linalg_probe_i8_rows.hlsl`        - groupshared operand load
- `linalg_probe_i8_desc.hlsl`        - descriptor operand load
- `linalg_probe_i8_align.hlsl`       - alignment the descriptor load requires

### GetCoordinate works here, and the layout is trivial

We had been treating `GetCoordinate()` as unusable everywhere, on the strength
of `linalg_repro_getcoordinate.hlsl`. That repro is real, but it is an *AMD*
result: RX 9070 XT, driver 32.0.23041.2023. Carrying it over to Intel was an
assumption, not a measurement, and it was wrong.

On B390 at 8x32x16, GetCoordinate agrees with Store on all 128 cells, and the
mapping is about as friendly as it gets at wave 16:

```
lane = column,  element index = row
lane0: e0=(0,0) e1=(1,0) e2=(2,0) ... e7=(7,0)
lane1: e0=(0,1) e1=(1,1) e2=(2,1) ... e7=(7,1)
```

So an accumulator can be drained straight out of registers. The LDS round trip
that the AMD result forces is not needed on this part. Check per-vendor before
assuming either way.

### The multiply is exact

With operands fed correctly, `C[r][c] = 32*(r+1)*(c+1)` came back exact across
the whole 8x16 tile, corner included (r7c15 = 4096). s32 accumulation means
results stay bit-comparable with the dp4a path - no tolerance argument needed.

### Groupshared operand loads are broken

`Matrix::Load(groupshared T Arr[], ...)` does not read the array. Every
configuration returned the same answer - 127*127 per accumulated term, i.e. both
operands filled with 0x7F - and the result did not move when the input moved,
when the probe operand switched from A to B, or when Stride changed.

That last part is what makes it a driver defect rather than a convention we
guessed wrong. A wrong stride or a wrong layout enum gives *different* wrong
answers; an answer that ignores its input entirely is not reading it. Note the
accumulator's groupshared `Store` works fine in the same dispatch, so this is
specific to loading operands.

Do not spend time deducing the packed-I8 stride units from this path. The
question is unanswerable while the path returns fill.

**Follow-up (2026-09), minimal repro for Intel.**
`shaders/linalg_repro_groupshared_load.hlsl` reduces this to a multiply of two
constant matrices, so the expected answer needs no reference implementation:

    A[r][k] = FILL_A (8 x 32, RowMajor)   B[k][c] = FILL_B (32 x 16, ColMajor)
    C[r][c] = 32 * FILL_A * FILL_B

Measured, s8 x s8 -> s32 at 8x32x16, wave 16, driver 32.0.101.8992:

| FILL_A | expected C | observed C[0][0] | distinct values in C     |
|--------|-----------|------------------|--------------------------|
| 1      | 32        | 516128           | 129032 258064 387096 516128 |
| 3      | 96        | 516128           | 129032 258064 387096 516128 |
| 100    | 3200      | 516128           | 129032 258064 387096 516128 |

Bit-identical across all three. 516128 = 32 * 127 * 127, and 127 is never
written by the shader - the largest byte it stores is FILL_A - so the value
cannot be misread input. The same dispatch reads the two groupshared arrays
back with ordinary loads as a control: they hold exactly `0x03030303` and
`0x01010101` with no other distinct value, so the data is present and the
barrier is correct at the moment of the matrix load.

One detail worth passing on: the last three columns come back as 24, 16 and 8
accumulated terms rather than 32 (387096 / 258064 / 129032 = 24/16/8 * 127*127).
So the path is not purely returning fill - something bounded is happening at the
tile edge - which may be the more useful end to pull on.

### Descriptor loads work, but demand 4-byte alignment

`Matrix::Load(ByteAddressBuffer, StartOffset, Stride, Layout, Align)` is
correct, and its offset and stride are in bytes, so there are no units to
guess. The `Align` argument, however, does not buy sub-dword addressing. Ask
for less than 4 and the load quietly truncates to a 4-byte boundary, dropping
the leading bytes:

```
off 8192 stride 32 Align 128   32 64 96 128 160 192 224 256   correct
off 8196 stride 32 Align 4     32 64 96 128 160 192 224 256   correct
off 8194 stride 32 Align 2     30 62 94 126 158 190 222 254   2 elements lost
off 8193 stride 32 Align 1     31 63 95 127 159 191 223 255   1 element lost
off 8192 stride 34 Align 4     32 60 96 120 160 180 224 240   odd rows lose 2
```

The stride-34 row is the important one. Losing elements only on odd rows is the
signature: at a 34-byte pitch every second row starts 2 mod 4, and loses exactly
the 2 bytes needed to reach the next dword. It is silent - no failure, no
warning, just a slightly wrong number.

**Correction (Intel, 2026-09).** This is not a driver defect. Sub-dword aligned
descriptor loads are not permitted by the Linear Algebra specification, so the
observed behaviour is conformant - we were asking for something the API does not
offer. The practical consequence below is unchanged, but the outlook is: this is
permanent, not something a driver update will lift. Design around it.

The only thing still worth asking for is diagnostics. A spec violation that
returns a slightly wrong number in silence is expensive to find; a debug-layer
error would have turned this into a five-minute fix.

### What this costs a Q8_0 GEMM

Q8_0 is a bad fit for what remains. A block is 34 bytes and its 32 quants start
2 bytes in, so along K the data is neither 4-byte aligned nor at a 4-byte pitch,
and by the rule above it cannot be handed to the matrix load directly. The
obvious workaround - stage a tile in LDS and load from there - is the path that
does not work.

That leaves repacking into a 4-byte-aligned scratch UAV, which puts a global
memory round trip in front of every tile. Against a dp4a baseline that reads the
weights once and needs no repack, that is a real handicap, and it should be
measured before more of the GEMM is written. K=32 matching QK8_0 exactly is
still the nicest property on offer here, so the shape is right even though the
addressing is not.

Method note: use one all-ones operand when probing a matrix multiply. An
all-ones tile reads the same under any stride or layout, so the result depends
on the other operand only, and one unknown gets measured at a time. Trying to
read both operands' conventions out of a single `C = A*B` is how the first three
attempts here produced garbage that looked like a hardware fault.

Also: do not pass a Matrix by value into a helper or reuse one Matrix variable
across measurements. These are opaque handles and nothing else in the tree does
it; a probe written that way produced results that were wrong in a way that
mimicked a layout bug.

## 22. Shipping a GEMM on the 8x16x16 shape

Section 21 established that the Intel Xe3 driver offers exactly one f16
matrix shape - `8(M) x 16(K) x 16(N)`, f16 accumulator, wave scope - and
that its groupshared operand load is broken. `mul_mat_linalg_wave_f16_i.hlsl`
is a GEMM built inside that box: nothing is staged, both operands come
straight out of buffers via descriptor loads, and the f16 accumulator is
drained into f32 registers every FOLD_BLOCKS tiles so the long K reduction
happens in f32.

Result on Phi-3-mini F16, Arc B390: `pp512` 349 -> 469 t/s, **+34%**.
`tg64` is unchanged (14.5) - decode is M=1 and the matvec kernels own it.

### An 8-row tile is not a GEMM

The first working version tiled `8(M) x 64(N)` per threadgroup, one 8x16
tile per wave. It was correct (1216/1216) and **35% slower** than the
existing tiled kernel it was replacing (202 vs 303 t/s).

Arithmetic intensity for a BMxBN output tile is `BM*BN / (BM+BN)`. At
8x64 that is 7.1; the 64x64 kernel it displaced sits at 32. The 8-row
tile re-reads the entire weight matrix for eight output rows.

The fix is to give each wave MTILE row tiles and load the weight tile
**once** per k-step for all of them:

```hlsl
MatBf b = MatBf::Load(src0, b_base + kb, nb01, MatrixLayout::ColMajor, 4u);
[unroll] for (uint mj = 0; mj < MTILE; mj++) {
    acc[mj].MultiplyAccumulate(MatAf::Load(...), b);
}
```

Two things worth knowing, both contrary to what section 21 suggested:
arrays of `Matrix` compile and work, and a `Matrix` held in a variable and
passed to `MultiplyAccumulate` gives correct results. The "never reassign,
never pass by value" rule from bring-up was narrower than it looked.

Measured, Phi-3 F16 `pp512`, interleaved A/B against the same binary:

| MTILE | BM | intensity | pp512 |
|---|---|---|---|
| 1  | 8  | 7.1  | 202, 196 |
| 4  | 32 | 21.3 | 442, 419 |
| 8  | 64 | 32.0 | **467, 471** |
| 16 | 128| 42.7 | 394, 424 |

MTILE=16 falls off - eight accumulators plus 64 f32 partials is already
near the register budget at 64 threads, and doubling it spills.

### The real hang: the matrix load faults at 2 GB

The shader passed `test-backend-ops` 1216/1216 but TDR'd the device on a
real model. Bisecting by graph position pinned it to a single dispatch:

```
take 34  ne0=3072  s0off=2065784832  -> clean
take 35  ne0=16384 s0off=2135003136  -> DEVICE_HUNG
```

Take 34 ends at 2,084,646,912. Take 35 ends at 2,135,003,136 + 16384*6144
= **2,235,666,432**, the first flag-264 access to reach past `0x80000000`.
Every earlier dispatch of that same shape at a lower offset was fine.

**The Intel matrix `Load` sign-extends its buffer offset.** Any byte at or
above 2 GB faults. This is invisible to `test-backend-ops` (tiny buffers),
invisible below `-ngl 12` on Phi-3 (weights stay under the line), and
independent of shape, K, M, fold depth, submission granularity and the
replay cache - which is why it read for a long time like a per-dispatch
driver leak. The host gate now requires
`tensor_offset(src0) + ne01*nb01 <= 0x80000000`.

The cost is real: on a 7 GB F16 model only the layers below 2 GB take the
path, so the +34% above is what a third of the model buys. Worth reporting
to Intel.

### Two traps that cost whole rounds of bisecting

**CMake evaluates `$ENV{VAR}` at configure time, not build time.** Driving
a shader define with `-D FOO=$ENV{FOO}` bakes whatever the value was when
CMakeLists.txt was last processed. An entire sweep of a K-truncation knob
reported false "clean" results, and the binary that came out of it passed
the hang test while emitting garbage and dropping MUL_MAT to 1191/1216.
Runtime `getenv` on the host is fine; CMake `$ENV{}` is not. There are no
`$ENV{}` defines in the DX12 shader rules and there must not be.

**A stale bisect define can outlive its use silently.** `IW_ENABLE_LOOP`
stayed `#define`d but unreferenced after an edit replaced it, so the one
"clean" data point in the table came from a shader that no longer existed.
Grep for a knob's *uses*, not its definition, before trusting a result.

### Also fixed here

- The activation convert pre-pass sized the *conversion* by the padded
  destination extent, reading up to `BM*K*4` bytes past the end of `src1`.
  Destination padding is not source extent: `n_elems` sizes the buffer and
  the cache key, `c_elems` bounds the read, and the shader zeroes anything
  past `ne1`. The pad rows must be zeroed, not left undefined - they are
  read by the matrix load even though they are never stored.
- The convert reuses the Q8_1 scratch and its cache key, so a Q8_1
  quantize and an F16 convert could each take a false cache hit on the
  other's encoding. `last_q8_1_kind` (1 = q8_1, 2 = flat f16) now
  discriminates at all three reuse sites.

## 23. Vulkan sees one capability on this part that LinAlg does not

Intel asked for a groupshared repro (section 21), and closed the alignment
item as spec-conformant. That prompted the obvious cross-check: the same
silicon also exposes `VK_KHR_cooperative_matrix`. Does Vulkan advertise the
same set?

`tools/vk_coopmat_probe.cpp` calls
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` directly - vulkaninfo
reports the extension but does not dump the shape list.

B390, driver 32.0.101.8992, subgroup 32:

| M | N | K | A | B | C | Result |
|---|---|---|---|---|---|--------|
| 8 | 16| 16| f16 | f16 | **f32** | **f32** |
| 8 | 16| 16| f16 | f16 | f16 | f16 |
| 8 | 16| 32| s8  | s8  | s32 | s32 |
| 8 | 16| 32| u8  | u8  | u32 | u32 |

D3D12 LinAlg on the same part (`DX12_LINALG_CAPS=1`):

```
wave f16xf16 -> f16  shapes: 8x16x16
wave s8xs8   -> s32  shapes: 8x32x16
wave u8xu8   -> s32  shapes: 8x32x16
```

Mind the convention: Vulkan reports (M, N, K), D3D12 reports M x K x N. So
Vulkan's 8/16/16 is D3D12's 8x16x16 and Vulkan's 8/16/32 is D3D12's 8x32x16.
The shapes match exactly, which is a useful independent confirmation that our
reading of the D3D12 shape triple is right.

**The gap is exactly one entry: `f16 x f16 -> f32`.** Vulkan has it, LinAlg
does not.

That is the entry we actually wanted. Section 20 chased 439 wrong results
because our shaders asked for an f32 accumulator and the PSO built anyway; we
concluded the part could not do it. It can - it just is not reachable through
LinAlg. And the f16-accumulator workaround in
`mul_mat_linalg_wave_f16_i.hlsl` - drain to f32 registers every FOLD_BLOCKS
tiles, because an f16 accumulator cannot carry a K=3072 reduction - exists
only to work around its absence.

### It is exercised, not merely advertised

Section 20's lesson was that an advertised capability can still be wrong, so
advertisement alone is not the claim here. `ggml-vulkan` hard-requires the f32
accumulator:

```cpp
if (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
    GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. "
                   "Disabling matrix cores.\n");
    device->coopmat_support = false;
}
```

On this device that warning does not fire, the backend reports
`matrix cores: KHR_coopmat`, and `test-backend-ops test -o MUL_MAT` passes
**1021/1021** through the coopmat GEMM. So the f32 accumulator is real,
reachable and correct on this silicon and this driver package.

Generalised: when one API says a part cannot do something, ask another API on
the same silicon before believing it. A capability list is a statement about
an implementation, not about hardware.

### How much headroom is there, in numbers

Since a working Vulkan build now exists on this machine, it is worth having the
number. SmolVLM2-256M f16 (small enough that every layer clears the 2 GB gate,
so our LinAlg path is fully engaged), pp512, interleaved to control for the
thermal drift noted in section 16:

| round | DX12 (LinAlg, MTILE=8) | Vulkan (KHR_coopmat) |
|-------|------------------------|----------------------|
| 1     | 13032 +/- 1776         | 23633 +/- 257        |
| 2     | 11773 +/- 1028         | 22703 +/- 1117       |

Vulkan is about **1.9x** our prefill on the same silicon.

Do not read that as the cost of the missing f32 accumulator. It is not a
controlled comparison: `ggml-vulkan` runs a 128x128/BK32 tile against our
64x64, gets subgroup 32 against our wave 16, has no fold/drain structure, and
is a far more mature backend generally. The number says how much room is left
above us, not which change closes it.

It is still the useful framing for prioritisation. The section 22 work bought
+35% and the remaining gap is another 90%, so the DX12 GEMM is not near a
hardware ceiling - it is near the ceiling of this shader.

Note also the error bars: our runs spread 8-15%, Vulkan's 1-5%. Consistent
with section 16 - hold any future A/B in one thermal state.

## 24. The 2 GB fault is in the offset argument, not the address

Section 22 gated flag 264 to weights ending below 2 GB, which cost Phi-3 F16
two thirds of its layers. That gate is now gone.

Two causes fit the evidence equally well:

- **(a)** the `StartOffset` the shader hands `Matrix::Load` is treated as
  signed 32-bit
- **(b)** the matrix path mishandles a *virtual address* at or above 2 GB

A plain `ByteAddressBuffer` load at the same address always worked, which
hinted at (a), but hinting is not knowing, and the two call for opposite fixes.

The experiment that separates them: `src0` is bound as a **root SRV**, so the
base is a GPU virtual address we choose. Binding at the weight rather than at
the heap leaves the VA of every byte touched **bit-identical** and shrinks only
the offset argument. Nothing else changes.

It runs clean. **(a) confirmed** - the defect is in the offset argument.

That makes it a complete workaround, not just a mitigation: no single weight
tensor is remotely near 2 GB, so the offset can never get close again however
large the model. Intel still has a real bug; we no longer have to wait for it.

Phi-3 F16, `-ngl all`, per-layer coverage went from a third to all of it:

| | before | after |
|---|---|---|
| layers on fl=264 | ~11/32 | **31/32** |
| pp512, LinAlg on | - | 538.6, 536.9 |
| pp512, LinAlg off | - | 463.6, 363.1 |

Conservatively **+16%** (round 1; the off-side spread is thermal, section 16).
`test-backend-ops test` 18035/18035, greedy output byte-identical on vs off.

The generalisable part: when a driver mishandles a value, check whether you get
to choose where that value starts from. A root descriptor is a free change of
origin, and it converted an indefinite wait on a vendor fix into a closed item.

### Where the time goes now

Phi-3 F16 pp512, share of GPU time:

| op | share |
|----|-------|
| MUL_MAT fl=264 | **90.9** |
| FLASH_ATTN_EXT | 4.7 |
| GLU | 1.4 |
| ADD | 1.1 |

SmolVLM2-256M f16 is less lopsided - MUL_MAT 65.3, FA 19.4, ADD 7.1 - and the
two Q4_K_M models put 83.7 into the *quantized* MUL_MAT, which has no LinAlg
path at all. Across all four benchmark models the GEMM is the only target that
matters; nothing else is worth touching until it is done.

## 25. Reuse each operand more, because LDS is not available

Section 24 left the GEMM at 90.9% of Phi-3 F16 prefill, so it is the only
thing worth tuning. The obvious move was Vulkan's tile: it runs 128x128 where
we run 64x64.

Widening N by adding waves does not work:

| NWAVE | BN | pp512 |
|-------|-----|-------|
| **4** | **64** | **534.7** |
| 8 | 128 | 500.8 |
| 16| 256 | 509.3 |

That result is the useful part. Arithmetic intensity by the usual
`BM*BN/(BM+BN)` measure rises from 32 to 42.7 across that sweep, and
performance falls. The formula does not apply here, because it assumes the
operands are staged once per workgroup in LDS and shared. Ours are not: the
groupshared matrix `Load` is broken on this driver (section 21), so every wave
re-reads both operands from the descriptor. Adding waves to a group therefore
adds no reuse at all - it just makes groups fatter and fewer.

So Vulkan's 1.9x (section 23) is not mainly the f32 accumulator. It is that
`ggml-vulkan` stages A and B in shared memory and we cannot. **Of the open
Intel defects, the groupshared one is the one capping throughput.**

What is still available is reuse *within* a wave, which needs no LDS. MTILE
already did it in one direction - hoist a weight tile, sweep it over MTILE row
tiles. NTILE does the other - hold NTILE weight tiles and sweep each activation
tile over all of them. Per wave per k-step the traffic is
`128*MTILE + 256*NTILE` for `128*MTILE*NTILE` outputs, so the product is what
matters and a square tile is the best shape for a given register budget.

Phi-3 F16 pp512, all 1216/1216 on `-o MUL_MAT` at every step:

| MTILE | NTILE | wave tile | traffic/output | pp512 |
|-------|-------|-----------|----------------|-------|
| 8 | 1 | 64x16 | 1.25 | 547.7 |
| **4** | **2** | **32x32** | **1.00** | **566.1** |
| 8 | 2 | 64x32 | 0.75 | 529.5 |
| 4 | 4 | 32x64 | 0.75 | 557.4 |

Note that traffic keeps falling past the winner and performance does not
follow: (8,2) and (4,4) move the same bytes and differ by 5%. Past the square
tile the limit stops being bandwidth and becomes live accumulator registers.
Two knobs, and the second one only shows up once the first stops binding.

Settled on **MTILE=4, NTILE=2**.

### Where this landed

`test-backend-ops test` 18035/18035. Phi-3 F16 greedy output byte-identical
with the path on and off.

| model | LinAlg off | LinAlg on | gain |
|-------|-----------|-----------|------|
| Phi-3-mini F16 | 440.6, 286.7 | 567.1, 528.6 | +29% (round 1) |
| SmolVLM2-256M f16 | 9477.7, 9098.2 | 15979.6, 16932.9 | **+69 to +86%** |

SmolVLM2 gains far more than Phi-3 because its ne0 values are small (192, 576,
1536) and a 32x32 wave tile fits them where 64x16 wasted most of a tile. It
also started this section at 12.2-12.5k, so NTILE alone bought it about +31%.

The off-side spread is thermal (section 16); compare within a round.

### Validated on the four benchmark models

All four run end to end on the shipping tile (MTILE=4, NTILE=2, NWAVE=4,
FOLD_BLOCKS=4), coherent output, no faults:

| # | model | prompt t/s | gen t/s |
|---|-------|-----------|---------|
| 1 | SmolVLM2-256M f16 + image | 962.0 | 198.2 |
| 2 | SmolVLM2-256M Q4_K_M + image | 667.3 | 256.5 |
| 3 | Phi-3-mini F16 | 100.0 | 14.4 |
| 4 | Phi-3-mini Q4_K_M | 95.1 | 35.0 |

The four tile knobs are duplicated between the shader (`-D` in
ggml-dx12/CMakeLists.txt) and the host (`DX12_IW_*` in ggml-dx12.cpp), which
sets the group count, the M-tail pad and the divisibility gates. They must
match. A stale `DX12_IW_NWAVE` left over from a sweep produced pp512 of 1877
against a true 535 - the host dispatched a quarter of the groups and
`-o MUL_MAT` dropped to 1215/1216. If a tuning result looks too good, check
these six lines before believing it.

`ne00 % 64` in the gate was really `LA_K * FOLD_BLOCKS` written out; it is now
spelled that way so retuning FOLD_BLOCKS cannot silently break the gate.

## 26. How Vulkan feeds quants to coopmat, and why our GEMM is 3.7x off

> CORRECTION: this section assumes LDS staging is unavailable. It is not.
> The sweep is still valid data, but the conclusion does not follow. See
> section 35.

The question that started this: Vulkan and D3D12 see the same matrix shapes on
this part (section 23 - only f16xf16->f32 is missing from LinAlg), so why does
Vulkan win on every quant?

### Vulkan has exactly one matrix GEMM, and quant type is just a fill function

`mul_mm.comp` declares the A tile as groupshared:

    shared FLOAT_TYPEV2 buf_a[BM * SHMEM_STRIDE];

and loads the matrix operand out of it:

    coopMatLoad(cache_a, buf_a, a_shmem_index(...), a_shmem_stride(), RowMajor);

The quant never reaches the matrix core. `mul_mm_funcs.glsl` dequantizes into
that tile - the Q4_K arm ends:

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;
    store_a(col, k_pair,     FLOAT_TYPEV2(fma(d, q.x, m), fma(d, q.y, m)));

So Q4_K, Q5_K, Q6_K, Q8_0 and F16 all run the *same* coopmat GEMM. Only the
per-type fill differs. Vulkan needs no s8 matrix shape for K-quants at all.

`mul_mmq.comp` is a separate integer path on GL_EXT_integer_dot_product - dp4a,
no matrix cores. It is the fallback, not the quant strategy.

### Measured, and it shows

Phi-3-mini, B390, -fa 1, prefill t/s:

| build      | type   | pp512  | pp6144 |
|------------|--------|--------|--------|
| Vulkan     | F16    | 2091.7 |  582.8 |
| Vulkan     | Q4_K_M | 2132.1 |  573.4 |
| DX12 LinAlg| F16    |  559.5 |  313.9 |
| DX12 LinAlg| Q4_K_M |  759.9 |  363.2 |

Vulkan Q4_K equals Vulkan F16 to within noise. Dequantizing to f16 in LDS is
free - it is hidden behind the same memory traffic the GEMM already pays. That
is the whole design, and it is why quant type barely moves Vulkan's number.

Two further readings of the table:

- Our Q4_K (759.9) beats our F16 (559.5). The Q4_K number is the tuned dp4a
  matvec/GEMM moving a quarter of the bytes; the F16 number is the LinAlg path.
  We are not losing because of the quant - we are losing on the GEMM itself.
- The gap narrows with prompt length (3.7x at 512 -> 1.9x at 6144) because both
  backends become attention-bound as n^2 grows. Short-context prefill is where
  the GEMM deficit is visible; that is the regime to tune in.

### What this means for the quant-LinAlg plan

The parked `iw-quant-linalg` note assumed reaching the matrix core for Q4_K
needed an int8 repack onto the s8 8x32x16 shape. Vulkan shows that assumption
was wrong: dequantize to f16 and reuse the f16 GEMM.

We cannot copy it directly. Vulkan stages in LDS, and groupshared Matrix::Load
is the broken path on this driver (section 21). The workaround is a device
scratch tile rather than groupshared, which costs bandwidth Vulkan does not pay.

But the table says do not start there. A quant path built on our current GEMM
inherits a GEMM that is 3.7x off Vulkan on the same silicon and the same
shapes. Vulkan's lesson is that once the GEMM is right the quant support is a
per-type fill function - cheap, and worth little before then.

Order of work: fix operand reuse in the f16 GEMM first (section 25 - every wave
re-reads both operands because we cannot stage in LDS), then add per-type fill.
Not the reverse.

### The GEMM cannot be fixed without the groupshared fix

Where pp512 goes on Phi-3 F16 (DX12_PROFILE, graph #2): flag 264 is 90.6% of
the frame (51.1 + 20.6 + 14.8 + 4.1), FLASH_ATTN_EXT is 4.9%. There is no cheap
win outside the GEMM.

Per-dispatch efficiency, same frame:

| shape                | ms/dispatch | TFLOP/s |
|----------------------|-------------|---------|
| K=3072 N=16384 M=512 | 14.92       | 3.45    |
| K=3072 N= 9216 M=512 |  5.84       | 4.97    |
| K=8192 N= 3072 M=512 |  4.30       | 5.99    |
| K=3072 N= 3072 M=512 |  1.15       | 8.42    |

Efficiency falls as the weight matrix grows. BN tiles the weight rows (dst
ne[0]) and BM tiles the tokens (dst ne[1]), so the weight matrix is re-read
M/BM times - 16 times at BM=32. The obvious fix is a bigger BM.

It does not work. Sweeping past the earlier MTILE<=8 range, pp512:

| MTILE | NTILE | BM  | BN  | pp512  |
|-------|-------|-----|-----|--------|
| 4     | 2     |  32 | 128 | 566.1  |  <- shipping
| 8     | 1     |  64 |  64 | 547.7  |
| 16    | 1     | 128 |  64 | 406.0  |
| 8     | 4     |  64 | 256 | 425.6  |
| 16    | 2     | 128 | 128 | 262.1  |

Monotonically worse. The tile cannot grow because registers are the only place
we have to stage operands, and the accumulator count is MTILE*NTILE. Vulkan
runs BM=BN=128 (l_warptile) precisely because it stages in LDS and the matrix
load reads from there.

So the 3.7x is not a tuning deficit we can close. Both the bandwidth (every wave
re-reads both operands) and the tile-size ceiling (registers, not LDS) trace to
the same defect: groupshared Matrix::Load returns wrong data on this driver
(section 21, Intel issue 1).

That reorders the work. The groupshared bug is not a correctness curiosity to
mention alongside the f32-accumulator request - it is the single blocker
standing between us and Vulkan-class prefill, and it should lead the filing.
Quant support is cheap once the GEMM is right (this section, top) and worth
little before then.

### What is NOT blocked: ordinary groupshared works, and FA is the real cost

The defect is narrow. It is `Matrix::Load` from a groupshared array, not
groupshared memory. Ordinary LDS is fine and we already lean on it hard:
`flash_attn_pf.hlsli:153-173` declares 15 groupshared arrays and
`mul_mat_wmma64.hlsl:46-47` stages both operands in LDS, and both pass
test-backend-ops 18035/18035. The Intel repro says the same thing - reading the
arrays back with ordinary loads right after the failing matrix load returns the
correct data.

So the blocker applies only to code that wants operands in LDS *for the matrix
core*. Everything else is open.

And at long context the largest item is not the GEMM. Phi-3 F16 pp6144,
FLASH_ATTN_EXT share of frame as the cache fills:

| nkv  | FA %  | FA ms  |
|------|-------|--------|
| 1024 | 12.2  |  127.3 |
| 4096 | 44.4  |  836.7 |
| 4608 | 46.8  |  970.4 |
| 5120 | 54.2  | 1354.5 |
| 6144 | 55.5  | 1267.6 |

At pp512 FA was 4.9%, which is why earlier passes dismissed it. At pp6144 it is
over half the frame.

It is also slow in absolute terms. At nkv=6144: 39.6 ms/dispatch for roughly
1.9e10 causal FLOPs = about 0.49 TFLOP/s, against 3.45-8.42 TFLOP/s for our own
GEMM on the same part. KV traffic is only ~15 GB/s, so it is not bandwidth
bound - there is real headroom.

FA on the matrix core (`iw-fa-linalg`) would hit the same operand-staging wall.
Plain FA tuning would not. That is the work to do while the driver bug is open.

## 27. A Matrix must not live in an array

Symptom: the flag-264 GEMM returned NaN and stored out of bounds whenever
`dst->ne[0] % 64 != 0`. Row 0 of the tile was correct, every row above it was
never written. Deterministic - five runs gave the same 1241/1262.

The cause was one declaration:

```hlsl
MatBf b[NTILE];
[unroll] for (uint nb = 0; nb < NTILE; nb++) b[nb] = MatBf::Load(...);
...
[unroll] for (uint nj = 0; nj < NTILE; nj++) acc[mj][nj].MultiplyAccumulate(a, b[nj]);
```

Filled in one loop, read back in a later one. `NTILE=1` collapses the array to
one element and works; `NTILE=2` breaks. Named locals `b0`/`b1` fix it with no
other change.

Spec proposal 0035 explains why this is fragile rather than merely unlucky:
a matrix is an *intangible* object with no defined size or layout, and at the
DXIL level "the matrix object stores a pointer in the IR". An array of those is
an array of pointers that the compiler must fully resolve away. It does not
here.

The shader header already carried this warning from bring-up. The array crept
back in anyway. Rule: **name every matrix, never index one.**

Note `MatAccf acc[MTILE][NTILE]` is still an array and passes 18035/18035.
It is written and read at unrolled constant indices inside the same scope,
which appears to be enough. Treat it as a known risk, not a proven-safe case -
`b[]` also passed everything the models touch.

### Why it hid for so long

`-o MUL_MAT` dispatches flag 264 with **four** shapes, every one of them
`ne0` = 64 or 128. Both multiples of 64, so all four passed. The path was
broken the whole time and the op-targeted run said 1216/1216.

`MUL_MAT_VEC_FUSION` is what caught it, because it uses n=32.

**After touching flag 264, run the full suite, not `-o MUL_MAT`.**

Two smaller traps from the same hunt:

- `Select-String ': FAIL'` under-reports; the console wraps the line and splits
  the marker. Grep `'NaN|ERR ='`, or read the `N/M tests passed` summary.
- In `MUL_MAT_VEC_FUSION` names, `n_mats` and `n_used` are inert when
  `use_id=0` (`tests/test-backend-ops.cpp:6689`). They do not mean the case is
  MoE. Chasing that cost most of a session.

### Cost of the fix

None. Phi-3 F16 pp512, interleaved, `DX12_LINALG_F16_WAVE` on vs off:

| pass | LinAlg on | off   |
| ---- | --------- | ----- |
| 1    | 565.8     | 392.7 |
| 2    | 568.3     | 456.5 |

Same as the pre-fix 566.1 for this tile shape.

## 28. Where FA prefill time actually goes

FA is 52% of the pp6144 frame on Phi-3 F16 (D=96, BR=32, BC=32, 256 threads),
so three FA experiments were queued off the back of section 26. Before writing
any of them, two probes attributed the time. Each probe is a deliberately wrong
shader that removes one term and keeps the FMA count, measured by swapping
`ggml-dx12.dll` in and out so the A/B is interleaved in one thermal state.

FLASH_ATTN_EXT fl=114 at nkv=6144, sum over 32 dispatches:

| build                        | ms         | delta   |
| ---------------------------- | ---------- | ------- |
| baseline                     | 1009, 1010 | -       |
| PV without the s_scores reads |  893, 897  | -11.7%  |
| QK inner loop removed        |  588, 587  | -42%    |

So: QK 42%, PV score traffic 12%, and the remaining 46% is tile staging, the
two softmax reductions, the barriers and the PV math itself.

### What this rules out

`fa-exp-v-vec` is dead. The PV loop reads V once and feeds `FA_PF_ACC`
accumulators, and `FA_PF_ACC` is 16 at D=96, so the scalar 16-bit V read is one
LDS op in 17. Widening the whole V tile to f32 (which fits: 30852 B) measured a
wash - 998 vs 1010 ms interleaved, inside the run-to-run spread. Section 12's
"scalar 16-bit LDS is slow" still holds; it just is not reachable from here.

Vectorizing the s_scores reads into float4 is bounded above by the 11.7% probe,
so about 9% of FA and 4.5% of the frame at best. It also needs the QK thread
mapping changed from one column per thread to four, which is the online-softmax
indexing. Not worth that risk for 4.5%.

### What is left

QK, at 42%, is the only large term. Per thread per tile it runs 24 iterations of
1 `s_kh` half4 read + 4 `s_qh` half4 reads + 8 `dot2add` - 120 LDS reads for 192
dot2add. Both tiles are 8-byte reads that could be 16-byte ones.

Growing the tile is not an option. At D=96 the LDS budget is already at 24708 B
of 32768; BR=64 needs 35716 and BC=64 needs 41988. The tile is at its maximum
unless K and V share one slot, which costs the barrier the fp16 path was built
to avoid.

## 29. Splitting the Vulkan gap: how much of it is LinAlg?

Section 26 measured the gap but not its parts. A four-way run, all in one
thermal window, Phi-3 F16 pp512, matrix path toggled by env var on each
backend (`GGML_VK_DISABLE_COOPMAT`+`COOPMAT2` / `DX12_LINALG_F16_WAVE=0`):

| backend | matrix path on | off    | uplift |
| ------- | -------------- | ------ | ------ |
| Vulkan  | 2053.0         | 577.5  | 3.56x  |
| DX12    |  575.1         | 470.7  | 1.22x  |

Two things fall straight out.

**Our matrix path performs like Vulkan's scalar path.** 575.1 vs 577.5. The
matrix engine is buying us almost nothing over a good non-matrix GEMM.

**The gap is mostly, but not only, LinAlg.** Of the 3.57x total:
- 1.23x is generic - our non-matrix GEMM is already behind Vulkan's (470.7 vs 577.5)
- 2.9x is matrix-specific - coopmat gives Vulkan 3.56x where LinAlg gives us 1.22x

### Why 2.9x, and it is the groupshared bug

Reuse per operand element loaded:

| | tile | staged in | MACs per element |
| - | - | - | - |
| Vulkan `l_warptile` (ggml-vulkan.cpp:4342) | BM=128 BN=128 | LDS | 128*128/256 = 64 |
| ours, per wave | 32 x 32 (MTILE 4 x NTILE 2) | registers | 32*32/64 = 16 |

4x less reuse, against a 2.9x deficit. The threadgroup tile is BM=32 BN=128,
but that number does not help us: with `Matrix::Load` from groupshared broken
(section 21) every wave re-reads both operands itself, so only the wave tile
counts.

Sanity check on the operand traffic each one needs:

| | TFLOP/s | FLOP/byte | implied operand traffic |
| - | - | - | - |
| Vulkan coopmat | 15.7 | 64 | ~245 GB/s |
| ours           |  4.4 | 16 | ~274 GB/s |

Both land at roughly the same delivered bandwidth. They are limited by the same
thing; Vulkan simply buys 4x more arithmetic with each byte. (Estimate - it
ignores whatever reuse L1 gives us across waves in a group.)

### Consequence

We cannot close this by tuning. Section 25 already swept the tile and registers
are the wall: (4,2)=566.1, (8,1)=547.7, (8,4)=425.6, (16,2)=262.1. Bigger tiles
spill because registers are the only staging area we have.

**Fixing the Intel groupshared `Matrix::Load` bug is worth up to ~2.9x on F16
prefill.** That is the number to attach to issue 1 of the report. The other
~1.23x is ours to fix and needs no driver change.

Section 30 confirms the causal step this section assumed: the load cost is
per-byte, not per-issue, which is what makes LDS staging able to recover it.

**Caveat, added later: do not quote the 2.9x.** It was measured against a
baseline that section 31 then improved by 2x with a group swizzle. The gap is
now 1.13-1.22x, so whatever the groupshared bug is worth, it is worth much less
than 2.9x. The bug is still real and still worth reporting on its own terms -
just not with this number attached.

## 30. What the LinAlg GEMM actually spends its time on

> CORRECTION: the per-byte load attribution below stands, but the claim that
> LDS staging is unavailable to fix it is wrong. See section 35.

Section 29 split the Vulkan gap into a 1.23x generic part and a 2.9x
matrix-path part, and argued the matrix part is operand reuse. That was an
argument from tile arithmetic, not a measurement. Three probes settle it.

Each probe keeps the MultiplyAccumulate count identical and changes only the
loads. All are deliberately wrong - they are timing instruments, not
candidate optimizations. Phi-3-mini F16, pp512, DLL-swap interleaved, two
passes, one thermal window.

| build                                            | pp512        | vs base |
|--------------------------------------------------|--------------|---------|
| base                                             | 574.3, 571.0 | 1.00x   |
| same-address (same load count, all cache hits)   | 2016, 2412   | ~3.9x   |
| hoisted (4x fewer loads, same MACs)              | 3097, 3145   | 5.4x    |
| Vulkan coopmat, for reference                    | 2053         | 3.57x   |

Reading:

- The GEMM is almost entirely `Matrix::Load`. Remove the memory traffic and
  the same shader runs 3.9x faster; remove three quarters of the load
  instructions too and it runs 5.4x faster, past Vulkan.
- Cost is mostly per-byte, not per-issue. The same-address probe issues every
  load the base build issues and still wins 3.9x, so ~3.9x of the 5.4x is data
  movement and the remaining ~1.4x is issue overhead.
- That per-byte split is what section 29 needed. Groupshared staging does not
  reduce the load count, so if the cost had been per-issue, fixing the Intel
  groupshared bug would have bought nothing. It is per-byte, so staging the
  tile once per group and re-reading it from LDS attacks the dominant term.
  The 2.9x measured Vulkan matrix advantage sits inside the 3.9x ceiling.
- MAC throughput is not the limit anywhere in this range.

Two hypotheses died on the way:

- **The f16 accumulator drain does not matter.** `sum[MTILE][NTILE][ACC_E]`
  exists only because Intel exposes f16xf16->f16 and not ->f32 (section 23),
  so the accumulator must be drained to f32 every FOLD_BLOCKS tiles. Halving
  the drain count, FOLD_BLOCKS 4 -> 8, measured 571/573 -> 476/476, i.e. 17%
  **slower**. The longer live range costs more than the drains save. Reverted;
  FOLD_BLOCKS stays 4.
- **The Align argument is inert.** All offsets and strides on this path are
  32-byte multiples, so `Matrix::Load(..., 16u)` is legal here. It measured
  571/567 vs 570/567 - a wash, and correct (1262/1262). This confirms from the
  production shader what section 21 found from the probes: the descriptor load
  truncates to 4 bytes whatever Align says. Reverted to 4u.

So the ranked list for this kernel is: groupshared `Matrix::Load` (Intel issue
1) first and by a wide margin, then the missing f16xf16->f32 accumulator shape
(section 23) - not because the drain is expensive, but because an f32
accumulator would free the registers `sum` occupies and let the tile grow,
which is the only other way to cut bytes per MAC. Nothing else here is worth
touching until one of those lands.

## 31. Group swizzle: 2x on F16 prefill, no driver fix needed

Section 30 found the GEMM is bound by bytes fetched. The first place to look
for bytes is not the inner loop at all - it is the order the groups run in.

The dispatch is `groups_x = ne1/BM` (tokens) and `groups_y = ne0/BN`
(channels), and D3D12 walks x fastest. So the hardware runs a whole column of
token tiles against one weight block, then moves to the next weight block and
reads **every activation row again**. Over the dispatch the activations are
read `n_y` times.

For one Phi-3 QKV projection at pp512 (ne0 9216, ne1 512, K 3072): n_x = 16,
n_y = 72, activations 3 MB, weights 56.6 MB.

    traffic = A * n_y + B = 3 * 72 + 56.6 = 273 MB

Band `SWIZZLE_Y` column blocks together and run y fastest inside the band, and
`SWIZZLE_Y` consecutive groups share one activation strip:

    traffic = A * n_y/SWIZZLE_Y + B

The weights still stream once either way. Only the re-read count changes.

Measured, Phi-3-mini F16 pp512, DLL-swap interleaved, two passes:

| SWIZZLE_Y | pp512        | vs 1  |
|-----------|--------------|-------|
| 1 (none)  | 565, 575     | 1.00x |
| 4         | 765, 788     | 1.35x |
| 8         | 883, 930     | 1.58x |
| **16**    | **1136, 1183** | **2.03x** |
| 32        | 1083, 847    | noisy |
| 64        | 1103, 1137   | 1.96x |

16 is the peak and is where the resident weight blocks still fit: at K 3072 a
block is 128*3072*2 = 786 KB, so 16 of them is 12.6 MB. Past that the bands
start evicting each other and the curve turns over and gets noisy.

Implementation is a remap of the group id at the top of the shader, the Triton
`swizzle2d` idiom. The only subtlety is the last band, which is short when n_y
is not a multiple of SWIZZLE_Y; using its real height `bh` instead of
SWIZZLE_Y keeps the remap a bijection, so no tile is dropped or done twice.
The host dispatch is unchanged.

### The guard: bands need something to band

Applied unconditionally the swizzle is a **loss** on small models. SmolLM2-135M
measured -24%. The reason is in the formula: when `n_y <= SWIZZLE_Y` the whole
grid is one band, the remap becomes plain column-major, and it is now the
*weights* being re-read `n_x` times instead of the activations. That is the
wrong trade whenever B is bigger than A, which is every small model - their
largest n_y is 12, so they never had enough column blocks to band in the first
place.

So the shader takes the swizzle only when `n_y >= SWIZZLE_Y`. The test is
uniform across the dispatch and costs no divergence.

Per-model, pp512, F16, sw1 vs guarded sw16, interleaved:

| model                | largest n_y | result                        |
|----------------------|-------------|-------------------------------|
| Phi-3-mini 3.8B      | 128         | 565/575 -> 1136/1183, 2.03x   |
| granite-3.0 1B MoE   | 8           | 2617/2617 -> 2680/2623, wash  |
| SmolLM2-135M         | 12          | mean 15219 -> 15632, wash     |
| SmolVLM2-256M        | 12          | wash (see note)               |

The three small models all fall under the guard and run the identity path, as
intended. Note their absolute numbers are useless for A/B on their own: back to
back runs of the *same* DLL on SmolVLM2 spanned 13756-16723, an 18% spread. The
SmolLM2 row needed four alternations at -r 4 before the sign settled; two
passes had shown a convincing-looking -6% that was not there.

### Where this leaves the Vulkan gap

Phi-3-mini F16, both backends measured in one window:

| workload | DX12 before | DX12 after | Vulkan | gap after |
|----------|-------------|------------|--------|-----------|
| pp512    | ~570        | 1195.6     | 1461.3 | 1.22x     |
| pp2048   | ~500        | 951.8      | 1077.1 | 1.13x     |
| tg128    | 14.37       | 14.37      | 14.60  | 1.02x     |

From 3.57x behind to 1.13-1.22x, with a shader-side change and no driver fix.
Decode is untouched, as expected - flag 264 is a prefill path.

This is orthogonal to the groupshared `Matrix::Load` bug: swizzling cuts how
many times a tile is fetched, LDS staging would cut how many times it is
fetched *within* a group. Both attack bytes, at different levels. The
groupshared fix is still worth having, but it is no longer the whole story, and
the 2.9x in section 29 was measured against a baseline that no longer exists.

## 32. The same swizzle does nothing for the quant path, and why

Section 31 is a 2x on F16 prefill, so the obvious next move is to apply it to
the quantized GEMM. It does nothing, and the reason is worth writing down
because it redirects where the quant work should go.

Phi-3-mini Q4_K_M prefill is 84% three dispatches of the register-blocked Q8_1
integer-dot GEMM (flags 127/128/129, `mul_mat_q*k_q8_1_mmq.hlsl`):

    250.6 ms  36.8%  fl=127  K=3072  N=16384  M=512   3.5 GB/s
    144.8 ms  21.3%  fl=128  K=3072  N= 9216  M=512   4.3 GB/s
     64.8 ms   9.5%  fl=129  K=8192  N= 3072  M=512   4.8 GB/s

These have the axes the other way round from the F16 kernel - `groups_x` is
channels and `groups_y` is tokens - so the mirrored remap is wanted: y fastest,
so concurrent groups share a weight tile. At pp512 the grid is 256 x 4, and the
native order streams all 256 weight tiles once per token tile.

Built and measured, band 1 / 8 / 32 / 1024, all correct at 1216/1216:

| band | pp512        |
|------|--------------|
| 1    | 794.4, 789.5 |
| 8    | 789.1, 786.1 |
| 32   | 795.8, 788.6 |
| 1024 | 794.2, 798.7 |

Flat. Reverted, along with the shared `gemm_swizzle.hlsli` it was written
against - one no-op user is not worth a new header.

Two reasons, and the second is the interesting one:

1. This kernel stages both operands in groupshared and *its* groupshared works
   - it is an ordinary `uint` array, not a `Matrix::Load` (section 21). Its
   128x64 tile already gets 42 MACs per element loaded. The reuse the swizzle
   buys at the grid level, it already has at the group level.
2. It is not close to any hardware limit. 3.5-4.8 GB/s against the ~110 GB/s
   this part sustains, and about 6.4 TOP/s of dp4a against a much higher peak.
   It is neither bandwidth bound nor arithmetic bound, so there was nothing for
   a traffic optimization to recover.

That 4%-of-bandwidth figure is the finding. The quant prefill path is bound
inside its own inner loop - the K-quant unpack and the LDS traffic around it -
not by anything the dispatch shape can influence. That is the next place to
look for quant prefill, and it is a much bigger job than a group remap.

## 33. Where the Q4_K prefill GEMM actually spends its time (and why LinAlg will not help it)

Section 32 said the quant path runs at 4% of sustainable bandwidth and is
bound "inside its own inner loop". That was a guess. This section prices each
part of that inner loop with deliberately-wrong probes, holding everything
else fixed. Model Phi-3-mini-4k Q4_K_M, pp512, flag 127, interleaved DLL swap,
two passes each. Base measured 789.6 / 795.2 / 793.7 / 795.2 / 801.0 / 783.3.

| probe | what it removes | pp512 | delta |
|---|---|---|---|
| base | - | ~793 | - |
| MMQ_PROBE_NODP4A | 256 dot4add per thread per K block, replaced by integer add | 737.8, 743.7 | **-6%** |
| MMQ_PROBE_NOSCALE | the packed 6-bit scale decode (5 scalar global loads per K block) | 814.5, 810.5 | +3% |
| MMQ_PROBE_NOBAR | both GroupMemoryBarrierWithGroupSync per K block (192 total) | 866.8, 864.3 | +9% |
| MMQ_PROBE_NOLDS | 7 of every 8 LDS reads (quad 0 reread, dp4a count held) | 886.7, 883.4 | +12% |

Two things fall out.

**The arithmetic is free.** Deleting every dp4a made the kernel *slower*. The
MACs are fully hidden behind memory. This is the direct answer to "can LinAlg
accelerate Q4_K_M or Q8_0": matrix cores accelerate MACs, and the MACs here
cost nothing. Even a perfect s8 matrix path would buy approximately zero.

There are hard blockers on top of that, in case the arithmetic ever does
become the constraint:
- groupshared Matrix::Load is broken on this part (section 21), so operands
  must come from a descriptor as plain int8 planes.
- descriptor Load truncates offset and stride to 4 bytes regardless of Align,
  and stride 34 specifically loses 2 elements on odd rows (section 21).
  Q8_0's block layout is 2-byte scale + 32 bytes of quants = stride 34. It
  hits that bug exactly.
- so any LinAlg quant path needs a de-interleaving pre-pass writing quants at
  stride K and scales into a separate plane. Weights are static across a run
  so that is cacheable, and the flag-264 F16 convert pre-pass is a precedent.
  It is buildable. It is just not worth building for a bottleneck that is not
  arithmetic.


**Everything named adds up to a quarter.** 6 + 3 + 9 + 12 is well under half
the runtime even before allowing that the probes overlap. The obvious suspect
for the rest is the weight staging global loads. Per K block each loader
thread issues

    raw = src0.Load4(blk_off + 16u + il * 32u + ld_part * 4u)

where il = tile_in_block / 2 and high_nibble = tile_in_block & 1. Blocks 2k
and 2k+1 share il, so they load the identical 16 bytes and differ only in
which nibble they keep. Half of all weight bytes fetched are thrown away.

That looked like a free 2x on weight traffic. It was built twice and it is
worth nothing. Both variants step the block loop by two and unpack both
nibbles from one Load4; they differ only in where the second tile lives.

| variant | where the extra tile lives | pp512, 4 alternations | vs base |
|---|---|---|---|
| base | - | 798.8 819.1 798.0 805.8 -> 805.4 | - |
| double-buffered LDS | tile_w_qs[2][..], LDS 7.5 KB -> 15 KB | 803.4 824.9 815.9 801.4 -> 811.4 | +0.7%, noise |
| register-held | raw kept in registers across the barrier, LDS unchanged | 769.4 773.7 769.7 771.3 -> 771.0 | **-2.6%, 4/4** |

Both passed MUL_MAT 1216/1216, so this is a performance result and not a
correctness artifact. The double-buffered form also halves the barrier count,
which the NOBAR probe above prices at 9% on its own, and it still could not
clear the noise floor. The register-held form gives up 2.6%, which is the cost
of the extra live state.

So the weight global loads are not the bottleneck either. The full accounting:

| component | share of runtime |
|---|---|
| dp4a arithmetic | none (negative) |
| scale decode | 3% |
| barriers | 9% |
| LDS reads | 12% |
| weight global loads | none |

Nothing in this kernel is expensive. That pattern - every component priced
near zero, and any change that trades per-thread state for less work coming
out negative - is what latency-bound and occupancy-limited looks like. The
group is 256 threads carrying 32 accumulator floats each plus 7.5 KB of LDS,
and there are not enough groups resident to hide memory latency.

The lever is therefore the opposite of everything tried here: shrink
per-thread state (MMQ_TM / MMQ_TN) to raise occupancy, rather than spend state
to save traffic. That is a tile sweep, and it is entangled with the host-side
GGML_DX12_MMQ_BM / GGML_DX12_MMQ_BN constants and the separate _64 variant, so
it has to be done as one deliberate host-plus-shader change and not a shader
edit. Logged as todo mmq-occupancy.

Method note: NODP4A coming out slower than base is the useful kind of result,
and so is this section's headline. Two builds killed a change that a traffic
count said was a free 2x. Count bytes to form a hypothesis, never to conclude.
---

## 34. Q2_0 portable dp4a matvec

`mul_mat_vec_q2_0_dp4a_portable.hlsl` extends the Q2_0 Q8_1/dp4a matvec to wave16 and wave32 devices without changing the wave64 RDNA4 shader. Each wave reduces locally and group shared memory combines the wave sums for the four output rows.

The path is default-on for `DX12_ARCH_INTEL_UHD`, where Qwen3-0.6B Q2_0 decode improved from 23.85 to 38.48 tokens/s (+61%) and the 4096x14336 operator improved from 2569 to 1117 us (+2.30x). It is also default-on for AMD Strix Point device `0x150E`, where SmolLM2-135M Q2_0 decode improved from 184.4 to 263.4 tokens/s (+42.8%) and prompt processing improved from 373.0 to 378.6 tokens/s (+1.5%). All 1215 MUL_MAT backend cases pass with the route forced, and four fixed Wikitext chunks produced identical perplexity with the portable path on and off.

NVIDIA stays on the generic path by default. The isolated operator improved from 66.18 to 37.96 us (+1.74x), but Qwen3-0.6B decode regressed from 471.7 to 366.0 tokens/s because the Q8_1 prepass cost is not recovered across the full graph.

`DX12_Q2_0_DP4A_PORTABLE=1` forces the portable path on other dp4a devices. Set it to `0` to disable a device default. The existing `DX12_Q2_0_DP4A` threshold still applies.

## 20. RDNA4 prefill sweep: pp128 through pp6144

RX 9070 XT Q8_0 profiling used SmolLM2-135M, Phi-3-mini-4k, and Qwen3-4B. At
pp6144, flash attention and GEMM are both material: final-chunk FA time was
8.7 ms at D=64, 51.7 ms at D=96, and 86.2 ms at D=128. The corresponding
static GEMMs took about 5, 70, and 75 ms.

Reverse causal query-group scheduling is retained in `flash_attn_linalg.hlsl`.
It dispatches the query groups with the most causal work first, matching
`flash_attn_pf.hlsli`. SmolLM2 and Phi-3 were neutral. Six paired Qwen3-4B
pp6144 rounds improved from 3848.9 to 3868.3 t/s (+0.5%). The correctness
envelope is unchanged: D=64 and D=96 pass, quantized D=128 cache cases pass,
and the existing 271 D=128 F16 failures remain.

NVIDIA keeps forward query-group order in its dedicated D=64/D=96/D=128
blobs. On RTX 5070, applying the reverse order globally reduced Qwen3-4B F16
pp6144 by about 0.6% in balanced A/B testing.

The most useful non-kernel lever is ubatch size. `-ub 1024 -b 2048` reduced
the number of pp6144 chunks from 12 to 6 and improved:

| model | `-ub 512` | `-ub 1024` | change |
|---|---:|---:|---:|
| SmolLM2 Q8_0 | about 42100 | about 50300 | +19% |
| Phi-3 Q8_0 | about 4710 | about 4830 | +2.5% |
| Qwen3-4B Q8_0 | about 3850 | about 3990 | +3.7% |

Larger ubatches retained the narrow-model gain but regressed Phi-3 and Qwen.
This remains a caller setting because its memory cost and batching semantics
cannot be selected safely by the backend.

The following kernel and routing experiments were rejected:

- `DX12_FA_SPLIT_GROUPS=256..2304`: the existing target of 512 was best.
  More splits added reduction overhead and lost 2-7%.
- `DX12_LINALG_MM_GROUPS=32..192` at M=1024: 32-64 were effectively tied
  for the large models. Values at or above 96 severely regressed SmolLM2,
  and 192 also regressed Qwen.
- Native F16 probability exponentiation in LinAlg FA: SmolLM2 -1.0%,
  Phi-3 -0.4%, Qwen -0.5%.
- D=128 BR=16/BC=64, copied from the Vulkan coopmat1 geometry: Qwen pp6144
  fell from 3887 to 2539 t/s (-34.7%). More query groups did not compensate
  for the extra output and barrier work.
- A runtime aligned-tile predicate intended to skip Q8_0 GEMM bounds paths:
  Phi-3 lost 2.6-8.1% and Qwen lost 6.1-8.2% at pp128/512/1024. The extra
  dynamic predicate inhibited shader optimization more than the checks cost.

Short-prompt profiles confirm that another FA-only change is not the main
opportunity below pp1024:

| model | prompt | FA | GEMM |
|---|---:|---:|---:|
| SmolLM2 | 128 / 512 / 1024 | 6% / 20% / 24% | 84% / 70% / 66% |
| Phi-3 | 128 / 512 / 1024 | 6% / 6% / 9% | 88% / 88% / 85% |
| Qwen3-4B | 128 / 512 / 1024 | 3% / 7% / 11% | 91% / 85% / 82% |

The current tile selector and group target already survived independent
sweeps, and adding a runtime full-tile path regressed. Further short-prompt
gains need a structurally different quantized GEMM path, not another bounds
check, group threshold, or existing tile permutation.

### Live RDNA4 CoopMat and LinAlg capability comparison

The RX 9070 XT Vulkan driver reports `KHR_coopmat`, not NV coopmat2. The active
Vulkan path is subgroup-scope CoopMat1. The D3D12 Agility 1.721.3 preview
reports LinAlg tier 1.0 with these relevant operations:

- wave32 and wave64 F16 x F16 -> F16/F32, shape 16x16x16;
- wave32 and wave64 S8/U8 matrix multiply -> S32, shape 16x16x16;
- F16 thread vector-matrix multiply and F16 -> F32 outer product;
- no threadgroup matrix multiply on this AMD driver.

The basic matrix arithmetic is therefore similar to Vulkan KHR CoopMat, but
the usable dataflow is not. Vulkan can directly load and store aligned
cooperative matrices and can access subgroup cooperative-matrix elements.
On this driver, LinAlg descriptor `ColMajor` loads are incorrect,
`GetCoordinate()` exposes only half of an accumulator, direct descriptor
stores from the production quantized GEMM are incorrect, and accumulator
throughput falls as a wave owns more accumulators. These are driver/runtime
limits, not missing tile-selection logic.

Three further experiments after the capability audit were rejected:

- Reusing the per-tile mask scan to classify all-zero mask tiles avoided mask
  arithmetic but lost 5-7% on SmolLM2 and 1-3% on Phi-3/Qwen at both
  `-ub 512` and `-ub 1024`.
- Direct `Matrix::Store` to fully aligned F32 GEMM output tiles failed five
  `MUL_MAT` cases with about 1.0 relative error. The correct LDS store/reload
  remains required on RDNA4.
- Replacing per-thread mask-scan atomics with one atomic per wave was correct
  but neutral to -0.3% across the same six pp6144 configurations.

The integer LinAlg GEMM was already tested separately and remains rejected:
Q8_0/Q8_1 scales force an S32 accumulator close and rescale every 32 K
elements, and the required LDS readback leaves it slower than the mature DP4A
kernel.

This leaves no measured shader-only optimization with a positive return on
driver `32.0.23041.2023`. Vulkan's remaining lead is explained by its better
working matrix load/store and accumulator behavior, not by an untried
equivalent LinAlg switch. Revisit after an AMD driver or Agility update adds
working descriptor matrix operations, complete accumulator element access,
or threadgroup matrix multiply.

The exact wave-shape capability gate must query the selected wave size. This AMD
driver reports the 16x16x16 F16/F32 shape for wave32 and wave64, but returns no
shapes when queried with `WaveSize=0`. Treating the zero-wave result as final
disabled all LinAlg GEMM, attention, and convolution routes and reduced
pp6144 from 42.8k to 10.8k t/s on SmolLM2, 4.79k to 2.70k on Phi-3, and 3.91k
to 1.89k on Qwen3-4B. Querying the selected wave size first, with a zero-wave
fallback, restores the reported capability and the previous correctness and
performance envelope. A shape advertised only for a different wave size does
not establish support for the compiled shader.

## 35. Correction: the groupshared matrix load was never broken

Section 21 reported that `Matrix::Load` from groupshared returns 0x7F fill on
Xe3, and sections 26 and 30 built on that: if LDS cannot stage a matrix, then
registers are the only staging area, the tile cannot grow past BM=32/BN=128,
and every wave must re-read both operands from the descriptor. That premise is
wrong. The probe was miscoded.

Intel's reply: the groupshared load/store semantics changed in late July, and
shipping drivers still implement the older rule - the value is CONVERTED when
the array element type does not match the matrix component type.

Under that rule the old repro measured a conversion, not a load. It declared
`groupshared int a_arr[]` and packed four i8 per slot, against a matrix of
`ComponentType::I8`. Each slot held 0x03030303 = 50529027, and converting that
to i8 saturates to 127. Hence 32*127*127 = 516128, and hence the result never
moved when FILL_A went 1 -> 3 -> 100: every value saturates to the same 127.

The old repro already contained the counter-evidence. Its accumulator `Store`
worked, and it is the only operation there whose types already matched
(`ComponentType::I32` into `groupshared int`).

`linalg_probe_gs_typed.hlsl` re-tests with matching types on the same part and
driver (B390, 32.0.101.8992):

| test | expected | observed |
|---|---|---|
| f16 8x16x16 from `groupshared float16_t`, stride K | 32 | 32, all 128 cells |
| f16 control, ordinary load | 1 | 1 |
| I8 8x32x16 from `int[]`, one element per slot, stride K | 96 | 96, all 128 cells |
| I8 control, ordinary load | 3 | 3 |

Both work. Note the I8 case: with the conversion rule the array is a plain
array of elements, so the stride is K, not K/4, and no packing is involved.

What this reopens - none of it measured yet:

- Tile size. Section 26's sweep stands as data about the register-only shader,
  but "cannot grow past BM=32/BN=128" no longer follows.
- Operand reuse. Section 30 measured `Matrix::Load` as the whole runtime and
  per-byte in cost. Staging in LDS is the standard answer to that and was
  ruled out on the bad probe.
- Quants. Section 33 concluded LinAlg cannot help the Q4_K path. The dp4a
  attribution there is still valid, but Vulkan's design - dequantize to f16
  into groupshared, then one matrix load for every type - is now reproducible
  here. That also sidesteps the stride-34 descriptor truncation that blocked
  Q8_0, since the dequantized tile is dense f16.

Caveat before anyone spends a week on it: this proves the groupshared load is
FUNCTIONAL, not that it is FAST. Whether staging through LDS beats the
descriptor path on this hardware is a separate measurement.

The descriptor-side findings in section 21 - the 4-byte offset/stride
truncation and the stride-34 element loss - were measured through a different
path and are not affected by this correction.

### Forward compatibility: the semantics are changing under us

Intel's reply points at hlsl-specs PR 879, which revises groupshared Load,
Store and InterlockedAccumulate. Shipping drivers implement the OLD rule (the
conversion described above). Two things follow for any LDS staging work.

1. Match the types. The spec constraint on the groupshared overload accepts an
   array whose element type either matches the matrix component type OR is
   integral / packed-integral, which is why `groupshared int` with an I8 matrix
   compiled at all. PR 879 exists to let the conversion be BYPASSED for
   reinterpretation, aligning with SPIR-V and Metal. Code that already matches
   types should read the same under both rules, since the conversion is a no-op
   when there is nothing to convert. Not yet confirmed by Intel - ask before
   relying on it.

2. New alignment validation. PR 879 adds a rule that the groupshared stride is
   16-byte aligned, and the offset 128-byte aligned when both are constants
   (DirectXShaderCompiler issues 8634 and 8636). This constrains the staging
   tile: an f16 row pitch must be a multiple of 8 halves, so the usual trick of
   padding LDS rows by 1-2 elements to break bank conflicts is not free - pad to
   8. Tile start offsets want 64-half alignment.

   `linalg_probe_gs_typed.hlsl` already complies: the f16 stride is 16 halves
   (32 bytes), the I8 stride 32 ints (128 bytes), and both offsets are 0. So it
   should stay valid when drivers move to the new rule.

The risk to watch is the mixed configuration - a DXC new enough to enforce the
879 validation running against a driver still on the old semantics.

The Intel Xe3 wave GEMM must pin its compiled wave size and only route weights
contiguous along K. Its indexing derives wave ownership from `WAVE_SIZE`, and
its weight loads assume the packed element or block layout. The production
route therefore uses `WAVE_SIZE_ATTR` and requires
`nb[0] == ggml_type_size(type)`.

## 36. LDS staging the activations: 1.4x on the Xe3 wave GEMM

Section 35 proved the groupshared matrix load works. It did not say whether it
is FAST. It is.

The flag-264 GEMM read both operands straight from the descriptor. Look at the
addresses: `b_base` depends on `wave`, but `a_base` does not. Every wave in the
group was loading the SAME activation rows. At NWAVE=4 that is four fetches of
one strip.

Traffic per group per 16-deep k-block, at BM=32 / BN=128:

| operand | unique bytes | bytes actually fetched |
|---|---|---|
| A activations | 1 KB | 4 KB (NWAVE copies) |
| B weights     | 4 KB | 4 KB (each wave owns its own columns) |

So a quarter of the fetched bytes were the only ones worth fetching on the A
side. `LDS_STAGE_A` moves that strip into `groupshared float16_t lds_a[BM*FOLD_K]`
once per fold window, with all THREADS cooperating on uint4 loads, and the
matrix load then reads LDS. The array element type matches the matrix component
type, so nothing converts (section 35).

Measured on Phi-3-mini F16 pp512, interleaved in one thermal window after 420 s
idle, three repeats per point:

| round | descriptor | LDS-staged A |
|---|---|---|
| 1 | 1048.31 | 1638.32 |
| 2 | 1186.70 | 1625.98 |

That is 1.37x on the settled pair and 1.46x on the first. Note also that the
staged numbers barely move between rounds while the descriptor numbers swing
13%: the descriptor path is bandwidth bound, so it tracks thermal state, and
the staged path does not.

This is consistent with section 30, which found the descriptor load to be
essentially all of the runtime and its cost to be per-byte. Cutting fetched
bytes 8 KB -> 5 KB per group per k-block predicts about 1.6x, and 1.4x is what
lands after the barriers and the staging stores are paid for.

Two consequences.

- B is NOT worth staging. Each wave owns its own column strip, so there is no
  redundancy inside the group to remove. Its traffic falls out of the tile
  shape instead: bytes per output element are 32*(1/BM + 1/BN), which is what
  MTILE and NTILE control.
- The quantized path is now plausible on this part. `mul_mat_linalg_f16.hlsl`
  already dequantizes Q8_0 and Q4_K into a `groupshared float16_t` tile
  (QUANT_MODE 8 and 4) and was only ever blocked here by needing a 16x16x16
  shape with an F32 accumulator. A dense f16 staged tile also sidesteps the
  stride-34 descriptor truncation of section 21, since the staged tile is
  dense and the 34-byte Q8_0 block is never handed to a matrix load.

Inactive waves must not return early any more: the staging barriers are
group-wide, so a wave whose columns are all past ne0 stays in the loop and
just skips its own multiply and store.

### The tile still cannot grow, but the reason changed

Section 26 blamed the tile cap on having no staging area. That reason is gone,
so the sweep was repeated with the activations staged. MTILE 4 -> 8, holding
NTILE=2, so BM 32 -> 64:

| round | BM=32 BN=128 | BM=64 BN=128 |
|---|---|---|
| 1 | 1611.38 | 1350.22 |
| 2 | 1609.23 | 1356.15 |

16% slower, even though bytes per output element drop from 1.25 to 0.75. The
binding constraint is the accumulator set, not the operands: MTILE*NTILE
accumulators of ACC_E halves each, plus the same count of F32 drain registers
(`sum[MTILE][NTILE][ACC_E]`). Going 4x2 -> 8x2 doubles both, from 8 to 16
accumulators and 64 to 128 drain floats, and it spills.

So the cap in section 26 holds. Staging cannot lift it, because staging does
not touch the accumulators.

### And the last untested tile shape loses too

Section 26's sweep never tried MTILE=8 with NTILE=1, the one variant that grows
BM without growing the accumulator set: 8x1 has the same 8 accumulators as 4x2,
while bytes per output element drop from 1.25 to 1.0. With staging in place it
was worth a measurement.

Phi-3 F16 pp512, order balanced within each round (m4n2, m8n1, m8n1, m4n2) so
the warm-up ramp cannot favour either arm:

| round | BM=32 BN=128 | BM=64 BN=64 |
|---|---|---|
| 1 | 1551.34 / 1599.68 | 1423.34 / 1429.67 |
| 2 | 1573.25 / 1631.95 | 1398.17 / 1463.81 |
| 3 | 1636.95 / 1607.65 | 1453.81 / 1474.06 |

About 10% slower, every pair, despite fetching fewer bytes per output.

The reason is A operand reuse inside the wave. Each loaded MatAf feeds NTILE
MultiplyAccumulates, so NTILE=2 halves the A loads per MAC and NTILE=1 gives
that back. That outweighs the traffic saved.

So the tile is fixed from both sides: MTILE cannot grow because of the
accumulators, and NTILE cannot shrink because of A reuse. BM=32/BN=128 stands.

### A note on measuring this box

Phi-3 F16 pp512 read 1253 and then 1299 immediately after two full
test-backend-ops runs, against 1610 in section 36 - which looks like a 20%
regression and is not one. The same binary read 1475 on a warm-up pass and
1600-1637 once warm. Idling is not enough; the part also needs a discarded
warm-up run before the first arm, and arms must be order balanced. An A/B
against a HEAD-built DLL confirmed F16 was untouched: 1528.7 vs 1535.0.
## 37. Q8_0 through the matrix path: a tie, three times over

Section 36 made the F16 wave GEMM 1.4x faster by staging the activations in
LDS. Section 35 had already removed the reason quants were excluded from the
matrix path at all. So the obvious next step was Vulkan's design: dequantize
the weights into a groupshared F16 tile and let one matrix kernel serve every
type. That also sidesteps the stride-34 descriptor truncation of section 21,
because the staged tile is dense F16.

It was built (flag 265, the same shader with `QUANT_B=8`) and it works. It is
also, on this part, exactly as fast as the dp4a MMQ kernel it would replace.

### The measurements

Qwen3-4B Q8_0, pp512, interleaved in one thermal window, `-r 2`, three rounds.
`DX12_LINALG_Q8_WAVE=1` selects the matrix path, `0` the dp4a MMQ path.

| round | matrix path | dp4a MMQ |
|---|---|---|
| 1 | 814.16 | 817.64 |
| 2 | 815.43 | 813.01 |
| 3 | 818.93 | 822.39 |

pp6144 agrees: 336.86 / 323.62 against 324.57 / 322.14.

Three separate attempts to move it, each measured the same way:

| change | rationale | result |
|---|---|---|
| FOLD_BLOCKS 4 -> 2 | 20 KB of LDS per group caps occupancy; halve it | 812.99 / 809.84 / 815.93 vs 815.94 / 819.12 / 816.84 |
| MTILE 4x NTILE 2 -> MTILE 8x NTILE 1 | BM 32 -> 64 doubles how far the dequant cost amortizes, and holds the accumulator count at 8 | 809.29 / 813.29 / 814.90 vs 805.74 / 806.31 / 812.88 |
| one thread per Q8_0 block | the first version re-read the block scale for every four quants and used two unaligned dword loads, so it fetched about three bytes for every useful one | 809.13 / 815.40 / 816.86 |

Every number in this section is between 805 and 823. Nothing moves it.

### Why

A 2.7x cut in weight read amplification changing nothing is not a small
result - it says the Q8_0 prefill is not bound by the weight fetch, and by
extension not by this GEMM's operand traffic at all.

That is the same conclusion section 33 reached from the other direction, by
pricing the dp4a MMQ kernel component by component and finding its arithmetic
already free. Matrix cores accelerate arithmetic. Section 33 measured that
removing the arithmetic entirely was *slower*. So there was never headroom for
a matrix kernel to take, and swapping one for the other lands on the same
number - which is what happened.

### The small-N cliff

Where the matrix path does differ, it is worse. `test-backend-ops perf`,
m=4096, k=14336, GFLOPS, two rounds averaged:

| n | matrix path | dp4a matvec | ratio |
|---|---|---|---|
| 2 | 68 | 375 | 5.5x slower |
| 3 | 110 | 542 | 4.9x slower |
| 4 | 147 | 685 | 4.6x slower |
| 5 | 183 | 725 | 4.0x slower |
| 8 | 289 | 817 | 2.8x slower |

The cause is occupancy, not the kernel. At n=8 the dispatch is
ceil(8/BM) x ceil(4096/BN) = 1 x 32 groups, and the machine idles; the cost is
flat at about 3.2 ms for every n in that range. So the gate now needs
`ne[1] >= DX12_IWQ8_BM` on the Q8_0 route.

Worth noting the F16 route has no such cliff - it is about 2x *faster* than its
fallback at the same shapes, because there the fallback is a plain F16 kernel
rather than a tuned dp4a matvec. The same gate is right for one type and wrong
for the other.

### What ships

`DX12_LINALG_Q8_WAVE` defaults to 0. The dp4a MMQ path keeps the traffic. The
code stays because it is a one-variable A/B, and the interesting question is
not settled forever - the groupshared semantics change of PR 879 is still not
in a shipping driver.

Q4_K was the other requested type and is deliberately not attempted. Its decode
is strictly heavier than Q8_0's - 256-element super blocks with 6-bit packed
scales - so it would carry a larger staging cost into the same kernel that
already fails to beat dp4a with the cheapest possible decode. Section 33 priced
Q4_K MMQ directly and reached the same place.



## 38. The Vulkan quant gap is not the GEMM tile, the drain, or operand traffic

Section 37 measured the Q8_0 matrix route at parity with dp4a in a model run
and concluded the prefill was not bound by that GEMM. User benchmarks then
contradicted the spirit of that: at pp6144 Vulkan beats us by 25-32% on Q8_0
and Q4_K while our F16 sits at parity. Worse, Vulkan's Q8_0 (402 t/s on
Qwen3-4B) is faster than Vulkan's own F16 (335) - it turns the halved weight
traffic into speed, and we do not.

### Isolate the GEMM

`test-backend-ops perf -o MUL_MAT`, m=4096, n=512, k=14336, same shape both
backends:

| type | Vulkan | DX12 | gap |
|------|--------|------|-----|
| f16  | 16.07 TFLOPS | 10.31 | 1.56x |
| q8_0 | 14.89 | 5.07 MMQ / 5.87 matrix | 2.5-2.9x |
| q4_K | 11.74 | 4.40 | 2.7x |

Two things fall out immediately. Vulkan's quant GEMMs run at 93% and 73% of
its own F16 GEMM; ours run at 57% and 43%. And our F16 GEMM, which looks fine
at model level, is itself 1.56x behind - model-level F16 parity was masking a
real deficit, because F16 prefill is bound by weight traffic on this UMA part
rather than by the GEMM.

Note also that the flag-265 matrix route is 16% faster than MMQ here (5.87 vs
5.07). Section 37 measured parity at model level and shipped it off by
default. Both readings are correct; the model run simply does not spend enough
of its time in this GEMM for 16% of it to show.

### Three hypotheses, all measured, all wrong

**Operand traffic per output.** Vulkan's workgroup tile is 128x128, ours is
BM=32 x BN=128. Per output that is 0.0156 operand elements against 0.039 - a
2.5x advantage, suspiciously close to the 2.5x gap. Section 26 could not test
this because it only ever grew MTILE/NTILE, which grows the per-wave
accumulator set and spills. The wave grid is the other knob: arrange NWAVE as
a WAVE_M x WAVE_N grid and the group tile grows while the per-wave
accumulators stay put. NWAVE=16 as 4x4 gives exactly Vulkan's 128x128.

Built, 1216/1216 and 18035/18035 correct. Order-balanced A/B on the isolated
GEMM: base 10.36/10.27, grid 10.91/10.55. **+4%.** A 4x bigger tile and a 2.5x
traffic cut buy 4%, so operand traffic is not what binds. At model level
(Phi-3 F16 pp512, order-balanced) it is 1431.7 against 1420.7 - nothing.

**The f32 drain.** LinAlg has no f16xf16->f32 shape on this part (section 23),
so the f16 accumulator has to be drained to f32 registers every fold window:
MTILE*NTILE*ACC_E = 64 scalar Get()+add against FOLD_BLOCKS*MTILE*NTILE = 32
MultiplyAccumulates. Two scalar ops per matrix op looked like a plausible
1.5x. Probe: drain on the last window only, which keeps every load and MAC
live but gives wrong results, so it prices the drain exactly. Base 10.22/10.34,
no-drain 10.82/9.62. **Nothing.** The absence of an f32 accumulator costs us
approximately zero.

**Bigger tile for the quant route.** Dequant cost per output scales as 1/BM,
so the wave grid should pay here even if it does not for F16. Q8_0 at
WAVE_M=2, NWAVE=16 gives BM=128/BN=128. Measured +2.4% over MMQ, against
+16% for the shipped 64x64. Worse. The likely cause is LDS: lds_a and lds_b
at BM=BN=128 and FOLD_K=64 are 16 KB each, exactly the 32 KB budget, which
caps the group count per EU.

### Where that leaves it

The 8x16x16 shape is identical on both APIs (section 23), the drain the
missing f32 accumulator forces is free, and a 2.5x operand traffic cut is
worth 4%. So the deficit is not in the operand path or the accumulator path.
What is left is issue rate and occupancy around the matrix op itself, and the
F32->F16 activation convert pre-pass that Vulkan does inline in the shader
(flag 263 reads ~29 MB and writes ~15 MB for this shape, order 15% of the
5.83 ms run). Those are the next things to price.

The WAVE_M generalization is kept, defaulted to 1 so both shipped shaders are
byte-identical to before. It costs nothing at that setting, it is the
apparatus that closed the tile question, and it lets the other LinAlg parts
re-test the same sweep on different silicon - the accumulator budget that
makes 128x128 pay for Vulkan may exist there.

## 39. Coopmat vs LinAlg: the concrete differences, itemised

Section 38 left "issue rate and occupancy" as the residue. This section
enumerates every difference between the two APIs on this part that could
plausibly cost 1.56x, and prices the ones that are reachable.

### The inventory

| | Vulkan `VK_KHR_cooperative_matrix` | D3D12 LinAlg (SM 6.10 preview) |
|---|---|---|
| shapes | 8x16x16 f16, 8x32x16 s8/u8 | identical (section 23) |
| accumulator | f16 **and f32** | f16 only |
| scope | subgroup | wave (threadgroup shape advertised, unusable) |
| lane width used | 32 | 16 |
| load from groupshared | yes, stride in elements | yes, but see section 35 |
| load from a descriptor | no (must stage) | yes, stride truncated to 4 bytes (section 21) |
| status | shipping | preview, semantics still moving (PR 879) |

Only four entries actually differ, and three of them are now measured.

### Priced and rejected

**The missing f32 accumulator** costs approximately nothing. Section 38
removed the drain it forces and the GEMM did not move.

**Lane width.** `D3D12_FEATURE_DATA_D3D12_OPTIONS1` on the B390 reports
`WaveLaneCountMin=16, WaveLaneCountMax=32` (`tools/wave_caps_probe.cpp`). We
take Min on Intel, so the matrix GEMM runs 16-lane waves while Vulkan drives
the same XMX hardware from a 32-lane subgroup. That is a real difference and
the obvious suspect for a ~1.5x.

It is reachable: `[WaveSize(32)]` plus `WAVE_SIZE=32` recompiles the same
shader for 32 lanes (ACC_E falls from 8 to 4, THREADS rises to 128). Built,
1216/1216 correct. Order-balanced A/B on the isolated GEMM: wave16
10.28/10.71, wave32 10.22/9.76. **Wave 32 is about 5% slower.** The driver
evidently maps the 8x16x16 wave matrix onto the same hardware either way, and
at 32 lanes we simply get half as many waves to hide latency with.

**Descriptor loads**, which Vulkan does not have at all, are ours to keep -
section 30 measured them as the whole runtime, and section 36 replaced the A
side with LDS staging for 1.4x.

### One real defect found

`mul_mat_linalg_wave_f16_i.hlsl` carried `[numthreads(THREADS,1,1)]` with no
`WAVE_SIZE_ATTR`, unlike its sibling `mul_mat_linalg_f16.hlsl`, which pins the
wave and documents why. Both `ACC_E = LA_M*LA_N/WAVE_SIZE` and the
`tid / WAVE_SIZE` wave index are only valid if the dispatch really runs at
WAVE_SIZE lanes, and the header of `ggml_common.hlsli` records that Intel Xe
drivers have been seen picking a wider wave than the one a shader was tuned
for. It happened to run at 16 and be correct; nothing was asking for it.

Now pinned. Measured identical (10.28/10.71 against an unpinned 10.36/10.27),
which also confirms the driver was already choosing 16.

### What is left

Four hypotheses are now dead: tile size, operand traffic, the f32 drain, and
lane width. What remains unpriced is the F32->F16 activation convert pre-pass
(flag 263), which Vulkan folds into its GEMM and we run as a separate
dispatch - order 44 MB of traffic against a 5.83 ms GEMM for this shape. That
is the next thing to measure, and unlike the four above it is a scheduling
problem rather than a capability gap, so it is ours to fix.

## 40. The convert pre-pass was 15% off the memory roof; now it is on it

Section 38 left one unpriced lever: the F32 -> F16 activation convert (flag 263)
runs as a separate dispatch, where Vulkan converts inside its GEMM. Unlike the
five dead hypotheses in 38-39 this is a scheduling choice we own, not a driver
capability gap, so it was worth measuring properly.

### Pricing it

A convert cannot simply be switched off - the GEMM needs its output. So the
probe dispatches it N EXTRA times (`DX12_CVT_REP`, default 0) and reads the
slope. One dispatch is far below run-to-run noise on this part (two identical
arms measured 6056 vs 5195 us, 14% apart); at N=8 the signal clears it.

Order-balanced ladder, MUL_MAT f16 m=4096 n=512 k=14336, us/run:

    rep   run A   run B    mean
    0     5332.9  6130.6   5731.7
    4     8486.3  7743.9   8115.1
    8    10006.2  9667.5   9836.9

Slope = 513 us per convert. That is 9% of the whole 5732 us op - larger than
anything section 38 found, and the pre-pass had never been looked at.

### Why it was slow

The shader read TWO elements per thread with scalar `Load`s, i.e. 4-byte loads.
It moves 29.4 MB in + 14.7 MB out = 44 MB, so 513 us is 86 GB/s.

Widened to 8 elements per thread: two `Load4` feeding one `Store4`. The tail and
the pad region keep the old scalar path, so the `ne1` source-extent guard and the
zero fill are unchanged. Host group count moves from ceil(n/2) to ceil(n/8) in
lockstep - the shader and the host linearisation must always agree.

Same ladder after the change:

    rep   run A   run B    mean
    0     5458.0  5059.3   5258.6
    4     7099.0  7266.5   7182.7
    8     8659.5  8836.3   8747.9

Slope = 436 us per convert, 15% off. 44 MB / 436 us = 101 GB/s.

### Why this lever is now CLOSED

101 GB/s is not an arbitrary number. `test-backend-ops perf -o CPY` on the same
device tops out at 101.66 GB/s (f32->f32, 64 MB) and 99.72 GB/s (24 MB); every
smaller case falls off into latency. The kB/run figure counts both sides, so it
is directly comparable to the convert's 44 MB.

The convert is therefore ON the measured streaming roof. No further shader work
can help; only removing the traffic could.

### Do not fold the convert into the GEMM

The obvious "just convert during LDS staging" idea is a traffic regression, and
the arithmetic says so before any code is written. A activations are re-read once
per column group, i.e. ne0/BN = 4096/128 = 32 times for the reference shape.
Reading them as F32 there costs an extra 14.7 MB x 32; the pre-pass costs 44 MB
ONCE. The separate dispatch is the cheap option by more than an order of
magnitude - Vulkan's fused approach is not a model to copy on this part.

### Net

Convert 513 -> 436 us, so about 1.5% of the reference GEMM op, and less at model
level because the existing reuse cache already shares one conversion across the
qkv projections of a layer. Small, but free and correct (1216/1216 MUL_MAT,
18035/18035 full suite).

The probe stays in at `DX12_CVT_REP=0` - it is the only way to price this
dispatch, and it cost a rebuild to write.

## 41. MMQ is at a local optimum: both tile directions lose

Section 33 priced every MMQ component near zero (dp4a none, scales 3%, barriers
9%, LDS reads 12%, weight global loads none) and concluded latency/occupancy.
The standing todo was to shrink per-thread state to raise occupancy. Tested in
both directions; neither wins.

The tile is MMQ_BM x MMQ_BN = 128 x 64, from MMQ_TM=8 x MMQ_TN=4 per thread
over a 16x16 thread group, so 32 accumulator floats per thread.

### Down: 64x64 (MMQ_TM=4)

The `_mmq_64` variants already exist but are gated to NV Pascal+ AND ne0 <= 2560,
so they could not be reached on this part at the reference shape. Added
`DX12_MMQ_NARROW64=2` to force them regardless of width - the gate is otherwise
unreachable for measurement.

MUL_MAT m=4096 n=512 k=14336, us/run, order-balanced:

    type   128x64          64x64
    q8_0   8836 / 8768     (22835 first, PSO compile) / 10764
    q4_K   9019 / 9015     10672 / 10639

q4_K 18% worse on two clean pairs. Halving MMQ_TM halves the outputs a group
produces while its LDS staging and barrier count stay put, so the dp4a work per
staged byte halves. Fewer accumulators did not buy back occupancy.

### Up: 128x128 (MMQ_TN=8)

Matches Vulkan's `l_warptile`. 64 accumulator floats per thread, LDS ~10 KB.
All five shaders on the shared host dispatch (q8_0, q4_K, q5_K, q6_K, q2_K) take
MMQ_TN by `#ifndef`, and `GGML_DX12_MMQ_BN` must move with them or the group
count covers a fraction of the output.

Correct (1216/1216) but:

    type   128x64   128x128
    q8_0    9368     12759     36% worse
    q4_K   11796     12090     2.5%, noise

### A methodology note worth keeping

The first q4_K reading made 128x128 look 34% worse. It was not: the 128x64
baseline had been taken with a narrow `-p` filter and the variant with a broad
one, and q4_K's baseline moves 9017 -> 11796 between them. The broad filter runs
many shapes first and leaves the part in a different thermal and cache state.

Re-measured with matched filters the q4_K difference is noise. ALWAYS take both
arms through the identical harness invocation, not just the identical binary.
The q8_0 result survived the correction; the q4_K one did not.

### Conclusion

128x64 is a local optimum on this part - down loses clearly, up loses clearly on
q8_0 and gains nothing on q4_K. The `mmq-occupancy` idea is closed. The 2.5x
Vulkan quant gap is not the MMQ tile shape.

`DX12_MMQ_NARROW64=2` stays as a measurement tool; the default gate is unchanged.

## 42. VTune: occupancy was never the problem, but fixing it won anyway

First profiler run on this backend. VTune 2026.3 `gpu-hotspots` attaches to
D3D12 fine; `characterization-mode=instruction-count` does NOT (it needs GTPin,
which asserts on D3D12), so only counter-level data is available.

### Two harness traps found first

`GPU Time` is NOT a comparison metric here. test-backend-ops targets a fixed
wall time per shape, so both backends spend ~1.0s of GPU time by construction.
An early DX12-vs-Vulkan reading of 40.5s vs 27.2s was pure artifact: the two
backends support a different number of types, so the sweep ran more shapes.
Compare `us/run` at equal GPU time instead.

Single-shape `-p` filters lose the result line to the exit truncation bug. Use
`-p "f16,type_b=f32,..."` which also matches `bf16`; the f16 line completes and
bf16 absorbs the truncation.

### The measurement

Same shape (m=4096 n=512 k=14336), equal GPU time, profiled:

    backend/type   runs   us/run   Occupancy   XVE Stalled
    DX12  f16       194     5206      54.9%       69.2%
    Vulkan f16      244     4116      80.1%       73.8%
    DX12  q8_0       94    10790      73.0%       58.1%
    Vulkan q8_0     134     7522      70.8%       57.6%
    DX12  q4_K       90    11395      73.9%       56.4%
    Vulkan q4_K     172     5855      70.4%       48.1%

On the quants Vulkan wins 1.4-1.9x at MATCHING occupancy and MATCHING stall.
Equal residency, equal stall fraction, more work done: its instruction stream
does more math per instruction. That is not reachable by tuning tiles, and it
is why sections 38-41 found nothing.

F16 was different: a real 54.9% vs 80.1% occupancy deficit, previously inferred
(section 26 blamed spills) but never measured.

### Fixing the F16 occupancy

`float sum[MTILE][NTILE][ACC_E]` at line 220 is 4*2*8 = 64 floats per thread,
live across the entire K loop, plus MTILE*NTILE=8 f16 accumulator matrices.
That array exists ONLY because LinAlg has no f32 accumulator (section 39).
NTILE is the cheapest lever on it.

NTILE 2 -> 1 with NWAVE 4 -> 8 holds BM=32 and BN=128 exactly, so the tile, the
group count and all host routing are unchanged; only per-thread registers halve
and threads per group go 64 -> 128.

Result: occupancy 54.9% -> 82.4% (now above Vulkan), stall 69.2% -> 60.6%.

### Why the first reading said "no change"

A single profiled pair read 5206 (base) vs 5320 (new) and looked neutral. It was
not. The baseline's run-to-run spread is enormous and the new config's is not:

    base  : 4963, 6138, 6033, 6111, 5404, 5638   mean 5714  stdev ~450
    NTILE1: 5097, 5081, 5068, 5082, 5091, 5105   mean 5087  stdev ~13

Interleaved A/B/A/B with the DLL swapped between every run, repeated in both
orders. 11% faster on the mean and 35x more stable. A single sample of the
baseline can land anywhere in a 24% band, which is what made the first read
meaningless.

Model level, Phi-3 F16 pp512, interleaved both orders - all four B samples beat
all four A samples:

    base  : 1459.8, 1533.0, 1532.0, 1560.0   mean 1521.2
    NTILE1: 1538.1, 1567.2, 1562.1, 1599.0   mean 1566.6   +3.0%

SmolLM2-135M f16 pp6144 is neutral: the first block ran B second in every pair
and looked negative, but with the order reversed all three pairs favour B. When
thermal drift is this strong, only WITHIN-PAIR comparison means anything.

### Standing conclusions

- Occupancy is not the constraint on the quant paths - it already matches
  Vulkan. Do not spend more time there.
- Occupancy WAS suppressed on F16, and raising it is worth 11% GEMM / 3% model
  even though the earlier probes said the drain itself is free. Section 38
  priced the drain ARITHMETIC and correctly found it near zero; the drain
  REGISTERS are the real cost, and no A/B could have separated the two.
- The remaining Vulkan quant gap is instruction efficiency. Counter-level data
  cannot break it down further on D3D12, and GTPin is unavailable, so the next
  honest step is comparing generated ISA, not more tile sweeps.

## 43. Where the quant gap actually comes from: Vulkan runs quants on the matrix units

**Superseded by section 46.** The Xe3 quantized wave selections were overwritten by the later MMQ routing for output widths of at least 256. The reported large-shape opt-in comparisons did not exercise flags 265/266. The causal conclusions below are not established.

Section 42 left one thing unexplained: on q8_0 and q4_K, Vulkan does 1.4-1.9x
more work than us at the *same* occupancy and the *same* stall fraction. Equal
residency and equal stall rule out both the machine being empty and the machine
waiting. The only term left is what a running instruction does.

### The mechanism, read out of the Vulkan source

Vulkan does not have a separate quantized kernel on this path. It dequantizes
the weights into a groupshared F16 tile and then runs the *same*
cooperative-matrix GEMM it uses for F16:

- `vulkan-shaders/mul_mm_funcs.glsl` - `load_a_to_shmem()` has a per-type arm
  (Q8_0 at line 141, Q4_K at line 239) that unpacks to floats and writes
  `buf_a` (line 17).
- `vulkan-shaders/mul_mm.comp` - `buf_a` is F16 (line 132), and the COOPMAT
  variant loads it with `coopMatLoad` and multiplies with `coopMatMulAdd`
  (lines 314-319).

So its quant GEMM issues matrix instructions. Ours issues `dot4add` (MMQ, flags
104/162/163) on the vector ALUs. One 8x16x16 matrix op is 2048 MACs for the
wave; one dp4a is 4 MACs for a lane. That is the whole of "more math per
instruction", and it is exactly the kind of gap that tile tuning cannot close -
which is why sections 38-41 all came back empty.

### Our matrix path is not the weak part

We already have this architecture: flag 265, `QUANT_B=8` in
`mul_mat_linalg_wave_f16_i.hlsl`, dequantizing Q8_0 into a groupshared F16 tile.
Compare how far each backend falls from its own F16 GEMM going to Q8_0:

    Vulkan   q8_0 7522 / f16 4116 = 1.83x
    DX12     q8_0 8870 / f16 ~5000 = ~1.77x

We degrade no worse than Vulkan does. The quant staging is fine. The deficit on
q8_0 is inherited almost entirely from the base F16 matrix GEMM still being
behind, not from anything quant-specific.

### Negative: the section 42 register cut does not transfer to Q8

Q8 runs its own tile (`DX12_IWQ8_*`) with `sum[8][1][8]` - the same 64 floats
per thread that capped F16 at 54.9%. Tried the analogous cut, MTILE 8->4 with
WAVE_M 1->2 and NWAVE 4->8, which holds BM=64/BN=64 exactly and halves the
registers. 1216/1216. Interleaved A/B, four pairs:

    base 8351 / new 8825   (base is the low outlier here, first after warmup)
    base 8865 / new 8862
    base 8913 / new 8834
    base 8870 / new 8885

Neutral. Do not retry.

The profiler predicted this before the build: F16 was at 54.9% occupancy and
gained 11% from the cut; Q8 was already at 70.3% and gained nothing. Register
pressure is only worth cutting where a counter says residency is actually
capped. This is the first time a measurement here was correctly called in
advance instead of found by trial.

### The live lever

q4_K is where the gap is worst (11395 vs 5855, 1.95x) and it is the one type
with *no* matrix path on our side - it is MMQ dp4a only. The tell is that
Vulkan's q4_K is *faster* than its own q8_0 (5855 vs 7522) while ours is
*slower* (11395 vs 9273). With dequant amortized into a shared tile, q4_K wins
on traffic; with dequant done per dp4a step, it loses on ALU. Extending the
existing `QUANT_B` staging to Q4_K is the next thing to try, and unlike
sections 38-41 it is a capability gap, not a tuning knob.

## 44. Correction to 43: the quant GEMM is dequant-bound, not multiply-bound

**Invalid diagnosis; see section 46.** The comparison used the MMQ path on both sides. The instruction-count estimate also compares scalar element work with wave-wide matrix operations without measuring generated instructions, and cannot establish a dequantization bottleneck.

Section 43 read the Vulkan source correctly - it does dequantize Q8_0 and Q4_K
into a shared F16 tile and run them through `coopMatMulAdd` - and then drew the
wrong conclusion from it. The conclusion was that we lose because MMQ issues
`dot4add` on the vector ALUs while Vulkan issues matrix instructions. That is
testable: build the same thing for Q4_K and see.

Built it. New flag 266, `QUANT_B=4` in `mul_mat_linalg_wave_f16_i.hlsl`, one
thread per 32-element sub-block so the 6-bit scale pair is unpacked once and
reused for all 32 values. Correct first try, 1216/1216 and 18035/18035.

m=4096 n=512 k=14336, profiled back to back:

    q4_K MMQ (dp4a)     10356 us   Occ 65.2%   Stall 58.8%
    q4_K matrix (266)   10403 us   Occ 66.0%   Stall 59.3%

A dead tie, on all three numbers. Interleaved A/B without the profiler agrees:
neutral to +5%, inside the drift of that run.

### Why it ties, and what section 43 got wrong

Count the two terms per group per K window. The group dequantizes
`BN * FOLD_K` = 64 * 64 = 4096 values. It issues `MTILE * NTILE * FOLD_BLOCKS`
= 32 matrix ops per wave, 128 for the group. Over the whole GEMM:

    dequants   (4096/BN) * (512/BM) * ne00 * BN = ~470M, at ~7 instructions each
    matrix ops 4096 * 512 * 14336 MACs / 2048 per op = ~15M

The dequant is roughly 200x the matrix work. Moving the multiply from dp4a to
the matrix units optimizes a term that was never the cost - it is ~0.5% of the
instruction stream either way. That is why flag 265 tied MMQ for Q8_0 (section
37), why 266 ties it for Q4_K, and why the two tie each other.

So "Vulkan does more math per instruction" is real as an observation, but the
instructions that matter are the *unpack* ones, not the multiply ones. The
matrix unit is not the lever. Section 43's mechanism paragraph stands as a
description of Vulkan; its diagnosis of our gap does not.

### What the lever actually is

Dequant cost per output scales as 1/BM - each staged weight element is reused
by BM activation rows. We stage at BM=64; Vulkan's `l_warptile` stages at
BM=128, which halves its unpack cost per MAC for free. That is a tile-size
lever, and section 26 closed it only because groupshared staging was believed
broken. Section 35 retracted that. `LDS_STAGE_A` already exists, so both
operands can now be staged and the tile can grow past the register limit that
capped it at 32x128 / 64x64.

Next, in order:
1. Raise BM for the staged quant paths and re-measure. This is the one term
   that provably dominates, and it is the one Vulkan is ahead on.
2. Widen the unpack itself - the staging loop stores `lds_b` one `float16_t` at
   a time; Vulkan writes a vec2 per store.

Flag 266 stays in, default off behind `DX12_LINALG_Q4K_WAVE`, because it is the
only Q4_K path that can benefit from a bigger tile: MMQ's cost is spread over
every dp4a step and does not amortize at all.

## 45. BM=128 for the staged quants: neutral, so it is not dequant-bound either

**Invalid negative result; see section 46.** The modified quantized wave tile was not selected at the reported shape. Neither this experiment nor aggregate occupancy/stall counters rule out dequantization, tiling, or memory bottlenecks.

Section 44 predicted the win. Dequant cost per output scales as 1/BM, section
44 measured it as the dominant term, so doubling BM should have taken a large
bite. Built it: `DX12_IWQ8_WAVE_M` 1->2 with `NWAVE` 4->8, MTILE unchanged, so
BM 64->128 and BN stays 64. LDS goes 16 KB -> 24 KB, still inside the 32 KB
budget. 1216/1216.

q4_K m=4096 n=512 k=14336, matrix path against MMQ as the fixed reference,
interleaved (pair 1 is the usual cold-start artifact, MMQ ran first):

    BM=64    mmq 10329 / matrix 12913, 14384/14204, 15004/14347, 14206/13381
    BM=128   mmq 11165 / matrix 13366, 16016/15219, 16042/15400, 17547/16453

Both sit at about +5% over MMQ in the settled pairs. Doubling BM bought
nothing. Reverted - at equal speed the smaller tile is better, it holds 8 KB
less LDS and keeps the `ne[1] >= BM` gate at 64 instead of 128.

### What this rules out

Halving the number of dequants outright and getting no time back means the
dequant ALU work is not what the kernel is waiting on, so section 44's
arithmetic was directionally wrong even though its measurement (265 and 266 tie
MMQ) was right. Three explanations are now dead:

- the multiply unit (section 43 - matrix vs dp4a ties)
- the dequant ALU cost (this section - halving it is free)
- occupancy and tile shape (sections 38-41, and section 44's counters: MMQ and
  the matrix path sit within 1 point of each other on occupancy and stall)

Every path we can build lands at 10-11 ms while Vulkan does 5.9 ms. The paths
tie each other *because* they all wait on the same thing, and that thing is
none of the above.

### What is left

Weight re-read traffic is the untested candidate. The tile reads the weight
matrix `nrows/BM` times: 33 MB of q4_K re-read 8 times at BM=64 is 264 MB, and
at the 101 GB/s roof from section 40 that is 2.6 ms of the 10 ms. But BM=128
should then have returned ~1.3 ms and did not, which argues the re-reads are
being absorbed by cache rather than paid at DRAM.

The next honest step is not another tile probe. It is to measure DRAM bytes
directly on both backends at this shape. Attempted here with
`-knob collect-memory-bandwidth=true`: the collection runs, but the only
report the CLI will produce for a D3D12 or Vulkan result is `-report summary`,
and that emits `Max DRAM Single-Package Bandwidth` (the 124 GB/s ceiling) and
no measured traffic rows at all. Same limitation as the instruction-count knob
in section 42. Do not retry it from the command line - it needs the GUI, or a
different instrument.

Every structural hypothesis reachable by editing this kernel has now been tried
and priced at zero, so the next move has to come from a measurement, not a
guess.

## 46. Routing audit: the quantized wave opt-ins were overwritten

Quality caveat: the Q4 model timings in this section predate the activation scratch synchronization repair in section 50. They are historical timing observations, not validated correct-inference baselines.

On September 9, 2026, source inspection and actual per-operation traces found that the final Q8_0 and Q4_K MMQ selectors replaced flags 265 and 266. Their `linalg_took` predicates recognized only the older 211-218 variants. At output widths of at least 256, the default MMQ width threshold therefore applied even when the Xe3 wave opt-in was enabled.

Before the fix, SmolLM2 Q8_0 pp512 with `DX12_LINALG_Q8_WAVE=1` reported flag 104 for N=576 and N=1536, but flag 265 for N=192. After guarding the two final MMQ selectors, all three eligible widths report flag 265. During this routing audit the switches remained default-off; the later rollout is recorded in section 47.

The existing correctness sweep did not cover Q4_K at a wide output and at least 64 activation rows. Four cases were added to the existing MUL_MAT test list: Q8_0 and Q4_K at N=256/M=64/K=256, plus N=272/M=65/K=1536 with two batches. With both switches enabled, dispatch traces confirm flags 265/266 rather than inferring coverage from a suite pass count.

Vulkan dispatch tracing on the same SmolLM2 Q4_K_M file identifies `matmul_q4_k_f32_f16acc_aligned_l`, `matmul_q5_0_f32_f16acc_l`, `matmul_q6_k_f32_f16acc_aligned_l`, and `matmul_q8_0_f32_f16acc_l`. These are created from the coopmat1 shaders on this device. Its attention dispatch is `flash_attn_f32_f16_aligned`, the scalar path, not the cooperative-matrix attention path. The Intel scalar configuration uses four query rows, 32 KV columns, 128 threads, D-split eight, and no K/V shared-memory staging.

The actual pp6144 traces also change the optimization target. In the final SmolLM2 Q4_K_M microbatch, DX12 attributes 134.9 ms of 166.7 ms to attention (80.9%); most projections are Q5_0 because K=576. In the final Granite Q4_K_M microbatch, attention is 38.2% and expert MUL_MAT_ID is 52.1%. These are instrumented attribution samples, not uninstrumented throughput comparisons. A single large dense Q4_K GEMM cannot explain either workload.

Earlier absolute comparisons also used executables reporting different source commits. Both benchmark builds were rebuilt from the same source before continuing the audit. Profiler totals collected over multiple types or shapes must not be treated as per-kernel counters.

Across timed graphs 13-36, rather than only the final microbatch, SmolLM2 Q4_K_M attributes 69.7% to attention and 25.4% to MUL_MAT. Granite Q4_K_M attributes 63.3% to MUL_MAT_ID and 24.9% to attention. These percentages describe the instrumented DX12 runs only.

### Controlled experiments after the routing correction

A D64 attention experiment split each QK dot product across two lanes while retaining the existing wide tile. All 2036 D64 attention cases passed, but DLL-swapped ABBA SmolLM2 Q4_K_M pp6144 fell from 4714.01/4683.34 to 3591.21/3593.40 tok/s. The experiment was reverted. This rejects that layout, not Vulkan's different four-query, eight-way split design.

For quantized wave GEMM, changing MTILE/NWAVE/WAVE_M from 8/4/1 to 4/8/2 keeps BM=BN=64, halves each thread's long-lived F32 sum array, and doubles the threads available for staging. Unlike the earlier tile experiments, both arms now dispatch the intended matrix kernels. DLL-swapped ABBA at test dimensions m=4096, n=512, k=14336:

| Type | Original wave tile, us/run | Retuned wave tile, us/run | Throughput change |
| --- | --- | --- | --- |
| Q8_0 | 8852.25, 8990.68 | 7492.95, 7401.93 | +19.8% |
| Q4_K | 9530.61, 9158.75 | 7318.67, 7219.21 | +28.6% |

These are isolated GEMM gains over the corrected original matrix route, not gains over the default MMQ route or model throughput. The experiment changes both per-thread state and staging parallelism; it does not isolate a single hardware bottleneck. Q8_0/Q4_K dense and expert correctness coverage passed 244/244 after removing a separate unsuccessful Intel override of the AMD Q8 expert MMQ kernel. Quantized dense matrix routes were kept default-off until the model comparisons below.

The console test printer now flushes each result. Earlier native redirected perf runs sometimes exited successfully with the last result missing; the complete Q8 ABBA above confirms the measurements are now retained. An exit code without a matching result row or nonzero executed-case count is not evidence of coverage.

Uninstrumented pp6144, two repetitions per process, ordered default DX12 / retuned wave / Vulkan / Vulkan / retuned wave / default DX12, with 15-second gaps and an initial seven-minute cooldown:

| Model | Default DX12, tok/s | Retuned wave opt-ins, tok/s | DX12 change | Vulkan, tok/s |
| --- | --- | --- | --- | --- |
| SmolLM2-135M Q8_0 | 4961.52, 4865.13 | 5220.58, 5333.10 | +7.4% | 7747.30, 7769.24 |
| SmolLM2-135M Q4_K_M | 4660.11, 4720.45 | 4778.57, 4819.22 | +2.3% | 7645.04, 7514.82 |
| Qwen3-4B-Instruct-2507 Q8_0 | 320.07, 318.19 | 358.07, 350.64 | +11.0% | 430.64, 467.99 |
| Qwen3-4B-Instruct-2507 Q4_K_M | 312.76, 312.55 | 338.72, 341.03 | +8.7% | 469.46, 480.73 |

Each table entry is a process mean, not an individual repetition. Qwen's Vulkan runs have appreciably more variation (within-process standard deviations 11-30 tok/s) than the DX12 arms. Both dense opt-ins were enabled together. The paired DX12 improvements are real, but these runs do not close the Vulkan gap. The machine remained on its existing Balanced power plan; do not compare these absolute numbers directly with earlier runs or scripts that switch power plans.

### Scoped VTune collection

Fresh `gpu-hotspots` collections used `characterization-mode=global-memory-accesses`, `collect-memory-bandwidth=true`, and the exact single Q4_K m=4096/n=512/k=14336 filter. The instrumented MMQ / retuned wave / Vulkan runs reported 9937.34 / 7491.25 / 3776.42 us per operation. Occupancy was 73.9% / 71.7% / 70.2%; stalled-or-idle was 59.0% / 57.9% / 57.7%.

GPU time was approximately one second in all three because the perf harness runs each case for approximately one second: it completed 102 / 134 / 266 repetitions. Equal GPU time is not equal work. Similar aggregate occupancy and stall percentages still do not identify the limiting instruction sequence or eliminate cache/bandwidth effects.

The collection succeeded, but the attempted computing-task hotspot report was empty and the suggested `/GPUAvgGpuCoreFrequencyMHzMetric` query was unavailable in this installation. Those query failures did not establish missing counters; the follow-up below recovered time-windowed adapter metrics. VTune warned about Remote Desktop; Windows session state was explicitly checked and was `Disc`, not an active RDP session. Its automatic peak-bandwidth calibration also runs a separate workload before collection, another reason not to mix these timings with the uninstrumented comparisons.

The completed CLI audit established working reports: `-R hotspots -group-by=gpu-adapter` for hardware counters, and `-R timeline -report-knob query-type=overtime` with `GPUCoreFrequency`, `GPUMemoryReadGB`, `GPUMemoryWriteGB`, and `GPUL3ShaderReadThroughputGB`. `OvertimeBandwidth` separately exposes system-wide DRAM traffic. The accepted `GPUMemoryReadBandwidth` timeline query is not interchangeable with the GUI's `GPUMemoryReadGB` query and gave inappropriate scaling in the audit.

Applied to the three fresh single-shape Q4_K captures, equal-duration 0.6-second compute windows give:

| Route | Window, seconds | GPU memory read, GB/s | L3 read, GB/s | Interior-bin clock, GHz |
| --- | --- | --- | --- | --- |
| MMQ | 6.3-6.9 | 9.318 | 124.956 | 2.500 |
| Retuned wave | 7.4-8.0 | 18.992 | 189.807 | 2.492 |
| Vulkan | 4.05-4.65 | 61.280 | 477.560 | 2.482 |

Bandwidth comes from adapter hotspot reports with the stated `-time-filter` in seconds; clock is the mean of the complete frequency bins inside each window. Windows exclude the startup ramp and capture tail. These are adapter-level metrics during a controlled GEMM workload, including any associated prepasses, not individually named shader counters. The selected steady-state clock is comparable; it does not explain the remaining throughput gap. Higher bandwidth is a rate, not evidence of fewer bytes per operation, and no exact per-operation DRAM total has been established.

## 47. Xe3 expert GEMM using the existing buckets

Quality caveat: the Q4 model timings below predate the scratch synchronization repair in section 50 and must not be treated as validated correct-inference baselines.

The B390 never selected the existing matrix MUL_MAT_ID path: that path requires a 16x16x16 F32-accumulator shape the device does not advertise. Its actual grouped expert route was a scalar F16 tiled GEMM, not LinAlg.

Flag 267 adds the supported 8x16x16 F16 matrix multiply to `mul_mat_id_gemm.hlsli`. It reuses the existing expert bucket pass, quant decoders, activation loading, and output mapping. Activations convert to F16 while loading the shared tile, with no separate conversion dispatch. The matrix load uses matching `float16_t` arrays; partial F16 sums drain to F32 registers. This is gated on Intel, the exact advertised shape, wave16, and at least 128 routed rows per expert on average. It was introduced behind `DX12_LINALG_MMID_WAVE=1` for the comparisons below.

The first BK=16 implementation passed 237/237 F16/Q8_0/Q4_K/Q6_K expert cases. Dispatch traces confirm flag267, including output-column tails and long K=1536 reductions. Four small cases extend the existing test list rather than introducing a new test executable.

Paired pp6144 comparison using the same protocol as section 46, with both dense quant opt-ins and the new expert opt-in enabled in the candidate:

| Model | Default DX12, tok/s | Candidate, tok/s | DX12 change | Vulkan, tok/s |
| --- | --- | --- | --- | --- |
| Granite Q8_0 | 1486.54, 1487.43 | 2175.29, 2176.63 | +46.3% | 4282.97, 4255.66 |
| Granite Q4_K_M | 1242.99, 1261.81 | 1673.26, 1666.79 | +33.3% | 4354.72, 4349.94 |
| Falcon-H1-7B Q8_0 | 272.51, 275.09 | 319.20, 314.88 | +15.8% | 550.46, 550.73 |
| Falcon-H1-7B Q4_K_M | 261.89, 258.70 | 309.49, 308.38 | +18.7% | 532.40, 530.93 |

Falcon has no expert matmul in this comparison; its gain comes from the dense quantized wave route. These are improvements over DX12's previous defaults, not Vulkan parity.

Granite F16 was also compared independently with three repetitions per process. Scalar expert GEMM reported 1815.98 +/- 3.76 and 1735.40 +/- 146.21 tok/s; wave expert GEMM reported 2392.54 +/- 5.10 and 2399.00 +/- 3.41. The latter is about 32% faster even against the higher, stable baseline arm. Vulkan reported 3990.12 +/- 7.05 and 3960.02 +/- 29.25, so F16 also retains a substantial gap.

Increasing BK from 16 to 32 passed the same 237 correctness cases but regressed every measured Granite type. BK16/BK32 ABBA means were 2174.83/2177.24 versus 2019.02/2023.28 for Q8_0, 1670.49/1666.67 versus 1485.92/1556.91 for Q4_K_M, and 2395.99/2391.86 versus 2193.40/2196.53 for F16. That implementation changed B-load reuse as well as the shared-memory footprint and reduction depth. It was reverted; this does not rule out other larger-K layouts.

After these comparisons, the retained dense Q8_0/Q4_K wave routes use the existing Intel architecture-default helper instead of requiring opt-in. The expert wave route also defaults on behind its existing shape, wave-width, type, and routed-row gates. Other vendors retain their previous defaults. No environment setup is needed on the supported Intel configuration. Set `DX12_LINALG_Q8_WAVE=0`, `DX12_LINALG_Q4K_WAVE=0`, or `DX12_LINALG_MMID_WAVE=0` to restore the corresponding old route. The existing `DX12_LINALG_MMID=0` switch also disables the new expert route.

With the final defaults, the existing MUL_MAT, MUL_MAT_ID, MUL_MAT_VEC_FUSION, and MUL_MAT_ID_FUSION selections pass 3374/3374. A separate nine-case routing control observes flags 265/266/267 with no opt-ins, the old 104/119/122/127 paths with the three switches disabled, and no 267 when only the global MMID switch is disabled. All three controls pass.

An unfiltered DX12 `test-backend-ops test` run now repeats the supported default and opt-out route checks through the backend flag sink. The checks are capability-gated, use a replay-disabled backend, and restore the process environment afterward.

The remaining cost is model-dependent. Final default Q4_K_M traces, summed over timed graphs 13-24 of pp6144, attribute Qwen3-4B 58.7% to attention and 35.7% to dense MUL_MAT. Falcon-H1-7B attributes 57.0% to dense MUL_MAT, 22.0% to attention, and 10.9% to SSM_SCAN. These are instrumented DX12 operation shares, not clean backend throughput comparisons. Vulkan tracing confirms the F16-accumulator quant matrix pipelines in both models and scalar flash attention in Qwen. Further tuning must target these actual workloads rather than treating the gap as one universal dequantization or instruction-efficiency problem.

## 48. Reviewing the eight incoming commits on B390

The incoming range ends at `9f93739f1`. Local Intel work was saved as `9d648c196` before merging, and both histories are preserved.

| Commit | Change | B390 applicability |
| --- | --- | --- |
| `21b7bd2ce` | Reverse causal LinAlg attention groups | The active Intel scalar attention shader already uses reverse order; the incoming shader requires the unsupported 16x16x16 F32-accumulator shape. |
| `9b1641913` | NVIDIA small-model tile thresholds; Xe3 wave and stride defenses | Retain NVIDIA-only thresholds. Preserve the packed-K stride check and exactly one wave-size annotation on the Intel shader. |
| `048d57448` | NVIDIA Q4_K mid-projection exception | Leave vendor-gated. Intel flags 265/266 remain protected from later MMQ selection. |
| `a510e24ce` | NVIDIA D128 QK wave-limit documentation | A driver-specific rejected experiment, not a B390 capability or performance result. |
| `6fca6943c` | Merge Xe3 work while retaining packing checks | The type-sized `nb[0]` check also covers the local Q4_K wave route. |
| `5d85887c0` | Selected-wave capability queries and explicit conversion failure | Both defenses apply here. Failed conversion preparation now returns an error instead of continuing without valid F16 activations. |
| `2bcd60aec` | Restrict broken AMD D128 direct-K matrix load | Keep the direct-K specialization NVIDIA-only; no change to B390 scalar attention. |
| `9f93739f1` | Restore NVIDIA forward causal order | Preserve the vendor split rather than assuming one scheduling order wins everywhere. |

The capability-query documentation was corrected to match production: query the selected shader wave size, then the driver's zero-wave fallback, rather than accepting an arbitrary supported width. The merged B390 still reports wave16 and only the 8x16x16 F16-accumulator shape.

The optional capability dump previously queried only wave sizes 0, 32, and 64. Add wave16 so its omission cannot be mistaken for a driver restriction. The updated dump reports native wave16 F16xF16 -> F16 with shape 8x16x16, and S8xS8/U8xU8 -> S32 with shape 8x32x16. The integer entries are capability reports, not new implemented or benchmarked matrix paths.

The merged build passes 10824/10824 cases selected for MUL_MAT, MUL_MAT_ID, their vector/expert fusions, FLASH_ATTN_EXT, CONV_2D, CONV_3D, and OUT_PROD. A timestamped pre-merge/merged replay also passes 10824/10824 on each DLL. This is B390 runtime coverage; it does not substitute for running the AMD/NVIDIA-specific shaders on those devices.

There is a pre-existing timeout caveat despite those numerical passes. Windows records LiveKernelEvent 141 during the F16 attention case with D=256, 24 query heads, 4 KV heads, KV=16384, query batch=512, and a KV view. The isolated case reproduces on both DLLs and still passes 1/1. Instrumentation shows the same generic flag-0 route taking 3362.66 ms before the merge and 3393.39 ms after it; this is not the LinAlg attention shader changed by the incoming commits. Do not describe the suite as free of runtime issues or suppress the watchdog to hide this result.

The isolated reproduction uses the existing harness:

```powershell
.\build_linalg\bin\Release\test-backend-ops.exe test -o FLASH_ATTN_EXT -b DX120 -p "hsk=256,hsv=256,nh=4,nr23=\[6,1\],kv=16384,nb=512,mask=1"
```

### Before/after throughput

Compare saved pre-merge and merged backend DLLs with batch2048, ubatch512, pp6144, FA enabled, and the same existing Balanced power plan. Each table entry lists two process means in tok/s. SmolLM2 F16 and Granite use the initial balanced before/after/wide/wide/after/before sequence with two repetitions per process. The other rows use a follow-up with three repetitions per process and before/after/after/before order, except SmolLM2 Q8, which also includes the two middle wide arms. Each sweep starts after seven minutes idle following the preceding GPU/build work, with 15-second process gaps in the initial sweep and 20-second gaps in the follow-up.

| Model | Before merge, tok/s | After merge, tok/s |
| --- | --- | --- |
| SmolLM2-135M F16 | 5437.10, 5367.69 | 5371.65, 5355.81 |
| SmolLM2-135M Q8_0 | 5225.70, 5279.55 | 5212.06, 5217.45 |
| Granite MoE Q8_0 | 2166.72, 2176.55 | 2172.72, 2173.13 |
| Qwen3-4B Q4_K_M | 358.54, 347.98 | 347.60, 347.82 |
| Falcon-H1-7B Q4_K_M | 298.08, 311.49 | 309.95, 308.02 |

The initial SmolLM2 Q8 sequence was repeated because an RDP reconnection overlapped its last baseline arm. Initial Qwen/Falcon outliers also required follow-up rather than attributing variation to the merge. No RDP reconnect occurred during the follow-up. The first follow-up Qwen baseline remains noisy at 358.54 +/- 21.89; its final baseline and both merged arms agree near 348. The two SmolLM2 mean shifts are about -0.7%, within the observed variation, and the changing Falcon baseline does not establish a merge speedup. These measurements do not show a material, repeatable merge-related regression; they are not evidence of exact performance equality.

A SmolLM2 Q8 tg128 before/after/after/before control reports 314.76, 332.91, 323.76, and 329.44 tok/s, respectively, with three repetitions per process. It shows no decode regression in this sample, but its variability does not establish a decode optimization. No further GPU timeout report was observed during the follow-up benchmarks.

### Larger microbatches on B390

The incoming RDNA tuning notes suggest testing larger microbatches. This is also useful on B390 for some models. With the merged backend and batch2048, changing only ubatch512 to ubatch1024 gives these means across the paired processes:

| Model | ubatch512, tok/s | ubatch1024, tok/s | Change |
| --- | --- | --- | --- |
| SmolLM2-135M F16 | 5363.73 | 5838.10 | +8.8% |
| SmolLM2-135M Q8_0 | 5214.76 | 5611.00 | +7.6% |
| Granite MoE Q8_0 | 2172.93 | 2382.69 | +9.7% |

Use `-b 2048 -ub 1024` to reproduce that caller-side setting. It is not an incoming shader speedup or a new backend default, and it changes graph chunking and memory requirements. The initial Falcon comparison instead moved from 306.37 to 300.36 tok/s with the larger microbatch. Qwen's initial noisy sequence does not establish a robust microbatch benefit. Keep the default unchanged rather than applying this setting indiscriminately.

## 49. RDNA4 Q8_0 GEMM input specialization (2026-09-09)

RX 9070 XT, driver 32.0.23041.2023, Agility 1.721.3-preview, DXC 1.10.2605.37. Baseline: `9f93739f1`. This section concerns RDNA4 discrete GPUs, not the Xe3 measurements above.

### Retained changes

The live Vulkan path on this GPU is CoopMat1 with tile-local dequantization into F16 shared memory, not the integer MMQ fallback. Comparing its input specialization and packed loads exposed useful work beyond the earlier tile-size sweeps.

Two additional Q8_0 shader variants retain the existing 128x64 and 128x128 tile geometry, double-buffered BK16 staging, F32 accumulation, and LDS accumulator drain. They specialize full tiles with contiguous F32 activations and use native `uint16_t`/`uint16_t2` loads for Q8_0 scales and quants. Q8_0 blocks are 34 bytes, so two-byte alignment is sufficient; reconstructing every four quant bytes from shifted dwords was unnecessary for this path.

Flags 268 and 269 select the 128x128 and 128x64 variants respectively. They were renumbered during integration because 266 and 267 already select the Intel Q4_K and expert wave paths. Selection requires discrete RDNA4, a wave64 blob, full M/N tiles, contiguous F32 activation/output tensors, a 16-byte-aligned activation offset, and two-byte-aligned weight rows/batches. Other layouts and devices retain the generic shaders. `DX12_LINALG_Q8_ALIGNED=0` disables these variants.

The MMQ crossover is still shape-specific. Qwen's K=9728/N=2560 down-projection stays on LinAlg for M=768..2048 in multiples of 256; the optimized shader extends that to M=512. Phi's K=8192/N=3072 down-projection benefits at M=256/512 only with the optimized shader. Larger Phi microbatches remain on MMQ: the new LinAlg kernel still loses badly there. Explicit `DX12_MMQ_MIN_M`, `DX12_MMQ_MIN_N`, and `DX12_MMQ_MIN_K` settings retain control of this crossover.

Cached flags 268/269 must revalidate the layout predicate before replay. The general decision identity omits strides and offsets. A same-graph activation K-stride transition from 4 to 8 bytes reproduced incorrect output with the cached specialized pipeline; invalidating that decision and selecting the generic shader fixes both tile variants. This does not disable graph replay.

### Model measurements

Balanced baseline/new/new/baseline runs, all GPU work serialized. Qwen/Phi use five repetitions per arm; the small models use ten. All models are Q8_0, full GPU offload, FA enabled. Values below are means of both arms, in tokens/s. SmolVLM2 is the 256M model's text backbone, not vision encoding.

With `-ub 1024 -b 2048`:

| Model | pp512 gain | pp1024 gain | pp6144 before | pp6144 after | pp6144 gain |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | +17.0% | +23.2% | 4324.24 | 5055.00 | +16.9% |
| Phi-3-mini-4k | +20.2% | +14.7% | 4885.54 | 5448.70 | +11.5% |
| SmolLM2-135M | +1.4% | +10.0% | 50686.89 | 53318.50 | +5.2% |
| SmolVLM2-256M | +2.4% | +7.7% | 50702.18 | 53476.02 | +5.5% |

Small-model pp128/256 differences were within run-to-run noise. Their modest pp512 gains are less conclusive than the pp1024/6144 results.

The improvement also applies to the default microbatch size, without increasing `-ub`:

| Model, pp6144, ub512 | Before | After | Gain |
|---|---:|---:|---:|
| Qwen3-4B | 4161.78 | 4675.22 | +12.3% |
| Phi-3-mini-4k | 4696.74 | 5403.04 | +15.0% |

The final Qwen pp6144 profile's last chunk reduced gate/up GEMM time from 62.9 to 48.7 ms, down-projection from 43.2 to 28.9 ms, and query projection from 14.9 to 11.3 ms. These are aggregate GPU times per chunk, not single-dispatch or end-to-end timings. Attention remained around 128 ms and became roughly half the chunk's GPU time.

A fresh same-session Vulkan comparison puts the remaining pp6144 gap around 6% for Qwen and 8% for Phi. Some isolated GEMMs now beat Vulkan: Qwen gate/up measured about 768 us against Vulkan's 864 us with default precision and 878 us with F32 accumulation. This is not a claim that every operator or full workload is faster.

### Rejected experiments and coverage

Reversing the matrix operand roles while retaining LDS staging was correct but essentially neutral. Combining reversed roles with Vulkan-like two-wave ownership, BK32, a single staging buffer, and a 40-half row stride was correct but 9-39% slower on the sampled large Qwen shapes. Reusing the Q8 scale across two BK16 steps regressed KV projections by about 10-12%; native packed loads without that scale cache were better.

The existing operator suite plus six added full-tile, tail, broadcast, strided-view, and fused-bias cases passed 1222/1222. An isolated same-graph stride-transition probe reproduced the replay bug before the guard and passed both specialized tile variants afterward. Non-LinAlg DX12 also builds; Vulkan passes the four new cases it supports and skips the two K=2080 broadcast/view cases.

No accumulation-precision change, global dequantized-weight cache, activation prepass, direct descriptor matrix load, or additional accumulator-access workaround is required.

## 50. Pulling the RDNA4 specialization and checking model quality on B390

The new incoming commit is `aa17df904`, compared with local baseline `45b72561a`. Its aligned RDNA4 Q8 variants initially reused flags 266/267, already assigned to the Intel Q4_K/expert paths. Integration moves the new variants to 268/269, preserves the full-parameter dispatch conditions for both families, and retains the incoming RDNA4-only eligibility and replay checks. B390 does not run these wave64/F32-accumulator kernels.

### Model-level checks exposed a pre-existing scratch race

The initial merged build passed 3380/3380 matrix and fusion cases, but Q4 model quality was broken on both the pre-pull and merged DLLs. SmolLM2 Q4_K_M produced NaNs, and Granite/Qwen Q4_K_M perplexity reached millions. The user's existing SmolVLM2 Q4 image transcript also contained only dots. Disabling flash attention, fusion, or replay did not repair the failure. Forcing barriers did.

Q8_1 quantization and the Intel F16 activation prepass reuse one scratch buffer. The tensor hazard tracker cannot see earlier GEMMs reading that internal buffer. A barrier after writing protects subsequent reads but does not protect earlier reads against the next scratch overwrite. Add buffer-scoped compute barriers before all four writers: dense quantization, matvec quantization, F16 conversion, and fused RMS/MUL/quantization. The enhanced barrier's prior access mask includes both shader-resource reads and UAV access; existing callers retain their default mask. Cache reuse and optimized routes remain enabled.

Two independent-input, mixed Q8_0/Q5_0 GEMMs in one graph reproduce the missing dependency. The existing `test_mul_mat` helper now covers both type orders. The original DLL fails with NaN; the repaired DLL passes. The final affected selection passes 3382/3382, including the six incoming cases and these two whole-graph regressions. No new test executable or test file was added.

### Quality results

WikiText-2 raw test set, four chunks, context512, batch2048, ubatch512, full GPU offload, FA on. Corpus SHA256: `173C87A53759E0201F33E0CCF978E510C2042D7F2CB78229D9A50D79B9E7DD08`. These are small matched regression samples, not a full-corpus quality evaluation or comparisons between different models.

| Model | Pre-pull DX12 PPL | Repaired DX12 PPL | CPU Q4 reference |
| --- | --- | --- | --- |
| SmolLM2 F16 | 17.8620 | 17.8620 | - |
| SmolLM2 Q8_0 | 17.8813 | 17.8813 | - |
| SmolLM2 Q4_K_M | NaN | 18.3421 | 18.4188 |
| Granite MoE Q8_0 | 9.3745 | 9.3745 | - |
| Granite MoE Q4_K_M | 12835331.4240 | 9.6024 | 9.6938 |
| Qwen3-4B Q8_0 | 9.4669 | 9.4669 | - |
| Qwen3-4B Q4_K_M | 18314152.2766 | 9.6718 | 9.7009 |

Before the scratch repair, the incoming-only candidate had identical saved quantized logits for all four F16/Q8 samples. After the repair, SmolLM2 F16/Q8 saved logits remained identical, and all four F16/Q8 PPL values remained unchanged. A cached SmolVLM2 Q4_K_M image run with `stalib.jpg`, greedy sampling, seed1, and 128 generated tokens now produces a fluent description instead of only dots; this checks recovery from degeneration, not factual accuracy of every generated detail.

Earlier Q4 timing comparisons did not establish correct model output and must not be reused as quality-validated performance baselines. New Q4 throughput measurements use the repaired candidate only. The generic D256/KV16384 watchdog issue documented in section 48 is separate and was not addressed by this change.

### Performance with the repaired candidate

Serialized B390 runs use the existing Balanced plan, batch2048, ubatch512, full GPU offload, FA on, three repetitions per process, and 20-second process gaps. The sweep starts after seven minutes idle. F16/Q8 use before/after/after/before order; Q4 uses two repaired-only processes. No RDP reconnect or new GPU timeout event was observed during these measurements. Values are means across the two processes, in tok/s.

| Model | Before pp6144 | Repaired pp6144 | Before tg128 | Repaired tg128 |
| --- | --- | --- | --- | --- |
| SmolLM2 F16 | 5383.97 | 5379.31 | 253.97 | 249.18 |
| SmolLM2 Q8_0 | 5198.80 | 5202.82 | 312.85 | 310.31 |
| Granite MoE Q8_0 | 2171.99 | 2175.08 | 148.71 | 148.06 |
| Qwen3-4B Q8_0 | 345.75 | 355.11 | 22.30 | 23.11 |

The first three prefill changes are within 0.2%; their decode shifts are small compared with the observed variation. Qwen measured about +2.7% prefill, but its first baseline process was noisy at 346.68 +/- 14.97 tok/s. These controls show no material regression, not a demonstrated B390 speedup from the RDNA4 shader specialization.

| Repaired Q4_K_M model | pp6144 | tg128 |
| --- | --- | --- |
| SmolLM2 | 4701.03 | 352.83 |
| Granite MoE | 1605.42 | 146.56 |
| Qwen3-4B | 344.97 | 38.10 |

These Q4 numbers establish new correctness-checked baselines. Qwen's two prefill processes were 349.34 and 340.60 tok/s, so retain run-to-run variation when comparing future changes. GPU runtime validation here is limited to B390; the incoming AMD specialization still needs its own device coverage.

## 51. Main integration and Q5_0 MMQ on B390 (2026-09-10)

Merged the five main-only commits from `4d2d3ee5d` through `b7bfa804b` into `dx12-linalg-phase0`, using `78ec9245b` as the comparison baseline. These add Strix Point Q2_0 routing, quantized SET_ROWS eligibility guards, a Vulkan Strix Point matvec correction, Strix Point FP16 prefill attention, Q5_0 MMQ, and NVIDIA Pascal+ iGPU mask prescan. Intel already had the attention defaults. Keep the other vendor defaults scoped to their original devices.

Preserved the late LinAlg-aware MMQ selectors, flags 265-269, the wide route-cache flag field, the shared activation scratch barriers, and Vulkan tracing. Q5_0 MMQ flags 147/149 replace only the existing flag-58 integer-dot route; they do not overwrite a LinAlg selection.

### Paired-lane correctness

The incoming wave-share shader failed three Q5_0 cases on B390, including both independent-input mixed Q8_0/Q5_0 graphs. Errors were approximately 0.80 for the mixed graphs and 1.20 for the single Q5_0 GEMM, against a 0.0005 tolerance. The ordinary MMQ variant passed all 19 selected cases.

Changed the wave-share variant to a 256x1x1 threadgroup and reconstructed its logical 16x16 tile coordinates from `SV_GroupIndex`. This removes the assumption that the original two-dimensional group supplies the expected adjacent lane pairs. Both variants then passed all 19 cases. Three additional assertions in the existing Intel route fixture cover flags 147, 149 and the flag-58 opt-out, using output tails, odd Q5 block counts per row, and batch broadcasting.

Keep the ordinary variant's original `SV_GroupThreadID` entrypoint. An initial shared-prologue refactor also changed its performance, so the final fix is restricted to the wave-share entrypoint. Both final variants passed the 19 selected cases. Their wave16 bytecode was also matched against the respective benchmark DLLs.

### Correctness and quality

The integration candidate passed 18035/18035 numerical cases and 10/10 route assertions before the final entrypoint-only refinement above. The incoming SET_ROWS guards move 16 broadcasted Q8_0/Q5_1 cases to the reference fallback, accounting for the difference from the previous 18051 supported cases. The Vulkan build passed 73/73 selected eight-column MUL_MAT cases on B390. This is not AMD or NVIDIA runtime coverage.

All seven four-chunk WikiText perplexities from section 50 were unchanged. Forcing either Q5_0 MMQ variant also preserved SmolLM2 Q4_K_M PPL 18.3421 and SmolVLM2 Q4_K_M PPL 21.7803. SmolLM2 Q8_0 with quantized K and V caches was unchanged at 17.8677 for Q8_0 cache and 17.8971 for Q5_1 cache.

The full run again produced the previously documented LiveKernelEvent 141 during long-context attention. Numerical success does not resolve that watchdog issue. The baseline DLL reproduced the OS timeout in the standalone D=256, KV=16384, 512-query case at 13:33:58, followed by event 141, while still reporting 1/1 numerical success. Its generic attention implementation is unchanged by this merge.

### Q5_0 MMQ applicability

Balanced power plan, RDP disconnected, seven-minute idle before the first timing series, pp6144/tg128, `-b 2048 -ub 512`, full offload, FA on, three repetitions per process. The repaired shared variant used baseline/nonshared/shared/shared/nonshared/baseline order. The original nonshared entrypoint was then measured with baseline/nonshared/nonshared/baseline order in a separate window. Each row uses its own paired baseline; values are means of two process means in tok/s.

| Q4_K_M model | Variant | Baseline pp6144 | Variant pp6144 | Change |
| --- | --- | ---: | ---: | ---: |
| SmolLM2-135M | Original MMQ 147 | 4716.10 | 4660.88 | -1.17% |
| SmolVLM2-256M | Original MMQ 147 | 4696.59 | 4665.83 | -0.65% |
| SmolLM2-135M | Repaired wave-share 149 | 4701.62 | 4627.45 | -1.58% |
| SmolVLM2-256M | Repaired wave-share 149 | 4681.78 | 4619.57 | -1.33% |

Neither variant earns a B390 default. Keep `DX12_Q50_MMQ=1` as an opt-in, with `DX12_Q50_MMQ_WAVE_SHARE=0/1` selecting the variants. Decode scatter was larger than the small differences between their means; these GEMM routes are not selected at a single token.

### Default-route regression controls

The seven-model comparison used baseline/merged/merged/baseline order with the same settings, leaving Q5_0 MMQ at its device default. No B390 optimization default changed.

| Model | Baseline pp6144 | Merged pp6144 | Change | Baseline tg128 | Merged tg128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2 F16 | 5329.78 | 5400.52 | +1.33% | 252.13 | 250.19 |
| SmolLM2 Q8_0 | 5174.28 | 5234.42 | +1.16% | 295.35 | 308.36 |
| SmolLM2 Q4_K_M | 4655.82 | 4674.07 | +0.39% | 334.13 | 357.20 |
| Granite Q8_0 | 2173.61 | 2173.23 | -0.02% | 147.77 | 148.02 |
| Granite Q4_K_M | 1602.39 | 1606.89 | +0.28% | 146.49 | 146.78 |
| Qwen3-4B Q8_0 | 358.72 | 354.57 | -1.16% | 23.15 | 23.36 |
| Qwen3-4B Q4_K_M | 348.12 | 352.63 | +1.30% | 38.33 | 38.58 |

These controls show no material regression, not a demonstrated B390 speedup. Qwen Q8's two merged prefill process means were 360.22 and 348.91 tok/s, wider than its aggregate change. Smol decode had substantial within-process scatter, up to 71.45 tok/s standard deviation, so do not interpret its higher means as a tuning win. RDP reconnected after all throughput measurements had finished.

## 52. Intel Q5_0 wave GEMM (2026-09-10)

Added `QUANT_B=50` to `mul_mat_linalg_wave_f16_i.hlsl`, using the existing 8x16x16 F16 matrix shape and 64x64 quantized output tile. One thread decodes a complete 32-value Q5_0 block into F16 LDS. Native halfword loads read exactly the 22-byte block, including blocks starting two bytes into a dword. The existing F32 activation conversion, padded scratch allocation, pre-write barriers, root-SRV rebasing and bounded F16 accumulation/F32 drain are reused.

Flag 270 is default-on under the same feature and architecture gates as the Intel Q8_0/Q4_K wave variants. `DX12_LINALG_Q50_WAVE=0` opts out; the existing `DX12_LINALG_F16_WAVE=0` also disables it. The late Q5_0 integer-dot selector cannot overwrite a selected wave route. Other shapes and devices keep their previous paths. No MMQ default changed.

The shape constraints remain K divisible by 64, output channels divisible by 16, at least 64 tokens, contiguous F32 activations, and four-byte-aligned weight offset/strides. Partial 64-row token tiles and partial 64-channel tiles are handled by the existing padding and wave guards. K tails, smaller channel tails and noncontiguous activation views fall back instead of reading outside a tensor.

### Paired model performance

Baseline: `2cc43b234`. Balanced power plan, RDP disconnected, seven-minute cooldown after correctness/quality work, serial baseline/wave/wave/baseline order, three repetitions per process, pp6144/tg128, `-b 2048 -ub 512`, full offload and FA on. Values are means of two process means in tok/s.

| Model | Baseline pp6144 | Q5 wave enabled | Change | Baseline tg128 | Q5 wave enabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2 Q4_K_M | 4681.07 | 4959.22 | +5.94% | 341.49 | 336.55 |
| SmolVLM2 Q4_K_M | 4696.18 | 4952.20 | +5.45% | 343.88 | 352.57 |
| SmolLM2 F16 control | 5369.90 | 5387.04 | +0.32% | 256.06 | 253.72 |
| SmolLM2 Q8_0 control | 5202.98 | 5237.26 | +0.66% | 308.30 | 312.80 |

Both Q4 models show repeatable prefill gains, unlike the MMQ variants in section 51. The two wave prefill process means were 4956.95/4961.48 for SmolLM2 and 4951.84/4952.56 for SmolVLM2. The route is not selected for single-token decode; decode scatter was substantial, so these means do not establish a decode gain or loss.

### Quality and fallback coverage

The four-chunk WikiText sample changed from 18.3421 to 18.3063 PPL for SmolLM2 Q4_K_M and from 21.7803 to 21.8390 for SmolVLM2 Q4_K_M. These are small changes, not bitwise equivalence: the wave path uses F16 activations and bounded F16 matrix accumulation rather than Q8_1 activations and integer-dot accumulation. Disabling the new route restores the baseline values. F16 and Q8_0 control perplexities were unchanged.

The existing route fixture covers the actual flag-270 default, mixed independent Q8_1/F16 scratch consumers, priority over the late integer-dot selector, local/global opt-outs, K/channel fallbacks and a noncontiguous-view fallback. A Q5_0 case with K=16384 exercises repeated accumulator drains. The strided-view case uses generic flag 0, not tiled flag 58, because its activation tensor is also noncontiguous.

The final default-on build passed 18036/18036 numerical cases and 17/17 route assertions. The longer 16-chunk sample changed from 23.4723 to 23.4473 PPL for SmolLM2 and from 27.6604 to 27.6641 for SmolVLM2, within 0.11% of baseline in both cases. SmolVLM2 image generation remained fluent. These are regression samples, not a full quality evaluation. The pre-existing generic long-context attention watchdog remains outside this change.

## 53. Additional Intel wave coverage (2026-09-10)

The following routes target the queried 8x16x16 F16 matrix shape on Intel wave16 hardware. They are opt-in: native matrix coverage does not establish a speedup over the existing route. The previously shipped F16/Q8_0/Q4_K/Q5_0 defaults remain unchanged.

| Switch | Coverage | Flags |
| --- | --- | --- |
| `DX12_LINALG_Q6K_WAVE=1` | Dense Q6_K block loader | 271 |
| `DX12_LINALG_Q5K_WAVE=1` | Dense Q5_K block loader | 272 |
| `DX12_LINALG_QUANT_WAVE=1` | Dense Q4_0, Q4_1, Q5_1, IQ4_NL, MXFP4 block loaders | 273-277 |
| `DX12_LINALG_DENSE_STAGED=1` | F16 and 25 quant formats through the shared dequantizing tiled kernel | 279 |
| `DX12_LINALG_MMID_EXTRA=1` | Additional quant formats in the existing bucketed expert kernel | 267 |
| `DX12_LINALG_WAVE_BIAS=1` | Single-consumer, contiguous F32 channel bias after an eligible dense wave GEMM | Base flag plus 512 |
| `DX12_LINALG_AUX_F16=1` | Eligible F16-staged convolution and OUT_PROD wave kernels | 214-217, interpreted per operation |
| `DX12_FA_LINALG_WAVE=1` or `2` | Native attention with D=64/96/128 and float, Q8_0 or Q4_0 KV | 280-282, 284-286, 288-290 |

The optimized dense loaders retain K%64, output-channel%16, contiguous F32 activation and aligned weight-layout gates. Q6_K's 210-byte and MXFP4's 17-byte blocks use bounded loads rather than assuming every block starts on a dword boundary. The staged route handles F16/F32 activations, guarded dimension tails and batched/broadcast layouts without the activation-conversion dispatch. It does not displace an already selected optimized wave kernel and needs at least 16 token rows. Existing expert occupancy gates still apply to the expanded format set.

`DX12_LINALG_Q50_MIN_M` can lower the Q5_0 token threshold from its default 64, with a minimum of 2. This reuses the 64x64 shader; it is not a new small-tile kernel. Bias variants are compiled separately so the unbiased shipped shader bytecode is unchanged. Dispatch dimensions and scratch sizing must use the base flag after removing bit 512.

`MUL_MAT` with F32 or BF16 weights remains on its existing paths. The available preview API does not provide a BF16 wave matrix component, and converting arbitrary F32/BF16 weights to F16 is not an equivalent implementation. The explicit `AUX_F16` option does convert supported F32 auxiliary tensors to F16 staging. Unlike attention, auxiliary matrix products do not have runtime range safeguards; large finite inputs can overflow a partial sum. Do not enable that reduced-precision option indiscriminately.

### Attention precision and limits

Mode `1` honors `GGML_PREC_F32`; mode `2` explicitly permits reduced-precision matrix products even for that request. Model graphs normally request F32 attention, so model experiments require mode `2`. Neither mode is enabled by default. `DX12_FA_LINALG=0` disables both. BF16 KV and D=256 remain on the existing implementation.

The native implementation preserves masks, ALiBi, softcap, sinks, GQA, batches and split-KV output layout. Range flags are computed from original values in F32. Q/K groups within magnitude 32 stay unscaled. Larger eligible groups use Q/4 and/or K/16, with the corresponding score restoration in F32. Inputs beyond Q=128/K=512, raw half subnormals, and groups where necessary scaling would produce half subnormals use original-input F32 arithmetic. Large V retains scaled-PV/F32 alternatives. F16 partial sums are drained into F32 every 16 products. These safeguards are part of the implementation, not optional performance instrumentation.

Eight wave16 waves share each 32-query group. The first four compute matrix QK; pairs of PV waves split the output width without duplicating QK or softmax. All waves participate in the group barriers. This reduces each thread's output accumulator and mask arrays while preserving the host dispatch and split metadata layout.

### Route and numerical coverage

Run `test-backend-ops.exe test -b DX120 -o DX12_ROUTES` for the 109-case route/numerical fixture. It exercises selected shader flags, opt-outs, mixed independent scratch consumers, all 11 bias formats, additional expert formats, staged tails and attention modifiers/range handling. The initial integrated numerical suite passed 18107/18107 ordinary cases. The final fixture passed 109/109, including F32/F16 scaling-underflow cases, attention precision and environment-refresh transitions. A separate final attention slice passed 2708/2708 ordinary D64/D96/D128 cases. Route capture excludes unrelated copy/add dispatches, which otherwise make a flag-0 assertion ambiguous. The full run also generated the previously observed long-context attention watchdog at 17:11 Pacific; a passing numerical count does not resolve that separate issue.

Four-chunk WikiText controls cover all 18 text configurations in `bench_linalg.bat`, using the actual cached assets. The F16 labels for Qwen3-0.6B and Falcon-H1 resolve to BF16 files. Enabling the extra dense/expert routes changed PPL by at most 0.33% in absolute relative terms; native attention mode `2` changed it by at most 0.32%. These short samples are regression checks, not proof of general quality equivalence.

Two stale selector checks were corrected: Q5_K/Q6_K generic wave flags are 219/223 rather than 119/123, and the IQ1_S tiled predecessor is 158 rather than 130. Route capture now occurs after late attention selection. The attention opt-out uses the existing refresh-aware environment cache so fixtures can change it between graphs.

### Initial performance and profiling

Baseline `0c060bbb0`; High Performance power plan, RDP disconnected, seven-minute idle, full offload, FA on, `-b 2048 -ub 512`. The initial pilot used three repetitions per process at pp512/1024/6144 and baseline/dense/attention/attention/dense/baseline order. The dense arm enables the extra dense/expert formats, not staged or auxiliary kernels. The attention column is the first implementation before contention/layout/range tuning.

| Model | Baseline pp6144 | Extra dense pp6144 | Dense change | Initial native FA pp6144 |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2 F16 | 5446.42 | 5467.50 | +0.39% | 2932.64 |
| SmolLM2 Q4_K_M | 5017.91 | 4873.20 | -2.88% | 2809.88 |
| Granite Q4_K_M | 1600.00 | 1573.58 | -1.65% | 527.03 |
| Qwen3-4B Q4_K_M | 339.99 | 320.48 | -5.74% | 195.58 |
| Phi-3 Q4_K_M | 433.87 | 432.18 | -0.39% | 339.88 |

These results do not justify enabling the new dense defaults. Native FA also required further work, not promotion based on its matrix instructions alone.

VTune global-memory-access captures exposed about 1.28 billion SLM atomics and 21.6 billion bank-conflict events in the initial SmolLM2 native-FA capture. Aggregating mask visibility per wave reduced atomics to about 75 million. Padding F32 scores reduced conflict events to about 11.1 billion. Padding the F16 matrix pitches by eight elements reduced them further to about 4.1 billion, but also changed occupancy. These are whole-capture counters, not per-kernel attribution; post-build GPU clocks differed substantially, so the profiler's throughput values are not controlled speed comparisons.

The F16 pitches remain multiples of eight halves. Score padding doubles as storage for rescale factors, keeping the D64 declared LDS footprint at 16268 bytes. Matrix operands must not use the single-float padding applied to scores.

A scheduler callback captured actual Q/K ranges for 512 deterministic random input tokens. SmolLM2 and Phi had no Q/K groups above the original magnitude-32 bound. Granite had 1644/3072 unsafe K tiles and 637/6144 unsafe Q groups; Qwen had 736/4608 unsafe K tiles. Maximum observed K was 296.75 for Granite and 317.75 for Qwen. This motivated the bounded power-of-two scaling above instead of removing the overflow fallback. These short-context diagnostics are not a universal bound on model activations.

Do not detect scaling underflow by comparing two native half casts: both can flush the value to zero. With Q=128, tiny nonzero K and an attention scale of 512, that approach produced 0.5 instead of approximately 0.73106. Original-value F32 flags now reject unsafe subnormal staging before it can affect the matrix result. The route fixture includes zero-erasure and nonzero-rounding counterexamples.

After padding and Q/K scaling, all 18 four-chunk perplexity samples remained within 0.37% of their paired baseline. Sixteen-chunk samples were 23.4409 versus 23.4473 for SmolLM2 Q4_K_M, 12.3125 versus 12.3073 for Granite Q4_K_M, and 10.2962 versus 10.2858 for Falcon Q8_0. The experimental precision override remains explicit.

The final eight-wave, subnormal-safe implementation was checked again across all 18 four-chunk samples: the largest increase was 0.13%, and the largest absolute change was a 0.31% decrease for Granite Q4_K_M. Final 16-chunk values versus baseline were SmolLM2 Q4_K_M 23.4450/23.4473, Granite Q4_K_M 12.3035/12.3073, Qwen3-4B Q4_K_M 12.5097/12.5096, Phi-3 Q4_K_M 7.1360/7.1351, and Falcon Q8_0 10.3014/10.2858.

### Additional-format operator measurements

Interleaved baseline/extra/extra/baseline, High Performance plan, RDP disconnected, seven-minute idle beforehand. Dense cases have 4096 output channels, 512 tokens and K=14336. Values are means of the two process means. These are isolated operator results, not model-level speedups.

| Weights / activations | Baseline us | Extra routes us | Speedup |
| --- | ---: | ---: | ---: |
| Q4_0 / F32 | 42524.73 | 8712.83 | 4.88x |
| Q4_1 / F32 | 42062.68 | 7684.33 | 5.47x |
| Q5_1 / F32 | 998472.00 | 7464.86 | 133.76x |
| IQ4_NL / F32 | 1065981.25 | 15746.60 | 67.70x |
| MXFP4 / F32 | 30260.66 | 15878.14 | 1.91x |
| Q5_K / F32 | 9867.32 | 7842.05 | 1.26x |
| Q6_K / F32 | 11204.40 | 18658.82 | 0.60x |
| IQ1_S / F32, staged | 974881.00 | 51483.88 | 18.94x |
| IQ2_XS / F32, staged | 55201.61 | 54649.01 | 1.01x |
| F16 / F16, staged | 21994.03 | 22342.78 | 0.98x |

IQ1_S includes the corrected tiled-route predecessor as well as the optional staged path; the table does not isolate the native contribution from that routing fix. The legacy scalar Q5_1/IQ4_NL baselines explain their very large ratios. Those ratios must not be presented as whole-model gains. F16/F16 and IQ2_XS dense results do not establish a gain.

An IQ2_XS expert case with 32 experts, eight selected experts, 512 tokens, 768 output channels and K=2048 improved from 15901.99 to 10067.87 us, or 1.58x. Other newly supported expert formats still need representative performance measurements before a blanket default change.

The format switches remain opt-in: large-shape operator wins are not sufficient evidence to change every model/layout default. In particular, Q6_K loses both in the isolated case and in the primary Q4_K_M model pilot. Small-Q5 and bias image experiments also had substantial cold-prompt variation, so their defaults remain unchanged.

### Final primary-workload controls

All 18 text configurations in `bench_linalg.bat` were measured at pp6144 and tg512, five repetitions per process, with baseline/final pairs, full offload, FA on, `-b 2048 -ub 512`, a two-second repetition delay and High Performance power plan. New experimental switches were left at their default off values.

| Model | Baseline pp6144 | Final pp6144 | Baseline tg512 | Final tg512 |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2 F16 | 5357.66 | 5252.55 | 250.69 | 245.97 |
| SmolLM2 Q8_0 | 5186.28 | 5221.89 | 309.25 | 307.48 |
| SmolLM2 Q4_K_M | 4955.30 | 4936.50 | 353.61 | 357.26 |
| Qwen3-4B F16 | 371.64 | 367.36 | 13.34 | 13.40 |
| Qwen3-4B Q8_0 | 354.08 | 352.30 | 22.765 | 22.810 |
| Qwen3-4B Q4_K_M | 346.01 | 348.00 | 36.760 | 36.675 |
| Phi-3 F16 | 518.33 | 529.31 | 14.24 | 14.31 |
| Phi-3 Q8_0 | 472.86 | 470.03 | 24.06 | 24.31 |
| Phi-3 Q4_K_M | 444.53 | 446.29 | 38.32 | 38.48 |
| Granite F16 | 2341.51 | 2309.60 | 95.94 | 95.91 |
| Granite Q8_0 | 2107.62 | 2112.22 | 141.94 | 143.07 |
| Granite Q4_K_M | 1531.90 | 1530.51 | 133.02 | 132.45 |
| Qwen3-0.6B BF16 | 859.09 | 870.30 | 78.32 | 78.54 |
| Qwen3-0.6B Q8_0 | 1311.81 | 1314.69 | 114.99 | 115.35 |
| Qwen3-0.6B Q4_K_M | 1297.37 | 1288.60 | 164.33 | 164.34 |
| Falcon-H1 BF16 | 96.08 | 113.02 | 7.14 | 7.09 |
| Falcon-H1 Q8_0 | 317.31 | 317.28 | 11.55 | 11.59 |
| Falcon-H1 Q4_K_M | 308.83 | 310.80 | 18.08 | 18.13 |

The initial Qwen Q8/Q4 decode pairs appeared 10.70%/4.26% slower. Those drops did not reproduce in independent tg512-only baseline/final/final/baseline runs with five repetitions: the means above differ by less than 0.24%. Do not diagnose a code regression from the earlier unbracketed pair alone.

A brief RDP reconnection on September 10 at 21:37:13-21:37:48 Pacific invalidated the original Falcon BF16 final measurement. Its entire pair was repeated without a reconnection; the table uses that replacement. The helper now checks session-reconnection events during each measurement, not just whether RDP is active at its endpoints. Falcon BF16 prefill varied substantially between windows, so its apparent increase is not claimed as an optimization win.

After these replacements, decode changes range from -1.88% to +1.04%. Excluding the unstable Falcon BF16 prefill result, prefill changes range from -1.96% to +2.12%. These are default-route regression controls, not evidence of a new speedup for the primary model set. The original F16/Q8_0/Q4_K/Q5_0 wave16 shader blobs were also found byte-identical inside the final DLL.

The final experimental native-FA mode remained slower in its separate pp6144 baseline/native/native/baseline comparison:

| Q4_K_M model | Existing FA | Native FA, explicit mode 2 |
| --- | ---: | ---: |
| SmolLM2 | 4953.07 | 1946.55 |
| Granite | 1605.45 | 793.61 |
| Qwen3-4B | 341.55 | 112.83 |

Keep native FA off on B390 by default. Its broader functional coverage, range handling and matrix use do not establish competitive throughput. The existing single-token decode, F32/BF16 weight, and unsupported attention-shape paths are intentionally retained.

## 54. Q4_K traffic accounting and controlled staging experiments (2026-09-11, Pacific)

The bandwidth figures in section 46 do not establish a DX12 hardware bandwidth ceiling. Vulkan completes more GEMMs per second and also generates more memory/cache read traffic per GEMM. These two factors multiply. The investigation below recovered the old work denominators, collected new equal-work captures, and tested seven staging changes. None earned a retained optimization; the pre-investigation shader and CMake files were restored exactly, including the uncommitted coverage work in section 53.

### Recovering the original denominators

The three September 9 captures all target `MUL_MAT(type_a=q4_K,type_b=f32,m=4096,n=512,k=14336)`: 60,129,542,144 mathematical FLOPs per GEMM. Their approximately one-second benchmark loops complete 102 MMQ, 134 wave-labelled, and 266 Vulkan operations. Equal captured GPU time is not equal work.

The DX12 traces contain generic short/long Dispatch pairs. Integrating sampled counters over 60 complete MMQ pairs and 79 complete wave-labelled pairs inside the quoted steady windows gives:

| Original capture | GPU memory read, MB/GEMM | L3 read, MB/GEMM |
| --- | ---: | ---: |
| MMQ, complete pairs | 92.0 | 1238 |
| Wave-labelled, complete pairs | 142 | 1423 |
| Vulkan, whole-capture/266 proxy | 223 | 1728 |

The Vulkan archive lacks dispatch timestamps/counts. Its row includes possible warmup/setup traffic and is not an exact steady-window normalization. The DX12 archive also lacks shader names and inherited environment values, so directory labels alone are not independent routing evidence. The new captures below remove these limitations for the current native route.

The captured Intel definitions map GPU-memory read/write to interface byte counters and L3 read/write to device-cache events multiplied by 64 bytes. These are not unique tensor bytes or necessarily physical LPDDR wire bytes. Do not add L3 and memory bytes as independent payload. Whole-adapter rates divide accumulated traffic by the entire capture duration, including idle/startup; percentages use their own event/clock denominators.

### New fixed-work captures

A session-only probe uses the existing ggml API, deterministic identical inputs, two GEMMs per graph, 32 warmup GEMMs, and exactly 1024 measured GEMMs. It brackets the measured loop with VTune pause/resume calls and checks completion and output hashes. The native dispatch trace confirms flag 266; a separate Vulkan trace confirms `matmul_q4_k_f32_f16acc_aligned_l`, groups `(32,4,1)`.

ITT task records are absent in these GPU-hotspots captures, but `dd_paused_range` supplies collection-control boundaries. Merge the paused ranges and integrate raw GPU counter deltas over the common active gap. Hardware counters still include warmup activity outside that gap: dividing whole-capture bytes by 1024 would be wrong. Each DX12 measured interval contains exactly 2048 Dispatch records, covering the conversion/GEMM pairs. Vulkan uses the same completed fixed loop and pause boundaries, without inventing missing dispatch records.

| Fixed-work route | GPU memory read, MB/GEMM | L3 read, MB/GEMM | SLM read, GB/GEMM | Issued instructions, million/GEMM | SLM bank conflicts, million/GEMM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing native Q4_K | 138.1 | 1414.4 | 5.637 | 699.4 | 14.680 |
| Four values/thread | 137.9 | 1308.3 | 5.637 | 1066.8 | 0 |
| K-sliced weight LDS | 137.8 | 1386.9 | 5.637 | 696.7 | 22.020 |
| Vulkan, pipeline tracing disabled | 224.1 | 1722.8 | 3.758 | 212.1 | 0 |

MB/GB are decimal. Instruction counts exclude the scalar pipe; they are not scalar-operation counts, and wave widths differ between APIs. All metrics are adapter-level samples during the controlled workload, including associated prepasses, not individual named shader instrumentation.

The active windows are 7.7175 seconds for native DX12 and 3.7436 seconds for Vulkan. Thus Vulkan's approximately 3.35x GPU-memory read rate is 2.06x operation rate times 1.62x bytes per operation. Its approximately 2.51x L3 read rate is the same 2.06x operation rate times 1.22x bytes per operation. This directly explains why reproducing Vulkan's GB/s is not itself the optimization target.

The active-cycle clock ratios are approximately 2.48 GHz for native DX12 and 2.45 GHz for the accepted Vulkan capture. They do not explain Vulkan's advantage. This is not a DRAM-clock measurement. In the old steady windows, native DX12 has roughly 47.5% L3 Busy, 1.4% L3 queue-full, and 41% memory-active, versus approximately 95%, 2%, and 93% for Vulkan. Those observations support lower request-generation/issue pressure in DX12 rather than proving external-memory saturation. Busy means pending requests, not percentage of peak byte bandwidth.

### Causal experiments

The current native Q4_K kernel stages 32 consecutive decoded values per thread, with 128 threads, a 64x64 output tile, and K=64 staging windows. The selected Vulkan large-tile path uses 512 threads, a 128x128 output tile, K=32 staging, and four decoded values per loader invocation. It converts F32 activations during staging; DX12 uses a separate F16 activation prepass. Larger tiles, different wave widths, matrix loads and accumulator handling remain important differences. Copying only the Vulkan loader granularity is not equivalent to copying its whole kernel.

All variants preserved matrix arithmetic, bounded F16 partial accumulation, global scratch synchronization and output coverage. Both fused-bias and ordinary Q4_K variants were built. Each candidate passed the existing 109 route assertions; ordinary Q4_K numerical cases also passed for the four-value, layout, rotation and uniform-header candidates. Sixteen/eight-value variants were covered by the route fixture.

Uninstrumented operator measurements used the exact original test-harness case, High Performance, disconnected RDP with reconnect-event rejection, seven-minute cooldown after builds/tests, and reversed-order DLL-swapped measurements. Each table value is the mean of two process means. Comparisons use their own bracketed baseline, not an absolute baseline imported from another cohort.

| Variant | Baseline us/GEMM | Candidate us/GEMM | Throughput change |
| --- | ---: | ---: | ---: |
| 16 decoded values/thread | 7249.42 | 7438.79 | -2.55% |
| 8 decoded values/thread | 7249.42 | 8341.82 | -13.10% |
| 4 decoded values/thread | 7249.42 | 10445.98 | -30.60% |
| Weight LDS row padded by eight halves | 7169.72 | 7708.38 | -6.99% |
| Weight LDS rearranged into K=16 planes | 7169.72 | 7192.84 | -0.32% |
| XOR-staggered 32-value staging order | 7149.82 | 7763.28 | -7.90% |
| Four-value loader with wave-uniform header/pair decoding | 7149.82 | 10922.41 | -34.54% |

Vulkan in the first uninstrumented cohort averages 3733.10 us/GEMM versus native DX12's 7249.42 us/GEMM, approximately 1.94x faster. This gap was not closed.

The four-value experiment is a useful counterexample: it eliminates the measured SLM bank conflicts and reduces L3 read bytes by about 7.5%, yet increases issued instructions by about 52.5% and loses 30.6% throughput. The source change repeats header/scale decoding and loop/address work more often; the counters do not provide an opcode-level attribution of every added instruction. Its output hash is bit-identical to the baseline. Removing a large conflict count or improving coalescing alone is therefore insufficient. Conversely, K-slicing raises the conflict count by 50% without a material timing change; conflict counts are not a measurement of critical-path stall cycles.

The remaining explanation must account for work per GEMM, particularly the 1.5x shared-memory read volume and the different generated instruction sequences, rather than just trying to increase a bandwidth counter. These experiments do not isolate the contribution of each matrix-load, accumulator, compiler-lowering or tile-reuse difference, nor rule out other layouts. No native-ISA causal proof or universal hardware limit was established.

### Measurement and restoration safeguards

Pipeline logging is not timing-neutral. With `GGML_VK_TRACE_PIPELINES=1`, the fixed Vulkan probe takes 5.823 ms/GEMM and reads approximately 341 MB/GEMM from GPU memory. Removing only that logging gives 3.654 ms/GEMM and 224 MB/GEMM in the repeated capture; an unprofiled no-trace run gives 3.628 ms/GEMM. Inputs and output hashes match. The logged capture is retained as a rejected measurement, not mixed into the accepted table. Routing was established separately.

Autotune cache identity includes the DLL translation unit's build timestamp. The A/B helpers therefore hold the known baseline decisions constant with the existing force controls: Q4_K/Q5_K 32-thread choices, and F16/BF16/F32 K thresholds 1495/1690/807. Otherwise rebuilding or swapping DLLs can change unrelated decode decisions. These are process-local measurement controls, not new backend defaults.

A build helper incorrectly cleared `build_linalg` during setup. The source and pre-investigation DLL snapshot were preserved, but the original generated build/cache contents were not backed up. All standard targets were rebuilt with the pinned preview DXC, Agility 721, activation staging, and BoringSSL HTTPS support. No measurements from the resulting temporary non-LinAlg builds were accepted. A targeted build also requires the `ggml-dx12-agility-stage` target on a fresh tree; building only `ggml-dx12` does not run that unrelated ALL target.

The restored shader and CMake files match their pre-investigation snapshots byte-for-byte. The rebuilt DLL retains the original F16/Q8_0/Q4_K/Q5_0 wave16 blobs byte-for-byte, passes 109 route assertions and 395 selected matrix cases, and reproduces the four-chunk SmolLM2/Qwen3-4B/Granite/Falcon Q4 perplexities exactly at printed precision. The final installed DLL is the saved pre-investigation F85715DC... binary, not an experimental or rebuilt candidate. Its original autotune decisions are restored as well; regenerated calibration values are archived rather than left as an unrelated behavior change. Earlier uncommitted expansion changes remain intact. None of the seven experimental staging variants or their build switches is retained.

Evidence is saved under the session `files` directory: `traffic-audit-20260911-*` contains the archived capture audit; `traffic-20260911-fixed-*` contains the new captures and integrated metrics; `traffic-20260911-ops.csv`, `traffic-20260911-layout-ops.csv` and `traffic-20260911-targeted-ops.csv` contain the uninstrumented cohorts. The probe, integration script, experiment source snapshots and DLLs are retained for reproduction. Sample-boundary bounds cover interpolation only, not total measurement uncertainty; one zero-width clock sample is accounted at its timestamp and does not affect byte counters.

## 55. Complete Vulkan-style Q4_K GEMM (2026-09-14, Pacific)

This is a complete opt-in kernel, not another isolated loader replacement. On the B390 it reduces the large Q4_K operator from roughly 7.2 ms to 4.6 ms. Vulkan still takes roughly 3.7 ms: the target of throughput within 5% of Vulkan has not been reached. No existing routing default changes.

### Implementation and precision policy

`mul_mat_q4_k_vkport.hlsl` follows Vulkan's `mul_mm.comp` and Q4_K loader in `mul_mm_funcs.glsl`: weights are operand A, activations are operand B, the workgroup output tile is 128x128, K staging is 32, and each shared-memory row has 40 half elements. The 16-wave workgroup uses four decoded weights per loader invocation and eight F32 activations per invocation. Activation conversion is inside the GEMM, without a scratch conversion dispatch. The retained variant reads each weight header with `Load4` and uses fixed-trip-count staging loops.

The runtime's reported native 8x16x16 F16 tile is not a maximum matrix size. Microsoft's `D3D12LinearAlgebraRuntimeFeatureSupport.md` draft describes composition from integer multiples of native dimensions. Both 16x16 and 32x32 F16 output matrices were accepted and numerically exercised on this driver. The retained wave output matrix is 32x32, using 32x16 A and 16x32 B matrices. This does not imply F16 x F16 -> F32 MMA support; that type combination is still not advertised.

`DX12_LINALG_Q4K_VKPORT` accepts one digit:

| Value | Dispatch flag | Wave size | Accumulation |
| --- | ---: | ---: | --- |
| unset or `0` | Existing selection | Existing selection | Unchanged |
| `1` | 292 | 16 | F16 partials drained to F32 every 64 K elements |
| `2` | 293 | 16 | Full-K F16 |
| `3` | 294 | 32 | F16 partials drained to F32 every 64 K elements |
| `4` | 295 | 32 | Full-K F16 |

Mode 4 is the fastest measured port. Modes 2 and 4 deliberately have reduced accumulation range and precision; like the Vulkan F16-accumulator shader, they clamp the final F16 values to +/-65504 before storing F32 output. An explicit `GGML_PREC_F32` request retains the original route instead of these full-K modes. Modes 1 and 3 retain the original kernel's 64-element partial-reduction policy, not a promise of pure F32 arithmetic.

Selection only replaces an otherwise eligible native Q4_K flag 266. The native/global opt-outs still apply. Output dimensions must fill 128x128 tiles; activations and output must be contiguous F32 with compatible batch dimensions. Full-F16 matrix stores also require a 128-byte-aligned output tensor base. Wave32 requires both the advertised wave-size range and an independent matrix-shape query; wave16 modes require the Intel device's selected wave size to be 16. Each tensor must fit within `INT32_MAX` bytes, and dispatch dimensions must fit D3D12 limits. Both input SRVs and the output UAV are rebased onto their tensors, with matching root-address cache updates. Small, ragged, unsupported and explicitly opted-out cases keep the existing kernels. Bias ADD remains a separate operation.

### Operator comparisons and discarded experiments

The fixed test is `MUL_MAT(type_a=q4_K,type_b=f32,m=4096,n=512,k=14336)`. Comparisons retain the section 54 controls: pinned toolchain, original autotune decisions, High Performance, disconnected RDP and reconnect-event rejection, cooldown after CPU-heavy work, and reversed-order DLL swaps. Each entry below averages two process means from one cohort.

| Clean-build cohort | us/GEMM | Throughput relative to original DX12 |
| --- | ---: | ---: |
| Original native DX12 | 7141.21 | 1.00x |
| Wave32 bounded-accumulation port | 6115.42 | 1.17x |
| Wave32 full-F16 port | 4600.51 | 1.55x |
| Vulkan | 3705.77 | 1.93x |

The full-F16 port has approximately 24.1% longer runtime than Vulkan in this cohort. The earlier native-8-row version took about 5.24 ms in its own cohort. Composed 16-row fragments and eight-value activation loading reduced this to about 4.73 ms; 32x32 wave output matrices made a further small improvement. These experiments do not isolate each instruction-level cause. Cleanup replaced the one-element matrix arrays with one accumulator per wave; it was not a measured performance optimization. The preceding candidate's samples in the final cohort were 4352.32 and 4586.78 us, versus 4633.37 and 4567.65 us after cleanup.

The other trials did not earn retained complexity: native F32 matrix totals were dramatically slower in diagnostic runs; K=64 staging with no row padding regressed; composing K=32 did not improve over K=16; and cooperative descriptor-load/cast/shared-store activation staging took about 4.79 ms versus 4.62 ms for the ordinary loader in its matched cohort. A vectorized header load and explicit staging-loop unrolling were approximately neutral independently. The scalar accumulation totals and ordinary activation loader remain. Experimental build switches and losing shader branches are removed.

### Fixed-work profile

The same section 54 probe executes 1024 measured GEMMs with the same input hashes. Both pre-cleanup and clean-build traces confirm flag 295 and exactly 1024 dispatches, rather than the original 2048 conversion/GEMM dispatches. Both output hashes are `8263474046ee4e47`, matching the archived untraced Vulkan probe.

| Route | GPU memory read, MB/GEMM | L3 read, MB/GEMM | SLM read, GB/GEMM | Issued instructions, million/GEMM | SLM bank conflicts, million/GEMM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original DX12, section 54 capture | 138.1 | 1414.4 | 5.637 | 699.4 | 14.680 |
| Pre-cleanup wave32 full-F16 port | 143.0 | 1659.3 | 3.758 | 333.0 | 0.459 |
| Clean build, interval estimate | 139.5 | 1606.5 | 3.654 | 323.8 | 0.446 |
| Vulkan, section 54 untraced capture | 224.1 | 1722.8 | 3.758 | 212.1 | 0 |

The clean capture has wider boundary samples: its SLM read bounds are 3.568-3.758 GB/GEMM and issued-instruction bounds are 316.1-333.0 million/GEMM. Uniform interpolation within those boundary samples produces the table's estimates; do not interpret the lower point estimates as a demonstrated instruction or traffic improvement from cleanup. The upper bounds match the earlier profile closely. These bounds cover interpolation only, not total measurement uncertainty.

The port removes the original tile's excess shared-memory read volume and roughly halves its issued-instruction count, but still issues substantially more vector-pipe instructions than the Vulkan reference. It reads fewer GPU-memory bytes, not more. These are work-normalized comparisons with archived reference captures, not a simultaneous rate comparison or an opcode-level causal attribution. The pre-cleanup capture's active clock is about 2.48 GHz and its profiled wall time is 4.875 ms/GEMM; the clean capture is about 2.41 GHz and 4.839 ms/GEMM. Neither similar SLM bytes nor eliminating the prepass establishes Vulkan parity or an unavoidable API/driver limit.

### Model quality

Six pinned Q4_K_M model files from `bench_linalg.bat` were evaluated on the same first 16 WikiText-2 chunks, context 512, batch 2048, microbatch 512, GPU offload and flash attention enabled.

| Model | Original PPL | Mode 3 PPL | Mode 4 PPL |
| --- | ---: | ---: | ---: |
| SmolLM2-135M | 23.4473 | 23.4473 | 23.4473 |
| Qwen3-0.6B | 26.2046 | 26.1766 | 26.1959 |
| Qwen3-4B-Instruct-2507 | 12.5096 | 12.5048 | 12.5083 |
| Phi-3-mini-4k | 7.1351 | 7.1365 | 7.1335 |
| Granite-3.0-1B-A400M | 12.3073 | 12.3175 | 12.3162 |
| Falcon-H1-7B | 10.6869 | 10.6845 | 10.6989 |

Changes are within 0.12% on this sample. This is not a guarantee for arbitrary activations or equivalent numerical precision, and is not sufficient to promote full-K F16 accumulation to a default.

### Model performance and controls

The first full-model cohort used five repetitions, `pp512,pp1024,pp6144,tg512` in each process, and original/candidate/candidate/original ordering. These are throughput changes for the pre-cleanup mode-4 candidate, not cross-API model comparisons:

| Model | pp512 | pp1024 | pp6144 | tg512 |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2-135M | -9.82% | +1.48% | -0.32% | +0.08% |
| Qwen3-0.6B | +15.82% | +13.23% | +4.19% | +0.32% |
| Qwen3-4B | +31.90% | +29.39% | +9.15% | +1.09% |
| Phi-3-mini | +19.79% | +18.51% | +6.92% | +2.09% |
| Granite-3.0 | +0.26% | +2.89% | +5.66% | -8.65% |
| Falcon-H1-7B | +19.40% | +18.24% | +11.63% | +0.33% |

The two apparent regressions were not dismissed. SmolLM2's traced graph has no port dispatches, and single-token decode is outside this kernel's eligibility. A ten-repetition follow-up used original/clean-disabled/clean-mode4/clean-mode4/clean-disabled/original ordering, with `pp512,tg512`:

| Control, tokens/s | Original DLL | Clean DLL, mode 0 | Clean DLL, mode 4 |
| --- | ---: | ---: | ---: |
| SmolLM2 pp512 | 13182.36 | 13249.48 | 12670.70 |
| SmolLM2 tg512 | 358.45 | 359.58 | 358.85 |
| Granite pp512 | 2292.17 | 2275.54 | 2301.96 |
| Granite tg512 | 142.37 | 139.55 | 140.94 |

The follow-up does not reproduce a comparable default-path loss. It still shows several-percent movement for an unused mode on SmolLM2's short prefill, so this is not a blanket regression-free claim or evidence of a shader regression. Different prompt/repetition lists also consume different random-token sequences and must not be pooled. No decode speedup is claimed. The clean dispatcher avoids querying the port environment switch for ineligible nodes.

The clean build was then rechecked on the primary `pp6144` workload alone, three repetitions and original/candidate/candidate/original ordering:

| Model, tokens/s | Original | Clean mode 4 | Change |
| --- | ---: | ---: | ---: |
| Qwen3-4B | 347.675 | 364.530 | +4.85% |
| Falcon-H1-7B | 307.670 | 344.820 | +12.07% |

### Final state

The clean shader reproduces all six mode-3 and all six mode-4 perplexities above at printed precision. Modes 0 through 4 pass the four aligned numerical cases, including long K, broadcast batches and mixed Q4_K/Q8_0 scratch use. The route fixture passes 126 assertions, including all port modes, separate bias operations, explicit F32 precision, opt-outs and tail fallbacks. Malformed mode values are rejected. After the final wave-range hardening, all four port modes and the route fixture pass again; the four measured port shader blobs remain byte-identical.

The existing F16/Q8_0/Q4_K/Q5_0 wave16 blobs remain byte-identical to the original DLL, and the original native shader source is unchanged. The installed DLL is the clean implementation with wave-range guards, SHA256 `9ECE262D9553364BFBA809FCBDC47D7D33E5D45F884C09EEAEE0E57F35882CBD`. The original tuning file is restored; the new compilation stamp can trigger the existing cache-recalibration policy on an unforced launch. The terminal-event audit found no reconnects from the first operator cohort through final validation.

Evidence is in the session `files` directory under `gemm-20260914-*`: baseline source/build/DLL backups, distinct candidate DLLs, numerical and routing logs, reversed-order operator CSVs, the fixed-work profile and its integrated counters, and pinned model quality results. The previous uncommitted coverage expansion is preserved, and the build tree was not cleared.

## 56. Shared tiled GEMM and automatic bounded dispatch (2026-09-14)

The Q4_K-only experiment in section 55 is now `mul_mat_linalg_tiled_i.hlsl`. The same matrix core supports F16, Q8_0, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL and MXFP4. F16 and Q4_K/Q5_K have vector weight loaders; the other formats reuse `quant_dequant.hlsli`. CMake builds 44 specializations: eleven weight formats, two wave sizes and two accumulation modes. Selection is per tensor, not per model filename.

### Selection and precision

No runtime override is needed for qualified shapes on the B390 (`8086:B080`). Automatic selection uses wave32, F16 matrix partial sums drained into F32 totals every 64 reduction elements, and direct F32-to-F16 activation staging. It never selects full-K F16 accumulation. The runtime shape/capability checks still apply; other devices retain their previous defaults.

Here `m` is the number of output channels, `n` the token count of this operation, and `k` the reduction length. All automatic cases require `m >= 1024`, `n >= 512`, `k >= 1024`, whole 128x128 output tiles, and the existing layout, alignment and tensor-span guards.

| Weight type | Additional automatic selection rule |
| --- | --- |
| Q5_K, Q4_0, Q4_1, Q5_1, IQ4_NL, MXFP4 | None |
| Q4_K | `n == 512`; either `m == 1024` and `2048 <= k <= 4096`, or `m >= 3072`, `k >= 8192` and `2*k >= 5*m` |
| Q8_0 | `n == 512`; either `m == 1024` and `2048 <= k <= 4096`, or `m >= 4096` and `k >= 4096` |
| F16, Q5_0, Q6_K | Keep the previous default kernel |

These are conservative performance regions, not kernel correctness limits. Smaller batches, tails, other shapes and unsupported layouts retain the previous dispatcher. In particular, single-token decode does not enter this selector. F16/Q5_0/Q6_K remain available through the shared kernel's explicit override, but replacing their defaults did not win consistently.

`DX12_LINALG_TILED_GEMM` controls the shared family:

| Value | Meaning |
| --- | --- |
| Unset | Mode 2 on wave16 devices (since 2026-10-02); automatic mode 3 only with `DX12_LINALG_TILED_W32=1` |
| `0` | Disable the shared tiled family |
| `1` | Request wave16, bounded accumulation |
| `2` | Request wave16, full-K F16 accumulation |
| `3` | Request wave32, bounded accumulation |
| `4` | Request wave32, full-K F16 accumulation |

The old `DX12_LINALG_Q4K_VKPORT` name is a Q4_K-only compatibility fallback when the new variable is absent. The new variable takes precedence, including an explicit `0`. Existing global and per-format wave opt-outs also take precedence over tiled requests. Overrides do not bypass safety or capability checks. Full-K F16 modes remain opt-in, reject explicit F32 precision and require the conservative 128-byte output alignment. Bounded accumulation is not numerically identical to an all-F32 GEMM.

### Operator qualification

Four reversed-order cohorts compared the task-start DLL with explicitly requested bounded wave32: the original large GEMM, nine representative model shapes, lower-size/batch crossover cases and additional long-reduction boundaries. Together they cover 154 distinct format/shape combinations. Each cohort used baseline/candidate/candidate/baseline ordering, fixed tuning decisions and a seven-minute cooldown after CPU-heavy work. A separate boundary attempt stopped on RDP reconnection and was not accepted.

Examples from the crossover and clean boundary cohorts:

| Type | m, n, k | Previous default, us | Tiled bounded, us | Throughput change |
| --- | --- | ---: | ---: | ---: |
| Q4_K | 1024, 512, 2048 | 301.69 | 238.96 | +26.3% |
| Q8_0 | 1024, 512, 2048 | 305.06 | 253.78 | +20.2% |
| Q4_K | 4096, 512, 12288 | 6310.62 | 5303.28 | +19.0% |
| Q8_0 | 4096, 512, 12288 | 6264.62 | 5573.23 | +12.4% |
| Q5_K | 1024, 512, 1024 | 201.99 | 150.49 | +34.2% |

The blanket replacement also produced losses: Q4_K at 4096x512x8192 was 6.6% slower in throughput, Q8_0 at 2560x512x9728 was 25.1% slower, and large F16 GEMMs generally regressed. Those regions are not enabled automatically. Q4_0/Q4_1/MXFP4 and especially Q5_1/IQ4_NL show much larger gains over their previous defaults, which include slow non-LinAlg routes. Those gains are not comparisons against the fastest possible opt-in native wave kernel, or against Vulkan. Section 55's remaining Vulkan gap has not been closed by format generalization.

### Numerical coverage

All four explicit modes pass 24 aligned numerical cases, including long reductions and broadcast batches. The final route fixture passes 220 assertions, including automatic thresholds, precision overrides, opt-out precedence, tail fallbacks and mixed tiled/native scratch use. Reusing test objects exposed stale sentinel pointers from freed evaluation contexts; `test_case::eval()` now clears that per-evaluation state.

Fourteen paired model evaluations use the first 16 WikiText-2 chunks, context 512, batch 2048 and microbatch 512. These include Q4_K_M/Q8_0 versions of Qwen3-0.6B, Qwen3-4B, Phi-3-mini and Falcon-H1, plus six local pure-quant Qwen3-0.6B files generated from the same cached BF16 source. The largest PPL increase is approximately 0.021%. The pure Q5_K model changes from 26.0442 to 25.9197; lower PPL on this small sample is not a general quality-improvement claim. Candidate dispatch traces confirm use of the automatic family. Full-K F16 is not used in these comparisons.

The local pure MXFP4 file was generated with the existing `--pure MXFP4_MOE` quantizer option; the ordinary mixed-format `MXFP4_MOE` preset would not exercise dense MXFP4 weights on this model. An initial model-helper invocation generated no results because a PowerShell parameter named `Switch` interfered with the automatic `$switch` variable. It was discarded, the parameter was renamed, and empty-result checks were added before rerunning the complete quality cohort.

Evidence and task-start snapshots are under session `files\general-gemm-20260914-*`. The baseline is the completed section 55 DLL, not the older pre-port DLL. Existing uncommitted backend work and the configured build tree are preserved.

The first full-model performance run was interrupted before completion and the machine was rebooted at 17:30 on September 14. A 15:53 RDP reconnect fell inside the Qwen3-4B Q4_K_M comparison; that comparison and the incomplete Phi-3 Q8_0 comparison were discarded and restarted as whole four-arm cohorts. The helper now checks reconnects across each complete model comparison, including gaps between arms, and records per-arm timestamps. Completed unaffected comparisons remain separate from the post-reboot cohorts; absolute rates from opposite sides of the reboot are not pooled.

`GpuDebuggingLog.txt` also records a timeout notification at 15:02 during the old DLL's pure-Q5_1 perplexity run, before the candidate arm began. That old route is exceptionally slow, and produced a final PPL despite the notification. This is not evidence of a tiled-kernel timeout; it is also not a clean stability result for the old route. Post-reboot routing assertions pass again with the saved candidate DLL.

### Automatic model performance and controls

The completed comparison covers all eighteen text model/format combinations from `bench_linalg.bat` and three local pure-quant Qwen3-0.6B files. Each model uses three repetitions per arm, baseline/automatic/automatic/baseline ordering, batch 2048 and microbatch 512. Percentages below compare mean tokens/s within each model's uninterrupted cohort. They include variability, not just statistically established improvements.

| Model | pp512 | pp1024 | pp6144 | tg512 |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2 F16 | -3.25% | -9.80% | -2.68% | -2.02% |
| SmolLM2 Q8_0 | -5.04% | +3.76% | -0.75% | -0.42% |
| SmolLM2 Q4_K_M | -2.20% | +1.68% | +0.29% | +0.03% |
| Qwen3-4B F16 | +1.69% | +1.34% | +1.38% | +0.15% |
| Qwen3-4B Q8_0 | -0.28% | -0.84% | -0.09% | -0.31% |
| Qwen3-4B Q4_K_M | +1.89% | +2.85% | -1.74% | -0.89% |
| Phi-3-mini F16 | -0.08% | -1.21% | +1.07% | +0.10% |
| Phi-3-mini Q8_0 | +0.73% | +0.44% | -0.76% | -0.44% |
| Phi-3-mini Q4_K_M | +13.01% | +10.97% | +10.46% | -1.16% |
| Granite F16 | +1.08% | +0.12% | +0.19% | -2.32% |
| Granite Q8_0 | +0.64% | +0.02% | +0.11% | +1.28% |
| Granite Q4_K_M | +1.30% | -0.10% | -1.17% | +1.31% |
| Qwen3-0.6B BF16 | +0.02% | +0.70% | -0.40% | +0.21% |
| Qwen3-0.6B Q8_0 | +7.69% | +4.05% | +1.00% | +0.07% |
| Qwen3-0.6B Q4_K_M | -0.82% | +2.50% | +4.81% | +4.15% |
| Falcon-H1 BF16 | +1.30% | +0.50% | -0.26% | +0.07% |
| Falcon-H1 Q8_0 | -0.23% | +0.40% | -0.97% | +0.13% |
| Falcon-H1 Q4_K_M | +0.63% | -7.22% | +3.32% | +3.50% |
| Qwen3-0.6B pure Q4_0 | +232.29% | +215.14% | +121.16% | -1.27% |
| Qwen3-0.6B pure Q5_K | +35.48% | +32.98% | +12.57% | -0.16% |
| Qwen3-0.6B pure MXFP4 | +112.35% | +112.03% | +63.28% | -1.57% |

The pure Q5_K filename retains `Q5_K_M` from the quantizer invocation, but `--pure` makes this a pure-Q5_K comparison, not the standard mixed Q5_K_M recipe. BF16 rows above correspond to the cached BF16 files selected by the script's F16 model aliases.

The conspicuous negative results received separate baseline/new-disabled/new-automatic/new-automatic/new-disabled/baseline controls. SmolLM2 uses ten repetitions with all four workloads, Falcon uses seven repetitions of pp1024 alone, and Qwen3-4B uses five repetitions of pp6144 alone.

| Control, tokens/s | Baseline | New DLL, disabled | New DLL, automatic |
| --- | ---: | ---: | ---: |
| SmolLM2 F16 pp512 | 17807.49 | 17969.42 | 17816.60 |
| SmolLM2 F16 pp1024 | 15141.16 | 14687.36 | 14732.56 |
| SmolLM2 F16 pp6144 | 5336.99 | 5335.75 | 5335.53 |
| SmolLM2 F16 tg512 | 252.22 | 251.06 | 251.64 |
| Falcon-H1 Q4_K_M pp1024 | 465.31 | 454.03 | 472.06 |
| Qwen3-4B Q4_K_M pp6144 | 334.86 | 335.00 | 335.88 |

The initial Falcon pp1024 decrease includes a candidate sample of 413.56 +/- 54.16 t/s, versus 502.50 +/- 2.39 in the second candidate arm. The controlled rerun does not reproduce an automatic-path loss, but remains noisy in all arms. Qwen3-4B's small pp6144 loss also does not reproduce. SmolLM2's pp6144 and pp512 are neutral in the control, while pp1024 still averages approximately 2.7% lower with the new DLL, similarly with the tiled route disabled. Its individual sample standard deviations reach about 1000 t/s, and the two baseline arms move from 15527.69 to 14754.63 t/s. No tiled dispatch applies to this F16 model. This evidence does not support a blanket regression-free claim. No single-token decode speedup is attributed to this change.

### Shipping state

The final DLL SHA256 is `32E0B309163CD3EC905E492D93EB070A9505C9AB0F6EFF7D660363047C022CF1`. All 44 tiled shader blobs match the first generalized candidate used in the operator cohorts. The four existing F16/Q8_0/Q4_K/Q5_0 wave16 blobs match the task-start baseline. The installed DLL matches the saved final artifact, and all four explicit modes pass the 24 numerical cases again after reboot. The original tuning snapshot is unchanged. No probe executable is left in the runtime directory.

Accepted post-reboot model/control intervals have no RDP reconnects, and the timeout log has not changed since the old Q5_1 run. Detailed model results are in `general-gemm-final-model-summary.csv`, `general-gemm-postboot-*-summary.csv`, the per-arm logs/CSVs and the three control cohorts. The pre-reboot incomplete CSV and reconnect-contaminated comparison are retained as rejected evidence, not silently merged into the completed cohorts.

## 57. Packed Q8_0, Q5_0 and Q6_K staging; rejected FP16 variants

September 15, 2026, Intel Arc B390. This work starts from section 56's completed generalized DLL, SHA256 `32E0B309163CD3EC905E492D93EB070A9505C9AB0F6EFF7D660363047C022CF1`, not the earlier Q4_K-only implementation. The retained changes specialize weight staging inside the existing shared tiled kernel. Matrix geometry, bounded accumulation, activation staging and output stores are unchanged.

### Packed loaders and automatic selection

Q8_0 now assigns eight adjacent values to each staging invocation. It reads the block scale once and unpacks two DWORDs. Q5_0 and Q6_K assign four adjacent values, reuse their shared scale, and unpack low/high bits together instead of calling the scalar decoder four times. The local DWORD reader handles halfword-aligned packed blocks without reading a second DWORD when the address is already aligned.

All non-MXFP4 tiled routes now require halfword-aligned row and batch strides. F16/Q4_K/Q5_K retain the stricter DWORD alignment required by their raw loaders. Root-offset alignment, tensor-span limits, capability checks, tail fallbacks and format/global opt-outs remain in effect.

The automatic B390 policy extends section 56 as follows, with its existing common size and full-tile guards:

| Type | Updated rule |
| --- | --- |
| Q5_0, Q6_K | Enable bounded wave32 for `m >= 1024`, `n >= 512`, `k >= 1024` |
| Q8_0 | Require `n == 512`; either `m == 1024` and `1024 <= k <= 4096`, or `m >= 2048`, `k >= 4096` and (`m >= 4096` or `k >= 2*m`) |
| F16 | Keep the previous default and shader bytecode |

Other format rules are unchanged. No new runtime switch is required. `DX12_LINALG_TILED_GEMM=0` disables the entire shared family, including section 56's changes, rather than isolating this specialization. Use the section 56 DLL for a true before/after comparison. Full-K F16 remains opt-in; single-token decode does not use these new routes.

### Operator evidence and rejected quant loaders

Accepted cohorts use reversed ordering, fixed tuning choices, disconnected RDP, and a 420-second idle cooldown after builds or numerical work. Throughput percentages are `(old_us / new_us - 1) * 100`, not reductions in execution time.

| Type | m, n, k | Section 56 default, us | Specialized, us | Throughput change |
| --- | --- | ---: | ---: | ---: |
| Q8_0 | 1024, 512, 3072 | 355.31 | 304.71 | +16.61% |
| Q8_0 | 4096, 512, 14336 | 6845.98 | 6015.69 | +13.80% |
| Q5_0 | 4096, 512, 14336 | 9546.99 | 5988.74 | +59.42% |
| Q6_K | 4096, 512, 14336 | 11166.77 | 6724.25 | +66.07% |

The eight-value Q8_0 loader improves all ten measured shapes versus the old shared Q8_0 shader by 7.59-18.74%. It does not beat the old default at every shape: 3072x512x1024 loses 22.53%, and 9728x512x2560 loses 4.91%. A crossover test at 1536x512x3072 loses 14.87%. These remain on the previous route. Q5_0 and Q6_K improve all ten original model-shaped points and all ten additional crossover points per type, including both 512- and 1024-token batches.

A four-value Q8_0 loader reduced static DXIL raw-load call sites from 16 to 6 but regressed badly on some shapes. A whole-block loader was mostly neutral or mildly beneficial. Neither is retained. The Q5_0/Q6_K four-value loaders reduce the same static count from 48 to 10 and 32 to 12. These are compiler call-site counts, not executed hardware instructions or memory transactions.

### FP16: no default promotion

Three native layouts with integrated F32 activation conversion were evaluated: 32x128, 16x256 and 64x128 token-by-channel tiles. None consistently beats the existing preconverted FP16 route. The reusable activation conversion pass is not simply removable overhead.

A 64x128 layout retaining that pass showed some promising isolated wins, but a 19-shape follow-up did not justify broad selection. For example, an earlier gain at `m=2560,k=9728` shrank to 0.79%, and one at `m=8192,k=3072` shrank to 0.34%. The follow-up improved `m=2048/2304,k=8192` by about 18%, but lost 8.76% at `m=4096,k=2560`. No general model-level benefit was established.

A final experiment kept the shared kernel's composed 32x32 matrices but loaded FP16 weights directly from the descriptor instead of staging them in LDS. It passed the selected numerical cases in all four modes, but lost 12.49-30.00% versus the old shared kernel and 21.30-55.78% versus the previous automatic default over ten shapes.

All experimental FP16 flags, generated variants and activation branches were removed. The existing native FP16 source and CMake configuration match the task-start snapshot. Do not use isolated optimistic FP16 points as evidence that a new default is faster. These experiments do not establish Vulkan parity.

### Validation and artifacts

The cleaned candidate passes all 24 aligned numerical cases in each of the four explicit modes and 238 route assertions. Coverage includes new Q8_0 boundaries, Q5_0/Q6_K format and global opt-outs, broadcast batches, long reductions, precision requirements, and mixed native/tiled scratch use.

Its DLL SHA256 is `CF6BAF616659D404897697DF92C442A1D635708298FAEA54EB3A0C24DBFB2D4B`. All twelve specialized shader blobs match the measured candidate, while the other 32 tiled blobs and four checked native wave16 blobs match the section 56 baseline. The build cache is unchanged.

Evidence is stored under session `files\specialize-gemm-*`, including the task-start source/DLL snapshot, rejected candidate DLLs, per-arm operator logs, paired summaries, numerical logs and bytecode checks.

Fourteen paired model-quality runs compare the first 16 WikiText-2 chunks at context 512, batch 2048 and microbatch 512. They cover Q8_0 and Q4_K_M versions of all six text-model families in `bench_linalg.bat`, plus local pure-Q5_0 and pure-Q6_K Qwen3-0.6B files generated from the same BF16 source. This includes Q4_K_M models because their Q6_K tensors can now change routes.

| Selected PPL comparisons | Section 56 | Specialized |
| --- | ---: | ---: |
| Phi-3-mini Q4_K_M | 7.1321 | 7.1323 |
| Qwen3-4B Q4_K_M | 12.5100 | 12.5020 |
| Qwen3-0.6B pure Q5_0 | 26.5255 | 26.5310 |
| Qwen3-0.6B pure Q6_K | 25.1519 | 25.1616 |

The largest PPL increase is 0.0386%, on pure Q6_K. All pairs remain below the 0.1% increase gate. Small PPL decreases on this sample are not claims of improved model quality. These comparisons use automatic bounded selection, not full-K F16.

### Model performance

Sixteen model combinations use three repetitions per arm and baseline/automatic/automatic/baseline ordering at batch 2048 and microbatch 512. These include all twelve Q8_0/Q4_K_M text combinations in the user's script, the two pure-quant files, and two unchanged FP16 controls. Each row is one uninterrupted comparison; absolute results from different cohorts are not pooled.

| Model | pp512 | pp1024 | pp6144 | tg512 |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2 Q8_0 | -8.86% | +2.12% | -0.08% | +0.31% |
| SmolLM2 Q4_K_M | +3.60% | +1.49% | +0.20% | +1.81% |
| Qwen3-0.6B Q8_0 | -1.74% | +0.70% | -1.39% | +0.03% |
| Qwen3-0.6B Q4_K_M | +13.46% | +4.34% | +1.95% | -0.27% |
| Phi-3-mini Q8_0 | +3.29% | +2.82% | +0.81% | -1.78% |
| Phi-3-mini Q4_K_M | +6.54% | +6.28% | +3.00% | +0.21% |
| Qwen3-4B Q8_0 | +0.70% | +2.22% | +1.43% | +0.38% |
| Qwen3-4B Q4_K_M | +6.34% | +5.73% | +2.74% | +0.41% |
| Granite Q8_0 | +0.53% | -0.37% | +0.37% | -0.43% |
| Granite Q4_K_M | +0.22% | +0.02% | +0.39% | +5.33% |
| Falcon-H1 Q8_0 | +5.32% | +7.22% | +3.69% | -0.35% |
| Falcon-H1 Q4_K_M | +6.91% | +6.78% | +5.52% | -0.67% |
| Qwen3-0.6B pure Q5_0 | +17.26% | +14.80% | +10.20% | -0.58% |
| Qwen3-0.6B pure Q6_K | +34.00% | +27.08% | +13.88% | -0.16% |
| SmolLM2 F16 control | +1.56% | +2.30% | -0.66% | -0.05% |
| Phi-3-mini F16 control | -0.70% | +0.30% | -1.59% | -0.07% |

These are observed mean changes, not confidence intervals or universal speedups. In particular, the decode variations are not attributed to GEMM specialization. The Q8_0 short-prefill losses and unchanged Phi-3 FP16 pp6144 result receive separate disabled/automatic controls rather than being omitted from the report.

The controls use baseline/new-disabled/new-automatic/new-automatic/new-disabled/baseline ordering. The two Q8_0 models use ten repetitions per arm for each prefill workload; Phi-3 FP16 uses five repetitions of pp6144. The disabled arm disables the entire tiled family, including section 56.

| Control, tokens/s | Baseline | New DLL, disabled | New DLL, automatic |
| --- | ---: | ---: | ---: |
| SmolLM2 Q8_0 pp512 | 14909.58 | 15037.67 | 15190.19 |
| SmolLM2 Q8_0 pp1024 | 13278.22 | 13435.91 | 13289.90 |
| SmolLM2 Q8_0 pp6144 | 5181.81 | 5185.74 | 5191.84 |
| Qwen3-0.6B Q8_0 pp512 | 5460.28 | 5116.85 | 5418.98 |
| Qwen3-0.6B Q8_0 pp1024 | 4536.84 | 4363.10 | 4573.34 |
| Qwen3-0.6B Q8_0 pp6144 | 1219.85 | 1196.99 | 1228.25 |
| Phi-3-mini F16 pp6144 | 520.46 | 521.48 | 518.38 |

SmolLM2's initial Q8_0 pp512 loss does not reproduce. Qwen3-0.6B Q8_0 pp6144 changes to +0.69%, while pp512 remains slightly lower at -0.76%; no universal Q8_0 speedup is claimed. Its absolute pp6144 rates differ substantially between cohorts, reinforcing the need for within-cohort comparisons. The unchanged Phi-3 FP16 control is -0.40%, rather than the initial -1.59%. The FP16 implementation and bytecode are unchanged.

All accepted model/control comparisons completed without RDP reconnects. The installed DLL matches the cleaned candidate after the helpers restore it, and `GpuDebuggingLog.txt` remains unchanged since the old Q5_1 timeout reported in section 56. Complete rows are in `specialize-gemm-final-performance-summary.csv`, `specialize-gemm-q8-controls-summary.csv` and `specialize-gemm-f16-control-summary.csv`, with per-arm timestamps, hashes and logs retained alongside them.

## 58. Attention reprofile, scalar-port comparison, and remaining packed loaders

The September 15 follow-up starts from section 57's DLL, `CF6BAF616659D404897697DF92C442A1D635708298FAEA54EB3A0C24DBFB2D4B`. It tests attention independently of the GEMM work: a Vulkan-inspired GEMM win does not establish that the same approach will improve FA.

### Current attention share

The shipping DX12 backend was profiled at pp6144, batch 2048, microbatch 512, with F16 KV and FA enabled. The table sums GPU operation timestamps for graphs 13-24, excluding the warmup prompt and its zero-timestamp first graph. These are instrumented operation shares, not clean backend throughput.

| Q4_K_M model | Summed operation time, ms | FA share | Dense GEMM share |
| --- | ---: | ---: | ---: |
| SmolLM2-135M | 1156.31 | 74.61% | 19.99% |
| Phi-3-mini | 14166.17 | 55.51% | 39.40% |
| Qwen3-4B | 18063.25 | 60.76% | 33.43% |
| Falcon-H1-7B | 19500.22 | 23.50% | 55.32% |

Separate Vulkan pipeline traces still select scalar `flash_attn_f32_f16_aligned`. The current `get_fa_tuning_params_scalar()` and actual dispatches give Br4 for D64, but Br8 with row-split4 for D96/D128, not Br4 for every dimension. All three use 128 threads, D-split8 and Bc32, with direct K/V loads on Intel. Pipeline tracing is excluded from accepted performance comparisons.

### Scalar FA port: tested, rejected

Two DX12 arithmetic variants reproduce that partition, direct vectorized F16 K/V loading, per-column-stream online softmax, and final stream merge. The first retains F32 Q, probabilities and output state. The second uses F16 Q/probability/output storage, F32 score/softmax statistics, `dot2add`, and Vulkan's `3*log(2)` maximum offset. Both support masks, ALiBi, softcap, sinks, GQA, batches, tails, and the existing split-KV partial layout. Wave16 and wave32 were tested separately.

This is a dataflow comparison, not an identical shader/compiler comparison. The port classifies masks inline instead of using Vulkan's separate mask-optimization prepass. The mixed variant also uses HLSL `dot2add`; Vulkan's generated scalar shader has separate floating-point and optional mixed-dot variants.

Both arithmetic variants passed 444 selected FA cases per wave size and 256 route assertions, including 18 temporary scalar-route assertions. Neither variant wins any of the nine paired operator cases: D64/96/128, 512 queries, and KV512/1024/6144. The mixed variant's best wave choice still loses 11.60-34.15% throughput versus the shipping FA implementation.

The matched Vulkan comparison is much narrower than the earlier GEMM gap. At KV512/1024 it ranges from 1.02x to 1.21x shipping DX12 throughput. At KV6144, Vulkan is 1.12x DX12 for D64, but 0.77x for D96 and 0.82x for D128. These tests use explicit F32 precision, masked F16 KV, and the same test generator on both backends; they are not end-to-end model comparisons or a claim that DX12 FA always beats Vulkan.

Actual pp6144 model comparisons confirm the rejection. Baseline/mixed-wave32/mixed-wave32/baseline ordering, with two repetitions per arm:

| Q4_K_M model | Shipping, tokens/s | Mixed scalar port, tokens/s | Change |
| --- | ---: | ---: | ---: |
| SmolLM2-135M | 4953.42 | 3883.64 | -21.60% |
| Phi-3-mini | 481.74 | 380.66 | -20.98% |
| Qwen3-4B | 339.44 | 305.53 | -9.99% |

All temporary scalar shaders, host selectors, build rules and route assertions were removed. The previous FA source, dispatch and default remain unchanged. Eight new model-shaped cases remain in the existing performance generator so the nine-point comparison can be repeated without adding a new test binary. The rejected source variants and DLLs are archived under session `files\fa-packed-*`; `DX12_FA_VK_SCALAR` was only an experiment and is not a retained switch.

### Packed quant staging

The shared tiled GEMM now decodes four neighboring values together for Q4_0, Q4_1, Q5_1, IQ4_NL and MXFP4. It reuses each scale, minimum and packed quant word; Q5_1 also reuses its high-bit word. Arithmetic order and final F16 conversion are preserved. Matrix geometry, activation staging, accumulation, precision guards and automatic dispatch thresholds are unchanged.

MXFP4's 17-byte blocks require byte-aligned word assembly, unlike the existing halfword-aligned helper. Its second load addresses the DWORD containing the last requested byte, not an unconditional following DWORD. This also keeps a speculatively evaluated aligned-tail load inside the existing DWORD-aligned buffer span.

Paired baseline/new/new/baseline operator results at `m=4096,n=512,k=14336`. The baseline selects bounded wave32 automatically; the new arm explicitly selects the same route with mode 3:

| Type | Before, us | Packed, us | Throughput change |
| --- | ---: | ---: | ---: |
| Q4_0 | 6383.20 | 5898.40 | +8.22% |
| Q4_1 | 6761.56 | 5882.80 | +14.94% |
| Q5_1 | 10177.01 | 7055.02 | +44.25% |
| IQ4_NL | 8151.68 | 7409.62 | +10.01% |
| MXFP4 | 8271.97 | 7553.70 | +9.51% |

All 20 updated tiled blobs are present in the measured candidate; the other 24 tiled blobs are byte-identical to section 57. The existing F16, Q8_0, Q4_K, Q5_K, Q5_0 and Q6_K tiled implementations are not changed by this work. No new environment flag is needed, and full-K F16 accumulation remains opt-in.

### Broad sweep and tighter controls

A subsequent ten-shape sweep per format showed substantial variation, including apparent regressions as large as 34.67% for IQ4_NL. It is not used to claim an all-shape speedup. For example, the same new Q4_1 shader at 1024x512x3072 measured 444.01 and 303.50 us in its two arms, while the old arms were 353.21 and 357.06 us.

Every shape with an apparent regression received a shorter, separate comparison: `(m,k) = (1024,1024), (1024,3072), (3072,1024), (2560,9728), (3072,8192), (8192,3072)`, all with `n=512`. Each comparison includes all five changed formats and a byte-identical Q8_0 control. Both DLLs explicitly select bounded wave32 mode 3, with old/new/new/old ordering.

| Format | Throughput change over the six focused shapes |
| --- | ---: |
| Q4_0 | +8.44% to +12.77% |
| Q4_1 | +12.01% to +16.71% |
| Q5_1 | +36.52% to +98.09% |
| IQ4_NL | +3.22% to +11.10% |
| MXFP4 | +2.55% to +8.90% |
| Q8_0, unchanged control | -1.38% to +0.54% |

The earlier losses do not reproduce: all 30 changed-format points improve, while the unchanged control stays close to parity. No shape exclusions or runtime branches were added on the basis of the noisy broad sweep. Both the original sweep and the focused comparisons remain in the artifacts, rather than replacing the unfavorable observations with only their repeats.

### Numerical qualification

The cleaned backend passes all 24 aligned cases in each of the four tiled modes and all 238 retained route assertions. Small reductions, long reductions, broadcast batches, all eleven tiled formats, and the existing precision/fallback guards remain covered. The temporary FA route tests are not retained after removing the rejected implementation.

Five paired quality comparisons use local Qwen3-0.6B quantizations, the first 16 WikiText-2 chunks, context 512, batch 2048 and microbatch 512. Dispatch traces confirm the automatic bounded tiled route in every candidate run.

| Weight format | Before PPL | Packed PPL |
| --- | ---: | ---: |
| Q4_0 | 28.7348 | 28.7348 |
| Q4_1 | 30.9181 | 30.9181 |
| Q5_1 | 26.2868 | 26.2868 |
| IQ4_NL | 28.7079 | 28.7079 |
| MXFP4 | 32.1348 | 32.1348 |

PPL is unchanged at the reported precision on this sample; this is not a claim of bit-identical outputs for every input. These comparisons use automatic bounded accumulation, not the optional full-K F16 mode.

### Model performance and confirmation

The first complete model comparison uses three repetitions per arm, baseline/automatic/automatic/baseline ordering, and the same five local Qwen3-0.6B weight files as the quality comparison:

| Format | pp512 | pp1024 | pp6144 | tg512 |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | +11.71% | +3.16% | +3.71% | +6.48% |
| Q4_1 | +8.92% | +4.78% | +6.50% | +5.66% |
| Q5_1 | +39.25% | +30.05% | +9.26% | +7.13% |
| IQ4_NL | +3.31% | +3.24% | +4.95% | +3.30% |
| MXFP4 | +3.82% | +6.46% | +4.66% | -0.33% |

These are observed means, not confidence intervals. The decode changes cannot be attributed to the packed GEMM loaders: those shaders do not run at one token. Several initial arms also have substantial within-arm variation. Therefore the primary pp6144 result was repeated separately with five repetitions per arm and the same reversed ordering:

| Format | Before, tokens/s | Packed, tokens/s | pp6144 change |
| --- | ---: | ---: | ---: |
| Q4_0 | 1294.23 | 1298.67 | +0.34% |
| Q4_1 | 1253.07 | 1302.96 | +3.98% |
| Q5_1 | 1168.34 | 1265.36 | +8.30% |
| IQ4_NL | 1211.08 | 1224.12 | +1.08% |
| MXFP4 | 1177.32 | 1205.44 | +2.39% |

An unchanged Qwen3-0.6B Q8_0 control, also using five repetitions per arm, changes by -0.47% at pp512, +0.26% at pp1024, +0.62% at pp6144, and +0.28% at tg512. Q4_0 is effectively neutral at pp6144, and IQ4_NL's model-level improvement is small. Q5_1 has the clearest end-to-end benefit; the isolated loader gains do not transfer proportionally to the attention-heavy long prompt. No decode speedup or broad Vulkan parity is claimed.

All reported comparisons audit RDP reconnects and serialize GPU work. Performance follows a 420-second idle cooldown after builds, numerical work and CPU-heavy checks; no builds or heavy analysis run during accepted timing. Fixed tuning choices avoid DLL-timestamp-dependent autotuning differences. Absolute rates from different cohorts are not pooled.

### Final state and artifacts

The installed `build_linalg\bin\Release\ggml-dx12.dll` has SHA256 `4D19B8EA848D64A629446D3A21A448DC93155DEEC7A7EADFEF073AAEE7C61643`. A whole-DLL comparison finds exactly 20 replaced DXIL containers and 1701 unchanged containers, including the existing attention shaders. The host source and CMake configuration match the task-start snapshot, and the build cache is unchanged. `GpuDebuggingLog.txt` remains unchanged since September 14.

Session `files\fa-packed-*` contains the task-start source/DLL snapshot, profile JSONL files, Vulkan route traces, both rejected scalar arithmetic variants, per-arm operator/model logs, quality traces, timestamps, DLL hashes, and the final bytecode comparison. Key summaries are `fa-packed-final-quality-summary.csv`, `fa-packed-final-models-summary.csv`, `fa-packed-focused-summary.csv`, `fa-packed-focused-models-summary.csv`, and `fa-packed-unchanged-control-summary.csv`. An initial model invocation with an incorrectly formatted prompt-list argument failed before benchmarking; it is not included in the completed comparisons.

## 59. VTune-guided attention investigation and distributed Q caching (2026-09-15)

### Measurement and Vulkan architecture

The new baseline is commit `fca93cf51`, with `ggml-dx12.dll` SHA256 `3B0999F36943402CF00025EA63EC4079071AFA2DDE95A329E97765BA8D46F98D`. This is not the earlier section 58 measured DLL. GPU, driver, pinned DXC and preview SDK are unchanged.

A session-local harness runs one FA node with fixed, hashed inputs, eight warmups outside collection, and 128 synchronized executions inside an ITT resume/pause interval. The primary workload is D128, eight KV heads, GQA4, KV6144, 512 queries, contiguous F16 KV, explicit F32 precision, and a last-block causal mask. VTune 2026.3 collects global-memory-access hardware counters. These captures are diagnostic, not clean operator benchmarks.

The current Vulkan implementation has three distinct architectures:

- Scalar Intel tuning uses D-split8, direct KV loads and independent column-stream softmax, with a final stream merge. Intel disables enforced subgroup sizing; its nominal tuning width of 32 is not proof of actual SIMD32 execution. With FP16 enabled, explicit F32 selects F32 QK accumulation but does not make Q, P and persistent O all F32.
- CM1 uses Br16/Bc64 and explicit shared-memory exchanges between QK, row softmax and PV ownership. Its F16 KV path still uses half PV accumulation even when QK requests F32.
- CM2 keeps Q/O as workgroup-scoped matrices and exposes matrix reductions/conversions. Native wave-scope multiplication alone does not provide this architecture.

Source references are `ggml-vulkan.cpp` (`ggml_vk_flash_attn` and pipeline tuning), `flash_attn.comp`, `flash_attn_cm1.comp`, `flash_attn_cm2.comp`, `flash_attn_base.glsl`, and `vulkan-shaders-gen.cpp`. Vulkan's mask prepass classifies actual mask values, skips all-masked tiles, and removes mask staging for all-zero tiles. Large scalar direct-KV iterations can then avoid workgroup barriers until the final stream merge. The section 58 model traces select `flash_attn_f32_f16_aligned` on this machine; the generic `KHR_coopmat` device banner does not establish the FA route.

### Initial hardware-counter evidence

| Route | Profiled ms/op | XVE active | XVE stalled | Thread occupancy | GPU memory reads, GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shipping scalar DX12 | 49.212 | 70.4% | 27.6% | 75.6% | 9.413 |
| Native LinAlg FA | 151.364 | 35.8% | 53.7% | 44.4% | 1.810 |
| Vulkan | 43.914 | 32.7% | 65.5% | 75.2% | 10.304 |

Integrated counters divided by the fixed operation count:

| Counter per FA operation | Scalar DX12 | Native LinAlg | Vulkan |
| --- | ---: | ---: | ---: |
| GPU memory read bytes | 463.4 MB | 274.0 MB | 452.6 MB |
| L3 read bytes | 3.106 GB | 3.750 GB | 4.334 GB |
| SLM read bytes | 30.353 GB | 9.666 GB | 3.139 GB |
| SLM write bytes | 4.422 GB | 3.993 GB | 0.0467 GB |
| Issued-instruction counter | 8.194 billion | 12.096 billion | 2.494 billion |
| SLM fence-message counter | 21.194 million | 6.125 million | 0.527 million |

Native LinAlg is not approaching external-memory bandwidth saturation. Its lower occupancy and higher instruction count suggest that conversion, synchronization and fallback work must be priced before assuming matrix throughput will help. Scalar DX12's much larger SLM traffic motivates reducing repeated Q loads without changing its successful geometry.

These are adapter-level counters, not independently process-exclusive measurements. The captures have no dispatch table, so the 128-operation count comes from the synchronized harness, not 128 profiler dispatch records. ITT bounds agree with the harness interval. The L3 counter requires a 64-byte scale; the memory/SLM byte counters do not. A separate source-analysis/stall-sampling probe produced raw stall data but no attributed computing tasks or shader report. No source-line stall attribution is claimed.

### Rejected native fallback experiment

The native FA guard treats a raw subnormal anywhere in a Br32 query tile or Bc32 key tile as unsafe for half staging. That sends the entire QK tile through original-value F32 dot products. Emulating these unchanged guards on the exact captured inputs predicts 35,953 of 94,464 active tiles, or 38.060002%, taking that fallback. This is input/source evidence, not a GPU branch counter, and must not be generalized to every model's activations.

The first experiment vectorizes Q loads and divides each original-value score over eight lanes, retaining all range, scaling and subnormal guards. It passes 800 selected numerical cases and 238 route assertions. However, paired old-native/new-native comparisons are mixed: D64 gains 7.17-12.63% throughput, D96 gains 5.45-52.47%, and D128 loses 4.60-26.57%. Every new-native point is still much slower than shipping scalar FA.

That implementation was archived and reverted in full. A frequent fallback is worth understanding, but parallelizing it alone does not make this native matrix architecture competitive. The native FA default and numerical safeguards are unchanged.

### Retained scalar experiment

As of September 16, wave-distributed packed Q caching is enabled by default for the existing B390 D64 and D128 prefill routes. Unset or `DX12_FA_Q_REGS=1` enables it; `DX12_FA_Q_REGS=0` opts out. It requires native FP16, wave16, the wave16 blob selection, and the existing scalar prescan route. `DX12_FA_PF_FP16=0` retains F32 staging instead. Native LinAlg selections, D96, other head dimensions, decode and other GPUs retain their previous routes.

The shader stages Q exactly as before, then distributes half4 values across wave lanes as packed integer pairs. Each lane holds eight DWORDs of Q cache for the retained geometries. QK broadcasts the required pair rather than reading Q from shared memory again for every KV tile. K loading, QK arithmetic order, score/softmax/PV precision, masks, tile geometry and split policy are unchanged. This is a scalar optimization; it does not require the LinAlg preview build.

The initial D96 version used sixteen DWORDs per lane and lost 17.06-18.16% throughput. It is not built or selectable in the retained implementation.

Nine-point operator cohort, baseline/cache/Vulkan/Vulkan/cache/baseline ordering; times are microseconds per operation:

| D | KV | Shipping DX12 | Q cache | Throughput change | Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | 445.89 | 442.04 | +0.87% | 452.04 |
| 64 | 1024 | 819.76 | 789.52 | +3.83% | 794.01 |
| 64 | 6144 | 4891.84 | 4651.08 | +5.18% | 4373.96 |
| 128 | 512 | 3208.80 | 2966.46 | +8.17% | 2648.77 |
| 128 | 1024 | 5728.76 | 5351.03 | +7.06% | 5204.74 |
| 128 | 6144 | 38341.50 | 32770.84 | +17.00% | 46984.34 |

D64/KV512 is effectively neutral in this cohort. D128 improves materially, but Vulkan still leads at KV512/1024. The KV6144 advantage over Vulkan is workload-specific, not general FA parity. An earlier invocation matched an unwanted tenth shape and stopped after its baseline arm; only the corrected complete cohort is used above.

The cache passes 800 selected FA cases plus 113 BF16 KV cases. Six fixed-input comparisons, D64/D128 with no mask, last-block causal mask and first-block causal mask, have identical output hashes. Two paired 16-chunk WikiText-2 comparisons also retain the reported PPL: SmolLM2-135M Q4_K_M 23.4473, Qwen3-4B Q4_K_M 12.5020. Dispatch traces confirm the new routes. These observations are not a proof of bit identity for every possible input.

### Paired profile of the retained D128 cache

A fresh shipping/cache VTune pair uses the same fixed workload and identical input/output hashes. Both captures follow the 420-second cooldown. A separate trace of the exact Vulkan workload confirms `flash_attn_f32_f16_aligned`, rather than a cooperative FA pipeline.

| Per-operation metric | Shipping | Q cache | Change |
| --- | ---: | ---: | ---: |
| Instrumented time | 38.724 ms | 32.896 ms | -15.05% |
| SLM read bytes | 29.782 GB | 23.741 GB | -20.29% |
| SLM write bytes | 4.339 GB | 4.343 GB | +0.10% |
| L3 read bytes | 2.959 GB | 3.190 GB | +7.78% |
| GPU memory read bytes | 465.36 MB | 507.38 MB | +9.03% |
| Issued-instruction counter | 8.040 billion | 8.337 billion | +3.69% |
| Stalled-XVE counter | 2.704 billion | 1.321 billion | -51.14% |
| SLM fence-message counter | 20.796 million | 20.817 million | +0.10% |

The average GPU clock inferred from clocks/time is approximately 2.44 GHz in both captures. SLM bank-conflict count is effectively unchanged (+0.05%). The result supports reducing repeated shared-Q access and its dependency cost, not reducing total instruction count, external-memory traffic or the number of barriers. The stalled-XVE row is an integrated counter per operation, not a percentage of execution time or source-line attribution.

### End-to-end results

The model cohort uses three repetitions per arm, baseline/cache/cache/baseline ordering, fixed tuning choices, and RDP auditing. All models below are Q4_K_M:

| Model | pp512 change | pp1024 change | pp6144 before, tokens/s | pp6144 cache, tokens/s | pp6144 change | tg512 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M | +5.03% | +3.26% | 4899.05 | 5092.26 | +3.94% | +0.64% |
| Phi-3-mini, unchanged D96 control | -0.10% | -1.19% | 458.16 | 458.23 | +0.01% | -0.65% |
| Qwen3-4B | +0.96% | +1.57% | 341.62 | 355.17 | +3.97% | -0.06% |

These are observed means, not confidence intervals. SmolLM2's pp512 baseline arms vary substantially, so its short-prompt percentage is not a robust standalone speedup claim. The primary pp6144 result is more consistent: SmolLM2's baseline arms are 4895.25/4902.84 and cache arms 5076.96/5107.55; Qwen3's are 340.52/342.71 and 355.40/354.94. The approximately 4% model gains are smaller than the isolated long-KV D128 gain because whole-model prefill contains other operations, earlier query blocks and shorter KV lengths. No decode speedup is attributed to this change.

### September 15 state and artifacts

The September 15 backend retained only the two opt-in scalar Q-cache shaders, with flags 296 and 298. The native fallback experiment and D96 cache are not shipped. Route assertions passed 245/245, including opt-out, D96 fallback and preservation of the F32-staging override. The final host cleanup added explicit F32-staging/blob-width guards; its retained shader blobs are byte-identical to those measured above.

The September 15 `build_linalg\bin\Release\ggml-dx12.dll` has SHA256 `684CF6ADB17375C0AF74282E5FC58F762F577BD15FC7B39ACDA67734DDF8A20A`. Whole-DLL inspection found all 1721 baseline DXIL containers unchanged and exactly two additional cache shaders. No production shader was replaced, and the cache was initially opt-in.

Session `files\fa-deep-*` contains the baseline, rejected source/DLL experiments, fixed-work ITT harness, exact-input fallback prediction, VTune captures, integration script, numerical/route logs, per-arm operator/model results and final bytecode inspection. Main result tables are `fa-deep-qregs-models-summary.csv` and `fa-deep-qregs-d128-global-metrics.csv`. Performance is serialized, audited for RDP reconnects, and separated from builds/CPU-heavy work by the cooldown; absolute rates from separate cohorts are not pooled.

### Default enablement (2026-09-16)

The cache is now automatic within the same B390-only hardware and D64/D128 prefill guards. This changes only the unset environment-variable behavior, not the shaders or supported shapes. Explicit `0` still opts out, explicit `1` retains the same gates, and invalid values remain errors. The existing route fixtures cover unset, enabled and disabled selection for D64/D96/D128.

## NVIDIA Q5_K/Q6_K routing after flag-ID correction (2026-09-16)

RTX 5070, wave32, DXC 1.10.2605.37, Agility 1.721.3-preview. Integrated `fca93cf51` and `b963d707c` from `0c060bbb0`. The corrected LinAlg ranges are Q5_K 219-222 and Q6_K 223-226, not 119-122/123-126. The old check failed to recognize these routes, so its effective NVIDIA default was MMQ for N>=256, including wide Q6_K. This is a routing change, not only a comment fix.

Serial A/B/B/A measurements used `DX12_NO_AUTOTUNE=1` and `DX12_TUNE_REFRESH=1` in every child process, with other DX12 overrides unset, ten seconds of idle before each process, and no concurrent build. Operator timing uses the existing `test-backend-ops perf --test-file` runner at M=512. Route/correctness profiles are separate from timing. N means output channels; K means reduction width.

| Format, N, K | Old HEAD, us | Incoming, us | Incoming latency change |
| --- | ---: | ---: | ---: |
| Q5_K, 128, 4096 | 196.70 | 181.08 | -7.9% |
| Q5_K, 192, 4096 | 199.17 | 180.97 | -9.1% |
| Q5_K, 3072, 3072 | 390.84 | 410.76 | +5.1% |
| Q5_K, 3072, 8192 | 1239.93 | 1162.76 | -6.2% |
| Q6_K, 128, 4096 | 213.79 | 171.31 | -19.9% |
| Q6_K, 192, 4096 | 221.36 | 171.22 | -22.6% |
| Q6_K, 3072, 3072 | 393.81 | 402.39 | +2.2% |
| Q6_K, 3072, 8192 | 1110.17 | 1081.78 | -2.6% |
| Q6_K, 4096, 14336 | 2863.33 | 3522.46 | +23.0% |
| Q6_K, 9216, 3072 | 1101.05 | 1162.53 | +5.6% |
| Q6_K, 14336, 4096 | 2347.20 | 2540.60 | +8.2% |

Retain the corrected IDs and new narrow-output behavior, but restore NVIDIA's wide Q6_K MMQ crossover at N>=4096. Q5_K's incoming crossover is unchanged. A second A/B/B/A cohort, changing only that Q6_K policy on the incoming source, measured incoming/retained latencies of 3530.40/2870.05, 1164.41/1100.84, and 2543.82/2352.43 us for the last three shapes. Unchanged Q5_K controls differed by at most 0.3%. This avoids the reproducible wide-Q6_K regression without changing Intel or adding a runtime knob.

Observed default flags are 164/165 for narrow Q5_K/Q6_K, 220/224 at N=3072, and 128/129 at wide outputs after the fix. The old tiny-N flags were 221/225. Keep LinAlg for the mid-width band despite its small square-matrix losses: Phi-3's long-K projections and whole-model measurements favor the incoming policy there. This is not a claim that every mid-width shape is optimal.

Model timing uses cached GGUFs, `-p 6144 -n 0 -b 2048 -ub 512 -ngl 99 -fa on -dev DX120 -r 5 --delay 2`. Each entry is the mean of two five-sample process means:

| Matched cohort | Model | A, tokens/s | B, tokens/s | B/A change |
| --- | --- | ---: | ---: | ---: |
| Old HEAD / incoming | Phi-3-mini Q4_K_M | 2380.37 | 2404.74 | +1.02% |
| Old HEAD / incoming | Phi-3-mini Q5_K_M | 2286.73 | 2318.39 | +1.38% |
| Old HEAD / incoming | Qwen3-4B Q4_K_M | 2080.08 | 2082.90 | +0.14% |
| Old HEAD / incoming | Qwen3-4B Q8_0 control | 2240.88 | 2242.23 | +0.06% |
| Incoming with old Q5/Q6 policy / incoming | Phi-3-mini Q4_K_M | 2382.19 | 2403.24 | +0.88% |
| Incoming with old Q5/Q6 policy / incoming | Phi-3-mini Q5_K_M | 2289.00 | 2315.71 | +1.17% |
| Incoming / retained wide-Q6 policy | Phi-3-mini Q5_K_M | 2314.07 | 2321.54 | +0.32% |
| Incoming / retained wide-Q6 policy | Phi-3-mini Q4_K_M | 2404.03 | 2403.02 | -0.04% |
| Incoming / retained wide-Q6 policy | Qwen3-4B Q4_K_M | 2080.34 | 2081.10 | +0.04% |
| Incoming / retained wide-Q6 policy | Qwen3-4B Q8_0 control | 2241.58 | 2240.46 | -0.05% |

The isolated old-policy DLL changes only NVIDIA Q5_K/Q6_K routing; no `DX12_MMQ_MIN_N` override is used. Qwen3-0.6B Q4_K_M, BF16 and SmolLM2 Q4_K_M were additional controls; their small differences and SmolLM2's drifting endpoint are not speedup claims. Do not pool absolute rates across cohorts. Qwen3-4B's measured Q6_K projections use N=1024/2560 and remain flag 165, so its neutral end-to-end result does not contradict the wide synthetic regression.

Final correctness: 30/30 existing Q5_K/Q6_K MUL_MAT fixtures plus 18/18 explicit shape fixtures through the existing test-file loader, including narrow, mid-width and wide outputs. DX12 and Vulkan targeted Release builds passed without changing BoringSSL configuration. No B390 hardware was tested and no scalar-Q-cache route-test coverage changes are included.

## Grouped packed loads in generic LinAlg (2026-09-16)

Stage 2 retains only grouped Q4_0/Q4_1 nibble loads in `mul_mat_linalg_f16.hlsl`. Q6_K already uses `la_q8_quads` and `LA_B_STEP`; Q8_0, Q4_K, Q5_K and MXFP4 also already have packed fetch loops. None of those paths changed. The trial covered the remaining Q5_0, Q4_0, Q4_1, Q5_1 and IQ4_NL per-element loops, not a new Q6_K loader.

The retained loops reuse `la_q8_quads`, with the byte offset rounded down inside the 16-byte nibble half and each byte selected from the returned word. This also keeps two-element runs from reading a quad past the nibble plane. Scale/min arithmetic, F16 staging, F32 matrix accumulation, zero-fill, row strides and MMID addressing are unchanged. The actual NVIDIA 128x64 variant has BK=16, BN=64, eight wave32 waves and B_PER_THREAD=4; the non-NVIDIA-specific four-wave 128x64 variant has B_PER_THREAD=8 at wave32. Other emitted configurations use runs of 2, 4, 8 or 16 elements.

### Measurements and rejected formats

RTX 5070, driver 620.12, DXC 1.10.2605.37, Agility 1.721.3-preview. Baseline is the accepted stage-1 snapshot, including its Q6_K crossover. Serial A/B/B/A and reversed B/A/A/B cohorts use fixed autotune defaults (`DX12_NO_AUTOTUNE=1`, `DX12_TUNE_REFRESH=1`), ten seconds idle before each process, no concurrent compilation and separate profiling runs. Operator timing uses the existing test-file runner with M=512.

Forced same-flag measurements set `DX12_Q50_INTDOT=0`, `DX12_LINALG_MM_Q8=1`, and optionally `DX12_LINALG_TILE=0`. These switches are measurement controls only. The table below instead uses unchanged production routing in a reversed cohort; each latency is the mean of two process results:

| Format, N, K | Flag | Stage 1, us | Retained, us | Latency change | Throughput change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_0, 576, 576 | 236 | 45.97 | 45.99 | +0.05% | -0.05% |
| Q4_0, 576, 1536 | 236 | 117.26 | 112.41 | -4.13% | +4.31% |
| Q4_0, 1536, 576 | 234 | 57.67 | 55.84 | -3.17% | +3.28% |
| Q4_0, 3072, 3072 | 235 | 380.18 | 353.29 | -7.07% | +7.61% |
| Q4_0, 4096, 14336 | 235 | 3638.85 | 3076.70 | -15.45% | +18.27% |
| Q4_1, 576, 576 | 240 | 46.27 | 45.25 | -2.19% | +2.24% |
| Q4_1, 576, 1536 | 240 | 128.24 | 113.93 | -11.16% | +12.56% |
| Q4_1, 1536, 576 | 238 | 58.02 | 57.30 | -1.24% | +1.26% |
| Q4_1, 3072, 3072 | 239 | 446.18 | 371.30 | -16.78% | +20.17% |
| Q4_1, 4096, 14336 | 239 | 4050.41 | 3117.16 | -23.04% | +29.94% |

The preceding automatic-tile cohort reproduced the large-shape latency reductions: Q4_0 -15.62% and Q4_1 -23.25% at N=4096/K=14336. With tile 0 fixed instead, the initial trial measured Q4_0 636.58 -> 458.48 us and Q4_1 682.06 -> 629.34 us at N=K=3072. These are distinct cohorts and tile selections, not interchangeable absolute rates.

Q5_0, Q5_1 and IQ4_NL grouped-load candidates were archived and reverted with no routing changes. Their large-shape gains were mixed with repeatable small-shape losses: forced tile-0 Q5_0 N=K=576 rose 40.92 -> 42.97 us (+5.0% latency), Q5_1 rose 39.73 -> 42.24 us (+6.3%), and automatic-tile IQ4_NL rose 47.42 -> 48.78 us (+2.9%). Reverting these formats leaves their emitted shader headers byte-identical to stage 1.

Cached TinyLlama-1.1B Q4_0 provides a production-model check without forcing LinAlg. Its projections already select flags 235/236. At pp6144 (`-b 2048 -ub 512 -ngl 99 -fa on -r 5 --delay 2`), reversed B/A/A/B means were 6079.28 -> 6757.29 tokens/s: +11.15% throughput, -10.03% latency. A separate A/B/B/A repeat measured 6088.63 -> 6757.31 tokens/s: +10.98% throughput, -9.90% latency.

Cached SmolLM-135M Q5_0 remains on integer-dot flag 58, not the modified loader. Its two production cohorts changed +0.04% and -0.20% throughput; SmolLM2 Q4_K_M changed -0.42% in the retained cohort. Treat these as controls, not gains. No Q4_1 pure model was cached, no model was downloaded or converted, and no default route was broadened to expose an optimization.

### Correctness and compiled scope

The retained build recompiled all 269 variants emitted from this shared source. Exactly 48 generated shader headers changed: Q4_0/Q4_1 across six dense tile variants and two MMID tile variants, each at wave16/32/64. The other 2397 generated headers are byte-identical to stage 1. CMake, BoringSSL and the accepted stage-1 host policy are unchanged.

On RTX 5070, existing changed-format MUL_MAT fixtures passed 144/144 for each forced tile 0-3, 144/144 with production defaults, and 144/144 with the existing non-NVIDIA-specific 128x64 override. These include existing permutation/noncontiguous cases; unsupported CPU-reference cases remain excluded. Twenty additional strided-row/tail fixtures through the existing test-file loader passed for each of those five forced tile configurations. Production-route shape fixtures passed 25/25. Existing MUL_MAT_ID fixtures passed 158/158 with defaults and 158/158 with the existing tall-tile threshold forced to zero; profiles confirm Q4_0/Q4_1 flags 200 and 202.

All ten Q4 shape routes match the baseline, as do all 24 available default-route profile entries. A pre-existing missing Q5 profile entry is not counted as observed. Packed-byte/address checks cover 262144 comparisons across all byte values, nibble positions, run widths and four byte alignments, including the last aligned dword. Wave16/wave64 variants compile, but other-vendor hardware performance and full GPU numerical behavior were not measured. No precision tolerance, test source, production helper or runtime knob was added.

## NVIDIA composed F32 GEMM (2026-09-16, stage 3)

RTX 5070, driver 620.12, DXC 1.10.2605.37 and Agility 1.721.3-preview: retain the 128x64, eight-wave32 variant of the shared `mul_mat_linalg_tiled_i.hlsl` shader for F16, Q8_0, Q4_K, Q6_K and Q4_0. Each wave composes a 32x32 accumulator from native 16x16x16 F16/F16/F32 operations. BK=32 and LDS pitch=40 are inherited from the Intel implementation, but the NVIDIA branch accumulates in F32 throughout K and stores F32 directly. There is no F16 accumulation, periodic F16 drain or clipping in this branch. Integrated F32 activation conversion was already present in the generic baseline and is not a new benefit.

Flag 301 is automatic only on device ID 0x2F04 with the existing NVIDIA wave32/native-F16 capability gates, exactly 512 token columns, output N>=512 and K>=1024. Full 128x64 tiles, block/alignment/span checks, contiguous F32 activations/output and dispatch limits still apply; other cases fall back. The existing rebased-root and broadcast-batch handling is reused. `DX12_LINALG_NV_COMPOSED=0` restores the accepted stage-2 routing; `=1` opts in beyond the measured automatic shape/device-ID restriction but cannot bypass the NVIDIA capability, type or memory-safety gates. Other values fail explicitly. `DX12_LINALG_MM`, `DX12_LINALG_MM_KQ` and `DX12_LINALG_MM_Q8` remain effective. Explicit `DX12_LINALG_TILE` or `DX12_MMQ_MIN_N` suppresses automatic selection. The accepted Q6_K crossover and generic Q4_0/Q4_1 loaders remain unchanged underneath this route.

### Geometry experiments and operator results

The existing standalone probe executed native16 and composed32 wave matrices with zero mismatches. Its exact-wave capability query initially reported no support even for native16; a session-only host copy using the production backend's default-wave query fallback executed both successfully. This is a local query observation, not a universal API or driver limit. Full operator runs, not probe timing, determined retention.

All three geometries executed the same 25 full-GEMM cases: five formats, M=512, and N/K pairs 512/1024, 1536/1024, 2560/4096, 3072/8192 and 4096/14336. Serial reversed cohorts used fixed default autotune decisions (`DX12_NO_AUTOTUNE=1 DX12_TUNE_REFRESH=1`) and ten seconds idle before each process. Profiles were separate from `test-backend-ops perf` timing.

The 16-wave 128x128 variant won 20/25 comparisons but regressed F16 N=2560/K=4096 from 324.17 to 389.09 us (+20.02% latency) and Q8_0 N=512/K=1024 from 42.66 to 56.81 us (+33.18%). The four-wave 64x64 alternative won 24/25, favored some small shapes, but lost much of the medium/wide benefit; F16 N=3072/K=8192 regressed 2.76%. Neither geometry ships. Eight-wave 128x64 won all 25 comparisons in both its initial production-policy cohort and the final automatic-policy repeat.

Selected final B/A/A/B means against accepted stage 2:

| Format, N, K (M=512) | Baseline, us | Retained, us | Latency change | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| F16, 2560, 4096 | 324.21 | 300.60 | -7.28% | +7.86% |
| Q8_0, 512, 1024 | 42.81 | 32.91 | -23.12% | +30.07% |
| Q4_K, 3072, 8192 | 1145.02 | 724.25 | -36.75% | +58.10% |
| Q6_K, 4096, 14336 | 2870.23 | 1896.89 | -33.91% | +51.31% |

Across all 25 final cases, latency decreased 7.28%-57.50%. A separate forced-generic-LinAlg A/B/B/A cohort (`DX12_MMQ_MIN_N=0 DX12_LINALG_TG=0`, with the experimental composed mode explicitly selected) decreased latency 26.74%-63.41%. That cohort changes the baseline route and must not be described as an isolated Q6_K policy or loader effect. Latency change is `new/old-1`; fixed-work throughput change is `old/new-1`.

### Qualified production-model results

Cached models, pp6144, `-b 2048 -ub 512 -ngl 99 -fa on -r 5 --delay 2`; no forced composed mode on retained arms. Each cohort has two five-sample processes per arm. Columns below keep the independent cohorts separate.

| Model | A/B/B/A baseline -> retained, tokens/s | Throughput change | B/A/A/B baseline -> retained, tokens/s | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B Q4_K_M | 2080.29 -> 2515.91 | +20.94% | 2079.08 -> 2509.63 | +20.71% |
| Qwen3-4B Q8_0 | 2239.17 -> 2638.18 | +17.82% | 2237.94 -> 2635.28 | +17.75% |
| Qwen3-4B F16 | 2455.11 -> 2598.48 | +5.84% | 2451.92 -> 2597.78 | +5.95% |
| Phi-3-mini Q4_K_M | 2402.42 -> 2854.18 | +18.80% | 2401.87 -> 2848.81 | +18.61% |
| TinyLlama-1.1B Q4_0 | 6738.52 -> 8741.57 | +29.73% | 6725.07 -> 8745.38 | +30.04% |
| Qwen3-0.6B Q4_K_M | 7783.80 -> 8784.42 | +12.86% | 7778.50 -> 8775.33 | +12.82% |
| Qwen3-0.6B BF16 control | 7341.77 -> 7344.29 | +0.03% | 7333.48 -> 7330.90 | -0.04% |
| SmolLM2-135M Q4_K_M control | 24118.92 -> 24245.68 | +0.53% | 24187.35 -> 24291.73 | +0.43% |

The BF16 and Smol controls have no observed flag-301 dispatches; their small changes are not kernel gains. The six improved models show 5.52%-23.10% lower fixed-work latency, not 5.84%-30.04% lower latency. The earlier broader explicit-mode TinyLlama result was faster but included smaller unqualified projections; it is not the retained default result.

### Correctness, preservation and limits

Each experimental geometry passed 25 shape and 20 padded-row/tail cases. The final build passed the existing affected-format MUL_MAT selection (446/446), padded-row/tail cases (20/20), and batched/broadcast/strided-activation cases (15/15), separately with automatic, forced and disabled composed selection. Automatic and disabled shape profiles each passed 25/25 numerical cases. All 25 automatic shape flags are 301. Batched profiles show two composed cases and one strided-activation fallback per format. Global/format kill switches and explicit tile/MMQ overrides also passed 25/25 numerical cases each.

Only five new wave32 headers are included in the final build; their DXIL is identical to the measured eight-wave experiment. All 44 existing Intel tiled headers remain byte-identical, as does the accepted generic stage-2 shader. Existing Intel behavior, MMID, bias-fusion handling, accumulation tolerances and unrelated B390 fixtures were not changed. Other NVIDIA devices can explicitly opt in but are not performance-qualified; other-vendor GPU execution and performance were not tested here. No claim is made for decode or other automatic microbatch sizes.

The disabled-route operator profile contains 22 usable entries: three wide Q8_0 cases emit zero/missing profiler timestamps. Gate profiles have similar omissions on some fallback paths. These are not counted as observed flags or used for timing. Raw commands, exact model paths, full operator tables, numerical logs, source patches and both experimental/final binaries are preserved in the session artifact directory `stage3-20260916`.

### Permanent route coverage (2026-09-16)

The existing `test-backend-ops -o DX12_ROUTES` fixture now independently admits qualified NVIDIA hardware. Its 23 CPU-reference cases cover all five retained formats at N=M=512/K=1024 with default flag 301, composed/global/per-format opt-outs, a Q6_K weight-only strided view at byte offset 420 with contiguous activations, and automatic M/N/K boundary fallbacks. Captures require actual dispatch flags, and an admitted fixture cannot pass with zero executed cases. Environment refresh remains active during the fixture; saved variables are restored and compared afterward. The unfiltered default numerical suite already invokes this fixture.

RTX 5070 passed 23/23 both with ordinary defaults and with inherited opt-outs that the fixture clears and restores. Existing default-mode MUL_MAT/MUL_MAT_ID tests for F16, Q8_0, Q4_K, Q6_K, Q4_0 and the retained stage-2 Q4_1 loader passed 879/879. Unsupported local Intel skipped cleanly. Scalar Q-cache flags 296/298 are now queried outside the preview guard and can independently admit the existing B390 fixture; the full host translation unit compiled without `GGML_DX12_LINALG_PREVIEW`, but B390 hardware was not available. Production routing, shaders and generated shader headers are unchanged, so performance cohorts were not repeated. Follow-up evidence is archived in `stage3-coverage-20260916`.

## RDNA4 selective composed F32 GEMM (2026-09-16)

RX 9070 XT, driver 32.0.23041.2023, DXC 1.10.2605.37 and Agility 1.721.3-preview: flag 302 ports the eight-wave32, 128-output-channel x 64-token composed kernel to Q4_K and Q6_K. It reuses the packed dequantization, BK=32, LDS pitch=40, rebased roots and broadcast-batch handling. F16 operands feed F32 accumulation throughout K. F32 activation conversion was already integrated in the baseline; this change does not remove a separate conversion pass.

### Output stage and routing

The unchanged NVIDIA direct-output variant failed all 25 full-GEMM numerical fixtures on this AMD device at both wave32 and wave64. Small standalone native16/composed32 direct-store probes passed, so they were insufficient qualification. Changing the wave-index calculation did not fix the full kernel. The retained AMD output stage instead serializes the eight wave owners through one shared 32x32 F32 tile, with group barriers around each store and scalar F32 drain. This stays below 32 KiB LDS without F16 accumulation, clipping or a whole-weight cache. `VP_LDS_OUTPUT` defaults to zero; existing Intel and NVIDIA output paths are unchanged. This is a workaround for the observed full-kernel behavior, not a claim that all AMD direct matrix stores are unsupported.

Automatic eligibility requires an RDNA4+ discrete GPU, LinAlg and FP16 support, and an exact native 16x16x16 F16/F16/F32 wave32 capability query. There is no default-wave fallback for that query. The measured shape region is exactly M=512 tokens, N=2048..4096 output channels and K=4096..16384. Full 128x64 tiles, quant-block alignment, aligned/even weight offsets and strides, contiguous F32 activations/output, signed span limits and dispatch limits remain mandatory. Other shapes and formats retain their previous routing.

`DX12_LINALG_AMD_COMPOSED=0` restores the previous route. Setting it to 1 does not bypass the hardware, shape or layout gates. Global/per-format LinAlg opt-outs remain effective; explicit `DX12_LINALG_TILE` or any `DX12_MMQ_MIN_M/N/K` override suppresses this path. Q8_0, small matrices, decode, MMID, NVIDIA's Q6_K crossover and B390's attention-cache default are not broadened. Earlier blanket AMD experiments regressed some Q8_0 cases by 3%-6% and small shapes by 25%-40%, which is why those cases are excluded.

### Measurements

Baseline is `b8a374b96`, including the incoming generic Q4_0/Q4_1 loader improvements. Serial DLL-swapped A/B/B/A comparisons use `DX12_NO_AUTOTUNE=1`, `DX12_TUNE_REFRESH=1`, ten seconds idle before each process, and separate profiling runs. A shorter operator repeat resolves noisy initial cases. Selected M=512 means:

| Format, N, K | Baseline, us | Retained, us | Throughput change | Cohort |
| --- | ---: | ---: | ---: | --- |
| Q4_K, 2048, 4096 | 260.48 | 188.38 | +38.27% | Short repeat |
| Q4_K, 2560, 4096 | 315.90 | 265.21 | +19.11% | Initial |
| Q4_K, 3072, 8192 | 630.66 | 540.89 | +16.60% | Initial |
| Q4_K, 4096, 14336 | 1499.22 | 1138.18 | +31.72% | Short repeat |
| Q6_K, 2560, 9728 | 901.39 | 778.69 | +15.76% | Short repeat |
| Q6_K, 3072, 8192 | 866.07 | 706.45 | +22.60% | Initial |

The Q4_K N=4096/K=16384 corner was approximately neutral: -0.33% throughput initially and +1.89% in the repeat. Do not claim every admitted shape has a material gain. An initial Q6_K N=2560/K=9728 baseline outlier suggested +56.89%; the stable shorter repeat above supersedes that estimate.

Cached whole models use `-p 512,1024,6144 -n 128 -b 2048 -ub 512 -ngl 99 -fa on -r 5 --delay 2 -dev DX120`. Each arm has two five-sample processes. Independent reversed B/A/A/B repeats confirm the pp6144 improvements:

| Model | A/B/B/A baseline -> retained, tokens/s | Throughput change | B/A/A/B baseline -> retained, tokens/s | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B Q4_K_M | 3801.88 -> 3986.84 | +4.87% | 3835.14 -> 4030.14 | +5.08% |
| Phi-3-mini-4k Q4_K_M | 4157.69 -> 4327.07 | +4.07% | 4165.48 -> 4320.09 | +3.71% |
| Qwen3-4B Q8_0 control | 4601.90 -> 4583.99 | -0.39% | 4602.14 -> 4591.82 | -0.22% |

Across the two cohorts, Qwen Q4_K_M pp512 improved 5.11%-6.94% and pp1024 improved 7.03%-8.11%; Phi improved 3.86%-4.44% and 4.87%-5.04%, respectively. Their tg128 controls changed -0.15%..+0.21%. The initial Qwen Q8_0 pp512 change of -1.43% did not repeat (+0.29% reversed); do not treat it as a demonstrated routing regression. SmolLM2-135M Q4_K_M/Q8_0 and SmolVLM2-256M Q4_K_M text-only pp6144 controls changed +0.28%, +0.44% and +0.11%. These are noise controls, not kernel gains.

Profiles attribute Qwen's flag-302 work to Q4_K N=2560/K=4096 and Q4_K/Q6_K N=2560/K=9728. Phi uses it for Q4_K/Q6_K N=3072/K=8192. Qwen Q8_0 and SmolLM2 Q4_K_M profiles have no flag-302 entries. Gains are for 512-token microbatches, not a claim for ub1024 or arbitrary shapes.

### Correctness and scope

The final candidate passed 22/22 boundary/model-shaped F32-precision cases, 879/879 existing affected-format MUL_MAT/MUL_MAT_ID cases and 46/46 permanent AMD route assertions. Route coverage includes opt-outs, explicit overrides, dimension boundaries, unchanged formats, broadcast batches, strided activations, padded/offset Q6_K weights, F32 precision and a separate bias ADD.

Cached flag-302 decisions recheck eligibility because graph identities omit strides and offsets. A session-only public-API probe exercised dense -> strided -> dense and enabled -> disabled -> enabled graphs for both formats. Each scenario recorded 98 decision-cache hits and two rebuilds, rejected the stale composed route, and preserved numerical results. A safe generic decision may remain cached when eligibility returns. Ordinary whole-command replay was disabled for this probe; decision replay remained enabled.

Paired perplexity smoke runs used the same existing mixed repository-document corpus, 16 chunks at context 512 and ub512. This is not WikiText or a general model-quality benchmark. Qwen Q4_K_M printed 5.7135 +/- 0.26635 and Phi Q4_K_M printed 4.5122 +/- 0.18569 for both binaries; profiles confirmed the new route executed. No numerical tolerance was changed.

Preview-enabled inference targets retain BoringSSL, and the ordinary preview-disabled DX12 backend also builds. Other-vendor execution was not measured on this machine. Raw operator/model cohorts, replay source/logs and baseline/candidate DLLs are retained as `amd_port_sep16_*` session artifacts.

## RDNA4 Vulkan-gap follow-up (2026-09-17)

RX 9070 XT, driver 32.0.23041.2023, DXC 1.10.2605.37 and Agility 1.721.3-preview. Measurements use FA, F16 KV, b2048 and ub512; pp6144 is not an ub1024 result. Vulkan comparisons use the same models and batching. The remaining Vulkan advantage is real; these changes do not establish parity.

### Smol routing and expert buckets

`DX12_LINALG_Q8_SMOL=0` disables the default-on Smol Q8 selection. At 512 tokens, N576/K576, N576/K1536 and N192/K576 use existing aligned flag269 instead of the small generic tile. Explicit tile/MMQ overrides, unsupported layouts, other types and other GPUs retain their prior routing. In a same-binary ABBA cohort, Smol Q8 pp6144 averaged 43021.60 off and 46496.10 on (+8.08%). A global tile override is not equivalent: it regresses Smol Q4.

`DX12_LINALG_MMID_BUCKET=0` retains the per-workgroup routing scan. Flags303/304 consume the existing compact expert buckets with the original 32x32 or 128x64 F32-accumulating matrix arithmetic. The default is limited to discrete RDNA4 wave64, at least 128 tokens, at most 256 experts, and validated weight/activation/IDs/output layouts. Unsupported layouts retain the original matrix path.

The shared bucket prepass must upload op0 as well as the 30 base root-constant DWORDs. Uploading only the base fields leaves a stale pairs offset; CBV mode hid this defect. Bucket reuse is confined to compatible IDs resources and expert counts, invalidated by overlapping writes, and reset between graph executions. Cached route decisions recheck eligibility. Explicit in-process environment refresh disables decision replay so tuning changes are not hidden by cached routes.

Before native-load refinements, Granite Q8 pp6144 averaged 16931.94 off and 18694.52 on (+10.41%). This is matrix MMID consuming buckets, not substitution with the much slower ordinary scalar MoE GEMM.

### D128 attention

`DX12_FA_RDNA4=0` disables flag320. Its default-on scope is F16 K/V, D128, discrete RDNA4 wave64, at least 1024 keys and at least 256 unsplit query/head/batch groups. It uses Br32/Bc64, two score tiles per wave, exact zero/mixed/fully-masked classification and wave-local row reductions. QK, PV, normalization and persistent output remain F32; K retains the padded LDS transpose.

Mask classification scans the actual mask values, including broadcast dimensions and partial tiles. It does not infer causality from shape or corner values. Wave reductions preserve the cross-wave barriers needed by the P and output-correction consumers; fully masked iterations also synchronize before reusing classification storage.

Qwen3-0.6B Q8 pp6144 averaged 17069.73 with the old attention and 17813.64 with the wider candidate (+4.36%). The wider tile regressed Falcon's smaller, split-prone dispatch by about 1-2%, so the default group-count gate excludes that case. D64 and small-key dispatches remain unchanged. Wave32 and Br16/Bc64 alternatives passed numerical checks but reduced Qwen pp6144 to roughly 14.7-14.8k; neither is selected.

### F16 PV was evaluated, not enabled

Native F16/F16/F16 16x16x16 matrices executed correctly in isolated wave32 and wave64 probes. That capability did not make half-precision attention safe or faster:

- F16 PV with F32 persistent output passed the ordinary D128 cases, but overflowed for 32 keys with V=3000 even though the normalized result is only 3000. Its Qwen timing showed no gain over F32 PV.
- F16 PV with F16 persistent output also failed existing attention cases and long-context constant-value controls. At 131072 keys it produced about half the correct normalized value.
- F32 PV and F32 persistent output passed those controls, including arbitrary masks, fully masked rows, GQA, partial tiles and split/non-split execution.

The half-precision variants are retained only as session experiments, not production flags or defaults. No numerical tolerance was relaxed. Vulkan's half PV/carry is therefore not treated as a drop-in precision-equivalent optimization.

### Remaining dense experiments

Native16 weight loads were evaluated independently of aligned/full-tile bounds removal. Granite Q8 gained about 1.5% beyond bucketing. Packed Q5_0 nibble loads alone regressed Smol Q4, while packed plus native16 loads improved pp6144 from roughly 40.0k to 41.3k in forward and reversed cohorts.

The retained Q8 change enables native16 loads only in the already layout-gated AMD bucket shaders; generic dense Q8 is unchanged. Flags310-313 provide packed/native16 Q5_0 versions of the four existing dense tiles, with the same F32 accumulation and bounds checks. They require discrete RDNA4 wave64 and even weight offsets/strides; cached decisions recheck these requirements. `DX12_LINALG_Q50_PACKED=0` retains flags230-233. Other formats, layouts and vendors keep their existing shaders.

A genuine 64x64/BK32 F32-accumulating experiment changed both host geometry tables and the shader, rather than merely overscheduling a replacement shader with the old grid. It improved over forced 32x32, but remained slower than the existing automatic routes: Smol Q8 39615.71 versus about 46500, and Qwen Q8 14795.08 versus about 17800. The geometry change was rejected and both tables restored.

Raw measurements, isolated shader sources, overflow probes and DLLs are retained in session artifacts prefixed `rdna4_` and `vk_gap_sep16_`. Falcon's large BF16 model alternates between substantially different timing regimes even with the original backend; a single such result is not evidence of a new kernel regression or improvement.

### Combined delivery

Final original-DLL/installed-DLL confirmation, three repetitions per result:

| Model | Format | Original pp6144 | Installed pp6144 | Change |
|---|---|---:|---:|---:|
| SmolLM2-135M | Q8_0 | 43157.80 | 46251.93 | +7.2% |
| Qwen3-0.6B | Q8_0 | 17073.05 | 17694.30 | +3.6% |
| Granite-3.0-1B-A400M | Q8_0 | 16916.59 | 18839.06 | +11.4% |
| SmolLM2-135M | Q4_K_M | 39914.51 | 40860.02 | +2.4% |
| Falcon-H1-7B | Q8_0 | 2704.25 | 2689.20 | -0.6% |

The preceding independent final cohort measured 46634/17910/18964 tok/s for the first three Q8 models. Falcon retains its original attention route; its sub-percent variation is not a claimed optimization. F16/BF16 and Q4 controls showed the expected scope: Qwen and Granite improved, while Smol F16 and Falcon quantized routes remained effectively unchanged. Paired tg128 means were 1075.25/1075.60 for Smol, 487.60/486.35 for Qwen, 505.60/513.81 for Granite and 60.71/60.74 for Falcon.

The retained implementation passed 1232 affected dense/expert cases, 536 D128 F16 attention cases, 138 route assertions and the existing 46 composed assertions. A separate real decision-replay probe recorded 99 hits/one miss while expert IDs changed in place, with correct outputs. Root-constant and CBV bucket paths were exercised separately.

Paired whole-corpus perplexity at context2048/ub512 covered 12 chunks for Smol Q4 and Granite Q8 and 10 for Qwen Q8. Original/installed estimates were 7.9651/7.9651, 5.5477/5.5477 and 9.1599/9.1600 respectively. This is a local mixed-document smoke corpus, not a general model-quality claim.

`llama-cli`, `llama-bench` and `llama-mtmd-cli` are built with BoringSSL in both the preview-enabled `build-linalg` and preview-disabled `build-dx12` trees. Benchmark the former for these LinAlg changes.

## Range-safe FP16 PV follow-up (2026-09-17)

FP16 PV is an opt-in experiment, not a new performance default. `DX12_FA_PV_F16=1` selects it for discrete RDNA4, wave64, D128 and F16 K/V, after an exact native F16/F16/F16 16x16x16 capability query. Unset or zero preserves the previously qualified F32 paths. Flags321/322 preserve the generic/wide query and KV geometry respectively, including existing split boundaries and the `DX12_FA_RDNA4` selection policy.

QK, softmax maxima/sums, persistent output, split partials and normalization remain F32. Only the current PV matrix increment accumulates in F16. The increment is widened into the existing canonical F32 staging slots. A wave that produces a nonfinite increment recomputes that whole increment from zero in F32; widening an already overflowed result cannot recover it. A matrix-wide subnormal result is also recomputed when its P fragments contain nonzero probabilities. This protects tiny-value workloads without redoing every ordinary near-zero output element. It is still reduced-precision arithmetic, not bit-identical F32.

The range probes include V=3000, maximum finite half values (+/-65504), subnormal values, large cancellation, arbitrary masks and 131072-key distribution shifts. The original unnormalized half-output carry is not used. P/V scaling was not adopted because it can discard small but meaningful contributions before multiplication.

### Throughput remains unresolved

The corrected half paths did not produce a reliable end-to-end improvement over the delivered F32 baseline. Native register-resident matrix probes with matched operands, sustained warmup and zero mismatches measured F32 at 96-102 us and F16 at 91-98 us for the same work. They do not establish a large arithmetic advantage that can simply pay for extra range checks and data movement.

The follow-up also evaluated typed-half shared staging, reuse of dead Q/K/P storage, a one-slot F32 control, direct accumulator extraction, both 32- and 64-key tiles, and deferred probability checks. Direct-coordinate and integer-backed staging attempts failed full-shader correctness and were not retained. Data-derived coordinate calibration recovered correctness in later extraction experiments, but did not deliver a performance win and was also removed. Some overlapping late experimental timing arms were excluded; accepted comparisons serialize all GPU work.

The faster F32 defaults and all previously measured GEMM/MMID gains are preserved. Raw sources, analytical probes, sustained native-matrix measurements and candidate DLLs are retained as `fp16_pv_*` session artifacts.

The integrated on/off ABBA Qwen3-0.6B Q8 cohort (pp6144, ub512/b2048) measured F32 at 17918.96/17896.22 tok/s and safe FP16 PV at 16144.82/16132.69 tok/s, about 9.9% slower. An explicitly unsafe control without recovery reached 17071.36 tok/s and was then removed; recovery overhead is not the sole remaining cost. These results do not justify enabling FP16 PV by default.

The permanent overflow fixture uses an analytic reference: Q/K are zero, V is 3000, and the normalized result must be finite 3000. The CPU F16 attention implementation itself returns infinity for this fixture, so it cannot serve as its numerical reference. Separate analytic probes cover the full finite-half range, cancellation and long-context distribution shifts. The opt-in path also passes the existing D128 F16 attention cases without tolerance changes. The local Qwen smoke-corpus perplexity is 9.1595 with FP16 PV versus 9.1600 with the preceding F32 path; this is not a general quality guarantee.

## Resource-led D128 attention follow-up (2026-09-17)

The AMD shader-analyzer extension, hosted with the same Agility721 runtime as inference, now supplies native ISA, ELF metadata and resource statistics. Stock RGA could not create this preview pipeline. RGP connected but failed trace collection with result -2, including a single-dispatch retry; no current runtime occupancy or stall-counter claim is made.

Both original wide kernels allocated 192 VGPRs and 29696 bytes of LDS, with zero scratch. Live-register analysis placed the peak at PV matrix work, not at the F16 accumulator alone. The final compact kernels retain Br32/Bc64, four wave64s, padded LDS-transposed K and F32 persistent output:

| Wide kernel | Driver-reported VGPRs | Hardware-allocated VGPRs | Peak live VGPRs | Allocated LDS bytes | Scratch bytes |
|---|---:|---:|---:|---:|---:|
| Original F32 | 192 | 192 | 181 | 29696 | 0 |
| Compact F32 | 160 | 168 | 143 | 25600 | 0 |
| Original safe FP16 PV | 192 | 192 | 181 | 29696 | 0 |
| Compact safe FP16 PV | 157 | 168 | 141 | 25600 | 0 |

Flags323/324 specialize the previously eligible wide RDNA4 F32/FP16 paths for complete 64-key F16 tiles, D128 and 32-byte-aligned K/V offsets and outer strides. V loads directly into matrix fragments, so Q and P staging can reuse the dead K allocation without another barrier. Removing generic V staging also removes its overlapping register lifetimes. Query tails, GQA, batches, masks, sinks and softcap remain supported. KV tails and misaligned layouts retain flags320/322; the smaller-dispatch, other-head-dimension and other-hardware routes are unchanged. Decision replay revalidates the compact layout when bindings change.

`DX12_FA_COMPACT=0` disables this layout specialization. `DX12_FA_PV_F16=1` remains required for half accumulation. Independently of layout specialization, FP16 PV now retains the values already read for its range check and uses them for the output update. Only an actual F32 repair reloads the staging slot. This removes a redundant LDS read without removing either the overflow or whole-matrix underflow recovery.

K prefetch remains enabled. Removing it reduced compact F32 allocation further to 144 VGPRs but regressed Qwen pp6144 from about 18124 to 16881 tok/s. The ELF reports CU mode, and the lower resource counts alone did not establish a residency benefit. Dynamically looping over PV output tiles introduced 64 bytes of scratch and was rejected. Neither experiment remains in production.

Final uncontended binary ABBA comparison against the saved start-of-follow-up DLL, Qwen3-0.6B Q8_0, pp6144/ub512/b2048, F16 K/V, three repetitions per arm:

| Accumulation | Original arms, tok/s | Final arms, tok/s | Mean change |
|---|---|---|---:|
| F32 | 18071.65 / 18063.51 | 18679.80 / 18764.24 | +3.6% |
| Safe FP16 PV | 16258.39 / 16202.63 | 17796.82 / 17816.25 | +9.7% |

The cached-value change alone improved compact FP16 from 16866.83/16914.19 to 17896.30/17878.75 tok/s in a separate binary ABBA run. FP16 is still about 5% slower than the final F32 path, so F32 remains the default. Short pp512 means were effectively unchanged. Single paired F32 format controls measured Qwen BF16 at 16863.97/17504.73 and Q4_K_M at 15510.15/16020.31 tok/s; those are corroborating pairs, not additional ABBA cohorts.

The final paths pass 536 D128 F16 attention cases in each accumulation mode and 159 route assertions plus 46 composed assertions in both root-constant and CBV modes. Permanent route coverage includes aligned tiles, KV/query tails, misaligned K/V views, GQA/batches/sinks/softcap, opt-outs and analytic FP16 overflow recovery on both generic and compact paths. Real decision replay records 99 hits/one miss, falls back correctly when V moves by two bytes, and restores the compact route after alignment is restored. The FP16 cache change also passes the existing analytical finite-half, subnormal, cancellation and 131072-key controls without tolerance changes.

Paired local-corpus Qwen perplexity is 9.1600 for both original and final F32 and 9.1595 for final FP16. This is a smoke corpus, not a general quality guarantee. Raw native analysis, benchmark CSVs, quality logs and rejected variants are session artifacts named `fa_resources_*` and `vk_gap_sep16_fa_*`. The requested inference targets remain built with BoringSSL in `build-linalg` and `build-dx12`; use `build-linalg` for these optimizations.

## Vulkan attention attribution (2026-09-17)

Matched controls separate the output precision policy from mask processing and data layout. All measurements below use the RX 9070 XT, Qwen3-0.6B Q8_0, pp6144, ub512/b2048 and F16 K/V unless stated otherwise. The discrete adapter enumerated as DX120 and Vulkan1 during this experiment; Vulkan0 was the integrated GPU. GPU workloads were serialized. Diagnostic Vulkan DLLs did not change the installed Vulkan policy.

### Precision and mask counterfactuals

Each mean below combines two bracketed ABBA arms, with four repetitions per arm:

| Vulkan policy | Original, tok/s | Control, tok/s | Control change |
|---|---:|---:|---:|
| F32 PV, F32 staging/carry, no maximum offset | 23150.83 | 22444.51 | -3.1% |
| Original precision, mask prepass disabled | 23135.66 | 22509.76 | -2.7% |

These effects are not additive. The first control still uses F16 matrix operands; it does not make the whole attention operation F32. A separate screening control that changed only persistent carry to F32 was within one percent of original throughput. Output precision alone therefore does not explain the remaining DX12 gap.

Instrumented attention totals across the twelve 512-query microbatches were 122.091 ms for original Vulkan, 125.635 ms for its F32-output/no-offset control, and 128.391 ms without its mask prepass. These are separate profiled runs, not the uninstrumented ABBA measurements. Driver-reported masked-pipeline resources were 81 VGPRs/15872 bytes LDS for original Vulkan and 89 VGPRs/17920 bytes LDS for its F32-output control, with no scratch. Vulkan used Br16/Bc64 and four wave64s; DX12 compact attention used Br32/Bc64 and four wave64s. Static resource counts are not measured occupancy.

A DX12 exact mask-classification prepass was also implemented and measured independently of precision. It scanned each distinct mask tile once per FA operation, reused the existing unsplit scratch binding, and skipped all-zero loads and all-negative-infinity K/V tiles. Combined prepass/consumer/barrier wall time lost: 18523.06 versus 18774.00 tok/s (-1.3%). Its consumer reduced reported VGPRs to 138 but retained 25600 allocated LDS bytes. The prototype was removed, not enabled by default or left as an unqualified opt-in. This result does not rule out a differently packed or safely cached prepass, but does not justify adding that complexity without another measured benefit.

### Scaled half carry is not range-safe

An independent double-precision stable-softmax reference exposed failures in the actual original Vulkan shader, not only in a DX12 approximation of it. With the maximum offset retained, constant V=3000 returned 65504; alternating large positive/negative blocks also clamped to 65504 instead of zero. At 131072 keys, constant V=3 returned 3.666015625, and a small-update fixture returned 1 instead of 1.046875. Clamping can conceal intermediate overflow, so a finite final result alone is not a correctness criterion.

Changing Vulkan carry to F32 fixed the constant-range and long-carry cases, but not cancellation. F32 PV with F16 staging and F32 carry fixed cancellation. The positive maximum offset still damaged a directed tiny-probability case; F32 output state with zero offset passed all 17 analytical cases. The corresponding DX12 scaled-half-carry controls reproduced the relevant range/long-state failures and were rejected. No scaled half carry, half final normalization, or final clamp was added to DX12.

The same probe exposed a pre-existing tiny-probability conversion limitation in DX12's compact F32 and safe-FP16 paths: 0.000304767367 versus the analytical 0.000363725471. Native packed conversion truncates these probabilities before PV; retaining F32 output cannot recover the lost mass. Explicit integer round-to-nearest-even reduced the error sufficiently to pass the case, but raised VGPRs to 192 and reduced throughput to about 16652 tok/s. That costly implementation was not retained. The earlier range tests did not establish accuracy for every probability distribution.

The delivered compact paths instead construct half-subnormal bits with `round(p * 2^24)`, using a bounded unconditional calculation and integer selection. This avoids the extra divergent branches and register pressure of the first implementation. The denominator still sums the original F32 probabilities; normal probabilities retain the existing conversion. Both compact accumulation modes now return 0.000365720858 on the directed case, NMSE 3.01e-5, within the unchanged 5e-4 threshold. This is rounded half input, not exact full-F32 attention. Generic and non-RDNA4 routes are unchanged.

### Retained data-layout improvement

Compact F32 attention now assigns each lane four adjacent output columns within one row, rather than one column across four rows. Persistent output, softmax correction, sinks, normalization and stores use the same mapping. Matrix arithmetic, padded AMD K transpose, prefetch, geometry, routing gates and precision remain unchanged. Other shader variants retain the original mapping; in particular, the analogous safe-FP16 change regressed about 1.5% and was rejected.

The production-build ABBA measured 19450.61/19452.06 versus 18726.27/18731.99 tok/s, a 3.9% improvement over the start-of-attribution compact F32 baseline. A separate isolated-DLL ABBA measured +3.8%. Profiled DX12 attention time fell from 169.585 to 154.914 ms across the twelve microbatches, about 8.7%, while the whole model improved less. Qwen3-4B Q8_0 improved from a mean 4732.56 to 4844.18 tok/s (+2.4%). Falcon, Smol and Granite retain their existing routes and showed no material change. Short pp512 remained unchanged. Single paired BF16 and Q4_K_M Qwen controls corroborated the improvement but are not additional ABBA cohorts.

Native output-update inspection shows why the mapping helps: 16 scalar PV reloads become four 128-bit LDS loads, and 16 correction reloads become four scalar LDS loads. The 16 carry FMAs remain. The compiler also consumes the first PV tile earlier instead of retaining four output-tile chains until all sixteen WMMAs finish. Matrix instruction counts, group barriers and scalar global-store counts are unchanged. Used VGPRs fall from 160 to 157 and static peak live from 143 to 141; this is not evidence of an occupancy gain.

### Final combined result

Correct subnormal rounding costs throughput even without additional VGPR allocation: in a separate ABBA against layout-only code, compact F32 measured 18995.29 versus 19535.37 tok/s, and safe FP16 measured 17718.07 versus 17873.19 tok/s. A cheaper precise-add rounding control also passed the analytical cases but did not materially improve throughput, so the clearer explicit-round implementation was retained.

Final production-DLL ABBA against the start-of-attribution baseline, including both the output mapping and probability fix:

| Model, Q8_0 | Baseline pp6144 | Final pp6144 | Net change |
|---|---:|---:|---:|
| Qwen3-0.6B | 18756.32 | 18963.89 | +1.1% |
| Qwen3-4B | 4755.93 | 4797.19 | +0.9% |

The earlier +3.9%/+2.4% figures isolate the output mapping; they are not the final net gains after correcting the pre-existing numerical issue. F32 remains faster than safe FP16.

The combined implementation passed all 17 analytical cases in both accumulation modes, 536 existing D128 F16 attention cases in each mode, 161 route assertions and 46 composed assertions in root-constant and CBV modes, and real decision replay with alignment changes. The permanent route fixtures now include the tiny-probability regression in both compact modes. Ten-chunk local Qwen perplexity was 9.1600 before this work, 9.1602 with final F32, and 9.1596 with final safe FP16; these tiny changes are not a general model-quality claim.

Vulkan's original policy is still faster: roughly 23150 versus 18964 tok/s in these cohorts. Even its F32-output/no-offset control is faster. The useful changes here are cheaper output data handling and corrected compact-path subnormal conversion, not proof that FP16 PV is intrinsically faster or that the remaining gap has been closed. F32 remains the DX12 default. Diagnostic sources, DLLs, analytical results, native resources, profiles and balanced timings are retained as `fa_attr_*` session artifacts.

## RDNA4 compact F32 register-owned rows (2026-09-17)

The next pass retains only a register-owned softmax/output path in flag 323. Its gate and Br32/Bc64 geometry are unchanged. The padded AMD K transpose, K prefetch, direct V loads, explicit subnormal probability rounding, and original F32 probability sums remain intact. Flag 324 and all noncompact shaders retain their previous dataflow.

Each wave owns eight softmax rows, with eight lanes per row. A lane now carries that same row through maximum, denominator, correction, F32 output accumulation, sinks and final stores. Each lane holds two adjacent columns from each of the eight output column tiles. Matrix outputs still use canonical LDS staging; no matrix-coordinate assumptions or duplicated softmax work are needed.

The PV producer layout remains two waves per 16-row block. Producers write two matrix tiles per phase, synchronize, and consumers read the appropriate row from either producer's canonical slots. A second barrier protects slot reuse. There are two phases. This trades cross-wave PV handoff for removal of the repeated `s_gmax`, `s_gsum` and `s_corr` exchanges, without increasing the PV staging allocation.

### Balanced results against the saved current baseline

RX 9070 XT / DX120, Q8_0 models, F16 K/V, `-p 512,6144 -n 0 -ub 512 -b 2048 -ngl 99 -fa on -r 4 --delay 1`. Each pair uses ABBA binary order with eight timed samples per arm, except the initial tile comparison, which used ABC-CBA. GPU workloads were serialized; no compilation overlapped timing.

| Model | Baseline pp6144 | Register-owned F32 pp6144 | Change |
|---|---:|---:|---:|
| Qwen3-0.6B, final production confirmation | 18783.44 | 19083.33 | +1.60% |
| Qwen3-4B | 4744.63 | 4788.01 | +0.91% |
| SmolLM2-135M control | 46523.40 | 46469.44 | -0.12% |
| Granite-3.0-1B-A400M control | 18903.04 | 18895.34 | -0.04% |
| Falcon-H1-7B control | 2686.60 | 2686.13 | -0.02% |

The initial three-arm cohort measured 18816.76 baseline versus 19137.19 Br32 (+1.70%). The Qwen3-4B experiment and final production use byte-identical compact F32 CSOs. The pp512 route is outside the compact gate; its noisy changes are not attributed to this optimization. These are incremental gains over the already-corrected prior baseline, not over the historical 18756 tok/s baseline. No Vulkan parity is claimed.

### Rejected experiments

- Br16/Bc64 was implemented in the same register-owned dataflow, not by changing the tile define alone: four rows per wave, 16 lanes per row, one PV handoff phase, and a matching 16-row host dispatch. It passed all 17 analytical cases but measured 16150.03 tok/s versus Br32's 19137.19 in the balanced tile cohort. Its 141 requested VGPRs did not compensate for the smaller query tile; it retained 24848 bytes of LDS and doubled the query-group count.
- Diagnostic-only shader counters measured the actual positive F32 probabilities and wave-level subnormal decisions during Qwen3-0.6B pp6144. Across 616 compact dispatches from warmup and timed prefill, 10372246937 of 16796516352 positive probabilities (61.75%) were below 2^-14. A subnormal was present in 32057617 of 33116160 executed wave/tile softmax passes (96.80%). At 6144 keys alone, those rates were 67.60% and 97.97%. Counts exclude fully skipped tiles and zero probabilities; positive probabilities that round to half zero are included. Instrumentation preserved model outputs and was excluded from timing comparisons.
- The wave-uniform ordinary-conversion fast path retained the corrected RNE fallback and F32 sums and passed all 17 cases. It measured 18742.68 tok/s versus 19114.92 for unconditional corrected conversion (-1.95%). With only 3.20% of measured waves eligible for the fast path, the added vote/branch did not pay off.
- Safe FP16 PV in the redesigned dataflow preserved F32 persistent carry, overflow recovery and whole-matrix underflow recovery. All 17 analytical cases passed, including the recovery fixtures. It measured 17524.77 tok/s versus improved F32's 19114.92 (-8.32%). The previous opt-in half shader remains unchanged and is byte-identical to its saved baseline CSO.

Only the Br32 F32 redesign ships. The Br16 geometry, alternate host dispatch, probability counters, wave-fast branch and redesigned half path are confined to the session experiment archive.

### Native evidence and final qualification

Matched AMD analysis reports 157 -> 161 requested VGPRs, 67 SGPRs in both, and 25232 -> 24848 bytes of LDS; allocated LDS per group changes from 25600 to 25088 bytes. Scratch remains zero and all 32 static matrix instructions remain F32 WMMAs. Static group barrier signal/wait counts rise from 24/24 to 26/26. This is a dataflow improvement despite extra barriers, not evidence of improved occupancy.

The rebuilt production DLL passed 17 independent analytical cases in each PV mode, all 536 existing D128/F16 attention cases in each mode, and 161 route assertions plus 46 composed assertions in each of root-constant and CBV modes. Replay measured 99 hits/1 miss and correctly fell back from 323 to 320 after moving V by two bytes, then returned to 323 after realignment; half mode retained the corresponding 324/322 behavior. Ten actual chunks of the existing 2048-context Qwen corpus measured PPL 9.1602 for the saved baseline and 9.1599 for final F32, within the reported +/-0.2813 uncertainty.

Production shader headers were regenerated from source and the preview `llama-cli`, `llama-bench`, `llama-mtmd-cli`, and `test-backend-ops` targets rebuilt successfully. All three user-facing tools passed startup checks. No CMake or host dispatch changes were added by this pass.

Artifacts are in the session's `wave-owned-lab` archive, outside the Git worktree: preserved baseline and experimental DLLs, shader sources/CSOs/native analyses, balanced benchmark logs, probability counters, quality checks, replay logs and production qualification logs. `production_shader.patch` isolates this pass from earlier dirty changes. The installed `build-linalg\bin\ggml-dx12.dll` SHA256 is `BCFE347A6394217B1586CC22E2C187FD968BAA635A73E87BF36DC77150FA73E9`; its compact F32 CSO SHA256 is `D3F84D1EF698CED89995ED61E8D6BB4621B1E118DE7BE7643AE683407BA822E4`.

### Independent-review controls

The Br16 result above is conditional on the retained K-staging and PV-producer architecture, not a general rejection of Br16 or larger dataflow changes. An independent review requested two additional controls before drawing architectural conclusions. Both were implemented in isolated DLLs against the newly qualified production baseline. Section 10i already records correct but performance-neutral Q prescaling on an earlier dataflow; the new control is an interaction recheck, not a novel hypothesis.

Moving the scale multiply to F32 Q before F16 conversion, and removing it from each score, passed the original 17 analytical cases and 536 attention cases. It improved Qwen3-0.6B by 0.43% in an initial balanced cohort and 0.35% in a repeated ABBA with 16 timed samples per arm (18900.52 -> 18966.50 tok/s). Qwen3-4B measured +0.23% (4793.65 -> 4804.84) with appreciable drift between the first and last baseline runs.

However, prescaling changes the input precision boundary. A further analytical fixture uses Q=2^-23, K=0 for the first 512 keys and 65504 for the next 512, and corresponding V=0/1. With the usual D128 scale, qualified production returns 0.522071898 against reference 0.522071943, while prescaling returns 0.5 (NMSE 0.001787, above the existing 0.0005 tolerance): scaling Q before conversion rounds a previously representable input to half zero. The current original-policy Vulkan dGPU also returns 0.5 on this fixture. Thus this is a precision tradeoff, not a free removal of repeated work. The unguarded control is not promoted; a future guarded prescale or scale policy needs separate qualification.

The D128 transposed-QK control computes K times Q-transpose and stores the scores transposed. Direct row-major global K loads through `MatA::Load` failed the tiny-probability fixture (0.869894207 versus reference 0.000363725471) and two existing random compact fixtures, leaving 534/536 passing. Reducing the declared alignment from 32 to 16 bytes gave the same failure. The same orientation with K staged in natural row-major LDS passed all 17 analytical and 536 existing cases, isolating the failure to the direct-load variant rather than proving the score-transpose or consumer layout invalid. The correct staged control measured 15829.80 versus production's 19162.56 tok/s (-17.39%). Incorrect direct-load variants were not benchmarked or promoted. A compiler/driver defect is not established by this isolation alone.

Independent review reconciled the earlier `fa_attr_dx_qkt`/`qkt_manual` CSOs and DLLs: they already used direct row-major descriptor K loads as MatA with 32-byte alignment. The first used ColMajor LDS Q/MatB loads and accumulator stores; the manual control transposed Q in LDS and used RowMajor MatB loads/stores with a matching score index. Both were 256-thread, 12688-byte LDS shaders, each embedded exactly once in its corresponding DLL, and both failed the asymmetric tiny-probability fixture at approximately 0.869886 versus 0.000363725 in F32 and half variants. The follow-up above is therefore a recheck, not a new architecture test. Retaining the padded K path is justified for this ABI/driver. Only a smaller isolated descriptor-MatA correctness probe or materially different API/driver evidence should reopen direct K.

These controls do not exhaust the larger design space: Br16 with smaller staging, different matrix producer ownership, or a correct direct-K implementation remains distinct from the tested Br16 configuration. Earlier Qwen3-4B profiling found GEMM parity, while more recent Qwen3-0.6B attribution profiles suggest gaps at several GEMM shapes. Those observations concern different workloads and need a fresh controlled audit before a general conclusion; attention-only results cannot establish backend parity. Additional sources, native analysis, balanced logs and the new boundary probe are archived under `wave-owned-lab\review*`.

Device enumeration was rechecked for the boundary comparison: `build-vulkan\bin\llama-bench.exe --list-devices` currently reports Vulkan0 as RX 9070 XT and Vulkan1 as the integrated AMD Graphics. Historical Vulkan1=dGPU instructions must not be reused without checking. `vulkan_dgpu_tiny_query.txt` records the confirmed index-0 run; `vulkan_tiny_query.txt` is the separate initial integrated-GPU run.

### Independent parity review: remaining scope

GPT-5.6 Sol reviewed the plan, implementation and results. The small qualified improvements do not establish that attention-only tuning can close backend parity. The safe-output Vulkan control still differs in Q scaling and in whether rounded P feeds the denominator, so the remaining difference cannot yet be assigned entirely to API or code generation. Current Qwen3-0.6B GEMM shapes also need a fresh matched audit rather than inheriting the earlier Qwen3-4B conclusion.

The retained cross-wave PV handoff is internally consistent and resembles Vulkan's high-level handoff; a fully wave-local PV rewrite is not an obvious missing step. A distinct unmeasured attention hypothesis is wave-local K staging: each compact QK wave consumes its own 16-key block, but current K staging is group-striped and requires a group rendezvous. Remapping load and prefetch ownership could first isolate that rendezvous cost while preserving the proven padded transpose. A subsequent bounded 64-D wave-private slab control could test smaller K storage, at the cost of additional phases. Neither is a measured gain or a reason to drop necessary memory ordering.

The rejected mask prototype also differs from a graph-scoped packed prepass: it used a DWORD and a workgroup per tile, recomputed before each FA consumer, and paid repeated synchronization. Reuse across compatible consumers and Vulkan-style packing remain untested; resource identity, offsets, spans, strides, broadcasts, geometry and intervening writes must all participate in validity checks. These are reasons to consider a bounded kernel/dataflow refactor, not evidence that a wholesale backend rewrite is necessary.

## Vulkan pipeline replacement experiments (September 17, 2026)

The qualified baseline was preserved in commit `4f1fa66d38bfb11cf261452f720675ace9202580` on `dx12-linalg-phase0`. Replacement work is on `dx12-vulkan-parity`. `DX12_FA_PIPELINE=1` opts into a separate D128 Br16/Bc64 pipeline on the RDNA4 discrete GPU; it does not change the default or replace the existing safe-half path. `DX12_FA_LINALG=0` and `DX12_FA_PV_F16=1` disable this experiment.

### Graph-reused mask metadata

The producer packs 16 classifications into each DWORD, with two bits per Br16/Bc64 tile: mixed, all negative infinity, or all zero. Signed zero counts as zero; finite biases, positive infinity and NaNs remain mixed. Metadata uses the actual query count, supports mask head/batch broadcasting, and skips padded query rows. The consumer retains the previous infinity suppression policy, which is distinct from exact producer classification.

The cache is local to one graph execution and keyed by resource, byte offset/span, shape, strides, type and attention geometry. Existing write tracking invalidates overlapping entries, including fused outputs. A dedicated per-context arena provides bounded storage, orders producer/consumer accesses, and submits and drains an open command list before growth can release the old resource. Whole-command-list replay is bypassed for candidate graphs; ordinary decision replay remains available because late attention selection is recomputed.

Whole-graph probes demonstrate one producer for two consumers, invalidation after an in-place mask write, and zero/negative-infinity/zero refresh across executions. A real Qwen graph reuses one producer across 28 attention consumers. The scratch-growth probe submits 302 consumers across small/large/small graphs without intermediate readback; the large graph builds once and reuses 299 times. These are pipeline validity results, not independent speedup claims.

### Direct K and literal matrix strides

The initial wave-private staged-K replacement passed all 18 analytical and 536 D128/F16 cases but regressed Qwen3-0.6B Q8_0 pp6144 from approximately 19180 to 12994 tok/s. Smaller LDS alone did not make that architecture faster.

Independent descriptor probes then passed RowMajor and ColMajor loads for both matrix roles, asymmetric supported 16x16x16 products, padded strides/offsets, wave32/wave64, and preloaded-Q multiwave patterns. These invalidate a generic descriptor-load prohibition. In the complete attention shader, runtime-stride K loads still produced incorrect canonical QK scores: the first tile accumulated one or fifteen products instead of 128 in the asymmetric tiny-probability case. Replacing only that stride with the matching compile-time constant corrected the scores.

The direct path therefore uses natural Q as MatA, descriptor K as ColMajor MatB, eight supported 16x16x16 QK steps, and exact host-gated stride specializations. Flag 325 requires 256 bytes; flag 327 requires 2048 bytes, including the interleaved Qwen layout. Both preserve F32 scores, maxima, denominator, output carry and normalization, plus the qualified subnormal-P conversion. Interleaved tiny-query, tiny-probability and mixed-mask/sink/softcap references pass with flag 327. The full-shader lowering dependency is not evidence that every dynamic-stride descriptor load fails.

Caching eight Q fragments outside the key loop removes repeated LDS loads while keeping 10240 bytes of LDS and zero scratch. Requested VGPRs increase from approximately 55 to 68. This is a DX12 experiment, not a claim that Vulkan's GLSL explicitly caches Q across its key loop. Balanced Qwen3-0.6B pp6144 measured baseline 19236.82/18942.50 versus Q-cache 19380.83/19431.45 tok/s. Qwen3-4B measured baseline 4789.25/4737.66 versus Q-cache 4830.97/4818.76. These small gains do not meet the gap-closure goal.

An earlier cohort labeled direct K actually selected fallback flag 323 because its literal-256 eligibility gate excluded Qwen. It is not direct-kernel performance evidence. Subsequent candidate measurements must confirm the executed flag, not merely the environment variable.

### Arithmetic-matched Vulkan and GEMM attribution

An isolated Vulkan control preserves unscaled RTZ-rounded Q, post-QK scaling, zero softmax maximum offset, original F32 denominator probabilities, normal-P RTZ/subnormal-P RNE conversion, and F32 PV/staging/carry/normalization without clamping. All 18 analytical references pass. Original Vulkan measured 23433.28/23348.28 tok/s on Qwen3-0.6B pp6144; the matched control measured 21269.95/21244.43. This matches arithmetic policy, not native operation ordering or bit-identical execution.

The implausible full-model Q8 GEMM timing of 7.228 us is excluded as standalone latency evidence. Warm isolated default-precision measurements are shape-dependent: for m/n/k=1024/512/1024, Vulkan measured 46.46/46.24 us and DX12 31.39/31.59 us; for 3072/512/1024 they measured 61.41/62.23 and 76.25/76.68 us; for 1024/512/2048 they measured 48.86/49.25 and 55.28/55.46 us. These are repeated one-node throughput measurements, not unique-activation full-graph dispatch timings.

Actual RX 9070 XT traces select `matmul_q8_0_f32_f16acc_aligned_m` at default precision and `matmul_q8_0_f32_aligned_m` at explicit F32 precision, without Q8_1 quantization. An initial audit's integer Q8_0 x Q8_1 attribution was incorrect: the active cooperative-matrix pipeline family does not populate the scalar integer-MMQ Q8_1 pipeline set. The trace-confirmed floating cooperative-matrix route must be used for comparisons.

### Qualified replacement

The final shader uses padded key-major score/P storage, four query rows per wave, full-wave softmax reductions, persistent Q fragments, and P fragments reused across both D128 PV phases. Direct RowMajor K times cached ColMajor Q-transpose now passes the asymmetric references with literal strides, so score stores also use RowMajor. Both PV phases write disjoint columns of a full `[query][128]` exchange buffer before one synchronized readback, rather than synchronizing and exchanging after each phase.

The successive measured Qwen3-0.6B pp6144 controls were approximately 19400 tok/s for the first Q-cache pipeline, 20348 for padded Vulkan rows with P caching, and 20542 for transposed QK. Full-width PV exchange provided the final improvement. Q reload with P caching regressed to 17947 despite lower register allocation; lower register counts alone are not a performance objective.

The final source is 288 lines rather than the 731-line experimental version. Rejected staging, alternative row layouts, reload paths and unsafe precision controls remain in the experiment archive, not in production. Clean stride-256/2048 CSOs are byte-identical to the qualified experimental variants. Native analysis reports 96 requested/allocated VGPRs, 68 SGPRs, 15360 bytes of LDS, zero scratch, 16 static F32 WMMAs and five barrier pairs. The full-width exchange removes two barrier pairs while increasing LDS by 2048 bytes.

Balanced whole-model results use Q8_0, F16 K/V caches, `-ub 512 -b 2048 -ngl 99 -fa on`, four repetitions per process, and two process means per arm. The baseline is the preserved pre-branch runtime, not an intermediate experiment.

| Model | Prompt tokens | Baseline tok/s | Replacement tok/s | Gain |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 6144 | 18984.54 | 20900.26 | 10.09% |
| Qwen3-0.6B | 16384 | 10591.32 | 12187.97 | 15.08% |
| Qwen3-4B | 6144 | 4724.31 | 5122.03 | 8.42% |
| Qwen3-4B | 16384 | 3167.14 | 3570.84 | 12.75% |

Qwen pp512 remains outside the new route and is essentially unchanged. SmolLM2-135M also retains its D64 route: pp6144 measured 46733.39 baseline versus 46596.52 with the opt-in enabled, within the process/run variation. This does not claim a Smol optimization; the D64 shader compilation is not a qualified D64 host route.

In the same balanced Qwen3-0.6B cohort, arithmetic-matched Vulkan measured 20982.64 tok/s at pp6144 and 12060.39 at pp16384. The replacement is within approximately 1% of those results. Original-policy Vulkan still measured 23041.30 and 13742.27 respectively. Thus the rewrite closes the measured arithmetic-matched pipeline gap, not the full original-policy Vulkan gap. Different numerical policies remain material, and the remaining original-policy gap must not be described as solved.

A separate balanced DX12 profiling cohort reduced the final 28 attention operations from 24.308 ms to 19.2975 ms, a 20.61% time reduction, with flags 323 and 327 respectively. The graph-reused mask producer is included in whole-model measurements. Backend-specific timestamps are not used to claim exact cross-API shader latency parity.

The final binaries pass 21 analytical references in both root-constant and CBV modes, 536 D128/F16 cases, and 178 route plus 46 composed assertions. The existing range fixture now covers pipeline overflow and asymmetric tiny probabilities with both literal K strides. Replay retains 99 hits/1 miss and correctly changes routes after alignment changes; the 302-dispatch scratch-growth probe passes. Ten 2048-context corpus chunks give PPL 9.1596 versus the preserved baseline's 9.1602, within the reported +/-0.2813 uncertainty.

`llama-cli`, `llama-bench` and `llama-mtmd-cli` are rebuilt in `build-linalg` and standard `build-dx12`. The replacement requires the LinAlg build and remains opt-in:

```powershell
$env:DX12_FA_LINALG = "1"
$env:DX12_FA_PV_F16 = "0"
$env:DX12_FA_PIPELINE = "1"
```

Use the existing benchmark commands with `build-linalg\bin\llama-bench.exe`. Set `DX12_FA_PIPELINE=0` for the unchanged path. This route remains restricted to RDNA4 discrete wave64, eligible D128 F16 K/V layouts, full 64-key tiles, and exact K strides of 256 or 2048 bytes. Other devices, dimensions and unsupported layouts retain existing routing. Final `build-linalg\bin\ggml-dx12.dll` SHA256: `28F085A8F3AE14801E5C4FB79D4E38BE80556B85BB863791A2AC3B07CA1A01DB`.

## Revised parity priorities (September 18, 2026)

The September 17 Vulkan report changed the priorities from further D128 half carry to D64 attention, quantized expert decoding, large Q6_K output heads, multimodal GEMM, and selective D96/BF16 work. The baseline for this section is the qualified D128 replacement above, not commit `4f1fa66d3` alone. These changes remain on `dx12-vulkan-parity`; no new commit or push was made.

### D64 and D96 attention

Simply compiling the replacement for D64 did not work as an optimization. Its original Br16 full-wave row reductions lost approximately 3.6% on SmolLM2 and were flat on Granite. Expanding to Br32 increased row-state pressure and regressed both models by approximately 7-9%. Neither version is retained.

The accepted D64 shader partitions each wave into independent query rows. Br16 uses 16 lanes per row and four keys/output dimensions per lane; Br32 uses eight lanes per row and eight keys/output dimensions per lane. Partitioned reductions carry one maximum and denominator per lane rather than four or eight independent rows. Q/P shared storage is aliased after Q fragments are cached. QK, PV, output carry and normalization retain F32 accumulation, and the explicit subnormal-P conversion is unchanged.

Native resource extraction reports Br16 allocation falling from 72 to 56 VGPRs and LDS from 11264 to 9216 bytes. Br32 falls from 100 to 80 VGPRs and from 19456 to 15360 bytes of LDS. Both have zero scratch. Br16 is selected for K strides 128, 384 and 1536; Br32 is selected for stride 1024. Granite benefits more from Br32 while Smol benefits more from Br16. These are literal byte strides, not interchangeable shader variants.

The packed-mask producer, consumer dispatch, arena sizing and cache identity now share the selected query-tile height. Mixed Br16/Br32 consumers cannot reuse incompatible metadata. Whole-graph regression cases use alternating 16-row visible/masked bands, shared masks, head broadcasting and intervening overwrites.

D96 uses Br16 and literal K strides 192 or 6144. Q staging rounds its iteration count upward, and the last PV phase runs only on waves owning dimensions 64-95. All waves still reach the group barriers. The D128 generated shaders remain byte-identical to the earlier qualified replacement.

`DX12_FA_PIPELINE=1` continues to opt into the attention pipeline. The existing F16 K/V, alignment, full 64-key tile, >=1024 KV, hardware and mask guards remain. The workgroup floor is 256 for Br16 and 128 for Br32. Unsupported layouts and short contexts retain the previous routes.

### Quantized decoding

Flag 340 replaces the existing Q6_K DP4A flag 23 only on discrete RDNA4 wave64, at default precision, one token and at least 65536 output rows. Its packed mapping reuses ql/qh bytes across four groups instead of repeatedly reconstructing the same payload. It retains two output rows per group, the existing large-vocabulary chunk splitting and the existing Q8_1 activation quantizer/cache. Explicit F32 precision and smaller outputs retain their previous routes. `DX12_Q6K_PACKED_MMV=0` disables this default-on specialization.

Expert flags 341/342 process four Q4_K/Q6_K output rows per group and share activation slices across those rows. Q4_K replaces eligible flag 159; weighted Q4_K/Q6_K replace the existing flag-18 router-weight epilogue. Weighted output remains F32, and the existing router binding and strides are retained. These expert routes use Q8_1 activations, including weighted down projections which previously consumed F32 activations directly; this is an activation-quantization policy change, not bit-identical arithmetic. CPU-reference comparisons retain the existing numerical tolerance. Decision replay preserves the selected expert flag and quantization state.

Expert selection is restricted to discrete RDNA4 wave64, contiguous F32 activation input, K divisible by 256, one outer batch and at most eight tokens. The later LinAlg prefill selection retains precedence. `DX12_MOE_KQ_ROWS4=0` disables the default-on expert specialization. Both quantized optimizations are available in standard DX12 as well as the LinAlg build; NVIDIA and integrated-GPU routes are unchanged.

### Multimodal GEMM

The accepted full-tile variants remove bounds/type-selection work only when tensor layout proves it unnecessary. Default selection is limited to the measured SmolVLM2 shapes:

| Input weights | M | K / N | Staging |
| --- | ---: | --- | --- |
| F16 | 1024 | 768 / 768, 3072 / 768 | Single buffer, 128x64/BK16 |
| F16 | 1024 | 768 / 3072 | Double buffer, 128x64/BK16 |
| Q8_0 | 64 | 576 / 192, 576 / 576, 576 / 1536, 1536 / 576 | Existing 32x16 or 32x32/BK32 geometry, double buffer |

Inputs must be K-contiguous with aligned offsets/outer strides, activations must be F32, output must be contiguous F32, and M/N/K must be exact tile multiples. Small tails, including M=12, retain the edge-capable kernels. Late specialization rechecks the actual tensors rather than trusting a cached geometry decision. Bias handling and F32 accumulation are unchanged.

`DX12_LINALG_FULL_F16=0` and `DX12_LINALG_FULL_Q8=0` disable these defaults. Explicit value 1 permits the respective full-tile double-buffer variant on other eligible shapes; F16 value 2 selects single buffering. Q8 single buffering and forcing all small Q8 tiles to 32x32 were measured and rejected. Single buffering is not generally faster: it wins the two retained N=768 vision shapes but loses to double buffering at N=3072.

### Whole-model results

The following are balanced baseline/candidate/candidate/baseline process means on RX 9070 XT. Text measurements use three timed repetitions per process, pp6144 or tg512, `-ngl 99 -fa on -dev DX120`. They include preprocessing and dispatch overhead, not just isolated shaders.

| Workload | Baseline | Candidate | Throughput gain |
| --- | ---: | ---: | ---: |
| Granite Q4_K_M tg512 | 412.25 | 535.08 | 29.8% |
| Qwen3-0.6B Q4_K_M tg512 | 479.92 | 533.49 | 11.2% |
| Qwen3-4B Q4_K_M tg512 | 173.45 | 176.75 | 1.9% |
| Granite F16 pp6144 | 18061.70 | 19665.07 | 8.9% |
| Granite Q8_0 pp6144 | 18991.74 | 20786.64 | 9.5% |
| Granite Q4_K_M pp6144 | 16924.36 | 18357.25 | 8.5% |
| SmolLM2 F16 pp6144 | 43264.03 | 44909.80 | 3.8% |
| SmolLM2 Q8_0 pp6144 | 46397.95 | 48009.68 | 3.5% |
| SmolLM2 Q4_K_M pp6144 | 41314.30 | 42184.69 | 2.1% |
| Phi-3 F16 pp6144 | 4706.01 | 4941.99 | 5.0% |
| Phi-3 Q8_0 pp6144 | 5478.57 | 5821.40 | 6.3% |
| Phi-3 Q4_K_M pp6144 | 4384.78 | 4597.91 | 4.9% |

SmolVLM2 uses the user's image/prompt, 480 prompt tokens, 512 generated tokens, temperature zero and seed 1. Three launches per format/arm were recorded; the final two launches from each arm form the warm means below. Cold launches are not pooled with them.

| SmolVLM2 format | Baseline prompt time | Candidate prompt time | Time reduction |
| --- | ---: | ---: | ---: |
| F16 | 167.25 ms | 159.49 ms | 4.6% |
| Q8_0 | 170.04 ms | 159.28 ms | 6.3% |
| Q4_K_M | 169.23 ms | 159.67 ms | 5.7% |

Warm multimodal timings have appreciable launch-to-launch variation. Exact GEMM shape controls and graph profiles support the retained routing, but these results do not establish Vulkan parity. Granite/Qwen quantized generation still trails the supplied Vulkan report despite the larger gains.

### Falcon BF16 and rejected work

Full-tile BF16 variants did not produce a stable full-offload gain; single buffering regressed. They were removed rather than enabled speculatively. Falcon's 14.13 GiB weight footprint is close to this adapter's reported budget. A controlled `-ngl 40` profile changed the large BF16 GEMM aggregate from 86 operations/200.084 ms to 76 operations/41.836 ms; GLU changed from 43 operations/54.338 ms to 38 operations/3.948 ms. Multiple operation classes recover, so attributing the entire slowdown to BF16 GEMM arithmetic is not supported.

This is evidence of capacity-sensitive behavior, not a direct paging measurement or a completed residency fix. No automatic offload reduction or global allocation-policy change is made. Falcon BF16 remains an open performance gap.

### Qualification and artifacts

The final pipeline passes 21 analytical range/mask references for each of D64/D96/D128 in both root-constant and CBV modes, 882 existing F16 attention cases, and 226 route plus 46 composed assertions in both parameter modes. Existing test infrastructure covers expert broadcast/weighted tails, vocabulary chunk boundaries, explicit F32 fallback, full-tile shape fallback, mixed mask geometry and mask writes. The standard DX12 build passes 26 route assertions in each parameter mode.

A final balanced Qwen3-0.6B Q8_0 pp6144 comparison measured 21055.37 versus 21067.78 tok/s, consistent with unchanged D128 performance rather than another claimed gain. Granite Q4_K_M also completes generation with whole-command-list replay explicitly enabled.

Raw balanced logs, rejected DLLs and isolated controls are under the session artifact directory with `revised-*` names. `revised-baseline.dll` is the September 17 qualified D128 baseline. The inference executables `llama-cli`, `llama-bench` and `llama-mtmd-cli` are built in both `build-linalg` and `build-dx12`. Keep the existing attention opt-in settings above for the extended pipeline; quantized decoding and the qualified multimodal GEMM shapes need no additional opt-in.

Final installed backend SHA256:

- LinAlg: `0A53D72579F0B71F2864AE0913E214C449980228D703EF052224B42575655765`
- Standard DX12: `70876A9D5F733E2F2B77CC417BB9A0D4FEF2449F7EABC98DE2BB8DBBAE4ED711`

## Remaining-gap follow-up (September 18, 2026)

Commit `61954a8d5` preserves the preceding pipeline integration and kernels, except for the D64 source mismatch corrected below. The September 18 user report confirms Granite Q4_K_M generation improving from 420.96 to 537.45 tok/s and Qwen3-0.6B Q4_K_M from 483.17 to 530.23 tok/s, but both still trail Vulkan. Fresh profiles also identify normalization as a material decoding cost, alongside the remaining expert/dense matvec work.

### Fixed-width normalization and packed KV stores

On discrete RDNA4 wave64, flags 381/382 specialize 1024-element F32 ADD+RMS_NORM+MUL and RMS_NORM+MUL rows. One wave owns a row, loads four float4 vectors per lane, keeps those values in registers, and reduces without group barriers. The ADD variant also writes the residual intermediate. This avoids the generic ADD kernel's dynamically indexed cache and runtime element-type/store branches. Host guards require aligned vector accesses, matching widths and F32 types; ADD also requires a contiguous residual layout. Unsupported widths, dim0 broadcasting and layouts retain the previous kernels.

These routes default to single-row decoding. `DX12_ADD_RMS_FIXED=0` and `DX12_RMS_FIXED=0` opt out; explicit value 1 also permits eligible multi-row controls. The output remains F32, with no additional activation quantization.

Flag 380 keeps the existing 256-thread RMS+MUL+NEOX+SET_ROWS arithmetic but exchanges adjacent lane results to store two F16 values per DWORD. This removes the generic half-store atomic retry loops. Its host contract requires full D128 NEOX rotation and aligned, contiguous F16 head elements; the default is restricted to decoding. `DX12_NORM_ROPE_PACKED=0` opts out, and explicit value 1 permits eligible multi-token controls.

Balanced off/on/on/off tg512 controls, three repetitions per process unless noted:

| Isolated change | Model/format | Off tok/s | On tok/s | Gain |
| --- | --- | ---: | ---: | ---: |
| Fixed ADD/norm | Granite F16 | 299.85 | 310.88 | 3.7% |
| Fixed ADD/norm | Granite Q8_0 | 496.83 | 532.06 | 7.1% |
| Fixed ADD/norm | Granite Q4_K_M | 544.06 | 584.53 | 7.4% |
| Fixed RMS multiply | Qwen3-0.6B BF16 | 346.20 | 360.71 | 4.2% |
| Fixed RMS multiply | Qwen3-0.6B Q8_0 | 482.18 | 511.19 | 6.0% |
| Fixed RMS multiply | Qwen3-0.6B Q4_K_M | 538.80 | 560.94 | 4.1% |
| Packed KV stores, fixed RMS already on | Qwen3-0.6B Q4_K_M | 559.23 | 571.92 | 2.3% |

The packed-store control uses five repetitions per process. These are separate controls, not additive percentages or a substitute for a final combined comparison. The fresh Granite profile reduces 48 fused ADD/norm dispatches from approximately 0.249 ms to 0.119 ms.

Rejected controls include a generic 64-thread ADD/norm unroll (approximately 20% slower Granite Q4 generation), a single-wave version of the entire K-normalization shader (less consistent than changing only its stores), and enabling the existing RMS/Q8_1 quantization fusion on RDNA4 (Qwen Q4/Q8 generation regressed). No such routing changes are promoted.

The existing fused-operation CPU comparisons pass, including the added 1024-element, batched and broadcast cases. Route introspection now includes normalization so these variants and their opt-outs can be asserted directly. LinAlg passes 240 route and 46 composed assertions; standard DX12 passes 40 route assertions, each in root-constant and CBV modes. Raw controls use the `round2-*norm*` artifact prefix.

### Vectorized F16 expert decoding

Flag 360 loads four contiguous F16 weights and four F32 activation values per lane, accumulating four expert output rows per group in F32. It replaces the cooperative scalar-load path without quantizing activations or introducing scratch buffers. Existing expert IDs, broadcast activation slots, weighted-router epilogues and decision replay are retained.

The initial eight-row candidate raised Granite F16 tg512 from 315.38 to 365.22 tok/s (+15.8%) with the normalization improvements already enabled. A separate balanced geometry comparison favored four rows: 370.86 versus 364.57 tok/s (+1.7%). The four-row version is retained. The eight-row profile reduced the up/gate aggregate from approximately 1.196 to 0.863 ms and weighted down from 0.786 to 0.588 ms; those profile numbers are not attributed to the later four-row version.

Default routing is limited to the measured K1024/N512 and K512/N1024 F16 expert shapes at one token on discrete RDNA4 wave64. `DX12_MOE_F16_VEC=0` opts out; explicit value 1 permits other eligible shapes with at most eight tokens. Inputs must be contiguous F16 weights and F32 activations with 8-byte and 16-byte aligned offsets/outer strides respectively, K divisible by 256, a single outer batch and legal dispatch bounds. Unsupported precision requests, K tails and unaligned views keep the previous routes. Q8_0 and K-quant experts are not changed by this selector.

The final geometry passes 265 route plus 46 composed assertions in LinAlg and 65 route assertions in standard DX12, in both root-constant and CBV modes. These include weighted 1/3/8-token cases, output-row tails, activation broadcasting, defaults, opt-outs and independent weight/activation offset fallbacks. Artifacts use `round2-f16-*`.

### D64 source reproducibility correction

The preceding delivery DLL contained the qualified partitioned-row D64 shader, but commit `61954a8d5` accidentally retained its earlier wave-per-row source. A subsequent backend rebuild therefore regressed D64 prefill without changing its route flags. The shader source is restored to the qualified implementation (SHA256 `5096B7EAF8B0D2042982E5F9C00BD8F0BCE0571587C2A18129E9E6B067B9B7E1`), including shared Q/P storage and subgroup row reductions. D96/D128 arithmetic is unchanged.

With identical full attention opt-ins, Granite Q8_0 pp6144 returns from approximately 17,620 to 20,862 tok/s. This is recovery of the previous delivery, not a new improvement over the September 18 user benchmark. Decoding results are unaffected because these D64 prefill routes do not handle one-query generation.

### Combined decoding results

Balanced baseline/candidate/candidate/baseline tg512 runs, with three repetitions per process, compare the September 18 delivery backend against fixed normalization, packed K stores and four-row F16 experts together:

| Model | Format | Baseline tok/s | Updated tok/s | Gain |
| --- | --- | ---: | ---: | ---: |
| Granite | F16 | 301.61 | 370.07 | 22.7% |
| Granite | Q8_0 | 503.81 | 539.65 | 7.1% |
| Granite | Q4_K_M | 544.30 | 586.69 | 7.8% |
| Qwen3-0.6B | BF16 | 344.45 | 360.25 | 4.6% |
| Qwen3-0.6B | Q8_0 | 482.78 | 512.68 | 6.2% |
| Qwen3-0.6B | Q4_K_M | 537.76 | 568.35 | 5.7% |

These are same-machine paired controls, not percentages against the user's separate Vulkan report. Artifacts use `round2-combined-*`.

### Rejected prefill geometry controls

Q8 bucket MMID BK32 variants passed CPU-reference routing comparisons, including partial expert tiles and activation-slot broadcasting, but regressed Granite Q8_0 pp6144. With the restored attention source, the existing 128x64/BK16 route measured 20,862 tok/s, versus 16,855 for 128x64/BK32 and 17,436 for 64x64/BK32. Halving K-loop synchronization did not compensate for the resource and tile tradeoffs. Both variants and their experimental selector were removed.

Routing SmolVLM2's M2/M6/M12 Q8 GEMMs to the existing one-wave 16x16 shader also failed to improve the four measured shape families. At M2, K1536/N576 rose from 36.01 to 38.32 us and K576/N576 from 18.19 to 19.56 us; K576/N1536 was effectively flat. Larger MMQ controls, including their activation quantization, and the non-LinAlg fallback were slower still. The qualified two-wave 32x16 route is retained rather than adding an unproven multimodal default.

A two-wave D64 attention implementation reduced Q/P fragment loads and used eight-lane row reductions with F32 state. With score padding 8, Smol pp6144 was effectively unchanged in F16/Q8_0/Q4_K_M. Padding 4 lowered LDS use further, but balanced ten-repetition Q8_0 controls measured only 48,879 versus 49,295 tok/s (+0.85%), smaller than the approximately 1.5% within-arm standard deviation. Warm Q8 multimodal prompt means for padding 8 were 155.22 versus 159.10 ms, not an improvement. Granite's alternative two-wave Br16/pad4 route measured 20,730 tok/s versus the existing Br32 route's 20,862. These candidates were removed; the qualified partitioned four-wave source is retained.

A Q4_K gate/up DP4A candidate consumed the existing RMS+MUL Q8_1 producer cache and fused SwiGLU. CPU-reference route assertions passed, including producer/consumer selection and explicit F32 fallbacks, but the full normalized producer plus consumer regressed Qwen3-0.6B Q4_K_M tg512 from 564.55 to 545.25 tok/s (-3.4%). Consumer-only instruction counts were not a sufficient reason to retain it. Its shader, host selector and experimental fixtures were removed.

The additional prefill and multimodal candidates therefore do not establish another improvement or Vulkan parity. Remaining gaps include Granite expert prefill, Smol prefill, warm multimodal prompt processing and quantized Qwen/Granite decoding. The retained gains in this follow-up are the decoding changes above and the D64 source reproducibility correction.

### Follow-up delivery

Both `build-linalg` and `build-dx12` contain rebuilt `llama-cli`, `llama-bench` and `llama-mtmd-cli`. The final LinAlg backend passes 265 route plus 46 composed assertions in both parameter modes, 2,426 F16 attention comparisons, and 126 analytical attention cases across D64/D96/D128 and both parameter modes. Standard DX12 passes 65 route assertions in each mode.

Final installed-backend runs measured Granite F16 tg512 at 369.32 tok/s, Qwen3-0.6B Q4_K_M tg512 at 560.23 tok/s and Smol Q8_0 pp6144 at 48,829 tok/s. Warm SmolVLM2 Q8_0 prompt time averaged 153.75 ms across launches 2 and 3; this confirms retention of the preceding delivery, not a new multimodal gain. Granite F16 also completes with whole-command-list replay explicitly enabled.

Final backend artifacts are saved under `round2-delivery-*`. Installed DLL SHA256 values:

- LinAlg: `60AD380C8226AB92B2DD32AB5821FE75B615052C8C110DDE7826231B2A6DC952`
- Standard DX12: `2DDD9B2324FC56CE0E1131FCED1A168DBBF4DA12A6CD25640318412AA0706ABB`

## NVIDIA D128 attention pipeline port (September 18, 2026)

Phase 1 on RTX 5070 (PCI 0x2F04, wave32, driver 620.12) ports the qualified F16-input/F32-accumulator D128 dataflow and packed actual-mask metadata. It remains opt-in with `DX12_FA_PIPELINE=1`; unset, 0, or any other value retains the previous NVIDIA route. `DX12_FA_LINALG=0` and `DX12_FA_PV_F16=1` exclude the pipeline. There are no normalization, expert, full-tile GEMM, or default dense flag301 changes in this phase.

The NVIDIA shader uses four wave32 waves (128 threads), Br16/Bc64, and two keys per lane for each owned softmax row. Both keys participate in the row maximum and F32 denominator and are written to the padded key-major P tile. The port retains hoisted Q fragments, cached P fragments, full-width PV shared exchange, F32 persistent output/max/sum, and the existing subnormal-P conversion. It does not use F16 persistent accumulation or carry approximations. NVIDIA already had direct K loads on flag177; these gains are not a new direct-K workaround.

Flags 325 and 327 select wave-specific bytecode for literal K strides 256 and 2048 bytes. The NVIDIA gate is limited to discrete RTX 5070 wave32, D128, the existing LinAlg/F16 capability checks, F32 Q, aligned F16 K/V, at least 1024 keys divisible by 64, and the existing group-count, broadcast, stride, mask, and address-range guards. Dynamic V strides remain supported. D64/D96, unsupported K strides, unaligned views, F32/quantized KV, short KV, and small query grids retain their previous routes.

The packed-mask shader also uses four waves and advances two query rows per iteration on wave32, rather than skipping half the rows. Its 2-bit classifications and sixteen 64-key tiles per DWORD are unchanged. It reuses the existing graph-local metadata arena, resource/view/shape/stride/type/query-geometry cache key, write invalidation, graph reset, and replay exclusion. Qwen3-0.6B and Qwen3-4B profiles show one metadata build and respectively 27 and 35 reuses per eligible graph.

### Matched measurements

The baseline is commit `203d2cfb22c58a674943e99c482cc0fc69a41577`, with DX12 DLL SHA256 `e07fc0a302bff8d28f32d834c4a0269e14cb945c94620b160fdc045b271367c8`. All measurements use the same machine and model files, serial independent processes, alternating ABBA/BAAB order, ten-second idle intervals, llama-bench warmup, three repetitions per process, and a two-second repetition delay. Inherited `DX12_` variables are removed, then `DX12_NO_AUTOTUNE=1` and `DX12_TUNE_REFRESH=1` are set; only the candidate enables the pipeline. Parameters are `-p 6144 -n 0 -b 2048 -ub 512 -ngl 99 -fa on -dev DX120 -r 3 --delay 2 -o jsonl`. Q4/Q8 below describe model weights; KV remains F16.

| Model | Weight format | Baseline tok/s | Pipeline tok/s | Change |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B | Q4_K_M | 8816.84 | 12208.95 | +38.47% |
| Qwen3-0.6B | Q8_0 | 8949.00 | 12545.86 | +40.19% |
| Qwen3-4B Instruct 2507 | Q4_K_M | 2521.20 | 3051.17 | +21.02% |
| Qwen3-4B Instruct 2507 | Q8_0 | 2641.67 | 3240.67 | +22.68% |
| Phi-3-mini D96 control | Q4_K_M | 2851.07 | 2849.78 | -0.05% |
| TinyLlama D64 control | Q4_0 | 8749.77 | 8739.75 | -0.11% |
| SmolLM2-135M D64 control | Q4_K_M | 24250.39 | 24202.75 | -0.20% |

At pp16384 with the other parameters unchanged, Qwen3-0.6B Q4_K_M improves from 4153.38 to 6478.33 tok/s (+55.98%), and Qwen3-4B Q4_K_M from 1434.49 to 1978.12 (+37.90%). Separate explicit `DX12_FA_PIPELINE=0` controls against the baseline change Qwen3-0.6B Q4, Qwen3-4B Q4, and Qwen3-4B Q8 by -0.34%, +0.20%, and +0.11%. The small control differences do not establish performance changes.

Separate profiles confirm real flag327 dispatches, with flag177 retained for the first 512-key batch. At the final 6144-key batch, the aggregate attention time changes from 73.776 to 42.148 ms for Qwen3-0.6B Q4 (28 dispatches), 173.889 to 105.242 ms for Qwen3-4B Q4 (36), and 174.595 to 104.352 ms for Qwen3-4B Q8 (36). These are positive GPU timestamps, not timing-run throughput measurements. Phi remains on flag179; TinyLlama and Smol remain on flag178.

### Qualification and limits

The delivered build passes 55/55 actual-dispatch route assertions in each of root-constant and CBV modes: the existing 23 NVIDIA composed assertions, 25 pipeline/guard/cache assertions, and seven analytical range assertions. The analytical cases include constant 3000-valued outputs, 8192-key cancellation, and tiny probabilities at 1024 and 8192 keys with both literal K strides. The cache fixtures test F16 and backend-internal F32 masks, one/two mask heads, zero/-infinity/mixed tiles, graph-to-graph reset, within-graph reuse, and mask overwrite invalidation. F32 mask fixtures attach the backend input directly because the public ggml constructor accepts F16 masks only.

The existing F16 D64/D96/D128 attention matrix passes 882/882 comparisons in each parameter mode, including D128 CPU-reference comparisons at 4096 and 16384 keys. Only two of those general-matrix cases select the new pipeline; the dedicated route and analytical tests provide the additional positive pipeline coverage. The route suite's separate `0/0` generic-case counter is not the route assertion count.

All ten affected AMD wave64 attention/metadata binaries compile byte-identically to the baseline source. An initial shared softmax-loop refactor changed AMD bytecode and was not retained; the existing wave64 softmax body is preserved separately. No AMD hardware performance claim is made by this NVIDIA qualification. The NVIDIA shader headers remain byte-identical between the first measured candidate and the final AMD-preserving rebuild.

This is an accepted opt-in attention improvement, not a claim of Vulkan parity or a default promotion. Quantized-KV LinAlg PSO failures on flags 169-174 were already present in the baseline; this F16-KV qualification does not certify those routes. Other NVIDIA GPUs, model generation quality/perplexity, F16 model weights, and whole-command-list replay with packed metadata are not newly qualified. Normalization/packed stores and expert bucketing/vector decoding remain separate phases.

Raw stdout/stderr, exact commands, paired JSON summaries, source diff, build logs, shader comparisons, baseline and candidate binaries, and `handoff.txt` are saved in the session artifact directory `nv-parity-port-20260918`. The build uses the existing `build-dx12-phase0` preview SDK/BoringSSL configuration without new dependencies.

## NVIDIA fixed normalization qualification (September 18, 2026)

Phase 2 enables the existing 64-thread, width-1024 fixed RMS/MUL and ADD/RMS/MUL shaders for single-row decode on discrete RTX 5070 (PCI 0x2F04, native/blob wave32). Flags 382 and 381 retain all F32, 16-byte alignment, width, stride, fusion and residual-layout guards. `DX12_RMS_FIXED=0` and `DX12_ADD_RMS_FIXED=0` opt out; value 1 also permits eligible multiple-row inputs. Other NVIDIA devices, wide flag85 normalization, quantization policy, attention and expert kernels are unchanged. AMD routing and all normalization shader bytecode are unchanged.

The immutable baseline is the phase-1 candidate DLL `b26d91bfa60ee09d61c0838f15ab22331f8748c05e3e745fd179093cecf59ce1`. Serial independent ABBA/BAAB processes use `-p 0 -n 512 -b 2048 -ub 512 -ngl 99 -fa on -dev DX120`, warmup, ten-second idle intervals, and `--delay 2`. Inherited `DX12_` variables are cleared; both arms set `DX12_NO_AUTOTUNE=1`, `DX12_TUNE_REFRESH=1` and the identical `DX12_FA_PIPELINE=1`. Final default qualification uses five repetitions per process and two processes per arm:

| Model | Weights | Baseline tok/s | Fixed norms tok/s | Change |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B | Q4_K_M | 497.12 | 503.40 | +1.26% |
| Qwen3-0.6B | Q8_0 | 400.89 | 410.21 | +2.33% |
| Granite 1B A400M | F16 | 211.90 | 215.63 | +1.76% |
| Granite 1B A400M | Q8_0 | 380.66 | 391.45 | +2.83% |
| Granite 1B A400M | Q4_K_M | 433.59 | 446.85 | +3.06% |

Earlier three-repetition controls measured +1.30%, +1.71%, +1.73%, +2.77% and +2.49%, respectively. Granite Q4's first final series had a slow baseline process and reported +5.20%; the table uses a complete repeat, not a filtered average. Qwen3-4B Q4's unchanged-width control is -0.04%; explicit-off Qwen0.6 Q4 and Granite Q4 controls are -0.12% and -0.06%.

Separate positive-timestamp profiles show Qwen0.6 Q4 RMS flag2 -> 382 at 29 dispatches, 0.114 -> 0.087 ms per graph; Q8 at 56 dispatches, 0.225 -> 0.177 ms. Granite F16/Q8/Q4 ADD flag3 -> 381 at 48 dispatches changes approximately 0.239/0.241/0.243 -> 0.168/0.168/0.171 ms. Final root-constant and CBV runs each pass 71 actual-dispatch assertions (16 normalization, 55 pre-existing), 897 general operator comparisons and 255 fused-graph comparisons. Normalization assertions include defaults, opt-outs, multiple rows, width tails, aligned/unaligned weight views and residual broadcasting.

The native 32-thread/eight-float4 alternative was rejected: Qwen Q4 and Granite F16 tg512 changed -2.19% and -0.44%, and their normalization profiles were slower. The source and rebuilt blobs were restored to the 64-thread version.

The standalone wave32 packed-KV store experiment passed actual flag380 CPU comparisons but is not retained. Normal Qwen decode uses merged Q/K normalization flag104, so `DX12_NORM_ROPE_PACKED=1` did not select flag380. A diagnostic with `DX12_QK_NORM_MERGE=0` selected flag380 and improved 485.06 -> 496.32 tok/s (+2.32%), with K-normalization time 0.075 -> 0.032 ms; this is not a gain over normal merged decode. Default packed-only results had no flag380 dispatches and are not credited. The AMD packed-store path and NVIDIA merged Q/K shader remain unchanged; a future merged-store optimization requires separate qualification.

Artifacts, rejected candidates, exact commands, raw logs, source/binary provenance and `handoff.txt` are in `nv-parity-decode-20260918`. No Vulkan-parity or model-quality claim is made.

## NVIDIA expert bucketing and F16 vector decoding (September 18, 2026)

Phase 3 enables measured expert routes on discrete RTX 5070 (PCI 0x2F04, native/blob wave32). Bucketing reuses the existing compact producer, resource/view/expert-count cache key, write invalidation and graph reset. Flags 303/304 select new wave32 32x32/128x64 consumers, not the AMD wave64 blobs. Q8 wave32 uses the existing alignment-safe packed loader; native16 loads remain wave64-only. All 2,530 pre-existing shader headers are byte-identical, including AMD, attention, normalization and the retained 64-thread F16 vector shader.

The bucket default is limited to 32 experts, eight selected experts, 512 tokens, flag202's tall tile, and K/N pairs 1024/512 or 512/1024. Measured formats are F16, Q8_0 and Q4_K, plus Q6_K for K512/N1024. Other safe shapes/formats require `DX12_LINALG_MMID_BUCKET=1`; value 0 disables bucketing. Existing LinAlg/F16 capabilities, global/format opt-outs, 128-token floor, 256-expert bound, layout, dispatch and 32-bit address guards remain. AMD defaults are unchanged.

F16 flag360 reuses four weight/activation values per lane, four output rows, 64 threads, precise F32 accumulation and the weighted-router epilogue. Its NVIDIA default is one token, 32 experts, eight selected experts, and K/N1024/512 or 512/1024. `DX12_MOE_F16_VEC=1` permits other eligible small-token shapes; value 0 disables it. Contiguous F16 weights/F32 activations, K divisible by 256, at most eight tokens, default precision, dispatch limits and weight/activation alignment remain required. Ordinary eight-token LinAlg routing still takes precedence; eligible weighted eight-token fusion may use flag360. No Q8 activation conversion is introduced. NVIDIA weighted Q4_K/Q6_K remains on flag18, not flags341/342.

### Whole-model qualification

The immutable baseline is phase 2's final DLL `3fee7d69a4c78bfe475ab577bf17d8100b9dc43ed4abf4f20cc09f279f288816`, not a rebuild of HEAD. Both arms retain phase-1 attention and phase-2 normalization. Serial independent ABBA/BAAB processes use `-b 2048 -ub 512 -ngl 99 -fa on -dev DX120 -r 5 --delay 2 -o jsonl`, warmup and ten-second idle intervals. Inherited `DX12_` variables are removed; both arms set `DX12_NO_AUTOTUNE=1`, `DX12_TUNE_REFRESH=1` and `DX12_FA_PIPELINE=1`. Each table entry averages two process means.

| Granite 1B A400M workload | Weights | Baseline tok/s | Expert routes tok/s | Change |
| --- | --- | ---: | ---: | ---: |
| pp512 | F16 | 11007.11 | 11975.08 | +8.79% |
| pp512 | Q8_0 | 10767.18 | 12039.95 | +11.82% |
| pp512 | Q4_K_M | 10634.70 | 11551.98 | +8.63% |
| pp6144 | F16 | 8132.04 | 8665.59 | +6.56% |
| pp6144 | Q8_0 | 8156.31 | 8690.61 | +6.55% |
| pp6144 | Q4_K_M | 8000.17 | 8510.02 | +6.37% |
| tg512 | F16 | 216.12 | 393.18 | +81.93% |

Initial three-repetition runs independently measured +5.08% to +8.10% at pp512, +6.35% to +6.56% at pp6144, and +81.15% for F16 tg512. All runs are retained, including noisy initial Q8 pp512. F16 prefill also benefits from vector routing on final-token work; with `DX12_MOE_F16_VEC=0` in both arms, bucket-only repeats still improve pp512 by +10.11% and pp6144 by +6.45%.

Unchanged tg512 controls are Granite Q8 -0.11%, Granite Q4 +0.29% and Qwen3-0.6B Q4 +0.03%; dense Qwen pp6144 is +0.03%. Explicit-off controls are F16 tg512 +0.04% and Q8 pp6144 -0.01%. These small differences do not establish gains. A 32-thread vector variant passed correctness but did not beat the existing 64-thread shader in a direct five-repetition ABBA comparison (394.60 versus 395.93 tok/s, -0.34%). It was removed and rebuilt; no unhelpful geometry switch remains.

Separate positive GPU timestamps show flag202 -> 304 in the final 6144-key prompt graph: F16 34.295 -> 30.518 ms, Q8 33.964 -> 30.172 ms and Q4_K_M 34.931 -> 31.341 ms, each across 69 eligible expert dispatches. F16 decode changes flag1 -> 360 at K1024/N512 (48 dispatches, 2.028 -> 0.683 ms) and weighted flag18 -> 360 at K512/N1024 (24, 1.042 -> 0.362 ms), averaged over three generation graphs. Q8 decode retains flag17; Q4_K_M retains flags159/18. Granite D64 attention remains unchanged.

Final root-constant and CBV modes each pass 169 actual-dispatch assertions: 37 vector (24 positive, 13 exclusions), 61 bucket (39 positive, 22 exclusions), and 71 preserved earlier-phase assertions. Bucket cases cover all supported formats under opt-in, exact expert bounds, tails, skew/sparse IDs, offsets/strides, global/format opt-outs, qualified defaults, cache reuse/expert-count keys and in-place ID writes. Two analytical fixtures execute the same graph three times with changed nonzero-offset IDs, both with reuse and intervening overwrite, and assert two real flag303 dispatches per execution. The generic route runner's separate 0/0 count is not certification.

The existing expert suites pass 892/892 CPU comparisons in each mode; the full backend suite passes 18,136/18,136 runnable comparisons in each mode. Unsupported cases, including pre-existing quantized-KV PSO failures169-174, are not newly certified. Explicit command-replay smoke runs complete, but Granite's ARGSORT nodes prevent whole-list capture; they are not evidence of captured-list replay. Other GPUs, broader default shapes, model quality/perplexity and multimodal GEMM are outside this qualification.

Final installed DLL: `fd15ad1ef04e46e48bd01f6e016db30d8bf45e2323a2eb3cd86dea0269520bbb`. Sources, all candidate binaries, exact commands, raw logs, shader preservation checks and `handoff.txt` are archived in `nv-parity-experts-20260918`. No commits, pushes or submissions were made.

## NVIDIA merged Q/K packed stores (September 18, 2026)

The follow-up evaluates the actual merged Q/K normalization path, without disabling `DX12_QK_NORM_MERGE`. `DX12_QK_NORM_PACKED=1` selects flag383 after the original flag104 parameter packing, resource binding and dispatch sizing. During this experiment, unset, 0 and other values retained flag104; the later default promotion is documented below. The opt-in is limited to discrete RTX 5070 PCI0x2F04, native/blob wave32, full D128 NEOX and a DWORD-aligned F16 K cache with DWORD-aligned row stride. F32 caches, partial rotary dimensions, normal RoPE and unsupported devices retain their previous shaders. No parameter slots, cache keys or root signatures are added.

One wave32-only wrapper changes just the K-side stores: adjacent lanes exchange their rotated values and even lanes write F16 pairs. Q remains F32; RMS arithmetic, shared storage, 256-thread geometry, RoPE and the merged dispatch are unchanged. All pre-existing shader bytecode, including AMD, stays identical. Testing also found that flag104's root UAV could not preserve a two-byte cache-base offset; the merge now excludes that layout so the existing standalone flag8 path handles it correctly.

The immutable baseline is the phase-3 candidate `fd15ad1ef04e46e48bd01f6e016db30d8bf45e2323a2eb3cd86dea0269520bbb`. Serial independent tg512 processes use `-p 0 -n 512 -b 2048 -ub 512 -ngl 99 -fa on -dev DX120 -r 5 --delay 2`, warmup and ten-second idle intervals. Both arms clear inherited `DX12_` settings and set `DX12_NO_AUTOTUNE=1`, `DX12_TUNE_REFRESH=1`, `DX12_FA_PIPELINE=1`; only the candidate adds the packed-store opt-in.

| Model | Comparison | Baseline tok/s | Packed tok/s | Change |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B Q4_K_M | ABBA + BAAB, four processes/arm | 503.87 | 511.63 | +1.54% |
| Qwen3-0.6B Q8_0 | ABBA + BAAB, four processes/arm | 408.98 | 409.71 | +0.18% |
| Qwen3-4B Q4_K_M | BAAB repeat, two processes/arm | 157.45 | 158.24 | +0.51% |

Earlier Qwen0.6 Q4 series measured +1.44% and +0.17%; the latter contains a noisy candidate process (490.07 tok/s, standard deviation 19.99). Q8 measured +0.63% and +1.46% before the stronger nearly-flat series. All raw runs remain available without outlier removal. Qwen4's first series measured +0.41%; explicit-off Qwen0.6 Q4 changes -0.07%. This is a small reproducible Q4 benefit, not a general decode improvement, so the route remains opt-in. No alternate geometry was introduced.

Separate positive-timestamp profiles confirm flag104 -> 383: Qwen0.6 Q4 at 28 dispatches changes 0.118 -> 0.067 ms, Q8 at 28 changes 0.111 -> 0.064 ms, and Qwen4 at 36 changes 0.165 -> 0.101 ms. These are complete merged-dispatch times, not isolated K-store timings.

Root-constant and CBV qualification each passes 180 actual-route assertions, 897 general operator comparisons and 255 fused-graph comparisons. The 11 new assertions cover both Q and the full K cache, aligned/unaligned cache views, F32, partial/normal RoPE, row-stride alignment, exact opt-in/default/off behavior and repeated execution. The repeated fixture compares changing positions and cache row IDs with CPU results over 129 executions. Root mode records all 129; CBV logs one real capture, 125 replays, three normal records and no invalidations. The fixture disables the test-only environment-refresh hook during replay because that hook intentionally disables decision caching.

Artifacts and the final handoff are in `nv-parity-packed-qk-20260918`. Phase-1 attention, phase-2 fixed normalization and phase-3 expert defaults are preserved. No other-GPU, model-quality or Vulkan-parity claim is made.

## NVIDIA combined delivery comparison (September 19, 2026)

The final installed backend is compared directly with the immutable, unmodified `203d2cfb2` branch build, not with an intermediate phase. Baseline DLL SHA256 is `e07fc0a302bff8d28f32d834c4a0269e14cb945c94620b160fdc045b271367c8`; final is `27896ac0e398dfafae22d660da477dd71efcd1ece1294afd6308aec79ab12d9a`. Both retain BoringSSL and the same preview SDK/toolchain configuration. This cohort was rerun after the decision-cache correction described below.

All six workloads use independent ABBA/BAAB processes, three timed repetitions per process, built-in warmup, ten-second idle intervals, `-b 2048 -ub 512 -ngl 99 -fa on -dev DX120 --delay 2`, and no concurrent builds or GPU jobs. Both arms clear inherited `DX12_` settings, then set `DX12_NO_AUTOTUNE=1`, `DX12_TUNE_REFRESH=1`, `DX12_FA_PIPELINE=1` and `DX12_QK_NORM_PACKED=1`. The original baseline has neither NVIDIA opt-in implementation. Thus the final results include the opt-in attention/packed routes alongside qualified normalization/expert defaults; they do not describe out-of-box attention performance.

| Model | Workload | Original tok/s | Final tok/s | Change |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B Q4_K_M | pp6144 | 8858.37 | 12197.11 | +37.69% |
| Qwen3-4B Q8_0 | pp6144 | 2640.72 | 3239.22 | +22.66% |
| Qwen3-4B F16 | pp6144 | 2598.71 | 3184.92 | +22.56% |
| Granite 1B A400M F16 | tg512 | 212.02 | 394.69 | +86.15% |
| Granite 1B A400M Q4_K_M | pp6144 | 8035.57 | 8510.21 | +5.91% |
| Qwen3-0.6B Q4_K_M | tg512 | 499.98 | 510.05 | +2.02% |

These are directly measured combined changes, not sums or products of phase percentages. The six-workload cohort adds F16-weight Qwen prefill coverage; the earlier pp16384 measurements remain phase-1 results rather than a repeated final long-context cohort. No model quality/perplexity or cross-API parity claim is made. Full-tile multimodal GEMM remains outside this change.

Raw results from all 24 processes, executable provenance and exact commands are archived under `nv-parity-final-cache-fix-20260918`; the earlier pre-review cohort remains separately archived under `nv-parity-final-20260918`. The reusable driver is `build-dx12-phase0\nv_parity_final.py --output nv-parity-final-cache-fix-20260918`. Both opt-ins remain off unless explicitly set to 1; qualified fixed normalization and expert defaults retain their existing opt-outs.

### Expert decision-cache validation follow-up

The NVIDIA default bucket guard must accept both the incoming tall flag202 and the selected/cached flag304. Rejecting cached304 rebuilt decisions on every unchanged graph despite executing the correct bucket shader. The guard now accepts202/304 under the same hardware, type, shape and layout checks; small-tile303 remains excluded from default selection.

One additional variant of the existing repeated bucket fixture runs a default-qualified graph three times with the bucket variable unset, environment refresh disabled, decision caching enabled and whole-command replay disabled. It asserts existing cache-counter deltas as well as numerical output and six actual flag304 dispatches. The unfixed guard produces zero hits, three misses and three rebuilds in both root/CBV modes, failing180/181 route assertions. The fixed guard produces two hits, one miss and one rebuild, passing181/181 routes and892/892 expert CPU comparisons in each mode. A read-only test introspection hook exposes the existing counters; no cache behavior or instrumentation counters were added.

Raw failing/passing logs, the fix-only diff, preserved-source checks, binaries and final hash are in `nv-parity-cache-fix-20260918`. The combined table above is a fresh integrated comparison of the corrected binary, not a relabeling of the preceding cohort.

## NVIDIA composed Q5_K and BF16 GEMM coverage (September 20, 2026)

Flag301 now supports Q5_K through the existing packed decoder and BF16 through a dedicated packed BF16-to-F32-to-F16 loader. BF16 bits are not interpreted as F16. Both operands retain the existing composed F16 tile policy with native F32 accumulation; BF16 follows the operand conversion already used by flag208. There is no half accumulator or integer/Q8 activation conversion. This does not extend the operand range to arbitrary BF16/F32 values outside the existing F16 tile policy.

The new defaults are limited to RTX 5070 PCI0x2F04, 512-token microbatches and one matrix batch. Q5_K is enabled only for K3072/N9216. BF16 is enabled for K/N3072/1024, 1024/3072, 2048/1024, 1024/1024 and 1024/2048. Other previously supported composed types keep their original defaults. `DX12_LINALG_NV_COMPOSED=0` disables the route; value 1 allows other eligible shapes for experiments. `DX12_LINALG_MM=0` excludes both new types, and `DX12_LINALG_MM_KQ=0` excludes Q5_K. Existing automatic tile/min-N override handling is unchanged.

The existing full-tile, contiguous F32 activation/output, source block stride, broadcasting, offset, address-range and dispatch limits remain. BF16 and Q5_K join the raw-load formats requiring DWORD-aligned weight strides as well as offsets. New NVIDIA wave32 blobs reuse the composed 128x64 output tile and parameter layout. All 2,553 pre-existing shader headers, including AMD/Intel variants, are byte-identical; no AMD/Intel routing or hardware qualification is claimed.

### Matched immutable-baseline measurements

The baseline is the incoming dirty-source delivery at `203d2cfb2`, DLL SHA256 `27896ac0e398dfafae22d660da477dd71efcd1ece1294afd6308aec79ab12d9a`, not a rebuild of clean HEAD. The retained DLL is `687cf68e8739b86b344b28e73a7a872f469df507ac7799e403ec23209ed186c3`. Complete baseline/candidate/final runtime directories are archived independently of the mutable build output.

Each clean row averages two independent process means per arm in ABBA/BAAB order, with built-in warmup, three repetitions, two-second repetition delay and ten-second serial cooldown. Both arms remove inherited `DX12_` settings, then set `DX12_NO_AUTOTUNE=1`, `DX12_TUNE_REFRESH=1`, `DX12_FA_PIPELINE=1` and `DX12_QK_NORM_PACKED=1`. Prompt arguments are `-p 6144 -n 0 -b 2048 -ub 512 -ngl 99 -fa on -dev DX120 -r 3 --delay 2 -o jsonl`; decode uses `-p 0 -n 512`. Profiles are separate and never counted as clean throughput.

| Workload | Baseline tok/s | Candidate tok/s | Ratio |
| --- | ---: | ---: | ---: |
| Phi-3-mini Q4_K_M pp6144 | 2848.57 | 3084.79 | 1.0829 |
| Qwen3-0.6B BF16 pp6144 | 9551.79 | 12661.83 | 1.3256 |
| Phi-3-mini F16 pp6144 control | 2799.17 | 2800.17 | 1.0004 |
| Qwen3-4B Q4_K_M pp6144 control | 3044.96 | 3043.65 | 0.9996 |
| Phi-3-mini Q4_K_M tg512 control | 189.71 | 189.62 | 0.9996 |
| Qwen3-0.6B BF16 tg512 control | 320.15 | 319.88 | 0.9991 |
| Granite 1B A400M F16 tg512 control | 383.42 | 388.81 | 1.0141 |

The noisy Granite control is not credited as a gain: expert routing and bytecode are unchanged. Explicit composed-off comparisons in both arms produce ratios 0.9992 for Phi Q4 and 1.0024 for Qwen BF16. Disabling composed also disables older composed types, so those Phi absolute speeds are not the normal baseline.

Separate profiles confirm Phi Q5_K flag128 -> 301: 428.923 -> 272.388 ms across measured prompt graphs13-24 (384 dispatches), and 35.647 -> 22.653 ms in graph24. Qwen's five BF16 shapes change flag208 -> 301: 307.709 -> 151.198 ms across graphs13-24, and 25.697 -> 12.601 ms in graph24. Phi D96 remains flag179; Qwen D128 remains flag327 after the short-KV batch. These are device-node intervals, not clean model throughput.

### Correctness and limits

The retained executable passes 208/208 actual-route assertions in both root-constant and CBV modes, including 27 new cases and all prior attention/norm/expert guards. The new cases exercise all six qualified shapes, default/forced/disabled routing, global/format opt-outs, broadcast batches, N/M/K tails, strided activations, nonzero weight/activation offsets, unaligned BF16 exclusion and explicit F32 precision. Nine changing-input executions of each target format produce exact analytical outputs beyond the F16 accumulator range, eight decision-cache hits, one miss and nine flag301 dispatches in each mode.

Whole-command-list replay intentionally excludes prefill. An initial fixture incorrectly expected captured CBV replay for M512; only that expectation was corrected. The existing merged-Q/K decode fixture still records one capture and 125 replays over129 executions. The composed fixtures certify parameter rebinding and decision-cache reuse, not captured prefill command lists.

The existing Q5_K/BF16 selector passes165/165 CPU comparisons per mode. The old composed-format/D128 F16-attention selector passes982/982 runnable comparisons per mode, with99 unsupported cases excluded. No tolerance was loosened. The first optimization candidate won both targets; no rejected performance configuration or experimental switch remains. Model quality/perplexity, other GPUs, larger default shape coverage and full-range BF16 arithmetic are not newly qualified. Falcon BF16 was not run.

Artifacts are under the ignored `build-dx12-phase0\nv-q5-bf16-20260920`: `run.py`, `analyze.py`, exact `commands.jsonl`, `clean-*-summary.json`, `profile-analysis.json`, raw logs, source snapshots/diffs, executable/model hashes and `handoff.txt`. This phase makes no Vulkan-parity claim; the old Vulkan comparator requires `-fa 1`, and its source is not a matched baseline. D96/D64 attention and Granite quantized-expert optimization remain later phases.

## NVIDIA D96/D64 pipeline attention coverage (September 20, 2026)

Phase2 adds three wave32 blobs: D96/Br16/K-stride6144, D64/Br32/K-stride1024 and the Br32 packed-mask producer. Existing flags333 and330 select the NVIDIA blobs on wave32; AMD retains its original wave64 blobs. D128 flags325/327 and all 2,555 incoming compiled shader headers are byte-identical. Phase1 GEMM and all expert implementations are unchanged.

The shared pipeline already contains the necessary wave32 dataflow. D96 uses two keys per lane to cover all64 keys in each row reduction, six16-wide head blocks and a second PV phase with only two active waves. All four waves still participate in the group barriers. D64/Br32 divides128 threads into32 rows with four lanes per row: each lane handles16 scores and16 output values; XOR reductions stay within each four-lane row. The existing Br32 query cache and two PV row tiles remain. Merely retaining one key per lane from the wave64 D96 path would be incorrect.

Both variants keep F32 QK/PV accumulators, online maxima, denominator, correction and output state. Q/K/V operands and the probability tile remain F16. The existing explicit `round(p * 2^24)` conversion preserves subnormal probabilities; no half carry or new approximation was introduced. The mask producer retains two bits per tile,16 tiles per DWORD, with32 lanes covering two query rows at a time. The existing scratch allocation, geometry-sensitive reuse, write-overlap invalidation and command-list replay exclusion are reused.

### Qualified routing

`DX12_FA_PIPELINE=1` is still an exact opt-in, not a global default. The new routes require discrete RTX5070 PCI0x2F04, native/blob wave32 and the existing LinAlg/F16 capabilities:

| Flag | D | Q/KV heads | K and V key stride | K and V head stride | Br/Bc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 330 | 64 | 16/8 | 1024 bytes | 128 bytes | 32/64 |
| 333 | 96 | 32/32 | 6144 bytes | 192 bytes | 16/64 |

The new layouts require one Q/K/V batch,256-1024 queries and1024-16384 keys in multiples of64. Existing F32-Q/F16-KV, offset/stride alignment, mask, group-count and32-bit address-range guards remain. Other head ratios, padded KV key strides, key tails, short KV, query counts outside that interval and additional batches fall back. Query tails, aligned offsets and padded query strides are supported. `DX12_FA_PIPELINE=0`, unset or any value other than1 disables pipeline routing; `DX12_FA_LINALG=0` and experimental `DX12_FA_PV_F16=1` also exclude it. D128 eligibility is unchanged.

### Incremental results against phase1

The immutable baseline DLL is `687cf68e8739b86b344b28e73a7a872f469df507ac7799e403ec23209ed186c3`; final DLL is `23e67e6c6aa538069e4f4ae76440e2f65fa6478e5d7158e54afad57faca8a1b4`. Clean process-isolated ABBA/BAAB runs use the preceding phase's environment and pp6144/tg512 commands, full built-in warmup, r3, two-second delay and ten-second serial cooldown. Both arms retain phase1.

| Workload | Baseline tok/s | Candidate tok/s | Ratio |
| --- | ---: | ---: | ---: |
| Phi F16 pp6144 | 2800.53 | 3257.66 | 1.1632 |
| Phi Q4_K_M pp6144 | 3085.45 | 3622.91 | 1.1742 |
| Granite Q4_K_M pp6144 | 8491.69 | 10246.53 | 1.2067 |
| Granite Q8_0 pp6144 | 8642.74 | 10499.21 | 1.2148 |
| Granite F16 pp6144 | 8616.19 | 10427.57 | 1.2102 |
| Qwen4 Q4_K_M pp6144 control | 3047.78 | 3052.34 | 1.0015 |
| Qwen0.6 BF16 pp6144 control | 12667.82 | 12730.18 | 1.0049 |
| Phi Q4_K_M tg512 control | 190.49 | 190.67 | 1.0009 |
| Granite Q4_K_M tg512 control | 436.19 | 435.99 | 0.9995 |
| Granite F16 tg512 control | 388.76 | 390.03 | 1.0033 |
| Qwen0.6 BF16 tg512 control | 323.59 | 324.10 | 1.0016 |

Explicit pipeline-off ratios are1.0006 for Phi Q4 and1.0003 for Granite Q4. Existing-backend-op ABBA microbenchmarks cover both new layouts at queries256/257/511/512/513/1024 and keys1024/6144: all24 shapes improve, with1.1264-1.5691x ratios. No losing geometry or speculative arithmetic variant remains.

Separate profiles over measured graphs13-24 show attention device intervals753.658 ->442.897 ms for Phi F16,756.075 ->446.714 ms for Phi Q4 and225.766 ->97.764 ms for Granite Q4. The first512-key batch retains flags179/178; subsequent batches use333/330. Last-graph attention intervals are119.530 ->69.718,119.675 ->70.175 and35.776 ->15.036 ms respectively. Baseline full-prompt attention shares are34.9%,38.7% and31.9%; last-context shares are not full-prompt shares. Profile throughput is not credited as clean throughput.

### Qualification and limitations

Root constants and CBV each pass282/282 route assertions, preserving all208 incoming cases, plus1501 CPU comparisons across seven GEMM formats and D64/D96/D128 F16 attention. The combined selector enumerates1600 cases;99 unsupported comparisons are excluded, including66 CPU-only and33 unsupported on both backends. Quantized-KV PSO failures169-174 are outside this selector and are not claimed as passes.

Coverage includes real512-query layouts through16384 keys, finite mask bias, shifted windows crossing64-key/1024-key boundaries, all-masked rows, tails, offsets, padding, head ratios, sinks, softcap and opt-outs. Analytical tests retain the existing tolerances for subnormal probabilities, cancellation over16384 keys and outputs that require F32 accumulation.

`test-backend-ops test -b DX120 -o DX12_ROUTES_FA` isolates106 attention assertions in a fresh process. Eight repeated-graph fixtures each record18 pipeline dispatches over nine changing-Q/K/V/mask executions, eight decision-cache hits, one miss and one rebuild. Same-geometry metadata reuse, Br16/Br32 separation and in-graph mask overwrite invalidation pass. Root and CBV rebind parameters without captured prefill replay. The unchanged packed-QK decode fixture still captures once and replays125 times. Initial cache-fixture failures came from cached opt-outs initialized by unrelated route cases, not numerical errors; the isolated selector avoids changing production environment-cache or replay policy.

Artifacts, exact commands, complete runtime snapshots, source-only diff, hashes and resolved failures are in `build-dx12-phase0\nv-d96-d64-20260920\handoff.txt` and adjacent logs/scripts. No model downloads, Falcon BF16 run, other-GPU qualification, model-quality evaluation or Vulkan-parity claim is included. Granite quantized-expert optimization remains phase3.

## NVIDIA Granite quantized experts (September 20, 2026 campaign)

Phase3 uses the retained phase2 runtime as its immutable baseline. Qualification is limited to RTX5070 PCI `0x2F04`, native/blob wave32, driver `32.0.16.2012`. F16 expert flags360/304, phase1 composed GEMM, phase2 attention and all AMD/Intel blobs and routing are preserved.

### Quantized prefill

The wave32 Q4_K, Q8_0 and Q6_K `bucket_128x64` specializations now drain F32 matrix registers directly through the existing per-expert scatter offsets. This removes the LDS accumulator store/reload and its epilogue barriers. Tile geometry stays128x64x16 with four32-lane waves, two row tiles and four column tiles per wave. Weight decoding, F16 staged operands, F32 accumulation, bucket construction and padding are unchanged.

Flag304 and the existing default gate remain: 32 experts, eight selected, 512 tokens, K/N1024/512 or512/1024; Q6_K defaults only for512/1024. The layout-safe `DX12_LINALG_MMID_BUCKET=1` override and `=0` opt-out retain their meanings. Cache eligibility still accepts initial202 or cached304, not303. All non-target bucket types and wave64 variants remain byte-identical.

### Q8_0 decode

Flag361 uses64 threads and four output rows instead of flag17's32 threads and two rows. Eight lanes share a32-weight block; each lane applies its packed four activation bytes to four weight rows. Two F32 wave reductions feed the final row stores. The established Q8_1 activation pre-pass/cache, packed integer dot products and precise F32 accumulators remain; there is no new activation quantization policy.

The default is limited to the same RTX5070, Q8_0,32 experts/eight selected, one token, K/N1024/512 or512/1024, contiguous weights/F32 activations/F32 output, one batch, default precision and DWORD-aligned offsets. Activations can broadcast across experts or have eight independent slots. Other shapes, views that violate the guards and explicit F32 precision retain previous routes. `DX12_MOE_Q8_ROWS4=0` disables361; `=1` does not expand its shape gate. `DX12_MOE_Q8_DP4A=0` also excludes it. The separate flag keeps PSO identity and four-row dispatch geometry consistent.

### Clean measurements

Serial independent ABBA/BAAB processes, two process means per arm, full warmup, `-r3 --delay2`, ten-second cooldown, `-b2048 -ub512 -ngl99 -fa on -dev DX120`; prefill6144 or decode512. Both arms strip inherited DX12 variables and set `NO_AUTOTUNE=1`, `TUNE_REFRESH=1`, `FA_PIPELINE=1`, `QK_NORM_PACKED=1` with the `DX12_` prefix. Every runtime file is hashed before/after each process. Profile throughput is excluded.

| workload | phase2 tok/s | phase3 tok/s | change |
| --- | ---: | ---: | ---: |
| Granite Q4_K_M pp6144 | 10279.20 | 10504.40 | +2.19% |
| Granite Q8_0 pp6144 | 10494.41 | 10646.54 | +1.45% |
| Granite Q8_0 tg512 | 390.75 | 441.96 | +13.11% |
| Granite F16 pp6144 control | 10487.35 | 10467.70 | -0.19% |
| Granite F16 tg512 control | 391.87 | 392.64 | +0.20% |
| Granite Q4_K_M tg512 control | 445.26 | 446.75 | +0.33% |
| Phi Q4_K_M pp6144 control | 3626.03 | 3628.79 | +0.08% |
| Qwen0.6 BF16 pp6144 control | 12695.22 | 12699.51 | +0.03% |
| Qwen4 Q4_K_M pp6144 control | 3053.92 | 3051.11 | -0.09% |

Q8 decode with361 disabled in both arms is392.61 vs392.05 tok/s (-0.14%). Unchanged-route controls are not credited as wins. An independent earlier four-process series reproduced the prefill and decode gains.

Separate profiles show measured-prompt expert device intervals374.930 ->362.013 ms for Q4 and361.058 ->353.518 ms for Q8, over graphs13-24. Both arms already use304; this is not a fallback-to-bucketing comparison. For Q8 decode, mean expert intervals over the final256 graphs fall0.983 ->0.709 ms, with48 gate/up and24 down dispatches per token. Their flags change17 ->361 and their x group counts halve256/512 ->128/256. The GPU span/node gap stays about0.114 ms, so this is GPU-kernel work saved, not a large CPU-idle fix.

### Correctness and rejected experiments

Root constants and CBV each pass318/318 actual-route assertions, including all282 incoming assertions, plus106/106 isolated attention assertions. The combined existing MUL_MAT_ID/MUL_MAT/F16-attention selector passes1822 CPU comparisons per mode;99 unsupported cases are excluded, not counted as passes. Seven quantized expert fixtures each verify nine changed-input/ID executions, eight decision-cache hits, one miss/rebuild and exact large F32 outputs. They cover full, partial, skewed and empty buckets, independent activation slots and scatter mappings. Layout/disable/default/excluded cases and all old-format bucket routes pass. Existing F16 cache tests and125 packed-QK CBV command replays remain intact; prefill replay eligibility is unchanged.

K32 staging was numerically correct but lost7.04% Q4 and6.80% Q8 prefill; it was removed. The retained K16 register epilogue won instead. The first analytical fixture used non-exact large F32 sums and failed an unjustified bitwise-equality expectation; power-of-two-scaled operands make both reduction orders exact without loosening any tolerance. A misplaced shader preprocessor terminator was corrected before any Q8 candidate ran. No unresolved build/numerical failure remains.

Exactly three pre-existing generated headers change: Q4_K/Q8_0/Q6_K tall bucket wave32. All other2555 incoming headers are byte-identical. Three ordinary wave-size variants are added for the Q8 four-row wrapper; only wave32 is selected by the new gate. Artifacts, raw process means, exact commands, hashes, rejected runtime snapshots, phase3-only diff and handoff are in `build-dx12-phase0\nv-granite-quants-20260920`. There is no non-RTX5070, model-quality or Vulkan-parity claim.

## NVIDIA combined GEMM, attention and quantized-expert delivery

The final comparison measures all three phases together against the incoming runtime, not a product of separately measured ratios. Baseline DLL SHA256 is `27896ac0e398dfafae22d660da477dd71efcd1ece1294afd6308aec79ab12d9a`; retained DLL is `2b7d65d3e17eba300a32e6781091b3e87d3ddd9833a72a04b25546f069026491`. Both complete runtimes are immutable snapshots, independent of the installed build.

The 48 clean processes cover twelve workloads, each in ABBA or BAAB order with two process means per arm. Each process uses full built-in warmup, three repetitions, two-second repetition delay and ten-second serial cooldown. Arguments are `-b 2048 -ub 512 -ngl 99 -fa on -dev DX120 -r 3 --delay 2 -o jsonl`, with `-p 6144 -n 0` or `-p 0 -n 512`. Both arms clear inherited DX12 variables and set `DX12_NO_AUTOTUNE=1`, `DX12_TUNE_REFRESH=1`, `DX12_FA_PIPELINE=1` and `DX12_QK_NORM_PACKED=1`, matching the local LinAlg benchmark script. No build or other GPU job runs concurrently; profiling is disabled.

| Workload | Before tok/s | Final tok/s | Change |
| --- | ---: | ---: | ---: |
| Phi-3-mini F16 pp6144 | 2804.69 | 3262.90 | +16.34% |
| Phi-3-mini Q4_K_M pp6144 | 2851.03 | 3634.26 | +27.47% |
| Granite F16 pp6144 | 8663.28 | 10498.34 | +21.18% |
| Granite Q4_K_M pp6144 | 8522.32 | 10510.05 | +23.32% |
| Granite Q8_0 pp6144 | 8705.26 | 10665.76 | +22.52% |
| Qwen3-0.6B BF16 pp6144 | 9588.17 | 12732.04 | +32.79% |
| Granite Q8_0 tg512 | 387.71 | 439.21 | +13.28% |
| Qwen3-4B Q4_K_M pp6144 control | 3057.50 | 3056.51 | -0.03% |
| Granite Q4_K_M tg512 control | 443.09 | 443.61 | +0.12% |
| Granite F16 tg512 control | 390.21 | 391.27 | +0.27% |
| Phi-3-mini Q4_K_M tg512 control | 190.24 | 190.31 | +0.04% |
| Qwen3-0.6B BF16 tg512 control | 317.30 | 323.22 | +1.87% |

Unchanged-route controls are not credited as gains. Qwen BF16 decode's baseline process means were310.80 and323.80, versus322.43 and324.02 for the final binary; its apparent mean increase is noise in an unchanged route, not a decode optimization. All samples are retained.

The final installed backend and test executable match phase3's retained runtime. Its root/CBV qualification covers318 route assertions,106 isolated attention assertions and1822 runnable CPU comparisons per mode, with99 unsupported cases excluded. An independent read-only review of the three phase diffs and their interactions found no significant issue; it did not independently execute the tests or certify model quality. Existing F16 experts, D128 attention, AMD/Intel blobs and the qualified precision policies remain preserved as described above.

The local, git-ignored `bench_vulkan.bat` now uses `-fa 1` on all36 `llama-bench` commands. Its nine multimodal CLI commands retain `-fa on`. The installed old Vulkan benchmark parses `on` as false and numeric1 as true; this was established from its emitted JSON, not inferred from throughput. The Vulkan executable was not rebuilt in this campaign, so these DX12-to-DX12 results are not a same-source Vulkan-parity claim.

The retained default gates and opt-outs are documented in each phase. The attention extensions still require the existing exact `DX12_FA_PIPELINE=1` opt-in; the qualified new GEMM and Q8 expert-decode routes need no new switch. Installed binaries remain in `build-dx12-phase0\bin\Release`. Combined snapshots, commands, all samples, hashes, the corrected batch-file copy and handoff are under `build-dx12-phase0\nv-campaign-20260920`. No commit or push was made.

## NVIDIA same-source Vulkan prefill investigation, September 21, 2026

The remaining gap survives a current-source Vulkan build and same-process RTX 5070 selection. At pp6144, Phi-3-mini F16 measures DX12 3262.72 versus Vulkan 8665.94 tok/s (2.66x), and Granite Q4_K_M measures 10502.47 versus 24400.58 tok/s (2.32x). These are averages of two independent process means per API, using ABBA/BAAB, batch2048, ubatch512, full GPU offload and FA enabled. The isolated Vulkan build is `build-vulkan-investigate-20260921`; the installed DX12 and old user Vulkan runtimes were not replaced.

Matched operator inputs expose a precision-policy difference as well as an implementation gap. The following times are medians of four independent process means per API, in microseconds per operation. Each process uses a warmed, synchronized repeated-node graph, so these are amortized operation timings, not individual-dispatch timestamps.

| Operation | Default DX12 / Vulkan us | Default ratio | Explicit F32 DX12 / Vulkan us | F32 ratio |
| --- | ---: | ---: | ---: | ---: |
| Phi F16 QKV, K3072 N9216 M512 | 722.32 / 313.70 | 2.30x | 762.69 / 511.14 | 1.49x |
| Phi F16 gate/up, K3072 N16384 M512 | 1945.18 / 625.54 | 3.11x | 2131.34 / 1009.52 | 2.11x |
| Small F16 dense, K1024 N512 M512 | 31.93 / 19.42 | 1.64x | 31.90 / 31.88 | 1.00x |
| D96 attention, nq512 nkv6144 | 2348.24 / 567.06 | 4.14x | 2344.41 / 694.81 | 3.37x |
| D64 attention, nq512 nkv6144 | 661.21 / 191.25 | 3.46x | 657.24 / 241.95 | 2.72x |
| D128 attention, nq512 nkv6144 | 1583.88 / 407.93 | 3.88x | 1578.92 / 538.31 | 2.93x |
| Granite F16 expert gate/up | 499.07 / 142.31 | 3.51x | 498.65 / 211.60 | 2.36x |
| Granite F16 expert down | 474.62 / 165.75 | 2.86x | 472.69 / 226.76 | 2.09x |

Vulkan's default F16 GEMM/expert routes use F16 accumulation; explicit F32 selects their F32-accumulator shaders. For quantized experts the F32 field does NOT change the observed Vulkan `f16acc` route. Default Q4_K gate/up is 488.44/135.82 us (3.60x), Q8_0 gate/up is 478.36/130.47 us (3.67x), and Q6_K down is 476.93/153.10 us (3.12x), but these are not matched-F32 comparisons. Explicit F32 also does not make every attention arithmetic detail identical: Vulkan forms the row sum from the converted probability matrix, while DX12 sums F32 probabilities before their F16 operand conversion. No lower-precision DX12 policy is proposed.

The expert fixtures use 32 experts, 512 tokens and eight distinct selected experts per token, with deterministic counts109..156 across all32 experts. Gate/up uses broadcast activations `[1024,1,512]`; down uses per-slot activations `[512,8,512]`. Useful work is4.295 GFLOP per expert operation, including all4096 selected-token slots. Cross-API source hashes and ID histograms match. Fourteen full-shape CPU-reference cases per API pass the existing5e-4 NMSE threshold; this is not a model-quality claim.

### Source attribution and bounded next experiments

Granite's observed DX12 flag304 is a wave32 kernel with a128-token x64-output tile, BK16 and four waves. Both operands are staged in LDS; F16 uses the existing LDS accumulator drain, while Q4/Q8/Q6 tall buckets already use the retained register-scatter epilogue. Vulkan's F16 large cooperative-matrix-2 configuration uses128 output channels x256 selected rows, BK64 and256 threads, with smaller row matrices for eligible tails. Quantized expert tiles instead use128 output channels x128 selected rows and BK32 on this build, which has no vector-decode support. Do not swap the token/output axes when comparing the two implementations.

The Vulkan name `matmul_id_subgroup` describes its ID-handling specialization, NOT the scope of matrix arithmetic: `vulkan-shaders-gen.cpp::matmul_shaders` selects `mul_mm_cm2.comp` when cooperative-matrix-2 is enabled, and that shader uses workgroup-scope matrices and tensor-load/decode callbacks. Vulkan still dispatches `count_experts` and resolves selected rows. It is incorrect to attribute its advantage to having no expert metadata machinery.

For attention, Vulkan uses64-query x64-key tiles with128 threads for these three shapes, versus DX12's16-query D96/D128 and32-query D64 tiles. `flash_attn_cm2.comp` keeps the online state in cooperative matrices and uses matrix row reductions and tensor loads. DX12's `flash_attn_pipeline.hlsl` stores scores and PV fragments through shared memory, stages probabilities, and synchronizes between those phases. These are concrete reuse and fragment-movement differences, not a measured decomposition of their individual costs.

Disabling NV cooperative-matrix-2 makes D96 F32 attention about2.92x slower, but makes F16 expert gate/up about21% faster. Thus the attention gap strongly motivates that dataflow comparison, while the expert gap does not require a cooperative-matrix-2-only explanation. A perfectly grouped batched dense expert-work control also helps both APIs, but changes DX12's route to261, activation footprint and row-count distribution. Its time cannot be subtracted from MMID to claim an isolated bucket-construction cost.

Prioritize Granite F16 expert gate/up as the no-dequant control: investigate more output-channel reuse per group while holding BK16, F32 accumulation and bucket semantics fixed, then qualify Q4 gate/up and Q6 down. This differs from the already-rejected BK32-only staging change. Next investigate larger D96 query tiles and fewer shared-memory fragment round trips without changing the current normalization/probability policy. Retain the small F32 dense parity case as a regression control. These are hypotheses for bounded experiments, not implemented or credited speedups.

### Measurement limitations and retained artifacts

The first D96 default DX12 process took5980.47 us versus2336.58..2357.46 us in the other three; all samples remain retained and the table uses the median. Late-cohort Vulkan quantized-down timings also varied substantially. A fresh-process Q6 single-case control gave151.32/151.50 us for default and151.81/151.45 us for the F32 field, using the same shader. Do not interpret the late-cohort slowdown as evidence that Vulkan honored F32 accumulation, or use the anomalous means as hard acceptance thresholds.

Exact cases, commands, input hashes, per-process samples, separate GPU profiles, precision controls, correctness results and source-preservation records are under `build-dx12-phase0\nv-api-investigation-20260921\measurements\HANDOFF.md`. The diagnostic harness copies the existing test runner into ignored scratch space to fix unseeded expert IDs/masks and select the physical GPU within the process; no tracked test code was changed for this investigation. The delivered DX12 DLL remains `2b7d65d3e17eba300a32e6781091b3e87d3ddd9833a72a04b25546f069026491`; the isolated Vulkan DLL is `a5af39a5770f97d76e424f813699609bcdafab7b7b582b613846334c7399dc3d`. No production kernel change, commit or push was made.

## NVIDIA Granite expert output-channel reuse, September 21, 2026

The wider-output experiment wins without changing BK16, F32 accumulation, operand conversion or expert buckets. The existing wave GEMM specialization grows from128 selected-token rows x64 output channels in four wave32 groups to128x128 in eight waves. Each wave still owns eight16x16 F32 accumulators: `LA_NWAVE=8`, `LA_WN=2`, `LA_MT=2`, `LA_NT=4`, `LA_BK=16`. Gathered activation tiles serve twice as many output channels, and the output-block workgroup count halves. Workgroup scheduling and LDS allocation also change, so this is not an isolated measurement of activation traffic or bucket cost.

The existing shader source is unchanged. Four new wave32 specializations cover F16/Q4_K/Q6_K/Q8_0. F16 retains its original LDS epilogue; the quantized variants retain the previous register-scatter epilogue. All incoming generated shader headers are unchanged. Optimized DXIL has256 threads, wave32, eight F32 accumulator matrices and no `alloca`; groupshared allocation is25608 bytes for F16 and17416 bytes for the quantized variants. DXIL does not establish final driver register allocation or absence of machine-code spills.

New flag305 is a narrow default after the existing bucket eligibility check succeeds: RTX5070 PCI0x2F04, physical/blob wave32,32 experts, eight selected experts,512 tokens, K1024/N512 or K512/N1024. It covers F16/Q4_K/Q8_0 and Q6_K down only. Q6_K gate/up stays202; decode and other devices do not qualify. `DX12_LINALG_MMID_WIDE=0` restores304. Host dispatch geometry, blob selection, bucket-consumer classification, support reporting and cached-decision validation include305. This does not change flag304's dimensions or widen the existing format/layout eligibility.

### Matched baseline/candidate results

Final staged-backend measurements use independent ABBA/BAAB processes, the same deterministic full32-expert inputs, F32 accumulation in both arms, batch2048, ubatch512, FA enabled, and three model repetitions per process. All builds and GPU work run serially with10-second inter-process cooldown. Useful expert work remains4.295 GFLOP, including all4096 selected-token slots. Times below are synchronized repeated-graph microseconds per operation, not individual-dispatch timestamps.

| Operation | Baseline process means, us | Wider-output process means, us | Speedup |
| --- | ---: | ---: | ---: |
| F16 gate/up | 498.08, 500.44 | 376.20, 388.00 | 1.31x |
| F16 down | 471.93, 474.01 | 417.69, 421.24 | 1.13x |
| Q4_K gate/up | 489.84, 490.72 | 251.46, 255.19 | 1.94x |
| Q4_K down | 460.18, 455.99 | 254.50, 257.92 | 1.79x |
| Q8_0 gate/up | 478.17, 474.46 | 252.34, 247.95 | 1.90x |
| Q8_0 down | 447.87, 448.95 | 246.44, 246.94 | 1.82x |
| Q6_K down | 478.16, 477.68 | 289.72, 293.60 | 1.64x |
| Q6_K gate/up, unchanged202 | 592.35, 588.48 | 589.03, 592.07 | 1.00x |

| Model, pp6144 | Baseline process means, tok/s | Wider-output process means, tok/s | Change |
| --- | ---: | ---: | ---: |
| Granite F16 | 10485.64, 10489.31 | 11127.42, 11135.46 | +6.14% |
| Granite Q4_K_M | 10470.90, 10488.12 | 13937.96, 13931.44 | +32.97% |

Earlier independent opt-in and default cohorts reproduced the model gains. Explicit-F32 operator controls also retain the gains. F16/Q4 decode controls have no material regression across two balanced cohorts; unrelated dense controls remain within approximately0.2% in cohort averages. No decode gain is credited. This stage measures DX12 against DX12, not a fresh Vulkan comparison.

Sixteen full-shape CPU-reference cases pass per root/CBV mode at the unchanged5e-4 NMSE threshold; maximum observed NMSE is1.5445e-5. Existing route assertions pass331/331 in each mode, including opt-out304, default305, skewed/offset/strided layouts, repeated graphs and raw-reference precision. Quantized cached graphs record nine305 dispatches with eight hits/one miss; F16 records six305 dispatches with two hits/one miss/one rebuild. No new test file was added.

The requested128x128 shape succeeded, so there was no alternative tile sweep or rejected performance variant. An initial host-build command quoting error was corrected before measurements. Artifacts, incremental `stage-only.patch`, input and runtime manifests, per-process results, optimized DXIL and source archives are in `build-dx12-phase0\nv-expert-reuse-20260921`. The stage's qualified runtime is `final\runtime`, backend SHA256 `6f72cba63135695f39e7e7e8a51464dab077816cdf853bfd862a3e33a48c09cc`; it was staged rather than installed while the subsequent D96 experiment proceeded. The original delivered runtime remains the immutable incoming baseline.

## NVIDIA D96 query reuse and intermediate-transfer experiments, September 21, 2026

A narrow Br32/Bc64 D96 route is retained, but simply doubling query reuse was not sufficient. The retained shader also applies the existing D64 four-lane row-partition approach to D96, covering all96 output dimensions and reducing duplicated softmax state. It uses four wave32 groups, two query tiles and six16-wide head blocks; only two waves perform the second PV dimension phase, while all waves reach group barriers. Q/P share6144 bytes after Q is cached in matrix fragments; score/PV workspace is12288 bytes, for18432 bytes total groupshared memory.

F32 QK/PV accumulators, running maxima, denominator, rescaling and output remain. The denominator sums unrounded F32 probabilities; only the P operand is converted to F16 using the unchanged explicit subnormal conversion. The four-lane reduction changes summation order within F32, not the precision policy. PV still stores and reloads through shared memory in the retained shader. The attempts to remove that transfer were measured and rejected.

New flag335 is confined to the existing RTX5070 D96/K6144 pipeline eligibility. Device-aware row selection drives layout/group checks, scratch sizing, packed-mask metadata/cache keys, dispatch geometry and PSO selection; AMD wave64 D96 remains Br16. The existing Br32 packed-mask producer is reused. Cached decisions are revalidated against the selected geometry. `DX12_FA_D96_BR32=0` restores flag333/Br16; the overall route still requires exact `DX12_FA_PIPELINE=1`. All12 preexisting attention shader blobs remain byte-identical, including NVIDIA D64/D128 and AMD wave64.

### Retained and rejected candidates

| Candidate, nq512/nkv6144 | Default operator time versus paired Br16 baseline | Disposition |
| --- | ---: | --- |
| Br32 with original vector-output ownership | +26.75% | Rejected |
| Br32 scalar-held output and coordinate scatter, removing PV shared transfer | +25.68% | Rejected |
| Br64 scalar-held output and coordinate scatter | +10.55% | Rejected |
| Br32 with four-lane row partitions and existing PV shared transfer | -3.62% | Retained |

Plain Br64 exceeded the compiler's32768-byte groupshared limit with46080 bytes. Some matrix-state candidates failed PSO creation with0x8007000E, including persistent Get/Set output after matrix `alloca` removal; the cause is not established and must not be labeled hardware register exhaustion. Scalar-held Get/coordinate-scatter variants executed and passed CPU comparisons but lost performance. Loop-local PV matrices also exposed allocation/PSO issues; the retained D96 specialization declares its fixed PV matrix array outside the key loop and has no matrix `alloca` in optimized DXIL. None of these observations establishes native GPU spill counts.

Failed-PSO attempts that fell back to generic flag108 are retained in the artifacts but excluded from candidate measurements. Initial host-switch insertion and command/evidence-parser errors were corrected before credited results. No fallback timing is presented as optimized-route performance.

### Incremental results against the post-Granite baseline

Independent ABBA/BAAB processes use the same nq512 fixtures, full32-head D96 layout, causal masks and F32 arithmetic in both DX12 arms. Times are synchronized amortized graph microseconds per operation.

| KV length | Br16 default process means, us | Retained Br32 default process means, us | Time change | Explicit-F32 time change |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 328.14, 317.02 | 294.87, 296.60 | -8.32% | -9.29% |
| 1088 | 343.46, 342.92 | 318.23, 317.55 | -7.37% | -6.78% |
| 6144 | 2322.99, 2360.91 | 2268.67, 2245.80 | -3.62% | -4.78% |
| 16384 | 6339.02, 6412.65 | 6313.23, 6212.14 | -1.77% | -4.43% |

| Model, pp6144 | Post-Granite baseline process means, tok/s | Retained D96 process means, tok/s | Change |
| --- | ---: | ---: | ---: |
| Phi-3-mini F16 | 3260.73, 3260.82 | 3292.88, 3292.29 | +0.98% |
| Phi-3-mini Q4_K_M | 3632.42, 3626.64 | 3698.70, 3700.86 | +1.94% |

The earlier reversed-order model cohort measured+1.13% F16 and+1.83% Q4. These modest gains are far smaller than the residual Vulkan attention gap; no parity claim is made. Decode cohorts varied below1%, without a consistent F16 direction or a demonstrated Q4 regression; unchanged eligibility excludes decode. D64/D128 timings varied in both directions with identical shader blobs and are not credited as improvements.

Root and CBV each pass333/333 full existing route assertions and108/108 attention-focused assertions. Fourteen CPU cases per mode cover default/F32, KV1024/1088/6144/16384 and D64/D128 controls, with maximum NMSE0.0003507815 against the unchanged0.0005 threshold. Br16 opt-out comparisons also pass. Existing analytical fixtures cover subnormal probabilities, cancellation, sinks, softcap, masks, tails, offsets and cached mask mutation. Model shader audits record1408 flag335 dispatches,128 unchanged short-context flag179 dispatches and44 Br32 packed-mask dispatches per profiling process.

Artifacts are under `build-dx12-phase0\nv-d96-reuse-20260921`, including rejected source/runtime snapshots, `stage-only.patch`, `source-archive.zip`, full process manifests and evidence. The manually built performance reference is `final\runtime`, backend SHA256 `be6d84f4b2ab7fbba1e1a520c657b80606fe81b62d30ae20c7291463ae937264`. A normal existing CMake build subsequently compiled and linked `ggml-dx12`, `test-backend-ops` and `llama-bench`, staged at `cmake-integration\runtime`, backend SHA256 `63b9127f6652de0becad3474b3828342430b63d47286c935a4b410622a379f1c`. Its17 relevant shader identities match the qualified four expert shaders, new D96 shader and12 existing attention shaders. The CMake-built runtime passes333/333 routes in each binding mode and both Phi prefill smoke runs. The incoming delivered runtime was restored after that integration build pending final balanced delivery measurements.

## NVIDIA expert/query reuse combined delivery

The normal CMake-built combined runtime was compared directly against the original incoming runtime, not by multiplying the incremental phase results. Each workload uses ABBA then BAAB: four fresh-process means per arm, three repetitions per process, full warmup and10-second inter-process cooldown. Settings are pp6144, batch2048, ubatch512, full GPU offload and FA enabled. Both arms clear inherited DX12/Vulkan/probe variables and use `DX12_NO_AUTOTUNE=1`, `DX12_TUNE_REFRESH=1`, `DX12_FA_PIPELINE=1` and `DX12_QK_NORM_PACKED=1`; neither new feature switch needs to be set. All52 processes passed device/runtime guards:48 balanced model measurements and four separate route audits.

| Workload | Incoming tok/s | Combined tok/s | Change |
| --- | ---: | ---: | ---: |
| Granite F16 pp6144 | 10475.76 | 11129.40 | +6.24% |
| Granite Q4_K_M pp6144 | 10513.14 | 13957.66 | +32.76% |
| Granite Q8_0 pp6144 | 10650.76 | 14147.45 | +32.83% |
| Phi-3-mini F16 pp6144 | 3262.46 | 3292.73 | +0.93% |
| Phi-3-mini Q4_K_M pp6144 | 3624.06 | 3691.55 | +1.86% |
| Granite Q8_0 tg128 control | 450.09 | 451.13 | +0.23% |

All five prefill improvements occur in both orders. The Q8 decode control changes+0.72% in ABBA and-0.25% in BAAB with overlapping sample ranges; it is effectively unchanged and no decode speedup is credited. Its audited expert route remains361. Q8 prefill changes304 to305 for3312 expert dispatches across warmup and three repetitions, with36 short-row361 dispatches unchanged.

The D96 default's query-count endpoints were measured separately before delivery. Four explicit-F32 cases use nq256/1024 x nkv1024/6144 with the same32-head layout and causal mask policy. Eight independent processes in ABBA+BAAB confirm333 versus335 and identical runtime identities. Mean times fall194.34->179.26 us,1243.06->1147.59 us,456.72->420.19 us and4283.00->3884.05 us respectively, improvements of7.76%,7.68%,8.00% and9.31%. Every final process beats every baseline process at each shape, so no query-gate narrowing was indicated. One fast final nq1024/nkv6144 sample remains included; the other final samples also beat baseline.

The installed runtime in `build-dx12-phase0\bin\Release` now matches the measured CMake-built candidate across all99 files. Only `ggml-dx12.dll` and `test-backend-ops.exe` differed and were replaced; their prior copies are retained in `nv-reuse-delivery-20260921\before-install`. Installed backend SHA256 is `63b9127f6652de0becad3474b3828342430b63d47286c935a4b410622a379f1c`; incoming immutable baseline SHA256 remains `2b7d65d3e17eba300a32e6781091b3e87d3ddd9833a72a04b25546f069026491`. `DX12_LINALG_MMID_WIDE=0` and `DX12_FA_D96_BR32=0` independently restore the prior expert and D96 routes. Attention still requires the existing `DX12_FA_PIPELINE=1`.

Exact process values, model/runtime provenance, commands and installation hashes are in `build-dx12-phase0\nv-reuse-delivery-20260921`; boundary evidence is in its `d96-boundaries` subdirectory. The old user Vulkan runtime and isolated same-source Vulkan runtime were not changed. All earlier uncommitted work remains preserved; no commit or push was made.
## Barrier-free small-batch matrix multiplication (September 18, 2026)

Flags 390 (Q8_0) and 392 (F16) replace tiled matrix multiplication for the measured SmolVLM2 text-segment shapes. One wave owns an output row, loads eight weights per lane and reuses them across the active input columns. Activations and accumulation remain F32. There is no activation-quantization pass, groupshared staging or group barrier. This changes the data movement rather than substituting another LinAlg tile geometry.

The defaults are restricted to discrete RDNA4 wave64, M in {2, 3, 6, 12}, K576/N192, K576/N576, K576/N1536 and K1536/N576. All tensors must be contiguous with four-byte-aligned offsets, fit 32-bit addressing and have a single outer batch. Bias remains a separate ADD. Explicit F32 precision is supported. `DX12_Q8_SMALL_M=0` and `DX12_F16_SMALL_M=0` restore the previous routes; setting 1 does not broaden eligibility. Single-token generation and large text-prefill batches retain their previous selection.

Balanced Q8 operator controls reduced K1536/N576/M2 from 35.67 to 7.85 us and K576/N576/M6 from 17.84 to 8.80 us. K576/N1536/M12 was much closer, 19.14 versus 18.46 us. These are isolated operator measurements, not whole-model speedups.

### Whole-workload results

Two off/on/on/off experiments used `stalib.jpg`, the prompt `Can you describe this image?`, 480 prompt tokens, 512 generated tokens, temperature zero and seed 1. Both small-batch controls were toggled together. Each arm launched three processes; launches 2 and 3 were pooled, yielding eight warm measurements per format and setting. Attention settings were `DX12_FA_LINALG=1`, `DX12_FA_PIPELINE=1` and `DX12_FA_PV_F16=0`.

| SmolVLM2 format | Previous-route prompt time | Small-batch prompt time | Latency reduction |
| --- | ---: | ---: | ---: |
| F16 | 152.83 ms | 140.92 ms | 7.8% |
| Q8_0 | 156.27 ms | 141.66 ms | 9.4% |
| Q4_K_M | 154.81 ms | 157.98 ms | -2.0% |

Q4 has no demonstrated improvement. Its profile does select flag 390 for a small subset of Q8 K/V matrices, so it is not an entirely unaffected control. Its observed slowdown is small relative to launch-to-launch variation; these measurements do not establish whether it is causal. Neither format reaches the supplied Vulkan warm-prompt results.

An earlier Q8-only experiment contained startup outliers of 1065 and 741 ms. It was repeated rather than using those values to claim a gain. The table uses only the later complete balanced experiments (`round3-mtmd-balanced-*` and `round3-mtmd-repeat-*`).

After the replay correction below, final installed default runs averaged 141.16 ms for F16 and 142.65 ms for Q8 across launches 2 and 3. A separate balanced SmolLM2 Q8 control changed pp6144 and tg512 by less than 0.4%, consistent with their unchanged routes.

### Replay, qualification and rejected controls

Decision-cache identities omit fourth-dimension sizes and tensor offsets. Reusing a cached small-batch flag after changing the fourth dimension could therefore dispatch a single-batch shader on a batched graph. The complete eligibility predicate is shared between initial selection and replay validation. A sequential whole-graph regression seeds a single-batch decision, repeats it, then changes to fourth-dimension activation broadcasting. The Q8 regression failed with NMSE 0.888 before this correction and passes afterward.

The final LinAlg build passes 316 route plus 46 composed assertions in both root-constant and CBV modes. Standard DX12 passes 116 route assertions in each mode. Coverage includes every supported M/shape/type combination, explicit F32 precision, separate bias, opt-outs, unaligned weights, outer batches and decision replay. Both F16 and Q8 multimodal workloads also complete with command replay explicitly enabled; small-batch prompt graphs themselves remain outside the whole-command-list replay fast path.

Granite expert 64x64/BK16 controls produced only approximately 1.4% Q8 and 1.3% F16 long-prefill gains, with short F16 prefill flat and Q4 approximately 3.4% slower. A precomputed expert tile map was effectively neutral: tall geometry measured 20797/20883 tok/s without/with the map, and medium geometry 21105/21027. The medium variants and tile-map machinery were removed.

Both `build-linalg` and `build-dx12` contain `llama-cli`, `llama-bench` and `llama-mtmd-cli` with BoringSSL retained. The F16 wrapper explicitly depends on its shared Q8 shader source in CMake. The qualified attention shader is unchanged. Backend artifacts are saved as `round3-delivery-linalg.dll` and `round3-delivery-standard.dll`:

- LinAlg SHA256: `BEA56BF16FF8259B7676BF4A5E95B92945A6B6BBA34E4E32F5D6F9CFB3FF94FC`
- Standard DX12 SHA256: `BDA8DA3DF9E0218F9F8BDA2ACFC2EA23D79133DF38BE6796B7E28B9EAAB93928`

## Short-K Q4 decoding and Granite Q6 output (September 18, 2026)

RDNA4 wave64 uses a single-wave Q4_K matvec for K1024, N <= 2048 and one input column/outer batch. Flag 394 consumes F32 activations directly for N <= 1024; flag 395 uses Q8_1 activations and integer dots for larger N. Both reuse the existing two-row kernels with wave-only reduction instead of the 256-thread workgroup. The direct path wins at N1024, while integer dots win at N2048. Odd output rows load a valid weight row before the guarded store. Contiguous tensors, four-byte-aligned offsets and bounded addressing are required; the layout guard is also applied when replaying graph decisions. Existing fused gate/up routes retain their selection.

`DX12_Q4K_SHORT=0` restores the old route. Values 1 and 2 force the direct-F32 and integer-dot variants within the same eligible shapes; 3 selects the default shape policy. The existing Q6 packed route now also covers K1024/N32768+ output projections below its previous N65536 threshold. `DX12_Q6K_PACKED_HEAD=0` disables only this extension; `DX12_Q6K_PACKED_MMV=0` disables the whole packed route. Explicit F32 precision remains outside the Q6 route.

Balanced isolated measurements reduced Qwen's Q4 query projection from 5.80 to 3.68 us and Granite's Q4 projection from 3.84 to 2.76 us. Granite's Q6 head measured approximately 53 us with the packed kernel, versus 114 us in the preceding baseline cohort. An existing direct-F32 Q6 alternative measured 55 us on Granite and 247 us on Qwen's larger head, so it was not selected. These hot-cache operator numbers are not whole-model speedups.

The combined off/on/on/off experiment used tg512, five repetitions per process, one-second delay, DX120, and `DX12_FA_LINALG=1`, `DX12_FA_PIPELINE=1`, `DX12_FA_PV_F16=0`. Each table entry averages the two process means.

| Q4_K_M workload | Previous routes | New routes | Throughput change |
| --- | ---: | ---: | ---: |
| Granite-3.0-1B-A400M | 579.52 tok/s | 623.49 tok/s | +7.6% |
| Qwen3-0.6B | 564.93 tok/s | 582.27 tok/s | +3.1% |

With the Q4 route held on, the Q6 head extension separately changed Granite from 598.81 to 625.39 tok/s. Extending the candidate to Q4 K2048 and narrow Q6 projections gave little additional whole-model improvement and was not promoted. No prefill gain is attributed to these decode-only changes.

Artifacts are `architecture-short-q4-*`, `architecture-q6-head-*`, `architecture-decode-combined-*` and `architecture-extra-*` in the session directory. Route coverage includes both Q4 variants, odd rows, fused bias, larger-shape and batch fallbacks, default/opt-out selection, whole-graph fourth-dimension replay, and Q6 head tails.

LinAlg passes 349 route plus 46 composed assertions, and standard DX12 passes 149 route assertions, with both root constants and `DX12_PARAM_CBV=1`. The mode lookup uses a fresh environment value while recording so test-controlled replay does not retain an invalidated environment-string pointer. Single-output numerical fixtures were replaced with multi-output odd tails because relative NMSE against the CPU's quantized-activation reference becomes unstable near a single cancelled dot product; the tolerance was not relaxed.

Ten 512-token chunks of the existing varied-text corpus were evaluated with `-ub 1` to exercise decoding rather than batched GEMM. Previous/new perplexity was 5.3239/5.3318 for Granite and 8.3790/8.3762 for Qwen3-0.6B. These small sample changes do not establish bitwise equivalence or a general quality guarantee. The direct-F32 route intentionally removes activation quantization, while the integer-dot route retains it.

## Register-tiled F32 router (September 18, 2026)

Flag 396 replaces the generic scalar GEMM for the Granite router shape: K1024/N32, 32..512 input columns, contiguous F32 tensors and a single outer batch on discrete RDNA4 wave64. Each wave computes four tokens by eight output channels. Lanes load four consecutive K values, reuse the input and weight vectors across 32 F32 partial sums, and reduce each result within the wave. There is no activation conversion, LDS staging or group barrier. The output stays F32 and explicit F32 precision is supported. Bias remains a separate operation. `DX12_F32_ROUTER=0` restores the generic kernel.

The first candidate used a 16x32/BK32 LDS tile with coalesced loads and measured 42.7 us versus 48.8 us for the old kernel. It was replaced rather than promoted. The register-tiled candidate measured 13.01 us for the same K1024/N32/M512 operation; the earlier Vulkan measurement was 11.10 us. The rejected shader is retained only as `architecture-router-lds-candidate.hlsl` in the session artifacts.

Balanced off/on/on/off pp6144 runs, five repetitions per process and one-second delay, produced the following means with the other backend settings held fixed:

| Granite format | Generic router | Register-tiled router | Throughput change |
| --- | ---: | ---: | ---: |
| F16 | 19697.56 tok/s | 20428.55 tok/s | +3.7% |
| Q8_0 | 20672.61 tok/s | 21328.88 tok/s | +3.2% |
| Q4_K_M | 18270.96 tok/s | 18874.78 tok/s | +3.3% |

Ten 512-token chunks of the existing varied-text corpus, evaluated with `-ub 512` to exercise the prefill router, changed Granite Q8 perplexity from 5.2817 to 5.2842. F32 summation order changes, so no bitwise-equivalence claim is made. Whole-model and numerical logs are `architecture-router-model-*` and `architecture-router-ppl-*`.

The combined delivery passes 365 route plus 46 composed assertions on LinAlg and 165 route assertions on standard DX12, in root-constant and CBV modes. Review identified a direct-Q4 interaction with the opt-in RMS/Q8_1 fusion: its skip-F32 proof previously classified consumers only by weight type and could omit an activation read by flag 394. Direct-only and mixed-consumer graphs reproduced NMSE near 2.0. The consumer proof now shares the short-K selection helper; mixed consumers preserve the F32 materialization. Both regression graphs pass with the fix. Router coverage includes token tails, explicit F32, separate bias, layout/batch fallbacks and fourth-dimension decision replay.

## Native SmolLM2 Q8 attribution (September 18, 2026)

Radeon Developer Panel captures now provide decoded RDNA4 dispatches and native ISA, without either backend's per-node timing logger. The workload is SmolLM2-135M Q8_0 pp6144 on RX 9070 XT, using the committed `cd7886328` DX12 binaries and the existing Vulkan baseline. These captures are diagnostic runs, not throughput acceptance runs.

The RDP CLI requires `--rgp-render-op-count` separately from `--rgp-auto-capture dispatch:<start>`. The first files named `architecture-smol-*-full.rgp` actually requested one render operation and must not be treated as full prompt passes. Replacement `architecture-smol-*-window.rgp` files explicitly request 5500 DX12 and 7000 Vulkan operations. The native decoder finds 5508 and 7053 dispatch records per shader engine, respectively, with no reported lost-data or incomplete-wave warnings. Those are different windows, not matched complete passes.

Compiled resources for the dominant dense matrix pipeline are:

| Native property | DX12 (`co_7.elf`) | Vulkan (`co_1.elf`) |
| --- | ---: | ---: |
| Wave width | 64 | 64 |
| Workgroup threads | 256 | 128 |
| Metadata VGPR count | 89 | 64 |
| LDS bytes | 16384 | 11264 |
| Scratch bytes | 0 | 0 |
| Matrix accumulator instruction | `v_wmma_f32_16x16x16_f16` | `v_wmma_f16_16x16x16_f16` |

These counts are native code-object metadata, not calculated occupancy limits. The difference includes geometry, accumulator precision and compiler scheduling; it does not isolate the benefit of any one of them. Both kernels use LDS and neither spills to scratch. In these windows the matrix pipelines appear 2699 and 2898 times, respectively.

Separate instruction-enabled captures request 64 operations and decode 77 DX12 and 117 Vulkan dispatches. All 2693405 DX12 and 951776 Vulkan instruction records resolve to the captured ISA. Within the dense matrix pipeline, wait instructions account for 5803191 of 12294687 summed instruction-duration cycles on DX12 (47.2%) and 3698866 of 11813974 on Vulkan (31.3%). Barrier waits add 1679465 and 1181522 cycles, respectively; they are classified separately from the wait category. The DX12 D64 attention pipeline also spends most sampled instruction-duration cycles on waits: 93220672 of 123858256. These are sums across traced waves, not kernel wall times. Attention KV extents and dispatch mixes are not matched between these captures, so their absolute cycle totals must not be compared as speedups.

Hardware-counter interpretation is deliberately narrower. The fixed-size SPM ring retains only the tail of the longer captures. In the DX12 window its retained timestamps begin after the final decoded shader activity, leaving most GPU counters at zero. Those zeros are not evidence of low cache traffic or low LDS contention, and no DX12/Vulkan cache-hit or bandwidth comparison is claimed from these files.

The concrete next experiment is a matched-shape, two-wave dense GEMM with cheaper operand staging and explicit precision controls. Native evidence supports investigating load/LDS waits and resource footprint rather than register spills or a presumed faster matrix instruction. Attention load scheduling remains a separate candidate; the captures do not justify promoting unsafe half carry or assigning a percentage of the whole-model gap to it. The RDF parser, decoded reports, disassembly and captures are retained as `architecture_native_smol.py`, `architecture-smol-*-window.native.json` and `architecture-smol-*-instructions*` in the session artifacts.

## Expert-major Granite FFN (September 18, 2026)

The retained path changes the complete gate/up/SwiGLU/down/weight chain, not an isolated GEMM. Gate and up gather their activation rows directly from the original token-major input while writing compact expert-major outputs. SwiGLU preserves this ordering. Down reads the compact activations, multiplies each result by its router weight and scatters directly into the original weighted token-slot output. The existing deterministic expert sum is unchanged. There is no separate activation-copy pass, extra activation scratch, intermediate down-output materialization or standalone router-weight MUL.

Flags 399 and 400 use dedicated expert-major 32x32 and 128x64 shader variants. Ordinary bucket shaders retain their previous code path. Matrix inputs remain F16-converted with F32 accumulation; this is not a half-accumulator experiment. Automatic selection is limited to the measured Granite prefill shape: K1024/H512/D1024, 32 experts, eight selected experts and 512 tokens on discrete RDNA4 wave64. `DX12_MOE_EXPERT_MAJOR=0` disables it. `=1` also permits other structurally eligible shapes for experiments; it does not bypass layout, device or numerical guards. Decode does not run the automatic chain planner.

Eligibility requires an adjacent MMID/MMID/SwiGLU/MMID/MUL chain, identical input and routing IDs, private contiguous F32 intermediates, supported bucket kernels and default matrix precision. Intermediate outputs, extra consumers, bias/adapters, unsupported layouts and tensor dumps fall back before changing layout. The weighted destination cannot overlap any live down/epilogue input because it is written one graph node earlier. The bucket order remains pinned through down, including command-list resets. Participating graphs bypass decision/command replay. Hidden router-weight reads are included in write-after-read tracking even though the separate MUL is skipped.

Two complete implementations were rejected before retaining this one. Explicitly gathering activations into 16 MiB scratch and scattering after down changed Granite Q8 pp6144 from 21563.58 to 20746.31 tok/s (-3.8%). Disabling replay on both sides still lost 3.9%, so replay alone did not explain the regression. Fusing the weighted down epilogue but retaining that activation copy reduced the loss to 1.8%. Removing the copy, while keeping the intermediate FFN tensors expert-major, produced the retained improvement. The rejected source variants are archived only in the session artifacts.

Balanced off/on/on/off runs, five repetitions per process, one-second delay and the new router held on:

| Granite format | Existing FFN | Expert-major FFN | Change |
| --- | ---: | ---: | ---: |
| F16 | 20456.68 tok/s | 21201.88 tok/s | +3.6% |
| Q8_0 | 21531.05 tok/s | 22228.61 tok/s | +3.2% |
| Q4_K_M | 18908.37 tok/s | 19468.08 tok/s | +3.0% |

A separate balanced run compared both the router and FFN improvements disabled against production defaults:

| Granite format | Previous router/FFN | New defaults | Combined change |
| --- | ---: | ---: | ---: |
| F16 | 19749.87 tok/s | 21164.97 tok/s | +7.2% |
| Q8_0 | 20831.58 tok/s | 22222.85 tok/s | +6.7% |
| Q4_K_M | 18303.48 tok/s | 19479.82 tok/s | +6.4% |

Ten varied-text c512 chunks with ub512 produced identical printed old/new perplexities: F16 5.2763, Q8_0 5.2842 and Q4_K_M 5.3307. This is a limited numerical check, not a general quality or bitwise-equivalence claim. LinAlg passes 396 route and 46 composed assertions in root-constant and CBV modes. Full-chain cases cover 12 weight formats, both tiles, token/K/N tails, public-intermediate fallback, the qualified default, unqualified default fallback, early destination reuse and router weights overwritten immediately after the epilogue. Evidence is retained in `architecture-expert-major-*` and `architecture-router-ffn-combined-*`.

## Second parity round: wave-only decode (September 18, 2026)

The supplied cd3bb760e report confirms the previous work: Granite pp6144 improves 6.9-7.6%, Granite Q4_K_M tg512 improves 7.2%, and Qwen3-0.6B Q4_K_M tg512 improves 4.5% against the 46e485424 report. Across all 36 matched text rows, the largest remaining relative losses against the supplied Vulkan report are Falcon BF16 prefill (27.9%), Smol F16/Q4 prefill (21.6%/20.5%), Granite Q4 prefill (17.9%) and Qwen0.6 Q4 prefill (16.2%). These percentages are throughput deficits, not the larger speedups required to reach parity.

The selected work areas are shared small-model prefill, Falcon full-offload memory behavior, and the remaining Granite/Qwen decode kernels. The retained decode changes remove LDS staging and group barriers rather than changing accumulation precision:

- F16 expert flag 407 loads eight adjacent K values per lane with packed weight loads, reuses activations across four output rows, and reduces entirely within wave64. It preserves the existing expert IDs, weighted epilogue, F32 accumulation and flag-360 eligibility. `DX12_MOE_F16_WAVE=0` selects the previous implementation.
- Q4 gate/up flags 408/409 decode headers in registers and reuse each activation vector across two gate rows and two up rows. Flag 409 also retains the RMS fold. Automatic selection is limited to discrete RDNA4 wave64, K1024/N3072, with the existing fusion/layout guards. `DX12_Q4K_GLU_WAVE=0` restores the old shaders; `=1` permits other eligible output widths.

Fresh off/on/on/off tg512 runs use five repetitions per process and two-second delays. Vulkan was rerun locally with the same model arguments:

| Workload | Previous DX12 | New DX12 | Change | Fresh Vulkan | Remaining deficit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Granite F16 | 367.63 | 420.19 | +14.3% | 437.20 | 3.9% |
| Qwen3-0.6B Q4_K_M | 584.42 | 618.76 | +5.9% | 670.75 | 7.8% |

Separate intrusive profiles confirm the intended kernels execute. Granite's two F16 expert shape groups drop from approximately 1.365 to 1.059 ms in the sampled decode graph. Qwen's fused gate/up group drops from approximately 0.324 to 0.225 ms. Profiled throughput is not used for acceptance.

Root-constant and CBV runs pass 447 route assertions and 46 composed assertions each. Coverage includes expert broadcast, weighted output, K/N tails, misaligned-view fallback, Q4 odd rows, batches, opt-outs and the replay-enabled RMS fold. Ten c512/ub1 varied-text chunks exercise decode, not the prefill kernels: Granite PPL is 5.2750 old versus 5.2738 new; Qwen is 8.3762 versus 8.3896. These are small but nonzero differences from F32 reduction/reassociation, not a bitwise-equivalence or general quality claim. Results are retained in `parity2-decode-*` and `parity2-profile-new-*`.

### Matched F32/F16 accumulator experiment

Three dense GEMM geometries were compiled with both F32 and F16 accumulators. All use BK16, F16 operands, the same staging within each pair, F32 LDS epilogues and F32 global outputs. The F16 result is explicitly cast to F32 before the epilogue. Thus changing accumulator precision alone does not reduce the input/output VRAM byte count.

Smol pp6144 screening, three repetitions per process:

| Geometry | Smol F16: F32 accumulator | Smol F16: F16 accumulator | Smol Q8: F32 accumulator | Smol Q8: F16 accumulator |
| --- | ---: | ---: | ---: | ---: |
| 64x64, two waves | 44163.17 | 43605.87 | 41025.57 | 40200.38 |
| 128x64, two waves | 34932.26 | 35244.10 | 33409.47 | 33272.01 |
| 128x64, four waves | 43266.66 | 41822.76 | 40818.27 | 39458.61 |

Production controls were approximately 45k tok/s for F16 and 48k tok/s for Q8. None of these blanket geometry replacements wins. Half accumulation is neutral or slower within each pair except a 0.9% difference in an already substantially slower F16 configuration. No half-accumulator path is retained or enabled. This rejects this particular implementation, not every possible mixed-precision/staging design. The screening logs and rejected prototype patch are session artifacts under `parity2-probe-*` and `parity2-matched-accumulator-prototype.patch`.

A separate raw-prefetch Q8 kernel retained production's 128x64/four-wave/BK16 geometry and F32 accumulation, moving conversion into the following loop iteration. Balanced Smol Q8 pp6144 was neutral: 48194.10 versus 48170.67 tok/s. A fresh instruction capture identified the actual candidate as `co_24`: next-tile loads at 0x5c8-0x5fc still encounter load waits at 0x608/0x614 before the first current-tile WMMA at 0x6d8. Subword register moves prevent the intended overlap. VGPRs increase from 89 to 93; LDS remains 16384 bytes and scratch remains zero. A second implementation using full-width raw payloads also fails to establish a meaningful gain (48150.53 versus 48328.61 tok/s). Neither is retained. These experiments did not close the Smol prefill gap; the first native trace proves that moving conversion in HLSL did not by itself produce load/compute overlap in machine code.

### Falcon memory accounting and split-KV reservation

All three memory-reporting APIs now query current DXGI budgets instead of returning the startup snapshot. Successful zero-budget/zero-headroom results remain zero. Initialization seeds a descriptor-based fallback before the first query; later query failures use the immutable startup snapshot, including a legitimate zero snapshot. No shared cached fields are written by live queries.

Preallocation and dispatch now use the same split-count policy and environment controls. Falcon's default partial-buffer reservation decreases from 19169280 to 9584640 bytes (18.28 to 9.14 MiB). Pipeline-eligible graphs still reserve fallback capacity because pipeline creation can fail. Existing resource-draining and capacity clamps are preserved. This is a bounded allocation correction, not a claim that 9 MiB explains the full performance deficit.

`DX12_MEMORY_LOG=1` enables budget, usage, reservation, allocation and scratch diagnostics without adding GPU waits; unset it to disable logging. In the sampled full-offload Falcon run, local budget is 15299694592 bytes and usage is 15100334080 bytes, leaving about 190 MiB headroom. Nonlocal usage is 848830464 bytes, while explicitly tracked transfer staging is 821624832 bytes. This is tight budget evidence, but the nonlocal total is not proof that model weights were paged out.

`DX12_BUFFER_MAX_1G=1` is a diagnostic 1-GiB allocation-chunk cap; unset or `0` preserves the previous maximum. Identical full-offload pp6144 off/on/on/off runs, three repetitions per process, measured 793.79 versus 823.07 tok/s (+3.7%). The logged occupied bytes were almost unchanged. The control remains opt-in: this does not establish a universal resource-size policy or close Falcon's Vulkan gap. No automatic layer-offload reduction, eviction or residency-priority change is introduced.

All three live memory APIs were exercised around a 128-MiB allocation and release; each reported the same corresponding decrease and recovery in free memory. Existing D128 attention cases pass with pipeline selection, forced fallback and split-policy overrides. Runtime evidence is retained in `parity2-live-memory-api.txt`, `parity2-fa-*`, `parity2-falcon-memory-*` and `parity2-falcon-cap-final-*`.

## Explicit two-stage prefill pipeline (September 20, 2026)

The retained dense GEMM uses a separate prologue, two K stages per steady-state iteration and an explicit final pair. Each stage has a distinct full-width raw register payload and a constant LDS slot. Conversion and publication remain separate from fetching. Unlike the previous raw-prefetch experiment, this arrangement produces some actual load/WMMA overlap in the native shader. F16 operands, F32 accumulation, F32 output, fused bias and the previous tile geometries are preserved.

Flags 410/411 use 128x64 output tiles, BK16 and four wave64 groups for Q8_0/F16. Flag 412 uses the existing F16 32x32/BK32/two-wave geometry. These replace eligible flags 269/207/209; they do not override generic fallback, precision, format or device selection. Automatic routing is limited to discrete RDNA4 wave64, 512-token microbatches and the measured Smol shapes: K576/N192, K576/N576, K576/N1536 and K1536/N576. Explicit tile, tile-minimum-K and group controls suppress automatic selection. Full tiles, K divisible by 64, four-byte-aligned weight offsets/strides and contiguous, 16-byte-aligned F32 activations are required. Replay revalidates these layout constraints.

`DX12_LINALG_Q8_PIPELINE=0` and `DX12_LINALG_F16_PIPELINE=0` restore the previous kernels. `=1` permits other eligible shapes for experiments. F16 `=2` selects only the wide variant and `=3` only the small variant. They are not blanket defaults for other models or token counts.

Balanced off/default/default/off runs use five repetitions per process, two-second delays, FA enabled and the RX 9070 XT. FA settings are held at `DX12_FA_LINALG=1`, `DX12_FA_PIPELINE=1`, `DX12_FA_PV_F16=0`. The final table uses the shared Q8/F16 implementation and automatic shape selection:

| SmolLM2-135M workload | Previous DX12 | New DX12 | Change | Fresh Vulkan | Remaining throughput deficit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 pp6144 | 48266.92 | 50283.29 | +4.2% | 55416.90 | 9.3% |
| F16 pp6144 | 45156.93 | 47593.26 | +5.4% | 56507.31 | 15.8% |
| Q8_0 pp512 | 63207.38 | 66866.50 | +5.8% | 92793.18 | 27.9% |
| F16 pp512 | 60326.33 | 63681.00 | +5.6% | 95166.63 | 33.1% |

Short pp512 measurements have larger run-to-run variance than pp6144. Decode is not selected by these defaults; tg128 controls are 1038.55/1050.03 tok/s for Q8 and 990.00/987.35 for F16, within the observed variation. The earlier Q8-only implementation measured +5.1% at pp6144. F16 screening separates the wide-only and small-only improvements, approximately +2.6% and +2.3%, with both reaching +5.5%. These are whole-model gains, not inferred speedups from instruction counts.

The fresh Vulkan runs use the existing f48b338bf/build-11502 binary and the same model and benchmark arguments. Device enumeration places the RX 9070 XT at Vulkan1 in these processes; Vulkan0 is the integrated GPU. Initial runs against Vulkan0 are excluded. Always check the physical adapter rather than assuming DX12 and Vulkan ordinals match.

Instruction-enabled captures identify actual dispatched code objects, with 77 dispatches in each of four decoded streams and no decoder warnings:

| Kernel | Code object | VGPRs | SGPRs | LDS bytes | Scratch bytes | Threads |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q8 wide | pipeline3-pingpong/co_40 | 77 | 42 | 16384 | 0 | 256 |
| F16 wide | pipeline3-f16-pingpong/co_42 | 70 | 44 | 16384 | 0 | 256 |
| F16 small | pipeline3-f16-pingpong/co_43 | 56 | 44 | 10240 | 0 | 128 |

The Q8 production comparator used 89 VGPRs. In the new Q8 kernel, activation loads at 0x88c/0x898 span WMMA at 0x8b0-0x8f8 before a later load wait at 0x964. F16 wide loads at 0x6f4-0x70c span WMMA at 0x71c-0x754 before waits at 0x774/0x788; F16 small loads at 0x65c-0x674 span WMMA at 0x688-0x6a0 before waits at 0x6c0/0x6c8. Other loads still wait before compute. This establishes partial overlap, not complete latency hiding, and reduced register pressure also contributes. Expanded steady-state/tail code increases static WMMA counts without increasing the arithmetic required for a given K. Capture throughput is not used for acceptance.

LinAlg root-constant and CBV runs each pass 516 route assertions plus 46 composed assertions; the non-LinAlg build passes 216. Cases include all variants, batched/repeated weights, fused bias, aligned and misaligned activation views, explicit opt-outs/tile controls, K/N fallback and replay. Ten varied-text c512/ub512 chunks produce identical printed old/new PPL: Q8_0 6.4570 and F16 6.4609. This is a bounded numerical check, not a general quality or bitwise-equivalence claim. Native captures, balanced runs and numerical evidence are retained in the session artifacts under `pipeline3-*`.

## NVIDIA integration of incoming AMD work, September 22, 2026

Remote `048768418` adds six commits after the common base `203d2cfb2`; local `8d1df3f02` adds the three NVIDIA commits above. The merge retains both implementations, including NVIDIA expert flag305, D96 flag335 and Q8 expert decode flag361. Conflicts in the host, CMake, route tests and this log were resolved without dropping either campaign's coverage or changing numerical tolerances.

Two semantic interactions needed explicit device guards, beyond removing conflict markers. The incoming flag407 override sits inside the F16 expert-vector eligibility which the NVIDIA branch had widened; it must not select a wave64 shader merely because NVIDIA supports flag360. Likewise, `DX12_LINALG_MMID_BUCKET=1` widens NVIDIA bucket eligibility, so the expert-major matcher must explicitly restrict its wave64 variants rather than rely on the old implicit AMD restriction. Existing fixtures now cover both forced-override fallbacks. A later native-wave32 port would require its own qualification, not just removal of these guards.

The incoming small-M and router shaders use one wave per reduction. Running their 64-thread form as two NVIDIA wave32s would not combine both partial sums. The explicit dense pipeline also hardcodes wave64 load geometry. Expert-major layout is potentially portable, but its incoming weighted epilogue exists only in the LDS drain; the NVIDIA quantized register-scatter epilogue needs equivalent weighting before that specialization could be enabled. Replacing NVIDIA's wider expert tile with the incoming narrower tile would confound any layout comparison.

### Initial merged candidate and regression hold

The normal CMake build, retaining BoringSSL, produced staged backend SHA256 `2679e10e08fe33560b0e9e771f190ddb05fd72d7dc7abfb60413b6e15afe747a`. The installed `63b9127f6652de0becad3474b3828342430b63d47286c935a4b410622a379f1c` runtime was preserved across all99 files. The candidate was not installed: matching routes and CPU accuracy were insufficient to accept measured decode losses.

Fresh-process comparisons used pp6144 or tg128, batch2048, ubatch512, full offload, FA enabled and r3. Initial ABBA cohorts were followed by BAAB for apparent losses. Prefill changes across Granite F16/Q4/Q8, Phi F16/Q4, Smol F16/Q8 and Qwen0.6 BF16/Q4 ranged from -0.25% to +0.08%. Granite F16, Granite Q8 and Smol F16 decode changed -0.95%, -2.36% and -1.60%, respectively, with losses in both orders and disjoint process-mean ranges. Qwen0.6 Q4 decode changed -0.77% with overlapping ranges. No gain is credited to the one-cohort Granite Q4 decode result.

Eleven paired model audits have identical shader identities, routes and dispatch counts. The existing attention and flags305/335/360/361 blobs are byte-identical. Of70 changed existing generated blobs,69 are Q4 matvec variants and one is Q6 packed wave64;23 changes are wave32 Q4 variants. This excludes an obvious shader takeover for the F16/Q8 losses but does not by itself establish a CPU cause.

A four-runtime crossover subsequently located a Granite Q8 penalty on the backend side: swapping only the backend cost0.77% with the old remaining libraries and1.69% with rebuilt libraries, whereas changing the remaining libraries with the old backend measured+0.16%. Their executable code sections match; other binary sections differ. GPU graph timings were effectively unchanged (Granite1.931->1.924 ms, Smol0.984->0.988 ms). Disabling decision replay did not remove the deficit; forced command replay behaved identically for Smol and remained unavailable to Granite's ARGSORT graph. CPU preparation/inter-token work remained a hypothesis, not an attribution to a particular statement. Smol crossover samples were inconclusive and are retained.

### Parameter-mode coverage correction

`DX12_PARAM_CBV` is presence-based: setting it to `"0"` still enables CBV. Earlier artifacts named `cbv0` and `cbv1` in this integration therefore exercised CBV twice. Their results remain valid CBV evidence, but the label is not evidence of root-constant coverage. The same limitation applies to any earlier campaign artifact that inferred root mode solely from this value.

True root-mode runs were added with `DX12_PARAM_CBV` absent and `DX12_COMMAND_REPLAY=0`. The merged runtime passes349/349 full route assertions and108/108 attention assertions in this mode; the explicit CBV1 runs also pass. Seventeen deterministic CPU cases per runtime and binding mode cover odd-output Q4/Q6 matvecs, full32-expert/eight-selected Granite operations and D96 attention. Baseline and merged error vectors match, with maximum NMSE0.00012526547 below the unchanged0.0005 threshold. These bounded checks do not establish model-level numerical equivalence or validate AMD execution on NVIDIA hardware.

Initial build, raw samples, source/runtime provenance and crossover evidence are under `build-dx12-phase0\nv-incoming-048768418`. Correct mode labels and additional root evidence are in `QUALIFICATION-LABEL-CORRECTION.txt` and `root-correction`; attribution data are in `attribution`. No historical root coverage is inferred from incorrectly labeled runs, no samples were discarded, and Falcon BF16 was excluded because of the previously observed device-hang risk.

### Retained host-overhead remedy

The disabled diagnostic and planner paths must not tax unrelated graphs. Three hot environment reads now use the existing `DX12_GETENV` cache and refresh hook: memory logging, the RMS/Q8 skip-F32 override and expert-major selection. The graph-local expert-step map is created only after a valid chain and all required pipelines are ready. Inactive graphs neither allocate the map nor look up each node in it. Scratch failure discards the map; active graphs retain the same matching, lifetime, bucket ordering, replay exclusion and dispatch behavior.

Caching alone recovered the Granite controls but did not consistently recover Smol F16. A lookup-only guard was inconsistent across reversed cohorts. Lazy map creation improved both Smol comparison orders; the final combined remedy was then compared independently with both the original and unfixed merged runtimes. This supports removing inactive-path overhead as a category, not attributing a measured number of cycles to one allocator or environment call.

| Decode control, tg128 | Original63b9 tok/s | Unfixed merge tok/s | Host remedy tok/s | Remedy vs original |
| --- | ---: | ---: | ---: | ---: |
| Granite Q8_0 | 449.956 | 444.954 | 449.301 | -0.15% |
| Smol F16 | 826.397 | 811.049 | 826.208 | -0.02% |
| Granite F16 | 398.581 | 394.903 | 398.484 | -0.02% |

Each arm has four fresh processes with r3, using symmetric three-arm orders and the same tg128 settings. Residual differences are below the observed process spread; exact zero regression is not established. All2612 generated shader headers are unchanged. True-root and CBV runs each pass349/349 route assertions and108/108 attention assertions;34 additional CPU cases match the original error vectors, with maximum NMSE0.00012526547. Actual model routes are unchanged.

The normal-CMake-built remedy is staged under `nv-incoming-048768418\remedy\cache3-lazymap`, backend SHA256 `66afe7ed5a8f953513695777698dc98a618238474bb95be0a79dda61de0b12a6`. `remedy\handoff.txt` and its source snapshots preserve all intermediate candidates and126 performance processes. This stage restored measured baseline-level decode before testing any new NVIDIA kernel. It was not installed at this point.

### Native-wave32 candidates

The router and small-M kernels were specialized for one native wave32, changing both group size and K progression rather than launching two independent partial reductions. Admission remains limited to the measured discrete RTX5070 device and the existing narrow shapes. AMD wave64 shader binaries remain byte-identical.

| Candidate | NVIDIA result | Decision |
| --- | --- | --- |
| F32 router396, K1024/N32/M32,33,128,512 | Isolated latency -75.54% to -80.88%; same-DLL Granite prefill +3.08% F16, +4.09% Q4, +3.85% Q8 | Retain |
| Small-M390/392, existing Smol shapes | Thirty endpoints improve10.12% to83.99%; K576/N1536/M12 loses46.65% F16 and50.64% Q8 | Exclude that endpoint on NVIDIA, including explicit opt-in |
| Native-wave32 F16 expert407 | Corrected actual360/407 comparisons lose26.75% gate/up and36.98% down; model decode has no gain | Reject NVIDIA port; retain original AMD-only shader and wiring |

Existing flag305 expert geometry and flag335 attention remain unchanged. The incoming AMD expert-major layout and dense two-stage pipeline remain AMD-only; neither is automatically promoted based on AMD timings.

Cached SmolVLM2 image trials used the native CLI, `stalib.jpg`, prompt `Can you describe this image?`, temperature0/seed1, FA enabled and512 generated tokens. Four fresh processes per arm in ABBA+BAAB reduce prompt evaluation from224.34 to216.24 ms for F16 (-3.61%) and218.79 to211.11 ms for Q8 (-3.51%). Image encoding, startup and generation are reported separately and are not credited to small-M. Shader audits record1805 optimized small-M dispatches in each enabled workflow.

All eight F16 greedy outputs match. Q8 outputs are repeatable within each arm but differ between arms. Passing operator tolerances is not sufficient to claim model-quality equivalence; therefore NVIDIA Q8 small-M is experimental and requires explicit `DX12_Q8_SMALL_M=1`. Its default remains the previous route. AMD defaults are unchanged. NVIDIA F16 small-M keeps its qualified default, and `DX12_F16_SMALL_M=0` disables it. `DX12_F32_ROUTER=0` restores the old router.

The interim normal-build backend `44ff1275fa3eb9fdf0e0a0ee58685de24f3049c47e1b684e470eca380540c068` passed413/413 route assertions and108/108 attention assertions in both true-root and CBV modes, plus144 signed deterministic CPU comparisons. Router NMSE improved from3.83e-13 to6.13e-14; small-M worst NMSE was1.5167e-5 under unchanged tolerances. These tests cover explicit F32 precision, offsets, tails, bias, batches, replay and opt-outs. This interim build still defaulted Q8 small-M on; it is not the final conservative-default build.

Direct interim44ff versus63b9 prefill results were+3.15%/+3.65%/+3.30% for Granite F16/Q4/Q8, with other prefill controls between-0.21% and+0.25%. Two decode controls remained negative after balanced repeats: Granite F16 -0.61%, Smol F16 -0.80%. An earlier source-equivalent build had positive results on those controls, but that does not erase the negative measurements or establish no regression. This triggered a separate fixed-size, longer-decode qualification before any installation. Both cohorts, including all contrary samples and measurement corrections, remain under `nv-incoming-048768418\native-wave32`.

### Conservative-default delivery and measured uncertainty

Final backend SHA256 `79de8616909c0c3f7c954f222bdb07221c132cd7665e0b30e2f30d1baa34a5c1` was built with the existing CMake/BoringSSL configuration. Q8 small-M is now opt-in only on NVIDIA. The cached-environment replay fixture was corrected to exercise defaults: changing an environment variable after its call-site static cache was initialized does not replace the cached value while refresh is disabled. Forced-on positive coverage remains in the normal suite and in fresh-process probes. The initial411/413 fixture failure and its rejected runtime remain in the artifacts.

The final true-root and CBV suites each pass413/413 route assertions and108/108 attention assertions. Ninety-six additional signed Q8 CPU comparisons cover16 shapes, default/off/forced-on settings and both binding modes, including three graph executions per case. All default/off Q8 cases retain flag214; forced-on390 still excludes the losing K576/N1536/M12 endpoint. Generated headers and the generated master match the preceding shader-qualified build.

A predeclared three-arm decode study compared original63b9, host-remedy66afe and final79de. It used eight fresh process means per arm/model, r5/tg512,72 processes total, in symmetric block orders. All360 within-process samples were retained. The independent unit is a process mean. Paired log-throughput estimates and unadjusted95% Student-t intervals use eight same-model blocks (df7); these descriptive intervals assume independent block differences and are not simultaneous bounds.

| Decode workload | Final vs original | Paired95% interval |
| --- | ---: | ---: |
| Granite F16 | +0.016% | -0.229% to +0.262% |
| Smol F16 | -0.381% | -1.398% to +0.647% |
| Granite Q8_0 | -0.053% | -0.462% to +0.357% |

No contrast demonstrates a decode regression in this study. This is not proof of exact equivalence: the Smol interval does not exclude a slowdown greater than0.5%. Its valid slow process remains included. Comparisons against66afe also include zero. The predeclared adverse-result trigger was not met, so no speculative source adjustment or selective longer repeat followed. Separate decode audits show no390/392/396 dispatches; this does not prove host eligibility costs zero. The earlier negative tg128 cohorts remain historical results, not silently relabeled or discarded.

Exact-final-runtime ABBA comparisons against63b9 improve Granite pp6144 throughput by3.25% F16,3.94% Q4 and3.62% Q8. The F16 image workflow improves prompt latency from224.28 to214.62 ms (-4.31%) in ABBA+BAAB, with identical greedy output in all eight runs. Four Q8 default/off image runs have identical output matching the old opt-out output, with no390 dispatches. Experimental forced-on Q8 has no model-quality equivalence claim.

The accepted runtime is `nv-incoming-048768418\native-wave32\ports-q8-accept\runtime`. Its exact claims are limited to the final decode cohort, three Granite prefill models, image comparisons and CPU/route suites; the broader earlier44ff grid is not presented as final79de evidence. `acceptance-handoff.txt`, `acceptance-primary-analysis.json`, `acceptance-final99.json` and `acceptance-provenance.json` retain settings, samples, source/runtime hashes and limits.

Installed `build-dx12-phase0\bin\Release` now matches all99 qualified runtime files; eight files were replaced. The original99-file63b9 runtime remains in `nv-incoming-048768418\baseline`, and `installation.json` records every changed hash. Vulkan binaries and unrelated user artifacts were not changed. The incoming branch merge and the additional fixes remain uncommitted; no commit or push was performed in this integration task.

## RX 9070 XT review of the four incoming commits (September 22, 2026)

This review compares the local, immutable 048768418 runtime with incoming 687603e37: the NVIDIA port b9c4449a8, expert widening 1db39c6d3, D96 reuse 8d1df3f02 and merge 687603e37. The existing RX 9070 XT driver is 32.0.23041.2023. The branch was fast-forwarded; unrelated benchmark files were preserved.

One build regression was found and fixed. Windows SDK DXC 1.8.2502.11 crashes compiling the new four-row expert shader at every wave size, even without optimization. Giving the third `dot4add_i8packed` argument the explicit type `int(0)` avoids the compiler crash without changing arithmetic. The four-row wrapper also lacked a CMake dependency on its included implementation; this is now explicit. The standard build succeeds again. All 2854 generated preview shader headers remain byte-identical before/after the fix, preserving the NVIDIA shader implementation and the measured RDNA4 code.

### Runtime comparison

The 52 clean processes cover 13 model/format combinations, each in ABBA or BAAB order with two process means per arm. Each process runs pp6144 and tg512 with built-in warmup, three repetitions, one-second repetition delay, three-second inter-process idle, batch2048/ubatch512, full offload and FA enabled. Both arms clear inherited DX12 variables, then set `DX12_FA_LINALG=1`, `DX12_FA_PIPELINE=1`, `DX12_FA_PV_F16=0` and `DX12_COMMAND_REPLAY=0`. CBV is absent. Every process confirms DX120 is the RX 9070 XT. No concurrent build or GPU job runs.

| Model | Format | Prefill change | Decode change |
| --- | --- | ---: | ---: |
| SmolLM2-135M | F16 | +0.04% | +0.90% |
| SmolLM2-135M | Q8_0 | +0.51% | +1.19% |
| SmolLM2-135M | Q4_K_M | -0.15% | +0.36% |
| Granite 1B A400M | F16 | +0.03% | +0.62% |
| Granite 1B A400M | Q8_0 | +0.16% | +0.65% |
| Granite 1B A400M | Q4_K_M | -0.21% | +0.88% |
| Phi-3-mini 4K | F16 | +0.06% | +0.11% |
| Phi-3-mini 4K | Q8_0 | -0.17% | +0.28% |
| Phi-3-mini 4K | Q4_K_M | +0.17% | +0.48% |
| Qwen3-0.6B | BF16 (repository tag F16) | -0.10% | +0.44% |
| Qwen3-0.6B | Q8_0 | +0.06% | +2.36% |
| Qwen3-0.6B | Q4_K_M | +0.14% | +1.04% |
| Qwen3-4B Instruct 2507 | Q8_0 | +0.03% | +0.23% |

No material slowdown is observed in this cohort. Prefill remains essentially unchanged; the small decode increases are consistent with reduced inactive host-path overhead, not evidence that NVIDIA-only kernels execute on AMD. Two process means per arm do not establish exact equivalence or tight confidence bounds. Falcon and other devices were not benchmarked in this review.

Ten varied-text c512/ub512 chunks per model produce identical old/new PPL and byte-identical saved quantized log-probability files for all13 combinations. Additional ub1 runs for Smol F16, Granite Q8 and Qwen0.6 Q8 also match both PPL and saved files. This is stronger than comparing rounded PPL alone, but the saved files quantize log probabilities and do not establish raw-logit bitwise identity or general model-quality equivalence. SmolVLM2-256M F16/Q8/Q4 image runs use the same unchanged CLI launcher source with each runtime's libraries, `stalib.jpg`, temperature0/seed1 and512 generated tokens. All four outputs per format are byte-identical across old/new. Image prompt/decode latency shows no regression in this small cohort.

True root and CBV each pass590 route assertions, including43 pipeline-attention assertions. An additional ten D64/D96/D128 long-context cases through16384 keys pass with attention pipeline off/on in each binding mode;40 expert129-token boundary cases pass per mode. The standard backend passes249 route assertions. Unsupported cases are not counted in these additional selectors. The final preview rebuild retains all shader bytes and passes590 routes again.

### RDNA4 transfer priorities

1. Q8 expert four-row decode is the smallest high-value candidate. Granite Q8 still uses flag17; a sampled decode graph spends0.842 of1.785 ms in its two expert shape groups (47.2% of summed dispatch time). Flag361 already has a compiled wave64 variant but is gated to RTX5070. Test activation reuse and a native-wave64 reduction before changing that gate; NVIDIA's measured13.1% model gain is not an AMD prediction.
2. Wider expert output tiles and register output stores have the largest prefill opportunity. Granite Q4's final6144-key graph spends12.636 of28.261 ms in flag400 expert-major GEMMs (44.7%). Preserve the existing expert-major chain and weighted down output rather than replacing it with NVIDIA flag305. The current register epilogue does not apply the expert-major `op7 == 2` router-weight multiply, so simply enabling it for flag400 would be wrong. Eight wave64 groups also mean512 threads, unlike NVIDIA's256; occupancy must be measured.
3. D96 query reuse is relevant but not a direct wave32-to-wave64 switch. Phi Q8's final6144-key graph spends41.309 of104.623 ms in flag333 attention (39.5%). The new Br32/four-lane ownership is NVIDIA-specific; AMD still uses Br16. Adapt and measure ownership/resource use while retaining F32 online state and the current probability conversion. NVIDIA's rejected plain Br32/Br64 and direct-PV-transfer candidates are not accepted designs.

Merged packed Q/K stores are a smaller follow-up: Qwen0.6 Q8 still uses flag104, accounting for9.2% of the sampled short-context decode dispatch time. Flag383 is wave32-only. Fixed normalization, the router, small-M kernels, expert bucketing, expert-major FFN and the dense two-stage pipeline were already present on RDNA4; porting them back is not a new optimization. The shared host-overhead remedy and stricter misaligned QK-cache guard already apply.

Profile shares above refer to individual sampled graphs, not whole-prompt fractions or clean throughput. Review logs, immutable baseline runtime, exact commands, all process samples, PPL/hash records, image outputs, CPU comparisons and profiles are retained in the session artifacts under `incoming4-*`. No speculative RDNA4 kernel promotion is included.

## B390 portability experiments (September 22, 2026)

These experiments start from `b628233694d8ebc10e6686fc8ac880cb48d2ad31` on `dx12-vulkan-parity`, not from the previously installed September 16 runtime. The baseline was rebuilt with the existing DXC 1.10.2605.24 and Agility 1.721.3-preview configuration. Its backend SHA256 is `37DEE0E9081567AEE5678653535211DF46DF8EB12EB187B8416C51C20BCD9ADC`. The Intel Arc B390 uses driver 32.0.101.8992 and physical/blob wave16. Its supported wave8x16x16 F16-accumulator matrix shape does not admit the incoming native F32-accumulator kernels.

All new Intel admissions are opt-in and restricted to device B080 with FP16 and wave16. Existing AMD/NVIDIA defaults are unchanged. The existing B390 register-Q attention default is preserved. These are separate experiments, not a recommendation to enable every flag together:

| Environment variable, set to `1` | Route | Scope |
| --- | --- | --- |
| `DX12_MOE_F16_VEC` | 360 | Aligned small-token F16 expert vector decode, including the existing weighted output |
| `DX12_MOE_Q8_ROWS4` | 361 | One-token Granite Q8 expert shapes, using the existing Q8_1 activation prepass |
| `DX12_F32_ROUTER` | 396 | Contiguous F32 router K1024/N32/M32..512 |
| `DX12_RMS_FIXED` | 382 | Aligned width1024 RMS+multiply |
| `DX12_ADD_RMS_FIXED` | 381 | Aligned width1024 add+RMS+multiply |
| `DX12_QK_NORM_PACKED` | 383 | Existing merged full-D128 NEOX Q/K normalization with paired F16 K stores |
| `DX12_NORM_ROPE_PACKED` | 380 | Existing standalone full-D128 NEOX normalization/rope/cache-store fusion with paired F16 stores |
| `DX12_FA_SCALAR_MASK` | 420..424, producer425/426 | Graph-reused scalar attention mask classification |

The packed-QK port replaces the PSO only after the existing flag104 parameter packing. It does not break apart an existing fusion. The standalone packed-K port covers models that instead use separate Q and K dispatches. Fixed normalization does not replace the existing RMS/Q8_1 fusion. Expert and router admission reads use fresh environment pointers so the in-process route/replay fixtures do not retain invalid pointers after changing the environment.

### Expert decode, router and normalization

Model comparisons use immutable DLL swaps, a discarded warm-up, ABBA or BAAB process order, tg512 or pp6144, full offload, FA enabled, batch2048/ubatch512 and five repetitions per process. GPU work is serialized; measurement starts after a seven-minute idle following builds/correctness work. The helpers record DLL hashes, restore the installed runtime and power scheme, and reject active RDP/reconnects. Numbers below are means of the two retained process means in each arm, not independent token samples.

| Candidate/model | Baseline tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| F16 expert, Granite F16 tg512, ABBA | 93.485 | 97.550 | +4.35% |
| F16 expert, independent BAAB | 92.985 | 96.100 | +3.35% |
| Q8 rows4, Granite Q8 tg512, first cohort | 142.800 | 132.545 | -7.18% |
| Q8 rows4, independent confirmation | 143.825 | 145.340 | +1.05% |
| F32 router, Granite Q4 pp6144 | 1371.360 | 1355.280 | -1.17% |
| Fixed RMS, Granite F16 tg512 | 94.085 | 89.575 | -4.79% |
| Fixed add+RMS, Granite F16 tg512 | 91.695 | 93.225 | +1.67% |

F16 expert decode is the reproducible model-level benefit. Isolated actual Granite up projections drop from approximately32-33 to13-14 us, and down projections from31-32 to17-18 us. The weighted F16 epilogue is preserved. The Q8 first cohort contains a slow116.95 tok/s candidate process; it is retained, not discarded. Its confirmation and smaller operator gains do not establish a dependable model win.

A final combined-runtime three-arm comparison uses a discarded warm-up followed by baseline/off/on/on/off/baseline. Granite F16 tg512 is96.110 tok/s for the rebuilt baseline,96.850 with the combined runtime's expert path off, and99.315 with it on. This is+3.33% versus the original baseline and+2.55% in the same DLL. The off control does not show a regression in this small cohort; it is not a general equivalence bound. The earlier+3.35%/+4.35% cohorts used intermediate runtimes.

The F32 router improves the exact M32/M128/M512 operators from approximately80-91 to25-31 us, but the full Granite prefill comparison does not resolve a model benefit. Fixed RMS is negative/noisy and fixed add+RMS is inconclusive at model level. These flags remain off on Intel.

The initial merged packed-QK Qwen3-0.6B Q4 tg512 comparison was168.085 versus168.380 tok/s (+0.18%). A subsequent real-model dispatch audit showed that neither flag104 nor383 executed: this is a no-op control, not a packed-store gain. The separate standalone experiment does execute flag380 in this model.

Bounded fused-operator timings do not establish a large normalization win. Over ABBA+BAAB, fixed add+RMS changes2.4475 to2.3975 us for one add and2.4450 to2.3975 us for two adds, with overlapping individual process results. Standalone packed stores change2.510 to2.485 us in ABBA. These measurements include the whole fused operator; small differences around2.4 us are not evidence of a proportional model gain.

Standalone packed K stores improve Qwen3-0.6B Q4 tg512 from169.595 to171.080 tok/s in ABBA (+0.88%) and168.940 to170.600 in BAAB (+0.98%). A same-DLL BAAB comparison confirms169.110 off versus170.840 on (+1.02%), rather than attributing an unrelated binary change to the store path. All three cohorts use five repetitions per process and discard their first warm-up.

The standalone packed-store ub1/c128 PPL comparison reports26.2658 for both arms, but neither graph executes flag380, so it is not packed-route quality evidence. Four subsequent temperature0/seed1 CLI trials do exercise380 only in the candidate. Three response texts match, including the second unchanged baseline; the first baseline differs. The equality assertion failed and is retained as negative evidence. This establishes baseline generation variability in this sample, not a candidate-induced quality regression or bitwise equivalence. Packed stores remain opt-in with bounded operator correctness coverage, not a model-wide output-identity claim.

F16 greedy response text matched the baseline; timing footers make whole-stdout hashes unsuitable for that comparison. Q8 greedy output changed at the final suffix. Q8 operator tolerance results therefore do not establish model-level numerical equivalence, and Q8 rows4 stays experimental. The ub1 perplexity graph did not select rows4 and is not credited as its quality evidence.

### Graph-reused scalar mask metadata

`DX12_FA_SCALAR_MASK=1` reuses the existing context-owned arena and graph-local mask cache. Br32/Bc32 metadata is shared by D64 and D96; D128 needs Br16/Bc32. Both tile dimensions are part of cache identity. Overlapping writes invalidate entries and repeated graph executions rebuild classifications from current mask contents. This is not an assumption that a mask is immutable across graph executions.

The producer classifies sixteen tiles per DWORD. Separate consumer blobs read these classes instead of rescanning the entire mask in every layer/head. Existing Q staging/register caching, QK/softmax/PV arithmetic, sink handling and output precision remain unchanged. Scalar classification preserves the original prescan's treatment of both infinity signs, signed zero and NaN; native matrix-attention classification still uses its original negative-infinity rule.

Admission requires at least64 queries and64 keys, at most65536 keys, a key count divisible by4, aligned contiguous-key F16/F32 masks and valid broadcasts/32-bit addressing. Other layouts retain the old shader. Metadata graphs exclude baked command-list replay; decision-cache replay is retained. The metadata buffer address also participates in the replay resource signature.

Actual four-chunk c512 model graphs show one producer plus35 reuses for SmolLM2,31 for Phi-3 and29 for Qwen3-4B. Reported PPL matches baseline exactly at18.3063,6.9985 and9.6802 respectively. These rounded values are not a raw-logit identity claim.

Whole-model performance includes metadata production, barriers and all consumers, rather than timing only the faster consumer. Each cohort discards one warm-up process, retains two fresh processes per arm and uses three repetitions per process. Independent BAAB confirmation was run for SmolLM2 and Qwen3-4B; the small/noisy Phi-3 result was not repeated.

| Q4_K_M model | Prompt | ABBA throughput change | BAAB throughput change |
| --- | ---: | ---: | ---: |
| SmolLM2-135M | 512 | +12.06% | +7.72% |
| SmolLM2-135M | 6144 | +3.83% | +2.84% |
| Phi-3-mini | 512 | +1.11% | Not repeated |
| Phi-3-mini | 6144 | +0.66% | Not repeated |
| Qwen3-4B | 512 | +0.86% | -0.28% |
| Qwen3-4B | 6144 | +1.13% | -0.04% |

The repeated benefit is on SmolLM2. Its pp6144 means are5082.305->5276.830 tok/s in ABBA and5163.765->5310.175 in BAAB. Short-prompt rates are noisier; the7.7-12.1% range is not a precise universal speedup. Qwen confirmation is neutral and Phi's one-cohort changes are smaller than its run variability, so neither is credited with a resolved win. Retain the metadata path as an opt-in experiment, not a general D64/D96/D128 default.

A final same-DLL SmolLM2 pp6144 ABBA, with five repetitions per process and a discarded warm-up, confirms5129.535 tok/s off versus5327.265 on (+3.85%). Both arms have the same backend hash. This isolates metadata reuse from unrelated changes between the rebuilt baseline and combined runtime.

### Correctness and evidence limits

The combined runtime passes398/398 route assertions in true root-constant mode and399/399 in CBV mode with forced replay coverage. Root mode unsets `DX12_PARAM_CBV`; assigning `"0"` would still enable CBV. Repeated expert graphs change activations and IDs and check exact analytic output and decision-cache hits. Mask graphs cover cross-layer reuse, overwritten masks, differing tile geometry, F16/F32, broadcasts, exceptional values and exclusion of baked captures/replays. Eighteen additional file-based D64/D96/D128 FA cases cover65/512 queries and512/516/6144 keys against the existing CPU oracle without changing tolerance.

The CPU attention oracle reads masks as F16, so F32-mask coverage uses analytic graphs and metadata-on/off DX12 comparison instead. Two attempts to use the broad filtered FA generator produced initialization-only logs with no completed-test summary; neither is counted as passing. The bounded file-based run completed18/18 cases. No claim is made about AMD/NVIDIA execution on this Intel machine.

Artifacts are under the session's `files\intel-parity-20260922` directory, with model CSVs named `intel-parity-*.csv` in its parent. The old installed runtime, freshly rebuilt baseline, intermediate candidates, raw measurements, dispatch audits and contrary samples are retained. The combined candidate `mask.dll` has SHA256 `83F38834F2FEC2A91D6A0D9B75005ED28AE49514DB1983FE58469ADCB9ADC983`.

## RDNA4 expert port qualification (2026-09-22)

Baseline is an immutable copy of `b62823369` on RX 9070 XT, driver32.0.23041.2023, preview DXC1.10.2605.37 and Agility1.721.3-preview. GPU runs and builds are serialized. Attention settings remain `DX12_FA_LINALG=1`, `DX12_FA_PIPELINE=1`, `DX12_FA_PV_F16=0`.

### Q8 four-row decode: rejected

The NVIDIA four-row expert kernel was admitted only under an experimental AMD override for Granite's two exact expert-vector shapes. The original64-thread implementation measured538.08 versus542.54 tok/s (+0.83%), smaller than process-to-process spread. Removing its LDS reduction/barrier with a wave-only output measured538.75 versus533.77 (-0.92%); a32-thread wave-only variant measured536.67 versus528.21 (-1.58%). Each screen used four balanced processes, tg512/r5/delay2, and passed598 CPU/route assertions. None established a repeatable gain. All three AMD experiments and their temporary fixtures were removed; the NVIDIA route and AMD flag17 are unchanged.

### Wider expert-major output tiles: retained for quantized Granite

Flag413 adds an128x128/BK16 expert-major tile with eight wave64 groups, two wave columns, two row tiles and four output-column tiles per wave. It reuses the existing bucket producer and the complete fused gate/up/SwiGLU/down/weighted-output chain. F32 accumulators, LDS output drain, expert ordering, tails and weighted-down multiplication are unchanged. No layout-copy pass is added. This doubles output reuse without doubling accumulators per wave; its512-thread group is explicitly compiled and measured rather than substituting NVIDIA's wave32 blob.

`DX12_MOE_EXPERT_WIDE=0` disables it. The default selects only Q8_0/Q4_K/Q6_K with the existing Granite32-expert/eight-selected/K1024/H512/D1024/512-token chain. F16 remains on flag400: its opt-in screen improved only about1%, insufficient to expand the default. `DX12_MOE_EXPERT_WIDE=1` permits the four compiled formats, including F16, within the existing safe expert-major matcher; unsupported formats retain their old tiles.

Final immutable-old/current-new comparison uses new/old/old/new process order, r5/delay2, pp6144 and tg512. Values are means of the two process means, not profiler timings:

| Granite format | Old pp6144 tok/s | New pp6144 tok/s | Change | Old tg512 tok/s | New tg512 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 | 21090.33 | 21085.40 | -0.02% | 422.12 | 421.73 |
| Q8_0 | 21942.07 | 23838.65 | +8.64% | 534.10 | 534.74 |
| Q4_K_M | 19357.25 | 20975.51 | +8.36% | 618.92 | 619.39 |

A separate final-context Q4 profile places the three expert groups at6.459/2.150/2.098 ms, totaling10.707 ms, versus the previous12.636 ms. This supports expert-GEMM attribution; cross-run sampled dispatch sums are not a whole-prompt speedup estimate.

True root and CBV each pass604 route assertions, including full weighted chains for all four new formats, defaults/opt-out, output-column tails and router-weight lifetime. Ten c512/ub512 varied-text PPL chunks yield old/new F16=5.2763, Q8_0=5.2842 and Q4_K_M=5.3307, with byte-identical saved quantized log-probability files for each pair. This does not claim raw-logit bitwise identity or general quality equivalence.

### Direct register output: rejected on the current AMD driver

Separate128x64 and128x128 register-output candidates included the required expert-major router-weight multiplication. Both nevertheless failed CPU comparisons for all four formats (eight failures, normalized error about2), consistent with the already documented AMD `GetCoordinate()` problem. The LDS-output wide tile passed the same full-chain comparisons. No incorrect register-output variant, AMD route or special shader code is retained. Artifacts are under `rdna-port-*`, including the rejected patch, all screen samples, final balanced logs and quality hashes.

### D96 query reuse: native eight-lane wave64 ownership

The first Br32 candidate kept NVIDIA-style four-lane row ownership in the lower half of each wave64. It passed the numerical fixtures but improved Phi Q8 pp6144 only1.70%. Using all64 lanes with eight lanes per row improved the balanced screen4.79%. Each lane now handles eight keys and12 output dimensions rather than16 keys and24 outputs. Four waves cover32 queries, with256 threads and18 KiB LDS. The existing parameterized row-partition code already expresses this mapping, so the retained shader change only permits the explicit `FA_D96_W64_BR32` compilation. No half-wave ownership branches remain.

Flag335 selects a separately compiled wave64 blob on RDNA4 and the unchanged wave32 blob on NVIDIA. On RDNA4 the default is limited to D96,512 queries,32 query/KV heads, one batch,6144-byte K row stride and up to16384 keys; all existing alignment, type, capability, mask and minimum-key guards still apply. `DX12_FA_PIPELINE=1` and F32 PV policy remain prerequisites. `DX12_FA_D96_BR32=0` restores Br16; `1` permits other safe query counts through the existing layout validator. Dispatch, mask scratch sizing and mask-cache identity all use the same device-aware row helper.

Final old/new/new/old comparison against immutable `b62823369`, r5/delay2:

| Phi-3-mini format | Old pp6144 tok/s | New pp6144 tok/s | Change | pp2048 change | tg512 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 | 4991.35 | 5166.36 | +3.51% | +0.99% | +0.23% |
| Q8_0 | 5759.89 | 6032.98 | +4.74% | +1.38% | -0.21% |
| Q4_K_M | 4643.23 | 4810.64 | +3.61% | +0.65% | -0.29% |

The unchanged pp512 path ranged from-0.90% to-0.07% in this long cohort. A focused F16 new/old/old/new r12 control measured6001.27 versus6007.29 tok/s (+0.10%), so the initial small F16 decline did not reproduce. A separate final-context Q8 profile measures D96 flag335 at33.952 ms versus the earlier flag333 sample41.309 ms; clean throughput, not that cross-run dispatch difference, is the acceptance result.

All13 pre-existing attention shader blobs remain byte-identical. True root and CBV each pass619 full-route assertions, including58 pipeline-attention assertions. Coverage includes Br32 query tails, default/override routing, analytic large-value/cancellation/subnormal cases, mask mutation/replay and head-broadcast masks. Ten additional D64/D96/D128 CPU comparisons through16384 keys pass in each binding mode.

The arithmetic policy remains F32 for QK/PV accumulators, online maxima/sums, rescaling, output accumulation and normalization. Existing F16 matrix operands and subnormal-preserving probability conversion are unchanged. Reduction association does change: Br16 uses64-lane wave sums while Br32 combines local sums and eight-lane XOR reductions. Results are not bit-identical.

Four varied-text c2048/b2048/ub512 chunks exercise the new path rather than accidentally testing only the unchanged sub1024-key fallback:

| Phi format | Old PPL | New PPL | Mean KL versus old saved probabilities | Same top probability |
| --- | ---: | ---: | ---: | ---: |
| F16 | 4.8991 | 4.8991 | 0.000002 | 100.000% |
| Q8_0 | 4.8933 | 4.8932 | 0.000002 | 100.000% |
| Q4_K_M | 5.0361 | 5.0375 | 0.000037 | 99.756% |

Baseline-self KL is approximately zero. Saved probability files differ for all three formats. Q4 PPL increases0.0014 (0.028%); RMS token-probability difference is0.192 percentage points, versus0.034/0.036 for F16/Q8. These are small but real measured output differences, not raw-logit identity or proof of universal quality equivalence. The larger Q4 drift may reflect downstream amplification; these measurements do not establish its cause. No new precision approximation or FP16 accumulator policy is introduced. Source review found no ownership, resource-lifetime or precision-policy defect. Exact quality logs, self-controls, performance samples and rejected four-lane artifacts are retained under `rdna-port-d96-*`.

## Upstream integration: precision and shared contracts (2026-09-29)

The integration target is upstream `19e28a27702117d8f2eb16b825b9a308111f67d9`, with common upstream ancestor `2d8d612e4c68d3801e556a1b4a028f55ec33ecbb` and fork baseline `b26226e61d1700c7efce59e23f3a65d8e10e56fb`. This is 552 upstream commits with 544 distinct trailing PR references. The source merge is isolated on `integrate-upstream-20260929`; the installed DX12 and Vulkan runtimes are preserved separately from candidate outputs.

### Accumulator precision is not source precision

Upstream now defines a minimum accumulator type separately from permitted source conversion. DX12 reads the accumulator policy from matmul slot0 or attention slot3, and the activation conversion policy from matmul slot3. The public source setter currently accepts only source1 of `MUL_MAT` and `MUL_MAT_ID`; it does not expose an attention source policy. Unknown policies are rejected rather than silently treated as defaults.

An explicit F32 source requirement excludes F16 staging and Q8 activation quantization. BF16 source permission does not permit F16 staging, because their ranges differ. Existing F32-input fallback routes remain available. Fusion must check every absorbed consumer, not just the first projection: a relaxed sibling must not cause a strict sibling to consume a quantized buffer or lose its F32 normalization intermediate. Precision parameters and tensor flags participate in decision and command replay identity.

F32 or BF16 accumulation excludes kernels that accumulate partial results in F16 before widening to F32. This includes experimental AMD half-PV flags321/322/324 and Intel half-accumulator wave paths. Environment overrides do not override the graph's precision requirement. Ordinary llama attention graphs explicitly request F32 accumulation, so historical half-accumulator opt-in timings are not applicable to those graphs after this correction. NVIDIA's F32-accumulating attention paths are not excluded by this rule.

Positive tests of intentional half-accumulator routes use `GGML_PREC_UNDEFINED`; separate strict-policy cases require their exclusion. For CPU flash attention, UNDEFINED and F32 use the same F32 implementation, so this preserves the numerical oracle and tolerances without a special graph-cloning override. The CPU quantized-matmul implementation can quantize activations despite a source-precision request; strict-source qualification therefore needs analytical results rather than assuming that CPU comparison certifies F32 source handling.

Persistent autotuning stores thread choices and float-matvec thresholds, not arbitrary pipeline flags. These choices remain subordinate to precision gates. `DX12_NO_AUTOTUNE=1` disables benchmarking but does not prevent loading a valid disk cache.

### Shared integration boundaries

Vulkan dispatch tracing moves with upstream's dispatch helper into `ggml-vulkan-common.h`, preserving the new descriptor-set reuse implementation. The Strix Point matvec exclusion remains in place. The Metal event path keeps the fork's `MTLEvent` implementation together with upstream's autorelease pool, and ARM initialization keeps the clang/MSVC distinction.

The two removed example scripts differed from upstream only in executable mode, so their upstream deletion is accepted. The four QDC modify/delete conflicts retain the fork's earlier script removals and pytest replacement. Shared batching and graph-input changes come from upstream; custom fixtures must mark storage leaves, not computed views, as inputs under the new `GGML_OP_NONE` requirement.

Vulkan quantized GEMM is no longer uniformly an F16-dequant path: upstream adds selected RDNA3/RDNA4 int8 cooperative-matrix routes with Q8_1 activations. Compare actual device/format dispatches, not the older blanket assumption. AMD and B390 positive paths require their own hardware qualification; an RTX 5070 run cannot establish their speed or numerical equivalence.

### Integration qualification and accepted upstream performance change (2026-09-30)

Both isolated Windows Release candidates were built with BoringSSL. The qualified code/test tree is `b7e88adf9c85e303c71bd8ec8bfc3a22aef24da7`; this results section is a later documentation-only change. The executables still report `b26226e61-dirty` because the merge is uncommitted, so source archives and runtime hashes identify the candidates, not that version string. Artifacts are retained under `build-dx12-phase0\upstream-20260929`; `r4-handoff.json` contains full inventories, commands, samples and limitations, and `diagnostic-final.json` contains the separate performance investigation.

| Artifact | SHA256 |
| --- | --- |
| Candidate `ggml-dx12.dll` | `96ab24a3eb4bc56c78822d02e3f9abcf91239f7f2a01deed501aadb551268094` |
| Candidate `ggml-vulkan.dll` | `36c65827cc6dd58eac9f183cbd890950560d3a048018ed0cd6428e8a33418461` |
| Complete candidate runtime ZIP | `9c0e100ac473d4b87a88eb95dea4163da7c43eeb87d3edc196435be1cbfac04b` |

ROOT and CBV route suites passed 425/425 each, including both strict-source RMS analytical cases; attention passed 108/108 each, and forced command replay passed 428/428 with actual replays. Both QKV paired-reference cases passed. Signed operator comparisons passed 53 cases per binding mode on DX12 and 53 per arm on Vulkan. Selected PAD/CPY/ROPE/SET_ROWS cases passed 262/262 per candidate backend. Static batch/grammar tests, existing core tests, fragmented KV restore, save/load state, CLI completion and localhost server slot save/erase/restore passed. SmolVLM2 F16/Q8 greedy image outputs were byte-identical within each baseline/candidate cohort. All 2,628 DXIL blobs are unchanged.

Qualification exposed three fixture/build issues, with every failed run retained. The forced WMMA positive initially selected the legal NVIDIA TG route; changing output width from 128 to 64 isolates WMMA without changing precision or tolerance. The mixed RMS test initially had no eligible Q4_K quantized consumer on discrete NVIDIA; explicit fixture-local `DX12_NV_Q4K_DP4A=1` exercises fusion while the strict sibling still requires the F32 intermediate. The upstream batch test used a captured local array bound that MSVC rejected even after making it constexpr; initializer-deduced `pos[]` preserves the four positions and compiles.

The fixed ABBA+BAAB cohort contains 72 DX12 processes/264 samples and 24 Vulkan processes/88 samples. All 352 samples are retained. Runs use the physical RTX 5070, root constants (`DX12_PARAM_CBV` absent), replay off, FA on, F16 KV, batch 2048, ubatch 512 and 20 CPU threads. Deltas below are ratios of arithmetic process means; positive means faster. These are finite-cohort measurements, not claims about all models or devices.

| Backend | Model | Workload | Baseline tok/s | Candidate tok/s | Change |
| --- | --- | --- | ---: | ---: | ---: |
| DX12 | Granite F16 | pp6144 | 11538.50 | 11534.00 | -0.04% |
| DX12 | Granite Q4 | pp6144 | 14576.60 | 14568.32 | -0.06% |
| DX12 | Granite Q8 | pp6144 | 14805.28 | 14769.05 | -0.24% |
| DX12 | Phi F16 | pp6144 | 3307.81 | 3306.01 | -0.05% |
| DX12 | Smol F16 | pp6144 | 24962.74 | 24813.63 | -0.60% |
| DX12 | Qwen BF16 | pp6144 | 12782.12 | 12804.02 | +0.17% |
| DX12 | Granite F16 | tg512 | 396.25 | 391.65 | -1.16% |
| DX12 | Granite Q8 | tg512 | 445.38 | 444.47 | -0.20% |
| DX12 | Smol F16 | tg512 | 796.93 | 796.66 | -0.03% |
| Vulkan | Granite Q8 | pp6144 | 24737.30 | 24059.67 | -2.74% |
| Vulkan | Phi F16 | pp6144 | 8664.73 | 8613.90 | -0.59% |
| Vulkan | Smol F16 | tg512 | 615.37 | 638.98 | +3.84% |

The DX12 Granite F16 decode mean includes one entire slow candidate process (380.07 tok/s; other candidates 395.09, 396.04 and 395.41). A separately labeled balanced diagnostic cohort measured -0.30%; it does not replace the original -1.16%. Dispatch counts were identical, and sampled GPU spans were effectively unchanged (2.433906 versus 2.433875 ms). A historical scheduling/environment cause remains possible but unproved.

Vulkan Granite Q8 prefill reproduced at -2.75% in the separate diagnostic cohort. Serialized GPU profiling attributed increases to attention (+23.37%) and dense Q8 matmul (+18.36%), partly offset by faster routing. Attention/dense-matmul node counts were unchanged, while pipeline bindings, queue submissions and descriptor updates decreased. Dense Q8 pipeline names changed from per-type variants to `matmul_quant` variants. This establishes a GPU-path regression, not its exact shader/compiler mechanism; quantized specialization/tile changes and sparse-capable attention changes are the bounded next investigation. Serialized per-node timings are not interchangeable with clean wall-time throughput.

The slower Vulkan shader paths come from the pinned upstream update: candidate Vulkan shader sources match upstream, and the only retained fork backend differences are optional dispatch tracing and an AMD-only Strix Point gate. Neither changes the measured NVIDIA kernel implementations. The exact upstream commit/compiler mechanism has not been isolated. On September 30, the user accepted this inherited upstream performance change as non-blocking; the earlier promotion hold is removed without changing or discarding any measurements.

At the qualification handoff, installed baseline runtimes remained byte-identical and the integration merge was staged without a commit or push. The user subsequently authorized committing and publishing the integration to the private fork's `dx12-linalg-phase0` branch, accepting the measured DX12 results and inherited Vulkan change. This does not claim zero performance variation or hardware coverage beyond the recorded scope. The archive `r4-qualified-candidate-runtimes.zip` preserves complete coherent runtimes; do not combine old executables with new core/backend DLLs. Build caches currently target candidate output directories.

Remaining coverage limits: no AMD/B390 hardware run, mixed-policy GLU62 and homogeneous Q8 QKV99 lack dedicated paired-reference execution, and BF16 attention has support-query rather than execution coverage because the CPU reference rejects that precision. Optional backend-scheduler tests were unavailable; the standalone GGUF temporary-file test was omitted, while real GGUF model loading was exercised. Falcon BF16 remains excluded after earlier OOM/TDR failures.

## Opt-in NVIDIA F16 QKV row pairs (2026-09-30)

`DX12_QKV_F16_ROWS2=1` enables an experimental paired-row implementation of the fused RMS + F16 QKV + NORMAL RoPE + KV scatter shader (flag88). It is disabled by default and does not require LinAlg. NVIDIA Pascal-or-newer devices can select the variant; AMD and Intel retain their existing shaders. Only F16 KV caches use paired ownership. F32 KV caches retain the one-row algorithm inside the opt-in variant, and quantized weight routes are unchanged. Set the flag before starting the process; the blob selection and dispatch geometry stay fixed for its lifetime.

```powershell
$env:DX12_QKV_F16_ROWS2 = "1"
# Run the benchmark or application here.
Remove-Item Env:DX12_QKV_F16_ROWS2
```

Each group owns two adjacent rows, computes their dot products once, applies the existing F32 RMS reduction and RoPE, and writes both results. Aligned contiguous F16 cache pairs use one 32-bit store instead of two contending half-word atomic stores. Misaligned or strided destinations retain scalar stores. Existing fusion guards still require even Q/K/V row counts, NORMAL RoPE, head alignment, contiguous weights and a shared K/V cache resource.

The original stock-toolchain prototype on RTX5070 reduced the summed flag88 GPU time from0.146632 to0.124183 ms (-15.3%). Two ABBA+BAAB tg128 cohorts measured SmolVLM2 F16 decode gains of1.80% and1.26%; these small wall-clock differences are noisy, not a promised speedup. Fresh guarded-main measurements are recorded below. Phi F16 (+0.31%) and Smol Q8 (-0.30%) were inactive-route controls, not broader tests of paired QKV. The experiment does not explain or repair the earlier old-main versus promoted-main decode regression.

With F16 KV caches, all49,280 raw F32 logits across1,024 consecutive Smol F16 tokens were bit-identical in the original prototype's off/on comparisons under root constants, CBV and actual forced command-list replay. Decode-mode perplexity distributions and Smol F16/Q8 image outputs also matched. F32-cache qualification was withheld after repeated raw-logit runs differed even on untouched main OFF; this experiment is not a fix for that separate issue.

Before considering default enablement, test other NVIDIA generations, larger-K models that actually select flag88, partial rotary/frequency-factor inputs and non-native-FP16 operation. AMD/Intel currently need opt-in exclusion and default-path checks, not speed qualification of this NVIDIA-only shader. Measurement and failure records are retained with the September30 main-profiling artifacts.

Fresh guarded-main validation used three clean Release builds with BoringSSL0.20260903.0: `build-vulkan`, `build-dx12` (LinAlg OFF, stock DXC) and `build-dx12-linalg` (LinAlg ON, preview DXC and Agility721). The repository's LinAlg compile default remains OFF. Each complete runtime is in its build directory's `bin\Release`; keep the LinAlg runtime's `D3D12` directory beside the executables.

For both DX12 builds, the off/on F16-cache raw F32 logits were bit-identical across49,280 outputs x1,024 tokens under ROOT, CBV and actual forced replay. Shader audits confirmed the opt-in variant on RTX5070 and its exclusion on Intel UHD and Q8 weights. Unset/0/1 F16-cache text and off/on F16/Q8 image outputs matched within each build. The12 default QKV blobs (three wave sizes, ordinary/native-FP16, stock/preview DXC) were byte-identical to their pristine-main counterparts. The LinAlg build passed425/425 route assertions and108/108 attention assertions under each binding, plus428/428 with forced replay. These LinAlg-only assertion suites explicitly skip on the OFF build; separate selected operator checks passed53/53 on each DX12 GPU/build and Vulkan RTX5070. Selected core tests and Vulkan prefill/decode smoke runs passed.

F32-cache forced-length text runs exercised the opt-in blob's one-row branch, but outputs also differed between unset and0 controls. Exact F32-cache qualification remains withheld. No AMD device was available. These checks do not requalify the previously failing quantized-cache attention PSOs169-174 or remove their recorded compiler/driver caveat.

After a discarded warm-up, fresh-process ABBA+BAAB tg128 measurements used three repetitions per process with the same binary and only the ENV flag changed. Smol F16 mean throughput was910.39 ->924.27 tok/s (+1.52%) with LinAlg OFF and885.47 ->903.79 tok/s (+2.07%) with LinAlg ON. The ON cohort had slow processes in both arms, including its final enabled sample; all samples are retained. This is a small, noisy single-model gain, not a general speed claim.

Fresh build, accuracy and performance records are in `.build\qkv-env-20260930`. Older build trees and qualification records were archived under `.build\archive-20260930`, not deleted. Their two registered source worktrees were relocated to that archive's `worktrees` directory with tracked and untracked changes preserved. Old CMake caches and scripts can contain their former absolute paths; they are historical artifacts, not the three active builds.

## Fresh three-build model-quality matrix (2026-09-30)

The three fresh BoringSSL runtimes above were tested against a CPU reference from the fresh stock DX12 build. The matrix covers the 18 text configurations in `bench_linalg.bat` plus SmolVLM2 F16/Q8_0/Q4_K_M. Each model/quant uses the same cached GGUF in all four arms. The F16 benchmark labels for Qwen3-0.6B and Falcon-H1 actually resolve to BF16 files. Model hashes, GGUF architecture/layer metadata, runtime inventories, exact commands and complete logs are in `.build\model-quality-20260930`; `quality-matrix.csv` and `matrix-summary-final.json` contain the scorecard.

Perplexity uses the first 16 WikiText-2 raw test chunks, context 512, batch 2048, ubatch 512, threads 20, FA on and F16 K/V caches. Corpus SHA256 is `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. CPU uses `-ngl 0 -dev none`; GPU arms use RTX5070, explicit full offload and fitting disabled. Falcon BF16 is the one capacity exception: its 15.18 GB weight file exceeds this GPU's VRAM, so its three GPU runs use explicitly recorded `-ngl 20`. This does not qualify fully GPU-resident Falcon BF16. GPU placement and LinAlg dispatches were audited; no compatibility workaround replaced a failing normal execution.

All 84 perplexity runs completed with finite results and without PSO creation or device errors. Compare PPL within a row, not between different models/tokenizers. Positive ON/OFF values mean higher perplexity with LinAlg enabled. Review triggers were absolute CPU/GPU differences above 1% or LinAlg ON/OFF differences above 0.5%; these are investigation thresholds, not proof of general quality equivalence.

| Model | Weight format | CPU PPL | Vulkan PPL | DX12 OFF PPL | LinAlg ON PPL | ON/OFF |
|---|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | F16 | 22.6761 | 22.6837 | 22.6778 | 22.6775 | -0.0013% |
| SmolLM2-135M | Q8_0 | 22.8093 | 22.7592 | 22.8318 | 22.7957 | -0.1581% |
| SmolLM2-135M | Q4_K_M | 23.4859 | 23.4282 | 23.5081 | 23.4794 | -0.1221% |
| Qwen3-4B-Instruct-2507 | F16 | 12.1993 | 12.1803 | 12.2001 | 12.1993 | -0.0066% |
| Qwen3-4B-Instruct-2507 | Q8_0 | 12.2088 | 12.1985 | 12.2045 | 12.2061 | +0.0131% |
| Qwen3-4B-Instruct-2507 | Q4_K_M | 12.5338 | 12.4865 | 12.4814 | 12.4989 | +0.1402% |
| Phi-3-mini | F16 | 6.7508 | 6.7685 | 6.7516 | 6.7510 | -0.0089% |
| Phi-3-mini | Q8_0 | 6.7581 | 6.7742 | 6.7598 | 6.7572 | -0.0385% |
| Phi-3-mini | Q4_K_M | 7.1442 | 7.1361 | 7.1386 | 7.1325 | -0.0855% |
| Granite-3.0-1B-A400M | F16 | 12.0428 | 12.0241 | 12.0474 | 12.0445 | -0.0241% |
| Granite-3.0-1B-A400M | Q8_0 | 12.0316 | 12.0215 | 12.0520 | 12.0274 | -0.2041% |
| Granite-3.0-1B-A400M | Q4_K_M | 12.3959 | 12.2961 | 12.2937 | 12.2962 | +0.0203% |
| Qwen3-0.6B | BF16 | 25.2124 | 25.2175 | 25.2067 | 25.2075 | +0.0032% |
| Qwen3-0.6B | Q8_0 | 25.1357 | 25.1742 | 25.1481 | 25.1666 | +0.0736% |
| Qwen3-0.6B | Q4_K_M | 26.3348 | 26.2057 | 26.1567 | 26.1759 | +0.0734% |
| Falcon-H1-7B | BF16, partial offload | 10.1808 | 10.1829 | 10.1819 | 10.1815 | -0.0039% |
| Falcon-H1-7B | Q8_0 | 10.2755 | 10.2285 | 10.2802 | 10.2742 | -0.0584% |
| Falcon-H1-7B | Q4_K_M | 11.0439 | 10.5710 | 10.6416 | 10.6863 | +0.4200% |
| SmolVLM2-256M | F16 | 26.5647 | 26.5671 | 26.5650 | 26.5620 | -0.0113% |
| SmolVLM2-256M | Q8_0 | 26.6174 | 26.5942 | 26.6458 | 26.6545 | +0.0327% |
| SmolVLM2-256M | Q4_K_M | 27.7430 | 27.6300 | 27.7059 | 27.6495 | -0.2036% |

The largest LinAlg ON/OFF change is +0.4200% on Falcon Q4_K_M; the largest absolute difference on the other 20 configurations is 0.2041%. The largest absolute CPU/GPU difference outside Falcon Q4 is 0.8245%, on Granite Q4_K_M. Falcon Q4 is explicitly flagged: all three GPU paths give 3.24-4.28% lower PPL than CPU, which is not evidence of a general quality improvement. CPU repetition reproduced 11.0439; disabling repacking gave 11.0550 and disabling FA gave 11.1042. Neither control removes the difference.

Historical Falcon Q4 controls used coherent preserved runtimes and the same corpus/model/settings. Pre-integration main `b7bfa804b` gave CPU 11.0480 and DX12 GPU 10.6777, establishing that the CPU/GPU difference predates the integration. Pristine promoted main `8c6f6938e` gave DX12 GPU 10.6416, exactly matching fresh DX12 OFF with the QKV experiment disabled. CPU K-quant dot products use Q8_K activations, while the audited DX12 Q4 MMQ routes use Q8_1. These are different numerical contracts, but their exact contributions to the Falcon discrepancy were not isolated. The discrepancy remains a recorded caveat, not a new optimization-related failure.

All 63 greedy text runs (21 configurations x three GPU builds, up to 128 generated tokens) produced nonempty, readable output without obvious numerical degeneration. Qwen3-0.6B spends this short budget in its reasoning stream, so these runs are not checks of completed answers. SmolVLM2 F16/Q8/Q4 image generation was also checked on CPU and all three GPU builds with `stalib.jpg`, the same F16 projector, greedy sampling and up to 128 tokens. All 12 corrected image runs were readable. F16 captions matched across GPU builds; Q8 captions differed, while Q4 captions matched between the two DX12 builds. These captions contain ordinary model hallucinations/repetition and are not proof of factual image accuracy.

Two-chunk decode-mode PPL comparisons (`-b 1 -ub 1`) exercised the QKV experiment on five F16-weight models in both DX12 builds. Shader audits confirmed active paired QKV for SmolLM2, Granite and SmolVLM2; Qwen3-4B and Phi were inactive-route controls. All 10 off/on pairs had identical PPL and identical saved compressed log-probability distributions. These distribution files are not raw F32 logits; the separate raw-logit identity coverage remains as documented in the preceding section.

Initial harness mistakes are preserved separately: one chunk-count tuple/list assertion, a Vulkan device-log parser mismatch, and CPU projector selection with the invalid name `CPU`. Audited/revalidated records and corrected `--mmproj-device none` CPU image runs resolve these harness errors without changing any backend path. No numerical or driver failure was discarded.

This is a matched regression sample on RTX5070 with F16 KV caches, not a full-corpus, long-context, multi-GPU or universal quality evaluation. It does not qualify F32/quantized KV caches, erase their previously recorded failures, or justify default enablement of the QKV experiment.

## Imported upstream feature coverage (2026-10-01)

Audited the 552 upstream commits in `2d8d612e4c68d3801e556a1b4a028f55ec33ecbb..19e28a27702117d8f2eb16b825b9a308111f67d9`, the August 31 through September 30 import. There are no new `GGML_OP_*` or `GGML_TYPE_*` enum entries in this range. The backend-relevant changes are new operator variants, precision contracts, model graphs and regression cases, not a new set of enum values.

New text architecture files are HRM/Mimir (`hrm-text`), Tencent Hy4 (`hy-v4`), Maple ternary MoE (`maple`) and Spark2.5 (`spark2-5`). New multimodal architecture files are DeepSeek4 vision (`deepseek4v`) and Ling3 VL (`ling3vl`). Their graphs use the existing graph-building infrastructure and primitives, including sparse attention/indexing, offset RoPE, MoE, padding and multimodal RoPE. Existing Qwen4 experimental graphs also gained gated HC PRE and nullable-combination/identity HC POST. This source review is not end-to-end model qualification of these architectures.

### Confirmed gaps and implementation

Before this change, the selected DX12 OFF matrix executed 322 cases successfully but skipped 45: 38 HC cases and seven scan/rollback cases. The corresponding Vulkan baseline executed 351 cases successfully and skipped 16: 12 non-four-stream HC PRE cases and four scan cases. Unsupported cases are not numerical passes.

| Surface | Change |
|---|---|
| DX12 HC COMB | Native F32 HC4 softmax/Sinkhorn combination kernel, with token blocks that remain valid across wave/workgroup boundaries |
| DX12 HC PRE | Native ordinary and sigmoid-gated collapse, with arbitrary positive stream counts and strided inputs |
| DX12 HC POST | Native combination-weighted and identity residual expansion, with arbitrary positive stream counts and strided inputs |
| DX12 selective scan | Add Mamba-2 state width 96 and K-slot rollback snapshots; preserve widths 128/256 and final-state slot 0 |
| Vulkan HC PRE/POST | Extend stream counts beyond four using the existing shaders; retain the four-stream shared-memory/unrolled path |
| Vulkan selective scan | Add state width 96 using existing specialization constants; pad workgroups to whole subgroups and guard state lanes and final partial workgroups |

The DX12 kernels are ordinary shaders, not LinAlg shaders. They work in both stock and preview builds; the LinAlg compile default remains OFF. HC metadata is included in full root-constant/CBV uploads, and auxiliary HC tensor offsets are baked into SRV addresses once. Scan snapshot slot `s` contains the state `s` tokens back, with slot 0 holding the final state. Slots beyond the available token history remain untouched. Dispatch decomposition uses the compiled DX12 blob wave size, including Intel's w16 blobs despite its reported minimum wave size of eight.

Five cases were added to the existing backend-ops test file: offset/strided COMB, ordinary and gated strided PRE, strided eight-stream combination POST, and strided 65-stream gated identity POST. The gated POST test makes the gate input contiguous before the generic sigmoid, as required by the CPU reference; the HC x/residual inputs still exercise their views. No new file was added under `tests`.

### Qualification

Rebuilt all three BoringSSL Release distributions: `build-dx12`, `build-dx12-linalg` and `build-vulkan`. Stock DX12 retains SDK DXC; the preview build retains its preview DXC/Agility configuration. The pinned BoringSSL source and TLS options were unchanged. Commands, stdout/stderr, executable/backend DLL hashes, device assertions and failure records are retained under `.build\upstream-feature-audit-20260930`.

| Physical device / build | Selected feature cases passed | Unsupported | Gather-view and sparse-attention cases passed |
|---|---:|---:|---:|
| RTX5070 / DX12 OFF | 371 | 1 | 138 |
| RTX5070 / DX12 LinAlg ON | 371 | 1 | 138 |
| RTX5070 / Vulkan | 371 | 1 | 138 |
| Intel Graphics / DX12 OFF | 371 | 1 | 138 |
| Intel Graphics / Vulkan | 370 | 2 | 138 |

The 138-case group comprises 117 GET_ROWS source-view cases, including the nonzero column offset, and 21 sparse FLASH_ATTN_EXT cases. The feature group covers HC, scan/rollback, GDN cache-copy graphs, W4A8/W4A4 regular and expert matmul graphs, Hadamard, normalization/scale, ADD_ADD, MoE reduction, batched L2 norm, padding and lightning indexing. A passing graph can execute unfused; these results do not establish that every named fusion exists or that FP4 uses native tensor-core arithmetic.

The common skip is the older Mamba-1 vector-A scan with state width 16/head dimension 1. It remains a CPU fallback, not a newly imported feature. Intel Vulkan also skips the 16384-square F32 Hadamard matmul test; its explicit transform matrix is 1 GiB and is rejected by the existing tensor-size/device buffer limits. Smaller Hadamard cases execute. Do not count either skip as qualified GPU execution or bypass the device-limit check.

Additional HC/scan qualification passed 59 cases per run, with the Mamba-1 case explicitly unsupported: Intel LinAlg ON, forced Intel root constants, and explicit CBV/no-command-replay on both RTX DX12 builds. Stock RTX DX12 also passed the same group with the D3D12 debug layer and GPU-based validation actually enabled; there were no ERROR/CORRUPTION messages. The retained log includes shutdown live-object warnings.

Actual whole-command-list replay was checked separately against CPU on seven graphs: HC COMB, gated 65-stream PRE, eight-stream combination POST, gated 65-stream identity POST, and widths 96/128/256 scan with K=3. Each graph ran 129 times in both DX12 builds while input values changed; scan sequence IDs alternated too. Every output comparison passed NMSE < 2e-7, with worst measured NMSE below 1e-13. Each graph's 128-execution statistics checkpoint showed 125 replays, three records and one capture. This establishes replay execution rather than merely setting its ENV flag. The bounded standalone harness is retained in the ignored audit directory, not added to the permanent test suite.

Falcon-H1-7B Q8_0 was rechecked with the preceding quality matrix's exact 16-chunk WikiText-2 settings, F16 K/V, context 512, batch 2048, ubatch 512 and full 45/45 layer offload. Its model SHA256 was verified against the previous record. PPL matched the preceding results at reported precision: DX12 OFF 10.2802, LinAlg ON 10.2742, Vulkan 10.2285. Shader audits confirmed 748 `ssm_scan_d256` dispatches in each DX12 run. This checks the changed shared scan body on an existing real model; it does not qualify new HC models, all new architectures, different KV types or long contexts.

The initial draft's failures are retained, not discarded: missing full HC root-parameter uploads caused numerical failures and a GPU timeout, and the first strided gated POST fixture hit the CPU sigmoid contiguity assertion. Both causes were corrected before the final runs. The first Vulkan shader compile rejected a reserved identifier; that source error was fixed normally. One Intel Vulkan harness assertion expected an over-specific device description; a fresh run verified the actual `Intel(R) Graphics` description. No runtime compatibility workaround or relaxed numerical threshold was used.

Useful repeat commands, after clearing inherited backend overrides and verifying the physical device shown by each process:

```powershell
$ops = 'DSV4.*,SSM_SCAN.*,GATED_DELTA_NET_CACHE_FUSION,MUL_MAT_W4A8,MUL_MAT_W4A4,MUL_MAT_ID_W4A8,MUL_MAT_ID_W4A4,MUL_MAT_HADAMARD,NORM_SCALE,RMS_NORM_SCALE,ADD_ADD,MOE_REDUCE,L2_NORM_BATCH,PAD.*,LIGHTNING_INDEXER'
.\build-dx12\bin\Release\test-backend-ops.exe test -b DX120 -o $ops
.\build-dx12-linalg\bin\Release\test-backend-ops.exe test -b DX120 -o $ops
.\build-vulkan\bin\Release\test-backend-ops.exe test -b Vulkan1 -o $ops
.\build-dx12\bin\Release\test-backend-ops.exe test -b DX121 -o $ops
.\build-vulkan\bin\Release\test-backend-ops.exe test -b Vulkan0 -o $ops
.\build-dx12\bin\Release\test-backend-ops.exe test -b DX120 -o 'GET_ROWS,FLASH_ATTN_EXT' -p 'vs0=1|n_kv_max=[1-9]'
```

All listed commands passed with the counts above. Repeat the final gather/sparse command with the other executable/backend pairs to obtain the 138-case column. Device indices can change, especially Vulkan's Intel/NVIDIA ordering. For DX12 CBV coverage set `DX12_PARAM_CBV=1` and `DX12_COMMAND_REPLAY=0`; for root constants leave `DX12_PARAM_CBV` unset and set `DX12_COMMAND_REPLAY=0`, including on Intel. Setting `DX12_PARAM_CBV=0` still selects CBV because this override is presence-based.

### Remaining acceleration opportunities and limits

At the audit baseline, sparse attention was numerically supported, but DX12 treated `n_kv_max` as a hint and processed the masked dense range; Vulkan already had compaction/gather support. Fast Hadamard instead of generic matmul, direct GDN cache-copy fusion, HC POST gate-chain fusion and specialized norm/scale/add-chain kernels were performance opportunities, not newly missing enum operators. The follow-up below implements bounded sparse decode and GDN cache-copy acceleration behind default-OFF switches. DX12 HC PRE already incorporates its sigmoid gate; HC POST consumes the existing gate chain's output. Vulkan retains its existing POST gate fusion.

There is no AMD hardware qualification here. DX12 w16/w32/w64 blobs compile, but only NVIDIA w32 and Intel w16 were exercised. New-architecture model-level PPL/generation and native FP4 throughput remain unqualified. Earlier F32 KV nondeterminism, quantized-KV PSO creation failures and the historical Falcon Q4 CPU/GPU discrepancy remain separate recorded limitations; these tests neither exercise nor resolve them.

## Sparse decode compaction and GDN cache-copy fusion (2026-10-01)

Both paths are ordinary DX12 shaders available in stock and LinAlg-preview builds. Both remain OFF unless their switch is exactly `1`; unset, `0` and other values retain the existing route. LinAlg's compile default remains OFF. The two BoringSSL Release distributions were rebuilt without changing their TLS settings.

| Switch | Operation |
|---|---|
| `DX12_FA_SPARSE=1` | Compact selected mask positions, then gather the original K/V rows during scalar attention |
| `DX12_GDN_CACHE_FUSION=1` | Write eligible recurrent-cache snapshots directly from GDN while retaining its full ordinary output |
| `DX12_FA_SPARSE_STATS=1` | Report mask builds/reuse using the existing `[DX12_FA_PIPELINE]` statistics |
| `DX12_GDN_CACHE_STATS=1` | Report fused dispatches recorded for the graph; replay does not re-record those dispatches |

### Sparse decode bounds and fallback

The compact path requires one F32 query, F16 K/V, equal query/K/V sequence counts and equal K/V head counts, head widths no larger than 1024, an eligible mask, no ALiBi/softcap, and a positive hint with original KV depth at least `max(4096, 4*hint)`. Metadata is capped at 64 MiB. Prompts, quantized KV, unsuitable layouts/hints and smaller ranges retain their normal attention route.

The prepass omits exact negative infinity and produces ordered original KV indices plus a GPU count for each mask row. It does not materialize compact K/V tensors. Capacity is the original KV depth, not the hint, so an understated hint cannot truncate entries; the public API still requires the hint to bound finite mask entries. Only split selection uses the hint. Both score/mask loads and V loads use the original indices. Empty rows produce zero output; empty split vectors are initialized using the V head width, including mixed K/V widths.

Metadata is context-owned, reused for matching masks within one graph, invalidated by overlapping writes, and rebuilt between graph executions. Sparse graphs conservatively exclude whole-command-list replay. The existing decision cache remains usable.

### GDN cache layout and replay

The finder accepts a contiguous direct view of the GDN snapshot tail and only view-like/NONE nodes before the following CPY. The cache must be F32 `[S_v*S_v*H_v, n_seqs, n_copied_slots, 1]`, with aligned, nonoverlapping sequence/snapshot strides and 32-bit-addressable offsets. It copies no more than the produced `min(tokens, K)` snapshots. Destination overlap with any GDN input or its ordinary output rejects fusion. Unsupported/aliased copies retain the ordinary dispatch; its decision is cached before per-graph elision so a later rejection does not leave a stale SKIP decision.

The shader writes both its original output and the cache. This removes a copy read and dispatch, not the original snapshot output or both writes. The optional cache uses UAV `u1`; otherwise-unused destination metadata carries copied-slot count (`ne0`), cache byte offset (`ne1`) and sequence/snapshot byte strides (`nb1`/`nb2`). The ordinary path sets `ne0=0`. Cache writes participate in resource barriers and write-range/attention-metadata invalidation.

GDN replay signatures cover all six inputs, normal output, K, fresh eligible-copy index and cache resource/offset/type/dimensions/strides. Changing cache metadata forces recapture; changing tensor values does not. Four CPU-comparison probes ran 257 executions each: stock NVIDIA/Intel used one-token K=3 with changing sequence stride, and preview NVIDIA/Intel used eight-token K=3 with changing snapshot stride. Inputs changed every execution and strides changed every 16 executions. All passed; worst NMSE was below `2.46e-14`. Each statistics checkpoint showed 125 actual replays, three records, nine captures and eight invalidations.

The earlier 129-execution stride-changing probes did not reach the replay statistics checkpoint: captures do not contribute to its replay+record total. Absence of that line was not proof that replay failed. The longer probes establish actual replay and metadata recapture. Async timing loops that leave the ordinary command list open do not establish command-list replay merely by setting its ENV flag.

### Correctness and measured performance

The final backend-ops selection executes 23 sparse-hint attention cases and 47 GDN cases, including eleven cache fixtures. Each OFF/ON arm passes 70/70 with zero unsupported cases on RTX5070 and Intel Graphics, in both stock and preview builds: 560 executed cases. Nine cache fixtures fuse; input-state and output-alias fixtures retain CPY. Shader audits confirm sparse compaction/consumer flags 431/430 and eligible-copy dispatch removal. Existing tests were extended; no file was added under `tests`.

Repeated sparse CPU comparisons also cover shared-mask reuse, head/sequence broadcasts, nonzero offsets, empty/one-key rows and changing dense/compact eligibility. The private harness additionally checks an understated hint, outside the valid common-suite contract. It uses the existing attention family's `5e-4` CPU NMSE threshold: an initial `2e-7` threshold also rejected the unchanged dense control at about `1.06e-5`. GDN retains `2e-7`. A separate earlier dense/compact GPU comparison measured NMSE `4.35e-17`, maximum absolute difference `2.80e-9`; its original runtime hashes and dumps are retained. D3D12 GPU-based validation passed targeted fixtures and the preview NVIDIA 129-execution sparse probe without ERROR/CORRUPTION messages.

Full-graph timings use a discarded process first, then OFF/ON/ON/OFF, with 128 warm-up executions and 2048 measured executions per process. Both arms use the same backend DLL and physical GPU. The sparse graph contains two attention nodes sharing one mask, K/V width 512, eight query heads, one KV head, KV16384 and 512 selected positions. The GDN graphs include Q/K L2 normalization and a padded cache copy, with S_v=128, sixteen heads and one sequence. These are bounded synthetic graph measurements, not model token-rate claims.

| Device / build | Sparse OFF -> ON (us) | GDN token1 OFF -> ON (us) | GDN token8 OFF -> ON (us) |
|---|---:|---:|---:|
| RTX5070 / stock | 820.56 -> 227.44 (3.61x) | 17.25 -> 13.16 (1.31x) | 30.53 -> 19.60 (1.56x) |
| RTX5070 / preview | 823.54 -> 227.36 (3.62x) | 17.20 -> 13.02 (1.32x) | 30.12 -> 19.91 (1.51x) |
| Intel Graphics / stock | 14013.85 -> 1837.29 (7.63x) | 80.31 -> 31.30 (2.57x) | 281.51 -> 185.50 (1.52x) |
| Intel Graphics / preview | 13953.36 -> 1834.76 (7.60x) | 90.69 -> 32.07 (2.83x) | 282.17 -> 183.50 (1.54x) |

All table entries disable command-list replay and leave the CBV override unset. Separate token1 measurements synchronize every graph to allow actual command-list replay, force CBV and remove its FLOP gate. With positive replay/capture counters in every arm, fusion gives stock/preview NVIDIA 73.73 -> 69.57 us (1.06x) / 76.51 -> 70.89 us (1.08x), and Intel 240.58 -> 186.34 us (1.29x) / 249.86 -> 184.79 us (1.35x). These include per-graph CPU/fence latency and are not directly comparable to the async table.

Qwen3.5-0.8B UD-IQ2_XXS, present in the broader benchmark logs, was checked on RTX5070 in both builds with F16 K/V, context 512, full 25/25 layer offload and 64 greedy generation steps. OFF/ON produced identical generated text in each build; ON recorded eighteen fused cache writes per eligible graph. Only generated text was compared, excluding the CLI's changing timing line. This is a real-model integration smoke check, not comprehensive PPL qualification of Qwen3.5, every quant or long-context rollback.

Commands, device assertions, hashes, failure logs, CPU comparisons, shader audits and timing arms are retained under `.build\dx12-acceleration-20261001`. The initial perf harness returned exit 0 without its stdout timing line; those incomplete records were rejected and retained. Final measurements require an explicit flushed stderr result as well as successful exit/device checks. A first real-model comparison incorrectly included the CLI's timing line; the generated-text-only comparison resolved that reporting mismatch without relaxing a numerical threshold or changing the backend.

Useful repeat commands after clearing inherited backend overrides and confirming the process's physical GPU:

```powershell
$env:DX12_COMMAND_REPLAY = '0'
$env:DX12_FA_SPARSE = '1'
$env:DX12_GDN_CACHE_FUSION = '1'
.\build-dx12\bin\Release\test-backend-ops.exe test -b DX120 -o 'FLASH_ATTN_EXT,GATED_DELTA_NET.*' -p 'n_kv_max=[1-9]|head_size='
.\build-dx12-linalg\bin\Release\test-backend-ops.exe test -b DX121 -o 'FLASH_ATTN_EXT,GATED_DELTA_NET.*' -p 'n_kv_max=[1-9]|head_size='
```

Both commands passed 70/70; use the other build/device combinations and unset both opt-ins for their OFF controls. For forced CBV controls set `DX12_PARAM_CBV=1`. Setting it to `0` still enables CBV because that override is presence-based.

### Fast Hadamard: current benchmark applicability

Reviewed all seventeen root `*bench*.txt` inventories, not just the 21-model quality subset. They include SmolLM2, SmolVLM2, Qwen3-0.6B/4B, Phi-3, Granite, Falcon-H1, TinyLlama, Llama-3.2-1B, Qwen3.5-0.8B and Gemma-4-E2B, with multiple weight quantizations. The recorded command lines do not request quantized KV. Both CLI and llama-bench default to F16 K/V; the preceding quality runs explicitly request it and log `attn_rot_k=0` / `attn_rot_v=0`. The new Qwen3.5 smoke also logs both rotations off. Weight labels such as Q4_K_M, Q8_0 or IQ2 do not enable KV rotation.

For ordinary attention, `src\llama-kv-cache.cpp` enables K/V rotation when that cache side is quantized, rotations are not disabled, and its head width is divisible by 64. `src\llama-impl.h` lowers the transform to ordinary MUL_MAT carrying `GGML_HINT_SRC0_IS_HADAMARD`; this is not a new GGML operator. Vulkan already recognizes that hint for its 64/128/256/512 F32 fast Walsh-Hadamard kernels. DX12 uses generic matmul, but there is no active transform to accelerate in the recorded default-F16 benchmark configurations.

Quantized-KV versions of eligible benchmark models could benefit and should be profiled before implementing item 3. SmolLM2/SmolVLM2 and Granite have 64-wide heads; Qwen3 and Falcon-H1 have 128-wide heads in the qualified metadata. Phi-3's 96-wide heads fail the current divisible-by-64 rotation gate, so simply selecting quantized KV does not activate this transform there. DeepSeek3.2/4 and related DSA/indexer architectures have additional mandatory-rotation paths even with floating-point KV; none is in this benchmark inventory. Those architectures are a separate reason to revisit fast Hadamard.

No fast-Hadamard kernel was added. AMD runtime qualification, full new-architecture quality matrices, quantized-KV PSO limitations and long-context real-model sparse-attention throughput remain open; the switches stay OFF by default.

## Local opt-in model qualification (2026-10-01)

This is a fresh qualification of `DX12_GDN_CACHE_FUSION`, `DX12_QK_NORM_PACKED`, `DX12_FA_PIPELINE` and `DX12_FA_SPARSE`, not a reuse of the earlier 21-model OFF-only quality matrix. Defaults and backend code were not changed. Evidence is in `.build\optin-qualification-20261001`: exact per-process commands/environments, model/runtime/corpus hashes, shader audits, saved distributions, generated text and retained failures. Both distributions retain BoringSSL (`LLAMA_OPENSSL=ON`); stock has LinAlg compilation OFF and preview has it ON.

Runs clear inherited backend/CLI overrides, use F16 K/V and hold `DX12_NO_AUTOTUNE=1` / `DX12_TUNE_REFRESH=1` fixed in both arms. Ordinary controls disable command-list replay and leave the CBV override unset. GPU-heavy processes are serialized. RTX5070 and Intel Graphics are present; the Intel device is Intel-UHD, not B390, and does not meet the packed-QK or FA-pipeline hardware guards. No AMD device is available.

The corpus is the same Wikitext-2 test file, SHA256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. Comparisons are within one model/weight format and one binary. A 0.5% absolute paired PPL change and mean saved-distribution KL above `1e-4` are review triggers, not universal quality guarantees. Saved files contain scaled 16-bit log probabilities clipped to a 16-logit range, not raw FP32 logits; the comparison verifies headers/tokens/file sizes, decodes and normalizes them before calculating KL and probability differences. Identical OFF repetitions were used to check numerical repeatability.

### Broad matrix and packed Q/K normalization

The preview RTX5070 matrix covers all 21 earlier model/weight configurations at context 2048, batch 2048, ubatch 512 and four chunks, with all four switches OFF versus ON. All executions have the intended offload; Falcon-H1 BF16 remains the explicitly limited 20-layer case. The largest paired PPL change is 0.034893%. All 21 short-prompt greedy generation comparisons match exactly.

Actual pipeline dispatches occur for Granite D64, Phi-3 D96 and Qwen3-0.6B/4B D128. SmolLM2, SmolVLM2 and Falcon-H1 are fallback controls for these switches, not positive pipeline qualification. The cached SmolVLM2 language model has nine query heads and three KV heads, so it does not meet the NVIDIA D64 16/8-head guard. Qwen3 generation records flag 383 for packed normalization, including Q4_K_M.

Twelve packed-only PPL pairs cover both binaries, Qwen3-0.6B/4B, BF16/F16, Q8_0 and Q4_K_M, context 512, batch/ubatch 1 and two chunks. All saved files match byte-for-byte OFF/ON. Eight pairs actually dispatch flag 383; the four Q4_K_M PPL pairs do not and are explicitly fallback controls. A further logical-batch experiment is also a fallback control. Q4_K_M active-path evidence comes from generated-text comparisons and the replay generation checks, not those no-op PPL runs. Current local model-quality evidence therefore supports the RTX5070 packed path; B390 enabled-path model quality remains unavailable here.

### Attention pipeline: small average changes do not clear every gate

Pipeline-only PPL and saved-distribution comparisons cover all three weight formats for Granite/Phi-3 at context 4096 and Qwen3-0.6B/4B at context 8192, two chunks each. Audits prove active flags 330, 335 and 327. Further context-1024, batch-1024, ubatch-256 Q4_K_M controls activate the pipeline in each family. This exercises smaller eligible query batches rather than only 512-query prompts.

All paired PPL changes remain inside 0.5%; the largest across these contexts is 0.225382%. However, several distributions exceed the separate KL review threshold. At context 4096 Granite mean KL is `2.51e-4` to `3.69e-4`, with isolated maximum probability differences up to 0.239. At context 1024 the four family means range from `2.40e-4` to `8.93e-4`. Repeated unchanged OFF runs for Granite F16 and Phi-3 F16 have identical saved distributions and zero decoded KL, so these changes cannot be dismissed as ordinary OFF-run noise.

Long-prompt Q4_K_M generation activates the pipeline in all four families. Qwen3-4B text matches; Granite, Phi-3 and Qwen3-0.6B text differs. Qwen3-0.6B retains a coherent summarization plan but changes wording; Granite changes details and repetition. These results are not bit-identical-output claims or grounds to silently raise the review threshold.

Phi-3 has a separate serious baseline limitation: context-4096 Q4_K_M PPL is 2031.4599 OFF / 2028.8656 ON, and the matched CPU control is 2168.4842. Long-prompt generated text is visibly poor in both OFF and ON arms. Logs warn that Phi SWA is disabled, but this work does not establish that warning as the complete cause. This is not a new-pipeline-only failure, and a close OFF/ON ratio does not qualify an already-bad baseline. D96 long-context quality therefore remains blocked; the pipeline also has unresolved distribution-review cases and no local AMD model qualification.

The first long-generation harness run rejected a successful execution because the CLI shortened its prompt echo to `... (truncated)`. The original record is retained. Fresh tagged runs parse the explicitly marked shortened echo and compare only generated text, excluding prompt text and timing. No model error or numerical threshold was suppressed.

### GDN: rollback and Intel baseline blockers

Actual-model rollback testing uses `test-recurrent-state-rollback.exe`, not a synthetic replacement. The bounded matrix supplies context/batch 256 and ubatch 16: all five cached Qwen3.5 weight/model variants on stock NVIDIA, plus Qwen3.5-0.8B IQ2 on stock Intel and preview NVIDIA/Intel, OFF/ON. Sixteen executions return failure. In each, single-sequence checkpoint restoration passes the existing exact comparison, but multi-sequence split replay fails at sequence 0, position 16. Within every pair the reported maximum difference and NMSE match OFF/ON.

Stock NVIDIA NMSE values are 0.0100993 (0.8B IQ2), 0.00492031 (0.8B IQ3), 0.00448566 (0.8B Q4_K_M), 0.00294063 (4B Q3_K_M) and 0.0173259 (4B Q4_K_M), versus the test's `1e-4` bound. Intel IQ2 is 0.0102069. Both ordinary-context and bounded-context CPU IQ2 controls also fail, NMSE 0.0149671. This demonstrates a baseline blocker independent of enabling cache fusion; it does not diagnose or fix the complete cause.

The test stops after its first failing zero-fill variant, so dirty-cache fill and the subsequent sequence-isolation check are not claimed as completed real-model coverage. The exact single-sequence threshold and multi-sequence threshold were not relaxed. Earlier unbounded-context failures and an interrupted redundant Intel run are retained separately, not converted into passes.

Intel stock IQ2 prefill has an additional baseline failure with fusion OFF: `MUL_MAT flags=58` PSO creation reports `0x887A0005`. The process still exits 0 and prints PPL 274637.0668. The harness rejects this from the error log despite finite output and full reported offload. Fresh device enumeration succeeds afterward, but that does not qualify the failed workload. No shader-disable switch, alternate compiler route or backend compatibility workaround was applied; broader Intel model quality is blocked.

NVIDIA quality completed separately in both binaries for Qwen3.5-0.8B UD-IQ2_XXS, UD-IQ3_XXS and Q4_K_M, plus Qwen3.5-4B Q3_K_M and Q4_K_M. These are the ordinary target models, including target files from MTP repositories; MTP speculative decoding is not claimed as tested. Ten PPL pairs use context 2048, batch 2048, ubatch 512 and two chunks. All saved distributions match byte-for-byte, with identical PPL OFF/ON. All ten 128-step greedy generation pairs also match. Audits and cache statistics show eighteen fused writes per eligible 0.8B graph and twenty-four per 4B graph; the PPL runs execute fusion beyond warm-up, not just in initialization.

Actual NVIDIA command-list replay is qualified separately for both binaries. Explicit `DX12_COMMAND_REPLAY=1` selects CBV; `DX12_COMMAND_REPLAY_STATS=1` and `DX12_REPLAY_MIN_GFLOP=0` make the intended coverage observable. GDN tests use 257 generation steps for 0.8B IQ2 and 4B Q4_K_M, OFF/ON. Packed-QK tests use two-chunk batch/ubatch-1 BF16 PPL and 257-step Q4_K_M generation. All sixteen native arms show positive `[DX12_CMD_REPLAY]` counters. Generation checkpoints report 125 replays, three records and one capture; the last PPL checkpoints report 883 replays, thirteen records and four captures. Packed flag 383 and fused GDN cache writes are verified in the ON arms. All paired text and saved distributions match. Command-list replay is distinct from the failing recurrent-state rollback test.

### Sparse real-model coverage

All 47 cached non-projector GGUF files were inspected. Their architectures are llama, qwen3, phi3, granitemoe, falcon-h1, qwen35, gemma3 and gemma4; none is a DSA sparse-attention architecture. Generic attention builders pass a zero sparse bound. Positive bounds come from DSA builders and DeepSeek4's dedicated construction, not ordinary Gemma sliding-window attention. Dense-model ON runs and zero sparse-dispatch counts are fallback controls only.

The earlier synthetic sparse correctness, alias, validation and timing evidence remains useful, but this local machine cannot supply the requested long-context real sparse-model quality or replay-inclusive model throughput comparison from its cached models. No large model was downloaded or synthetic graph relabeled as a real sparse model. Sparse default promotion remains blocked, and AMD runtime qualification remains unavailable.

### Local promotion status and reproduction

| Switch | Local conclusion |
|---|---|
| `DX12_QK_NORM_PACKED` | RTX5070 enabled-path model quality and actual replay pass; B390 model qualification remains unavailable |
| `DX12_GDN_CACHE_FUSION` | NVIDIA ordinary-model distributions, generation and command replay pass; recurrent rollback and Intel prefill block broader/default qualification |
| `DX12_FA_PIPELINE` | Average PPL changes are small, but distribution review, changed long-prompt text and the bad Phi-3 baseline prevent a clean promotion; AMD is unavailable |
| `DX12_FA_SPARSE` | Real sparse-model gate cannot be exercised from local cached models; do not substitute dense fallback or synthetic results |

The final evidence check verifies expected pair counts, positive optimized routes, positive actual command replay, matching saved distributions/text where claimed, preserved failing rollback statuses, and unchanged source/runtime hashes. Its summary is `.build\optin-qualification-20261001\final-summary.json`. No default, commit or push was made.

Commands used for the local evidence collection are below. The runner pins binaries/models, clears inherited overrides, retains failures, and reuses only matching existing records if invoked again. Actual executable commands and exit codes are in each process JSON; use a fresh output namespace for new measurements rather than counting a cached rerun as fresh qualification.

```powershell
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\optin-qualification-20261001\qualify.py quality
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\optin-qualification-20261001\qualify.py controls
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\optin-qualification-20261001\qualify.py rollback
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\optin-qualification-20261001\qualify.py gdn
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\optin-qualification-20261001\qualify.py gdn-nvidia
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\optin-qualification-20261001\qualify.py replay-nvidia
```

`quality` completed after the explicitly documented echo-parser correction; `controls` completed with recorded distribution-review triggers. `rollback` collects sixteen failing native-test records and is not a passing rollback qualification even though the collection runner itself finishes. `gdn` rejects the Intel PSO failure; `gdn-nvidia` and `replay-nvidia` complete the stated NVIDIA comparisons without hiding or bypassing the failed Intel workload.

## Packed Q/K default and GDN blocker investigation (2026-10-01)

### Packed Q/K default promotion

`DX12_QK_NORM_PACKED` now defaults ON. Unset or exactly `1` selects flag383 on eligible graphs; `0` and other explicit values retain flag104. Existing RTX5070 wave32 and B390 wave16, full-D128 NEOX, merged-Q/K, F16-cache and DWORD-alignment guards are unchanged. Unsupported hardware and layouts still fall back. No shader, packing, root signature or dispatch geometry changes accompany this promotion. B390 model quality remains unqualified locally. GDN cache fusion, FA pipeline and sparse FA remain opt-ins.

Both stock and LinAlg Release runtimes were rebuilt with their existing BoringSSL configurations. Existing route fixtures now expect the packed default and retain explicit OFF, invalid-value, cache-offset, F32, normal-RoPE, partial-rotary and odd-stride checks. The route-suite entry guard also recognizes flag383 support, so stock RTX5070 no longer skips the suite merely because LinAlg routes are unavailable.

On RTX5070, stock ordinary route assertions pass 171/171 and preview ordinary assertions pass 425/425. Final CBV suites pass 172/172 stock and 428/428 preview, including the existing 129-execution packed graph comparison against CPU. Both report actual command-list replay: 125 replays, three records, one capture and four recorded packed dispatches. Fresh Qwen3-0.6B Q4_K_M generation comparisons use unset/0/1 in each binary, both ordinary and forced-command-replay modes, with generation caps 128/257. All twelve outputs match within each three-arm group. Audits show flag383 for unset and 1, zero packed dispatches for 0, and positive actual command-replay counters in all forced-replay arms.

For normal use, remove the old explicit ON setting; to compare against the old route:

```bat
set "DX12_QK_NORM_PACKED="
rem Explicit old-path control:
set "DX12_QK_NORM_PACKED=0"
```

### Baseline recurrent rollback: two separate findings

The current three-token test exercises an unsupported snapshot history but the memory API accepts it. `test_multi_seq_split_replay` first decodes sixteen tokens, advances three more, then removes three. GDN writes `min(n_seq_tokens, K)` newest snapshots, with slot0 holding the latest state. The three-token advance writes only slots0..2; restoring three tokens back selects slot3, which still holds older prefill history. In this setup, the requested state is after position15, but the stale GDN slot is after position12. Convolution history has different short-batch handling, so the two state components do not provide a valid paired restore. This is a graph/memory contract problem, independent of the DX12 fusion.

Relevant implementation: `tests\test-recurrent-state-rollback.cpp` setup, `src\models\delta-net-base.cpp::build_recurrent_attn` snapshot copy, `ggml\src\ggml-cpu\ops.cpp` GDN snapshot mapping and `src\llama-memory-recurrent.cpp::seq_rm`. The API currently bounds rollback by configured `n_rs_seq`, not the valid history actually produced. The existing single-sequence test already contains a TODO warning that a repeated rollback after a short ubatch is invalid.

An isolated probe linked to the preserved original stock runtime reproduces CPU IQ2 NMSE 0.0149670745 for the exact three-token setup. Deferring all removals until both sequences finish prefill does not change it. Increasing the advance to four or nine tokens reduces NMSE to 0.0027999663 / 0.0028009831 but does not clear the original `1e-4` bound. Importantly, controls that do not roll back at all, but group the same prefix tokens differently, also diverge: NMSE 0.0028928383 / 0.0028613756. Therefore valid-tail residuals cannot automatically be called rollback corruption; there is a separate batch-shape-dependent baseline to isolate. No threshold was raised, and the probe's successful exit denotes collection, not passing rollback qualification.

Resolution plan:

1. Make the valid-depth contract explicit. Either track and refuse unavailable snapshot depths, or carry prior history correctly across short ubatches if that behavior is required. Update that metadata consistently across placement dry-runs, copies, clears, restore and sequence ownership. Do not infer validity from the allocated snapshot count.
2. Calibrate a matched-batch reference or state-level GDN/conv oracle before diagnosing the remaining logit drift. Compare cache tensors at the restore boundary, then the first differing layer. Separate normal batch-shape divergence from wrong restored state; keep the original failing case as a rejection/history regression, not a numerical pass.
3. Extend the existing rollback test, without adding a new permanent test file: valid and invalid depths, rejected-call immutability, short-batch history, split replay, dirty fill and sequence isolation. Then repeat all five cached Qwen3.5 variants on CPU/NVIDIA and the available Intel cases with fusion OFF/ON.

Upstream `ggml-org/llama.cpp#25004` is an open, relevant proposal covering rolling history and valid-depth bookkeeping; it is not a merged or locally validated fix. `ggml-org/llama.cpp#28019` concerns a different architecture and does not establish the cause here. No upstream patch was copied or production rollback behavior changed during this investigation.

### Intel prefill: device hang, not the first reported PSO

The original stock IQ2 workload was reproduced with all four optimizations explicitly OFF and only debug-layer, DRED and PSO-trace diagnostics added. It still fails. The new PSO error logging reports the actual device reason `0x887A0006` (`DEVICE_HUNG`). The debug layer independently reports TDR/device removal. PSO tracing identifies flags58, src0 type13 as `mul_mat_q5k_q8_1_tiled` (Q5_K, not Q5_0); its creation attempt takes only 0.065ms and happens about 96 seconds after trace start, following the earlier hang. Do not blame that PSO as the initiating failure.

DRED records an earlier command list stopped at breadcrumb506 of568 in dispatch operations. No page-fault address or useful per-node breadcrumb context was printed; that does not prove memory access is safe or identify an exact shader. The diagnostic native run still exits0 and prints invalid PPL 307856.8035 after losing the device.

A separate attribution run uses the existing `DX12_SYNC_PER_OP=1`, with `DX12_SYNC_PER_OP_MS=120000`, on the same model, tokens, batch settings, compiler and shader-selection controls. It completes without a device or PSO error and prints PPL 48.5915, taking about 175 seconds for evaluation. This is a diagnostic scheduling change, not a fix or a qualified workaround. It narrows the issue to normal batching, synchronization or long/unpreemptible submission behavior; it does not distinguish those possibilities. The ordinary failing path and its bad output are retained.

There is also a definite error-propagation defect: the record path skips a node whose pipeline is missing, caches `DX12_DEC_NO_PIPELINE`, and the decision-cache path skips it again. A required GPU computation can consequently be omitted while the caller receives success. PSO/device failures need to reach the scheduler/application rather than produce plausible-looking PPL.

Resolution plan:

1. Propagate missing required pipelines and device-loss/fence errors as failures in both record and cached paths. Keep the new removed-reason/DRED reporting. Verify native nonzero/error results and that perplexity never reports successful estimates from failed decode/output.
2. Localize the unchanged batched path with the existing barrier trace/PIX hooks and a minimized graph or existing backend-ops fixture. Check resource lifetimes, RAW/WAW ordering and per-submission duration/preemption. Per-op synchronization is an attribution control only; do not ship it as a performance fix.
3. If the defect is missing dependencies, fix their precise ordering. If a normal submission exceeds the device's watchdog/preemption budget, evaluate bounded submissions without per-node waits. If a valid minimal graph still hangs, preserve the same DXIL/driver failure for driver triage instead of switching compilers, disabling the shader or changing system TDR settings.
4. Rerun the original two-chunk IQ2 prefill successfully without diagnostic synchronization, verify model distributions against controls, then repeat Intel GDN fusion OFF/ON, replay and valid rollback coverage before promotion.

### Evidence and commands

Evidence is under `.build\packed-default-gdn-investigation-20261001`, including the preserved stock failure runtime, source/runtime hashes, unmodified original failure references, CPU probe source/results, fresh default comparisons, route logs and both Intel diagnostic runs. The earlier qualification namespace remains tied to its original hashes; its finalizer must not be rerun against these rebuilt binaries.

```powershell
# CMake is the existing Visual Studio bundled executable.
& 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build-dx12 --config Release --target test-backend-ops --parallel 8
& 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build-dx12-linalg --config Release --target test-backend-ops --parallel 8
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\packed-default-gdn-investigation-20261001\run.py packed
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\packed-default-gdn-investigation-20261001\run.py rollback
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\packed-default-gdn-investigation-20261001\run.py intel
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\packed-default-gdn-investigation-20261001\run.py intel-sync
```

Both builds and the packed runner pass. The rollback runner records eight diagnostic cases, including failing numerical comparisons. The ordinary Intel native run fails diagnostically despite exit0; the per-op run completes but is not ordinary-path qualification. Exact commands/environments are retained in the process JSON records. No commit or push was made.

## GDN baseline corrections and Intel submission budgeting (2026-10-01)

### Recurrent history contract

Recurrent memory now tracks valid rollback depth per sequence instead of treating all allocated snapshot planes as initialized history. The conservative bound is `min(n_rs_seq, latest_ubatch.n_seq_tokens - 1)`. No rolling-history carry was added. A three-token advance therefore permits depths1..2, not depth3; single-token replay leaves depth0. Unavailable depths and partial rollback of shared recurrent cells return false before changing state. A pending rollback remains single-use.

Copies inherit the source's pending index and valid depth. Clear/keep operations reset discarded owners; checkpoint restore resets history to depth0 because serialization contains only the logical current state. Input preparation consumes pending indices consistently for all owners of a shared cell. Inactive cells materialized by a graph lose their deeper-history claim because only their current plane is copied. Placement dry-runs do not update valid-depth metadata. Negative-sequence full removal now clears recurrent tails and metadata consistently.

The existing rollback fixture retains its zero-error checkpoint comparisons and `1e-4` split-replay bound. Split replay now compares the pending in-place restore with a separately materialized checkpoint from the same prefill batching; it no longer compares against a differently grouped prefix. New checks cover the original invalid three-token history, rejected-call byte/position immutability, copied/shared ownership, keep, restored history, short-batch invalidation and clear-all. The formerly commented-out repeated single-token rollback is now an explicit rejection check. No permanent test file was added.

The preserved original runtime fails these new rejection checks. The corrected stock CPU runtime passes all five cached Qwen3.5 model/weight variants with both zero and `0x3e` dirty fills, including split replay and sequence isolation.

### Batch-shape baseline isolated

An ignored CPU tensor-callback probe reproduces the no-rollback four-token versus one-plus-three control at NMSE 0.00289283825373. Callback and non-callback final logits are bit-identical in each arm. Matching logical token slices have identical layer0 embeddings, normalization, QKV projection, convolution, GDN inputs and GDN attention outputs. The first difference is the Q4_K linear-attention output projection, with maximum differences about `2e-7`. The subsequent IQ2_XXS FFN grows the difference to about `1e-3`, followed by the Q2_K down projection.

CPU repacking explicitly chooses GEMM for more than three RHS rows and GEMV for the one-/three-row controls (`ggml\src\ggml-cpu\repack.cpp::forward_mul_mat_one_chunk`). The probe identifies batch-route numerical differences outside the cache restore, including amplification at later quantized projections; it does not justify raising the rollback threshold or accepting the original stale history. The matched-checkpoint fixture avoids that confound and requires the original bounds.

### Required GPU failures are no longer successful output

Both pipeline-decision paths now return `GGML_STATUS_FAILED` for a missing required pipeline. An ignored direct-backend probe uses an unsupported loss node to verify first-record and cached-decision failure in stock and preview runtimes. Both return status-1 rather than silently skipping the node.

Compute/transfer fence reads recognize the device-removal sentinel `UINT64_MAX`, fence event setup/waits are checked, and idle queue signals no longer ignore failures. Existing fatal device-loss handling reports the removed reason and optional DRED data. A fresh ordinary Intel hang exits `0xC0000409`, reports `DEVICE_HUNG`, and produces no final PPL estimate. Standard perplexity also returns native exit1 for an unsuccessful result; a too-short input verifies that path without GPU failure.

### Intel scalar IQ submission policy

Reducing the command-ring depth to2 still hangs. Forcing an inter-node UAV barrier also still hangs. Submitting every node without per-node CPU waits succeeds with PPL48.5915 and no debug-layer errors. A synchronized profiling control places about83% of the roughly87-second prompt-graph GPU time in scalar IQ2_XXS matmuls (flag43); the largest bucket averages about1.17 seconds per dispatch. Q5_K flags58 is a later vocabulary projection, not the initiating fault. The profiler-plus-per-op combination emits three closed-list debug errors, so it is attribution evidence, not a clean qualification run.

A selective submission diagnostic separates scalar IQ dispatches from the surrounding command list without changing compiler, DXIL, shader choice or adding GPU waits. It succeeds in about178 seconds, with no device/PSO/debug errors, PPL48.5915 and byte-identical saved distributions to the earlier synchronized control. This supports a submission-budget explanation rather than an ordinary inter-node UAV dependency fix; it does not identify a vendor-internal preemption defect.

The final policy is narrower: on Intel-UHD only, prompt flag43 IQ matmuls estimated at least1 GFLOP are submitted separately, retaining command-ring pipelining. Small prompts, generation, other shaders and other hardware keep their old submission policy. An explicit `DX12_FLUSH_THRESHOLD` overrides this automatic policy, including `24` for reproducing the original batching during driver triage. No per-op synchronization, compiler downgrade, shader fallback or TDR registry change is shipped.

The final stock ordinary two-chunk IQ2 prefill completes with fusion OFF and ON, both at PPL48.5915. Saved distributions are byte-identical to each other and the old synchronized control, without per-op waits or a flush override. An independent CPU-only control (0/25 layers offloaded, CPU model/cache/compute buffers) gives PPL49.4765. CPU versus Intel saved-distribution mean KL is0.007556, with94.71% identical top tokens; this is not a strict CPU-equivalence pass. Within-backend OFF/ON comparisons do not inherit that cross-backend tolerance.

### Evidence

Fresh artifacts are under `.build\gdn-resolution-20261001`: the original-runtime rejection test, CPU callback traces and calibration, checked-failure probes, Intel ring/barrier/submission controls, source/runtime manifests and final model qualification records. Earlier namespaces stay pinned to their original runtimes and failure evidence. `budget-final-runtime-manifest.json` identifies the binaries for final qualification; the earlier `implementation-runtime-manifest.json` identifies the first selective-submission diagnostic.

### Separate Intel large-offset baseline failure

Broader qualification exposed a second Intel failure: stock Qwen3.5-4B Q3_K_M rollback with fusion OFF hangs during the initial nine-token prefill. The preserved original runtime also reports `DEVICE_HUNG` on this workload. Per-op synchronization does not eliminate it; the failing dispatch is `linear_attn_out-29`, Q4_K flag30, with weight shape `[4096,2560,1,1]` and activation shape `[4096,9,1,1]`. This is not an unavailable-history rejection or an ON-only fusion failure.

An ignored capture/replay probe isolates the dispatch without running GDN or the rest of the model. Its original weight offset is `2,151,486,464` bytes, above 2 GiB. On Intel, the captured weight/activation inputs complete 200 isolated flag30 dispatches at compact offsets. The same inputs also complete below 2 GiB, at weight offset `2,141,581,312`, with unchanged finite output. At the original offset, one isolated flag30 dispatch reports `DEVICE_HUNG` and exits `0xC0000409`. NVIDIA completes both its normal flag163 MMQ route and an explicitly selected flag30 LDS control at that original offset. The Intel compact/below-offset outputs and NVIDIA original-offset flag30 output are byte-identical (SHA256 `3854198cacb84b03c23ac6e60586dfbe89ffda7e483ec8dcc2b39cf68cb6bb8c`).

This establishes a large-offset dependency in the Intel Q4_K LDS path. It does not yet prove a particular signed-address compiler or driver defect, or an exact boundary for every shader. The first padding-probe draft failed a GGML view-size assertion before GPU execution; only the corrected quantized-backing `-v2` controls support the offset comparison. The synchronized model probe and isolated failing probe remain diagnostic controls, not qualification passes. A debug/DRED model control was stopped after it stalled while reporting removal.

Address arithmetic in the shader is unsigned. DXC's `utils/hct/gen_intrin_main.txt`, namespace `ByteAddressBufferMethods`, declares ordinary `Load` with `uint byteOffset`; the legacy Learn page's `Load(int)` synopsis is not evidence that this offset must be capped at 2 GiB. The captured offset fits the backend's 32-bit parameter and remains within its allocation. Further triage must identify the actual failing address lowering or resource-access behavior rather than silently imposing a signed-offset limit.

No shader fallback, root-address rebasing, compiler change or default gate was added to hide this failure. The IQ2 submission fix stays narrowly scoped to its measured workload. Full Intel 4B rollback qualification remains blocked, but an independent baseline failure must be evaluated separately from incremental GDN fusion risk; see the matched ON control below. Captured inputs and the standalone executable are retained locally for driver/compiler triage; nothing was uploaded.

### Final completed qualification and remaining block

Stock and LinAlg-preview DX12 BoringSSL builds are rebuilt. Provenance verification checks the implementation sources, runtime/executable hashes, five model hashes and corpus hash. The completed results are:

| Coverage | Result |
|---|---|
| CPU, all five Qwen3.5 variants | Five native rollback processes pass both fills |
| RTX5070, both builds, all five variants, fusion OFF/ON | Twenty native rollback processes pass both fills, including checkpoint restore, split replay, sequence isolation and history lifecycle |
| Intel-UHD, both builds, IQ2/IQ3/0.8B Q4_K_M, fusion OFF/ON | Twelve native rollback processes pass both fills |
| Model quality, both builds | Twelve OFF/ON pairs: ten NVIDIA model/build pairs and two ordinary Intel IQ2 pairs; saved distributions are byte-identical within every pair |
| Actual command replay, both builds | Six real-model OFF/ON pairs have identical response text and 125 actual replays in each arm |
| Ordinary/root-constant and CBV/replay route suites | Stock 171/171 and 172/172; preview 425/425 and 428/428 |

This is 37 successful native rollback processes and 74 successful fill runs. The fixture's original zero-error checkpoint comparisons and `1e-4` split-replay bound are unchanged; GPU ON processes have positive fused dispatch counts. Intel ordinary prefill is now qualified on both stock and preview without per-op waits or a flush override: both OFF/ON pairs give PPL48.5915 and the same saved-distribution SHA256 `66deecd80cc354dc1f2ac4ad05f86c346402c211b193a722022a945597cc3f5d`, also matching the old synchronized control.

The full 45-process rollback matrix is not signed off. Stock Intel 4B Q3_K_M OFF and ON processes fail with the independently reproduced large-offset hang; six remaining Intel 4B rollback arms are unqualified, not counted as passes. `qualification-status.json` records `incomplete`, the two native failures and the six missing arms. No AMD hardware was available. GDN cache fusion, FA pipeline and sparse FA remain opt-in; packed Q/K remains default ON under its existing guards. No commit or push was made.

The qualification collector was corrected to exclude variable CLI performance footers from response-text comparisons and to use the same stock Intel workload definition for the previously unrecorded preview prefill pair. Those were collection/template errors, not native numerical failures. The original records and failed controls remain preserved.

Reproduction commands from the repository root:

```powershell
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\gdn-resolution-20261001\validate.py nv-matrix
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\gdn-resolution-20261001\validate.py intel-small-matrix
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\gdn-resolution-20261001\summarize-qualification.py
```

The completed NVIDIA matrix passes. All Intel small-model rollback/replay processes and the separately completed preview prefill pair pass; rerunning `intel-small-matrix` reuses the pinned completed records with the corrected workload template. The summarizer verifies those passes but reports incomplete full qualification. To finish Intel 4B coverage, first resolve or externally qualify the preserved isolated large-offset failure, then run the original full matrix and strict finalizer without changing the fixture thresholds.

### Intel 4B fusion-ON matched failure control (2026-10-01)

The stock 4B Q3_K_M workload was rerun with `DX12_GDN_CACHE_FUSION=1`, retaining the same executable, runtime DLL hashes, model, offload, cache types and batching as the failing OFF control. The ordinary ON run exits `0xC0000409` with `DEVICE_HUNG`. A second ON run uses the same per-op diagnostic controls as the earlier OFF attribution run, plus a read/write trace. Both stop at node2312/2454, `MUL_MAT linear_attn_out-29`, Q4_K flag30, with identical destination `[2560,9,1,1]`, weight `[4096,2560,1,1]` and activation `[4096,9,1,1]` shapes, and removal reason `0x887A0006`.

The ON trace records 23 GDN dispatches with additional cache-write ranges before that projection, confirming the fusion path executed rather than merely accepting the environment flag. No earlier native error is reported. This confirms that OFF and ON reach the same independently isolated large-offset failure; this failure is not evidence of an incremental GDN fusion regression. It remains a backend bug to fix, not an automatic veto on promotion for separately qualified hardware/models. This control does not establish 4B output quality after the failing projection, qualify every Intel GPU or promote any default.

Evidence is in `intel-4b-fusion-on-confirmation.json`, the ordinary `*-f1-matrix` record, and `intel-q3-fusion-on-sync-confirmation.{json,stderr,trace.jsonl}` under the existing ignored artifact directory. The comparison runner returns success only after checking native failure codes, matching OFF/ON signatures, identical runtime/executable hashes and actual fused cache writes:

```powershell
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\gdn-resolution-20261001\validate.py intel-4b-on-control
```

## GDN cache default promotion (2026-10-01)

`DX12_GDN_CACHE_FUSION` now defaults ON for the qualified RTX5070 wave32 path and Intel-UHD with wave16 shader blobs, in both stock and LinAlg-preview builds. Unset selects that hardware default; exactly `1` explicitly enables the existing fusion on any supported hardware. `0` and other explicit values disable it. AMD, other NVIDIA devices and other Intel families remain exact-1 opt-in until qualified.

Graph execution and command-replay signatures use the same enable policy. The existing F32 snapshot-tail, stride, offset and input/output alias guards are unchanged; unsuitable copies still execute normally. No shader arithmetic, root layout, snapshot contract or submission policy changed. The independently reproduced Intel 4B large-offset failure remains a separate backend bug, not a fusion-default gate.

Remove the explicit ON setting from scripts using the qualified devices. For a baseline comparison, retain an explicit opt-out:

```bat
set "DX12_GDN_CACHE_FUSION="
rem Explicit old-path control:
set "DX12_GDN_CACHE_FUSION=0"
```

Packed Q/K remains default ON under its existing guards. FA pipeline and sparse FA remain opt-in.

Both BoringSSL Release distributions were rebuilt, including llama-bench, CLI, perplexity, server, backend-ops and recurrent rollback executables. Fresh evidence is under `.build\gdn-default-20261001`; the earlier qualification records remain unchanged and pinned to their original binaries.

The existing eleven cache-copy fixtures passed unset/0/1/invalid-value controls on both GPUs and both builds: 16 processes and 176 cases. Unset and `1` each fuse the nine eligible copies, while the two alias fixtures retain their normal copies; `0` and `2` fuse none. Four default-ON Qwen3.5 IQ2 rollback processes pass both cache fills, checkpoint restore, split replay, sequence isolation and history lifecycle checks.

Four fresh default/explicit-1 generation pairs match, also matching the preserved OFF/ON responses. Every arm records 125 actual command-list replays and positive fused dispatches. Four fresh default-ON perplexity runs have byte-identical saved distributions and the same reported PPL as the preserved qualified explicit-1 references. This verifies the default change on the qualified paths; it does not add AMD/other-family qualification or resolve the separate Intel 4B hang.

```powershell
.\.build\optin-qualification-20261001\venv\Scripts\python.exe .\.build\gdn-default-20261001\validate.py
```

The validation command passed all 32 native processes. It creates fresh records exclusively and does not silently reuse previous measurements; use a fresh artifact directory for another run.

## Explicit-target preview runtime staging (2026-10-01)

A fresh `build-dx12-linalg` configured with preview LinAlg compiled successfully but reported SM 6.8, LinAlg unavailable and experimental shader models unavailable on the RX 9070 XT. Building named consumers did not run the Agility staging target: that target depended on `ggml-dx12`, rather than being a prerequisite of it.

The backend now depends on the staging target, which depends on the two SDK DLL inputs. An explicit `cmake --build build-dx12-linalg --target llama-cli` stages `bin\D3D12` and the actual device banner reports SM 6.9, LinAlg available and experimental shader models available on the RX 9070 XT. The AMD integrated GPU correctly remains non-LinAlg. Stock builds and the default-off LinAlg configuration are unchanged.

This fixes runtime packaging, not numerical or performance qualification. The broader merged-main AMD campaign is separate.

## AMD integrated-GPU indexer qualification (2026-10-01)

The merged-main `LIGHTNING_INDEXER` path removed the device with `DXGI_ERROR_DEVICE_HUNG` on the AMD integrated GPU (PCI 13C0, driver 32.0.21045.1000). The first failing operator case used D128, 64 heads, 256 KV rows, 512 query tokens and four streams with F32 keys. The original group assigned one head to each lane and loaded query elements serially at a 512-byte inter-lane stride.

The shared shader now assigns a head to a wave, with adjacent lanes loading adjacent query elements. A 256-thread group shares the staged key across its waves, reduces each dot product with `WaveActiveSum`, and combines the per-wave head sums. Key formats, offsets, weight and mask broadcasting, output layout and F32 arithmetic remain unchanged. Reduction association changes; this is not a bit-identical transformation.

All 156 existing indexer fixtures pass CPU comparison on the RX 9070 XT and the integrated GPU in stock and preview builds, with both root constants and CBV parameters (1,248 operator comparisons). The formerly hanging four-stream F32 case also passes in isolation. The broader new-operator selector passes 366 supported cases per arm; its one unsupported SSM_SCAN shape is excluded, not counted as a pass. No timeout suppression, unsupported-op fallback or numerical-tolerance change was used.

These are operator-level results, not an end-to-end DSA-model quality or performance claim. Artifacts are the `qual-fixed-*-newops.csv` files in the October 1 AMD qualification campaign.

## AMD integrated-GPU cache allocation isolation (2026-10-01)

Qwen3.5-0.8B IQ2_XXS checkpoint replay initially produced nonfinite logits on the AMD integrated GPU in both stock and preview builds, with GDN cache fusion either disabled or enabled. The first nonfinite operation was scalar flash attention, but its active K/V cache rows had already been overwritten during the initial prompt. Skipping zero-probability V loads did not fix it and was not retained.

A seven-buffer allocation/clear probe reproduced the overwrite without dispatching any shaders: the first 2 MiB of a 3 MiB allocation acquired the fill pattern of a distinct 173.4 MiB allocation. An independent D3D12-only reproduction established the same failure when transfer command objects were created during the lifetime of a short-lived committed resource. The defect reproduced without background PSO creation and with the transient adapter probe skipped. Debug-layer activation prevented it, but is not used as a production fix.

On AMD integrated GPUs, the transfer allocator, command list, fence and event are now initialized during device initialization, before committed tensor-buffer lifetimes. Upload/readback buffers remain lazy, transfer arithmetic and synchronization are unchanged, and other device families retain their existing initialization. This prevents the reproduced allocation-order failure on driver 32.0.21045.1000; it does not establish a universal driver root cause or repair the driver itself.

The seven-buffer isolation probe passes after this change. Full recurrent rollback passes in all eight combinations of stock/preview, discrete/integrated GPU and GDN cache fusion off/on. Each run covers both zero and 0x3e cache fills, checkpoint and dirty-context restore, split replay, sequence isolation and rollback-history lifecycle, with the existing exact-logit requirement unchanged. `DX12_FA_PIPELINE=1` remains enabled; fusion-on runs record positive fused dispatch counts. The same binaries also pass 1,008 supported CPU-reference operator cases per DX12 build/device across root constants and CBV.

Artifacts include `alias-faithful-backend-order.txt`, `alias-fixed-final.txt` and `qual-fixed-*-rollback-*.txt` in the October 1 AMD qualification campaign. Performance and model-distribution comparisons are separate, still-pending qualifications.

## B390 driver 9033: wave32 tiled GEMM gated off (2026-10-01)

Intel driver 32.0.101.9033 breaks the wave32 tiled LinAlg quant GEMM (flags 294/295). `test-backend-ops -o MUL_MAT -p type_a=q4_K` with `DX12_LINALG_TILED_GEMM=3` gives ERR ~85 at m=256,n=128,k=256, then device removal (0x887A0006). The bc682902f runtime fails the same way, so the source is not the cause. Driver 8992 passed. Wave16 modes 1/2 pass.

Model impact with automatic mode 3 (wikitext c512, 20 chunks): Phi-3 Q4_K_M PPL 9.85e7, Qwen3-4B Q4_K_M 2.6e44, Qwen3-0.6B Q8_0 NaN. With `DX12_LINALG_TILED_GEMM=0` they are 6.8637 / 11.6773 / 23.6328.

Modes 3/4 (explicit or automatic) now need `DX12_LINALG_TILED_W32=1`. Without it they fall back to the non-tiled route. The route tests for 294/295 and the B390 automatic block also run only with that variable set. Remove the gate once a fixed driver passes `test-backend-ops -o MUL_MAT` and `-o DX12_ROUTES` with `DX12_LINALG_TILED_W32=1`.

## GQA-shared decode attention (2026-10-01)

`flash_attn_cd_gqa.hlsli` (flags 440-445: D64/96/128, F16 and Q8_0 KV) is the decode path when G = n_heads / n_kv_heads >= 3. One wave serves up to 4 Q heads of one KV head, so each K/V row is read once for the group instead of once per Q head. Dispatch y = n_kv_heads * ceil(G/4). Partials use the existing reduce layout. ALiBi (max_bias != 0) and per-head masks (mask ne2 > 1) keep the per-head path. `DX12_FA_CD_GQA=0` disables it.

The kernel launches G times fewer waves, so it needs more KV splits: the cap is `DX12_FA_CD_GQA_MAX_SPLITS` (default 64, not 32), also clamped to what fits in `splitkv_temp`. The replay record stores this cap so replay recomputes the same split count. At cap 32, SmolVLM2 (3 KV heads) got slower (FA 2.50 -> 2.72 ms at d4096); at 64 it got faster (~1.9-2.4 ms).

G=2 (Qwen3-0.6B) is gated off: half the head slots idle, and FA went 5.0 -> 5.5 ms.

B390, llama-bench tg64, -d 4096, two order-balanced rounds (t/s):

| Model | KV | off | on |
|---|---|---|---|
| Qwen3-4B Q4_K_M (G=4) | f16 | 25.9 / 27.7 | 30.3 / 30.5 |
| Qwen3-4B Q4_K_M | q8_0 | 23.9 / 24.8 | 27.2 / 27.2 |
| SmolVLM2 Q4_K_M (G=3) | f16 | 202.1 / 198.4 | 227.2 / 229.3 |
| SmolVLM2 Q4_K_M | q8_0 | 139.8 / 139.3 | 157.2 / 158.1 |

Qwen3-4B FA op time at d4096 is 10.7 -> 6.6 ms. Vulkan at the same point is 30.7 (Qwen3-4B) and 237.6 (SmolVLM2). The gain shrinks with depth: SmolVLM2 is -1.5% at d256, +1.5% at d1024, +5% at d2048. Qwen3-4B at d256 is +1-3%.

Decode PPL (c512, 3 chunks, `-ub 1`), off / on: SmolVLM2 Q4_K_M 21.5017 / 21.5247, Qwen3-4B Q4_K_M 9.9317 / 9.8940. `test-backend-ops -o FLASH_ATTN_EXT` passes 5324/5324.

## Upstream Vulkan review: 1945e0920, 83dd71f86, fusions (2026-10-01)

Isolated `MUL_MAT q4_K m=4096,n=512,k=14336` on B390: Vulkan 16.3-22.4 TFLOPS, DX12 5.7-6.3, LinAlg default 7.0-8.4, LinAlg tiled wave16 mode 1 7.4-8.1 and mode 2 7.9-10.4. Vulkan's Q4_K GEMM is now about 2x ours after 1945e0920 (Intel coopmat1 warptiles, F16 B operand).

- 1945e0920 tile: our tiled GEMM (`mul_mat_linalg_tiled_i.hlsl`) with VP_BN=256 (512 threads, 128x256 like Vulkan's Xe2/Xe3 mmq tile) was flat against VP_BN=128 in modes 1/2 (order-balanced, three rounds). Not kept. The F16 B operand (convert src1 once, as flag 263 does for the wave GEMM, instead of converting in each tile load) is not tried yet. The non-LinAlg build has no matrix path, so this commit only maps to the LinAlg build.
- 83dd71f86 (F32 A two at a time): F32 matvec n=1 is at parity (2.09 vs 2.07 ms, ~113 GB/s). n=2/4 are 1.5x slower than Vulkan (3.06 vs 2.08 ms), but F32 weights are rare in GGUF models.
- 9ac8c408a: DX12 already fuses ADD+RMS_NORM+MUL and RMS_NORM+MUL+ROPE(+SET_ROWS). RMS_NORM+MUL+ADD (Gemma post-norm) is missing; none of the benchmark models use it.
- 64e9bceb2 (UNARY+MUL): llama-family FFNs use GLU, already handled. DX12_FUSE_MTP_GATE covers SIGMOID+MUL gates.
- 50182a53f: DX12 has no fused top-k MoE kernel. On Granite-1B-A400M Q4_K_M decode, ARGSORT+SOFT_MAX+GET_ROWS cost 0.29 of 6.2 ms (~4.6%), the upper bound for a fused router.

## Oct 2 pull review on B390 (2026-10-02)

Pulled the five origin/main commits (HC ops, SSM d96, sparse FA, GDN cache fusion, failure propagation, packed Q/K default) under the local GQA decode work. B390 results:

- `test-backend-ops -b DX120`: 18907/18907, no "missing pipeline" failures. New-op list 377/377 in both builds. LinAlg MUL_MAT/MUL_MAT_ID/FA 7649/7649.
- PPL (8 models, c512, 20 chunks): identical to the Oct 1 baseline in both builds.
- Packed Q/K (flag383) does not fire on B390 Qwen3: the graph takes the fl8/fl7 5-way SET_ROWS path, not flag104. `DX12_QK_NORM_PACKED=0` gives identical PPL, normal and `-ub 1`. No B390 exposure.
- FA pipeline and sparse FA do not apply here (not RDNA4/RTX5070, no DSA model).

### GDN cache fusion now default on B390

`dx12_gdn_cache_enabled()` now includes `dx12_is_b390_wave16()`. Qwen3.5-0.8B Q4_K_M, OFF vs ON:

| check | OFF | ON |
|---|---|---|
| PPL c512 4 chunks | 16.6267 | 16.6267 |
| PPL `-ub 1` | 16.6592 | 16.6592 |
| `test-recurrent-state-rollback` (c256, ub16) | pass | pass |
| tg128, 3 order-balanced pairs | 119.5 / 119.3 / 118.0 | 122.9 / 122.8 / 121.0 |
| pp512 | 2350 / 2359 / 2367 | 2326 / 2383 / 2400 |

tg is +2.8% in every pair. The profile shows the 0.29 ms state CPY (`K=128 N=262144`, 18 calls) gone and GDN up 0.02 ms. `DX12_GDN_CACHE_FUSION=0` restores the old path.

## LinAlg tiled GEMM: F16 activations from the pre-pass (2026-10-02)

The wave16 tiled GEMM (`DX12_LINALG_TILED_GEMM=1/2`) converted F32 B to F16 per tile, inside the K loop: two `Load4` and eight casts per thread per 32-K step. It now reuses the flag-263 convert pre-pass (same Q8_1 scratch and per-activation cache as the wave GEMM) and loads 8 halves with one `Load4` (`VP_B_F16`). Vulkan's Intel coopmat path also uses an F16 B operand. All eleven tiled types have `_w16_bf16` blobs. Default on; `DX12_TILED_B_F16=0` restores the old path. Wave32 modes 3/4 are unchanged (driver-gated anyway).

Isolated `MUL_MAT m=4096,n=512,k=14336`, order-balanced pairs, TFLOPS (old / f16-B):

| type | mode 1 | mode 2 |
|---|---|---|
| q4_K | 8.39, 7.15 / 9.53, 8.53 | 9.02, 9.06 / 9.32, 9.77 |
| f16 | 7.92, 7.98 / 9.21, 8.69 | 10.14, 9.61 / 11.65, 11.75 |
| q8_0 | 8.14, 8.05 / 8.66, 8.57 | 10.25, 10.31 / 11.42, 11.40 |
| q6_K | 6.52, 6.32 / 6.59, 6.52 | 7.89, 8.02 / 8.56, 8.50 |

Model pp512 (two order-balanced rounds, LinAlg build):

| model | default | m1 | m1 f16-B | m2 | m2 f16-B |
|---|---|---|---|---|---|
| Phi-3 Q4_K_M | 1033/864 | 1058/1020 | 1129/1118 | 1535/1291 | 1552/1548 |
| Phi-3 F16 | 1596/1613 | 1134/1151 | 1276/1280 | 1545/1536 | 1701/1708 |
| Qwen3-4B Q4_K_M | 983/850 | 985/917 | 977/978 | 1299/1192 | 1184/1275 |

PPL c512, 20 chunks (default / m1 f16-B / m2 f16-B): Phi-3 Q4 6.8637/6.8613/6.8592, Phi-3 F16 6.5132/6.5132/6.5153, Qwen3-4B Q4 11.6773/11.6659/11.6460, Granite Q4 11.2209/11.2275/11.2158. `test-backend-ops -o MUL_MAT` passes in modes 1 and 2 with and without f16-B; routes 331/331.

So the per-tile convert was a real cost after all (sec 38 priced the convert pre-pass for the wave GEMM, not the in-loop convert of the tiled one). Mode 2 + f16-B is about 1.5x the LinAlg default on Phi-3 Q4 prefill and +6% on Phi-3 F16, with no PPL loss on these four models. Mode 2 accumulates fully in F16, so a default promotion needs a wider quality matrix (large-activation models) first.

## Fused MoE router (flag 450, 2026-10-02)

The MoE router ran as three dispatches per layer: SOFT_MAX, the small top-k ARGSORT (fl52) and the GET_ROWS selected-weight norm (fl60). `moe_router.hlsl` does all of it in one wave per row: softmax over up to 256 experts (16 per lane), K rounds of wave argmax (lowest id wins ties), then optional sum/clamp/div normalization of the K weights. It writes the full softmax row, the top-k ids into the ARGSORT buffer, and the weights into the GET_ROWS or DIV output. Pattern: SOFT_MAX (no mask, scale 1) -> RESHAPE -> ARGSORT DESC -> VIEW (k <= 16) -> GET_ROWS, with optional RESHAPE -> SUM_ROWS -> CLAMP -> DIV. All outputs must share one buffer. Default on, `DX12_FUSE_MOE_ROUTER=0` disables it.

Granite-3.0-1B-A400M Q4_K_M decode, 24 router calls: ARGSORT 0.165 + SOFT_MAX 0.068 + GET_ROWS 0.052 ms -> one 0.13 ms dispatch.

Interleaved llama-bench (on / off), three order-balanced pairs: tg128 154.1/150.9, 153.0/148.8, 153.7/147.1 (+3.1%); pp512 1636/1683, 1657/1644, 1650/1651 (flat).

PPL c512, 20 chunks (on / off): f16 10.9815/10.9850, Q4_K_M 11.2172/11.2249, Q8_0 10.9707/10.9945. CORRECTION (2026-10-02): these deltas were a race, not sum order. In prefill the allocator puts the DIV weights inside the in-place logits/SOFT_MAX rows, so early weight writes clobber rows other waves have not read. The AMD integration added an overlap guard (see "Private-origin DX12 integration on AMD"). On the B390 with the guard, Granite PPL matches the unfused values exactly (10.9850/11.2249/10.9945). Prefill now fuses 0 of 23 router layers on Granite (no overlap-free layout), costing about 1.1 ms per 512-token batch (~0.35%); decode has no overlap and still fuses, which is where the +3% tg comes from. Interleaved A/B of the guarded vs racy DLL: Granite Q4 pp512 1641/1612 vs 1620/1633, tg 151.7/150.1 vs 150.8/150.7. `test-backend-ops` TOPK_MOE 416/416, SOFT_MAX, ARGSORT, GET_ROWS pass in both builds.

## AMD integrated-GPU fused GDN submission boundaries (2026-10-01)

The larger Qwen3.5-0.8B IQ2_XXS perplexity workload (`-c 512 -b 512 -ub 128 --chunks 2`) exposed a separate `DEVICE_HUNG` on the AMD integrated GPU with GDN cache fusion enabled. The short rollback matrix had passed. Fusion disabled completed normally; root constants with replay disabled still hung, and an extra UAV barrier did not resolve it. Per-op synchronization and submission boundaries after fused GDN dispatches both completed.

AMD integrated GPUs now submit each fused GDN dispatch separately outside command-replay capture. Fusion, shader arithmetic, cache writes and numerical thresholds remain unchanged. Other devices and unfused dispatches retain their submission policy. This resolves the observed command-list accumulation failure on PCI 13C0, driver 32.0.21045.1000; the experiments do not establish the driver's internal cause.

The exact two-chunk IQ2 workload passes in stock and preview with PPL 22.3903. Both saved distributions match the unfused reference byte-for-byte (SHA256 `07CD89B735A841ED4E6EE3097FA5488CF660A568AE4DE529556782C140AF4774`). Stock IQ3 also has identical fused/unfused distributions in a one-chunk control. Both builds pass the iGPU rollback OFF/ON cases and all 36 GATED_DELTA_NET fixtures. The combined attention/GDN selector additionally passes all 70 cases on both GPUs, both builds and root/CBV bindings, with positive fused-dispatch evidence in every arm (560 comparisons).

Evidence is in `diag-gdn-*`, `fixed-gdn-*`, `qual-gdnfix-*` and `qual-gdnsubmission-*` in the AMD qualification artifacts. Pre-fix runtimes and failed logs remain preserved. The broader model and performance campaign must use a fresh runtime manifest; these results alone do not establish full performance parity.

## Merged-main AMD qualification snapshot (2026-10-02)

The broad campaign completed after the three AMD fixes above, with `DX12_GDN_CACHE_FUSION=1` and `DX12_FA_PIPELINE=1`. It covers stock DX12, preview DX12 and Vulkan on the RX 9070 XT and AMD integrated GPU. This is an intermediate result, not a regression-free sign-off: short-prefill DX12 host overhead and Vulkan performance/numerical changes remain under investigation.

The dGPU matrix contains SmolLM2-135M, SmolVLM2-256M, Granite-3.0-1B-A400M, Phi-3-mini-4k, Qwen3-0.6B and Qwen3-4B, each with F16/BF16, Q8_0 and Q4_K_M weights. The iGPU matrix contains all three SmolLM2 formats, Granite Q4 and Qwen3-0.6B Q8/Q4. Performance uses a discarded full warm-up for each arm followed by old/new/new/old processes with three repetitions each: 72 complete model/backend/device cohorts, or 216 endpoint comparisons. dGPU endpoints are pp512, pp6144 and empty-context tg128; iGPU endpoints are pp128, pp512 and empty-context tg32. The prompt lengths do not imply that generation starts at those depths.

Median throughput changes across models:

| Device | Backend | Short prefill | Long prefill | Empty-context decode |
|--------|---------|---------------|--------------|----------------------|
| RX 9070 XT | stock DX12 | -0.70% | -0.06% | -0.38% |
| RX 9070 XT | preview DX12 | -0.53% | -0.14% | -0.45% |
| RX 9070 XT | Vulkan | -3.13% | -2.39% | -0.21% |
| AMD iGPU | stock DX12 | -0.04% | +0.07% | +0.03% |
| AMD iGPU | preview DX12 | -0.14% | -0.06% | -0.39% |
| AMD iGPU | Vulkan | -2.94% | -4.56% | -1.56% |

These medians do not hide individual regressions. A separate warmed new/old/old/new Smol Q8 comparison with ten raw timing samples per endpoint reproduces pp512 losses of 2.25% stock and 4.42% preview. Matched preview host profiles locate about 0.31 ms of extra time outside the backend graph interval; graph preparation, recording and submission are nearly unchanged. GPU shapes, flags and dispatch counts match, with only about 0.03 ms of added graph time. Batch conversion is a candidate, not yet a proven cause.

Vulkan profiling identifies separate losses: scalar D64 flash attention explains nearly all of the iGPU Smol F16 prefill slowdown, while two K576 Q8 matrix shapes explain nearly all of the sampled dGPU SmolVLM Q8 prefill slowdown. Other Q8 shapes improve. A blanket rollback of integer cooperative matrices is therefore not justified by this evidence.

Quality uses two 2048-token chunks on the dGPU and four 512-token chunks on the iGPU. Saved files contain quantized log probabilities, not raw logits. All 18 dGPU preview distributions match the preserved preview baseline byte-for-byte; five of six iGPU distributions also match. Same-backend historical perplexity tools were rebuilt separately because the preserved stock perplexity files had an incompatible older ABI and the preserved Vulkan runtime lacked that tool. The exact stock source baseline is `d131d541f`, whereas Vulkan is `f48b338bf`; the rebuilt historical tools are numerical controls, not performance baselines.

All 24 same-backend stock model/device controls are covered: 23 match byte-for-byte. Historical iGPU Granite Q4 varies across repeated runs; the current result is stable and matches two historical runs exactly, so the initial single-run difference is not a demonstrated regression. Vulkan has 10 byte-identical controls out of 24. Reproducible dGPU perplexity increases include Smol Q8 +0.3025%, Smol Q4 +0.3255%, SmolVLM Q8 +0.1878% and Granite Q8 +0.1628%; other quantized models decrease. These corpus-specific changes remain unresolved, rather than being dismissed as cross-backend differences. In contrast, the apparent stock Smol Q8 and Vulkan Phi F16 differences against preview were already present historically.

The recurrent IQ2/IQ3 matrix covers both DX12 builds, both GPUs and fusion off/on: all eight pairs have byte-identical saved distributions and generated response text. Eighteen multimodal smokes produce nonempty image descriptions; this is execution coverage, not a full multimodal quality benchmark. Vulkan ordinals changed during the campaign, so accepted runs resolve the physical GPU and check the workload's own identity log; multimodal Vulkan runs also check the projector backend. Rejected or unattributed earlier runs are archived separately. The pre-existing Vulkan iGPU short-K Q8 NaN remains a separately recorded baseline failure.

Canonical evidence is in `performance-comparison-{discrete,integrated}.json`, `temporal-matched-comparison.json`, `model-smokes.json`, `short-prefill-attribution-*` and `vulkan-attribution-*` in the AMD qualification artifacts. Production and preserved runtime hashes remained unchanged during historical-baseline reconstruction. Final regression fixes, rebuilt delivery binaries and the requested upstream optimization review are still pending.

## Legacy batch conversion overhead (2026-10-02)

The short-prefill host investigation confirmed unnecessary work in the legacy-to-extended batch conversion introduced by `fc343a84b`. Each token was built locally, including its sequence-ID unordered set, then copied into a growing vector. Conversion now reserves room for the incoming tokens and constructs each token in place. Existing destination entries, field conversion, position inference, sequence deduplication, output flags and embedding layouts are unchanged.

Matched temporary instrumentation measured conversion at 163.8 us before versus 27.2 us after, while allocator initialization stayed near 35 us. The 136.6 us conversion reduction agrees with the 134.7 us benchmark reduction in that paired experiment. Timings varied between campaigns, so they must not be combined with a different run's larger conversion cost to inflate the gain. No instrumentation is retained in the production change.

For unprofiled Smol Q8 pp512, a current/fixed/old/old/fixed/current comparison used 30 repetitions per process. Pooling repetitions 11-30, preview mean latency fell from 7.331 ms to 7.054 ms, with mean sample throughput up 3.91%. The preserved old runtime measured 6.991 ms, so the fix remains about 0.86% below that reference; it does not establish exact performance parity. Stock conversion also became cheaper, but stock unprofiled throughput improved only 0.36%, small relative to run variation. Separate long-prefill/decode controls did not establish a material new regression.

The existing batch-allocation test source passes all 50 tests and 327 assertions, including its compatibility cases for explicit/missing fields, multiple sequences, inferred positions, M-RoPE embeddings, mixed token/embedding input and encoder-width override. Smol Q8 and Granite Q4 perplexity comparisons have byte-identical saved distributions before/after in both DX12 builds. Candidate source and runtime evidence is under `.diagnostics\batch-compat-inplace.patch`, `.diagnostics\batch-attribution-report.txt` and `.diagnostics\batch-quality`; final production rebuild remains part of delivery.

## Vulkan scalar dense-attention regression (2026-10-02)

The sparse-attention update replaced scalar dense K/V and mask indexing with calls to `fa_kv_index()`. On the AMD iGPU, dense D64 Smol attention became about 17-19% slower despite unchanged tuning, explaining nearly all of the model's prefill loss. Keeping dense bounds and offsets explicit at the load sites restores the earlier performance. The sparse specialization still uses the existing index helper; precision, host dispatch, barriers and sparse gathering are unchanged.

A same-host comparison, changing only the Vulkan backend DLL, improves Smol F16 pp128 by 4.85% and pp512 by 5.63%. Its perplexity and saved distributions are byte-identical. Ten dense and eight sparse existing operator cases pass; a read-only review found no significant correctness issue. Mixed prefill/decode runs had large decode variation, while a separate final decode-only comparison was +0.10%; no decode improvement is claimed.

The independent shape-limited RDNA4 Q8 experiment is not promoted. It accelerates two K576 matrix shapes but improves model pp6144 only 1.52% and slightly increases perplexity versus the current path. Further native-F32-B versus integer-activation attribution remains separate. Evidence for both experiments is in `vk-experiments-20261002\final-measured-report.json`; only `fa-explicit-dense.patch` is applied to production source at this stage.

## Vulkan integer-activation arithmetic trade-off (2026-10-02)

Follow-up isolation identifies Q8_1 activation quantization as the cause of the tested dense-model distribution changes. Routing all eligible RDNA4 Q8 GEMMs through the existing floating cooperative-matrix path restores the exact historical saved distributions for Smol Q8, Qwen3-0.6B Q8 and Phi Q8. The path accepts F32 B inputs but still converts operands to half internally where applicable; this is not full-FP32 arithmetic.

Mixed Q4_K_M filenames do not describe every tensor. The Smol Q4 model contains 166 Q5_0, 15 Q8_0, 14 Q6_K and only 16 Q4_K tensors. Restoring the floating path consistently for Q8_0, Q5_0 and Q6_K restores the exact historical Smol and SmolVLM Q4 distributions. Changing Q4_K staging alone leaves both current distributions byte-identical, despite verified execution of the changed route. Thus Q4_K's separate F16 staging is not responsible for these numerical changes.

The arithmetic choice has a measured performance cost. Versus the current integer path, the all-Q8 floating route improves Smol pp512/pp6144 by 9.44%/7.81% and Qwen3-0.6B by 13.18%/7.95%, but slows Phi by 9.10%/8.23%. The combined Q8/Q5_0/Q6_K route improves Smol and SmolVLM Q4 prefill by about 2-5%. These results do not justify model-name gates or choosing whichever mixed arithmetic minimizes perplexity on this corpus. Granite expert routing was deliberately not changed, so no MoE-wide historical-compatibility claim follows.

No diagnostic arithmetic override is promoted. The delivered Vulkan build retains the merged upstream arithmetic policy rather than silently exchanging large-matrix gains for historical compatibility. Consequently, this campaign does not certify universal old/new numerical equivalence or eliminate every Vulkan performance regression. A production arithmetic-policy change requires an explicit compatibility/performance decision; the experimental patches and exact controls are preserved in `vk-experiments-20261002\native-attribution-summary.json`.

Additional sparse-attention performance controls cover three existing large-head fixtures in repeated old/new order. All 24 measurements pass, with median latency changes below one percent and no observed sparse regression from the dense-index fix.

## Review of the 44 incoming Vulkan-titled commits (2026-10-02)

All 44 actual diffs were reviewed, covering 144 changed-file entries. The earlier count of 48 included three Metal subjects and one CI subject whose commit bodies mentioned Vulkan. The canonical list, per-commit relevance/coverage ledger and archived-diff hashes are in `vulkan44-review-20261002`; no proposed optimization below is implemented or claimed as a measured gain.

| Priority | Experiment | Evidence and constraints |
|----------|------------|--------------------------|
| 1 | Tune RDNA4 short-K integer GEMM scheduling without changing activation arithmetic | Two K576 Q8 shapes account for about 14.66 ms of the sampled 14.80 ms SmolVLM pp6144 GPU increase. Try the existing smaller integer tile first, then separately examine BK_STEP and LDS/prefetch scheduling. Preserve Q8_1 generation, scale accumulation and the improving M192 control; check neighboring sizes and Phi before choosing any rule. Transfer only proven scheduling lessons to DX12, whose API/driver cooperative-matrix capabilities must be established independently. |
| 2 | Fuse the remaining DX12 MoE router boundaries | Granite Q4 decode still spends about 0.138 ms in softmax, top-K argsort and selected-weight normalization across 72 dispatches. DX12 already has small top-K and weight-normalization kernels. First confirm the actual fusible chain and consumers, then use allocation-dependency guards for every external input/output. This is about 9.7% of sampled kernel time, not a predicted 9.7% model speedup. Preserve tie ordering, clamping and normalization; do not enable broad graph reordering. |
| 3 | Remove work from partially occupied expert tiles | DX12 already selects tiles by routed rows, buckets expert work and skips empty workgroups. Collect actual Granite and second-model prefill row histograms before adding a wave-uniform mask for wholly inactive output strips. All shared loads and workgroup barriers required by live waves must still execute. Decode matvec time does not prove this prefill bottleneck. |

The specific upstream changes are not all missing DX12 features. `ggml-org/llama.cpp#29182` overlaps existing per-expert tile selection in both generic and LinAlg dispatch. `ggml-org/llama.cpp#29280` overlaps DX12's root-SRV GPU-address/PSO binding cache and command replay. The paired F32-load change in `ggml-org/llama.cpp#29254` is already present in the DX12 matvec shader. Their reported gains are not transferable estimates.

The lifetime handling in `ggml-org/llama.cpp#28422` is useful for a future router fusion: external input lifetimes must extend through all fused outputs, rather than assuming graph order prevents allocator aliasing. Idle CPU copies are not a reason to change AMD DX12 memory placement wholesale; the existing backend documents a substantial GPU-read penalty for mapped CUSTOM L0 heaps. Intel/Adreno/NVIDIA-specific tuning, new untested model formats and build refactors are not AMD speedup evidence.

The integer cooperative-matrix addition in `ggml-org/llama.cpp#27952` provides a real scheduling research target, but its Q8_1 activation arithmetic has the measured compatibility trade-off above. Neither blanket native routing nor a model/corpus-fitted gate is recommended. The next useful experiment is arithmetic-preserving tile selection, with existing operator checks, unchanged current-policy perplexity, and warmed interleaved model measurements.

## Final scalar-attention refinement and delivery qualification (2026-10-02)

Final multimodal comparison exposed deterministic vision-output drift from the initial broad dense-index rewrite, despite passing text distributions and operator tolerances. In the first differing SmolVLM vision attention layer, 55 of 786,432 outputs differed, with maximum absolute difference 0.000244140625. Later vision layers amplified this enough to change a close greedy token decision. Same-input FP64-reference errors were nearly equal, so this demonstrates output incompatibility, not a general loss of accuracy. The exact compiler/ISA rounding mechanism was not established.

The delivered shader retains explicit dense indexing only for mask/QK loads and restores the original V/PV indexing. Sparse indexing and arithmetic policy remain unchanged; no model or shape gate was added. Isolated comparisons restore all 84 attention outputs, seven embeddings and 48-step common-history decoder logits/tokens byte-for-byte. This supersedes the broader `fa-explicit-dense.patch` described above; `vk-fa-multimodal-20261002\scalar-fa-qk-only.patch` is the complete delivered shader change.

All six Vulkan image responses now match the pre-fix current baseline from the production build location, including RX 9070 XT Q8. Copied-runtime Q8 image controls had also faulted in `amdvlk64.dll` with unchanged shaders; that location-dependent driver behavior remains unexplained and is not hidden by a compatibility workaround. The production-location candidate succeeds normally.

Final delivery coverage combines the unchanged DX12 runtime results with the rebuilt, refined Vulkan results: 560 DX12 and 240 Vulkan supported operator comparisons, DX12 route checks, eight recurrent rollback arms, 72 byte-identical saved text distributions and 18 exact image responses. The text and image comparisons here are against the pre-fix merged-main cohort, not a claim that all historical Vulkan output differences disappeared. Earlier historical comparisons and the integer-activation trade-off remain applicable. Both requested DX12 fusion flags were enabled for the applicable coverage.

All three requested BoringSSL distributions contain `llama-cli`, `llama-bench`, `llama-mtmd-cli`, `llama-perplexity`, `test-backend-ops` and `test-recurrent-state-rollback`. Stock DX12 has LinAlg preview disabled; `build-dx12-linalg` has it enabled, with the non-LinAlg iGPU using its supported path. Evidence is in `final-qual-summary.json`, `delivery-vulkan-ops-summary.json`, `final-quality-{discrete,integrated}.json`, `delivery-quality-{discrete,integrated}.json`, `final-multimodal.json` and `delivery-vulkan-multimodal.json`. The immutable final executable/DLL hashes are in `delivery-runtime-hashes.json`; the earlier Vulkan final manifest is superseded.

Final targeted production benchmarks use two discarded warm-up processes followed by before/final/final/before, with ten raw samples per endpoint in each measured process. The reference is the preserved merged-main runtime before the batch and attention fixes, not the historical September baseline. Mean process throughput changes:

| Workload | Short prefill | Long prefill | Empty-context decode |
|----------|---------------|--------------|----------------------|
| RX 9070 XT, stock DX12, Smol Q8 | pp512 +0.58% | pp6144 +0.15% | tg128 -0.52% |
| RX 9070 XT, preview DX12, Smol Q8 | pp512 +5.12% | pp6144 +0.52% | tg128 -0.73% |
| AMD iGPU, Vulkan, Smol F16 | pp128 +4.85% | pp512 +5.56% | tg32 -1.08% |

The preview and Vulkan prefill gains reproduce in both final processes. Stock's small gain is within process variation. Decode samples overlap between arms, and the Vulkan before-process means range from 68.03 to 73.19 tokens/s; these runs do not establish a decode improvement or a new material regression. All 104 executable/DLL hashes still match the delivery manifest after benchmarking. Raw samples and aggregates are in `final-performance-results.json` and `delivery-performance-summary.json`.

This completes the bounded delivery campaign, not universal regression-free certification. The unchanged upstream Vulkan Q8_1 arithmetic policy retains the historical performance/numerical trade-offs above, and the historical Vulkan iGPU short-K Q8 NaN remains recorded as a pre-existing failure. No arithmetic override, threshold relaxation or speculative optimization from the 44-commit review is included.

## Private-origin DX12 integration on AMD (2026-10-02)

This follow-up pulls only this fork's `origin/main`, through `a0fc5bd43`, not official llama.cpp. Seven incoming commits add shared-head decode attention, router fusion, Intel tiled-GEMM changes, a B390 GDN default and runtime/fusion corrections. The existing three local AMD/runtime commits and the qualified batch, GDN and scalar-attention fixes are preserved. The overlapping Agility change keeps the explicit staging prerequisite so named consumer builds still stage the runtime. Vulkan arithmetic regressions are now upstream-owned, not blockers for DX12 integration.

The incoming router matcher needed an allocation-lifetime guard. ARGSORT and selected-weight outputs can reuse dead logits storage in the original multi-dispatch schedule, but their early writes race with other rows' logits loads after fusion. The matcher now rejects resource-relative byte overlap between logits and either later output, and between the three fused outputs. Normal in-place SOFT_MAX remains safe: its input and output share the exact row layout, and a wave reads its entire row before writing it. The change retains the unfused path for unsafe layouts rather than changing allocator policy.

Eight cases in the existing route suite exercise independent and combined IDs/weights aliases, with and without normalization, plus non-alias controls. The fixtures explicitly place the outputs in logits storage; merely adding a view to the test's separately allocated tensors would not reproduce graph-allocator reuse. Route introspection now records SOFT_MAX and makes the router checks available on the iGPU. CPU-reference output checks and flag-selection assertions both pass on both AMD GPUs in stock and preview builds.

The shared-head decode kernel is correct within the existing attention tolerances, but its incoming default is a substantial AMD performance regression. Same-binary, discarded-warm-up, OFF/ON/ON/OFF comparisons use five repetitions per measured process:

| Device/build and model | Context depth | GQA sharing OFF | GQA sharing ON | Throughput change |
|------------------------|---------------|-----------------|----------------|-------------------|
| RX 9070 XT, stock, Smol Q8 | 4096 | 708.37 tokens/s | 470.64 tokens/s | -33.56% |
| RX 9070 XT, preview, SmolVLM Q8 | 4096 | 710.85 tokens/s | 525.64 tokens/s | -26.06% |
| AMD iGPU, stock, Smol Q8 | 1024 | 86.84 tokens/s | 67.74 tokens/s | -22.00% |

AMD therefore retains the per-Q-head decode path by default. `DX12_FA_CD_GQA=1` still enables the shared-head experiment; explicit opt-out and non-AMD defaults are unchanged. The decision is per device, not a process-global vendor choice. Four route cases check explicit ON/OFF with F16 and Q8_0 KV, G=3 and a 513-row KV tail; a fifth covers the D96 Q8_0 shader specifically. No numerical threshold or driver compatibility workaround is added.

Shared-head default ON also changed four SmolVLM Q8 image responses; the AMD default gate restores all 12 stock/preview image responses exactly. The final explicit-opt-in attention and router matrix passes 5,776 supported CPU-reference comparisons across both builds, both GPUs and root/CBV parameters. Profiles cover flags 440/441/442/443/445; the added route assertion covers flag 444. Final route totals are 265 stock dGPU, 652 preview dGPU and 14 per iGPU build, each passing with root constants and CBV. The earlier focused attention/GDN, matmul and dense selectors pass 1,104 comparisons, and all eight recurrent rollback arms pass. The Intel tiled-F16-activation path and wave32 opt-in remain Intel-gated; these AMD results do not qualify enabling them on other architectures.

One unseeded TOPK_MOE run reported a near-tie selected-ID mismatch for the separate sqrt-softplus gate, which cannot match the new SOFT_MAX router fusion. The failure log is retained. Twelve matched-seed single-case controls pass on both the preserved baseline and candidate, as do both full 416-case seed-1234 controls. The final matrix uses that reproducible seed without relaxing its tolerances; the original unseeded input cannot be reconstructed.

Evidence is under `origin-main-20261002` in the session artifacts. `baseline-runtime-hashes.json` pins the pre-pull binaries; `gqa-controls.json` records the rejected default's performance; `gated-operator-results.json` and `gated-multimodal-results.json` contain the final extended coverage. The interrupted ungated performance campaign is preserved separately and is not counted as a complete comparison.

With the AMD-safe default, all 48 final stock/preview text distributions and reported perplexities match the qualified pre-pull baseline exactly. Final performance covers 16 model/build/device cohorts and 48 endpoints in warmed before/after/after/before order. Smol, SmolVLM and Qwen3-4B include empty and populated-context decode; Granite and Phi provide router and non-GQA controls. Granite Q8 tg128 improves 4.13% stock and 4.30% preview on the RX 9070 XT; a profile confirms 24 fused flag-450 router dispatches per decode graph. Granite Q4 and the iGPU controls are approximately flat. Router fusion remains enabled with the alias fallback.

Short LinAlg Smol prefill initially read about 3% slower, but the result did not survive longer attribution controls. The original r10 workload repeated at +0.63% (empty context) and +1.27% (depth 4096); r60 interleaving measured -0.20% and -0.73%. An isolated DLL crossover measured -0.22% for the merged pair and -0.43% for the DX12-only change. All 422 dispatches, flags and barrier decisions match, and measured graph-decision time is unchanged. Process-startup samples vary by several percent even after separate-process warmups. No speculative kernel, host or model-specific change is justified by this warning; its original results remain preserved.

The final matrix plus the sustained follow-up does not establish a material remaining AMD performance regression, but is not a claim of universal bit identity or zero timing overhead for arbitrary workloads. Detailed raw samples, quality metrics and the Smol attribution are in `gated-performance-comparison.json`, `final-quality-comparison.json` and `smol-attribution\findings.txt`. The router implementation has now arrived and been qualified, so the earlier proposed router experiment is no longer wholly unimplemented; further work should target remaining boundaries or allocation-safe opportunities rather than duplicate this fusion.

## LinAlg tiled GEMM: mode 2 default (2026-10-02)

Wave16 full-K F16 accumulation (`DX12_LINALG_TILED_GEMM=2`) with F16 activations is now the LinAlg default on wave16 Intel devices. F16 weights at n < 256 keep the wave GEMM. `DX12_LINALG_TILED_GEMM=0` restores the old routing. Upstream Vulkan also accumulates in F16 on Intel when the op precision is default, and both honor `GGML_PREC_F32`.

Quality: KLD vs the non-LinAlg build, wikitext c512, 10 chunks (mean KLD, default / m2 / Vulkan):

| model | default | m2 | Vulkan |
|---|---|---|---|
| Phi-3 F16 | 0.000011 | 0.000113 | 0.001996 |
| Phi-3 Q4_K_M | 0.000735 | 0.001013 | 0.002772 |
| Phi-3 Q8_0 | 0.000715 | 0.000804 | 0.002680 |
| Phi-3 Q5_0 | 0.000833 | 0.000972 | 0.002727 |
| Qwen3-4B F16 | 0.000009 | 0.000408 | 0.000382 |
| Qwen3-4B Q4_K_M | 0.000642 | 0.001172 | 0.001167 |
| Qwen3-4B Q8_0 | 0.000806 | 0.001168 | 0.001211 |
| Qwen3-0.6B Q8_0 | 0.001404 | 0.001638 | 0.001591 |
| Falcon-H1-7B Q4_K_M | 0.004845 | 0.005762 | 0.012359 |
| Falcon-H1-7B Q8_0 | 0.005156 | 0.005818 | 0.011925 |
| Granite-3.0 1B f16 | 0.000482 | 0.000512 | 0.000592 |
| Granite-3.0 1B Q4_K_M | 0.002643 | 0.002849 | 0.002846 |
| Granite-4.1 3B Q4_K_M | 0.001978 | 0.002606 | 0.002498 |
| SmolVLM2 Q4_K_M | 0.001733 | 0.001762 | 0.001779 |
| SmolLM2-360M Q8_0 | 0.002002 | 0.002092 | 0.002133 |
| Qwen3.5-0.8B Q4_K_M | 0.000792 | 0.000843 | 0.000855 |
| Phi-3.1 Q4_K_M | 0.000612 | 0.000902 | 0.002650 |

No overflow (Falcon-H1 has large activations). Q2_K and BF16 do not take this path. Phi-3 Q2_K PPL is 155.8 on DX12 and Vulkan alike; that file is bad.

Prefill, interleaved pp512 at ub512, two order-balanced pairs (m0 -> m2): Phi-3 F16 +7.4%, Q4_K_M +50.9%, Q8_0 +40.7%, Q5_0 +93.4%; Qwen3-4B F16 +2.4%, Q4_K_M +33.0%, Q8_0 +30.4%; Falcon-H1-7B Q4_K_M +27.7%; Granite-4.1 3B Q4 +42.5%; Granite-3.0 1B Q4 +1.9%; Qwen3-0.6B Q8 +21.5%; SmolLM2-360M Q8 +14.9%. At ub256: Phi-3 F16 -0.4%, quants +22..+41%. At ub128: Phi-3 F16 -8.8% (hence the F16 n < 256 gate), quants +6..+37%. Decode does not use this path.

`test-backend-ops -o MUL_MAT` 1374/1374; route assertions 331/331.

## NVIDIA UMA short-K Q4 recovery and opt-in D96 BR16 (2026-10-01)

On the NVIDIA RTX Spark N1X UMA device (wave32, driver 32.0.16.1662), matched September/current profiles attribute Granite Q4 decode's largest increase to dense projections rather than expert kernels. The current tuner chooses the 256-thread Q4 matvec globally, while the preserved runtime chooses 32 threads. For 48 K1024/N1024 projections per token, summed intrusive GPU time increases from 0.443 to 0.553 ms; expert timing is effectively unchanged. Both runtimes dispatch 498 nodes per sampled decode graph.

NVIDIA Pascal+ UMA wave32 now defaults to the existing 32-thread Q4_K dp4a shader for single-vector K1024 projections with 512 or 1024 output rows and no outer batching. Other shapes retain the tuner decision. Explicit `DX12_Q4K_DP4A_THREADS` and `DX12_TUNE_FORCE_Q4K_DP4A_32` overrides retain priority. This is a narrow measured shape policy, not a global 32-thread preference for larger models.

Phi-3 Q4 long-prefill profiles instead show D96 attention increasing from 339.68 to 383.16 ms per 512-query chunk, while dense matmul decreases from 265.76 to 261.20 ms. The preserved attention tile has 16 query rows; current stock has 32. Increasing per-group resources and reducing independent groups are plausible mechanisms, not hardware-counter measurements.

`DX12_FA_PF_D96_BR16=1` opts NVIDIA Pascal+ UMA wave32 into a 16-row D96 prefill tile. It is disabled by default; unset or 0 keeps the 32-row tile, and other devices remain unchanged. Flags 111/112 select ordinary/prescan BR16 blobs, with host query-group geometry matched to the shader. Both native-FP16 and F32 shader variants reuse the shared attention body and retain the existing precision/mask policies. Invalid values fail graph execution. Existing `DX12_FA_PF_PRESCAN=0` and `DX12_FA_PF_FP16=0` controls also work with BR16.

Balanced clean comparisons used four fresh processes per arm, three repetitions per process, full offload, FA on, F16 KV, batch2048/ubatch512, 18 threads and two-second repetition delays. Granite's control is the same candidate binary with `DX12_Q4K_DP4A_THREADS=256`; its candidate uses the new unset default. Phi's control/candidate differ only by BR16=0/1. The third arm is the complete preserved September runtime. No build or GPU profiling runs concurrently.

| Workload | Preserved September | Same-runtime control | Candidate | Candidate/control |
|---|---:|---:|---:|---:|
| Granite Q4_K_M tg512 | 265.80 tok/s | 253.04 tok/s | 264.34 tok/s | +4.47% |
| Phi-3 Q4_K_M pp6144 | 746.32 tok/s | 708.51 tok/s | 750.61 tok/s | +5.94% |

The Granite candidate is 0.55% below September with overlapping process ranges. Phi is 0.57% above September; the first Phi control/candidate pair is faster than later pairs, and all samples remain included. These results recover the sampled losses on this device, not a universal speed claim or approval to enable BR16 elsewhere.

The default full operator suite passes 18,852/18,852, including two short-K projection cases added to the existing test file. BR16 passes 5,324/5,324 attention cases with the usual mask policy, plus 122/122 D96 cases with prescan disabled and 122/122 with native-FP16 disabled. Coverage includes 75-query ragged tails, sinks, masks and KV views. Shader audits confirm both BR16 blobs execute, and existing overrides restore Granite flag 10 while its default uses 13.

Granite single-token WikiText PPL (c512/b1/ub1, eight chunks) is 10.6649 control versus 10.6607 candidate. Phi PPL (c2048/b2048/ub512, twenty chunks) is 5.8998 BR32 versus 5.8975 BR16. These bounded samples show no material quality loss, not raw-logit identity. BR16 remains opt-in as an experiment.

## MoE normalization raw-weight liveness (2026-10-02)

Both router normalization and selected-weight normalization require the raw reshaped weights to be consumed only by SUM_ROWS and DIV. Another reader of the raw weights previously observed stale storage because fusion wrote only the normalized output. The matchers also retain rows, reshape, sum and clamp intermediates marked as graph outputs.

Unsafe normalization layouts fall back without changing arithmetic or gates. The router can still fuse softmax, top-k and the raw gather; sum/clamp/div then execute separately. Ordinary graphs without external readers retain the full normalized fusion.

Twelve cases in the existing route suite cover normal graphs, an extra raw-weight reader and explicit outputs of the rows, reshape, sum and clamp intermediates, with both direct-softmax and biased selection. The biased cases exercise the older selected-weight normalization independently of the router. Both normal controls pass before the fix; the ten external-reader/output cases fail before and pass afterward.
