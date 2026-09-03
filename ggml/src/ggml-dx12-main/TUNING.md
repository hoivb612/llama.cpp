# ggml-dx12 cross-vendor tuning guide

How to measure, gate and validate a performance change in the DX12 backend
across AMD, NVIDIA and Intel parts.

Read [GOTCHAS.md](GOTCHAS.md) first - it lists the correctness traps that
will silently corrupt a "win" if you step on them.

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
- **Rebuild the agility stage.** The build target list must always include
  `ggml-dx12-agility-stage`; without it a rebuilt shader is not staged and
  the benchmark silently measures the *old* blob.
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
| `DX12_PHASE_PROFILE=1` | per-phase timing |
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
  collisions after the merge** - git will not flag them.

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

**AMD RDNA4 - LinAlg driver defects.** See `AMD_LinAlg_Driver_Bug.md` on the
`dx12-linalg-phase0` branch. A
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
| Intel D=64 prefill FA | mask prescan blob | `DX12_FA_PF_PRESCAN=0` | pp6144 +13% granite, +46-52% Smol |
| Intel D=96/128 prefill FA | mask prescan blob | `DX12_FA_PF_PRESCAN=0` | pp4096 +5.8% Phi-3, +5.3% Qwen3-4B |
| Quantized MoE GEMM | tall BM=128 tile when >=128 pairs/expert | `DX12_MOE_GEMM_TALL=0` | granite Q4_K_M pp512 +17%, pp6144 +15% |
| Quantized KV cache | q8_0 + legacy-quant `SET_ROWS` enabled | `DX12_SET_ROWS_Q8_0=0`, `DX12_SET_ROWS_LEGACY_QUANT=0` | `-ctk/-ctv q8_0`/`q4_0` work; SET_ROWS 135/135 |
| Q8_0 MMID | 64-thread DP4A workgroup | `DX12_MOE_Q8_G64=0` | Granite pp512 +34% |
| F16 decode | wave64 matvec | `DX12_F16_WAVE64=0` | SmolLM2 decode +28% |
| K<=1024 Q8_0 SwiGLU | retain the DP4A GLU kernel instead of folding RMS norm into a scalar kernel | `DX12_Q80_GLU_RMS_FOLD=1` restores the fold | Smol models decode +8% |

Strix Point UMA (device ID `0x150E`, including Radeon 880M/890M) also uses
the large-K Q8_0 and D=64 prefill FA defaults above. On Radeon 880M, skipping
`mr256v` improved Qwen3 Q8_0 decode by 22% and Phi-3 Q8_0 decode by 16%;
the wide FA shader improved F16 pp512 by 12%, pp2048 by 25%, and pp6144 by
54%. The other RDNA1/2 UMA defaults remain unchanged for this device.

### AMD projection fusion defaults

BF16 gate/up/SwiGLU fusion is default-on for AMD RDNA1/2 and RDNA4. It
improved Qwen3-0.6B decode by about 1% on RDNA4 and 3-5% on RDNA2. RDNA3
remains unchanged pending measurements. `DX12_MMV_GLU_FUSION_BF16=0`
disables it.

Q/K/V resource partitioning is enabled for F16, BF16, and Q8_0 on RDNA4.
The measured Qwen3-0.6B decode gains were 1.3-1.4%, 3.6%, and 2.8%
respectively. RDNA1/2 remains disabled after neutral F16/BF16 results and
a 5.6% Q8_0 regression. `DX12_QKV_RESOURCE_PARTITION=0` disables the route.

---

## 7. Rejected experiments

Recorded so they are not retried. Full context and measurements are in
`docs/linalg_how_to_setup.md` on the `dx12-linalg-phase0` branch.

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

GLU/UNARY `supports_op` also test `F32 || F16`, but those ops run on F32
activations even under BF16 weights, so BF16 there is moot.

When adding a type to a kernel, grep for the flag number and check every
gate that mentions it.

---

## 7c. Q4_K MoE decode: dp4a, and why one thread must own a whole sub-block

Q8_0 had a dp4a MMID kernel (fl=17) and Q4_K had none. The Q8_1 activation
pre-pass is already type-agnostic, so the only missing piece was the kernel:
`shaders/mul_mat_id_q4k_dp4a.hlsl`, fl=117, env `DX12_MOE_Q4K_DP4A`.

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

## 8. MoE decode routing and MTP gate fusion

Granite MoE decode exposed three avoidable routing and aggregation costs:

- Apply the selected router weight in the down-projection MMID epilogue,
  then reduce selected expert outputs with one `moe_sum` dispatch.
- Select small descending top-K prefixes directly instead of sorting a
  padded 1024-entry row.
- Fuse `GET_ROWS -> RESHAPE -> SUM_ROWS -> CLAMP -> DIV` router-weight
  normalization.

On Granite 3.0 1B A400M Q4_K_M, RX 9070 XT, these changes improved tg64
from 296.37 +/- 2.44 to 341.56 +/- 2.68 tokens/s, a 15.2% gain.
Perplexity was identical at 3.6711 on the same 512-token corpus.

The small top-K shader sizes its group scratch for 32 waves so a 256-thread
group is safe on Intel UHD wave8 hardware. A/B controls are
`DX12_NO_FUSE_MOE_SUM`, `DX12_NO_SMALL_TOPK`, and
`DX12_NO_FUSE_MOE_WEIGHT_NORM`.

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

Qwen3.5 MTP fuses `CONT(mtp_gate) -> SIGMOID -> MUL(attention)`. Direct
dumps at 8,192, 45,056, and 16,384 elements were bit-identical, two
dispatches were removed per graph, and steady graph time improved by about
0.4-0.6%. Set `DX12_NO_FUSE_MTP_GATE=1` for the unfused path.

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

`DX12_FA_PF_PRESCAN=0` remains the kill switch. Other vendors are still
excluded - the blob has only been measured on the two Intel families.

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
