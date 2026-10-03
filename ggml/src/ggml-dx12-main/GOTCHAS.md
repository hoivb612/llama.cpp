# DX12 Backend — Known Gotchas

## NVIDIA Q4_K Precision

Q4_K/Q5_K/Q6_K quantized batch MUL_MAT can produce cumulative precision errors on NVIDIA GPUs (tested RTX 6000 Ada). Intel (UHD, B390) and AMD (880M) produce correct results. The root cause appears to be NVIDIA's shader JIT floating-point behavior.

The dp4a matvec path (`mul_mat_vec_q4k_dp4a.hlsl`) adds Q8_1 activation quantization on top of Q4_K weight quantization. To avoid amplifying NVIDIA precision drift, the dispatcher in `ggml-dx12.cpp` gates the dp4a Q4_K matvec (`flags=10`) on `!nvidia`, falling back to the float multi-row path (`flags=9`) on NVIDIA. Intel and AMD use the dp4a path for ~1.5-2x throughput on Q4_K matvec.

Exception: the Pascal+ Tegra iGPU (GB20B, `arch_family == DX12_ARCH_NV_PASCAL_PLUS && is_igpu`) re-enables `flags=10`. The precision drift above was measured in the batched GEMM path on discrete Ada; the M=1 matvec path was re-validated on the GB20B via perplexity forced through the matvec route (`llama-perplexity -ub 1`, which routes single-token eval as `ne[1]==1`). Phi-3 Q4_K PPL was 3.0523 (dp4a) vs 3.0495 (scalar) - a 0.09% delta, well inside the +/-0.18 stderr - for +9.3% decode throughput. Discrete NVIDIA stays on the scalar path.

## DXC Internal Compiler Error with `dot4add_i8packed`

DXC 1.8.2502.11 crashes (ICE) when `dot4add_i8packed` is called with a literal `0` as the accumulator:

```hlsl
// BAD — causes DXC ICE
int r = dot4add_i8packed(a, b, 0);

// GOOD — use a variable instead
int r = 0;
r = dot4add_i8packed(a, b, r);
```

## ByteAddressBuffer Alignment (NVIDIA)

`ByteAddressBuffer.Load()` requires 4-byte aligned addresses. Intel and AMD tolerate misaligned loads, but NVIDIA returns incorrect data silently. Always use `buf.Load(addr & ~3u)` and shift to extract sub-word data when the address may not be aligned.

## Wave/SIMD Size Variation

DX12 wave size varies by vendor: NVIDIA=32, AMD=64, Intel UHD=8–16. Two-level WaveActiveSum reduction (wave + shared memory) fails when `num_waves > wave_size`. Use tree reduction on shared memory for the second level to ensure cross-vendor correctness.

## Integer-dot GEMMs (`mul_mat_*_q8_1_mmq.hlsl`) — cross-vendor status

`mul_mat_q8_0_q8_1_mmq`, `mul_mat_q4k_q8_1_mmq`, `mul_mat_q5k_q8_1_mmq` and
`mul_mat_q6k_q8_1_mmq` are register-blocked prefill GEMMs (128x64 tile, 32
accumulators per thread, quad-major groupshared tiles) that replace the 32x32
integer tile for wide outputs. They need only SM 6.6 `dot4add_i8packed`.

Selection: automatic when the output width `ne[0] >= 256` and `ne[1] >= 64`.
`DX12_MMQ_MIN_N` overrides the width threshold; `DX12_MMQ_MIN_N=0` disables the
path entirely, which is the A/B switch.

**Measured on AMD** (RX 9070 XT and an RDNA2 iGPU): Phi-3-mini
pp4096 Q8_0 1186->2801, Q4_K_M 1124->2569, Q6_K 1108->2192, Qwen2.5-3B Q5_K_M
1478->3359, all with perplexity and decode unchanged.

Measured on RTX 6000 Ada across SmolLM2-135M, Qwen3-4B, and Phi-3: Q8_0 gained
14-349% and Q4_K_M gained 2-283% from pp32 through pp6144. The smallest Q4_K_M
result repeated at +5% in reverse order. F16 routing is unchanged, and the
enabled DX120 MUL_MAT suite passed 1013/1013. Tiled integer-dot therefore
defaults on for discrete NVIDIA Pascal-or-newer devices; `DX12_TILED_INTDOT=0`
retains the full-path kill switch.

**Validated on the GB20B NVIDIA Pascal+ iGPU** (also default-on via the
`tiled_intdot` auto set): Phi-3/Qwen3-4B Q4_K_M and Q8_0 pp512 +223-382%,
decode unchanged, and Phi-3 Q4_K 47-chunk PPL 5.8235 against CPU fp32 5.8313 /
float-dequant 5.8190 (within 0.2%). Q4_K/Q5_K/Q6_K/Q8_0 all exercised;
Qwen3-4B Q5_K_M PPL 6.7909 against CPU fp32 6.7623 (within the error bar) with
pp512 +504%. `DX12_TILED_INTDOT=0` opts out.

All weight loads go through `Load(addr & ~3u)` plus funnel shifts (Q8_0's
34-byte and Q6_K's 210-byte blocks are not 4-byte aligned), so the NVIDIA
alignment rule above is respected. Intel UHD is deliberately excluded and keeps
the flag-59 aligned-byte-extraction variant.

## Root SRVs are not bounds checked, so speculated loads can fault the device

The buffers bound with `SetComputeRootShaderResourceView` (weights, K/V, mask)
are *root* SRVs. D3D12 clamps out-of-bounds reads through descriptor-table
SRVs, but root SRVs get no bounds checking at all: an over-read walks straight
into whatever follows the allocation, and if that page is unmapped the device
page-faults and the whole submission dies as `DEVICE_HUNG`.

The funnel-shift helpers above are where this bites. They load one extra
trailing word that is only needed for a misaligned start, and they guard it:

```hlsl
// BAD - the guarded load is still speculated by DXC
if (shift == 0u) { packed1 = w1; }
else { uint w2 = buf.Load(aligned + 8u); ... }

// GOOD - the address is in bounds no matter which way the branch goes
uint w2 = buf.Load(aligned + (shift == 0u ? 4u : 8u));
```

An early `return` in front of the load is no protection either; DXC flattens
that just the same.

This is easy to miss because it is silent almost everywhere: a tensor in the
middle of an allocation just over-reads into its neighbour. It only faults when
the tensor ends flush with the end of the allocation - in the KV cache that is
the last layer's V, which is why it surfaced as a TDR only at large N_kv, and
always on the final FLASH_ATTN_EXT node. See GRAPH_REORDER_PLAN.md section 18.

Rule: any `Load` whose address is only valid under a condition must be made
unconditionally in-bounds. When touching these helpers, grep the shader tree
for `Load(aligned +` / `Load(base +` rather than assuming one call site.

## Never size a root-descriptor scratch buffer to an exact fit

The same missing bounds check bites allocations, not just shader addressing.
The hardware reads a root descriptor at cacheline granularity, so even a
provably in-range access to the *last* element of a buffer can pull in the
bytes that follow. If the allocation ends on a page boundary and the next page
is unmapped, the device hangs.

This is load-bearing for `argsort_scratch` and `q8_1_scratch`, both of which
are bound as root descriptors and sized from the workload. Both over-allocate
by `dx12_device::DX12_ROOT_SCRATCH_SLACK` while recording only the usable
capacity. Do not "tidy up" the extra bytes.

Two properties make this miserable to debug:

- GPU-based validation cannot see it. Root descriptors carry no size, so GBV
  has nothing to bounds-check against and reports a clean run right up to the
  `DEVICE_HUNG`. Its silence is not evidence.
- It is allocation-history dependent. A geometric growth policy means the
  exact-fit case only occurs for particular size sequences, and the page after
  the allocation is only unmapped when something else has churned the address
  space. `test-backend-ops -o ARGSORT` alone always passed; only
  `-o MUL_MAT,ARGSORT` hung.

Intel's driver reports the resulting page fault as `DEVICE_HUNG (0x887A0006)`,
whose D3D12 message blames a slow kernel. Time the dispatch before believing
it: the ARGSORT case that produced this message ran in 0.59 s against a 2 s
watchdog.
