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
