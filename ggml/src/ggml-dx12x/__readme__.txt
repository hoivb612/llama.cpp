● Here's the complete MUL_MAT → DX12 shader mapping:

  --------------------------------------------------------------------------------------------------------------------------------------

  Pipeline Key

  Every tensor op is mapped via dx12_pipeline_key:

   { op=GGML_OP_MUL_MAT, src0_type, src1_type, dst_type, flags }

  The flags field selects the shader variant:

  ┌───────┬─────────────────────────────┬───────────────────────────────────────────┐
  │ flags │ Path                        │ When used                                 │
  ├───────┼─────────────────────────────┼───────────────────────────────────────────┤
  │ 0     │ Generic/flat matmul         │ Batch (M>1) for types without WMMA shader │
  ├───────┼─────────────────────────────┼───────────────────────────────────────────┤
  │ 1     │ Matvec (32 threads)         │ Token generation (M=1)                    │
  ├───────┼─────────────────────────────┼───────────────────────────────────────────┤
  │ 4     │ WMMA tiled (32×32)          │ Batch (M>1) for Q4_K/Q5_K/Q6_K/F16/F32    │
  ├───────┼─────────────────────────────┼───────────────────────────────────────────┤
  │ 5     │ Alt variant (256t or load4) │ Matvec if auto-tune selects it            │
  └───────┴─────────────────────────────┴───────────────────────────────────────────┘

  --------------------------------------------------------------------------------------------------------------------------------------

  Step 1: Flag Assignment (graph traversal, ~line 1431)

   if M == 1:  → flags=1 (matvec)     [auto-tune may override to flags=5]
   if M > 1 AND type ∈ {Q4_K,Q5_K,Q6_K,F16,F32}:  → flags=4 (WMMA tiled)
   if M > 1 AND type ∈ {Q8_0,Q5_0,Q4_0,...}:       → flags=0 (flat fallback)

  --------------------------------------------------------------------------------------------------------------------------------------

  Step 2: Shader Selection (get_or_create_pipeline, ~line 2444)

  flags=4 (WMMA tiled batch, M>1):

  ┌───────────┬─────────────────────────┐
  │ src0_type │ Shader                  │
  ├───────────┼─────────────────────────┤
  │ Q4_K      │ mul_mat_q4k_wmma.hlsl   │
  ├───────────┼─────────────────────────┤
  │ Q5_K      │ mul_mat_q5k_wmma.hlsl   │
  ├───────────┼─────────────────────────┤
  │ Q6_K      │ mul_mat_q6k_wmma.hlsl   │
  ├───────────┼─────────────────────────┤
  │ F16/F32   │ mul_mat_wmma.hlsl       │
  └───────────┴─────────────────────────┘

  flags=1 (matvec, M=1):

  ┌───────────┬─────────────────────────┬─────────────────────────────┐
  │ src0_type │ Shader (32t)            │ Shader (256t, flags=5)      │
  ├───────────┼─────────────────────────┼─────────────────────────────┤
  │ Q4_K      │ mul_mat_vec_q4k.hlsl    │ —                           │
  ├───────────┼─────────────────────────┼─────────────────────────────┤
  │ Q5_K      │ mul_mat_vec_q5k.hlsl    │ —                           │
  ├───────────┼─────────────────────────┼─────────────────────────────┤
  │ Q6_K      │ mul_mat_vec_q6k.hlsl    │ mul_mat_vec_q6k_32.hlsl     │
  ├───────────┼─────────────────────────┼─────────────────────────────┤
  │ Q5_0      │ mul_mat_vec_q5_0.hlsl   │ mul_mat_vec_q5_0_256.hlsl   │
  ├───────────┼─────────────────────────┼─────────────────────────────┤
  │ Q8_0      │ mul_mat_vec_q8_0.hlsl   │ mul_mat_vec_q8_0_256.hlsl   │
  ├───────────┼─────────────────────────┼─────────────────────────────┤
  │ F16/F32   │ mul_mat_vec.hlsl        │ mul_mat_vec_load4.hlsl      │
  └───────────┴─────────────────────────┴─────────────────────────────┘

  flags=0 (flat fallback, batch with no WMMA):

  ┌───────────┬──────────────────────────────┐
  │ src0_type │ Shader                       │
  ├───────────┼──────────────────────────────┤
  │ Q4_K      │ mul_mat_q4k.hlsl             │
  ├───────────┼──────────────────────────────┤
  │ Q8_0      │ mul_mat_q8_0.hlsl            │
  ├───────────┼──────────────────────────────┤
  │ Q5_0      │ mul_mat_q5_0.hlsl            │
  ├───────────┼──────────────────────────────┤
  │ etc.      │ mul_mat_q*.hlsl              │
  ├───────────┼──────────────────────────────┤
  │ F16/F32   │ mul_mat.hlsl (16×16 tiled)   │
  └───────────┴──────────────────────────────┘

  --------------------------------------------------------------------------------------------------------------------------------------

  Step 3: Dispatch Dimensions (~line 1681)

  ┌───────────────┬─────────────────────────────────────────────────────────────────────────────────┐
  │ Path          │ Thread groups                                                                   │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ Matvec        │ (N, 1, batches) — one group per output row                                      │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ WMMA tiled    │ ((N+31)/32, (M+31)/32, batches) — 32×32 output tiles, [numthreads(16,16,1)]     │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ Flat quant    │ ((N×M×batches+255)/256, 1, 1) — 1 output/thread, 256 threads/group              │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ F16/F32 tiled │ ((N+15)/16, (M+15)/16, batches) — 16×16 tiles                                   │
  └───────────────┴─────────────────────────────────────────────────────────────────────────────────┘

  --------------------------------------------------------------------------------------------------------------------------------------

  Step 4: Parameters (Root Constants)

  All shaders share one root signature. Slot 0 = dx12_shader_params (≤64 DWORDs):

   src0: ne[4], nb[4], offset, esize    ← weight matrix (quantized)
   src1: ne[4], nb[4], offset, esize    ← activation (F32)
   dst:  ne[4], nb[4], offset, esize    ← output (F32)
   op_params[16]                        ← fusion flags, bias offset, etc.

  GPU bindings:

   - Slot 1 (SRV t0): src0 buffer (weight)
   - Slot 2 (SRV t1): src1 buffer (activation)
   - Slot 3 (UAV u0): dst buffer (output)
   - Slot 4 (SRV t2): optional (fused bias)

● SRV = Shader Resource View — read-only access to a GPU buffer. In HLSL, these are ByteAddressBuffer (registers t0, t1, etc.). Used for
  input tensors (weights, activations) that the shader reads but never writes.

  UAV = Unordered Access View — read/write access to a GPU buffer. In HLSL, these are RWByteAddressBuffer (register u0). Used for the
  output tensor that the shader writes results to.

  In the DX12 backend's root signature:

  ┌──────┬──────────────────┬───────────────┬─────────────────────────────────────────────────┐
  │ Slot │ Type             │ HLSL Register │ Purpose                                         │
  ├──────┼──────────────────┼───────────────┼─────────────────────────────────────────────────┤
  │ 0    │ Root Constants   │ b0            │ dx12_shader_params (shapes, strides, offsets)   │
  ├──────┼──────────────────┼───────────────┼─────────────────────────────────────────────────┤
  │ 1    │ SRV (read)       │ t0            │ src0 — weight matrix                            │
  ├──────┼──────────────────┼───────────────┼─────────────────────────────────────────────────┤
  │ 2    │ SRV (read)       │ t1            │ src1 — activations                              │
  ├──────┼──────────────────┼───────────────┼─────────────────────────────────────────────────┤
  │ 3    │ UAV (read/write) │ u0            │ dst — output                                    │
  ├──────┼──────────────────┼───────────────┼─────────────────────────────────────────────────┤
  │ 4    │ SRV (read)       │ t2            │ src2 — optional (bias, KV cache)                │
  ├──────┼──────────────────┼───────────────┼─────────────────────────────────────────────────┤
  │ 5    │ SRV (read)       │ t3            │ src3 — optional (attention mask)                │
  └──────┴──────────────────┴───────────────┴─────────────────────────────────────────────────┘

  The distinction matters because D3D12 enforces it at the hardware level — writing to an SRV is a GPU fault, and the driver can optimize
  read-only resources differently (caching, prefetch). Vulkan has a similar concept (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER for read/write vs 
  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER for read-only).

   --------------------------------------------------------------------------------------------------------------------------------------

● The Three MUL_MAT Paths

  For C[N×M] = A[N×K] × B[K×M] where A=weights (quantized), B=activations (F32):

  --------------------------------------------------------------------------------------------------------------------------------------

  🟢 Matvec (M=1, token generation)

  One output = one dot product. All threads in a group cooperate on a single row.

   Thread group #row (32 or 256 threads):
     Each thread handles a chunk of K elements
     Partial sums → wave reduction → one final C[row]

   - Groups launched: N (one per output row)
   - Why fast: All threads collaborate on one K-length reduction. Wave shuffles eliminate shared memory barriers. Full memory bandwidth 
  utilization.
   - Use case: Generating one token at a time

  --------------------------------------------------------------------------------------------------------------------------------------

  🟢 WMMA Tiled (M>1, batch prompt)
  
A 16×16 thread group computes a 32×32 output tile using register blocking (each thread owns a 2×2 sub-tile).

 Thread group at tile (tx, ty):
   for k_chunk in range(0, K, TILE_K):
     Load A[ty*32..ty*32+31, k_chunk] into shared memory (dequantize)
     Load B[k_chunk, tx*32..tx*32+31] into shared memory
     barrier()
     Each thread accumulates its 2×2 sub-tile from shared data
   Write 32×32 output tile to C

 - Groups launched: (N/32) × (M/32) × batches
 - Why fast: Each weight element loaded from memory is reused 32 times (across 32 columns of B). Shared memory acts as a fast cache. 
Arithmetic intensity: O(32) ops per memory load.
 - Use case: Processing 556 tokens of prompt at once

--------------------------------------------------------------------------------------------------------------------------------------

🔴 Flat Quant (M>1, no WMMA shader)

One thread = one output element. No cooperation, no tiling, no data reuse.

 Thread #idx:
   n = idx / M;  m = idx % M
   sum = 0
   for k in range(K):
     sum += dequant(A[n, k]) * B[k, m]
   C[n, m] = sum

 - Groups launched: (N×M×batches) / 256
 - Why slow: Every thread independently reads the entire K dimension. The same weight row gets loaded N×M times across all threads. 
Zero data reuse. Completely memory-bandwidth bound.
 - Use case: Fallback when no tiled shader exists (Q8_0, Q5_0, Q4_0, etc.)

--------------------------------------------------------------------------------------------------------------------------------------

Performance Comparison (for 556-token prompt, N=3072, K=3072)

┌────────────┬────────────┬──────────────────────────────────┬────────────────────┐
│ Path       │ Data reuse │ Effective reads of weight matrix │ Relative speed     │
├────────────┼────────────┼──────────────────────────────────┼────────────────────┤
│ WMMA Tiled │ 32×        │ ~17× (556/32 tiles)              │ 1× (baseline)      │
├────────────┼────────────┼──────────────────────────────────┼────────────────────┤
│ Flat       │ 1×         │ 556× (once per M row)            │ ~30-60× slower     │
└────────────┴────────────┴──────────────────────────────────┴────────────────────┘

The flat path reads weights 556× more than it needs to. That's why Q8_0 (flat) was 128 seconds while Q6_K (WMMA) was 2 seconds for the
same prompt.

--------------------------------------------------------------------------------------------------------------------------------------

shaders are GPU-agnostic at the HLSL source level, but can be written to exploit specific GPU
features for dramatic performance gains.

The Compilation Chain

 HLSL source → DXC compiler → DXIL bytecode → GPU driver → native ISA
                                                 (at load time)

DXIL is an intermediate representation (like LLVM IR). The GPU driver's JIT compiler converts it to the actual machine instructions for
your specific GPU (RDNA 3, Ada Lovelace, etc.). So the same DXIL runs on any D3D12-capable GPU — it's portable.

But Performance is NOT Portable

The same shader can run 10× faster or slower depending on how well it maps to the hardware. Key GPU-specific differences:

┌─────────────────────────────┬───────────────────────────┬──────────────────────────────┬──────────────┐
│ Resource                    │ AMD RDNA 3 (your 7700 XT) │ NVIDIA Ada (RTX 4xxx)        │ Intel Arc    │
├─────────────────────────────┼───────────────────────────┼──────────────────────────────┼──────────────┤
│ Wave/warp size              │ 32 or 64                  │ 32                           │ 16           │
├─────────────────────────────┼───────────────────────────┼──────────────────────────────┼──────────────┤
│ Shared memory (LDS)         │ 64 KB per workgroup       │ 48-100 KB                    │ 64 KB        │
├─────────────────────────────┼───────────────────────────┼──────────────────────────────┼──────────────┤
│ Cache line                  │ 128 bytes                 │ 128 bytes                    │ 64 bytes     │
├─────────────────────────────┼───────────────────────────┼──────────────────────────────┼──────────────┤
│ Tensor cores / Matrix units │ None (WaveMMA future)     │ Yes (via Cooperative Vector) │ XMX units    │
├─────────────────────────────┼───────────────────────────┼──────────────────────────────┼──────────────┤
│ FP16 throughput             │ 2× FP32                   │ 2× FP32                      │ 2× FP32      │
├─────────────────────────────┼───────────────────────────┼──────────────────────────────┼──────────────┤
│ Memory bandwidth            │ 432 GB/s                  │ 504-1008 GB/s                │ 256-512 GB/s │
└─────────────────────────────┴───────────────────────────┴──────────────────────────────┴──────────────┘

What GPU-Specific Shaders Can Exploit

1. Wave size tuning — This is exactly what the auto-tune does:

 Q8_0 K=3072: 32t=13884  256t=250904 ticks  → 32-thread variant 18× faster on AMD

AMD RDNA uses wave32 natively. A shader designed for 256-thread groups with wave64 assumptions would be suboptimal.

2. Shared memory (LDS) tile sizes — The WMMA shaders use 32×32 tiles. But RDNA 3 has 64 KB LDS which could support larger tiles
(64×32), increasing data reuse. NVIDIA's 48 KB would need smaller tiles.

3. Memory access patterns — RDNA 3's cache hierarchy is different from NVIDIA's. Coalesced reads of 128-byte cache lines matter, but
the optimal access pattern (row-major vs column-major traversal) depends on the GPU's L1/L2 topology.

4. Hardware matrix acceleration — The DX12 backend already checks for this:

 // Cooperative Vector (NVIDIA, requires Agility SDK)
 cooperative_vector_supported = ...;   // → mul_mat_cv.hlsl
 
 // WaveMMA (SM 6.9, future AMD/Intel)
 wave_mma_supported = ...;            // → hardware matrix multiply

When available, these replace hundreds of scalar multiply-add instructions with a single hardware matrix op (e.g., 16×16×16 matmul in
one cycle).

5. Dequantization strategy — Q4_K packs 4-bit weights in a specific layout. The optimal unpack sequence (bit shifts, masks, LUT vs
arithmetic) varies by GPU ALU design.

What Would Help Your RX 7700 XT Specifically

The current shaders were likely developed primarily for NVIDIA. For AMD RDNA 3:

 - Wave32 native dispatches — avoid 256-thread groups that force wave64 scheduling overhead
 - Larger LDS tiles — RDNA 3 has 64 KB LDS, could do 64×32 WMMA tiles for 2× data reuse
 - VGPR-conscious register blocking — RDNA 3 has 256 VGPRs/wave32; more aggressive register blocking (4×4 sub-tiles instead of 2×2) 
could hide memory latency
 - Q8_0 WMMA shader — the missing one that caused your 128s TTFT

In short: the HLSL/DXIL portability guarantees correctness across GPUs, but performance requires GPU-aware shader design — tile sizes,
wave sizes, memory patterns, and hardware intrinsics all need to match the target architecture.

--------------------------------------------------------------------------------------------------------------------------------------

Gemma-4 TDR Root Cause Summary

Symptom: GPU hang (TDR / DEVICE_HUNG 0x887A0006) during first prompt processing with Gemma-4 on DX12 + AMD RDNA 3. Phi-3 worked
fine.

There were actually TWO separate TDR issues:

Issue 1: D3D12 Dispatch Limit Overflow (Checkpoints 18–19)

Gemma-4's vocabulary is 262,144 tokens (vs Phi-3's 32,000). The lm_head MUL_MAT projection dispatches groups_x = vocab_size =
262144, which exceeds D3D12's per-axis limit of 65535. The GPU silently hangs.

Fix: 2D dispatch linearization — when groups_x > 65535, split into (65535, ceil(N/65535), batches) and the shader reconstructs row
= gid.y * 65535 + gid.x.

Issue 2: Stale Root Descriptors (Checkpoints 20–22) — The Hard One

After fixing the dispatch limit, TDR persisted. This took extensive isolation:

 - Unconditional UAV barriers → didn't help
 - Aggressive flush thresholds → BSOD (too many fence ops)
 - Full binding reset every dispatch → finally fixed it

Root cause: AMD RDNA 3 hardware prefetches/validates ALL bound root descriptor slots when a PSO (Pipeline State Object) is
switched, even if the new shader doesn't use those slots. Flash attention had previously bound KV cache addresses to slots 4/5
(src2/src3). After those buffers were released or reused, later dispatches (GLU, MUL_MAT) that don't use slots 4/5 would still
trigger a GPU hang because the hardware tried to validate the stale addresses.

Fix: Always bind a valid fallback address (src0's VA) to slots 4 and 5 before every dispatch. Negligible perf cost due to VA
caching.

Why Phi-3 Never Hit This

 - Phi-3 vocab = 32K → never exceeds 65535 dispatch limit  
 - Phi-3 has 32 attention heads → after FA, the next PSO switch still had "live" addresses in slots 4/5 by coincidence (more graph 
nodes between FA and the problematic ops)
