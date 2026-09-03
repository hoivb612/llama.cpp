// Shared tiled GEMM MUL_MAT_ID implementation.
//
// The MMID matvec kernels dispatch one threadgroup per (token, expert) pair,
// which re-reads the whole expert matrix for every token routed to it. On
// granite-3.0-1b-a400m F16 that is ~4 GiB of weight traffic per 512-token
// ubatch node for 4.3 GFLOP of work - ~198 GFLOP/s against the ~1500 the
// dense F16 GEMM reaches on the same device.
//
// moe_expert_bucket.hlsl groups the pairs by expert first, so here one
// threadgroup owns a 64(pairs) x 64(outputs) tile of a single expert and
// reads that expert's weights once per tile.
//
// Tile shape mirrors mul_mat_wmma_fp16.hlsl: 64x64 output tile, 4x4 register
// blocking, half LDS tiles, fp16 multiply promoted into an fp32 accumulator
// once per K-tile.
//
// Wrapper shaders either define nothing (dense F32/F16/BF16 weights) or
// exactly one MMID_<TYPE> macro before including this file, which selects the
// matching mmid_dequant() and turns the weight tile loader into a
// dequant-to-LDS pass.
//
// Dispatch:
//   gid.x = output-feature tile (ne0 / BN)
//   gid.y = pair tile within the expert
//   gid.z = expert index
//
// Bindings:
//   src0 = expert weights [ne00 x ne01 x n_expert]
//   src1 = F32 activations
//   temp = bucket scratch (offsets at 0, pair indices at op6)
//   dst  = [ne0, n_expert_used, n_tokens]
//
// Requires native 16-bit shader ops; the host keeps devices without them on
// the matvec route.
#ifdef MMID_QUANT
#include "quant_dequant.hlsli"
#else
#include "ggml_common.hlsli"
#endif

// Tile height. Wrappers that define MMID_BM=128 build the "tall" variant,
// which halves how often an expert's weight tile is re-read but only pays off
// when a model routes at least BM pairs to a single expert. The host picks
// between the two blob sets from the pair-per-expert estimate; see
// dx12_mmid_gemm_bm() in ggml-dx12.cpp.
#ifndef MMID_BM
#define MMID_BM 64
#endif

#define BM      MMID_BM
#define BN      64
#define BK      16
#define TM      (BM / 16)
#define TN      4
#define THREADS 256

// Elements each thread moves into LDS per K-tile.
#define A_PER_THREAD ((BM * BK) / THREADS)
#define B_PER_THREAD ((BK * BN) / THREADS)

#define PAIR_INVALID 0xFFFFFFFFu

groupshared float16_t tile_a[BM][BK]; // activations: pairs x K
groupshared float16_t tile_b[BK][BN]; // weights:     K x outputs

// Resolved once per tile so the K-loop costs no integer divides.
groupshared uint g_rowoff[BM];  // activation row byte offset
groupshared uint g_slot[BM];    // dst expert slot  (PAIR_INVALID = padding row)
groupshared uint g_token[BM];   // dst token

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint tx      = gtid.x; // 0..15 -> output column within tile
    const uint ty      = gtid.y; // 0..15 -> pair row within tile
    const uint flat_id = ty * 16 + tx;

    const uint col_block = gid.x;
    const uint pair_tile = gid.y;
    const uint expert    = gid.z;

    const uint start = temp.Load(expert * 4u);
    const uint end   = temp.Load((expert + 1u) * 4u);
    const uint cnt   = end - start;

    const uint row_base = pair_tile * BM;
    if (row_base >= cnt) {
        return; // this expert has no pairs left for the tile
    }

    const uint nei0 = ne1; // n_expert_used

    if (flat_id < BM) {
        uint r = row_base + flat_id;
        if (r < cnt) {
            uint pair  = temp.Load(op6 + (start + r) * 4u);
            uint slot  = pair % nei0;
            uint token = pair / nei0;
            g_slot[flat_id]   = slot;
            g_token[flat_id]  = token;
            g_rowoff[flat_id] = src1_offset + (slot % ne11) * nb11 + token * nb12;
        } else {
            g_slot[flat_id]   = PAIR_INVALID;
            g_token[flat_id]  = 0;
            g_rowoff[flat_id] = 0;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint K           = ne00;
    const uint num_k_tiles = (K + BK - 1) / BK;
    const uint src0_base   = src0_offset + expert * nb02;

    // Weight row this thread's slice of the B tile belongs to.  The tile loads
    // stride by THREADS so a thread walks BK/4 k-values within one row: for
    // quantized weights that keeps every load inside one block and lets the
    // block scale be folded out of the unrolled loop.
    const uint b_n   = flat_id % BN;
    const uint b_col = col_block * BN + b_n;
    const uint b_row_off = src0_base + b_col * nb01;

    precise float acc[TM][TN];
    [unroll] for (uint im = 0; im < TM; im++) {
        [unroll] for (uint in_ = 0; in_ < TN; in_++) {
            acc[im][in_] = 0.0f;
        }
    }

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // tile_a: BM x BK activations.
        [unroll] for (uint e = 0; e < A_PER_THREAD; e++) {
            uint idx = flat_id + e * THREADS;
            uint m   = idx / BK;
            uint k   = idx % BK;
            uint global_k = k_start + k;
            float16_t val = (float16_t)0;
            if (g_slot[m] != PAIR_INVALID && global_k < K) {
                val = (float16_t)load_auto(src1, g_rowoff[m] + global_k * nb10, src1_esize);
            }
            tile_a[m][k] = val;
        }

        // tile_b: BK x BN expert weights.
        [unroll] for (uint e2 = 0; e2 < B_PER_THREAD; e2++) {
            uint k = e2 * (THREADS / BN) + flat_id / BN;
            uint global_k = k_start + k;
            float16_t val = (float16_t)0;
            if (global_k < K && b_col < ne0) {
#ifdef MMID_QUANT
                val = (float16_t)mmid_dequant(src0, b_row_off, global_k);
#else
                val = (float16_t)load_auto(src0, b_row_off + global_k * nb00, src0_esize);
#endif
            }
            tile_b[k][b_n] = val;
        }

        GroupMemoryBarrierWithGroupSync();

        float16_t tacc[TM][TN];
        [unroll] for (uint im = 0; im < TM; im++) {
            [unroll] for (uint in_ = 0; in_ < TN; in_++) {
                tacc[im][in_] = (float16_t)0;
            }
        }
        [unroll]
        for (uint k = 0; k < BK; k++) {
            float16_t a[TM];
            float16_t b[TN];
            [unroll] for (uint im = 0; im < TM; im++) {
                a[im] = tile_a[ty * TM + im][k];
            }
            [unroll] for (uint in_ = 0; in_ < TN; in_++) {
                b[in_] = tile_b[k][tx * TN + in_];
            }
            [unroll] for (uint im = 0; im < TM; im++) {
                [unroll] for (uint in_ = 0; in_ < TN; in_++) {
                    tacc[im][in_] += a[im] * b[in_];
                }
            }
        }
        [unroll] for (uint im = 0; im < TM; im++) {
            [unroll] for (uint in_ = 0; in_ < TN; in_++) {
                acc[im][in_] += (float)tacc[im][in_];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint im = 0; im < TM; im++) {
        uint m = ty * TM + im;
        uint slot = g_slot[m];
        if (slot == PAIR_INVALID) continue;
        uint token = g_token[m];
        [unroll] for (uint in_ = 0; in_ < TN; in_++) {
            uint global_n = col_block * BN + tx * TN + in_;
            if (global_n >= ne0) continue;
            uint off = offset_4d(global_n, slot, token, 0, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off, acc[im][in_], dst_esize);
        }
    }
}
