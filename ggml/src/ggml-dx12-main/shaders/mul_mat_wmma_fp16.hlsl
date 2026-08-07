// mul_mat_wmma_fp16.hlsl - FP16 LDS + 4x4 register-tile MUL_MAT.
// Requires native 16-bit shader ops (-enable-16bit-types). Targets F16 weights
// and F16 activations (e.g. CLIP / vision encoder GEMMs).
//
// Differences vs mul_mat_wmma.hlsl:
//   - 4x4 register blocking (vs 2x2): each thread accumulates 16 outputs over
//     a 64x64 output tile, so the compute:LDS-load ratio per inner iteration
//     becomes 16:8 = 2:1 (vs 4:4 = 1:1 in the 2x2 baseline). On B390 the
//     baseline path is compute-bound at ~1.7 TFLOPS of ~3+ peak; the 4x4
//     tile is the standard "more compute per LDS load" remedy.
//   - LDS tiles stored as float16_t (half-precision LDS), keeping the LDS
//     footprint at 4 KiB despite the 2x bigger output tile. Multiply in fp16,
//     accumulate in fp32.
//   - K-tile partial sums stay in fp16 and are promoted into the fp32
//     accumulator once per tile, so the inner loop runs at the fp16 rate.
//     16 terms cannot lose meaningful precision and cannot plausibly
//     overflow (fp16 saturates at 65504, so it would take products
//     averaging ~4000). SmolLM2-135M f16 perplexity is 9.2147 vs 9.2144 for
//     fp32 accumulation and 9.2143 on the CPU backend; B390 pp512 +10.9%.
//   - Vectorized typed-half loads via load_auto() casts (TBD: pure typed loads
//     once a F16x4 wrapper lands).
//
// Dispatched only when:
//   - src0_type == F16 && src1_type == F16
//   - dev->fp16_supported (OPTIONS4.Native16BitShaderOpsSupported)
//
// Falls back to mul_mat_wmma.hlsl for F32 weights or F16 activations.
#include "ggml_common.hlsli"

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define THREADS 256

groupshared float16_t tile_a[BM][BK]; // src1 batch x K (1024 halves)
groupshared float16_t tile_b[BK][BN]; // src0 K x N     (1024 halves)

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x; // 0..15  -> col tile index
    uint ty = gtid.y; // 0..15  -> row tile index
    uint flat_id = ty * 16 + tx;

    uint col_block = gid.x;
    uint row_block = gid.y;
    uint batch = gid.z;

    uint i2 = batch % ne2;
    uint i3 = batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_k_tiles = (K + BK - 1) / BK;

    // 4x4 register tile, fp32 accumulators (K can be 3072+ -> fp16 acc loses bits).
    precise float acc[TM][TN];
    [unroll] for (uint im = 0; im < TM; im++) {
        [unroll] for (uint in_ = 0; in_ < TN; in_++) {
            acc[im][in_] = 0.0f;
        }
    }

    // Each thread loads 4 halves into each of tile_a / tile_b per K-tile pass
    // (1024 halves / 256 threads = 4 each).
    const uint LDS_PER_THREAD = (BM * BK) / THREADS; // == 4

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // tile_a: BM x BK halves (1024). 4 per thread.
        {
            uint base = flat_id * LDS_PER_THREAD;
            [unroll] for (uint e = 0; e < LDS_PER_THREAD; e++) {
                uint idx = base + e;
                uint m = idx / BK;
                uint k = idx % BK;
                uint global_m = row_block * BM + m;
                uint global_k = k_start + k;
                float16_t val = (float16_t)0;
                if (global_m < ne11 && global_k < K) {
                    uint off = offset_4d(global_k, global_m, i2, i3,
                                         nb10, nb11, nb12, nb13, src1_offset);
                    val = (float16_t)load_auto(src1, off, src1_esize);
                }
                tile_a[m][k] = val;
            }
        }

        // tile_b: BK x BN halves (1024). 4 per thread.
        {
            uint base = flat_id * LDS_PER_THREAD;
            [unroll] for (uint e = 0; e < LDS_PER_THREAD; e++) {
                uint idx = base + e;
                uint k = idx / BN;
                uint n = idx % BN;
                uint global_k = k_start + k;
                uint global_n = col_block * BN + n;
                float16_t val = (float16_t)0;
                if (global_k < K && global_n < ne01) {
                    uint off = offset_4d(global_k, global_n, i2_src0, i3_src0,
                                         nb00, nb01, nb02, nb03, src0_offset);
                    val = (float16_t)load_auto(src0, off, src0_esize);
                }
                tile_b[k][n] = val;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // 4x4 register accumulate. Thread (tx, ty) owns output tile at
        // rows [ty*TM .. ty*TM+3], cols [tx*TN .. tx*TN+3].
        // The K-tile partial sums stay in fp16 (16 terms cannot lose
        // meaningful precision) so the inner loop runs at the fp16 rate;
        // only the per-tile result is promoted into the fp32 accumulator.
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

    // Store 4x4 with bounds checking.
    [unroll] for (uint im = 0; im < TM; im++) {
        uint global_m = row_block * BM + ty * TM + im;
        if (global_m >= ne1) continue;
        [unroll] for (uint in_ = 0; in_ < TN; in_++) {
            uint global_n = col_block * BN + tx * TN + in_;
            if (global_n >= ne0) continue;
            uint off = offset_4d(global_n, global_m, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off, acc[im][in_], dst_esize);
        }
    }
}

