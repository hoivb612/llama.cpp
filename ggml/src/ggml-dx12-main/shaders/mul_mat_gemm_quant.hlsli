// Tiled dense MUL_MAT for quantized weights, dequantising into LDS.
//
// The default batch path for the IQ/TQ types (mul_mat_quant.hlsli) runs one
// thread per output element, so each thread walks the whole of K decoding a
// codebook as it goes and nothing is reused: the weight row is re-decoded
// once per token and the activation row once per output column.  That costs
// ne1 * ne0 * K dequants and is slow enough to need row-chunking to stay
// under the Windows TDR.
//
// Here a threadgroup owns a 64(token) x 64(output) tile, so a weight element
// is decoded once per 64 tokens and an activation element read once per 64
// outputs.  Same tile shape as mul_mat_id_gemm.hlsli.
//
// Wrapper shaders define exactly one MMID_<TYPE> macro before including this.

#include "quant_dequant.hlsli"

#define BM      64
#define BN      64
#define BK      16
#define TM      (BM / 16)
#define TN      4
#define THREADS 256

// Elements each thread moves into LDS per K-tile.
#define A_PER_THREAD ((BM * BK) / THREADS)
#define B_PER_THREAD ((BK * BN) / THREADS)

groupshared float16_t tile_a[BM][BK]; // activations: tokens x K
groupshared float16_t tile_b[BK][BN]; // weights:     K x outputs

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint tx      = gtid.x; // 0..15 -> output column within tile
    const uint ty      = gtid.y; // 0..15 -> token row within tile
    const uint flat_id = ty * 16 + tx;

    const uint col_block = gid.x;
    const uint row_block = gid.y;
    const uint batch     = gid.z;

    const uint i2 = batch % ne2;
    const uint i3 = batch / ne2;

    const uint i2_src0 = i2 * ne02 / ne2;
    const uint i3_src0 = i3 * ne03 / ne3;

    const uint row_base = row_block * BM;
    if (row_base >= ne1) {
        return;
    }

    const uint K           = ne00;
    const uint num_k_tiles = (K + BK - 1) / BK;

    const uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    const uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    // Weight row this thread's slice of the B tile belongs to.  Both tile
    // loads stride by THREADS, which keeps them coalesced and gives each
    // thread BK/4 k-values inside a single weight row: for a quantized tile
    // they all land in one block so the block header is decoded once.
    const uint b_n       = flat_id % BN;
    const uint b_col     = col_block * BN + b_n;
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
            uint row = row_base + m;
            float16_t val = (float16_t)0;
            if (row < ne1 && global_k < K) {
                val = (float16_t)load_auto(src1, src1_base + row * nb11 + global_k * nb10, src1_esize);
            }
            tile_a[m][k] = val;
        }

        // tile_b: BK x BN weights.
        [unroll] for (uint e2 = 0; e2 < B_PER_THREAD; e2++) {
            uint k = e2 * (THREADS / BN) + flat_id / BN;
            uint global_k = k_start + k;
            float16_t val = (float16_t)0;
            if (global_k < K && b_col < ne0) {
                val = (float16_t)mmid_dequant(src0, b_row_off, global_k);
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
        uint row = row_base + ty * TM + im;
        if (row >= ne1) continue;
        [unroll] for (uint in_ = 0; in_ < TN; in_++) {
            uint global_n = col_block * BN + tx * TN + in_;
            if (global_n >= ne0) continue;
            uint off = offset_4d(global_n, row, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off, acc[im][in_], dst_esize);
        }
    }
}
