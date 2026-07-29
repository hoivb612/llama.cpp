// mul_mat_wmma_kfull.hlsl - Batch MUL_MAT specialized for tiny K (<=64).
// Targets CLIP attention shapes: Q.K^T (K=64 N=1024 M=1024) and similar
// small-head-dim batched matmuls where the K loop is so short that the
// K-tile sync overhead in mul_mat_wmma.hlsl dominates the actual compute.
//
// vs mul_mat_wmma.hlsl:
//   - BK = 64 (not 16): the entire K is one tile. Zero K-loop syncs.
//   - Same 32x32 output tile, same 2x2 register block, same dispatch grid.
//   - LDS footprint: 2*32*64*4 = 16 KiB (vs 4 KiB baseline). Still well
//     within the 32 KiB D3D12 minimum.
//
// Constraint: ne00 (K) must be <= 64. Host gates this.
//
// ggml MUL_MAT: dst[i1, i0] = sum_k(src0[i0, k] * src1[i1, k])
// src0: weights, ne00 = K, ne01 = N (output features)
// src1: input,   ne10 = K, ne11 = M (batch)
// dst:  output,  ne0  = N, ne1  = M
#include "ggml_common.hlsli"

#define BM 32
#define BN 32
#define BK 64

groupshared float tile_a[BM][BK]; // src1: batch x K
groupshared float tile_b[BK][BN]; // src0: K x output features

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;
    uint flat_id = ty * 16 + tx;

    uint col_block = gid.x;
    uint row_block = gid.y;
    uint batch = gid.z;

    uint i2 = batch % ne2;
    uint i3 = batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;

    precise float acc00 = 0.0f, acc01 = 0.0f;
    precise float acc10 = 0.0f, acc11 = 0.0f;

    // Load tile_a: BM x BK from src1. 2048 elements / 256 threads = 8 each.
    {
        uint base = flat_id * 8;
        [unroll] for (uint e = 0; e < 8; e++) {
            uint idx = base + e;
            uint m = idx / BK;
            uint k = idx % BK;
            uint global_m = row_block * BM + m;
            float val = 0.0f;
            if (global_m < ne11 && k < K) {
                uint off = offset_4d(k, global_m, i2, i3,
                                     nb10, nb11, nb12, nb13, src1_offset);
                val = load_auto(src1, off, src1_esize);
            }
            tile_a[m][k] = val;
        }
    }

    // Load tile_b: BK x BN from src0. 2048 elements / 256 threads = 8 each.
    {
        uint base = flat_id * 8;
        [unroll] for (uint e = 0; e < 8; e++) {
            uint idx = base + e;
            uint k = idx / BN;
            uint n = idx % BN;
            uint global_n = col_block * BN + n;
            float val = 0.0f;
            if (k < K && global_n < ne01) {
                uint off = offset_4d(k, global_n, i2_src0, i3_src0,
                                     nb00, nb01, nb02, nb03, src0_offset);
                val = load_auto(src0, off, src0_esize);
            }
            tile_b[k][n] = val;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Single K loop, no inner sync. 2x2 register accumulation.
    [unroll]
    for (uint k = 0; k < BK; k++) {
        if (k >= K) break;
        float a0 = tile_a[ty * 2    ][k];
        float a1 = tile_a[ty * 2 + 1][k];
        float b0 = tile_b[k][tx * 2    ];
        float b1 = tile_b[k][tx * 2 + 1];
        acc00 += a0 * b0;
        acc01 += a0 * b1;
        acc10 += a1 * b0;
        acc11 += a1 * b1;
    }

    uint m0 = row_block * BM + ty * 2;
    uint m1 = m0 + 1;
    uint n0 = col_block * BN + tx * 2;
    uint n1 = n0 + 1;

    if (m0 < ne1 && n0 < ne0) {
        uint off = offset_4d(n0, m0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc00, dst_esize);
    }
    if (m0 < ne1 && n1 < ne0) {
        uint off = offset_4d(n1, m0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc01, dst_esize);
    }
    if (m1 < ne1 && n0 < ne0) {
        uint off = offset_4d(n0, m1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc10, dst_esize);
    }
    if (m1 < ne1 && n1 < ne0) {
        uint off = offset_4d(n1, m1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc11, dst_esize);
    }
}
