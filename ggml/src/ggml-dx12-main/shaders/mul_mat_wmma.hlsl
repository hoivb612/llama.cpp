// mul_mat_wmma.hlsl - Register-blocked tiled matrix multiplication
// Optimized batch MUL_MAT (M > 1) for prompt processing.
// Uses 2×2 register blocking: each thread computes 4 output elements,
// giving 2× better compute/load ratio vs the scalar 16×16 tiled approach.
//
// When WaveMatrix HLSL types ship in DXC, this shader can be upgraded
// to use hardware MMA units for an additional 2-4× speedup.
//
// ggml MUL_MAT: dst[i1, i0] = sum_k(src0[i0, k] * src1[i1, k])
//
// src0: weights, ne00 = K, ne01 = N (output features) — F32 or F16
// src1: input,   ne10 = K, ne11 = M (batch)            — F32 or F16
// dst:  output,  ne0  = N, ne1  = M                    — F32 or F16
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3
#include "ggml_common.hlsli"

#define BM 32
#define BN 32
#define BK 16

groupshared float tile_a[BM][BK]; // src1: batch × K
groupshared float tile_b[BK][BN]; // src0: K × output features

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x; // 0..15
    uint ty = gtid.y; // 0..15
    uint flat_id = ty * 16 + tx;

    uint col_block = gid.x; // output feature block
    uint row_block = gid.y; // batch block
    uint batch = gid.z;

    uint i2 = batch % ne2;
    uint i3 = batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_k_tiles = (K + BK - 1) / BK;

    // 2×2 register blocking: each thread accumulates 4 output elements
    precise float acc00 = 0.0f, acc01 = 0.0f;
    precise float acc10 = 0.0f, acc11 = 0.0f;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // Load tile_a: BM × BK from src1 (512 elements, 256 threads, 2 each)
        {
            uint base = flat_id * 2;
            [unroll] for (uint e = 0; e < 2; e++) {
                uint idx = base + e;
                uint m = idx / BK;
                uint k = idx % BK;
                uint global_m = row_block * BM + m;
                uint global_k = k_start + k;
                float val = 0.0f;
                if (global_m < ne11 && global_k < K) {
                    uint off = offset_4d(global_k, global_m, i2, i3,
                                         nb10, nb11, nb12, nb13, src1_offset);
                    val = load_auto(src1, off, src1_esize);
                }
                tile_a[m][k] = val;
            }
        }

        // Load tile_b: BK × BN from src0 (512 elements, 256 threads, 2 each)
        {
            uint base = flat_id * 2;
            [unroll] for (uint e = 0; e < 2; e++) {
                uint idx = base + e;
                uint k = idx / BN;
                uint n = idx % BN;
                uint global_k = k_start + k;
                uint global_n = col_block * BN + n;
                float val = 0.0f;
                if (global_k < K && global_n < ne01) {
                    uint off = offset_4d(global_k, global_n, i2_src0, i3_src0,
                                         nb00, nb01, nb02, nb03, src0_offset);
                    val = load_auto(src0, off, src0_esize);
                }
                tile_b[k][n] = val;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Accumulate: 2×2 register blocking
        // Thread (tx, ty) computes outputs at (ty*2+0..1, tx*2+0..1)
        [unroll]
        for (uint k = 0; k < BK; k++) {
            float a0 = tile_a[ty * 2    ][k];
            float a1 = tile_a[ty * 2 + 1][k];
            float b0 = tile_b[k][tx * 2    ];
            float b1 = tile_b[k][tx * 2 + 1];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Store 2×2 results with bounds checking
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
