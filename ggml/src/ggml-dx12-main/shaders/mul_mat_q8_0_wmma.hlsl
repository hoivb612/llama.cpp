// mul_mat_q8_0_wmma.hlsl - Register-blocked tiled MUL_MAT with Q8_0 weights
// Mirrors mul_mat_q4k_wmma.hlsl. Q8_0 block = 34 bytes: f16 scale d + int8 qs[32].
// BK=32 == QK8_0 so each K-tile sits inside a single Q8_0 block per row.
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3
#include "ggml_common.hlsli"

#define QK8_0 32
#define Q8_0_BSIZE 34

#define BM 32
#define BN 32
#define BK 32

uint read_byte_q80(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float read_f16_q80(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

groupshared float tile_a[BM][BK];
groupshared float tile_b[BK][BN];

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;
    uint flat_id = ty * 16 + tx;

    uint col_block = gid.x;
    uint row_block = gid.y;
    uint batch     = gid.z;

    uint i2 = batch % ne2;
    uint i3 = batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_k_tiles = (K + BK - 1) / BK;

    precise float acc00 = 0.0f, acc01 = 0.0f;
    precise float acc10 = 0.0f, acc11 = 0.0f;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // Load tile_a: BM × BK from src1 (1024 elements, 256 threads, 4 each)
        {
            uint base = flat_id * 4;
            [unroll] for (uint e = 0; e < 4; e++) {
                uint idx = base + e;
                uint m_local = idx / BK;
                uint k_local = idx % BK;
                uint global_m = row_block * BM + m_local;
                uint global_k = k_start + k_local;
                float val = 0.0f;
                if (global_m < ne11 && global_k < K) {
                    uint off = offset_4d(global_k, global_m, i2, i3,
                                         nb10, nb11, nb12, nb13, src1_offset);
                    val = load_auto(src1, off, src1_esize);
                }
                tile_a[m_local][k_local] = val;
            }
        }

        // Load tile_b: BK × BN from src0 — dequantize Q8_0 (1024 elements, 256 threads, 4 each).
        // BK == QK8_0 so all 32 K-elements for a given (k_tile, row) live in the same Q8_0 block;
        // the per-row scale is read once per element here (the cache makes this near-free, and
        // we keep the pattern simple to mirror the Q4_K wmma shader).
        {
            uint base = flat_id * 4;
            [unroll] for (uint e = 0; e < 4; e++) {
                uint idx = base + e;
                uint k_local = idx / BN;
                uint n_local = idx % BN;
                uint global_k = k_start + k_local;
                uint global_n = col_block * BN + n_local;
                float val = 0.0f;
                if (global_k < K && global_n < ne01) {
                    uint block_idx = global_k / QK8_0;
                    uint elem_in_block = global_k % QK8_0;
                    uint row_off  = src0_offset + global_n * nb01
                                  + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q8_0_BSIZE;
                    float d   = read_f16_q80(src0, block_off);
                    uint  qb  = read_byte_q80(src0, block_off + 2u + elem_in_block);
                    int   qi  = (int)(qb << 24u) >> 24;  // sign-extend int8
                    val = d * (float)qi;
                }
                tile_b[k_local][n_local] = val;
            }
        }

        GroupMemoryBarrierWithGroupSync();

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
