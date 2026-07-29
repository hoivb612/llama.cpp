// mul_mat_q4_1_wmma.hlsl - Register-blocked tiled MUL_MAT with Q4_1 weights
// Mirrors mul_mat_q4_0_wmma.hlsl. Q4_1 block = 20 bytes:
//   d(f16) + m(f16) + qs[16] (each byte holds 2 nibbles)
//   val(l)        = (qs[l]      & 0x0F) * d + m  for l in 0..15
//   val(l + 16)   = (qs[l] >> 4)         * d + m  for l in 0..15
// BK=32 == QK4_1 so each K-tile lives inside a single Q4_1 block per row.
// Q4_1 block stride is 20 bytes (4-aligned), but read_byte_q4_1 still uses
// aligned Load + bit-shift for the qs nibble (qs offsets within a block are
// 1-byte aligned to byte_idx).
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3
#include "ggml_common.hlsli"

#define QK4_1       32
#define Q4_1_BSIZE  20

#define BM 32
#define BN 32
#define BK 32

uint read_byte_q4_1(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float read_f16_q4_1(ByteAddressBuffer buf, uint byte_off) {
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

        // Load tile_b: BK × BN from src0 — dequantize Q4_1 (1024 elements, 256 threads, 4 each).
        // BK == QK4_1 so all 32 K-elements for a given (k_tile, row) live in the same Q4_1 block.
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
                    uint block_idx = global_k / QK4_1;
                    uint elem_in_block = global_k % QK4_1;
                    uint row_off  = src0_offset + global_n * nb01
                                  + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q4_1_BSIZE;
                    // dm packed: low half = d, high half = m
                    uint dm_raw = src0.Load(block_off);
                    float d  = f16_to_f32(dm_raw & 0xFFFFu);
                    float mm = f16_to_f32(dm_raw >> 16);
                    uint byte_idx = elem_in_block % 16;
                    uint qs_byte  = read_byte_q4_1(src0, block_off + 4u + byte_idx);
                    uint nibble = (elem_in_block < 16) ? (qs_byte & 0x0Fu) : (qs_byte >> 4);
                    val = d * (float)nibble + mm;
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
