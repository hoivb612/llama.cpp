// mul_mat_q5k_wmma.hlsl - Register-blocked tiled MUL_MAT with Q5_K weights
//
// Cooperative dequant: threads cooperatively load raw Q5_K block bytes (qs + qh)
// into shared memory, decode scales once per column, then dequant from LDS.
// Q5_K block layout: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128] = 176 bytes
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3
#include "ggml_common.hlsli"

#define QK_K 256
#define Q5K_BLOCK_SIZE 176

#define BM 32
#define BN 32
#define BK 32

// Shared memory for cooperative Q5_K dequant
groupshared uint  raw_qs[BN][8];   // 32 qs bytes as 8 uint32 per column
groupshared uint  raw_qh[BN][8];   // 32 qh bytes as 8 uint32 per column
groupshared float col_sc[BN][2];   // (d*scale, dmin*min) per column

groupshared float tile_a[BM][BK];
groupshared float tile_b[BK][BN];

uint read_byte_q5k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

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
    uint num_k_tiles = (K + BK - 1) / BK;

    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // Load tile_a from src1
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

        // Load tile_b: cooperative Q5_K dequant
        {
            uint block_idx    = k_start / QK_K;
            uint elem_in_blk  = k_start % QK_K;
            uint sub_chunk    = elem_in_blk / 64;
            bool is_high_half = (elem_in_blk % 64) >= 32;

            // Phase 1a: First 32 threads load headers and scales
            if (flat_id < BN) {
                uint n = flat_id;
                uint global_n = col_block * BN + n;
                if (global_n < ne01) {
                    uint row_off = src0_offset + global_n * nb01
                                 + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q5K_BLOCK_SIZE;

                    uint dm_raw = src0.Load(block_off);
                    float dall = f16_to_f32(dm_raw & 0xFFFFu);
                    float dmin = f16_to_f32(dm_raw >> 16);

                    uint scales_off = block_off + 4;
                    uint is = sub_chunk * 2 + (is_high_half ? 1u : 0u);
                    uint sc_raw, mb_raw;
                    if (is < 4) {
                        sc_raw = read_byte_q5k(src0, scales_off + is) & 0x3Fu;
                        mb_raw = read_byte_q5k(src0, scales_off + is + 4) & 0x3Fu;
                    } else {
                        uint lo_sc = read_byte_q5k(src0, scales_off + is + 4) & 0x0Fu;
                        uint hi_sc = (read_byte_q5k(src0, scales_off + is - 4) >> 6) & 0x03u;
                        sc_raw = lo_sc | (hi_sc << 4);
                        uint lo_mb = (read_byte_q5k(src0, scales_off + is + 4) >> 4) & 0x0Fu;
                        uint hi_mb = (read_byte_q5k(src0, scales_off + is) >> 6) & 0x03u;
                        mb_raw = lo_mb | (hi_mb << 4);
                    }
                    col_sc[n][0] = dall * (float)sc_raw;
                    col_sc[n][1] = dmin * (float)mb_raw;
                } else {
                    col_sc[n][0] = 0.0f; col_sc[n][1] = 0.0f;
                }
            }

            // Phase 1b: 256 threads load qs bytes (32 cols × 8 uint32 = 256)
            {
                uint col = flat_id / 8;
                uint qw  = flat_id % 8;
                uint global_n = col_block * BN + col;
                if (global_n < ne01 && k_start < K) {
                    uint row_off = src0_offset + global_n * nb01
                                 + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q5K_BLOCK_SIZE;
                    uint qs_base = block_off + 48 + sub_chunk * 32;
                    raw_qs[col][qw] = src0.Load(qs_base + qw * 4);
                } else {
                    raw_qs[col][qw] = 0;
                }
            }

            // Phase 1c: Load qh bytes -- need all 32 qh bytes per column
            // (qh bits are indexed by elem_in_half, not sub_chunk)
            // 256 threads load 32 cols × 8 uint32 = 256
            {
                uint col = flat_id / 8;
                uint qw  = flat_id % 8;
                uint global_n = col_block * BN + col;
                if (global_n < ne01 && k_start < K) {
                    uint row_off = src0_offset + global_n * nb01
                                 + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q5K_BLOCK_SIZE;
                    uint qh_base = block_off + 16;
                    raw_qh[col][qw] = src0.Load(qh_base + qw * 4);
                } else {
                    raw_qh[col][qw] = 0;
                }
            }

            GroupMemoryBarrierWithGroupSync();

            // Phase 2: Dequant from shared memory
            {
                uint hm = is_high_half ? (1u << (2u * sub_chunk + 1u)) : (1u << (2u * sub_chunk));

                uint base = flat_id * 4;
                [unroll] for (uint e = 0; e < 4; e++) {
                    uint idx = base + e;
                    uint k_local = idx / BN;
                    uint n_local = idx % BN;

                    float val = 0.0f;
                    if ((k_start + k_local) < K && (col_block * BN + n_local) < ne01) {
                        uint word_idx = k_local / 4;
                        uint byte_in_word = k_local % 4;
                        uint qs_byte = (raw_qs[n_local][word_idx] >> (byte_in_word * 8)) & 0xFFu;
                        uint qh_byte = (raw_qh[n_local][word_idx] >> (byte_in_word * 8)) & 0xFFu;

                        float d = col_sc[n_local][0];
                        float m = col_sc[n_local][1];

                        uint q_base;
                        if (!is_high_half) {
                            q_base = qs_byte & 0x0Fu;
                        } else {
                            q_base = qs_byte >> 4;
                        }
                        uint q5_bit = ((qh_byte & hm) != 0u) ? 16u : 0u;
                        val = d * (float)(q_base + q5_bit) - m;
                    }
                    tile_b[k_local][n_local] = val;
                }
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
