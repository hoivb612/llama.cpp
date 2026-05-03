// mul_mat_q4k_wmma.hlsl - Register-blocked tiled MUL_MAT with Q4_K weights
// Dequantizes Q4_K into groupshared memory, then uses 2×2 register blocking.
// BK=32 aligns with Q4_K sub-block boundaries (8 sub-blocks of 32 per block).
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3
#include "ggml_common.hlsli"

#define QK_K 256
#define Q4K_BLOCK_SIZE 144

#define BM 32
#define BN 32
#define BK 32

uint read_byte_q4k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

// Dequantize one element from a Q4_K block (same logic as mul_mat_q4k.hlsl)
float dequant_q4k(ByteAddressBuffer buf, uint block_off, uint elem_in_block) {
    uint dm_raw = buf.Load(block_off);
    float dall = f16_to_f32(dm_raw & 0xFFFFu);
    float dmin_val = f16_to_f32(dm_raw >> 16);
    uint scales_off = block_off + 4;

    uint il = elem_in_block / 64;
    uint elem_in_chunk = elem_in_block % 64;
    bool is_high = (elem_in_chunk >= 32);
    uint elem_in_half = elem_in_chunk % 32;
    uint is = 2 * il;

    uint sc_idx, mb_val;
    uint is_eff = is_high ? (is + 1) : is;
    {
        uint scidx0 = (is < 4) ? is_eff : (is_eff + 4);
        uint scidx1 = (is < 4) ? is_eff : (is_eff - 4);
        uint scmask1 = (is < 4) ? 0x30u : 0xC0u;
        uint scshift1 = (is < 4) ? 0u : 2u;
        uint mbidx0 = is_eff + 4;
        uint mbidx1 = (is < 4) ? is_eff + 4 : is_eff;
        uint mbmask0 = (is < 4) ? 0x0Fu : 0xF0u;
        uint mbshift0 = (is < 4) ? 0u : 4u;
        uint mbmask1 = (is < 4) ? 0x30u : 0xC0u;
        uint mbshift1 = (is < 4) ? 0u : 2u;

        sc_idx = (read_byte_q4k(buf, scales_off + scidx0) & 0x0Fu) |
                 ((read_byte_q4k(buf, scales_off + scidx1) & scmask1) >> scshift1);
        mb_val = ((read_byte_q4k(buf, scales_off + mbidx0) & mbmask0) >> mbshift0) |
                 ((read_byte_q4k(buf, scales_off + mbidx1) & mbmask1) >> mbshift1);
    }

    float d = dall * (float)sc_idx;
    float m = dmin_val * (float)mb_val;

    uint qs_off = block_off + 16;
    uint qs_byte = read_byte_q4k(buf, qs_off + il * 32 + elem_in_half);

    float q;
    if (!is_high) {
        q = (float)(qs_byte & 0x0Fu);
    } else {
        q = (float)(qs_byte >> 4);
    }

    return d * q - m;
}

groupshared float tile_a[BM][BK]; // src1: batch × K
groupshared float tile_b[BK][BN]; // src0: K × output features (dequantized)

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

        // Load tile_b: BK × BN from src0 — dequantize Q4_K (1024 elements, 256 threads, 4 each)
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
                    uint block_idx = global_k / QK_K;
                    uint elem_in_block = global_k % QK_K;
                    uint row_off = src0_offset + global_n * nb01
                                 + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q4K_BLOCK_SIZE;
                    val = dequant_q4k(src0, block_off, elem_in_block);
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
