// Tiled Q5_K x Q8_1 batch MUL_MAT using packed int8 dot products.
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK_K          256
#define Q5K_BSIZE     176
#define Q8_1_BSIZE     36
#define BM             32
#define BN             32

groupshared uint  tile_w_qs[BN][8];
groupshared float tile_w_d[BN];
groupshared float tile_w_m[BN];
groupshared uint  tile_a_qs[BM][8];
groupshared float tile_a_d[BM];
groupshared float tile_a_s[BM];

uint read_byte_q5k_intdot(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;
    uint flat_id = ty * 16u + tx;

    uint i2 = gid.z % ne2;
    uint i3 = gid.z / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_q8_blocks = K / 32u;

    precise float acc00 = 0.0f;
    precise float acc01 = 0.0f;
    precise float acc10 = 0.0f;
    precise float acc11 = 0.0f;

    for (uint block = 0; block < num_q8_blocks; ++block) {
        uint local_row = flat_id / 8u;
        uint packed_idx = flat_id & 7u;

        uint q5_block = block / 8u;
        uint tile_in_block = block & 7u;
        uint il = tile_in_block / 2u;
        bool high_nibble = (tile_in_block & 1u) != 0u;

        uint global_n = gid.x * BN + local_row;
        uint w_qs = 0u;
        if (global_n < ne01) {
            uint row_off = src0_offset + global_n * nb01
                         + i2_src0 * nb02 + i3_src0 * nb03;
            uint block_off = row_off + q5_block * Q5K_BSIZE;
            uint raw_qs = src0.Load(block_off + 48u + il * 32u + packed_idx * 4u);
            uint qh = src0.Load(block_off + 16u + packed_idx * 4u);
            uint low_bits = high_nibble ? ((raw_qs >> 4u) & 0x0F0F0F0Fu)
                                        : (raw_qs & 0x0F0F0F0Fu);
            uint high_bits = ((qh >> tile_in_block) & 0x01010101u) << 4u;
            w_qs = low_bits | high_bits;

            if (packed_idx == 0u) {
                uint dm_raw = src0.Load(block_off);
                float dall = f16_to_f32(dm_raw & 0xFFFFu);
                float dmin = f16_to_f32(dm_raw >> 16);
                uint scales_off = block_off + 4u;
                uint is_eff = tile_in_block;
                bool low_half = il < 2u;

                uint scidx0 = low_half ? is_eff : (is_eff + 4u);
                uint scidx1 = low_half ? is_eff : (is_eff - 4u);
                uint scmask1 = low_half ? 0x30u : 0xC0u;
                uint scshift1 = low_half ? 0u : 2u;
                uint mbidx0 = is_eff + 4u;
                uint mbidx1 = low_half ? (is_eff + 4u) : is_eff;
                uint mbmask0 = low_half ? 0x0Fu : 0xF0u;
                uint mbshift0 = low_half ? 0u : 4u;
                uint mbmask1 = low_half ? 0x30u : 0xC0u;
                uint mbshift1 = low_half ? 0u : 2u;

                uint sc = (read_byte_q5k_intdot(src0, scales_off + scidx0) & 0x0Fu)
                        | ((read_byte_q5k_intdot(src0, scales_off + scidx1) & scmask1) >> scshift1);
                uint m = ((read_byte_q5k_intdot(src0, scales_off + mbidx0) & mbmask0) >> mbshift0)
                       | ((read_byte_q5k_intdot(src0, scales_off + mbidx1) & mbmask1) >> mbshift1);

                tile_w_d[local_row] = dall * float(sc);
                tile_w_m[local_row] = dmin * float(m);
            }
        } else if (packed_idx == 0u) {
            tile_w_d[local_row] = 0.0f;
            tile_w_m[local_row] = 0.0f;
        }
        tile_w_qs[local_row][packed_idx] = w_qs;

        uint global_m = gid.y * BM + local_row;
        uint a_qs = 0u;
        if (global_m < ne11) {
            uint flat_row = (i3 * ne12 + i2) * ne11 + global_m;
            uint block_off = src1_offset
                           + (flat_row * num_q8_blocks + block) * Q8_1_BSIZE;
            uint ds = src1.Load(block_off);
            a_qs = src1.Load(block_off + 4u + packed_idx * 4u);
            if (packed_idx == 0u) {
                tile_a_d[local_row] = f16_to_f32(ds & 0xFFFFu);
                tile_a_s[local_row] = f16_to_f32(ds >> 16);
            }
        } else if (packed_idx == 0u) {
            tile_a_d[local_row] = 0.0f;
            tile_a_s[local_row] = 0.0f;
        }
        tile_a_qs[local_row][packed_idx] = a_qs;

        GroupMemoryBarrierWithGroupSync();

        uint m0 = ty * 2u;
        uint m1 = m0 + 1u;
        uint n0 = tx * 2u;
        uint n1 = n0 + 1u;

        int dot00 = 0;
        int dot01 = 0;
        int dot10 = 0;
        int dot11 = 0;
        [unroll]
        for (uint q = 0; q < 8u; ++q) {
            uint a0 = tile_a_qs[m0][q];
            uint a1 = tile_a_qs[m1][q];
            dot00 = dot4add_i8packed(tile_w_qs[n0][q], a0, dot00);
            dot01 = dot4add_i8packed(tile_w_qs[n1][q], a0, dot01);
            dot10 = dot4add_i8packed(tile_w_qs[n0][q], a1, dot10);
            dot11 = dot4add_i8packed(tile_w_qs[n1][q], a1, dot11);
        }

        acc00 += tile_w_d[n0] * tile_a_d[m0] * float(dot00) - tile_w_m[n0] * tile_a_s[m0];
        acc01 += tile_w_d[n1] * tile_a_d[m0] * float(dot01) - tile_w_m[n1] * tile_a_s[m0];
        acc10 += tile_w_d[n0] * tile_a_d[m1] * float(dot10) - tile_w_m[n0] * tile_a_s[m1];
        acc11 += tile_w_d[n1] * tile_a_d[m1] * float(dot11) - tile_w_m[n1] * tile_a_s[m1];

        GroupMemoryBarrierWithGroupSync();
    }

    uint m0 = gid.y * BM + ty * 2u;
    uint m1 = m0 + 1u;
    uint n0 = gid.x * BN + tx * 2u;
    uint n1 = n0 + 1u;

    if (m0 < ne1 && n0 < ne0) {
        store_auto(dst, offset_4d(n0, m0, i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc00, dst_esize);
    }
    if (m0 < ne1 && n1 < ne0) {
        store_auto(dst, offset_4d(n1, m0, i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc01, dst_esize);
    }
    if (m1 < ne1 && n0 < ne0) {
        store_auto(dst, offset_4d(n0, m1, i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc10, dst_esize);
    }
    if (m1 < ne1 && n1 < ne0) {
        store_auto(dst, offset_4d(n1, m1, i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc11, dst_esize);
    }
}
