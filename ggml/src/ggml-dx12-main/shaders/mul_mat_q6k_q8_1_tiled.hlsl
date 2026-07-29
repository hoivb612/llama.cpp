// Tiled Q6_K x Q8_1 batch MUL_MAT using packed int8 dot products.
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK_K          256
#define Q6K_BSIZE     210
#define Q8_1_BSIZE     36
#define BM             32
#define BN             32

groupshared uint  tile_w_qs[BN][8];
groupshared float tile_w_d0[BN];
groupshared float tile_w_d1[BN];
groupshared uint  tile_a_qs[BM][8];
groupshared float tile_a_d[BM];

// Q6_K superblocks are 210 bytes, so neither the block base nor the row base
// is 4-byte aligned; every load has to go through the unaligned path.
uint read_byte_q6k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

int read_i8_q6k(ByteAddressBuffer buf, uint byte_off) {
    return int(read_byte_q6k(buf, byte_off) ^ 0x80u) - 128;
}

float read_f16_q6k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint read_u32_q6k(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) {
        return lo;
    }
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
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

        uint q6_block = block / 8u;
        uint tile_in_block = block & 7u;
        // Each 32-element tile sits in one 128-element half (h) and one of the
        // four 32-element quarters (sub) that ggml's dequant walks in lockstep.
        uint sub = tile_in_block & 3u;
        uint h = tile_in_block >> 2u;

        uint global_n = gid.x * BN + local_row;
        uint w_qs = 0u;
        if (global_n < ne01) {
            uint row_off = src0_offset + global_n * nb01
                         + i2_src0 * nb02 + i3_src0 * nb03;
            uint block_off = row_off + q6_block * Q6K_BSIZE;

            uint ql_word = read_u32_q6k(src0, block_off + h * 64u
                                            + (sub & 1u) * 32u + packed_idx * 4u);
            uint low = (sub < 2u) ? (ql_word & 0x0F0F0F0Fu)
                                  : ((ql_word >> 4u) & 0x0F0F0F0Fu);
            uint qh_word = read_u32_q6k(src0, block_off + 128u + h * 32u
                                            + packed_idx * 4u);
            uint high = ((qh_word >> (2u * sub)) & 0x03030303u) << 4u;

            // v holds 0..63 per byte; subtract 32 by keeping the low 5 bits and
            // sign-filling the top 3 when bit 5 is clear (0x20 * 0x07 == 0xE0).
            uint v = low | high;
            uint neg = (v & 0x20202020u) ^ 0x20202020u;
            w_qs = (v & 0x1F1F1F1Fu) | (neg * 0x07u);

            if (packed_idx == 0u) {
                float d = read_f16_q6k(src0, block_off + 208u);
                // 16 scales per superblock, so one tile spans two of them:
                // elements 0-15 take sc0 and 16-31 take sc0 + 1.
                uint sidx0 = h * 8u + 2u * sub;
                tile_w_d0[local_row] = d * float(read_i8_q6k(src0, block_off + 192u + sidx0));
                tile_w_d1[local_row] = d * float(read_i8_q6k(src0, block_off + 192u + sidx0 + 1u));
            }
        } else if (packed_idx == 0u) {
            tile_w_d0[local_row] = 0.0f;
            tile_w_d1[local_row] = 0.0f;
        }
        tile_w_qs[local_row][packed_idx] = w_qs;

        uint global_m = gid.y * BM + local_row;
        uint a_qs = 0u;
        if (global_m < ne11) {
            uint flat_row = (i3 * ne12 + i2) * ne11 + global_m;
            uint block_off = src1_offset
                           + (flat_row * num_q8_blocks + block) * Q8_1_BSIZE;
            a_qs = src1.Load(block_off + 4u + packed_idx * 4u);
            if (packed_idx == 0u) {
                tile_a_d[local_row] = f16_to_f32(src1.Load(block_off) & 0xFFFFu);
            }
        } else if (packed_idx == 0u) {
            tile_a_d[local_row] = 0.0f;
        }
        tile_a_qs[local_row][packed_idx] = a_qs;

        GroupMemoryBarrierWithGroupSync();

        uint m0 = ty * 2u;
        uint m1 = m0 + 1u;
        uint n0 = tx * 2u;
        uint n1 = n0 + 1u;

        int dot00a = 0, dot01a = 0, dot10a = 0, dot11a = 0;
        int dot00b = 0, dot01b = 0, dot10b = 0, dot11b = 0;
        [unroll]
        for (uint q = 0; q < 4u; ++q) {
            uint a0 = tile_a_qs[m0][q];
            uint a1 = tile_a_qs[m1][q];
            dot00a = dot4add_i8packed(tile_w_qs[n0][q], a0, dot00a);
            dot01a = dot4add_i8packed(tile_w_qs[n1][q], a0, dot01a);
            dot10a = dot4add_i8packed(tile_w_qs[n0][q], a1, dot10a);
            dot11a = dot4add_i8packed(tile_w_qs[n1][q], a1, dot11a);
        }
        [unroll]
        for (uint q2 = 4u; q2 < 8u; ++q2) {
            uint a0 = tile_a_qs[m0][q2];
            uint a1 = tile_a_qs[m1][q2];
            dot00b = dot4add_i8packed(tile_w_qs[n0][q2], a0, dot00b);
            dot01b = dot4add_i8packed(tile_w_qs[n1][q2], a0, dot01b);
            dot10b = dot4add_i8packed(tile_w_qs[n0][q2], a1, dot10b);
            dot11b = dot4add_i8packed(tile_w_qs[n1][q2], a1, dot11b);
        }

        acc00 += tile_a_d[m0] * (tile_w_d0[n0] * float(dot00a) + tile_w_d1[n0] * float(dot00b));
        acc01 += tile_a_d[m0] * (tile_w_d0[n1] * float(dot01a) + tile_w_d1[n1] * float(dot01b));
        acc10 += tile_a_d[m1] * (tile_w_d0[n0] * float(dot10a) + tile_w_d1[n0] * float(dot10b));
        acc11 += tile_a_d[m1] * (tile_w_d0[n1] * float(dot11a) + tile_w_d1[n1] * float(dot11b));

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
