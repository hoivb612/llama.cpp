// Tiled Q8_0 x Q8_1 batch MUL_MAT using packed int8 dot products.
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK8_0       32
#define Q8_0_BSIZE  34
#define Q8_1_BSIZE  36
#define BM           32
#define BN           32

groupshared uint  tile_w_qs[BN][8];
groupshared float tile_w_d[BN];
groupshared uint  tile_a_qs[BM][8];
groupshared float tile_a_d[BM];

uint read_byte_q80(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float read_f16_q80(ByteAddressBuffer buf, uint byte_off) {
#ifdef Q8_TILED_BYTE_LOADS
    return f16_to_f32(read_byte_q80(buf, byte_off) |
                      (read_byte_q80(buf, byte_off + 1u) << 8u));
#else
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
#endif
}

uint read_u32_q80(ByteAddressBuffer buf, uint byte_off) {
#ifdef Q8_TILED_BYTE_LOADS
    return read_byte_q80(buf, byte_off) |
           (read_byte_q80(buf, byte_off + 1u) << 8u) |
           (read_byte_q80(buf, byte_off + 2u) << 16u) |
           (read_byte_q80(buf, byte_off + 3u) << 24u);
#else
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) {
        return lo;
    }
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
#endif
}

int dot4_i8_q80(uint a, uint b, int acc) {
    return dot4add_i8packed(a, b, acc);
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
    uint num_blocks = K / QK8_0;

    precise float acc00 = 0.0f;
    precise float acc01 = 0.0f;
    precise float acc10 = 0.0f;
    precise float acc11 = 0.0f;

    for (uint block = 0; block < num_blocks; ++block) {
        uint local_row = flat_id / 8u;
        uint packed_idx = flat_id & 7u;

        uint global_n = gid.x * BN + local_row;
        uint w_qs = 0u;
        if (global_n < ne01) {
            uint row_off = src0_offset + global_n * nb01
                         + i2_src0 * nb02 + i3_src0 * nb03;
            uint block_off = row_off + block * Q8_0_BSIZE;
            w_qs = read_u32_q80(src0, block_off + 2u + packed_idx * 4u);
            if (packed_idx == 0u) {
                tile_w_d[local_row] = read_f16_q80(src0, block_off);
            }
        } else if (packed_idx == 0u) {
            tile_w_d[local_row] = 0.0f;
        }
        tile_w_qs[local_row][packed_idx] = w_qs;

        uint global_m = gid.y * BM + local_row;
        uint a_qs = 0u;
        if (global_m < ne11) {
            uint flat_row = (i3 * ne12 + i2) * ne11 + global_m;
            uint block_off = src1_offset
                           + (flat_row * num_blocks + block) * Q8_1_BSIZE;
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

        int dot00 = 0;
        int dot01 = 0;
        int dot10 = 0;
        int dot11 = 0;
        [unroll]
        for (uint q = 0; q < 8u; ++q) {
            uint a0 = tile_a_qs[m0][q];
            uint a1 = tile_a_qs[m1][q];
            dot00 = dot4_i8_q80(tile_w_qs[n0][q], a0, dot00);
            dot01 = dot4_i8_q80(tile_w_qs[n1][q], a0, dot01);
            dot10 = dot4_i8_q80(tile_w_qs[n0][q], a1, dot10);
            dot11 = dot4_i8_q80(tile_w_qs[n1][q], a1, dot11);
        }

        acc00 += tile_w_d[n0] * tile_a_d[m0] * float(dot00);
        acc01 += tile_w_d[n1] * tile_a_d[m0] * float(dot01);
        acc10 += tile_w_d[n0] * tile_a_d[m1] * float(dot10);
        acc11 += tile_w_d[n1] * tile_a_d[m1] * float(dot11);

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
