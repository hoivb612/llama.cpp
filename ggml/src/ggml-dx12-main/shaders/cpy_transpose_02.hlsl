#include "ggml_common.hlsli"

#define TILE_DIM 32

groupshared uint tile[TILE_DIM][TILE_DIM + 1];

uint load_bits(ByteAddressBuffer buf, uint off, uint esize) {
    if (esize == 4u) {
        return buf.Load(off);
    }
    uint word = buf.Load(off & ~3u);
    return (word >> ((off & 2u) * 8u)) & 0xFFFFu;
}

void store_bits16(RWByteAddressBuffer buf, uint off, uint bits) {
    uint word_off = off & ~3u;
    uint shift = (off & 2u) * 8u;
    uint mask = 0xFFFFu << shift;
    uint expected;
    uint original;
    [allow_uav_condition] do {
        buf.InterlockedOr(word_off, 0u, expected);
        uint desired = (expected & ~mask) | ((bits & 0xFFFFu) << shift);
        buf.InterlockedCompareExchange(word_off, expected, desired, original);
    } while (original != expected);
}

[numthreads(32, 8, 1)]
void main(uint3 group_id : SV_GroupID, uint3 local_id : SV_GroupThreadID) {
    uint i1 = group_id.z % ne1;
    uint i3 = group_id.z / ne1;
    uint src_esize = src0_esize == 3u ? 2u : src0_esize;
    uint out_esize = dst_esize == 3u ? 2u : dst_esize;

    [unroll]
    for (uint y = 0; y < 4u; ++y) {
        uint i0 = group_id.x * TILE_DIM + local_id.y + 8u * y;
        uint i2 = group_id.y * TILE_DIM + local_id.x;
        if (i0 < ne00 && i2 < ne02) {
            uint src_off = src0_offset + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
            tile[local_id.y + 8u * y][local_id.x] = load_bits(src0, src_off, src_esize);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint y = 0; y < 4u; ++y) {
        uint i0 = group_id.x * TILE_DIM + local_id.x;
        uint i2 = group_id.y * TILE_DIM + local_id.y + 8u * y;
        if (i0 < ne0 && i2 < ne2) {
            uint bits = tile[local_id.x][local_id.y + 8u * y];
            uint dst_off = dst_offset + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3;
            if (out_esize == 4u) {
                dst.Store(dst_off, bits);
            } else {
                store_bits16(dst, dst_off, bits);
            }
        }
    }
}
