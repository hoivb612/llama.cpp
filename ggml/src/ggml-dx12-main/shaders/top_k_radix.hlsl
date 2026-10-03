#include "ggml_common.hlsli"

#define GROUP_SIZE 1024
#define RADIX_BITS 8
#define RADIX_SIZE (1u << RADIX_BITS)

groupshared uint histo[RADIX_SIZE];
groupshared uint selected_bucket;
groupshared uint selected_above;
groupshared uint out_count;

uint float_key(float value) {
    uint key = asuint(value);
    return (key & 0x80000000u) != 0u ? key ^ 0xFFFFFFFFu : key | 0x80000000u;
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint nrows = ne01 * ne02 * ne03;
    uint row = group_id.y * 65535u + group_id.x;
    if (row >= nrows) {
        return;
    }

    uint prefix = 0u;
    uint desired = ne0;

    [unroll]
    for (int shift = 24; shift >= 0; shift -= RADIX_BITS) {
        if (tid < RADIX_SIZE) {
            histo[tid] = 0u;
        }
        GroupMemoryBarrierWithGroupSync();

        uint hi_mask = shift == 24 ? 0u : 0xFFFFFFFFu << (shift + RADIX_BITS);
        uint prefix_hi = prefix & hi_mask;
        for (uint i = tid; i < ne00; i += GROUP_SIZE) {
            uint src_off = src0_offset + (row * ne00 + i) * 4u;
            uint key = float_key(asfloat(src0.Load(src_off)));
            if ((key & hi_mask) == prefix_hi) {
                InterlockedAdd(histo[(key >> shift) & (RADIX_SIZE - 1u)], 1u);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid == 0u) {
            uint above = 0u;
            uint bucket = 0u;
            for (int b = RADIX_SIZE - 1; b >= 0; --b) {
                uint count = histo[b];
                if (above + count >= desired) {
                    bucket = (uint)b;
                    break;
                }
                above += count;
            }
            selected_bucket = bucket;
            selected_above = above;
        }
        GroupMemoryBarrierWithGroupSync();

        prefix |= selected_bucket << shift;
        desired -= selected_above;
    }

    if (tid == 0u) {
        out_count = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint dst_row = dst_offset + row * ne0 * 4u;
    for (uint i = tid; i < ne00; i += GROUP_SIZE) {
        uint src_off = src0_offset + (row * ne00 + i) * 4u;
        if (float_key(asfloat(src0.Load(src_off))) > prefix) {
            uint pos;
            InterlockedAdd(out_count, 1u, pos);
            dst.Store(dst_row + pos * 4u, i);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = tid; i < ne00; i += GROUP_SIZE) {
        uint src_off = src0_offset + (row * ne00 + i) * 4u;
        if (float_key(asfloat(src0.Load(src_off))) == prefix) {
            uint pos;
            InterlockedAdd(out_count, 1u, pos);
            if (pos < ne0) {
                dst.Store(dst_row + pos * 4u, i);
            }
        }
    }
}
