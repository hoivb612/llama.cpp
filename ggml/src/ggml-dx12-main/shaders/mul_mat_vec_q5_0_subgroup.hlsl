#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE 64
#endif
#define QK5_0 32
#define Q5_0_BSIZE 22
#define VALUES_PER_THREAD 8

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
groupshared float wave_sums[GROUP_SIZE / WAVE_SIZE];
#endif

uint read_u32_unaligned(ByteAddressBuffer buf, uint byte_offset) {
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) {
        return lo;
    }
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

void read_u32x2_unaligned(ByteAddressBuffer buf, uint byte_offset, out uint lo, out uint hi) {
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 3u) * 8u;
    uint w0 = buf.Load(aligned);
    uint w1 = buf.Load(aligned + 4u);
    if (shift == 0u) {
        lo = w0;
        hi = w1;
        return;
    }
    uint w2 = buf.Load(aligned + 8u);
    lo = (w0 >> shift) | (w1 << (32u - shift));
    hi = (w1 >> shift) | (w2 << (32u - shift));
}

float4 decode_q5_0_4(uint qs, uint qh, uint nibble_shift) {
    uint4 q = uint4(
        (qs >> (nibble_shift +  0u)) & 0x0fu,
        (qs >> (nibble_shift +  8u)) & 0x0fu,
        (qs >> (nibble_shift + 16u)) & 0x0fu,
        (qs >> (nibble_shift + 24u)) & 0x0fu);
    uint4 h = uint4(
        (qh & 0x1u) << 4u,
        ((qh >> 1u) & 0x1u) << 4u,
        ((qh >> 2u) & 0x1u) << 4u,
        ((qh >> 3u) & 0x1u) << 4u);
    return float4(q | h) - 16.0f;
}

WAVE_SIZE_ATTR
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row = group_x_2d(group_id);
    if (row >= ne0) {
        return;
    }

    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + i2_src0 * nb02 + i3_src0 * nb03 + row * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    float acc = 0.0f;
    for (uint k = local_id * VALUES_PER_THREAD; k < ne00; k += GROUP_SIZE * VALUES_PER_THREAD) {
        uint block = k / QK5_0;
        uint elem = k & (QK5_0 - 1u);
        uint block_offset = src0_row + block * Q5_0_BSIZE;

        uint d_word = src0.Load(block_offset & ~3u);
        float d = f16_to_f32((d_word >> ((block_offset & 2u) * 8u)) & 0xffffu);
        uint qh = read_u32_unaligned(src0, block_offset + 2u);

        uint qs0;
        uint qs1;
        read_u32x2_unaligned(src0, block_offset + 6u + (elem & 15u), qs0, qs1);

        uint nibble_shift = elem >= 16u ? 4u : 0u;
        float4 w0 = decode_q5_0_4(qs0, qh >> elem, nibble_shift);
        float4 w1 = decode_q5_0_4(qs1, qh >> (elem + 4u), nibble_shift);
        float4 x0 = asfloat(src1.Load4(src1_base + k * 4u));
        float4 x1 = asfloat(src1.Load4(src1_base + (k + 4u) * 4u));

        acc += d * (dot(w0, x0) + dot(w1, x1));
    }

    float sum = WaveActiveSum(acc);

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
    if (WaveIsFirstLane()) {
        wave_sums[local_id / WAVE_SIZE] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_id == 0u) {
        sum = 0.0f;
        [unroll]
        for (uint wave = 0; wave < GROUP_SIZE / WAVE_SIZE; ++wave) {
            sum += wave_sums[wave];
        }
    }
#endif

    if (local_id == 0u) {
        sum += load_fused_bias(row, i2, i3);
        uint dst_offset_row = offset_4d(row, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, dst_offset_row, sum, dst_esize);
    }
}
