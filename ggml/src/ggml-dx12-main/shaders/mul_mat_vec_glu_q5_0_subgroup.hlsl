#include "ggml_common.hlsli"

// RMS_FUSED variant (mul_mat_vec_glu_q5_0_subgroup_rms.hlsl): folds the
// preceding RMS_NORM + MUL(norm_weight) in. src1 carries the pre-norm
// activation x and src6 the norm weight g; one pass accumulates the dots
// against x*g plus sum(x*x) and applies 1/rms once. op14 = eps (float bits).

#ifndef GROUP_SIZE
#define GROUP_SIZE        64
#endif
#define QK5_0             32
#define Q5_0_BSIZE        22
#define VALUES_PER_THREAD  8

// Cross-wave reduction scratch (multi-wave devices only). Sized for the
// smallest compiled wave (16) => GROUP_SIZE/16 partials per accumulator.
// Unused on the single-wave (GROUP_SIZE == WAVE_SIZE) fast path.
#if !(defined(WAVE_SIZE) && (GROUP_SIZE == WAVE_SIZE))
groupshared float glu_wave_gate[GROUP_SIZE / 16];
groupshared float glu_wave_up[GROUP_SIZE / 16];
#if RMS_FUSED
groupshared float glu_wave_ss[GROUP_SIZE / 16];
#endif
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

float q5_0_dot(ByteAddressBuffer weights, uint block_offset, uint elem, float4 x0, float4 x1) {
    uint d_word = weights.Load(block_offset & ~3u);
    float d = f16_to_f32((d_word >> ((block_offset & 2u) * 8u)) & 0xffffu);
    uint qh = read_u32_unaligned(weights, block_offset + 2u);
    uint qs0;
    uint qs1;
    read_u32x2_unaligned(weights, block_offset + 6u + (elem & 15u), qs0, qs1);
    uint nibble_shift = elem >= 16u ? 4u : 0u;
    float4 w0 = decode_q5_0_4(qs0, qh >> elem, nibble_shift);
    float4 w1 = decode_q5_0_4(qs1, qh >> (elem + 4u), nibble_shift);
    return d * (dot(w0, x0) + dot(w1, x1));
}

WAVE_SIZE_ATTR
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row = group_x_2d(group_id);
    if (row >= ne0) {
        return;
    }

    uint i2 = group_id.z % ne2;
    uint i3 = group_id.z / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint gate_row = src0_offset + i2_src0 * nb02 + i3_src0 * nb03 + row * nb01;
    uint up_row = op1 + i2_src0 * nb02 + i3_src0 * nb03 + row * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    float gate = 0.0f;
    float up = 0.0f;
#if RMS_FUSED
    float ss = 0.0f;
#endif
    for (uint k = local_id * VALUES_PER_THREAD; k < ne00; k += GROUP_SIZE * VALUES_PER_THREAD) {
        uint block = k / QK5_0;
        uint elem = k & (QK5_0 - 1u);
        float4 x0 = asfloat(src1.Load4(src1_base + k * 4u));
        float4 x1 = asfloat(src1.Load4(src1_base + (k + 4u) * 4u));
#if RMS_FUSED
        ss += dot(x0, x0) + dot(x1, x1);
        x0 *= asfloat(src6.Load4(k * 4u));
        x1 *= asfloat(src6.Load4((k + 4u) * 4u));
#endif
        gate += q5_0_dot(src0, gate_row + block * Q5_0_BSIZE, elem, x0, x1);
        up += q5_0_dot(src2, up_row + block * Q5_0_BSIZE, elem, x0, x1);
    }

#if defined(WAVE_SIZE) && (GROUP_SIZE == WAVE_SIZE)
    gate = WaveActiveSum(gate);
    up = WaveActiveSum(up);
#if RMS_FUSED
    float rms_scale = 1.0f / sqrt(WaveActiveSum(ss) / (float)ne00 + asfloat(op14));
    gate *= rms_scale;
    up   *= rms_scale;
#endif
    if (local_id == 0u) {
        float result = (gate / (1.0f + exp(-gate))) * up;
        uint dst_row = offset_4d(row, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, dst_row, result, dst_esize);
    }
#else
    float wave_gate = WaveActiveSum(gate);
    float wave_up   = WaveActiveSum(up);
#if RMS_FUSED
    float wave_ss   = WaveActiveSum(ss);
#endif
    uint  wave_id   = local_id / WARP_SIZE;
    uint  num_waves = GROUP_SIZE / WARP_SIZE;
    if (WaveIsFirstLane()) {
        glu_wave_gate[wave_id] = wave_gate;
        glu_wave_up[wave_id]   = wave_up;
#if RMS_FUSED
        glu_wave_ss[wave_id]   = wave_ss;
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_id == 0u) {
        float g = 0.0f;
        float u = 0.0f;
#if RMS_FUSED
        float s = 0.0f;
#endif
        for (uint w = 0u; w < num_waves; ++w) {
            g += glu_wave_gate[w];
            u += glu_wave_up[w];
#if RMS_FUSED
            s += glu_wave_ss[w];
#endif
        }
#if RMS_FUSED
        float rms_scale = 1.0f / sqrt(s / (float)ne00 + asfloat(op14));
        g *= rms_scale;
        u *= rms_scale;
#endif
        float result = (g / (1.0f + exp(-g))) * u;
        uint dst_row = offset_4d(row, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, dst_row, result, dst_esize);
    }
#endif
}
