// Q6_K dp4a matvec with direct packed-block loads.
// Each lane loads the unique Q6_K bytes for four 4-element groups.
#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE 64
#endif

#define QK_K        256
#define Q6K_BSIZE   210
#define Q8_1_BSIZE  36
#define NUM_ROWS    2

groupshared float shared_acc0[64];
groupshared float shared_acc1[64];

uint load_u32_unaligned(ByteAddressBuffer buf, uint byte_off) {
    uint base = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(base);
    if (shift == 0u) {
        return lo;
    }
    uint hi = buf.Load(base + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

int load_s8(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    uint value = (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
    return value < 128u ? (int)value : (int)value - 256;
}

float load_f16_unaligned(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

void load_q8_chunk(uint q8_super, uint block_idx, uint byte_idx,
                   out float d, out uint qs, out int sum) {
    uint off = q8_super + block_idx * Q8_1_BSIZE;
    d = f16_to_f32(src1.Load(off) & 0xFFFFu);
    qs = src1.Load(off + 4u + byte_idx);
    sum = 0;
    sum = dot4add_i8packed(0x01010101u, qs, sum);
}

float dot_q6_row(uint block_off, uint v_im, uint v_in,
                 float d0, float d1, float d2, float d3,
                 uint a0, uint a1, uint a2, uint a3,
                 int asum0, int asum1, int asum2, int asum3) {
    uint l0 = 4u * v_in;
    uint ql0 = load_u32_unaligned(src0, block_off + 64u * v_im + l0);
    uint ql1 = load_u32_unaligned(src0, block_off + 64u * v_im + l0 + 32u);
    uint qh = load_u32_unaligned(src0, block_off + 128u + 32u * v_im + l0);

    uint q0 = (ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4u);
    uint q1 = (ql1 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2u);
    uint q2 = ((ql0 >> 4u) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
    uint q3 = ((ql1 >> 4u) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2u);

    int dot0 = 0; dot0 = dot4add_i8packed(q0, a0, dot0);
    int dot1 = 0; dot1 = dot4add_i8packed(q1, a1, dot1);
    int dot2 = 0; dot2 = dot4add_i8packed(q2, a2, dot2);
    int dot3 = 0; dot3 = dot4add_i8packed(q3, a3, dot3);

    uint scale_base = block_off + 192u + 8u * v_im + v_in / 4u;
    float s0 = float(load_s8(src0, scale_base));
    float s1 = float(load_s8(src0, scale_base + 2u));
    float s2 = float(load_s8(src0, scale_base + 4u));
    float s3 = float(load_s8(src0, scale_base + 6u));
    float d = load_f16_unaligned(src0, block_off + 208u);

    float sum = s0 * d0 * float(dot0 - 32 * asum0);
    sum = mad(s1 * d1, float(dot1 - 32 * asum1), sum);
    sum = mad(s2 * d2, float(dot2 - 32 * asum2), sum);
    sum = mad(s3 * d3, float(dot3 - 32 * asum3), sum);
    return d * sum;
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * NUM_ROWS;
    if (row0 >= ne0) {
        return;
    }

    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint num_blocks = ne00 / QK_K;
    uint num_q8 = ne00 / 32u;
    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint row1 = min(row0 + 1u, ne0 - 1u);
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + row1 * nb01;

    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_vec_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_q8 * Q8_1_BSIZE;

    uint itid = tid & 15u;
    uint block_slot = tid >> 4u;
    uint block_stride = GROUP_SIZE / 16u;
    uint v_im = itid >> 3u;
    uint v_in = itid & 7u;
    uint byte_idx = 4u * v_in;
    uint q8_block0 = 4u * v_im;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    for (uint block_idx = block_slot; block_idx < num_blocks; block_idx += block_stride) {
        uint q8_super = q8_vec_base + block_idx * 8u * Q8_1_BSIZE;
        float d0, d1, d2, d3;
        uint a0, a1, a2, a3;
        int asum0, asum1, asum2, asum3;
        load_q8_chunk(q8_super, q8_block0 + 0u, byte_idx, d0, a0, asum0);
        load_q8_chunk(q8_super, q8_block0 + 1u, byte_idx, d1, a1, asum1);
        load_q8_chunk(q8_super, q8_block0 + 2u, byte_idx, d2, a2, asum2);
        load_q8_chunk(q8_super, q8_block0 + 3u, byte_idx, d3, a3, asum3);

        acc0 += dot_q6_row(src0_row0 + block_idx * Q6K_BSIZE, v_im, v_in,
                           d0, d1, d2, d3, a0, a1, a2, a3,
                           asum0, asum1, asum2, asum3);
        acc1 += dot_q6_row(src0_row1 + block_idx * Q6K_BSIZE, v_im, v_in,
                           d0, d1, d2, d3, a0, a1, a2, a3,
                           asum0, asum1, asum2, asum3);
    }

    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
#if !defined(WAVE_SIZE) || GROUP_SIZE != WAVE_SIZE
    uint wave_lanes = WaveGetLaneCount();
    uint wave_id = tid / wave_lanes;
    uint num_waves = (GROUP_SIZE + wave_lanes - 1u) / wave_lanes;
    if (WaveIsFirstLane()) {
        shared_acc0[wave_id] = wave_sum0;
        shared_acc1[wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();
#endif

    if (tid == 0u) {
#if defined(WAVE_SIZE) && GROUP_SIZE == WAVE_SIZE
        float result0 = wave_sum0;
        float result1 = wave_sum1;
#else
        float result0 = shared_acc0[0];
        float result1 = shared_acc1[0];
        for (uint w = 1u; w < num_waves; ++w) {
            result0 += shared_acc0[w];
            result1 += shared_acc1[w];
        }
#endif
        result0 += load_fused_bias(row0, i2, i3);
        uint off0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off0, result0, dst_esize);

        if (row0 + 1u < ne0) {
            result1 += load_fused_bias(row0 + 1u, i2, i3);
            uint off1 = offset_4d(row0 + 1u, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off1, result1, dst_esize);
        }
    }
}
