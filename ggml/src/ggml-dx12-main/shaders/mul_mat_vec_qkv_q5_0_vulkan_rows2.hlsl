// Combined Q/K/V projection matvec for a single-token (M=1) decode step,
// Q5_0 weights, two output rows per group. One dispatch produces all three
// projections. Wave64 reduces directly; smaller waves use a shared-memory
// cross-wave reduction.
//
// Structure mirrors mul_mat_vec_qkv_f16_wave64.hlsl (region-selected post-ops)
// but reuses the exact Q5_0 rows2 dequant/lane math from
// mul_mat_vec_q5_0_vulkan_rows2.hlsl: each group owns a (row0, row1) pair that
// shares the activation loads, and for the Q/K regions that pair is also the
// NORMAL rotation pair. The three weight matrices are contiguous in one buffer
// (Wq | Wk | Wv, host-verified), so the global output row indexes them directly:
//   Q rows   [0, q_rows)              rotate, write the rope output (dst/u0)
//   K rows   [q_rows, q_rows+k_rows)  rotate, scatter into the KV cache (temp/u1)
//   V rows   [q_rows+k_rows, ne0)     no rope, scatter into the KV cache (temp/u1)
// Per-row results are bit-identical to the three separate fused Q5_0 matvecs
// this replaces.
//
// op-param map (identical to the F16 combined path; see host QKV-shared dispatch):
//   op0  q_rows                 op2  k_rows            ne0 = total rows (guard)
//   rope (shared Q & K, consumed by mmv_rope_pair; positions src2, ff src4):
//     op1 n_dims  op3 freq_base  op4 freq_scale  op5 ext_factor
//     op6 attn_factor  op8 corr_low  op9 corr_high  op10 head_dim  op11 has_ff
//   op7  KV cache element size (== cache nb0)   op12 Q rope-out element stride
//   op13 K cache base byte offset   op14 V cache base byte offset
//   op15 KV cache row stride (nb1)
//   src0/t0 Wq|Wk|Wv   src1/t1 activation   src3/t3 K indices   src5/t5 V indices
//   dst/u0 Q rope output   temp/u1 KV cache
// op15 rides the KV cache row stride here, so the group index comes from
// group_id.x directly (group_x_2d would fold op15 into the row).
//
// RMS_FUSED variants (fl=94 pure Q5_0, fl=97 mixed Q5_0/Q8_0): the preceding
// RMS_NORM+MUL is folded in. src1 carries the pre-norm activation x and
// MMV_G_BUF the norm weight g; one pass accumulates the dots against x*g plus
// sum(x*x) and applies 1/rms once at the end (RoPE is linear in the pair, so
// scaling after the rotation is equivalent).
//   op10 bit31 carries has_ff (see MMV_ROPE_HAS_FF)   op11 rms eps (float bits)
// The mixed variant already binds src6/t6 for the Q8_0 Wv weights, so it reads g
// from src4/t4 instead and the host declines the fold when freq_factors exist.

#if RMS_FUSED
// op11 is repurposed to carry eps, so has_ff moves to the top bit of op10.
#define MMV_ROPE_HAS_FF (op10 >> 31u)
#ifndef MMV_G_BUF
#define MMV_G_BUF src6
#endif
#endif

#include "ggml_common.hlsli"
#include "rope_yarn.hlsli"

#define GROUP_SIZE 64
#define QK5_0 32
#define Q5_0_BSIZE 22

#if RMS_FUSED
#define QKV_HEAD_DIM (op10 & 0x7fffffffu)
#define QKV_ACC_SLOTS 3
#else
#define QKV_HEAD_DIM op10
#define QKV_ACC_SLOTS 2
#endif

#ifdef QKV_V_Q8_0
#define QK8_0 32
#define Q8_0_BSIZE 34
#endif

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
groupshared float wave_sums[QKV_ACC_SLOTS][GROUP_SIZE / WAVE_SIZE];
#endif

uint read_u32_unaligned(uint byte_offset) {
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = src0.Load(aligned);
    uint hi = src0.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (hi << (32u - shift));
}

uint read_u16_unaligned(uint byte_offset) {
    uint word = src0.Load(byte_offset & ~3u);
    return (word >> ((byte_offset & 2u) * 8u)) & 0xffffu;
}

float read_f16_unaligned(uint byte_offset) {
    return f16_to_f32(read_u16_unaligned(byte_offset));
}

float4 decode_q5_0_packed16(uint packed, uint qh, uint iqs) {
    uint h0 = ((qh >> iqs) << 4u) & 0x10u;
    uint h16 = (qh >> (iqs + 12u)) & 0x10u;
    uint h1 = ((qh >> (iqs + 1u)) << 4u) & 0x10u;
    uint h17 = (qh >> (iqs + 13u)) & 0x10u;
    return float4(
        (packed & 0x0fu) | h0,
        ((packed >> 4u) & 0x0fu) | h16,
        ((packed >> 8u) & 0x0fu) | h1,
        ((packed >> 12u) & 0x0fu) | h17) - 16.0f;
}

float q5_dot(uint row_offset, uint block, uint iqs, float4 x0, float4 x1) {
    uint block_offset = row_offset + block * Q5_0_BSIZE;
    float d = read_f16_unaligned(block_offset);
    uint qh = read_u32_unaligned(block_offset + 2u);
    uint packed0 = read_u16_unaligned(block_offset + 6u + iqs);
    uint packed1 = read_u16_unaligned(block_offset + 8u + iqs);
    float4 w0 = decode_q5_0_packed16(packed0, qh, iqs);
    float4 w1 = decode_q5_0_packed16(packed1, qh, iqs + 2u);
    return d * (dot(w0, x0) + dot(w1, x1));
}

#ifdef QKV_V_Q8_0
uint read_src6_u32_unaligned(uint byte_offset) {
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = src6.Load(aligned);
    uint hi = src6.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (hi << (32u - shift));
}

float read_src6_f16_unaligned(uint byte_offset) {
    uint word = src6.Load(byte_offset & ~3u);
    return f16_to_f32((word >> ((byte_offset & 2u) * 8u)) & 0xffffu);
}

float dot_q8_0(uint packed, float4 x) {
    int4 q = int4(
        (int)(packed << 24) >> 24,
        (int)(packed << 16) >> 24,
        (int)(packed <<  8) >> 24,
        (int)packed >> 24);
    return dot(float4(q), x);
}

float q8_dot(uint row_offset, uint col, float4 x0, float4 x1) {
    uint block = col / QK8_0;
    uint element = col & (QK8_0 - 1u);
    uint block_offset = row_offset + block * Q8_0_BSIZE;
    float d = read_src6_f16_unaligned(block_offset);
    uint packed0 = read_src6_u32_unaligned(block_offset + 2u + element);
    uint packed1 = read_src6_u32_unaligned(block_offset + 6u + element);
    return d * (dot_q8_0(packed0, x0) + dot_q8_0(packed1, x1));
}
#endif

WAVE_SIZE_ATTR
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row0 = group_id.x * 2u;
    if (row0 >= ne0) {
        return;
    }
    uint row1 = row0 + 1u;
    bool has_row1 = row1 < ne0;

    uint q_rows  = op0;
    uint qk_rows = op0 + op2;

    // Region (uniform across the pair): 0 = Q, 1 = K, 2 = V. Region boundaries
    // are head_dim-aligned (even), so row0 and row1 share a region.
    uint region;
    uint region_base;
    if (row0 < q_rows)       { region = 0u; region_base = 0u; }
    else if (row0 < qk_rows) { region = 1u; region_base = q_rows; }
    else                     { region = 2u; region_base = qk_rows; }
    bool rope = (region != 2u);

    uint src0_row0 = src0_offset + row0 * nb01;
    uint src0_row1 = src0_offset + row1 * nb01;
    uint src1_base = src1_offset;
#ifdef QKV_V_Q8_0
    uint v_row0 = (row0 - qk_rows) * nb02;
    uint v_row1 = (row1 - qk_rows) * nb02;
#endif

    float acc0 = 0.0f;
    float acc1 = 0.0f;
#if RMS_FUSED
    precise float acc_ss = 0.0f;
#endif
    for (uint col = local_id * 8u; col < ne00; col += GROUP_SIZE * 8u) {
#ifdef QKV_V_Q8_0
        if (region == 2u) {
            float4 x0 = asfloat(src1.Load4(src1_base + col * 4u));
            float4 x1 = asfloat(src1.Load4(src1_base + (col + 4u) * 4u));
#if RMS_FUSED
            acc_ss += dot(x0, x0) + dot(x1, x1);
            x0 *= asfloat(MMV_G_BUF.Load4(col * 4u));
            x1 *= asfloat(MMV_G_BUF.Load4((col + 4u) * 4u));
#endif
            acc0 += q8_dot(v_row0, col, x0, x1);
            if (has_row1) {
                acc1 += q8_dot(v_row1, col, x0, x1);
            }
        } else {
#endif
        uint block_start = col & ~(QK5_0 - 1u);
        uint block = col / QK5_0;
        uint iqs = (col & (QK5_0 - 1u)) / 2u;
        float4 x_low = asfloat(src1.Load4(src1_base + (block_start + iqs) * 4u));
        float4 x_high = asfloat(src1.Load4(src1_base + (block_start + iqs + 16u) * 4u));
#if RMS_FUSED
        acc_ss += dot(x_low, x_low) + dot(x_high, x_high);
        x_low  *= asfloat(MMV_G_BUF.Load4((block_start + iqs) * 4u));
        x_high *= asfloat(MMV_G_BUF.Load4((block_start + iqs + 16u) * 4u));
#endif
        float4 x0 = float4(x_low.x, x_high.x, x_low.y, x_high.y);
        float4 x1 = float4(x_low.z, x_high.z, x_low.w, x_high.w);
        acc0 += q5_dot(src0_row0, block, iqs, x0, x1);
        if (has_row1) {
            acc1 += q5_dot(src0_row1, block, iqs, x0, x1);
        }
#ifdef QKV_V_Q8_0
        }
#endif
    }

    float sum0 = WaveActiveSum(acc0);
    float sum1 = WaveActiveSum(acc1);
#if RMS_FUSED
    float sum_ss = WaveActiveSum(acc_ss);
#endif

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
    if (WaveIsFirstLane()) {
        uint wave = local_id / WAVE_SIZE;
        wave_sums[0][wave] = sum0;
        wave_sums[1][wave] = sum1;
#if RMS_FUSED
        wave_sums[2][wave] = sum_ss;
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_id == 0u) {
        sum0 = 0.0f;
        sum1 = 0.0f;
#if RMS_FUSED
        sum_ss = 0.0f;
#endif
        [unroll]
        for (uint wave = 0; wave < GROUP_SIZE / WAVE_SIZE; ++wave) {
            sum0 += wave_sums[0][wave];
            sum1 += wave_sums[1][wave];
#if RMS_FUSED
            sum_ss += wave_sums[2][wave];
#endif
        }
    }
#endif

    if (local_id == 0u) {
#if RMS_FUSED
        float rms_scale = 1.0f / sqrt(sum_ss / (float)ne00 + asfloat(op11));
        sum0 *= rms_scale;
        sum1 *= rms_scale;
#endif
        uint cache_esize = op7;
        uint cache_nb1   = op15;
        uint local_row0  = row0 - region_base;
        if (rope) {
            uint pair_in_head = (row0 % QKV_HEAD_DIM) / 2u;
            float out0, out1;
            mmv_rope_pair(pair_in_head, sum0, sum1, out0, out1);
            if (region == 0u) {
                store_auto(dst, dst_offset + local_row0 * op12, out0, dst_esize);
                if (has_row1) {
                    store_auto(dst, dst_offset + (local_row0 + 1u) * op12, out1, dst_esize);
                }
            } else {
                int row_idx = asint(src3.Load(0));
                uint base = op13 + (uint)row_idx * cache_nb1;
                store_auto(temp, base + local_row0 * cache_esize, out0, cache_esize);
                if (has_row1) {
                    store_auto(temp, base + (local_row0 + 1u) * cache_esize, out1, cache_esize);
                }
            }
        } else {
            int row_idx = asint(src5.Load(0));
            uint base = op14 + (uint)row_idx * cache_nb1;
            store_auto(temp, base + local_row0 * cache_esize, sum0, cache_esize);
            if (has_row1) {
                store_auto(temp, base + (local_row0 + 1u) * cache_esize, sum1, cache_esize);
            }
        }
    }
}
