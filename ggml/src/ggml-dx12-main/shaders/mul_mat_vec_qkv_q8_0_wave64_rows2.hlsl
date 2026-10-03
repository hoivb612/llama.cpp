// Combined Q/K/V projection matvec for a single-token (M=1) decode step,
// Q8_0 weights, AMD wave64, two output rows per group. One dispatch produces
// all three projections.
//
// Structure mirrors mul_mat_vec_qkv_f16_wave64.hlsl (region-selected post-ops)
// but reuses the exact Q8_0 rows2 decode math from
// mul_mat_vec_q8_0_wave64_rows2.hlsl: each group owns a (row0, row1) pair that
// shares the activation loads, and for the Q/K regions that pair is also the
// NORMAL rotation pair. The three weight matrices are contiguous in one buffer
// (Wq | Wk | Wv, host-verified), so the global output row indexes them directly:
//   Q rows   [0, q_rows)              rotate, write the rope output (dst/u0)
//   K rows   [q_rows, q_rows+k_rows)  rotate, scatter into the KV cache (temp/u1)
//   V rows   [q_rows+k_rows, ne0)     no rope, scatter into the KV cache (temp/u1)
// Per-row results are bit-identical to the three separate fused Q8_0 matvecs
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
// RMS_FUSED variant (mul_mat_vec_qkv_q8_0_mr256_rms.hlsl):
//   Folds the preceding RMS_NORM + MUL(norm_weight) into this dispatch. src1
//   carries the pre-norm activation x, src6 the norm weight g, and op11 is
//   repurposed for eps (has_ff moves into bit 31 of op10). The RMS scale is a
//   scalar over the row, so one pass accumulates both the dot products and
//   sum(x*x); rope is linear in (sum0, sum1) so scaling before it is exact.
//
// QKV_DP4A variant (mul_mat_vec_qkv_q8_0_dp4a.hlsl):
//   src1 is the Q8_1 scratch from the quantize pre-pass instead of the F32
//   activation, and the dot products use dot4add_i8packed. Mutually exclusive
//   with RMS_FUSED, which needs the un-normalized F32 activation.

#include "ggml_common.hlsli"
#if RMS_FUSED
#define MMV_ROPE_HAS_FF (op10 >> 31u)
#endif
#include "rope_yarn.hlsli"

#if RMS_FUSED
#define QKV_HEAD_DIM  (op10 & 0x7fffffffu)
#define QKV_ACC_SLOTS 3
#else
#define QKV_HEAD_DIM  op10
#define QKV_ACC_SLOTS 2
#endif

#ifndef GROUP_SIZE
#define GROUP_SIZE       64
#endif
#define QK8_0            32
#define Q8_0_BSIZE       34
#define Q8_1_BSIZE       36
#define VALUES_PER_LANE   8
#if QKV_DP4A
// 8 lanes cooperate on one Q8_0 block, so a group covers this many blocks
// per iteration.
#define BLOCKS_PER_ITER  (GROUP_SIZE / 8)
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

float read_f16_unaligned(uint byte_offset) {
    uint word = src0.Load(byte_offset & ~3u);
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
    uint src1_row  = src1_offset;

#if QKV_DP4A
    // Match the precise accumulation of the fl=17 matvec this replaces, so the
    // fused result stays bit-identical to the three separate dispatches.
    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;
#else
    float acc0 = 0.0f;
    float acc1 = 0.0f;
#endif
#if RMS_FUSED
    float acc_ss = 0.0f;
#endif
#if QKV_DP4A
    // src1 is the Q8_1 scratch produced by the quantize pre-pass, so the
    // activation is read once as packed int8 and consumed by dot4add.
    // M=1, so the flat row base is src1_offset.
    uint num_blocks = ne00 / QK8_0;
    uint sub  = local_id / 8u;
    uint lane = local_id % 8u;
    uint l0   = lane * 4u;
    for (uint block_idx = sub; block_idx < num_blocks;
         block_idx += BLOCKS_PER_ITER) {
        uint q8_off = src1_row + block_idx * Q8_1_BSIZE;
        float a_d = f16_to_f32(src1.Load(q8_off) & 0xffffu);
        uint a_packed = src1.Load(q8_off + 4u + l0);

        uint w_off0 = src0_row0 + block_idx * Q8_0_BSIZE;
        float d0 = read_f16_unaligned(w_off0);
        uint w_packed0 = read_u32_unaligned(w_off0 + 2u + l0);
        int isum0 = 0;
        isum0 = dot4add_i8packed(w_packed0, a_packed, isum0);
        acc0 += d0 * a_d * float(isum0);

        if (has_row1) {
            uint w_off1 = src0_row1 + block_idx * Q8_0_BSIZE;
            float d1 = read_f16_unaligned(w_off1);
            uint w_packed1 = read_u32_unaligned(w_off1 + 2u + l0);
            int isum1 = 0;
            isum1 = dot4add_i8packed(w_packed1, a_packed, isum1);
            acc1 += d1 * a_d * float(isum1);
        }
    }
#else
    for (uint k = local_id * VALUES_PER_LANE; k < ne00;
         k += GROUP_SIZE * VALUES_PER_LANE) {
        uint block = k / QK8_0;
        uint element = k & (QK8_0 - 1u);
        uint block_offset0 = src0_row0 + block * Q8_0_BSIZE;
        uint block_offset1 = src0_row1 + block * Q8_0_BSIZE;
        float4 x0 = asfloat(src1.Load4(src1_row + k * 4u));
        float4 x1 = asfloat(src1.Load4(src1_row + (k + 4u) * 4u));
#if RMS_FUSED
        acc_ss += dot(x0, x0) + dot(x1, x1);
        x0 *= asfloat(src6.Load4(k * 4u));
        x1 *= asfloat(src6.Load4((k + 4u) * 4u));
#endif

        float d0 = read_f16_unaligned(block_offset0);
        uint packed00 = read_u32_unaligned(block_offset0 + 2u + element);
        uint packed01 = read_u32_unaligned(block_offset0 + 6u + element);
        acc0 += d0 * (dot_q8_0(packed00, x0) + dot_q8_0(packed01, x1));

        if (has_row1) {
            float d1 = read_f16_unaligned(block_offset1);
            uint packed10 = read_u32_unaligned(block_offset1 + 2u + element);
            uint packed11 = read_u32_unaligned(block_offset1 + 6u + element);
            acc1 += d1 * (dot_q8_0(packed10, x0) + dot_q8_0(packed11, x1));
        }
    }
#endif

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

#if RMS_FUSED
    {
        float rms_scale = 1.0f / sqrt(sum_ss / (float)ne00 + asfloat(op11));
        sum0 *= rms_scale;
        sum1 *= rms_scale;
    }
#endif

    if (local_id == 0u) {
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
