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

#include "ggml_common.hlsli"
#include "rope_yarn.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE       64
#endif
#define QK8_0            32
#define Q8_0_BSIZE       34
#define VALUES_PER_LANE   8

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
groupshared float wave_sums[2][GROUP_SIZE / WAVE_SIZE];
#endif

uint read_u32_unaligned(uint byte_offset) {
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = src0.Load(aligned);
    if (shift == 0u) {
        return lo;
    }
    uint hi = src0.Load(aligned + 4u);
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

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint k = local_id * VALUES_PER_LANE; k < ne00;
         k += GROUP_SIZE * VALUES_PER_LANE) {
        uint block = k / QK8_0;
        uint element = k & (QK8_0 - 1u);
        uint block_offset0 = src0_row0 + block * Q8_0_BSIZE;
        uint block_offset1 = src0_row1 + block * Q8_0_BSIZE;
        float4 x0 = asfloat(src1.Load4(src1_row + k * 4u));
        float4 x1 = asfloat(src1.Load4(src1_row + (k + 4u) * 4u));

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

    float sum0 = WaveActiveSum(acc0);
    float sum1 = WaveActiveSum(acc1);

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
    if (WaveIsFirstLane()) {
        uint wave = local_id / WAVE_SIZE;
        wave_sums[0][wave] = sum0;
        wave_sums[1][wave] = sum1;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_id == 0u) {
        sum0 = 0.0f;
        sum1 = 0.0f;
        [unroll]
        for (uint wave = 0; wave < GROUP_SIZE / WAVE_SIZE; ++wave) {
            sum0 += wave_sums[0][wave];
            sum1 += wave_sums[1][wave];
        }
    }
#endif

    if (local_id == 0u) {
        uint cache_esize = op7;
        uint cache_nb1   = op15;
        uint local_row0  = row0 - region_base;
        if (rope) {
            uint pair_in_head = (row0 % op10) / 2u;
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
