// Combined Q/K/V projection matvec for a single-token (M=1) decode step,
// F16 weights, AMD wave64. One dispatch produces all three projections.
//
// The three weight matrices are contiguous in one buffer (Wq | Wk | Wv, host-
// verified), so the global output row indexes them directly. The region (Q/K/V)
// is chosen by the row vs the op0/op2 boundaries and is uniform across a group:
//   Q rows   [0, q_rows)              rotate, write the rope output (dst/u0)
//   K rows   [q_rows, q_rows+k_rows)  rotate, scatter into the KV cache (temp/u1)
//   V rows   [q_rows+k_rows, ne0)     no rope, scatter into the KV cache (temp/u1)
// Decode math (accumulate4) and rotation (mmv_rope_pair) reuse the standalone
// mul_mat_vec_f16_wave64.hlsl path exactly, so per-row results are bit-identical
// to the three separate fused matvecs this replaces.
//
// op-param map (see host dx12_graph_compute QKV-shared dispatch):
//   op0  q_rows                 op2  k_rows            ne0 = total rows (guard)
//   rope (shared Q & K, consumed by mmv_rope_pair; positions src2, ff src4):
//     op1 n_dims  op3 freq_base  op4 freq_scale  op5 ext_factor
//     op6 attn_factor  op8 corr_low  op9 corr_high  op10 head_dim  op11 has_ff
//   op7  KV cache element size (== cache nb0)   op12 Q rope-out element stride
//   op13 K cache base byte offset   op14 V cache base byte offset
//   op15 KV cache row stride (nb1)
//   src0/t0 Wq|Wk|Wv   src1/t1 activation   src3/t3 K indices   src5/t5 V indices
//   dst/u0 Q rope output   temp/u1 KV cache

#include "ggml_common.hlsli"
#include "rope_yarn.hlsli"

#define GROUP_SIZE 64

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
groupshared float wave_sums[2][GROUP_SIZE / WAVE_SIZE];
#endif

void accumulate4(uint weight_offset, uint activation_offset, inout float acc) {
#if NATIVE_FP16
    vector<float16_t, 4> w = src0.Load<vector<float16_t, 4> >(weight_offset);
    float4 x = asfloat(src1.Load4(activation_offset));
    acc = mad((float)w.x, x.x,
          mad((float)w.y, x.y,
          mad((float)w.z, x.z,
          mad((float)w.w, x.w, acc))));
#else
    uint2 packed_w = src0.Load2(weight_offset);
    float4 x = asfloat(src1.Load4(activation_offset));
    float w0 = f16_to_f32(packed_w.x & 0xffffu);
    float w1 = f16_to_f32(packed_w.x >> 16);
    float w2 = f16_to_f32(packed_w.y & 0xffffu);
    float w3 = f16_to_f32(packed_w.y >> 16);
    acc = mad(w0, x.x, mad(w1, x.y, mad(w2, x.z, mad(w3, x.w, acc))));
#endif
}

float load_f16_scalar(uint byte_offset) {
    uint word = src0.Load(byte_offset & ~3u);
    return f16_to_f32((word >> ((byte_offset & 2u) * 8u)) & 0xffffu);
}

WAVE_SIZE_ATTR
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row = group_id.x;
    if (row >= ne0) {
        return;
    }

    uint q_rows  = op0;
    uint qk_rows = op0 + op2;

    // Region (uniform across the group): 0 = Q, 1 = K, 2 = V.
    uint region;
    uint region_base;
    if (row < q_rows)       { region = 0u; region_base = 0u; }
    else if (row < qk_rows) { region = 1u; region_base = q_rows; }
    else                    { region = 2u; region_base = qk_rows; }
    uint local_row = row - region_base;
    bool rope = (region != 2u);

    uint src0_row   = src0_offset + row * nb01;
    uint src0_row_p = src0_offset + (row ^ 1u) * nb01;
    uint src1_row   = src1_offset;

    float acc   = 0.0f;
    float acc_p = 0.0f;
    uint k = local_id * 4u;
    const uint stride = GROUP_SIZE * 4u;

    for (; k + 3u < ne00; k += stride) {
        accumulate4(src0_row + k * 2u, src1_row + k * 4u, acc);
        if (rope) {
            accumulate4(src0_row_p + k * 2u, src1_row + k * 4u, acc_p);
        }
    }
    for (; k < ne00; ++k) {
        float x = asfloat(src1.Load(src1_row + k * 4u));
        acc = mad(load_f16_scalar(src0_row + k * 2u), x, acc);
        if (rope) {
            acc_p = mad(load_f16_scalar(src0_row_p + k * 2u), x, acc_p);
        }
    }

    float sum   = WaveActiveSum(acc);
    float sum_p = 0.0f;
    if (rope) {
        sum_p = WaveActiveSum(acc_p);
    }

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
    if (WaveIsFirstLane()) {
        wave_sums[0][local_id / WAVE_SIZE] = sum;
        wave_sums[1][local_id / WAVE_SIZE] = sum_p;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_id == 0u) {
        sum   = 0.0f;
        sum_p = 0.0f;
        [unroll]
        for (uint wave = 0; wave < GROUP_SIZE / WAVE_SIZE; ++wave) {
            sum   += wave_sums[0][wave];
            sum_p += wave_sums[1][wave];
        }
    }
#endif

    if (local_id == 0u) {
        uint cache_esize = op7;
        uint cache_nb1   = op15;
        if (rope) {
            uint row0 = row & ~1u;
            float sum0 = (row == row0) ? sum   : sum_p;
            float sum1 = (row == row0) ? sum_p : sum;
            uint pair_in_head = (row0 % op10) / 2u;
            float out0, out1;
            mmv_rope_pair(pair_in_head, sum0, sum1, out0, out1);
            float outv = (row == row0) ? out0 : out1;
            if (region == 0u) {
                store_auto(dst, dst_offset + local_row * op12, outv, dst_esize);
            } else {
                int row_idx = asint(src3.Load(0));
                uint off = op13 + local_row * cache_esize + (uint)row_idx * cache_nb1;
                store_auto(temp, off, outv, cache_esize);
            }
        } else {
            int row_idx = asint(src5.Load(0));
            uint off = op14 + local_row * cache_esize + (uint)row_idx * cache_nb1;
            store_auto(temp, off, sum, cache_esize);
        }
    }
}
