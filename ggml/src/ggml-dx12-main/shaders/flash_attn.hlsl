// flash_attn.hlsl - Flash Attention for DX12
//
// Computes: output = softmax(QK^T * scale + softcap(tanh) + slope*mask + sinks) @ V
//
// op_params layout (16 dwords, FA-specific):
//   op0: src2_offset    op1: src2_nb0    op2: src2_nb1    op3: src2_nb2
//   op4: src2_nb3       op5: src2_esize  op6: scale(f32)
//   op7: logit_softcap(f32, 0 = softcap off)
//   op8: mask info (bit 0 = has_mask, bits 8-15 = mask_nb0, bits 16-23 = mask_es, bit 24 = has_sinks)
//   op9: mask_offset op10: mask_nb1   op11: mask_nb2   op12: mask_nb3
//   op13: packed mask_ne2 (low 16) | mask_ne3 (high 16)
//   op14: max_bias (f32, 0 = ALiBi disabled). Shader derives m0, m1, n_head_log2
//         from max_bias and n_head (= ne02) to match ggml-cpu reference.
//   op15: packed split_kv (low 16) | gqa_ratio (high 16)
//
// n_kv_heads is read directly from ne12 (= src1->ne[2]).
//
// src4 binding (when has_sinks): F32 vector of length n_heads. Each
// workgroup reads sinks[head_idx] once at the tail of the KV loop and
// applies a single online-softmax update with no V contribution.

#include "ggml_common.hlsli"

// KV quant variants: set MMID_<TYPE> + KV_QUANT, then pull in the shared
// per-element dequant for K/V via quant_dequant.hlsli.
#if defined(KV_Q4_0)
#define MMID_Q4_0
#define KV_QUANT
#elif defined(KV_Q4_1)
#define MMID_Q4_1
#define KV_QUANT
#elif defined(KV_Q5_0)
#define MMID_Q5_0
#define KV_QUANT
#elif defined(KV_Q5_1)
#define MMID_Q5_1
#define KV_QUANT
#elif defined(KV_Q8_0)
#define MMID_Q8_0
#define KV_QUANT
#elif defined(KV_IQ4_NL)
#define MMID_IQ4_NL
#define KV_QUANT
#endif

#ifdef KV_QUANT
#include "quant_dequant.hlsli"
#endif

#ifndef TILE_KV
#define TILE_KV 256
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 256
#endif

groupshared float s_scores[TILE_KV];
groupshared float s_reduce[GROUP_SIZE];

float load_mask(uint byte_offset, uint elem_stride) {
    return load_auto(src3, byte_offset, elem_stride);
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint local_id = gtid.x;
    uint query_idx = gid.x;
    uint head_idx  = gid.y;

    // Split-KV: decompose gid.z into split_id and batch_idx
    uint n_splits = op15 & 0xFFFFu;  // GQA-folded FA packs gqa_ratio in high 16 bits
    uint split_id, batch_idx;
    if (n_splits > 1) {
        split_id  = gid.z % n_splits;
        batch_idx = gid.z / n_splits;
    } else {
        split_id  = 0;
        batch_idx = gid.z;
    }

    if (query_idx >= ne01) return;

    float scale      = asfloat(op6);
    uint  n_kv_heads = ne12;
    uint  src2_off   = op0;
    uint  src2_nb0   = op1;
    uint  src2_nb1   = op2;
    uint  src2_nb2   = op3;
    uint  src2_nb3   = op4;
    // op5: low 8 bits = src2 element size, high 24 bits = D_v (= hsv).
    // Equal to D when hsk == hsv (most models); only diverges for MLA-style
    // attention where the K head_dim exceeds the V head_dim.
    uint  src2_es    = op5 & 0xFFu;
    uint  D_v        = op5 >> 8;

    uint  mask_info   = op8;
    uint  has_mask    = mask_info & 1u;
    uint  has_sinks   = (mask_info >> 24) & 1u;
    uint  mask_nb0    = (mask_info >> 8) & 0xFFu;
    uint  mask_es     = (mask_info >> 16) & 0xFFu;
    uint  mask_off    = op9;
    uint  mask_nb1    = op10;
    uint  mask_nb2    = op11;
    uint  mask_nb3    = op12;
    uint  mask_ne2    = op13 & 0xFFFFu;
    uint  mask_ne3    = (op13 >> 16) & 0xFFFFu;

    // ALiBi / softcap parameters (0 = features disabled).
    float max_bias      = asfloat(op14);
    float logit_softcap = asfloat(op7);

    // Per-head ALiBi slope. Matches ggml-cpu reference:
    //   slope = h < n_head_log2 ? pow(m0, h+1) : pow(m1, 2*(h-n_head_log2)+1)
    //   m0 = 2^(-max_bias / n_head_log2)
    //   m1 = 2^(-max_bias / 2 / n_head_log2)
    //   n_head_log2 = largest power of two <= n_head (= ne02).
    float slope = 1.0f;
    if (max_bias > 0.0f) {
        uint n_head      = ne02;
        uint n_head_log2 = (n_head > 0u) ? (1u << firstbithigh(n_head)) : 1u;
        float n_head_log2_f = (float)n_head_log2;
        float m0 = exp2(-max_bias        / n_head_log2_f);
        float m1 = exp2(-max_bias * 0.5f / n_head_log2_f);
        if (head_idx < n_head_log2) {
            slope = pow(m0, (float)(head_idx + 1u));
        } else {
            slope = pow(m1, (float)(2u * (head_idx - n_head_log2) + 1u));
        }
    }

    uint D    = ne00;
    uint N_kv = ne11;
    uint kv_head = head_idx * n_kv_heads / ne02;

    // Split-KV: compute this group's KV range.
    // The inner loop already handles partial last tiles via min(tile_end, kv_end),
    // so do NOT round kv_per_split up to TILE_KV — that would cause most splits
    // to early-exit when N_kv is small (e.g., decode at ~330 tokens with 11 splits
    // would leave only 2 splits with work).
    uint kv_per_split = (N_kv + n_splits - 1) / n_splits;
    uint kv_start = split_id * kv_per_split;
    uint kv_end   = min(kv_start + kv_per_split, N_kv);
    if (kv_start >= N_kv) {
        // This split has no work — can happen with rounding
        if (n_splits > 1 && local_id == 0) {
            // Write zero partial to temp buffer
            uint n_heads = ne02;
            uint partial_stride = (D + 2) * 4;  // bytes per partial: D floats + max + sum
            uint partial_off = ((batch_idx * n_heads + head_idx) * (uint)ne01 + query_idx) * n_splits + split_id;
            partial_off *= partial_stride;
            temp.Store(partial_off, asuint(-3.402823466e+38f));  // max
            temp.Store(partial_off + 4, asuint(0.0f));            // sum
        }
        return;
    }

    uint mask_base = 0;
    if (has_mask) {
        mask_base = mask_off
                  + query_idx * mask_nb1
                  + (head_idx % mask_ne2) * mask_nb2
                  + (batch_idx % mask_ne3) * mask_nb3;
    }

    uint q_base = src0_offset + query_idx * nb01 + head_idx * nb02 + batch_idx * nb03;

    float global_max = -3.402823466e+38f;
    float global_sum = 0.0f;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint tile_start = kv_start; tile_start < kv_end; tile_start += TILE_KV) {
        uint tile_end = min(tile_start + TILE_KV, kv_end);
        uint tile_size = tile_end - tile_start;

        // Pass 1: Each thread computes one Q·K dot product
        float my_score = -3.402823466e+38f;
        if (local_id < tile_size) {
            uint kv = tile_start + local_id;

            float mv = 0.0f;
            if (has_mask) {
                mv = load_mask(mask_base + kv * mask_nb0, mask_es);
                mv *= slope;  // ALiBi: scale mask by per-head slope (no-op when slope==1)
            }

            if (!isinf(mv)) {
                precise float dot = 0.0f;
                uint k_base = src1_offset + kv * nb11 + kv_head * nb12 + batch_idx * nb13;

                // Vectorized QK dot product with Load2
                if (src0_esize == 4 && nb00 == 4 && nb10 == 4) {
                    // Both Q and K are contiguous F32
                    uint d = 0;
                    for (; d + 1 < D; d += 2) {
                        uint2 qp = src0.Load2(q_base + d * 4);
                        uint2 kp = src1.Load2(k_base + d * 4);
                        dot += asfloat(qp.x) * asfloat(kp.x) + asfloat(qp.y) * asfloat(kp.y);
                    }
                    if (d < D) {
                        dot += asfloat(src0.Load(q_base + d * 4)) * asfloat(src1.Load(k_base + d * 4));
                    }
#if NATIVE_FP16
                } else if (src0_esize == 4 && nb00 == 4 && src1_esize == 2 && nb10 == 2) {
                    // Q = contiguous F32, K = contiguous F16 (typical KV-cache case).
                    // Native fp16 path: load 4 K halves at once via templated Load.
                    uint d = 0;
                    for (; d + 3 < D; d += 4) {
                        uint4 qp = src0.Load4(q_base + d * 4);
                        vector<float16_t,4> kh = src1.Load<vector<float16_t,4> >(k_base + d * 2);
                        dot = mad(asfloat(qp.x), (float)kh.x, mad(asfloat(qp.y), (float)kh.y,
                              mad(asfloat(qp.z), (float)kh.z, mad(asfloat(qp.w), (float)kh.w, dot))));
                    }
                    for (; d < D; d++) {
                        dot += asfloat(src0.Load(q_base + d * 4)) * load_auto(src1, k_base + d * 2, 2);
                    }
#endif
                } else {
#ifdef KV_QUANT
                    // Per-element dequant: K row starts at k_base, element d
                    // decoded via mmid_dequant (handles block decode internally).
                    // Q is F32 contiguous in all tested cases (src0_esize == 4).
                    for (uint d = 0; d < D; d++) {
                        float q = asfloat(src0.Load(q_base + d * 4));
                        float k = mmid_dequant(src1, k_base, d);
                        dot = mad(q, k, dot);
                    }
#else
                    for (uint d = 0; d < D; d++) {
                        uint q_off = q_base + d * nb00;
                        uint k_off = k_base + d * nb10;
                        dot += load_auto(src0, q_off, src0_esize) * load_auto(src1, k_off, src1_esize);
                    }
#endif
                }
                // CPU: s = s*scale; if (softcap!=0) s = softcap*tanh(s); s += mv.
                // Host has already folded scale/=softcap so the inner expr stays in range.
                float scaled = dot * scale;
                if (logit_softcap != 0.0f) {
                    scaled = logit_softcap * tanh(scaled);
                }
                my_score = scaled + mv;
            }
        }
        // Pass 2a: Find tile max using wave reduction
        float local_max = (local_id < tile_size) ? my_score : -3.402823466e+38f;
        float wave_max = WaveActiveMax(local_max);
#if defined(WAVE_SIZE) && (GROUP_SIZE == WAVE_SIZE)
        float tile_max = WaveReadLaneFirst(wave_max);
#else
        uint lane_count = WaveGetLaneCount();
        uint wave_id = local_id / lane_count;
        uint num_waves = (GROUP_SIZE + lane_count - 1) / lane_count;
        if (WaveIsFirstLane()) s_reduce[wave_id] = wave_max;
        GroupMemoryBarrierWithGroupSync();
        if (local_id == 0) {
            float v = s_reduce[0];
            for (uint w = 1; w < num_waves; ++w) v = max(v, s_reduce[w]);
            s_reduce[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        float tile_max = s_reduce[0];
#endif
        if (tile_max == -3.402823466e+38f) {
            continue;
        }

        // Online softmax correction
        float new_max = max(global_max, tile_max);
        float correction = (global_sum > 0.0f) ? exp(global_max - new_max) : 0.0f;
        [unroll] for (uint a = 0; a < 4; a++) acc[a] *= correction;
        global_sum *= correction;
        global_max = new_max;

        // Exponentiate scores
        if (local_id < tile_size) {
            s_scores[local_id] = exp(my_score - global_max);
        } else {
            s_scores[local_id] = 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        // Pass 2b: Sum exponentiated scores using wave reduction
        float local_score = s_scores[local_id];
        float wave_sum = WaveActiveSum(local_score);
#if defined(WAVE_SIZE) && (GROUP_SIZE == WAVE_SIZE)
        global_sum += WaveReadLaneFirst(wave_sum);
#else
        if (WaveIsFirstLane()) s_reduce[wave_id] = wave_sum;
        GroupMemoryBarrierWithGroupSync();
        if (local_id == 0) {
            float v = s_reduce[0];
            for (uint w = 1; w < num_waves; ++w) v += s_reduce[w];
            s_reduce[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        global_sum += s_reduce[0];
#endif

        // Pass 3: Accumulate weighted V
        for (uint ai = 0; ai < 4; ai++) {
            uint d_out = local_id + ai * GROUP_SIZE;
            if (d_out < D_v) {
                precise float tile_acc = 0.0f;
                for (uint t = 0; t < tile_size; t++) {
                    uint kv = tile_start + t;
#ifdef KV_QUANT
                    // V row (head_dim contiguous along dim 0). Per-token row
                    // base is fixed; element d_out is decoded out of the block.
                    uint v_row_base = src2_off + kv * src2_nb1 + kv_head * src2_nb2 + batch_idx * src2_nb3;
                    tile_acc += s_scores[t] * mmid_dequant(src2, v_row_base, d_out);
#else
                    uint v_off = src2_off + d_out * src2_nb0 + kv * src2_nb1 + kv_head * src2_nb2 + batch_idx * src2_nb3;
                    tile_acc += s_scores[t] * load_auto(src2, v_off, src2_es);
#endif
                }
                acc[ai] += tile_acc;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Attention sinks tail: one extra online-softmax step with score = sinks[head_idx]
    // and zero V contribution. Apply only on the first KV split so the reduce
    // step sees exactly one sink contribution per (batch, head, query). Matches
    // ggml-cpu reference (apply_only_on_first_kv_chunk).
    if (has_sinks != 0u && split_id == 0u) {
        float sink_s = asfloat(src4.Load(head_idx * 4u));

        float new_max = max(global_max, sink_s);
        float correction = (global_sum > 0.0f) ? exp(global_max - new_max) : 0.0f;
        float vs = exp(sink_s - new_max);
        [unroll] for (uint a = 0; a < 4; a++) acc[a] *= correction;
        global_sum = global_sum * correction + vs;
        global_max = new_max;
    }

    if (n_splits <= 1) {
        // Single split: write final normalized output directly
        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
        for (uint ai = 0; ai < 4; ai++) {
            uint d_out = local_id + ai * GROUP_SIZE;
            if (d_out < D_v) {
                uint out_off = dst_offset + d_out * nb0 + head_idx * nb1 + query_idx * nb2 + batch_idx * nb3;
                store_auto(dst, out_off, acc[ai] * inv_sum, dst_esize);
            }
        }
    } else {
        // Split-KV: write unnormalized partial (max, sum, O[D_v]) to temp buffer
        // Layout: [batch][head][query][split] × (max + sum + D_v floats)
        uint n_heads = ne02;
        uint partial_stride = (D_v + 2) * 4;  // bytes per partial
        uint partial_off = ((batch_idx * n_heads + head_idx) * (uint)ne01 + query_idx) * n_splits + split_id;
        partial_off *= partial_stride;

        if (local_id == 0) {
            temp.Store(partial_off,     asuint(global_max));
            temp.Store(partial_off + 4, asuint(global_sum));
        }
        for (uint ai = 0; ai < 4; ai++) {
            uint d_out = local_id + ai * GROUP_SIZE;
            if (d_out < D_v) {
                temp.Store(partial_off + 8 + d_out * 4, asuint(acc[ai]));
            }
        }
    }
}
