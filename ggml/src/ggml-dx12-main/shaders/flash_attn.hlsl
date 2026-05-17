// flash_attn.hlsl - Flash Attention for DX12
//
// Computes: output = softmax(Q @ K^T * scale + mask) @ V
//
// op_params layout:
//   op0: src2_offset    op1: src2_nb0    op2: src2_nb1    op3: src2_nb2
//   op4: src2_nb3       op5: src2_esize  op6: scale(f32)  op7: n_kv_heads
//   op8: mask info      op9: mask_offset op10: mask_nb1   op11: mask_nb2
//   op12: mask_nb3      op13: mask_ne2   op14: mask_ne3

#include "ggml_common.hlsli"

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
    uint  n_kv_heads = op7;
    uint  src2_off   = op0;
    uint  src2_nb0   = op1;
    uint  src2_nb1   = op2;
    uint  src2_nb2   = op3;
    uint  src2_nb3   = op4;
    uint  src2_es    = op5;

    uint  mask_info   = op8;
    uint  has_mask    = mask_info & 1u;
    uint  mask_nb0    = (mask_info >> 8) & 0xFFu;
    uint  mask_es     = (mask_info >> 16) & 0xFFu;
    uint  mask_off    = op9;
    uint  mask_nb1    = op10;
    uint  mask_nb2    = op11;
    uint  mask_nb3    = op12;
    uint  mask_ne2    = op13;
    uint  mask_ne3    = op14;

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
                    for (uint d = 0; d < D; d++) {
                        uint q_off = q_base + d * nb00;
                        uint k_off = k_base + d * nb10;
                        dot += load_auto(src0, q_off, src0_esize) * load_auto(src1, k_off, src1_esize);
                    }
                }
                my_score = dot * scale + mv;
            }
        }
        s_scores[local_id] = my_score;
        GroupMemoryBarrierWithGroupSync();

        // Pass 2a: Find tile max using wave reduction
        float local_max = (local_id < tile_size) ? s_scores[local_id] : -3.402823466e+38f;
        float wave_max = WaveActiveMax(local_max);
        uint wave_id = local_id / WARP_SIZE;
        if (WaveIsFirstLane()) s_reduce[wave_id] = wave_max;
        GroupMemoryBarrierWithGroupSync();
        uint num_waves = GROUP_SIZE / WARP_SIZE;
        if (local_id < num_waves) {
            float v = s_reduce[local_id];
            v = WaveActiveMax(v);
            if (local_id == 0) s_reduce[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        float tile_max = s_reduce[0];
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
            s_scores[local_id] = exp(s_scores[local_id] - global_max);
        } else {
            s_scores[local_id] = 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        // Pass 2b: Sum exponentiated scores using wave reduction
        float local_score = s_scores[local_id];
        float wave_sum = WaveActiveSum(local_score);
        if (WaveIsFirstLane()) s_reduce[wave_id] = wave_sum;
        GroupMemoryBarrierWithGroupSync();
        if (local_id < num_waves) {
            float v = s_reduce[local_id];
            v = WaveActiveSum(v);
            if (local_id == 0) s_reduce[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        global_sum += s_reduce[0];

        // Pass 3: Accumulate weighted V
        for (uint ai = 0; ai < 4; ai++) {
            uint d_out = local_id + ai * GROUP_SIZE;
            if (d_out < D) {
                precise float tile_acc = 0.0f;
                for (uint t = 0; t < tile_size; t++) {
                    uint kv = tile_start + t;
                    uint v_off = src2_off + d_out * src2_nb0 + kv * src2_nb1 + kv_head * src2_nb2 + batch_idx * src2_nb3;
                    tile_acc += s_scores[t] * load_auto(src2, v_off, src2_es);
                }
                acc[ai] += tile_acc;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (n_splits <= 1) {
        // Single split: write final normalized output directly
        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
        for (uint ai = 0; ai < 4; ai++) {
            uint d_out = local_id + ai * GROUP_SIZE;
            if (d_out < D) {
                uint out_off = dst_offset + d_out * nb0 + head_idx * nb1 + query_idx * nb2 + batch_idx * nb3;
                store_auto(dst, out_off, acc[ai] * inv_sum, dst_esize);
            }
        }
    } else {
        // Split-KV: write unnormalized partial (max, sum, O[D]) to temp buffer
        // Layout: [batch][head][query][split] × (max + sum + D floats)
        uint n_heads = ne02;
        uint partial_stride = (D + 2) * 4;  // bytes per partial
        uint partial_off = ((batch_idx * n_heads + head_idx) * (uint)ne01 + query_idx) * n_splits + split_id;
        partial_off *= partial_stride;

        if (local_id == 0) {
            temp.Store(partial_off,     asuint(global_max));
            temp.Store(partial_off + 4, asuint(global_sum));
        }
        for (uint ai = 0; ai < 4; ai++) {
            uint d_out = local_id + ai * GROUP_SIZE;
            if (d_out < D) {
                temp.Store(partial_off + 8 + d_out * 4, asuint(acc[ai]));
            }
        }
    }
}
