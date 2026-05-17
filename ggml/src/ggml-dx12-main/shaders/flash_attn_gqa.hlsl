// flash_attn_gqa.hlsl - GQA-folded Flash Attention for DX12
//
// Variant of flash_attn.hlsl optimized for grouped-query attention. One
// workgroup per (kv_head, batch, split) processes all `gqa_ratio` query
// heads that share that KV head, loading K and V from VRAM once and reusing
// across all gqa_ratio Q-heads. For SmolVLM2 (gqa_ratio=3) this cuts KV
// bandwidth ~3x compared to the per-Q-head dispatch in flash_attn.hlsl.
//
// op_params layout (matches flash_attn.hlsl with one extension):
//   op0..op6:  same as flash_attn (src2_*, scale)
//   op7:       n_kv_heads
//   op8..op14: mask params (same as flash_attn)
//   op15:      packed: low16 = n_splits, high16 = gqa_ratio (1..MAX_GQA)
//
// Dispatch: groups_x = N_queries, groups_y = n_kv_heads,
//           groups_z = batch * n_splits

#include "ggml_common.hlsli"

#define TILE_KV    256
#define GROUP_SIZE 256
#define MAX_GQA    8

groupshared float s_scores[MAX_GQA][TILE_KV];
groupshared float s_reduce[GROUP_SIZE];

float load_mask(uint byte_offset, uint elem_stride) {
    return load_auto(src3, byte_offset, elem_stride);
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint local_id  = gtid.x;
    uint query_idx = gid.x;
    uint kv_head   = gid.y;

    uint packed     = op15;
    uint n_splits   = packed & 0xFFFFu;
    uint gqa_ratio  = (packed >> 16) & 0xFFFFu;
    if (gqa_ratio == 0) gqa_ratio = 1;

    uint split_id, batch_idx;
    if (n_splits > 1) {
        split_id  = gid.z % n_splits;
        batch_idx = gid.z / n_splits;
    } else {
        split_id  = 0;
        batch_idx = gid.z;
    }

    if (query_idx >= ne01) return;

    float scale     = asfloat(op6);
    uint  src2_off  = op0;
    uint  src2_nb0  = op1;
    uint  src2_nb1  = op2;
    uint  src2_nb2  = op3;
    uint  src2_nb3  = op4;
    uint  src2_es   = op5;
    uint  mask_info = op8;
    uint  has_mask  = mask_info & 1u;
    uint  mask_nb0  = (mask_info >> 8) & 0xFFu;
    uint  mask_es   = (mask_info >> 16) & 0xFFu;
    uint  mask_off  = op9;
    uint  mask_nb1  = op10;
    uint  mask_nb2  = op11;
    uint  mask_nb3  = op12;
    uint  mask_ne2  = op13;
    uint  mask_ne3  = op14;

    uint D    = ne00;
    uint N_kv = ne11;
    uint n_heads = ne02;

    uint kv_per_split = (N_kv + n_splits - 1) / n_splits;
    uint kv_start     = split_id * kv_per_split;
    uint kv_end       = min(kv_start + kv_per_split, N_kv);

    if (kv_start >= N_kv) {
        if (n_splits > 1 && local_id == 0) {
            uint partial_stride = (D + 2) * 4;
            [loop] for (uint g = 0; g < gqa_ratio; g++) {
                uint head_idx_g = kv_head * gqa_ratio + g;
                if (head_idx_g >= n_heads) break;
                uint partial_off = ((batch_idx * n_heads + head_idx_g) * (uint)ne01 + query_idx) * n_splits + split_id;
                partial_off *= partial_stride;
                temp.Store(partial_off,     asuint(-3.402823466e+38f));
                temp.Store(partial_off + 4, asuint(0.0f));
            }
        }
        return;
    }

    // Per-Q-head accumulators (sized for MAX_GQA so the compiler can keep
    // them in registers; only the first `gqa_ratio` slots are live).
    float global_max[MAX_GQA];
    float global_sum[MAX_GQA];
    float acc[MAX_GQA][4];
    uint  q_base[MAX_GQA];
    uint  mask_base[MAX_GQA];

    [unroll] for (uint gi = 0; gi < MAX_GQA; gi++) {
        global_max[gi] = -3.402823466e+38f;
        global_sum[gi] = 0.0f;
        [unroll] for (uint ai = 0; ai < 4; ai++) acc[gi][ai] = 0.0f;
        q_base[gi]    = 0;
        mask_base[gi] = 0;
    }

    [loop] for (uint g = 0; g < gqa_ratio; g++) {
        uint head_idx_g = kv_head * gqa_ratio + g;
        q_base[g] = src0_offset + query_idx * nb01 + head_idx_g * nb02 + batch_idx * nb03;
        if (has_mask) {
            mask_base[g] = mask_off
                         + query_idx * mask_nb1
                         + (head_idx_g % mask_ne2) * mask_nb2
                         + (batch_idx % mask_ne3) * mask_nb3;
        }
    }

    uint num_waves = GROUP_SIZE / WARP_SIZE;
    uint wave_id   = local_id / WARP_SIZE;

    for (uint tile_start = kv_start; tile_start < kv_end; tile_start += TILE_KV) {
        uint tile_end  = min(tile_start + TILE_KV, kv_end);
        uint tile_size = tile_end - tile_start;

        // Pass 1: each thread reads K[t] (and mask[t]) once and computes a
        // QK dot product for every Q-head sharing this kv_head. K loads are
        // shared across the gqa_ratio dot products (the win vs flash_attn).
        if (local_id < tile_size) {
            uint kv = tile_start + local_id;
            uint k_base = src1_offset + kv * nb11 + kv_head * nb12 + batch_idx * nb13;

            bool ctg_f32 = (src0_esize == 4 && nb00 == 4 && nb10 == 4);
#if NATIVE_FP16
            bool ctg_qf32_kf16 = (src0_esize == 4 && nb00 == 4 && src1_esize == 2 && nb10 == 2);
#endif

            [loop] for (uint g = 0; g < gqa_ratio; g++) {
                float mv = 0.0f;
                if (has_mask) mv = load_mask(mask_base[g] + kv * mask_nb0, mask_es);

                float my_score = -3.402823466e+38f;
                if (!isinf(mv)) {
                    precise float dot = 0.0f;
                    if (ctg_f32) {
                        uint d = 0;
                        for (; d + 1 < D; d += 2) {
                            uint2 qp = src0.Load2(q_base[g] + d * 4);
                            uint2 kp = src1.Load2(k_base    + d * 4);
                            dot += asfloat(qp.x) * asfloat(kp.x) + asfloat(qp.y) * asfloat(kp.y);
                        }
                        if (d < D) {
                            dot += asfloat(src0.Load(q_base[g] + d * 4)) * asfloat(src1.Load(k_base + d * 4));
                        }
#if NATIVE_FP16
                    } else if (ctg_qf32_kf16) {
                        uint d = 0;
                        for (; d + 3 < D; d += 4) {
                            uint4 qp = src0.Load4(q_base[g] + d * 4);
                            vector<float16_t,4> kh = src1.Load<vector<float16_t,4> >(k_base + d * 2);
                            dot = mad(asfloat(qp.x), (float)kh.x, mad(asfloat(qp.y), (float)kh.y,
                                  mad(asfloat(qp.z), (float)kh.z, mad(asfloat(qp.w), (float)kh.w, dot))));
                        }
                        for (; d < D; d++) {
                            dot += asfloat(src0.Load(q_base[g] + d * 4)) * load_auto(src1, k_base + d * 2, 2);
                        }
#endif
                    } else {
                        for (uint d = 0; d < D; d++) {
                            dot += load_auto(src0, q_base[g] + d * nb00, src0_esize)
                                 * load_auto(src1, k_base    + d * nb10, src1_esize);
                        }
                    }
                    my_score = dot * scale + mv;
                }
                s_scores[g][local_id] = my_score;
            }
        } else {
            [loop] for (uint g = 0; g < gqa_ratio; g++) {
                s_scores[g][local_id] = -3.402823466e+38f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Pass 2: per-Q-head softmax stats (max, then exp, then sum). All
        // threads in the workgroup execute the same g-iteration in lockstep
        // so the in-loop barriers are well-defined.
        [loop] for (uint g = 0; g < gqa_ratio; g++) {
            float local_max = (local_id < tile_size) ? s_scores[g][local_id] : -3.402823466e+38f;
            float wave_max  = WaveActiveMax(local_max);
            if (WaveIsFirstLane()) s_reduce[wave_id] = wave_max;
            GroupMemoryBarrierWithGroupSync();
            if (local_id < num_waves) {
                float v = s_reduce[local_id];
                v = WaveActiveMax(v);
                if (local_id == 0) s_reduce[0] = v;
            }
            GroupMemoryBarrierWithGroupSync();
            float tile_max = s_reduce[0];
            if (tile_max == -3.402823466e+38f) {
                if (local_id < tile_size) {
                    s_scores[g][local_id] = 0.0f;
                }
                GroupMemoryBarrierWithGroupSync();
                continue;
            }

            float new_max    = max(global_max[g], tile_max);
            float correction = (global_sum[g] > 0.0f) ? exp(global_max[g] - new_max) : 0.0f;
            [unroll] for (uint ai = 0; ai < 4; ai++) acc[g][ai] *= correction;
            global_sum[g] *= correction;
            global_max[g]  = new_max;

            if (local_id < tile_size) {
                s_scores[g][local_id] = exp(s_scores[g][local_id] - global_max[g]);
            } else {
                s_scores[g][local_id] = 0.0f;
            }
            GroupMemoryBarrierWithGroupSync();

            float local_score = s_scores[g][local_id];
            float wave_sum    = WaveActiveSum(local_score);
            if (WaveIsFirstLane()) s_reduce[wave_id] = wave_sum;
            GroupMemoryBarrierWithGroupSync();
            if (local_id < num_waves) {
                float v = s_reduce[local_id];
                v = WaveActiveSum(v);
                if (local_id == 0) s_reduce[0] = v;
            }
            GroupMemoryBarrierWithGroupSync();
            global_sum[g] += s_reduce[0];
        }

        // Pass 3: each thread reads V[d_out, t] once and multiplies by the
        // corresponding s_scores[g][t] for every g — V VRAM loads are
        // amortised across all gqa_ratio Q-heads (the second win vs flash_attn).
        for (uint a = 0; a < 4; a++) {
            uint d_out = local_id + a * GROUP_SIZE;
            if (d_out < D) {
                float tile_acc[MAX_GQA];
                [unroll] for (uint gi = 0; gi < MAX_GQA; gi++) tile_acc[gi] = 0.0f;

                for (uint t = 0; t < tile_size; t++) {
                    uint kv = tile_start + t;
                    uint v_off = src2_off + d_out * src2_nb0 + kv * src2_nb1 + kv_head * src2_nb2 + batch_idx * src2_nb3;
                    float v = load_auto(src2, v_off, src2_es);
                    [loop] for (uint g = 0; g < gqa_ratio; g++) {
                        tile_acc[g] += s_scores[g][t] * v;
                    }
                }
                [loop] for (uint g = 0; g < gqa_ratio; g++) {
                    acc[g][a] += tile_acc[g];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write outputs (one per Q-head this workgroup owns).
    if (n_splits <= 1) {
        [loop] for (uint g = 0; g < gqa_ratio; g++) {
            uint head_idx_g = kv_head * gqa_ratio + g;
            if (head_idx_g >= n_heads) break;
            float inv_sum = (global_sum[g] > 0.0f) ? (1.0f / global_sum[g]) : 0.0f;
            for (uint a = 0; a < 4; a++) {
                uint d_out = local_id + a * GROUP_SIZE;
                if (d_out < D) {
                    uint out_off = dst_offset + d_out * nb0 + head_idx_g * nb1 + query_idx * nb2 + batch_idx * nb3;
                    store_auto(dst, out_off, acc[g][a] * inv_sum, dst_esize);
                }
            }
        }
    } else {
        uint partial_stride = (D + 2) * 4;
        [loop] for (uint g = 0; g < gqa_ratio; g++) {
            uint head_idx_g = kv_head * gqa_ratio + g;
            if (head_idx_g >= n_heads) break;
            uint partial_off = ((batch_idx * n_heads + head_idx_g) * (uint)ne01 + query_idx) * n_splits + split_id;
            partial_off *= partial_stride;
            if (local_id == 0) {
                temp.Store(partial_off,     asuint(global_max[g]));
                temp.Store(partial_off + 4, asuint(global_sum[g]));
            }
            for (uint a = 0; a < 4; a++) {
                uint d_out = local_id + a * GROUP_SIZE;
                if (d_out < D) {
                    temp.Store(partial_off + 8 + d_out * 4, asuint(acc[g][a]));
                }
            }
        }
    }
}
