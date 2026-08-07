// flash_attn_pf.hlsli - prefill-oriented tiled Flash Attention.
//
// flash_attn_tiled.hlsl is shaped for the decode-adjacent case: 128 threads, a
// runtime head dim, and a PV pass that only keeps D_v of those threads busy
// while extracting V lanes through a dynamic float4 component index. At
// prefill sizes that leaves most of the machine idle - measured at 1.5 TFLOPS
// on a 91 TFLOPS part, and 94% of prompt time.
//
// This variant specializes on HEAD_DIM so every loop bound is a constant, runs
// 256 threads, keeps both the QK and PV passes at full occupancy, and reads V
// from LDS as plain floats. K/V share one LDS tile, so the per-tile cost is one
// extra barrier rather than a third of the LDS budget.
//
// Thread mapping (D=64, BR=16, BC=64, 256 threads):
//   QK  idx = tid + j*256 -> r = idx/BC, c = idx%BC   (4 scores/thread)
//   PV  d   = tid % D, r_j = tid/D + j*(256/D)        (4 accumulators/thread)
// Both keep the fast-varying index in the low bits, so LDS reads across a wave
// are consecutive and conflict-free.

#include "ggml_common.hlsli"

#ifndef HEAD_DIM
#error "flash_attn_pf.hlsli requires HEAD_DIM"
#endif

#define FA_D HEAD_DIM

#ifndef FA_PF_BR
#define FA_PF_BR 8
#endif

#ifndef FA_PF_BC
#if HEAD_DIM <= 64
#define FA_PF_BC 64
#else
#define FA_PF_BC 32
#endif
#endif

#ifndef FA_PF_THREADS
#define FA_PF_THREADS 256
#endif

// One float of padding rotates the bank across rows, so a wave reading
// s_kv[c][d] with c varying hits FA_D+1 strides instead of a single bank.
#define FA_PF_STRIDE (FA_D + 1)
#define FA_PF_DVEC   (FA_D / 4)

// QK: rows covered per thread, and the row group it starts from.
#define FA_PF_CGROUPS (FA_PF_THREADS / FA_PF_BC)
#define FA_PF_QK_PER  (FA_PF_BR / FA_PF_CGROUPS)

// PV: D-sized groups that fit in the threadgroup. When FA_D does not divide
// FA_PF_THREADS the tail threads sit out the PV pass (D=96: 192 of 256 active).
#define FA_PF_DG      (FA_PF_THREADS / FA_D)
#define FA_PF_ACC     (FA_PF_BR / FA_PF_DG)

// Softmax reductions: FA_PF_RSPLIT threads cooperate on each query row.
#define FA_PF_RSPLIT  (FA_PF_THREADS / FA_PF_BR)

#if (FA_PF_THREADS % FA_PF_BC) != 0
#error "FA_PF_THREADS must be a multiple of FA_PF_BC"
#endif
#if (FA_PF_BR % FA_PF_CGROUPS) != 0
#error "FA_PF_BR must be a multiple of FA_PF_CGROUPS"
#endif
#if (FA_PF_BR % FA_PF_DG) != 0
#error "FA_PF_BR must be a multiple of FA_PF_DG"
#endif
#if (FA_PF_THREADS % FA_PF_BR) != 0
#error "FA_PF_THREADS must be a multiple of FA_PF_BR"
#endif

groupshared float s_q[FA_PF_BR][FA_PF_STRIDE];
groupshared float s_kv[FA_PF_BC][FA_PF_STRIDE];
groupshared float s_scores[FA_PF_BR][FA_PF_BC];
groupshared float s_red[FA_PF_BR][FA_PF_RSPLIT];
groupshared float s_max[FA_PF_BR];
groupshared float s_sum[FA_PF_BR];
groupshared float s_corr[FA_PF_BR];
groupshared uint  s_active[FA_PF_BR];
groupshared uint  s_tile_any;

float4 fa_pf_load4(ByteAddressBuffer buf, uint byte_offset, uint elem_size) {
    if (elem_size == 4) {
        return asfloat(buf.Load4(byte_offset));
    }

    // Half rows/views may begin at a 2-byte offset. Reconstruct two packed
    // words from aligned loads instead of issuing an undefined misaligned Load2.
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 2u) * 8u;
    uint w0 = buf.Load(aligned);
    uint w1 = buf.Load(aligned + 4u);
    uint packed0 = shift == 0u ? w0 : ((w0 >> 16) | (w1 << 16));
    uint packed1;
    if (shift == 0u) {
        packed1 = w1;
    } else {
        uint w2 = buf.Load(aligned + 8u);
        packed1 = (w1 >> 16) | (w2 << 16);
    }
    uint v0 =  packed0        & 0xFFFFu;
    uint v1 = (packed0 >> 16) & 0xFFFFu;
    uint v2 =  packed1        & 0xFFFFu;
    uint v3 = (packed1 >> 16) & 0xFFFFu;

    if (elem_size == 3) {
        return float4(
            asfloat(v0 << 16),
            asfloat(v1 << 16),
            asfloat(v2 << 16),
            asfloat(v3 << 16));
    }

    return float4(
        f16_to_f32(v0),
        f16_to_f32(v1),
        f16_to_f32(v2),
        f16_to_f32(v3));
}

#define FA_PF_NEG_MAX (-3.402823466e+38f)

// Online-softmax running-max update for one query row. Called by whichever
// single thread finished that row's max fold.
void fa_pf_fold_max(uint r, float tile_max, uint q_start, uint N_queries) {
    uint active = (q_start + r < N_queries && tile_max != FA_PF_NEG_MAX) ? 1u : 0u;
    s_active[r] = active;
    if (active != 0u) {
        float old_max = s_max[r];
        float new_max = max(old_max, tile_max);
        float correction = (s_sum[r] > 0.0f) ? exp(old_max - new_max) : 0.0f;
        s_max[r] = new_max;
        s_sum[r] *= correction;
        s_corr[r] = correction;
    } else {
        s_corr[r] = 1.0f;
    }
}

WAVE_SIZE_ATTR
[numthreads(FA_PF_THREADS, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid       = gtid.x;
    const uint q_start   = gid.x * FA_PF_BR;
    const uint head_idx  = gid.y;
    const uint batch_idx = gid.z;

    const uint N_queries  = ne01;
    const uint N_kv       = ne11;
    const uint n_heads    = ne02;
    const uint n_kv_heads = ne12;
    const uint kv_head    = head_idx * n_kv_heads / n_heads;

    if (q_start >= N_queries) {
        return;
    }

    const uint src2_off = op0;
    const uint src2_nb0 = op1;
    const uint src2_nb1 = op2;
    const uint src2_nb2 = op3;
    const uint src2_nb3 = op4;
    const uint src2_es  = op5 & 0xFFu;

    const uint mask_info = op8;
    const uint has_mask  = mask_info & 1u;
    const uint has_sinks = (mask_info >> 24) & 1u;
    const uint mask_nb0  = (mask_info >> 8) & 0xFFu;
    const uint mask_es   = (mask_info >> 16) & 0xFFu;
    const uint mask_off  = op9;
    const uint mask_nb1  = op10;
    const uint mask_nb2  = op11;
    const uint mask_nb3  = op12;
    const uint mask_ne2  = op13 & 0xFFFFu;
    const uint mask_ne3  = (op13 >> 16) & 0xFFFFu;

    const float scale         = asfloat(op6);
    const float logit_softcap = asfloat(op7);
    const float max_bias      = asfloat(op14);
    const float neg_max       = -3.402823466e+38f;

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        uint n_head_log2 = (n_heads > 0u) ? (1u << firstbithigh(n_heads)) : 1u;
        float n_head_log2_f = (float)n_head_log2;
        float m0 = exp2(-max_bias * 0.5f / n_head_log2_f * 2.0f);
        float m1 = exp2(-max_bias * 0.5f / n_head_log2_f);
        if (head_idx < n_head_log2) {
            slope = pow(m0, (float)(head_idx + 1u));
        } else {
            slope = pow(m1, (float)(2u * (head_idx - n_head_log2) + 1u));
        }
    }

    // Stage the query rows once for all KV tiles.
    for (uint qidx = tid; qidx < FA_PF_BR * FA_PF_DVEC; qidx += FA_PF_THREADS) {
        uint r  = qidx / FA_PF_DVEC;
        uint dv = qidx % FA_PF_DVEC;
        uint query_idx = q_start + r;
        float4 qv = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (query_idx < N_queries) {
            uint q_base = src0_offset + query_idx * nb01 + head_idx * nb02 + batch_idx * nb03;
            qv = asfloat(src0.Load4(q_base + dv * 16u));
        }
        s_q[r][dv * 4u + 0u] = qv.x;
        s_q[r][dv * 4u + 1u] = qv.y;
        s_q[r][dv * 4u + 2u] = qv.z;
        s_q[r][dv * 4u + 3u] = qv.w;
    }

    if (tid < FA_PF_BR) {
        s_max[tid] = neg_max;
        s_sum[tid] = 0.0f;
    }

    // QK ownership: FA_PF_QK_PER (r, c) pairs, c fast-varying.
    const uint qk_c  = tid % FA_PF_BC;
    const uint qk_r0 = tid / FA_PF_BC;

    // PV ownership: one output dim, FA_PF_ACC query rows.
    const uint pv_d    = tid % FA_D;
    const uint pv_r0   = tid / FA_D;
    const bool pv_live = (tid < FA_PF_DG * FA_D);

    // Softmax reduction ownership.
    const uint red_r = tid / FA_PF_RSPLIT;
    const uint red_s = tid % FA_PF_RSPLIT;

    precise float acc[FA_PF_ACC];
    [unroll]
    for (uint ai = 0; ai < FA_PF_ACC; ++ai) {
        acc[ai] = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint tile_start = 0; tile_start < N_kv; tile_start += FA_PF_BC) {
        uint tile_size = min((uint)FA_PF_BC, N_kv - tile_start);

        // A causal or sliding-window mask leaves whole KV tiles fully masked.
        // Scanning the mask first is cheaper than staging K/V and running QK,
        // softmax and PV only to discard them. Skipping is exact: a fully
        // masked tile leaves acc and the online-softmax state untouched.
        if (has_mask != 0u) {
            if (tid == 0) {
                s_tile_any = 0u;
            }
            GroupMemoryBarrierWithGroupSync();

            uint any_local = 0u;
            uint mhb = (head_idx % mask_ne2) * mask_nb2
                     + (batch_idx % mask_ne3) * mask_nb3;
            [unroll]
            for (uint mj = 0; mj < FA_PF_QK_PER; ++mj) {
                uint mr = qk_r0 + mj * FA_PF_CGROUPS;
                uint mq = q_start + mr;
                float mv = 0.0f;
                if (mq < N_queries && qk_c < tile_size) {
                    mv = load_auto(
                        src3, mask_off + mq * mask_nb1 + mhb + (tile_start + qk_c) * mask_nb0,
                        mask_es);
                    if (!isinf(mv)) {
                        any_local = 1u;
                    }
                }
                s_scores[mr][qk_c] = mv;
            }
            if (any_local != 0u) {
                InterlockedOr(s_tile_any, 1u);
            }
            GroupMemoryBarrierWithGroupSync();

            if (s_tile_any == 0u) {
                continue;
            }
        }

        // Stage K, shared by every query row.
        for (uint kidx = tid; kidx < FA_PF_BC * FA_PF_DVEC; kidx += FA_PF_THREADS) {
            uint c  = kidx / FA_PF_DVEC;
            uint dv = kidx % FA_PF_DVEC;
            float4 kval = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (c < tile_size) {
                uint k_base = src1_offset + (tile_start + c) * nb11
                            + kv_head * nb12 + batch_idx * nb13;
                kval = fa_pf_load4(src1, k_base + dv * 4u * nb10, src1_esize);
            }
            s_kv[c][dv * 4u + 0u] = kval.x;
            s_kv[c][dv * 4u + 1u] = kval.y;
            s_kv[c][dv * 4u + 2u] = kval.z;
            s_kv[c][dv * 4u + 3u] = kval.w;
        }
        GroupMemoryBarrierWithGroupSync();

        // QK. Every thread computes FA_PF_QK_PER full-length dot products.
        // The K element is the loop-carried value so it is fetched once per d
        // and shared across the rows; s_q reads are wave-uniform broadcasts.
        precise float sc[FA_PF_QK_PER];
        float mask_local[FA_PF_QK_PER];
        uint  live_local[FA_PF_QK_PER];
        [unroll]
        for (uint j0 = 0; j0 < FA_PF_QK_PER; ++j0) {
            uint r = qk_r0 + j0 * FA_PF_CGROUPS;
            sc[j0] = 0.0f;
            mask_local[j0] = 0.0f;
            live_local[j0] = (q_start + r < N_queries && qk_c < tile_size) ? 1u : 0u;
            if (has_mask != 0u && live_local[j0] != 0u) {
                mask_local[j0] = s_scores[r][qk_c] * slope;
                if (isinf(mask_local[j0])) {
                    live_local[j0] = 0u;
                }
            }
        }

        for (uint d = 0; d < FA_D; ++d) {
            float kval = s_kv[qk_c][d];
            [unroll]
            for (uint j1 = 0; j1 < FA_PF_QK_PER; ++j1) {
                sc[j1] += s_q[qk_r0 + j1 * FA_PF_CGROUPS][d] * kval;
            }
        }

        [unroll]
        for (uint j = 0; j < FA_PF_QK_PER; ++j) {
            uint r = qk_r0 + j * FA_PF_CGROUPS;
            float dp = neg_max;
            if (live_local[j] != 0u) {
                dp = sc[j] * scale;
                if (logit_softcap != 0.0f) {
                    dp = logit_softcap * tanh(dp);
                }
                dp += mask_local[j];
            }
            sc[j] = dp;
            s_scores[r][qk_c] = dp;
        }
        GroupMemoryBarrierWithGroupSync();

        // Row max: FA_PF_RSPLIT threads per row, then a short serial fold.
        // Collapsing this to one pass (each of FA_PF_BR threads scanning all
        // FA_PF_BC columns) saves a barrier but was measured slower even on
        // Intel UHD, the most barrier-sensitive part here: 494 vs 522 t/s at
        // pp6144. Folding via WaveActiveMax/WaveActiveSum instead, which
        // removes the same two barriers with no redundant work, was a wash on
        // both parts (NVIDIA 13857 vs 13777-14191 run-to-run, Intel 522.7 vs
        // 522.4), so these reductions are not the limiter.
        {
            float m = FA_PF_NEG_MAX;
            for (uint c = red_s; c < FA_PF_BC; c += FA_PF_RSPLIT) {
                m = max(m, s_scores[red_r][c]);
            }
            s_red[red_r][red_s] = m;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < FA_PF_BR) {
            float tile_max = FA_PF_NEG_MAX;
            [unroll]
            for (uint k = 0; k < FA_PF_RSPLIT; ++k) {
                tile_max = max(tile_max, s_red[tid][k]);
            }
            fa_pf_fold_max(tid, tile_max, q_start, N_queries);
        }
        GroupMemoryBarrierWithGroupSync();

        // Exponentiate in place, reusing the scores still held in registers.
        [unroll]
        for (uint j2 = 0; j2 < FA_PF_QK_PER; ++j2) {
            uint r = qk_r0 + j2 * FA_PF_CGROUPS;
            float p = 0.0f;
            if (s_active[r] != 0u && sc[j2] != neg_max && qk_c < tile_size) {
                p = exp(sc[j2] - s_max[r]);
            }
            s_scores[r][qk_c] = p;
        }
        if (pv_live) {
            [unroll]
            for (uint ci = 0; ci < FA_PF_ACC; ++ci) {
                acc[ci] *= s_corr[pv_r0 + ci * FA_PF_DG];
            }
        }
        GroupMemoryBarrierWithGroupSync();

        {
            float s = 0.0f;
            for (uint c = red_s; c < FA_PF_BC; c += FA_PF_RSPLIT) {
                s += s_scores[red_r][c];
            }
            s_red[red_r][red_s] = s;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < FA_PF_BR && s_active[tid] != 0u) {
            float tile_sum = 0.0f;
            [unroll]
            for (uint k2 = 0; k2 < FA_PF_RSPLIT; ++k2) {
                tile_sum += s_red[tid][k2];
            }
            s_sum[tid] += tile_sum;
        }

        // Reuse the K tile slot for V. The K reads finished at the QK barrier,
        // so no extra sync is needed before overwriting it.
        for (uint vidx = tid; vidx < FA_PF_BC * FA_PF_DVEC; vidx += FA_PF_THREADS) {
            uint c  = vidx / FA_PF_DVEC;
            uint dv = vidx % FA_PF_DVEC;
            float4 vval = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (c < tile_size) {
                uint v_base = src2_off + (tile_start + c) * src2_nb1
                            + kv_head * src2_nb2 + batch_idx * src2_nb3;
                vval = fa_pf_load4(src2, v_base + dv * 4u * src2_nb0, src2_es);
            }
            s_kv[c][dv * 4u + 0u] = vval.x;
            s_kv[c][dv * 4u + 1u] = vval.y;
            s_kv[c][dv * 4u + 2u] = vval.z;
            s_kv[c][dv * 4u + 3u] = vval.w;
        }
        GroupMemoryBarrierWithGroupSync();

        // PV. One LDS V read feeds FA_PF_ACC accumulators.
        if (pv_live) {
            for (uint vc = 0; vc < tile_size; ++vc) {
                float vval = s_kv[vc][pv_d];
                [unroll]
                for (uint ai2 = 0; ai2 < FA_PF_ACC; ++ai2) {
                    acc[ai2] += s_scores[pv_r0 + ai2 * FA_PF_DG][vc] * vval;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (has_sinks != 0u) {
        if (tid < FA_PF_BR && q_start + tid < N_queries) {
            float sink_score = asfloat(src4.Load(head_idx * 4u));
            float old_max = s_max[tid];
            float new_max = max(old_max, sink_score);
            float correction = (s_sum[tid] > 0.0f) ? exp(old_max - new_max) : 0.0f;
            float sink_weight = exp(sink_score - new_max);
            s_max[tid] = new_max;
            s_sum[tid] = s_sum[tid] * correction + sink_weight;
            s_corr[tid] = correction;
        }
        GroupMemoryBarrierWithGroupSync();
        if (pv_live) {
            [unroll]
            for (uint ai3 = 0; ai3 < FA_PF_ACC; ++ai3) {
                acc[ai3] *= s_corr[pv_r0 + ai3 * FA_PF_DG];
            }
        }
    }

    if (pv_live) {
        [unroll]
        for (uint ai4 = 0; ai4 < FA_PF_ACC; ++ai4) {
            uint r = pv_r0 + ai4 * FA_PF_DG;
            uint query_idx = q_start + r;
            if (query_idx < N_queries) {
                float inv_sum = s_sum[r] > 0.0f ? (1.0f / s_sum[r]) : 0.0f;
                uint out_off = dst_offset + pv_d * nb0 + head_idx * nb1
                             + query_idx * nb2 + batch_idx * nb3;
                store_auto(dst, out_off, acc[ai4] * inv_sum, dst_esize);
            }
        }
    }
}
