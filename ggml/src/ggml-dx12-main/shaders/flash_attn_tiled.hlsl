// flash_attn_tiled.hlsl - multi-query tiled Flash Attention.
//
// One workgroup handles FA_BR query rows for one head. K/V are staged in
// 32-row tiles and reused across all query rows, avoiding the per-query K/V
// rereads in flash_attn.hlsl. Q/K/V products and online softmax accumulate in
// F32. The default is 8 rows; flash_attn_tiled16.hlsl overrides FA_BR to 16.
#include "ggml_common.hlsli"

#ifndef FA_BR
#define FA_BR 8
#endif
// One KV column per lane. Tying the column count to the wave width makes each
// query row's 32/64 scores wave-local, so the online-softmax row reductions
// become WaveActiveMax/WaveActiveSum instead of a serial scan over the tile on
// FA_BR threads while the rest of the group waits at a barrier.
// Without native fp16 the tile stays f32, so keep the original 32-column tile:
// a wave64 device would otherwise double an f32 K/V tile and lose occupancy.
#if defined(WAVE_SIZE) && defined(NATIVE_FP16)
#define FA_BC WAVE_SIZE
#else
#define FA_BC 32
#endif
#define FA_MAX_D 128
#define FA_DVEC (FA_MAX_D / 4)
#define FA_THREADS 128
#define FA_ROW_GROUPS (FA_THREADS / FA_BC)
#define FA_ROW_SETS (FA_BR / FA_ROW_GROUPS)

// Q/K/V tiles are staged as half4 when the device has native 16-bit ops. K/V
// arrive as f16 in the common case, so this is lossless for them, and it keeps
// the wider FA_BC tile within the same LDS budget the f32 tile used at
// FA_BC=32. Devices without native fp16 keep the f32 tile: they use the
// narrower FA_BC that their smaller wave implies, so the LDS budget still fits.
#if defined(NATIVE_FP16)
typedef float16_t4 fa_vec4;
#else
typedef float4 fa_vec4;
#endif

groupshared fa_vec4 s_q[FA_BR][FA_DVEC];
// Pad the K/V tile stride by one half4. The QK inner loop reads s_kv[c][dv]
// with c = tid & (FA_BC-1) and dv loop-invariant; an unpadded stride of
// FA_DVEC puts every lane on the same LDS bank (32-way conflict). Padding
// makes the bank rotate with c.
groupshared fa_vec4 s_kv[FA_BC][FA_DVEC + 1];
groupshared float  s_scores[FA_BR][FA_BC];
groupshared float  s_tile_max[FA_BR];
groupshared float  s_global_max[FA_BR];
groupshared float  s_global_sum[FA_BR];
groupshared float  s_correction[FA_BR];
groupshared uint   s_tile_active[FA_BR];
groupshared uint   s_tile_any;

float4 fa_load4(ByteAddressBuffer buf, uint byte_offset, uint elem_size) {
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

// Half-rate multiply with f32 accumulate (v_dot2_f32_f16) where available.
float fa_dot4(fa_vec4 q, fa_vec4 k, float acc) {
#if defined(NATIVE_FP16)
    float a = dot2add(q.xy, k.xy, acc);
    return dot2add(q.zw, k.zw, a);
#else
    return acc + dot(q, k);
#endif
}

float fa_lane(fa_vec4 v, uint lane) {
    if (lane == 0) return (float)v.x;
    if (lane == 1) return (float)v.y;
    if (lane == 2) return (float)v.z;
    return (float)v.w;
}

WAVE_SIZE_ATTR
[numthreads(FA_THREADS, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid       = gtid.x;
    const uint q_start   = gid.x * FA_BR;
    const uint head_idx  = gid.y;
    const uint batch_idx = gid.z;

    const uint D          = ne00;
    const uint D_v        = op5 >> 8;
    const uint dvecs      = D / 4;
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
    for (uint idx = tid; idx < FA_BR * FA_DVEC; idx += FA_THREADS) {
        uint r  = idx / FA_DVEC;
        uint dv = idx % FA_DVEC;
        uint query_idx = q_start + r;
        float4 qv = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (query_idx < N_queries && dv < dvecs) {
            uint q_base = src0_offset + query_idx * nb01 + head_idx * nb02 + batch_idx * nb03;
            qv = asfloat(src0.Load4(q_base + dv * 16u));
        }
        s_q[r][dv] = (fa_vec4)qv;
    }

    if (tid < FA_BR) {
        s_global_max[tid] = neg_max;
        s_global_sum[tid] = 0.0f;
    }

    // The PV pass gives each thread one output element, so D_v threads are busy
    // and the rest idle. When D_v is only half the group, split the KV columns
    // across two sub-groups instead and fold the partials in at the end. The
    // online-softmax correction is a scalar multiply applied to both sub-groups
    // every tile, so the split stays exact. The fold reuses s_scores, which is
    // dead by then, so this costs no extra LDS - important, because another
    // 2 KB drops the group from 3 to 2 per CU and more than undoes the gain.
#if FA_BC * 2 >= FA_THREADS
    const uint pv_split = (D_v * 2u <= FA_THREADS) ? 2u : 1u;
#else
    const uint pv_split = 1u;
#endif
    const uint pv_threads = D_v * pv_split;
    const uint pv_sub     = pv_split == 1u ? 0u : tid / D_v;
    const uint pv_shift   = pv_split == 1u ? 0u : 1u;

    precise float acc[FA_BR];
    [unroll]
    for (uint r = 0; r < FA_BR; ++r) {
        acc[r] = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint tile_start = 0; tile_start < N_kv; tile_start += FA_BC) {
        uint tile_size = min((uint)FA_BC, N_kv - tile_start);

        // A causal or sliding-window mask leaves whole KV tiles fully masked.
        // Scanning the tile's mask first is far cheaper than staging K/V and
        // running the QK, softmax and PV passes only to discard them. Skipping
        // is exact: a fully masked tile leaves acc and the online-softmax state
        // untouched.
        if (has_mask != 0u) {
            if (tid == 0) {
                s_tile_any = 0u;
            }
            GroupMemoryBarrierWithGroupSync();

            uint any_local = 0u;
            uint mhb = (head_idx % mask_ne2) * mask_nb2
                     + (batch_idx % mask_ne3) * mask_nb3;
            for (uint midx = tid; midx < FA_BR * FA_BC; midx += FA_THREADS) {
                uint mr = midx / FA_BC;
                uint mc = midx % FA_BC;
                uint mq = q_start + mr;
                float mv = 0.0f;
                if (mq < N_queries && mc < tile_size) {
                    mv = load_auto(
                        src3, mask_off + mq * mask_nb1 + mhb + (tile_start + mc) * mask_nb0,
                        mask_es);
                    if (!isinf(mv)) {
                        any_local = 1u;
                    }
                }
                // Cache for the QK pass so a kept tile reads the mask once.
                s_scores[mr][mc] = mv;
            }
            if (any_local != 0u) {
                InterlockedOr(s_tile_any, 1u);
            }
            GroupMemoryBarrierWithGroupSync();

            if (s_tile_any == 0u) {
                continue;
            }
        }

        // Stage K once, shared by all query rows.
        const uint kv_vec_count = FA_BC * dvecs;
        for (uint idx = tid; idx < kv_vec_count; idx += FA_THREADS) {
            uint c  = idx / dvecs;
            uint dv = idx % dvecs;
            uint kv = tile_start + c;
            float4 kval = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (c < tile_size) {
                uint k_base = src1_offset + kv * nb11 + kv_head * nb12 + batch_idx * nb13;
                kval = fa_load4(src1, k_base + dv * 4u * nb10, src1_esize);
            }
            s_kv[c][dv] = (fa_vec4)kval;
        }
        GroupMemoryBarrierWithGroupSync();

        // FA_ROW_GROUPS waves of FA_BC lanes cover query rows 0..FA_ROW_GROUPS-1.
        // Each thread handles the corresponding row in every row set, reusing one
        // K vector across FA_ROW_SETS rows. Lanes within a wave differ only in c,
        // so a row's scores are wave-local and reduce without LDS or barriers.
        uint c = tid & (FA_BC - 1u);
        uint base_r = tid / FA_BC;
        precise float qk_local[FA_ROW_SETS];
        float score_local[FA_ROW_SETS];
        float mask_local[FA_ROW_SETS];
        uint active_local[FA_ROW_SETS];

        [unroll]
        for (uint rr = 0; rr < FA_ROW_SETS; ++rr) {
            qk_local[rr] = 0.0f;
            score_local[rr] = neg_max;
            mask_local[rr] = 0.0f;
            active_local[rr] = 0u;
        }

        if (c < tile_size) {
            [unroll]
            for (uint rr = 0; rr < FA_ROW_SETS; ++rr) {
                uint r = base_r + rr * FA_ROW_GROUPS;
                uint query_idx = q_start + r;
                if (query_idx < N_queries) {
                    active_local[rr] = 1u;
                }
                if (has_mask != 0u && active_local[rr] != 0u) {
                    mask_local[rr] = s_scores[r][c] * slope;
                    if (isinf(mask_local[rr])) {
                        active_local[rr] = 0u;
                    }
                }
            }

            for (uint dv = 0; dv < dvecs; ++dv) {
                fa_vec4 kval = s_kv[c][dv];
                [unroll]
                for (uint rr = 0; rr < FA_ROW_SETS; ++rr) {
                    if (active_local[rr] != 0u) {
                        uint r = base_r + rr * FA_ROW_GROUPS;
                        qk_local[rr] = fa_dot4(s_q[r][dv], kval, qk_local[rr]);
                    }
                }
            }

            [unroll]
            for (uint rr = 0; rr < FA_ROW_SETS; ++rr) {
                if (active_local[rr] != 0u) {
                    float scaled = qk_local[rr] * scale;
                    if (logit_softcap != 0.0f) {
                        scaled = logit_softcap * tanh(scaled);
                    }
                    score_local[rr] = scaled + mask_local[rr];
                }
            }
        }

        [unroll]
        for (uint rr = 0; rr < FA_ROW_SETS; ++rr) {
            uint r = base_r + rr * FA_ROW_GROUPS;
            s_scores[r][c] = score_local[rr];
        }

        // Row maximum. When the runtime wave matches FA_BC every lane of a wave
        // holds a distinct column of the same row, so this is a pure wave
        // reduction. The LDS scan is kept for drivers that widen the wave past
        // the requested [WaveSize(N)].
        const bool wave_exact = (WaveGetLaneCount() == FA_BC);
        if (wave_exact) {
            [unroll]
            for (uint rr = 0; rr < FA_ROW_SETS; ++rr) {
                float m = WaveActiveMax(score_local[rr]);
                if (WaveIsFirstLane()) {
                    s_tile_max[base_r + rr * FA_ROW_GROUPS] = m;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Update each row's online-softmax maximum and rescale old output.
        if (tid < FA_BR) {
            uint r = tid;
            float tile_max = neg_max;
            if (wave_exact) {
                tile_max = s_tile_max[r];
            } else {
                for (uint c = 0; c < tile_size; ++c) {
                    tile_max = max(tile_max, s_scores[r][c]);
                }
            }

            uint active = (q_start + r < N_queries && tile_max != neg_max) ? 1u : 0u;
            s_tile_active[r] = active;
            if (active != 0u) {
                float old_max = s_global_max[r];
                float new_max = max(old_max, tile_max);
                float correction = (s_global_sum[r] > 0.0f) ? exp(old_max - new_max) : 0.0f;
                s_global_max[r] = new_max;
                s_global_sum[r] *= correction;
                s_correction[r] = correction;
            } else {
                s_correction[r] = 1.0f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < pv_threads) {
            [unroll]
            for (uint r = 0; r < FA_BR; ++r) {
                acc[r] *= s_correction[r];
            }
        }

        [unroll]
        for (uint rr = 0; rr < FA_ROW_SETS; ++rr) {
            uint r = base_r + rr * FA_ROW_GROUPS;
            float p = 0.0f;
            if (c < tile_size && s_tile_active[r] != 0u && score_local[rr] != neg_max) {
                p = exp(score_local[rr] - s_global_max[r]);
            }
            s_scores[r][c] = p;
            if (wave_exact) {
                float row_sum = WaveActiveSum(p);
                if (WaveIsFirstLane() && s_tile_active[r] != 0u) {
                    s_global_sum[r] += row_sum;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (!wave_exact) {
            if (tid < FA_BR && s_tile_active[tid] != 0u) {
                float tile_sum = 0.0f;
                for (uint c = 0; c < tile_size; ++c) {
                    tile_sum += s_scores[tid][c];
                }
                s_global_sum[tid] += tile_sum;
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // Reuse the shared K tile for V, then update all query-row outputs.
        const uint v_vec_count = FA_BC * (D_v / 4);
        for (uint idx = tid; idx < v_vec_count; idx += FA_THREADS) {
            uint c  = idx / (D_v / 4);
            uint dv = idx % (D_v / 4);
            uint kv = tile_start + c;
            float4 vval = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (c < tile_size) {
                uint v_base = src2_off + kv * src2_nb1 + kv_head * src2_nb2 + batch_idx * src2_nb3;
                vval = fa_load4(src2, v_base + dv * 4u * src2_nb0, src2_es);
            }
            s_kv[c][dv] = (fa_vec4)vval;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < pv_threads) {
            uint dv = (tid - pv_sub * D_v) / 4;
            uint lane = tid & 3u;
            // Contiguous column range per sub-group, so the loop keeps a stride
            // of one and collapses to the original loop when pv_split == 1.
            uint vc_end = (tile_size * (pv_sub + 1u)) >> pv_shift;
            for (uint vc = (tile_size * pv_sub) >> pv_shift; vc < vc_end; ++vc) {
                float vval = fa_lane(s_kv[vc][dv], lane);
                [unroll]
                for (uint r = 0; r < FA_BR; ++r) {
                    if (q_start + r < N_queries) {
                        acc[r] += s_scores[r][vc] * vval;
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Fold the second KV-column sub-group's partials into sub-group 0.
    if (pv_split > 1u) {
        if (tid >= D_v && tid < pv_threads) {
            [unroll]
            for (uint r = 0; r < FA_BR; ++r) {
                s_scores[r][tid - D_v] = acc[r];
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < D_v) {
            [unroll]
            for (uint r = 0; r < FA_BR; ++r) {
                acc[r] += s_scores[r][tid];
            }
        }
    }

    if (has_sinks != 0u) {
        if (tid < FA_BR && q_start + tid < N_queries) {
            float sink_score = asfloat(src4.Load(head_idx * 4u));
            float old_max = s_global_max[tid];
            float new_max = max(old_max, sink_score);
            float correction = (s_global_sum[tid] > 0.0f) ? exp(old_max - new_max) : 0.0f;
            float sink_weight = exp(sink_score - new_max);
            s_global_max[tid] = new_max;
            s_global_sum[tid] = s_global_sum[tid] * correction + sink_weight;
            s_correction[tid] = correction;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < D_v) {
            [unroll]
            for (uint r = 0; r < FA_BR; ++r) {
                acc[r] *= s_correction[r];
            }
        }
    }

    if (tid < D_v) {
        [unroll]
        for (uint r = 0; r < FA_BR; ++r) {
            uint query_idx = q_start + r;
            if (query_idx < N_queries) {
                float inv_sum = s_global_sum[r] > 0.0f ? (1.0f / s_global_sum[r]) : 0.0f;
                uint out_off = dst_offset + tid * nb0 + head_idx * nb1
                             + query_idx * nb2 + batch_idx * nb3;
                store_auto(dst, out_off, acc[r] * inv_sum, dst_esize);
            }
        }
    }
}
