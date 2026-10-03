// D64/D96/D128 F16 flash attention with Vulkan-style Br16/Br32, Bc64 dataflow.
// The unsplit masked route binds packed tile metadata at the u1 root base.
#include "ggml_common.hlsli"
#include <dx/linalg.h>

using namespace dx::linalg;

#ifndef WAVE_SIZE
#error "flash_attn_pipeline requires WAVE_SIZE"
#endif
#if WAVE_SIZE != 64 && WAVE_SIZE != 32
#error "flash_attn_pipeline requires wave32 or wave64"
#endif
#ifndef FA_D
#define FA_D 128
#endif
#if FA_D != 64 && FA_D != 96 && FA_D != 128
#error "flash_attn_pipeline supports D64, D96, and D128"
#endif
#ifndef FA_PIPE_K_STRIDE
#define FA_PIPE_K_STRIDE (FA_D * 2u)
#endif
#ifndef FA_BR
#define FA_BR 16
#endif
#ifndef FA_BC
#define FA_BC 64
#endif
#ifndef FA_D96_W64_BR32
#define FA_D96_W64_BR32 0
#endif
#if FA_D96_W64_BR32 && !(FA_D == 96 && FA_BR == 32 && WAVE_SIZE == 64)
#error "FA_D96_W64_BR32 requires wave64 D96 Br32"
#endif
#if (FA_BR != 16 && FA_BR != 32) || FA_BC != 64
#error "flash_attn_pipeline requires Br16/Bc64 or Br32/Bc64"
#endif
#if FA_BR == 32 && FA_D != 64 && !(FA_D == 96 && WAVE_SIZE == 32) && !FA_D96_W64_BR32
#error "Br32 requires D64, wave32 D96, or FA_D96_W64_BR32"
#endif
#if WAVE_SIZE == 32 && !((FA_D == 128 && FA_BR == 16) || FA_D == 96 || (FA_D == 64 && FA_BR == 32))
#error "wave32 supports D128 Br16, D96 Br16/Br32, and D64 Br32"
#endif

#define TILE 16u
#define NWAVE 4u
#define THREADS (NWAVE * WAVE_SIZE)
#define KEYS_PER_LANE (FA_BC / WAVE_SIZE)
#define Q_TILES (FA_BR / TILE)
#define D_BLOCKS (FA_D / TILE)
#define D_PHASES ((FA_D + 63u) / 64u)
#define Q_VECS (FA_BR * (FA_D / 4u))
#define Q_LOAD_ITERS ((Q_VECS + THREADS - 1u) / THREADS)
#define VK_ROWS_PER_WAVE (FA_BR / NWAVE)
#define VK_SCORE_STRIDE (FA_BR + 8u)
#define Q_LEN (FA_BR * FA_D)
#define P_LEN (FA_BC * VK_SCORE_STRIDE)
#define SCORE_LEN (FA_BC * VK_SCORE_STRIDE)
#define FULL_PV_LEN (FA_BR * FA_D)

#define FA_ROW_PARTITION (FA_D == 64 || (FA_D == 96 && FA_BR == 32))
#if FA_ROW_PARTITION
#define D64_ROW_LANES (THREADS / FA_BR)
#define D64_KEYS_PER_LANE (FA_BC / D64_ROW_LANES)
#define D64_OUTPUT_PER_LANE (FA_D / D64_ROW_LANES)
#if (THREADS % FA_BR) != 0 || (FA_BC % D64_ROW_LANES) != 0 || (FA_D % D64_ROW_LANES) != 0
#error "Row partitions must divide the key and output dimensions"
#endif
#endif

#if SCORE_LEN > FULL_PV_LEN
#define WORK_LEN SCORE_LEN
#else
#define WORK_LEN FULL_PV_LEN
#endif

#if FA_ROW_PARTITION
#if Q_LEN > P_LEN
#define QP_LEN Q_LEN
#else
#define QP_LEN P_LEN
#endif
groupshared float16_t s_qp[QP_LEN];
#define s_q s_qp
#define s_p s_qp
#else
groupshared float16_t s_q[Q_LEN];
groupshared float16_t s_p[P_LEN];
#endif
groupshared float s_work[WORK_LEN];

typedef Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Wave> MatA;
typedef Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave> MatB;
typedef Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;

float16_t fa_pipeline_probability(float p) {
    const uint rounded = (uint)round(min(p, 6.103515625e-5f) * 16777216.0f);
    return asfloat16((uint16_t)(p < 6.103515625e-5f ? rounded : f32tof16(p)));
}

uint fa_pipeline_mask_class(uint q_tile, uint k_tile, uint head_idx, uint batch_idx,
                            uint n_queries, uint n_kv, uint mask_heads, uint mask_batches) {
    const uint query_tiles = (n_queries + FA_BR - 1u) / FA_BR;
    const uint key_words = (n_kv + 16u * FA_BC - 1u) / (16u * FA_BC);
    const uint mh = head_idx % mask_heads;
    const uint mb = batch_idx % mask_batches;
    const uint word_idx = ((mb * mask_heads + mh) * query_tiles + q_tile) * key_words + k_tile / 16u;
    return (temp.Load(word_idx * 4u) >> (2u * (k_tile % 16u))) & 3u;
}

WAVE_SIZE_ATTR
[numthreads(THREADS, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid = gtid.x;
    const uint wave = tid / WAVE_SIZE;
    const uint lane = tid % WAVE_SIZE;

    const uint n_queries = ne01;
    const uint n_kv = ne11;
    const uint n_heads = ne02;
    const uint n_kv_heads = ne12;
    const uint n_qgroups = (n_queries + FA_BR - 1u) / FA_BR;
    const uint q_tile = n_qgroups - 1u - gid.x;
    const uint q_start = q_tile * FA_BR;
    const uint head_idx = gid.y;
    const uint batch_idx = gid.z;
    const uint kv_head = head_idx * n_kv_heads / n_heads;
    // K and V share the broadcast batch shape on this route.
    const uint kv_batch = batch_idx / max(ne03 / ne13, 1u);

    if (q_start >= n_queries || max(op15 & 0xFFFFu, 1u) != 1u) {
        return;
    }

    const uint src2_off = op0;
    const uint src2_nb1 = op2;
    const uint src2_nb2 = op3;
    const uint src2_nb3 = op4;
    const uint mask_info = op8;
    const bool has_mask = (mask_info & 1u) != 0u;
    const bool has_sinks = ((mask_info >> 24) & 1u) != 0u;
    const uint mask_nb0 = (mask_info >> 8) & 0xFFu;
    const uint mask_es = (mask_info >> 16) & 0xFFu;
    const uint mask_off = op9;
    const uint mask_nb1 = op10;
    const uint mask_nb2 = op11;
    const uint mask_nb3 = op12;
    const uint mask_heads = op13 & 0xFFFFu;
    const uint mask_batches = op13 >> 16;
    const uint mask_hb = has_mask
        ? (head_idx % mask_heads) * mask_nb2 + (batch_idx % mask_batches) * mask_nb3
        : 0u;

    const float scale = asfloat(op6);
    const float logit_softcap = asfloat(op7);
    const float max_bias = asfloat(op14);
    const float neg_max = -3.402823466e+38f;

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const uint n_head_log2 = n_heads > 0u ? 1u << firstbithigh(n_heads) : 1u;
        const float m0 = exp2(-max_bias / (float)n_head_log2);
        const float m1 = exp2(-0.5f * max_bias / (float)n_head_log2);
        slope = head_idx < n_head_log2
            ? pow(m0, (float)(head_idx + 1u))
            : pow(m1, (float)(2u * (head_idx - n_head_log2) + 1u));
    }

    [unroll] for (uint qit = 0u; qit < Q_LOAD_ITERS; ++qit) {
        const uint qi = qit * THREADS + tid;
        const uint qr = qi / (FA_D / 4u);
        const uint d4 = (qi % (FA_D / 4u)) * 4u;
#if FA_D == 96
        if (qi < Q_VECS) {
#endif
        float4 qv = 0.0f;
        if (q_start + qr < n_queries) {
            const uint q_base = src0_offset + (q_start + qr) * nb01 + head_idx * nb02 + batch_idx * nb03;
            const uint4 qbits = src0.Load4(q_base + d4 * 4u);
            qv = float4(asfloat(qbits.x), asfloat(qbits.y), asfloat(qbits.z), asfloat(qbits.w));
        }
        s_q[qr * FA_D + d4] = (float16_t)qv.x;
        s_q[qr * FA_D + d4 + 1u] = (float16_t)qv.y;
        s_q[qr * FA_D + d4 + 2u] = (float16_t)qv.z;
        s_q[qr * FA_D + d4 + 3u] = (float16_t)qv.w;
#if FA_D == 96
        }
#endif
    }
    GroupMemoryBarrierWithGroupSync();

#if FA_BR >= 32
    MatB q_cache[Q_TILES * D_BLOCKS];
    [unroll] for (uint qt = 0u; qt < Q_TILES; ++qt) {
        [unroll] for (uint dblk = 0u; dblk < D_BLOCKS; ++dblk) {
            q_cache[qt * D_BLOCKS + dblk] =
                MatB::Load(s_q, qt * TILE * FA_D + dblk * TILE, FA_D, MatrixLayout::ColMajor);
        }
    }
#else
    MatB q_cache[D_BLOCKS];
    [unroll] for (uint dblk = 0u; dblk < D_BLOCKS; ++dblk) {
        q_cache[dblk] = MatB::Load(s_q, dblk * TILE, FA_D, MatrixLayout::ColMajor);
    }
#endif

    // Partition each wave into independent row reductions.
#if FA_ROW_PARTITION
    const uint owned_row = tid / D64_ROW_LANES;
    const uint owned_lane = tid % D64_ROW_LANES;
    float output[D64_OUTPUT_PER_LANE];
    float running_max = neg_max;
    float running_sum = 0.0f;
    [unroll] for (uint d = 0u; d < D64_OUTPUT_PER_LANE; ++d) {
        output[d] = 0.0f;
    }
#elif FA_BR == 32 && FA_D != 96
    float output[VK_ROWS_PER_WAVE];
    float running_max[VK_ROWS_PER_WAVE];
    float running_sum[VK_ROWS_PER_WAVE];
    [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
        output[r] = 0.0f;
        running_max[r] = neg_max;
        running_sum[r] = 0.0f;
    }
#else
    float4 output[VK_ROWS_PER_WAVE];
    float running_max[VK_ROWS_PER_WAVE];
    float running_sum[VK_ROWS_PER_WAVE];
    [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
        output[r] = 0.0f;
        running_max[r] = neg_max;
        running_sum[r] = 0.0f;
    }
#endif

#if FA_D == 96 && FA_BR >= 32
    MatAcc pv_tile[D_PHASES * Q_TILES];
#endif
    [loop] for (uint tile_start = 0u; tile_start < n_kv; tile_start += FA_BC) {
        const uint tile_size = min(FA_BC, n_kv - tile_start);
        uint mask_class = 2u;
        if (has_mask) {
            if (lane == 0u) {
                mask_class = fa_pipeline_mask_class(q_tile, tile_start / FA_BC, head_idx, batch_idx,
                                                       n_queries, n_kv, mask_heads, mask_batches);
            }
            mask_class = WaveReadLaneFirst(mask_class);
            if (mask_class == 1u) {
                continue;
            }
        }
        const bool apply_mask = has_mask && mask_class != 2u;

#if FA_D == 96 && FA_BR >= 32
        MatA k_cache[D_BLOCKS];
        [unroll] for (uint dblk = 0u; dblk < D_BLOCKS; ++dblk) {
            const uint k_base = src1_offset + (tile_start + wave * TILE) * nb11
                              + kv_head * nb12 + kv_batch * nb13 + dblk * TILE * 2u;
            k_cache[dblk] = MatA::Load(src1, k_base, FA_PIPE_K_STRIDE, MatrixLayout::RowMajor, 32u);
        }
        [unroll] for (uint qt = 0u; qt < Q_TILES; ++qt) {
            MatAcc score = MatAcc::Splat(0.0f);
            [unroll] for (uint dblk = 0u; dblk < D_BLOCKS; ++dblk) {
                score.MultiplyAccumulate(k_cache[dblk], q_cache[qt * D_BLOCKS + dblk]);
            }
            score.Store(s_work, wave * TILE * VK_SCORE_STRIDE + qt * TILE, VK_SCORE_STRIDE, MatrixLayout::RowMajor);
        }
#elif FA_BR == 32
        MatA k_cache[D_BLOCKS];
        [unroll] for (uint dblk = 0u; dblk < D_BLOCKS; ++dblk) {
            const uint k_base = src1_offset + (tile_start + wave * TILE) * nb11
                              + kv_head * nb12 + kv_batch * nb13 + dblk * TILE * 2u;
            k_cache[dblk] = MatA::Load(src1, k_base, FA_PIPE_K_STRIDE, MatrixLayout::RowMajor, 32u);
        }

        MatAcc score0 = MatAcc::Splat(0.0f);
        [unroll] for (uint dblk = 0u; dblk < D_BLOCKS; ++dblk) {
            score0.MultiplyAccumulate(k_cache[dblk], q_cache[dblk]);
        }
        score0.Store(s_work, wave * TILE * VK_SCORE_STRIDE, VK_SCORE_STRIDE, MatrixLayout::RowMajor);

        MatAcc score1 = MatAcc::Splat(0.0f);
        [unroll] for (uint dblk = 0u; dblk < D_BLOCKS; ++dblk) {
            score1.MultiplyAccumulate(k_cache[dblk], q_cache[D_BLOCKS + dblk]);
        }
        score1.Store(s_work, wave * TILE * VK_SCORE_STRIDE + TILE, VK_SCORE_STRIDE, MatrixLayout::RowMajor);
#else
        MatAcc score = MatAcc::Splat(0.0f);
        [unroll] for (uint dblk = 0u; dblk < D_BLOCKS; ++dblk) {
            const uint k_base = src1_offset + (tile_start + wave * TILE) * nb11
                              + kv_head * nb12 + kv_batch * nb13 + dblk * TILE * 2u;
            const MatB q = q_cache[dblk];
            const MatA k = MatA::Load(src1, k_base, FA_PIPE_K_STRIDE, MatrixLayout::RowMajor, 32u);
            score.MultiplyAccumulate(k, q);
        }
        score.Store(s_work, wave * TILE * VK_SCORE_STRIDE, VK_SCORE_STRIDE, MatrixLayout::RowMajor);
#endif
        GroupMemoryBarrierWithGroupSync();

#if FA_ROW_PARTITION
        const uint query_abs = q_start + owned_row;
        const bool row_valid = query_abs < n_queries;
        float scores[D64_KEYS_PER_LANE];
        float tile_max = neg_max;
        [unroll] for (uint i = 0u; i < D64_KEYS_PER_LANE; ++i) {
            const uint key = owned_lane * D64_KEYS_PER_LANE + i;
            float sv = neg_max;
            if (row_valid && key < tile_size) {
                sv = s_work[key * VK_SCORE_STRIDE + owned_row] * scale;
                if (logit_softcap != 0.0f) {
                    sv = logit_softcap * tanh(sv);
                }
                if (apply_mask) {
                    const float mv = load_auto(src3, mask_off + query_abs * mask_nb1 + mask_hb
                                                    + (tile_start + key) * mask_nb0, mask_es);
                    sv = isinf(mv) ? neg_max : sv + mv * slope;
                }
            }
            scores[i] = sv;
            tile_max = max(tile_max, sv);
        }
        [unroll] for (uint delta = D64_ROW_LANES / 2u; delta > 0u; delta /= 2u) {
            tile_max = max(tile_max, WaveReadLaneAt(tile_max, lane ^ delta));
        }

        float correction = 1.0f;
        if (row_valid && tile_max != neg_max) {
            const float next_max = max(running_max, tile_max);
            correction = running_sum > 0.0f ? exp(running_max - next_max) : 0.0f;
            running_max = next_max;
            running_sum *= correction;
        }

        float tile_sum = 0.0f;
        [unroll] for (uint i = 0u; i < D64_KEYS_PER_LANE; ++i) {
            const uint key = owned_lane * D64_KEYS_PER_LANE + i;
            const float p = scores[i] == neg_max ? 0.0f : exp(scores[i] - running_max);
            s_p[key * VK_SCORE_STRIDE + owned_row] = fa_pipeline_probability(p);
            tile_sum += p;
        }
        [unroll] for (uint delta = D64_ROW_LANES / 2u; delta > 0u; delta /= 2u) {
            tile_sum += WaveReadLaneAt(tile_sum, lane ^ delta);
        }
        running_sum += tile_sum;
#elif WAVE_SIZE == 64
        float scores[VK_ROWS_PER_WAVE];
        float correction[VK_ROWS_PER_WAVE];
        [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
            const uint query_row = wave * VK_ROWS_PER_WAVE + r;
            const uint query_abs = q_start + query_row;
            const bool row_valid = query_abs < n_queries;
            float sv = neg_max;
            if (row_valid && lane < tile_size) {
                sv = s_work[lane * VK_SCORE_STRIDE + query_row] * scale;
                if (logit_softcap != 0.0f) {
                    sv = logit_softcap * tanh(sv);
                }
                if (apply_mask) {
                    const float mv = load_auto(src3, mask_off + query_abs * mask_nb1 + mask_hb
                                                    + (tile_start + lane) * mask_nb0, mask_es);
                    sv = isinf(mv) ? neg_max : sv + mv * slope;
                }
            }
            scores[r] = sv;

            const float tile_max = WaveActiveMax(sv);
            correction[r] = 1.0f;
            if (row_valid && tile_max != neg_max) {
                const float next_max = max(running_max[r], tile_max);
                correction[r] = running_sum[r] > 0.0f ? exp(running_max[r] - next_max) : 0.0f;
                running_max[r] = next_max;
                running_sum[r] *= correction[r];
            }
        }

        [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
            const float p = scores[r] == neg_max ? 0.0f : exp(scores[r] - running_max[r]);
            s_p[lane * VK_SCORE_STRIDE + wave * VK_ROWS_PER_WAVE + r] = fa_pipeline_probability(p);
            running_sum[r] += WaveActiveSum(p);
        }
#else
        float scores[VK_ROWS_PER_WAVE * KEYS_PER_LANE];
        float correction[VK_ROWS_PER_WAVE];
        [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
            const uint query_row = wave * VK_ROWS_PER_WAVE + r;
            const uint query_abs = q_start + query_row;
            const bool row_valid = query_abs < n_queries;
            float local_max = neg_max;
            [unroll] for (uint i = 0u; i < KEYS_PER_LANE; ++i) {
                const uint key = lane + i * WAVE_SIZE;
                float sv = neg_max;
                if (row_valid && key < tile_size) {
                    sv = s_work[key * VK_SCORE_STRIDE + query_row] * scale;
                    if (logit_softcap != 0.0f) {
                        sv = logit_softcap * tanh(sv);
                    }
                    if (apply_mask) {
                        const float mv = load_auto(src3, mask_off + query_abs * mask_nb1 + mask_hb
                                                        + (tile_start + key) * mask_nb0, mask_es);
                        sv = isinf(mv) ? neg_max : sv + mv * slope;
                    }
                }
                scores[r * KEYS_PER_LANE + i] = sv;
                local_max = max(local_max, sv);
            }

            const float tile_max = WaveActiveMax(local_max);
            correction[r] = 1.0f;
            if (row_valid && tile_max != neg_max) {
                const float next_max = max(running_max[r], tile_max);
                correction[r] = running_sum[r] > 0.0f ? exp(running_max[r] - next_max) : 0.0f;
                running_max[r] = next_max;
                running_sum[r] *= correction[r];
            }
        }

        [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
            float local_sum = 0.0f;
            [unroll] for (uint i = 0u; i < KEYS_PER_LANE; ++i) {
                const float sv = scores[r * KEYS_PER_LANE + i];
                const float p = sv == neg_max ? 0.0f : exp(sv - running_max[r]);
                s_p[(lane + i * WAVE_SIZE) * VK_SCORE_STRIDE + wave * VK_ROWS_PER_WAVE + r] = fa_pipeline_probability(p);
                local_sum += p;
            }
            running_sum[r] += WaveActiveSum(local_sum);
        }
#endif
        GroupMemoryBarrierWithGroupSync();

#if FA_BR == 16
        MatA p_cache[FA_BC / TILE];
        [unroll] for (uint kc = 0u; kc < FA_BC / TILE; ++kc) {
            p_cache[kc] = MatA::Load(s_p, kc * TILE * VK_SCORE_STRIDE, VK_SCORE_STRIDE, MatrixLayout::ColMajor);
        }
#endif

        [unroll] for (uint phase = 0u; phase < D_PHASES; ++phase) {
            const uint dblk = phase * NWAVE + wave;
#if FA_D == 96
            if (dblk < D_BLOCKS) {
#endif
#if FA_D == 96 && FA_BR >= 32
            MatB v_cache[FA_BC / TILE];
            [unroll] for (uint kc = 0u; kc < FA_BC / TILE; ++kc) {
                const uint v_base = src2_off + (tile_start + kc * TILE) * src2_nb1
                                  + kv_head * src2_nb2 + kv_batch * src2_nb3 + dblk * TILE * 2u;
                v_cache[kc] = MatB::Load(src2, v_base, src2_nb1, MatrixLayout::RowMajor, 32u);
            }
            [unroll] for (uint qt = 0u; qt < Q_TILES; ++qt) {
                pv_tile[phase * Q_TILES + qt] = MatAcc::Splat(0.0f);
                [unroll] for (uint kc = 0u; kc < FA_BC / TILE; ++kc) {
                    const MatA p = MatA::Load(s_p, kc * TILE * VK_SCORE_STRIDE + qt * TILE, VK_SCORE_STRIDE, MatrixLayout::ColMajor);
                    pv_tile[phase * Q_TILES + qt].MultiplyAccumulate(p, v_cache[kc]);
                }
                pv_tile[phase * Q_TILES + qt].Store(s_work, qt * TILE * FA_D + dblk * TILE, FA_D, MatrixLayout::RowMajor);
            }
#elif FA_BR == 32
            MatB v_cache[FA_BC / TILE];
            [unroll] for (uint kc = 0u; kc < FA_BC / TILE; ++kc) {
                const uint v_base = src2_off + (tile_start + kc * TILE) * src2_nb1
                                  + kv_head * src2_nb2 + kv_batch * src2_nb3 + dblk * TILE * 2u;
                v_cache[kc] = MatB::Load(src2, v_base, src2_nb1, MatrixLayout::RowMajor, 32u);
            }

            MatAcc pv0 = MatAcc::Splat((float16_t)0.0f);
            [unroll] for (uint kc = 0u; kc < FA_BC / TILE; ++kc) {
                const MatA p0 = MatA::Load(s_p, kc * TILE * VK_SCORE_STRIDE, VK_SCORE_STRIDE, MatrixLayout::ColMajor);
                pv0.MultiplyAccumulate(p0, v_cache[kc]);
            }
            pv0.Store(s_work, dblk * TILE, FA_D, MatrixLayout::RowMajor);

            MatAcc pv1 = MatAcc::Splat((float16_t)0.0f);
            [unroll] for (uint kc = 0u; kc < FA_BC / TILE; ++kc) {
                const MatA p1 = MatA::Load(s_p, kc * TILE * VK_SCORE_STRIDE + TILE, VK_SCORE_STRIDE, MatrixLayout::ColMajor);
                pv1.MultiplyAccumulate(p1, v_cache[kc]);
            }
            pv1.Store(s_work, TILE * FA_D + dblk * TILE, FA_D, MatrixLayout::RowMajor);
#else
            MatAcc pv = MatAcc::Splat((float16_t)0.0f);
            [unroll] for (uint kc = 0u; kc < FA_BC / TILE; ++kc) {
                const MatA p = p_cache[kc];
                const uint v_base = src2_off + (tile_start + kc * TILE) * src2_nb1
                                  + kv_head * src2_nb2 + kv_batch * src2_nb3 + dblk * TILE * 2u;
                const MatB v = MatB::Load(src2, v_base, src2_nb1, MatrixLayout::RowMajor, 32u);
                pv.MultiplyAccumulate(p, v);
            }
            pv.Store(s_work, dblk * TILE, FA_D, MatrixLayout::RowMajor);
#endif
#if FA_D == 96
            }
#endif
        }

        GroupMemoryBarrierWithGroupSync();
#if FA_ROW_PARTITION
        [unroll] for (uint d = 0u; d < D64_OUTPUT_PER_LANE; ++d) {
            const uint dim = owned_lane * D64_OUTPUT_PER_LANE + d;
            const float value = s_work[owned_row * FA_D + dim];
            output[d] = output[d] * correction + value;
        }
#elif FA_BR == 32 && FA_D != 96
        [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
            const uint query_row = wave * VK_ROWS_PER_WAVE + r;
            const float value = s_work[query_row * FA_D + lane];
            output[r] = output[r] * correction[r] + value;
        }
#else
        if (lane < FA_D / 4u) {
            [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
                const uint query_row = wave * VK_ROWS_PER_WAVE + r;
                const uint pv_base = query_row * FA_D + lane * 4u;
                const float4 value = float4(s_work[pv_base],
                                            s_work[pv_base + 1u],
                                            s_work[pv_base + 2u],
                                            s_work[pv_base + 3u]);
                output[r] = output[r] * correction[r] + value;
            }
        }
#endif
        GroupMemoryBarrierWithGroupSync();
    }

    if (has_sinks) {
        const float sink = asfloat(src4.Load(head_idx * 4u));
#if FA_ROW_PARTITION
        if (q_start + owned_row < n_queries) {
            const float next_max = max(running_max, sink);
            const float correction = running_sum > 0.0f ? exp(running_max - next_max) : 0.0f;
            running_sum = running_sum * correction + exp(sink - next_max);
            running_max = next_max;
            [unroll] for (uint d = 0u; d < D64_OUTPUT_PER_LANE; ++d) {
                output[d] *= correction;
            }
        }
#else
        [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
            const uint query_abs = q_start + wave * VK_ROWS_PER_WAVE + r;
            if (query_abs < n_queries) {
                const float next_max = max(running_max[r], sink);
                const float correction = running_sum[r] > 0.0f ? exp(running_max[r] - next_max) : 0.0f;
                running_sum[r] = running_sum[r] * correction + exp(sink - next_max);
                running_max[r] = next_max;
                output[r] *= correction;
            }
        }
#endif
    }

#if FA_ROW_PARTITION
    const uint query_abs = q_start + owned_row;
    if (query_abs < n_queries) {
        const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        [unroll] for (uint d = 0u; d < D64_OUTPUT_PER_LANE; ++d) {
            const uint dim = owned_lane * D64_OUTPUT_PER_LANE + d;
            const uint out_off = dst_offset + dim * nb0 + head_idx * nb1 + query_abs * nb2 + batch_idx * nb3;
            store_auto(dst, out_off, output[d] * inv_sum, dst_esize);
        }
    }
#elif FA_BR == 32 && FA_D != 96
    [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
        const uint query_abs = q_start + wave * VK_ROWS_PER_WAVE + r;
        if (query_abs < n_queries) {
            const float inv_sum = running_sum[r] > 0.0f ? 1.0f / running_sum[r] : 0.0f;
            const uint out_off = dst_offset + lane * nb0 + head_idx * nb1 + query_abs * nb2 + batch_idx * nb3;
            store_auto(dst, out_off, output[r] * inv_sum, dst_esize);
        }
    }
#else
    if (lane < FA_D / 4u) {
        [unroll] for (uint r = 0u; r < VK_ROWS_PER_WAVE; ++r) {
            const uint query_abs = q_start + wave * VK_ROWS_PER_WAVE + r;
            if (query_abs < n_queries) {
                const float inv_sum = running_sum[r] > 0.0f ? 1.0f / running_sum[r] : 0.0f;
                [unroll] for (uint c = 0u; c < 4u; ++c) {
                    const uint d = lane * 4u + c;
                    const uint out_off = dst_offset + d * nb0 + head_idx * nb1 + query_abs * nb2 + batch_idx * nb3;
                    store_auto(dst, out_off, output[r][c] * inv_sum, dst_esize);
                }
            }
        }
    }
#endif
}
