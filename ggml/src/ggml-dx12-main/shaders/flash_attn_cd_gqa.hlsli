// flash_attn_cd_gqa.hlsli - cooperative decode FA, GQA-shared K/V (n_q == 1)
//
// Same lane layout and numerics as flash_attn_cd.hlsli, but one wave serves up
// to GQA_G Q heads that share one KV head. Each K/V row is loaded once and used
// by all of them, instead of once per Q head.
// Grid: x = query (0), y = kv_head * n_chunks + chunk, z = batch * n_splits + split.
// op15 high 16 bits = Q heads per KV head; a chunk covers up to GQA_G of them.
// Host gates: no ALiBi, mask (if any) shared by all heads (mask ne2 == 1).

#include "ggml_common.hlsli"

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif
#ifndef D_SPLIT
#define D_SPLIT 8
#endif
#ifndef GQA_G
#define GQA_G 4
#endif

#ifndef WAVE_SIZE
#error "flash_attn_cd_gqa requires a compile-time WAVE_SIZE"
#endif

#define GROUP_SIZE WAVE_SIZE
#define COLS       (WAVE_SIZE / D_SPLIT)
#define VPT        (HEAD_DIM / (D_SPLIT * 4))

#define FA_NEG_MAX (-3.402823466e+38f)

#if defined(CD_KV_Q8_0)
float4 cd_load_kv4(ByteAddressBuffer buf, uint row_base, uint vidx) {
    const uint e   = vidx * 4u;
    const uint off = row_base + (e >> 5) * 34u;
    const float d  = f16_to_f32((buf.Load(off & ~3u) >> ((off & 2u) * 8u)) & 0xFFFFu);

    const uint qo = off + 2u + (e & 31u);
    const uint a  = qo & ~3u;
    const uint sh = (qo & 3u) * 8u;
    const uint w0 = buf.Load(a);
    const uint w1 = buf.Load(a + (sh == 0u ? 0u : 4u));
    const uint w  = (sh == 0u) ? w0 : ((w0 >> sh) | (w1 << (32u - sh)));

    return float4((float)((int)(w << 24) >> 24),
                  (float)((int)(w << 16) >> 24),
                  (float)((int)(w <<  8) >> 24),
                  (float)((int) w        >> 24)) * d;
}
#else
float4 cd_load_kv4(ByteAddressBuffer buf, uint row_base, uint vidx) {
    vector<float16_t, 4> h = buf.Load<vector<float16_t, 4> >(row_base + vidx * 8u);
    return float4((float)h.x, (float)h.y, (float)h.z, (float)h.w);
}
#endif

[WaveSize(WAVE_SIZE)]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint lane    = WaveGetLaneIndex();
    const uint d_tid   = lane % D_SPLIT;
    const uint col_tid = lane / D_SPLIT;

    const uint query_idx = gid.x;

    const uint n_splits = op15 & 0xFFFFu;
    const uint gqa      = op15 >> 16;
    uint split_id, batch_idx;
    if (n_splits > 1) {
        split_id  = gid.z % n_splits;
        batch_idx = gid.z / n_splits;
    } else {
        split_id  = 0;
        batch_idx = gid.z;
    }

    if (query_idx >= ne01) return;

    const uint n_chunks  = (gqa + GQA_G - 1) / GQA_G;
    const uint kv_head   = gid.y / n_chunks;
    const uint chunk     = gid.y % n_chunks;
    const uint head0     = kv_head * gqa + chunk * GQA_G;
    const uint g_cnt     = min((uint)GQA_G, gqa - chunk * GQA_G);

    const float scale      = asfloat(op6);
    const uint  src2_off   = op0;
    const uint  src2_nb1   = op2;
    const uint  src2_nb2   = op3;
    const uint  src2_nb3   = op4;

    const uint  mask_info  = op8;
    const uint  has_mask   = mask_info & 1u;
    const uint  has_sinks  = (mask_info >> 24) & 1u;
    const uint  mask_nb0   = (mask_info >> 8) & 0xFFu;
    const uint  mask_es    = (mask_info >> 16) & 0xFFu;
    const uint  mask_off   = op9;
    const uint  mask_nb1   = op10;
    const uint  mask_nb3   = op12;
    const uint  mask_ne3   = (op13 >> 16) & 0xFFFFu;

    const float logit_softcap = asfloat(op7);

    const uint N_kv    = ne11;
    const uint n_heads = ne02;

    const uint kv_per_split = (N_kv + n_splits - 1) / n_splits;
    const uint kv_start = split_id * kv_per_split;
    const uint kv_end   = min(kv_start + kv_per_split, N_kv);
    const uint partial_stride = (HEAD_DIM + 2) * 4;

    if (kv_start >= N_kv) {
        if (n_splits > 1 && lane == 0) {
            [unroll] for (uint g = 0; g < GQA_G; g++) {
                if (g < g_cnt) {
                    uint partial_off = (((batch_idx * n_heads + head0 + g) * (uint)ne01 + query_idx) * n_splits + split_id) * partial_stride;
                    temp.Store(partial_off,     asuint(FA_NEG_MAX));
                    temp.Store(partial_off + 4, asuint(0.0f));
                }
            }
        }
        return;
    }

    uint mask_base = 0;
    if (has_mask) {
        mask_base = mask_off + query_idx * mask_nb1 + (batch_idx % mask_ne3) * mask_nb3;
    }

    // Unused head slots load head0's Q; their results are never stored.
    float4 qreg[GQA_G][VPT];
    [unroll] for (uint g = 0; g < GQA_G; g++) {
        const uint h = head0 + (g < g_cnt ? g : 0u);
        const uint q_base = src0_offset + query_idx * nb01 + h * nb02 + batch_idx * nb03;
        [unroll] for (uint qi = 0; qi < VPT; qi++) {
            uint vidx = qi * D_SPLIT + d_tid;
            uint4 qw  = src0.Load4(q_base + vidx * 16u);
            qreg[g][qi] = float4(asfloat(qw.x), asfloat(qw.y), asfloat(qw.z), asfloat(qw.w)) * scale;
        }
    }

    float  m_state[GQA_G];
    float  l_state[GQA_G];
    float4 o_state[GQA_G][VPT];
    [unroll] for (uint g = 0; g < GQA_G; g++) {
        m_state[g] = FA_NEG_MAX;
        l_state[g] = 0.0f;
        [unroll] for (uint oi = 0; oi < VPT; oi++) o_state[g][oi] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    for (uint kv = kv_start + col_tid; kv < kv_end; kv += COLS) {
        float mv = 0.0f;
        if (has_mask) {
            mv = load_auto(src3, mask_base + kv * mask_nb0, mask_es);
        }
        if (isinf(mv)) {
            continue;
        }

        const uint k_base = src1_offset + kv * nb11 + kv_head * nb12 + batch_idx * nb13;
        float4 kreg[VPT];
        [unroll] for (uint di = 0; di < VPT; di++) {
            kreg[di] = cd_load_kv4(src1, k_base, di * D_SPLIT + d_tid);
        }
        float partial[GQA_G];
        [unroll] for (uint g = 0; g < GQA_G; g++) {
            partial[g] = 0.0f;
            [unroll] for (uint di = 0; di < VPT; di++) {
                partial[g] += dot(qreg[g][di], kreg[di]);
            }
        }
        [unroll] for (uint s = D_SPLIT / 2u; s > 0u; s >>= 1) {
            [unroll] for (uint g = 0; g < GQA_G; g++) {
                partial[g] += WaveReadLaneAt(partial[g], lane ^ s);
            }
        }

        const uint v_row_base = src2_off + kv * src2_nb1 + kv_head * src2_nb2 + batch_idx * src2_nb3;
        float4 vreg[VPT];
        [unroll] for (uint vi = 0; vi < VPT; vi++) {
            vreg[vi] = cd_load_kv4(src2, v_row_base, vi * D_SPLIT + d_tid);
        }

        [unroll] for (uint g = 0; g < GQA_G; g++) {
            float score = partial[g];
            if (logit_softcap != 0.0f) {
                score = logit_softcap * tanh(score);
            }
            score += mv;
            float new_max = max(m_state[g], score);
            float corr    = (l_state[g] > 0.0f) ? exp(m_state[g] - new_max) : 0.0f;
            float p       = exp(score - new_max);
            [unroll] for (uint vi = 0; vi < VPT; vi++) {
                o_state[g][vi] = o_state[g][vi] * corr + p * vreg[vi];
            }
            l_state[g] = l_state[g] * corr + p;
            m_state[g] = new_max;
        }
    }

    [unroll] for (uint ms = D_SPLIT; ms < WAVE_SIZE; ms <<= 1) {
        [unroll] for (uint g = 0; g < GQA_G; g++) {
            float other_m = WaveReadLaneAt(m_state[g], lane ^ ms);
            float other_l = WaveReadLaneAt(l_state[g], lane ^ ms);
            float new_max = max(m_state[g], other_m);
            float a = (m_state[g] > FA_NEG_MAX) ? exp(m_state[g] - new_max) : 0.0f;
            float b = (other_m > FA_NEG_MAX) ? exp(other_m - new_max) : 0.0f;
            [unroll] for (uint mi = 0; mi < VPT; mi++) {
                float4 other_o = WaveReadLaneAt(o_state[g][mi], lane ^ ms);
                o_state[g][mi] = a * o_state[g][mi] + b * other_o;
            }
            l_state[g] = a * l_state[g] + b * other_l;
            m_state[g] = new_max;
        }
    }

    if (col_tid != 0u) return;

    [unroll] for (uint g = 0; g < GQA_G; g++) {
        if (g >= g_cnt) break;
        const uint head_idx = head0 + g;

        if (has_sinks != 0u && split_id == 0u) {
            float sink_s  = asfloat(src4.Load(head_idx * 4u));
            float new_max = max(m_state[g], sink_s);
            float corr    = (l_state[g] > 0.0f) ? exp(m_state[g] - new_max) : 0.0f;
            float p       = exp(sink_s - new_max);
            [unroll] for (uint si = 0; si < VPT; si++) o_state[g][si] *= corr;
            l_state[g] = l_state[g] * corr + p;
            m_state[g] = new_max;
        }

        if (n_splits <= 1) {
            float inv = (l_state[g] > 0.0f) ? (1.0f / l_state[g]) : 0.0f;
            uint base = dst_offset + head_idx * nb1 + query_idx * nb2 + batch_idx * nb3;
            [unroll] for (uint wi = 0; wi < VPT; wi++) {
                uint d_out = (wi * D_SPLIT + d_tid) * 4u;
                float4 ov  = o_state[g][wi] * inv;
                store_auto(dst, base + (d_out + 0u) * nb0, ov.x, dst_esize);
                store_auto(dst, base + (d_out + 1u) * nb0, ov.y, dst_esize);
                store_auto(dst, base + (d_out + 2u) * nb0, ov.z, dst_esize);
                store_auto(dst, base + (d_out + 3u) * nb0, ov.w, dst_esize);
            }
        } else {
            uint partial_off = (((batch_idx * n_heads + head_idx) * (uint)ne01 + query_idx) * n_splits + split_id) * partial_stride;
            if (lane == 0) {
                temp.Store(partial_off,     asuint(m_state[g]));
                temp.Store(partial_off + 4, asuint(l_state[g]));
            }
            [unroll] for (uint wi = 0; wi < VPT; wi++) {
                uint d_out = (wi * D_SPLIT + d_tid) * 4u;
                temp.Store4(partial_off + 8u + d_out * 4u, asuint(o_state[g][wi]));
            }
        }
    }
}
