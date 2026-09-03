// flash_attn_cd.hlsli - Cooperative decode Flash Attention (single query, n_q == 1)
//
// A D-split cooperative variant of flash_attn.hlsl for the decode hot path.
// Instead of one thread per KV position (scalar Q.K dot) plus one thread per
// output dim, a single-wave workgroup is organised as D_SPLIT x COLS:
//   d_tid  = lane % D_SPLIT   -> owns HEAD_DIM / D_SPLIT of the head dimension
//   col_tid= lane / D_SPLIT   -> owns one of COLS parallel KV columns
// D_SPLIT threads cooperate on each KV column's dot product (coalesced,
// vectorised f16 loads) and reduce it with a wave butterfly. Each thread keeps
// a private online-softmax partial over the KV columns it owns; the COLS
// partials are merged once with a second butterfly at the end. No LDS, no
// GroupMemoryBarrier - all reductions are wave shuffles within one wave.
//
// Mirrors flash_attn.hlsl's op_params layout and scale / softcap / mask / ALiBi
// / sink / split-KV semantics exactly, so numerics match the scalar shader and
// the shared flash_attn_reduce combines its split partials unchanged.
//
// Compile-time config (set by the thin wrappers):
//   HEAD_DIM : head size (HSK == HSV), must be a multiple of D_SPLIT*4
//   D_SPLIT  : head-dim lanes per KV column (default 8)
//   CD_KV_Q8_0 : K/V cache is Q8_0 instead of F16
// Requires NATIVE_FP16 for the vectorised f16 K/V path; the host only routes
// here on fp16-capable devices with contiguous F16 (or Q8_0) K/V and F32 Q.

#include "ggml_common.hlsli"

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif
#ifndef D_SPLIT
#define D_SPLIT 8
#endif

#ifndef WAVE_SIZE
#error "flash_attn_cd requires a compile-time WAVE_SIZE"
#endif

#define GROUP_SIZE WAVE_SIZE
#define COLS       (WAVE_SIZE / D_SPLIT)     // KV columns processed per wave step
#define VPT        (HEAD_DIM / (D_SPLIT * 4)) // f16-vec4 loads per thread and dim

#define FA_NEG_MAX (-3.402823466e+38f)

// K/V tile fetch: four consecutive head-dim elements starting at vidx*4.
#if defined(CD_KV_Q8_0)
// Q8_0 = 34-byte blocks of 32 elements (f16 scale, then 32 int8). vidx*4 is
// 4-aligned and 32 is a multiple of 4, so the four elements never straddle a
// block: one scale fetch plus one 32-bit quant fetch covers all four. The
// generic per-element mmid_dequant costs 8 loads for the same data.
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

float cd_load_mask(uint byte_offset, uint elem_stride) {
    return load_auto(src3, byte_offset, elem_stride);
}

[WaveSize(WAVE_SIZE)]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint lane    = WaveGetLaneIndex();
    const uint d_tid   = lane % D_SPLIT;
    const uint col_tid = lane / D_SPLIT;

    const uint query_idx = gid.x;   // 0 for decode
    const uint head_idx  = gid.y;

    const uint n_splits = op15 & 0xFFFFu;
    uint split_id, batch_idx;
    if (n_splits > 1) {
        split_id  = gid.z % n_splits;
        batch_idx = gid.z / n_splits;
    } else {
        split_id  = 0;
        batch_idx = gid.z;
    }

    if (query_idx >= ne01) return;

    const float scale      = asfloat(op6);
    const uint  n_kv_heads = ne12;
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
    const uint  mask_nb2   = op11;
    const uint  mask_nb3   = op12;
    const uint  mask_ne2   = op13 & 0xFFFFu;
    const uint  mask_ne3   = (op13 >> 16) & 0xFFFFu;

    const float max_bias      = asfloat(op14);
    const float logit_softcap = asfloat(op7);

    // Per-head ALiBi slope (matches flash_attn.hlsl / ggml-cpu reference).
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

    const uint N_kv    = ne11;
    const uint n_heads = ne02;
    const uint kv_head = head_idx * n_kv_heads / n_heads;

    const uint kv_per_split = (N_kv + n_splits - 1) / n_splits;
    const uint kv_start = split_id * kv_per_split;
    const uint kv_end   = min(kv_start + kv_per_split, N_kv);

    if (kv_start >= N_kv) {
        // Empty split: write a zero partial so the reduce sees a well-formed slot.
        if (n_splits > 1 && lane == 0) {
            uint partial_stride = (HEAD_DIM + 2) * 4;
            uint partial_off = ((batch_idx * n_heads + head_idx) * (uint)ne01 + query_idx) * n_splits + split_id;
            partial_off *= partial_stride;
            temp.Store(partial_off,     asuint(FA_NEG_MAX));
            temp.Store(partial_off + 4, asuint(0.0f));
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

    const uint q_base = src0_offset + query_idx * nb01 + head_idx * nb02 + batch_idx * nb03;

    // Load this thread's slice of Q (F32, contiguous) once, pre-scaled.
    float4 qreg[VPT];
    [unroll] for (uint qi = 0; qi < VPT; qi++) {
        uint vidx = qi * D_SPLIT + d_tid;              // f32-vec4 index in the head dim
        uint4 qw  = src0.Load4(q_base + vidx * 16u);
        qreg[qi]  = float4(asfloat(qw.x), asfloat(qw.y), asfloat(qw.z), asfloat(qw.w)) * scale;
    }

    // Private online-softmax state for the KV columns this col_tid owns.
    float m_state = FA_NEG_MAX;
    float l_state = 0.0f;
    float4 o_state[VPT];
    [unroll] for (uint oi = 0; oi < VPT; oi++) o_state[oi] = float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Each col_tid strides through its KV columns; D_SPLIT lanes cooperate per column.
    for (uint kv = kv_start + col_tid; kv < kv_end; kv += COLS) {
        float mv = 0.0f;
        if (has_mask) {
            mv = cd_load_mask(mask_base + kv * mask_nb0, mask_es);
            mv *= slope;
        }
        if (isinf(mv)) {
            continue;   // fully masked column - all D_SPLIT lanes agree
        }

        const uint k_base = src1_offset + kv * nb11 + kv_head * nb12 + batch_idx * nb13;
        float partial = 0.0f;
        [unroll] for (uint di = 0; di < VPT; di++) {
            uint vidx = di * D_SPLIT + d_tid;
            partial += dot(qreg[di], cd_load_kv4(src1, k_base, vidx));
        }
        // Reduce the dot across the D_SPLIT lanes of this column (butterfly).
        [unroll] for (uint s = D_SPLIT / 2u; s > 0u; s >>= 1) {
            partial += WaveReadLaneAt(partial, lane ^ s);
        }

        float score = partial;   // scale already folded into qreg
        if (logit_softcap != 0.0f) {
            score = logit_softcap * tanh(score);
        }
        score += mv;

        // Online-softmax update for this column-group.
        float new_max = max(m_state, score);
        float corr    = (l_state > 0.0f) ? exp(m_state - new_max) : 0.0f;
        float p       = exp(score - new_max);

        const uint v_row_base = src2_off + kv * src2_nb1 + kv_head * src2_nb2 + batch_idx * src2_nb3;
        [unroll] for (uint vi = 0; vi < VPT; vi++) {
            uint vidx = vi * D_SPLIT + d_tid;
            o_state[vi] = o_state[vi] * corr + p * cd_load_kv4(src2, v_row_base, vidx);
        }
        l_state = l_state * corr + p;
        m_state = new_max;
    }

    // Merge the COLS private partials (across col_tid) with a second butterfly.
    [unroll] for (uint ms = D_SPLIT; ms < WAVE_SIZE; ms <<= 1) {
        float other_m = WaveReadLaneAt(m_state, lane ^ ms);
        float other_l = WaveReadLaneAt(l_state, lane ^ ms);
        float new_max = max(m_state, other_m);
        float a = (m_state > FA_NEG_MAX) ? exp(m_state - new_max) : 0.0f;
        float b = (other_m > FA_NEG_MAX) ? exp(other_m - new_max) : 0.0f;
        [unroll] for (uint mi = 0; mi < VPT; mi++) {
            float4 other_o = WaveReadLaneAt(o_state[mi], lane ^ ms);
            o_state[mi] = a * o_state[mi] + b * other_o;
        }
        l_state = a * l_state + b * other_l;
        m_state = new_max;
    }

    // Attention sinks tail (once per head/query, first split only).
    if (has_sinks != 0u && split_id == 0u) {
        float sink_s  = asfloat(src4.Load(head_idx * 4u));
        float new_max = max(m_state, sink_s);
        float corr    = (l_state > 0.0f) ? exp(m_state - new_max) : 0.0f;
        float p       = exp(sink_s - new_max);
        [unroll] for (uint si = 0; si < VPT; si++) o_state[si] *= corr;
        l_state = l_state * corr + p;
        m_state = new_max;
    }

    // Only the first column-group's lanes write (o_state is replicated across col_tid).
    if (col_tid != 0u) return;

    if (n_splits <= 1) {
        float inv = (l_state > 0.0f) ? (1.0f / l_state) : 0.0f;
        [unroll] for (uint wi = 0; wi < VPT; wi++) {
            uint d_out = (wi * D_SPLIT + d_tid) * 4u;
            float4 ov  = o_state[wi] * inv;
            uint base  = dst_offset + head_idx * nb1 + query_idx * nb2 + batch_idx * nb3;
            store_auto(dst, base + (d_out + 0u) * nb0, ov.x, dst_esize);
            store_auto(dst, base + (d_out + 1u) * nb0, ov.y, dst_esize);
            store_auto(dst, base + (d_out + 2u) * nb0, ov.z, dst_esize);
            store_auto(dst, base + (d_out + 3u) * nb0, ov.w, dst_esize);
        }
    } else {
        uint partial_stride = (HEAD_DIM + 2) * 4;
        uint partial_off = ((batch_idx * n_heads + head_idx) * (uint)ne01 + query_idx) * n_splits + split_id;
        partial_off *= partial_stride;
        if (lane == 0) {
            temp.Store(partial_off,     asuint(m_state));
            temp.Store(partial_off + 4, asuint(l_state));
        }
        [unroll] for (uint wi = 0; wi < VPT; wi++) {
            uint d_out = (wi * D_SPLIT + d_tid) * 4u;
            temp.Store4(partial_off + 8u + d_out * 4u, asuint(o_state[wi]));
        }
    }
}
