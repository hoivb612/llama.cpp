// lightning_indexer.hlsli - shared body for the DSA index-score kernels.
//
//   res[ik, t, s] = sum_h relu(dot(q[:, h, t, s], k[:, ik, s])) * w[h, t, s]
//                   + mask[ik, t, s % nem3]
//
// src0 = q       [n_embd, n_head, n_tokens, n_stream]  F32
// src1 = k       [n_embd, 1,      n_kv,     n_stream]  float or quantized
// src2 = weights [n_head, n_tokens, 1,      n_stream]  F32, prescaled
// src3 = mask    [n_kv,   n_tokens, 1,      nem3   ]  F16
//
// One group per (ik, t, s). The k row is staged in LDS once and reused by
// every head. Each wave computes one head with adjacent lanes reading adjacent q elements.
//
// Quantized-K wrappers define one MMID_<TYPE> macro before including this.
#pragma once

#include "ggml_common.hlsli"
#ifdef LI_QUANT
#include "quant_dequant.hlsli"
#endif

#define GROUP_SIZE 256
#define MAX_EMBD  256

groupshared float k_sh[MAX_EMBD];
groupshared float part[GROUP_SIZE];

WAVE_SIZE_ATTR
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint ik = group_id.x;
    uint t  = group_id.y;
    uint s  = group_id.z;

    if (ik >= ne0 || t >= ne1 || s >= ne3) return;

    uint n_embd = ne00;
    uint n_head = ne01;

    uint k_base = src1_offset + ik * nb12 + s * nb13;
    for (uint e = tid; e < n_embd; e += GROUP_SIZE) {
#ifdef LI_QUANT
        k_sh[e] = mmid_dequant(src1, k_base, e);
#else
        k_sh[e] = load_auto(src1, k_base + e * nb10, src1_esize);
#endif
    }
    GroupMemoryBarrierWithGroupSync();

    uint q_base = src0_offset + t * nb02 + s * nb03;
    uint w_base = (t + s * ne1) * n_head * 4u;
    uint wave_size = WaveGetLaneCount();
    uint lane = WaveGetLaneIndex();
    uint wave = tid / wave_size;
    uint n_waves = GROUP_SIZE / wave_size;

    float acc = 0.0f;
    for (uint h = wave; h < n_head; h += n_waves) {
        uint q_row = q_base + h * nb01;
        float qk = 0.0f;
        for (uint e = lane; e < n_embd; e += wave_size) {
            qk += asfloat(src0.Load(q_row + e * nb00)) * k_sh[e];
        }
        qk = WaveActiveSum(qk);
        if (lane == 0) {
            acc += max(qk, 0.0f) * load_auto(src2, w_base + h * 4u, 4u);
        }
    }

    if (lane == 0) {
        part[wave] = acc;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float score = 0.0f;
        for (uint w = 0; w < n_waves; ++w) {
            score += part[w];
        }
        uint m_off = (ik + (t + (s % ne2) * ne1) * ne0) * 2u;
        float res  = score + load_auto(src3, m_off, 2u);
        store_auto(dst, dst_offset + ik * nb0 + t * nb1 + s * nb3, res, dst_esize);
    }
}
