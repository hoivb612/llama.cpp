// wkv6.hlsl - RWKV WKV6 recurrent kernel (Phase 6).
//
// CPU reference: ggml_compute_forward_rwkv_wkv6_f32 (ggml/src/ggml-cpu/ops.cpp).
// Vulkan reference: ggml/src/ggml-vulkan/vulkan-shaders/wkv6.comp.
//
// Layout (all F32, contiguous):
//   src0 (k)        : [S, H, T]
//   src1 (v)        : [S, H, T]
//   src2 (r)        : [S, H, T]
//   src3 (tf)       : [S, H]
//   src4 (td)       : [S, H, T]
//   src5 (state_in) : [S*S*H, n_seqs]   (== S*S*H*n_seqs floats)
//   dst             : [S*H, T + S*n_seqs] = packed [token-outputs | new-state]
// where S = head_size, H = head_count, T = n_tokens.
//
// op_params:
//   op0 = B (n_seqs)
//
// Dispatch: groups_x = H * B (one workgroup per (batch, head)).
// Each workgroup runs S threads (S == BLOCK_SIZE == 64); each thread holds
// state[S] in registers and processes one output channel per token.

#include "ggml_common.hlsli"

#define BLOCK_SIZE 64

groupshared float _k[BLOCK_SIZE];
groupshared float _r[BLOCK_SIZE];
groupshared float _tf[BLOCK_SIZE];
groupshared float _td[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint S = ne00;       // head_size
    uint H = ne01;       // head_count
    uint T = ne02;       // n_tokens
    uint B = op0;        // n_seqs
    uint C = S * H;

    uint head_size = BLOCK_SIZE;
    uint batch_id = gid.x / H;
    uint head_id  = gid.x % H;

    if (batch_id >= B || head_id >= H) {
        return;
    }

    uint state_size  = C * head_size;
    uint n_seq_tokens = T / B;

    float state[BLOCK_SIZE];
    {
        uint state_base = batch_id * state_size + head_id * head_size * head_size;
        [unroll]
        for (uint i = 0; i < head_size; ++i) {
            state[i] = asfloat(src5.Load((state_base + i * head_size + tid) * 4u));
        }
    }

    GroupMemoryBarrierWithGroupSync();
    _tf[tid] = asfloat(src3.Load((head_id * head_size + tid) * 4u));
    GroupMemoryBarrierWithGroupSync();

    uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;
    uint end_t   = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += C) {
        GroupMemoryBarrierWithGroupSync();
        _k[tid]  = asfloat(src0.Load(t * 4u + src0_offset));
        _r[tid]  = asfloat(src2.Load(t * 4u));
        _td[tid] = asfloat(src4.Load(t * 4u));
        GroupMemoryBarrierWithGroupSync();

        float v_val = asfloat(src1.Load(t * 4u + src1_offset));
        float y = 0.0f;

        [unroll]
        for (uint j = 0; j < head_size; j += 4) {
            float4 k_vec  = float4(_k[j],  _k[j+1],  _k[j+2],  _k[j+3]);
            float4 r_vec  = float4(_r[j],  _r[j+1],  _r[j+2],  _r[j+3]);
            float4 tf_vec = float4(_tf[j], _tf[j+1], _tf[j+2], _tf[j+3]);
            float4 td_vec = float4(_td[j], _td[j+1], _td[j+2], _td[j+3]);
            float4 s_vec  = float4(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv = k_vec * v_val;
            float4 tmp = tf_vec * kv + s_vec;
            y += dot(r_vec, tmp);

            s_vec = s_vec * td_vec + kv;
            state[j  ] = s_vec.x;
            state[j+1] = s_vec.y;
            state[j+2] = s_vec.z;
            state[j+3] = s_vec.w;
        }

        dst.Store(t * 4u + dst_offset, asuint(y));
    }

    {
        uint state_out_base = T * C + batch_id * state_size + head_id * head_size * head_size;
        [unroll]
        for (uint i = 0; i < head_size; ++i) {
            dst.Store((state_out_base + i * head_size + tid) * 4u + dst_offset, asuint(state[i]));
        }
    }
}
