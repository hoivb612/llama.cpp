// gla.hlsl - Gated Linear Attention recurrent kernel.
//
// CPU reference: ggml_compute_forward_gla_f32 (ggml/src/ggml-cpu/ops.cpp).
//
// Layout (all F32, contiguous):
//   src0 (k)        : [S, H, T]
//   src1 (v)        : [S, H, T]
//   src2 (q)        : [S, H, T]
//   src3 (g)        : [S, H, T]
//   src4 (state_in) : [S*S*H, n_seqs]
//   dst             : [S*H, T + S*n_seqs] = packed [token-outputs | new-state]
// where S = head_size, H = head_count, T = n_tokens.
//
// op_params:
//   op0 = B (n_seqs)
//   op1 = scale (float bits)
//
// Dispatch: groups_x = H * B. Each workgroup runs S threads; thread tid owns
// output channel tid and holds state[i][tid] for all i in registers.

#include "ggml_common.hlsli"

#define BLOCK_SIZE 64

groupshared float _k[BLOCK_SIZE];
groupshared float _q[BLOCK_SIZE];
groupshared float _g[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint S = ne00;
    uint H = ne01;
    uint T = ne02;
    uint B = op0;
    uint C = S * H;
    float scale = asfloat(op1);

    uint head_size = BLOCK_SIZE;
    uint batch_id = gid.x / H;
    uint head_id  = gid.x % H;

    if (batch_id >= B || head_id >= H) {
        return;
    }

    uint state_size   = C * head_size;
    uint n_seq_tokens = T / B;

    float state[BLOCK_SIZE];
    {
        uint state_base = batch_id * state_size + head_id * head_size * head_size;
        [unroll]
        for (uint i = 0; i < head_size; ++i) {
            state[i] = asfloat(src4.Load((state_base + i * head_size + tid) * 4u));
        }
    }

    uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;
    uint end_t   = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += C) {
        GroupMemoryBarrierWithGroupSync();
        _k[tid] = asfloat(src0.Load(t * 4u + src0_offset));
        _q[tid] = asfloat(src2.Load(t * 4u)) * scale;
        _g[tid] = asfloat(src3.Load(t * 4u));
        GroupMemoryBarrierWithGroupSync();

        float v_val = asfloat(src1.Load(t * 4u + src1_offset));
        float y = 0.0f;

        [unroll]
        for (uint j = 0; j < head_size; j += 4) {
            float4 k_vec = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            float4 q_vec = float4(_q[j], _q[j+1], _q[j+2], _q[j+3]);
            float4 g_vec = float4(_g[j], _g[j+1], _g[j+2], _g[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);

            s_vec = s_vec * g_vec + k_vec * v_val;
            y += dot(q_vec, s_vec);

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
