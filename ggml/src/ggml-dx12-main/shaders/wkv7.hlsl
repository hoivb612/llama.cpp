// wkv7.hlsl - RWKV WKV7 recurrent kernel (Phase 6).
//
// CPU reference: ggml_compute_forward_rwkv_wkv7_f32 (ggml/src/ggml-cpu/ops.cpp).
// Vulkan reference: ggml/src/ggml-vulkan/vulkan-shaders/wkv7.comp.
//
// Layout (all F32, contiguous):
//   src0 (r)        : [S, H, T]
//   src1 (w)        : [S, H, T]
//   src2 (k)        : [S, H, T]
//   src3 (v)        : [S, H, T]
//   src4 (a)        : [S, H, T]
//   src5 (b)        : [S, H, T]
//   src6 (state_in) : [S*S*H, n_seqs]
//   dst             : [S*H, T + S*n_seqs] = packed [token-outputs | new-state]
//
// op_params:
//   op0 = B (n_seqs)
//
// Dispatch: groups_x = H * B; BLOCK_SIZE threads per group (BLOCK_SIZE == S == 64).

#include "ggml_common.hlsli"

#define BLOCK_SIZE 64

groupshared float _r[BLOCK_SIZE];
groupshared float _w[BLOCK_SIZE];
groupshared float _k[BLOCK_SIZE];
groupshared float _a[BLOCK_SIZE];
groupshared float _b[BLOCK_SIZE];

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

    // Note: state indexing differs from WKV6 — here tid indexes the OUTER row
    // (the "i" in CPU's `state_prev[h_2d_i_offset + j]`) instead of the inner
    // column. See ggml/src/ggml-vulkan/vulkan-shaders/wkv7.comp for the same
    // [tid*head_size + i] vs WKV6's [i*head_size + tid] split.
    float state[BLOCK_SIZE];
    {
        uint state_base = batch_id * state_size + head_id * head_size * head_size;
        [unroll]
        for (uint i = 0; i < head_size; ++i) {
            state[i] = asfloat(src6.Load((state_base + tid * head_size + i) * 4u));
        }
    }

    uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;
    uint end_t   = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += C) {
        GroupMemoryBarrierWithGroupSync();
        _r[tid] = asfloat(src0.Load(t * 4u + src0_offset));
        _w[tid] = asfloat(src1.Load(t * 4u + src1_offset));
        _k[tid] = asfloat(src2.Load(t * 4u));
        _a[tid] = asfloat(src4.Load(t * 4u));
        _b[tid] = asfloat(src5.Load(t * 4u));
        GroupMemoryBarrierWithGroupSync();

        float sa = 0.0f;
        [unroll]
        for (uint j = 0; j < head_size; j += 4) {
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);
            float4 a_vec = float4(_a[j],   _a[j+1],   _a[j+2],   _a[j+3]);
            sa += dot(s_vec, a_vec);
        }

        float v_val = asfloat(src3.Load(t * 4u));
        float y = 0.0f;

        [unroll]
        for (uint j = 0; j < head_size; j += 4) {
            float4 r_vec = float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            float4 w_vec = float4(_w[j], _w[j+1], _w[j+2], _w[j+3]);
            float4 k_vec = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            float4 b_vec = float4(_b[j], _b[j+1], _b[j+2], _b[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv = k_vec * v_val;
            s_vec = s_vec * w_vec + kv + sa * b_vec;
            y += dot(r_vec, s_vec);

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
            dst.Store((state_out_base + tid * head_size + i) * 4u + dst_offset, asuint(state[i]));
        }
    }
}
