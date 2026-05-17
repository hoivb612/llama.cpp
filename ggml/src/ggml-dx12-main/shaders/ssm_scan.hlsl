// ssm_scan.hlsl - Mamba2 selective scan (SSM_SCAN op).
//
// Inputs:
//   src0 (t0) = ssm initial state : [D_STATE, head_dim, n_head, n_seqs]            F32
//   src1 (t1) = x                 : [head_dim, n_head, n_tok, n_seqs]              F32
//   src2 (t2) = dt                : [n_head, n_tok, n_seqs]                        F32
//   src3 (t3) = A                 : [n_head] (Mamba2: scalar per head)             F32
//   src4 (t4) = B                 : [D_STATE, n_group, n_tok, n_seqs]              F32
//   src5 (t5) = C                 : [D_STATE, n_group, n_tok, n_seqs]              F32
//   src6 (t6) = ids               : [n_seqs]                                       I32
//
// Output (dst, packed):
//   per-token y outputs : [head_dim*n_head, n_tok, n_seqs]
//   then new state      : [D_STATE, head_dim, n_head, n_seqs] at byte offset s_off
//
// Push constants (mapped from op_params):
//   op0  = nb02   (src0 stride bytes for dim 2 — head dim)
//   op1  = nb03   (src0 stride bytes for dim 3 — seq)
//   op2  = nb12   (src1 stride bytes — token)
//   op3  = nb13   (src1 stride bytes — seq)
//   op4  = nb21   (src2 stride bytes — token)
//   op5  = nb22   (src2 stride bytes — seq)
//   op6  = nb31   (src3 stride bytes — head)
//   op7  = nb42   (src4 stride bytes — token)
//   op8  = nb43   (src4 stride bytes — seq)
//   op9  = nb52   (src5 stride bytes — token)
//   op10 = nb53   (src5 stride bytes — seq)
//   op11 = s_off  (byte offset to state region in dst)
//   op12 = n_head
//   op13 = d_head (head_dim)
//   op14 = n_group
//   op15 = n_tok
//
// Hardcoded D_STATE = 128 (Mamba2 / qwen3.5).
// 128 threads per workgroup; each wave handles one (head_idx, head_off) location.
#include "ggml_common.hlsli"

#define D_STATE 128
#define GROUP_SIZE 128

// c_factor = D_STATE / WAVE_SIZE = each thread owns this many state slots.
#define C_FACTOR (D_STATE / WARP_SIZE)
#define NUM_WAVES (GROUP_SIZE / WARP_SIZE)

groupshared float temp_reduce[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint nb02   = op_param_uint(0);
    const uint nb03   = op_param_uint(1);
    const uint nb12   = op_param_uint(2);
    const uint nb13   = op_param_uint(3);
    const uint nb21   = op_param_uint(4);
    const uint nb22   = op_param_uint(5);
    const uint nb31   = op_param_uint(6);
    const uint nb42   = op_param_uint(7);
    const uint nb43   = op_param_uint(8);
    const uint nb52   = op_param_uint(9);
    const uint nb53   = op_param_uint(10);
    const uint s_off  = op_param_uint(11);
    const uint n_head = op_param_uint(12);
    const uint d_head = op_param_uint(13);
    const uint n_group = op_param_uint(14);
    const uint n_tok  = op_param_uint(15);

    const uint tid           = gtid.x;
    const uint subgroup      = tid / WARP_SIZE;
    const uint lane          = tid % WARP_SIZE;
    const uint subgroup_idx  = gid.x * NUM_WAVES + subgroup;
    const uint head_idx      = subgroup_idx / d_head;
    const uint head_off      = (subgroup_idx % d_head) * 4u;
    const uint seq_idx       = gid.y;

    // Bounds: avoid OOB if dispatch over-rounds
    if (head_idx >= n_head) return;

    const uint heads_per_group = n_head / n_group;
    const uint group_off       = (head_idx / heads_per_group) * D_STATE * 4u;

    const int  seq_id_int = asint(src6.Load(seq_idx * 4u));
    const uint seq_id     = (uint)seq_id_int;

    // s0_base index in elements
    const uint s0_base_idx = (seq_id * nb03 + head_idx * nb02 + head_off * D_STATE) / 4u;
    const uint x_base_idx  = (seq_idx * nb13 + subgroup_idx * 4u) / 4u;
    const uint dt_base_idx = (seq_idx * nb22 + head_idx * 4u) / 4u;
    const uint A_base_idx  = (head_idx * nb31) / 4u;
    const uint B_base_idx  = (seq_idx * nb43 + group_off) / 4u;
    const uint C_base_idx  = (seq_idx * nb53 + group_off) / 4u;
    const uint y_base_idx  = seq_idx * n_tok * n_head * d_head + subgroup_idx;
    const uint s_base_idx  = (s_off + seq_idx * nb03 + head_idx * nb02 + head_off * D_STATE) / 4u;

    const uint stride_x  = nb12 / 4u;
    const uint stride_dt = nb21 / 4u;
    const uint stride_B  = nb42 / 4u;
    const uint stride_C  = nb52 / 4u;
    const uint stride_y  = n_head * d_head;

    float state[C_FACTOR];
    [unroll] for (uint j = 0; j < C_FACTOR; j++) {
        state[j] = asfloat(src0.Load((s0_base_idx + WARP_SIZE * j + lane) * 4u + src0_offset));
    }

    float a = asfloat(src3.Load(A_base_idx * 4u));

    for (uint i = 0; i < n_tok; i++) {
        float dt_v = asfloat(src2.Load((dt_base_idx + i * stride_dt) * 4u));
        // softplus
        float dt_sp = (dt_v <= 20.0f) ? log(1.0f + exp(dt_v)) : dt_v;

        float state_sum = 0.0f;
        const float dA   = exp(dt_sp * a);
        const float x_dt = asfloat(src1.Load((x_base_idx + i * stride_x) * 4u + src1_offset)) * dt_sp;

        [unroll] for (uint j2 = 0; j2 < C_FACTOR; j2++) {
            float B_val = asfloat(src4.Load((B_base_idx + i * stride_B + WARP_SIZE * j2 + lane) * 4u));
            float C_val = asfloat(src5.Load((C_base_idx + i * stride_C + WARP_SIZE * j2 + lane) * 4u));
            state[j2] = (state[j2] * dA) + (B_val * x_dt);
            state_sum += state[j2] * C_val;
        }

        // wave-wide reduction
        state_sum = WaveActiveSum(state_sum);

        if (lane == 0u) {
            dst.Store((y_base_idx + i * stride_y) * 4u + dst_offset, asuint(state_sum));
        }
    }

    [unroll] for (uint j3 = 0; j3 < C_FACTOR; j3++) {
        dst.Store((s_base_idx + WARP_SIZE * j3 + lane) * 4u + dst_offset, asuint(state[j3]));
    }
}
