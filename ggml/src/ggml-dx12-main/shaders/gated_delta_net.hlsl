// gated_delta_net.hlsl - Fused Gated Delta Net (Mamba2-style attention).
//
// Inputs:
//   src0 (t0) = q : [S_v, H_q, n_tokens, n_seqs]                    F32
//   src1 (t1) = k : [S_v, H_q, n_tokens, n_seqs]                    F32
//   src2 (t2) = v : [S_v, H_v, n_tokens, n_seqs]                    F32
//   src3 (t3) = g : [1 or S_v, H_v, n_tokens, n_seqs]               F32 (gate; KDA when ne0==S_v)
//   src4 (t4) = beta : [1, H_v, n_tokens, n_seqs]                   F32
//   src5 (t5) = state : [S_v, S_v, H_v, n_seqs]                     F32 (initial state)
//
// Output (dst, packed):
//   per-token attn outputs : [S_v, H_v, n_tokens, n_seqs] (length S_v*H_v*n_tokens*n_seqs floats)
//   then new state         : [S_v, S_v, H_v, n_seqs]                at byte offset s_off*4
//
// Dispatch: (H_v, n_seqs, S_v) workgroups, S_v threads each.
// Each workgroup processes one (head, seq, column) — column-parallel rows are owned by lanes.
//
// Push constants (mapped from op_params):
//   op0  = H            (n_heads, value side)
//   op1  = n_tokens
//   op2  = n_seqs
//   op3  = s_off        (number of float elements before the state output region in dst)
//   op4..op6  = sq1, sq2, sq3 (q strides in elements)
//   op7..op9  = sv1, sv2, sv3 (v strides in elements)
//   op10..op12= sb1, sb2, sb3 (beta strides in elements)
//   op13 = neq1         (q->ne[1], i.e. H_q for GQA-aware q indexing)
//   op14 = rq3          (n_seqs / q->ne[3])
//   op15 = scale        (float bits)
//
// Currently compiled only for S_v = 128 (the qwen3.5 case). Smaller S_v
// (32/64) would need a separate variant or runtime masking.
#include "ggml_common.hlsli"

// gated_delta_net.hlsl - Fused Gated Delta Net (Mamba2-style attention).
//
// Inputs:
//   src0 (t0) = q : [S_v, H_q, n_tokens, n_seqs]                    F32
//   src1 (t1) = k : [S_v, H_q, n_tokens, n_seqs]                    F32
//   src2 (t2) = v : [S_v, H_v, n_tokens, n_seqs]                    F32
//   src3 (t3) = g : [1 or S_v, H_v, n_tokens, n_seqs]               F32 (gate; KDA when ne0==S_v)
//   src4 (t4) = beta : [1, H_v, n_tokens, n_seqs]                   F32
//   src5 (t5) = state : [S_v, S_v, H_v, n_seqs]                     F32 (initial state)
//
// Output (dst, packed):
//   per-token attn outputs : [S_v, H_v, n_tokens, n_seqs] (length S_v*H_v*n_tokens*n_seqs floats)
//   then new state         : [S_v, S_v, H_v, n_seqs]                at byte offset s_off*4
//
// Dispatch: (H_v, n_seqs, S_v) workgroups, WARP_SIZE threads each.
// Each workgroup processes one (head, seq, column) — the S_V rows of that
// column are sharded across the wave's lanes (ROWS_PER_LANE rows per lane).
// Single wave per WG so reductions use WaveActiveSum (no barriers, no shmem).
//
// Push constants (mapped from op_params): see file for layout.
//
// Currently compiled only for S_v = 128 (the qwen3.5 case). Smaller S_v
// (32/64) would need a separate variant or runtime masking.

#define S_V 128
#define ROWS_PER_LANE (S_V / WARP_SIZE)

WAVE_SIZE_ATTR
[numthreads(WARP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint head_id = gid.x;
    const uint seq_id  = gid.y;
    const uint col     = gid.z;
    const uint lane    = gtid.x;

    const uint H        = op_param_uint(0);
    const uint n_tokens = op_param_uint(1);
    const uint s_off    = op_param_uint(3);
    const uint sq1      = op_param_uint(4);
    const uint sq2      = op_param_uint(5);
    const uint sq3      = op_param_uint(6);
    const uint sv1      = op_param_uint(7);
    const uint sv2      = op_param_uint(8);
    const uint sv3      = op_param_uint(9);
    const uint sb1      = op_param_uint(10);
    const uint sb2      = op_param_uint(11);
    const uint sb3      = op_param_uint(12);
    const uint neq1     = op_param_uint(13);
    const uint rq3      = op_param_uint(14);
    const float scale   = op_param_f32(15);

    // Bounds: dispatch is exact for (H, n_seqs, S_V) but guard anyway.
    if (head_id >= H) return;

    // GQA Q-head mapping (q has H_q heads, v has H heads; H % H_q == 0).
    const uint iq1 = (neq1 != 0) ? (head_id % neq1) : 0;
    const uint iq3 = (rq3  != 0) ? (seq_id  / rq3)  : 0;

    const uint state_size = S_V * S_V;
    const uint state_base = (seq_id * H + head_id) * state_size;

    // Each lane owns ROWS_PER_LANE rows of the column: rows {r*WARP_SIZE + lane}.
    float s_shard[ROWS_PER_LANE];
    [unroll] for (uint r0 = 0; r0 < ROWS_PER_LANE; r0++) {
        const uint i = r0 * WARP_SIZE + lane;
        s_shard[r0] = asfloat(src5.Load((state_base + col * S_V + i) * 4u));
    }

    uint attn_out_off = (seq_id * n_tokens * H + head_id) * S_V;  // element offset

    for (uint t = 0; t < n_tokens; t++) {
        const uint q_off  = iq3 * sq3 + t * sq2 + iq1 * sq1;
        const uint k_off  = q_off;                             // same layout
        const uint v_off  = seq_id * sv3 + t * sv2 + head_id * sv1;
        const uint gb_off = seq_id * sb3 + t * sb2 + head_id * sb1;

        const float beta_val = asfloat(src4.Load((gb_off) * 4u));
        const float g_val    = exp(asfloat(src3.Load((gb_off) * 4u)));  // KDA=0: scalar gate

        float k_reg[ROWS_PER_LANE];
        float q_reg[ROWS_PER_LANE];
        [unroll] for (uint r1 = 0; r1 < ROWS_PER_LANE; r1++) {
            const uint i = r1 * WARP_SIZE + lane;
            k_reg[r1] = asfloat(src1.Load((k_off + i) * 4u + src1_offset));
            q_reg[r1] = asfloat(src0.Load((q_off + i) * 4u + src0_offset));
        }

        // src2 (v) has its tensor offset baked into the SRV GPU VA in the
        // dispatch code, so byte offset 0 here = tensor base.
        const float v_val = asfloat(src2.Load((v_off + col) * 4u));

        // First reduction: kv_col = sum_row{ g * s[row] * k[row] }
        float kv_shard = 0.0f;
        [unroll] for (uint r2 = 0; r2 < ROWS_PER_LANE; r2++) {
            kv_shard += g_val * s_shard[r2] * k_reg[r2];
        }
        const float kv_col    = WaveActiveSum(kv_shard);
        const float delta_col = (v_val - kv_col) * beta_val;

        // Update state and accumulate attn: s[r] = g*s[r] + k[r]*delta;
        // attn = sum_r(s[r]*q[r])
        float attn_partial = 0.0f;
        [unroll] for (uint r3 = 0; r3 < ROWS_PER_LANE; r3++) {
            s_shard[r3] = g_val * s_shard[r3] + k_reg[r3] * delta_col;
            attn_partial += s_shard[r3] * q_reg[r3];
        }
        const float attn_col = WaveActiveSum(attn_partial);

        if (lane == 0u) {
            dst.Store((attn_out_off + col) * 4u + dst_offset, asuint(attn_col * scale));
        }

        attn_out_off += S_V * H;
    }

    // Write final state column rows
    [unroll] for (uint r4 = 0; r4 < ROWS_PER_LANE; r4++) {
        const uint i = r4 * WARP_SIZE + lane;
        dst.Store((s_off + state_base + col * S_V + i) * 4u + dst_offset, asuint(s_shard[r4]));
    }
}
