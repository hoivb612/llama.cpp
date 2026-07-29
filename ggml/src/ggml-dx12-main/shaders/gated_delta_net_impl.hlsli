// gated_delta_net_impl.hlsli - Shared body for the S_V/KDA variants of
// the Gated Delta Net shader.  The including file MUST define S_V to one
// of {16, 32, 64, 128} before #including this header.  KDA defaults to 0
// (scalar gate); set #define KDA 1 to load g per-row from a [S_V, H_v,
// n_tokens, n_seqs] gate tensor (sized like a Mamba2 Kappa-Delta-Anchor).
//
// Threading model:
//   numthreads(EFFECTIVE_THREADS) with EFFECTIVE_THREADS = min(S_V, WARP_SIZE).
//   Each lane owns ROWS_PER_LANE rows of one column (rows {r*ET + lane}).
//   When S_V < WARP_SIZE, only S_V lanes are active in the wave and
//   WaveActiveSum reduces across that subset.
//
// See gated_delta_net.hlsl for the original notes on inputs, output
// layout, and push-constant order.

#ifndef S_V
#error "S_V must be defined before including gated_delta_net_impl.hlsli"
#endif

#ifndef KDA
#define KDA 0
#endif

#include "ggml_common.hlsli"

#if (S_V) < (WARP_SIZE)
    #define EFFECTIVE_THREADS (S_V)
#else
    #define EFFECTIVE_THREADS (WARP_SIZE)
#endif
#define ROWS_PER_LANE ((S_V) / EFFECTIVE_THREADS)

WAVE_SIZE_ATTR
[numthreads(EFFECTIVE_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint head_id = gid.x;
    const uint seq_id  = gid.y;
    const uint col     = gid.z;
    const uint lane    = gtid.x;

    const uint H        = op_param_uint(0);
    const uint n_tokens = op_param_uint(1);
    const uint K        = op_param_uint(2);   // snapshot slot count; 1 = final-only
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

    if (head_id >= H) return;

    const uint iq1 = (neq1 != 0) ? (head_id % neq1) : 0;
    const uint iq3 = (rq3  != 0) ? (seq_id  / rq3)  : 0;

    const uint state_size = S_V * S_V;
    // input state layout (D, H, n_seqs): s0 only (upstream); per-seq stride is H*D.
    const uint state_in_base  = (seq_id * H + head_id) * state_size;
    // output state layout per slot: same per-(seq,head) offset as the single-slot case.
    const uint state_out_base = (seq_id * H + head_id) * state_size;
    // distance between snapshot slots in the dst buffer.
    // n_seqs * H = s_off / (S_V * n_tokens), so per-snap stride = state_size * n_seqs * H.
    const uint state_size_per_snap = (n_tokens != 0u) ? ((S_V * s_off) / n_tokens) : 0u;
    // snapshot slot mapping (upstream): slot 0 = final state, slot s = state s tokens
    // back. When n_tokens < K only slots 0..n_tokens-1 are written; earlier slots are
    // left untouched (caller-owned).

    float s_shard[ROWS_PER_LANE];
    [unroll] for (uint r0 = 0; r0 < ROWS_PER_LANE; r0++) {
        const uint i = r0 * EFFECTIVE_THREADS + lane;
        s_shard[r0] = asfloat(src5.Load((state_in_base + col * S_V + i) * 4u));
    }

    uint attn_out_off = (seq_id * n_tokens * H + head_id) * S_V;

    for (uint t = 0; t < n_tokens; t++) {
        const uint q_off  = iq3 * sq3 + t * sq2 + iq1 * sq1;
        const uint k_off  = q_off;
        const uint v_off  = seq_id * sv3 + t * sv2 + head_id * sv1;
        const uint gb_off = seq_id * sb3 + t * sb2 + head_id * sb1;

        const float beta_val = asfloat(src4.Load((gb_off) * 4u));

        float g_exp[ROWS_PER_LANE];
#if KDA == 0
        const float g_scalar = exp(asfloat(src3.Load((gb_off) * 4u)));
        [unroll] for (uint rg = 0; rg < ROWS_PER_LANE; rg++) {
            g_exp[rg] = g_scalar;
        }
#else
        const uint g_base = gb_off * S_V;
        [unroll] for (uint rg = 0; rg < ROWS_PER_LANE; rg++) {
            const uint i = rg * EFFECTIVE_THREADS + lane;
            g_exp[rg] = exp(asfloat(src3.Load((g_base + i) * 4u)));
        }
#endif

        float k_reg[ROWS_PER_LANE];
        float q_reg[ROWS_PER_LANE];
        [unroll] for (uint r1 = 0; r1 < ROWS_PER_LANE; r1++) {
            const uint i = r1 * EFFECTIVE_THREADS + lane;
            k_reg[r1] = asfloat(src1.Load((k_off + i) * 4u + src1_offset));
            q_reg[r1] = asfloat(src0.Load((q_off + i) * 4u + src0_offset));
        }

        const float v_val = asfloat(src2.Load((v_off + col) * 4u));

        float kv_shard = 0.0f;
        [unroll] for (uint r2 = 0; r2 < ROWS_PER_LANE; r2++) {
            kv_shard += g_exp[r2] * s_shard[r2] * k_reg[r2];
        }
        const float kv_col    = WaveActiveSum(kv_shard);
        const float delta_col = (v_val - kv_col) * beta_val;

        float attn_partial = 0.0f;
        [unroll] for (uint r3 = 0; r3 < ROWS_PER_LANE; r3++) {
            s_shard[r3] = g_exp[r3] * s_shard[r3] + k_reg[r3] * delta_col;
            attn_partial += s_shard[r3] * q_reg[r3];
        }
        const float attn_col = WaveActiveSum(attn_partial);

        if (lane == 0u) {
            dst.Store((attn_out_off + col) * 4u + dst_offset, asuint(attn_col * scale));
        }

        // K-snapshot output (Multi-Token Prediction): when K > 1, write the
        // running state into the matching slot of the last K tokens.  For
        // K == 1 this branch never fires and the post-loop single-state write
        // below produces bit-identical output to the pre-MTP path.
        if (K > 1u) {
            const int target_slot = int(n_tokens) - 1 - int(t);
            if (target_slot >= 0 && target_slot < int(K)) {
                const uint slot_base = s_off
                                     + uint(target_slot) * state_size_per_snap
                                     + state_out_base;
                [unroll] for (uint rs = 0; rs < ROWS_PER_LANE; rs++) {
                    const uint i = rs * EFFECTIVE_THREADS + lane;
                    dst.Store((slot_base + col * S_V + i) * 4u + dst_offset, asuint(s_shard[rs]));
                }
            }
        }

        attn_out_off += S_V * H;
    }

    if (K == 1u) {
        [unroll] for (uint r4 = 0; r4 < ROWS_PER_LANE; r4++) {
            const uint i = r4 * EFFECTIVE_THREADS + lane;
            dst.Store((s_off + state_out_base + col * S_V + i) * 4u + dst_offset, asuint(s_shard[r4]));
        }
    }
}
