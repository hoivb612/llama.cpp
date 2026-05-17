// mul_mat_vec_glu.hlsl - Fused MUL_MAT(M=1, W_gate) + MUL_MAT(M=1, W_up) + GLU(SWIGLU split)
//
// Collapses three ops into one dispatch:
//   t_gate = W_gate @ x       (matvec)
//   t_up   = W_up   @ x       (matvec, same x)
//   y      = silu(t_gate) * t_up  (split-mode SwiGLU)
//
// The two weight matrices are sibling FFN projections that share the
// same activation x.  By doing both dot products in a single K-loop we
// load x once and avoid materialising t_gate and t_up to memory.
//
// In the fused graph the gate matvec lands at node[i] (its W is bound
// as src0 by the existing matvec routing) and the up matvec lands at
// node[i+1] (its W is bound as src2 with op1 = byte offset).
//
// Bindings:
//   src0 (t0): W_gate weights, F16, ne00=K, ne01=N
//   src1 (t1): x       activation, F32, contiguous, ne10=K
//   src2 (t2): W_up    weights, F16, same shape and stride as W_gate.
//              Bound at the resource base; W_up's tensor byte offset
//              is passed in op1.
//   dst  (u0): y       fused output, F32, ne0=N (= GLU split-mode output width)
//
// op_params:
//   op1 = W_up base byte offset (within src2)
//
// Only SWIGLU is supported; that is the activation used by every
// LLaMA-class FFN we currently care about.  Other GLU variants would
// require a per-variant shader; defer until we have a workload.
//
// Dispatch geometry mirrors mul_mat_vec_mr.hlsl:
//   groups_x = (N+1)/2 (two output rows per group)
//   groups_y = 1
//   groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 256
#define NUM_ROWS   2

groupshared float shared_acc[128];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;
    uint row0 = group_x_2d(gid) * NUM_ROWS;
    if (row0 >= ne0) return;
    uint row1 = min(row0 + 1, ne0 - 1);

    uint i2 = gid.z % ne2;
    uint i3 = gid.z / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint up_base   = op1          + i2_src0 * nb02 + i3_src0 * nb03;
    uint gate0_base = src0_base + row0 * nb01;
    uint up0_base   = up_base   + row0 * nb01;
    uint gate1_base = src0_base + row1 * nb01;
    uint up1_base   = up_base   + row1 * nb01;
    uint x_base     = src1_offset + i2 * nb12 + i3 * nb13;

    bool gate_pair_aligned = ((gate0_base | gate1_base) & 3u) == 0;
    bool up_pair_aligned   = ((up0_base   | up1_base)   & 3u) == 0;
    bool x_contiguous      = nb10 == 4;

    precise float acc_gate0 = 0.0f;
    precise float acc_up0   = 0.0f;
    precise float acc_gate1 = 0.0f;
    precise float acc_up1   = 0.0f;

    if (x_contiguous && gate_pair_aligned && up_pair_aligned) {
        // F16 weights + contiguous F32 input — process two rows per group
        // and four K values per participating lane, matching the standalone
        // MR shader's geometry while sharing activation loads across gate/up.
        uint k = tid * 4;
        for (; k + 3 < K; k += GROUP_SIZE * 4) {
            uint4 x4 = src1.Load4(x_base + k * 4);
            float x0 = asfloat(x4.x); float x1 = asfloat(x4.y);
            float x2 = asfloat(x4.z); float x3 = asfloat(x4.w);

#if NATIVE_FP16
            vector<float16_t,4> wg0 = src0.Load<vector<float16_t,4> >(gate0_base + k * 2);
            acc_gate0 = mad((float)wg0.x, x0, mad((float)wg0.y, x1,
                       mad((float)wg0.z, x2, mad((float)wg0.w, x3, acc_gate0))));

            vector<float16_t,4> wu0 = src2.Load<vector<float16_t,4> >(up0_base + k * 2);
            acc_up0 = mad((float)wu0.x, x0, mad((float)wu0.y, x1,
                    mad((float)wu0.z, x2, mad((float)wu0.w, x3, acc_up0))));

            vector<float16_t,4> wg1 = src0.Load<vector<float16_t,4> >(gate1_base + k * 2);
            acc_gate1 = mad((float)wg1.x, x0, mad((float)wg1.y, x1,
                       mad((float)wg1.z, x2, mad((float)wg1.w, x3, acc_gate1))));

            vector<float16_t,4> wu1 = src2.Load<vector<float16_t,4> >(up1_base + k * 2);
            acc_up1 = mad((float)wu1.x, x0, mad((float)wu1.y, x1,
                    mad((float)wu1.z, x2, mad((float)wu1.w, x3, acc_up1))));
#else
            uint2 wg0 = src0.Load2(gate0_base + k * 2);
            acc_gate0 = mad(f16tof32(wg0.x & 0xFFFFu), x0, mad(f16tof32(wg0.x >> 16), x1,
                       mad(f16tof32(wg0.y & 0xFFFFu), x2, mad(f16tof32(wg0.y >> 16), x3, acc_gate0))));

            uint2 wu0 = src2.Load2(up0_base + k * 2);
            acc_up0 = mad(f16tof32(wu0.x & 0xFFFFu), x0, mad(f16tof32(wu0.x >> 16), x1,
                    mad(f16tof32(wu0.y & 0xFFFFu), x2, mad(f16tof32(wu0.y >> 16), x3, acc_up0))));

            uint2 wg1 = src0.Load2(gate1_base + k * 2);
            acc_gate1 = mad(f16tof32(wg1.x & 0xFFFFu), x0, mad(f16tof32(wg1.x >> 16), x1,
                       mad(f16tof32(wg1.y & 0xFFFFu), x2, mad(f16tof32(wg1.y >> 16), x3, acc_gate1))));

            uint2 wu1 = src2.Load2(up1_base + k * 2);
            acc_up1 = mad(f16tof32(wu1.x & 0xFFFFu), x0, mad(f16tof32(wu1.x >> 16), x1,
                    mad(f16tof32(wu1.y & 0xFFFFu), x2, mad(f16tof32(wu1.y >> 16), x3, acc_up1))));
#endif
        }
        for (; k < K; k++) {
            float x = asfloat(src1.Load(x_base + k * 4));
            acc_gate0 = mad(load_auto(src0, gate0_base + k * 2, 2), x, acc_gate0);
            acc_up0   = mad(load_auto(src2, up0_base   + k * 2, 2), x, acc_up0);
            acc_gate1 = mad(load_auto(src0, gate1_base + k * 2, 2), x, acc_gate1);
            acc_up1   = mad(load_auto(src2, up1_base   + k * 2, 2), x, acc_up1);
        }
    } else {
        // Generic path — handles non-contiguous activation or unaligned
        // weight base (rare for FFN projections but kept for safety).
        for (uint k = tid; k < K; k += GROUP_SIZE) {
            float x = load_auto(src1, x_base + k * nb10, src1_esize);
            acc_gate0 = mad(load_auto(src0, gate0_base + k * 2, 2), x, acc_gate0);
            acc_up0   = mad(load_auto(src2, up0_base   + k * 2, 2), x, acc_up0);
            acc_gate1 = mad(load_auto(src0, gate1_base + k * 2, 2), x, acc_gate1);
            acc_up1   = mad(load_auto(src2, up1_base   + k * 2, 2), x, acc_up1);
        }
    }

    float wave_gate0 = WaveActiveSum(acc_gate0);
    float wave_up0   = WaveActiveSum(acc_up0);
    float wave_gate1 = WaveActiveSum(acc_gate1);
    float wave_up1   = WaveActiveSum(acc_up1);

    uint wave_id   = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id]      = wave_gate0;
        shared_acc[32 + wave_id] = wave_up0;
        shared_acc[64 + wave_id] = wave_gate1;
        shared_acc[96 + wave_id] = wave_up1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid]      += shared_acc[tid + s];
            shared_acc[32 + tid] += shared_acc[32 + tid + s];
            shared_acc[64 + tid] += shared_acc[64 + tid + s];
            shared_acc[96 + tid] += shared_acc[96 + tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float gate0 = shared_acc[0];
        float up0   = shared_acc[32];
        float result0 = (gate0 / (1.0f + exp(-gate0))) * up0;
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float gate1 = shared_acc[64];
            float up1   = shared_acc[96];
            float result1 = (gate1 / (1.0f + exp(-gate1))) * up1;
            uint off_d1 = offset_4d(row1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
