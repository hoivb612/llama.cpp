// solve_tri.hlsl - Forward substitution for lower-triangular linear systems.
//
// Solves A * X = B for X, where:
//   A (src0): N x N lower triangular   (per (i02, i03) batch)
//   B (src1): N x K                    (per (i02, i03) batch)
//   X (dst):  N x K
//
// CPU reference: ggml_compute_forward_solve_tri_f32 (ggml/src/ggml-cpu/ops.cpp).
// Only the lower / right / non-unitriangular variant is implemented today.
//
// Dispatch: groups_x = ceil(K / WG_SIZE) * (ne02 * ne03).
//   Each workgroup handles up to WG_SIZE consecutive columns of one batch.
//   Each thread owns one column and holds X[N] in registers.
//
// Within a workgroup, A is streamed row-by-row through groupshared `shA`
// (no full preload — N can be up to MAX_N=256, full preload would need
// N*N*4 = 256 KB shmem). Per-row cost is one barrier-protected load of
// N floats plus per-thread forward substitution against X[0..r-1].

#include "ggml_common.hlsli"

#define WG_SIZE 64
#define MAX_N 256

groupshared float shA[MAX_N];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid_idx : SV_GroupIndex) {
    uint N = ne00;
    uint K = ne10;
    uint K_chunks = (K + WG_SIZE - 1) / WG_SIZE;

    uint flat    = gid.x;
    uint k_chunk = flat % K_chunks;
    uint batch   = flat / K_chunks;

    uint i02 = batch % ne02;
    uint i03 = batch / ne02;

    uint c_abs = k_chunk * WG_SIZE + tid_idx;
    bool active = (c_abs < K);

    uint a_base = src0_offset + i02 * nb02 + i03 * nb03;
    uint b_base = src1_offset + i02 * nb12 + i03 * nb13;
    uint x_base = dst_offset  + i02 * nb2  + i03 * nb3;

    float X[MAX_N];

    [loop]
    for (uint r = 0; r < N; ++r) {
        for (uint t = tid_idx; t < N; t += WG_SIZE) {
            shA[t] = asfloat(src0.Load(a_base + r * nb01 + t * nb00));
        }
        GroupMemoryBarrierWithGroupSync();

        if (active) {
            float sum = 0.0f;
            for (uint t = 0; t < r; ++t) {
                sum += shA[t] * X[t];
            }
            float b    = asfloat(src1.Load(b_base + r * nb11 + c_abs * nb10));
            float diag = shA[r];
            float x    = (b - sum) / diag;
            X[r] = x;
            dst.Store(x_base + r * nb1 + c_abs * nb0, asuint(x));
        }

        GroupMemoryBarrierWithGroupSync();
    }
}
