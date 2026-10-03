// moe_expert_bucket.hlsl - group MUL_MAT_ID token/expert pairs by expert.
//
// The MMID matvec kernels give one threadgroup to a single (token, expert)
// pair, so at prefill every expert matrix is re-read once per token routed to
// it. Grouping the pairs by expert first lets a tiled GEMM read each expert
// once per pair tile instead.
//
// Reads the ids tensor as src0 and writes, into dst:
//   [0 .. n_expert]                 exclusive prefix sums (n_expert+1 uints)
//   [op0 .. op0 + nei0*nei1*4)      flat pair indices grouped by expert
//
// A pair index is `token * nei0 + slot`, so the GEMM recovers both with one
// divide per row rather than one per element.
//
// One threadgroup: the whole ids tensor is at most n_expert_used * n_tokens
// entries and the prefix sum has to be serial across experts anyway. Host
// gates on n_expert <= MOE_BUCKET_MAX_EXPERTS.
#include "ggml_common.hlsli"

#define BUCKET_THREADS 256

#ifndef MOE_BUCKET_MAX_EXPERTS
#define MOE_BUCKET_MAX_EXPERTS 512
#endif

groupshared uint g_count[MOE_BUCKET_MAX_EXPERTS];
groupshared uint g_cursor[MOE_BUCKET_MAX_EXPERTS];

[numthreads(BUCKET_THREADS, 1, 1)]
void main(uint tid : SV_GroupIndex) {
    const uint n_expert = ne00;
    const uint nei0     = ne01;
    const uint nei1     = ne02;
    const uint total    = nei0 * nei1;

    for (uint z = tid; z < n_expert; z += BUCKET_THREADS) {
        g_count[z] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = tid; i < total; i += BUCKET_THREADS) {
        uint slot  = i % nei0;
        uint token = i / nei0;
        uint id    = (uint)asint(src0.Load(src0_offset + slot * nb00 + token * nb01));
        if (id < n_expert) {
            uint prev;
            InterlockedAdd(g_count[id], 1u, prev);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        uint run = 0;
        for (uint e = 0; e < n_expert; ++e) {
            g_cursor[e] = run;
            dst.Store(dst_offset + e * 4u, run);
            run += g_count[e];
        }
        dst.Store(dst_offset + n_expert * 4u, run);
    }
    GroupMemoryBarrierWithGroupSync();

    // Pair order within an expert is unspecified. Each pair owns a distinct
    // output row, so no result depends on it.
    for (uint j = tid; j < total; j += BUCKET_THREADS) {
        uint slot  = j % nei0;
        uint token = j / nei0;
        uint id    = (uint)asint(src0.Load(src0_offset + slot * nb00 + token * nb01));
        if (id < n_expert) {
            uint pos;
            InterlockedAdd(g_cursor[id], 1u, pos);
            dst.Store(dst_offset + op0 + pos * 4u, j);
        }
    }
}
