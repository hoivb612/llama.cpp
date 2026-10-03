// quantize_q8_1.hlsl - Quantize F32 input to Q8_1 format
// One thread group (32 threads) per Q8_1 block
// Q8_1 block (36 bytes): ds(2xf16 packed) + qs[32](int8 packed as 8 x uint32)
//
// src0: F32 input (contiguous)
// dst:  Q8_1 output scratch buffer (contiguous blocks of 36 bytes)
//
// Dispatch: groups_x <= 65535, groups_y = ceil(total_blocks/groups_x)

#include "ggml_common.hlsli"

#define QK8_1 32

// Per-lane storage for amax / qsum reduction. We avoid wave intrinsics
// (WaveActiveMax / WaveActiveSum) entirely because AMD wave64 with the
// 32-thread workgroup runs as a single wave with only 32 of 64 lanes
// active. Driver-side handling of inactive lanes for those reductions
// produced a small but model-fatal bias vs CPU when Q8_1 was consumed
// by dp4a matvecs (DX12_FORCE_DP4A_WAVE64 retest, 2026-05-13).
// Pure shared-memory reductions with explicit barriers are deterministic
// across all wave widths.
groupshared float gs_aval[32];
groupshared int   gs_q[32];

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint block_idx = gid.x + gid.y * 65535u;
    if (block_idx >= ne0) {
        return;
    }

    // Load F32 input value
    uint src_off = src0_offset + (block_idx * QK8_1 + tid) * 4;
    float val = asfloat(src0.Load(src_off));

    // amax reduction via shared memory tree.
    gs_aval[tid] = abs(val);
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 16u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            gs_aval[tid] = max(gs_aval[tid], gs_aval[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float amax = gs_aval[0];

    // Compute scale (now consistent across the whole 32-element block)
    float d  = amax / 127.0f;
    float id = (d > 0.0f) ? (127.0f / amax) : 0.0f;

    // Quantize to int8
    int q = (int)round(val * id);
    q = clamp(q, -128, 127);
    gs_q[tid] = q;
    GroupMemoryBarrierWithGroupSync();

    // qsum reduction via shared memory tree (integer, deterministic).
    [unroll] for (uint stride2 = 16u; stride2 > 0u; stride2 >>= 1) {
        if (tid < stride2) {
            gs_q[tid] = gs_q[tid] + gs_q[tid + stride2];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    int q_sum = gs_q[0];

    // Write Q8_1 block: [ds(4 bytes)][qs(32 bytes)] = 36 bytes total
    uint dst_block = dst_offset + block_idx * 36;

    if (tid == 0) {
        uint d_bits = f32tof16(d);
        uint s_bits = f32tof16(d * float(q_sum));
        dst.Store(dst_block, d_bits | (s_bits << 16));
    }

    // Re-store q values to shared (the reduction overwrote gs_q[0..15]).
    // Recompute and write out the int8 packed qs.
    gs_q[tid] = q;
    GroupMemoryBarrierWithGroupSync();

    // 8 threads write 8 packed uint32s (4 int8 each)
    if (tid < 8) {
        uint base = tid * 4;
        uint packed = ((uint)(gs_q[base]   & 0xFF))       |
                      ((uint)(gs_q[base+1] & 0xFF) <<  8) |
                      ((uint)(gs_q[base+2] & 0xFF) << 16) |
                      ((uint)(gs_q[base+3] & 0xFF) << 24);
        dst.Store(dst_block + 4 + tid * 4, packed);
    }
}