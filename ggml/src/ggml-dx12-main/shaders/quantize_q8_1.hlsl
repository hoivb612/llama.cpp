// quantize_q8_1.hlsl - Quantize F32 input to Q8_1 format
// One thread group (32 threads) per Q8_1 block
// Q8_1 block (36 bytes): ds(2xf16 packed) + qs[32](int8 packed as 8 x uint32)
//
// src0: F32 input (contiguous)
// dst:  Q8_1 output scratch buffer (contiguous blocks of 36 bytes)
//
// Dispatch: groups_x = total_blocks (K/32 * M * batch)

#include "ggml_common.hlsli"

#define QK8_1 32

groupshared int shared_qs[32];
// Cross-wave reduction slots. Sized for wave size as small as 8 (4 waves max).
// Required because Intel Xe-LPG (B390) reports WaveLaneCount = 16, so the
// 32-thread workgroup spans 2 waves and a single WaveActiveMax/WaveActiveSum
// covers only half the block — producing the wrong per-block scale and the
// wrong 's' field for the second half. AMD/NVIDIA report wave 32 and the
// original single-wave reduction was correct only there.
groupshared float gs_amax[4];
groupshared float gs_qsum[4];

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint block_idx = gid.x;

    // Load F32 input value
    uint src_off = src0_offset + (block_idx * QK8_1 + tid) * 4;
    float val = asfloat(src0.Load(src_off));

    uint lane_count = WaveGetLaneCount();
    uint wave_id    = tid / lane_count;
    uint num_waves  = (32u + lane_count - 1u) / lane_count;

    // Find block max absolute value across all waves in the workgroup.
    float wmax = WaveActiveMax(abs(val));
    if (WaveIsFirstLane()) gs_amax[wave_id] = wmax;
    GroupMemoryBarrierWithGroupSync();
    float amax = gs_amax[0];
    [unroll] for (uint w = 1u; w < 4u; ++w) {
        if (w < num_waves) amax = max(amax, gs_amax[w]);
    }

    // Compute scale (now consistent across the whole 32-element block)
    float d  = amax / 127.0f;
    float id = (d > 0.0f) ? (127.0f / amax) : 0.0f;

    // Quantize to int8
    int q = (int)round(val * id);
    q = clamp(q, -128, 127);
    shared_qs[tid] = q;

    // Compute weighted sum for 's' field across all waves.
    float wsum = WaveActiveSum((float)q);
    if (WaveIsFirstLane()) gs_qsum[wave_id] = wsum;
    GroupMemoryBarrierWithGroupSync();
    float q_sum = gs_qsum[0];
    [unroll] for (uint w2 = 1u; w2 < 4u; ++w2) {
        if (w2 < num_waves) q_sum += gs_qsum[w2];
    }

    // Write Q8_1 block: [ds(4 bytes)][qs(32 bytes)] = 36 bytes total
    uint dst_block = dst_offset + block_idx * 36;

    if (tid == 0) {
        // Write d and s as packed f16x2
        uint d_bits = f32tof16(d);
        uint s_bits = f32tof16(d * q_sum);
        dst.Store(dst_block, d_bits | (s_bits << 16));
    }

    // 8 threads write 8 packed uint32s (4 int8 each)
    if (tid < 8) {
        uint base = tid * 4;
        uint packed = ((uint)(shared_qs[base]   & 0xFF))       |
                      ((uint)(shared_qs[base+1] & 0xFF) <<  8) |
                      ((uint)(shared_qs[base+2] & 0xFF) << 16) |
                      ((uint)(shared_qs[base+3] & 0xFF) << 24);
        dst.Store(dst_block + 4 + tid * 4, packed);
    }
}