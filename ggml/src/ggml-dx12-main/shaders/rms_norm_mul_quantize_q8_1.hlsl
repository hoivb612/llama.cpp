// rms_norm_mul_quantize_q8_1.hlsl
//
// Fused RMS_NORM + MUL (rms_norm_mul) that ALSO writes a Q8_1-quantized copy
// of the normalized output to a secondary UAV (u1, the shared `temp` buffer).
// Eliminates the standalone quantize_q8_1 pre-pass that the downstream dp4a
// MUL_MAT would otherwise have dispatched. The F32 destination (u0) is still
// populated so any non-dp4a consumer of the rms_norm_mul output works
// unchanged.
//
// Inputs:
//   src0 (t0): raw F32 input vector (pre-norm)
//   src1 (t1): learned RMS_NORM weight (broadcast along inner dim)
// Outputs:
//   dst  (u0): F32 normalized * weight  (same layout as rms_norm_mul)
//   temp (u1): Q8_1 quantized rows, contiguous blocks of 36 bytes, row-major
//              (matches stock quantize_q8_1.hlsl block layout: global_block =
//              row * (ne00 / 32) + block_within_row)
//
// op_param[0]: epsilon (float, same as rms_norm_mul.hlsl)
// op_param[1]: skip_f32_dst flag — when non-zero, the F32 dst store is
//              elided. Dispatcher sets this only when all consumers of the
//              rms_norm_mul output are downstream dp4a matmuls that will
//              consume the Q8_1 scratch via bctx->q8_1_scratch cache (and
//              no intermediate dispatch can invalidate that cache). Saves
//              ne00 * sizeof(float) bytes of bandwidth per row.
//
// Dispatch: one workgroup per (i1, i2, i3) row, 256 threads each.
//
// Phase 2 reduction strategy:
//   - Each Q8_1 block is 32 lanes. With 256-thread workgroups, the per-block
//     reduction sits within a single wave on AMD wave64 (2 blocks/wave) and
//     NVIDIA/Intel wave32 (1 block/wave). We use a manual 32-lane butterfly
//     via WaveReadLaneAt (a deterministic lane shuffle, not a wave-reduction
//     intrinsic) so we don't trip the inactive-lane bias issue documented in
//     quantize_q8_1.hlsl. All 256 lanes are active, so the shuffle is exact.
//   - For wave=16 (Intel UHD/Arc-class), a 32-lane block spans 2 waves, so
//     WaveReadLaneAt across waves is undefined. Fall back to a shared-memory
//     reduction with one barrier per stride (same as v1; still a win since
//     wave=16 path is rare on iGPUs of interest).
//   - Phase 1 (SOS reduction) is identical to rms_norm_mul.hlsl.

#include "ggml_common.hlsli"

// q8_1 output uses the shared `temp : register(u1)` UAV declared in
// ggml_common.hlsli (also used by split-KV flash attention; bindings are
// per-dispatch so the two paths never collide).

groupshared float wave_sums[32];

#if !defined(WAVE_SIZE) || (WAVE_SIZE < 32 && WAVE_SIZE != 16)
// Shared-memory fallback for sub-32 waves not covered by the wave16 path
// below: a 32-lane Q8_1 block spans several waves, so WaveReadLaneAt cannot
// reach across it.
groupshared uint  gs_amax_bits[256];
groupshared int   gs_qsum[256];
#endif

WAVE_SIZE_ATTR
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint total_rows = ne1 * ne2 * ne3;
    if (row >= total_rows) return;

    uint i3 = row / (ne1 * ne2);
    uint rem = row % (ne1 * ne2);
    uint i2 = rem / ne1;
    uint i1 = rem % ne1;

    float eps = op_param_f32(0);
    uint skip_f32_dst = op_param_uint(1);
    uint local_id = gtid.x;
    uint lane_count = WaveGetLaneCount();
    uint wave_count = (256 + lane_count - 1) / lane_count;
    uint wave_id = local_id / lane_count;

    // --------------------------------------------------------------------
    // Phase 1: sum-of-squares reduction (identical to rms_norm_mul.hlsl)
    // --------------------------------------------------------------------
    precise float local_sum = 0.0f;
    for (uint i0 = local_id; i0 < ne00; i0 += 256) {
        uint off = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        float val = load_auto(src0, off, src0_esize);
        local_sum += val * val;
    }

    float wave_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) wave_sums[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    float total = 0.0f;
    if (local_id == 0) {
        float acc = wave_sums[0];
        for (uint w = 1; w < wave_count; ++w) acc += wave_sums[w];
        wave_sums[0] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
    total = wave_sums[0];

    float rms = sqrt(total / (float)ne00 + eps);
    float scale_val = 1.0f / rms;

    // --------------------------------------------------------------------
    // Phase 2: chunked Q8_1 quantize (8 blocks of 32 elements per chunk).
    // For wave>=32, per-block reductions use WaveReadLaneAt butterfly with
    // NO group syncs. For wave<32, fall back to shared-memory reduction.
    // --------------------------------------------------------------------
    uint num_blocks     = ne00 / 32;

#if defined(WAVE_SIZE) && (WAVE_SIZE == 16)
    // wave16 (Intel Xe): give each wave one whole Q8_1 block, 16 lanes x 2
    // elements, so both per-block reductions stay inside a single wave. The
    // shared-memory butterfly used by the generic sub-32 path needs 10
    // GroupMemoryBarrierWithGroupSync per chunk, which dominates the shader
    // on this hardware.
    uint w_id   = local_id / 16u;   // wave index == block within chunk
    uint w_lane = local_id & 15u;   // == WaveGetLaneIndex() for 256/16 groups

    for (uint chunk_start = 0; chunk_start < num_blocks; chunk_start += 16u) {
        uint block_idx = chunk_start + w_id;
        bool in_range = (block_idx < num_blocks);

        float n0 = 0.0f;
        float n1 = 0.0f;
        if (in_range) {
            uint i0 = block_idx * 32u + w_lane * 2u;
            [unroll] for (uint e = 0u; e < 2u; ++e) {
                uint off_src = offset_4d(i0 + e, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
                uint off_wt  = offset_4d((i0 + e) % ne10, i1 % ne11, i2 % ne12, i3 % ne13,
                                          nb10, nb11, nb12, nb13, src1_offset);
                float val = load_auto(src0, off_src, src0_esize);
                float wt  = load_auto(src1, off_wt, src1_esize);
                float nv  = val * scale_val * wt;
                if (e == 0u) { n0 = nv; } else { n1 = nv; }
                if (skip_f32_dst == 0u) {
                    uint off_dst = offset_4d(i0 + e, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                    store_auto(dst, off_dst, nv, dst_esize);
                }
            }
        }

        float amax      = WaveActiveMax(max(abs(n0), abs(n1)));
        float d_scale   = amax / 127.0f;
        float inv_scale = (d_scale > 0.0f) ? (127.0f / amax) : 0.0f;
        int q0 = clamp((int)round(n0 * inv_scale), -128, 127);
        int q1 = clamp((int)round(n1 * inv_scale), -128, 127);
        if (!in_range) { q0 = 0; q1 = 0; }
        int qsum = WaveActiveSum(q0 + q1);

        uint global_block = row * num_blocks + block_idx;
        uint q8_off = global_block * 36u;

        if (in_range && w_lane == 0u) {
            uint d_bits = f32tof16(d_scale);
            uint s_bits = f32tof16(d_scale * float(qsum));
            temp.Store(q8_off, d_bits | (s_bits << 16));
        }

        // Pack 4 quants per word: lanes (2L, 2L+1) supply word L. WaveReadLaneAt
        // must be reached from wave-uniform control flow, so evaluate it on
        // every lane and gate only the store.
        int p0 = WaveReadLaneAt(q0, (w_lane + 1u) & 15u);
        int p1 = WaveReadLaneAt(q1, (w_lane + 1u) & 15u);
        if (in_range && (w_lane & 1u) == 0u) {
            uint packed = (((uint)q0 & 0xFFu))       |
                          (((uint)q1 & 0xFFu) <<  8) |
                          (((uint)p0 & 0xFFu) << 16) |
                          (((uint)p1 & 0xFFu) << 24);
            temp.Store(q8_off + 4u + (w_lane >> 1) * 4u, packed);
        }
    }
#else
    uint block_in_chunk = local_id / 32;   // 0..7
    uint lane_in_block  = local_id & 31;   // 0..31
#if !defined(WAVE_SIZE) || (WAVE_SIZE < 32)
    uint chunk_base     = block_in_chunk * 32;
#endif

    for (uint chunk_start = 0; chunk_start < num_blocks; chunk_start += 8) {
        uint block_idx = chunk_start + block_in_chunk;
        bool in_range = (block_idx < num_blocks);

        // Compute normalized * weight for this lane's element and write F32.
        float normed = 0.0f;
        if (in_range) {
            uint i0 = block_idx * 32 + lane_in_block;
            uint off_src = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
            uint off_wt  = offset_4d(i0 % ne10, i1 % ne11, i2 % ne12, i3 % ne13,
                                      nb10, nb11, nb12, nb13, src1_offset);
            uint off_dst = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            float val = load_auto(src0, off_src, src0_esize);
            float wt  = load_auto(src1, off_wt, src1_esize);
            normed = val * scale_val * wt;
            if (skip_f32_dst == 0u) {
                store_auto(dst, off_dst, normed, dst_esize);
            }
        }

        // ---- per-block (32-lane) amax reduction ----
        float amax = abs(normed);
#if defined(WAVE_SIZE) && (WAVE_SIZE >= 32)
        // Compute per-32-lane-block amax using WaveActiveMax with a
        // predicate-masked input. All 64 lanes participate (wave-uniform call),
        // but lanes outside the target block contribute 0 (amax is always
        // >= 0 so 0 is the natural identity). For wave=32, all lanes are in
        // the same block, so amax_block0 == amax_block1 == wave max.
        //
        // This avoids manual WaveReadLaneAt butterflies entirely. On AMD
        // wave=64, the manual butterfly produced incorrect results in
        // real-model inference (test-backend-ops never exercises this op
        // chain, so the bug only surfaced on Phi-3 Q4_K_M text generation).
        // WaveActiveMax is a single instruction cascade and avoids any
        // cross-lane index trouble.
        uint wave_lane_a = WaveGetLaneIndex();
        float amax_block0 = WaveActiveMax((wave_lane_a < 32u) ? amax : 0.0f);
        float amax_block1 = WaveActiveMax((wave_lane_a >= 32u) ? amax : 0.0f);
        amax = (wave_lane_a < 32u) ? amax_block0 : amax_block1;
#else
        gs_amax_bits[local_id] = asuint(amax);
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint s = 16u; s > 0u; s >>= 1) {
            if (lane_in_block < s) {
                uint a = gs_amax_bits[chunk_base + lane_in_block];
                uint b = gs_amax_bits[chunk_base + lane_in_block + s];
                gs_amax_bits[chunk_base + lane_in_block] = asuint(max(asfloat(a), asfloat(b)));
            }
            GroupMemoryBarrierWithGroupSync();
        }
        amax = asfloat(gs_amax_bits[chunk_base]);
#endif

        float d_scale  = amax / 127.0f;
        float inv_scale = (d_scale > 0.0f) ? (127.0f / amax) : 0.0f;
        int q = (int)round(normed * inv_scale);
        q = clamp(q, -128, 127);
        if (!in_range) q = 0;

        // ---- per-block (32-lane) qsum reduction ----
        int qsum;
#if defined(WAVE_SIZE) && (WAVE_SIZE >= 32)
        // Same predicate-masking pattern as the amax reduction above.
        uint wave_lane_b = WaveGetLaneIndex();
        int qsum_block0 = WaveActiveSum((wave_lane_b < 32u) ? q : 0);
        int qsum_block1 = WaveActiveSum((wave_lane_b >= 32u) ? q : 0);
        qsum = (wave_lane_b < 32u) ? qsum_block0 : qsum_block1;
#else
        gs_qsum[local_id] = q;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint s2 = 16u; s2 > 0u; s2 >>= 1) {
            if (lane_in_block < s2) {
                gs_qsum[chunk_base + lane_in_block] =
                    gs_qsum[chunk_base + lane_in_block] + gs_qsum[chunk_base + lane_in_block + s2];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        qsum = gs_qsum[chunk_base];
#endif

        // ---- write Q8_1 block (36 bytes per block, row-major) ----
        uint global_block = row * num_blocks + block_idx;
        uint q8_off = global_block * 36;

        if (in_range && lane_in_block == 0) {
            uint d_bits = f32tof16(d_scale);
            uint s_bits = f32tof16(d_scale * float(qsum));
            temp.Store(q8_off, d_bits | (s_bits << 16));
        }

#if defined(WAVE_SIZE) && (WAVE_SIZE >= 32)
        // Pack 4 int8 quants per uint32 via lane shuffle (no shared mem).
        //
        // CRITICAL: WaveReadLaneAt must be called from wave-uniform control flow.
        // Calling it inside `if ((lane_in_block & 3u) == 0u)` would make 3 of every
        // 4 lanes inactive, and reading from an inactive lane returns undefined
        // values. We therefore evaluate the lane shuffles unconditionally for all
        // lanes (cheap on AMD/NV — issued as a single ds_bpermute / shfl.idx /
        // ReadFirstLane sequence) and gate only the store on the pack-leader
        // predicate. Source lanes are masked into the 32-lane block for safety
        // (wave_lane3=60 reads 61/62/63, all in block; mask is a no-op there but
        // protects the upper block on wave64 generically).
        uint wave_lane3 = WaveGetLaneIndex();
        uint block_base = wave_lane3 & ~31u;
        int q1 = WaveReadLaneAt(q, block_base | ((wave_lane3 + 1u) & 31u));
        int q2 = WaveReadLaneAt(q, block_base | ((wave_lane3 + 2u) & 31u));
        int q3 = WaveReadLaneAt(q, block_base | ((wave_lane3 + 3u) & 31u));
        if (in_range && (lane_in_block & 3u) == 0u) {
            uint packed = ((uint)(q  & 0xFF))       |
                          ((uint)(q1 & 0xFF) <<  8) |
                          ((uint)(q2 & 0xFF) << 16) |
                          ((uint)(q3 & 0xFF) << 24);
            temp.Store(q8_off + 4 + (lane_in_block >> 2) * 4, packed);
        }
#else
        // Shared-mem path: restore q values (qsum reduction overwrote them).
        gs_qsum[local_id] = q;
        GroupMemoryBarrierWithGroupSync();
        if (in_range && lane_in_block < 8) {
            uint base = chunk_base + lane_in_block * 4;
            uint packed = ((uint)(gs_qsum[base]   & 0xFF))       |
                          ((uint)(gs_qsum[base+1] & 0xFF) <<  8) |
                          ((uint)(gs_qsum[base+2] & 0xFF) << 16) |
                          ((uint)(gs_qsum[base+3] & 0xFF) << 24);
            temp.Store(q8_off + 4 + lane_in_block * 4, packed);
        }
        GroupMemoryBarrierWithGroupSync();
#endif
    }
#endif  // WAVE_SIZE == 16
}
