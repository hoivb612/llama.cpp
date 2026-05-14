// mul_mat_vec_q8_0_mr64.hlsl - AMD wave64 Q8_0 matvec (M=1, 4 rows/group)
//
// Single 64-lane wave per workgroup → no GroupMemoryBarrier and no
// shared memory needed; WaveActiveSum closes the reduction directly.
// Each thread processes one Q8_0 element per K iteration and accumulates
// across 4 output rows in registers, amortising the activation Load.
//
// Targets the SmolLM2 / SmolVLM2 K=576 working point where the existing
// 32-thread mr shader leaves half the AMD wave idle and the 256-thread
// mr256v variant has too much idle cross-wave plumbing for K=576.
//
// Dispatch: groups_x = (N+3)/4, groups_y = 1, groups_z = batch * ne2 * ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 64
#define QK8_0      32
#define Q8_0_BSIZE 34
#define NUM_ROWS   4

float read_f16_v(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * NUM_ROWS;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK8_0;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    uint src0_rows[NUM_ROWS];
    [unroll] for (uint r = 0; r < NUM_ROWS; r++) {
        src0_rows[r] = src0_base + (row0 + r) * nb01;
    }

    precise float acc[NUM_ROWS];
    [unroll] for (uint i = 0; i < NUM_ROWS; i++) acc[i] = 0.0f;

    // Each thread (lane 0..63) advances by GROUP_SIZE elements. A Q8_0 block
    // contains 32 elements, so two consecutive threads cover one block and
    // each thread advances one block per outer step (block += 2 per pair, +1
    // per lane within the block).
    for (uint k = local_id; k < K; k += GROUP_SIZE) {
        float x = asfloat(src1.Load(src1_base + k * 4));
        uint block = k / QK8_0;
        uint qoff_in_block = k - block * QK8_0; // 0..31

        [unroll]
        for (uint r = 0; r < NUM_ROWS; r++) {
            uint boff = src0_rows[r] + block * Q8_0_BSIZE;
            // d is shared across all 32 lanes touching this block; read once
            // per thread. AMD will fold this into an SMEM/scalar load.
            float d = read_f16_v(src0, boff);
            uint qs_off = boff + 2u + qoff_in_block;
            uint raw = (src0.Load(qs_off & ~3u) >> ((qs_off & 3u) * 8u)) & 0xFFu;
            int q = (raw < 128u) ? (int)raw : (int)raw - 256;
            acc[r] = mad(d * float(q), x, acc[r]);
        }
    }

    // Single-wave reduction: GROUP_SIZE == WARP_SIZE on the dispatched-to GPU
    // (gated by wave_size>=64 in the dispatcher). No shared memory, no barrier.
    [unroll]
    for (uint r = 0; r < NUM_ROWS; r++) {
        float s = WaveActiveSum(acc[r]);
        if (local_id == 0 && (row0 + r) < ne0) {
            s += load_fused_bias(row0 + r, i2, i3);
            uint off_d = offset_4d(row0 + r, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d, s, dst_esize);
        }
    }
}
