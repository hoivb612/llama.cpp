// mul_mat_vec_glu_q8_0.hlsl - Fused MUL_MAT(W_gate Q8_0) + MUL_MAT(W_up Q8_0) + SwiGLU split
//
// Same fusion as mul_mat_vec_glu_q5_0.hlsl but for Q8_0 weights.
//
// Bindings:
//   src0 (t0): W_gate weights, Q8_0, ne00=K, ne01=N
//   src1 (t1): x       activation, F32, contiguous, ne10=K
//   src2 (t2): W_up    weights, Q8_0, same shape and stride as W_gate.
//              Bound at the resource base; W_up's tensor byte offset is
//              passed in op1.
//   dst  (u0): y       fused output, F32, ne0=N (= GLU split-mode output width)
//
// op_params:
//   op1 = W_up base byte offset (within src2)
//
// Q8_0 block: d(f16=2) + qs[32 signed int8] = 34 bytes per 32 elements.
//
// GROUP_SIZE = 64 fully populates one AMD wave64 (was 32 = half-wave).
// On wave32 devices the group runs as 2 waves and a tree reduction combines
// them; on wave16 as 4 waves.  Each iteration processes BLOCKS_PER_ITER
// blocks in parallel — the wave is split into BLOCKS_PER_ITER × 32-lane
// sub-groups indexed by sub_block.
//
// LDS pre-decode of scales: the 4 × num_blocks d scales (gate0, gate1,
// up0, up1 × every block) are cooperatively loaded into shared memory once
// before the K loop.  Q8_0 has no qh high-bits, so only the d float is
// pre-loaded.  Capped at MAX_BLOCKS = 32 (handles K up to 1024).

#include "ggml_common.hlsli"

#define GROUP_SIZE       64
#define BLOCKS_PER_ITER  2
#define QK8_0            32
#define Q8_0_BSIZE       34
#define MAX_BLOCKS       32

groupshared float  scales_d[4 * MAX_BLOCKS];     // dg0, dg1, du0, du1 per block
groupshared float  shared_acc[256];

float read_f16_v(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

// Read a single signed int8 from the qs payload of a Q8_0 block.
// block_off points at the start of the 34-byte block: 2-byte d, then 32 int8 qs.
int read_q8_0_qs(ByteAddressBuffer buf, uint block_off, uint elem) {
    // qs starts at block_off + 2 (after the f16 d).
    uint qs_off = block_off + 2u + elem;
    uint aligned = qs_off & ~3u;
    uint shift   = (qs_off & 3u) * 8u;
    uint word    = buf.Load(aligned);
    uint b       = (word >> shift) & 0xFFu;
    // sign-extend
    return (int)(b ^ 0x80u) - 128;
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2;
    if (row0 >= ne0) return;
    uint row1 = min(row0 + 1, ne0 - 1);

    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K          = ne00;
    uint num_blocks = K / QK8_0;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint up_base   = op1          + i2_src0 * nb02 + i3_src0 * nb03;
    uint gate_row0 = src0_base + row0 * nb01;
    uint gate_row1 = src0_base + row1 * nb01;
    uint up_row0   = up_base   + row0 * nb01;
    uint up_row1   = up_base   + row1 * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    // Phase A: cooperatively pre-decode all d scales into LDS.
    // Each lane handles one (group, block) entry; with 64 lanes and
    // 4 * num_blocks entries (e.g. 4*18 = 72 for K=576), we issue
    // ceil(4*num_blocks / 64) iterations.
    uint total_entries = 4u * num_blocks;
    for (uint e = local_id; e < total_entries; e += GROUP_SIZE) {
        uint group_idx = e & 3u;            // 0=g0, 1=g1, 2=u0, 3=u1
        uint b         = e >> 2;            // block index
        uint base;
        if (group_idx == 0u) { base = gate_row0; }
        else if (group_idx == 1u) { base = gate_row1; }
        else if (group_idx == 2u) { base = up_row0; }
        else                      { base = up_row1; }
        uint blk_off = base + b * Q8_0_BSIZE;
        if (group_idx < 2u) {
            scales_d[e] = read_f16_v(src0, blk_off);
        } else {
            scales_d[e] = read_f16_v(src2, blk_off);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint elem      = local_id & (QK8_0 - 1);   // 0..31
    uint sub_block = local_id >> 5;            // 0..BLOCKS_PER_ITER-1

    precise float acc_g0 = 0.0f;
    precise float acc_g1 = 0.0f;
    precise float acc_u0 = 0.0f;
    precise float acc_u1 = 0.0f;

    for (uint block = 0; block < num_blocks; block += BLOCKS_PER_ITER) {
        uint b = block + sub_block;
        if (b >= num_blocks) {
            continue;
        }

        uint k = b * QK8_0 + elem;
        float x = asfloat(src1.Load(src1_base + k * 4));

        uint blk_g0 = gate_row0 + b * Q8_0_BSIZE;
        uint blk_g1 = gate_row1 + b * Q8_0_BSIZE;
        uint blk_u0 = up_row0   + b * Q8_0_BSIZE;
        uint blk_u1 = up_row1   + b * Q8_0_BSIZE;

        uint sb = b * 4u;
        float dg0 = scales_d[sb + 0u];
        float dg1 = scales_d[sb + 1u];
        float du0 = scales_d[sb + 2u];
        float du1 = scales_d[sb + 3u];

        int vg0 = read_q8_0_qs(src0, blk_g0, elem);
        int vg1 = read_q8_0_qs(src0, blk_g1, elem);
        int vu0 = read_q8_0_qs(src2, blk_u0, elem);
        int vu1 = read_q8_0_qs(src2, blk_u1, elem);

        acc_g0 += dg0 * float(vg0) * x;
        acc_g1 += dg1 * float(vg1) * x;
        acc_u0 += du0 * float(vu0) * x;
        acc_u1 += du1 * float(vu1) * x;
    }

    float wave_g0 = WaveActiveSum(acc_g0);
    float wave_g1 = WaveActiveSum(acc_g1);
    float wave_u0 = WaveActiveSum(acc_u0);
    float wave_u1 = WaveActiveSum(acc_u1);

    uint wave_id   = local_id / WARP_SIZE;
    uint num_waves = (GROUP_SIZE + WARP_SIZE - 1) / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id]                  = wave_g0;
        shared_acc[num_waves     + wave_id]  = wave_u0;
        shared_acc[num_waves * 2 + wave_id]  = wave_g1;
        shared_acc[num_waves * 3 + wave_id]  = wave_u1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_acc[local_id]                  += shared_acc[local_id + s];
            shared_acc[num_waves     + local_id]  += shared_acc[num_waves     + local_id + s];
            shared_acc[num_waves * 2 + local_id]  += shared_acc[num_waves * 2 + local_id + s];
            shared_acc[num_waves * 3 + local_id]  += shared_acc[num_waves * 3 + local_id + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        float gate0 = shared_acc[0];
        float up0   = shared_acc[num_waves];
        float result0 = (gate0 / (1.0f + exp(-gate0))) * up0;
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float gate1 = shared_acc[num_waves * 2];
            float up1   = shared_acc[num_waves * 3];
            float result1 = (gate1 / (1.0f + exp(-gate1))) * up1;
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
