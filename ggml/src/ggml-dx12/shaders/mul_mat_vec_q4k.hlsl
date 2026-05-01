// mul_mat_vec_q4k.hlsl - Optimized matrix-vector multiply for Q4_K weights (M=1)
//
// Matches Vulkan mul_mat_vec_q4_k.comp approach:
// - 16 threads cooperate per Q4_K block
// - Vectorized uint32 loads for qs (2 loads → 16 elements)
// - Packed scale decoding via bitwise ops
// - 256 threads / group, 1 output row / group
//
// Dispatch: groups_x = N, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q4K_BSIZE   144   // 2+2+12+128 bytes per Q4_K block

groupshared float shared_acc[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i0 = group_id.x;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    // 16 threads cooperate per Q4_K block (matching Vulkan)
    uint it_size = GROUP_SIZE / 16;  // 16 block groups
    uint itid = tid % 16;           // thread index within block (0..15)
    uint ix = tid / 16;             // which block group (0..15)

    // Map thread to Q4_K sub-block structure
    uint il = itid / 4;             // 0..3 (which chunk pair)
    uint ir = itid % 4;             // 0..3 (position within chunk)
    uint v_im = il / 2;             // 0 or 1 (which 128-element half)
    uint v_in = il % 2;             // 0 or 1 (which sub-chunk)

    uint l0 = 4 * (2 * ir + v_in);  // starting element: 0,4,8,...,28
    uint q_offset = 32 * v_im + l0; // qs byte offset within qs[128]
    uint y_offset = 64 * v_im + l0; // activation element offset within block

    float acc = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        uint block_off = src0_row + block_idx * Q4K_BSIZE;

        // Load block header (d, dmin as packed f16 pair)
        uint dm_raw = src0.Load(block_off);
        float dall = f16_to_f32(dm_raw & 0xFFFFu);
        float dmin = f16_to_f32(dm_raw >> 16);

        // Load and decode scales (12 bytes at block_off + 4)
        // v_im selects which pair of uint16 values within each uint32 load
        uint scales_off = block_off + 4;
        uint s_raw0, s_raw4, s_raw8;
        if (v_im == 0) {
            s_raw0 = src0.Load(scales_off) & 0xFFFFu;
            s_raw4 = src0.Load(scales_off + 4) & 0xFFFFu;
            s_raw8 = src0.Load(scales_off + 8) & 0xFFFFu;
        } else {
            s_raw0 = (src0.Load(scales_off) >> 16) & 0xFFFFu;
            s_raw4 = (src0.Load(scales_off + 4) >> 16) & 0xFFFFu;
            s_raw8 = (src0.Load(scales_off + 8) >> 16) & 0xFFFFu;
        }

        // Decode 6-bit scales and mins (Vulkan approach)
        uint scale_0_4_l = (s_raw4 << 16) | s_raw0;
        uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2;

        float sc0 = float((scale_0_4_l >>  0) & 0x3Fu);
        float sc1 = float((scale_0_4_l >>  8) & 0x3Fu);
        float sc2 = float((scale_0_4_l >> 16) & 0x3Fu);
        float sc3 = float((scale_0_4_l >> 24) & 0x3Fu);

        uint combined_8 = (((s_raw8 << 12) | s_raw8) & 0x0F0F0F0Fu) | scale_0_4_h;
        float sc4 = float((combined_8 >>  0) & 0xFFu);
        float sc5 = float((combined_8 >>  8) & 0xFFu);
        float sc6 = float((combined_8 >> 16) & 0xFFu);
        float sc7 = float((combined_8 >> 24) & 0xFFu);

        // Load 2 × uint32 for qs data (8 bytes = 16 nibbles)
        uint qs_off = block_off + 16;  // qs starts after 4-byte header + 12-byte scales
        uint qs0 = src0.Load(qs_off + q_offset);
        uint qs64 = src0.Load(qs_off + q_offset + 64);

        // Extract 16 4-bit values
        float q0  = float((qs0  >>  0) & 0xFu);
        float q1  = float((qs0  >>  8) & 0xFu);
        float q2  = float((qs0  >> 16) & 0xFu);
        float q3  = float((qs0  >> 24) & 0xFu);
        float q4  = float((qs0  >>  4) & 0xFu);
        float q5  = float((qs0  >> 12) & 0xFu);
        float q6  = float((qs0  >> 20) & 0xFu);
        float q7  = float((qs0  >> 28) & 0xFu);
        float q8  = float((qs64 >>  0) & 0xFu);
        float q9  = float((qs64 >>  8) & 0xFu);
        float q10 = float((qs64 >> 16) & 0xFu);
        float q11 = float((qs64 >> 24) & 0xFu);
        float q12 = float((qs64 >>  4) & 0xFu);
        float q13 = float((qs64 >> 12) & 0xFu);
        float q14 = float((qs64 >> 20) & 0xFu);
        float q15 = float((qs64 >> 28) & 0xFu);

        // Load 16 activation values using Load4 (128-bit reads, 4× bandwidth vs scalar)
        uint y1_off = src1_base + (block_idx * QK_K + y_offset) * 4;
        uint y2_off = y1_off + 128 * 4;

        uint4 a0 = src1.Load4(y1_off);
        uint4 a1 = src1.Load4(y1_off + 128);
        uint4 a2 = src1.Load4(y2_off);
        uint4 a3 = src1.Load4(y2_off + 128);

        float by0  = asfloat(a0.x); float by1  = asfloat(a0.y);
        float by2  = asfloat(a0.z); float by3  = asfloat(a0.w);
        float by4  = asfloat(a1.x); float by5  = asfloat(a1.y);
        float by6  = asfloat(a1.z); float by7  = asfloat(a1.w);
        float by8  = asfloat(a2.x); float by9  = asfloat(a2.y);
        float by10 = asfloat(a2.z); float by11 = asfloat(a2.w);
        float by12 = asfloat(a3.x); float by13 = asfloat(a3.y);
        float by14 = asfloat(a3.z); float by15 = asfloat(a3.w);

        // Compute partial dot products using mad() FMA chains for better ILP
        float sx = mad(q0, by0, mad(q1, by1, mad(q2, by2, q3*by3)));
        float sy = mad(q4, by4, mad(q5, by5, mad(q6, by6, q7*by7)));
        float sz = mad(q8, by8, mad(q9, by9, mad(q10, by10, q11*by11)));
        float sw = mad(q12, by12, mad(q13, by13, mad(q14, by14, q15*by15)));

        // Min compensation with mad() chains
        float smin = mad(sc2, by0+by1+by2+by3, mad(sc3, by4+by5+by6+by7,
                     mad(sc6, by8+by9+by10+by11, sc7*(by12+by13+by14+by15))));

        acc += dall * mad(sx, sc0, mad(sy, sc1, mad(sz, sc4, sw*sc5))) - dmin * smin;
    }

    // Wave-intrinsic reduction (2 barriers instead of 8)
    float wave_sum = WaveActiveSum(acc);
    uint wave_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    uint num_waves = GROUP_SIZE / WaveGetLaneCount();
    if (tid < num_waves) {
        float v = shared_acc[tid];
        v = WaveActiveSum(v);
        if (tid == 0) shared_acc[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float result = shared_acc[0];
        if (op0 == 1u) result += asfloat(src2.Load(op1 + i0 * 4));
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}
