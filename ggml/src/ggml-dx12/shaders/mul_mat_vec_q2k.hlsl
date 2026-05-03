// mul_mat_vec_q2k.hlsl - Cooperative matrix-vector multiply for Q2_K weights (M=1)
//
// 16 threads cooperate per Q2_K superblock (256 elements).
// Each thread processes 16 elements across 8 scale groups via 4 bit-shifts.
//
// Q2_K block layout (84 bytes per 256 elements):
//   scales[16]: 4-bit scale (low) + 4-bit min (high) per group of 16 elements
//   qs[64]:     2-bit quantized values, 4 elements per byte
//   d (f16):    super-block scale
//   dmin (f16): super-block min
//
// Dispatch: groups_x = N, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE        256
#define QK_K              256
#define Q2K_BSIZE         84    // 16 + 64 + 2 + 2
#define THREADS_PER_BLOCK 16
#define BLOCKS_PER_GROUP  (GROUP_SIZE / THREADS_PER_BLOCK)

groupshared float shared_acc[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i0 = group_id.y * 65535u + group_id.x;  // linearized 2D for large N (>65535)
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    uint lane = tid % THREADS_PER_BLOCK;
    uint block_group = tid / THREADS_PER_BLOCK;
    uint num_blocks = K / QK_K;

    // Thread mapping (matches Vulkan mul_mat_vec_q2_k.comp)
    uint v_im = lane / 8;           // 0 or 1: which 128-element half
    uint v_in = lane % 8;           // 0..7: position within half
    uint l0 = 2 * v_in;             // 0,2,4,...,14
    uint q_offset = 32 * v_im + l0; // byte offset within qs[64]
    uint y_offset = 128 * v_im + l0; // element offset within block

    float acc = 0.0f;

    for (uint bg = block_group; bg < num_blocks; bg += BLOCKS_PER_GROUP) {
        uint block_off = src0_row + bg * Q2K_BSIZE;

        // Load d and dmin (f16 pair at byte offset 80, 4-byte aligned)
        uint dm_raw = src0.Load(block_off + 80);
        float d = f16_to_f32(dm_raw & 0xFFFFu);
        float dmin = f16_to_f32(dm_raw >> 16);

        // Load 8 scale bytes for this half (2 × uint32)
        uint sc_base = block_off + 8 * v_im;
        uint sc_word0 = src0.Load(sc_base);
        uint sc_word1 = src0.Load(sc_base + 4);

        // Extract scale (low 4 bits) and min (high 4 bits) for 8 groups
        float sc0 = float((sc_word0 >>  0) & 0xFu);
        float sc1 = float((sc_word0 >>  8) & 0xFu);
        float sc2 = float((sc_word0 >> 16) & 0xFu);
        float sc3 = float((sc_word0 >> 24) & 0xFu);
        float sc4 = float((sc_word1 >>  0) & 0xFu);
        float sc5 = float((sc_word1 >>  8) & 0xFu);
        float sc6 = float((sc_word1 >> 16) & 0xFu);
        float sc7 = float((sc_word1 >> 24) & 0xFu);

        float mn0 = float((sc_word0 >>  4) & 0xFu);
        float mn1 = float((sc_word0 >> 12) & 0xFu);
        float mn2 = float((sc_word0 >> 20) & 0xFu);
        float mn3 = float((sc_word0 >> 28) & 0xFu);
        float mn4 = float((sc_word1 >>  4) & 0xFu);
        float mn5 = float((sc_word1 >> 12) & 0xFu);
        float mn6 = float((sc_word1 >> 20) & 0xFu);
        float mn7 = float((sc_word1 >> 28) & 0xFu);

        // Load 4 qs bytes packed into uint32:
        // bytes [q_offset, q_offset+1, q_offset+16, q_offset+17]
        // q_offset is always even, qs starts at block_off+16 which is 4-byte aligned
        uint qs_abs = block_off + 16;
        uint addr_a = qs_abs + q_offset;
        uint word_a = src0.Load(addr_a & ~3u);
        uint pair_a = (word_a >> ((addr_a & 2u) * 8u)) & 0xFFFFu;

        uint addr_b = qs_abs + q_offset + 16;
        uint word_b = src0.Load(addr_b & ~3u);
        uint pair_b = (word_b >> ((addr_b & 2u) * 8u)) & 0xFFFFu;

        uint qs_u32 = pair_a | (pair_b << 16);

        // Extract 16 2-bit values via 4 bit-shifts (0, 2, 4, 6)
        // Shift 0: elements at y_offset+{0,1,16,17}
        float q0  = float( qs_u32        & 3u);
        float q1  = float((qs_u32 >>  8) & 3u);
        float q2  = float((qs_u32 >> 16) & 3u);
        float q3  = float((qs_u32 >> 24) & 3u);
        // Shift 2: elements at y_offset+{32,33,48,49}
        float q4  = float((qs_u32 >>  2) & 3u);
        float q5  = float((qs_u32 >> 10) & 3u);
        float q6  = float((qs_u32 >> 18) & 3u);
        float q7  = float((qs_u32 >> 26) & 3u);
        // Shift 4: elements at y_offset+{64,65,80,81}
        float q8  = float((qs_u32 >>  4) & 3u);
        float q9  = float((qs_u32 >> 12) & 3u);
        float q10 = float((qs_u32 >> 20) & 3u);
        float q11 = float((qs_u32 >> 28) & 3u);
        // Shift 6: elements at y_offset+{96,97,112,113}
        float q12 = float((qs_u32 >>  6) & 3u);
        float q13 = float((qs_u32 >> 14) & 3u);
        float q14 = float((qs_u32 >> 22) & 3u);
        float q15 = float((qs_u32 >> 30) & 3u);

        // Load 16 activations as 8 × Load2 (pairs at stride 16 elements = 64 bytes)
        uint y_off = src1_base + (bg * QK_K + y_offset) * 4;
        uint2 a0 = src1.Load2(y_off);
        uint2 a1 = src1.Load2(y_off + 64);
        uint2 a2 = src1.Load2(y_off + 128);
        uint2 a3 = src1.Load2(y_off + 192);
        uint2 a4 = src1.Load2(y_off + 256);
        uint2 a5 = src1.Load2(y_off + 320);
        uint2 a6 = src1.Load2(y_off + 384);
        uint2 a7 = src1.Load2(y_off + 448);

        float by0  = asfloat(a0.x); float by1  = asfloat(a0.y);
        float by2  = asfloat(a1.x); float by3  = asfloat(a1.y);
        float by4  = asfloat(a2.x); float by5  = asfloat(a2.y);
        float by6  = asfloat(a3.x); float by7  = asfloat(a3.y);
        float by8  = asfloat(a4.x); float by9  = asfloat(a4.y);
        float by10 = asfloat(a5.x); float by11 = asfloat(a5.y);
        float by12 = asfloat(a6.x); float by13 = asfloat(a6.y);
        float by14 = asfloat(a7.x); float by15 = asfloat(a7.y);

        // Dot product: sum1 = sum(scale_i * q_i * act_i)
        float sum1 = mad(sc0, mad(q0, by0, q1 * by1),
                     mad(sc1, mad(q2, by2, q3 * by3),
                     mad(sc2, mad(q4, by4, q5 * by5),
                     mad(sc3, mad(q6, by6, q7 * by7),
                     mad(sc4, mad(q8, by8, q9 * by9),
                     mad(sc5, mad(q10, by10, q11 * by11),
                     mad(sc6, mad(q12, by12, q13 * by13),
                         sc7 * mad(q14, by14, q15 * by15))))))));

        // Min compensation: sum2 = sum(min_i * (act_pair_sum))
        float sum2 = mad(mn0, by0 + by1,
                     mad(mn1, by2 + by3,
                     mad(mn2, by4 + by5,
                     mad(mn3, by6 + by7,
                     mad(mn4, by8 + by9,
                     mad(mn5, by10 + by11,
                     mad(mn6, by12 + by13,
                         mn7 * (by14 + by15))))))));

        acc += d * sum1 - dmin * sum2;
    }

    // Wave-intrinsic reduction then cross-wave via shared memory
    float wave_sum = WaveActiveSum(acc);
    uint wave_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
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
