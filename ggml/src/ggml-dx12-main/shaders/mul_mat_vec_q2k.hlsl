// mul_mat_vec_q2k.hlsl - Optimized matrix-vector multiply for Q2_K weights (M=1)
//
// Q2_K block (84 bytes): scales[16] + qs[64] + d(f16) + dmin(f16)
//   16 sub-blocks of 16 elements each (256 total per superblock).
//   scales[is]: low 4 bits = scale, high 4 bits = min (per sub-block).
//   qs:        2 bits per element. Layout per dequantize_row_q2_K:
//              for sub-block index `is`:
//                qs_base = (is>>3)*32           // 0 or 32
//                qs_off  = qs_base + (is&1)*16  // start of 16 contiguous qs bytes
//                shift   = ((is>>1) & 3) * 2    // 0,2,4,6
//   Per element l in [0,16):
//                q = (qs[qs_off + l] >> shift) & 3
//                w = d * (scales[is] & 0xF) * q  -  dmin * (scales[is] >> 4)
//
// 256 threads, 16 cooperating per superblock (it_size = 16). 1 output row per group.
//
// Dispatch: groups_x = N, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  256
#endif
#define QK_K        256
#define Q2K_BSIZE   84

groupshared float shared_acc[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i0 = group_x_2d(group_id);
    if (i0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint src0_row  = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    uint it_size = GROUP_SIZE / 16;  // 16 threads per superblock
    uint is = tid % 16;              // sub-block index 0..15
    uint ix = tid / 16;              // superblock stride (interleaved)

    // Per sub-block constants
    uint qs_off  = ((is >> 3) & 1u) * 32u + (is & 1u) * 16u;  // 0..63
    uint shift   = ((is >> 1) & 3u) * 2u;                      // 0,2,4,6

    float acc = 0.0f;

    for (uint blk = ix; blk < num_blocks; blk += it_size) {
        uint block_off = src0_row + blk * Q2K_BSIZE;

        // Load d, dmin from end of block (offsets 80, 82 -> one 4-byte word)
        uint dm_raw = src0.Load(block_off + 80);
        float d    = f16_to_f32(dm_raw & 0xFFFFu);
        float dmin = f16_to_f32(dm_raw >> 16);

        // Load scales[is] byte (16 bytes total, aligned word read + shift)
        uint sc_word = src0.Load(block_off + (is & ~3u));
        uint sc_byte = (sc_word >> ((is & 3u) * 8u)) & 0xFFu;
        float scale_d = d    * float(sc_byte & 0x0Fu);
        float min_v   = dmin * float(sc_byte >> 4);

        // Load 16 qs bytes for this sub-block (4 word loads, 4-byte aligned)
        uint qs_block_off = block_off + 16u + qs_off;
        uint qw0 = src0.Load(qs_block_off + 0);
        uint qw1 = src0.Load(qs_block_off + 4);
        uint qw2 = src0.Load(qs_block_off + 8);
        uint qw3 = src0.Load(qs_block_off + 12);

        // Extract 4 quants from each word at the sub-block's bit shift
        uint qp0 = (qw0 >> shift) & 0x03030303u;
        uint qp1 = (qw1 >> shift) & 0x03030303u;
        uint qp2 = (qw2 >> shift) & 0x03030303u;
        uint qp3 = (qw3 >> shift) & 0x03030303u;

        // Load 16 activation floats (4 Load4 = 64 bytes coalesced)
        uint y_off = src1_base + (blk * QK_K + is * 16u) * 4u;
        uint4 y0 = src1.Load4(y_off + 0);
        uint4 y1 = src1.Load4(y_off + 16);
        uint4 y2 = src1.Load4(y_off + 32);
        uint4 y3 = src1.Load4(y_off + 48);

        float by0  = asfloat(y0.x); float by1  = asfloat(y0.y);
        float by2  = asfloat(y0.z); float by3  = asfloat(y0.w);
        float by4  = asfloat(y1.x); float by5  = asfloat(y1.y);
        float by6  = asfloat(y1.z); float by7  = asfloat(y1.w);
        float by8  = asfloat(y2.x); float by9  = asfloat(y2.y);
        float by10 = asfloat(y2.z); float by11 = asfloat(y2.w);
        float by12 = asfloat(y3.x); float by13 = asfloat(y3.y);
        float by14 = asfloat(y3.z); float by15 = asfloat(y3.w);

        // Per-element q values (cast 2-bit to float)
        float q0  = float((qp0      ) & 0xFFu);
        float q1  = float((qp0 >>  8) & 0xFFu);
        float q2  = float((qp0 >> 16) & 0xFFu);
        float q3  = float((qp0 >> 24) & 0xFFu);
        float q4  = float((qp1      ) & 0xFFu);
        float q5  = float((qp1 >>  8) & 0xFFu);
        float q6  = float((qp1 >> 16) & 0xFFu);
        float q7  = float((qp1 >> 24) & 0xFFu);
        float q8  = float((qp2      ) & 0xFFu);
        float q9  = float((qp2 >>  8) & 0xFFu);
        float q10 = float((qp2 >> 16) & 0xFFu);
        float q11 = float((qp2 >> 24) & 0xFFu);
        float q12 = float((qp3      ) & 0xFFu);
        float q13 = float((qp3 >>  8) & 0xFFu);
        float q14 = float((qp3 >> 16) & 0xFFu);
        float q15 = float((qp3 >> 24) & 0xFFu);

        float sum_qy = q0*by0 + q1*by1 + q2*by2  + q3*by3
                     + q4*by4 + q5*by5 + q6*by6  + q7*by7
                     + q8*by8 + q9*by9 + q10*by10+ q11*by11
                     + q12*by12+q13*by13+q14*by14+q15*by15;

        float sum_y  = by0+by1+by2+by3 + by4+by5+by6+by7
                     + by8+by9+by10+by11 + by12+by13+by14+by15;

        acc += scale_d * sum_qy - min_v * sum_y;
    }

    // Wave-intrinsic reduction (mirrors Q5_K pattern)
    float wave_sum = WaveActiveSum(acc);
    uint wave_id = tid / WARP_SIZE;
    if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    uint num_waves = GROUP_SIZE / WARP_SIZE;
    if (num_waves <= WARP_SIZE) {
        if (tid < num_waves) {
            float v = shared_acc[tid];
            v = WaveActiveSum(v);
            if (tid == 0) shared_acc[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
    } else {
        for (uint s = num_waves / 2; s > 0; s /= 2) {
            if (tid < s) shared_acc[tid] += shared_acc[tid + s];
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (tid == 0) {
        float result = shared_acc[0];
        result += load_fused_bias(i0, i2, i3);
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}
