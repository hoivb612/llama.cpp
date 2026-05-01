// mul_mat_q2k.hlsl - Cooperative batch matmul for Q2_K quantized weights
//
// 16 output rows per group, 16 K-threads cooperate per row (same packed-read
// pattern as the matvec shader).  Each K-thread handles 16 elements per block
// via 5 packed weight loads + 16 activation loads + mad() chains.
//
// Q2_K block layout (84 bytes per 256 elements):
//   scales[16]: 4-bit scale (low) + 4-bit min (high) per group of 16 elements
//   qs[64]:     2-bit quantized values, 4 elements per byte
//   d (f16):    super-block scale
//   dmin (f16): super-block min
//
// Dispatch: groups_x = ceil(N/16), groups_y = M, groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE       256
#define ROWS_PER_GROUP   16
#define THREADS_PER_ROW  16
#define QK_K             256
#define Q2K_BSIZE        84

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row_local = tid / THREADS_PER_ROW;   // 0..15
    uint k_tid     = tid % THREADS_PER_ROW;   // 0..15

    uint i0 = group_id.x * ROWS_PER_GROUP + row_local;
    uint i1 = group_id.y;
    uint i2 = group_id.z % ne2;
    uint i3 = group_id.z / ne2;

    uint N = ne0;
    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row  = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i1 * nb11 + i2 * nb12 + i3 * nb13;

    // Thread mapping within the 16-thread cooperative block (matches matvec)
    uint v_im     = k_tid / 8;            // 0 or 1: which 128-element half
    uint v_in     = k_tid % 8;            // 0..7: position within half
    uint l0       = 2 * v_in;             // 0,2,4,...,14
    uint q_offset = 32 * v_im + l0;       // byte offset within qs[64]
    uint y_offset = 128 * v_im + l0;      // element offset within block

    float acc = 0.0f;

    if (i0 < N) {
        for (uint bg = 0; bg < num_blocks; bg++) {
            uint block_off = src0_row + bg * Q2K_BSIZE;

            // --- Weight decode (5 packed Loads, identical to matvec) ---

            uint dm_raw = src0.Load(block_off + 80);
            float d    = f16_to_f32(dm_raw & 0xFFFFu);
            float dmin = f16_to_f32(dm_raw >> 16);

            uint sc_base  = block_off + 8 * v_im;
            uint sc_word0 = src0.Load(sc_base);
            uint sc_word1 = src0.Load(sc_base + 4);

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

            // Load 4 qs bytes packed into uint32
            uint qs_abs = block_off + 16;
            uint addr_a = qs_abs + q_offset;
            uint word_a = src0.Load(addr_a & ~3u);
            uint pair_a = (word_a >> ((addr_a & 2u) * 8u)) & 0xFFFFu;

            uint addr_b = qs_abs + q_offset + 16;
            uint word_b = src0.Load(addr_b & ~3u);
            uint pair_b = (word_b >> ((addr_b & 2u) * 8u)) & 0xFFFFu;

            uint qs_u32 = pair_a | (pair_b << 16);

            // Extract 16 2-bit values via 4 bit-shifts
            float q0  = float( qs_u32        & 3u);
            float q1  = float((qs_u32 >>  8) & 3u);
            float q2  = float((qs_u32 >> 16) & 3u);
            float q3  = float((qs_u32 >> 24) & 3u);
            float q4  = float((qs_u32 >>  2) & 3u);
            float q5  = float((qs_u32 >> 10) & 3u);
            float q6  = float((qs_u32 >> 18) & 3u);
            float q7  = float((qs_u32 >> 26) & 3u);
            float q8  = float((qs_u32 >>  4) & 3u);
            float q9  = float((qs_u32 >> 12) & 3u);
            float q10 = float((qs_u32 >> 20) & 3u);
            float q11 = float((qs_u32 >> 28) & 3u);
            float q12 = float((qs_u32 >>  6) & 3u);
            float q13 = float((qs_u32 >> 14) & 3u);
            float q14 = float((qs_u32 >> 22) & 3u);
            float q15 = float((qs_u32 >> 30) & 3u);

            // --- Load 16 activations (scattered positions within K-block) ---
            uint y_base = src1_base + (bg * QK_K + y_offset) * src1_esize;
            float by0  = load_auto(src1, y_base,                    src1_esize);
            float by1  = load_auto(src1, y_base +      src1_esize,  src1_esize);
            float by2  = load_auto(src1, y_base + 16 * src1_esize,  src1_esize);
            float by3  = load_auto(src1, y_base + 17 * src1_esize,  src1_esize);
            float by4  = load_auto(src1, y_base + 32 * src1_esize,  src1_esize);
            float by5  = load_auto(src1, y_base + 33 * src1_esize,  src1_esize);
            float by6  = load_auto(src1, y_base + 48 * src1_esize,  src1_esize);
            float by7  = load_auto(src1, y_base + 49 * src1_esize,  src1_esize);
            float by8  = load_auto(src1, y_base + 64 * src1_esize,  src1_esize);
            float by9  = load_auto(src1, y_base + 65 * src1_esize,  src1_esize);
            float by10 = load_auto(src1, y_base + 80 * src1_esize,  src1_esize);
            float by11 = load_auto(src1, y_base + 81 * src1_esize,  src1_esize);
            float by12 = load_auto(src1, y_base + 96 * src1_esize,  src1_esize);
            float by13 = load_auto(src1, y_base + 97 * src1_esize,  src1_esize);
            float by14 = load_auto(src1, y_base + 112 * src1_esize, src1_esize);
            float by15 = load_auto(src1, y_base + 113 * src1_esize, src1_esize);

            // --- Dot product (same mad() chains as matvec) ---
            float sum1 = mad(sc0, mad(q0, by0, q1 * by1),
                         mad(sc1, mad(q2, by2, q3 * by3),
                         mad(sc2, mad(q4, by4, q5 * by5),
                         mad(sc3, mad(q6, by6, q7 * by7),
                         mad(sc4, mad(q8, by8, q9 * by9),
                         mad(sc5, mad(q10, by10, q11 * by11),
                         mad(sc6, mad(q12, by12, q13 * by13),
                             sc7 * mad(q14, by14, q15 * by15))))))));

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
    }

    // Butterfly reduction across 16 K-threads (XOR stays within 16-lane block)
    uint wave_lane = WaveGetLaneIndex();
    [unroll]
    for (uint offset = 1; offset < THREADS_PER_ROW; offset <<= 1) {
        acc += WaveReadLaneAt(acc, wave_lane ^ offset);
    }

    if (k_tid == 0 && i0 < N) {
        uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, acc, dst_esize);
    }
}
