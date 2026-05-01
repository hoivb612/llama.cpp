// mul_mat_q3k.hlsl - Cooperative batch matmul for Q3_K quantized weights
//
// 16 output rows per group, 16 K-threads cooperate per row (same packed-read
// pattern as the matvec shader).  Each K-thread handles 16 elements per block
// via packed weight loads + hmask correction + mad() chains.
//
// Q3_K block layout (110 bytes per 256 elements, NOT 4-byte aligned):
//   hmask[32]: high bit (1 bit per element, bit-packed)
//   qs[64]:    low 2 bits (4 elements per byte)
//   scales[12]: 6-bit scales, packed
//   d (f16):   super-block scale (NO dmin)
//
// Dispatch: groups_x = ceil(N/16), groups_y = M, groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE       256
#define ROWS_PER_GROUP   16
#define THREADS_PER_ROW  16
#define QK_K             256
#define Q3K_BSIZE        110

// Misalignment-safe uint32 load (Q3_K block size 110 is not 4-byte aligned)
uint load_u32(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift   = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row_local = tid / THREADS_PER_ROW;
    uint k_tid     = tid % THREADS_PER_ROW;

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

    // Thread mapping (same as matvec)
    uint v_im  = k_tid / 8;
    uint v_in  = k_tid % 8;
    uint v_im4 = v_im * 4;
    uint l0       = 2 * v_in;
    uint q_offset = 32 * v_im + l0;
    uint y_offset = 128 * v_im + l0;

    // Pre-compute hmask bit masks for 4 shift groups
    uint m_base = 0x01010101u << v_im4;
    uint hm_m0  = m_base;
    uint hm_m1  = m_base << 1;
    uint hm_m2  = m_base << 2;
    uint hm_m3  = m_base << 3;

    float acc = 0.0f;

    if (i0 < N) {
        for (uint bg = 0; bg < num_blocks; bg++) {
            uint block_off = src0_row + bg * Q3K_BSIZE;

            // --- Weight decode (identical to matvec) ---

            // d (f16 at byte 108, potentially misaligned)
            uint d_word = load_u32(src0, block_off + 108);
            float d_all = f16_to_f32(d_word & 0xFFFFu);

            // 6-bit scales from 12 bytes at offset 96
            uint sc_base = block_off + 96;
            uint sc0_raw = load_u32(src0, sc_base);
            uint sc1_raw = load_u32(src0, sc_base + 4);
            uint sc2_raw = load_u32(src0, sc_base + 8);

            uint aux0 = (sc0_raw & 0x0F0F0F0Fu) | (((sc2_raw >> 0) & 0x03030303u) << 4);
            uint aux1 = (sc1_raw & 0x0F0F0F0Fu) | (((sc2_raw >> 2) & 0x03030303u) << 4);
            uint aux2 = ((sc0_raw >> 4) & 0x0F0F0F0Fu) | (((sc2_raw >> 4) & 0x03030303u) << 4);
            uint aux3 = ((sc1_raw >> 4) & 0x0F0F0F0Fu) | (((sc2_raw >> 6) & 0x03030303u) << 4);

            uint sc_lo = (v_im == 0) ? aux0 : aux2;
            uint sc_hi = (v_im == 0) ? aux1 : aux3;

            float s0 = d_all * (float(int( sc_lo        & 0x3Fu)) - 32.0f);
            float s1 = d_all * (float(int((sc_lo >>  8) & 0x3Fu)) - 32.0f);
            float s2 = d_all * (float(int((sc_lo >> 16) & 0x3Fu)) - 32.0f);
            float s3 = d_all * (float(int((sc_lo >> 24) & 0x3Fu)) - 32.0f);
            float s4 = d_all * (float(int( sc_hi        & 0x3Fu)) - 32.0f);
            float s5 = d_all * (float(int((sc_hi >>  8) & 0x3Fu)) - 32.0f);
            float s6 = d_all * (float(int((sc_hi >> 16) & 0x3Fu)) - 32.0f);
            float s7 = d_all * (float(int((sc_hi >> 24) & 0x3Fu)) - 32.0f);

            // qs (2-bit low values at offset 32)
            uint qs_abs = block_off + 32;
            uint addr_a = qs_abs + q_offset;
            uint pair_a = load_u32(src0, addr_a) & 0xFFFFu;
            uint addr_b = qs_abs + q_offset + 16;
            uint pair_b = load_u32(src0, addr_b) & 0xFFFFu;
            uint qs_u32 = pair_a | (pair_b << 16);

            float ql0  = float( qs_u32        & 3u);
            float ql1  = float((qs_u32 >>  8) & 3u);
            float ql2  = float((qs_u32 >> 16) & 3u);
            float ql3  = float((qs_u32 >> 24) & 3u);
            float ql4  = float((qs_u32 >>  2) & 3u);
            float ql5  = float((qs_u32 >> 10) & 3u);
            float ql6  = float((qs_u32 >> 18) & 3u);
            float ql7  = float((qs_u32 >> 26) & 3u);
            float ql8  = float((qs_u32 >>  4) & 3u);
            float ql9  = float((qs_u32 >> 12) & 3u);
            float ql10 = float((qs_u32 >> 20) & 3u);
            float ql11 = float((qs_u32 >> 28) & 3u);
            float ql12 = float((qs_u32 >>  6) & 3u);
            float ql13 = float((qs_u32 >> 14) & 3u);
            float ql14 = float((qs_u32 >> 22) & 3u);
            float ql15 = float((qs_u32 >> 30) & 3u);

            // hmask correction (offset 0)
            uint hm_pair_a = load_u32(src0, block_off + l0) & 0xFFFFu;
            uint hm_pair_b = load_u32(src0, block_off + l0 + 16) & 0xFFFFu;
            uint hmk = ~(hm_pair_a | (hm_pair_b << 16));

            uint hc0_u = ((hmk & hm_m0) >> (v_im4 + 0)) << 2;
            uint hc1_u = ((hmk & hm_m1) >> (v_im4 + 1)) << 2;
            uint hc2_u = ((hmk & hm_m2) >> (v_im4 + 2)) << 2;
            uint hc3_u = ((hmk & hm_m3) >> (v_im4 + 3)) << 2;

            float h00 = float(hc0_u & 0xFFu); float h01 = float((hc0_u >> 8) & 0xFFu);
            float h02 = float((hc0_u >> 16) & 0xFFu); float h03 = float((hc0_u >> 24) & 0xFFu);
            float h10 = float(hc1_u & 0xFFu); float h11 = float((hc1_u >> 8) & 0xFFu);
            float h12 = float((hc1_u >> 16) & 0xFFu); float h13 = float((hc1_u >> 24) & 0xFFu);
            float h20 = float(hc2_u & 0xFFu); float h21 = float((hc2_u >> 8) & 0xFFu);
            float h22 = float((hc2_u >> 16) & 0xFFu); float h23 = float((hc2_u >> 24) & 0xFFu);
            float h30 = float(hc3_u & 0xFFu); float h31 = float((hc3_u >> 8) & 0xFFu);
            float h32 = float((hc3_u >> 16) & 0xFFu); float h33 = float((hc3_u >> 24) & 0xFFu);

            // --- Load 16 activations ---
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

            // --- Dot product with hmask correction ---
            acc += s0 * mad(ql0  - h00, by0,  (ql1  - h01) * by1)
                +  s1 * mad(ql2  - h02, by2,  (ql3  - h03) * by3)
                +  s2 * mad(ql4  - h10, by4,  (ql5  - h11) * by5)
                +  s3 * mad(ql6  - h12, by6,  (ql7  - h13) * by7)
                +  s4 * mad(ql8  - h20, by8,  (ql9  - h21) * by9)
                +  s5 * mad(ql10 - h22, by10, (ql11 - h23) * by11)
                +  s6 * mad(ql12 - h30, by12, (ql13 - h31) * by13)
                +  s7 * mad(ql14 - h32, by14, (ql15 - h33) * by15);
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
