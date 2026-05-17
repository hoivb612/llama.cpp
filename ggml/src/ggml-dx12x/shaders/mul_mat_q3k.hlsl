// mul_mat_q3k.hlsl - Cooperative batch matmul for Q3_K quantized weights
//
// 16 output rows per group, 16 K-threads cooperate per row.
// Processes TILE_M=4 activation columns per group — weight data decoded once
// per K-block, reused across all 4 columns.
// Uses Load2 for f32 activations.
//
// Q3_K block layout (110 bytes, NOT 4-byte aligned):
//   hmask[32]: high bit per element
//   qs[64]:    low 2 bits per element
//   scales[12]: 6-bit scales, packed
//   d (f16):   super-block scale
//
// Dispatch: groups_x = ceil(N/16), groups_y = ceil(M/4), groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE       256
#define ROWS_PER_GROUP   16
#define THREADS_PER_ROW  16
#define QK_K             256
#define Q3K_BSIZE        110
#define TILE_M           4

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

    uint i0      = group_id.x * ROWS_PER_GROUP + row_local;
    uint i1_base = group_id.y * TILE_M;
    uint i2      = group_id.z % ne2;
    uint i3      = group_id.z / ne2;

    uint N = ne0;
    uint M = ne1;
    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_bc  = i2 * nb12 + i3 * nb13;

    uint v_im  = k_tid / 8;
    uint v_in  = k_tid % 8;
    uint v_im4 = v_im * 4;
    uint l0       = 2 * v_in;
    uint q_offset = 32 * v_im + l0;
    uint y_offset = 128 * v_im + l0;

    uint m_base = 0x01010101u << v_im4;
    uint hm_m0  = m_base;
    uint hm_m1  = m_base << 1;
    uint hm_m2  = m_base << 2;
    uint hm_m3  = m_base << 3;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    if (i0 < N) {
        for (uint bg = 0; bg < num_blocks; bg++) {
            uint block_off = src0_row + bg * Q3K_BSIZE;

            // --- Weight decode (once, reused for TILE_M columns) ---
            uint d_word = load_u32(src0, block_off + 108);
            float d_all = f16_to_f32(d_word & 0xFFFFu);

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

            // Pre-compute corrected qs values (weight-only, reused across columns)
            float w0  = ql0  - h00; float w1  = ql1  - h01;
            float w2  = ql2  - h02; float w3  = ql3  - h03;
            float w4  = ql4  - h10; float w5  = ql5  - h11;
            float w6  = ql6  - h12; float w7  = ql7  - h13;
            float w8  = ql8  - h20; float w9  = ql9  - h21;
            float w10 = ql10 - h22; float w11 = ql11 - h23;
            float w12 = ql12 - h30; float w13 = ql13 - h31;
            float w14 = ql14 - h32; float w15 = ql15 - h33;

            // --- Process TILE_M activation columns ---
            uint y_elem = bg * QK_K + y_offset;

            [unroll]
            for (uint m = 0; m < TILE_M; m++) {
                if (i1_base + m >= M) break;

                uint src1_m = src1_offset + (i1_base + m) * nb11 + src1_bc;
                float by0, by1, by2, by3, by4, by5, by6, by7;
                float by8, by9, by10, by11, by12, by13, by14, by15;

                if (src1_esize == 4u) {
                    uint y_off = src1_m + y_elem * 4u;
                    uint2 a0 = src1.Load2(y_off);
                    uint2 a1 = src1.Load2(y_off + 64);
                    uint2 a2 = src1.Load2(y_off + 128);
                    uint2 a3 = src1.Load2(y_off + 192);
                    uint2 a4 = src1.Load2(y_off + 256);
                    uint2 a5 = src1.Load2(y_off + 320);
                    uint2 a6 = src1.Load2(y_off + 384);
                    uint2 a7 = src1.Load2(y_off + 448);
                    by0 =asfloat(a0.x); by1 =asfloat(a0.y);
                    by2 =asfloat(a1.x); by3 =asfloat(a1.y);
                    by4 =asfloat(a2.x); by5 =asfloat(a2.y);
                    by6 =asfloat(a3.x); by7 =asfloat(a3.y);
                    by8 =asfloat(a4.x); by9 =asfloat(a4.y);
                    by10=asfloat(a5.x); by11=asfloat(a5.y);
                    by12=asfloat(a6.x); by13=asfloat(a6.y);
                    by14=asfloat(a7.x); by15=asfloat(a7.y);
                } else {
                    uint yb = src1_m + y_elem * src1_esize;
                    by0  = load_auto(src1, yb,                    src1_esize);
                    by1  = load_auto(src1, yb +      src1_esize,  src1_esize);
                    by2  = load_auto(src1, yb + 16 * src1_esize,  src1_esize);
                    by3  = load_auto(src1, yb + 17 * src1_esize,  src1_esize);
                    by4  = load_auto(src1, yb + 32 * src1_esize,  src1_esize);
                    by5  = load_auto(src1, yb + 33 * src1_esize,  src1_esize);
                    by6  = load_auto(src1, yb + 48 * src1_esize,  src1_esize);
                    by7  = load_auto(src1, yb + 49 * src1_esize,  src1_esize);
                    by8  = load_auto(src1, yb + 64 * src1_esize,  src1_esize);
                    by9  = load_auto(src1, yb + 65 * src1_esize,  src1_esize);
                    by10 = load_auto(src1, yb + 80 * src1_esize,  src1_esize);
                    by11 = load_auto(src1, yb + 81 * src1_esize,  src1_esize);
                    by12 = load_auto(src1, yb + 96 * src1_esize,  src1_esize);
                    by13 = load_auto(src1, yb + 97 * src1_esize,  src1_esize);
                    by14 = load_auto(src1, yb + 112 * src1_esize, src1_esize);
                    by15 = load_auto(src1, yb + 113 * src1_esize, src1_esize);
                }

                float partial = s0 * mad(w0, by0, w1 * by1)
                    + s1 * mad(w2,  by2,  w3  * by3)
                    + s2 * mad(w4,  by4,  w5  * by5)
                    + s3 * mad(w6,  by6,  w7  * by7)
                    + s4 * mad(w8,  by8,  w9  * by9)
                    + s5 * mad(w10, by10, w11 * by11)
                    + s6 * mad(w12, by12, w13 * by13)
                    + s7 * mad(w14, by14, w15 * by15);

                if (m == 0u) acc0 += partial;
                else if (m == 1u) acc1 += partial;
                else if (m == 2u) acc2 += partial;
                else acc3 += partial;
            }
        }
    }

    // Butterfly reduction across 16 K-threads for all 4 accumulators
    uint wave_lane = WaveGetLaneIndex();
    [unroll]
    for (uint offset = 1; offset < THREADS_PER_ROW; offset <<= 1) {
        acc0 += WaveReadLaneAt(acc0, wave_lane ^ offset);
        acc1 += WaveReadLaneAt(acc1, wave_lane ^ offset);
        acc2 += WaveReadLaneAt(acc2, wave_lane ^ offset);
        acc3 += WaveReadLaneAt(acc3, wave_lane ^ offset);
    }

    if (k_tid == 0 && i0 < N) {
        if (i1_base     < M) store_auto(dst, offset_4d(i0, i1_base,   i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc0, dst_esize);
        if (i1_base + 1 < M) store_auto(dst, offset_4d(i0, i1_base+1, i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc1, dst_esize);
        if (i1_base + 2 < M) store_auto(dst, offset_4d(i0, i1_base+2, i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc2, dst_esize);
        if (i1_base + 3 < M) store_auto(dst, offset_4d(i0, i1_base+3, i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc3, dst_esize);
    }
}
