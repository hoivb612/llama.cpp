// Shared per-element quantized dequant tables. Wrappers (mul_mat_quant.hlsli
// and mul_mat_id_quant.hlsli) include this file after defining exactly one
// MMID_<TYPE> macro. Defines MMID_QK, MMID_BLOCK_SIZE, and
// loat mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k).
#pragma once
#include "ggml_common.hlsli"

uint mmid_read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

int mmid_read_sbyte(ByteAddressBuffer buf, uint byte_off) {
    uint b = mmid_read_byte(buf, byte_off);
    return (b < 128u) ? (int)b : (int)b - 256;
}

float mmid_read_f16(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint mmid_read_u32_unaligned(ByteAddressBuffer buf, uint byte_off) {
    uint b0 = mmid_read_byte(buf, byte_off);
    uint b1 = mmid_read_byte(buf, byte_off + 1);
    uint b2 = mmid_read_byte(buf, byte_off + 2);
    uint b3 = mmid_read_byte(buf, byte_off + 3);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

#if defined(MMID_Q4_0)
#define MMID_QK 32
#define MMID_BLOCK_SIZE 18
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    float d = mmid_read_f16(buf, block_off);
    uint qs = mmid_read_byte(buf, block_off + 2 + (elem % 16));
    int q = (elem < 16) ? ((int)(qs & 0x0Fu) - 8) : ((int)(qs >> 4) - 8);
    return d * (float)q;
}
#elif defined(MMID_Q4_1)
#define MMID_QK 32
#define MMID_BLOCK_SIZE 20
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    float d = mmid_read_f16(buf, block_off);
    float m = mmid_read_f16(buf, block_off + 2);
    uint qs = mmid_read_byte(buf, block_off + 4 + (elem % 16));
    uint q = (elem < 16) ? (qs & 0x0Fu) : (qs >> 4);
    return (float)q * d + m;
}
#elif defined(MMID_Q5_0)
#define MMID_QK 32
#define MMID_BLOCK_SIZE 22
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    float d = mmid_read_f16(buf, block_off);
    uint qh = mmid_read_u32_unaligned(buf, block_off + 2);
    uint qs = mmid_read_byte(buf, block_off + 6 + (elem % 16));
    uint xh = (elem < 16) ? (((qh >> elem) << 4) & 0x10u) : ((qh >> (elem - 4)) & 0x10u);
    uint ql = (elem < 16) ? (qs & 0x0Fu) : (qs >> 4);
    return d * (float)((int)(ql | xh) - 16);
}
#elif defined(MMID_Q5_1)
#define MMID_QK 32
#define MMID_BLOCK_SIZE 24
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    float d = mmid_read_f16(buf, block_off);
    float m = mmid_read_f16(buf, block_off + 2);
    uint qh = mmid_read_u32_unaligned(buf, block_off + 4);
    uint qs = mmid_read_byte(buf, block_off + 8 + (elem % 16));
    uint xh = (elem < 16) ? (((qh >> elem) << 4) & 0x10u) : ((qh >> (elem - 4)) & 0x10u);
    uint ql = (elem < 16) ? (qs & 0x0Fu) : (qs >> 4);
    return (float)(ql | xh) * d + m;
}
#elif defined(MMID_Q8_0)
#define MMID_QK 32
#define MMID_BLOCK_SIZE 34
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    float d = mmid_read_f16(buf, block_off);
    int q = mmid_read_sbyte(buf, block_off + 2 + (k % MMID_QK));
    return d * (float)q;
}
#elif defined(MMID_Q4_K)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 144
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint dm_raw = buf.Load(block_off);
    float dall = f16_to_f32(dm_raw & 0xFFFFu);
    float dmin_val = f16_to_f32(dm_raw >> 16);
    uint il = elem / 64;
    uint elem_in_chunk = elem % 64;
    bool is_high = (elem_in_chunk >= 32);
    uint elem_in_half = elem_in_chunk % 32;
    uint is = 2 * il;
    uint is_eff = is_high ? (is + 1) : is;
    uint scales_off = block_off + 4;

    uint scidx0 = (is < 4) ? is_eff : (is_eff + 4);
    uint scidx1 = (is < 4) ? is_eff : (is_eff - 4);
    uint scmask1 = (is < 4) ? 0x30u : 0xC0u;
    uint scshift1 = (is < 4) ? 0u : 2u;
    uint mbidx0 = is_eff + 4;
    uint mbidx1 = (is < 4) ? is_eff + 4 : is_eff;
    uint mbmask0 = (is < 4) ? 0x0Fu : 0xF0u;
    uint mbshift0 = (is < 4) ? 0u : 4u;
    uint mbmask1 = (is < 4) ? 0x30u : 0xC0u;
    uint mbshift1 = (is < 4) ? 0u : 2u;

    uint sc = (mmid_read_byte(buf, scales_off + scidx0) & 0x0Fu) |
              ((mmid_read_byte(buf, scales_off + scidx1) & scmask1) >> scshift1);
    uint mb = ((mmid_read_byte(buf, scales_off + mbidx0) & mbmask0) >> mbshift0) |
              ((mmid_read_byte(buf, scales_off + mbidx1) & mbmask1) >> mbshift1);
    uint qs = mmid_read_byte(buf, block_off + 16 + il * 32 + elem_in_half);
    uint q = is_high ? (qs >> 4) : (qs & 0x0Fu);
    return dall * (float)sc * (float)q - dmin_val * (float)mb;
}
#elif defined(MMID_Q5_K)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 176
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint dm_raw = buf.Load(block_off);
    float dall = f16_to_f32(dm_raw & 0xFFFFu);
    float dmin_val = f16_to_f32(dm_raw >> 16);
    uint il = elem / 64;
    uint elem_in_chunk = elem % 64;
    bool is_high = (elem_in_chunk >= 32);
    uint elem_in_half = elem_in_chunk % 32;
    uint is = 2 * il;
    uint is_eff = is_high ? (is + 1) : is;
    uint scales_off = block_off + 4;

    uint scidx0 = (is < 4) ? is_eff : (is_eff + 4);
    uint scidx1 = (is < 4) ? is_eff : (is_eff - 4);
    uint scmask1 = (is < 4) ? 0x30u : 0xC0u;
    uint scshift1 = (is < 4) ? 0u : 2u;
    uint mbidx0 = is_eff + 4;
    uint mbidx1 = (is < 4) ? is_eff + 4 : is_eff;
    uint mbmask0 = (is < 4) ? 0x0Fu : 0xF0u;
    uint mbshift0 = (is < 4) ? 0u : 4u;
    uint mbmask1 = (is < 4) ? 0x30u : 0xC0u;
    uint mbshift1 = (is < 4) ? 0u : 2u;

    uint sc = (mmid_read_byte(buf, scales_off + scidx0) & 0x0Fu) |
              ((mmid_read_byte(buf, scales_off + scidx1) & scmask1) >> scshift1);
    uint mb = ((mmid_read_byte(buf, scales_off + mbidx0) & mbmask0) >> mbshift0) |
              ((mmid_read_byte(buf, scales_off + mbidx1) & mbmask1) >> mbshift1);
    uint qs = mmid_read_byte(buf, block_off + 48 + il * 32 + elem_in_half);
    uint qh = mmid_read_byte(buf, block_off + 16 + elem_in_half);
    uint hm = is_high ? (1u << (2u * il + 1u)) : (1u << (2u * il));
    uint q = (is_high ? (qs >> 4) : (qs & 0x0Fu)) + (((qh & hm) != 0u) ? 16u : 0u);
    return dall * (float)sc * (float)q - dmin_val * (float)mb;
}
#elif defined(MMID_Q6_K)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 210
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint d_off = block_off + 208;
    uint d_word = buf.Load(d_off & ~3u);
    float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);
    uint ip = elem / 128;
    uint il = elem % 128;
    int scale = mmid_read_sbyte(buf, block_off + 192 + 8 * ip + il / 16);
    uint ql = mmid_read_byte(buf, block_off + 64 * ip + (il % 64));
    uint qh = mmid_read_byte(buf, block_off + 128 + 32 * ip + (il % 32));
    int q;
    if (il < 32) {
        q = (int)((ql & 0x0Fu) | (((qh >> 0) & 3u) << 4)) - 32;
    } else if (il < 64) {
        q = (int)((ql & 0x0Fu) | (((qh >> 2) & 3u) << 4)) - 32;
    } else if (il < 96) {
        q = (int)((ql >> 4) | (((qh >> 4) & 3u) << 4)) - 32;
    } else {
        q = (int)((ql >> 4) | (((qh >> 6) & 3u) << 4)) - 32;
    }
    return d * (float)scale * (float)q;
}
#elif defined(MMID_IQ4_NL)
#define MMID_QK 32
#define MMID_BLOCK_SIZE 18
int mmid_kvalues_iq4nl(uint idx) {
    static const uint packed[4] = {
        0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u
    };
    uint w = packed[idx >> 2];
    uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    return (int)(b << 24) >> 24;
}
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    float d = mmid_read_f16(buf, block_off);
    uint qs = mmid_read_byte(buf, block_off + 2 + (elem % 16));
    uint q = (elem < 16) ? (qs & 0x0Fu) : ((qs >> 4) & 0x0Fu);
    return d * (float)mmid_kvalues_iq4nl(q);
}
#elif defined(MMID_Q2_K)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 84
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    float d    = mmid_read_f16(buf, block_off + 80);
    float dmin = mmid_read_f16(buf, block_off + 82);
    uint sc_idx = elem / 16;
    uint sc_byte = mmid_read_byte(buf, block_off + sc_idx);
    float scale = d * (float)(sc_byte & 0x0Fu);
    float min_val = dmin * (float)(sc_byte >> 4);
    uint qs_pos = ((sc_idx >> 3) & 1u) * 32u + (sc_idx & 1u) * 16u + (elem & 0xFu);
    uint qs_shift = ((sc_idx >> 1) & 3u) * 2u;
    uint qs_byte = mmid_read_byte(buf, block_off + 16u + qs_pos);
    int q = (int)((qs_byte >> qs_shift) & 3u);
    return scale * (float)q - min_val;
}
#elif defined(MMID_Q3_K)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 110
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint j = k % MMID_QK;
    float d = mmid_read_f16(buf, block_off + 108);
    uint sc_idx = j / 16;
    uint shift  = ((sc_idx >> 1) & 3u) * 2u;
    uint qs_pos = ((sc_idx >> 3) & 1u) * 32u + (sc_idx & 1u) * 16u + (j & 0xFu);
    uint hm_pos = (sc_idx & 1u) * 16u + (j & 0xFu);
    uint m_bit  = sc_idx >> 1;
    uint qs_byte = mmid_read_byte(buf, block_off + 32u + qs_pos);
    uint ql      = (qs_byte >> shift) & 3u;
    uint hm_byte = mmid_read_byte(buf, block_off + 0u + hm_pos);
    uint qh      = (hm_byte >> m_bit) & 1u;
    int q3 = (int)ql - (qh != 0u ? 0 : 4);
    uint scales_off = block_off + 96u;
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0F0F0F0Fu;
    uint raw0 = mmid_read_byte(buf, scales_off + 0u)
              | (mmid_read_byte(buf, scales_off + 1u) << 8u)
              | (mmid_read_byte(buf, scales_off + 2u) << 16u)
              | (mmid_read_byte(buf, scales_off + 3u) << 24u);
    uint raw4 = mmid_read_byte(buf, scales_off + 4u)
              | (mmid_read_byte(buf, scales_off + 5u) << 8u)
              | (mmid_read_byte(buf, scales_off + 6u) << 16u)
              | (mmid_read_byte(buf, scales_off + 7u) << 24u);
    uint raw8 = mmid_read_byte(buf, scales_off + 8u)
              | (mmid_read_byte(buf, scales_off + 9u) << 8u)
              | (mmid_read_byte(buf, scales_off + 10u) << 16u)
              | (mmid_read_byte(buf, scales_off + 11u) << 24u);
    uint sub = sc_idx >> 2;
    uint pos = (sc_idx & 3u) * 8u;
    uint aux_word;
    if      (sub == 0u) aux_word = (raw0 & kmask2)         | (((raw8 >> 0u) & kmask1) << 4u);
    else if (sub == 1u) aux_word = (raw4 & kmask2)         | (((raw8 >> 2u) & kmask1) << 4u);
    else if (sub == 2u) aux_word = ((raw0 >> 4u) & kmask2) | (((raw8 >> 4u) & kmask1) << 4u);
    else                aux_word = ((raw4 >> 4u) & kmask2) | (((raw8 >> 6u) & kmask1) << 4u);
    int scale = int((aux_word >> pos) & 0xFFu) - 32;
    return d * (float)scale * (float)q3;
}
#elif defined(MMID_IQ4_XS)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 136
int mmid_kvalues_iq4xs(uint idx) {
    static const uint packed[4] = {
        0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u
    };
    uint w = packed[idx >> 2];
    uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    return (int)(b << 24) >> 24;
}
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint ib32 = elem / 32u;
    uint l_in_sub = elem % 32u;
    bool is_high = l_in_sub >= 16u;
    uint j_in_half = l_in_sub & 15u;
    float d = mmid_read_f16(buf, block_off);
    uint scales_h = buf.Load(block_off) >> 16;
    uint scales_l = buf.Load(block_off + 4u);
    uint slo = (scales_l >> ((ib32 / 2u) * 8u + (ib32 & 1u) * 4u)) & 0xFu;
    uint shi = (scales_h >> (2u * ib32)) & 0x3u;
    int  s6  = (int)(slo | (shi << 4u));
    float dl = d * (float)(s6 - 32);
    uint qbyte = mmid_read_byte(buf, block_off + 8u + 16u * ib32 + j_in_half);
    int v = mmid_kvalues_iq4xs(is_high ? (qbyte >> 4) : (qbyte & 0xFu));
    return dl * (float)v;
}
#elif defined(MMID_IQ2_XXS)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 66
// IQ2_XXS grid: 256 entries x 8 bytes (uint2). Matches iq2xxs_grid in ggml-common.h.
static const uint2 mmid_iq2xxs_grid[256] = {
    uint2(0x08080808u,0x08080808u), uint2(0x0808082bu,0x08080808u), uint2(0x08081919u,0x08080808u), uint2(0x08082b08u,0x08080808u),
    uint2(0x08082b2bu,0x08080808u), uint2(0x08190819u,0x08080808u), uint2(0x08191908u,0x08080808u), uint2(0x082b0808u,0x08080808u),
    uint2(0x082b082bu,0x08080808u), uint2(0x082b2b08u,0x08080808u), uint2(0x082b2b2bu,0x08080808u), uint2(0x19080819u,0x08080808u),
    uint2(0x19081908u,0x08080808u), uint2(0x19190808u,0x08080808u), uint2(0x19192b08u,0x08080808u), uint2(0x192b0819u,0x08080808u),
    uint2(0x192b1908u,0x08080808u), uint2(0x2b080808u,0x08080808u), uint2(0x2b08082bu,0x08080808u), uint2(0x2b082b2bu,0x08080808u),
    uint2(0x2b2b082bu,0x08080808u), uint2(0x08080819u,0x08080819u), uint2(0x08081908u,0x08080819u), uint2(0x08190808u,0x08080819u),
    uint2(0x08191919u,0x08080819u), uint2(0x19080808u,0x08080819u), uint2(0x2b081908u,0x08080819u), uint2(0x2b192b08u,0x08080819u),
    uint2(0x08080808u,0x0808082bu), uint2(0x0808082bu,0x0808082bu), uint2(0x082b082bu,0x0808082bu), uint2(0x2b08082bu,0x0808082bu),
    uint2(0x08080819u,0x08081908u), uint2(0x08081908u,0x08081908u), uint2(0x08190808u,0x08081908u), uint2(0x082b0819u,0x08081908u),
    uint2(0x082b1908u,0x08081908u), uint2(0x19080808u,0x08081908u), uint2(0x1908082bu,0x08081908u), uint2(0x19082b08u,0x08081908u),
    uint2(0x192b0808u,0x08081908u), uint2(0x2b080819u,0x08081908u), uint2(0x2b081908u,0x08081908u), uint2(0x2b190808u,0x08081908u),
    uint2(0x2b2b1908u,0x08081908u), uint2(0x08080808u,0x08081919u), uint2(0x0808082bu,0x08081919u), uint2(0x08082b08u,0x08081919u),
    uint2(0x082b0808u,0x08081919u), uint2(0x1908192bu,0x08081919u), uint2(0x192b2b19u,0x08081919u), uint2(0x2b080808u,0x08081919u),
    uint2(0x2b190819u,0x08081919u), uint2(0x08082b19u,0x0808192bu), uint2(0x08190808u,0x0808192bu), uint2(0x19080808u,0x0808192bu),
    uint2(0x2b081908u,0x0808192bu), uint2(0x2b2b1908u,0x0808192bu), uint2(0x08080808u,0x08082b08u), uint2(0x08081919u,0x08082b08u),
    uint2(0x08082b08u,0x08082b08u), uint2(0x08191908u,0x08082b08u), uint2(0x082b2b08u,0x08082b08u), uint2(0x19080819u,0x08082b08u),
    uint2(0x19081908u,0x08082b08u), uint2(0x19190808u,0x08082b08u), uint2(0x1919082bu,0x08082b08u), uint2(0x2b082b08u,0x08082b08u),
    uint2(0x08081908u,0x08082b19u), uint2(0x19080808u,0x08082b19u), uint2(0x0808082bu,0x08082b2bu), uint2(0x08191908u,0x08082b2bu),
    uint2(0x08080819u,0x08190808u), uint2(0x08081908u,0x08190808u), uint2(0x08190808u,0x08190808u), uint2(0x082b0819u,0x08190808u),
    uint2(0x19080808u,0x08190808u), uint2(0x192b0808u,0x08190808u), uint2(0x2b081908u,0x08190808u), uint2(0x2b190808u,0x08190808u),
    uint2(0x2b191919u,0x08190808u), uint2(0x08080808u,0x08190819u), uint2(0x08082b08u,0x08190819u), uint2(0x082b0808u,0x08190819u),
    uint2(0x19190808u,0x08190819u), uint2(0x19192b2bu,0x08190819u), uint2(0x2b080808u,0x08190819u), uint2(0x082b1908u,0x0819082bu),
    uint2(0x19081919u,0x0819082bu), uint2(0x08080808u,0x08191908u), uint2(0x08082b08u,0x08191908u), uint2(0x082b0808u,0x08191908u),
    uint2(0x082b1919u,0x08191908u), uint2(0x19082b19u,0x08191908u), uint2(0x2b080808u,0x08191908u), uint2(0x08192b08u,0x08191919u),
    uint2(0x192b082bu,0x08191919u), uint2(0x08080808u,0x0819192bu), uint2(0x0819192bu,0x0819192bu), uint2(0x08080819u,0x08192b08u),
    uint2(0x08081908u,0x08192b08u), uint2(0x08190808u,0x08192b08u), uint2(0x19080808u,0x08192b08u), uint2(0x2b080819u,0x08192b08u),
    uint2(0x08080808u,0x08192b19u), uint2(0x08081919u,0x08192b19u), uint2(0x2b2b0808u,0x08192b19u), uint2(0x19190819u,0x08192b2bu),
    uint2(0x08080808u,0x082b0808u), uint2(0x0808082bu,0x082b0808u), uint2(0x08082b2bu,0x082b0808u), uint2(0x19081908u,0x082b0808u),
    uint2(0x192b0819u,0x082b0808u), uint2(0x2b080808u,0x082b0808u), uint2(0x2b08082bu,0x082b0808u), uint2(0x082b2b19u,0x082b0819u),
    uint2(0x19082b08u,0x082b0819u), uint2(0x08080808u,0x082b082bu), uint2(0x0808082bu,0x082b082bu), uint2(0x08080819u,0x082b1908u),
    uint2(0x08081908u,0x082b1908u), uint2(0x08190808u,0x082b1908u), uint2(0x19080808u,0x082b1908u), uint2(0x1919192bu,0x082b1908u),
    uint2(0x08080808u,0x082b1919u), uint2(0x19080819u,0x082b1919u), uint2(0x192b1908u,0x082b1919u), uint2(0x2b190808u,0x082b192bu),
    uint2(0x08082b08u,0x082b2b08u), uint2(0x082b0808u,0x082b2b08u), uint2(0x2b191908u,0x082b2b08u), uint2(0x19081908u,0x082b2b2bu),
    uint2(0x08080819u,0x19080808u), uint2(0x08081908u,0x19080808u), uint2(0x08190808u,0x19080808u), uint2(0x08192b08u,0x19080808u),
    uint2(0x082b0819u,0x19080808u), uint2(0x082b1908u,0x19080808u), uint2(0x19080808u,0x19080808u), uint2(0x19082b08u,0x19080808u),
    uint2(0x1919192bu,0x19080808u), uint2(0x192b0808u,0x19080808u), uint2(0x2b080819u,0x19080808u), uint2(0x2b081908u,0x19080808u),
    uint2(0x2b190808u,0x19080808u), uint2(0x08080808u,0x19080819u), uint2(0x082b0808u,0x19080819u), uint2(0x192b0819u,0x19080819u),
    uint2(0x2b080808u,0x19080819u), uint2(0x2b081919u,0x19080819u), uint2(0x08080819u,0x1908082bu), uint2(0x08190808u,0x1908082bu),
    uint2(0x19082b08u,0x1908082bu), uint2(0x1919192bu,0x1908082bu), uint2(0x192b2b08u,0x1908082bu), uint2(0x08080808u,0x19081908u),
    uint2(0x08082b08u,0x19081908u), uint2(0x082b0808u,0x19081908u), uint2(0x2b080808u,0x19081908u), uint2(0x2b192b19u,0x19081908u),
    uint2(0x0819082bu,0x19081919u), uint2(0x082b1908u,0x19081919u), uint2(0x08080808u,0x1908192bu), uint2(0x08080819u,0x19082b08u),
    uint2(0x08081908u,0x19082b08u), uint2(0x08190808u,0x19082b08u), uint2(0x19080808u,0x19082b08u), uint2(0x19081919u,0x19082b08u),
    uint2(0x08080808u,0x19082b19u), uint2(0x19192b08u,0x19082b19u), uint2(0x192b0819u,0x19082b19u), uint2(0x2b08082bu,0x19082b19u),
    uint2(0x19081919u,0x19082b2bu), uint2(0x2b190808u,0x19082b2bu), uint2(0x08080808u,0x19190808u), uint2(0x08082b08u,0x19190808u),
    uint2(0x08190819u,0x19190808u), uint2(0x08192b19u,0x19190808u), uint2(0x082b0808u,0x19190808u), uint2(0x2b080808u,0x19190808u),
    uint2(0x2b082b08u,0x19190808u), uint2(0x08081908u,0x19190819u), uint2(0x1908082bu,0x19190819u), uint2(0x2b2b1908u,0x19190819u),
    uint2(0x2b190819u,0x1919082bu), uint2(0x2b190808u,0x19191908u), uint2(0x2b19082bu,0x19191908u), uint2(0x08082b2bu,0x19191919u),
    uint2(0x08080819u,0x1919192bu), uint2(0x19191908u,0x1919192bu), uint2(0x08080808u,0x19192b08u), uint2(0x08190819u,0x19192b08u),
    uint2(0x08192b19u,0x19192b08u), uint2(0x192b1908u,0x19192b08u), uint2(0x19080808u,0x19192b19u), uint2(0x08082b08u,0x19192b2bu),
    uint2(0x08081908u,0x192b0808u), uint2(0x08190808u,0x192b0808u), uint2(0x19080808u,0x192b0808u), uint2(0x192b2b08u,0x192b0808u),
    uint2(0x08080808u,0x192b0819u), uint2(0x19191919u,0x192b0819u), uint2(0x08192b08u,0x192b082bu), uint2(0x192b0808u,0x192b082bu),
    uint2(0x08080808u,0x192b1908u), uint2(0x08081919u,0x192b1908u), uint2(0x08190808u,0x192b1919u), uint2(0x0819082bu,0x192b1919u),
    uint2(0x2b081908u,0x192b1919u), uint2(0x1908082bu,0x192b2b08u), uint2(0x08080808u,0x2b080808u), uint2(0x0808082bu,0x2b080808u),
    uint2(0x08082b2bu,0x2b080808u), uint2(0x19080819u,0x2b080808u), uint2(0x2b08082bu,0x2b080808u), uint2(0x08081908u,0x2b080819u),
    uint2(0x08192b08u,0x2b080819u), uint2(0x19080808u,0x2b080819u), uint2(0x08190819u,0x2b08082bu), uint2(0x08080819u,0x2b081908u),
    uint2(0x08081908u,0x2b081908u), uint2(0x08190808u,0x2b081908u), uint2(0x08191919u,0x2b081908u), uint2(0x19080808u,0x2b081908u),
    uint2(0x192b0808u,0x2b081908u), uint2(0x08080808u,0x2b081919u), uint2(0x1908192bu,0x2b081919u), uint2(0x2b191908u,0x2b081919u),
    uint2(0x08082b19u,0x2b08192bu), uint2(0x19080808u,0x2b08192bu), uint2(0x192b0808u,0x2b08192bu), uint2(0x0808082bu,0x2b082b08u),
    uint2(0x08081908u,0x2b082b19u), uint2(0x08190819u,0x2b082b2bu), uint2(0x08081908u,0x2b190808u), uint2(0x08190808u,0x2b190808u),
    uint2(0x082b1908u,0x2b190808u), uint2(0x19080808u,0x2b190808u), uint2(0x2b2b0819u,0x2b190808u), uint2(0x0819192bu,0x2b190819u),
    uint2(0x2b080808u,0x2b190819u), uint2(0x19081919u,0x2b19082bu), uint2(0x08080808u,0x2b191908u), uint2(0x082b082bu,0x2b191908u),
    uint2(0x19081908u,0x2b191908u), uint2(0x19190819u,0x2b191919u), uint2(0x2b080819u,0x2b192b08u), uint2(0x082b0808u,0x2b192b19u),
    uint2(0x0808082bu,0x2b2b0808u), uint2(0x19190808u,0x2b2b0808u), uint2(0x2b081919u,0x2b2b0808u), uint2(0x08082b19u,0x2b2b0819u),
    uint2(0x08080808u,0x2b2b082bu), uint2(0x08192b08u,0x2b2b1908u), uint2(0x19190808u,0x2b2b2b08u), uint2(0x08081908u,0x2b2b2b19u)
};
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint ib32 = elem / 32u;
    uint elem_in_sub = elem % 32u;
    uint l = elem_in_sub / 8u;
    uint j = elem_in_sub % 8u;
    float d = mmid_read_f16(buf, block_off);
    uint sub_off = block_off + 2u + 8u * ib32;
    uint aux32_g = mmid_read_u32_unaligned(buf, sub_off);
    uint aux32_s = mmid_read_u32_unaligned(buf, sub_off + 4u);
    uint grid_idx = (aux32_g >> (l * 8u)) & 0xFFu;
    uint signs    = (aux32_s >> (l * 7u)) & 0x7Fu;
    uint parity   = countbits(signs) & 1u;
    float scale = d * 0.25f * (0.5f + (float)(aux32_s >> 28));
    uint2 grid8 = mmid_iq2xxs_grid[grid_idx];
    uint gword = (j < 4u) ? grid8.x : grid8.y;
    uint gbyte = (gword >> ((j & 3u) * 8u)) & 0xFFu;
    uint sbit  = (j < 7u) ? ((signs >> j) & 1u) : parity;
    float sgn  = (sbit != 0u) ? -1.0f : 1.0f;
    return scale * sgn * (float)gbyte;
}
#elif defined(MMID_IQ2_XS)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 74
#include "iq2xs_grid.hlsli"
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint ib32 = elem / 32u;
    uint l    = (elem % 32u) / 8u;
    uint j    = elem & 7u;

    float d = mmid_read_f16(buf, block_off);
    uint qs_off = block_off + 2u + (4u * ib32 + l) * 2u;
    uint q2_lo = mmid_read_byte(buf, qs_off);
    uint q2_hi = mmid_read_byte(buf, qs_off + 1u);
    uint q2 = q2_lo | (q2_hi << 8u);
    uint grid_idx = q2 & 0x1FFu;
    uint signs    = (q2 >> 9) & 0x7Fu;
    uint parity   = countbits(signs) & 1u;

    uint il = l >> 1;
    uint scale_byte = mmid_read_byte(buf, block_off + 66u + ib32);
    float dl = d * (0.5f + (float)((scale_byte >> (4u * il)) & 0xFu)) * 0.25f;

    uint2 grid8 = iq2xs_grid[grid_idx];
    uint gword = (j < 4u) ? grid8.x : grid8.y;
    uint gbyte = (gword >> ((j & 3u) * 8u)) & 0xFFu;
    uint sbit  = (j < 7u) ? ((signs >> j) & 1u) : parity;
    float sgn  = (sbit != 0u) ? -1.0f : 1.0f;
    return dl * sgn * (float)gbyte;
}
#elif defined(MMID_IQ2_S)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 82
#include "iq2s_grid.hlsli"
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint ib32 = elem / 32u;
    uint l    = (elem % 32u) / 8u;
    uint j    = elem & 7u;

    float d = mmid_read_f16(buf, block_off);
    uint q_lo  = mmid_read_byte(buf, block_off + 2u + 4u * ib32 + l);
    uint qh    = mmid_read_byte(buf, block_off + 66u + ib32);
    uint q_hi2 = (qh >> (2u * l)) & 0x3u;
    uint grid_idx = q_lo | (q_hi2 << 8u);

    uint signs = mmid_read_byte(buf, block_off + 2u + 32u + 4u * ib32 + l);

    uint il = l >> 1;
    uint scale_byte = mmid_read_byte(buf, block_off + 74u + ib32);
    float dl = d * (0.5f + (float)((scale_byte >> (4u * il)) & 0xFu)) * 0.25f;

    uint2 grid8 = iq2s_grid[grid_idx];
    uint gword = (j < 4u) ? grid8.x : grid8.y;
    uint gbyte = (gword >> ((j & 3u) * 8u)) & 0xFFu;
    uint sbit = (signs >> j) & 1u;
    float sgn = (sbit != 0u) ? -1.0f : 1.0f;
    return dl * sgn * (float)gbyte;
}
#elif defined(MMID_IQ3_XXS)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 98
#include "iq3xxs_grid.hlsli"
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint ib32 = elem / 32u;
    uint l    = (elem % 32u) / 8u;
    uint j    = elem & 7u;

    float d = mmid_read_f16(buf, block_off);
    uint q3_off = block_off + 2u + 8u * ib32 + 2u * l;
    uint q3a = mmid_read_byte(buf, q3_off);
    uint q3b = mmid_read_byte(buf, q3_off + 1u);

    uint aux32 = mmid_read_u32_unaligned(buf, block_off + 66u + 4u * ib32);
    uint signs = (aux32 >> (l * 7u)) & 0x7Fu;
    uint parity = countbits(signs) & 1u;
    float scale = d * 0.5f * (0.5f + (float)(aux32 >> 28));

    uint gword = (j < 4u) ? iq3xxs_grid[q3a] : iq3xxs_grid[q3b];
    uint gbyte = (gword >> ((j & 3u) * 8u)) & 0xFFu;
    uint sbit = (j < 7u) ? ((signs >> j) & 1u) : parity;
    float sgn = (sbit != 0u) ? -1.0f : 1.0f;
    return scale * sgn * (float)gbyte;
}
#elif defined(MMID_IQ3_S)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 110
#include "iq3s_grid.hlsli"
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint ib32 = elem / 32u;
    uint l    = (elem % 32u) / 8u;
    uint j    = elem & 7u;

    float d = mmid_read_f16(buf, block_off);
    uint qs_a = mmid_read_byte(buf, block_off + 2u + 8u * ib32 + 2u * l);
    uint qs_b = mmid_read_byte(buf, block_off + 2u + 8u * ib32 + 2u * l + 1u);
    uint qh   = mmid_read_byte(buf, block_off + 66u + ib32);
    uint bit_a = (qh >> (2u * l))      & 1u;
    uint bit_b = (qh >> (2u * l + 1u)) & 1u;
    uint idx_a = qs_a | (bit_a << 8u);
    uint idx_b = qs_b | (bit_b << 8u);

    uint signs = mmid_read_byte(buf, block_off + 74u + 4u * ib32 + l);

    uint scale_byte = mmid_read_byte(buf, block_off + 106u + (ib32 >> 1));
    uint scale_nib = (scale_byte >> (4u * (ib32 & 1u))) & 0xFu;
    float dl = d * (1.0f + 2.0f * (float)scale_nib);

    uint gword = (j < 4u) ? iq3s_grid[idx_a] : iq3s_grid[idx_b];
    uint gbyte = (gword >> ((j & 3u) * 8u)) & 0xFFu;
    uint sbit = (signs >> j) & 1u;
    float sgn = (sbit != 0u) ? -1.0f : 1.0f;
    return dl * sgn * (float)gbyte;
}
#elif defined(MMID_IQ1_S)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 50
#define MMID_IQ1S_DELTA 0.125f
#include "iq1_grid.hlsli"
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint ib32 = elem / 32u;
    uint l    = (elem % 32u) / 8u;
    uint j    = elem & 7u;

    float d = mmid_read_f16(buf, block_off);
    uint qs_byte = mmid_read_byte(buf, block_off + 2u + 4u * ib32 + l);
    uint qh_lo = mmid_read_byte(buf, block_off + 34u + 2u * ib32);
    uint qh_hi = mmid_read_byte(buf, block_off + 34u + 2u * ib32 + 1u);
    uint qh_full = qh_lo | (qh_hi << 8u);

    uint hi3 = (qh_full >> (3u * l)) & 0x7u;
    uint grid_idx = qs_byte | (hi3 << 8u);

    float dl = d * (2.0f * (float)((qh_full >> 12) & 0x7u) + 1.0f);
    float ml = dl * (((qh_full & 0x8000u) != 0u) ? (-1.0f - MMID_IQ1S_DELTA) : (-1.0f + MMID_IQ1S_DELTA));

    uint grid = iq1s_grid_gpu[grid_idx];
    uint b = j & 3u;
    uint nib_high = j >> 2;
    uint gbyte = (grid >> (b * 8u)) & 0xFFu;
    uint nib   = (nib_high == 0u) ? (gbyte & 0xFu) : ((gbyte >> 4) & 0xFu);
    return dl * (float)nib + ml;
}
#elif defined(MMID_IQ1_M)
#define MMID_QK 256
#define MMID_BLOCK_SIZE 56
#define MMID_IQ1M_DELTA 0.125f
#include "iq1_grid.hlsli"
float mmid_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / MMID_QK) * MMID_BLOCK_SIZE;
    uint elem = k % MMID_QK;
    uint ib32 = elem / 32u;
    uint l    = (elem % 32u) / 8u;
    uint j    = elem & 7u;

    uint sc0_lo = mmid_read_byte(buf, block_off + 48u);
    uint sc0_hi = mmid_read_byte(buf, block_off + 49u);
    uint sc1_lo = mmid_read_byte(buf, block_off + 50u);
    uint sc1_hi = mmid_read_byte(buf, block_off + 51u);
    uint sc2_lo = mmid_read_byte(buf, block_off + 52u);
    uint sc2_hi = mmid_read_byte(buf, block_off + 53u);
    uint sc3_lo = mmid_read_byte(buf, block_off + 54u);
    uint sc3_hi = mmid_read_byte(buf, block_off + 55u);
    uint sc0 = sc0_lo | (sc0_hi << 8u);
    uint sc1 = sc1_lo | (sc1_hi << 8u);
    uint sc2 = sc2_lo | (sc2_hi << 8u);
    uint sc3 = sc3_lo | (sc3_hi << 8u);
    uint d_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0u) | ((sc2 >> 4) & 0x0f00u) | (sc3 & 0xf000u);
    float d = f16_to_f32(d_u16);

    uint il = l >> 1;
    uint pair_in_il = l & 1u;

    uint qs_byte = mmid_read_byte(buf, block_off + 4u * ib32 + l);
    uint qh_byte = mmid_read_byte(buf, block_off + 32u + 2u * ib32 + il);

    uint sh = (pair_in_il == 0u) ? 8u : 4u;
    uint hi3 = (qh_byte << sh) & 0x700u;
    uint grid_idx = qs_byte | hi3;

    uint sc_word = (ib32 < 2u) ? sc0 : ((ib32 < 4u) ? sc1 : ((ib32 < 6u) ? sc2 : sc3));
    uint dl_bits = (sc_word >> (6u * (ib32 & 1u) + 3u * il)) & 0x7u;
    float dl = d * (2.0f * (float)dl_bits + 1.0f);

    uint sign_mask = (pair_in_il == 0u) ? 0x08u : 0x80u;
    float ml = dl * (((qh_byte & sign_mask) != 0u) ? (-1.0f - MMID_IQ1M_DELTA) : (-1.0f + MMID_IQ1M_DELTA));

    uint grid = iq1s_grid_gpu[grid_idx];
    uint b = j & 3u;
    uint nib_high = j >> 2;
    uint gbyte = (grid >> (b * 8u)) & 0xFFu;
    uint nib   = (nib_high == 0u) ? (gbyte & 0xFu) : ((gbyte >> 4) & 0xFu);
    return dl * (float)nib + ml;
}
#else
#error "mul_mat_id_quant.hlsli included without an MMID_* quant macro"
#endif