// quant_dequant_kv.hlsli - runtime-selected per-element dequant for the
// flash-attention KV cache, used when K and V have different types (the
// macro-selected mmid_dequant in quant_dequant.hlsli can only carry one).
// Bodies mirror quant_dequant.hlsli; keep the two in sync.
//
// Type ids (host: dx12_fa_kv_type_id):
//   0 = float (F32/F16/BF16, handled by the caller via load_auto)
//   1 = Q4_0  2 = Q4_1  3 = Q5_0  4 = Q5_1  5 = Q8_0
//   6 = IQ4_NL  7 = Q1_0  8 = Q2_0
#pragma once
#include "ggml_common.hlsli"

uint kvq_read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

int kvq_read_sbyte(ByteAddressBuffer buf, uint byte_off) {
    uint b = kvq_read_byte(buf, byte_off);
    return (b < 128u) ? (int)b : (int)b - 256;
}

float kvq_read_f16(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint kvq_read_u32_unaligned(ByteAddressBuffer buf, uint byte_off) {
    return kvq_read_byte(buf, byte_off) |
           (kvq_read_byte(buf, byte_off + 1) << 8) |
           (kvq_read_byte(buf, byte_off + 2) << 16) |
           (kvq_read_byte(buf, byte_off + 3) << 24);
}

int kvq_kvalues_iq4nl(uint idx) {
    static const uint packed[4] = {
        0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u
    };
    uint w = packed[idx >> 2];
    uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    return (int)(b << 24) >> 24;
}

float kvq_dequant(ByteAddressBuffer buf, uint row_off, uint k, uint t) {
    if (t == 1u) {          // Q4_0
        uint block_off = row_off + (k / 32u) * 18u;
        uint elem = k % 32u;
        float d = kvq_read_f16(buf, block_off);
        uint qs = kvq_read_byte(buf, block_off + 2 + (elem % 16u));
        int q = (elem < 16u) ? ((int)(qs & 0x0Fu) - 8) : ((int)(qs >> 4) - 8);
        return d * (float)q;
    }
    if (t == 2u) {          // Q4_1
        uint block_off = row_off + (k / 32u) * 20u;
        uint elem = k % 32u;
        float d = kvq_read_f16(buf, block_off);
        float m = kvq_read_f16(buf, block_off + 2);
        uint qs = kvq_read_byte(buf, block_off + 4 + (elem % 16u));
        uint q = (elem < 16u) ? (qs & 0x0Fu) : (qs >> 4);
        return (float)q * d + m;
    }
    if (t == 3u) {          // Q5_0
        uint block_off = row_off + (k / 32u) * 22u;
        uint elem = k % 32u;
        float d = kvq_read_f16(buf, block_off);
        uint qh = kvq_read_u32_unaligned(buf, block_off + 2);
        uint qs = kvq_read_byte(buf, block_off + 6 + (elem % 16u));
        uint xh = (elem < 16u) ? (((qh >> elem) << 4) & 0x10u) : ((qh >> (elem - 4u)) & 0x10u);
        uint ql = (elem < 16u) ? (qs & 0x0Fu) : (qs >> 4);
        return d * (float)((int)(ql | xh) - 16);
    }
    if (t == 4u) {          // Q5_1
        uint block_off = row_off + (k / 32u) * 24u;
        uint elem = k % 32u;
        float d = kvq_read_f16(buf, block_off);
        float m = kvq_read_f16(buf, block_off + 2);
        uint qh = kvq_read_u32_unaligned(buf, block_off + 4);
        uint qs = kvq_read_byte(buf, block_off + 8 + (elem % 16u));
        uint xh = (elem < 16u) ? (((qh >> elem) << 4) & 0x10u) : ((qh >> (elem - 4u)) & 0x10u);
        uint ql = (elem < 16u) ? (qs & 0x0Fu) : (qs >> 4);
        return (float)(ql | xh) * d + m;
    }
    if (t == 5u) {          // Q8_0
        uint block_off = row_off + (k / 32u) * 34u;
        float d = kvq_read_f16(buf, block_off);
        return d * (float)kvq_read_sbyte(buf, block_off + 2 + (k % 32u));
    }
    if (t == 6u) {          // IQ4_NL
        uint block_off = row_off + (k / 32u) * 18u;
        uint elem = k % 32u;
        float d = kvq_read_f16(buf, block_off);
        uint qs = kvq_read_byte(buf, block_off + 2 + (elem % 16u));
        uint q = (elem < 16u) ? (qs & 0x0Fu) : ((qs >> 4) & 0x0Fu);
        return d * (float)kvq_kvalues_iq4nl(q);
    }
    if (t == 7u) {          // Q1_0
        uint block_off = row_off + (k / 128u) * 18u;
        uint elem = k % 128u;
        float d = kvq_read_f16(buf, block_off);
        uint b = kvq_read_byte(buf, block_off + 2 + (elem >> 3));
        return ((b >> (elem & 7u)) & 1u) != 0u ? d : -d;
    }
    // t == 8u: Q2_0
    uint block_off = row_off + (k / 64u) * 18u;
    uint elem = k % 64u;
    float d = kvq_read_f16(buf, block_off);
    uint b = kvq_read_byte(buf, block_off + 2 + (elem >> 2));
    uint q = (b >> ((elem & 3u) * 2u)) & 3u;
    return ((float)q - 1.0f) * d;
}
