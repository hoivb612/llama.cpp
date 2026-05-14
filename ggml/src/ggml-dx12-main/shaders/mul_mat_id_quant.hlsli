// Shared quantized MUL_MAT_ID implementation. Wrapper shaders define exactly
// one MMID_* macro before including this file.
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
#else
#error "mul_mat_id_quant.hlsli included without an MMID_* quant macro"
#endif

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint idx = flat_idx_2d(group_id, local_id);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint ids_off = op0 + i1 * op1 + i2 * op2;
    int expert_id = asint(src2.Load(ids_off));

    uint K = ne00;
    uint i3_src0 = i3 * ne03 / ne3;
    uint src0_row = src0_offset + i0 * nb01 + (uint)expert_id * nb02 + i3_src0 * nb03;
    uint i1_src1 = i1 * ne11 / ne1;
    uint src1_row = src1_offset + i1_src1 * nb11 + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        float w = mmid_dequant(src0, src0_row, k);
        float x = load_auto(src1, src1_row + k * nb10, src1_esize);
        acc += w * x;
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
