// mul_mat_q3k.hlsl - Generic MUL_MAT (any M) for Q3_K weights.
// 1 thread per output element (slow but correct). Used for batch (M>1) prompt
// processing; matvec path uses mul_mat_vec_q3k.hlsl.
//
// Q3_K block (110 bytes):
//   offset   0..31  : hmask[32] (1 bit per element, high bit of 3-bit quant)
//   offset  32..95  : qs[64]    (2 bits per element, low bits)
//   offset  96..107 : scales[12] (16 packed 6-bit values, biased; subtract 32)
//   offset 108..109 : d (fp16)
//
// Scale unpack from C dequantize_row_q3_K:
//   memcpy(aux[0..2], &scales[0], 12)
//   tmp = aux[2]
//   aux[0] = (aux[0] & 0x0F0F0F0F) | (((tmp>>0) & 0x03030303) << 4)  -> scales[ 0..3]
//   aux[1] = (aux[1] & 0x0F0F0F0F) | (((tmp>>2) & 0x03030303) << 4)  -> scales[ 4..7]
//   aux[2] = ((aux[0]>>4) & 0x0F0F0F0F) | (((tmp>>4) & 0x03030303) << 4)  -> scales[ 8..11]
//                       ^ the *original* aux[0]
//   aux[3] = ((aux[1]>>4) & 0x0F0F0F0F) | (((tmp>>6) & 0x03030303) << 4)  -> scales[12..15]
//                       ^ the *original* aux[1]
//
// Element ordering (canonical):
//   is = 0; m = 1
//   for n in [0,128) step 128:
//     for j in [0,4):
//       shift = 2*j
//       dl = d * (scales[is++] - 32)
//       for l in [0,16): y[n + j*32 + l + 0]  = dl * ((q[l   ]>>shift)&3) - ((hm[l   ]&m)?0:4)
//       dl = d * (scales[is++] - 32)
//       for l in [0,16): y[n + j*32 + l + 16] = dl * ((q[l+16]>>shift)&3) - ((hm[l+16]&m)?0:4)
//       m <<= 1
//     q += 32
//   (hmask is *not* advanced across n)
//
// Q3_K block size 110 is NOT 4-aligned, so use safe byte/word readers.

#include "ggml_common.hlsli"

#define QK_K 256
#define Q3K_BSIZE 110

uint read_byte_q3k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

uint read_u16_q3k(ByteAddressBuffer buf, uint byte_off) {
    // 2-byte aligned read; handle straddle across 4-byte boundary
    uint lo = read_byte_q3k(buf, byte_off);
    uint hi = read_byte_q3k(buf, byte_off + 1u);
    return lo | (hi << 8u);
}

float read_f16_q3k(ByteAddressBuffer buf, uint byte_off) {
    return f16_to_f32(read_u16_q3k(buf, byte_off));
}

uint read_u32_q3k(ByteAddressBuffer buf, uint byte_off) {
    uint b0 = read_byte_q3k(buf, byte_off);
    uint b1 = read_byte_q3k(buf, byte_off + 1u);
    uint b2 = read_byte_q3k(buf, byte_off + 2u);
    uint b3 = read_byte_q3k(buf, byte_off + 3u);
    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint idx = flat_idx_2d(group_id, local_id);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_row = src1_offset + i1 * nb11 + i2 * nb12 + i3 * nb13;

    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0F0F0F0Fu;

    precise float acc = 0.0f;

    for (uint blk = 0; blk < num_blocks; blk++) {
        uint block_off = src0_row + blk * Q3K_BSIZE;

        float d = read_f16_q3k(src0, block_off + 108);

        // Decode all 16 scales into 4 uint32 words (matches C exactly)
        uint scales_off = block_off + 96;
        uint raw0 = read_u32_q3k(src0, scales_off + 0);
        uint raw4 = read_u32_q3k(src0, scales_off + 4);
        uint raw8 = read_u32_q3k(src0, scales_off + 8);

        uint sc_w0 = (raw0 & kmask2)        | (((raw8 >> 0u) & kmask1) << 4u);  // scales[ 0..3]
        uint sc_w1 = (raw4 & kmask2)        | (((raw8 >> 2u) & kmask1) << 4u);  // scales[ 4..7]
        uint sc_w2 = ((raw0 >> 4u) & kmask2)| (((raw8 >> 4u) & kmask1) << 4u);  // scales[ 8..11]
        uint sc_w3 = ((raw4 >> 4u) & kmask2)| (((raw8 >> 6u) & kmask1) << 4u);  // scales[12..15]

        uint y_block = blk * QK_K;
        uint is = 0;
        uint q_base = 0;
        uint m = 1;

        [unroll] for (uint n_iter = 0; n_iter < 2; n_iter++) {
            [unroll] for (uint j = 0; j < 4; j++) {
                uint shift = j * 2;
                uint y_pos_base = n_iter * 128u + j * 32u;

                // Pull two scales for this j
                uint sc_word_a;
                uint sc_word_b;
                {
                    uint sub_a = is >> 2;
                    uint sub_b = (is + 1u) >> 2;
                    if      (sub_a == 0u) sc_word_a = sc_w0;
                    else if (sub_a == 1u) sc_word_a = sc_w1;
                    else if (sub_a == 2u) sc_word_a = sc_w2;
                    else                  sc_word_a = sc_w3;
                    if      (sub_b == 0u) sc_word_b = sc_w0;
                    else if (sub_b == 1u) sc_word_b = sc_w1;
                    else if (sub_b == 2u) sc_word_b = sc_w2;
                    else                  sc_word_b = sc_w3;
                }
                int sc_a = int((sc_word_a >> ((is & 3u) * 8u)) & 0xFFu) - 32;
                int sc_b = int((sc_word_b >> (((is + 1u) & 3u) * 8u)) & 0xFFu) - 32;
                float dla = d * float(sc_a);
                float dlb = d * float(sc_b);
                is += 2u;

                // Half 1: q[l + 0], hm[l + 0]
                [unroll] for (uint l = 0; l < 16; l++) {
                    uint qb = read_byte_q3k(src0, block_off + 32u + q_base + l);
                    uint hb = read_byte_q3k(src0, block_off + 0u + l);
                    int q_lo = int((qb >> shift) & 3u);
                    int q_hi = ((hb & m) != 0u) ? 0 : 4;
                    float w = dla * float(q_lo - q_hi);
                    uint k = y_block + y_pos_base + l;
                    float x = load_auto(src1, src1_row + k * nb10, src1_esize);
                    acc += w * x;
                }
                // Half 2: q[l + 16], hm[l + 16]
                [unroll] for (uint l = 0; l < 16; l++) {
                    uint qb = read_byte_q3k(src0, block_off + 32u + q_base + 16u + l);
                    uint hb = read_byte_q3k(src0, block_off + 0u + 16u + l);
                    int q_lo = int((qb >> shift) & 3u);
                    int q_hi = ((hb & m) != 0u) ? 0 : 4;
                    float w = dlb * float(q_lo - q_hi);
                    uint k = y_block + y_pos_base + 16u + l;
                    float x = load_auto(src1, src1_row + k * nb10, src1_esize);
                    acc += w * x;
                }
                m <<= 1u;
            }
            q_base += 32u;
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
