// SPDX-FileCopyrightText: Copyright 2024 Arm Ltd.
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-q4.h"
#include "ggml-repack.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>

ggml_tensor_repacking_mode_t tensor_repacking_mode = TENSOR_REPACKING_MODE_NONE;

inline
ggml_tensor_repacking_mode_t
ggml_tensor_repacking_mode (
    void
    )
{
    return tensor_repacking_mode;
}

void 
ggml_set_tensor_repacking_mode (
    ggml_tensor_repacking_mode_t mode
    ) 
{
    tensor_repacking_mode = mode;
}

void
make_q4_0_repack_quant (
    uint64_t ne,
    block_q4_0_repack * out,
    block_q4_0 * in
    )

//
// Convert groups of eight q4_0 quant blocks to one q4_0_repack quant block. The q4_0
// quant blocks are interleaved in the q4_0_repack quant block such that they can be
// loaded and acted on directly.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    block_q8_0 qs_in[8];
    block_q8_0_repack qs_out;

    const __m256i m4 = _mm256_set1_epi8(0xf);

    //
    // Convert groups of eight q4_0 quant blocks into one q4_0_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {
        for (i = 0; i < 8; i += 1) {

            //
            // Unpack 8 q4_0 quant blocks into 8 temporary q8_0 quant blocks.
            //
            // N.B. The multiplier value from the q4_0 quant blocks is moved
            //      directly to the temporary q8_0_repack quant block.
            //

            qs_out.d[i] = in[i].d;
            const __m128i tmp1 = _mm_loadu_si128((const __m128i *)in[i].qs);
            const __m128i tmp2 = _mm_srli_epi16(tmp1, 4);
            __m256i qx = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp1), tmp2, 1);
            qx = _mm256_and_si256(m4, qx);
            _mm256_storeu_si256((__m256i *)qs_in[i].qs, qx);
        }
    
        //
        // Rearrange the quant bytes into lanes of four bytes interleaved into the
        // temporary q8_0_repack quant block.
        // 
    
        for (i = 0; i < 8; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out.qs[offset];
                uint32_t * qs_src = (uint32_t *)&qs_in[i].qs[j * 4 + 0];
                *qs_dst = *qs_src;
                offset += 32;
            }
        }
    
        //
        // Repack the temporary q8_0_repack quant block into the q4_0_repack quant block.
        //
        // N.B. The packing of the low and high nibbles in the q4_0_repack format is performed
        //      64 quant values rather than the 32 quant values of the q4_0 format. This
        //      makes unpacking of the nibble values in the eventual vector dot function
        //      more efficient.
        //
    
        for (i = 0; i < 4; i += 1) {
            const __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&qs_out.qs[i * 64 + 0]);
            __m256i tmp2 = _mm256_loadu_si256((const __m256i *)&qs_out.qs[i * 64 + 32]);
            tmp2 = _mm256_slli_epi16(tmp2, 4);
            tmp2 = _mm256_or_epi32(tmp1, tmp2);
            _mm256_storeu_si256((__m256i *)&out->qs[i * 32], tmp2);
        }

        //
        // Copy the multiplier values to the q4_0_repack quant block.
        //

        const __m128i d = _mm_loadu_si128((__m128i *)qs_out.d);
        _mm_storeu_si128((__m128i *)out->d, d);

        out += 1;
        in += 8;
    }
}

void
make_q2_k_repack_quant (
    uint64_t ne,
    block_q2_K_repack * out,
    block_q2_K * in
    )

//
// Convert one q2_K quant block to one q2_K_repack quant block. The q2_K quant block
// values are interleaved in the q2_K_repack quant block such that they can be loaded
// and acted on directly by the q2_k_q8_k_x8 vector dot code.
//
// N.B. The q2_k quant block and q2_K_repack quant block have the same layout and storage
//      requirements.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    uint8_t qs_in[QK_K];
    uint8_t qs_out[QK_K];

    //
    // Convert one q2_k quant block to one q2_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multipliers/(d-dmin) and scales/(mins-scales) values directly
        // from the input quant block to the output quant block.
        //
    
        out->dm = in->dm;
        memcpy(out->scales, in->scales, sizeof(in->scales));
    
        //
        // Unpack the 2-bit q2_k quant values into an array of 8-bit quant values.
        //

        const __m256i m3 = _mm256_set1_epi8(0x3);

        for (i = 0; i < (QK_K / 128); i += 1) {
            __m256i q2bits = _mm256_loadu_si256((const __m256i*)&in->qs[i * 32]);

            for (j = 0; j < (QK_K / 64); j += 1) {
                __m256i q2x = _mm256_and_si256(q2bits, m3);
                _mm256_storeu_si256((__m256i *)&qs_in[i * 128 + j * 32], q2x);

                q2bits = _mm256_srli_epi16(q2bits, 2);
            }
        }

        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        //
        // N.B. There are 16 scale values for q2_k, and thus, 16 lanes are required.
        //
    
        for (i = 0; i < 16; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 4; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&qs_in[i * 16 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 64;
            }
        }

        //
        // Repack the 8-bit quant values into the q2_k_repack quant block.
        //

        __m512i q2bytes = _mm512_loadu_si512((const __m512i*)&qs_out[0]);
        const __m512i q2bytes_0 = _mm512_loadu_si512((const __m512i*)&qs_out[64]);
        const __m512i q2bytes_1 = _mm512_loadu_si512((const __m512i*)&qs_out[128]);
        const __m512i q2bytes_2 = _mm512_loadu_si512((const __m512i*)&qs_out[192]);

        q2bytes = _mm512_or_si512(q2bytes, _mm512_slli_epi16(q2bytes_0, 2));
        q2bytes = _mm512_or_si512(q2bytes, _mm512_slli_epi16(q2bytes_1, 4));
        q2bytes = _mm512_or_si512(q2bytes, _mm512_slli_epi16(q2bytes_2, 6));

        _mm512_storeu_si512((__m512i *)out->qs, q2bytes);

        out += 1;
        in += 1;
    }
}
                      
void
make_q3_k_repack_quant (
    uint64_t ne,
    block_q3_K_repack * out,
    block_q3_K * in
    )

//
// Convert one q3_K quant block to one q3_K_repack quant block. The q3_K quant block
// values are interleaved in the q3_K_repack quant block such that they can be loaded
// and acted on directly by the q3_k_q8_k_x8 vector dot code.
//
// N.B. The q3_k quant block and q3_K_repack quant block have the same layout and storage
//      requirements.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    uint8_t qs_in[QK_K];
    uint8_t qs_out[QK_K];

    //
    // Convert one q3_k quant block to one q3_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    const __m256i m3_256 = _mm256_set1_epi8(3);
    const __m512i m3_512 = _mm512_set1_epi8(3);
    const __m256i m4_256 = _mm256_set1_epi8(4);

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multiplier and scales values directly from the input quant block
        // to the output quant block.
        //
    
        out->d = in->d;
        memcpy(out->scales, in->scales, sizeof(in->scales));

        //
        // Load the high bits and preshift to first set of bits.
        //

        __m256i hbits = _mm256_loadu_si256((const __m256i*)in->hmask);
        hbits = _mm256_rol_epi32(hbits, 2);

        //
        // Unpack the 3-bit q3_k quant values into an array of 8-bit quant values.
        //

        for (i = 0; i < (QK_K / 128); i += 1) {
            __m256i q2bits = _mm256_loadu_si256((const __m256i*)&in->qs[i * 32]);

            for (j = 0; j < (QK_K / 64); j += 1) {

                //
                // Isolate 2-bit values, merge with high bits, and store the value.
                //

                const __m256i q3l = _mm256_and_si256(q2bits, m3_256);
                const __m256i q3h = _mm256_and_si256(hbits, m4_256);
                const __m256i q3 = _mm256_or_si256(q3l, q3h);

                _mm256_storeu_si256((__m256i *)&qs_in[i * 128 + j * 32], q3);

                hbits = _mm256_ror_epi32(hbits, 1);
                q2bits = _mm256_srli_epi16(q2bits, 2);
            }
        }

        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        //
        // N.B. There are 16 scale values for q3_k, and thus, 16 lanes are required.
        //
    
        for (i = 0; i < 16; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 4; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&qs_in[i * 16 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 64;
            }
        }

        //
        // Repack the 8-bit quant values into the q3_k_repack quant block.
        //
        // N.B. The repacking is on 64-byte boundaries instead of the normal 32-byte
        //      boundaries. This is to facilitate more efficent computation of the
        //      dot product.
        //

        const __m512i q3bytes_0 = _mm512_loadu_si512((const __m512i*)&qs_out[0]);
        const __m512i q3bytes_1 = _mm512_loadu_si512((const __m512i*)&qs_out[64]);
        const __m512i q3bytes_2 = _mm512_loadu_si512((const __m512i*)&qs_out[128]);
        const __m512i q3bytes_3 = _mm512_loadu_si512((const __m512i*)&qs_out[192]);

        //
        // Isolate low bits.
        //

        const __m512i q3lbits_0 = _mm512_and_si512(q3bytes_0, m3_512);
        const __m512i q3lbits_1 = _mm512_and_si512(q3bytes_1, m3_512);
        const __m512i q3lbits_2 = _mm512_and_si512(q3bytes_2, m3_512);
        const __m512i q3lbits_3 = _mm512_and_si512(q3bytes_3, m3_512);

        //
        // Pack low bits into 64 bytes.
        //

        __m512i q2bytes = _mm512_or_si512(q3lbits_0, _mm512_slli_epi16(q3lbits_1, 2));
        q2bytes = _mm512_or_si512(q2bytes, _mm512_slli_epi16(q3lbits_2, 4));
        q2bytes = _mm512_or_si512(q2bytes, _mm512_slli_epi16(q3lbits_3, 6));

        _mm512_storeu_si512((__m512i *)out->qs, q2bytes);

        //
        // Shift the high bits into the sign bit.
        //

        const __m512i q3hbits_0 = _mm512_slli_epi16(q3bytes_0, 5);
        const __m512i q3hbits_1 = _mm512_slli_epi16(q3bytes_1, 5);
        const __m512i q3hbits_2 = _mm512_slli_epi16(q3bytes_2, 5);
        const __m512i q3hbits_3 = _mm512_slli_epi16(q3bytes_3, 5);

        //
        // Collect the high bits into 64-bit masks.
        //
        // N.B. The high bits are stored as the mask of sign bits.
        //

        const __mmask64 high_bits_mask_0 = _mm512_movepi8_mask(q3hbits_0);
        const __mmask64 high_bits_mask_1 = _mm512_movepi8_mask(q3hbits_1);
        const __mmask64 high_bits_mask_2 = _mm512_movepi8_mask(q3hbits_2);
        const __mmask64 high_bits_mask_3 = _mm512_movepi8_mask(q3hbits_3);

        *((uint64_t *)out->hmask + 0) = _cvtmask64_u64(high_bits_mask_0);
        *((uint64_t *)out->hmask + 1) = _cvtmask64_u64(high_bits_mask_1);
        *((uint64_t *)out->hmask + 2) = _cvtmask64_u64(high_bits_mask_2);
        *((uint64_t *)out->hmask + 3) = _cvtmask64_u64(high_bits_mask_3);

        out += 1;
        in += 1;
    }
}
                      
void
make_q4_k_repack_quant (
    uint64_t ne,
    block_q4_K_repack * out,
    block_q4_K * in
    )

//
// Convert one q4_K quant block to one q4_K_repack quant block. The q4_K quant block
// values are interleaved in the q4_K_repack quant block such that they can be loaded
// and acted on directly.
//
// N.B. The q4_k quant block and q4_K_repack quant block have the same layout and storage
//      requirements.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    uint8_t qs_in[QK_K];
    uint8_t qs_out[QK_K];

    //
    // Convert one q4_k quant block to one q4_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multipliers/(d-dmin) and scales/(mins-scales) values directly
        // from the input quant block to the output quant block.
        //
    
        out->dm = in->dm;
        memcpy(out->scales, in->scales, sizeof(in->scales));
    
        //
        // Unpack the 4-bit q4_k quant values into an array of 8-bit quant values.
        //
    
        const __m512i m4 = _mm512_set1_epi8(0xf);
    
        for (i = 0; i < (QK_K / 64); i += 1) {
            __m256i tmp1 = _mm256_loadu_si256((const __m256i *)(in->qs + i * 32));
            __m256i tmp2 = _mm256_srli_epi16(tmp1, 4);
    
            __m512i qx = _mm512_castsi256_si512(tmp1); 
            qx = _mm512_inserti32x8(qx, tmp2, 1);
    
            qx = _mm512_and_si512(m4, qx);
            _mm512_storeu_si512((__m512i *)(&qs_in[i * 64]), qx);
        }
    
        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        // 
    
        for (i = 0; i < 8; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&qs_in[i * 32 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 32;
            }
        }
    
        //
        // Repack the q8_k quant block into the q4_k_repack quant block.
        //
        // N.B. The packing of the low and high nibbles in the q4_k_repack format is
        //      performed as 64 quant values rather than the 32 quant values of the
        //      q4_k format. This makes unpacking of the nibble values in the eventual
        //      vector dot function more efficient.
        //
    
        for (i = 0; i < 4; i += 1) {
            __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&qs_out[i * 64 + 0]);
            __m256i tmp2 = _mm256_loadu_si256((const __m256i *)&qs_out[i * 64 + 32]);
    
            tmp2 = _mm256_slli_epi16(tmp2, 4);
            tmp2 = _mm256_or_epi32(tmp1, tmp2);
            _mm256_storeu_si256((__m256i *)&out->qs[i * 32], tmp2);
        }

        out += 1;
        in += 1;
    }
}

void
make_q6_k_repack_quant (
    uint64_t ne,
    block_q6_K_repack * out,
    block_q6_K * in
    )

//
// Convert one q6_K quant block to one q6_K_repack quant block. The q6_K quant block
// values are interleaved in the q6_K_repack quant block such that they can be loaded
// and acted on directly by the q6_k_q8_k_x8 vector dot code.
//
// N.B. The q6_k quant block and q6_K_repack quant block have the same layout and storage
//      requirements.
//

{

    uint64_t offset;
    uint8_t qs_in[QK_K];
    uint8_t qs_out[QK_K];

    //
    // Convert one q6_k quant block to one q6_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (uint64_t k = 0; k < nb; k += 1) {

        //
        // Copy the multiplier and scales values directly from the input quant block
        // to the output quant block.
        //
    
        out->d = in->d;
        memcpy(out->scales, in->scales, sizeof(in->scales));
    
        //
        // Unpack the 6-bit q6_k quant values into an array of 8-bit quant values.
        //

        const __m256i m4_256 = _mm256_set1_epi8(0xf);
        const __m512i m4_512 = _mm512_set1_epi8(0xf);
        const __m256i m2_256 = _mm256_set1_epi8(0x30);
        const __m512i m2_512 = _mm512_set1_epi8(0x30);

        for (uint64_t j = 0; j < (QK_K / 128); j += 1) {

            //
            // Load the low 4-bit values and the high 2-bit values.
            //

            __m256i q4bits1 = _mm256_loadu_si256((__m256i *)(in->ql + (j * 64) + 0));
            __m256i q4bits2 = _mm256_loadu_si256((__m256i *)(in->ql + (j * 64) + 32));
            const __m256i q2bitsH = _mm256_loadu_si256((__m256i *)(in->qh + (j * 32)));

            //
            // Unpack the high (<5:4>) 2-bit values.
            //

            const __m256i q2_0 = _mm256_and_si256(_mm256_rol_epi64(q2bitsH, 4), m2_256);
            const __m256i q2_1 = _mm256_and_si256(_mm256_rol_epi64(q2bitsH, 2), m2_256);
            const __m256i q2_2 = _mm256_and_si256(q2bitsH, m2_256);
            const __m256i q2_3 = _mm256_and_si256(_mm256_ror_epi64(q2bitsH, 2), m2_256);

            //
            // Unpack the low (<3:0>) 4-bit values and or in the high 2-bit values.
            //

            const __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4_256), q2_0);
            const __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4_256), q2_1);

            q4bits1 = _mm256_srli_epi16(q4bits1, 4);
            const __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4_256), q2_2);

            q4bits2 = _mm256_srli_epi16(q4bits2, 4);
            const __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4_256), q2_3);

            //
            // Store the 8-bit quant values.
            //

            _mm256_storeu_si256((__m256i *)&qs_in[(j * 128) + 0], q6_0);
            _mm256_storeu_si256((__m256i *)&qs_in[(j * 128) + 32], q6_1);
            _mm256_storeu_si256((__m256i *)&qs_in[(j * 128) + 64], q6_2);
            _mm256_storeu_si256((__m256i *)&qs_in[(j * 128) + 96], q6_3);
        }

        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        //
        // N.B. There are 16 scale values for q6_k, and thus, 16 lanes are required.
        //
    
        for (uint64_t i = 0; i < 16; i += 1) {
            offset = i * 4;
    
            for (uint64_t j = 0; j < 4; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&qs_in[i * 16 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 64;
            }
        }

        //
        // Repack the 8-bit quant values into the q6_k_repack quant block.
        //

        const __m512i qs_0 = _mm512_loadu_si512(&qs_out[0]);
        const __m512i qs_1 = _mm512_loadu_si512(&qs_out[64]);
        const __m512i qs_2 = _mm512_loadu_si512(&qs_out[128]);
        const __m512i qs_3 = _mm512_loadu_si512(&qs_out[192]);

        __m512i qs4_l = _mm512_and_si512(qs_0, m4_512);
        __m512i qs4_h = _mm512_and_si512(qs_1, m4_512);
        __m512i qs4 = _mm512_or_si512(qs4_l, _mm512_slli_epi16(qs4_h, 4));
        _mm512_storeu_si512((__m512i *)&out->ql[0], qs4);

        qs4_l = _mm512_and_si512(qs_2, m4_512);
        qs4_h = _mm512_and_si512(qs_3, m4_512);
        qs4 = _mm512_or_si512(qs4_l, _mm512_slli_epi16(qs4_h, 4));
        _mm512_storeu_si512((__m512i *)&out->ql[64], qs4);

        const __m512i qs2_0 = _mm512_srli_epi16(_mm512_and_si512(qs_0, m2_512), 4);
        const __m512i qs2_1 = _mm512_srli_epi16(_mm512_and_si512(qs_1, m2_512), 2);
        const __m512i qs2_2 = _mm512_and_si512(qs_2, m2_512);
        const __m512i qs2_3 = _mm512_slli_epi16(_mm512_and_si512(qs_3, m2_512), 2);
        __m512i qs2 = _mm512_or_si512(qs2_0, qs2_1);
        qs2 = _mm512_or_si512(qs2, qs2_2);
        qs2 = _mm512_or_si512(qs2, qs2_3);
        _mm512_storeu_si512((__m512i *)&out->qh[0], qs2);

        out += 1;
        in += 1;
    }
}
                      
void
make_q8_0_repack_quant (
    uint64_t ne,
    block_q8_0_repack * out,
    block_q8_0 * in
    )

//
// Convert groups of eight q8_0 quant blocks to one q8_0_repack quant block. The q8_0
// quant blocks are interleaved in the q8_0_repack quant block such that they can be
// loaded and acted on directly.
//

{

    int8_t * dst;
    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    int8_t qs_out[QK_K];
    int8_t * src;

    //
    // Convert groups of eight q8_0 quant blocks into one q8_0_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        //
        // N.B. There are 8 scale values, and thus, 8 lanes are required.
        //
        // N.B. A separate output buffer is required since the in and out quants
        //      may share the same memory.
        //

        for (i = 0; i < 8; i += 1) {
            out->d[i] = in[i].d;
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&in[i].qs[j * 4 + 0];
                *qs_dst = *qs_src;
                offset += 32;
            }
        }

        //
        // Copy the rearranged q8_0 quants into the q8_0_repack quants.
        //

        src = qs_out;
        dst = out->qs;

        const __m512i q0 = _mm512_loadu_si512(src);
        const __m512i q1 = _mm512_loadu_si512(src + 64);
        const __m512i q2 = _mm512_loadu_si512(src + 128);
        const __m512i q3 = _mm512_loadu_si512(src + 192);

        _mm512_storeu_si512(dst, q0);
        _mm512_storeu_si512(dst + 64, q1);
        _mm512_storeu_si512(dst + 128, q2);
        _mm512_storeu_si512(dst + 192, q3);

        out += 1;
        in += 8;
    }
}
    
void
make_q236_k_q8_k_repack_quant (
    uint64_t ne,
    block_q8_K_repack * out,
    block_q8_K * in
    )

//
// Convert one q8_K quant block to one q8_K_repack quant block. The q8_K quant block
// values are interleaved in the block_q8_K_repack quant block such that they can be
// loaded and acted on directly in the q2_k_q8_k_x8 vector dot code.
//
// N.B. The block_q8_K quant block and block_q8_K_repack quant block have the same
//      layout and storage requirements.
//

{

    int8_t * dst;
    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    int8_t qs_out[QK_K];
    int8_t * src;

    //
    // Convert one q8_k quant block to one q8_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multiplier/(d) value from the input quant block to the output
        // quant block.
        //
    
        out->d = in->d;

        //
        // Copy the bsum values from the input quant block to the output quant block.
        //

        memcpy(out->bsums, in->bsums, sizeof(out->bsums));

        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        //
        // N.B. There are 16 scale values, and thus, 16 lanes are required.
        //
        // N.B. A separate output buffer is required since the in and out quants
        //      may share the same memory.
        //

        for (i = 0; i < 16; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 4; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&in->qs[i * 16 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 64;
            }
        }
    
        //
        // Copy the rearranged q8_k quant block into the q8_k_repack quant block.
        //

        src = qs_out;
        dst = out->qs;

        const __m512i q0 = _mm512_loadu_si512(src);
        const __m512i q1 = _mm512_loadu_si512(src + 64);
        const __m512i q2 = _mm512_loadu_si512(src + 128);
        const __m512i q3 = _mm512_loadu_si512(src + 192);

        _mm512_storeu_si512(dst, q0);
        _mm512_storeu_si512(dst + 64, q1);
        _mm512_storeu_si512(dst + 128, q2);
        _mm512_storeu_si512(dst + 192, q3);

        out += 1;
        in += 1;
    }
}

void
make_q4_k_q8_k_repack_quant (
    uint64_t ne,
    block_q8_K_repack * out,
    block_q8_K * in
    )

//
// Convert one q8_K quant block to one q8_K_repack quant block. The q8_K quant block
// values are interleaved in the block_q8_K_repack quant block such that they can be
// loaded and acted on directly in the q4_k_q8_k vector dot code.
//
// N.B. The block_q8_K quant block and block_q8_K_repack quant block have the same
//      layout and storage requirements.
//

{

    int8_t * dst;
    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    int8_t qs_out[QK_K];
    int8_t * src;

    //
    // Convert one q8_k quant block to one q8_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multiplier/(d) value from the input quant block to the output
        // quant block.
        //
    
        out->d = in->d;

        //
        // Precompute bsums half add that is required in the q4_k_q8_k vector dot
        // function.
        //

        __m128i bsums0 = _mm_loadu_si128((__m128i *)&in->bsums[0]);
        __m128i bsums1 = _mm_loadu_si128((__m128i *)&in->bsums[8]);
        bsums0 = _mm_hadd_epi16(bsums0, bsums1);
        _mm_storeu_si128((__m128i *)&out->bsums[0], bsums0);
        _mm_storeu_si128((__m128i *)&out->bsums[8], bsums0);

        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        //
        // N.B. There are 8 scale values, and thus, 8 lanes are required.
        //
        // N.B. A separate output buffer is required since the in and out quants
        //      may share the same memory.
        //
    
        for (i = 0; i < 8; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&in->qs[i * 32 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 32;
            }
        }
    
        //
        // Copy the rearranged q8_k quant block into the q8_k_repack quant block.
        //

        src = qs_out;
        dst = out->qs;

        const __m512i q0 = _mm512_loadu_si512(src);
        const __m512i q1 = _mm512_loadu_si512(src + 64);
        const __m512i q2 = _mm512_loadu_si512(src + 128);
        const __m512i q3 = _mm512_loadu_si512(src + 192);

        _mm512_storeu_si512(dst, q0);
        _mm512_storeu_si512(dst + 64, q1);
        _mm512_storeu_si512(dst + 128, q2);
        _mm512_storeu_si512(dst + 192, q3);

        out += 1;
        in += 1;
    }
}

//#define ORIGINAL_VERSION 1

#if ORIGINAL_VERSION

void
xx_vec_dot_q4_0_q8_0_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q4_0_repack * vx,
    size_t bx,
    const block_q8_0_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i offset = _mm512_set1_epi8(8);
    const __m512i m4 = _mm512_set1_epi8(0xf);

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_0_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate throught the specified number of rows
        //

        const block_q4_0_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {
        
                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();
        
                //
                // Compute combined multiplier for the quant block.
                //
        
                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m128h yd = _mm_loadu_ph(y[i].d);
                const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                                   _mm256_cvtph_ps(yd));
        
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);
        
                //
                // Compute the dot product and accumulate.
                //
        
                for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {
                    const __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&x[i].qs[j * QK4_0 + 0]);
                    __m512i qy = _mm512_loadu_si512((const __m512i *)&y[i].qs[j * QK8_0 * 2 + 0]);
        
                    const __m256i tmp2 = _mm256_srli_epi16(tmp1, 4);
                    __m512i qx = _mm512_inserti32x8(_mm512_castsi256_si512(tmp1), tmp2, 1);
                    qx = _mm512_and_si512(m4, qx);
        
                    //
                    // Multiply unsigned scaled q4 quant values by signed q8 quant
                    // values and acculate the results.
                    //
        
                    sumi = _mm512_dpbusd_epi32(sumi, qx, qy);
        
                    //
                    // Multiply the unsigned bias quant values by the signed q8 quant
                    // values and acculate the results.
                    //
        
                    bias = _mm512_dpbusd_epi32(bias, offset, qy);
                }
        
                //
                // Subtract the bias values from the sum, convert to float, multiply
                // by the block multiplier, and accumulate the results.
                //
        
                sumi = _mm512_sub_epi32(sumi, bias);
                acc = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(sumi), acc);
            }
        
            const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                             _mm512_extractf32x8_ps(acc, 1));
        
            const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                         _mm256_extractf128_ps(res, 1));
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    
            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

#else

//
// Batch-optimized Q4_0 x Q8_0 vec_dot for prompt prefill (ncols >= 4).
//
// Key optimization: processes 4 activation columns per weight load, reducing
// weight memory traffic by ~4x.
//
// Q4_0 difference from K-quants: d is a VECTOR of 8 fp16 values (one per
// sub-block), not a single scalar. SHARED: weight d[8] loaded once.
// PER-COLUMN: activation d[8] loaded, element-wise multiply with weight d.
//
// Register budget (AVX-512, 32 zmm):
//   4 x acc (zmm) + 4 x sumi (zmm) + 4 xx bias (zmm) +
//   1 x qx (zmm) + 1 x qy (zmm) + offset + m4 x 15 zmm
//

void
xx_vec_dot_q4_0_q8_0_x8 (
    uint32_t n,
    float * s_base,
    size_t nr_nb1,
    const block_q4_0_repack * x_base,
    size_t bx,
    const block_q8_0_repack * y_base,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i offset = _mm512_set1_epi8(8);
    const __m512i m4 = _mm512_set1_epi8(0xf);

    const uint64_t ncols4 = ncols & ~3ULL;
    uint64_t l = 0;

    //
    // Phase 1: Process 4 activation columns at a time.
    //
    // For each weight row + super-block, load weight qs and d once,
    // unpack nibbles once, then dpbusd against 4 activation columns.
    //

    for (; l < ncols4; l += 4) {

        const block_q8_0_repack * y0 = y_base + (l + 0) * nb;
        const block_q8_0_repack * y1 = y_base + (l + 1) * nb;
        const block_q8_0_repack * y2 = y_base + (l + 2) * nb;
        const block_q8_0_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        const block_q4_0_repack * x = x_base;

        for (uint64_t k = 0; k < nrows; k += 1) {

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                //
                // Per-column integer accumulators.
                //

                __m512i sumi0 = _mm512_setzero_si512();
                __m512i sumi1 = _mm512_setzero_si512();
                __m512i sumi2 = _mm512_setzero_si512();
                __m512i sumi3 = _mm512_setzero_si512();

                __m512i bias0 = _mm512_setzero_si512();
                __m512i bias1 = _mm512_setzero_si512();
                __m512i bias2 = _mm512_setzero_si512();
                __m512i bias3 = _mm512_setzero_si512();

                //
                // SHARED: Load weight q4 data once per sub-block, unpack nibbles,
                // then dpbusd against 4 activation columns.
                //

                for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {

                    //
                    // SHARED: unpack weight nibbles.
                    //

                    const __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&x[i].qs[j * QK4_0 + 0]);
                    const __m256i tmp2 = _mm256_srli_epi16(tmp1, 4);
                    __m512i qx = _mm512_inserti32x8(_mm512_castsi256_si512(tmp1), tmp2, 1);
                    qx = _mm512_and_si512(m4, qx);

                    //
                    // Per-column: load q8 and dpbusd against shared qx.
                    //

                    const __m512i qy0 = _mm512_loadu_si512((const __m512i *)&y0[i].qs[j * QK8_0 * 2]);
                    sumi0 = _mm512_dpbusd_epi32(sumi0, qx, qy0);
                    bias0 = _mm512_dpbusd_epi32(bias0, offset, qy0);

                    const __m512i qy1 = _mm512_loadu_si512((const __m512i *)&y1[i].qs[j * QK8_0 * 2]);
                    sumi1 = _mm512_dpbusd_epi32(sumi1, qx, qy1);
                    bias1 = _mm512_dpbusd_epi32(bias1, offset, qy1);

                    const __m512i qy2 = _mm512_loadu_si512((const __m512i *)&y2[i].qs[j * QK8_0 * 2]);
                    sumi2 = _mm512_dpbusd_epi32(sumi2, qx, qy2);
                    bias2 = _mm512_dpbusd_epi32(bias2, offset, qy2);

                    const __m512i qy3 = _mm512_loadu_si512((const __m512i *)&y3[i].qs[j * QK8_0 * 2]);
                    sumi3 = _mm512_dpbusd_epi32(sumi3, qx, qy3);
                    bias3 = _mm512_dpbusd_epi32(bias3, offset, qy3);
                }

                //
                // SHARED: Load weight d[8] (fp16 - fp32) once.
                //

                const __m128h xd = _mm_loadu_ph(x[i].d);
                __m256 x_fp32 = _mm256_cvtph_ps(xd);

                //
                // Per-column: subtract bias, build per-column scale vector, FMA.
                //

#define Q40_COL_FMA(col, y_ptr, sumi_v, bias_v, acc_v) \
                do { \
                    const __m128h yd_##col = _mm_loadu_ph((y_ptr)[i].d); \
                    const __m256 scale_##col = _mm256_mul_ps(x_fp32, _mm256_cvtph_ps(yd_##col)); \
                    __m512 d_##col = _mm512_castps256_ps512(scale_##col); \
                    d_##col = _mm512_insertf32x8(d_##col, scale_##col, 1); \
                    sumi_v = _mm512_sub_epi32(sumi_v, bias_v); \
                    acc_v = _mm512_fmadd_ps(d_##col, _mm512_cvtepi32_ps(sumi_v), acc_v); \
                } while (0)

                Q40_COL_FMA(0, y0, sumi0, bias0, acc0);
                Q40_COL_FMA(1, y1, sumi1, bias1, acc1);
                Q40_COL_FMA(2, y2, sumi2, bias2, acc2);
                Q40_COL_FMA(3, y3, sumi3, bias3, acc3);

#undef Q40_COL_FMA

            }

            //
            // Reduce and store results for each of the 4 columns.
            //

#define REDUCE_Q40_CP(acc, dst) \
            do { \
                const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc), \
                                                 _mm512_extractf32x8_ps(acc, 1)); \
                const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res), \
                                             _mm256_extractf128_ps(res, 1)); \
                const __m128 t1 = _mm_hadd_ps(t0, t0); \
                dst[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1)); \
            } while (0)

            REDUCE_Q40_CP(acc0, s0);
            REDUCE_Q40_CP(acc1, s1);
            REDUCE_Q40_CP(acc2, s2);
            REDUCE_Q40_CP(acc3, s3);

#undef REDUCE_Q40_CP

            x += nb;
        }
    }

    //
    // Phase 2: Process remaining columns one at a time (0-3 columns).
    //

    const block_q8_0_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        const block_q4_0_repack * x = x_base;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();

                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m128h yd = _mm_loadu_ph(y_rem[i].d);
                const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                                   _mm256_cvtph_ps(yd));
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);

                for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {
                    const __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&x[i].qs[j * QK4_0 + 0]);
                    __m512i qy = _mm512_loadu_si512((const __m512i *)&y_rem[i].qs[j * QK8_0 * 2 + 0]);

                    const __m256i tmp2 = _mm256_srli_epi16(tmp1, 4);
                    __m512i qx = _mm512_inserti32x8(_mm512_castsi256_si512(tmp1), tmp2, 1);
                    qx = _mm512_and_si512(m4, qx);

                    sumi = _mm512_dpbusd_epi32(sumi, qx, qy);
                    bias = _mm512_dpbusd_epi32(bias, offset, qy);
                }

                sumi = _mm512_sub_epi32(sumi, bias);
                acc = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(sumi), acc);
            }

            const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                             _mm512_extractf32x8_ps(acc, 1));

            const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                          _mm256_extractf128_ps(res, 1));

            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s_rem[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s_rem = (float *)((char *)s_rem + nr_nb1);
        y_rem += nb;
    }
}

#endif // ORIGINAL_VERSION

#if ORIGINAL_VERSION

void
xx_vec_dot_q2_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q2_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i m3 = _mm512_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_K_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        const block_q2_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();
            __m256 mins_acc = _mm256_setzero_ps();
        
            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
                const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);
        
                const uint8_t * q2 = x[i].qs;
                const int8_t * q8 = y[i].qs;
        
                const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
                __m512i q2bits = _mm512_loadu_si512((const __m512i*)q2);
        
                const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
                const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        
                const __m512i scale = _mm512_cvtepi8_epi32(scales8);
        
                __m512i sumi = _mm512_setzero_si512();
        
                //
                // Compute the integer product of the q2 and q8 quants and accumulate the
                // integer results.
                //
        
                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));
                    const __m512i q2v = _mm512_and_si512(q2bits, m3);
                    q2bits = _mm512_srli_epi16(q2bits, 2);
        
                    sumi = _mm512_dpbusd_epi32(sumi, q2v, q8v);
                }
        
                //
                // Multiply the accumulated integer result by the q2 scale, convert to float,
                // multiply by the q8 multiplier, and accumulate the results.
                //
        
                sumi = _mm512_mullo_epi32(sumi, scale);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
        
                //
                // Load 16 q8 bsums values, multiply by mins values, convert to float,
                // and accumulate the floating results.
                //
        
                const __m256i bsums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
                const __m256i mins = _mm256_cvtepi8_epi16(mins8);
                const __m256i prod = _mm256_madd_epi16(mins, bsums);
                mins_acc = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), mins_acc);
            }
        
            __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                       _mm512_extractf32x8_ps(acc, 1));
        
            res = _mm256_sub_ps(res, mins_acc);
        
            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

#else

void
xx_vec_dot_q2_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q2_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i m3 = _mm512_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_K_repack * y_base = vy;
    float * s_base = s;

    const uint64_t ncols4 = ncols & ~3ULL;
    uint64_t l = 0;

    //
    // Phase 1: Process 4 activation columns at a time.
    //

    for (; l < ncols4; l += 4) {

        const block_q8_K_repack * y0 = y_base + (l + 0) * nb;
        const block_q8_K_repack * y1 = y_base + (l + 1) * nb;
        const block_q8_K_repack * y2 = y_base + (l + 2) * nb;
        const block_q8_K_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        //
        // Iterate through the specified number of rows.
        //

        const block_q2_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            __m256 mins_acc0 = _mm256_setzero_ps();
            __m256 mins_acc1 = _mm256_setzero_ps();
            __m256 mins_acc2 = _mm256_setzero_ps();
            __m256 mins_acc3 = _mm256_setzero_ps();

            //
            // Iterate through the quant blocks 4 columns at a time.
            //

            for (uint64_t i = 0; i < nb; ++i) {

                __m512i sumi0 = _mm512_setzero_si512();
                __m512i sumi1 = _mm512_setzero_si512();
                __m512i sumi2 = _mm512_setzero_si512();
                __m512i sumi3 = _mm512_setzero_si512();
        
                //
                // Compute the integer product of the q2 and q8 quants against 4
                // columns and accumulate the integer results.
                //
        
                __m512i q2bits = _mm512_loadu_si512((const __m512i*)x[i].qs);

                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m512i q2v = _mm512_and_si512(q2bits, m3);
                    q2bits = _mm512_srli_epi16(q2bits, 2);

                    sumi0 = _mm512_dpbusd_epi32(sumi0, q2v, _mm512_loadu_si512(&y0[i].qs[j * 64]));
                    sumi1 = _mm512_dpbusd_epi32(sumi1, q2v, _mm512_loadu_si512(&y1[i].qs[j * 64]));
                    sumi2 = _mm512_dpbusd_epi32(sumi2, q2v, _mm512_loadu_si512(&y2[i].qs[j * 64]));
                    sumi3 = _mm512_dpbusd_epi32(sumi3, q2v, _mm512_loadu_si512(&y3[i].qs[j * 64]));
                }

                //
                // Compute row scale and mins values.
                //

                const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        
                const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
                const __m512i scale = _mm512_cvtepi8_epi32(scales8);

                const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
                const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        
                //
                // Apply scale to all 4 column accumlators.
                //

                sumi0 = _mm512_mullo_epi32(sumi0, scale);
                sumi1 = _mm512_mullo_epi32(sumi1, scale);
                sumi2 = _mm512_mullo_epi32(sumi2, scale);
                sumi3 = _mm512_mullo_epi32(sumi3, scale);

                //
                // Compute d-scale for 4 columns.
                //

                const float x_d = GGML_FP16_TO_FP32(x[i].d);

                const float d0 = y0[i].d * x_d;
                const float d1 = y1[i].d * x_d;
                const float d2 = y2[i].d * x_d;
                const float d3 = y3[i].d * x_d;

                //
                // Accumulate 4 columns into accumulators.
                //

                acc0 = _mm512_fmadd_ps(_mm512_set1_ps(d0), _mm512_cvtepi32_ps(sumi0), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_set1_ps(d1), _mm512_cvtepi32_ps(sumi1), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_set1_ps(d2), _mm512_cvtepi32_ps(sumi2), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_set1_ps(d3), _mm512_cvtepi32_ps(sumi3), acc3);

                //
                // Compute dmin for 4 columns.
                //

                const float x_dmin = GGML_FP16_TO_FP32(x[i].dmin);
        
                const float dmin0 = y0[i].d * x_dmin;
                const float dmin1 = y1[i].d * x_dmin;
                const float dmin2 = y2[i].d * x_dmin;
                const float dmin3 = y3[i].d * x_dmin;

                //
                // Load 16 q8 bsums values, multiply by mins values, convert to float,
                // and accumulate the floating results for 4 columns.
                //
        
                const __m256i bsums0 = _mm256_loadu_si256((const __m256i*)y0[i].bsums);
                const __m256i bsums1 = _mm256_loadu_si256((const __m256i*)y1[i].bsums);
                const __m256i bsums2 = _mm256_loadu_si256((const __m256i*)y2[i].bsums);
                const __m256i bsums3 = _mm256_loadu_si256((const __m256i*)y3[i].bsums);

                const __m256i prod0 = _mm256_madd_epi16(mins, bsums0);
                const __m256i prod1 = _mm256_madd_epi16(mins, bsums1);
                const __m256i prod2 = _mm256_madd_epi16(mins, bsums2);
                const __m256i prod3 = _mm256_madd_epi16(mins, bsums3);

                mins_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(dmin0), _mm256_cvtepi32_ps(prod0), mins_acc0);
                mins_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(dmin1), _mm256_cvtepi32_ps(prod1), mins_acc1);
                mins_acc2 = _mm256_fmadd_ps(_mm256_set1_ps(dmin2), _mm256_cvtepi32_ps(prod2), mins_acc2);
                mins_acc3 = _mm256_fmadd_ps(_mm256_set1_ps(dmin3), _mm256_cvtepi32_ps(prod3), mins_acc3);
            }

            //
            // Reduce and store results for 4 columns.
            //

#define REDUCE_Q2K_MINS(acc, mins_acc, dst) \
            do { \
                __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc), \
                                           _mm512_extractf32x8_ps(acc, 1)); \
                res = _mm256_sub_ps(res, mins_acc); \
                __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res), \
                                       _mm256_extractf128_ps(res, 1)); \
                const __m128 t1 = _mm_hadd_ps(t0, t0); \
                dst[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1)); \
            } while (0);

            REDUCE_Q2K_MINS(acc0, mins_acc0, s0);
            REDUCE_Q2K_MINS(acc1, mins_acc1, s1);
            REDUCE_Q2K_MINS(acc2, mins_acc2, s2);
            REDUCE_Q2K_MINS(acc3, mins_acc3, s3);

#undef REDUCE_Q2K_MINS

            x += nb;
        }
    }

    //
    // Phase 2: Process remaining columns one at a time (original path).
    //

    const block_q8_K_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        const block_q2_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();
            __m256 mins_acc = _mm256_setzero_ps();
        
            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y_rem[i].d * GGML_FP16_TO_FP32(x[i].d);
                const float dmin = y_rem[i].d * GGML_FP16_TO_FP32(x[i].dmin);
        
                const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
                __m512i q2bits = _mm512_loadu_si512((const __m512i*)x[i].qs);
        
                const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
                const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        
                const __m512i scale = _mm512_cvtepi8_epi32(scales8);
        
                __m512i sumi = _mm512_setzero_si512();
        
                //
                // Compute the integer product of the q2 and q8 quants and accumulate the
                // integer results.
                //
        
                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m512i q8v = _mm512_loadu_si512(&y_rem[i].qs[j * 64]);
                    const __m512i q2v = _mm512_and_si512(q2bits, m3);
                    q2bits = _mm512_srli_epi16(q2bits, 2);
        
                    sumi = _mm512_dpbusd_epi32(sumi, q2v, q8v);
                }
        
                //
                // Multiply the accumulated integer result by the q2 scale, convert to float,
                // multiply by the q8 multiplier, and accumulate the results.
                //
        
                sumi = _mm512_mullo_epi32(sumi, scale);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
        
                //
                // Load 16 q8 bsums values, multiply by mins values, convert to float,
                // and accumulate the floating results.
                //
        
                const __m256i bsums = _mm256_loadu_si256((const __m256i*)y_rem[i].bsums);
                const __m256i mins = _mm256_cvtepi8_epi16(mins8);
                const __m256i prod = _mm256_madd_epi16(mins, bsums);
                mins_acc = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), mins_acc);
            }
        
            __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                       _mm512_extractf32x8_ps(acc, 1));
        
            res = _mm256_sub_ps(res, mins_acc);
        
            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s_rem[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s_rem = (float *)((char *)s_rem + nr_nb1);
        y_rem += nb;
    }
}

#endif // ORIGINAL_VERSION

#if ORIGINAL_VERSION

void
xx_vec_dot_q3_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q3_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t kmask1 = 0x0000000030303030ull;
    const uint64_t kmask2 = 0x0f0f0f0f0f0f0f0full;

    const uint64_t nb = n / QK_K;

    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i m4 = _mm512_set1_epi8(4);
    const __m128i m32 = _mm_set1_epi8(32);

    const __m128i zero128 = _mm_setzero_si128();
    const __m512i zero512 = _mm512_setzero_si512();

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_K_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        const block_q3_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();
        
            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        
                const uint8_t * q3 = x[i].qs;
                const int8_t * q8 = y[i].qs;
        
                //
                // Set up scales. There are 16 scales packed in 6-bit format.
                //

                uint64_t scales_h = *((uint64_t *)(x[i].scales));
                uint64_t scales_l = scales_h & kmask2;
                scales_h = (scales_h >> 4) & kmask2;

                uint64_t high_2bits = *((uint32_t *)(x[i].scales + 8));

                scales_h |= ((high_2bits >> 2) & kmask1) << 32;
                scales_h |= (high_2bits & kmask1);

                scales_l |= ((high_2bits << 2) & kmask1) << 32;
                scales_l |= ((high_2bits << 4) & kmask1);

                __m128i scales8 = _mm_insert_epi64(zero128, scales_l, 0);
                scales8 = _mm_insert_epi64(scales8, scales_h, 1);

                scales8 = _mm_sub_epi8(scales8, m32);
                const __m512i scales = _mm512_cvtepi8_epi32(scales8);
        
                //
                // Compute the integer product of the q3 and q8 quants and accumulate the
                // integer results.
                //
        
                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();
        
                __m512i q3bytes = _mm512_loadu_si512((const __m512i*)q3);
        
                for (uint64_t j = 0; j < QK_K / 64; j += 1) {
        
                    //
                    // Load the q8 values.
                    //
        
                    const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));
        
                    //
                    // Isolate the low 2-bits of the q3 values and move to the next value.
                    //
        
                    const __m512i q3lv = _mm512_and_si512(q3bytes, m3);
                    q3bytes = _mm512_srli_epi16(q3bytes, 2);
        
                    //
                    // Multiply the low 2-bit unsigned values by the signed q8 quant
                    // values and acculate the results.
                    //
        
                    sumi = _mm512_dpbusd_epi32(sumi, q3lv, q8v);
        
                    //
                    // Compute the high bits from the q3 mask.
                    //
                    // N.B. The high bit is flipped to conform with the unpacked
                    //      implementation, i.e., and_not.
                    //
        
                    uint64_t high_bits = *((uint64_t *)x[i].hmask + j);
                    const __mmask64 hmask = _cvtu64_mask64(high_bits);
                    const __m512i q3hv = _mm512_mask_blend_epi8(hmask, m4, zero512);
        
                    //
                    // Multiply the unsigned high values by the signed q8 quant
                    // values and accumulate the results.
                    //
        
                    bias = _mm512_dpbusd_epi32(bias, q3hv, q8v);
                }
        
                //
                // Subtract the bias values from the sum, multiply the accumulated sum
                // by the q3 scale, convert to float, multiply by the block multiplier,
                // and accumulate the results.
                //
        
                sumi = _mm512_sub_epi32(sumi, bias);
                sumi = _mm512_mullo_epi32(sumi, scales);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
            }
        
            __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                       _mm512_extractf32x8_ps(acc, 1));
        
            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

#else

void
xx_vec_dot_q3_k_q8_k_x8 (
    uint32_t n,
    float * s_base,
    size_t nr_nb1,
    const block_q3_K_repack * x_base,
    size_t bx,
    const block_q8_K_repack * y_base,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t kmask1 = 0x0000000030303030ull;
    const uint64_t kmask2 = 0x0f0f0f0f0f0f0f0full;

    const uint64_t nb = n / QK_K;

    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i m4 = _mm512_set1_epi8(4);
    const __m128i m32 = _mm_set1_epi8(32);

    const __m128i zero128 = _mm_setzero_si128();
    const __m512i zero512 = _mm512_setzero_si512();

    //
    // Iterate through the specified number of columns and rows.
    //

    const uint64_t ncols4 = ncols & ~3ULL;
    uint64_t l = 0;

    //
    // Phase 1: Process 4 columns at a time.
    //

    for (; l < ncols4; l += 4) {

        const block_q8_K_repack * y0 = y_base + (l + 0) * nb;
        const block_q8_K_repack * y1 = y_base + (l + 1) * nb;
        const block_q8_K_repack * y2 = y_base + (l + 2) * nb;
        const block_q8_K_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        //
        // Iterate through the specified number of rows.
        //

        const block_q3_K_repack * x = x_base;

        for (uint64_t k = 0; k < nrows; k += 1) {

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            //
            // Iterate through the quant blocks.
            //

            for (uint64_t i = 0; i < nb; ++i) {

                __m512i sumi0 = _mm512_setzero_si512();
                __m512i sumi1 = _mm512_setzero_si512();
                __m512i sumi2 = _mm512_setzero_si512();
                __m512i sumi3 = _mm512_setzero_si512();

                __m512i bias0 = _mm512_setzero_si512();
                __m512i bias1 = _mm512_setzero_si512();
                __m512i bias2 = _mm512_setzero_si512();
                __m512i bias3 = _mm512_setzero_si512();

                //
                // Compute the integer product of the q3 and q8 quants and accumulate the
                // integer results.
                //

                __m512i q3bytes = _mm512_loadu_si512(x[i].qs);
        
                for (uint64_t j = 0; j < QK_K / 64; j += 1) {
        
                    //
                    // Isolate the low 2-bits of the q3 values and move to the next value.
                    //
        
                    const __m512i q3lv = _mm512_and_si512(q3bytes, m3);
                    q3bytes = _mm512_srli_epi16(q3bytes, 2);
        
                    //
                    // Compute the high bits from the q3 mask.
                    //
                    // N.B. The high bit is flipped to conform with the unpacked
                    //      implementation, i.e., and_not.
                    //
        
                    uint64_t high_bits = *((uint64_t *)x[i].hmask + j);
                    const __mmask64 hmask = _cvtu64_mask64(high_bits);
                    const __m512i q3hv = _mm512_mask_blend_epi8(hmask, m4, zero512);
        
                    //
                    // Multiply the low 2-bit unsigned values by the signed q8 quant
                    // values and acculate the results.
                    //
                    // Multiply the unsigned high values by the signed q8 quant values
                    // and accumulate the results.
                    //

                    const __m512i q8v0 = _mm512_loadu_si512(&y0[i].qs[j * 64]);
                    sumi0 = _mm512_dpbusd_epi32(sumi0, q3lv, q8v0);
                    bias0 = _mm512_dpbusd_epi32(bias0, q3hv, q8v0);

                    const __m512i q8v1 = _mm512_loadu_si512(&y1[i].qs[j * 64]);
                    sumi1 = _mm512_dpbusd_epi32(sumi1, q3lv, q8v1);
                    bias1 = _mm512_dpbusd_epi32(bias1, q3hv, q8v1);

                    const __m512i q8v2 = _mm512_loadu_si512(&y2[i].qs[j * 64]);
                    sumi2 = _mm512_dpbusd_epi32(sumi2, q3lv, q8v2);
                    bias2 = _mm512_dpbusd_epi32(bias2, q3hv, q8v2);

                    const __m512i q8v3 = _mm512_loadu_si512(&y3[i].qs[j * 64]);
                    sumi3 = _mm512_dpbusd_epi32(sumi3, q3lv, q8v3);
                    bias3 = _mm512_dpbusd_epi32(bias3, q3hv, q8v3);
                }
        
                //
                // Set up scales. There are 16 scales packed in 6-bit format.
                //

                uint64_t scales_h = *((uint64_t *)(x[i].scales));
                uint64_t scales_l = scales_h & kmask2;
                scales_h = (scales_h >> 4) & kmask2;

                uint64_t high_2bits = *((uint32_t *)(x[i].scales + 8));

                scales_h |= ((high_2bits >> 2) & kmask1) << 32;
                scales_h |= (high_2bits & kmask1);

                scales_l |= ((high_2bits << 2) & kmask1) << 32;
                scales_l |= ((high_2bits << 4) & kmask1);

                __m128i scales8 = _mm_insert_epi64(zero128, scales_l, 0);
                scales8 = _mm_insert_epi64(scales8, scales_h, 1);

                scales8 = _mm_sub_epi8(scales8, m32);
                const __m512i scales = _mm512_cvtepi8_epi32(scales8);

                //
                // Subtract the bias values from the sum and scale the result
                // sum values.
                //

                sumi0 = _mm512_sub_epi32(sumi0, bias0);
                sumi0 = _mm512_mullo_epi32(sumi0, scales);

                sumi1 = _mm512_sub_epi32(sumi1, bias1);
                sumi1 = _mm512_mullo_epi32(sumi1, scales);

                sumi2 = _mm512_sub_epi32(sumi2, bias2);
                sumi2 = _mm512_mullo_epi32(sumi2, scales);

                sumi3 = _mm512_sub_epi32(sumi3, bias3);
                sumi3 = _mm512_mullo_epi32(sumi3, scales);

                //
                // Compute the d-scale values.
                //

                const float x_d = GGML_FP16_TO_FP32(x[i].d);

                const float d0 = y0[i].d * x_d;
                const float d1 = y1[i].d * x_d;
                const float d2 = y2[i].d * x_d;
                const float d3 = y3[i].d * x_d;

                //
                // Compute the d-scaled result and accumulate.
                //

                acc0 = _mm512_fmadd_ps(_mm512_set1_ps(d0), _mm512_cvtepi32_ps(sumi0), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_set1_ps(d1), _mm512_cvtepi32_ps(sumi1), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_set1_ps(d2), _mm512_cvtepi32_ps(sumi2), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_set1_ps(d3), _mm512_cvtepi32_ps(sumi3), acc3);
            }

            //
            // Reduce and store results.
            //

#define REDUCE_Q3K(acc, dst) \
            do { \
                __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc), \
                                           _mm512_extractf32x8_ps(acc, 1)); \
                __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res), \
                                       _mm256_extractf128_ps(res, 1)); \
                const __m128 t1 = _mm_hadd_ps(t0, t0); \
                dst[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1)); \
            } while (0);

            REDUCE_Q3K(acc0, s0);
            REDUCE_Q3K(acc1, s1);
            REDUCE_Q3K(acc2, s2);
            REDUCE_Q3K(acc3, s3);

#undef REDUCE_Q3K

            x += nb;
        }
    }

    //
    // Phase 2: Process remaining columns one at a time (original path).
    //

    const block_q8_K_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        const block_q3_K_repack * x = x_base;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();
        
            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y_rem[i].d * GGML_FP16_TO_FP32(x[i].d);
        
                //
                // Set up scales. There are 16 scales packed in 6-bit format.
                //

                uint64_t scales_h = *((uint64_t *)(x[i].scales));
                uint64_t scales_l = scales_h & kmask2;
                scales_h = (scales_h >> 4) & kmask2;

                uint64_t high_2bits = *((uint32_t *)(x[i].scales + 8));

                scales_h |= ((high_2bits >> 2) & kmask1) << 32;
                scales_h |= (high_2bits & kmask1);

                scales_l |= ((high_2bits << 2) & kmask1) << 32;
                scales_l |= ((high_2bits << 4) & kmask1);

                __m128i scales8 = _mm_insert_epi64(zero128, scales_l, 0);
                scales8 = _mm_insert_epi64(scales8, scales_h, 1);

                scales8 = _mm_sub_epi8(scales8, m32);
                const __m512i scales = _mm512_cvtepi8_epi32(scales8);
        
                //
                // Compute the integer product of the q3 and q8 quants and accumulate the
                // integer results.
                //
        
                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();
        
                __m512i q3bytes = _mm512_loadu_si512(x[i].qs);
        
                for (uint64_t j = 0; j < QK_K / 64; j += 1) {
        
                    //
                    // Load the q8 values.
                    //
        
                    const __m512i q8v = _mm512_loadu_si512(&y_rem[i].qs[j * 64]);
        
                    //
                    // Isolate the low 2-bits of the q3 values and move to the next value.
                    //
        
                    const __m512i q3lv = _mm512_and_si512(q3bytes, m3);
                    q3bytes = _mm512_srli_epi16(q3bytes, 2);
        
                    //
                    // Multiply the low 2-bit unsigned values by the signed q8 quant
                    // values and acculate the results.
                    //
        
                    sumi = _mm512_dpbusd_epi32(sumi, q3lv, q8v);
        
                    //
                    // Compute the high bits from the q3 mask.
                    //
                    // N.B. The high bit is flipped to conform with the unpacked
                    //      implementation, i.e., and_not.
                    //
        
                    uint64_t high_bits = *((uint64_t *)x[i].hmask + j);
                    const __mmask64 hmask = _cvtu64_mask64(high_bits);
                    const __m512i q3hv = _mm512_mask_blend_epi8(hmask, m4, zero512);
        
                    //
                    // Multiply the unsigned high values by the signed q8 quant
                    // values and accumulate the results.
                    //
        
                    bias = _mm512_dpbusd_epi32(bias, q3hv, q8v);
                }
        
                //
                // Subtract the bias values from the sum, multiply the accumulated sum
                // by the q3 scale, convert to float, multiply by the block multiplier,
                // and accumulate the results.
                //
        
                sumi = _mm512_sub_epi32(sumi, bias);
                sumi = _mm512_mullo_epi32(sumi, scales);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
            }
        
            __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                       _mm512_extractf32x8_ps(acc, 1));
        
            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s_rem[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s_rem = (float *)((char *)s_rem + nr_nb1);
        y_rem += nb;
    }
}

#endif // ORIGINAL_VERSION

#if ORIGINAL_VERSION

void
xx_vec_dot_q4_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q4_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t kmask4 = 0xc0c0c0c0u;

    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m128i zero128 = _mm_setzero_si128();

    uint64_t utmp[2];

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_K_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate throught the specified number of rows
        //

        const block_q4_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();
            __m128 mins_acc = _mm_setzero_ps();
        
            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
                const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);
        
                const uint32_t * vscales = (uint32_t *)x[i].scales;
                utmp[1] = (uint64_t)(((vscales[2] >> 4) & kmask2) | ((vscales[1] & kmask4) >> 2)) << 32;
                utmp[1] |= (uint64_t)(vscales[1] & kmask1);
                utmp[0] = (uint64_t)((vscales[2] & kmask2) | ((vscales[0] & kmask4) >> 2)) << 32;
                utmp[0] |= (uint64_t)(vscales[0] & kmask1);
        
                const uint8_t * q4 = x[i].qs;
                const int8_t  * q8 = y[i].qs;
        
                //
                // Insert 8 q4 mins and 8 q4 scales.
                //
                // N.B. Both mins and scales are 6-bit unsigned values.
                //
        
                const __m128i scales8 = _mm_insert_epi64(zero128, utmp[0], 0);
                const __m128i mins8 = _mm_insert_epi64(zero128, utmp[1], 0);
        
                //
                // Compute the scale vector.
                //
                // N.B. The 8 scale values are replicated to 16 scale values.
                //
        
                __m512i scale = _mm512_cvtepi8_epi32(scales8);
                scale = _mm512_inserti64x4(scale, _mm512_castsi512_si256(scale), 1);
        
                __m512i sumi = _mm512_setzero_si512();
        
                //
                // Compute the integer product of the q4 and q8 quants and accumulate the
                // integer results.
                //
        
                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j * 32)));
                    const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));
        
                    __m512i q4v = _mm512_castsi256_si512(q4bits);
                    q4v = _mm512_inserti32x8(q4v, _mm256_srli_epi16(q4bits, 4), 1);
                    q4v = _mm512_and_si512(q4v, m4);
        
                    sumi = _mm512_dpbusd_epi32(sumi, q4v, q8v);
                }
        
                //
                // Multiply the accumulated integer result by the q4 scale, convert to float,
                // multiply by the q8 multiplier, and accumulate the results.
                //
        
                sumi = _mm512_mullo_epi32(sumi, scale);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
        
                //
                // Load 8 q8 bsums values, multiply by the mins values, and accumulate
                // the floating results.
                //
                // N.B. The half add to fold the 16 bsum values into 8 values is performed
                //      in the make q8_k quant code.
                //
        
                const __m128i q8s = _mm_loadu_si128((const __m128i *)y[i].bsums);
                const __m128i mins = _mm_cvtepi8_epi16(mins8);
                const __m128i prod = _mm_madd_epi16(mins, q8s);
                mins_acc = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), mins_acc);
            }
        
            const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                             _mm512_extractf32x8_ps(acc, 1));
        
            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));
        
            t0 = _mm_sub_ps(t0, mins_acc);
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    
            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

#else

//
// Drop-in replacement with identical signature. Two-phase approach:
//
// How it works: For each weight row + super-block:
//
// 1. Load & decode weight q4 data + 6-bit scales/mins once
// 2. dpbusd the same q4 vector against 4 different q8 activation columns
// 3. Apply shared scale vector to all 4 integer accumulators
// 4. Per-column: FMA with column-specific d values, accumulate mins
//
// Why faster for batch >= 4:
//
//  - 4x reduction in weight memory traffic (the bottleneck)
//  - Scale/min extraction amortized across 4 columns
//  - ~12 zmm registers used - no spill pressure
//

void
xx_vec_dot_q4_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q4_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t kmask4 = 0xc0c0c0c0u;

    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m128i zero128 = _mm_setzero_si128();

    uint64_t utmp[2];

    const block_q8_K_repack * y_base = vy;
    float * s_base = s;

    const uint64_t ncols4 = ncols & ~3ULL;
    uint64_t l = 0;

    //
    // Phase 1: Process 4 activation columns at a time.
    //
    // For each weight row, load weight super-blocks once and dot-product
    // against 4 activation columns simultaneously. This amortizes weight
    // memory traffic across 4 columns.
    //

    for (; l < ncols4; l += 4) {

        const block_q8_K_repack * y0 = y_base + (l + 0) * nb;
        const block_q8_K_repack * y1 = y_base + (l + 1) * nb;
        const block_q8_K_repack * y2 = y_base + (l + 2) * nb;
        const block_q8_K_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        const block_q4_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            __m128 mins_acc0 = _mm_setzero_ps();
            __m128 mins_acc1 = _mm_setzero_ps();
            __m128 mins_acc2 = _mm_setzero_ps();
            __m128 mins_acc3 = _mm_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                //
                // SHARED: extract weight d/dmin and decode 6-bit scales/mins.
                // Done once per super-block, reused across all 4 columns.
                //

                const float x_d = GGML_FP16_TO_FP32(x[i].d);
                const float x_dmin = GGML_FP16_TO_FP32(x[i].dmin);

                const uint32_t * vscales = (uint32_t *)x[i].scales;

                utmp[1] = (uint64_t)(((vscales[2] >> 4) & kmask2) | ((vscales[1] & kmask4) >> 2)) << 32;
                utmp[1] |= (uint64_t)(vscales[1] & kmask1);
                utmp[0] = (uint64_t)((vscales[2] & kmask2) | ((vscales[0] & kmask4) >> 2)) << 32;
                utmp[0] |= (uint64_t)(vscales[0] & kmask1);

                const __m128i scales8 = _mm_insert_epi64(zero128, utmp[0], 0);
                const __m128i mins8   = _mm_insert_epi64(zero128, utmp[1], 0);

                __m512i scale = _mm512_cvtepi8_epi32(scales8);
                scale = _mm512_inserti64x4(scale, _mm512_castsi512_si256(scale), 1);

                const uint8_t * q4 = x[i].qs;

                //
                // SHARED: Load weight q4 data once per sub-block iteration,
                // then dpbusd against 4 activation columns.
                //

                __m512i sumi0 = _mm512_setzero_si512();
                __m512i sumi1 = _mm512_setzero_si512();
                __m512i sumi2 = _mm512_setzero_si512();
                __m512i sumi3 = _mm512_setzero_si512();

                for (uint64_t j = 0; j < QK_K / 64; ++j) {

                    //
                    // Load weight nibbles once
                    //

                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j * 32)));

                    __m512i q4v = _mm512_castsi256_si512(q4bits);
                    q4v = _mm512_inserti32x8(q4v, _mm256_srli_epi16(q4bits, 4), 1);
                    q4v = _mm512_and_si512(q4v, m4);

                    //
                    // Dot product against 4 activation columns
                    //

                    sumi0 = _mm512_dpbusd_epi32(sumi0, q4v, _mm512_loadu_si512(y0[i].qs + (j * 64)));
                    sumi1 = _mm512_dpbusd_epi32(sumi1, q4v, _mm512_loadu_si512(y1[i].qs + (j * 64)));
                    sumi2 = _mm512_dpbusd_epi32(sumi2, q4v, _mm512_loadu_si512(y2[i].qs + (j * 64)));
                    sumi3 = _mm512_dpbusd_epi32(sumi3, q4v, _mm512_loadu_si512(y3[i].qs + (j * 64)));
                }

                //
                // SHARED: Apply scale vector to all 4 column accumulators.
                //

                sumi0 = _mm512_mullo_epi32(sumi0, scale);
                sumi1 = _mm512_mullo_epi32(sumi1, scale);
                sumi2 = _mm512_mullo_epi32(sumi2, scale);
                sumi3 = _mm512_mullo_epi32(sumi3, scale);

                //
                // Per-column: multiply by d-scale and FMA into float accumulators.
                //

                const float d0 = y0[i].d * x_d;
                const float d1 = y1[i].d * x_d;
                const float d2 = y2[i].d * x_d;
                const float d3 = y3[i].d * x_d;

                acc0 = _mm512_fmadd_ps(_mm512_set1_ps(d0), _mm512_cvtepi32_ps(sumi0), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_set1_ps(d1), _mm512_cvtepi32_ps(sumi1), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_set1_ps(d2), _mm512_cvtepi32_ps(sumi2), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_set1_ps(d3), _mm512_cvtepi32_ps(sumi3), acc3);

                //
                // SHARED mins extraction, per-column bsums.
                //

                const __m128i mins = _mm_cvtepi8_epi16(mins8);

                const float dmin0 = y0[i].d * x_dmin;
                const float dmin1 = y1[i].d * x_dmin;
                const float dmin2 = y2[i].d * x_dmin;
                const float dmin3 = y3[i].d * x_dmin;

                const __m128i prod0 = _mm_madd_epi16(mins, _mm_loadu_si128((const __m128i *)y0[i].bsums));
                const __m128i prod1 = _mm_madd_epi16(mins, _mm_loadu_si128((const __m128i *)y1[i].bsums));
                const __m128i prod2 = _mm_madd_epi16(mins, _mm_loadu_si128((const __m128i *)y2[i].bsums));
                const __m128i prod3 = _mm_madd_epi16(mins, _mm_loadu_si128((const __m128i *)y3[i].bsums));

                mins_acc0 = _mm_fmadd_ps(_mm_set1_ps(dmin0), _mm_cvtepi32_ps(prod0), mins_acc0);
                mins_acc1 = _mm_fmadd_ps(_mm_set1_ps(dmin1), _mm_cvtepi32_ps(prod1), mins_acc1);
                mins_acc2 = _mm_fmadd_ps(_mm_set1_ps(dmin2), _mm_cvtepi32_ps(prod2), mins_acc2);
                mins_acc3 = _mm_fmadd_ps(_mm_set1_ps(dmin3), _mm_cvtepi32_ps(prod3), mins_acc3);
            }

            //
            // Horizontal reduction: 512 -> 256 -> 128, subtract mins, hadd to scalar.
            //

#define REDUCE_CP(acc_v, mins_v, dest, idx) \
            { \
                const __m256 _res = _mm256_add_ps(_mm512_castps512_ps256(acc_v), \
                                                   _mm512_extractf32x8_ps(acc_v, 1)); \
                __m128 _t0 = _mm_add_ps(_mm256_castps256_ps128(_res), \
                                         _mm256_extractf128_ps(_res, 1)); \
                _t0 = _mm_sub_ps(_t0, mins_v); \
                const __m128 _t1 = _mm_hadd_ps(_t0, _t0); \
                dest[idx] = _mm_cvtss_f32(_mm_hadd_ps(_t1, _t1)); \
            }

            REDUCE_CP(acc0, mins_acc0, s0, k);
            REDUCE_CP(acc1, mins_acc1, s1, k);
            REDUCE_CP(acc2, mins_acc2, s2, k);
            REDUCE_CP(acc3, mins_acc3, s3, k);

#undef REDUCE_CP

            x += nb;
        }
    }

    //
    // Phase 2: Process remaining columns one at a time (original path).
    //

    const block_q8_K_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        const block_q4_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();
            __m128 mins_acc = _mm_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                const float d = y_rem[i].d * GGML_FP16_TO_FP32(x[i].d);
                const float dmin = y_rem[i].d * GGML_FP16_TO_FP32(x[i].dmin);

                const uint32_t * vscales = (uint32_t *)x[i].scales;
                utmp[1] = (uint64_t)(((vscales[2] >> 4) & kmask2) | ((vscales[1] & kmask4) >> 2)) << 32;
                utmp[1] |= (uint64_t)(vscales[1] & kmask1);
                utmp[0] = (uint64_t)((vscales[2] & kmask2) | ((vscales[0] & kmask4) >> 2)) << 32;
                utmp[0] |= (uint64_t)(vscales[0] & kmask1);

                const uint8_t * q4 = x[i].qs;
                const int8_t  * q8 = y_rem[i].qs;

                const __m128i scales8 = _mm_insert_epi64(zero128, utmp[0], 0);
                const __m128i mins8 = _mm_insert_epi64(zero128, utmp[1], 0);

                __m512i scale = _mm512_cvtepi8_epi32(scales8);
                scale = _mm512_inserti64x4(scale, _mm512_castsi512_si256(scale), 1);

                __m512i sumi = _mm512_setzero_si512();

                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j * 32)));
                    const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));

                    __m512i q4v = _mm512_castsi256_si512(q4bits);
                    q4v = _mm512_inserti32x8(q4v, _mm256_srli_epi16(q4bits, 4), 1);
                    q4v = _mm512_and_si512(q4v, m4);

                    sumi = _mm512_dpbusd_epi32(sumi, q4v, q8v);
                }

                sumi = _mm512_mullo_epi32(sumi, scale);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);

                const __m128i q8s = _mm_loadu_si128((const __m128i *)y_rem[i].bsums);
                const __m128i mins = _mm_cvtepi8_epi16(mins8);
                const __m128i prod = _mm_madd_epi16(mins, q8s);
                mins_acc = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), mins_acc);
            }

            const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                             _mm512_extractf32x8_ps(acc, 1));

            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));

            t0 = _mm_sub_ps(t0, mins_acc);

            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s_rem[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s_rem = (float *)((char *)s_rem + nr_nb1);
        y_rem += nb;
    }
}

#endif // ORIGINAL_VERSION

#if ORIGINAL_VERSION

void
xx_vec_dot_q6_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q6_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m512i m2 = _mm512_set1_epi8(0x30);
    const __m512i m32s = _mm512_set1_epi8(32);

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_K_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        const block_q6_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        
                const uint8_t * q4 = x[i].ql;
                const uint8_t * q2 = x[i].qh;
                const int8_t * q8 = y[i].qs;
        
                const __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
                const __m512i scales = _mm512_cvtepi8_epi32(scales8);
        
                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();

                //
                // Load the high 2-bits and rotate the bits left by 4 - 256 2-bit values.
                //
                // N.B. This lines up the first set of 2-bits in the proper position to
                //      merge with the first set of 4-bits.
                //

                __m512i q2bits = _mm512_loadu_si512((const __m512i*)q2);
                q2bits = _mm512_rol_epi32(q2bits, 4);

                //
                // Unpack the 6-bit quant values into unsigned 8-bit quant values
                // and perform dot product.
                //

                for (uint64_t j = 0; j < QK_K / 128; ++j) {

                    //
                    // Load 64 nibble packed 4-bit quant values - 128 4-bit values.
                    //

                    __m512i q4bits = _mm512_loadu_si512((const __m512i*)(q4 + (j * 64)));

                    //
                    // Merge the low 4-bits and current high 2-bits of the 6-bit quant
                    // values into unsigned 6-bit quant values.
                    //
        
                    const __m512i q4bits1 = _mm512_and_si512(q4bits, m4);
                    const __m512i q2bits1 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_0 = _mm512_or_si512(q2bits1, q4bits1);

                    //
                    // Shift next set of high 2-bits into place.
                    //

                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    //
                    // Load 8-bit quant values - 64 8-bit values.
                    //

                    const __m512i q8_0 = _mm512_loadu_si512(q8 + (j * 128) + 0);

                    //
                    // Multiply the unsigned q6 quant values by the signed q8 quant
                    // values and accumulate the results.
                    // 

                    sumi = _mm512_dpbusd_epi32(sumi, q6_0, q8_0);

                    //
                    // Multiply the unsigned bias values by the signed q8 quant
                    // values, and accumulate the results.
                    //

                    bias = _mm512_dpbusd_epi32(bias, m32s, q8_0);

                    //
                    // Shift the next nibble into place.
                    //

                    __m512i q4bits2 = _mm512_srli_epi16(q4bits, 4);

                    //
                    // Merge the high 4-bits and current high 2-bits of the 6-bit
                    // quant value into unsigned 6-bit quant values.
                    //

                    q4bits2 = _mm512_and_si512(q4bits2, m4);
                    const __m512i q2bits2 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_1 = _mm512_or_si512(q2bits2, q4bits2);

                    //
                    // Shift next set of high 2-bits into place.
                    //

                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    //
                    // Load 8-bit quant values - 64 8-bit values.
                    //

                    const __m512i q8_1 = _mm512_loadu_si512(q8 + (j * 128) + 64);

                    //
                    // Multiply the unsigned q6 quant values by the signed q8 quant
                    // values and accumulate the results.
                    // 

                    sumi = _mm512_dpbusd_epi32(sumi, q6_1, q8_1);

                    //
                    // Multiply the unsigned bias values by the signed q8 quant
                    // values and accumulate the results.
                    //

                    bias = _mm512_dpbusd_epi32(bias, m32s, q8_1);
                }

                //
                // Subtract the bias values from the sum, multiply the accumulated
                // sum by the q6 scale, convert to float, multiply by the block
                // multiplier, and accumulate the results.
                //

                sumi = _mm512_sub_epi32(sumi, bias);
                sumi = _mm512_mullo_epi32(sumi, scales);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
            }

            __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                       _mm512_extractf32x8_ps(acc, 1));

            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));

            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

#else

void
xx_vec_dot_q6_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q6_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m512i m2 = _mm512_set1_epi8(0x30);
    const __m512i m32s = _mm512_set1_epi8(32);

    const block_q8_K_repack * y_base = vy;
    float * s_base = s;

    const uint64_t ncols4 = ncols & ~3ULL;
    uint64_t l = 0;

    //
    // Phase 1: Process 4 activation columns at a time.
    //

    for (; l < ncols4; l += 4) {

        const block_q8_K_repack * y0 = y_base + (l + 0) * nb;
        const block_q8_K_repack * y1 = y_base + (l + 1) * nb;
        const block_q8_K_repack * y2 = y_base + (l + 2) * nb;
        const block_q8_K_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        //
        // Iterate through the specified number of rows.
        //

        const block_q6_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                //
                // Per-column integer accumulators.
                //

                __m512i sumi0 = _mm512_setzero_si512();
                __m512i sumi1 = _mm512_setzero_si512();
                __m512i sumi2 = _mm512_setzero_si512();
                __m512i sumi3 = _mm512_setzero_si512();

                __m512i bias0 = _mm512_setzero_si512();
                __m512i bias1 = _mm512_setzero_si512();
                __m512i bias2 = _mm512_setzero_si512();
                __m512i bias3 = _mm512_setzero_si512();

                //
                // Load the high 2-bits and rotate the bits left by 6 - 256 2-bit values.
                //
                // N.B. This lines up the first set of high 2-bits in the proper position.
                //

                __m512i q2bits = _mm512_loadu_si512((const __m512i*)&x[i].qh[0]);
                q2bits = _mm512_rol_epi32(q2bits, 6);

                //
                // Unpack the 6-bit quant values into unsigned 8-bit quant values
                // and perform dot product.
                //

                for (uint64_t j = 0; j < QK_K / 128; j += 1) {

                    //
                    // Load 64 nibble packed 4-bit quant values - 128 4-bit values
                    // and shift the next high 2-bits into place.
                    //

                    __m512i q4bits = _mm512_loadu_si512((const __m512i*)(&x[i].ql[j * 64]));
                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    //
                    // Merge the low 4-bits and high 2-bits of the 6-bit quant values
                    // into unsigned 6-bit quant value.
                    //
        
                    const __m512i q4bits1 = _mm512_and_si512(q4bits, m4);
                    const __m512i q2bits1 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_0 = _mm512_or_si512(q2bits1, q4bits1);

                    //
                    // Shift the next nibble into place and shift next high 2-bits into
                    // place.
                    //

                    q4bits = _mm512_srli_epi16(q4bits, 4);
                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    //
                    // Merge the high 4-bits and high 2-bits of the 6-bit quant value
                    // into unsigned 6-bit quant values.
                    //

                    const __m512i q4bits2 = _mm512_and_si512(q4bits, m4);
                    const __m512i q2bits2 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_1 = _mm512_or_si512(q2bits2, q4bits2);

                    //
                    // Per-column: load q8-x and dpbusd against shared q6_0.
                    //

                    const __m512i q8_0 = _mm512_loadu_si512((const __m512i *)&y0[i].qs[j * 128 + 0]);
                    sumi0 = _mm512_dpbusd_epi32(sumi0, q6_0, q8_0);
                    bias0 = _mm512_dpbusd_epi32(bias0, m32s, q8_0);

                    const __m512i q8_1 = _mm512_loadu_si512((const __m512i *)&y1[i].qs[j * 128 + 0]);
                    sumi1 = _mm512_dpbusd_epi32(sumi1, q6_0, q8_1);
                    bias1 = _mm512_dpbusd_epi32(bias1, m32s, q8_1);

                    const __m512i q8_2 = _mm512_loadu_si512((const __m512i *)&y2[i].qs[j * 128 + 0]);
                    sumi2 = _mm512_dpbusd_epi32(sumi2, q6_0, q8_2);
                    bias2 = _mm512_dpbusd_epi32(bias2, m32s, q8_2);

                    const __m512i q8_3 = _mm512_loadu_si512((const __m512i *)&y3[i].qs[j * 128 + 0]);
                    sumi3 = _mm512_dpbusd_epi32(sumi3, q6_0, q8_3);
                    bias3 = _mm512_dpbusd_epi32(bias3, m32s, q8_3);

                    //
                    // Per-column: load q8-x and dpbusd against shared q6_1.
                    //

                    const __m512i q8_4 = _mm512_loadu_si512((const __m512i *)&y0[i].qs[j * 128 + 64]);
                    sumi0 = _mm512_dpbusd_epi32(sumi0, q6_1, q8_4);
                    bias0 = _mm512_dpbusd_epi32(bias0, m32s, q8_4);

                    const __m512i q8_5 = _mm512_loadu_si512((const __m512i *)&y1[i].qs[j * 128 + 64]);
                    sumi1 = _mm512_dpbusd_epi32(sumi1, q6_1, q8_5);
                    bias1 = _mm512_dpbusd_epi32(bias1, m32s, q8_5);

                    const __m512i q8_6 = _mm512_loadu_si512((const __m512i *)&y2[i].qs[j * 128 + 64]);
                    sumi2 = _mm512_dpbusd_epi32(sumi2, q6_1, q8_6);
                    bias2 = _mm512_dpbusd_epi32(bias2, m32s, q8_6);

                    const __m512i q8_7 = _mm512_loadu_si512((const __m512i *)&y3[i].qs[j * 128 + 64]);
                    sumi3 = _mm512_dpbusd_epi32(sumi3, q6_1, q8_7);
                    bias3 = _mm512_dpbusd_epi32(bias3, m32s, q8_7);
                }

                //
                // Shared: load weight d[8] (fp16 -> fp32) and scales.
                //

                const float xd = GGML_FP16_TO_FP32(x[i].d);

                const __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
                const __m512i scales = _mm512_cvtepi8_epi32(scales8);

#define Q6K_COL_FMA(col, y_ptr, sumi_v, bias_v, acc_v) \
                do { \
                    const float d_##col = xd * y_ptr[i].d; \
                    sumi_v = _mm512_sub_epi32(sumi_v, bias_v); \
                    sumi_v = _mm512_mullo_epi32(sumi_v, scales); \
                    acc_v = _mm512_fmadd_ps(_mm512_set1_ps(d_##col), \
                                            _mm512_cvtepi32_ps(sumi_v), \
                                            acc_v); \
                } while (0)

                Q6K_COL_FMA(0, y0, sumi0, bias0, acc0);
                Q6K_COL_FMA(1, y1, sumi1, bias1, acc1);
                Q6K_COL_FMA(2, y2, sumi2, bias2, acc2);
                Q6K_COL_FMA(3, y3, sumi3, bias3, acc3);

#undef Q6K_COL_FMA

            }

            //
            // Reduce and store results for each of the 4 columns.
            //

#define REDUCE_Q6K_CP(acc, dst) \
            do { \
                const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc), \
                                                 _mm512_extractf32x8_ps(acc, 1)); \
                const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res), \
                                             _mm256_extractf128_ps(res, 1)); \
                const __m128 t1 = _mm_hadd_ps(t0, t0); \
                dst[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1)); \
            } while (0)

            REDUCE_Q6K_CP(acc0, s0);
            REDUCE_Q6K_CP(acc1, s1);
            REDUCE_Q6K_CP(acc2, s2);
            REDUCE_Q6K_CP(acc3, s3);

#undef REDUCE_Q6K_CP

            x += nb;
        }
    }

    //
    // Phase 2: Process remaining columns one at a time (0-3 columns).
    //

    const block_q8_K_repack * y_rem = y_base + l * nb;;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        const block_q6_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y_rem[i].d * GGML_FP16_TO_FP32(x[i].d);
        
                const __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
                const __m512i scales = _mm512_cvtepi8_epi32(scales8);
        
                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();

                //
                // Load the high 2-bits and rotate the bits left by 6 - 256 2-bit values.
                //
                // N.B. This lines up the first set of 2-bits in the proper position.
                //

                __m512i q2bits = _mm512_loadu_si512((const __m512i*)&x[i].qh[0]);
                q2bits = _mm512_rol_epi32(q2bits, 6);

                //
                // Unpack the 6-bit quant values into unsigned 8-bit quant values
                // and perform dot product.
                //

                for (uint64_t j = 0; j < QK_K / 128; j += 1) {

                    //
                    // Load 64 nibble packed 4-bit quant values - 128 4-bit values
                    // and shift the next high 2-bits into place.
                    //

                    __m512i q4bits = _mm512_loadu_si512((const __m512i*)(&x[i].ql[j * 64]));
                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    //
                    // Merge the low 4-bits and high 2-bits of the 6-bit quant values
                    // into unsigned 6-bit quant values.
                    //
        
                    const __m512i q4bits1 = _mm512_and_si512(q4bits, m4);
                    const __m512i q2bits1 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_0 = _mm512_or_si512(q2bits1, q4bits1);

                    //
                    // Load 8-bit quant values - 64 8-bit values and dpbusd against q6_0.
                    //

                    const __m512i q8_0 = _mm512_loadu_si512(&y_rem[i].qs[j * 128 + 0]);
                    sumi = _mm512_dpbusd_epi32(sumi, q6_0, q8_0);
                    bias = _mm512_dpbusd_epi32(bias, m32s, q8_0);

                    //
                    // Shift the next nibble into place and shift next high 2-bits into
                    // place.
                    //

                    q4bits = _mm512_srli_epi16(q4bits, 4);
                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    //
                    // Merge the high 4-bits and high 2-bits of the 6-bit quant value
                    // into unsigned 6-bit quant values.
                    //

                    const __m512i q4bits2 = _mm512_and_si512(q4bits, m4);
                    const __m512i q2bits2 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_1 = _mm512_or_si512(q2bits2, q4bits2);

                    //
                    // Load 8-bit quant values - 64 8-bit values and dpbusd against q6_1.
                    //

                    const __m512i q8_1 = _mm512_loadu_si512(&y_rem[i].qs[j * 128 + 64]);
                    sumi = _mm512_dpbusd_epi32(sumi, q6_1, q8_1);
                    bias = _mm512_dpbusd_epi32(bias, m32s, q8_1);
                }

                //
                // Subtract the bias values from the sum, multiply the accumulated
                // sum by the q6 scale, convert to float, multiply by the block
                // multiplier, and accumulate the results.
                //

                sumi = _mm512_sub_epi32(sumi, bias);
                sumi = _mm512_mullo_epi32(sumi, scales);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
            }

            __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                       _mm512_extractf32x8_ps(acc, 1));

            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));

            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s_rem[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s = (float *)((char *)s_rem + nr_nb1);
        y_rem += nb;
    }
}

#endif // ORIGINAL_VERSION

#if ORIGINAL_VERSION

void
xx_vec_dot_q8_0_q8_0_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q8_0_repack * vx,
    size_t bx,
    const block_q8_0_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i zero512 = _mm512_setzero_si512();

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_0_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate throught the specified number of rows.
        //

        const block_q8_0_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {
        
                __m512i sumi = _mm512_setzero_si512();
        
                //
                // Compute combined scale for the an entire quant block.
                //
        
                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m128h yd = _mm_loadu_ph(y[i].d);
                const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                                   _mm256_cvtph_ps(yd));
        
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);
        
                //
                // Compute the dot product and accumulate.
                //
        
                for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {
                    const __m512i qx = _mm512_loadu_si512((const __m512i *)&x[i].qs[j * QK8_0 * 2 + 0]);
                    const __m512i qy = _mm512_loadu_si512((const __m512i *)&y[i].qs[j * QK8_0 * 2 + 0]);
    
                    //
                    // Compute the absolute value of qx and generate a mask of the corresponding
                    // values of qx that are negative.
                    //
    
                    const __m512i ax = _mm512_abs_epi8(qx);
                    const __mmask64 is_negative_qx = _mm512_movepi8_mask(qx);
    
                    //
                    // Compute the signed value of qy taking into account the negative values
                    // in the corresponding byte of qx.
                    //
    
                    const __m512i sy = _mm512_mask_sub_epi8(qy, is_negative_qx, zero512, qy);
    
                    //
                    // mul (ax * sy) + sumi directly to epi32
                    //
                    // N.B. __AVX512VNNI__ and __AVX512VL__ are always defined.
                    //
            
                    sumi = _mm512_dpbusd_epi32(sumi, ax, sy);
                }
        
                //
                // Multiply q with scale and accumulate.
                //
        
                const __m512 q = _mm512_cvtepi32_ps(sumi);
                acc = _mm512_fmadd_ps(d, q, acc);
            }
        
            const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                             _mm512_extractf32x8_ps(acc, 1));
        
            const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                         _mm256_extractf128_ps(res, 1));
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    
            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

#else

void
xx_vec_dot_q8_0_q8_0_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q8_0_repack * vx,
    size_t bx,
    const block_q8_0_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i zero512 = _mm512_setzero_si512();

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_0_repack * y_base = vy;
    float * s_base = s;

    const uint64_t ncols4 = ncols & ~3ULL;
    uint64_t l = 0;

    //
    // Phase 1: Process 4 activation columns at a time.
    //

    for (; l < ncols4; l += 4) {

        const block_q8_0_repack * y0 = y_base + (l + 0) * nb;
        const block_q8_0_repack * y1 = y_base + (l + 1) * nb;
        const block_q8_0_repack * y2 = y_base + (l + 2) * nb;
        const block_q8_0_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        const block_q8_0_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; i += 1) {

                //
                // Per-column integer accumulators.
                //

                __m512i sumi0 = _mm512_setzero_si512();
                __m512i sumi1 = _mm512_setzero_si512();
                __m512i sumi2 = _mm512_setzero_si512();
                __m512i sumi3 = _mm512_setzero_si512();

                //
                // Compute the dot product and accumulate.
                //
        
                for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {
    
                    //
                    // Compute the absolute value of the row and generate a mask of the
                    // corresponding values of qx that are negative.
                    //
    
                    const __m512i qx = _mm512_loadu_si512((const __m512i *)&x[i].qs[j * QK8_0 * 2 + 0]);
                    const __m512i ax = _mm512_abs_epi8(qx);
                    const __mmask64 is_negative_qx = _mm512_movepi8_mask(qx);
    
                    //
                    // Compute the signed value of qy taking into account the negative values
                    // in the corresponding byte of qx.
                    //
    
                    const __m512i qy0 = _mm512_loadu_si512((const __m512i *)&y0[i].qs[j * QK8_0 * 2 + 0]);
                    const __m512i sy0 = _mm512_mask_sub_epi8(qy0, is_negative_qx, zero512, qy0);
    
                    const __m512i qy1 = _mm512_loadu_si512((const __m512i *)&y1[i].qs[j * QK8_0 * 2 + 0]);
                    const __m512i sy1 = _mm512_mask_sub_epi8(qy1, is_negative_qx, zero512, qy1);

                    const __m512i qy2 = _mm512_loadu_si512((const __m512i *)&y2[i].qs[j * QK8_0 * 2 + 0]);
                    const __m512i sy2 = _mm512_mask_sub_epi8(qy2, is_negative_qx, zero512, qy2);

                    const __m512i qy3 = _mm512_loadu_si512((const __m512i *)&y3[i].qs[j * QK8_0 * 2 + 0]);
                    const __m512i sy3 = _mm512_mask_sub_epi8(qy3, is_negative_qx, zero512, qy3);

                    //
                    // mul (ax * sy) + sumi directly to epi32
                    //
                    // N.B. __AVX512VNNI__ and __AVX512VL__ are always defined.
                    //
            
                    sumi0 = _mm512_dpbusd_epi32(sumi0, ax, sy0);
                    sumi1 = _mm512_dpbusd_epi32(sumi1, ax, sy1);
                    sumi2 = _mm512_dpbusd_epi32(sumi2, ax, sy2);
                    sumi3 = _mm512_dpbusd_epi32(sumi3, ax, sy3);
                }

                //
                // Load row scale and convert to fp32.
                //

                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m256 x_fp32 = _mm256_cvtph_ps(xd);

                //
                // Per-column: build per-column scale vector, FMA.
                //

#define Q80_COL_FMA(col, y_ptr, sumi_v, acc_v) \
                do { \
                    const __m128h yd_##col = _mm_loadu_ph((y_ptr)[i].d); \
                    const __m256 scale_##col = _mm256_mul_ps(x_fp32, _mm256_cvtph_ps(yd_##col)); \
                    __m512 d_##col = _mm512_castps256_ps512(scale_##col); \
                    d_##col = _mm512_insertf32x8(d_##col, scale_##col, 1); \
                    acc_v = _mm512_fmadd_ps(d_##col, _mm512_cvtepi32_ps(sumi_v), acc_v); \
                } while (0)

                Q80_COL_FMA(0, y0, sumi0, acc0);
                Q80_COL_FMA(1, y1, sumi1, acc1);
                Q80_COL_FMA(2, y2, sumi2, acc2);
                Q80_COL_FMA(3, y3, sumi3, acc3);

#undef Q80_COL_FMA

            }

            //
            // Reduce and store results for each of the 4 columns.
            //

#define REDUCE_Q80_CP(acc, dst) \
            do { \
                const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc), \
                                                 _mm512_extractf32x8_ps(acc, 1)); \
                const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res), \
                                             _mm256_extractf128_ps(res, 1)); \
                const __m128 t1 = _mm_hadd_ps(t0, t0); \
                dst[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1)); \
            } while (0)

            REDUCE_Q80_CP(acc0, s0);
            REDUCE_Q80_CP(acc1, s1);
            REDUCE_Q80_CP(acc2, s2);
            REDUCE_Q80_CP(acc3, s3);

#undef REDUCE_Q80_CP

            x += nb;
        }
    }

    //
    // Phase 2: Process remaining columns one at a time (0-3 columns).
    //

    const block_q8_0_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        //
        // Iterate throught the remaining rows.
        //

        const block_q8_0_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {
        
                __m512i sumi = _mm512_setzero_si512();
        
                //
                // Compute combined scale for the an entire quant block.
                //
        
                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m128h yd = _mm_loadu_ph(y_rem[i].d);
                const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                                   _mm256_cvtph_ps(yd));
        
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);
        
                //
                // Compute the dot product and accumulate.
                //
        
                for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {
                    const __m512i qx = _mm512_loadu_si512((const __m512i *)&x[i].qs[j * QK8_0 * 2 + 0]);
                    const __m512i qy = _mm512_loadu_si512((const __m512i *)&y_rem[i].qs[j * QK8_0 * 2 + 0]);
    
                    //
                    // Compute the absolute value of qx and generate a mask of the corresponding
                    // values of qx that are negative.
                    //
    
                    const __m512i ax = _mm512_abs_epi8(qx);
                    const __mmask64 is_negative_qx = _mm512_movepi8_mask(qx);
    
                    //
                    // Compute the signed value of qy taking into account the negative values
                    // in the corresponding byte of qx.
                    //
    
                    const __m512i sy = _mm512_mask_sub_epi8(qy, is_negative_qx, zero512, qy);
    
                    //
                    // mul (ax * sy) + sumi directly to epi32
                    //
                    // N.B. __AVX512VNNI__ and __AVX512VL__ are always defined.
                    //
            
                    sumi = _mm512_dpbusd_epi32(sumi, ax, sy);
                }
        
                //
                // Multiply q with scale and accumulate.
                //
        
                acc = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(sumi), acc);
            }
        
            const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                             _mm512_extractf32x8_ps(acc, 1));
        
            const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                         _mm256_extractf128_ps(res, 1));
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s_rem[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    
            x += nb;
        }

        s_rem = (float *)((char *)s_rem + nr_nb1);
        y_rem += nb;
    }
}

#endif // ORIGINAL_VERSION

void
quantize_row_q4_0_x8 (
    const float * x,
    block_q4_0 * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q4_0 quants.
    //

    quantize_row_q4_0(x, y, vec_size);

    //
    // Make q4_0_repack quant blocks
    //

    make_q4_0_repack_quant(vec_size, (block_q4_0_repack *)y, y);
}

void                   
quantize_row_q2_k_x8 (
    const float * x,
    block_q2_K * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q2_K quants.
    //

    quantize_row_q2_K(x, y, vec_size);


    //
    // Make q2_k_repack quant blocks.
    //

    make_q2_k_repack_quant(vec_size, (block_q2_K_repack *)y, y);
}

void                   
quantize_row_q3_k_x8 (
    const float * x,
    block_q3_K * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q3_K quants.
    //

    quantize_row_q3_K(x, y, vec_size);

    //
    // Make q3_k_repack quant blocks.
    //

    make_q3_k_repack_quant(vec_size, (block_q3_K_repack *)y, y);
}

void                   
quantize_row_q4_k_x8 (
    const float * x,
    block_q4_K * y,
    uint64_t vec_size
    )
{

    //
    // Quantize the x vector into q4_K quants.
    //

    quantize_row_q4_K(x, y, vec_size);

    //
    // Make q4_k_repack quant blocks.
    //

    make_q4_k_repack_quant(vec_size, (block_q4_K_repack *)y, y);
}

void                   
quantize_row_q6_k_x8 (
    const float * x,
    block_q6_K * y,
    uint64_t vec_size
    )
{

    //
    // Quantize the x vector into q6_K quants.
    //

    quantize_row_q6_K(x, y, vec_size);

    //
    // Make q6_k_repack quant blocks.
    //

    make_q6_k_repack_quant(vec_size, (block_q6_K_repack *)y, y);
}

void                   
quantize_row_q236_k_q8_k_x8 (
    const float * x,
    block_q8_K * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q8_K quants.
    //

    quantize_row_q8_K(x, y, vec_size);

    //
    // Make q8_k_repack quant blocks
    //

    make_q236_k_q8_k_repack_quant(vec_size, (block_q8_K_repack *)y, y);
}

void                   
quantize_row_q4_k_q8_k_x8 (
    const float * x,
    block_q8_K * y,
    uint64_t vec_size
    )
{

    //
    // Quantize the x vector into q8_K quants.
    //

    quantize_row_q8_K(x, y, vec_size);

    //
    // Make q8_k_repack quant blocks.
    //

    make_q4_k_q8_k_repack_quant(vec_size, (block_q8_K_repack *)y, y);
}

void
quantize_row_q8_0_x8 (
    const float * x,
    block_q8_0 * y,
    uint64_t vec_size
    )
{

    //
    // Quantize the x vector into q8_0 guants.
    //

    quantize_row_q8_0(x, y, vec_size);

    //
    // Make q8_0_repack quant blocks.
    //

    make_q8_0_repack_quant(vec_size, (block_q8_0_repack *)y, y);
}

void
ggml_repack_tensor (
    const struct ggml_compute_params * params,
    struct ggml_tensor *tensor
    ) 
{
    enum ggml_type type = tensor->type;

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_ASSERT((type == GGML_TYPE_Q4_0) ||
                (type == GGML_TYPE_Q2_K) ||
                (type == GGML_TYPE_Q3_K) ||
                (type == GGML_TYPE_Q4_K) ||
                (type == GGML_TYPE_Q6_K) ||
                (type == GGML_TYPE_Q8_0));

    switch (tensor_repacking_mode) {

        //
        // Repack GGML mode.
        //

    case TENSOR_REPACKING_MODE_GGML:

        //
        // N.B. Repacking is single threaded on the zeroth cpu for ggml.
        //

        enum ggml_type repack_type = type;
        if (type == GGML_TYPE_Q4_0) {
            repack_type = GGML_TYPE_Q4_0_8_8;

        } else if (type == GGML_TYPE_Q4_K) {
            repack_type = GGML_TYPE_Q4_K_8_8;

        } else {
            break;
        }

        if (!ith) {
            size_t data_size = ggml_nbytes(tensor);
            void *src_data = tensor->data;

            if (ggml_aarch64_repack_tensor(tensor, repack_type, src_data, data_size)) {

                // printf("*** converted tensor %s - type %s - size %zd succeeded\n",
                //        ggml_get_name(tensor),
                //        ggml_type_name(type),
                //        data_size);

                type = repack_type;
            }

            //
            //
            // Wait for all other threads to arrive at the barrier below before
            // potentially changing the tensor type.
            //
            // N.B. The tensor type cannot be changed until it is guaranteed that
            //      all other threads are waiting on the barrier below.
            //

            ggml_wait_to_finalize(params);
            tensor->type = type;
        }

        ggml_wait_for_done(params);
        break;

        //
        // Repack Xbox mode.
        //

    case TENSOR_REPACKING_MODE_XBOX:
    case TENSOR_REPACKING_MODE_XBCG:

        //
        // Check if the tensor is contiguous and the number of elements is 0 mod QK_K.
        //

        uint64_t ne = tensor->ne[0];
        if (!ggml_is_contiguous(tensor) || ((ne % QK_K) != 0)) {
            break;
        }

        //
        // Make transformed quant based on current type.
        //
        // N.B. The original data and the repacked data share the same memory. They are
        //      exactly the same size and layout. The make repack quant function does
        //      the repack such that no extra memory needs to be allocated and there are
        //      no extra copies.
        //
        // N.B. Repacking is multithreaded for xbox.
        //

        int64_t i;
        char * src_data = tensor->data;
        int64_t nrows = tensor->ne[1];
        int64_t stride = tensor->nb[1];
        const int64_t rows_per_thread = (nrows + nth - 1) / nth;
        const int64_t start_row = rows_per_thread * ith;
        const int64_t end_row = MIN(start_row + rows_per_thread, nrows);
        src_data += start_row * stride;

        if (type == GGML_TYPE_Q4_0) {
            type = GGML_TYPE_Q4_0_x8;

            for (i = start_row; i < end_row; i += 1) {
                make_q4_0_repack_quant(ne,
                                       (block_q4_0_repack *)src_data,
                                       (block_q4_0 *)src_data);

                src_data += stride;
            }

        } else if (type == GGML_TYPE_Q2_K) {
            type = GGML_TYPE_Q2_K_x8;

            for (i = start_row; i < end_row; i += 1) {
                make_q2_k_repack_quant(ne,
                                       (block_q2_K_repack *)src_data,
                                       (block_q2_K *)src_data);

                src_data += stride;
            }

        } else if (type == GGML_TYPE_Q3_K) {
            type = GGML_TYPE_Q3_K_x8;

            for (i = start_row; i < end_row; i += 1) {
                make_q3_k_repack_quant(ne,
                                       (block_q3_K_repack *)src_data,
                                       (block_q3_K *)src_data);

                src_data += stride;
            }

        } else if (type == GGML_TYPE_Q4_K) {
            type = GGML_TYPE_Q4_K_x8;

            for (i = start_row; i < end_row; i += 1) {
                make_q4_k_repack_quant(ne,
                                       (block_q4_K_repack *)src_data,
                                       (block_q4_K *)src_data);

                src_data += stride;
            }

        } else if (type == GGML_TYPE_Q6_K) {
            type = GGML_TYPE_Q6_K_x8;

            for (i = start_row; i < end_row; i += 1) {
                make_q6_k_repack_quant(ne,
                                       (block_q6_K_repack *)src_data,
                                       (block_q6_K *)src_data);

                src_data += stride;
            }

        } else if (type == GGML_TYPE_Q8_0) {
            type = GGML_TYPE_Q8_0_x8;

            for (i = start_row; i < end_row; i += 1) {
                make_q8_0_repack_quant(ne,
                                       (block_q8_0_repack *)src_data,
                                       (block_q8_0 *)src_data);

                src_data += stride;
            }
        }

        ggml_wait_for_done(params);

        //
        // N.B. All threads write the same value to tensor type so no special
        //      synchronization is required.
        //

        tensor->type = type;
        break;

    case TENSOR_REPACKING_MODE_NONE:
    default:
        break;
    }

    return;
}
