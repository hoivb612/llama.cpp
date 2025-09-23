#define GGML_COMMON_IMPL_C

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "ggml-cpu-repack.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>

//
// Repacking Xbox style
//

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
#pragma comment(linker, "/EXPORT:make_q4_0_repack_quant=" __FUNCTION__)

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
                uint32_t * qs_dst = (uint32_t *)&qs_out.qs[offset + 0];
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
#pragma comment(linker, "/EXPORT:make_q2_k_repack_quant=" __FUNCTION__)

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
#pragma comment(linker, "/EXPORT:make_q3_k_repack_quant=" __FUNCTION__)

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
#pragma comment(linker, "/EXPORT:make_q4_k_repack_quant=" __FUNCTION__)

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
#pragma comment(linker, "/EXPORT:make_q6_k_repack_quant=" __FUNCTION__)

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
#pragma comment(linker, "/EXPORT:make_q8_0_repack_quant=" __FUNCTION__)

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
#pragma comment(linker, "/EXPORT:make_q236_k_q8_k_repack_quant=" __FUNCTION__)

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
#pragma comment(linker, "/EXPORT:make_q4_k_q8_k_repack_quant=" __FUNCTION__)

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

void
xx_vec_dot_q4_0_q8_0_x8 (
    const int n,
    float * s,
    size_t nr_nb1,
    const block_q4_0_repack * vx,
    size_t bx,
    const block_q8_0_repack * vy,
    size_t ncols,
    int nrc
    )
{
    #pragma comment(linker, "/EXPORT:xx_vec_dot_q4_0_q8_0_x8=" __FUNCTION__)

    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const __m512i offset = _mm512_set1_epi8(8);
    const __m512i m4 = _mm512_set1_epi8(0xf);

    //
    // Iterate through the specified number of columns.
    //

    block_q8_0_repack * y = (block_q8_0_repack *)vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate throught the specified number of rows
        //

        block_q4_0_repack * x = (block_q4_0_repack *)vx;

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

void
xx_vec_dot_q2_k_q8_k_x8 (
    const int n,
    float * s,
    size_t nr_nb1,
    const block_q2_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    int nrc
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const __m512i m3 = _mm512_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    //
    // Iterate through the specified number of columns.
    //

    block_q8_K_repack * y = (block_q8_K_repack *)vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        block_q2_K_repack * x = (block_q2_K_repack *)vx;

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

void
xx_vec_dot_q3_k_q8_k_x8 (
    const int n,
    float * s,
    size_t nr_nb1,
    const block_q3_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    int nrc
    )
{
    GGML_UNUSED(bx);

    const uint64_t kmask1 = 0x0000000030303030ull;
    const uint64_t kmask2 = 0x0f0f0f0f0f0f0f0full;

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i m4 = _mm512_set1_epi8(4);
    const __m128i m32 = _mm_set1_epi8(32);

    const __m128i zero128 = _mm_setzero_si128();
    const __m512i zero512 = _mm512_setzero_si512();

    //
    // Iterate through the specified number of columns.
    //

    block_q8_K_repack * y = (block_q8_K_repack *)vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        block_q3_K_repack * x = (block_q3_K_repack *)vx;

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

#if 0
                uint32_t * aux = (uint32_t *)x[i].scales;
                __m128i scales8 = _mm_set_epi32(
                        ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                        ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                        (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                        (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
#endif // #if 0
        
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

void
xx_vec_dot_q4_k_q8_k_x8 (
    const int n,
    float * s,
    size_t nr_nb1,
    const block_q4_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    int nrc
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t kmask4 = 0xc0c0c0c0u;

    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m128i zero128 = _mm_setzero_si128();

    uint64_t utmp[2];

    //
    // Iterate through the specified number of columns.
    //

    block_q8_K_repack * y = (block_q8_K_repack *)vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate throught the specified number of rows
        //

        block_q4_K_repack * x = (block_q4_K_repack *)vx;

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
#if 1 // __512_AVXVNNI__ (best)

                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j * 32)));
                    const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));

                    __m512i q4v = _mm512_castsi256_si512(q4bits);
                    q4v = _mm512_inserti32x8(q4v, _mm256_srli_epi16(q4bits, 4), 1);
                    q4v = _mm512_and_si512(q4v, m4);

                    sumi = _mm512_dpbusd_epi32(sumi, q4v, q8v);
                }

#elif __256_AVXVNNI_NO_UNROLL__

                __m256i sumi0 = _mm256_setzero_si256();
                __m256i sumi1 = _mm256_setzero_si256();

                const __m256i m4 = _mm256_set1_epi8(0xF);

                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + j * 32));

                    // Unpack 4-bit values into two 32-byte vectors
                    __m256i q4lo = _mm256_and_si256(q4bits, m4);
                    __m256i q4hi = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                    // Load q8 (64 bytes) as two 256-bit vectors
                    const __m256i q8lo = _mm256_loadu_si256((const __m256i*)(q8 + j * 64 + 0));
                    const __m256i q8hi = _mm256_loadu_si256((const __m256i*)(q8 + j * 64 + 32));

                    // Do 4*dot product per 32-byte vectors
                    sumi0 = _mm256_dpbusd_epi32(sumi0, q4lo, q8lo);
                    sumi1 = _mm256_dpbusd_epi32(sumi1, q4hi, q8hi);
                }
                // Combine two 256-bit results into 512-bit
                sumi = _mm512_castsi256_si512(sumi0);
                sumi = _mm512_inserti32x8(sumi, sumi1, 1);

#elif __256_AVXVNNI_UNROLL4__

                __m256i sumi_lo = _mm256_setzero_si256();
                __m256i sumi_hi = _mm256_setzero_si256();
                const __m256i m4 = _mm256_set1_epi8(0x0F);

                for (uint64_t j = 0; j < QK_K / 64; j += 4) {
                    // Unroll iteration 0
                    {
                        __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j + 0) * 32));
                        __m256i q4lo = _mm256_and_si256(q4bits, m4);
                        __m256i q4hi = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                        __m256i q8lo = _mm256_loadu_si256((const __m256i*)(q8 + (j + 0) * 64 + 0));
                        __m256i q8hi = _mm256_loadu_si256((const __m256i*)(q8 + (j + 0) * 64 + 32));
                        sumi_lo = _mm256_dpbusd_epi32(sumi_lo, q4lo, q8lo);
                        sumi_hi = _mm256_dpbusd_epi32(sumi_hi, q4hi, q8hi);
                    }
                    // Unroll iteration 1
                    {
                        __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j + 1) * 32));
                        __m256i q4lo = _mm256_and_si256(q4bits, m4);
                        __m256i q4hi = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                        __m256i q8lo = _mm256_loadu_si256((const __m256i*)(q8 + (j + 1) * 64 + 0));
                        __m256i q8hi = _mm256_loadu_si256((const __m256i*)(q8 + (j + 1) * 64 + 32));
                        sumi_lo = _mm256_dpbusd_epi32(sumi_lo, q4lo, q8lo);
                        sumi_hi = _mm256_dpbusd_epi32(sumi_hi, q4hi, q8hi);
                    }
                    // Unroll iteration 2
                    {
                        __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j + 2) * 32));
                        __m256i q4lo = _mm256_and_si256(q4bits, m4);
                        __m256i q4hi = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                        __m256i q8lo = _mm256_loadu_si256((const __m256i*)(q8 + (j + 2) * 64 + 0));
                        __m256i q8hi = _mm256_loadu_si256((const __m256i*)(q8 + (j + 2) * 64 + 32));
                        sumi_lo = _mm256_dpbusd_epi32(sumi_lo, q4lo, q8lo);
                        sumi_hi = _mm256_dpbusd_epi32(sumi_hi, q4hi, q8hi);
                    }
                    // Unroll iteration 3
                    {
                        __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j + 3) * 32));
                        __m256i q4lo = _mm256_and_si256(q4bits, m4);
                        __m256i q4hi = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                        __m256i q8lo = _mm256_loadu_si256((const __m256i*)(q8 + (j + 3) * 64 + 0));
                        __m256i q8hi = _mm256_loadu_si256((const __m256i*)(q8 + (j + 3) * 64 + 32));
                        sumi_lo = _mm256_dpbusd_epi32(sumi_lo, q4lo, q8lo);
                        sumi_hi = _mm256_dpbusd_epi32(sumi_hi, q4hi, q8hi);
                    }
                }

                // Combine
                sumi = _mm512_castsi256_si512(sumi_lo);
                sumi = _mm512_inserti32x8(sumi, sumi_hi, 1);

#elif __512_maddubs__ // very slow

#pragma message("xx_vec_dot_q4_k_q8_k - no AVXVNNI")
                const __m512i m4 = _mm512_set1_epi8(0xf); // mask for low 4 bits

                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    // Load 32 bytes of 4-bit packed data
                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j * 32)));

                    // Load 64 bytes of q8 data
                    const __m512i q8v = _mm512_loadu_si512((const void*)(q8 + (j * 64)));

                    __m512i q4v = _mm512_castsi256_si512(q4bits);
                    q4v = _mm512_inserti32x8(q4v, _mm256_srli_epi16(q4bits, 4), 1);
                    q4v = _mm512_and_si512(q4v, m4);

                    // Step 1: multiply unsigned q4 × signed q8 → int16 pairs
                    __m512i prod16 = _mm512_maddubs_epi16(q4v, q8v); // 64 pairs → 32 int16
                    _512_madd_count++;

                    // Step 2: sum 2× int16 → int32 using madd
                    const __m512i ones = _mm512_set1_epi16(1);
                    __m512i prod32 = _mm512_madd_epi16(prod16, ones); // pairwise sum

                    // Step 3: accumulate
                    sumi = _mm512_add_epi32(sumi, prod32);
                }

#else // __512_fmadd_ps

#pragma message("xx_vec_dot_q4_k_q8_k - no AVXVNNI with fmadds_ps only")

                #if 0 // first try
                __m512 acc = _mm512_setzero_ps();

                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + j * 32));

                    // Unpack 4-bit pairs into 2 x 32 bytes (64 values total)
                    __m512i q4v = _mm512_castsi256_si512(q4bits);
                    q4v = _mm512_inserti32x8(q4v, _mm256_srli_epi16(q4bits, 4), 1);
                    q4v = _mm512_and_si512(q4v, m4);

                    __m512i q8v = _mm512_loadu_si512((const __m512i*)(q8 + j * 64));

                    // Convert to float
                    // __m512 q4f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si256(q4v)));
                    __m512 q4f = _mm512_cvtepi32_ps(q4v);
                    // __m512 q8f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si256(q8v)));
                    __m512 q8f = _mm512_cvtepi32_ps(q8v);

                    // Multiply and accumulate: acc += q4 * q8
                    acc = _mm512_fmadd_ps(q4f, q8f, acc);

                    // Do same for upper half
                    // __m512 q4f_hi = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti64x4_epi64(q4v, 1)));
                    __m512 q4f_hi = _mm512_cvtepi32_ps(q4v);

                    // __m512 q8f_hi = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti64x4_epi64(q8v, 1)));
                    __m512 q8f_hi = _mm512_cvtepi32_ps(q8v);

                    acc = _mm512_fmadd_ps(q4f_hi, q8f_hi, acc);

                    sumi = _mm512_castps_si512(acc);
                }
                #endif

                __m512 sum = _mm512_setzero_ps();
                const __m512i mask_0xf = _mm512_set1_epi8(0xf);

                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    // Load q4 (packed 4-bit values) into 256 bits
                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + j * 32));

                    // Load q8 (uint8_t) into 512 bits
                    const __m512i q8v_u8 = _mm512_loadu_si512((const void*)(q8 + j * 64));

                    // Unpack q4 values into two 128-bit lanes
                    __m128i q4_low = _mm256_extracti128_si256(q4bits, 0);
                    __m128i q4_high = _mm256_extracti128_si256(q4bits, 1);

                    // Extract low and high 4-bit values
                    __m128i q4l_lo = _mm_and_si128(q4_low, _mm_set1_epi8(0xf));
                    __m128i q4l_hi = _mm_and_si128(_mm_srli_epi16(q4_low, 4), _mm_set1_epi8(0xf));
                    __m128i q4h_lo = _mm_and_si128(q4_high, _mm_set1_epi8(0xf));
                    __m128i q4h_hi = _mm_and_si128(_mm_srli_epi16(q4_high, 4), _mm_set1_epi8(0xf));

                    // Pack into __m256i
                    __m256i q4u_lo = _mm256_set_m128i(q4l_hi, q4l_lo);
                    __m256i q4u_hi = _mm256_set_m128i(q4h_hi, q4h_lo);

                    // Combine both to get 64 values
                    __m512i q4v_u8 = _mm512_inserti64x4(_mm512_castsi256_si512(q4u_lo), q4u_hi, 1);

                    // Convert to float
                    __m512 q4f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(q4v_u8, 0))); // lower 32
                    __m512 q8f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(q8v_u8, 0))); // lower 32
                    sum = _mm512_fmadd_ps(q4f, q8f, sum);

                    q4f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(q4v_u8, 1))); // upper 32
                    q8f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(q8v_u8, 1))); // upper 32
                    sum = _mm512_fmadd_ps(q4f, q8f, sum);
                }

                sumi = _mm512_castps_si512(sum);

#endif
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

void
xx_vec_dot_q6_k_q8_k_x8 (
    const int n,
    float * s,
    size_t nr_nb1,
    const block_q6_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    int nrc
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m512i m2 = _mm512_set1_epi8(0x30);
    const __m512i m32s = _mm512_set1_epi8(32);

    //
    // Iterate through the specified number of columns.
    //

    block_q8_K_repack * y = (block_q8_K_repack *)vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        block_q6_K_repack * x = (block_q6_K_repack *)vx;

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

void
xx_vec_dot_q8_0_q8_0_x8 (
    const int n,
    float * s,
    size_t nr_nb1,
    const block_q8_0_repack * vx,
    size_t bx,
    const block_q8_0_repack * vy,
    size_t ncols,
    int nrc
    )
{
    #pragma comment(linker, "/EXPORT:xx_vec_dot_q8_0_q8_0_x8=" __FUNCTION__)

    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const __m512i zero512 = _mm512_setzero_si512();

    //
    // Iterate through the specified number of columns.
    //

    block_q8_0_repack * y = (block_q8_0_repack *)vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate throught the specified number of rows.
        //

        block_q8_0_repack * x = (block_q8_0_repack *)vx;

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

/*
    static uint32_t count = 128;

    if (count != 0) {
        count -= 1;
        printf("xx_vec_dot_q8_0_q8_0_x8 %08x\n", *(uint32_t *)s);
    }
*/

}

void
quantize_row_q4_0_x8 (
    const float * x,
    block_q4_0 * y,
    uint32_t vec_size
    )
{
#pragma comment(linker, "/EXPORT:quantize_row_q4_0_x8=" __FUNCTION__)

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
#pragma comment(linker, "/EXPORT:quantize_row_q4_k_x8=" __FUNCTION__)

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
#pragma comment(linker, "/EXPORT:quantize_row_q6_k_x8=" __FUNCTION__)

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
#pragma comment(linker, "/EXPORT:quantize_row_q4_k_q8_k_x8=" __FUNCTION__)

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
#pragma comment(linker, "/EXPORT:quantize_row_q8_0_x8=" __FUNCTION__)

    //
    // Quantize the x vector into q8_0 guants.
    //

    quantize_row_q8_0(x, y, vec_size);

    //
    // Make q8_0_repack quant blocks.
    //

    make_q8_0_repack_quant(vec_size, (block_q8_0_repack *)y, y);
}

enum ggml_type
ggml_repack_tensor_single_thread (
    const struct ggml_compute_params * params,
    struct ggml_tensor *tensor
    ) 
{

    enum ggml_type type = tensor->type;

    const int ith = params->ith;
    const int nth = params->nth;

#if 0
    if (tensor->flags & GGML_TENSOR_FLAG_DUP) {
        //
        // this tensor is duplicated in multiple operations so it is not safe for repacking
        // 

        if (ith == 0) {
            mul_mat_repack_failed_count += 1;
        }
        return type;
    }
#endif

    GGML_ASSERT((type == GGML_TYPE_Q4_0) ||
                (type == GGML_TYPE_Q2_K) ||
                (type == GGML_TYPE_Q3_K) ||
                (type == GGML_TYPE_Q4_K) ||
                (type == GGML_TYPE_Q6_K) ||
                (type == GGML_TYPE_Q8_0));

    //
    // Check if the tensor is contiguous and the number of elements is 0 mod QK_K.
    //
    uint64_t ne = tensor->ne[0];
    if (!ggml_is_contiguous(tensor) || ((ne % QK_K) != 0)) {
        if (ith == 0) {
            mul_mat_repack_failed_count += 1;
        }
        return type;
    }

    if (tensor->flags & GGML_TENSOR_FLAG_DUP) {
        //
        // Duplication is active meaning there is a copy of the 
        // tensor data being used by some other OPs. Allocate
        // a new buffer for this specific instance for repacking so 
        // the original copy remains intact for the other OPs.
        //
        char *duplicate_data = (char *)malloc(ggml_nbytes(tensor));
        memcpy(duplicate_data, tensor->data, ggml_nbytes(tensor));
        tensor->data = duplicate_data;
    }

    //
    // Make transformed quant based on current type.
    //
    // N.B. The original data and the repacked data share the same memory. They are
    //      exactly the same size and layout. The make repack quant function does
    //      the repack such that no extra memory needs to be allocated and there are
    //      no extra copies.
    //
    char * src_data = tensor->data;
    uint64_t nrows = tensor->ne[1];
    uint64_t stride = tensor->nb[1];

    if (type == GGML_TYPE_Q4_0) {
        type = GGML_TYPE_Q4_0_x8;
        for (uint64_t i = 0; i < nrows; i += 1) {
            make_q4_0_repack_quant(ne,
                                   (block_q4_0_repack *)src_data,
                                   (block_q4_0 *)src_data);
            src_data += stride;
        }
    } else if (type == GGML_TYPE_Q2_K) {
        type = GGML_TYPE_Q2_K_x8;
        for (uint64_t i = 0; i < nrows; i += 1) {
            make_q2_k_repack_quant(ne,
                                   (block_q2_K_repack *)src_data,
                                   (block_q2_K *)src_data);
            src_data += stride;
        }
    } else if (type == GGML_TYPE_Q3_K) {
        type = GGML_TYPE_Q3_K_x8;
        for (uint64_t i = 0; i < nrows; i += 1) {
            make_q3_k_repack_quant(ne,
                                   (block_q3_K_repack *)src_data,
                                   (block_q3_K *)src_data);
            src_data += stride;
        }
    } else if (type == GGML_TYPE_Q4_K) {
        type = GGML_TYPE_Q4_K_x8;
        for (uint64_t i = 0; i < nrows; i += 1) {
            make_q4_k_repack_quant(ne,
                                   (block_q4_K_repack *)src_data,
                                   (block_q4_K *)src_data);
            src_data += stride;
        }
    } else if (type == GGML_TYPE_Q6_K) {
        type = GGML_TYPE_Q6_K_x8;

        for (uint64_t i = 0; i < nrows; i += 1) {
            make_q6_k_repack_quant(ne,
                                    (block_q6_K_repack *)src_data,
                                    (block_q6_K *)src_data);
            src_data += stride;
        }
    } else if (type == GGML_TYPE_Q8_0) {
        type = GGML_TYPE_Q8_0_Q8_0_x8;
        for (uint64_t i = 0; i < nrows; i += 1) {
            make_q8_0_repack_quant(ne,
                                   (block_q8_0_repack *)src_data,
                                   (block_q8_0 *)src_data);
            src_data += stride;
        }
    }
    /*
    if (type != tensor->type) {
        printf("*** XBOX convert tensor %s - type %s - elements %zd succeeded\n",
               ggml_get_name(tensor),
               ggml_type_name(type),
               tensor->ne[0]);
    }
    */
    return type;
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

#if 0 // do not give up - try later on below
    if (tensor->flags & GGML_TENSOR_FLAG_DUP) {
        //
        // this tensor is duplicated in multiple operations so it is not safe for repacking
        // 

        if (ith == 0) {
            mul_mat_repack_failed_count += 1;
        }
        return type;
    }
#endif

    GGML_ASSERT((type == GGML_TYPE_Q4_0) ||
                (type == GGML_TYPE_Q2_K) ||
                (type == GGML_TYPE_Q3_K) ||
                (type == GGML_TYPE_Q4_K) ||
                (type == GGML_TYPE_Q6_K) ||
                (type == GGML_TYPE_Q8_0));

    //
    // Check if the tensor is contiguous and the number of elements is 0 mod QK_K.
    //
    uint64_t ne = tensor->ne[0];
    if (!ggml_is_contiguous(tensor) || ((ne % QK_K) != 0)) {
        if (ith == 0) {
            mul_mat_repack_failed_count += 1;
        }
        return;
    }

    if (tensor->flags & GGML_TENSOR_FLAG_DUP) {
        //
        // Duplication is active meaning there is a copy of the 
        // tensor data being used by some other OPs. Allocate
        // a new buffer for this specific instance for repacking so 
        // the original copy remains intact for the other OPs.
        //
        if (!ith) {
            char *duplicate_data = (char *)malloc(ggml_nbytes(tensor));
            memcpy(duplicate_data, tensor->data, ggml_nbytes(tensor));

            //
            // wait for all threads to arrive before we change tensor->data
            //
            ggml_wait_to_finalize_xbox(params);
            tensor->data = duplicate_data;
        }

        //
        // wait for !ith to be done and tensor->data to be updated
        //
        ggml_wait_for_done_xbox(params);
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

    /*
    static uint32_t count = 8;
    if (count != 0) {
        count -= 1;
        uint32_t contiguous = ggml_is_contiguous(tensor);
        printf("contiguous %u, row size %zu, stride %zu\n",
               contiguous,
               ggml_row_size(tensor->type, ne),
               tensor->nb[1]);
    }
    */

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

    ggml_wait_for_done_xbox(params);

    //
    // N.B. All threads write the same value to tensor type so no special
    //      synchronization is required.
    //
    tensor->type = type;

    return;
}
