#define GGML_COMMON_IMPL_C

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "quants.h"
#include "ggml-cpu-repack.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>

// Clang requires explicit target attributes for VNNI intrinsics even when
// the translation unit is intended for AVX-512 targets. MSVC does not enforce this.
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("avx512f,avx512vnni,avx512vl"))), apply_to=function)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:make_q4_0_repack_quant=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:make_q2_k_repack_quant=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:make_q3_k_repack_quant=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:make_q4_k_repack_quant=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:make_q6_k_repack_quant=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:make_q8_0_repack_quant=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:make_q236_k_q8_k_repack_quant=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:make_q4_k_q8_k_repack_quant=" __FUNCTION__)
#endif

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

/*
Analysis: Q4_0 vec_dot — b612 vs Upstream

b612: xx_vec_dot_q4_0_q8_0_x8 (lines 866–998, 133 lines)

Data Layout (block_q4_0_repack, custom struct):

 - d[8] — 8 × fp16 scales (one per original block_q4_0)
 - qs[QK_K/2] = qs[128] — nibble-packed, interleaved from 8 block_q4_0 blocks
 - Total: 16 + 128 = 144 bytes per super-block (packs 8 original 20-byte blocks → 144 vs 160)

Key differences from K-quants:

 - Q4_0 is unsigned (0–15), bias-subtracted by 8 → signed
 - 8 separate per-block scales (not a shared super-scale + sub-scales)
 - Scales are a vector of 8 fp16, not a single fp16 d

Vec-dot structure:

 for each column:
   for each row:
     for each super-block:
       ── d = combined 512-bit scale vector (8 fp16×fp16 → 8 fp32, duplicated)
       ── inner loop (4 iterations, QK_K/(QK8_0*2) = 256/64 = 4):
            load 256-bit q4 packed → split low/high nibbles → 512-bit unsigned
            load 512-bit q8 signed
            sumi = dpbusd(sumi, qx, qy)       [1 VNNI]
            bias = dpbusd(bias, offset_8, qy)  [1 VNNI]
       ── sumi = sumi - bias
       ── acc = fmadd(d, cvt(sumi), acc)
     ── reduce 512→256→128→scalar

Key characteristics:

 - 2 dpbusd per inner iteration × 4 iterations = 8 dpbusd per super-block
 - No scale decode needed (direct fp16 load)
 - Scale combination: d is a vector (8 distinct float multipliers), not a scalar
  - Uses _mm512_insertf32x8 to broadcast 8 scales → 16 lanes (duplicated)
  - This is key: the FMA applies 8 different scales simultaneously
 - No mins (Q4_0 is purely scale + offset)
 - Weight bytes loaded per super-block: 128 (qs) + 16 (d) = 144 bytes
 - Register budget: ~8 zmm (acc + sumi + bias + qx + qy + d + offset + m4)
 - Clang-specific workaround for _mm_loadu_ph not being available

Upstream: ggml_gemm_q4_0_8x8_q8_0 (lines 1165–1910, ~745 lines)

Data Layout (block_q4_0x8 = block<4, 8>):

 - d[8] — 8 × fp16 scales
 - qs[QK8_0
  * 4] = qs[128] — 8 blocks × 16 bytes interleaved

Same storage as b612's block_q4_0_repack (both 144 bytes). But the interleaving pattern differs.

GEMM structure (AVX-512 path):

 outer: groups of 4 rows (a_ptrs[4])
   inner: pairs of 2 × block_q4_0x8 columns (16 columns total)
     acc_rows[16] — 16 × 512-bit FP accumulators
     per block:
       ── Load 8 × 256-bit from two b_ptrs (weight data for 16 cols)
       ── 6 blend + 6 permute → 4 × 512-bit patterns
       ── 8 × shuffle_epi8 (signextendlut) for nibble→signed byte
       ── 16 × shuffle_epi32 sp1 + 16 × shuffle_epi32 sp2 (two shuffle patterns)
       ── per row-pair (4 iterations):
            4 × loadu_256 → 8 × permute → 8 × inserti32x8 → 8 × shuffle_epi32 × 2
            8 × mul_sum_i8_pairs_acc (= 8 × maddubs + 8 × madd_epi16)
            → sp1 + sp2 → add → 4 blend → 4 FMA
     ── store 16 output elements

Key characteristics:

 - Output tile: 16 rows × 16 cols per outer iteration
 - Q4_0 weights decoded via signextendlut (shuffle_epi8 LUT) — signed representation, no bias needed
 - Uses mul_sum_i8_pairs_acc (= maddubs_epi16 + madd_epi16) — NOT VNNI dpbusd
 - Massive shuffle overhead: 32+ shuffle_epi32 per block per row-pair
 - Two shuffle patterns (sp1, sp2) needed for the 2×2 MMLA-style computation
 - 16 FP accumulators (acc_rows[16]) = 16 zmm dedicated to results
 - Weight loaded from 2 × block_q4_0x8 simultaneously (16 columns)
 - Each weight load serves 16 rows × 16 cols = 256 output elements
 - Has fallback paths for nr%16 ≠ 0 (single row-pair at lines 1399+)

Comparative Table

┌────────────────────────────────┬──────────────────────────────────┬───────────────────────────────────────────┐
│ Metric                         │ b612 (xx_vec_dot_q4_0_q8_0_x8)   │ Upstream (ggml_gemm_q4_0_8x8_q8_0)        │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ Lines of code                  │ 133                              │ ~745                                      │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ Output tile                    │ 1 row × 1 col                    │ 16 rows × 16 cols                         │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ Weight struct                  │ block_q4_0_repack (144B)         │ block_q4_0x8 (144B)                       │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ Signed repr                    │ Unsigned + bias subtraction      │ Signed via LUT (shuffle_epi8)             │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ Core multiply                  │ dpbusd (VNNI, 1 µop)             │ maddubs + madd (2 µops)                   │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ dpbusd/maddubs per super-block │ 8 dpbusd + 8 bias                │ 128 maddubs + 64 madd (per tile)          │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ Shuffle overhead               │ None                             │ Massive (32+ shuffle_epi32 per block-row) │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ Scale handling                 │ Vector 8 × fp16→fp32 direct      │ Vector + cross-column broadcast           │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ Register pressure              │ ~8 zmm                           │ ~30+ zmm (16 acc + data + temps)          │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ Weight reuse per load          │ 1 (1 row)                        │ 16 (16 rows)                              │
├────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────┤
│ Activation reuse per load      │ 1 (1 col)                        │ 16 (16 cols)                              │
└────────────────────────────────┴──────────────────────────────────┴───────────────────────────────────────────┘

Performance: Prefill vs Token Generation

Token generation (batch=1): b612 wins decisively:

 - VNNI dpbusd = 1 µop vs maddubs+madd = 2-3 µops
 - Tiny 8-zmm footprint = zero spill pressure, fits µop cache perfectly
 - At batch=1, memory bandwidth is the bottleneck — b612 loads 144 bytes once, upstream loads ~the same but with enormous decode 
overhead
 - The 16×16 tile in upstream is wasted at batch=1 (only 1 row/col active)

Prompt prefill (batch ≥ 4): Upstream wins:

 - Weight data loaded once for 16 rows × 16 cols = 256 FMAs per weight load
 - b612 re-reads weights ncols times — at batch=16, that's 16× the weight bandwidth
 - Despite 2× compute cost per multiply, the 16× bandwidth savings dominate

Batch-tiling potential for b612

Q4_0 is uniquely well-suited for the _cp batch-tiling treatment:

 1. Simplest inner loop of any quant — no scale decode, no mins, just dpbusd + bias
 2. Register budget is extremely light: only ~8 of 32 zmm used → room for 4× (sumi + bias + d) + acc = ~20 zmm
 3. Scale is a vector, not scalar — but this works fine: each column gets the same weight d vector, just multiplied by a different 
activation d

Unique Q4_0 consideration: The scale d is a vector of 8 fp16 values (one per sub-block). In the K-quants, the shared part was just a
single scalar x_d. For Q4_0:

 - SHARED: load x[i].d (weight 8×fp16 → 8×fp32 vector) ONCE
 - PER-COLUMN: load y_col[i].d (activation 8×fp16 → 8×fp32), compute d_col = x_d
  * y_col_d (element-wise fp32 multiply)
 - The weight d load + conversion is shared; only the per-column activation d differs

Estimated speedup at batch=4: ~3.5× reduction in weight bandwidth (same as other quants). The absolute savings per super-block (144
bytes × 3) are comparable to Q4_K (128 bytes × 3) and less than Q6_K (210 bytes × 3).

Relative benefit ranking (updated):

┌───────┬────────────────────────┬─────────────────────────┐
│ Quant │ Weight bytes/block     │ Benefit of _cp tiling   │
├───────┼────────────────────────┼─────────────────────────┤
│ Q6_K  │ 210                    │ Highest                 │
├───────┼────────────────────────┼─────────────────────────┤
│ Q4_0  │ 144                    │ High                    │
├───────┼────────────────────────┼─────────────────────────┤
│ Q4_K  │ 144 (128 qs + 16 meta) │ High                    │
├───────┼────────────────────────┼─────────────────────────┤
│ Q3_K  │ 110                    │ Medium                  │
├───────┼────────────────────────┼─────────────────────────┤
│ Q2_K  │ 84                     │ Moderate                │
└───────┴────────────────────────┴─────────────────────────┘

Q4_0 is tied with Q4_K for second place. The batch-tiling is straightforward to implement following the same pattern — the only wrinkle
is the 8-element vector scale instead of a scalar, but that actually works identically since the shared part is the weight's d[8], and
only the per-column activation d[8] differs.

*/
void
xx_vec_dot_q4_0_q8_0_x8_dc (
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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:xx_vec_dot_q4_0_q8_0_x8_dc=" __FUNCTION__)
#endif

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

#ifdef __clang__

                uint16_t* x_half = (uint16_t*)(x[i].d);
                uint16_t* y_half = (uint16_t*)(y[i].d);

                // Load as 128-bit integer vectors
                __m128i x_raw = _mm_loadu_si128((__m128i*)x_half);
                __m128i y_raw = _mm_loadu_si128((__m128i*)y_half);

                // Convert to 256-bit single-precision float vectors
                __m256 x_fp32 = _mm256_cvtph_ps(x_raw);
                __m256 y_fp32 = _mm256_cvtph_ps(y_raw);

                // Multiply
                __m256 scale = _mm256_mul_ps(x_fp32, y_fp32);

                // Cast and insert into __m512
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);

#else

                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m128h yd = _mm_loadu_ph(y[i].d);
                const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                                   _mm256_cvtph_ps(yd));
        
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);

#endif // __clang__
        
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

//
// Batch-optimized Q4_0 × Q8_0 vec_dot for prompt prefill (ncols >= 4).
//
// Key optimization: processes 4 activation columns per weight load, reducing
// weight memory traffic by ~4×. Uses the same repack format as
// xx_vec_dot_q4_0_q8_0_x8_dc — no changes to make_q4_0_repack_quant needed.
//
// Q4_0 difference from K-quants: d is a VECTOR of 8 fp16 values (one per
// sub-block), not a single scalar. SHARED: weight d[8] loaded once.
// PER-COLUMN: activation d[8] loaded, element-wise multiply with weight d.
//
// Register budget (AVX-512, 32 zmm):
//   4 × acc (zmm) + 4 × sumi (zmm) + 4 × bias (zmm) +
//   1 × qx (zmm) + 1 × qy (zmm) + offset + m4 ≈ 15 zmm
//
void
xx_vec_dot_q4_0_q8_0_x8 ( // Batch tiled Q4_0 CP
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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:xx_vec_dot_q4_0_q8_0_x8_cp=" __FUNCTION__)
#endif

    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const __m512i offset = _mm512_set1_epi8(8);
    const __m512i m4 = _mm512_set1_epi8(0xf);

    block_q8_0_repack * y_base = (block_q8_0_repack *)vy;
    float * s_base = s;

    const uint64_t ncols4 = ncols & ~3ULL;
    uint64_t l = 0;

    //
    // Phase 1: Process 4 activation columns at a time.
    //
    // For each weight row + super-block, load weight qs and d once,
    // unpack nibbles once, then dpbusd against 4 activation columns.
    //

    for (; l < ncols4; l += 4) {

        block_q8_0_repack * y0 = y_base + (l + 0) * nb;
        block_q8_0_repack * y1 = y_base + (l + 1) * nb;
        block_q8_0_repack * y2 = y_base + (l + 2) * nb;
        block_q8_0_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        block_q4_0_repack * x = (block_q4_0_repack *)vx;

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
                // SHARED: Load weight d[8] (fp16 → fp32) once.
                //

#ifdef __clang__
                __m128i x_raw = _mm_loadu_si128((__m128i*)x[i].d);
                __m256 x_fp32 = _mm256_cvtph_ps(x_raw);
#else
                const __m128h xd = _mm_loadu_ph(x[i].d);
                __m256 x_fp32 = _mm256_cvtph_ps(xd);
#endif

                //
                // Per-column: subtract bias, build per-column scale vector, FMA.
                //

#ifdef __clang__

#define Q40_COL_FMA(col, y_ptr, sumi_v, bias_v, acc_v) \
                do { \
                    __m128i y_raw_##col = _mm_loadu_si128((__m128i*)(y_ptr)[i].d); \
                    __m256 y_fp32_##col = _mm256_cvtph_ps(y_raw_##col); \
                    __m256 scale_##col = _mm256_mul_ps(x_fp32, y_fp32_##col); \
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

#else

#define Q40_COL_FMA(col, y_ptr, sumi_v, bias_v, acc_v) \
                do { \
                    const __m128h yd_##col = _mm_loadu_ph((y_ptr)[i].d); \
                    const __m256 scale_##col = _mm256_mul_ps(x_fp32, \
                                                _mm256_cvtph_ps(yd_##col)); \
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

#endif // __clang__
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
    // Falls back to the original single-column loop.
    //

    block_q8_0_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        block_q4_0_repack * x = (block_q4_0_repack *)vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();

#ifdef __clang__
                __m128i x_raw = _mm_loadu_si128((__m128i*)x[i].d);
                __m128i y_raw = _mm_loadu_si128((__m128i*)y_rem[i].d);
                __m256 x_fp32 = _mm256_cvtph_ps(x_raw);
                __m256 y_fp32 = _mm256_cvtph_ps(y_raw);
                __m256 scale = _mm256_mul_ps(x_fp32, y_fp32);
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);
#else
                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m128h yd = _mm_loadu_ph(y_rem[i].d);
                const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                                    _mm256_cvtph_ps(yd));
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);
#endif

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

/*
Q2_K Repack Comparison: b612 vs Upstream

b612 xx_vec_dot_q2_k_q8_k_x8 (106 lines)

┌───────────────────────┬─────────────────────────────────────────────────────┐
│ Aspect                │ Detail                                              │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Tile size             │ 1 row × 1 col (pure vec-dot)                        │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ ISA                   │ AVX-512 VNNI (_mm512_dpbusd_epi32)                  │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Q2 decode             │ Load 512-bit, AND+shift×4 to extract 4 sub-blocks   │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Scales                │ 16 × 4-bit packed (simple AND/shift decode)         │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Mins                  │ 256-bit path (_mm256_madd_epi16 with 16 bsums)      │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Ops/super-block       │ 4 dpbusd + scale decode + 1 fmadd_ps                │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Repack struct         │ typedef block_q2_K — same layout, in-place repack   │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Register pressure     │ ~10 zmm registers                                   │
└───────────────────────┴─────────────────────────────────────────────────────┘

Upstream ggml_gemm_q2_K_8x8_q8_K (~2881 lines)

┌───────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
│ Aspect                │ Detail                                                                                   │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ Tile size             │ 16 rows × 16 cols (true GEMM)                                                            │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ ISA                   │ AVX-512 maddubs_epi16 (NOT VNNI) + madd_epi16 for scale                                  │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ Q2 decode             │ 8 sub-blocks/super-block (2× outer, 4× shift) — 64 shuffled RHS vectors per sb iteration │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ Scales                │ 16 scale vectors (8 sb pairs × 2 col groups) extracted via shuffle+mask                  │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ Mins                  │ 512-bit path: 4 bsum groups × 4 min vectors × 4 rows = 16 madd_epi16 per rp              │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ Ops/super-block       │ 128 maddubs + 32 madd(scale) + 16 madd(mins) + 4 fmadd_ps                                │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ Repack struct         │ block_q2_Kx8: 8 blocks interleaved (d[8], dmin[8], scales[128], qs[512])                 │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ Register pressure     │ Extremely high — 64 RHS shuffle results + 32 LHS loads + 16 acc + 16 min_acc             │
└───────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘

Key Differences from Q4_K Comparison

Q2_K is harder than Q4_K for the upstream GEMM because:

 1. 8 sub-blocks vs 4 in Q4_K → 2× more iterations in the inner loop
 2. 64 shuffled RHS vectors per sb pass (vs ~32 for Q4_K) → massive shuffle overhead
 3. 16 scale/min pairs vs 8 → more complex scale extraction
 4. Function is ~2× larger than Q4_K GEMM (2881 vs 1453 lines)

Verdict

┌─────────┬────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Batch   │ Winner         │ Why                                                                                                      │
│ size    │                │                                                                                                          │
├─────────┼────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ batch = │ b612 (big win) │ VNNI dpbusd is 1 instruction vs maddubs+madd. 106 lines vs 2881. Minimal shuffle overhead. Clean 512-bit │
│ 1       │                │ paths.                                                                                                   │
├─────────┼────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ batch ≥ │ Upstream       │ 16×16 tiling amortizes weight loads. But the advantage is smaller than Q4_K because the 8-sub-block      │
│ 4       │ (moderate win) │ inner loop creates enormous shuffle pressure that partially negates the data reuse benefit.              │
├─────────┼────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ batch ≥ │ Upstream       │ 16-row tiling fully utilized. Weight data loaded once for 16 rows.                                       │
│ 8       │ (clear win)    │                                                                                                          │
└─────────┴────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Bottom line: Same pattern as Q4_K — b612 wins decisively at batch=1 (token generation), upstream wins at batch≥4 (prefill). However,
the gap is narrower for Q2_K because the upstream's complexity cost is higher (8 sub-blocks create disproportionate shuffle overhead),
making the 4-column tiling optimization even more impactful for a hypothetical xx_vec_dot_q2_k_q8_k_x8_cp().

*/

void
xx_vec_dot_q2_k_q8_k_x8_dc (
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

//
// Batch-optimized Q2_K × Q8_K vec_dot for prompt prefill (ncols >= 4).
//
// Key optimization: processes 4 activation columns per weight load, reducing
// weight memory traffic by ~4×. Uses the same in-place repack format as
// xx_vec_dot_q2_k_q8_k_x8 — no changes to make_q2_k_repack_quant needed.
//
// Inner loop: load q2 weights ONCE per super-block, then VNNI dpbusd against
// 4 q8 activation streams. Scales and mins extracted once and shared.
//
// Q2_K specifics vs Q4_K:
//   - 2-bit quants: single 512-bit load, extract via AND(3) + shift-right-2
//   - 16 scales (4-bit packed) vs Q4_K's 8 scales (6-bit packed) — simpler decode
//   - 256-bit mins path (16 bsums) vs Q4_K's 128-bit path (8 bsums)
//
// Register budget (AVX-512, 32 zmm):
//   4 × acc (zmm)  + 4 × sumi (zmm)  + 1 × scale (zmm) +
//   1 × q2v (zmm)  + 1 × q2bits (zmm) + 1 × m3 (zmm)   ≈ 12 zmm
//   4 × mins_acc (ymm, bottom of zmm)
// Fits comfortably with headroom for compiler temporaries.
//

/*
xx_vec_dot_q2_k_q8_k_x8_cp() — Batch-Tiled Q2_K Vec_Dot

Design (same pattern as Q4_K):

 - Phase 1: Groups of 4 columns — load q2 weights ONCE per super-block, dpbusd against 4 activation streams
 - Phase 2: Remaining 0-3 columns processed one-at-a-time (original path)
 - No repack format change — same block_q2_K_repack

Q2_K-specific details vs Q4_K:

┌──────────────────┬───────────────────────────────────────┬───────────────────────────────────────────┐
│ Aspect           │ Q4_K_cp                               │ Q2_K_cp                                   │
├──────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│ Weight load      │ 256-bit per sub-block (nibble unpack) │ 512-bit once, extract via AND+shift×4     │
├──────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│ Inner loop iters │ 4 (QK_K/64)                           │ 4 (QK_K/64)                               │
├──────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│ Scale decode     │ 6-bit packed (complex bitmask)        │ 4-bit packed (simple AND/shift)           │
├──────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│ Mins path        │ 128-bit (8 bsums)                     │ 256-bit (16 bsums)                        │
├──────────────────┼───────────────────────────────────────┼───────────────────────────────────────────┤
│ Reduction        │ 512→256→128, sub 128-bit mins         │ 512→256, sub 256-bit mins, →128           │
└──────────────────┴───────────────────────────────────────┴───────────────────────────────────────────┘

Register budget: ~17 zmm of 32 available — comfortable.
*/
void
xx_vec_dot_q2_k_q8_k_x8 ( // batch-tiled Q2_K
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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:xx_vec_dot_q2_k_q8_k_x8_cp=" __FUNCTION__)
#endif

    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const __m512i m3 = _mm512_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    block_q8_K_repack * y_base = (block_q8_K_repack *)vy;
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

        block_q8_K_repack * y0 = y_base + (l + 0) * nb;
        block_q8_K_repack * y1 = y_base + (l + 1) * nb;
        block_q8_K_repack * y2 = y_base + (l + 2) * nb;
        block_q8_K_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        block_q2_K_repack * x = (block_q2_K_repack *)vx;

        for (uint64_t k = 0; k < nrows; k += 1) {

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            __m256 mins_acc0 = _mm256_setzero_ps();
            __m256 mins_acc1 = _mm256_setzero_ps();
            __m256 mins_acc2 = _mm256_setzero_ps();
            __m256 mins_acc3 = _mm256_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                //
                // SHARED: extract weight d/dmin and decode 4-bit scales/mins.
                // Done once per super-block, reused across all 4 columns.
                //

                const float x_d    = GGML_FP16_TO_FP32(x[i].d);
                const float x_dmin = GGML_FP16_TO_FP32(x[i].dmin);

                const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
                const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
                const __m128i mins8   = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);

                const __m512i scale = _mm512_cvtepi8_epi32(scales8);

                //
                // SHARED: Load weight q2 data once, extract 2-bit values in inner loop,
                // then dpbusd against 4 activation columns.
                //

                __m512i q2bits = _mm512_loadu_si512((const __m512i*)x[i].qs);

                __m512i sumi0 = _mm512_setzero_si512();
                __m512i sumi1 = _mm512_setzero_si512();
                __m512i sumi2 = _mm512_setzero_si512();
                __m512i sumi3 = _mm512_setzero_si512();

                for (uint64_t j = 0; j < QK_K / 64; ++j) {

                    // Extract 2-bit values from q2bits (shared across all 4 columns)
                    const __m512i q2v = _mm512_and_si512(q2bits, m3);
                    q2bits = _mm512_srli_epi16(q2bits, 2);

                    // Dot product against 4 activation columns
                    sumi0 = _mm512_dpbusd_epi32(sumi0, q2v, _mm512_loadu_si512(y0[i].qs + (j * 64)));
                    sumi1 = _mm512_dpbusd_epi32(sumi1, q2v, _mm512_loadu_si512(y1[i].qs + (j * 64)));
                    sumi2 = _mm512_dpbusd_epi32(sumi2, q2v, _mm512_loadu_si512(y2[i].qs + (j * 64)));
                    sumi3 = _mm512_dpbusd_epi32(sumi3, q2v, _mm512_loadu_si512(y3[i].qs + (j * 64)));
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
                // SHARED mins extraction, per-column bsums (256-bit path).
                // Q2_K has 16 bsums (not 8 like Q4_K), so mins uses 256-bit.
                //

                const __m256i mins = _mm256_cvtepi8_epi16(mins8);

                const float dmin0 = y0[i].d * x_dmin;
                const float dmin1 = y1[i].d * x_dmin;
                const float dmin2 = y2[i].d * x_dmin;
                const float dmin3 = y3[i].d * x_dmin;

                const __m256i prod0 = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *)y0[i].bsums));
                const __m256i prod1 = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *)y1[i].bsums));
                const __m256i prod2 = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *)y2[i].bsums));
                const __m256i prod3 = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *)y3[i].bsums));

                mins_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(dmin0), _mm256_cvtepi32_ps(prod0), mins_acc0);
                mins_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(dmin1), _mm256_cvtepi32_ps(prod1), mins_acc1);
                mins_acc2 = _mm256_fmadd_ps(_mm256_set1_ps(dmin2), _mm256_cvtepi32_ps(prod2), mins_acc2);
                mins_acc3 = _mm256_fmadd_ps(_mm256_set1_ps(dmin3), _mm256_cvtepi32_ps(prod3), mins_acc3);
            }

            //
            // Horizontal reduction: 512 → 256, subtract 256-bit mins, 256 → 128, hadd to scalar.
            //

#define REDUCE_Q2_CP(acc_v, mins_v, dest, idx) \
            { \
                const __m256 _res = _mm256_sub_ps( \
                    _mm256_add_ps(_mm512_castps512_ps256(acc_v), \
                                  _mm512_extractf32x8_ps(acc_v, 1)), \
                    mins_v); \
                __m128 _t0 = _mm_add_ps(_mm256_castps256_ps128(_res), \
                                         _mm256_extractf128_ps(_res, 1)); \
                const __m128 _t1 = _mm_hadd_ps(_t0, _t0); \
                dest[idx] = _mm_cvtss_f32(_mm_hadd_ps(_t1, _t1)); \
            }

            REDUCE_Q2_CP(acc0, mins_acc0, s0, k);
            REDUCE_Q2_CP(acc1, mins_acc1, s1, k);
            REDUCE_Q2_CP(acc2, mins_acc2, s2, k);
            REDUCE_Q2_CP(acc3, mins_acc3, s3, k);

#undef REDUCE_Q2_CP

            x += nb;
        }
    }

    //
    // Phase 2: Process remaining columns one at a time (original path).
    //

    block_q8_K_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        block_q2_K_repack * x = (block_q2_K_repack *)vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();
            __m256 mins_acc = _mm256_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                const float d = y_rem[i].d * GGML_FP16_TO_FP32(x[i].d);
                const float dmin = y_rem[i].d * GGML_FP16_TO_FP32(x[i].dmin);

                const uint8_t * q2 = x[i].qs;
                const int8_t * q8 = y_rem[i].qs;

                const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
                __m512i q2bits = _mm512_loadu_si512((const __m512i*)q2);

                const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
                const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);

                const __m512i scale = _mm512_cvtepi8_epi32(scales8);

                __m512i sumi = _mm512_setzero_si512();

                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));
                    const __m512i q2v = _mm512_and_si512(q2bits, m3);
                    q2bits = _mm512_srli_epi16(q2bits, 2);

                    sumi = _mm512_dpbusd_epi32(sumi, q2v, q8v);
                }

                sumi = _mm512_mullo_epi32(sumi, scale);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);

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

/* 
Analysis: Q3_K and Q6_K vec_dot — Prefill vs Token Gen

xx_vec_dot_q3_k_q8_k_x8 (lines 1484–1642)

Data Layout (block_q3_K_repack = block_q3_K, 110 bytes/super-block):

 - hmask[32] — high bits (1 bit per weight, packed 8/byte)
 - qs[64] — low 2 bits (2 bits per weight, packed 4/byte)
 - scales[12] — 16 scales in 6-bit packed format
 - d — 1× fp16 scale (no mins/dmin — Q3_K is scale-only)

Vec-dot structure (159 lines):

 for each column:
   for each row:
     for each super-block:
       ── scale decode: 6-bit unpack → 16 scales (insert_epi64 × 2, bitwise)
       ── q3bytes = load 512-bit (low 2-bits, packed 4/subblock)
       ── inner loop (4 iterations per super-block):
            q3lv = AND(q3bytes, 3); shift q3bytes >>= 2     [2-bit extraction]
            sumi = dpbusd(sumi, q3lv, q8v)                  [low-bits dot]
            hmask = load uint64 → kmask64                    [high-bit mask]
            q3hv = mask_blend(hmask, 4, 0)                   [high bit = 4 or 0]
            bias = dpbusd(bias, q3hv, q8v)                   [bias accumulation]
       ── sumi = sumi - bias; mullo(sumi, scales)
       ── fmadd to acc
     ── reduce 512→128→scalar

Key characteristics:

 - 16 scales (like Q2_K), decoded from 6-bit packed format — more complex than Q6_K's 8-bit scales
 - 2 dpbusd per inner iteration (sumi + bias) × 4 iterations = 8 dpbusd per super-block
 - High bits read as uint64 → __mmask64 → mask_blend — unique branchy-mask approach
 - No mins/dmin accumulator — Q3_K uses signed scales (scales - 32), bias via q3hv term
 - Block size: 110 bytes (smallest K-quant weight struct)

Register budget: ~11 zmm (1 acc + 1 sumi + 1 bias + 1 q3bytes + 1 q3lv + 1 q3hv + 1 q8v + m3 + m4 + zero512 + scales)

--------------------------------------------------------------------------------------------------------------------------------------

Comparative Table

┌──────────────────────────────┬──────────────────────────────┐
│ Metric                       │ Q3_K                         │
├──────────────────────────────┼──────────────────────────────┤
│ Bits per weight              │ 3.4375                       │
├──────────────────────────────┼──────────────────────────────┤
│ Block size (bytes)           │ 110                          │
├──────────────────────────────┼──────────────────────────────┤
│ Scales count                 │ 16 (6-bit packed)            │
├──────────────────────────────┼──────────────────────────────┤
│ Scale decode cost            │ High (bitwise)               │
├──────────────────────────────┼──────────────────────────────┤
│ dpbusd per super-block       │ 8                            │
├──────────────────────────────┼──────────────────────────────┤
│ Has mins/dmin?               │ No                           │
├──────────────────────────────┼──────────────────────────────┤
│ Bit unpacking method         │ 2-bit AND+shift + mask_blend │
├──────────────────────────────┼──────────────────────────────┤
│ Weight loads per super-block │ 1×512b (qs) + 4×64b (hmask)  │
├──────────────────────────────┼──────────────────────────────┤
│ Weight bytes loaded          │ 64 + 32 = 96                 │
├──────────────────────────────┼──────────────────────────────┤
│ Activation bytes loaded      │ 4×512b = 256                 │
├──────────────────────────────┼──────────────────────────────┤
│ zmm registers used           │ ~11 of 32                    │
└──────────────────────────────┴──────────────────────────────┘

--------------------------------------------------------------------------------------------------------------------------------------

Performance: Prefill vs Token Generation

Token generation (batch=1): Both routines are well-suited. The 1-row × 1-col inner loop means:

 - Weight data streams sequentially through each super-block
 - Q3_K is faster here because it reads 96 bytes of weight per super-block vs Q6_K's 192 bytes. With batch=1, the bottleneck is memory 
bandwidth (weights >> activations), so smaller quant = faster.
 - Both use identical computation: 8 dpbusd + scale multiply. Compute cost is equal; memory cost favors Q3_K by ~2×.

Prompt prefill (batch ≥ 4): This is where both routines lose efficiency. The outer loop structure is:

 for each column:        ← sequential over batch columns
   for each row:         ← sequential over output rows
     for each block: ... ← weight data loaded here

Each column re-reads the entire weight matrix. For batch=B, weight data is read B times. This is the same bottleneck we fixed in Q4_K
and Q2_K with the _cp variants.

Q3_K prefill penalty: Weight matrix re-read B× despite only 96 bytes/block. Total weight traffic = B × 96 × nb × nrows. The fix would
save (B-1)/B ≈ 75% of weight bandwidth at batch=4.

Q6_K prefill penalty: Much worse — 192 bytes/block loaded B times. Total weight traffic = B × 192 × nb × nrows. Since Q6_K has ~2× the
weight data of Q3_K, the absolute waste from re-reading is 2× larger.

Both are strong candidates for the _cp batch-tiling optimization, for the same reason as Q4_K/Q2_K:

 - Plenty of register headroom (10–11 of 32 zmm used → room for 4× accumulators)
 - No repack format change needed
 - Weight loads dominate the inner loop and can be shared across 4 columns

Relative benefit ranking:

┌───────┬────────────────────┬──────────────────────────┬──────────────────┐
│ Quant │ Weight bytes/block │ Prefill batch=4 waste    │ Benefit of _cp   │
├───────┼────────────────────┼──────────────────────────┼──────────────────┤
│ Q6_K  │ 192                │ Highest (largest blocks) │ Highest          │
├───────┼────────────────────┼──────────────────────────┼──────────────────┤
│ Q4_K  │ 128                │ High                     │ High             │
├───────┼────────────────────┼──────────────────────────┼──────────────────┤
│ Q3_K  │ 96                 │ Medium                   │ Medium           │
├───────┼────────────────────┼──────────────────────────┼──────────────────┤
│ Q2_K  │ 64                 │ Lowest (smallest blocks) │ Moderate         │
└───────┴────────────────────┴──────────────────────────┴──────────────────┘

Q6_K would benefit most from the _cp treatment because it has the most weight data to reload. Q3_K benefits less in absolute terms but
the relative speedup (4× weight bandwidth reduction) is the same for all.

Additional Q3_K note: The hmask reads (uint64 loads from x[i].hmask + j) are particularly wasteful in the prefill path — these 4 ×
8-byte scalar loads per super-block are non-SIMD and would benefit from being shared across columns.

*/

void
xx_vec_dot_q3_k_q8_k_x8_dc (
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

//
// Batch-optimized Q3_K × Q8_K vec_dot for prompt prefill (ncols >= 4).
//
// Key optimization: processes 4 activation columns per weight load, reducing
// weight memory traffic by ~4×. Uses the same in-place repack format as
// xx_vec_dot_q3_k_q8_k_x8_dc — no changes to make_q3_k_repack_quant needed.
//
// Inner loop: load q3 weights + hmask ONCE per super-block, then VNNI dpbusd
// against 4 q8 activation streams. Scale decode done once and shared.
//
// Register budget (AVX-512, 32 zmm):
//   4 × acc (zmm) + 4 × sumi (zmm) + 4 × bias (zmm) + 1 × q3bytes (zmm) +
//   m3 + m4 + zero512 + temps ≈ 16 zmm. Fits comfortably.
//
void
xx_vec_dot_q3_k_q8_k_x8 ( // batch tiled Q3_K CP
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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:xx_vec_dot_q3_k_q8_k_x8_cp=" __FUNCTION__)
#endif

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

    block_q8_K_repack * y_base = (block_q8_K_repack *)vy;
    float * s_base = s;

    const uint64_t ncols4 = ncols & ~3ULL;
    uint64_t l = 0;

    //
    // Phase 1: Process 4 activation columns at a time.
    //
    // For each weight row + super-block, load weights and hmask once,
    // then dpbusd against 4 activation columns simultaneously.
    //

    for (; l < ncols4; l += 4) {

        block_q8_K_repack * y0 = y_base + (l + 0) * nb;
        block_q8_K_repack * y1 = y_base + (l + 1) * nb;
        block_q8_K_repack * y2 = y_base + (l + 2) * nb;
        block_q8_K_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        block_q3_K_repack * x = (block_q3_K_repack *)vx;

        for (uint64_t k = 0; k < nrows; k += 1) {

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                //
                // SHARED: extract weight scale and decode 6-bit scales.
                // Done once per super-block, reused across all 4 columns.
                //

                const float x_d = GGML_FP16_TO_FP32(x[i].d);

                const uint8_t * q3 = x[i].qs;

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
                // SHARED: Load the q3 low-bits once.
                //

                __m512i q3bytes = _mm512_loadu_si512((const __m512i*)q3);

                //
                // Per-column accumulators.
                //

                __m512i sumi0 = _mm512_setzero_si512();
                __m512i sumi1 = _mm512_setzero_si512();
                __m512i sumi2 = _mm512_setzero_si512();
                __m512i sumi3 = _mm512_setzero_si512();

                __m512i bias0 = _mm512_setzero_si512();
                __m512i bias1 = _mm512_setzero_si512();
                __m512i bias2 = _mm512_setzero_si512();
                __m512i bias3 = _mm512_setzero_si512();

                for (uint64_t j = 0; j < QK_K / 64; j += 1) {

                    //
                    // SHARED: isolate low 2-bits of q3 values.
                    //

                    const __m512i q3lv = _mm512_and_si512(q3bytes, m3);
                    q3bytes = _mm512_srli_epi16(q3bytes, 2);

                    //
                    // SHARED: compute high bit mask and blend.
                    //

                    uint64_t high_bits = *((uint64_t *)x[i].hmask + j);
                    const __mmask64 hmask = _cvtu64_mask64(high_bits);
                    const __m512i q3hv = _mm512_mask_blend_epi8(hmask, m4, zero512);

                    //
                    // Per-column: load q8 and dpbusd against shared q3 low and high.
                    //

                    const __m512i q8v0 = _mm512_loadu_si512(y0[i].qs + (j * 64));
                    sumi0 = _mm512_dpbusd_epi32(sumi0, q3lv, q8v0);
                    bias0 = _mm512_dpbusd_epi32(bias0, q3hv, q8v0);

                    const __m512i q8v1 = _mm512_loadu_si512(y1[i].qs + (j * 64));
                    sumi1 = _mm512_dpbusd_epi32(sumi1, q3lv, q8v1);
                    bias1 = _mm512_dpbusd_epi32(bias1, q3hv, q8v1);

                    const __m512i q8v2 = _mm512_loadu_si512(y2[i].qs + (j * 64));
                    sumi2 = _mm512_dpbusd_epi32(sumi2, q3lv, q8v2);
                    bias2 = _mm512_dpbusd_epi32(bias2, q3hv, q8v2);

                    const __m512i q8v3 = _mm512_loadu_si512(y3[i].qs + (j * 64));
                    sumi3 = _mm512_dpbusd_epi32(sumi3, q3lv, q8v3);
                    bias3 = _mm512_dpbusd_epi32(bias3, q3hv, q8v3);
                }

                //
                // Per-column: subtract bias, multiply by scales, FMA with d.
                //

                sumi0 = _mm512_mullo_epi32(_mm512_sub_epi32(sumi0, bias0), scales);
                sumi1 = _mm512_mullo_epi32(_mm512_sub_epi32(sumi1, bias1), scales);
                sumi2 = _mm512_mullo_epi32(_mm512_sub_epi32(sumi2, bias2), scales);
                sumi3 = _mm512_mullo_epi32(_mm512_sub_epi32(sumi3, bias3), scales);

                const float d0 = y0[i].d * x_d;
                const float d1 = y1[i].d * x_d;
                const float d2 = y2[i].d * x_d;
                const float d3 = y3[i].d * x_d;

                acc0 = _mm512_fmadd_ps(_mm512_set1_ps(d0), _mm512_cvtepi32_ps(sumi0), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_set1_ps(d1), _mm512_cvtepi32_ps(sumi1), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_set1_ps(d2), _mm512_cvtepi32_ps(sumi2), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_set1_ps(d3), _mm512_cvtepi32_ps(sumi3), acc3);
            }

            //
            // Reduce and store results for each of the 4 columns.
            //

#define REDUCE_Q3_CP(acc, dst) \
            do { \
                const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc), \
                                                  _mm512_extractf32x8_ps(acc, 1)); \
                __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res), \
                                        _mm256_extractf128_ps(res, 1)); \
                const __m128 t1 = _mm_hadd_ps(t0, t0); \
                dst[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1)); \
            } while (0)

            REDUCE_Q3_CP(acc0, s0);
            REDUCE_Q3_CP(acc1, s1);
            REDUCE_Q3_CP(acc2, s2);
            REDUCE_Q3_CP(acc3, s3);

#undef REDUCE_Q3_CP

            x += nb;
        }
    }

    //
    // Phase 2: Process remaining columns one at a time (0-3 columns).
    // Falls back to the original single-column loop.
    //

    block_q8_K_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        block_q3_K_repack * x = (block_q3_K_repack *)vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                const float d = y_rem[i].d * GGML_FP16_TO_FP32(x[i].d);

                const uint8_t * q3 = x[i].qs;
                const int8_t * q8 = y_rem[i].qs;

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

                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();

                __m512i q3bytes = _mm512_loadu_si512((const __m512i*)q3);

                for (uint64_t j = 0; j < QK_K / 64; j += 1) {

                    const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));

                    const __m512i q3lv = _mm512_and_si512(q3bytes, m3);
                    q3bytes = _mm512_srli_epi16(q3bytes, 2);

                    sumi = _mm512_dpbusd_epi32(sumi, q3lv, q8v);

                    uint64_t high_bits = *((uint64_t *)x[i].hmask + j);
                    const __mmask64 hmask = _cvtu64_mask64(high_bits);
                    const __m512i q3hv = _mm512_mask_blend_epi8(hmask, m4, zero512);

                    bias = _mm512_dpbusd_epi32(bias, q3hv, q8v);
                }

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

void
xx_vec_dot_q4_k_q8_k_x8_dc (
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

//
// Batch-optimized Q4_K × Q8_K vec_dot for prompt prefill (ncols >= 4).
//
// Key optimization: processes 4 activation columns per weight load, reducing
// weight memory traffic by ~4×. Uses the same in-place repack format as
// xx_vec_dot_q4_k_q8_k_x8 — no changes to make_q4_k_repack_quant needed.
//
// Inner loop: load q4 weights ONCE per super-block, then VNNI dpbusd against
// 4 q8 activation streams. Scales and mins extracted once and shared.
//
// Register budget (AVX-512, 32 zmm):
//   4 × acc (zmm)  + 4 × sumi (zmm)  + 1 × scale (zmm) +
//   1 × q4v (zmm)  + 1 × m4 (zmm)    + temps ≈ 12 zmm
//   4 × mins_acc (xmm, bottom of zmm)
// Fits comfortably with headroom for compiler temporaries.
//
/* 
Compare & Contrast: Q4_K Vec Dot Implementations

1. Data Layout

┌─────────────────┬────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│                 │ b612 block_q4_K_repack                                         │ x86 block_q4_Kx8                                 │
├─────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ Structure       │ typedef block_q4_K — same 144-byte block                       │ 8 blocks fused: d[8], dmin[8], scales[96],       │
│                 │                                                                │ qs[1024]                                         │
├─────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ Storage         │ Zero — in-place repack, same size                              │ 1x — 8 columns packed together (1152 bytes vs    │
│ overhead        │                                                                │ 8×144)                                           │
├─────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ Repack          │ Rearranges nibble order within existing qs[] for efficient     │ Interleaves 8 full Q4_K blocks column-wise into  │
│                 │ 64-element unpacking                                           │ a single struct                                  │
└─────────────────┴────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────┘

2. Inner Loop — Core Dot Product

b612 (xx_vec_dot_q4_k_q8_k_x8) — per super-block, 4 iterations:

 Load 256-bit q4 → split low/high nibbles → 512-bit
 Load 512-bit q8
 _mm512_dpbusd_epi32  ← 1 VNNI instruction

4 dpbusd per super-block. Total: ~4 instructions in the hot loop.

x86/repack (ggml_gemm_q4_K_8x8_q8_K) — per sub-block:

 Load 16×256-bit RHS (8 columns × 2 groups)
 Shuffle/blend/permute into 2 patterns (sp1, sp2)
 Load 16×256-bit LHS (4 rows × 2 halves)
 32 × _mm512_maddubs_epi16  ← 2-step: maddubs→int16
 16 × _mm512_add_epi16      ← then madd→int32
 8  × _mm512_madd_epi16     ← scale multiply

~56 compute instructions per sub-block + massive shuffle overhead.

3. Parallelism Strategy

┌───────────────────────┬──────────────────────────────────────┬───────────────────────────────────────────────────┐
│                       │ b612                                 │ x86 GEMM                                          │
├───────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────┤
│ Output tile           │ 1 row × 1 col per inner loop         │ 16 rows × 16 cols per outer iteration             │
├───────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────┤
│ Weight reuse          │ None — each row re-reads all weights │ Each weight load serves 4 rows                    │
├───────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────┤
│ Register pressure     │ ~10 live 512-bit regs                │ 60+ live 512-bit regs (massive spilling risk)     │
├───────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────┤
│ Instruction           │ dpbusd (VNNI) — 1 op                 │ maddubs + madd — 2 ops for same work              │
└───────────────────────┴──────────────────────────────────────┴───────────────────────────────────────────────────┘

4. Scale/Min Handling

b612: Extracts 8 scales into __m512i, multiplies accumulated sumi by scale vector in a single _mm512_mullo_epi32. Mins via 128-bit 
_mm_madd_epi16.

x86 GEMM: Extracts scales from 4 interleaved blocks via memcpy + bitmask decode (kmask1/2/3), builds per-column scale vectors, applies
via _mm512_madd_epi16. Much more decode overhead per sub-block.

5. Which Is Faster?

For token generation (batch=1): b612 wins. Why:

 - VNNI dpbusd does in 1 µop what maddubs + madd takes 2-3 µops
 - Tiny inner loop (4 iterations) fits entirely in µop cache
 - Minimal register pressure → no spills
 - In-place repack → no extra memory traffic
 - At batch=1, both are memory-bandwidth-bound; the simpler code has less decode/issue overhead

For prompt processing (batch≥4): x86 GEMM wins. Why:

 - Weight data reuse — loads weight once, multiplies against 4-16 rows. At batch=1 this doesn't help, but at batch≥4 it cuts memory 
traffic by 4-16×
 - 16×16 output tile means 256 FMAs per weight load vs 1 in b612
 - The 8-interleaved layout means one cache line serves 8 columns simultaneously
 - The massive instruction overhead is amortized over 256 output elements

Bottom line: b612's approach is a vec_dot (MV — matrix-vector) optimized for latency. The x86 approach is a GEMM (MM — matrix-matrix)
optimized for throughput. In LLM inference, token generation is batch=1 (b612 wins), while prompt prefill is batch=N (x86 GEMM wins).
The best system uses both: GEMM for prefill, vec_dot for generation.

======================= Implementation Details =======================

xx_vec_dot_q4_k_q8_k_x8_cp — Batch-Tiled Q4_K Vec Dot

Files modified:

 - ggml/include/ggml-cpu-repack.h — added declaration (lines 98-110)
 - ggml/src/ggml-cpu/b612/ggml-cpu-repack.c — added implementation (lines 1584-1867)

Design: Drop-in replacement with identical signature. Two-phase approach:

┌─────────────────────────┬─────────────┬──────────────────────────────┐
│ Phase                   │ Columns     │ Weight loads per super-block │
├─────────────────────────┼─────────────┼──────────────────────────────┤
│ Phase 1 (batch)         │ 4 at a time │ 1 (shared across 4 cols)     │
├─────────────────────────┼─────────────┼──────────────────────────────┤
│ Phase 2 (remainder)     │ 1 at a time │ 1 (original path)            │
└─────────────────────────┴─────────────┴──────────────────────────────┘

How it works: For each weight row + super-block:

 1. Load & decode weight q4 data + 6-bit scales/mins once
 2. dpbusd the same q4 vector against 4 different q8 activation columns
 3. Apply shared scale vector to all 4 integer accumulators
 4. Per-column: FMA with column-specific d values, accumulate mins

Why faster for batch ≥ 4:

 - 4× reduction in weight memory traffic (the bottleneck)
 - Scale/min extraction amortized across 4 columns
 - ~12 zmm registers used — no spill pressure
 - No repack format change needed — same block_q4_K_repack
*/
void
xx_vec_dot_q4_k_q8_k_x8 ( // Batch tiled Q4_K CP
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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:xx_vec_dot_q4_k_q8_k_x8_cp=" __FUNCTION__)
#endif

    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t kmask4 = 0xc0c0c0c0u;

    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m128i zero128 = _mm_setzero_si128();

    uint64_t utmp[2];

    block_q8_K_repack * y_base = (block_q8_K_repack *)vy;
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

        block_q8_K_repack * y0 = y_base + (l + 0) * nb;
        block_q8_K_repack * y1 = y_base + (l + 1) * nb;
        block_q8_K_repack * y2 = y_base + (l + 2) * nb;
        block_q8_K_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        block_q4_K_repack * x = (block_q4_K_repack *)vx;

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

                const float x_d    = GGML_FP16_TO_FP32(x[i].d);
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

                    // Load weight nibbles once
                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j * 32)));

                    __m512i q4v = _mm512_castsi256_si512(q4bits);
                    q4v = _mm512_inserti32x8(q4v, _mm256_srli_epi16(q4bits, 4), 1);
                    q4v = _mm512_and_si512(q4v, m4);

                    // Dot product against 4 activation columns
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
            // Horizontal reduction: 512 → 256 → 128, subtract mins, hadd to scalar.
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

    block_q8_K_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        block_q4_K_repack * x = (block_q4_K_repack *)vx;

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

void
xx_vec_dot_q6_k_q8_k_x8_dc (
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

//
// Batch-optimized Q6_K × Q8_K vec_dot for prompt prefill (ncols >= 4).
//
// Key optimization: processes 4 activation columns per weight load, reducing
// weight memory traffic by ~4×. Uses the same in-place repack format as
// _dc — no changes to make_q6_k_repack_quant needed.
//
// Inner loop: load q6 ql/qh ONCE per super-block, unpack 6-bit values once,
// then VNNI dpbusd against 4 q8 activation streams. Scales loaded once.
//
// Q6_K benefits most from batch-tiling because it has the largest weight
// blocks (210 bytes) — the absolute bandwidth savings are the highest.
//
// Register budget (AVX-512, 32 zmm):
//   4 × acc (zmm) + 4 × sumi (zmm) + 4 × bias (zmm) + 1 × q2bits (zmm) +
//   1 × q4bits (zmm) + m4 + m2 + m32s + scales + temps ≈ 17 zmm
//

/*
Analysis: Q6_K vec_dot — Prefill vs Token Gen

Data Layout (block_q6_K_repack = block_q6_K, 210 bytes/super-block):

 - ql[128] — low 4 bits (4 bits/weight, nibble-packed)
 - qh[64] — high 2 bits (2 bits/weight, packed 4/byte)
 - scales[16] — 16 × 8-bit scales (simple, no packing!)
 - d — 1× fp16 scale (no mins — Q6_K is scale-only like Q3_K)

Vec-dot structure (178 lines):

 for each column:
   for each row:
     for each super-block:
       ── scales = loadu_128 → cvtepi8_epi32 → 512-bit [trivial!]
       ── q2bits = load 512-bit(qh); rol_epi32(q2bits, 4) [pre-shift high bits]
       ── inner loop (2 iterations per super-block):
            q4bits = load 512-bit(ql)                     [low 4-bit nibbles]
            q6_0 = OR(AND(q4bits,0xf), AND(q2bits,0x30))  [merge low+high → 6-bit]
            q2bits = ror_epi32(q2bits, 2)                  [advance high bits]
            sumi = dpbusd(sumi, q6_0, q8_0)               [dot product]
            bias = dpbusd(bias, 32, q8_0)                  [bias = 32 × sum(q8)]
            q4bits2 = (q4bits >> 4) & 0xf                  [upper nibble]
            q6_1 = OR(q4bits2, AND(q2bits,0x30))           [merge]
            q2bits = ror_epi32(q2bits, 2)
            sumi = dpbusd(sumi, q6_1, q8_1)
            bias = dpbusd(bias, 32, q8_1)
       ── sumi = sumi - bias; mullo(sumi, scales)
       ── fmadd to acc
     ── reduce 512→128→scalar

Key characteristics:

 - 16 scales, 8-bit each — simplest scale decode of any K-quant (_mm_loadu_si128 + cvtepi8_epi32)
 - 4 dpbusd per inner iteration × 2 iterations = 8 dpbusd per super-block (same as Q3_K)
 - 6-bit unpacking: 4-bit low from ql + 2-bit high from qh, merged via OR
 - q2bits register rotated (ror_epi32) to advance through the 4 sets of 2-bit high values
 - Bias: constant 32 × q8 sums (since values are unsigned 0–63, centered at 32)
 - Largest weight block: 210 bytes

Register budget: ~10 zmm (1 acc + 1 sumi + 1 bias + 1 q2bits + 1 q4bits + q6 temps + m4 + m2 + m32s + scales)

--------------------------------------------------------------------------------------------------------------------------------------

Comparative Table

┌──────────────────────────────┬───────────────────────────────┐
│ Metric                       │ Q6_K                          │
├──────────────────────────────┼───────────────────────────────┤
│ Bits per weight              │ 6.5625                        │
├──────────────────────────────┼───────────────────────────────┤
│ Block size (bytes)           │ 210                           │
├──────────────────────────────┼───────────────────────────────┤
│ Scales count                 │ 16 (8-bit direct)             │
├──────────────────────────────┼───────────────────────────────┤
│ Scale decode cost            │ Trivial (load+cvt)            │
├──────────────────────────────┼───────────────────────────────┤
│ dpbusd per super-block       │ 8                             │
├──────────────────────────────┼───────────────────────────────┤
│ Has mins/dmin?               │ No                            │
├──────────────────────────────┼───────────────────────────────┤
│ Bit unpacking method         │ 4-bit nibble + 2-bit OR merge │
├──────────────────────────────┼───────────────────────────────┤
│ Weight loads per super-block │ 2×512b (ql) + 1×512b (qh)     │
├──────────────────────────────┼───────────────────────────────┤
│ Weight bytes loaded          │ 128 + 64 = 192                │
├──────────────────────────────┼───────────────────────────────┤
│ Activation bytes loaded      │ 4×512b = 256                  │
├──────────────────────────────┼───────────────────────────────┤
│ zmm registers used           │ ~10 of 32                     │
└──────────────────────────────┴───────────────────────────────┘

--------------------------------------------------------------------------------------------------------------------------------------

Performance: Prefill vs Token Generation

Token generation (batch=1): Both routines are well-suited. The 1-row × 1-col inner loop means:

 - Weight data streams sequentially through each super-block
 - Q3_K is faster here because it reads 96 bytes of weight per super-block vs Q6_K's 192 bytes. With batch=1, the bottleneck is memory 
bandwidth (weights >> activations), so smaller quant = faster.
 - Both use identical computation: 8 dpbusd + scale multiply. Compute cost is equal; memory cost favors Q3_K by ~2×.

Prompt prefill (batch ≥ 4): This is where both routines lose efficiency. The outer loop structure is:

 for each column:        ← sequential over batch columns
   for each row:         ← sequential over output rows
     for each block: ... ← weight data loaded here

Each column re-reads the entire weight matrix. For batch=B, weight data is read B times. This is the same bottleneck we fixed in Q4_K
and Q2_K with the _cp variants.

Q3_K prefill penalty: Weight matrix re-read B× despite only 96 bytes/block. Total weight traffic = B × 96 × nb × nrows. The fix would
save (B-1)/B ≈ 75% of weight bandwidth at batch=4.

Q6_K prefill penalty: Much worse — 192 bytes/block loaded B times. Total weight traffic = B × 192 × nb × nrows. Since Q6_K has ~2× the
weight data of Q3_K, the absolute waste from re-reading is 2× larger.

Both are strong candidates for the _cp batch-tiling optimization, for the same reason as Q4_K/Q2_K:

 - Plenty of register headroom (10–11 of 32 zmm used → room for 4× accumulators)
 - No repack format change needed
 - Weight loads dominate the inner loop and can be shared across 4 columns

Relative benefit ranking:

┌───────┬────────────────────┬──────────────────────────┬──────────────────┐
│ Quant │ Weight bytes/block │ Prefill batch=4 waste    │ Benefit of _cp   │
├───────┼────────────────────┼──────────────────────────┼──────────────────┤
│ Q6_K  │ 192                │ Highest (largest blocks) │ Highest          │
├───────┼────────────────────┼──────────────────────────┼──────────────────┤
│ Q4_K  │ 128                │ High                     │ High             │
├───────┼────────────────────┼──────────────────────────┼──────────────────┤
│ Q3_K  │ 96                 │ Medium                   │ Medium           │
├───────┼────────────────────┼──────────────────────────┼──────────────────┤
│ Q2_K  │ 64                 │ Lowest (smallest blocks) │ Moderate         │
└───────┴────────────────────┴──────────────────────────┴──────────────────┘

Q6_K would benefit most from the _cp treatment because it has the most weight data to reload. Q3_K benefits less in absolute terms but
the relative speedup (4× weight bandwidth reduction) is the same for all.

*/
void
xx_vec_dot_q6_k_q8_k_x8 ( // Batch tiled Q6_K CP
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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:_cp=" __FUNCTION__)
#endif

    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m512i m2 = _mm512_set1_epi8(0x30);
    const __m512i m32s = _mm512_set1_epi8(32);

    block_q8_K_repack * y_base = (block_q8_K_repack *)vy;
    float * s_base = s;

    const uint64_t ncols4 = ncols & ~3ULL;
    uint64_t l = 0;

    //
    // Phase 1: Process 4 activation columns at a time.
    //
    // For each weight row + super-block, load ql/qh once, unpack 6-bit values,
    // then dpbusd against 4 activation columns simultaneously.
    //

    for (; l < ncols4; l += 4) {

        block_q8_K_repack * y0 = y_base + (l + 0) * nb;
        block_q8_K_repack * y1 = y_base + (l + 1) * nb;
        block_q8_K_repack * y2 = y_base + (l + 2) * nb;
        block_q8_K_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        block_q6_K_repack * x = (block_q6_K_repack *)vx;

        for (uint64_t k = 0; k < nrows; k += 1) {

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                //
                // SHARED: extract weight scale and load 8-bit scales.
                // Done once per super-block, reused across all 4 columns.
                //

                const float x_d = GGML_FP16_TO_FP32(x[i].d);

                const uint8_t * q4 = x[i].ql;
                const uint8_t * q2 = x[i].qh;

                const __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
                const __m512i scales = _mm512_cvtepi8_epi32(scales8);

                //
                // Per-column accumulators.
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
                // SHARED: Load the high 2-bits and pre-rotate.
                //

                __m512i q2bits = _mm512_loadu_si512((const __m512i*)q2);
                q2bits = _mm512_rol_epi32(q2bits, 4);

                for (uint64_t j = 0; j < QK_K / 128; ++j) {

                    //
                    // SHARED: Load 64 nibble packed 4-bit values.
                    //

                    __m512i q4bits = _mm512_loadu_si512((const __m512i*)(q4 + (j * 64)));

                    //
                    // SHARED: Unpack first set of 6-bit values (low nibble + high 2-bits).
                    //

                    const __m512i q4bits1 = _mm512_and_si512(q4bits, m4);
                    const __m512i q2bits1 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_0 = _mm512_or_si512(q2bits1, q4bits1);

                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    //
                    // Per-column: dpbusd for first 64 values of this 128-value chunk.
                    //

                    const __m512i q8v0_0 = _mm512_loadu_si512(y0[i].qs + (j * 128) + 0);
                    sumi0 = _mm512_dpbusd_epi32(sumi0, q6_0, q8v0_0);
                    bias0 = _mm512_dpbusd_epi32(bias0, m32s, q8v0_0);

                    const __m512i q8v1_0 = _mm512_loadu_si512(y1[i].qs + (j * 128) + 0);
                    sumi1 = _mm512_dpbusd_epi32(sumi1, q6_0, q8v1_0);
                    bias1 = _mm512_dpbusd_epi32(bias1, m32s, q8v1_0);

                    const __m512i q8v2_0 = _mm512_loadu_si512(y2[i].qs + (j * 128) + 0);
                    sumi2 = _mm512_dpbusd_epi32(sumi2, q6_0, q8v2_0);
                    bias2 = _mm512_dpbusd_epi32(bias2, m32s, q8v2_0);

                    const __m512i q8v3_0 = _mm512_loadu_si512(y3[i].qs + (j * 128) + 0);
                    sumi3 = _mm512_dpbusd_epi32(sumi3, q6_0, q8v3_0);
                    bias3 = _mm512_dpbusd_epi32(bias3, m32s, q8v3_0);

                    //
                    // SHARED: Unpack second set of 6-bit values (high nibble + next high 2-bits).
                    //

                    __m512i q4bits2 = _mm512_srli_epi16(q4bits, 4);
                    q4bits2 = _mm512_and_si512(q4bits2, m4);
                    const __m512i q2bits2 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_1 = _mm512_or_si512(q2bits2, q4bits2);

                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    //
                    // Per-column: dpbusd for second 64 values of this 128-value chunk.
                    //

                    const __m512i q8v0_1 = _mm512_loadu_si512(y0[i].qs + (j * 128) + 64);
                    sumi0 = _mm512_dpbusd_epi32(sumi0, q6_1, q8v0_1);
                    bias0 = _mm512_dpbusd_epi32(bias0, m32s, q8v0_1);

                    const __m512i q8v1_1 = _mm512_loadu_si512(y1[i].qs + (j * 128) + 64);
                    sumi1 = _mm512_dpbusd_epi32(sumi1, q6_1, q8v1_1);
                    bias1 = _mm512_dpbusd_epi32(bias1, m32s, q8v1_1);

                    const __m512i q8v2_1 = _mm512_loadu_si512(y2[i].qs + (j * 128) + 64);
                    sumi2 = _mm512_dpbusd_epi32(sumi2, q6_1, q8v2_1);
                    bias2 = _mm512_dpbusd_epi32(bias2, m32s, q8v2_1);

                    const __m512i q8v3_1 = _mm512_loadu_si512(y3[i].qs + (j * 128) + 64);
                    sumi3 = _mm512_dpbusd_epi32(sumi3, q6_1, q8v3_1);
                    bias3 = _mm512_dpbusd_epi32(bias3, m32s, q8v3_1);
                }

                //
                // Per-column: subtract bias, multiply by scales, FMA with d.
                //

                sumi0 = _mm512_mullo_epi32(_mm512_sub_epi32(sumi0, bias0), scales);
                sumi1 = _mm512_mullo_epi32(_mm512_sub_epi32(sumi1, bias1), scales);
                sumi2 = _mm512_mullo_epi32(_mm512_sub_epi32(sumi2, bias2), scales);
                sumi3 = _mm512_mullo_epi32(_mm512_sub_epi32(sumi3, bias3), scales);

                const float d0 = y0[i].d * x_d;
                const float d1 = y1[i].d * x_d;
                const float d2 = y2[i].d * x_d;
                const float d3 = y3[i].d * x_d;

                acc0 = _mm512_fmadd_ps(_mm512_set1_ps(d0), _mm512_cvtepi32_ps(sumi0), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_set1_ps(d1), _mm512_cvtepi32_ps(sumi1), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_set1_ps(d2), _mm512_cvtepi32_ps(sumi2), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_set1_ps(d3), _mm512_cvtepi32_ps(sumi3), acc3);
            }

            //
            // Reduce and store results for each of the 4 columns.
            //

#define REDUCE_Q6_CP(acc, dst) \
            do { \
                const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc), \
                                                  _mm512_extractf32x8_ps(acc, 1)); \
                __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res), \
                                        _mm256_extractf128_ps(res, 1)); \
                const __m128 t1 = _mm_hadd_ps(t0, t0); \
                dst[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1)); \
            } while (0)

            REDUCE_Q6_CP(acc0, s0);
            REDUCE_Q6_CP(acc1, s1);
            REDUCE_Q6_CP(acc2, s2);
            REDUCE_Q6_CP(acc3, s3);

#undef REDUCE_Q6_CP

            x += nb;
        }
    }

    //
    // Phase 2: Process remaining columns one at a time (0-3 columns).
    // Falls back to the original single-column loop.
    //

    block_q8_K_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        block_q6_K_repack * x = (block_q6_K_repack *)vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                const float d = y_rem[i].d * GGML_FP16_TO_FP32(x[i].d);

                const uint8_t * q4 = x[i].ql;
                const uint8_t * q2 = x[i].qh;
                const int8_t * q8 = y_rem[i].qs;

                const __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
                const __m512i scales = _mm512_cvtepi8_epi32(scales8);

                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();

                __m512i q2bits = _mm512_loadu_si512((const __m512i*)q2);
                q2bits = _mm512_rol_epi32(q2bits, 4);

                for (uint64_t j = 0; j < QK_K / 128; ++j) {

                    __m512i q4bits = _mm512_loadu_si512((const __m512i*)(q4 + (j * 64)));

                    const __m512i q4bits1 = _mm512_and_si512(q4bits, m4);
                    const __m512i q2bits1 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_0 = _mm512_or_si512(q2bits1, q4bits1);

                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    const __m512i q8_0 = _mm512_loadu_si512(q8 + (j * 128) + 0);
                    sumi = _mm512_dpbusd_epi32(sumi, q6_0, q8_0);
                    bias = _mm512_dpbusd_epi32(bias, m32s, q8_0);

                    __m512i q4bits2 = _mm512_srli_epi16(q4bits, 4);
                    q4bits2 = _mm512_and_si512(q4bits2, m4);
                    const __m512i q2bits2 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_1 = _mm512_or_si512(q2bits2, q4bits2);

                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    const __m512i q8_1 = _mm512_loadu_si512(q8 + (j * 128) + 64);
                    sumi = _mm512_dpbusd_epi32(sumi, q6_1, q8_1);
                    bias = _mm512_dpbusd_epi32(bias, m32s, q8_1);
                }

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

/*
Q8_0 × Q8_0 Analysis: xx_vec_dot_q8_0_q8_0_x8

Structure

block_q8_0_repack (header line 21-24):

 - d[8]: 8 × fp16 scales (one per sub-block) — same vector-scale design as Q4_0
 - qs[QK_K]: 256 × int8 quants, interleaved from 8 original block_q8_0
 - Size: 16 + 256 = 272 bytes — the largest of all repack blocks

Inner loop (lines 3734-3760, 4 iterations per super-block)

Each iteration:

 1. SHARED (weight): Load 512-bit qx from x[i].qs[] — 1 zmm load
 2. PER-COL (activation): Load 512-bit qy from y[i].qs[] — 1 zmm load
 3. abs_epi8(qx) → unsigned ax
 4. movepi8_mask(qx) → __mmask64 negative mask
 5. mask_sub_epi8(qy, mask, 0, qy) → signed-adjusted sy
 6. dpbusd(sumi, ax, sy) — 1 dpbusd per iteration (4 total per super-block)

Key difference from other quants

Q8_0 uses signed × signed dot product, but AVX-512 VNNI dpbusd requires unsigned × signed. The workaround:

 - Take abs(qx) → unsigned
 - Negate qy where qx was negative → sy
 - dpbusd(ax, sy) = effectively signed × signed

This adds 3 extra instructions per inner iteration (abs_epi8, movepi8_mask, mask_sub_epi8) compared to Q4_0's inner loop which is just 
load + nibble-unpack + dpbusd.

No bias subtraction

Unlike Q4_0 (unsigned 0-15, needs bias=8 subtraction), Q8_0's sign-correction approach produces the correct result directly — no bias
accumulator needed.

Scale handling

Same as Q4_0: d[8] is a vector of 8 fp16. Weight x[i].d[8] → fp32, activation y[i].d[8] → fp32, element-wise multiply → 8-element scale
broadcast to zmm via insertf32x8.

Batch-tiling potential

┌─────────────────────────┬──────────────────────────────────────────────────────┐
│ Aspect                  │ Assessment                                           │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ Weight bytes/block      │ 272 (largest of all quants)                          │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ Shared weight load/iter │ 1 × 512-bit qx + abs + mask (shared across 4 cols)   │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ Per-col work/iter       │ 1 × 512-bit qy load + mask_sub + dpbusd              │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ Bias accumulators       │ None needed (saves 4 zmm in batch version)           │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ Scale: shared part      │ weight d[8] fp16→fp32 (1 load + cvt)                 │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ Scale: per-col part     │ activation d[8] fp16→fp32 + element-wise mul         │
└─────────────────────────┴──────────────────────────────────────────────────────┘

Register budget (4-col batch):

 - 4 × acc (zmm) + 4 × sumi (zmm) = 8 accumulators
 - 1 × qx (shared) + 1 × ax (shared) + 1 × mask64 = shared weight
 - 4 × qy + 4 × sy = 8 per-column (but qy/sy can alias) → ~4 zmm
 - offset/m4 not needed (no bias, no nibble mask)
 - Total: ~14-16 zmm — well within 32

Estimated savings per super-block at batch=4:

┌──────────────────┬─────────────────┬──────────────────────────────────────┐
│ Data             │ Single-col (×4) │ Batch-tiled (×1 shared + ×4 per-col) │
├──────────────────┼─────────────────┼──────────────────────────────────────┤
│ Weight qs loads  │ 4×4 = 16 zmm    │ 1×4 = 4 zmm (shared)                 │
├──────────────────┼─────────────────┼──────────────────────────────────────┤
│ Weight d loads   │ 4×1 = 4         │ 1 (shared)                           │
├──────────────────┼─────────────────┼──────────────────────────────────────┤
│ abs+mask ops     │ 4×4 = 16        │ 1×4 = 4 (shared)                     │
├──────────────────┼─────────────────┼──────────────────────────────────────┤
│ Activation loads │ 4×4 = 16 zmm    │ 4×4 = 16 zmm (same)                  │
├──────────────────┼─────────────────┼──────────────────────────────────────┤
│ dpbusd           │ 4×4 = 16        │ 4×4 = 16 (same)                      │
└──────────────────┴─────────────────┴──────────────────────────────────────┘

Weight bandwidth saved: 272 bytes × 3 = 816 bytes per super-block — the highest absolute savings of any quant.

Relative benefit ranking (final, all quants):

┌──────────┬────────────────────┬───────────────────────┬─────────────┐
│ Quant    │ Weight bytes/block │ Shared ops/iter       │ Benefit     │
├──────────┼────────────────────┼───────────────────────┼─────────────┤
│ Q8_0     │ 272                │ load + abs + mask     │ Highest     │
├──────────┼────────────────────┼───────────────────────┼─────────────┤
│ Q6_K     │ 210                │ ql+qh unpack          │ Very High   │
├──────────┼────────────────────┼───────────────────────┼─────────────┤
│ Q4_0     │ 144                │ nibble unpack         │ High        │
├──────────┼────────────────────┼───────────────────────┼─────────────┤
│ Q4_K     │ 144                │ nibble + scale decode │ High        │
├──────────┼────────────────────┼───────────────────────┼─────────────┤
│ Q3_K     │ 110                │ low bits + hmask      │ Medium      │
├──────────┼────────────────────┼───────────────────────┼─────────────┤
│ Q2_K     │ 84                 │ 2-bit shift           │ Moderate    │
└──────────┴────────────────────┴───────────────────────┴─────────────┘

Additional shared computation bonus

Beyond the weight loads, the sign-correction ops (abs_epi8 + movepi8_mask + mask_sub_epi8) only need the weight qx — they're fully
shareable across all 4 columns. In the single-col version these run 4×; batch-tiled reduces to 1×. This is unique to Q8_0 (other quants
don't have this extra shared compute).

Verdict

Q8_0 is the strongest candidate for batch tiling:

 1. Largest weight blocks (272 bytes) → most bandwidth saved
 2. Extra sign-correction ops are fully shareable → unique compute savings
 3. No bias accumulators → lower register pressure than Q4_0 or K-quants
 4. Simplest scale handling (same vector-d as Q4_0)
 5. ~14 zmm for 4-col batch → plenty of headroom

Expected speedup at batch=4: ~3.5-4× for the weight-bandwidth-bound portion, with additional ~20% from shared abs/mask compute. Overall
prompt prefill improvement likely higher than any other quant due to the combination of bandwidth + compute sharing.

*/
void
xx_vec_dot_q8_0_q8_0_x8_dc (
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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:xx_vec_dot_q8_0_q8_0_x8_dc=" __FUNCTION__)
#endif

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

#ifdef __clang__

                __m128i xd_raw = _mm_loadu_si128((__m128i*)x[i].d);
                __m128i yd_raw = _mm_loadu_si128((__m128i*)y[i].d);

                // Convert to single-precision floats
                __m256 xd_fp32 = _mm256_cvtph_ps(xd_raw);
                __m256 yd_fp32 = _mm256_cvtph_ps(yd_raw);

                // Multiply
                __m256 scale = _mm256_mul_ps(xd_fp32, yd_fp32);

                // Expand to __m512
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);

#else

                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m128h yd = _mm_loadu_ph(y[i].d);
                const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                                   _mm256_cvtph_ps(yd));

                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);

#endif // __clang__

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

//
// Batch-optimized Q8_0 × Q8_0 vec_dot for prompt prefill (ncols >= 4).
//
// Key optimization: processes 4 activation columns per weight load, reducing
// weight memory traffic by ~4×. Additional savings from shared sign-correction
// ops (abs_epi8 + movepi8_mask) which are weight-only and run once per 4 cols.
//
// Same repack format as xx_vec_dot_q8_0_q8_0_x8_dc — drop-in replacement.
// No bias accumulators needed (sign-correction produces exact result).
//
// Register budget (AVX-512, 32 zmm):
//   4 × acc (zmm) + 4 × sumi (zmm) +
//   1 × qx (shared) + 1 × ax (shared) + 4 × sy (can alias qy) +
//   1 × zero512 ≈ 14 zmm
//
void
xx_vec_dot_q8_0_q8_0_x8 ( // Batch tiled Q8_0
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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:xx_vec_dot_q8_0_q8_0_x8_cp=" __FUNCTION__)
#endif

    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;
    const uint64_t nrows = (uint32_t)nrc;

    const __m512i zero512 = _mm512_setzero_si512();

    block_q8_0_repack * y_base = (block_q8_0_repack *)vy;
    float * s_base = s;

    const uint64_t ncols4 = ncols & ~3ULL;
    uint64_t l = 0;

    //
    // Phase 1: Process 4 activation columns at a time.
    //
    // For each weight row + super-block, load weight qs once, compute
    // abs + negative mask once, then apply against 4 activation columns.
    //

    for (; l < ncols4; l += 4) {

        block_q8_0_repack * y0 = y_base + (l + 0) * nb;
        block_q8_0_repack * y1 = y_base + (l + 1) * nb;
        block_q8_0_repack * y2 = y_base + (l + 2) * nb;
        block_q8_0_repack * y3 = y_base + (l + 3) * nb;

        float * s0 = (float *)((char *)s_base + (l + 0) * nr_nb1);
        float * s1 = (float *)((char *)s_base + (l + 1) * nr_nb1);
        float * s2 = (float *)((char *)s_base + (l + 2) * nr_nb1);
        float * s3 = (float *)((char *)s_base + (l + 3) * nr_nb1);

        block_q8_0_repack * x = (block_q8_0_repack *)vx;

        for (uint64_t k = 0; k < nrows; k += 1) {

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                //
                // Per-column integer accumulators (no bias needed for Q8_0).
                //

                __m512i sumi0 = _mm512_setzero_si512();
                __m512i sumi1 = _mm512_setzero_si512();
                __m512i sumi2 = _mm512_setzero_si512();
                __m512i sumi3 = _mm512_setzero_si512();

                //
                // SHARED: Load weight data once per sub-block, compute abs and
                // negative mask, then dpbusd against 4 activation columns.
                //

                for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {

                    //
                    // SHARED: load weight qx, compute unsigned abs and sign mask.
                    //

                    const __m512i qx = _mm512_loadu_si512((const __m512i *)&x[i].qs[j * QK8_0 * 2]);
                    const __m512i ax = _mm512_abs_epi8(qx);
                    const __mmask64 is_negative_qx = _mm512_movepi8_mask(qx);

                    //
                    // Per-column: load qy, apply sign correction, dpbusd.
                    //

                    const __m512i qy0 = _mm512_loadu_si512((const __m512i *)&y0[i].qs[j * QK8_0 * 2]);
                    const __m512i sy0 = _mm512_mask_sub_epi8(qy0, is_negative_qx, zero512, qy0);
                    sumi0 = _mm512_dpbusd_epi32(sumi0, ax, sy0);

                    const __m512i qy1 = _mm512_loadu_si512((const __m512i *)&y1[i].qs[j * QK8_0 * 2]);
                    const __m512i sy1 = _mm512_mask_sub_epi8(qy1, is_negative_qx, zero512, qy1);
                    sumi1 = _mm512_dpbusd_epi32(sumi1, ax, sy1);

                    const __m512i qy2 = _mm512_loadu_si512((const __m512i *)&y2[i].qs[j * QK8_0 * 2]);
                    const __m512i sy2 = _mm512_mask_sub_epi8(qy2, is_negative_qx, zero512, qy2);
                    sumi2 = _mm512_dpbusd_epi32(sumi2, ax, sy2);

                    const __m512i qy3 = _mm512_loadu_si512((const __m512i *)&y3[i].qs[j * QK8_0 * 2]);
                    const __m512i sy3 = _mm512_mask_sub_epi8(qy3, is_negative_qx, zero512, qy3);
                    sumi3 = _mm512_dpbusd_epi32(sumi3, ax, sy3);
                }

                //
                // SHARED: Load weight d[8] (fp16 → fp32) once.
                //

#ifdef __clang__
                __m128i x_raw = _mm_loadu_si128((__m128i*)x[i].d);
                __m256 x_fp32 = _mm256_cvtph_ps(x_raw);
#else
                const __m128h xd = _mm_loadu_ph(x[i].d);
                __m256 x_fp32 = _mm256_cvtph_ps(xd);
#endif

                //
                // Per-column: build scale vector, convert sumi to float, FMA.
                //

#ifdef __clang__

#define Q80_COL_FMA(col, y_ptr, sumi_v, acc_v) \
                do { \
                    __m128i y_raw_##col = _mm_loadu_si128((__m128i*)(y_ptr)[i].d); \
                    __m256 y_fp32_##col = _mm256_cvtph_ps(y_raw_##col); \
                    __m256 scale_##col = _mm256_mul_ps(x_fp32, y_fp32_##col); \
                    __m512 d_##col = _mm512_castps256_ps512(scale_##col); \
                    d_##col = _mm512_insertf32x8(d_##col, scale_##col, 1); \
                    acc_v = _mm512_fmadd_ps(d_##col, _mm512_cvtepi32_ps(sumi_v), acc_v); \
                } while (0)

                Q80_COL_FMA(0, y0, sumi0, acc0);
                Q80_COL_FMA(1, y1, sumi1, acc1);
                Q80_COL_FMA(2, y2, sumi2, acc2);
                Q80_COL_FMA(3, y3, sumi3, acc3);

#undef Q80_COL_FMA

#else

#define Q80_COL_FMA(col, y_ptr, sumi_v, acc_v) \
                do { \
                    const __m128h yd_##col = _mm_loadu_ph((y_ptr)[i].d); \
                    const __m256 scale_##col = _mm256_mul_ps(x_fp32, \
                                                _mm256_cvtph_ps(yd_##col)); \
                    __m512 d_##col = _mm512_castps256_ps512(scale_##col); \
                    d_##col = _mm512_insertf32x8(d_##col, scale_##col, 1); \
                    acc_v = _mm512_fmadd_ps(d_##col, _mm512_cvtepi32_ps(sumi_v), acc_v); \
                } while (0)

                Q80_COL_FMA(0, y0, sumi0, acc0);
                Q80_COL_FMA(1, y1, sumi1, acc1);
                Q80_COL_FMA(2, y2, sumi2, acc2);
                Q80_COL_FMA(3, y3, sumi3, acc3);

#undef Q80_COL_FMA

#endif // __clang__
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

    block_q8_0_repack * y_rem = y_base + l * nb;
    float * s_rem = (float *)((char *)s_base + l * nr_nb1);

    for (; l < ncols; l += 1) {

        block_q8_0_repack * x = (block_q8_0_repack *)vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {

                __m512i sumi = _mm512_setzero_si512();

#ifdef __clang__
                __m128i x_raw = _mm_loadu_si128((__m128i*)x[i].d);
                __m128i y_raw = _mm_loadu_si128((__m128i*)y_rem[i].d);
                __m256 x_fp32 = _mm256_cvtph_ps(x_raw);
                __m256 y_fp32 = _mm256_cvtph_ps(y_raw);
                __m256 scale = _mm256_mul_ps(x_fp32, y_fp32);
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);
#else
                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m128h yd = _mm_loadu_ph(y_rem[i].d);
                const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                                    _mm256_cvtph_ps(yd));
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);
#endif

                for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {
                    const __m512i qx = _mm512_loadu_si512((const __m512i *)&x[i].qs[j * QK8_0 * 2]);
                    const __m512i qy = _mm512_loadu_si512((const __m512i *)&y_rem[i].qs[j * QK8_0 * 2]);

                    const __m512i ax = _mm512_abs_epi8(qx);
                    const __mmask64 is_negative_qx = _mm512_movepi8_mask(qx);
                    const __m512i sy = _mm512_mask_sub_epi8(qy, is_negative_qx, zero512, qy);

                    sumi = _mm512_dpbusd_epi32(sumi, ax, sy);
                }

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

void
quantize_row_q4_0_x8 (
    const float * x,
    block_q4_0 * y,
    uint32_t vec_size
    )
{
#ifndef __clang__
#pragma comment(linker, "/EXPORT:quantize_row_q4_0_x8=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:quantize_row_q4_k_x8=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:quantize_row_q6_k_x8=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:quantize_row_q4_k_q8_k_x8=" __FUNCTION__)
#endif

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
#ifndef __clang__
#pragma comment(linker, "/EXPORT:quantize_row_q8_0_x8=" __FUNCTION__)
#endif

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

    int ith = 0;
    if (params != NULL) {
        ith = params->ith;
    }

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
        size_t tensor_size = ggml_nbytes(tensor);
        char * duplicate_data = (char *) malloc(tensor_size);
        if (duplicate_data == NULL) {
            if (ith == 0) {
                mul_mat_repack_failed_count += 1;
            }
            return type;
        }

        memcpy(duplicate_data, tensor->data, tensor_size);
        tensor->data = duplicate_data;

        if (ith == 0) {
            mul_mat_repack_duplicate_tensor_count += 1;
            mul_mat_repack_duplicate_tensor_total_size += tensor_size;
        }
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

    //
    // If the tensor data is shared with non-MUL_MAT ops (detected by
    // ggml_repack_scan_aliased_data_pointers at graph build time),
    // allocate a new buffer so the original data remains intact.
    //
    if (tensor->flags & GGML_TENSOR_FLAG_DUP) {
        if (!ith) {
            size_t tensor_size = ggml_nbytes(tensor);
            char *duplicate_data = (char *)malloc(tensor_size);
            if (duplicate_data == NULL) {
                mul_mat_repack_failed_count += 1;
                return;
            }
            memcpy(duplicate_data, tensor->data, tensor_size);

            //
            // wait for all threads to arrive before we publis tensor->data
            //
            ggml_wait_to_finalize_xbox(params);
            tensor->data = duplicate_data;

            mul_mat_repack_duplicate_tensor_count += 1;
            mul_mat_repack_duplicate_tensor_total_size += tensor_size;
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

#ifdef __clang__
#pragma clang attribute pop
#endif
