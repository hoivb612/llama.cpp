// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.
//
// matmulnbits_avx512.cpp
//
// AVX-512F int4 dequant dot product (M3++). This translation unit contains only
// the AVX512F fallback contract; optional-extension variants live in separate
// translation units. It must never be entered on a CPU without AVX-512F --
// callers gate on HasAvx512f() (defined in the baseline TU fusion_cpu_ep.cpp).

#include "matmulnbits_avx512.h"

#include <immintrin.h>
#include "fp16_utils.h"

#include <stdio.h>
#include <string.h>


#if defined(_MSC_VER)
#define FUSIONCPU_NOINLINE __declspec(noinline)
#else
#define FUSIONCPU_NOINLINE __attribute__((noinline))
#endif

// AVX512F-only horizontal reduction. Pair the four 128-bit quarters in the
// same order as the AVX512DQ implementation so runtime dispatch does not
// change floating-point results.

static inline float ReduceAddAvx512F(__m512 v) {

    //
    // Compute (q3 + q1) - (q2 + q0) -> 256-bits.
    //

    const __m256 qsum = _mm256_add_ps(_mm512_castps512_ps256(v),
                                      _mm512_extractf32x8_ps(v, 1));

    //
    // Compute (q3 + q1) + (q2 + q0) -> 128-bits.
    // 

    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(qsum),
                            _mm256_extractf128_ps(qsum, 1));

#if 0
    const __m128 q0 = _mm512_castps512_ps128(v);
    const __m128 q1 = _mm512_extractf32x4_ps(v, 1);
    const __m128 q2 = _mm512_extractf32x4_ps(v, 2);
    const __m128 q3 = _mm512_extractf32x4_ps(v, 3);
    __m128 sum = _mm_add_ps(_mm_add_ps(q0, q2), _mm_add_ps(q1, q3));
#endif

    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    return _mm_cvtss_f32(_mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55)));
}

// AVX512F-only int32 horizontal reduction for dpbusd / weight-sum accumulators.
// Uses the 128-bit _mm512_extracti32x4_epi32 (AVX512F) quarter tree rather than
// _mm512_reduce_add_epi32, whose compiler lowering can emit the 256-bit
// _mm512_extracti32x8_epi32 (AVX512DQ) and so is unsafe in an AVX512F-only TU.
// The 16 lane sums fit in int32 for all realistic block sizes; callers widen to
// int64 as before. No AVX512DQ specialization: these reductions are either cold
// (weight prepack) or a per-block tail dwarfed by the dpbusd loop, so a 256-bit
// integer extract would save ~1 instruction on a non-bottleneck.
static inline int32_t ReduceAddEpi32Avx512(__m512i v) {
    const __m128i q0 = _mm512_castsi512_si128(v);
    const __m128i q1 = _mm512_extracti32x4_epi32(v, 1);
    const __m128i q2 = _mm512_extracti32x4_epi32(v, 2);
    const __m128i q3 = _mm512_extracti32x4_epi32(v, 3);
    __m128i sum = _mm_add_epi32(_mm_add_epi32(q0, q2), _mm_add_epi32(q1, q3));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

FUSIONCPU_NOINLINE void DequantColumnAvx512(float* out, const uint8_t* b_row,
                                            const float* scales_row, const uint8_t* zp_row,
                                            bool has_zp, int64_t k_blocks, int64_t block_size,
                                            int default_zp) {
    const int64_t blob = block_size / 2;  // packed int4 bytes per block
    const __m128i nib = _mm_set1_epi8(0x0F);

    for (int64_t b = 0; b < k_blocks; ++b) {
        const float scale = scales_row[b];
        const __m512 sv = _mm512_set1_ps(scale);
        const int zpv = has_zp ? ((b & 1) ? (zp_row[b >> 1] >> 4) : (zp_row[b >> 1] & 0x0F))
                               : default_zp;
        const __m512i zv = _mm512_set1_epi32(zpv);
        const uint8_t* blk = b_row + b * blob;
        float* o = out + b * block_size;

        for (int64_t j = 0; j < block_size; j += 16) {
            // Load 8 packed bytes -> 16 nibbles (k_j .. k_j+15).
            __m128i packed = _mm_loadl_epi64((const __m128i*)(blk + (j >> 1)));
            __m128i lo = _mm_and_si128(packed, nib);                      // low nibbles: k_even
            __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nib);   // high nibbles: k_odd
            __m128i inter = _mm_unpacklo_epi8(lo, hi);                    // k_j, k_j+1, ... k_j+15
            __m512i qi = _mm512_cvtepu8_epi32(inter);                     // 16x int32 in [0,15]
            __m512 wv = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(qi, zv)), sv);
            _mm512_storeu_ps(o + j, wv);
        }
    }
}

FUSIONCPU_NOINLINE float DotFp32Avx512(const float* a, const float* w, int64_t k) {
    // Four independent accumulators to hide FMA latency (a single dependent
    // accumulator chain is latency-bound at ~1 FMA / 4 cycles; four chains
    // saturate the two FMA ports). k is a multiple of 16; the tail handles the
    // remaining 16-wide groups when k is not a multiple of 64.
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 64 <= k; i += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(w + i),      a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(w + i + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(w + i + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(w + i + 48), a3);
    }
    for (; i < k; i += 16) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(w + i), a0);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    return ReduceAddAvx512F(acc);
}

// Per-column VNNI int8 dot (decode-optimal). Computes one output element y[m,n]
// by streaming the weight column re-centered to int8 (wq8) against the int8
// activation row, accumulating with dpbusd over 64-lane chunks and reducing
// horizontally per block. Unlike the register-blocked tile kernel this needs no
// weight prepack/interleave, so it wins when M is small (notably M=1 decode)
// where the tile kernel's prepack cost is not amortized over rows.
//
// Precondition: block_size % 64 == 0 and HasAvx512Vnni().
FUSIONCPU_NOINLINE float DotInt8VnniColumn(const uint8_t* au8, const float* ascale,
                                           const int8_t* wq8, const float* wscale,
                                           const int32_t* wsum, int64_t k_blocks,
                                           int64_t block_size) {
    float y = 0.0f;
    for (int64_t b = 0; b < k_blocks; ++b) {
        const int64_t base = b * block_size;
        __m512i racc = _mm512_setzero_si512();
        for (int64_t j = 0; j < block_size; j += 64) {
            const __m512i av = _mm512_loadu_si512(
                (const __m512i*)(au8 + base + j));
            const __m512i wv = _mm512_loadu_si512(
                (const __m512i*)(wq8 + base + j));
            racc = _mm512_dpbusd_epi32(racc, av, wv);
        }
        const int64_t raw = ReduceAddEpi32Avx512(racc);
        // Undo the +128 activation bias: au8 = aq + 128, so
        //   sum(aq * wq8) = sum((au8 - 128) * wq8) = raw - 128 * sum(wq8).
        const int64_t block_dot = raw - 128 * (int64_t)(wsum[b]);
        y += ascale[b] * wscale[b] * (float)(block_dot);
    }
    return y;
}

// Contiguous int4 -> int8 weight re-pack for the per-column VNNI path. Writes
// wq8_out[b*block_size + t] = (q - zp) as signed int8, the per-block fp32 scale,
// and wsum_out[b] = sum_t (q - zp).
//
// Precondition: block_size % 16 == 0 and HasAvx512f().
FUSIONCPU_NOINLINE void RepackWeightColumnInt8Avx512(int8_t* wq8_out, float* wscale_out,
                                                     int32_t* wsum_out, const uint8_t* b_row,
                                                     const float* scales_row, const uint8_t* zp_row,
                                                     bool has_zp, int64_t k_blocks, int64_t block_size,
                                                     int default_zp) {
    const int64_t blob = block_size / 2;  // packed int4 bytes per block
    const __m128i nib = _mm_set1_epi8(0x0F);

    for (int64_t b = 0; b < k_blocks; ++b) {
        wscale_out[b] = scales_row[b];
        const int zpv = has_zp ? ((b & 1) ? (zp_row[b >> 1] >> 4) : (zp_row[b >> 1] & 0x0F))
                               : default_zp;
        const __m512i zv = _mm512_set1_epi32(zpv);
        const uint8_t* blk = b_row + b * blob;
        int8_t* o = wq8_out + b * block_size;
        __m512i sumq = _mm512_setzero_si512();

        for (int64_t j = 0; j < block_size; j += 16) {
            __m128i packed = _mm_loadl_epi64((const __m128i*)(blk + (j >> 1)));
            __m128i lo = _mm_and_si128(packed, nib);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nib);
            __m128i inter = _mm_unpacklo_epi8(lo, hi);
            __m512i qi = _mm512_cvtepu8_epi32(inter);                 // 16x int32 in [0,15]
            sumq = _mm512_add_epi32(sumq, qi);
            __m512i wi = _mm512_sub_epi32(qi, zv);                    // 16x int32 in [-15,15]
            __m128i b8 = _mm512_cvtepi32_epi8(wi);                    // 16x int8 (exact here)
            _mm_storeu_si128((__m128i*)(o + j), b8);
        }

        const int64_t sq = ReduceAddEpi32Avx512(sumq);
        wsum_out[b] = (int32_t)(sq - block_size * (int64_t)(zpv));
    }
}

FUSIONCPU_NOINLINE void DotInt8VnniTile16Avx512(float* out16, const uint8_t* au8_row,
                                                const float* ascale, const int8_t* wq8p,
                                                const float* wscale_packed,
                                                const int32_t* wsum_packed,
                                                int64_t k_blocks, int64_t block_size) {
    // Register-blocked VNNI microkernel (MLAS-style). The 16 int32 lanes of the
    // accumulator hold the dot products of 16 DIFFERENT output columns, so the
    // per-block dequant is a single 16-wide vector op with NO horizontal
    // reduction. The activation row is streamed once (broadcast 4 bytes per
    // dpbusd) instead of being re-read per column.
    const int64_t blk_stride = block_size * 16;  // int8 per block in the packed tile
    const __m512i c128 = _mm512_set1_epi32(128);
    __m512 yacc = _mm512_setzero_ps();
    for (int64_t b = 0; b < k_blocks; ++b) {
        const int8_t*  wp = wq8p + b * blk_stride;
        const uint8_t* ap = au8_row + b * block_size;
        __m512i acc = _mm512_setzero_si512();
        // Each k-strip is 4 consecutive k-values. Broadcast the row's 4 uint8
        // activations to all 16 lanes; the weight vector holds, per lane i, the
        // 4 weights of column i for the same strip. dpbusd then accumulates a
        // 4-wide partial dot into lane i (uint8 x int8 -> int32, no saturation).
        for (int64_t s = 0; s < block_size; s += 4) {
            int32_t a4;
            memcpy(&a4, ap + s, sizeof(a4));
            const __m512i abroad = _mm512_set1_epi32(a4);
            const __m512i wv = _mm512_loadu_si512(
                (const __m512i*)(wp + s * 16));
            acc = _mm512_dpbusd_epi32(acc, abroad, wv);
        }
        // Undo the +128 activation bias per column: au8 = aq + 128, so
        //   sum(aq * wq8) = acc - 128 * sum(wq8) = acc - 128 * wsum.
        const __m512i wsum16 = _mm512_loadu_si512(
            (const __m512i*)(wsum_packed + b * 16));
        const __m512i bd = _mm512_sub_epi32(acc, _mm512_mullo_epi32(c128, wsum16));
        const __m512 bdf = _mm512_cvtepi32_ps(bd);
        const __m512 wscale16 = _mm512_loadu_ps(wscale_packed + b * 16);
        const __m512 coef = _mm512_mul_ps(_mm512_set1_ps(ascale[b]), wscale16);
        yacc = _mm512_fmadd_ps(coef, bdf, yacc);
    }
    _mm512_storeu_ps(out16, yacc);
}

FUSIONCPU_NOINLINE void RepackWeightTile16Int8Avx512(
    int8_t* wq8p, float* wscale_packed, int32_t* wsum_packed,
    const uint8_t* b_base, int64_t b_stride,
    const float* sc_base, int64_t sc_stride,
    const uint8_t* zp_base, int64_t zp_stride, bool has_zp,
    int cnt, int64_t k_blocks, int64_t block_size, int default_zp) {
    const int64_t blob = block_size / 2;          // packed int4 bytes per block
    const int64_t blk_stride = block_size * 16;   // int8 per block in the packed tile
    const __m128i nib = _mm_set1_epi8(0x0F);

    for (int c = 0; c < 16; ++c) {
        // Pad columns beyond the valid count with zeros: zero weights contribute
        // nothing to any dpbusd lane, so the tile dot still yields the correct
        // outputs for the valid columns and 0 for the padding (which the caller
        // discards).
        if (c >= cnt) {
            for (int64_t b = 0; b < k_blocks; ++b) {
                wscale_packed[b * 16 + c] = 0.0f;
                wsum_packed[b * 16 + c] = 0;
                int8_t* obase = wq8p + b * blk_stride + (int64_t)(c) * 4;
                for (int64_t s = 0; s < block_size; s += 4) {
                    int8_t* dst = obase + (s >> 2) * 64;
                    dst[0] = dst[1] = dst[2] = dst[3] = 0;
                }
            }
            continue;
        }

        const uint8_t*  brow = b_base + (int64_t)(c) * b_stride;
        const float*    srow = sc_base + (int64_t)(c) * sc_stride;
        const uint8_t*  zrow = has_zp ? zp_base + (int64_t)(c) * zp_stride : NULL;

        for (int64_t b = 0; b < k_blocks; ++b) {
            wscale_packed[b * 16 + c] = srow[b];
            const int zpv = has_zp ? ((b & 1) ? (zrow[b >> 1] >> 4) : (zrow[b >> 1] & 0x0F))
                                   : default_zp;
            const __m512i zv = _mm512_set1_epi32(zpv);
            const uint8_t* blk = brow + b * blob;
            int8_t* obase = wq8p + b * blk_stride + (int64_t)(c) * 4;
            __m512i sumq = _mm512_setzero_si512();

            for (int64_t j = 0; j < block_size; j += 16) {
                __m128i packed = _mm_loadl_epi64((const __m128i*)(blk + (j >> 1)));
                __m128i lo = _mm_and_si128(packed, nib);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nib);
                __m128i inter = _mm_unpacklo_epi8(lo, hi);            // k_j .. k_j+15
                __m512i qi = _mm512_cvtepu8_epi32(inter);             // 16x int32 in [0,15]
                sumq = _mm512_add_epi32(sumq, qi);
                __m512i wi = _mm512_sub_epi32(qi, zv);                // 16x int32 in [-15,15]
                __m128i b8 = _mm512_cvtepi32_epi8(wi);                // 16x int8 (exact here)
                _Alignas(16) int8_t tmp16[16];
                _mm_store_si128((__m128i*)(tmp16), b8);
                // Scatter the 16 weights into 4 k-strips (4 bytes each, stride 64),
                // matching the [strip][16 cols][4 k] tile layout the dot consumes.
                const int64_t s0 = j >> 2;  // first strip index for this group
                for (int sub = 0; sub < 4; ++sub) {
                    int8_t* dst = obase + (s0 + sub) * 64;
                    dst[0] = tmp16[sub * 4 + 0];
                    dst[1] = tmp16[sub * 4 + 1];
                    dst[2] = tmp16[sub * 4 + 2];
                    dst[3] = tmp16[sub * 4 + 3];
                }
            }

            const int64_t sq = ReduceAddEpi32Avx512(sumq);
            wsum_packed[b * 16 + c] = (int32_t)(sq - block_size * (int64_t)(zpv));
        }
    }
}

FUSIONCPU_NOINLINE void DotInt8VnniTile16Int4Avx512(float* out16, const uint8_t* au8_row,
                                                    const float* ascale, const uint8_t* wq4p,
                                                    const float* wscale_packed,
                                                    const int32_t* wsum_packed,
                                                    int64_t k_blocks, int64_t block_size) {
    // Same register-blocked VNNI microkernel as DotInt8VnniTile16Avx512, but the
    // weights are nibble-packed (half the bytes) and expanded to signed int8 in
    // registers before each dpbusd. Reading half the weight bytes is the whole
    // point: decode is cold-cache bandwidth bound. block_size % 16 == 0, so each
    // block has block_size/4 strips of 32 packed bytes (= 64 nibbles).
    const int64_t blk_stride4 = block_size * 8;   // packed int4 bytes per block (16 cols)
    const __m512i c128 = _mm512_set1_epi32(128);
    const __m512i nib16 = _mm512_set1_epi16(0x000F);
    const __m512i c8b = _mm512_set1_epi8(8);
    __m512 yacc = _mm512_setzero_ps();
    for (int64_t b = 0; b < k_blocks; ++b) {
        const uint8_t* wp = wq4p + b * blk_stride4;
        const uint8_t* ap = au8_row + b * block_size;
        __m512i acc = _mm512_setzero_si512();
        for (int64_t s = 0; s < block_size; s += 4) {
            int32_t a4;
            memcpy(&a4, ap + s, sizeof(a4));
            const __m512i abroad = _mm512_set1_epi32(a4);
            // Load 32 packed bytes (= 64 nibbles = 16 cols x 4 k) for this strip
            // and expand to 64 signed int8 in the [c0:4k][c1:4k]... lane order the
            // dpbusd consumes. Each packed byte i holds value[2i] (low nibble) and
            // value[2i+1] (high nibble); we widen bytes to 16-bit lanes, split the
            // nibbles, and re-pack them as adjacent int8 (byte 2i=lo, 2i+1=hi).
            const __m256i raw = _mm256_loadu_si256(
                (const __m256i*)(wp + (s >> 2) * 32));
            const __m512i w16 = _mm512_cvtepu8_epi16(raw);            // 32x uint16 = packed bytes
            const __m512i lo = _mm512_and_si512(w16, nib16);
            const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(w16, 4), nib16);
            const __m512i comb = _mm512_or_si512(lo, _mm512_slli_epi16(hi, 8));  // [lo0,hi0,lo1,hi1,...]
            // Sign-extend each 4-bit value stored as (q-zp)&0xF: (n ^ 8) - 8.
            const __m512i wv = _mm512_sub_epi8(_mm512_xor_si512(comb, c8b), c8b);
            acc = _mm512_dpbusd_epi32(acc, abroad, wv);
        }
        const __m512i wsum16 = _mm512_loadu_si512(
            (const __m512i*)(wsum_packed + b * 16));
        const __m512i bd = _mm512_sub_epi32(acc, _mm512_mullo_epi32(c128, wsum16));
        const __m512 bdf = _mm512_cvtepi32_ps(bd);
        const __m512 wscale16 = _mm512_loadu_ps(wscale_packed + b * 16);
        const __m512 coef = _mm512_mul_ps(_mm512_set1_ps(ascale[b]), wscale16);
        yacc = _mm512_fmadd_ps(coef, bdf, yacc);
    }
    _mm512_storeu_ps(out16, yacc);
}

FUSIONCPU_NOINLINE void RepackWeightTile16Int4Avx512(
    uint8_t* wq4p,
    float* wscale_packed,
    int32_t* wsum_packed,
    const uint8_t* b_base,
    int64_t b_stride,
    const float* sc_base,
    int64_t sc_stride,
    int cnt,
    int64_t k_blocks,
    int64_t block_size,
    int default_zp)
{
    const int64_t blob = block_size / 2;          // packed int4 bytes per block (source)
    const int64_t blk_stride4 = block_size * 8;   // packed int4 bytes per block (dest tile)
    const __m128i nib = _mm_set1_epi8(0x0F);
    const __m512i zv = _mm512_set1_epi32(default_zp); // no zero_points: (q-default_zp) in [-8,7]

    for (int64_t c = 0; c < 16; ++c) {
        if (c < cnt) {
            const uint8_t* brow = b_base + c * b_stride;
            const float*   srow = sc_base + c * sc_stride;
    
            for (int64_t b = 0; b < k_blocks; ++b) {
                wscale_packed[b * 16 + c] = srow[b];
                const uint8_t* blk = brow + b * blob;
                uint8_t* obase = wq4p + b * blk_stride4 + c * 2;
                __m512i sumq = _mm512_setzero_si512();
    
                for (int64_t j = 0; j < block_size; j += 16) {
                    __m128i packed = _mm_loadl_epi64((const __m128i*)(blk + (j >> 1)));
                    __m128i lo = _mm_and_si128(packed, nib);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nib);
                    __m128i inter = _mm_unpacklo_epi8(lo, hi);            // k_j .. k_j+15
                    __m512i qi = _mm512_cvtepu8_epi32(inter);             // 16x int32 in [0,15]
                    __m512i wi = _mm512_sub_epi32(qi, zv);                // 16x int32 in [-8,7]
                    __m128i b8 = _mm512_cvtepi32_epi8(wi);                // 16x int8

                    // Pack the 16 weights into 4 strips x 2 bytes (4 nibbles/strip for
                    // this column), matching the [strip][16 cols][2 bytes] tile layout.
    
                    uint8_t* dst = obase + (j << 3);

                    dst[0] = (uint8_t)((_mm_extract_epi8(b8, 0) & 0xf) |
                                       (_mm_extract_epi8(b8, 1) << 4));

                    dst[1] = (uint8_t)((_mm_extract_epi8(b8, 2) & 0xf) |
                                       (_mm_extract_epi8(b8, 3) << 4));

                    dst[32] = (uint8_t)((_mm_extract_epi8(b8, 4) & 0xf) |
                                        (_mm_extract_epi8(b8, 5) << 4));

                    dst[33] = (uint8_t)((_mm_extract_epi8(b8, 6) & 0xf) |
                                        (_mm_extract_epi8(b8, 7) << 4));

                    dst[64] = (uint8_t)((_mm_extract_epi8(b8, 8) & 0xf) |
                                        (_mm_extract_epi8(b8, 9) << 4));

                    dst[65] = (uint8_t)((_mm_extract_epi8(b8, 10) & 0xf) |
                                        (_mm_extract_epi8(b8, 11) << 4));

                    dst[96] = (uint8_t)((_mm_extract_epi8(b8, 12) & 0xf) |
                                        (_mm_extract_epi8(b8, 13) << 4));

                    dst[97] = (uint8_t)((_mm_extract_epi8(b8, 14) & 0xf) |
                                        (_mm_extract_epi8(b8, 15) << 4));

                    sumq = _mm512_add_epi32(sumq, wi);
                }
    
                const int64_t sq = ReduceAddEpi32Avx512(sumq);
                wsum_packed[b * 16 + c] = (int32_t)sq;
            }

        } else {

            // Zero columns: nibble 0 expands to (0^8)-8 = 0, so zero bytes give a
            // zero-weight column that contributes nothing to any dpbusd lane.

            for (int64_t b = 0; b < k_blocks; ++b) {
                wscale_packed[b * 16 + c] = 0.0f;
                wsum_packed[b * 16 + c] = 0;
                uint8_t* obase = wq4p + b * blk_stride4 + c * 2;

                for (int64_t s = 0; s < block_size; s += 4) {
                    uint8_t* dst = obase + (s << 3);
                    dst[0] = dst[1] = 0;
                }
            }
        }
    }
}

// Prepack a 16-column tile into the compact-zp int4 layout: RAW nibbles q (0..15),
// per-(block,col) fp32 scale, int32 sum of q, and int32 zero_point. The caller
// narrows scale->fp16 and qsum/zp->int16 into the resident side arrays. Mirrors
// RepackWeightTile16Int4Avx512's tile layout but stores the raw quantized value (no
// zp recenter) so the dot can subtract zp*asum with a per-row activation sum.
FUSIONCPU_NOINLINE void RepackWeightTile16Int4ZpAvx512(
    uint8_t* wq4p, float* wscale_packed, int32_t* qsum_packed, int32_t* zp_packed,
    const uint8_t* b_base, int64_t b_stride,
    const float* sc_base, int64_t sc_stride,
    const uint8_t* zp_base, int64_t zp_stride,
    int cnt, int64_t k_blocks, int64_t block_size, bool interleaved) {
    const int64_t blob = block_size / 2;          // packed int4 bytes per block (source)
    const int64_t blk_stride4 = block_size * 8;   // packed int4 bytes per block (dest tile)
    const __m128i nib = _mm_set1_epi8(0x0F);

    for (int c = 0; c < 16; ++c) {
        if (c >= cnt) {
            // Zero columns: raw nibble 0 expands to weight 0, contributing nothing.
            for (int64_t b = 0; b < k_blocks; ++b) {
                wscale_packed[b * 16 + c] = 0.0f;
                qsum_packed[b * 16 + c] = 0;
                zp_packed[b * 16 + c] = 0;
                if (interleaved) {
                    // Planar layout: column c occupies bytes [4c, 4c+4) of every 64-byte unit.
                    uint8_t* ubase = wq4p + b * blk_stride4 + (int64_t)(c) * 4;
                    for (int64_t u = 0; u < block_size / 8; ++u) {
                        uint8_t* dst = ubase + u * 64;
                        dst[0] = dst[1] = dst[2] = dst[3] = 0;
                    }
                } else {
                    uint8_t* obase = wq4p + b * blk_stride4 + (int64_t)(c) * 2;
                    for (int64_t s = 0; s < block_size; s += 4) {
                        uint8_t* dst = obase + (s >> 2) * 32;
                        dst[0] = dst[1] = 0;
                    }
                }
            }
            continue;
        }

        const uint8_t* brow = b_base + (int64_t)(c) * b_stride;
        const float*   srow = sc_base + (int64_t)(c) * sc_stride;
        const uint8_t* zrow = zp_base + (int64_t)(c) * zp_stride;

        for (int64_t b = 0; b < k_blocks; ++b) {
            wscale_packed[b * 16 + c] = srow[b];
            const int zpv = (b & 1) ? (zrow[b >> 1] >> 4) : (zrow[b >> 1] & 0x0F);
            zp_packed[b * 16 + c] = zpv;
            const uint8_t* blk = brow + b * blob;
            uint8_t* obase = wq4p + b * blk_stride4 + (int64_t)(c) * 2;
            uint8_t* uobase = wq4p + b * blk_stride4 + (int64_t)(c) * 4;  // planar
            __m512i sumq = _mm512_setzero_si512();

            for (int64_t j = 0; j < block_size; j += 16) {
                __m128i packed = _mm_loadl_epi64((const __m128i*)(blk + (j >> 1)));
                __m128i lo = _mm_and_si128(packed, nib);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nib);
                __m128i inter = _mm_unpacklo_epi8(lo, hi);            // k_j .. k_j+15 raw q
                __m512i qi = _mm512_cvtepu8_epi32(inter);             // 16x int32 in [0,15]
                sumq = _mm512_add_epi32(sumq, qi);
                __m128i b8 = _mm512_cvtepi32_epi8(qi);                // 16x int8 raw q (0..15)
                _Alignas(16) int8_t tmp16[16];
                _mm_store_si128((__m128i*)(tmp16), b8);
                if (interleaved) {
                    // The 16 k-values j..j+15 span two 8-k units. For unit u the byte at
                    // (4c + jj) holds q[c, 8u+jj] in its low nibble and q[c, 8u+4+jj] in
                    // its high nibble, so a single AND / SRLI-AND recovers each 4-k strip.
                    const int64_t u0 = j >> 3;
                    uint8_t* d0 = uobase + u0 * 64;
                    for (int jj = 0; jj < 4; ++jj)
                        d0[jj] = (uint8_t)((tmp16[jj] & 0x0F) | ((tmp16[jj + 4] & 0x0F) << 4));
                    uint8_t* d1 = uobase + (u0 + 1) * 64;
                    for (int jj = 0; jj < 4; ++jj)
                        d1[jj] = (uint8_t)((tmp16[8 + jj] & 0x0F) | ((tmp16[12 + jj] & 0x0F) << 4));
                } else {
                    const int64_t s0 = j >> 2;  // first strip index for this group
                    for (int sub = 0; sub < 4; ++sub) {
                        uint8_t* dst = obase + (s0 + sub) * 32;
                        dst[0] = (uint8_t)((tmp16[sub * 4 + 0] & 0x0F) |
                                                      ((tmp16[sub * 4 + 1] & 0x0F) << 4));
                        dst[1] = (uint8_t)((tmp16[sub * 4 + 2] & 0x0F) |
                                                      ((tmp16[sub * 4 + 3] & 0x0F) << 4));
                    }
                }
            }

            const int64_t sq = ReduceAddEpi32Avx512(sumq);
            qsum_packed[b * 16 + c] = (int32_t)(sq);
        }
    }
}

// --- Alignment-stable hot kernels -----------------------------------------
// The SIMD kernels below are the ones on the active decode/prefill path
// (compact int4 MatMulNBits + fp16 GQA attend). Place them in a dedicated,
// separately-named code section: the linker gives a distinct code section a
// page-aligned base (PE SectionAlignment, default 0x1000), which decouples
// their code alignment from the total size of everything linked before them.
// This .obj's .text is placed AFTER fusion_cpu_ep.obj, so an unrelated edit
// there (e.g. the stray fprintf that regressed decode) previously shifted this
// section's base and moved these hot loops mod-64. With a page-aligned base the
// compiler's intra-function loop alignment is honored on every build, so
// unrelated code changes can no longer perturb these kernels.
#pragma code_seg(push, ".hotkern")

//
// Sign extend 4-bit nibble to 8-bits.
//

int8_t signextend_lut[64] = {
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1
};

FUSIONCPU_NOINLINE
void
DotInt8VnniTile16Int4CompactAvx512(
    float* out16,
    const uint8_t* au8_row,
    const float* ascale,
    const uint8_t* wq4p,
    const uint16_t* wscale_h,
    const int16_t* wsum16,
    int64_t k_blocks,
    int64_t block_size) {

    // Same nibble-expand VNNI microkernel as DotInt8VnniTile16Int4Avx512; only the
    // two per-(block,col) side arrays are read as half-width and widened in-register:
    //   * fp16 scale -> fp32 via vcvtph2ps (lossless for fp16-source models),
    //   * int16 wsum -> int32 via vpmovsxwd (lossless for the no-zp path).

    const int64_t blk_stride4 = block_size * 8;   // packed int4 bytes per block (16 cols)
    const __m512i c128 = _mm512_set1_epi32(128);

    const __m512i nib16 = _mm512_set1_epi16(0x0F0F);
    const __m512i lut8 = _mm512_loadu_si512(signextend_lut);

    __m512 yacc = _mm512_setzero_ps();
    for (int64_t b = 0; b < k_blocks; ++b) {
        const uint8_t* wp = wq4p + b * blk_stride4;
        const uint8_t* ap = au8_row + b * block_size;
        __m512i acc = _mm512_setzero_si512();
        for (int64_t s = 0; s < block_size; s += 4) {
            int32_t a4;
            memcpy(&a4, ap + s, sizeof(a4));
            const __m512i abroad = _mm512_set1_epi32(a4);

            const __m256i raw = _mm256_loadu_si256((const __m256i*)(wp + (s << 3)));
            const __m512i w16 = _mm512_cvtepu8_epi16(raw);
            const __m512i hi = _mm512_slli_epi16(w16, 4);
            const __m512i comb = _mm512_and_si512(_mm512_or_si512(w16, hi), nib16);
            const __m512i wv = _mm512_shuffle_epi8(lut8, comb);

            acc = _mm512_dpbusd_epi32(acc, abroad, wv);
        }

        // int16 wsum -> int32 (sign-extend); exact for the no-zp range.

        const __m256i wsum_i16 = _mm256_loadu_si256((const __m256i*)(wsum16 + b * 16));
        const __m512i wsum32 = _mm512_cvtepi16_epi32(wsum_i16);
        const __m512i bd = _mm512_sub_epi32(acc, _mm512_mullo_epi32(c128, wsum32));
        const __m512 bdf = _mm512_cvtepi32_ps(bd);

        // fp16 scale -> fp32.

        const __m256i sh = _mm256_loadu_si256((const __m256i*)(wscale_h + b * 16));
        const __m512 wscale16 = _mm512_cvtph_ps(sh);
        const __m512 coef = _mm512_mul_ps(_mm512_set1_ps(ascale[b]), wscale16);
        yacc = _mm512_fmadd_ps(coef, bdf, yacc);
    }

    _mm512_storeu_ps(out16, yacc);
}

// Single-row compact-zp int4 tile dot. RAW nibbles expand WITHOUT recentering; the
// zp correction is applied via zp*asum (per-row per-block activation sum). See
// RepackWeightTile16Int4ZpAvx512 for the layout and math derivation.
FUSIONCPU_NOINLINE void DotInt8VnniTile16Int4CompactZpAvx512(
    float* out16, const uint8_t* au8_row, const float* ascale, const int32_t* asum_row,
    const uint8_t* wq4p, const uint16_t* wscale_h, const int16_t* qsum16, const int16_t* zp16,
    int64_t k_blocks, int64_t block_size) {
    const int64_t blk_stride4 = block_size * 8;   // packed int4 bytes per block (16 cols)
    const __m512i c128 = _mm512_set1_epi32(128);
    const __m512i nib16 = _mm512_set1_epi16(0x000F);
    __m512 yacc = _mm512_setzero_ps();
    for (int64_t b = 0; b < k_blocks; ++b) {
        const uint8_t* wp = wq4p + b * blk_stride4;
        const uint8_t* ap = au8_row + b * block_size;
        __m512i acc = _mm512_setzero_si512();
        for (int64_t s = 0; s < block_size; s += 4) {
            int32_t a4;
            memcpy(&a4, ap + s, sizeof(a4));
            const __m512i abroad = _mm512_set1_epi32(a4);
            const __m256i raw = _mm256_loadu_si256(
                (const __m256i*)(wp + (s >> 2) * 32));
            const __m512i w16 = _mm512_cvtepu8_epi16(raw);
            const __m512i lo = _mm512_and_si512(w16, nib16);
            const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(w16, 4), nib16);
            // RAW quantized bytes q in [0,15]; positive so valid as the signed dpbusd
            // operand. NO xor-8/sub-8 recenter (that is the no-zp path).
            const __m512i comb = _mm512_or_si512(lo, _mm512_slli_epi16(hi, 8));
            acc = _mm512_dpbusd_epi32(acc, abroad, comb);
        }
        // acc = sum_k(a_u8*q) = sum_k(a_s8*q) + 128*qsum.
        const __m256i qsum_i16 = _mm256_loadu_si256(
            (const __m256i*)(qsum16 + b * 16));
        const __m512i qsum32 = _mm512_cvtepi16_epi32(qsum_i16);
        const __m256i zp_i16 = _mm256_loadu_si256(
            (const __m256i*)(zp16 + b * 16));
        const __m512i zp32 = _mm512_cvtepi16_epi32(zp_i16);
        const __m512i asum_b = _mm512_set1_epi32(asum_row[b]);
        // bd = acc - 128*qsum - zp*asum = sum_k(a_s8*(q - zp)).
        __m512i bd = _mm512_sub_epi32(acc, _mm512_mullo_epi32(c128, qsum32));
        bd = _mm512_sub_epi32(bd, _mm512_mullo_epi32(zp32, asum_b));
        const __m512 bdf = _mm512_cvtepi32_ps(bd);
        const __m256i sh = _mm256_loadu_si256(
            (const __m256i*)(wscale_h + b * 16));
        const __m512 wscale16 = _mm512_cvtph_ps(sh);
        const __m512 coef = _mm512_mul_ps(_mm512_set1_ps(ascale[b]), wscale16);
        yacc = _mm512_fmadd_ps(coef, bdf, yacc);
    }
    _mm512_storeu_ps(out16, yacc);
}

// Single-row INTERLEAVED compact-zp int4 tile dot. Reads the planar 8-k-unit layout
// (RepackWeightTile16Int4ZpAvx512 interleaved=true): one 64-byte load yields the low-
// nibble strip (k=8u..8u+3) and, via SRLI+AND, the high-nibble strip (k=8u+4..8u+7),
// so two dpbusd share a single load and ~1.5 unpack ops each. Same k order, same side-
// data math -> bit-identical to DotInt8VnniTile16Int4CompactZpAvx512.
FUSIONCPU_NOINLINE void DotInt8VnniTile16Int4CompactZpItlAvx512(
    float* out16, const uint8_t* au8_row, const float* ascale, const int32_t* asum_row,
    const uint8_t* wq4p, const uint16_t* wscale_h, const int16_t* qsum16, const int16_t* zp16,
    int64_t k_blocks, int64_t block_size) {
    const int64_t blk_stride4 = block_size * 8;   // packed int4 bytes per block (16 cols)
    const int64_t n_units = block_size >> 3;      // 8-k units per block
    const __m512i c128 = _mm512_set1_epi32(128);
    const __m512i nib8 = _mm512_set1_epi8(0x0F);
    __m512 yacc = _mm512_setzero_ps();
    for (int64_t b = 0; b < k_blocks; ++b) {
        const uint8_t* wp = wq4p + b * blk_stride4;
        const uint8_t* ap = au8_row + b * block_size;
        __m512i acc = _mm512_setzero_si512();
        for (int64_t u = 0; u < n_units; ++u) {
            const __m512i v = _mm512_loadu_si512(
                (const __m512i*)(wp + u * 64));
            const __m512i bv_lo = _mm512_and_si512(v, nib8);                           // k=8u..8u+3
            const __m512i bv_hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), nib8);     // k=8u+4..8u+7
            int32_t a_lo, a_hi;
            memcpy(&a_lo, ap + u * 8, sizeof(a_lo));
            memcpy(&a_hi, ap + u * 8 + 4, sizeof(a_hi));
            acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(a_lo), bv_lo);
            acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(a_hi), bv_hi);
        }
        // acc = sum_k(a_u8*q) = sum_k(a_s8*q) + 128*qsum. (Same as the strided kernel.)
        const __m256i qsum_i16 = _mm256_loadu_si256(
            (const __m256i*)(qsum16 + b * 16));
        const __m512i qsum32 = _mm512_cvtepi16_epi32(qsum_i16);
        const __m256i zp_i16 = _mm256_loadu_si256(
            (const __m256i*)(zp16 + b * 16));
        const __m512i zp32 = _mm512_cvtepi16_epi32(zp_i16);
        const __m512i asum_b = _mm512_set1_epi32(asum_row[b]);
        __m512i bd = _mm512_sub_epi32(acc, _mm512_mullo_epi32(c128, qsum32));
        bd = _mm512_sub_epi32(bd, _mm512_mullo_epi32(zp32, asum_b));
        const __m512 bdf = _mm512_cvtepi32_ps(bd);
        const __m256i sh = _mm256_loadu_si256(
            (const __m256i*)(wscale_h + b * 16));
        const __m512 wscale16 = _mm512_cvtph_ps(sh);
        const __m512 coef = _mm512_mul_ps(_mm512_set1_ps(ascale[b]), wscale16);
        yacc = _mm512_fmadd_ps(coef, bdf, yacc);
    }
    _mm512_storeu_ps(out16, yacc);
}


// Explicit 4-row M-blocked compact int4 tile core. The 4 int32 accumulators and
// 4 fp32 output accumulators are NAMED locals (not a stack array) so MSVC keeps
// them in zmm registers across the strip loop -- the whole point is to expand each
// weight strip ONCE and reuse it across the 4 rows. Bit-identical, per row, to
// DotInt8VnniTile16Int4CompactAvx512.

FUSIONCPU_NOINLINE
void
DotInt8VnniTile16Int4CompactMBAvx512(
    float* out,
    const uint8_t* au8_base,
    int64_t arow,
    const float* asc_base,
    int64_t scrow,
    const uint8_t* wq4p,
    const uint16_t* wscale_h,
    const int16_t* wsum16,
    int64_t k_blocks,
    int64_t block_size) {

    const int64_t blk_stride4 = block_size * 8;   // packed int4 bytes per block (16 cols)
    const __m512i c128 = _mm512_set1_epi32(128);

    const __m512i nib16 = _mm512_set1_epi16(0x0F0F);
    const __m512i lut8 = _mm512_loadu_si512(signextend_lut);

    const uint8_t* a0 = au8_base;
    const uint8_t* a1 = au8_base + arow;
    const uint8_t* a2 = au8_base + 2 * arow;
    const uint8_t* a3 = au8_base + 3 * arow;
    __m512 y0 = _mm512_setzero_ps(), y1 = _mm512_setzero_ps();
    __m512 y2 = _mm512_setzero_ps(), y3 = _mm512_setzero_ps();

    for (int64_t b = 0; b < k_blocks; ++b) {
        const uint8_t* wp = wq4p + b * blk_stride4;
        const int64_t boff = b * block_size;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
        // Expand each weight strip ONCE, accumulate against all 4 rows.
        for (int64_t s = 0; s < block_size; s += 4) {

            const __m256i raw = _mm256_loadu_si256((const __m256i*)(wp + (s << 3)));
            const __m512i w16 = _mm512_cvtepu8_epi16(raw);
            const __m512i hi = _mm512_slli_epi16(w16, 4);
            const __m512i comb = _mm512_and_si512(_mm512_or_si512(w16, hi), nib16);
            const __m512i wv = _mm512_shuffle_epi8(lut8, comb);

            int32_t v0, v1, v2, v3;
            memcpy(&v0, a0 + boff + s, sizeof(v0));
            memcpy(&v1, a1 + boff + s, sizeof(v1));
            memcpy(&v2, a2 + boff + s, sizeof(v2));
            memcpy(&v3, a3 + boff + s, sizeof(v3));
            acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(v0), wv);
            acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(v1), wv);
            acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(v2), wv);
            acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(v3), wv);
        }
        // Side data loaded + the 128*wsum bias computed ONCE, shared across rows.
        const __m256i wsum_i16 = _mm256_loadu_si256(
            (const __m256i*)(wsum16 + b * 16));
        const __m512i wbias = _mm512_mullo_epi32(c128, _mm512_cvtepi16_epi32(wsum_i16));
        const __m256i sh = _mm256_loadu_si256(
            (const __m256i*)(wscale_h + b * 16));
        const __m512 wscale16 = _mm512_cvtph_ps(sh);
        const __m512 bdf0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc0, wbias));
        const __m512 bdf1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc1, wbias));
        const __m512 bdf2 = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc2, wbias));
        const __m512 bdf3 = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc3, wbias));
        y0 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[b]), wscale16), bdf0, y0);
        y1 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[scrow + b]), wscale16), bdf1, y1);
        y2 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[2 * scrow + b]), wscale16), bdf2, y2);
        y3 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[3 * scrow + b]), wscale16), bdf3, y3);
    }
    _mm512_storeu_ps(out + 0 * 16, y0);
    _mm512_storeu_ps(out + 1 * 16, y1);
    _mm512_storeu_ps(out + 2 * 16, y2);
    _mm512_storeu_ps(out + 3 * 16, y3);
}

// Explicit 4-row M-blocked compact-ZP int4 tile core. Mirrors DotCompactMB4 but for
// the compact-zp tile: RAW nibbles (no xor-8/sub-8 recenter), and the per-block
// correction is  bd = acc - 128*qsum - zp*asum. The weight-only 128*qsum bias, the zp
// vector, and the fp16 scale are loaded/derived ONCE per block and shared across rows;
// zp*asum and the ascale coefficient are per-row. Bit-identical, per row, to
// DotInt8VnniTile16Int4CompactZpAvx512.
FUSIONCPU_NOINLINE void DotCompactZpMB4(float* out, const uint8_t* au8_base, int64_t arow,
                                        const float* asc_base, int64_t scrow,
                                        const int32_t* asum_base, int64_t asrow,
                                        const uint8_t* wq4p, const uint16_t* wscale_h,
                                        const int16_t* qsum16, const int16_t* zp16,
                                        int64_t k_blocks, int64_t block_size) {
    const int64_t blk_stride4 = block_size * 8;   // packed int4 bytes per block (16 cols)
    const __m512i c128 = _mm512_set1_epi32(128);
    const __m512i nib16 = _mm512_set1_epi16(0x000F);
    const uint8_t* a0 = au8_base;
    const uint8_t* a1 = au8_base + arow;
    const uint8_t* a2 = au8_base + 2 * arow;
    const uint8_t* a3 = au8_base + 3 * arow;
    __m512 y0 = _mm512_setzero_ps(), y1 = _mm512_setzero_ps();
    __m512 y2 = _mm512_setzero_ps(), y3 = _mm512_setzero_ps();

    for (int64_t b = 0; b < k_blocks; ++b) {
        const uint8_t* wp = wq4p + b * blk_stride4;
        const int64_t boff = b * block_size;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
        // Expand each RAW weight strip ONCE, accumulate against all 4 rows.
        for (int64_t s = 0; s < block_size; s += 4) {
            const __m256i raw = _mm256_loadu_si256(
                (const __m256i*)(wp + (s >> 2) * 32));
            const __m512i w16 = _mm512_cvtepu8_epi16(raw);
            const __m512i lo = _mm512_and_si512(w16, nib16);
            const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(w16, 4), nib16);
            // RAW q in [0,15] (positive -> valid signed dpbusd operand); no recenter.
            const __m512i comb = _mm512_or_si512(lo, _mm512_slli_epi16(hi, 8));
            int32_t v0, v1, v2, v3;
            memcpy(&v0, a0 + boff + s, sizeof(v0));
            memcpy(&v1, a1 + boff + s, sizeof(v1));
            memcpy(&v2, a2 + boff + s, sizeof(v2));
            memcpy(&v3, a3 + boff + s, sizeof(v3));
            acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(v0), comb);
            acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(v1), comb);
            acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(v2), comb);
            acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(v3), comb);
        }
        // Weight-only bias 128*qsum, zp vector, and fp16 scale: loaded ONCE, shared.
        const __m256i qsum_i16 = _mm256_loadu_si256(
            (const __m256i*)(qsum16 + b * 16));
        const __m512i wbias = _mm512_mullo_epi32(c128, _mm512_cvtepi16_epi32(qsum_i16));
        const __m256i zp_i16 = _mm256_loadu_si256(
            (const __m256i*)(zp16 + b * 16));
        const __m512i zp32 = _mm512_cvtepi16_epi32(zp_i16);
        const __m256i sh = _mm256_loadu_si256(
            (const __m256i*)(wscale_h + b * 16));
        const __m512 wscale16 = _mm512_cvtph_ps(sh);
        // Per-row: bd = acc - 128*qsum - zp*asum_row.
        const __m512i zpa0 = _mm512_mullo_epi32(zp32, _mm512_set1_epi32(asum_base[b]));
        const __m512i zpa1 = _mm512_mullo_epi32(zp32, _mm512_set1_epi32(asum_base[asrow + b]));
        const __m512i zpa2 = _mm512_mullo_epi32(zp32, _mm512_set1_epi32(asum_base[2 * asrow + b]));
        const __m512i zpa3 = _mm512_mullo_epi32(zp32, _mm512_set1_epi32(asum_base[3 * asrow + b]));
        const __m512 bdf0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_sub_epi32(acc0, wbias), zpa0));
        const __m512 bdf1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_sub_epi32(acc1, wbias), zpa1));
        const __m512 bdf2 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_sub_epi32(acc2, wbias), zpa2));
        const __m512 bdf3 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_sub_epi32(acc3, wbias), zpa3));
        y0 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[b]), wscale16), bdf0, y0);
        y1 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[scrow + b]), wscale16), bdf1, y1);
        y2 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[2 * scrow + b]), wscale16), bdf2, y2);
        y3 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[3 * scrow + b]), wscale16), bdf3, y3);
    }
    _mm512_storeu_ps(out + 0 * 16, y0);
    _mm512_storeu_ps(out + 1 * 16, y1);
    _mm512_storeu_ps(out + 2 * 16, y2);
    _mm512_storeu_ps(out + 3 * 16, y3);
}

// Explicit 4-row M-blocked INTERLEAVED compact-zp int4 tile core. Mirrors DotCompactZpMB4
// but reads the planar 8-k-unit layout: one 64-byte load per unit is unpacked ONCE (AND +
// SRLI-AND) into the low/high 4-k strips and reused across all 4 rows (8 dpbusd per unit).
// Per-row math + accumulation order are IDENTICAL to DotInt8VnniTile16Int4CompactZpItlAvx512.
FUSIONCPU_NOINLINE void DotCompactZpItlMB4(float* out, const uint8_t* au8_base, int64_t arow,
                                           const float* asc_base, int64_t scrow,
                                           const int32_t* asum_base, int64_t asrow,
                                           const uint8_t* wq4p, const uint16_t* wscale_h,
                                           const int16_t* qsum16, const int16_t* zp16,
                                           int64_t k_blocks, int64_t block_size) {
    const int64_t blk_stride4 = block_size * 8;   // packed int4 bytes per block (16 cols)
    const int64_t n_units = block_size >> 3;      // 8-k units per block
    const __m512i c128 = _mm512_set1_epi32(128);
    const __m512i nib8 = _mm512_set1_epi8(0x0F);
    const uint8_t* a0 = au8_base;
    const uint8_t* a1 = au8_base + arow;
    const uint8_t* a2 = au8_base + 2 * arow;
    const uint8_t* a3 = au8_base + 3 * arow;
    __m512 y0 = _mm512_setzero_ps(), y1 = _mm512_setzero_ps();
    __m512 y2 = _mm512_setzero_ps(), y3 = _mm512_setzero_ps();

    for (int64_t b = 0; b < k_blocks; ++b) {
        const uint8_t* wp = wq4p + b * blk_stride4;
        const int64_t boff = b * block_size;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
        // Unpack each 8-k unit ONCE (low/high nibble strips), accumulate all 4 rows.
        // The two 4-k strips are consumed sequentially so only one weight vector +
        // four activation broadcasts are live at a time (keeps register pressure at
        // the level of the proven strided DotCompactZpMB4 core).
        for (int64_t u = 0; u < n_units; ++u) {
            const __m512i v = _mm512_loadu_si512(
                (const __m512i*)(wp + u * 64));
            const int64_t koff = boff + u * 8;
            // Low-nibble strip: k = 8u..8u+3.
            {
                const __m512i bv = _mm512_and_si512(v, nib8);
                int32_t v0, v1, v2, v3;
                memcpy(&v0, a0 + koff, 4);
                memcpy(&v1, a1 + koff, 4);
                memcpy(&v2, a2 + koff, 4);
                memcpy(&v3, a3 + koff, 4);
                acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(v0), bv);
                acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(v1), bv);
                acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(v2), bv);
                acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(v3), bv);
            }
            // High-nibble strip: k = 8u+4..8u+7.
            {
                const __m512i bv = _mm512_and_si512(_mm512_srli_epi16(v, 4), nib8);
                int32_t v0, v1, v2, v3;
                memcpy(&v0, a0 + koff + 4, 4);
                memcpy(&v1, a1 + koff + 4, 4);
                memcpy(&v2, a2 + koff + 4, 4);
                memcpy(&v3, a3 + koff + 4, 4);
                acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(v0), bv);
                acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(v1), bv);
                acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(v2), bv);
                acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(v3), bv);
            }
        }
        const __m256i qsum_i16 = _mm256_loadu_si256(
            (const __m256i*)(qsum16 + b * 16));
        const __m512i wbias = _mm512_mullo_epi32(c128, _mm512_cvtepi16_epi32(qsum_i16));
        const __m256i zp_i16 = _mm256_loadu_si256(
            (const __m256i*)(zp16 + b * 16));
        const __m512i zp32 = _mm512_cvtepi16_epi32(zp_i16);
        const __m256i sh = _mm256_loadu_si256(
            (const __m256i*)(wscale_h + b * 16));
        const __m512 wscale16 = _mm512_cvtph_ps(sh);
        const __m512i zpa0 = _mm512_mullo_epi32(zp32, _mm512_set1_epi32(asum_base[b]));
        const __m512i zpa1 = _mm512_mullo_epi32(zp32, _mm512_set1_epi32(asum_base[asrow + b]));
        const __m512i zpa2 = _mm512_mullo_epi32(zp32, _mm512_set1_epi32(asum_base[2 * asrow + b]));
        const __m512i zpa3 = _mm512_mullo_epi32(zp32, _mm512_set1_epi32(asum_base[3 * asrow + b]));
        const __m512 bdf0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_sub_epi32(acc0, wbias), zpa0));
        const __m512 bdf1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_sub_epi32(acc1, wbias), zpa1));
        const __m512 bdf2 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_sub_epi32(acc2, wbias), zpa2));
        const __m512 bdf3 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_sub_epi32(acc3, wbias), zpa3));
        y0 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[b]), wscale16), bdf0, y0);
        y1 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[scrow + b]), wscale16), bdf1, y1);
        y2 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[2 * scrow + b]), wscale16), bdf2, y2);
        y3 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(asc_base[3 * scrow + b]), wscale16), bdf3, y3);
    }
    _mm512_storeu_ps(out + 0 * 16, y0);
    _mm512_storeu_ps(out + 1 * 16, y1);
    _mm512_storeu_ps(out + 2 * 16, y2);
    _mm512_storeu_ps(out + 3 * 16, y3);
}

FUSIONCPU_NOINLINE void DotInt8VnniTile16Int4CompactZpMBAvx512(
    float* out, const uint8_t* au8_base, int64_t au8_row_stride,
    const float* ascale_base, int64_t ascale_row_stride,
    const int32_t* asum_base, int64_t asum_row_stride,
    const uint8_t* wq4p, const uint16_t* wscale_h, const int16_t* qsum16, const int16_t* zp16,
    int mrows, int64_t k_blocks, int64_t block_size) {
    // Full 4-row blocks take the register-blocked kernel; 1..3 tail rows fall back to
    // the single-row compact-zp kernel (negligible at prefill M, one fast path).
    if (mrows == FUSIONCPU_COMPACT_MB_MAX) {
        DotCompactZpMB4(out, au8_base, au8_row_stride, ascale_base, ascale_row_stride,
                        asum_base, asum_row_stride, wq4p, wscale_h, qsum16, zp16,
                        k_blocks, block_size);
        return;
    }
    for (int r = 0; r < mrows; ++r) {
        DotInt8VnniTile16Int4CompactZpAvx512(out + r * 16,
                                             au8_base + r * au8_row_stride,
                                             ascale_base + r * ascale_row_stride,
                                             asum_base + r * asum_row_stride,
                                             wq4p, wscale_h, qsum16, zp16, k_blocks, block_size);
    }
}

FUSIONCPU_NOINLINE void DotInt8VnniTile16Int4CompactZpItlMBAvx512(
    float* out, const uint8_t* au8_base, int64_t au8_row_stride,
    const float* ascale_base, int64_t ascale_row_stride,
    const int32_t* asum_base, int64_t asum_row_stride,
    const uint8_t* wq4p, const uint16_t* wscale_h, const int16_t* qsum16, const int16_t* zp16,
    int mrows, int64_t k_blocks, int64_t block_size) {
    // Full 4-row blocks take the interleaved register-blocked kernel; 1..3 tail rows
    // fall back to the single-row interleaved kernel (same planar layout).
    if (mrows == FUSIONCPU_COMPACT_MB_MAX) {
        DotCompactZpItlMB4(out, au8_base, au8_row_stride, ascale_base, ascale_row_stride,
                           asum_base, asum_row_stride, wq4p, wscale_h, qsum16, zp16,
                           k_blocks, block_size);
        return;
    }
    for (int r = 0; r < mrows; ++r) {
        DotInt8VnniTile16Int4CompactZpItlAvx512(out + r * 16,
                                                au8_base + r * au8_row_stride,
                                                ascale_base + r * ascale_row_stride,
                                                asum_base + r * asum_row_stride,
                                                wq4p, wscale_h, qsum16, zp16, k_blocks, block_size);
    }
}

FUSIONCPU_NOINLINE float GqaDotF16Avx512(const float* q, const uint16_t* k, int64_t hs) {
    __m512 acc = _mm512_setzero_ps();
    int64_t d = 0;
    for (; d + 16 <= hs; d += 16) {
        const __m256i kh = _mm256_loadu_si256((const __m256i*)(k + d));
        const __m512 kf = _mm512_cvtph_ps(kh);
        const __m512 qf = _mm512_loadu_ps(q + d);
        acc = _mm512_fmadd_ps(qf, kf, acc);
    }

    float sum = ReduceAddAvx512F(acc);
    for (; d < hs; ++d) sum += q[d] * FusionCpuHalfToFloat(k[d]);
    return sum;
}

FUSIONCPU_NOINLINE void GqaAccumVF16Avx512(float* acc, float w, const uint16_t* v, int64_t hs) {
    const __m512 wv = _mm512_set1_ps(w);
    int64_t d = 0;
    for (; d + 16 <= hs; d += 16) {
        const __m256i vh = _mm256_loadu_si256((const __m256i*)(v + d));
        const __m512 vf = _mm512_cvtph_ps(vh);
        __m512 af = _mm512_loadu_ps(acc + d);
        af = _mm512_fmadd_ps(wv, vf, af);
        _mm512_storeu_ps(acc + d, af);
    }
    for (; d < hs; ++d) acc[d] += w * FusionCpuHalfToFloat(v[d]);
}

FUSIONCPU_NOINLINE void RmsNormRowF16Avx512(const uint16_t* in, const uint16_t* skip,
                                            const float* gamma, uint16_t* out,
                                            uint16_t* sum, int64_t width, float eps) {
    const int rnd = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    __m512 acc = _mm512_setzero_ps();
    int64_t d = 0;
    // Pass 1: accumulate sum of squares of value = in (+ skip); emit sum if wanted.
    for (; d + 16 <= width; d += 16) {
        __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(in + d)));
        if (skip != NULL) {
            v = _mm512_add_ps(v, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(skip + d))));
        }
        acc = _mm512_fmadd_ps(v, v, acc);
        if (sum != NULL) {
            _mm256_storeu_si256((__m256i*)(sum + d), _mm512_cvtps_ph(v, rnd));
        }
    }
    float ms = ReduceAddAvx512F(acc);
    for (int64_t t = d; t < width; ++t) {
        float v = FusionCpuHalfToFloat(in[t]);
        if (skip != NULL) v += FusionCpuHalfToFloat(skip[t]);
        ms += v * v;
        if (sum != NULL) sum[t] = FusionCpuFloatToHalf(v);
    }
    const float inv = 1.0f / sqrtf(ms / (float)width + eps);
    const __m512 invv = _mm512_set1_ps(inv);
    // Pass 2: out = value * inv * gamma.
    for (d = 0; d + 16 <= width; d += 16) {
        __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(in + d)));
        if (skip != NULL) {
            v = _mm512_add_ps(v, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(skip + d))));
        }
        const __m512 g = _mm512_loadu_ps(gamma + d);
        const __m512 o = _mm512_mul_ps(_mm512_mul_ps(v, invv), g);
        _mm256_storeu_si256((__m256i*)(out + d), _mm512_cvtps_ph(o, rnd));
    }
    for (int64_t t = d; t < width; ++t) {
        float v = FusionCpuHalfToFloat(in[t]);
        if (skip != NULL) v += FusionCpuHalfToFloat(skip[t]);
        out[t] = FusionCpuFloatToHalf(v * inv * gamma[t]);
    }
}

// Widen a contiguous fp16 (uint16_t) buffer into fp32. fp16->fp32 is exact, so
// this is bit-identical to a scalar FusionCpuHalfToFloat loop. Precondition:
// HasAvx512f().
FUSIONCPU_NOINLINE void FusionCpuWidenHalfToFloatAvx512(float* dst, const uint16_t* src,
                                                        int64_t count) {
    int64_t i = 0;
    for (; i + 16 <= count; i += 16) {
        _mm512_storeu_ps(dst + i,
                         _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(src + i))));
    }
    for (; i < count; ++i) dst[i] = FusionCpuHalfToFloat(src[i]);
}

// Narrow fp32 -> fp16 with round-to-nearest-even; bit-identical to a scalar
// FusionCpuFloatToHalf loop. Precondition: HasAvx512f().
FUSIONCPU_NOINLINE void FusionCpuNarrowFloatToHalfAvx512(uint16_t* dst, const float* src,
                                                         int64_t count) {
    const int rnd = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    int64_t i = 0;
    for (; i + 16 <= count; i += 16) {
        _mm256_storeu_si256((__m256i*)(dst + i),
                            _mm512_cvtps_ph(_mm512_loadu_ps(src + i), rnd));
    }
    for (; i < count; ++i) dst[i] = FusionCpuFloatToHalf(src[i]);
}

// fp16 SwiGLU: out = fp16( fp16(gate*up) * fp16(sigmoid(gate)) ), with the
// per-element intermediate rounding to fp16 preserved to match the scalar
// reference bit-for-bit. Conversions and multiplies are vectorized; sigmoid's
// expf stays scalar (no vectorized exp available), so results are identical to
// the baseline scalar loop -- only the fp16<->fp32 conversion overhead is
// removed. gate/up/out are fp16; count is the element count.
// When fast_exp is true, the sigmoid uses the vectorized FusionCpuExpPsAvx512
// approximation instead of scalar expf (faster, not bit-exact).
FUSIONCPU_NOINLINE void FusionCpuSwiGluF16Avx512(const uint16_t* gate, const uint16_t* up,
                                                 uint16_t* out, int64_t count) {
    const int rnd = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    int64_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m512 g = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(gate + i)));
        const __m512 u = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(up + i)));
        // product = fp16(u * g) widened back to fp32.
        const __m512 product = _mm512_cvtph_ps(_mm512_cvtps_ph(_mm512_mul_ps(u, g), rnd));
        __m512 sig_f;
        float gbuf[16];
       float sbuf[16];
        _mm512_storeu_ps(gbuf, g);
        for (int l = 0; l < 16; ++l) sbuf[l] = 1.0f / (1.0f + expf(-gbuf[l]));
        sig_f = _mm512_loadu_ps(sbuf);
        const __m512 sigmoid = _mm512_cvtph_ps(_mm512_cvtps_ph(sig_f, rnd));
        _mm256_storeu_si256((__m256i*)(out + i),
                            _mm512_cvtps_ph(_mm512_mul_ps(product, sigmoid), rnd));
    }
    for (; i < count; ++i) {
        const float g = FusionCpuHalfToFloat(gate[i]);
        const float u = FusionCpuHalfToFloat(up[i]);
        const float product = FusionCpuHalfToFloat(FusionCpuFloatToHalf(u * g));
        const float sigmoid = FusionCpuHalfToFloat(
            FusionCpuFloatToHalf(1.0f / (1.0f + expf(-g))));
        out[i] = FusionCpuFloatToHalf(product * sigmoid);
    }
}

#pragma code_seg(pop)  // .hotkern

