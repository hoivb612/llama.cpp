#include "vec-b612.h"

#include <cassert>

#if defined(_MSC_VER)
// disable "possible loss of data" to avoid hundreds of casts
// we should just be careful :)
#pragma warning(disable: 4244 4267)
#endif

// For AVX512_BF16 full time
#ifndef __AVX512BF16__
    #define __AVX512BF16__
#endif // __AVX512BF16__
#define GGML_BF16_STEP32 128
#define GGML_BF16_EPR32 32
#define GGML_BF16_STEP16 64
#define GGML_BF16_EPR16 16

// precomputed gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

float ggml_cosine_similarity_f32(int n, float *x, float *y) {
#pragma comment(linker, "/EXPORT:ggml_cosine_similarity_f32=" __FUNCTION__)

    float denom_x;
    float denom_y;
    float dot;

    ggml_vec_dot_f32(n, &dot, 0, x, 0, y, 0, 1);
    ggml_vec_sumsq_f32(n, &denom_x, x);
    ggml_vec_sumsq_f32(n, &denom_y, y);
    return dot / (float)sqrt(denom_x * denom_y);
}

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc) {
#pragma comment(linker, "/EXPORT:ggml_vec_dot_f32=" __FUNCTION__)

   assert(nrc == 1);
   GGML_UNUSED(nrc);
   GGML_UNUSED(bx);
   GGML_UNUSED(by);
   GGML_UNUSED(bs);

    float sumf = 0.0f;
    int64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_dot_f32 version")

    const int64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    if (xn) {
        __m512 sum[GGML_F32_ARR];
        __m512 ax[GGML_F32_ARR];
        __m512 ay[GGML_F32_ARR];

        const int64_t np = (n & ~(GGML_F32_STEP16 - 1));

        sum[0] = _mm512_setzero_ps();
        sum[1] = _mm512_setzero_ps();
        sum[2] = _mm512_setzero_ps();
        sum[3] = _mm512_setzero_ps();

        if (np) {
            do {
              for (int64_t j = 0; j < GGML_F32_ARR; j++) {
                  ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
                  ay[j] = _mm512_loadu_ps(y + i + j * GGML_F32_EPR16);
                  sum[j] = _mm512_fmadd_ps(ax[j], ay[j], sum[j]); 
              }

              i += GGML_F32_STEP16;
            } while (i < np);
        }

        if (xn > np) {
            do {
                ax[0] = _mm512_loadu_ps(x + i);
                ay[0] = _mm512_loadu_ps(y + i);
                sum[0] = _mm512_fmadd_ps(ax[0], ay[0], sum[0]);
                i += GGML_F32_EPR16;
            } while (i < xn);
        }

        // reduce sum0..sum3 to sumf
        GGML_F32_VEC_REDUCE512(sumf, sum);
    }

    // leftovers
    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            sumf += x[i] * y[i];
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    const int64_t xn = (n & ~(GGML_F32_EPR - 1));

    if (xn) {
         __m256 sum[GGML_F32_ARR];
        __m256 ax[GGML_F32_ARR];
        __m256 ay[GGML_F32_ARR];

       const int64_t np = (n & ~(GGML_F32_STEP - 1));

        sum[0] = _mm256_setzero_ps();
        sum[1] = _mm256_setzero_ps();
        sum[2] = _mm256_setzero_ps();
        sum[3] = _mm256_setzero_ps();

        if (np) {
            do {
                for (int64_t j = 0; j < GGML_F32_ARR; j++) {
                    ax[j] = _mm256_loadu_ps(x + i + j * GGML_F32_EPR);
                    ay[j] = _mm256_loadu_ps(y + i + j * GGML_F32_EPR);
                    sum[j] = _mm256_fmadd_ps(ax[j], ay[j], sum[j]);
                }
    
                i += GGML_F32_STEP;
            } while (i < np);
        }

        if (xn > np) {
            do {
                ax[0] = _mm256_loadu_ps(x + i);
                ay[0] = _mm256_loadu_ps(y + i);
                sum[0] = _mm256_fmadd_ps(ax[0], ay[0], sum[0]);
                i += GGML_F32_EPR;
            } while (i < xn);
        }

        // reduce sum0..sum3 to sumf
        GGML_F32_VEC_REDUCE(sumf, sum);
    }

    // leftovers
    if (n & (GGML_F32_EPR - 1)) {
        do {
            sumf += x[i] * y[i];
            i += 1;
        } while (i < n);
    }

#else

    // scalar
    for (int64_t i = 0; i < n; ++i) {
        sumf += x[i] * y[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

    *s = sumf;
}

float ggml_cosine_similarity_bf16(int n, ggml_bf16_t *x, ggml_bf16_t *y) {
#pragma comment(linker, "/EXPORT:ggml_cosine_similarity_bf16=" __FUNCTION__)

    float denom_x;
    float denom_y;
    float dot;

    ggml_vec_dot_bf16(n, &dot, 0, x, 0, y, 0, 1);
    ggml_vec_sumsq_bf16(n, &denom_x, x);
    ggml_vec_sumsq_bf16(n, &denom_y, y);
    return dot / (float)sqrt(denom_x * denom_y);
}

void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc) {
#pragma comment(linker, "/EXPORT:ggml_vec_dot_bf16=" __FUNCTION__)

    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    uint64_t nc = n;
    uint64_t i = 0;
    float sumf = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__) && !defined(__gnu_linux__)

    const uint64_t xn = (nc & ~(GGML_BF16_EPR32 - 1));

    if (xn) {
        __m512 sum[GGML_F32_ARR];
        __m512i ax[GGML_F32_ARR];
        __m512i ay[GGML_F32_ARR];

        const uint64_t np = (nc & ~(GGML_BF16_STEP32 - 1));

        sum[0] = _mm512_setzero_ps();
        sum[1] = _mm512_setzero_ps();
        sum[2] = _mm512_setzero_ps();
        sum[3] = _mm512_setzero_ps();

        if (np) {
            do {
              for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                  ax[j] = _mm512_loadu_si512(x + i + j * GGML_BF16_EPR32);
                  ay[j] = _mm512_loadu_si512(y + i + j * GGML_BF16_EPR32);
                  sum[j] = _mm512_dpbf16_ps(sum[j], ax[j], ay[j]);
              }

              i += GGML_BF16_STEP32;
            } while (i < np);
        }

        if (xn > np) {
            do {
                ax[0] = _mm512_loadu_si512(x + i);
                ay[0] = _mm512_loadu_si512(y + i);
                sum[0] = _mm512_dpbf16_ps(sum[0], ax[0], ay[0]);
                i += GGML_BF16_EPR32;
            } while (i < xn);
        }

        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE512(sumf, sum);
    }

    // leftovers

    if (nc & (GGML_BF16_EPR32 - 1)) {
        do {
            sumf += (GGML_BF16_TO_FP32(x[i]) * GGML_BF16_TO_FP32(y[i]));
            i += 1;
        } while (i < nc);
    }

#elif defined(__AVX2__) && !defined(__gnu_linux__)

    const uint64_t xn = (nc & ~(GGML_BF16_EPR16 - 1));

    if (xn) {
        __m256 sum[GGML_F32_ARR];
        __m256i ax[GGML_F32_ARR];
        __m256i ay[GGML_F32_ARR];

        const uint64_t np = (nc & ~(GGML_BF16_STEP16 - 1));

        sum[0] = _mm256_setzero_ps();
        sum[1] = _mm256_setzero_ps();
        sum[2] = _mm256_setzero_ps();
        sum[3] = _mm256_setzero_ps();

        if (np) {
            do {
                for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                    ax[j] = _mm256_loadu_si256((__m256i *)(x + i + j * GGML_BF16_EPR16));
                    ay[j] = _mm256_loadu_si256((__m256i *)(y + i + j * GGML_BF16_EPR16));
                    sum[j] = _mm256_dpbf16_ps(sum[j], ax[j], ay[j]);
                }

                i += GGML_BF16_STEP16;
            } while (i < np);
        }

        if (xn > np) {
            do {
                ax[0] = _mm256_loadu_si256((__m256i *)(x + i));
                ay[0] = _mm256_loadu_si256((__m256i *)(y + i));
                sum[0] = _mm256_dpbf16_ps(sum[0], ax[0], ay[0]);

                i += GGML_BF16_EPR16;
            } while (i < xn);
        }

        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE(sumf, sum);
    }

    // leftovers

    if (nc & (GGML_BF16_EPR16 - 1)) {
        do {
            sumf += (GGML_BF16_TO_FP32(x[i]) * GGML_BF16_TO_FP32(y[i]));
            i += 1;
        } while (i < nc);
    }

#else

    for (; i < nc; ++i) {
        sumf += (GGML_BF16_TO_FP32(x[i]) * GGML_BF16_TO_FP32(y[i]));
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

    *s = sumf;
}

void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
#pragma comment(linker, "/EXPORT:ggml_vec_dot_f16=" __FUNCTION__)

    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

     float sumf = 0.0;
    int64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_dot_f16 version")

    const int64_t xn = (n & ~(GGML_F16_EPR16 - 1));

    if (xn) {
        __m512 sum[GGML_F16_ARR];
        __m512 ax[GGML_F16_ARR];
        __m512 ay[GGML_F16_ARR];

        sum[0] = _mm512_setzero_ps();
        sum[1] = _mm512_setzero_ps();
        sum[2] = _mm512_setzero_ps();
        sum[3] = _mm512_setzero_ps();

        const int64_t np = (n & ~(GGML_F16_STEP16 - 1));

        if (np) {
            do {
                for (int64_t j = 0; j < GGML_F16_ARR; j++) {
                    ax[j] = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(x + i + j * GGML_F16_EPR16)));
                    ay[j] = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(y + i + j * GGML_F16_EPR16)));
                    sum[j] = _mm512_fmadd_ps(ax[j], ay[j], sum[j]);
                }
    
                i += GGML_F16_STEP16;
            } while (i < np);
        }

        if (xn > np) {
            do {
                ax[0] = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(x + i)));
                ay[0] = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(y + i)));
                sum[0] = _mm512_fmadd_ps(ax[0], ay[0], sum[0]);
                i += GGML_F16_EPR16;
            } while (i < xn);
        }

        // reduce sum0..sum3 to sumf
        GGML_F16_VEC_REDUCE512(sumf, sum);
    }

    // leftovers
    if (n & (GGML_F16_EPR16 - 1)) {
        do {
            sumf += GGML_FP16_TO_FP32(x[i]) * GGML_FP16_TO_FP32(y[i]);
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    const int64_t xn = (n & ~(GGML_F16_EPR - 1));

    if (xn) {
        __m256 sum[GGML_F16_ARR];
        __m256 ax[GGML_F16_ARR];
        __m256 ay[GGML_F16_ARR];

        sum[0] = _mm256_setzero_ps();
        sum[1] = _mm256_setzero_ps();
        sum[2] = _mm256_setzero_ps();
        sum[3] = _mm256_setzero_ps();

        const int64_t np = (n & ~(GGML_F16_STEP - 1));

        if (np) {
            do {
                for (int64_t j = 0; j < GGML_F16_ARR; j++) {
                    ax[j] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(x + i + j * GGML_F16_EPR)));
                    ay[j] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(y + i + j * GGML_F16_EPR)));
                    sum[j] = _mm256_fmadd_ps(ax[j], ay[j], sum[j]);
                }
    
                i += GGML_F16_STEP;
            } while (i < np);
        }

        if (xn > np) {
            do {
                ax[0] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(x + i)));
                ay[0] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(y + i)));
                sum[0] = _mm256_fmadd_ps(ax[0], ay[0], sum[0]);
                i += GGML_F16_EPR;
            } while (i < xn);
        }

        // reduce sum0..sum3 to sumf
        GGML_F16_VEC_REDUCE(sumf, sum);
    }

    // leftovers
    if (n & (GGML_F16_EPR - 1)) {

        do {
            sumf += (GGML_FP16_TO_FP32(x[i])*GGML_FP16_TO_FP32(y[i]));
            i += 1;
        } while (i < n);
    }

#else

    for (int64_t i = 0; i < n; ++i) {
        sumf += GGML_FP16_TO_FP32(x[i]) * GGML_FP16_TO_FP32(y[i]);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

    *s = sumf;
}

void ggml_vec_dot_bf16_f32(const int n, float * GGML_RESTRICT s, size_t bs, const ggml_bf16_t * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc) {
#pragma comment(linker, "/EXPORT:ggml_vec_dot_bf16_f32=" __FUNCTION__)

    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const uint64_t nc = n;
    float sumf = 0.0;
    uint64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_dot_bf16_f32 version")

    const uint64_t xn = (nc & ~(GGML_F32_EPR16 - 1));

    if (xn) {
        __m256i au[GGML_F32_ARR];
        __m512i av[GGML_F32_ARR];
        __m512 sum[GGML_F32_ARR];
        __m512 ax[GGML_F32_ARR];
        __m512 ay[GGML_F32_ARR];

        const uint64_t np = (nc & ~(GGML_F32_STEP16 - 1));

        sum[0] = _mm512_setzero_ps();
        sum[1] = _mm512_setzero_ps();
        sum[2] = _mm512_setzero_ps();
        sum[3] = _mm512_setzero_ps();

        if (np) {
            do {
                for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                    au[j] = _mm256_loadu_si256((__m256i *)(x + i + j * GGML_F16_EPR16));
                    ay[j] = _mm512_loadu_ps(y + i + j * GGML_F32_EPR16);
                    av[j] = _mm512_cvtepu16_epi32(au[j]);
                    av[j] = _mm512_slli_epi32(av[j], 16);
                    ax[j] = _mm512_castsi512_ps(av[j]);
                    sum[j] = _mm512_fmadd_ps(ax[j], ay[j], sum[j]);
                }
    
                i += GGML_F32_STEP16;
            } while (i < np);
        }

        if (xn > np) {
            do {
                au[0] = _mm256_loadu_si256((__m256i *)(x + i));
                ay[0] = _mm512_loadu_ps(y + i);
                av[0] = _mm512_cvtepu16_epi32(au[0]);
                av[0] = _mm512_slli_epi32(av[0], 16);
                ax[0] = _mm512_castsi512_ps(av[0]);
                sum[0] = _mm512_fmadd_ps(ax[0], ay[0], sum[0]);
                i += GGML_F32_EPR16;
            } while (i < xn);
        }

        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE512(sumf, sum);
    }

    // leftovers

    if (nc & (GGML_F32_EPR16 - 1)) {
        do {
            sumf += GGML_BF16_TO_FP32(x[i]) * y[i];
            i += 1;
        } while (i < nc);
    }

#elif defined(__AVX2__)

    const uint64_t xn = (nc & ~(GGML_F32_EPR - 1));

    if (xn) {
        __m128i au[GGML_F32_ARR];
        __m256i av[GGML_F32_ARR];
        __m256 sum[GGML_F32_ARR];
        __m256 ax[GGML_F32_ARR];
        __m256 ay[GGML_F32_ARR];

        const uint64_t np = (nc & ~(GGML_F32_STEP - 1));

        sum[0] = _mm256_setzero_ps();
        sum[1] = _mm256_setzero_ps();
        sum[2] = _mm256_setzero_ps();
        sum[3] = _mm256_setzero_ps();

        if (np) {
            do {
                for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                    au[j] = _mm_loadu_si128((__m128i *)(x + i + j * GGML_F32_EPR));
                    ay[j] = _mm256_loadu_ps(y + i + j * GGML_F32_EPR);
                    av[j] = _mm256_cvtepu16_epi32(au[j]);
                    av[j] = _mm256_slli_epi32(av[j], 16);
                    ax[j] = _mm256_castsi256_ps(av[j]);
                    sum[j] = _mm256_fmadd_ps(ax[j], ay[j], sum[j]);
                }
    
                i += GGML_F32_STEP;
            } while (i < np);
        }

        if (xn > np) {
            do {
                au[0] = _mm_loadu_si128((__m128i *)(x + i));
                ay[0] = _mm256_loadu_ps(y + i);
                av[0] = _mm256_cvtepu16_epi32(au[0]);
                av[0] = _mm256_slli_epi32(av[0], 16);
                ax[0] = _mm256_castsi256_ps(av[0]);
                sum[0] = _mm256_fmadd_ps(ax[0], ay[0], sum[0]);
                i += GGML_F32_EPR;
            } while (i < xn);
        }

        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE(sumf, sum);
    }

    // leftovers

    if (nc & (GGML_F32_EPR - 1)) {
        do {
            sumf += GGML_BF16_TO_FP32(x[i]) * y[i];
            i += 1;
        } while (i < nc);
    }

#else

    for (; i < nc; ++i) {
        sumf += GGML_BF16_TO_FP32(x[i]) * y[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__) 

    *s = sumf;
}

void ggml_vec_dot_f16_f32(const int64_t n, float * GGML_RESTRICT s, size_t bs, const ggml_fp16_t * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc) {
#pragma comment(linker, "/EXPORT:ggml_vec_dot_f16_f32=" __FUNCTION__)

    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const uint64_t nc = n;
    float sumf = 0.0;
    uint64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_dot_f16_f32 version")

    const uint64_t xn = (nc & ~(GGML_F32_EPR16 - 1));

    if (xn) {
        __m512 sum[GGML_F32_ARR];
        __m512 ax[GGML_F32_ARR];
        __m512 ay[GGML_F32_ARR];

        const uint64_t np = (nc & ~(GGML_F32_STEP16 - 1));

        sum[0] = _mm512_setzero_ps();
        sum[1] = _mm512_setzero_ps();
        sum[2] = _mm512_setzero_ps();
        sum[3] = _mm512_setzero_ps();

        if (np) {
            do {
                for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                    ax[j] = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i *)(x + i + j * GGML_F32_EPR16)));
                    ay[j] = _mm512_loadu_ps(y + i + j * GGML_F32_EPR16);
                    sum[j] = _mm512_fmadd_ps(ax[j], ay[j], sum[j]);
                }
    
                i += GGML_F32_STEP16;
            } while (i < np);
        }

        if (xn > np) {
            do {
                ax[0] = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i *)(x + i)));
                ay[0] = _mm512_loadu_ps(y + i);
                sum[0] = _mm512_fmadd_ps(ax[0], ay[0], sum[0]);
                i += GGML_F32_EPR16;
            } while (i < xn);
        }

        // reduce sum0..sum3 to sumf
        GGML_F32_VEC_REDUCE512(sumf, sum);
    }

    // leftovers

    if (nc & (GGML_F32_EPR16 - 1)) {
        do {
            sumf += GGML_FP16_TO_FP32(x[i]) * y[i];
            i += 1;
        } while (i < nc);
    }

#elif defined(__AVX2__)

    const uint64_t xn = (nc & ~(GGML_F32_EPR - 1));

    if (xn) {
        __m256 sum[GGML_F32_ARR];
        __m256 ax[GGML_F32_ARR];
        __m256 ay[GGML_F32_ARR];

        const uint64_t np = (nc & ~(GGML_F32_STEP - 1));

        sum[0] = _mm256_setzero_ps();
        sum[1] = _mm256_setzero_ps();
        sum[2] = _mm256_setzero_ps();
        sum[3] = _mm256_setzero_ps();

        if (np) {
            do {
                for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                    ax[j] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(x + i + j * GGML_F32_EPR)));
                    ay[j] = _mm256_loadu_ps(y + i + j * GGML_F32_EPR);
                    sum[j] = _mm256_fmadd_ps(ax[j], ay[j], sum[j]);
                }
    
                i += GGML_F32_STEP;
            } while (i < np);
        }

        if (xn > np) {
            do {
                ax[0] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(x + i)));
                ay[0] = _mm256_loadu_ps(y + i);
                sum[0] = _mm256_fmadd_ps(ax[0], ay[0], sum[0]);
                i += GGML_F32_EPR;
            } while (i < xn);
        }

        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE(sumf, sum);
    }

    // leftovers

    if (nc & (GGML_F32_EPR - 1)) {
        do {
            sumf += GGML_FP16_TO_FP32(x[i]) * y[i];
            i += 1;
        } while (i < nc);
    }

#else

    for (; i < nc; ++i) {
        sumf += GGML_FP16_TO_FP32(x[i]) * y[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__) 

    *s = sumf;
}

void ggml_vec_silu_f32(const int n, float * y, const float * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_silu_f32=" __FUNCTION__)

    uint64_t nc = n;
    uint64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__) && defined(__AVX512DQ__)
#pragma message("Building AVX512F ggml_vec_silu_f32")

    const uint64_t xn = (nc & ~(GGML_F32_EPR16 - 1)); 

    for (; i < xn; i += GGML_F32_EPR16) {
        const __m512 ax = _mm512_loadu_ps(x + i);
        const __m512 ay = ggml_v_silu(ax);
        _mm512_storeu_ps(y + i, ay);
    }

    // leftovers

    if (nc & (GGML_F32_EPR16 - 1)) {
        do {
            y[i] = ggml_silu_f32(x[i]);
            i += 1;
        } while (i < nc);
    }

#elif defined(__AVX2__) && defined(__FMA__)

    const uint64_t xn = (nc & ~(GGML_F32_EPR - 1)); 

    for (; i < xn; i += GGML_F32_EPR) {
        const __m256 ax = _mm256_loadu_ps(x + i);
        const __m256 ay = ggml_v_silu(ax);
        _mm256_storeu_ps(y + i, ay);
    }

    // leftovers

    if (nc & (GGML_F32_EPR - 1)) {
        do {
            y[i] = ggml_silu_f32(x[i]);
            i += 1;
        } while (i < nc);
    }

#else

    for (; i < nc; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__) && defined(__AVX512DQ__)

}

ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
#pragma comment(linker, "/EXPORT:ggml_vec_soft_max_f32=" __FUNCTION__)

    uint64_t nc = n;
    uint64_t i = 0;
    float sumf = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__) && defined(__AVX512DQ__)
#pragma message("Building AVX512F ggml_vec_soft_max_f32")

    const uint64_t xn = (nc & ~(GGML_F32_EPR16 - 1)); 

    if (xn) {
        __m512 vmax = _mm512_set1_ps(max);
        __m512 sum = _mm512_setzero_ps();
        __m512 val;

        do {
            val = _mm512_loadu_ps(x + i);
            val = _mm512_sub_ps(val, vmax);
            val = ggml_v_expf(val);
            _mm512_storeu_ps(y + i, val);
            sum = _mm512_add_ps(sum, val);
            i += GGML_F32_EPR16;
        } while (i < xn);

        // reduce sum

        sumf = _mm512_reduce_add_ps(sum);
    }

    // leftovers

    if (nc & (GGML_F32_EPR16 - 1)) {
        do {
            float val = expf(x[i] - max);
            y[i] = val;
            sumf += val;
            i += 1;
        } while (i < nc);
    }

#elif defined(__AVX2__) && defined(__FMA__)

    const uint64_t xn = (nc & ~(GGML_F32_EPR - 1)); 

    if (xn) {
        __m256 vmax = _mm256_set1_ps(max);
        __m256 sum = _mm256_setzero_ps();
        __m256 val;

        do {
            val = _mm256_loadu_ps(x + i);
            val = _mm256_sub_ps(val, vmax);
            val = ggml_v_expf(val);
            _mm256_storeu_ps(y + i, val);
            sum = _mm256_add_ps(sum, val);
            i += GGML_F32_EPR;
        } while (i < xn);

        // reduce sum

        const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(sum),
                                     _mm256_extractf128_ps(sum, 1));

        const __m128 t1 = _mm_hadd_ps(t0, t0);
        sumf = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    }

    // leftovers

    if (nc & (GGML_F32_EPR - 1)) {
        do {
            float val = expf(x[i] - max);
            y[i] = val;
            sumf += val;
            i += 1;
        } while (i < nc);
    }

#else

    for (; i < nc; ++i) {
        float val = expf(x[i] - max);
        y[i] = val;
        sumf += val;
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__) && defined(__AVX512DQ__)

    return (ggml_float)sumf;
}

ggml_float ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max) {
    // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

    int i = 0;
    ggml_float sum = 0;
    for (; i < n; ++i) {
        float val = x[i] - max;
        y[i] = val;
        sum += (ggml_float)expf(val);
    }
    return sum = (ggml_float)logf(sum);
}

void ggml_vec_max_f32(const uint32_t n, float * s, const float * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_max_f32=" __FUNCTION__)

#ifndef GGML_USE_ACCELERATE
    uint64_t nc = n;
    uint64_t i = 0;
    float max = -INFINITY;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)

    const uint64_t xn = (nc & ~(GGML_F32_EPR16 - 1)); 

    if (xn) {
        __m512 maxvx[GGML_F32_ARR];
        __m512 ax[GGML_F32_ARR];

        const uint64_t np = (nc &~(GGML_F32_STEP16 - 1));

        maxvx[0] = _mm512_set1_ps(max);
        maxvx[1] = _mm512_set1_ps(max);
        maxvx[2] = _mm512_set1_ps(max);
        maxvx[3] = _mm512_set1_ps(max);

        for (; i < np; i += GGML_F32_STEP16) {
            for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
                maxvx[j] = _mm512_max_ps(maxvx[j], ax[j]);
            }
        }

        for (; i < xn; i += GGML_F32_EPR16) {
            ax[0] = _mm512_loadu_ps(x + i);
            maxvx[0] = _mm512_max_ps(maxvx[0], ax[0]);
        }

        // reduce

        GGML_F32_VEC_REDUCE512_MAX(max, maxvx);
    }

    // leftovers

    if (nc & (GGML_F32_EPR16 - 1)) {
        do {
            max = MAX(max, x[i]);
            i += 1;
        } while (i < nc);
    }

#elif defined(__AVX2__)

    const uint64_t xn = (nc & ~(GGML_F32_EPR - 1)); 

    if (xn) {
        GGML_F32_VEC maxvx[GGML_F32_ARR];
        GGML_F32_VEC ax[GGML_F32_ARR];

        const uint64_t np = (nc &~(GGML_F32_STEP - 1));

        maxvx[0] = _mm256_set1_ps(max);
        maxvx[1] = _mm256_set1_ps(max);
        maxvx[2] = _mm256_set1_ps(max);
        maxvx[3] = _mm256_set1_ps(max);

        for (; i < np; i += GGML_F32_STEP) {
            for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
                maxvx[j] = GGML_F32_VEC_MAX(maxvx[j], ax[j]);
            }
        }

        for (; i < xn; i += GGML_F32_EPR) {
            ax[0] = GGML_F32_VEC_LOAD(x + i);
            maxvx[0] = GGML_F32_VEC_MAX(maxvx[0], ax[0]);
        }

        // reduce

        GGML_F32_VEC_REDUCE_MAX(max, maxvx);
    }

    // leftovers

    if (nc & (GGML_F32_EPR - 1)) {
        do {
            max = MAX(max, x[i]);
            i += 1;
        } while (i < nc);
    }

#else

    for (; i < nc; ++i) {
        max = MAX(max, x[i]);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

    *s = max;

#else // GGML_USE_ACCELERATE
    vDSP_maxv(x, 1, s, n);
#endif // GGML_USE_ACCELERATE
}

void ggml_vec_sum_f32(const uint64_t n, float * s, const float * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_sum_f32=" __FUNCTION__)

    float sumf = 0.0f;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_sum_f32 version")

    uint64_t i = 0;
    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    if (xn) {
        __m512 sum[GGML_F32_ARR];
        __m512 ax[GGML_F32_ARR];

        const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

        sum[0] = _mm512_setzero_ps();
        sum[1] = _mm512_setzero_ps();
        sum[2] = _mm512_setzero_ps();
        sum[3] = _mm512_setzero_ps();

        for (; i < np; i += GGML_F32_STEP16) {
            for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
                sum[j] = _mm512_add_ps(ax[j], sum[j]); 
            }
        }

        for (; i < xn; i += GGML_F32_EPR16) {
            ax[0] = _mm512_loadu_ps(x + i);
            sum[0] = _mm512_add_ps(ax[0], sum[0]); 
        }
    
        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE512(sumf, sum);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            sumf += x[i];
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)
#pragma message("Building AVX2 ggml_vec_sum_f32 version")

    uint64_t i = 0;
    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    if (xn) {
        GGML_F32_VEC sum[GGML_F32_ARR];
        GGML_F32_VEC ax[GGML_F32_ARR];

        const uint64_t np = (n & ~(GGML_F32_STEP - 1));

        sum[0] = _mm256_setzero_ps();
        sum[1] = _mm256_setzero_ps();
        sum[2] = _mm256_setzero_ps();
        sum[3] = _mm256_setzero_ps();

        for (; i < np; i += GGML_F32_STEP) {
            for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
                sum[j] = GGML_F32_VEC_ADD(ax[j], sum[j]); 
            }
        }

        for (; i < xn; i += GGML_F32_EPR) {
            ax[0] = GGML_F32_VEC_LOAD(x + i);
            sum[0] = GGML_F32_VEC_ADD(ax[0], sum[0]); 
        }
    
        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE(sumf, sum);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            sumf += x[i];
            i += 1;
        } while (i < n);
    }

#else

    // scalar

    for (uint64_t i = 0; i < n; ++i) {
        sumf += x[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__) 

    *s = sumf;
}

void ggml_vec_sumsq_f32(const uint64_t n, float * s, const float * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_sumsq_f32=" __FUNCTION__)

    float sumf = 0.0f;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_sumsq_f32 version")

    uint64_t i = 0;
    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    if (xn) {
        __m512 sum[GGML_F32_ARR];
        __m512 ax[GGML_F32_ARR];

        const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

        sum[0] = _mm512_setzero_ps();
        sum[1] = _mm512_setzero_ps();
        sum[2] = _mm512_setzero_ps();
        sum[3] = _mm512_setzero_ps();

        for (; i < np; i += GGML_F32_STEP16) {
            for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
                sum[j] = _mm512_fmadd_ps(ax[j], ax[j], sum[j]); 
            }
        }

        for (; i < xn; i += GGML_F32_EPR16) {
            ax[0] = _mm512_loadu_ps(x + i);
            sum[0] = _mm512_fmadd_ps(ax[0], ax[0], sum[0]); 
        }
    
        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE512(sumf, sum);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            sumf += x[i] * x[i];
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;
    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    if (xn) {
        GGML_F32_VEC sum[GGML_F32_ARR];
        GGML_F32_VEC ax[GGML_F32_ARR];

        const uint64_t np = (n & ~(GGML_F32_STEP - 1));

        sum[0] = _mm256_setzero_ps();
        sum[1] = _mm256_setzero_ps();
        sum[2] = _mm256_setzero_ps();
        sum[3] = _mm256_setzero_ps();

        for (; i < np; i += GGML_F32_STEP) {
            for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
                sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ax[j]); 
            }
        }

        for (; i < xn; i += GGML_F32_EPR) {
            ax[0] = GGML_F32_VEC_LOAD(x + i);
            sum[0] = GGML_F32_VEC_FMA(sum[0], ax[0], ax[0]);
        }

        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE(sumf, sum);
    }

    // leftovers

    if (n & (GGML_F32_STEP - 1)) {
        do {
            sumf += x[i] * x[i];
            i += 1;
        } while (i < n);
    }

#else

    // scalar

    for (uint64_t i = 0; i < n; ++i) {
        sumf += x[i] * x[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

    *s = sumf;
}

void ggml_vec_sumsq_bf16(const uint64_t n, float * s, const ggml_bf16_t * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_sumsq_bf16=" __FUNCTION__)

    uint64_t i = 0;
    float sumf = 0.0f;

#if defined(__AVX512F__) && defined(__GEN_AVX512__) && !defined(__gnu_linux__)

    const uint64_t xn = (n & ~(GGML_BF16_EPR32 - 1));

    if (xn) {
        __m512 sum[GGML_F32_ARR];
        __m512i ax[GGML_F32_ARR];

        const uint64_t np = (n & ~(GGML_BF16_STEP32 - 1));

        sum[0] = _mm512_setzero_ps();
        sum[1] = _mm512_setzero_ps();
        sum[2] = _mm512_setzero_ps();
        sum[3] = _mm512_setzero_ps();

        for (; i < np; i += GGML_BF16_STEP32) {
            for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = _mm512_loadu_si512(x + i + j * GGML_BF16_EPR32);
                sum[j] = _mm512_dpbf16_ps(sum[j], ax[j], ax[j]);
            }
        }

        for (; i < xn; i += GGML_BF16_EPR32) {
            ax[0] = _mm512_loadu_si512(x + i);
            sum[0] = _mm512_dpbf16_ps(sum[0], ax[0], ax[0]);
        }

        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE512(sumf, sum);
    }

    // leftovers

    if (n & (GGML_BF16_EPR32 - 1)) {
        do {
            float xc = GGML_BF16_TO_FP32(x[i]);
            sumf += xc * xc;
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__) && !defined(__gnu_linux__)

    const uint64_t xn = (n & ~(GGML_BF16_EPR16 - 1));

    if (xn) {
        __m256 sum[GGML_F32_ARR];
        __m256i ax[GGML_F32_ARR];

        const uint64_t np = (n & ~(GGML_BF16_STEP16 - 1));

        sum[0] = _mm256_setzero_ps();
        sum[1] = _mm256_setzero_ps();
        sum[2] = _mm256_setzero_ps();
        sum[3] = _mm256_setzero_ps();

        for (; i < np; i += GGML_BF16_STEP16) {
            for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = _mm256_loadu_si256((__m256i *)(x + i + j * GGML_BF16_EPR16));
                sum[j] = _mm256_dpbf16_ps(sum[j], ax[j], ax[j]);
            }
        }

        for (; i < xn; i += GGML_BF16_EPR16) {
            ax[0] = _mm256_loadu_si256((__m256i *)(x + i));
            sum[0] = _mm256_dpbf16_ps(sum[0], ax[0], ax[0]);
        }

        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE(sumf, sum);
    }

    // leftovers

    if (n & (GGML_BF16_EPR16 - 1)) {
        do {
            float xc = GGML_BF16_TO_FP32(x[i]);
            sumf += xc * xc;
            i += 1;
        } while (i < n);
    }

#else

    for (; i < n; ++i) {
        float xc = GGML_BF16_TO_FP32(x[i]);
        sumf += xc * xc;
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

    *s = sumf;
}

void ggml_vec_add_f32(const uint64_t n, float * z, const float * x, const float * y) {
#pragma comment(linker, "/EXPORT:ggml_vec_add_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_add_f32 version")

    uint64_t i = 0;

    GGML_F32_VEC512 ax[GGML_F32_ARR];
    GGML_F32_VEC512 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD512(x + i + j * GGML_F32_EPR16);
            ay[j] = GGML_F32_VEC_LOAD512(y + i + j * GGML_F32_EPR16);
            ay[j] = GGML_F32_VEC_ADD512(ax[j], ay[j]);
            GGML_F32_VEC_STORE512(z + i + j * GGML_F32_EPR16, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = GGML_F32_VEC_LOAD512(x + i);
        ay[0] = GGML_F32_VEC_LOAD512(y + i);
        ay[0] = GGML_F32_VEC_ADD512(ax[0], ay[0]);
        GGML_F32_VEC_STORE512(z + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            z[i]  = x[i] + y[i];
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_ADD(ax[j], ay[j]);
            GGML_F32_VEC_STORE(z + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = GGML_F32_VEC_LOAD(x + i);
        ay[0] = GGML_F32_VEC_LOAD(y + i);
        ay[0] = GGML_F32_VEC_ADD(ax[0], ay[0]);
        GGML_F32_VEC_STORE(z + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            z[i]  = x[i] + y[i];
            i += 1;
        } while (i < n);
    }

#else

    for (uint64_t i = 0; i < n; ++i) {
        z[i]  = x[i] + y[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_add1_f32(const uint64_t n, float * z, const float * x, const float v) {
#pragma comment(linker, "/EXPORT:ggml_vec_add1_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_add1_f32 version")

    uint64_t i = 0;

    __m512 ax[GGML_F32_ARR];
    __m512 vy = _mm512_set1_ps(v);

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            ax[j] = _mm512_add_ps(ax[j], vy);
            _mm512_storeu_ps(z + i + j * GGML_F32_EPR16, ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        ax[0] = _mm512_add_ps(ax[0], vy);
        _mm512_storeu_ps(z + i, ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            z[i]  = x[i] + v;
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);
    GGML_F32_VEC ax[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
            ax[j] = GGML_F32_VEC_ADD(ax[j], vx);
            GGML_F32_VEC_STORE(z + i + j * GGML_F32_EPR, ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = GGML_F32_VEC_LOAD(x + i);
        ax[0] = GGML_F32_VEC_ADD(ax[0], vx);
        GGML_F32_VEC_STORE(z + i, ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            z[i]  = x[i] + v;
            i += 1;
        } while (i < n);
    }

#else

    for (uint64_t i = 0; i < n; ++i) {
        z[i]  = x[i] + v;
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__) 

}

void ggml_vec_acc_f32(const uint64_t n, float * y, const float * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_acc_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_acc_f32 version")

    uint64_t i = 0;

    __m512 ax[GGML_F32_ARR];
    __m512 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_loadu_ps(y + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_add_ps(ax[j], ay[j]);
            _mm512_storeu_ps(y + i + j * GGML_F32_EPR16, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        ay[0] = _mm512_loadu_ps(y + i);
        ay[0] = _mm512_add_ps(ax[0], ay[0]);
        _mm512_storeu_ps(y + i, ay[0]);
    }


    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            y[i] += x[i];
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_ADD(ax[j], ay[j]);
            GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = GGML_F32_VEC_LOAD(x + i);
        ay[0] = GGML_F32_VEC_LOAD(y + i);
        ay[0] = GGML_F32_VEC_ADD(ax[0], ay[0]);
        GGML_F32_VEC_STORE(y + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            y[i] += x[i];
            i += 1;
        } while (i < n);
    }

#else

    for (uint64_t i = 0; i < n; ++i) {
        y[i] += x[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_acc1_f32(const uint64_t n, float * y, const float v) {
#pragma comment(linker, "/EXPORT:ggml_vec_acc1_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_acc1_f32 version")

    uint64_t i = 0;

    __m512 vx = _mm512_set1_ps(v); 
    __m512 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = _mm512_loadu_ps(y + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_add_ps(ay[j], vx);
            _mm512_storeu_ps(y + i + j * GGML_F32_EPR16, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ay[0] = _mm512_loadu_ps(y + i);
        ay[0] = _mm512_add_ps(ay[0], vx);
        _mm512_storeu_ps(y + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            y[i] += v;
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);
    GGML_F32_VEC ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_ADD(ay[j], vx);
            GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ay[0] = GGML_F32_VEC_LOAD(y + i);
        ay[0] = GGML_F32_VEC_ADD(ay[0], vx);
        GGML_F32_VEC_STORE(y + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            y[i] += v;
            i += 1;
        } while (i < n);
    }

#else

    for (uint64_t i = 0; i < n; ++i) {
        y[i] += v;
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_sub_f32(const uint64_t n, float * z, const float * x, const float * y) {
#pragma comment(linker, "/EXPORT:ggml_vec_sub_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_sub_f32 version")

    uint64_t i = 0;

    __m512 ax[GGML_F32_ARR];
    __m512 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_loadu_ps(y + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_sub_ps(ax[j], ay[j]);
            _mm512_storeu_ps(z + i + j * GGML_F32_EPR16, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        ay[0] = _mm512_loadu_ps(y + i);
        ay[0] = _mm512_sub_ps(ax[0], ay[0]);
        _mm512_storeu_ps(z + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            z[i]  = x[i] - y[i];
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_SUB(ax[j], ay[j]);
            GGML_F32_VEC_STORE(z + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = GGML_F32_VEC_LOAD(x + i);
        ay[0] = GGML_F32_VEC_LOAD(y + i);
        ay[0] = GGML_F32_VEC_SUB(ax[0], ay[0]);
        GGML_F32_VEC_STORE(z + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            z[i]  = x[i] - y[i];
            i += 1;
        } while (i < n);
    }

#else

    for (uint64_t i = 0; i < n; ++i) {
        z[i]  = x[i] - y[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_set_f32(const uint64_t n, float * x, const float v) {
#pragma comment(linker, "/EXPORT:ggml_vec_set_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_set_f32 version")

    uint64_t i = 0;

    __m512 vx = _mm512_set1_ps(v);

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            _mm512_storeu_ps(x + i + j * GGML_F32_EPR16, vx);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        _mm512_storeu_ps(x + i, vx);
    }

    // left overs

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            x[i] = v;
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            GGML_F32_VEC_STORE(x + i + j * GGML_F32_EPR, vx);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        GGML_F32_VEC_STORE(x + i, vx);
    }

    // left overs

    if (n & (GGML_F32_EPR - 1)) {
        do {
            x[i] = v;
            i+= 1;
        } while (i < n);
    }

#else

    for (uint64_t i = 0; i < n; ++i) {
        x[i]  = v;
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_cpy_f32(const uint64_t n, float * y, const float * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_cpy_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_cpy_f32 version")

    uint64_t i = 0;

    __m512 ax[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            _mm512_storeu_ps(y + i + j * GGML_F32_EPR16, ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        _mm512_storeu_ps(y + i, ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            y[i] = x[i];
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    GGML_F32_VEC ax[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
            GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = GGML_F32_VEC_LOAD(x + i);
        GGML_F32_VEC_STORE(y + i, ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            y[i] = x[i];
            i += 1;
        } while (i < n);
    }

#else

    for (uint64_t i = 0; i < n; ++i) {
        y[i]  = x[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_neg_f32(const uint64_t n, float * y, const float * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_neg_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_neg_f32 version")

    uint64_t i = 0;
    const uint32_t xor_pat = 0x80000000;

    __m512 vx = _mm512_set1_ps(*(float *)&xor_pat);
    __m512 ax[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            ax[j] = _mm512_xor_ps(vx, ax[j]);
            _mm512_storeu_ps(y + i + j * GGML_F32_EPR16, ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        ax[0] = _mm512_xor_ps(vx, ax[0]);
        _mm512_storeu_ps(y + i, ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            y[i] = -x[i];
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;
    const uint32_t xor_pat = 0x80000000;

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(*(float *)&xor_pat);
    GGML_F32_VEC ax[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
            ax[j] = GGML_F32_VEC_XOR(vx, ax[j]);
            GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = GGML_F32_VEC_LOAD(x + i);
        ax[0] = GGML_F32_VEC_XOR(vx, ax[0]);
        GGML_F32_VEC_STORE(y + i, ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            y[i] = -x[i];
            i += 1;
        } while (i < n);
    }

#else

    for (uint64_t i = 0; i < n; ++i) {
        y[i] = -x[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_mul_f32(const uint32_t n, float * z, const float * x, const float * y) {
#pragma comment(linker, "/EXPORT:ggml_vec_mul_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_mul_f32 version")

    uint64_t i = 0;

    GGML_F32_VEC512 ax[GGML_F32_ARR];
    GGML_F32_VEC512 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD512(x + i + j * GGML_F32_EPR16);
            ay[j] = GGML_F32_VEC_LOAD512(y + i + j * GGML_F32_EPR16);
            ay[j] = GGML_F32_VEC_MUL512(ax[j], ay[j]);
            GGML_F32_VEC_STORE512(z + i + j * GGML_F32_EPR16, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = GGML_F32_VEC_LOAD512(x + i);
        ay[0] = GGML_F32_VEC_LOAD512(y + i);
        ay[0] = GGML_F32_VEC_MUL512(ax[0], ay[0]);
        GGML_F32_VEC_STORE512(z + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            z[i]  = x[i] * y[i];
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            ax[j] = GGML_F32_VEC_MUL(ax[j], ay[j]);
            GGML_F32_VEC_STORE(z + i + j * GGML_F32_EPR, ax[j]); 
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1)); 

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = GGML_F32_VEC_LOAD(x + i);
        ay[0] = GGML_F32_VEC_LOAD(y + i);
        ax[0] = GGML_F32_VEC_MUL(ax[0], ay[0]);
        GGML_F32_VEC_STORE(z + i, ax[0]); 
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {

        do {
            z[i] = x[i] * y[i];
            i += 1;
        } while (i < n);
    }

#else

    // scalar

    for (uint64_t i = 0; i < n; ++i) {
        z[i]  = x[i] * y[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_mul1_f32(const uint64_t n, float * z, const float * x, const float v) {
#pragma comment(linker, "/EXPORT:ggml_vec_mul1_f32=" __FUNCTION__)

    uint64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)

    __m512 vx = _mm512_set1_ps(v);
    __m512 ax[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            ax[j] = _mm512_mul_ps(ax[j], vx);
            _mm512_storeu_ps(z + i + j * GGML_F32_EPR16, ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        ax[0] = _mm512_mul_ps(ax[0], vx);
        _mm512_storeu_ps(z + i, ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            z[i]  = x[i] * v;
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    __m256 vx = GGML_F32_VEC_SET1(v);
    __m256 ax[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm256_loadu_ps(x + i + j * GGML_F32_EPR);
            ax[j] = _mm256_mul_ps(ax[j], vx);
            _mm256_storeu_ps(z + i + j * GGML_F32_EPR, ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = _mm256_loadu_ps(x + i);
        ax[0] = _mm256_mul_ps(ax[0], vx);
        _mm256_storeu_ps(z + i, ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            z[i]  = x[i] * v;
            i += 1;
        } while (i < n);
    }

#else

    for (; i < n; ++i) {
        z[i]  = x[i] * v;
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__) 

}

void ggml_vec_div_f32(const uint64_t n, float * z, const float * x, const float * y) {
#pragma comment(linker, "/EXPORT:ggml_vec_div_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_div_f32 version")

    uint64_t i = 0;

    __m512 ax[GGML_F32_ARR];
    __m512 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_loadu_ps(y + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_div_ps(ax[j], ay[j]);
            _mm512_storeu_ps(z + i + j * GGML_F32_EPR16, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        ay[0] = _mm512_loadu_ps(y + i);
        ay[0] = _mm512_div_ps(ax[0], ay[0]);
        _mm512_storeu_ps(z + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            z[i]  = x[i] / y[i];
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_DIV(ax[j], ay[j]);
            GGML_F32_VEC_STORE(z + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = GGML_F32_VEC_LOAD(x + i);
        ay[0] = GGML_F32_VEC_LOAD(y + i);
        ay[0] = GGML_F32_VEC_DIV(ax[0], ay[0]);
        GGML_F32_VEC_STORE(z + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            z[i]  = x[i] / y[i];
            i += 1;
        } while (i < n);
    }

#else

    for (uint64_t i = 0; i < n; ++i) {
        z[i]  = x[i] / y[i];
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_normsq_f32(const uint64_t n, float * s, const float mean, float * y, const float * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_normsq_f32=" __FUNCTION__)

    float sumf = 0.0f;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_normsq_f32 version")

    uint64_t i = 0;
    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1)); 

    if (xn) {
        __m512 vx = _mm512_set1_ps(mean);
        __m512 sum[GGML_F32_ARR];
        __m512 ax[GGML_F32_ARR];

        const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

        sum[0] = _mm512_setzero_ps(); 
        sum[1] = _mm512_setzero_ps(); 
        sum[2] = _mm512_setzero_ps(); 
        sum[3] = _mm512_setzero_ps();

        for (; i < np; i += GGML_F32_STEP16) {
            for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
                ax[j] = _mm512_sub_ps(ax[j], vx);
                _mm512_storeu_ps(y + i + j * GGML_F32_EPR16, ax[j]);
                sum[j] = _mm512_fmadd_ps(ax[j], ax[j], sum[j]);
            }
        }

        for (; i < xn; i += GGML_F32_EPR16) {
            ax[0] = _mm512_loadu_ps(x + i);
            ax[0] = _mm512_sub_ps(ax[0], vx);
            _mm512_storeu_ps(y + i, ax[0]);
            sum[0] = _mm512_fmadd_ps(ax[0], ax[0], sum[0]);
        }

        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE512(sumf, sum); 
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        float bx;

        do {
            bx = x[i] - mean;
            y[i] = bx;
            sumf += bx * bx;
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;
    const uint64_t xn = (n & ~(GGML_F32_EPR - 1)); 

    if (xn) {
        __m256 vx = _mm256_set1_ps(mean);
        __m256 sum[GGML_F32_ARR];
        __m256 ax[GGML_F32_ARR];

        const uint64_t np = (n & ~(GGML_F32_STEP - 1));

        sum[0] = _mm256_setzero_ps(); 
        sum[1] = _mm256_setzero_ps(); 
        sum[2] = _mm256_setzero_ps(); 
        sum[3] = _mm256_setzero_ps();

        for (; i < np; i += GGML_F32_STEP) {
            for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = _mm256_loadu_ps(x + i + j * GGML_F32_EPR);
                ax[j] = _mm256_sub_ps(ax[j], vx);
                _mm256_storeu_ps(y + i + j * GGML_F32_EPR, ax[j]);
                sum[j] = _mm256_fmadd_ps(ax[j], ax[j], sum[j]);
            }
        }

        for (; i < xn; i += GGML_F32_EPR) {
            ax[0] = _mm256_loadu_ps(x + i);
            ax[0] = _mm256_sub_ps(ax[0], vx);
            _mm256_storeu_ps(y + i, ax[0]);
            sum[0] = _mm256_fmadd_ps(ax[0], ax[0], sum[0]);
        }

        // reduce sum0..sum3 to sumf

        GGML_F32_VEC_REDUCE(sumf, sum); 
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        float bx;

        do {
            bx = x[i] - mean;
            y[i] = bx;
            sumf += bx * bx;
            i += 1;
        } while (i < n);
    }

#else

    // scaler

    for (uint64_t i = 0; i < n; ++i) {
        float bx;

        bx = x[i] - mean;
        y[i] = bx;
        sumf += bx * bx;
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__) 

    *s =sumf;
}

void ggml_vec_sqrt_f32(const uint64_t n, float * y, const float * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_sqrt_f32=" __FUNCTION__)

    uint64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_sqrt_f32 version")

    __m512 ax[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            ax[j] = _mm512_sqrt_ps(ax[j]);
            _mm512_storeu_ps((y + i + j * GGML_F16_EPR16), ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        ax[0] = _mm512_sqrt_ps(ax[0]);
        _mm512_storeu_ps((y + i), ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            y[i]  = sqrtf(x[i]);
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    __m256 ax[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm256_loadu_ps(x + i + j * GGML_F32_EPR);
            ax[j] = _mm256_sqrt_ps(ax[j]);
            _mm256_storeu_ps((y + i + j * GGML_F16_EPR), ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = _mm256_loadu_ps(x + i);
        ax[0] = _mm256_sqrt_ps(ax[0]);
        _mm256_storeu_ps((y + i), ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            y[i]  = sqrtf(x[i]);
            i += 1;
        } while (i < n);
    }

#else

    for (i = 0; i < n; ++i) {
        y[i] = sqrtf(x[i]);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_abs_f32(const uint64_t n, float * y, const float * x) {
#pragma comment(linker, "/EXPORT:ggml_vec_abs_f32=" __FUNCTION__)

    uint64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_abs_f32 version")

    __m512 ax[GGML_F32_ARR];
    const __m512 signBit = _mm512_set1_ps(-0.0f);

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            ax[j] = _mm512_andnot_ps(signBit, ax[j]);
            _mm512_storeu_ps((y + i + j * GGML_F16_EPR16), ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        ax[0] = _mm512_andnot_ps(signBit, ax[0]);
        _mm512_storeu_ps((y + i), ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            y[i]  = fabsf(x[i]);
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    __m256 ax[GGML_F32_ARR];
    const __m256 signBit = _mm256_set1_ps(-0.0f);

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm256_loadu_ps(x + i + j * GGML_F32_EPR);
            ax[j] = _mm256_andnot_ps(signBit, ax[j]);
            _mm256_storeu_ps((y + i + j * GGML_F16_EPR), ax[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = _mm256_loadu_ps(x + i);
        ax[0] = _mm256_andnot_ps(signBit, ax[0]);
        _mm256_storeu_ps((y + i), ax[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            y[i]  = fabsf(x[i]);
            i += 1;
        } while (i < n);
    }

#else

    for (i = 0; i < n; ++i) {
        y[i] = fabsf(x[i]);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_mad_f32(const uint64_t n, float * GGML_RESTRICT y, const float * GGML_RESTRICT x, const float v) {
#pragma comment(linker, "/EXPORT:ggml_vec_mad_f32=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_mad_f32 version")

    uint64_t i = 0;

    __m512 vx = _mm512_set1_ps(v);
    __m512 ax[GGML_F32_ARR];
    __m512 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_loadu_ps(y + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_fmadd_ps(ax[j], vx, ay[j]);
            _mm512_storeu_ps(y + i + j * GGML_F32_EPR16, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        ay[0] = _mm512_loadu_ps(y + i);
        ay[0] = _mm512_fmadd_ps(ax[0], vx, ay[0]);
        _mm512_storeu_ps(y + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 -1)) {
        do {
            y[i] += x[i]*v;
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    __m256 vx = _mm256_set1_ps(v);
    __m256 ax[GGML_F32_ARR];
    __m256 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm256_loadu_ps(x + i + j * GGML_F32_EPR);
            ay[j] = _mm256_loadu_ps(y + i + j * GGML_F32_EPR);
            ay[j] = _mm256_fmadd_ps(ax[j], vx, ay[j]);
            _mm256_storeu_ps(y + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (;i < xn ; i += GGML_F32_EPR) {
        ax[0] = _mm256_loadu_ps(x + i);
        ay[0] = _mm256_loadu_ps(y + i);
        ay[0] = _mm256_fmadd_ps(ax[0], vx, ay[0]);
        _mm256_storeu_ps(y + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            y[i] += x[i]*v;
            i+= 1;
        } while (i < n);
    }

#else

    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_mad_f16(const uint64_t n, ggml_fp16_t * GGML_RESTRICT y, const ggml_fp16_t * GGML_RESTRICT x, const float v) {
#pragma comment(linker, "/EXPORT:ggml_vec_mad_f16=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_mad_f16 version")

    uint64_t i = 0;

    __m512 vx = _mm512_set1_ps(v);
    __m512 ax[GGML_F32_ARR];
    __m512 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F16_STEP16 - 1));

    for (; i < np; i += GGML_F16_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i + j * GGML_F16_EPR16)));
            ay[j] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(y + i + j * GGML_F16_EPR16)));
            ay[j] = _mm512_fmadd_ps(ax[j], vx, ay[j]);
            _mm256_storeu_si256((__m256i *)(y + i + j * GGML_F16_EPR16), _mm512_cvtps_ph(ay[j], 0));
        }
    }

    const uint64_t xn = (n & ~(GGML_F16_EPR16 - 1));

    for (; i < xn; i += GGML_F16_EPR16) {
        ax[0] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i)));
        ay[0] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(y + i)));
        ay[0] = _mm512_fmadd_ps(ax[0], vx, ay[0]);
        _mm256_storeu_si256((__m256i *)(y + i), _mm512_cvtps_ph(ay[0], 0));
    }

    // leftovers

    if (n & (GGML_F16_EPR16 - 1)) {
        do {
            y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) + GGML_FP16_TO_FP32(x[i])*v);
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    __m256 vx = _mm256_set1_ps(v);
    __m256 ax[GGML_F32_ARR];
    __m256 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F16_STEP - 1));

    for (; i < np; i += GGML_F16_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i + j * GGML_F16_EPR)));
            ay[j] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(y + i + j * GGML_F16_EPR)));
            ay[j] = _mm256_fmadd_ps(ax[j], vx, ay[j]);
            _mm_storeu_si128((__m128i *)(y + i + j * GGML_F16_EPR), _mm256_cvtps_ph(ay[j], 0));
        }
    }

    const uint64_t xn = (n & ~(GGML_F16_EPR - 1));

    for (; i < xn; i += GGML_F16_EPR) {
       ax[0] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i)));
       ay[0] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(y + i)));
       ay[0] = _mm256_fmadd_ps(ax[0], vx, ay[0]);
       _mm_storeu_si128((__m128i *)(y + i), _mm256_cvtps_ph(ay[0], 0));
    }

    // leftovers

    if (n & (GGML_F16_EPR - 1)) {
        do {
            y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) + GGML_FP16_TO_FP32(x[i])*v);
            i += 1;
        } while (i < n);
    }

#else

    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) + GGML_FP16_TO_FP32(x[i])*v);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_vec_scale_f32(const uint64_t n, float * y, const float   v) {
#pragma comment(linker, "/EXPORT:ggml_vec_scale_f32=" __FUNCTION__)

#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_scale_f32 version")

    uint64_t i = 0;

    __m512 vx = _mm512_set1_ps(v);
    __m512 ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = _mm512_loadu_ps(y + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_mul_ps(ay[j], vx);
            _mm512_storeu_ps(y + i + j * GGML_F32_EPR16, ay[j]);
        } 
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ay[0] = _mm512_loadu_ps(y + i);
        ay[0] = _mm512_mul_ps(ay[0], vx);
        _mm512_storeu_ps(y + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR16 - 1)) {
        do {
            y[i] *= v;
            i += 1;
        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);
    GGML_F32_VEC ay[GGML_F32_ARR];

    const uint64_t np = (n & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_MUL(ay[j], vx);
            GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
        } 
    }

    const uint64_t xn = (n & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ay[0] = GGML_F32_VEC_LOAD(y + i);
        ay[0] = GGML_F32_VEC_MUL(ay[0], vx);
        GGML_F32_VEC_STORE(y + i, ay[0]);
    }

    // leftovers

    if (n & (GGML_F32_EPR - 1)) {
        do {
            y[i] *= v;
            i += 1;
        } while (i < n);
    }

#else

    // scalar
    for (int64_t i = 0; i < n; ++i) {
        y[i] *= v;
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__) 

}

void ggml_vec_scale_f16(const uint64_t n, ggml_fp16_t * y, const float v) {
#pragma comment(linker, "/EXPORT:ggml_vec_scale_f16=" __FUNCTION__)

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_vec_scale_f16 version")

    uint64_t i = 0;

    __m512 vx = _mm512_set1_ps(v);
    __m512 ay[GGML_F16_ARR];
    __m256i az[GGML_F16_ARR];

    const uint64_t np = (n & ~(GGML_F16_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F16_ARR; j++) {
            ay[j] = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(y + i + j * GGML_F16_EPR16)));
            ay[j] = _mm512_mul_ps(ay[j], vx);
            az[j] = _mm512_cvtps_ph(ay[j], _MM_FROUND_TO_NEAREST_INT);
            _mm256_storeu_si256((__m256i *)(y + i + j * GGML_F16_EPR16), az[j]);
        }
    }

    const uint64_t xn = (n & ~(GGML_F16_EPR16 - 1));

    for (; i < xn; i += GGML_F16_EPR16) {
        ay[0] = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(y + i)));
        ay[0] = _mm512_mul_ps(ay[0], vx);
        az[0] = _mm512_cvtps_ph(ay[0], _MM_FROUND_TO_NEAREST_INT);  
        _mm256_storeu_si256((__m256i *)(y + i), az[0]); 
    }

    // leftovers

    if (n & (GGML_F16_EPR16 - 1)) {
        do {
            y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) * v);
            i += 1;

        } while (i < n);
    }

#elif defined(__AVX2__)

    uint64_t i = 0;

    __m256 vx = _mm256_set1_ps(v);
    __m256 ay[GGML_F16_ARR];
    __m128i az[GGML_F16_ARR];

    const uint64_t np = (n & ~(GGML_F16_STEP - 1));

    for (; i < np; i += GGML_F16_STEP) {
        for (uint64_t j = 0; j < GGML_F16_ARR; j++) {
            ay[j] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(y + i + j * GGML_F16_EPR)));
            ay[j] = _mm256_mul_ps(ay[j], vx);
            az[j] = _mm256_cvtps_ph(ay[j], _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i + j * GGML_F16_EPR), az[j]); 
        } 
    }

    const uint64_t xn = (n & ~(GGML_F16_EPR - 1));

    for (; i < xn; i += GGML_F16_EPR) {
        ay[0] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(y + i)));
        ay[0] = _mm256_mul_ps(ay[0], vx);
        az[0] = _mm256_cvtps_ph(ay[0], _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), az[0]);
    }

    // leftovers

    if (n & (GGML_F16_EPR - 1)) {
        do {
            y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) * v);
            i += 1;
        } while (i < n);
    }

#else

    // scalar
    for (int64_t i = 0; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) * v);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__) 

}

