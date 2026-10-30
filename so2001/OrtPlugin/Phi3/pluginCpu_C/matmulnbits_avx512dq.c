// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.
//
// AVX512DQ-specialized reductions for kernels whose AVX512F implementation
// otherwise needs three 128-bit extracts. Callers must gate entry on
// HasAvx512Dq(); keeping these functions in their own translation unit prevents
// DQ instructions from leaking into the AVX512F fallback.

#include "matmulnbits_avx512.h"

#include <immintrin.h>
#include "fp16_utils.h"

#if defined(_MSC_VER)
#define FUSIONCPU_NOINLINE __declspec(noinline)
#else
#define FUSIONCPU_NOINLINE __attribute__((noinline))
#endif

static inline float ReduceAddAvx512Dq(__m512 v) {
    const __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(v),
                                        _mm512_extractf32x8_ps(v, 1));

    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(sum256),
                            _mm256_extractf128_ps(sum256, 1));

    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    return _mm_cvtss_f32(_mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55)));
}

FUSIONCPU_NOINLINE float DotFp32Avx512Dq(const float* a, const float* w, int64_t k) {
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
    return ReduceAddAvx512Dq(
        _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
}

FUSIONCPU_NOINLINE float GqaDotF16Avx512Dq(const float* q, const uint16_t* k, int64_t hs) {
    __m512 acc = _mm512_setzero_ps();
    int64_t d = 0;
    for (; d + 16 <= hs; d += 16) {
        const __m256i kh = _mm256_loadu_si256((const __m256i*)(k + d));
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + d), _mm512_cvtph_ps(kh), acc);
    }
    float sum = ReduceAddAvx512Dq(acc);
    for (; d < hs; ++d) sum += q[d] * FusionCpuHalfToFloat(k[d]);
    return sum;
}

