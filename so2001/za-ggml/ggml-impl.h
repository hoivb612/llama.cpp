#pragma once

#include "ggml.h"

// GGML internal header

#include <assert.h>
#include <stdlib.h>

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef __cplusplus
extern "C" {
#endif

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976

#ifndef __cplusplus
#ifndef static_assert
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#else
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif
#endif
#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512

#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#endif

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available

#if defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ifndef __SSE3__
#define __SSE3__
#endif
#ifndef __AVXVNNIINT8__
//#define __AVXVNNIINT8__
#endif
#ifndef __SSSE3__
#define __SSSE3__
#endif
#endif

#if defined(_MSC_VER) && defined(__F16C__)

#define m512bh(p) p
#define m512i(p) p

#include <intrin.h>

inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;

    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

/**
 * Converts float32 to brain16.
 *
 * Subnormals are flushed to zero, and NANs are quiet.
 */

inline ggml_bf16_t ggml_compute_fp32_to_bf16(float s) {
    ggml_bf16_t h;

    h.bits = (uint16_t)_mm_extract_epi16(_mm_cvtneps_pbh(_mm_set_ss(s)), 0);
    return h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

#define GGML_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define GGML_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)

#endif

// precomputed f32 table for f16 (256 KB)
// defined in ggml.c, initialized in ggml_init()

extern float ggml_table_f32_f16[1 << 16];

inline float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
    uint16_t s;
    s = *(uint16_t *)&f;
    return ggml_table_f32_f16[s];
}

#define GGML_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)

#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#define GGML_HASHTABLE_FULL ((size_t)-1)
#define GGML_HASHTABLE_ALREADY_EXISTS ((size_t)-2)

struct ggml_hash_set ggml_hash_set_new(size_t size);

bool ggml_hash_contains(const struct ggml_hash_set hash_set, struct ggml_tensor * key);

// returns GGML_HASHTABLE_FULL if table is full, otherwise the current index of the key
// or where it should be inserted

size_t ggml_hash_find(const struct ggml_hash_set hash_set, struct ggml_tensor * key);

// returns GGML_HASHTABLE_ALREADY_EXISTS if key already exists, index otherwise, asserts
// if table is full.

size_t ggml_hash_insert(struct ggml_hash_set hash_set, struct ggml_tensor * key);

// return index, asserts if table is full

size_t ggml_hash_find_or_insert(struct ggml_hash_set hash_set, struct ggml_tensor * key);

#ifdef __cplusplus
}
#endif
