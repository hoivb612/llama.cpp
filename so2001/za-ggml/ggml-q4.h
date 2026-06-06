// SPDX-FileCopyrightText: Copyright 2024 Arm Ltd.
#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void from_float_to_q8_0_4_8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void from_float_to_q8_K_4_8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

// GEMV
void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);

// GEMM
void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);

bool           ggml_aarch64_repack_tensor(struct ggml_tensor * cur, enum ggml_type repack_type, const void * data, size_t data_size);
enum ggml_type ggml_aarch64_get_optimal_repack_type(const struct ggml_tensor * cur);

#ifdef __cplusplus
}
#endif

