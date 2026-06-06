// SPDX-FileCopyrightText: Copyright 2024 Arm Ltd.
#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

//
// for tensor repacking
//

typedef enum {
    TENSOR_REPACKING_MODE_NONE = 0,
    TENSOR_REPACKING_MODE_GGML = 1,
    TENSOR_REPACKING_MODE_XBOX = 2,
    TENSOR_REPACKING_MODE_XBCG = 3,
    TENSOR_REPACKING_MODE_MAX  = 4
} ggml_tensor_repacking_mode_t;

typedef struct {
    ggml_half d[8];                     // delta (scale)
    int8_t qs[QK_K / 2];                // quants interleaved packed two per byte
} block_q4_0_repack;

C_ASSERT((sizeof(block_q4_0) * 8) == sizeof(block_q4_0_repack));

typedef struct {
    ggml_half d[8];                     // delta (scale)
    int8_t qs[QK_K];                    // quants interleaved
} block_q8_0_repack;

C_ASSERT((sizeof(block_q8_0) * 8) == sizeof(block_q8_0_repack));

typedef block_q2_K block_q2_K_repack;
typedef block_q3_K block_q3_K_repack;
typedef block_q4_K block_q4_K_repack;
typedef block_q6_K block_q6_K_repack;
typedef block_q8_K block_q8_K_repack;

extern ggml_tensor_repacking_mode_t tensor_repacking_mode;

ggml_tensor_repacking_mode_t ggml_tensor_repacking_mode();

void ggml_set_tensor_repacking_mode(ggml_tensor_repacking_mode_t type);

void
ggml_repack_tensor(
    const struct ggml_compute_params * params,
    struct ggml_tensor *tensor);

void
ggml_wait_for_done(
    const struct ggml_compute_params * params);

void
ggml_wait_to_finalize(
    const struct ggml_compute_params * params);

// vec_dot routines for Xbox repacked tensors

void
xx_vec_dot_q4_0_q8_0_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q4_0_repack * x,
    size_t bx,
    const block_q8_0_repack * y,
    size_t ncols,
    uint32_t nrows);

void
xx_vec_dot_q2_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q2_K_repack * x,
    size_t bx,
    const block_q8_K_repack * y,
    size_t ncols,
    uint32_t nrows);

void
xx_vec_dot_q3_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q3_K_repack * x,
    size_t bx,
    const block_q8_K_repack * y,
    size_t ncols,
    uint32_t nrows);

void
xx_vec_dot_q4_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q4_K_repack * x,
    size_t bx,
    const block_q8_K_repack * y,
    size_t ncols,
    uint32_t nrows);

void
xx_vec_dot_q6_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q6_K_repack * x,
    size_t bx,
    const block_q8_K_repack * y,
    size_t ncols,
    uint32_t nrows);

void
xx_vec_dot_q8_0_q8_0_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q8_0_repack * x,
    size_t bx,
    const block_q8_0_repack * y,
    size_t ncols,
    uint32_t nrows);

void                   
quantize_row_q2_k_x8 (
    const float * x,
    block_q2_K * y,
    uint32_t vec_size);

void                   
quantize_row_q3_k_x8 (
    const float * x,
    block_q3_K * y,
    uint32_t vec_size);

void                   
quantize_row_q4_k_x8 (
    const float * x,
    block_q4_K * y,
    uint64_t vec_size);

void                   
quantize_row_q6_k_x8 (
    const float * x,
    block_q6_K * y,
    uint64_t vec_size);

void                   
quantize_row_q236_k_q8_k_x8 (
    const float * x,
    block_q8_K * y,
    uint32_t vec_size);

void                   
quantize_row_q4_k_q8_k_x8 (
    const float * x,
    block_q8_K * y,
    uint64_t vec_size);

void
quantize_row_q8_0_x8 (
    const float * x,
    block_q8_0 * y,
    uint64_t vec_size);

#ifdef __cplusplus
}
#endif
