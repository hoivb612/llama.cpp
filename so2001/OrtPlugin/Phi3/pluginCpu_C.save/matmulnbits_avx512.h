// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#define FUSIONCPU_COMPACT_MB_MAX 4

void DequantColumnAvx512(float* out, const uint8_t* b_row, const float* scales_row,
                         const uint8_t* zp_row, bool has_zp, int64_t k_blocks,
                         int64_t block_size, int default_zp);
float DotFp32Avx512(const float* a, const float* w, int64_t k);
float DotInt8VnniColumn(const uint8_t* au8, const float* ascale, const int8_t* wq8,
                        const float* wscale, const int32_t* wsum, int64_t k_blocks,
                        int64_t block_size);
void RepackWeightColumnInt8Avx512(int8_t* wq8_out, float* wscale_out, int32_t* wsum_out,
                                  const uint8_t* b_row, const float* scales_row,
                                  const uint8_t* zp_row, bool has_zp, int64_t k_blocks,
                                  int64_t block_size, int default_zp);
void DotInt8VnniTile16Avx512(float* out16, const uint8_t* au8_row, const float* ascale,
                             const int8_t* wq8p, const float* wscale_packed,
                             const int32_t* wsum_packed, int64_t k_blocks,
                             int64_t block_size);
void RepackWeightTile16Int8Avx512(int8_t* wq8p, float* wscale_packed,
                                  int32_t* wsum_packed, const uint8_t* b_base,
                                  int64_t b_stride, const float* sc_base,
                                  int64_t sc_stride, const uint8_t* zp_base,
                                  int64_t zp_stride, bool has_zp, int cnt,
                                  int64_t k_blocks, int64_t block_size, int default_zp);
void RepackWeightTile16Int4Avx512(uint8_t* wq4p, float* wscale_packed,
                                  int32_t* wsum_packed, const uint8_t* b_base,
                                  int64_t b_stride, const float* sc_base,
                                  int64_t sc_stride, int cnt, int64_t k_blocks,
                                  int64_t block_size, int default_zp);
void DotInt8VnniTile16Int4Avx512(float* out16, const uint8_t* au8_row,
                                 const float* ascale, const uint8_t* wq4p,
                                 const float* wscale_packed,
                                 const int32_t* wsum_packed, int64_t k_blocks,
                                 int64_t block_size);
void DotInt8VnniTile16Int4CompactAvx512(float* out16, const uint8_t* au8_row,
                                        const float* ascale, const uint8_t* wq4p,
                                        const uint16_t* wscale_h,
                                        const int16_t* wsum16, int64_t k_blocks,
                                        int64_t block_size);
void DotInt8VnniTile16Int4CompactMBAvx512(float* out, const uint8_t* au8_base,
                                          int64_t au8_row_stride,
                                          const float* ascale_base,
                                          int64_t ascale_row_stride,
                                          const uint8_t* wq4p,
                                          const uint16_t* wscale_h,
                                          const int16_t* wsum16, int mrows,
                                          int64_t k_blocks, int64_t block_size);
float GqaDotF16Avx512(const float* q, const uint16_t* k, int64_t hs);
void GqaAccumVF16Avx512(float* acc, float w, const uint16_t* v, int64_t hs);
