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
float DotFp32Avx512Dq(const float* a, const float* w, int64_t k);
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

void
DotInt8VnniTile16Int4CompactAvx512 (
    float* out16,
    const uint8_t* au8_row,
    const float* ascale,
    const uint8_t* wq4p,
    const uint16_t* wscale_h,
    const int16_t* wsum16,
    int64_t k_blocks,
    int64_t block_size);

void
DotInt8VnniTile16Int4CompactMBAvx512 (
    float* out,
    const uint8_t* au8_base,
    int64_t au8_row_stride,
    const float* ascale_base,
    int64_t ascale_row_stride,
    const uint8_t* wq4p,
    const uint16_t* wscale_h,
    const int16_t* wsum16,
    int64_t k_blocks,
    int64_t block_size);

void RepackWeightTile16Int4ZpAvx512(uint8_t* wq4p, float* wscale_packed,
                                    int32_t* qsum_packed, int32_t* zp_packed,
                                    const uint8_t* b_base, int64_t b_stride,
                                    const float* sc_base, int64_t sc_stride,
                                    const uint8_t* zp_base, int64_t zp_stride,
                                    int cnt, int64_t k_blocks, int64_t block_size,
                                    bool interleaved);
void DotInt8VnniTile16Int4CompactZpAvx512(float* out16, const uint8_t* au8_row,
                                          const float* ascale, const int32_t* asum_row,
                                          const uint8_t* wq4p, const uint16_t* wscale_h,
                                          const int16_t* qsum16, const int16_t* zp16,
                                          int64_t k_blocks, int64_t block_size);
void DotInt8VnniTile16Int4CompactZpItlAvx512(float* out16, const uint8_t* au8_row,
                                             const float* ascale, const int32_t* asum_row,
                                             const uint8_t* wq4p, const uint16_t* wscale_h,
                                             const int16_t* qsum16, const int16_t* zp16,
                                             int64_t k_blocks, int64_t block_size);
void DotInt8VnniTile16Int4CompactZpMBAvx512(float* out, const uint8_t* au8_base,
                                            int64_t au8_row_stride,
                                            const float* ascale_base,
                                            int64_t ascale_row_stride,
                                            const int32_t* asum_base,
                                            int64_t asum_row_stride,
                                            const uint8_t* wq4p,
                                            const uint16_t* wscale_h,
                                            const int16_t* qsum16,
                                            const int16_t* zp16, int mrows,
                                            int64_t k_blocks, int64_t block_size);
void DotInt8VnniTile16Int4CompactZpItlMBAvx512(float* out, const uint8_t* au8_base,
                                               int64_t au8_row_stride,
                                               const float* ascale_base,
                                               int64_t ascale_row_stride,
                                               const int32_t* asum_base,
                                               int64_t asum_row_stride,
                                               const uint8_t* wq4p,
                                               const uint16_t* wscale_h,
                                               const int16_t* qsum16,
                                               const int16_t* zp16, int mrows,
                                               int64_t k_blocks, int64_t block_size);
float GqaDotF16Avx512(const float* q, const uint16_t* k, int64_t hs);
float GqaDotF16Avx512Dq(const float* q, const uint16_t* k, int64_t hs);
void GqaAccumVF16Avx512(float* acc, float w, const uint16_t* v, int64_t hs);
// fp16 RMSNorm row (fp16 I/O, fp32 accumulation). gamma is fp32 (caller caches
// the widened constant). value = in (+ skip if non-NULL);
// out = value * rsqrt(mean(value^2)+eps) * gamma; optional sum = value (fp16).
void RmsNormRowF16Avx512(const uint16_t* in, const uint16_t* skip,
                         const float* gamma, uint16_t* out, uint16_t* sum,
                         int64_t width, float eps);
// fp16 SwiGLU (fp16 gate/up/out). Bit-exact to the scalar reference when
// fast_exp==0 (fp16 intermediate rounding preserved, expf scalar). When
// fast_exp!=0 the sigmoid uses a vectorized exp approximation (faster, not
// bit-exact). Precondition: HasAvx512f().
void FusionCpuSwiGluF16Avx512(const uint16_t* gate, const uint16_t* up,
                              uint16_t* out, int64_t count);
// Widen fp16 -> fp32 (exact; bit-identical to scalar). Precondition: HasAvx512f().
void FusionCpuWidenHalfToFloatAvx512(float* dst, const uint16_t* src, int64_t count);
// Narrow fp32 -> fp16 RNE (bit-identical to scalar). Precondition: HasAvx512f().
void FusionCpuNarrowFloatToHalfAvx512(uint16_t* dst, const float* src, int64_t count);
