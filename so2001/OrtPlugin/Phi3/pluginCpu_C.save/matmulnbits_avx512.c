// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#include "matmulnbits_avx512.h"
#include "fp16_utils.h"

#include <immintrin.h>
#include <string.h>

#if defined(_MSC_VER)
#define FUSIONCPU_NOINLINE __declspec(noinline)
#else
#define FUSIONCPU_NOINLINE __attribute__((noinline))
#endif

static int FusionCpuNibble(const uint8_t* packed, int64_t index) {
    const uint8_t byte = packed[index >> 1];
    return (index & 1) != 0 ? (byte >> 4) : (byte & 0x0F);
}

FUSIONCPU_NOINLINE void DequantColumnAvx512(float* out, const uint8_t* b_row,
                                            const float* scales_row, const uint8_t* zp_row,
                                            bool has_zp, int64_t k_blocks,
                                            int64_t block_size, int default_zp) {
    const int64_t blob = block_size / 2;
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    int64_t block;
    for (block = 0; block < k_blocks; ++block) {
        const int zero = has_zp ? ((block & 1) ? (zp_row[block >> 1] >> 4)
                                                    : (zp_row[block >> 1] & 0x0F))
                                : default_zp;
        const __m512 scale = _mm512_set1_ps(scales_row[block]);
        const __m512i zero_vector = _mm512_set1_epi32(zero);
        const uint8_t* source = b_row + block * blob;
        float* destination = out + block * block_size;
        int64_t offset;
        for (offset = 0; offset < block_size; offset += 16) {
            const __m128i packed = _mm_loadl_epi64((const __m128i*)(source + (offset >> 1)));
            const __m128i low = _mm_and_si128(packed, nibble_mask);
            const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            const __m128i expanded = _mm_unpacklo_epi8(low, high);
            const __m512i integers = _mm512_cvtepu8_epi32(expanded);
            const __m512 values = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(integers, zero_vector)), scale);
            _mm512_storeu_ps(destination + offset, values);
        }
    }
}

FUSIONCPU_NOINLINE float DotFp32Avx512(const float* a, const float* w, int64_t k) {
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    int64_t index = 0;
    _Alignas(64) float lanes[16];
    float result = 0.0f;
    int lane;
    for (; index + 64 <= k; index += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + index), _mm512_loadu_ps(w + index), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + index + 16), _mm512_loadu_ps(w + index + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + index + 32), _mm512_loadu_ps(w + index + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + index + 48), _mm512_loadu_ps(w + index + 48), a3);
    }
    for (; index + 16 <= k; index += 16) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + index), _mm512_loadu_ps(w + index), a0);
    }
    _mm512_store_ps(lanes, _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
    for (lane = 0; lane < 16; ++lane) result += lanes[lane];
    for (; index < k; ++index) result += a[index] * w[index];
    return result;
}

FUSIONCPU_NOINLINE float DotInt8VnniColumn(const uint8_t* au8, const float* ascale,
                                           const int8_t* wq8, const float* wscale,
                                           const int32_t* wsum, int64_t k_blocks,
                                           int64_t block_size) {
    float result = 0.0f;
    int64_t block;
    for (block = 0; block < k_blocks; ++block) {
        int64_t element;
        int64_t raw = 0;
        const int64_t base = block * block_size;
        for (element = 0; element < block_size; ++element) {
            raw += (int64_t)au8[base + element] * (int64_t)wq8[base + element];
        }
        result += ascale[block] * wscale[block] * (float)(raw - 128 * (int64_t)wsum[block]);
    }
    return result;
}

FUSIONCPU_NOINLINE void RepackWeightColumnInt8Avx512(
    int8_t* wq8_out, float* wscale_out, int32_t* wsum_out, const uint8_t* b_row,
    const float* scales_row, const uint8_t* zp_row, bool has_zp, int64_t k_blocks,
    int64_t block_size, int default_zp) {
    const int64_t blob = block_size / 2;
    int64_t block;
    for (block = 0; block < k_blocks; ++block) {
        const int zero = has_zp ? ((block & 1) ? (zp_row[block >> 1] >> 4)
                                                    : (zp_row[block >> 1] & 0x0F))
                                : default_zp;
        const uint8_t* source = b_row + block * blob;
        int64_t sum = 0;
        int64_t element;
        wscale_out[block] = scales_row[block];
        for (element = 0; element < block_size; ++element) {
            const int value = FusionCpuNibble(source, element) - zero;
            wq8_out[block * block_size + element] = (int8_t)value;
            sum += value;
        }
        wsum_out[block] = (int32_t)sum;
    }
}

FUSIONCPU_NOINLINE void DotInt8VnniTile16Avx512(
    float* out16, const uint8_t* au8_row, const float* ascale, const int8_t* wq8p,
    const float* wscale_packed, const int32_t* wsum_packed, int64_t k_blocks,
    int64_t block_size) {
    int column;
    for (column = 0; column < 16; ++column) {
        float result = 0.0f;
        int64_t block;
        for (block = 0; block < k_blocks; ++block) {
            int64_t element;
            int64_t raw = 0;
            for (element = 0; element < block_size; ++element) {
                const int64_t strip = element >> 2;
                const int64_t within = element & 3;
                const int8_t weight = wq8p[block * block_size * 16 + strip * 64 + column * 4 + within];
                raw += (int64_t)au8_row[block * block_size + element] * weight;
            }
            result += ascale[block] * wscale_packed[block * 16 + column] *
                      (float)(raw - 128 * (int64_t)wsum_packed[block * 16 + column]);
        }
        out16[column] = result;
    }
}

FUSIONCPU_NOINLINE void RepackWeightTile16Int8Avx512(
    int8_t* wq8p, float* wscale_packed, int32_t* wsum_packed, const uint8_t* b_base,
    int64_t b_stride, const float* sc_base, int64_t sc_stride, const uint8_t* zp_base,
    int64_t zp_stride, bool has_zp, int count, int64_t k_blocks, int64_t block_size,
    int default_zp) {
    int column;
    memset(wq8p, 0, (size_t)k_blocks * (size_t)block_size * 16);
    memset(wscale_packed, 0, (size_t)k_blocks * 16 * sizeof(*wscale_packed));
    memset(wsum_packed, 0, (size_t)k_blocks * 16 * sizeof(*wsum_packed));
    for (column = 0; column < count; ++column) {
        const uint8_t* b_row = b_base + (int64_t)column * b_stride;
        const float* scales = sc_base + (int64_t)column * sc_stride;
        const uint8_t* zp = has_zp ? zp_base + (int64_t)column * zp_stride : NULL;
        const int64_t blob = block_size / 2;
        int64_t block;
        for (block = 0; block < k_blocks; ++block) {
            int64_t sum = 0;
            int64_t element;
            const int zero = has_zp ? ((block & 1) ? (zp[block >> 1] >> 4) : (zp[block >> 1] & 0x0F))
                                    : default_zp;
            wscale_packed[block * 16 + column] = scales[block];
            for (element = 0; element < block_size; ++element) {
                const int value = FusionCpuNibble(b_row + block * blob, element) - zero;
                const int64_t strip = element >> 2;
                const int64_t within = element & 3;
                wq8p[block * block_size * 16 + strip * 64 + column * 4 + within] = (int8_t)value;
                sum += value;
            }
            wsum_packed[block * 16 + column] = (int32_t)sum;
        }
    }
}

FUSIONCPU_NOINLINE void RepackWeightTile16Int4Avx512(
    uint8_t* wq4p, float* wscale_packed, int32_t* wsum_packed, const uint8_t* b_base,
    int64_t b_stride, const float* sc_base, int64_t sc_stride, int count,
    int64_t k_blocks, int64_t block_size, int default_zp) {
    int column;
    memset(wq4p, 0, (size_t)k_blocks * (size_t)block_size * 8);
    memset(wscale_packed, 0, (size_t)k_blocks * 16 * sizeof(*wscale_packed));
    memset(wsum_packed, 0, (size_t)k_blocks * 16 * sizeof(*wsum_packed));
    for (column = 0; column < count; ++column) {
        const uint8_t* b_row = b_base + (int64_t)column * b_stride;
        const float* scales = sc_base + (int64_t)column * sc_stride;
        const int64_t blob = block_size / 2;
        int64_t block;
        for (block = 0; block < k_blocks; ++block) {
            int64_t sum = 0;
            int64_t element;
            wscale_packed[block * 16 + column] = scales[block];
            for (element = 0; element < block_size; ++element) {
                const int value = FusionCpuNibble(b_row + block * blob, element) - default_zp;
                const int64_t strip = element >> 2;
                const int64_t byte_index = element & 3;
                uint8_t* destination = wq4p + block * block_size * 8 + strip * 32 + column * 2;
                if (byte_index < 2)
                    destination[0] |= (uint8_t)((value & 0x0F) << (byte_index * 4));
                else
                    destination[1] |= (uint8_t)((value & 0x0F) << ((byte_index - 2) * 4));
                sum += value;
            }
            wsum_packed[block * 16 + column] = (int32_t)sum;
        }
    }
}

FUSIONCPU_NOINLINE void DotInt8VnniTile16Int4Avx512(
    float* out16, const uint8_t* au8_row, const float* ascale, const uint8_t* wq4p,
    const float* wscale_packed, const int32_t* wsum_packed, int64_t k_blocks,
    int64_t block_size) {
    int column;
    for (column = 0; column < 16; ++column) {
        float result = 0.0f;
        int64_t block;
        for (block = 0; block < k_blocks; ++block) {
            int64_t raw = 0;
            int64_t element;
            for (element = 0; element < block_size; ++element) {
                const int64_t strip = element >> 2;
                const int byte = (int)(element & 3) >> 1;
                const int shift = ((int)element & 1) * 4;
                uint8_t packed = wq4p[block * block_size * 8 + strip * 32 + column * 2 + byte];
                int weight = (packed >> shift) & 0x0F;
                if (weight >= 8) weight -= 16;
                raw += (int64_t)au8_row[block * block_size + element] * weight;
            }
            result += ascale[block] * wscale_packed[block * 16 + column] *
                      (float)(raw - 128 * (int64_t)wsum_packed[block * 16 + column]);
        }
        out16[column] = result;
    }
}

FUSIONCPU_NOINLINE void DotInt8VnniTile16Int4CompactAvx512(
    float* out16, const uint8_t* au8_row, const float* ascale, const uint8_t* wq4p,
    const uint16_t* wscale_h, const int16_t* wsum16, int64_t k_blocks,
    int64_t block_size) {
    int column;
    for (column = 0; column < 16; ++column) {
        float result = 0.0f;
        int64_t block;
        for (block = 0; block < k_blocks; ++block) {
            int64_t raw = 0;
            int64_t element;
            for (element = 0; element < block_size; ++element) {
                const int64_t strip = element >> 2;
                const int byte = (int)(element & 3) >> 1;
                const int shift = ((int)element & 1) * 4;
                uint8_t packed = wq4p[block * block_size * 8 + strip * 32 + column * 2 + byte];
                int weight = (packed >> shift) & 0x0F;
                if (weight >= 8) weight -= 16;
                raw += (int64_t)au8_row[block * block_size + element] * weight;
            }
            result += ascale[block] * FusionCpuHalfToFloat(wscale_h[block * 16 + column]) *
                      (float)(raw - 128 * (int64_t)wsum16[block * 16 + column]);
        }
        out16[column] = result;
    }
}

FUSIONCPU_NOINLINE void DotInt8VnniTile16Int4CompactMBAvx512(
    float* out, const uint8_t* au8_base, int64_t au8_row_stride, const float* ascale_base,
    int64_t ascale_row_stride, const uint8_t* wq4p, const uint16_t* wscale_h,
    const int16_t* wsum16, int mrows, int64_t k_blocks, int64_t block_size) {
    int row;
    for (row = 0; row < mrows; ++row) {
        DotInt8VnniTile16Int4CompactAvx512(out + row * 16,
            au8_base + (int64_t)row * au8_row_stride,
            ascale_base + (int64_t)row * ascale_row_stride,
            wq4p, wscale_h, wsum16, k_blocks, block_size);
    }
}

FUSIONCPU_NOINLINE float GqaDotF16Avx512(const float* q, const uint16_t* k, int64_t hs) {
    float result = 0.0f;
    int64_t index;
    for (index = 0; index < hs; ++index) result += q[index] * FusionCpuHalfToFloat(k[index]);
    return result;
}

FUSIONCPU_NOINLINE void GqaAccumVF16Avx512(float* acc, float weight, const uint16_t* value,
                                           int64_t hs) {
    int64_t index;
    for (index = 0; index < hs; ++index) acc[index] += weight * FusionCpuHalfToFloat(value[index]);
}
