/*
 * Optimized C11 MatMulNBits and GroupQueryAttention kernels.  This file is
 * included by fusion_cpu_ep.c after the ABI helpers and kernel state types.
 */

static inline bool FusionCpuSizeProduct(size_t a, size_t b, size_t* result) {
    uint64_t high;

    uint64_t low = UnsignedMultiply128(a, b, &high);
    if (high) {
        return false;

    } else {
        *result = low;
        return true;
    }

#if 0
    if (a != 0 && b > SIZE_MAX / a) return false;
    *result = a * b;
    return true;
#endif // #if 0
}

static void* FusionCpuCallocArray(size_t count, size_t element_size) {
    size_t bytes;
    if (!FusionCpuSizeProduct(count, element_size, &bytes)) return NULL;
    return calloc(bytes == 0 ? 1 : count, element_size);
}

static void* FusionCpuMallocArray(size_t count, size_t element_size) {
    size_t bytes;
    if (!FusionCpuSizeProduct(count, element_size, &bytes)) return NULL;
    return malloc(bytes == 0 ? 1 : bytes);
}

static inline int FusionCpuMinInt(int left, int right) {
    return left < right ? left : right;
}

static inline int64_t FusionCpuMinI64(int64_t left, int64_t right) {
    return left < right ? left : right;
}

static bool FusionCpuMatMulScalePointer(const ONNXTensorElementDataType type,
                                        const void* scale_data, int64_t k_blocks,
                                        int64_t n0, int64_t columns,
                                        float* scratch, const float** result) {
    size_t count;
    size_t offset;
    size_t i;
    if (!FusionCpuSizeProduct((size_t)columns, (size_t)k_blocks, &count) ||
        !FusionCpuSizeProduct((size_t)n0, (size_t)k_blocks, &offset)) {
        return false;
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        *result = (const float*)scale_data + offset;
        return true;
    }
    if (scratch == NULL) return false;
    for (i = 0; i < count; ++i) {
        scratch[i] = FusionCpuHalfToFloat(((const uint16_t*)scale_data)[offset + i]);
    }
    *result = scratch;
    return true;
}

static void FusionCpuMatMulReleasePrepack(FusionCpuMatMulNBitsState* state) {
    free(state->pp_wq8);
    free(state->pp_wq4);
    free(state->pp_wscale);
    free(state->pp_wsum);
    free(state->pp_wscale_h);
    free(state->pp_wsum16);
    free(state->pp_qsum16);
    free(state->pp_zp16);
    state->pp_wq8 = NULL;
    state->pp_wq4 = NULL;
    state->pp_wscale = NULL;
    state->pp_wsum = NULL;
    state->pp_wscale_h = NULL;
    state->pp_wsum16 = NULL;
    state->pp_qsum16 = NULL;
    state->pp_zp16 = NULL;
    state->prepack_ready = false;
    state->prepack_b = NULL;
    state->pp_n_tiles = 0;
    state->pp_k_blocks = 0;
    state->pp_block_size = 0;
}

typedef struct FusionCpuMatMulDispatch {
    FusionCpuMatMulNBitsState* state;
    ONNXTensorElementDataType a_type;
    const float* af;
    const uint8_t* au8;
    const float* ascale;
    const int32_t* asum;
    const uint8_t* b_data;
    const void* scale_data;
    const uint8_t* zp_data;
    void* y_data;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t k_blocks;
    int64_t blob;
    int64_t zp_cols;
    bool has_zp;
    bool io_fp16;
    bool use_avx512;
    bool use_vnni;
    bool use_prepack;
    bool want_int4;
    bool want_compact;
    bool want_compact_zp;
    int mb_step;
    int ways;
    float* scratch_wdq;
    int8_t* scratch_wq8;
    float* scratch_scale;
    int32_t* scratch_sum;
    volatile LONG failure;
} FusionCpuMatMulDispatch;

static inline void FusionCpuMatMulStore(const FusionCpuMatMulDispatch* dispatch,
                                 size_t offset, float value) {
    if (dispatch->io_fp16) {
        ((uint16_t*)dispatch->y_data)[offset] = FusionCpuFloatToHalf(value);
    } else {
        ((float*)dispatch->y_data)[offset] = value;
    }
}

typedef struct FusionCpuMatMulPrepackContext {
    FusionCpuMatMulDispatch* dispatch;
    volatile LONG failure;
} FusionCpuMatMulPrepackContext;

static inline void FusionCpuMatMulSetPrepackFailure(FusionCpuMatMulPrepackContext* context) {
    (void)InterlockedCompareExchange(&context->failure, 1, 0);
}

static void FusionCpuMatMulPrepackBody(void* opaque, int task, int nways) {
    FusionCpuMatMulPrepackContext* context = (FusionCpuMatMulPrepackContext*)opaque;
    FusionCpuMatMulDispatch* dispatch = context->dispatch;
    FusionCpuMatMulNBitsState* state = dispatch->state;
    const int64_t n_tiles = state->pp_n_tiles;
    const int64_t begin = (int64_t)task * n_tiles / nways;
    const int64_t end = (int64_t)(task + 1) * n_tiles / nways;
    size_t side_count;
    float* scale_scratch = NULL;
    float* wide_scale = NULL;
    int32_t* wide_sum = NULL;
    int32_t* wide_zp = NULL;
    int64_t tile;

    if (!FusionCpuSizeProduct((size_t)dispatch->k_blocks, 16, &side_count)) {
        FusionCpuMatMulSetPrepackFailure(context);
        return;
    }
    if (dispatch->a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        scale_scratch = (float*)FusionCpuMallocArray(side_count, sizeof(*scale_scratch));
    }
    if (state->pp_compact || state->pp_compact_zp) {
        wide_scale = (float*)FusionCpuMallocArray(side_count, sizeof(*wide_scale));
        wide_sum = (int32_t*)FusionCpuMallocArray(side_count, sizeof(*wide_sum));
        if (state->pp_compact_zp) {
            wide_zp = (int32_t*)FusionCpuMallocArray(side_count, sizeof(*wide_zp));
        }
    }
    if ((dispatch->a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && scale_scratch == NULL) ||
        ((state->pp_compact || state->pp_compact_zp) &&
         (wide_scale == NULL || wide_sum == NULL ||
          (state->pp_compact_zp && wide_zp == NULL)))) {
        FusionCpuMatMulSetPrepackFailure(context);
        free(scale_scratch);
        free(wide_scale);
        free(wide_sum);
        free(wide_zp);
        return;
    }

    for (tile = begin; tile < end; ++tile) {
        const int64_t n0 = tile * 16;
        const int count = (int)FusionCpuMinI64(16, dispatch->n - n0);
        const float* scales;
        size_t tile_side_offset;
        size_t tile_weight_offset;
        size_t i;
        if (InterlockedCompareExchange(&context->failure, 0, 0) != 0) break;
        if (!FusionCpuMatMulScalePointer(dispatch->a_type, dispatch->scale_data,
                                         dispatch->k_blocks, n0, count,
                                         scale_scratch, &scales) ||
            !FusionCpuSizeProduct((size_t)tile, side_count, &tile_side_offset) ||
            !FusionCpuSizeProduct((size_t)tile,
                                   (size_t)dispatch->k_blocks *
                                       (size_t)dispatch->state->block_size * 8,
                                   &tile_weight_offset)) {
            FusionCpuMatMulSetPrepackFailure(context);
            break;
        }
        if (state->pp_int4) {
            uint8_t* wq4 = state->pp_wq4 + tile_weight_offset;
            if (state->pp_compact) {
                RepackWeightTile16Int4Avx512(
                    wq4, wide_scale, wide_sum,
                    dispatch->b_data + (size_t)n0 * (size_t)dispatch->k_blocks *
                        (size_t)dispatch->blob,
                    dispatch->k_blocks * dispatch->blob, scales, dispatch->k_blocks, count,
                    dispatch->k_blocks, state->block_size, 8);
                for (i = 0; i < side_count; ++i) {
                    state->pp_wscale_h[tile_side_offset + i] = FusionCpuFloatToHalf(wide_scale[i]);
                    state->pp_wsum16[tile_side_offset + i] = (int16_t)wide_sum[i];
                }
            } else {
                RepackWeightTile16Int4Avx512(
                    wq4, state->pp_wscale + tile_side_offset, state->pp_wsum + tile_side_offset,
                    dispatch->b_data + (size_t)n0 * (size_t)dispatch->k_blocks *
                        (size_t)dispatch->blob,
                    dispatch->k_blocks * dispatch->blob, scales, dispatch->k_blocks, count,
                    dispatch->k_blocks, state->block_size, 8);
            }
        } else if (state->pp_compact_zp) {
            uint8_t* wq4 = state->pp_wq4 + tile_weight_offset;
            RepackWeightTile16Int4ZpAvx512(
                wq4, wide_scale, wide_sum, wide_zp,
                dispatch->b_data + (size_t)n0 * (size_t)dispatch->k_blocks *
                    (size_t)dispatch->blob,
                dispatch->k_blocks * dispatch->blob, scales, dispatch->k_blocks,
                dispatch->zp_data + (size_t)n0 * (size_t)dispatch->zp_cols,
                dispatch->zp_cols, count, dispatch->k_blocks, state->block_size,
                state->pp_zp_itl);
            for (i = 0; i < side_count; ++i) {
                state->pp_wscale_h[tile_side_offset + i] = FusionCpuFloatToHalf(wide_scale[i]);
                state->pp_qsum16[tile_side_offset + i] = (int16_t)wide_sum[i];
                state->pp_zp16[tile_side_offset + i] = (int16_t)wide_zp[i];
            }
        } else {
            size_t int8_weight_offset;
            if (!FusionCpuSizeProduct((size_t)tile,
                                      (size_t)dispatch->k_blocks *
                                          (size_t)state->block_size * 16,
                                      &int8_weight_offset)) {
                FusionCpuMatMulSetPrepackFailure(context);
                break;
            }
            RepackWeightTile16Int8Avx512(
                state->pp_wq8 + int8_weight_offset,
                state->pp_wscale + tile_side_offset, state->pp_wsum + tile_side_offset,
                dispatch->b_data + (size_t)n0 * (size_t)dispatch->k_blocks *
                    (size_t)dispatch->blob,
                dispatch->k_blocks * dispatch->blob, scales, dispatch->k_blocks,
                dispatch->has_zp ? dispatch->zp_data + (size_t)n0 * (size_t)dispatch->zp_cols : NULL,
                dispatch->zp_cols, dispatch->has_zp, count, dispatch->k_blocks,
                state->block_size, 8);
        }
    }
    free(scale_scratch);
    free(wide_scale);
    free(wide_sum);
    free(wide_zp);
}

static bool FusionCpuMatMulBuildPrepack(FusionCpuMatMulDispatch* dispatch) {
    FusionCpuMatMulNBitsState* state = dispatch->state;
    const int64_t n_tiles = (dispatch->n + 15) / 16;
    size_t side_count;
    size_t q4_count;
    size_t q8_count;
    FusionCpuMatMulPrepackContext context;
    int ways;

    if (state->prepack_ready) {
        const bool usable = state->prepack_b == dispatch->b_data &&
                            state->pp_n_tiles == n_tiles &&
                            state->pp_k_blocks == dispatch->k_blocks &&
                            state->pp_block_size == state->block_size;
        return usable;
    }
    if (!FusionCpuSizeProduct((size_t)n_tiles, (size_t)dispatch->k_blocks * 16, &side_count) ||
        !FusionCpuSizeProduct((size_t)n_tiles,
                               (size_t)dispatch->k_blocks * (size_t)state->block_size * 8,
                               &q4_count) ||
        !FusionCpuSizeProduct((size_t)n_tiles,
                               (size_t)dispatch->k_blocks * (size_t)state->block_size * 16,
                               &q8_count)) {
        return false;
    }
    state->pp_n_tiles = n_tiles;
    state->pp_k_blocks = dispatch->k_blocks;
    state->pp_block_size = state->block_size;
    state->pp_int4 = dispatch->want_int4;
    state->pp_compact = dispatch->want_compact;
    state->pp_compact_zp = dispatch->want_compact_zp;
    state->pp_zp_itl = dispatch->want_compact_zp &&
                       FusionCpuMatMulInterleaveEnabled() &&
                       (state->block_size % 8 == 0);
    if (state->pp_int4 || state->pp_compact_zp) {
        state->pp_wq4 = (uint8_t*)FusionCpuCallocArray(q4_count, sizeof(*state->pp_wq4));
    } else {
        state->pp_wq8 = (int8_t*)FusionCpuCallocArray(q8_count, sizeof(*state->pp_wq8));
    }
    if (state->pp_compact || state->pp_compact_zp) {
        state->pp_wscale_h =
            (uint16_t*)FusionCpuCallocArray(side_count, sizeof(*state->pp_wscale_h));
    } else {
        state->pp_wscale = (float*)FusionCpuCallocArray(side_count, sizeof(*state->pp_wscale));
        state->pp_wsum = (int32_t*)FusionCpuCallocArray(side_count, sizeof(*state->pp_wsum));
    }
    if (state->pp_compact) {
        state->pp_wsum16 =
            (int16_t*)FusionCpuCallocArray(side_count, sizeof(*state->pp_wsum16));
    }
    if (state->pp_compact_zp) {
        state->pp_qsum16 =
            (int16_t*)FusionCpuCallocArray(side_count, sizeof(*state->pp_qsum16));
        state->pp_zp16 =
            (int16_t*)FusionCpuCallocArray(side_count, sizeof(*state->pp_zp16));
    }
    if ((state->pp_int4 || state->pp_compact_zp ? state->pp_wq4 != NULL : state->pp_wq8 != NULL) &&
        (state->pp_compact || state->pp_compact_zp ? state->pp_wscale_h != NULL
                                                    : (state->pp_wscale != NULL && state->pp_wsum != NULL)) &&
        (!state->pp_compact || state->pp_wsum16 != NULL) &&
        (!state->pp_compact_zp || (state->pp_qsum16 != NULL && state->pp_zp16 != NULL))) {
        memset(&context, 0, sizeof(context));
        context.dispatch = dispatch;
        ways = FusionCpuMinInt(fusion_pool_size(),
                               n_tiles > INT_MAX ? INT_MAX : (int)n_tiles);

        if (ways < 1) {
            ways = 1;
        }

        fusion_pool_parallel_for(ways, FusionCpuMatMulPrepackBody, &context);
        if (InterlockedCompareExchange(&context.failure, 0, 0) == 0) {
            state->prepack_b = dispatch->b_data;
            state->prepack_ready = true;
            return true;
        }
    }
    FusionCpuMatMulReleasePrepack(state);
    return false;
}

static float FusionCpuMatMulFp32Dot(const FusionCpuMatMulDispatch* dispatch,
                                    const float* activation, const float* weight) {
    if (dispatch->use_avx512) {
        return FusionCpuHasAvx512Dq()
            ? DotFp32Avx512Dq(activation, weight, dispatch->k)
            : DotFp32Avx512(activation, weight, dispatch->k);

    } else {

        float value = 0.0f;
        int64_t k;
        for (k = 0; k < dispatch->k; ++k) value += activation[k] * weight[k];
        return value;
    }
}

static void FusionCpuMatMulComputeCachedRange(FusionCpuMatMulDispatch* dispatch,
                                              int64_t n_begin, int64_t n_end) {
    FusionCpuMatMulNBitsState* state = dispatch->state;
    const int64_t first_tile = n_begin / 16;
    const int64_t last_tile = (n_end + 15) / 16;
    int64_t tile;
    for (tile = first_tile; tile < last_tile; ++tile) {
        const int64_t n0 = tile * 16;
        const int c0 = (int)(n_begin > n0 ? n_begin - n0 : 0);
        const int c1 = (int)FusionCpuMinI64(16, n_end - n0);
        size_t side_offset = (size_t)tile * (size_t)dispatch->k_blocks * 16;
        size_t q4_offset = (size_t)tile * (size_t)dispatch->k_blocks *
                           (size_t)state->block_size * 8;
        size_t q8_offset = (size_t)tile * (size_t)dispatch->k_blocks *
                           (size_t)state->block_size * 16;
        float out_mb[FUSIONCPU_COMPACT_MB_MAX * 16];
        float out16[16];
        int64_t m0;
        if (c0 >= c1) continue;
        if (state->pp_compact_zp) {
            for (m0 = 0; m0 < dispatch->m; m0 += dispatch->mb_step) {
                const int rows = (int)FusionCpuMinI64(dispatch->mb_step, dispatch->m - m0);
                int row;
                if (state->pp_zp_itl) {
                    DotInt8VnniTile16Int4CompactZpItlMBAvx512(
                        out_mb, dispatch->au8 + (size_t)m0 * (size_t)dispatch->k, dispatch->k,
                        dispatch->ascale + (size_t)m0 * (size_t)dispatch->k_blocks,
                        dispatch->k_blocks,
                        dispatch->asum + (size_t)m0 * (size_t)dispatch->k_blocks,
                        dispatch->k_blocks, state->pp_wq4 + q4_offset,
                        state->pp_wscale_h + side_offset, state->pp_qsum16 + side_offset,
                        state->pp_zp16 + side_offset, rows, dispatch->k_blocks,
                        state->block_size);
                } else {
                    DotInt8VnniTile16Int4CompactZpMBAvx512(
                        out_mb, dispatch->au8 + (size_t)m0 * (size_t)dispatch->k, dispatch->k,
                        dispatch->ascale + (size_t)m0 * (size_t)dispatch->k_blocks,
                        dispatch->k_blocks,
                        dispatch->asum + (size_t)m0 * (size_t)dispatch->k_blocks,
                        dispatch->k_blocks, state->pp_wq4 + q4_offset,
                        state->pp_wscale_h + side_offset, state->pp_qsum16 + side_offset,
                        state->pp_zp16 + side_offset, rows, dispatch->k_blocks,
                        state->block_size);
                }
                for (row = 0; row < rows; ++row) {
                    int col;
                    for (col = c0; col < c1; ++col) {
                        FusionCpuMatMulStore(dispatch,
                            (size_t)(m0 + row) * (size_t)dispatch->n + (size_t)n0 + (size_t)col,
                            out_mb[row * 16 + col]);
                    }
                }
            }
        } else if (state->pp_compact) {
            for (m0 = 0; m0 < dispatch->m; m0 += dispatch->mb_step) {
                const int rows = (int)FusionCpuMinI64(dispatch->mb_step, dispatch->m - m0);

                //
                // Check if full 4-row blocks for the register-blocked kernel.
                //

                int64_t k_blocks = dispatch->k_blocks;
                const uint8_t* au8_base = dispatch->au8 + m0 * dispatch->k;
                int64_t au8_row_stride = dispatch->k;
                const float* ascale_base = dispatch->ascale + m0 * k_blocks;
                int64_t ascale_row_stride = dispatch->k_blocks;
                const uint8_t* wq4p = state->pp_wq4 + q4_offset;
                const uint16_t* wscale_h = state->pp_wscale_h + side_offset;
                const int16_t* wsum16 = state->pp_wsum16 + side_offset;
                int64_t block_size = state->block_size;

                if (rows == FUSIONCPU_COMPACT_MB_MAX) {
                    DotInt8VnniTile16Int4CompactMBAvx512(out_mb,
                                                         au8_base,
                                                         au8_row_stride,
                                                         ascale_base,
                                                         ascale_row_stride,
                                                         wq4p,
                                                         wscale_h,
                                                         wsum16,
                                                         k_blocks,
                                                         block_size);

                } else {
                    float* out = out_mb;

                    for (int64_t r = 0; r < rows; r += 1) {
                        DotInt8VnniTile16Int4CompactAvx512(out,
                                                           au8_base,
                                                           ascale_base,
                                                           wq4p,
                                                           wscale_h,
                                                           wsum16,
                                                           k_blocks,
                                                           block_size);

                        out += 16;
                        au8_base += au8_row_stride;
                        ascale_base += ascale_row_stride;
                    }
                }

                int64_t row;
                for (row = 0; row < rows; ++row) {
                    int64_t col;
                    for (col = c0; col < c1; ++col) {
                        FusionCpuMatMulStore(dispatch,
                            (size_t)(m0 + row) * (size_t)dispatch->n + (size_t)n0 + (size_t)col,
                            out_mb[row * 16 + col]);
                    }
                }
            }

        } else {
            int64_t row;
            for (row = 0; row < dispatch->m; ++row) {
                int col;
                const uint8_t* activation =
                    dispatch->au8 + (size_t)row * (size_t)dispatch->k;
                const float* scale =
                    dispatch->ascale + (size_t)row * (size_t)dispatch->k_blocks;
                if (state->pp_int4) {
                    DotInt8VnniTile16Int4Avx512(out16, activation, scale,
                        state->pp_wq4 + q4_offset, state->pp_wscale + side_offset,
                        state->pp_wsum + side_offset, dispatch->k_blocks, state->block_size);
                } else {
                    DotInt8VnniTile16Avx512(out16, activation, scale,
                        state->pp_wq8 + q8_offset, state->pp_wscale + side_offset,
                        state->pp_wsum + side_offset, dispatch->k_blocks, state->block_size);
                }
                for (col = c0; col < c1; ++col) {
                    FusionCpuMatMulStore(dispatch,
                        (size_t)row * (size_t)dispatch->n + (size_t)n0 + (size_t)col,
                        out16[col]);
                }
            }
        }
    }
}

static void FusionCpuMatMulComputeUncachedRange(FusionCpuMatMulDispatch* dispatch,
                                                int task, int64_t n_begin, int64_t n_end) {
    FusionCpuMatMulNBitsState* state = dispatch->state;
    float* wdq = dispatch->scratch_wdq + (size_t)task * (size_t)dispatch->k;
    int8_t* wq8 = dispatch->scratch_wq8 + (size_t)task * (size_t)dispatch->k * 16;
    float* scales = dispatch->scratch_scale +
        (size_t)task * (size_t)dispatch->k_blocks * 16;
    int32_t* sums = dispatch->scratch_sum +
        (size_t)task * (size_t)dispatch->k_blocks * 16;
    if (dispatch->use_vnni) {
        const bool per_column = dispatch->m < 16 && state->block_size % 64 == 0;
        if (per_column) {
            int64_t n;
            for (n = n_begin; n < n_end; ++n) {
                const float* scale;
                int64_t row;
                if (!FusionCpuMatMulScalePointer(dispatch->a_type, dispatch->scale_data,
                                                 dispatch->k_blocks, n, 1, scales, &scale)) {
                    (void)InterlockedCompareExchange(&dispatch->failure, 1, 0);
                    return;
                }
                RepackWeightColumnInt8Avx512(wq8, scales + dispatch->k_blocks,
                    sums, dispatch->b_data + (size_t)n * (size_t)dispatch->k_blocks *
                        (size_t)dispatch->blob, scale,
                    dispatch->has_zp ? dispatch->zp_data + (size_t)n * (size_t)dispatch->zp_cols : NULL,
                    dispatch->has_zp, dispatch->k_blocks, state->block_size, 8);
                for (row = 0; row < dispatch->m; ++row) {
                    const float value = DotInt8VnniColumn(
                        dispatch->au8 + (size_t)row * (size_t)dispatch->k,
                        dispatch->ascale + (size_t)row * (size_t)dispatch->k_blocks,
                        wq8, scales + dispatch->k_blocks, sums,
                        dispatch->k_blocks, state->block_size);
                    FusionCpuMatMulStore(dispatch,
                        (size_t)row * (size_t)dispatch->n + (size_t)n, value);
                }
            }
            return;
        }
        {
            int64_t n0;
            float out16[16];
            for (n0 = n_begin; n0 < n_end; n0 += 16) {
                const int count = (int)FusionCpuMinI64(16, n_end - n0);
                const float* scale;
                int64_t row;
                if (!FusionCpuMatMulScalePointer(dispatch->a_type, dispatch->scale_data,
                                                 dispatch->k_blocks, n0, count, scales, &scale)) {
                    (void)InterlockedCompareExchange(&dispatch->failure, 1, 0);
                    return;
                }
                RepackWeightTile16Int8Avx512(
                    wq8, scales + (size_t)dispatch->k_blocks * 16, sums,
                    dispatch->b_data + (size_t)n0 * (size_t)dispatch->k_blocks *
                        (size_t)dispatch->blob,
                    dispatch->k_blocks * dispatch->blob, scale, dispatch->k_blocks,
                    dispatch->has_zp ? dispatch->zp_data + (size_t)n0 * (size_t)dispatch->zp_cols : NULL,
                    dispatch->zp_cols, dispatch->has_zp, count, dispatch->k_blocks,
                    state->block_size, 8);
                for (row = 0; row < dispatch->m; ++row) {
                    int col;
                    DotInt8VnniTile16Avx512(out16,
                        dispatch->au8 + (size_t)row * (size_t)dispatch->k,
                        dispatch->ascale + (size_t)row * (size_t)dispatch->k_blocks,
                        wq8, scales + (size_t)dispatch->k_blocks * 16, sums,
                        dispatch->k_blocks, state->block_size);
                    for (col = 0; col < count; ++col) {
                        FusionCpuMatMulStore(dispatch,
                            (size_t)row * (size_t)dispatch->n + (size_t)n0 + (size_t)col,
                            out16[col]);
                    }
                }
            }
            return;
        }
    }
    {
        int64_t n;
        for (n = n_begin; n < n_end; ++n) {
            const float* scale;
            int64_t block;
            int64_t row;
            if (!FusionCpuMatMulScalePointer(dispatch->a_type, dispatch->scale_data,
                                             dispatch->k_blocks, n, 1, scales, &scale)) {
                (void)InterlockedCompareExchange(&dispatch->failure, 1, 0);
                return;
            }
            if (dispatch->use_avx512) {
                DequantColumnAvx512(wdq,
                    dispatch->b_data + (size_t)n * (size_t)dispatch->k_blocks *
                        (size_t)dispatch->blob,
                    scale,
                    dispatch->has_zp ? dispatch->zp_data + (size_t)n * (size_t)dispatch->zp_cols : NULL,
                    dispatch->has_zp, dispatch->k_blocks, state->block_size, 8);
            } else {
                const uint8_t* b_row = dispatch->b_data +
                    (size_t)n * (size_t)dispatch->k_blocks * (size_t)dispatch->blob;
                const uint8_t* zp = dispatch->has_zp
                    ? dispatch->zp_data + (size_t)n * (size_t)dispatch->zp_cols : NULL;
                for (block = 0; block < dispatch->k_blocks; ++block) {
                    const int zero = dispatch->has_zp
                        ? ((block & 1) ? (zp[block >> 1] >> 4) : (zp[block >> 1] & 15)) : 8;
                    int64_t element;
                    for (element = 0; element < state->block_size; ++element) {
                        const uint8_t byte = b_row[(size_t)block * (size_t)dispatch->blob +
                                                   (size_t)(element >> 1)];
                        const int quant = (element & 1) ? (byte >> 4) : (byte & 15);
                        wdq[block * state->block_size + element] =
                            ((float)quant - (float)zero) * scale[block];
                    }
                }
            }
            for (row = 0; row < dispatch->m; ++row) {
                FusionCpuMatMulStore(dispatch, (size_t)row * (size_t)dispatch->n + (size_t)n,
                    FusionCpuMatMulFp32Dot(dispatch,
                        dispatch->af + (size_t)row * (size_t)dispatch->k, wdq));
            }
        }
    }
}

static void FusionCpuMatMulWorker(void* opaque, int task, int nways) {
    FusionCpuMatMulDispatch* dispatch = (FusionCpuMatMulDispatch*)opaque;
    const int64_t begin = (int64_t)task * dispatch->n / nways;
    const int64_t end = (int64_t)(task + 1) * dispatch->n / nways;
    if (InterlockedCompareExchange(&dispatch->failure, 0, 0) != 0) return;
    if (dispatch->use_prepack) {
        FusionCpuMatMulComputeCachedRange(dispatch, begin, end);

    } else {
        FusionCpuMatMulComputeUncachedRange(dispatch, task, begin, end);
    }
}

static OrtStatus* ORT_API_CALL FusionCpuMatMulNBitsCompute(void* opaque,
                                                            OrtKernelContext* context) {
    FusionCpuMatMulNBitsState* state = (FusionCpuMatMulNBitsState*)opaque;
    const OrtValue* a_value;
    const OrtValue* b_value;
    const OrtValue* scale_value;
    const OrtValue* aux_value;
    ONNXTensorElementDataType a_type;
    ONNXTensorElementDataType b_type;
    ONNXTensorElementDataType scale_type;
    ONNXTensorElementDataType aux_type;
    const int64_t* a_shape;
    const int64_t* b_shape;
    const int64_t* scale_shape;
    const int64_t* aux_shape = NULL;
    size_t a_rank;
    size_t b_rank;
    size_t scale_rank;
    size_t aux_rank = 0;
    const void* a_data;
    const void* b_void;
    const void* scale_void;
    const void* aux_void = NULL;
    const uint16_t* awq = NULL;
    const uint8_t* zp_data = NULL;
    int64_t m = 1;
    int64_t k;
    int64_t n;
    int64_t k_blocks;
    int64_t blob;
    int64_t zp_cols;
    size_t count;
    size_t awq_count = 0;
    size_t af_count;
    int64_t dim;
    int64_t* y_shape = NULL;
    OrtValue* y_value;
    void* y_data;
    float* af = NULL;
    uint8_t* au8 = NULL;
    float* ascale = NULL;
    int32_t* asum = NULL;
    FusionCpuMatMulDispatch dispatch;
    bool has_awq;
    bool has_zp;
    bool use_avx512;
    bool use_vnni;
    bool want_int4;
    bool want_compact;
    bool want_compact_zp;
    int ways;
    int64_t total_macs;
    OrtStatus* status;

    {
        size_t input_count;
        size_t output_count;
        status = state->api->KernelContext_GetInputCount(context, &input_count);
        if (status != NULL) return status;
        status = state->api->KernelContext_GetOutputCount(context, &output_count);
        if (status != NULL) return status;
        if (output_count != 1 || state->idx_a < 0 || state->idx_b < 0 ||
            state->idx_scales < 0 || (size_t)state->idx_a >= input_count ||
            (size_t)state->idx_b >= input_count || (size_t)state->idx_scales >= input_count ||
            (state->idx_scale >= 0 && (size_t)state->idx_scale >= input_count) ||
            (state->idx_zp >= 0 && (size_t)state->idx_zp >= input_count)) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
                "MatMulNBits: fused-node input or output index is invalid.");
        }
    }
    status = FusionCpuGetInput(state->api, context, (size_t)state->idx_a, &a_value,
                               &a_type, &a_shape, &a_rank, &a_data);
    if (status != NULL) return status;
    status = FusionCpuGetInput(state->api, context, (size_t)state->idx_b, &b_value,
                               &b_type, &b_shape, &b_rank, &b_void);
    if (status != NULL) return status;
    status = FusionCpuGetInput(state->api, context, (size_t)state->idx_scales, &scale_value,
                               &scale_type, &scale_shape, &scale_rank, &scale_void);
    if (status != NULL) return status;
    has_awq = state->idx_scale >= 0;
    has_zp = state->idx_zp >= 0;
    if (has_awq) {
        status = FusionCpuGetInput(state->api, context, (size_t)state->idx_scale, &aux_value,
                                   &aux_type, &aux_shape, &aux_rank, &aux_void);
        if (status != NULL) return status;
        if (aux_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
            !FusionCpuElementCount(aux_shape, aux_rank, &awq_count)) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
                "MatMulNBits: AWQ scale must be a float16 tensor.");
        }
        awq = (const uint16_t*)aux_void;
    }
    if (has_zp) {
        status = FusionCpuGetInput(state->api, context, (size_t)state->idx_zp, &aux_value,
                                   &aux_type, &aux_shape, &aux_rank, &aux_void);
        if (status != NULL) return status;
        if (aux_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
                "MatMulNBits: zero_points must be uint8.");
        }
        zp_data = (const uint8_t*)aux_void;
    }
    if ((a_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
         a_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
        b_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 || scale_type != a_type ||
        state->bits != 4 || a_rank == 0 || b_rank != 3) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
            "MatMulNBits: unsupported tensor dtype, rank, or bit width.");
    }
    k = a_shape[a_rank - 1];
    n = b_shape[0];
    k_blocks = b_shape[1];
    blob = b_shape[2];
    if (k <= 0 || n <= 0 || k_blocks <= 0 || state->block_size <= 0 ||
        blob != (int64_t)state->block_size * state->bits / 8 ||
        k_blocks * state->block_size != k) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
            "MatMulNBits: B shape, K, or block_size is inconsistent.");
    }
    for (dim = 0; dim + 1 < (int64_t)a_rank; ++dim) {
        if (a_shape[dim] < 0 || (a_shape[dim] != 0 && m > INT64_MAX / a_shape[dim])) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
                "MatMulNBits: invalid A shape.");
        }
        m *= a_shape[dim];
    }
    if (!FusionCpuElementCount(scale_shape, scale_rank, &count) ||
        count != (size_t)n * (size_t)k_blocks ||
        (has_awq && count == 0)) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
            "MatMulNBits: scale tensor shape is inconsistent.");
    }
    if (has_awq && awq_count != (size_t)k) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
            "MatMulNBits: AWQ scale must have K elements.");
    }
    zp_cols = (k_blocks * state->bits + 7) / 8;
    if (has_zp && (!FusionCpuElementCount(aux_shape, aux_rank, &count) ||
                   count != (size_t)n * (size_t)zp_cols)) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
            "MatMulNBits: zero_points tensor shape is inconsistent.");
    }
    if (!FusionCpuSizeProduct((size_t)m, (size_t)k, &af_count)) {
        return state->api->CreateStatus(ORT_EP_FAIL, "MatMulNBits: activation size overflows.");
    }
    af = (float*)FusionCpuMallocArray(af_count, sizeof(*af));
    y_shape = (int64_t*)FusionCpuMallocArray(a_rank, sizeof(*y_shape));
    if (af == NULL || y_shape == NULL) {
        free(af);
        free(y_shape);
        return state->api->CreateStatus(ORT_EP_FAIL, "MatMulNBits: allocation failed.");
    }
    for (dim = 0; dim < m; ++dim) {
        const uint16_t* a16 = a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            ? (const uint16_t*)a_data + (size_t)dim * (size_t)k : NULL;
        const float* a32 = a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            ? (const float*)a_data + (size_t)dim * (size_t)k : NULL;
        int64_t element;
        for (element = 0; element < k; ++element) {
            float value = a16 != NULL ? FusionCpuHalfToFloat(a16[element]) : a32[element];
            if (has_awq) {
                value = FusionCpuHalfToFloat(
                    FusionCpuFloatToHalf(value * FusionCpuHalfToFloat(awq[element])));
            }
            af[(size_t)dim * (size_t)k + (size_t)element] = value;
        }
    }
    memcpy(y_shape, a_shape, a_rank * sizeof(*y_shape));
    y_shape[a_rank - 1] = n;
    status = FusionCpuGetOutput(state->api, context, 0, y_shape, a_rank, &y_value, &y_data);
    free(y_shape);
    if (status != NULL) {
        free(af);
        return status;
    }

    use_avx512 = FusionCpuHasAvx512f() &&
                 (state->block_size % 16 == 0) &&
                 !FusionCpuForceScalarMatMul();

    use_vnni = use_avx512 &&
               FusionCpuHasAvx512Vnni() &&
               FusionCpuMatMulVnniEnabled();

    want_int4 = !has_zp && FusionCpuMatMulInt4TileEnabled();

    want_compact = want_int4 &&
                   FusionCpuMatMulCompactEnabled() &&
                   (state->block_size <= 4096);

    want_compact_zp = has_zp &&
                      FusionCpuMatMulInt4TileEnabled() &&
                      FusionCpuMatMulCompactEnabled() &&
                      FusionCpuMatMulCompactZpEnabled() &&
                      ((int64_t)state->block_size * 15 <= INT16_MAX);

    if (FusionCpuMatMulVerbose()) {
        static volatile LONG logged = 0;
        if (InterlockedCompareExchange(&logged, 1, 0) == 0) {
            fprintf(stderr,
                "[FusionCpuEP] MatMulNBits dispatch: avx512=%d vnni=%d prepack=%d "
                "int4=%d compact=%d compact_zp=%d block=%d\n",
                use_avx512, use_vnni, use_vnni && FusionCpuMatMulPrepackEnabled(),
                want_int4, want_compact, want_compact_zp, state->block_size);
            fflush(stderr);
        }
    }

    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.state = state;
    dispatch.a_type = a_type;
    dispatch.af = af;
    dispatch.b_data = (const uint8_t*)b_void;
    dispatch.scale_data = scale_void;
    dispatch.zp_data = zp_data;
    dispatch.y_data = y_data;
    dispatch.m = m;
    dispatch.n = n;
    dispatch.k = k;
    dispatch.k_blocks = k_blocks;
    dispatch.blob = blob;
    dispatch.zp_cols = zp_cols;
    dispatch.has_zp = has_zp;
    dispatch.io_fp16 = a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    dispatch.use_avx512 = use_avx512;
    dispatch.use_vnni = use_vnni;
    dispatch.want_int4 = want_int4;
    dispatch.want_compact = want_compact;
    dispatch.want_compact_zp = want_compact_zp;
    dispatch.mb_step = FusionCpuMatMulMBlockEnabled() ? FUSIONCPU_COMPACT_MB_MAX : 1;
    if (use_vnni) {
        size_t au8_count;
        size_t block_count;
        if (!FusionCpuSizeProduct((size_t)m, (size_t)k, &au8_count) ||
            !FusionCpuSizeProduct((size_t)m, (size_t)k_blocks, &block_count)) {
            free(af);
            return state->api->CreateStatus(ORT_EP_FAIL, "MatMulNBits: quantization size overflows.");
        }
        au8 = (uint8_t*)FusionCpuMallocArray(au8_count, sizeof(*au8));
        ascale = (float*)FusionCpuMallocArray(block_count, sizeof(*ascale));
        if (want_compact_zp) asum = (int32_t*)FusionCpuMallocArray(block_count, sizeof(*asum));
        if (au8 == NULL || ascale == NULL || (want_compact_zp && asum == NULL)) {
            free(af); free(au8); free(ascale); free(asum);
            return state->api->CreateStatus(ORT_EP_FAIL, "MatMulNBits: activation quantization allocation failed.");
        }
        for (dim = 0; dim < m; ++dim) {
            int64_t block;
            for (block = 0; block < k_blocks; ++block) {
                const int64_t base = block * state->block_size;
                float maxabs = 0.0f;
                float scale;
                float inverse;
                int32_t sum = 0;
                int64_t element;
                for (element = 0; element < state->block_size; ++element) {
                    const float value = fabsf(af[(size_t)dim * (size_t)k + (size_t)(base + element)]);
                    if (value > maxabs) maxabs = value;
                }
                scale = maxabs / 127.0f;
                inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
                ascale[(size_t)dim * (size_t)k_blocks + (size_t)block] = scale;
                for (element = 0; element < state->block_size; ++element) {
                    int quant = (int)lrintf(
                        af[(size_t)dim * (size_t)k + (size_t)(base + element)] * inverse);
                    if (quant < -127) quant = -127;
                    else if (quant > 127) quant = 127;
                    au8[(size_t)dim * (size_t)k + (size_t)(base + element)] =
                        (uint8_t)(quant + 128);
                    sum += quant;
                }
                if (asum != NULL) asum[(size_t)dim * (size_t)k_blocks + (size_t)block] = sum;
            }
        }
        dispatch.au8 = au8;
        dispatch.ascale = ascale;
        dispatch.asum = asum;
    }
    dispatch.use_prepack = use_vnni && FusionCpuMatMulPrepackEnabled();
    if (dispatch.use_prepack) {
        if (!FusionCpuMatMulBuildPrepack(&dispatch)) {
            dispatch.use_prepack = false;
            if (state->prepack_ready == false && state->prepack_b == NULL) {
                free(af); free(au8); free(ascale); free(asum);
                return state->api->CreateStatus(ORT_EP_FAIL, "MatMulNBits: prepack allocation failed.");
            }
        }
    }
    total_macs = m > 0 && n > 0 && k > 0 && m <= INT64_MAX / n &&
                 m * n <= INT64_MAX / k ? m * n * k : INT64_MAX;
    ways = (int)FusionCpuMinI64(total_macs / (1 << 20), n);
    if (ways < 1) {
        ways = 1;
    }

    ways = FusionCpuMinInt(ways, fusion_pool_size());

    dispatch.ways = ways;
    if (!dispatch.use_prepack) {
        size_t wdq_count;
        size_t wq8_count;
        size_t side_count;
        if (!FusionCpuSizeProduct((size_t)ways, (size_t)k, &wdq_count) ||
            !FusionCpuSizeProduct(wdq_count, 16, &wq8_count) ||
            !FusionCpuSizeProduct((size_t)ways, (size_t)k_blocks * 16, &side_count)) {
            free(af); free(au8); free(ascale); free(asum);
            return state->api->CreateStatus(ORT_EP_FAIL, "MatMulNBits: worker scratch size overflows.");
        }
        dispatch.scratch_wdq = (float*)FusionCpuMallocArray(wdq_count, sizeof(*dispatch.scratch_wdq));
        dispatch.scratch_wq8 = (int8_t*)FusionCpuMallocArray(wq8_count, sizeof(*dispatch.scratch_wq8));
        dispatch.scratch_scale = (float*)FusionCpuMallocArray(side_count, sizeof(*dispatch.scratch_scale));
        dispatch.scratch_sum = (int32_t*)FusionCpuMallocArray(side_count, sizeof(*dispatch.scratch_sum));
        if (dispatch.scratch_wdq == NULL || dispatch.scratch_wq8 == NULL ||
            dispatch.scratch_scale == NULL || dispatch.scratch_sum == NULL) {
            free(af); free(au8); free(ascale); free(asum);
            free(dispatch.scratch_wdq); free(dispatch.scratch_wq8);
            free(dispatch.scratch_scale); free(dispatch.scratch_sum);
            return state->api->CreateStatus(ORT_EP_FAIL, "MatMulNBits: worker scratch allocation failed.");
        }
    }
    if (FusionCpuTraceEnabled()) {
        fprintf(stderr,
            "[FusionCpuEP][trace] MatMulNBits m=%lld n=%lld k=%lld ways=%d "
            "avx512=%d vnni=%d prepack=%d zp=%d\n",
            (long long)m, (long long)n, (long long)k, ways, use_avx512, use_vnni,
            dispatch.use_prepack, has_zp);
        fflush(stderr);
    }
    fusion_pool_parallel_for(ways, FusionCpuMatMulWorker, &dispatch);
    free(dispatch.scratch_wdq);
    free(dispatch.scratch_wq8);
    free(dispatch.scratch_scale);
    free(dispatch.scratch_sum);
    free(af);
    free(au8);
    free(ascale);
    free(asum);
    if (InterlockedCompareExchange(&dispatch.failure, 0, 0) != 0) {
        return state->api->CreateStatus(ORT_EP_FAIL, "MatMulNBits: worker scratch conversion failed.");
    }
    return NULL;
}

static OrtStatus* ORT_API_CALL FusionCpuSwiGluCompute(void* opaque,
                                                       OrtKernelContext* context) {
    FusionCpuSwiGluState* state = (FusionCpuSwiGluState*)opaque;
    const OrtValue* gate_value;
    const OrtValue* up_value;
    ONNXTensorElementDataType gate_type;
    ONNXTensorElementDataType up_type;
    const int64_t* gate_shape;
    const int64_t* up_shape;
    size_t gate_rank;
    size_t up_rank;
    const void* gate_data;
    const void* up_data;
    OrtValue* output_value;
    void* output_data;
    size_t count;
    size_t i;
    OrtStatus* status;
    status = FusionCpuGetInput(state->api, context, (size_t)state->idx_g, &gate_value,
                               &gate_type, &gate_shape, &gate_rank, &gate_data);
    if (status != NULL) return status;
    status = FusionCpuGetInput(state->api, context, (size_t)state->idx_u, &up_value,
                               &up_type, &up_shape, &up_rank, &up_data);
    if (status != NULL) return status;
    if (gate_type != up_type ||
        (gate_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
         gate_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
        !FusionCpuShapeEqual(gate_shape, gate_rank, up_shape, up_rank) ||
        !FusionCpuElementCount(gate_shape, gate_rank, &count)) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
            "FusionCPU SwiGLU requires matching float16 or float32 gate/up tensors.");
    }
    status = FusionCpuGetOutput(state->api, context, 0, gate_shape, gate_rank,
                                &output_value, &output_data);
    if (status != NULL) return status;
    if (gate_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const uint16_t* gate = (const uint16_t*)gate_data;
        const uint16_t* up = (const uint16_t*)up_data;
        uint16_t* output = (uint16_t*)output_data;
        if (FusionCpuHasAvx512f()) {
            FusionCpuSwiGluF16Avx512(gate, up, output, (int64_t)count);
        } else {
            for (i = 0; i < count; ++i) {
                const float g = FusionCpuHalfToFloat(gate[i]);
                const float u = FusionCpuHalfToFloat(up[i]);
                const float product = FusionCpuHalfToFloat(FusionCpuFloatToHalf(u * g));
                const float sigmoid = FusionCpuHalfToFloat(
                    FusionCpuFloatToHalf(1.0f / (1.0f + expf(-g))));
                output[i] = FusionCpuFloatToHalf(product * sigmoid);
            }
        }
    } else {
        const float* gate = (const float*)gate_data;
        const float* up = (const float*)up_data;
        float* output = (float*)output_data;
        for (i = 0; i < count; ++i) {
            output[i] = gate[i] * (1.0f / (1.0f + expf(-gate[i]))) * up[i];
        }
    }
    return NULL;
}

static void FusionCpuRotateRow(const float* input, const float* cosine, const float* sine,
                               int rotary_dim, int head_size, bool interleaved, float* output) {
    int index;
    const int half = rotary_dim / 2;
    for (index = 0; index < rotary_dim; ++index) {
        int cache_index;
        int peer;
        bool negative;
        if (interleaved) {
            cache_index = (index / 2) % half;
            negative = (index & 1) != 0;
            peer = negative ? index - 1 : index + 1;
        } else {
            cache_index = index % half;
            negative = index >= half;
            peer = (index + half) % rotary_dim;
        }
        output[index] = input[index] * cosine[cache_index] +
                        (negative ? input[peer] : -input[peer]) * sine[cache_index];
    }
    for (; index < head_size; ++index) output[index] = input[index];
}

typedef struct FusionCpuGqaRun {
    FusionCpuGqaState* state;
    const float* q_base;
    const float* k_base;
    const float* v_base;
    const float* past_k;
    const float* past_v;
    const float* cosine;
    const float* sine;
    float* output;
    float* present_k;
    float* present_v;
    int64_t sequence;
    int64_t past_length;
    int64_t total;
    int64_t base;
    int64_t cache_total;
    int64_t cache_cap;
    int64_t head_size;
    int64_t q_hidden;
    int64_t q_stride;
    int64_t k_stride;
    int64_t v_stride;
    int rotary_dim;
    int rot_half;
    int group;
    float scale;
} FusionCpuGqaRun;

static inline int FusionCpuGqaWays(int count) {
    return FusionCpuMinInt(fusion_pool_size(), count > 0 ? count : 1);
}

static void FusionCpuGqaBuildOne(const FusionCpuGqaRun* run, int kv_head) {
    const int64_t present_offset = (int64_t)kv_head * run->total * run->head_size;
    const int64_t past_offset = (int64_t)kv_head * run->past_length * run->head_size;
    float* destination_k = run->present_k + present_offset;
    float* destination_v = run->present_v + present_offset;
    int64_t token;
    if (run->past_length > 0) {
        memcpy(destination_k, run->past_k + past_offset,
               (size_t)run->past_length * (size_t)run->head_size * sizeof(float));
        memcpy(destination_v, run->past_v + past_offset,
               (size_t)run->past_length * (size_t)run->head_size * sizeof(float));
    }
    for (token = 0; token < run->sequence; ++token) {
        const float* source_k = run->k_base + token * run->k_stride +
            (int64_t)kv_head * run->head_size;
        const float* source_v = run->v_base + token * run->v_stride +
            (int64_t)kv_head * run->head_size;
        float* key = destination_k + (run->past_length + token) * run->head_size;
        float* value = destination_v + (run->past_length + token) * run->head_size;
        if (run->state->do_rotary) {
            FusionCpuRotateRow(source_k,
                run->cosine + (run->past_length + token) * run->rot_half,
                run->sine + (run->past_length + token) * run->rot_half,
                run->rotary_dim, (int)run->head_size, run->state->rotary_interleaved, key);
        } else {
            memcpy(key, source_k, (size_t)run->head_size * sizeof(float));
        }
        memcpy(value, source_v, (size_t)run->head_size * sizeof(float));
    }
}

static void FusionCpuGqaAttendFp32One(const FusionCpuGqaRun* run, int head,
                                      float* scores, float* qrot) {
    const int kv_head = head / run->group;
    const float* keys = run->present_k + (int64_t)kv_head * run->total * run->head_size;
    const float* values = run->present_v + (int64_t)kv_head * run->total * run->head_size;
    int64_t token;
    for (token = 0; token < run->sequence; ++token) {
        const int64_t position = run->past_length + token;
        const int64_t limit = position + 1;
        const int64_t first = run->state->local_window > 0 &&
                              limit > run->state->local_window
            ? limit - run->state->local_window : 0;
        const float* query = run->q_base + token * run->q_stride +
            (int64_t)head * run->head_size;
        float maximum = -FLT_MAX;
        float sum = 0.0f;
        float* output = run->output + token * run->q_hidden + (int64_t)head * run->head_size;
        int64_t key_index;
        if (run->state->do_rotary) {
            FusionCpuRotateRow(query, run->cosine + position * run->rot_half,
                run->sine + position * run->rot_half, run->rotary_dim,
                (int)run->head_size, run->state->rotary_interleaved, qrot);
            query = qrot;
        }
        for (key_index = first; key_index < limit; ++key_index) {
            const float* key = keys + key_index * run->head_size;
            float dot = 0.0f;
            int64_t dimension;
            for (dimension = 0; dimension < run->head_size; ++dimension) {
                dot += query[dimension] * key[dimension];
            }
            scores[key_index] = dot * run->scale;
            if (scores[key_index] > maximum) maximum = scores[key_index];
        }
        for (key_index = first; key_index < limit; ++key_index) {
            scores[key_index] = expf(scores[key_index] - maximum);
            sum += scores[key_index];
        }
        memset(output, 0, (size_t)run->head_size * sizeof(float));
        for (key_index = first; key_index < limit; ++key_index) {
            const float weight = sum > 0.0f ? scores[key_index] / sum : 0.0f;
            const float* value = values + key_index * run->head_size;
            int64_t dimension;
            for (dimension = 0; dimension < run->head_size; ++dimension) {
                output[dimension] += weight * value[dimension];
            }
        }
    }
}

static void FusionCpuGqaAttendF16One(const FusionCpuGqaRun* run, int head,
                                     float* scores, float* qrot) {
    const int kv_head = head / run->group;
    const uint16_t* keys = run->state->kcache +
        (int64_t)kv_head * run->cache_cap * run->head_size;
    const uint16_t* values = run->state->vcache +
        (int64_t)kv_head * run->cache_cap * run->head_size;
    const bool use_dq = FusionCpuHasAvx512Dq();
    int64_t token;
    for (token = 0; token < run->sequence; ++token) {
        const int64_t position = run->base + token;
        const int64_t limit = position + 1;
        const int64_t first = run->state->local_window > 0 &&
                              limit > run->state->local_window
            ? limit - run->state->local_window : 0;
        const float* query = run->q_base + token * run->q_stride +
            (int64_t)head * run->head_size;
        float maximum = -FLT_MAX;
        float sum = 0.0f;
        float* output = run->output + token * run->q_hidden + (int64_t)head * run->head_size;
        int64_t key_index;
        if (run->state->do_rotary) {
            FusionCpuRotateRow(query, run->cosine + position * run->rot_half,
                run->sine + position * run->rot_half, run->rotary_dim,
                (int)run->head_size, run->state->rotary_interleaved, qrot);
            query = qrot;
        }
        for (key_index = first; key_index < limit; ++key_index) {
            const float dot = (use_dq
                ? GqaDotF16Avx512Dq(query, keys + key_index * run->head_size, run->head_size)
                : GqaDotF16Avx512(query, keys + key_index * run->head_size, run->head_size)) *
                run->scale;
            scores[key_index] = dot;
            if (dot > maximum) maximum = dot;
        }
        for (key_index = first; key_index < limit; ++key_index) {
            scores[key_index] = expf(scores[key_index] - maximum);
            sum += scores[key_index];
        }
        memset(output, 0, (size_t)run->head_size * sizeof(float));
        for (key_index = first; key_index < limit; ++key_index) {
            const float weight = sum > 0.0f ? scores[key_index] / sum : 0.0f;
            GqaAccumVF16Avx512(output, weight,
                values + key_index * run->head_size, run->head_size);
        }
    }
}

typedef struct FusionCpuGqaCallback {
    const FusionCpuGqaRun* run;
    float* scores;
    float* qrot;
    float* krot;
    int count;
} FusionCpuGqaCallback;

static void FusionCpuGqaBuildBody(void* opaque, int task, int nways) {
    FusionCpuGqaCallback* context = (FusionCpuGqaCallback*)opaque;
    const int begin = task * context->count / nways;
    const int end = (task + 1) * context->count / nways;
    int index;
    for (index = begin; index < end; ++index) FusionCpuGqaBuildOne(context->run, index);
}

static void FusionCpuGqaFp32AttendBody(void* opaque, int task, int nways) {
    FusionCpuGqaCallback* context = (FusionCpuGqaCallback*)opaque;
    const int begin = task * context->count / nways;
    const int end = (task + 1) * context->count / nways;
    float* scores = context->scores + (size_t)task * (size_t)context->run->total;
    float* qrot = context->qrot + (size_t)task * (size_t)context->run->head_size;
    int index;
    for (index = begin; index < end; ++index) {
        FusionCpuGqaAttendFp32One(context->run, index, scores, qrot);
    }
}

static void FusionCpuGqaFp32FusedBody(void* opaque, int task, int nways) {
    FusionCpuGqaCallback* context = (FusionCpuGqaCallback*)opaque;
    const int begin = task * context->count / nways;
    const int end = (task + 1) * context->count / nways;
    float* scores = context->scores + (size_t)task * (size_t)context->run->total;
    float* qrot = context->qrot + (size_t)task * (size_t)context->run->head_size;
    int kv_head;
    for (kv_head = begin; kv_head < end; ++kv_head) {
        int head;
        FusionCpuGqaBuildOne(context->run, kv_head);
        for (head = kv_head * context->run->group;
             head < (kv_head + 1) * context->run->group; ++head) {
            FusionCpuGqaAttendFp32One(context->run, head, scores, qrot);
        }
    }
}

static void FusionCpuGqaReseedBody(void* opaque, int task, int nways) {
    FusionCpuGqaCallback* context = (FusionCpuGqaCallback*)opaque;
    const FusionCpuGqaRun* run = context->run;
    const int begin = task * context->count / nways;
    const int end = (task + 1) * context->count / nways;
    int kv_head;
    for (kv_head = begin; kv_head < end; ++kv_head) {
        uint16_t* keys = run->state->kcache +
            (int64_t)kv_head * run->cache_cap * run->head_size;
        uint16_t* values = run->state->vcache +
            (int64_t)kv_head * run->cache_cap * run->head_size;
        const float* past_k = run->past_k +
            (int64_t)kv_head * run->past_length * run->head_size;
        const float* past_v = run->past_v +
            (int64_t)kv_head * run->past_length * run->head_size;
        int64_t i;
        for (i = 0; i < run->past_length * run->head_size; ++i) {
            keys[i] = FusionCpuFloatToHalf(past_k[i]);
            values[i] = FusionCpuFloatToHalf(past_v[i]);
        }
    }
}

static void FusionCpuGqaAppendBody(void* opaque, int task, int nways) {
    FusionCpuGqaCallback* context = (FusionCpuGqaCallback*)opaque;
    const FusionCpuGqaRun* run = context->run;
    const int begin = task * context->count / nways;
    const int end = (task + 1) * context->count / nways;
    float* krot = context->krot + (size_t)task * (size_t)run->head_size;
    int kv_head;
    for (kv_head = begin; kv_head < end; ++kv_head) {
        uint16_t* keys = run->state->kcache +
            (int64_t)kv_head * run->cache_cap * run->head_size;
        uint16_t* values = run->state->vcache +
            (int64_t)kv_head * run->cache_cap * run->head_size;
        int64_t token;
        for (token = 0; token < run->sequence; ++token) {
            const float* source_k = run->k_base + token * run->k_stride +
                (int64_t)kv_head * run->head_size;
            const float* source_v = run->v_base + token * run->v_stride +
                (int64_t)kv_head * run->head_size;
            uint16_t* key = keys + (run->base + token) * run->head_size;
            uint16_t* value = values + (run->base + token) * run->head_size;
            int64_t dimension;
            if (run->state->do_rotary) {
                FusionCpuRotateRow(source_k,
                    run->cosine + (run->base + token) * run->rot_half,
                    run->sine + (run->base + token) * run->rot_half,
                    run->rotary_dim, (int)run->head_size,
                    run->state->rotary_interleaved, krot);
                for (dimension = 0; dimension < run->head_size; ++dimension) {
                    key[dimension] = FusionCpuFloatToHalf(krot[dimension]);
                }
            } else {
                for (dimension = 0; dimension < run->head_size; ++dimension) {
                    key[dimension] = FusionCpuFloatToHalf(source_k[dimension]);
                }
            }
            for (dimension = 0; dimension < run->head_size; ++dimension) {
                value[dimension] = FusionCpuFloatToHalf(source_v[dimension]);
            }
        }
    }
}

static void FusionCpuGqaF16AttendBody(void* opaque, int task, int nways) {
    FusionCpuGqaCallback* context = (FusionCpuGqaCallback*)opaque;
    const FusionCpuGqaRun* run = context->run;
    const int begin = task * context->count / nways;
    const int end = (task + 1) * context->count / nways;
    float* scores = context->scores + (size_t)task * (size_t)run->cache_total;
    float* qrot = context->qrot + (size_t)task * (size_t)run->head_size;
    int kv_head;
    for (kv_head = begin; kv_head < end; ++kv_head) {
        int head;
        for (head = kv_head * run->group; head < (kv_head + 1) * run->group; ++head) {
            FusionCpuGqaAttendF16One(run, head, scores, qrot);
        }
    }
}

static bool FusionCpuGqaEnsureCache(FusionCpuGqaState* state, int kv_heads,
                                    int64_t head_size, int64_t need_rows,
                                    int64_t keep_rows) {
    int64_t new_cap;
    size_t count;
    uint16_t* keys;
    uint16_t* values;
    int kv_head;
    if (state->cache_cap > 0 &&
        (state->cache_head_size != head_size || state->cache_kv_heads != kv_heads)) {
        free(state->kcache);
        free(state->vcache);
        state->kcache = NULL;
        state->vcache = NULL;
        state->cache_cap = 0;
        state->cache_len = 0;
        keep_rows = 0;
    }
    if (need_rows <= state->cache_cap) return true;
    new_cap = state->cache_cap > 0 ? state->cache_cap : 256;
    while (new_cap < need_rows) {
        if (new_cap > INT64_MAX / 2) {
            new_cap = need_rows;
            break;
        }
        new_cap *= 2;
    }
    if (!FusionCpuSizeProduct((size_t)kv_heads, (size_t)new_cap, &count) ||
        !FusionCpuSizeProduct(count, (size_t)head_size, &count)) {
        return false;
    }
    keys = (uint16_t*)FusionCpuMallocArray(count, sizeof(*keys));
    values = (uint16_t*)FusionCpuMallocArray(count, sizeof(*values));
    if (keys == NULL || values == NULL) {
        free(keys);
        free(values);
        return false;
    }
    if (keep_rows > 0 && state->cache_cap > 0) {
        for (kv_head = 0; kv_head < kv_heads; ++kv_head) {
            memcpy(keys + (int64_t)kv_head * new_cap * head_size,
                   state->kcache + (int64_t)kv_head * state->cache_cap * head_size,
                   (size_t)keep_rows * (size_t)head_size * sizeof(uint16_t));
            memcpy(values + (int64_t)kv_head * new_cap * head_size,
                   state->vcache + (int64_t)kv_head * state->cache_cap * head_size,
                   (size_t)keep_rows * (size_t)head_size * sizeof(uint16_t));
        }
    }
    free(state->kcache);
    free(state->vcache);
    state->kcache = keys;
    state->vcache = values;
    state->cache_cap = new_cap;
    state->cache_head_size = head_size;
    state->cache_kv_heads = kv_heads;
    return true;
}

// Widen a contiguous fp16 (uint16_t) buffer into a freshly malloc'd fp32 buffer.
// Returns NULL on OOM. Used by the q4f16 GQA path to bridge fp16 model I/O into
// the fp32 attention workers.
static float* FusionCpuGqaWidenHalf(const void* src, int64_t count) {
    float* dst = (float*)FusionCpuMallocArray((size_t)count, sizeof(float));
    int64_t i;
    if (dst == NULL) return NULL;
    if (FusionCpuHasAvx512f()) {
        FusionCpuWidenHalfToFloatAvx512(dst, (const uint16_t*)src, count);
        return dst;
    }
    for (i = 0; i < count; ++i) {
        dst[i] = FusionCpuHalfToFloat(((const uint16_t*)src)[i]);
    }
    return dst;
}

static OrtStatus* ORT_API_CALL FusionCpuGqaCompute(void* opaque,
                                                    OrtKernelContext* context) {
    FusionCpuGqaState* state = (FusionCpuGqaState*)opaque;
    const OrtValue* qkv_value;
    const OrtValue* past_k_value;
    const OrtValue* past_v_value;
    const OrtValue* key_value;
    const OrtValue* value_value;
    ONNXTensorElementDataType type;
    const int64_t* qkv_shape;
    const int64_t* past_k_shape;
    const int64_t* past_v_shape;
    const int64_t* key_shape;
    const int64_t* value_shape;
    const int64_t* aux_shape;
    size_t qkv_rank;
    size_t past_k_rank;
    size_t past_v_rank;
    size_t key_rank;
    size_t value_rank;
    size_t aux_rank;
    const void* qkv_void;
    const void* past_k_void;
    const void* past_v_void;
    const void* aux_void;
    const float* q_base;
    const float* k_base;
    const float* v_base;
    const float* cosine = NULL;
    const float* sine = NULL;
    int64_t batch;
    int64_t sequence;
    int64_t head_size;
    int64_t q_hidden;
    int64_t kv_hidden;
    int64_t q_stride;
    int64_t k_stride;
    int64_t v_stride;
    int64_t past_length;
    int64_t total;
    int64_t own_base;
    int64_t base;
    int64_t cache_total;
    int rotary_dim = 0;
    int rot_half = 0;
    int group;
    bool unpacked;
    bool own_kv;
    int64_t output_shape[3];
    int64_t present_shape[4];
    OrtValue* unused_output;
    void* output_void;
    void* present_k_void;
    void* present_v_void;
    const void* key_void = NULL;
    const void* value_void = NULL;
    FusionCpuGqaRun run;
    FusionCpuGqaCallback callback;
    float* scores = NULL;
    float* qrot = NULL;
    float* krot = NULL;
    int ways;
    OrtStatus* status;
    // fp16 (q4f16) I/O bridging: the workers operate on fp32, so on the fp16 path
    // we widen inputs into these scratch buffers up front and narrow the fp32
    // output/present scratch back to fp16 at the end. All NULL / unused on fp32.
    bool io_fp16 = false;
    float* qkv_f32 = NULL;
    float* key_f32 = NULL;
    float* value_f32 = NULL;
    float* past_k_f32 = NULL;
    float* past_v_f32 = NULL;
    float* out_f32 = NULL;
    float* present_k_f32 = NULL;
    float* present_v_f32 = NULL;
    bool present_written = false;
    ONNXTensorElementDataType io_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

    {
        size_t input_count;
        size_t output_count;
        status = state->api->KernelContext_GetInputCount(context, &input_count);
        if (status != NULL) return status;
        status = state->api->KernelContext_GetOutputCount(context, &output_count);
        if (status != NULL) return status;
        if (output_count != 3 || state->idx_qkv < 0 || state->idx_past_key < 0 ||
            state->idx_past_value < 0 || state->idx_out < 0 ||
            state->idx_present_key < 0 || state->idx_present_value < 0 ||
            (size_t)state->idx_qkv >= input_count ||
            (size_t)state->idx_past_key >= input_count ||
            (size_t)state->idx_past_value >= input_count ||
            (size_t)state->idx_out >= output_count ||
            (size_t)state->idx_present_key >= output_count ||
            (size_t)state->idx_present_value >= output_count) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
                "GQA: fused-node I/O indices are invalid.");
        }
    }
    status = FusionCpuGetInput(state->api, context, (size_t)state->idx_qkv, &qkv_value,
                               &type, &qkv_shape, &qkv_rank, &qkv_void);
    if (status != NULL) return status;
    if ((type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
         type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) || qkv_rank != 3) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: query/qkv must be rank-3 float32/float16.");
    }
    io_type = type;
    io_fp16 = (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    status = FusionCpuGetInput(state->api, context, (size_t)state->idx_past_key, &past_k_value,
                               &type, &past_k_shape, &past_k_rank, &past_k_void);
    if (status != NULL) return status;
    if (type != io_type || past_k_rank != 4) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: past_key dtype/rank mismatch.");
    }
    status = FusionCpuGetInput(state->api, context, (size_t)state->idx_past_value, &past_v_value,
                               &type, &past_v_shape, &past_v_rank, &past_v_void);
    if (status != NULL) return status;
    if (type != io_type || past_v_rank != 4) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: past_value dtype/rank mismatch.");
    }
    batch = qkv_shape[0];
    sequence = qkv_shape[1];
    if (batch != 1 || sequence < 0 || state->num_heads <= 0 || state->kv_num_heads <= 0 ||
        state->num_heads % state->kv_num_heads != 0) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: unsupported batch, sequence, or head count.");
    }
    group = state->num_heads / state->kv_num_heads;
    unpacked = state->idx_key >= 0 || state->idx_value >= 0;
    if (unpacked && (state->idx_key < 0 || state->idx_value < 0)) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: only one of key/value is mapped.");
    }
    q_base = (const float*)qkv_void;
    if (unpacked) {
        if (qkv_shape[2] <= 0 || qkv_shape[2] % state->num_heads != 0) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: invalid query hidden size.");
        }
        head_size = qkv_shape[2] / state->num_heads;
        q_hidden = qkv_shape[2];
        kv_hidden = (int64_t)state->kv_num_heads * head_size;
        status = FusionCpuGetInput(state->api, context, (size_t)state->idx_key, &key_value,
                                   &type, &key_shape, &key_rank, &aux_void);
        if (status != NULL) return status;
        if (type != io_type || key_rank != 3 ||
            key_shape[0] != batch || key_shape[1] != sequence || key_shape[2] != kv_hidden) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: invalid key tensor.");
        }
        k_base = (const float*)aux_void;
        key_void = aux_void;
        status = FusionCpuGetInput(state->api, context, (size_t)state->idx_value, &value_value,
                                   &type, &value_shape, &value_rank, &aux_void);
        if (status != NULL) return status;
        if (type != io_type || value_rank != 3 ||
            value_shape[0] != batch || value_shape[1] != sequence || value_shape[2] != kv_hidden) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: invalid value tensor.");
        }
        v_base = (const float*)aux_void;
        value_void = aux_void;
        q_stride = q_hidden;
        k_stride = kv_hidden;
        v_stride = kv_hidden;
    } else {
        const int64_t packed_hidden = qkv_shape[2];
        if (packed_hidden <= 0 ||
            packed_hidden % (state->num_heads + 2 * state->kv_num_heads) != 0) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: invalid packed QKV hidden size.");
        }
        head_size = packed_hidden / (state->num_heads + 2 * state->kv_num_heads);
        q_hidden = (int64_t)state->num_heads * head_size;
        kv_hidden = (int64_t)state->kv_num_heads * head_size;
        k_base = q_base + q_hidden;
        v_base = q_base + q_hidden + kv_hidden;
        q_stride = packed_hidden;
        k_stride = packed_hidden;
        v_stride = packed_hidden;
    }
    past_length = past_k_shape[2];
    if (head_size <= 0 || past_length < 0 || past_k_shape[0] != 1 ||
        past_k_shape[1] != state->kv_num_heads || past_k_shape[3] != head_size ||
        !FusionCpuShapeEqual(past_k_shape, past_k_rank, past_v_shape, past_v_rank) ||
        past_length > INT64_MAX - sequence) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: past cache shape is invalid.");
    }
    total = past_length + sequence;
    own_kv = FusionCpuGqaOwnKvEnabled();
    own_base = past_length;
    if (own_kv) {
        int64_t total_kv;
        size_t seq_count;
        if (!(FusionCpuGqaFp16KvEnabled() && FusionCpuHasAvx512f()) ||
            state->idx_seqlens_k < 0) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
                "GQA own-KV requires fp16 KV caching and a mapped seqlens_k input.");
        }
        status = FusionCpuGetInput(state->api, context, (size_t)state->idx_seqlens_k, &key_value,
                                   &type, &aux_shape, &aux_rank, &aux_void);
        if (status != NULL) return status;
        if (!FusionCpuElementCount(aux_shape, aux_rank, &seq_count) || seq_count == 0) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA own-KV: invalid seqlens_k.");
        }
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            total_kv = (int64_t)((const int32_t*)aux_void)[0] + 1;
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            total_kv = ((const int64_t*)aux_void)[0] + 1;
        } else {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
                "GQA own-KV: seqlens_k must be int32 or int64.");
        }
        own_base = total_kv - sequence;
        if (own_base < 0) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
                "GQA own-KV: sequence exceeds seqlens_k.");
        }
    }
    base = own_kv ? own_base : past_length;
    if (base > INT64_MAX - sequence) {
        return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: KV length overflows.");
    }
    cache_total = base + sequence;
    if (state->do_rotary) {
        const int64_t* cosine_shape;
        const int64_t* sine_shape;
        size_t cosine_rank;
        size_t sine_rank;
        status = FusionCpuGetInput(state->api, context, (size_t)state->idx_cos, &key_value,
                                   &type, &cosine_shape, &cosine_rank, &aux_void);
        if (status != NULL) return status;
        if (type != io_type || cosine_rank != 2 ||
            cosine_shape[1] <= 0 || cosine_shape[1] > INT_MAX / 2) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: invalid cosine cache.");
        }
        cosine = (const float*)aux_void;
        rotary_dim = (int)(cosine_shape[1] * 2);
        if (rotary_dim > head_size || cache_total > cosine_shape[0]) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT,
                "GQA: total_sequence_length exceeds cos_cache rows (rotary position out of range).");
        }
        if (io_fp16) {
            int64_t count = cosine_shape[0] * cosine_shape[1];
            if (state->cos_f32 == NULL || state->cos_f32_count != count) {
                float* widened = (float*)FusionCpuMallocArray((size_t)count, sizeof(float));
                if (widened == NULL) {
                    return state->api->CreateStatus(ORT_EP_FAIL, "GQA: cos cache widening failed.");
                }
                for (int64_t i = 0; i < count; ++i) {
                    widened[i] = FusionCpuHalfToFloat(((const uint16_t*)aux_void)[i]);
                }
                free(state->cos_f32);
                state->cos_f32 = widened;
                state->cos_f32_count = count;
            }
            cosine = state->cos_f32;
        }
        status = FusionCpuGetInput(state->api, context, (size_t)state->idx_sin, &value_value,
                                   &type, &sine_shape, &sine_rank, &aux_void);
        if (status != NULL) return status;
        if (type != io_type || sine_rank != 2 ||
            sine_shape[0] < cache_total || sine_shape[1] != cosine_shape[1]) {
            return state->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA: invalid sine cache.");
        }
        sine = (const float*)aux_void;
        if (io_fp16) {
            int64_t count = sine_shape[0] * sine_shape[1];
            if (state->sin_f32 == NULL || state->sin_f32_count != count) {
                float* widened = (float*)FusionCpuMallocArray((size_t)count, sizeof(float));
                if (widened == NULL) {
                    return state->api->CreateStatus(ORT_EP_FAIL, "GQA: sin cache widening failed.");
                }
                for (int64_t i = 0; i < count; ++i) {
                    widened[i] = FusionCpuHalfToFloat(((const uint16_t*)aux_void)[i]);
                }
                free(state->sin_f32);
                state->sin_f32 = widened;
                state->sin_f32_count = count;
            }
            sine = state->sin_f32;
        }
        rot_half = rotary_dim / 2;
    }
    output_shape[0] = 1;
    output_shape[1] = sequence;
    output_shape[2] = q_hidden;
    present_shape[0] = 1;
    present_shape[1] = state->kv_num_heads;
    present_shape[2] = total;
    present_shape[3] = head_size;
    status = FusionCpuGetOutput(state->api, context, (size_t)state->idx_out, output_shape, 3,
                                &unused_output, &output_void);
    if (status != NULL) return status;
    status = FusionCpuGetOutput(state->api, context, (size_t)state->idx_present_key,
                                present_shape, 4, &unused_output, &present_k_void);
    if (status != NULL) return status;
    status = FusionCpuGetOutput(state->api, context, (size_t)state->idx_present_value,
                                present_shape, 4, &unused_output, &present_v_void);
    if (status != NULL) return status;
    if (io_fp16) {
        if (unpacked) {
            qkv_f32 = FusionCpuGqaWidenHalf(qkv_void, sequence * q_hidden);
            key_f32 = FusionCpuGqaWidenHalf(key_void, sequence * kv_hidden);
            value_f32 = FusionCpuGqaWidenHalf(value_void, sequence * kv_hidden);
            if (qkv_f32 == NULL || key_f32 == NULL || value_f32 == NULL) {
                status = state->api->CreateStatus(ORT_EP_FAIL, "GQA: fp16 input widening failed.");
                goto gqa_done;
            }
            q_base = qkv_f32;
            k_base = key_f32;
            v_base = value_f32;
        } else {
            qkv_f32 = FusionCpuGqaWidenHalf(qkv_void, sequence * q_stride);
            if (qkv_f32 == NULL) {
                status = state->api->CreateStatus(ORT_EP_FAIL, "GQA: fp16 input widening failed.");
                goto gqa_done;
            }
            q_base = qkv_f32;
            k_base = qkv_f32 + q_hidden;
            v_base = qkv_f32 + q_hidden + kv_hidden;
        }
        out_f32 = (float*)FusionCpuMallocArray((size_t)(sequence * q_hidden), sizeof(float));
        if (out_f32 == NULL) {
            status = state->api->CreateStatus(ORT_EP_FAIL, "GQA: fp16 output scratch allocation failed.");
            goto gqa_done;
        }
    }
    memset(&run, 0, sizeof(run));
    run.state = state;
    run.q_base = q_base;
    run.k_base = k_base;
    run.v_base = v_base;
    run.past_k = io_fp16 ? NULL : (const float*)past_k_void;
    run.past_v = io_fp16 ? NULL : (const float*)past_v_void;
    run.cosine = cosine;
    run.sine = sine;
    run.output = io_fp16 ? out_f32 : (float*)output_void;
    run.present_k = io_fp16 ? NULL : (float*)present_k_void;
    run.present_v = io_fp16 ? NULL : (float*)present_v_void;
    run.sequence = sequence;
    run.past_length = past_length;
    run.total = total;
    run.base = base;
    run.cache_total = cache_total;
    run.head_size = head_size;
    run.q_hidden = q_hidden;
    run.q_stride = q_stride;
    run.k_stride = k_stride;
    run.v_stride = v_stride;
    run.rotary_dim = rotary_dim;
    run.rot_half = rot_half;
    run.group = group;
    run.scale = state->scale != 0.0f ? state->scale : 1.0f / sqrtf((float)head_size);
    if (FusionCpuTraceEnabled()) {
        fprintf(stderr,
            "[FusionCpuEP][trace] GQA sequence=%lld past=%lld base=%lld heads=%d kv=%d "
            "hs=%lld fp16=%d own_kv=%d\n",
            (long long)sequence, (long long)past_length, (long long)base,
            state->num_heads, state->kv_num_heads, (long long)head_size,
            FusionCpuGqaFp16KvEnabled() && FusionCpuHasAvx512f(), own_kv);
        fflush(stderr);
    }

    if (FusionCpuGqaFp16KvEnabled() && FusionCpuHasAvx512f()) {
        const bool reseed = own_kv ? false : state->cache_len != past_length;
        int64_t keep;
        if (own_kv) {
            if (base > state->cache_len) {
                status = state->api->CreateStatus(ORT_INVALID_ARGUMENT,
                    "GQA own-KV: position gap beyond cached history.");
                goto gqa_done;
            }
            state->cache_len = base;
        }
        keep = reseed ? 0 : state->cache_len;
        if (!FusionCpuGqaEnsureCache(state, state->kv_num_heads, head_size,
                                     cache_total, keep)) {
            status = state->api->CreateStatus(ORT_EP_FAIL, "GQA: fp16 KV cache allocation failed.");
            goto gqa_done;
        }
        run.cache_cap = state->cache_cap;
        memset(&callback, 0, sizeof(callback));
        callback.run = &run;
        callback.count = state->kv_num_heads;
        ways = FusionCpuGqaWays(state->kv_num_heads);
        if (reseed && past_length > 0) {
            if (io_fp16) {
                int64_t pcount = kv_hidden * past_length;
                past_k_f32 = FusionCpuGqaWidenHalf(past_k_void, pcount);
                past_v_f32 = FusionCpuGqaWidenHalf(past_v_void, pcount);
                if (past_k_f32 == NULL || past_v_f32 == NULL) {
                    status = state->api->CreateStatus(ORT_EP_FAIL, "GQA: fp16 past widening failed.");
                    goto gqa_done;
                }
                run.past_k = past_k_f32;
                run.past_v = past_v_f32;
            }
            fusion_pool_parallel_for(ways, FusionCpuGqaReseedBody, &callback);
        }
        qrot = (float*)FusionCpuMallocArray((size_t)ways * (size_t)head_size, sizeof(*qrot));
        krot = (float*)FusionCpuMallocArray((size_t)ways * (size_t)head_size, sizeof(*krot));
        scores = (float*)FusionCpuMallocArray((size_t)ways * (size_t)cache_total, sizeof(*scores));
        if (qrot == NULL || krot == NULL || scores == NULL) {
            status = state->api->CreateStatus(ORT_EP_FAIL, "GQA: fp16 worker scratch allocation failed.");
            goto gqa_done;
        }
        callback.qrot = qrot;
        callback.krot = krot;
        callback.scores = scores;
        fusion_pool_parallel_for(ways, FusionCpuGqaAppendBody, &callback);
        state->cache_len = cache_total;
        fusion_pool_parallel_for(ways, FusionCpuGqaF16AttendBody, &callback);
        status = NULL;
        goto gqa_done;
    }
    if (own_kv) {
        status = state->api->CreateStatus(ORT_INVALID_ARGUMENT,
            "GQA own-KV requires AVX-512 fp16 KV caching.");
        goto gqa_done;
    }
    if (io_fp16) {
        // fp32 fallback paths (fused / two-phase) read past and write present as
        // fp32, so widen past and provide fp32 present scratch (narrowed at the end).
        if (past_length > 0) {
            int64_t pcount = kv_hidden * past_length;
            past_k_f32 = FusionCpuGqaWidenHalf(past_k_void, pcount);
            past_v_f32 = FusionCpuGqaWidenHalf(past_v_void, pcount);
            if (past_k_f32 == NULL || past_v_f32 == NULL) {
                status = state->api->CreateStatus(ORT_EP_FAIL, "GQA: fp16 past widening failed.");
                goto gqa_done;
            }
            run.past_k = past_k_f32;
            run.past_v = past_v_f32;
        }
        {
            int64_t prcount = kv_hidden * total;
            present_k_f32 = (float*)FusionCpuMallocArray((size_t)prcount, sizeof(float));
            present_v_f32 = (float*)FusionCpuMallocArray((size_t)prcount, sizeof(float));
            if (present_k_f32 == NULL || present_v_f32 == NULL) {
                status = state->api->CreateStatus(ORT_EP_FAIL, "GQA: fp16 present scratch allocation failed.");
                goto gqa_done;
            }
            run.present_k = present_k_f32;
            run.present_v = present_v_f32;
            present_written = true;
        }
    }
    if (FusionCpuGqaUseFusedPath()) {
        ways = FusionCpuGqaWays(state->kv_num_heads);
        scores = (float*)FusionCpuMallocArray((size_t)ways * (size_t)total, sizeof(*scores));
        qrot = (float*)FusionCpuMallocArray((size_t)ways * (size_t)head_size, sizeof(*qrot));
        if (scores == NULL || qrot == NULL) {
            status = state->api->CreateStatus(ORT_EP_FAIL, "GQA: fused worker scratch allocation failed.");
            goto gqa_done;
        }
        memset(&callback, 0, sizeof(callback));
        callback.run = &run;
        callback.scores = scores;
        callback.qrot = qrot;
        callback.count = state->kv_num_heads;
        fusion_pool_parallel_for(ways, FusionCpuGqaFp32FusedBody, &callback);
    } else {
        ways = FusionCpuGqaWays(state->kv_num_heads);
        memset(&callback, 0, sizeof(callback));
        callback.run = &run;
        callback.count = state->kv_num_heads;
        fusion_pool_parallel_for(ways, FusionCpuGqaBuildBody, &callback);
        ways = FusionCpuGqaWays(state->num_heads);
        scores = (float*)FusionCpuMallocArray((size_t)ways * (size_t)total, sizeof(*scores));
        qrot = (float*)FusionCpuMallocArray((size_t)ways * (size_t)head_size, sizeof(*qrot));
        if (scores == NULL || qrot == NULL) {
            status = state->api->CreateStatus(ORT_EP_FAIL, "GQA: worker scratch allocation failed.");
            goto gqa_done;
        }
        callback.scores = scores;
        callback.qrot = qrot;
        callback.count = state->num_heads;
        fusion_pool_parallel_for(ways, FusionCpuGqaFp32AttendBody, &callback);
    }
    status = NULL;

gqa_done:
    if (status == NULL && io_fp16) {
        int64_t ocount = sequence * q_hidden;
        int64_t i;
        for (i = 0; i < ocount; ++i) {
            ((uint16_t*)output_void)[i] = FusionCpuFloatToHalf(out_f32[i]);
        }
        if (present_written) {
            int64_t prcount = kv_hidden * total;
            for (i = 0; i < prcount; ++i) {
                ((uint16_t*)present_k_void)[i] = FusionCpuFloatToHalf(present_k_f32[i]);
                ((uint16_t*)present_v_void)[i] = FusionCpuFloatToHalf(present_v_f32[i]);
            }
        }
    }
    free(scores);
    free(qrot);
    free(krot);
    free(qkv_f32);
    free(key_f32);
    free(value_f32);
    free(past_k_f32);
    free(past_v_f32);
    free(out_f32);
    free(present_k_f32);
    free(present_v_f32);
    return status;
}

static void ORT_API_CALL FusionCpuFreeMatMulState(void* opaque) {
    FusionCpuMatMulNBitsState* state = (FusionCpuMatMulNBitsState*)opaque;
    if (state == NULL) return;
    FusionCpuMatMulReleasePrepack(state);
    free(state);
}

static void ORT_API_CALL FusionCpuFreeGqaOptimizedState(void* opaque) {
    FusionCpuGqaState* state = (FusionCpuGqaState*)opaque;
    if (state == NULL) return;
    free(state->kcache);
    free(state->vcache);
    free(state->cos_f32);
    free(state->sin_f32);
    free(state);
}
