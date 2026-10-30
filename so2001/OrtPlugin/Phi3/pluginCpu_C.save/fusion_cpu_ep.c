// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#include "fusion_cpu_ep.h"
#include "fusion_cpu_ep_sync.h"
#include "fusion_cpu_extra_ops.h"
#include "fp16_utils.h"
#include "matmulnbits_avx512.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#include <intrin.h>
#endif

static INIT_ONCE g_fusion_pool_once = INIT_ONCE_STATIC_INIT;
static FusionThreadPool* g_fusion_pool;

static int FusionCpuThreadCount(void) {
    const char* value = getenv("ORT_FUSIONCPU_MATMULNBITS_THREADS");
    if (value != NULL && value[0] != '\0') {
        char* end = NULL;
        long parsed = strtol(value, &end, 10);
        if (end != value && parsed >= 1) return (int)(parsed > 256 ? 256 : parsed);
    }
#if defined(_WIN32)
    {
        DWORD count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
        if (count == 0) count = 1;
        return (int)(count > 64 ? 64 : count);
    }
#else
    return 1;
#endif
}

int fusion_ccx_pin_enabled(void) {
    const char* value = getenv("ORT_FUSIONCPU_CCX_PIN");
    return value == NULL || value[0] != '0';
}

void fusion_pin_worker_to_core(int worker_idx, int nthreads) {
#if defined(_WIN32)
    DWORD logical = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    DWORD physical;
    DWORD core;
    DWORD logical_processor;
    GROUP_AFFINITY affinity;
    if (logical == 0) logical = 1;
    physical = logical / 2;
    if (physical == 0) physical = logical;
    if (nthreads <= 0) nthreads = 1;
    core = (DWORD)(((uint64_t)(unsigned)worker_idx * physical) / (unsigned)nthreads);
    logical_processor = (core * 2) % logical;
    memset(&affinity, 0, sizeof(affinity));
    affinity.Group = (WORD)(logical_processor / 64);
    affinity.Mask = (KAFFINITY)(1ull << (logical_processor % 64));
    SetThreadGroupAffinity(GetCurrentThread(), &affinity, NULL);
#else
    (void)worker_idx;
    (void)nthreads;
#endif
}

static BOOL CALLBACK FusionCpuInitializeThreadPool(
    PINIT_ONCE once, PVOID parameter, PVOID* context) {
    (void)once;
    (void)parameter;
    (void)context;
    g_fusion_pool = fusion_pool_create(FusionCpuThreadCount());
    return TRUE;
}

static FusionThreadPool* FusionCpuThreadPool(void) {
    InitOnceExecuteOnce(&g_fusion_pool_once, FusionCpuInitializeThreadPool, NULL, NULL);
    return g_fusion_pool;
}

void FusionCpuParallelFor(int count, fusion_body_fn fn, void* context) {
    FusionThreadPool* pool;
    int ways;
    if (count <= 1) {
        fn(context, 0, 1);
        return;
    }
    pool = FusionCpuThreadPool();
    ways = pool != NULL ? fusion_pool_size(pool) : 1;
    if (ways > count) ways = count;
    fusion_pool_parallel_for(pool, ways, fn, context);
}

typedef struct FusionCpuMulState {
    const OrtApi* api;
} FusionCpuMulState;

typedef struct FusionCpuSigmoidState {
    const OrtApi* api;
} FusionCpuSigmoidState;

typedef struct FusionCpuMatMulNBitsState {
    const OrtApi* api;
    int bits;
    int block_size;
    int idx_a;
    int idx_b;
    int idx_scales;
    int idx_zp;
} FusionCpuMatMulNBitsState;

typedef struct FusionCpuGqaState {
    const OrtApi* api;
    int num_heads;
    int kv_num_heads;
    float scale;
    bool do_rotary;
    bool rotary_interleaved;
    int idx_qkv;
    int idx_key;
    int idx_value;
    int idx_past_key;
    int idx_past_value;
    int idx_cos;
    int idx_sin;
    int idx_out;
    int idx_present_key;
    int idx_present_value;
    int64_t local_window;
} FusionCpuGqaState;

typedef struct FusionCpuNodeComputeInfo {
    OrtNodeComputeInfo base;
    FusionCpuEp* ep;
} FusionCpuNodeComputeInfo;

static char* FusionCpuDuplicateString(const char* text) {
    size_t length;
    char* copy;
    if (text == NULL) return NULL;
    length = strlen(text) + 1;
    copy = (char*)malloc(length);
    if (copy != NULL) memcpy(copy, text, length);
    return copy;
}

#if defined(_WIN32)
static CRITICAL_SECTION* FusionCpuKernelLock(FusionCpuEp* ep) {
    return (CRITICAL_SECTION*)ep->kernel_lock;
}
#endif

static void FusionCpuLockKernels(FusionCpuEp* ep) {
#if defined(_WIN32)
    EnterCriticalSection(FusionCpuKernelLock(ep));
#else
    (void)ep;
#endif
}

static void FusionCpuUnlockKernels(FusionCpuEp* ep) {
#if defined(_WIN32)
    LeaveCriticalSection(FusionCpuKernelLock(ep));
#else
    (void)ep;
#endif
}

FusionCpuKernelEntry* FusionCpuEp_FindKernel(FusionCpuEp* ep, const char* name) {
    FusionCpuKernelEntry* entry;
    if (ep == NULL || name == NULL) return NULL;
    FusionCpuLockKernels(ep);
    entry = ep->kernels;
    while (entry != NULL && strcmp(entry->name, name) != 0) entry = entry->next;
    FusionCpuUnlockKernels(ep);
    return entry;
}

OrtStatus* FusionCpuEp_AddKernel(FusionCpuEp* ep, const char* name, void* state,
                                 FusionCpuComputeFn compute, FusionCpuFreeFn destroy) {
    FusionCpuKernelEntry* entry;
    if (ep == NULL || name == NULL || compute == NULL || destroy == NULL) {
        return ep != NULL ? ep->ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                                       "FusionCPU EP: invalid kernel registration.")
                          : NULL;
    }
    entry = (FusionCpuKernelEntry*)calloc(1, sizeof(*entry));
    if (entry == NULL) {
        return ep->ort_api->CreateStatus(ORT_EP_FAIL,
                                         "FusionCPU EP: out of memory registering kernel.");
    }
    entry->name = FusionCpuDuplicateString(name);
    if (entry->name == NULL) {
        free(entry);
        return ep->ort_api->CreateStatus(ORT_EP_FAIL,
                                         "FusionCPU EP: out of memory copying kernel name.");
    }
    entry->state = state;
    entry->compute = compute;
    entry->destroy = destroy;
    FusionCpuLockKernels(ep);
    entry->next = ep->kernels;
    ep->kernels = entry;
    FusionCpuUnlockKernels(ep);
    return NULL;
}

static void FusionCpuFreeKernelEntries(FusionCpuEp* ep) {
    FusionCpuKernelEntry* entry = ep->kernels;
    while (entry != NULL) {
        FusionCpuKernelEntry* next = entry->next;
        entry->destroy(entry->state);
        free(entry->name);
        free(entry);
        entry = next;
    }
    ep->kernels = NULL;
}

static const char* ORT_API_CALL FusionCpuEpGetName(const OrtEp* ep) {
    return ((const FusionCpuEp*)ep)->name;
}

static bool FusionCpuHasAvx512f(void) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    static int initialized = 0;
    static bool has_avx512f = false;
    int info[4];
    unsigned __int64 xcr0;
    if (initialized) return has_avx512f;
    __cpuid(info, 0);
    if (info[0] >= 7) {
        __cpuidex(info, 1, 0);
        if ((info[2] & (1 << 27)) != 0 && (info[2] & (1 << 28)) != 0) {
            xcr0 = _xgetbv(0);
            if ((xcr0 & 0xE6u) == 0xE6u) {
                __cpuidex(info, 7, 0);
                has_avx512f = (info[1] & (1 << 16)) != 0;
            }
        }
    }
    initialized = 1;
    return has_avx512f;
#else
    return false;
#endif
}

static bool FusionCpuForceScalarMatMul(void) {
    const char* value = getenv("ORT_FUSIONCPU_MATMULNBITS_SCALAR");
    return value != NULL && value[0] == '1';
}

static OrtStatus* FusionCpuGetInput(const OrtApi* api, const OrtKernelContext* context,
                                    size_t index, const OrtValue** value,
                                    ONNXTensorElementDataType* type,
                                    const int64_t** shape, size_t* rank,
                                    const void** data) {
    OrtStatus* status = api->KernelContext_GetInput(context, index, value);
    if (status != NULL) return status;
    if (*value == NULL) {
        return api->CreateStatus(ORT_INVALID_ARGUMENT, "FusionCPU EP: required input is absent.");
    }
    status = api->GetTensorElementTypeAndShapeDataReference(*value, type, shape, rank);
    if (status != NULL) return status;
    return api->GetTensorData(*value, data);
}

static OrtStatus* FusionCpuGetOutput(const OrtApi* api, OrtKernelContext* context,
                                     size_t index, const int64_t* shape, size_t rank,
                                     OrtValue** value, void** data) {
    OrtStatus* status = api->KernelContext_GetOutput(context, index, shape, rank, value);
    if (status != NULL) return status;
    if (*value == NULL) {
        return api->CreateStatus(ORT_INVALID_ARGUMENT, "FusionCPU EP: required output is absent.");
    }
    return api->GetTensorMutableData(*value, data);
}

static bool FusionCpuShapeEqual(const int64_t* a, size_t a_rank,
                                const int64_t* b, size_t b_rank) {
    size_t i;
    if (a_rank != b_rank) return false;
    for (i = 0; i < a_rank; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

static bool FusionCpuElementCount(const int64_t* shape, size_t rank, size_t* count) {
    size_t i;
    size_t result = 1;
    for (i = 0; i < rank; ++i) {
        if (shape[i] < 0 || (shape[i] != 0 && result > SIZE_MAX / (size_t)shape[i])) return false;
        result *= (size_t)shape[i];
    }
    *count = result;
    return true;
}

static OrtStatus* ORT_API_CALL FusionCpuMulCompute(void* state,
                                                    OrtKernelContext* context) {
    FusionCpuMulState* self = (FusionCpuMulState*)state;
    const OrtValue* left;
    const OrtValue* right;
    OrtValue* output;
    ONNXTensorElementDataType left_type;
    ONNXTensorElementDataType right_type;
    const int64_t* left_shape;
    const int64_t* right_shape;
    size_t left_rank;
    size_t right_rank;
    const void* left_data;
    const void* right_data;
    void* output_data;
    size_t i;
    size_t count;
    size_t output_rank;
    int64_t* output_shape;
    int64_t* left_strides;
    int64_t* right_strides;
    int64_t* coordinates;
    OrtStatus* status;

    status = FusionCpuGetInput(self->api, context, 0, &left, &left_type, &left_shape,
                               &left_rank, &left_data);
    if (status != NULL) return status;
    status = FusionCpuGetInput(self->api, context, 1, &right, &right_type, &right_shape,
                               &right_rank, &right_data);
    if (status != NULL) return status;
    if (left_type != right_type ||
        (left_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
         left_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT,
            "FusionCPU Mul requires matching float32 or float16 tensors.");
    }
    output_rank = left_rank > right_rank ? left_rank : right_rank;
    output_shape = (int64_t*)calloc(output_rank, sizeof(*output_shape));
    left_strides = (int64_t*)calloc(output_rank, sizeof(*left_strides));
    right_strides = (int64_t*)calloc(output_rank, sizeof(*right_strides));
    coordinates = (int64_t*)calloc(output_rank, sizeof(*coordinates));
    if ((output_shape == NULL || left_strides == NULL || right_strides == NULL ||
         coordinates == NULL) && output_rank != 0) {
        free(output_shape); free(left_strides); free(right_strides); free(coordinates);
        return self->api->CreateStatus(ORT_EP_FAIL, "FusionCPU Mul allocation failed.");
    }
    for (i = 0; i < output_rank; ++i) {
        int64_t left_dim = i + left_rank >= output_rank ? left_shape[i - (output_rank - left_rank)] : 1;
        int64_t right_dim = i + right_rank >= output_rank ? right_shape[i - (output_rank - right_rank)] : 1;
        if (left_dim < 0 || right_dim < 0 ||
            (left_dim != right_dim && left_dim != 1 && right_dim != 1)) {
            free(output_shape); free(left_strides); free(right_strides); free(coordinates);
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "FusionCPU Mul broadcast shapes are incompatible.");
        }
        output_shape[i] = left_dim > right_dim ? left_dim : right_dim;
    }
    {
        int64_t stride = 1;
        int64_t dimension;
        for (dimension = (int64_t)output_rank - 1; dimension >= 0; --dimension) {
            const int64_t left_dim = (size_t)dimension + left_rank >= output_rank
                ? left_shape[dimension - ((int64_t)output_rank - (int64_t)left_rank)] : 1;
            const int64_t right_dim = (size_t)dimension + right_rank >= output_rank
                ? right_shape[dimension - ((int64_t)output_rank - (int64_t)right_rank)] : 1;
            left_strides[dimension] = left_dim == 1 ? 0 : stride;
            right_strides[dimension] = right_dim == 1 ? 0 : stride;
            stride *= output_shape[dimension];
        }
        count = (size_t)stride;
    }
    status = FusionCpuGetOutput(self->api, context, 0, output_shape, output_rank, &output, &output_data);
    if (status != NULL) {
        free(output_shape); free(left_strides); free(right_strides); free(coordinates);
        return status;
    }
    for (i = 0; i < count; ++i) {
        int64_t left_offset = 0;
        int64_t right_offset = 0;
        int64_t dimension;
        for (dimension = 0; dimension < (int64_t)output_rank; ++dimension) {
            left_offset += coordinates[dimension] * left_strides[dimension];
            right_offset += coordinates[dimension] * right_strides[dimension];
        }
        if (left_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            ((float*)output_data)[i] = ((const float*)left_data)[left_offset] *
                                       ((const float*)right_data)[right_offset];
        } else {
            const float product = FusionCpuHalfToFloat(((const uint16_t*)left_data)[left_offset]) *
                                  FusionCpuHalfToFloat(((const uint16_t*)right_data)[right_offset]);
            ((uint16_t*)output_data)[i] = FusionCpuFloatToHalf(product);
        }
        for (dimension = (int64_t)output_rank - 1; dimension >= 0; --dimension) {
            if (++coordinates[dimension] < output_shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    free(output_shape);
    free(left_strides);
    free(right_strides);
    free(coordinates);
    return NULL;
}

static OrtStatus* ORT_API_CALL FusionCpuSigmoidCompute(void* state,
                                                        OrtKernelContext* context) {
    FusionCpuSigmoidState* self = (FusionCpuSigmoidState*)state;
    const OrtValue* input;
    OrtValue* output;
    ONNXTensorElementDataType type;
    const int64_t* shape;
    size_t rank;
    const void* input_data;
    void* output_data;
    size_t i;
    size_t count;
    OrtStatus* status = FusionCpuGetInput(self->api, context, 0, &input, &type, &shape,
                                           &rank, &input_data);
    if (status != NULL) return status;
    if ((type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
         type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ||
        !FusionCpuElementCount(shape, rank, &count)) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "FusionCPU Sigmoid requires float or float16 input.");
    }
    status = FusionCpuGetOutput(self->api, context, 0, shape, rank, &output, &output_data);
    if (status != NULL) return status;
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        for (i = 0; i < count; ++i) {
            ((float*)output_data)[i] = 1.0f / (1.0f + expf(-((const float*)input_data)[i]));
        }
    } else {
        for (i = 0; i < count; ++i) {
            const float input_value = FusionCpuHalfToFloat(((const uint16_t*)input_data)[i]);
            ((uint16_t*)output_data)[i] =
                FusionCpuFloatToHalf(1.0f / (1.0f + expf(-input_value)));
        }
    }
    return NULL;
}

typedef struct FusionCpuMatMulParallelContext {
    FusionCpuMatMulNBitsState* state;
    ONNXTensorElementDataType a_type;
    const void* a_data;
    const void* b_data;
    const void* scale_data;
    const void* zp_data;
    void* y_data;
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t k_blocks;
    int64_t blob;
    int64_t zp_cols;
    bool has_zp;
    float* dequant_scratch;
    float* scale_scratch;
} FusionCpuMatMulParallelContext;

static void FusionCpuMatMulParallelBody(void* opaque, int task, int nways) {
    FusionCpuMatMulParallelContext* context =
        (FusionCpuMatMulParallelContext*)opaque;
    FusionCpuMatMulNBitsState* self = context->state;
    const int64_t begin = (int64_t)task * context->n / nways;
    const int64_t end = (int64_t)(task + 1) * context->n / nways;
    float* dequant = context->dequant_scratch + (size_t)task * (size_t)context->k;
    float* scales = context->scale_scratch +
        (size_t)task * (size_t)context->k_blocks;
    int64_t col;
    for (col = begin; col < end; ++col) {
        const uint8_t* b_row = (const uint8_t*)context->b_data +
            (size_t)col * (size_t)context->k_blocks * (size_t)context->blob;
        const uint8_t* z_row = context->has_zp
            ? (const uint8_t*)context->zp_data +
                (size_t)col * (size_t)context->zp_cols
            : NULL;
        int64_t block;
        int64_t row;
        for (block = 0; block < context->k_blocks; ++block) {
            scales[block] = context->a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                ? ((const float*)context->scale_data)[
                    (size_t)col * (size_t)context->k_blocks + (size_t)block]
                : FusionCpuHalfToFloat(((const uint16_t*)context->scale_data)[
                    (size_t)col * (size_t)context->k_blocks + (size_t)block]);
        }
        if (FusionCpuHasAvx512f() && self->block_size % 16 == 0 &&
            !FusionCpuForceScalarMatMul()) {
            DequantColumnAvx512(dequant, b_row, scales, z_row, context->has_zp,
                                context->k_blocks, self->block_size, 8);
        } else {
            for (block = 0; block < context->k_blocks; ++block) {
                const int zero = context->has_zp
                    ? ((block & 1) ? (z_row[block >> 1] >> 4)
                                   : (z_row[block >> 1] & 0x0F))
                    : 8;
                int64_t element;
                for (element = 0; element < self->block_size; ++element) {
                    const uint8_t packed = b_row[
                        (size_t)block * (size_t)context->blob +
                        (size_t)(element >> 1)];
                    const int quantized =
                        (element & 1) ? (packed >> 4) : (packed & 0x0F);
                    dequant[block * self->block_size + element] =
                        ((float)quantized - (float)zero) * scales[block];
                }
            }
        }
        for (row = 0; row < context->m; ++row) {
            int64_t element;
            float result;
            if (context->a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                const float* a_row =
                    (const float*)context->a_data + row * context->k;
                if (FusionCpuHasAvx512f() && context->k % 16 == 0 &&
                    !FusionCpuForceScalarMatMul()) {
                    result = DotFp32Avx512(a_row, dequant, context->k);
                } else {
                    result = 0.0f;
                    for (element = 0; element < context->k; ++element) {
                        result += a_row[element] * dequant[element];
                    }
                }
                ((float*)context->y_data)[row * context->n + col] = result;
            } else {
                const uint16_t* a_row =
                    (const uint16_t*)context->a_data + row * context->k;
                result = 0.0f;
                for (element = 0; element < context->k; ++element) {
                    result += FusionCpuHalfToFloat(a_row[element]) * dequant[element];
                }
                ((uint16_t*)context->y_data)[row * context->n + col] =
                    FusionCpuFloatToHalf(result);
            }
        }
    }
}

static OrtStatus* ORT_API_CALL FusionCpuMatMulNBitsCompute(void* state,
                                                            OrtKernelContext* context) {
    FusionCpuMatMulNBitsState* self = (FusionCpuMatMulNBitsState*)state;
    const OrtValue* a_value;
    const OrtValue* b_value;
    const OrtValue* scale_value;
    const OrtValue* zp_value = NULL;
    ONNXTensorElementDataType a_type;
    ONNXTensorElementDataType b_type;
    ONNXTensorElementDataType scale_type;
    ONNXTensorElementDataType ignored_type;
    const int64_t* a_shape;
    const int64_t* b_shape;
    const int64_t* scale_shape;
    const int64_t* ignored_shape = NULL;
    size_t a_rank;
    size_t b_rank;
    size_t scale_rank;
    size_t ignored_rank = 0;
    const void* a_data;
    const void* b_data;
    const void* scale_data;
    const void* zp_data = NULL;
    int64_t* y_shape;
    OrtValue* y_value;
    void* y_data;
    int64_t m = 1;
    int64_t k;
    int64_t n;
    int64_t k_blocks;
    int64_t blob;
    int64_t zp_cols;
    size_t scale_count;
    bool has_zp;
    float* dequant = NULL;
    float* scales = NULL;
    FusionThreadPool* pool;
    FusionCpuMatMulParallelContext parallel_context;
    int ways;
    int64_t row;
    OrtStatus* status;

    status = FusionCpuGetInput(self->api, context, (size_t)self->idx_a, &a_value, &a_type,
                               &a_shape, &a_rank, &a_data);
    if (status != NULL) return status;
    status = FusionCpuGetInput(self->api, context, (size_t)self->idx_b, &b_value, &b_type,
                               &b_shape, &b_rank, &b_data);
    if (status != NULL) return status;
    status = FusionCpuGetInput(self->api, context, (size_t)self->idx_scales, &scale_value,
                               &scale_type, &scale_shape, &scale_rank, &scale_data);
    if (status != NULL) return status;
    has_zp = self->idx_zp >= 0;
    if (has_zp) {
        status = FusionCpuGetInput(self->api, context, (size_t)self->idx_zp, &zp_value,
                                   &ignored_type, &ignored_shape, &ignored_rank, &zp_data);
        if (status != NULL) return status;
        if (ignored_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT,
                                            "MatMulNBits zero_points must be uint8.");
        }
    }
    if ((a_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
         a_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ||
        scale_type != a_type || b_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
        a_rank == 0 || b_rank != 3 || self->bits != 4) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT,
                                        "FusionCPU MatMulNBits received unsupported tensor types.");
    }
    k = a_shape[a_rank - 1];
    n = b_shape[0];
    k_blocks = b_shape[1];
    blob = b_shape[2];
    if (k <= 0 || n < 0 || k_blocks <= 0 || self->block_size <= 0 ||
        k_blocks * self->block_size != k || blob != self->block_size / 2) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT,
                                        "FusionCPU MatMulNBits received inconsistent shapes.");
    }
    for (row = 0; row + 1 < (int64_t)a_rank; ++row) {
        if (a_shape[row] < 0 || (a_shape[row] != 0 && m > INT64_MAX / a_shape[row])) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "MatMulNBits invalid A shape.");
        }
        m *= a_shape[row];
    }
    if (!FusionCpuElementCount(scale_shape, scale_rank, &scale_count) ||
        scale_count != (size_t)n * (size_t)k_blocks) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "MatMulNBits scales shape mismatch.");
    }
    zp_cols = (k_blocks + 1) / 2;
    if (has_zp && (!FusionCpuElementCount(ignored_shape, ignored_rank, &scale_count) ||
                   scale_count != (size_t)n * (size_t)zp_cols)) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT,
                                        "MatMulNBits zero_points shape mismatch.");
    }
    y_shape = (int64_t*)malloc(a_rank * sizeof(*y_shape));
    pool = FusionCpuThreadPool();
    ways = pool != NULL ? fusion_pool_size(pool) : 1;
    if ((int64_t)ways > n) ways = (int)n;
    if (ways < 1) ways = 1;
    dequant = (float*)malloc((size_t)ways * (size_t)k * sizeof(*dequant));
    scales = (float*)malloc((size_t)ways * (size_t)k_blocks * sizeof(*scales));
    if (y_shape == NULL || dequant == NULL || scales == NULL) {
        free(y_shape);
        free(dequant);
        free(scales);
        return self->api->CreateStatus(ORT_EP_FAIL, "FusionCPU MatMulNBits out of memory.");
    }
    memcpy(y_shape, a_shape, a_rank * sizeof(*y_shape));
    y_shape[a_rank - 1] = n;
    status = FusionCpuGetOutput(self->api, context, 0, y_shape, a_rank, &y_value, &y_data);
    free(y_shape);
    if (status != NULL) {
        free(dequant);
        free(scales);
        return status;
    }
    memset(&parallel_context, 0, sizeof(parallel_context));
    parallel_context.state = self;
    parallel_context.a_type = a_type;
    parallel_context.a_data = a_data;
    parallel_context.b_data = b_data;
    parallel_context.scale_data = scale_data;
    parallel_context.zp_data = zp_data;
    parallel_context.y_data = y_data;
    parallel_context.m = m;
    parallel_context.k = k;
    parallel_context.n = n;
    parallel_context.k_blocks = k_blocks;
    parallel_context.blob = blob;
    parallel_context.zp_cols = zp_cols;
    parallel_context.has_zp = has_zp;
    parallel_context.dequant_scratch = dequant;
    parallel_context.scale_scratch = scales;
    fusion_pool_parallel_for(pool, ways, FusionCpuMatMulParallelBody, &parallel_context);
    free(dequant);
    free(scales);
    return NULL;
}

static void FusionCpuRotateRow(const float* input, const float* cosine, const float* sine,
                               int rotary_dim, int head_size, bool interleaved, float* output) {
    int index;
    int half = rotary_dim / 2;
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

typedef struct FusionCpuGqaParallelContext {
    FusionCpuGqaState* state;
    const float* q_data;
    const float* k_data;
    const float* v_data;
    const float* past_k_data;
    const float* past_v_data;
    const float* cos_data;
    const float* sin_data;
    float* output_data;
    float* present_k_data;
    float* present_v_data;
    float* q_rot_scratch;
    float* scores_scratch;
    int64_t sequence;
    int64_t head_size;
    int64_t q_hidden;
    int64_t q_stride;
    int64_t k_stride;
    int64_t v_stride;
    int64_t past_length;
    int64_t total;
    int group;
} FusionCpuGqaParallelContext;

static void FusionCpuGqaParallelBody(void* opaque, int task, int nways) {
    FusionCpuGqaParallelContext* context = (FusionCpuGqaParallelContext*)opaque;
    FusionCpuGqaState* self = context->state;
    const int begin = task * self->kv_num_heads / nways;
    const int end = (task + 1) * self->kv_num_heads / nways;
    float* q_rot = context->q_rot_scratch +
        (size_t)task * (size_t)context->head_size;
    float* scores = context->scores_scratch +
        (size_t)task * (size_t)context->total;
    int kv_head;
    for (kv_head = begin; kv_head < end; ++kv_head) {
        const int64_t past_offset =
            (int64_t)kv_head * context->past_length * context->head_size;
        const int64_t present_offset =
            (int64_t)kv_head * context->total * context->head_size;
        const float* keys = context->present_k_data + present_offset;
        const float* values = context->present_v_data + present_offset;
        int64_t token;
        int head;
        if (context->past_length > 0) {
            memcpy(context->present_k_data + present_offset,
                   context->past_k_data + past_offset,
                   (size_t)context->past_length * (size_t)context->head_size *
                       sizeof(float));
            memcpy(context->present_v_data + present_offset,
                   context->past_v_data + past_offset,
                   (size_t)context->past_length * (size_t)context->head_size *
                       sizeof(float));
        }
        for (token = 0; token < context->sequence; ++token) {
            const float* source_k = context->k_data +
                token * context->k_stride + (int64_t)kv_head * context->head_size;
            const float* source_v = context->v_data +
                token * context->v_stride + (int64_t)kv_head * context->head_size;
            float* destination_k = context->present_k_data + present_offset +
                (context->past_length + token) * context->head_size;
            float* destination_v = context->present_v_data + present_offset +
                (context->past_length + token) * context->head_size;
            if (self->do_rotary) {
                const int rotary_dim = (int)context->head_size;
                FusionCpuRotateRow(
                    source_k,
                    context->cos_data +
                        (context->past_length + token) * (rotary_dim / 2),
                    context->sin_data +
                        (context->past_length + token) * (rotary_dim / 2),
                    rotary_dim, (int)context->head_size,
                    self->rotary_interleaved, destination_k);
            } else {
                memcpy(destination_k, source_k,
                       (size_t)context->head_size * sizeof(float));
            }
            memcpy(destination_v, source_v,
                   (size_t)context->head_size * sizeof(float));
        }
        for (head = kv_head * context->group;
             head < (kv_head + 1) * context->group; ++head) {
            for (token = 0; token < context->sequence; ++token) {
                const int64_t limit = context->past_length + token + 1;
                const int64_t start =
                    self->local_window > 0 && limit > self->local_window
                    ? limit - self->local_window : 0;
                const float* query = context->q_data +
                    token * context->q_stride +
                    (int64_t)head * context->head_size;
                float* result = context->output_data +
                    token * context->q_hidden +
                    (int64_t)head * context->head_size;
                float maximum = -FLT_MAX;
                float sum = 0.0f;
                const float scale = self->scale != 0.0f
                    ? self->scale : 1.0f / sqrtf((float)context->head_size);
                int64_t key_index;
                if (self->do_rotary) {
                    const int rotary_dim = (int)context->head_size;
                    FusionCpuRotateRow(
                        query,
                        context->cos_data +
                            (context->past_length + token) * (rotary_dim / 2),
                        context->sin_data +
                            (context->past_length + token) * (rotary_dim / 2),
                        rotary_dim, (int)context->head_size,
                        self->rotary_interleaved, q_rot);
                    query = q_rot;
                }
                for (key_index = start; key_index < limit; ++key_index) {
                    const float* key = keys + key_index * context->head_size;
                    float dot = 0.0f;
                    int64_t dim;
                    for (dim = 0; dim < context->head_size; ++dim) {
                        dot += query[dim] * key[dim];
                    }
                    scores[key_index] = dot * scale;
                    if (scores[key_index] > maximum) maximum = scores[key_index];
                }
                for (key_index = start; key_index < limit; ++key_index) {
                    scores[key_index] = expf(scores[key_index] - maximum);
                    sum += scores[key_index];
                }
                memset(result, 0, (size_t)context->head_size * sizeof(float));
                for (key_index = start; key_index < limit; ++key_index) {
                    const float weight =
                        sum > 0.0f ? scores[key_index] / sum : 0.0f;
                    const float* value =
                        values + key_index * context->head_size;
                    int64_t dim;
                    for (dim = 0; dim < context->head_size; ++dim) {
                        result[dim] += weight * value[dim];
                    }
                }
            }
        }
    }
}

static OrtStatus* ORT_API_CALL FusionCpuGqaCompute(void* state, OrtKernelContext* context) {
    FusionCpuGqaState* self = (FusionCpuGqaState*)state;
    const OrtValue* qkv_value;
    const OrtValue* past_k_value;
    const OrtValue* past_v_value;
    const OrtValue* key_value = NULL;
    const OrtValue* value_value = NULL;
    const OrtValue* cos_value = NULL;
    const OrtValue* sin_value = NULL;
    ONNXTensorElementDataType type;
    const int64_t* qkv_shape;
    const int64_t* past_k_shape;
    const int64_t* past_v_shape;
    const int64_t* ignored_shape;
    size_t qkv_rank;
    size_t past_k_rank;
    size_t past_v_rank;
    size_t ignored_rank;
    const void* qkv_void;
    const void* past_k_void;
    const void* past_v_void;
    const void* ignored_void;
    const float* q_data;
    const float* k_data;
    const float* v_data;
    const float* past_k_data;
    const float* past_v_data;
    const float* cos_data = NULL;
    const float* sin_data = NULL;
    bool unpacked;
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
    int group;
    int64_t output_shape[3];
    int64_t present_shape[4];
    OrtValue* ignored_output;
    void* output_void;
    void* present_k_void;
    void* present_v_void;
    float* output_data;
    float* present_k_data;
    float* present_v_data;
    float* q_rot = NULL;
    float* scores = NULL;
    FusionThreadPool* pool;
    FusionCpuGqaParallelContext parallel_context;
    int ways;
    OrtStatus* status;

    status = FusionCpuGetInput(self->api, context, (size_t)self->idx_qkv, &qkv_value, &type,
                               &qkv_shape, &qkv_rank, &qkv_void);
    if (status != NULL) return status;
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || qkv_rank != 3) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA query must be rank-3 float.");
    }
    status = FusionCpuGetInput(self->api, context, (size_t)self->idx_past_key, &past_k_value,
                               &type, &past_k_shape, &past_k_rank, &past_k_void);
    if (status != NULL) return status;
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || past_k_rank != 4) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA past_key must be rank-4 float.");
    }
    status = FusionCpuGetInput(self->api, context, (size_t)self->idx_past_value, &past_v_value,
                               &type, &past_v_shape, &past_v_rank, &past_v_void);
    if (status != NULL) return status;
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || past_v_rank != 4) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA past_value must be rank-4 float.");
    }

    unpacked = self->idx_key >= 0 && self->idx_value >= 0;
    batch = qkv_shape[0];
    sequence = qkv_shape[1];
    if (batch != 1 || sequence < 0 || self->kv_num_heads <= 0 ||
        self->num_heads <= 0 || self->num_heads % self->kv_num_heads != 0) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported GroupQueryAttention shape.");
    }
    group = self->num_heads / self->kv_num_heads;
    q_data = (const float*)qkv_void;
    if (unpacked) {
        const int64_t* key_shape;
        const int64_t* value_shape;
        size_t key_rank;
        size_t value_rank;
        if (qkv_shape[2] % self->num_heads != 0) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA query hidden size is invalid.");
        }
        head_size = qkv_shape[2] / self->num_heads;
        q_hidden = qkv_shape[2];
        kv_hidden = (int64_t)self->kv_num_heads * head_size;
        status = FusionCpuGetInput(self->api, context, (size_t)self->idx_key, &key_value, &type,
                                   &key_shape, &key_rank, &ignored_void);
        if (status != NULL) return status;
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || key_rank != 3 ||
            key_shape[0] != batch || key_shape[1] != sequence || key_shape[2] != kv_hidden) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA key shape is invalid.");
        }
        k_data = (const float*)ignored_void;
        status = FusionCpuGetInput(self->api, context, (size_t)self->idx_value, &value_value, &type,
                                   &value_shape, &value_rank, &ignored_void);
        if (status != NULL) return status;
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || value_rank != 3 ||
            value_shape[0] != batch || value_shape[1] != sequence || value_shape[2] != kv_hidden) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA value shape is invalid.");
        }
        v_data = (const float*)ignored_void;
        q_stride = q_hidden;
        k_stride = kv_hidden;
        v_stride = kv_hidden;
    } else {
        int64_t packed_hidden = qkv_shape[2];
        if (packed_hidden % (self->num_heads + 2 * self->kv_num_heads) != 0) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA packed hidden size is invalid.");
        }
        head_size = packed_hidden / (self->num_heads + 2 * self->kv_num_heads);
        q_hidden = (int64_t)self->num_heads * head_size;
        kv_hidden = (int64_t)self->kv_num_heads * head_size;
        k_data = q_data + q_hidden;
        v_data = q_data + q_hidden + kv_hidden;
        q_stride = packed_hidden;
        k_stride = packed_hidden;
        v_stride = packed_hidden;
    }
    past_length = past_k_shape[2];
    total = past_length + sequence;
    if (head_size <= 0 || past_k_shape[0] != 1 || past_k_shape[1] != self->kv_num_heads ||
        past_k_shape[3] != head_size || !FusionCpuShapeEqual(past_k_shape, past_k_rank,
        past_v_shape, past_v_rank)) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA past cache shape is invalid.");
    }
    if (self->do_rotary) {
        status = FusionCpuGetInput(self->api, context, (size_t)self->idx_cos, &cos_value, &type,
                                   &ignored_shape, &ignored_rank, &ignored_void);
        if (status != NULL) return status;
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA cosine cache must be float.");
        }
        cos_data = (const float*)ignored_void;
        status = FusionCpuGetInput(self->api, context, (size_t)self->idx_sin, &sin_value, &type,
                                   &ignored_shape, &ignored_rank, &ignored_void);
        if (status != NULL) return status;
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "GQA sine cache must be float.");
        }
        sin_data = (const float*)ignored_void;
    }
    output_shape[0] = 1;
    output_shape[1] = sequence;
    output_shape[2] = q_hidden;
    present_shape[0] = 1;
    present_shape[1] = self->kv_num_heads;
    present_shape[2] = total;
    present_shape[3] = head_size;
    status = FusionCpuGetOutput(self->api, context, (size_t)self->idx_out, output_shape, 3,
                                &ignored_output, &output_void);
    if (status != NULL) return status;
    output_data = (float*)output_void;
    status = FusionCpuGetOutput(self->api, context, (size_t)self->idx_present_key, present_shape, 4,
                                &ignored_output, &present_k_void);
    if (status != NULL) return status;
    status = FusionCpuGetOutput(self->api, context, (size_t)self->idx_present_value, present_shape, 4,
                                &ignored_output, &present_v_void);
    if (status != NULL) return status;
    present_k_data = (float*)present_k_void;
    present_v_data = (float*)present_v_void;
    past_k_data = (const float*)past_k_void;
    past_v_data = (const float*)past_v_void;

    pool = FusionCpuThreadPool();
    ways = pool != NULL ? fusion_pool_size(pool) : 1;
    if (ways > self->kv_num_heads) ways = self->kv_num_heads;
    if (ways < 1) ways = 1;
    q_rot = (float*)malloc((size_t)ways * (size_t)head_size * sizeof(*q_rot));
    scores = (float*)malloc((size_t)ways * (size_t)total * sizeof(*scores));
    if (q_rot == NULL || scores == NULL) {
        free(q_rot);
        free(scores);
        return self->api->CreateStatus(ORT_EP_FAIL, "GQA scratch allocation failed.");
    }
    memset(&parallel_context, 0, sizeof(parallel_context));
    parallel_context.state = self;
    parallel_context.q_data = q_data;
    parallel_context.k_data = k_data;
    parallel_context.v_data = v_data;
    parallel_context.past_k_data = past_k_data;
    parallel_context.past_v_data = past_v_data;
    parallel_context.cos_data = cos_data;
    parallel_context.sin_data = sin_data;
    parallel_context.output_data = output_data;
    parallel_context.present_k_data = present_k_data;
    parallel_context.present_v_data = present_v_data;
    parallel_context.q_rot_scratch = q_rot;
    parallel_context.scores_scratch = scores;
    parallel_context.sequence = sequence;
    parallel_context.head_size = head_size;
    parallel_context.q_hidden = q_hidden;
    parallel_context.q_stride = q_stride;
    parallel_context.k_stride = k_stride;
    parallel_context.v_stride = v_stride;
    parallel_context.past_length = past_length;
    parallel_context.total = total;
    parallel_context.group = group;
    fusion_pool_parallel_for(pool, ways, FusionCpuGqaParallelBody, &parallel_context);
    free(q_rot);
    free(scores);
    return NULL;
}

static void ORT_API_CALL FusionCpuFreeSimpleState(void* state) {
    free(state);
}

static void ORT_API_CALL FusionCpuFreeGqaState(void* state) {
    free(state);
}

static bool FusionCpuGetNodeIo(const OrtApi* api, const OrtNode* node, bool inputs,
                               const OrtValueInfo*** values, size_t* count) {
    OrtStatus* status;
    const OrtValueInfo** list;
    *values = NULL;
    *count = 0;
    status = inputs ? api->Node_GetNumInputs(node, count) : api->Node_GetNumOutputs(node, count);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    if (*count == 0) return true;
    list = (const OrtValueInfo**)calloc(*count, sizeof(*list));
    if (list == NULL) return false;
    status = inputs ? api->Node_GetInputs(node, list, *count) : api->Node_GetOutputs(node, list, *count);
    if (status != NULL) {
        api->ReleaseStatus(status);
        free((void*)list);
        return false;
    }
    *values = list;
    return true;
}

static bool FusionCpuValueNameEquals(const OrtApi* api, const OrtValueInfo* value,
                                     const char* name) {
    const char* value_name;
    OrtStatus* status;
    if (value == NULL || name == NULL) return false;
    status = api->GetValueInfoName(value, &value_name);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    return value_name != NULL && strcmp(value_name, name) == 0;
}

static int FusionCpuFindInputForValue(const OrtApi* api, const OrtNode* fused,
                                      const OrtValueInfo* value) {
    const OrtValueInfo** inputs;
    size_t count;
    size_t i;
    const char* name = NULL;
    OrtStatus* status;
    if (value == NULL) return -1;
    status = api->GetValueInfoName(value, &name);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return -1;
    }
    if (!FusionCpuGetNodeIo(api, fused, true, &inputs, &count)) return -1;
    for (i = 0; i < count; ++i) {
        if (FusionCpuValueNameEquals(api, inputs[i], name)) {
            free((void*)inputs);
            return (int)i;
        }
    }
    free((void*)inputs);
    return -1;
}

static int FusionCpuFindOutputForValue(const OrtApi* api, const OrtNode* fused,
                                       const OrtValueInfo* value) {
    const OrtValueInfo** outputs;
    size_t count;
    size_t i;
    const char* name = NULL;
    OrtStatus* status;
    if (value == NULL) return -1;
    status = api->GetValueInfoName(value, &name);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return -1;
    }
    if (!FusionCpuGetNodeIo(api, fused, false, &outputs, &count)) return -1;
    for (i = 0; i < count; ++i) {
        if (FusionCpuValueNameEquals(api, outputs[i], name)) {
            free((void*)outputs);
            return (int)i;
        }
    }
    free((void*)outputs);
    return -1;
}

static bool FusionCpuReadIntAttr(const OrtApi* api, const OrtNode* node,
                                 const char* name, int64_t* value) {
    const OrtOpAttr* attribute = NULL;
    size_t byte_count = 0;
    OrtStatus* status = api->Node_GetAttributeByName(node, name, &attribute);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    if (attribute == NULL) return false;
    status = api->ReadOpAttr(attribute, ORT_OP_ATTR_INT, value, sizeof(*value), &byte_count);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    return byte_count == sizeof(*value);
}

static bool FusionCpuReadFloatAttr(const OrtApi* api, const OrtNode* node,
                                   const char* name, float* value) {
    const OrtOpAttr* attribute = NULL;
    size_t byte_count = 0;
    OrtStatus* status = api->Node_GetAttributeByName(node, name, &attribute);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    if (attribute == NULL) return false;
    status = api->ReadOpAttr(attribute, ORT_OP_ATTR_FLOAT, value, sizeof(*value), &byte_count);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    return byte_count == sizeof(*value);
}

static bool FusionCpuIsSupportedNode(const OrtApi* api, const OrtNode* node) {
    const char* op = NULL;
    const char* domain = NULL;
    size_t input_count;
    size_t output_count;
    OrtStatus* status = api->Node_GetOperatorType(node, &op);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    status = api->Node_GetDomain(node, &domain);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    if (FusionCpuExtraSupportsNode(api, node)) return true;
    status = api->Node_GetNumInputs(node, &input_count);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    status = api->Node_GetNumOutputs(node, &output_count);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    if ((domain == NULL || domain[0] == '\0') &&
        (strcmp(op, "Mul") == 0 || strcmp(op, "Sigmoid") == 0)) {
        return (strcmp(op, "Mul") == 0 && input_count == 2 && output_count == 1) ||
               (strcmp(op, "Sigmoid") == 0 && input_count == 1 && output_count == 1);
    }
    if (domain != NULL && strcmp(domain, "com.microsoft") == 0 &&
        strcmp(op, "MatMulNBits") == 0) {
        return (input_count == 3 || input_count == 4) && output_count == 1;
    }
    if (domain != NULL && strcmp(domain, "com.microsoft") == 0 &&
        strcmp(op, "GroupQueryAttention") == 0) {
        return output_count == 3;
    }
    return false;
}

static OrtStatus* ORT_API_CALL FusionCpuEpGetCapability(OrtEp* ep, const OrtGraph* graph,
                                                         OrtEpGraphSupportInfo* support_info) {
    FusionCpuEp* self = (FusionCpuEp*)ep;
    const OrtNode** nodes = NULL;
    size_t count = 0;
    size_t i;
    OrtStatus* status = self->ort_api->Graph_GetNumNodes(graph, &count);
    if (status != NULL) return status;
    if (count == 0) return NULL;
    nodes = (const OrtNode**)calloc(count, sizeof(*nodes));
    if (nodes == NULL) {
        return self->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU EP: out of memory reading graph.");
    }
    status = self->ort_api->Graph_GetNodes(graph, nodes, count);
    if (status != NULL) {
        free((void*)nodes);
        return status;
    }
    for (i = 0; i < count; ++i) {
        OrtNodeFusionOptions options;
        if (!FusionCpuIsSupportedNode(self->ort_api, nodes[i])) continue;
        memset(&options, 0, sizeof(options));
        options.ort_version_supported = ORT_API_VERSION;
        options.drop_constant_initializers = false;
        status = self->ep_api->EpGraphSupportInfo_AddNodesToFuse(support_info, &nodes[i], 1,
                                                                 &options);
        if (status != NULL) {
            free((void*)nodes);
            return status;
        }
    }
    free((void*)nodes);
    return NULL;
}

static OrtStatus* FusionCpuCreateGqaState(FusionCpuEp* ep, const OrtNode* fused,
                                          const OrtNode* inner, void** state) {
    FusionCpuGqaState* result;
    const OrtValueInfo** inputs = NULL;
    const OrtValueInfo** outputs = NULL;
    size_t input_count;
    size_t output_count;
    int64_t num_heads;
    int64_t kv_num_heads;
    int64_t do_rotary = 0;
    int64_t rotary_interleaved = 0;
    int64_t local_window = -1;
    float scale = 0.0f;
    if (!FusionCpuGetNodeIo(ep->ort_api, inner, true, &inputs, &input_count) ||
        !FusionCpuGetNodeIo(ep->ort_api, inner, false, &outputs, &output_count) ||
        input_count < 5 || output_count < 3 ||
        !FusionCpuReadIntAttr(ep->ort_api, inner, "num_heads", &num_heads) ||
        !FusionCpuReadIntAttr(ep->ort_api, inner, "kv_num_heads", &kv_num_heads)) {
        free((void*)inputs);
        free((void*)outputs);
        return ep->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU GQA missing required metadata.");
    }
    FusionCpuReadFloatAttr(ep->ort_api, inner, "scale", &scale);
    FusionCpuReadIntAttr(ep->ort_api, inner, "do_rotary", &do_rotary);
    FusionCpuReadIntAttr(ep->ort_api, inner, "rotary_interleaved", &rotary_interleaved);
    FusionCpuReadIntAttr(ep->ort_api, inner, "local_window_size", &local_window);
    result = (FusionCpuGqaState*)calloc(1, sizeof(*result));
    if (result == NULL) {
        free((void*)inputs);
        free((void*)outputs);
        return ep->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU GQA allocation failed.");
    }
    result->api = ep->ort_api;
    result->num_heads = (int)num_heads;
    result->kv_num_heads = (int)kv_num_heads;
    result->scale = scale;
    result->do_rotary = do_rotary != 0;
    result->rotary_interleaved = rotary_interleaved != 0;
    result->local_window = local_window;
    result->idx_qkv = FusionCpuFindInputForValue(ep->ort_api, fused, inputs[0]);
    result->idx_key = input_count > 1 ? FusionCpuFindInputForValue(ep->ort_api, fused, inputs[1]) : -1;
    result->idx_value = input_count > 2 ? FusionCpuFindInputForValue(ep->ort_api, fused, inputs[2]) : -1;
    result->idx_past_key = FusionCpuFindInputForValue(ep->ort_api, fused, inputs[3]);
    result->idx_past_value = FusionCpuFindInputForValue(ep->ort_api, fused, inputs[4]);
    result->idx_cos = input_count > 7 ? FusionCpuFindInputForValue(ep->ort_api, fused, inputs[7]) : -1;
    result->idx_sin = input_count > 8 ? FusionCpuFindInputForValue(ep->ort_api, fused, inputs[8]) : -1;
    result->idx_out = FusionCpuFindOutputForValue(ep->ort_api, fused, outputs[0]);
    result->idx_present_key = FusionCpuFindOutputForValue(ep->ort_api, fused, outputs[1]);
    result->idx_present_value = FusionCpuFindOutputForValue(ep->ort_api, fused, outputs[2]);
    free((void*)inputs);
    free((void*)outputs);
    if (result->idx_qkv < 0 || result->idx_past_key < 0 || result->idx_past_value < 0 ||
        result->idx_out < 0 || result->idx_present_key < 0 || result->idx_present_value < 0 ||
        (result->do_rotary && (result->idx_cos < 0 || result->idx_sin < 0))) {
        free(result);
        return ep->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU GQA could not map fused I/O.");
    }
    *state = result;
    return NULL;
}

static OrtStatus* FusionCpuCreateKernel(FusionCpuEp* ep, const OrtNode* fused,
                                        const OrtNode* inner, void** state,
                                        FusionCpuComputeFn* compute,
                                        FusionCpuFreeFn* destroy) {
    const char* op = NULL;
    const char* domain = NULL;
    OrtStatus* status = ep->ort_api->Node_GetOperatorType(inner, &op);
    if (status != NULL) return status;
    status = ep->ort_api->Node_GetDomain(inner, &domain);
    if (status != NULL) return status;
    *state = NULL;
    if ((domain == NULL || domain[0] == '\0') && strcmp(op, "Mul") == 0) {
        FusionCpuMulState* result = (FusionCpuMulState*)calloc(1, sizeof(*result));
        if (result == NULL) goto oom;
        result->api = ep->ort_api;
        *state = result;
        *compute = FusionCpuMulCompute;
        *destroy = FusionCpuFreeSimpleState;
        return NULL;
    }
    if ((domain == NULL || domain[0] == '\0') && strcmp(op, "Sigmoid") == 0) {
        FusionCpuSigmoidState* result = (FusionCpuSigmoidState*)calloc(1, sizeof(*result));
        if (result == NULL) goto oom;
        result->api = ep->ort_api;
        *state = result;
        *compute = FusionCpuSigmoidCompute;
        *destroy = FusionCpuFreeSimpleState;
        return NULL;
    }
    if (domain != NULL && strcmp(domain, "com.microsoft") == 0 &&
        strcmp(op, "MatMulNBits") == 0) {
        FusionCpuMatMulNBitsState* result;
        int64_t bits = 4;
        int64_t block_size = 0;
        size_t input_count = 0;
        status = ep->ort_api->Node_GetNumInputs(inner, &input_count);
        if (status != NULL) return status;
        FusionCpuReadIntAttr(ep->ort_api, inner, "bits", &bits);
        if (!FusionCpuReadIntAttr(ep->ort_api, inner, "block_size", &block_size) ||
            bits != 4 || block_size <= 0) {
            return ep->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU MatMulNBits attributes unsupported.");
        }
        result = (FusionCpuMatMulNBitsState*)calloc(1, sizeof(*result));
        if (result == NULL) goto oom;
        result->api = ep->ort_api;
        result->bits = (int)bits;
        result->block_size = (int)block_size;
        result->idx_a = 0;
        result->idx_b = 1;
        result->idx_scales = 2;
        result->idx_zp = input_count >= 4 ? 3 : -1;
        *state = result;
        *compute = FusionCpuMatMulNBitsCompute;
        *destroy = FusionCpuFreeSimpleState;
        return NULL;
    }
    if (domain != NULL && strcmp(domain, "com.microsoft") == 0 &&
        strcmp(op, "GroupQueryAttention") == 0) {
        status = FusionCpuCreateGqaState(ep, fused, inner, state);
        if (status != NULL) return status;
        *compute = FusionCpuGqaCompute;
        *destroy = FusionCpuFreeGqaState;
        return NULL;
    }
    status = FusionCpuExtraCreate(ep->ort_api, fused, inner, state, compute, destroy);
    if (status != NULL || *state != NULL) return status;
    return ep->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU EP attempted to compile an unsupported operator.");
oom:
    return ep->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU EP kernel allocation failed.");
}

static OrtStatus* ORT_API_CALL FusionCpuNodeCreateState(OrtNodeComputeInfo* info,
                                                         OrtNodeComputeContext* context,
                                                         void** state) {
    FusionCpuNodeComputeInfo* self = (FusionCpuNodeComputeInfo*)info;
    const char* name = self->ep->ep_api->NodeComputeContext_NodeName(context);
    FusionCpuKernelEntry* entry = FusionCpuEp_FindKernel(self->ep, name);
    if (entry == NULL) {
        return self->ep->ort_api->CreateStatus(ORT_EP_FAIL,
                                                "FusionCPU EP could not locate compiled kernel.");
    }
    *state = entry;
    return NULL;
}

static OrtStatus* ORT_API_CALL FusionCpuNodeCompute(OrtNodeComputeInfo* info, void* state,
                                                     OrtKernelContext* context) {
    FusionCpuKernelEntry* entry = (FusionCpuKernelEntry*)state;
    (void)info;
    return entry->compute(entry->state, context);
}

static void ORT_API_CALL FusionCpuNodeReleaseState(OrtNodeComputeInfo* info, void* state) {
    (void)info;
    (void)state;
}

static OrtStatus* ORT_API_CALL FusionCpuEpCompile(OrtEp* ep, const OrtGraph** graphs,
                                                   const OrtNode** fused_nodes, size_t count,
                                                   OrtNodeComputeInfo** infos,
                                                   OrtNode** ep_context_nodes) {
    FusionCpuEp* self = (FusionCpuEp*)ep;
    size_t i;
    (void)ep_context_nodes;
    for (i = 0; i < count; ++i) {
        const OrtNode** nodes;
        size_t node_count = 0;
        const char* name = NULL;
        void* state = NULL;
        FusionCpuComputeFn compute = NULL;
        FusionCpuFreeFn destroy = NULL;
        FusionCpuNodeComputeInfo* info;
        OrtStatus* status = self->ort_api->Graph_GetNumNodes(graphs[i], &node_count);
        if (status != NULL) return status;
        if (node_count != 1) {
            return self->ort_api->CreateStatus(ORT_EP_FAIL,
                                                "FusionCPU EP only compiles single-node partitions.");
        }
        nodes = (const OrtNode**)calloc(1, sizeof(*nodes));
        if (nodes == NULL) return self->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU EP allocation failed.");
        status = self->ort_api->Graph_GetNodes(graphs[i], nodes, 1);
        if (status == NULL) status = self->ort_api->Node_GetName(fused_nodes[i], &name);
        if (status == NULL) status = FusionCpuCreateKernel(self, fused_nodes[i], nodes[0],
                                                            &state, &compute, &destroy);
        free((void*)nodes);
        if (status != NULL) return status;
        status = FusionCpuEp_AddKernel(self, name != NULL ? name : "", state, compute, destroy);
        if (status != NULL) {
            destroy(state);
            return status;
        }
        info = (FusionCpuNodeComputeInfo*)calloc(1, sizeof(*info));
        if (info == NULL) {
            return self->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU EP allocation failed.");
        }
        info->ep = self;
        info->base.ort_version_supported = ORT_API_VERSION;
        info->base.CreateState = FusionCpuNodeCreateState;
        info->base.Compute = FusionCpuNodeCompute;
        info->base.ReleaseState = FusionCpuNodeReleaseState;
        infos[i] = &info->base;
    }
    return NULL;
}

static void ORT_API_CALL FusionCpuEpReleaseInfos(OrtEp* ep, OrtNodeComputeInfo** infos,
                                                  size_t count) {
    size_t i;
    (void)ep;
    if (infos == NULL) return;
    for (i = 0; i < count; ++i) free(infos[i]);
}

static OrtStatus* ORT_API_CALL FusionCpuEpSync(OrtEp* ep) {
    (void)ep;
    return NULL;
}

FusionCpuEp* FusionCpuEp_Create(const OrtApi* ort_api, const OrtEpApi* ep_api,
                                const OrtLogger* logger, const char* name) {
    FusionCpuEp* result;
    if (ort_api == NULL || ep_api == NULL) return NULL;
    result = (FusionCpuEp*)calloc(1, sizeof(*result));
    if (result == NULL) return NULL;
    result->name = FusionCpuDuplicateString(name);
    if (result->name == NULL) {
        free(result);
        return NULL;
    }
    result->ort_api = ort_api;
    result->ep_api = ep_api;
    result->logger = logger;
#if defined(_WIN32)
    result->kernel_lock = malloc(sizeof(CRITICAL_SECTION));
    if (result->kernel_lock == NULL) {
        free(result->name);
        free(result);
        return NULL;
    }
    InitializeCriticalSection(FusionCpuKernelLock(result));
#endif
    result->base.ort_version_supported = ORT_API_VERSION;
    result->base.GetName = FusionCpuEpGetName;
    result->base.GetCapability = FusionCpuEpGetCapability;
    result->base.Compile = FusionCpuEpCompile;
    result->base.ReleaseNodeComputeInfos = FusionCpuEpReleaseInfos;
    result->base.Sync = FusionCpuEpSync;
    return result;
}

void FusionCpuEp_Destroy(FusionCpuEp* ep) {
    if (ep == NULL) return;
    FusionCpuLockKernels(ep);
    FusionCpuFreeKernelEntries(ep);
    FusionCpuUnlockKernels(ep);
#if defined(_WIN32)
    DeleteCriticalSection(FusionCpuKernelLock(ep));
    free(ep->kernel_lock);
#endif
    free(ep->name);
    free(ep);
}
