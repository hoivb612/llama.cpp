// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#include "fusion_cpu_extra_ops.h"

#include <intrin.h>
#include "fp16_utils.h"
#include "matmulnbits_avx512.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum FusionCpuExtraKind {
    FUSION_CPU_EXTRA_RMS_NORM,
    FUSION_CPU_EXTRA_SHAPE,
    FUSION_CPU_EXTRA_CAST,
    FUSION_CPU_EXTRA_GATHER,
    FUSION_CPU_EXTRA_REDUCE_SUM,
    FUSION_CPU_EXTRA_SUB
} FusionCpuExtraKind;

typedef struct FusionCpuExtraState {
    const OrtApi* api;
    FusionCpuExtraKind kind;
    int input0;
    int input1;
    int input2;
    int output0;
    int output1;
    float epsilon;
    int64_t attr0;
    int64_t attr1;
    bool flag0;
    bool flag1;
    ONNXTensorElementDataType type;
    // fp16 RMSNorm: fp32-widened copy of the constant gamma (scale) initializer,
    // populated once on the first fp16 Compute and reused across Run() calls.
    float* gamma_f32;
    int64_t gamma_f32_width;
} FusionCpuExtraState;

static OrtStatus* FusionCpuExtraInput(const OrtApi* api, const OrtKernelContext* context,
                                      size_t index, const OrtValue** value,
                                      ONNXTensorElementDataType* type,
                                      const int64_t** shape, size_t* rank,
                                      const void** data) {
    OrtStatus* status = api->KernelContext_GetInput(context, index, value);
    if (status != NULL) {
        FusionCpuDiagnosticStatus(api, "Extra KernelContext_GetInput",
                                  (int)index, status);
        return status;
    }
    if (*value == NULL) {
        status = api->CreateStatus(ORT_INVALID_ARGUMENT,
                                   "FusionCPU extra op input is absent.");
        FusionCpuDiagnosticStatus(api, "Extra KernelContext_GetInput returned NULL",
                                  (int)index, status);
        return status;
    }
    status = api->GetTensorElementTypeAndShapeDataReference(*value, type, shape, rank);
    if (status != NULL) {
        FusionCpuDiagnosticStatus(api,
                                  "Extra GetTensorElementTypeAndShapeDataReference",
                                  (int)index, status);
        return status;
    }
    status = api->GetTensorData(*value, data);
    if (status != NULL) {
        FusionCpuDiagnosticStatus(api, "Extra GetTensorData", (int)index, status);
    }
    return status;
}

static OrtStatus* FusionCpuExtraOutput(const OrtApi* api, OrtKernelContext* context,
                                       size_t index, const int64_t* shape, size_t rank,
                                       void** data) {
    OrtValue* value = NULL;
    OrtStatus* status = api->KernelContext_GetOutput(context, index, shape, rank, &value);
    if (status != NULL) {
        FusionCpuDiagnosticStatus(api, "Extra KernelContext_GetOutput",
                                  (int)index, status);
        return status;
    }
    if (value == NULL) {
        return api->CreateStatus(ORT_INVALID_ARGUMENT, "FusionCPU extra op output is absent.");
    }
    status = api->GetTensorMutableData(value, data);
    if (status != NULL) {
        FusionCpuDiagnosticStatus(api, "Extra GetTensorMutableData",
                                  (int)index, status);
    }
    return status;
}

static bool FusionCpuExtraCount(const int64_t* shape, size_t rank, size_t* count) {
    size_t result = 1;
    size_t i;
    for (i = 0; i < rank; ++i) {
        if (shape[i] < 0 || (shape[i] != 0 && result > SIZE_MAX / (size_t)shape[i])) return false;
        result *= (size_t)shape[i];
    }
    *count = result;
    return true;
}

static size_t FusionCpuExtraElementSize(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return 1;
        default:
            return 0;
    }
}

static bool FusionCpuExtraReadInt(const OrtApi* api, const OrtNode* node,
                                  const char* name, int64_t* output) {
    const OrtOpAttr* attr = NULL;
    size_t bytes = 0;
    OrtStatus* status = api->Node_GetAttributeByName(node, name, &attr);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    if (attr == NULL) return false;
    status = api->ReadOpAttr(attr, ORT_OP_ATTR_INT, output, sizeof(*output), &bytes);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    return bytes == sizeof(*output);
}

static bool FusionCpuExtraReadFloat(const OrtApi* api, const OrtNode* node,
                                    const char* name, float* output) {
    const OrtOpAttr* attr = NULL;
    size_t bytes = 0;
    OrtStatus* status = api->Node_GetAttributeByName(node, name, &attr);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    if (attr == NULL) return false;
    status = api->ReadOpAttr(attr, ORT_OP_ATTR_FLOAT, output, sizeof(*output), &bytes);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    return bytes == sizeof(*output);
}

static bool FusionCpuExtraGetNodeValues(const OrtApi* api, const OrtNode* node,
                                        bool inputs, const OrtValueInfo*** values,
                                        size_t* count) {
    const OrtValueInfo** result;
    OrtStatus* status;
    *values = NULL;
    *count = 0;
    status = inputs ? api->Node_GetNumInputs(node, count)
                    : api->Node_GetNumOutputs(node, count);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    if (*count == 0) return true;
    result = (const OrtValueInfo**)calloc(*count, sizeof(*result));
    if (result == NULL) return false;
    status = inputs ? api->Node_GetInputs(node, result, *count)
                    : api->Node_GetOutputs(node, result, *count);
    if (status != NULL) {
        api->ReleaseStatus(status);
        free((void*)result);
        return false;
    }
    *values = result;
    return true;
}

static bool FusionCpuExtraValueType(const OrtApi* api, const OrtValueInfo* value,
                                    ONNXTensorElementDataType* type) {
    const OrtTypeInfo* type_info = NULL;
    const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
    OrtStatus* status;
    if (value == NULL) return false;
    status = api->GetValueInfoTypeInfo(value, &type_info);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    status = api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (status != NULL || tensor_info == NULL) {
        if (status != NULL) api->ReleaseStatus(status);
        return false;
    }
    status = api->GetTensorElementType(tensor_info, type);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    return true;
}

static bool FusionCpuExtraIsConstant(const OrtApi* api, const OrtValueInfo* value) {
    bool result = false;
    OrtStatus* status;
    if (value == NULL) return false;
    status = api->ValueInfo_IsConstantInitializer(value, &result);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    return result;
}

static int FusionCpuExtraFindValue(const OrtApi* api, const OrtNode* fused_node,
                                   const OrtValueInfo* inner_value, bool input) {
    const OrtValueInfo** fused_values = NULL;
    size_t count = 0;
    size_t index;
    const char* inner_name = NULL;
    OrtStatus* status;
    int result = -1;
    if (inner_value == NULL) return -1;
    status = api->GetValueInfoName(inner_value, &inner_name);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return -1;
    }
    if (!FusionCpuExtraGetNodeValues(api, fused_node, input,
                                     &fused_values, &count)) {
        return -1;
    }
    for (index = 0; index < count; ++index) {
        const char* fused_name = NULL;
        if (fused_values[index] == NULL) continue;
        status = api->GetValueInfoName(fused_values[index], &fused_name);
        if (status != NULL) {
            api->ReleaseStatus(status);
            continue;
        }
        if (inner_name != NULL && fused_name != NULL &&
            strcmp(inner_name, fused_name) == 0) {
            result = (int)index;
            break;
        }
    }
    free((void*)fused_values);
    return result;
}

typedef struct FusionCpuRmsNormParallelContext {
    const float* input;
    const float* skip;
    const float* gamma;
    float* output;
    float* sum;
    size_t rows;
    int64_t width;
    float epsilon;
} FusionCpuRmsNormParallelContext;

static void FusionCpuRmsNormParallelBody(void* opaque, int task, int nways) {
    FusionCpuRmsNormParallelContext* context =
        (FusionCpuRmsNormParallelContext*)opaque;
    const size_t begin = (size_t)task * context->rows / (size_t)nways;
    const size_t end = (size_t)(task + 1) * context->rows / (size_t)nways;
    size_t row;
    for (row = begin; row < end; ++row) {
        const float* input = context->input + row * (size_t)context->width;
        const float* residual = context->skip != NULL
            ? context->skip + row * (size_t)context->width : NULL;
        float* output = context->output + row * (size_t)context->width;
        float* sum = context->sum != NULL
            ? context->sum + row * (size_t)context->width : NULL;
        float mean_square = 0.0f;
        int64_t column;
        for (column = 0; column < context->width; ++column) {
            const float value =
                residual != NULL ? input[column] + residual[column] : input[column];
            mean_square += value * value;
            if (sum != NULL) sum[column] = value;
        }
        mean_square =
            1.0f / sqrtf(mean_square / (float)context->width + context->epsilon);
        for (column = 0; column < context->width; ++column) {
            const float value =
                residual != NULL ? input[column] + residual[column] : input[column];
            output[column] = value * mean_square * context->gamma[column];
        }
    }
}

typedef struct FusionCpuRmsNormF16ParallelContext {
    const uint16_t* input;
    const uint16_t* skip;
    const float* gamma;
    uint16_t* output;
    uint16_t* sum;
    size_t rows;
    int64_t width;
    float epsilon;
} FusionCpuRmsNormF16ParallelContext;

// AVX512F fp16 RMSNorm: reads fp16 input/skip directly and writes fp16
// output/sum with fp32 accumulation, using the cached fp32 gamma. Avoids the
// whole-tensor widen/narrow scratch that the scalar fallback needs.
static void FusionCpuRmsNormF16ParallelBody(void* opaque, int task, int nways) {
    FusionCpuRmsNormF16ParallelContext* context =
        (FusionCpuRmsNormF16ParallelContext*)opaque;
    const size_t begin = (size_t)task * context->rows / (size_t)nways;
    const size_t end = (size_t)(task + 1) * context->rows / (size_t)nways;
    size_t row;
    for (row = begin; row < end; ++row) {
        const size_t offset = row * (size_t)context->width;
        RmsNormRowF16Avx512(context->input + offset,
                            context->skip != NULL ? context->skip + offset : NULL,
                            context->gamma,
                            context->output + offset,
                            context->sum != NULL ? context->sum + offset : NULL,
                            context->width, context->epsilon);
    }
}

static OrtStatus* FusionCpuExtraRmsNorm(FusionCpuExtraState* self,
                                        OrtKernelContext* context) {
    const OrtValue* input;
    const OrtValue* gamma;
    const OrtValue* skip = NULL;
    ONNXTensorElementDataType type;
    const int64_t* shape;
    const int64_t* input_shape;
    size_t rank;
    size_t input_rank;
    const void* input_void;
    const void* gamma_void;
    const void* skip_void = NULL;
    void* output_void;
    void* sum_void = NULL;
    size_t count;
    int64_t width;
    FusionCpuRmsNormParallelContext parallel_context;
    OrtStatus* status;
    bool io_fp16 = false;
    float* in_f32 = NULL;
    float* skip_f32 = NULL;
    float* out_f32 = NULL;
    float* sum_f32 = NULL;

    status = FusionCpuExtraInput(self->api, context, (size_t)self->input0, &input, &type,
                                 &shape, &rank, &input_void);
    if (status != NULL) return status;
    if ((type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
         type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) || rank == 0 ||
        !FusionCpuExtraCount(shape, rank, &count) || shape[rank - 1] <= 0) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "RMSNorm requires non-empty float/float16 input.");
    }
    io_fp16 = (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    width = shape[rank - 1];
    input_shape = shape;
    input_rank = rank;
    status = FusionCpuExtraInput(self->api, context, (size_t)self->input2, &gamma, &type,
                                 &shape, &rank, &gamma_void);
    if (status != NULL) return status;
    if (type != (io_fp16 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                         : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "RMSNorm gamma dtype mismatch.");
    }
    if (self->flag0) {
        status = FusionCpuExtraInput(self->api, context, (size_t)self->input1, &skip, &type,
                                     &shape, &rank, &skip_void);
        if (status != NULL) return status;
        if (type != (io_fp16 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                             : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "RMSNorm skip dtype mismatch.");
        }
    }
    status = FusionCpuExtraOutput(self->api, context, (size_t)self->output0,
                                  input_shape, input_rank, &output_void);
    if (status != NULL) return status;
    if (self->flag0 && self->output1 >= 0) {
        size_t output_count = 0;
        status = self->api->KernelContext_GetOutputCount(context, &output_count);
        if (status != NULL) return status;
        if ((size_t)self->output1 < output_count) {
            status = FusionCpuExtraOutput(self->api, context, (size_t)self->output1,
                                          input_shape, input_rank, &sum_void);
            if (status != NULL) return status;
        }
    }
    if (io_fp16) {
        /* Cache the widened constant gamma once; reused across Run() calls. */
        if (self->gamma_f32 == NULL || self->gamma_f32_width != width) {
            float* g = (float*)malloc((size_t)width * sizeof(float));
            int64_t i;
            if (g == NULL) return self->api->CreateStatus(ORT_EP_FAIL, "RMSNorm gamma cache alloc failed.");
            for (i = 0; i < width; ++i) g[i] = FusionCpuHalfToFloat(((const uint16_t*)gamma_void)[i]);
            free(self->gamma_f32);
            self->gamma_f32 = g;
            self->gamma_f32_width = width;
        }
        if (FusionCpuHasAvx512f()) {
            /* Fast path: fp16 I/O in place, fp32 accumulation, cached gamma. */
            FusionCpuRmsNormF16ParallelContext f16;
            f16.input = (const uint16_t*)input_void;
            f16.skip = self->flag0 ? (const uint16_t*)skip_void : NULL;
            f16.gamma = self->gamma_f32;
            f16.output = (uint16_t*)output_void;
            f16.sum = (uint16_t*)sum_void;  /* NULL when no second output */
            f16.rows = count / (size_t)width;
            f16.width = width;
            f16.epsilon = self->epsilon;
            fusion_pool_parallel_for((int)f16.rows, FusionCpuRmsNormF16ParallelBody, &f16);
            return NULL;
        }
        /* Scalar fallback (no AVX512): widen input/skip, narrow output/sum. */
        {
            size_t i;
            in_f32 = (float*)malloc(count * sizeof(float));
            out_f32 = (float*)malloc(count * sizeof(float));
            if (self->flag0) skip_f32 = (float*)malloc(count * sizeof(float));
            if (sum_void != NULL) sum_f32 = (float*)malloc(count * sizeof(float));
            if (in_f32 == NULL || out_f32 == NULL ||
                (self->flag0 && skip_f32 == NULL) ||
                (sum_void != NULL && sum_f32 == NULL)) {
                free(in_f32); free(out_f32); free(skip_f32); free(sum_f32);
                return self->api->CreateStatus(ORT_EP_FAIL, "RMSNorm fp16 scratch allocation failed.");
            }
            for (i = 0; i < count; ++i) in_f32[i] = FusionCpuHalfToFloat(((const uint16_t*)input_void)[i]);
            if (self->flag0) {
                for (i = 0; i < count; ++i) skip_f32[i] = FusionCpuHalfToFloat(((const uint16_t*)skip_void)[i]);
            }
        }
    }
    parallel_context.input = io_fp16 ? in_f32 : (const float*)input_void;
    parallel_context.skip = self->flag0 ? (io_fp16 ? skip_f32 : (const float*)skip_void) : NULL;
    parallel_context.gamma = io_fp16 ? self->gamma_f32 : (const float*)gamma_void;
    parallel_context.output = io_fp16 ? out_f32 : (float*)output_void;
    parallel_context.sum = io_fp16 ? sum_f32 : (float*)sum_void;
    parallel_context.rows = count / (size_t)width;
    parallel_context.width = width;
    parallel_context.epsilon = self->epsilon;
    fusion_pool_parallel_for((int)parallel_context.rows,
                         FusionCpuRmsNormParallelBody, &parallel_context);
    if (io_fp16) {
        size_t i;
        for (i = 0; i < count; ++i) ((uint16_t*)output_void)[i] = FusionCpuFloatToHalf(out_f32[i]);
        if (sum_void != NULL && sum_f32 != NULL) {
            for (i = 0; i < count; ++i) ((uint16_t*)sum_void)[i] = FusionCpuFloatToHalf(sum_f32[i]);
        }
        free(in_f32); free(out_f32); free(skip_f32); free(sum_f32);
    }
    return NULL;
}

static OrtStatus* FusionCpuExtraShape(FusionCpuExtraState* self, OrtKernelContext* context) {
    const OrtValue* input;
    ONNXTensorElementDataType type;
    const int64_t* shape;
    size_t rank;
    const void* data;
    int64_t start;
    int64_t end;
    int64_t output_shape[1];
    void* output;
    int64_t i;
    OrtStatus* status = FusionCpuExtraInput(self->api, context, (size_t)self->input0, &input,
                                            &type, &shape, &rank, &data);
    if (status != NULL) return status;
    start = self->attr0 < 0 ? self->attr0 + (int64_t)rank : self->attr0;
    end = self->flag0 ? (self->attr1 < 0 ? self->attr1 + (int64_t)rank : self->attr1)
                      : (int64_t)rank;
    if (start < 0) start = 0;
    if (start > (int64_t)rank) start = (int64_t)rank;
    if (end < 0) end = 0;
    if (end > (int64_t)rank) end = (int64_t)rank;
    output_shape[0] = end > start ? end - start : 0;
    status = FusionCpuExtraOutput(self->api, context, (size_t)self->output0,
                                  output_shape, 1, &output);
    if (status != NULL) return status;
    for (i = 0; i < output_shape[0]; ++i) ((int64_t*)output)[i] = shape[start + i];
    return NULL;
}

static void FusionCpuExtraCastOne(const void* source, void* destination, size_t count,
                                  ONNXTensorElementDataType from, ONNXTensorElementDataType to) {
    size_t i;
    /* Fast paths for the fp16<->fp32 Casts that appear in q4f16 graphs. Both are
       bit-identical to the generic (double-routed) loop below: fp16->fp32 is
       exact, and fp32->fp16 uses the same RNE conversion as FusionCpuFloatToHalf. */
    if (FusionCpuHasAvx512f()) {
        if (from == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
            to == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            FusionCpuWidenHalfToFloatAvx512((float*)destination, (const uint16_t*)source,
                                            (int64_t)count);
            return;
        }
        if (from == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            to == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            FusionCpuNarrowFloatToHalfAvx512((uint16_t*)destination, (const float*)source,
                                             (int64_t)count);
            return;
        }
    }
    for (i = 0; i < count; ++i) {
        double value;
        switch (from) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: value = ((const float*)source)[i]; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: value = FusionCpuHalfToFloat(((const uint16_t*)source)[i]); break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: value = ((const double*)source)[i]; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: value = (double)((const int64_t*)source)[i]; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: value = (double)((const int32_t*)source)[i]; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: value = (double)((const uint8_t*)source)[i]; break;
            default: value = 0.0; break;
        }
        switch (to) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: ((float*)destination)[i] = (float)value; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: ((uint16_t*)destination)[i] = FusionCpuFloatToHalf((float)value); break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: ((double*)destination)[i] = value; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: ((int64_t*)destination)[i] = (int64_t)value; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: ((int32_t*)destination)[i] = (int32_t)value; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: ((uint8_t*)destination)[i] = (uint8_t)value; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: ((uint8_t*)destination)[i] = value != 0.0; break;
            default: break;
        }
    }
}

static OrtStatus* FusionCpuExtraCast(FusionCpuExtraState* self, OrtKernelContext* context) {
    const OrtValue* input;
    ONNXTensorElementDataType from;
    const int64_t* shape;
    size_t rank;
    const void* source;
    void* destination;
    size_t count;
    OrtStatus* status = FusionCpuExtraInput(self->api, context, (size_t)self->input0, &input,
                                            &from, &shape, &rank, &source);
    if (status != NULL) return status;
    if (!FusionCpuExtraCount(shape, rank, &count) || FusionCpuExtraElementSize(from) == 0 ||
        FusionCpuExtraElementSize(self->type) == 0) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "Cast received an unsupported type or shape.");
    }
    status = FusionCpuExtraOutput(self->api, context, (size_t)self->output0, shape, rank, &destination);
    if (status != NULL) return status;
    FusionCpuExtraCastOne(source, destination, count, from, self->type);
    return NULL;
}

static OrtStatus* FusionCpuExtraGather(FusionCpuExtraState* self, OrtKernelContext* context) {
    const OrtValue* data_value;
    const OrtValue* index_value;
    ONNXTensorElementDataType data_type;
    ONNXTensorElementDataType index_type;
    const int64_t* data_shape;
    const int64_t* index_shape;
    size_t data_rank;
    size_t index_rank;
    const void* data;
    const void* indexes;
    int64_t axis;
    int64_t outer = 1;
    int64_t inner = 1;
    int64_t index_count = 1;
    int64_t* output_shape;
    int64_t output_rank;
    uint8_t* output;
    size_t element_size;
    int64_t i;
    OrtStatus* status = FusionCpuExtraInput(self->api, context, (size_t)self->input0, &data_value,
                                            &data_type, &data_shape, &data_rank, &data);
    if (status != NULL) return status;
    status = FusionCpuExtraInput(self->api, context, (size_t)self->input1, &index_value,
                                 &index_type, &index_shape, &index_rank, &indexes);
    if (status != NULL) return status;
    axis = self->attr0 < 0 ? self->attr0 + (int64_t)data_rank : self->attr0;
    element_size = FusionCpuExtraElementSize(data_type);
    if (axis < 0 || axis >= (int64_t)data_rank || element_size == 0 ||
        (index_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
         index_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "Gather received unsupported arguments.");
    }
    for (i = 0; i < axis; ++i) outer *= data_shape[i];
    for (i = axis + 1; i < (int64_t)data_rank; ++i) inner *= data_shape[i];
    for (i = 0; i < (int64_t)index_rank; ++i) index_count *= index_shape[i];
    output_rank = (int64_t)data_rank - 1 + (int64_t)index_rank;
    output_shape = (int64_t*)malloc((size_t)output_rank * sizeof(*output_shape));
    if (output_shape == NULL && output_rank != 0) {
        return self->api->CreateStatus(ORT_EP_FAIL, "Gather output shape allocation failed.");
    }
    for (i = 0; i < axis; ++i) output_shape[i] = data_shape[i];
    for (i = 0; i < (int64_t)index_rank; ++i) output_shape[axis + i] = index_shape[i];
    for (i = axis + 1; i < (int64_t)data_rank; ++i) {
        output_shape[i - 1 + (int64_t)index_rank] = data_shape[i];
    }
    status = FusionCpuExtraOutput(self->api, context, (size_t)self->output0, output_shape,
                                  (size_t)output_rank, (void**)&output);
    free(output_shape);
    if (status != NULL) return status;
    for (i = 0; i < outer * index_count; ++i) {
        int64_t index = index_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            ? ((const int64_t*)indexes)[i % index_count]
            : (int64_t)((const int32_t*)indexes)[i % index_count];
        const int64_t outer_index = i / index_count;
        const size_t bytes = (size_t)inner * element_size;
        if (index < 0) index += data_shape[axis];
        if (index < 0 || index >= data_shape[axis]) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "Gather index is out of range.");
        }
        memcpy(output + (size_t)i * bytes,
               (const uint8_t*)data + (size_t)((outer_index * data_shape[axis] + index) * inner) * element_size,
               bytes);
    }
    return NULL;
}

static OrtStatus* FusionCpuExtraReduceSum(FusionCpuExtraState* self, OrtKernelContext* context) {
    const OrtValue* data_value;
    const OrtValue* axes_value = NULL;
    ONNXTensorElementDataType type;
    ONNXTensorElementDataType axes_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    const int64_t* data_shape;
    const int64_t* axes_shape;
    size_t data_rank;
    size_t axes_rank;
    const void* data;
    const void* axes_data = NULL;
    uint8_t* reduced;
    int64_t* output_shape;
    size_t input_count;
    size_t output_count = 0;
    int64_t i;
    int64_t numel = 1;
    int64_t output_numel = 1;
    OrtStatus* status;
    status = FusionCpuExtraInput(self->api, context, (size_t)self->input0, &data_value, &type,
                                 &data_shape, &data_rank, &data);
    if (status != NULL) return status;
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "ReduceSum supports int64 and float only.");
    }
    status = self->api->KernelContext_GetInputCount(context, &input_count);
    if (status != NULL) return status;
    if (self->input1 >= 0 && (size_t)self->input1 < input_count) {
        status = FusionCpuExtraInput(self->api, context, (size_t)self->input1, &axes_value, &axes_type,
                                     &axes_shape, &axes_rank, &axes_data);
        if (status != NULL) return status;
        if (axes_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
            axes_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "ReduceSum axes must be int32 or int64.");
        }
    } else {
        axes_rank = 0;
        axes_shape = NULL;
    }
    reduced = (uint8_t*)calloc(data_rank, 1);
    output_shape = (int64_t*)malloc(data_rank * sizeof(*output_shape));
    if ((reduced == NULL && data_rank != 0) || (output_shape == NULL && data_rank != 0)) {
        free(reduced);
        free(output_shape);
        return self->api->CreateStatus(ORT_EP_FAIL, "ReduceSum allocation failed.");
    }
    if (axes_data != NULL) {
        int64_t axes_count = 1;
        for (i = 0; i < (int64_t)axes_rank; ++i) axes_count *= axes_shape[i];
        for (i = 0; i < axes_count; ++i) {
            int64_t axis = axes_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                ? ((const int64_t*)axes_data)[i] : (int64_t)((const int32_t*)axes_data)[i];
            if (axis < 0) axis += (int64_t)data_rank;
            if (axis < 0 || axis >= (int64_t)data_rank) {
                free(reduced);
                free(output_shape);
                return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "ReduceSum axis is out of range.");
            }
            reduced[axis] = 1;
        }
        if (axes_count == 0 && !self->flag1) {
            memset(reduced, 1, data_rank);
        }
        if (axes_count == 0 && self->flag1) {
            free(reduced);
            free(output_shape);
            /* Identity requires output allocated with the original shape. */
            {
                void* output;
                status = FusionCpuExtraOutput(self->api, context, (size_t)self->output0,
                                              data_shape, data_rank, &output);
                if (status != NULL) return status;
                for (i = 0; i < (int64_t)data_rank; ++i) numel *= data_shape[i];
                memcpy(output, data, (size_t)numel * FusionCpuExtraElementSize(type));
                return NULL;
            }
        }
    } else {
        memset(reduced, 1, data_rank);
    }
    for (i = 0; i < (int64_t)data_rank; ++i) {
        numel *= data_shape[i];
        if (reduced[i]) {
            if (self->flag0) output_shape[output_count++] = 1;
        } else {
            output_shape[output_count++] = data_shape[i];
            output_numel *= data_shape[i];
        }
    }
    {
        void* output;
        int64_t* indices = (int64_t*)calloc(data_rank, sizeof(*indices));
        int64_t* strides = (int64_t*)calloc(data_rank, sizeof(*strides));
        int64_t stride = 1;
        if ((indices == NULL && data_rank != 0) || (strides == NULL && data_rank != 0)) {
            free(indices); free(strides); free(reduced); free(output_shape);
            return self->api->CreateStatus(ORT_EP_FAIL, "ReduceSum allocation failed.");
        }
        for (i = (int64_t)data_rank - 1; i >= 0; --i) {
            strides[i] = reduced[i] ? 0 : stride;
            if (!reduced[i]) stride *= data_shape[i];
        }
        status = FusionCpuExtraOutput(self->api, context, (size_t)self->output0, output_shape,
                                      output_count, &output);
        free(output_shape);
        if (status != NULL) {
            free(indices); free(strides); free(reduced);
            return status;
        }
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            memset(output, 0, (size_t)output_numel * sizeof(int64_t));
        } else {
            memset(output, 0, (size_t)output_numel * sizeof(float));
        }
        for (i = 0; i < numel; ++i) {
            int64_t offset = 0;
            int64_t dim;
            for (dim = 0; dim < (int64_t)data_rank; ++dim) offset += indices[dim] * strides[dim];
            if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                ((int64_t*)output)[offset] += ((const int64_t*)data)[i];
            else
                ((float*)output)[offset] += ((const float*)data)[i];
            for (dim = (int64_t)data_rank - 1; dim >= 0; --dim) {
                if (++indices[dim] < data_shape[dim]) break;
                indices[dim] = 0;
            }
        }
        free(indices);
        free(strides);
    }
    free(reduced);
    return NULL;
}

static OrtStatus* FusionCpuExtraSub(FusionCpuExtraState* self, OrtKernelContext* context) {
    const OrtValue* left_value;
    const OrtValue* right_value;
    ONNXTensorElementDataType left_type;
    ONNXTensorElementDataType right_type;
    const int64_t* left_shape;
    const int64_t* right_shape;
    size_t left_rank;
    size_t right_rank;
    const void* left;
    const void* right;
    size_t rank;
    int64_t* output_shape;
    int64_t* left_strides;
    int64_t* right_strides;
    int64_t* indices;
    int64_t numel = 1;
    size_t i;
    OrtStatus* status = FusionCpuExtraInput(self->api, context, (size_t)self->input0, &left_value,
                                            &left_type, &left_shape, &left_rank, &left);
    if (status != NULL) return status;
    status = FusionCpuExtraInput(self->api, context, (size_t)self->input1, &right_value,
                                 &right_type, &right_shape, &right_rank, &right);
    if (status != NULL) return status;
    if (left_type != right_type || (left_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
                                    left_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
        return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "Sub supports matching int64 or float tensors.");
    }
    rank = left_rank > right_rank ? left_rank : right_rank;
    output_shape = (int64_t*)calloc(rank, sizeof(*output_shape));
    left_strides = (int64_t*)calloc(rank, sizeof(*left_strides));
    right_strides = (int64_t*)calloc(rank, sizeof(*right_strides));
    indices = (int64_t*)calloc(rank, sizeof(*indices));
    if ((output_shape == NULL || left_strides == NULL || right_strides == NULL || indices == NULL) && rank != 0) {
        free(output_shape); free(left_strides); free(right_strides); free(indices);
        return self->api->CreateStatus(ORT_EP_FAIL, "Sub allocation failed.");
    }
    for (i = 0; i < rank; ++i) {
        int64_t ld = i + left_rank >= rank ? left_shape[i - (rank - left_rank)] : 1;
        int64_t rd = i + right_rank >= rank ? right_shape[i - (rank - right_rank)] : 1;
        if (ld != rd && ld != 1 && rd != 1) {
            free(output_shape); free(left_strides); free(right_strides); free(indices);
            return self->api->CreateStatus(ORT_INVALID_ARGUMENT, "Sub broadcast shapes are incompatible.");
        }
        output_shape[i] = ld > rd ? ld : rd;
        numel *= output_shape[i];
    }
    {
        int64_t left_stride = 1;
        int64_t right_stride = 1;
        int64_t dimension;
        void* output;
        /* Broadcast strides must be each INPUT's OWN contiguous stride (product of
           that input's trailing dims), 0 where the input dim is 1 or absent -- NOT
           the output stride. Using the output stride over-counts the offset when an
           operand is broadcast on a trailing dim (e.g. left [4,1] vs right [4,3]),
           driving the index past the input buffer (OOB read / access violation). */
        for (dimension = (int64_t)rank - 1; dimension >= 0; --dimension) {
            int64_t li = dimension - ((int64_t)rank - (int64_t)left_rank);
            int64_t ri = dimension - ((int64_t)rank - (int64_t)right_rank);
            int64_t ld = li >= 0 ? left_shape[li] : 1;
            int64_t rd = ri >= 0 ? right_shape[ri] : 1;
            left_strides[dimension] = ld == 1 ? 0 : left_stride;
            right_strides[dimension] = rd == 1 ? 0 : right_stride;
            if (li >= 0) left_stride *= ld;
            if (ri >= 0) right_stride *= rd;
        }
        status = FusionCpuExtraOutput(self->api, context, (size_t)self->output0,
                                      output_shape, rank, &output);
        if (status != NULL) {
            free(output_shape); free(left_strides); free(right_strides); free(indices);
            return status;
        }
        for (i = 0; i < (size_t)numel; ++i) {
            int64_t left_offset = 0;
            int64_t right_offset = 0;
            int64_t offset_dimension;
            for (offset_dimension = 0; offset_dimension < (int64_t)rank; ++offset_dimension) {
                left_offset += indices[offset_dimension] * left_strides[offset_dimension];
                right_offset += indices[offset_dimension] * right_strides[offset_dimension];
            }
            if (left_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                ((int64_t*)output)[i] = ((const int64_t*)left)[left_offset] -
                                        ((const int64_t*)right)[right_offset];
            else
                ((float*)output)[i] = ((const float*)left)[left_offset] -
                                      ((const float*)right)[right_offset];
            for (dimension = (int64_t)rank - 1; dimension >= 0; --dimension) {
                if (++indices[dimension] < output_shape[dimension]) break;
                indices[dimension] = 0;
            }
        }
        free(output_shape);
    }
    free(left_strides);
    free(right_strides);
    free(indices);
    return NULL;
}

static OrtStatus* ORT_API_CALL FusionCpuExtraCompute(void* state, OrtKernelContext* context) {
    FusionCpuExtraState* self = (FusionCpuExtraState*)state;
    switch (self->kind) {
        case FUSION_CPU_EXTRA_RMS_NORM: return FusionCpuExtraRmsNorm(self, context);
        case FUSION_CPU_EXTRA_SHAPE: return FusionCpuExtraShape(self, context);
        case FUSION_CPU_EXTRA_CAST: return FusionCpuExtraCast(self, context);
        case FUSION_CPU_EXTRA_GATHER: return FusionCpuExtraGather(self, context);
        case FUSION_CPU_EXTRA_REDUCE_SUM: return FusionCpuExtraReduceSum(self, context);
        case FUSION_CPU_EXTRA_SUB: return FusionCpuExtraSub(self, context);
        default: return self->api->CreateStatus(ORT_EP_FAIL, "Unknown FusionCPU extra kernel.");
    }
}

static void ORT_API_CALL FusionCpuExtraDestroy(void* state) {
    if (state != NULL) {
        free(((FusionCpuExtraState*)state)->gamma_f32);
    }
    free(state);
}

bool FusionCpuExtraSupportsNode(const OrtApi* api, const OrtNode* node) {
    const char* op = NULL;
    const char* domain = NULL;
    const OrtValueInfo** inputs = NULL;
    const OrtValueInfo** outputs = NULL;
    size_t input_count = 0;
    size_t output_count = 0;
    ONNXTensorElementDataType type0;
    ONNXTensorElementDataType type1;
    ONNXTensorElementDataType output_type;
    int64_t axis = -1;
    const char* enabled;
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
    enabled = getenv("ORT_FUSIONCPU_EXTRA_OPS");
    if (enabled != NULL && enabled[0] == '0') return false;
    if (!FusionCpuExtraGetNodeValues(api, node, true, &inputs, &input_count) ||
        !FusionCpuExtraGetNodeValues(api, node, false, &outputs, &output_count)) {
        free((void*)inputs);
        free((void*)outputs);
        return false;
    }
    if (domain == NULL || domain[0] == '\0') {
        if (strcmp(op, "SimplifiedLayerNormalization") == 0) {
            enabled = getenv("ORT_FUSIONCPU_EXTRA_NORM");
            if ((enabled == NULL || enabled[0] != '0') && input_count == 2 &&
                output_count >= 1 && FusionCpuExtraValueType(api, inputs[0], &type0) &&
                FusionCpuExtraValueType(api, inputs[1], &type1) &&
                FusionCpuExtraValueType(api, outputs[0], &output_type) &&
                (type0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                 type0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) &&
                type1 == type0 && output_type == type0 &&
                FusionCpuExtraIsConstant(api, inputs[1])) {
                FusionCpuExtraReadInt(api, node, "axis", &axis);
                free((void*)inputs);
                free((void*)outputs);
                return axis == -1;
            }
        } else {
            enabled = getenv("ORT_FUSIONCPU_EXTRA_MASKPREP");
            if (enabled == NULL || enabled[0] != '0') {
                bool result = false;
                if (strcmp(op, "Shape") == 0) {
                    result = input_count == 1 && output_count == 1 &&
                        inputs[0] != NULL && outputs[0] != NULL;
                } else if (strcmp(op, "Cast") == 0) {
                    result = input_count == 1 && output_count == 1 &&
                        FusionCpuExtraValueType(api, inputs[0], &type0) &&
                        FusionCpuExtraValueType(api, outputs[0], &output_type) &&
                        FusionCpuExtraElementSize(type0) != 0 &&
                        FusionCpuExtraElementSize(output_type) != 0;
                } else if (strcmp(op, "Gather") == 0) {
                    result = input_count == 2 && output_count == 1 &&
                        FusionCpuExtraValueType(api, inputs[0], &type0) &&
                        FusionCpuExtraValueType(api, inputs[1], &type1) &&
                        FusionCpuExtraElementSize(type0) != 0 &&
                        (type1 == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
                         type1 == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
                } else if (strcmp(op, "ReduceSum") == 0) {
                    result = (input_count == 1 || input_count == 2) && output_count == 1 &&
                        FusionCpuExtraValueType(api, inputs[0], &type0) &&
                        (type0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                         type0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
                } else if (strcmp(op, "Sub") == 0) {
                    result = input_count == 2 && output_count == 1 &&
                        FusionCpuExtraValueType(api, inputs[0], &type0) &&
                        FusionCpuExtraValueType(api, inputs[1], &type1) &&
                        type0 == type1 &&
                        (type0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                         type0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
                }
                free((void*)inputs);
                free((void*)outputs);
                return result;
            }
        }
    } else if (strcmp(domain, "com.microsoft") == 0 &&
               strcmp(op, "SkipSimplifiedLayerNormalization") == 0) {
        enabled = getenv("ORT_FUSIONCPU_EXTRA_NORM");
        if ((enabled == NULL || enabled[0] != '0') && input_count == 3 && output_count >= 1 &&
            FusionCpuExtraValueType(api, inputs[0], &type0) &&
            FusionCpuExtraValueType(api, inputs[1], &type1) &&
            FusionCpuExtraValueType(api, inputs[2], &output_type) &&
            (type0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
             type0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) &&
            type1 == type0 && output_type == type0 &&
            FusionCpuExtraIsConstant(api, inputs[2]) &&
            FusionCpuExtraValueType(api, outputs[0], &output_type) &&
            output_type == type0) {
            free((void*)inputs);
            free((void*)outputs);
            return true;
        }
    }
    free((void*)inputs);
    free((void*)outputs);
    return false;
}

OrtStatus* FusionCpuExtraCreate(const OrtApi* api, const OrtNode* fused_node,
                                const OrtNode* inner_node, void** state,
                                FusionCpuComputeFn* compute,
                                FusionCpuFreeFn* destroy) {
    FusionCpuExtraState* result;
    const OrtValueInfo** inputs = NULL;
    const OrtValueInfo** outputs = NULL;
    const char* op = NULL;
    const char* domain = NULL;
    int64_t value;
    size_t input_count = 0;
    size_t output_count = 0;
    OrtStatus* status;
    *state = NULL;
    status = api->Node_GetOperatorType(inner_node, &op);
    if (status != NULL) return status;
    status = api->Node_GetDomain(inner_node, &domain);
    if (status != NULL) return status;
    if (!FusionCpuExtraSupportsNode(api, inner_node)) return NULL;
    if (!FusionCpuExtraGetNodeValues(api, inner_node, true,
                                     &inputs, &input_count) ||
        !FusionCpuExtraGetNodeValues(api, inner_node, false,
                                     &outputs, &output_count)) {
        free((void*)inputs);
        free((void*)outputs);
        return api->CreateStatus(ORT_EP_FAIL,
                                 "FusionCPU extra kernel could not read node I/O.");
    }
    result = (FusionCpuExtraState*)calloc(1, sizeof(*result));
    if (result == NULL) {
        free((void*)inputs);
        free((void*)outputs);
        return api->CreateStatus(ORT_EP_FAIL,
                                 "FusionCPU extra kernel allocation failed.");
    }
    result->api = api;
    result->input0 = input_count > 0
        ? FusionCpuExtraFindValue(api, fused_node, inputs[0], true) : -1;
    result->input1 = input_count > 1
        ? FusionCpuExtraFindValue(api, fused_node, inputs[1], true) : -1;
    result->input2 = input_count > 2
        ? FusionCpuExtraFindValue(api, fused_node, inputs[2], true) : -1;
    result->output0 = output_count > 0
        ? FusionCpuExtraFindValue(api, fused_node, outputs[0], false) : -1;
    result->output1 = -1;
    if (strcmp(op, "SkipSimplifiedLayerNormalization") == 0) {
        result->kind = FUSION_CPU_EXTRA_RMS_NORM;
        result->flag0 = true;
        result->epsilon = 1e-5f;
        FusionCpuExtraReadFloat(api, inner_node, "epsilon", &result->epsilon);
        if (output_count > 3 && outputs[3] != NULL) {
            result->output1 =
                FusionCpuExtraFindValue(api, fused_node, outputs[3], false);
        }
    } else if (strcmp(op, "SimplifiedLayerNormalization") == 0) {
        result->kind = FUSION_CPU_EXTRA_RMS_NORM;
        result->flag0 = false;
        result->input2 = result->input1;
        result->epsilon = 1e-5f;
        FusionCpuExtraReadFloat(api, inner_node, "epsilon", &result->epsilon);
    } else if (strcmp(op, "Shape") == 0) {
        result->kind = FUSION_CPU_EXTRA_SHAPE;
        result->attr0 = 0;
        result->flag0 = FusionCpuExtraReadInt(api, inner_node, "end", &result->attr1);
        FusionCpuExtraReadInt(api, inner_node, "start", &result->attr0);
    } else if (strcmp(op, "Cast") == 0) {
        result->kind = FUSION_CPU_EXTRA_CAST;
        if (!FusionCpuExtraReadInt(api, inner_node, "to", &value)) {
            free((void*)inputs);
            free((void*)outputs);
            free(result);
            return api->CreateStatus(ORT_EP_FAIL, "Cast is missing its to attribute.");
        }
        result->type = (ONNXTensorElementDataType)value;
    } else if (strcmp(op, "Gather") == 0) {
        result->kind = FUSION_CPU_EXTRA_GATHER;
        result->attr0 = 0;
        FusionCpuExtraReadInt(api, inner_node, "axis", &result->attr0);
    } else if (strcmp(op, "ReduceSum") == 0) {
        result->kind = FUSION_CPU_EXTRA_REDUCE_SUM;
        result->flag0 = true;
        result->flag1 = false;
        if (FusionCpuExtraReadInt(api, inner_node, "keepdims", &value)) result->flag0 = value != 0;
        if (FusionCpuExtraReadInt(api, inner_node, "noop_with_empty_axes", &value)) result->flag1 = value != 0;
    } else if (strcmp(op, "Sub") == 0) {
        result->kind = FUSION_CPU_EXTRA_SUB;
    }
    free((void*)inputs);
    free((void*)outputs);
    if (result->output0 < 0 || result->input0 < 0 ||
        (result->kind == FUSION_CPU_EXTRA_RMS_NORM &&
         (result->input2 < 0 || (result->flag0 && result->input1 < 0))) ||
        ((result->kind == FUSION_CPU_EXTRA_GATHER ||
          result->kind == FUSION_CPU_EXTRA_SUB) && result->input1 < 0)) {
        free(result);
        return api->CreateStatus(
            ORT_EP_FAIL,
            "FusionCPU extra kernel could not map inner node I/O to fused node I/O.");
    }
    *state = result;
    *compute = FusionCpuExtraCompute;
    *destroy = FusionCpuExtraDestroy;
    return NULL;
}
