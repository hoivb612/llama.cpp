#include "fusion_cpu_ep.h"
#include "fusion_cpu_ep_sync.h"
#include "fusion_cpu_extra_ops.h"

#include "matmulnbits_avx512.h"

#include <intrin.h>
#include "fp16_utils.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

static volatile LONG64 g_fusion_kernel_sequence;

#if defined(_MSC_VER)
__declspec(thread) static const char* g_fusion_diagnostic_node;
__declspec(thread) static const char* g_fusion_diagnostic_op;
#else
static _Thread_local const char* g_fusion_diagnostic_node;
static _Thread_local const char* g_fusion_diagnostic_op;
#endif

static double FusionCpuElapsedMilliseconds(
    const LARGE_INTEGER* start, const LARGE_INTEGER* end) {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return (double)(end->QuadPart - start->QuadPart) * 1000.0 /
        (double)frequency.QuadPart;
}

void FusionCpuDiagnosticStatus(const OrtApi* api, const char* operation,
                               int input_index, const OrtStatus* status) {
    const char* message = status != NULL && api != NULL
        ? api->GetErrorMessage(status) : "no ORT status message";
    fprintf(stderr,
            "[FusionCpuEP][diagnostic] node='%s' op='%s' call='%s' input=%d: %s\n",
            g_fusion_diagnostic_node != NULL ? g_fusion_diagnostic_node : "<unknown>",
            g_fusion_diagnostic_op != NULL ? g_fusion_diagnostic_op : "<unknown>",
            operation != NULL ? operation : "<unknown>", input_index,
            message != NULL ? message : "<null>");
    fflush(stderr);
}

typedef struct FusionCpuMulState {
    const OrtApi* api;
} FusionCpuMulState;

typedef struct FusionCpuSigmoidState {
    const OrtApi* api;
} FusionCpuSigmoidState;

typedef struct FusionCpuSwiGluState {
    const OrtApi* api;
    int idx_g;
    int idx_u;
} FusionCpuSwiGluState;

typedef struct FusionCpuMatMulNBitsState {
    const OrtApi* api;
    int bits;
    int block_size;
    int idx_a;
    int idx_scale;
    int idx_b;
    int idx_scales;
    int idx_zp;
    bool prepack_ready;
    const uint8_t* prepack_b;
    int64_t pp_n_tiles;
    int64_t pp_k_blocks;
    int64_t pp_block_size;
    bool pp_int4;
    bool pp_compact;
    bool pp_compact_zp;
    bool pp_zp_itl;
    int8_t* pp_wq8;
    uint8_t* pp_wq4;
    float* pp_wscale;
    int32_t* pp_wsum;
    uint16_t* pp_wscale_h;
    int16_t* pp_wsum16;
    int16_t* pp_qsum16;
    int16_t* pp_zp16;
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
    int idx_seqlens_k;
    int idx_cos;
    int idx_sin;
    int idx_out;
    int idx_present_key;
    int idx_present_value;
    int64_t local_window;
    uint16_t* kcache;
    uint16_t* vcache;
    int64_t cache_len;
    int64_t cache_cap;
    int64_t cache_head_size;
    int cache_kv_heads;
    // fp16 (q4f16) support: persistent fp32-widened copies of the constant cos/sin
    // RoPE caches, populated once on the first fp16 Compute and reused across every
    // Run(). NULL on the fp32 path (raw pointers used directly).
    float* cos_f32;
    float* sin_f32;
    int64_t cos_f32_count;
    int64_t sin_f32_count;
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

FusionCpuKernelEntry* FusionCpuEp_FindKernel(FusionCpuEp* ep, const char* name) {
    FusionCpuKernelEntry* entry;
    if (ep == NULL || name == NULL) return NULL;
    entry = ep->kernels;
    while (entry != NULL && strcmp(entry->name, name) != 0) entry = entry->next;
    return entry;
}

OrtStatus* FusionCpuEp_AddKernel(FusionCpuEp* ep, const char* name,
                                 const char* op_type, void* state,
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
    entry->op_type = FusionCpuDuplicateString(op_type != NULL ? op_type : "<unknown>");
    if (entry->name == NULL || entry->op_type == NULL) {
        free(entry->name);
        free(entry->op_type);
        free(entry);
        return ep->ort_api->CreateStatus(ORT_EP_FAIL,
                                         "FusionCPU EP: out of memory copying kernel name.");
    }
    entry->state = state;
    entry->compute = compute;
    entry->destroy = destroy;
    entry->next = ep->kernels;
    ep->kernels = entry;
    return NULL;
}

static void FusionCpuFreeKernelEntries(FusionCpuEp* ep) {
    FusionCpuKernelEntry* entry = ep->kernels;
    while (entry != NULL) {
        FusionCpuKernelEntry* next = entry->next;
        entry->destroy(entry->state);
        free(entry->name);
        free(entry->op_type);
        free(entry);
        entry = next;
    }
    ep->kernels = NULL;
}

static const char* ORT_API_CALL FusionCpuEpGetName(const OrtEp* ep) {
    return ((const FusionCpuEp*)ep)->name;
}

//
// Generation/query of hardware features.
//

static bool g_FusionCpuHasAvx512f = 0;

bool FusionCpuHasAvx512f(void) {
    return g_FusionCpuHasAvx512f;
}

static bool FusionCpuAvx512fGen(void) {
    int info[4];
    unsigned __int64 xcr0;
    bool detected = 0;

    __cpuid(info, 0);
    if (info[0] >= 7) {
        __cpuidex(info, 1, 0);

        if ((info[2] & (1 << 27)) != 0 && (info[2] & (1 << 28)) != 0) {
            xcr0 = _xgetbv(0);
            if ((xcr0 & 0xE6u) == 0xE6u) {
                __cpuidex(info, 7, 0);
                detected = (info[1] & (1 << 16)) != 0;
            }
        }
    }

    return detected;
}

static bool g_FusionCpuHasAvx512Dq = 0;

static inline bool FusionCpuHasAvx512Dq(void) {
    return g_FusionCpuHasAvx512Dq;
}

static bool FusionCpuAvx512DqGen(void) {
    int info[4];
    bool detected = 0;

    if (FusionCpuHasAvx512f()) {
        __cpuidex(info, 7, 0);
        detected = (info[1] & (1 << 17)) != 0;
    }

    return detected;
}

static bool g_FusionCpuHasAvx512Vnni = 0;

static bool FusionCpuHasAvx512Vnni(void) {
    return g_FusionCpuHasAvx512Vnni;
}

static bool FusionCpuAvx512VnniGen(void) {
    int info[4];
    bool detected = 0;

    if (FusionCpuHasAvx512f()) {
        __cpuidex(info, 7, 0);
        detected = (info[2] & (1 << 11)) != 0;
    }

    return detected;
}

static bool g_FusionCpuSpinEnabled = 0;
static double g_FusionCpuSpinMs = 2.0; // default spin budget in milliseconds
static int64_t g_FusionCpuSpinTicks = 0; // spin budget in ticks (QPC units)

static bool FusionCpuSpinTicksGen(void) {
    const char* env_ms = getenv("ORT_FUSIONCPU_SPIN_BUDGET_MS");
    if (env_ms != NULL && env_ms[0] != '\0') {
        char* end = NULL;
        g_FusionCpuSpinMs = strtod(env_ms, &end);
        if ((end != env_ms) && (g_FusionCpuSpinMs >= 0.0)) {
            // Max is 100ms. Anything higher is likely a misconfiguration.
            if (g_FusionCpuSpinMs > 100.0) {
                g_FusionCpuSpinMs = 100.0;
            }

        } else {
            // "0" if number is not parsable or negative. This disables the spin entirely.
            g_FusionCpuSpinMs = 0.0;
        }
    }

    // Convert ms to ticks
    LARGE_INTEGER qpf;
    QueryPerformanceFrequency(&qpf);
    g_FusionCpuSpinTicks = (int64_t)((double)qpf.QuadPart * g_FusionCpuSpinMs / 1000.0);

    g_FusionCpuSpinEnabled = (g_FusionCpuSpinTicks > 0);

    return g_FusionCpuSpinEnabled;
}

const int64_t FusionCpuSpinTicks(void) {
    return g_FusionCpuSpinTicks;
}

//
// Generation/query of environment features.
//

#define gen_default_disabled(func, env) \
static bool g_ ## func = 0;             \
extern inline bool func(void) {         \
    return g_ ## func;                  \
}                                       \
static bool func ## Gen(void) {         \
    const char* value = getenv(# env);  \
    return ((value != NULL) && (value[0] == '1')); \
}

#define gen_default_enabled(func, env)  \
static bool g_ ## func = 0;             \
extern inline bool func(void) {         \
    return g_ ## func;                  \
}                                       \
static bool func ## Gen(void) {         \
    const char* value = getenv(# env);  \
    return ((value == NULL) || (value[0] != '0')); \
}

gen_default_disabled(FusionCpuForceScalarMatMul, ORT_FUSIONCPU_MATMULNBITS_SCALAR)
gen_default_enabled(FusionCpuMatMulVnniEnabled, ORT_FUSIONCPU_MATMULNBITS_VNNI)
gen_default_enabled(FusionCpuMatMulPrepackEnabled, ORT_FUSIONCPU_MATMULNBITS_PREPACK)
gen_default_enabled(FusionCpuMatMulNBitsEnabled, ORT_FUSIONCPU_MATMULNBITS)
gen_default_enabled(FusionCpuMatMulInt4TileEnabled, ORT_FUSIONCPU_MATMULNBITS_INT4TILE)
gen_default_enabled(FusionCpuMatMulCompactEnabled, ORT_FUSIONCPU_MATMULNBITS_INT4COMPACT)
gen_default_enabled(FusionCpuMatMulCompactZpEnabled, ORT_FUSIONCPU_MATMULNBITS_INT4COMPACTZP)
gen_default_enabled(FusionCpuMatMulMBlockEnabled, ORT_FUSIONCPU_MATMULNBITS_MBLOCK)
gen_default_enabled(FusionCpuMatMulInterleaveEnabled, ORT_FUSIONCPU_MATMULNBITS_INTERLEAVE)
gen_default_enabled(fusion_ccx_pin_enabled, ORT_FUSIONCPU_CCX_PIN)
gen_default_disabled(FusionCpuMatMulVerbose, ORT_FUSIONCPU_MATMULNBITS_VERBOSE)
gen_default_disabled(FusionCpuTraceEnabled, ORT_FUSIONCPU_TRACE)
gen_default_enabled(FusionCpuGqaEnabled, ORT_FUSIONCPU_GQA)
gen_default_enabled(FusionCpuGqaUseFusedPath, ORT_FUSIONCPU_GQA_FUSED)
gen_default_enabled(FusionCpuGqaFp16KvEnabled, ORT_FUSIONCPU_GQA_FP16KV)
gen_default_disabled(FusionCpuGqaOwnKvEnabled, ORT_FUSIONCPU_GQA_OWNKV)

//
// Generate environment/hardware configuration.
//

typedef bool (*fusion_gen_env_fn)(void);

typedef struct {
    fusion_gen_env_fn fn;
    bool *result;
} fusion_config_entry;

fusion_config_entry fusion_gen_env_table[] = {
    {FusionCpuAvx512fGen, &g_FusionCpuHasAvx512f},
    {FusionCpuAvx512DqGen, &g_FusionCpuHasAvx512Dq},
    {FusionCpuAvx512VnniGen, &g_FusionCpuHasAvx512Vnni},
    {FusionCpuForceScalarMatMulGen, &g_FusionCpuForceScalarMatMul},
    {FusionCpuMatMulVnniEnabledGen, &g_FusionCpuMatMulVnniEnabled},
    {FusionCpuMatMulPrepackEnabledGen, &g_FusionCpuMatMulPrepackEnabled},
    {FusionCpuMatMulNBitsEnabledGen, &g_FusionCpuMatMulNBitsEnabled},
    {FusionCpuMatMulInt4TileEnabledGen, &g_FusionCpuMatMulInt4TileEnabled},
    {FusionCpuMatMulCompactEnabledGen, &g_FusionCpuMatMulCompactEnabled},
    {FusionCpuMatMulCompactZpEnabledGen, &g_FusionCpuMatMulCompactZpEnabled},
    {FusionCpuMatMulMBlockEnabledGen, &g_FusionCpuMatMulMBlockEnabled},
    {FusionCpuMatMulInterleaveEnabledGen, &g_FusionCpuMatMulInterleaveEnabled},
    {fusion_ccx_pin_enabledGen, &g_fusion_ccx_pin_enabled},
    {FusionCpuMatMulVerboseGen, &g_FusionCpuMatMulVerbose},
    {FusionCpuTraceEnabledGen, &g_FusionCpuTraceEnabled},
    {FusionCpuGqaEnabledGen, &g_FusionCpuGqaEnabled},
    {FusionCpuGqaUseFusedPathGen, &g_FusionCpuGqaUseFusedPath},
    {FusionCpuGqaFp16KvEnabledGen, &g_FusionCpuGqaFp16KvEnabled},
    {FusionCpuGqaOwnKvEnabledGen, &g_FusionCpuGqaOwnKvEnabled},
    {FusionCpuSpinTicksGen, &g_FusionCpuSpinEnabled},
};

void FusionInitEP(void) {

    //
    // Initialize the environment and hardware configuration.
    //

    printf("generating environment/hardware configuation\n");

    for (uint32_t i = 0; i < ARRAYSIZE(fusion_gen_env_table); i += 1) {
        *fusion_gen_env_table[i].result = fusion_gen_env_table[i].fn();
    }

    printf("FusionCpuHasAvx512f %d\n", FusionCpuHasAvx512f());
    printf("FusionCpuHasAvx512Dq %d\n", FusionCpuHasAvx512Dq());
    printf("FusionCpuHasAvx512Vnni %d\n", FusionCpuHasAvx512Vnni());
    printf("FusionCpuForceScalarMatMul %d\n", FusionCpuForceScalarMatMul());
    printf("FusionCpuMatMulVnniEnabled %d\n", FusionCpuMatMulVnniEnabled());
    printf("FusionCpuMatMulPrepackEnabled %d\n", FusionCpuMatMulPrepackEnabled());
    printf("FusionCpuMatMulNBitsEnabled %d\n", FusionCpuMatMulNBitsEnabled());
    printf("FusionCpuMatMulInt4TileEnabled %d\n", FusionCpuMatMulInt4TileEnabled());
    printf("FusionCpuMatMulCompactEnabled %d\n", FusionCpuMatMulCompactEnabled());
    printf("FusionCpuMatMulCompactZpEnabled %d\n", FusionCpuMatMulCompactZpEnabled());
    printf("FusionCpuMatMulMBlockEnabled %d\n", FusionCpuMatMulMBlockEnabled());
    printf("FusionCpuMatMulInterleaveEnabled %d\n", FusionCpuMatMulInterleaveEnabled());
    printf("fusion_ccx_pin_enabled %d\n", fusion_ccx_pin_enabled());
    printf("FusionCpuMatMulVerbose %d\n", FusionCpuMatMulVerbose());
    printf("FusionCpuTraceEnabled %d\n", FusionCpuTraceEnabled());
    printf("FusionCpuGqaEnabled %d\n", FusionCpuGqaEnabled());
    printf("FusionCpuGqaUseFusedPath %d\n", FusionCpuGqaUseFusedPath());
    printf("FusionCpuGqaFp16KvEnabled %d\n", FusionCpuGqaFp16KvEnabled());
    printf("FusionCpuGqaOwnKvEnabled %d\n", FusionCpuGqaOwnKvEnabled());
    printf("FusionCpuSpinEnabled %d: %.1fms\n", g_FusionCpuSpinEnabled, g_FusionCpuSpinMs);

    //
    // Initilialize the thread pool.
    //
    // N.B. There is no return from this function if the thread pool cannot be
    //      allocated and initialized.
    //

    fusion_pool_create();
}

static OrtStatus* FusionCpuGetInput(const OrtApi* api, const OrtKernelContext* context,
                                    size_t index, const OrtValue** value,
                                    ONNXTensorElementDataType* type,
                                    const int64_t** shape, size_t* rank,
                                    const void** data) {
    OrtStatus* status = api->KernelContext_GetInput(context, index, value);
    if (status != NULL) {
        FusionCpuDiagnosticStatus(api, "KernelContext_GetInput", (int)index, status);
        return status;
    }
    if (*value == NULL) {
        status = api->CreateStatus(ORT_INVALID_ARGUMENT,
                                   "FusionCPU EP: required input is absent.");
        FusionCpuDiagnosticStatus(api, "KernelContext_GetInput returned NULL",
                                  (int)index, status);
        return status;
    }
    status = api->GetTensorElementTypeAndShapeDataReference(*value, type, shape, rank);
    if (status != NULL) {
        FusionCpuDiagnosticStatus(api, "GetTensorElementTypeAndShapeDataReference",
                                  (int)index, status);
        return status;
    }
    status = api->GetTensorData(*value, data);
    if (status != NULL) {
        FusionCpuDiagnosticStatus(api, "GetTensorData", (int)index, status);
    }
    return status;
}

static OrtStatus* FusionCpuGetOutput(const OrtApi* api, OrtKernelContext* context,
                                     size_t index, const int64_t* shape, size_t rank,
                                     OrtValue** value, void** data) {
    OrtStatus* status = api->KernelContext_GetOutput(context, index, shape, rank, value);
    if (status != NULL) {
        FusionCpuDiagnosticStatus(api, "KernelContext_GetOutput", (int)index, status);
        return status;
    }
    if (*value == NULL) {
        status = api->CreateStatus(ORT_INVALID_ARGUMENT,
                                   "FusionCPU EP: required output is absent.");
        FusionCpuDiagnosticStatus(api, "KernelContext_GetOutput returned NULL",
                                  (int)index, status);
        return status;
    }
    status = api->GetTensorMutableData(*value, data);
    if (status != NULL) {
        FusionCpuDiagnosticStatus(api, "GetTensorMutableData", (int)index, status);
    }
    return status;
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

#include "fusion_cpu_ep_optimized.inc"

static void ORT_API_CALL FusionCpuFreeSimpleState(void* state) {
    free(state);
}

static void ORT_API_CALL FusionCpuFreeGqaState(void* state) {
    FusionCpuFreeGqaOptimizedState(state);
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

static bool FusionCpuNodeMatches(const OrtApi* api, const OrtNode* node,
                                 const char* op_name, const char* domain_name) {
    const char* op = NULL;
    const char* domain = NULL;
    OrtStatus* status;
    status = api->Node_GetOperatorType(node, &op);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    status = api->Node_GetDomain(node, &domain);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    return op != NULL && strcmp(op, op_name) == 0 &&
           ((domain_name[0] == '\0' && (domain == NULL || domain[0] == '\0')) ||
            (domain != NULL && strcmp(domain, domain_name) == 0));
}

static bool FusionCpuGetValueTensorShape(const OrtApi* api, const OrtValueInfo* value,
                                          ONNXTensorElementDataType* element_type,
                                          int64_t** dimensions, size_t* rank) {
    const OrtTypeInfo* type_info = NULL;
    const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
    OrtStatus* status;
    int64_t* shape = NULL;
    size_t count = 0;
    *dimensions = NULL;
    *rank = 0;
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
    status = api->GetTensorElementType(tensor_info, element_type);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    status = api->GetDimensionsCount(tensor_info, &count);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    shape = (int64_t*)FusionCpuMallocArray(count, sizeof(*shape));
    if (shape == NULL) return false;
    if (count != 0) {
        status = api->GetDimensions(tensor_info, shape, count);
        if (status != NULL) {
             api->ReleaseStatus(status);
             free(shape);
             return false;
        }
    }
    *dimensions = shape;
    *rank = count;
    return true;
}

static bool FusionCpuValueIsFp16PerK(const OrtApi* api, const OrtValueInfo* value,
                                      int64_t k) {
    ONNXTensorElementDataType type;
    int64_t* shape = NULL;
    size_t rank = 0;
    size_t count = 1;
    size_t i;
    bool result = false;
    if (!FusionCpuGetValueTensorShape(api, value, &type, &shape, &rank) ||
        type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 || rank == 0 ||
        shape[rank - 1] != k) {
        free(shape);
        return false;
    }
    for (i = 0; i < rank; ++i) {
        if (shape[i] < 0 || (shape[i] != 0 && count > SIZE_MAX / (size_t)shape[i])) {
             free(shape);
             return false;
        }
        count *= (size_t)shape[i];
    }
    result = count == (size_t)k;
    free(shape);
    return result;
}

static bool FusionCpuMulHasFp32IdenticalInputs(const OrtApi* api, const OrtNode* node) {
    const OrtValueInfo** inputs = NULL;
    const OrtValueInfo** outputs = NULL;
    ONNXTensorElementDataType type_a;
    ONNXTensorElementDataType type_b;
    ONNXTensorElementDataType type_y;
    int64_t* shape_a = NULL;
    int64_t* shape_b = NULL;
    int64_t* shape_y = NULL;
    size_t rank_a = 0;
    size_t rank_b = 0;
    size_t rank_y = 0;
    size_t input_count = 0;
    size_t output_count = 0;
    bool result = false;
    if (!FusionCpuGetNodeIo(api, node, true, &inputs, &input_count) ||
        !FusionCpuGetNodeIo(api, node, false, &outputs, &output_count) ||
        input_count != 2 || output_count != 1 || inputs[0] == NULL ||
        inputs[1] == NULL || outputs[0] == NULL ||
        !FusionCpuGetValueTensorShape(api, inputs[0], &type_a, &shape_a, &rank_a) ||
        !FusionCpuGetValueTensorShape(api, inputs[1], &type_b, &shape_b, &rank_b) ||
        !FusionCpuGetValueTensorShape(api, outputs[0], &type_y, &shape_y, &rank_y)) {
        goto done;
    }
    result = type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
              type_b == type_a && type_y == type_a &&
              FusionCpuShapeEqual(shape_a, rank_a, shape_b, rank_b) &&
              FusionCpuShapeEqual(shape_a, rank_a, shape_y, rank_y);
done:
    free((void*)inputs);
    free((void*)outputs);
    free(shape_a);
    free(shape_b);
    free(shape_y);
    return result;
}

static bool FusionCpuIsAwqMulForMatMul(const OrtApi* api, const OrtNode* matmul,
                                        const OrtNode** mul_out) {
    const OrtValueInfo** mm_inputs = NULL;
    const OrtValueInfo** mul_inputs = NULL;
    const OrtValueInfo** mul_outputs = NULL;
    size_t mm_input_count = 0;
    size_t mul_input_count = 0;
    size_t mul_output_count = 0;
    size_t consumers = 0;
    bool graph_output = false;
    bool first_constant = false;
    bool second_constant = false;
    int64_t k = 0;
    int64_t bits = 4;
    int64_t block_size = 0;
    const OrtNode* mul = NULL;
    OrtStatus* status;
    bool result = false;

    *mul_out = NULL;
    if (!FusionCpuNodeMatches(api, matmul, "MatMulNBits", "com.microsoft") ||
        !FusionCpuGetNodeIo(api, matmul, true, &mm_inputs, &mm_input_count) ||
        mm_input_count < 3 || mm_input_count > 4 || mm_inputs[0] == NULL) {
        free((void*)mm_inputs);
        return false;
    }
    FusionCpuReadIntAttr(api, matmul, "bits", &bits);
    if (!FusionCpuReadIntAttr(api, matmul, "K", &k) ||
        !FusionCpuReadIntAttr(api, matmul, "block_size", &block_size) ||
        bits != 4 || k <= 0 || block_size <= 0 || k % block_size != 0) {
        goto done;
    }
    status = api->ValueInfo_GetValueProducer(mm_inputs[0], &mul, NULL);
    if (status != NULL) {
        api->ReleaseStatus(status);
        goto done;
    }
    if (mul == NULL || !FusionCpuNodeMatches(api, mul, "Mul", "") ||
        !FusionCpuGetNodeIo(api, mul, true, &mul_inputs, &mul_input_count) ||
        !FusionCpuGetNodeIo(api, mul, false, &mul_outputs, &mul_output_count) ||
        mul_input_count != 2 || mul_output_count != 1 || mul_outputs[0] == NULL) {
        goto done;
    }
    {
        const char* mm_name = NULL;
        const char* mul_name = NULL;
        status = api->GetValueInfoName(mm_inputs[0], &mm_name);
        if (status != NULL) {
            api->ReleaseStatus(status);
            goto done;
        }
        status = api->GetValueInfoName(mul_outputs[0], &mul_name);
        if (status != NULL) {
            api->ReleaseStatus(status);
            goto done;
        }
        if (mm_name == NULL || mul_name == NULL || strcmp(mm_name, mul_name) != 0) goto done;
    }
    status = api->ValueInfo_GetValueNumConsumers(mm_inputs[0], &consumers);
    if (status != NULL) {
        api->ReleaseStatus(status);
        goto done;
    }
    status = api->ValueInfo_IsGraphOutput(mm_inputs[0], &graph_output);
    if (status != NULL) {
        api->ReleaseStatus(status);
        goto done;
    }
    status = api->ValueInfo_IsConstantInitializer(mul_inputs[0], &first_constant);
    if (status != NULL) {
        api->ReleaseStatus(status);
        goto done;
    }
    status = api->ValueInfo_IsConstantInitializer(mul_inputs[1], &second_constant);
    if (status != NULL) {
        api->ReleaseStatus(status);
        goto done;
    }
    if (consumers == 1 && !graph_output && first_constant != second_constant &&
        FusionCpuValueIsFp16PerK(api, first_constant ? mul_inputs[0] : mul_inputs[1], k)) {
        *mul_out = mul;
        result = true;
    }
done:
    free((void*)mm_inputs);
    free((void*)mul_inputs);
    free((void*)mul_outputs);
    return result;
}

static bool FusionCpuValueHasType(const OrtApi* api, const OrtValueInfo* value,
                                  ONNXTensorElementDataType expected) {
    ONNXTensorElementDataType type;
    int64_t* shape = NULL;
    size_t rank = 0;
    bool result = FusionCpuGetValueTensorShape(api, value, &type, &shape, &rank) &&
                  type == expected;
    free(shape);
    return result;
}

// Returns the tensor element type of a value, or UNDEFINED on failure. Used by
// the GQA claim to require a single consistent io type (FLOAT or FLOAT16) across
// all fp inputs so the fp16 (q4f16) model is claimed too. 
static ONNXTensorElementDataType FusionCpuValueElemType(const OrtApi* api,
                                                        const OrtValueInfo* value) {
    ONNXTensorElementDataType type;
    int64_t* shape = NULL;
    size_t rank = 0;
    if (!FusionCpuGetValueTensorShape(api, value, &type, &shape, &rank)) {
        free(shape);
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    free(shape);
    return type;
}

static bool FusionCpuGqaCanClaim(const OrtApi* api, const OrtNode* node) {
    const OrtValueInfo** inputs = NULL;
    const OrtValueInfo** outputs = NULL;
    size_t input_count = 0;
    size_t output_count = 0;
    int64_t num_heads = 0;
    int64_t kv_heads = 0;
    int64_t do_rotary = 0;
    int64_t smooth = 0;
    int64_t qk_output = 0;
    int64_t kv_bits = 0;
    float softcap = 0.0f;
    size_t i;
    bool packed;
    bool first_constant = false;
    bool second_constant = false;
    OrtStatus* status;
    bool result = false;
    ONNXTensorElementDataType io_t;

    if (!FusionCpuGetNodeIo(api, node, true, &inputs, &input_count) ||
        !FusionCpuGetNodeIo(api, node, false, &outputs, &output_count) ||
        input_count < 7 || output_count != 3 ||
        outputs[0] == NULL || outputs[1] == NULL || outputs[2] == NULL ||
        inputs[0] == NULL || inputs[3] == NULL || inputs[4] == NULL ||
        inputs[5] == NULL || inputs[6] == NULL ||
        !FusionCpuReadIntAttr(api, node, "num_heads", &num_heads) ||
        !FusionCpuReadIntAttr(api, node, "kv_num_heads", &kv_heads) ||
        num_heads <= 0 || kv_heads <= 0 || num_heads % kv_heads != 0) {
        goto done;
    }
    // Accept FLOAT or FLOAT16 for the model I/O, but require a single consistent
    // io type across query/key/value/past/cos/sin (reject mixed dtype). The
    // internal GQA math is fp32; the Compute path widens fp16 inputs up front and
    // narrows fp16 outputs at the end.
    io_t = FusionCpuValueElemType(api, inputs[0]);
    if (io_t != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        io_t != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        goto done;
    }
    if (!FusionCpuValueHasType(api, inputs[3], io_t) ||
        !FusionCpuValueHasType(api, inputs[4], io_t)) {
        goto done;
    }
    packed = inputs[1] == NULL && inputs[2] == NULL;
    if (!packed && (inputs[1] == NULL || inputs[2] == NULL ||
                    !FusionCpuValueHasType(api, inputs[1], io_t) ||
                    !FusionCpuValueHasType(api, inputs[2], io_t))) {
        goto done;
    }
    for (i = 9; i < input_count; ++i) {
        if (inputs[i] != NULL) goto done;
    }
    if (FusionCpuReadIntAttr(api, node, "smooth_softmax", &smooth) && smooth == 1) goto done;
    if (FusionCpuReadIntAttr(api, node, "qk_output", &qk_output) && qk_output != 0) goto done;
    if (FusionCpuReadIntAttr(api, node, "kv_cache_bit_width", &kv_bits) && kv_bits != 0) goto done;
    if (FusionCpuReadFloatAttr(api, node, "softcap", &softcap) && softcap != 0.0f) goto done;
    FusionCpuReadIntAttr(api, node, "do_rotary", &do_rotary);
    if (do_rotary != 0) {
        if (input_count <= 8 || inputs[7] == NULL || inputs[8] == NULL ||
            !FusionCpuValueHasType(api, inputs[7], io_t) ||
            !FusionCpuValueHasType(api, inputs[8], io_t)) {
            goto done;
        }
        status = api->ValueInfo_IsConstantInitializer(inputs[7], &first_constant);
        if (status != NULL) {
            api->ReleaseStatus(status);
            goto done;
        }
        status = api->ValueInfo_IsConstantInitializer(inputs[8], &second_constant);
        if (status != NULL) {
            api->ReleaseStatus(status);
            goto done;
        }
        if (!first_constant || !second_constant) goto done;
    }
    result = true;
done:
    free((void*)inputs);
    free((void*)outputs);
    return result;
}

static bool FusionCpuShapeNumel(const int64_t* shape, size_t rank, size_t* count) {
    size_t product = 1;
    size_t i;
    for (i = 0; i < rank; ++i) {
        if (shape[i] < 0 || (shape[i] != 0 && product > SIZE_MAX / (size_t)shape[i])) {
            return false;
        }
        product *= (size_t)shape[i];
    }
    *count = product;
    return true;
}

static bool FusionCpuMatMulCanClaim(const OrtApi* api, const OrtNode* node) {
    const OrtValueInfo** inputs = NULL;
    const OrtValueInfo** outputs = NULL;
    ONNXTensorElementDataType a_type;
    ONNXTensorElementDataType b_type;
    ONNXTensorElementDataType scale_type;
    ONNXTensorElementDataType zp_type;
    int64_t* a_shape = NULL;
    int64_t* b_shape = NULL;
    int64_t* scale_shape = NULL;
    int64_t* zp_shape = NULL;
    size_t a_rank = 0;
    size_t b_rank = 0;
    size_t scale_rank = 0;
    size_t zp_rank = 0;
    size_t input_count = 0;
    size_t output_count = 0;
    size_t count;
    int64_t bits = 4;
    int64_t block_size = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t accuracy = 0;
    int64_t k_blocks;
    int64_t blob;
    bool b_constant = false;
    bool scale_constant = false;
    bool zp_constant = false;
    OrtStatus* status;
    bool result = false;

    if (!FusionCpuGetNodeIo(api, node, true, &inputs, &input_count) ||
        !FusionCpuGetNodeIo(api, node, false, &outputs, &output_count) ||
        (input_count != 3 && input_count != 4) || output_count != 1 ||
        inputs[0] == NULL || inputs[1] == NULL || inputs[2] == NULL ||
        outputs[0] == NULL || !FusionCpuReadIntAttr(api, node, "K", &k) ||
        !FusionCpuReadIntAttr(api, node, "N", &n) ||
        !FusionCpuReadIntAttr(api, node, "block_size", &block_size)) {
        goto done;
    }
    FusionCpuReadIntAttr(api, node, "bits", &bits);
    FusionCpuReadIntAttr(api, node, "accuracy_level", &accuracy);
    if (bits != 4 || block_size <= 0 || k <= 0 || n <= 0 || k % block_size != 0 ||
        (accuracy != 0 && accuracy != 4) ||
        !FusionCpuGetValueTensorShape(api, inputs[0], &a_type, &a_shape, &a_rank) ||
        !FusionCpuGetValueTensorShape(api, inputs[1], &b_type, &b_shape, &b_rank) ||
        !FusionCpuGetValueTensorShape(api, inputs[2], &scale_type, &scale_shape, &scale_rank) ||
        (a_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
         a_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
        b_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 || scale_type != a_type ||
        a_rank == 0 || a_shape[a_rank - 1] != k || b_rank != 3) {
        goto done;
    }
    status = api->ValueInfo_IsConstantInitializer(inputs[1], &b_constant);
    if (status != NULL) {
        api->ReleaseStatus(status);
        goto done;
    }
    status = api->ValueInfo_IsConstantInitializer(inputs[2], &scale_constant);
    if (status != NULL) {
        api->ReleaseStatus(status);
        goto done;
    }
    k_blocks = k / block_size;
    blob = block_size * bits / 8;
    if (!b_constant || !scale_constant || b_shape[0] != n || b_shape[1] != k_blocks ||
        b_shape[2] != blob || !FusionCpuShapeNumel(scale_shape, scale_rank, &count) ||
        count != (size_t)n * (size_t)k_blocks) {
        goto done;
    }
    if (input_count == 4) {
        if (inputs[3] == NULL ||
            !FusionCpuGetValueTensorShape(api, inputs[3], &zp_type, &zp_shape, &zp_rank) ||
            zp_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            goto done;
        }
        status = api->ValueInfo_IsConstantInitializer(inputs[3], &zp_constant);
        if (status != NULL) {
            api->ReleaseStatus(status);
            goto done;
        }
        if (!zp_constant || !FusionCpuShapeNumel(zp_shape, zp_rank, &count) ||
            count != (size_t)n * (size_t)((k_blocks * bits + 7) / 8)) {
            goto done;
        }
    }
    result = true;
done:
    free((void*)inputs);
    free((void*)outputs);
    free(a_shape);
    free(b_shape);
    free(scale_shape);
    free(zp_shape);
    return result;
}

static bool FusionCpuSameValueName(const OrtApi* api, const OrtValueInfo* left,
                                   const OrtValueInfo* right) {
    const char* left_name = NULL;
    const char* right_name = NULL;
    OrtStatus* status;
    if (left == NULL || right == NULL) return false;
    status = api->GetValueInfoName(left, &left_name);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    status = api->GetValueInfoName(right, &right_name);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    return left_name != NULL && right_name != NULL && strcmp(left_name, right_name) == 0;
}

static bool FusionCpuSwiGluValuePairIsSupported(const OrtApi* api,
                                                 const OrtValueInfo* gate,
                                                 const OrtValueInfo* up) {
    ONNXTensorElementDataType gate_type;
    ONNXTensorElementDataType up_type;
    int64_t* gate_shape = NULL;
    int64_t* up_shape = NULL;
    size_t gate_rank = 0;
    size_t up_rank = 0;
    bool result = false;
    if (FusionCpuGetValueTensorShape(api, gate, &gate_type, &gate_shape, &gate_rank) &&
        FusionCpuGetValueTensorShape(api, up, &up_type, &up_shape, &up_rank) &&
        gate_type == up_type &&
        (gate_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         gate_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) &&
        FusionCpuShapeEqual(gate_shape, gate_rank, up_shape, up_rank)) {
        result = true;
    }
    free(gate_shape);
    free(up_shape);
    return result;
}

static bool FusionCpuSwiGluIntermediateIsPrivate(const OrtApi* api,
                                                  const OrtValueInfo* value) {
    size_t consumers = 0;
    bool graph_output = false;
    OrtStatus* status;
    status = api->ValueInfo_GetValueNumConsumers(value, &consumers);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    status = api->ValueInfo_IsGraphOutput(value, &graph_output);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return false;
    }
    return consumers == 1 && !graph_output;
}

static bool FusionCpuIdentifySwiGlu(const OrtApi* api, const OrtNode* sigmoid,
                                    const OrtNode* first_mul, const OrtNode* second_mul,
                                    const OrtValueInfo** gate, const OrtValueInfo** up) {
    const OrtValueInfo** sig_inputs = NULL;
    const OrtValueInfo** sig_outputs = NULL;
    const OrtValueInfo** first_inputs = NULL;
    const OrtValueInfo** first_outputs = NULL;
    const OrtValueInfo** second_inputs = NULL;
    const OrtValueInfo** second_outputs = NULL;
    size_t sig_input_count = 0;
    size_t sig_output_count = 0;
    size_t first_input_count = 0;
    size_t first_output_count = 0;
    size_t second_input_count = 0;
    size_t second_output_count = 0;
    const OrtValueInfo* s;
    const OrtValueInfo* p;
    const OrtValueInfo* u;
    bool result = false;

    *gate = NULL;
    *up = NULL;
    if (!FusionCpuNodeMatches(api, sigmoid, "Sigmoid", "") ||
        !FusionCpuNodeMatches(api, first_mul, "Mul", "") ||
        !FusionCpuNodeMatches(api, second_mul, "Mul", "") ||
        !FusionCpuGetNodeIo(api, sigmoid, true, &sig_inputs, &sig_input_count) ||
        !FusionCpuGetNodeIo(api, sigmoid, false, &sig_outputs, &sig_output_count) ||
        !FusionCpuGetNodeIo(api, first_mul, true, &first_inputs, &first_input_count) ||
        !FusionCpuGetNodeIo(api, first_mul, false, &first_outputs, &first_output_count) ||
        !FusionCpuGetNodeIo(api, second_mul, true, &second_inputs, &second_input_count) ||
        !FusionCpuGetNodeIo(api, second_mul, false, &second_outputs, &second_output_count) ||
        sig_input_count != 1 || sig_output_count != 1 ||
        first_input_count != 2 || first_output_count != 1 ||
        second_input_count != 2 || second_output_count != 1 ||
        sig_inputs[0] == NULL || sig_outputs[0] == NULL ||
        first_inputs[0] == NULL || first_inputs[1] == NULL || first_outputs[0] == NULL ||
        second_inputs[0] == NULL || second_inputs[1] == NULL || second_outputs[0] == NULL ||
        !FusionCpuSwiGluIntermediateIsPrivate(api, sig_outputs[0])) {
        goto done;
    }
    s = sig_outputs[0];
    /* Association A: p = Mul(u, g); y = Mul(p, Sigmoid(g)). */
    if ((FusionCpuSameValueName(api, second_inputs[0], s) ||
         FusionCpuSameValueName(api, second_inputs[1], s)) &&
        FusionCpuSwiGluIntermediateIsPrivate(api, first_outputs[0]) &&
        (FusionCpuSameValueName(api, second_inputs[0], first_outputs[0]) ||
         FusionCpuSameValueName(api, second_inputs[1], first_outputs[0])) &&
        (FusionCpuSameValueName(api, first_inputs[0], sig_inputs[0]) ||
         FusionCpuSameValueName(api, first_inputs[1], sig_inputs[0]))) {
        u = FusionCpuSameValueName(api, first_inputs[0], sig_inputs[0])
            ? first_inputs[1] : first_inputs[0];
        if (FusionCpuSwiGluValuePairIsSupported(api, sig_inputs[0], u)) {
            *gate = sig_inputs[0];
            *up = u;
            result = true;
            goto done;
        }
    }
    /*
     * Association B: p = Mul(g, Sigmoid(g)); y = Mul(p, u). The first Mul
     * consumes the Sigmoid result and the gate, and the second consumes p.
     */
    if ((FusionCpuSameValueName(api, first_inputs[0], s) ||
         FusionCpuSameValueName(api, first_inputs[1], s)) &&
        (FusionCpuSameValueName(api, first_inputs[0], sig_inputs[0]) ||
         FusionCpuSameValueName(api, first_inputs[1], sig_inputs[0])) &&
        FusionCpuSwiGluIntermediateIsPrivate(api, first_outputs[0]) &&
        (FusionCpuSameValueName(api, second_inputs[0], first_outputs[0]) ||
         FusionCpuSameValueName(api, second_inputs[1], first_outputs[0]))) {
        p = first_outputs[0];
        u = FusionCpuSameValueName(api, second_inputs[0], p)
            ? second_inputs[1] : second_inputs[0];
        if (FusionCpuSwiGluValuePairIsSupported(api, sig_inputs[0], u)) {
            *gate = sig_inputs[0];
            *up = u;
            result = true;
        }
    }
done:
    free((void*)sig_inputs);
    free((void*)sig_outputs);
    free((void*)first_inputs);
    free((void*)first_outputs);
    free((void*)second_inputs);
    free((void*)second_outputs);
    return result;
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
        return (strcmp(op, "Mul") == 0 && input_count == 2 && output_count == 1 &&
                FusionCpuMulHasFp32IdenticalInputs(api, node)) ||
               (strcmp(op, "Sigmoid") == 0 && input_count == 1 && output_count == 1);
    }
    if (domain != NULL && strcmp(domain, "com.microsoft") == 0 &&
        strcmp(op, "MatMulNBits") == 0) {
        return FusionCpuMatMulNBitsEnabled() &&
               (input_count == 3 || input_count == 4) && output_count == 1 &&
               FusionCpuMatMulCanClaim(api, node);
    }
    if (domain != NULL && strcmp(domain, "com.microsoft") == 0 &&
        strcmp(op, "GroupQueryAttention") == 0) {
        return FusionCpuGqaEnabled() && output_count == 3 &&
               FusionCpuGqaCanClaim(api, node);
    }
    return false;
}

static OrtStatus* ORT_API_CALL FusionCpuEpGetCapability(OrtEp* ep, const OrtGraph* graph,
                                                         OrtEpGraphSupportInfo* support_info) {
    FusionCpuEp* self = (FusionCpuEp*)ep;
    const OrtNode** nodes = NULL;
    bool* claimed = NULL;
    size_t count = 0;
    size_t i;
    OrtStatus* status = self->ort_api->Graph_GetNumNodes(graph, &count);
    if (status != NULL) return status;
    if (count == 0) return NULL;
    nodes = (const OrtNode**)calloc(count, sizeof(*nodes));
    claimed = (bool*)calloc(count, sizeof(*claimed));
    if (nodes == NULL || claimed == NULL) {
        free((void*)nodes);
        free(claimed);
        return self->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU EP: out of memory reading graph.");
    }
    status = self->ort_api->Graph_GetNodes(graph, nodes, count);
    if (status != NULL) {
        free((void*)nodes);
        free(claimed);
        return status;
    }
    /* Claim foldable AWQ {Mul, MatMulNBits} pairs before singleton nodes. */
    for (i = 0; i < count; ++i) {
        const OrtNode* mul = NULL;
        size_t mul_index;
        OrtNodeFusionOptions options;
        if (!FusionCpuMatMulNBitsEnabled() ||
            !FusionCpuMatMulCanClaim(self->ort_api, nodes[i]) ||
            !FusionCpuIsAwqMulForMatMul(self->ort_api, nodes[i], &mul)) {
            continue;
        }
        for (mul_index = 0; mul_index < count && nodes[mul_index] != mul; ++mul_index) {
        }
        if (mul_index == count || claimed[i] || claimed[mul_index]) continue;
        memset(&options, 0, sizeof(options));
        options.ort_version_supported = ORT_API_VERSION;
        options.drop_constant_initializers = false;
        {
            const OrtNode* pair[2] = {mul, nodes[i]};
            status = self->ep_api->EpGraphSupportInfo_AddNodesToFuse(support_info, pair, 2, &options);
        }
        if (status != NULL) {
            free((void*)nodes);
            free(claimed);
            return status;
        }
        claimed[i] = true;
        claimed[mul_index] = true;
    }
    /* Claim complete SwiGLU triples before their singleton Sigmoid/Mul nodes. */
    for (i = 0; i < count; ++i) {
        size_t first;
        bool found = false;
        if (claimed[i] || !FusionCpuNodeMatches(self->ort_api, nodes[i], "Sigmoid", "")) continue;
        for (first = 0; first < count && !found; ++first) {
            size_t second;
            if (first == i || claimed[first]) continue;
            for (second = 0; second < count; ++second) {
                const OrtValueInfo* gate = NULL;
                const OrtValueInfo* up = NULL;
                OrtNodeFusionOptions options;
                const OrtNode* triple[3];
                if (second == i || second == first || claimed[second] ||
                    !FusionCpuIdentifySwiGlu(self->ort_api, nodes[i], nodes[first], nodes[second],
                                              &gate, &up)) {
                    continue;
                }
                (void)gate;
                (void)up;
                memset(&options, 0, sizeof(options));
                options.ort_version_supported = ORT_API_VERSION;
                options.drop_constant_initializers = false;
                triple[0] = nodes[i];
                triple[1] = nodes[first];
                triple[2] = nodes[second];
                status = self->ep_api->EpGraphSupportInfo_AddNodesToFuse(
                    support_info, triple, 3, &options);
                if (status != NULL) {
                    free((void*)nodes);
                    free(claimed);
                    return status;
                }
                claimed[i] = true;
                claimed[first] = true;
                claimed[second] = true;
                found = true;
                break;
            }
        }
    }
    for (i = 0; i < count; ++i) {
        OrtNodeFusionOptions options;
        if (claimed[i]) continue;
        if (!FusionCpuIsSupportedNode(self->ort_api, nodes[i])) continue;
        memset(&options, 0, sizeof(options));
        options.ort_version_supported = ORT_API_VERSION;
        options.drop_constant_initializers = false;
        status = self->ep_api->EpGraphSupportInfo_AddNodesToFuse(support_info, &nodes[i], 1,
                                                                 &options);
        if (status != NULL) {
            free((void*)nodes);
            free(claimed);
            return status;
        }
    }
    free((void*)nodes);
    free(claimed);
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
    result->idx_seqlens_k = input_count > 5
        ? FusionCpuFindInputForValue(ep->ort_api, fused, inputs[5]) : -1;
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
        FusionCpuFreeGqaOptimizedState(result);
        return ep->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU GQA could not map fused I/O.");
    }
    *state = result;
    return NULL;
}

static OrtStatus* FusionCpuCreateAwqMatMulState(FusionCpuEp* ep, const OrtNode* fused,
                                                 const OrtNode* first, const OrtNode* second,
                                                 void** state, const char** op_type) {
    const OrtNode* mul = NULL;
    const OrtNode* matmul = NULL;
    const OrtValueInfo** mul_inputs = NULL;
    const OrtValueInfo** mul_outputs = NULL;
    const OrtValueInfo** mm_inputs = NULL;
    size_t mul_input_count = 0;
    size_t mul_output_count = 0;
    size_t mm_input_count = 0;
    bool first_constant = false;
    bool second_constant = false;
    const OrtValueInfo* activation;
    const OrtValueInfo* awq_scale;
    FusionCpuMatMulNBitsState* result = NULL;
    int64_t bits = 4;
    int64_t block_size = 0;
    OrtStatus* status;

    if (FusionCpuNodeMatches(ep->ort_api, first, "Mul", "")) mul = first;
    if (FusionCpuNodeMatches(ep->ort_api, second, "Mul", "")) mul = second;
    if (FusionCpuNodeMatches(ep->ort_api, first, "MatMulNBits", "com.microsoft")) matmul = first;
    if (FusionCpuNodeMatches(ep->ort_api, second, "MatMulNBits", "com.microsoft")) matmul = second;
    if (mul == NULL || matmul == NULL ||
        !FusionCpuGetNodeIo(ep->ort_api, mul, true, &mul_inputs, &mul_input_count) ||
        !FusionCpuGetNodeIo(ep->ort_api, mul, false, &mul_outputs, &mul_output_count) ||
        !FusionCpuGetNodeIo(ep->ort_api, matmul, true, &mm_inputs, &mm_input_count) ||
        mul_input_count != 2 || mul_output_count != 1 || mm_input_count < 3 ||
        mm_input_count > 4 || mul_outputs[0] == NULL || mm_inputs[0] == NULL) {
        status = ep->ort_api->CreateStatus(ORT_EP_FAIL,
            "FusionCPU AWQ fold has an unsupported Mul/MatMulNBits topology.");
        goto done;
    }
    {
        const char* mul_name = NULL;
        const char* mm_name = NULL;
        status = ep->ort_api->GetValueInfoName(mul_outputs[0], &mul_name);
        if (status != NULL) goto done;
        status = ep->ort_api->GetValueInfoName(mm_inputs[0], &mm_name);
        if (status != NULL) goto done;
        if (mul_name == NULL || mm_name == NULL || strcmp(mul_name, mm_name) != 0) {
            status = ep->ort_api->CreateStatus(ORT_EP_FAIL,
                "FusionCPU AWQ fold Mul output does not feed MatMulNBits A.");
            goto done;
        }
    }
    status = ep->ort_api->ValueInfo_IsConstantInitializer(mul_inputs[0], &first_constant);
    if (status != NULL) goto done;
    status = ep->ort_api->ValueInfo_IsConstantInitializer(mul_inputs[1], &second_constant);
    if (status != NULL) goto done;
    if (first_constant == second_constant) {
        status = ep->ort_api->CreateStatus(ORT_EP_FAIL,
            "FusionCPU AWQ fold requires exactly one constant Mul operand.");
        goto done;
    }
    activation = first_constant ? mul_inputs[1] : mul_inputs[0];
    awq_scale = first_constant ? mul_inputs[0] : mul_inputs[1];
    FusionCpuReadIntAttr(ep->ort_api, matmul, "bits", &bits);
    if (!FusionCpuReadIntAttr(ep->ort_api, matmul, "block_size", &block_size) ||
        bits != 4 || block_size <= 0 || block_size > INT_MAX) {
        status = ep->ort_api->CreateStatus(ORT_EP_FAIL,
            "FusionCPU AWQ MatMulNBits attributes are unsupported.");
        goto done;
    }
    result = (FusionCpuMatMulNBitsState*)calloc(1, sizeof(*result));
    if (result == NULL) {
        status = ep->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU AWQ state allocation failed.");
        goto done;
    }
    result->api = ep->ort_api;
    result->bits = (int)bits;
    result->block_size = (int)block_size;
    result->idx_a = FusionCpuFindInputForValue(ep->ort_api, fused, activation);
    result->idx_scale = FusionCpuFindInputForValue(ep->ort_api, fused, awq_scale);
    result->idx_b = FusionCpuFindInputForValue(ep->ort_api, fused, mm_inputs[1]);
    result->idx_scales = FusionCpuFindInputForValue(ep->ort_api, fused, mm_inputs[2]);
    result->idx_zp = mm_input_count == 4 && mm_inputs[3] != NULL
        ? FusionCpuFindInputForValue(ep->ort_api, fused, mm_inputs[3]) : -1;
    if (result->idx_a < 0 || result->idx_scale < 0 || result->idx_b < 0 ||
        result->idx_scales < 0 || (mm_input_count == 4 && result->idx_zp < 0)) {
        FusionCpuFreeMatMulState(result);
        result = NULL;
        status = ep->ort_api->CreateStatus(ORT_EP_FAIL,
            "FusionCPU AWQ fold could not map fused-node inputs by name.");
        goto done;
    }
    *state = result;
    *op_type = "MatMulNBits";
    status = NULL;
done:
    if (status != NULL && result != NULL) FusionCpuFreeMatMulState(result);
    free((void*)mul_inputs);
    free((void*)mul_outputs);
    free((void*)mm_inputs);
    return status;
}

static OrtStatus* FusionCpuCreateSwiGluState(FusionCpuEp* ep, const OrtNode* fused,
                                             const OrtNode* const* nodes, size_t count,
                                             void** state, const char** op_type) {
    size_t sig_index;
    FusionCpuSwiGluState* result = NULL;
    if (count != 3) {
        return ep->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU SwiGLU partition must contain 3 nodes.");
    }
    for (sig_index = 0; sig_index < count; ++sig_index) {
        size_t first;
        if (!FusionCpuNodeMatches(ep->ort_api, nodes[sig_index], "Sigmoid", "")) continue;
        for (first = 0; first < count; ++first) {
            size_t second;
            if (first == sig_index) continue;
            for (second = 0; second < count; ++second) {
                const OrtValueInfo* gate = NULL;
                const OrtValueInfo* up = NULL;
                if (second == sig_index || second == first ||
                    !FusionCpuIdentifySwiGlu(ep->ort_api, nodes[sig_index], nodes[first],
                                              nodes[second], &gate, &up)) {
                    continue;
                }
                result = (FusionCpuSwiGluState*)calloc(1, sizeof(*result));
                if (result == NULL) {
                    return ep->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU SwiGLU allocation failed.");
                }
                result->api = ep->ort_api;
                result->idx_g = FusionCpuFindInputForValue(ep->ort_api, fused, gate);
                result->idx_u = FusionCpuFindInputForValue(ep->ort_api, fused, up);
                if (result->idx_g < 0 || result->idx_u < 0 || result->idx_g == result->idx_u) {
                    free(result);
                    return ep->ort_api->CreateStatus(ORT_EP_FAIL,
                        "FusionCPU SwiGLU could not map fused-node inputs by name.");
                }
                *state = result;
                *op_type = "SwiGlu";
                return NULL;
            }
        }
    }
    return ep->ort_api->CreateStatus(ORT_EP_FAIL,
        "FusionCPU fused 3-node partition does not match SwiGLU topology.");
}

static OrtStatus* FusionCpuCreateKernel(FusionCpuEp* ep, const OrtNode* fused,
                                        const OrtNode* inner, void** state,
                                        const char** op_type,
                                        FusionCpuComputeFn* compute,
                                        FusionCpuFreeFn* destroy) {
    const char* op = NULL;
    const char* domain = NULL;
    OrtStatus* status = ep->ort_api->Node_GetOperatorType(inner, &op);
    if (status != NULL) return status;
    *op_type = op;
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
        result->idx_scale = -1;
        result->idx_b = 1;
        result->idx_scales = 2;
        result->idx_zp = input_count >= 4 ? 3 : -1;
        *state = result;
        *compute = FusionCpuMatMulNBitsCompute;
        *destroy = FusionCpuFreeMatMulState;
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
    FusionCpuNodeComputeInfo* self = (FusionCpuNodeComputeInfo*)info;
    FusionCpuKernelEntry* entry = (FusionCpuKernelEntry*)state;
    OrtStatus* status;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    LONG64 sequence = 0;
    g_fusion_diagnostic_node = entry->name;
    g_fusion_diagnostic_op = entry->op_type;
    if (FusionCpuTraceEnabled()) {
        sequence = InterlockedIncrement64(&g_fusion_kernel_sequence);
        QueryPerformanceCounter(&start);
        fprintf(stderr,
                "[FusionCpuEP][trace] kernel #%lld BEGIN node='%s' op='%s'\n",
                (long long)sequence, entry->name, entry->op_type);
        fflush(stderr);
    }
    status = entry->compute(entry->state, context);
    if (FusionCpuTraceEnabled()) {
        QueryPerformanceCounter(&end);
        fprintf(stderr,
                "[FusionCpuEP][trace] kernel #%lld END node='%s' op='%s' "
                "status=%s elapsed=%.3f ms\n",
                (long long)sequence, entry->name, entry->op_type,
                status == NULL ? "OK" : "ERROR",
                FusionCpuElapsedMilliseconds(&start, &end));
        fflush(stderr);
    }
    if (status != NULL) {
        FusionCpuDiagnosticStatus(self->ep->ort_api, "kernel compute", -1, status);
    }
    g_fusion_diagnostic_node = NULL;
    g_fusion_diagnostic_op = NULL;
    return status;
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
        const char* op_type = NULL;
        void* state = NULL;
        FusionCpuComputeFn compute = NULL;
        FusionCpuFreeFn destroy = NULL;
        FusionCpuNodeComputeInfo* info;
        OrtStatus* status = self->ort_api->Graph_GetNumNodes(graphs[i], &node_count);
        if (status != NULL) return status;
        if (node_count != 1 && node_count != 2 && node_count != 3) {
            return self->ort_api->CreateStatus(ORT_EP_FAIL,
                                                "FusionCPU EP received an unsupported fused partition size.");
        }
        nodes = (const OrtNode**)calloc(node_count, sizeof(*nodes));
        if (nodes == NULL) return self->ort_api->CreateStatus(ORT_EP_FAIL, "FusionCPU EP allocation failed.");
        status = self->ort_api->Graph_GetNodes(graphs[i], nodes, node_count);
        if (status == NULL) status = self->ort_api->Node_GetName(fused_nodes[i], &name);
        if (status == NULL && node_count == 1) {
            status = FusionCpuCreateKernel(self, fused_nodes[i], nodes[0],
                                           &state, &op_type, &compute, &destroy);
        } else if (status == NULL && node_count == 2) {
            status = FusionCpuCreateAwqMatMulState(self, fused_nodes[i], nodes[0], nodes[1],
                                                    &state, &op_type);
            if (status == NULL) {
                compute = FusionCpuMatMulNBitsCompute;
                destroy = FusionCpuFreeMatMulState;
            }
        } else if (status == NULL) {
            status = FusionCpuCreateSwiGluState(self, fused_nodes[i], nodes, node_count,
                                                 &state, &op_type);
            if (status == NULL) {
                compute = FusionCpuSwiGluCompute;
                destroy = FusionCpuFreeSimpleState;
            }
        }
        free((void*)nodes);
        if (status != NULL) return status;
        status = FusionCpuEp_AddKernel(self, name != NULL ? name : "",
                                       op_type, state, compute, destroy);
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
    result->base.ort_version_supported = ORT_API_VERSION;
    result->base.GetName = FusionCpuEpGetName;
    result->base.GetCapability = FusionCpuEpGetCapability;
    result->base.Compile = FusionCpuEpCompile;
    result->base.ReleaseNodeComputeInfos = FusionCpuEpReleaseInfos;
    result->base.Sync = FusionCpuEpSync;

    //
    // Initialize execution provider.
    //

    FusionInitEP();

    return result;
}

void FusionCpuEp_Destroy(FusionCpuEp* ep) {
    if (ep == NULL) return;
    FusionCpuFreeKernelEntries(ep);
    free(ep->name);
    free(ep);
}
