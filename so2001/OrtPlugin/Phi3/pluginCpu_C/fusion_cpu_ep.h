// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#pragma once

#include <stdbool.h>
#include <stddef.h>

#include <onnxruntime_c_api.h>

#include "fusion_cpu_ep_sync.h"

typedef struct FusionCpuEp FusionCpuEp;
typedef struct FusionCpuKernelEntry FusionCpuKernelEntry;

typedef OrtStatus* (ORT_API_CALL* FusionCpuComputeFn)(void* state,
                                                       OrtKernelContext* context);
typedef void (ORT_API_CALL* FusionCpuFreeFn)(void* state);

struct FusionCpuKernelEntry {
    char* name;
    char* op_type;
    void* state;
    FusionCpuComputeFn compute;
    FusionCpuFreeFn destroy;
    FusionCpuKernelEntry* next;
};

struct FusionCpuEp {
    OrtEp base; /* Must be first: ORT owns this C ABI object. */
    const OrtApi* ort_api;
    const OrtEpApi* ep_api;
    const OrtLogger* logger;
    char* name;
    FusionCpuKernelEntry* kernels;
};

FusionCpuEp* FusionCpuEp_Create(const OrtApi* ort_api, const OrtEpApi* ep_api,
                                const OrtLogger* logger, const char* name);
void FusionCpuEp_Destroy(FusionCpuEp* ep);

OrtStatus* FusionCpuEp_AddKernel(FusionCpuEp* ep, const char* name,
                                 const char* op_type, void* state,
                                 FusionCpuComputeFn compute, FusionCpuFreeFn destroy);
FusionCpuKernelEntry* FusionCpuEp_FindKernel(FusionCpuEp* ep, const char* name);

void FusionCpuDiagnosticStatus(const OrtApi* api, const char* operation,
                               int input_index, const OrtStatus* status);

bool fusion_ccx_pin_enabled(void);

bool FusionCpuMatMulVerbose(void);

bool FusionCpuTraceEnabled(void);

bool FusionCpuHasAvx512f(void);

const int64_t FusionCpuSpinTicks(void);