// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#pragma once

#include <onnxruntime_c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

OrtEpFactory* CreateFusionCpuEpFactory(const char* registered_name,
                                       const OrtApi* ort_api,
                                       const OrtLogger* default_logger);
void DestroyFusionCpuEpFactory(OrtEpFactory* factory);

#ifdef __cplusplus
}
#endif
