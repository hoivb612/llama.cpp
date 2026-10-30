// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#pragma once

#include <stdbool.h>

#include "fusion_cpu_ep.h"

#ifdef __cplusplus
extern "C" {
#endif

bool FusionCpuExtraSupportsNode(const OrtApi* api, const OrtNode* node);
OrtStatus* FusionCpuExtraCreate(const OrtApi* api, const OrtNode* fused_node,
                                const OrtNode* inner_node, void** state,
                                FusionCpuComputeFn* compute,
                                FusionCpuFreeFn* destroy);

#ifdef __cplusplus
}
#endif

