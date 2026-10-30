// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#pragma once

#include <onnxruntime_c_api.h>

namespace fusion_cpu_ep {

// Construct a single OrtEpFactory that advertises a CPU OrtEpDevice.
// Ownership transfers to the caller; release via DestroyFusionCpuEpFactory().
//
// `registered_name` is the name ORT registered this library under.
// `ort_api` must be a non-null OrtApi pointer from OrtApiBase::GetApi().
// `default_logger` may be nullptr.
OrtEpFactory* CreateFusionCpuEpFactory(const char* registered_name,
                                       const OrtApi* ort_api,
                                       const OrtLogger* default_logger);

void DestroyFusionCpuEpFactory(OrtEpFactory* factory);

}  // namespace fusion_cpu_ep
