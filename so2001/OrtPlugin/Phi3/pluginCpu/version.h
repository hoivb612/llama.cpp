// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#pragma once

#define FUSION_CPU_EP_VERSION_MAJOR 0
#define FUSION_CPU_EP_VERSION_MINOR 0
#define FUSION_CPU_EP_VERSION_PATCH 1
#define FUSION_CPU_EP_VERSION_STRING "0.0.1-dev"

// Display / registration name. Distinct from the built-in "CPUExecutionProvider"
// so both can coexist in one session: FusionCPU claims a subset of ops (to beat
// MLAS via cross-op fusion) and the built-in CPU EP runs everything else, all in
// the same CPU memory with zero copies between them.
#define FUSION_CPU_EP_NAME "FusionCpuEP"
#define FUSION_CPU_EP_VENDOR "ONNX Runtime FusionCPU EP contributors"
// No IHV PCI vendor ID for a pure-CPU EP; 0 means "non-IHV-specific".
#define FUSION_CPU_EP_VENDOR_ID 0

// Minimum ONNX Runtime version we bind against (when the plugin EP API
// stabilised). Overridable at CMake configure time.
#ifndef FUSION_CPU_EP_MIN_ORT_VERSION
#define FUSION_CPU_EP_MIN_ORT_VERSION "1.24.4"
#endif
