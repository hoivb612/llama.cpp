// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.
//
// dllmain.cpp
//
// The two C ABI entry points ORT looks up by name in every plugin EP library:
//   CreateEpFactories — produce one or more OrtEpFactory instances.
//   ReleaseEpFactory  — destroy a factory created above.

#include "fusion_cpu_ep_factory.h"

#include <onnxruntime_c_api.h>

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "version.h"

#include <new>

#if defined(_WIN32)
#define FUSION_CPU_EP_EXPORT extern "C" __declspec(dllexport)
#else
#define FUSION_CPU_EP_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr size_t kFactoriesProduced = 1;
}  // namespace

FUSION_CPU_EP_EXPORT OrtStatus* CreateEpFactories(const char* registered_name,
                                                  const OrtApiBase* ort_api_base,
                                                  const OrtLogger* default_logger,
                                                  OrtEpFactory** factories,
                                                  size_t max_factories,
                                                  size_t* num_factories) {
    if (!ort_api_base || !factories || !num_factories) {
        if (num_factories) *num_factories = 0;
        return nullptr;
    }

    const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
    if (!ort_api) {
        *num_factories = 0;
        return nullptr;
    }

    // Bind the C++ API wrapper's global Ort::Global<>::api_ — every TU here
    // includes onnxruntime_cxx_api.h with ORT_API_MANUAL_INIT.
    Ort::InitApi(ort_api);

    if (max_factories < kFactoriesProduced) {
        *num_factories = 0;
        return nullptr;
    }

    auto* factory = fusion_cpu_ep::CreateFusionCpuEpFactory(
        registered_name ? registered_name : FUSION_CPU_EP_NAME,
        ort_api,
        default_logger);
    if (!factory) {
        *num_factories = 0;
        return nullptr;
    }

    factories[0] = factory;
    *num_factories = kFactoriesProduced;
    return nullptr;
}

FUSION_CPU_EP_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
    if (factory) {
        fusion_cpu_ep::DestroyFusionCpuEpFactory(factory);
    }
    return nullptr;
}
