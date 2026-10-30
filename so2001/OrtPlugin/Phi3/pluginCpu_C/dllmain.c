// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#include "fusion_cpu_ep_factory.h"
#include "version.h"

#include <stddef.h>

#if defined(_WIN32)
#define FUSION_CPU_EP_EXPORT __declspec(dllexport)
#else
#define FUSION_CPU_EP_EXPORT __attribute__((visibility("default")))
#endif

FUSION_CPU_EP_EXPORT OrtStatus* ORT_API_CALL CreateEpFactories(
    const char* registered_name, const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger, OrtEpFactory** factories,
    size_t max_factories, size_t* num_factories) {
    const OrtApi* ort_api;

    if (num_factories != NULL) *num_factories = 0;
    if (ort_api_base == NULL || factories == NULL || num_factories == NULL ||
        max_factories == 0) {
        return NULL;
    }
    ort_api = ort_api_base->GetApi(ORT_API_VERSION);
    if (ort_api == NULL) return NULL;

    factories[0] = CreateFusionCpuEpFactory(
        registered_name != NULL ? registered_name : FUSION_CPU_EP_NAME,
        ort_api, default_logger);
    if (factories[0] == NULL) return NULL;
    *num_factories = 1;
    return NULL;
}

FUSION_CPU_EP_EXPORT OrtStatus* ORT_API_CALL ReleaseEpFactory(OrtEpFactory* factory) {
    if (factory != NULL) DestroyFusionCpuEpFactory(factory);
    return NULL;
}
