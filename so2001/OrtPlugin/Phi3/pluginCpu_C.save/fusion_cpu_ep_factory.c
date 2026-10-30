// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#include "fusion_cpu_ep_factory.h"
#include "fusion_cpu_ep.h"
#include "version.h"

#include <stdlib.h>
#include <string.h>

typedef struct FusionCpuEpFactory {
    OrtEpFactory base; /* Must be first. */
    const OrtApi* ort_api;
    const OrtEpApi* ep_api;
    const OrtLogger* logger;
    char* registered_name;
} FusionCpuEpFactory;

static FusionCpuEpFactory* FusionCpuFactorySelf(OrtEpFactory* factory) {
    return (FusionCpuEpFactory*)factory;
}

static const FusionCpuEpFactory* FusionCpuFactoryConstSelf(const OrtEpFactory* factory) {
    return (const FusionCpuEpFactory*)factory;
}

static char* FusionCpuDupString(const char* source) {
    size_t len;
    char* result;
    if (source == NULL) return NULL;
    len = strlen(source) + 1;
    result = (char*)malloc(len);
    if (result != NULL) memcpy(result, source, len);
    return result;
}

static const char* ORT_API_CALL FusionCpuFactoryGetName(const OrtEpFactory* factory) {
    return FusionCpuFactoryConstSelf(factory)->registered_name;
}

static const char* ORT_API_CALL FusionCpuFactoryGetVendor(const OrtEpFactory* factory) {
    (void)factory;
    return FUSION_CPU_EP_VENDOR;
}

static uint32_t ORT_API_CALL FusionCpuFactoryGetVendorId(const OrtEpFactory* factory) {
    (void)factory;
    return FUSION_CPU_EP_VENDOR_ID;
}

static const char* ORT_API_CALL FusionCpuFactoryGetVersion(const OrtEpFactory* factory) {
    (void)factory;
    return FUSION_CPU_EP_VERSION_STRING;
}

static OrtStatus* ORT_API_CALL FusionCpuFactoryGetSupportedDevices(
    OrtEpFactory* factory, const OrtHardwareDevice* const* devices, size_t num_devices,
    OrtEpDevice** ep_devices, size_t max_ep_devices, size_t* num_ep_devices) {
    FusionCpuEpFactory* self = FusionCpuFactorySelf(factory);
    size_t i;

    if (num_ep_devices == NULL) {
        return self->ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                           "FusionCPU EP: num_ep_devices is null.");
    }
    *num_ep_devices = 0;
    if (devices == NULL || ep_devices == NULL) return NULL;

    for (i = 0; i < num_devices && *num_ep_devices < max_ep_devices; ++i) {
        OrtStatus* status;
        OrtEpDevice* device = NULL;
        if (devices[i] == NULL ||
            self->ort_api->HardwareDevice_Type(devices[i]) != OrtHardwareDeviceType_CPU) {
            continue;
        }
        status = self->ep_api->CreateEpDevice(factory, devices[i], NULL, NULL, &device);
        if (status != NULL) return status;
        ep_devices[(*num_ep_devices)++] = device;
    }
    return NULL;
}

static OrtStatus* ORT_API_CALL FusionCpuFactoryCreateEp(
    OrtEpFactory* factory, const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* ep_metadata, size_t num_devices,
    const OrtSessionOptions* session_options, const OrtLogger* logger, OrtEp** ep) {
    FusionCpuEpFactory* self = FusionCpuFactorySelf(factory);
    FusionCpuEp* result;
    (void)devices;
    (void)ep_metadata;
    (void)session_options;

    if (ep == NULL) {
        return self->ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                           "FusionCPU EP CreateEp: output is null.");
    }
    *ep = NULL;
    if (num_devices == 0) {
        return self->ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                           "FusionCPU EP CreateEp: expected a CPU device.");
    }
    result = FusionCpuEp_Create(self->ort_api, self->ep_api,
                                logger != NULL ? logger : self->logger,
                                self->registered_name);
    if (result == NULL) {
        return self->ort_api->CreateStatus(ORT_EP_FAIL,
                                           "FusionCPU EP CreateEp: out of memory.");
    }
    *ep = &result->base;
    return NULL;
}

static void ORT_API_CALL FusionCpuFactoryReleaseEp(OrtEpFactory* factory, OrtEp* ep) {
    (void)factory;
    FusionCpuEp_Destroy((FusionCpuEp*)ep);
}

static OrtStatus* ORT_API_CALL FusionCpuFactoryCreateAllocator(
    OrtEpFactory* factory, const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* allocator_options, OrtAllocator** allocator) {
    FusionCpuEpFactory* self = FusionCpuFactorySelf(factory);
    (void)memory_info;
    (void)allocator_options;
    if (allocator != NULL) *allocator = NULL;
    return self->ort_api->CreateStatus(
        ORT_INVALID_ARGUMENT,
        "FusionCPU EP uses ORT's default CPU allocator and has no custom allocator.");
}

static void ORT_API_CALL FusionCpuFactoryReleaseAllocator(OrtEpFactory* factory,
                                                           OrtAllocator* allocator) {
    (void)factory;
    (void)allocator;
}

static OrtStatus* ORT_API_CALL FusionCpuFactoryCreateDataTransfer(
    OrtEpFactory* factory, OrtDataTransferImpl** data_transfer) {
    (void)factory;
    if (data_transfer != NULL) *data_transfer = NULL;
    return NULL;
}

static bool ORT_API_CALL FusionCpuFactoryIsStreamAware(const OrtEpFactory* factory) {
    (void)factory;
    return false;
}

static OrtStatus* ORT_API_CALL FusionCpuFactoryCreateSyncStream(
    OrtEpFactory* factory, const OrtMemoryDevice* memory_device,
    const OrtKeyValuePairs* options, OrtSyncStreamImpl** stream) {
    FusionCpuEpFactory* self = FusionCpuFactorySelf(factory);
    (void)memory_device;
    (void)options;
    if (stream != NULL) *stream = NULL;
    return self->ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "FusionCPU EP has no synchronization stream.");
}

OrtEpFactory* CreateFusionCpuEpFactory(const char* registered_name,
                                       const OrtApi* ort_api,
                                       const OrtLogger* default_logger) {
    FusionCpuEpFactory* result;
    if (ort_api == NULL || ort_api->GetEpApi == NULL || ort_api->GetEpApi() == NULL) return NULL;

    result = (FusionCpuEpFactory*)calloc(1, sizeof(*result));
    if (result == NULL) return NULL;
    result->registered_name = FusionCpuDupString(
        registered_name != NULL ? registered_name : FUSION_CPU_EP_NAME);
    if (result->registered_name == NULL) {
        free(result);
        return NULL;
    }
    result->ort_api = ort_api;
    result->ep_api = ort_api->GetEpApi();
    result->logger = default_logger;

    result->base.ort_version_supported = ORT_API_VERSION;
    result->base.GetName = FusionCpuFactoryGetName;
    result->base.GetVendor = FusionCpuFactoryGetVendor;
    result->base.GetVendorId = FusionCpuFactoryGetVendorId;
    result->base.GetVersion = FusionCpuFactoryGetVersion;
    result->base.GetSupportedDevices = FusionCpuFactoryGetSupportedDevices;
    result->base.CreateEp = FusionCpuFactoryCreateEp;
    result->base.ReleaseEp = FusionCpuFactoryReleaseEp;
    result->base.CreateAllocator = FusionCpuFactoryCreateAllocator;
    result->base.ReleaseAllocator = FusionCpuFactoryReleaseAllocator;
    result->base.CreateDataTransfer = FusionCpuFactoryCreateDataTransfer;
    result->base.IsStreamAware = FusionCpuFactoryIsStreamAware;
    result->base.CreateSyncStreamForDevice = FusionCpuFactoryCreateSyncStream;
    return &result->base;
}

void DestroyFusionCpuEpFactory(OrtEpFactory* factory) {
    FusionCpuEpFactory* self;
    if (factory == NULL) return;
    self = FusionCpuFactorySelf(factory);
    free(self->registered_name);
    free(self);
}
