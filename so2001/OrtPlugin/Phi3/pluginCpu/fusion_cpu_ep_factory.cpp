// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.
//
// fusion_cpu_ep_factory.cpp
//
// FusionCPU OrtEpFactory. A pure-CPU plugin EP, modelled on ORT's own
// onnxruntime/core/session/plugin_ep/ep_factory_cpu.cc:
//   * GetSupportedDevices matches OrtHardwareDeviceType_CPU and emits one
//     OrtEpDevice via CreateEpDevice(factory, hw, nullptr, nullptr, ...).
//   * NO EpDevice_AddAllocatorInfo, NO CreateAllocator, NO CreateDataTransfer,
//     NO streams. The EP's outputs therefore live in plain CPU memory — the
//     SAME memory the built-in CPU EP uses — so nodes we claim and nodes that
//     fall through to the built-in CPU EP exchange tensors with zero copies and
//     no inserted Memcpy nodes (ORT's MemcpyTransformer only touches non-CPU
//     providers).
//   * CreateEp constructs a per-session FusionCpuEp (no device handle).

#include "fusion_cpu_ep_factory.h"

#include "version.h"
#include "fusion_cpu_ep.h"

#include <cstring>
#include <memory>
#include <new>
#include <string>

namespace fusion_cpu_ep {
namespace {

// Concrete factory. First member MUST be OrtEpFactory by value so we can cast
// the C-ABI `this_ptr` back to FusionCpuEpFactory* in every trampoline.
struct FusionCpuEpFactory {
    OrtEpFactory     base;            // MUST be first
    const OrtApi*    ort_api;
    const OrtEpApi*  ep_api;
    const OrtLogger* logger;
    std::string      registered_name;
    std::string      version_str;
};

FusionCpuEpFactory* Self(OrtEpFactory* p) {
    return reinterpret_cast<FusionCpuEpFactory*>(p);
}
const FusionCpuEpFactory* Self(const OrtEpFactory* p) {
    return reinterpret_cast<const FusionCpuEpFactory*>(p);
}

// --- OrtEpFactory v-table -------------------------------------------------

const char* ORT_API_CALL GetName(const OrtEpFactory* this_ptr) noexcept {
    return Self(this_ptr)->registered_name.c_str();
}

const char* ORT_API_CALL GetVendor(const OrtEpFactory* /*this_ptr*/) noexcept {
    return FUSION_CPU_EP_VENDOR;
}

uint32_t ORT_API_CALL GetVendorId(const OrtEpFactory* /*this_ptr*/) noexcept {
    return FUSION_CPU_EP_VENDOR_ID;
}

const char* ORT_API_CALL GetVersion(const OrtEpFactory* this_ptr) noexcept {
    return Self(this_ptr)->version_str.c_str();
}

OrtStatus* ORT_API_CALL GetSupportedDevices(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* hw_devices,
                                            size_t num_hw_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* num_ep_devices) noexcept {
    auto& self = *Self(this_ptr);
    *num_ep_devices = 0;

    for (size_t i = 0; i < num_hw_devices && *num_ep_devices < max_ep_devices; ++i) {
        const OrtHardwareDevice* hw = hw_devices[i];
        if (self.ort_api->HardwareDevice_Type(hw) != OrtHardwareDeviceType_CPU) {
            continue;
        }

        // CPU EP: no ep_metadata / ep_options, and (crucially) no allocator
        // info registration. ORT then treats our device as plain CPU memory.
        OrtEpDevice* ep_device = nullptr;
        OrtStatus* status = self.ep_api->CreateEpDevice(
            this_ptr, hw, /*ep_metadata=*/nullptr, /*ep_options=*/nullptr, &ep_device);
        if (status != nullptr) {
            return status;
        }

        ep_devices[(*num_ep_devices)++] = ep_device;
    }

    return nullptr;
}

OrtStatus* ORT_API_CALL CreateEp(OrtEpFactory* this_ptr,
                                 const OrtHardwareDevice* const* /*devices*/,
                                 const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 size_t num_devices,
                                 const OrtSessionOptions* /*session_options*/,
                                 const OrtLogger* logger,
                                 OrtEp** ep) noexcept {
    auto& self = *Self(this_ptr);
    *ep = nullptr;
    if (num_devices == 0) {
        return self.ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
            "FusionCPU EP CreateEp: expected at least one device.");
    }
    const OrtLogger* session_logger = logger ? logger : self.logger;
    *ep = new (std::nothrow) FusionCpuEp(self.ort_api, self.ep_api, session_logger,
                                         self.registered_name);
    if (!*ep) {
        return self.ort_api->CreateStatus(ORT_EP_FAIL,
            "FusionCPU EP CreateEp: out of memory constructing FusionCpuEp.");
    }
    return nullptr;
}

void ORT_API_CALL ReleaseEp(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
    delete static_cast<FusionCpuEp*>(ep);
}

// ORT calls CreateDataTransfer on every factory unconditionally during plugin
// registration (environment.cc: CreateDataTransferForFactory). A pure-CPU EP
// needs no custom transfer — return a null impl so ORT uses its default copy.
OrtStatus* ORT_API_CALL CreateDataTransfer(OrtEpFactory* /*this_ptr*/,
                                           OrtDataTransferImpl** data_transfer) noexcept {
    *data_transfer = nullptr;
    return nullptr;
}

// We never register a stream; ORT may still query this. CPU EP is not stream-aware.
bool ORT_API_CALL IsStreamAware(const OrtEpFactory* /*this_ptr*/) noexcept {
    return false;
}

// We register no allocator info (plain CPU memory via ORT's default allocator),
// so ORT should never call this. Provide a non-null stub anyway to avoid any
// chance of a null-pointer call, and fail loudly if it is ever invoked.
OrtStatus* ORT_API_CALL CreateAllocator(OrtEpFactory* this_ptr,
                                        const OrtMemoryInfo* /*memory_info*/,
                                        const OrtKeyValuePairs* /*allocator_options*/,
                                        OrtAllocator** allocator) noexcept {
    auto& self = *Self(this_ptr);
    *allocator = nullptr;
    return self.ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
        "FusionCPU EP CreateAllocator: pure-CPU EP registers no custom allocator.");
}

void ORT_API_CALL ReleaseAllocator(OrtEpFactory* /*this_ptr*/,
                                   OrtAllocator* /*allocator*/) noexcept {
    // Nothing to release — CreateAllocator never hands out an allocator.
}

OrtStatus* ORT_API_CALL CreateSyncStreamForDevice(OrtEpFactory* this_ptr,
                                                  const OrtMemoryDevice* /*device*/,
                                                  const OrtKeyValuePairs* /*options*/,
                                                  OrtSyncStreamImpl** stream) noexcept {
    auto& self = *Self(this_ptr);
    *stream = nullptr;
    return self.ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
        "FusionCPU EP CreateSyncStreamForDevice: pure-CPU EP is not stream-aware.");
}

void InitFactoryVtable(FusionCpuEpFactory& f) {
    auto& v = f.base;
    v.ort_version_supported     = ORT_API_VERSION;
    v.GetName                   = &GetName;
    v.GetVendor                 = &GetVendor;
    v.GetVendorId               = &GetVendorId;
    v.GetVersion                = &GetVersion;
    v.GetSupportedDevices       = &GetSupportedDevices;
    v.CreateEp                  = &CreateEp;
    v.ReleaseEp                 = &ReleaseEp;
    // ORT invokes these factory members unconditionally during registration /
    // session setup, so they MUST be non-null even though a pure-CPU EP has no
    // custom allocator / data transfer / streams.
    v.CreateAllocator           = &CreateAllocator;
    v.ReleaseAllocator          = &ReleaseAllocator;
    v.CreateDataTransfer        = &CreateDataTransfer;
    v.IsStreamAware             = &IsStreamAware;
    v.CreateSyncStreamForDevice = &CreateSyncStreamForDevice;
}

}  // namespace

OrtEpFactory* CreateFusionCpuEpFactory(const char* registered_name,
                                       const OrtApi* ort_api,
                                       const OrtLogger* default_logger) {
    auto self = std::make_unique<FusionCpuEpFactory>();
    std::memset(&self->base, 0, sizeof(self->base));
    self->ort_api         = ort_api;
    self->ep_api          = ort_api ? ort_api->GetEpApi() : nullptr;
    self->logger          = default_logger;
    self->registered_name = registered_name ? registered_name : FUSION_CPU_EP_NAME;
    self->version_str     = FUSION_CPU_EP_VERSION_STRING;

    InitFactoryVtable(*self);
    return reinterpret_cast<OrtEpFactory*>(self.release());
}

void DestroyFusionCpuEpFactory(OrtEpFactory* factory) {
    delete reinterpret_cast<FusionCpuEpFactory*>(factory);
}

}  // namespace fusion_cpu_ep
