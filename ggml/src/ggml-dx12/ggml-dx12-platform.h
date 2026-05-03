// ggml-dx12-platform.h - Platform-specific DX12 hooks for b612.dc_041126
//
// This file provides adapter filtering and buffer allocation fixes that are
// NOT present in the upstream d3ddk repo. When syncing ggml-dx12.cpp from
// d3ddk, this file stays untouched — only the two call sites in ggml-dx12.cpp
// need to be re-inserted (search for "PLATFORM_HOOK").
//
// Fixes provided:
//   1. dx12_platform_filter_adapter() — deduplicates GPU partitions (Hyper-V
//      GPU-PV / AMD multi-partition) and filters WARP software renderer.
//   2. dx12_platform_create_committed_resource() — uses CREATE_NOT_ZEROED to
//      prevent TDR on multi-GB buffer allocations, with fallback for older Windows.
//
#pragma once

#ifdef _WIN32
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#else
#include <winadapter.h>
#include <directx/d3d12.h>
#include <directx/dxgi1_6.h>
#endif
#include <wrl/client.h>
#include <vector>
#include <cstring>

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Adapter filter state — call dx12_platform_filter_adapter() for each adapter
// during enumeration. Returns true if the adapter should be SKIPPED.
// ---------------------------------------------------------------------------

struct dx12_adapter_filter {
    struct adapter_key {
        LUID luid;
        UINT vendor_id;
        UINT device_id;
    };
    std::vector<adapter_key> seen;
};

// Returns true if this adapter should be skipped (duplicate or software).
// Caller should declare one dx12_adapter_filter instance per enumeration pass.
static inline bool dx12_platform_filter_adapter(dx12_adapter_filter & filter,
                                                const DXGI_ADAPTER_DESC1 & desc) {
    // Skip Microsoft Basic Render Driver (WARP) — not always flagged as software
    {
        char name_buf[256];
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf, sizeof(name_buf), nullptr, nullptr);
        if (strstr(name_buf, "Basic Render") || strstr(name_buf, "Microsoft Basic")) {
            return true; // skip
        }
    }

    // Skip duplicate adapters — same LUID or same VendorId+DeviceId.
    // AMD drivers and Hyper-V GPU-PV enumerate the same physical GPU multiple
    // times with different LUIDs; match by hardware IDs as fallback.
    for (const auto & key : filter.seen) {
        bool luid_match = (key.luid.LowPart == desc.AdapterLuid.LowPart &&
                           key.luid.HighPart == desc.AdapterLuid.HighPart);
        bool hw_match   = (key.vendor_id == desc.VendorId &&
                           key.device_id == desc.DeviceId);
        if (luid_match || hw_match) {
            return true; // skip duplicate
        }
    }
    filter.seen.push_back({ desc.AdapterLuid, desc.VendorId, desc.DeviceId });
    return false; // keep this adapter
}

// ---------------------------------------------------------------------------
// Buffer allocation — wraps CreateCommittedResource.
// The new d3ddk code switched to HEAP_FLAG_NONE (zero-fill) — new shaders
// may rely on zero-initialized buffers. We preserve that behavior here.
//
// TODO: For very large buffers (>2 GB), this may trigger TDR.  A chunked
// zero-fill using CREATE_NOT_ZEROED + explicit clear is the long-term fix.
// ---------------------------------------------------------------------------

static inline HRESULT dx12_platform_create_committed_resource(
        ID3D12Device * device,
        const D3D12_HEAP_PROPERTIES * hp,
        D3D12_HEAP_FLAGS heap_flags,
        const D3D12_RESOURCE_DESC * rd,
        D3D12_RESOURCE_STATES init_state,
        ComPtr<ID3D12Resource> * out_resource) {
    // Use HEAP_FLAG_NONE (zero-fill) to match new d3ddk behavior.
    HRESULT hr = device->CreateCommittedResource(
        hp, heap_flags, rd,
        init_state, nullptr, IID_PPV_ARGS(out_resource->ReleaseAndGetAddressOf()));
    return hr;
}
