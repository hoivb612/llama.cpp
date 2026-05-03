// ggml-dx12.cpp - DirectX 12 backend for ggml
//
// Implements a GPU compute backend using D3D12, with optional Cooperative Vector
// acceleration for matrix-vector operations (SM 6.9 / Agility SDK 1.717+).

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#else
// WSL2: use DirectX-Headers and DXCore (no DXGI on Linux)
#include <winadapter.h>
#include <directx/d3d12.h>
#include <directx/dxcore.h>
#include <directx/dxcore_interface.h>
#include <directx/dxgiformat.h>
#include <dxguids/dxguids.h>
#include <thread>
#include <chrono>
#endif
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <map>

#include "ggml-dx12.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#ifdef GGML_DX12_SHADERS_COMPILED
#include "ggml_dx12_shaders.h"
#endif

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// WSL2 compatibility shims
// ---------------------------------------------------------------------------
#ifndef _WIN32

// DXGI error codes (raw HRESULT values) used in D3D12 error checking
#ifndef DXGI_ERROR_NOT_FOUND
#define DXGI_ERROR_NOT_FOUND      ((HRESULT)0x887A0002L)
#endif
#ifndef DXGI_ERROR_DEVICE_REMOVED
#define DXGI_ERROR_DEVICE_REMOVED ((HRESULT)0x887A0005L)
#endif

// Win32 sync shims: fence wait uses polling on WSL2
static inline HANDLE dx12_create_event() { return nullptr; }
static inline void   dx12_close_event(HANDLE) {}
static inline void   dx12_wait_event(HANDLE, DWORD) {}

// WideCharToMultiByte replacement for DXCore (provides UTF-8 natively)
static inline void dx12_wide_to_utf8(const wchar_t * src, char * dst, int dst_size) {
    int i = 0;
    while (src[i] && i < dst_size - 1) { dst[i] = (char)src[i]; i++; }
    dst[i] = '\0';
}

#else // _WIN32

static inline HANDLE dx12_create_event() { return CreateEvent(nullptr, FALSE, FALSE, nullptr); }
static inline void   dx12_close_event(HANDLE h) { CloseHandle(h); }
static inline void   dx12_wait_event(HANDLE h, DWORD ms) { WaitForSingleObject(h, ms); }
#define dx12_wide_to_utf8(src, dst, sz) WideCharToMultiByte(CP_UTF8, 0, src, -1, dst, sz, nullptr, nullptr)

#endif // _WIN32

// Sentinel base address for non-host-accessible GPU buffers (matches Vulkan approach)
static void * const DX12_PTR_BASE = (void *)(uintptr_t)0x1000;

static uint64_t dx12_tensor_offset(const struct ggml_tensor * tensor) {
    return (uint8_t *)tensor->data - (uint8_t *)DX12_PTR_BASE;
}

// ---------------------------------------------------------------------------
// Debug logging
// ---------------------------------------------------------------------------

#ifdef GGML_DX12_DEBUG
#define DX12_LOG_DEBUG(...) GGML_LOG_DEBUG("ggml-dx12: " __VA_ARGS__)
#else
#define DX12_LOG_DEBUG(...)
#endif

#define DX12_LOG_INFO(...)  GGML_LOG_INFO ("ggml-dx12: " __VA_ARGS__)
#define DX12_LOG_WARN(...)  GGML_LOG_WARN ("ggml-dx12: " __VA_ARGS__)
#define DX12_LOG_ERROR(...) GGML_LOG_ERROR("ggml-dx12: " __VA_ARGS__)

// ---------------------------------------------------------------------------
// HR check helper
// ---------------------------------------------------------------------------

// Thread-local device pointer for error reporting
static thread_local ID3D12Device * g_tls_device = nullptr;

static inline void dx12_check_hr(HRESULT hr, const char * msg, const char * file, int line) {
    if (FAILED(hr)) {
        fprintf(stderr, "ggml-dx12: %s failed (HRESULT 0x%08X) at %s:%d\n", msg, (unsigned)hr, file, line);
        if (hr == (HRESULT)0x887A0005 /* DXGI_ERROR_DEVICE_REMOVED */ && g_tls_device) {
            HRESULT reason = g_tls_device->GetDeviceRemovedReason();
            fprintf(stderr, "ggml-dx12: Device removed reason: 0x%08X", (unsigned)reason);
            if (reason == (HRESULT)0x887A0006) fprintf(stderr, " (DEVICE_HUNG / TDR timeout)");
            if (reason == (HRESULT)0x887A0005) fprintf(stderr, " (DEVICE_REMOVED)");
            if (reason == (HRESULT)0x887A0007) fprintf(stderr, " (DEVICE_RESET)");
            if (reason == (HRESULT)0x887A0020) fprintf(stderr, " (DRIVER_INTERNAL_ERROR)");
            if (reason == (HRESULT)0x80070057) fprintf(stderr, " (E_INVALIDARG)");
            fprintf(stderr, "\n");
            fflush(stderr);
        }
        GGML_ABORT("DX12 fatal error");
    }
}
#define DX12_CHECK(hr, msg) dx12_check_hr(hr, msg, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

struct dx12_device;
struct dx12_buffer;
struct dx12_backend;
struct dx12_pipeline;

// ---------------------------------------------------------------------------
// Pipeline key — identifies a unique shader variant
// ---------------------------------------------------------------------------

struct dx12_pipeline_key {
    ggml_op       op;
    ggml_type     src0_type;
    ggml_type     src1_type;
    ggml_type     dst_type;
    uint32_t      flags; // contiguity, specialization

    bool operator==(const dx12_pipeline_key & o) const {
        return op == o.op && src0_type == o.src0_type && src1_type == o.src1_type
            && dst_type == o.dst_type && flags == o.flags;
    }
};

struct dx12_pipeline_key_hash {
    size_t operator()(const dx12_pipeline_key & k) const {
        size_t h = std::hash<int>()(k.op);
        h ^= std::hash<int>()(k.src0_type) << 4;
        h ^= std::hash<int>()(k.src1_type) << 8;
        h ^= std::hash<int>()(k.dst_type)  << 12;
        h ^= std::hash<uint32_t>()(k.flags) << 16;
        return h;
    }
};

// ---------------------------------------------------------------------------
// Pipeline — a root signature + PSO pair for one shader variant
// ---------------------------------------------------------------------------

struct dx12_pipeline {
    ComPtr<ID3D12RootSignature> root_sig;
    ComPtr<ID3D12PipelineState> pso;
    uint32_t num_root_constants = 0;
    uint32_t num_srvs           = 0;
    uint32_t num_uavs           = 0;
};

// ---------------------------------------------------------------------------
// Push constants / root constants structure for shader parameters
// ---------------------------------------------------------------------------

// Generic parameter block passed via root constants.
// We keep this at ≤ 64 DWORDs to fit comfortably in root signature space.
struct dx12_shader_params {
    uint32_t ne00, ne01, ne02, ne03;   // src0 shape
    uint32_t nb00, nb01, nb02, nb03;   // src0 strides (in bytes)
    uint32_t ne10, ne11, ne12, ne13;   // src1 shape
    uint32_t nb10, nb11, nb12, nb13;   // src1 strides (in bytes)
    uint32_t ne0, ne1, ne2, ne3;       // dst shape
    uint32_t nb0, nb1, nb2, nb3;       // dst strides (in bytes)
    uint32_t src0_offset;              // byte offset into src0 buffer
    uint32_t src1_offset;              // byte offset into src1 buffer
    uint32_t dst_offset;               // byte offset into dst buffer
    uint32_t src0_esize;               // src0 element size in bytes (2=F16, 4=F32)
    uint32_t src1_esize;               // src1 element size in bytes
    uint32_t dst_esize;                // dst element size in bytes
    uint32_t op_params[16];            // extra op-specific params
};
static_assert(sizeof(dx12_shader_params) % 4 == 0, "must be DWORD-aligned");
static_assert(sizeof(dx12_shader_params) / 4 <= 64, "must fit in root constants");

// ---------------------------------------------------------------------------
// Device — represents one D3D12 adapter + device
// ---------------------------------------------------------------------------

struct dx12_device {
#ifdef _WIN32
    ComPtr<IDXGIAdapter1>     adapter;
    DXGI_ADAPTER_DESC1        adapter_desc = {};
#else
    ComPtr<IDXCoreAdapter>    adapter;
#endif
    ComPtr<ID3D12Device>      device;
    ComPtr<ID3D12CommandQueue> compute_queue;

    size_t                    vram_total   = 0;
    size_t                    vram_free    = 0;
    uint32_t                  vendor_id    = 0;
    uint32_t                  device_id    = 0;

    bool cooperative_vector_supported = false;

    // WaveMMA (SM 6.9 Wave Matrix) support
    bool wave_mma_supported = false;
    uint32_t wave_mma_K      = 0;     // hardware K dimension (even multiple of 16)
    uint32_t wave_mma_M      = 0;     // M dimension (16 or 64)
    uint32_t wave_mma_N      = 0;     // N dimension (16 or 64)
    uint32_t wave_mma_wave_size = 0;  // required wave size for WaveMMA
    bool     wave_mma_f16_acc32 = false; // F16 input with F32 accumulator

    // Auto-tuning: optimal shader variants per quant type
    // Determined by GPU microbenchmark at first model load
    // Bump TUNE_VERSION when adding new dimensions to invalidate cache
    static constexpr int TUNE_VERSION = 2;
    bool tuning_done = false;
    bool q5_0_use_256 = false;  // Q5_0 matvec: true=256 threads, false=32 threads
    bool q8_0_use_256 = false;  // Q8_0 matvec: true=256 threads, false=32 threads
    bool q6k_use_32   = false;  // Q6_K matvec: true=32 threads, false=256 threads (default=256)
    bool f16_use_load4 = false; // F16 matvec: true=Load4 variant, false=Load2 (default)

    void run_autotune();

    // Pipeline cache
    std::mutex pipeline_mutex;
    std::unordered_map<dx12_pipeline_key, dx12_pipeline, dx12_pipeline_key_hash> pipeline_cache;

    // Common root signature for most shaders
    ComPtr<ID3D12RootSignature> common_root_sig;

    // Persistent transfer context — reused for all set_tensor/get_tensor calls
    // instead of creating/destroying D3D12 objects per call
    struct {
        ComPtr<ID3D12CommandAllocator>    cmd_alloc;
        ComPtr<ID3D12GraphicsCommandList> cmd_list;
        ComPtr<ID3D12Fence>              fence;
        HANDLE                           fence_event = nullptr;
        uint64_t                         fence_value = 0;
        ComPtr<ID3D12Resource>           upload_staging;
        size_t                           upload_size = 0;
        ComPtr<ID3D12Resource>           readback_staging;
        size_t                           readback_size = 0;
        bool                             initialized = false;
    } xfer;

    void init_xfer() {
        if (xfer.initialized) return;
        HRESULT hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                     IID_PPV_ARGS(&xfer.cmd_alloc));
        DX12_CHECK(hr, "CreateCommandAllocator(xfer)");
        hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        xfer.cmd_alloc.Get(), nullptr,
                                        IID_PPV_ARGS(&xfer.cmd_list));
        DX12_CHECK(hr, "CreateCommandList(xfer)");
        xfer.cmd_list->Close(); // start in closed state
        hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&xfer.fence));
        DX12_CHECK(hr, "CreateFence(xfer)");
        xfer.fence_event = dx12_create_event();
        xfer.initialized = true;
    }

    void xfer_wait() {
        if (xfer.fence_value == 0) return;
        if (xfer.fence->GetCompletedValue() >= xfer.fence_value) return;
#ifdef _WIN32
        xfer.fence->SetEventOnCompletion(xfer.fence_value, xfer.fence_event);
        dx12_wait_event(xfer.fence_event, INFINITE);
#else
        while (xfer.fence->GetCompletedValue() < xfer.fence_value) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
#endif
    }

    void xfer_ensure_staging(size_t up_size, size_t rb_size) {
        auto make_buf = [&](ComPtr<ID3D12Resource> & res, size_t & cur, size_t need, D3D12_HEAP_TYPE ht) {
            if (cur >= need) return;
            need = (need + 0xFFFF) & ~(size_t)0xFFFF;
            res.Reset();
            D3D12_HEAP_PROPERTIES hp = {}; hp.Type = ht;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = need; rd.Height = 1; rd.DepthOrArraySize = 1;
            rd.MipLevels = 1; rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            rd.Flags = D3D12_RESOURCE_FLAG_NONE;
            HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                ht == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr, IID_PPV_ARGS(&res));
            DX12_CHECK(hr, "CreateCommittedResource(xfer staging)");
            cur = need;
        };
        if (up_size > 0) make_buf(xfer.upload_staging, xfer.upload_size, up_size, D3D12_HEAP_TYPE_UPLOAD);
        if (rb_size > 0) make_buf(xfer.readback_staging, xfer.readback_size, rb_size, D3D12_HEAP_TYPE_READBACK);
    }

    // Device index in the global list
    size_t dev_index = 0;
    std::string name;        // "DX120", "DX121", etc. (for --dev matching)
    std::string description; // GPU name from adapter desc

    dx12_device() = default;
    dx12_device(const dx12_device &) = delete;
    dx12_device & operator=(const dx12_device &) = delete;

    ~dx12_device() {
        if (xfer.fence_event) {
            xfer_wait();
            dx12_close_event(xfer.fence_event);
        }
    }

#ifdef _WIN32
    void init(ComPtr<IDXGIAdapter1> adapter_, size_t idx);
#else
    void init(ComPtr<IDXCoreAdapter> adapter_, size_t idx);
#endif
    void create_common_root_signature();
    dx12_pipeline * get_or_create_pipeline(const dx12_pipeline_key & key);
};

// ---------------------------------------------------------------------------
// Buffer context
// ---------------------------------------------------------------------------

struct dx12_buffer_context {
    dx12_device *          dev       = nullptr;
    ComPtr<ID3D12Resource> resource;
    size_t                 size      = 0;
    D3D12_HEAP_TYPE        heap_type = D3D12_HEAP_TYPE_DEFAULT;
    void *                 mapped    = nullptr; // non-null for upload/readback heaps
};

// ---------------------------------------------------------------------------
// Backend context (stream)
// ---------------------------------------------------------------------------

static const int CMD_RING_SIZE = 3;

struct dx12_backend_context {
    dx12_device * dev = nullptr;

    // Command allocator ring — 3 allocators so CPU can record while GPU executes
    ComPtr<ID3D12CommandAllocator>    cmd_allocs[CMD_RING_SIZE];
    uint64_t                          cmd_alloc_fence[CMD_RING_SIZE] = {}; // fence value when submitted
    int                               cmd_ring_head = 0; // next allocator to use
    ComPtr<ID3D12GraphicsCommandList> cmd_list;
    ComPtr<ID3D12Fence>              fence;
    HANDLE                           fence_event = nullptr;
    uint64_t                         fence_value = 0;

    // Staging buffers for set/get tensor
    ComPtr<ID3D12Resource> upload_staging;
    size_t                 upload_staging_size   = 0;
    ComPtr<ID3D12Resource> readback_staging;
    size_t                 readback_staging_size = 0;

    bool cmd_list_open = false;

    // --- Redundant D3D12 call elimination state ---
    ID3D12PipelineState *      last_pso      = nullptr;
    ID3D12RootSignature *      last_root_sig = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS  last_src0_va  = 0;
    D3D12_GPU_VIRTUAL_ADDRESS  last_src1_va  = 0;
    D3D12_GPU_VIRTUAL_ADDRESS  last_dst_va   = 0;
    D3D12_GPU_VIRTUAL_ADDRESS  last_src2_va  = 0;
    D3D12_GPU_VIRTUAL_ADDRESS  last_src3_va  = 0;
    ID3D12Resource *           last_barrier_res = nullptr;
    ID3D12Resource *           last_uav_res = nullptr;

    void reset_binding_cache() {
        last_pso      = nullptr;
        last_root_sig = nullptr;
        last_src0_va  = 0;
        last_src1_va  = 0;
        last_dst_va   = 0;
        last_src2_va  = 0;
        last_src3_va  = 0;
        last_barrier_res = nullptr;
        last_uav_res = nullptr;
    }

    ~dx12_backend_context() {
        // RAII cleanup: wait for ALL GPU work and close event handle
        if (fence && fence_event && dev) {
            wait_for_gpu();
        }
        if (fence_event) {
            dx12_close_event(fence_event);
            fence_event = nullptr;
        }
    }

    void ensure_cmd_list_open();
    void close_and_execute();
    void wait_for_gpu();
    void wait_for_fence(uint64_t value);
    void ensure_staging(size_t upload_size, size_t readback_size);
};

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static struct {
    bool                                        initialized = false;
#ifdef _WIN32
    ComPtr<IDXGIFactory4>                       factory;
#else
    ComPtr<IDXCoreAdapterFactory>               factory;
#endif
    std::vector<std::unique_ptr<dx12_device>>   devices;
    std::mutex                                  init_mutex;

    // Backend device & registry objects
    std::vector<ggml_backend_device> backend_devices;
    ggml_backend_reg               backend_reg_obj = {};
} g_dx12;

// ---------------------------------------------------------------------------
// Device initialization
// ---------------------------------------------------------------------------

static void dx12_ensure_initialized() {
    std::lock_guard<std::mutex> lock(g_dx12.init_mutex);
    if (g_dx12.initialized) return;

    // Enable debug layer when DX12_DEBUG env var is set
    if (getenv("DX12_DEBUG")) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            DX12_LOG_INFO("D3D12 debug layer enabled\n");
        }
    }

    // Enable experimental features for Cooperative Vector (best-effort, non-fatal)
    {
        UUID features[] = { D3D12ExperimentalShaderModels };
        HRESULT hr = D3D12EnableExperimentalFeatures(1, features, nullptr, nullptr);
        if (SUCCEEDED(hr)) {
            DX12_LOG_DEBUG("Experimental shader models enabled\n");
        }
    }

#ifdef _WIN32
    // --- Windows: DXGI-based adapter enumeration ---
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&g_dx12.factory));
    DX12_CHECK(hr, "CreateDXGIFactory1");

    // Track seen adapters to skip duplicates for the same physical GPU.
    // AMD drivers (and Hyper-V GPU-PV) can enumerate the same GPU multiple times
    // with different LUIDs, so we check both LUID and VendorId+DeviceId.
    struct adapter_key {
        LUID   luid;
        UINT   vendor_id;
        UINT   device_id;
    };
    std::vector<adapter_key> seen_adapters;

    // Enumerate adapters
    for (UINT i = 0; ; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        hr = g_dx12.factory->EnumAdapters1(i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        DX12_CHECK(hr, "EnumAdapters1");

        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        // Skip software adapters (flag check)
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        // Skip Microsoft Basic Render Driver (WARP) — not always flagged as software
        {
            char name_buf[256];
            dx12_wide_to_utf8(desc.Description, name_buf, sizeof(name_buf));
            if (strstr(name_buf, "Basic Render") || strstr(name_buf, "Microsoft Basic")) {
                DX12_LOG_DEBUG("Skipping software adapter: %s\n", name_buf);
                continue;
            }
        }

        // Skip duplicate adapters — same LUID or same VendorId+DeviceId
        {
            bool duplicate = false;
            for (const auto & key : seen_adapters) {
                bool luid_match = (key.luid.LowPart == desc.AdapterLuid.LowPart &&
                                   key.luid.HighPart == desc.AdapterLuid.HighPart);
                bool hw_match   = (key.vendor_id == desc.VendorId &&
                                   key.device_id == desc.DeviceId);
                if (luid_match || hw_match) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                char name_buf[256];
                dx12_wide_to_utf8(desc.Description, name_buf, sizeof(name_buf));
                DX12_LOG_DEBUG("Skipping duplicate adapter: %s\n", name_buf);
                continue;
            }
            seen_adapters.push_back({ desc.AdapterLuid, desc.VendorId, desc.DeviceId });
        }

        // Try to create a D3D12 device to verify support
        ComPtr<ID3D12Device> test_device;
        hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&test_device));
        if (FAILED(hr)) continue;

        // Validate compute capability: try a small UAV buffer allocation.
        // Some integrated GPUs pass device creation but fail actual compute allocations.
        {
            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width            = 4096;
            rd.Height           = 1;
            rd.DepthOrArraySize = 1;
            rd.MipLevels        = 1;
            rd.SampleDesc.Count = 1;
            rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            ComPtr<ID3D12Resource> test_buf;
            hr = test_device->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&test_buf));
            if (FAILED(hr)) {
                char name_buf[128];
                dx12_wide_to_utf8(desc.Description, name_buf, sizeof(name_buf));
                DX12_LOG_WARN("Skipping %s: UAV allocation failed (HRESULT 0x%08X)\n", name_buf, (unsigned)hr);
                continue;
            }
        }
        test_device.Reset();

        if (g_dx12.devices.size() >= GGML_DX12_MAX_DEVICES) break;

        g_dx12.devices.push_back(std::make_unique<dx12_device>());
        g_dx12.devices.back()->init(std::move(adapter), g_dx12.devices.size() - 1);
    }

#else
    // --- WSL2: DXCore-based adapter enumeration ---
    HRESULT hr = DXCoreCreateAdapterFactory(IID_PPV_ARGS(&g_dx12.factory));
    DX12_CHECK(hr, "DXCoreCreateAdapterFactory");

    const GUID filter_attrs[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
    ComPtr<IDXCoreAdapterList> adapter_list;
    hr = g_dx12.factory->CreateAdapterList(1, filter_attrs, IID_PPV_ARGS(&adapter_list));
    DX12_CHECK(hr, "CreateAdapterList");

    const uint32_t adapter_count = adapter_list->GetAdapterCount();
    for (uint32_t i = 0; i < adapter_count; ++i) {
        ComPtr<IDXCoreAdapter> adapter;
        hr = adapter_list->GetAdapter(i, IID_PPV_ARGS(&adapter));
        if (FAILED(hr)) continue;

        // Skip software adapters (IsHardware == false)
        bool is_hardware = false;
        if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::IsHardware,
                                           sizeof(is_hardware), &is_hardware)) && !is_hardware) {
            continue;
        }

        // Get adapter description (UTF-8 natively from DXCore)
        size_t desc_size = 0;
        adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &desc_size);
        std::string adapter_name(desc_size, '\0');
        adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, desc_size, adapter_name.data());
        if (!adapter_name.empty() && adapter_name.back() == '\0') adapter_name.pop_back();

        // Skip WARP / Basic Render
        if (adapter_name.find("Basic Render") != std::string::npos ||
            adapter_name.find("Microsoft Basic") != std::string::npos) {
            DX12_LOG_DEBUG("Skipping software adapter: %s\n", adapter_name.c_str());
            continue;
        }

        // Try to create a D3D12 device
        ComPtr<ID3D12Device> test_device;
        hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&test_device));
        if (FAILED(hr)) {
            DX12_LOG_DEBUG("Skipping %s: D3D12CreateDevice failed (0x%08X)\n", adapter_name.c_str(), (unsigned)hr);
            continue;
        }

        // Validate compute capability with UAV allocation test
        {
            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width            = 4096;
            rd.Height           = 1;
            rd.DepthOrArraySize = 1;
            rd.MipLevels        = 1;
            rd.SampleDesc.Count = 1;
            rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            ComPtr<ID3D12Resource> test_buf;
            hr = test_device->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&test_buf));
            if (FAILED(hr)) {
                DX12_LOG_WARN("Skipping %s: UAV allocation failed (0x%08X)\n", adapter_name.c_str(), (unsigned)hr);
                continue;
            }
        }
        test_device.Reset();

        if (g_dx12.devices.size() >= GGML_DX12_MAX_DEVICES) break;

        g_dx12.devices.push_back(std::make_unique<dx12_device>());
        g_dx12.devices.back()->init(std::move(adapter), g_dx12.devices.size() - 1);
    }
#endif // _WIN32

    DX12_LOG_INFO("Found %zu D3D12 device(s)\n", g_dx12.devices.size());
    g_dx12.initialized = true;
}

// ---------------------------------------------------------------------------
// dx12_device implementation
// ---------------------------------------------------------------------------

#ifdef _WIN32
void dx12_device::init(ComPtr<IDXGIAdapter1> adapter_, size_t idx) {
    adapter   = std::move(adapter_);
    dev_index = idx;

    adapter->GetDesc1(&adapter_desc);

    // Convert wide name to narrow
    char narrow[256];
    dx12_wide_to_utf8(adapter_desc.Description, narrow, sizeof(narrow));
    description = narrow;
    name = std::string(GGML_DX12_NAME) + std::to_string(idx);

    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    DX12_CHECK(hr, "D3D12CreateDevice");

    // Create compute command queue
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type     = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    qd.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&compute_queue));
    DX12_CHECK(hr, "CreateCommandQueue(compute)");

    // VRAM: use dedicated video memory. For integrated GPUs with little dedicated
    // memory, use a conservative estimate to avoid the scheduler preferring them
    // over discrete GPUs with large VRAM.
    bool is_integrated = (adapter_desc.DedicatedVideoMemory < (size_t)512 * 1024 * 1024);
    vram_total = adapter_desc.DedicatedVideoMemory;
    if (is_integrated) {
        // Integrated GPU: dedicated memory is small (DVMT pre-allocated).
        // Use it as-is — don't inflate with shared system memory or DXGI budget.
        vram_total = adapter_desc.DedicatedVideoMemory;
        if (vram_total == 0) {
            vram_total = adapter_desc.SharedSystemMemory / 4;
        }
    }

    // Query actual budget via DXGI 1.4 — only for discrete GPUs.
    // iGPU DXGI local segment includes most of system RAM, which inflates the budget.
    ComPtr<IDXGIAdapter3> adapter3;
    if (!is_integrated && SUCCEEDED(adapter.As(&adapter3))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO mem_info = {};
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mem_info))
            && mem_info.Budget > 0) {
            vram_total = (size_t)mem_info.Budget;
            vram_free  = mem_info.Budget > mem_info.CurrentUsage
                       ? (size_t)(mem_info.Budget - mem_info.CurrentUsage)
                       : 0;
        }
    } else {
        vram_free = vram_total;
    }
    vendor_id = adapter_desc.VendorId;
    device_id = adapter_desc.DeviceId;
#else
void dx12_device::init(ComPtr<IDXCoreAdapter> adapter_, size_t idx) {
    adapter   = std::move(adapter_);
    dev_index = idx;

    // Get adapter description (UTF-8 from DXCore)
    size_t desc_size = 0;
    adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &desc_size);
    description.resize(desc_size);
    adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, desc_size, description.data());
    if (!description.empty() && description.back() == '\0') description.pop_back();
    name = std::string(GGML_DX12_NAME) + std::to_string(idx);

    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    DX12_CHECK(hr, "D3D12CreateDevice");

    // Create compute command queue
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type     = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    qd.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&compute_queue));
    DX12_CHECK(hr, "CreateCommandQueue(compute)");

    // VRAM: query dedicated memory from DXCore
    size_t dedicated_mem = 0;
    if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::DedicatedAdapterMemory,
                                       sizeof(dedicated_mem), &dedicated_mem))) {
        vram_total = dedicated_mem;
    }
    if (vram_total == 0) {
        // Fallback: use shared memory estimate
        size_t shared_mem = 0;
        if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::SharedSystemMemory,
                                           sizeof(shared_mem), &shared_mem))) {
            vram_total = shared_mem / 4;
        }
    }
    vram_free = vram_total;

    // Get hardware IDs from DXCore
    DXCoreHardwareID hw_id = {};
    if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::HardwareID, sizeof(hw_id), &hw_id))) {
        vendor_id = hw_id.vendorID;
        device_id = hw_id.deviceID;
    }
#endif

    // Check Cooperative Vector support
    cooperative_vector_supported = false;
    {
        // Try to query CV support — requires preview Agility SDK headers
        // For now, try the feature check and see if the driver supports it
        struct {
            UINT CooperativeVectorTier;
        } exp_opts = {};
        // D3D12_FEATURE value 52 = D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL (preview)
        HRESULT hr2 = device->CheckFeatureSupport((D3D12_FEATURE)52, &exp_opts, sizeof(exp_opts));
        if (SUCCEEDED(hr2) && exp_opts.CooperativeVectorTier >= 1) {
            cooperative_vector_supported = true;
        }
    }

    // Check WaveMMA (SM 6.9 Wave Matrix) support
    // D3D12_FEATURE_WAVE_MMA queries hardware matrix multiply-accumulate capability
    wave_mma_supported = false;
    {
        // D3D12_FEATURE_WAVE_MMA — feature enum value TBD (not yet in public headers)
        // Structure matches the spec: input DataType + M + N, output Supported + K + AccumPrecision + RequiredWaveSize
        // Try F16 with 16x16 first (most common and useful for LLM inference)
        struct {
            UINT DataType;          // 0=BYTE, 1=FLOAT16, 2=FLOAT
            UINT M;                 // 0=16, 1=64
            UINT N;                 // 0=16, 1=64
            BOOL Supported;
            UINT K;
            UINT AccumPrecision;    // flags: 0x1=16-bit, 0x2=32-bit
            UINT RequiredWaveSize;
        } wave_mma_caps = {};
        wave_mma_caps.DataType = 1;  // D3D12_WAVE_MMA_DATATYPE_FLOAT16
        wave_mma_caps.M = 0;        // D3D12_WAVE_MMA_DIMENSION_16
        wave_mma_caps.N = 0;        // D3D12_WAVE_MMA_DIMENSION_16

        // D3D12_FEATURE_WAVE_MMA = 53 (tentative — may vary by SDK version)
        // Try a range of feature enum values since the exact value depends on the SDK
        bool found = false;
        for (UINT feat_id = 53; feat_id <= 60 && !found; feat_id++) {
            HRESULT hr2 = device->CheckFeatureSupport((D3D12_FEATURE)feat_id, &wave_mma_caps, sizeof(wave_mma_caps));
            if (SUCCEEDED(hr2) && wave_mma_caps.Supported) {
                wave_mma_supported = true;
                wave_mma_K = wave_mma_caps.K;
                wave_mma_M = 16;
                wave_mma_N = 16;
                wave_mma_wave_size = wave_mma_caps.RequiredWaveSize;
                wave_mma_f16_acc32 = (wave_mma_caps.AccumPrecision & 0x2) != 0;
                found = true;
            }
        }
    }

    create_common_root_signature();

    DX12_LOG_INFO("Device %zu: %s (%s, VRAM: %.1f GB, CV: %s, WaveMMA: %s%s)\n",
                  idx, name.c_str(), description.c_str(),
                  (double)vram_total / (1024.0 * 1024.0 * 1024.0),
                  cooperative_vector_supported ? "yes" : "no",
                  wave_mma_supported ? "yes" : "no",
                  wave_mma_supported ? (std::string(" K=") + std::to_string(wave_mma_K) +
                                        " wave=" + std::to_string(wave_mma_wave_size) +
                                        (wave_mma_f16_acc32 ? " f16→f32" : " f16→f16")).c_str() : "");
}

void dx12_device::create_common_root_signature() {
    // Common root signature layout:
    //   Slot 0: Root constants (dx12_shader_params)
    //   Slot 1: SRV root descriptor (src0 ByteAddressBuffer)
    //   Slot 2: SRV root descriptor (src1 ByteAddressBuffer)
    //   Slot 3: UAV root descriptor (dst  RWByteAddressBuffer)
    //   Slot 4: SRV root descriptor (src2 ByteAddressBuffer) [optional]
    //   Slot 5: SRV root descriptor (src3 ByteAddressBuffer) [optional, mask]

    D3D12_ROOT_PARAMETER1 params[6] = {};

    // Slot 0: Root constants
    params[0].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister  = 0; // b0
    params[0].Constants.RegisterSpace   = 0;
    params[0].Constants.Num32BitValues  = sizeof(dx12_shader_params) / 4;
    params[0].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 1: src0 SRV (t0)
    params[1].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 0; // t0
    params[1].Descriptor.RegisterSpace  = 0;
    params[1].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 2: src1 SRV (t1)
    params[2].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].Descriptor.ShaderRegister = 1; // t1
    params[2].Descriptor.RegisterSpace  = 0;
    params[2].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 3: dst UAV (u0)
    params[3].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[3].Descriptor.ShaderRegister = 0; // u0
    params[3].Descriptor.RegisterSpace  = 0;
    params[3].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 4: src2 SRV (t2)
    params[4].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[4].Descriptor.ShaderRegister = 2; // t2
    params[4].Descriptor.RegisterSpace  = 0;
    params[4].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 5: src3 SRV (t3) — mask for flash attention
    params[5].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[5].Descriptor.ShaderRegister = 3; // t3
    params[5].Descriptor.RegisterSpace  = 0;
    params[5].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rsd = {};
    rsd.Version                  = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rsd.Desc_1_1.NumParameters   = 6;
    rsd.Desc_1_1.pParameters     = params;
    rsd.Desc_1_1.NumStaticSamplers = 0;
    rsd.Desc_1_1.Flags           = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sig_blob, err_blob;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rsd, &sig_blob, &err_blob);
    if (FAILED(hr) && err_blob) {
        DX12_LOG_ERROR("Root signature serialization: %s\n", (char *)err_blob->GetBufferPointer());
    }
    DX12_CHECK(hr, "D3D12SerializeVersionedRootSignature");

    hr = device->CreateRootSignature(0, sig_blob->GetBufferPointer(), sig_blob->GetBufferSize(),
                                     IID_PPV_ARGS(&common_root_sig));
    DX12_CHECK(hr, "CreateRootSignature");
}

// ---------------------------------------------------------------------------
// Backend context (stream) implementation
// ---------------------------------------------------------------------------

void dx12_backend_context::ensure_cmd_list_open() {
    if (cmd_list_open) return;

    // Pick the next allocator in the ring
    int slot = cmd_ring_head;
    cmd_ring_head = (cmd_ring_head + 1) % CMD_RING_SIZE;

    // Only wait if THIS allocator's previous submission hasn't finished
    // Other allocators may still be in-flight — that's fine
    wait_for_fence(cmd_alloc_fence[slot]);

    HRESULT hr;
    if (!cmd_allocs[slot]) {
        hr = dev->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                  IID_PPV_ARGS(&cmd_allocs[slot]));
        DX12_CHECK(hr, "CreateCommandAllocator");
    }
    if (!cmd_list) {
        hr = dev->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             cmd_allocs[slot].Get(), nullptr,
                                             IID_PPV_ARGS(&cmd_list));
        DX12_CHECK(hr, "CreateCommandList");
    } else {
        hr = cmd_allocs[slot]->Reset();
        DX12_CHECK(hr, "CommandAllocator::Reset");
        hr = cmd_list->Reset(cmd_allocs[slot].Get(), nullptr);
        DX12_CHECK(hr, "CommandList::Reset");
    }
    reset_binding_cache();
    cmd_list_open = true;
}

void dx12_backend_context::close_and_execute() {
    if (!cmd_list_open) return;

    HRESULT hr = cmd_list->Close();
    DX12_CHECK(hr, "CommandList::Close");

    ID3D12CommandList * lists[] = { cmd_list.Get() };
    dev->compute_queue->ExecuteCommandLists(1, lists);

    fence_value++;
    hr = dev->compute_queue->Signal(fence.Get(), fence_value);
    DX12_CHECK(hr, "Signal fence");

    // Record which fence value this allocator was submitted with
    int submitted_slot = (cmd_ring_head + CMD_RING_SIZE - 1) % CMD_RING_SIZE;
    cmd_alloc_fence[submitted_slot] = fence_value;

    cmd_list_open = false;
}

void dx12_backend_context::wait_for_fence(uint64_t value) {
    if (value == 0) return; // never submitted
    if (fence->GetCompletedValue() >= value) return;

    HRESULT hr = fence->SetEventOnCompletion(value, fence_event);
    DX12_CHECK(hr, "SetEventOnCompletion");
#ifdef _WIN32
    dx12_wait_event(fence_event, INFINITE);
#else
    while (fence->GetCompletedValue() < value) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
#endif
}

void dx12_backend_context::wait_for_gpu() {
    if (fence_value == 0) return;
    wait_for_fence(fence_value);
    // Verify GPU didn't TDR during execution — device removal signals all fences
    HRESULT removed = dev->device->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        fprintf(stderr, "ggml-dx12: Device removed detected in wait_for_gpu! HRESULT 0x%08lX\n", (unsigned long)removed);
        fflush(stderr);
    }
}

void dx12_backend_context::ensure_staging(size_t upload_size, size_t readback_size) {
    auto create_buffer = [&](ComPtr<ID3D12Resource> & res, size_t & cur_size,
                             size_t needed, D3D12_HEAP_TYPE heap_type) {
        if (cur_size >= needed) return;
        // Round up to 64 KB
        needed = (needed + 0xFFFF) & ~(size_t)0xFFFF;
        res.Reset();

        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = heap_type;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width              = needed;
        rd.Height             = 1;
        rd.DepthOrArraySize   = 1;
        rd.MipLevels          = 1;
        rd.SampleDesc.Count   = 1;
        rd.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags              = D3D12_RESOURCE_FLAG_NONE;

        D3D12_RESOURCE_STATES init_state =
            (heap_type == D3D12_HEAP_TYPE_UPLOAD)   ? D3D12_RESOURCE_STATE_GENERIC_READ :
            (heap_type == D3D12_HEAP_TYPE_READBACK) ? D3D12_RESOURCE_STATE_COPY_DEST    :
                                                      D3D12_RESOURCE_STATE_COMMON;

        HRESULT hr = dev->device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd, init_state, nullptr, IID_PPV_ARGS(&res));
        DX12_CHECK(hr, "CreateCommittedResource(staging)");
        cur_size = needed;
    };

    if (upload_size > 0)   create_buffer(upload_staging,   upload_staging_size,   upload_size,   D3D12_HEAP_TYPE_UPLOAD);
    if (readback_size > 0) create_buffer(readback_staging, readback_staging_size, readback_size, D3D12_HEAP_TYPE_READBACK);
}

// ---------------------------------------------------------------------------
// Helper: create a GPU buffer (D3D12_HEAP_TYPE_DEFAULT)
// ---------------------------------------------------------------------------

static ComPtr<ID3D12Resource> dx12_create_buffer(dx12_device * dev, size_t size,
                                                  D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = std::max<size_t>(size, 256); // minimum 256 bytes for root descriptors
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = flags;

    ComPtr<ID3D12Resource> res;
    // Use CREATE_NOT_ZEROED to skip the runtime's internal GPU zero-fill.
    // Large buffers (multi-GB) can trigger TDR during zero-init on some drivers.
    // The allocator's clear() callback will zero the buffer in controlled chunks.
    HRESULT hr = dev->device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_CREATE_NOT_ZEROED, &rd,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&res));
    if (FAILED(hr)) {
        // Fallback for older Windows versions that don't support CREATE_NOT_ZEROED
        hr = dev->device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr, IID_PPV_ARGS(&res));
    }
    if (FAILED(hr)) {
        DX12_LOG_WARN("CreateCommittedResource(buffer) failed on %s (HRESULT 0x%08X, size=%zu)\n",
                      dev->name.c_str(), (unsigned)hr, size);
        return nullptr;
    }
    return res;
}

// ---------------------------------------------------------------------------
// Buffer type interface
// ---------------------------------------------------------------------------

static const char * dx12_buft_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "DX12";
}

static ggml_backend_buffer_t dx12_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dx12_device * dev = (dx12_device *)buft->context;

    auto * ctx = new dx12_buffer_context();
    ctx->dev       = dev;
    ctx->size      = size;
    ctx->heap_type = D3D12_HEAP_TYPE_DEFAULT;

    if (size > 0) {
        ctx->resource = dx12_create_buffer(dev, size);
        if (!ctx->resource) {
            delete ctx;
            return nullptr;
        }
    }

    static const ggml_backend_buffer_i iface = {
        /* .free_buffer   = */ [](ggml_backend_buffer_t buffer) {
            delete (dx12_buffer_context *)buffer->context;
        },
        /* .get_base      = */ [](ggml_backend_buffer_t buffer) -> void * {
            // D3D12 buffers aren't host-accessible; return a sentinel for offset math
            // tensor->data will be set to base + offset by the allocator
            GGML_UNUSED(buffer);
            return (void *)(uintptr_t)0x1000;
        },
        /* .init_tensor   = */ [](ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) -> ggml_status {
            GGML_UNUSED(buffer);
            GGML_UNUSED(tensor);
            return GGML_STATUS_SUCCESS;
        },
        /* .memset_tensor = */ [](ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                  uint8_t value, size_t offset, size_t size) {
            auto * ctx = (dx12_buffer_context *)buffer->context;
            if (size == 0) return;

            ctx->dev->init_xfer();
            ctx->dev->xfer_wait();
            ctx->dev->xfer_ensure_staging(size, 0);

            void * mapped = nullptr;
            D3D12_RANGE read_range = { 0, 0 };
            HRESULT hr = ctx->dev->xfer.upload_staging->Map(0, &read_range, &mapped);
            DX12_CHECK(hr, "Map(memset staging)");
            memset(mapped, value, size);
            ctx->dev->xfer.upload_staging->Unmap(0, nullptr);

            hr = ctx->dev->xfer.cmd_alloc->Reset();
            DX12_CHECK(hr, "xfer cmd_alloc Reset(memset)");
            hr = ctx->dev->xfer.cmd_list->Reset(ctx->dev->xfer.cmd_alloc.Get(), nullptr);
            DX12_CHECK(hr, "xfer cmd_list Reset(memset)");

            size_t tensor_offset = dx12_tensor_offset(tensor) + offset;
            ctx->dev->xfer.cmd_list->CopyBufferRegion(ctx->resource.Get(), tensor_offset,
                                                       ctx->dev->xfer.upload_staging.Get(), 0, size);
            ctx->dev->xfer.cmd_list->Close();

            ID3D12CommandList * lists[] = { ctx->dev->xfer.cmd_list.Get() };
            ctx->dev->compute_queue->ExecuteCommandLists(1, lists);
            ctx->dev->xfer.fence_value++;
            ctx->dev->compute_queue->Signal(ctx->dev->xfer.fence.Get(), ctx->dev->xfer.fence_value);
            ctx->dev->xfer_wait();
        },
        /* .set_tensor    = */ [](ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                  const void * data, size_t offset, size_t size) {
            auto * ctx = (dx12_buffer_context *)buffer->context;
            if (size == 0) return;

            g_tls_device = ctx->dev->device.Get();
            
            // CRITICAL: Ensure compute command list is closed before transfer
            // The scheduler may call set_tensor between graph splits
            // We need all compute work to be submitted first
            
            ctx->dev->init_xfer();
            ctx->dev->xfer_wait(); // wait for any previous transfer
            ctx->dev->xfer_ensure_staging(size, 0);

            // Map upload staging, copy data
            void * mapped = nullptr;
            D3D12_RANGE read_range = { 0, 0 };
            HRESULT hr = ctx->dev->xfer.upload_staging->Map(0, &read_range, &mapped);
            DX12_CHECK(hr, "Map upload staging");
            memcpy(mapped, data, size);
            ctx->dev->xfer.upload_staging->Unmap(0, nullptr);

            // Reset and record copy command
            hr = ctx->dev->xfer.cmd_alloc->Reset();
            DX12_CHECK(hr, "xfer cmd_alloc Reset");
            hr = ctx->dev->xfer.cmd_list->Reset(ctx->dev->xfer.cmd_alloc.Get(), nullptr);
            DX12_CHECK(hr, "xfer cmd_list Reset");

            size_t tensor_offset = dx12_tensor_offset(tensor) + offset;
            ctx->dev->xfer.cmd_list->CopyBufferRegion(ctx->resource.Get(), tensor_offset,
                                                       ctx->dev->xfer.upload_staging.Get(), 0, size);
            ctx->dev->xfer.cmd_list->Close();

            ID3D12CommandList * lists[] = { ctx->dev->xfer.cmd_list.Get() };
            ctx->dev->compute_queue->ExecuteCommandLists(1, lists);
            ctx->dev->xfer.fence_value++;
            ctx->dev->compute_queue->Signal(ctx->dev->xfer.fence.Get(), ctx->dev->xfer.fence_value);
            ctx->dev->xfer_wait();
        },
        /* .get_tensor    = */ [](ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                  void * data, size_t offset, size_t size) {
            auto * ctx = (dx12_buffer_context *)buffer->context;
            if (size == 0) return;

            g_tls_device = ctx->dev->device.Get();
            ctx->dev->init_xfer();
            ctx->dev->xfer_wait(); // wait for any previous transfer
            ctx->dev->xfer_ensure_staging(0, size);

            size_t tensor_offset = dx12_tensor_offset(tensor) + offset;

            // Always check if device is already dead before attempting copy
            {
                HRESULT removed = ctx->dev->device->GetDeviceRemovedReason();
                if (FAILED(removed)) {
                    fprintf(stderr, "ggml-dx12: get_tensor: device already removed (0x%08lX) before copy! offset=%zu size=%zu\n",
                            (unsigned long)removed, tensor_offset, size);
                    fflush(stderr);
                }
            }

            // Diagnostic: log readback parameters
            static bool dx12_diag_gt = (getenv("GGML_DX12_DIAG") != nullptr);
            if (dx12_diag_gt && size > 100000) {
                fprintf(stderr, "ggml-dx12: get_tensor: offset=%zu size=%zu buf_size=%zu ne=(%lld,%lld,%lld,%lld)\n",
                        tensor_offset, size, ctx->size,
                        (long long)tensor->ne[0], (long long)tensor->ne[1],
                        (long long)tensor->ne[2], (long long)tensor->ne[3]);
                fflush(stderr);
            }

            // Reset and record copy command
            HRESULT hr = ctx->dev->xfer.cmd_alloc->Reset();
            DX12_CHECK(hr, "xfer cmd_alloc Reset(get)");
            hr = ctx->dev->xfer.cmd_list->Reset(ctx->dev->xfer.cmd_alloc.Get(), nullptr);
            DX12_CHECK(hr, "xfer cmd_list Reset(get)");

            ctx->dev->xfer.cmd_list->CopyBufferRegion(ctx->dev->xfer.readback_staging.Get(), 0,
                                                       ctx->resource.Get(), tensor_offset, size);
            ctx->dev->xfer.cmd_list->Close();

            ID3D12CommandList * lists[] = { ctx->dev->xfer.cmd_list.Get() };
            ctx->dev->compute_queue->ExecuteCommandLists(1, lists);
            ctx->dev->xfer.fence_value++;
            ctx->dev->compute_queue->Signal(ctx->dev->xfer.fence.Get(), ctx->dev->xfer.fence_value);
            ctx->dev->xfer_wait();

            // Map readback and copy
            void * mapped = nullptr;
            D3D12_RANGE read_range = { 0, size };
            hr = ctx->dev->xfer.readback_staging->Map(0, &read_range, &mapped);
            DX12_CHECK(hr, "Map readback staging");
            memcpy(data, mapped, size);
            D3D12_RANGE written = { 0, 0 };
            ctx->dev->xfer.readback_staging->Unmap(0, &written);

            // NaN check: scan ALL readback tensors for NaN (early detection)
            if (dx12_diag_gt && size >= 4) {
                static int nan_report_count = 0;
                if (nan_report_count < 5) {
                    const float * fp = (const float *)data;
                    size_t n_floats = size / 4;
                    int nan_count = 0;
                    size_t first_nan_idx = 0;
                    for (size_t i = 0; i < n_floats; i++) {
                        if (isnan(fp[i])) {
                            if (nan_count == 0) first_nan_idx = i;
                            nan_count++;
                        }
                    }
                    if (nan_count > 0) {
                        nan_report_count++;
                        fprintf(stderr, "ggml-dx12: NaN DETECTED in get_tensor! offset=%zu size=%zu ne=(%lld,%lld,%lld,%lld) "
                                "nan_count=%d/%zu first_nan@%zu val_at[0]=%.6g buf_size=%zu\n",
                                tensor_offset, size,
                                (long long)tensor->ne[0], (long long)tensor->ne[1],
                                (long long)tensor->ne[2], (long long)tensor->ne[3],
                                nan_count, n_floats, first_nan_idx,
                                fp[0], ctx->size);
                        // Print a few values around first NaN
                        size_t start = first_nan_idx > 4 ? first_nan_idx - 4 : 0;
                        fprintf(stderr, "  values around first NaN [%zu..%zu]: ", start, start+7);
                        for (size_t i = start; i < start + 8 && i < n_floats; i++) {
                            fprintf(stderr, "%.4g ", fp[i]);
                        }
                        fprintf(stderr, "\n");
                        fflush(stderr);
                    }
                }
            }
            if (dx12_diag_gt && tensor->ne[0] > 100000 && tensor->ne[1] == 1 && size == tensor->ne[0] * 4) {
                static int logits_diag_count = 0;
                if (logits_diag_count < 3) {
                    logits_diag_count++;
                    const float * logits = (const float *)data;
                    int64_t n = tensor->ne[0];
                    // Find top-5
                    int top_ids[5] = {0,0,0,0,0};
                    float top_vals[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
                    float sum = 0.0f;
                    int nan_count = 0, inf_count = 0, zero_count = 0;
                    for (int64_t i = 0; i < n; i++) {
                        float v = logits[i];
                        if (isnan(v)) { nan_count++; continue; }
                        if (isinf(v)) { inf_count++; continue; }
                        if (v == 0.0f) zero_count++;
                        sum += v;
                        for (int j = 0; j < 5; j++) {
                            if (v > top_vals[j]) {
                                for (int k = 4; k > j; k--) { top_ids[k] = top_ids[k-1]; top_vals[k] = top_vals[k-1]; }
                                top_ids[j] = (int)i;
                                top_vals[j] = v;
                                break;
                            }
                        }
                    }
                    fprintf(stderr, "ggml-dx12: LOGITS DIAG #%d: n=%lld sum=%.4f mean=%.6f zeros=%d nans=%d infs=%d\n",
                            logits_diag_count, (long long)n, sum, sum/n, zero_count, nan_count, inf_count);
                    fprintf(stderr, "  top5: [%d]=%.4f [%d]=%.4f [%d]=%.4f [%d]=%.4f [%d]=%.4f\n",
                            top_ids[0], top_vals[0], top_ids[1], top_vals[1],
                            top_ids[2], top_vals[2], top_ids[3], top_vals[3],
                            top_ids[4], top_vals[4]);
                    // Also print first 8 and last 8 values
                    fprintf(stderr, "  first8: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                            logits[0], logits[1], logits[2], logits[3],
                            logits[4], logits[5], logits[6], logits[7]);
                    fprintf(stderr, "  last8:  %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                            logits[n-8], logits[n-7], logits[n-6], logits[n-5],
                            logits[n-4], logits[n-3], logits[n-2], logits[n-1]);
                    fflush(stderr);
                }
            }
        },
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ nullptr,
        /* .clear         = */ [](ggml_backend_buffer_t buffer, uint8_t value) {
            auto * ctx = (dx12_buffer_context *)buffer->context;
            if (!ctx->resource || ctx->size == 0) return;

            ctx->dev->init_xfer();
            ctx->dev->xfer_wait();

            const size_t chunk = 16 * 1024 * 1024;
            ctx->dev->xfer_ensure_staging(std::min(ctx->size, chunk), 0);

            void * mapped = nullptr;
            D3D12_RANGE read_range = { 0, 0 };
            HRESULT hr = ctx->dev->xfer.upload_staging->Map(0, &read_range, &mapped);
            DX12_CHECK(hr, "Map(clear staging)");
            memset(mapped, value, std::min(ctx->size, chunk));
            ctx->dev->xfer.upload_staging->Unmap(0, nullptr);

            hr = ctx->dev->xfer.cmd_alloc->Reset();
            DX12_CHECK(hr, "xfer cmd_alloc Reset(clear)");
            hr = ctx->dev->xfer.cmd_list->Reset(ctx->dev->xfer.cmd_alloc.Get(), nullptr);
            DX12_CHECK(hr, "xfer cmd_list Reset(clear)");

            for (size_t off = 0; off < ctx->size; off += chunk) {
                size_t copy_size = std::min(chunk, ctx->size - off);
                ctx->dev->xfer.cmd_list->CopyBufferRegion(ctx->resource.Get(), off,
                                                           ctx->dev->xfer.upload_staging.Get(), 0, copy_size);
            }
            ctx->dev->xfer.cmd_list->Close();

            ID3D12CommandList * lists[] = { ctx->dev->xfer.cmd_list.Get() };
            ctx->dev->compute_queue->ExecuteCommandLists(1, lists);
            ctx->dev->xfer.fence_value++;
            ctx->dev->compute_queue->Signal(ctx->dev->xfer.fence.Get(), ctx->dev->xfer.fence_value);
            ctx->dev->xfer_wait();
        },
        /* .reset         = */ nullptr,
    };

    return ggml_backend_buffer_init(buft, iface, ctx, size);
}

static size_t dx12_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 256; // D3D12 requires 256-byte alignment for constant buffers; good default
}

static size_t dx12_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // D3D12 buffer max is 4 GB on most hardware
    return (size_t)4 * 1024 * 1024 * 1024 - 1;
}

static bool dx12_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i dx12_buffer_type_interface = {
    /* .get_name      = */ dx12_buft_get_name,
    /* .alloc_buffer  = */ dx12_buft_alloc_buffer,
    /* .get_alignment = */ dx12_buft_get_alignment,
    /* .get_max_size  = */ dx12_buft_get_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host       = */ dx12_buft_is_host,
};

static ggml_backend_buffer_type g_dx12_buffer_types[GGML_DX12_MAX_DEVICES];

// ---------------------------------------------------------------------------
// Supported ops check
// ---------------------------------------------------------------------------

static bool dx12_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONCAT:
        case GGML_OP_REPEAT:
        case GGML_OP_ROPE:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_DIAG_MASK_INF:
            // Only support F32 output and F32 sources
            if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) {
                // Check source types too — our shaders only handle F32 data
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    if (op->src[s] && op->src[s]->type != GGML_TYPE_F32 &&
                        op->src[s]->type != GGML_TYPE_F16 &&
                        op->src[s]->type != GGML_TYPE_I32) {
                        return false;
                    }
                }
                return true;
            }
            return false;

        case GGML_OP_SET_ROWS:
            // KV cache writes: src0 is F32, dst can be F16 or F32
            if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) {
                return true;
            }
            return false;

        case GGML_OP_GLU: {
            // Gated Linear Unit: supports SWIGLU, REGLU, GEGLU etc.
            enum ggml_glu_op glu_op = (enum ggml_glu_op)op->op_params[0];
            switch (glu_op) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) {
                        // src0 must be F32 or F16
                        if (op->src[0] && op->src[0]->type != GGML_TYPE_F32 &&
                            op->src[0]->type != GGML_TYPE_F16) return false;
                        // src1 (if present) must be F32 or F16
                        if (op->src[1] && op->src[1]->type != GGML_TYPE_F32 &&
                            op->src[1]->type != GGML_TYPE_F16) return false;
                        return true;
                    }
                    return false;
                default:
                    return false;
            }
        }

        case GGML_OP_UNARY: {
            // Only support specific unary ops that have shaders
            enum ggml_unary_op uop = ggml_get_unary_op(op);
            switch (uop) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_SIGMOID:
                    if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) {
                        return true;
                    }
                    return false;
                default:
                    return false;
            }
        }

        case GGML_OP_MUL_MAT:
            if (op->type != GGML_TYPE_F32) return false;
            if (op->src[0]) {
                ggml_type t = op->src[0]->type;
                if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 &&
                    t != GGML_TYPE_Q2_K && t != GGML_TYPE_Q3_K &&
                    t != GGML_TYPE_Q4_K && t != GGML_TYPE_Q5_K && t != GGML_TYPE_Q6_K &&
                    t != GGML_TYPE_Q4_0 && t != GGML_TYPE_Q4_1 &&
                    t != GGML_TYPE_Q5_0 && t != GGML_TYPE_Q5_1 &&
                    t != GGML_TYPE_Q8_0 && t != GGML_TYPE_Q8_1) return false;
            }
            if (op->src[1] && op->src[1]->type != GGML_TYPE_F32) return false;
            return true;

        case GGML_OP_GET_ROWS:
            if (op->type != GGML_TYPE_F32) return false;
            if (op->src[0]) {
                ggml_type t = op->src[0]->type;
                if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 &&
                    t != GGML_TYPE_Q2_K && t != GGML_TYPE_Q3_K &&
                    t != GGML_TYPE_Q4_K && t != GGML_TYPE_Q5_K && t != GGML_TYPE_Q6_K &&
                    t != GGML_TYPE_Q4_0 && t != GGML_TYPE_Q4_1 &&
                    t != GGML_TYPE_Q5_0 && t != GGML_TYPE_Q5_1 &&
                    t != GGML_TYPE_Q8_0 && t != GGML_TYPE_Q8_1) return false;
            }
            return true;

        case GGML_OP_FLASH_ATTN_EXT:
            // Re-enabled: NaN was in model head (dispatches 1598-1608), not attention.
            // FA is MQA-safe and handles attention softcapping internally.
            if (op->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_F16) {
                return true;
            }
            return false;

        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Graph compute: dispatch shaders for a compute graph
// ---------------------------------------------------------------------------

static void dx12_fill_params(const struct ggml_tensor * tensor, dx12_shader_params & p) {
    memset(&p, 0, sizeof(p));

    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];

    if (src0) {
        p.ne00 = (uint32_t)src0->ne[0]; p.ne01 = (uint32_t)src0->ne[1];
        p.ne02 = (uint32_t)src0->ne[2]; p.ne03 = (uint32_t)src0->ne[3];
        GGML_ASSERT(src0->nb[1] <= UINT32_MAX && "src0 nb1 exceeds 4GB");
        GGML_ASSERT(src0->nb[2] <= UINT32_MAX && "src0 nb2 exceeds 4GB");
        GGML_ASSERT(src0->nb[3] <= UINT32_MAX && "src0 nb3 exceeds 4GB");
        p.nb00 = (uint32_t)src0->nb[0]; p.nb01 = (uint32_t)src0->nb[1];
        p.nb02 = (uint32_t)src0->nb[2]; p.nb03 = (uint32_t)src0->nb[3];
        uint64_t off = dx12_tensor_offset(src0);
        GGML_ASSERT(off <= UINT32_MAX && "src0 offset exceeds 4GB");
        p.src0_offset = (uint32_t)off;
        p.src0_esize  = (uint32_t)ggml_type_size(src0->type);
    }
    if (src1) {
        p.ne10 = (uint32_t)src1->ne[0]; p.ne11 = (uint32_t)src1->ne[1];
        p.ne12 = (uint32_t)src1->ne[2]; p.ne13 = (uint32_t)src1->ne[3];
        p.nb10 = (uint32_t)src1->nb[0]; p.nb11 = (uint32_t)src1->nb[1];
        p.nb12 = (uint32_t)src1->nb[2]; p.nb13 = (uint32_t)src1->nb[3];
        uint64_t off = dx12_tensor_offset(src1);
        GGML_ASSERT(off <= UINT32_MAX && "src1 offset exceeds 4GB");
        p.src1_offset = (uint32_t)off;
        p.src1_esize  = (uint32_t)ggml_type_size(src1->type);
    }

    uint64_t dst_off = dx12_tensor_offset(tensor);
    GGML_ASSERT(dst_off <= UINT32_MAX && "dst offset exceeds 4GB");
    p.ne0 = (uint32_t)tensor->ne[0]; p.ne1 = (uint32_t)tensor->ne[1];
    p.ne2 = (uint32_t)tensor->ne[2]; p.ne3 = (uint32_t)tensor->ne[3];
    p.nb0 = (uint32_t)tensor->nb[0]; p.nb1 = (uint32_t)tensor->nb[1];
    p.nb2 = (uint32_t)tensor->nb[2]; p.nb3 = (uint32_t)tensor->nb[3];
    p.dst_offset = (uint32_t)dst_off;
    p.dst_esize  = (uint32_t)ggml_type_size(tensor->type);

    // Copy op_params — for FLASH_ATTN_EXT, repurpose to carry src2 + mask info
    if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
        const struct ggml_tensor * src2 = tensor->src[2];
        const struct ggml_tensor * mask = tensor->src[3];
        float scale = 0.0f;
        memcpy(&scale, tensor->op_params, sizeof(float));

        p.op_params[0] = src2 ? (uint32_t)dx12_tensor_offset(src2) : 0; // src2_offset
        p.op_params[1] = src2 ? (uint32_t)src2->nb[0] : 0;             // src2_nb0
        p.op_params[2] = src2 ? (uint32_t)src2->nb[1] : 0;             // src2_nb1
        p.op_params[3] = src2 ? (uint32_t)src2->nb[2] : 0;             // src2_nb2
        p.op_params[4] = src2 ? (uint32_t)src2->nb[3] : 0;             // src2_nb3
        p.op_params[5] = src2 ? (uint32_t)ggml_type_size(src2->type) : 4; // src2_esize
        memcpy(&p.op_params[6], &scale, sizeof(float));                 // scale
        p.op_params[7] = (uint32_t)src1->ne[2];                        // n_kv_heads

        // Mask (src3) parameters
        p.op_params[8]  = mask ? 1u : 0u;                                // has_mask
        p.op_params[9]  = mask ? (uint32_t)dx12_tensor_offset(mask) : 0; // mask_offset
        p.op_params[10] = mask ? (uint32_t)mask->nb[1] : 0;              // mask_nb1
        p.op_params[11] = mask ? (uint32_t)mask->nb[2] : 0;              // mask_nb2
        p.op_params[12] = mask ? (uint32_t)mask->nb[3] : 0;              // mask_nb3
        p.op_params[13] = mask ? (uint32_t)mask->ne[2] : 1;              // mask_ne2
        p.op_params[14] = mask ? (uint32_t)mask->ne[3] : 1;              // mask_ne3
    } else if (tensor->op == GGML_OP_SOFT_MAX) {
        // SOFT_MAX: op_params layout:
        //   [0] scale (float)
        //   [1] max_bias (float)
        //   [2] m0 (float, pre-computed ALiBi base for h < n_head_log2)
        //   [3] m1 (float, pre-computed ALiBi base for h >= n_head_log2)
        //   [4] n_head_log2 (uint)
        //   [5] has_sinks (uint, 1 if src2 present)
        //   [6] src2_offset (uint)
        memcpy(p.op_params, tensor->op_params, 2 * sizeof(uint32_t)); // scale, max_bias

        float max_bias = 0.0f;
        memcpy(&max_bias, (float *)tensor->op_params + 1, sizeof(float));

        const uint32_t n_head      = p.ne02;
        const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2((double)n_head));

        float m0 = powf(2.0f, -(max_bias       ) / (float)n_head_log2);
        float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

        memcpy(&p.op_params[2], &m0, sizeof(float));
        memcpy(&p.op_params[3], &m1, sizeof(float));
        p.op_params[4] = n_head_log2;

        const struct ggml_tensor * src2 = tensor->src[2];
        p.op_params[5] = src2 ? 1u : 0u;
        p.op_params[6] = src2 ? (uint32_t)dx12_tensor_offset(src2) : 0u;
    } else {
        static_assert(sizeof(tensor->op_params) >= sizeof(p.op_params), "op_params size mismatch");
        memcpy(p.op_params, tensor->op_params, sizeof(p.op_params));
    }
}

static ID3D12Resource * dx12_get_resource(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) return nullptr;
    auto * ctx = (dx12_buffer_context *)tensor->buffer->context;
    return ctx ? ctx->resource.Get() : nullptr;
}

static ggml_status dx12_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * bctx = (dx12_backend_context *)backend->context;

    g_tls_device = bctx->dev->device.Get();

    // Run auto-tuning on first graph compute
    if (!bctx->dev->tuning_done) {
        bctx->dev->run_autotune();
    }

    bctx->ensure_cmd_list_open();

    // Profiling: profile only actual generation graphs (M=1 in MUL_MATs)
    static bool profiling = (getenv("DX12_PROFILE") != nullptr);
    static int profile_graph = 0;
    static int gen_graph = 0;
    profile_graph++;

    // Detect if this is a prompt processing graph (M > 1 in MUL_MATs)
    bool is_prompt = false;
    for (int j = 0; j < std::min(cgraph->n_nodes, 30); j++) {
        struct ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_MUL_MAT && n->ne[1] > 1) {
            is_prompt = true;
            break;
        }
    }
    if (!is_prompt) gen_graph++;
    // Profile the 3rd-5th actual generation graphs (skip warmup/reserve)
    bool do_profile = profiling && !is_prompt && gen_graph >= 3 && gen_graph <= 5;
    std::map<std::string, double> op_times;

    int dispatch_weight = 0;
    // AMD RDNA 3 workaround: flush threshold for prompt phase.
    // Lower values = more frequent CL splits = more correct but slower.
    // Configurable via GGML_DX12_FLUSH_INTERVAL env var (default: 16).
    static int TDR_FLUSH_THRESHOLD = 0;
    if (TDR_FLUSH_THRESHOLD == 0) {
        const char * env = getenv("GGML_DX12_FLUSH_INTERVAL");
        TDR_FLUSH_THRESHOLD = env ? atoi(env) : 16;
        if (TDR_FLUSH_THRESHOLD <= 0) TDR_FLUSH_THRESHOLD = 16;
    }

    // Track unsynced tensor writes for smart barrier insertion
    std::unordered_set<uintptr_t> unsynced_writes;

    static bool dx12_diag = (getenv("GGML_DX12_DIAG") != nullptr);

    // Track wall time for TDR yield: GPU must periodically yield to DWM
    // to prevent display-level TDR from cumulative compute monopolization.
    // (WSL2 has no DWM, so this is Windows-only)
#ifdef _WIN32
    LARGE_INTEGER tdr_t0, tdr_freq;
    QueryPerformanceFrequency(&tdr_freq);
    QueryPerformanceCounter(&tdr_t0);
    static constexpr double TDR_YIELD_MS = 800.0;  // yield to DWM every 800ms
#endif

    // Prompt diagnostics gated under GGML_DX12_DIAG
    static int prompt_eval_count = 0;
    bool trace_prompt = false;
    if (is_prompt && dx12_diag) {
        trace_prompt = true;
        prompt_eval_count++;
        fprintf(stderr, "ggml-dx12: PROMPT GRAPH #%d: n_nodes=%d, cl_open=%d\n",
                prompt_eval_count, cgraph->n_nodes, (int)bctx->cmd_list_open);
        fflush(stderr);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_NONE ||
            node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            // Propagate unsynced status through view/reshape aliases
            for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                if (unsynced_writes.count((uintptr_t)node->src[s])) {
                    unsynced_writes.insert((uintptr_t)node);
                    break;
                }
            }
            continue;
        }

        // Build pipeline key
        dx12_pipeline_key key = {};
        key.op       = node->op;
        key.dst_type = node->type;
        key.src0_type = node->src[0] ? node->src[0]->type : GGML_TYPE_F32;
        key.src1_type = node->src[1] ? node->src[1]->type : GGML_TYPE_F32;

        // Op fusion: RMS_NORM + MUL → rms_norm_mul (single dispatch)
        // Also detects ADD + RMS_NORM + MUL → add_rms_norm_mul (triple fusion)
        struct ggml_tensor * fused_mul_node = nullptr;
        struct ggml_tensor * fused_add_rms_node = nullptr;  // the ADD preceding RMS_NORM+MUL
        struct ggml_tensor * fused_rms_node = nullptr;      // the RMS_NORM in triple fusion

        // Try ADD + RMS_NORM + MUL triple fusion first
        static bool no_fusion = (getenv("GGML_DX12_NO_FUSION") != nullptr);
        bool skip_fusion = no_fusion || (is_prompt && getenv("GGML_DX12_NO_PROMPT_FUSION"));
        if (!skip_fusion && node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
            struct ggml_tensor * rms = cgraph->nodes[i + 1];
            struct ggml_tensor * mul = cgraph->nodes[i + 2];
            if (rms->op == GGML_OP_RMS_NORM && mul->op == GGML_OP_MUL &&
                rms->src[0] == node && mul->src[0] == rms &&
                node->type == GGML_TYPE_F32 && rms->type == GGML_TYPE_F32 &&
                mul->type == GGML_TYPE_F32 && mul->src[1]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(mul->src[1])) {
                fused_add_rms_node = node;
                fused_rms_node = rms;
                fused_mul_node = mul;
                key.op = GGML_OP_RMS_NORM;
                key.flags = 3;  // flags=3 means fused add_rms_norm_mul
            }
        }

        // Fallback: try RMS_NORM + MUL + ROPE triple fusion, or RMS_NORM + MUL double fusion
        struct ggml_tensor * fused_rope_after_rms = nullptr;
        if (!skip_fusion && !fused_add_rms_node && node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes) {
            struct ggml_tensor * next = cgraph->nodes[i + 1];
            if (next->op == GGML_OP_MUL && next->src[0] == node) {
                // Check for RMS_NORM + MUL + ROPE triple fusion
                if (i + 2 < cgraph->n_nodes) {
                    struct ggml_tensor * rope = cgraph->nodes[i + 2];
                    int mode = rope->op == GGML_OP_ROPE ? ((const int32_t *)rope->op_params)[2] : -1;
                    if (rope->op == GGML_OP_ROPE && rope->src[0] == next &&
                        ggml_is_contiguous(next) && ggml_is_contiguous(rope) &&
                        next->ne[0] <= 1024 &&
                        (mode == 0 || mode == 2)) {  // NORMAL or NEOX
                        fused_mul_node = next;
                        fused_rope_after_rms = rope;
                        key.op = GGML_OP_RMS_NORM;
                        key.flags = 7;  // flags=7 means fused rms_norm_mul_rope
                    }
                }
                // If triple fusion didn't trigger, use double fusion
                if (!fused_rope_after_rms) {
                    fused_mul_node = next;
                    key.op = GGML_OP_RMS_NORM;
                    key.flags = 2;  // flags=2 means fused rms_norm_mul
                }
            }
        }

        // Op fusion: ROPE + VIEW + SET_ROWS → fused rope_set_rows
        // Eliminates 2 dispatches per KV cache write
        struct ggml_tensor * fused_rope_set_rows = nullptr;  // the SET_ROWS node
        struct ggml_tensor * fused_rope_view = nullptr;       // the VIEW node
        if (!skip_fusion && node->op == GGML_OP_ROPE && i + 2 < cgraph->n_nodes && !fused_add_rms_node) {
            struct ggml_tensor * view = cgraph->nodes[i + 1];
            struct ggml_tensor * set_rows = cgraph->nodes[i + 2];
            if (view->op == GGML_OP_VIEW && set_rows->op == GGML_OP_SET_ROWS &&
                view->src[0] == node && set_rows->src[0] == view &&
                node->src[0]->ne[3] == 1 &&
                (set_rows->type == GGML_TYPE_F32 || set_rows->type == GGML_TYPE_F16) &&
                ggml_is_contiguous(view) &&
                view->ne[0] == node->ne[0] * node->ne[1]) {
                fused_rope_set_rows = set_rows;
                fused_rope_view = view;
                key.op = GGML_OP_ROPE;
                key.flags = 6;  // flags=6 means fused rope_set_rows
            }
        }

        // For unary ops, store the unary op type in flags
        if (node->op == GGML_OP_UNARY) {
            key.flags = (uint32_t)ggml_get_unary_op(node);
        }
        // For MUL_MAT with M=1, use matvec pipeline (flags=1, or flags=5 for 256-thread auto-tuned)
        // Only for types that have matvec shaders
        bool is_matvec_dispatch = false;
        if (node->op == GGML_OP_MUL_MAT && node->ne[1] == 1 && node->src[0]) {
            ggml_type t = node->src[0]->type;
            if (t == GGML_TYPE_F16 || t == GGML_TYPE_F32 ||
                t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q5_0 ||
                t == GGML_TYPE_Q8_0) {
                key.flags = 1;
                // Multi-row matvec for Q4_K (2 rows per group, halves activation traffic)
                if (t == GGML_TYPE_Q4_K) key.flags = 2;
                // Auto-tuning: use alternate variant if benchmarked as faster
                if (t == GGML_TYPE_Q5_0 && bctx->dev->q5_0_use_256) key.flags = 5;
                if (t == GGML_TYPE_Q8_0 && bctx->dev->q8_0_use_256) key.flags = 5;
                if (t == GGML_TYPE_Q6_K && bctx->dev->q6k_use_32)   key.flags = 5;
                if ((t == GGML_TYPE_F16 || t == GGML_TYPE_F32) && bctx->dev->f16_use_load4) key.flags = 5;
                is_matvec_dispatch = true;
            }
        }
        // For batch MUL_MAT (M > 1), use register-blocked tiled path (flags=4)
        // for types that have wmma shaders
        if (node->op == GGML_OP_MUL_MAT && node->ne[1] > 1 && node->src[0]) {
            ggml_type t = node->src[0]->type;
            if (t == GGML_TYPE_F16 || t == GGML_TYPE_F32 ||
                t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                t == GGML_TYPE_Q6_K) {
                key.flags = 4;
            }
        }

        // Op fusion: MUL_MAT(M=1) + ADD → matvec with fused bias add
        struct ggml_tensor * fused_bias_add = nullptr;
        struct ggml_tensor * fused_bias_tensor = nullptr;
        if (!skip_fusion && is_matvec_dispatch && i + 1 < cgraph->n_nodes) {
            struct ggml_tensor * next = cgraph->nodes[i + 1];
            if (next->op == GGML_OP_ADD) {
                struct ggml_tensor * bias = nullptr;
                if (next->src[0] == node) bias = next->src[1];
                else if (next->src[1] == node) bias = next->src[0];
                // Bias must be F32, same shape as output, contiguous
                if (bias && bias->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
                    bias->ne[0] == node->ne[0] && ggml_is_contiguous(bias)) {
                    fused_bias_add = next;
                    fused_bias_tensor = bias;
                }
            }
        }

        // Look up or create pipeline
        dx12_pipeline * pipeline = bctx->dev->get_or_create_pipeline(key);
        if (!pipeline || !pipeline->pso) {
            continue;
        }

        // Set pipeline state — skip if unchanged from previous dispatch
        ID3D12RootSignature * root_sig = bctx->dev->common_root_sig.Get();
        if (root_sig != bctx->last_root_sig) {
            bctx->cmd_list->SetComputeRootSignature(root_sig);
            bctx->last_root_sig = root_sig;
        }
        if (pipeline->pso.Get() != bctx->last_pso) {
            bctx->cmd_list->SetPipelineState(pipeline->pso.Get());
            bctx->last_pso = pipeline->pso.Get();
        }

        // Set root constants (shader params)
        dx12_shader_params params;
        if (fused_add_rms_node) {
            // Triple fusion: ADD + RMS_NORM + MUL
            // node = ADD, fused_rms_node = RMS_NORM, fused_mul_node = MUL
            dx12_fill_params(node, params);  // src0 and src1 are ADD's inputs
            // dst shape/strides from MUL's output
            params.ne0 = (uint32_t)fused_mul_node->ne[0]; params.ne1 = (uint32_t)fused_mul_node->ne[1];
            params.ne2 = (uint32_t)fused_mul_node->ne[2]; params.ne3 = (uint32_t)fused_mul_node->ne[3];
            params.nb0 = (uint32_t)fused_mul_node->nb[0]; params.nb1 = (uint32_t)fused_mul_node->nb[1];
            params.nb2 = (uint32_t)fused_mul_node->nb[2]; params.nb3 = (uint32_t)fused_mul_node->nb[3];
            params.dst_offset = (uint32_t)dx12_tensor_offset(fused_mul_node);
            params.dst_esize = (uint32_t)ggml_type_size(fused_mul_node->type);
            // op_params: ADD dst offset, weight offset, epsilon, ADD dst esize
            params.op_params[0] = (uint32_t)dx12_tensor_offset(node);  // ADD's output offset
            params.op_params[1] = (uint32_t)dx12_tensor_offset(fused_mul_node->src[1]);  // weight offset
            float eps = 0.0f;
            memcpy(&eps, fused_rms_node->op_params, sizeof(float));
            memcpy(&params.op_params[2], &eps, sizeof(uint32_t));
            params.op_params[3] = (uint32_t)ggml_type_size(node->type);  // ADD dst esize
        } else if (fused_mul_node) {
            // For fused rms_norm_mul or rms_norm_mul_rope
            dx12_fill_params(node, params);
            // Override src1 with MUL's weight tensor
            const struct ggml_tensor * wt = fused_mul_node->src[1];
            if (wt) {
                params.ne10 = (uint32_t)wt->ne[0]; params.ne11 = (uint32_t)wt->ne[1];
                params.ne12 = (uint32_t)wt->ne[2]; params.ne13 = (uint32_t)wt->ne[3];
                params.nb10 = (uint32_t)wt->nb[0]; params.nb11 = (uint32_t)wt->nb[1];
                params.nb12 = (uint32_t)wt->nb[2]; params.nb13 = (uint32_t)wt->nb[3];
                params.src1_offset = (uint32_t)dx12_tensor_offset(wt);
                params.src1_esize = (uint32_t)ggml_type_size(wt->type);
            }
            if (fused_rope_after_rms) {
                // RMS_NORM + MUL + ROPE: dst is ROPE's output
                params.ne0 = (uint32_t)fused_rope_after_rms->ne[0]; params.ne1 = (uint32_t)fused_rope_after_rms->ne[1];
                params.ne2 = (uint32_t)fused_rope_after_rms->ne[2]; params.ne3 = (uint32_t)fused_rope_after_rms->ne[3];
                params.nb0 = (uint32_t)fused_rope_after_rms->nb[0]; params.nb1 = (uint32_t)fused_rope_after_rms->nb[1];
                params.nb2 = (uint32_t)fused_rope_after_rms->nb[2]; params.nb3 = (uint32_t)fused_rope_after_rms->nb[3];
                params.dst_offset = (uint32_t)dx12_tensor_offset(fused_rope_after_rms);
                params.dst_esize = (uint32_t)ggml_type_size(fused_rope_after_rms->type);
                // Copy ROPE's op_params (n_dims, mode, freq_base, etc.) into our op_params
                // op_params[0] = epsilon (already set by fill_params from RMS_NORM node)
                // op_params[1..7] = ROPE params
                memcpy(&params.op_params[1], &fused_rope_after_rms->op_params[1], 7 * sizeof(uint32_t));
            } else {
                // Override dst with MUL's output
                params.ne0 = (uint32_t)fused_mul_node->ne[0]; params.ne1 = (uint32_t)fused_mul_node->ne[1];
                params.ne2 = (uint32_t)fused_mul_node->ne[2]; params.ne3 = (uint32_t)fused_mul_node->ne[3];
                params.nb0 = (uint32_t)fused_mul_node->nb[0]; params.nb1 = (uint32_t)fused_mul_node->nb[1];
                params.nb2 = (uint32_t)fused_mul_node->nb[2]; params.nb3 = (uint32_t)fused_mul_node->nb[3];
                params.dst_offset = (uint32_t)dx12_tensor_offset(fused_mul_node);
                params.dst_esize = (uint32_t)ggml_type_size(fused_mul_node->type);
            }
            params.dst_esize = (uint32_t)ggml_type_size(fused_mul_node->type);
        } else {
            dx12_fill_params(node, params);
        }

        // Fused bias add: set op_params and override dst to ADD's output
        if (fused_bias_tensor) {
            params.op_params[0] = 1;  // bias fusion flag
            params.op_params[1] = (uint32_t)dx12_tensor_offset(fused_bias_tensor);  // bias byte offset
            // Use ADD's output as destination
            params.dst_offset = (uint32_t)dx12_tensor_offset(fused_bias_add);
        }

        // Fused ROPE+SET_ROWS: override dst to SET_ROWS output, pass stride info
        if (fused_rope_set_rows) {
            // op_params[8] = set_rows_stride (elements per KV row, for flat indexing)
            params.op_params[8] = (uint32_t)(fused_rope_set_rows->nb[1] / ggml_type_size(fused_rope_set_rows->type));
            // op_params[9] = set_rows nb1 (byte stride between rows)
            params.op_params[9] = (uint32_t)fused_rope_set_rows->nb[1];
            // Override dst to SET_ROWS output (KV cache)
            params.ne0 = (uint32_t)fused_rope_set_rows->ne[0]; params.ne1 = (uint32_t)fused_rope_set_rows->ne[1];
            params.ne2 = (uint32_t)fused_rope_set_rows->ne[2]; params.ne3 = (uint32_t)fused_rope_set_rows->ne[3];
            params.nb0 = (uint32_t)fused_rope_set_rows->nb[0]; params.nb1 = (uint32_t)fused_rope_set_rows->nb[1];
            params.nb2 = (uint32_t)fused_rope_set_rows->nb[2]; params.nb3 = (uint32_t)fused_rope_set_rows->nb[3];
            params.dst_offset = (uint32_t)dx12_tensor_offset(fused_rope_set_rows);
            params.dst_esize = (uint32_t)ggml_type_size(fused_rope_set_rows->type);
        }

        // Upload root constants — only upload op_params for ops that need them
        static constexpr uint32_t BASE_PARAMS = 30;  // ne/nb/offsets/esizes = 30 DWORDs
        bool needs_op_params = (node->op == GGML_OP_SOFT_MAX || 
                                 node->op == GGML_OP_FLASH_ATTN_EXT || 
                                 node->op == GGML_OP_ROPE ||
                                 node->op == GGML_OP_RMS_NORM ||
                                 node->op == GGML_OP_NORM ||
                                 node->op == GGML_OP_GLU ||
                                 node->op == GGML_OP_SCALE ||
                                 fused_bias_tensor ||
                                 fused_add_rms_node);
        uint32_t num_constants = needs_op_params ? (uint32_t)(sizeof(params) / 4) : BASE_PARAMS;
        bctx->cmd_list->SetComputeRoot32BitConstants(0, num_constants, &params, 0);

        // Bind resources — for fused ops, use the fused node's resources
        ID3D12Resource * src0_res = dx12_get_resource(node->src[0]);
        ID3D12Resource * src1_res;
        if (fused_add_rms_node) {
            // Triple fusion: src1 stays as ADD's src1 (NOT MUL's weight)
            src1_res = dx12_get_resource(node->src[1]);
        } else if (fused_mul_node) {
            src1_res = dx12_get_resource(fused_mul_node->src[1]);
        } else {
            src1_res = dx12_get_resource(node->src[1]);
        }
        ID3D12Resource * dst_res;
        if (fused_rope_after_rms) {
            dst_res = dx12_get_resource(fused_rope_after_rms);
        } else if (fused_mul_node) {
            dst_res = dx12_get_resource(fused_mul_node);
        } else if (fused_bias_add) {
            dst_res = dx12_get_resource(fused_bias_add);
        } else if (fused_rope_set_rows) {
            dst_res = dx12_get_resource(fused_rope_set_rows);
        } else {
            dst_res = dx12_get_resource(node);
        }

        // Bind resources — cached for slots 1-3, always-bind fallback for slots 4/5
        if (src0_res) {
            D3D12_GPU_VIRTUAL_ADDRESS va = src0_res->GetGPUVirtualAddress();
            if (va != bctx->last_src0_va) {
                bctx->cmd_list->SetComputeRootShaderResourceView(1, va);
                bctx->last_src0_va = va;
            }
        }
        if (src1_res) {
            D3D12_GPU_VIRTUAL_ADDRESS va = src1_res->GetGPUVirtualAddress();
            if (va != bctx->last_src1_va) {
                bctx->cmd_list->SetComputeRootShaderResourceView(2, va);
                bctx->last_src1_va = va;
            }
        }
        if (dst_res) {
            D3D12_GPU_VIRTUAL_ADDRESS va = dst_res->GetGPUVirtualAddress();
            if (va != bctx->last_dst_va) {
                bctx->cmd_list->SetComputeRootUnorderedAccessView(3, va);
                bctx->last_dst_va = va;
            }
        }

        // Always bind slots 4/5: AMD RDNA 3 can hang if root descriptors point to
        // stale/freed memory when switching PSOs, even if the new shader doesn't use them.
        bool needs_src2 = (node->op == GGML_OP_FLASH_ATTN_EXT) || (node->op == GGML_OP_SOFT_MAX) || (fused_bias_tensor != nullptr) || (fused_add_rms_node != nullptr) || (fused_rope_after_rms != nullptr);
        bool needs_src3 = (node->op == GGML_OP_FLASH_ATTN_EXT) || (fused_rope_set_rows != nullptr);

        {
            D3D12_GPU_VIRTUAL_ADDRESS src2_va;
            if (needs_src2 || needs_src3) {
                ID3D12Resource * src2_res;
                if (fused_rope_after_rms) {
                    src2_res = dx12_get_resource(fused_rope_after_rms->src[1]);
                } else if (fused_add_rms_node) {
                    src2_res = dx12_get_resource(fused_mul_node->src[1]);
                } else if (fused_bias_tensor) {
                    src2_res = dx12_get_resource(fused_bias_tensor);
                } else {
                    src2_res = dx12_get_resource(node->src[2]);
                }
                src2_va = (src2_res ? src2_res : src0_res)->GetGPUVirtualAddress();
            } else {
                src2_va = src0_res ? src0_res->GetGPUVirtualAddress() : 0;
            }
            if (src2_va && src2_va != bctx->last_src2_va) {
                bctx->cmd_list->SetComputeRootShaderResourceView(4, src2_va);
                bctx->last_src2_va = src2_va;
            }
        }

        {
            D3D12_GPU_VIRTUAL_ADDRESS src3_va;
            if (needs_src3) {
                ID3D12Resource * src3_res;
                if (fused_rope_set_rows) {
                    src3_res = dx12_get_resource(fused_rope_set_rows->src[1]);
                } else {
                    src3_res = dx12_get_resource(node->src[3]);
                }
                src3_va = (src3_res ? src3_res : src0_res)->GetGPUVirtualAddress();
            } else {
                src3_va = src0_res ? src0_res->GetGPUVirtualAddress() : 0;
            }
            if (src3_va && src3_va != bctx->last_src3_va) {
                bctx->cmd_list->SetComputeRootShaderResourceView(5, src3_va);
                bctx->last_src3_va = src3_va;
            }
        }



        // Calculate dispatch dimensions
        uint32_t groups_x = 1, groups_y = 1, groups_z = 1;

        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                bool is_matvec = (node->ne[1] == 1); // M=1: single token generation

                if (is_matvec) {
                    // Matvec dispatch
                    uint32_t N = (uint32_t)node->ne[0];
                    uint32_t batches = (uint32_t)(node->ne[2] * node->ne[3]);

                    // Chunked dispatch for very large N to prevent TDR
                    // A single dispatch of 262K+ rows can exceed the Windows TDR timeout
                    static constexpr uint32_t MATVEC_CHUNK = 32768;
                    bool no_fusions = (!fused_bias_add && !fused_mul_node &&
                                       !fused_add_rms_node && !fused_rope_after_rms && !fused_rope_set_rows);

                    if (N > MATVEC_CHUNK && key.flags != 2 && no_fusions) {
                        uint32_t saved_ne0      = params.ne0;
                        uint32_t saved_ne01     = params.ne01;
                        uint32_t saved_src0_off = params.src0_offset;
                        uint32_t saved_dst_off  = params.dst_offset;

                        for (uint32_t cs = 0; cs < N; cs += MATVEC_CHUNK) {
                            uint32_t cr = std::min(MATVEC_CHUNK, N - cs);

                            params.ne0         = cr;
                            params.ne01        = cr;
                            params.src0_offset = saved_src0_off + cs * params.nb01;
                            params.dst_offset  = saved_dst_off  + cs * params.nb0;

                            bctx->cmd_list->SetComputeRoot32BitConstants(0, BASE_PARAMS, &params, 0);

                            if (dx12_diag && is_prompt) {
                                int src0t = node->src[0] ? (int)node->src[0]->type : -1;
                                fprintf(stderr, "ggml-dx12: dispatch #%d MUL_MAT(src0t=%d) chunk [%u..%u/%u] groups=(%u,1,%u)\n",
                                        i, src0t, cs, cs + cr, N, cr, batches);
                                fflush(stderr);
                            }

                            bctx->cmd_list->Dispatch(cr, 1, batches);

                            // Flush between chunks for TDR safety
                            if (cs + MATVEC_CHUNK < N) {
                                bctx->close_and_execute();
                                bctx->wait_for_gpu();
                                if (dx12_diag) {
                                    fprintf(stderr, "ggml-dx12: chunk flush OK after [%u..%u]\n", cs, cs + cr);
                                    fflush(stderr);
                                }
                                bctx->ensure_cmd_list_open();
                                // Re-bind on new command list
                                bctx->cmd_list->SetComputeRootSignature(root_sig);
                                bctx->cmd_list->SetPipelineState(pipeline->pso.Get());
                                if (src0_res)
                                    bctx->cmd_list->SetComputeRootShaderResourceView(1, src0_res->GetGPUVirtualAddress());
                                if (src1_res)
                                    bctx->cmd_list->SetComputeRootShaderResourceView(2, src1_res->GetGPUVirtualAddress());
                                if (dst_res)
                                    bctx->cmd_list->SetComputeRootUnorderedAccessView(3, dst_res->GetGPUVirtualAddress());
                            }
                        }

                        // Restore params
                        params.ne0         = saved_ne0;
                        params.ne01        = saved_ne01;
                        params.src0_offset = saved_src0_off;
                        params.dst_offset  = saved_dst_off;

                        unsynced_writes.insert((uintptr_t)node);  // no fusions, dst = node
                        dispatch_weight = 0;
                        continue;  // skip normal dispatch path
                    }

                    // Normal matvec dispatch (N <= MATVEC_CHUNK)
                    if (key.flags == 2) {
                        // Multi-row: 2 rows per thread group
                        groups_x = (N + 1) / 2;
                    } else {
                        groups_x = N;
                    }
                    // D3D12 dispatch limit: 65535 per axis
                    // Linearize into 2D: shader computes row = gid.y * 65535 + gid.x
                    if (groups_x > 65535) {
                        groups_y = (groups_x + 65534) / 65535;
                        groups_x = 65535;
                    } else {
                        groups_y = 1;
                    }
                    groups_z = batches;
                } else if (key.flags == 4) {
                    // Register-blocked tiled dispatch (32×32 tile) [numthreads(16,16,1)]
                    uint32_t N = (uint32_t)node->ne[0];
                    uint32_t M = (uint32_t)node->ne[1];
                    uint32_t batches = (uint32_t)(node->ne[2] * node->ne[3]);
                    groups_x = (N + 31) / 32;
                    groups_y = (M + 31) / 32;
                    groups_z = batches;
                } else if (node->src[0] && (node->src[0]->type == GGML_TYPE_Q2_K ||
                                            node->src[0]->type == GGML_TYPE_Q3_K)) {
                    // Cooperative batch matmul: 16 rows/group, 16 K-threads/row, TILE_M=4
                    uint32_t N = (uint32_t)node->ne[0];
                    uint32_t M = (uint32_t)node->ne[1];
                    uint32_t batches = (uint32_t)(node->ne[2] * node->ne[3]);
                    groups_x = (N + 15) / 16;
                    groups_y = (M + 3) / 4;
                    groups_z = batches;
                } else if (node->src[0] && (node->src[0]->type == GGML_TYPE_Q4_K ||
                                            node->src[0]->type == GGML_TYPE_Q5_K ||
                                            node->src[0]->type == GGML_TYPE_Q6_K ||
                                            node->src[0]->type == GGML_TYPE_Q4_0 ||
                                            node->src[0]->type == GGML_TYPE_Q4_1 ||
                                            node->src[0]->type == GGML_TYPE_Q5_0 ||
                                            node->src[0]->type == GGML_TYPE_Q5_1 ||
                                            node->src[0]->type == GGML_TYPE_Q8_0 ||
                                            node->src[0]->type == GGML_TYPE_Q8_1)) {
                    // Quantized flat shaders: 1 output per thread, 256 threads/group
                    uint32_t total = (uint32_t)(node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3]);
                    groups_x = (total + 255) / 256;
                } else {
                    // Tiled dispatch for F32/F16 [numthreads(16,16,1)]
                    uint32_t N = (uint32_t)node->ne[0];
                    uint32_t M = (uint32_t)node->ne[1];
                    uint32_t batches = (uint32_t)(node->ne[2] * node->ne[3]);
                    groups_x = (N + 15) / 16;
                    groups_y = (M + 15) / 16;
                    groups_z = batches;
                }
                break;
            }
            case GGML_OP_RMS_NORM:
            case GGML_OP_NORM:
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SUM_ROWS: {
                // Row-based ops: one thread group per row
                uint32_t total_rows = (uint32_t)(node->ne[1] * node->ne[2] * node->ne[3]);
                groups_x = total_rows;
                break;
            }
            case GGML_OP_ROPE: {
                // Process pairs
                uint32_t n_pairs = (uint32_t)(node->ne[0] / 2);
                uint32_t total = n_pairs * (uint32_t)node->ne[1] * (uint32_t)node->ne[2] * (uint32_t)node->ne[3];
                groups_x = (total + 255) / 256;
                break;
            }
            case GGML_OP_FLASH_ATTN_EXT: {
                // One thread group per query row per head per batch
                groups_x = (uint32_t)node->src[0]->ne[1]; // N_queries
                groups_y = (uint32_t)node->src[0]->ne[2]; // n_heads
                groups_z = (uint32_t)node->src[0]->ne[3]; // batch
                break;
            }
            default: {
                // Element-wise: one thread per element
                uint32_t total_elements = (uint32_t)(ggml_nelements(node));
                groups_x = (total_elements + 255) / 256;
                break;
            }
        }

        // Override dispatch dimensions for triple fusion (ADD+RMS_NORM+MUL uses row-based dispatch)
        if (fused_add_rms_node) {
            uint32_t total_rows = (uint32_t)(fused_mul_node->ne[1] * fused_mul_node->ne[2] * fused_mul_node->ne[3]);
            groups_x = total_rows;
            groups_y = 1;
            groups_z = 1;
        }
        // RMS_NORM+MUL+ROPE also uses row-based dispatch
        if (fused_rope_after_rms) {
            uint32_t total_rows = (uint32_t)(fused_rope_after_rms->ne[1] * fused_rope_after_rms->ne[2] * fused_rope_after_rms->ne[3]);
            groups_x = total_rows;
            groups_y = 1;
            groups_z = 1;
        }

        if (do_profile) {
            // Flush and time each dispatch individually
            bctx->close_and_execute();
            bctx->wait_for_gpu();
            bctx->ensure_cmd_list_open();
            bctx->reset_binding_cache();
            // Re-bind all
            bctx->cmd_list->SetComputeRootSignature(bctx->dev->common_root_sig.Get());
            bctx->last_root_sig = bctx->dev->common_root_sig.Get();
            bctx->cmd_list->SetPipelineState(pipeline->pso.Get());
            bctx->last_pso = pipeline->pso.Get();
            if (src0_res) { bctx->cmd_list->SetComputeRootShaderResourceView(1, src0_res->GetGPUVirtualAddress()); bctx->last_src0_va = src0_res->GetGPUVirtualAddress(); }
            if (src1_res) { bctx->cmd_list->SetComputeRootShaderResourceView(2, src1_res->GetGPUVirtualAddress()); bctx->last_src1_va = src1_res->GetGPUVirtualAddress(); }
            if (dst_res)  { bctx->cmd_list->SetComputeRootUnorderedAccessView(3, dst_res->GetGPUVirtualAddress()); bctx->last_dst_va = dst_res->GetGPUVirtualAddress(); }
            // Bind src0 as fallback for src2/src3
            if (src0_res) {
                D3D12_GPU_VIRTUAL_ADDRESS fallback_va = src0_res->GetGPUVirtualAddress();
                bctx->cmd_list->SetComputeRootShaderResourceView(4, fallback_va); bctx->last_src2_va = fallback_va;
                bctx->cmd_list->SetComputeRootShaderResourceView(5, fallback_va); bctx->last_src3_va = fallback_va;
            }
            bctx->cmd_list->SetComputeRoot32BitConstants(0, sizeof(params) / 4, &params, 0);
        }

#ifdef _WIN32
        LARGE_INTEGER t0, t1, freq;
        if (do_profile) { QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&t0); }
#else
        std::chrono::steady_clock::time_point t0;
        if (do_profile) { t0 = std::chrono::steady_clock::now(); }
#endif

        // Determine the effective destination tensor (accounting for fusion)
        struct ggml_tensor * dst_tensor = fused_rope_after_rms ? fused_rope_after_rms :
                                          (fused_mul_node ? fused_mul_node : 
                                          (fused_bias_add ? fused_bias_add : 
                                          (fused_rope_set_rows ? fused_rope_set_rows : node)));

        // Dependency-tracked UAV barriers (like Vulkan's sync_buffers)
        // Only insert a barrier when the current dispatch reads a tensor that was
        // written by a previous unsynced dispatch. This eliminates ~40% of barriers.
        {
            bool need_barrier = false;
            
            // Always barrier before/after SET_ROWS, FA, and fused ROPE+SET_ROWS (KV cache views overlap)
            if (node->op == GGML_OP_SET_ROWS || node->op == GGML_OP_FLASH_ATTN_EXT || fused_rope_set_rows) {
                need_barrier = true;
            } else {
                // Check if current dispatch reads from any unsynced written tensor
                for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                    if (unsynced_writes.count((uintptr_t)node->src[s])) {
                        need_barrier = true;
                        break;
                    }
                }
                if (!need_barrier && fused_mul_node) {
                    for (int s = 0; s < GGML_MAX_SRC && fused_mul_node->src[s]; s++) {
                        if (unsynced_writes.count((uintptr_t)fused_mul_node->src[s])) {
                            need_barrier = true;
                            break;
                        }
                    }
                }
                if (unsynced_writes.count((uintptr_t)dst_tensor)) {
                    need_barrier = true;
                }
            }

            if (need_barrier) {
                D3D12_RESOURCE_BARRIER barrier = {};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.UAV.pResource = nullptr;
                bctx->cmd_list->ResourceBarrier(1, &barrier);
                unsynced_writes.clear();
            }

            // DEBUG: unconditional UAV barrier before every dispatch to test if
            // the dependency-tracked barriers are missing a case.
            static bool force_barriers = (getenv("GGML_DX12_BARRIER_ALL") != nullptr);
            if (force_barriers) {
                D3D12_RESOURCE_BARRIER barrier = {};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.UAV.pResource = nullptr;
                bctx->cmd_list->ResourceBarrier(1, &barrier);
                unsynced_writes.clear();
            }
        }

        // TDR diagnostic: print every dispatch during first prompt
        if (dx12_diag && is_prompt) {
            int src0t = node->src[0] ? (int)node->src[0]->type : -1;
            fprintf(stderr, "ggml-dx12: dispatch #%d %s(src0t=%d) groups=(%u,%u,%u) ne=(%lld,%lld,%lld,%lld)\n",
                    i, ggml_op_name(node->op), src0t, groups_x, groups_y, groups_z,
                    (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3]);
            fflush(stderr);
        }

        // D3D12 dispatch limit: 65535 per axis
        if (groups_x > 65535 || groups_y > 65535 || groups_z > 65535) {
            fprintf(stderr, "ggml-dx12: SKIP dispatch #%d %s — groups (%u,%u,%u) exceeds D3D12 limit 65535\n",
                    i, ggml_op_name(node->op), groups_x, groups_y, groups_z);
            continue;
        }

        bctx->cmd_list->Dispatch(groups_x, groups_y, groups_z);

        // GLU diagnostic: read back first few values of src0, src1, dst after first GLU dispatch
        if (dx12_diag && node->op == GGML_OP_GLU) {
            static bool glu_diag_done = false;
            if (!glu_diag_done) {
                glu_diag_done = true;
                // Flush current work so GLU dispatch completes
                bctx->close_and_execute();
                bctx->wait_for_gpu();

                fprintf(stderr, "ggml-dx12: GLU DIAG: src0 type=%d ne=(%lld,%lld) nb=(%lld,%lld) off=%u\n",
                        (int)node->src[0]->type,
                        (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                        (long long)node->src[0]->nb[0], (long long)node->src[0]->nb[1],
                        params.src0_offset);
                fprintf(stderr, "ggml-dx12: GLU DIAG: src1 type=%d ne=(%lld,%lld) nb=(%lld,%lld) off=%u\n",
                        node->src[1] ? (int)node->src[1]->type : -1,
                        node->src[1] ? (long long)node->src[1]->ne[0] : 0,
                        node->src[1] ? (long long)node->src[1]->ne[1] : 0,
                        node->src[1] ? (long long)node->src[1]->nb[0] : 0,
                        node->src[1] ? (long long)node->src[1]->nb[1] : 0,
                        params.src1_offset);
                fprintf(stderr, "ggml-dx12: GLU DIAG: dst type=%d ne=(%lld,%lld) nb=(%lld,%lld) off=%u\n",
                        (int)node->type,
                        (long long)node->ne[0], (long long)node->ne[1],
                        (long long)node->nb[0], (long long)node->nb[1],
                        params.dst_offset);
                fprintf(stderr, "ggml-dx12: GLU DIAG: op_params glu_op=%u swapped=%u\n",
                        params.op_params[0], params.op_params[1]);

                // Read back 8 floats from src0, src1, and dst
                const uint32_t DIAG_BYTES = 32; // 8 floats
                float src0_vals[8] = {}, src1_vals[8] = {}, dst_vals[8] = {};
                auto * dev = bctx->dev;
                dev->init_xfer();
                dev->xfer_ensure_staging(0, DIAG_BYTES);

                // Helper lambda to read from a resource at an offset
                auto readback = [&](ID3D12Resource * res, uint32_t offset, float * out, const char* label) {
                    if (!res) { fprintf(stderr, "ggml-dx12: GLU DIAG: %s res=NULL!\n", label); return; }
                    dev->xfer_wait(); // ensure prior xfer is done
                    HRESULT hr = dev->xfer.cmd_alloc->Reset();
                    if (FAILED(hr)) { fprintf(stderr, "ggml-dx12: GLU DIAG: %s cmd_alloc Reset failed 0x%08lX\n", label, (unsigned long)hr); return; }
                    hr = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
                    if (FAILED(hr)) { fprintf(stderr, "ggml-dx12: GLU DIAG: %s cmd_list Reset failed 0x%08lX\n", label, (unsigned long)hr); return; }
                    dev->xfer.cmd_list->CopyBufferRegion(
                        dev->xfer.readback_staging.Get(), 0, res, offset, DIAG_BYTES);
                    dev->xfer.cmd_list->Close();
                    ID3D12CommandList * lists[] = { dev->xfer.cmd_list.Get() };
                    dev->compute_queue->ExecuteCommandLists(1, lists);
                    dev->xfer.fence_value++;
                    dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                    dev->xfer_wait();
                    void * mapped = nullptr;
                    D3D12_RANGE read_range = { 0, DIAG_BYTES };
                    hr = dev->xfer.readback_staging->Map(0, &read_range, &mapped);
                    if (FAILED(hr)) { fprintf(stderr, "ggml-dx12: GLU DIAG: %s Map failed 0x%08lX\n", label, (unsigned long)hr); return; }
                    memcpy(out, mapped, DIAG_BYTES);
                    D3D12_RANGE written = { 0, 0 };
                    dev->xfer.readback_staging->Unmap(0, &written);
                };

                readback(src0_res, params.src0_offset, src0_vals, "src0");
                readback(src1_res, params.src1_offset, src1_vals, "src1");
                readback(dst_res, params.dst_offset, dst_vals, "dst");

                fprintf(stderr, "ggml-dx12: GLU DIAG src0[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                        src0_vals[0], src0_vals[1], src0_vals[2], src0_vals[3],
                        src0_vals[4], src0_vals[5], src0_vals[6], src0_vals[7]);
                fprintf(stderr, "ggml-dx12: GLU DIAG src1[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                        src1_vals[0], src1_vals[1], src1_vals[2], src1_vals[3],
                        src1_vals[4], src1_vals[5], src1_vals[6], src1_vals[7]);
                fprintf(stderr, "ggml-dx12: GLU DIAG dst[0..7]:  %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                        dst_vals[0], dst_vals[1], dst_vals[2], dst_vals[3],
                        dst_vals[4], dst_vals[5], dst_vals[6], dst_vals[7]);

                // Expected: dst[i] = silu(src0[i]) * src1[i]
                fprintf(stderr, "ggml-dx12: GLU DIAG expected[0..3]: ");
                for (int d = 0; d < 4; d++) {
                    float g = src0_vals[d];
                    float silu_g = g / (1.0f + expf(-g));
                    fprintf(stderr, "%.6f ", silu_g * src1_vals[d]);
                }
                fprintf(stderr, "\n");
                fflush(stderr);

                // Re-open command list
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                unsynced_writes.clear();
            }
        }

        // Track this dispatch's output as unsynced
        unsynced_writes.insert((uintptr_t)dst_tensor);
        // For triple fusion, also track the ADD intermediate output as unsynced
        if (fused_add_rms_node) {
            unsynced_writes.insert((uintptr_t)fused_add_rms_node);
        }

        // Skip fused nodes
        if (fused_add_rms_node) {
            i += 2;  // skip the RMS_NORM and MUL nodes
        } else if (fused_rope_after_rms) {
            i += 2;  // skip the MUL and ROPE nodes
        } else if (fused_mul_node) {
            i++;  // skip the MUL node
        }
        if (fused_bias_add) {
            i++;  // skip the ADD node
        }
        if (fused_rope_set_rows) {
            i += 2;  // skip the VIEW and SET_ROWS nodes
        }

        // Periodic CL flush for decode phase (M=1) to work around AMD RDNA 3
        // stale cache issue.  UAV barriers alone are insufficient — full pipeline
        // drains (close CL + execute + wait) are required.  The prompt path has
        // its own weight-based TDR flushing.  For decode, flush every N dispatches
        // throughout the graph body.  The model head CL splits below handle the
        // tail.  Configurable via GGML_DX12_FLUSH_INTERVAL (default: 16).
        if (!is_prompt && cgraph->n_nodes > 100) {
            static int decode_flush_interval = 0;
            if (decode_flush_interval == 0) {
                const char * env = getenv("GGML_DX12_FLUSH_INTERVAL");
                decode_flush_interval = env ? atoi(env) : 16;
                if (decode_flush_interval <= 0) decode_flush_interval = 16;
            }
            dispatch_weight++;
            if (dispatch_weight >= decode_flush_interval) {
                bctx->close_and_execute();
                bctx->wait_for_gpu();
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                unsynced_writes.clear();
                dispatch_weight = 0;
            }
        }

        // Force CL split at every dispatch in model head region to fix NaN and sync.
        // AMD RDNA 3 requires full pipeline drains (not just UAV barriers) between
        // the final FFN/norm layers and the vocab projection + logit softcapping.
        // Without these splits, the model head produces NaN due to stale GPU caches.
        if (cgraph->n_nodes > 1000 && i >= cgraph->n_nodes - 25 && i <= cgraph->n_nodes - 2) {
            bctx->close_and_execute();
            bctx->wait_for_gpu();
            bctx->ensure_cmd_list_open();
            bctx->reset_binding_cache();
            unsynced_writes.clear();
            dispatch_weight = 0;
        }

        // Per-dispatch NaN isolation: force flush + readback for target range
        if (false && dx12_diag && cgraph->n_nodes > 1000 && i >= 1597 && i <= 1610) {
            static int iso_count = 0;
            if (iso_count < 15) {
                // Force flush
                bctx->close_and_execute();
                bctx->wait_for_gpu();
                // Read back destination tensor
                auto * probe_node = cgraph->nodes[i];
                if (probe_node && probe_node->buffer) {
                    size_t elem_size = ggml_type_size(probe_node->type);
                    size_t n_elem = probe_node->ne[0] * std::max((int64_t)1, probe_node->ne[1]);
                    size_t probe_bytes = std::min(n_elem * elem_size, (size_t)(256 * 4));
                    uint32_t poff = (uint32_t)dx12_tensor_offset(probe_node);
                    auto * dev = bctx->dev;
                    dev->init_xfer();
                    dev->xfer_ensure_staging(0, probe_bytes);
                    dev->xfer_wait();
                    HRESULT hr3 = dev->xfer.cmd_alloc->Reset();
                    if (SUCCEEDED(hr3)) {
                        hr3 = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
                        if (SUCCEEDED(hr3)) {
                            auto * pbuf = (dx12_buffer_context *)probe_node->buffer->context;
                            dev->xfer.cmd_list->CopyBufferRegion(
                                dev->xfer.readback_staging.Get(), 0,
                                pbuf->resource.Get(), poff, probe_bytes);
                            dev->xfer.cmd_list->Close();
                            ID3D12CommandList * plists[] = { dev->xfer.cmd_list.Get() };
                            dev->compute_queue->ExecuteCommandLists(1, plists);
                            dev->xfer.fence_value++;
                            dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                            dev->xfer_wait();
                            void * pm = nullptr;
                            D3D12_RANGE prr = { 0, probe_bytes };
                            hr3 = dev->xfer.readback_staging->Map(0, &prr, &pm);
                            if (SUCCEEDED(hr3)) {
                                float * pf = (float *)pm;
                                int n_check = (int)(probe_bytes / 4);
                                int pnans = 0;
                                for (int pi = 0; pi < n_check; pi++) if (isnan(pf[pi])) pnans++;
                                int src0t = probe_node->src[0] ? (int)probe_node->src[0]->type : -1;
                                fprintf(stderr, "ggml-dx12: ISO disp#%d %s(src0t=%d) ne=(%lld,%lld,%lld,%lld) off=%u: "
                                        "nans=%d/%d val[0..3]=%.4g %.4g %.4g %.4g\n",
                                        i, ggml_op_name(probe_node->op), src0t,
                                        (long long)probe_node->ne[0], (long long)probe_node->ne[1],
                                        (long long)probe_node->ne[2], (long long)probe_node->ne[3],
                                        poff, pnans, n_check,
                                        pf[0], n_check>1?pf[1]:0, n_check>2?pf[2]:0, n_check>3?pf[3]:0);
                                fflush(stderr);
                                D3D12_RANGE pwr = { 0, 0 };
                                dev->xfer.readback_staging->Unmap(0, &pwr);
                                iso_count++;
                            }
                        }
                    }
                }
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                unsynced_writes.clear();
                dispatch_weight = 0;
            }
        }

        if (do_profile) {
            bctx->close_and_execute();
            bctx->wait_for_gpu();
            bctx->ensure_cmd_list_open();
            bctx->reset_binding_cache();
            unsynced_writes.clear();
#ifdef _WIN32
            QueryPerformanceCounter(&t1);
            double ms = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart * 1000.0;
#else
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
            char key[128];
            int src0t = node->src[0] ? (int)node->src[0]->type : -1;
            snprintf(key, sizeof(key), "%s(src0t=%d,grp=%u)", ggml_op_name(node->op), src0t, groups_x);
            op_times[key] += ms;
        }

        // TDR prevention: flush during prompt processing
        if (is_prompt) {
            int weight = 1;
            if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_FLASH_ATTN_EXT) {
                // Use total thread groups — not just groups_x — because 2D dispatch
                // (e.g., groups=(256,128,1) for prompt batches) hides the true
                // GPU workload in the Y/Z axes.
                uint64_t total_groups = (uint64_t)groups_x * groups_y * groups_z;
                if (total_groups >= 8000) weight = 16;
                else if (total_groups >= 1000) weight = 4;
                else weight = 2;
            }
            dispatch_weight += weight;
            if (dispatch_weight >= TDR_FLUSH_THRESHOLD) {
                static int flush_num = 0;
                flush_num++;
#ifdef _WIN32
                if (trace_prompt) {
                    LARGE_INTEGER tdr_now;
                    QueryPerformanceCounter(&tdr_now);
                    double elapsed_ms = (double)(tdr_now.QuadPart - tdr_t0.QuadPart) / tdr_freq.QuadPart * 1000.0;
                    fprintf(stderr, "  F#%d @d%d w=%d t=%.0fms\n", flush_num, i, dispatch_weight, elapsed_ms);
                    fflush(stderr);
                }
#else
                if (trace_prompt) {
                    fprintf(stderr, "  F#%d @d%d w=%d\n", flush_num, i, dispatch_weight);
                    fflush(stderr);
                }
#endif
                bctx->close_and_execute();
                bctx->wait_for_gpu();
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                unsynced_writes.clear();
                if (trace_prompt) {
                    HRESULT dev_stat = bctx->dev->device->GetDeviceRemovedReason();
                    if (FAILED(dev_stat)) {
                        fprintf(stderr, "  F#%d TDR! dev=0x%08lX\n", flush_num, (unsigned long)dev_stat);
                        fflush(stderr);
                    }
                }

                // Yield to DWM periodically: continuous back-to-back compute CLs
                // monopolize the GPU and prevent display compositing.  The Windows
                // TDR timer covers the ENTIRE adapter — if DWM can't draw for ~2s,
                // the OS resets the GPU even though individual CLs are fast.
#ifdef _WIN32
                {
                    LARGE_INTEGER tdr_now;
                    QueryPerformanceCounter(&tdr_now);
                    double elapsed_ms = (double)(tdr_now.QuadPart - tdr_t0.QuadPart) / tdr_freq.QuadPart * 1000.0;
                    if (elapsed_ms >= TDR_YIELD_MS) {
                        Sleep(2);  // yield GPU time to display compositor
                        QueryPerformanceCounter(&tdr_t0);  // reset yield timer
                        if (trace_prompt) {
                            fprintf(stderr, "  [yield at %.0fms]\n", elapsed_ms);
                            fflush(stderr);
                        }
                    }
                }
#endif

                if (dx12_diag) {
                    fprintf(stderr, "ggml-dx12: flush #%d OK after dispatch %d\n", flush_num, i);
                    fflush(stderr);
                }

                // NaN probe: after first few flushes of large graph, check last dst for NaN
                if (dx12_diag && cgraph->n_nodes > 1000) {
                    static int nan_probe_count = 0;
                    // Binary search: probe at specific dispatch thresholds
                    static int probe_targets[] = {1598, 1599, 1600, 1601, 1603, 1604, 1605, 1606, 1607, 1608};
                    static int next_target_idx = 0;
                    bool should_probe = false;
                    if (next_target_idx < 10 && i >= probe_targets[next_target_idx]) {
                        should_probe = true;
                        next_target_idx++;
                    }
                    if (should_probe && nan_probe_count < 15) {
                        // Read back last dispatched dst tensor
                        auto * last_node = cgraph->nodes[i];
                        if (last_node) {
                            size_t elem_size = ggml_type_size(last_node->type);
                            size_t elem_count = last_node->ne[0] * std::max((int64_t)1, last_node->ne[1]);
                            size_t probe_bytes = std::min(elem_count * elem_size, (size_t)(256 * 4));
                            uint32_t dst_off = (uint32_t)dx12_tensor_offset(last_node);
                            auto * dev = bctx->dev;
                            dev->init_xfer();
                            dev->xfer_ensure_staging(0, probe_bytes);
                            dev->xfer_wait();
                            HRESULT hr2 = dev->xfer.cmd_alloc->Reset();
                            if (SUCCEEDED(hr2)) {
                                hr2 = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
                                if (SUCCEEDED(hr2)) {
                                    auto * buf_ctx = (dx12_buffer_context *)last_node->buffer->context;
                                    dev->xfer.cmd_list->CopyBufferRegion(
                                        dev->xfer.readback_staging.Get(), 0,
                                        buf_ctx->resource.Get(), dst_off, probe_bytes);
                                    dev->xfer.cmd_list->Close();
                                    ID3D12CommandList * xlists[] = { dev->xfer.cmd_list.Get() };
                                    dev->compute_queue->ExecuteCommandLists(1, xlists);
                                    dev->xfer.fence_value++;
                                    dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                                    dev->xfer_wait();
                                    void * mapped2 = nullptr;
                                    D3D12_RANGE rr = { 0, probe_bytes };
                                    hr2 = dev->xfer.readback_staging->Map(0, &rr, &mapped2);
                                    if (SUCCEEDED(hr2)) {
                                        // Check for NaN — handle both F32 and F16
                                        int n_probe = (int)(probe_bytes / elem_size);
                                        int nans = 0, infs = 0;
                                        float v0=0, v1=0, v2=0, v3=0;
                                        if (last_node->type == GGML_TYPE_F32) {
                                            float * fvals = (float *)mapped2;
                                            for (int pi = 0; pi < n_probe && pi < 256; pi++) {
                                                if (isnan(fvals[pi])) nans++;
                                                if (isinf(fvals[pi])) infs++;
                                            }
                                            v0 = fvals[0]; v1 = n_probe>1?fvals[1]:0;
                                            v2 = n_probe>2?fvals[2]:0; v3 = n_probe>3?fvals[3]:0;
                                        } else if (last_node->type == GGML_TYPE_F16) {
                                            uint16_t * hvals = (uint16_t *)mapped2;
                                            for (int pi = 0; pi < n_probe && pi < 256; pi++) {
                                                uint16_t h = hvals[pi];
                                                if ((h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0) nans++;
                                                if ((h & 0x7FFF) == 0x7C00) infs++;
                                            }
                                            // Convert first 4 to float for display
                                            auto h2f = [](uint16_t h) -> float {
                                                uint32_t sign = (uint32_t)(h >> 15) << 31;
                                                uint32_t exp = (h >> 10) & 0x1F;
                                                uint32_t mant = h & 0x3FF;
                                                if (exp == 0) { if (mant == 0) { uint32_t r = sign; float f; memcpy(&f,&r,4); return f; } exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF; }
                                                else if (exp == 31) { uint32_t r = sign | 0x7F800000 | (mant << 13); float f; memcpy(&f,&r,4); return f; }
                                                uint32_t r = sign | ((exp + 112) << 23) | (mant << 13);
                                                float f; memcpy(&f, &r, 4); return f;
                                            };
                                            v0 = h2f(hvals[0]); v1 = n_probe>1?h2f(hvals[1]):0;
                                            v2 = n_probe>2?h2f(hvals[2]):0; v3 = n_probe>3?h2f(hvals[3]):0;
                                        }
                                        fprintf(stderr, "ggml-dx12: NaN PROBE @disp#%d op=%d type=%d ne=(%lld,%lld,%lld,%lld) "
                                                "off=%u: nans=%d infs=%d/%d val[0..3]=%.4g %.4g %.4g %.4g\n",
                                                i, (int)last_node->op, (int)last_node->type,
                                                (long long)last_node->ne[0], (long long)last_node->ne[1],
                                                (long long)last_node->ne[2], (long long)last_node->ne[3],
                                                dst_off, nans, infs, n_probe,
                                                v0, v1, v2, v3);
                                        fflush(stderr);
                                        D3D12_RANGE wr = { 0, 0 };
                                        dev->xfer.readback_staging->Unmap(0, &wr);
                                        nan_probe_count++;
                                    }
                                }
                            }
                        }
                    }
                }
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                dispatch_weight = 0;
            }
        }

    }

    // Diagnostic: sync all remaining work before returning
    if (dx12_diag && is_prompt && bctx->cmd_list_open) {
        bctx->close_and_execute();
        bctx->wait_for_gpu();
        fprintf(stderr, "ggml-dx12: final flush OK (all %d dispatches completed)\n", cgraph->n_nodes);
        fflush(stderr);

        // Read back the hidden state (src[1] of the vocab projection MUL_MAT)
        // to compare with CPU reference. Find last MUL_MAT with large output.
        static int hidden_diag_count = 0;
        if (hidden_diag_count < 2 && cgraph->n_nodes > 1000) {
            hidden_diag_count++;
            struct ggml_tensor * vocab_mm = nullptr;
            for (int ni = cgraph->n_nodes - 1; ni >= cgraph->n_nodes - 10; ni--) {
                auto * nd = cgraph->nodes[ni];
                if (nd->op == GGML_OP_MUL_MAT && nd->ne[0] >= 100000) {
                    vocab_mm = nd;
                    break;
                }
            }
            if (vocab_mm && vocab_mm->src[1] && vocab_mm->src[1]->buffer &&
                vocab_mm->src[1]->type == GGML_TYPE_F32) {
                auto * hs = vocab_mm->src[1];  // hidden state
                size_t hs_bytes = hs->ne[0] * sizeof(float);
                uint32_t hs_off = (uint32_t)dx12_tensor_offset(hs);
                auto * dev = bctx->dev;
                dev->init_xfer();
                dev->xfer_ensure_staging(0, hs_bytes);
                dev->xfer_wait();
                HRESULT hr_hs = dev->xfer.cmd_alloc->Reset();
                if (SUCCEEDED(hr_hs)) hr_hs = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
                if (SUCCEEDED(hr_hs)) {
                    auto * hsbuf = (dx12_buffer_context *)hs->buffer->context;
                    dev->xfer.cmd_list->CopyBufferRegion(
                        dev->xfer.readback_staging.Get(), 0,
                        hsbuf->resource.Get(), hs_off, hs_bytes);
                    dev->xfer.cmd_list->Close();
                    ID3D12CommandList * hlists[] = { dev->xfer.cmd_list.Get() };
                    dev->compute_queue->ExecuteCommandLists(1, hlists);
                    dev->xfer.fence_value++;
                    dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                    dev->xfer_wait();
                    void * hm = nullptr;
                    D3D12_RANGE hrr = { 0, hs_bytes };
                    hr_hs = dev->xfer.readback_staging->Map(0, &hrr, &hm);
                    if (SUCCEEDED(hr_hs)) {
                        float * hf = (float *)hm;
                        int n = (int)(hs->ne[0]);
                        double sum_sq = 0, sum = 0;
                        int nans = 0;
                        for (int j = 0; j < n; j++) {
                            if (isnan(hf[j])) { nans++; continue; }
                            sum += hf[j];
                            sum_sq += (double)hf[j] * hf[j];
                        }
                        double rms = sqrt(sum_sq / n);
                        fprintf(stderr, "ggml-dx12: HIDDEN STATE (vocab_mm src[1]) ne=(%lld,%lld) off=%u:\n",
                                (long long)hs->ne[0], (long long)hs->ne[1], hs_off);
                        fprintf(stderr, "  rms=%.6f mean=%.6f nans=%d\n", rms, sum/n, nans);
                        fprintf(stderr, "  first16: ");
                        for (int j = 0; j < 16 && j < n; j++) fprintf(stderr, "%.4f ", hf[j]);
                        fprintf(stderr, "\n  last8: ");
                        for (int j = n-8; j < n; j++) fprintf(stderr, "%.4f ", hf[j]);
                        fprintf(stderr, "\n");
                        fflush(stderr);
                        D3D12_RANGE hwr = { 0, 0 };
                        dev->xfer.readback_staging->Unmap(0, &hwr);
                    }
                }
            }
        }

        bctx->ensure_cmd_list_open();  // leave CL open as expected
    }

    // Keep the command list open — UAV barriers between dispatches ensure
    // correct ordering within a single command list.  The list is flushed
    // in synchronize(), get_tensor(), or set_tensor() when results are
    // actually needed.  This avoids 300+ close/execute/wait round-trips
    // per generation that were pegging the CPU at 100%.

    // Dump profiling results
    if (do_profile && !op_times.empty()) {
        fprintf(stderr, "\n=== DX12 Profile (graph #%d) ===\n", profile_graph);
        std::vector<std::pair<double, std::string>> sorted;
        double total = 0;
        for (auto & kv : op_times) { sorted.push_back({kv.second, kv.first}); total += kv.second; }
        std::sort(sorted.rbegin(), sorted.rend());
        for (auto & p : sorted) {
            if (p.first > 0.01) fprintf(stderr, "  %8.3f ms  %5.1f%%  %s\n", p.first, p.first/total*100, p.second.c_str());
        }
        fprintf(stderr, "  %8.3f ms  TOTAL\n", total);
    }

    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Auto-tuning: benchmark shader variants and pick the fastest per GPU
// ---------------------------------------------------------------------------

void dx12_device::run_autotune() {
    if (tuning_done) return;
    tuning_done = true;

    // Check for cache file first
    char cache_path[512];
#ifdef _WIN32
    snprintf(cache_path, sizeof(cache_path), "%s/.ggml_dx12_tune_%04X_%04X.txt",
             getenv("LOCALAPPDATA") ? getenv("LOCALAPPDATA") : ".",
             vendor_id, device_id);
#else
    snprintf(cache_path, sizeof(cache_path), "%s/.ggml_dx12_tune_%04X_%04X.txt",
             getenv("HOME") ? getenv("HOME") : "/tmp",
             vendor_id, device_id);
#endif

    FILE * f = fopen(cache_path, "r");
    if (f) {
        int ver = 0, q5 = 0, q8 = 0, q6 = 0, f16l4 = 0;
        if (fscanf(f, "v=%d q5_0_256=%d q8_0_256=%d q6k_32=%d f16_load4=%d", &ver, &q5, &q8, &q6, &f16l4) == 5
            && ver == TUNE_VERSION) {
            q5_0_use_256 = (q5 != 0);
            q8_0_use_256 = (q8 != 0);
            q6k_use_32   = (q6 != 0);
            f16_use_load4 = (f16l4 != 0);
            fclose(f);
            DX12_LOG_INFO("Auto-tune v%d loaded: Q5_0=%s Q8_0=%s Q6_K=%s F16=%s\n", ver,
                          q5_0_use_256 ? "256t" : "32t", q8_0_use_256 ? "256t" : "32t",
                          q6k_use_32 ? "32t" : "256t", f16_use_load4 ? "load4" : "load2");
            return;
        }
        fclose(f);
        // Version mismatch or parse failure — re-benchmark
    }

    DX12_LOG_INFO("Running auto-tune benchmark...\n");

    // Create a temporary buffer for benchmarking
    // Must be large enough for max test: N=256 rows × max K stride
    // For K=3072 matvec: 256 rows × 3072 byte stride = 768KB + shader read range
    ComPtr<ID3D12Resource> bench_buf;
    {
        D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 4 * 1024 * 1024;  // 4MB — covers all benchmark configurations
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&bench_buf));
        if (FAILED(hr)) {
            DX12_LOG_WARN("Auto-tune: failed to create benchmark buffer\n");
            return;
        }
    }

    // Create timestamp query heap
    ComPtr<ID3D12QueryHeap> ts_heap;
    ComPtr<ID3D12Resource> ts_readback;
    {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = 4;  // start + end for 2 variants
        HRESULT hr = device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&ts_heap));
        if (FAILED(hr)) { DX12_LOG_WARN("Auto-tune: failed to create query heap\n"); return; }

        D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 4 * sizeof(uint64_t);
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ts_readback));
        if (FAILED(hr)) { DX12_LOG_WARN("Auto-tune: failed to create readback buffer\n"); return; }
    }

    // Helper: benchmark a pipeline variant
    // Returns GPU time in ticks, or UINT64_MAX on failure
    auto bench_pipeline = [&](dx12_pipeline_key key, uint32_t K, uint32_t N, uint32_t ts_start) -> uint64_t {
        dx12_pipeline * pl = get_or_create_pipeline(key);
        if (!pl || !pl->pso) return UINT64_MAX;

        // Create command allocator + list for benchmarking
        ComPtr<ID3D12CommandAllocator> alloc;
        ComPtr<ID3D12GraphicsCommandList> cl;
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));

        cl->SetComputeRootSignature(common_root_sig.Get());
        cl->SetPipelineState(pl->pso.Get());
        D3D12_GPU_VIRTUAL_ADDRESS va = bench_buf->GetGPUVirtualAddress();
        cl->SetComputeRootShaderResourceView(1, va);
        cl->SetComputeRootShaderResourceView(2, va);
        cl->SetComputeRootUnorderedAccessView(3, va);
        cl->SetComputeRootShaderResourceView(4, va);
        cl->SetComputeRootShaderResourceView(5, va);

        // Set minimal params
        dx12_shader_params params = {};
        params.ne00 = K; params.ne01 = N;
        params.ne02 = 1; params.ne03 = 1;
        params.nb00 = 1; params.nb01 = K;  // fake strides
        params.ne10 = K; params.ne11 = 1;
        params.ne12 = 1; params.ne13 = 1;
        params.nb10 = 4;  // F32 stride
        params.ne0 = N; params.ne1 = 1; params.ne2 = 1; params.ne3 = 1;
        params.nb0 = 4; params.nb1 = N * 4;
        params.src0_esize = 2;  // Q5_0/Q8_0 block size doesn't matter for benchmarking
        params.src1_esize = 4;
        params.dst_esize = 4;
        cl->SetComputeRoot32BitConstants(0, (uint32_t)(sizeof(params)/4), &params, 0);

        // Warmup dispatch
        cl->Dispatch(N, 1, 1);
        D3D12_RESOURCE_BARRIER barrier = {}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        cl->ResourceBarrier(1, &barrier);

        // Timed dispatch
        cl->EndQuery(ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts_start);
        for (int rep = 0; rep < 10; rep++) {
            cl->Dispatch(N, 1, 1);
            cl->ResourceBarrier(1, &barrier);
        }
        cl->EndQuery(ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts_start + 1);
        cl->ResolveQueryData(ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts_start, 2, ts_readback.Get(), ts_start * sizeof(uint64_t));

        cl->Close();
        ID3D12CommandList * lists[] = { cl.Get() };
        compute_queue->ExecuteCommandLists(1, lists);

        // Wait
        ComPtr<ID3D12Fence> fence;
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        // Wait with timeout — GPU hangs should not block indefinitely
        HANDLE event = dx12_create_event();
        fence->SetEventOnCompletion(1, event);
        compute_queue->Signal(fence.Get(), 1);
#ifdef _WIN32
        DWORD wait_result = WaitForSingleObject(event, 5000);
        dx12_close_event(event);

        if (wait_result == WAIT_TIMEOUT) {
            DX12_LOG_WARN("Auto-tune: GPU benchmark timed out\n");
            return UINT64_MAX;
        }
#else
        // WSL2: poll with timeout
        auto start = std::chrono::steady_clock::now();
        while (fence->GetCompletedValue() < 1) {
            if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(5000)) {
                DX12_LOG_WARN("Auto-tune: GPU benchmark timed out\n");
                return UINT64_MAX;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        dx12_close_event(event);
#endif

        // Read timestamps
        uint64_t * ts = nullptr;
        D3D12_RANGE range = { ts_start * sizeof(uint64_t), (ts_start + 2) * sizeof(uint64_t) };
        ts_readback->Map(0, &range, (void**)&ts);
        uint64_t dt = ts[ts_start + 1] - ts[ts_start];
        ts_readback->Unmap(0, nullptr);
        return dt;
    };

    // Benchmark Q5_0 matvec: 32 threads vs 256 threads
    // Test with K=576 (SmolVLM2-like) and K=3072 (Phi-3-like)
    uint32_t test_K[] = { 576, 3072 };
    uint32_t test_N = 256;  // number of output rows to benchmark

    uint64_t q50_32_total = 0, q50_256_total = 0;

    for (uint32_t K : test_K) {
        dx12_pipeline_key key32 = {}; key32.op = GGML_OP_MUL_MAT; key32.src0_type = GGML_TYPE_Q5_0; key32.flags = 1;
        dx12_pipeline_key key256 = {}; key256.op = GGML_OP_MUL_MAT; key256.src0_type = GGML_TYPE_Q5_0; key256.flags = 5;

        uint64_t t32 = bench_pipeline(key32, K, test_N, 0);
        uint64_t t256 = bench_pipeline(key256, K, test_N, 2);

        if (t32 != UINT64_MAX) q50_32_total += t32;
        if (t256 != UINT64_MAX) q50_256_total += t256;

        DX12_LOG_INFO("  Q5_0 K=%u: 32t=%llu 256t=%llu ticks\n", K, (unsigned long long)t32, (unsigned long long)t256);
    }

    q5_0_use_256 = (q50_256_total < q50_32_total && q50_256_total > 0);

    // Benchmark Q8_0 matvec: 32 threads vs 256 threads
    uint64_t q80_32_total = 0, q80_256_total = 0;

    for (uint32_t K : test_K) {
        dx12_pipeline_key key32 = {}; key32.op = GGML_OP_MUL_MAT; key32.src0_type = GGML_TYPE_Q8_0; key32.flags = 1;
        dx12_pipeline_key key256 = {}; key256.op = GGML_OP_MUL_MAT; key256.src0_type = GGML_TYPE_Q8_0; key256.flags = 5;

        uint64_t t32 = bench_pipeline(key32, K, test_N, 0);
        uint64_t t256 = bench_pipeline(key256, K, test_N, 2);

        if (t32 != UINT64_MAX) q80_32_total += t32;
        if (t256 != UINT64_MAX) q80_256_total += t256;

        DX12_LOG_INFO("  Q8_0 K=%u: 32t=%llu 256t=%llu ticks\n", K, (unsigned long long)t32, (unsigned long long)t256);
    }

    q8_0_use_256 = (q80_256_total < q80_32_total && q80_256_total > 0);

    // Benchmark Q6_K matvec: 256 threads (current default) vs 32 threads
    uint64_t q6k_256_total = 0, q6k_32_total = 0;

    for (uint32_t K : test_K) {
        dx12_pipeline_key key256 = {}; key256.op = GGML_OP_MUL_MAT; key256.src0_type = GGML_TYPE_Q6_K; key256.flags = 1;
        dx12_pipeline_key key32 = {}; key32.op = GGML_OP_MUL_MAT; key32.src0_type = GGML_TYPE_Q6_K; key32.flags = 5;

        uint64_t t256 = bench_pipeline(key256, K, test_N, 0);
        uint64_t t32 = bench_pipeline(key32, K, test_N, 2);

        if (t256 != UINT64_MAX) q6k_256_total += t256;
        if (t32 != UINT64_MAX) q6k_32_total += t32;

        DX12_LOG_INFO("  Q6_K K=%u: 256t=%llu 32t=%llu ticks\n", K, (unsigned long long)t256, (unsigned long long)t32);
    }

    q6k_use_32 = (q6k_32_total < q6k_256_total && q6k_32_total > 0);

    // Benchmark F16 matvec: Load2 (current) vs Load4
    uint64_t f16_load2_total = 0, f16_load4_total = 0;

    for (uint32_t K : test_K) {
        dx12_pipeline_key key_l2 = {}; key_l2.op = GGML_OP_MUL_MAT; key_l2.src0_type = GGML_TYPE_F16; key_l2.flags = 1;
        dx12_pipeline_key key_l4 = {}; key_l4.op = GGML_OP_MUL_MAT; key_l4.src0_type = GGML_TYPE_F16; key_l4.flags = 5;

        uint64_t t_l2 = bench_pipeline(key_l2, K, test_N, 0);
        uint64_t t_l4 = bench_pipeline(key_l4, K, test_N, 2);

        if (t_l2 != UINT64_MAX) f16_load2_total += t_l2;
        if (t_l4 != UINT64_MAX) f16_load4_total += t_l4;

        DX12_LOG_INFO("  F16 K=%u: load2=%llu load4=%llu ticks\n", K, (unsigned long long)t_l2, (unsigned long long)t_l4);
    }

    f16_use_load4 = (f16_load4_total < f16_load2_total && f16_load4_total > 0);

    DX12_LOG_INFO("Auto-tune result: Q5_0=%s Q8_0=%s Q6_K=%s F16=%s\n",
                  q5_0_use_256 ? "256t" : "32t", q8_0_use_256 ? "256t" : "32t",
                  q6k_use_32 ? "32t" : "256t", f16_use_load4 ? "load4" : "load2");

    // Save to cache
    f = fopen(cache_path, "w");
    if (f) {
        fprintf(f, "v=%d q5_0_256=%d q8_0_256=%d q6k_32=%d f16_load4=%d\n",
                TUNE_VERSION,
                q5_0_use_256 ? 1 : 0, q8_0_use_256 ? 1 : 0,
                q6k_use_32 ? 1 : 0, f16_use_load4 ? 1 : 0);
        fclose(f);
    }
}

static const char * dx12_backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_DX12_NAME;
}

static void dx12_backend_free(ggml_backend_t backend) {
    auto * ctx = (dx12_backend_context *)backend->context;
    delete ctx; // RAII destructor handles fence wait + event close
}

static void dx12_backend_synchronize(ggml_backend_t backend) {
    auto * ctx = (dx12_backend_context *)backend->context;

    // Submit pending work. The wait happens in get_tensor when data is actually needed.
    // This allows GPU to run asynchronously with CPU graph planning.
    if (ctx->cmd_list_open) {
        ctx->close_and_execute();
    }
}

static const ggml_backend_i dx12_backend_interface = {
    /* .get_name            = */ dx12_backend_get_name,
    /* .free                = */ dx12_backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ dx12_backend_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ dx12_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

// ---------------------------------------------------------------------------
// Backend GUID
// ---------------------------------------------------------------------------

static ggml_guid_t dx12_backend_get_guid() {
    static ggml_guid guid = {
        0xd3, 0xd1, 0x2b, 0xac, 0x6e, 0x77, 0x4f, 0xa2,
        0x8d, 0x1e, 0xc0, 0x0a, 0xee, 0x12, 0x34, 0x56
    };
    return &guid;
}

// ---------------------------------------------------------------------------
// Device interface
// ---------------------------------------------------------------------------

static const char * dx12_dev_get_name(ggml_backend_dev_t dev) {
    auto * d = (dx12_device *)dev->context;
    return d->name.c_str();
}

static const char * dx12_dev_get_description(ggml_backend_dev_t dev) {
    auto * d = (dx12_device *)dev->context;
    return d->description.c_str();
}

static void dx12_dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * d = (dx12_device *)dev->context;
    if (free)  *free  = d->vram_free;
    if (total) *total = d->vram_total;
}

static enum ggml_backend_dev_type dx12_dev_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void dx12_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    auto * d = (dx12_device *)dev->context;
    props->name         = d->name.c_str();
    props->description  = d->name.c_str();
    props->memory_free  = d->vram_free;
    props->memory_total = d->vram_total;
    props->type         = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id    = nullptr;
    props->caps = {
        /* .async             = */ false,
        /* .host_buffer       = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events            = */ false,
    };
}

static ggml_backend_t dx12_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * d = (dx12_device *)dev->context;

    auto * ctx = new dx12_backend_context();
    ctx->dev = d;

    HRESULT hr = d->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx->fence));
    DX12_CHECK(hr, "CreateFence");
    ctx->fence_event = dx12_create_event();

    auto * backend = new ggml_backend();
    backend->guid    = dx12_backend_get_guid();
    backend->iface   = dx12_backend_interface;
    backend->device  = dev;
    backend->context = ctx;
    return backend;
}

static ggml_backend_buffer_type_t dx12_dev_get_buffer_type(ggml_backend_dev_t dev) {
    auto * d = (dx12_device *)dev->context;
    size_t idx = d->dev_index;
    if (!g_dx12_buffer_types[idx].context) {
        g_dx12_buffer_types[idx].iface   = dx12_buffer_type_interface;
        g_dx12_buffer_types[idx].device  = dev;
        g_dx12_buffer_types[idx].context = d;
    }
    return &g_dx12_buffer_types[idx];
}

static bool dx12_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return dx12_supports_op(dev, op);
}

static bool dx12_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != dx12_buft_get_name) {
        return false;
    }
    // Ensure the buffer type belongs to the same D3D12 device —
    // resources created on one GPU cannot be used on another.
    dx12_device * this_dev = (dx12_device *)dev->context;
    dx12_device * buft_dev = (dx12_device *)buft->context;
    return this_dev == buft_dev;
}

static const ggml_backend_device_i dx12_device_interface = {
    /* .get_name              = */ dx12_dev_get_name,
    /* .get_description       = */ dx12_dev_get_description,
    /* .get_memory            = */ dx12_dev_get_memory,
    /* .get_type              = */ dx12_dev_get_type,
    /* .get_props             = */ dx12_dev_get_props,
    /* .init_backend          = */ dx12_dev_init_backend,
    /* .get_buffer_type       = */ dx12_dev_get_buffer_type,
    /* .get_host_buffer_type  = */ nullptr,
    /* .buffer_from_host_ptr  = */ nullptr,
    /* .supports_op           = */ dx12_dev_supports_op,
    /* .supports_buft         = */ dx12_dev_supports_buft,
    /* .offload_op            = */ nullptr,
    /* .event_new             = */ nullptr,
    /* .event_free            = */ nullptr,
    /* .event_synchronize     = */ nullptr,
};

// ---------------------------------------------------------------------------
// Registry interface
// ---------------------------------------------------------------------------

static const char * dx12_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_DX12_NAME;
}

static size_t dx12_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    dx12_ensure_initialized();
    return g_dx12.devices.size();
}

static ggml_backend_dev_t dx12_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    dx12_ensure_initialized();
    GGML_ASSERT(index < g_dx12.devices.size());

    // Lazy-initialize backend device objects
    if (g_dx12.backend_devices.empty()) {
        g_dx12.backend_devices.resize(g_dx12.devices.size());
        for (size_t i = 0; i < g_dx12.devices.size(); i++) {
            g_dx12.backend_devices[i].iface   = dx12_device_interface;
            g_dx12.backend_devices[i].reg     = &g_dx12.backend_reg_obj;
            g_dx12.backend_devices[i].context = g_dx12.devices[i].get();
        }
    }
    return &g_dx12.backend_devices[index];
}

static const ggml_backend_reg_i dx12_reg_interface = {
    /* .get_name         = */ dx12_reg_get_name,
    /* .get_device_count = */ dx12_reg_get_device_count,
    /* .get_device       = */ dx12_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

// ---------------------------------------------------------------------------
// Shader blob registry
// ---------------------------------------------------------------------------

struct dx12_shader_blob {
    const void * data;
    size_t       size;
};

#ifdef GGML_DX12_SHADERS_COMPILED
static const std::unordered_map<int, dx12_shader_blob> g_shader_blobs = {
    { GGML_OP_ADD,           { g_add_dxil,           sizeof(g_add_dxil)           } },
    { GGML_OP_MUL,           { g_mul_dxil,           sizeof(g_mul_dxil)           } },
    { GGML_OP_SCALE,         { g_scale_dxil,         sizeof(g_scale_dxil)         } },
    { GGML_OP_SQR,           { g_sqr_dxil,           sizeof(g_sqr_dxil)           } },
    { GGML_OP_SQRT,          { g_sqrt__dxil,         sizeof(g_sqrt__dxil)         } },
    { GGML_OP_CLAMP,         { g_clamp_dxil,         sizeof(g_clamp_dxil)         } },
    { GGML_OP_CONT,          { g_cpy_dxil,           sizeof(g_cpy_dxil)           } },
    { GGML_OP_CPY,           { g_cpy_dxil,           sizeof(g_cpy_dxil)           } },
    { GGML_OP_DUP,           { g_cpy_dxil,           sizeof(g_cpy_dxil)           } },
    { GGML_OP_RMS_NORM,      { g_rms_norm_dxil,      sizeof(g_rms_norm_dxil)      } },
    { GGML_OP_NORM,          { g_norm_dxil,          sizeof(g_norm_dxil)          } },
    { GGML_OP_SOFT_MAX,      { g_soft_max_dxil,      sizeof(g_soft_max_dxil)      } },
    { GGML_OP_MUL_MAT,       { g_mul_mat_dxil,       sizeof(g_mul_mat_dxil)       } },
    { GGML_OP_GET_ROWS,      { g_get_rows_dxil,      sizeof(g_get_rows_dxil)      } },
    { GGML_OP_DIAG_MASK_INF, { g_diag_mask_inf_dxil, sizeof(g_diag_mask_inf_dxil) } },
    { GGML_OP_ROPE,          { g_rope_dxil,          sizeof(g_rope_dxil)          } },
    { GGML_OP_CONCAT,        { g_concat_dxil,        sizeof(g_concat_dxil)        } },
    { GGML_OP_REPEAT,        { g_repeat_dxil,        sizeof(g_repeat_dxil)        } },
    { GGML_OP_SUM_ROWS,      { g_sum_rows_dxil,      sizeof(g_sum_rows_dxil)      } },
    { GGML_OP_FLASH_ATTN_EXT,{ g_flash_attn_dxil,    sizeof(g_flash_attn_dxil)    } },
    { GGML_OP_SET_ROWS,      { g_set_rows_dxil,      sizeof(g_set_rows_dxil)      } },
    { GGML_OP_GLU,           { g_glu_dxil,           sizeof(g_glu_dxil)           } },
};

// Unary op shaders keyed by GGML_UNARY_OP_*
static const std::unordered_map<int, dx12_shader_blob> g_unary_shader_blobs = {
    { GGML_UNARY_OP_SILU,       { g_silu_dxil,       sizeof(g_silu_dxil)       } },
    { GGML_UNARY_OP_GELU,       { g_gelu_dxil,       sizeof(g_gelu_dxil)       } },
    { GGML_UNARY_OP_GELU_QUICK, { g_gelu_quick_dxil, sizeof(g_gelu_quick_dxil) } },
    { GGML_UNARY_OP_RELU,       { g_relu_dxil,       sizeof(g_relu_dxil)       } },
    { GGML_UNARY_OP_TANH,       { g_tanh__dxil,      sizeof(g_tanh__dxil)      } },
    { GGML_UNARY_OP_SIGMOID,    { g_sigmoid_dxil,    sizeof(g_sigmoid_dxil)    } },
};
#endif

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

dx12_pipeline * dx12_device::get_or_create_pipeline(const dx12_pipeline_key & key) {
    std::lock_guard<std::mutex> lock(pipeline_mutex);

    auto it = pipeline_cache.find(key);
    if (it != pipeline_cache.end()) {
        return &it->second;
    }

#ifdef GGML_DX12_SHADERS_COMPILED
    const dx12_shader_blob * blob = nullptr;

    // For UNARY ops, look up by the unary sub-op stored in flags
    if (key.op == GGML_OP_UNARY) {
        auto uit = g_unary_shader_blobs.find((int)key.flags);
        if (uit != g_unary_shader_blobs.end()) {
            blob = &uit->second;
        }
    } else if (key.op == GGML_OP_RMS_NORM && key.flags == 2) {
        // Fused RMS_NORM + MUL
        static const dx12_shader_blob fused_blob = { g_rms_norm_mul_dxil, sizeof(g_rms_norm_mul_dxil) };
        blob = &fused_blob;
    } else if (key.op == GGML_OP_RMS_NORM && key.flags == 3) {
        // Fused ADD + RMS_NORM + MUL (triple fusion)
        static const dx12_shader_blob fused_blob = { g_add_rms_norm_mul_dxil, sizeof(g_add_rms_norm_mul_dxil) };
        blob = &fused_blob;
    } else if (key.op == GGML_OP_RMS_NORM && key.flags == 7) {
        // Fused RMS_NORM + MUL + ROPE
        static const dx12_shader_blob fused_blob = { g_rms_norm_mul_rope_dxil, sizeof(g_rms_norm_mul_rope_dxil) };
        blob = &fused_blob;
    } else if (key.op == GGML_OP_ROPE && key.flags == 6) {
        // Fused ROPE + VIEW + SET_ROWS
        static const dx12_shader_blob fused_blob = { g_rope_set_rows_dxil, sizeof(g_rope_set_rows_dxil) };
        blob = &fused_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 4) {
        // Register-blocked tiled batch MUL_MAT (M > 1)
        if (key.src0_type == GGML_TYPE_Q4_K) {
            static const dx12_shader_blob wmma_q4k_blob = { g_mul_mat_q4k_wmma_dxil, sizeof(g_mul_mat_q4k_wmma_dxil) };
            blob = &wmma_q4k_blob;
        } else if (key.src0_type == GGML_TYPE_Q5_K) {
            static const dx12_shader_blob wmma_q5k_blob = { g_mul_mat_q5k_wmma_dxil, sizeof(g_mul_mat_q5k_wmma_dxil) };
            blob = &wmma_q5k_blob;
        } else if (key.src0_type == GGML_TYPE_Q6_K) {
            static const dx12_shader_blob wmma_q6k_blob = { g_mul_mat_q6k_wmma_dxil, sizeof(g_mul_mat_q6k_wmma_dxil) };
            blob = &wmma_q6k_blob;
        } else {
            // F16/F32 register-blocked tiled batch MUL_MAT
            static const dx12_shader_blob wmma_blob = { g_mul_mat_wmma_dxil, sizeof(g_mul_mat_wmma_dxil) };
            blob = &wmma_blob;
        }
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 2) {
        // Multi-row matvec (2 rows per group)
        if (key.src0_type == GGML_TYPE_Q4_K) {
            static const dx12_shader_blob mr_q4k_blob = { g_mul_mat_vec_q4k_mr_dxil, sizeof(g_mul_mat_vec_q4k_mr_dxil) };
            blob = &mr_q4k_blob;
        }
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 1) {
        // Matvec path (M=1 single-token generation)
        if (key.src0_type == GGML_TYPE_Q2_K) {
            static const dx12_shader_blob mv_q2k_blob = { g_mul_mat_vec_q2k_dxil, sizeof(g_mul_mat_vec_q2k_dxil) };
            blob = &mv_q2k_blob;
        } else if (key.src0_type == GGML_TYPE_Q3_K) {
            static const dx12_shader_blob mv_q3k_blob = { g_mul_mat_vec_q3k_dxil, sizeof(g_mul_mat_vec_q3k_dxil) };
            blob = &mv_q3k_blob;
        } else if (key.src0_type == GGML_TYPE_Q4_K) {
            static const dx12_shader_blob mv_q4k_blob = { g_mul_mat_vec_q4k_dxil, sizeof(g_mul_mat_vec_q4k_dxil) };
            blob = &mv_q4k_blob;
        } else if (key.src0_type == GGML_TYPE_Q5_K) {
            static const dx12_shader_blob mv_q5k_blob = { g_mul_mat_vec_q5k_dxil, sizeof(g_mul_mat_vec_q5k_dxil) };
            blob = &mv_q5k_blob;
        } else if (key.src0_type == GGML_TYPE_Q6_K) {
            if (key.flags == 5) {
                static const dx12_shader_blob mv_q6k_32_blob = { g_mul_mat_vec_q6k_32_dxil, sizeof(g_mul_mat_vec_q6k_32_dxil) };
                blob = &mv_q6k_32_blob;
            } else {
                static const dx12_shader_blob mv_q6k_blob = { g_mul_mat_vec_q6k_dxil, sizeof(g_mul_mat_vec_q6k_dxil) };
                blob = &mv_q6k_blob;
            }
        } else if (key.src0_type == GGML_TYPE_Q5_0) {
            if (key.flags == 5) {
                static const dx12_shader_blob mv_q50_256_blob = { g_mul_mat_vec_q5_0_256_dxil, sizeof(g_mul_mat_vec_q5_0_256_dxil) };
                blob = &mv_q50_256_blob;
            } else {
                static const dx12_shader_blob mv_q50_blob = { g_mul_mat_vec_q5_0_dxil, sizeof(g_mul_mat_vec_q5_0_dxil) };
                blob = &mv_q50_blob;
            }
        } else if (key.src0_type == GGML_TYPE_Q8_0) {
            if (key.flags == 5) {
                static const dx12_shader_blob mv_q80_256_blob = { g_mul_mat_vec_q8_0_256_dxil, sizeof(g_mul_mat_vec_q8_0_256_dxil) };
                blob = &mv_q80_256_blob;
            } else {
                static const dx12_shader_blob mv_q80_blob = { g_mul_mat_vec_q8_0_dxil, sizeof(g_mul_mat_vec_q8_0_dxil) };
                blob = &mv_q80_blob;
            }
        } else {
            // F16/F32 matvec
            if (key.flags == 5) {
                static const dx12_shader_blob mv_l4_blob = { g_mul_mat_vec_load4_dxil, sizeof(g_mul_mat_vec_load4_dxil) };
                blob = &mv_l4_blob;
            } else {
                static const dx12_shader_blob mv_blob = { g_mul_mat_vec_dxil, sizeof(g_mul_mat_vec_dxil) };
                blob = &mv_blob;
            }
        }
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q2_K) {
        static const dx12_shader_blob q2k_blob = { g_mul_mat_q2k_dxil, sizeof(g_mul_mat_q2k_dxil) };
        blob = &q2k_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q3_K) {
        static const dx12_shader_blob q3k_blob = { g_mul_mat_q3k_dxil, sizeof(g_mul_mat_q3k_dxil) };
        blob = &q3k_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q4_K) {
        // Q4_K quantized matmul (batch path)
        static const dx12_shader_blob q4k_blob = { g_mul_mat_q4k_dxil, sizeof(g_mul_mat_q4k_dxil) };
        blob = &q4k_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q5_K) {
        static const dx12_shader_blob q5k_blob = { g_mul_mat_q5k_dxil, sizeof(g_mul_mat_q5k_dxil) };
        blob = &q5k_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q6_K) {
        static const dx12_shader_blob q6k_blob = { g_mul_mat_q6k_dxil, sizeof(g_mul_mat_q6k_dxil) };
        blob = &q6k_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q8_0) {
        static const dx12_shader_blob q80_blob = { g_mul_mat_q8_0_dxil, sizeof(g_mul_mat_q8_0_dxil) };
        blob = &q80_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q5_0) {
        static const dx12_shader_blob q50_blob = { g_mul_mat_q5_0_dxil, sizeof(g_mul_mat_q5_0_dxil) };
        blob = &q50_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q4_0) {
        static const dx12_shader_blob q40_blob = { g_mul_mat_q4_0_dxil, sizeof(g_mul_mat_q4_0_dxil) };
        blob = &q40_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q4_1) {
        static const dx12_shader_blob q41_blob = { g_mul_mat_q4_1_dxil, sizeof(g_mul_mat_q4_1_dxil) };
        blob = &q41_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q5_1) {
        static const dx12_shader_blob q51_blob = { g_mul_mat_q5_1_dxil, sizeof(g_mul_mat_q5_1_dxil) };
        blob = &q51_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q8_1) {
        static const dx12_shader_blob q81_blob = { g_mul_mat_q8_1_dxil, sizeof(g_mul_mat_q8_1_dxil) };
        blob = &q81_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q2_K) {
        static const dx12_shader_blob q2k_gr_blob = { g_get_rows_q2k_dxil, sizeof(g_get_rows_q2k_dxil) };
        blob = &q2k_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q3_K) {
        static const dx12_shader_blob q3k_gr_blob = { g_get_rows_q3k_dxil, sizeof(g_get_rows_q3k_dxil) };
        blob = &q3k_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q4_K) {
        // Q4_K dequantizing get_rows
        static const dx12_shader_blob q4k_gr_blob = { g_get_rows_q4k_dxil, sizeof(g_get_rows_q4k_dxil) };
        blob = &q4k_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q5_K) {
        static const dx12_shader_blob q5k_gr_blob = { g_get_rows_q5k_dxil, sizeof(g_get_rows_q5k_dxil) };
        blob = &q5k_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q6_K) {
        static const dx12_shader_blob q6k_gr_blob = { g_get_rows_q6k_dxil, sizeof(g_get_rows_q6k_dxil) };
        blob = &q6k_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q8_0) {
        static const dx12_shader_blob q80_gr_blob = { g_get_rows_q8_0_dxil, sizeof(g_get_rows_q8_0_dxil) };
        blob = &q80_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q5_0) {
        static const dx12_shader_blob q50_gr_blob = { g_get_rows_q5_0_dxil, sizeof(g_get_rows_q5_0_dxil) };
        blob = &q50_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q4_0) {
        static const dx12_shader_blob q40_gr_blob = { g_get_rows_q4_0_dxil, sizeof(g_get_rows_q4_0_dxil) };
        blob = &q40_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q4_1) {
        static const dx12_shader_blob q41_gr_blob = { g_get_rows_q4_1_dxil, sizeof(g_get_rows_q4_1_dxil) };
        blob = &q41_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q5_1) {
        static const dx12_shader_blob q51_gr_blob = { g_get_rows_q5_1_dxil, sizeof(g_get_rows_q5_1_dxil) };
        blob = &q51_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q8_1) {
        static const dx12_shader_blob q81_gr_blob = { g_get_rows_q8_1_dxil, sizeof(g_get_rows_q8_1_dxil) };
        blob = &q81_gr_blob;
    } else {
        auto sit = g_shader_blobs.find((int)key.op);
        if (sit != g_shader_blobs.end()) {
            blob = &sit->second;
        }
    }

    if (!blob) {
        DX12_LOG_WARN("No shader blob for op %d (flags=%u, src0_type=%d)\n", key.op, key.flags, key.src0_type);
        pipeline_cache[key] = {};
        return &pipeline_cache[key];
    }

    dx12_pipeline pipeline;

    // Create PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
    pso_desc.pRootSignature = common_root_sig.Get();
    pso_desc.CS.pShaderBytecode = blob->data;
    pso_desc.CS.BytecodeLength  = blob->size;

    HRESULT hr = device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pipeline.pso));
    if (FAILED(hr)) {
        DX12_LOG_ERROR("Failed to create PSO for op %d (HRESULT 0x%08X)\n", key.op, (unsigned)hr);
        pipeline_cache[key] = {};
        return &pipeline_cache[key];
    }

    pipeline.root_sig = common_root_sig;
    pipeline_cache[key] = std::move(pipeline);
    return &pipeline_cache[key];
#else
    DX12_LOG_WARN("Shaders not compiled - op %d unavailable\n", key.op);
    pipeline_cache[key] = {};
    return &pipeline_cache[key];
#endif
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

ggml_backend_reg_t ggml_backend_dx12_reg(void) {
    dx12_ensure_initialized();

    g_dx12.backend_reg_obj.api_version = GGML_BACKEND_API_VERSION;
    g_dx12.backend_reg_obj.iface       = dx12_reg_interface;
    g_dx12.backend_reg_obj.context     = nullptr;
    return &g_dx12.backend_reg_obj;
}

ggml_backend_t ggml_backend_dx12_init(size_t dev_num) {
    dx12_ensure_initialized();
    if (dev_num >= g_dx12.devices.size()) {
        DX12_LOG_ERROR("Device %zu not found (have %zu)\n", dev_num, g_dx12.devices.size());
        return nullptr;
    }

    ggml_backend_dev_t dev = dx12_reg_get_device(nullptr, dev_num);
    return dx12_dev_init_backend(dev, nullptr);
}

bool ggml_backend_is_dx12(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, dx12_backend_get_guid());
}

int ggml_backend_dx12_get_device_count(void) {
    dx12_ensure_initialized();
    return (int)g_dx12.devices.size();
}

void ggml_backend_dx12_get_device_description(int device, char * description, size_t description_size) {
    dx12_ensure_initialized();
    if (device < 0 || (size_t)device >= g_dx12.devices.size()) {
        snprintf(description, description_size, "Unknown");
        return;
    }
    snprintf(description, description_size, "%s", g_dx12.devices[device]->name.c_str());
}

void ggml_backend_dx12_get_device_memory(int device, size_t * free, size_t * total) {
    dx12_ensure_initialized();
    if (device < 0 || (size_t)device >= g_dx12.devices.size()) {
        if (free) *free = 0;
        if (total) *total = 0;
        return;
    }
    if (free)  *free  = g_dx12.devices[device]->vram_free;
    if (total) *total = g_dx12.devices[device]->vram_total;
}

ggml_backend_buffer_type_t ggml_backend_dx12_buffer_type(size_t dev_num) {
    dx12_ensure_initialized();
    GGML_ASSERT(dev_num < g_dx12.devices.size());
    ggml_backend_dev_t dev = dx12_reg_get_device(nullptr, dev_num);
    return dx12_dev_get_buffer_type(dev);
}

ggml_backend_buffer_type_t ggml_backend_dx12_host_buffer_type(void) {
    // TODO: Implement upload-heap based host buffer type
    return nullptr;
}

GGML_BACKEND_DL_IMPL(ggml_backend_dx12_reg)
