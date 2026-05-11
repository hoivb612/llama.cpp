// ggml-dx12.cpp - DirectX 12 backend for ggml
//
// Implements a GPU compute backend using D3D12, with optional Cooperative Vector
// acceleration for matrix-vector operations (SM 6.9 / Agility SDK 1.717+).

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#if defined(GGML_DX12_XBOX_GDKX)
// Xbox Series X (Scarlett, GDKX). The Gaming.Xbox.Scarlett.x64 platform
// supplies a Scarlett-specific d3d12_xs.h instead of the desktop d3d12.h.
// There is no DXGI on this partition -- adapter enumeration is replaced
// by a single D3D12XboxCreateDevice call.
#include <windows.h>
#include <d3d12_xs.h>
#include <XGameRuntime.h>
// d3d12_xs.h doesn't define D3D12_HEAP_FLAG_CREATE_NOT_ZEROED -- that flag
// is a desktop D3D12 addition (Win10 19H1) that never made it to the
// Scarlett D3D12 partition because the Scarlett runtime already does not
// zero-init committed resources by default. Alias it to NONE so call sites
// compile; the result is identical (no zero-init) on Xbox.
#ifndef D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
#define D3D12_HEAP_FLAG_CREATE_NOT_ZEROED D3D12_HEAP_FLAG_NONE
#endif
#elif defined(_WIN32)
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

#if defined(GGML_DX12_XBOX_GDKX)
// On the GDKX, D3D12 interfaces inherit from IGraphicsUnknown rather than
// IUnknown. WRL's <wrl/client.h> still works (Microsoft::WRL::ComPtr<T> only
// calls AddRef()/Release() by name and IGraphicsUnknown supplies both), but
// its IID_PPV_ARGS_Helper overload for ComPtrRef static_asserts that
// T::InterfaceType derives from IUnknown:
//   client.h(915,19): error C2338: T has to derive from IUnknown
// d3d12_xs.h provides IID_GRAPHICS_PPV_ARGS as the GDKX equivalent, but it's
// designed for raw T** pointers and doesn't compose with WRL's ComPtrRef
// (which is what `&comptr` produces).
//
// Override IID_PPV_ARGS to a definition that does NOT route through the
// WRL helper. ComPtrRef has its own non-asserting `operator void**() const`
// which returns reinterpret_cast<void**>(ptr_->ReleaseAndGetAddressOf()), so
// a static_cast triggers that conversion and produces a usable void** for
// CreateXxx(... IID, void**). __uuidof(**(ppType)) still resolves correctly
// because ComPtrRef has operator*() returning InterfaceType*.
//
// This only affects the ggml-dx12 TU; we don't poison the macro for any
// other code in the build.
#undef IID_PPV_ARGS
#define IID_PPV_ARGS(ppType) __uuidof(**(ppType)), static_cast<void**>(ppType)
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <deque>
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

// DX12_USE_ZEROCOPY_IO selects the transport for set_tensor / get_tensor /
// memset_tensor on a CCUMA buffer (CUSTOM + L0 + WRITE_BACK heap):
//
//   1 = zero-copy. Map() the GPU buffer, memcpy CPU<->mapped, Unmap() with
//       an explicit written-range. No queue work, no fence wait. Relies on
//       the D3D12 CCUMA cache contract (driver flushes/invalidates the
//       right cache lines around Map/Unmap) AND on the runtime to have
//       quiesced any in-flight GPU writers/readers of the affected range
//       before calling us. ggml_backend_sched calls synchronize before
//       set_tensor on the same backend, so that ordering is normally fine.
//
//   0 = staging. Allocate UPLOAD / READBACK heaps, memcpy CPU<->staging,
//       record a CopyBufferRegion on the compute queue, ExecuteCommandLists,
//       Signal an xfer fence, CPU-wait on it. The wait incidentally drains
//       all prior compute work too, which masks any missing pre-set_tensor
//       sync at the cost of an extra GPU copy + fence round-trip.
//
// Both paths are correctness-safe when the runtime synchronizes before
// set_tensor (which it does today). The Xbox GDKX driver, however,
// CheckFeatureSupport(ARCHITECTURE1) often (mis)reports UMA=false; the
// CCUMA cache contract is therefore not verified by Microsoft for that
// driver. Empirically on GDKX the staging path is also a slight perf win
// for the small per-token tensors (inp_pos, inp_embd, inp_out_ids,
// logits): the GPU-side DMA copy plus a clean READBACK readback are
// cheaper than CPU-side WRITE_BACK store + Map/Unmap overhead on the
// multi-GB device buffer.
//
// Note: dx12_create_buffer still forces is_uma + is_cache_coherent_uma=true
// on Xbox so device buffers go through CUSTOM/L0/WRITE_BACK. That's a
// budget concern (the L1/DEFAULT pool is only ~10 GB of the 13.5 GB title
// budget, and the KV cache otherwise OOMs with HRESULT 0x8007000E) and is
// independent of which set_tensor/get_tensor path runs.
#if defined(GGML_DX12_XBOX_GDKX)
#  define DX12_USE_ZEROCOPY_IO 0
#else
#  define DX12_USE_ZEROCOPY_IO 1
#endif

// Skip leading single/double quotes -- on Windows cmd.exe, `set FOO="1"` stores
// the literal 3-char string `"1"`, which atoi/strtoull parse as 0. Used for all
// numeric GGML_DX12_* env-var reads.
static inline const char * dx12_env_unquote(const char * s) {
    if (!s) return nullptr;
    while (*s == '"' || *s == '\'') ++s;
    return s;
}

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Per-op perf stats (gated by GGML_DX12_PERF env var)
// ---------------------------------------------------------------------------

static std::atomic<int64_t> dx12_op_counts[GGML_OP_COUNT] = {};
static std::atomic<int64_t> dx12_op_time_us[GGML_OP_COUNT] = {};

// MUL_MAT breakdown by quant type: VEC (ne[1]==1) vs GEMM (ne[1]>1)
static std::atomic<int64_t> dx12_mm_vec_counts[GGML_TYPE_COUNT] = {};
static std::atomic<int64_t> dx12_mm_vec_time_us[GGML_TYPE_COUNT] = {};
static std::atomic<int64_t> dx12_mm_gemm_counts[GGML_TYPE_COUNT] = {};
static std::atomic<int64_t> dx12_mm_gemm_time_us[GGML_TYPE_COUNT] = {};

// CPY/CONT/DUP breakdown by shader variant.
//   0 = generic (cpy.hlsl, scalar element-wise, may use atomic CAS for half-word stores)
//   1 = contig fast path (cpy_contig.hlsl, uint4-vectorized, src and dst contiguous + same dtype)
//   (additional variants reserved)
enum dx12_cpy_variant : uint8_t {
    DX12_CPY_GENERIC = 0,
    DX12_CPY_CONTIG  = 1,
    DX12_CPY_VARIANT_COUNT
};
static const char * dx12_cpy_variant_name(uint8_t v) {
    switch (v) {
        case DX12_CPY_GENERIC: return "generic";
        case DX12_CPY_CONTIG:  return "contig";
        default:               return "?";
    }
}
static std::atomic<int64_t> dx12_cpy_counts[DX12_CPY_VARIANT_COUNT] = {};
static std::atomic<int64_t> dx12_cpy_time_us[DX12_CPY_VARIANT_COUNT] = {};

// Forward declaration: drain any deferred PERF readbacks across all live
// backend contexts. Defined further down (needs the complete
// dx12_backend_context type). Called at the top of
// ggml_dx12_print_tensor_op_perf so end-of-run stats include the most recent
// graphs, and can be safely called from anywhere that holds the print path.
static void dx12_drain_all_pending_perf();

static void ggml_dx12_print_tensor_op_perf() {
    // Drain any in-flight PERF readbacks across all live backend contexts so
    // the final stats include the most recent graphs. This is a one-time
    // wait on the last graph or two; insignificant compared to the run.
    dx12_drain_all_pending_perf();
    int64_t total_count = 0;
    int64_t total_time  = 0;
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        total_count += dx12_op_counts[i].load();
        total_time  += dx12_op_time_us[i].load();
    }
    if (total_count == 0) return;

    printf("\n\n[DX12] Per-Op GPU Timing (GGML_DX12_PERF)\n");
    printf("          Total     Total  Tensor\n");
    printf("   Count Time(sec)   %%     Time(us) Tensor Op\n");

    double total_percent = 0.0;
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        int64_t c = dx12_op_counts[i].load();
        int64_t t = dx12_op_time_us[i].load();
        if (c > 0) {
            double percent = (double)t * 100.0 / (double)total_time;
            total_percent += percent;
            printf("%8lld %8.2f  %5.2f   %8.2f GGML_OP_%s\n",
                   (long long)c,
                   (double)t / 1e6,
                   percent,
                   (double)t / (double)c,
                   ggml_op_name((enum ggml_op)i));
        }
    }
    printf("\n%8lld %8.2f %6.2f\n",
           (long long)total_count,
           (double)total_time / 1e6,
           total_percent);

    // MUL_MAT breakdown by src0 quant type
    int64_t mm_total_count = 0, mm_total_time = 0;
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        mm_total_count += dx12_mm_gemm_counts[i].load() + dx12_mm_vec_counts[i].load();
        mm_total_time  += dx12_mm_gemm_time_us[i].load() + dx12_mm_vec_time_us[i].load();
    }
    if (mm_total_count > 0) {
        printf("\nMUL_MAT Src0 Type Frequency\n");
        printf("          Total     Total  Tensor\n");
        printf("   Count Time(sec)   %%     Time(us) Shader / Src0 Type\n");

        for (int i = 0; i < GGML_TYPE_COUNT; i++) {
            int64_t c = dx12_mm_gemm_counts[i].load();
            int64_t t = dx12_mm_gemm_time_us[i].load();
            if (c > 0) {
                printf("%8lld %8.2f  %5.2f   %8.2f MUL_MAT      %s\n",
                       (long long)c, (double)t / 1e6,
                       (double)t * 100.0 / (double)mm_total_time,
                       (double)t / (double)c,
                       ggml_type_name((enum ggml_type)i));
            }
        }
        for (int i = 0; i < GGML_TYPE_COUNT; i++) {
            int64_t c = dx12_mm_vec_counts[i].load();
            int64_t t = dx12_mm_vec_time_us[i].load();
            if (c > 0) {
                printf("%8lld %8.2f  %5.2f   %8.2f MUL_MAT_VEC  %s\n",
                       (long long)c, (double)t / 1e6,
                       (double)t * 100.0 / (double)mm_total_time,
                       (double)t / (double)c,
                       ggml_type_name((enum ggml_type)i));
            }
        }
        printf("\n%8lld %8.2f 100.00\n",
                (long long)mm_total_count, (double)mm_total_time / 1e6);
    }

    // CPY/CONT/DUP breakdown by shader variant
    int64_t cpy_total_count = 0, cpy_total_time = 0;
    for (int i = 0; i < DX12_CPY_VARIANT_COUNT; i++) {
        cpy_total_count += dx12_cpy_counts[i].load();
        cpy_total_time  += dx12_cpy_time_us[i].load();
    }
    if (cpy_total_count > 0) {
        printf("\nCPY/CONT/DUP Variant Frequency\n");
        printf("          Total     Total  Tensor\n");
        printf("   Count Time(sec)   %%     Time(us) Variant\n");
        for (int i = 0; i < DX12_CPY_VARIANT_COUNT; i++) {
            int64_t c = dx12_cpy_counts[i].load();
            int64_t t = dx12_cpy_time_us[i].load();
            if (c > 0) {
                printf("%8lld %8.2f  %5.2f   %8.2f cpy %s\n",
                       (long long)c, (double)t / 1e6,
                       (double)t * 100.0 / (double)cpy_total_time,
                       (double)t / (double)c,
                       dx12_cpy_variant_name((uint8_t)i));
            }
        }
        printf("\n%8lld %8.2f 100.00\n",
                (long long)cpy_total_count, (double)cpy_total_time / 1e6);
    }
    printf("\n");
}

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

// Safe mode: single allocator, sync after every CL, no binding cache.
// Enabled via GGML_DX12_SAFE_MODE=1 env var. Useful for debugging WSL2 issues.
static bool dx12_safe_mode = false;

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
#if defined(GGML_DX12_XBOX_GDKX)
    // No adapter object on Scarlett -- single device per console partition.
#elif defined(_WIN32)
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

    // UMA (Unified Memory Architecture) — APU/integrated GPU
    /* 
    UMA indicates that the GPU shares system memory with the CPU (integrated GPU / APU),
    which allows zero-copy access to buffers allocated in shared memory. On UMA systems,
    if the GPU cache is coherent with the CPU cache (CC-UMA), then writes from the CPU
    are immediately visible to the GPU and vice versa without explicit flushes.

    On non-CacheCoherentUMA (AMD 8060S reports UMA: yes but NOT CC), using WRITE_COMBINE + L0 
    custom heap for UAV buffers causes data corruption. The GPU's L2 cache writes back through 
    write-combine pages inconsistently — coherent for CPU→GPU loads, but not for GPU UAV read-modify-write 
    patterns (KV cache, intermediate compute).

    Caution: Only use D3D12_HEAP_TYPE_CUSTOM (zero-copy) on CacheCoherentUMA systems where hardware 
    guarantees coherency. For plain UMA (like this 8060S), use D3D12_HEAP_TYPE_DEFAULT — the driver 
    still allocates from shared system memory (you still get the 15.2 GB), just without the zero-copy CPU path.

    The VRAM expansion (Dedicated → Shared VRAM) still works — that's separate from the heap type.
    */
    bool is_uma = false;
    bool is_cache_coherent_uma = false;

    // WaveMMA (SM 6.9 Wave Matrix) support
    bool wave_mma_supported = false;
    uint32_t wave_mma_K      = 0;     // hardware K dimension (even multiple of 16)
    uint32_t wave_mma_M      = 0;     // M dimension (16 or 64)
    uint32_t wave_mma_N      = 0;     // N dimension (16 or 64)
    uint32_t wave_mma_wave_size = 0;  // required wave size for WaveMMA
    bool     wave_mma_f16_acc32 = false; // F16 input with F32 accumulator

    // Native 16-bit shader op support (D3D12_FEATURE_D3D12_OPTIONS4).
    // When true, shaders compiled with -enable-16bit-types may use float16_t /
    // int16_t natively. Used to gate the F16 variants of compute shaders
    // (mul_mat WMMA, etc). Probed at device init; honor GGML_DX12_F16=0 to
    // force off for A/B testing.
    bool native16bit_supported = false;
    bool use_f16_shaders       = false;  // final decision after env override

    // Auto-tuning: optimal shader variants per quant type
    // Determined by GPU microbenchmark at first model load
    // Bump TUNE_VERSION when adding new dimensions to invalidate cache
    static constexpr int TUNE_VERSION = 2;
    bool tuning_done = false;
    bool q5_0_use_256 = false;  // Q5_0 matvec: true=256 threads, false=32 threads
    bool q8_0_use_256 = false;  // Q8_0 matvec: true=256 threads, false=32 threads
    bool q6k_use_32   = false;  // Q6_K matvec: true=32 threads, false=256 threads (default=256)
    bool f16_use_load4 = false; // F16 matvec: true=Load4 variant, false=Load2 (default)
    bool q4k_use_32   = false;  // Q4_K matvec: true=32t wave-only, false=256t (UMA optimization)
    bool q5k_use_mr   = false;  // Q5_K matvec: true=multi-row (2 rows/group), false=single-row
    bool q6k_use_mr   = false;  // Q6_K matvec: true=multi-row (2 rows/group), false=single-row
    bool fa_use_uma   = false;  // Flash Attention: true=UMA variant (128t, D≤128), false=standard (256t)

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

#if defined(GGML_DX12_XBOX_GDKX)
    void init(size_t idx);
#elif defined(_WIN32)
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

static const int CMD_RING_SIZE = 16;

struct dx12_backend_context {
    dx12_device * dev = nullptr;

    // Command allocator ring — multiple allocators so CPU can record while GPU executes
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

    // Split-KV flash attention temp buffer
    ComPtr<ID3D12Resource> splitkv_temp;
    size_t                 splitkv_temp_size = 0;

    // FA mask staging buffer -- protects against compute buffer aliasing
    // where a prior FA output overwrites a later FA's mask input.
    ComPtr<ID3D12Resource> fa_mask_staging;
    size_t                 fa_mask_staging_size = 0;

    bool cmd_list_open = false;

    // --- Graph cache (GGML_DX12_GRAPH_CACHE, on by default) ---
    //
    // Each unique cgraph topology is recorded into a dedicated CL once, then
    // replayed via ExecuteCommandLists on subsequent matching graphs. This
    // amortizes the per-dispatch CPU cost (root constants, descriptor binds,
    // PSO swap) over hundreds of decode iterations -- the runtime's
    // "graphs reused" counter shows ~97% of decode graphs share topology.
    //
    // Eligibility is decided per graph in dx12_graph_compute (see
    // dx12_graph_cache_eligible). On a miss for an eligible graph we record;
    // on a hit we replay. Live (non-cached) work continues to use the
    // CMD_RING_SIZE allocator ring.
    struct dx12_graph_cache_entry {
        uint64_t                            hash         = 0;
        int                                 n_nodes      = 0;
        int                                 n_dispatches = 0;
        ComPtr<ID3D12CommandAllocator>      alloc;
        ComPtr<ID3D12GraphicsCommandList>   cl;     // closed, ready to ExecuteCommandLists
    };
    std::unordered_map<uint64_t, dx12_graph_cache_entry> graph_cache;

    // While true, the dispatch loop is recording into a cached CL. Mid-graph
    // CL splits (decode_flush, head split, etc) are suppressed.
    bool cache_recording = false;
    // Set to the entry being recorded (so ts/diag code can read n_dispatches)
    dx12_graph_cache_entry * cache_recording_entry = nullptr;

    // --- GPU-timestamp perf (GGML_DX12_PERF) ---
    // Deferred-readback model: each graph reserves a contiguous slot range in
    // a ring inside ts_heap, emits EndQuery(TIMESTAMP) before+after each
    // Dispatch on the live compute CL, then at end of graph issues a single
    // ResolveQueryData on a fresh CL (queue-ordered after the data CLs --
    // submission order alone gives the right read-after-write semantics on a
    // compute queue) and pushes a {slot range, fence value, records} entry
    // onto a pending queue. No CPU-side fence wait. The next graph polls the
    // queue front for completion (no wait), accumulates whatever is ready,
    // then reserves its own slot range. ggml_dx12_print_tensor_op_perf does a
    // final drain (with wait) across all live backend contexts.
    static constexpr uint32_t TS_HEAP_SIZE = 32768;  // ~14 Phi-3 graphs in flight
    ComPtr<ID3D12QueryHeap> ts_heap;
    ComPtr<ID3D12Resource>  ts_readback;
    uint64_t                ts_freq = 0;        // ticks per second (GetTimestampFrequency)
    uint32_t                ts_next_slot = 0;   // ring write head (slot index)
    uint32_t                ts_graph_start_slot = 0; // first slot of the in-progress graph
    bool                    ts_graph_active = false; // ts_begin_graph reserved a range
    struct ts_record {
        uint16_t op;          // ggml_op
        uint8_t  src0_type;   // ggml_type (or 0xFF if N/A)
        uint8_t  variant;     // sub-shader id; meaning depends on op:
                              //   MUL_MAT: 0=GEMM, 1=matvec
                              //   CPY/CONT/DUP: dx12_cpy_variant
                              //   other:   unused (0)
        uint32_t slot_pair;   // absolute slot index in ts_heap (end is +1)
    };
    std::vector<ts_record> ts_records;       // records for the in-progress graph
    struct ts_pending {
        uint32_t                slot_lo = 0; // [slot_lo, slot_hi) range in ts_heap / ts_readback
        uint32_t                slot_hi = 0;
        uint64_t                fence_value = 0; // signaled when ResolveQueryData is done
        std::vector<ts_record>  records;
    };
    std::deque<ts_pending> ts_pending_q;
    bool ts_init_failed = false; // set if heap creation failed; falls back to wall-clock

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
        if (fence && dev) {
            wait_for_gpu();
            // Drain any remaining PERF pending entries so their timing data
            // makes it into the globals before this context is gone. Safe
            // because wait_for_gpu has already flushed all GPU work.
            ts_drain_all_with_wait();
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

    // GPU-timestamp perf helpers. ts_init() is idempotent; the rest are no-ops
    // unless ts_init() succeeded.
    bool ts_init();
    void ts_begin_graph(int upper_dispatches); // reserve slot range; poll-drain completed pending
    void ts_record_dispatch_pre();          // emits start EndQuery on cmd_list
    void ts_record_dispatch_post(uint16_t op, uint8_t src0_type, uint8_t variant); // emits end EndQuery + records
    void ts_resolve_pending();              // legacy stub; resolve happens once in ts_finalize_graph
    void ts_finalize_graph();               // submit ResolveQueryData CL, push pending, return without wait
    void ts_poll_drain();                   // accumulate any pending entries whose fence has signaled
    void ts_drain_one_with_wait();          // accumulate front pending, blocking on its fence
    void ts_drain_all_with_wait();          // drain entire pending queue (used at print / shutdown)
    void ts_accumulate(const ts_pending & p); // map readback range and add to globals
};

// Global registry of live backend contexts. Populated by dx12_register_bctx
// from dx12_dev_init_backend, drained by dx12_unregister_bctx from
// dx12_backend_free. Used by ggml_dx12_print_tensor_op_perf (via the
// dx12_drain_all_pending_perf forward decl above) to drain any deferred
// PERF readbacks at end-of-run.
static std::mutex                          g_bctx_list_mutex;
static std::vector<dx12_backend_context *> g_bctx_list;

static void dx12_register_bctx(dx12_backend_context * b) {
    std::lock_guard<std::mutex> lk(g_bctx_list_mutex);
    g_bctx_list.push_back(b);
}
static void dx12_unregister_bctx(dx12_backend_context * b) {
    std::lock_guard<std::mutex> lk(g_bctx_list_mutex);
    auto it = std::find(g_bctx_list.begin(), g_bctx_list.end(), b);
    if (it != g_bctx_list.end()) g_bctx_list.erase(it);
}
static void dx12_drain_all_pending_perf() {
    std::lock_guard<std::mutex> lk(g_bctx_list_mutex);
    for (auto * b : g_bctx_list) {
        if (b) b->ts_drain_all_with_wait();
    }
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static struct dx12_global_state {
    bool                                        initialized = false;
#if defined(GGML_DX12_XBOX_GDKX)
    // No factory on Scarlett -- D3D12XboxCreateDevice is called directly.
#elif defined(_WIN32)
    ComPtr<IDXGIFactory4>                       factory;
#else
    ComPtr<IDXCoreAdapterFactory>               factory;
#endif
    std::vector<std::unique_ptr<dx12_device>>   devices;
    std::mutex                                  init_mutex;

    // Backend device & registry objects
    std::vector<ggml_backend_device> backend_devices;
    ggml_backend_reg               backend_reg_obj = {};

#ifndef _WIN32
    // WSL2: D3D12 runtime (libd3d12core.so) calls abort() if COM objects are
    // released during __cxa_finalize. Leak intentionally — OS reclaims at exit.
    ~dx12_global_state() {
        for (auto & dev : devices) {
            dev.release(); // release unique_ptr ownership without calling destructor
        }
        factory.Detach(); // release ComPtr ownership without calling Release()
    }
#endif
} g_dx12;

// ---------------------------------------------------------------------------
// Device initialization
// ---------------------------------------------------------------------------

static void dx12_ensure_initialized() {
    std::lock_guard<std::mutex> lock(g_dx12.init_mutex);
    if (g_dx12.initialized) return;

    // Check safe mode
    dx12_safe_mode = (getenv("GGML_DX12_SAFE_MODE") != nullptr);
    if (dx12_safe_mode) {
        DX12_LOG_INFO("Safe mode enabled (single allocator, sync after every CL)\n");
    }

#if defined(GGML_DX12_XBOX_GDKX)
    // ---- Xbox Series X (Scarlett) single-device init ----
    //
    // No DXGI / DXCore enumeration on this partition. The platform exposes
    // exactly one D3D12 device per title, created directly via
    // D3D12XboxCreateDevice.
    //
    // The debug layer and experimental-feature surfaces don't exist here --
    // use PIX-on-Xbox for capture and validation instead.
    {
        // XGameRuntime services (XSystem, XUser, XStore, XGameSave, XLaunch,
        // XClosedCaptions, XGameInvite, ...) require XGameRuntimeInitialize()
        // to be called once before any other XGame* API. None of these are
        // currently used by the dx12 backend itself, but a real Scarlett
        // package deploy needs the runtime initialized for the OS to treat
        // the process as a game title (HDMI handoff, suspend/resume, QoS
        // scheduling on the title cores). Calling it here makes the backend
        // self-sufficient instead of pushing the requirement up to every
        // host (minslm-cli, server, etc.).
        //
        // Failure is non-fatal: when the process runs as a plain desktop
        // .exe against the GDKX runtime (no MicrosoftGame.config, no game
        // manifest), XGameRuntimeInitialize returns an error and we just
        // continue without it -- D3D12 device creation still works.
        HRESULT hr_xg = XGameRuntimeInitialize();
        if (SUCCEEDED(hr_xg)) {
            DX12_LOG_INFO("XGameRuntimeInitialize: ok\n");
        } else {
            DX12_LOG_WARN("XGameRuntimeInitialize failed (HRESULT 0x%08X) -- continuing without XGame* services (desktop .exe mode)\n",
                          (unsigned)hr_xg);
        }

        D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {};
        params.Version             = D3D12_SDK_VERSION;
        params.GraphicsCommandQueueRingSizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
        params.GraphicsScratchMemorySizeBytes    = D3D12XBOX_DEFAULT_SIZE_BYTES;
        params.ComputeScratchMemorySizeBytes     = D3D12XBOX_DEFAULT_SIZE_BYTES;
        if (getenv("DX12_DEBUG")) {
            params.ProcessDebugFlags = D3D12_PROCESS_DEBUG_FLAG_DEBUG_LAYER_ENABLED;
            DX12_LOG_INFO("D3D12 debug layer requested (DX12_DEBUG)\n");
        }

        ComPtr<ID3D12Device> probe_device;
        HRESULT hr = D3D12XboxCreateDevice(nullptr, &params, IID_PPV_ARGS(&probe_device));
        DX12_CHECK(hr, "D3D12XboxCreateDevice");
        probe_device.Reset(); // dx12_device::init() will create its own

        g_dx12.devices.push_back(std::make_unique<dx12_device>());
        g_dx12.devices.back()->init(0);
    }
#else
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
#endif // GGML_DX12_XBOX_GDKX

    DX12_LOG_INFO("Found %zu D3D12 device(s)\n", g_dx12.devices.size());
    g_dx12.initialized = true;
}

// ---------------------------------------------------------------------------
// dx12_device implementation
// ---------------------------------------------------------------------------

#if defined(GGML_DX12_XBOX_GDKX)
void dx12_device::init(size_t idx) {
    dev_index   = idx;
    description = "Xbox Series X";
    name        = std::string(GGML_DX12_NAME) + std::to_string(idx);

    // Single device per partition. Re-create here -- the probe in
    // dx12_ensure_initialized() was discarded after success.
    D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {};
    params.Version             = D3D12_SDK_VERSION;
    params.GraphicsCommandQueueRingSizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
    params.GraphicsScratchMemorySizeBytes    = D3D12XBOX_DEFAULT_SIZE_BYTES;
    params.ComputeScratchMemorySizeBytes     = D3D12XBOX_DEFAULT_SIZE_BYTES;
    if (getenv("DX12_DEBUG")) {
        params.ProcessDebugFlags = D3D12_PROCESS_DEBUG_FLAG_DEBUG_LAYER_ENABLED;
    }
    HRESULT hr = D3D12XboxCreateDevice(nullptr, &params, IID_PPV_ARGS(&device));
    DX12_CHECK(hr, "D3D12XboxCreateDevice");

    // Compute command queue
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type     = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    qd.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&compute_queue));
    DX12_CHECK(hr, "CreateCommandQueue(compute)");

    // VRAM: Series X reports ~13.5 GB of game-available memory (16 GB physical
    // minus OS reservation). The actual budget can be queried via
    // XGameRuntime title-memory APIs; for the prototype we use a fixed budget
    // and let the caller override via GGML_DX12_VRAM_OVERRIDE_MB.
    vram_total = (size_t)13ull * 1024 * 1024 * 1024 + (size_t)512 * 1024 * 1024; // 13.5 GiB
    if (const char * ovr = getenv("GGML_DX12_VRAM_OVERRIDE_MB")) {
        size_t mb = (size_t)strtoull(dx12_env_unquote(ovr), nullptr, 10);
        if (mb > 0) {
            vram_total = mb * 1024ull * 1024ull;
            DX12_LOG_INFO("VRAM override: %zu MiB\n", mb);
        }
    }
    vram_free = vram_total;

    // AMD Scarlett (RDNA 2 custom). Vendor 0x1002, device id is custom and
    // doesn't matter for our purposes.
    vendor_id = 0x1002;
    device_id = 0;
#elif defined(_WIN32)
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
#ifndef GGML_DX12_XBOX_GDKX
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
#endif // !GGML_DX12_XBOX_GDKX

    // Check WaveMMA (SM 6.9 Wave Matrix) support
    // D3D12_FEATURE_WAVE_MMA queries hardware matrix multiply-accumulate capability
    wave_mma_supported = false;
#ifndef GGML_DX12_XBOX_GDKX
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
#endif // !GGML_DX12_XBOX_GDKX

    // Native 16-bit shader op support (D3D12_FEATURE_D3D12_OPTIONS4 = 23).
    // The struct layout is:
    //   BOOL MSAA64KBAlignedTextureSupported;
    //   D3D12_SHARED_RESOURCE_COMPATIBILITY_TIER SharedResourceCompatibilityTier;
    //   BOOL Native16BitShaderOpsSupported;
    // We only care about the last field. Defined inline to avoid a hard
    // dependency on a specific Agility SDK header revision.
    native16bit_supported = false;
#if defined(GGML_DX12_XBOX_GDKX)
    // Scarlett supports native 16-bit shader ops unconditionally.
    native16bit_supported = true;
#else
    {
        struct {
            BOOL MSAA64KBAlignedTextureSupported;
            UINT SharedResourceCompatibilityTier;
            BOOL Native16BitShaderOpsSupported;
        } opts4 = {};
        // D3D12_FEATURE_D3D12_OPTIONS4 = 23
        HRESULT hr2 = device->CheckFeatureSupport((D3D12_FEATURE)23, &opts4, sizeof(opts4));
        if (SUCCEEDED(hr2) && opts4.Native16BitShaderOpsSupported) {
            native16bit_supported = true;
        }
    }
#endif

    // GGML_DX12_F16 env var: 0 = force off, 1 = use if hardware supports, unset = auto (on if supported).
    use_f16_shaders = native16bit_supported;
    if (const char * env = getenv("GGML_DX12_F16")) {
        int v = atoi(env);
        if (v == 0) {
            if (use_f16_shaders) {
                DX12_LOG_INFO("GGML_DX12_F16=0 set: disabling F16 shader variants\n");
            }
            use_f16_shaders = false;
        } else if (v != 0 && !native16bit_supported) {
            DX12_LOG_INFO("GGML_DX12_F16=%d but device lacks Native16BitShaderOps -- staying on F32 shaders\n", v);
            use_f16_shaders = false;
        }
    }

    // Query wave lane count (D3D12_FEATURE_D3D12_OPTIONS1)
    uint32_t wave_lane_min = 0, wave_lane_max = 0;
    {
        struct {
            BOOL WaveOps;
            UINT WaveLaneCountMin;
            UINT WaveLaneCountMax;
            UINT TotalLaneCount;
        } opts1 = {};
        HRESULT hr2 = device->CheckFeatureSupport((D3D12_FEATURE)8, &opts1, sizeof(opts1));
        if (SUCCEEDED(hr2)) {
            wave_lane_min = opts1.WaveLaneCountMin;
            wave_lane_max = opts1.WaveLaneCountMax;
        }
    }

    // Query UMA (Unified Memory Architecture) — APU/integrated GPU zero-copy path
    {
        D3D12_FEATURE_DATA_ARCHITECTURE1 arch1 = {};
        HRESULT hr2 = device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch1, sizeof(arch1));
        if (SUCCEEDED(hr2)) {
            is_uma = (arch1.UMA != FALSE);
            is_cache_coherent_uma = (arch1.CacheCoherentUMA != FALSE);
        } else {
            // Fallback: D3D12_FEATURE_ARCHITECTURE (no CacheCoherent field)
            D3D12_FEATURE_DATA_ARCHITECTURE arch0 = {};
            hr2 = device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch0, sizeof(arch0));
            if (SUCCEEDED(hr2)) {
                is_uma = (arch0.UMA != FALSE);
                is_cache_coherent_uma = (arch0.CacheCoherentUMA != FALSE);
            }
        }
    }

#if defined(GGML_DX12_XBOX_GDKX)
    // Scarlett is architecturally UMA + cache-coherent UMA: a single chip with
    // a single GDDR6 pool shared between the Zen 2 CPU cores and the RDNA 2
    // GPU, fully cache-coherent. The GDKX driver, however, presents two memory
    // pools (L0/L1) to mimic the desktop "VRAM + system RAM" split, and
    // CheckFeatureSupport(ARCHITECTURE1) often (mis)reports UMA=false on it.
    //
    // When UMA=false, dx12_create_buffer() routes through D3D12_HEAP_TYPE_DEFAULT,
    // which on Scarlett maps to the ~10 GB L1 ("GPU-optimal") segment only --
    // not the full ~13.5 GB title-memory budget. Once the model + D3D12XBOX
    // scratch + system reservations consume enough of L1, large contiguous
    // allocations (e.g. the KV cache, hundreds of MiB) fail with
    // E_OUTOFMEMORY (HRESULT 0x8007000E) even though plenty of title memory
    // is free.
    //
    // Forcing the (correct) UMA + CC-UMA values here makes dx12_create_buffer
    // use D3D12_HEAP_TYPE_CUSTOM + L0 + WRITE_BACK, which sees the full title
    // budget and is also the zero-copy fast path.
    is_uma = true;
    is_cache_coherent_uma = true;
#endif

    // Environment override: GGML_DX12_NO_UMA=1 disables UMA optimizations for testing
    if (is_uma && getenv("GGML_DX12_NO_UMA")) {
        DX12_LOG_INFO("GGML_DX12_NO_UMA set: disabling UMA/CC-UMA optimizations\n");
        is_uma = false;
        is_cache_coherent_uma = false;
    }

    // UMA memory adjustment: on UMA systems, the GPU can access all system RAM.
    // Report SharedSystemMemory so the model loader puts all layers on GPU.
    // Xbox already reports the correct game-available budget directly, so
    // the desktop-style expansion is skipped there.
    if (is_uma && vram_total < (size_t)2 * 1024 * 1024 * 1024) {
#if defined(GGML_DX12_XBOX_GDKX)
        size_t shared = vram_total;
#elif defined(_WIN32)
        size_t shared = adapter_desc.SharedSystemMemory;
#else
        size_t shared = 0;
        adapter->GetProperty(DXCoreAdapterProperty::SharedSystemMemory, sizeof(shared), &shared);
#endif
        if (shared > vram_total) {
            DX12_LOG_INFO("UMA detected: expanding VRAM from %.1f GB to %.1f GB (shared system memory)\n",
                          (double)vram_total / (1024.0 * 1024.0 * 1024.0),
                          (double)shared / (1024.0 * 1024.0 * 1024.0));
            vram_total = shared;
            vram_free  = shared;
        }
    }

    create_common_root_signature();

    DX12_LOG_INFO("Device %zu: %s (%s, VRAM: %.1f GB, Wave: %u-%u, CV: %s, WaveMMA: %s%s, F16: %s, UMA: %s)\n",
                  idx, name.c_str(), description.c_str(),
                  (double)vram_total / (1024.0 * 1024.0 * 1024.0),
                  wave_lane_min, wave_lane_max,
                  cooperative_vector_supported ? "yes" : "no",
                  wave_mma_supported ? "yes" : "no",
                  wave_mma_supported ? (std::string(" K=") + std::to_string(wave_mma_K) +
                                        " wave=" + std::to_string(wave_mma_wave_size) +
                                        (wave_mma_f16_acc32 ? " f16->f32" : " f16->f16")).c_str() : "",
                  use_f16_shaders ? "on" : (native16bit_supported ? "off" : "no"),
                  is_uma ? (is_cache_coherent_uma ? "CC" : "yes") : "no");
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

    // Pick the next allocator in the ring (safe mode: always use slot 0)
    int slot = dx12_safe_mode ? 0 : cmd_ring_head;
    if (!dx12_safe_mode) {
        cmd_ring_head = (cmd_ring_head + 1) % CMD_RING_SIZE;
    }

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
    int submitted_slot = dx12_safe_mode ? 0 : (cmd_ring_head + CMD_RING_SIZE - 1) % CMD_RING_SIZE;
    cmd_alloc_fence[submitted_slot] = fence_value;

    cmd_list_open = false;

    // Safe mode: force full GPU sync after every submission
    if (dx12_safe_mode) {
        wait_for_gpu();
    }
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

// ---------------------------------------------------------------------------
// GPU-timestamp perf (GGML_DX12_PERF)
// Pattern matches Xbox-GDK-Samples/Kits/ATGTK/PerformanceTimers (GDKX-tested
// on a COMPUTE command queue via FastBlockCompress). Stock D3D12 APIs only:
//   - D3D12_QUERY_HEAP_TYPE_TIMESTAMP heap on the device
//   - EndQuery(TIMESTAMP) on the live compute CL (TIMESTAMP queries don't use
//     BeginQuery -- a single EndQuery records the GPU clock at that pipeline
//     point).
//   - ResolveQueryData() on the same CL just before Close()
//   - Fence wait, then map the readback heap and convert ticks -> us using
//     ID3D12CommandQueue::GetTimestampFrequency().
// ---------------------------------------------------------------------------
bool dx12_backend_context::ts_init() {
    if (ts_heap)         return true;
    if (ts_init_failed)  return false;

    HRESULT hr = dev->compute_queue->GetTimestampFrequency(&ts_freq);
    if (FAILED(hr) || ts_freq == 0) {
        DX12_LOG_WARN("GGML_DX12_PERF: GetTimestampFrequency failed (hr=0x%08lX) -- per-op timing disabled\n",
                      (unsigned long)hr);
        ts_init_failed = true;
        return false;
    }

    D3D12_QUERY_HEAP_DESC qhd = {};
    qhd.Type  = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.Count = TS_HEAP_SIZE;
    hr = dev->device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&ts_heap));
    if (FAILED(hr)) {
        DX12_LOG_WARN("GGML_DX12_PERF: CreateQueryHeap(TIMESTAMP) failed (hr=0x%08lX) -- per-op timing disabled\n",
                      (unsigned long)hr);
        ts_init_failed = true;
        return false;
    }

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width              = TS_HEAP_SIZE * sizeof(uint64_t);
    rd.Height             = 1;
    rd.DepthOrArraySize   = 1;
    rd.MipLevels          = 1;
    rd.Format             = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count   = 1;
    rd.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    hr = dev->device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&ts_readback));
    if (FAILED(hr)) {
        DX12_LOG_WARN("GGML_DX12_PERF: CreateCommittedResource(readback) failed (hr=0x%08lX) -- per-op timing disabled\n",
                      (unsigned long)hr);
        ts_heap.Reset();
        ts_init_failed = true;
        return false;
    }

    ts_records.reserve(TS_HEAP_SIZE / 2);
    DX12_LOG_INFO("GGML_DX12_PERF: GPU-timestamp mode (freq=%llu Hz, %u slots)\n",
                  (unsigned long long)ts_freq, (unsigned)TS_HEAP_SIZE);
    return true;
}

void dx12_backend_context::ts_record_dispatch_pre() {
    if (!ts_heap || !ts_graph_active) return;
    if (ts_next_slot + 2 > TS_HEAP_SIZE) return; // shouldn't happen if begin reserved correctly
    cmd_list->EndQuery(ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts_next_slot);
}

void dx12_backend_context::ts_record_dispatch_post(uint16_t op, uint8_t src0_type, uint8_t variant) {
    if (!ts_heap || !ts_graph_active) return;
    if (ts_next_slot + 2 > TS_HEAP_SIZE) return;
    cmd_list->EndQuery(ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts_next_slot + 1);
    ts_records.push_back({ op, src0_type, variant, ts_next_slot });
    ts_next_slot += 2;
}

void dx12_backend_context::ts_resolve_pending() {
    // Legacy stub. With the deferred-readback model, ResolveQueryData is
    // issued exactly once per graph in ts_finalize_graph() on a CL submitted
    // *after* all of that graph's data CLs. Compute-queue submission order
    // is the read-after-write barrier; no CPU wait is required.
}

// Reserve a slot range in the ring for the upcoming graph. Polls completed
// pending entries first (no wait), then waits only if a still-in-flight entry
// occupies the prospective range -- in steady state this never blocks because
// fence completion runs sub-millisecond behind GPU execution.
void dx12_backend_context::ts_begin_graph(int upper_dispatches) {
    if (!ts_heap) return;
    ts_records.clear();
    ts_graph_active = false;

    // Drain any already-completed entries -- updates the globals at near-zero
    // CPU cost and frees ring slots.
    ts_poll_drain();

    if (upper_dispatches <= 0) return;
    uint32_t needed = (uint32_t)upper_dispatches * 2;
    if (needed > TS_HEAP_SIZE) {
        // Graph too large to fit. Skip PERF for this one rather than abort.
        DX12_LOG_WARN("GGML_DX12_PERF: graph dispatch upper bound (%d) exceeds heap (%u slots) -- skipping PERF for this graph\n",
                      upper_dispatches, (unsigned)TS_HEAP_SIZE);
        return;
    }

    // Wrap if the prospective range would run off the end of the ring.
    if (ts_next_slot + needed > TS_HEAP_SIZE) {
        ts_next_slot = 0;
    }

    // If a not-yet-completed pending entry overlaps the prospective range,
    // drain it (with wait) until the range is clear. Pending entries are
    // FIFO and slot ranges are allocated in order, so this only ever waits
    // on the oldest few entries.
    auto overlaps_prospective = [&](const ts_pending & p) {
        uint32_t a = ts_next_slot;
        uint32_t b = ts_next_slot + needed;
        return (a < p.slot_hi) && (p.slot_lo < b);
    };
    while (!ts_pending_q.empty() && overlaps_prospective(ts_pending_q.front())) {
        ts_drain_one_with_wait();
    }

    ts_graph_start_slot = ts_next_slot;
    ts_graph_active     = true;
}

// Map the readback range corresponding to a pending entry's slots, accumulate
// per-record deltas into the globals, unmap. No fence interaction.
void dx12_backend_context::ts_accumulate(const ts_pending & p) {
    if (!ts_readback || p.slot_hi <= p.slot_lo || p.records.empty()) return;

    D3D12_RANGE rr = { p.slot_lo * sizeof(uint64_t), p.slot_hi * sizeof(uint64_t) };
    void * mapped = nullptr;
    HRESULT hr = ts_readback->Map(0, &rr, &mapped);
    if (FAILED(hr) || !mapped) {
        DX12_LOG_WARN("GGML_DX12_PERF: ts_readback->Map failed (hr=0x%08lX) -- dropping %zu records\n",
                      (unsigned long)hr, p.records.size());
        return;
    }

    const uint64_t * timings   = (const uint64_t *)mapped;
    const double     tick_to_us = 1000000.0 / (double)ts_freq;

    // One-shot diagnostic: print first few records of the first drained entry
    // so we can sanity-check the deferred path is producing the same shape of
    // data as the old synchronous path.
    static bool ts_diag_done = false;
    if (!ts_diag_done) {
        ts_diag_done = true;
        fprintf(stderr,
            "ggml-dx12: TS DIAG (first deferred drain): freq=%llu Hz, slot_range=[%u..%u), n_records=%zu\n",
            (unsigned long long)ts_freq, p.slot_lo, p.slot_hi, p.records.size());
        size_t n_show = std::min<size_t>(8, p.records.size());
        for (size_t i = 0; i < n_show; i++) {
            const auto & r = p.records[i];
            uint64_t t0 = timings[r.slot_pair];
            uint64_t t1 = timings[r.slot_pair + 1];
            int64_t  dt = (int64_t)t1 - (int64_t)t0;
            fprintf(stderr,
                "  rec[%zu] op=%u src0=%u var=%u slot=%u  t0=%llu t1=%llu  delta=%lld ticks (%.2f us)\n",
                i, (unsigned)r.op, (unsigned)r.src0_type, (unsigned)r.variant,
                r.slot_pair,
                (unsigned long long)t0, (unsigned long long)t1,
                (long long)dt, (double)dt * tick_to_us);
        }
        fflush(stderr);
    }

    for (const auto & r : p.records) {
        uint64_t t0 = timings[r.slot_pair];
        uint64_t t1 = timings[r.slot_pair + 1];
        if (t1 <= t0) continue;
        int64_t us = (int64_t)((double)(t1 - t0) * tick_to_us);
        if (us <= 0) continue;

        dx12_op_counts[r.op].fetch_add(1, std::memory_order_relaxed);
        dx12_op_time_us[r.op].fetch_add(us, std::memory_order_relaxed);
        if (r.op == GGML_OP_MUL_MAT && r.src0_type < GGML_TYPE_COUNT) {
            if (r.variant) {
                dx12_mm_vec_counts[r.src0_type].fetch_add(1, std::memory_order_relaxed);
                dx12_mm_vec_time_us[r.src0_type].fetch_add(us, std::memory_order_relaxed);
            } else {
                dx12_mm_gemm_counts[r.src0_type].fetch_add(1, std::memory_order_relaxed);
                dx12_mm_gemm_time_us[r.src0_type].fetch_add(us, std::memory_order_relaxed);
            }
        } else if ((r.op == GGML_OP_CPY || r.op == GGML_OP_CONT || r.op == GGML_OP_DUP)
                   && r.variant < DX12_CPY_VARIANT_COUNT) {
            dx12_cpy_counts[r.variant].fetch_add(1, std::memory_order_relaxed);
            dx12_cpy_time_us[r.variant].fetch_add(us, std::memory_order_relaxed);
        }
    }

    D3D12_RANGE empty = { 0, 0 };
    ts_readback->Unmap(0, &empty);
}

void dx12_backend_context::ts_poll_drain() {
    while (!ts_pending_q.empty()) {
        const auto & front = ts_pending_q.front();
        if (fence->GetCompletedValue() < front.fence_value) break;
        ts_accumulate(front);
        ts_pending_q.pop_front();
    }
}

void dx12_backend_context::ts_drain_one_with_wait() {
    if (ts_pending_q.empty()) return;
    const auto & front = ts_pending_q.front();
    wait_for_fence(front.fence_value);
    ts_accumulate(front);
    ts_pending_q.pop_front();
}

void dx12_backend_context::ts_drain_all_with_wait() {
    while (!ts_pending_q.empty()) {
        ts_drain_one_with_wait();
    }
}

// End-of-graph: submit ResolveQueryData on a fresh CL, capture its fence
// value, push pending entry, return WITHOUT waiting. The compute queue
// guarantees the resolve sees all prior EndQuery writes by submission order.
void dx12_backend_context::ts_finalize_graph() {
    if (!ts_heap || !ts_graph_active) {
        ts_records.clear();
        ts_graph_active = false;
        return;
    }
    if (ts_records.empty() || ts_next_slot == ts_graph_start_slot) {
        ts_records.clear();
        ts_graph_active = false;
        return;
    }

    // Submit any pending data CL so its EndQueries are queued for execution
    // before the resolve CL. Submission order on the compute queue is the
    // synchronization point.
    if (cmd_list_open) {
        close_and_execute();
    }

    const uint32_t lo = ts_graph_start_slot;
    const uint32_t hi = ts_next_slot;

    ensure_cmd_list_open();
    cmd_list->ResolveQueryData(
        ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
        lo, hi - lo,
        ts_readback.Get(), (UINT64)lo * sizeof(uint64_t));
    close_and_execute();   // bumps fence_value and signals

    ts_pending p;
    p.slot_lo      = lo;
    p.slot_hi      = hi;
    p.fence_value  = fence_value;
    p.records.swap(ts_records); // ts_records left empty for next graph
    ts_pending_q.push_back(std::move(p));

    ts_graph_active = false;
    // ts_next_slot stays as-is so the next ts_begin_graph allocates after us.
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

    if (dev->is_cache_coherent_uma) {
        // CacheCoherentUMA: CUSTOM heap with WRITE_BACK is zero-copy for both CPU and GPU.
        // GPU UAV + CPU Map/Unmap with full cache coherency guaranteed.
        hp.Type                 = D3D12_HEAP_TYPE_CUSTOM;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        hp.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    } else {
        // Non-CC UMA or discrete: use DEFAULT heap. On UMA systems, DEFAULT still
        // allocates from shared system memory (driver manages coherency internally).
        // WRITE_COMBINE + L0 caused data corruption on AMD APUs (non-coherent GPU
        // cache writeback through write-combine pages produces stale UAV data).
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    }

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
    ctx->heap_type = dev->is_cache_coherent_uma ? D3D12_HEAP_TYPE_CUSTOM : D3D12_HEAP_TYPE_DEFAULT;

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

            size_t tensor_offset = dx12_tensor_offset(tensor) + offset;

            // UMA zero-copy: direct memset
            if (DX12_USE_ZEROCOPY_IO && ctx->heap_type == D3D12_HEAP_TYPE_CUSTOM) {
                void * mapped = nullptr;
                D3D12_RANGE read_range = { 0, 0 };
                HRESULT hr = ctx->resource->Map(0, &read_range, &mapped);
                DX12_CHECK(hr, "Map(UMA memset)");
                memset((uint8_t *)mapped + tensor_offset, value, size);
                D3D12_RANGE written = { tensor_offset, tensor_offset + size };
                ctx->resource->Unmap(0, &written);
                return;
            }

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

            size_t tensor_offset = dx12_tensor_offset(tensor) + offset;

            // UMA zero-copy: buffer is CPU-writable, write directly
            if (DX12_USE_ZEROCOPY_IO && ctx->heap_type == D3D12_HEAP_TYPE_CUSTOM) {
                void * mapped = nullptr;
                D3D12_RANGE read_range = { 0, 0 };
                HRESULT hr = ctx->resource->Map(0, &read_range, &mapped);
                DX12_CHECK(hr, "Map(UMA set_tensor)");
                memcpy((uint8_t *)mapped + tensor_offset, data, size);
                D3D12_RANGE written = { tensor_offset, tensor_offset + size };
                ctx->resource->Unmap(0, &written);
                return;
            }
            
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

            size_t tensor_offset = dx12_tensor_offset(tensor) + offset;

            // UMA zero-copy: buffer is CPU-readable (WRITE_BACK) or slow-read (WRITE_COMBINE)
            if (DX12_USE_ZEROCOPY_IO && ctx->heap_type == D3D12_HEAP_TYPE_CUSTOM) {
                void * mapped = nullptr;
                D3D12_RANGE read_range = { tensor_offset, tensor_offset + size };
                HRESULT hr = ctx->resource->Map(0, &read_range, &mapped);
                DX12_CHECK(hr, "Map(UMA get_tensor)");
                memcpy(data, (const uint8_t *)mapped + tensor_offset, size);
                D3D12_RANGE written = { 0, 0 };
                ctx->resource->Unmap(0, &written);
                return;
            }

            ctx->dev->init_xfer();
            ctx->dev->xfer_wait(); // wait for any previous transfer
            ctx->dev->xfer_ensure_staging(0, size);

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
                if (logits_diag_count < 30) {
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
                    // Spotlight: token 30 (the empty-piece token the model gets stuck on)
                    if (n > 30) {
                        fprintf(stderr, "  token30=%.4f  (top1 - token30 = %.4f)\n",
                                logits[30], top_vals[0] - logits[30]);
                    }
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
            if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16) {
                // Check source types too — our shaders handle F32/F16/BF16 via load_auto
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    if (op->src[s] && op->src[s]->type != GGML_TYPE_F32 &&
                        op->src[s]->type != GGML_TYPE_F16 &&
                        op->src[s]->type != GGML_TYPE_BF16 &&
                        op->src[s]->type != GGML_TYPE_I32) {
                        return false;
                    }
                }
                return true;
            }
            return false;

        case GGML_OP_SET_ROWS:
            // KV cache writes: src0 is F32, dst can be F16, BF16, or F32
            if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16) {
                return true;
            }
            return false;

        case GGML_OP_SET:
            // Recurrent state overlay: all tensors must be same type (F32 or I32)
            if (op->src[0] && op->src[1] &&
                op->src[0]->type == op->src[1]->type &&
                op->src[0]->type == op->type &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32)) {
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
                    if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16) {
                        // src0 must be F32, F16, or BF16
                        if (op->src[0] && op->src[0]->type != GGML_TYPE_F32 &&
                            op->src[0]->type != GGML_TYPE_F16 &&
                            op->src[0]->type != GGML_TYPE_BF16) return false;
                        // src1 (if present) must be F32, F16, or BF16
                        if (op->src[1] && op->src[1]->type != GGML_TYPE_F32 &&
                            op->src[1]->type != GGML_TYPE_F16 &&
                            op->src[1]->type != GGML_TYPE_BF16) return false;
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
                    if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16) {
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
                if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_BF16 &&
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
                if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_BF16 &&
                    t != GGML_TYPE_Q2_K && t != GGML_TYPE_Q3_K &&
                    t != GGML_TYPE_Q4_K && t != GGML_TYPE_Q5_K && t != GGML_TYPE_Q6_K &&
                    t != GGML_TYPE_Q4_0 && t != GGML_TYPE_Q4_1 &&
                    t != GGML_TYPE_Q5_0 && t != GGML_TYPE_Q5_1 &&
                    t != GGML_TYPE_Q8_0 && t != GGML_TYPE_Q8_1) return false;
            }
            return true;

        case GGML_OP_FLASH_ATTN_EXT:
            // Honor the runtime --flash-attn / -fa selection. The non-FA
            // path (separate QK matmul + softmax + attn*V) is currently
            // faster for prompt processing on Xbox/RDNA2 due to better
            // dispatch parallelism, so callers may prefer to leave FA off
            // -- minslm-cli's default is OFF on the GDKX build for that
            // reason. We always report supported here when the dtypes match
            // and let the runtime / cli decide.
            if (op->type == GGML_TYPE_F32 && op->src[1] &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_BF16)) {
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

// Map ggml type to shader esize: 2=F16, 3=BF16 (sentinel), 4=F32
// BF16 uses 3 as dispatch sentinel so shaders can distinguish from F16.
static uint32_t dx12_esize(ggml_type t) {
    if (t == GGML_TYPE_BF16) return 3;
    return (uint32_t)ggml_type_size(t);
}

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
        p.src0_esize  = dx12_esize(src0->type);
    }
    if (src1) {
        p.ne10 = (uint32_t)src1->ne[0]; p.ne11 = (uint32_t)src1->ne[1];
        p.ne12 = (uint32_t)src1->ne[2]; p.ne13 = (uint32_t)src1->ne[3];
        p.nb10 = (uint32_t)src1->nb[0]; p.nb11 = (uint32_t)src1->nb[1];
        p.nb12 = (uint32_t)src1->nb[2]; p.nb13 = (uint32_t)src1->nb[3];
        uint64_t off = dx12_tensor_offset(src1);
        GGML_ASSERT(off <= UINT32_MAX && "src1 offset exceeds 4GB");
        p.src1_offset = (uint32_t)off;
        p.src1_esize  = dx12_esize(src1->type);
    }

    uint64_t dst_off = dx12_tensor_offset(tensor);
    GGML_ASSERT(dst_off <= UINT32_MAX && "dst offset exceeds 4GB");
    p.ne0 = (uint32_t)tensor->ne[0]; p.ne1 = (uint32_t)tensor->ne[1];
    p.ne2 = (uint32_t)tensor->ne[2]; p.ne3 = (uint32_t)tensor->ne[3];
    p.nb0 = (uint32_t)tensor->nb[0]; p.nb1 = (uint32_t)tensor->nb[1];
    p.nb2 = (uint32_t)tensor->nb[2]; p.nb3 = (uint32_t)tensor->nb[3];
    p.dst_offset = (uint32_t)dst_off;
    p.dst_esize  = dx12_esize(tensor->type);

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

// ---------------------------------------------------------------------------
// Q5_K matvec unit test: feed known data through the shader and verify output
// Triggered by GGML_DX12_TEST_MATVEC=1 env var (runs once at first graph_compute)
// ---------------------------------------------------------------------------
static void dx12_test_matvec_q5k(dx12_device * dev) {
    fprintf(stderr, "ggml-dx12: === Q5_K MATVEC UNIT TEST ===\n");

    // ---- Reference: dequantize_row_q5_K from ggml-quants.c ----
    auto get_scale_min_k4 = [](int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
        if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
        else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
    };

    // Create a synthetic Q5_K block (176 bytes) with known values
    const uint32_t QK_K = 256;
    const uint32_t Q5K_BSIZE = 176;
    uint8_t block[Q5K_BSIZE] = {};

    // d = 0.01, dmin = 0.005 (as F16)
    uint16_t d_f16 = 0x211E;   // ~0.01 in F16
    uint16_t dmin_f16 = 0x1D0E; // ~0.005 in F16
    memcpy(&block[0], &d_f16, 2);
    memcpy(&block[2], &dmin_f16, 2);

    // Scales: all set to known values
    // For j<4: d=scales[j]&63, m=scales[j+4]&63
    // Simple: set scales[0..11] to sequential values
    for (int i = 0; i < 12; i++) block[4 + i] = (uint8_t)(10 + i);

    // qh (32 bytes): set bit patterns
    for (int i = 0; i < 32; i++) block[16 + i] = (uint8_t)(i & 0xFF);

    // qs (128 bytes): set to known nibble pairs
    for (int i = 0; i < 128; i++) block[48 + i] = (uint8_t)((i % 16) | (((i + 3) % 16) << 4));

    // Create activation vector: simple ramp
    float activations[QK_K];
    for (uint32_t i = 0; i < QK_K; i++) activations[i] = 0.01f * (float)(i % 32) - 0.15f;

    // ---- CPU reference dot product ----
    float ref_dequant[QK_K];
    {
        float dall, dmin_val;
        { uint16_t h; memcpy(&h, &block[0], 2); uint32_t s=(uint32_t)(h>>15)<<31, e=(h>>10)&0x1F, m_=h&0x3FF;
          if(e==0){if(m_==0){uint32_t r=s;memcpy(&dall,&r,4);dall=dall;}else{e=1;while(!(m_&0x400)){m_<<=1;e--;}m_&=0x3FF;uint32_t r=s|((e+112)<<23)|(m_<<13);memcpy(&dall,&r,4);}}
          else if(e==31){uint32_t r=s|0x7F800000|(m_<<13);memcpy(&dall,&r,4);}
          else{uint32_t r=s|((e+112)<<23)|(m_<<13);memcpy(&dall,&r,4);}}
        { uint16_t h; memcpy(&h, &block[2], 2); uint32_t s=(uint32_t)(h>>15)<<31, e=(h>>10)&0x1F, m_=h&0x3FF;
          if(e==0){if(m_==0){uint32_t r=s;memcpy(&dmin_val,&r,4);dmin_val=dmin_val;}else{e=1;while(!(m_&0x400)){m_<<=1;e--;}m_&=0x3FF;uint32_t r=s|((e+112)<<23)|(m_<<13);memcpy(&dmin_val,&r,4);}}
          else if(e==31){uint32_t r=s|0x7F800000|(m_<<13);memcpy(&dmin_val,&r,4);}
          else{uint32_t r=s|((e+112)<<23)|(m_<<13);memcpy(&dmin_val,&r,4);}}

        const uint8_t * ql = &block[48];
        const uint8_t * qh = &block[16];
        const uint8_t * scales = &block[4];
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < (int)QK_K; j += 64) {
            uint8_t sc, mi;
            get_scale_min_k4(is + 0, scales, &sc, &mi);
            float d1 = dall * sc, m1 = dmin_val * mi;
            get_scale_min_k4(is + 1, scales, &sc, &mi);
            float d2 = dall * sc, m2 = dmin_val * mi;
            for (int l = 0; l < 32; ++l) ref_dequant[j + l]      = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) ref_dequant[j + 32 + l] = d2 * ((ql[l] >> 4)  + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
    float ref_dot = 0.0f;
    for (uint32_t i = 0; i < QK_K; i++) ref_dot += ref_dequant[i] * activations[i];
    fprintf(stderr, "  CPU reference dot product: %.8f\n", ref_dot);

    // ---- GPU test ----
    // Create buffers: src0 (1 Q5_K block), src1 (256 F32 activations), dst (1 F32 output)
    auto make_gpu_buf = [&](size_t size) -> ComPtr<ID3D12Resource> {
        D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = std::max(size, (size_t)256);
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> res;
        dev->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res));
        return res;
    };

    ComPtr<ID3D12Resource> src0_buf = make_gpu_buf(Q5K_BSIZE);
    ComPtr<ID3D12Resource> src1_buf = make_gpu_buf(QK_K * 4);
    ComPtr<ID3D12Resource> dst_buf  = make_gpu_buf(256);  // need at least 1 float output

    // Upload data via staging
    dev->init_xfer();
    auto upload = [&](ID3D12Resource * gpu_buf, const void * data, size_t size) {
        dev->xfer_wait();
        dev->xfer_ensure_staging(size, 0);
        void * mapped = nullptr;
        D3D12_RANGE rr = { 0, 0 };
        dev->xfer.upload_staging->Map(0, &rr, &mapped);
        memcpy(mapped, data, size);
        dev->xfer.upload_staging->Unmap(0, nullptr);
        HRESULT hr = dev->xfer.cmd_alloc->Reset();
        if (SUCCEEDED(hr)) hr = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
        if (SUCCEEDED(hr)) {
            dev->xfer.cmd_list->CopyBufferRegion(gpu_buf, 0, dev->xfer.upload_staging.Get(), 0, size);
            dev->xfer.cmd_list->Close();
            ID3D12CommandList * lists[] = { dev->xfer.cmd_list.Get() };
            dev->compute_queue->ExecuteCommandLists(1, lists);
            dev->xfer.fence_value++;
            dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
            dev->xfer_wait();
        }
    };

    upload(src0_buf.Get(), block, Q5K_BSIZE);
    upload(src1_buf.Get(), activations, QK_K * 4);

    // Zero the output
    float zero = 0.0f;
    upload(dst_buf.Get(), &zero, 4);

    // Get the Q5_K matvec pipeline
    dx12_pipeline_key key = {};
    key.op = GGML_OP_MUL_MAT;
    key.src0_type = GGML_TYPE_Q5_K;
    key.flags = 1;  // standard matvec
    dx12_pipeline * pl = dev->get_or_create_pipeline(key);

    // Also test flags=4 (tiled batch MUL_MAT) -- used when ne[1] > 1
    dx12_pipeline_key key4 = {};
    key4.op = GGML_OP_MUL_MAT;
    key4.src0_type = GGML_TYPE_Q5_K;
    key4.flags = 4;  // tiled batch
    dx12_pipeline * pl4 = dev->get_or_create_pipeline(key4);

    if (!pl || !pl->pso) {
        fprintf(stderr, "  FAILED: Q5_K matvec pipeline not available\n");
        return;
    }

    // Set up shader params for K=256, N=1 (1 block, 1 output row)
    dx12_shader_params params = {};
    params.ne00 = QK_K;  // K dimension (elements per row)
    params.ne01 = 1;     // N=1 (1 output row)
    params.ne02 = 1; params.ne03 = 1;
    params.nb00 = Q5K_BSIZE;  // stride = 1 block
    params.nb01 = Q5K_BSIZE;  // row stride
    params.nb02 = Q5K_BSIZE; params.nb03 = Q5K_BSIZE;
    params.ne10 = QK_K;  // activation length
    params.ne11 = 1; params.ne12 = 1; params.ne13 = 1;
    params.nb10 = 4;  // F32 stride
    params.nb11 = QK_K * 4; params.nb12 = QK_K * 4; params.nb13 = QK_K * 4;
    params.ne0 = 1;  // output: 1 element
    params.ne1 = 1; params.ne2 = 1; params.ne3 = 1;
    params.nb0 = 4; params.nb1 = 4; params.nb2 = 4; params.nb3 = 4;
    params.src0_offset = 0;
    params.src1_offset = 0;
    params.dst_offset = 0;
    params.src0_esize = Q5K_BSIZE;  // block size for quantized types
    params.src1_esize = 4;
    params.dst_esize = 4;

    // Dispatch
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    dev->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
    dev->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));

    cl->SetComputeRootSignature(dev->common_root_sig.Get());
    cl->SetPipelineState(pl->pso.Get());
    cl->SetComputeRoot32BitConstants(0, sizeof(params) / 4, &params, 0);
    cl->SetComputeRootShaderResourceView(1, src0_buf->GetGPUVirtualAddress());
    cl->SetComputeRootShaderResourceView(2, src1_buf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(3, dst_buf->GetGPUVirtualAddress());
    // Bind src0 as fallback for slots 4/5
    cl->SetComputeRootShaderResourceView(4, src0_buf->GetGPUVirtualAddress());
    cl->SetComputeRootShaderResourceView(5, src0_buf->GetGPUVirtualAddress());

    // Dispatch 1 group (1 output row)
    cl->Dispatch(1, 1, 1);
    cl->Close();

    ID3D12CommandList * lists[] = { cl.Get() };
    dev->compute_queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> f;
    dev->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f));
    dev->compute_queue->Signal(f.Get(), 1);
    HANDLE ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    f->SetEventOnCompletion(1, ev);
    WaitForSingleObject(ev, 5000);
    CloseHandle(ev);

    // Readback result
    dev->xfer_ensure_staging(0, 256);
    dev->xfer_wait();
    HRESULT hr = dev->xfer.cmd_alloc->Reset();
    if (SUCCEEDED(hr)) hr = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
    if (SUCCEEDED(hr)) {
        dev->xfer.cmd_list->CopyBufferRegion(dev->xfer.readback_staging.Get(), 0, dst_buf.Get(), 0, 4);
        dev->xfer.cmd_list->Close();
        ID3D12CommandList * rl[] = { dev->xfer.cmd_list.Get() };
        dev->compute_queue->ExecuteCommandLists(1, rl);
        dev->xfer.fence_value++;
        dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
        dev->xfer_wait();
    }

    void * mapped = nullptr;
    D3D12_RANGE rr = { 0, 4 };
    hr = dev->xfer.readback_staging->Map(0, &rr, &mapped);
    float gpu_result = 0.0f;
    if (SUCCEEDED(hr)) {
        gpu_result = *(float *)mapped;
        D3D12_RANGE wr = { 0, 0 };
        dev->xfer.readback_staging->Unmap(0, &wr);
    }

    fprintf(stderr, "  GPU shader result:         %.8f\n", gpu_result);
    fprintf(stderr, "  Difference:                %.8e\n", (double)(gpu_result - ref_dot));
    float rel_err = (ref_dot != 0.0f) ? fabsf((gpu_result - ref_dot) / ref_dot) : fabsf(gpu_result);
    fprintf(stderr, "  Relative error:            %.6f%%\n", rel_err * 100.0f);
    fprintf(stderr, "  RESULT: %s\n", rel_err < 0.01f ? "PASS" : "*** FAIL ***");

    // ---- Test 2: tiled batch MUL_MAT (flags=4) ----
    if (pl4 && pl4->pso) {
        fprintf(stderr, "\n  --- Tiled batch Q5_K (flags=4) ---\n");

        // Zero the output
        upload(dst_buf.Get(), &zero, 4);

        // Tiled shader expects different params layout
        dx12_shader_params p4 = {};
        p4.ne00 = QK_K;  // K
        p4.ne01 = 1;     // N (output rows) -- but the weight matrix is [K, N]
        p4.ne02 = 1; p4.ne03 = 1;
        p4.nb00 = Q5K_BSIZE;  // block size
        p4.nb01 = Q5K_BSIZE;  // row stride in weight buffer
        p4.nb02 = Q5K_BSIZE; p4.nb03 = Q5K_BSIZE;
        p4.ne10 = QK_K;  // activation K
        p4.ne11 = 1;     // M (number of input vectors)
        p4.ne12 = 1; p4.ne13 = 1;
        p4.nb10 = 4;  // F32
        p4.nb11 = QK_K * 4;
        p4.nb12 = QK_K * 4; p4.nb13 = QK_K * 4;
        p4.ne0 = 1;  // output N
        p4.ne1 = 1;  // output M
        p4.ne2 = 1; p4.ne3 = 1;
        p4.nb0 = 4; p4.nb1 = 4; p4.nb2 = 4; p4.nb3 = 4;
        p4.src0_offset = 0;
        p4.src1_offset = 0;
        p4.dst_offset = 0;
        p4.src0_esize = Q5K_BSIZE;
        p4.src1_esize = 4;
        p4.dst_esize = 4;

        ComPtr<ID3D12CommandAllocator> alloc4;
        ComPtr<ID3D12GraphicsCommandList> cl4;
        dev->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc4));
        dev->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc4.Get(), nullptr, IID_PPV_ARGS(&cl4));

        cl4->SetComputeRootSignature(dev->common_root_sig.Get());
        cl4->SetPipelineState(pl4->pso.Get());
        cl4->SetComputeRoot32BitConstants(0, sizeof(p4) / 4, &p4, 0);
        cl4->SetComputeRootShaderResourceView(1, src0_buf->GetGPUVirtualAddress());
        cl4->SetComputeRootShaderResourceView(2, src1_buf->GetGPUVirtualAddress());
        cl4->SetComputeRootUnorderedAccessView(3, dst_buf->GetGPUVirtualAddress());
        cl4->SetComputeRootShaderResourceView(4, src0_buf->GetGPUVirtualAddress());
        cl4->SetComputeRootShaderResourceView(5, src0_buf->GetGPUVirtualAddress());

        // Tiled: groups_x = ceil(N/32) = 1, groups_y = ceil(M/32) = 1
        cl4->Dispatch(1, 1, 1);
        cl4->Close();

        ID3D12CommandList * l4[] = { cl4.Get() };
        dev->compute_queue->ExecuteCommandLists(1, l4);
        ComPtr<ID3D12Fence> f4;
        dev->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f4));
        dev->compute_queue->Signal(f4.Get(), 1);
        HANDLE ev4 = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        f4->SetEventOnCompletion(1, ev4);
        WaitForSingleObject(ev4, 5000);
        CloseHandle(ev4);

        // Readback
        dev->xfer_ensure_staging(0, 256);
        dev->xfer_wait();
        hr = dev->xfer.cmd_alloc->Reset();
        if (SUCCEEDED(hr)) hr = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
        if (SUCCEEDED(hr)) {
            dev->xfer.cmd_list->CopyBufferRegion(dev->xfer.readback_staging.Get(), 0, dst_buf.Get(), 0, 4);
            dev->xfer.cmd_list->Close();
            ID3D12CommandList * rl4[] = { dev->xfer.cmd_list.Get() };
            dev->compute_queue->ExecuteCommandLists(1, rl4);
            dev->xfer.fence_value++;
            dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
            dev->xfer_wait();
        }
        mapped = nullptr;
        hr = dev->xfer.readback_staging->Map(0, &rr, &mapped);
        float gpu4 = 0.0f;
        if (SUCCEEDED(hr)) {
            gpu4 = *(float *)mapped;
            D3D12_RANGE wr4 = { 0, 0 };
            dev->xfer.readback_staging->Unmap(0, &wr4);
        }
        fprintf(stderr, "  GPU tiled result:          %.8f\n", gpu4);
        fprintf(stderr, "  CPU reference:             %.8f\n", ref_dot);
        float re4 = (ref_dot != 0.0f) ? fabsf((gpu4 - ref_dot) / ref_dot) : fabsf(gpu4);
        fprintf(stderr, "  Relative error:            %.6f%%\n", re4 * 100.0f);
        fprintf(stderr, "  RESULT: %s\n", re4 < 0.01f ? "PASS" : "*** FAIL ***");
    }

    // ---- Test 3: matvec with non-zero offsets (mimics real dispatch) ----
    {
        fprintf(stderr, "\n  --- Matvec with offset (mimics real dispatch) ---\n");

        // Create larger buffers and place data at non-zero offsets
        size_t weight_offset = 80813568;  // same offset as real blk.0.attn_qkv.weight
        size_t act_offset = 14692352;     // same offset as real attn_norm-0

        ComPtr<ID3D12Resource> big_src0 = make_gpu_buf(weight_offset + Q5K_BSIZE);
        ComPtr<ID3D12Resource> big_src1 = make_gpu_buf(act_offset + QK_K * 4);
        ComPtr<ID3D12Resource> big_dst  = make_gpu_buf(1024);

        // Upload data at the real offsets
        dev->xfer_wait();
        dev->xfer_ensure_staging(std::max(Q5K_BSIZE, QK_K * 4), 0);

        // Upload weight block at weight_offset
        {
            void * m = nullptr;
            D3D12_RANGE r0 = { 0, 0 };
            dev->xfer.upload_staging->Map(0, &r0, &m);
            memcpy(m, block, Q5K_BSIZE);
            dev->xfer.upload_staging->Unmap(0, nullptr);
            HRESULT h = dev->xfer.cmd_alloc->Reset();
            if (SUCCEEDED(h)) h = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
            if (SUCCEEDED(h)) {
                dev->xfer.cmd_list->CopyBufferRegion(big_src0.Get(), weight_offset,
                    dev->xfer.upload_staging.Get(), 0, Q5K_BSIZE);
                dev->xfer.cmd_list->Close();
                ID3D12CommandList * ll[] = { dev->xfer.cmd_list.Get() };
                dev->compute_queue->ExecuteCommandLists(1, ll);
                dev->xfer.fence_value++;
                dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                dev->xfer_wait();
            }
        }
        // Upload activations at act_offset
        {
            void * m = nullptr;
            D3D12_RANGE r0 = { 0, 0 };
            dev->xfer.upload_staging->Map(0, &r0, &m);
            memcpy(m, activations, QK_K * 4);
            dev->xfer.upload_staging->Unmap(0, nullptr);
            HRESULT h = dev->xfer.cmd_alloc->Reset();
            if (SUCCEEDED(h)) h = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
            if (SUCCEEDED(h)) {
                dev->xfer.cmd_list->CopyBufferRegion(big_src1.Get(), act_offset,
                    dev->xfer.upload_staging.Get(), 0, QK_K * 4);
                dev->xfer.cmd_list->Close();
                ID3D12CommandList * ll[] = { dev->xfer.cmd_list.Get() };
                dev->compute_queue->ExecuteCommandLists(1, ll);
                dev->xfer.fence_value++;
                dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                dev->xfer_wait();
            }
        }

        // Set up params with real offsets
        dx12_shader_params p3 = {};
        p3.ne00 = QK_K; p3.ne01 = 1; p3.ne02 = 1; p3.ne03 = 1;
        p3.nb00 = Q5K_BSIZE; p3.nb01 = Q5K_BSIZE;
        p3.nb02 = Q5K_BSIZE; p3.nb03 = Q5K_BSIZE;
        p3.ne10 = QK_K; p3.ne11 = 1; p3.ne12 = 1; p3.ne13 = 1;
        p3.nb10 = 4; p3.nb11 = QK_K * 4;
        p3.nb12 = QK_K * 4; p3.nb13 = QK_K * 4;
        p3.ne0 = 1; p3.ne1 = 1; p3.ne2 = 1; p3.ne3 = 1;
        p3.nb0 = 4; p3.nb1 = 4; p3.nb2 = 4; p3.nb3 = 4;
        p3.src0_offset = (uint32_t)weight_offset;
        p3.src1_offset = (uint32_t)act_offset;
        p3.dst_offset = 0;
        p3.src0_esize = Q5K_BSIZE;
        p3.src1_esize = 4;
        p3.dst_esize = 4;

        ComPtr<ID3D12CommandAllocator> a3;
        ComPtr<ID3D12GraphicsCommandList> c3;
        dev->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&a3));
        dev->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, a3.Get(), nullptr, IID_PPV_ARGS(&c3));

        c3->SetComputeRootSignature(dev->common_root_sig.Get());
        c3->SetPipelineState(pl->pso.Get());
        c3->SetComputeRoot32BitConstants(0, sizeof(p3) / 4, &p3, 0);
        c3->SetComputeRootShaderResourceView(1, big_src0->GetGPUVirtualAddress());
        c3->SetComputeRootShaderResourceView(2, big_src1->GetGPUVirtualAddress());
        c3->SetComputeRootUnorderedAccessView(3, big_dst->GetGPUVirtualAddress());
        c3->SetComputeRootShaderResourceView(4, big_src0->GetGPUVirtualAddress());
        c3->SetComputeRootShaderResourceView(5, big_src0->GetGPUVirtualAddress());
        c3->Dispatch(1, 1, 1);
        c3->Close();

        ID3D12CommandList * l3[] = { c3.Get() };
        dev->compute_queue->ExecuteCommandLists(1, l3);
        ComPtr<ID3D12Fence> f3;
        dev->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f3));
        dev->compute_queue->Signal(f3.Get(), 1);
        HANDLE ev3 = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        f3->SetEventOnCompletion(1, ev3);
        WaitForSingleObject(ev3, 5000);
        CloseHandle(ev3);

        // Readback
        dev->xfer_ensure_staging(0, 256);
        dev->xfer_wait();
        hr = dev->xfer.cmd_alloc->Reset();
        if (SUCCEEDED(hr)) hr = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
        if (SUCCEEDED(hr)) {
            dev->xfer.cmd_list->CopyBufferRegion(dev->xfer.readback_staging.Get(), 0, big_dst.Get(), 0, 4);
            dev->xfer.cmd_list->Close();
            ID3D12CommandList * rl3[] = { dev->xfer.cmd_list.Get() };
            dev->compute_queue->ExecuteCommandLists(1, rl3);
            dev->xfer.fence_value++;
            dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
            dev->xfer_wait();
        }
        mapped = nullptr;
        hr = dev->xfer.readback_staging->Map(0, &rr, &mapped);
        float gpu3 = 0.0f;
        if (SUCCEEDED(hr)) {
            gpu3 = *(float *)mapped;
            D3D12_RANGE wr3 = { 0, 0 };
            dev->xfer.readback_staging->Unmap(0, &wr3);
        }
        fprintf(stderr, "  GPU result with offset:    %.8f\n", gpu3);
        fprintf(stderr, "  CPU reference:             %.8f\n", ref_dot);
        float re3 = (ref_dot != 0.0f) ? fabsf((gpu3 - ref_dot) / ref_dot) : fabsf(gpu3);
        fprintf(stderr, "  Relative error:            %.6f%%\n", re3 * 100.0f);
        fprintf(stderr, "  RESULT: %s\n", re3 < 0.01f ? "PASS" : "*** FAIL ***");
    }

    // ---- Test 4: K=3072 (12 blocks, matching real Phi-3 K dimension) ----
    {
        fprintf(stderr, "\n  --- K=3072 matvec (12 blocks, real K) ---\n");

        const uint32_t K4 = 3072;
        const uint32_t NBLOCKS4 = K4 / QK_K;  // 12
        const size_t ROW_BYTES = NBLOCKS4 * Q5K_BSIZE;  // 12 * 176 = 2112

        // Create 12 blocks of weight data
        std::vector<uint8_t> weight_data(ROW_BYTES, 0);
        for (uint32_t b = 0; b < NBLOCKS4; b++) {
            uint8_t * blk = &weight_data[b * Q5K_BSIZE];
            memcpy(blk, &d_f16, 2);
            memcpy(blk + 2, &dmin_f16, 2);
            for (int i = 0; i < 12; i++) blk[4 + i] = (uint8_t)(10 + i + b);
            for (int i = 0; i < 32; i++) blk[16 + i] = (uint8_t)((i + b * 7) & 0xFF);
            for (int i = 0; i < 128; i++) blk[48 + i] = (uint8_t)(((i + b) % 16) | ((((i + b) + 3) % 16) << 4));
        }

        // Create K4 activations
        std::vector<float> act4(K4);
        for (uint32_t i = 0; i < K4; i++) act4[i] = 0.01f * (float)(i % 32) - 0.15f;

        // CPU reference: dequant all 12 blocks and dot product
        float ref4 = 0.0f;
        for (uint32_t b = 0; b < NBLOCKS4; b++) {
            const uint8_t * blk = &weight_data[b * Q5K_BSIZE];
            float dall4, dmin4;
            // Simplified: use ggml's fp16 conversion
            dall4 = GGML_FP16_TO_FP32(*(ggml_fp16_t *)(blk));
            dmin4 = GGML_FP16_TO_FP32(*(ggml_fp16_t *)(blk + 2));

            const uint8_t * ql4 = blk + 48;
            const uint8_t * qh4 = blk + 16;
            const uint8_t * sc4 = blk + 4;
            int is4 = 0;
            uint8_t u14 = 1, u24 = 2;
            for (int j = 0; j < (int)QK_K; j += 64) {
                uint8_t sd, sm;
                get_scale_min_k4(is4 + 0, sc4, &sd, &sm);
                float dd1 = dall4 * sd, mm1 = dmin4 * sm;
                get_scale_min_k4(is4 + 1, sc4, &sd, &sm);
                float dd2 = dall4 * sd, mm2 = dmin4 * sm;
                for (int l = 0; l < 32; ++l) {
                    float w = dd1 * ((ql4[l] & 0xF) + (qh4[l] & u14 ? 16 : 0)) - mm1;
                    ref4 += w * act4[b * QK_K + j + l];
                }
                for (int l = 0; l < 32; ++l) {
                    float w = dd2 * ((ql4[l] >> 4) + (qh4[l] & u24 ? 16 : 0)) - mm2;
                    ref4 += w * act4[b * QK_K + j + 32 + l];
                }
                ql4 += 32; is4 += 2; u14 <<= 2; u24 <<= 2;
            }
        }

        // GPU test
        ComPtr<ID3D12Resource> s0_4 = make_gpu_buf(ROW_BYTES);
        ComPtr<ID3D12Resource> s1_4 = make_gpu_buf(K4 * 4);
        ComPtr<ID3D12Resource> d_4  = make_gpu_buf(256);
        upload(s0_4.Get(), weight_data.data(), ROW_BYTES);
        upload(s1_4.Get(), act4.data(), K4 * 4);
        float z4 = 0.0f;
        upload(d_4.Get(), &z4, 4);

        dx12_shader_params p44 = {};
        p44.ne00 = K4; p44.ne01 = 1; p44.ne02 = 1; p44.ne03 = 1;
        p44.nb00 = Q5K_BSIZE; p44.nb01 = (uint32_t)ROW_BYTES;
        p44.nb02 = (uint32_t)ROW_BYTES; p44.nb03 = (uint32_t)ROW_BYTES;
        p44.ne10 = K4; p44.ne11 = 1; p44.ne12 = 1; p44.ne13 = 1;
        p44.nb10 = 4; p44.nb11 = K4 * 4; p44.nb12 = K4 * 4; p44.nb13 = K4 * 4;
        p44.ne0 = 1; p44.ne1 = 1; p44.ne2 = 1; p44.ne3 = 1;
        p44.nb0 = 4; p44.nb1 = 4; p44.nb2 = 4; p44.nb3 = 4;
        p44.src0_offset = 0; p44.src1_offset = 0; p44.dst_offset = 0;
        p44.src0_esize = Q5K_BSIZE; p44.src1_esize = 4; p44.dst_esize = 4;

        ComPtr<ID3D12CommandAllocator> a44;
        ComPtr<ID3D12GraphicsCommandList> c44;
        dev->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&a44));
        dev->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, a44.Get(), nullptr, IID_PPV_ARGS(&c44));
        c44->SetComputeRootSignature(dev->common_root_sig.Get());
        c44->SetPipelineState(pl->pso.Get());
        c44->SetComputeRoot32BitConstants(0, sizeof(p44) / 4, &p44, 0);
        c44->SetComputeRootShaderResourceView(1, s0_4->GetGPUVirtualAddress());
        c44->SetComputeRootShaderResourceView(2, s1_4->GetGPUVirtualAddress());
        c44->SetComputeRootUnorderedAccessView(3, d_4->GetGPUVirtualAddress());
        c44->SetComputeRootShaderResourceView(4, s0_4->GetGPUVirtualAddress());
        c44->SetComputeRootShaderResourceView(5, s0_4->GetGPUVirtualAddress());
        c44->Dispatch(1, 1, 1);
        c44->Close();
        ID3D12CommandList * l44[] = { c44.Get() };
        dev->compute_queue->ExecuteCommandLists(1, l44);
        ComPtr<ID3D12Fence> f44;
        dev->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f44));
        dev->compute_queue->Signal(f44.Get(), 1);
        HANDLE ev44 = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        f44->SetEventOnCompletion(1, ev44);
        WaitForSingleObject(ev44, 5000);
        CloseHandle(ev44);

        dev->xfer_ensure_staging(0, 256);
        dev->xfer_wait();
        hr = dev->xfer.cmd_alloc->Reset();
        if (SUCCEEDED(hr)) hr = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
        if (SUCCEEDED(hr)) {
            dev->xfer.cmd_list->CopyBufferRegion(dev->xfer.readback_staging.Get(), 0, d_4.Get(), 0, 4);
            dev->xfer.cmd_list->Close();
            ID3D12CommandList * rl44[] = { dev->xfer.cmd_list.Get() };
            dev->compute_queue->ExecuteCommandLists(1, rl44);
            dev->xfer.fence_value++;
            dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
            dev->xfer_wait();
        }
        mapped = nullptr;
        hr = dev->xfer.readback_staging->Map(0, &rr, &mapped);
        float gpu44 = 0.0f;
        if (SUCCEEDED(hr)) {
            gpu44 = *(float *)mapped;
            D3D12_RANGE wr44 = { 0, 0 };
            dev->xfer.readback_staging->Unmap(0, &wr44);
        }
        fprintf(stderr, "  GPU K=3072 result:         %.8f\n", gpu44);
        fprintf(stderr, "  CPU reference:             %.8f\n", ref4);
        float re44 = (ref4 != 0.0f) ? fabsf((gpu44 - ref4) / ref4) : fabsf(gpu44);
        fprintf(stderr, "  Relative error:            %.6f%%\n", re44 * 100.0f);
        fprintf(stderr, "  RESULT: %s\n", re44 < 0.01f ? "PASS" : "*** FAIL ***");

        // Per-block isolation: dispatch each block separately and sum
        float block_sum = 0.0f;
        fprintf(stderr, "\n  Per-block isolation (12 separate dispatches):\n");
        for (uint32_t b = 0; b < NBLOCKS4; b++) {
            float zb = 0.0f;
            upload(d_4.Get(), &zb, 4);

            // Create a view: offset into the weight buffer, activation offset
            dx12_shader_params pb = {};
            pb.ne00 = QK_K;  // K=256 per block
            pb.ne01 = 1; pb.ne02 = 1; pb.ne03 = 1;
            pb.nb00 = Q5K_BSIZE; pb.nb01 = Q5K_BSIZE;
            pb.nb02 = Q5K_BSIZE; pb.nb03 = Q5K_BSIZE;
            pb.ne10 = QK_K; pb.ne11 = 1; pb.ne12 = 1; pb.ne13 = 1;
            pb.nb10 = 4; pb.nb11 = QK_K * 4;
            pb.nb12 = QK_K * 4; pb.nb13 = QK_K * 4;
            pb.ne0 = 1; pb.ne1 = 1; pb.ne2 = 1; pb.ne3 = 1;
            pb.nb0 = 4; pb.nb1 = 4; pb.nb2 = 4; pb.nb3 = 4;
            pb.src0_offset = b * Q5K_BSIZE;      // offset to this block
            pb.src1_offset = b * QK_K * 4;         // offset to this block's activations
            pb.dst_offset = 0;
            pb.src0_esize = Q5K_BSIZE; pb.src1_esize = 4; pb.dst_esize = 4;

            ComPtr<ID3D12CommandAllocator> ab;
            ComPtr<ID3D12GraphicsCommandList> cb;
            dev->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&ab));
            dev->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, ab.Get(), nullptr, IID_PPV_ARGS(&cb));
            cb->SetComputeRootSignature(dev->common_root_sig.Get());
            cb->SetPipelineState(pl->pso.Get());
            cb->SetComputeRoot32BitConstants(0, sizeof(pb) / 4, &pb, 0);
            cb->SetComputeRootShaderResourceView(1, s0_4->GetGPUVirtualAddress());
            cb->SetComputeRootShaderResourceView(2, s1_4->GetGPUVirtualAddress());
            cb->SetComputeRootUnorderedAccessView(3, d_4->GetGPUVirtualAddress());
            cb->SetComputeRootShaderResourceView(4, s0_4->GetGPUVirtualAddress());
            cb->SetComputeRootShaderResourceView(5, s0_4->GetGPUVirtualAddress());
            cb->Dispatch(1, 1, 1);
            cb->Close();
            ID3D12CommandList * lb[] = { cb.Get() };
            dev->compute_queue->ExecuteCommandLists(1, lb);
            ComPtr<ID3D12Fence> fb;
            dev->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fb));
            dev->compute_queue->Signal(fb.Get(), 1);
            HANDLE evb = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            fb->SetEventOnCompletion(1, evb);
            WaitForSingleObject(evb, 5000);
            CloseHandle(evb);

            dev->xfer_ensure_staging(0, 256);
            dev->xfer_wait();
            hr = dev->xfer.cmd_alloc->Reset();
            if (SUCCEEDED(hr)) hr = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
            if (SUCCEEDED(hr)) {
                dev->xfer.cmd_list->CopyBufferRegion(dev->xfer.readback_staging.Get(), 0, d_4.Get(), 0, 4);
                dev->xfer.cmd_list->Close();
                ID3D12CommandList * rlb[] = { dev->xfer.cmd_list.Get() };
                dev->compute_queue->ExecuteCommandLists(1, rlb);
                dev->xfer.fence_value++;
                dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                dev->xfer_wait();
            }
            mapped = nullptr;
            hr = dev->xfer.readback_staging->Map(0, &rr, &mapped);
            float gpub = 0.0f;
            if (SUCCEEDED(hr)) {
                gpub = *(float *)mapped;
                D3D12_RANGE wrb = { 0, 0 };
                dev->xfer.readback_staging->Unmap(0, &wrb);
            }
            block_sum += gpub;
            fprintf(stderr, "    Block %2u: GPU=%.6f  sum_so_far=%.6f\n", b, gpub, block_sum);
        }
        fprintf(stderr, "  Sum of 12 separate dispatches: %.8f\n", block_sum);
        fprintf(stderr, "  Single dispatch (12 blocks):   %.8f\n", gpu44);
        fprintf(stderr, "  CPU reference:                 %.8f\n", ref4);
        float re_iso = (ref4 != 0.0f) ? fabsf((block_sum - ref4) / ref4) : fabsf(block_sum);
        fprintf(stderr, "  Isolated sum error:            %.6f%%\n", re_iso * 100.0f);
        fprintf(stderr, "  Isolated RESULT: %s\n", re_iso < 0.01f ? "PASS" : "*** FAIL ***");
    }

    fflush(stderr);
}

static ggml_status dx12_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * bctx = (dx12_backend_context *)backend->context;

    g_tls_device = bctx->dev->device.Get();

    // Wall-clock timing of the entire graph_compute call (GGML_DX12_GC_TIMING).
    // Useful for separating dx12-backend CPU cost from upstream scheduler /
    // sampling / buffer-IO cost when the host process shows one core pegged.
    // Prints a rolling avg every 50 calls. Implemented as a small RAII guard
    // so all return paths are covered uniformly.
    static const bool gc_timing = (getenv("GGML_DX12_GC_TIMING") != nullptr);
    struct gc_timer {
        bool   active;
#ifdef _WIN32
        LARGE_INTEGER t0;
        LARGE_INTEGER freq;
#else
        std::chrono::steady_clock::time_point t0;
#endif
        int n_nodes;
        gc_timer(bool on, int nn) : active(on), n_nodes(nn) {
            if (!active) return;
#ifdef _WIN32
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&t0);
#else
            t0 = std::chrono::steady_clock::now();
#endif
        }
        ~gc_timer() {
            if (!active) return;
            int64_t us;
#ifdef _WIN32
            LARGE_INTEGER t1; QueryPerformanceCounter(&t1);
            us = (int64_t)((double)(t1.QuadPart - t0.QuadPart) * 1e6 / (double)freq.QuadPart);
#else
            auto t1 = std::chrono::steady_clock::now();
            us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
#endif
            static int64_t n_calls = 0;
            static int64_t total_us = 0;
            static int64_t window_us = 0;
            n_calls++;
            total_us  += us;
            window_us += us;
            if ((n_calls % 50) == 0) {
                fprintf(stderr,
                    "[gc] calls=%lld last=%lld us nodes=%d  win50_avg=%.1f us  total=%.2f ms\n",
                    (long long)n_calls, (long long)us, n_nodes,
                    (double)window_us / 50.0, (double)total_us / 1000.0);
                fflush(stderr);
                window_us = 0;
            }
        }
    };
    gc_timer _gc_timer(gc_timing, cgraph->n_nodes);

    // Graph identity hash + cache decision (Stage 2 of GGML_DX12_GRAPH_CACHE).
    //
    // We hash a small but topology-defining slice of each node:
    //   op + type + ne[0..3] + nb[0..3] + dst data ptr
    //   + src[s]->{ne, data ptr} for each src
    //   + first 4 op_params DWORDs (captures flag-style ops: UNARY sub-op,
    //     ROPE n_dims/mode, SOFT_MAX scale, etc).
    //
    // The dx12 backend places tensors at fixed offsets inside a small set of
    // device buffers, so the (data) pointer for any given node is stable
    // across decode iterations as long as the runtime is reusing the cgraph.
    // The llama-context "graphs reused" counter validates this externally
    // (see `n_reused++` in llama-context.cpp's process_ubatch). With FA on,
    // KV cache size grows but the cgraph topology is reused, and our hash
    // captures the FA src ne[d] so the per-N_kv variant does get its own
    // cache entry.
    //
    // Eligibility (cache_hash != 0): n_nodes > 32, no flash-attn split-K
    // metadata that depends on N_kv (the cached CL would bake stale params).
    // Hit  -> ExecuteCommandLists(cached_cl), signal fence, ts_finalize, return.
    // Miss & eligible -> record into a fresh CL, also submit it for execution
    //                    this time around. Subsequent matching calls hit.
    // Miss & ineligible -> normal record-fresh path.

    static const bool cache_diag    = (getenv("GGML_DX12_GRAPH_CACHE_DIAG") != nullptr);
    static const bool cache_enabled = []() {
        const char * env = getenv("GGML_DX12_GRAPH_CACHE");
        bool on = !env || atoi(dx12_env_unquote(env)) != 0;  // default ON
        DX12_LOG_INFO("Graph cache: %s (set GGML_DX12_GRAPH_CACHE=0 to disable)\n",
                      on ? "enabled" : "disabled");
        return on;
    }();

    uint64_t cache_hash = 0;
    {
        uint64_t h = 1469598103934665603ull;  // FNV-1a 64-bit offset basis
        auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
        mix((uint64_t)cgraph->n_nodes);
        for (int ni = 0; ni < cgraph->n_nodes; ++ni) {
            const ggml_tensor * n = cgraph->nodes[ni];
            if (!n) continue;
            mix((uint64_t)n->op);
            mix((uint64_t)n->type);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                mix((uint64_t)n->ne[d]);
                mix((uint64_t)n->nb[d]);
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * sr = n->src[s];
                if (sr) {
                    mix((uint64_t)(uintptr_t)sr->data);
                    // Hash src ne too, so that any data-shape change (e.g.
                    // FA's K/V getting larger as KV cache grows) produces a
                    // different cache entry instead of replaying a stale CL.
                    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                        mix((uint64_t)sr->ne[d]);
                    }
                } else {
                    mix(0ull);
                }
            }
            mix((uint64_t)(uintptr_t)n->data);
            const uint32_t * op = (const uint32_t *)n->op_params;
            mix(((uint64_t)op[0] << 32) | op[1]);
            mix(((uint64_t)op[2] << 32) | op[3]);
        }
        cache_hash = h;
    }

    // ----- Cache HIT path (replay) ---------------------------------------
    if (cache_enabled) {
        auto cit = bctx->graph_cache.find(cache_hash);
        if (cit != bctx->graph_cache.end()) {
            // Make sure any pending live CL is submitted first so submission
            // order on the queue matches graph-execution order.
            if (bctx->cmd_list_open) {
                bctx->close_and_execute();
            }
            // Submit the cached CL.
            ID3D12CommandList * lists[] = { cit->second.cl.Get() };
            bctx->dev->compute_queue->ExecuteCommandLists(1, lists);
            bctx->fence_value++;
            HRESULT hr = bctx->dev->compute_queue->Signal(bctx->fence.Get(), bctx->fence_value);
            DX12_CHECK(hr, "Signal fence (cache replay)");

            if (cache_diag) {
                static int hit_total = 0;
                ++hit_total;
                if ((hit_total % 50) == 0 || hit_total <= 3) {
                    fprintf(stderr, "ggml-dx12: GRAPH CACHE replay hit=%d hash=0x%016llX nodes=%d disp=%d\n",
                            hit_total, (unsigned long long)cache_hash,
                            cit->second.n_nodes, cit->second.n_dispatches);
                    fflush(stderr);
                }
            }

            // GGML_DX12_PERF accounting still runs (but with no per-op
            // breakdown -- the cached CL doesn't emit our timestamp queries
            // for the recorded dispatches). Skip ts_finalize_graph here.
            return GGML_STATUS_SUCCESS;
        }
    }

    // Run auto-tuning on first graph compute
    if (!bctx->dev->tuning_done) {
        bctx->dev->run_autotune();
    }

    // Run auto-tuning on first graph compute
    if (!bctx->dev->tuning_done) {
        bctx->dev->run_autotune();
    }

    // Q5_K matvec unit test (GGML_DX12_TEST_MATVEC=1)
    {
        static bool test_done = false;
        if (!test_done && getenv("GGML_DX12_TEST_MATVEC")) {
            test_done = true;
            dx12_test_matvec_q5k(bctx->dev);
        }
    }

    bctx->ensure_cmd_list_open();

    // Profiling: profile only actual generation graphs (M=1 in MUL_MATs)
    static bool profiling = (getenv("DX12_PROFILE") != nullptr);
    static bool dx12_perf = (getenv("GGML_DX12_PERF") != nullptr);

    // GGML_DX12_PERF: try to use GPU-timestamp queries (zero-stall, accurate
    // per-Dispatch GPU time). Falls back to wall-clock+sync mode if heap
    // creation fails. ts_init() is idempotent; subsequent graphs reuse the
    // same heap.
    bool ts_active = dx12_perf && bctx->ts_init();

    // PERF deferred-readback: reserve a slot range for this graph, polling
    // the pending queue for completed prior graphs (no wait in steady state).
    if (ts_active) {
        bctx->ts_begin_graph(cgraph->n_nodes);
    }

    // ---- Cache MISS path: decide whether to record this graph ---------
    //
    // Eligibility: cache enabled, large enough graph to be worth caching,
    // and none of the per-dispatch instrumentation modes are active (those
    // need to interleave readbacks with dispatches and would race the
    // recording).
    //
    // GGML_DX12_GRAPH_CACHE_PROMPT env var (default OFF):
    //   Prompt-eval graphs are unique per prompt (different leaf input
    //   pointers -> different hash) so they're recorded once and never
    //   replayed. The recording overhead + loss of TDR-flush yields adds
    //   ~20% to prefill latency. Default is to skip recording prompt graphs;
    //   set GGML_DX12_GRAPH_CACHE_PROMPT=1 to record them too (useful when a
    //   workload reuses the same prompt verbatim, e.g. perplexity sweeps).
    static bool dx12_diag_for_cache = (getenv("GGML_DX12_DIAG") != nullptr);
    static bool cache_prompt = []() {
        const char * env = getenv("GGML_DX12_GRAPH_CACHE_PROMPT");
        bool on = env && atoi(dx12_env_unquote(env)) != 0;
        if (on) DX12_LOG_INFO("Graph cache: also caching prompt-eval graphs (GGML_DX12_GRAPH_CACHE_PROMPT=1)\n");
        return on;
    }();
    bool record_this_graph =
        cache_enabled &&
        cgraph->n_nodes > 32 &&
        !ts_active &&            // PERF mode emits per-dispatch EndQuery; cached CLs would replay them
        !profiling &&
        !dx12_diag_for_cache &&
        getenv("GGML_DX12_DUMP_DISPATCH") == nullptr;
    if (record_this_graph && !cache_prompt) {
        // Quick prompt detection (mirror of the is_prompt logic later in the
        // function): if any of the first 30 MUL_MATs has ne[1] > 1, this is
        // a prompt-eval graph. Skip recording it unless cache_prompt is set.
        for (int j = 0; j < std::min(cgraph->n_nodes, 30); ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (n && n->op == GGML_OP_MUL_MAT && n->ne[1] > 1) {
                record_this_graph = false;
                break;
            }
        }
    }
    // The chunked matvec path (huge lm_head with N > 32768) does mid-graph
    // close+wait+rebind cycles. We can't record those into a single CL.
    // Cheap pre-scan to disqualify graphs that hit it.
    if (record_this_graph) {
        for (int j = 0; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (n && n->op == GGML_OP_MUL_MAT && n->ne[1] == 1 && n->ne[0] > 32768) {
                record_this_graph = false;
                break;
            }
        }
    }

    // If recording, redirect bctx->cmd_list to a fresh dedicated allocator+CL
    // for the duration of this graph. The dispatch loop is unmodified -- it
    // just writes into bctx->cmd_list. We restore the live ring state at the
    // end. Mid-graph close_and_execute calls become no-ops while recording
    // (see cache_recording check in the periodic flush block below).
    ComPtr<ID3D12CommandAllocator>    saved_cmd_alloc_unused; // not used; we restore by restart
    ComPtr<ID3D12GraphicsCommandList> saved_cmd_list;
    bool saved_cmd_list_open = false;
    dx12_backend_context::dx12_graph_cache_entry recording_entry;
    if (record_this_graph) {
        // Make sure any pending live work is submitted first (so submission
        // order on the queue matches graph-execution order).
        if (bctx->cmd_list_open) {
            bctx->close_and_execute();
        }
        // Allocate fresh, dedicated objects for the cache entry.
        HRESULT hr = bctx->dev->device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&recording_entry.alloc));
        if (FAILED(hr)) {
            DX12_LOG_WARN("Graph cache: CreateCommandAllocator failed (0x%08lX) -- skipping record\n",
                          (unsigned long)hr);
            record_this_graph = false;
        } else {
            hr = bctx->dev->device->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_COMPUTE, recording_entry.alloc.Get(), nullptr,
                IID_PPV_ARGS(&recording_entry.cl));
            if (FAILED(hr)) {
                DX12_LOG_WARN("Graph cache: CreateCommandList failed (0x%08lX) -- skipping record\n",
                              (unsigned long)hr);
                record_this_graph = false;
                recording_entry.alloc.Reset();
            }
        }
        if (record_this_graph) {
            // Swap: dispatch loop writes into recording_entry.cl, not the
            // live ring. Save the previous live cmd_list (closed; the live
            // CL will be re-opened from the ring at end of graph).
            saved_cmd_list      = bctx->cmd_list;
            saved_cmd_list_open = bctx->cmd_list_open;
            bctx->cmd_list      = recording_entry.cl;
            bctx->cmd_list_open = true;
            bctx->cache_recording        = true;
            bctx->cache_recording_entry  = &recording_entry;
            bctx->reset_binding_cache();
        }
    }


    // Per-dispatch tensor dump: GGML_DX12_DUMP_DISPATCH=<dir> dumps each node's
    // output immediately after its dispatch completes (with full GPU sync).
    // This captures the true output before aliased memory gets overwritten.
    // GGML_DX12_DUMP_DISPATCH_GRAPH=N selects which graph_compute call to dump (default: 4).
    static const char * dump_dispatch_dir = getenv("GGML_DX12_DUMP_DISPATCH");
    static int dump_dispatch_graph_target = []() {
        const char * e = getenv("GGML_DX12_DUMP_DISPATCH_GRAPH");
        return (e && atoi(e) > 0) ? atoi(e) : 4;
    }();
    static int dump_dispatch_graph_count = 0;
    static bool dump_dispatch_done = false;
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

    // Per-dispatch dump activation
    bool dump_this_dispatch_graph = false;
    if (dump_dispatch_dir && !dump_dispatch_done) {
        dump_dispatch_graph_count++;
        if (dump_dispatch_graph_count == dump_dispatch_graph_target) {
            dump_this_dispatch_graph = true;
            dump_dispatch_done = true;
            fprintf(stderr, "ggml-dx12: DUMP_DISPATCH: dumping graph #%d (%d nodes) to '%s'\n",
                    dump_dispatch_graph_count, cgraph->n_nodes, dump_dispatch_dir);
            fflush(stderr);
        }
    }

    int dispatch_weight = 0;
    // AMD RDNA 3 workaround: flush threshold for prompt phase.
    // Lower values = more frequent CL splits = more correct but slower.
    // Configurable via GGML_DX12_FLUSH_INTERVAL env var (default: 16).
    static int TDR_FLUSH_THRESHOLD = 0;
    if (TDR_FLUSH_THRESHOLD == 0) {
        const char * env = getenv("GGML_DX12_FLUSH_INTERVAL");
        TDR_FLUSH_THRESHOLD = env ? atoi(dx12_env_unquote(env)) : 16;
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

    // Prompt diagnostics gated under GGML_DX12_DIAG.
    // Also trace the first decode graph so we can see SET_ROWS / FA dispatch
    // shapes when output is broken (n_tokens=1 path).
    static int prompt_eval_count = 0;
    static int decode_eval_count = 0;
    bool trace_prompt = false;
    if (is_prompt && dx12_diag) {
        trace_prompt = true;
        prompt_eval_count++;
        fprintf(stderr, "ggml-dx12: PROMPT GRAPH #%d: n_nodes=%d, cl_open=%d\n",
                prompt_eval_count, cgraph->n_nodes, (int)bctx->cmd_list_open);
        fflush(stderr);
    } else if (!is_prompt && dx12_diag) {
        decode_eval_count++;
        // Default: trace decode #1. Override via GGML_DX12_TRACE_DECODE=N to
        // trace the Nth decode iteration instead (e.g. N=15 catches the
        // KV-corruption point where logits flip from sane to all-NaN).
        static int trace_decode_target = []() {
            const char * env = getenv("GGML_DX12_TRACE_DECODE");
            int v = env ? atoi(dx12_env_unquote(env)) : 1;
            if (v <= 0) v = 1;
            return v;
        }();
        if (decode_eval_count == trace_decode_target) {
            trace_prompt = true;  // reuse the same trace switch
            fprintf(stderr, "ggml-dx12: DECODE GRAPH #%d (TRACED): n_nodes=%d, cl_open=%d\n",
                    decode_eval_count, cgraph->n_nodes, (int)bctx->cmd_list_open);
            fflush(stderr);

            // Probe the first F32 leaf input tensor (inp_embd) before any
            // dispatches run -- this tells us if set_tensor delivered correct
            // data into the decode input, or if the embedding itself is NaN.
            // Also probe inp_pos (I32) for ROPE position correctness.
            struct ggml_tensor * probe_embd = nullptr;
            struct ggml_tensor * probe_pos  = nullptr;
            for (int ni = 0; ni < cgraph->n_nodes && (!probe_embd || !probe_pos); ni++) {
                auto * nd = cgraph->nodes[ni];
                for (int si = 0; si < GGML_MAX_SRC; si++) {
                    auto * sr = nd->src[si];
                    if (!sr || !sr->buffer) continue;
                    if (sr->op != GGML_OP_NONE) continue;  // not a leaf
                    if (sr->view_src) continue;             // not a base tensor
                    if (!probe_embd && sr->type == GGML_TYPE_F32 && sr->ne[0] >= 256 && sr->ne[0] <= 8192) {
                        probe_embd = sr;
                    }
                    if (!probe_pos && sr->type == GGML_TYPE_I32 && sr->ne[0] >= 1 && sr->ne[0] <= 16) {
                        probe_pos = sr;
                    }
                }
            }
            auto * dev = bctx->dev;
            dev->init_xfer();
            dev->xfer_wait();  // ensure any pending set_tensor xfer landed
            // Probe inp_embd (F32)
            if (probe_embd) {
                size_t bytes = (size_t)ggml_nelements(probe_embd) * sizeof(float);
                if (bytes > 0 && bytes <= 64*1024) {
                    uint32_t off = (uint32_t)dx12_tensor_offset(probe_embd);
                    auto * sbuf = (dx12_buffer_context *)probe_embd->buffer->context;
                    dev->xfer_ensure_staging(0, bytes);
                    HRESULT hr = dev->xfer.cmd_alloc->Reset();
                    if (SUCCEEDED(hr)) hr = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
                    if (SUCCEEDED(hr)) {
                        dev->xfer.cmd_list->CopyBufferRegion(
                            dev->xfer.readback_staging.Get(), 0, sbuf->resource.Get(), off, bytes);
                        dev->xfer.cmd_list->Close();
                        ID3D12CommandList * lists[] = { dev->xfer.cmd_list.Get() };
                        dev->compute_queue->ExecuteCommandLists(1, lists);
                        dev->xfer.fence_value++;
                        dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                        dev->xfer_wait();
                        void * mp = nullptr;
                        D3D12_RANGE rr = { 0, bytes };
                        if (SUCCEEDED(dev->xfer.readback_staging->Map(0, &rr, &mp))) {
                            float * f = (float *)mp;
                            int n = (int)ggml_nelements(probe_embd);
                            int nans = 0;
                            double sum = 0, sum_sq = 0;
                            for (int j = 0; j < n; j++) {
                                if (isnan(f[j])) { nans++; continue; }
                                sum += f[j];
                                sum_sq += (double)f[j] * f[j];
                            }
                            double rms = sqrt(sum_sq / n);
                            fprintf(stderr, "ggml-dx12: INPUT EMBD ne=(%lld,%lld) off=%u: rms=%.6f mean=%.6f nans=%d/%d  first8:",
                                    (long long)probe_embd->ne[0], (long long)probe_embd->ne[1],
                                    off, rms, sum/n, nans, n);
                            for (int j = 0; j < 8 && j < n; j++) fprintf(stderr, " %.4f", f[j]);
                            fprintf(stderr, "\n");
                            fflush(stderr);
                            D3D12_RANGE wr = { 0, 0 };
                            dev->xfer.readback_staging->Unmap(0, &wr);
                        }
                    }
                }
            }
            // Probe inp_pos (I32) — first 8 positions
            if (probe_pos) {
                size_t bytes = (size_t)ggml_nelements(probe_pos) * sizeof(int32_t);
                if (bytes > 0 && bytes <= 4096) {
                    uint32_t off = (uint32_t)dx12_tensor_offset(probe_pos);
                    auto * sbuf = (dx12_buffer_context *)probe_pos->buffer->context;
                    dev->xfer_ensure_staging(0, bytes);
                    HRESULT hr = dev->xfer.cmd_alloc->Reset();
                    if (SUCCEEDED(hr)) hr = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
                    if (SUCCEEDED(hr)) {
                        dev->xfer.cmd_list->CopyBufferRegion(
                            dev->xfer.readback_staging.Get(), 0, sbuf->resource.Get(), off, bytes);
                        dev->xfer.cmd_list->Close();
                        ID3D12CommandList * lists[] = { dev->xfer.cmd_list.Get() };
                        dev->compute_queue->ExecuteCommandLists(1, lists);
                        dev->xfer.fence_value++;
                        dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                        dev->xfer_wait();
                        void * mp = nullptr;
                        D3D12_RANGE rr = { 0, bytes };
                        if (SUCCEEDED(dev->xfer.readback_staging->Map(0, &rr, &mp))) {
                            int32_t * ip = (int32_t *)mp;
                            int n = (int)ggml_nelements(probe_pos);
                            fprintf(stderr, "ggml-dx12: INPUT POS  ne=(%lld) off=%u: vals[0..%d]:",
                                    (long long)probe_pos->ne[0], off, std::min(n,8)-1);
                            for (int j = 0; j < 8 && j < n; j++) fprintf(stderr, " %d", ip[j]);
                            fprintf(stderr, "\n");
                            fflush(stderr);
                            D3D12_RANGE wr = { 0, 0 };
                            dev->xfer.readback_staging->Unmap(0, &wr);
                        }
                    }
                }
            }
            // Re-open compute CL so dispatch loop has a recording surface.
            // (xfer cmd lists are separate from compute CL, but ensure_cmd_list_open
            // is idempotent if already open.)
            bctx->ensure_cmd_list_open();
        }
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
        static bool no_fusion = []() {
            bool v = (getenv("GGML_DX12_NO_FUSION") != nullptr);
            if (v) DX12_LOG_INFO("GGML_DX12_NO_FUSION set: disabling all op fusion\n");
            return v;
        }();
        static bool no_prompt_fusion = []() {
            bool v = (getenv("GGML_DX12_NO_PROMPT_FUSION") != nullptr);
            if (v) DX12_LOG_INFO("GGML_DX12_NO_PROMPT_FUSION set: disabling op fusion during prompt eval\n");
            return v;
        }();
        bool skip_fusion = no_fusion || (is_prompt && no_prompt_fusion);
        // Per-fusion control: GGML_DX12_FUSION_MASK bitmask enables specific fusions:
        //   bit 0 (1):  ADD+RMS_NORM+MUL
        //   bit 1 (2):  RMS_NORM+MUL
        //   bit 2 (4):  RMS_NORM+MUL+ROPE  (disabled by default — broken on AMD RDNA 3)
        //   bit 3 (8):  ROPE+VIEW+SET_ROWS
        //   bit 4 (16): MUL_MAT+ADD bias
        // When set, only fusions with set bits are attempted. When not set, all fusions work.
        static int fusion_mask = -1;
        if (fusion_mask == -1) {
            const char * env = getenv("GGML_DX12_FUSION_MASK");
            fusion_mask = env ? atoi(dx12_env_unquote(env)) : 0x1B;  // 27 = all except RMS+MUL+ROPE
        }
        if (!skip_fusion && !(fusion_mask & 1) && !(fusion_mask & 2) && !(fusion_mask & 4) && !(fusion_mask & 8) && !(fusion_mask & 16))
            skip_fusion = true;

        if (!skip_fusion && (fusion_mask & 1) && node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
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
         if (!skip_fusion && (fusion_mask & 6) && !fused_add_rms_node && node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes) {
            struct ggml_tensor * next = cgraph->nodes[i + 1];
            if (next->op == GGML_OP_MUL && next->src[0] == node) {
                // Check for RMS_NORM + MUL + ROPE triple fusion
                if ((fusion_mask & 4) && i + 2 < cgraph->n_nodes) {
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
                if (!fused_rope_after_rms && (fusion_mask & 2)) {
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
        if (!skip_fusion && (fusion_mask & 8) && node->op == GGML_OP_ROPE && i + 2 < cgraph->n_nodes && !fused_add_rms_node) {
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

        // CPY/CONT/DUP fast path: src and dst both fully contiguous, same dtype,
        // and total bytes is a multiple of 16. The vast majority of CONT calls
        // for KV cache writes / residual reshape / output projection fixups land
        // here. Avoids the per-element flat_to_4d index math and (critically)
        // the InterlockedCompareExchange retry loop that the generic shader
        // uses for half-word stores.
        uint8_t cpy_variant = DX12_CPY_GENERIC;
        if ((node->op == GGML_OP_CPY || node->op == GGML_OP_CONT || node->op == GGML_OP_DUP)
            && node->src[0]) {
            const struct ggml_tensor * src = node->src[0];
            bool same_dtype  = (src->type == node->type);
            bool src_contig  = ggml_is_contiguous(src);
            bool dst_contig  = ggml_is_contiguous(node);
            bool same_count  = (ggml_nelements(src) == ggml_nelements(node));
            size_t total_bytes = (size_t)ggml_nelements(node) * ggml_type_size(node->type);
            bool aligned16   = ((total_bytes & 15) == 0);
            if (same_dtype && src_contig && dst_contig && same_count && aligned16) {
                key.flags  = 1;  // cpy_contig
                cpy_variant = DX12_CPY_CONTIG;
            }
        }

        // For MUL_MAT with M=1, use matvec pipeline (flags=1, or flags=5 for 256-thread auto-tuned)
        // Only for types that have matvec shaders
        bool is_matvec_dispatch = false;
        if (node->op == GGML_OP_MUL_MAT && node->ne[1] == 1 && node->src[0]) {
            ggml_type t = node->src[0]->type;
            if (t == GGML_TYPE_F16 || t == GGML_TYPE_F32 ||
                t == GGML_TYPE_BF16 ||
                t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q5_0 ||
                t == GGML_TYPE_Q8_0) {
                key.flags = 1;
                // Multi-row matvec (2 rows per group, halves activation traffic)
                if (t == GGML_TYPE_Q4_K) {
                    if (bctx->dev->q4k_use_32) key.flags = 6;  // 32t wave-only
                    else key.flags = 2;  // multi-row 256t (default)
                }
                if (t == GGML_TYPE_Q5_K && bctx->dev->q5k_use_mr) key.flags = 2;
                if (t == GGML_TYPE_Q6_K && bctx->dev->q6k_use_mr) key.flags = 2;
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
                t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q5_0) {
                key.flags = 4;
            }
        }

        // Op fusion: MUL_MAT(M=1) + ADD -> matvec with fused bias add
        struct ggml_tensor * fused_bias_add = nullptr;
        struct ggml_tensor * fused_bias_tensor = nullptr;
        if (!skip_fusion && (fusion_mask & 16) && is_matvec_dispatch && i + 1 < cgraph->n_nodes) {
            struct ggml_tensor * next = cgraph->nodes[i + 1];
            if (next->op == GGML_OP_ADD) {
                struct ggml_tensor * bias = nullptr;
                if (next->src[0] == node) bias = next->src[1];
                else if (next->src[1] == node) bias = next->src[0];
                // Bias must be:
                //   - F32, same ne[0] as output, contiguous
                //   - a LEAF model weight (op == NONE) NOT a computed tensor.
                //     Without this check the residual stream (which has the
                //     same shape as a bias for matvec rows) gets mis-fused as
                //     a bias on every layer. Mathematically equivalent in
                //     theory, but the in-place read/write on the residual
                //     buffer corrupts FFN-down output on UMA -- residual mean
                //     drifts to +40 by layer 9 and decoder NaNs by layer 11.
                bool bias_is_leaf = bias && bias->op == GGML_OP_NONE && bias->view_src == nullptr;
                if (bias && bias_is_leaf && bias->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
                    bias->ne[0] == node->ne[0] && ggml_is_contiguous(bias)) {
                    fused_bias_add = next;
                    fused_bias_tensor = bias;
                }
            }
        }

        // Flash Attention: use UMA-optimized variant for decode only (D≤128, single query)
        // Prefill (N_queries > 1) keeps the standard 256t shader for better batch throughput
        if (node->op == GGML_OP_FLASH_ATTN_EXT && bctx->dev->fa_use_uma) {
            uint32_t D = (uint32_t)node->src[0]->ne[0];
            uint32_t N_queries = (uint32_t)node->src[0]->ne[1];
            if (D <= 128 && N_queries == 1) key.flags = 2;  // UMA FA (128t, smaller tile)
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
            params.dst_esize = dx12_esize(fused_mul_node->type);
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
                params.src1_esize = dx12_esize(wt->type);
            }
            if (fused_rope_after_rms) {
                // RMS_NORM + MUL + ROPE: dst is ROPE's output
                params.ne0 = (uint32_t)fused_rope_after_rms->ne[0]; params.ne1 = (uint32_t)fused_rope_after_rms->ne[1];
                params.ne2 = (uint32_t)fused_rope_after_rms->ne[2]; params.ne3 = (uint32_t)fused_rope_after_rms->ne[3];
                params.nb0 = (uint32_t)fused_rope_after_rms->nb[0]; params.nb1 = (uint32_t)fused_rope_after_rms->nb[1];
                params.nb2 = (uint32_t)fused_rope_after_rms->nb[2]; params.nb3 = (uint32_t)fused_rope_after_rms->nb[3];
                params.dst_offset = (uint32_t)dx12_tensor_offset(fused_rope_after_rms);
                params.dst_esize = dx12_esize(fused_rope_after_rms->type);
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
                params.dst_esize = dx12_esize(fused_mul_node->type);
            }
            params.dst_esize = dx12_esize(fused_mul_node->type);
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
            params.dst_esize = dx12_esize(fused_rope_set_rows->type);
        }

        // Split-KV flash attention: precompute split_k before params are pushed
        uint32_t fa_split_k = 1;
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            static int splitk_env = []() {
                const char * env = getenv("GGML_DX12_SPLIT_K");
                if (!env) return 0;
                int v = atoi(dx12_env_unquote(env));
                DX12_LOG_INFO("GGML_DX12_SPLIT_K env=\"%s\" parsed=%d (0=auto, 1=disabled, >=2=force)\n", env, v);
                return v;
            }();

            uint32_t N_queries = (uint32_t)node->src[0]->ne[1];
            uint32_t N_kv      = (uint32_t)node->src[1]->ne[1];
            uint32_t n_heads   = (uint32_t)node->src[0]->ne[2];
            uint32_t batch     = (uint32_t)node->src[0]->ne[3];

            if (N_queries == 1 && N_kv >= 512) {
                if (splitk_env == 1) {
                    // Explicit disable: force fa_split_k = 1 (no split)
                    fa_split_k = 1;
                } else if (splitk_env > 1) {
                    fa_split_k = (uint32_t)splitk_env;
                    // Still cap to avoid too many empty splits (wastes temp memory)
                    uint32_t max_split = (N_kv + 255) / 256;
                    if (fa_split_k > max_split) fa_split_k = max_split;
                    if (fa_split_k < 2) fa_split_k = 1;
                } else {
                    // Auto split-KV heuristic: target enough thread groups to
                    // saturate the GPU's compute units with multiple wavefronts.
                    //
                    // RDNA 2 (Xbox, AMD discrete) needs 4-8+ wavefronts/CU to hide
                    // memory latency. With 52 CUs on Xbox, that means 200-400+ groups.
                    // NVIDIA and smaller APUs are less sensitive but still benefit.
                    //
                    // Platform-specific targets:
                    //   - Xbox (52 CUs, UMA):  256 groups -- ~5 waves/CU
                    //   - Discrete GPU:        128 groups -- conservative, good for 32-96 CUs
                    //   - APU/iGPU (16 CUs):    64 groups -- 4 waves/CU, sufficient
                    //
                    // Formula: split_k = ceil(target / (n_heads * batch)),
                    // capped so each split has at least one tile worth of KV tokens.
                    uint32_t target_groups;
#if defined(GGML_DX12_XBOX_GDKX)
                    target_groups = 256;
#else
                    target_groups = bctx->dev->is_uma ? 128 : 128;
#endif
                    uint32_t tile_kv = (key.flags == 2) ? 128 : 256; // UMA shader uses TILE_KV=128
                    uint32_t total_wgs = N_queries * n_heads * batch;
                    if (total_wgs < target_groups) {
                        fa_split_k = (target_groups + total_wgs - 1) / total_wgs;
                        uint32_t max_split = (N_kv + tile_kv - 1) / tile_kv;
                        if (fa_split_k > max_split) fa_split_k = max_split;
                        if (fa_split_k < 2) fa_split_k = 1;
                    }
                }
            }

            if (fa_split_k > 1) {
                // Allocate/grow temp buffer for partial results
                uint32_t D    = (uint32_t)node->src[0]->ne[0];
                uint32_t ne01 = N_queries;
                uint32_t ne02 = n_heads;
                uint32_t ne03 = batch;
                size_t o_size  = (size_t)D * ne01 * fa_split_k * ne02 * ne03 * sizeof(float);
                size_t ml_size = (size_t)ne01 * fa_split_k * ne02 * ne03 * sizeof(float);
                size_t temp_size = o_size + 2 * ml_size;
                temp_size = (temp_size + 255) & ~255;

                if (temp_size > bctx->splitkv_temp_size) {
                    D3D12_HEAP_PROPERTIES hp = {};
                    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
                    D3D12_RESOURCE_DESC rd = {};
                    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                    rd.Width     = temp_size;
                    rd.Height    = 1;
                    rd.DepthOrArraySize = 1;
                    rd.MipLevels = 1;
                    rd.SampleDesc.Count = 1;
                    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                    rd.Flags  = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

                    bctx->splitkv_temp.Reset();
                    HRESULT hr = bctx->dev->device->CreateCommittedResource(
                        &hp, D3D12_HEAP_FLAG_NONE, &rd,
                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                        IID_PPV_ARGS(&bctx->splitkv_temp));
                    if (FAILED(hr)) {
                        DX12_LOG_WARN("Failed to allocate split-KV temp buffer (%zu bytes)\n", temp_size);
                        fa_split_k = 1;
                    } else {
                        bctx->splitkv_temp_size = temp_size;
                    }
                }
            }

            params.op_params[15] = fa_split_k;
        }

        // Upload root constants -- only upload op_params for ops that need them
        static constexpr uint32_t BASE_PARAMS = 30;  // ne/nb/offsets/esizes = 30 DWORDs
        bool needs_op_params = (node->op == GGML_OP_SOFT_MAX || 
                                 node->op == GGML_OP_FLASH_ATTN_EXT || 
                                 node->op == GGML_OP_ROPE ||
                                 node->op == GGML_OP_RMS_NORM ||
                                 node->op == GGML_OP_NORM ||
                                 node->op == GGML_OP_GLU ||
                                 node->op == GGML_OP_SCALE ||
                                 node->op == GGML_OP_MUL_MAT ||  // matvec shaders read op0/op1 for bias fusion
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
        ID3D12Resource * fa_real_dst_res = nullptr;  // saved for split-KV reduce pass
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

        // Split-KV: redirect main FA dispatch to write partials to temp buffer
        if (fa_split_k > 1 && bctx->splitkv_temp) {
            fa_real_dst_res = dst_res;
            dst_res = bctx->splitkv_temp.Get();
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

                            if (trace_prompt) {
                                int src0t = node->src[0] ? (int)node->src[0]->type : -1;
                                fprintf(stderr, "ggml-dx12: dispatch #%d MUL_MAT(src0t=%d) chunk [%u..%u/%u] groups=(%u,1,%u)\n",
                                        i, src0t, cs, cs + cr, N, cr, batches);
                                fflush(stderr);
                            }

                            if (ts_active) bctx->ts_record_dispatch_pre();
                            bctx->cmd_list->Dispatch(cr, 1, batches);
                            if (ts_active) {
                                uint8_t s0t = (node->src[0] && node->src[0]->type < GGML_TYPE_COUNT)
                                              ? (uint8_t)node->src[0]->type : (uint8_t)0xFFu;
                                bctx->ts_record_dispatch_post((uint16_t)GGML_OP_MUL_MAT, s0t, /*is_matvec=*/1);
                            }

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

                        // Flush after chunked dispatch for decode: the `continue` below
                        // skips the periodic decode flush, so explicitly drain here.
                        if (!is_prompt) {
                            bctx->close_and_execute();
                            bctx->wait_for_gpu();
                            bctx->ensure_cmd_list_open();
                            bctx->reset_binding_cache();
                        }

                        unsynced_writes.insert((uintptr_t)node);  // no fusions, dst = node
                        dispatch_weight = 0;
                        continue;  // skip normal dispatch path
                    }

                    // Normal matvec dispatch (N <= MATVEC_CHUNK)
                    if (key.flags == 2) {
                        // Multi-row: 2 rows per thread group
                        groups_x = (N + 1) / 2;
                    } else {
                        // flags==1 (standard), flags==5 (auto-tune alt), flags==6 (32t wave-only)
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
                // groups_x encodes query_idx * split_k + split_k_index (split_k precomputed above)
                uint32_t N_queries = (uint32_t)node->src[0]->ne[1];
                groups_x = N_queries * fa_split_k;
                groups_y = (uint32_t)node->src[0]->ne[2]; // n_heads
                groups_z = (uint32_t)node->src[0]->ne[3]; // batch
                break;
            }
            case GGML_OP_SET_ROWS: {
                // 2D dispatch to handle tensors exceeding 65535 groups in X
                uint32_t total_elements = (uint32_t)(ggml_nelements(node));
                uint32_t total_groups = (total_elements + 255) / 256;
                if (total_groups <= 65535) {
                    groups_x = total_groups;
                } else {
                    groups_x = 65535;
                    groups_y = (total_groups + 65534) / 65535;
                }
                break;
            }
            default: {
                // Element-wise: one thread per element
                uint32_t total_elements = (uint32_t)(ggml_nelements(node));
                groups_x = (total_elements + 255) / 256;
                break;
            }
        }

        // CPY/CONT/DUP fast path override: one thread per 16-byte uint4.
        if (cpy_variant == DX12_CPY_CONTIG) {
            size_t total_bytes = (size_t)ggml_nelements(node) * ggml_type_size(node->type);
            uint32_t total_quads = (uint32_t)(total_bytes >> 4);
            groups_x = (total_quads + 255) / 256;
            groups_y = 1;
            groups_z = 1;
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
        if (do_profile || dx12_perf) { QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&t0); }
#else
        std::chrono::steady_clock::time_point t0;
        if (do_profile || dx12_perf) { t0 = std::chrono::steady_clock::now(); }
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

            // Force unconditional UAV barriers before every dispatch.
            // On Xbox, the pointer-based dependency tracker above misses
            // byte-range aliasing (ggml allocator reuses offsets). Force
            // unconditional barriers as a correctness requirement.
            // On desktop, enable via GGML_DX12_BARRIER_ALL=1 for debugging.
            static bool force_barriers = []() {
#if defined(GGML_DX12_XBOX_GDKX)
                bool v = true;  // default ON for Xbox
                if (getenv("GGML_DX12_BARRIER_ALL") && atoi(dx12_env_unquote(getenv("GGML_DX12_BARRIER_ALL"))) == 0) {
                    v = false;  // allow explicit disable for testing
                    DX12_LOG_INFO("GGML_DX12_BARRIER_ALL=0: forced barriers disabled on Xbox\n");
                }
#else
                bool v = (getenv("GGML_DX12_BARRIER_ALL") != nullptr);
#endif
                if (v) DX12_LOG_INFO("GGML_DX12_BARRIER_ALL: forcing UAV barrier before every dispatch\n");
                return v;
            }();
            if (force_barriers) {
                D3D12_RESOURCE_BARRIER barrier = {};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.UAV.pResource = nullptr;
                bctx->cmd_list->ResourceBarrier(1, &barrier);
                unsynced_writes.clear();
            }
        }

        // TDR diagnostic: print every dispatch during traced graphs (prompt + first decode)
        if (trace_prompt) {
            int src0t = node->src[0] ? (int)node->src[0]->type : -1;
            if (node->op == GGML_OP_FLASH_ATTN_EXT && fa_split_k > 1) {
                fprintf(stderr, "ggml-dx12: dispatch #%d FA(split_k=%u) groups=(%u,%u,%u) ne=(%lld,%lld,%lld,%lld)\n",
                        i, fa_split_k, groups_x, groups_y, groups_z,
                        (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3]);
            } else {
                fprintf(stderr, "ggml-dx12: dispatch #%d %s(src0t=%d) groups=(%u,%u,%u) ne=(%lld,%lld,%lld,%lld)\n",
                        i, ggml_op_name(node->op), src0t, groups_x, groups_y, groups_z,
                        (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3]);
            }
            fflush(stderr);
        }

        // D3D12 dispatch limit: 65535 per axis
        if (groups_x > 65535 || groups_y > 65535 || groups_z > 65535) {
            if (dx12_diag) {
                fprintf(stderr, "ggml-dx12: SKIP dispatch #%d %s — groups (%u,%u,%u) exceeds D3D12 limit 65535\n",
                        i, ggml_op_name(node->op), groups_x, groups_y, groups_z);
            }
            continue;
        }

        // FA input probe: when tracing decode, read the first 32 elements of
        // each FA source (Q/K/V/mask) and report NaN counts. This isolates
        // whether the FA shader is generating NaN from clean inputs (bug in
        // shader/op_params) or whether the K/V cache contains NaN (upstream
        // SET_ROWS or projection bug).
        if (dx12_diag && trace_prompt && !is_prompt && node->op == GGML_OP_FLASH_ATTN_EXT) {
            auto probe_src = [&](const char * label, struct ggml_tensor * src) {
                if (!src || !src->buffer) return;
                size_t off = (size_t)dx12_tensor_offset(src);
                size_t esz = ggml_type_size(src->type);
                size_t n_to_read = 32;
                size_t bytes = n_to_read * esz;
                bool is_f16 = (src->type == GGML_TYPE_F16);
                bool is_f32 = (src->type == GGML_TYPE_F32);
                if (!is_f16 && !is_f32) {
                    fprintf(stderr, "ggml-dx12: FA INPUT d#%d %s type=%d off=%zu (skipped, not F16/F32)\n",
                            i, label, (int)src->type, off);
                    fflush(stderr);
                    return;
                }
                bctx->close_and_execute();
                bctx->wait_for_gpu();
                auto * dev = bctx->dev;
                dev->init_xfer();
                dev->xfer_ensure_staging(0, bytes);
                dev->xfer_wait();
                HRESULT hrp = dev->xfer.cmd_alloc->Reset();
                if (SUCCEEDED(hrp)) hrp = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
                if (SUCCEEDED(hrp)) {
                    auto * pbuf = (dx12_buffer_context *)src->buffer->context;
                    dev->xfer.cmd_list->CopyBufferRegion(
                        dev->xfer.readback_staging.Get(), 0, pbuf->resource.Get(), off, bytes);
                    dev->xfer.cmd_list->Close();
                    ID3D12CommandList * lists[] = { dev->xfer.cmd_list.Get() };
                    dev->compute_queue->ExecuteCommandLists(1, lists);
                    dev->xfer.fence_value++;
                    dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                    dev->xfer_wait();
                    void * mp = nullptr;
                    D3D12_RANGE rr = { 0, bytes };
                    if (SUCCEEDED(dev->xfer.readback_staging->Map(0, &rr, &mp))) {
                        int nans = 0, infs = 0;
                        float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
                        if (is_f32) {
                            const float * f = (const float *)mp;
                            for (size_t j = 0; j < n_to_read; ++j) {
                                if (isnan(f[j])) nans++;
                                else if (isinf(f[j])) infs++;
                            }
                            v0 = f[0]; v1 = f[1]; v2 = f[2]; v3 = f[3];
                        } else { // F16
                            const uint16_t * h = (const uint16_t *)mp;
                            auto h2f = [](uint16_t x) -> float {
                                uint32_t s = (uint32_t)(x & 0x8000) << 16;
                                uint32_t e = (x >> 10) & 0x1f;
                                uint32_t m = x & 0x3ff;
                                uint32_t r;
                                if (e == 0) {
                                    if (m == 0) r = s;
                                    else { while (!(m & 0x400)) { m <<= 1; e--; } e++; m &= 0x3ff;
                                           r = s | ((e + 112) << 23) | (m << 13); }
                                } else if (e == 31) {
                                    r = s | 0x7f800000 | (m << 13);
                                } else {
                                    r = s | ((e + 112) << 23) | (m << 13);
                                }
                                float f; memcpy(&f, &r, 4); return f;
                            };
                            for (size_t j = 0; j < n_to_read; ++j) {
                                float fv = h2f(h[j]);
                                if (isnan(fv)) nans++;
                                else if (isinf(fv)) infs++;
                            }
                            v0 = h2f(h[0]); v1 = h2f(h[1]); v2 = h2f(h[2]); v3 = h2f(h[3]);
                        }
                        fprintf(stderr, "ggml-dx12: FA INPUT d#%d %s type=%d off=%zu ne=(%lld,%lld,%lld,%lld) nb=(%zu,%zu,%zu,%zu) nans=%d/%zu infs=%d v[0..3]=%.4g %.4g %.4g %.4g\n",
                                i, label, (int)src->type, off,
                                (long long)src->ne[0], (long long)src->ne[1], (long long)src->ne[2], (long long)src->ne[3],
                                src->nb[0], src->nb[1], src->nb[2], src->nb[3],
                                nans, n_to_read, infs, v0, v1, v2, v3);
                        fflush(stderr);
                        D3D12_RANGE wr = { 0, 0 };
                        dev->xfer.readback_staging->Unmap(0, &wr);
                    }
                }
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                unsynced_writes.clear();
            };
            probe_src("Q   (src0)", node->src[0]);
            probe_src("K   (src1)", node->src[1]);
            probe_src("V   (src2)", node->src[2]);
            probe_src("Mask(src3)", node->src[3]);
            // Print dst offset and nb for FA to check for buffer aliasing
            fprintf(stderr, "ggml-dx12: FA DST  d#%d off=%u nb=(%u,%u,%u,%u) ne=(%lld,%lld,%lld,%lld) esize=%u\n",
                    i, params.dst_offset, params.nb0, params.nb1, params.nb2, params.nb3,
                    (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3],
                    params.dst_esize);
            fflush(stderr);
        }

        if (ts_active) bctx->ts_record_dispatch_pre();
        bctx->cmd_list->Dispatch(groups_x, groups_y, groups_z);

        // Split-KV reduce pass: merge partial O, M, L into final output
        if (fa_split_k > 1 && fa_real_dst_res) {
            // UAV barrier between main FA and reduce
            D3D12_RESOURCE_BARRIER barrier = {};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = nullptr;
            bctx->cmd_list->ResourceBarrier(1, &barrier);

            // Get/create reduce pipeline
            dx12_pipeline_key reduce_key = {};
            reduce_key.op    = GGML_OP_FLASH_ATTN_EXT;
            reduce_key.flags = 1; // reduce variant
            dx12_pipeline * reduce_pipeline = bctx->dev->get_or_create_pipeline(reduce_key);
            if (reduce_pipeline && reduce_pipeline->pso) {
                bctx->cmd_list->SetPipelineState(reduce_pipeline->pso.Get());
                bctx->last_pso = reduce_pipeline->pso.Get();

                // Bind temp buffer as src0 (SRV) for reduce shader
                D3D12_GPU_VIRTUAL_ADDRESS temp_va = bctx->splitkv_temp->GetGPUVirtualAddress();
                bctx->cmd_list->SetComputeRootShaderResourceView(1, temp_va);
                bctx->last_src0_va = temp_va;

                // Bind real destination as UAV for reduce output
                D3D12_GPU_VIRTUAL_ADDRESS real_dst_va = fa_real_dst_res->GetGPUVirtualAddress();
                bctx->cmd_list->SetComputeRootUnorderedAccessView(3, real_dst_va);
                bctx->last_dst_va = real_dst_va;

                // Build reduce params: op0=D, op1=ne01, op2=split_k, op3=ne02, op4=ne03, op5=dst_esize
                dx12_shader_params reduce_params = {};
                uint32_t D    = (uint32_t)node->src[0]->ne[0];
                uint32_t ne01 = (uint32_t)node->src[0]->ne[1];
                uint32_t ne02 = (uint32_t)node->src[0]->ne[2];
                uint32_t ne03 = (uint32_t)node->src[0]->ne[3];

                // Copy dst layout from original params for final output addressing
                reduce_params.ne0 = params.ne0; reduce_params.ne1 = params.ne1;
                reduce_params.ne2 = params.ne2; reduce_params.ne3 = params.ne3;
                reduce_params.nb0 = params.nb0; reduce_params.nb1 = params.nb1;
                reduce_params.nb2 = params.nb2; reduce_params.nb3 = params.nb3;
                reduce_params.dst_offset = params.dst_offset;
                reduce_params.dst_esize  = params.dst_esize;

                reduce_params.op_params[0] = D;
                reduce_params.op_params[1] = ne01;
                reduce_params.op_params[2] = fa_split_k;
                reduce_params.op_params[3] = ne02;
                reduce_params.op_params[4] = ne03;
                reduce_params.op_params[5] = params.dst_esize;

                bctx->cmd_list->SetComputeRoot32BitConstants(0, (uint32_t)(sizeof(reduce_params) / 4), &reduce_params, 0);

                // Dispatch reduce: one group per (query, head, batch)
                // REDUCE_GROUP_SIZE=256, each thread handles D/256 dimensions
                bctx->cmd_list->Dispatch(ne01, ne02, ne03);
            }
        }

        if (ts_active) {
            uint8_t s0t  = (node->src[0] && node->src[0]->type < GGML_TYPE_COUNT)
                           ? (uint8_t)node->src[0]->type : (uint8_t)0xFFu;
            uint8_t variant;
            if (node->op == GGML_OP_MUL_MAT) {
                variant = (node->ne[1] == 1) ? 1 : 0;  // 1=matvec, 0=GEMM
            } else if (node->op == GGML_OP_CPY || node->op == GGML_OP_CONT || node->op == GGML_OP_DUP) {
                variant = cpy_variant;
            } else {
                variant = 0;
            }
            bctx->ts_record_dispatch_post((uint16_t)node->op, s0t, variant);
        }

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

        // Periodic CL flush for decode phase (M=1) to prevent stale GPU caches
        // on AMD RDNA 3.  CL boundaries provide GPU pipeline drains (separate
        // ExecuteCommandLists calls are implicitly ordered).  No CPU-side wait
        // is needed — the cmd allocator ring handles CPU-GPU pipelining.
        // Default: flush after every dispatch (interval=1).
        // Override via GGML_DX12_DECODE_FLUSH env var.
        // Suppress while recording into a graph-cache CL: cached graphs are
        // submitted as a single CL on replay, so we must not split the
        // recording CL mid-graph either.
        if (!is_prompt && cgraph->n_nodes > 10 && !bctx->cache_recording) {
            static int decode_flush_interval = 0;
            if (decode_flush_interval == 0) {
                const char * env = getenv("GGML_DX12_DECODE_FLUSH");
                if (!env) env = getenv("GGML_DX12_FLUSH_INTERVAL");
                decode_flush_interval = env ? atoi(dx12_env_unquote(env)) : 1;
                if (decode_flush_interval <= 0) decode_flush_interval = 1;
            }
            dispatch_weight++;
            if (dispatch_weight >= decode_flush_interval) {
                bctx->close_and_execute();
                // No wait_for_gpu() -- CL boundary is the GPU pipeline drain.
                // The ring of CMD_RING_SIZE allocators handles CPU pipelining.
                bctx->ensure_cmd_list_open();
                unsynced_writes.clear();
                dispatch_weight = 0;
            }
        }

        // Force CL split at every dispatch in model head region to fix NaN and sync.
        // AMD RDNA 3 requires full pipeline drains (not just UAV barriers) between
        // the final FFN/norm layers and the vocab projection + logit softcapping.
        // Without these splits, the model head produces NaN due to stale GPU caches.
        // Suppressed during cache recording for the same reason as the decode
        // flush: the cached CL is one indivisible unit on replay.
        if (cgraph->n_nodes > 1000 && i >= cgraph->n_nodes - 25 && i <= cgraph->n_nodes - 2 && !bctx->cache_recording) {
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
        } else if (dx12_perf && !ts_active) {
            // Fallback: GPU-timestamp init failed; use the legacy wall-clock
            // path that does close+wait per dispatch. CPU-saturating but
            // accurate for the GPU portion.
            bctx->close_and_execute();
            bctx->wait_for_gpu();
            bctx->ensure_cmd_list_open();
            bctx->reset_binding_cache();
            unsynced_writes.clear();
#ifdef _WIN32
            QueryPerformanceCounter(&t1);
            int64_t elapsed_us = (int64_t)((double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart * 1e6);
#else
            auto t1 = std::chrono::steady_clock::now();
            int64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
#endif
            dx12_op_counts[node->op].fetch_add(1, std::memory_order_relaxed);
            dx12_op_time_us[node->op].fetch_add(elapsed_us, std::memory_order_relaxed);
            // MUL_MAT breakdown by src0 type
            if (node->op == GGML_OP_MUL_MAT && node->src[0]) {
                int type = node->src[0]->type;
                if (type >= 0 && type < GGML_TYPE_COUNT) {
                    if (node->ne[1] == 1) {
                        dx12_mm_vec_counts[type].fetch_add(1, std::memory_order_relaxed);
                        dx12_mm_vec_time_us[type].fetch_add(elapsed_us, std::memory_order_relaxed);
                    } else {
                        dx12_mm_gemm_counts[type].fetch_add(1, std::memory_order_relaxed);
                        dx12_mm_gemm_time_us[type].fetch_add(elapsed_us, std::memory_order_relaxed);
                    }
                }
            }
        }
        // ts_active path: per-op accumulation happens in ts_finalize_graph()
        // at end of graph_compute. No per-dispatch CPU work.

        // Per-dispatch tensor dump: sync GPU, read back output, write to file
        if (dump_this_dispatch_graph) {
            // Determine the effective output tensor (accounting for fusion)
            struct ggml_tensor * dump_tensor = fused_rope_after_rms ? fused_rope_after_rms :
                                               (fused_mul_node ? fused_mul_node :
                                               (fused_bias_add ? fused_bias_add :
                                               (fused_rope_set_rows ? fused_rope_set_rows : node)));

            // GPU sync (if not already synced).
            // The legacy wall-clock dx12_perf path syncs after every dispatch.
            // The new GPU-timestamp path (ts_active) does NOT, so we still need
            // to sync here to safely read back the dispatched tensor.
            bool already_synced = (!ts_active && dx12_perf) || do_profile;
            if (!already_synced) {
                bctx->close_and_execute();
                bctx->wait_for_gpu();
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                unsynced_writes.clear();
            }

            auto * dump_buf_ctx = (dx12_buffer_context *)(dump_tensor->buffer ? dump_tensor->buffer->context : nullptr);
            if (dump_buf_ctx && dump_buf_ctx->resource) {
                size_t offset = (size_t)dx12_tensor_offset(dump_tensor);
                size_t nbytes = ggml_nbytes(dump_tensor);
                int64_t nelements = ggml_nelements(dump_tensor);
                auto * dev = bctx->dev;

                dev->init_xfer();
                dev->xfer_ensure_staging(0, nbytes);
                dev->xfer_wait();

                HRESULT hr_dd = dev->xfer.cmd_alloc->Reset();
                if (SUCCEEDED(hr_dd)) hr_dd = dev->xfer.cmd_list->Reset(dev->xfer.cmd_alloc.Get(), nullptr);
                if (SUCCEEDED(hr_dd)) {
                    dev->xfer.cmd_list->CopyBufferRegion(
                        dev->xfer.readback_staging.Get(), 0,
                        dump_buf_ctx->resource.Get(), offset, nbytes);
                    dev->xfer.cmd_list->Close();
                    ID3D12CommandList * dlists[] = { dev->xfer.cmd_list.Get() };
                    dev->compute_queue->ExecuteCommandLists(1, dlists);
                    dev->xfer.fence_value++;
                    dev->compute_queue->Signal(dev->xfer.fence.Get(), dev->xfer.fence_value);
                    dev->xfer_wait();

                    void * mapped = nullptr;
                    D3D12_RANGE rr = { 0, nbytes };
                    hr_dd = dev->xfer.readback_staging->Map(0, &rr, &mapped);
                    if (SUCCEEDED(hr_dd)) {
                        // Convert to F32
                        std::vector<float> f32_data;
                        uint32_t n_floats = 0;

                        if (dump_tensor->type == GGML_TYPE_F32) {
                            n_floats = (uint32_t)nelements;
                            f32_data.assign((float *)mapped, (float *)mapped + nelements);
                        } else if (dump_tensor->type == GGML_TYPE_F16) {
                            n_floats = (uint32_t)nelements;
                            f32_data.resize(nelements);
                            uint16_t * hp = (uint16_t *)mapped;
                            for (int64_t j = 0; j < nelements; j++) {
                                uint16_t h = hp[j];
                                uint32_t sign = (uint32_t)(h >> 15) << 31;
                                uint32_t exp = (h >> 10) & 0x1F;
                                uint32_t mant = h & 0x3FF;
                                if (exp == 0) {
                                    if (mant == 0) { f32_data[j] = (h & 0x8000) ? -0.0f : 0.0f; continue; }
                                    exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF;
                                } else if (exp == 31) {
                                    uint32_t r = sign | 0x7F800000 | (mant << 13);
                                    memcpy(&f32_data[j], &r, 4); continue;
                                }
                                uint32_t r = sign | ((exp + 112) << 23) | (mant << 13);
                                memcpy(&f32_data[j], &r, 4);
                            }
                        } else if (dump_tensor->type == GGML_TYPE_I32) {
                            n_floats = (uint32_t)nelements;
                            f32_data.resize(nelements);
                            int32_t * ip = (int32_t *)mapped;
                            for (int64_t j = 0; j < nelements; j++) f32_data[j] = (float)ip[j];
                        }

                        char filepath[1024];
                        snprintf(filepath, sizeof(filepath), "%s/node_%04d_%s.bin",
                                 dump_dispatch_dir, i, ggml_op_name(dump_tensor->op));
                        FILE * df = fopen(filepath, "wb");
                        if (df) {
                            uint32_t magic = 0x31444747; // "GGD1"
                            uint32_t version = 1;
                            uint32_t nidx = (uint32_t)i;
                            uint32_t op_id = (uint32_t)dump_tensor->op;
                            uint32_t type_id = (uint32_t)dump_tensor->type;
                            uint32_t nlen = (uint32_t)strlen(dump_tensor->name);
                            uint32_t nf = n_floats;
                            fwrite(&magic, 4, 1, df);
                            fwrite(&version, 4, 1, df);
                            fwrite(&nidx, 4, 1, df);
                            fwrite(&op_id, 4, 1, df);
                            fwrite(&type_id, 4, 1, df);
                            fwrite(dump_tensor->ne, sizeof(int64_t), 4, df);
                            fwrite(&nlen, 4, 1, df);
                            fwrite(&nf, 4, 1, df);
                            fwrite(dump_tensor->name, 1, nlen, df);
                            if (n_floats > 0) {
                                fwrite(f32_data.data(), sizeof(float), n_floats, df);
                            } else {
                                fwrite(mapped, 1, nbytes, df);
                            }
                            fclose(df);
                        }

                        D3D12_RANGE wr_dd = { 0, 0 };
                        dev->xfer.readback_staging->Unmap(0, &wr_dd);
                    }
                }
            }
            // Ensure command list is open for the next dispatch
            bctx->ensure_cmd_list_open();
            bctx->reset_binding_cache();
            unsynced_writes.clear();
        }

        // TDR prevention: flush during prompt processing.
        // Suppressed during cache recording (the cached CL is one unit).
        if (is_prompt && !bctx->cache_recording) {
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
    if (dx12_diag && bctx->cmd_list_open) {
        bctx->close_and_execute();
        bctx->wait_for_gpu();
        fprintf(stderr, "ggml-dx12: final flush OK (all %d dispatches completed, %s)\n",
                cgraph->n_nodes, is_prompt ? "prompt" : "decode");
        fflush(stderr);

        // Read back the hidden state (src[1] of the vocab projection MUL_MAT)
        // to compare with CPU reference. Find last MUL_MAT with large output.
        // Fires for both prompt and decode graphs so we can compare the two.
        static int hidden_diag_count = 0;
        if (hidden_diag_count < 4) {
            hidden_diag_count++;
            struct ggml_tensor * vocab_mm = nullptr;
            int search_lo = std::max(0, cgraph->n_nodes - 10);
            for (int ni = cgraph->n_nodes - 1; ni >= search_lo; ni--) {
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

    // ---- Cache RECORD: finalize the recording CL, submit it for this
    //      execution, store the entry. Subsequent matching graphs hit. ----
    if (record_this_graph && bctx->cache_recording) {
        // The recording CL is currently 'open' (bctx->cmd_list_open=true,
        // bctx->cmd_list = recording_entry.cl). Close it.
        HRESULT hr = bctx->cmd_list->Close();
        DX12_CHECK(hr, "CommandList::Close (cache record)");
        bctx->cmd_list_open = false;

        // Capture dispatch count before storing
        recording_entry.hash    = cache_hash;
        recording_entry.n_nodes = cgraph->n_nodes;
        // (n_dispatches not tracked precisely; could add a counter if needed)

        // Submit it for execution this time around.
        ID3D12CommandList * lists[] = { recording_entry.cl.Get() };
        bctx->dev->compute_queue->ExecuteCommandLists(1, lists);
        bctx->fence_value++;
        hr = bctx->dev->compute_queue->Signal(bctx->fence.Get(), bctx->fence_value);
        DX12_CHECK(hr, "Signal fence (cache record)");

        // Restore live cmd_list state. The next graph_compute (or any code
        // that calls ensure_cmd_list_open) will allocate a fresh slot from
        // the ring, so we just clear the swap-saved fields and let the
        // ring take over again.
        bctx->cmd_list      = saved_cmd_list;       // may be null; that's OK
        bctx->cmd_list_open = false;                // any saved state was already closed before the swap
        (void)saved_cmd_list_open;
        bctx->reset_binding_cache();

        // Store the cache entry. Use std::move so the ComPtrs don't refcount
        // bump unnecessarily.
        bctx->graph_cache.emplace(cache_hash, std::move(recording_entry));
        bctx->cache_recording        = false;
        bctx->cache_recording_entry  = nullptr;

        if (cache_diag) {
            fprintf(stderr, "ggml-dx12: GRAPH CACHE record hash=0x%016llX nodes=%d total_entries=%zu\n",
                    (unsigned long long)cache_hash, cgraph->n_nodes,
                    bctx->graph_cache.size());
            fflush(stderr);
        }
    }

    // GPU-timestamp finalize: drain CL, wait once, read back, accumulate.
    // Single fence wait per graph (~one per token) instead of per-dispatch.
    if (ts_active) {
        bctx->ts_finalize_graph();
    }

    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Auto-tuning: benchmark shader variants and pick the fastest per GPU
// ---------------------------------------------------------------------------

void dx12_device::run_autotune() {
    if (tuning_done) return;
    tuning_done = true;

#ifndef _WIN32
    // WSL2: timestamp queries and benchmark dispatches can hang on some
    // GPU-PV configurations.  Use safe defaults instead of benchmarking.
    if (is_uma) {
        q5_0_use_256  = false;
        q8_0_use_256  = false;
        q6k_use_32    = false;
        f16_use_load4 = false;
        q5k_use_mr    = true;    // multi-row Q5_K: 14% faster than standard 256t
        q6k_use_mr    = false;   // disabled: q6k_mr shader 2.1x slower than standard 256t
        fa_use_uma    = true;    // UMA FA: 128t, fewer barriers, no idle threads (D≤128)
        DX12_LOG_INFO("Auto-tune: skipped on WSL2 (UMA defaults: Q5_K=mr FA=uma)\n");
    } else {
        DX12_LOG_INFO("Auto-tune: skipped on WSL2 (using defaults)\n");
    }
    return;
#endif

    // Check for cache file in current directory (easy to verify/delete)
    char cache_path[512];
    snprintf(cache_path, sizeof(cache_path), ".ggml_dx12_tune_%04X_%04X.txt",
             vendor_id, device_id);

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
            // On UMA, also enable multi-row variants (activation sharing)
            if (is_uma) {
                q5k_use_mr = true;   // multi-row Q5_K: 14% faster than standard 256t
                q6k_use_mr = false;  // disabled: q6k_mr shader 2.1x slower than standard 256t
                fa_use_uma = true;   // UMA FA: 128t, fewer barriers, no idle threads
            }
            // Per-flag env overrides (apply regardless of cache)
            if (const char * e = getenv("GGML_DX12_Q5_0_USE_256")) q5_0_use_256 = (atoi(dx12_env_unquote(e)) != 0);
            if (const char * e = getenv("GGML_DX12_Q8_0_USE_256")) q8_0_use_256 = (atoi(dx12_env_unquote(e)) != 0);
            if (const char * e = getenv("GGML_DX12_Q5K_USE_MR"))   q5k_use_mr   = (atoi(dx12_env_unquote(e)) != 0);
            if (const char * e = getenv("GGML_DX12_FA_USE_UMA"))   fa_use_uma   = (atoi(dx12_env_unquote(e)) != 0);
            DX12_LOG_INFO("Auto-tune v%d loaded from cache '%s': Q5_0=%s Q8_0=%s Q6_K=%s F16=%s%s\n", ver,
                          cache_path,
                          q5_0_use_256 ? "256t" : "32t", q8_0_use_256 ? "256t" : "32t",
                          q6k_use_32 ? "32t" : "256t", f16_use_load4 ? "load4" : "load2",
                          is_uma ? (std::string(" +UMA(Q5_K=") + (q5k_use_mr ? "mr" : "std") +
                                     " FA=" + (fa_use_uma ? "uma" : "std") + ")").c_str() : "");
            return;
        }
        fclose(f);
        DX12_LOG_INFO("Auto-tune cache '%s' version mismatch or corrupt — regenerating\n", cache_path);
    }

    // UMA fallback (no cache available): use conservative defaults that skip
    // the expensive GPU benchmark. Only override settings where 32t is safe.
    // Note: Q6_K 32t uses per-element scalar access — slower for large K on
    // some UMA iGPUs. Leave q6k_use_32=false (use 256t cooperative variant).
    if (is_uma) {
        q5_0_use_256  = false;   // 32t wave-only (proven better on all UMA)
        q8_0_use_256  = false;   // 32t wave-only (proven better on all UMA)
        q6k_use_32    = false;   // keep 256t cooperative (faster for large K)
        f16_use_load4 = false;   // load2 sufficient for shared-memory bandwidth
        q5k_use_mr    = true;    // multi-row Q5_K: 14% faster than standard 256t
        q6k_use_mr    = false;   // disabled: q6k_mr shader 2.1x slower than standard 256t
        fa_use_uma    = true;    // UMA FA: 128t, fewer barriers, no idle threads (D<=128)
        // Per-flag env overrides for UMA bisection.
        // GGML_DX12_Q5_0_USE_256=1 forces 256t cooperative shader for Q5_0
        // GGML_DX12_Q8_0_USE_256=1 forces 256t cooperative shader for Q8_0
        // GGML_DX12_Q5K_USE_MR=0   disables multi-row Q5_K
        // GGML_DX12_FA_USE_UMA=0   disables UMA FA shader (use standard FA)
        if (const char * e = getenv("GGML_DX12_Q5_0_USE_256")) q5_0_use_256 = (atoi(dx12_env_unquote(e)) != 0);
        if (const char * e = getenv("GGML_DX12_Q8_0_USE_256")) q8_0_use_256 = (atoi(dx12_env_unquote(e)) != 0);
        if (const char * e = getenv("GGML_DX12_Q5K_USE_MR"))   q5k_use_mr   = (atoi(dx12_env_unquote(e)) != 0);
        if (const char * e = getenv("GGML_DX12_FA_USE_UMA"))   fa_use_uma   = (atoi(dx12_env_unquote(e)) != 0);
        DX12_LOG_INFO("Auto-tune v%d UMA: Q5_0=%s Q8_0=%s Q6_K=256t Q5_K=%s FA=%s F16=load2\n",
                      TUNE_VERSION,
                      q5_0_use_256 ? "256t" : "32t",
                      q8_0_use_256 ? "256t" : "32t",
                      q5k_use_mr   ? "mr"   : "std",
                      fa_use_uma   ? "uma"  : "std");
        // Allow forcing the benchmark on UMA via GGML_DX12_FORCE_AUTOTUNE=1
        if (!getenv("GGML_DX12_FORCE_AUTOTUNE")) return;
        DX12_LOG_INFO("GGML_DX12_FORCE_AUTOTUNE set -- running benchmark on UMA\n");
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

    // GPU TIMESTAMP queries on the compute command list. Pattern matches
    // Xbox-GDK-Samples/Kits/ATGTK/PerformanceTimers (and is verified on a
    // GDKX compute queue via the FastBlockCompress sample, where the same
    // GPUTimer is bound to a D3D12_COMMAND_LIST_TYPE_COMPUTE queue and
    // calls EndQuery(TIMESTAMP) on the compute CL).
    //
    // Wall-clock timing remains the fallback if heap creation, the
    // GetTimestampFrequency call, or the post-execution readback fails on
    // some specific driver -- bench_pipeline checks ts_ok per call.
    ComPtr<ID3D12QueryHeap> ts_heap;
    ComPtr<ID3D12Resource>  ts_readback;
    uint64_t                ts_freq = 0;
    bool                    ts_ok   = false;
    {
        HRESULT hr = compute_queue->GetTimestampFrequency(&ts_freq);
        if (SUCCEEDED(hr) && ts_freq != 0) {
            D3D12_QUERY_HEAP_DESC qhd = {};
            qhd.Type  = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            qhd.Count = 4;  // start + end for 2 variants (slots 0/1 and 2/3)
            hr = device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&ts_heap));
        }
        if (SUCCEEDED(hr) && ts_heap) {
            D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width              = 4 * sizeof(uint64_t);
            rd.Height             = 1;
            rd.DepthOrArraySize   = 1;
            rd.MipLevels          = 1;
            rd.Format             = DXGI_FORMAT_UNKNOWN;
            rd.SampleDesc.Count   = 1;
            rd.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            hr = device->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_COMMON, nullptr,
                IID_PPV_ARGS(&ts_readback));
            if (SUCCEEDED(hr)) {
                ts_ok = true;
                DX12_LOG_INFO("Auto-tune: GPU timestamp queries enabled (freq=%llu Hz)\n",
                              (unsigned long long)ts_freq);
            }
        }
        if (!ts_ok) {
            DX12_LOG_INFO("Auto-tune: GPU timestamp queries unavailable -- falling back to wall-clock timing\n");
            ts_heap.Reset();
            ts_readback.Reset();
        }
    }

    // Helper: benchmark a pipeline variant
    // Returns GPU time in ticks, or UINT64_MAX on failure
    auto bench_pipeline = [&](dx12_pipeline_key key, uint32_t K, uint32_t N, uint32_t ts_start) -> uint64_t {
        DX12_LOG_INFO("  bench: op=%d src0=%d flags=%u K=%u N=%u\n",
                      key.op, key.src0_type, key.flags, K, N);
        fflush(stderr);
        dx12_pipeline * pl = get_or_create_pipeline(key);
        if (!pl || !pl->pso) {
            DX12_LOG_WARN("  bench: no PSO!\n");
            return UINT64_MAX;
        }

        // Create command allocator + list for benchmarking
        ComPtr<ID3D12CommandAllocator> alloc;
        ComPtr<ID3D12GraphicsCommandList> cl;
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));

        auto submit_and_wait = [&]() {
            cl->Close();
            ID3D12CommandList * lists[] = { cl.Get() };
            compute_queue->ExecuteCommandLists(1, lists);
            ComPtr<ID3D12Fence> f;
            device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f));
            compute_queue->Signal(f.Get(), 1);
            HANDLE ev = dx12_create_event();
            f->SetEventOnCompletion(1, ev);
#ifdef _WIN32
            DWORD wr = WaitForSingleObject(ev, 5000);
            dx12_close_event(ev);
            if (wr == WAIT_TIMEOUT) { DX12_LOG_WARN("  bench: GPU timeout!\n"); return false; }
#else
            auto t0 = std::chrono::steady_clock::now();
            while (f->GetCompletedValue() < 1) {
                if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(5000)) {
                    DX12_LOG_WARN("  bench: GPU timeout!\n"); dx12_close_event(ev); return false;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            dx12_close_event(ev);
#endif
            return true;
        };

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

        // Warmup dispatch + GPU sync
        cl->Dispatch(N, 1, 1);
        D3D12_RESOURCE_BARRIER barrier = {}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        cl->ResourceBarrier(1, &barrier);
        if (!submit_and_wait()) return UINT64_MAX;

        // Timed dispatches. Prefer GPU TIMESTAMP queries on the same compute
        // CL (pattern from Xbox-GDK-Samples ATGTK PerformanceTimers, also used
        // by the GGML_DX12_PERF mode in this file). Wall-clock timing is the
        // fallback if ts_ok is false or the readback returns a bad range.
        alloc->Reset();
        cl->Reset(alloc.Get(), nullptr);
        cl->SetComputeRootSignature(common_root_sig.Get());
        cl->SetPipelineState(pl->pso.Get());
        D3D12_GPU_VIRTUAL_ADDRESS va2 = bench_buf->GetGPUVirtualAddress();
        cl->SetComputeRootShaderResourceView(1, va2);
        cl->SetComputeRootShaderResourceView(2, va2);
        cl->SetComputeRootUnorderedAccessView(3, va2);
        cl->SetComputeRootShaderResourceView(4, va2);
        cl->SetComputeRootShaderResourceView(5, va2);
        cl->SetComputeRoot32BitConstants(0, (uint32_t)(sizeof(params)/4), &params, 0);

        if (ts_ok) {
            cl->EndQuery(ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts_start);
        }
        for (int rep = 0; rep < 10; rep++) {
            cl->Dispatch(N, 1, 1);
            cl->ResourceBarrier(1, &barrier);
        }
        if (ts_ok) {
            cl->EndQuery(ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts_start + 1);
            cl->ResolveQueryData(
                ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                ts_start, 2,
                ts_readback.Get(), ts_start * sizeof(uint64_t));
        }
#ifdef _WIN32
        LARGE_INTEGER wc0, wc1, wcf;
        QueryPerformanceFrequency(&wcf);
        QueryPerformanceCounter(&wc0);
#else
        auto wc0 = std::chrono::steady_clock::now();
#endif
        if (!submit_and_wait()) return UINT64_MAX;

        // Prefer GPU timestamps -- only count the time the GPU spent on the
        // 10 dispatches, not submission/wait overhead.
        if (ts_ok) {
            D3D12_RANGE rr = { ts_start * sizeof(uint64_t),
                               (ts_start + 2) * sizeof(uint64_t) };
            void * mapped = nullptr;
            HRESULT hrm = ts_readback->Map(0, &rr, &mapped);
            if (SUCCEEDED(hrm) && mapped) {
                const uint64_t * timings = (const uint64_t *)mapped;
                uint64_t t0t = timings[ts_start];
                uint64_t t1t = timings[ts_start + 1];
                D3D12_RANGE wr = { 0, 0 };
                ts_readback->Unmap(0, &wr);
                if (t1t > t0t) {
                    uint64_t dt_us = (uint64_t)((double)(t1t - t0t) * 1e6 / (double)ts_freq);
                    DX12_LOG_INFO("    => %llu us (gpu)\n", (unsigned long long)dt_us);
                    return dt_us;
                }
            }
            // Bad readback -- fall through to wall-clock
        }
#ifdef _WIN32
        QueryPerformanceCounter(&wc1);
        uint64_t dt = (uint64_t)((double)(wc1.QuadPart - wc0.QuadPart) / wcf.QuadPart * 1e6);
#else
        auto wc1 = std::chrono::steady_clock::now();
        uint64_t dt = std::chrono::duration_cast<std::chrono::microseconds>(wc1 - wc0).count();
#endif
        DX12_LOG_INFO("    => %llu us (wall)\n", (unsigned long long)dt);
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
        DX12_LOG_INFO("Auto-tune cache written to '%s'\n", cache_path);
    }
}

static const char * dx12_backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_DX12_NAME;
}

static void dx12_backend_free(ggml_backend_t backend) {
    auto * ctx = (dx12_backend_context *)backend->context;
    dx12_unregister_bctx(ctx);
    delete ctx; // RAII destructor handles fence wait + event close + ts drain
}

static void dx12_backend_synchronize(ggml_backend_t backend) {
    auto * ctx = (dx12_backend_context *)backend->context;

    // Submit pending work, then wait for GPU completion. Must wait here:
    // some buffer paths (notably the CUSTOM-heap zero-copy I/O on systems
    // where it is enabled) read/write tensor data via Map() with no fence
    // wait of their own, and depend on synchronize() having quiesced the
    // device. The non-zero-copy staging paths queue an xfer copy on the
    // same compute_queue and effectively wait via the xfer fence, so the
    // extra wait here is harmless for them (the queue is already drained).
    if (ctx->cmd_list_open) {
        ctx->close_and_execute();
    }
    ctx->wait_for_gpu();
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
#if defined(_WIN32) && !defined(GGML_DX12_XBOX_GDKX)
    // Query fresh memory info from DXGI to reflect current allocations
    ComPtr<IDXGIAdapter3> adapter3;
    if (SUCCEEDED(d->adapter.As(&adapter3))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO mem_info = {};
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mem_info))
            && mem_info.Budget > 0) {
            if (total) *total = (size_t)mem_info.Budget;
            if (free)  *free  = mem_info.Budget > mem_info.CurrentUsage
                              ? (size_t)(mem_info.Budget - mem_info.CurrentUsage) : 0;
            return;
        }
    }
#endif
    // Fallback to cached values (always used on WSL2 and Xbox)
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
    dx12_register_bctx(ctx);
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

static void * dx12_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_cpu_print_tensor_op_perf") == 0) {
        return (void *)ggml_dx12_print_tensor_op_perf;
    }
    return nullptr;
}

static const ggml_backend_reg_i dx12_reg_interface = {
    /* .get_name         = */ dx12_reg_get_name,
    /* .get_device_count = */ dx12_reg_get_device_count,
    /* .get_device       = */ dx12_reg_get_device,
    /* .get_proc_address = */ dx12_reg_get_proc_address,
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
    { GGML_OP_SET,           { g_set_dxil,           sizeof(g_set_dxil)           } },
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
    } else if ((key.op == GGML_OP_CPY || key.op == GGML_OP_CONT || key.op == GGML_OP_DUP)
               && key.flags == 1) {
        // Fast path: contiguous same-dtype copy (uint4-vectorized, no atomic CAS)
        static const dx12_shader_blob cpy_contig_blob = { g_cpy_contig_dxil, sizeof(g_cpy_contig_dxil) };
        blob = &cpy_contig_blob;
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
    } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags == 1) {
        // Split-KV reduce shader
        static const dx12_shader_blob reduce_blob = { g_flash_attn_reduce_dxil, sizeof(g_flash_attn_reduce_dxil) };
        blob = &reduce_blob;
    } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags == 2) {
        // UMA-optimized FA (GROUP_SIZE=128, TILE_KV=128, D≤128)
        static const dx12_shader_blob uma_blob = { g_flash_attn_uma_dxil, sizeof(g_flash_attn_uma_dxil) };
        blob = &uma_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 4) {
        // Register-blocked tiled batch MUL_MAT (M > 1)
        if (key.src0_type == GGML_TYPE_Q4_K) {
            static const dx12_shader_blob wmma_q4k_blob     = { g_mul_mat_q4k_wmma_dxil,     sizeof(g_mul_mat_q4k_wmma_dxil) };
#ifdef GGML_DX12_HAVE_F16_SHADERS
            static const dx12_shader_blob wmma_q4k_f16_blob = { g_mul_mat_q4k_wmma_f16_dxil, sizeof(g_mul_mat_q4k_wmma_f16_dxil) };
            blob = use_f16_shaders ? &wmma_q4k_f16_blob : &wmma_q4k_blob;
#else
            blob = &wmma_q4k_blob;
#endif
        } else if (key.src0_type == GGML_TYPE_Q5_K) {
            static const dx12_shader_blob wmma_q5k_blob     = { g_mul_mat_q5k_wmma_dxil,     sizeof(g_mul_mat_q5k_wmma_dxil) };
#ifdef GGML_DX12_HAVE_F16_SHADERS
            static const dx12_shader_blob wmma_q5k_f16_blob = { g_mul_mat_q5k_wmma_f16_dxil, sizeof(g_mul_mat_q5k_wmma_f16_dxil) };
            blob = use_f16_shaders ? &wmma_q5k_f16_blob : &wmma_q5k_blob;
#else
            blob = &wmma_q5k_blob;
#endif
        } else if (key.src0_type == GGML_TYPE_Q6_K) {
            static const dx12_shader_blob wmma_q6k_blob     = { g_mul_mat_q6k_wmma_dxil,     sizeof(g_mul_mat_q6k_wmma_dxil) };
#ifdef GGML_DX12_HAVE_F16_SHADERS
            static const dx12_shader_blob wmma_q6k_f16_blob = { g_mul_mat_q6k_wmma_f16_dxil, sizeof(g_mul_mat_q6k_wmma_f16_dxil) };
            blob = use_f16_shaders ? &wmma_q6k_f16_blob : &wmma_q6k_blob;
#else
            blob = &wmma_q6k_blob;
#endif
        } else if (key.src0_type == GGML_TYPE_Q5_0) {
            static const dx12_shader_blob wmma_q50_blob     = { g_mul_mat_q5_0_wmma_dxil,     sizeof(g_mul_mat_q5_0_wmma_dxil) };
#ifdef GGML_DX12_HAVE_F16_SHADERS
            static const dx12_shader_blob wmma_q50_f16_blob = { g_mul_mat_q5_0_wmma_f16_dxil, sizeof(g_mul_mat_q5_0_wmma_f16_dxil) };
            blob = use_f16_shaders ? &wmma_q50_f16_blob : &wmma_q50_blob;
#else
            blob = &wmma_q50_blob;
#endif
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
        } else if (key.src0_type == GGML_TYPE_Q5_K) {
            static const dx12_shader_blob mr_q5k_blob = { g_mul_mat_vec_q5k_mr_dxil, sizeof(g_mul_mat_vec_q5k_mr_dxil) };
            blob = &mr_q5k_blob;
        } else if (key.src0_type == GGML_TYPE_Q6_K) {
            static const dx12_shader_blob mr_q6k_blob = { g_mul_mat_vec_q6k_mr_dxil, sizeof(g_mul_mat_vec_q6k_mr_dxil) };
            blob = &mr_q6k_blob;
        }
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 6) {
        // 32-thread wave-only Q4_K matvec (UMA optimized, no barriers)
        if (key.src0_type == GGML_TYPE_Q4_K) {
            static const dx12_shader_blob mv_q4k_32_blob = { g_mul_mat_vec_q4k_32_dxil, sizeof(g_mul_mat_vec_q4k_32_dxil) };
            blob = &mv_q4k_32_blob;
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
