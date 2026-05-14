// ggml-dx12.cpp - DirectX 12 backend for ggml
//
// Implements a GPU compute backend using D3D12, with optional Cooperative Vector
// acceleration for matrix-vector operations (SM 6.9 / Agility SDK 1.717+).

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
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

// Sentinel base address for non-host-accessible GPU buffers (matches Vulkan approach).
// Used only when a DX12 buffer has no CPU-mapped backing (DEFAULT heap, dGPU).
// On UMA Intel iGPUs we instead persistently map the resource and get_base()
// returns the real CPU pointer so CPU-fallback ops can read/write the same
// memory the GPU sees.  dx12_tensor_offset() therefore must compute the offset
// relative to whatever base the owning buffer reported.
static void * const DX12_PTR_BASE = (void *)(uintptr_t)0x1000;

struct dx12_buffer_context; // forward decl; defined below

static uint64_t dx12_tensor_offset(const struct ggml_tensor * tensor);

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
// Per-node decision cache ("graph replay")
// ---------------------------------------------------------------------------
// Stable across decode tokens once a graph reaches steady state.  Each cgraph
// node maps to one decision; the decision captures the result of the per-node
// "decision" block (pipeline lookup, fusion lookahead, route flags) so the
// fast path can skip ~250 lines of branching.  Per-token state (tensor
// pointers, offsets, root constants, group counts that depend on N_kv,
// barrier dependency tracking) is still recomputed every call.
//
// Modeled after CUDA's ggml_cuda_graph_update_required (ggml-cuda.cu:3135-3175):
// capture per-node identity bytes, memcmp at the start of every graph compute,
// invalidate-and-rebuild on mismatch.

enum dx12_decision_kind : uint8_t {
    DX12_DEC_SKIP         = 0,  // view/reshape/permute/transpose — alias propagation only
    DX12_DEC_NO_PIPELINE  = 1,  // pipeline missing or build failed — silently skip
    DX12_DEC_COMPUTE      = 2,  // real dispatch (possibly fused)
};

enum dx12_fusion_kind : uint8_t {
    DX12_FUSE_NONE             = 0,
    DX12_FUSE_ADD_RMS_MUL      = 1,  // ADD + RMS_NORM + MUL  (skip 2)
    DX12_FUSE_RMS_MUL          = 2,  // RMS_NORM + MUL        (skip 1)
    DX12_FUSE_RMS_MUL_ROPE3    = 3,  // RMS_NORM + MUL + ROPE (skip 2)
    DX12_FUSE_RMS_MUL_ROPE5    = 4,  // + VIEW + SET_ROWS     (skip 4)
    DX12_FUSE_ROPE_SET_ROWS    = 5,  // ROPE + VIEW + SET_ROWS (skip 2)
    DX12_FUSE_MMV_GLU_SPLIT    = 6,  // MUL_MAT(up) + MUL_MAT(gate) + SWIGLU split (skip 2)
};

// Per-node identity used for cache invalidation.  Layout-stable across tokens
// once a graph reaches steady state — RoPE positions live in src[1] (not
// op_params), KV-cache row indices live in tensor data (not shape).  The
// fields here cover everything the decision block branches on.
struct dx12_node_identity {
    uint8_t  op;             // ggml_op fits in a byte
    uint8_t  dst_type;       // ggml_type
    uint8_t  src0_type;
    uint8_t  src1_type;
    uint8_t  src2_type;
    uint8_t  src3_type;
    uint8_t  has_src2;
    uint8_t  has_src3;
    int64_t  src0_ne0;       // K dimension — drives MUL_MAT key.flags thresholds
    int64_t  src0_ne2;       // n_heads     — drives FA gqa_ratio
    int64_t  src1_ne0;
    int64_t  src1_ne2;       // n_kv_heads  — drives FA gqa_ratio
    int64_t  dst_ne1;        // M dimension — matvec vs batch routing
    int64_t  dst_ne0;        // output width — drives matvec_row_groups
    int32_t  op_params[12];  // 48 bytes; covers RoPE mode/ext_factor/attn_factor, FA flags, ROPE sections, etc.
    uint8_t  src0_pad[7];    // explicit zero-pad so memcmp is deterministic
};
static_assert(sizeof(dx12_node_identity) % 8 == 0, "identity must be 8-byte aligned for safe memcmp");

// Per-node cached decision.  Populated on the first graph compute (or after
// invalidation) and reused as long as identity matches.
struct dx12_node_decision {
    dx12_node_identity identity;

    dx12_decision_kind kind;
    dx12_fusion_kind   fusion_kind;
    uint8_t            skip_count;            // nodes to advance past after this dispatch (fusion)
    uint8_t            key_flags;             // dx12_pipeline_key.flags (matvec route, etc.)

    bool               is_matvec_dispatch;
    bool               use_dp4a;
    bool               use_dp4a_matvec;
    bool               needs_op_params;
    bool               conservative_barrier;  // SET_ROWS / FA / fused_rope_set_rows
    bool               has_bias_add;          // matvec fused with following ADD bias

    dx12_pipeline *    pipeline;
};

struct dx12_replay_cache {
    std::vector<dx12_node_decision> decisions;

    // Diagnostics
    uint64_t hits     = 0;
    uint64_t misses   = 0;
    uint64_t rebuilds = 0;
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

struct dx12_shader_blob {
    const void * data;
    size_t       size;
};

// ---------------------------------------------------------------------------
// Device — represents one D3D12 adapter + device
// ---------------------------------------------------------------------------

struct dx12_device {
    ComPtr<IDXGIAdapter1>     adapter;
    ComPtr<ID3D12Device>      device;
    ComPtr<ID3D12CommandQueue> compute_queue;

    DXGI_ADAPTER_DESC1        adapter_desc = {};
    size_t                    vram_total   = 0;
    size_t                    vram_free    = 0;

    bool cooperative_vector_supported = false;

    // WaveMMA (SM 6.9 Wave Matrix) support
    bool wave_mma_supported = false;
    uint32_t wave_mma_K      = 0;     // hardware K dimension (even multiple of 16)
    uint32_t wave_mma_M      = 0;     // M dimension (16 or 64)
    uint32_t wave_mma_N      = 0;     // N dimension (16 or 64)
    uint32_t wave_mma_wave_size = 0;  // required wave size for WaveMMA
    bool     wave_mma_f16_acc32 = false; // F16 input with F32 accumulator

    // dp4a (integer dot product) support — SM 6.4+
    bool dp4a_supported = false;

    // Native 16-bit shader operations (half / float16_t / int16_t) — D3D12_OPTIONS4.
    // Required to consume `_fp16_dxil` shader variants compiled with
    // -enable-16bit-types. Otherwise we fall back to the FP32 blob.
    bool fp16_supported = false;

    // GPU wave (warp/subgroup) size — detected at init, used for shader variant selection
    uint32_t wave_size = 32;

    // Memory architecture detection (for ReBAR / UMA fast-paths).
    // Memory architecture detection (UMA fast-path for set_tensor on iGPU).
    // Mirrors Vulkan's ggml_vk_create_buffer_device strategy at ggml-vulkan.cpp:2800-2835.
    //   - is_uma: integrated GPU, all memory is host-shared.  On Intel iGPUs
    //     we allocate as CUSTOM heap with WRITE_BACK + L0 for direct memcpy.
    //     AMD RDNA iGPUs skip this (CUSTOM L0 regresses GPU reads 25-30%).
    // ReBAR's "DEVICE_LOCAL | HOST_VISIBLE" pattern is INTENTIONALLY NOT
    // implemented for dGPU because D3D12 doesn't expose the equivalent flag
    // combination -- see detect_memory_architecture() for the analysis.
    // Override via DX12_NO_UMA=1 to force the staging path.
    bool is_uma = false;
    void detect_memory_architecture();

    // Hot-path pipeline pointer caches.  Both `quantize_q8_1` (flags=99) and
    // `flash_attn_split_k_reduce` (flags=8 with op=FLASH_ATTN_EXT) keys are
    // compile-time constants but were previously looked up via
    // get_or_create_pipeline (mutex + unordered_map::find) on every dp4a /
    // every split-KV FA dispatch -- and the lookup also clobbered the device's
    // last_pipeline_key fast-path so the NEXT main-pipeline lookup also went
    // through the mutex.  Caching these directly here eliminates ~60-80 mutex
    // acquisitions per token on dp4a models.
    dx12_pipeline * quantize_q8_1_pipeline = nullptr;
    dx12_pipeline * flash_attn_reduce_pipeline = nullptr;

    // Per-device shader blob maps — populated at init from wave-size-specific compiled variants
    std::unordered_map<int, dx12_shader_blob> shader_blobs;
    std::unordered_map<int, dx12_shader_blob> unary_shader_blobs;
    void init_shader_blobs();

    // Auto-tuning: optimal shader variants per quant type
    // Determined by GPU microbenchmark at first model load.
    // Bump TUNE_VERSION when adding/removing dimensions to invalidate cache.
    //
    // History:
    //  v5 -> v6: removed 5 dead dimensions (q5_0/q8_0/q6k/q5k/f16_load4) whose
    //            autotune results were silently overwritten in dispatch.
    //  v6 -> v7: added f16_mr_use_256 (F16/F32 matvec: 256-thread mr vs
    //            32-thread mr32; was previously hardcoded by wave_size>=64).
    //  v7 -> v8: added f16_mr_k_256_threshold (K-aware F16 mr selection — on
    //            B390 the 256t variant beats 32t at large K but loses at
    //            small K; pick per-dispatch based on src0->ne[0]).
    //  v8 -> v9: extended autotune K coverage (added K=8192) so the K-aware
    //            crossover sees Phi-3 FFN-down shapes; crossover now uses
    //            test_K[0] / test_K[NK-1] instead of fixed [0]/[1].
    static constexpr int TUNE_VERSION = 9;
    bool tuning_done = false;
    bool q4k_dp4a_use_32 = false; // Q4_K dp4a matvec: true=32 threads, false=256 threads (default=256)
    bool q5k_dp4a_use_32 = false; // Q5_K dp4a matvec: true=32 threads, false=256 threads (default=256)
    bool f16_mr_use_256  = false; // F16/F32 matvec:   true=256 threads (mr), false=32 threads (mr32)
    // K-aware F16 mr selection: when K (src0->ne[0]) >= this threshold use the
    // 256-thread mr variant, otherwise use the 32-thread mr32 variant. Set to
    // UINT32_MAX to always use mr32 (the historical default for non-AMD-wave64
    // devices) and 0 to always use mr256 (matches f16_mr_use_256=true).
    uint32_t f16_mr_k_256_threshold = 0xFFFFFFFFu;

    void run_autotune();

    // Pipeline cache
    std::mutex pipeline_mutex;
    std::unordered_map<dx12_pipeline_key, dx12_pipeline, dx12_pipeline_key_hash> pipeline_cache;

    // Fast-path: skip mutex + map lookup when consecutive nodes use the same pipeline
    dx12_pipeline_key  last_pipeline_key = {};
    dx12_pipeline *    last_pipeline_ptr = nullptr;

    // Common root signature for most shaders
    ComPtr<ID3D12RootSignature> common_root_sig;

    // Split-KV temp buffer for flash attention (1 MB, lazily created)
    ComPtr<ID3D12Resource> splitkv_temp;
    static constexpr size_t SPLITKV_TEMP_SIZE = 1024 * 1024; // 1 MB

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
        void *                           upload_mapped = nullptr;  // persistent map ptr
        ComPtr<ID3D12Resource>           readback_staging;
        size_t                           readback_size = 0;
        void *                           readback_mapped = nullptr; // persistent map ptr
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
        xfer.fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        xfer.initialized = true;
    }

    void xfer_wait() {
        if (xfer.fence_value == 0) return;
        if (xfer.fence->GetCompletedValue() >= xfer.fence_value) return;
        xfer.fence->SetEventOnCompletion(xfer.fence_value, xfer.fence_event);
        WaitForSingleObject(xfer.fence_event, INFINITE);
    }

    void xfer_ensure_staging(size_t up_size, size_t rb_size) {
        // Helper: allocate-or-grow.  Optionally persistently maps when ht ==
        // UPLOAD (write-combined memory, no cache invalidation needed).
        // For READBACK heaps we DO NOT persistently map: D3D12_HEAP_TYPE_READBACK
        // uses CPU_PAGE_PROPERTY_WRITE_BACK which is cached, and Map() with
        // a non-empty read_range is the documented cache-invalidation point.
        // Persistent mapping would require manual cache flushes that D3D12
        // doesn't expose.
        auto make_buf = [&](ComPtr<ID3D12Resource> & res, size_t & cur, void *& mapped, size_t need, D3D12_HEAP_TYPE ht) {
            if (cur >= need) return;
            need = (need + 0xFFFF) & ~(size_t)0xFFFF;
            if (mapped && res) {
                D3D12_RANGE wr = { 0, 0 };
                res->Unmap(0, &wr);
                mapped = nullptr;
            }
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
            // Persistent map only for UPLOAD (write-combined; no cache issues).
            if (ht == D3D12_HEAP_TYPE_UPLOAD) {
                D3D12_RANGE no_read = { 0, 0 };
                hr = res->Map(0, &no_read, &mapped);
                DX12_CHECK(hr, "Map upload staging (persistent)");
            }
        };
        if (up_size > 0) make_buf(xfer.upload_staging, xfer.upload_size, xfer.upload_mapped, up_size, D3D12_HEAP_TYPE_UPLOAD);
        if (rb_size > 0) make_buf(xfer.readback_staging, xfer.readback_size, xfer.readback_mapped, rb_size, D3D12_HEAP_TYPE_READBACK);
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
            CloseHandle(xfer.fence_event);
        }
        // Unmap persistently-mapped staging before ComPtr releases the resource.
        // D3D12 will assert in debug if a resource is released while still mapped.
        if (xfer.upload_mapped && xfer.upload_staging) {
            D3D12_RANGE wr = { 0, 0 };
            xfer.upload_staging->Unmap(0, &wr);
            xfer.upload_mapped = nullptr;
        }
    }

    void init(ComPtr<IDXGIAdapter1> adapter_, size_t idx);
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

// Compute byte offset of `tensor` within its DX12 buffer.  When the buffer is
// persistently mapped (UMA host-visible path), get_base() returns the real
// mapped pointer; otherwise it returns the DX12_PTR_BASE sentinel.  Either way
// the offset is (tensor->data - buffer_base), and ggml's allocator computed
// tensor->data as base + offset.
static inline uint64_t dx12_tensor_offset(const struct ggml_tensor * tensor) {
    auto * ctx = (dx12_buffer_context *)tensor->buffer->context;
    void * base = (ctx && ctx->mapped) ? ctx->mapped : DX12_PTR_BASE;
    return (uint8_t *)tensor->data - (uint8_t *)base;
}

// ---------------------------------------------------------------------------
// Backend context (stream)
// ---------------------------------------------------------------------------

static const int CMD_RING_SIZE = 4;

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

    // "Almost-ready" fence: signaled partway through graph compute so the
    // CPU can overlap readback prep with the GPU's tail dispatches.
    uint64_t                         almost_ready_fence = 0;

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
    D3D12_GPU_VIRTUAL_ADDRESS  last_src4_va  = 0;
    D3D12_GPU_VIRTUAL_ADDRESS  last_src5_va  = 0;
    D3D12_GPU_VIRTUAL_ADDRESS  last_src6_va  = 0;

    // Scratch buffer for dp4a Q8_1 quantized input
    ComPtr<ID3D12Resource> q8_1_scratch;
    size_t                 q8_1_scratch_size = 0;
    // Old q8_1_scratch buffers retained until end of current graph_compute,
    // because the open cmd_list still has dispatches recorded with their VAs.
    std::vector<ComPtr<ID3D12Resource>> q8_1_scratch_retired;

    // Quantize-dispatch caching: track the last src1 (input activation) tensor
    // we quantized into q8_1_scratch.  When consecutive MUL_MAT dispatches share
    // the same src1 (e.g. Q/K/V projections all reading the post-RMS_NORM_MUL
    // output, or gate/up reading the post-attn-norm output), we can skip the
    // redundant quantize+barrier pair before each subsequent matmul.  Cache is
    // invalidated on cmd-list reset and on any graph node that writes to the
    // cached src1 tensor.
    D3D12_GPU_VIRTUAL_ADDRESS last_q8_1_src_va   = 0;
    uint32_t                  last_q8_1_src_off  = 0;
    uint32_t                  last_q8_1_size     = 0;
    uintptr_t                 last_q8_1_src_id   = 0;

    // Cross-frame adaptive submit threshold (Vulkan ggml-vulkan.cpp:14512-14524).
    // The previous graph_compute records its total MUL_MAT/MUL_MAT_ID weight bytes;
    // the next call uses last/40 as the per-submit byte threshold (clamped to 100MB).
    // This kicks the GPU early on dense models (few large matmuls) and avoids
    // over-submitting on sparse models (many tiny ops).
    uint64_t last_total_mul_mat_bytes = 0;

    // Deferred memcpy queue for get_tensor_async (Vulkan parity:
    // ggml-vulkan.cpp:13890 deferred_memcpy + out_memcpys).  Async readback
    // records a CopyBufferRegion to the readback staging buffer (or, for UMA,
    // captures the source mapped pointer directly), then registers a (dst,
    // src, size) entry here.  At synchronize() — after wait_for_fence — we
    // execute the queued memcpys to deliver the data to the caller's buffer.
    // This lets multiple get_tensor_async calls amortize a single fence wait
    // and pipeline tile-by-tile readbacks (e.g. mtmd vision encoder loops).
    struct deferred_memcpy_t {
        void *          dst;
        const uint8_t * src;     // either readback_staging map ptr + offset, or UMA mapped + offset
        size_t          size;
        ComPtr<ID3D12Resource> staging;  // keeps readback staging alive until flushed
    };
    std::vector<deferred_memcpy_t> pending_get_memcpys;

    // Per-async-call readback staging ring.  Distinct from the device-level
    // xfer.readback_staging used by sync get_tensor — that path Map/Unmaps per
    // call and only supports one in-flight transfer.  For async we may have
    // multiple readbacks queued before synchronize, so each get_tensor_async
    // allocates (or reuses) a staging buffer that lives in pending_get_memcpys
    // until flushed.  Keep a small free-list to avoid repeated allocation.
    std::vector<ComPtr<ID3D12Resource>> async_readback_pool;

    // Persistent dependency-tracking set used during graph_compute.  Promoted
    // from a per-call local because the default-constructed unordered_set
    // allocates its bucket array on the heap (one malloc/free per token);
    // reusing the bucket array across tokens eliminates ~200-600ns of heap
    // overhead per token plus avoids occasional rehash-induced reallocations.
    std::unordered_set<uintptr_t> unsynced_writes;

    // R1 — per-graph cached decisions ("graph replay").  Persistent across
    // calls; invalidated by per-node identity memcmp at the start of every
    // graph_compute.  Skips pipeline lookup, fusion lookahead, and key
    // construction on cache hits — typically 95%+ of decode tokens after
    // warmup.  Disable via DX12_NO_GRAPH_REPLAY=1.
    dx12_replay_cache replay_cache;

    void reset_binding_cache() {
        last_pso      = nullptr;
        last_root_sig = nullptr;
        last_src0_va  = 0;
        last_src1_va  = 0;
        last_dst_va   = 0;
        last_src2_va  = 0;
        last_src3_va  = 0;
        last_src4_va  = 0;
        last_src5_va  = 0;
        last_src6_va  = 0;
        last_q8_1_src_va  = 0;
        last_q8_1_src_off = 0;
        last_q8_1_size    = 0;
        last_q8_1_src_id  = 0;
    }

    ~dx12_backend_context() {
        // RAII cleanup: wait for ALL GPU work and close event handle
        if (fence && fence_event && dev) {
            wait_for_gpu();
        }
        if (fence_event) {
            CloseHandle(fence_event);
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

struct dx12_globals_t {
    bool                                        initialized = false;
    ComPtr<IDXGIFactory4>                       factory;
    std::vector<std::unique_ptr<dx12_device>>   devices;
    std::mutex                                  init_mutex;

    // Backend device & registry objects
    std::vector<ggml_backend_device> backend_devices;
    ggml_backend_reg               backend_reg_obj = {};
};

// Heap-allocate the globals and intentionally leak at process exit.
// Static destruction order is unsafe on Windows/D3D12: by the time the
// dtor for a file-scope object runs, the Intel UMD (igd12um64xe3.dll) may
// already be partially unloaded, causing ComPtr Release() calls to fault
// with STATUS_STACK_BUFFER_OVERRUN (0xC0000409). The OS reclaims handles
// and GPU resources at process exit regardless, so leaking is safe.
static dx12_globals_t & g_dx12 = *(new dx12_globals_t());

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

    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&g_dx12.factory));
    DX12_CHECK(hr, "CreateDXGIFactory1");

    // Enumerate adapters
    for (UINT i = 0; ; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        hr = g_dx12.factory->EnumAdapters1(i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        DX12_CHECK(hr, "EnumAdapters1");

        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        // Skip software adapters
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

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
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf, sizeof(name_buf), nullptr, nullptr);
                DX12_LOG_WARN("Skipping %s: UAV allocation failed (HRESULT 0x%08X)\n", name_buf, (unsigned)hr);
                continue;
            }
        }
        test_device.Reset();

        if (g_dx12.devices.size() >= GGML_DX12_MAX_DEVICES) break;

        g_dx12.devices.push_back(std::make_unique<dx12_device>());
        g_dx12.devices.back()->init(std::move(adapter), g_dx12.devices.size() - 1);
    }

    DX12_LOG_INFO("Found %zu D3D12 device(s)\n", g_dx12.devices.size());
    g_dx12.initialized = true;
}

// ---------------------------------------------------------------------------
// dx12_device implementation
// ---------------------------------------------------------------------------

void dx12_device::init(ComPtr<IDXGIAdapter1> adapter_, size_t idx) {
    adapter   = std::move(adapter_);
    dev_index = idx;

    adapter->GetDesc1(&adapter_desc);

    // Convert wide name to narrow
    char narrow[256];
    WideCharToMultiByte(CP_UTF8, 0, adapter_desc.Description, -1, narrow, sizeof(narrow), nullptr, nullptr);
    description = narrow;
    name = std::string(GGML_DX12_NAME) + std::to_string(idx);

    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    DX12_CHECK(hr, "D3D12CreateDevice");

    // Create compute command queue
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type     = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    qd.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&compute_queue));
    DX12_CHECK(hr, "CreateCommandQueue(compute)");

    // VRAM: use DXGI budget for accuracy.
    // For iGPUs (small dedicated VRAM), also include the non-local (shared system RAM)
    // segment. The OS-managed DXGI budget caps this sensibly (~50% of physical RAM),
    // preventing over-reporting. For dGPUs, only use the local segment.
    vram_total = adapter_desc.DedicatedVideoMemory;

    ComPtr<IDXGIAdapter3> adapter3;
    if (SUCCEEDED(adapter.As(&adapter3))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO local_info = {};
        DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal_info = {};
        bool have_local = SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info))
                          && local_info.Budget > 0;
        bool have_nonlocal = SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal_info))
                             && nonlocal_info.Budget > 0;

        if (have_local) {
            vram_total = (size_t)local_info.Budget;
            vram_free  = local_info.Budget > local_info.CurrentUsage
                       ? (size_t)(local_info.Budget - local_info.CurrentUsage) : 0;
        }

        // iGPU: add non-local (shared system RAM) budget.
        // DedicatedVideoMemory < 512MB is a reliable iGPU indicator on Windows.
        // The DXGI non-local budget is OS-managed but can be very large (84%+ of
        // physical RAM). Cap total at 32GB to match Vulkan's behavior and prevent
        // iGPU from appearing to have more memory than a dGPU in multi-GPU systems.
        static constexpr size_t IGPU_MAX_TOTAL = (size_t)32 * 1024 * 1024 * 1024;
        bool is_igpu = (adapter_desc.DedicatedVideoMemory < (size_t)512 * 1024 * 1024);
        if (is_igpu && have_nonlocal) {
            size_t nonlocal_free = nonlocal_info.Budget > nonlocal_info.CurrentUsage
                                 ? (size_t)(nonlocal_info.Budget - nonlocal_info.CurrentUsage) : 0;
            vram_total += (size_t)nonlocal_info.Budget;
            vram_free  += nonlocal_free;
            // Only cap if the combined total exceeds the limit
            if (vram_total > IGPU_MAX_TOTAL) {
                vram_total = IGPU_MAX_TOTAL;
                if (vram_free > IGPU_MAX_TOTAL) vram_free = IGPU_MAX_TOTAL;
            }
        }
    }

    // Fallback if budget queries didn't work
    if (vram_total < (size_t)512 * 1024 * 1024 && adapter_desc.SharedSystemMemory > 0) {
        vram_total = adapter_desc.SharedSystemMemory;
        vram_free = vram_total;
    }
    if (vram_free == 0) {
        vram_free = vram_total;
    }

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

    // dp4a (dot4add_i8packed) — check SM 6.4 support via shader model feature query
    // Query highest supported shader model
    D3D_SHADER_MODEL highest_sm = D3D_SHADER_MODEL_6_0;
    {
        D3D12_FEATURE_DATA_SHADER_MODEL sm = {};
        sm.HighestShaderModel = D3D_SHADER_MODEL_6_9;
        HRESULT hr2 = device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm));
        if (SUCCEEDED(hr2)) highest_sm = sm.HighestShaderModel;
        dp4a_supported = highest_sm >= D3D_SHADER_MODEL_6_4;
    }

    // Native 16-bit shader ops — required for the `_fp16_dxil` blob variants.
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS4 opts4 = {};
        HRESULT hr2 = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &opts4, sizeof(opts4));
        if (SUCCEEDED(hr2) && opts4.Native16BitShaderOpsSupported) {
            fp16_supported = true;
        }
    }

    // Query wave (warp/subgroup) size for shader variant selection.
    // AMD RDNA: use WaveLaneCountMax because compute shaders run in wave64
    // mode even though WaveLaneCountMin=32. Using Min causes compile-time
    // WARP_SIZE=32 vs runtime wave64 mismatch in reductions.
    // Intel/NVIDIA: use WaveLaneCountMin for best performance.
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts1 = {};
        HRESULT hr2 = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opts1, sizeof(opts1));
        if (SUCCEEDED(hr2) && opts1.WaveLaneCountMin > 0) {
            bool is_amd = (adapter_desc.VendorId == 0x1002);
            wave_size = is_amd ? opts1.WaveLaneCountMax : opts1.WaveLaneCountMin;
        }
    }

    detect_memory_architecture();

    create_common_root_signature();
    init_shader_blobs();

    DX12_LOG_INFO("Device %zu: %s (%s, VRAM: %.1f GB, SM: 6.%d, wave: %u, CV: %s, WaveMMA: %s%s, dp4a: %s, fp16: %s)\n",
                  idx, name.c_str(), description.c_str(),
                  (double)vram_total / (1024.0 * 1024.0 * 1024.0),
                  (int)(highest_sm & 0xF),
                  wave_size,
                  cooperative_vector_supported ? "yes" : "no",
                  wave_mma_supported ? "yes" : "no",
                  wave_mma_supported ? (std::string(" K=") + std::to_string(wave_mma_K) +
                                        " wave=" + std::to_string(wave_mma_wave_size) +
                                        (wave_mma_f16_acc32 ? " f16→f32" : " f16→f16")).c_str() : "",
                  dp4a_supported ? "yes" : "no",
                  fp16_supported ? "yes" : "no");
}

void dx12_device::create_common_root_signature() {
    // Common root signature layout:
    //   Slot 0: Root constants (dx12_shader_params)
    //   Slot 1: SRV root descriptor (src0 ByteAddressBuffer)
    //   Slot 2: SRV root descriptor (src1 ByteAddressBuffer)
    //   Slot 3: UAV root descriptor (dst  RWByteAddressBuffer)
    //   Slot 4: SRV root descriptor (src2 ByteAddressBuffer) [optional]
    //   Slot 5: SRV root descriptor (src3 ByteAddressBuffer) [optional, mask]
    //   Slot 6: UAV root descriptor (u1)  [optional, splitkv temp]
    //   Slot 7: SRV root descriptor (src4 ByteAddressBuffer) [optional, GDN/SSM_SCAN]
    //   Slot 8: SRV root descriptor (src5 ByteAddressBuffer) [optional, GDN/SSM_SCAN]
    //   Slot 9: SRV root descriptor (src6 ByteAddressBuffer) [optional, SSM_SCAN ids]

    D3D12_ROOT_PARAMETER1 params[10] = {};

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
    params[1].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[1].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 2: src1 SRV (t1)
    params[2].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].Descriptor.ShaderRegister = 1; // t1
    params[2].Descriptor.RegisterSpace  = 0;
    params[2].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[2].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 3: dst UAV (u0)
    params[3].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[3].Descriptor.ShaderRegister = 0; // u0
    params[3].Descriptor.RegisterSpace  = 0;
    params[3].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[3].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 4: src2 SRV (t2)
    params[4].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[4].Descriptor.ShaderRegister = 2; // t2
    params[4].Descriptor.RegisterSpace  = 0;
    params[4].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[4].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 5: src3 SRV (t3) — mask for flash attention
    params[5].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[5].Descriptor.ShaderRegister = 3; // t3
    params[5].Descriptor.RegisterSpace  = 0;
    params[5].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[5].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 6: temp UAV (u1) — auxiliary temp buffer for split-KV flash attention
    params[6].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[6].Descriptor.ShaderRegister = 1; // u1
    params[6].Descriptor.RegisterSpace  = 0;
    params[6].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[6].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 7: src4 SRV (t4) — GDN beta / SSM_SCAN B
    params[7].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[7].Descriptor.ShaderRegister = 4; // t4
    params[7].Descriptor.RegisterSpace  = 0;
    params[7].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[7].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 8: src5 SRV (t5) — GDN state / SSM_SCAN C
    params[8].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[8].Descriptor.ShaderRegister = 5; // t5
    params[8].Descriptor.RegisterSpace  = 0;
    params[8].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[8].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 9: src6 SRV (t6) — SSM_SCAN ids
    params[9].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[9].Descriptor.ShaderRegister = 6; // t6
    params[9].Descriptor.RegisterSpace  = 0;
    params[9].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[9].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rsd = {};
    rsd.Version                  = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rsd.Desc_1_1.NumParameters   = 10;
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

    // Vulkan-style two-stage wait: when an `almost_ready` fence has already
    // signaled (or is close to signaling), the bulk of the GPU work is done
    // and the final ~20% of the graph completes within microseconds.  An
    // OS-level WaitForSingleObject incurs a syscall round-trip (~10-30us on
    // Windows) that is comparable to the remaining GPU work, so for the
    // *final* fence we briefly spin polling GetCompletedValue (yielding to
    // the scheduler each iteration) before falling back to event-wait.
    //
    // Heuristic: spin for up to ~500us if the almost-ready fence has fired,
    // otherwise go straight to event-wait (large remaining tail).  Disable
    // via DX12_NO_SPIN_WAIT=1 if it ever causes thermal/power problems.
    static const bool spin_disabled = (getenv("DX12_NO_SPIN_WAIT") != nullptr);
    const bool early_done =
        almost_ready_fence != 0 && fence->GetCompletedValue() >= almost_ready_fence;
    if (early_done && !spin_disabled) {
        // Tight loop with YieldProcessor; bounded by ~500us wall clock.
        LARGE_INTEGER qfreq, t0, tnow;
        QueryPerformanceFrequency(&qfreq);
        QueryPerformanceCounter(&t0);
        const LONGLONG spin_ticks = qfreq.QuadPart / 2000; // 500us
        for (int spins = 0; spins < 256; spins++) {
            for (int j = 0; j < 64; j++) YieldProcessor();
            if (fence->GetCompletedValue() >= value) return;
            QueryPerformanceCounter(&tnow);
            if (tnow.QuadPart - t0.QuadPart > spin_ticks) break;
        }
    }

    HRESULT hr = fence->SetEventOnCompletion(value, fence_event);
    DX12_CHECK(hr, "SetEventOnCompletion");
    WaitForSingleObject(fence_event, INFINITE);
}

void dx12_backend_context::wait_for_gpu() {
    if (fence_value == 0) return;
    wait_for_fence(fence_value);
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
// Memory architecture detection (UMA / ReBAR fast paths for set_tensor)
// ---------------------------------------------------------------------------

void dx12_device::detect_memory_architecture() {
    is_uma = false;

    // 1) UMA detection.  D3D12 reports UMA via D3D12_FEATURE_DATA_ARCHITECTURE
    //    (.UMA = TRUE on iGPUs / SoCs where there is no separate VRAM).
    {
        D3D12_FEATURE_DATA_ARCHITECTURE arch = {};
        arch.NodeIndex = 0;
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch)))) {
            is_uma = (arch.UMA != FALSE);
        }
    }
    // Fallback: DedicatedVideoMemory < 512MB is a reliable iGPU indicator on Windows.
    if (!is_uma && adapter_desc.DedicatedVideoMemory < (size_t)512 * 1024 * 1024) {
        is_uma = true;
    }

    static const bool no_uma = (getenv("DX12_NO_UMA") != nullptr);
    if (no_uma) is_uma = false;

    // NOTE: a "ReBAR-style direct VRAM write" path is INTENTIONALLY NOT
    // implemented on dGPU.  Vulkan exploits ReBAR via the flag combination
    // DEVICE_LOCAL | HOST_VISIBLE (ggml-vulkan.cpp:2800-2835).  D3D12 does
    // NOT expose this combination:
    //   - HEAP_TYPE_DEFAULT is GPU-only (no CPU access).
    //   - HEAP_TYPE_CUSTOM with MEMORY_POOL_L1 does not permit Map() (L1 is
    //     GPU-side VRAM, not visible to the CPU under any standard config).
    //   - HEAP_TYPE_CUSTOM with MEMORY_POOL_L0 places the buffer in SYSTEM
    //     RAM, which on a dGPU forces the GPU to read weights across PCIe at
    //     BAR bandwidth -- a 5-7x slowdown for inference.  Measured on
    //     RTX 6000 Ada: SmolVLM2 256M Q8_0 dropped from 381 t/s to 57 t/s.
    //
    // UMA (iGPU) is unaffected: L0 *is* the GPU's memory there, so CUSTOM
    // L0 + WRITE_BACK is read by the GPU from the exact same physical pages
    // without any cross-bus transfer.

    DX12_LOG_INFO("Memory architecture: %s\n",
                  is_uma ? "UMA (host-shared, direct write enabled)" : "discrete-VRAM (staging required)");
}

// ---------------------------------------------------------------------------
// Helper: create a GPU buffer (D3D12_HEAP_TYPE_DEFAULT)
// ---------------------------------------------------------------------------

// Create a CPU-writable backing buffer for a model-weights ggml_buffer.  When
// UMA or ReBAR is available we use a CUSTOM heap on memory pool L0 so the CPU
// can memcpy directly into the buffer's VRAM/RAM, skipping the staging copy
// + GPU CopyBufferRegion + fence wait that set_tensor would otherwise require.
//
// Returns the resource and (on success) a persistent CPU mapping in *mapped_out.
// Returns nullptr if the host-shared allocation fails (caller should fall back
// to the standard DEFAULT-heap path).
static ComPtr<ID3D12Resource> dx12_create_host_visible_buffer(dx12_device * dev,
                                                               size_t size,
                                                               D3D12_CPU_PAGE_PROPERTY page_prop,
                                                               void ** mapped_out) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type                 = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty      = page_prop;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    hp.CreationNodeMask     = 1;
    hp.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = std::max<size_t>(size, 256);
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // ALLOW_UNORDERED_ACCESS is required because the GGML buffer allocator
    // hands out the same buffer for both weight tensors (read-only via SRV)
    // and intermediate / KV-cache tensors (read-write via UAV).  Without
    // the flag the shader silently fails to write outputs and we get
    // garbage tokens (verified on Intel iGPU with the flag missing).
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ComPtr<ID3D12Resource> res;
    HRESULT hr = dev->device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res));
    if (FAILED(hr)) {
        return nullptr;
    }

    void * ptr = nullptr;
    D3D12_RANGE no_read = { 0, 0 };
    hr = res->Map(0, &no_read, &ptr);
    if (FAILED(hr) || !ptr) {
        return nullptr;
    }
    *mapped_out = ptr;
    return res;
}

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
    HRESULT hr = dev->device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&res));
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
    ctx->mapped    = nullptr;

    if (size > 0) {
        // Try the CPU-accessible fast paths first: CUSTOM heap on memory pool L0
        // with WRITE_BACK (cached) for UMA, WRITE_COMBINE for ReBAR-exposed VRAM.
        // When this works `set_tensor` becomes a direct memcpy with no staging.
        // Only used for buffer types we know are weight buffers -- the buffer
        // type is the same for all DX12 allocations today, but if any future
        // CPU-shader-write path appears it should opt out of the host-mapped
        // backing because CUSTOM heap allocations don't support UAV writes from
        // the GPU as flexibly as DEFAULT heap.
        if (dev->is_uma) {
            // AMD RDNA iGPUs: CUSTOM heap with L0 causes 25-30% GPU read
            // regression regardless of page property (WRITE_BACK or WRITE_COMBINE).
            // The DEFAULT heap lets the AMD driver choose optimal placement.
            // Intel iGPUs: CUSTOM L0 + WRITE_BACK works well (no snooping penalty).
            constexpr UINT VENDOR_AMD = 0x1002;
            if (dev->adapter_desc.VendorId != VENDOR_AMD) {
                ctx->resource = dx12_create_host_visible_buffer(dev, size,
                    D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, &ctx->mapped);
                if (ctx->resource) {
                    ctx->heap_type = D3D12_HEAP_TYPE_CUSTOM;
                }
            }
        }
        // Fall back to DEFAULT heap (GPU-only, staging required for set_tensor).
        if (!ctx->resource) {
            ctx->resource = dx12_create_buffer(dev, size);
        }
        if (!ctx->resource) {
            delete ctx;
            return nullptr;
        }
    }

    static const ggml_backend_buffer_i iface = {
        /* .free_buffer   = */ [](ggml_backend_buffer_t buffer) {
            auto * ctx = (dx12_buffer_context *)buffer->context;
            // Unmap host-mapped buffer before destroying (CUSTOM L0 path).
            if (ctx->mapped && ctx->resource) {
                D3D12_RANGE wr = { 0, ctx->size };
                ctx->resource->Unmap(0, &wr);
                ctx->mapped = nullptr;
            }
            delete ctx;
        },
        /* .get_base      = */ [](ggml_backend_buffer_t buffer) -> void * {
            // On UMA we persistently map the resource (CUSTOM L0 + WRITE_BACK).
            // Returning the real CPU pointer lets CPU-fallback ops read/write
            // the same memory the GPU sees -- essential when scheduler routes
            // an op (e.g. GATED_DELTA_NET fallback) to CPU whose dst tensor
            // lives in this DX12 buffer.  Without this, tensor->data would
            // be DX12_PTR_BASE+offset and CPU memcpy would AV.
            // Discrete-GPU buffers (DEFAULT heap, no Map) keep the sentinel;
            // ops on such tensors must run on DX12 (we report supports_op
            // accordingly), and dx12_tensor_offset still computes the right
            // offset because get_base is consistent for the same buffer.
            auto * ctx = (dx12_buffer_context *)buffer->context;
            return ctx->mapped ? ctx->mapped : (void *)(uintptr_t)0x1000;
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

            // UMA / ReBAR fast path: buffer is CPU-mapped (CUSTOM heap on L0) --
            // memset directly, no staging copy, no GPU command list.
            if (ctx->mapped) {
                size_t tensor_offset = dx12_tensor_offset(tensor) + offset;
                memset((uint8_t *)ctx->mapped + tensor_offset, value, size);
                return;
            }

            ctx->dev->init_xfer();
            ctx->dev->xfer_wait();
            ctx->dev->xfer_ensure_staging(size, 0);

            // Persistently mapped: just memset the existing pointer.
            memset(ctx->dev->xfer.upload_mapped, value, size);

            HRESULT hr = ctx->dev->xfer.cmd_alloc->Reset();
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

            // UMA / ReBAR fast path: buffer is CPU-mapped (CUSTOM heap on L0) --
            // memcpy directly into the destination, no staging copy, no GPU
            // command list, no fence wait.  This is the Vulkan ReBAR pattern
            // (ggml-vulkan.cpp:6748-6751 fast-direct-write).
            if (ctx->mapped) {
                size_t tensor_offset = dx12_tensor_offset(tensor) + offset;
                memcpy((uint8_t *)ctx->mapped + tensor_offset, data, size);
                return;
            }

            g_tls_device = ctx->dev->device.Get();
            
            // CRITICAL: Ensure compute command list is closed before transfer
            // The scheduler may call set_tensor between graph splits
            // We need all compute work to be submitted first
            
            ctx->dev->init_xfer();
            ctx->dev->xfer_wait(); // wait for any previous transfer
            ctx->dev->xfer_ensure_staging(size, 0);

            // Persistently mapped (since xfer_ensure_staging) -- just memcpy.
            memcpy(ctx->dev->xfer.upload_mapped, data, size);

            // Reset and record copy command
            HRESULT hr = ctx->dev->xfer.cmd_alloc->Reset();
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

            // UMA / ReBAR fast path: read directly from the host-mapped buffer.
            // For UMA (WRITE_BACK / cached) this is a normal cached read.
            // For ReBAR (WRITE_COMBINE) this is an uncached read across PCIe;
            // the read is correct but slower than a staging-readback cached
            // transfer.  We accept that since ReBAR readback isn't a common
            // hot path (mostly used for output sampling, which is small).
            if (ctx->mapped) {
                size_t tensor_offset = dx12_tensor_offset(tensor) + offset;
                memcpy(data, (uint8_t *)ctx->mapped + tensor_offset, size);
                return;
            }

            g_tls_device = ctx->dev->device.Get();
            ctx->dev->init_xfer();
            ctx->dev->xfer_wait(); // wait for any previous transfer
            ctx->dev->xfer_ensure_staging(0, size);

            // Reset and record copy command
            HRESULT hr = ctx->dev->xfer.cmd_alloc->Reset();
            DX12_CHECK(hr, "xfer cmd_alloc Reset(get)");
            hr = ctx->dev->xfer.cmd_list->Reset(ctx->dev->xfer.cmd_alloc.Get(), nullptr);
            DX12_CHECK(hr, "xfer cmd_list Reset(get)");

            size_t tensor_offset = dx12_tensor_offset(tensor) + offset;
            ctx->dev->xfer.cmd_list->CopyBufferRegion(ctx->dev->xfer.readback_staging.Get(), 0,
                                                       ctx->resource.Get(), tensor_offset, size);
            ctx->dev->xfer.cmd_list->Close();

            ID3D12CommandList * lists[] = { ctx->dev->xfer.cmd_list.Get() };
            ctx->dev->compute_queue->ExecuteCommandLists(1, lists);
            ctx->dev->xfer.fence_value++;
            ctx->dev->compute_queue->Signal(ctx->dev->xfer.fence.Get(), ctx->dev->xfer.fence_value);
            ctx->dev->xfer_wait();

            // Readback: D3D12_HEAP_TYPE_READBACK is cached (D3D12_CPU_PAGE_PROPERTY_WRITE_BACK).
            // The CPU cache may hold stale data after a GPU write; Map() with a non-empty
            // read_range is the documented cache-invalidation point.  We therefore Map/Unmap
            // per call rather than persistently mapping.
            void * mapped = nullptr;
            D3D12_RANGE read_range = { 0, size };
            hr = ctx->dev->xfer.readback_staging->Map(0, &read_range, &mapped);
            DX12_CHECK(hr, "Map readback staging");
            memcpy(data, mapped, size);
            D3D12_RANGE written = { 0, 0 };
            ctx->dev->xfer.readback_staging->Unmap(0, &written);
        },
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ nullptr,
        /* .clear         = */ [](ggml_backend_buffer_t buffer, uint8_t value) {
            auto * ctx = (dx12_buffer_context *)buffer->context;
            if (!ctx->resource || ctx->size == 0) return;

            // UMA / ReBAR fast path: memset the entire mapped buffer in one shot.
            if (ctx->mapped) {
                memset(ctx->mapped, value, ctx->size);
                return;
            }

            ctx->dev->init_xfer();
            ctx->dev->xfer_wait();

            const size_t chunk = 16 * 1024 * 1024;
            ctx->dev->xfer_ensure_staging(std::min(ctx->size, chunk), 0);

            // Persistently mapped: just memset the existing pointer.
            memset(ctx->dev->xfer.upload_mapped, value, std::min(ctx->size, chunk));

            HRESULT hr = ctx->dev->xfer.cmd_alloc->Reset();
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
    dx12_device * dev = (dx12_device *)buft->context;

    // D3D12 buffer max is 4 GB on most hardware. AMD UMA drivers can fail
    // near that cliff, so cap iGPU allocations lower and let ggml's generic
    // allocator split large model ranges into multiple DX12 buffers. AMD dGPUs
    // keep the normal limit.
    constexpr size_t max_d3d12_buffer_size = (size_t)4 * 1024 * 1024 * 1024 - 1;
    constexpr size_t max_amd_uma_buffer_size = (size_t)2 * 1024 * 1024 * 1024 - 1;
    constexpr UINT VENDOR_AMD = 0x1002;
    if (dev && dev->is_uma && dev->adapter_desc.VendorId == VENDOR_AMD) {
        return max_amd_uma_buffer_size;
    }
    return max_d3d12_buffer_size;
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
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONCAT:
        case GGML_OP_REPEAT:
        case GGML_OP_ROPE:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_IM2COL:
        case GGML_OP_PAD:
        case GGML_OP_UPSCALE:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_CONV_2D:
            // Support F32, F16, and BF16
            if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16) {
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
            // KV cache writes: src0 is F32, dst can be F16 or F32
            if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) {
                if (op->src[1] && op->src[1]->type != GGML_TYPE_I32 &&
                    op->src[1]->type != GGML_TYPE_I64) return false;
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
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
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
                if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_BF16 &&
                    t != GGML_TYPE_Q4_K && t != GGML_TYPE_Q5_K && t != GGML_TYPE_Q6_K &&
                    t != GGML_TYPE_Q4_0 && t != GGML_TYPE_Q4_1 &&
                    t != GGML_TYPE_Q5_0 && t != GGML_TYPE_Q5_1 &&
                    t != GGML_TYPE_Q8_0 && t != GGML_TYPE_Q8_1 &&
                    t != GGML_TYPE_Q2_K && t != GGML_TYPE_Q3_K &&
                    t != GGML_TYPE_IQ4_NL /* MM-on, GR-on */) return false;
                // Q3_K matvec/batch shaders both produce wrong results when K<4096
                // (root cause not yet identified). Force CPU fallback for those tensors.
                // SmolLM2 ffn_down (K=1536) is the canonical trigger; Phi-3 K>=3072 is fine.
                if (t == GGML_TYPE_Q3_K && op->src[0]->ne[0] < 4096) return false;
            }
            if (op->src[1] && op->src[1]->type != GGML_TYPE_F32 &&
                op->src[1]->type != GGML_TYPE_F16) return false;
            return true;

        case GGML_OP_MUL_MAT_ID:
            // MoE: src0 = expert weights, src1 = input, src2 = expert ids (I32)
            if (op->type != GGML_TYPE_F32) return false;
            if (op->src[0]) {
                ggml_type t = op->src[0]->type;
                if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_BF16 &&
                    t != GGML_TYPE_Q4_K && t != GGML_TYPE_Q5_K && t != GGML_TYPE_Q6_K &&
                    t != GGML_TYPE_Q4_0 && t != GGML_TYPE_Q4_1 &&
                    t != GGML_TYPE_Q5_0 && t != GGML_TYPE_Q5_1 &&
                    t != GGML_TYPE_Q8_0 && t != GGML_TYPE_IQ4_NL) return false;
            }
            if (op->src[1] && op->src[1]->type != GGML_TYPE_F32) return false;
            if (op->src[2] && op->src[2]->type != GGML_TYPE_I32) return false;
            return true;

        case GGML_OP_GET_ROWS:
            if (op->type != GGML_TYPE_F32) return false;
            if (op->src[0]) {
                ggml_type t = op->src[0]->type;
                if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_BF16 &&
                    t != GGML_TYPE_Q4_K && t != GGML_TYPE_Q5_K && t != GGML_TYPE_Q6_K &&
                    t != GGML_TYPE_Q4_0 && t != GGML_TYPE_Q4_1 &&
                    t != GGML_TYPE_Q5_0 && t != GGML_TYPE_Q5_1 &&
                    t != GGML_TYPE_Q8_0 && t != GGML_TYPE_Q8_1 &&
                    t != GGML_TYPE_Q2_K && t != GGML_TYPE_Q3_K &&
                    t != GGML_TYPE_IQ4_NL) return false;
            }
            return true;

        case GGML_OP_FLASH_ATTN_EXT:
            if (op->src[0]->type != GGML_TYPE_F32) return false;
            if (op->src[1]->type != GGML_TYPE_F32 && op->src[1]->type != GGML_TYPE_F16) return false;
            if (op->src[2]->type != GGML_TYPE_F32 && op->src[2]->type != GGML_TYPE_F16) return false;
            // The DX12 FA shaders currently implement the base
            // softmax(QK^T*scale + mask) @ V path. They use a single head
            // dimension for Q/K and V, and do not carry params/bindings for
            // sinks, ALiBi/max_bias, or logit softcap.
            if (op->src[1]->ne[0] != op->src[2]->ne[0]) return false;
            if (op->src[4] != nullptr) return false;
            {
                float max_bias = 0.0f;
                float logit_softcap = 0.0f;
                memcpy(&max_bias,      (const float *) op->op_params + 1, sizeof(float));
                memcpy(&logit_softcap, (const float *) op->op_params + 2, sizeof(float));
                if (max_bias != 0.0f || logit_softcap != 0.0f) return false;
            }
            return true;

        case GGML_OP_ROLL:
            // F32 only; dst shape == src0 shape (handled by default unary fill_params path)
            return op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32;

        case GGML_OP_SSM_CONV:
            // F32 only (Mamba/Gated Delta Net 1D depthwise convolution)
            return op->type == GGML_TYPE_F32 &&
                   op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1] && op->src[1]->type == GGML_TYPE_F32;

        case GGML_OP_L2_NORM:
            // F32 only; row-based normalization analogous to RMS_NORM
            return op->type == GGML_TYPE_F32 &&
                   op->src[0] && op->src[0]->type == GGML_TYPE_F32;

        case GGML_OP_GATED_DELTA_NET: {
            // Fused gated delta net (Mamba2-style attention substitute used by qwen3.5).
            // src0=q, src1=k, src2=v, src3=g, src4=beta, src5=state.
            // Output dst is interpreted as packed [token-attn outputs | new state].
            // Kill switch: DX12_DISABLE_FGDN=1 forces CPU fallback (debug aid).
            static int disable_fgdn = -1;
            if (disable_fgdn < 0) {
                const char * v = getenv("DX12_DISABLE_FGDN");
                disable_fgdn = (v && v[0] && v[0] != '0') ? 1 : 0;
            }
            if (disable_fgdn) return false;
            if (op->type != GGML_TYPE_F32) return false;
            for (int i = 0; i < 6; i++) {
                if (!op->src[i] || op->src[i]->type != GGML_TYPE_F32) return false;
            }
            const uint32_t S_v = (uint32_t)op->src[2]->ne[0];
            // Shader currently supports S_v in {32, 64, 128} (matches Vulkan).
            if (S_v != 32 && S_v != 64 && S_v != 128) return false;
            return true;
        }

        case GGML_OP_SSM_SCAN: {
            // Mamba2 selective scan. src0=ssm_state, src1=x, src2=dt, src3=A,
            // src4=B, src5=C, src6=ids (I32).
            // Kill switch: DX12_DISABLE_SSM_SCAN=1 forces CPU fallback (debug aid).
            static int disable_ssm = -1;
            if (disable_ssm < 0) {
                const char * v = getenv("DX12_DISABLE_SSM_SCAN");
                disable_ssm = (v && v[0] && v[0] != '0') ? 1 : 0;
            }
            if (disable_ssm) return false;
            if (op->type != GGML_TYPE_F32) return false;
            for (int i = 0; i < 6; i++) {
                if (!op->src[i] || op->src[i]->type != GGML_TYPE_F32) return false;
            }
            if (op->src[6] && op->src[6]->type != GGML_TYPE_I32) return false;
            // Mamba2 detection: A is a single float per head (n_dims == 1
            // means src3 is shape [n_head] only). Earlier code checked
            // `nb[1] != sizeof(float)`, but for a 1D tensor ggml may set
            // nb[1] = nb[0] * ne[0], not sizeof(float) — that test was
            // unreliable. Use ggml_n_dims directly.
            if (ggml_n_dims(op->src[3]) != 1) return false;
            const uint32_t d_state = (uint32_t)op->src[0]->ne[0];
            const uint32_t head_dim = (uint32_t)op->src[0]->ne[1];
            if (d_state != 128 && d_state != 256) return false;
            if (head_dim % 16 != 0) return false;
            return true;
        }

        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Graph compute: dispatch shaders for a compute graph
// ---------------------------------------------------------------------------

// Precompute YaRN correction range (corr_low, corr_high) from ROPE op_params.
// Mirrors the in-shader computation in rope_multi.hlsl / rope.hlsl so that
// fused shaders with no spare op_params slots for n_ctx_orig/beta_fast/beta_slow
// can still apply YaRN.
static void dx12_rope_corr_dims(const struct ggml_tensor * rope,
                                float & corr_low, float & corr_high) {
    const float * fp = (const float *)rope->op_params;
    const int32_t * ip = (const int32_t *)rope->op_params;
    const uint32_t n_dims     = (uint32_t)ip[1];
    const uint32_t n_ctx_orig = (uint32_t)ip[4];
    const float    freq_base  = fp[5];
    const float    beta_fast  = fp[9];
    const float    beta_slow  = fp[10];
    const float two_pi = 6.2831853071795864769f;
    const float corr_start = floorf((float)n_dims * logf((float)n_ctx_orig / (beta_fast * two_pi)) / (2.0f * logf(freq_base)));
    const float corr_end   = ceilf ((float)n_dims * logf((float)n_ctx_orig / (beta_slow * two_pi)) / (2.0f * logf(freq_base)));
    corr_low  = fmaxf(0.0f, corr_start);
    corr_high = fminf((float)n_dims - 1.0f, corr_end);
}

// Single source of truth for ROPE op_params packing across the 5 ROPE-family
// shaders. Each shader reads a slightly different subset of slots (see
// per-shader comments at the top of rope*.hlsl and rms_norm_mul_rope*.hlsl).
// Historically these slots were populated inline at each dispatch site with
// duplicated memcpy + override sequences, which was the root cause of every
// ROPE fusion regression we've shipped (Phi-3 KV truncation, Gemma-vision
// CLAMP, missing attn_factor/freq_factors). One helper, one place to fix.
//
// Canonical layout (across all kinds; "—" = unused / shader-ignored):
//
//   slot | STANDALONE     ROPE_SET_ROWS    FUSED_RMS_MUL_ROPE3   FUSED_..._ROPE5
//   -----+--------------------------------------------------------------------
//   [0]  | n_past(0)      n_past(0)        eps                   eps
//   [1]  | n_dims         n_dims           n_dims                n_dims
//   [2]  | mode           mode             mode                  mode
//   [3]  | n_ctx          corr_high        corr_high             corr_high
//   [4]  | n_ctx_orig     corr_low         corr_low              corr_low
//   [5]  | freq_base      freq_base        freq_base             freq_base
//   [6]  | freq_scale     freq_scale       freq_scale            freq_scale
//   [7]  | ext_factor     ext_factor       ext_factor            ext_factor
//   [8]  | attn_factor    set_rows_stride  —                     set_rows_stride
//   [9]  | beta_fast      set_rows_nb1     —                     set_rows_nb1
//   [10] | beta_slow      sr_idx_offset    pos_offset            pos_offset
//   [11] | mrope sec[0]   sr_idx_nb0       —                     sr_idx_offset
//   [12] | mrope sec[1]   —                pos_nb0               pos_nb0
//   [13] | mrope sec[2]   —                —                     sr_idx_nb0
//   [14] | mrope sec[3]   attn_factor      attn_factor           attn_factor
//   [15] | has_ff         has_ff           has_ff                has_ff
//
// Note: STANDALONE preserves the ggml-native layout so the same packing
// drives both rope.hlsl (NORMAL/NEOX) and rope_multi.hlsl (mrope/vision/
// imrope). The mrope sections live at [11..14] and would be clobbered by
// any of the non-standalone packings — that's why the fusion gates exclude
// mrope.
enum class dx12_rope_pack_kind : uint8_t {
    STANDALONE,           // rope.hlsl, rope_multi.hlsl
    ROPE_SET_ROWS,        // rope_set_rows.hlsl
    FUSED_RMS_MUL_ROPE3,  // rms_norm_mul_rope.hlsl
    FUSED_RMS_MUL_ROPE5,  // rms_norm_mul_rope_set_rows.hlsl
};

// Populate p.op_params[0..15] for the given ROPE dispatch.
//   rope_tensor:     the GGML_OP_ROPE tensor (always required)
//   set_rows_tensor: the SET_ROWS dst (only for ROPE_SET_ROWS / ..._ROPE5)
//   eps:             RMS_NORM epsilon (only for FUSED_RMS_MUL_ROPE3/5)
// All other p.* fields (ne/nb/offsets/esizes) must be set by the caller
// before/after this call as appropriate.
static void dx12_pack_rope_op_params(
        const struct ggml_tensor * rope_tensor,
        const struct ggml_tensor * set_rows_tensor,
        dx12_rope_pack_kind kind,
        float eps,
        dx12_shader_params & p) {
    GGML_ASSERT(rope_tensor && rope_tensor->op == GGML_OP_ROPE);
    const uint32_t * rope_up = (const uint32_t *)rope_tensor->op_params;

    // STANDALONE: ggml-native layout + has_ff at slot 15. This matches
    // dx12_fill_params' ROPE post-memcpy, included here so callers have a
    // single uniform code path.
    if (kind == dx12_rope_pack_kind::STANDALONE) {
        static_assert(sizeof(rope_tensor->op_params) >= sizeof(p.op_params),
                      "ggml op_params must be >= dx12_shader_params op_params");
        memcpy(p.op_params, rope_tensor->op_params, sizeof(p.op_params));
        p.op_params[15] = (rope_tensor->src[2] != nullptr) ? 1u : 0u;
        return;
    }

    // Non-standalone variants: rebuild op_params from scratch so stale ggml
    // slots (mrope sections, n_ctx, beta_fast/slow) cannot leak into shader
    // slots that have been repurposed.
    memset(p.op_params, 0, sizeof(p.op_params));

    // [0] eps for fused-with-RMS variants, otherwise n_past (always 0).
    if (kind == dx12_rope_pack_kind::FUSED_RMS_MUL_ROPE3 ||
        kind == dx12_rope_pack_kind::FUSED_RMS_MUL_ROPE5) {
        memcpy(&p.op_params[0], &eps, sizeof(uint32_t));
    }
    // [1..2] n_dims, mode (uint, ggml-native)
    p.op_params[1] = rope_up[1];
    p.op_params[2] = rope_up[2];
    // [3]/[4] host-precomputed YaRN corr_high/corr_low (overwriting ggml's
    // n_ctx/n_ctx_orig — the shaders do not need those once corr_* is known).
    {
        float corr_low = 0.0f, corr_high = 0.0f;
        dx12_rope_corr_dims(rope_tensor, corr_low, corr_high);
        memcpy(&p.op_params[3], &corr_high, sizeof(uint32_t));
        memcpy(&p.op_params[4], &corr_low,  sizeof(uint32_t));
    }
    // [5..7] freq_base, freq_scale, ext_factor (float, ggml-native)
    p.op_params[5] = rope_up[5];
    p.op_params[6] = rope_up[6];
    p.op_params[7] = rope_up[7];

    // SET_ROWS-derived slots
    if (kind == dx12_rope_pack_kind::ROPE_SET_ROWS ||
        kind == dx12_rope_pack_kind::FUSED_RMS_MUL_ROPE5) {
        GGML_ASSERT(set_rows_tensor && set_rows_tensor->op == GGML_OP_SET_ROWS);
        const struct ggml_tensor * row_idx = set_rows_tensor->src[1];
        // [8] elements per KV row (for flat indexing into KV cache)
        p.op_params[8] = (uint32_t)(set_rows_tensor->nb[1] / ggml_type_size(set_rows_tensor->type));
        // [9] byte stride between KV rows
        p.op_params[9] = (uint32_t)set_rows_tensor->nb[1];
        if (kind == dx12_rope_pack_kind::ROPE_SET_ROWS) {
            // ROPE_SET_ROWS: pos comes from src1 directly (no slot needed).
            // SET_ROWS row indices live at [10]/[11].
            p.op_params[10] = (uint32_t)dx12_tensor_offset(row_idx);
            p.op_params[11] = (uint32_t)row_idx->nb[0];
        } else {
            // 5-way: pos at src2 ([10]/[12]), row indices at src3 ([11]/[13])
            p.op_params[11] = (uint32_t)dx12_tensor_offset(row_idx);
            p.op_params[13] = (uint32_t)row_idx->nb[0];
        }
    }

    // ROPE position-tensor offset/stride for FUSED_RMS_MUL_ROPE3/5
    // (ROPE_SET_ROWS reads pos from src1 directly via src1_offset/nb10).
    if (kind == dx12_rope_pack_kind::FUSED_RMS_MUL_ROPE3 ||
        kind == dx12_rope_pack_kind::FUSED_RMS_MUL_ROPE5) {
        const struct ggml_tensor * pos = rope_tensor->src[1];
        p.op_params[10] = (uint32_t)dx12_tensor_offset(pos);
        p.op_params[12] = (uint32_t)pos->nb[0];
    }

    // [14] attn_factor (always uniform in slot 14 across non-standalone
    // variants — see slot table comment above).
    p.op_params[14] = rope_up[8];
    // [15] has_ff
    p.op_params[15] = (rope_tensor->src[2] != nullptr) ? 1u : 0u;
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
        p.src0_esize  = (src0->type == GGML_TYPE_BF16) ? 3 : (uint32_t)ggml_type_size(src0->type);
    }
    if (src1) {
        p.ne10 = (uint32_t)src1->ne[0]; p.ne11 = (uint32_t)src1->ne[1];
        p.ne12 = (uint32_t)src1->ne[2]; p.ne13 = (uint32_t)src1->ne[3];
        p.nb10 = (uint32_t)src1->nb[0]; p.nb11 = (uint32_t)src1->nb[1];
        p.nb12 = (uint32_t)src1->nb[2]; p.nb13 = (uint32_t)src1->nb[3];
        uint64_t off = dx12_tensor_offset(src1);
        GGML_ASSERT(off <= UINT32_MAX && "src1 offset exceeds 4GB");
        p.src1_offset = (uint32_t)off;
        p.src1_esize  = (src1->type == GGML_TYPE_BF16) ? 3 : (uint32_t)ggml_type_size(src1->type);
    }

    uint64_t dst_off = dx12_tensor_offset(tensor);
    GGML_ASSERT(dst_off <= UINT32_MAX && "dst offset exceeds 4GB");
    p.ne0 = (uint32_t)tensor->ne[0]; p.ne1 = (uint32_t)tensor->ne[1];
    p.ne2 = (uint32_t)tensor->ne[2]; p.ne3 = (uint32_t)tensor->ne[3];
    p.nb0 = (uint32_t)tensor->nb[0]; p.nb1 = (uint32_t)tensor->nb[1];
    p.nb2 = (uint32_t)tensor->nb[2]; p.nb3 = (uint32_t)tensor->nb[3];
    p.dst_offset = (uint32_t)dst_off;
    p.dst_esize  = (tensor->type == GGML_TYPE_BF16) ? 3 : (uint32_t)ggml_type_size(tensor->type);

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
        const uint32_t mask_esize = mask ? (mask->type == GGML_TYPE_BF16 ? 3u : (uint32_t)ggml_type_size(mask->type)) : 0u;
        p.op_params[8]  = mask ? (1u | ((uint32_t)mask->nb[0] << 8) | (mask_esize << 16)) : 0u; // mask info
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
    } else if (tensor->op == GGML_OP_MUL_MAT_ID) {
        // MUL_MAT_ID: pass src2 (expert ids) info via op_params
        const struct ggml_tensor * ids = tensor->src[2];
        if (ids) {
            p.op_params[0] = (uint32_t)dx12_tensor_offset(ids);  // src2_offset
            p.op_params[1] = (uint32_t)ids->nb[0];               // src2_nb0
            p.op_params[2] = (uint32_t)ids->nb[1];               // src2_nb1
        }
    } else if (tensor->op == GGML_OP_CPY ||
               tensor->op == GGML_OP_CONT ||
               tensor->op == GGML_OP_DUP) {
        // CPY/CONT/DUP need explicit I32 src/dst flags to distinguish from F32
        // (both have ggml_type_size==4 and the same src*_esize sentinel).
        // BF16 is already differentiated via src*_esize==3 sentinel.
        p.op_params[0] = (tensor->src[0] && tensor->src[0]->type == GGML_TYPE_I32) ? 1u : 0u;
        p.op_params[1] = (tensor->type == GGML_TYPE_I32) ? 1u : 0u;
    } else if (tensor->op == GGML_OP_GATED_DELTA_NET) {
        // GDN op_params layout (matches gated_delta_net.hlsl):
        //   [0]=H [1]=n_tokens [2]=n_seqs [3]=s_off
        //   [4..6]=sq1,sq2,sq3 [7..9]=sv1,sv2,sv3 [10..12]=sb1,sb2,sb3
        //   [13]=neq1 [14]=rq3 [15]=scale (float bits)
        const struct ggml_tensor * src_q    = tensor->src[0];
        const struct ggml_tensor * src_v    = tensor->src[2];
        const struct ggml_tensor * src_beta = tensor->src[4];
        const uint32_t S_v      = (uint32_t)src_v->ne[0];
        const uint32_t H        = (uint32_t)src_v->ne[1];
        const uint32_t n_tokens = (uint32_t)src_v->ne[2];
        const uint32_t n_seqs   = (uint32_t)src_v->ne[3];
        const uint32_t s_off    = S_v * H * n_tokens * n_seqs;
        p.op_params[0]  = H;
        p.op_params[1]  = n_tokens;
        p.op_params[2]  = n_seqs;
        p.op_params[3]  = s_off;
        p.op_params[4]  = (uint32_t)(src_q->nb[1] / sizeof(float));
        p.op_params[5]  = (uint32_t)(src_q->nb[2] / sizeof(float));
        p.op_params[6]  = (uint32_t)(src_q->nb[3] / sizeof(float));
        p.op_params[7]  = (uint32_t)(src_v->nb[1] / sizeof(float));
        p.op_params[8]  = (uint32_t)(src_v->nb[2] / sizeof(float));
        p.op_params[9]  = (uint32_t)(src_v->nb[3] / sizeof(float));
        p.op_params[10] = (uint32_t)(src_beta->nb[1] / sizeof(float));
        p.op_params[11] = (uint32_t)(src_beta->nb[2] / sizeof(float));
        p.op_params[12] = (uint32_t)(src_beta->nb[3] / sizeof(float));
        p.op_params[13] = (uint32_t)src_q->ne[1];
        p.op_params[14] = (uint32_t)(src_v->ne[3] / src_q->ne[3]);
        const float scale = 1.0f / sqrtf((float)S_v);
        memcpy(&p.op_params[15], &scale, sizeof(float));
    } else if (tensor->op == GGML_OP_SSM_SCAN) {
        // SSM_SCAN op_params layout (matches ssm_scan.hlsl):
        //   [0]=nb02 [1]=nb03 [2]=nb12 [3]=nb13 [4]=nb21 [5]=nb22 [6]=nb31
        //   [7]=nb42 [8]=nb43 [9]=nb52 [10]=nb53 [11]=s_off
        //   [12]=n_head [13]=d_head [14]=n_group [15]=n_tok
        const struct ggml_tensor * s0 = tensor->src[0];
        const struct ggml_tensor * x  = tensor->src[1];
        const struct ggml_tensor * dt = tensor->src[2];
        const struct ggml_tensor * A  = tensor->src[3];
        const struct ggml_tensor * B  = tensor->src[4];
        const struct ggml_tensor * C  = tensor->src[5];
        p.op_params[0]  = (uint32_t)s0->nb[2];
        p.op_params[1]  = (uint32_t)s0->nb[3];
        p.op_params[2]  = (uint32_t)x->nb[2];
        p.op_params[3]  = (uint32_t)x->nb[3];
        p.op_params[4]  = (uint32_t)dt->nb[1];
        p.op_params[5]  = (uint32_t)dt->nb[2];
        p.op_params[6]  = (uint32_t)A->nb[1];
        p.op_params[7]  = (uint32_t)B->nb[2];
        p.op_params[8]  = (uint32_t)B->nb[3];
        p.op_params[9]  = (uint32_t)C->nb[2];
        p.op_params[10] = (uint32_t)C->nb[3];
        p.op_params[11] = (uint32_t)(ggml_nelements(x) * sizeof(float));
        p.op_params[12] = (uint32_t)x->ne[1];   // n_head
        p.op_params[13] = (uint32_t)s0->ne[1];  // d_head
        p.op_params[14] = (uint32_t)B->ne[1];   // n_group
        p.op_params[15] = (uint32_t)x->ne[2];   // n_tok
    } else {
        static_assert(sizeof(tensor->op_params) >= sizeof(p.op_params), "op_params size mismatch");
        memcpy(p.op_params, tensor->op_params, sizeof(p.op_params));
        // ROPE: signal has_ff=1 if freq_factors (src2) tensor is bound.
        // op_params[15] is unused by ggml ROPE (sections only fills [11..14])
        // and the standalone rope.hlsl shader reads it as the has_ff flag.
        if (tensor->op == GGML_OP_ROPE) {
            p.op_params[15] = (tensor->src[2] != nullptr) ? 1u : 0u;
        }
    }
}

static ID3D12Resource * dx12_get_resource(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) return nullptr;
    auto * ctx = (dx12_buffer_context *)tensor->buffer->context;
    return ctx ? ctx->resource.Get() : nullptr;
}

// R1 — compute the per-node identity used to validate the replay cache.
// Captures everything the decision block branches on, but nothing that varies
// per token (positions, KV row indices, tensor pointers, byte offsets).
static inline void dx12_compute_node_identity(const struct ggml_tensor * node,
                                              dx12_node_identity & id) {
    memset(&id, 0, sizeof(id));
    id.op       = (uint8_t)node->op;
    id.dst_type = (uint8_t)node->type;
    if (node->src[0]) {
        id.src0_type = (uint8_t)node->src[0]->type;
        id.src0_ne0  = node->src[0]->ne[0];
        id.src0_ne2  = node->src[0]->ne[2];
    }
    if (node->src[1]) {
        id.src1_type = (uint8_t)node->src[1]->type;
        id.src1_ne0  = node->src[1]->ne[0];
        id.src1_ne2  = node->src[1]->ne[2];
    }
    if (node->src[2]) { id.has_src2 = 1; id.src2_type = (uint8_t)node->src[2]->type; }
    if (node->src[3]) { id.has_src3 = 1; id.src3_type = (uint8_t)node->src[3]->type; }
    id.dst_ne0 = node->ne[0];
    id.dst_ne1 = node->ne[1];
    // op_params: 12 i32 = 48 bytes covers RoPE mode/freq_base/freq_scale/ext_factor/attn_factor,
    // FA scale/max_bias/logit_softcap, ROPE mrope sections (slots 11..14 are inside this window).
    memcpy(id.op_params, node->op_params, sizeof(id.op_params));
}

// DX12_DUMP_TENSOR helper. Returns true if the tensor name matched any
// comma-separated token in `dump_name` and the dump succeeded. `call_idx` is
// the graph_compute call counter, `node_idx` is the index of the node within
// the current graph (for end-of-graph dumps) or -1 for per-dispatch dumps.
// Caller is responsible for ensuring all GPU writes to `node` have completed
// before calling this (via close_and_execute + wait_for_gpu).
static bool dx12_dump_tensor_if_matched(
        const ggml_tensor * node,
        const char * dump_name,
        const char * suffix,
        int call_idx,
        int node_idx) {
    if (!node || !node->name[0] || !node->buffer || !dump_name) return false;
    // Match if node->name contains any comma-separated token from dump_name.
    {
        const char * pat = dump_name;
        bool matched = false;
        while (*pat) {
            const char * comma = strchr(pat, ',');
            size_t tlen = comma ? (size_t)(comma - pat) : strlen(pat);
            if (tlen > 0 && tlen < 64) {
                char tok[64]; memcpy(tok, pat, tlen); tok[tlen] = 0;
                if (strstr(node->name, tok)) { matched = true; break; }
            }
            if (!comma) break;
            pat = comma + 1;
        }
        if (!matched) return false;
    }
    size_t nb = ggml_nbytes(node);
    std::vector<uint8_t> tmp(nb);
    node->buffer->iface.get_tensor(node->buffer, const_cast<ggml_tensor *>(node), tmp.data(), 0, nb);
    char fname[512];
    if (node_idx >= 0) {
        snprintf(fname, sizeof(fname), "dx12_dump_%s_call%d_node%d_%s.txt",
                 suffix, call_idx, node_idx, node->name);
    } else {
        snprintf(fname, sizeof(fname), "dx12_dump_%s_call%d_disp_%s.txt",
                 suffix, call_idx, node->name);
    }
    for (char * p = fname; *p; ++p) if (*p == '/' || *p == '\\' || *p == ':') *p = '_';
    FILE * f = fopen(fname, "w");
    if (!f) {
        fprintf(stderr, "[DX12_DUMP] failed to open %s\n", fname);
        return false;
    }
    fprintf(f, "# tensor=%s type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
            node->name, (int)node->type,
            (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3],
            node->nb[0], node->nb[1], node->nb[2], node->nb[3]);
    if (node->type == GGML_TYPE_F32) {
        const float * fp = (const float *)tmp.data();
        size_t n_floats = nb / sizeof(float);
        for (size_t k = 0; k < n_floats; ++k) fprintf(f, "%.9g\n", fp[k]);
    } else if (node->type == GGML_TYPE_F16) {
        const ggml_fp16_t * hp = (const ggml_fp16_t *)tmp.data();
        size_t n_h = nb / sizeof(ggml_fp16_t);
        for (size_t k = 0; k < n_h; ++k) fprintf(f, "%.9g\n", (double)ggml_fp16_to_fp32(hp[k]));
    } else {
        for (size_t k = 0; k < nb; ++k) fprintf(f, "%02x\n", tmp[k]);
    }
    fclose(f);
    fprintf(stderr, "[DX12_DUMP] wrote %s (%zu bytes)\n", fname, nb);
    return true;
}

static ggml_status dx12_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * bctx = (dx12_backend_context *)backend->context;

    g_tls_device = bctx->dev->device.Get();

    static const int dx12_trace = (getenv("DX12_TRACE_GRAPH") != nullptr) ? atoi(getenv("DX12_TRACE_GRAPH")) : 0;
    static int dx12_trace_call = 0;
    int trace_call = ++dx12_trace_call;
    if (dx12_trace) {
        fprintf(stderr, "[DX12_TRACE] graph_compute #%d enter: n_nodes=%d\n", trace_call, cgraph->n_nodes);
        fflush(stderr);
    }

    // Per-dispatch tensor dump: capture matching tensors immediately after
    // their producing dispatch, before later ops can clobber the workspace
    // buffer they alias. Slow (forces flush + GPU wait per match) — diagnostic
    // only. Without DX12_DUMP_PER_DISPATCH, dumps happen only at end-of-graph
    // (which captures stale memory for workspace-aliased intermediates).
    static const char * const dump_name_env = getenv("DX12_DUMP_TENSOR");
    static const bool dump_per_dispatch = (getenv("DX12_DUMP_PER_DISPATCH") != nullptr);
    static int dump_per_dispatch_call = 0;
    int dump_call_idx = dump_per_dispatch_call++;  // captured per graph_compute

    // Run auto-tuning on first graph compute
    if (!bctx->dev->tuning_done) {
        bctx->dev->run_autotune();
    }

    bctx->ensure_cmd_list_open();

    // Pre-allocate ancillary device buffers BEFORE the dispatch loop so that
    // their CreateCommittedResource (~100-500us driver stall) doesn't fire
    // mid-loop on the first token of a new session.
    //   - splitkv_temp (1MB): used by FA split-KV reduction; created the
    //     first time a node has n_splits > 1.
    if (!bctx->dev->splitkv_temp) {
        bctx->dev->splitkv_temp = dx12_create_buffer(bctx->dev, dx12_device::SPLITKV_TEMP_SIZE);
    }

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
    // Profile the 3rd-5th actual generation graphs (skip warmup/reserve).
    // Set DX12_PROFILE_PROMPT=1 to also profile prompt graphs (e.g. CLIP
    // encode is the largest prompt graph in vision models).
    static bool profile_prompt = (getenv("DX12_PROFILE_PROMPT") != nullptr);
    bool do_profile = profiling && ((!is_prompt && gen_graph >= 3 && gen_graph <= 5) ||
                                    (is_prompt && profile_prompt));
    std::map<std::string, double> op_times;
    std::map<std::string, uint32_t> op_counts;

    // GPU-side timestamp profiling — record per-dispatch start/end timestamps
    // into a query heap, then resolve and read after the graph completes. This
    // avoids the per-dispatch close/execute/wait/rebind dance which has proved
    // fragile (root-binding cache vs cmd-list reset interactions cause TDR).
    ComPtr<ID3D12QueryHeap> prof_heap;
    ComPtr<ID3D12Resource>  prof_readback;
    uint32_t prof_capacity = 0;
    uint32_t prof_idx = 0;
    std::vector<std::string> prof_keys;   // one per dispatched node
    UINT64 prof_freq = 1;
    if (do_profile) {
        prof_capacity = (uint32_t)cgraph->n_nodes * 2 + 32;
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = prof_capacity;
        HRESULT hr = bctx->dev->device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&prof_heap));
        if (FAILED(hr)) { do_profile = false; }
        if (do_profile) {
            D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = (UINT64)prof_capacity * sizeof(uint64_t);
            rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
            rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            hr = bctx->dev->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&prof_readback));
            if (FAILED(hr)) { do_profile = false; }
        }
        if (do_profile) {
            hr = bctx->dev->compute_queue->GetTimestampFrequency(&prof_freq);
            if (FAILED(hr) || prof_freq == 0) { do_profile = false; }
        }
        if (do_profile) {
            prof_keys.reserve(cgraph->n_nodes);
        }
    }

    int dispatch_weight = 0;
    int stream_nodes = 0;  // dispatched-node counter for stream-submit
    uint64_t mul_mat_bytes = 0;        // accumulated weight bytes since last submit
    uint64_t total_mul_mat_bytes = 0;  // grand total this graph (saved for next call)
    int submit_count = 0;              // for cross-frame doubling heuristic
    static constexpr int TDR_FLUSH_THRESHOLD = 24;
    // Stream-submit threshold: flush every N dispatched nodes so the GPU
    // starts executing the early part of the graph while the CPU is still
    // recording the rest. Vulkan does this every ~100 nodes. Tune via
    // DX12_STREAM_NODES env var (default 96; 0 disables streaming).
    // Values below 64 cause excessive cmd-list overhead and have hung
    // GPUs (Phi-3 Q4_K_M @ 32) — clamped to >=64 unless 0.
    static int stream_threshold = []() {
        const char * s = getenv("DX12_STREAM_NODES");
        if (!s) return 96;
        int v = atoi(s);
        if (v == 0) return 0;
        return v < 64 ? 64 : v;
    }();
    // Bytes-per-submit threshold (Vulkan ggml-vulkan.cpp:14513-14521).
    // Initial value: previous graph's total / 40, clamped to <= 100MB.  When
    // last_total == 0 (first call ever, or stream-bytes disabled) the value
    // is 0 and the bytes trigger does nothing -- the existing node-count
    // stream_threshold drives submission instead.
    //
    // For very large models (Phi-3 F16 ~7GB total per token) the 100MB cap
    // would cause over-submission since each individual MUL_MAT is ~50MB.
    // We cap at max(100MB, last_total/8) so very large models still see
    // ~8 submits/token max -- matches Vulkan's effective behaviour, since
    // their per-submit overhead is lower than D3D12's on NVIDIA.
    // Disable via DX12_STREAM_BYTES=0.
    static const bool stream_bytes_disabled = []() {
        const char * s = getenv("DX12_STREAM_BYTES");
        return s && atoi(s) == 0;
    }();
    uint64_t bytes_per_submit = 0;
    if (!stream_bytes_disabled && bctx->last_total_mul_mat_bytes > 0) {
        const uint64_t base = bctx->last_total_mul_mat_bytes / 40u;
        const uint64_t cap  = std::max<uint64_t>(100u * 1000u * 1000u,
                                                  bctx->last_total_mul_mat_bytes / 8u);
        bytes_per_submit = std::min(cap, base);
    }

    // Track unsynced tensor writes for smart barrier insertion.  Persistent
    // across tokens (field of bctx) -- just clear and reuse the bucket array.
    std::unordered_set<uintptr_t> & unsynced_writes = bctx->unsynced_writes;
    unsynced_writes.clear();

    // Debug: DX12_NO_FUSION=1 disables all op fusions for correctness testing
    static bool no_fusion = (getenv("DX12_NO_FUSION") != nullptr);
    // Debug: per-fusion-type bypasses for bisecting correctness issues
    static bool no_fuse_add_rms_mul   = (getenv("DX12_NO_FUSE_ADD_RMS_MUL")   != nullptr);
    static bool no_fuse_rms_mul_rope5 = (getenv("DX12_NO_FUSE_RMS_MUL_ROPE5") != nullptr);
    static bool no_fuse_rms_mul_rope3 = (getenv("DX12_NO_FUSE_RMS_MUL_ROPE3") != nullptr);
    static bool no_fuse_rms_mul       = (getenv("DX12_NO_FUSE_RMS_MUL")       != nullptr);
    static bool no_fuse_rope_set_rows = (getenv("DX12_NO_FUSE_ROPE_SET_ROWS") != nullptr);
    // Diagnostic-only: bypass the Qwen3 QK-Norm gate so the same binary can A/B
    // fused vs gated. Do not commit a change that ships with this enabled.
    static bool force_fuse_qk_norm = (getenv("DX12_FORCE_FUSE_QK_NORM") != nullptr);

    // R1 — replay-cache validation pass.  Compute current node identities and
    // compare against the cached decisions.  On mismatch (graph topology
    // change, shape change that crosses a routing threshold, or first call),
    // we mark the cache as needing rebuild and the per-node loop will compute
    // and store decisions as it runs.  On match (steady-state decode), the
    // per-node loop reads decisions directly from the cache and skips ~250
    // lines of pipeline lookup / fusion lookahead / route flag computation.
    static const bool no_replay = (getenv("DX12_NO_GRAPH_REPLAY") != nullptr);
    bool replay = false;
    dx12_replay_cache & rcache = bctx->replay_cache;
    if (!no_replay) {
        // Eager validation: O(n_nodes) memcmp of identity bytes.  ~5 µs for
        // a 400-node decode graph; cheap relative to ~90 µs of decision-block
        // work it skips when the cache is hot.
        bool match = ((int)rcache.decisions.size() == cgraph->n_nodes);
        if (match) {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                dx12_node_identity cur;
                dx12_compute_node_identity(cgraph->nodes[i], cur);
                if (memcmp(&cur, &rcache.decisions[i].identity, sizeof(cur)) != 0) {
                    match = false;
                    break;
                }
            }
        }
        if (match) {
            replay = true;
            rcache.hits++;
        } else {
            rcache.misses++;
            rcache.rebuilds++;
            rcache.decisions.assign(cgraph->n_nodes, dx12_node_decision{});
            for (int i = 0; i < cgraph->n_nodes; i++) {
                dx12_compute_node_identity(cgraph->nodes[i], rcache.decisions[i].identity);
            }
        }
        static const bool replay_stats = (getenv("DX12_REPLAY_STATS") != nullptr);
        if (replay_stats && ((rcache.hits + rcache.misses) % 100) == 0) {
            fprintf(stderr, "[DX12_REPLAY] hit=%llu miss=%llu rebuild=%llu n_nodes=%d\n",
                    (unsigned long long)rcache.hits, (unsigned long long)rcache.misses,
                    (unsigned long long)rcache.rebuilds, cgraph->n_nodes);
            fflush(stderr);
        }
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (dx12_trace >= 2) {
            fprintf(stderr, "[DX12_TRACE]  node %d/%d: op=%s name=%s\n",
                    i, cgraph->n_nodes, ggml_op_name(node->op), node->name);
            fflush(stderr);
        }

        // R1 — fast-path skip of view/reshape/permute/transpose nodes via cache
        if (replay && rcache.decisions[i].kind == DX12_DEC_SKIP) {
            for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                if (unsynced_writes.count((uintptr_t)node->src[s])) {
                    unsynced_writes.insert((uintptr_t)node);
                    break;
                }
            }
            continue;
        }
        if (replay && rcache.decisions[i].kind == DX12_DEC_NO_PIPELINE) {
            continue;
        }

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
            if (!replay && !no_replay) {
                rcache.decisions[i].kind = DX12_DEC_SKIP;
            }
            continue;
        }

        // Build pipeline key
        dx12_pipeline_key key = {};
        key.op       = node->op;
        key.dst_type = node->type;
        key.src0_type = node->src[0] ? node->src[0]->type : GGML_TYPE_F32;
        key.src1_type = node->src[1] ? node->src[1]->type : GGML_TYPE_F32;

        // Locals derived by either the decision block (record path) or the
        // replay cache (fast path).  Declared up front so both paths share
        // them with the dispatch/binding code below.
        struct ggml_tensor * fused_mul_node       = nullptr;
        struct ggml_tensor * fused_add_rms_node   = nullptr;
        struct ggml_tensor * fused_rms_node       = nullptr;
        struct ggml_tensor * fused_rope_after_rms = nullptr;
        struct ggml_tensor * fused_5way_set_rows  = nullptr;
        struct ggml_tensor * fused_rope_set_rows  = nullptr;
        struct ggml_tensor * fused_rope_view      = nullptr;
        struct ggml_tensor * fused_bias_add       = nullptr;
        struct ggml_tensor * fused_bias_tensor    = nullptr;
        // R9 fusion handles: in topological order the gate matvec comes
        // first (because ggml_swiglu_split's src[0] is gate, visited before
        // src[1]=up), then the up matvec, then the SWIGLU node.
        struct ggml_tensor * fused_mmv_glu_up     = nullptr;  // R9: 2nd matvec (up proj at i+1)
        struct ggml_tensor * fused_mmv_glu_glu    = nullptr;  // R9: SWIGLU split output at i+2
        bool is_matvec_dispatch = false;
        bool use_dp4a           = false;
        bool use_dp4a_matvec    = false;
        dx12_pipeline * pipeline = nullptr;

        if (replay) {
            // R1 fast path: pull cached decision and reconstruct fused_*
            // tensor pointers from cgraph (relative indices).
            const dx12_node_decision & d = rcache.decisions[i];
            pipeline           = d.pipeline;
            key.flags          = d.key_flags;
            is_matvec_dispatch = d.is_matvec_dispatch;
            use_dp4a           = d.use_dp4a;
            use_dp4a_matvec    = d.use_dp4a_matvec;

            switch (d.fusion_kind) {
                case DX12_FUSE_ADD_RMS_MUL:
                    fused_add_rms_node = node;
                    fused_rms_node     = cgraph->nodes[i + 1];
                    fused_mul_node     = cgraph->nodes[i + 2];
                    key.op             = GGML_OP_RMS_NORM;
                    break;
                case DX12_FUSE_RMS_MUL:
                    fused_mul_node = cgraph->nodes[i + 1];
                    key.op         = GGML_OP_RMS_NORM;
                    break;
                case DX12_FUSE_RMS_MUL_ROPE3:
                    fused_mul_node       = cgraph->nodes[i + 1];
                    fused_rope_after_rms = cgraph->nodes[i + 2];
                    key.op               = GGML_OP_RMS_NORM;
                    break;
                case DX12_FUSE_RMS_MUL_ROPE5:
                    fused_mul_node       = cgraph->nodes[i + 1];
                    fused_rope_after_rms = cgraph->nodes[i + 2];
                    fused_5way_set_rows  = cgraph->nodes[i + 4];
                    key.op               = GGML_OP_RMS_NORM;
                    break;
                case DX12_FUSE_ROPE_SET_ROWS:
                    fused_rope_view     = cgraph->nodes[i + 1];
                    fused_rope_set_rows = cgraph->nodes[i + 2];
                    key.op              = GGML_OP_ROPE;
                    break;
                case DX12_FUSE_MMV_GLU_SPLIT:
                    fused_mmv_glu_up  = cgraph->nodes[i + 1];
                    fused_mmv_glu_glu = cgraph->nodes[i + 2];
                    break;
                case DX12_FUSE_NONE:
                default:
                    break;
            }
            // Bias-add fusion can co-occur with MUL_MAT(M=1).
            if (d.has_bias_add && i + 1 < cgraph->n_nodes) {
                struct ggml_tensor * next = cgraph->nodes[i + 1];
                if (next->src[0] == node) fused_bias_tensor = next->src[1];
                else if (next->src[1] == node) fused_bias_tensor = next->src[0];
                if (fused_bias_tensor) fused_bias_add = next;
            }
        } else {

        // Op fusion: RMS_NORM + MUL → rms_norm_mul (single dispatch)
        // Also detects ADD + RMS_NORM + MUL → add_rms_norm_mul (triple fusion)

        // Try ADD + RMS_NORM + MUL triple fusion first
        if (!no_fusion && !no_fuse_add_rms_mul && node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
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

        // Fallback: try RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS) fusion, or RMS_NORM + MUL double fusion
        if (!no_fusion && !no_fuse_rms_mul && !fused_add_rms_node && node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes) {
            struct ggml_tensor * next = cgraph->nodes[i + 1];
            if (next->op == GGML_OP_MUL && next->src[0] == node) {
                const struct ggml_tensor * wt = next->src[1];
                const bool rms_mul_rope_weight_compatible =
                    wt != nullptr &&
                    wt->ne[0] == node->ne[0] &&
                    wt->ne[1] == 1 &&
                    wt->ne[2] == 1 &&
                    wt->ne[3] == 1;
                // Check for RMS_NORM + MUL + ROPE triple fusion.
                if (rms_mul_rope_weight_compatible && !no_fuse_rms_mul_rope3 && i + 2 < cgraph->n_nodes) {
                    struct ggml_tensor * rope = cgraph->nodes[i + 2];
                    int mode = rope->op == GGML_OP_ROPE ? ((const int32_t *)rope->op_params)[2] : -1;
                    // The fused RMS+MUL+ROPE shaders implement attn_factor,
                    // freq_factors, and YaRN ext_factor (corr_low/high are
                    // precomputed host-side and forwarded into the shader).
                    bool rope_ext_compatible = (rope->op == GGML_OP_ROPE);
                    // The fused RMS+MUL+ROPE shaders implement attn_factor,
                    // freq_factors, and YaRN ext_factor (corr_low/high are
                    // precomputed host-side and forwarded into the shader).
                    // Gate fusion on `node->src[0]->ne[1] == 1` to disable
                    // fused 3-way / 5-way for QK-Norm-style models
                    // (Qwen3, AFMoE, etc.) where the RMS_NORM operates per
                    // attention head (ne[1] == n_head > 1) on a broadcast
                    // weight. Attempted to remove this gate after op_params
                    // packing was centralized via dx12_pack_rope_op_params
                    // (commit 80572dc): all isolated test-backend-ops cases
                    // pass, prompt-eval + first decode token result_output is
                    // bit-identical, but subsequent decode tokens diverge (
                    // verified call3+ produces wrong logits on Qwen3-0.6B
                    // Q4_K_M with seed 42). Root cause not yet found; gate
                    // retained for runtime correctness.
                    if (rope->op == GGML_OP_ROPE && rope->src[0] == next &&
                        ggml_is_contiguous(next) && ggml_is_contiguous(rope) &&
                        next->ne[0] <= 1024 &&
                        node->src[0] && (force_fuse_qk_norm || node->src[0]->ne[1] == 1) &&
                        (mode == 0 || mode == 2) &&
                        rope_ext_compatible) {
                        // Check for 5-way: ROPE + VIEW + SET_ROWS
                        if (!no_fuse_rms_mul_rope5 && i + 4 < cgraph->n_nodes) {
                            struct ggml_tensor * view5 = cgraph->nodes[i + 3];
                            struct ggml_tensor * sr5 = cgraph->nodes[i + 4];
                            if (view5->op == GGML_OP_VIEW && sr5->op == GGML_OP_SET_ROWS &&
                                view5->src[0] == rope && sr5->src[0] == view5 &&
                                rope->src[0]->ne[3] == 1 &&
                                (sr5->type == GGML_TYPE_F32 || sr5->type == GGML_TYPE_F16) &&
                                ggml_is_contiguous(view5) &&
                                view5->ne[0] == rope->ne[0] * rope->ne[1] &&
                                (sr5->src[1]->type == GGML_TYPE_I32 || sr5->src[1]->type == GGML_TYPE_I64)) {
                                fused_mul_node = next;
                                fused_rope_after_rms = rope;
                                fused_5way_set_rows = sr5;
                                key.op = GGML_OP_RMS_NORM;
                                key.flags = 8;  // flags=8 means 5-way fusion
                            }
                        }
                        // Fallback: 3-way RMS+MUL+ROPE
                        if (!fused_5way_set_rows) {
                            fused_mul_node = next;
                            fused_rope_after_rms = rope;
                            key.op = GGML_OP_RMS_NORM;
                            key.flags = 7;
                        }
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
        if (!no_fusion && !no_fuse_rope_set_rows && node->op == GGML_OP_ROPE && i + 2 < cgraph->n_nodes && !fused_add_rms_node) {
            int rope_mode = ((const int32_t *)node->op_params)[2];
            // The fused rope_set_rows shader implements attn_factor,
            // freq_factors, and YaRN ext_factor (corr_low/high precomputed host-side).
            // Only fuse standard ROPE (mode 0/2), not mrope/imrope
            if (rope_mode == 0 || rope_mode == 2) {
                struct ggml_tensor * view = cgraph->nodes[i + 1];
                struct ggml_tensor * set_rows = cgraph->nodes[i + 2];
                if (view->op == GGML_OP_VIEW && set_rows->op == GGML_OP_SET_ROWS &&
                    view->src[0] == node && set_rows->src[0] == view &&
                    node->src[0]->ne[3] == 1 &&
                    (set_rows->type == GGML_TYPE_F32 || set_rows->type == GGML_TYPE_F16) &&
                    ggml_is_contiguous(view) &&
                    view->ne[0] == node->ne[0] * node->ne[1] &&
                    (set_rows->src[1]->type == GGML_TYPE_I32 || set_rows->src[1]->type == GGML_TYPE_I64)) {
                    fused_rope_set_rows = set_rows;
                    fused_rope_view = view;
                    key.op = GGML_OP_ROPE;
                    key.flags = 6;  // flags=6 means fused rope_set_rows
                }
            }
        }

        // For unary ops, store the unary op type in flags
        if (node->op == GGML_OP_UNARY) {
            key.flags = (uint32_t)ggml_get_unary_op(node);
        }

        // Detect mrope (multi-dimensional ROPE) — uses sections in op_params[11..14]
        if (node->op == GGML_OP_ROPE && key.flags == 0) {
            const int32_t * sections = (const int32_t *)node->op_params + 11;
            if (sections[0] > 0 || sections[1] > 0 || sections[2] > 0 || sections[3] > 0) {
                key.flags = 13;  // flags=13 means mrope (rope_multi shader)
            }
        }
        // For MUL_MAT with M=1, use matvec pipeline (flags=1, or flags=5 for 256-thread auto-tuned)
        // Only for types that have matvec shaders
        // Wave64 DP4A/Q8_1 routes can accumulate enough numerical drift to
        // corrupt model-level output. Keep wave64 on the MR/exact paths.
        const bool allow_dp4a_wave = bctx->dev->wave_size < 64;
        if (node->op == GGML_OP_MUL_MAT && node->ne[1] == 1 && node->src[0]) {
            ggml_type t = node->src[0]->type;
            if (t == GGML_TYPE_F16 || t == GGML_TYPE_F32 || t == GGML_TYPE_BF16 ||
                t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q5_0 ||
                t == GGML_TYPE_Q5_1 ||
                t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                t == GGML_TYPE_IQ4_NL ||
                t == GGML_TYPE_Q8_0) {
                key.flags = 1;
                // F16/F32 multi-row matvec — autotuned: 256-thread (mr,
                // flag=11) vs 32-thread (mr32, flag=12).  These shaders use
                // vector loads for F32 activations and packed weight rows, so
                // keep non-F32 src1 and potentially 2-byte-aligned F16 rows on
                // the generic matvec path.
                if (t == GGML_TYPE_F16 || t == GGML_TYPE_F32) {
                    const bool src1_f32_contiguous = node->src[1] &&
                                                     node->src[1]->type == GGML_TYPE_F32 &&
                                                     node->src[1]->nb[0] == sizeof(float);
                    bool src0_vector_aligned = true;
                    if (t == GGML_TYPE_F16) {
                        const uint64_t src0_off = dx12_tensor_offset(node->src[0]);
                        src0_vector_aligned = (src0_off & 3u) == 0 &&
                                              (node->src[0]->nb[1] & 3) == 0 &&
                                              (node->src[0]->nb[2] & 3) == 0 &&
                                              (node->src[0]->nb[3] & 3) == 0;
                    }
                    if (src1_f32_contiguous && src0_vector_aligned) {
                        bool is_amd_wave64 = (bctx->dev->wave_size >= 64);
                        uint32_t K = (uint32_t)node->src[0]->ne[0];
                        bool use_256 = is_amd_wave64
                                    || bctx->dev->f16_mr_use_256
                                    || (K >= bctx->dev->f16_mr_k_256_threshold);
                        key.flags = use_256 ? 11 : 12;
                    }
                }
                // Q5_K/Q6_K/Q8_0/Q5_0/Q5_1 multi-row matvec
                if (t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q8_0 ||
                    t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1) {
                    key.flags = 9;
                }
                // Q2_K multi-row matvec.
                // Default: 256-thread block-level shader (16 threads/block, 16
                // elements/thread, two output rows per workgroup, bias factorisation).
                // Big decode win on AMD wave64: +43% on Phi-3 Q2_K, neutral on
                // SmolLM2 Q2_K. Set DX12_Q2K_BLOCKED=0 to revert to fl=19.
                if (t == GGML_TYPE_Q2_K) {
                    static const char * q2k_blk_env = getenv("DX12_Q2K_BLOCKED");
                    bool q2k_blocked = (q2k_blk_env == nullptr) ||
                                       (q2k_blk_env[0] != '0');
                    constexpr UINT VENDOR_AMD = 0x1002;
                    bool is_amd = (bctx->dev->adapter_desc.VendorId == VENDOR_AMD);
                    // Block-level shader uses 2-byte aligned ByteAddressBuffer.Load4
                    // (Q2_K block = 84 bytes); only AMD GCN/RDNA tolerates the misaligned
                    // load. Other vendors produce wrong results.
                    if (q2k_blocked && is_amd && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float) &&
                        (node->src[0]->ne[0] % 256) == 0) {
                        key.flags = 27;     // Q2_K block-level matvec (default)
                    } else {
                        key.flags = 19;
                    }
                }
                // Q3_K multi-row matvec (2 rows/group, 256 threads).
                // Only safe for K >= 4096 — supports_op already routes K<4096 to CPU.
                // Diagnostic block-level variant available via DX12_Q3K_BLOCKED=1
                // (neutral on Phi-3 Q3_K_M, kept opt-in for further tuning).
                if (t == GGML_TYPE_Q3_K && node->src[0]->ne[0] >= 4096) {
                    static const char * q3k_blk_env = getenv("DX12_Q3K_BLOCKED");
                    bool q3k_blocked = (q3k_blk_env != nullptr) && (q3k_blk_env[0] != '0');
                    constexpr UINT VENDOR_AMD = 0x1002;
                    bool is_amd = (bctx->dev->adapter_desc.VendorId == VENDOR_AMD);
                    // Block-level Q3_K (110-byte block) uses 2-byte aligned Load4 that
                    // only AMD tolerates; opt-in via env var, AMD only.
                    if (q3k_blocked && is_amd && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float) &&
                        (node->src[0]->ne[0] % 256) == 0) {
                        key.flags = 26;     // Q3_K block-level matvec (diagnostic)
                    } else {
                        key.flags = 20;
                    }
                }
                // Q8_0 on AMD wave64 with large K: use vectorized 256-thread multi-row.
                // Processes 4 elements/thread via packed loads. Only for K >= 1536
                // where there's enough work per thread — K=576 regresses with 256 threads.
                if (t == GGML_TYPE_Q8_0 && bctx->dev->wave_size >= 64 &&
                    node->src[0]->ne[0] >= 1536) {
                    key.flags = 18;  // Q8_0 mr256v (256-thread, AMD wave64)
                    use_dp4a_matvec = false;
                }
                // Q8_0 on AMD wave64 with small K (< 1536): single-wave 64-thread
                // WG with NUM_ROWS=4. The default 32-thread mr leaves half the
                // wave idle on AMD; mr256v over-pads. Default-on for AMD wave64
                // (verified +5-9% decode on SmolLM2/SmolVLM2/Phi-3 Q8_0); opt
                // out via DX12_Q8_MR64=0.
                if (t == GGML_TYPE_Q8_0 && bctx->dev->wave_size >= 64 &&
                    node->src[0]->ne[0] < 1536 && (node->src[0]->ne[0] % 32) == 0) {
                    static const char * q8_mr64_env = getenv("DX12_Q8_MR64");
                    bool q8_mr64 = (q8_mr64_env == nullptr) || (q8_mr64_env[0] != '0');
                    if (q8_mr64 && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float)) {
                        key.flags = 28;     // Q8_0 mr64 (4 rows/group, AMD wave64)
                        use_dp4a_matvec = false;
                    }
                }
                // Q5_K subgroup matvec: single-wave (32-thread) WG with subgroup
                // reduction (no shmem). Big win on wave==32 GPUs (NVIDIA) over
                // the 256-thread MR variant which leaves most lanes idle when
                // num_blocks_per_row is small. Vulkan-parity port.
                if (t == GGML_TYPE_Q5_K && bctx->dev->wave_size == 32) {
                    key.flags = 15;
                }
                // Q4_K: prefer dp4a matvec when supported. Vulkan uses dotPacked4x8EXT
                // here and gets ~2x throughput on Intel for the dominant SmolVLM2 weight type.
                // Gate off NVIDIA per GOTCHAS.md (cumulative precision drift on NVIDIA JIT).
                // Wave-portable since the shader's reduction was ported to use
                // WaveGetLaneCount() + linear final sum (works on Intel UHD wave=8).
                if (t == GGML_TYPE_Q4_K) {
                    key.flags = 9;
                    constexpr UINT VENDOR_NVIDIA = 0x10DE;
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == VENDOR_NVIDIA);
                    if (bctx->dev->dp4a_supported && allow_dp4a_wave &&
                        !nvidia &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = 10;          // Q4_K dp4a multi-row matvec
                        if (bctx->dev->q4k_dp4a_use_32) key.flags = 13; // 32-thread variant
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                }
                // Q5_K: same dp4a treatment as Q4_K (5th bit merged into nibble).
                if (t == GGML_TYPE_Q5_K) {
                    constexpr UINT VENDOR_NVIDIA = 0x10DE;
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == VENDOR_NVIDIA);
                    if (bctx->dev->dp4a_supported && allow_dp4a_wave &&
                        !nvidia &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = 14;          // Q5_K dp4a multi-row matvec
                        if (bctx->dev->q5k_dp4a_use_32) key.flags = 16; // 32-thread variant
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                }
                // Q6_K: dp4a matvec. q = (ql_nibble | (qh<<4)) - 32 fits int8;
                // we keep q as unsigned [0,63] for dp4a and subtract the bias
                // 32*sum(q8) at the end (no min/dmin term, just per-subblock
                // int8 scale). Same gating as Q4_K/Q5_K dp4a.
                if (t == GGML_TYPE_Q6_K) {
                    constexpr UINT VENDOR_NVIDIA = 0x10DE;
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == VENDOR_NVIDIA);
                    // Block-level Q6_K matvec is the default: it amortizes the
                    // block decode across 16 threads/block instead of decoding
                    // per-element, and shares the activation reads across two
                    // output rows. Set DX12_Q6K_BLOCKED=0 to revert to fl=9.
                    static const char * q6k_blk_env = getenv("DX12_Q6K_BLOCKED");
                    bool q6k_blocked = (q6k_blk_env == nullptr) ||
                                       (q6k_blk_env[0] != '0');
                    constexpr UINT VENDOR_AMD = 0x1002;
                    bool is_amd = (bctx->dev->adapter_desc.VendorId == VENDOR_AMD);
                    // Block-level Q6_K (210-byte block) uses 2-byte aligned Load4 that
                    // only AMD GCN/RDNA tolerates; other vendors get wrong results.
                    if (q6k_blocked && is_amd && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float) &&
                        (node->src[0]->ne[0] % 256) == 0) {
                        key.flags = 25;         // Q6_K block-level matvec
                    } else if (bctx->dev->dp4a_supported && allow_dp4a_wave &&
                        !nvidia &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = 23;          // Q6_K dp4a multi-row matvec
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                }
                // Q8_0: dp4a matvec — pure int8 dot (no min term, no precision
                // drift), so safe on NVIDIA unlike Q4_K/Q5_K dp4a matvec.
                // Mirrors the Q8_0 batch dp4a path (flag=8) which already runs
                // on NVIDIA. ~2x speedup on Phi-3 Q8_0 generation.
                // Skip when flag=18 (mr256v) was already selected above for
                // AMD wave64 + K>=1536 — that path was explicitly chosen and
                // sets use_dp4a_matvec=false on purpose.
                if (t == GGML_TYPE_Q8_0 && key.flags != 18) {
                    bool small_wave = (bctx->dev->wave_size < 16);
                    if (bctx->dev->dp4a_supported && allow_dp4a_wave && !small_wave &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = 17;          // Q8_0 dp4a multi-row matvec
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                }
                // Q5_0 / Q5_1 dp4a multi-row matvec (dot4add_i8packed + Q8_1 activations)
                // Same gating as Q4_K dp4a: requires SM 6.4 dp4a, non-tiny wave,
                // F32 contiguous src1, K%32==0 (Q5 block size = 32). Skip on NVIDIA
                // for safety (Q4_K/Q5_K dp4a were observed to drift there).
                if (t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1) {
                    constexpr UINT VENDOR_NVIDIA = 0x10DE;
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == VENDOR_NVIDIA);
                    bool small_wave = (bctx->dev->wave_size < 16);
                    if (bctx->dev->dp4a_supported && allow_dp4a_wave &&
                        !nvidia && !small_wave &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = (t == GGML_TYPE_Q5_0) ? 21 : 22;
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                }
                // Q5_0 single-wave variant for AMD wave64. The default 32-thread
                // dp4a shader leaves half the wave idle; mr64 (GROUP_SIZE=64,
                // NUM_ROWS=4) fills one full AMD wave and amortizes activation
                // reads 4x. Opt-in via DX12_Q50_MR64=1 pending validation;
                // promote to default after benches confirm gain on SmolVLM2/SmolLM2.
                if (t == GGML_TYPE_Q5_0 && bctx->dev->wave_size >= 64 &&
                    node->src[0]->ne[0] >= 32 && (node->src[0]->ne[0] % 32) == 0) {
                    constexpr UINT VENDOR_NVIDIA_Q50 = 0x10DE;
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == VENDOR_NVIDIA_Q50);
                    if (bctx->dev->dp4a_supported && allow_dp4a_wave && !nvidia &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1])) {
                        static const char * q50_mr64_env = getenv("DX12_Q50_MR64");
                        bool q50_mr64 = (q50_mr64_env != nullptr) && (q50_mr64_env[0] != '0');
                        if (q50_mr64) {
                            key.flags = 29;     // Q5_0 mr64 (single-wave AMD wave64, 4 rows/group)
                            use_dp4a_matvec = true;
                        }
                    }
                }
                is_matvec_dispatch = true;
            }
        }
        // For batch MUL_MAT (M > 1), use register-blocked tiled path (flags=4)
        // for types that have wmma shaders
        if (node->op == GGML_OP_MUL_MAT && node->ne[1] > 1 && node->src[0]) {
            ggml_type t = node->src[0]->type;
            // Q8_0: the wmma tiled kernel (32x32 tile, on-the-fly dequant into
            // groupshared) beats the flat dp4a kernel (mul_mat_q8_0_q8_1)
            // on PP-sized batches because it gets K-tile reuse across the
            // 256-thread workgroup. The flat shader had no tile/no register
            // block and re-read the full weight row per output element, which
            // on Phi-3 PP512 was so slow it could trip the Windows TDR.
            // Set DX12_FORCE_Q8_0_BATCH_DP4A=1 to fall back to the flat dp4a
            // shader for A/B testing.
            if (t == GGML_TYPE_F16 || t == GGML_TYPE_F32 || t == GGML_TYPE_BF16 ||
                t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q8_0) {
                static const bool force_q80_dp4a = (getenv("DX12_FORCE_Q8_0_BATCH_DP4A") != nullptr);
                if (t == GGML_TYPE_Q8_0 && force_q80_dp4a &&
                    bctx->dev->dp4a_supported && allow_dp4a_wave &&
                    node->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(node->src[1]) &&
                    (node->src[1]->ne[0] % 32) == 0) {
                    key.flags = 8;  // dp4a flat batch path (override)
                    use_dp4a = true;
                } else {
                    key.flags = 4;
                    // Cooperative-LDS Q4_K wmma variant: pre-decodes Q4_K
                    // scales/mins per (n_local, kt) into LDS once instead of
                    // having every thread re-decode per element. Originally
                    // shipped default-on as "portable to all DX12 vendors",
                    // but reproducibly triggers DXGI_ERROR_DEVICE_REMOVED
                    // (HRESULT 0x887A0005) on Intel Arc B390 (wave=16) the
                    // first time the PSO is dispatched. Until the wave16 path
                    // is debugged, default the LDS variant on only for AMD
                    // (where +15-23% PP is verified). Other vendors can still
                    // opt in explicitly with DX12_Q4K_WMMA_LDS=1, and AMD can
                    // opt out with DX12_Q4K_WMMA_LDS=0.
                    if (t == GGML_TYPE_Q4_K) {
                        constexpr UINT VENDOR_AMD = 0x1002;
                        const bool is_amd_q4k = (bctx->dev->adapter_desc.VendorId == VENDOR_AMD);
                        static const char * q4k_lds_env = getenv("DX12_Q4K_WMMA_LDS");
                        bool q4k_lds_enabled;
                        if (q4k_lds_env == nullptr) {
                            q4k_lds_enabled = is_amd_q4k;
                        } else {
                            q4k_lds_enabled = (q4k_lds_env[0] != '0');
                        }
                        if (q4k_lds_enabled) {
                            key.flags = 30;  // mul_mat_q4k_wmma_lds
                        }
                    }
                }
            }
        }

        // Op fusion: MUL_MAT(M=1) + ADD → matvec with fused bias add
        if (!no_fusion && is_matvec_dispatch && i + 1 < cgraph->n_nodes) {
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

        // R9 op fusion: MUL_MAT(W_gate, M=1) + MUL_MAT(W_up, M=1) + GLU(SWIGLU split)
        // In topological order ggml_swiglu_split's src[0] (gate) is visited
        // before src[1] (up), so the gate matvec lands at node[i] and the up
        // matvec at node[i+1].  GLU lands at node[i+2].
        // Only F16 weights are wired in v1 — the hot path for SmolLM2 / SmolVLM2
        // LLM blocks.  Phi-3 uses LLM_FFN_SWIGLU (single 2*n_ff projection)
        // and never matches.
        static const bool no_mmv_glu = (getenv("DX12_NO_MMV_GLU_FUSION") != nullptr);
        if (!no_fusion && !no_mmv_glu && !fused_bias_add && is_matvec_dispatch &&
            i + 2 < cgraph->n_nodes &&
            node->op == GGML_OP_MUL_MAT && node->src[0] && node->src[1] &&
            node->src[0]->type == GGML_TYPE_F16 && node->ne[1] == 1) {
            struct ggml_tensor * mm_up = cgraph->nodes[i + 1];
            struct ggml_tensor * glu   = cgraph->nodes[i + 2];
            if (mm_up->op == GGML_OP_MUL_MAT && glu->op == GGML_OP_GLU &&
                mm_up->src[0] && mm_up->src[1] &&
                mm_up->src[0]->type == GGML_TYPE_F16 &&
                mm_up->src[1] == node->src[1] &&            // share activation
                mm_up->ne[0] == node->ne[0] &&              // same output width N
                mm_up->ne[1] == 1 &&
                mm_up->src[0]->ne[0] == node->src[0]->ne[0] &&  // same K
                mm_up->src[0]->nb[1] == node->src[0]->nb[1] &&  // same row stride
                glu->src[0] == node && glu->src[1] == mm_up &&  // gate first, up second
                ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                ((const int32_t *)glu->op_params)[1] == 0 &&    // swapped == false
                glu->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32) {
                fused_mmv_glu_up   = mm_up;
                fused_mmv_glu_glu  = glu;
                key.flags = 24;  // mul_mat_vec_glu shader
                static const bool log_mmv_glu = (getenv("DX12_R9_LOG") != nullptr);
                static int mmv_glu_log_count = 0;
                if (log_mmv_glu && mmv_glu_log_count < 1) {
                    fprintf(stderr, "[DX12_R9] MMV+GLU fusion firing: K=%d N=%d (one-shot log)\n",
                            (int)node->src[0]->ne[0], (int)node->ne[0]);
                    fflush(stderr);
                    mmv_glu_log_count++;
                }
            }
        }

        // End of record-path decision block.  Look up pipeline and store the
        // decision into the replay cache so subsequent tokens can fast-path.
        pipeline = bctx->dev->get_or_create_pipeline(key);
        if (!pipeline || !pipeline->pso) {
            if (!no_replay) {
                rcache.decisions[i].kind = DX12_DEC_NO_PIPELINE;
            }
            continue;
        }
        if (!no_replay) {
            dx12_node_decision & d = rcache.decisions[i];
            d.kind               = DX12_DEC_COMPUTE;
            d.pipeline           = pipeline;
            d.key_flags          = (uint8_t)key.flags;
            d.is_matvec_dispatch = is_matvec_dispatch;
            d.use_dp4a           = use_dp4a;
            d.use_dp4a_matvec    = use_dp4a_matvec;
            // fusion_kind / skip_count are filled in below at the existing
            // `i += N` site.  needs_op_params and conservative_barrier are
            // filled in at their respective sites.
            if (fused_add_rms_node)        d.fusion_kind = DX12_FUSE_ADD_RMS_MUL;
            else if (fused_5way_set_rows)  d.fusion_kind = DX12_FUSE_RMS_MUL_ROPE5;
            else if (fused_rope_after_rms) d.fusion_kind = DX12_FUSE_RMS_MUL_ROPE3;
            else if (fused_mul_node)       d.fusion_kind = DX12_FUSE_RMS_MUL;
            else if (fused_rope_set_rows)  d.fusion_kind = DX12_FUSE_ROPE_SET_ROWS;
            else if (fused_mmv_glu_up)     d.fusion_kind = DX12_FUSE_MMV_GLU_SPLIT;
            else                            d.fusion_kind = DX12_FUSE_NONE;
            // skip_count derived from fusion_kind (matches the `i += N` block below).
            switch (d.fusion_kind) {
                case DX12_FUSE_ADD_RMS_MUL:    d.skip_count = 2; break;
                case DX12_FUSE_RMS_MUL_ROPE5:  d.skip_count = 4; break;
                case DX12_FUSE_RMS_MUL_ROPE3:  d.skip_count = 2; break;
                case DX12_FUSE_RMS_MUL:        d.skip_count = 1; break;
                case DX12_FUSE_ROPE_SET_ROWS:  d.skip_count = 2; break;
                case DX12_FUSE_MMV_GLU_SPLIT:  d.skip_count = 2; break;
                default:                       d.skip_count = 0; break;
            }
            if (fused_bias_add) {
                d.has_bias_add = true;
                d.skip_count  += 1;
            }
        }
        } // end of `if (replay) { ... } else { ... }`

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
            // op_params: ADD dst offset, weight offset, epsilon, ADD dst esize,
            // plus weight nb11/12/13 and ne11/12/13 for broadcast-aware indexing.
            params.op_params[0] = (uint32_t)dx12_tensor_offset(node);  // ADD's output offset
            params.op_params[1] = (uint32_t)dx12_tensor_offset(fused_mul_node->src[1]);  // weight offset
            float eps = 0.0f;
            memcpy(&eps, fused_rms_node->op_params, sizeof(float));
            memcpy(&params.op_params[2], &eps, sizeof(uint32_t));
            params.op_params[3] = (uint32_t)ggml_type_size(node->type);  // ADD dst esize
            const struct ggml_tensor * arm_wt = fused_mul_node->src[1];
            params.op_params[4] = (uint32_t)arm_wt->nb[1];
            params.op_params[5] = (uint32_t)arm_wt->nb[2];
            params.op_params[6] = (uint32_t)arm_wt->nb[3];
            params.op_params[7] = (uint32_t)arm_wt->ne[1];
            params.op_params[8] = (uint32_t)arm_wt->ne[2];
            params.op_params[9] = (uint32_t)arm_wt->ne[3];
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
                if (fused_5way_set_rows) {
                    // 5-way: RMS+MUL+ROPE+VIEW+SET_ROWS — dst is SET_ROWS output (KV cache)
                    params.ne0 = (uint32_t)fused_5way_set_rows->ne[0]; params.ne1 = (uint32_t)fused_5way_set_rows->ne[1];
                    params.ne2 = (uint32_t)fused_5way_set_rows->ne[2]; params.ne3 = (uint32_t)fused_5way_set_rows->ne[3];
                    params.nb0 = (uint32_t)fused_5way_set_rows->nb[0]; params.nb1 = (uint32_t)fused_5way_set_rows->nb[1];
                    params.nb2 = (uint32_t)fused_5way_set_rows->nb[2]; params.nb3 = (uint32_t)fused_5way_set_rows->nb[3];
                    params.dst_offset = (uint32_t)dx12_tensor_offset(fused_5way_set_rows);
                    params.dst_esize = (uint32_t)ggml_type_size(fused_5way_set_rows->type);
                } else {
                    // 3-way: RMS+MUL+ROPE — dst is ROPE output
                    params.ne0 = (uint32_t)fused_rope_after_rms->ne[0]; params.ne1 = (uint32_t)fused_rope_after_rms->ne[1];
                    params.ne2 = (uint32_t)fused_rope_after_rms->ne[2]; params.ne3 = (uint32_t)fused_rope_after_rms->ne[3];
                    params.nb0 = (uint32_t)fused_rope_after_rms->nb[0]; params.nb1 = (uint32_t)fused_rope_after_rms->nb[1];
                    params.nb2 = (uint32_t)fused_rope_after_rms->nb[2]; params.nb3 = (uint32_t)fused_rope_after_rms->nb[3];
                    params.dst_offset = (uint32_t)dx12_tensor_offset(fused_rope_after_rms);
                    params.dst_esize = (uint32_t)ggml_type_size(fused_rope_after_rms->type);
                }
                // Copy ROPE's op_params via the canonical packing helper.
                {
                    float eps = 0.0f;
                    memcpy(&eps, fused_rms_node ? fused_rms_node->op_params : node->op_params, sizeof(float));
                    dx12_pack_rope_op_params(
                        fused_rope_after_rms,
                        fused_5way_set_rows,  // null for 3-way
                        fused_5way_set_rows ? dx12_rope_pack_kind::FUSED_RMS_MUL_ROPE5
                                            : dx12_rope_pack_kind::FUSED_RMS_MUL_ROPE3,
                        eps, params);
                }
            } else {
                // Override dst with MUL's output
                params.ne0 = (uint32_t)fused_mul_node->ne[0]; params.ne1 = (uint32_t)fused_mul_node->ne[1];
                params.ne2 = (uint32_t)fused_mul_node->ne[2]; params.ne3 = (uint32_t)fused_mul_node->ne[3];
                params.nb0 = (uint32_t)fused_mul_node->nb[0]; params.nb1 = (uint32_t)fused_mul_node->nb[1];
                params.nb2 = (uint32_t)fused_mul_node->nb[2]; params.nb3 = (uint32_t)fused_mul_node->nb[3];
                params.dst_offset = (uint32_t)dx12_tensor_offset(fused_mul_node);
                params.dst_esize = (uint32_t)ggml_type_size(fused_mul_node->type);
            }
            // (was: unconditional dst_esize overwrite here that incorrectly
            //  clobbered the ROPE/SET_ROWS dst_esize on the rope path -- removed)
        } else {
            dx12_fill_params(node, params);
        }

        // Fused bias add: set op_params and override dst to ADD's output
        if (fused_bias_tensor) {
            params.op_params[0] = 1;  // bias fusion flag
            params.op_params[1] = (uint32_t)dx12_tensor_offset(fused_bias_tensor);  // bias byte offset
            params.op_params[2] = (uint32_t)fused_bias_tensor->nb[0];
            params.op_params[3] = (uint32_t)fused_bias_tensor->nb[2];
            params.op_params[4] = (uint32_t)fused_bias_tensor->nb[3];
            params.op_params[5] = (uint32_t)fused_bias_tensor->ne[2];
            params.op_params[6] = (uint32_t)fused_bias_tensor->ne[3];
            // Use ADD's output as destination
            params.dst_offset = (uint32_t)dx12_tensor_offset(fused_bias_add);
        }

        // Fused ROPE+SET_ROWS: override dst to SET_ROWS output, pass stride info
        if (fused_rope_set_rows) {
            dx12_pack_rope_op_params(node, fused_rope_set_rows,
                                     dx12_rope_pack_kind::ROPE_SET_ROWS,
                                     0.0f, params);
            // Override dst to SET_ROWS output (KV cache)
            params.ne0 = (uint32_t)fused_rope_set_rows->ne[0]; params.ne1 = (uint32_t)fused_rope_set_rows->ne[1];
            params.ne2 = (uint32_t)fused_rope_set_rows->ne[2]; params.ne3 = (uint32_t)fused_rope_set_rows->ne[3];
            params.nb0 = (uint32_t)fused_rope_set_rows->nb[0]; params.nb1 = (uint32_t)fused_rope_set_rows->nb[1];
            params.nb2 = (uint32_t)fused_rope_set_rows->nb[2]; params.nb3 = (uint32_t)fused_rope_set_rows->nb[3];
            params.dst_offset = (uint32_t)dx12_tensor_offset(fused_rope_set_rows);
            params.dst_esize = (uint32_t)ggml_type_size(fused_rope_set_rows->type);
        }

        // R9 fused MMV+GLU: override dst to SWIGLU output, encode W_up offset
        // in op_params[1] (mirrors fused_bias_tensor's slot-1 encoding pattern).
        if (fused_mmv_glu_glu) {
            params.op_params[1] = (uint32_t)dx12_tensor_offset(fused_mmv_glu_up->src[0]);
            params.dst_offset   = (uint32_t)dx12_tensor_offset(fused_mmv_glu_glu);
        }

        if (is_matvec_dispatch) {
            params.op_params[15] = 0;
        }

        // Upload root constants — only upload op_params for ops that need them
        static constexpr uint32_t BASE_PARAMS = 30;  // ne/nb/offsets/esizes = 30 DWORDs
        bool needs_op_params = (node->op == GGML_OP_SOFT_MAX || 
                                 node->op == GGML_OP_FLASH_ATTN_EXT || 
                                 node->op == GGML_OP_ROPE ||
                                 node->op == GGML_OP_RMS_NORM ||
                                 node->op == GGML_OP_NORM ||
                                 node->op == GGML_OP_L2_NORM ||
                                 node->op == GGML_OP_GATED_DELTA_NET ||
                                 node->op == GGML_OP_SSM_SCAN ||
                                 node->op == GGML_OP_GROUP_NORM ||
                                 node->op == GGML_OP_GLU ||
                                 node->op == GGML_OP_SCALE ||
                                 node->op == GGML_OP_CLAMP ||
                                 node->op == GGML_OP_UPSCALE ||
                                 node->op == GGML_OP_IM2COL ||
                                 node->op == GGML_OP_POOL_2D ||
                                 node->op == GGML_OP_POOL_1D ||
                                 node->op == GGML_OP_PAD ||
                                 node->op == GGML_OP_ROLL ||
                                 node->op == GGML_OP_CONV_2D ||
                                 node->op == GGML_OP_CONCAT ||
                                 node->op == GGML_OP_MUL_MAT_ID ||
                                 node->op == GGML_OP_CPY ||
                                 node->op == GGML_OP_CONT ||
                                 node->op == GGML_OP_DUP ||
                                 fused_bias_tensor ||
                                 fused_add_rms_node ||
                                 fused_rope_set_rows ||
                                 fused_mmv_glu_up);
        uint32_t num_constants = needs_op_params ? (uint32_t)(sizeof(params) / 4) : BASE_PARAMS;
        // FLASH_ATTN_EXT re-uploads the full params block at line ~2425 after
        // computing n_splits + gqa_ratio (which are encoded into op_params[15]
        // and read by the shader).  Skipping the upload here saves one
        // 184-byte SetComputeRoot32BitConstants per attention block per token.
        if (node->op != GGML_OP_FLASH_ATTN_EXT) {
            bctx->cmd_list->SetComputeRoot32BitConstants(0, num_constants, &params, 0);
        }

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
        if (fused_5way_set_rows) {
            dst_res = dx12_get_resource(fused_5way_set_rows);
        } else if (fused_rope_after_rms) {
            dst_res = dx12_get_resource(fused_rope_after_rms);
        } else if (fused_mul_node) {
            dst_res = dx12_get_resource(fused_mul_node);
        } else if (fused_bias_add) {
            dst_res = dx12_get_resource(fused_bias_add);
        } else if (fused_rope_set_rows) {
            dst_res = dx12_get_resource(fused_rope_set_rows);
        } else if (fused_mmv_glu_glu) {
            dst_res = dx12_get_resource(fused_mmv_glu_glu);
        } else {
            dst_res = dx12_get_resource(node);
        }

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

        // GATED_DELTA_NET / SSM_SCAN need src2 and src3 with their per-tensor
        // byte offsets baked into the GPU VA (the shaders read these via
        // tensor-base-relative addressing). The general src2/src3 path below
        // does NOT include the offset, which is fine for ops whose tensors
        // happen to live at offset 0 in their buffers (typical for weights),
        // but src2 (V) / src3 (g) here are activation tensors with non-zero
        // offsets — without this, the SRV points into another tensor's data
        // and the GPU reads OOB → page fault → device removed.
        bool gdn_or_ssm = (node->op == GGML_OP_GATED_DELTA_NET) || (node->op == GGML_OP_SSM_SCAN);
        if (gdn_or_ssm) {
            if (node->src[2]) {
                ID3D12Resource * src2_res = dx12_get_resource(node->src[2]);
                D3D12_GPU_VIRTUAL_ADDRESS src2_va = src2_res
                    ? (src2_res->GetGPUVirtualAddress() + dx12_tensor_offset(node->src[2]))
                    : src0_res->GetGPUVirtualAddress();
                if (src2_va != bctx->last_src2_va) {
                    bctx->cmd_list->SetComputeRootShaderResourceView(4, src2_va);
                    bctx->last_src2_va = src2_va;
                }
            }
            if (node->src[3]) {
                ID3D12Resource * src3_res = dx12_get_resource(node->src[3]);
                D3D12_GPU_VIRTUAL_ADDRESS src3_va = src3_res
                    ? (src3_res->GetGPUVirtualAddress() + dx12_tensor_offset(node->src[3]))
                    : src0_res->GetGPUVirtualAddress();
                if (src3_va != bctx->last_src3_va) {
                    bctx->cmd_list->SetComputeRootShaderResourceView(5, src3_va);
                    bctx->last_src3_va = src3_va;
                }
            }
        }

        // Optional src2/src3 — only bind for ops that use them
        bool needs_src2 = (node->op == GGML_OP_SOFT_MAX) || (node->op == GGML_OP_MUL_MAT_ID) || (fused_bias_tensor != nullptr) || (fused_add_rms_node != nullptr) || (fused_rope_after_rms != nullptr) ||
                          (fused_mmv_glu_up != nullptr) ||
                          (node->op == GGML_OP_ROPE && node->src[2] != nullptr);
        bool needs_src3 = (node->op == GGML_OP_FLASH_ATTN_EXT) || (fused_rope_set_rows != nullptr) || (fused_5way_set_rows != nullptr) ||
                          (fused_rope_after_rms != nullptr && fused_5way_set_rows == nullptr && fused_rope_after_rms->src[2] != nullptr);

        if (needs_src2 || needs_src3) {
            ID3D12Resource * src2_res;
            D3D12_GPU_VIRTUAL_ADDRESS src2_offset = 0;
            if (fused_rope_after_rms) {
                src2_res = dx12_get_resource(fused_rope_after_rms->src[1]);  // ROPE position indices
            } else if (fused_add_rms_node) {
                src2_res = dx12_get_resource(fused_mul_node->src[1]);  // weight tensor
            } else if (fused_bias_tensor) {
                src2_res = dx12_get_resource(fused_bias_tensor);
            } else if (fused_mmv_glu_up) {
                // R9: bind W_up as src2; per-tensor byte offset is encoded in op_params[1]
                // and consumed by the mul_mat_vec_glu shader (matches fused_bias_tensor pattern).
                src2_res = dx12_get_resource(fused_mmv_glu_up->src[0]);
            } else if (node->op == GGML_OP_ROPE && node->src[2]) {
                // freq_factors tensor (Llama-3.1, Phi-3 LongRope)
                src2_res = dx12_get_resource(node->src[2]);
                src2_offset = dx12_tensor_offset(node->src[2]);
            } else {
                src2_res = dx12_get_resource(node->src[2]);
            }
            D3D12_GPU_VIRTUAL_ADDRESS src2_va =
                (src2_res ? src2_res : src0_res)->GetGPUVirtualAddress() + src2_offset;
            if (src2_va != bctx->last_src2_va) {
                bctx->cmd_list->SetComputeRootShaderResourceView(4, src2_va);
                bctx->last_src2_va = src2_va;
            }
        }

        if (needs_src3) {
            ID3D12Resource * src3_res;
            D3D12_GPU_VIRTUAL_ADDRESS src3_offset = 0;
            if (fused_5way_set_rows) {
                src3_res = dx12_get_resource(fused_5way_set_rows->src[1]);  // SET_ROWS row indices
            } else if (fused_rope_set_rows) {
                src3_res = dx12_get_resource(fused_rope_set_rows->src[1]);  // SET_ROWS row indices
            } else if (fused_rope_after_rms && fused_rope_after_rms->src[2]) {
                // 3-way RMS+MUL+ROPE freq_factors: bind to src3/t3
                src3_res = dx12_get_resource(fused_rope_after_rms->src[2]);
                src3_offset = dx12_tensor_offset(fused_rope_after_rms->src[2]);
            } else {
                src3_res = dx12_get_resource(node->src[3]);
            }
            D3D12_GPU_VIRTUAL_ADDRESS src3_va =
                (src3_res ? src3_res : src0_res)->GetGPUVirtualAddress() + src3_offset;
            if (src3_va != bctx->last_src3_va) {
                bctx->cmd_list->SetComputeRootShaderResourceView(5, src3_va);
                bctx->last_src3_va = src3_va;
            }
        }

        // 5-way RMS+MUL+ROPE+VIEW+SET_ROWS freq_factors: bind to src4/t4
        // (src3 is occupied by SET_ROWS row indices in this fusion.)
        if (fused_5way_set_rows && fused_rope_after_rms && fused_rope_after_rms->src[2]) {
            ID3D12Resource * ff_res = dx12_get_resource(fused_rope_after_rms->src[2]);
            D3D12_GPU_VIRTUAL_ADDRESS ff_va = ff_res
                ? (ff_res->GetGPUVirtualAddress() + dx12_tensor_offset(fused_rope_after_rms->src[2]))
                : src0_res->GetGPUVirtualAddress();
            if (ff_va != bctx->last_src4_va) {
                bctx->cmd_list->SetComputeRootShaderResourceView(7, ff_va);
                bctx->last_src4_va = ff_va;
            }
        }

        // Optional src4/src5/src6 — bound for hybrid SSM ops with >4 input tensors
        // (GATED_DELTA_NET needs src0..src5; SSM_SCAN needs src0..src6).
        bool needs_src4 = (node->op == GGML_OP_GATED_DELTA_NET) || (node->op == GGML_OP_SSM_SCAN);
        bool needs_src5 = needs_src4;
        bool needs_src6 = (node->op == GGML_OP_SSM_SCAN);

        // For src4/src5/src6 we bake the per-tensor byte offset into the
        // GPU virtual address so the shader can treat byte offset 0 as the
        // start of the tensor (no spare op_params slots for src{4,5,6}_offset).
        if (needs_src4 && node->src[4]) {
            ID3D12Resource * src4_res = dx12_get_resource(node->src[4]);
            D3D12_GPU_VIRTUAL_ADDRESS src4_va = src4_res
                ? (src4_res->GetGPUVirtualAddress() + dx12_tensor_offset(node->src[4]))
                : src0_res->GetGPUVirtualAddress();
            if (src4_va != bctx->last_src4_va) {
                bctx->cmd_list->SetComputeRootShaderResourceView(7, src4_va);
                bctx->last_src4_va = src4_va;
            }
        }
        if (needs_src5 && node->src[5]) {
            ID3D12Resource * src5_res = dx12_get_resource(node->src[5]);
            D3D12_GPU_VIRTUAL_ADDRESS src5_va = src5_res
                ? (src5_res->GetGPUVirtualAddress() + dx12_tensor_offset(node->src[5]))
                : src0_res->GetGPUVirtualAddress();
            if (src5_va != bctx->last_src5_va) {
                bctx->cmd_list->SetComputeRootShaderResourceView(8, src5_va);
                bctx->last_src5_va = src5_va;
            }
        }
        if (needs_src6 && node->src[6]) {
            ID3D12Resource * src6_res = dx12_get_resource(node->src[6]);
            D3D12_GPU_VIRTUAL_ADDRESS src6_va = src6_res
                ? (src6_res->GetGPUVirtualAddress() + dx12_tensor_offset(node->src[6]))
                : src0_res->GetGPUVirtualAddress();
            if (src6_va != bctx->last_src6_va) {
                bctx->cmd_list->SetComputeRootShaderResourceView(9, src6_va);
                bctx->last_src6_va = src6_va;
            }
        }



        // Calculate dispatch dimensions
        uint32_t groups_x = 1, groups_y = 1, groups_z = 1;
        uint32_t matvec_row_groups = 0;

        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                bool is_matvec = (node->ne[1] == 1); // M=1: single token generation

                if (is_matvec) {
                    // Matvec dispatch
                    uint32_t N = (uint32_t)node->ne[0];
                    uint32_t batches = (uint32_t)(node->ne[2] * node->ne[3]);
                    if (key.flags == 9 || key.flags == 10 || key.flags == 11 ||
                        key.flags == 12 || key.flags == 13 || key.flags == 14 ||
                        key.flags == 15 || key.flags == 16 || key.flags == 17 ||
                        key.flags == 18 || key.flags == 19 || key.flags == 20 ||
                        key.flags == 21 || key.flags == 22 || key.flags == 23 ||
                        key.flags == 24 || key.flags == 25 || key.flags == 26 ||
                        key.flags == 27) {
                        // Multi-row: 2 rows per group
                        matvec_row_groups = (N + 1) / 2;
                    } else if (key.flags == 28 || key.flags == 29) {
                        // 4 rows per group: Q8_0 mr64 (28), Q5_0 mr64 (29)
                        matvec_row_groups = (N + 3) / 4;
                    } else {
                        // Default: one group per output row
                        matvec_row_groups = N;
                    }
                    if (matvec_row_groups > 65535) {
                        groups_x = 65535;
                        groups_y = (matvec_row_groups + 65534) / 65535;
                    } else {
                        groups_x = matvec_row_groups;
                        groups_y = 1;
                    }
                    groups_z = batches;
                } else if (key.flags == 4 || key.flags == 30) {
                    // Register-blocked tiled dispatch (32×32 tile) [numthreads(16,16,1)]
                    // fl=30 = Q4_K wmma cooperative-LDS variant (same dispatch)
                    uint32_t N = (uint32_t)node->ne[0];
                    uint32_t M = (uint32_t)node->ne[1];
                    uint32_t batches = (uint32_t)(node->ne[2] * node->ne[3]);
                    groups_x = (N + 31) / 32;
                    groups_y = (M + 31) / 32;
                    groups_z = batches;
                } else if (node->src[0] && (node->src[0]->type == GGML_TYPE_Q4_K ||
                                            node->src[0]->type == GGML_TYPE_Q5_K ||
                                            node->src[0]->type == GGML_TYPE_Q6_K ||
                                            node->src[0]->type == GGML_TYPE_Q4_0 ||
                                            node->src[0]->type == GGML_TYPE_Q4_1 ||
                                            node->src[0]->type == GGML_TYPE_Q5_0 ||
                                            node->src[0]->type == GGML_TYPE_Q5_1 ||
                                            node->src[0]->type == GGML_TYPE_Q8_0 ||
                                            node->src[0]->type == GGML_TYPE_Q8_1 ||
                                            node->src[0]->type == GGML_TYPE_Q2_K ||
                                            node->src[0]->type == GGML_TYPE_Q3_K ||
                                            node->src[0]->type == GGML_TYPE_IQ4_NL)) {
                    // Quantized flat shaders: 1 output per thread, 256 threads/group
                    uint32_t total = (uint32_t)(node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3]);
                    uint32_t total_groups = (total + 255) / 256;
                    // D3D12 limits dispatch to 65535 groups per dimension.
                    // Split into 2D dispatch if needed (shader uses group_id.y for overflow).
                    if (total_groups > 65535) {
                        groups_x = 65535;
                        groups_y = (total_groups + 65534) / 65535;
                    } else {
                        groups_x = total_groups;
                    }
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
            case GGML_OP_L2_NORM:
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SUM_ROWS: {
                // Row-based ops: one thread group per row
                uint32_t total_rows = (uint32_t)(node->ne[1] * node->ne[2] * node->ne[3]);
                groups_x = total_rows;
                break;
            }
            case GGML_OP_GROUP_NORM: {
                // One thread group per (batch, group) pair
                uint32_t num_groups = node->op_params[0];
                uint32_t batches = (uint32_t)node->ne[3];
                groups_x = num_groups * batches;
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
                // Split-KV: increase parallelism by splitting KV across groups
                uint32_t N_queries = (uint32_t)node->src[0]->ne[1];
                uint32_t n_heads   = (uint32_t)node->src[0]->ne[2];
                uint32_t batch     = (uint32_t)node->src[0]->ne[3];
                uint32_t N_kv      = (uint32_t)node->src[1]->ne[1];
                uint32_t n_kv_heads = (uint32_t)node->src[1]->ne[2];

                // GQA fold: when multiple Q-heads share one KV-head, launch
                // one workgroup per kv_head and have it process all gqa_ratio
                // Q-heads. This shares K/V VRAM loads across the gqa_ratio
                // dot products (e.g. 3x bandwidth reduction for SmolVLM2).
                // Falls back to per-Q-head dispatch when gqa_ratio == 1 or
                // the ratio exceeds MAX_GQA in the shader (currently 8).
                constexpr uint32_t MAX_GQA = 8;
                uint32_t gqa_ratio = 1;
                bool gqa_fold = false;
                // GQA-folded FA shader is opt-in pending performance tuning.
                // The current implementation regresses vs the per-head path on
                // Intel Arc (likely register pressure + per-g barriers).
                static const bool gqa_enabled = []{
                    const char * v = std::getenv("GGML_DX12_GQA_FA");
                    return v && v[0] && v[0] != '0';
                }();
                if (gqa_enabled &&
                    n_kv_heads > 0 && n_heads > n_kv_heads &&
                    (n_heads % n_kv_heads) == 0) {
                    uint32_t r = n_heads / n_kv_heads;
                    if (r <= MAX_GQA) {
                        gqa_ratio = r;
                        gqa_fold  = true;
                        // Re-bind to GQA pipeline (key.flags = 1)
                        dx12_pipeline_key gqa_key = key;
                        gqa_key.flags = 1;
                        dx12_pipeline * gqa_pl = bctx->dev->get_or_create_pipeline(gqa_key);
                        if (gqa_pl && gqa_pl->pso) {
                            bctx->cmd_list->SetPipelineState(gqa_pl->pso.Get());
                            bctx->last_pso = gqa_pl->pso.Get();
                            pipeline = gqa_pl;
                        } else {
                            // Pipeline build failed; fall back to non-folded path
                            gqa_fold  = false;
                            gqa_ratio = 1;
                        }
                    }
                }

                uint32_t dispatch_heads = gqa_fold ? n_kv_heads : n_heads;

                // Small-D decode-friendly variants: when D is small we prefer
                // a smaller GROUP_SIZE so Pass-3 V accumulation has higher
                // thread utilization. Smaller workgroups also let more
                // workgroups run concurrently on small-wave GPUs (Intel Arc).
                //   D <= 64  → flash_attn_64  (GROUP_SIZE=TILE_KV=64)
                //   D <= 128 → flash_attn_128 (GROUP_SIZE=TILE_KV=128) — covers
                //              ViT (D=72/80) and many Q-heads (D=80/96/128)
                uint32_t head_dim = (uint32_t)node->src[0]->ne[0];
                if (!gqa_fold && key.flags == 0 && head_dim <= 128) {
                    dx12_pipeline_key small_key = key;
                    small_key.flags = (head_dim <= 64) ? 2 : 3;
                    dx12_pipeline * small_pl = bctx->dev->get_or_create_pipeline(small_key);
                    if (small_pl && small_pl->pso) {
                        bctx->cmd_list->SetPipelineState(small_pl->pso.Get());
                        bctx->last_pso = small_pl->pso.Get();
                        pipeline = small_pl;
                    }
                }

                // Heuristic: split when total groups < target to increase GPU utilization.
                // 256 chosen as a safe default for both NVIDIA (RTX 6000 Ada, 142 SMs) and
                // many-EU iGPUs. Larger targets cause excessive splits on small models with
                // many heads (Phi-3 32 heads → 12 splits at 384 vs 8 at 256), regressing
                // F16/Q4_K generation throughput on NVIDIA.
                uint32_t total_groups_no_split = N_queries * dispatch_heads * batch;
                uint32_t target_groups = 256;
                uint32_t n_splits = 1;
                if (total_groups_no_split < target_groups && N_kv > 32) {
                    n_splits = (target_groups + total_groups_no_split - 1) / total_groups_no_split;
                    n_splits = std::min(n_splits, (N_kv + 31) / 32);  // min 32 KV per split
                    n_splits = std::min(n_splits, (uint32_t)32);      // cap at 32 splits
                }

                // op_params[15]: low 16 bits = n_splits, high 16 bits = gqa_ratio.
                // Always pack so flash_attn.hlsl, flash_attn_gqa.hlsl, and the
                // reduce shader can use the same convention (they all mask low16).
                params.op_params[15] = (n_splits & 0xFFFFu) | ((gqa_ratio & 0xFFFFu) << 16);
                // Re-upload params since we modified op_params
                bctx->cmd_list->SetComputeRoot32BitConstants(0, (uint32_t)(sizeof(params) / 4), &params, 0);

                groups_x = N_queries;
                groups_y = dispatch_heads;
                groups_z = batch * n_splits;

                // Bind temp buffer for split-KV (allocated eagerly at the
                // top of graph_compute -- this branch is the rare fallback
                // for the case where pre-allocation failed).
                if (n_splits > 1) {
                    if (!bctx->dev->splitkv_temp) {
                        bctx->dev->splitkv_temp = dx12_create_buffer(bctx->dev, dx12_device::SPLITKV_TEMP_SIZE);
                    }
                    if (bctx->dev->splitkv_temp) {
                        bctx->cmd_list->SetComputeRootUnorderedAccessView(6, bctx->dev->splitkv_temp->GetGPUVirtualAddress());
                    }
                }
                break;
            }
            case GGML_OP_IM2COL: {
                // IM2COL: one thread per output element (no paired F16, uses store_auto)
                uint32_t total_elements = (uint32_t)(ggml_nelements(node));
                groups_x = (total_elements + 255) / 256;
                break;
            }
            case GGML_OP_MUL_MAT_ID: {
                // Flat one-output-per-thread shader; split into 2D dispatch if
                // the output exceeds D3D12's per-dimension group limit.
                uint32_t total_elements = (uint32_t)(ggml_nelements(node));
                uint32_t total_groups = (total_elements + 255) / 256;
                if (total_groups > 65535) {
                    groups_x = 65535;
                    groups_y = (total_groups + 65534) / 65535;
                } else {
                    groups_x = total_groups;
                }
                break;
            }
            case GGML_OP_SSM_CONV: {
                // One thread per output element (i1, i2, i3) over [nr, n_t, n_s]
                uint32_t total_elements = (uint32_t)(node->ne[0] * node->ne[1] * node->ne[2]);
                groups_x = (total_elements + 255) / 256;
                break;
            }
            case GGML_OP_GATED_DELTA_NET: {
                // Dispatch (H, n_seqs, S_v) — one workgroup per (head, seq, column).
                const struct ggml_tensor * src_v = node->src[2];
                groups_x = (uint32_t)src_v->ne[1]; // H
                groups_y = (uint32_t)src_v->ne[3]; // n_seqs
                groups_z = (uint32_t)src_v->ne[0]; // S_v
                break;
            }
            case GGML_OP_SSM_SCAN: {
                // Vulkan-style: groups_x = ceil(n_head*head_dim / num_subgroups), groups_y = n_seq
                const struct ggml_tensor * src0 = node->src[0];
                const struct ggml_tensor * src1 = node->src[1];
                const uint32_t d_state  = (uint32_t)src0->ne[0];
                const uint32_t head_dim = (uint32_t)src0->ne[1];
                const uint32_t n_head   = (uint32_t)src1->ne[1];
                const uint32_t n_seq    = (uint32_t)src1->ne[3];
                const uint32_t wave     = bctx->dev->wave_size ? bctx->dev->wave_size : 32u;
                const uint32_t num_subgroups = d_state / wave;
                groups_x = (n_head * head_dim + num_subgroups - 1) / num_subgroups;
                groups_y = n_seq;
                groups_z = 1;
                break;
            }
            default: {
                // Element-wise: one thread per element
                // For paired F16 output: halve dispatch since each thread handles 2 elements.
                // ONLY ops whose shaders implement store_f16_pair() can use this optimization.
                // Adding an op here without paired-store support in the shader writes only
                // half the output (other half = uninitialized garbage).
                // Currently supported: ADD, SUB, MUL.
                // (CPY and SET_ROWS handle pairing in their own dispatch geometry above.)
                // For most elementwise ops, dispatch one thread per dst element.
                // SET_ROWS is special: dst is the full KV cache (e.g. 256K rows
                // × 1024 cols = 268M elements) but src0 only contains the new
                // rows to write (typically 1 per gen step = 1024 elements).
                // The shader's early-exit (`if (idx >= src0_total) return;`)
                // means oversizing dispatch wastes 100,000× the work — so size
                // by src0_nelements for SET_ROWS instead.
                uint32_t total_elements;
                if (node->op == GGML_OP_SET_ROWS && node->src[0]) {
                    total_elements = (uint32_t)ggml_nelements(node->src[0]);
                } else {
                    total_elements = (uint32_t)ggml_nelements(node);
                }
                // Only ops whose shaders implement store_f16_pair() are safe.
                // For CPY/DUP/CONT, cpy.hlsl pairs only when src0 is also
                // contiguous F16 along dim0, so mirror that predicate here.
                // SET_ROWS pairs based on dst alone (shader reads src as F32).
                bool op_pairs_dst_only = (node->op == GGML_OP_ADD ||
                                          node->op == GGML_OP_SUB ||
                                          node->op == GGML_OP_MUL ||
                                          node->op == GGML_OP_SET_ROWS);
                bool op_pairs_cpy = (node->op == GGML_OP_CPY ||
                                     node->op == GGML_OP_DUP ||
                                     node->op == GGML_OP_CONT) &&
                                    node->src[0] &&
                                    node->src[0]->type == GGML_TYPE_F16 &&
                                    node->src[0]->nb[0] == ggml_type_size(node->src[0]->type) &&
                                    (node->src[0]->ne[0] & 1) == 0;
                bool paired_f16 = (op_pairs_dst_only || op_pairs_cpy) &&
                                  node->type == GGML_TYPE_F16 &&
                                  node->nb[0] == 2 &&
                                  (node->ne[0] & 1) == 0 &&
                                  (dx12_tensor_offset(node) & 3) == 0 &&
                                  (node->nb[1] & 3) == 0;
                if (paired_f16) total_elements /= 2;
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

        if (do_profile && prof_idx + 2 <= prof_capacity) {
            // Record start timestamp into query heap
            bctx->cmd_list->EndQuery(prof_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, prof_idx);
        }

        LARGE_INTEGER t0, t1, freq;
        if (do_profile) { QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&t0); }

        // Determine the effective destination tensor (accounting for fusion)
        struct ggml_tensor * dst_tensor = fused_5way_set_rows ? fused_5way_set_rows :
                                          (fused_rope_after_rms ? fused_rope_after_rms :
                                          (fused_mul_node ? fused_mul_node :
                                          (fused_bias_add ? fused_bias_add : 
                                          (fused_rope_set_rows ? fused_rope_set_rows :
                                          (fused_mmv_glu_glu ? fused_mmv_glu_glu : node)))));

        // Dependency-tracked UAV barriers
        // Only insert when the current dispatch reads a tensor written by a previous unsynced dispatch.
        {
            bool need_barrier = false;

            // Conservative-by-default: always barrier before SET_ROWS / FA /
            // fused ROPE+SET_ROWS (KV cache views overlap and the alias-aware
            // dependency tracker can miss those write hazards).  Set
            // DX12_RELAXED_KV_BARRIERS=1 to skip the conservative path and rely
            // purely on dependency tracking for these ops -- saves one UAV
            // barrier per attention block on dispatch-bound graphs.
            static const bool relaxed_kv = (getenv("DX12_RELAXED_KV_BARRIERS") != nullptr);
            if (!relaxed_kv && (node->op == GGML_OP_SET_ROWS || node->op == GGML_OP_FLASH_ATTN_EXT || fused_rope_set_rows || fused_5way_set_rows)) {
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
                // Quantize-cache: invalidate ONLY when the cached src1 was one
                // of the unsynced writes being flushed -- otherwise the cached
                // quantized data is still valid (Vulkan tracks this with
                // per-scratch prealloc_*_need_sync flags; we use the unified
                // unsynced_writes set, so we can be precise as long as we do
                // the lookup *before* clearing the set).
                if (bctx->last_q8_1_src_id != 0 &&
                    unsynced_writes.count(bctx->last_q8_1_src_id)) {
                    bctx->last_q8_1_src_id = 0;
                }
                unsynced_writes.clear();
            }
        }

        // dp4a path: quantize src1 to Q8_1 before the main MUL_MAT dispatch
        if (use_dp4a) {
            // Total F32 elements to quantize: K * M * ne2 * ne3
            uint32_t K = (uint32_t)node->src[1]->ne[0];
            uint32_t total_src1_elements = (uint32_t)ggml_nelements(node->src[1]);
            uint32_t num_q8_blocks = total_src1_elements / 32;
            size_t q8_1_size = (size_t)num_q8_blocks * 36;

            // Ensure scratch buffer is large enough
            if (q8_1_size > bctx->q8_1_scratch_size) {
                // CRITICAL: any dispatches already recorded on the open command
                // list reference the OLD scratch buffer's GPU VA. Releasing it
                // before submission causes page faults / TDR. Retain the old
                // resource until graph_compute completes (synchronize() drains
                // q8_1_scratch_retired after wait_for_gpu).
                if (bctx->q8_1_scratch) {
                    bctx->q8_1_scratch_retired.push_back(bctx->q8_1_scratch);
                }
                bctx->q8_1_scratch.Reset();
                D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_RESOURCE_DESC rd = {};
                rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                rd.Width = q8_1_size;
                rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
                rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count = 1;
                rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                bctx->dev->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&bctx->q8_1_scratch));
                bctx->q8_1_scratch_size = q8_1_size;
                // Scratch reallocation invalidates the cache (different VA).
                bctx->last_q8_1_src_id = 0;
                bctx->last_q8_1_size   = 0;
            }

            // Quantize-cache: when consecutive MUL_MATs share src1 (e.g. Q/K/V
            // projections all reading post-RMS_NORM_MUL output, or gate/up
            // reading post-attention-RMS), skip the quantize+barrier and reuse
            // the prior quantized data in q8_1_scratch.  Cache is invalidated
            // by `reset_binding_cache()` (cmd-list flush) and by the barrier
            // dispatch above (which clears unsynced_writes -- we mirror that).
            uint32_t this_src_off = (uint32_t)dx12_tensor_offset(node->src[1]);
            bool reuse_q8_1 = (bctx->last_q8_1_src_id == (uintptr_t)node->src[1] &&
                               bctx->last_q8_1_src_off == this_src_off &&
                               bctx->last_q8_1_size == (uint32_t)q8_1_size);

            // Dispatch quantize_q8_1 shader (skipped on reuse).  Cache the
            // pipeline pointer on the device the first time we look it up;
            // the key is a compile-time constant so this runs at most once.
            if (!bctx->dev->quantize_q8_1_pipeline) {
                dx12_pipeline_key q_key = {};
                q_key.op = GGML_OP_NONE;
                q_key.flags = 99;
                bctx->dev->quantize_q8_1_pipeline = bctx->dev->get_or_create_pipeline(q_key);
            }
            dx12_pipeline * q_pipeline = bctx->dev->quantize_q8_1_pipeline;
            if (q_pipeline && q_pipeline->pso) {
                if (!reuse_q8_1) {
                    bctx->cmd_list->SetPipelineState(q_pipeline->pso.Get());
                    bctx->last_pso = q_pipeline->pso.Get();

                    // Set params: src0_offset = src1's offset, dst_offset = 0
                    dx12_shader_params q_params = {};
                    q_params.src0_offset = this_src_off;
                    q_params.dst_offset = 0;
                    bctx->cmd_list->SetComputeRoot32BitConstants(0, 30, &q_params, 0);

                    // Bind src1 as src0 (input to quantize), scratch as dst
                    bctx->cmd_list->SetComputeRootShaderResourceView(1, src1_res->GetGPUVirtualAddress());
                    bctx->cmd_list->SetComputeRootUnorderedAccessView(3, bctx->q8_1_scratch->GetGPUVirtualAddress());
                    bctx->last_src0_va = src1_res->GetGPUVirtualAddress();
                    bctx->last_dst_va = bctx->q8_1_scratch->GetGPUVirtualAddress();

                    bctx->cmd_list->Dispatch(num_q8_blocks, 1, 1);

                    // Barrier before MUL_MAT reads the quantized data
                    D3D12_RESOURCE_BARRIER barrier = {};
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                    barrier.UAV.pResource = nullptr;
                    bctx->cmd_list->ResourceBarrier(1, &barrier);

                    // Update cache
                    bctx->last_q8_1_src_id   = (uintptr_t)node->src[1];
                    bctx->last_q8_1_src_off  = this_src_off;
                    bctx->last_q8_1_size     = (uint32_t)q8_1_size;
                    bctx->last_q8_1_src_va   = src1_res->GetGPUVirtualAddress();
                }

                // Re-bind for the MUL_MAT dispatch
                bctx->cmd_list->SetPipelineState(pipeline->pso.Get());
                bctx->last_pso = pipeline->pso.Get();
                bctx->cmd_list->SetComputeRootShaderResourceView(1, src0_res->GetGPUVirtualAddress());
                bctx->last_src0_va = src0_res->GetGPUVirtualAddress();
                // Bind Q8_1 scratch as src1 (quantized input)
                bctx->cmd_list->SetComputeRootShaderResourceView(2, bctx->q8_1_scratch->GetGPUVirtualAddress());
                bctx->last_src1_va = bctx->q8_1_scratch->GetGPUVirtualAddress();
                if (dst_res) {
                    bctx->cmd_list->SetComputeRootUnorderedAccessView(3, dst_res->GetGPUVirtualAddress());
                    bctx->last_dst_va = dst_res->GetGPUVirtualAddress();
                }
                // Update params for Q8_1 addressing
                params.src1_offset = 0;  // scratch buffer starts at 0
                // ne10/ne11/ne12/ne13 stay as original (shader uses them for flat row calc)
                bctx->cmd_list->SetComputeRoot32BitConstants(0, num_constants, &params, 0);
            }
        }

        // dp4a matvec path: quantize src1 to Q8_1 before Q4_K matvec dispatch
        if (use_dp4a_matvec) {
            uint32_t total_src1_elements = (uint32_t)ggml_nelements(node->src[1]);
            uint32_t num_q8_blocks = total_src1_elements / 32;
            size_t q8_1_size = (size_t)num_q8_blocks * 36;

            if (q8_1_size > bctx->q8_1_scratch_size) {
                // CRITICAL: see comment above — retain old buffer until graph completes.
                if (bctx->q8_1_scratch) {
                    bctx->q8_1_scratch_retired.push_back(bctx->q8_1_scratch);
                }
                bctx->q8_1_scratch.Reset();
                D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_RESOURCE_DESC rd = {};
                rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                rd.Width = q8_1_size;
                rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
                rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count = 1;
                rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                bctx->dev->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&bctx->q8_1_scratch));
                bctx->q8_1_scratch_size = q8_1_size;
                bctx->last_q8_1_src_id = 0;
                bctx->last_q8_1_size   = 0;
            }

            // Quantize-cache (see use_dp4a path above for details)
            uint32_t this_src_off = (uint32_t)dx12_tensor_offset(node->src[1]);
            bool reuse_q8_1 = (bctx->last_q8_1_src_id == (uintptr_t)node->src[1] &&
                               bctx->last_q8_1_src_off == this_src_off &&
                               bctx->last_q8_1_size == (uint32_t)q8_1_size);

            // Cached quantize pipeline pointer (see use_dp4a path)
            if (!bctx->dev->quantize_q8_1_pipeline) {
                dx12_pipeline_key q_key = {};
                q_key.op = GGML_OP_NONE;
                q_key.flags = 99;
                bctx->dev->quantize_q8_1_pipeline = bctx->dev->get_or_create_pipeline(q_key);
            }
            dx12_pipeline * q_pipeline = bctx->dev->quantize_q8_1_pipeline;
            if (q_pipeline && q_pipeline->pso) {
                if (!reuse_q8_1) {
                    bctx->cmd_list->SetPipelineState(q_pipeline->pso.Get());
                    bctx->last_pso = q_pipeline->pso.Get();

                    dx12_shader_params q_params = {};
                    q_params.src0_offset = this_src_off;
                    q_params.dst_offset = 0;
                    bctx->cmd_list->SetComputeRoot32BitConstants(0, 30, &q_params, 0);

                    bctx->cmd_list->SetComputeRootShaderResourceView(1, src1_res->GetGPUVirtualAddress());
                    bctx->cmd_list->SetComputeRootUnorderedAccessView(3, bctx->q8_1_scratch->GetGPUVirtualAddress());
                    bctx->last_src0_va = src1_res->GetGPUVirtualAddress();
                    bctx->last_dst_va = bctx->q8_1_scratch->GetGPUVirtualAddress();

                    bctx->cmd_list->Dispatch(num_q8_blocks, 1, 1);

                    D3D12_RESOURCE_BARRIER barrier = {};
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                    barrier.UAV.pResource = nullptr;
                    bctx->cmd_list->ResourceBarrier(1, &barrier);

                    bctx->last_q8_1_src_id   = (uintptr_t)node->src[1];
                    bctx->last_q8_1_src_off  = this_src_off;
                    bctx->last_q8_1_size     = (uint32_t)q8_1_size;
                    bctx->last_q8_1_src_va   = src1_res->GetGPUVirtualAddress();
                }

                // Re-bind for the Q4_K dp4a matvec dispatch
                bctx->cmd_list->SetPipelineState(pipeline->pso.Get());
                bctx->last_pso = pipeline->pso.Get();
                bctx->cmd_list->SetComputeRootShaderResourceView(1, src0_res->GetGPUVirtualAddress());
                bctx->last_src0_va = src0_res->GetGPUVirtualAddress();
                bctx->cmd_list->SetComputeRootShaderResourceView(2, bctx->q8_1_scratch->GetGPUVirtualAddress());
                bctx->last_src1_va = bctx->q8_1_scratch->GetGPUVirtualAddress();
                if (dst_res) {
                    bctx->cmd_list->SetComputeRootUnorderedAccessView(3, dst_res->GetGPUVirtualAddress());
                    bctx->last_dst_va = dst_res->GetGPUVirtualAddress();
                }
                params.src1_offset = 0;
                bctx->cmd_list->SetComputeRoot32BitConstants(0, num_constants, &params, 0);
            }
        }

        if (is_matvec_dispatch && matvec_row_groups > 32768) {
            // Large vocab logits can exceed D3D12 dispatch/TDR-friendly sizes
            // as one kernel; split by output row-group.  Each chunk is presented
            // to the shader as a local row range by advancing src0/dst offsets,
            // so the matvec shaders keep their normal row indexing semantics.
            constexpr uint32_t MATVEC_CHUNK_GROUPS = 32768;
            const uint32_t rows_per_group = (key.flags == 9 || key.flags == 10 || key.flags == 11 ||
                                             key.flags == 12 || key.flags == 13 || key.flags == 14 ||
                                             key.flags == 15 || key.flags == 16 || key.flags == 17 ||
                                             key.flags == 18 || key.flags == 19 || key.flags == 20 ||
                                             key.flags == 21 || key.flags == 22 || key.flags == 23 ||
                                             key.flags == 24 || key.flags == 25 || key.flags == 26 ||
                                             key.flags == 27) ? 2 :
                                            (key.flags == 28 || key.flags == 29) ? 4 : 1;
            const uint32_t full_ne0 = params.ne0;
            const uint32_t src0_offset_base = params.src0_offset;
            const uint32_t dst_offset_base = params.dst_offset;
            const uint32_t bias_offset_base = params.op_params[1];
            const uint32_t src2_offset_base = params.op_params[1];
            for (uint32_t base_group = 0; base_group < matvec_row_groups; base_group += MATVEC_CHUNK_GROUPS) {
                uint32_t chunk_groups = std::min(MATVEC_CHUNK_GROUPS, matvec_row_groups - base_group);
                const uint32_t base_row = base_group * rows_per_group;
                const uint32_t chunk_rows = std::min(full_ne0 - base_row, chunk_groups * rows_per_group);
                params.ne0 = chunk_rows;
                params.src0_offset = src0_offset_base + base_row * params.nb01;
                params.dst_offset = dst_offset_base + base_row * params.nb0;
                if (params.op_params[0] == 1u) {
                    params.op_params[1] = bias_offset_base + base_row * sizeof(float);
                } else if (key.flags == 24) {
                    params.op_params[1] = src2_offset_base + base_row * params.nb01;
                }
                params.op_params[15] = 0;
                bctx->cmd_list->SetComputeRoot32BitConstants(0, (uint32_t)(sizeof(params) / 4), &params, 0);
                bctx->cmd_list->Dispatch(chunk_groups, 1, groups_z);
            }
        } else {
            if (is_matvec_dispatch && !needs_op_params) {
                uint32_t matvec_ops[16] = {};
                bctx->cmd_list->SetComputeRoot32BitConstants(0, 16, matvec_ops, BASE_PARAMS);
            }
            bctx->cmd_list->Dispatch(groups_x, groups_y, groups_z);
        }

        // Split-KV reduction pass: combine partial results
        // op_params[15] is packed: low16 = n_splits, high16 = gqa_ratio
        if (node->op == GGML_OP_FLASH_ATTN_EXT && (params.op_params[15] & 0xFFFFu) > 1) {
            uint32_t n_splits = params.op_params[15] & 0xFFFFu;

            // UAV barrier between pass 1 and pass 2
            D3D12_RESOURCE_BARRIER barrier = {};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = nullptr;
            bctx->cmd_list->ResourceBarrier(1, &barrier);

            // Switch to reduction pipeline (cached pointer; key is constant)
            if (!bctx->dev->flash_attn_reduce_pipeline) {
                dx12_pipeline_key reduce_key = {};
                reduce_key.op = GGML_OP_FLASH_ATTN_EXT;
                reduce_key.flags = 8;  // flags=8 = split-KV reduction
                bctx->dev->flash_attn_reduce_pipeline = bctx->dev->get_or_create_pipeline(reduce_key);
            }
            dx12_pipeline * reduce_pl = bctx->dev->flash_attn_reduce_pipeline;
            if (reduce_pl && reduce_pl->pso) {
                bctx->cmd_list->SetPipelineState(reduce_pl->pso.Get());
                bctx->last_pso = reduce_pl->pso.Get();
                // Reduction dispatch: original groups without splits
                uint32_t N_queries = (uint32_t)node->src[0]->ne[1];
                uint32_t n_heads   = (uint32_t)node->src[0]->ne[2];
                uint32_t batch     = (uint32_t)node->src[0]->ne[3];
                bctx->cmd_list->Dispatch(N_queries, n_heads, batch);
            }
        }

        // Track this dispatch's output as unsynced
        unsynced_writes.insert((uintptr_t)dst_tensor);

        // For triple fusion, also track the ADD intermediate output as unsynced
        if (fused_add_rms_node) {
            unsynced_writes.insert((uintptr_t)fused_add_rms_node);
        }

        // DX12_DUMP_PER_DISPATCH: capture matching tensors immediately, before
        // workspace pool reuse can clobber them. Match against the dispatched
        // dst tensor *and* the names of fused-away nodes (so e.g. "Qcur-0"
        // resolves to the actual buffer the 3-way RMS+MUL+ROPE shader wrote
        // to). On match, flush + wait + readback (slow — diagnostic only).
        if (dump_name_env && dump_per_dispatch) {
            const ggml_tensor * candidates[8] = { dst_tensor, node,
                fused_rope_after_rms, fused_5way_set_rows, fused_rope_set_rows,
                fused_mul_node, fused_add_rms_node, fused_bias_tensor };
            const ggml_tensor * matched = nullptr;
            for (const ggml_tensor * c : candidates) {
                if (!c || !c->name[0]) continue;
                const char * pat = dump_name_env;
                bool name_match = false;
                while (*pat) {
                    const char * comma = strchr(pat, ',');
                    size_t tlen = comma ? (size_t)(comma - pat) : strlen(pat);
                    if (tlen > 0 && tlen < 64) {
                        char tok[64]; memcpy(tok, pat, tlen); tok[tlen] = 0;
                        if (strstr(c->name, tok)) { name_match = true; break; }
                    }
                    if (!comma) break;
                    pat = comma + 1;
                }
                if (name_match) { matched = c; break; }
            }
            if (matched) {
                const char * suffix = getenv("DX12_DUMP_SUFFIX");
                if (!suffix) suffix = "";
                bctx->close_and_execute();
                bctx->wait_for_gpu();
                bctx->ensure_cmd_list_open();
                // Re-bind PSO + roots — flush cleared the cmd-list state cache.
                bctx->reset_binding_cache();
                dx12_dump_tensor_if_matched(matched, dump_name_env, suffix, dump_call_idx, i);
            }
        }

        // Skip fused nodes
        if (fused_add_rms_node) {
            i += 2;  // skip the RMS_NORM and MUL nodes
        } else if (fused_5way_set_rows) {
            i += 4;  // skip MUL, ROPE, VIEW, SET_ROWS
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
        if (fused_mmv_glu_glu) {
            i += 2;  // skip the gate matvec and SWIGLU split
        }

        if (do_profile && prof_idx + 2 <= prof_capacity) {
            // Record end timestamp into query heap
            bctx->cmd_list->EndQuery(prof_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, prof_idx + 1);
            char keybuf[160];
            int src0t = node->src[0] ? (int)node->src[0]->type : -1;
            uint32_t N = (uint32_t)node->ne[0];
            uint32_t M = (uint32_t)node->ne[1];
            uint32_t K = node->src[0] ? (uint32_t)node->src[0]->ne[0] : 0;
            // For FA, also surface N_kv (src1->ne[1]) and n_heads / n_kv_heads.
            if (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[1]) {
                uint32_t Nkv  = (uint32_t)node->src[1]->ne[1];
                uint32_t nh   = (uint32_t)node->src[0]->ne[2];
                uint32_t nkvh = (uint32_t)node->src[1]->ne[2];
                snprintf(keybuf, sizeof(keybuf),
                         "%-13s fl=%2u D=%4u nq=%5u nh=%3u/%3u nkv=%5u grp=%u",
                         ggml_op_name(node->op), key.flags, K, M, nh, nkvh, Nkv, groups_x);
            } else {
                snprintf(keybuf, sizeof(keybuf), "%-13s s0=%2d fl=%2u K=%5u N=%5u M=%4u grp=%u",
                         ggml_op_name(node->op), src0t, key.flags, K, N, M, groups_x);
            }
            prof_keys.emplace_back(keybuf);
            prof_idx += 2;
            (void)t0; (void)t1; (void)freq;
        }

        // TDR prevention: flush command list periodically to prevent GPU timeout.
        // Prompt: always flush (batch ops are heavy, threshold 24).
        // Generation: only flush for large models (>500 nodes) that risk TDR
        //   on iGPUs. Use a high weight threshold (2000) so only very heavy
        //   models actually flush — empirically SmolLM2/SmolVLM2 accumulate
        //   ~600-700 weight per token and previously flushed ~3x at threshold
        //   200, causing 5-10 tok/s regression from cmd-list ring stalls and
        //   binding-cache wipes. Phi-3 3.8B accumulates ~3500/token and now
        //   flushes ~1x — still well within the TDR window.
        //
        // Pipelining: the cmd-list ring (CMD_RING_SIZE=4) already provides
        //   natural backpressure — `ensure_cmd_list_open` waits on the slot
        //   it's about to reuse, so up to 3 submissions can be in flight at
        //   once.  This is enough to keep the GPU saturated while the CPU
        //   records the next batch.  Previously we called `wait_for_gpu()`
        //   after every flush when total_groups >= 20000 (vision-scale
        //   dispatches like Qwen3-VL CLIP with 6600 patches × 16 heads), but
        //   that drained the queue to zero between every flush and produced
        //   a visible 100→0→100 ping-pong on the GPU monitor.  The ring's
        //   3-deep pipelining keeps each individual submission well under
        //   the 2s TDR window without serializing CPU↔GPU.  Set
        //   DX12_FLUSH_DRAIN=1 to restore the old drain-after-flush behavior
        //   if a specific workload ever revisits TDR territory.
        {
            uint64_t total_groups = (uint64_t)groups_x * groups_y * groups_z;
            int weight = 1;
            if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_FLASH_ATTN_EXT) {
                if (total_groups >= 8000) weight = 16;
                else if (total_groups >= 1000) weight = 4;
                else weight = 2;
            }
            dispatch_weight += weight;

            // Accumulate weight bytes for adaptive streaming.  Use src0 nbytes
            // (weight matrix) which is what Vulkan tracks (ggml-vulkan.cpp:14528).
            if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[0]) {
                uint64_t nb = ggml_nbytes(node->src[0]);
                mul_mat_bytes      += nb;
                total_mul_mat_bytes += nb;
            }

            bool needs_gen_flush = !is_prompt && cgraph->n_nodes > 500;
            int flush_threshold = is_prompt ? 24 : 2000;
            const bool bytes_trigger = (bytes_per_submit > 0 && mul_mat_bytes >= bytes_per_submit);
            if ((is_prompt || needs_gen_flush) && dispatch_weight >= flush_threshold) {
                bctx->close_and_execute();
                static const bool flush_drain = (getenv("DX12_FLUSH_DRAIN") != nullptr);
                if (flush_drain && is_prompt && total_groups >= 20000) {
                    bctx->wait_for_gpu();
                }
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                dispatch_weight = 0;
                stream_nodes = 0;
                mul_mat_bytes = 0;
                submit_count++;
                if (submit_count <= 3 && bytes_per_submit > 0) bytes_per_submit *= 2;
            } else if (bytes_trigger ||
                       (stream_threshold > 0 && ++stream_nodes >= stream_threshold)) {
                // Stream-submit: kick the GPU so it can overlap with CPU recording.
                bctx->close_and_execute();
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                stream_nodes = 0;
                dispatch_weight = 0;
                mul_mat_bytes = 0;
                submit_count++;
                if (submit_count <= 3 && bytes_per_submit > 0) bytes_per_submit *= 2;
            }
        }

        // Almost-ready fence: submit the first ~80% of the generation graph so
        // the GPU starts early, and the CPU can OS-sleep on this fence while
        // the early dispatches complete.  Once the fence fires, wait_for_fence
        // switches to a brief spin loop for the remaining ~20% of work, which
        // catches GPU completion within microseconds rather than waiting for
        // a syscall round-trip.
        //
        // This mirrors the Vulkan reference's almost-ready pattern
        // (ggml-vulkan.cpp:14719: `(n_nodes - i) < n_nodes / 5`).  Vulkan has
        // it always-on; we gate by a node-count floor so very tiny graphs
        // (vision encoder warmup etc.) don't pay the extra submit overhead
        // for negligible overlap, and we use a dynamic n_nodes/5 trigger so
        // small Smol-class graphs (~301-330 nodes) aren't excluded by an
        // arbitrary >300 cutoff.  Disable via DX12_NO_ALMOST_READY_FENCE=1.
        static const bool almost_ready_disabled = (getenv("DX12_NO_ALMOST_READY_FENCE") != nullptr);
        const int almost_ready_remaining = cgraph->n_nodes / 5;
        if (!almost_ready_disabled &&
            !is_prompt && bctx->almost_ready_fence == 0 &&
            cgraph->n_nodes >= 80 &&
            (cgraph->n_nodes - i) <= almost_ready_remaining &&
            bctx->cmd_list_open) {
            bctx->close_and_execute();
            bctx->almost_ready_fence = bctx->fence_value;
            bctx->ensure_cmd_list_open();
            bctx->reset_binding_cache();
        }
    }

    // Save grand total for next call's bytes-per-submit threshold heuristic.
    bctx->last_total_mul_mat_bytes = total_mul_mat_bytes;

    // Drain retired q8_1_scratch buffers that were displaced by reallocation.
    // They must outlive any in-flight dispatch that referenced their VA.
    // We close+wait here only if there's actually something to free.
    if (!bctx->q8_1_scratch_retired.empty()) {
        if (bctx->cmd_list_open) {
            bctx->close_and_execute();
        }
        bctx->wait_for_gpu();
        bctx->q8_1_scratch_retired.clear();
    }

    // Keep the command list open — UAV barriers between dispatches ensure
    // correct ordering within a single command list.  The list is flushed
    // in synchronize(), get_tensor(), or set_tensor() when results are
    // actually needed.  This avoids 300+ close/execute/wait round-trips
    // per generation that were pegging the CPU at 100%.

    // Dump profiling results: resolve query heap and aggregate per op
    if (do_profile && prof_idx > 0) {
        // Resolve query data into the readback buffer, then flush+wait so
        // we can map and read the GPU timestamps.
        bctx->cmd_list->ResolveQueryData(prof_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                          0, prof_idx, prof_readback.Get(), 0);
        bctx->close_and_execute();
        bctx->wait_for_gpu();
        bctx->ensure_cmd_list_open();
        bctx->reset_binding_cache();

        uint64_t * ts = nullptr;
        D3D12_RANGE rr = { 0, (size_t)prof_idx * sizeof(uint64_t) };
        HRESULT hr = prof_readback->Map(0, &rr, (void **)&ts);
        if (SUCCEEDED(hr) && ts) {
            for (size_t k = 0; k < prof_keys.size(); k++) {
                uint64_t t_start = ts[k * 2];
                uint64_t t_end   = ts[k * 2 + 1];
                if (t_end < t_start) continue;
                double ms = (double)(t_end - t_start) * 1000.0 / (double)prof_freq;
                op_times[prof_keys[k]] += ms;
                op_counts[prof_keys[k]] += 1;
            }
            D3D12_RANGE wr = { 0, 0 };
            prof_readback->Unmap(0, &wr);
        }
    }
    if (do_profile && !op_times.empty()) {
        fprintf(stderr, "\n=== DX12 Profile (graph #%d) ===\n", profile_graph);
        std::vector<std::pair<double, std::string>> sorted;
        double total = 0;
        for (auto & kv : op_times) { sorted.push_back({kv.second, kv.first}); total += kv.second; }
        std::sort(sorted.rbegin(), sorted.rend());
        fprintf(stderr, "  %8s  %5s %5s  %s\n", "ms", "%", "n", "op");
        for (auto & p : sorted) {
            if (p.first > 0.01) {
                uint32_t n = op_counts[p.second];
                fprintf(stderr, "  %8.3f  %5.1f  %4u  %s\n", p.first, p.first/total*100, n, p.second.c_str());
            }
        }
        fprintf(stderr, "  %8.3f  TOTAL\n", total);
    }

    if (dx12_trace) {
        fprintf(stderr, "[DX12_TRACE] graph_compute #%d exit: success\n", trace_call);
        fflush(stderr);
    }

    // DX12_DUMP_TENSOR: post-dispatch tensor dump diagnostic. Set env to a
    // comma-separated list of name substrings (e.g. "Qcur-0,Kcur-0"); writes
    // the bytes of any matching node to a file. Used to root-cause
    // fused-vs-unfused divergence by diffing dumps from two runs (e.g. one
    // with fusion enabled, one with it gated off via DX12_NO_FUSE_*). Set
    // DX12_DUMP_SUFFIX to disambiguate output files between runs.
    //
    // For tensors that live in the workspace pool and may be aliased / reused
    // by later ops within the same graph_compute (most non-cache, non-output
    // intermediates), the end-of-graph dump captures stale memory. Use
    // DX12_DUMP_PER_DISPATCH=1 to also capture each matching tensor
    // immediately after its producing dispatch (slow: causes a flush + GPU
    // wait per match) — see the dispatch loop above.
    if (const char * dump_name = getenv("DX12_DUMP_TENSOR")) {
        const char * suffix = getenv("DX12_DUMP_SUFFIX");
        if (!suffix) suffix = "";
        bctx->close_and_execute();
        bctx->wait_for_gpu();
        bctx->ensure_cmd_list_open();
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            dx12_dump_tensor_if_matched(cgraph->nodes[i], dump_name, suffix, dump_call_idx, i);
        }
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
    const char * localappdata = getenv("LOCALAPPDATA");
    snprintf(cache_path, sizeof(cache_path), "%s/.ggml_dx12_tune_%04X_%04X.txt",
             localappdata ? localappdata : ".",
             adapter_desc.VendorId, adapter_desc.DeviceId);

    FILE * f = fopen(cache_path, "r");
    if (f) {
        int ver = 0, q4kdp = 0, q5kdp = 0, f16mr256 = 0;
        unsigned int f16mr_kthresh = 0xFFFFFFFFu;
        if (fscanf(f, "v=%d q4k_dp4a_32=%d q5k_dp4a_32=%d f16_mr_256=%d f16_mr_k_thresh=%u",
                   &ver, &q4kdp, &q5kdp, &f16mr256, &f16mr_kthresh) == 5
            && ver == TUNE_VERSION) {
            q4k_dp4a_use_32 = (q4kdp != 0);
            q5k_dp4a_use_32 = (q5kdp != 0);
            f16_mr_use_256  = (f16mr256 != 0);
            f16_mr_k_256_threshold = (uint32_t)f16mr_kthresh;
            fclose(f);
            DX12_LOG_INFO("Auto-tune v%d loaded: Q4_K_dp4a=%s Q5_K_dp4a=%s F16_mr=%s (K>=%u uses 256t)\n", ver,
                          q4k_dp4a_use_32 ? "32t" : "256t",
                          q5k_dp4a_use_32 ? "32t" : "256t",
                          f16_mr_use_256  ? "256t" : "32t",
                          (unsigned)f16_mr_k_256_threshold);
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
        rd.Width = 16 * 1024 * 1024;  // 16MB — covers worst-case (F16 mr, NUM_ROWS=2, N=256, K=8192) reads
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
        HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        fence->SetEventOnCompletion(1, event);
        compute_queue->Signal(fence.Get(), 1);
        DWORD wait_result = WaitForSingleObject(event, 5000);
        CloseHandle(event);

        if (wait_result == WAIT_TIMEOUT) {
            DX12_LOG_WARN("Auto-tune: GPU benchmark timed out\n");
            return UINT64_MAX;
        }

        // Read timestamps
        uint64_t * ts = nullptr;
        D3D12_RANGE range = { ts_start * sizeof(uint64_t), (ts_start + 2) * sizeof(uint64_t) };
        ts_readback->Map(0, &range, (void**)&ts);
        uint64_t dt = ts[ts_start + 1] - ts[ts_start];
        ts_readback->Unmap(0, nullptr);
        return dt;
    };

    // Benchmark Q5_0 matvec: 32 threads vs 256 threads
    // Test with K=576 (SmolVLM2-like), K=3072 (Phi-3 attn / Smol FFN), and
    // K=8192 (Phi-3 FFN-down). Bigger K matters because the per-thread work in
    // the 256t variant only becomes profitable once K is large enough to
    // amortize the extra waves' setup; without an 8192 sample the crossover
    // estimator is extrapolating well past its measured range.
    uint32_t test_K[] = { 576, 3072, 8192 };
    uint32_t test_N = 256;  // number of output rows to benchmark

    // Capture per-K timings to write into the cache file as diagnostic comments.
    // Layout: for each variant we store [K0_256, K0_32, K1_256, K1_32, ...].
    constexpr size_t NK = sizeof(test_K) / sizeof(test_K[0]);
    uint64_t q4k_per_k[NK*2]; for (size_t i = 0; i < NK*2; ++i) q4k_per_k[i] = UINT64_MAX;
    uint64_t q5k_per_k[NK*2]; for (size_t i = 0; i < NK*2; ++i) q5k_per_k[i] = UINT64_MAX;
    uint64_t f16_per_k[NK*2]; for (size_t i = 0; i < NK*2; ++i) f16_per_k[i] = UINT64_MAX;

    // Benchmark Q4_K dp4a matvec: 256 threads (default) vs 32 threads
    // Only meaningful if the device supports SM 6.4 / dp4a
    uint64_t q4k_256_total = 0, q4k_32_total = 0;
    if (dp4a_supported) {
        for (size_t ki = 0; ki < NK; ++ki) {
            uint32_t K = test_K[ki];
            dx12_pipeline_key key256 = {}; key256.op = GGML_OP_MUL_MAT; key256.src0_type = GGML_TYPE_Q4_K; key256.flags = 10;
            dx12_pipeline_key key32  = {}; key32.op  = GGML_OP_MUL_MAT; key32.src0_type  = GGML_TYPE_Q4_K; key32.flags  = 13;

            uint64_t t256 = bench_pipeline(key256, K, test_N, 0);
            uint64_t t32  = bench_pipeline(key32,  K, test_N, 2);

            q4k_per_k[ki*2 + 0] = t256;
            q4k_per_k[ki*2 + 1] = t32;
            if (t256 != UINT64_MAX) q4k_256_total += t256;
            if (t32  != UINT64_MAX) q4k_32_total  += t32;

            DX12_LOG_INFO("  Q4_K_dp4a K=%u: 256t=%llu 32t=%llu ticks\n", K,
                          (unsigned long long)t256, (unsigned long long)t32);
        }
        q4k_dp4a_use_32 = (q4k_32_total < q4k_256_total && q4k_32_total > 0);
    }

    // Benchmark Q5_K dp4a matvec: 256 threads (default) vs 32 threads
    uint64_t q5k_dp4a_256_total = 0, q5k_dp4a_32_total = 0;
    if (dp4a_supported) {
        for (size_t ki = 0; ki < NK; ++ki) {
            uint32_t K = test_K[ki];
            dx12_pipeline_key key256 = {}; key256.op = GGML_OP_MUL_MAT; key256.src0_type = GGML_TYPE_Q5_K; key256.flags = 14;
            dx12_pipeline_key key32  = {}; key32.op  = GGML_OP_MUL_MAT; key32.src0_type  = GGML_TYPE_Q5_K; key32.flags  = 16;

            uint64_t t256 = bench_pipeline(key256, K, test_N, 0);
            uint64_t t32  = bench_pipeline(key32,  K, test_N, 2);

            q5k_per_k[ki*2 + 0] = t256;
            q5k_per_k[ki*2 + 1] = t32;
            if (t256 != UINT64_MAX) q5k_dp4a_256_total += t256;
            if (t32  != UINT64_MAX) q5k_dp4a_32_total  += t32;

            DX12_LOG_INFO("  Q5_K_dp4a K=%u: 256t=%llu 32t=%llu ticks\n", K,
                          (unsigned long long)t256, (unsigned long long)t32);
        }
        q5k_dp4a_use_32 = (q5k_dp4a_32_total < q5k_dp4a_256_total && q5k_dp4a_32_total > 0);
    }

    // Benchmark F16 matvec: 256 threads (mr, flags=11) vs 32 threads (mr32, flags=12).
    // Default for non-AMD-wave64 is 32t; this autotune may flip to 256t on
    // NVIDIA wave32 or Intel Arc when their occupancy benefits outweigh the
    // partial-wave waste at small K.
    uint64_t f16_mr_256_total = 0, f16_mr_32_total = 0;
    {
        for (size_t ki = 0; ki < NK; ++ki) {
            uint32_t K = test_K[ki];
            dx12_pipeline_key key256 = {}; key256.op = GGML_OP_MUL_MAT; key256.src0_type = GGML_TYPE_F16; key256.flags = 11;
            dx12_pipeline_key key32  = {}; key32.op  = GGML_OP_MUL_MAT; key32.src0_type  = GGML_TYPE_F16; key32.flags  = 12;

            uint64_t t256 = bench_pipeline(key256, K, test_N, 0);
            uint64_t t32  = bench_pipeline(key32,  K, test_N, 2);

            f16_per_k[ki*2 + 0] = t256;
            f16_per_k[ki*2 + 1] = t32;
            if (t256 != UINT64_MAX) f16_mr_256_total += t256;
            if (t32  != UINT64_MAX) f16_mr_32_total  += t32;

            DX12_LOG_INFO("  F16_mr K=%u: 256t=%llu 32t=%llu ticks\n", K,
                          (unsigned long long)t256, (unsigned long long)t32);
        }
        // Pick K-aware threshold via linear-interp crossover between the
        // smallest and largest tested K (test_K[0] and test_K[NK-1]).
        // When 256t wins both → threshold=0; when 32t wins both → UINT32_MAX.
        f16_mr_use_256 = (f16_mr_256_total < f16_mr_32_total && f16_mr_256_total > 0);
        f16_mr_k_256_threshold = 0xFFFFFFFFu;
        if (NK >= 2) {
            constexpr size_t lo = 0;
            constexpr size_t hi = NK - 1;
            uint64_t a256 = f16_per_k[lo*2 + 0];
            uint64_t a32  = f16_per_k[lo*2 + 1];
            uint64_t b256 = f16_per_k[hi*2 + 0];
            uint64_t b32  = f16_per_k[hi*2 + 1];
            if (a256 != UINT64_MAX && a32 != UINT64_MAX &&
                b256 != UINT64_MAX && b32 != UINT64_MAX) {
                bool a_256_wins = (a256 < a32);
                bool b_256_wins = (b256 < b32);
                if (a_256_wins && b_256_wins) {
                    f16_mr_k_256_threshold = 0;
                } else if (!a_256_wins && !b_256_wins) {
                    f16_mr_k_256_threshold = 0xFFFFFFFFu;
                } else {
                    // Split decision — interpolate crossover K. Solve
                    //   a256 + (b256-a256)*x = a32 + (b32-a32)*x
                    // for x in [0,1] over K range [test_K[lo], test_K[hi]].
                    double da = (double)a256 - (double)a32;
                    double db = (double)b32  - (double)b256;
                    double denom = da + db;
                    double x = (denom > 0) ? (da / denom) : 0.5;
                    if (x < 0.0) x = 0.0;
                    if (x > 1.0) x = 1.0;
                    double K_cross = (double)test_K[lo]
                                   + x * ((double)test_K[hi] - (double)test_K[lo]);
                    f16_mr_k_256_threshold = b_256_wins
                        ? (uint32_t)(K_cross + 0.5)   // small-K=32t, large-K=256t
                        : 0xFFFFFFFFu;                 // small-K=256t, large-K=32t (rare; keep 32t default)
                }
            }
        }
    }

    DX12_LOG_INFO("Auto-tune result: Q4_K_dp4a=%s Q5_K_dp4a=%s F16_mr=%s (K>=%u uses 256t)\n",
                  q4k_dp4a_use_32 ? "32t" : "256t",
                  q5k_dp4a_use_32 ? "32t" : "256t",
                  f16_mr_use_256  ? "256t" : "32t",
                  (unsigned)f16_mr_k_256_threshold);

    // Save to cache (with per-K diagnostic comments after the result line)
    f = fopen(cache_path, "w");
    if (f) {
        fprintf(f, "v=%d q4k_dp4a_32=%d q5k_dp4a_32=%d f16_mr_256=%d f16_mr_k_thresh=%u\n",
                TUNE_VERSION,
                q4k_dp4a_use_32 ? 1 : 0,
                q5k_dp4a_use_32 ? 1 : 0,
                f16_mr_use_256  ? 1 : 0,
                (unsigned)f16_mr_k_256_threshold);
        for (size_t ki = 0; ki < NK; ++ki) {
            fprintf(f, "# Q4_K_dp4a K=%u: 256t=%llu 32t=%llu ticks\n",
                    test_K[ki],
                    (unsigned long long)q4k_per_k[ki*2 + 0],
                    (unsigned long long)q4k_per_k[ki*2 + 1]);
        }
        for (size_t ki = 0; ki < NK; ++ki) {
            fprintf(f, "# Q5_K_dp4a K=%u: 256t=%llu 32t=%llu ticks\n",
                    test_K[ki],
                    (unsigned long long)q5k_per_k[ki*2 + 0],
                    (unsigned long long)q5k_per_k[ki*2 + 1]);
        }
        for (size_t ki = 0; ki < NK; ++ki) {
            fprintf(f, "# F16_mr     K=%u: 256t=%llu 32t=%llu ticks\n",
                    test_K[ki],
                    (unsigned long long)f16_per_k[ki*2 + 0],
                    (unsigned long long)f16_per_k[ki*2 + 1]);
        }
        fclose(f);
    }
}

// ---------------------------------------------------------------------------
// R13 — Graph reorder pre-pass
// ---------------------------------------------------------------------------
//
// Greedy reorder of cgraph->nodes that pulls fusion-eligible neighbours
// adjacent so the record path in dx12_graph_compute can match its existing
// patterns more often.  Mirrors Vulkan's ggml_vk_graph_optimize, restricted
// to the patterns DX12 actually fuses today:
//
//   - ADD + RMS_NORM + MUL                         (DX12_FUSE_ADD_RMS_MUL)
//   - RMS_NORM + MUL                               (DX12_FUSE_RMS_MUL)
//   - RMS_NORM + MUL + ROPE [+ VIEW + SET_ROWS]    (DX12_FUSE_RMS_MUL_ROPE3/5)
//   - ROPE + VIEW + SET_ROWS                       (DX12_FUSE_ROPE_SET_ROWS)
//   - MUL_MAT(M=1) + ADD                           (matvec + bias)
//   - MUL_MAT(W_gate) + MUL_MAT(W_up) + GLU(SwiGLU split)  (R9, DX12_FUSE_MMV_GLU_SPLIT)
//
// The reorder also preserves the topk-MoE op sequences upstream so a future
// MoE fused kernel slots in cleanly; DX12 doesn't fuse MoE today but we do
// not want to spread its 6..11 nodes across other dispatches.
//
// Off by default — opt-in with DX12_ENABLE_GRAPH_OPTIMIZE=1.  Reorder is
// topologically safe (every node's srcs still appear earlier in the array)
// but does not track implicit write-aliasing through SET_ROWS / CPY into
// shared buffers (e.g. gemma-3n's per-layer-input cache, AltUp scaler), so
// some architectures generate incoherent output when the reorder is active.
// The cheap topological-order verifier below stays on whenever the reorder
// runs and falls back to the original order if any src/use ordering breaks.

namespace {

// MoE topk patterns — kept identical to Vulkan so a future port matches.
constexpr std::initializer_list<ggml_op> dx12_topk_moe_early_softmax_norm{
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
    GGML_OP_VIEW,     GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
    GGML_OP_SUM_ROWS, GGML_OP_CLAMP,    GGML_OP_DIV,
    GGML_OP_RESHAPE };

constexpr std::initializer_list<ggml_op> dx12_topk_moe_sigmoid_norm_bias{
    GGML_OP_UNARY,    GGML_OP_RESHAPE,  GGML_OP_ADD,
    GGML_OP_ARGSORT,  GGML_OP_VIEW,     GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE,  GGML_OP_SUM_ROWS, GGML_OP_CLAMP,
    GGML_OP_DIV,      GGML_OP_RESHAPE };

constexpr std::initializer_list<ggml_op> dx12_topk_moe_early_softmax{
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
    GGML_OP_VIEW,     GGML_OP_GET_ROWS };

constexpr std::initializer_list<ggml_op> dx12_topk_moe_late_softmax{
    GGML_OP_ARGSORT,  GGML_OP_VIEW,
    GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

inline bool dx12_node_is_empty(const ggml_tensor * n) {
    return n->op == GGML_OP_NONE || n->op == GGML_OP_RESHAPE ||
           n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_VIEW ||
           n->op == GGML_OP_PERMUTE;
}

inline bool dx12_node_is_src_of(const ggml_tensor * dst, const ggml_tensor * src) {
    for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
        if (dst->src[s] == src) return true;
    }
    // implicit dependency through view aliasing
    const ggml_tensor * d = dst->view_src ? dst->view_src : dst;
    const ggml_tensor * s = src->view_src ? src->view_src : src;
    return d == s;
}

} // anonymous namespace

static void dx12_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_UNUSED(backend);
    static const bool enabled = (getenv("DX12_ENABLE_GRAPH_OPTIMIZE") != nullptr);
    if (!enabled || graph->n_nodes <= 1) return;

    auto match_pattern = [&](const std::initializer_list<ggml_op> & pattern,
                             const std::vector<bool> & used,
                             int start) -> bool {
        if (start + (int)pattern.size() > graph->n_nodes) return false;
        for (size_t j = 0; j < pattern.size(); ++j) {
            if (graph->nodes[start + j]->op != pattern.begin()[j] || used[start + j]) {
                return false;
            }
        }
        return true;
    };

    std::vector<ggml_tensor *> new_order;
    new_order.reserve(graph->n_nodes);
    std::vector<bool> used(graph->n_nodes, false);
    std::unordered_set<ggml_tensor *> used_node_set;

    int first_unused = 0;
    while (first_unused < graph->n_nodes) {
        // Preserve MoE topk sequences as a contiguous block.
        auto keep_pattern = [&](const std::initializer_list<ggml_op> & pattern) -> bool {
            if (match_pattern(pattern, used, first_unused)) {
                for (size_t j = 0; j < pattern.size(); ++j) {
                    new_order.push_back(graph->nodes[first_unused + j]);
                    used_node_set.insert(graph->nodes[first_unused + j]);
                    used[first_unused + j] = true;
                }
                while (first_unused < graph->n_nodes && used[first_unused]) {
                    first_unused++;
                }
                return true;
            }
            return false;
        };

        if (keep_pattern(dx12_topk_moe_early_softmax_norm))   continue;
        if (keep_pattern(dx12_topk_moe_sigmoid_norm_bias))    continue;
        if (keep_pattern(dx12_topk_moe_early_softmax))        continue;
        if (keep_pattern(dx12_topk_moe_late_softmax))         continue;

        std::vector<int> current_set;
        current_set.push_back(first_unused);

        // First pass: real (non-empty) nodes that don't depend on intervening
        // unprocessed nodes — except when the dependency itself is a fusion
        // pair we want to preserve.
        const int NUM_TO_CHECK = 20;
        const int last = std::min(first_unused + NUM_TO_CHECK, graph->n_nodes);
        for (int j = first_unused + 1; j < last; ++j) {
            if (used[j])                       continue;
            if (dx12_node_is_empty(graph->nodes[j])) continue;
            if (match_pattern(dx12_topk_moe_early_softmax_norm, used, j) ||
                match_pattern(dx12_topk_moe_sigmoid_norm_bias,  used, j) ||
                match_pattern(dx12_topk_moe_early_softmax,      used, j) ||
                match_pattern(dx12_topk_moe_late_softmax,       used, j)) {
                continue;
            }
            bool ok = true;
            for (int c = first_unused; c < j; ++c) {
                if (used[c]) continue;
                if (!dx12_node_is_src_of(graph->nodes[j], graph->nodes[c])) continue;
                // Allow the fusion-pair exceptions DX12 actually exploits.
                bool back = (j == c + 1 && c == current_set.back());
                ggml_op pc = graph->nodes[c]->op;
                ggml_op pj = graph->nodes[j]->op;
                if (back &&
                    ((pc == GGML_OP_RMS_NORM && pj == GGML_OP_MUL)     ||
                     (pc == GGML_OP_MUL_MAT  && pj == GGML_OP_ADD)     ||
                     (pc == GGML_OP_ADD      && pj == GGML_OP_RMS_NORM))) {
                    continue;
                }
                ok = false;
                break;
            }
            if (!ok) continue;
            current_set.push_back(j);

            int rope_idx = j;

            // Pull a ROPE that consumes RMS+MUL right behind it.
            if (j > 0 &&
                graph->nodes[j]->op   == GGML_OP_MUL &&
                graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                const int rope_last = std::min(j + 15, graph->n_nodes);
                for (int k = j + 1; k < rope_last; ++k) {
                    if (graph->nodes[k]->op == GGML_OP_ROPE &&
                        graph->nodes[k]->src[0] == graph->nodes[j] &&
                        graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                        (graph->nodes[k]->src[2] == nullptr ||
                         graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                        rope_idx = k;
                        current_set.push_back(rope_idx);
                        used[rope_idx] = true;
                        break;
                    }
                }
            }

            // Pull VIEW + SET_ROWS behind a ROPE so the 5-way / ROPE+SET_ROWS
            // fusions in dx12_graph_compute can match.
            if (graph->nodes[rope_idx]->op == GGML_OP_ROPE) {
                int view_idx     = -1;
                int set_rows_idx = -1;
                const int sr_last = std::min(rope_idx + 10, graph->n_nodes);
                for (int k = rope_idx + 1; k < sr_last; ++k) {
                    if (view_idx == -1 &&
                        graph->nodes[k]->op == GGML_OP_VIEW &&
                        graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                        view_idx = k;
                        continue;
                    }
                    if (view_idx != -1 && set_rows_idx == -1 &&
                        graph->nodes[k]->op == GGML_OP_SET_ROWS &&
                        graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                        set_rows_idx = k;
                        break;
                    }
                }
                if (set_rows_idx != -1) {
                    current_set.push_back(view_idx);
                    current_set.push_back(set_rows_idx);
                    used[view_idx]     = true;
                    used[set_rows_idx] = true;
                }
            }

            // R9: pull a sibling MUL_MAT and the consuming GLU(SWIGLU split)
            // adjacent to a matvec we just admitted.  Only triggers when both
            // matvecs share an activation, which is the FFN gate/up topology.
            if (graph->nodes[j]->op == GGML_OP_MUL_MAT &&
                graph->nodes[j]->ne[1] == 1) {
                const ggml_tensor * mm0 = graph->nodes[j];
                int mm1_idx = -1;
                int glu_idx = -1;
                const int mmv_last = std::min(j + 10, graph->n_nodes);
                for (int k = j + 1; k < mmv_last; ++k) {
                    if (used[k]) continue;
                    if (mm1_idx == -1 &&
                        graph->nodes[k]->op == GGML_OP_MUL_MAT &&
                        graph->nodes[k]->ne[1] == 1 &&
                        graph->nodes[k]->src[1] == mm0->src[1]) {
                        mm1_idx = k;
                        continue;
                    }
                    if (mm1_idx != -1 && glu_idx == -1 &&
                        graph->nodes[k]->op == GGML_OP_GLU &&
                        graph->nodes[k]->src[0] == mm0 &&
                        graph->nodes[k]->src[1] == graph->nodes[mm1_idx]) {
                        glu_idx = k;
                        break;
                    }
                }
                if (mm1_idx != -1 && glu_idx != -1) {
                    current_set.push_back(mm1_idx);
                    current_set.push_back(glu_idx);
                    used[mm1_idx] = true;
                    used[glu_idx] = true;
                }
            }
        }

        // Second pass: views/empty nodes whose data is now visible.
        // Skip if it would split a known fusion pair (Vulkan parity).
        if (graph->nodes[current_set.back()]->op != GGML_OP_ADD) {
            for (int j = first_unused + 1; j < last; ++j) {
                if (used[j])                              continue;
                if (!dx12_node_is_empty(graph->nodes[j])) continue;
                bool ok = true;
                for (int c = first_unused; c < j; ++c) {
                    if (used[c]) continue;
                    bool in_set = std::find(current_set.begin(), current_set.end(), c) != current_set.end();
                    if (!in_set && dx12_node_is_src_of(graph->nodes[j], graph->nodes[c])) {
                        ok = false;
                        break;
                    }
                }
                if (ok) current_set.push_back(j);
            }
        }

        for (int c : current_set) {
            new_order.push_back(graph->nodes[c]);
            used_node_set.insert(graph->nodes[c]);
            used[c] = true;
        }
        while (first_unused < graph->n_nodes && used[first_unused]) {
            first_unused++;
        }
    }

    // Defensive: only commit the rewrite if the reorder produced exactly the
    // same nodes (no drops, no duplicates).  If a bug ever leaves new_order
    // short, fall back to the original order rather than corrupting the graph.
    if ((int)new_order.size() == graph->n_nodes) {
        // Topological-order verification: every node's srcs must appear at a
        // strictly earlier index in new_order.  Skip the rewrite (keep the
        // original order) if any violation is detected.  Cheap O(n_nodes *
        // GGML_MAX_SRC) hash lookup; runs only when the optimize path is
        // explicitly enabled via DX12_ENABLE_GRAPH_OPTIMIZE.
        std::unordered_map<ggml_tensor *, int> pos;
        pos.reserve(graph->n_nodes);
        for (int i = 0; i < graph->n_nodes; ++i) {
            pos[new_order[i]] = i;
        }
        bool topo_ok = true;
        for (int i = 0; i < graph->n_nodes && topo_ok; ++i) {
            ggml_tensor * n = new_order[i];
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                ggml_tensor * src = n->src[s];
                if (!src) continue;
                auto it = pos.find(src);
                if (it != pos.end() && it->second >= i) {
                    topo_ok = false;
                    break;
                }
            }
        }
        if (topo_ok) {
            for (int i = 0; i < graph->n_nodes; ++i) {
                graph->nodes[i] = new_order[i];
            }
        }
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

    static const int dx12_trace = (getenv("DX12_TRACE_GRAPH") != nullptr) ? atoi(getenv("DX12_TRACE_GRAPH")) : 0;
    if (dx12_trace) {
        fprintf(stderr, "[DX12_TRACE] synchronize enter (cmd_list_open=%d, fence_value=%llu)\n",
                (int)ctx->cmd_list_open, (unsigned long long)ctx->fence_value);
        fflush(stderr);
    }

    // Submit pending work immediately so GPU can start executing
    // while CPU proceeds to sampling/scheduling.
    if (ctx->cmd_list_open) {
        ctx->close_and_execute();
    }

    // synchronize() is the formal sync primitive — it must guarantee that ALL
    // submitted GPU work is complete before returning, otherwise CPU code that
    // touches tensor memory directly (e.g. sampling reading logits) can race
    // with in-flight dispatches. The almost-ready fence already gave the CPU
    // a head start during graph_compute; here we must wait on the latest fence
    // value (covers both the early submit and the tail submit).
    ctx->wait_for_fence(ctx->fence_value);
    ctx->almost_ready_fence = 0;

    // Flush deferred get_tensor_async memcpys (Vulkan parity).  All recorded
    // CopyBufferRegion → readback staging operations are now complete and the
    // GPU has flushed its writes, so we can safely deliver the data to the
    // caller's buffers.  For UMA fast-path entries, src points directly into
    // the device-mapped buffer and is also safe to read after the fence wait.
    for (auto & m : ctx->pending_get_memcpys) {
        if (m.staging) {
            // Map readback staging with non-empty range to invalidate cache,
            // copy, and unmap.  Persistent mapping is unsafe for READBACK
            // heaps (CPU_PAGE_PROPERTY_WRITE_BACK is cached and Map() is the
            // documented invalidation point — same reasoning as the sync
            // get_tensor path at ggml-dx12.cpp:1349-1359).
            void * mapped = nullptr;
            D3D12_RANGE rr = { 0, m.size };
            HRESULT hr = m.staging->Map(0, &rr, &mapped);
            if (SUCCEEDED(hr) && mapped) {
                memcpy(m.dst, mapped, m.size);
                D3D12_RANGE wr = { 0, 0 };
                m.staging->Unmap(0, &wr);
            }
            // Return staging to pool for reuse.
            ctx->async_readback_pool.push_back(std::move(m.staging));
        } else {
            memcpy(m.dst, m.src, m.size);
        }
    }
    ctx->pending_get_memcpys.clear();

    if (dx12_trace) {
        fprintf(stderr, "[DX12_TRACE] synchronize exit\n");
        fflush(stderr);
    }
}

// ---------------------------------------------------------------------------
// Async tensor I/O (Vulkan parity: ggml-vulkan.cpp:13814-13950)
// ---------------------------------------------------------------------------
//
// These three functions let the ggml scheduler and any caller using the
// _async public APIs queue tensor transfers without forcing a CPU↔GPU
// rendezvous (which is what the synchronous fallback path does — see
// ggml-backend.cpp:260-265 / 274-278: `if iface.set_tensor_async == NULL
// then synchronize() + sync set/get`).  With async I/O implemented:
//
//   - set_tensor_async: queue the upload onto the shared compute cmd_list
//     so it serializes with subsequent dispatches via the natural cmd-list
//     ordering.  No fence wait.
//
//   - get_tensor_async: queue a CopyBufferRegion → readback staging onto
//     the cmd_list, then register a (dst, staging, size) entry that
//     synchronize() flushes after waiting for the fence.  Multiple readbacks
//     amortize a single fence wait.
//
//   - cpy_tensor_async: same-device DX12-to-DX12 → CopyBufferRegion onto
//     the shared cmd_list.  Cross-device → return false, scheduler falls
//     back to host round-trip.
//
// UMA buffers (CPU-mapped via ctx->mapped) get fast paths that avoid the
// staging detour: set_tensor_async direct-writes the mapped buffer, and
// get_tensor_async captures the source pointer for a deferred memcpy after
// fence wait.  These mirror the sync UMA fast paths at lines 1273-1278 and
// 1321-1325 but defer the memcpy on get to satisfy async semantics.

static void async_alloc_readback_staging(dx12_backend_context * ctx,
                                         size_t size,
                                         ComPtr<ID3D12Resource> & out) {
    // Try to reuse a pooled staging buffer of sufficient size.
    for (auto it = ctx->async_readback_pool.begin(); it != ctx->async_readback_pool.end(); ++it) {
        D3D12_RESOURCE_DESC d = (*it)->GetDesc();
        if (d.Width >= size) {
            out = std::move(*it);
            ctx->async_readback_pool.erase(it);
            return;
        }
    }
    // Allocate a new READBACK heap.  Round up to 64K to reduce fragmentation
    // and increase reuse probability.
    size_t alloc = (size + 0xFFFF) & ~(size_t)0xFFFF;
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = alloc; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;
    HRESULT hr = ctx->dev->device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&out));
    DX12_CHECK(hr, "CreateCommittedResource(async readback staging)");
}

static void dx12_backend_set_tensor_async(ggml_backend_t backend,
                                          ggml_tensor * tensor,
                                          const void * data,
                                          size_t offset, size_t size) {
    auto * ctx = (dx12_backend_context *)backend->context;
    if (size == 0) return;

    GGML_ASSERT(tensor->buffer && "set_tensor_async on tensor without buffer");
    auto * buf_ctx = (dx12_buffer_context *)tensor->buffer->context;
    size_t tensor_offset = dx12_tensor_offset(tensor) + offset;

    // UMA fast path: direct memcpy to mapped buffer.
    //
    // Race analysis: this writes data that subsequent compute reads.  The
    // recorded cmd_list is built sequentially, and any dispatch that reads
    // this buffer is recorded AFTER set_tensor_async (the typical pattern is
    // "set inputs → record graph_compute → execute").  CPU memcpy completes
    // before this function returns, so by the time the cmd_list is later
    // submitted and executed, the GPU sees the new data.
    //
    // The remaining concern is a previously-recorded but not-yet-submitted
    // dispatch that READS this buffer.  We defensively close+execute the
    // current cmd_list before writing, then re-open.  This forces the prior
    // dispatch into the GPU queue (still race-free since GPU executes serially
    // after the upload memcpy completes — UMA writes from CPU are visible to
    // GPU as soon as the cmd_list is submitted).
    //
    // For typical usage (set inputs at top of new graph), cmd_list is closed
    // already so this is just a memcpy.
    if (buf_ctx->mapped) {
        memcpy((uint8_t *)buf_ctx->mapped + tensor_offset, data, size);
        return;
    }

    // Non-UMA: DEFAULT-heap device buffer.  Upload via shared compute cmd
    // list using the device-level xfer.upload_staging buffer.  Record a
    // CopyBufferRegion that will execute in order with subsequent dispatches.
    //
    // Important: the xfer staging is single-buffered, so concurrent async
    // uploads with different data would clobber.  In practice the scheduler
    // serializes set_tensor_async calls (each completes before next), and we
    // submit the cmd_list before the staging is reused.  For multiple inputs
    // in one batch, we close+execute between uploads to release staging.
    ctx->dev->init_xfer();
    ctx->dev->xfer_wait();   // ensure prior xfer staging usage has drained
    ctx->dev->xfer_ensure_staging(size, 0);
    memcpy(ctx->dev->xfer.upload_mapped, data, size);

    ctx->ensure_cmd_list_open();
    ctx->cmd_list->CopyBufferRegion(buf_ctx->resource.Get(), tensor_offset,
                                     ctx->dev->xfer.upload_staging.Get(), 0, size);

    // Force the upload to become visible by flushing the cmd_list.  This
    // releases the upload staging for reuse by the next set_tensor_async and
    // gives the GPU a head start.
    ctx->close_and_execute();
    // Mark the xfer staging as in-use so the next set_tensor_async waits.
    ctx->dev->xfer.fence_value = ctx->fence_value;
    ctx->dev->compute_queue->Signal(ctx->dev->xfer.fence.Get(), ctx->dev->xfer.fence_value);
    // Don't re-open cmd_list yet; subsequent record will reopen lazily.
}

static void dx12_backend_get_tensor_async(ggml_backend_t backend,
                                          const ggml_tensor * tensor,
                                          void * data,
                                          size_t offset, size_t size) {
    auto * ctx = (dx12_backend_context *)backend->context;
    if (size == 0) return;

    GGML_ASSERT(tensor->buffer && "get_tensor_async on tensor without buffer");
    auto * buf_ctx = (dx12_buffer_context *)tensor->buffer->context;
    size_t tensor_offset = dx12_tensor_offset(tensor) + offset;

    // UMA fast path: register a deferred memcpy from the mapped buffer.
    // Synchronize() will execute it after wait_for_fence, so we read coherent
    // data.  No staging copy required.
    if (buf_ctx->mapped) {
        dx12_backend_context::deferred_memcpy_t m;
        m.dst  = data;
        m.src  = (const uint8_t *)buf_ctx->mapped + tensor_offset;
        m.size = size;
        // staging stays empty — UMA path uses the direct memcpy branch
        // in synchronize().
        ctx->pending_get_memcpys.push_back(std::move(m));
        return;
    }

    // Non-UMA: queue a CopyBufferRegion to a per-call READBACK staging
    // buffer, register the deferred memcpy.
    //
    // The source is usually a compute output just written through a UAV. Keep
    // the readback copy in a fresh command list so the completed compute list
    // decays the buffer back to COMMON and the copy can promote it to
    // COPY_SOURCE. Recording the copy after UAV writes in the same command list
    // leaves the buffer in the wrong state on NVIDIA and can return stale logits.
    ComPtr<ID3D12Resource> staging;
    async_alloc_readback_staging(ctx, size, staging);

    if (ctx->cmd_list_open) {
        ctx->close_and_execute();
    }
    ctx->ensure_cmd_list_open();

    ctx->cmd_list->CopyBufferRegion(staging.Get(), 0,
                                     buf_ctx->resource.Get(), tensor_offset, size);

    dx12_backend_context::deferred_memcpy_t m;
    m.dst     = data;
    m.src     = nullptr;          // staging branch in synchronize() handles map+memcpy
    m.size    = size;
    m.staging = std::move(staging);
    ctx->pending_get_memcpys.push_back(std::move(m));
}

static bool dx12_backend_cpy_tensor_async(ggml_backend_t backend_src,
                                          ggml_backend_t backend_dst,
                                          const ggml_tensor * src,
                                          ggml_tensor * dst) {
    if (ggml_nbytes(src) == 0) return true;

    // Both tensors must be on DX12 buffers backed by the same device for an
    // intra-GPU CopyBufferRegion.  Cross-device or cross-backend → return
    // false; scheduler will fall back to a host round-trip.
    auto * dst_ctx = (dx12_backend_context *)backend_dst->context;
    if (src->buffer == nullptr || dst->buffer == nullptr) return false;

    // Verify both buffers are DX12-owned.  We identify them by checking the
    // buffer-type interface.  (We cannot rely on backend_src == backend_dst
    // because two distinct DX12 backends may share the same device.)
    auto * src_buf_ctx_raw = src->buffer->context;
    auto * dst_buf_ctx_raw = dst->buffer->context;
    if (!src_buf_ctx_raw || !dst_buf_ctx_raw) return false;

    auto * src_buf_ctx = (dx12_buffer_context *)src_buf_ctx_raw;
    auto * dst_buf_ctx = (dx12_buffer_context *)dst_buf_ctx_raw;

    // Cross-device DX12-to-DX12 not supported via direct copy.
    if (src_buf_ctx->dev != dst_buf_ctx->dev) return false;
    if (dst_buf_ctx->dev != dst_ctx->dev) return false;

    // Ensure the source tensor is a DX12 buffer (the cross-backend case where
    // src is a host buffer must fall through to the scheduler's fallback).
    // We additionally guard against null GPU resources (pre-allocation).
    if (!src_buf_ctx->resource || !dst_buf_ctx->resource) return false;

    size_t src_off = dx12_tensor_offset(src);
    size_t dst_off = dx12_tensor_offset(dst);
    size_t bytes   = ggml_nbytes(src);

    dst_ctx->ensure_cmd_list_open();
    dst_ctx->cmd_list->CopyBufferRegion(dst_buf_ctx->resource.Get(), dst_off,
                                         src_buf_ctx->resource.Get(), src_off, bytes);

    // No fence/event required: the copy is recorded onto the same cmd_list
    // as subsequent dispatches and serializes naturally.  The dst tensor's
    // contents become visible to those dispatches via the cmd_list ordering.
    GGML_UNUSED(backend_src);
    return true;
}


static const ggml_backend_i dx12_backend_interface = {
    /* .get_name            = */ dx12_backend_get_name,
    /* .free                = */ dx12_backend_free,
    /* .set_tensor_async    = */ dx12_backend_set_tensor_async,
    /* .get_tensor_async    = */ dx12_backend_get_tensor_async,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ dx12_backend_cpy_tensor_async,
    /* .synchronize         = */ dx12_backend_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ dx12_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ dx12_graph_optimize,
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
    ctx->fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

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
    bool ok = dx12_supports_op(dev, op);
    if (!ok) {
        static const bool log_unsup = []{
            const char * e = getenv("DX12_LOG_UNSUPPORTED_OPS");
            return e && *e && *e != '0';
        }();
        if (log_unsup) {
            const char * name = ggml_op_name(op->op);
            const char * tname = ggml_type_name(op->type);
            const char * s0n = (op->src[0]) ? ggml_type_name(op->src[0]->type) : "-";
            const char * s1n = (op->src[1]) ? ggml_type_name(op->src[1]->type) : "-";
            fprintf(stderr, "ggml-dx12: unsupported op=%s dst=%s src0=%s src1=%s ne=[%lld,%lld,%lld,%lld]\n",
                name ? name : "?", tname ? tname : "?", s0n, s1n,
                (long long)op->ne[0], (long long)op->ne[1], (long long)op->ne[2], (long long)op->ne[3]);
        }
    }
    return ok;
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

#ifdef GGML_DX12_SHADERS_COMPILED
// Macro to select wave-size-specific blob at init time
#define WB_INNER(name, ws) { g_##name##_w##ws##_dxil, sizeof(g_##name##_w##ws##_dxil) }
#define WB(name, ws) WB_INNER(name, ws)

// FP16-aware variant: picks the `_fp16_dxil` blob when the device supports
// native 16-bit shader ops AND DX12_ENABLE_FP16 is set, otherwise falls back
// to the FP32 blob. Default is OFF because the dual-compiled shaders still
// accumulate in fp32 (precise float + (float) casts), so on bandwidth-bound
// matvec paths the load-instruction tweak is a measured no-op on RTX 6000 Ada
// (within ±2 t/s of the FP32 path) and only "trending up within noise" on
// Intel Arc B390. Kept as opt-in for diagnostic A/B and future tuning.
// Evaluated at init time so the per-dispatch path is unchanged.
#define WB_FP16_INNER(name, ws) (this->fp16_supported && getenv("DX12_ENABLE_FP16") \
    ? dx12_shader_blob{ g_##name##_w##ws##_fp16_dxil, sizeof(g_##name##_w##ws##_fp16_dxil) } \
    : dx12_shader_blob{ g_##name##_w##ws##_dxil,      sizeof(g_##name##_w##ws##_dxil) })
#define WB_FP16(name, ws) WB_FP16_INNER(name, ws)

void dx12_device::init_shader_blobs() {
    // Round wave_size to nearest compiled variant: 16, 32, or 64
    uint32_t ws = wave_size <= 16 ? 16 : (wave_size <= 32 ? 32 : 64);

    // Helper lambdas that return the right blob for each compiled wave size
    // This runs once at init — zero per-dispatch overhead
    #define POPULATE_BLOBS(WS) do { \
        shader_blobs = { \
            { GGML_OP_ADD,           WB(add, WS)           }, \
            { GGML_OP_SUB,           WB(sub, WS)           }, \
            { GGML_OP_MUL,           WB(mul, WS)           }, \
            { GGML_OP_DIV,           WB(div, WS)           }, \
            { GGML_OP_SCALE,         WB(scale, WS)         }, \
            { GGML_OP_SQR,           WB(sqr, WS)           }, \
            { GGML_OP_SQRT,          WB(sqrt_, WS)         }, \
            { GGML_OP_SIN,           WB(sin, WS)           }, \
            { GGML_OP_COS,           WB(cos, WS)           }, \
            { GGML_OP_LOG,           WB(log, WS)           }, \
            { GGML_OP_CLAMP,         WB(clamp, WS)         }, \
            { GGML_OP_CONT,          WB(cpy, WS)           }, \
            { GGML_OP_CPY,           WB(cpy, WS)           }, \
            { GGML_OP_DUP,           WB(cpy, WS)           }, \
            { GGML_OP_RMS_NORM,      WB(rms_norm, WS)      }, \
            { GGML_OP_NORM,          WB(norm, WS)          }, \
            { GGML_OP_GROUP_NORM,    WB(group_norm, WS)    }, \
            { GGML_OP_SOFT_MAX,      WB(soft_max, WS)      }, \
            { GGML_OP_MUL_MAT,       WB(mul_mat, WS)       }, \
            { GGML_OP_MUL_MAT_ID,    WB(mul_mat_id, WS)    }, \
            { GGML_OP_GET_ROWS,      WB(get_rows, WS)      }, \
            { GGML_OP_DIAG_MASK_INF, WB(diag_mask_inf, WS) }, \
            { GGML_OP_ROPE,          WB(rope, WS)          }, \
            { GGML_OP_CONCAT,        WB(concat, WS)        }, \
            { GGML_OP_REPEAT,        WB(repeat, WS)        }, \
            { GGML_OP_SUM_ROWS,      WB(sum_rows, WS)      }, \
            { GGML_OP_PAD,           WB(pad, WS)           }, \
            { GGML_OP_ROLL,          WB(roll, WS)          }, \
            { GGML_OP_SSM_CONV,      WB(ssm_conv, WS)      }, \
            { GGML_OP_UPSCALE,       WB(upscale, WS)       }, \
            { GGML_OP_IM2COL,        WB(im2col, WS)        }, \
            { GGML_OP_POOL_2D,       WB(pool_2d, WS)       }, \
            { GGML_OP_POOL_1D,       WB(pool_1d, WS)       }, \
            { GGML_OP_CONV_2D,       WB(conv_2d, WS)       }, \
            { GGML_OP_FLASH_ATTN_EXT,WB_FP16(flash_attn, WS)    }, \
            { GGML_OP_SET_ROWS,      WB(set_rows, WS)      }, \
            { GGML_OP_GLU,           WB(glu, WS)           }, \
            { GGML_OP_L2_NORM,       WB(l2_norm, WS)       }, \
            { GGML_OP_GATED_DELTA_NET, WB(gated_delta_net, WS) }, \
            { GGML_OP_SSM_SCAN,      WB(ssm_scan, WS)      }, \
        }; \
        unary_shader_blobs = { \
            { GGML_UNARY_OP_SILU,       WB(silu, WS)       }, \
            { GGML_UNARY_OP_GELU,       WB(gelu, WS)       }, \
            { GGML_UNARY_OP_GELU_QUICK, WB(gelu_quick, WS) }, \
            { GGML_UNARY_OP_GELU_ERF,   WB(gelu_erf, WS)   }, \
            { GGML_UNARY_OP_RELU,       WB(relu, WS)       }, \
            { GGML_UNARY_OP_TANH,       WB(tanh_, WS)      }, \
            { GGML_UNARY_OP_SIGMOID,    WB(sigmoid, WS)    }, \
            { GGML_UNARY_OP_EXP,        WB(exp, WS)        }, \
            { GGML_UNARY_OP_SOFTPLUS,   WB(softplus, WS)   }, \
        }; \
    } while(0)

    if (ws == 16) {
        POPULATE_BLOBS(16);
    } else if (ws == 32) {
        POPULATE_BLOBS(32);
    } else {
        POPULATE_BLOBS(64);
    }
    #undef POPULATE_BLOBS

    DX12_LOG_INFO("Shader blobs: using wave=%u variant (device wave=%u)\n", ws, wave_size);
}
#else
void dx12_device::init_shader_blobs() {}
#endif

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

dx12_pipeline * dx12_device::get_or_create_pipeline(const dx12_pipeline_key & key) {
    // Fast path: skip mutex + map lookup for repeated pipeline keys
    if (key == last_pipeline_key && last_pipeline_ptr) {
        return last_pipeline_ptr;
    }

    std::lock_guard<std::mutex> lock(pipeline_mutex);

    auto it = pipeline_cache.find(key);
    if (it != pipeline_cache.end()) {
        last_pipeline_key = key;
        last_pipeline_ptr = &it->second;
        return last_pipeline_ptr;
    }

#ifdef GGML_DX12_SHADERS_COMPILED
    const dx12_shader_blob * blob = nullptr;

    // Wave-size blob selection helper — returns the right compiled variant
    auto wblob = [this](const void* d16, size_t s16, const void* d32, size_t s32, const void* d64, size_t s64) -> dx12_shader_blob {
        if (wave_size <= 16) return { d16, s16 };
        if (wave_size <= 32) return { d32, s32 };
        return { d64, s64 };
    };
    #define WBLOB(name) wblob( \
        g_##name##_w16_dxil, sizeof(g_##name##_w16_dxil), \
        g_##name##_w32_dxil, sizeof(g_##name##_w32_dxil), \
        g_##name##_w64_dxil, sizeof(g_##name##_w64_dxil))

    // FP16 variant selector: pick the `_fp16_dxil` blob when the device
    // supports native 16-bit shader ops (D3D12_OPTIONS4) AND the user opts in
    // via DX12_ENABLE_FP16=1. Default is OFF because the dual-compiled
    // shaders still accumulate in fp32, so the load-instruction tweak is a
    // no-op on bandwidth-bound matvec on the GPUs measured so far.
    static const bool enable_fp16 = (getenv("DX12_ENABLE_FP16") != nullptr);
    auto wblob_fp16_pick = [this](
        const void* d16, size_t s16, const void* d32, size_t s32, const void* d64, size_t s64,
        const void* d16_fp16, size_t s16_fp16, const void* d32_fp16, size_t s32_fp16, const void* d64_fp16, size_t s64_fp16) -> dx12_shader_blob {
        const bool use_fp16 = fp16_supported && enable_fp16;
        if (wave_size <= 16) return use_fp16 ? dx12_shader_blob{ d16_fp16, s16_fp16 } : dx12_shader_blob{ d16, s16 };
        if (wave_size <= 32) return use_fp16 ? dx12_shader_blob{ d32_fp16, s32_fp16 } : dx12_shader_blob{ d32, s32 };
        return use_fp16 ? dx12_shader_blob{ d64_fp16, s64_fp16 } : dx12_shader_blob{ d64, s64 };
    };
    #define WBLOB_FP16(name) wblob_fp16_pick( \
        g_##name##_w16_dxil,      sizeof(g_##name##_w16_dxil), \
        g_##name##_w32_dxil,      sizeof(g_##name##_w32_dxil), \
        g_##name##_w64_dxil,      sizeof(g_##name##_w64_dxil), \
        g_##name##_w16_fp16_dxil, sizeof(g_##name##_w16_fp16_dxil), \
        g_##name##_w32_fp16_dxil, sizeof(g_##name##_w32_fp16_dxil), \
        g_##name##_w64_fp16_dxil, sizeof(g_##name##_w64_fp16_dxil))

    // For UNARY ops, look up by the unary sub-op stored in flags
    if (key.op == GGML_OP_UNARY) {
        auto uit = unary_shader_blobs.find((int)key.flags);
        if (uit != unary_shader_blobs.end()) {
            blob = &uit->second;
        }
    } else if (key.op == GGML_OP_RMS_NORM && key.flags == 2) {
        // Fused RMS_NORM + MUL
        const dx12_shader_blob fused_blob = WBLOB(rms_norm_mul); blob = &fused_blob;
    } else if (key.op == GGML_OP_RMS_NORM && key.flags == 3) {
        // Fused ADD + RMS_NORM + MUL (triple fusion)
        const dx12_shader_blob fused_blob = WBLOB(add_rms_norm_mul); blob = &fused_blob;
    } else if (key.op == GGML_OP_RMS_NORM && key.flags == 7) {
        // Fused RMS_NORM + MUL + ROPE
        const dx12_shader_blob fused_blob = WBLOB(rms_norm_mul_rope); blob = &fused_blob;
    } else if (key.op == GGML_OP_RMS_NORM && key.flags == 8) {
        // Fused RMS_NORM + MUL + ROPE + VIEW + SET_ROWS (5-way)
        const dx12_shader_blob fused_blob = WBLOB(rms_norm_mul_rope_set_rows); blob = &fused_blob;
    } else if (key.op == GGML_OP_ROPE && key.flags == 6) {
        // Fused ROPE + VIEW + SET_ROWS
        const dx12_shader_blob fused_blob = WBLOB(rope_set_rows); blob = &fused_blob;
    } else if (key.op == GGML_OP_ROPE && key.flags == 13) {
        // mrope (multi-dimensional ROPE for Qwen3-VL etc.)
        const dx12_shader_blob mrope_blob = WBLOB(rope_multi); blob = &mrope_blob;
    } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags == 1) {
        // GQA-folded flash attention (one workgroup per kv_head, loops over Q-heads)
        const dx12_shader_blob fa_gqa_blob = WBLOB_FP16(flash_attn_gqa); blob = &fa_gqa_blob;
    } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags == 2) {
        // Small-D (<=64) decode-friendly flash attention: GROUP_SIZE = TILE_KV = 64
        const dx12_shader_blob fa_64_blob = WBLOB(flash_attn_64); blob = &fa_64_blob;
    } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags == 3) {
        // Mid-D (65..128) flash attention: GROUP_SIZE = TILE_KV = 128
        const dx12_shader_blob fa_128_blob = WBLOB(flash_attn_128); blob = &fa_128_blob;
    } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags == 8) {
        // Split-KV reduction shader
        const dx12_shader_blob reduce_blob = WBLOB(flash_attn_reduce); blob = &reduce_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 4) {
        // Register-blocked tiled batch MUL_MAT (M > 1)
        if (key.src0_type == GGML_TYPE_Q4_K) {
            const dx12_shader_blob wmma_q4k_blob = WBLOB(mul_mat_q4k_wmma); blob = &wmma_q4k_blob;
        } else if (key.src0_type == GGML_TYPE_Q5_K) {
            const dx12_shader_blob wmma_q5k_blob = WBLOB(mul_mat_q5k_wmma); blob = &wmma_q5k_blob;
        } else if (key.src0_type == GGML_TYPE_Q6_K) {
            const dx12_shader_blob wmma_q6k_blob = WBLOB(mul_mat_q6k_wmma); blob = &wmma_q6k_blob;
        } else if (key.src0_type == GGML_TYPE_Q8_0) {
            const dx12_shader_blob wmma_q80_blob = WBLOB(mul_mat_q8_0_wmma); blob = &wmma_q80_blob;
        } else {
            // F16/F32 register-blocked tiled batch MUL_MAT
            const dx12_shader_blob wmma_blob = WBLOB(mul_mat_wmma); blob = &wmma_blob;
        }
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 8) {
        // dp4a batch MUL_MAT: Q8_0 weights × Q8_1 quantized input
        const dx12_shader_blob dp4a_blob = WBLOB(mul_mat_q8_0_q8_1); blob = &dp4a_blob;
    } else if (key.op == GGML_OP_NONE && key.flags == 99) {
        // Quantize F32 → Q8_1
        const dx12_shader_blob q_blob = WBLOB(quantize_q8_1); blob = &q_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 9) {
        // Multi-row matvec (2 rows per group, float dequant)
        if (key.src0_type == GGML_TYPE_Q4_K) {
            const dx12_shader_blob mv_q4k_mr_blob = WBLOB(mul_mat_vec_q4k_mr); blob = &mv_q4k_mr_blob;
        } else if (key.src0_type == GGML_TYPE_Q5_K) {
            const dx12_shader_blob mv_q5k_mr_blob = WBLOB(mul_mat_vec_q5k_mr); blob = &mv_q5k_mr_blob;
        } else if (key.src0_type == GGML_TYPE_Q6_K) {
            const dx12_shader_blob mv_q6k_mr_blob = WBLOB(mul_mat_vec_q6k_mr); blob = &mv_q6k_mr_blob;
        } else if (key.src0_type == GGML_TYPE_Q8_0) {
            const dx12_shader_blob mv_q80_mr_blob = WBLOB(mul_mat_vec_q8_0_mr); blob = &mv_q80_mr_blob;
        } else if (key.src0_type == GGML_TYPE_Q5_0) {
            const dx12_shader_blob mv_q50_mr_blob = WBLOB(mul_mat_vec_q5_0_mr); blob = &mv_q50_mr_blob;
        } else if (key.src0_type == GGML_TYPE_Q5_1) {
            const dx12_shader_blob mv_q51_mr_blob = WBLOB(mul_mat_vec_q5_1_mr); blob = &mv_q51_mr_blob;
        }
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 10) {
        // Q4_K dp4a multi-row matvec (dot4add_i8packed + Q8_1 activations)
        const dx12_shader_blob mv_q4k_dp4a_blob = WBLOB(mul_mat_vec_q4k_dp4a); blob = &mv_q4k_dp4a_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 25) {
        // Q6_K block-level matvec (diagnostic, opt-in via DX12_Q6K_BLOCKED=1)
        const dx12_shader_blob mv_q6k_blk_blob = WBLOB(mul_mat_vec_q6k_mr_blocked); blob = &mv_q6k_blk_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 26) {
        // Q3_K block-level matvec (diagnostic, opt-in via DX12_Q3K_BLOCKED=1)
        const dx12_shader_blob mv_q3k_blk_blob = WBLOB(mul_mat_vec_q3k_mr_blocked); blob = &mv_q3k_blk_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 27) {
        // Q2_K block-level matvec (default; opt out via DX12_Q2K_BLOCKED=0)
        const dx12_shader_blob mv_q2k_blk_blob = WBLOB(mul_mat_vec_q2k_mr_blocked); blob = &mv_q2k_blk_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 28) {
        // Q8_0 mr64 (single-wave AMD wave64, 4 rows/group; default-on, opt out via DX12_Q8_MR64=0)
        const dx12_shader_blob mv_q8_mr64_blob = WBLOB(mul_mat_vec_q8_0_mr64); blob = &mv_q8_mr64_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 29) {
        // Q5_0 mr64 (single-wave AMD wave64, 4 rows/group; opt-in DX12_Q50_MR64=1)
        const dx12_shader_blob mv_q50_mr64_blob = WBLOB(mul_mat_vec_q5_0_mr64); blob = &mv_q50_mr64_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 30) {
        // Q4_K cooperative-LDS WMMA (default-on, opt out via DX12_Q4K_WMMA_LDS=0)
        const dx12_shader_blob wmma_q4k_lds_blob = WBLOB(mul_mat_q4k_wmma_lds); blob = &wmma_q4k_lds_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 13) {
        // Q4_K dp4a multi-row matvec — 32-thread variant (better on small-wave GPUs)
        const dx12_shader_blob mv_q4k_dp4a_32_blob = WBLOB(mul_mat_vec_q4k_dp4a_32); blob = &mv_q4k_dp4a_32_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 14) {
        // Q5_K dp4a multi-row matvec (dot4add_i8packed + Q8_1 activations)
        const dx12_shader_blob mv_q5k_dp4a_blob = WBLOB(mul_mat_vec_q5k_dp4a); blob = &mv_q5k_dp4a_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 15) {
        // Q5_K subgroup matvec — 32-thread single-wave (NVIDIA), 2 rows/WG
        const dx12_shader_blob mv_q5k_sg_blob = WBLOB(mul_mat_vec_q5k_subgroup); blob = &mv_q5k_sg_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 16) {
        // Q5_K dp4a multi-row matvec — 32-thread variant (better on small-wave GPUs)
        const dx12_shader_blob mv_q5k_dp4a_32_blob = WBLOB(mul_mat_vec_q5k_dp4a_32); blob = &mv_q5k_dp4a_32_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 17) {
        // Q8_0 dp4a multi-row matvec (dot4add_i8packed + Q8_1 activations)
        const dx12_shader_blob mv_q80_dp4a_blob = WBLOB(mul_mat_vec_q8_0_dp4a); blob = &mv_q80_dp4a_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 18) {
        // Q8_0 vectorized multi-row matvec — 256 threads, 4 elements/thread
        const dx12_shader_blob mv_q80_mr256v_blob = WBLOB(mul_mat_vec_q8_0_mr256v); blob = &mv_q80_mr256v_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 19) {
        // Q2_K multi-row matvec — 256 threads, 2 rows/group
        const dx12_shader_blob mv_q2k_mr_blob = WBLOB(mul_mat_vec_q2k_mr); blob = &mv_q2k_mr_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 20) {
        // Q3_K multi-row matvec — 256 threads, 2 rows/group (K>=4096 only)
        const dx12_shader_blob mv_q3k_mr_blob = WBLOB(mul_mat_vec_q3k_mr); blob = &mv_q3k_mr_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 21) {
        // Q5_0 dp4a multi-row matvec (dot4add_i8packed + Q8_1 activations)
        const dx12_shader_blob mv_q50_dp4a_blob = WBLOB(mul_mat_vec_q5_0_dp4a); blob = &mv_q50_dp4a_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 22) {
        // Q5_1 dp4a multi-row matvec (dot4add_i8packed + Q8_1 activations)
        const dx12_shader_blob mv_q51_dp4a_blob = WBLOB(mul_mat_vec_q5_1_dp4a); blob = &mv_q51_dp4a_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 23) {
        // Q6_K dp4a multi-row matvec (dot4add_i8packed + Q8_1 activations)
        const dx12_shader_blob mv_q6k_dp4a_blob = WBLOB(mul_mat_vec_q6k_dp4a); blob = &mv_q6k_dp4a_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 24) {
        // R9: fused MUL_MAT(W_up) + MUL_MAT(W_gate) + GLU(SWIGLU split)
        // Two F16 matvecs sharing the same activation, collapsed into one
        // K-loop, output = silu(gate) * up.
        const dx12_shader_blob mv_glu_blob = WBLOB_FP16(mul_mat_vec_glu); blob = &mv_glu_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 11) {
        // F16/F32 multi-row matvec — 256 threads (2 rows per group)
        const dx12_shader_blob mv_mr_blob = WBLOB_FP16(mul_mat_vec_mr); blob = &mv_mr_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 12) {
        // F16/F32 multi-row matvec — 32 threads (compact, better for small K)
        const dx12_shader_blob mv_mr32_blob = WBLOB_FP16(mul_mat_vec_mr32); blob = &mv_mr32_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.flags == 1) {
        // Matvec path (M=1) — only Q2_K/Q3_K/BF16 actually reach here today.
        // Q4_K/Q5_K/Q6_K/Q5_0/Q5_1/Q8_0 are routed by their dedicated flags
        // (9-18) earlier in the dispatch path; F16/F32 use flags=11 or 12.
        if (key.src0_type == GGML_TYPE_Q2_K) {
            const dx12_shader_blob mv_q2k_blob = WBLOB(mul_mat_vec_q2k); blob = &mv_q2k_blob;
        } else if (key.src0_type == GGML_TYPE_Q3_K) {
            const dx12_shader_blob mv_q3k_blob = WBLOB(mul_mat_vec_q3k); blob = &mv_q3k_blob;
        } else if (key.src0_type == GGML_TYPE_IQ4_NL) {
            const dx12_shader_blob mv_iq4nl_blob = WBLOB(mul_mat_vec_iq4_nl); blob = &mv_iq4nl_blob;
        } else {
            // BF16 / generic fallback
            const dx12_shader_blob mv_blob = WBLOB_FP16(mul_mat_vec); blob = &mv_blob;
        }
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q8_0) {
        const dx12_shader_blob q80_blob = WBLOB(mul_mat_q8_0); blob = &q80_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q5_0) {
        const dx12_shader_blob q50_blob = WBLOB(mul_mat_q5_0); blob = &q50_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q4_0) {
        const dx12_shader_blob q40_blob = WBLOB(mul_mat_q4_0); blob = &q40_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q4_1) {
        const dx12_shader_blob q41_blob = WBLOB(mul_mat_q4_1); blob = &q41_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q5_1) {
        const dx12_shader_blob q51_blob = WBLOB(mul_mat_q5_1); blob = &q51_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q8_1) {
        const dx12_shader_blob q81_blob = WBLOB(mul_mat_q8_1); blob = &q81_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q2_K) {
        const dx12_shader_blob q2k_blob = WBLOB(mul_mat_q2k); blob = &q2k_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_Q3_K) {
        const dx12_shader_blob q3k_blob = WBLOB(mul_mat_q3k); blob = &q3k_blob;
    } else if (key.op == GGML_OP_MUL_MAT && key.src0_type == GGML_TYPE_IQ4_NL) {
        const dx12_shader_blob iq4nl_blob = WBLOB(mul_mat_iq4_nl); blob = &iq4nl_blob;
    } else if (key.op == GGML_OP_MUL_MAT_ID && key.src0_type == GGML_TYPE_Q4_0) {
        const dx12_shader_blob mmi_q40_blob = WBLOB(mul_mat_id_q4_0); blob = &mmi_q40_blob;
    } else if (key.op == GGML_OP_MUL_MAT_ID && key.src0_type == GGML_TYPE_Q4_1) {
        const dx12_shader_blob mmi_q41_blob = WBLOB(mul_mat_id_q4_1); blob = &mmi_q41_blob;
    } else if (key.op == GGML_OP_MUL_MAT_ID && key.src0_type == GGML_TYPE_Q5_0) {
        const dx12_shader_blob mmi_q50_blob = WBLOB(mul_mat_id_q5_0); blob = &mmi_q50_blob;
    } else if (key.op == GGML_OP_MUL_MAT_ID && key.src0_type == GGML_TYPE_Q5_1) {
        const dx12_shader_blob mmi_q51_blob = WBLOB(mul_mat_id_q5_1); blob = &mmi_q51_blob;
    } else if (key.op == GGML_OP_MUL_MAT_ID && key.src0_type == GGML_TYPE_Q8_0) {
        const dx12_shader_blob mmi_q80_blob = WBLOB(mul_mat_id_q8_0); blob = &mmi_q80_blob;
    } else if (key.op == GGML_OP_MUL_MAT_ID && key.src0_type == GGML_TYPE_Q4_K) {
        const dx12_shader_blob mmi_q4k_blob = WBLOB(mul_mat_id_q4k); blob = &mmi_q4k_blob;
    } else if (key.op == GGML_OP_MUL_MAT_ID && key.src0_type == GGML_TYPE_Q5_K) {
        const dx12_shader_blob mmi_q5k_blob = WBLOB(mul_mat_id_q5k); blob = &mmi_q5k_blob;
    } else if (key.op == GGML_OP_MUL_MAT_ID && key.src0_type == GGML_TYPE_Q6_K) {
        const dx12_shader_blob mmi_q6k_blob = WBLOB(mul_mat_id_q6k); blob = &mmi_q6k_blob;
    } else if (key.op == GGML_OP_MUL_MAT_ID && key.src0_type == GGML_TYPE_IQ4_NL) {
        const dx12_shader_blob mmi_iq4nl_blob = WBLOB(mul_mat_id_iq4_nl); blob = &mmi_iq4nl_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q4_K) {
        // Q4_K dequantizing get_rows
        const dx12_shader_blob q4k_gr_blob = WBLOB(get_rows_q4k); blob = &q4k_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q5_K) {
        const dx12_shader_blob q5k_gr_blob = WBLOB(get_rows_q5k); blob = &q5k_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q6_K) {
        const dx12_shader_blob q6k_gr_blob = WBLOB(get_rows_q6k); blob = &q6k_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q8_0) {
        const dx12_shader_blob q80_gr_blob = WBLOB(get_rows_q8_0); blob = &q80_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q5_0) {
        const dx12_shader_blob q50_gr_blob = WBLOB(get_rows_q5_0); blob = &q50_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q4_0) {
        const dx12_shader_blob q40_gr_blob = WBLOB(get_rows_q4_0); blob = &q40_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q4_1) {
        const dx12_shader_blob q41_gr_blob = WBLOB(get_rows_q4_1); blob = &q41_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q5_1) {
        const dx12_shader_blob q51_gr_blob = WBLOB(get_rows_q5_1); blob = &q51_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q8_1) {
        const dx12_shader_blob q81_gr_blob = WBLOB(get_rows_q8_1); blob = &q81_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q2_K) {
        const dx12_shader_blob q2k_gr_blob = WBLOB(get_rows_q2k); blob = &q2k_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_Q3_K) {
        const dx12_shader_blob q3k_gr_blob = WBLOB(get_rows_q3k); blob = &q3k_gr_blob;
    } else if (key.op == GGML_OP_GET_ROWS && key.src0_type == GGML_TYPE_IQ4_NL) {
        const dx12_shader_blob iq4nl_gr_blob = WBLOB(get_rows_iq4_nl); blob = &iq4nl_gr_blob;
    } else {
        auto sit = shader_blobs.find((int)key.op);
        if (sit != shader_blobs.end()) {
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
        DX12_LOG_ERROR("Failed to create PSO for op %d (%s, flags=%u) (HRESULT 0x%08X)\n",
                       key.op, ggml_op_name((enum ggml_op)key.op), (unsigned)key.flags, (unsigned)hr);
        pipeline_cache[key] = {};
        return &pipeline_cache[key];
    }

    pipeline.root_sig = common_root_sig;
    pipeline_cache[key] = std::move(pipeline);
    last_pipeline_key = key;
    last_pipeline_ptr = &pipeline_cache[key];
    return last_pipeline_ptr;
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
