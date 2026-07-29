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

static uint64_t dx12_qpc_us() {
    static const double ticks_to_us = []() {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        return 1000000.0 / (double)frequency.QuadPart;
    }();
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)((double)counter.QuadPart * ticks_to_us);
}

static uint64_t g_dx12_buf_set_us    = 0;
static uint64_t g_dx12_buf_set_calls = 0;

// ---------------------------------------------------------------------------
// Env-var refresh hook (for in-process variant sweep by llama-mmv-tune)
// ---------------------------------------------------------------------------
// Most DX12_* dispatch-gate env vars are cached at first call via
// `static const ... = getenv(...)`. This makes mid-process env flips a no-op,
// which forces the tuner to re-exec for every variant. When this TLS flag is
// set (via ggml_backend_dx12_set_env_refresh below), DX12_GETENV evaluates
// getenv() on every call instead. Cost: one TLS read + branch per dispatch
// per env (~ns). Tuner pays only during the bench loop.
namespace {
thread_local bool g_dx12_env_refresh = false;
}
// Read a DX12_* env var, returning either the cached value (default) or a
// fresh getenv() result (when refresh is enabled). Each invocation has its own
// static cache via lambda-local static, so different sites cache independently.
#define DX12_GETENV(name) ([]() -> const char * { \
    if (g_dx12_env_refresh) return getenv(name); \
    static const char * v = getenv(name); \
    return v; \
}())

// Read a DX12_* tuning flag that ships ENABLED by default. Unset => on; an
// explicit leading '0' (e.g. DX12_FOO=0) turns it off. Mirrors the opt-in
// "v && v[0] && v[0] != '0'" idiom but flips the default so the shipped decode
// fast paths are active without any environment setup.
static inline bool dx12_flag_default_on(const char * name) {
    const char * v = getenv(name);
    return !v || !v[0] || v[0] != '0';
}

// Heuristic-pick capture sink: when set, the dispatch path appends each
// dispatched MUL_MAT(_ID) node's selected key.flags here. Used by the tuner
// to confirm which variant the heuristic picked for the shape it benched.
namespace {
thread_local std::vector<uint32_t> * g_dx12_flag_sink = nullptr;
}

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

// Device-init banner: write directly to stderr so it survives the upstream
// llama-cli verbosity default (LOG_LEVEL_ERROR), which filters GGML_LOG_INFO.
// Used only for the one-time device-enumeration banner so users always see
// which adapters were picked up and their feature/wave classification.
#define DX12_LOG_BANNER(...) do {              \
        fputs("ggml-dx12: ", stderr);          \
        fprintf(stderr, __VA_ARGS__);          \
        fflush(stderr);                        \
    } while (0)

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
    DX12_FUSE_NONE                = 0,
    DX12_FUSE_ADD_RMS_MUL         = 1,  // ADD + RMS_NORM + MUL  (skip 2)
    DX12_FUSE_RMS_MUL             = 2,  // RMS_NORM + MUL        (skip 1)
    DX12_FUSE_RMS_MUL_ROPE3       = 3,  // RMS_NORM + MUL + ROPE (skip 2)
    DX12_FUSE_RMS_MUL_ROPE5       = 4,  // + VIEW + SET_ROWS     (skip 4)
    DX12_FUSE_ROPE_SET_ROWS       = 5,  // ROPE + VIEW + SET_ROWS (skip 2)
    DX12_FUSE_MMV_GLU_SPLIT       = 6,  // MUL_MAT(up) + MUL_MAT(gate) + SWIGLU split (skip 2)
    DX12_FUSE_SSM_CONV_SILU       = 7,  // SSM_CONV + UNARY(SILU)         (skip 1)
    DX12_FUSE_SSM_CONV_BIAS_SILU  = 8,  // SSM_CONV + ADD + UNARY(SILU)   (skip 2)
    DX12_FUSE_RMS_MUL_QUANT_Q8_1  = 9,  // RMS_NORM + MUL + Q8_1 pre-pass for downstream dp4a matmul (skip 1; consumer matmul not absorbed)
    DX12_FUSE_MMV_SET_ROWS        = 10, // MUL_MAT(V proj, M=1) -> ... -> VIEW -> SET_ROWS: matvec scatters into KV cache (absorbs only the SET_ROWS, non-contiguous)
    DX12_FUSE_MMV_Q_ROPE          = 11, // MUL_MAT(Q proj, M=1) -> RESHAPE -> ROPE: matvec applies RoPE, writes rope output (absorbs the ROPE, non-contiguous)
    DX12_FUSE_MMV_K_ROPE_SET_ROWS = 12, // MUL_MAT(K proj, M=1) -> RESHAPE -> ROPE -> VIEW -> SET_ROWS: matvec applies RoPE + scatters into KV cache (absorbs ROPE+VIEW+SET_ROWS)
    DX12_FUSE_MMV_QKV_SHARED      = 13, // MUL_MAT(Q proj, M=1) recorded once, absorbing the V and K projection matvecs (which share the activation) plus all three post-ops (Q RoPE, K RoPE+scatter, V scatter) into a single combined dispatch
    DX12_FUSE_QK_ROPE_SCALE_SET_ROWS = 14, // Q ROPE+SCALE plus sibling K ROPE+VIEW+SET_ROWS
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

    // DX12_FUSE_MMV_SET_ROWS: node offset (relative to this MUL_MAT) of the
    // absorbed SET_ROWS. The SET_ROWS is not adjacent (K-projection nodes sit
    // between), so it cannot ride the contiguous skip_count path — the replay
    // fast path reconstructs it via cgraph->nodes[i + mmv_set_rows_rel].
    int16_t            mmv_set_rows_rel;      // 0 = no MMV_SET_ROWS fusion on this node

    // DX12_FUSE_MMV_Q_ROPE / DX12_FUSE_MMV_K_ROPE_SET_ROWS: node offsets
    // (relative to this MUL_MAT) of the absorbed ROPE and, for K, the SET_ROWS.
    // Non-adjacent (other projections sit between), so the replay fast path
    // reconstructs them via cgraph->nodes[i + *_rel]. 0 = not fused. The K VIEW
    // needs no tracking — it self-skips as an ordinary view node.
    int16_t            mmv_rope_rel;          // absorbed ROPE (Q and K)
    int16_t            mmv_rope_set_rows_rel; // absorbed SET_ROWS (K only)

    // DX12_FUSE_MMV_QKV_SHARED: node offsets (relative to this Q MUL_MAT) of the
    // two absorbed projection matvecs and all three post-ops the combined
    // dispatch replaces. Non-adjacent; the replay fast path reconstructs them via
    // cgraph->nodes[i + *_rel]. 0 = not fused.
    int16_t            qkv_v_matvec_rel;      // V projection MUL_MAT
    int16_t            qkv_k_matvec_rel;      // K projection MUL_MAT
    int16_t            qkv_q_rope_rel;        // Q ROPE (rope output)
    int16_t            qkv_k_rope_rel;        // K ROPE
    int16_t            qkv_k_set_rows_rel;    // K SET_ROWS (K cache)
    int16_t            qkv_v_set_rows_rel;    // V SET_ROWS (V cache)

    // DX12_FUSE_QK_ROPE_SCALE_SET_ROWS: offsets relative to the Q ROPE node.
    // Q SCALE is adjacent and uses skip_count; the K chain is non-contiguous.
    int16_t            qk_scale_rel;
    int16_t            qk_k_rope_rel;
    int16_t            qk_k_set_rows_rel;

    bool               is_matvec_dispatch;
    bool               use_dp4a;
    bool               use_dp4a_matvec;
    bool               needs_op_params;
    bool               conservative_barrier;  // SET_ROWS / FA / fused_rope_set_rows
    bool               has_bias_add;          // matvec fused with following ADD bias
    bool               fusion_skip_f32;       // RMS+Q8_1 fusion: skip F32 dst write (all consumers go through Q8_1 cache)

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

// Vendor IDs (PCI). Re-exposed here so the arch-family classifier and any
// scattered VendorId checks share one set of constants.
namespace dx12_vendor {
constexpr UINT NVIDIA   = 0x10DE;
constexpr UINT AMD      = 0x1002;
constexpr UINT INTEL    = 0x8086;
constexpr UINT QUALCOMM = 0x5143;
constexpr UINT APPLE    = 0x106B;
constexpr UINT MICROSOFT = 0x1414; // WARP, basic render
}  // namespace dx12_vendor

// Architecture family — a coarse classification of the runtime GPU into one
// of a handful of buckets that share dispatcher-relevant behavior. The
// raw VendorId + wave_size + dp4a/CV/WMMA capability bits are still around
// for fine-grained checks; this enum exists so the dispatcher can express
// intent ("AMD wave64 datacenter/GCN") instead of bare proxies
// ("wave_size >= 64"). Mirrors what the Vulkan backend infers via
// VkPhysicalDeviceProperties + VK_DRIVER_ID, just at lower resolution.
//
// Pascal is bucketed with Turing+ under NV_PASCAL_PLUS because the only
// runtime distinction the dispatcher cares about today is dp4a availability
// (SM 6.4 = Pascal). Hardware MMA / tensor-core gating uses the separate
// wave_mma_supported / cooperative_vector_supported flags.
enum dx12_arch_family {
    DX12_ARCH_UNKNOWN = 0,
    DX12_ARCH_NV_LEGACY,        // pre-Pascal NVIDIA (no dp4a)
    DX12_ARCH_NV_PASCAL_PLUS,   // Pascal/Volta/Turing/Ampere/Ada/Hopper/Blackwell
    DX12_ARCH_AMD_WAVE64,       // GCN, CDNA (Vega, MI-series, wave64 mode)
    DX12_ARCH_AMD_RDNA,         // RDNA1/2/3/4 (wave32 consumer)
    DX12_ARCH_INTEL_UHD,             // Gen9, Xe-LP — wave8 integrated
    DX12_ARCH_INTEL_XE_HPG_PLUS,     // Xe-HPG (Arc A / Alchemist), Xe2 (Arc B / Battlemage, Lunar Lake iGPU), Xe3 (Panther Lake iGPU) — wave>=16
    DX12_ARCH_QUALCOMM,
    DX12_ARCH_APPLE,
    DX12_ARCH_MICROSOFT_WARP,
    DX12_ARCH_OTHER,
};

static const char * dx12_arch_family_str(dx12_arch_family a) {
    switch (a) {
        case DX12_ARCH_NV_LEGACY:       return "NV-legacy";
        case DX12_ARCH_NV_PASCAL_PLUS:  return "NV-Pascal+";
        case DX12_ARCH_AMD_WAVE64:      return "AMD-wave64";
        case DX12_ARCH_AMD_RDNA:        return "AMD-RDNA";
        case DX12_ARCH_INTEL_UHD:       return "Intel-UHD";
        case DX12_ARCH_INTEL_XE_HPG_PLUS: return "Intel-Xe-HPG+";
        case DX12_ARCH_QUALCOMM:        return "Qualcomm";
        case DX12_ARCH_APPLE:           return "Apple";
        case DX12_ARCH_MICROSOFT_WARP:  return "MS-WARP";
        case DX12_ARCH_OTHER:           return "other";
        case DX12_ARCH_UNKNOWN:
        default:                        return "unknown";
    }
}

// Sub-family — capability-keyed refinement of arch_family. Only AMD has
// useful splits today; an authoritative AMD DeviceId lookup table follows.
//
// IMPORTANT: finer numbered splits (RDNA3 vs 3.5 vs 4 vs 5) are *not*
// detectable from D3D12 capabilities alone. Add them only when a
// dispatcher decision actually needs the distinction, and bring a
// device-id range table with you.
enum dx12_arch_subfamily {
    DX12_SUBARCH_UNKNOWN = 0,
    DX12_SUBARCH_AMD_GCN,          // wave64, no WMMA — Vega, Polaris
    DX12_SUBARCH_AMD_CDNA,         // wave64, matrix HW — MI100/200/300 datacenter
    DX12_SUBARCH_AMD_RDNA1_2,      // wave32, no WMMA — Navi10/14 (RDNA1), Navi21-24 (RDNA2), PS5/XSX, Steam Deck
    DX12_SUBARCH_AMD_RDNA3_PLUS,   // wave32, WMMA32 — Navi31-44 (RDNA3/3.5/4), RDNA3+ APUs (780M/880M/890M/Strix Halo/Krackan)
};

static const char * dx12_arch_subfamily_str(dx12_arch_subfamily s) {
    switch (s) {
        case DX12_SUBARCH_AMD_GCN:          return "AMD-GCN";
        case DX12_SUBARCH_AMD_CDNA:         return "AMD-CDNA";
        case DX12_SUBARCH_AMD_RDNA1_2:      return "AMD-RDNA1/2";
        case DX12_SUBARCH_AMD_RDNA3_PLUS:   return "AMD-RDNA3+";
        case DX12_SUBARCH_UNKNOWN:
        default:                            return "";
    }
}

// AMD DeviceId → sub-family lookup table. Sourced from libdrm
// `amdgpu.ids` (the canonical Mesa list of AMD PCI device IDs). Covers
// every chip currently likely to run a llama.cpp DX12 build on Windows.
// Pre-RDNA chips are bucketed as GCN because their gating needs (wave64,
// no WMMA) are identical from our perspective regardless of whether
// they're GCN1/2/3/4 or Vega/Vega20.
//
// Why a table at all: D3D12 exposes no capability that cleanly separates
// RDNA1/2 from RDNA3+ on Windows. The SM 6.9 WaveMMA tier was a Microsoft
// preview that was deprecated and never shipped, so even genuine WMMA32
// hardware (e.g. AMD 880M, RDNA 3.5) reports WaveMMA=no. The DeviceId is
// the only authoritative signal.
//
// Maintenance: when a new AMD chip ships, add the DeviceId(s) here. The
// authoritative source is `amdgpu.ids` in libdrm:
//   https://gitlab.freedesktop.org/mesa/drm/-/blob/main/data/amdgpu.ids
//
// IMPORTANT: this is for *sub-family* classification only. arch_family
// (RDNA vs WAVE64) is derived from the chosen sub-family in
// dx12_classify_arch_family below.
struct dx12_amd_did_range {
    uint16_t lo;
    uint16_t hi;
    dx12_arch_subfamily sub;
};

static const dx12_amd_did_range AMD_DID_TABLE[] = {
    // -- RDNA 3.5 iGPU (gfx115x) --
    { 0x1114, 0x1114, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Krackan Point (840M / 860M)
    { 0x150E, 0x150E, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Strix Point (880M / 890M)
    { 0x1586, 0x1586, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Strix Halo (8050S / 8060S)

    // -- RDNA 3 iGPU (gfx110x) --
    { 0x15BF, 0x15BF, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Phoenix (740M / 760M / 780M)
    { 0x15C8, 0x15C8, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Phoenix2 (740M)
    { 0x1900, 0x1902, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Hawk Point variants (740M / 760M / 780M / 820M / 840M)

    // -- RDNA 3 dGPU (gfx110x: Navi 31/32/33) --
    { 0x7448, 0x744B, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 31 Pro W7800/7900 series
    { 0x744C, 0x744C, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 31 RX 7900 XTX / XT / GRE / M
    { 0x745E, 0x745E, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 31 Pro W7800
    { 0x7460, 0x7461, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 31 cloud V710
    { 0x7470, 0x7470, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 32 Pro W7700
    { 0x747E, 0x747E, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 32 RX 7800 XT / 7700 / 7700 XT / 7800M
    { 0x7480, 0x7483, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 33 RX 7600 / 7600M / 7700S / Steam Machine
    { 0x7489, 0x7489, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 33 Pro W7500
    { 0x7499, 0x7499, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 33 Pro W7400 / RX 7400 / 7300

    // -- RDNA 4 dGPU (gfx12: Navi 48) --
    { 0x7550, 0x7551, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 48 RX 9070 / 9070 XT / 9070 GRE / AI Pro R9700 / R9600D
    { 0x7590, 0x7590, DX12_SUBARCH_AMD_RDNA3_PLUS }, // Navi 48 RX 9060 / 9060 XT / 9060 XT LP

    // -- RDNA 2 iGPU (gfx103x) --
    { 0x1506, 0x1506, DX12_SUBARCH_AMD_RDNA1_2 },    // Mendocino (610M)
    { 0x164E, 0x164E, DX12_SUBARCH_AMD_RDNA1_2 },    // Raphael (610M)
    { 0x1681, 0x1681, DX12_SUBARCH_AMD_RDNA1_2 },    // Rembrandt (660M / 680M)

    // -- RDNA 2 dGPU (gfx103x: Navi 21/22/23/24) --
    { 0x73A0, 0x73FF, DX12_SUBARCH_AMD_RDNA1_2 },    // Navi 21/22/23 RX 6600..6950 + V620 + Pro W6600/6800
    { 0x7421, 0x743F, DX12_SUBARCH_AMD_RDNA1_2 },    // Navi 24 RX 6300/6400/6500 + Pro W6300/W6400/W6500M

    // -- RDNA 1 dGPU (gfx101x: Navi 10/12/14) --
    { 0x731F, 0x731F, DX12_SUBARCH_AMD_RDNA1_2 },    // Navi 10 RX 5600/5700/5700 XT
    { 0x7312, 0x7312, DX12_SUBARCH_AMD_RDNA1_2 },    // Navi 10 Pro W5700
    { 0x7340, 0x734F, DX12_SUBARCH_AMD_RDNA1_2 },    // Navi 14 RX 5300/5500 + Pro W5500
    { 0x7360, 0x7362, DX12_SUBARCH_AMD_RDNA1_2 },    // Navi 12 Pro V520/V540 + Pro 5600M

    // -- CDNA (gfx9xx datacenter) --
    { 0x738C, 0x738C, DX12_SUBARCH_AMD_CDNA },       // CDNA1 MI100
    { 0x7408, 0x7408, DX12_SUBARCH_AMD_CDNA },       // CDNA2 MI250X
    { 0x740C, 0x740C, DX12_SUBARCH_AMD_CDNA },       // CDNA2 MI250/250X
    { 0x740F, 0x740F, DX12_SUBARCH_AMD_CDNA },       // CDNA2 MI210
    { 0x74A0, 0x74BD, DX12_SUBARCH_AMD_CDNA },       // CDNA3 MI300A/300X/308X/325X (+ HF / VF variants)
    { 0x75A0, 0x75B3, DX12_SUBARCH_AMD_CDNA },       // CDNA4 MI350X/355X (+ VF variants)
    { 0x66A0, 0x66AF, DX12_SUBARCH_AMD_CDNA },       // Vega 20 datacenter (MI50/60/Pro VII/Radeon VII — first AMD matrix-capable parts)

    // -- GCN / Vega consumer iGPU + dGPU (wave64, no useful matrix HW) --
    { 0x15D8, 0x15D8, DX12_SUBARCH_AMD_GCN },        // Raven Ridge / Picasso iGPU (Vega, GCN5)
    { 0x15DD, 0x15DD, DX12_SUBARCH_AMD_GCN },        // Raven Ridge / Picasso (Vega)
    { 0x6860, 0x687F, DX12_SUBARCH_AMD_GCN },        // Vega 10 (Vega 56/64, Frontier, MI25, Pro WX 8200/9100, Pro V340)
    { 0x694C, 0x694E, DX12_SUBARCH_AMD_GCN },        // Kaby Lake-G Vega M (GH / GL)
    { 0x6980, 0x699F, DX12_SUBARCH_AMD_GCN },        // Polaris 12 Pro WX 2100/3100/3200 + RX 540/640 + E9171
    { 0x67C0, 0x67FF, DX12_SUBARCH_AMD_GCN },        // Polaris 10/11 RX 460/470/480/560/570/580/590 + Pro WX 4100/5100/7100
    { 0x6FDF, 0x6FDF, DX12_SUBARCH_AMD_GCN },        // Polaris 30 (RX 580 2048SP / 590 GME)
    { 0x6600, 0x66AF, DX12_SUBARCH_AMD_GCN },        // GCN1-3 mobile + HD 7xxx/R7/R9 200/300 series + Mars/Tropo
    { 0x67A0, 0x67BF, DX12_SUBARCH_AMD_GCN },        // Hawaii R9 290/290X + FirePro W8100/W9100
    { 0x6780, 0x679F, DX12_SUBARCH_AMD_GCN },        // Tahiti HD 7900 + W8000/W9000
    { 0x6800, 0x683F, DX12_SUBARCH_AMD_GCN },        // GCN Cape Verde / Pitcairn HD 7700/7800/7900M
    { 0x6900, 0x693F, DX12_SUBARCH_AMD_GCN },        // Tonga / Iceland R5/R7/R9 + Fiji R9 Fury
    { 0x7300, 0x7300, DX12_SUBARCH_AMD_GCN },        // Fiji R9 Fury / Fury X / Pro Duo
};

static dx12_arch_subfamily dx12_amd_subfamily_from_device_id(uint32_t device_id) {
    uint16_t did = (uint16_t)device_id;
    for (const auto & r : AMD_DID_TABLE) {
        if (did >= r.lo && did <= r.hi) {
            return r.sub;
        }
    }
    return DX12_SUBARCH_UNKNOWN;
}

// Map a (known) AMD sub-family to its arch_family bucket. Centralised so
// the classifier and the sub-family lookup agree.
static dx12_arch_family dx12_amd_arch_from_subfamily(dx12_arch_subfamily sub) {
    switch (sub) {
        case DX12_SUBARCH_AMD_RDNA1_2:
        case DX12_SUBARCH_AMD_RDNA3_PLUS:
            return DX12_ARCH_AMD_RDNA;
        case DX12_SUBARCH_AMD_GCN:
        case DX12_SUBARCH_AMD_CDNA:
            return DX12_ARCH_AMD_WAVE64;
        default:
            return DX12_ARCH_OTHER;
    }
}

// Classify the GPU into an arch family from values already probed by init().
// Heuristic: VendorId + DeviceId (AMD) or VendorId + wave_size + dp4a
// (NVIDIA / Intel). For AMD we use the static DeviceId table above for
// accuracy; if a chip is too new for the table we fall back to
// wave_size_min (RDNA reports Min=32 Max=64, GCN/CDNA Min=Max=64) for a
// safe-ish guess.
//
// `wave_size_min` is the raw WaveLaneCountMin probe (vs `wave_size` which
// is forced to WaveLaneCountMax on AMD for DXIL compile compatibility).
static dx12_arch_family dx12_classify_arch_family(UINT vendor_id,
                                                  uint32_t wave_size,
                                                  uint32_t wave_size_min,
                                                  uint32_t device_id,
                                                  bool dp4a) {
    switch (vendor_id) {
        case dx12_vendor::NVIDIA:
            return dp4a ? DX12_ARCH_NV_PASCAL_PLUS : DX12_ARCH_NV_LEGACY;
        case dx12_vendor::AMD: {
            dx12_arch_subfamily known = dx12_amd_subfamily_from_device_id(device_id);
            if (known != DX12_SUBARCH_UNKNOWN) {
                return dx12_amd_arch_from_subfamily(known);
            }
            // Unknown DeviceId — fall back to wave_size_min heuristic.
            // RDNA chips report Min=32 (compute may pick 32 or 64);
            // GCN/CDNA report Min=Max=64.
            return (wave_size_min >= 64) ? DX12_ARCH_AMD_WAVE64 : DX12_ARCH_AMD_RDNA;
        }
        case dx12_vendor::INTEL:
            // Gen9 / Xe-LP iGPUs report wave8. Xe-HPG (Arc A / Alchemist),
            // Xe2 (Arc B / Battlemage, Lunar Lake iGPU) and Xe3 (Panther
            // Lake iGPU) all report wave>=16. Bucket any wave>=16 Intel as
            // Xe-HPG+. Note: this lumps discrete Arc with modern integrated
            // Xe — gating purely on this family is "wave>=16 architecture",
            // not "discrete vs integrated".
            return (wave_size >= 16) ? DX12_ARCH_INTEL_XE_HPG_PLUS : DX12_ARCH_INTEL_UHD;
        case dx12_vendor::QUALCOMM:   return DX12_ARCH_QUALCOMM;
        case dx12_vendor::APPLE:      return DX12_ARCH_APPLE;
        case dx12_vendor::MICROSOFT:  return DX12_ARCH_MICROSOFT_WARP;
        default:                      return DX12_ARCH_OTHER;
    }
}

// Classify into a sub-family. For AMD: uses the DeviceId table for an
// accurate answer; falls back to wave_mma_supported if the DeviceId is
// unknown (newer chip we haven't tabled yet). For all other vendors
// today: returns DX12_SUBARCH_UNKNOWN — caller should fall back to
// arch_family. NV/Intel sub-arch would need their own DeviceId tables.
static dx12_arch_subfamily dx12_classify_arch_subfamily(dx12_arch_family fam,
                                                        UINT vendor_id,
                                                        uint32_t device_id,
                                                        bool wave_mma) {
    if (vendor_id == dx12_vendor::AMD) {
        dx12_arch_subfamily known = dx12_amd_subfamily_from_device_id(device_id);
        if (known != DX12_SUBARCH_UNKNOWN) {
            return known;
        }
        // Unknown DeviceId — fall back to old wave_mma heuristic. WaveMMA
        // is a deprecated D3D12 preview that no driver currently exposes
        // for AMD, so in practice this only fires on chips too new for
        // our table.
        switch (fam) {
            case DX12_ARCH_AMD_WAVE64:
                return wave_mma ? DX12_SUBARCH_AMD_CDNA : DX12_SUBARCH_AMD_GCN;
            case DX12_ARCH_AMD_RDNA:
                return wave_mma ? DX12_SUBARCH_AMD_RDNA3_PLUS : DX12_SUBARCH_AMD_RDNA1_2;
            default:
                return DX12_SUBARCH_UNKNOWN;
        }
    }
    return DX12_SUBARCH_UNKNOWN;
}


struct dx12_device {
    ComPtr<IDXGIAdapter1>     adapter;
    ComPtr<ID3D12Device>      device;
    ComPtr<ID3D12CommandQueue> compute_queue;

    DXGI_ADAPTER_DESC1        adapter_desc = {};
    size_t                    vram_total   = 0;
    size_t                    vram_free    = 0;

    bool cooperative_vector_supported = false;

    // D3D12 Enhanced Barriers (ID3D12GraphicsCommandList7::Barrier). Enabled by
    // default when the device reports EnhancedBarriersSupported (disable via
    // DX12_ENHANCED_BARRIERS=0); hot-path UAV barriers are emitted as scoped
    // GLOBAL/BUFFER barriers (COMPUTE_SHADING -> COMPUTE_SHADING,
    // UNORDERED_ACCESS access) instead of legacy full-drain UAV barriers,
    // mirroring Vulkan's fine-grained model.
    bool enhanced_barriers = false;

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

    // Reported `WaveLaneCountMin` from D3D12_FEATURE_DATA_D3D12_OPTIONS1. On
    // AMD RDNA this is 32 while wave_size (= WaveLaneCountMax) is 64; on
    // AMD GCN/CDNA both are 64; on NVIDIA both are 32; on Intel UHD both are
    // 8 (Xe-HPG/Arc=16, Xe2=16/32). Required to distinguish AMD-RDNA from
    // AMD-wave64 because we deliberately set wave_size from `WaveLaneCountMax`
    // on AMD for DXIL compile compatibility, which erases the RDNA signal in
    // wave_size alone.
    uint32_t wave_size_min = 32;

    // Coarse architecture family (NV-Pascal+, AMD-RDNA, Intel-Xe-HPG+, ...).
    // Populated by dx12_classify_arch_family() at end of init() once
    // vendor/wave/dp4a are known. Use this instead of raw VendorId or
    // wave_size proxies when expressing dispatcher intent.
    dx12_arch_family arch_family = DX12_ARCH_UNKNOWN;

    // Capability-keyed sub-family refinement. Populated alongside
    // arch_family. Today only AMD has useful splits (RDNA1/2 vs RDNA3+,
    // GCN vs CDNA), keyed off wave_mma_supported. UNKNOWN for NV / Intel /
    // everything we don't refine.
    dx12_arch_subfamily sub_family = DX12_SUBARCH_UNKNOWN;

    // Highest supported shader model (raw enum from CheckFeatureSupport).
    // Surfaced so dispatchers can gate on SM 6.6+ (BF16 intrinsics, packed
    // ops) without re-probing.
    D3D_SHADER_MODEL highest_shader_model = D3D_SHADER_MODEL_6_0;

    // BF16 (bfloat16) shader support. SM 6.6 introduced bfloat16 intrinsics
    // (CONVERT_TO_BFLOAT, dot2add_bf16packed, etc.). We don't have any
    // BF16-only shaders today but the field lets future paths gate cleanly,
    // mirroring Vulkan's `bf16` capability flag.
    bool bf16_supported = false;

    // UMD (user-mode driver) version, packed into a 64-bit LARGE_INTEGER by
    // IDXGIAdapter::CheckInterfaceSupport(D3D11Device). Surfaced for the
    // device-init banner and for any future driver-version-keyed
    // workarounds. Zero means the query failed (rare; happens on WARP).
    uint64_t driver_version_raw = 0;
    std::string driver_version_str;  // "31.0.15.5222" form

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
    // is_igpu mirrors the architectural classification at detection time and
    // is NOT affected by DX12_NO_UMA (which is a perf opt-out, not a
    // hardware change).  Used to report GGML_BACKEND_DEVICE_TYPE_IGPU so
    // llama.cpp's default device-select skips integrated GPUs when a
    // discrete GPU is present (matches Vulkan behavior).
    bool is_igpu = false;
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
    //  v9 -> v10: added q5k_dp4a_m_32_threshold (M-aware Q5_K selection — on
    //             AMD 880M Phi-3 ffn_up K=3072 M=9216 prefers 32t while
    //             M=256 prefers 256t; pick per-dispatch based on src0->ne[1]).
    //  v10 -> v11: extended Q5_K M-sweep to 3 points (added M=4096 mid-point,
    //              raised large endpoint from 8192 to 32768 to cover Phi-3
    //              vocab projection M=32064). Crossover detection is now
    //              piecewise (lo->mid or mid->hi) instead of single linear
    //              interp over the full range; preserves small-M resolution
    //              while making the autotune see real large-M workload shapes.
    //              Also fixed a latent bug in the linear-interp formula that
    //              made it silently fall through to the midpoint of the
    //              probed M range whenever 256t won at small M and 32t won
    //              at large M (the entire intended Q5_K use case) — the
    //              v10 thresholds were effectively just (lo+hi)/2 = 4224.
    static constexpr int TUNE_VERSION = 11;
    bool tuning_done = false;
    bool q4k_dp4a_use_32 = false; // Q4_K dp4a matvec: true=32 threads, false=256 threads (default=256)
    bool q5k_dp4a_use_32 = false; // Q5_K dp4a matvec: true=32 threads, false=256 threads (default=256)
    bool f16_mr_use_256  = false; // F16/F32 matvec:   true=256 threads (mr), false=32 threads (mr32)
    // K-aware F16 mr selection: when K (src0->ne[0]) >= this threshold use the
    // 256-thread mr variant, otherwise use the 32-thread mr32 variant. Set to
    // UINT32_MAX to always use mr32 (the historical default for non-AMD-wave64
    // devices) and 0 to always use mr256 (matches f16_mr_use_256=true).
    uint32_t f16_mr_k_256_threshold = 0xFFFFFFFFu;
    // M-aware Q5_K dp4a selection: when M (src0->ne[1]) >= this threshold use
    // the 32-thread variant, otherwise use the 256-thread variant. Set to
    // UINT32_MAX to never use 32t (preserves the historical default of
    // q5k_dp4a_use_32=false meaning always 256t) and 0 to always use 32t
    // (matches q5k_dp4a_use_32=true). Inverse semantics from f16 because the
    // default Q5_K dp4a shader is 256t and 32t is the opt-in for large M.
    uint32_t q5k_dp4a_m_32_threshold = 0xFFFFFFFFu;

    void run_autotune();

    // Pipeline cache
    std::mutex pipeline_mutex;
    std::unordered_map<dx12_pipeline_key, dx12_pipeline, dx12_pipeline_key_hash> pipeline_cache;

    // Fast-path: skip mutex + map lookup when consecutive nodes use the same pipeline
    dx12_pipeline_key  last_pipeline_key = {};
    dx12_pipeline *    last_pipeline_ptr = nullptr;

    // Common root signature for most shaders
    ComPtr<ID3D12RootSignature> common_root_sig;
    bool use_param_cbv = false;

    // Split-KV temp buffer for flash attention (1 MB, lazily created)
    ComPtr<ID3D12Resource> splitkv_temp;
    static constexpr size_t SPLITKV_TEMP_SIZE = 1024 * 1024; // 1 MB

    // ARGSORT/TOP_K large-N scratch (lazily created, grows on demand).
    // Layout: per row, ncols_padded slots of int2(col_idx, value_bits).
    // Bound at root slot 6 (register u1) for the argsort_large dispatches.
    // Released after the cmd-list drains (retired list pattern matches
    // q8_1_scratch).
    ComPtr<ID3D12Resource>              argsort_scratch;
    size_t                              argsort_scratch_size = 0;
    std::vector<ComPtr<ID3D12Resource>> argsort_scratch_retired;

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
        flush_uploads();
        if (xfer.fence_value == 0) return;
        if (xfer.fence->GetCompletedValue() >= xfer.fence_value) return;
        xfer.fence->SetEventOnCompletion(xfer.fence_value, xfer.fence_event);
        WaitForSingleObject(xfer.fence_event, INFINITE);
    }

    void xfer_wait_value(uint64_t value) {
        if (value == 0) return;
        if (xfer.fence->GetCompletedValue() >= value) return;
        xfer.fence->SetEventOnCompletion(value, xfer.fence_event);
        WaitForSingleObject(xfer.fence_event, INFINITE);
    }

    // Upload staging ring.  The single-buffered xfer staging forces a blocking
    // fence wait on every host->device tensor upload; the scheduler issues one
    // per split input on each token, so that wait dominates the inter-graph CPU
    // window and leaves the GPU idle.  Each ring slot owns its allocator, list
    // and staging buffer, so an upload only blocks when the ring wraps onto a
    // slot whose copy has not completed yet.
    struct upload_slot {
        ComPtr<ID3D12CommandAllocator>    cmd_alloc;
        ComPtr<ID3D12GraphicsCommandList> cmd_list;
        ComPtr<ID3D12Resource>            staging;
        void *                            mapped      = nullptr;
        size_t                            capacity    = 0;
        uint64_t                          fence_value = 0;
    };
    static constexpr size_t UPLOAD_RING_SIZE = 8;
    // Per-slot upload cap; bounds the ring at UPLOAD_RING_SIZE x this value.
    static constexpr size_t UPLOAD_SLOT_MAX  = 4 * 1024 * 1024;
    upload_slot upload_ring[UPLOAD_RING_SIZE];
    size_t      upload_ring_head = 0;
    std::vector<upload_slot *> pending_uploads;

    // Submit every recorded upload copy in a single ExecuteCommandLists call.
    // Called before any operation that must observe the uploaded bytes.
    void flush_uploads() {
        if (pending_uploads.empty()) return;
        ID3D12CommandList * lists[UPLOAD_RING_SIZE];
        size_t n = 0;
        for (auto * s : pending_uploads) {
            lists[n++] = s->cmd_list.Get();
        }
        compute_queue->ExecuteCommandLists((UINT)n, lists);
        xfer.fence_value++;
        compute_queue->Signal(xfer.fence.Get(), xfer.fence_value);
        for (auto * s : pending_uploads) {
            s->fence_value = xfer.fence_value;
        }
        pending_uploads.clear();
    }

    upload_slot * upload_ring_acquire(size_t size) {
        if (pending_uploads.size() >= UPLOAD_RING_SIZE) {
            flush_uploads();
        }
        upload_slot & s = upload_ring[upload_ring_head];
        upload_ring_head = (upload_ring_head + 1) % UPLOAD_RING_SIZE;

        xfer_wait_value(s.fence_value);

        if (!s.cmd_alloc) {
            HRESULT hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                        IID_PPV_ARGS(&s.cmd_alloc));
            DX12_CHECK(hr, "CreateCommandAllocator(upload ring)");
            hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                           s.cmd_alloc.Get(), nullptr,
                                           IID_PPV_ARGS(&s.cmd_list));
            DX12_CHECK(hr, "CreateCommandList(upload ring)");
            s.cmd_list->Close();
        }

        if (s.capacity < size) {
            size_t need = (size + 0xFFFF) & ~(size_t)0xFFFF;
            if (s.mapped && s.staging) {
                D3D12_RANGE wr = { 0, 0 };
                s.staging->Unmap(0, &wr);
                s.mapped = nullptr;
            }
            s.staging.Reset();
            D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = need; rd.Height = 1; rd.DepthOrArraySize = 1;
            rd.MipLevels = 1; rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s.staging));
            DX12_CHECK(hr, "CreateCommittedResource(upload ring staging)");
            D3D12_RANGE no_read = { 0, 0 };
            hr = s.staging->Map(0, &no_read, &s.mapped);
            DX12_CHECK(hr, "Map upload ring staging");
            s.capacity = need;
        }
        return &s;
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
    // Stable string of the form "DX12:<VID>_<DID>" (hex, 4 digits each), used
    // for ggml_backend_dev_props::device_id so external tools (notably
    // llama-mmv-tune validate-cache) can map the chosen backend device to the
    // matching autotune cache file ($LOCALAPPDATA/.ggml_dx12_tune_<VID>_<DID>.txt).
    std::string device_id_str;

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
        for (auto & s : upload_ring) {
            if (s.mapped && s.staging) {
                D3D12_RANGE wr = { 0, 0 };
                s.staging->Unmap(0, &wr);
                s.mapped = nullptr;
            }
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

// Command-allocator ring depth. The CPU may run at most (ring - 1) submissions
// ahead of the GPU before `ensure_cmd_list_open` has to block on the slot it is
// about to recycle. A decode token issues ~16 submissions (uploads + compute
// stream chunks + logits readback), so a shallow ring forces several CPU<->GPU
// lock-steps per token. CMD_RING_MAX bounds the fixed-size arrays; the live
// depth is per-context and overridable via DX12_CMD_RING.
static const int CMD_RING_MAX     = 16;
static const int CMD_RING_DEFAULT = 4;

// ---------------------------------------------------------------------------
// Whole-graph command-list replay (enabled by default; disable via DX12_COMMAND_REPLAY=0)
// ---------------------------------------------------------------------------
// Prototype that eliminates per-token D3D12 command recording for stable M=1
// decode graphs.  On the first stable token we record the entire decode graph
// into a dedicated command allocator + closed command list, with shader params
// written into a dedicated persistent CBV region at stable GPU VAs (distinct
// from the 4-slot ring's param_upload).  On subsequent tokens we validate that
// nothing the baked list depends on has changed (per-node identity via the
// existing replay_cache, resource base VAs, and per-FA n_splits/groups) and, if
// so, simply re-ExecuteCommandLists the already-closed list -- skipping the
// ~1.67ms of SetPipelineState / root-descriptor / Dispatch calls.  The only
// per-token dynamic state baked into the CBV in a steady decode graph is flash
// attention's N_kv-derived n_splits; when N_kv changes without changing n_splits
// the fresh FA params are written into the same fixed CBV slot; when n_splits
// (and thus the baked group count) changes the capture is invalidated and
// re-recorded.  Projection post-op fusion (DX12_MMV_POSTOP_FUSION) is replay-
// compatible: the fused matvec bakes only static RoPE/scatter op_params, and the
// per-token positions and KV row indices ride DATA_VOLATILE root SRVs the host
// refreshes each token -- no CBV patching needed for them.
struct dx12_cmd_replay {
    bool enabled = false;   // DX12_COMMAND_REPLAY
    bool stats   = false;   // DX12_COMMAND_REPLAY_STATS

    ComPtr<ID3D12CommandAllocator>    alloc;
    ComPtr<ID3D12GraphicsCommandList> list;    // closed after capture, re-executed on replay
    ComPtr<ID3D12GraphicsCommandList> saved_cmd_list; // bctx->cmd_list stashed during capture

    // Dedicated persistent parameter region (UPLOAD heap, stable GPU VA).
    ComPtr<ID3D12Resource> cbv;
    uint8_t *                 cbv_mapped = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS cbv_base   = 0;
    size_t                    cbv_size   = 0;
    size_t                    cbv_cursor = 0;   // monotonic during capture
    size_t                    last_param_offset = 0; // offset of the most recent capture write

    bool     capturing      = false;  // recording into the dedicated list right now
    bool     captured       = false;  // a valid closed list is available for replay
    bool     capture_ok     = true;   // cleared on overflow / unexpected realloc during capture
    bool     disabled_perm  = false;  // permanently disabled after excessive thrashing
    uint64_t last_fence     = 0;      // fence value of the last capture/replay submit
    uint64_t signature      = 0;      // resource-base-VA signature at capture time
    int      n_nodes        = 0;      // graph node count at capture time
    int      stable_streak  = 0;      // consecutive identity-stable tokens
    int      replays_since_capture = 0;
    int      thrash_count   = 0;

    // Per-FLASH_ATTN_EXT dynamic linkage captured during recording.
    struct fa_rec {
        int      node_index;
        size_t   cbv_offset;
        uint32_t total_groups_no_split;
        uint32_t target_groups;
        uint32_t min_kv_per_split;
        uint32_t gqa_ratio;
        uint32_t n_kv;      // updated when the CBV slot is patched
        uint32_t n_splits;
    };
    std::vector<fa_rec> fa;

    // Diagnostics
    uint64_t captures = 0, replays = 0, invalidations = 0, records = 0, patches = 0;
};

struct dx12_backend_context {
    dx12_device * dev = nullptr;

    // Command allocator ring — 3 allocators so CPU can record while GPU executes
    ComPtr<ID3D12CommandAllocator>    cmd_allocs[CMD_RING_MAX];
    uint64_t                          cmd_alloc_fence[CMD_RING_MAX] = {}; // fence value when submitted
    int                               cmd_ring_head = 0; // next allocator to use
    int                               cmd_ring_size = CMD_RING_DEFAULT;
    ComPtr<ID3D12GraphicsCommandList> cmd_list;
    // Cached ID3D12GraphicsCommandList7 view of cmd_list, for Enhanced
    // Barriers. Re-queried whenever the underlying cmd_list pointer changes
    // (e.g. COMMAND_REPLAY swaps in a captured list).
    ComPtr<ID3D12GraphicsCommandList7> cmd_list7;
    ID3D12GraphicsCommandList *        cmd_list7_src = nullptr;

    // Lazily resolve (and cache) the CommandList7 view of the current cmd_list.
    // Returns nullptr if the runtime does not expose the interface.
    ID3D12GraphicsCommandList7 * get_cmd_list7() {
        ID3D12GraphicsCommandList * raw = cmd_list.Get();
        if (raw != cmd_list7_src) {
            cmd_list7.Reset();
            if (raw) raw->QueryInterface(IID_PPV_ARGS(&cmd_list7));
            cmd_list7_src = raw;
        }
        return cmd_list7.Get();
    }

    // Global compute UAV ordering point. Legacy: null-resource UAV barrier
    // (full pipeline drain). Enhanced: GLOBAL barrier scoped to compute
    // shading + unordered-access, which lets the driver avoid a full drain.
    // Global barriers flush all cached memory, so this is only used where the
    // hazard set is unknown/broad; prefer emit_uav_barrier_buffer/_scoped.
    void emit_uav_barrier_global() {
        if (dev->enhanced_barriers) {
            if (ID3D12GraphicsCommandList7 * cl7 = get_cmd_list7()) {
                D3D12_GLOBAL_BARRIER gb = {};
                gb.SyncBefore   = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
                gb.SyncAfter    = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
                gb.AccessBefore = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
                gb.AccessAfter  = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS | D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
                D3D12_BARRIER_GROUP grp = {};
                grp.Type            = D3D12_BARRIER_TYPE_GLOBAL;
                grp.NumBarriers     = 1;
                grp.pGlobalBarriers = &gb;
                cl7->Barrier(1, &grp);
                return;
            }
        }
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = nullptr;
        cmd_list->ResourceBarrier(1, &barrier);
    }

    // Single-buffer compute UAV ordering point. Legacy: null-resource UAV
    // barrier (preserves default behavior). Enhanced: a scoped BUFFER barrier
    // over just this resource -- far cheaper than a GLOBAL/legacy full drain.
    void emit_uav_barrier_buffer(ID3D12Resource * res) {
        if (dev->enhanced_barriers && res) {
            if (ID3D12GraphicsCommandList7 * cl7 = get_cmd_list7()) {
                D3D12_BUFFER_BARRIER bb = {};
                bb.SyncBefore   = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
                bb.SyncAfter    = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
                bb.AccessBefore = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
                bb.AccessAfter  = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS | D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
                bb.pResource    = res;
                bb.Offset       = 0;
                bb.Size         = UINT64_MAX;
                D3D12_BARRIER_GROUP grp = {};
                grp.Type            = D3D12_BARRIER_TYPE_BUFFER;
                grp.NumBarriers     = 1;
                grp.pBufferBarriers = &bb;
                cl7->Barrier(1, &grp);
                return;
            }
        }
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = nullptr;
        cmd_list->ResourceBarrier(1, &barrier);
    }

    // Resource-scoped compute UAV ordering point over a small set of buffers.
    // Legacy: one UAV barrier per resource. Enhanced: BUFFER barriers scoped
    // to compute shading + unordered-access over each resource's full range.
    void emit_uav_barrier_scoped(ID3D12Resource * const * resources, int n) {
        if (n <= 0) return;
        if (dev->enhanced_barriers) {
            if (ID3D12GraphicsCommandList7 * cl7 = get_cmd_list7()) {
                D3D12_BUFFER_BARRIER bb[16];
                for (int i = 0; i < n; i++) {
                    bb[i] = {};
                    bb[i].SyncBefore   = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
                    bb[i].SyncAfter    = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
                    bb[i].AccessBefore = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
                    bb[i].AccessAfter  = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS | D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
                    bb[i].pResource    = resources[i];
                    bb[i].Offset       = 0;
                    bb[i].Size         = UINT64_MAX;
                }
                D3D12_BARRIER_GROUP grp = {};
                grp.Type            = D3D12_BARRIER_TYPE_BUFFER;
                grp.NumBarriers     = (UINT32) n;
                grp.pBufferBarriers = bb;
                cl7->Barrier(1, &grp);
                return;
            }
        }
        D3D12_RESOURCE_BARRIER barriers[16];
        for (int i = 0; i < n; i++) {
            barriers[i] = {};
            barriers[i].Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barriers[i].UAV.pResource = resources[i];
        }
        cmd_list->ResourceBarrier((UINT) n, barriers);
    }
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
    // Set by the fused rms_norm_mul_quantize_q8_1 dispatch to flag the q8_1
    // cache as pre-populated by *us*. The matmul barrier path uses this to
    // skip the "cached src1 in unsynced_writes → invalidate cache" check that
    // would otherwise wipe our fresh quantized scratch. Cleared after the
    // first matmul barrier consumes it.
    bool                      q8_1_cache_safe   = false;

    // Diagnostic counters (DX12_QUANT_STATS): per-graph tally of Q8_1
    // activation-prep dispatches actually issued vs skipped via the reuse
    // cache vs pre-populated by a fused norm+quantize dispatch. Used to
    // quantify the norm/activation fusion's dispatch reduction. Reset each
    // graph_compute; printed at graph end when the env flag is set.
    uint64_t                  dbg_q8_quant_dispatched = 0;
    uint64_t                  dbg_q8_quant_reused     = 0;
    uint64_t                  dbg_q8_norm_prepop      = 0;

    // DX12_BARRIER_STATS: per-graph tally of UAV barriers emitted, split by
    // scoped (resource-range) vs global (full-pipeline drain).  Used to
    // quantify the barrier-elision path's serialization reduction.
    uint64_t                  dbg_barrier_scoped      = 0;
    uint64_t                  dbg_barrier_global      = 0;

    // Cross-frame adaptive submit threshold (Vulkan ggml-vulkan.cpp:16308-16322).
    // The previous graph_compute records its total estimated FLOPs (matmul,
    // conv, attention); the next call uses last/40 as the per-submit FLOP
    // threshold, clamped to a cap (200 GFLOP on discrete GPUs, lower on weak
    // iGPUs — see the flops_cap logic in dx12_graph_compute).  This kicks the
    // GPU early on dense models (few large matmuls) and avoids over-submitting
    // on sparse models (many tiny ops).  Init UINT64_MAX so the first graph
    // uses the full flops_cap rather than 0 (which would disable the trigger).
    uint64_t last_total_flops = UINT64_MAX;

    // Same estimate, but recorded only for decode (non-prompt) graphs, for the
    // command-replay size gate below.  A prompt graph is 10-35x larger than the
    // decode graph of the same model (SmolLM2-135M Q4_K: 7.47 vs 0.220 GFLOP),
    // so last_total_flops cannot be used to classify decode graph size.
    // UINT64_MAX until the first decode graph runs, so the gate stays open.
    uint64_t last_decode_flops = UINT64_MAX;

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

    // Companion to unsynced_writes for write-after-read hazards.  Records the
    // GPU memory regions (resource + byte range) read by dispatches since the
    // last UAV barrier, so a later dispatch that writes into memory the graph
    // allocator has recycled -- while an earlier dispatch may still be reading
    // the previous tensor -- forces a barrier.  Keyed by memory, not by
    // ggml_tensor pointer, because the recycled tensors have unrelated views.
    struct unsynced_read {
        void *    res;
        uintptr_t lo;
        uintptr_t hi;
        uintptr_t root;
    };
    std::vector<unsynced_read> unsynced_reads;

    // Byte-range companion to unsynced_writes (which is keyed by ggml_tensor
    // pointer).  Records the memory region each unsynced dispatch wrote so a
    // later read/write into the same memory through an aliased view can be
    // detected precisely instead of via the conservative KV blanket barrier.
    // KV write and read views can share the same root, so range overlap alone
    // determines the hazard. Used only when DX12_PRECISE_KV_BARRIERS is set.
    std::vector<unsynced_read> unsynced_write_ranges;

    // R1 — per-graph cached decisions ("graph replay").  Persistent across
    // calls; invalidated by per-node identity memcmp at the start of every
    // graph_compute.  Skips pipeline lookup, fusion lookahead, and key
    // construction on cache hits — typically 95%+ of decode tokens after
    // warmup.  Disable via DX12_NO_GRAPH_REPLAY=1.
    dx12_replay_cache replay_cache;

    // DX12_PHASE_PROFILE timing state. Values cover one graph_compute followed
    // by its synchronize call and are only populated when explicitly enabled.
    uint64_t phase_graph_start_us  = 0;
    uint64_t phase_record_start_us = 0;
    uint64_t phase_graph_return_us = 0;
    uint64_t phase_submit_us       = 0;
    uint64_t phase_submit_record_us = 0;
    uint64_t phase_alloc_wait_us    = 0;
    uint64_t phase_alloc_wait_post_us = 0;
    uint64_t phase_first_submit_us  = 0;
    uint64_t phase_get_tensor_us    = 0;
    bool     phase_is_prompt       = false;
    // Set by graph_compute, cleared by the synchronize that accounts for it.
    // The scheduler issues ~13 synchronize calls per token but only one follows
    // a graph, so without this every later call would re-account the same graph
    // against an ever-later sync entry and inflate post_graph.
    bool     phase_pending         = false;
    uint64_t phase_sync_calls      = 0;   // total synchronize() entries (all splits)
    uint64_t phase_last_sync_end_us = 0;  // wait_end of the previous accounted graph
    uint64_t phase_sum_gap_us      = 0;   // CPU time outside the graph->sync window
    uint64_t phase_sum_gapsync_us  = 0;   // time inside synchronize() calls made during the gap
    uint64_t phase_sum_settensor_us = 0;  // time inside set_tensor(_async) during the gap
    bool     phase_gap_sync_accounted = false;
    uint64_t phase_decode_count    = 0;
    uint64_t phase_sum_prep_us     = 0;
    uint64_t phase_sum_record_us   = 0;
    uint64_t phase_sum_submit_us   = 0;
    uint64_t phase_sum_wait_us     = 0;
    uint64_t phase_sum_total_us    = 0;
    uint64_t phase_sum_post_graph_us = 0;
    uint64_t phase_sum_get_tensor_us = 0;
    uint64_t phase_sum_alloc_wait_us = 0;
    uint64_t phase_sum_alloc_wait_post_us = 0;
    uint64_t phase_sum_first_submit_us = 0;
    uint64_t phase_decision_us     = 0;
    uint64_t phase_params_us       = 0;
    uint64_t phase_setup_us        = 0;
    uint64_t phase_barrier_us      = 0;
    uint64_t phase_dispatch_us     = 0;
    uint64_t phase_sum_decision_us = 0;
    uint64_t phase_sum_params_us   = 0;
    uint64_t phase_sum_setup_us    = 0;
    uint64_t phase_sum_barrier_us  = 0;
    uint64_t phase_sum_dispatch_us = 0;

    static constexpr size_t PARAM_STRIDE = 256;
    static constexpr size_t PARAM_ENTRIES_PER_SLOT = 8192;
    static constexpr size_t PARAM_SLOT_SIZE = PARAM_STRIDE * PARAM_ENTRIES_PER_SLOT;
    ComPtr<ID3D12Resource> param_upload;
    uint8_t * param_upload_mapped = nullptr;
    size_t param_cursor[CMD_RING_MAX] = {};
    int active_cmd_slot = 0;

    // Whole-graph command-list replay state (opt-in; see dx12_cmd_replay).
    dx12_cmd_replay replay;

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
        q8_1_cache_safe   = false;
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
        if (param_upload && param_upload_mapped) {
            param_upload->Unmap(0, nullptr);
            param_upload_mapped = nullptr;
        }
        if (replay.cbv && replay.cbv_mapped) {
            replay.cbv->Unmap(0, nullptr);
            replay.cbv_mapped = nullptr;
        }
    }

    void ensure_cmd_list_open();
    void set_shader_params(const dx12_shader_params & params, uint32_t num_constants);
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

    bool debug_layer_enabled = false;
    auto try_enable_debug_layer = [&]() {
        if (debug_layer_enabled) return;
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            debug_layer_enabled = true;
            DX12_LOG_INFO("D3D12 debug layer enabled\n");
        }
    };

    // Enable debug layer when DX12_DEBUG env var is set
    if (getenv("DX12_DEBUG")) {
        try_enable_debug_layer();
        // GPU-based validation (catches OOB UAV reads/writes, missing barriers)
        ComPtr<ID3D12Debug1> debug1;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug1)))) {
            if (getenv("DX12_DEBUG_GBV")) {
                debug1->SetEnableGPUBasedValidation(TRUE);
                DX12_LOG_INFO("D3D12 GPU-based validation enabled\n");
            }
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

    // Enumerate adapters. Some drivers (observed: Intel Arc B-series Xe3 iGPU) return
    // DXGI_ERROR_NOT_CURRENTLY_AVAILABLE from CreateCommandQueue unless the D3D12 debug
    // layer has been enabled before factory creation. We detect this and auto-retry
    // with the debug layer enabled. This adds validation overhead but is the only known
    // way to get the device working on those drivers.
    bool saw_queue_creation_failure = false;

    auto enumerate_adapters = [&]() {
        saw_queue_creation_failure = false;
        g_dx12.devices.clear();
        g_dx12.factory.Reset();

        HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&g_dx12.factory));
        DX12_CHECK(hr, "CreateDXGIFactory1");

        // Prefer IDXGIFactory6::EnumAdapterByGpuPreference so the OS gives
        // us discrete GPUs before integrated ones on hybrid systems
        // (otherwise DXGI returns the panel-driving adapter first, which
        // on most laptops is the iGPU). Fall back to EnumAdapters1 on
        // older Windows 10 builds that don't expose IDXGIFactory6.
        ComPtr<IDXGIFactory6> factory6;
        const bool have_factory6 = SUCCEEDED(g_dx12.factory.As(&factory6));

        for (UINT i = 0; ; ++i) {
            ComPtr<IDXGIAdapter1> adapter;
            if (have_factory6) {
                hr = factory6->EnumAdapterByGpuPreference(
                    i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                    IID_PPV_ARGS(&adapter));
            } else {
                hr = g_dx12.factory->EnumAdapters1(i, &adapter);
            }
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            DX12_CHECK(hr, "EnumAdapter");

            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);

            // Skip software adapters
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

            // Skip Microsoft virtual adapters (Basic Render Driver, Remote/Indirect
            // Display under RDP). These advertise D3D12 but fail CreateCommandQueue.
            if (desc.VendorId == 0x1414) {
                char name_buf[128];
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf, sizeof(name_buf), nullptr, nullptr);
                DX12_LOG_INFO("Skipping Microsoft virtual adapter: %s\n", name_buf);
                continue;
            }
            {
                char name_buf[128];
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf, sizeof(name_buf), nullptr, nullptr);
                if (strstr(name_buf, "Remote Display") || strstr(name_buf, "Indirect Display")) {
                    DX12_LOG_INFO("Skipping remote/indirect display adapter: %s\n", name_buf);
                    continue;
                }
            }

            ComPtr<ID3D12Device> test_device;
            hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&test_device));
            if (FAILED(hr)) continue;

            // Validate compute capability: try a small UAV buffer allocation.
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

            // Validate compute queue creation up front so we can fall back cleanly.
            {
                D3D12_COMMAND_QUEUE_DESC qd_test = {};
                qd_test.Type     = D3D12_COMMAND_LIST_TYPE_COMPUTE;
                qd_test.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
                qd_test.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
                ComPtr<ID3D12CommandQueue> test_q;
                HRESULT q_hr = test_device->CreateCommandQueue(&qd_test, IID_PPV_ARGS(&test_q));
                if (FAILED(q_hr)) {
                    char name_buf[128];
                    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf, sizeof(name_buf), nullptr, nullptr);
                    DX12_LOG_WARN("Skipping %s: CreateCommandQueue failed (HRESULT 0x%08X)\n", name_buf, (unsigned)q_hr);
                    saw_queue_creation_failure = true;
                    continue;
                }
            }
            test_device.Reset();

            if (g_dx12.devices.size() >= GGML_DX12_MAX_DEVICES) break;

            g_dx12.devices.push_back(std::make_unique<dx12_device>());
            g_dx12.devices.back()->init(std::move(adapter), g_dx12.devices.size() - 1);
        }
    };

    enumerate_adapters();

    // Auto-fallback:if every otherwise-usable adapter failed CreateCommandQueue and
    // we ended up with no devices, retry with the debug layer enabled. Skipped when
    // DX12_NO_DEBUG_FALLBACK=1.
    if (g_dx12.devices.empty() && saw_queue_creation_failure &&
        !debug_layer_enabled && !getenv("DX12_NO_DEBUG_FALLBACK")) {
        DX12_LOG_WARN("All adapters failed CreateCommandQueue; retrying with D3D12 debug layer enabled\n");
        try_enable_debug_layer();
        if (debug_layer_enabled) {
            enumerate_adapters();
        }
    }

    DX12_LOG_BANNER("Found %zu D3D12 device(s)\n", g_dx12.devices.size());
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
    {
        char id_buf[32];
        snprintf(id_buf, sizeof(id_buf), "DX12:%04X_%04X",
                 adapter_desc.VendorId, adapter_desc.DeviceId);
        device_id_str = id_buf;
    }

    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    DX12_CHECK(hr, "D3D12CreateDevice");

    // Route debug-layer messages to stderr when DX12_DEBUG is set.
    // Uses ID3D12InfoQueue1::RegisterMessageCallback (Win10 20H1+) so messages
    // appear in console output instead of OutputDebugString.
    if (getenv("DX12_DEBUG")) {
        ComPtr<ID3D12InfoQueue> info_queue;
        if (SUCCEEDED(device.As(&info_queue))) {
            info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, FALSE);
            info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
        }
        ComPtr<ID3D12InfoQueue1> info_queue1;
        if (SUCCEEDED(device.As(&info_queue1))) {
            DWORD cookie = 0;
            auto cb = [](D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY sev,
                         D3D12_MESSAGE_ID id, LPCSTR desc, void *) {
                const char * sev_str = "INFO";
                if (sev == D3D12_MESSAGE_SEVERITY_CORRUPTION) sev_str = "CORRUPTION";
                else if (sev == D3D12_MESSAGE_SEVERITY_ERROR) sev_str = "ERROR";
                else if (sev == D3D12_MESSAGE_SEVERITY_WARNING) sev_str = "WARNING";
                else if (sev == D3D12_MESSAGE_SEVERITY_MESSAGE) sev_str = "MESSAGE";
                fprintf(stderr, "[D3D12 %s id=%u] %s\n", sev_str, (unsigned)id, desc ? desc : "(null)");
                fflush(stderr);
            };
            HRESULT cb_hr = info_queue1->RegisterMessageCallback(
                cb, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie);
            if (SUCCEEDED(cb_hr)) {
                DX12_LOG_INFO("D3D12 InfoQueue callback registered (cookie=%lu)\n", cookie);
            }
        }
    }

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
    highest_shader_model = highest_sm;

    // Native 16-bit shader ops — required for the `_fp16_dxil` blob variants.
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS4 opts4 = {};
        HRESULT hr2 = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &opts4, sizeof(opts4));
        if (SUCCEEDED(hr2) && opts4.Native16BitShaderOpsSupported) {
            fp16_supported = true;
        }
    }

    // D3D12 Enhanced Barriers -- enabled by default when the device reports
    // support (OPTIONS12). Disable via DX12_ENHANCED_BARRIERS=0.
    enhanced_barriers = false;
    if (dx12_flag_default_on("DX12_ENHANCED_BARRIERS")) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS12 opts12 = {};
        HRESULT hr2 = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &opts12, sizeof(opts12));
        if (SUCCEEDED(hr2) && opts12.EnhancedBarriersSupported) {
            enhanced_barriers = true;
            fprintf(stderr, "ggml-dx12: Enhanced Barriers enabled (scoped compute UAV sync)\n");
        } else {
            fprintf(stderr, "ggml-dx12: Enhanced Barriers not supported by device; using legacy UAV barriers\n");
        }
    }

    // BF16 (bfloat16) support. SM 6.6 introduced bfloat16 intrinsics in DXIL.
    // The D3D12_FEATURE_DATA_D3D12_OPTIONS19 cap (NativeShaderBfloat16Support)
    // is the authoritative bit but is only present on recent Agility SDK
    // headers; treat it as best-effort and fall back to a conservative
    // SM6.6 + Native16BitShaderOps proxy that matches every shipping NVIDIA
    // Ampere+/AMD RDNA2+/Intel Arc-class GPU.
    bf16_supported = false;
    {
        // D3D12_FEATURE_D3D12_OPTIONS19 = 57 (preview value, may shift with SDK).
        // Layout starts with a 32-bit cap (MismatchingOutputDimensionsSupport)
        // followed by NativeShaderBfloat16Support a few fields in. Use a
        // padded buffer + raw byte index probe so we don't depend on the
        // exact struct definition.
        struct Options19Probe {
            UINT32 dword_0;  // MismatchingOutputDimensionsSupported
            UINT32 dword_1;  // SupportedSampleCountsWithNoOutputs
            UINT32 dword_2;  // PointSamplingAddressesNeverRoundUp
            UINT32 dword_3;  // RasterizerDesc2Supported
            UINT32 dword_4;  // NarrowQuadrilateralLinesSupported
            UINT32 dword_5;  // AnisoFilterWithPointMipSupported
            UINT32 dword_6;  // MaxSamplerDescriptorHeapSize
            UINT32 dword_7;  // MaxSamplerDescriptorHeapSizeWithStaticSamplers
            UINT32 dword_8;  // MaxViewDescriptorHeapSize
            UINT32 dword_9;  // ComputeOnlyCustomHeapSupported
            UINT32 dword_10; // NativeShaderBfloat16Support (slot per current spec)
        } probe = {};
        HRESULT hr2 = device->CheckFeatureSupport((D3D12_FEATURE)57, &probe, sizeof(probe));
        if (SUCCEEDED(hr2) && probe.dword_10 != 0) {
            bf16_supported = true;
        }
    }
    // Fallback proxy when Options19 isn't exposed by the runtime. SM 6.6+
    // with native 16-bit ops is necessary but not sufficient — Intel UHD
    // (Gen9 / Xe-LP) advertises SM 6.7 + fp16 yet has no BF16 hardware. So
    // we additionally require an arch family known to ship BF16 (NVIDIA
    // Pascal+, AMD RDNA, Intel Arc-class). Conservative but matches what
    // we'd actually want to dispatch to.
    if (!bf16_supported && highest_sm >= D3D_SHADER_MODEL_6_6 && fp16_supported) {
        // Note: arch_family is filled in below from vendor/wave/dp4a, so we
        // re-derive the family inline here. Same call, same inputs.
        dx12_arch_family af_probe = dx12_classify_arch_family(adapter_desc.VendorId, 0, 0, adapter_desc.DeviceId, dp4a_supported);
        // Re-classify with the wave_size we'll set further down so AMD
        // wave64 vs RDNA wave32 split is correct. wave_size isn't queried
        // yet at this point, so do that probe early enough to use here.
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts1_bf = {};
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opts1_bf, sizeof(opts1_bf))) && opts1_bf.WaveLaneCountMin > 0) {
            bool is_amd = (adapter_desc.VendorId == dx12_vendor::AMD);
            uint32_t ws     = is_amd ? opts1_bf.WaveLaneCountMax : opts1_bf.WaveLaneCountMin;
            uint32_t ws_min = opts1_bf.WaveLaneCountMin;
            af_probe = dx12_classify_arch_family(adapter_desc.VendorId, ws, ws_min, adapter_desc.DeviceId, dp4a_supported);
        }
        switch (af_probe) {
            case DX12_ARCH_NV_PASCAL_PLUS:
            case DX12_ARCH_AMD_RDNA:
            case DX12_ARCH_AMD_WAVE64:
            case DX12_ARCH_INTEL_XE_HPG_PLUS:
                bf16_supported = true;
                break;
            default:
                break;  // Intel-UHD, NV-legacy, Qualcomm, Apple, WARP, other
        }
    }

    // Query wave (warp/subgroup) size for shader variant selection.
    // AMD RDNA: use WaveLaneCountMax because compute shaders run in wave64
    // mode even though WaveLaneCountMin=32. Using Min causes compile-time
    // WARP_SIZE=32 vs runtime wave64 mismatch in reductions.
    // Intel/NVIDIA: use WaveLaneCountMin for best performance.
    // We also store the raw Min separately as `wave_size_min` so the arch
    // classifier can tell AMD-RDNA (Min=32) from AMD-wave64 GCN/CDNA (Min=64).
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts1 = {};
        HRESULT hr2 = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opts1, sizeof(opts1));
        if (SUCCEEDED(hr2) && opts1.WaveLaneCountMin > 0) {
            bool is_amd = (adapter_desc.VendorId == dx12_vendor::AMD);
            wave_size     = is_amd ? opts1.WaveLaneCountMax : opts1.WaveLaneCountMin;
            wave_size_min = opts1.WaveLaneCountMin;
        }
    }

    // Classify into a coarse architecture family. Must come after vendor,
    // wave_size / wave_size_min and dp4a are known. For AMD, the static
    // DeviceId table inside the classifier is authoritative; the wave-size
    // fields are fallback for chips too new to be in the table.
    arch_family = dx12_classify_arch_family(adapter_desc.VendorId, wave_size, wave_size_min, adapter_desc.DeviceId, dp4a_supported);
    sub_family  = dx12_classify_arch_subfamily(arch_family, adapter_desc.VendorId, adapter_desc.DeviceId, wave_mma_supported);

    // User-mode driver version. IDXGIAdapter::CheckInterfaceSupport with
    // __uuidof(IDXGIDevice) returns the UMD version as a packed
    // LARGE_INTEGER. Format is HighPart=(prod<<16)|ver, LowPart=(sub<<16)|build
    // — Microsoft documents the layout in DXGI_ADAPTER_DESC docs but the
    // most-quoted form is "31.0.15.5222" so we decode that way.
    {
        LARGE_INTEGER umd = {};
        HRESULT hr2 = adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd);
        if (SUCCEEDED(hr2)) {
            driver_version_raw = (uint64_t) umd.QuadPart;
            char buf[64];
            unsigned prod  = (unsigned)((umd.HighPart >> 16) & 0xFFFF);
            unsigned ver   = (unsigned)( umd.HighPart        & 0xFFFF);
            unsigned sub   = (unsigned)((umd.LowPart  >> 16) & 0xFFFF);
            unsigned build = (unsigned)( umd.LowPart         & 0xFFFF);
            std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u", prod, ver, sub, build);
            driver_version_str = buf;
        }
    }

    detect_memory_architecture();

    create_common_root_signature();
    init_shader_blobs();

    // Build the arch token: "<family>" or "<family> [<subfamily>]" when a
    // sub-family classification is available (currently AMD only).
    std::string arch_token = dx12_arch_family_str(arch_family);
    const char * sub_str = dx12_arch_subfamily_str(sub_family);
    if (sub_str && sub_str[0] != '\0') {
        arch_token += " [";
        arch_token += sub_str;
        arch_token += "]";
    }

    DX12_LOG_BANNER("Device %zu: %s (%s, VRAM: %.1f GB, arch: %s, SM: 6.%d, wave: %u, CV: %s, WaveMMA: %s%s, dp4a: %s, fp16: %s, bf16: %s, driver: %s)\n",
                  idx, name.c_str(), description.c_str(),
                  (double)vram_total / (1024.0 * 1024.0 * 1024.0),
                  arch_token.c_str(),
                  (int)(highest_sm & 0xF),
                  wave_size,
                  cooperative_vector_supported ? "yes" : "no",
                  wave_mma_supported ? "yes" : "no",
                  wave_mma_supported ? (std::string(" K=") + std::to_string(wave_mma_K) +
                                        " wave=" + std::to_string(wave_mma_wave_size) +
                                        (wave_mma_f16_acc32 ? " f16→f32" : " f16→f16")).c_str() : "",
                  dp4a_supported ? "yes" : "no",
                  fp16_supported ? "yes" : "no",
                  bf16_supported ? "yes" : "no",
                  driver_version_str.empty() ? "?" : driver_version_str.c_str());
}

void dx12_device::create_common_root_signature() {
    // Common root signature layout:
    //   Slot 0: Root constants or a root CBV (dx12_shader_params)
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

    // Whole-graph command-list replay requires CBV params so the baked list's
    // slot-0 binding is a stable GPU VA whose contents can be refreshed.
    // Replay's upload-heap CBV is beneficial on UMA, but costs more than replay
    // saves on discrete GPUs. Preserve explicit replay and CBV overrides.
    const char * replay_env = DX12_GETENV("DX12_COMMAND_REPLAY");
    const bool replay_requested = replay_env && replay_env[0] && replay_env[0] != '0';
    use_param_cbv = (DX12_GETENV("DX12_PARAM_CBV") != nullptr) ||
                    ((is_igpu || replay_requested) &&
                     dx12_flag_default_on("DX12_COMMAND_REPLAY"));
    if (use_param_cbv) {
        params[0].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[0].Descriptor.ShaderRegister = 0; // b0
        params[0].Descriptor.RegisterSpace  = 0;
        params[0].Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    } else {
        params[0].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.ShaderRegister  = 0; // b0
        params[0].Constants.RegisterSpace   = 0;
        params[0].Constants.Num32BitValues  = sizeof(dx12_shader_params) / 4;
    }
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
    cmd_ring_head = (cmd_ring_head + 1) % cmd_ring_size;

    // Only wait if THIS allocator's previous submission hasn't finished
    // Other allocators may still be in-flight — that's fine
    const bool phase_profile = DX12_GETENV("DX12_PHASE_PROFILE") != nullptr;
    const uint64_t alloc_wait_start_us = phase_profile ? dx12_qpc_us() : 0;
    wait_for_fence(cmd_alloc_fence[slot]);
    if (phase_profile) {
        const uint64_t elapsed = dx12_qpc_us() - alloc_wait_start_us;
        phase_alloc_wait_us += elapsed;
        if (phase_graph_return_us != 0) {
            phase_alloc_wait_post_us += elapsed;
        }
    }

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
    active_cmd_slot = slot;
    param_cursor[slot] = 0;
    cmd_list_open = true;
}

void dx12_backend_context::set_shader_params(const dx12_shader_params & params, uint32_t num_constants) {
    if (!dev->use_param_cbv) {
        cmd_list->SetComputeRoot32BitConstants(0, num_constants, &params, 0);
        return;
    }

    // Capture path: write into the dedicated persistent CBV region at a stable
    // GPU VA (distinct from the ring's param_upload) and bind that VA into the
    // dedicated list.  A monotonic cursor gives every dispatch its own slot so
    // the whole closed list can be re-executed without slot aliasing.
    if (replay.capturing) {
        const size_t off = replay.cbv_cursor;
        if (!replay.cbv_mapped || off + PARAM_STRIDE > replay.cbv_size) {
            replay.capture_ok = false;   // overflow -> capture will be discarded
            return;
        }
        memcpy(replay.cbv_mapped + off, &params, sizeof(params));
        cmd_list->SetComputeRootConstantBufferView(0, replay.cbv_base + off);
        replay.last_param_offset = off;
        replay.cbv_cursor = off + PARAM_STRIDE;
        return;
    }

    if (!param_upload) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = PARAM_SLOT_SIZE * cmd_ring_size;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        HRESULT hr = dev->device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&param_upload));
        DX12_CHECK(hr, "CreateCommittedResource(param_upload)");

        D3D12_RANGE read_range = { 0, 0 };
        hr = param_upload->Map(0, &read_range, (void **) &param_upload_mapped);
        DX12_CHECK(hr, "Map(param_upload)");
    }

    size_t & cursor = param_cursor[active_cmd_slot];
    if (cursor + PARAM_STRIDE > PARAM_SLOT_SIZE) {
        GGML_ABORT("DX12 parameter buffer exhausted");
    }

    const size_t offset = (size_t) active_cmd_slot * PARAM_SLOT_SIZE + cursor;
    memcpy(param_upload_mapped + offset, &params, sizeof(params));
    cmd_list->SetComputeRootConstantBufferView(0, param_upload->GetGPUVirtualAddress() + offset);
    cursor += PARAM_STRIDE;
}

void dx12_backend_context::close_and_execute() {
    dev->flush_uploads();

    if (!cmd_list_open) return;

    const bool phase_profile = DX12_GETENV("DX12_PHASE_PROFILE") != nullptr;
    const uint64_t submit_start_us = phase_profile ? dx12_qpc_us() : 0;

    HRESULT hr = cmd_list->Close();
    DX12_CHECK(hr, "CommandList::Close");

    ID3D12CommandList * lists[] = { cmd_list.Get() };
    dev->compute_queue->ExecuteCommandLists(1, lists);

    fence_value++;
    hr = dev->compute_queue->Signal(fence.Get(), fence_value);
    DX12_CHECK(hr, "Signal fence");

    // Record which fence value this allocator was submitted with
    cmd_alloc_fence[active_cmd_slot] = fence_value;

    cmd_list_open = false;
    if (phase_profile) {
        if (phase_first_submit_us == 0 && phase_graph_start_us != 0) {
            phase_first_submit_us = dx12_qpc_us() - phase_graph_start_us;
        }
        const uint64_t elapsed = dx12_qpc_us() - submit_start_us;
        phase_submit_us += elapsed;
        if (phase_graph_return_us == 0) {
            phase_submit_record_us += elapsed;
        }
    }
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

    // Snapshot the architectural classification *before* the DX12_NO_UMA
    // perf opt-out flips is_uma — is_igpu reports what the hardware is,
    // not how we choose to map memory.
    is_igpu = is_uma;

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

    // AMD and NVIDIA iGPUs deliberately skip the host-visible mapped path (see
    // the VendorId check in dx12_buft_alloc_buffer), so UMA alone does not imply
    // direct write.
    const bool direct_write = is_uma &&
                              adapter_desc.VendorId != dx12_vendor::AMD &&
                              adapter_desc.VendorId != dx12_vendor::NVIDIA;
    DX12_LOG_BANNER("Memory architecture: %s\n",
                    !is_uma       ? "discrete-VRAM (staging required)" :
                    direct_write  ? "UMA (host-shared, direct write enabled)"
                                  : "UMA (host-shared, staged writes)");
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

// Maximum size of a single D3D12 committed buffer resource on this device.
// D3D12 caps a single buffer at ~4 GB on most hardware; some UMA drivers fail
// (or remove the device) below that, so iGPUs get a conservative cap. ggml's
// allocator splits large *weight* ranges across multiple buffers to respect
// this, but the compute/scratch buffer is a single allocation that cannot be
// split (e.g. the full-context attention-scores tensor when flash attention is
// disabled).
static size_t dx12_max_single_resource_size(const dx12_device * dev) {
    constexpr size_t max_d3d12_buffer_size   = (size_t)4 * 1024 * 1024 * 1024 - 1;
    constexpr size_t max_amd_uma_buffer_size = (size_t)2 * 1024 * 1024 * 1024 - 1;
    if (dev && dev->is_uma && dev->adapter_desc.VendorId == dx12_vendor::AMD) {
        return max_amd_uma_buffer_size;
    }
    return max_d3d12_buffer_size;
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
        // On integrated GPUs a single committed resource larger than the D3D12
        // per-resource limit does not fail cleanly: the Intel/AMD UMA driver
        // returns DXGI_ERROR_DEVICE_REMOVED (0x887A0005) and destabilizes the
        // device, which then crashes the process during teardown. ggml's
        // allocator already splits weight buffers to respect get_max_size(), so
        // only the compute/scratch buffer -- a single indivisible allocation,
        // e.g. the full-context attention-scores tensor when flash attention is
        // disabled -- can exceed the cap. Reject it up front with an actionable
        // message instead of removing the device. Discrete GPUs (NVIDIA / AMD
        // dGPU) can back allocations well above the nominal 4 GB limit, so only
        // guard UMA devices.
        if (dev->is_uma) {
            const size_t max_res = dx12_max_single_resource_size(dev);
            if (size > max_res) {
                DX12_LOG_ERROR(
                    "cannot allocate a single %.2f GiB buffer on integrated GPU %s "
                    "(D3D12 single-resource limit is %.2f GiB). This usually means the "
                    "attention buffers are too large: enable flash attention (-fa 1) or "
                    "reduce the context size (-c).\n",
                    (double)size    / (1024.0*1024.0*1024.0),
                    dev->name.c_str(),
                    (double)max_res / (1024.0*1024.0*1024.0));
                delete ctx;
                return nullptr;
            }
        }
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
            // NVIDIA iGPUs (Tegra/SoC UMA): a persistently-mapped CUSTOM L0
            // resource is locked resident (non-evictable). When the working set
            // approaches physical RAM the driver cannot page it out and removes
            // the device (DXGI_ERROR_DEVICE_RESET) at the first dispatch instead
            // of paging. The DEFAULT heap is evictable, so it both fits under
            // memory pressure and reads faster on this hardware. (NVIDIA dGPUs
            // are not UMA, so this only affects NVIDIA iGPUs.)
            // Intel iGPUs: CUSTOM L0 + WRITE_BACK works well (no snooping penalty).
            if (dev->adapter_desc.VendorId != dx12_vendor::AMD &&
                dev->adapter_desc.VendorId != dx12_vendor::NVIDIA) {
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

            struct buf_set_timer {
                uint64_t t0;
                bool on;
                ~buf_set_timer() { if (on) { g_dx12_buf_set_us += dx12_qpc_us() - t0; g_dx12_buf_set_calls++; } }
            } timer { 0, DX12_GETENV("DX12_PHASE_PROFILE") != nullptr };
            if (timer.on) timer.t0 = dx12_qpc_us();

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

            // Upload via the staging ring: copy the caller's bytes into a free
            // slot, submit the copy on the compute queue and return without
            // waiting.  Queue order guarantees the data lands before any later
            // dispatch reads it, and get_tensor/xfer_wait() cover the fence.
            ctx->dev->init_xfer();

            // Large transfers (model-load weights) keep the single-buffer
            // blocking path.  Ring slots grow to the largest request and never
            // shrink, so routing a 200 MB weight through them would pin
            // 8 x 200 MB of UPLOAD staging for the process lifetime -- fatal on
            // a memory-constrained iGPU.  Blocking costs nothing at load time;
            // the decode split inputs this ring exists for are kilobytes.
            if (size > dx12_device::UPLOAD_SLOT_MAX) {
                ctx->dev->xfer_wait();
                ctx->dev->xfer_ensure_staging(size, 0);

                memcpy(ctx->dev->xfer.upload_mapped, data, size);

                HRESULT hr = ctx->dev->xfer.cmd_alloc->Reset();
                DX12_CHECK(hr, "xfer cmd_alloc Reset");
                hr = ctx->dev->xfer.cmd_list->Reset(ctx->dev->xfer.cmd_alloc.Get(), nullptr);
                DX12_CHECK(hr, "xfer cmd_list Reset");

                size_t big_offset = dx12_tensor_offset(tensor) + offset;
                ctx->dev->xfer.cmd_list->CopyBufferRegion(ctx->resource.Get(), big_offset,
                                                          ctx->dev->xfer.upload_staging.Get(), 0, size);
                ctx->dev->xfer.cmd_list->Close();

                ID3D12CommandList * lists[] = { ctx->dev->xfer.cmd_list.Get() };
                ctx->dev->compute_queue->ExecuteCommandLists(1, lists);
                ctx->dev->xfer.fence_value++;
                ctx->dev->compute_queue->Signal(ctx->dev->xfer.fence.Get(), ctx->dev->xfer.fence_value);
                ctx->dev->xfer_wait();
                return;
            }

            auto * slot = ctx->dev->upload_ring_acquire(size);

            memcpy(slot->mapped, data, size);

            HRESULT hr = slot->cmd_alloc->Reset();
            DX12_CHECK(hr, "upload ring cmd_alloc Reset");
            hr = slot->cmd_list->Reset(slot->cmd_alloc.Get(), nullptr);
            DX12_CHECK(hr, "upload ring cmd_list Reset");

            size_t tensor_offset = dx12_tensor_offset(tensor) + offset;
            slot->cmd_list->CopyBufferRegion(ctx->resource.Get(), tensor_offset,
                                             slot->staging.Get(), 0, size);
            slot->cmd_list->Close();

            ctx->dev->pending_uploads.push_back(slot);
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

    // See dx12_max_single_resource_size(): D3D12 caps a single buffer near 4 GB
    // (lower on AMD UMA). ggml's generic allocator splits large model ranges
    // into multiple DX12 buffers to respect this.
    return dx12_max_single_resource_size(dev);
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

// Opt-out for the large-N multi-pass ARGSORT/TOP_K path.  Default-on because
// it expands test-backend-ops coverage (the small-N shader caps at 1024
// columns); users running real inference with vocab-sized TOP_K can set
// DX12_DISABLE_LARGE_SORT=1 to fall back to CPU sort, in case the
// O(log^2 ncols) dispatch storm becomes a hot spot.
static bool dx12_env_disable_large_sort() {
    static const bool flag = (getenv("DX12_DISABLE_LARGE_SORT") != nullptr);
    return flag;
}

// Tri-state opt for the block-level MUL_MAT_ID Q4_K shader
// (mul_mat_id_q4k_block.hlsl). The default per-element MMID Q4_K shader
// recomputes the full Q4_K scale/min decode per K iteration; on Intel UHD
// wave=8 with large K this produces wrong results for later thread groups
// (cumulative per-thread register pressure / driver scratch state).
// Returns: -1 = unset (auto), 0 = force off, 1 = force on.
static int dx12_env_mmid_q4k_block() {
    static const int value = []() {
        const char * e = getenv("DX12_MMID_Q4K_BLOCK");
        if (!e) return -1;
        return (atoi(e) != 0) ? 1 : 0;
    }();
    return value;
}

// Accumulated wall time inside dx12_supports_op (DX12_PHASE_PROFILE only). The
// scheduler calls this once per graph node per token, so it sits on the
// critical path between tokens.
static uint64_t g_dx12_supports_op_us = 0;

static bool dx12_supports_op_impl(ggml_backend_dev_t dev, const struct ggml_tensor * op);

static bool dx12_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (!DX12_GETENV("DX12_PHASE_PROFILE")) {
        return dx12_supports_op_impl(dev, op);
    }
    const uint64_t t0 = dx12_qpc_us();
    const bool r = dx12_supports_op_impl(dev, op);
    g_dx12_supports_op_us += dx12_qpc_us() - t0;
    return r;
}

static bool dx12_supports_op_impl(ggml_backend_dev_t dev, const struct ggml_tensor * op) {

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
        case GGML_OP_LOG:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_FILL:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
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

        case GGML_OP_SUM:
        case GGML_OP_MEAN:
            // Parallel reduction is only numerically stable when src0 is
            // contiguous along ne[0]; permuted inputs accumulate enough float
            // drift (vs CPU's f64 accumulator) to fail the strict 1e-7 test
            // tolerance. Match Vulkan's gate (ggml_is_contiguous_rows).
            return op->type == GGML_TYPE_F32 && op->src[0] &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous_rows(op->src[0]);
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
        case GGML_OP_IM2COL_3D:
        case GGML_OP_PAD:
        case GGML_OP_UPSCALE:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            // Same-type quantized CPY/DUP: byte-level block copy. Only CPY
            // and DUP go through the quant block-copy shader; CONCAT/REPEAT/
            // etc. on quants still fall through to the general gate below
            // (which rejects them, matching the prior behavior).
            // Same-type I16 DUP/CPY also routes here because the load_auto
            // path's esize=2 branch goes through f16_to_f32 (lossy for
            // integer bit patterns). The byte-level uint16 copy preserves
            // the bits exactly.
            if ((op->op == GGML_OP_CPY || op->op == GGML_OP_DUP) &&
                op->src[0] && op->src[0]->type == op->type &&
                (op->type == GGML_TYPE_I16 ||
                 (ggml_is_quantized(op->type) &&
                  (ggml_type_size(op->type) & 1u) == 0u))) {
                // Shader uses native uint16_t Load/Store (some quant blocks
                // start at 2-byte but not 4-byte aligned addresses, e.g. the
                // 18-byte Q4_0 block at offset 18). Requires fp16_supported.
                if (!dev) return false;
                auto * d = (dx12_device *)dev->context;
                if (!d || !d->fp16_supported) return false;
                return true;
            }
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
            // I32 bit-preserving copy paths (CPY/DUP/CONT/CONCAT/REPEAT).
            // CPY/DUP shader handles F32 <-> I32 conversion via op_params
            // flags; CONT/CONCAT/REPEAT use load_auto/store_auto with
            // elem_stride=4 which round-trips the I32 bit pattern via
            // asfloat/asuint (no arithmetic on the value).
            if (op->type == GGML_TYPE_I32 &&
                (op->op == GGML_OP_CPY || op->op == GGML_OP_DUP ||
                 op->op == GGML_OP_CONT || op->op == GGML_OP_CONCAT ||
                 op->op == GGML_OP_REPEAT)) {
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    if (!op->src[s]) continue;
                    if (op->op == GGML_OP_CPY || op->op == GGML_OP_DUP) {
                        if (op->src[s]->type != GGML_TYPE_I32 &&
                            op->src[s]->type != GGML_TYPE_F32) return false;
                    } else {
                        if (op->src[s]->type != GGML_TYPE_I32) return false;
                    }
                }
                return true;
            }
            return false;

        case GGML_OP_SET_ROWS:
            // KV cache writes: src0 is F32, dst can be F16 or F32.
            // Phase 9a (opt-in): Q8_0 dst via dedicated quantize-on-store shader.
            if (op->src[1] && op->src[1]->type != GGML_TYPE_I32 &&
                op->src[1]->type != GGML_TYPE_I64) return false;
            if (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 ||
                op->type == GGML_TYPE_BF16) {
                // The SET_ROWS shader reads src0 as F32; decline non-F32 sources
                // (upstream added F16-src cases) so they fall back to CPU.
                if (op->src[0] && op->src[0]->type != GGML_TYPE_F32) return false;
                return true;
            }
            if (op->type == GGML_TYPE_Q8_0) {
                // Opt-in via DX12_SET_ROWS_Q8_0=1.  Produces coherent KV
                // cache writes for Phi-3 (head_dim=96), gemma Q4_K_M and
                // SmolVLM2 F16 (text + vision) with both --cache-type-k q8_0
                // and --cache-type-v q8_0.  The earlier SmolVLM2-F16
                // dual-cache break no longer reproduces (fixed by the
                // RDNA1/2 F16 WAR + precise-KV barrier work).  Remaining: 4
                // test-backend-ops fails at ~1.3e-7 NMSE on broadcast cases,
                // from round-half-even (HLSL) vs round-half-away (CPU ref)
                // tie noise -- negligible for real KV use.
                // Default OFF pending multi-vendor validation.
                static const bool enable_q8_0 = (getenv("DX12_SET_ROWS_Q8_0") != nullptr);
                if (!enable_q8_0) return false;
                // Requires native 16-bit shader ops (uint16_t Store) and
                // row column count to be a multiple of the Q8_0 block size
                // (mirrors the CPU `from_float` constraint).
                if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32) return false;
                if (op->src[0]->ne[0] % 32 != 0) return false;
                if (!dev) return false;
                auto * d = (dx12_device *)dev->context;
                if (!d || !d->fp16_supported) return false;
                return true;
            }
            // Phase F4: legacy quants (Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL) via
            // dedicated quantize-on-store shaders.  Opt-in via
            // DX12_SET_ROWS_LEGACY_QUANT=1 to mirror the Q8_0 gating —
            // these have not been validated on multi-KV-cache models.
            if (op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 ||
                op->type == GGML_TYPE_Q5_0 || op->type == GGML_TYPE_Q5_1 ||
                op->type == GGML_TYPE_IQ4_NL) {
                static const bool enable_lq = (getenv("DX12_SET_ROWS_LEGACY_QUANT") != nullptr);
                if (!enable_lq) return false;
                if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32) return false;
                if (op->src[0]->ne[0] % 32 != 0) return false;
                if (!dev) return false;
                auto * d = (dx12_device *)dev->context;
                if (!d || !d->fp16_supported) return false;
                return true;
            }
            return false;

        case GGML_OP_ARGMAX:
            // F32 matrix in, I32 1D out
            return op->type == GGML_TYPE_I32 &&
                   op->src[0] && op->src[0]->type == GGML_TYPE_F32;

        case GGML_OP_ARGSORT:
            // F32 in, I32 out. Small-N path (ncols<=1024) uses the LDS bitonic
            // shader; large-N path (1024 < ncols <= 2^20) uses the multi-pass
            // argsort_large shader with a global scratch buffer.  The large
            // path issues O(log^2 N) dispatches per op, so users running
            // real inference with vocab-sized TOP_K can opt back to CPU
            // fallback via DX12_DISABLE_LARGE_SORT=1.
            if (op->type != GGML_TYPE_I32) return false;
            if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) return false;
            if (op->src[0]->ne[0] <= 1024) return true;
            if (dx12_env_disable_large_sort()) return false;
            return op->src[0]->ne[0] <= (1 << 20);

        case GGML_OP_TOP_K:
            // F32 in, I32 out. Same small/large split as ARGSORT.
            if (op->type != GGML_TYPE_I32) return false;
            if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) return false;
            if (op->src[0]->ne[0] <= 1024) return true;
            if (dx12_env_disable_large_sort()) return false;
            return op->src[0]->ne[0] <= (1 << 20);

        case GGML_OP_ADD_ID:
            // src0: F32 [n_embd, n_used, n_tok]; src1: F32 [n_embd, n_experts];
            // src2 (ids): I32 [n_used, n_tok] (possibly view of [n_experts, n_tok]).
            return op->type == GGML_TYPE_F32 &&
                   op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[2] && op->src[2]->type == GGML_TYPE_I32 &&
                   op->src[0]->nb[0] == sizeof(float) &&
                   op->src[1]->nb[0] == sizeof(float);

        case GGML_OP_COUNT_EQUAL:
            // Two I32 tensors in, I64 scalar out
            return op->type == GGML_TYPE_I64 &&
                   op->src[0] && op->src[0]->type == GGML_TYPE_I32 &&
                   op->src[1] && op->src[1]->type == GGML_TYPE_I32;

        case GGML_OP_ACC:
        case GGML_OP_SET:
            // src0 and dst are same-shape and both contiguous; src1 is
            // the patch tensor. Strides for the dst-view come from
            // op_params[0..2] and must be a multiple of the dst element
            // size. SET is bit-preserving (asfloat/asuint roundtrip) so
            // also accepts I32; ACC needs float arithmetic, F32 only.
            if (op->op == GGML_OP_SET &&
                op->type == GGML_TYPE_I32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_I32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0])) {
                return true;
            }
            return op->type == GGML_TYPE_F32 &&
                   op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);

        case GGML_OP_CUMSUM:
            // F32 in/out, per-row inclusive prefix scan. src0 must have
            // contiguous innermost dim (nb00 == sizeof(float)).
            return op->type == GGML_TYPE_F32 &&
                   op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[0]->nb[0] == sizeof(float);

        case GGML_OP_SOLVE_TRI:
            // Lower-triangular solve A*X=B for F32. Square A (ne00==ne01),
            // capped at MAX_N=256 by the shader's groupshared budget.
            return op->type == GGML_TYPE_F32 &&
                   op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[0]->ne[0] == op->src[0]->ne[1] &&
                   op->src[0]->ne[0] <= 256;

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
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                case GGML_UNARY_OP_XIELU:
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
                    t != GGML_TYPE_IQ4_NL /* MM-on, GR-on */ &&
                    t != GGML_TYPE_IQ2_XXS && t != GGML_TYPE_IQ4_XS &&
                    t != GGML_TYPE_IQ3_XXS && t != GGML_TYPE_IQ2_XS &&
                    t != GGML_TYPE_IQ2_S   && t != GGML_TYPE_IQ3_S &&
                    t != GGML_TYPE_IQ1_S   && t != GGML_TYPE_IQ1_M) return false;
            }
            if (op->src[1] && op->src[1]->type != GGML_TYPE_F32 &&
                op->src[1]->type != GGML_TYPE_F16) return false;
            // Quantized matvec shaders walk K in fixed-size blocks (QK_K=256,
            // QK4_0=32, etc.) and cannot handle non-contiguous K. F16/F32
            // shaders (mul_mat_vec.hlsl) have a strided fallback that honors
            // nb00 / nb10, used by permuted V-cache matvec under -fa 0.
            if (op->src[0] && ggml_is_quantized(op->src[0]->type) &&
                op->src[0]->nb[0] != ggml_type_size(op->src[0]->type)) return false;
            if (op->src[1] && ggml_is_quantized(op->src[1]->type) &&
                op->src[1]->nb[0] != ggml_type_size(op->src[1]->type)) return false;
            return true;

        case GGML_OP_MUL_MAT_ID:
            // MoE: src0 = expert weights, src1 = input, src2 = expert ids (I32)
            if (op->type != GGML_TYPE_F32) return false;
            if (op->src[0]) {
                ggml_type t = op->src[0]->type;
                if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_BF16 &&
                    t != GGML_TYPE_Q4_K && t != GGML_TYPE_Q5_K && t != GGML_TYPE_Q6_K &&
                    t != GGML_TYPE_Q2_K && t != GGML_TYPE_Q3_K &&
                    t != GGML_TYPE_Q4_0 && t != GGML_TYPE_Q4_1 &&
                    t != GGML_TYPE_Q5_0 && t != GGML_TYPE_Q5_1 &&
                    t != GGML_TYPE_Q8_0 &&
                    t != GGML_TYPE_IQ4_NL && t != GGML_TYPE_IQ4_XS &&
                    t != GGML_TYPE_IQ2_XXS && t != GGML_TYPE_IQ2_XS &&
                    t != GGML_TYPE_IQ2_S && t != GGML_TYPE_IQ3_XXS &&
                    t != GGML_TYPE_IQ3_S &&
                    t != GGML_TYPE_IQ1_S && t != GGML_TYPE_IQ1_M) return false;
            }
            if (op->src[1] && op->src[1]->type != GGML_TYPE_F32) return false;
            if (op->src[2] && op->src[2]->type != GGML_TYPE_I32) return false;
            return true;

        case GGML_OP_GET_ROWS:
            if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_I32) return false;
            if (op->src[0]) {
                ggml_type t = op->src[0]->type;
                // I32 dst requires I32 src (bit-preserving via load_auto/
                // store_auto esize=4 asfloat/asuint roundtrip).
                if (op->type == GGML_TYPE_I32) {
                    if (t != GGML_TYPE_I32) return false;
                } else if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_BF16 &&
                    t != GGML_TYPE_Q4_K && t != GGML_TYPE_Q5_K && t != GGML_TYPE_Q6_K &&
                    t != GGML_TYPE_Q4_0 && t != GGML_TYPE_Q4_1 &&
                    t != GGML_TYPE_Q5_0 && t != GGML_TYPE_Q5_1 &&
                    t != GGML_TYPE_Q8_0 && t != GGML_TYPE_Q8_1 &&
                    t != GGML_TYPE_Q2_K && t != GGML_TYPE_Q3_K &&
                    t != GGML_TYPE_IQ4_NL) return false;
            }
            return true;

        case GGML_OP_FLASH_ATTN_EXT: {
            if (op->src[0]->type != GGML_TYPE_F32) return false;
            // K (src[1]) and V (src[2]) accepted types: F32, F16, BF16, plus the
            // 6 "legacy" quant types (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL). For quant
            // KV, the shader does per-element on-the-fly dequant; head_dim must
            // be a multiple of the block size (QK=32 for all 6 quant types) and
            // src[1] and src[2] must be the same type (matches every kv-cache).
            const ggml_type kt = op->src[1]->type;
            const ggml_type vt = op->src[2]->type;
            auto fa_type_ok = [](ggml_type t) {
                return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 ||
                       t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 ||
                       t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1 ||
                       t == GGML_TYPE_Q8_0 || t == GGML_TYPE_IQ4_NL;
            };
            auto fa_is_quant = [](ggml_type t) {
                return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 ||
                       t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1 ||
                       t == GGML_TYPE_Q8_0 || t == GGML_TYPE_IQ4_NL;
            };
            if (!fa_type_ok(kt)) return false;
            if (!fa_type_ok(vt)) return false;
            // Quant KV: require K type == V type and head dims that are a
            // multiple of the block size (QK=32 for all 6 legacy quant types).
            if (fa_is_quant(kt) || fa_is_quant(vt)) {
                if (kt != vt) return false;
                const int64_t QK = 32;
                if (op->src[1]->ne[0] % QK != 0) return false;
                if (op->src[2]->ne[0] % QK != 0) return false;
            }
            // The DX12 FA shaders implement the base
            // softmax(QK^T*scale + slope*mask + softcap(tanh) + sinks) @ V path.
            // Mixed hsk/hsv (e.g. DeepSeek-V2 MLA 576/512, V3 192/128) is
            // supported by passing D_v in the high bits of op_params[5]; the
            // shader uses ne00 (= hsk) for Q*K and D_v (= hsv) for V/output.
            // Cap at acc[4] * GROUP_SIZE = 1024 (both hsk and hsv).
            if (op->src[1]->ne[0] > 1024 || op->src[2]->ne[0] > 1024) return false;
            // sinks (src4): F32 vector of length n_heads (= Q's ne[2])
            if (op->src[4] != nullptr) {
                if (op->src[4]->type != GGML_TYPE_F32) return false;
                if (op->src[4]->ne[0] != op->src[0]->ne[2]) return false;
            }
            return true;
        }

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
            // Shader is compiled in four S_V variants (16/32/64/128). Other
            // sizes would OOB the state buffer (sized S_v*S_v*H*n_seqs).
            if (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) return false;
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
            // Mamba2 detection: A is a scalar per head, so total elements
            // equal n_head regardless of whether A is stored 1D [n_head] or
            // 2D [1, n_head]. ggml_n_dims previously rejected the 2D form,
            // missing all real Mamba-2 / Falcon-H1 cases. Mamba-1 has A
            // shaped [d_state, n_head] so total elements = d_state * n_head,
            // which won't match.
            const uint32_t n_head_check = (uint32_t)op->src[1]->ne[1];
            if (ggml_nelements(op->src[3]) != (int64_t)n_head_check) return false;
            const uint32_t d_state = (uint32_t)op->src[0]->ne[0];
            const uint32_t head_dim = (uint32_t)op->src[0]->ne[1];
            if (d_state != 128 && d_state != 256) return false;
            if (head_dim % 16 != 0) return false;
            // Wave-gate the Mamba-2 path: on small waves (Intel UHD wave=8,
            // Intel Arc wave=16) the per-lane state[C_FACTOR] array grows
            // large (8 floats at wave=16/d=128, 16 floats at wave=8/d=128,
            // 32 floats at wave=8/d=256) and the wave-reduction precision
            // is insufficient to meet test-backend-ops' nmse tolerance.
            // Substituting an explicit shared-mem tree reduction (matching
            // Vulkan's !USE_SUBGROUP_ADD path) produces the same magnitude
            // of error on Intel Arc, indicating a deeper precision issue
            // specific to small-wave hardware on this op. Real-world
            // deployments (NVIDIA wave=32, AMD wave=64) are unaffected and
            // Mamba/Falcon-H1 models work correctly on those backends. Fall
            // back to CPU on small-wave devices.
            {
                auto * d = (dx12_device *)dev->context;
                if (d && d->wave_size <= 16) return false;
            }
            return true;
        }

        case GGML_OP_RWKV_WKV6: {
            // RWKV WKV6 recurrent kernel.
            // src0=k, src1=v, src2=r, src3=tf, src4=td, src5=state.
            // Shader assumes head_size == BLOCK_SIZE == 64.
            if (op->type != GGML_TYPE_F32) return false;
            for (int i = 0; i < 6; i++) {
                if (!op->src[i] || op->src[i]->type != GGML_TYPE_F32) return false;
                if (!ggml_is_contiguous(op->src[i])) return false;
            }
            return op->src[0]->ne[0] == 64;
        }

        case GGML_OP_RWKV_WKV7: {
            // RWKV WKV7 recurrent kernel.
            // src0=r, src1=w, src2=k, src3=v, src4=a, src5=b, src6=state.
            // Shader assumes head_size == BLOCK_SIZE == 64.
            if (op->type != GGML_TYPE_F32) return false;
            for (int i = 0; i < 7; i++) {
                if (!op->src[i] || op->src[i]->type != GGML_TYPE_F32) return false;
                if (!ggml_is_contiguous(op->src[i])) return false;
            }
            return op->src[0]->ne[0] == 64;
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
        static_assert(sizeof(rope_tensor->op_params) <= sizeof(p.op_params),
                      "ggml op_params must fit in dx12_shader_params op_params");
        memset(p.op_params, 0, sizeof(p.op_params));
        memcpy(p.op_params, rope_tensor->op_params, sizeof(rope_tensor->op_params));
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

// Pack the shared ROPE metadata for a Q/K projection matvec post-op fusion
// (DX12_MMV_{Q,K}_ROPE_FUSION). Reuses the canonical dx12_rope_corr_dims and
// ggml-native op_params so the rotation matches rope.hlsl / rope_set_rows.hlsl
// exactly. Slots op1..op6, op8..op11 hold RoPE state (op0 stays 0 — the fusion
// is bias-free); op7 (mode selector), op12..op14 (Q store / K scatter strides)
// are filled by the caller. Positions ride src2, freq_factors ride src4.
static void dx12_pack_mmv_rope_op_params(
        const struct ggml_tensor * rope_tensor,
        dx12_shader_params & p) {
    GGML_ASSERT(rope_tensor && rope_tensor->op == GGML_OP_ROPE);
    const uint32_t * rope_up = (const uint32_t *)rope_tensor->op_params;
    float corr_low = 0.0f, corr_high = 0.0f;
    dx12_rope_corr_dims(rope_tensor, corr_low, corr_high);
    p.op_params[1]  = rope_up[1];                                     // n_dims
    p.op_params[2]  = rope_up[2];                                     // mode
    p.op_params[3]  = rope_up[5];                                     // freq_base  (float bits)
    p.op_params[4]  = rope_up[6];                                     // freq_scale (float bits)
    p.op_params[5]  = rope_up[7];                                     // ext_factor (float bits)
    p.op_params[6]  = rope_up[8];                                     // attn_factor(float bits)
    memcpy(&p.op_params[8], &corr_low,  sizeof(uint32_t));            // corr_low  (float bits)
    memcpy(&p.op_params[9], &corr_high, sizeof(uint32_t));            // corr_high (float bits)
    p.op_params[10] = (uint32_t)rope_tensor->ne[0];                   // head_dim (rope ne0)
    p.op_params[11] = (rope_tensor->src[2] != nullptr) ? 1u : 0u;    // has_ff
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
        const struct ggml_tensor * src2  = tensor->src[2];
        const struct ggml_tensor * mask  = tensor->src[3];
        const struct ggml_tensor * sinks = tensor->src[4];
        float scale         = 0.0f;
        float max_bias      = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&scale,         (const float *) tensor->op_params + 0, sizeof(float));
        memcpy(&max_bias,      (const float *) tensor->op_params + 1, sizeof(float));
        memcpy(&logit_softcap, (const float *) tensor->op_params + 2, sizeof(float));

        // CPU reference (ggml-cpu/ops.cpp): when softcap is active the score
        // becomes `softcap * tanh(QK * scale / softcap)`, which is implemented
        // by folding `scale /= softcap` host-side and applying
        // `softcap * tanh(scaled_score)` in the shader.
        if (logit_softcap != 0.0f) {
            scale /= logit_softcap;
        }

        p.op_params[0] = src2 ? (uint32_t)dx12_tensor_offset(src2) : 0; // src2_offset
        p.op_params[1] = src2 ? (uint32_t)src2->nb[0] : 0;             // src2_nb0
        p.op_params[2] = src2 ? (uint32_t)src2->nb[1] : 0;             // src2_nb1
        p.op_params[3] = src2 ? (uint32_t)src2->nb[2] : 0;             // src2_nb2
        p.op_params[4] = src2 ? (uint32_t)src2->nb[3] : 0;             // src2_nb3
        // op5: low 8 bits = src2 element size, high 24 bits = D_v (= src2->ne[0],
        // the V head_dim hsv). Packing D_v here is what unlocks mixed hsk/hsv
        // (e.g. DeepSeek-V2 MLA 576/512, V3-style 192/128) without taking
        // a free op slot — every op[0..15] is already in use for FA params.
        {
            const uint32_t src2_es = src2 ? ((src2->type == GGML_TYPE_BF16) ? 3u : (uint32_t)ggml_type_size(src2->type)) : 4u;
            const uint32_t src2_ne0 = src2 ? (uint32_t)src2->ne[0] : 0u;
            GGML_ASSERT(src2_es <= 0xFFu);
            GGML_ASSERT(src2_ne0 <= 0xFFFFFFu);
            p.op_params[5] = (src2_es & 0xFFu) | (src2_ne0 << 8);
        }
        memcpy(&p.op_params[6], &scale, sizeof(float));                 // scale (post-softcap fold)
        // op7 reused for logit_softcap (float, 0 = softcap off). The shader
        // reads n_kv_heads from ne12 directly (it's the same value as src1->ne[2]).
        memcpy(&p.op_params[7], &logit_softcap, sizeof(float));

        // Mask (src3) parameters
        const uint32_t mask_esize = mask ? (mask->type == GGML_TYPE_BF16 ? 3u : (uint32_t)ggml_type_size(mask->type)) : 0u;
        p.op_params[8]  = mask ? (1u | ((uint32_t)mask->nb[0] << 8) | (mask_esize << 16)) : 0u; // mask info
        if (sinks) {
            p.op_params[8] |= (1u << 24);  // has_sinks bit
        }
        p.op_params[9]  = mask ? (uint32_t)dx12_tensor_offset(mask) : 0; // mask_offset
        p.op_params[10] = mask ? (uint32_t)mask->nb[1] : 0;              // mask_nb1
        p.op_params[11] = mask ? (uint32_t)mask->nb[2] : 0;              // mask_nb2
        p.op_params[12] = mask ? (uint32_t)mask->nb[3] : 0;              // mask_nb3
        // Pack mask_ne2 (low 16) | mask_ne3 (high 16); both are head/batch
        // broadcast counts and easily fit in 16 bits. Frees op14 for max_bias.
        {
            const uint32_t mask_ne2 = mask ? (uint32_t)mask->ne[2] : 1u;
            const uint32_t mask_ne3 = mask ? (uint32_t)mask->ne[3] : 1u;
            GGML_ASSERT(mask_ne2 < 65536u && mask_ne3 < 65536u);
            p.op_params[13] = (mask_ne2 & 0xFFFFu) | ((mask_ne3 & 0xFFFFu) << 16);
        }
        // ALiBi: pass max_bias (float). The shader derives m0, m1, n_head_log2
        // from max_bias and n_head (= ne02) — matches the ggml-cpu reference.
        memcpy(&p.op_params[14], &max_bias, sizeof(float));
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
        // Same-type quantized (or I16) CPY/DUP routes to cpy_quant_block.hlsl,
        // which reads blocks_per_row from op_params[2]. For quantized types
        // ne[0] is the logical element count and divides evenly by the
        // type's block size (supports_op gate + ggml's own constraints).
        // For I16 the block size is 1, so blocks_per_row == ne[0].
        if (tensor->src[0] && tensor->src[0]->type == tensor->type &&
            (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_I16)) {
            const int64_t blck = (int64_t)ggml_blck_size(tensor->type);
            p.op_params[2] = (uint32_t)((tensor->ne[0] + blck - 1) / blck);
        }
    } else if (tensor->op == GGML_OP_GATED_DELTA_NET) {
        // GDN op_params layout (matches gated_delta_net.hlsl):
        //   [0]=H [1]=n_tokens [2]=K (snapshot slot count) [3]=s_off
        //   [4..6]=sq1,sq2,sq3 [7..9]=sv1,sv2,sv3 [10..12]=sb1,sb2,sb3
        //   [13]=neq1 [14]=rq3 [15]=scale (float bits)
        // n_seqs is not passed directly: derive in shader as (s_off / (S_V * n_tokens)) / H,
        // freeing slot 2 for K. K==1 keeps backward-compatible single-slot semantics.
        const struct ggml_tensor * src_q     = tensor->src[0];
        const struct ggml_tensor * src_v     = tensor->src[2];
        const struct ggml_tensor * src_beta  = tensor->src[4];
        const uint32_t S_v      = (uint32_t)src_v->ne[0];
        const uint32_t H        = (uint32_t)src_v->ne[1];
        const uint32_t n_tokens = (uint32_t)src_v->ne[2];
        const uint32_t n_seqs   = (uint32_t)src_v->ne[3];
        // K (snapshot slot count) is stored as op_params[0] by upstream; the state
        // input is now s0-only ([S_v, S_v, H, n_seqs]) and no longer carries K in ne[1].
        const int32_t  K_param  = ggml_get_op_params_i32(tensor, 0);
        const uint32_t K        = K_param > 0 ? (uint32_t)K_param : 1u;
        const uint32_t s_off    = S_v * H * n_tokens * n_seqs;
        p.op_params[0]  = H;
        p.op_params[1]  = n_tokens;
        p.op_params[2]  = K;
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
    } else if (tensor->op == GGML_OP_RWKV_WKV6 ||
               tensor->op == GGML_OP_RWKV_WKV7) {
        // RWKV WKV6/WKV7 op_params layout:
        //   [0] = B (n_seqs)
        // All other dimensions are derived from cbuffer ne00/ne01/ne02
        // (S = head_size, H = head_count, T = n_tokens). The shader assumes
        // contiguous F32 srcs throughout.
        const struct ggml_tensor * state =
            (tensor->op == GGML_OP_RWKV_WKV6) ? tensor->src[5] : tensor->src[6];
        memset(p.op_params, 0, sizeof(p.op_params));
        p.op_params[0] = (uint32_t)state->ne[1];
    } else if (tensor->op == GGML_OP_ADD_ID) {
        // ADD_ID op_params layout (matches add_id.hlsl):
        //   [0] = src2 nb0 (ids stride along dim 0, bytes — usually 4)
        //   [1] = src2 nb1 (ids stride along dim 1, bytes)
        // src2's per-tensor byte offset is baked into the bound GPU VA via the
        // gdn_or_ssm-style binding path, so the shader reads from byte 0.
        const struct ggml_tensor * ids = tensor->src[2];
        memset(p.op_params, 0, sizeof(p.op_params));
        if (ids) {
            p.op_params[0] = (uint32_t)ids->nb[0];
            p.op_params[1] = (uint32_t)ids->nb[1];
        }
    } else if (tensor->op == GGML_OP_ARGSORT) {
        // ARGSORT op_params layout.
        // Small-N path (matches argsort.hlsl): [0]=order.
        // Large-N path (matches argsort_large.hlsl): [0]=order, [1]=ncols,
        // [2]=ncols_padded, [3]=kind (init=0 initially; swap/writeout phases
        // patch [3..6] live during dispatch), [4]=K (unused for ARGSORT),
        // [5]=k, [6]=j.  The large-path values are still valid for the small
        // shader (which only reads op0).
        memset(p.op_params, 0, sizeof(p.op_params));
        p.op_params[0] = (uint32_t)ggml_get_op_params_i32(tensor, 0);
        const uint32_t ncols = (uint32_t)tensor->src[0]->ne[0];
        if (ncols > 1024) {
            uint32_t ncols_padded = 1;
            while (ncols_padded < ncols) ncols_padded <<= 1;
            if (ncols_padded < 256u) ncols_padded = 256u;
            p.op_params[1] = ncols;
            p.op_params[2] = ncols_padded;
            p.op_params[3] = 0u; // kind = INIT
        }
    } else if (tensor->op == GGML_OP_TOP_K) {
        // TOP_K op_params layout (large-N path only — small path reads none).
        //   [1]=ncols, [2]=ncols_padded, [3]=kind, [4]=K, [5]=k, [6]=j.
        memset(p.op_params, 0, sizeof(p.op_params));
        const uint32_t ncols = (uint32_t)tensor->src[0]->ne[0];
        const uint32_t K     = (uint32_t)tensor->ne[0];
        if (ncols > 1024) {
            uint32_t ncols_padded = 1;
            while (ncols_padded < ncols) ncols_padded <<= 1;
            if (ncols_padded < 256u) ncols_padded = 256u;
            p.op_params[1] = ncols;
            p.op_params[2] = ncols_padded;
            p.op_params[3] = 0u; // kind = INIT
            p.op_params[4] = K;
        }
    } else {
        static_assert(sizeof(tensor->op_params) <= sizeof(p.op_params), "op_params size mismatch");
        memset(p.op_params, 0, sizeof(p.op_params));
        memcpy(p.op_params, tensor->op_params, sizeof(tensor->op_params));
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

// Overflow-safe ceil division (Vulkan CEIL_DIV fix, commit 86961efd5).
// `(m + n - 1) / n` overflows when m is within (n-1) of the type max — e.g.
// IM2COL_3D perf tensors reach ~4B elements, so `total_elements + 255` wraps
// a uint32.  Compute `(m / n) + (m % n != 0)` instead, which never overflows.
static inline uint32_t dx12_ceil_div(uint32_t m, uint32_t n) {
    return (m / n) + ((m % n) != 0u);
}

// Estimate the FLOP count of a single graph node for the adaptive submit
// heuristic.  Ported from Vulkan's ggml_vk_get_node_flops
// (ggml-vulkan.cpp:1910).  Unlike the older DX12 heuristic (which summed
// only MUL_MAT/MUL_MAT_ID weight *bytes*) this covers the compute-heavy op
// families and reflects actual arithmetic work, so submission batching tracks
// GPU cost instead of memory traffic.  Unhandled ops return 0 (they are cheap
// relative to matmul / attention and don't need to gate submission size).
static uint64_t dx12_get_node_flops(const ggml_tensor * node) {
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        const uint64_t m     = node->ne[0];
        const uint64_t n     = node->ne[1];
        const uint64_t k     = node->src[1]->ne[0];
        const uint64_t batch = node->ne[2] * node->ne[3];
        return m * n * (k + (k - 1)) * batch;
    }
    if (node->op == GGML_OP_CONV_2D || node->op == GGML_OP_CONV_TRANSPOSE_2D) {
        const ggml_tensor * knl = node->src[0];
        const uint64_t Cout   = node->ne[2];
        const uint64_t size_K = node->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        const uint64_t size_N = node->ne[3] * node->ne[0] * node->ne[1];
        return Cout * size_N * (size_K + (size_K - 1));
    }
    if (node->op == GGML_OP_CONV_3D) {
        const ggml_tensor * knl = node->src[0];
        const uint64_t OC     = ggml_get_op_params_i32(node, 11);
        const uint64_t IC     = ggml_get_op_params_i32(node, 9);
        if (OC == 0) return 0;
        const uint64_t size_K = IC * knl->ne[0] * knl->ne[1] * knl->ne[2];
        const uint64_t size_N = node->ne[3] / OC * node->ne[0] * node->ne[1] * node->ne[2];
        return OC * size_N * (size_K + (size_K - 1));
    }
    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        const ggml_tensor * q = node->src[0];
        const ggml_tensor * k = node->src[1];
        const ggml_tensor * v = node->src[2];
        return 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Whole-graph command-list replay helpers (DX12_COMMAND_REPLAY)
// ---------------------------------------------------------------------------

// Recompute a flash-attention dispatch's n_splits from the captured formula
// inputs and the current N_kv.  Mirrors the split-KV heuristic in the recording
// loop; the inputs (total_groups_no_split, target_groups, min_kv_per_split) are
// identity-stable so only N_kv varies per token.
static uint32_t dx12_replay_fa_n_splits(const dx12_cmd_replay::fa_rec & f, uint32_t n_kv) {
    uint32_t ns = 1;
    if (f.total_groups_no_split < f.target_groups && n_kv > f.min_kv_per_split) {
        ns = (f.target_groups + f.total_groups_no_split - 1) / f.total_groups_no_split;
        ns = std::min(ns, (n_kv + f.min_kv_per_split - 1) / f.min_kv_per_split);
        ns = std::min(ns, (uint32_t)32);
    }
    return ns;
}

// FNV-1a hash of the resource base GPU VAs the captured list bakes into its
// root descriptors, plus the on-demand device scratch buffers.  A change here
// (buffer / scratch reallocation) means the list is stale and must be
// re-recorded.  Only real-dispatch nodes are hashed: the per-node identity
// match already proves the graph is unchanged (so the graph-allocator layout,
// KV cache and weights keep the same bases), leaving the growable device
// scratch buffers as the primary reallocation risk.
static uint64_t dx12_replay_signature(dx12_backend_context * bctx, const ggml_cgraph * cgraph) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
    auto mix_res = [&](const ggml_tensor * t) {
        ID3D12Resource * r = dx12_get_resource(t);
        mix(r ? (uint64_t) r->GetGPUVirtualAddress() : 0);
    };
    const dx12_replay_cache & rc = bctx->replay_cache;
    const bool have_dec = ((int) rc.decisions.size() == cgraph->n_nodes);
    // Projection post-op fusion (Q RoPE, K RoPE+scatter, V scatter) rewrites the
    // ROPE/SET_ROWS it absorbs into DX12_DEC_SKIP nodes, so the src0/src1/dst
    // hashing below stops covering the KV-cache / positions / row-index /
    // freq-factor buffers the baked list binds as root SRVs.  Re-add them via the
    // decision cache's relative indices so a reallocation of any absorbed buffer
    // invalidates the capture (matching the coverage the non-fused graph gets
    // from its standalone ROPE/SET_ROWS COMPUTE nodes).
    auto mix_absorbed = [&](int i, const dx12_node_decision & d) {
        auto node_at = [&](int rel) -> const ggml_tensor * {
            const int j = i + rel;
            return (rel != 0 && j >= 0 && j < cgraph->n_nodes) ? cgraph->nodes[j] : nullptr;
        };
        switch (d.fusion_kind) {
            case DX12_FUSE_MMV_SET_ROWS:
                if (const ggml_tensor * sr = node_at(d.mmv_set_rows_rel)) {
                    mix_res(sr);          // KV cache (scatter dst)
                    mix_res(sr->src[1]);  // row indices
                }
                break;
            case DX12_FUSE_MMV_Q_ROPE:
                if (const ggml_tensor * rp = node_at(d.mmv_rope_rel)) {
                    mix_res(rp);          // RoPE output (dst)
                    mix_res(rp->src[1]);  // positions
                    mix_res(rp->src[2]);  // freq_factors (optional)
                }
                break;
            case DX12_FUSE_MMV_K_ROPE_SET_ROWS: {
                const ggml_tensor * rp = node_at(d.mmv_rope_rel);
                const ggml_tensor * sr = node_at(d.mmv_rope_set_rows_rel);
                if (rp) { mix_res(rp->src[1]); mix_res(rp->src[2]); }  // positions, freq_factors
                if (sr) { mix_res(sr); mix_res(sr->src[1]); }         // KV cache, row indices
                break;
            }
            case DX12_FUSE_MMV_QKV_SHARED: {
                // The combined dispatch binds the Q ROPE output, both extra
                // weights, the shared KV cache and the K/V indices/positions as
                // root descriptors the src0/src1/dst hashing above does not cover;
                // fold them in so a reallocation of any invalidates the capture.
                const ggml_tensor * q_rope = node_at(d.qkv_q_rope_rel);
                const ggml_tensor * v_mm   = node_at(d.qkv_v_matvec_rel);
                const ggml_tensor * v_sr   = node_at(d.qkv_v_set_rows_rel);
                const ggml_tensor * k_mm   = node_at(d.qkv_k_matvec_rel);
                const ggml_tensor * k_sr   = node_at(d.qkv_k_set_rows_rel);
                if (q_rope) { mix_res(q_rope); mix_res(q_rope->src[1]); mix_res(q_rope->src[2]); }
                if (v_mm)   { mix_res(v_mm->src[0]); }              // Wv
                if (v_sr)   { mix_res(v_sr); mix_res(v_sr->src[1]); }  // V cache, V indices
                if (k_mm)   { mix_res(k_mm->src[0]); }              // Wk
                if (k_sr)   { mix_res(k_sr); mix_res(k_sr->src[1]); }  // K cache, K indices
                break;
            }
            case DX12_FUSE_QK_ROPE_SCALE_SET_ROWS: {
                const ggml_tensor * scale = node_at(d.qk_scale_rel);
                const ggml_tensor * k_rp  = node_at(d.qk_k_rope_rel);
                const ggml_tensor * k_sr  = node_at(d.qk_k_set_rows_rel);
                if (scale) mix_res(scale);
                if (k_rp) {
                    mix_res(k_rp->src[0]);
                    mix_res(k_rp->src[1]);
                    mix_res(k_rp->src[2]);
                }
                if (k_sr) {
                    mix_res(k_sr);
                    mix_res(k_sr->src[1]);
                }
                break;
            }
            default:
                break;
        }
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (have_dec && rc.decisions[i].kind != DX12_DEC_COMPUTE) continue;
        const ggml_tensor * n = cgraph->nodes[i];
        if (!n) continue;
        mix_res(n->src[0]);
        mix_res(n->src[1]);
        mix_res(n);
        if (have_dec) mix_absorbed(i, rc.decisions[i]);
    }
    mix(bctx->q8_1_scratch        ? (uint64_t) bctx->q8_1_scratch->GetGPUVirtualAddress()        : 0);
    mix(bctx->dev->splitkv_temp   ? (uint64_t) bctx->dev->splitkv_temp->GetGPUVirtualAddress()   : 0);
    mix(bctx->dev->argsort_scratch? (uint64_t) bctx->dev->argsort_scratch->GetGPUVirtualAddress(): 0);
    return h;
}

// Lazily create the dedicated persistent CBV region (UPLOAD heap, mapped once).
// Sized above the ring's per-slot param budget so one graph's blocks never
// overflow (the recording path already guarantees a token fits in PARAM_SLOT_SIZE).
static bool dx12_replay_ensure_cbv(dx12_backend_context * bctx) {
    dx12_cmd_replay & R = bctx->replay;
    if (R.cbv) return true;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = (UINT64) dx12_backend_context::PARAM_SLOT_SIZE * 2;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    HRESULT hr = bctx->dev->device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr, IID_PPV_ARGS(&R.cbv));
    if (FAILED(hr)) return false;
    D3D12_RANGE rr = { 0, 0 };
    hr = R.cbv->Map(0, &rr, (void **) &R.cbv_mapped);
    if (FAILED(hr)) { R.cbv.Reset(); return false; }
    R.cbv_base = R.cbv->GetGPUVirtualAddress();
    R.cbv_size = (size_t) rd.Width;
    return true;
}

// Submit the already-closed dedicated list and signal the ordinary backend
// fence (mirrors close_and_execute for the ring list).
static void dx12_replay_execute(dx12_backend_context * bctx) {
    // Flush deferred input uploads before the baked list reads them. set_tensor
    // stages decode inputs into the upload ring; close_and_execute drains it on
    // the record path, but replay bypasses that, so drain it here too.
    bctx->dev->flush_uploads();
    ID3D12CommandList * lists[] = { bctx->replay.list.Get() };
    bctx->dev->compute_queue->ExecuteCommandLists(1, lists);
    bctx->fence_value++;
    HRESULT hr = bctx->dev->compute_queue->Signal(bctx->fence.Get(), bctx->fence_value);
    DX12_CHECK(hr, "Signal fence (cmd-replay)");
    bctx->replay.last_fence = bctx->fence_value;
}

static void dx12_replay_stats_dump(dx12_backend_context * bctx) {
    dx12_cmd_replay & R = bctx->replay;
    if (!R.stats) return;
    const uint64_t total = R.replays + R.records;
    if (total == 0 || (total % 128) != 0) return;
    const double pct = 100.0 * (double) R.replays / (double) total;
    fprintf(stderr,
            "[DX12_CMD_REPLAY] replay=%llu record=%llu capture=%llu invalidate=%llu patch=%llu (replay %.1f%%)\n",
            (unsigned long long) R.replays, (unsigned long long) R.records,
            (unsigned long long) R.captures, (unsigned long long) R.invalidations,
            (unsigned long long) R.patches, pct);
    fflush(stderr);
}

// Redirect recording into the dedicated allocator+list and dedicated CBV.  The
// allocator/list are only reset after their last submit has completed, so the
// closed list stays valid across replays.
static bool dx12_replay_begin_capture(dx12_backend_context * bctx) {
    dx12_cmd_replay & R = bctx->replay;
    if (!dx12_replay_ensure_cbv(bctx)) return false;
    if (R.last_fence) bctx->wait_for_fence(R.last_fence);
    HRESULT hr;
    if (!R.alloc) {
        hr = bctx->dev->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&R.alloc));
        if (FAILED(hr)) return false;
    } else {
        hr = R.alloc->Reset();
        if (FAILED(hr)) return false;
    }
    if (!R.list) {
        hr = bctx->dev->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, R.alloc.Get(), nullptr, IID_PPV_ARGS(&R.list));
        if (FAILED(hr)) return false;
    } else {
        hr = R.list->Reset(R.alloc.Get(), nullptr);
        if (FAILED(hr)) return false;
    }
    R.saved_cmd_list = bctx->cmd_list;
    bctx->cmd_list   = R.list;
    R.cbv_cursor  = 0;
    R.capture_ok  = true;
    R.capturing   = true;
    R.fa.clear();
    bctx->reset_binding_cache();
    return true;
}

// Close the dedicated list, restore the ring list, execute the recording to
// produce this token's output, and (if the capture is reusable) store the
// validation signature so subsequent tokens can replay it.
static void dx12_replay_finalize_capture(dx12_backend_context * bctx, const ggml_cgraph * cgraph) {
    dx12_cmd_replay & R = bctx->replay;
    HRESULT hr = bctx->cmd_list->Close();  // == R.list
    const bool close_ok = SUCCEEDED(hr);
    bctx->cmd_list = R.saved_cmd_list;
    R.saved_cmd_list.Reset();
    R.capturing = false;

    if (!close_ok || !R.capture_ok) {
        // Unreachable for real decode graphs (CBV is sized above the per-token
        // budget and Close only fails on recording errors that would also break
        // the ring path).  Bail out safely: drop the capture and re-record via
        // the ring on the next call.
        R.captured = false;
        R.list.Reset();
        GGML_ABORT("DX12 command-replay capture failed (close_ok=%d capture_ok=%d)",
                   (int) close_ok, (int) R.capture_ok);
    }

    dx12_replay_execute(bctx);

    const bool realloc_during_capture =
        !bctx->q8_1_scratch_retired.empty() || !bctx->dev->argsort_scratch_retired.empty();
    if (realloc_during_capture) {
        // A scratch buffer grew while recording: the baked list is valid for
        // this submit but its base VAs may be freed once the retired buffers
        // are drained, so do not reuse it.  Drain like the normal path.
        bctx->wait_for_gpu();
        bctx->q8_1_scratch_retired.clear();
        bctx->dev->argsort_scratch_retired.clear();
        R.captured = false;
        return;
    }

    R.signature = dx12_replay_signature(bctx, cgraph);
    R.n_nodes   = cgraph->n_nodes;
    R.captured  = true;
    R.replays_since_capture = 0;
    R.captures++;
}

// Validate an existing capture against the current graph and, if valid, refresh
// the dynamic FA params and re-execute the closed list.  Returns false (leaving
// the capture untouched) when anything the baked list depends on has changed.
static bool dx12_replay_try(dx12_backend_context * bctx, const ggml_cgraph * cgraph) {
    dx12_cmd_replay & R = bctx->replay;
    if (!R.captured || R.n_nodes != cgraph->n_nodes) return false;
    if (dx12_replay_signature(bctx, cgraph) != R.signature) return false;

    for (dx12_cmd_replay::fa_rec & f : R.fa) {
        if (f.node_index < 0 || f.node_index >= cgraph->n_nodes) return false;
        ggml_tensor * n = cgraph->nodes[f.node_index];
        if (!n || n->op != GGML_OP_FLASH_ATTN_EXT || !n->src[1]) return false;
        const uint32_t cur_nkv = (uint32_t) n->src[1]->ne[1];
        const uint32_t ns = dx12_replay_fa_n_splits(f, cur_nkv);
        if (ns != f.n_splits) return false;  // baked group count would be wrong
        if (cur_nkv != f.n_kv) {
            // N_kv changed but the split count (and thus the baked dispatch
            // dimensions) did not: refresh this dispatch's fixed CBV slot.  Wait
            // for the previous submit so the GPU is no longer reading the slot.
            bctx->wait_for_fence(R.last_fence);
            dx12_shader_params p;
            dx12_fill_params(n, p);
            p.op_params[15] = (ns & 0xFFFFu) | ((f.gqa_ratio & 0xFFFFu) << 16);
            if (R.cbv_mapped && f.cbv_offset + sizeof(p) <= R.cbv_size) {
                memcpy(R.cbv_mapped + f.cbv_offset, &p, sizeof(p));
            }
            f.n_kv = cur_nkv;
            R.patches++;
        }
    }

    dx12_replay_execute(bctx);
    R.replays++;
    R.replays_since_capture++;
    return true;
}

// V-cache SET_ROWS matvec fusion (DX12_MMV_SET_ROWS_FUSION) alias guard.
// The matvec at mm_idx would write its result straight into the KV cache the
// SET_ROWS at sr_idx scatters into, so the intermediate Vcur buffer is never
// materialized. Verify mm_idx and every RESHAPE/VIEW between it and the
// SET_ROWS are consumed *only* by that chain — if anything else reads the Vcur
// buffer (or a view of it) it would see stale data, so the fusion must not
// fire. Returns true only when the chain is exclusive.
static bool dx12_mmv_scatter_chain_exclusive(const ggml_cgraph * cgraph,
                                             int mm_idx, int sr_idx) {
    const ggml_tensor * mm = cgraph->nodes[mm_idx];
    const ggml_tensor * sr = cgraph->nodes[sr_idx];
    const ggml_tensor * chain[8];
    int n_chain = 0;
    const ggml_tensor * s = sr->src[0];
    while (s && s != mm && n_chain < 7) {
        if (!(s->op == GGML_OP_RESHAPE || s->op == GGML_OP_VIEW)) {
            return false;
        }
        chain[n_chain++] = s;
        s = s->src[0];
    }
    if (s != mm) {
        return false;
    }
    chain[n_chain++] = mm;
    for (int k = 0; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * x = cgraph->nodes[k];
        if (!x || x == sr) {
            continue;  // SET_ROWS legitimately consumes its own src[0]
        }
        bool x_in_chain = false;
        for (int c = 0; c < n_chain; ++c) {
            if (chain[c] == x) { x_in_chain = true; break; }
        }
        if (x_in_chain) {
            continue;  // internal chain links (view->src[0]) are expected
        }
        for (int j = 0; j < GGML_MAX_SRC && x->src[j]; ++j) {
            for (int c = 0; c < n_chain; ++c) {
                if (x->src[j] == chain[c]) {
                    return false;  // external reader of the Vcur buffer
                }
            }
        }
    }
    return true;
}

// Q/K projection ROPE matvec fusion (DX12_MMV_{Q,K}_ROPE_FUSION) alias guard.
// The matvec at mm_idx would apply RoPE and write straight into the endpoint's
// output (the ROPE result for Q, the KV cache for K), so the intermediate
// Qcur/Kcur buffer and the ROPE/VIEW nodes between mm_idx and the endpoint are
// never materialized. Verify every such link is consumed *only* along this
// chain — anything else reading them would see stale (unrotated) data, so the
// fusion must not fire. The chain may contain RESHAPE/VIEW (both Q and K) and,
// for K, the ROPE and its VIEW. The endpoint (ep_idx = the ROPE for Q, the
// SET_ROWS for K) legitimately has external consumers and is skipped here.
static bool dx12_mmv_rope_chain_exclusive(const ggml_cgraph * cgraph,
                                          int mm_idx, int ep_idx) {
    const ggml_tensor * mm = cgraph->nodes[mm_idx];
    const ggml_tensor * ep = cgraph->nodes[ep_idx];
    const ggml_tensor * chain[8];
    int n_chain = 0;
    const ggml_tensor * s = ep->src[0];
    while (s && s != mm && n_chain < 7) {
        if (!(s->op == GGML_OP_RESHAPE || s->op == GGML_OP_VIEW || s->op == GGML_OP_ROPE)) {
            return false;
        }
        chain[n_chain++] = s;
        s = s->src[0];
    }
    if (s != mm) {
        return false;
    }
    chain[n_chain++] = mm;
    for (int k = 0; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * x = cgraph->nodes[k];
        if (!x || x == ep) {
            continue;  // endpoint legitimately consumes its own src[0]
        }
        bool x_in_chain = false;
        for (int c = 0; c < n_chain; ++c) {
            if (chain[c] == x) { x_in_chain = true; break; }
        }
        if (x_in_chain) {
            continue;  // internal chain links (view->src[0]) are expected
        }
        for (int j = 0; j < GGML_MAX_SRC && x->src[j]; ++j) {
            for (int c = 0; c < n_chain; ++c) {
                if (x->src[j] == chain[c]) {
                    return false;  // external reader of the pre-RoPE buffer
                }
            }
        }
    }
    return true;
}

static bool dx12_tensor_consumed_only_by(const ggml_cgraph * cgraph,
                                         const ggml_tensor * tensor,
                                         const ggml_tensor * consumer) {
    bool found = false;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
            if (node->src[s] != tensor) {
                continue;
            }
            if (node != consumer || found) {
                return false;
            }
            found = true;
        }
    }
    return found;
}

static ggml_status dx12_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * bctx = (dx12_backend_context *)backend->context;
    const bool phase_profile = DX12_GETENV("DX12_PHASE_PROFILE") != nullptr;
    if (phase_profile) {
        bctx->phase_graph_start_us  = dx12_qpc_us();
        bctx->phase_pending         = true;
        bctx->phase_record_start_us = 0;
        bctx->phase_graph_return_us = 0;
        bctx->phase_submit_us       = 0;
        bctx->phase_submit_record_us = 0;
        bctx->phase_alloc_wait_us    = 0;
        bctx->phase_alloc_wait_post_us = 0;
        bctx->phase_first_submit_us  = 0;
        bctx->phase_get_tensor_us    = 0;
        bctx->phase_decision_us     = 0;
        bctx->phase_params_us       = 0;
        bctx->phase_setup_us        = 0;
        bctx->phase_barrier_us      = 0;
        bctx->phase_dispatch_us     = 0;
    }

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

    // DX12_QUANT_STATS: per-graph tally of Q8_1 activation-prep dispatches.
    static const bool quant_stats = (getenv("DX12_QUANT_STATS") != nullptr);
    bctx->dbg_q8_quant_dispatched = 0;
    bctx->dbg_q8_quant_reused     = 0;
    bctx->dbg_q8_norm_prepop      = 0;

    static const bool barrier_stats = (getenv("DX12_BARRIER_STATS") != nullptr);
    bctx->dbg_barrier_scoped = 0;
    bctx->dbg_barrier_global = 0;

    // DX12_FUSION_AUDIT: one-shot histogram of adjacent op pairs in the first
    // decode graph, printed sorted by frequency.  Decode graphs repeat
    // identically per token, so a single dump captures the full per-token
    // adjacency.  Cross-referenced against the fusions we already apply, it
    // shows which op sequences still run as separate dispatches -- i.e. the
    // real fusion gaps for this model, chosen from data rather than guesswork.
    // Read-only over cgraph; fully isolated from the dispatch path.
    static const bool fusion_audit = (getenv("DX12_FUSION_AUDIT") != nullptr);
    if (fusion_audit) {
        static bool fusion_audited = false;
        if (!fusion_audited && cgraph->n_nodes > 1 &&
            cgraph->nodes[cgraph->n_nodes - 1] &&
            cgraph->nodes[cgraph->n_nodes - 1]->ne[1] == 1) {
            fusion_audited = true;
            std::map<std::string, int> pairs;
            for (int i = 0; i + 1 < cgraph->n_nodes; i++) {
                if (!cgraph->nodes[i] || !cgraph->nodes[i + 1]) continue;
                std::string k = std::string(ggml_op_name(cgraph->nodes[i]->op)) + " -> " +
                                ggml_op_name(cgraph->nodes[i + 1]->op);
                pairs[k]++;
            }
            std::vector<std::pair<std::string, int>> sorted(pairs.begin(), pairs.end());
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto & a, const auto & b) { return a.second > b.second; });
            fprintf(stderr, "[DX12_FUSION_AUDIT] decode graph nodes=%d -- adjacent op pairs by frequency:\n",
                    cgraph->n_nodes);
            for (const auto & p : sorted) {
                fprintf(stderr, "[DX12_FUSION_AUDIT]   %5d  %s\n", p.second, p.first.c_str());
            }
            fflush(stderr);
        }
    }

    // Run auto-tuning on first graph compute
    if (!bctx->dev->tuning_done) {
        bctx->dev->run_autotune();
    }

    // NOTE: the command list is opened below, after the command-list replay
    // decision, so a replay token never opens (or records into) the ring list.

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
    if (phase_profile) {
        bctx->phase_is_prompt = is_prompt;
    }
    // Profile the 3rd-5th actual generation graphs (skip warmup/reserve).
    // Set DX12_PROFILE_PROMPT=1 to also profile prompt graphs (e.g. CLIP
    // encode is the largest prompt graph in vision models).
    static bool profile_prompt = (getenv("DX12_PROFILE_PROMPT") != nullptr);
    // Tuner GPU-timing hook: DX12_TUNE_PROFILE_JSON=<path> forces per-dispatch
    // GPU timestamp profiling on every graph_compute and appends one JSON-lines
    // record per graph to the path. Used by llama-mmv-tune to read kernel-side
    // dispatch time (free of CPU-sync cost). When set, normal stderr profile
    // table is suppressed; the JSON dump is the only output.
    static const char * tune_profile_json = getenv("DX12_TUNE_PROFILE_JSON");
    const bool tune_profile_active = (tune_profile_json != nullptr);
    bool do_profile = profiling && ((!is_prompt && gen_graph >= 3 && gen_graph <= 5) ||
                                    (is_prompt && profile_prompt));
    if (tune_profile_active) do_profile = true;
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
    uint64_t batch_flops = 0;   // accumulated estimated FLOPs since last submit
    uint64_t total_flops = 0;   // grand total this graph (saved for next call)
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
    // FLOPs-per-submit threshold (Vulkan ggml-vulkan.cpp:16308-16322).
    // Value: previous graph's total estimated FLOPs / 40, clamped to flops_cap.
    // last_total_flops is UINT64_MAX on the very first call so the first graph
    // uses the full flops_cap; the node-count stream_threshold still drives
    // submission on small graphs where /40 never reaches flops_per_submit.
    // Disable via DX12_STREAM_FLOPS=0.
    static const bool stream_flops_disabled = []() {
        const char * s = getenv("DX12_STREAM_FLOPS");
        return s && atoi(s) == 0;
    }();
    // flops_cap: the largest amount of work we let accumulate into one
    // submission.  On discrete GPUs 200 GFLOP keeps ~one big matmul group per
    // submit while overlapping CPU recording with GPU execution.  Weak GPUs
    // (integrated) can hit the OS GPU-timeout (TDR / DEVICE_REMOVED) on large
    // single submissions, so we submit more often there.
    //
    // Vulkan scales this by AMD compute-unit count (ggml-vulkan.cpp:16315-16320);
    // D3D12 exposes no CU/EU count, so we approximate "weak GPU" with the
    // is_igpu classification (integrated parts — which is exactly the weak AMD
    // APU class Vulkan targets, plus Intel iGPUs the user runs).  Override the
    // cap in GFLOP via DX12_FLOPS_CAP.
    uint64_t flops_cap = bctx->dev->is_igpu ? 40'000'000'000ULL : 200'000'000'000ULL;
    if (const char * s = getenv("DX12_FLOPS_CAP")) {
        const long long v = atoll(s);
        if (v > 0) flops_cap = (uint64_t)v * 1'000'000'000ULL;
    }
    uint64_t flops_per_submit = 0;
    if (!stream_flops_disabled) {
        flops_per_submit = std::min(flops_cap, bctx->last_total_flops / 40u);
    }

    // Track unsynced tensor writes for smart barrier insertion.  Persistent
    // across tokens (field of bctx) -- just clear and reuse the bucket array.
    std::unordered_set<uintptr_t> & unsynced_writes = bctx->unsynced_writes;
    unsynced_writes.clear();
    auto & unsynced_reads = bctx->unsynced_reads;
    unsynced_reads.clear();
    auto & unsynced_write_ranges = bctx->unsynced_write_ranges;
    unsynced_write_ranges.clear();
    auto tensor_root = [](const struct ggml_tensor * t) -> uintptr_t {
        while (t->view_src) t = t->view_src;
        return (uintptr_t) t;
    };

    // Debug: DX12_NO_FUSION=1 disables all op fusions for correctness testing
    static bool no_fusion = (getenv("DX12_NO_FUSION") != nullptr);
    // Opt-in V-cache SET_ROWS matvec fusion: a standalone M=1 V-projection
    // matvec (F16 fl=63, Q8_0 fl=67, Q5_0 fl=72) scatters its output directly
    // into the KV cache, eliminating the later V SET_ROWS dispatch.
    // DX12_MMV_POSTOP_FUSION is the master gate for all three projection
    // post-op fusions (Q ROPE, K ROPE+scatter, V scatter); the per-op flags
    // allow enabling each in isolation for A/B benchmarking. Enabled by
    // default; disable the master gate via DX12_MMV_POSTOP_FUSION=0.
    static const bool mmv_postop_fusion = dx12_flag_default_on("DX12_MMV_POSTOP_FUSION");
    static const bool mmv_set_rows_fusion = mmv_postop_fusion || []{
        const char * v = getenv("DX12_MMV_SET_ROWS_FUSION");
        return v && v[0] && v[0] != '0';
    }();
    // Opt-in Q-projection ROPE matvec fusion: the M=1 Q matvec applies RoPE to
    // its output and writes the ROPE result directly, eliminating the standalone
    // ROPE dispatch. NORMAL mode only (falls back for NeoX/mrope).
    static const bool mmv_q_rope_fusion = mmv_postop_fusion || []{
        const char * v = getenv("DX12_MMV_Q_ROPE_FUSION");
        return v && v[0] && v[0] != '0';
    }();
    // Opt-in K-projection ROPE+SET_ROWS matvec fusion: the M=1 K matvec applies
    // RoPE and scatters directly into the K KV cache, eliminating the fused
    // ROPE+VIEW+SET_ROWS (fl=6) dispatch. NORMAL mode only.
    static const bool mmv_k_rope_fusion = mmv_postop_fusion || []{
        const char * v = getenv("DX12_MMV_K_ROPE_FUSION");
        return v && v[0] && v[0] != '0';
    }();
    // Combined Q/K/V projection dispatch: recognize the three M=1
    // projection matvecs that share the normalized activation and collapse them
    // into a single dispatch, absorbing the V and K matvecs plus all three
    // post-ops (Q RoPE, K RoPE+scatter, V scatter). Builds on the projection
    // post-op fusion machinery (node_absorbed[] / mmv_any_postop) and is
    // replay-compatible in the same way. Homogeneous F16, Q8_0, and Q5_0 use
    // flags 75, 76/84, and 77. Mixed Q5_0 Q/K with Q8_0 V uses flag 79.
    // Enabled by default; disable via DX12_QKV_SHARED_DISPATCH=0.
    static const bool qkv_shared_dispatch = dx12_flag_default_on("DX12_QKV_SHARED_DISPATCH");
    static const bool qkv_q8_portable = dx12_flag_default_on("DX12_QKV_Q8_PORTABLE");
    static const bool qkv_mixed_q5_q8 = dx12_flag_default_on("DX12_QKV_MIXED_Q5_Q8");
    static const bool rms_norm_mul_1024 = dx12_flag_default_on("DX12_RMS_NORM_MUL_1024");
    static const bool qk_rope_scale_fusion =
        dx12_flag_default_on("DX12_QK_ROPE_SCALE_FUSION");
    // Opt-in F16 rope-pair de-duplication (fl=63 only): a NORMAL-mode Q/K
    // projection matvec that applies RoPE currently computes both dots of the
    // rotation pair in every group but stores only one row, so the partner row
    // is recomputed by a second group (2x the weight/activation loads on the
    // rope rows). When enabled, one group owns the full pair (row = 2*group_x,
    // row+1), stores both rotated outputs, and the dispatch runs at half the
    // row-group count. Default off; conservative because it only helps the
    // standalone rope matvec path (not the combined QKV dispatch).
    static const bool f16_rope_rows2 = []{
        const char * v = getenv("DX12_F16_ROPE_ROWS2");
        return v && v[0] && v[0] != '0';
    }();
    // Any projection post-op fusion active -- gates the node_absorbed bookkeeping
    // so the record/capture loop skips the absorbed ROPE/SET_ROWS dispatches.
    // Whole-command-list replay stays enabled: the fused matvec bakes only static
    // RoPE/scatter op_params into the CBV, while the per-token positions and KV
    // row indices ride DATA_VOLATILE root SRVs (the pos/index buffers) that the
    // host refreshes each token and the driver re-reads on every execute, so a
    // baked list stays correct across tokens.  dx12_replay_signature hashes the
    // absorbed nodes' bound resources so a buffer reallocation forces a re-record.
    const bool mmv_any_postop = mmv_set_rows_fusion || mmv_q_rope_fusion ||
                                mmv_k_rope_fusion || qkv_shared_dispatch;
    // Debug: per-fusion-type bypasses for bisecting correctness issues
    static bool no_fuse_add_rms_mul   = (getenv("DX12_NO_FUSE_ADD_RMS_MUL")   != nullptr);
    static bool env_no_fuse_rms_mul_rope5 = (getenv("DX12_NO_FUSE_RMS_MUL_ROPE5") != nullptr);
    static bool env_no_fuse_rms_mul_rope3 = (getenv("DX12_NO_FUSE_RMS_MUL_ROPE3") != nullptr);
    // AMD RDNA1/2 (wave-flexible 32/64) produces wrong values from the
    // 3-way / 5-way RMS+MUL+ROPE[+VIEW+SET_ROWS] fusion when the upstream
    // matvec is F16xF32 (verified on RX 6800 with SmolLM2-135M F16: looping
    // output disappears when these fusions are bypassed). Same shader works
    // fine for Q4/Q8 weight models because their dp4a matvec produces
    // numerically different (slightly smoother) F32 activations. Until the
    // exact precision interaction is understood, skip the fusion on RDNA1/2
    // so the unfused rms_norm_mul + rope path is used. No impact on Intel /
    // NVIDIA / AMD RDNA3+ / GCN / CDNA (they keep the fusion perf win).
    const bool amd_rdna12_skip_rope_fusion =
        (bctx->dev->sub_family == DX12_SUBARCH_AMD_RDNA1_2);
    const bool no_fuse_rms_mul_rope5 = env_no_fuse_rms_mul_rope5 || amd_rdna12_skip_rope_fusion;
    const bool no_fuse_rms_mul_rope3 = env_no_fuse_rms_mul_rope3 || amd_rdna12_skip_rope_fusion;
    static bool no_fuse_rms_mul       = (getenv("DX12_NO_FUSE_RMS_MUL")       != nullptr);
    static bool no_fuse_rope_set_rows = (getenv("DX12_NO_FUSE_ROPE_SET_ROWS") != nullptr);
    // Opt-in: DX12_RMSNORM_QUANT_FUSION=1 enables RMS_NORM_MUL + Q8_1
    // pre-pass fusion (rms_norm_mul_quantize_q8_1.hlsl). Default OFF until
    // bench shows a win on a target model; see plan.md "RMS_NORM + Q8_1
    // fusion" section for the per-model A/B/A bench protocol.
    static bool fuse_rms_quant_q8_1   = (getenv("DX12_RMSNORM_QUANT_FUSION") != nullptr);
    // v3 pivot 1: when fusion is on AND we can prove ALL downstream consumers
    // of the rms_norm_mul output are dp4a matmuls that will hit the q8_1 cache
    // (no intermediate dispatch can invalidate), skip the F32 dst write to
    // save ne00*4 bytes of bandwidth per row. Default OFF (env opt-in) so a
    // bad gate cannot silently corrupt model output; enable after bench.
    static bool fuse_rms_quant_q8_1_skip_f32 = (getenv("DX12_RMSNORM_QUANT_FUSION_SKIP_F32") != nullptr);
    const char * q50_subgroup_env = DX12_GETENV("DX12_Q50_SUBGROUP");
    const bool q50_subgroup_auto =
        bctx->dev->sub_family == DX12_SUBARCH_AMD_RDNA3_PLUS;
    // Default on across all vendors: the wave-portable Q5_0 subgroup shader is a
    // boost-or-neutral win on AMD / NVIDIA / Intel (dGPU + iGPU). Opt out with
    // DX12_Q50_SUBGROUP=0. (q50_subgroup_auto still selects the rows2 variant.)
    const bool q50_subgroup_active =
        q50_subgroup_env ? q50_subgroup_env[0] != '0' : true;
    // Diagnostic-only: bypass the Qwen3 QK-Norm gate so the same binary can A/B
    // fused vs gated. Do not commit a change that ships with this enabled.
    static bool force_fuse_qk_norm = (getenv("DX12_FORCE_FUSE_QK_NORM") != nullptr);

    // Diagnostic: DX12_SYNC_PER_OP=1 closes + executes + waits after every
    // node's dispatches so that a TDR can be pinpointed to the exact op that
    // caused it. The timed wait (default 5s, override via DX12_SYNC_PER_OP_MS)
    // detects hangs even before the OS TDR fires; on timeout or device-removed
    // the offending op + key.flags + shapes are dumped and the process aborts.
    // Very slow (no command batching) — for debugging only.
    static const bool sync_per_op = (getenv("DX12_SYNC_PER_OP") != nullptr);
    static const DWORD sync_per_op_ms = []() -> DWORD {
        const char * s = getenv("DX12_SYNC_PER_OP_MS");
        if (!s) return 5000;
        long v = strtol(s, nullptr, 10);
        return (v > 0) ? (DWORD)v : 5000;
    }();

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
            if (phase_profile) {
                bctx->phase_record_start_us = dx12_qpc_us();
            }
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

    // === Whole-graph command-list replay (DX12_COMMAND_REPLAY) ==================
    // Opt-in fast path: for a stable non-prompt M=1 decode graph, re-execute a
    // previously recorded whole-graph command list instead of re-recording every
    // dispatch.  The per-node identity match above (`replay`) already proves the
    // graph topology/shapes are unchanged; dx12_replay_try additionally checks
    // resource base VAs and the dynamic FA n_splits/groups before re-executing.
    // Anything unproven falls back to normal recording below.
    dx12_cmd_replay & cr = bctx->replay;
    bool cr_capture = false;

    // Discrete GPUs lose to a baked list on small decode graphs.  Measured on
    // sub-2-GFLOP decode graphs (SmolLM2-135M 0.22, Qwen3.5-0.8B 1.52): RX 6800
    // +12-14%/+7-9% and RTX 6000 Ada +7.3%/+7.9%/+2.8% from *disabling* replay,
    // and neutral above the threshold (Llama-3.2-1B 2.50, Phi-3-mini 7.54).
    // Integrated GPUs show the opposite sign - an NVIDIA iGPU gains 4-14% *from*
    // replay in the same band - so the gate is deliberately discrete-only.  Tune
    // with DX12_REPLAY_MIN_GFLOP; 0 disables the gate entirely.
    static const uint64_t cr_min_flops = [] {
        const char * env = DX12_GETENV("DX12_REPLAY_MIN_GFLOP");
        return (uint64_t)((env ? atof(env) : 2.0) * 1e9);
    }();
    const bool cr_size_ok = bctx->dev->is_igpu || cr_min_flops == 0 ||
                            bctx->last_decode_flops >= cr_min_flops;

    const bool cr_eligible =
        cr.enabled && !cr.disabled_perm && bctx->dev->use_param_cbv &&
        cr_size_ok &&
        !is_prompt && replay && !bctx->cmd_list_open &&
        !do_profile && !tune_profile_active && !sync_per_op &&
        dump_name_env == nullptr &&
        bctx->q8_1_scratch_retired.empty() &&
        bctx->dev->argsort_scratch_retired.empty();

    if (cr_eligible) {
        cr.stable_streak++;
        if (cr.captured) {
            if (dx12_replay_try(bctx, cgraph)) {
                dx12_replay_stats_dump(bctx);
                if (phase_profile) bctx->phase_graph_return_us = dx12_qpc_us();
                return GGML_STATUS_SUCCESS;
            }
            // Baked list is stale (FA n_splits/groups or a base VA changed).
            cr.captured = false;
            cr.invalidations++;
            if (cr.replays_since_capture < 2) {
                if (++cr.thrash_count >= 8) cr.disabled_perm = true;  // stop thrashing
            } else {
                cr.thrash_count = 0;
            }
        }
        // Arm a fresh capture once the graph has been stable for a few tokens and
        // contains no op that bypasses the CBV param path (ARGSORT/TOP_K use root
        // 32-bit constants directly and cannot be replayed via the CBV region).
        if (!cr.captured && cr.stable_streak >= 3) {
            bool graph_ok = true;
            for (int j = 0; j < cgraph->n_nodes; j++) {
                const ggml_op op = cgraph->nodes[j]->op;
                if (op == GGML_OP_ARGSORT || op == GGML_OP_TOP_K) { graph_ok = false; break; }
            }
            if (graph_ok && dx12_replay_begin_capture(bctx)) {
                cr_capture = true;
            }
        }
    } else {
        cr.stable_streak = 0;
        if (!replay) cr.captured = false;  // topology/shape changed: drop capture
    }

    if (!cr_capture) {
        bctx->ensure_cmd_list_open();
        if (cr.enabled && !is_prompt) cr.records++;
    }
    // === end command-list replay setup =========================================

    // DX12_FUSE_MMV_SET_ROWS / _Q_ROPE / _K_ROPE_SET_ROWS: nodes whose dispatch
    // a preceding projection matvec absorbed (the SET_ROWS it now scatters
    // directly, or the ROPE[+VIEW+SET_ROWS] it now applies inline). Marked at
    // the matvec node (always earlier in the graph) and skipped when the loop
    // reaches them, on both the record and replay passes.
    std::vector<char> node_absorbed(mmv_any_postop ? cgraph->n_nodes : 0, 0);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        uint64_t phase_detail_start_us = phase_profile ? dx12_qpc_us() : 0;
        struct ggml_tensor * node = cgraph->nodes[i];
        if (dx12_trace >= 2) {
            fprintf(stderr, "[DX12_TRACE]  node %d/%d: op=%s name=%s\n",
                    i, cgraph->n_nodes, ggml_op_name(node->op), node->name);
            fflush(stderr);
        }

        // A preceding projection matvec already produced this node's data (the
        // V SET_ROWS scatter, or the Q/K ROPE[+VIEW+SET_ROWS] applied inline).
        // Skip its dispatch on both passes; propagate unsynced status (the
        // destination is already tracked as written by the matvec) and record
        // the skip so the replay fast path bypasses it too.
        if (mmv_any_postop && node_absorbed[i]) {
            if (!replay && !no_replay) {
                rcache.decisions[i].kind = DX12_DEC_SKIP;
            }
            for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                if (unsynced_writes.count((uintptr_t)node->src[s])) {
                    unsynced_writes.insert((uintptr_t)node);
                    break;
                }
            }
            continue;
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
        // SSM_CONV + (optional ADD) + UNARY(SILU) fusion (mirrors Vulkan PR #22653):
        //   fused_ssm_silu      = the trailing UNARY(SILU) node, used as dst
        //   fused_ssm_bias      = the per-channel F32 bias tensor (only when ADD is fused)
        //   fused_ssm_bias_add  = the ADD node (only when bias is fused)
        struct ggml_tensor * fused_ssm_silu       = nullptr;
        struct ggml_tensor * fused_ssm_bias       = nullptr;
        struct ggml_tensor * fused_ssm_bias_add   = nullptr;
        // DX12_FUSE_MMV_SET_ROWS: the SET_ROWS the V-projection matvec absorbs
        // (writes directly into the KV cache). Not adjacent to the matvec, so
        // its graph index is tracked explicitly for the skip + replay paths.
        struct ggml_tensor * fused_mmv_set_rows   = nullptr;
        int                  fused_mmv_set_rows_idx = -1;
        // DX12_FUSE_MMV_Q_ROPE / _K_ROPE_SET_ROWS: the ROPE the Q/K matvec
        // applies inline, plus (K only) the VIEW + SET_ROWS it scatters through.
        // Non-adjacent to the matvec; tracked explicitly for skip + replay.
        struct ggml_tensor * fused_mmv_q_rope     = nullptr;  // Q: absorbed ROPE (dst = rope out)
        int                  fused_mmv_q_rope_idx  = -1;
        struct ggml_tensor * fused_mmv_k_rope     = nullptr;  // K: absorbed ROPE
        int                  fused_mmv_k_rope_idx  = -1;
        struct ggml_tensor * fused_mmv_k_set_rows = nullptr;  // K: absorbed SET_ROWS (dst = KV cache)
        int                  fused_mmv_k_set_rows_idx = -1;
        // DX12_FUSE_MMV_QKV_SHARED: the combined dispatch recorded at the Q
        // projection matvec absorbs the V and K projection matvecs (which share
        // the activation) plus all three post-ops. Handles for the two absorbed
        // matvecs and the three post-op endpoints (Q ROPE out, K/V SET_ROWS KV
        // cache writes); the K ROPE rides fused_mmv_k_rope above.
        bool                 fused_qkv                = false;
        struct ggml_tensor * fused_qkv_q_rope         = nullptr;  int fused_qkv_q_rope_idx     = -1;
        struct ggml_tensor * fused_qkv_v_matvec       = nullptr;  int fused_qkv_v_matvec_idx   = -1;
        struct ggml_tensor * fused_qkv_v_set_rows     = nullptr;  int fused_qkv_v_set_rows_idx = -1;
        struct ggml_tensor * fused_qkv_k_matvec       = nullptr;  int fused_qkv_k_matvec_idx   = -1;
        struct ggml_tensor * fused_qkv_k_set_rows     = nullptr;  int fused_qkv_k_set_rows_idx = -1;
        uint32_t fused_qkv_q_rows = 0;
        uint32_t fused_qkv_k_rows = 0;
        uint32_t fused_qkv_v_rows = 0;
        // Packed-QKV post-op fusion: dispatch at Q ROPE, writing the following
        // SCALE output and the sibling K ROPE cache scatter.
        bool                 fused_qk_postop             = false;
        struct ggml_tensor * fused_qk_scale              = nullptr; int fused_qk_scale_idx      = -1;
        struct ggml_tensor * fused_qk_k_rope             = nullptr; int fused_qk_k_rope_idx     = -1;
        struct ggml_tensor * fused_qk_k_set_rows         = nullptr; int fused_qk_k_set_rows_idx = -1;
        // Q8_1-pre-pass consumer detected by the rms_norm_mul lookahead. When
        // non-null, we promote the rms_norm_mul fusion to flags=12 (the
        // rms_norm_mul_quantize_q8_1 shader) so the standalone quantize_q8_1
        // dispatch for the downstream dp4a matvec is eliminated.
        struct ggml_tensor * fused_rms_quant_consumer = nullptr;
        // v3 pivot 1: when true AND fused_rms_quant_consumer is set, the
        // rms_norm_mul_quantize_q8_1 shader will skip its F32 dst store
        // (consumers all read Q8_1 from bctx->q8_1_scratch via cache).
        bool fused_rms_quant_skip_f32 = false;
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
                case DX12_FUSE_RMS_MUL_QUANT_Q8_1:
                    fused_mul_node = cgraph->nodes[i + 1];
                    key.op         = GGML_OP_RMS_NORM;
                    key.flags      = 12;
                    fused_rms_quant_consumer  = cgraph->nodes[i + 1]; // sentinel non-null (real consumer not needed downstream)
                    fused_rms_quant_skip_f32  = d.fusion_skip_f32;
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
                case DX12_FUSE_SSM_CONV_SILU:
                    fused_ssm_silu = cgraph->nodes[i + 1];
                    key.flags      = 1;
                    break;
                case DX12_FUSE_SSM_CONV_BIAS_SILU: {
                    struct ggml_tensor * add_node = cgraph->nodes[i + 1];
                    fused_ssm_bias_add = add_node;
                    fused_ssm_bias     = (add_node->src[0] == node) ? add_node->src[1] : add_node->src[0];
                    fused_ssm_silu     = cgraph->nodes[i + 2];
                    key.flags          = 2;
                    break;
                }
                case DX12_FUSE_MMV_SET_ROWS: {
                    int sr_idx = i + d.mmv_set_rows_rel;
                    if (d.mmv_set_rows_rel > 0 && sr_idx < cgraph->n_nodes &&
                        cgraph->nodes[sr_idx]->op == GGML_OP_SET_ROWS) {
                        fused_mmv_set_rows     = cgraph->nodes[sr_idx];
                        fused_mmv_set_rows_idx = sr_idx;
                    }
                    break;
                }
                case DX12_FUSE_MMV_Q_ROPE: {
                    int r_idx = i + d.mmv_rope_rel;
                    if (d.mmv_rope_rel > 0 && r_idx < cgraph->n_nodes &&
                        cgraph->nodes[r_idx]->op == GGML_OP_ROPE) {
                        fused_mmv_q_rope     = cgraph->nodes[r_idx];
                        fused_mmv_q_rope_idx = r_idx;
                    }
                    break;
                }
                case DX12_FUSE_MMV_K_ROPE_SET_ROWS: {
                    int r_idx  = i + d.mmv_rope_rel;
                    int sr_idx = i + d.mmv_rope_set_rows_rel;
                    if (d.mmv_rope_rel > 0 && d.mmv_rope_set_rows_rel > 0 &&
                        r_idx < cgraph->n_nodes && sr_idx < cgraph->n_nodes &&
                        cgraph->nodes[r_idx]->op == GGML_OP_ROPE &&
                        cgraph->nodes[sr_idx]->op == GGML_OP_SET_ROWS) {
                        fused_mmv_k_rope         = cgraph->nodes[r_idx];
                        fused_mmv_k_rope_idx     = r_idx;
                        fused_mmv_k_set_rows     = cgraph->nodes[sr_idx];
                        fused_mmv_k_set_rows_idx = sr_idx;
                    }
                    break;
                }
                case DX12_FUSE_MMV_QKV_SHARED: {
                    // Reconstruct the two absorbed projection matvecs and the
                    // three post-op endpoints from the stored relative indices.
                    auto at = [&](int rel, ggml_op expect) -> struct ggml_tensor * {
                        int j = i + rel;
                        if (rel != 0 && j > i && j < cgraph->n_nodes &&
                            cgraph->nodes[j]->op == expect) {
                            return cgraph->nodes[j];
                        }
                        return nullptr;
                    };
                    fused_qkv_q_rope     = at(d.qkv_q_rope_rel,     GGML_OP_ROPE);
                    fused_qkv_v_matvec   = at(d.qkv_v_matvec_rel,   GGML_OP_MUL_MAT);
                    fused_qkv_v_set_rows = at(d.qkv_v_set_rows_rel, GGML_OP_SET_ROWS);
                    fused_qkv_k_matvec   = at(d.qkv_k_matvec_rel,   GGML_OP_MUL_MAT);
                    fused_mmv_k_rope     = at(d.qkv_k_rope_rel,     GGML_OP_ROPE);
                    fused_qkv_k_set_rows = at(d.qkv_k_set_rows_rel, GGML_OP_SET_ROWS);
                    fused_qkv_q_rope_idx     = fused_qkv_q_rope     ? i + d.qkv_q_rope_rel     : -1;
                    fused_qkv_v_matvec_idx   = fused_qkv_v_matvec   ? i + d.qkv_v_matvec_rel   : -1;
                    fused_qkv_v_set_rows_idx = fused_qkv_v_set_rows ? i + d.qkv_v_set_rows_rel : -1;
                    fused_qkv_k_matvec_idx   = fused_qkv_k_matvec   ? i + d.qkv_k_matvec_rel   : -1;
                    fused_mmv_k_rope_idx     = fused_mmv_k_rope     ? i + d.qkv_k_rope_rel     : -1;
                    fused_qkv_k_set_rows_idx = fused_qkv_k_set_rows ? i + d.qkv_k_set_rows_rel : -1;
                    fused_qkv = fused_qkv_q_rope && fused_qkv_v_matvec &&
                                fused_qkv_v_set_rows && fused_qkv_k_matvec &&
                                fused_mmv_k_rope && fused_qkv_k_set_rows;
                    if (fused_qkv) {
                        fused_qkv_q_rows = (uint32_t)fused_qkv_q_rope->ne[0] *
                                           (uint32_t)fused_qkv_q_rope->ne[1];
                        fused_qkv_k_rows = (uint32_t)fused_qkv_k_matvec->ne[0];
                        fused_qkv_v_rows = (uint32_t)fused_qkv_v_matvec->ne[0];
                    }
                    break;
                }
                case DX12_FUSE_QK_ROPE_SCALE_SET_ROWS: {
                    const int scale_idx = i + d.qk_scale_rel;
                    const int rope_idx  = i + d.qk_k_rope_rel;
                    const int sr_idx    = i + d.qk_k_set_rows_rel;
                    if (d.qk_scale_rel > 0 && d.qk_k_rope_rel > 0 &&
                        d.qk_k_set_rows_rel > 0 &&
                        scale_idx < cgraph->n_nodes &&
                        rope_idx < cgraph->n_nodes &&
                        sr_idx < cgraph->n_nodes &&
                        cgraph->nodes[scale_idx]->op == GGML_OP_SCALE &&
                        cgraph->nodes[rope_idx]->op == GGML_OP_ROPE &&
                        cgraph->nodes[sr_idx]->op == GGML_OP_SET_ROWS) {
                        fused_qk_scale              = cgraph->nodes[scale_idx];
                        fused_qk_scale_idx          = scale_idx;
                        fused_qk_k_rope             = cgraph->nodes[rope_idx];
                        fused_qk_k_rope_idx         = rope_idx;
                        fused_qk_k_set_rows         = cgraph->nodes[sr_idx];
                        fused_qk_k_set_rows_idx     = sr_idx;
                        fused_qk_postop             = true;
                        key.op                      = GGML_OP_ROPE;
                        key.flags                   = 87;
                    }
                    break;
                }
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

        // GATED_DELTA_NET picks the S_V-sized + KDA-or-not shader variant.
        // S_V=128 + KDA=0 (the original) goes through the default WB mapping
        // (flags=0); other combinations encode S_V in low bits and KDA in
        // bit 0x100:
        //   flags = S_v (16/32/64) | (kda ? 0x100 : 0)
        //   flags = 0x100 means S_V=128, KDA=1.
        if (node->op == GGML_OP_GATED_DELTA_NET && node->src[2] && node->src[3]) {
            const uint32_t S_v = (uint32_t)node->src[2]->ne[0];
            const uint32_t kda = ((uint32_t)node->src[3]->ne[0] == S_v) ? 1u : 0u;
            uint32_t f = 0u;
            if (S_v != 128) f |= S_v;
            if (kda)        f |= 0x100u;
            key.flags = f;
        }

        // ARGSORT/TOP_K large-N path: when ncols > 1024 we route to the
        // multi-pass argsort_large shader (flags=50) instead of the single-WG
        // bitonic shader (flags=0). The dispatch case below detects flags=50
        // and issues the init + log2(N)*(log2(N)+1)/2 swap + writeout sequence.
        if ((node->op == GGML_OP_ARGSORT || node->op == GGML_OP_TOP_K) &&
            node->src[0] && node->src[0]->ne[0] > 1024) {
            key.flags = 50;
        }

        // SSM_SCAN d_state variant selection: the default ssm_scan shader is
        // built with D_STATE=128. For d_state=256 (Falcon-H1 etc.) route to
        // ssm_scan_d256 (key.flags=256). Other d_state values are rejected
        // by supports_op.
        if (node->op == GGML_OP_SSM_SCAN && node->src[0] && node->src[0]->ne[0] == 256) {
            key.flags = 256;
        }

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
                    // Phi-3 decode has one 3072-element row per dispatch. A full
                    // Ada workgroup reduces the single-row latency substantially.
                    if (rms_norm_mul_1024 &&
                        bctx->dev->adapter_desc.VendorId == dx12_vendor::NVIDIA &&
                        bctx->dev->wave_size == 32 &&
                        node->ne[0] == 3072 && node->ne[1] == 1 &&
                        node->ne[2] == 1 && node->ne[3] == 1) {
                        key.flags = 85;
                    }

                    // Q8_1 pre-pass fusion (opt-in via DX12_RMSNORM_QUANT_FUSION):
                    // look ahead up to 8 nodes for a MUL_MAT whose src[1] is
                    // the MUL output and whose src[0] type triggers the dp4a
                    // Q8_1 pre-pass. On match, promote flags=2 -> flags=12
                    // (rms_norm_mul_quantize_q8_1 shader) and pre-populate
                    // the q8_1 cache after dispatch so the matmul reuses our
                    // scratch instead of dispatching its own quantize_q8_1.
                    if (fuse_rms_quant_q8_1 && next->ne[0] % 32 == 0) {
                        int max_look = (i + 1 + 8 < cgraph->n_nodes) ? (i + 1 + 8) : cgraph->n_nodes;
                        for (int j = i + 2; j < max_look; ++j) {
                            struct ggml_tensor * c = cgraph->nodes[j];
                            if (!c) continue;
                            if (c->op == GGML_OP_MUL_MAT && c->src[1] == next && c->src[0]) {
                                ggml_type w = c->src[0]->type;
                                bool dp4a_consumer =
                                    w == GGML_TYPE_Q4_K || w == GGML_TYPE_Q5_K ||
                                    w == GGML_TYPE_Q6_K || w == GGML_TYPE_Q3_K ||
                                    w == GGML_TYPE_Q8_0 || w == GGML_TYPE_Q4_0 ||
                                    w == GGML_TYPE_Q4_1 ||
                                    (w == GGML_TYPE_Q5_0 && !q50_subgroup_active) ||
                                    w == GGML_TYPE_Q5_1 || w == GGML_TYPE_Q2_K;
                                if (dp4a_consumer) {
                                    fused_rms_quant_consumer = c;
                                    key.flags = 12;

                                    // v3 pivot 1: prove all consumers of `next`
                                    // are dp4a-eligible MUL_MATs that will hit
                                    // the q8_1 cache, AND that no intermediate
                                    // dispatch between i+1 and the last consumer
                                    // can invalidate the cache. If so, set
                                    // skip_f32 — the F32 dst write is dead.
                                    if (fuse_rms_quant_q8_1_skip_f32 &&
                                        next->ne[1] * next->ne[2] * next->ne[3] == 1) {
                                        bool safe = true;
                                        int last_consumer_idx = j;
                                        // Walk i+2..i+8: every node MUST be either
                                        // (a) a dp4a MUL_MAT consuming `next` as src[1], or
                                        // (b) a node that doesn't touch `next` at all.
                                        // If we see any other read of `next`, abort.
                                        for (int k = i + 2; k < max_look && safe; ++k) {
                                            struct ggml_tensor * d_n = cgraph->nodes[k];
                                            if (!d_n) continue;
                                            for (int si = 0; si < GGML_MAX_SRC; ++si) {
                                                if (d_n->src[si] == next) {
                                                    if (d_n->op == GGML_OP_MUL_MAT && si == 1 && d_n->src[0]) {
                                                        ggml_type dw = d_n->src[0]->type;
                                                        bool dc =
                                                            dw == GGML_TYPE_Q4_K || dw == GGML_TYPE_Q5_K ||
                                                            dw == GGML_TYPE_Q6_K || dw == GGML_TYPE_Q3_K ||
                                                            dw == GGML_TYPE_Q8_0 || dw == GGML_TYPE_Q4_0 ||
                                                            dw == GGML_TYPE_Q4_1 ||
                                                            (dw == GGML_TYPE_Q5_0 && !q50_subgroup_active) ||
                                                            dw == GGML_TYPE_Q5_1 || dw == GGML_TYPE_Q2_K;
                                                        // Also require src[1]->ne[1..3] == 1 so the
                                                        // matmul takes the dp4a M=1 path that uses
                                                        // the q8_1 scratch cache.
                                                        bool m1 =
                                                            d_n->src[1]->ne[1] *
                                                            d_n->src[1]->ne[2] *
                                                            d_n->src[1]->ne[3] == 1;
                                                        if (!dc || !m1) safe = false;
                                                        else last_consumer_idx = k;
                                                    } else {
                                                        // any non-dp4a-matmul reader of `next`
                                                        safe = false;
                                                    }
                                                    break;
                                                }
                                            }
                                        }
                                        // Also scan the whole graph past the
                                        // lookahead window: if anything else
                                        // reads `next`, we can't skip F32.
                                        for (int k = max_look; k < cgraph->n_nodes && safe; ++k) {
                                            struct ggml_tensor * d_n = cgraph->nodes[k];
                                            if (!d_n) continue;
                                            for (int si = 0; si < GGML_MAX_SRC; ++si) {
                                                if (d_n->src[si] == next) {
                                                    safe = false;  // far-away consumer; cache invalidated by then
                                                    break;
                                                }
                                            }
                                        }
                                        fused_rms_quant_skip_f32 = safe;
                                        (void)last_consumer_idx;
                                        if (getenv("DX12_RMSNORM_QUANT_FUSION_DEBUG")) {
                                            fprintf(stderr,
                                                "[rms_q81_skipf32] node=%d next=%p safe=%d consumer=%p type=%d K=%lld\n",
                                                i, (void*)next, safe, (void*)c, (int)w, (long long)next->ne[0]);
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Phi-3 packed-QKV post-op fusion. Q and K are views of one packed
        // projection output, so projection fusion cannot apply; combine the
        // following Q ROPE+SCALE and sibling K ROPE+VIEW+SET_ROWS instead.
        // Validated on RTX 6000 Ada: byte-identical Q8 logits and positive
        // interleaved Q4/Q8 decode results. Set DX12_QK_ROPE_SCALE_FUSION=0
        // to restore the standalone Q/K post-ops.
        if (!no_fusion && qk_rope_scale_fusion && node->op == GGML_OP_ROPE &&
            !fused_add_rms_node && i + 1 < cgraph->n_nodes &&
            bctx->dev->adapter_desc.VendorId == dx12_vendor::NVIDIA &&
            bctx->dev->wave_size == 32 &&
            node->type == GGML_TYPE_F32 && node->src[0] &&
            node->src[0]->type == GGML_TYPE_F32 &&
            node->ne[0] == 96 && node->ne[1] == 32 &&
            node->ne[2] == 1 && node->ne[3] == 1 &&
            ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(node)) {
            ggml_tensor * scale = cgraph->nodes[i + 1];
            float scale_value = 0.0f;
            float scale_bias  = 0.0f;
            if (scale && scale->op == GGML_OP_SCALE && scale->src[0] == node) {
                memcpy(&scale_value, scale->op_params, sizeof(float));
                memcpy(&scale_bias, scale->op_params + sizeof(float), sizeof(float));
            }
            const int q_mode = ((const int32_t *)node->op_params)[2];
            const bool q_ok =
                scale && scale->op == GGML_OP_SCALE && scale->src[0] == node &&
                scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) &&
                scale_bias == 0.0f && std::isfinite(scale_value) &&
                (q_mode == 0 || q_mode == 2) &&
                dx12_tensor_consumed_only_by(cgraph, node, scale) &&
                dx12_get_resource(node->src[0]) &&
                dx12_get_resource(node->src[1]) &&
                dx12_get_resource(scale);
            if (q_ok) {
                const int scan_end = std::min(cgraph->n_nodes - 3, i + 16);
                for (int r = i + 2; r <= scan_end; ++r) {
                    ggml_tensor * k_rope = cgraph->nodes[r];
                    ggml_tensor * k_view = cgraph->nodes[r + 1];
                    ggml_tensor * k_sr   = cgraph->nodes[r + 2];
                    if (!k_rope || !k_view || !k_sr ||
                        k_rope->op != GGML_OP_ROPE ||
                        k_view->op != GGML_OP_VIEW ||
                        k_sr->op != GGML_OP_SET_ROWS ||
                        k_view->src[0] != k_rope ||
                        k_sr->src[0] != k_view) {
                        continue;
                    }
                    const ggml_tensor * k_idx = k_sr->src[1];
                    const bool same_layout =
                        k_rope->type == GGML_TYPE_F32 && k_rope->src[0] &&
                        k_rope->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(k_rope->src[0]) &&
                        k_rope->ne[0] == node->ne[0] &&
                        k_rope->ne[1] == node->ne[1] &&
                        k_rope->ne[2] == node->ne[2] &&
                        k_rope->ne[3] == node->ne[3] &&
                        k_rope->src[0]->nb[0] == node->src[0]->nb[0] &&
                        k_rope->src[0]->nb[1] == node->src[0]->nb[1] &&
                        k_rope->src[0]->nb[2] == node->src[0]->nb[2] &&
                        k_rope->src[0]->nb[3] == node->src[0]->nb[3];
                    const bool shared_rope =
                        k_rope->src[1] == node->src[1] &&
                        k_rope->src[2] == node->src[2] &&
                        memcmp(k_rope->op_params, node->op_params,
                               sizeof(node->op_params)) == 0;
                    const bool cache_ok =
                        (k_sr->type == GGML_TYPE_F16 || k_sr->type == GGML_TYPE_F32) &&
                        ggml_is_contiguous(k_view) &&
                        k_view->ne[0] == k_rope->ne[0] * k_rope->ne[1] &&
                        k_view->ne[1] == 1 &&
                        k_idx && (k_idx->type == GGML_TYPE_I32 ||
                                  k_idx->type == GGML_TYPE_I64) &&
                        dx12_get_resource(k_rope->src[0]) &&
                        dx12_get_resource(k_idx) &&
                        dx12_get_resource(k_sr);
                    if (same_layout && shared_rope && cache_ok &&
                        dx12_mmv_rope_chain_exclusive(cgraph, r, r + 2)) {
                        fused_qk_postop             = true;
                        fused_qk_scale              = scale;
                        fused_qk_scale_idx          = i + 1;
                        fused_qk_k_rope             = k_rope;
                        fused_qk_k_rope_idx         = r;
                        fused_qk_k_set_rows         = k_sr;
                        fused_qk_k_set_rows_idx     = r + 2;
                        key.op                      = GGML_OP_ROPE;
                        key.flags                   = 87;
                        break;
                    }
                }
            }
        }

        // Op fusion: ROPE + VIEW + SET_ROWS → fused rope_set_rows
        // Eliminates 2 dispatches per KV cache write
        if (!fused_qk_postop && !no_fusion && !no_fuse_rope_set_rows && node->op == GGML_OP_ROPE && i + 2 < cgraph->n_nodes && !fused_add_rms_node) {
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
        // Wave64 dp4a was previously disabled because the Q8_1 activation
        // quantize shader used wave intrinsics (WaveActiveMax/WaveActiveSum)
        // inside a 32-thread workgroup. On AMD wave64 hardware that runs as
        // a single wave with only 32 of 64 lanes active, and the partial-
        // wave reductions produced a small per-block bias that compounded
        // across layers into model-level output corruption (catastrophic on
        // Phi-3 / Qwen3 / SmolLM2 with K>=576). Switching the quantize
        // shader to explicit shared-memory tree reductions fixed the bias;
        // dp4a is now correct on wave64 (test-backend-ops MUL_MAT passes,
        // 250-token Phi-3 generation coherent). Allow dp4a by default;
        // DX12_NO_DP4A_WAVE64=1 falls back to MR if a regression appears.
        const bool no_dp4a_wave64 = (DX12_GETENV("DX12_NO_DP4A_WAVE64") != nullptr);
        const bool allow_dp4a_wave = !(no_dp4a_wave64 && bctx->dev->wave_size >= 64);
        if (node->op == GGML_OP_MUL_MAT && node->ne[1] == 1 && node->src[0]) {
            ggml_type t = node->src[0]->type;
            if (t == GGML_TYPE_F16 || t == GGML_TYPE_F32 || t == GGML_TYPE_BF16 ||
                t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q5_0 ||
                t == GGML_TYPE_Q5_1 ||
                t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 ||
                t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                t == GGML_TYPE_IQ4_NL ||
                t == GGML_TYPE_IQ2_XXS ||
                t == GGML_TYPE_IQ4_XS ||
                t == GGML_TYPE_IQ3_XXS ||
                t == GGML_TYPE_IQ2_XS ||
                t == GGML_TYPE_IQ2_S ||
                t == GGML_TYPE_IQ3_S ||
                t == GGML_TYPE_IQ1_S ||
                t == GGML_TYPE_IQ1_M ||
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
                        if (t == GGML_TYPE_F16 &&
                            bctx->dev->adapter_desc.VendorId == dx12_vendor::AMD &&
                            bctx->dev->wave_size == 64) {
                            const char * f16_wave64_env = DX12_GETENV("DX12_F16_WAVE64");
                            const bool f16_wave64_auto =
                                bctx->dev->sub_family == DX12_SUBARCH_AMD_RDNA3_PLUS;
                            const bool f16_wave64 =
                                f16_wave64_env ? f16_wave64_env[0] != '0' : f16_wave64_auto;
                            if (f16_wave64) {
                                key.flags = 63;
                            }
                        }
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
                //
                // Q2_K block is 84 bytes = 4-aligned, and every Load4 in the
                // shader is at a 4-aligned offset within the block, so the
                // path is safe on all vendors (verified by inspection of
                // mul_mat_vec_q2k_mr_blocked.hlsl). No vendor gate.
                if (t == GGML_TYPE_Q2_K) {
                    const char * q2k_blk_env = DX12_GETENV("DX12_Q2K_BLOCKED");
                    bool q2k_blocked = (q2k_blk_env == nullptr) ||
                                       (q2k_blk_env[0] != '0');
                    if (q2k_blocked && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float) &&
                        (node->src[0]->ne[0] % 256) == 0) {
                        key.flags = 27;     // Q2_K block-level matvec (default)
                    } else {
                        key.flags = 19;
                    }
                }
                // Q3_K multi-row matvec (2 rows/group, 256 threads).
                // Diagnostic block-level variant available via DX12_Q3K_BLOCKED=1
                // (neutral on Phi-3 Q3_K_M, kept opt-in for further tuning).
                // Multi-row shader requires K to be a multiple of QK_K (256) AND
                // at least one full superblock per row (K >= 256). Smaller K
                // could in principle work but no models use Q3_K with K<256.
                // Gate K>=4096: single-row matvec wins below this on AMD wave64
                // (validated on Qwen3-0.6B Q3_K_M K=1024/2048/3072: MR regressed
                // 53.2 -> 45.6 t/s, -14%).
                if (t == GGML_TYPE_Q3_K && node->src[0]->ne[0] >= 4096) {
                    const char * q3k_blk_env = DX12_GETENV("DX12_Q3K_BLOCKED");
                    bool q3k_blocked = (q3k_blk_env != nullptr) && (q3k_blk_env[0] != '0');
                    // Q3_K block is 110 bytes (2-aligned, not 4-aligned).
                    // mul_mat_vec_q3k_mr_blocked.hlsl uses load4_u_q3k that
                    // reconstructs misaligned Load4 from aligned Loads
                    // (commit 7f7ae83), so the path is safe on all vendors.
                    // Still opt-in pending perf validation on non-AMD.
                    if (q3k_blocked && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float) &&
                        (node->src[0]->ne[0] % 256) == 0) {
                        key.flags = 26;     // Q3_K block-level matvec (opt-in)
                    } else {
                        key.flags = 20;
                    }
                }
                // Q3_K dp4a matvec (opt-in DX12_Q3K_DP4A=1): dot4add_i8packed
                // against Q8_1 activations. Weight q = (qs_2bit | (hbit<<2))
                // kept unsigned in [0,7]; the -4 centering is applied as a
                // -4*sum(q8) bias at the end (mirrors the Q6_K -32 bias trick).
                // Same gating as Q4_K/Q5_K/Q6_K dp4a (off NVIDIA per the
                // cumulative-precision-drift policy); opt-in pending fleet perf
                // validation. Overrides the scalar/blocked selection above.
                if (t == GGML_TYPE_Q3_K) {
                    const char * q3k_dp4a_env = DX12_GETENV("DX12_Q3K_DP4A");
                    bool q3k_dp4a = (q3k_dp4a_env != nullptr) && (q3k_dp4a_env[0] != '0');
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == dx12_vendor::NVIDIA);
                    if (q3k_dp4a && bctx->dev->dp4a_supported && allow_dp4a_wave &&
                        !nvidia &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[0]->ne[0] % 256) == 0) {
                        key.flags = 78;          // Q3_K dp4a multi-row matvec
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
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
                //
                // Opt-in dp4a variant (DX12_Q8_DP4A_MR64=1): same layout but
                // uses dot4add_i8packed against Q8_1-quantised activations.
                // Replaces 1 fp_mad/int8 with ~1/4 fp_mad/int8 in the K loop
                // (one dp4a per 4 weights). Triggers the Q8_1 quantize pre-pass.
                if (t == GGML_TYPE_Q8_0 && bctx->dev->wave_size >= 64 &&
                    node->src[0]->ne[0] < 1536 && (node->src[0]->ne[0] % 32) == 0) {
                    const char * q8_mr64_env = DX12_GETENV("DX12_Q8_MR64");
                    bool q8_mr64 = (q8_mr64_env == nullptr) || (q8_mr64_env[0] != '0');
                    if (q8_mr64 && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float)) {
                        key.flags = 28;     // Q8_0 mr64 (4 rows/group, AMD wave64, scalar)
                        use_dp4a_matvec = false;
                        const char * q8_dp4a_mr64_env = DX12_GETENV("DX12_Q8_DP4A_MR64");
                        bool q8_dp4a_mr64 = (q8_dp4a_mr64_env != nullptr) && (q8_dp4a_mr64_env[0] != '0');
                        if (q8_dp4a_mr64 && bctx->dev->dp4a_supported && allow_dp4a_wave &&
                            ggml_is_contiguous(node->src[1])) {
                            key.flags = 45;       // Q8_0 dp4a mr64 (opt-in)
                            use_dp4a_matvec = true; // triggers Q8_1 quantize pre-pass
                        }
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
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == dx12_vendor::NVIDIA);
                    if (bctx->dev->dp4a_supported && allow_dp4a_wave &&
                        !nvidia &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = 10;          // Q4_K dp4a multi-row matvec
                        if (bctx->dev->q4k_dp4a_use_32) key.flags = 13; // 32-thread variant
                        // Opt-in 4-row variant (DX12_Q4K_DP4A_MR4=1): gated to
                        // AMD wave>=64 only (Vulkan parity — `rm_kq=4` on AMD
                        // GCN). Halves dispatch count vs the 2-row default; pays
                        // an extra Q4_K decode/row but amortises Q8_1 activation
                        // loads across all 4 rows. Override forces flag=46 even
                        // when the 32-thread variant would have been selected.
                        if (bctx->dev->wave_size >= 64) {
                            const char * q4k_mr4_env = DX12_GETENV("DX12_Q4K_DP4A_MR4");
                            bool q4k_mr4 = (q4k_mr4_env != nullptr) && (q4k_mr4_env[0] != '0');
                            if (q4k_mr4) {
                                key.flags = 46;  // Q4_K dp4a NUM_ROWS=4
                            }
                        }
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                }
                // Q5_K: same dp4a treatment as Q4_K (5th bit merged into nibble).
                if (t == GGML_TYPE_Q5_K) {
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == dx12_vendor::NVIDIA);
                    if (bctx->dev->dp4a_supported && allow_dp4a_wave &&
                        !nvidia &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = 14;          // Q5_K dp4a multi-row matvec (256t)
                        // 32t opt-in: global override OR per-dispatch M-threshold.
                        // Larger M dispatches more workgroups with 32t, which on
                        // wave64 iGPUs (AMD 880M) wins by ~2x at M>=9216.
                        uint32_t M = (uint32_t)node->src[0]->ne[1];
                        if (bctx->dev->q5k_dp4a_use_32 ||
                            M >= bctx->dev->q5k_dp4a_m_32_threshold) {
                            key.flags = 16; // 32-thread variant
                        }
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                }
                // Q6_K: dp4a matvec. q = (ql_nibble | (qh<<4)) - 32 fits int8;
                // we keep q as unsigned [0,63] for dp4a and subtract the bias
                // 32*sum(q8) at the end (no min/dmin term, just per-subblock
                // int8 scale). Default-on for NVIDIA; opt out with
                // DX12_NV_Q6K_DP4A=0.
                if (t == GGML_TYPE_Q6_K) {
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == dx12_vendor::NVIDIA);
                    const char * nv_k_dp4a_env = DX12_GETENV("DX12_NV_Q6K_DP4A");
                    const bool use_nv_q6k_dp4a =
                        nvidia && (!nv_k_dp4a_env || nv_k_dp4a_env[0] != '0');
                    // Block-level Q6_K matvec is the default: it amortizes the
                    // block decode across 16 threads/block instead of decoding
                    // per-element, and shares the activation reads across two
                    // output rows. Set DX12_Q6K_BLOCKED=0 to revert to fl=9.
                    //
                    // Q6_K block is 210 bytes (2-aligned, not 4-aligned).
                    // mul_mat_vec_q6k_mr_blocked.hlsl uses load4_u_q6k that
                    // reconstructs misaligned Load4 from aligned Loads
                    // (commit 7f7ae83), so the path is safe on all vendors.
                    const char * q6k_blk_env = DX12_GETENV("DX12_Q6K_BLOCKED");
                    // Q6_K block path TDRs on Intel Arc B-series (wave=16,
                    // shader model 6.7) at K=8192/M=3072 (Phi-3 ffn_down-31)
                    // despite shape passing test-backend-ops. Root cause not
                    // pinned (no obvious shader bug; misaligned-load helper
                    // verified safe; reduction correct for num_waves=16) —
                    // gate the path off for small-wave hardware until a
                    // proper PIX capture can localize the dispatch failure.
                    // Env var still overrides (set DX12_Q6K_BLOCKED=1 to
                    // force, =0 to disable).
                    bool q6k_blocked;
                    if (q6k_blk_env != nullptr) {
                        q6k_blocked = (q6k_blk_env[0] != '0');
                    } else {
                        q6k_blocked = (bctx->dev->wave_size > 16);
                    }
                    if (use_nv_q6k_dp4a) {
                        q6k_blocked = false;
                    }
                    if (q6k_blocked && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float) &&
                        (node->src[0]->ne[0] % 256) == 0) {
                        key.flags = 25;         // Q6_K block-level matvec
                        static const bool dbg_q6k = (getenv("DX12_DBG_Q6K") != nullptr);
                        if (dbg_q6k) {
                            uint64_t src0_off = dx12_tensor_offset(node->src[0]);
                            uint64_t src1_off = dx12_tensor_offset(node->src[1]);
                            uint64_t dst_off  = dx12_tensor_offset(node);
                            fprintf(stderr, "[DX12_DBG_Q6K] node %d name=%s K=%lld M=%lld "
                                            "src0_off=%llu(%%4=%llu,%%16=%llu,nb1=%lld) "
                                            "src1_off=%llu(%%4=%llu,%%16=%llu,nb1=%lld) "
                                            "dst_off=%llu(%%4=%llu)\n",
                                    i, node->name,
                                    (long long)node->src[0]->ne[0],
                                    (long long)node->ne[0],
                                    (unsigned long long)src0_off,
                                    (unsigned long long)(src0_off & 3),
                                    (unsigned long long)(src0_off & 15),
                                    (long long)node->src[0]->nb[1],
                                    (unsigned long long)src1_off,
                                    (unsigned long long)(src1_off & 3),
                                    (unsigned long long)(src1_off & 15),
                                    (long long)node->src[1]->nb[1],
                                    (unsigned long long)dst_off,
                                    (unsigned long long)(dst_off & 3));
                            fflush(stderr);
                        }
                    } else if (bctx->dev->dp4a_supported && allow_dp4a_wave &&
                        (!nvidia || use_nv_q6k_dp4a) &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = 23;          // Q6_K dp4a multi-row matvec
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                    const char * q6k_subgroup_env = DX12_GETENV("DX12_Q6K_SUBGROUP");
                    // Default on across all vendors (wave-portable shader;
                    // boost-or-neutral). Opt out with DX12_Q6K_SUBGROUP=0.
                    const bool q6k_subgroup =
                        q6k_subgroup_env ? q6k_subgroup_env[0] != '0' : true;
                    if (q6k_subgroup && !use_dp4a_matvec &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float) &&
                        (node->src[0]->ne[0] % 256) == 0) {
                        key.flags = 61;
                        use_dp4a_matvec = false;
                        const char * q6k_g64_env = DX12_GETENV("DX12_Q6K_LARGE_G64");
                        const bool q6k_g64 = q6k_g64_env ? q6k_g64_env[0] != '0' : true;
                        if (nvidia && q6k_g64 && node->src[0]->ne[0] >= 3072) {
                            key.flags = 82;
                        }
                    }
                }
                // Q8_0: dp4a matvec — pure int8 dot (no min term, no precision
                // drift), so safe on NVIDIA unlike Q4_K/Q5_K dp4a matvec.
                // Mirrors the Q8_0 batch dp4a path (flag=8) which already runs
                // on NVIDIA. ~2x speedup on Phi-3 Q8_0 generation.
                // Preserve the AMD wave64 routes selected above. The scalar
                // mr64 path (28), optional dp4a mr64 path (45), and large-K
                // mr256v path (18) each intentionally override this generic
                // two-row dp4a route.
                if (t == GGML_TYPE_Q8_0 &&
                    key.flags != 18 && key.flags != 28 && key.flags != 45) {
                    bool small_wave = (bctx->dev->wave_size < 16);
                    if (bctx->dev->dp4a_supported && allow_dp4a_wave && !small_wave &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = 17;          // Q8_0 dp4a multi-row matvec
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                }
                // Q8_0 mr256 (256-thread, scalar int8, 2 rows/group): wave-portable
                // multi-row matvec. The default fl=17 (32-thread dp4a) leaves wide
                // GPUs starving on small-K matvecs because the dispatch only fills
                // ~7% of L40's in-flight capacity. mr256 trades dp4a math for 8x
                // occupancy per launch.
                //
                // Measured L40 (wave=32) tg64/128/256 r=5 vs fl=17 baseline:
                //   SmolLM2-135M  K=576  +10.8% / +10.0% / +9.7%
                //   SmolVLM2-256M K=576  +9.1%  / +11.0% / +10.4%
                //   Phi-3-mini    K=3072 +0.4%  / +0.1%  / -0.7% (neutral)
                // Intel UHD (wave=8): neutral to slight regression at K=576 — the
                // 32-thread dp4a path already fills the device, mr256 just loses
                // dp4a math. Gate excludes wave<32.
                // AMD 880M (wave=64, RDNA3.5): measured -21% on SmolLM2-135M Q8_0
                // tg128 (185 vs 235 t/s). 256-thread group gives only 4 waves on
                // wave64 vs 8 waves on wave32, hurting latency hiding. The same
                // pattern is documented in the q5k_dp4a_32 autotune memory.
                // Gate is `wave_size == 32` to keep the win on validated NV/Intel
                // wave32 paths and exclude AMD wave64 chips.
                // Default-on for wave_size==32 && K<=2048; opt out via DX12_Q8_MR256=0.
                // K threshold bumped 1024 → 2048 to catch SmolLM2-135M ffn_down
                // (K=1536) and similar small-FFN-large-down shapes.
                if (t == GGML_TYPE_Q8_0 && key.flags == 17 &&
                    bctx->dev->wave_size == 32 &&
                    node->src[0]->ne[0] <= 2048) {
                    const char * q8_mr256_env = DX12_GETENV("DX12_Q8_MR256");
                    bool q8_mr256 = (q8_mr256_env == nullptr) || (q8_mr256_env[0] != '0');
                    if (q8_mr256 && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float)) {
                        key.flags = 44;          // Q8_0 mr256 (256-thread, scalar)
                        use_dp4a_matvec = false; // scalar path, no Q8_1 quantize
                    }
                }
                // Q5_0 / Q5_1 dp4a multi-row matvec (dot4add_i8packed + Q8_1 activations)
                // Requires SM 6.4 dp4a, non-tiny wave, F32 contiguous src1,
                // K%32==0 (Q5 block size = 32).
                //
                // NVIDIA gating policy:
                //   Q5_0 — ALLOWED on NVIDIA. Math is structurally identical to
                //          Q8_0 (which has been on NVIDIA since the dp4a path
                //          shipped): pure scale * int8_dot, NO min term, NO FP16
                //          accumulator. The -16 element bias is corrected via the
                //          integer-exact Q8_1 's' field (= d_a * sum(a_int8)),
                //          not via FP accumulation, so the cumulative-precision
                //          drift seen in Q4_K / Q5_K dp4a on NVIDIA cannot apply.
                //   Q5_1 — STILL gated off NVIDIA. Q5_1 has a per-block min term
                //          that the dp4a path folds into an FP16 accumulator,
                //          which is the same drift pattern as Q4_K / Q5_K.
                if (t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1) {
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == dx12_vendor::NVIDIA);
                    bool nvidia_blocked = nvidia && (t == GGML_TYPE_Q5_1);
                    bool small_wave = (bctx->dev->wave_size < 16);
                    if (bctx->dev->dp4a_supported && allow_dp4a_wave &&
                        !nvidia_blocked && !small_wave &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = (t == GGML_TYPE_Q5_0) ? 21 : 22;
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                    // Q5_0 dp4a mr256 (fl=57): 256-thread variant of fl=21.
                    // Mirrors Q4_0 / Q8_0 / IQ4_NL mr256 — on wave=32 hardware
                    // the 32t shader leaves SMs at 1 wave/group at small K;
                    // bumping GROUP_SIZE to 256 gives 8 waves/group and recovers
                    // throughput at small K. AMD wave64 chips would only get
                    // 4 waves/group and regress (see Q8_0 880M -21% measurement);
                    // gate is `wave_size == 32` to keep the win on NV/Intel
                    // wave32 paths only. Default-on for wave_size==32 && K<=2048;
                    // opt out via DX12_Q5_0_MR256=0.
                    if (key.flags == 21 && bctx->dev->wave_size == 32 &&
                        node->src[0]->ne[0] <= 2048) {
                        const char * q50_mr256_env = DX12_GETENV("DX12_Q5_0_MR256");
                        bool q50_mr256 = (q50_mr256_env == nullptr) || (q50_mr256_env[0] != '0');
                        if (q50_mr256) {
                            key.flags = 57;     // Q5_0 dp4a mr256 (256t, 2 rows/group)
                        }
                    }
                }
                // Q4_0: dp4a matvec — mirrors the Q5_0 dp4a path minus the qh
                // fifth-bit handling. Pure int8 dot with -8 bias correction via
                // the Q8_1 's' field, so safe on NVIDIA (same reasoning as Q5_0
                // / Q8_0). Opt out via DX12_NO_Q4_0_DP4A=1 (handy for A/B);
                // also disabled when DX12_NO_DP4A_WAVE64=1 silences the whole
                // wave64 dp4a class via allow_dp4a_wave above.
                if (t == GGML_TYPE_Q4_0) {
                    bool small_wave = (bctx->dev->wave_size < 16);
                    const char * no_q40_dp4a_env = DX12_GETENV("DX12_NO_Q4_0_DP4A");
                    bool no_q40_dp4a = (no_q40_dp4a_env != nullptr) && (no_q40_dp4a_env[0] != '0');
                    if (!no_q40_dp4a && bctx->dev->dp4a_supported && allow_dp4a_wave &&
                        !small_wave &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[1]->ne[0] % 32) == 0) {
                        key.flags = 38;          // Q4_0 dp4a multi-row matvec
                        use_dp4a_matvec = true;  // triggers Q8_1 quantize pre-pass
                    }
                    // Q4_0 dp4a mr256 (fl=56): 256-thread variant of fl=38.
                    // Mirrors Q8_0 mr256 / IQ4_NL mr256 — on wave=32 hardware
                    // the 32t shader leaves SMs at 1 wave/group at small K;
                    // bumping GROUP_SIZE to 256 gives 8 waves/group and
                    // recovers throughput at small K (TinyLlama 1.1B K=2048
                    // proj+ffn, SmolLM2 K=576). AMD wave64 chips would only
                    // get 4 waves/group and regress (see Q8_0 880M -21%
                    // measurement); gate is `wave_size == 32` to keep the
                    // win on NV/Intel wave32 paths only. Default-on for
                    // wave_size==32 && K<=2048; opt out via DX12_Q4_0_MR256=0.
                    if (key.flags == 38 && bctx->dev->wave_size == 32 &&
                        node->src[0]->ne[0] <= 2048) {
                        const char * q40_mr256_env = DX12_GETENV("DX12_Q4_0_MR256");
                        bool q40_mr256 = (q40_mr256_env == nullptr) || (q40_mr256_env[0] != '0');
                        if (q40_mr256) {
                            key.flags = 56;     // Q4_0 dp4a mr256 (256t, 2 rows/group)
                        }
                    }
                }
                // Q5_0 single-wave variant for AMD wave64. The default 32-thread
                // dp4a shader leaves half the wave idle; mr64 (GROUP_SIZE=64,
                // NUM_ROWS=4) fills one full AMD wave and amortizes activation
                // reads 4x. Opt-in via DX12_Q50_MR64=1 pending validation;
                // promote to default after benches confirm gain on SmolVLM2/SmolLM2.
                if (t == GGML_TYPE_Q5_0 && bctx->dev->wave_size >= 64 &&
                    node->src[0]->ne[0] >= 32 && (node->src[0]->ne[0] % 32) == 0) {
                    bool nvidia = (bctx->dev->adapter_desc.VendorId == dx12_vendor::NVIDIA);
                    if (bctx->dev->dp4a_supported && allow_dp4a_wave && !nvidia &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1])) {
                        const char * q50_mr64_env = DX12_GETENV("DX12_Q50_MR64");
                        bool q50_mr64 = (q50_mr64_env != nullptr) && (q50_mr64_env[0] != '0');
                        if (q50_mr64) {
                            key.flags = 29;     // Q5_0 mr64 (single-wave AMD wave64, 4 rows/group)
                            use_dp4a_matvec = true;
                        }
                    }
                }
                // Q5_0 standalone LDS pre-decode wave64 variant (fl=34): mirrors
                // the LDS-pre-decode trick from the R9 Q5_0 fused shader (+16% on
                // SmolLM2 Q4_K_M).  Default-on for AMD wave64 with K <= 1024
                // (LDS scales array bound).  Opt out via DX12_Q50_MR_LDS=0.
                // Placed last so it overrides any earlier Q5_0 routing decision
                // (dp4a fl=21, mr64 fl=29).  Reads src1 as F32 directly so it
                // also clears the dp4a Q8_1 pre-pass requirement.
                if (t == GGML_TYPE_Q5_0) {
                    const char * q50_lds_env = DX12_GETENV("DX12_Q50_MR_LDS");
                    bool q50_lds = (q50_lds_env == nullptr) || (q50_lds_env[0] != '0');
                    bool is_amd_q50 = (bctx->dev->adapter_desc.VendorId == dx12_vendor::AMD);
                    if (q50_lds && is_amd_q50 && bctx->dev->wave_size == 64 &&
                        node->src[0]->ne[0] <= 1024 &&
                        (node->src[0]->ne[0] % 32) == 0 &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1])) {
                        key.flags = 34;
                        use_dp4a_matvec = false;
                    }
                }
                if (t == GGML_TYPE_Q8_0) {
                    const char * q8_wave64_env = DX12_GETENV("DX12_Q8_WAVE64");
                    const bool q8_wave64_auto =
                        bctx->dev->sub_family == DX12_SUBARCH_AMD_RDNA3_PLUS;
                    const bool q8_wave64 =
                        q8_wave64_env ? q8_wave64_env[0] != '0' : q8_wave64_auto;
                    if (q8_wave64 &&
                        bctx->dev->adapter_desc.VendorId == dx12_vendor::AMD &&
                        bctx->dev->wave_size == 64 &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float) &&
                        (node->src[0]->ne[0] % 32) == 0) {
                        key.flags = 64;
                        use_dp4a_matvec = false;
                        const char * q8_wave64_rows2_env = DX12_GETENV("DX12_Q8_WAVE64_ROWS2");
                        const bool q8_wave64_rows2 =
                            q8_wave64_rows2_env ? q8_wave64_rows2_env[0] != '0' : q8_wave64_auto;
                        if (q8_wave64_rows2) {
                            key.flags = 67;
                        }
                    }
                }
                // Vulkan-style Q5_0 float-dequant matvec.
                if (t == GGML_TYPE_Q5_0) {
                    if (q50_subgroup_active &&
                        (node->src[0]->ne[0] % 32) == 0 &&
                        node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1])) {
                        key.flags = 60;
                        use_dp4a_matvec = false;
                        const char * q50_vulkan_rows2_env = DX12_GETENV("DX12_Q50_VULKAN_ROWS2");
                        const bool q50_vulkan_rows2 =
                            q50_vulkan_rows2_env ? q50_vulkan_rows2_env[0] != '0' : q50_subgroup_auto;
                        // The rows2 variant reduces within a single wave only;
                        // restrict it to wave64 so non-wave64 devices keep the
                        // wave-portable flag 60 shader.
                        if (q50_vulkan_rows2 && bctx->dev->wave_size == 64) {
                            key.flags = 72;
                        }
                    }
                }
                // IQ4_NL multi-row matvec (fl=36): 32 threads, 2 rows/group,
                // shares the activation load across both rows. Halves dispatch
                // count and src1 bandwidth vs the single-row mul_mat_vec_iq4_nl
                // (fl=1). SmolLM2-135M Q3_K_M stores its FFN gate/up as IQ4_NL
                // (60× MUL_MAT K=576 N=1536 = 43% of GPU time on baseline);
                // halving dispatches recovers a measurable chunk of that.
                // Default-on for all vendors; opt out via DX12_IQ4NL_MR=0.
                if (t == GGML_TYPE_IQ4_NL) {
                    const char * iq4nl_mr_env = DX12_GETENV("DX12_IQ4NL_MR");
                    bool iq4nl_mr = (iq4nl_mr_env == nullptr) || (iq4nl_mr_env[0] != '0');
                    if (iq4nl_mr && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float) &&
                        (node->src[0]->ne[0] % 32) == 0) {
                        key.flags = 36;     // IQ4_NL multi-row matvec
                    }
                    // mr256 (fl=55): same 2-rows/group layout but 256 threads,
                    // matches Q8_0 mr256 (fl=44) for the "wide GPU launch-bound
                    // on 32-thread shaders at small K" pattern. The 32t mr
                    // (fl=36) leaves wave=32 hardware at 1 wave/group; mr256
                    // puts 8 waves in each group, recovering decode throughput
                    // for small-K small-models (e.g. SmolLM2-135M Q3_K_M
                    // IQ4_NL FFN gate/up @ K=576 N=1536, 60x/token). AMD wave64
                    // chips would only get 4 waves/group and regress (see Q8_0
                    // 880M -21% measurement); gate is `wave_size == 32` to keep
                    // the win on NV/Intel wave32 paths only. Default-on for
                    // wave_size==32 && K<=1024; opt out via DX12_IQ4NL_MR256=0.
                    if (key.flags == 36 && bctx->dev->wave_size == 32 &&
                        node->src[0]->ne[0] <= 1024) {
                        const char * iq4nl_mr256_env = DX12_GETENV("DX12_IQ4NL_MR256");
                        bool iq4nl_mr256 = (iq4nl_mr256_env == nullptr) || (iq4nl_mr256_env[0] != '0');
                        if (iq4nl_mr256) {
                            key.flags = 55;     // IQ4_NL mr256 (256t, 2 rows/group)
                        }
                    }
                }
                // IQ2_XXS multi-row matvec (fl=37): same template as fl=36 but
                // for IQ2_XXS superblock decode. 32 threads, 2 rows/group,
                // shares the per-strip activation load across both rows.
                // Opt-in via DX12_IQ2XXS_MR=1 (Qwen3.5-0.8B IQ2_XXS shows
                // only ~+4% within noise; revisit when a larger pure-IQ2_XXS
                // model is available).
                if (t == GGML_TYPE_IQ2_XXS) {
                    const char * iq2xxs_mr_env = DX12_GETENV("DX12_IQ2XXS_MR");
                    bool iq2xxs_mr = (iq2xxs_mr_env != nullptr) && (iq2xxs_mr_env[0] != '0');
                    if (iq2xxs_mr && node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                        node->src[1]->nb[0] == sizeof(float) &&
                        (node->src[0]->ne[0] % 256) == 0) {
                        key.flags = 37;     // IQ2_XXS multi-row matvec
                    }
                }
                is_matvec_dispatch = true;
            }
        }
        // Q5_0 small-M batch path: two weight rows x up to eight columns per wave.
        if (node->op == GGML_OP_MUL_MAT && node->src[0] &&
            node->src[0]->type == GGML_TYPE_Q5_0 &&
            (node->ne[1] == 2 || (node->ne[1] >= 32 && node->ne[0] >= 1024)) &&
            node->ne[1] <= 64 &&
            node->ne[0] <= 65535 &&
            node->ne[2] == 1 && node->ne[3] == 1 &&
            node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
            node->src[1]->nb[0] == sizeof(float) &&
            (node->src[0]->ne[0] % 32) == 0 &&
            bctx->dev->wave_size == 32 &&
            bctx->dev->adapter_desc.VendorId == dx12_vendor::NVIDIA) {
            const char * env = DX12_GETENV("DX12_Q50_SMALL_M");
            if (env == nullptr || env[0] != '0') {
                key.flags          = 83;
                is_matvec_dispatch = true;
                goto skip_wmma_batch;
            }
        }

        // Q4_K NUM_COLS=2 matvec path: opt-in DX12_Q4K_DP4A_NC2=1, fires for
        if (node->op == GGML_OP_MUL_MAT && node->src[0] &&
            node->src[0]->type == GGML_TYPE_Q4_K &&
            node->ne[1] == 2 && node->ne[2] == 1 && node->ne[3] == 1 &&
            node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(node->src[1]) &&
            (node->src[0]->ne[0] % 256) == 0 &&
            bctx->dev->dp4a_supported && allow_dp4a_wave) {
            const char * q4k_nc2_env = DX12_GETENV("DX12_Q4K_DP4A_NC2");
            bool enable_q4k_nc2 = (q4k_nc2_env != nullptr) && (q4k_nc2_env[0] != '0');
            if (enable_q4k_nc2) {
                key.flags          = 47;
                use_dp4a_matvec    = true;
                is_matvec_dispatch = true;
                goto skip_wmma_batch;
            }
        }
        // Q4_K NUM_COLS=4 (flag=48) / NUM_COLS=8 (flag=49) batch matvec:
        // same template as NC2, opt-in via DX12_Q4K_DP4A_NC4=1 / DX12_Q4K_DP4A_NC8=1.
        // Targets speculative-decoding draft+target workloads (ne11=4 / ne11=8).
        if (node->op == GGML_OP_MUL_MAT && node->src[0] &&
            node->src[0]->type == GGML_TYPE_Q4_K &&
            node->ne[1] == 4 && node->ne[2] == 1 && node->ne[3] == 1 &&
            node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(node->src[1]) &&
            (node->src[0]->ne[0] % 256) == 0 &&
            bctx->dev->dp4a_supported && allow_dp4a_wave) {
            const char * env = DX12_GETENV("DX12_Q4K_DP4A_NC4");
            if (env != nullptr && env[0] != '0') {
                key.flags          = 48;
                use_dp4a_matvec    = true;
                is_matvec_dispatch = true;
                goto skip_wmma_batch;
            }
        }
        if (node->op == GGML_OP_MUL_MAT && node->src[0] &&
            node->src[0]->type == GGML_TYPE_Q4_K &&
            node->ne[1] == 8 && node->ne[2] == 1 && node->ne[3] == 1 &&
            node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(node->src[1]) &&
            (node->src[0]->ne[0] % 256) == 0 &&
            bctx->dev->dp4a_supported && allow_dp4a_wave) {
            const char * env = DX12_GETENV("DX12_Q4K_DP4A_NC8");
            if (env != nullptr && env[0] != '0') {
                key.flags          = 49;
                use_dp4a_matvec    = true;
                is_matvec_dispatch = true;
                goto skip_wmma_batch;
            }
        }
        // Q5_K NC2 (flag=50): opt-in DX12_Q5K_DP4A_NC2=1; ne11==2 only.
        if (node->op == GGML_OP_MUL_MAT && node->src[0] &&
            node->src[0]->type == GGML_TYPE_Q5_K &&
            node->ne[1] == 2 && node->ne[2] == 1 && node->ne[3] == 1 &&
            node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(node->src[1]) &&
            (node->src[0]->ne[0] % 256) == 0 &&
            bctx->dev->dp4a_supported && allow_dp4a_wave) {
            const char * env = DX12_GETENV("DX12_Q5K_DP4A_NC2");
            if (env != nullptr && env[0] != '0') {
                key.flags          = 50;
                use_dp4a_matvec    = true;
                is_matvec_dispatch = true;
                goto skip_wmma_batch;
            }
        }
        // Q6_K NC2 (flag=51): opt-in DX12_Q6K_DP4A_NC2=1; ne11==2 only.
        if (node->op == GGML_OP_MUL_MAT && node->src[0] &&
            node->src[0]->type == GGML_TYPE_Q6_K &&
            node->ne[1] == 2 && node->ne[2] == 1 && node->ne[3] == 1 &&
            node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(node->src[1]) &&
            (node->src[0]->ne[0] % 256) == 0 &&
            bctx->dev->dp4a_supported && allow_dp4a_wave) {
            const char * env = DX12_GETENV("DX12_Q6K_DP4A_NC2");
            if (env != nullptr && env[0] != '0') {
                key.flags          = 51;
                use_dp4a_matvec    = true;
                is_matvec_dispatch = true;
                goto skip_wmma_batch;
            }
        }
        // Q8_0 NC2 (flag=52): opt-in DX12_Q8_DP4A_NC2=1; ne11==2 only.
        if (node->op == GGML_OP_MUL_MAT && node->src[0] &&
            node->src[0]->type == GGML_TYPE_Q8_0 &&
            node->ne[1] == 2 && node->ne[2] == 1 && node->ne[3] == 1 &&
            node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(node->src[1]) &&
            (node->src[0]->ne[0] % 32) == 0 &&
            bctx->dev->dp4a_supported && allow_dp4a_wave) {
            const char * env = DX12_GETENV("DX12_Q8_DP4A_NC2");
            if (env != nullptr && env[0] != '0') {
                key.flags          = 52;
                use_dp4a_matvec    = true;
                is_matvec_dispatch = true;
                goto skip_wmma_batch;
            }
        }
        // For batch MUL_MAT (M > 1), use register-blocked tiled path (flags=4)
        // for types that have wmma shaders
        if (node->op == GGML_OP_MUL_MAT && node->ne[1] > 1 && node->src[0]) {
            ggml_type t = node->src[0]->type;
            // Tiled integer-dot GEMM. This keeps the existing flat Q8_1
            // scratch layout while adding 32x32 weight/activation tile reuse.
            // Q8_0/Q4_K/Q5_K default on for all AMD and Intel Xe-HPG+ at
            // N >= 32: the int-dot tile beats the float-dequant WMMA fallback
            // wherever DXC has no fast hardware matrix path. Measured Q4_K/Q5_K
            // PP: 880M (RDNA3.5) +60-79%, RX 6800 (RDNA1/2) +46-81%. Measured
            // Q8_0 PP on RX 6800: 135M +43-96%, Phi-3 3B +73-160%. On Intel
            // Xe-HPG+ (B390) flag 30 wmma_lds is TDR-disabled, so tiled is the
            // only accelerated path: measured Phi-3 3B Q4_K +173-232%, Q8_0
            // +121-266%. Intel UHD (wave8, flag-59 variant) and NVIDIA keep
            // their existing paths (opt-in here). Other types/devices remain
            // opt-in. DX12_TILED_INTDOT=0/1 disables/enables it.
            const char * tiled_intdot_env = DX12_GETENV("DX12_TILED_INTDOT");
            const bool tiled_intdot_type =
                t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_K ||
                t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q5_0 ||
                t == GGML_TYPE_Q6_K;
            // Q6_K joins the auto set on the same architectures as the other
            // K-quants: its wmma float-dequant fallback runs at 2.45 TFLOP/s
            // against 9.99 for Q4_K at the same shape on RX 6800, which is
            // ~24% of Phi-3 prefill. Every Q4_K_M/Q5_K_M model carries Q6_K
            // for ffn_down and output, so this is not a Phi-3 special case.
            const bool tiled_intdot_auto_type =
                t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_K ||
                t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K;
            const bool tiled_intdot_auto_arch =
                bctx->dev->adapter_desc.VendorId == dx12_vendor::AMD ||
                bctx->dev->arch_family == DX12_ARCH_INTEL_XE_HPG_PLUS;
            // Q5_0 has neither a wmma nor a flat dp4a batch shader, so without
            // the tile it lands on the naive per-element kernel: 0.154 TFLOP/s
            // against 7.55 for Q4_K at the same shape on RX 6800, and slow
            // enough to risk a TDR on larger models. There is no tuned path to
            // regress, so enable it anywhere dp4a exists rather than limiting
            // it to the architectures the other types were measured on.
            const bool tiled_intdot_auto =
                (tiled_intdot_auto_arch && tiled_intdot_auto_type &&
                 node->ne[1] >= 32) ||
                (t == GGML_TYPE_Q5_0 && node->ne[1] >= 32);
            const bool tiled_intdot_enabled = tiled_intdot_env
                ? tiled_intdot_env[0] != '0'
                : tiled_intdot_auto;
            const bool tiled_intdot_k_aligned =
                ((t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q5_0) &&
                 node->src[0]->ne[0] % 32 == 0) ||
                ((t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                  t == GGML_TYPE_Q6_K) &&
                 node->src[0]->ne[0] % 256 == 0);
            if (tiled_intdot_enabled && tiled_intdot_type &&
                node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(node->src[1]) &&
                bctx->dev->dp4a_supported && allow_dp4a_wave &&
                tiled_intdot_k_aligned &&
                node->ne[0] >= 16 &&
                node->ne[1] >= (tiled_intdot_env ? 16 : 32)) {
                // Intel UHD wave8 requires aligned byte extraction for Q8_0's
                // 34-byte blocks; cross-word packed-load reconstruction
                // produces invalid values on the tested driver.
                key.flags =
                    t == GGML_TYPE_Q8_0 &&
                    bctx->dev->arch_family == DX12_ARCH_INTEL_UHD
                        ? 59
                        : 58;
                use_dp4a = true;
                goto skip_wmma_batch;
            }
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
                t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q8_0 ||
                t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1) {
                // Q4_0/Q4_1 wmma is default-on; opt out for A/B with
                // DX12_NO_Q4_0_WMMA=1 / DX12_NO_Q4_1_WMMA=1 (falls back to
                // the scalar mul_mat_q4_0 / mul_mat_q4_1 batch shader).
                if (t == GGML_TYPE_Q4_0) {
                    const char * no_q40_wmma_env = DX12_GETENV("DX12_NO_Q4_0_WMMA");
                    if (no_q40_wmma_env != nullptr && no_q40_wmma_env[0] != '0') {
                        goto skip_wmma_batch;
                    }
                }
                if (t == GGML_TYPE_Q4_1) {
                    const char * no_q41_wmma_env = DX12_GETENV("DX12_NO_Q4_1_WMMA");
                    if (no_q41_wmma_env != nullptr && no_q41_wmma_env[0] != '0') {
                        goto skip_wmma_batch;
                    }
                }
                const bool force_q80_dp4a = (DX12_GETENV("DX12_FORCE_Q8_0_BATCH_DP4A") != nullptr);
                if (t == GGML_TYPE_Q8_0 && force_q80_dp4a &&
                    bctx->dev->dp4a_supported && allow_dp4a_wave &&
                    node->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(node->src[1]) &&
                    (node->src[1]->ne[0] % 32) == 0) {
                    key.flags = 8;  // dp4a flat batch path (override)
                    use_dp4a = true;
                } else {
                    key.flags = 4;
                    // Tiny-K wmma variant: when K<=64 the existing
                    // mul_mat_wmma.hlsl spends most of its time on
                    // GroupMemoryBarrierWithGroupSync between K-tiles
                    // (4 K-tiles × 2 syncs for 64 madds of real work).
                    // mul_mat_wmma_kfull.hlsl folds the whole K into one
                    // tile (BK=64) with zero K-loop syncs. Same 32x32
                    // output tile, same dispatch, just bigger LDS (16 KiB
                    // - still well under the 32 KiB D3D12 minimum).
                    // Targets CLIP attention Q.K^T (K=64 N=1024 M=1024):
                    // 27.55 -> 23.15 ms (-16%) on B390 wave=16.
                    // Default-on for K<=64 + F16/F32; DX12_NO_WMMA_KFULL=1 opts out.
                    if ((t == GGML_TYPE_F16 || t == GGML_TYPE_F32) &&
                        node->src[1] && node->src[0]->ne[0] <= 64) {
                        static const bool wmma_kfull_off =
                            ([]{ const char *e = DX12_GETENV("DX12_NO_WMMA_KFULL");
                                 return (e != nullptr && e[0] != '0'); })();
                        if (!wmma_kfull_off) {
                            key.flags = 54;
                            goto skip_wmma_batch;
                        }
                    }
                    // FP16 64x64 wmma variant for F16xF16 GEMM (e.g. CLIP
                    // vision encoder). 4x4 register tile -> 2:1 compute:LDS
                    // ratio (vs 1:1 for the 2x2 baseline). Half-precision
                    // LDS keeps the footprint at 4 KiB despite the larger
                    // output tile.
                    //
                    // Default on for large AMD RDNA3+ GEMMs. SmolVLM2 CLIP
                    // on the 880M improved 20% across the vision GEMMs and
                    // 8-15% end-to-end with M=1024, while its M=64 projection
                    // regressed by over 2x from reduced parallelism. Requiring
                    // both output dimensions >=256 keeps the validated win and
                    // avoids the small-M tail. Other vendors remain opt-in
                    // pending equivalent validation. DX12_WMMA_FP16=0/1
                    // explicitly disables/enables the path.
                    if (t == GGML_TYPE_F16 && bctx->dev->fp16_supported &&
                        node->ne[0] >= 64 && node->ne[1] >= 64) {
                        const char * wmma_fp16_env = DX12_GETENV("DX12_WMMA_FP16");
                        const bool wmma_fp16_auto =
                            bctx->dev->sub_family == DX12_SUBARCH_AMD_RDNA3_PLUS &&
                            node->ne[0] >= 256 && node->ne[1] >= 256;
                        const bool wmma_fp16 =
                            wmma_fp16_env ? wmma_fp16_env[0] != '0' : wmma_fp16_auto;
                        if (wmma_fp16) {
                            key.flags = 53;
                            goto skip_wmma_batch;
                        }
                    }
                    // Cooperative-LDS Q4_K wmma variant: pre-decodes Q4_K
                    // scales/mins per (n_local, kt) into LDS once instead of
                    // having every thread re-decode per element. Originally
                    // shipped default-on as "portable to all DX12 vendors",
                    // but reproducibly triggers DXGI_ERROR_DEVICE_REMOVED
                    // (HRESULT 0x887A0005) on Intel Arc B390 (wave=16) the
                    // first time the PSO is dispatched. Real-model PP A/B on
                    // Phi-3-mini Q4_K_M (measured 2026-05-21) shows:
                    //   AMD     wave=64  +15-23% (verified earlier)
                    //   NVIDIA  wave=32  +20-22% (L40)
                    //   Intel   wave=8   +9-14%  (UHD)
                    //   Intel   wave=16  TDR     (B390 only failing class)
                    // Gate on arch family != Intel Xe-HPG+ to enable the
                    // win everywhere safe while keeping the wave16 TDR path
                    // disabled until the B390 (Xe3 Panther Lake iGPU) issue
                    // is root-caused. Note Xe-HPG+ here lumps Alchemist,
                    // Battlemage, Lunar Lake and Panther Lake — all share
                    // wave>=16 and the TDR class. DX12_Q4K_WMMA_LDS=0/1
                    // env override still wins for opt-out / forced opt-in.
                    if (t == GGML_TYPE_Q4_K) {
                        const bool wave_safe = (bctx->dev->arch_family != DX12_ARCH_INTEL_XE_HPG_PLUS);
                        const char * q4k_lds_env = DX12_GETENV("DX12_Q4K_WMMA_LDS");
                        bool q4k_lds_enabled;
                        if (q4k_lds_env == nullptr) {
                            q4k_lds_enabled = wave_safe;
                        } else {
                            q4k_lds_enabled = (q4k_lds_env[0] != '0');
                        }
                        if (q4k_lds_enabled) {
                            key.flags = 30;  // mul_mat_q4k_wmma_lds
                        }
                    }
                }
            }
            skip_wmma_batch: ;
        }

        // SOFT_MAX cached variant for ne00 <= 1024 (CLIP attention,
        // short-context decode). One global read per element vs three.
        // Neutral on B390 (compute-bound by exp), kept opt-in for vendors
        // where global memory bandwidth dominates. DX12_SOFT_MAX_CACHED=1.
        if (node->op == GGML_OP_SOFT_MAX && node->src[0] &&
            node->src[0]->ne[0] <= 1024) {
            static const bool soft_max_cached_on =
                (DX12_GETENV("DX12_SOFT_MAX_CACHED") != nullptr);
            if (soft_max_cached_on) {
                key.flags = 1;
            }
        }

        // IQ batch (M > 1) routing: per-element dequant batch shader using
        // the shared mul_mat_quant.hlsli template. These IQ types have no
        // wmma/dp4a batch variant, so we fall back to a 1-output-per-thread
        // dispatch. Same dequant code as the MMID variants, just without
        // the expert id lookup.
        if (node->op == GGML_OP_MUL_MAT && node->ne[1] > 1 && node->src[0]) {
            ggml_type t = node->src[0]->type;
            if (t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ2_XS ||
                t == GGML_TYPE_IQ2_S   || t == GGML_TYPE_IQ3_XXS ||
                t == GGML_TYPE_IQ3_S   || t == GGML_TYPE_IQ1_S ||
                t == GGML_TYPE_IQ1_M   || t == GGML_TYPE_IQ4_XS) {
                key.flags = 43;
            }
        }

        // Cooperative small-token MoE matvec for dense and common quant types.
        // This is lossless relative to the scalar path: F32 activations are read
        // directly and K is split across a 32-thread group. Q8_0 has a faster
        // DP4A override below.
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[0]) {
            const ggml_type t = node->src[0]->type;
            const bool coop_type =
                t == GGML_TYPE_F32  || t == GGML_TYPE_F16  || t == GGML_TYPE_BF16 ||
                t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 ||
                t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1 ||
                t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K ||
                t == GGML_TYPE_IQ4_NL || t == GGML_TYPE_IQ4_XS;
            const char * moe_coop_env = DX12_GETENV("DX12_MOE_COOP");
            const bool moe_coop = !moe_coop_env || moe_coop_env[0] != '0';
            if (moe_coop && coop_type && node->src[1] && node->src[2] &&
                node->src[1]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(node->src[1]) &&
                node->src[2]->ne[1] <= 8 &&
                dx12_ceil_div((uint32_t)node->ne[0], 2) <= 65535) {
                key.flags = 1;
            }
        }

        // Q8_0 MoE generation: Vulkan routes <=8-token MUL_MAT_ID through an
        // expert-aware DP4A matvec after quantizing the F32 activation to Q8_1.
        // The generic DX12 path assigns one whole K-loop to every output thread,
        // which dominates PhiMoE generation. Reuse the existing Q8_1 scratch
        // and quantize cache, then process two output rows per 32-thread group.
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] &&
            node->src[0]->type == GGML_TYPE_Q8_0) {
            const char * moe_q8_env = DX12_GETENV("DX12_MOE_Q8_DP4A");
            const bool moe_q8_dp4a = !moe_q8_env || moe_q8_env[0] != '0';
            const bool small_wave = bctx->dev->wave_size < 16;
            if (moe_q8_dp4a && bctx->dev->dp4a_supported && allow_dp4a_wave &&
                !small_wave && node->src[1] && node->src[2] &&
                node->src[1]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(node->src[1]) &&
                node->src[2]->ne[1] <= 8 &&
                (node->src[0]->ne[0] % 32) == 0 &&
                dx12_ceil_div((uint32_t)node->ne[0], 2) <= 65535) {
                key.flags = 17;
                use_dp4a_matvec = true;
            }
        }

        // MUL_MAT_ID Q4_K: small-wave devices (Intel UHD, wave=8) cannot
        // sustain the per-element decode template in mul_mat_id_quant.hlsli
        // for large K — cumulative per-thread work yields wrong outputs
        // for later thread groups. Route them to a block-level decode
        // variant (mul_mat_id_q4k_block.hlsl) that hoists the Q4_K scale/
        // min unpack out of the K loop. Larger-wave devices keep the
        // existing per-element shader by default (no observed regression
        // there); override via DX12_MMID_Q4K_BLOCK=1/0.
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] &&
            node->src[0]->type == GGML_TYPE_Q4_K && (node->src[0]->ne[0] % 256) == 0) {
            const int  opt   = dx12_env_mmid_q4k_block();
            const bool intel_uhd = (bctx->dev->arch_family == DX12_ARCH_INTEL_UHD);
            const bool use_block = (opt == 1) || (opt == -1 && intel_uhd);
            if (use_block) {
                key.flags = 51;
            }
        }

        // Op fusion: MUL_MAT(M=1) + ADD → matvec with fused bias add
        if (!no_fusion && is_matvec_dispatch && key.flags != 83 && i + 1 < cgraph->n_nodes) {
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

        // Op fusion: SSM_CONV [+ ADD(bias)] + UNARY(SILU)
        //   Pattern 1: SSM_CONV -> UNARY(SILU)
        //   Pattern 2: SSM_CONV -> ADD -> UNARY(SILU)
        // Mirrors Vulkan PR #22653.  Opt-out via DX12_NO_FUSE_SSM_CONV_SILU=1.
        // F32-only because the ssm_conv shader is F32-only.
        static bool no_fuse_ssm_conv_silu = (getenv("DX12_NO_FUSE_SSM_CONV_SILU") != nullptr);
        if (!no_fusion && !no_fuse_ssm_conv_silu &&
            node->op == GGML_OP_SSM_CONV && node->type == GGML_TYPE_F32 &&
            i + 1 < cgraph->n_nodes) {
            struct ggml_tensor * next = cgraph->nodes[i + 1];
            // Pattern 1: SSM_CONV -> SILU
            if (next->op == GGML_OP_UNARY &&
                ggml_get_unary_op(next) == GGML_UNARY_OP_SILU &&
                next->src[0] == node &&
                next->type == GGML_TYPE_F32) {
                fused_ssm_silu = next;
                key.flags      = 1;
            }
            // Pattern 2: SSM_CONV -> ADD -> SILU
            else if (next->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
                     next->type == GGML_TYPE_F32) {
                struct ggml_tensor * silu = cgraph->nodes[i + 2];
                if (silu->op == GGML_OP_UNARY &&
                    ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU &&
                    silu->src[0] == next &&
                    silu->type == GGML_TYPE_F32) {
                    struct ggml_tensor * bias = nullptr;
                    if (next->src[0] == node) bias = next->src[1];
                    else if (next->src[1] == node) bias = next->src[0];
                    // Bias must be a contiguous F32 vector with one element per output channel (ne[0]).
                    if (bias && bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias) &&
                        bias->ne[0] == node->ne[0] && ggml_nelements(bias) == node->ne[0]) {
                        fused_ssm_bias_add = next;
                        fused_ssm_bias     = bias;
                        fused_ssm_silu     = silu;
                        key.flags          = 2;
                    }
                }
            }
        }

        // R9 op fusion: MUL_MAT(W_gate, M=1) + MUL_MAT(W_up, M=1) + GLU(SWIGLU split)
        // In topological order ggml_swiglu_split's src[0] (gate) is visited
        // before src[1] (up), so the gate matvec lands at node[i] and the up
        // matvec at node[i+1].  GLU lands at node[i+2].
        //
        // F16 weights ship default-on (mul_mat_vec_glu, fl=24).
        // Q5_0 weights are opt-in via DX12_MMV_GLU_FUSION_Q50=1 (mul_mat_vec_glu_q5_0,
        // fl=31) — fires for SmolLM2/SmolVLM2 K=576 FFN where Q4_K_M weights fall
        // back to Q5_0.  On AMD Radeon 880M the fusion eliminates 30 dispatches and
        // the GLU pass per token but the fused shader is ~2x slower per call (4 vs
        // 2 accumulators per group), so net is within noise on that workload.
        // Kept opt-in for users whose dispatch overhead dominates more than ours.
        // Phi-3 uses LLM_FFN_SWIGLU (single 2*n_ff projection) and never matches.
        const bool no_mmv_glu = (DX12_GETENV("DX12_NO_MMV_GLU_FUSION") != nullptr);
        // Q5_0 R9 fusion: default-on (no NVIDIA dp4a competition since Q5_0
        // dp4a is itself NVIDIA-skipped for safety; safe across vendors).
        // Gated off for Intel-class architectures (UHD / Xe-HPG+ including
        // Arc A/B and integrated Xe2/Xe3) where it regresses SmolLM2-135M
        // Q4_K_M tg256 by ~12% on B390 (281 -> 252 t/s).
        // Opt out via DX12_MMV_GLU_FUSION_Q50=0; force on via =1.
        const char * q50_glu_env = DX12_GETENV("DX12_MMV_GLU_FUSION_Q50");
        const bool q50_glu_arch_ok = (bctx->dev->arch_family != DX12_ARCH_INTEL_UHD &&
                                      bctx->dev->arch_family != DX12_ARCH_INTEL_XE_HPG_PLUS);
        bool enable_q50_glu = q50_subgroup_active ||
                              ((q50_glu_env == nullptr) ? q50_glu_arch_ok : (q50_glu_env[0] != '0'));
        // Q4_K / Q5_K R9 fusion: AMD-only by default. The R9 path clears
        // use_dp4a_matvec, which on NVIDIA would replace the tuned dp4a
        // kernel (fl=15/16) with an untested non-dp4a path -> likely
        // regression on RTX. Force-enable on any vendor with =1, force-off
        // with =0.
        constexpr UINT VENDOR_AMD_R9 = dx12_vendor::AMD;
        bool is_amd_r9 = (bctx->dev->adapter_desc.VendorId == VENDOR_AMD_R9);
        const char * q4k_glu_env = DX12_GETENV("DX12_MMV_GLU_FUSION_Q4K");
        bool enable_q4k_glu = (q4k_glu_env == nullptr) ? is_amd_r9 : (q4k_glu_env[0] != '0');
        const char * q5k_glu_env = DX12_GETENV("DX12_MMV_GLU_FUSION_Q5K");
        bool enable_q5k_glu = (q5k_glu_env == nullptr) ? is_amd_r9 : (q5k_glu_env[0] != '0');
        // Q3_K R9 fusion: opt-in via DX12_MMV_GLU_FUSION_Q3K=1. Q3_K decode in
        // SmolLM2 Q3_K_M burns 30%+ of decode on separate gate/up matvecs on
        // Intel B390; the fused dispatch turns 60 dispatches gate+up + 30 GLU
        // into 30 fused dispatches. Default off (un-tested across vendors).
        const char * q3k_glu_env = DX12_GETENV("DX12_MMV_GLU_FUSION_Q3K");
        bool enable_q3k_glu = (q3k_glu_env != nullptr) && (q3k_glu_env[0] != '0');
        // Q8_0 R9 fusion: NVIDIA-default-on (+9% per-graph on SmolLM2/SmolVLM2
        // K=576 FFN: fused fl=35 turns 60 dispatches gate+up + 30 GLU dispatches
        // into 30 fused dispatches; on RTX 6000 Ada the dispatch reduction wins
        // net +3-9%. AMD wave64 is default-off: repeated tg512 A/B/A runs on
        // the 880M were neutral-to-slower for both models.
        // Force-enable on any vendor with =1, force-off with =0.
        constexpr UINT VENDOR_NVIDIA_Q80 = dx12_vendor::NVIDIA;
        bool nvidia_q80 = (bctx->dev->adapter_desc.VendorId == VENDOR_NVIDIA_Q80);
        const char * q80_glu_env = DX12_GETENV("DX12_MMV_GLU_FUSION_Q80");
        const char * q80_glu_dp4a_env = DX12_GETENV("DX12_MMV_GLU_FUSION_Q80_DP4A");
        // Default on: the dp4a fused Q8_0 SwiGLU path (cross-wave reduction,
        // correct on any wave size; dp4a hardware still required by the guard
        // below) is a boost-or-neutral win across vendors. Opt out with
        // DX12_MMV_GLU_FUSION_Q80_DP4A=0.
        const bool q80_glu_dp4a = q80_glu_dp4a_env ? q80_glu_dp4a_env[0] != '0' : true;
        const char * q80_glu_wave64_env = DX12_GETENV("DX12_Q80_GLU_WAVE64_ROWS2");
        const bool q80_glu_wave64_auto =
            bctx->dev->sub_family == DX12_SUBARCH_AMD_RDNA3_PLUS;
        const bool q80_glu_wave64 =
            q80_glu_wave64_env ? q80_glu_wave64_env[0] != '0' : q80_glu_wave64_auto;
        bool enable_q80_glu = q80_glu_dp4a || q80_glu_wave64 ||
                              ((q80_glu_env == nullptr) ? nvidia_q80 : (q80_glu_env[0] != '0'));
        if (!no_fusion && !no_mmv_glu && !fused_bias_add && is_matvec_dispatch &&
            i + 2 < cgraph->n_nodes &&
            node->op == GGML_OP_MUL_MAT && node->src[0] && node->src[1] &&
            (node->src[0]->type == GGML_TYPE_F16 ||
             (enable_q50_glu && node->src[0]->type == GGML_TYPE_Q5_0) ||
             (enable_q4k_glu && node->src[0]->type == GGML_TYPE_Q4_K &&
              node->src[0]->ne[0] <= 4096) ||
             (enable_q5k_glu && node->src[0]->type == GGML_TYPE_Q5_K &&
              node->src[0]->ne[0] <= 4096) ||
             (enable_q3k_glu && node->src[0]->type == GGML_TYPE_Q3_K) ||
             (enable_q80_glu && node->src[0]->type == GGML_TYPE_Q8_0 &&
              node->src[0]->ne[0] <= 1024)) &&
            node->ne[1] == 1) {
            struct ggml_tensor * mm_up = cgraph->nodes[i + 1];
            struct ggml_tensor * glu   = cgraph->nodes[i + 2];
            if (mm_up->op == GGML_OP_MUL_MAT && glu->op == GGML_OP_GLU &&
                mm_up->src[0] && mm_up->src[1] &&
                mm_up->src[0]->type == node->src[0]->type &&    // gate and up same quant
                mm_up->src[1] == node->src[1] &&            // share activation
                mm_up->ne[0] == node->ne[0] &&              // same output width N
                mm_up->ne[1] == 1 &&
                mm_up->src[0]->ne[0] == node->src[0]->ne[0] &&  // same K
                mm_up->src[0]->nb[1] == node->src[0]->nb[1] &&  // same row stride
                glu->src[0] == node && glu->src[1] == mm_up &&  // gate first, up second
                ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                ((const int32_t *)glu->op_params)[1] == 0 &&    // swapped == false
                glu->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32) {
                // R9 reads activation (node->src[1]) and writes glu output in
                // a single dispatch.  The ggml memory allocator may alias the
                // activation buffer with the SwiGLU output buffer (since the
                // activation dies after the gate/up matvecs and the SwiGLU
                // output starts there in the unfused schedule).  In a fused
                // dispatch this becomes a read/write race: thread groups that
                // start later see partially-written activation values where
                // earlier groups have already written their output rows.
                // Skip the fusion when the activation and the SwiGLU output
                // ranges overlap so the unfused (correct) path runs instead.
                const uint8_t * act_lo = (const uint8_t *)node->src[1]->data;
                const uint8_t * act_hi = act_lo + ggml_nbytes(node->src[1]);
                const uint8_t * dst_lo = (const uint8_t *)glu->data;
                const uint8_t * dst_hi = dst_lo + ggml_nbytes(glu);
                bool aliases = (act_lo && dst_lo && act_lo < dst_hi && dst_lo < act_hi);
                if (aliases) {
                    static const bool log_alias = (getenv("DX12_R9_LOG") != nullptr);
                    static int alias_log_count = 0;
                    if (log_alias && alias_log_count < 8) {
                        fprintf(stderr,
                            "[DX12_R9] skip fusion (alias): act=%s data=%p nb=%zu  glu=%s data=%p nb=%zu\n",
                            node->src[1]->name, node->src[1]->data, ggml_nbytes(node->src[1]),
                            glu->name, glu->data, ggml_nbytes(glu));
                        fflush(stderr);
                        alias_log_count++;
                    }
                } else {
                    fused_mmv_glu_up   = mm_up;
                    fused_mmv_glu_glu  = glu;
                    // F16 -> mul_mat_vec_glu (fl=24); Q5_0 -> mul_mat_vec_glu_q5_0 (fl=31);
                    // Q4_K -> mul_mat_vec_glu_q4_k (fl=32); Q5_K -> mul_mat_vec_glu_q5_k (fl=33);
                    // Q8_0 -> mul_mat_vec_glu_q8_0 (fl=35); Q3_K -> mul_mat_vec_glu_q3_k (fl=39)
                    if (node->src[0]->type == GGML_TYPE_Q5_0) {
                        key.flags = q50_subgroup_active ? 66 : 31;
                        const char * q50_glu_vulkan_rows2_env = DX12_GETENV("DX12_Q50_GLU_VULKAN_ROWS2");
                        const bool q50_glu_vulkan_rows2 =
                            q50_glu_vulkan_rows2_env ? q50_glu_vulkan_rows2_env[0] != '0' : q50_subgroup_auto;
                        // rows2 reduces within a single wave only; keep it wave64.
                        if (key.flags == 66 && q50_glu_vulkan_rows2 && bctx->dev->wave_size == 64) {
                            key.flags = 73;
                        }
                    }
                    else if (node->src[0]->type == GGML_TYPE_Q4_K) key.flags = 32;
                    else if (node->src[0]->type == GGML_TYPE_Q5_K) key.flags = 33;
                    else if (node->src[0]->type == GGML_TYPE_Q8_0) {
                        key.flags = q80_glu_wave64 &&
                                    bctx->dev->adapter_desc.VendorId == dx12_vendor::AMD &&
                                    bctx->dev->wave_size == 64 &&
                                    node->src[1]->type == GGML_TYPE_F32 &&
                                    ggml_is_contiguous(node->src[1]) &&
                                    (node->src[0]->ne[0] % 32) == 0 ? 74 : 35;
                    }
                    else if (node->src[0]->type == GGML_TYPE_Q3_K) key.flags = 39;
                    else                                          key.flags = 24;
                    // DX12_MMV_GLU_FUSION_Q80_DP4A is default-on (opt out with =0).
                    // The dp4a GLU shader now has a cross-wave reduction, so it
                    // is correct on any wave size; dp4a support is still required.
                    if (node->src[0]->type == GGML_TYPE_Q8_0 && q80_glu_dp4a &&
                        bctx->dev->dp4a_supported &&
                        allow_dp4a_wave && node->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(node->src[1]) &&
                        (node->src[0]->ne[0] % 32) == 0) {
                        key.flags = 62;
                        use_dp4a_matvec = true;
                    }
                    // Existing R9 shaders read src1 as F32. Flag 62 is the
                    // exception and consumes the Q8_1 scratch buffer.
                    use_dp4a_matvec = key.flags == 62;
                }
            }
        }

        // Combined Q/K/V projection dispatch (DX12_QKV_SHARED_DISPATCH): the
        // three M=1 projection matvecs share the normalized activation. Recorded
        // once at the first (Q) matvec, a single dispatch produces all three
        // projections and applies all three post-ops (Q RoPE -> rope output, K
        // RoPE -> KV scatter, V -> KV scatter); the V and K matvecs plus every
        // post-op node are absorbed. Homogeneous weight type: F16 (fl=11/12/63
        // -> 75), Q8_0 (wave64 fl=67 -> 76, NVIDIA wave32 fl=44 -> 84),
        // Q5_0 (fl=72 -> 77). The F16 and Q5_0 combined shaders are portable to
        // NVIDIA wave32. DX12_QKV_Q8_PORTABLE=0 disables the 256-thread Q8_0
        // combined shader used for the validated small-K NVIDIA route.
        // Mixed Q5_0 Q/K + Q8_0 V uses fl=79, with contiguous Wq|Wk in t0 and
        // Wv bound independently at t6. K/V caches must share a buffer.
        // Structural detection only.
        const ggml_type qkv_wtype =
            (node->op == GGML_OP_MUL_MAT && node->src[0]) ? node->src[0]->type : GGML_TYPE_F32;
        const bool qkv_q8_portable_route =
            qkv_q8_portable &&
            bctx->dev->adapter_desc.VendorId == dx12_vendor::NVIDIA &&
            bctx->dev->wave_size == 32 &&
            key.flags == 44;
        const int qkv_combined_flag =
            (qkv_wtype == GGML_TYPE_F16 && (key.flags == 11 || key.flags == 12 || key.flags == 63)) ? 75 :
            (qkv_wtype == GGML_TYPE_Q8_0 && key.flags == 67) ? 76 :
            (qkv_wtype == GGML_TYPE_Q8_0 && qkv_q8_portable_route) ? 84 :
            (qkv_wtype == GGML_TYPE_Q5_0 && (key.flags == 60 || key.flags == 72)) ? 77 : 0;
        const bool qkv_type_ok =
            (qkv_combined_flag == 75 && qkv_wtype == GGML_TYPE_F16) ||
            ((qkv_combined_flag == 76 || qkv_combined_flag == 84) &&
             qkv_wtype == GGML_TYPE_Q8_0) ||
            (qkv_combined_flag == 77 && qkv_wtype == GGML_TYPE_Q5_0);
        if (qkv_shared_dispatch && !no_fusion && is_matvec_dispatch &&
            !fused_bias_add && !fused_mmv_glu_up && !fused_mmv_set_rows &&
            !use_dp4a && !use_dp4a_matvec &&
            node->op == GGML_OP_MUL_MAT && node->src[0] && node->src[1] &&
            qkv_type_ok && node->type == GGML_TYPE_F32 &&
            node->ne[1] == 1 && node->ne[2] == 1 && node->ne[3] == 1 &&
            (node->ne[0] % 2) == 0) {
            const int WINDOW = 24;
            struct ggml_tensor * activation = node->src[1];
            const int64_t  K_dim = node->src[0]->ne[0];
            const uint32_t wnb1  = (uint32_t)node->src[0]->nb[1];
            auto find_rope = [&](int mm_idx) -> int {
                struct ggml_tensor * mm = cgraph->nodes[mm_idx];
                for (int k = mm_idx + 1; k < cgraph->n_nodes && k <= mm_idx + WINDOW; ++k) {
                    struct ggml_tensor * c = cgraph->nodes[k];
                    if (!c || c->op != GGML_OP_ROPE) continue;
                    struct ggml_tensor * s = c->src[0]; int hops = 0;
                    while (s && s != mm && hops < 5 &&
                           (s->op == GGML_OP_RESHAPE || s->op == GGML_OP_VIEW)) { s = s->src[0]; ++hops; }
                    if (s == mm) return k;
                }
                return -1;
            };
            auto find_set_rows = [&](int src_idx) -> int {
                struct ggml_tensor * src = cgraph->nodes[src_idx];
                for (int k = src_idx + 1; k < cgraph->n_nodes && k <= src_idx + WINDOW; ++k) {
                    struct ggml_tensor * c = cgraph->nodes[k];
                    if (!c || c->op != GGML_OP_SET_ROWS) continue;
                    struct ggml_tensor * s = c->src[0]; int hops = 0;
                    while (s && s != src && hops < 5 &&
                           (s->op == GGML_OP_RESHAPE || s->op == GGML_OP_VIEW)) { s = s->src[0]; ++hops; }
                    if (s == src) return k;
                }
                return -1;
            };
            auto rope_ok = [&](struct ggml_tensor * mm, struct ggml_tensor * rope) -> bool {
                const int32_t * rp = (const int32_t *)rope->op_params;
                struct ggml_tensor * pos = rope->src[1];
                struct ggml_tensor * ff  = rope->src[2];
                if (rp[2] != 0) return false;                        // NORMAL only
                if (!(rp[1] > 0 && (rp[1] % 2) == 0)) return false;  // n_dims even
                if (rope->type != GGML_TYPE_F32) return false;
                if (!(rope->src[0] && rope->src[0]->ne[2] == 1 && rope->src[0]->ne[3] == 1)) return false;
                if (!(rope->ne[0] > 0 && (rope->ne[0] % 2) == 0)) return false;
                if ((int64_t)mm->ne[0] != rope->ne[0] * rope->ne[1]) return false;
                if (!(pos && pos->type == GGML_TYPE_I32 && dx12_get_resource(pos))) return false;
                if (!(ff == nullptr || dx12_get_resource(ff))) return false;
                const int32_t * sec = rp + 11;
                if (sec[0] || sec[1] || sec[2] || sec[3]) return false;  // no mrope sections
                return true;
            };
            int q_rope_idx = find_rope(i);
            if (q_rope_idx >= 0 && rope_ok(node, cgraph->nodes[q_rope_idx]) &&
                find_set_rows(q_rope_idx) < 0) {
                struct ggml_tensor * q_rope = cgraph->nodes[q_rope_idx];
                int v_mm_idx = -1, v_sr_idx = -1;
                int k_mm_idx = -1, k_rope_i = -1, k_sr_idx = -1;
                for (int k = i + 1; k < cgraph->n_nodes && k <= i + WINDOW; ++k) {
                    struct ggml_tensor * mm = cgraph->nodes[k];
                    if (!mm || mm->op != GGML_OP_MUL_MAT || mm->src[1] != activation) continue;
                    if (!mm->src[0]) continue;
                    const bool mixed_v_type =
                        qkv_mixed_q5_q8 && qkv_combined_flag == 77 &&
                        mm->src[0]->type == GGML_TYPE_Q8_0;
                    if (mm->src[0]->type != qkv_wtype && !mixed_v_type) continue;
                    if (mm->type != GGML_TYPE_F32) continue;
                    if (mm->ne[1] != 1 || mm->ne[2] != 1 || mm->ne[3] != 1) continue;
                    if (mm->src[0]->ne[0] != K_dim) continue;
                    if ((mm->ne[0] % 2) != 0) continue;
                    int rr = find_rope(k);
                    if (rr >= 0) {
                        if (mm->src[0]->type != qkv_wtype ||
                            (uint32_t)mm->src[0]->nb[1] != wnb1) continue;
                        if (k_mm_idx < 0 && rope_ok(mm, cgraph->nodes[rr])) {
                            int sr = find_set_rows(rr);
                            if (sr >= 0) { k_mm_idx = k; k_rope_i = rr; k_sr_idx = sr; }
                        }
                    } else if (v_mm_idx < 0) {
                        if (mm->src[0]->type == qkv_wtype &&
                            (uint32_t)mm->src[0]->nb[1] != wnb1) continue;
                        int sr = find_set_rows(k);
                        if (sr >= 0) { v_mm_idx = k; v_sr_idx = sr; }
                    }
                }
                if (v_mm_idx >= 0 && k_mm_idx >= 0) {
                    struct ggml_tensor * v_mm   = cgraph->nodes[v_mm_idx];
                    struct ggml_tensor * k_mm   = cgraph->nodes[k_mm_idx];
                    struct ggml_tensor * v_sr   = cgraph->nodes[v_sr_idx];
                    struct ggml_tensor * k_sr   = cgraph->nodes[k_sr_idx];
                    struct ggml_tensor * k_rope = cgraph->nodes[k_rope_i];
                    const int64_t q_rows   = node->ne[0];
                    const int64_t k_rows   = k_mm->ne[0];
                    const int64_t v_rows   = v_mm->ne[0];
                    const int64_t head_dim = q_rope->ne[0];
                    const uint64_t wq_off = dx12_tensor_offset(node->src[0]);
                    const uint64_t wk_off = dx12_tensor_offset(k_mm->src[0]);
                    const uint64_t wv_off = dx12_tensor_offset(v_mm->src[0]);
                    const bool homogeneous =
                        k_mm->src[0]->type == qkv_wtype && v_mm->src[0]->type == qkv_wtype;
                    const bool mixed_q5_q8v =
                        qkv_wtype == GGML_TYPE_Q5_0 &&
                        k_mm->src[0]->type == GGML_TYPE_Q5_0 &&
                        v_mm->src[0]->type == GGML_TYPE_Q8_0;
                    ID3D12Resource * w_res = dx12_get_resource(node->src[0]);
                    ID3D12Resource * wk_res = dx12_get_resource(k_mm->src[0]);
                    ID3D12Resource * wv_res = dx12_get_resource(v_mm->src[0]);
                    bool weights_layout_ok = w_res && wk_res == w_res &&
                        wk_off == wq_off + (uint64_t)q_rows * wnb1 &&
                        ((homogeneous && wv_res == w_res &&
                          wv_off == wk_off + (uint64_t)k_rows * wnb1) ||
                         (mixed_q5_q8v && wv_res));
                    ID3D12Resource * kc_res = dx12_get_resource(k_sr);
                    ID3D12Resource * vc_res = dx12_get_resource(v_sr);
                    bool cache_shared = kc_res && kc_res == vc_res;
                    bool head_aligned = head_dim > 0 &&
                        (q_rows % head_dim) == 0 && (k_rows % head_dim) == 0 &&
                        (v_rows % head_dim) == 0;
                    bool cache_ok =
                        (k_sr->type == GGML_TYPE_F16 || k_sr->type == GGML_TYPE_F32) &&
                        k_sr->type == v_sr->type &&
                        (uint32_t)k_sr->nb[0] == (uint32_t)ggml_type_size(k_sr->type) &&
                        (uint32_t)v_sr->nb[0] == (uint32_t)ggml_type_size(v_sr->type) &&
                        (uint32_t)k_sr->nb[1] == (uint32_t)v_sr->nb[1];
                    struct ggml_tensor * k_idx = k_sr->src[1];
                    struct ggml_tensor * v_idx = v_sr->src[1];
                    bool idx_ok = k_idx && v_idx &&
                        (k_idx->type == GGML_TYPE_I32 || k_idx->type == GGML_TYPE_I64) &&
                        (v_idx->type == GGML_TYPE_I32 || v_idx->type == GGML_TYPE_I64) &&
                        dx12_get_resource(k_idx) && dx12_get_resource(v_idx) &&
                        dx12_get_resource(k_sr) && dx12_get_resource(v_sr) &&
                        dx12_get_resource(q_rope);
                    bool ff_ok = (q_rope->src[2] == k_rope->src[2]);
                    bool chains_ok =
                        dx12_mmv_rope_chain_exclusive(cgraph, i, q_rope_idx) &&
                        dx12_mmv_rope_chain_exclusive(cgraph, k_mm_idx, k_sr_idx) &&
                        dx12_mmv_scatter_chain_exclusive(cgraph, v_mm_idx, v_sr_idx);
                    if ((homogeneous || mixed_q5_q8v) &&
                        weights_layout_ok && cache_shared && head_aligned && cache_ok &&
                        idx_ok && ff_ok && chains_ok &&
                        (q_rows + k_rows + v_rows) < 65535) {
                        fused_qkv                = true;
                        fused_qkv_q_rope         = q_rope; fused_qkv_q_rope_idx     = q_rope_idx;
                        fused_qkv_v_matvec       = v_mm;   fused_qkv_v_matvec_idx   = v_mm_idx;
                        fused_qkv_v_set_rows     = v_sr;   fused_qkv_v_set_rows_idx = v_sr_idx;
                        fused_qkv_k_matvec       = k_mm;   fused_qkv_k_matvec_idx   = k_mm_idx;
                        fused_mmv_k_rope         = k_rope; fused_mmv_k_rope_idx     = k_rope_i;
                        fused_qkv_k_set_rows     = k_sr;   fused_qkv_k_set_rows_idx = k_sr_idx;
                        fused_qkv_q_rows = (uint32_t)q_rows;
                        fused_qkv_k_rows = (uint32_t)k_rows;
                        fused_qkv_v_rows = (uint32_t)v_rows;
                        key.flags = mixed_q5_q8v ? 79 : qkv_combined_flag;
                    }
                }
            }
        }

        // Q/K projection ROPE matvec fusion (DX12_MMV_{Q,K}_ROPE_FUSION): a
        // standalone M=1 Q/K matvec on a retained AMD decode shader (F16 fl=63,
        // Q8_0 fl=67, Q5_0 fl=72) applies RoPE to its output inline. Q writes
        // the rotated result to the ROPE output (absorbing the standalone ROPE);
        // K scatters the rotated result into the KV cache (absorbing the fused
        // ROPE+VIEW+SET_ROWS fl=6 dispatch). NORMAL mode only: the (row0,row1)
        // matvec output pair maps 1:1 to a NORMAL rotation pair. Structural
        // detection only — never tensor names.
        if ((mmv_q_rope_fusion || mmv_k_rope_fusion) && !no_fusion && is_matvec_dispatch &&
            !fused_qkv &&
            !fused_bias_add && !fused_mmv_glu_up && !fused_mmv_set_rows &&
            !use_dp4a && !use_dp4a_matvec &&
            node->op == GGML_OP_MUL_MAT && node->src[0] && node->src[1] &&
            node->type == GGML_TYPE_F32 &&
            node->ne[1] == 1 && node->ne[2] == 1 && node->ne[3] == 1 &&
            (node->ne[0] % 2) == 0 &&
            (key.flags == 63 || key.flags == 67 || key.flags == 72)) {
            const int WINDOW = 16;
            // Find the ROPE whose RESHAPE/VIEW chain leads back to this matvec.
            int rope_idx = -1;
            for (int k = i + 1; k < cgraph->n_nodes && k <= i + WINDOW; ++k) {
                struct ggml_tensor * cand = cgraph->nodes[k];
                if (!cand || cand->op != GGML_OP_ROPE) continue;
                struct ggml_tensor * s = cand->src[0];
                int hops = 0;
                while (s && s != node && hops < 5 &&
                       (s->op == GGML_OP_RESHAPE || s->op == GGML_OP_VIEW)) {
                    s = s->src[0];
                    hops++;
                }
                if (s == node) { rope_idx = k; break; }
            }
            if (rope_idx >= 0) {
                struct ggml_tensor * rope = cgraph->nodes[rope_idx];
                const int32_t * rp = (const int32_t *)rope->op_params;
                const int32_t rope_mode = rp[2];
                const int32_t n_dims    = rp[1];
                struct ggml_tensor * pos = rope->src[1];
                struct ggml_tensor * ff  = rope->src[2];  // freq_factors (optional)
                bool rope_ok =
                    rope_mode == 0 &&                            // NORMAL only
                    n_dims > 0 && (n_dims % 2) == 0 &&
                    rope->type == GGML_TYPE_F32 &&
                    rope->src[0] && rope->src[0]->ne[2] == 1 && rope->src[0]->ne[3] == 1 &&
                    rope->ne[0] > 0 && (rope->ne[0] % 2) == 0 &&
                    (int64_t)node->ne[0] == rope->ne[0] * rope->ne[1] &&
                    pos && pos->type == GGML_TYPE_I32 && dx12_get_resource(pos) &&
                    (ff == nullptr || dx12_get_resource(ff));
                if (rope_ok) {
                    // mrope sections must be zero (NORMAL carries none; guard).
                    const int32_t * sec = rp + 11;
                    if (sec[0] || sec[1] || sec[2] || sec[3]) rope_ok = false;
                }
                if (rope_ok) {
                    // Classify K (rope -> VIEW -> SET_ROWS) vs Q (no SET_ROWS).
                    int sr_idx = -1;
                    for (int k = rope_idx + 1; k < cgraph->n_nodes && k <= rope_idx + WINDOW; ++k) {
                        struct ggml_tensor * cand = cgraph->nodes[k];
                        if (!cand || cand->op != GGML_OP_SET_ROWS) continue;
                        struct ggml_tensor * s = cand->src[0];
                        int hops = 0;
                        while (s && s != rope && hops < 5 &&
                               (s->op == GGML_OP_RESHAPE || s->op == GGML_OP_VIEW)) {
                            s = s->src[0];
                            hops++;
                        }
                        if (s == rope) { sr_idx = k; break; }
                    }
                    if (sr_idx >= 0 && mmv_k_rope_fusion) {
                        struct ggml_tensor * sr     = cgraph->nodes[sr_idx];
                        struct ggml_tensor * srview = sr->src[0];
                        struct ggml_tensor * kidx   = sr->src[1];
                        bool ok = kidx && srview &&
                                  (sr->type == GGML_TYPE_F16 || sr->type == GGML_TYPE_F32) &&
                                  (kidx->type == GGML_TYPE_I32 || kidx->type == GGML_TYPE_I64) &&
                                  sr->ne[2] == 1 && sr->ne[3] == 1 &&
                                  (uint32_t)sr->nb[0] == (uint32_t)ggml_type_size(sr->type) &&
                                  ggml_is_contiguous(srview) &&
                                  srview->ne[0] == node->ne[0] &&   // flattened feature dim
                                  srview->ne[1] == 1 &&             // single token (M=1)
                                  dx12_get_resource(sr) && dx12_get_resource(kidx);
                        if (ok) {
                            ok = dx12_mmv_rope_chain_exclusive(cgraph, i, sr_idx);
                        }
                        if (ok) {
                            fused_mmv_k_rope         = rope;
                            fused_mmv_k_rope_idx     = rope_idx;
                            fused_mmv_k_set_rows     = sr;
                            fused_mmv_k_set_rows_idx = sr_idx;
                        }
                    } else if (sr_idx < 0 && mmv_q_rope_fusion) {
                        bool ok = ggml_is_contiguous(rope) && dx12_get_resource(rope);
                        if (ok) {
                            ok = dx12_mmv_rope_chain_exclusive(cgraph, i, rope_idx);
                        }
                        if (ok) {
                            fused_mmv_q_rope     = rope;
                            fused_mmv_q_rope_idx = rope_idx;
                        }
                    }
                }
            }
        }

        // V-cache SET_ROWS matvec fusion (DX12_MMV_SET_ROWS_FUSION): a
        // standalone M=1 V-projection matvec on one of the retained AMD wave64
        // decode shaders (F16 fl=63, Q8_0 fl=67, Q5_0 fl=72) writes its result
        // straight into the scattered KV cache slot, eliminating the later V
        // SET_ROWS dispatch. The chain is not contiguous: K-projection nodes
        // (and the K ROPE+SET_ROWS fusion) sit between the V matvec and the V
        // SET_ROWS, so only the SET_ROWS node is absorbed (the RESHAPE/VIEW in
        // between are already view-skips).
        if (mmv_set_rows_fusion && !no_fusion && is_matvec_dispatch &&
            !fused_bias_add && !fused_mmv_glu_up && !use_dp4a && !use_dp4a_matvec &&
            !fused_qkv && !fused_mmv_q_rope && !fused_mmv_k_set_rows &&
            node->op == GGML_OP_MUL_MAT && node->src[0] && node->src[1] &&
            node->type == GGML_TYPE_F32 &&
            node->ne[1] == 1 && node->ne[2] == 1 && node->ne[3] == 1 &&
            (key.flags == 63 || key.flags == 67 || key.flags == 72)) {
            const int WINDOW = 16;
            int sr_idx = -1;
            for (int k = i + 1; k < cgraph->n_nodes && k <= i + WINDOW; ++k) {
                struct ggml_tensor * cand = cgraph->nodes[k];
                if (!cand || cand->op != GGML_OP_SET_ROWS) continue;
                struct ggml_tensor * s = cand->src[0];
                int hops = 0;
                while (s && s != node && hops < 5 &&
                       (s->op == GGML_OP_RESHAPE || s->op == GGML_OP_VIEW)) {
                    s = s->src[0];
                    hops++;
                }
                if (s == node) { sr_idx = k; break; }
            }
            if (sr_idx >= 0) {
                struct ggml_tensor * sr     = cgraph->nodes[sr_idx];
                struct ggml_tensor * srview = sr->src[0];
                struct ggml_tensor * idx    = sr->src[1];
                bool ok = idx && srview &&
                          (sr->type == GGML_TYPE_F16 || sr->type == GGML_TYPE_F32) &&
                          (idx->type == GGML_TYPE_I32 || idx->type == GGML_TYPE_I64) &&
                          sr->ne[2] == 1 && sr->ne[3] == 1 &&
                          (uint32_t)sr->nb[0] == (uint32_t)ggml_type_size(sr->type) &&
                          ggml_is_contiguous(srview) &&
                          srview->ne[0] == node->ne[0] &&   // flattened feature dim
                          srview->ne[1] == 1 &&             // single token (M=1)
                          dx12_get_resource(sr) && dx12_get_resource(idx);
                if (ok) {
                    ok = dx12_mmv_scatter_chain_exclusive(cgraph, i, sr_idx);
                }
                if (ok) {
                    fused_mmv_set_rows     = sr;
                    fused_mmv_set_rows_idx = sr_idx;
                }
            }
        }

        // End of record-path decision block.  Look up pipeline and store the
        // decision into the replay cache so subsequent tokens can fast-path.
        if (g_dx12_flag_sink && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) {
            g_dx12_flag_sink->push_back((uint32_t)key.flags);
        }
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
            if (fused_qk_postop)                   d.fusion_kind = DX12_FUSE_QK_ROPE_SCALE_SET_ROWS;
            else if (fused_add_rms_node)           d.fusion_kind = DX12_FUSE_ADD_RMS_MUL;
            else if (fused_5way_set_rows)          d.fusion_kind = DX12_FUSE_RMS_MUL_ROPE5;
            else if (fused_rope_after_rms)         d.fusion_kind = DX12_FUSE_RMS_MUL_ROPE3;
            else if (fused_rms_quant_consumer)     d.fusion_kind = DX12_FUSE_RMS_MUL_QUANT_Q8_1;
            else if (fused_mul_node)               d.fusion_kind = DX12_FUSE_RMS_MUL;
            else if (fused_rope_set_rows)          d.fusion_kind = DX12_FUSE_ROPE_SET_ROWS;
            else if (fused_qkv)                    d.fusion_kind = DX12_FUSE_MMV_QKV_SHARED;
            else if (fused_mmv_glu_up)             d.fusion_kind = DX12_FUSE_MMV_GLU_SPLIT;
            else if (fused_mmv_set_rows)           d.fusion_kind = DX12_FUSE_MMV_SET_ROWS;
            else if (fused_mmv_k_set_rows)         d.fusion_kind = DX12_FUSE_MMV_K_ROPE_SET_ROWS;
            else if (fused_mmv_q_rope)             d.fusion_kind = DX12_FUSE_MMV_Q_ROPE;
            else if (fused_ssm_bias)               d.fusion_kind = DX12_FUSE_SSM_CONV_BIAS_SILU;
            else if (fused_ssm_silu)               d.fusion_kind = DX12_FUSE_SSM_CONV_SILU;
            else                                    d.fusion_kind = DX12_FUSE_NONE;
            // skip_count derived from fusion_kind (matches the `i += N` block below).
            switch (d.fusion_kind) {
                case DX12_FUSE_ADD_RMS_MUL:        d.skip_count = 2; break;
                case DX12_FUSE_RMS_MUL_ROPE5:      d.skip_count = 4; break;
                case DX12_FUSE_RMS_MUL_ROPE3:      d.skip_count = 2; break;
                case DX12_FUSE_RMS_MUL:            d.skip_count = 1; break;
                case DX12_FUSE_RMS_MUL_QUANT_Q8_1: d.skip_count = 1; break;  // only MUL absorbed, consumer matmul stays
                case DX12_FUSE_ROPE_SET_ROWS:      d.skip_count = 2; break;
                case DX12_FUSE_MMV_GLU_SPLIT:      d.skip_count = 2; break;
                case DX12_FUSE_SSM_CONV_SILU:      d.skip_count = 1; break;
                case DX12_FUSE_SSM_CONV_BIAS_SILU: d.skip_count = 2; break;
                case DX12_FUSE_QK_ROPE_SCALE_SET_ROWS: d.skip_count = 1; break;
                default:                            d.skip_count = 0; break;  // incl. MMV_SET_ROWS (non-contiguous)
            }
            // MMV_SET_ROWS absorbs a non-adjacent SET_ROWS via node_absorbed[]
            // (not skip_count). Record its relative index for the replay path.
            d.mmv_set_rows_rel = (fused_mmv_set_rows_idx >= 0)
                               ? (int16_t)(fused_mmv_set_rows_idx - i) : 0;
            // MMV_Q_ROPE / MMV_K_ROPE_SET_ROWS absorb a non-adjacent ROPE (and,
            // for K, its SET_ROWS) via node_absorbed[]. Record relative indices.
            d.mmv_rope_rel = (fused_mmv_q_rope_idx >= 0) ? (int16_t)(fused_mmv_q_rope_idx - i)
                           : (fused_mmv_k_rope_idx >= 0) ? (int16_t)(fused_mmv_k_rope_idx - i)
                           : 0;
            d.mmv_rope_set_rows_rel = (fused_mmv_k_set_rows_idx >= 0)
                                    ? (int16_t)(fused_mmv_k_set_rows_idx - i) : 0;
            // MMV_QKV_SHARED absorbs the V and K matvecs plus all three post-op
            // endpoints (non-contiguous) via node_absorbed[]. Record the relative
            // indices so the replay fast path reconstructs every fused handle.
            if (fused_qkv) {
                d.qkv_q_rope_rel     = (int16_t)(fused_qkv_q_rope_idx     - i);
                d.qkv_v_matvec_rel   = (fused_qkv_v_matvec_idx >= 0)
                                     ? (int16_t)(fused_qkv_v_matvec_idx - i) : 0;
                d.qkv_v_set_rows_rel = (int16_t)(fused_qkv_v_set_rows_idx - i);
                d.qkv_k_matvec_rel   = (fused_qkv_k_matvec_idx >= 0)
                                     ? (int16_t)(fused_qkv_k_matvec_idx - i) : 0;
                d.qkv_k_rope_rel     = (int16_t)(fused_mmv_k_rope_idx     - i);
                d.qkv_k_set_rows_rel = (int16_t)(fused_qkv_k_set_rows_idx - i);
            }
            if (fused_qk_postop) {
                d.qk_scale_rel      = (int16_t)(fused_qk_scale_idx      - i);
                d.qk_k_rope_rel     = (int16_t)(fused_qk_k_rope_idx     - i);
                d.qk_k_set_rows_rel = (int16_t)(fused_qk_k_set_rows_idx - i);
            }
            d.fusion_skip_f32 = fused_rms_quant_skip_f32;
            if (fused_bias_add) {
                d.has_bias_add = true;
                d.skip_count  += 1;
            }
        }
        } // end of `if (replay) { ... } else { ... }`

        // Mark the absorbed SET_ROWS so the loop skips its dispatch when it is
        // reached (both record and replay). The matvec below writes the cache.
        if (fused_mmv_set_rows_idx >= 0 && mmv_set_rows_fusion &&
            fused_mmv_set_rows_idx < cgraph->n_nodes) {
            node_absorbed[fused_mmv_set_rows_idx] = 1;
        }
        // Mark the absorbed ROPE (Q and K) and SET_ROWS (K) so their dispatches
        // are skipped when reached. Skipping the ROPE also stops the standalone
        // fl=6 ROPE+SET_ROWS fusion from firing at that node. The intermediate
        // VIEW (K) self-skips as an ordinary view node.
        if (mmv_any_postop) {
            if (fused_mmv_q_rope_idx >= 0 && fused_mmv_q_rope_idx < cgraph->n_nodes) {
                node_absorbed[fused_mmv_q_rope_idx] = 1;
            }
            if (fused_mmv_k_rope_idx >= 0 && fused_mmv_k_rope_idx < cgraph->n_nodes) {
                node_absorbed[fused_mmv_k_rope_idx] = 1;
            }
            if (fused_mmv_k_set_rows_idx >= 0 && fused_mmv_k_set_rows_idx < cgraph->n_nodes) {
                node_absorbed[fused_mmv_k_set_rows_idx] = 1;
            }
        }
        // MMV_QKV_SHARED: mark the V and K matvecs plus all three post-op
        // endpoints absorbed so their dispatches are skipped; the combined
        // dispatch recorded at this Q matvec produces all of them. The
        // intermediate RESHAPE/VIEW nodes self-skip as ordinary view nodes.
        if (fused_qkv) {
            const int idxs[6] = {
                fused_qkv_q_rope_idx, fused_qkv_v_matvec_idx, fused_qkv_v_set_rows_idx,
                fused_qkv_k_matvec_idx, fused_mmv_k_rope_idx, fused_qkv_k_set_rows_idx };
            for (int a = 0; a < 6; ++a) {
                if (idxs[a] > i && idxs[a] < cgraph->n_nodes) node_absorbed[idxs[a]] = 1;
            }
        }
        if (fused_qk_postop) {
            if (fused_qk_k_rope_idx > i && fused_qk_k_rope_idx < cgraph->n_nodes) {
                node_absorbed[fused_qk_k_rope_idx] = 1;
            }
            if (fused_qk_k_set_rows_idx > i &&
                fused_qk_k_set_rows_idx < cgraph->n_nodes) {
                node_absorbed[fused_qk_k_set_rows_idx] = 1;
            }
        }
        if (phase_profile) {
            uint64_t now = dx12_qpc_us();
            bctx->phase_decision_us += now - phase_detail_start_us;
            phase_detail_start_us = now;
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
            params.op_params[10] = (uint32_t)arm_wt->ne[0];  // for dim0 broadcast modulo
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
                // v3 pivot 1: RMS+Q8_1 fusion uses op_params[1] as the
                // skip_f32 flag. RMS_NORM's tensor->op_params only owns slot 0
                // (eps); the rest carries undefined bytes which the v2 shader
                // ignored. The v3 shader READS op_params[1], so we MUST
                // explicitly initialize it (and only set to 1 when the safety
                // gate proved all downstream consumers go through the Q8_1
                // scratch cache).
                params.op_params[1] = 0u;
                if (fused_rms_quant_consumer && fused_rms_quant_skip_f32) {
                    params.op_params[1] = 1u;
                }
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

        // Fused SSM_CONV + (optional ADD) + UNARY(SILU): redirect dst to the SILU
        // node's tensor (shape is identical to the SSM_CONV output, only the
        // buffer/offset differs).  The fused shader reads bias from src2 (bound
        // below) and applies bias + silu inline, so op_params don't need to
        // change.  The plain SSM_CONV op currently consumes no op_params slots.
        if (fused_ssm_silu) {
            params.ne0 = (uint32_t)fused_ssm_silu->ne[0]; params.ne1 = (uint32_t)fused_ssm_silu->ne[1];
            params.ne2 = (uint32_t)fused_ssm_silu->ne[2]; params.ne3 = (uint32_t)fused_ssm_silu->ne[3];
            params.nb0 = (uint32_t)fused_ssm_silu->nb[0]; params.nb1 = (uint32_t)fused_ssm_silu->nb[1];
            params.nb2 = (uint32_t)fused_ssm_silu->nb[2]; params.nb3 = (uint32_t)fused_ssm_silu->nb[3];
            params.dst_offset = (uint32_t)dx12_tensor_offset(fused_ssm_silu);
            params.dst_esize  = (uint32_t)ggml_type_size(fused_ssm_silu->type);
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

        if (fused_qk_postop) {
            dx12_pack_rope_op_params(node, fused_qk_k_set_rows,
                                     dx12_rope_pack_kind::ROPE_SET_ROWS,
                                     0.0f, params);
            float scale = 0.0f;
            memcpy(&scale, fused_qk_scale->op_params, sizeof(float));
            memcpy(&params.op_params[8], &scale, sizeof(uint32_t));
            params.op_params[9]  = (uint32_t)dx12_tensor_offset(fused_qk_k_set_rows);
            params.op_params[10] = (uint32_t)fused_qk_k_set_rows->nb[0];
            params.op_params[11] = (uint32_t)fused_qk_k_set_rows->nb[1];
            params.op_params[12] = (uint32_t)ggml_type_size(fused_qk_k_set_rows->type);
            params.ne0 = (uint32_t)fused_qk_scale->ne[0];
            params.ne1 = (uint32_t)fused_qk_scale->ne[1];
            params.ne2 = (uint32_t)fused_qk_scale->ne[2];
            params.ne3 = (uint32_t)fused_qk_scale->ne[3];
            params.nb0 = (uint32_t)fused_qk_scale->nb[0];
            params.nb1 = (uint32_t)fused_qk_scale->nb[1];
            params.nb2 = (uint32_t)fused_qk_scale->nb[2];
            params.nb3 = (uint32_t)fused_qk_scale->nb[3];
            params.dst_offset = (uint32_t)dx12_tensor_offset(fused_qk_scale);
            params.dst_esize  = (uint32_t)ggml_type_size(fused_qk_scale->type);
        }

        // R9 fused MMV+GLU: override dst to SWIGLU output, encode W_up offset
        // in op_params[1] (mirrors fused_bias_tensor's slot-1 encoding pattern).
        if (fused_mmv_glu_glu) {
            params.op_params[1] = (uint32_t)dx12_tensor_offset(fused_mmv_glu_up->src[0]);
            params.dst_offset   = (uint32_t)dx12_tensor_offset(fused_mmv_glu_glu);
        }

        // Fused V-cache SET_ROWS matvec: the matvec writes its output row
        // directly into the scattered KV cache slot instead of the plain dst.
        // dst is bound to the SET_ROWS output (KV cache) and the row-index
        // buffer to src2/t2; op7..op13 carry the direct-scatter metadata (see
        // mmv_store_scatter in ggml_common.hlsli). The plain-matvec dst_* fields
        // are left untouched — the shader ignores them when op7 == 1.
        if (fused_mmv_set_rows) {
            const struct ggml_tensor * sr  = fused_mmv_set_rows;
            const struct ggml_tensor * idx = sr->src[1];
            params.op_params[7]  = 1u;                                       // direct-scatter mode
            params.op_params[8]  = (uint32_t)dx12_tensor_offset(sr);         // cache base byte offset
            params.op_params[9]  = (uint32_t)sr->nb[0];                      // cache element stride
            params.op_params[10] = (uint32_t)sr->nb[1];                      // cache row stride
            params.op_params[11] = (uint32_t)dx12_tensor_offset(idx);        // index byte offset
            params.op_params[12] = (uint32_t)idx->nb[0];                     // index token stride
            params.op_params[13] = (uint32_t)ggml_type_size(sr->type);       // cache esize (2=F16, 4=F32)
        }

        // Fused Q-projection ROPE matvec: the matvec rotates its output pair and
        // writes the ROPE result directly (dst bound to the ROPE output). op7==2
        // selects Q-ROPE mode; op1..op11 carry the RoPE metadata; op12 = rope-out
        // element stride. Positions ride src2, freq_factors src4. See
        // mmv_rope_pair / mmv_rope_store in the shader headers.
        if (fused_mmv_q_rope) {
            const struct ggml_tensor * rope = fused_mmv_q_rope;
            dx12_pack_mmv_rope_op_params(rope, params);
            params.op_params[7]  = 2u;                                   // Q-ROPE mode
            params.op_params[12] = (uint32_t)rope->nb[0];               // rope-out element stride
            params.dst_offset    = (uint32_t)dx12_tensor_offset(rope);  // rope-out base
            params.dst_esize     = (uint32_t)ggml_type_size(rope->type);
        }

        // Fused K-projection ROPE+SET_ROWS matvec: the matvec rotates its output
        // pair and scatters the result straight into the KV cache (dst bound to
        // the SET_ROWS output). op7==3 selects K-ROPE-scatter mode; op1..op11
        // carry the RoPE metadata; op12/op13/op14 = cache element stride / row
        // stride / esize. Positions ride src2, SET_ROWS indices src3,
        // freq_factors src4.
        if (fused_mmv_k_set_rows) {
            const struct ggml_tensor * rope = fused_mmv_k_rope;
            const struct ggml_tensor * sr   = fused_mmv_k_set_rows;
            dx12_pack_mmv_rope_op_params(rope, params);
            params.op_params[7]  = 3u;                                   // K-ROPE-scatter mode
            params.op_params[12] = (uint32_t)sr->nb[0];                 // cache element stride
            params.op_params[13] = (uint32_t)sr->nb[1];                 // cache row stride
            params.op_params[14] = (uint32_t)ggml_type_size(sr->type);  // cache esize
            params.dst_offset    = (uint32_t)dx12_tensor_offset(sr);    // cache base
            params.dst_esize     = (uint32_t)ggml_type_size(sr->type);
        }

        if (is_matvec_dispatch) {
            params.op_params[15] = 0;
        }

        // Opt-in F16 rope-pair de-dup (fl=63 only): tag the fused Q/K rope
        // matvec so one group owns the full rotation pair. Set here (after op7's
        // Q/K mode) and consumed at the dispatch group-count below.
        const bool rope_rows2 = f16_rope_rows2 && key.flags == 63 &&
                                (fused_mmv_q_rope || fused_mmv_k_set_rows);
        if (rope_rows2) {
            params.op_params[7] |= 4u;
        }

        // Combined Q/K/V projection dispatch params. dx12_fill_params(node) set
        // the K dim (ne00), weight row stride (nb01), Wq base (src0_offset) and
        // the shared activation (src1_offset). The three contiguous weight
        // matrices (Wq|Wk|Wv) are addressed by the global output row, so ne0
        // becomes the total row count (dispatch guard). Rope metadata (shared by
        // Q and K) is packed from the Q ROPE; positions ride src2, freq_factors
        // src4, K/V indices src3/src5, KV cache u1. See the op-param map in
        // mul_mat_vec_qkv_f16_wave64.hlsl.
        if (fused_qkv) {
            const struct ggml_tensor * q_rope = fused_qkv_q_rope;
            const struct ggml_tensor * q_out  = q_rope;
            const struct ggml_tensor * k_sr   = fused_qkv_k_set_rows;
            const struct ggml_tensor * v_sr   = fused_qkv_v_set_rows;
            const uint32_t q_rows = fused_qkv_q_rows;
            const uint32_t k_rows = fused_qkv_k_rows;
            const uint32_t v_rows = fused_qkv_v_rows;
            dx12_pack_mmv_rope_op_params(q_rope, params);            // op1,3,4,5,6,8,9,10,11 (rope)
            params.op_params[0]  = q_rows;                          // Q region row count
            params.op_params[2]  = k_rows;                          // K region row count
            params.op_params[7]  = (uint32_t)ggml_type_size(k_sr->type);  // KV cache esize (== nb0)
            params.op_params[12] = (uint32_t)q_out->nb[0];          // Q output element stride
            params.op_params[13] = (uint32_t)dx12_tensor_offset(k_sr);    // K cache base byte offset
            params.op_params[14] = (uint32_t)dx12_tensor_offset(v_sr);    // V cache base byte offset
            params.op_params[15] = (uint32_t)k_sr->nb[1];           // KV cache row stride (nb1)
            if (key.flags == 79) {
                params.nb02 = (uint32_t)fused_qkv_v_matvec->src[0]->nb[1];
            }
            params.ne0        = q_rows + k_rows + v_rows;           // total rows (dispatch guard)
            params.dst_offset = (uint32_t)dx12_tensor_offset(q_out);      // Q output base (u0 at res base)
            params.dst_esize  = (uint32_t)ggml_type_size(q_out->type);
        }
        static constexpr uint32_t BASE_PARAMS = 30;  // ne/nb/offsets/esizes = 30 DWORDs
        bool needs_op_params = (node->op == GGML_OP_SOFT_MAX || 
                                 node->op == GGML_OP_FLASH_ATTN_EXT || 
                                 node->op == GGML_OP_ROPE ||
                                 node->op == GGML_OP_RMS_NORM ||
                                 node->op == GGML_OP_NORM ||
                                 node->op == GGML_OP_L2_NORM ||
                                 node->op == GGML_OP_GATED_DELTA_NET ||
                                 node->op == GGML_OP_SSM_SCAN ||
                                 node->op == GGML_OP_RWKV_WKV6 ||
                                 node->op == GGML_OP_RWKV_WKV7 ||
                                 node->op == GGML_OP_GROUP_NORM ||
                                 node->op == GGML_OP_GLU ||
                                 node->op == GGML_OP_SCALE ||
                                 node->op == GGML_OP_CLAMP ||
                                 node->op == GGML_OP_UPSCALE ||
                                 node->op == GGML_OP_IM2COL ||
                                 node->op == GGML_OP_IM2COL_3D ||
                                 node->op == GGML_OP_POOL_2D ||
                                 node->op == GGML_OP_POOL_1D ||
                                 node->op == GGML_OP_PAD ||
                                 node->op == GGML_OP_ROLL ||
                                 node->op == GGML_OP_CONV_2D ||
                                 node->op == GGML_OP_CONV_2D_DW ||
                                 node->op == GGML_OP_CONV_3D ||
                                 node->op == GGML_OP_CONV_TRANSPOSE_1D ||
                                 node->op == GGML_OP_CONV_TRANSPOSE_2D ||
                                 node->op == GGML_OP_CONCAT ||
                                 node->op == GGML_OP_MUL_MAT_ID ||
                                 node->op == GGML_OP_CPY ||
                                 node->op == GGML_OP_CONT ||
                                 node->op == GGML_OP_DUP ||
                                 (node->op == GGML_OP_UNARY &&
                                  ggml_get_unary_op(node) == GGML_UNARY_OP_XIELU) ||
                                 node->op == GGML_OP_LEAKY_RELU ||
                                 node->op == GGML_OP_FILL ||
                                 node->op == GGML_OP_TRI ||
                                 node->op == GGML_OP_ARANGE ||
                                 node->op == GGML_OP_TIMESTEP_EMBEDDING ||
                                 node->op == GGML_OP_ACC ||
                                 node->op == GGML_OP_SET ||
                                 node->op == GGML_OP_ARGSORT ||
                                 node->op == GGML_OP_TOP_K ||
                                 node->op == GGML_OP_ADD_ID ||
                                 node->op == GGML_OP_DIAG_MASK_INF ||
                                 fused_bias_tensor ||
                                 fused_add_rms_node ||
                                 fused_rope_set_rows ||
                                 fused_mmv_glu_up);
        uint32_t num_constants = (needs_op_params || is_matvec_dispatch)
                               ? (uint32_t)(sizeof(params) / 4) : BASE_PARAMS;
        // FLASH_ATTN_EXT re-uploads the full params block at line ~2425 after
        // computing n_splits + gqa_ratio (which are encoded into op_params[15]
        // and read by the shader).  Skipping the upload here saves one
        // 184-byte SetComputeRoot32BitConstants per attention block per token.
        if (node->op != GGML_OP_FLASH_ATTN_EXT) {
            bctx->set_shader_params(params, num_constants);
        }
        if (phase_profile) {
            uint64_t now = dx12_qpc_us();
            bctx->phase_params_us += now - phase_detail_start_us;
            phase_detail_start_us = now;
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
        if (fused_qk_postop) {
            dst_res = dx12_get_resource(fused_qk_scale);
        } else if (fused_5way_set_rows) {
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
        } else if (fused_ssm_silu) {
            dst_res = dx12_get_resource(fused_ssm_silu);
        } else if (fused_mmv_set_rows) {
            dst_res = dx12_get_resource(fused_mmv_set_rows);  // KV cache buffer
        } else if (fused_mmv_q_rope) {
            dst_res = dx12_get_resource(fused_mmv_q_rope);    // ROPE output buffer
        } else if (fused_mmv_k_set_rows) {
            dst_res = dx12_get_resource(fused_mmv_k_set_rows);  // KV cache buffer
        } else if (fused_qkv) {
            dst_res = dx12_get_resource(fused_qkv_q_rope);      // Q ROPE output buffer (u0)
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

        if (fused_qk_postop) {
            const ggml_tensor * k_input = fused_qk_k_rope->src[0];
            const ggml_tensor * k_idx   = fused_qk_k_set_rows->src[1];
            ID3D12Resource * k_res     = dx12_get_resource(k_input);
            ID3D12Resource * idx_res   = dx12_get_resource(k_idx);
            ID3D12Resource * cache_res = dx12_get_resource(fused_qk_k_set_rows);
            if (k_res) {
                D3D12_GPU_VIRTUAL_ADDRESS va =
                    k_res->GetGPUVirtualAddress() + dx12_tensor_offset(k_input);
                if (va != bctx->last_src3_va) {
                    bctx->cmd_list->SetComputeRootShaderResourceView(5, va);
                    bctx->last_src3_va = va;
                }
            }
            if (idx_res) {
                D3D12_GPU_VIRTUAL_ADDRESS va =
                    idx_res->GetGPUVirtualAddress() + dx12_tensor_offset(k_idx);
                if (va != bctx->last_src4_va) {
                    bctx->cmd_list->SetComputeRootShaderResourceView(7, va);
                    bctx->last_src4_va = va;
                }
            }
            if (cache_res) {
                bctx->cmd_list->SetComputeRootUnorderedAccessView(
                    6, cache_res->GetGPUVirtualAddress());
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

        // Combined Q/K/V projection dispatch bindings: positions (t2, read at
        // element 0), K indices (t3), freq_factors (t4, optional), V indices
        // (t5), and the shared KV cache (u1). Weights (t0), activation (t1) and
        // the Q ROPE output (u0) are bound above. Positions and freq_factors are
        // consumed by mmv_rope_pair; K/V indices by the region scatters.
        if (fused_qkv) {
            const struct ggml_tensor * pos   = fused_qkv_q_rope->src[1];
            const struct ggml_tensor * ff    = fused_qkv_q_rope->src[2];
            const struct ggml_tensor * k_idx = fused_qkv_k_set_rows->src[1];
            const struct ggml_tensor * v_idx = fused_qkv_v_set_rows->src[1];
            ID3D12Resource * pos_res  = dx12_get_resource(pos);
            ID3D12Resource * kidx_res = dx12_get_resource(k_idx);
            ID3D12Resource * vidx_res = dx12_get_resource(v_idx);
            ID3D12Resource * kv_res   = dx12_get_resource(fused_qkv_k_set_rows);
            if (pos_res) {
                D3D12_GPU_VIRTUAL_ADDRESS va = pos_res->GetGPUVirtualAddress() + dx12_tensor_offset(pos);
                if (va != bctx->last_src2_va) {
                    bctx->cmd_list->SetComputeRootShaderResourceView(4, va);
                    bctx->last_src2_va = va;
                }
            }
            if (kidx_res) {
                D3D12_GPU_VIRTUAL_ADDRESS va = kidx_res->GetGPUVirtualAddress() + dx12_tensor_offset(k_idx);
                if (va != bctx->last_src3_va) {
                    bctx->cmd_list->SetComputeRootShaderResourceView(5, va);
                    bctx->last_src3_va = va;
                }
            }
            if (ff) {
                ID3D12Resource * ff_res = dx12_get_resource(ff);
                if (ff_res) {
                    D3D12_GPU_VIRTUAL_ADDRESS va = ff_res->GetGPUVirtualAddress() + dx12_tensor_offset(ff);
                    if (va != bctx->last_src4_va) {
                        bctx->cmd_list->SetComputeRootShaderResourceView(7, va);
                        bctx->last_src4_va = va;
                    }
                }
            }
            if (vidx_res) {
                D3D12_GPU_VIRTUAL_ADDRESS va = vidx_res->GetGPUVirtualAddress() + dx12_tensor_offset(v_idx);
                if (va != bctx->last_src5_va) {
                    bctx->cmd_list->SetComputeRootShaderResourceView(8, va);
                    bctx->last_src5_va = va;
                }
            }
            if (kv_res) {
                bctx->cmd_list->SetComputeRootUnorderedAccessView(6, kv_res->GetGPUVirtualAddress());
            }
            if (key.flags == 79 && fused_qkv_v_matvec && fused_qkv_v_matvec->src[0]) {
                const struct ggml_tensor * v_weights = fused_qkv_v_matvec->src[0];
                ID3D12Resource * vw_res = dx12_get_resource(v_weights);
                if (vw_res) {
                    D3D12_GPU_VIRTUAL_ADDRESS va =
                        vw_res->GetGPUVirtualAddress() + dx12_tensor_offset(v_weights);
                    if (va != bctx->last_src6_va) {
                        bctx->cmd_list->SetComputeRootShaderResourceView(9, va);
                        bctx->last_src6_va = va;
                    }
                }
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
        bool gdn_or_ssm = (node->op == GGML_OP_GATED_DELTA_NET) ||
                          (node->op == GGML_OP_SSM_SCAN) ||
                          (node->op == GGML_OP_RWKV_WKV6) ||
                          (node->op == GGML_OP_RWKV_WKV7) ||
                          (node->op == GGML_OP_ADD_ID);
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
                          (fused_ssm_bias != nullptr) ||
                          (fused_mmv_set_rows != nullptr) ||
                          (fused_mmv_q_rope != nullptr) ||
                          (fused_mmv_k_set_rows != nullptr) ||
                          (node->op == GGML_OP_ROPE && node->src[2] != nullptr);
        bool needs_src3 = (node->op == GGML_OP_FLASH_ATTN_EXT) || (fused_rope_set_rows != nullptr) || (fused_5way_set_rows != nullptr) ||
                          (fused_mmv_k_set_rows != nullptr) ||
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
            } else if (fused_ssm_bias) {
                // SSM_CONV+ADD+SILU fusion: bias is a contiguous F32 vector,
                // bake its per-tensor byte offset into the VA (the shader reads
                // src2.Load(i1*4) without any cbuffer offset).
                src2_res    = dx12_get_resource(fused_ssm_bias);
                src2_offset = dx12_tensor_offset(fused_ssm_bias);
            } else if (fused_mmv_glu_up) {
                // R9: bind W_up as src2; per-tensor byte offset is encoded in op_params[1]
                // and consumed by the mul_mat_vec_glu shader (matches fused_bias_tensor pattern).
                src2_res = dx12_get_resource(fused_mmv_glu_up->src[0]);
            } else if (fused_mmv_set_rows) {
                // V-cache SET_ROWS fusion: bind the row-index buffer as src2/t2.
                // Its per-tensor byte offset rides op_params[11] (base VA here).
                src2_res = dx12_get_resource(fused_mmv_set_rows->src[1]);
            } else if (fused_mmv_q_rope) {
                // Q-ROPE fusion: bind ROPE position indices as src2/t2. Read at
                // element 0 (single-token M=1 decode); offset baked into the VA.
                src2_res    = dx12_get_resource(fused_mmv_q_rope->src[1]);
                src2_offset = dx12_tensor_offset(fused_mmv_q_rope->src[1]);
            } else if (fused_mmv_k_set_rows) {
                // K-ROPE fusion: bind ROPE position indices as src2/t2 (element
                // 0, offset baked into the VA).
                src2_res    = dx12_get_resource(fused_mmv_k_rope->src[1]);
                src2_offset = dx12_tensor_offset(fused_mmv_k_rope->src[1]);
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
            } else if (fused_mmv_k_set_rows) {
                // K-ROPE fusion: bind SET_ROWS row indices as src3/t3. Read at
                // element 0 (single-token M=1 decode); offset baked into the VA.
                src3_res    = dx12_get_resource(fused_mmv_k_set_rows->src[1]);
                src3_offset = dx12_tensor_offset(fused_mmv_k_set_rows->src[1]);
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

        // Q/K projection ROPE fusion freq_factors: bind to src4/t4 (mmv_rope_pair
        // reads src4.Load(pair*4); per-tensor offset baked into the VA). Only
        // present for RoPE-scaled models (Llama-3.1, Phi-3); has_ff==0 otherwise.
        {
            const struct ggml_tensor * mmv_rope_ff =
                fused_mmv_q_rope     ? fused_mmv_q_rope->src[2] :
                fused_mmv_k_set_rows ? fused_mmv_k_rope->src[2] : nullptr;
            if (mmv_rope_ff) {
                ID3D12Resource * ff_res = dx12_get_resource(mmv_rope_ff);
                D3D12_GPU_VIRTUAL_ADDRESS ff_va = ff_res
                    ? (ff_res->GetGPUVirtualAddress() + dx12_tensor_offset(mmv_rope_ff))
                    : src0_res->GetGPUVirtualAddress();
                if (ff_va != bctx->last_src4_va) {
                    bctx->cmd_list->SetComputeRootShaderResourceView(7, ff_va);
                    bctx->last_src4_va = ff_va;
                }
            }
        }

        // Optional src4/src5/src6 — bound for hybrid SSM ops with >4 input tensors
        // (GATED_DELTA_NET needs src0..src5; SSM_SCAN needs src0..src6).
        // FLASH_ATTN_EXT also needs src4 when "attention sinks" are present
        // (e.g. gpt-oss). The sinks tensor is a F32 vector of length n_heads.
        bool needs_src4 = (node->op == GGML_OP_GATED_DELTA_NET) ||
                          (node->op == GGML_OP_SSM_SCAN) ||
                          (node->op == GGML_OP_RWKV_WKV6) ||
                          (node->op == GGML_OP_RWKV_WKV7) ||
                          (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[4] != nullptr);
        bool needs_src5 = needs_src4;
        bool needs_src6 = (node->op == GGML_OP_SSM_SCAN) ||
                          (node->op == GGML_OP_RWKV_WKV7);

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
                bool is_matvec = (node->ne[1] == 1) || (key.flags == 83) ||
                                 (key.flags == 47) || (key.flags == 48) || (key.flags == 49) ||
                                 (key.flags == 50) || (key.flags == 51) || (key.flags == 52); // M=1, or NC2/NC4/NC8 batch paths

                if (is_matvec) {
                    if (key.flags == 83) {
                        groups_x = ((uint32_t)node->ne[0] + 31u) / 32u;
                        groups_y = ((uint32_t)node->ne[1] + 7u) / 8u;
                        groups_z = (uint32_t)(node->ne[2] * node->ne[3]);
                        matvec_row_groups = groups_x;
                        break;
                    }
                    // Matvec dispatch. The combined Q/K/V dispatch (fl=75 F16,
                    // fl=76/84 Q8_0, fl=77 Q5_0, fl=79 mixed Q5_0/Q8_0) covers all three projections, so
                    // its group count is the total row count (Wq|Wk|Wv), which
                    // params.ne0 carries; the plain matvec uses the single output
                    // width node->ne[0].
                    uint32_t N = (key.flags == 75 || key.flags == 76 ||
                                  key.flags == 77 || key.flags == 79 || key.flags == 84)
                                     ? params.ne0 : (uint32_t)node->ne[0];
                    uint32_t batches = (uint32_t)(node->ne[2] * node->ne[3]);
                    if (key.flags == 9 || key.flags == 10 || key.flags == 11 ||
                        key.flags == 12 || key.flags == 13 || key.flags == 14 ||
                        key.flags == 15 || key.flags == 16 || key.flags == 17 ||
                        key.flags == 18 || key.flags == 19 || key.flags == 20 ||
                        key.flags == 21 || key.flags == 22 || key.flags == 23 ||
                        key.flags == 24 || key.flags == 25 || key.flags == 26 ||
                        key.flags == 27 || key.flags == 31 || key.flags == 32 ||
                        key.flags == 33 || key.flags == 34 || key.flags == 35 ||
                        key.flags == 36 || key.flags == 37 || key.flags == 38 ||
                        key.flags == 44 || key.flags == 47 || key.flags == 48 ||
                        key.flags == 49 || key.flags == 50 || key.flags == 51 ||
                        key.flags == 52 ||
                        key.flags == 55 || key.flags == 56 || key.flags == 57 ||
                        key.flags == 61 || key.flags == 67 || key.flags == 72 ||
                        key.flags == 73 || key.flags == 74 ||
                        key.flags == 76 || key.flags == 77 || key.flags == 79 || key.flags == 84 ||
                        key.flags == 82 ||
                        key.flags == 78) {
                        // Multi-row: 2 rows per group (NC2/4/8 variants produce 2 rows x NUM_COLS cols)
                        // (55/56/57 = IQ4_NL / Q4_0 / Q5_0 mr256 variants on this branch)
                        // (76/77/79/84 = combined Q/K/V Q8_0/Q5_0/mixed rows2 dispatch)
                        matvec_row_groups = (N + 1) / 2;
                    } else if (key.flags == 28 || key.flags == 29 || key.flags == 45 ||
                               key.flags == 46) {
                        // 4 rows per group: Q8_0 mr64 (28), Q5_0 mr64 (29),
                        //                   Q8_0 dp4a mr64 (45), Q4_K dp4a mr4 (46)
                        matvec_row_groups = (N + 3) / 4;
                    } else {
                        // Default: one group per output row
                        matvec_row_groups = N;
                    }
                    // F16 rope-pair de-dup: one group owns two rows (the pair).
                    if (rope_rows2) {
                        matvec_row_groups = (N + 1) / 2;
                    }
                    if (matvec_row_groups > 65535) {
                        groups_x = 65535;
                        groups_y = (matvec_row_groups + 65534) / 65535;
                    } else {
                        groups_x = matvec_row_groups;
                        groups_y = 1;
                    }
                    groups_z = batches;
                } else if (key.flags == 4 || key.flags == 30 || key.flags == 54 ||
                           key.flags == 58 || key.flags == 59) {
                    // Register-blocked tiled dispatch (32×32 tile) [numthreads(16,16,1)]
                    // fl=30 = Q4_K wmma cooperative-LDS variant (same dispatch)
                    // fl=54 = tiny-K wmma_kfull (same 32x32 tile)
                    // fl=58 = Q8_1 tiled integer-dot GEMM (same dispatch)
                    uint32_t N = (uint32_t)node->ne[0];
                    uint32_t M = (uint32_t)node->ne[1];
                    uint32_t batches = (uint32_t)(node->ne[2] * node->ne[3]);
                    groups_x = (N + 31) / 32;
                    groups_y = (M + 31) / 32;
                    groups_z = batches;
                } else if (key.flags == 53) {
                    // FP16 wmma 64x64 tile [numthreads(16,16,1)] for F16xF16 GEMM
                    uint32_t N = (uint32_t)node->ne[0];
                    uint32_t M = (uint32_t)node->ne[1];
                    uint32_t batches = (uint32_t)(node->ne[2] * node->ne[3]);
                    groups_x = (N + 63) / 64;
                    groups_y = (M + 63) / 64;
                    groups_z = batches;
                } else if (key.flags == 43 || (node->src[0] && (node->src[0]->type == GGML_TYPE_Q4_K ||
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
                                            node->src[0]->type == GGML_TYPE_IQ4_NL))) {
                    // Quantized flat shaders: 1 output per thread, 256 threads/group.
                    // fl=43 covers the IQ batch path (per-element dequant on the fly).
                    uint32_t total = (uint32_t)(node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3]);
                    uint32_t total_groups = dx12_ceil_div(total, 256);
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
            case GGML_OP_SUM_ROWS:
            case GGML_OP_MEAN:
            case GGML_OP_CUMSUM: {
                // Row-based ops: one thread group per row
                uint32_t total_rows = (uint32_t)(node->ne[1] * node->ne[2] * node->ne[3]);
                groups_x = total_rows;

                // RMS_NORM + MUL + Q8_1 fused (flags=12): also allocate &
                // bind the q8_1 scratch buffer to slot 6 (u1) for the
                // shader to write the quantized pre-pass into.
                if (node->op == GGML_OP_RMS_NORM && key.flags == 12 &&
                    fused_mul_node) {
                    uint32_t total_elems = (uint32_t)ggml_nelements(fused_mul_node);
                    uint32_t num_blocks  = total_elems / 32;
                    size_t   want_size   = (size_t)num_blocks * 36;
                    if (want_size > bctx->q8_1_scratch_size) {
                        // Geometric growth + retire-don't-release (same
                        // pattern as the use_dp4a path below).
                        if (bctx->q8_1_scratch) {
                            bctx->q8_1_scratch_retired.push_back(bctx->q8_1_scratch);
                        }
                        bctx->q8_1_scratch.Reset();
                        size_t alloc = want_size * 4u;
                        if (alloc > (64u << 20)) alloc = std::max(want_size, (size_t)(64u << 20));
                        bctx->q8_1_scratch = dx12_create_buffer(bctx->dev, alloc);
                        bctx->q8_1_scratch_size = alloc;
                        bctx->last_q8_1_src_id = 0;
                        bctx->last_q8_1_size   = 0;
                    }
                    if (bctx->q8_1_scratch) {
                        bctx->cmd_list->SetComputeRootUnorderedAccessView(
                            6, bctx->q8_1_scratch->GetGPUVirtualAddress());
                    }
                }
                break;
            }
            case GGML_OP_ARGMAX: {
                // src0 is a matrix; one thread group per row reduces ne00 columns
                // in parallel to a single argmax.
                groups_x = (uint32_t)node->src[0]->ne[1];
                break;
            }
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                if (key.flags == 50) {
                    // Large-N multi-pass shader: BLOCK_SIZE=256, dispatch geometry
                    // is set to cover the INIT phase (one thread per padded slot).
                    // Subsequent SWAP/WRITEOUT phases below issue their own
                    // Dispatch calls with the right group count.
                    uint32_t ncols = (uint32_t)node->src[0]->ne[0];
                    uint32_t ncols_padded = 1;
                    while (ncols_padded < ncols) ncols_padded <<= 1;
                    if (ncols_padded < 256u) ncols_padded = 256u;
                    uint32_t nrows = (uint32_t)(node->src[0]->ne[1] *
                                                node->src[0]->ne[2] *
                                                node->src[0]->ne[3]);
                    groups_x = dx12_ceil_div(ncols_padded, 256u);
                    groups_y = nrows;
                    groups_z = 1;

                    // Ensure scratch buffer is large enough: int2 per padded slot.
                    size_t needed = (size_t)nrows * (size_t)ncols_padded * 8u;
                    if (needed > bctx->dev->argsort_scratch_size) {
                        if (bctx->dev->argsort_scratch) {
                            // Retire old buffer until graph_compute drains
                            // (matches q8_1_scratch retire pattern).
                            bctx->dev->argsort_scratch_retired.push_back(
                                bctx->dev->argsort_scratch);
                        }
                        bctx->dev->argsort_scratch.Reset();
                        // Geometric growth: allocate 4x needed (clamped to 64 MB)
                        // to avoid frequent regrowth in mixed test workloads,
                        // which has been observed to cause TDR on Intel Arc B390.
                        size_t alloc = needed * 4u;
                        if (alloc > (64u << 20)) alloc = std::max(needed, (size_t)(64u << 20));
                        bctx->dev->argsort_scratch = dx12_create_buffer(bctx->dev, alloc);
                        // dx12_create_buffer returns null on failure; only record
                        // the capacity on success, or the next graph treats the
                        // missing buffer as large enough and never retries.
                        bctx->dev->argsort_scratch_size =
                            bctx->dev->argsort_scratch ? alloc : 0;
                    }
                    if (bctx->dev->argsort_scratch) {
                        bctx->cmd_list->SetComputeRootUnorderedAccessView(
                            6, bctx->dev->argsort_scratch->GetGPUVirtualAddress());
                    }
                } else {
                    // Small-N path: one thread group per row across (ne01, ne02, ne03).
                    // 1024 threads cooperate on a bitonic sort of ne00 columns.
                    groups_x = (uint32_t)node->src[0]->ne[1];
                    groups_y = (uint32_t)node->src[0]->ne[2];
                    groups_z = (uint32_t)node->src[0]->ne[3];
                }
                break;
            }
            case GGML_OP_ADD_ID: {
                // One thread group per dst row (n_used * n_token * ne3).
                uint32_t total_rows = (uint32_t)(node->ne[1] * node->ne[2] * node->ne[3]);
                groups_x = total_rows;
                break;
            }
            case GGML_OP_SOLVE_TRI: {
                // 64 threads per group; each thread handles one RHS column.
                // groups_x = ceil(K / 64) * (ne02 * ne03).
                uint32_t K = (uint32_t)node->src[1]->ne[0];
                uint32_t k_chunks = (K + 63) / 64;
                uint32_t batches = (uint32_t)(node->ne[2] * node->ne[3]);
                groups_x = k_chunks * batches;
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
                groups_x = dx12_ceil_div(total, 256);
                break;
            }
            case GGML_OP_FLASH_ATTN_EXT: {
                // Split-KV: increase parallelism by splitting KV across groups
                uint32_t N_queries = (uint32_t)node->src[0]->ne[1];
                uint32_t n_heads   = (uint32_t)node->src[0]->ne[2];
                uint32_t batch     = (uint32_t)node->src[0]->ne[3];
                uint32_t N_kv      = (uint32_t)node->src[1]->ne[1];
                uint32_t n_kv_heads = (uint32_t)node->src[1]->ne[2];

                // Multi-query tiled FA: reuse each staged K/V tile across 8
                // query rows by default. Intel UHD uses 16 rows, which is
                // consistently 20-25% faster across D=72/96/128; NVIDIA keeps
                // 8 rows because 16 regresses most tested shapes. Override with
                // DX12_FA_TILED_BR=8/16.
                static const bool fa_tiled_enabled = dx12_flag_default_on("DX12_FA_TILED");
                static const uint32_t fa_tiled_min_q = [] {
                    const char * env = DX12_GETENV("DX12_FA_TILED_MINQ");
                    return env ? (uint32_t)std::max(1, atoi(env)) : 64u;
                }();
                // The tile only pays for itself once the head is wide enough to
                // amortize the staged K/V tile. At D=64 the untiled path wins by
                // 19-30% PP on RX 6800 (135M Q8_0 8887->11526, 135M F16
                // 6218->7378, SmolVLM2-256M Q8_0 8850->11242) while at D=96 it
                // is 2x ahead (Phi-3 Q8_0 1024 vs 466), so gate on head width.
                // Override with DX12_FA_TILED_MIND.
                static const uint32_t fa_tiled_min_d = [] {
                    const char * env = DX12_GETENV("DX12_FA_TILED_MIND");
                    return env ? (uint32_t)std::max(1, atoi(env)) : 72u;
                }();
                // Each workgroup re-streams the whole K/V for its head, so 16 rows
                // halves that traffic and wins on long KV. Short KV stays cache
                // resident, where the larger tile's LDS/occupancy cost dominates
                // instead (measured on B390: kv=4096 465->503 GF/s, kv=512 539->517).
                static const uint32_t fa_tiled_br16_min_kv = [] {
                    const char * env = DX12_GETENV("DX12_FA_TILED_BR16_MINKV");
                    return env ? (uint32_t)std::max(1, atoi(env)) : 2048u;
                }();
                const char * fa_tiled_br_env = DX12_GETENV("DX12_FA_TILED_BR");
                const uint32_t fa_tiled_br = fa_tiled_br_env
                    ? (atoi(fa_tiled_br_env) == 16 ? 16u : 8u)
                    : (bctx->dev->arch_family == DX12_ARCH_INTEL_UHD ||
                       N_kv >= fa_tiled_br16_min_kv ? 16u : 8u);
                const uint32_t head_dim = (uint32_t)node->src[0]->ne[0];
                const uint32_t value_dim = (uint32_t)node->src[2]->ne[0];
                auto fa_tiled_type = [](ggml_type t) {
                    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
                };
                bool fa_tiled = false;
                if (fa_tiled_enabled && key.flags == 0 &&
                    N_queries >= fa_tiled_min_q &&
                    head_dim == value_dim && head_dim >= fa_tiled_min_d && head_dim <= 128 &&
                    (head_dim % 4) == 0 &&
                    n_heads > 0 && n_kv_heads > 0 && (n_heads % n_kv_heads) == 0 &&
                    node->src[0]->type == GGML_TYPE_F32 && node->src[0]->nb[0] == 4 &&
                    fa_tiled_type(node->src[1]->type) &&
                    fa_tiled_type(node->src[2]->type) &&
                    node->src[1]->nb[0] == ggml_type_size(node->src[1]->type) &&
                    node->src[2]->nb[0] == ggml_type_size(node->src[2]->type) &&
                    dx12_ceil_div(N_queries, fa_tiled_br) <= 65535) {
                    dx12_pipeline_key tiled_key = key;
                    tiled_key.flags = fa_tiled_br == 16 ? 29u : 28u;
                    dx12_pipeline * tiled_pl = bctx->dev->get_or_create_pipeline(tiled_key);
                    if (tiled_pl && tiled_pl->pso) {
                        bctx->cmd_list->SetPipelineState(tiled_pl->pso.Get());
                        bctx->last_pso = tiled_pl->pso.Get();
                        pipeline = tiled_pl;
                        key.flags = tiled_key.flags;
                        fa_tiled = true;
                    }
                }

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
                if (!fa_tiled && gqa_enabled &&
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

                // Cooperative decode flash attention (DX12_FA_COOP): the pure
                // decode step (n_q == 1) with contiguous F16 K/V and HSK == HSV
                // routes to a D-split single-wave shader (flags 26/30/27 for
                // head_dim 64/96/128) that loads K/V vectorised and coalesced
                // and reduces with wave butterflies. Enabled by default
                // (disable via DX12_FA_COOP=0); needs native fp16 ops.
                // Preserves the split-KV grid + reduce and all other paths.
                static const bool fa_coop_enabled = dx12_flag_default_on("DX12_FA_COOP");
                static const bool fa_coop_q8_0_enabled = dx12_flag_default_on("DX12_FA_COOP_Q8_0");
                bool fa_coop = false;
                const bool coop_kv_f16 =
                    node->src[1] && node->src[2] &&
                    node->src[1]->type == GGML_TYPE_F16 && node->src[1]->nb[0] == 2 &&
                    node->src[2]->type == GGML_TYPE_F16 && node->src[2]->nb[0] == 2;
                // Q8_0 K/V decode: cd_load_kv4 amortises the block scale over four
                // elements, versus the per-element mmid_dequant in flash_attn.hlsl
                // (8 loads for the same 4 values). Head dims 64/96/128 are all
                // multiples of the 32-element Q8_0 block, so a fetch never straddles.
                const bool coop_kv_q8_0 =
                    fa_coop_q8_0_enabled &&
                    node->src[1] && node->src[2] &&
                    node->src[1]->type == GGML_TYPE_Q8_0 &&
                    node->src[2]->type == GGML_TYPE_Q8_0;
                if (fa_coop_enabled && !gqa_fold && key.flags == 0 &&
                    bctx->dev->fp16_supported &&
                    N_queries == 1 && node->src[0] &&
                    node->src[0]->type == GGML_TYPE_F32 && node->src[0]->nb[0] == 4 &&
                    (coop_kv_f16 || coop_kv_q8_0) &&
                    node->src[0]->ne[0] == node->src[2]->ne[0] &&
                    ((uint32_t)node->src[0]->ne[0] == 64 ||
                     (uint32_t)node->src[0]->ne[0] == 96 ||
                     (uint32_t)node->src[0]->ne[0] == 128)) {
                    const uint32_t coop_d = (uint32_t)node->src[0]->ne[0];
                    uint32_t coop_flag = coop_kv_q8_0
                        ? (coop_d == 64 ? 31u : coop_d == 96 ? 32u : 33u)
                        : (coop_d == 64 ? 26u : coop_d == 96 ? 30u : 27u);
                    dx12_pipeline_key coop_key = key;
                    coop_key.flags = coop_flag;
                    dx12_pipeline * coop_pl = bctx->dev->get_or_create_pipeline(coop_key);
                    if (coop_pl && coop_pl->pso) {
                        bctx->cmd_list->SetPipelineState(coop_pl->pso.Get());
                        bctx->last_pso = coop_pl->pso.Get();
                        pipeline = coop_pl;
                        key.flags = coop_flag;
                        fa_coop = true;
                    }
                }

                // Quant KV cache: route to the per-quant FA wrapper shader.
                // flags 20..25 cover Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL. The
                // wrapper sets KV_<TYPE>, which pulls in quant_dequant.hlsli and
                // swaps the K/V load_auto calls in flash_attn.hlsl for per-element
                // mmid_dequant. The small-D 64/128 variants and GQA-fold remain
                // F32/F16/BF16-only for now (coverage-first).
                if (!gqa_fold && key.flags == 0) {
                    uint32_t fa_quant_flag = 0;
                    switch (node->src[1]->type) {
                        case GGML_TYPE_Q4_0:   fa_quant_flag = 20; break;
                        case GGML_TYPE_Q4_1:   fa_quant_flag = 21; break;
                        case GGML_TYPE_Q5_0:   fa_quant_flag = 22; break;
                        case GGML_TYPE_Q5_1:   fa_quant_flag = 23; break;
                        case GGML_TYPE_Q8_0:   fa_quant_flag = 24; break;
                        case GGML_TYPE_IQ4_NL: fa_quant_flag = 25; break;
                        default: break;
                    }
                    if (fa_quant_flag != 0) {
                        dx12_pipeline_key quant_key = key;
                        quant_key.flags = fa_quant_flag;
                        dx12_pipeline * quant_pl = bctx->dev->get_or_create_pipeline(quant_key);
                        if (quant_pl && quant_pl->pso) {
                            bctx->cmd_list->SetPipelineState(quant_pl->pso.Get());
                            bctx->last_pso = quant_pl->pso.Get();
                            pipeline = quant_pl;
                        }
                    }
                }

                // Small-D decode-friendly variants: when D is small we prefer
                // a smaller GROUP_SIZE so Pass-3 V accumulation has higher
                // thread utilization. Smaller workgroups also let more
                // workgroups run concurrently on small-wave GPUs (Intel Arc).
                //   D <= 64  → flash_attn_64  (GROUP_SIZE=TILE_KV=64)
                //   D <= 128 → flash_attn_128 (GROUP_SIZE=TILE_KV=128) — covers
                //              ViT (D=72/80) and many Q-heads (D=80/96/128)
                bool kv_is_quant_fa = (node->src[1]->type == GGML_TYPE_Q4_0 ||
                                       node->src[1]->type == GGML_TYPE_Q4_1 ||
                                       node->src[1]->type == GGML_TYPE_Q5_0 ||
                                       node->src[1]->type == GGML_TYPE_Q5_1 ||
                                       node->src[1]->type == GGML_TYPE_Q8_0 ||
                                       node->src[1]->type == GGML_TYPE_IQ4_NL);
                if (!gqa_fold && !kv_is_quant_fa && key.flags == 0 && head_dim <= 128) {
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
                // For small-D variants (GROUP_SIZE=64), each wg costs 4x fewer
                // threads than the default (256), so we can afford finer splits
                // to fill more SMs on small models like SmolVLM2 (n_heads=9).
                // Empirically: FA=1 on SmolVLM2 Q4_K_M decode is 12% slower than
                // FA=0 because n_splits caps at 4 (N_kv=128 / 32 min KV/split),
                // giving only ~36 wgs of work on a 142-SM RTX 6000 Ada.
                bool fa_is_small_d = (key.flags == 0 && head_dim <= 128 && !gqa_fold && !kv_is_quant_fa);
                // Cooperative decode wgs are single-wave (cheap), so give them the
                // same fine-split budget as the small-D variants they replace.
                bool fa_fine = fa_is_small_d || fa_coop;
                uint32_t min_kv_per_split = fa_fine ? 16u : 32u;
                uint32_t query_groups = fa_tiled
                    ? dx12_ceil_div(N_queries, fa_tiled_br)
                    : N_queries;
                uint32_t total_groups_no_split = query_groups * dispatch_heads * batch;
                uint32_t target_groups = fa_tiled ? 0u : (fa_fine ? 512u : 256u);
                uint32_t n_splits = 1;
                if (!fa_tiled &&
                    total_groups_no_split < target_groups && N_kv > min_kv_per_split) {
                    n_splits = (target_groups + total_groups_no_split - 1) / total_groups_no_split;
                    n_splits = std::min(n_splits, (N_kv + min_kv_per_split - 1) / min_kv_per_split);
                    n_splits = std::min(n_splits, (uint32_t)32);      // cap at 32 splits
                }

                // op_params[15]: low 16 bits = n_splits, high 16 bits = gqa_ratio.
                // Always pack so flash_attn.hlsl, flash_attn_gqa.hlsl, and the
                // reduce shader can use the same convention (they all mask low16).
                params.op_params[15] = (n_splits & 0xFFFFu) | ((gqa_ratio & 0xFFFFu) << 16);
                // Re-upload params since we modified op_params
                bctx->set_shader_params(params, (uint32_t)(sizeof(params) / 4));

                groups_x = query_groups;
                groups_y = dispatch_heads;
                groups_z = batch * n_splits;

                // Command-list replay: record this FA dispatch's dynamic linkage
                // so a later replay can revalidate n_splits (which drives the
                // baked groups_z) and refresh the CBV slot when N_kv changes.
                if (bctx->replay.capturing) {
                    dx12_cmd_replay::fa_rec rec;
                    rec.node_index            = i;
                    rec.cbv_offset            = bctx->replay.last_param_offset;
                    rec.total_groups_no_split = total_groups_no_split;
                    rec.target_groups         = target_groups;
                    rec.min_kv_per_split      = min_kv_per_split;
                    rec.gqa_ratio             = gqa_ratio;
                    rec.n_kv                  = N_kv;
                    rec.n_splits              = n_splits;
                    bctx->replay.fa.push_back(rec);
                }

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
                // One thread per output element; split groups across X/Y to
                // clear D3D12's 65535 per-dim limit for large IM2COL tensors.
                uint32_t total_elements = (uint32_t)(ggml_nelements(node));
                uint32_t total_groups = dx12_ceil_div(total_elements, 256);
                if (total_groups > 65535) {
                    groups_x = 65535;
                    groups_y = (total_groups + 65534) / 65535;
                } else {
                    groups_x = total_groups;
                }
                break;
            }
            case GGML_OP_IM2COL_3D: {
                // One thread per dst element; perf cases produce ~4B elements
                // so we need to split across X/Y for the 65535-per-dim limit.
                uint32_t total_elements = (uint32_t)(ggml_nelements(node));
                uint32_t total_groups = dx12_ceil_div(total_elements, 256);
                if (total_groups > 65535) {
                    groups_x = 65535;
                    groups_y = (total_groups + 65534) / 65535;
                } else {
                    groups_x = total_groups;
                }
                break;
            }
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_TRANSPOSE_1D: {
                // One thread per dst element; split groups across X/Y to clear
                // D3D12's 65535 per-dim limit (large perf tests have ~67M dst).
                uint32_t total_elements = (uint32_t)(ggml_nelements(node));
                uint32_t total_groups = dx12_ceil_div(total_elements, 256);
                if (total_groups > 65535) {
                    groups_x = 65535;
                    groups_y = (total_groups + 65534) / 65535;
                } else {
                    groups_x = total_groups;
                }
                break;
            }
            case GGML_OP_MUL_MAT_ID: {
                if (key.flags == 1 || key.flags == 17) {
                    groups_x = dx12_ceil_div((uint32_t)node->ne[0], 2);
                    groups_y = (uint32_t)node->ne[1];
                    groups_z = (uint32_t)(node->ne[2] * node->ne[3]);
                } else {
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
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_RWKV_WKV7: {
                // One workgroup per (batch, head); shader does BLOCK_SIZE
                // (==head_size==64) threads.
                const struct ggml_tensor * state =
                    (node->op == GGML_OP_RWKV_WKV6) ? node->src[5] : node->src[6];
                uint32_t H = (uint32_t)node->src[0]->ne[1];
                uint32_t B = (uint32_t)state->ne[1];
                groups_x = H * B;
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
            case GGML_OP_SET_ROWS: {
                // Quantized SET_ROWS (Q8_0 etc.): one thread per block,
                // 32 threads/workgroup (matches Vulkan copy_to_quant.comp).
                if (ggml_is_quantized(node->type) && node->src[0]) {
                    const uint32_t qk         = (uint32_t)ggml_blck_size(node->type);
                    const uint32_t total_elem = (uint32_t)ggml_nelements(node->src[0]);
                    const uint32_t total_blk  = (total_elem + qk - 1) / qk;
                    uint32_t ne = (total_blk + 31u) / 32u;
                    if (ne > 262144u) {
                        groups_x = 512u;
                        groups_y = 512u;
                        groups_z = (ne + 262143u) / 262144u;
                    } else if (ne > 512u) {
                        groups_x = 512u;
                        groups_y = (ne + 511u) / 512u;
                        groups_z = 1u;
                    } else {
                        groups_x = ne;
                        groups_y = 1u;
                        groups_z = 1u;
                    }
                    break;
                }
                // Non-quantized SET_ROWS (F32 / F16): same element-wise
                // geometry as the default branch, but inlined here so we
                // don't fall through into a goto. Dispatch is sized by
                // src0 nelements (not dst!) because dst is the full KV
                // cache while src0 only contains the new rows to write.
                if (node->src[0]) {
                    uint32_t total_elements = (uint32_t)ggml_nelements(node->src[0]);
                    // paired_f16 fast path (only valid for F16 dst with
                    // aligned ne0/nb0/dst_offset/nb1; mirrors set_rows.hlsl).
                    bool paired_f16 = (node->type == GGML_TYPE_F16) &&
                                      (node->nb[0] == 2) &&
                                      ((node->ne[0] & 1) == 0) &&
                                      ((dx12_tensor_offset(node) & 3) == 0) &&
                                      ((node->nb[1] & 3) == 0);
                    if (paired_f16) total_elements /= 2;
                    groups_x = (total_elements + 255) / 256;
                    break;
                }
                groups_x = 1;
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
        if (phase_profile) {
            uint64_t now = dx12_qpc_us();
            bctx->phase_setup_us += now - phase_detail_start_us;
            phase_detail_start_us = now;
        }

        // Determine the effective destination tensor (accounting for fusion)
        struct ggml_tensor * dst_tensor = fused_qk_postop ? fused_qk_scale :
                                          (fused_5way_set_rows ? fused_5way_set_rows :
                                          (fused_rope_after_rms ? fused_rope_after_rms :
                                          (fused_mul_node ? fused_mul_node :
                                          (fused_bias_add ? fused_bias_add : 
                                          (fused_rope_set_rows ? fused_rope_set_rows :
                                          (fused_mmv_glu_glu ? fused_mmv_glu_glu :
                                          (fused_ssm_silu ? fused_ssm_silu :
                                          (fused_mmv_set_rows ? fused_mmv_set_rows :
                                          (fused_mmv_q_rope ? fused_mmv_q_rope :
                                          (fused_qkv ? fused_qkv_q_rope :
                                          (fused_mmv_k_set_rows ? fused_mmv_k_set_rows : node)))))))))));

        // Dependency-tracked UAV barriers
        // Only insert when the current dispatch reads a tensor written by a previous unsynced dispatch.
        {
            bool need_barrier = false;

            // Diagnostic: DX12_FORCE_BARRIER=1 inserts a UAV barrier before
            // every dispatch.  Used to isolate synchronization regressions.
            static const bool force_barrier = (getenv("DX12_FORCE_BARRIER") != nullptr);
            // Barrier elision (cross-backend review opportunity #1): couple
            // precise byte-range RAW/WAW detection across ALL tensors with the
            // resource-scoped emit path below, so a hazard on one resource no
            // longer drains every in-flight dispatch (the Vulkan barrier model,
            // ggml-vulkan.cpp:14714).  Opt-in while validated across the
            // multi-vendor fleet; promote to default after a sweep (as with the
            // subgroup / GLU flags).  DX12_SCOPED_UAV_BARRIERS remains a
            // back-compat alias that enables only the scoped emit path.
            // Default OFF; enable with =1.  Robust to =0 (both `set X=` and
            // `set X=0` disable it), matching the other DX12_* flag idioms.
            static const bool barrier_elision = []{
                const char * v = getenv("DX12_BARRIER_ELISION");
                return v && v[0] && v[0] != '0';
            }();
            if (force_barrier) {
                need_barrier = true;
            } else
            // Conservative KV barriers are OFF by default now that precise
            // byte-range dependency tracking (DX12_PRECISE_KV_BARRIERS, enabled
            // by default) covers the SET_ROWS / FA / fused ROPE+SET_ROWS write
            // hazards. Set DX12_PRECISE_KV_BARRIERS=0 to fall back to the
            // conservative blanket, or DX12_RELAXED_KV_BARRIERS=1 to skip both
            // and rely purely on the alias-aware tracker for diagnostics.
            {
            static const bool relaxed_kv = (getenv("DX12_RELAXED_KV_BARRIERS") != nullptr);
            static const bool precise_kv = dx12_flag_default_on("DX12_PRECISE_KV_BARRIERS");
            const bool conservative_kv = (node->op == GGML_OP_SET_ROWS ||
                                          node->op == GGML_OP_FLASH_ATTN_EXT ||
                                          fused_rope_set_rows ||
                                          fused_5way_set_rows ||
                                          fused_qk_postop);
            if (!relaxed_kv && !precise_kv && conservative_kv) {
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
                if (!need_barrier && fused_qk_postop) {
                    if (unsynced_writes.count((uintptr_t)fused_qk_k_rope->src[0]) ||
                        unsynced_writes.count((uintptr_t)fused_qk_k_set_rows->src[1])) {
                        need_barrier = true;
                    }
                }
                if (unsynced_writes.count((uintptr_t)dst_tensor)) {
                    need_barrier = true;
                }
                // DX12_PRECISE_KV_BARRIERS: replace the conservative KV blanket
                // with a byte-range RAW/WAW test.  FA reads K(src1)/V(src2) and
                // SET_ROWS / fused *_set_rows write the KV dst via view roots the
                // pointer-keyed set above can't match; a memory-overlap check
                // against unsynced_write_ranges catches those real hazards
                // while dropping the blanket's false positives.
                if (precise_kv && !need_barrier && conservative_kv) {
                    auto overlaps_write = [&](const ggml_tensor * t) -> bool {
                        if (!t || !t->data) return false;
                        void *    res  = (void *) dx12_get_resource(t);
                        uintptr_t lo   = (uintptr_t) t->data;
                        uintptr_t hi   = lo + ggml_nbytes(t);
                        for (const auto & w : unsynced_write_ranges) {
                            if (w.res == res && lo < w.hi && w.lo < hi) return true;
                        }
                        return false;
                    };
                    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                        if (overlaps_write(node->src[1]) || overlaps_write(node->src[2])) need_barrier = true;
                    } else if (fused_qk_postop) {
                        if (overlaps_write(fused_qk_k_set_rows) ||
                            overlaps_write(dst_tensor)) need_barrier = true;
                    } else {
                        if (overlaps_write(dst_tensor)) need_barrier = true;
                    }
                }
            }
            }

            // Barrier elision: precise byte-range RAW/WAW across ALL tensors
            // (the pointer-keyed set above only catches same-tensor hazards).
            // When the graph allocator recycles a buffer, a later dispatch can
            // touch memory a prior unsynced dispatch wrote through a different
            // view root; global barriers masked this by draining on every
            // hazard, but scoped barriers must catch it precisely or an
            // unrelated resource's barrier will not cover the miss.  Same-
            // tensor cases already set need_barrier above, so this only adds
            // the cross-view aliases the pointer path misses.
            if (barrier_elision && !need_barrier) {
                auto range_hits_write = [&](const struct ggml_tensor * t, bool as_write) -> bool {
                    if (!t || !t->data) return false;
                    void *    res  = (void *) dx12_get_resource(t);
                    uintptr_t lo   = (uintptr_t) t->data;
                    uintptr_t hi   = lo + ggml_nbytes(t);
                    uintptr_t root = tensor_root(t);
                    for (const auto & w : unsynced_write_ranges) {
                        if (w.res != res || hi <= w.lo || w.hi <= lo) continue;
                        // WAW: skip the in-place same-root case (a same-view
                        // write is already covered by the pointer-keyed WAW).
                        // RAW: any overlap is a real read-after-write hazard.
                        if (as_write && w.root == root) continue;
                        return true;
                    }
                    return false;
                };
                for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                    if (range_hits_write(node->src[s], false)) { need_barrier = true; break; }
                }
                if (!need_barrier && fused_qk_postop) {
                    if (range_hits_write(fused_qk_k_rope->src[0], false) ||
                        range_hits_write(fused_qk_k_set_rows->src[1], false)) {
                        need_barrier = true;
                    }
                }
                if (!need_barrier && range_hits_write(dst_tensor, true)) need_barrier = true;
            }

            // Write-after-read: this dispatch writes a memory region that a
            // prior un-synchronized dispatch read.  The write-based tracking
            // above only covers read-after-write / write-after-write; without
            // this, an op can overwrite an activation buffer whose memory the
            // graph allocator has since recycled while an earlier dispatch is
            // still reading the old tensor.  RDNA1/2 overlaps dispatches
            // aggressively and exposes this as corrupted F16 decode; GPUs that
            // serialize compute hide it.  Aliasing is by resource + byte range
            // because the recycled tensors have unrelated view roots (the
            // same-root case is already covered by the write tracking and the
            // conservative KV-cache barriers above).
            if (!need_barrier && dst_tensor->data) {
                void *    dres  = (void *) dx12_get_resource(dst_tensor);
                uintptr_t dlo   = (uintptr_t) dst_tensor->data;
                uintptr_t dhi   = dlo + ggml_nbytes(dst_tensor);
                uintptr_t droot = tensor_root(dst_tensor);
                for (const auto & r : unsynced_reads) {
                    if (r.res == dres && dlo < r.hi && r.lo < dhi && r.root != droot) {
                        need_barrier = true;
                        break;
                    }
                }
            }

            if (need_barrier) {
                // barriers for only the resources implicated in the hazard instead
                // of one global (pResource=nullptr) barrier that drains all
                // in-flight UAV work.  A global drain serializes every dispatch;
                // scoping lets dispatches touching unrelated resources overlap
                // (closer to the Vulkan barrier model).  Falls back to a global
                // barrier whenever the implicated resource set cannot be fully
                // resolved, so correctness never depends on the gather succeeding.
                static const bool scoped_uav = (getenv("DX12_SCOPED_UAV_BARRIERS") != nullptr);

                bool did_scoped = false;
                if ((scoped_uav || barrier_elision) && !force_barrier) {
                    ID3D12Resource * hres[16];
                    int  n_hres    = 0;
                    bool gather_ok = true;
                    auto add_res = [&](ID3D12Resource * r) {
                        if (!r) { gather_ok = false; return; }
                        for (int i = 0; i < n_hres; i++) { if (hres[i] == r) return; }
                        if (n_hres < (int)(sizeof(hres) / sizeof(hres[0]))) hres[n_hres++] = r;
                        else gather_ok = false;
                    };

                    // Conservative KV hazards: FA reads K(src1)/V(src2); SET_ROWS
                    // and fused *_set_rows write the KV-cache dst.  These are the
                    // aliasing cases the dependency tracker can miss, so scope to
                    // the KV resource explicitly.
                    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                        add_res(dx12_get_resource(node->src[1]));
                        add_res(dx12_get_resource(node->src[2]));
                    } else if (node->op == GGML_OP_SET_ROWS || fused_rope_set_rows ||
                               fused_5way_set_rows || fused_qk_postop) {
                        add_res(dx12_get_resource(
                            fused_qk_postop ? fused_qk_k_set_rows : dst_tensor));
                    }
                    // Read-after-write: any src written by an unsynced dispatch.
                    for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                        if (unsynced_writes.count((uintptr_t)node->src[s])) {
                            add_res(dx12_get_resource(node->src[s]));
                        }
                    }
                    if (fused_mul_node) {
                        for (int s = 0; s < GGML_MAX_SRC && fused_mul_node->src[s]; s++) {
                            if (unsynced_writes.count((uintptr_t)fused_mul_node->src[s])) {
                                add_res(dx12_get_resource(fused_mul_node->src[s]));
                            }
                        }
                    }
                    if (fused_qk_postop) {
                        if (unsynced_writes.count((uintptr_t)fused_qk_k_rope->src[0])) {
                            add_res(dx12_get_resource(fused_qk_k_rope->src[0]));
                        }
                        if (unsynced_writes.count((uintptr_t)fused_qk_k_set_rows->src[1])) {
                            add_res(dx12_get_resource(fused_qk_k_set_rows->src[1]));
                        }
                    }
                    // Write-after-write on the destination.
                    if (unsynced_writes.count((uintptr_t)dst_tensor)) {
                        add_res(dx12_get_resource(dst_tensor));
                    }
                    // Write-after-read: dst overwrites memory an unsynced read still uses.
                    if (dst_tensor->data) {
                        void *    dres  = (void *) dx12_get_resource(dst_tensor);
                        uintptr_t dlo   = (uintptr_t) dst_tensor->data;
                        uintptr_t dhi   = dlo + ggml_nbytes(dst_tensor);
                        uintptr_t droot = tensor_root(dst_tensor);
                        for (const auto & r : unsynced_reads) {
                            if (r.res == dres && dlo < r.hi && r.lo < dhi && r.root != droot) {
                                add_res((ID3D12Resource *) r.res);
                            }
                        }
                    }

                    // Barrier elision: gather resources for the cross-view
                    // byte-range RAW/WAW hazards detected above, so the scoped
                    // set stays a superset of what triggered the barrier.  If
                    // any hazard's resource were omitted the scoped barrier
                    // would not cover it, so this MUST mirror the detection.
                    if (barrier_elision) {
                        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                            const struct ggml_tensor * t = node->src[s];
                            if (!t || !t->data) continue;
                            void *    res = (void *) dx12_get_resource(t);
                            uintptr_t lo  = (uintptr_t) t->data;
                            uintptr_t hi  = lo + ggml_nbytes(t);
                            for (const auto & w : unsynced_write_ranges) {
                                if (w.res == res && lo < w.hi && w.lo < hi) { add_res((ID3D12Resource *) res); break; }
                            }
                        }
                        if (fused_qk_postop) {
                            const ggml_tensor * hidden_srcs[2] = {
                                fused_qk_k_rope->src[0],
                                fused_qk_k_set_rows->src[1],
                            };
                            for (const ggml_tensor * t : hidden_srcs) {
                                if (!t || !t->data) continue;
                                void *    res = (void *)dx12_get_resource(t);
                                uintptr_t lo  = (uintptr_t)t->data;
                                uintptr_t hi  = lo + ggml_nbytes(t);
                                for (const auto & w : unsynced_write_ranges) {
                                    if (w.res == res && lo < w.hi && w.lo < hi) {
                                        add_res((ID3D12Resource *)res);
                                        break;
                                    }
                                }
                            }
                        }
                        if (dst_tensor->data) {
                            void *    res  = (void *) dx12_get_resource(dst_tensor);
                            uintptr_t lo   = (uintptr_t) dst_tensor->data;
                            uintptr_t hi   = lo + ggml_nbytes(dst_tensor);
                            uintptr_t root = tensor_root(dst_tensor);
                            for (const auto & w : unsynced_write_ranges) {
                                if (w.res == res && lo < w.hi && w.lo < hi && w.root != root) { add_res((ID3D12Resource *) res); break; }
                            }
                        }
                    }

                    if (gather_ok && n_hres > 0) {
                        bctx->emit_uav_barrier_scoped(hres, n_hres);
                        bctx->dbg_barrier_scoped++;

                        auto in_hres = [&](ID3D12Resource * r) {
                            for (int i = 0; i < n_hres; i++) { if (hres[i] == r) return true; }
                            return false;
                        };
                        // Quantize-cache: invalidate only if the cached src's
                        // resource is one we just barriered (see global path below).
                        if (bctx->last_q8_1_src_id != 0 &&
                            unsynced_writes.count(bctx->last_q8_1_src_id) &&
                            !bctx->q8_1_cache_safe &&
                            in_hres(dx12_get_resource((const ggml_tensor *)bctx->last_q8_1_src_id))) {
                            bctx->last_q8_1_src_id = 0;
                        }
                        bctx->q8_1_cache_safe = false;
                        // Selective clear: drop only tracking entries for the
                        // resources we synced; entries in other resources remain
                        // unsynced so their hazards still trigger later barriers.
                        for (auto it = unsynced_writes.begin(); it != unsynced_writes.end(); ) {
                            if (in_hres(dx12_get_resource((const ggml_tensor *)*it))) it = unsynced_writes.erase(it);
                            else ++it;
                        }
                        for (auto it = unsynced_reads.begin(); it != unsynced_reads.end(); ) {
                            if (in_hres((ID3D12Resource *) it->res)) it = unsynced_reads.erase(it);
                            else ++it;
                        }
                        for (auto it = unsynced_write_ranges.begin(); it != unsynced_write_ranges.end(); ) {
                            if (in_hres((ID3D12Resource *) it->res)) it = unsynced_write_ranges.erase(it);
                            else ++it;
                        }
                        did_scoped = true;
                    }
                }

                if (!did_scoped) {
                    bctx->emit_uav_barrier_global();
                    bctx->dbg_barrier_global++;
                    // Quantize-cache: invalidate ONLY when the cached src1 was one
                    // of the unsynced writes being flushed -- otherwise the cached
                    // quantized data is still valid (Vulkan tracks this with
                    // per-scratch prealloc_*_need_sync flags; we use the unified
                    // unsynced_writes set, so we can be precise as long as we do
                    // the lookup *before* clearing the set).
                    //
                    // Exception: when the cache was pre-populated by an immediately
                    // preceding rms_norm_mul_quantize_q8_1 fused dispatch, the Q8_1
                    // bytes are already protected by a UAV barrier emitted by that
                    // dispatch. The F32 dst is still in unsynced_writes (correctly
                    // triggering the barrier above for any F32-side reader), but
                    // the Q8_1 scratch is fresh. q8_1_cache_safe gates the
                    // invalidation skip; the flag is cleared after one use so the
                    // next matmul (with its own src1) sees normal cache rules.
                    if (bctx->last_q8_1_src_id != 0 &&
                        unsynced_writes.count(bctx->last_q8_1_src_id) &&
                        !bctx->q8_1_cache_safe) {
                        bctx->last_q8_1_src_id = 0;
                    }
                    bctx->q8_1_cache_safe = false;
                    unsynced_writes.clear();
                    unsynced_reads.clear();
                    unsynced_write_ranges.clear();
                }
            }
        }
        if (phase_profile) {
            uint64_t now = dx12_qpc_us();
            bctx->phase_barrier_us += now - phase_detail_start_us;
            phase_detail_start_us = now;
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
                HRESULT hr_scratch = bctx->dev->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&bctx->q8_1_scratch));
                // The dispatch below binds this unconditionally; fail loudly here
                // rather than dereferencing a null resource.
                DX12_CHECK(hr_scratch, "CreateCommittedResource(q8_1_scratch)");
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
                if (reuse_q8_1) bctx->dbg_q8_quant_reused++;
                if (!reuse_q8_1) {
                    bctx->dbg_q8_quant_dispatched++;
                    bctx->cmd_list->SetPipelineState(q_pipeline->pso.Get());
                    bctx->last_pso = q_pipeline->pso.Get();

                    // Set params: src0_offset = src1's offset, dst_offset = 0
                    dx12_shader_params q_params = {};
                    q_params.src0_offset = this_src_off;
                    q_params.dst_offset = 0;
                    q_params.ne0 = num_q8_blocks;
                    bctx->set_shader_params(q_params, 30);

                    // Bind src1 as src0 (input to quantize), scratch as dst
                    bctx->cmd_list->SetComputeRootShaderResourceView(1, src1_res->GetGPUVirtualAddress());
                    bctx->cmd_list->SetComputeRootUnorderedAccessView(3, bctx->q8_1_scratch->GetGPUVirtualAddress());
                    bctx->last_src0_va = src1_res->GetGPUVirtualAddress();
                    bctx->last_dst_va = bctx->q8_1_scratch->GetGPUVirtualAddress();

                    const uint32_t q_groups_x = std::min(num_q8_blocks, 65535u);
                    const uint32_t q_groups_y = (num_q8_blocks + 65534u) / 65535u;
                    bctx->cmd_list->Dispatch(q_groups_x, q_groups_y, 1);

                    // Barrier before MUL_MAT reads the quantized data
                    bctx->emit_uav_barrier_buffer(bctx->q8_1_scratch.Get());

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
                bctx->set_shader_params(params, num_constants);
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
                HRESULT hr_scratch = bctx->dev->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&bctx->q8_1_scratch));
                DX12_CHECK(hr_scratch, "CreateCommittedResource(q8_1_scratch)");
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
                if (reuse_q8_1) bctx->dbg_q8_quant_reused++;
                if (!reuse_q8_1) {
                    bctx->dbg_q8_quant_dispatched++;
                    bctx->cmd_list->SetPipelineState(q_pipeline->pso.Get());
                    bctx->last_pso = q_pipeline->pso.Get();

                    dx12_shader_params q_params = {};
                    q_params.src0_offset = this_src_off;
                    q_params.dst_offset = 0;
                    q_params.ne0 = num_q8_blocks;
                    bctx->set_shader_params(q_params, 30);

                    bctx->cmd_list->SetComputeRootShaderResourceView(1, src1_res->GetGPUVirtualAddress());
                    bctx->cmd_list->SetComputeRootUnorderedAccessView(3, bctx->q8_1_scratch->GetGPUVirtualAddress());
                    bctx->last_src0_va = src1_res->GetGPUVirtualAddress();
                    bctx->last_dst_va = bctx->q8_1_scratch->GetGPUVirtualAddress();

                    const uint32_t q_groups_x = std::min(num_q8_blocks, 65535u);
                    const uint32_t q_groups_y = (num_q8_blocks + 65534u) / 65535u;
                    bctx->cmd_list->Dispatch(q_groups_x, q_groups_y, 1);

                    bctx->emit_uav_barrier_buffer(bctx->q8_1_scratch.Get());

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
                bctx->set_shader_params(params, num_constants);
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
                                             key.flags == 27 || key.flags == 31 || key.flags == 32 ||
                                             key.flags == 33 || key.flags == 34 || key.flags == 35 ||
                                             key.flags == 36 || key.flags == 37 || key.flags == 38 ||
                                             key.flags == 44 || key.flags == 47 || key.flags == 48 ||
                                             key.flags == 49 || key.flags == 50 || key.flags == 51 ||
                                             key.flags == 52 ||
                                             key.flags == 55 || key.flags == 56 || key.flags == 57 ||
                                            key.flags == 61 || key.flags == 67 || key.flags == 72 ||
                                            key.flags == 73 || key.flags == 74 || key.flags == 78 ||
                                            key.flags == 82) ? 2 :
                                            (key.flags == 28 || key.flags == 29 || key.flags == 45 ||
                                             key.flags == 46) ? 4 : 1;
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
                } else if (key.flags == 24 || key.flags == 31 || key.flags == 32 ||
                           key.flags == 33 || key.flags == 35 || key.flags == 62 ||
                           key.flags == 73 || key.flags == 74) {
                    params.op_params[1] = src2_offset_base + base_row * params.nb01;
                }
                params.op_params[15] = 0;
                bctx->set_shader_params(params, (uint32_t)(sizeof(params) / 4));
                bctx->cmd_list->Dispatch(chunk_groups, 1, groups_z);
            }
        } else if (!is_matvec_dispatch && key.flags == 43 &&
                   node->op == GGML_OP_MUL_MAT && node->src[0]) {
            // The IQ batch kernel is one thread per output element, each
            // walking the whole K with on-the-fly codebook dequant. A single
            // dispatch over a large output therefore runs for seconds and
            // trips the Windows TDR, which surfaces as DXGI_ERROR_DEVICE_REMOVED
            // (observed on 1B-class IQ models at pp512, where the output
            // projection alone is ~1e11 MACs). Split by output row using the
            // same local-row-range trick as the matvec chunking above so the
            // shaders keep their normal indexing, bounding MACs per dispatch.
            //
            // Chunk size is a throughput knob as well as a safety one: the
            // kernel re-reads the activation column per output element, so
            // smaller chunks keep that working set resident. Measured pp512 on
            // RX 6800 across 4M..512M MACs peaks at 16M for both Qwen3.5-0.8B
            // IQ3_XXS (146 vs 34 t/s at 512M) and Llama-3.2-1B IQ2_M (74 vs
            // 14). Override with DX12_IQ_CHUNK_MACS.
            static const uint64_t iq_chunk_macs = [] {
                const char * env = DX12_GETENV("DX12_IQ_CHUNK_MACS");
                return env ? std::max<uint64_t>(1, strtoull(env, nullptr, 10))
                           : (1ull << 24);
            }();
            const uint64_t cols = (uint64_t)node->ne[1] * node->ne[2] * node->ne[3];
            const uint64_t K    = (uint64_t)node->src[0]->ne[0];
            const uint64_t per_row = std::max<uint64_t>(1, cols * K);
            const uint32_t full_ne0 = params.ne0;
            const uint32_t rows_per_chunk = (uint32_t)std::min<uint64_t>(
                std::max<uint64_t>(1, full_ne0),
                std::max<uint64_t>(1, iq_chunk_macs / per_row));
            const uint32_t src0_offset_base = params.src0_offset;
            const uint32_t dst_offset_base  = params.dst_offset;
            for (uint32_t base_row = 0; base_row < full_ne0; base_row += rows_per_chunk) {
                const uint32_t chunk_rows = std::min(rows_per_chunk, full_ne0 - base_row);
                params.ne0         = chunk_rows;
                params.src0_offset = src0_offset_base + base_row * params.nb01;
                params.dst_offset  = dst_offset_base  + base_row * params.nb0;
                bctx->set_shader_params(params, (uint32_t)(sizeof(params) / 4));

                const uint64_t total = (uint64_t)chunk_rows * cols;
                const uint32_t chunk_groups = (uint32_t)((total + 255u) / 256u);
                uint32_t gx = chunk_groups;
                uint32_t gy = 1;
                if (chunk_groups > 65535) {
                    gx = 65535;
                    gy = (chunk_groups + 65534) / 65535;
                }
                bctx->cmd_list->Dispatch(gx, gy, 1);
            }
        } else {
            bctx->cmd_list->Dispatch(groups_x, groups_y, groups_z);
        }
        // RMS_NORM_MUL + Q8_1 fused: emit UAV barrier on q8_1_scratch and
        // pre-populate the q8_1 reuse cache so the downstream dp4a matmul
        // skips its own quantize_q8_1 dispatch. The barrier above protected
        // unsynced_writes via the F32 dst; the scratch (u1) also needs an
        // explicit ordering point before the dp4a matvec reads it.
        if (node->op == GGML_OP_RMS_NORM && key.flags == 12 &&
            fused_mul_node && bctx->q8_1_scratch) {
            bctx->emit_uav_barrier_buffer(bctx->q8_1_scratch.Get());

            uint32_t total_elems   = (uint32_t)ggml_nelements(fused_mul_node);
            uint32_t num_blocks    = total_elems / 32;
            size_t   q8_1_size     = (size_t)num_blocks * 36;
            bctx->last_q8_1_src_id  = (uintptr_t)fused_mul_node;
            bctx->last_q8_1_src_off = (uint32_t)dx12_tensor_offset(fused_mul_node);
            bctx->last_q8_1_size    = (uint32_t)q8_1_size;
            ID3D12Resource * fmul_res = dx12_get_resource(fused_mul_node);
            if (fmul_res) {
                bctx->last_q8_1_src_va = fmul_res->GetGPUVirtualAddress();
            }
            // Tell the next matmul's barrier-path cache-invalidation check
            // to skip its check on this cache entry (q8_1 bytes are fresh
            // and barrier-protected; F32 dst sharing the unsynced_writes
            // entry should not invalidate the q8_1 scratch).
            bctx->q8_1_cache_safe = true;
            bctx->dbg_q8_norm_prepop++;
        }

        // ARGSORT/TOP_K large-N multi-pass: the main Dispatch above ran INIT
        // (kind=0).  Now patch op_params[3..6] for each SWAP step and re-
        // dispatch, with a UAV barrier between dispatches to serialize scratch
        // writes.  Finally issue the WRITEOUT (kind=2 for ARGSORT, kind=3 for
        // TOP_K).  This shader reuses the same pipeline for all phases.
        if ((node->op == GGML_OP_ARGSORT || node->op == GGML_OP_TOP_K) &&
            key.flags == 50) {
            const uint32_t ncols        = params.op_params[1];
            const uint32_t ncols_padded = params.op_params[2];
            const uint32_t nrows        = groups_y;

            // Use explicit-resource UAV barriers (vs nullptr/global) - some
            // drivers (observed on Intel Arc B390 in test-backend-ops mixed
            // workloads) handle named-resource barriers more reliably than
            // global UAV barriers between many sequential dispatches.
            D3D12_RESOURCE_BARRIER uav_barrier = {};
            uav_barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav_barrier.UAV.pResource = bctx->dev->argsort_scratch.Get();

            const uint32_t half_n          = ncols_padded >> 1u;
            const uint32_t swap_groups_x   = (half_n + 255u) / 256u;

            // Bitonic sweep: outer step k = 2..ncols_padded, inner j = k/2 .. 1.
            params.op_params[3] = 1u; // kind = SWAP
            for (uint32_t k_step = 2u; k_step <= ncols_padded; k_step <<= 1u) {
                for (uint32_t j_step = k_step >> 1u; j_step > 0u; j_step >>= 1u) {
                    bctx->cmd_list->ResourceBarrier(1, &uav_barrier);
                    params.op_params[5] = k_step;
                    params.op_params[6] = j_step;
                    // Full param upload (works for both the CBV and root-constant
                    // slot-0 layouts; the base fields are unchanged from INIT).
                    bctx->set_shader_params(params, (uint32_t)(sizeof(params) / 4));
                    bctx->cmd_list->Dispatch(swap_groups_x, nrows, 1);
                }
            }

            // Final WRITEOUT phase.
            bctx->cmd_list->ResourceBarrier(1, &uav_barrier);
            if (node->op == GGML_OP_ARGSORT) {
                params.op_params[3] = 2u;
                bctx->set_shader_params(params, (uint32_t)(sizeof(params) / 4));
                bctx->cmd_list->Dispatch((ncols + 255u) / 256u, nrows, 1);
            } else {
                const uint32_t K = params.op_params[4];
                params.op_params[3] = 3u;
                bctx->set_shader_params(params, (uint32_t)(sizeof(params) / 4));
                bctx->cmd_list->Dispatch((K + 255u) / 256u, nrows, 1);
            }
        }

        // Split-KV reduction pass: combine partial results
        // op_params[15] is packed: low16 = n_splits, high16 = gqa_ratio
        if (node->op == GGML_OP_FLASH_ATTN_EXT && (params.op_params[15] & 0xFFFFu) > 1) {
            uint32_t n_splits = params.op_params[15] & 0xFFFFu;

            // UAV barrier between pass 1 and pass 2
            bctx->emit_uav_barrier_global();

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

        // Byte-range companion (DX12_PRECISE_KV_BARRIERS): record the memory
        // written so a later cross-view-root read/write can be detected by
        // overlap.  Mirrors the read tracking below.
        auto track_write_range = [&](const struct ggml_tensor * t) {
            if (!t || !t->data) return;
            uintptr_t wlo = (uintptr_t) t->data;
            unsynced_write_ranges.push_back({ (void *) dx12_get_resource(t), wlo,
                                              wlo + ggml_nbytes(t), tensor_root(t) });
        };
        track_write_range(dst_tensor);

        // Track this dispatch's reads for write-after-read hazard detection.
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
            const struct ggml_tensor * sr = node->src[s];
            if (!sr->data) continue;
            uintptr_t rlo = (uintptr_t) sr->data;
            unsynced_reads.push_back({ (void *) dx12_get_resource(sr), rlo,
                                       rlo + ggml_nbytes(sr), tensor_root(sr) });
        }
        if (fused_qk_postop) {
            const ggml_tensor * hidden_srcs[2] = {
                fused_qk_k_rope->src[0],
                fused_qk_k_set_rows->src[1],
            };
            for (const ggml_tensor * sr : hidden_srcs) {
                if (!sr || !sr->data) continue;
                uintptr_t rlo = (uintptr_t)sr->data;
                unsynced_reads.push_back({ (void *)dx12_get_resource(sr), rlo,
                                           rlo + ggml_nbytes(sr), tensor_root(sr) });
            }
        }

        // For triple fusion, also track the ADD intermediate output as unsynced
        if (fused_add_rms_node) {
            unsynced_writes.insert((uintptr_t)fused_add_rms_node);
            track_write_range(fused_add_rms_node);
        }

        // Combined Q/K/V dispatch writes three destinations (Q rope output tracked
        // above as dst_tensor). Track the K and V KV-cache scatters too so the
        // dependency-aware barrier fires before their downstream readers (the
        // FLASH_ATTN also always barriers, but keep tracking exact for relaxed-KV).
        if (fused_qkv) {
            unsynced_writes.insert((uintptr_t)fused_qkv_k_set_rows);
            unsynced_writes.insert((uintptr_t)fused_qkv_v_set_rows);
            track_write_range(fused_qkv_k_set_rows);
            track_write_range(fused_qkv_v_set_rows);
        }
        if (fused_qk_postop) {
            unsynced_writes.insert((uintptr_t)fused_qk_k_set_rows);
            track_write_range(fused_qk_k_set_rows);
        }

        // DX12_SYNC_PER_OP: close+execute+timed-wait after every node so a TDR
        // can be attributed to the exact op that caused it.  On timeout or
        // device-removed we dump op/flags/shape info and abort.  Very slow.
        if (sync_per_op) {
            bctx->close_and_execute();
            // Custom timed wait (wait_for_gpu uses INFINITE which would hang).
            const uint64_t target = bctx->fence_value;
            if (target > 0 && bctx->fence->GetCompletedValue() < target) {
                HRESULT hr = bctx->fence->SetEventOnCompletion(target, bctx->fence_event);
                DWORD wait_result = WAIT_TIMEOUT;
                if (SUCCEEDED(hr)) {
                    wait_result = WaitForSingleObject(bctx->fence_event, sync_per_op_ms);
                }
                HRESULT removed = bctx->dev->device->GetDeviceRemovedReason();
                if (wait_result != WAIT_OBJECT_0 || FAILED(removed)) {
                    const char * reason_str = "(unknown)";
                    if (removed == (HRESULT)0x887A0006) reason_str = "DEVICE_HUNG / TDR";
                    else if (removed == (HRESULT)0x887A0005) reason_str = "DEVICE_REMOVED";
                    else if (removed == (HRESULT)0x887A0007) reason_str = "DEVICE_RESET";
                    else if (removed == (HRESULT)0x887A0020) reason_str = "DRIVER_INTERNAL_ERROR";
                    else if (removed == (HRESULT)0x80070057) reason_str = "E_INVALIDARG";
                    else if (SUCCEEDED(removed)) reason_str = "(device still alive — wait timed out)";
                    int s0t = node->src[0] ? (int)node->src[0]->type : -1;
                    int s1t = node->src[1] ? (int)node->src[1]->type : -1;
                    fprintf(stderr,
                        "[DX12_SYNC_PER_OP] HANG at node %d/%d: op=%s name=%s flags=%u\n"
                        "  dst type=%d shape=[%lld,%lld,%lld,%lld]\n"
                        "  src0 type=%d shape=[%lld,%lld,%lld,%lld]\n"
                        "  src1 type=%d shape=[%lld,%lld,%lld,%lld]\n"
                        "  wait_result=0x%08X (target_fence=%llu completed=%llu) DeviceRemovedReason=0x%08X %s\n",
                        i, cgraph->n_nodes, ggml_op_name(node->op), node->name, (unsigned)key.flags,
                        (int)node->type,
                        (long long)node->ne[0], (long long)node->ne[1],
                        (long long)node->ne[2], (long long)node->ne[3],
                        s0t,
                        node->src[0] ? (long long)node->src[0]->ne[0] : 0,
                        node->src[0] ? (long long)node->src[0]->ne[1] : 0,
                        node->src[0] ? (long long)node->src[0]->ne[2] : 0,
                        node->src[0] ? (long long)node->src[0]->ne[3] : 0,
                        s1t,
                        node->src[1] ? (long long)node->src[1]->ne[0] : 0,
                        node->src[1] ? (long long)node->src[1]->ne[1] : 0,
                        node->src[1] ? (long long)node->src[1]->ne[2] : 0,
                        node->src[1] ? (long long)node->src[1]->ne[3] : 0,
                        (unsigned)wait_result,
                        (unsigned long long)target,
                        (unsigned long long)bctx->fence->GetCompletedValue(),
                        (unsigned)removed, reason_str);
                    fflush(stderr);
                    GGML_ABORT("DX12 hang pinpointed via DX12_SYNC_PER_OP");
                }
            }
            bctx->ensure_cmd_list_open();
            bctx->reset_binding_cache();
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
        if (fused_qk_postop) {
            i++;  // skip the adjacent Q SCALE node
        }
        if (fused_mmv_glu_glu) {
            i += 2;  // skip the gate matvec and SWIGLU split
        }
        if (fused_ssm_silu) {
            // SSM_CONV + (optional ADD) + UNARY(SILU): skip the SILU node, plus
            // the ADD node when bias is fused.
            i += fused_ssm_bias_add ? 2 : 1;
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
                uint32_t Nq   = (uint32_t)node->src[0]->ne[1];
                uint32_t Nkv  = (uint32_t)node->src[1]->ne[1];
                uint32_t nh   = (uint32_t)node->src[0]->ne[2];
                uint32_t nkvh = (uint32_t)node->src[1]->ne[2];
                snprintf(keybuf, sizeof(keybuf),
                         "%-13s fl=%2u D=%4u nq=%5u nh=%3u/%3u nkv=%5u grp=%u",
                         ggml_op_name(node->op), key.flags, K, Nq, nh, nkvh, Nkv, groups_x);
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
        // Pipelining: the cmd-list ring (see CMD_RING_MAX / DX12_CMD_RING)
        //   already provides natural backpressure — `ensure_cmd_list_open`
        //   waits on the slot it's about to reuse, so up to (ring - 1)
        //   submissions can be in flight at once.  This keeps the GPU
        //   saturated while the CPU records the next batch.  Previously we
        //   called `wait_for_gpu()`
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

            // Accumulate estimated FLOPs for adaptive streaming over ALL
            // compute-heavy nodes (matmul, conv, attention), matching Vulkan
            // (ggml-vulkan.cpp:16330).  Cheap ops contribute 0.
            {
                uint64_t nf = dx12_get_node_flops(node);
                batch_flops += nf;
                total_flops += nf;
            }

            bool needs_gen_flush = !is_prompt && cgraph->n_nodes > 500;
            int flush_threshold = is_prompt ? 24 : 2000;
            const bool flops_trigger = (flops_per_submit > 0 && batch_flops >= flops_per_submit);
            // During a command-list replay capture the whole graph is recorded
            // into a single dedicated list with no stream submits.
            if (bctx->replay.capturing) {
                // no-op: keep everything in the dedicated capture list
            } else if ((is_prompt || needs_gen_flush) && dispatch_weight >= flush_threshold) {
                bctx->close_and_execute();
                static const bool flush_drain = (getenv("DX12_FLUSH_DRAIN") != nullptr);
                if (flush_drain && is_prompt && total_groups >= 20000) {
                    bctx->wait_for_gpu();
                }
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                dispatch_weight = 0;
                stream_nodes = 0;
                batch_flops = 0;
                submit_count++;
                if (submit_count <= 3 && flops_per_submit > 0) flops_per_submit *= 2;
            } else if (flops_trigger ||
                       (stream_threshold > 0 && ++stream_nodes >= stream_threshold)) {
                // Stream-submit: kick the GPU so it can overlap with CPU recording.
                bctx->close_and_execute();
                bctx->ensure_cmd_list_open();
                bctx->reset_binding_cache();
                stream_nodes = 0;
                dispatch_weight = 0;
                batch_flops = 0;
                submit_count++;
                if (submit_count <= 3 && flops_per_submit > 0) flops_per_submit *= 2;
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
        if (!almost_ready_disabled && !bctx->replay.capturing &&
            !is_prompt && bctx->almost_ready_fence == 0 &&
            cgraph->n_nodes >= 80 &&
            (cgraph->n_nodes - i) <= almost_ready_remaining &&
            bctx->cmd_list_open) {
            bctx->close_and_execute();
            bctx->almost_ready_fence = bctx->fence_value;
            bctx->ensure_cmd_list_open();
            bctx->reset_binding_cache();
        }
        if (phase_profile) {
            bctx->phase_dispatch_us += dx12_qpc_us() - phase_detail_start_us;
        }
    }

    // Save grand total for next call's flops-per-submit threshold heuristic.
    bctx->last_total_flops = total_flops;
    if (!is_prompt) {
        bctx->last_decode_flops = total_flops;
    }
    if (DX12_GETENV("DX12_LOG_GRAPH_FLOPS") != nullptr) {
        fprintf(stderr, "[DX12_GRAPH_FLOPS] %s nodes=%d flops=%.4f G\n",
                is_prompt ? "prompt" : "decode", cgraph->n_nodes, (double)total_flops / 1e9);
        fflush(stderr);
    }

    // DX12_QUANT_STATS: report Q8_1 activation-prep dispatch inventory for this
    // graph (only for M=1 decode graphs, which is where the norm/activation
    // fusion applies). "dispatched" = standalone quantize_q8_1 kernels issued;
    // "reused" = skipped via the reuse cache; "norm_prepop" = filled directly
    // by a fused norm+quantize dispatch (flags=12/14).
    if (quant_stats && cgraph->n_nodes > 0 && cgraph->nodes[cgraph->n_nodes - 1] &&
        cgraph->nodes[cgraph->n_nodes - 1]->ne[1] == 1) {
        fprintf(stderr,
            "[DX12_QUANT_STATS] nodes=%d q8_1_quantize_dispatched=%llu reused=%llu norm_prepop=%llu\n",
            cgraph->n_nodes,
            (unsigned long long)bctx->dbg_q8_quant_dispatched,
            (unsigned long long)bctx->dbg_q8_quant_reused,
            (unsigned long long)bctx->dbg_q8_norm_prepop);
        fflush(stderr);
    }

    // DX12_BARRIER_STATS: report UAV barrier inventory for this graph, split
    // by scoped (resource-range) vs global (full drain).  With DX12_BARRIER_
    // ELISION on, a higher scoped:global ratio means less cross-dispatch
    // serialization; compare against a baseline run to gauge the win.
    if (barrier_stats) {
        fprintf(stderr,
            "[DX12_BARRIER_STATS] nodes=%d barriers_scoped=%llu barriers_global=%llu\n",
            cgraph->n_nodes,
            (unsigned long long)bctx->dbg_barrier_scoped,
            (unsigned long long)bctx->dbg_barrier_global);
        fflush(stderr);
    }
    // here so the ring-list keep-open / profiling / dump tails are skipped (none
    // of them apply to a capture, which is gated off when they are active).
    if (bctx->replay.capturing) {
        dx12_replay_finalize_capture(bctx, cgraph);
        dx12_replay_stats_dump(bctx);
        if (phase_profile) bctx->phase_graph_return_us = dx12_qpc_us();
        return GGML_STATUS_SUCCESS;
    }

    // Drain retired q8_1_scratch buffers that were displaced by reallocation.
    // They must outlive any in-flight dispatch that referenced their VA.
    // We close+wait here only if there's actually something to free.
    if (!bctx->q8_1_scratch_retired.empty() ||
        !bctx->dev->argsort_scratch_retired.empty()) {
        if (bctx->cmd_list_open) {
            bctx->close_and_execute();
        }
        bctx->wait_for_gpu();
        bctx->q8_1_scratch_retired.clear();
        bctx->dev->argsort_scratch_retired.clear();
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
            uint64_t lo = ~0ull, hi = 0;
            double sum_ticks = 0.0;
            for (size_t k = 0; k < prof_keys.size(); k++) {
                uint64_t t_start = ts[k * 2];
                uint64_t t_end   = ts[k * 2 + 1];
                if (t_end < t_start) continue;
                double ms = (double)(t_end - t_start) * 1000.0 / (double)prof_freq;
                op_times[prof_keys[k]] += ms;
                op_counts[prof_keys[k]] += 1;
                if (t_start < lo) lo = t_start;
                if (t_end   > hi) hi = t_end;
                sum_ticks += (double)(t_end - t_start);
            }
            // span - sum exposes GPU idle *between* dispatches within one graph,
            // which busy-percentage counters cannot show.
            if (hi > lo) {
                const double to_ms = 1000.0 / (double)prof_freq;
                fprintf(stderr, "[GPU_SPAN] dispatches=%zu sum=%.3f ms span=%.3f ms idle=%.3f ms\n",
                        prof_keys.size(), sum_ticks * to_ms, (double)(hi - lo) * to_ms,
                        ((double)(hi - lo) - sum_ticks) * to_ms);
            }
            D3D12_RANGE wr = { 0, 0 };
            prof_readback->Unmap(0, &wr);
        }
    }
    if (do_profile && !op_times.empty() && tune_profile_active) {
        // JSON-lines append: one line per graph_compute call. Each line is a JSON
        // object: {"graph": <n>, "ops": [{"key":"<keystr>", "ms": <total_ms>,
        // "count": <n>}, ...]}. Aggregation across reps is the caller's job.
        FILE * jf = fopen(tune_profile_json, "ab");
        if (jf) {
            fprintf(jf, "{\"graph\":%d,\"ops\":[", profile_graph);
            bool first = true;
            for (auto & kv : op_times) {
                if (!first) fprintf(jf, ",");
                first = false;
                // simple JSON-string escape (keys are well-formed printf output;
                // only need to escape backslash and quote).
                fprintf(jf, "{\"key\":\"");
                for (char c : kv.first) {
                    if (c == '\\' || c == '"') fputc('\\', jf);
                    fputc(c, jf);
                }
                fprintf(jf, "\",\"ms\":%.6f,\"count\":%u}", kv.second, op_counts[kv.first]);
            }
            fprintf(jf, "]}\n");
            fclose(jf);
        }
    } else if (do_profile && !op_times.empty()) {
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

    if (phase_profile) {
        bctx->phase_graph_return_us = dx12_qpc_us();
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Auto-tuning: benchmark shader variants and pick the fastest per GPU
// ---------------------------------------------------------------------------

void dx12_device::run_autotune() {
    if (tuning_done) return;
    tuning_done = true;

    // Tuner override path: when these envs are set, skip cache+benchmark and use
    // the supplied values directly. Used by llama-mmv-tune validate-cache to
    // measure how each cached choice performs on real model shapes vs the
    // alternative. Each env is optional; unset envs fall through to cache load.
    const char * env_q4k    = getenv("DX12_TUNE_FORCE_Q4K_DP4A_32");
    const char * env_q5k    = getenv("DX12_TUNE_FORCE_Q5K_DP4A_32");
    const char * env_f16    = getenv("DX12_TUNE_FORCE_F16_MR_256");
    const char * env_kthr   = getenv("DX12_TUNE_FORCE_F16_MR_K_THRESH");
    const char * env_q5kmth = getenv("DX12_TUNE_FORCE_Q5K_DP4A_M_THRESH");
    if (env_q4k && env_q5k && env_f16 && env_kthr) {
        q4k_dp4a_use_32        = (atoi(env_q4k) != 0);
        q5k_dp4a_use_32        = (atoi(env_q5k) != 0);
        f16_mr_use_256         = (atoi(env_f16) != 0);
        f16_mr_k_256_threshold = (uint32_t) strtoul(env_kthr, nullptr, 10);
        // M-threshold env is optional; default UINT32_MAX preserves old behavior.
        q5k_dp4a_m_32_threshold = env_q5kmth
            ? (uint32_t) strtoul(env_q5kmth, nullptr, 10)
            : 0xFFFFFFFFu;
        DX12_LOG_INFO("Auto-tune FORCED via env: Q4_K_dp4a=%s Q5_K_dp4a=%s F16_mr=%s (K>=%u uses 256t, Q5K M>=%u uses 32t)\n",
                      q4k_dp4a_use_32 ? "32t" : "256t",
                      q5k_dp4a_use_32 ? "32t" : "256t",
                      f16_mr_use_256  ? "256t" : "32t",
                      (unsigned)f16_mr_k_256_threshold,
                      (unsigned)q5k_dp4a_m_32_threshold);
        return;
    }

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
        unsigned int q5k_m_thresh  = 0xFFFFFFFFu;
        if (fscanf(f, "v=%d q4k_dp4a_32=%d q5k_dp4a_32=%d f16_mr_256=%d f16_mr_k_thresh=%u q5k_dp4a_m_thresh=%u",
                   &ver, &q4kdp, &q5kdp, &f16mr256, &f16mr_kthresh, &q5k_m_thresh) == 6
            && ver == TUNE_VERSION) {
            q4k_dp4a_use_32 = (q4kdp != 0);
            q5k_dp4a_use_32 = (q5kdp != 0);
            f16_mr_use_256  = (f16mr256 != 0);
            f16_mr_k_256_threshold = (uint32_t)f16mr_kthresh;
            q5k_dp4a_m_32_threshold = (uint32_t)q5k_m_thresh;
            fclose(f);
            DX12_LOG_INFO("Auto-tune v%d loaded: Q4_K_dp4a=%s Q5_K_dp4a=%s F16_mr=%s (K>=%u uses 256t, Q5K M>=%u uses 32t)\n",
                          ver,
                          q4k_dp4a_use_32 ? "32t" : "256t",
                          q5k_dp4a_use_32 ? "32t" : "256t",
                          f16_mr_use_256  ? "256t" : "32t",
                          (unsigned)f16_mr_k_256_threshold,
                          (unsigned)q5k_dp4a_m_32_threshold);
            return;
        }
        fclose(f);
        // Version mismatch or parse failure — re-benchmark
    }

    DX12_LOG_INFO("Running auto-tune benchmark...\n");

    // Create a temporary buffer for benchmarking
    // Must be large enough for max test: K-sweep N=256 rows × max K stride,
    // plus M-sweep at M=test_M_large × K=test_K_for_m with fake byte strides
    // (nb01 = K). At M=32768, K=3072 that's 32768 * 3072 = 96 MB for src0.
    // 256 MB gives headroom for any future probe expansion.
    ComPtr<ID3D12Resource> bench_buf;
    {
        D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 256ull * 1024 * 1024;  // 256MB
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
        ComPtr<ID3D12Resource> param_buffer;
        uint8_t * param_mapped = nullptr;
        if (use_param_cbv) {
            D3D12_HEAP_PROPERTIES php = {};
            php.Type = D3D12_HEAP_TYPE_UPLOAD;
            D3D12_RESOURCE_DESC prd = {};
            prd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            prd.Width = 256;
            prd.Height = 1;
            prd.DepthOrArraySize = 1;
            prd.MipLevels = 1;
            prd.Format = DXGI_FORMAT_UNKNOWN;
            prd.SampleDesc.Count = 1;
            prd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            HRESULT phr = device->CreateCommittedResource(
                &php, D3D12_HEAP_FLAG_NONE, &prd, D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr, IID_PPV_ARGS(&param_buffer));
            if (FAILED(phr)) return UINT64_MAX;
            D3D12_RANGE read_range = { 0, 0 };
            phr = param_buffer->Map(0, &read_range, (void **) &param_mapped);
            if (FAILED(phr)) return UINT64_MAX;
            memcpy(param_mapped, &params, sizeof(params));
            cl->SetComputeRootConstantBufferView(0, param_buffer->GetGPUVirtualAddress());
        } else {
            cl->SetComputeRoot32BitConstants(0, (uint32_t)(sizeof(params)/4), &params, 0);
        }

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
            if (param_buffer && param_mapped) param_buffer->Unmap(0, nullptr);
            return UINT64_MAX;
        }

        // Read timestamps
        uint64_t * ts = nullptr;
        D3D12_RANGE range = { ts_start * sizeof(uint64_t), (ts_start + 2) * sizeof(uint64_t) };
        ts_readback->Map(0, &range, (void**)&ts);
        uint64_t dt = ts[ts_start + 1] - ts[ts_start];
        ts_readback->Unmap(0, nullptr);
        if (param_buffer && param_mapped) param_buffer->Unmap(0, nullptr);
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

    // Benchmark Q5_K dp4a matvec: 256 threads (default) vs 32 threads.
    // The K-sweep at fixed test_N feeds the global "always use 32t" fallback
    // (used when the M-aware crossover degenerates to one variant winning at
    // both endpoints). A separate M-sweep below (at fixed K) sets the
    // q5k_dp4a_m_32_threshold for per-dispatch M-aware routing.
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

    // Q5_K M-aware sweep at fixed K. Determines q5k_dp4a_m_32_threshold by
    // measuring 256t vs 32t at three M points (test_N, test_M_mid,
    // test_M_large) at a representative K. Three points let us detect a
    // crossover in either of two adjacent sub-ranges without losing the
    // small-M resolution we'd give up by simply pushing the large endpoint
    // out. Interpolation rule: walk lo->mid then mid->hi looking for the
    // first 32t-wins transition; linearly interpolate the crossover M inside
    // that bracket. Non-monotone responses (32t wins only in the middle, or
    // wins at one endpoint and loses at the other two) fall back to the safe
    // "never 32t" default so we don't enable the opt-in shader on
    // unpredictable shapes.
    uint64_t q5k_m_lo [2] = { UINT64_MAX, UINT64_MAX };  // [256t, 32t] at M=test_N
    uint64_t q5k_m_mid[2] = { UINT64_MAX, UINT64_MAX };  // [256t, 32t] at M=test_M_mid
    uint64_t q5k_m_hi [2] = { UINT64_MAX, UINT64_MAX };  // [256t, 32t] at M=test_M_large
    constexpr uint32_t test_K_for_m  = 3072;
    constexpr uint32_t test_M_mid    = 4096;
    constexpr uint32_t test_M_large  = 32768;
    if (dp4a_supported) {
        dx12_pipeline_key key256 = {}; key256.op = GGML_OP_MUL_MAT; key256.src0_type = GGML_TYPE_Q5_K; key256.flags = 14;
        dx12_pipeline_key key32  = {}; key32.op  = GGML_OP_MUL_MAT; key32.src0_type  = GGML_TYPE_Q5_K; key32.flags  = 16;

        // M=test_N point: reuse the corresponding result from q5k_per_k if
        // we have it (test_K_for_m == one of the test_K entries), otherwise
        // re-bench. test_K = {576, 3072, 8192} so K=3072 hits index 1.
        size_t k_idx = NK;
        for (size_t ki = 0; ki < NK; ++ki) if (test_K[ki] == test_K_for_m) { k_idx = ki; break; }
        if (k_idx < NK) {
            q5k_m_lo[0] = q5k_per_k[k_idx*2 + 0];
            q5k_m_lo[1] = q5k_per_k[k_idx*2 + 1];
        } else {
            q5k_m_lo[0] = bench_pipeline(key256, test_K_for_m, test_N, 0);
            q5k_m_lo[1] = bench_pipeline(key32,  test_K_for_m, test_N, 2);
        }

        // M=test_M_mid and M=test_M_large points: fresh benches
        q5k_m_mid[0] = bench_pipeline(key256, test_K_for_m, test_M_mid,   0);
        q5k_m_mid[1] = bench_pipeline(key32,  test_K_for_m, test_M_mid,   2);
        q5k_m_hi [0] = bench_pipeline(key256, test_K_for_m, test_M_large, 0);
        q5k_m_hi [1] = bench_pipeline(key32,  test_K_for_m, test_M_large, 2);

        DX12_LOG_INFO("  Q5_K_dp4a K=%u M=%u: 256t=%llu 32t=%llu ticks\n",
                      test_K_for_m, test_N,
                      (unsigned long long)q5k_m_lo[0], (unsigned long long)q5k_m_lo[1]);
        DX12_LOG_INFO("  Q5_K_dp4a K=%u M=%u: 256t=%llu 32t=%llu ticks\n",
                      test_K_for_m, test_M_mid,
                      (unsigned long long)q5k_m_mid[0], (unsigned long long)q5k_m_mid[1]);
        DX12_LOG_INFO("  Q5_K_dp4a K=%u M=%u: 256t=%llu 32t=%llu ticks\n",
                      test_K_for_m, test_M_large,
                      (unsigned long long)q5k_m_hi[0], (unsigned long long)q5k_m_hi[1]);

        // Helper: linearly interpolate the M at which 256t and 32t cost
        // the same, between two (M, t256, t32) points. Direction-agnostic:
        // works whether 32t wins at the smaller M (rare) or at the larger M
        // (typical Q5_K case). Models the cost gap (T32 - T256) as linear
        // in M and solves for the M where gap == 0. Returns rounded M.
        auto interp_crossover = [](double Ma, uint64_t a256, uint64_t a32,
                                   double Mb, uint64_t b256, uint64_t b32) -> uint32_t {
            double gap_a = (double)a32 - (double)a256;  // > 0 means 256t wins at Ma
            double gap_b = (double)b32 - (double)b256;  // > 0 means 256t wins at Mb
            double dg = gap_a - gap_b;                  // change in gap across the interval
            if (dg == 0.0) return (uint32_t)((Ma + Mb) * 0.5 + 0.5);  // parallel; midpoint
            double t = gap_a / dg;                      // fraction of [Ma, Mb] at zero-crossing
            if (t < 0.0) t = 0.0;
            if (t > 1.0) t = 1.0;
            double M_cross = Ma + t * (Mb - Ma);
            return (uint32_t)(M_cross + 0.5);
        };

        q5k_dp4a_m_32_threshold = 0xFFFFFFFFu;
        if (q5k_m_lo[0] != UINT64_MAX && q5k_m_lo[1] != UINT64_MAX &&
            q5k_m_mid[0] != UINT64_MAX && q5k_m_mid[1] != UINT64_MAX &&
            q5k_m_hi[0] != UINT64_MAX && q5k_m_hi[1] != UINT64_MAX) {
            bool lo_32_wins  = (q5k_m_lo[1]  < q5k_m_lo[0]);
            bool mid_32_wins = (q5k_m_mid[1] < q5k_m_mid[0]);
            bool hi_32_wins  = (q5k_m_hi[1]  < q5k_m_hi[0]);

            if (lo_32_wins && mid_32_wins && hi_32_wins) {
                q5k_dp4a_m_32_threshold = 0;                // always use 32t
            } else if (!lo_32_wins && !mid_32_wins && !hi_32_wins) {
                q5k_dp4a_m_32_threshold = 0xFFFFFFFFu;      // never use 32t
            } else if (!lo_32_wins && !mid_32_wins && hi_32_wins) {
                // monotone crossover in [mid, hi]
                q5k_dp4a_m_32_threshold = interp_crossover(
                    (double)test_M_mid,   q5k_m_mid[0], q5k_m_mid[1],
                    (double)test_M_large, q5k_m_hi[0],  q5k_m_hi[1]);
            } else if (!lo_32_wins && mid_32_wins && hi_32_wins) {
                // monotone crossover in [lo, mid]
                q5k_dp4a_m_32_threshold = interp_crossover(
                    (double)test_N,     q5k_m_lo[0],  q5k_m_lo[1],
                    (double)test_M_mid, q5k_m_mid[0], q5k_m_mid[1]);
            } else {
                // Non-monotone (e.g. 010, 100, 101, 110): treat as unreliable
                // signal and keep the safe default of never using 32t.
                q5k_dp4a_m_32_threshold = 0xFFFFFFFFu;
            }
        }
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

    DX12_LOG_INFO("Auto-tune result: Q4_K_dp4a=%s Q5_K_dp4a=%s F16_mr=%s (K>=%u uses 256t, Q5K M>=%u uses 32t)\n",
                  q4k_dp4a_use_32 ? "32t" : "256t",
                  q5k_dp4a_use_32 ? "32t" : "256t",
                  f16_mr_use_256  ? "256t" : "32t",
                  (unsigned)f16_mr_k_256_threshold,
                  (unsigned)q5k_dp4a_m_32_threshold);

    // Save to cache (with per-K diagnostic comments after the result line)
    f = fopen(cache_path, "w");
    if (f) {
        fprintf(f, "v=%d q4k_dp4a_32=%d q5k_dp4a_32=%d f16_mr_256=%d f16_mr_k_thresh=%u q5k_dp4a_m_thresh=%u\n",
                TUNE_VERSION,
                q4k_dp4a_use_32 ? 1 : 0,
                q5k_dp4a_use_32 ? 1 : 0,
                f16_mr_use_256  ? 1 : 0,
                (unsigned)f16_mr_k_256_threshold,
                (unsigned)q5k_dp4a_m_32_threshold);
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
        fprintf(f, "# Q5_K_dp4a K=%u M=%u: 256t=%llu 32t=%llu ticks\n",
                test_K_for_m, test_N,
                (unsigned long long)q5k_m_lo[0], (unsigned long long)q5k_m_lo[1]);
        fprintf(f, "# Q5_K_dp4a K=%u M=%u: 256t=%llu 32t=%llu ticks\n",
                test_K_for_m, test_M_mid,
                (unsigned long long)q5k_m_mid[0], (unsigned long long)q5k_m_mid[1]);
        fprintf(f, "# Q5_K_dp4a K=%u M=%u: 256t=%llu 32t=%llu ticks\n",
                test_K_for_m, test_M_large,
                (unsigned long long)q5k_m_hi[0], (unsigned long long)q5k_m_hi[1]);
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
    const bool phase_profile = DX12_GETENV("DX12_PHASE_PROFILE") != nullptr;
    const uint64_t sync_entry_us = phase_profile ? dx12_qpc_us() : 0;
    if (phase_profile) ctx->phase_sync_calls++;

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
    const uint64_t wait_start_us = phase_profile ? dx12_qpc_us() : 0;
    ctx->wait_for_fence(ctx->fence_value);
    const uint64_t wait_end_us = phase_profile ? dx12_qpc_us() : 0;
    ctx->almost_ready_fence = 0;

    if (phase_profile && !ctx->phase_is_prompt && ctx->phase_pending &&
        ctx->phase_graph_start_us != 0 && ctx->phase_record_start_us != 0 &&
        ctx->phase_graph_return_us >= ctx->phase_record_start_us) {
        ctx->phase_pending = false;
        ctx->phase_gap_sync_accounted = true;
        const uint64_t prep_us = ctx->phase_record_start_us - ctx->phase_graph_start_us;
        const uint64_t record_span_us = ctx->phase_graph_return_us - ctx->phase_record_start_us;
        const uint64_t record_us = record_span_us > ctx->phase_submit_record_us
            ? record_span_us - ctx->phase_submit_record_us : 0;
        const uint64_t post_graph_us = sync_entry_us >= ctx->phase_graph_return_us
            ? sync_entry_us - ctx->phase_graph_return_us : 0;
        const uint64_t wait_us = wait_end_us - wait_start_us;
        const uint64_t total_us = wait_end_us - ctx->phase_graph_start_us;

        ctx->phase_decode_count++;
        ctx->phase_sum_prep_us   += prep_us;
        ctx->phase_sum_record_us += record_us;
        ctx->phase_sum_submit_us += ctx->phase_submit_us;
        ctx->phase_sum_wait_us   += wait_us;
        ctx->phase_sum_total_us  += total_us;
        ctx->phase_sum_post_graph_us += post_graph_us;
        ctx->phase_sum_get_tensor_us += ctx->phase_get_tensor_us;
        ctx->phase_sum_alloc_wait_us += ctx->phase_alloc_wait_us;
        ctx->phase_sum_alloc_wait_post_us += ctx->phase_alloc_wait_post_us;
        ctx->phase_sum_first_submit_us += ctx->phase_first_submit_us;
        if (ctx->phase_last_sync_end_us != 0 &&
            ctx->phase_graph_start_us > ctx->phase_last_sync_end_us) {
            ctx->phase_sum_gap_us += ctx->phase_graph_start_us - ctx->phase_last_sync_end_us;
        }
        ctx->phase_last_sync_end_us = wait_end_us;
        ctx->phase_sum_decision_us += ctx->phase_decision_us;
        ctx->phase_sum_params_us   += ctx->phase_params_us;
        ctx->phase_sum_setup_us    += ctx->phase_setup_us;
        ctx->phase_sum_barrier_us  += ctx->phase_barrier_us;
        ctx->phase_sum_dispatch_us += ctx->phase_dispatch_us;

        if (ctx->phase_decode_count <= 8 || (ctx->phase_decode_count % 32) == 0) {
            const double count = (double)ctx->phase_decode_count;
            fprintf(stderr,
                    "[DX12_PHASE] n=%llu current_us prep=%llu record=%llu submit=%llu wait=%llu total=%llu "
                    "avg_us prep=%.1f record=%.1f submit=%.1f wait=%.1f total=%.1f\n",
                    (unsigned long long)ctx->phase_decode_count,
                    (unsigned long long)prep_us,
                    (unsigned long long)record_us,
                    (unsigned long long)ctx->phase_submit_us,
                    (unsigned long long)wait_us,
                    (unsigned long long)total_us,
                    ctx->phase_sum_prep_us / count,
                    ctx->phase_sum_record_us / count,
                    ctx->phase_sum_submit_us / count,
                    ctx->phase_sum_wait_us / count,
                    ctx->phase_sum_total_us / count);
            fprintf(stderr,
                    "[DX12_PHASE_DETAIL] current_us decision=%llu params=%llu setup=%llu barrier=%llu dispatch=%llu "
                    "avg_us decision=%.1f params=%.1f setup=%.1f barrier=%.1f dispatch=%.1f first_submit=%.1f\n",
                    (unsigned long long)ctx->phase_decision_us,
                    (unsigned long long)ctx->phase_params_us,
                    (unsigned long long)ctx->phase_setup_us,
                    (unsigned long long)ctx->phase_barrier_us,
                    (unsigned long long)ctx->phase_dispatch_us,
                    ctx->phase_sum_decision_us / count,
                    ctx->phase_sum_params_us / count,
                    ctx->phase_sum_setup_us / count,
                    ctx->phase_sum_barrier_us / count,
                    ctx->phase_sum_dispatch_us / count,
                    ctx->phase_sum_first_submit_us / count);
            fprintf(stderr,
                    "[DX12_PHASE_HOST] current_us post_graph=%llu get_tensor=%llu alloc_wait=%llu alloc_wait_post=%llu "
                    "avg_us post_graph=%.1f get_tensor=%.1f alloc_wait=%.1f alloc_wait_post=%.1f syncs_per_graph=%.1f gap=%.1f supports_op=%.1f gapsync=%.1f settensor=%.1f bufset=%.1f bufset_calls=%.1f\n",
                    (unsigned long long)post_graph_us,
                    (unsigned long long)ctx->phase_get_tensor_us,
                    (unsigned long long)ctx->phase_alloc_wait_us,
                    (unsigned long long)ctx->phase_alloc_wait_post_us,
                    ctx->phase_sum_post_graph_us / count,
                    ctx->phase_sum_get_tensor_us / count,
                    ctx->phase_sum_alloc_wait_us / count,
                    ctx->phase_sum_alloc_wait_post_us / count,
                    ctx->phase_sync_calls / count,
                    ctx->phase_sum_gap_us / count,
                    g_dx12_supports_op_us / count,
                    ctx->phase_sum_gapsync_us / count,
                    ctx->phase_sum_settensor_us / count,
                    g_dx12_buf_set_us / count,
                    g_dx12_buf_set_calls / count);
            fflush(stderr);
        }
    }

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

    if (phase_profile && !ctx->phase_gap_sync_accounted) {
        ctx->phase_sum_gapsync_us += dx12_qpc_us() - sync_entry_us;
    }
    ctx->phase_gap_sync_accounted = false;

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

    struct dx12_set_timer {
        dx12_backend_context * c;
        uint64_t t0;
        bool on;
        ~dx12_set_timer() { if (on) c->phase_sum_settensor_us += dx12_qpc_us() - t0; }
    } set_timer { ctx, 0, DX12_GETENV("DX12_PHASE_PROFILE") != nullptr };
    if (set_timer.on) set_timer.t0 = dx12_qpc_us();

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
    // xfer.fence_value indexes the device xfer fence -- a separate timeline from
    // the backend fence, so it must advance monotonically or xfer_wait() can
    // return early and let the next upload clobber in-flight staging.
    ctx->dev->xfer.fence_value++;
    ctx->dev->compute_queue->Signal(ctx->dev->xfer.fence.Get(), ctx->dev->xfer.fence_value);
    // Don't re-open cmd_list yet; subsequent record will reopen lazily.
}

static void dx12_backend_get_tensor_async(ggml_backend_t backend,
                                          const ggml_tensor * tensor,
                                          void * data,
                                          size_t offset, size_t size) {
    auto * ctx = (dx12_backend_context *)backend->context;
    if (size == 0) return;
    const bool phase_profile = DX12_GETENV("DX12_PHASE_PROFILE") != nullptr;
    const uint64_t get_start_us = phase_profile ? dx12_qpc_us() : 0;
    const uint64_t submit_start_us = phase_profile ? ctx->phase_submit_us : 0;
    const uint64_t alloc_wait_start_us = phase_profile ? ctx->phase_alloc_wait_us : 0;
    auto record_get_tensor_time = [&]() {
        if (!phase_profile) return;
        const uint64_t elapsed = dx12_qpc_us() - get_start_us;
        const uint64_t nested_submit = ctx->phase_submit_us - submit_start_us;
        const uint64_t nested_alloc_wait = ctx->phase_alloc_wait_us - alloc_wait_start_us;
        const uint64_t nested = nested_submit + nested_alloc_wait;
        ctx->phase_get_tensor_us += elapsed > nested ? elapsed - nested : 0;
    };

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
        record_get_tensor_time();
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
    record_get_tensor_time();
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
    auto * d = (dx12_device *)dev->context;
    return d->is_igpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void dx12_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    auto * d = (dx12_device *)dev->context;
    props->name         = d->name.c_str();
    props->description  = d->name.c_str();
    props->memory_free  = d->vram_free;
    props->memory_total = d->vram_total;
    props->type         = d->is_igpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id    = d->device_id_str.c_str();
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

    // Whole-graph command-list replay (see dx12_cmd_replay).  Default on, but
    // dx12_graph_compute additionally gates it by decode graph size on discrete
    // GPUs - see the DX12_REPLAY_MIN_GFLOP logic there.  It only has an effect
    // when the CBV root-param layout is active, which
    // create_common_root_signature auto-enables for the same env flag.
    ctx->replay.enabled = dx12_flag_default_on("DX12_COMMAND_REPLAY");
    ctx->replay.stats   = (getenv("DX12_COMMAND_REPLAY_STATS") != nullptr);

    // Command-allocator ring depth (see CMD_RING_MAX). Deeper rings let the CPU
    // run further ahead of the GPU before it has to block recycling a slot.
    if (const char * ring_env = getenv("DX12_CMD_RING")) {
        const int want = atoi(ring_env);
        if (want >= 2 && want <= CMD_RING_MAX) {
            ctx->cmd_ring_size = want;
        }
    }

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
            { GGML_OP_LEAKY_RELU,    WB(leaky_relu, WS)    }, \
            { GGML_OP_FILL,          WB(fill, WS)          }, \
            { GGML_OP_TRI,           WB(tri, WS)           }, \
            { GGML_OP_DIAG,          WB(diag, WS)          }, \
            { GGML_OP_ARANGE,        WB(arange, WS)        }, \
            { GGML_OP_TIMESTEP_EMBEDDING, WB(timestep_embedding, WS) }, \
            { GGML_OP_SUM,           WB(sum, WS)           }, \
            { GGML_OP_MEAN,          WB(mean, WS)          }, \
            { GGML_OP_ARGMAX,        WB(argmax, WS)        }, \
            { GGML_OP_ARGSORT,       WB(argsort, WS)       }, \
            { GGML_OP_TOP_K,         WB(top_k, WS)         }, \
            { GGML_OP_ADD_ID,        WB(add_id, WS)        }, \
            { GGML_OP_COUNT_EQUAL,   WB(count_equal, WS)   }, \
            { GGML_OP_ACC,           WB(acc, WS)           }, \
            { GGML_OP_SET,           WB(set, WS)           }, \
            { GGML_OP_CUMSUM,        WB(cumsum, WS)        }, \
            { GGML_OP_SOLVE_TRI,     WB(solve_tri, WS)     }, \
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
            { GGML_OP_IM2COL_3D,     WB(im2col_3d, WS)     }, \
            { GGML_OP_POOL_2D,       WB(pool_2d, WS)       }, \
            { GGML_OP_POOL_1D,       WB(pool_1d, WS)       }, \
            { GGML_OP_CONV_2D,       WB(conv_2d, WS)       }, \
            { GGML_OP_CONV_2D_DW,    WB(conv_2d_dw, WS)    }, \
            { GGML_OP_CONV_3D,       WB(conv_3d, WS)       }, \
            { GGML_OP_CONV_TRANSPOSE_1D, WB(conv_transpose_1d, WS) }, \
            { GGML_OP_CONV_TRANSPOSE_2D, WB(conv_transpose_2d, WS) }, \
            { GGML_OP_FLASH_ATTN_EXT,WB_FP16(flash_attn, WS)    }, \
            { GGML_OP_SET_ROWS,      WB(set_rows, WS)      }, \
            { GGML_OP_GLU,           WB(glu, WS)           }, \
            { GGML_OP_L2_NORM,       WB(l2_norm, WS)       }, \
            { GGML_OP_GATED_DELTA_NET, WB(gated_delta_net, WS) }, \
            { GGML_OP_SSM_SCAN,      WB(ssm_scan, WS)      }, \
            { GGML_OP_RWKV_WKV6,     WB(wkv6, WS)          }, \
            { GGML_OP_RWKV_WKV7,     WB(wkv7, WS)          }, \
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
            { GGML_UNARY_OP_ABS,        WB(abs, WS)        }, \
            { GGML_UNARY_OP_NEG,        WB(neg, WS)        }, \
            { GGML_UNARY_OP_SGN,        WB(sgn, WS)        }, \
            { GGML_UNARY_OP_STEP,       WB(step, WS)       }, \
            { GGML_UNARY_OP_ELU,        WB(elu, WS)        }, \
            { GGML_UNARY_OP_HARDSIGMOID,WB(hardsigmoid, WS)}, \
            { GGML_UNARY_OP_HARDSWISH,  WB(hardswish, WS)  }, \
            { GGML_UNARY_OP_FLOOR,      WB(floor, WS)      }, \
            { GGML_UNARY_OP_CEIL,       WB(ceil, WS)       }, \
            { GGML_UNARY_OP_ROUND,      WB(round, WS)      }, \
            { GGML_UNARY_OP_TRUNC,      WB(trunc, WS)      }, \
            { GGML_UNARY_OP_XIELU,      WB(xielu, WS)      }, \
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
    dx12_shader_blob selected_blob_storage = {};

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

    // Matvec threadgroup size. When DX12_MMV_GROUP_SIZE is set (16|32|64|128|256|
    // 512) it forces that size; otherwise the default is picked from the device
    // wave via default_mmv_group_size(). WBLOB_GS() maps the choice to the
    // matching g_<name>_g<GS>_w<WS>_dxil blob (wave selection still via WBLOB).
    //
    // Vendor-keyed defaults, derived from per-device GROUP_SIZE sweeps (mul_mat_vec
    // for Q5_0 / Q6_K / Q8_0-dp4a GLU). Wave alone is not enough: Intel and NVIDIA
    // both run wave32 yet prefer opposite ends. Refine as more devices are measured:
    //   Intel Xe/UHD (wave 16 & 32): 32  (Q50 +2-4%; groups >=128 collapse -7% to -65%)
    //   NVIDIA (wave 32):            128 (Q50 +3.2%, Q6K +3.9%, Q80 tie)
    //   AMD RDNA/wave64 + others:    64  (baseline; wider groups only add reduction cost)
    auto default_mmv_group_size = [](int wave, dx12_arch_family arch) -> int {
        if (arch == DX12_ARCH_INTEL_XE_HPG_PLUS || arch == DX12_ARCH_INTEL_UHD) return 32;
        if (arch == DX12_ARCH_NV_PASCAL_PLUS && wave <= 32) return 128;
        return 64;
    };
    static const int mmv_group_size_env = [] {
        const char * e = getenv("DX12_MMV_GROUP_SIZE");
        if (!e) return 0;
        int v = atoi(e);
        return (v == 16 || v == 32 || v == 64 || v == 128 || v == 256 || v == 512) ? v : 0;
    }();
    const int mmv_group_size = mmv_group_size_env ? mmv_group_size_env : default_mmv_group_size(wave_size, arch_family);
    {
        static bool mmv_gs_logged = false;
        if (!mmv_gs_logged) {
            mmv_gs_logged = true;
            if (mmv_group_size_env) {
                printf("ggml-dx12: MMV threadgroup GROUP_SIZE=%d (DX12_MMV_GROUP_SIZE override)\n", mmv_group_size);
            } else if (mmv_group_size != 64) {
                printf("ggml-dx12: MMV threadgroup GROUP_SIZE=%d (auto for %s, wave %d)\n",
                    mmv_group_size, dx12_arch_family_str(arch_family), (int) wave_size);
            }
        }
    }
    #define WBLOB_GS(name) ( \
        mmv_group_size == 16  ? WBLOB(name##_g16)  : \
        mmv_group_size == 32  ? WBLOB(name##_g32)  : \
        mmv_group_size == 128 ? WBLOB(name##_g128) : \
        mmv_group_size == 256 ? WBLOB(name##_g256) : \
        mmv_group_size == 512 ? WBLOB(name##_g512) : \
        WBLOB(name))

    auto set_selected_blob = [&](dx12_shader_blob selected) {
        selected_blob_storage = selected;
        blob = &selected_blob_storage;
    };

    auto select_mul_mat_blob = [&](dx12_shader_blob & selected) -> bool {
        auto pick = [&](dx12_shader_blob value) {
            selected = value;
            return true;
        };

        switch (key.flags) {
            case 1:
                switch (key.src0_type) {
                    case GGML_TYPE_Q2_K:    return pick(WBLOB(mul_mat_vec_q2k));
                    case GGML_TYPE_Q3_K:    return pick(WBLOB(mul_mat_vec_q3k));
                    case GGML_TYPE_IQ4_NL:  return pick(WBLOB(mul_mat_vec_iq4_nl));
                    case GGML_TYPE_IQ2_XXS: return pick(WBLOB(mul_mat_vec_iq2_xxs));
                    case GGML_TYPE_IQ4_XS:  return pick(WBLOB(mul_mat_vec_iq4_xs));
                    case GGML_TYPE_IQ3_XXS: return pick(WBLOB(mul_mat_vec_iq3_xxs));
                    case GGML_TYPE_IQ2_XS:  return pick(WBLOB(mul_mat_vec_iq2_xs));
                    case GGML_TYPE_IQ2_S:   return pick(WBLOB(mul_mat_vec_iq2_s));
                    case GGML_TYPE_IQ3_S:   return pick(WBLOB(mul_mat_vec_iq3_s));
                    case GGML_TYPE_IQ1_S:   return pick(WBLOB(mul_mat_vec_iq1_s));
                    case GGML_TYPE_IQ1_M:   return pick(WBLOB(mul_mat_vec_iq1_m));
                    case GGML_TYPE_Q4_0:    return pick(WBLOB(mul_mat_q4_0));
                    case GGML_TYPE_Q4_1:    return pick(WBLOB(mul_mat_q4_1));
                    default:                return pick(WBLOB_FP16(mul_mat_vec));
                }
            case 4:
                switch (key.src0_type) {
                    case GGML_TYPE_Q4_K: return pick(WBLOB(mul_mat_q4k_wmma));
                    case GGML_TYPE_Q5_K: return pick(WBLOB(mul_mat_q5k_wmma));
                    case GGML_TYPE_Q6_K: return pick(WBLOB(mul_mat_q6k_wmma));
                    case GGML_TYPE_Q8_0: return pick(WBLOB(mul_mat_q8_0_wmma));
                    case GGML_TYPE_Q4_0: return pick(WBLOB(mul_mat_q4_0_wmma));
                    case GGML_TYPE_Q4_1: return pick(WBLOB(mul_mat_q4_1_wmma));
                    default:             return pick(WBLOB(mul_mat_wmma));
                }
            case 8:  return pick(WBLOB(mul_mat_q8_0_q8_1));
            case 9:
                switch (key.src0_type) {
                    case GGML_TYPE_Q4_K: return pick(WBLOB(mul_mat_vec_q4k_mr));
                    case GGML_TYPE_Q5_K: return pick(WBLOB(mul_mat_vec_q5k_mr));
                    case GGML_TYPE_Q6_K: return pick(WBLOB(mul_mat_vec_q6k_mr));
                    case GGML_TYPE_Q8_0: return pick(WBLOB(mul_mat_vec_q8_0_mr));
                    case GGML_TYPE_Q5_0: return pick(WBLOB(mul_mat_vec_q5_0_mr));
                    case GGML_TYPE_Q5_1: return pick(WBLOB(mul_mat_vec_q5_1_mr));
                    default:             return false;
                }
            case 10: return pick(WBLOB(mul_mat_vec_q4k_dp4a));
            case 11: return pick(WBLOB_FP16(mul_mat_vec_mr));
            case 12: return pick(WBLOB_FP16(mul_mat_vec_mr32));
            case 13: return pick(WBLOB(mul_mat_vec_q4k_dp4a_32));
            case 14: return pick(WBLOB(mul_mat_vec_q5k_dp4a));
            case 15: return pick(WBLOB(mul_mat_vec_q5k_subgroup));
            case 16: return pick(WBLOB(mul_mat_vec_q5k_dp4a_32));
            // Q8_0 dp4a matvec wants wider threadgroups than the shared MMV
            // default. Narrow-N decode shapes (Phi-3 ffn_down 8192x3072 ->
            // 1536 groups) run one wave per group and starve occupancy.
            // RTX 6000 Ada: matvec GPU time 6.28 -> 5.57 ms/token, 152 -> 169 t/s.
            case 17: {
                const int q8_gs = mmv_group_size_env ? mmv_group_size_env :
                    ((arch_family == DX12_ARCH_NV_PASCAL_PLUS && wave_size <= 32) ? 256 : 32);
                return pick(
                    q8_gs == 16  ? WBLOB(mul_mat_vec_q8_0_dp4a_g16)  :
                    q8_gs == 128 ? WBLOB(mul_mat_vec_q8_0_dp4a_g128) :
                    q8_gs == 256 ? WBLOB(mul_mat_vec_q8_0_dp4a_g256) :
                    q8_gs == 512 ? WBLOB(mul_mat_vec_q8_0_dp4a_g512) :
                    WBLOB(mul_mat_vec_q8_0_dp4a_g32));
            }
            case 18: return pick(WBLOB(mul_mat_vec_q8_0_mr256v));
            case 19: return pick(WBLOB(mul_mat_vec_q2k_mr));
            case 20: return pick(WBLOB(mul_mat_vec_q3k_mr));
            case 21: return pick(WBLOB(mul_mat_vec_q5_0_dp4a));
            case 22: return pick(WBLOB(mul_mat_vec_q5_1_dp4a));
            case 23: return pick(WBLOB(mul_mat_vec_q6k_dp4a));
            case 24: return pick(WBLOB_FP16(mul_mat_vec_glu));
            case 25: return pick(WBLOB(mul_mat_vec_q6k_mr_blocked));
            case 26: return pick(WBLOB(mul_mat_vec_q3k_mr_blocked));
            case 27: return pick(WBLOB(mul_mat_vec_q2k_mr_blocked));
            case 28: return pick(WBLOB(mul_mat_vec_q8_0_mr64));
            case 29: return pick(WBLOB(mul_mat_vec_q5_0_mr64));
            case 30: return pick(WBLOB(mul_mat_q4k_wmma_lds));
            case 31: return pick(WBLOB(mul_mat_vec_glu_q5_0));
            case 32: return pick(WBLOB(mul_mat_vec_glu_q4_k));
            case 33: return pick(WBLOB(mul_mat_vec_glu_q5_k));
            case 34: return pick(WBLOB(mul_mat_vec_q5_0_mr_lds));
            case 35: return pick(WBLOB(mul_mat_vec_glu_q8_0));
            case 36: return pick(WBLOB(mul_mat_vec_iq4_nl_mr));
            case 37: return pick(WBLOB(mul_mat_vec_iq2_xxs_mr));
            case 38: return pick(WBLOB(mul_mat_vec_q4_0_dp4a));
            case 39: return pick(WBLOB(mul_mat_vec_glu_q3_k));
            case 43:
                switch (key.src0_type) {
                    case GGML_TYPE_IQ2_XXS: return pick(WBLOB(mul_mat_iq2_xxs_quant));
                    case GGML_TYPE_IQ2_XS:  return pick(WBLOB(mul_mat_iq2_xs_quant));
                    case GGML_TYPE_IQ2_S:   return pick(WBLOB(mul_mat_iq2_s_quant));
                    case GGML_TYPE_IQ3_XXS: return pick(WBLOB(mul_mat_iq3_xxs_quant));
                    case GGML_TYPE_IQ3_S:   return pick(WBLOB(mul_mat_iq3_s_quant));
                    case GGML_TYPE_IQ1_S:   return pick(WBLOB(mul_mat_iq1_s_quant));
                    case GGML_TYPE_IQ1_M:   return pick(WBLOB(mul_mat_iq1_m_quant));
                    case GGML_TYPE_IQ4_XS:  return pick(WBLOB(mul_mat_iq4_xs_quant));
                    default:                return false;
                }
            case 44: return pick(WBLOB(mul_mat_vec_q8_0_mr256));
            case 45: return pick(WBLOB(mul_mat_vec_q8_0_dp4a_mr64));
            case 46: return pick(WBLOB(mul_mat_vec_q4k_dp4a_mr4));
            case 47: return pick(WBLOB(mul_mat_vec_q4k_dp4a_nc2));
            case 48: return pick(WBLOB(mul_mat_vec_q4k_dp4a_nc4));
            case 49: return pick(WBLOB(mul_mat_vec_q4k_dp4a_nc8));
            case 50: return pick(WBLOB(mul_mat_vec_q5k_dp4a_nc2));
            case 51: return pick(WBLOB(mul_mat_vec_q6k_dp4a_nc2));
            case 52: return pick(WBLOB(mul_mat_vec_q8_0_dp4a_nc2));
            case 53: return pick(WBLOB(mul_mat_wmma_fp16));
            case 54: return pick(WBLOB(mul_mat_wmma_kfull));
            case 55: return pick(WBLOB(mul_mat_vec_iq4_nl_mr256));
            case 56: return pick(WBLOB(mul_mat_vec_q4_0_dp4a_mr256));
            case 57: return pick(WBLOB(mul_mat_vec_q5_0_dp4a_mr256));
            case 58:
                switch (key.src0_type) {
                    case GGML_TYPE_Q8_0: return pick(WBLOB(mul_mat_q8_0_q8_1_tiled));
                    case GGML_TYPE_Q4_K: return pick(WBLOB(mul_mat_q4k_q8_1_tiled));
                    case GGML_TYPE_Q5_K: return pick(WBLOB(mul_mat_q5k_q8_1_tiled));
                    case GGML_TYPE_Q5_0: return pick(WBLOB(mul_mat_q5_0_q8_1_tiled));
                    case GGML_TYPE_Q6_K: return pick(WBLOB(mul_mat_q6k_q8_1_tiled));
                    default:             return false;
                }
            case 59: return pick(WBLOB(mul_mat_q8_0_q8_1_tiled_intel));
            case 60: return pick(WBLOB_GS(mul_mat_vec_q5_0_subgroup));
            case 61: return pick(WBLOB_GS(mul_mat_vec_q6k_subgroup));
            case 62: return pick(WBLOB_GS(mul_mat_vec_glu_q8_0_dp4a_mr64));
            case 63: return pick(WBLOB_FP16(mul_mat_vec_f16_wave64));
            case 64: return pick(WBLOB(mul_mat_vec_q8_0_wave64));
            case 66: return pick(WBLOB_GS(mul_mat_vec_glu_q5_0_subgroup));
            case 67: return pick(WBLOB(mul_mat_vec_q8_0_wave64_rows2));
            case 72: return pick(WBLOB(mul_mat_vec_q5_0_vulkan_rows2));
            case 73: return pick(WBLOB(mul_mat_vec_glu_q5_0_vulkan_rows2));
            case 74: return pick(WBLOB(mul_mat_vec_glu_q8_0_wave64_rows2));
            case 75: return pick(WBLOB_FP16(mul_mat_vec_qkv_f16_wave64));
            case 76: return pick(WBLOB(mul_mat_vec_qkv_q8_0_wave64_rows2));
            case 77: return pick(WBLOB(mul_mat_vec_qkv_q5_0_vulkan_rows2));
            case 78: return pick(WBLOB(mul_mat_vec_q3k_dp4a));
            case 79: return pick(WBLOB(mul_mat_vec_qkv_q5_0_q8_0_rows2));
            case 82: return pick(WBLOB(mul_mat_vec_q6k_subgroup));
            case 83: return pick(WBLOB(mul_mat_vec_q5_0_nc8));
            case 84: return pick(WBLOB(mul_mat_vec_qkv_q8_0_mr256));
            default: break;
        }

        switch (key.src0_type) {
            case GGML_TYPE_Q8_0:   return pick(WBLOB(mul_mat_q8_0));
            case GGML_TYPE_Q5_0:   return pick(WBLOB(mul_mat_q5_0));
            case GGML_TYPE_Q4_0:   return pick(WBLOB(mul_mat_q4_0));
            case GGML_TYPE_Q4_1:   return pick(WBLOB(mul_mat_q4_1));
            case GGML_TYPE_Q5_1:   return pick(WBLOB(mul_mat_q5_1));
            case GGML_TYPE_Q8_1:   return pick(WBLOB(mul_mat_q8_1));
            case GGML_TYPE_Q2_K:   return pick(WBLOB(mul_mat_q2k));
            case GGML_TYPE_Q3_K:   return pick(WBLOB(mul_mat_q3k));
            case GGML_TYPE_IQ4_NL: return pick(WBLOB(mul_mat_iq4_nl));
            default:               return false;
        }
    };

    auto select_mul_mat_id_blob = [&](dx12_shader_blob & selected) -> bool {
        auto pick = [&](dx12_shader_blob value) {
            selected = value;
            return true;
        };

        switch (key.src0_type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
                return key.flags == 1 ? pick(WBLOB(mul_mat_id_coop)) : false;
            case GGML_TYPE_Q4_0:
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_q4_0_coop) : WBLOB(mul_mat_id_q4_0));
            case GGML_TYPE_Q4_1:
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_q4_1_coop) : WBLOB(mul_mat_id_q4_1));
            case GGML_TYPE_Q5_0:
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_q5_0_coop) : WBLOB(mul_mat_id_q5_0));
            case GGML_TYPE_Q5_1:
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_q5_1_coop) : WBLOB(mul_mat_id_q5_1));
            case GGML_TYPE_Q8_0:
                return pick(key.flags == 17 ? WBLOB(mul_mat_id_q8_0_dp4a) : WBLOB(mul_mat_id_q8_0));
            case GGML_TYPE_Q4_K:
                if (key.flags == 51) {
                    return pick(WBLOB(mul_mat_id_q4k_block));
                }
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_q4k_coop) : WBLOB(mul_mat_id_q4k));
            case GGML_TYPE_Q5_K:
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_q5k_coop) : WBLOB(mul_mat_id_q5k));
            case GGML_TYPE_Q6_K:
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_q6k_coop) : WBLOB(mul_mat_id_q6k));
            case GGML_TYPE_IQ4_NL:
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_iq4_nl_coop) : WBLOB(mul_mat_id_iq4_nl));
            case GGML_TYPE_IQ4_XS:
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_iq4_xs_coop) : WBLOB(mul_mat_id_iq4_xs));
            case GGML_TYPE_IQ2_XXS:
                return pick(WBLOB(mul_mat_id_iq2_xxs));
            case GGML_TYPE_Q2_K:
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_q2k_coop) : WBLOB(mul_mat_id_q2k));
            case GGML_TYPE_Q3_K:
                return pick(key.flags == 1 ? WBLOB(mul_mat_id_q3k_coop) : WBLOB(mul_mat_id_q3k));
            case GGML_TYPE_IQ2_XS:
                return pick(WBLOB(mul_mat_id_iq2_xs));
            case GGML_TYPE_IQ2_S:
                return pick(WBLOB(mul_mat_id_iq2_s));
            case GGML_TYPE_IQ3_XXS:
                return pick(WBLOB(mul_mat_id_iq3_xxs));
            case GGML_TYPE_IQ3_S:
                return pick(WBLOB(mul_mat_id_iq3_s));
            case GGML_TYPE_IQ1_S:
                return pick(WBLOB(mul_mat_id_iq1_s));
            case GGML_TYPE_IQ1_M:
                return pick(WBLOB(mul_mat_id_iq1_m));
            default:
                return false;
        }
    };

    auto select_get_rows_blob = [&](dx12_shader_blob & selected) -> bool {
        auto pick = [&](dx12_shader_blob value) {
            selected = value;
            return true;
        };

        switch (key.src0_type) {
            case GGML_TYPE_Q4_K:   return pick(WBLOB(get_rows_q4k));
            case GGML_TYPE_Q5_K:   return pick(WBLOB(get_rows_q5k));
            case GGML_TYPE_Q6_K:   return pick(WBLOB(get_rows_q6k));
            case GGML_TYPE_Q8_0:   return pick(WBLOB(get_rows_q8_0));
            case GGML_TYPE_Q5_0:   return pick(WBLOB(get_rows_q5_0));
            case GGML_TYPE_Q4_0:   return pick(WBLOB(get_rows_q4_0));
            case GGML_TYPE_Q4_1:   return pick(WBLOB(get_rows_q4_1));
            case GGML_TYPE_Q5_1:   return pick(WBLOB(get_rows_q5_1));
            case GGML_TYPE_Q8_1:   return pick(WBLOB(get_rows_q8_1));
            case GGML_TYPE_Q2_K:   return pick(WBLOB(get_rows_q2k));
            case GGML_TYPE_Q3_K:   return pick(WBLOB(get_rows_q3k));
            case GGML_TYPE_IQ4_NL: return pick(WBLOB(get_rows_iq4_nl));
            default:               return false;
        }
    };

    dx12_shader_blob operation_blob = {};
    switch (key.op) {
        case GGML_OP_MUL_MAT:
            if (select_mul_mat_blob(operation_blob)) {
                set_selected_blob(operation_blob);
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            if (select_mul_mat_id_blob(operation_blob)) {
                set_selected_blob(operation_blob);
            }
            break;
        case GGML_OP_GET_ROWS:
            if (select_get_rows_blob(operation_blob)) {
                set_selected_blob(operation_blob);
            }
            break;
        default:
            break;
    }

    if (!blob) {
        // For UNARY ops, look up by the unary sub-op stored in flags.
        if (key.op == GGML_OP_UNARY) {
            auto uit = unary_shader_blobs.find((int)key.flags);
            if (uit != unary_shader_blobs.end()) {
                set_selected_blob(uit->second);
            }
        } else if (key.op == GGML_OP_SET_ROWS && key.dst_type == GGML_TYPE_Q8_0) {
            set_selected_blob(WBLOB(set_rows_q8_0));
        } else if (key.op == GGML_OP_SET_ROWS && key.dst_type == GGML_TYPE_Q4_0) {
            set_selected_blob(WBLOB(set_rows_q4_0));
        } else if (key.op == GGML_OP_SET_ROWS && key.dst_type == GGML_TYPE_Q4_1) {
            set_selected_blob(WBLOB(set_rows_q4_1));
        } else if (key.op == GGML_OP_SET_ROWS && key.dst_type == GGML_TYPE_Q5_0) {
            set_selected_blob(WBLOB(set_rows_q5_0));
        } else if (key.op == GGML_OP_SET_ROWS && key.dst_type == GGML_TYPE_Q5_1) {
            set_selected_blob(WBLOB(set_rows_q5_1));
        } else if (key.op == GGML_OP_SET_ROWS && key.dst_type == GGML_TYPE_IQ4_NL) {
            set_selected_blob(WBLOB(set_rows_iq4_nl));
        } else if ((key.op == GGML_OP_CPY || key.op == GGML_OP_DUP) &&
                   key.dst_type == key.src0_type &&
                   (ggml_is_quantized(key.dst_type) || key.dst_type == GGML_TYPE_I16)) {
            set_selected_blob(WBLOB(cpy_quant_block));
        } else if ((key.op == GGML_OP_ARGSORT || key.op == GGML_OP_TOP_K) && key.flags == 50) {
            set_selected_blob(WBLOB(argsort_large));
        } else if (key.op == GGML_OP_GATED_DELTA_NET && key.flags == 16) {
            set_selected_blob(WBLOB(gated_delta_net_sv16));
        } else if (key.op == GGML_OP_GATED_DELTA_NET && key.flags == 32) {
            set_selected_blob(WBLOB(gated_delta_net_sv32));
        } else if (key.op == GGML_OP_GATED_DELTA_NET && key.flags == 64) {
            set_selected_blob(WBLOB(gated_delta_net_sv64));
        } else if (key.op == GGML_OP_GATED_DELTA_NET && key.flags == 0x100) {
            set_selected_blob(WBLOB(gated_delta_net_kda));
        } else if (key.op == GGML_OP_GATED_DELTA_NET && key.flags == (0x100 | 16)) {
            set_selected_blob(WBLOB(gated_delta_net_sv16_kda));
        } else if (key.op == GGML_OP_GATED_DELTA_NET && key.flags == (0x100 | 32)) {
            set_selected_blob(WBLOB(gated_delta_net_sv32_kda));
        } else if (key.op == GGML_OP_GATED_DELTA_NET && key.flags == (0x100 | 64)) {
            set_selected_blob(WBLOB(gated_delta_net_sv64_kda));
        } else if (key.op == GGML_OP_SSM_SCAN && key.flags == 256) {
            set_selected_blob(WBLOB(ssm_scan_d256));
        } else if (key.op == GGML_OP_SSM_CONV && key.flags == 1) {
            set_selected_blob(WBLOB(ssm_conv_silu));
        } else if (key.op == GGML_OP_SSM_CONV && key.flags == 2) {
            set_selected_blob(WBLOB(ssm_conv_bias_silu));
        } else if (key.op == GGML_OP_RMS_NORM && key.flags == 2) {
            set_selected_blob(WBLOB(rms_norm_mul));
        } else if (key.op == GGML_OP_RMS_NORM && key.flags == 85) {
            set_selected_blob(WBLOB(rms_norm_mul_1024));
        } else if (key.op == GGML_OP_RMS_NORM && key.flags == 12) {
            set_selected_blob(WBLOB(rms_norm_mul_quantize_q8_1));
        } else if (key.op == GGML_OP_RMS_NORM && key.flags == 3) {
            set_selected_blob(WBLOB(add_rms_norm_mul));
        } else if (key.op == GGML_OP_RMS_NORM && key.flags == 7) {
            set_selected_blob(WBLOB(rms_norm_mul_rope));
        } else if (key.op == GGML_OP_RMS_NORM && key.flags == 8) {
            set_selected_blob(WBLOB(rms_norm_mul_rope_set_rows));
        } else if (key.op == GGML_OP_ROPE && key.flags == 6) {
            set_selected_blob(WBLOB(rope_set_rows));
        } else if (key.op == GGML_OP_ROPE && key.flags == 87) {
            set_selected_blob(WBLOB(rope_scale_k_set_rows));
        } else if (key.op == GGML_OP_ROPE && key.flags == 13) {
            set_selected_blob(WBLOB(rope_multi));
        } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags == 1) {
            set_selected_blob(WBLOB_FP16(flash_attn_gqa));
        } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags == 2) {
            set_selected_blob(WBLOB(flash_attn_64));
        } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags == 3) {
            set_selected_blob(WBLOB(flash_attn_128));
        } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags == 8) {
            set_selected_blob(WBLOB(flash_attn_reduce));
        } else if (key.op == GGML_OP_FLASH_ATTN_EXT && key.flags >= 20 && key.flags <= 33) {
            switch (key.flags) {
                case 20: set_selected_blob(WBLOB(flash_attn_q4_0));    break;
                case 21: set_selected_blob(WBLOB(flash_attn_q4_1));    break;
                case 22: set_selected_blob(WBLOB(flash_attn_q5_0));    break;
                case 23: set_selected_blob(WBLOB(flash_attn_q5_1));    break;
                case 24: set_selected_blob(WBLOB(flash_attn_q8_0));    break;
                case 25: set_selected_blob(WBLOB(flash_attn_iq4_nl));  break;
                case 26: set_selected_blob(WBLOB(flash_attn_cd_64));   break;
                case 27: set_selected_blob(WBLOB(flash_attn_cd_128));  break;
                case 28: set_selected_blob(WBLOB(flash_attn_tiled));   break;
                case 29: set_selected_blob(WBLOB(flash_attn_tiled16)); break;
                case 30: set_selected_blob(WBLOB(flash_attn_cd_96));   break;
                case 31: set_selected_blob(WBLOB(flash_attn_cd_q8_0_64));  break;
                case 32: set_selected_blob(WBLOB(flash_attn_cd_q8_0_96));  break;
                case 33: set_selected_blob(WBLOB(flash_attn_cd_q8_0_128)); break;
                default: break;
            }
        } else if (key.op == GGML_OP_SOFT_MAX && key.flags == 1) {
            set_selected_blob(WBLOB(soft_max_cached));
        } else if (key.op == GGML_OP_NONE && key.flags == 99) {
            set_selected_blob(WBLOB(quantize_q8_1));
        }
    }

    if (!blob) {
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

void ggml_backend_dx12_set_env_refresh(bool on) {
    g_dx12_env_refresh = on;
}

void ggml_backend_dx12_set_flag_sink(void * sink) {
    g_dx12_flag_sink = static_cast<std::vector<uint32_t> *>(sink);
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
