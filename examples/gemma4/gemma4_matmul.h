// G4.1 - qquant matmul shim.
//
// Wraps ggml_mul_mat to bridge between our standalone hand-coded forward
// (host F32 activations) and the original quantized weight tensors loaded
// by llama_model_loader (Q4_K, Q5_K, F32, ...).
//
// Each matmul_qf32 call:
//   * resets a persistent ggml arena
//   * wraps the F32 activation x_in[n_in, n_cols] as a tensor in that arena
//   * builds a 1-op graph: y = ggml_mul_mat(W, x)
//   * runs the graph on the persistent ggml_threadpool (created once in
//     matmul_ctx_init) via ggml_graph_plan + ggml_graph_compute -- the
//     worker threads stay alive across calls, eliminating per-call
//     thread spawn/join overhead.
//   * memcpys y_t->data into y_out[n_out, n_cols]
//
// The W tensor is owned by the model loader (different ggml_context); ggml
// is happy to reference foreign tensors from another context's graph as
// long as W->data is in CPU-accessible memory (i.e. -ngl 0 on a CPU
// backend). This is the same trick the oracle uses with its F32 weights.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct ggml_cgraph;
struct ggml_threadpool;
struct ggml_cplan;

// Forward declare the full cplan struct's representation for inline storage.
// We can't forward-declare a struct AND store it by value, so include the
// canonical header form via ggml-cpu.h (which defines ggml_cplan).
#include "ggml.h"
#include "ggml-cpu.h"

namespace gemma4 {

// Custom deleter for ggml_threadpool so MatmulCtx is self-cleaning.
struct GgmlThreadpoolDeleter {
    void operator()(ggml_threadpool * p) const noexcept;
};

// Custom deleter for ggml_context so MatmulCacheEntry is self-cleaning.
struct GgmlContextDeleter {
    void operator()(ggml_context * c) const noexcept;
};

// ---------------------------------------------------------------------------
// G5.1 - tiny worker pool for parallelising the per-head attention loop in
// layer_forward_f32_cached. The existing ggml_threadpool inside MatmulCtx is
// only reachable via ggml_graph_compute, so we add a small purpose-built
// pool that can dispatch an arbitrary `JobFn(wid, n_workers, user_data)`.
//
// Workers park on a condition variable when idle (not busy-spin) so they do
// not oversubscribe cores while the ggml threadpool is running matmuls.
// CV wake cost is ~10-20 us per call * 35 layers = ~0.5 ms / decode token,
// negligible vs the ~4 ms / token attention work being parallelised.
//
// Main-as-worker convention (mirrors Phi3MatmulPool, commit 67230341e):
// pool->n_threads counts the caller as worker 0 plus (n_threads - 1)
// helpers. attn_pool_run executes wid=0 on the calling thread and waits
// for n_threads - 1 done counts from the helpers.
//
// NOT thread-safe at the run-driver level: at most one in-flight call to
// attn_pool_run per MatmulCtx (same single-caller invariant the arena /
// work_buf in this struct already require).
struct AttnWorkerPool;

// JobFn dispatch signature. Caller passes a function pointer + opaque
// pointer; pool replicates it across (n_workers - 1) helpers and the
// caller's own thread. Each invocation receives its worker id (wid in
// [0, n_workers)) and the total worker count W = pool->n_threads.
using AttnJobFn = void (*)(int wid, int W, void * user_data);

struct AttnPoolDeleter {
    void operator()(AttnWorkerPool * p) const noexcept;
};

// ---------------------------------------------------------------------------
// G5.2 - per-shape matmul graph cache. Each entry owns a private arena +
// ggml_context with a permanently-built 1-op graph (y = mul_mat(W, x)) and
// a cached cplan. On a cache hit we skip ggml_init / new_tensor /
// mul_mat / new_graph / build_forward_expand / graph_plan and go straight
// to memcpy x -> compute -> memcpy y. Eliminates ~7 ms/token of per-call
// graph rebuild overhead at typical tg64 contexts (~245 matmuls/token).
//
// Cache is single-caller (same invariant as the shared arena/work_buf).
// Only used when n_cols is small (decode); prefill matmuls bypass it.
// ---------------------------------------------------------------------------
struct MatmulCacheEntry {
    std::vector<uint8_t>                              arena;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor *  x_t   = nullptr;     // owned by ctx
    ggml_tensor *  y_t   = nullptr;     // owned by ctx; result of mul_mat
    ggml_cgraph *  gf    = nullptr;     // owned by ctx; 1-node graph
    ggml_cplan     cplan = {};          // work_data refreshed per call
    int            n_in     = 0;
    int            n_out    = 0;
    int            n_cols   = 0;
    int            w_type   = 0;        // ggml_type at build time; rebuilds on mismatch
    int            n_threads = 1;       // n_threads at build time
};

struct MatmulCacheKey {
    const ggml_tensor * W      = nullptr;
    int                 n_cols = 0;
    bool operator==(const MatmulCacheKey & o) const noexcept {
        return W == o.W && n_cols == o.n_cols;
    }
};

struct MatmulCacheKeyHash {
    std::size_t operator()(const MatmulCacheKey & k) const noexcept {
        // FNV-1a-ish mix; W pointers are 16-byte-aligned so the low bits
        // are zero -- shift before xoring with n_cols.
        const std::size_t p = reinterpret_cast<std::size_t>(k.W);
        return (p >> 4) ^ ((std::size_t) (uint32_t) k.n_cols * 0x9E3779B1u);
    }
};

// ---------------------------------------------------------------------------
// G5.3 - fused-MoE graph cache. matmul_moe_id_qf32 builds a multi-op graph
// (2x mul_mat_id + geglu + weight + expert-sum) inside a private ~32 MiB
// context, mallocing + rebuilding + planning every call. During decode the
// shape is fixed (n_tokens=1) and the same per-layer expert tensors recur
// every token, so we cache the built context/graph/cplan keyed by
// (down_exps, n_tokens) and on a hit only refresh the x/ids/weights inputs
// and recompute. Removes ~30 per-token 32 MiB mallocs + graph rebuilds.
//
// Single-caller (same invariant as the shared arena/work_buf). Only used
// when n_tokens is small (decode); prefill bypasses it (per-length keys
// would never recur).
// ---------------------------------------------------------------------------
struct MoeCacheEntry {
    std::vector<uint8_t>                              arena;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor *  x_t   = nullptr;     // owned by ctx; moe_in  [n_embd, n_tokens]
    ggml_tensor *  ids_t = nullptr;     // owned by ctx; ids     [n_expert_used, n_tokens]
    ggml_tensor *  w_t   = nullptr;     // owned by ctx; weights [1, n_expert_used, n_tokens]
    ggml_tensor *  moe   = nullptr;     // owned by ctx; result  [n_embd, n_tokens]
    ggml_cgraph *  gf    = nullptr;     // owned by ctx
    ggml_cplan     cplan = {};          // work_data refreshed per call
    int            n_embd        = 0;
    int            n_ff_exp      = 0;
    int            n_expert_used = 0;
    int            n_tokens      = 0;
    int            n_threads     = 1;   // n_threads at build time
    bool           merged        = false;
    const ggml_tensor * gate_up_exps = nullptr; // identity guard: rebuild on mismatch
    const ggml_tensor * gate_exps    = nullptr;
    const ggml_tensor * up_exps      = nullptr;
    const ggml_tensor * down_exps    = nullptr;
};

struct MoeCacheKey {
    const ggml_tensor * down_exps = nullptr;
    int                 n_tokens  = 0;
    bool operator==(const MoeCacheKey & o) const noexcept {
        return down_exps == o.down_exps && n_tokens == o.n_tokens;
    }
};

struct MoeCacheKeyHash {
    std::size_t operator()(const MoeCacheKey & k) const noexcept {
        const std::size_t p = reinterpret_cast<std::size_t>(k.down_exps);
        return (p >> 4) ^ ((std::size_t) (uint32_t) k.n_tokens * 0x9E3779B1u);
    }
};

struct MatmulCtx {
    // Arena memory used by the per-call ggml_context. Sized once at init
    // and reused (ggml_init resets the bump pointer to the start of this
    // buffer on every call).
    std::vector<uint8_t> arena;

    // ggml_cplan work buffer (sized lazily to max work_size seen).
    // Persists across calls so the compute path never reallocates once
    // it has run the largest matmul shape at least once.
    std::vector<uint8_t> work_buf;

    // Persistent CPU threadpool. nullptr -> single-thread fallback (uses
    // the plain ggml_graph_compute_with_ctx path). Created in
    // matmul_ctx_init when n_threads > 1.
    std::unique_ptr<ggml_threadpool, GgmlThreadpoolDeleter> pool;

    // G5.1 - persistent attention worker pool. nullptr when n_threads <= 1.
    // Created in matmul_ctx_init alongside the ggml pool.
    std::unique_ptr<AttnWorkerPool, AttnPoolDeleter> attn_pool;

    // G5.2 - matmul graph cache (see MatmulCacheEntry comment).
    std::unordered_map<MatmulCacheKey, MatmulCacheEntry, MatmulCacheKeyHash> matmul_cache;

    // G5.3 - fused-MoE graph cache (see MoeCacheEntry comment).
    std::unordered_map<MoeCacheKey, MoeCacheEntry, MoeCacheKeyHash> moe_cache;

    int n_threads = 1;
};

// G5.1 - dispatch fn across n_workers (= mm.n_threads). Caller thread runs
// fn(0, W, ud) inline; helpers run fn(wid, W, ud) for wid in [1, W).
// Returns after all helpers report done. If mm.attn_pool is nullptr (single
// thread fallback or n_threads <= 1), this is a serial call: fn(0, 1, ud).
void attn_pool_run(MatmulCtx & mm, AttnJobFn fn, void * user_data);

// G5.1 - global setter consumed by gemma4_forward.cpp. Set once from CLI
// (--gemma4-attn-parallel 0|1) before any decode. Default is ON; pass 0 to
// fall back to the serial per-head loop for A/B comparison.
void set_attn_parallel(bool on);
bool get_attn_parallel();

// G5.2 - global setter for the matmul graph cache. Default ON; pass 0 to
// fall back to the per-call build path for A/B comparison. Set once at
// startup via --gemma4-matmul-cache 0|1.
void set_matmul_cache(bool on);
bool get_matmul_cache();

// G6.1 - global setter for the fused ggml_mul_mat_id MoE path. Default ON;
// pass 0 to fall back to the per-expert matmul_expert_qf32 loop for A/B
// comparison. Set once at startup via --gemma4-moe-fused 0|1. Only engaged
// on the resident (all-experts-in-memory) path; the streaming ExpertStore
// path always uses the per-block loop.
void set_moe_fused(bool on);
bool get_moe_fused();

// G6.2 - global setter for the fused greedy lm_head+argmax path. When ON,
// the decode lm_head step computes the argmax token directly via a threaded
// per-vocab-row vec_dot over the (quantized) output-embedding weight,
// skipping (a) the 262144-element F32 logits materialization and (b) the
// final-logit softcap (cap*tanh(x/cap) is strictly monotonic, so it does
// not change the argmax). GREEDY ONLY -- callers that sample from the full
// (softcapped) logit distribution must NOT use the fused path. Default ON;
// set via --gemma4-lmhead-fused 0|1 for A/B.
void set_lmhead_fused(bool on);
bool get_lmhead_fused();

// G7 (prototype) - global setter for the fused per-layer prefill graph. When
// ON, the prefill (n_new > 1) path builds ONE ggml graph per layer (quantized
// weights, run multithreaded on mm.pool) instead of the ~7-per-layer scalar
// hand kernels + separate matmul dispatches. Collapses the per-layer dispatch
// count and lets ggml multithread every op. Decode (n_new == 1) always uses
// the hand path. Default OFF; set via --gemma4-prefill-fused 0|1 for A/B.
void set_prefill_fused(bool on);
bool get_prefill_fused();

// Allocate the arena, store the thread count, and (when n_threads > 1)
// spin up the persistent ggml_threadpool. arena_bytes should be large
// enough to hold one matmul's worth of tensor metadata + the F32
// activation copy + the F32 output + the graph + ggml's per-call work
// buffer; 32 MiB comfortably covers the lm_head case (262144 vocab).
bool matmul_ctx_init(MatmulCtx & mm, std::size_t arena_bytes, int n_threads,
                     std::string & error);

// y[n_out, n_cols] = W[n_in, n_out] (column-of-weights view) @ x[n_in, n_cols]
//
// Matches the existing matmul_f32 semantics:
//   x is interpreted as n_cols columns of length n_in,
//   y is interpreted as n_cols columns of length n_out.
//
// In ggml shape terms (row-major ne[0] = innermost):
//   W   : ne[0] = n_in,  ne[1] = n_out
//   x_t : ne[0] = n_in,  ne[1] = n_cols
//   y_t : ne[0] = n_out, ne[1] = n_cols
//
// W must be a 2D tensor; supported types are whatever ggml_mul_mat
// accepts (F32, Q4_K, Q5_K, ...). x_in / y_out are host F32 buffers.
bool matmul_qf32(MatmulCtx & mm, const ggml_tensor * W,
                 const float * x_in, float * y_out,
                 int n_in, int n_out, int n_cols, std::string & error);

// MoE per-expert matmul. W3d is a stacked expert tensor with the expert
// dimension outermost (ne[2] = n_expert), as produced by resolve():
//   ne[0] = n_in, ne[1] = n_out, ne[2] = n_expert
// This computes y[n_out, n_cols] = W3d[:,:,expert] @ x[n_in, n_cols] by
// building a 2D ggml_view over the single expert's contiguous sub-block and
// feeding it to ggml_mul_mat. Because the expert dim is outermost, expert e
// starts at byte offset e*nb[2] and each expert row keeps the tensor's
// native nb[1] row stride, so the view lands exactly on a K-quant block
// boundary (nb[1] = ggml_row_size(type, n_in), a whole number of blocks).
//
// Uncached (rebuilds a 1-op graph per call); intended for correctness
// validation and the initial all-resident MoE forward. x_in / y_out are
// host F32 buffers, same layout as matmul_qf32.
bool matmul_expert_qf32(MatmulCtx & mm, const ggml_tensor * W3d, int expert,
                        const float * x_in, float * y_out,
                        int n_in, int n_out, int n_cols, std::string & error);

// MoE per-expert matmul over a STREAMED expert block (P1). Instead of
// viewing a resident stacked tensor, this points a 2D ggml tensor of type
// `w_type` directly at `block` -- a contiguous quantized expert sub-block
// (exactly nb[2] bytes) fetched by ExpertStore::fetch. The block layout is
// identical to a standalone [n_in, n_out] quant tensor with the native row
// stride, so ggml_mul_mat consumes it unchanged. y[n_out, n_cols] =
// block[n_in, n_out] @ x[n_in, n_cols].
bool matmul_qblock_qf32(MatmulCtx & mm, int w_type, const void * block,
                        const float * x_in, float * y_out,
                        int n_in, int n_out, int n_cols, std::string & error);

// G6.1 - fused resident MoE via ggml_mul_mat_id. Computes, for all tokens at
// once and in a single graph, the routed-expert contribution
//   moe_out[:, t] = sum_j weights[j,t] * down_e( geglu( gate_up_e( moe_in[:,t] ) ) )
// where e = ids[j,t] selects one of the resident experts. This mirrors the
// upstream llm_graph_context::build_moe_ffn expert path (merged gate_up +
// ggml_geglu_split + down), fusing all n_expert_used experts and all tokens
// into one ggml_mul_mat_id per projection instead of the per-expert,
// per-token GEMV loop -- one threadpool dispatch instead of ~2*used*tokens.
//
// Weight/scale handling is done host-side by the caller: `weights[j,t]` must
// already be the renormalized routing weight (top-k softmax / sum, clamped)
// times any per-expert down scale (down_exps_s[e]); this function only does
// the elementwise multiply + expert-sum. Per-expert gate/up scales are NOT
// supported here (Gemma-4 Q4_K has none); callers with those must use the
// per-expert path.
//
// Shapes (ggml, ne[0] innermost):
//   gate_up_exps : [n_embd, 2*n_ff_exp, n_expert]  (merged; pass null for split)
//   gate_exps    : [n_embd, n_ff_exp,   n_expert]  (split path)
//   up_exps      : [n_embd, n_ff_exp,   n_expert]  (split path)
//   down_exps    : [n_ff_exp, n_embd,   n_expert]
//   moe_in       : host F32 [n_embd, n_tokens]
//   ids          : host I32  [n_expert_used, n_tokens] (n_expert_used innermost)
//   weights      : host F32  [n_expert_used, n_tokens] (same layout as ids)
//   moe_out      : host F32 [n_embd, n_tokens] (output; caller applies post-norm)
//
// Allocates a private, right-sized ggml context per call (independent of the
// shared mm.arena) and runs it on mm.pool. Requires the expert tensors to be
// host-resident (-ngl 0).
bool matmul_moe_id_qf32(MatmulCtx & mm,
                        const ggml_tensor * gate_up_exps,
                        const ggml_tensor * gate_exps,
                        const ggml_tensor * up_exps,
                        const ggml_tensor * down_exps,
                        const float * moe_in,
                        const int32_t * ids,
                        const float * weights,
                        float * moe_out,
                        int n_embd, int n_ff_exp,
                        int n_expert_used, int n_tokens,
                        std::string & error);

// G6.2 - fused greedy lm_head + argmax. Computes
//   out_tok = argmax_v ( sum_e W[e, v] * x[e] )
// where W is the (tied) output-embedding weight [n_embd, n_vocab] in any
// ggml quantized/float type. x is a host F32 vector of length n_embd (the
// post-output-norm hidden state for the single decode token). The dot for
// each vocab row uses the same CPU vec_dot ggml_mul_mat would use (x is
// quantized once to W's vec_dot_type), so the winning token is identical to
// argmax over the full matmul logits. The softcap is intentionally NOT
// applied (monotonic -> argmax-invariant). Rows are split across mm.pool
// workers with per-worker running maxima reduced at the end; ties resolve
// to the lowest vocab index (matching std::max_element on the logits).
bool lmhead_argmax_qf32(MatmulCtx & mm, const ggml_tensor * W,
                        const float * x, int n_embd, int n_vocab,
                        int32_t & out_tok, std::string & error);

} // namespace gemma4
