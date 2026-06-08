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
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_threadpool;

namespace gemma4 {

// Custom deleter for ggml_threadpool so MatmulCtx is self-cleaning.
struct GgmlThreadpoolDeleter {
    void operator()(ggml_threadpool * p) const noexcept;
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

} // namespace gemma4
