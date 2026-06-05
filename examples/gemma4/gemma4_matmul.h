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

    int n_threads = 1;
};

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
