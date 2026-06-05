#include "gemma4_matmul.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace gemma4 {

void GgmlThreadpoolDeleter::operator()(ggml_threadpool * p) const noexcept {
    if (p) ggml_threadpool_free(p);
}

bool matmul_ctx_init(MatmulCtx & mm, std::size_t arena_bytes, int n_threads,
                     std::string & error) {
    if (arena_bytes < (1u << 20)) {
        error = "matmul_ctx_init: arena too small (need >= 1 MiB)";
        return false;
    }
    mm.arena.assign(arena_bytes, 0);
    mm.work_buf.clear();
    mm.n_threads = std::max(1, n_threads);

    // Tear down any old pool first (init may be called more than once).
    mm.pool.reset();

    // Spin up a persistent threadpool when we have real parallelism. For
    // n_threads == 1 we stay on the single-thread compute path (saves a
    // worker thread sitting idle and one extra dependency tear-down).
    if (mm.n_threads > 1) {
        ggml_threadpool_params params = ggml_threadpool_params_default(mm.n_threads);
        ggml_threadpool * raw = ggml_threadpool_new(&params);
        if (!raw) {
            error = "matmul_ctx_init: ggml_threadpool_new failed";
            return false;
        }
        mm.pool.reset(raw);
    }
    return true;
}

bool matmul_qf32(MatmulCtx & mm, const ggml_tensor * W,
                 const float * x_in, float * y_out,
                 int n_in, int n_out, int n_cols, std::string & error) {
    if (!W) { error = "matmul_qf32: W is null"; return false; }
    if (mm.arena.empty()) { error = "matmul_qf32: MatmulCtx not initialized"; return false; }
    if (W->ne[0] != n_in || W->ne[1] != n_out) {
        std::ostringstream ss;
        ss << "matmul_qf32: W shape [" << W->ne[0] << "," << W->ne[1]
           << "] != expected [" << n_in << "," << n_out << "]";
        error = ss.str();
        return false;
    }

    ggml_init_params ip{ mm.arena.size(), mm.arena.data(), /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "matmul_qf32: ggml_init failed"; return false; }

    // Wrap host x_in as a tensor in our arena.
    ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_cols);
    if (!x_t) { error = "matmul_qf32: alloc x_t failed"; ggml_free(ctx); return false; }
    std::memcpy(x_t->data, x_in, (std::size_t) n_in * n_cols * sizeof(float));

    // y = W @ x. ggml_mul_mat reads W and x from their respective
    // contexts; the result tensor lives in our context.
    ggml_tensor * y_t = ggml_mul_mat(ctx, const_cast<ggml_tensor *>(W), x_t);
    if (!y_t) { error = "matmul_qf32: ggml_mul_mat returned null"; ggml_free(ctx); return false; }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y_t);

    ggml_status status;
    if (mm.pool) {
        // Multi-thread path: plan with the persistent pool and reuse the
        // shared work_buf. The pool's worker threads stay alive across
        // calls, eliminating the per-call spawn/join.
        ggml_cplan cplan = ggml_graph_plan(gf, mm.n_threads, mm.pool.get());
        if (cplan.work_size > mm.work_buf.size()) {
            mm.work_buf.assign(cplan.work_size, 0);
        }
        cplan.work_data = mm.work_buf.empty() ? nullptr : mm.work_buf.data();
        status = ggml_graph_compute(gf, &cplan);
    } else {
        // Single-thread fallback (n_threads <= 1): no need for a pool.
        status = ggml_graph_compute_with_ctx(ctx, gf, mm.n_threads);
    }

    if (status != GGML_STATUS_SUCCESS) {
        error = "matmul_qf32: graph compute failed";
        ggml_free(ctx);
        return false;
    }

    std::memcpy(y_out, y_t->data, (std::size_t) n_out * n_cols * sizeof(float));
    ggml_free(ctx);
    return true;
}

} // namespace gemma4
