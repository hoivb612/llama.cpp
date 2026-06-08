#include "gemma4_matmul.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace gemma4 {

void GgmlThreadpoolDeleter::operator()(ggml_threadpool * p) const noexcept {
    if (p) ggml_threadpool_free(p);
}

// ---------------------------------------------------------------------------
// G5.1 - AttnWorkerPool implementation (see gemma4_matmul.h for design).
// ---------------------------------------------------------------------------

namespace {

inline void cpu_pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    // no-op
#endif
}

} // namespace

struct AttnWorkerPool {
    int n_threads = 1;                       // total workers including caller
    std::vector<std::thread> workers;        // n_threads - 1 helpers
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> job_seq{0};
    std::atomic<int> done_count{0};
    std::mutex mtx;
    std::condition_variable cv_work;
    AttnJobFn fn = nullptr;
    void * user_data = nullptr;
};

void AttnPoolDeleter::operator()(AttnWorkerPool * p) const noexcept {
    if (!p) return;
    {
        std::lock_guard<std::mutex> lk(p->mtx);
        p->stop.store(true, std::memory_order_release);
    }
    p->cv_work.notify_all();
    for (auto & w : p->workers) {
        if (w.joinable()) w.join();
    }
    delete p;
}

static void attn_worker_loop(AttnWorkerPool * pool, int wid) {
    uint64_t last_seq = 0;
    while (true) {
        AttnJobFn fn = nullptr;
        void * ud = nullptr;
        int W = 1;
        uint64_t seq;
        {
            std::unique_lock<std::mutex> lk(pool->mtx);
            pool->cv_work.wait(lk, [&]{
                return pool->stop.load(std::memory_order_acquire) ||
                       pool->job_seq.load(std::memory_order_acquire) != last_seq;
            });
            if (pool->stop.load(std::memory_order_acquire)) return;
            seq = pool->job_seq.load(std::memory_order_acquire);
            fn  = pool->fn;
            ud  = pool->user_data;
            W   = pool->n_threads;
        }
        last_seq = seq;
        if (fn) fn(wid, W, ud);
        // Release: caller acquires done_count and must see all writes in fn.
        pool->done_count.fetch_add(1, std::memory_order_release);
    }
}

static bool attn_pool_init(MatmulCtx & mm, int n_threads, std::string & error) {
    mm.attn_pool.reset();
    if (n_threads <= 1) return true;

    std::unique_ptr<AttnWorkerPool, AttnPoolDeleter> p(new AttnWorkerPool());
    p->n_threads = n_threads;
    try {
        p->workers.reserve((size_t) (n_threads - 1));
        for (int i = 0; i < n_threads - 1; ++i) {
            p->workers.emplace_back(attn_worker_loop, p.get(), i + 1);
        }
    } catch (const std::exception & e) {
        error = std::string("attn_pool_init: thread spawn failed: ") + e.what();
        return false;
    }
    mm.attn_pool = std::move(p);
    return true;
}

void attn_pool_run(MatmulCtx & mm, AttnJobFn fn, void * user_data) {
    AttnWorkerPool * pool = mm.attn_pool.get();
    if (!pool || pool->n_threads <= 1 || !fn) {
        // Serial fallback: caller does all the work as wid=0 of W=1.
        if (fn) fn(0, 1, user_data);
        return;
    }
    const int W = pool->n_threads;

    pool->done_count.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(pool->mtx);
        pool->fn        = fn;
        pool->user_data = user_data;
        // Release under the lock so workers waking via the predicate observe
        // fn/user_data after the bump. The mutex itself also establishes
        // happens-before for any in-flight wait() that resumes after lock.
        pool->job_seq.fetch_add(1, std::memory_order_release);
    }
    pool->cv_work.notify_all();

    // Main-as-worker: this thread is wid 0.
    fn(0, W, user_data);

    // Wait for n_threads - 1 helpers. Acquire to make their writes visible.
    const int target = W - 1;
    for (;;) {
        const int d = pool->done_count.load(std::memory_order_acquire);
        if (d >= target) break;
        // Short pause-spin: each shard is sub-ms so we stay hot, but avoid
        // burning the SMT sibling. yield() is overkill here on Windows.
        for (int i = 0; i < 16; ++i) cpu_pause();
    }
}

// ---------------------------------------------------------------------------
// G5.1 - global attn-parallel toggle. Set once from CLI before any decode.
// ---------------------------------------------------------------------------
static std::atomic<bool> g_attn_parallel{false};

void set_attn_parallel(bool on) {
    g_attn_parallel.store(on, std::memory_order_relaxed);
}
bool get_attn_parallel() {
    return g_attn_parallel.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------

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
    mm.attn_pool.reset();

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

    // G5.1 - spin up the attention worker pool with the same thread count.
    // Failures here cascade out so the caller knows the matmul shim is
    // partially initialised.
    if (!attn_pool_init(mm, mm.n_threads, error)) {
        mm.pool.reset();
        return false;
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
