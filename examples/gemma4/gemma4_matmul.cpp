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

void GgmlContextDeleter::operator()(ggml_context * c) const noexcept {
    if (c) ggml_free(c);
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
// G5.2 - matmul graph cache toggle. Default ON for the qquant path; pass
// --gemma4-matmul-cache 0 to revert to the per-call build path.
// ---------------------------------------------------------------------------
static std::atomic<bool> g_matmul_cache{true};

void set_matmul_cache(bool on) {
    g_matmul_cache.store(on, std::memory_order_relaxed);
}
bool get_matmul_cache() {
    return g_matmul_cache.load(std::memory_order_relaxed);
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

    // G5.2 - drop any cached graphs first; their cplans hold a pointer to
    // the old mm.pool and assume the old mm.work_buf sizing. Must clear
    // BEFORE resetting pool/work_buf below.
    mm.matmul_cache.clear();

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

// ---------------------------------------------------------------------------
// G5.2 - matmul graph cache helpers
// ---------------------------------------------------------------------------

namespace {

// Eligibility gate for the cache. Only cache when we have a real pool
// (n_threads > 1: the original ggml_graph_compute_with_ctx fallback for
// single-thread allocates its own work_data inside the per-call ctx, which
// is not safe to reuse) and when n_cols is small enough that the per-entry
// arena stays cheap. Prefill (n_cols >> 1) skips the cache because each
// (W, n_cols) shape is only seen once per session, so the build cost is
// pure overhead without amortisation.
constexpr int kMaxCachedNCols = 1;

bool cache_eligible(const MatmulCtx & mm, int n_cols) {
    return get_matmul_cache() && mm.pool && n_cols >= 1 && n_cols <= kMaxCachedNCols;
}

// Arena size for one cached entry. Holds:
//   * x_t metadata + data        = ggml_tensor_overhead() + n_in*n_cols*4
//   * y_t metadata + data        = ggml_tensor_overhead() + n_out*n_cols*4
//   * one mul_mat node           = ggml_tensor_overhead() (no extra data)
//   * graph object               = ggml_graph_overhead_custom(4, false)
//   * ggml internal padding/objects -- add a 16 KiB margin to be safe.
// work_data is NOT stored here; it lives in mm.work_buf and is wired into
// cplan.work_data per call.
std::size_t entry_arena_bytes(int n_in, int n_out, int n_cols) {
    const std::size_t x_bytes = (std::size_t) n_in  * (std::size_t) n_cols * sizeof(float);
    const std::size_t y_bytes = (std::size_t) n_out * (std::size_t) n_cols * sizeof(float);
    const std::size_t graph   = ggml_graph_overhead_custom(/*size=*/4, /*grads=*/false);
    const std::size_t tensors = 3 * ggml_tensor_overhead();   // x, y, mul_mat node
    const std::size_t margin  = 16u * 1024u;
    return x_bytes + y_bytes + graph + tensors + margin;
}

// Build a fresh cache entry for (W, n_cols). Returns false on error.
bool build_cache_entry(MatmulCtx & mm, const ggml_tensor * W,
                       int n_in, int n_out, int n_cols,
                       MatmulCacheEntry & e, std::string & error) {
    e.arena.assign(entry_arena_bytes(n_in, n_out, n_cols), 0);

    ggml_init_params ip{ e.arena.size(), e.arena.data(), /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "matmul_qf32 cache: ggml_init failed"; return false; }
    e.ctx.reset(ctx);

    e.x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_cols);
    if (!e.x_t) { error = "matmul_qf32 cache: alloc x_t failed"; e.ctx.reset(); return false; }

    e.y_t = ggml_mul_mat(ctx, const_cast<ggml_tensor *>(W), e.x_t);
    if (!e.y_t) { error = "matmul_qf32 cache: ggml_mul_mat returned null"; e.ctx.reset(); return false; }

    // Tiny custom graph -- 4 node slots is plenty for x + mul_mat + y wrap.
    e.gf = ggml_new_graph_custom(ctx, /*size=*/4, /*grads=*/false);
    if (!e.gf) { error = "matmul_qf32 cache: ggml_new_graph_custom failed"; e.ctx.reset(); return false; }
    ggml_build_forward_expand(e.gf, e.y_t);

    e.cplan = ggml_graph_plan(e.gf, mm.n_threads, mm.pool.get());
    if (e.cplan.work_size > mm.work_buf.size()) {
        mm.work_buf.assign(e.cplan.work_size, 0);
    }
    // work_data is refreshed per call below; leave it dangling here so a
    // forgotten refresh trips immediately.

    e.n_in      = n_in;
    e.n_out     = n_out;
    e.n_cols    = n_cols;
    e.w_type    = (int) W->type;
    e.n_threads = mm.n_threads;
    return true;
}

} // namespace

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

    // -----------------------------------------------------------------------
    // G5.2 - cached path. On a hit, skip every per-call build step and go
    // straight to memcpy x -> ggml_graph_compute -> memcpy y.
    // -----------------------------------------------------------------------
    if (cache_eligible(mm, n_cols)) {
        const MatmulCacheKey key{ W, n_cols };
        auto it = mm.matmul_cache.find(key);
        if (it != mm.matmul_cache.end()) {
            // Invariant guard: w_type / n_threads must match the values at
            // build time. b612 has paths that may repack W in place, so a
            // type drift triggers a rebuild instead of using a stale cplan.
            MatmulCacheEntry & cur = it->second;
            if (cur.w_type != (int) W->type || cur.n_threads != mm.n_threads) {
                mm.matmul_cache.erase(it);
                it = mm.matmul_cache.end();
            }
        }
        if (it == mm.matmul_cache.end()) {
            MatmulCacheEntry e;
            if (!build_cache_entry(mm, W, n_in, n_out, n_cols, e, error)) {
                return false;
            }
            auto ins = mm.matmul_cache.emplace(key, std::move(e));
            it = ins.first;
        }
        MatmulCacheEntry & e = it->second;

        // Defensive: another entry may have grown mm.work_buf since we built
        // this cplan; the vector may also have been reallocated, so refresh
        // work_data each call.
        if (e.cplan.work_size > mm.work_buf.size()) {
            mm.work_buf.assign(e.cplan.work_size, 0);
        }
        e.cplan.work_data = mm.work_buf.empty() ? nullptr : mm.work_buf.data();

        std::memcpy(e.x_t->data, x_in,
                    (std::size_t) n_in * n_cols * sizeof(float));
        const ggml_status status = ggml_graph_compute(e.gf, &e.cplan);
        if (status != GGML_STATUS_SUCCESS) {
            error = "matmul_qf32 cache: graph compute failed";
            return false;
        }
        std::memcpy(y_out, e.y_t->data,
                    (std::size_t) n_out * n_cols * sizeof(float));
        return true;
    }

    // -----------------------------------------------------------------------
    // Original per-call build path. Used for n_cols > kMaxCachedNCols
    // (prefill), single-thread fallback (no pool), or when caching is
    // disabled via --gemma4-matmul-cache 0.
    // -----------------------------------------------------------------------
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
