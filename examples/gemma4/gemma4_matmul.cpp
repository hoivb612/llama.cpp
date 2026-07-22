#include "gemma4_matmul.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <limits>
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
    // Share the ggml threadpool's CCX-spread placement (no-op unless
    // GGML_B612_CCX_SPREAD is set). Pins attn helper `wid` to the same CCX as
    // ggml worker `wid`; the two pools never run concurrently, so co-locating
    // them keeps this pool off the pinned ggml cores' fabric contention.
    ggml_b612_ccx_pin_self(wid);
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

static std::atomic<bool> g_moe_fused{true};

void set_moe_fused(bool on) {
    g_moe_fused.store(on, std::memory_order_relaxed);
}
bool get_moe_fused() {
    return g_moe_fused.load(std::memory_order_relaxed);
}

static std::atomic<bool> g_lmhead_fused{true};

void set_lmhead_fused(bool on) {
    g_lmhead_fused.store(on, std::memory_order_relaxed);
}
bool get_lmhead_fused() {
    return g_lmhead_fused.load(std::memory_order_relaxed);
}

static std::atomic<bool> g_prefill_fused{true};

void set_prefill_fused(bool on) {
    g_prefill_fused.store(on, std::memory_order_relaxed);
}
bool get_prefill_fused() {
    return g_prefill_fused.load(std::memory_order_relaxed);
}

static std::atomic<bool> g_repack_active{false};

void set_repack_active(bool on) {
    g_repack_active.store(on, std::memory_order_relaxed);
}
bool get_repack_active() {
    return g_repack_active.load(std::memory_order_relaxed);
}

// Phase 2 - repack one resident expert bank [n_in, n_out, n_expert] in place.
// The bank is a stack of n_expert contiguous [n_in, n_out] slabs (expert dim
// outermost). ggml_cpu_repack_tensor_callgraph() (XBCG) repacks the src0 of
// each MUL_MAT node in place and flips that node's src0->type to the _x8
// variant, but the persistent bank tensor's own ->type is not touched by it.
// So we build a throwaway graph with one MUL_MAT per expert slab (each slab a
// distinct 2D src0 aliasing the bank's data at expert*nb[2]), run the
// callgraph to repack every slab's bytes, then flip the bank tensor's ->type
// once. The per-expert 2D view in matmul_expert_qf32 then inherits the _x8
// type and the fast kernel engages. Repack is a pure in-place reordering, so
// the bank's nb[] (row/slab byte strides) stay valid.
bool repack_expert_bank(const ggml_tensor * bank_c, std::string & error) {
    if (!bank_c) return true;
    ggml_tensor * bank = const_cast<ggml_tensor *>(bank_c);
    if (bank->ne[2] <= 0) return true;
    // Only XBCG at graph-build time actually repacks; the callgraph is a
    // no-op otherwise, and single_thread rejects n_in % 256 != 0 anyway.
    if (bank->ne[0] % 256 != 0) return true;   // e.g. down bank (n_ff_exp=704)

    const int64_t ne0 = bank->ne[0];
    const int64_t ne1 = bank->ne[1];
    const int64_t ne2 = bank->ne[2];           // n_expert
    const enum ggml_type base_type = bank->type;

    const size_t n_nodes = (size_t) ne2;
    const size_t mem = ggml_tensor_overhead() * (n_nodes * 2 + 8)
                     + ggml_graph_overhead_custom(n_nodes + 8, false)
                     + 4096;
    ggml_init_params ip{ mem, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "repack_expert_bank: ggml_init failed"; return false; }

    // Shared F32 rhs; no_alloc so data stays null (the callgraph never reads
    // it -- it only checks src1->type == F32).
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 1);
    if (!rhs) { error = "repack_expert_bank: alloc rhs failed"; ggml_free(ctx); return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes + 8, false);
    if (!gf) { error = "repack_expert_bank: new_graph failed"; ggml_free(ctx); return false; }

    for (int64_t e = 0; e < ne2; ++e) {
        ggml_tensor * slab = ggml_new_tensor_2d(ctx, base_type, ne0, ne1);
        if (!slab) { error = "repack_expert_bank: alloc slab failed"; ggml_free(ctx); return false; }
        // A fresh contiguous 2D tensor of the same type/ne0 has the same
        // row stride as the bank, so pointing its data at expert e's slab
        // makes single_thread repack exactly that slab's bytes.
        slab->data = (char *) bank->data + (size_t) e * bank->nb[2];
        ggml_tensor * y = ggml_mul_mat(ctx, slab, rhs);
        if (!y) { error = "repack_expert_bank: mul_mat failed"; ggml_free(ctx); return false; }
        ggml_build_forward_expand(gf, y);
    }

    ggml_cpu_repack_tensor_callgraph(gf);

    // Flip the bank type once to whatever the slabs became (unchanged if the
    // mode is off / not XBCG -- then this is a harmless no-op).
    enum ggml_type new_type = base_type;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        if (n->op == GGML_OP_MUL_MAT && n->src[0]) { new_type = n->src[0]->type; break; }
    }
    bank->type = new_type;
    ggml_free(ctx);
    return true;
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
    mm.moe_cache.clear();

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

    // Phase 1 dense-weight repack (XBCG). No-op unless a repack mode is set.
    // XBCG repacks src0 (W, the model weight) in place and flips its type;
    // planning below then reads the repacked type so the fast _x8 kernel is
    // selected. e.w_type is captured after this so the cache invariant guard
    // sees the post-repack type and does not thrash-rebuild.
    ggml_cpu_repack_tensor_callgraph(e.gf);

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

    // Phase 1 dense-weight repack (XBCG). No-op unless a repack mode is set.
    // Prefill (large n_cols) reaches this per-call path first and repacks W
    // in place once; later calls see the already-repacked type and skip.
    ggml_cpu_repack_tensor_callgraph(gf);

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

bool matmul_expert_qf32(MatmulCtx & mm, const ggml_tensor * W3d, int expert,
                        const float * x_in, float * y_out,
                        int n_in, int n_out, int n_cols, std::string & error) {
    if (!W3d) { error = "matmul_expert_qf32: W3d is null"; return false; }
    if (mm.arena.empty()) { error = "matmul_expert_qf32: MatmulCtx not initialized"; return false; }
    if (W3d->ne[0] != n_in || W3d->ne[1] != n_out) {
        std::ostringstream ss;
        ss << "matmul_expert_qf32: W3d shape [" << W3d->ne[0] << "," << W3d->ne[1] << "," << W3d->ne[2]
           << "] != expected [" << n_in << "," << n_out << ",*]";
        error = ss.str();
        return false;
    }
    if (expert < 0 || expert >= W3d->ne[2]) {
        std::ostringstream ss;
        ss << "matmul_expert_qf32: expert " << expert << " out of range [0," << W3d->ne[2] << ")";
        error = ss.str();
        return false;
    }

    ggml_init_params ip{ mm.arena.size(), mm.arena.data(), /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "matmul_expert_qf32: ggml_init failed"; return false; }

    ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_cols);
    if (!x_t) { error = "matmul_expert_qf32: alloc x_t failed"; ggml_free(ctx); return false; }
    std::memcpy(x_t->data, x_in, (std::size_t) n_in * n_cols * sizeof(float));

    // 2D view over expert e's contiguous sub-block. Expert dim is outermost
    // so the block begins at expert*nb[2] and keeps the native row stride.
    ggml_tensor * We = ggml_view_2d(ctx, const_cast<ggml_tensor *>(W3d),
                                    n_in, n_out, W3d->nb[1],
                                    (std::size_t) expert * W3d->nb[2]);
    if (!We) { error = "matmul_expert_qf32: ggml_view_2d returned null"; ggml_free(ctx); return false; }

    ggml_tensor * y_t = ggml_mul_mat(ctx, We, x_t);
    if (!y_t) { error = "matmul_expert_qf32: ggml_mul_mat returned null"; ggml_free(ctx); return false; }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y_t);

    ggml_status status;
    if (mm.pool) {
        ggml_cplan cplan = ggml_graph_plan(gf, mm.n_threads, mm.pool.get());
        if (cplan.work_size > mm.work_buf.size()) {
            mm.work_buf.assign(cplan.work_size, 0);
        }
        cplan.work_data = mm.work_buf.empty() ? nullptr : mm.work_buf.data();
        status = ggml_graph_compute(gf, &cplan);
    } else {
        status = ggml_graph_compute_with_ctx(ctx, gf, mm.n_threads);
    }

    if (status != GGML_STATUS_SUCCESS) {
        error = "matmul_expert_qf32: graph compute failed";
        ggml_free(ctx);
        return false;
    }

    std::memcpy(y_out, y_t->data, (std::size_t) n_out * n_cols * sizeof(float));
    ggml_free(ctx);
    return true;
}

bool matmul_qblock_qf32(MatmulCtx & mm, int w_type, const void * block,
                        const float * x_in, float * y_out,
                        int n_in, int n_out, int n_cols, std::string & error) {
    if (!block) { error = "matmul_qblock_qf32: block is null"; return false; }
    if (mm.arena.empty()) { error = "matmul_qblock_qf32: MatmulCtx not initialized"; return false; }

    ggml_init_params ip{ mm.arena.size(), mm.arena.data(), /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "matmul_qblock_qf32: ggml_init failed"; return false; }

    ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_cols);
    if (!x_t) { error = "matmul_qblock_qf32: alloc x_t failed"; ggml_free(ctx); return false; }
    std::memcpy(x_t->data, x_in, (std::size_t) n_in * n_cols * sizeof(float));

    // Wrap the streamed block as a 2D quant tensor. new_tensor_2d computes
    // the native contiguous strides (nb[1] = ggml_row_size(type, n_in)),
    // which match the block's on-disk layout; repoint its data at `block`
    // (the arena bytes it reserved are left unused -- cheap and simple).
    ggml_tensor * We = ggml_new_tensor_2d(ctx, (ggml_type) w_type, n_in, n_out);
    if (!We) { error = "matmul_qblock_qf32: alloc We failed"; ggml_free(ctx); return false; }
    We->data = const_cast<void *>(block);

    ggml_tensor * y_t = ggml_mul_mat(ctx, We, x_t);
    if (!y_t) { error = "matmul_qblock_qf32: ggml_mul_mat returned null"; ggml_free(ctx); return false; }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y_t);

    ggml_status status;
    if (mm.pool) {
        ggml_cplan cplan = ggml_graph_plan(gf, mm.n_threads, mm.pool.get());
        if (cplan.work_size > mm.work_buf.size()) {
            mm.work_buf.assign(cplan.work_size, 0);
        }
        cplan.work_data = mm.work_buf.empty() ? nullptr : mm.work_buf.data();
        status = ggml_graph_compute(gf, &cplan);
    } else {
        status = ggml_graph_compute_with_ctx(ctx, gf, mm.n_threads);
    }

    if (status != GGML_STATUS_SUCCESS) {
        error = "matmul_qblock_qf32: graph compute failed";
        ggml_free(ctx);
        return false;
    }

    std::memcpy(y_out, y_t->data, (std::size_t) n_out * n_cols * sizeof(float));
    ggml_free(ctx);
    return true;
}

// Build the fused-MoE subgraph inside `ctx`. On success, sets the four
// tensor handles (inputs x/ids/weights and the output `moe`). The graph is
// NOT computed here -- caller builds/plans/computes and (for the cache)
// keeps the tensors around to refresh + recompute on subsequent calls.
static bool build_moe_graph(ggml_context * ctx,
                            const ggml_tensor * gate_up_exps,
                            const ggml_tensor * gate_exps,
                            const ggml_tensor * up_exps,
                            const ggml_tensor * down_exps,
                            bool merged,
                            int n_embd, int n_expert_used, int n_tokens,
                            ggml_tensor ** out_x, ggml_tensor ** out_ids,
                            ggml_tensor ** out_w, ggml_tensor ** out_moe,
                            std::string & error) {
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    if (!x) { error = "matmul_moe_id_qf32: alloc x failed"; return false; }
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, n_embd, 1, n_tokens); // broadcast over used dim

    ggml_tensor * ids_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
    if (!ids_t) { error = "matmul_moe_id_qf32: alloc ids failed"; return false; }

    ggml_tensor * w_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_expert_used, n_tokens);
    if (!w_t) { error = "matmul_moe_id_qf32: alloc weights failed"; return false; }

    // ---- gate_up (+ split) ----
    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;
    if (merged) {
        ggml_tensor * gate_up = ggml_mul_mat_id(ctx, const_cast<ggml_tensor *>(gate_up_exps),
                                                x3, ids_t); // [2*n_ff_exp, used, tokens]
        if (!gate_up) { error = "matmul_moe_id_qf32: mul_mat_id(gate_up) failed"; return false; }
        const int64_t n_ff = gate_up->ne[0] / 2;
        gate = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2],
                            gate_up->nb[1], gate_up->nb[2], 0);
        up   = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2],
                            gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
    } else {
        up   = ggml_mul_mat_id(ctx, const_cast<ggml_tensor *>(up_exps),   x3, ids_t);
        gate = ggml_mul_mat_id(ctx, const_cast<ggml_tensor *>(gate_exps), x3, ids_t);
    }
    if (!gate || !up) { error = "matmul_moe_id_qf32: mul_mat_id(gate/up) failed"; return false; }

    // geglu(gate, up) = gelu(gate) * up  -> [n_ff_exp, used, tokens]
    ggml_tensor * act = ggml_geglu_split(ctx, gate, up);
    if (!act) { error = "matmul_moe_id_qf32: geglu_split failed"; return false; }

    // down projection -> [n_embd, used, tokens]
    ggml_tensor * experts = ggml_mul_mat_id(ctx, const_cast<ggml_tensor *>(down_exps), act, ids_t);
    if (!experts) { error = "matmul_moe_id_qf32: mul_mat_id(down) failed"; return false; }

    // apply routing weights (broadcast [1, used, tokens] over n_embd)
    experts = ggml_mul(ctx, experts, w_t);
    if (!experts) { error = "matmul_moe_id_qf32: mul(weights) failed"; return false; }

    // sum the n_expert_used contributions (views + adds, mirroring upstream)
    ggml_tensor * moe = ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0);
    for (int i = 1; i < n_expert_used; ++i) {
        ggml_tensor * ei = ggml_view_2d(ctx, experts, n_embd, n_tokens,
                                        experts->nb[2], (std::size_t) i * experts->nb[1]);
        moe = ggml_add(ctx, moe, ei);
    }
    if (n_expert_used == 1) {
        moe = ggml_cont(ctx, moe); // avoid a non-contiguous single-view result
    }
    if (!moe) { error = "matmul_moe_id_qf32: expert-sum failed"; return false; }

    *out_x   = x;
    *out_ids = ids_t;
    *out_w   = w_t;
    *out_moe = moe;
    return true;
}

// Eligibility for the fused-MoE graph cache: real pool + small n_tokens
// (decode). Prefill shapes (large n_tokens) recur at most once per length,
// so caching is pure overhead there.
static bool moe_cache_eligible(const MatmulCtx & mm, int n_tokens) {
    return get_matmul_cache() && mm.pool && n_tokens == 1;
}

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
                        std::string & error) {
    if (!down_exps) { error = "matmul_moe_id_qf32: down_exps is null"; return false; }
    const bool merged = (gate_up_exps != nullptr);
    if (!merged && (!gate_exps || !up_exps)) {
        error = "matmul_moe_id_qf32: need merged gate_up_exps or both gate_exps+up_exps";
        return false;
    }
    if (n_tokens <= 0 || n_expert_used <= 0) {
        error = "matmul_moe_id_qf32: n_tokens/n_expert_used must be > 0";
        return false;
    }

    const std::size_t nt  = (std::size_t) n_tokens;
    const std::size_t neu = (std::size_t) n_expert_used;

    // ---- G5.3 fused-MoE graph cache (decode fast path) ----
    if (moe_cache_eligible(mm, n_tokens)) {
        MoeCacheKey key{ down_exps, n_tokens };
        auto it = mm.moe_cache.find(key);
        if (it != mm.moe_cache.end()) {
            MoeCacheEntry & e = it->second;
            // Guard against a repack/model swap changing the expert tensors
            // or thread count under a recycled down_exps pointer.
            if (e.merged != merged || e.n_expert_used != n_expert_used ||
                e.n_embd != n_embd || e.n_ff_exp != n_ff_exp ||
                e.n_threads != mm.n_threads ||
                e.gate_up_exps != gate_up_exps || e.gate_exps != gate_exps ||
                e.up_exps != up_exps || e.down_exps != down_exps) {
                mm.moe_cache.erase(it);
                it = mm.moe_cache.end();
            }
        }
        if (it == mm.moe_cache.end()) {
            MoeCacheEntry e;
            std::size_t bytes = 0;
            bytes += (std::size_t) n_embd * nt * sizeof(float);
            bytes += neu * nt * sizeof(int32_t);
            bytes += neu * nt * sizeof(float);
            bytes += (std::size_t) 2 * n_ff_exp * neu * nt * sizeof(float);
            bytes += (std::size_t) n_ff_exp * neu * nt * sizeof(float);
            bytes += (std::size_t) 2 * n_embd * neu * nt * sizeof(float);
            bytes += (std::size_t) n_embd * nt * sizeof(float) * neu;
            bytes = bytes * 2 + ((std::size_t) 32u << 20);
            e.arena.resize(bytes);
            ggml_init_params ip{ e.arena.size(), e.arena.data(), /*no_alloc=*/false };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) { error = "matmul_moe_id_qf32: ggml_init (cache) failed"; return false; }
            e.ctx.reset(ctx);
            if (!build_moe_graph(ctx, gate_up_exps, gate_exps, up_exps, down_exps,
                                 merged, n_embd, n_expert_used, n_tokens,
                                 &e.x_t, &e.ids_t, &e.w_t, &e.moe, error)) {
                return false;
            }
            e.gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(e.gf, e.moe);
            e.cplan = ggml_graph_plan(e.gf, mm.n_threads, mm.pool.get());
            e.n_embd = n_embd; e.n_ff_exp = n_ff_exp;
            e.n_expert_used = n_expert_used; e.n_tokens = n_tokens;
            e.n_threads = mm.n_threads; e.merged = merged;
            e.gate_up_exps = gate_up_exps; e.gate_exps = gate_exps;
            e.up_exps = up_exps; e.down_exps = down_exps;
            auto ins = mm.moe_cache.emplace(key, std::move(e));
            it = ins.first;
        }

        MoeCacheEntry & e = it->second;
        std::memcpy(e.x_t->data,   moe_in,  (std::size_t) n_embd * nt * sizeof(float));
        std::memcpy(e.ids_t->data, ids,     neu * nt * sizeof(int32_t));
        std::memcpy(e.w_t->data,   weights, neu * nt * sizeof(float));

        if (e.cplan.work_size > mm.work_buf.size()) {
            mm.work_buf.assign(e.cplan.work_size, 0);
        }
        e.cplan.work_data = mm.work_buf.empty() ? nullptr : mm.work_buf.data();
        ggml_status status = ggml_graph_compute(e.gf, &e.cplan);
        if (status != GGML_STATUS_SUCCESS) {
            error = "matmul_moe_id_qf32: graph compute (cache) failed";
            return false;
        }
        std::memcpy(moe_out, e.moe->data, (std::size_t) n_embd * nt * sizeof(float));
        return true;
    }

    // ---- uncached path (prefill / cache disabled) ----
    // Right-size a private context (independent of mm.arena): sum of the F32
    // intermediates plus generous headroom for tensor metadata + graph nodes.
    std::size_t bytes = 0;
    bytes += (std::size_t) n_embd * nt * sizeof(float);            // x
    bytes += neu * nt * sizeof(int32_t);                          // ids
    bytes += neu * nt * sizeof(float);                            // weights
    bytes += (std::size_t) 2 * n_ff_exp * neu * nt * sizeof(float); // gate_up
    bytes += (std::size_t) n_ff_exp * neu * nt * sizeof(float);   // geglu
    bytes += (std::size_t) 2 * n_embd * neu * nt * sizeof(float); // down + weighted
    bytes += (std::size_t) n_embd * nt * sizeof(float) * neu;     // expert-sum chain
    bytes = bytes * 2 + ((std::size_t) 32u << 20);                // headroom + metadata

    ggml_init_params ip{ bytes, nullptr, /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "matmul_moe_id_qf32: ggml_init failed"; return false; }

    ggml_tensor * x = nullptr;
    ggml_tensor * ids_t = nullptr;
    ggml_tensor * w_t = nullptr;
    ggml_tensor * moe = nullptr;
    if (!build_moe_graph(ctx, gate_up_exps, gate_exps, up_exps, down_exps,
                         merged, n_embd, n_expert_used, n_tokens,
                         &x, &ids_t, &w_t, &moe, error)) {
        ggml_free(ctx);
        return false;
    }
    std::memcpy(x->data,     moe_in,  (std::size_t) n_embd * nt * sizeof(float));
    std::memcpy(ids_t->data, ids,     neu * nt * sizeof(int32_t));
    std::memcpy(w_t->data,   weights, neu * nt * sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, moe);

    ggml_status status;
    if (mm.pool) {
        ggml_cplan cplan = ggml_graph_plan(gf, mm.n_threads, mm.pool.get());
        if (cplan.work_size > mm.work_buf.size()) {
            mm.work_buf.assign(cplan.work_size, 0);
        }
        cplan.work_data = mm.work_buf.empty() ? nullptr : mm.work_buf.data();
        status = ggml_graph_compute(gf, &cplan);
    } else {
        status = ggml_graph_compute_with_ctx(ctx, gf, mm.n_threads);
    }
    if (status != GGML_STATUS_SUCCESS) {
        error = "matmul_moe_id_qf32: graph compute failed";
        ggml_free(ctx);
        return false;
    }

    std::memcpy(moe_out, moe->data, (std::size_t) n_embd * nt * sizeof(float));
    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------
// G6.2 - fused greedy lm_head + argmax
// ---------------------------------------------------------------------------
namespace {

struct LmHeadArgmaxJob {
    const char *   w_base    = nullptr;
    std::size_t    row_bytes = 0;
    const void *   xq        = nullptr;
    int            n_embd    = 0;
    int            n_vocab   = 0;
    ggml_vec_dot_t vec_dot   = nullptr;
    float *        best_score = nullptr; // [n_workers]
    int32_t *      best_idx   = nullptr; // [n_workers]
};

void lmhead_argmax_worker(int wid, int W, void * ud) {
    LmHeadArgmaxJob * j = static_cast<LmHeadArgmaxJob *>(ud);
    const int64_t nv = j->n_vocab;
    const int64_t lo = nv * wid / W;
    const int64_t hi = nv * (wid + 1) / W;
    float   best = -std::numeric_limits<float>::infinity();
    int32_t bidx = -1;
    for (int64_t v = lo; v < hi; ++v) {
        float s = 0.0f;
        j->vec_dot(j->n_embd, &s, 0, j->w_base + (std::size_t) v * j->row_bytes, 0, j->xq, 0, 1);
        if (s > best) { best = s; bidx = (int32_t) v; }
    }
    j->best_score[wid] = best;
    j->best_idx[wid]   = bidx;
}

} // namespace

bool lmhead_argmax_qf32(MatmulCtx & mm, const ggml_tensor * W,
                        const float * x, int n_embd, int n_vocab,
                        int32_t & out_tok, std::string & error) {
    if (!W)         { error = "lmhead_argmax_qf32: W is null"; return false; }
    if (!W->data)   { error = "lmhead_argmax_qf32: W->data is null (non-CPU tensor?)"; return false; }
    if (n_embd <= 0 || n_vocab <= 0) { error = "lmhead_argmax_qf32: bad dims"; return false; }
    if (W->ne[0] != n_embd || W->ne[1] != n_vocab) {
        error = "lmhead_argmax_qf32: W shape mismatch";
        return false;
    }

    const struct ggml_type_traits_cpu * wtr = ggml_get_type_traits_cpu(W->type);
    if (!wtr || !wtr->vec_dot) { error = "lmhead_argmax_qf32: no vec_dot for W type"; return false; }
    const ggml_type vdt = wtr->vec_dot_type;
    const struct ggml_type_traits_cpu * qtr = ggml_get_type_traits_cpu(vdt);
    if (!qtr || !qtr->from_float) { error = "lmhead_argmax_qf32: no from_float for vec_dot_type"; return false; }

    // Quantize the single activation column once into W's vec_dot layout.
    const std::size_t xq_bytes = ggml_row_size(vdt, n_embd);
    std::vector<uint8_t> xq(xq_bytes);
    qtr->from_float(x, xq.data(), n_embd);

    const int n_workers = std::max(1, mm.n_threads);
    std::vector<float>   best_score((std::size_t) n_workers, -std::numeric_limits<float>::infinity());
    std::vector<int32_t> best_idx((std::size_t) n_workers, -1);

    LmHeadArgmaxJob job;
    job.w_base     = static_cast<const char *>(W->data);
    job.row_bytes  = ggml_row_size(W->type, n_embd);
    job.xq         = xq.data();
    job.n_embd     = n_embd;
    job.n_vocab    = n_vocab;
    job.vec_dot    = wtr->vec_dot;
    job.best_score = best_score.data();
    job.best_idx   = best_idx.data();

    // attn_pool_run splits [0, n_workers) across the persistent pool (main as
    // worker 0) or falls back to a serial fn(0, 1) call when there is no pool.
    attn_pool_run(mm, lmhead_argmax_worker, &job);

    // Reduce; workers own contiguous ascending vocab ranges and each keeps
    // the first (lowest) index among ties, so scanning worker 0..n-1 with a
    // strict-greater test yields the lowest global argmax index -- matching
    // std::max_element over the full logit vector.
    float   best = -std::numeric_limits<float>::infinity();
    int32_t bidx = -1;
    for (int w = 0; w < n_workers; ++w) {
        if (best_idx[w] >= 0 && best_score[w] > best) {
            best = best_score[w];
            bidx = best_idx[w];
        }
    }
    if (bidx < 0) { error = "lmhead_argmax_qf32: no valid argmax"; return false; }
    out_tok = bidx;
    return true;
}

} // namespace gemma4
