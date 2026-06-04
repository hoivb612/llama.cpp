#include "phi3_fused_ops.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama_b612.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#  include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#endif

namespace {

// Pull weight bytes into a host-visible mirror so we can call the CPU vec_dot
// kernel directly. For non-host (e.g. CUDA) backends this is a one-time copy
// at init; for host backends we just borrow the live data pointer.
bool resolve_weight_data(Phi3FusedLmHead & lm, std::string & error) {
    const ggml_tensor * w = lm.weight;
    if (!w) {
        error = "phi3 fused lm_head: weight tensor not set";
        return false;
    }

    ggml_backend_buffer_t buf = w->buffer;
    if (buf && ggml_backend_buffer_is_host(buf) && w->data) {
        lm.weight_data = static_cast<const uint8_t *>(w->data);
        lm.host_weight.clear();
        return true;
    }

    const size_t total = ggml_nbytes(w);
    lm.host_weight.assign(total, 0);
    ggml_backend_tensor_get(w, lm.host_weight.data(), 0, total);
    lm.weight_data = lm.host_weight.data();
    return true;
}

} // namespace

// Tight spin used in both worker poll-loops and driver wait. We have plenty of
// cores idle during decode (the lm_head argmax burst is < 2 ms total), so
// burning a few hundred microseconds of cycles here is preferable to
// std::this_thread::yield(), which can introduce millisecond-scale wake-up
// latency on Windows when the OS deschedules the spinner.
static inline void phi3_cpu_relax() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

bool phi3_fused_lmhead_init(const llama_model * model, Phi3FusedLmHead & out, std::string & error) {
    out = {};

    if (!model) {
        error = "phi3 fused lm_head: model is null";
        return false;
    }

    // Prefer explicit lm_head weight; fall back to tied token-embedding for
    // small Phi3 variants that share the matrix.
    const ggml_tensor * w = llama_model_get_tensor_by_name(model, "output.weight");
    if (!w) {
        w = llama_model_get_tensor_by_name(model, "token_embd.weight");
    }
    if (!w) {
        error = "phi3 fused lm_head: cannot find output.weight or token_embd.weight";
        return false;
    }

    const auto * w_traits = ggml_get_type_traits_cpu(w->type);
    if (!w_traits || !w_traits->vec_dot) {
        error = "phi3 fused lm_head: weight type has no CPU vec_dot kernel";
        return false;
    }

    const ggml_type q_type   = w_traits->vec_dot_type;
    const auto *    q_traits = ggml_get_type_traits_cpu(q_type);
    if (!q_traits || !q_traits->from_float) {
        error = "phi3 fused lm_head: vec_dot_type has no from_float kernel";
        return false;
    }

    out.weight       = w;
    out.n_embd       = w->ne[0];
    out.n_vocab      = w->ne[1];
    out.weight_type  = (int) w->type;
    out.quant_type   = (int) q_type;
    out.weight_row_bytes = ggml_row_size(w->type, out.n_embd);

    out.quant_hidden.assign(ggml_row_size(q_type, out.n_embd), 0);

    if (!resolve_weight_data(out, error)) {
        return false;
    }

    error.clear();
    return true;
}

bool phi3_fused_lmhead_argmax(
        Phi3FusedLmHead & lm,
        Phi3LmHeadPool *  pool,
        const float *     hidden,
        int               n_threads,
        llama_token &     out_token,
        std::string &     error) {
    if (!hidden) {
        error = "phi3 fused lm_head: hidden state is null";
        return false;
    }
    if (lm.n_vocab <= 0 || lm.n_embd <= 0 || !lm.weight_data) {
        error = "phi3 fused lm_head: not initialized";
        return false;
    }

    const auto * w_traits = ggml_get_type_traits_cpu((ggml_type) lm.weight_type);
    const auto * q_traits = ggml_get_type_traits_cpu((ggml_type) lm.quant_type);
    if (!w_traits || !w_traits->vec_dot || !q_traits || !q_traits->from_float) {
        error = "phi3 fused lm_head: missing CPU traits";
        return false;
    }

    // Once-per-step: quantize hidden state into the format expected by vec_dot.
    q_traits->from_float(hidden, lm.quant_hidden.data(), lm.n_embd);

    const uint8_t * w_base       = lm.weight_data;
    const size_t    row_bytes    = lm.weight_row_bytes;
    const int64_t   n_vocab      = lm.n_vocab;
    const int       n_embd_int   = (int) lm.n_embd;
    const void *    q_hidden_ptr = lm.quant_hidden.data();

    // Persistent-pool fast path (Option 3): publish job state and spin-wait
    // on done_count. Avoids the ~50-100 us/thread spawn cost x N threads x 256
    // steps that dominated the prior std::thread fallback.
    if (pool && pool->initialized && pool->n_threads >= 2) {
        const int N = pool->n_threads;

        // Slice [0, n_vocab) into N contiguous chunks.
        const int64_t chunk = (n_vocab + N - 1) / N;
        for (int wid = 0; wid < N; ++wid) {
            const int64_t lo = (int64_t) wid * chunk;
            const int64_t hi = std::min<int64_t>(lo + chunk, n_vocab);
            pool->slots[wid].lo       = lo;
            pool->slots[wid].hi       = hi;
            pool->slots[wid].best_val = -std::numeric_limits<float>::infinity();
            pool->slots[wid].best_id  = lo;
        }
        pool->w_traits     = w_traits;
        pool->w_base       = w_base;
        pool->row_bytes    = row_bytes;
        pool->n_embd       = n_embd_int;
        pool->q_hidden_ptr = q_hidden_ptr;

        // Reset done_count BEFORE publishing job_seq so workers don't observe a
        // stale count from a previous dispatch.
        pool->done_count.store(0, std::memory_order_release);
        // Publish under the cv mutex so a parked worker can't miss the
        // wake-up (classic cv pattern: change predicate, then notify).
        {
            std::lock_guard<std::mutex> lk(pool->mtx);
            pool->job_seq.fetch_add(1, std::memory_order_acq_rel);
        }
        pool->cv_work.notify_all();

        // Driver tight-spins on done_count. The ~1 ms argmax burst happens
        // immediately after llama_decode returns, so spinning here keeps the
        // driver core hot and avoids any further wake-up latency.
        while (pool->done_count.load(std::memory_order_acquire) < N) {
            phi3_cpu_relax();
        }

        // Reduce N partials.
        float   best_val = -std::numeric_limits<float>::infinity();
        int64_t best_id  = 0;
        for (int wid = 0; wid < N; ++wid) {
            const auto & s = pool->slots[wid];
            if (s.hi <= s.lo) continue;
            if (s.best_val > best_val) {
                best_val = s.best_val;
                best_id  = s.best_id;
            }
        }
        out_token = (llama_token) best_id;
        error.clear();
        return true;
    }

    int n_workers = n_threads > 0 ? n_threads : 1;
    if (n_workers > n_vocab) {
        n_workers = (int) n_vocab;
    }

    if (n_workers <= 1) {
        float   best_val = -std::numeric_limits<float>::infinity();
        int64_t best_id  = 0;
        for (int64_t v = 0; v < n_vocab; ++v) {
            float s = 0.0f;
            w_traits->vec_dot(n_embd_int, &s, 0, w_base + (size_t) v * row_bytes, 0, q_hidden_ptr, 0, 1);
            if (s > best_val) {
                best_val = s;
                best_id  = v;
            }
        }
        out_token = (llama_token) best_id;
        error.clear();
        return true;
    }

    // Legacy fallback: per-call std::thread spawn. Retained for the case where
    // the pool is not provided (e.g. one-shot init failure, single-thread mode).
    struct Slot { float val; int64_t id; };
    std::vector<Slot> partials(n_workers, Slot{-std::numeric_limits<float>::infinity(), 0});
    std::vector<std::thread> workers;
    workers.reserve(n_workers);

    const int64_t chunk = (n_vocab + n_workers - 1) / n_workers;
    for (int wid = 0; wid < n_workers; ++wid) {
        const int64_t lo = (int64_t) wid * chunk;
        const int64_t hi = std::min<int64_t>(lo + chunk, n_vocab);
        if (lo >= hi) {
            continue;
        }
        workers.emplace_back([&, wid, lo, hi]() {
            float   best_val = -std::numeric_limits<float>::infinity();
            int64_t best_id  = lo;
            for (int64_t v = lo; v < hi; ++v) {
                float s = 0.0f;
                w_traits->vec_dot(n_embd_int, &s, 0, w_base + (size_t) v * row_bytes, 0, q_hidden_ptr, 0, 1);
                if (s > best_val) {
                    best_val = s;
                    best_id  = v;
                }
            }
            partials[wid].val = best_val;
            partials[wid].id  = best_id;
        });
    }
    for (auto & t : workers) {
        t.join();
    }

    float   best_val = -std::numeric_limits<float>::infinity();
    int64_t best_id  = 0;
    for (const auto & p : partials) {
        if (p.val > best_val) {
            best_val = p.val;
            best_id  = p.id;
        }
    }
    out_token = (llama_token) best_id;
    error.clear();
    return true;
}

namespace {

void phi3_lmhead_worker_loop(Phi3LmHeadPool * pool, int wid) {
    uint64_t last_seq = 0;
    while (true) {
        // Park on cv until the driver publishes a new job (or signals stop).
        // Parking matters because the driver thread runs ggml decode on its
        // own threadpool for ~40 ms between argmax calls; spinning during
        // decode would steal cycles from those workers.
        {
            std::unique_lock<std::mutex> lk(pool->mtx);
            pool->cv_work.wait(lk, [&]{
                return pool->stop.load(std::memory_order_acquire) ||
                       pool->job_seq.load(std::memory_order_acquire) != last_seq;
            });
            if (pool->stop.load(std::memory_order_acquire)) {
                return;
            }
            last_seq = pool->job_seq.load(std::memory_order_acquire);
        }

        // Acquire-load on job_seq above synchronizes with driver's release store,
        // so the per-job state below is safe to read.
        const auto *   w_traits  = pool->w_traits;
        const uint8_t * w_base    = pool->w_base;
        const size_t   row_bytes = pool->row_bytes;
        const int      n_embd    = pool->n_embd;
        const void *   q_hidden  = pool->q_hidden_ptr;
        auto &         s         = pool->slots[wid];

        const int64_t lo = s.lo;
        const int64_t hi = s.hi;
        if (lo < hi && w_traits && w_traits->vec_dot) {
            float   best_val = -std::numeric_limits<float>::infinity();
            int64_t best_id  = lo;
            for (int64_t v = lo; v < hi; ++v) {
                float ss = 0.0f;
                w_traits->vec_dot(n_embd, &ss, 0, w_base + (size_t) v * row_bytes, 0, q_hidden, 0, 1);
                if (ss > best_val) {
                    best_val = ss;
                    best_id  = v;
                }
            }
            s.best_val = best_val;
            s.best_id  = best_id;
        }

        pool->done_count.fetch_add(1, std::memory_order_acq_rel);
    }
}

} // namespace

bool phi3_fused_lmhead_pool_init(Phi3LmHeadPool & pool, int n_threads, std::string & error) {
    phi3_fused_lmhead_pool_free(pool);
    if (n_threads <= 1) {
        // Pool disabled; argmax will fall back to single-thread or legacy spawn.
        pool.n_threads   = 0;
        pool.initialized = false;
        error.clear();
        return true;
    }
    pool.n_threads = n_threads;
    pool.slots.assign(n_threads, Phi3LmHeadPool::Slot{});
    pool.stop.store(false, std::memory_order_release);
    pool.job_seq.store(0, std::memory_order_release);
    pool.done_count.store(0, std::memory_order_release);
    pool.workers.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        pool.workers.emplace_back(phi3_lmhead_worker_loop, &pool, i);
    }
    pool.initialized = true;
    error.clear();
    return true;
}

void phi3_fused_lmhead_pool_free(Phi3LmHeadPool & pool) {
    if (!pool.initialized && pool.workers.empty()) {
        pool.slots.clear();
        pool.n_threads = 0;
        return;
    }
    pool.stop.store(true, std::memory_order_release);
    // Bump job_seq AND notify under the cv mutex so parked workers observe both.
    {
        std::lock_guard<std::mutex> lk(pool.mtx);
        pool.job_seq.fetch_add(1, std::memory_order_acq_rel);
    }
    pool.cv_work.notify_all();
    for (auto & t : pool.workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    pool.workers.clear();
    pool.slots.clear();
    pool.initialized = false;
    pool.n_threads = 0;
}


// ---------------------------------------------------------------------------
// Phi3MatmulPool — spin-policy pool for the Phase A custom forward.
// See header for design rationale.
// ---------------------------------------------------------------------------

// A4.1 — returns true iff weight type `t` is in the _x8 repacked family. For
// these types BOTH the trait-pointed vec_dot variants (xx_vec_dot_q{N}_k_q8_k_x8
// and its _dc fallback) accept the batched call signature with `by=ncols` and
// `nrc=nrows`. The cp variant tiles 4 activation cols per weight load (the win);
// the dc variant iterates cols one-at-a-time but still emits correct output.
// Either way our batched call is safe regardless of which #if branch was compiled.
// Public so callers (e.g., phi3_fused_graph.cpp) can gate batched prefill setup.
bool phi3_x8_batched_eligible(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_K_x8:
        case GGML_TYPE_Q3_K_x8:
        case GGML_TYPE_Q6_K_x8:
            return true;
        default:
            return false;
    }
}

namespace {

inline void phi3_cpu_pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    // No-op on architectures without a pause hint; the wall-clock guards prevent
    // busy-spinning from getting out of hand.
#endif
}

void phi3_matmul_compute_range(const Phi3MatmulJob & j, int lo, int hi) {
    if (!j.w_traits || !j.w_traits->vec_dot || lo >= hi) {
        return;
    }

    // A4.1 — batched path: one trait-pointer call per worker for the full row
    // slice * all M_act columns. Eligibility requires the caller to have laid
    // out src_q (M_act columns of q8_K_repack super-blocks back-to-back) and
    // dst (column-major, stride dst_col_stride_bytes between cols). The trait
    // pointer's signature is positionally compatible: `bs -> nr_nb1`,
    // `by -> ncols`, `nrc -> nrows`.
    if (j.M_act > 1 && phi3_x8_batched_eligible(j.w_type) && j.dst_col_stride_bytes != 0) {
        j.w_traits->vec_dot(
            j.K,
            j.dst + lo,
            j.dst_col_stride_bytes,
            j.w_base + (size_t) lo * j.w_row_bytes, 0,
            j.src_q,
            (size_t) j.M_act,
            hi - lo);
        return;
    }

    for (int v = lo; v < hi; ++v) {
        float s = 0.0f;
        j.w_traits->vec_dot(j.K, &s, 0,
                            j.w_base + (size_t) v * j.w_row_bytes, 0,
                            j.src_q, 0, 1);
        j.dst[v] = s;
    }
}

// A3.y — per-head attention worker. Computes K*Q -> softmax -> V*scores for
// heads [h_lo, h_hi) using this worker's private scratch slice. Bit-identical
// to the serial loop in phi3_layer_forward_qquant section 5: same op order
// per head, same per-head F32/F16 reductions, and each head writes a
// disjoint slice of `ctx_base` so the cross-head order is irrelevant.
void phi3_attn_compute_heads(const Phi3AttnJob & j, int wid, int W) {
    if (W <= 0 || j.f16_traits == nullptr || j.f16_traits->vec_dot == nullptr ||
        j.f16_traits->from_float == nullptr) {
        return;
    }
    const int n_head    = j.n_head;
    const int n_head_kv = j.n_head_kv;
    const int head_dim  = j.head_dim;
    const int new_len   = j.new_len;
    const int ctx_max_v = j.ctx_max_v;
    const float scale_q = j.scale_q;

    const int h_lo = (int) (((int64_t) wid       * n_head) / W);
    const int h_hi = (int) (((int64_t) (wid + 1) * n_head) / W);
    if (h_lo >= h_hi) return;

    ggml_fp16_t * q_h_f16    = j.q_h_f16_base    + (size_t) wid * (size_t) j.stride_q_f16;
    float       * scores     = j.scores_base     + (size_t) wid * (size_t) j.stride_s_f32;
    ggml_fp16_t * scores_f16 = j.scores_f16_base + (size_t) wid * (size_t) j.stride_s_f16;

    const auto * tr = j.f16_traits;

    for (int h = h_lo; h < h_hi; ++h) {
        const float * q_h = j.q_base + (size_t) h * (size_t) head_dim;
        tr->from_float(q_h, q_h_f16, head_dim);

        float max_s = -std::numeric_limits<float>::infinity();
        for (int p = 0; p < new_len; ++p) {
            const ggml_fp16_t * k_p = j.K_base + ((size_t) p * (size_t) n_head_kv + (size_t) h) * (size_t) head_dim;
            float s = 0.0f;
            tr->vec_dot(head_dim, &s, 0, k_p, 0, q_h_f16, 0, 1);
            s *= scale_q;
            scores[p] = s;
            if (s > max_s) max_s = s;
        }
        float sum_exp = 0.0f;
        for (int p = 0; p < new_len; ++p) {
            scores[p] = std::exp(scores[p] - max_s);
            sum_exp += scores[p];
        }
        const float inv_sum = 1.0f / sum_exp;
        for (int p = 0; p < new_len; ++p) {
            scores[p] *= inv_sum;
        }
        tr->from_float(scores, scores_f16, new_len);

        float * ctx_h = j.ctx_base + (size_t) h * (size_t) head_dim;
        for (int d = 0; d < head_dim; ++d) {
            const ggml_fp16_t * v_strip =
                j.V_base + ((size_t) d * (size_t) n_head_kv + (size_t) h) * (size_t) ctx_max_v;
            float s = 0.0f;
            tr->vec_dot(new_len, &s, 0, v_strip, 0, scores_f16, 0, 1);
            ctx_h[d] = s;
        }
    }
}

void phi3_matmul_worker_loop(Phi3MatmulPool * pool, int wid) {
    uint64_t last_seq = 0;
    while (true) {
        // ---- Three-phase backoff ----
        // Phase 1: tight pause for spin_phase1_us.
        // Phase 2: pause loop with wall-clock check until spin_phase2_us.
        // Phase 3: cv-park until job_seq advances or stop is set.

        const auto t_phase_start = std::chrono::steady_clock::now();
        const auto phase1_deadline = t_phase_start + std::chrono::microseconds(pool->spin_phase1_us);
        const auto phase2_deadline = t_phase_start + std::chrono::microseconds(pool->spin_phase2_us);

        uint64_t seq = pool->job_seq.load(std::memory_order_acquire);
        bool got_work = false;

        // Phase 1: tight pause loop (~10 us). No wall-clock check inside.
        for (int i = 0; i < 256 && seq == last_seq; ++i) {
            phi3_cpu_pause();
            seq = pool->job_seq.load(std::memory_order_acquire);
        }
        if (seq != last_seq) {
            got_work = !pool->stop.load(std::memory_order_acquire);
        } else if (std::chrono::steady_clock::now() < phase1_deadline) {
            while (seq == last_seq && std::chrono::steady_clock::now() < phase1_deadline) {
                phi3_cpu_pause();
                seq = pool->job_seq.load(std::memory_order_acquire);
            }
            if (seq != last_seq) {
                got_work = !pool->stop.load(std::memory_order_acquire);
            }
        }

        // Phase 2: continued pause with wall-clock check, until phase2_deadline.
        if (seq == last_seq) {
            while (seq == last_seq && std::chrono::steady_clock::now() < phase2_deadline) {
                for (int i = 0; i < 64; ++i) {
                    phi3_cpu_pause();
                }
                seq = pool->job_seq.load(std::memory_order_acquire);
            }
            if (seq != last_seq) {
                got_work = !pool->stop.load(std::memory_order_acquire);
            }
        }

        // Phase 3: park on cv with predicate.
        if (seq == last_seq) {
            std::unique_lock<std::mutex> lk(pool->mtx);
            pool->cv_work.wait(lk, [&]{
                return pool->stop.load(std::memory_order_acquire) ||
                       pool->job_seq.load(std::memory_order_acquire) != last_seq;
            });
            seq = pool->job_seq.load(std::memory_order_acquire);
            got_work = !pool->stop.load(std::memory_order_acquire);
        }

        if (pool->stop.load(std::memory_order_acquire)) {
            return;
        }
        if (!got_work || seq == last_seq) {
            continue;
        }
        last_seq = seq;

        // Acquire on job_seq above synchronizes with the driver's release store,
        // so the job struct fields are safe to read.
        const Phi3PoolJobKind kind = pool->job_kind;
        if (kind == Phi3PoolJobKind::ATTN) {
            const Phi3AttnJob j = pool->attn_job;
            phi3_attn_compute_heads(j, wid, pool->n_threads);
        } else {
            const Phi3MatmulJob j = pool->job;
            const int N = j.N_total;
            const int W = pool->n_threads;
            const int lo = (int) (((int64_t) wid * N) / W);
            const int hi = (int) (((int64_t) (wid + 1) * N) / W);
            phi3_matmul_compute_range(j, lo, hi);
        }

        pool->done_count.fetch_add(1, std::memory_order_acq_rel);
    }
}

} // namespace

bool phi3_matmul_pool_init(Phi3MatmulPool & pool, int n_threads, std::string & error) {
    phi3_matmul_pool_free(pool);
    if (n_threads <= 1) {
        pool.n_threads   = 0;
        pool.initialized = false;
        error.clear();
        return true;
    }
    pool.n_threads = n_threads;
    pool.stop.store(false, std::memory_order_release);
    pool.job_seq.store(0, std::memory_order_release);
    pool.done_count.store(0, std::memory_order_release);
    pool.workers.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        pool.workers.emplace_back(phi3_matmul_worker_loop, &pool, i);
    }
    pool.initialized = true;
    error.clear();
    return true;
}

void phi3_matmul_pool_free(Phi3MatmulPool & pool) {
    if (!pool.initialized && pool.workers.empty()) {
        pool.n_threads = 0;
        return;
    }
    pool.stop.store(true, std::memory_order_release);
    // Bump job_seq AND notify under cv mutex so parked workers observe both.
    {
        std::lock_guard<std::mutex> lk(pool.mtx);
        pool.job_seq.fetch_add(1, std::memory_order_acq_rel);
    }
    pool.cv_work.notify_all();
    for (auto & t : pool.workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    pool.workers.clear();
    pool.initialized = false;
    pool.n_threads = 0;
}

void phi3_matmul_pool_run(Phi3MatmulPool * pool, const Phi3MatmulJob & job) {
    if (!pool || !pool->initialized || pool->n_threads <= 1) {
        // Fallback: run on the calling thread.
        phi3_matmul_compute_range(job, 0, job.N_total);
        return;
    }

    pool->job      = job;
    pool->job_kind = Phi3PoolJobKind::MATMUL;
    pool->done_count.store(0, std::memory_order_release);
    // Publish job under cv mutex so phase-3-parked workers see the seq bump
    // AND get woken. Workers in phases 1/2 observe via the acquire load.
    {
        std::lock_guard<std::mutex> lk(pool->mtx);
        pool->job_seq.fetch_add(1, std::memory_order_acq_rel);
    }
    pool->cv_work.notify_all();

    // Driver tight-spins on done_count for the short matmul burst.
    const int target = pool->n_threads;
    while (pool->done_count.load(std::memory_order_acquire) < target) {
        phi3_cpu_pause();
    }
}

void phi3_attn_pool_run(Phi3MatmulPool * pool, const Phi3AttnJob & job) {
    if (!pool || !pool->initialized || pool->n_threads <= 1) {
        // Fallback: run on the calling thread. The driver supplies its own
        // scratch slice at wid=0; pool stride fields may be unset in that
        // case, but the helper only indexes wid * stride = 0.
        phi3_attn_compute_heads(job, /*wid=*/0, /*W=*/1);
        return;
    }

    pool->attn_job = job;
    pool->job_kind = Phi3PoolJobKind::ATTN;
    pool->done_count.store(0, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(pool->mtx);
        pool->job_seq.fetch_add(1, std::memory_order_acq_rel);
    }
    pool->cv_work.notify_all();

    // Driver tight-spins on done_count for the attention burst.
    const int target = pool->n_threads;
    while (pool->done_count.load(std::memory_order_acquire) < target) {
        phi3_cpu_pause();
    }
}
