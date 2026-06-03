#pragma once

#include "llama.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct ggml_tensor;

// Fused lm_head + greedy argmax for Phi3 Phase C.
//
// When the Phi3 graph is built with cparams.fused_lmhead=true, the standard
// lm_head matmul is pruned and the post-final-norm hidden state is exposed via
// the embeddings buffer. This helper computes the next greedy token directly
// from that hidden state using the existing per-row vec_dot kernel from
// ggml_type_traits_cpu (Q4_K_M -> AVX-512 q4_K x q8_K on x86).
//
// Greedy-only. Callers must not feed the resulting token to a sampler chain
// that requires full logits.
struct Phi3FusedLmHead {
    const struct ggml_tensor * weight = nullptr; // [n_embd, n_vocab]
    int64_t                    n_embd = 0;
    int64_t                    n_vocab = 0;
    int                        weight_type = 0; // ggml_type of the weight tensor
    int                        quant_type  = 0; // ggml_type the hidden state must be quantized to (vec_dot_type)
    std::vector<uint8_t>       quant_hidden;    // scratch buffer for the q-quantized hidden state
    std::vector<uint8_t>       host_weight;     // mirror of the weight bytes (used iff backend buffer is not host-visible)
    const uint8_t *            weight_data = nullptr;
    size_t                     weight_row_bytes = 0;
};

// Option 3: persistent worker pool used by phi3_fused_lmhead_argmax to avoid
// per-step std::thread spawn/join overhead (which dominated the sample time at
// ~2.4 ms/step with 10 workers on Phi3-mini). Workers spin on an atomic
// job-sequence counter; the driver thread publishes the job state and waits
// (also spinning) for done_count == n_threads. Use only from a single driver
// thread (the runtime loop). Pool is sized at init and not resizable.
struct Phi3LmHeadPool {
    struct Slot {
        int64_t lo = 0;
        int64_t hi = 0;
        float   best_val = 0.0f;
        int64_t best_id  = 0;
    };

    int                       n_threads = 0;
    bool                      initialized = false;

    // Shared per-job state; driver writes before each kick, workers read after seeing job_seq change.
    const void *              q_hidden_ptr = nullptr;
    const uint8_t *           w_base       = nullptr;
    size_t                    row_bytes    = 0;
    int                       n_embd       = 0;
    const struct ggml_type_traits_cpu * w_traits = nullptr;

    std::vector<Slot>         slots;
    std::vector<std::thread>  workers;

    // Workers park on cv_work between jobs to avoid stealing cycles from the
    // decode threadpool during the ~40 ms decode phase. Driver tight-spins on
    // done_count for the short (~1 ms) argmax burst.
    std::mutex                          mtx;
    std::condition_variable             cv_work;
    alignas(64) std::atomic<uint64_t>   job_seq{0};      // incremented by driver per dispatch
    alignas(64) std::atomic<int>        done_count{0};   // incremented by workers as they finish
    alignas(64) std::atomic<bool>       stop{false};
};

// Resolve the lm_head weight tensor on the model and pre-allocate the scratch
// buffer for the per-step q-quantization of the hidden state.
// Returns false (and sets error) when the tensor is missing or the type is
// unsupported by the CPU type traits.
bool phi3_fused_lmhead_init(const struct llama_model * model, Phi3FusedLmHead & out, std::string & error);

// Spawn n_threads persistent worker threads. n_threads <= 1 leaves the pool
// uninitialized (caller falls back to single-thread argmax). Safe to call on
// an already-initialized pool: existing workers are joined and replaced.
bool phi3_fused_lmhead_pool_init(Phi3LmHeadPool & pool, int n_threads, std::string & error);

// Signal stop, wake spinners, join workers, reset state. Idempotent.
void phi3_fused_lmhead_pool_free(Phi3LmHeadPool & pool);

// Compute argmax_v ( weight[:, v] . hidden ) and write the winning row index.
// hidden must point to n_embd contiguous f32 values (e.g. from
// llama_get_embeddings_ith(ctx, -1)). When `pool` is non-null and initialized,
// the parallel reduction runs on the persistent pool; otherwise falls back to
// the legacy per-call std::thread spawn (or single-thread loop if n_threads<=1).
bool phi3_fused_lmhead_argmax(
    Phi3FusedLmHead & lm,
    Phi3LmHeadPool *  pool,
    const float *     hidden,
    int               n_threads,
    llama_token &     out_token,
    std::string &     error);

// ---------------------------------------------------------------------------
// Phi3MatmulPool — Phase A: persistent matmul-row pool for the custom Phi-3
// forward pass.
//
// Distinct from Phi3LmHeadPool above. The lm-head pool fires ONCE per token
// and cv-parks between calls so it doesn't steal cycles from the ggml decode
// threadpool during the ~40 ms graph evaluation. The matmul pool fires
// ~128 times per token (4 matmuls × 32 layers) and the cv-park overhead at
// that frequency would dominate; workers use a wall-clock-measured 3-phase
// backoff (tight pause → light pause with deadline → cv-park) so the
// typical sub-millisecond gap stays in the spin phases.
//
// Job shape: a single ggml-style K-quant × Q8_K-family matmul. Workers split
// the output-row range [0, N_total) evenly. Single driver thread (the
// runtime loop) publishes the job via job_seq and tight-spins on done_count.
// ---------------------------------------------------------------------------

struct Phi3MatmulJob {
    const struct ggml_type_traits_cpu * w_traits = nullptr;
    const uint8_t *                     w_base   = nullptr;  // base of weight matrix (row-contiguous K-quant)
    size_t                              w_row_bytes = 0;     // ggml_row_size(w->type, K)
    const void *                        src_q    = nullptr;  // pre-quantized src (q_type = w_traits->vec_dot_type)
    float *                             dst      = nullptr;  // F32 output, length N_total
    int                                 K        = 0;        // inner dim
    int                                 N_total  = 0;        // total output rows
};

struct Phi3MatmulPool {
    int  n_threads   = 0;
    bool initialized = false;

    Phi3MatmulJob job;

    std::vector<std::thread>          workers;
    std::mutex                        mtx;
    std::condition_variable           cv_work;

    alignas(64) std::atomic<uint64_t> job_seq{0};
    alignas(64) std::atomic<int>      done_count{0};
    alignas(64) std::atomic<bool>     stop{false};

    // Idle backoff thresholds (microseconds) — see spec §4. Phase 1 is tight
    // _mm_pause; phase 2 is _mm_pause with wall-clock check; phase 3 is
    // cv-park. Tunable; A2.5 will measure.
    int  spin_phase1_us = 10;
    int  spin_phase2_us = 1000;
};

// Spawn n_threads persistent worker threads. n_threads <= 1 leaves the pool
// uninitialized (caller falls back to single-thread serial matmul).
// Safe to call on an already-initialized pool: existing workers are joined
// and replaced.
bool phi3_matmul_pool_init(Phi3MatmulPool & pool, int n_threads, std::string & error);

// Signal stop, wake spinners, join workers, reset state. Idempotent.
void phi3_matmul_pool_free(Phi3MatmulPool & pool);

// Dispatch one matmul job and wait for completion. When pool is null or
// uninitialized, runs the matmul on the calling thread. The job struct is
// copied into the pool; caller may reuse its storage immediately after this
// returns.
void phi3_matmul_pool_run(Phi3MatmulPool * pool, const Phi3MatmulJob & job);
