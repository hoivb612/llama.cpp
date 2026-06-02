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
