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
struct ggml_type_traits_cpu;

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
    float *                             dst      = nullptr;  // F32 output, length N_total (or N_total * M_act if batched)
    int                                 K        = 0;        // inner dim
    int                                 N_total  = 0;        // total output rows

    // A4.1 — batched prefill matmul support. When M_act > 1 AND w_type is in the
    // x8 repacked family (--repack-ggml or --repack-xbcg), the worker calls into
    // xx_vec_dot_q{N}_k_q8_k_x8 once per row slice for ALL M_act activation
    // columns at once, amortizing weight memory traffic ~4x. When M_act == 1 or
    // w_type is not in the x8 family, falls back to the existing per-row path.
    //   - src_q layout when M_act>1: M_act columns laid back-to-back, each column
    //     is `nb` consecutive `block_q{N}_K_repack` super-blocks.
    //   - dst layout when M_act>1: column-major, dst[col*N_total + row].
    //   - dst_col_stride_bytes = N_total * sizeof(float) for the standard layout.
    int                                 M_act               = 1;
    size_t                              dst_col_stride_bytes = 0;
    enum ggml_type                      w_type              = GGML_TYPE_COUNT;
};

// A4.1 — returns true iff weight type `t` has an associated batched _x8 matmul
// kernel that this module can dispatch directly. Callers should use this to
// decide whether to lay out activations + dst in batched form before calling
// into the pool with M_act > 1. False for non-repacked types (NONE or XBOX).
bool phi3_x8_batched_eligible(enum ggml_type t);

// A3.y — per-head attention job dispatched through the same persistent pool.
// Workers each take a contiguous range of attention heads; outputs write to
// disjoint slices of `ctx_base` so no synchronization is needed inside the
// dispatch. Scratch (q_h_f16 / scores / scores_f16) is per-thread, cache-line
// padded by the caller to avoid false sharing.
//
// Bit-identical to the serial per-head loop in phi3_layer_forward_qquant
// section 5: each head's K*Q -> softmax -> V*scores sequence is preserved
// in op order; only the order between heads (which writes to disjoint
// output) changes.
struct Phi3AttnJob {
    // Per-attention inputs.
    const float *                       q_base    = nullptr;  // [n_head * head_dim] F32
    const ggml_fp16_t *                 K_base    = nullptr;  // K cache layer base, pos-major [new_len, n_head_kv, head_dim] F16
    const ggml_fp16_t *                 V_base    = nullptr;  // V cache layer base, transposed [head_dim, n_head_kv, ctx_max_v] F16
    float *                             ctx_base  = nullptr;  // [n_head * head_dim] F32 output

    int                                 n_head     = 0;
    int                                 n_head_kv  = 0;
    int                                 head_dim   = 0;
    int                                 new_len    = 0;
    int                                 ctx_max_v  = 0;
    float                               scale_q    = 0.0f;

    // F16 type traits (vec_dot + from_float). Same pointer used for K*Q,
    // V*scores, and the per-head Q quantization step.
    const struct ggml_type_traits_cpu * f16_traits = nullptr;

    // Per-thread scratch. Each worker w reads/writes its own slice starting
    // at base + w * stride. Strides are in element units.
    ggml_fp16_t *                       q_h_f16_base    = nullptr;  // [W * stride_q_f16]
    float *                             scores_base     = nullptr;  // [W * stride_s_f32]
    ggml_fp16_t *                       scores_f16_base = nullptr;  // [W * stride_s_f16]
    int                                 stride_q_f16    = 0;
    int                                 stride_s_f32    = 0;
    int                                 stride_s_f16    = 0;
};

enum class Phi3PoolJobKind : int {
    MATMUL = 0,
    ATTN   = 1,
};

struct Phi3MatmulPool {
    int  n_threads   = 0;
    bool initialized = false;

    Phi3MatmulJob   job;          // populated when job_kind == MATMUL
    Phi3AttnJob     attn_job;     // populated when job_kind == ATTN
    Phi3PoolJobKind job_kind = Phi3PoolJobKind::MATMUL;

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

// Dispatch one per-head attention job and wait for completion. Workers split
// the head range [0, n_head) evenly. When pool is null/uninitialized/single-
// threaded, runs the whole head range on the calling thread (the caller's
// per-thread scratch slice at wid=0 is used in that case). The job struct
// is copied into the pool; caller may reuse its storage immediately after
// this returns.
void phi3_attn_pool_run(Phi3MatmulPool * pool, const Phi3AttnJob & job);
