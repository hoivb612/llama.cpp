#pragma once

// Phi3 custom-forward scaffolding (Phase A).
//
// This header declares the Phi3Weights resolver — the foundation for the
// custom Phi-3 forward pass that bypasses `llama_decode` on gen steps.
//
// A0 (this commit) only resolves and validates the model tensors; no forward
// pass is built yet. The actual `phi3_custom_decode` / `phi3_custom_prefill`
// functions land in A2.
//
// References (for future readers):
//  - src/models/phi3.cpp: standard Phi3 graph implementation (the oracle).
//  - include/llama_b612.h: llama_model_get_tensor_by_name (used for lookup).

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

// Per-layer tensor pointers. Borrowed from the model — do NOT free.
struct Phi3LayerWeights {
    const ggml_tensor * attn_norm = nullptr; // [n_embd]              F32
    const ggml_tensor * wqkv      = nullptr; // [n_embd, n_qkv]       K-quant (Q4_K_M)
    const ggml_tensor * wo        = nullptr; // [n_embd, n_embd]      K-quant
    const ggml_tensor * ffn_norm  = nullptr; // [n_embd]              F32
    const ggml_tensor * ffn_up    = nullptr; // [n_embd, 2 * n_ff]    K-quant   gate|up fused
    const ggml_tensor * ffn_down  = nullptr; // [n_ff, n_embd]        K-quant
};

// Global tensors + per-layer slots. Borrowed from the model.
struct Phi3Weights {
    // Hyper-parameters resolved from the model.
    int n_layer       = 0;
    int n_embd        = 0;
    int n_head        = 0;
    int n_head_kv     = 0;
    int n_ff          = 0;       // half-width: ffn_up is [n_embd, 2*n_ff].
    int n_embd_head   = 0;       // = n_embd / n_head.
    int n_embd_gqa    = 0;       // = n_embd_head * n_head_kv.
    int n_qkv         = 0;       // = n_embd + 2 * n_embd_gqa.
    int n_vocab       = 0;

    // Global tensors.
    const ggml_tensor * tok_embd    = nullptr; // [n_embd, n_vocab]   K-quant
    const ggml_tensor * output_norm = nullptr; // [n_embd]            F32
    const ggml_tensor * output      = nullptr; // [n_embd, n_vocab]   K-quant (may alias tok_embd if tied)

    // Optional RoPE factor tables (shared across all layers — TENSOR_DUPLICATED).
    const ggml_tensor * rope_long   = nullptr; // [n_rot/2]           F32 (may be null)
    const ggml_tensor * rope_short  = nullptr; // [n_rot/2]           F32 (may be null)

    // Did the loader tie `output` to `tok_embd`?
    bool output_tied_to_embd = false;

    // Per-layer tensors.
    static constexpr int MAX_LAYERS = 64;
    Phi3LayerWeights layers[MAX_LAYERS];
};

// Resolve every Phi3 tensor by name and validate shapes/types.
// On failure, returns false and writes a human-readable reason to `error`.
//
// What is validated:
//  - All required tensors are present (or tied, for `output`).
//  - K-quant matmul weights have a vec_dot_type of GGML_TYPE_Q8_K
//    (matches the existing phi3_weight_accepts_q8K() semantics — accepts
//    Q4_K/Q5_K/Q6_K mixes used in K_M quants).
//  - Norm tensors are F32 and have shape [n_embd].
//  - Per-layer matmul shapes match (n_embd, n_qkv), (n_embd, n_embd),
//    (n_embd, 2*n_ff), (n_ff, n_embd).
//  - Hyper-parameters are self-consistent (n_embd_head * n_head == n_embd).
//
// Pre-condition: model must already be loaded.
bool phi3_weights_resolve(const llama_model * model, Phi3Weights & out, std::string & error);

// Print the resolved tensor table to stderr (shape, type, address). Useful for
// confirming A0 worked correctly before any forward code lands.
void phi3_weights_dump(const Phi3Weights & w);


// =============================================================================
// Phi3KV — custom contiguous KV cache for Phase A custom forward.
// =============================================================================
// Owns its own F16 K/V storage, independent of llama's internal kv cache.
//
// Layout (matches what manual scaled-dot-product attention prefers):
//   K[il]  : [head_dim, n_head_kv, ctx_max]    F16
//            (innermost dim = head_dim, contiguous; matches standard layout)
//   V_T[il]: [ctx_max,  n_head_kv, head_dim]   F16
//            (TRANSPOSED so attn[1..n_kv] . V_T[:, h, d] is a contiguous
//             reduction over the n_kv dimension)
//
// Single-sequence, contiguous positions. Multi-turn rewind is supported via
// kv_truncate (O(1)), kv_drop_range (memmove), and kv_keep_prefix.
//
// Indexing helpers (do NOT bounds-check; caller is responsible):
//   K[il] at position pos, head h, dim d:
//       k_ptr[il][pos*n_head_kv*head_dim + h*head_dim + d]
//   V_T[il] at position pos, head h, dim d:
//       v_ptr[il][d*n_head_kv*ctx_max + h*ctx_max + pos]
//
// Memory budget at ctx_max=4096 for Phi-3-mini-4k (n_layer=32, head_dim=96,
// n_head_kv=32) is ~1.5 GiB total (K and V combined). Logged at init.

#include "ggml.h"   // for ggml_fp16_t

struct Phi3KV {
    int n_layer       = 0;
    int n_head_kv     = 0;
    int head_dim      = 0;
    int ctx_max       = 0;
    int current_len   = 0;   // logical number of populated positions [0, current_len)

    // Per-layer base pointers into the K_buf / V_buf flat arenas.
    ggml_fp16_t * K[Phi3Weights::MAX_LAYERS] = {};
    ggml_fp16_t * V[Phi3Weights::MAX_LAYERS] = {};

private:
    // Single backing allocations to keep alloc/free simple.
    // Sized: n_layer * head_dim * n_head_kv * ctx_max F16s each.
    ggml_fp16_t * K_buf = nullptr;
    ggml_fp16_t * V_buf = nullptr;
    friend bool   phi3_kv_init  (Phi3KV & kv, int n_layer, int n_head_kv, int head_dim, int ctx_max, std::string & error);
    friend void   phi3_kv_free  (Phi3KV & kv);
    friend void   phi3_kv_drop_range(Phi3KV & kv, int p0, int p1);
};

// Allocate K_buf / V_buf and populate per-layer base pointers.
// On failure (e.g. OOM), returns false and writes a reason to `error`.
bool   phi3_kv_init      (Phi3KV & kv, int n_layer, int n_head_kv, int head_dim, int ctx_max, std::string & error);

// Release all KV memory. Safe to call on an uninitialized struct.
void   phi3_kv_free      (Phi3KV & kv);

// Total bytes allocated for K + V across all layers.
size_t phi3_kv_bytes     (const Phi3KV & kv);

// O(1) — drop tokens at positions [new_len, current_len). Used between turns.
// new_len must be in [0, current_len].
void   phi3_kv_truncate  (Phi3KV & kv, int new_len);

// Drop the range [p0, p1) and shift positions [p1, current_len) left by (p1-p0).
// Cost: O((current_len - p1) * n_layer * 2 * row_bytes). Used for middle drops
// (rare). Caller must ensure no RoPE-position-shifting concerns (tokens after
// p1 keep their original RoPE phase, which is correct for our usage).
void   phi3_kv_drop_range(Phi3KV & kv, int p0, int p1);

// Convenience: keep_prefix(N) == truncate(N).
inline void phi3_kv_keep_prefix(Phi3KV & kv, int n_keep) { phi3_kv_truncate(kv, n_keep); }

// One-shot self-test: allocate a tiny KV using the model's dimensions, write
// known sentinels, exercise truncate / drop_range / keep_prefix, verify, free.
// Prints PASS/FAIL to stderr. Returns true on PASS.
bool   phi3_kv_self_test (const Phi3Weights & w, std::string & error);


// ---------------------------------------------------------------------------
// Phi3MatmulPool self-test — Phase A.
// Builds a small F32 weight matrix + F32 src vector, dispatches many matmul
// jobs through the pool, compares against a serial reference, and verifies
// byte-equality. Stress-tests phase-1/2/3 backoff transitions with
// deliberate inter-job sleeps. Does NOT need a model.
// Prints PASS/FAIL to stderr. Returns true on PASS.
// ---------------------------------------------------------------------------
bool   phi3_matmul_pool_self_test(int n_threads, std::string & error);

// ---------------------------------------------------------------------------
// Per-kernel self-test — Phase A.
// Validates the three small F32 kernels needed by the upcoming custom forward
// pass against the corresponding ggml ops (RMSNorm, token-embedding dequant,
// NeoX RoPE), executed on the standard ggml-cpu backend.
//
// Coverage:
//   - RMSNorm: n_embd=3072, eps=1e-5, fused with F32 weight (matches
//     llm_build_norm(LLM_NORM_RMS)).
//   - Token embedding dequant: ggml_get_rows on a synthetic F16 token-embd
//     tensor for several tokens.
//   - NeoX RoPE: head_dim=96, two heads, three positions; tested with both
//     freq_factors == NULL (Phi-3-mini-4k case) and with synthetic factors
//     (Phi-3-medium long/short rope case).
// Does NOT need a model. Prints PASS/FAIL to stderr. Returns true on PASS.
// ---------------------------------------------------------------------------
bool   phi3_kernel_self_test(std::string & error);


// ===========================================================================
// Phi3FusedCtx — Phase A custom-forward execution context (A2.2 scaffolding).
// ===========================================================================
// This struct ties together the borrowed weights/pool with the owned KV cache,
// per-step scratch buffers, and resolved RoPE config. The actual decode and
// prefill functions land in A2.4; for A2.2 we only define the structs, the
// 24-feature validator, and the allocator/deallocator.

struct Phi3MatmulPool;       // fwd-decl from phi3_fused_ops.h

// RoPE configuration, resolved once at init from cparams/hparams.
struct Phi3RoPEConfig {
    int32_t       n_rot           = 0;     // dimensions to rotate (Phi-3: == head_dim)
    int32_t       n_ctx_orig      = 0;     // hparams.n_ctx_orig_yarn
    float         freq_base       = 0.0f;
    float         freq_scale      = 0.0f;
    float         ext_factor      = 0.0f;
    float         attn_factor     = 0.0f;
    float         beta_fast       = 0.0f;
    float         beta_slow       = 0.0f;
    const float * rope_factors    = nullptr;  // null → unit factors (no YARN)
};

// Per-step scratch buffers. Allocated once at init, reused every step.
struct Phi3ForwardScratch {
    // Quantization types resolved from weight traits at ctx_init.
    ggml_type            q_type_attn = GGML_TYPE_COUNT;
    ggml_type            q_type_ffn  = GGML_TYPE_COUNT;

    std::vector<float>   x_buf;       // [n_embd]
    std::vector<float>   h_buf;       // [n_embd]
    std::vector<uint8_t> hq_buf;      // [row_size(q_type_attn, n_embd)]
    std::vector<uint8_t> ffq_buf;     // [row_size(q_type_ffn, n_ff)]
    std::vector<float>   qkv_buf;     // [n_qkv]
    std::vector<float>   ctx_buf;     // [n_embd]
    std::vector<float>   scores_buf;  // [ctx_max]
    std::vector<float>   upgate_buf;  // [2*n_ff]
    std::vector<float>   ff_buf;      // [n_ff]

    // F32-debug mode: per-layer TRANSIENT F32 mirrors. Allocated only when
    // f32_debug is set at ctx_init time. Each holds ONE layer's dequantized
    // weights and is overwritten before processing the next layer.
    // Peak ~150 MiB at mini-4k Q4_K_M (one layer worth). lm_head is streamed
    // in vocab-row chunks (f32_vocab_chunk * n_embd) instead of a full mirror.
    bool                 f32_debug          = false;
    std::vector<float>   w_f32_attn_norm;     // [n_embd]
    std::vector<float>   w_f32_wqkv;          // [n_embd * n_qkv]
    std::vector<float>   w_f32_wo;            // [n_embd * n_embd]
    std::vector<float>   w_f32_ffn_norm;      // [n_embd]
    std::vector<float>   w_f32_ffn_up;        // [n_embd * 2 * n_ff]
    std::vector<float>   w_f32_ffn_down;      // [n_ff * n_embd]
    std::vector<float>   w_f32_vocab_chunk;   // [f32_vocab_chunk * n_embd]
    int32_t              f32_vocab_chunk    = 256;
    int32_t              f32_cached_layer   = -1;

    // A3.y — per-thread attention scratch for the parallel per-head dispatch
    // in phi3_layer_forward_qquant section 5. Allocated when the matmul pool
    // is present (n_threads > 1). Each worker w reads/writes its own slice:
    //   q_h_f16    = attn_q_h_f16_pool.data()    + w * attn_stride_q_f16
    //   scores     = attn_scores_pool.data()     + w * attn_stride_s_f32
    //   scores_f16 = attn_scores_f16_pool.data() + w * attn_stride_s_f16
    // Strides are rounded up so each slice starts on a 64-byte boundary;
    // this avoids false sharing between adjacent workers' scratch.
    // Off-path when pool is null/single-threaded or --phi3-fused-qquant-
    // attn-parallel is 0 — the existing serial loop allocates per-call.
    int                  attn_pool_workers  = 0;
    int                  attn_stride_q_f16  = 0;  // padded head_dim (F16 elts)
    int                  attn_stride_s_f32  = 0;  // padded ctx_max (F32 elts)
    int                  attn_stride_s_f16  = 0;  // padded ctx_max (F16 elts)
    std::vector<ggml_fp16_t> attn_q_h_f16_pool;
    std::vector<float>       attn_scores_pool;
    std::vector<ggml_fp16_t> attn_scores_f16_pool;
};

// A3.x — per-op profile accumulators (nanoseconds). All buckets are summed
// across n_gen tokens; per-layer buckets are summed across all n_layer
// invocations within each token (so the printed avg µs is "total time per
// token spent in this op type, summed across all 32 layers"). enabled=false
// keeps every PhiTimer call a 1-instruction branch — zero measurable cost.
// Gated by --phi3-fused-qquant-profile.
struct Phi3DecodeProfile {
    bool      enabled       = false;
    uint64_t  n_tokens      = 0;
    // Per-token (×1):
    uint64_t  embed_ns      = 0;
    uint64_t  final_norm_ns = 0;
    uint64_t  lmhead_ns     = 0;
    // Per-layer (accumulated across n_layer × n_tokens):
    uint64_t  attn_norm_ns  = 0;  // RMSNorm + mul (+ quantize when fused)
    uint64_t  wqkv_ns       = 0;
    uint64_t  rope_ns       = 0;
    uint64_t  kv_write_ns   = 0;
    uint64_t  attn_ns       = 0;  // QK + softmax + V combine, per-head loop
    uint64_t  wo_ns         = 0;
    uint64_t  res1_ns       = 0;
    uint64_t  ffn_norm_ns   = 0;
    uint64_t  ffn_up_ns     = 0;
    uint64_t  swiglu_ns     = 0;
    uint64_t  ffn_down_ns   = 0;
    uint64_t  res2_ns       = 0;
};

// Per-context state. Borrows weights & pool; owns KV, scratch, RoPE config.
struct Phi3FusedCtx {
    const Phi3Weights *  w           = nullptr;  // borrowed
    Phi3MatmulPool *     matmul_pool = nullptr;  // borrowed (may be null → serial)
    float                eps         = 1e-5f;

    Phi3RoPEConfig       rope;
    Phi3KV               kv;
    Phi3ForwardScratch   scratch;
    int32_t              cur_pos     = 0;

    // A3.2 (experiment) — fuse RMSNorm + multiply + quantize-to-Q8K at
    // the two attn_norm -> wqkv and ffn_norm -> ffn_up junctions. The
    // fused path is bit-identical to the unfused path; only gen_tps
    // differs. Measured impact on Phi-3-mini Q2_K/Q3_K_M/Q4_K_M at 8
    // threads UMA was neutral-to-negative (Q3_K_M: -13.6% gen_tps,
    // Q4_K_M: flat) -- the per-block call overhead of 12x
    // from_float(QK_K) outweighs the L1 locality win because
    // quantize_row_q8_K on x86 is the scalar ref and the activation
    // already fits in L1. Kept off by default and exposed via
    // --phi3-fused-qquant-rmsnorm-fuse 0|1 for educational reference.
    // See phi3_fused_rmsnorm_quant_q8K.
    bool                 fuse_rmsnorm_quant = false;

    // A3.y (experiment) — split the per-head attention loop in
    // phi3_layer_forward_qquant section 5 across the matmul pool's
    // workers. Each thread handles a contiguous range of heads and
    // writes to disjoint slices of `attn_ctx`, so the dispatch is
    // bit-identical to the serial loop. Only takes effect when the
    // pool is initialized with n_threads > 1. Default OFF for the
    // first commit; gated by --phi3-fused-qquant-attn-parallel 0|1.
    bool                 attn_parallel = false;

    // A3.x per-op profile. enabled=false (default) makes every timer a
    // no-op branch. Reset & emit are driven by phi3_run_qquant_decode.
    Phi3DecodeProfile    prof;
};

// 24-feature validator (see __phase_A2_spec.md §1.5). Returns true if the
// model + context can be safely served by the Phase A custom forward. On
// false, `error` names the first failing check. Cheap; no allocation.
//
// `lctx` is required (cparams checks: flash_attn, embeddings, causal_attn,
// n_seq_max, plus LoRA/cvec). Pass nullptr only if the caller intends to
// validate model-side checks only — in that case only the per-layer +
// model-global rejection rules are checked.
bool phi3_fused_validate_supported(
        const llama_model    * model,
        const Phi3Weights    & w,
        const llama_context  * lctx,
        std::string          & error);

// Allocate KV + scratch, resolve RoPE config, validate features. Returns
// false with a specific error message on any failure.
bool phi3_fused_ctx_init(
        Phi3FusedCtx         & out,
        const Phi3Weights    & w,
        Phi3MatmulPool       * matmul_pool,
        const llama_model    * model,
        const llama_context  * lctx,
        int                    ctx_max,
        bool                   f32_debug,
        std::string          & error);

void phi3_fused_ctx_free(Phi3FusedCtx & cx);

// Optional: log the resolved RoPE + scratch sizes to stderr (helpful when
// triaging "did we resolve this correctly?" questions). Safe to call on an
// uninitialized cx; just prints zeros.
void phi3_fused_ctx_dump(const Phi3FusedCtx & cx);


// ===========================================================================
// A2.3 — Single-layer F32 self-test.
// ===========================================================================
// Populate the per-layer F32 weight mirrors in cx.scratch from the model's
// quantized weights. Requires cx.scratch.f32_debug == true (i.e., ctx_init
// was called with f32_debug=true). Returns false with a specific error if
// the dequantize fails for any tensor.
bool phi3_layer_warmup_f32_mirrors(Phi3FusedCtx & cx, int il, std::string & error);

// Optional capture of per-step intermediates for diagnostics. All buffers
// the caller provides MUST be sized to the values shown; null pointers are
// skipped. Capture is the same for our impl and the oracle so a diverging
// step is immediately visible.
struct Phi3LayerCapture {
    float * norm1            = nullptr; // [n_embd]
    float * qkv              = nullptr; // [n_qkv]
    float * Q_post_rope      = nullptr; // [n_embd]
    float * K_post_rope      = nullptr; // [n_embd_kv]   (just the current token's K)
    float * V_cur            = nullptr; // [n_embd_kv]   (current token's V, pre-cache)
    float * attn_ctx         = nullptr; // [n_embd]      (post softmax · V)
    float * attn_out         = nullptr; // [n_embd]      (after wo)
    float * x_after_res1     = nullptr; // [n_embd]
    float * norm2            = nullptr; // [n_embd]
    float * upgate           = nullptr; // [2 * n_ff]
    float * ff               = nullptr; // [n_ff]
    float * ffn_out          = nullptr; // [n_embd]
};

// Our F32-everywhere single-layer forward for one token at absolute `pos`.
// Caller is responsible for the cache state at positions [0..pos); this call
// writes our newly-computed K, V at position `pos` (F16) and updates
// cx.kv.current_len if necessary.
//
// On entry x_inout = input residual stream (token embedding for layer 0).
// On exit  x_inout = output residual stream after this layer.
bool phi3_layer_forward_f32(
        Phi3FusedCtx       & cx,
        int                  il,
        int                  pos,
        float              * x_inout,
        Phi3LayerCapture   * capture,   // optional, may be nullptr
        std::string        & error);

// Run the single-layer F32 self-test against a ggml-cpu oracle graph.
// Loads layer 0 of the live model, pre-fills 4 prior positions of random
// K/V, then runs one token at pos=4 through both pipelines and compares
// every intermediate. Returns false with the first diverging step + index.
bool phi3_layer_self_test(const llama_model * model, std::string & error);


// ===========================================================================
// A2.4a — Full-network F32 forward + cross-layer self-test.
// ===========================================================================
// Run the full F32 hand path for ONE token:
//   tok_embd[token_id]  ->  layer 0..n_layer-1  ->  output_norm  ->  lm_head
// The KV cache cx.kv is updated at position `pos` for every layer. Caller
// owns positions [0..pos) (must already be set or 0). For F32-everywhere
// debug/decode mode this is called once per generated token; per-layer F32
// weight mirrors are lazily warmed by phi3_layer_forward_f32 (slow: ~450
// MiB dequant per layer per call — fine for tests, not production).
//
// On entry: out_logits is sized n_vocab.
// If tok_embd_capture != nullptr, it receives the F32-dequantized token
// embedding row (used for "tok_embd OK" stage in self-test).
// If capture_x_per_layer != nullptr, capture_x_per_layer[il] (for il in
// [0..n_layer)) receives the residual stream AFTER layer il. Slot index
// n_layer (if present) receives the post-final-norm residual.
// Pass capture_x_per_layer = nullptr to skip per-layer captures (fast path).
bool phi3_full_forward_f32(
        Phi3FusedCtx       & cx,
        int                  token_id,
        int                  pos,
        float              * out_logits,           // [n_vocab]
        float              * tok_embd_capture,     // optional, [n_embd]
        float * const      * capture_x_per_layer,  // optional, length (n_layer+1)
        std::string        & error);

// Full-network self-test: runs hand path vs ggml-cpu oracle across all
// layers and the lm_head, comparing at every layer boundary, the
// post-final-norm residual, and the full logits vector.
//
// Two sub-tests:
//   1. pos=0  with no prior KV (smoke; attention degenerates to V_cur).
//   2. pos=4  with random F16-rounded prior K/V seeded into all 32 layers
//             (exercises actual softmax / causal-attention / K dot
//             products / RoPE at non-zero pos).
//
// Numeric primary acceptance per stage:
//   max_abs <= tol_max  AND  rmse <= tol_rmse  (both depth-scaled)
// Argmax + top-5 overlap reported as diagnostics (do not gate pass/fail).
//
// Returns false with the first diverging stage on failure.
bool phi3_full_self_test(const llama_model * model, std::string & error);


// ===========================================================================
// A2.4b — F32-everywhere decode (debug / spot-check).
// ===========================================================================
// Run the F32 hand path end-to-end:
//   - For each prompt token, call phi3_full_forward_f32 at increasing pos.
//     This populates cx.kv at positions [0..n_prompt-1] for all 32 layers
//     and produces logits for the last prompt token.
//   - Greedy-argmax the last prefill logits => first generated token.
//   - For each remaining gen step: forward the previously-generated token
//     at the next position, argmax to produce the next token.
//
// Slow (~6-7s/token on Phi-3-mini-4k Q4_K_M because every layer re-
// dequantizes ~450 MiB of F32 weight mirrors per token). Intended for
// one-shot side-by-side comparison with the baseline (Q4_K_M via
// llama_decode), NOT for production decode or CI.
//
// Pre-conditions enforced inside:
//   - prompt_tokens non-empty
//   - n_gen > 0
//   - model resolves through phi3_weights_resolve
//   - n_head == n_head_kv (no GQA support yet; fails loudly otherwise)
//
// Output: out_generated is appended with exactly n_gen token ids.
bool phi3_run_f32_decode(
        const llama_model            * model,
        const std::vector<llama_token> & prompt_tokens,
        int                            n_gen,
        std::vector<llama_token>     & out_generated,
        std::string                  & error);


// ===========================================================================
// A2.5a.0 - Q-quant matmul primitive self-test.
// ===========================================================================
// For layer 0's wqkv/wo/ffn_up/ffn_down, compares the hand q_matmul_row
// (private impl) against ggml-cpu's ggml_mul_mat over the SAME quantized
// weight + same F32 input vector. Expected: very close to byte-identical
// (same vec_dot kernel + same from_float quantization). Tolerances:
// max_abs <= 1e-3, rmse <= 1e-4 (generous; printed for diagnostics).
//
// Use --phi3-qmatmul-test in Phi3.cpp.
bool phi3_qmatmul_self_test(const llama_model * model, std::string & error);


// ===========================================================================
// A2.5a.1 - Q-quant end-to-end decode (sibling of phi3_run_f32_decode).
// ===========================================================================
// Same shape as phi3_run_f32_decode, but:
//   - Uses the ORIGINAL quantized weights via q_matmul_row (no F32 mirrors,
//     no ~150MB transient, no per-token ~450ms dequant cost).
//   - Allocates Phi3FusedCtx with f32_debug=false.
//   - lm_head argmax uses the existing phi3_fused_lmhead_argmax (q-quant,
//     same path as production runtime).
//
// Attention path remains F32-dot-vs-F16-KV (matches A2.4b). This is a known
// minor divergence from ggml's ggml_vec_dot_f16 attention; A2.4 showed
// argmax still matches baseline.
//
// Pre-conditions enforced inside:
//   - prompt_tokens non-empty
//   - n_gen > 0
//   - n_head == n_head_kv (no GQA support yet)
//   - all prompt tokens in [0, n_vocab)
//
// Output: out_generated is appended with exactly n_gen token ids.
// Frees ctx + lm_head on every failure path.
bool phi3_run_qquant_decode(
        const llama_model            * model,
        const std::vector<llama_token> & prompt_tokens,
        int                            n_gen,
        int                            n_threads,
        bool                           fuse_rmsnorm_quant,
        bool                           profile_per_op,
        bool                           attn_parallel,
        std::vector<llama_token>     & out_generated,
        std::string                  & error);

// ---------------------------------------------------------------------------
// A5.2 — KV bridge: convert llama-format KV cache state into our Phi3KV
// layout so qquant-decode can pick up where llama_decode left off.
//
// LAYOUT NOTES (single-stream, F16):
//   llama K (per layer): linear[d + head*head_dim + pos*n_embd_kv]
//                        (= our K layout — straight memcpy per layer).
//   llama V (per layer, v_trans=true, the default for non-FA contexts):
//                        linear[pos + (head*head_dim + d)*src_kv_size]
//                        i.e. src_kv_size-long contiguous strip per (h,d),
//                        with strip order (h outer, d inner).
//   our V[il]:           linear[pos + (d*n_head_kv + h)*ctx_max]
//                        ctx_max-long strip per (h,d), strip order
//                        (d outer, h inner). Same strip CONTENT, different
//                        strip ORDER — the bridge re-arranges strips.
//
// The bridge is independent of llama — operates on raw F16 buffers + layout
// descriptors, so the same function will later serve a disk-loaded KV blob
// path (cached prefill / level cache).
//
// Low-level: copy `n_tokens` of K/V state from raw llama-format per-layer
// buffers into `cx.kv`, writing to destination positions
// [base_pos, base_pos + n_tokens) sourced from positions
// [src_pos_first, src_pos_first + n_tokens) of the source.
//
// Requirements (asserted; clear error on mismatch):
//   - src_type == GGML_TYPE_F16 (the only path exercised today).
//   - src_v_trans == true (we don't implement on-the-fly transpose yet).
//   - cx.kv.n_head_kv * cx.kv.head_dim is the per-token K row size (no GQA).
//   - base_pos >= 0 and base_pos + n_tokens <= cx.kv.ctx_max.
//   - src_pos_first + n_tokens <= src_kv_size.
//
// Does NOT update cx.kv.current_len. Caller is responsible for that
// (consistent with the per-token forward path which sets current_len
// explicitly after writing).
bool phi3_seed_kv_from_raw(
        Phi3FusedCtx       & cx,
        int                  base_pos,
        int                  n_tokens,
        int                  src_pos_first,
        bool                 src_v_trans,
        size_t               src_kv_size,           // source V (h,d) strip length, in elements
        enum ggml_type       src_type,              // must be GGML_TYPE_F16
        const void * const * src_K_layers,          // [cx.kv.n_layer] base ptrs to each layer's K data
        const void * const * src_V_layers,          // [cx.kv.n_layer] base ptrs to each layer's V data
        std::string        & error);

struct llama_context;

// Convenience wrapper: pull raw K/V layer pointers + v_trans flag from a
// llama_context (via llama_kv_self_layer_k / _layer_v / _v_trans) and feed
// them to phi3_seed_kv_from_raw. The context's KV cache must be the single
// unified (non-SWA, non-iSWA, non-hybrid) variety — Phi-3-mini qualifies.
//
// Reports a clear error on mismatch (e.g. flash_attn enabled -> v_trans=false;
// SWA model; non-F16 KV cache; n_head != n_head_kv).
bool phi3_seed_kv_from_llama(
        Phi3FusedCtx         & cx,
        struct llama_context * lctx,
        int                    base_pos,
        int                    n_tokens,
        int                    src_pos_first,
        std::string          & error);

// ---------------------------------------------------------------------------
// A2.5c — 3-prompt × n-token regression infrastructure.
//
// phi3_run_baseline_decode drives the production-equivalent path:
// llama_decode for layers/KV evolution + phi3_fused_lmhead_argmax for token
// selection (exact same selector the runtime uses in greedy mode). Creates a
// fresh llama_context with flash_attn DISABLED to match spec §0; uses
// llama_set_phi3_fused_lmhead(ctx, true) so the lm_head matmul is pruned
// from the graph and the hidden state is exposed via
// llama_get_embeddings_ith(ctx, -1). Never stops early on EOS — generates
// exactly n_gen tokens.
//
// This is the oracle for the A2.5c yes/no acceptance bar (spec §4 lines
// 472-479: "All 256 tokens MUST match").
bool phi3_run_baseline_decode(
        const llama_model              * model,
        const std::vector<llama_token> & prompt_tokens,
        int                              n_gen,
        int                              n_threads_prefill,
        int                              n_threads_gen,
        std::vector<llama_token>       & out_generated,
        std::string                    & error);

// phi3_run_qquant_regress: 3-prompt × n_gen-token regression. Compares
// phi3_run_baseline_decode against phi3_run_qquant_decode element-wise per
// prompt. PASS only if all tokens match for all prompts (yes/no bar). On
// divergence, logs the first diverging step + top-2 logit margin (computed
// from the baseline hidden state at that step) for diagnostics only — the
// bar is still "all match" per spec line 477. Prompts are hardcoded for
// reproducibility (see implementation).
bool phi3_run_qquant_regress(
        const llama_model * model,
        int                 n_gen,
        int                 n_threads_prefill,
        int                 n_threads_gen,
        int                 n_threads_qquant,
        bool                fuse_rmsnorm_quant,
        bool                attn_parallel,
        bool              & out_pass,
        std::string       & error);
