#pragma once

// G3.3 -- Gemma-4 single-layer F32 forward + ggml-graph oracle.
//
// Both paths consume the SAME dequantized F32 layer weights so the only
// numeric drift is operator order, not quantization. That makes a
// tight tolerance realistic (we use 1e-3 with stage-level fallbacks).
//
// Storage convention (matches ggml): all 2D buffers are stored as
// [innermost, n_tokens] with innermost contiguous. matmul semantics
// follow ggml_mul_mat: out[N, M] = W[K, M]^T @ x[K, N]. In code we
// pass W with shape [K, M] (per the GGUF schema) and x with shape
// [K, n_tokens]; the result is [M, n_tokens].
//
// G3.3 deliberately does NOT include a KV cache, the final lm_head,
// the softcap, or the per-layer-embedding global projection step --
// those land in G3.4+. The single-layer oracle covers the per-layer
// math only. For multi-token attention the test uses n_tokens >= 8
// with causal masking so attention has non-trivial structure.

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;
struct ggml_tensor;

namespace gemma4 {

struct Weights;       // forward decl (defined in gemma4_weights.h)
struct LayerWeights;

// Fully dequantized F32 weights for one layer. Caller owns; allocate
// fresh per-test and free when done.
struct LayerF32 {
    int   il        = -1;
    int   n_embd    = 0;
    int   n_head    = 0;
    int   n_head_kv = 0;
    int   head_dim  = 0;
    int   n_ff      = 0;
    int   n_embd_per_layer = 0;
    bool  is_swa    = false;
    float rms_eps   = 1e-6f;
    float rope_base = 0.0f;
    int   rope_dim  = 0;            // n_rot for ggml_rope_ext

    // freq_factors pointer (non-owning): non-null only for non-SWA layers
    // that carry rope_freqs. May still be null if model has no rope_freqs.
    const float * freq_factors = nullptr;  // [head_dim/2] F32

    // Norm weights (F32, copied from model).
    std::vector<float> attn_norm;        // [n_embd]
    std::vector<float> attn_q_norm;      // [head_dim]
    std::vector<float> attn_k_norm;      // [head_dim]
    std::vector<float> post_attn_norm;   // [n_embd]
    std::vector<float> ffn_norm;         // [n_embd]
    std::vector<float> post_ffw_norm;    // [n_embd]
    std::vector<float> post_norm;        // [n_embd]  (per-layer post-PLE)

    // Matmul weights (dequantized to F32).
    std::vector<float> wq;               // [n_embd, n_head * head_dim]
    std::vector<float> wk;               // [n_embd, n_head_kv * head_dim]
    std::vector<float> wv;               // [n_embd, n_head_kv * head_dim] (may be empty)
    std::vector<float> wo;               // [n_head * head_dim, n_embd]
    std::vector<float> ffn_gate;         // [n_embd, n_ff]
    std::vector<float> ffn_up;           // [n_embd, n_ff]
    std::vector<float> ffn_down;         // [n_ff,   n_embd]
    std::vector<float> inp_gate;         // [n_embd, n_embd_per_layer]
    std::vector<float> proj;             // [n_embd_per_layer, n_embd]

    // Optional scalar [1] -- multiplies hidden before next layer.
    bool  has_layer_output_scale = false;
    float layer_output_scale     = 1.0f;

    // Shared-KV pattern: if >=0, this layer reuses K/V computed by an earlier
    // layer (does NOT call wk/wv). Forward-pass storage of K/V at the
    // owning layer is the caller's responsibility (see network_forward_f32).
    int kv_reuse_il = -1;
};

// Dequantize layer `il` of the model into LayerF32. Reads w_global for
// hparams + rope settings. Returns false + error on any tensor type the
// installed ggml-cpu traits cannot dequant (should not happen for the
// K-quant family used by Q4_K_M).
bool dequant_layer(const llama_model * model, const Weights & w_global,
                   int il, LayerF32 & out, std::string & error);

// Hand-coded single-layer F32 forward (uses gemma4 kernels under the hood).
// Inputs/outputs are all F32, contiguous in ggml [innermost, n_tokens] order.
//   hidden_in        : [n_embd, n_tokens]
//   pos              : [n_tokens]  (one int per token, used for RoPE)
//   per_layer_input  : [n_embd_per_layer, n_tokens]  (already sliced for this layer)
//   hidden_out       : [n_embd, n_tokens] (written)
// G3.3 does NOT take past K/V -- attention is computed over the n_tokens
// in-batch with causal masking (no past tokens).
//
// G3.4a: optional KV-sharing args for Gemma-4's n_layer_kv_from_start pattern.
//   kv_K_self_out / kv_V_self_out  : if non-null, the function writes its
//                                    computed K (post-norm + post-RoPE) and
//                                    V (post-norm; no RoPE) into these
//                                    buffers, sized [n_head_kv * head_dim * n_tokens].
//   kv_K_reuse / kv_V_reuse        : if non-null, the function SKIPS its
//                                    own K/V projection and reuses these
//                                    instead. Pass these for layers with
//                                    L.kv_reuse_il >= 0 (taken from the
//                                    owning layer's kv_K_self_out / kv_V_self_out).
// All four nullptr -> classic behaviour (compute and discard locally),
// used by the G3.3 single-layer self-test.
bool layer_forward_f32(const LayerF32 & L,
                       int n_tokens,
                       const float * hidden_in,
                       const int32_t * pos,
                       const float * per_layer_input,
                       float * hidden_out,
                       std::string & error,
                       float * kv_K_self_out = nullptr,
                       float * kv_V_self_out = nullptr,
                       const float * kv_K_reuse = nullptr,
                       const float * kv_V_reuse = nullptr);

// ggml-graph oracle: same inputs and same F32 weights, but lowers each
// op to a ggml_* call and runs ggml_graph_compute_with_ctx. Used as the
// ground truth for layer_self_test().
bool oracle_layer_forward_f32(const LayerF32 & L,
                              int n_tokens,
                              const float * hidden_in,
                              const int32_t * pos,
                              const float * per_layer_input,
                              float * hidden_out,
                              std::string & error);

// Self-test: dequant layer `il`, generate random hidden_in + per_layer_input
// + monotonic pos, run both paths, element-wise compare with tolerance.
// n_tokens >= 4 recommended so attention is non-trivial.
// On failure, prints first-mismatch diagnostic to stderr.
bool layer_self_test(const llama_model * model, const Weights & w,
                     int il, int n_tokens, std::string & error);

// ---------------------------------------------------------------------
// G3.4 -- whole-network F32 forward.
// ---------------------------------------------------------------------
//
// ModelF32 holds dequantized F32 weights for every layer plus a few
// small F32 globals. Two large tensors (tok_embd and per_layer_tok_embd)
// are NOT dequantized eagerly to save ~3-4 GB on E4B; we keep pointers
// to the original ggml tensors and dequantize on-demand:
//   * tok_embd:           used per-token-row for input embedding lookup
//                         and per-vocab-row for the tied lm_head matmul
//   * per_layer_tok_embd: F32 in gemma4 already, so we just point at it
//
// As a result, ModelF32 requires the source llama_model to stay alive
// for the entire lifetime of ModelF32 -- we hold raw ggml_tensor * back
// at it. dequant_model() copies what it needs out of `Weights` so the
// `Weights` struct itself can be dropped if desired.

struct ModelF32 {
    // Hparams (copied from Weights so ModelF32 is self-describing).
    int   n_layer            = 0;
    int   n_embd             = 0;
    int   n_head             = 0;
    int   n_head_kv          = 0;
    int   n_vocab            = 0;
    int   n_embd_per_layer   = 0;
    int   n_swa              = 0;
    float rms_eps            = 1e-6f;
    float final_logit_softcap = 0.0f;
    bool  output_tied_to_embd = true;

    // Small global F32 buffers.
    std::vector<float> output_norm;           // [n_embd]
    std::vector<float> per_layer_model_proj;  // [n_embd, n_embd_per_layer * n_layer]  (dequantized)
    std::vector<float> per_layer_proj_norm;   // [n_embd_per_layer]
    std::vector<float> freq_factors_data;     // copy of rope_freqs->data (or empty)

    // Layer weights. LayerF32.freq_factors points INTO this->freq_factors_data
    // for non-SWA layers (or nullptr for SWA layers). Do not move ModelF32
    // after dequant_model() returns.
    std::vector<LayerF32> layers;

    // Pointers to quant tensors in the original model (NOT owned, do not free).
    // tok_embd is the embedding matrix; also used as the tied lm_head.
    const ggml_tensor * tok_embd_quant            = nullptr; // [n_embd, n_vocab]
    const ggml_tensor * per_layer_tok_embd_quant  = nullptr; // F32 [n_embd_per_layer * n_layer, n_vocab]
};

// Dequant the whole model into ModelF32. Does not copy tok_embd or
// per_layer_tok_embd (held as pointers to the live llama_model).
// Memory footprint: roughly 5.6 GB on E2B Q4_K_M, ~8 GB on E4B Q4_K_M.
bool dequant_model(const llama_model * model, const Weights & w,
                   ModelF32 & out, std::string & error);

// Full-network F32 forward over a single batch of n_tokens.
//   token_ids[n_tokens] - token IDs (used for tok_embd + PLE lookups)
//   pos[n_tokens]       - positions (used for RoPE)
//   logits_out          - if last_token_only == true:  [n_vocab]
//                         else:                         [n_vocab * n_tokens]
//
// Asserts n_tokens <= model.n_swa (SWA masking is NOT implemented for
// G3.4a; only causal masking inside each layer). Use small prompts for
// the first round.
//
// Note: this function does no allocation amortization between calls --
// it allocates and frees scratch per call. For a one-shot test that's
// fine; G3.4b will manage scratch + KV cache.
bool network_forward_f32(const ModelF32 & model,
                         int n_tokens,
                         const int32_t * token_ids,
                         const int32_t * pos,
                         bool last_token_only,
                         float * logits_out,
                         std::string & error);

// Network self-test: tokenize a fixed prompt with the model's tokenizer,
// run hand-coded network_forward_f32, run upstream llama_decode on the
// same prompt, compare the last-token logits with multi-metric reporting:
//   * top-1 token match
//   * top-5 / top-10 overlap (set agreement)
//   * upstream-top-1 rank in hand's ranking
//   * max_abs / mean_abs / RMS error on FULL logits
//   * cosine similarity on FULL logits
//
// Returns true if top-1 token matches; prints all metrics regardless.
// (Top-1 match is the user-visible correctness check; quantitative
// metrics are diagnostic.)
//
// `prompt` is a raw text prompt (no chat template applied). Choose a
// prompt where the model's continuation is high-confidence to maximize
// signal (e.g. "The capital of France is", "1, 2, 3, 4,").
bool network_self_test(const llama_model * model, const Weights & w,
                       const std::string & prompt, int n_threads,
                       std::string & error);

} // namespace gemma4
