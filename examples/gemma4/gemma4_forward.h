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
bool layer_forward_f32(const LayerF32 & L,
                       int n_tokens,
                       const float * hidden_in,
                       const int32_t * pos,
                       const float * per_layer_input,
                       float * hidden_out,
                       std::string & error);

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

} // namespace gemma4
