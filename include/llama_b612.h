#ifndef LLAMA_B612_H
#define LLAMA_B612_H

#ifndef LLAMA_H
#include "llama.h"
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API enum ggml_type llama_ftype_get_default_type(enum llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API struct quantize_state_impl * llama_quant_init(
        const struct llama_model * model,
        const struct llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(struct quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API struct llama_model * llama_quant_model_from_metadata(const struct llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const struct quantize_state_impl * qs,
        const struct ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        struct quantize_state_impl * qs,
        enum llama_ftype ftype,
        struct ggml_tensor ** tensors,
        enum ggml_type * result_types,
        size_t n_tensors);

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

// Set whether the context outputs pre-norm embeddings or not.
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0.
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits.
LLAMA_API void llama_set_embeddings_pre_norm(struct llama_context * ctx, bool value, bool masked);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_pre_norm    (struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_pre_norm_ith(struct llama_context * ctx, int32_t i);

// Phi3 Phase C: skip the lm_head matmul (and logits buffer write) on each decode.
// When enabled, the standard graph's output tensor (post-final-norm hidden state)
// is exposed via the embeddings buffer (llama_get_embeddings_ith) and the caller
// is responsible for computing the next token directly from it (see phi3_fused_ops.h).
// Greedy-only: caller must not invoke sampler chains that need full logits.
// No-op for non-Phi3 architectures.
LLAMA_API void llama_set_phi3_fused_lmhead(struct llama_context * ctx, bool value);

// Phi3 Option 1: in-graph fusion of RMSNorm + quantize-to-Q8_K at the two
// per-layer matmul sites (attn_norm+wqkv and ffn_norm+ffn_up). Enables the
// downstream `mul_mat` to skip its internal `from_float` step. Combinable
// with --phi3-fused-lmhead. No-op for non-Phi3 architectures.
LLAMA_API void llama_set_phi3_fused_decode(struct llama_context * ctx, bool value);

// Look up a model tensor by GGUF tensor name (e.g. "output.weight").
// Returns nullptr if the tensor is absent (some models tie lm_head to token_embd).
// The returned pointer is owned by the model; do not free.
LLAMA_API const struct ggml_tensor * llama_model_get_tensor_by_name(
        const struct llama_model * model, const char * name);

// Phi3 Phase A custom-forward fast-path guards: report whether the context
// currently has any active LoRA adapters or active control-vector slices.
// Returns true if active (the custom forward must fall back to the standard
// ggml graph for correctness). Cheap to call: no synchronization required.
LLAMA_API bool llama_b612_has_active_lora(const struct llama_context * ctx);
LLAMA_API bool llama_b612_has_active_cvec(const struct llama_context * ctx);

// Snapshot of the hparams / cparams fields needed by the Phi-3 custom-forward
// validator and RoPE config builder. One round trip into the llama internals
// instead of a dozen tiny accessors. All fields are non-mutating reads.
struct llama_b612_phi3_features {
    // --- cparams (per-context) ---
    bool     cp_flash_attn;
    bool     cp_embeddings;
    bool     cp_causal_attn;
    uint32_t cp_n_seq_max;
    uint32_t cp_n_ctx_seq;
    float    cp_rope_freq_base;
    float    cp_rope_freq_scale;
    float    cp_yarn_ext_factor;
    float    cp_yarn_attn_factor;
    float    cp_yarn_beta_fast;
    float    cp_yarn_beta_slow;

    // --- hparams (per-model) ---
    uint32_t hp_n_rot;             // for layer 0 (Phi-3 uses one rotation count)
    float    hp_f_norm_rms_eps;
    float    hp_f_clamp_kqv;
    float    hp_f_max_alibi_bias;
    bool     hp_use_alibi;
    bool     hp_attn_soft_cap;
    int      hp_swa_type;          // 0 == LLAMA_SWA_TYPE_NONE
    uint32_t hp_n_ctx_orig_yarn;
    float    hp_rope_freq_base_train;
};

LLAMA_API void llama_b612_get_phi3_features(
        const struct llama_model   * model,
        const struct llama_context * ctx,
        struct llama_b612_phi3_features * out);

// B612 / Phi3 hybrid prefill: raw access to the unified KV cache's per-layer
// K and V tensors. Intended for advanced consumers that want to read the
// post-prefill KV state and re-pack it into a custom decode-time format
// (see examples/phi3 hybrid prefill path). Returns nullptr if the context
// does not use a single unified KV cache (e.g. SWA/iSWA, hybrid memory) or
// if the requested model-layer index has no KV slot. The returned tensor is
// owned by the context; use ggml_backend_tensor_get to copy out the data,
// or read tensor->data directly when the cache lives on a CPU backend.
// Layout (single-stream KV cache):
//   K       : (head_dim, n_head_kv, kv_size)  row-major, fastest dim = head_dim
//   V trans : (kv_size,  n_head_kv, head_dim) row-major, fastest dim = pos
//              (i.e. kv_size-long contiguous strip per (h,d), strip order
//               (h outer, d inner))
//   V !trans: (head_dim, n_head_kv, kv_size)  row-major, fastest dim = head_dim
// `llama_kv_self_v_trans` tells you which V variant the context is using;
// it is `!cparams.flash_attn` (true by default on CPU).
LLAMA_API struct ggml_tensor * llama_kv_self_layer_k(struct llama_context * ctx, int32_t il);
LLAMA_API struct ggml_tensor * llama_kv_self_layer_v(struct llama_context * ctx, int32_t il);
LLAMA_API bool                 llama_kv_self_v_trans(const struct llama_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_B612_H
