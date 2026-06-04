#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP

#include "llama.h"
#if defined(LLAMA_B612_API)
#include "llama_b612.h"
#endif

#include <cstdint>
#include <map>

#if !defined(LLAMA_B612_API)
// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

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
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);
#endif // !defined(LLAMA_B612_API)

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

#if !defined(LLAMA_B612_API)
LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

//
// pre-norm embeddings (hidden state before the final output norm)
//

// Set whether the context outputs pre-norm embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_pre_norm(struct llama_context * ctx, bool value, bool masked);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_pre_norm    (struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
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
// with --phi3-fused-lmhead. Greedy-only (no impact on logits values, only
// on intermediate quantization paths). No-op for non-Phi3 architectures.
LLAMA_API void llama_set_phi3_fused_decode(struct llama_context * ctx, bool value);

// Look up a model tensor by GGUF tensor name (e.g. "output.weight").
// Returns nullptr if the tensor is absent (some models tie lm_head to token_embd).
// The returned pointer is owned by the model; do not free.
LLAMA_API const struct ggml_tensor * llama_model_get_tensor_by_name(
        const struct llama_model * model, const char * name);

// B612 / Phi3 hybrid prefill: raw access to the unified KV cache's per-layer
// K and V tensors. Intended for advanced consumers that want to read the
// post-prefill KV state and re-pack it into a custom decode-time format
// (see examples/phi3 hybrid prefill path). Returns nullptr if the context
// does not use a single unified KV cache (e.g. SWA/iSWA, hybrid memory) or
// if the requested model-layer index has no KV slot. The returned tensor is
// owned by the context; use ggml_backend_tensor_get to copy out the data.
// Layout (single-stream KV cache):
//   K       : (head_dim, n_head_kv, kv_size)  row-major, fastest dim = head_dim
//   V trans : (kv_size,  n_head_kv, head_dim) row-major, fastest dim = pos
//   V !trans: (head_dim, n_head_kv, kv_size)  row-major, fastest dim = head_dim
// `llama_kv_self_v_trans` tells you which V variant the context is using;
// it is `!cparams.flash_attn` (true by default on CPU).
LLAMA_API struct ggml_tensor * llama_kv_self_layer_k(struct llama_context * ctx, int32_t il);
LLAMA_API struct ggml_tensor * llama_kv_self_layer_v(struct llama_context * ctx, int32_t il);
LLAMA_API bool                 llama_kv_self_v_trans(const struct llama_context * ctx);
#endif // !defined(LLAMA_B612_API)
