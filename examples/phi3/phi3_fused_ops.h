#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
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

// Resolve the lm_head weight tensor on the model and pre-allocate the scratch
// buffer for the per-step q-quantization of the hidden state.
// Returns false (and sets error) when the tensor is missing or the type is
// unsupported by the CPU type traits.
bool phi3_fused_lmhead_init(const struct llama_model * model, Phi3FusedLmHead & out, std::string & error);

// Compute argmax_v ( weight[:, v] . hidden ) and write the winning row index.
// hidden must point to n_embd contiguous f32 values (e.g. from
// llama_get_embeddings_ith(ctx, -1)).
bool phi3_fused_lmhead_argmax(
    Phi3FusedLmHead & lm,
    const float *     hidden,
    int               n_threads,
    llama_token &     out_token,
    std::string &     error);
