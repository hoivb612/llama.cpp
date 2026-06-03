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
#include <string>

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
