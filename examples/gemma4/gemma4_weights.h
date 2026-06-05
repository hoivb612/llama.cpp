#pragma once

// Gemma-4 custom-forward weights resolver (G3 scaffolding).
//
// Resolves every tensor in a gemma4 dense GGUF, validates internal
// consistency, and records per-layer dims (which vary in gemma4):
//
//   * head_dim per layer: SWA layers use head_dim_swa (e.g. 256),
//     full-attention layers use head_dim_full (e.g. 512). Pattern is
//     fixed by tokenizer.ggml.* sliding_window_pattern.
//   * n_ff per layer:     gemma4 has a "double-wide MLP" mode where
//     the last few layers have 2x n_ff (e.g. E2B: 6144 for layers
//     0-29, 12288 for layers 30-34).
//   * attn_v is optional (if missing, V = K at the use site).
//
// No graph is built here -- this only resolves pointers and dims so a
// later pass (G3.2+) can hand-build the F32 forward.
//
// References:
//   src/models/gemma4.cpp - oracle graph (load_arch_hparams,
//                          load_arch_tensors, graph::graph).
//   examples/phi3/phi3_fused_graph.h - same idea, simpler model.

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;
struct llama_model;

namespace gemma4 {

struct LayerWeights {
    // Per-layer dims (read from actual tensor shapes, not hparams):
    int  head_dim   = 0;   // attn_q.ne[1] / n_head
    int  n_ff       = 0;   // ffn_gate.ne[1]
    bool is_swa     = false;
    bool has_kv     = false;  // attn_k present (always true for dense E2B/E4B)
    bool has_v_proj = false;  // attn_v present (E2B/E4B: yes; some variants reuse K)

    // Shared-KV pattern (Gemma 3n / Gemma 4):
    //   The last N layers reuse K/V cache from earlier layers instead of
    //   computing their own. Reuse rule (from llama-model.cpp):
    //     if il >= n_layer_kv_from_start:
    //         reuse = n_layer_kv_from_start - (is_swa(il) ? 2 : 1)
    //   For Gemma-4 E2B: n_layer_kv_from_start=15 (35 layers - 20 shared);
    //   SWA layers 15..34 reuse layer 13 (SWA); FULL layers 19,24,29,34
    //   reuse layer 14 (FULL).
    //   -1 means "compute own K/V" (the default).
    int  kv_reuse_il = -1;

    // Norms (all F32).
    const ggml_tensor * attn_norm        = nullptr; // [n_embd]
    const ggml_tensor * attn_q_norm      = nullptr; // [head_dim] -- applied per-head along last dim
    const ggml_tensor * attn_k_norm      = nullptr; // [head_dim]
    const ggml_tensor * post_attn_norm   = nullptr; // [n_embd]   "blk.N.post_attention_norm.weight"
    const ggml_tensor * post_norm        = nullptr; // [n_embd]   "blk.N.post_norm.weight"  -- per-layer post-MLP res-norm
    const ggml_tensor * ffn_norm         = nullptr; // [n_embd]
    const ggml_tensor * post_ffw_norm    = nullptr; // [n_embd]

    // Attention projections (K-quant in Q4_K_M).
    const ggml_tensor * wq = nullptr;  // [n_embd, n_head * head_dim]
    const ggml_tensor * wk = nullptr;  // [n_embd, n_head_kv * head_dim]
    const ggml_tensor * wv = nullptr;  // [n_embd, n_head_kv * head_dim]   may be null
    const ggml_tensor * wo = nullptr;  // [n_head * head_dim, n_embd]

    // FFN (SwiGLU with GELU as activation, parallel up/gate).
    const ggml_tensor * ffn_gate = nullptr;  // [n_embd, n_ff]
    const ggml_tensor * ffn_up   = nullptr;  // [n_embd, n_ff]
    const ggml_tensor * ffn_down = nullptr;  // [n_ff,   n_embd]

    // Per-layer-embedding (Gemma 3n / Gemma 4 PLE feature).
    const ggml_tensor * inp_gate  = nullptr;  // [n_embd, n_embd_per_layer]   F32
    const ggml_tensor * proj      = nullptr;  // [n_embd_per_layer, n_embd]   F32

    // Optional per-layer scalar that scales l_out before residual entering next layer.
    const ggml_tensor * layer_output_scale = nullptr;  // [1] F32; may be null
};

struct Weights {
    // Global hparams (resolved from GGUF metadata + tensor shapes).
    int  n_layer            = 0;
    int  n_embd             = 0;
    int  n_head             = 0;
    int  n_head_kv          = 0;
    int  n_embd_per_layer   = 0;
    int  n_vocab            = 0;
    int  n_swa              = 0;       // sliding-window size (e.g. 512)
    int  n_layer_kv_from_start = -1;   // Gemma-4 shared-KV boundary. -1 == all layers own their KV.
    int  rope_dim           = 0;       // rope.dimension_count (e.g. 256 or head_dim)
    float rope_freq_base    = 0.0f;    // e.g. 1e6 (full-attn layers)
    float rope_freq_base_swa = 0.0f;   // e.g. 1e4 (SWA layers)
    float rms_eps           = 0.0f;    // attention.layer_norm_rms_epsilon
    float final_logit_softcap = 0.0f;  // 0 means no softcap

    // Globals.
    const ggml_tensor * tok_embd             = nullptr;  // [n_embd, n_vocab]
    const ggml_tensor * output               = nullptr;  // may alias tok_embd (tied)
    bool                output_tied_to_embd  = false;
    const ggml_tensor * output_norm          = nullptr;  // [n_embd] F32
    const ggml_tensor * per_layer_tok_embd   = nullptr;  // [n_embd_per_layer * n_layer, n_vocab]
    const ggml_tensor * per_layer_model_proj = nullptr;  // [n_embd, n_embd_per_layer * n_layer]
    const ggml_tensor * per_layer_proj_norm  = nullptr;  // [n_embd_per_layer] F32
    const ggml_tensor * rope_freqs           = nullptr;  // [rope_dim/2] F32 (optional; if set, used by full-attn layers)

    // Per-layer.
    std::vector<LayerWeights> layers;

    // Convenience accessors.
    int n_ff_max() const {
        int m = 0;
        for (const auto & L : layers) if (L.n_ff > m) m = L.n_ff;
        return m;
    }
    int head_dim_max() const {
        int m = 0;
        for (const auto & L : layers) if (L.head_dim > m) m = L.head_dim;
        return m;
    }
};

// Resolve every tensor by name and validate shapes. On failure, returns
// false and sets `error` to a human-readable reason. The model must be
// loaded and must have general.architecture == "gemma4". MoE variants
// are rejected here (caller should switch to the upstream graph for those).
bool resolve(const llama_model * model, Weights & out, std::string & error);

// Dump the resolved schema to stderr (per-layer dims, swa pattern, tensor types).
void dump(const Weights & w);

} // namespace gemma4
