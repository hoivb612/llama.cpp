#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// G1 — Gemma-4 dense baseline loader.
//
// Validates the model declares general.architecture == "gemma4" and
// extracts the per-model hparams we care about. Per-layer arrays
// (n_ff varies per layer, sliding_window_pattern is a 0/1 mask) are
// surfaced as vectors so callers can introspect heterogeneity. The
// 26B-A4B (MoE) variant is loadable here -- the loader does not reject
// it -- but downstream paths that assume dense FFN will fail; see the
// is_moe flag.
struct Gemma4LoadParams {
    std::string model_path;
    int n_gpu_layers = 99;
};

struct Gemma4RawModel {
    llama_model       * model = nullptr;
    const llama_vocab * vocab = nullptr;

    std::string architecture;     // expected "gemma4"
    std::string model_name;       // e.g. "Gemma-4-E2B-It"

    // Per-model shape.
    int32_t n_layer       = 0;    // 35 for E2B, 42 for E4B, 30 for 26B-A4B
    int32_t n_embd        = 0;    // 1536 / 2560 / 2048
    int32_t n_head        = 0;    // 8 across the dense variants
    int32_t n_head_kv     = 0;    // 1 (E2B), 2 (E4B), ...
    int32_t head_dim_k    = 0;    // key length per head
    int32_t head_dim_v    = 0;    // value length per head
    int32_t context_length = 0;   // 131072 for the dense Gemma-4 variants
    int32_t sliding_window = 0;   // 512 by default

    // Heterogeneous per-layer values.
    std::vector<int32_t> n_ff_per_layer;             // feed-forward length, per layer
    std::vector<int32_t> sliding_window_pattern;     // 1 = SWA, 0 = full attention, per layer

    bool is_moe = false;  // true if any layer has ffn_gate_inp / ffn_*_exps

    // RoPE.
    int32_t rope_dim   = 0;       // typically 512 for Gemma-4
    float   rope_freq_base = 0.0f;
};

bool gemma4_load_raw_model(const Gemma4LoadParams & params,
                           Gemma4RawModel         & out,
                           std::string            & error);

void gemma4_unload_raw_model(Gemma4RawModel & model);

// Convenience: dump a one-line summary of the loaded model to stderr.
void gemma4_log_summary(const Gemma4RawModel & model);
