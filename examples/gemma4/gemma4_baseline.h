#pragma once

#include "llama.h"

#include <string>
#include <vector>

// G1 — Gemma-4 baseline decode (oracle).
//
// Drives the model through the upstream llama path:
//   * Batched prefill in one llama_decode call (the fast path; we'll
//     switch to sequential prefill in a later phase if/when we need
//     byte-exact parity with a custom forward).
//   * Greedy single-token generation via llama_get_logits_ith + argmax.
//
// flash_attn is forced OFF so that, in future phases, our custom
// forward can swap in without an alignment battle against the flash
// kernel's batched layout.
//
// Returns out_generated.size() == n_gen on success, sets error and
// returns false otherwise. Optional out-params receive the wall-clock
// per-stage timings.
bool gemma4_run_baseline_decode(
        const llama_model              * model,
        const std::vector<llama_token> & prompt_tokens,
        int                              n_gen,
        int                              n_threads_prefill,
        int                              n_threads_gen,
        std::vector<llama_token>       & out_generated,
        std::string                    & error,
        double                         * out_prefill_ms = nullptr,
        double                         * out_gen_ms     = nullptr);
