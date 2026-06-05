#include "gemma4_baseline.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Greedy argmax over the full logits vector. Returns the token id with
// the highest logit value. We deliberately don't apply any sampler
// (temp/min_p/top_k) here -- G1 is meant to be a deterministic oracle.
llama_token argmax_logits(const float * logits, int n_vocab) {
    llama_token best   = 0;
    float       best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best   = (llama_token) i;
        }
    }
    return best;
}

}  // namespace

bool gemma4_run_baseline_decode(
        const llama_model              * model,
        const std::vector<llama_token> & prompt_tokens,
        int                              n_gen,
        int                              n_threads_prefill,
        int                              n_threads_gen,
        std::vector<llama_token>       & out_generated,
        std::string                    & error,
        double                         * out_prefill_ms,
        double                         * out_gen_ms) {
    error.clear();
    if (model == nullptr)       { error = "gemma4_run_baseline_decode: model is null";       return false; }
    if (prompt_tokens.empty())  { error = "gemma4_run_baseline_decode: prompt_tokens empty"; return false; }
    if (n_gen <= 0)             { error = "gemma4_run_baseline_decode: n_gen must be > 0";   return false; }
    if (n_threads_prefill <= 0) n_threads_prefill = 1;
    if (n_threads_gen     <= 0) n_threads_gen     = 1;

    const int n_prompt = (int) prompt_tokens.size();
    const int n_ctx    = n_prompt + n_gen + 64;
    const int n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_ctx;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf         = true;

    llama_context * ctx = llama_init_from_model(const_cast<llama_model *>(model), cp);
    if (!ctx) { error = "gemma4_run_baseline_decode: llama_init_from_model failed"; return false; }

    std::fprintf(stderr,
                 "gemma4 baseline-decode: starting (n_prompt=%d, n_gen=%d, n_ctx=%d, "
                 "prefill_threads=%d, gen_threads=%d)\n",
                 n_prompt, n_gen, n_ctx, n_threads_prefill, n_threads_gen);

    // ---------- Prefill (batched) ----------
    llama_set_n_threads(ctx, n_threads_prefill, n_threads_prefill);
    llama_batch prefill = llama_batch_get_one(
        const_cast<llama_token *>(prompt_tokens.data()), n_prompt);

    const auto t_pre = std::chrono::steady_clock::now();
    if (llama_decode(ctx, prefill) != 0) {
        error = "gemma4_run_baseline_decode: llama_decode (prefill) failed";
        llama_free(ctx);
        return false;
    }
    const double t_pre_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_pre).count();
    if (out_prefill_ms) *out_prefill_ms = t_pre_ms;
    std::fprintf(stderr,
                 "gemma4 baseline-decode: prefill (batched) %d tokens in %.2f s (%.3f s/tok)\n",
                 n_prompt, t_pre_ms / 1000.0, t_pre_ms / 1000.0 / std::max(1, n_prompt));

    // ---------- Sample first gen token ----------
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        error = "gemma4_run_baseline_decode: missing logits after prefill";
        llama_free(ctx);
        return false;
    }
    llama_token next = argmax_logits(logits, n_vocab);
    out_generated.reserve(out_generated.size() + n_gen);
    out_generated.push_back(next);

    // ---------- Generation ----------
    llama_set_n_threads(ctx, n_threads_gen, n_threads_gen);
    const auto t_gen = std::chrono::steady_clock::now();
    for (int i = 1; i < n_gen; ++i) {
        llama_batch step = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, step) != 0) {
            error = "gemma4_run_baseline_decode: llama_decode (gen) failed at step " +
                    std::to_string(i);
            llama_free(ctx);
            return false;
        }
        const float * h = llama_get_logits_ith(ctx, -1);
        if (!h) {
            error = "gemma4_run_baseline_decode: missing logits at step " + std::to_string(i);
            llama_free(ctx);
            return false;
        }
        next = argmax_logits(h, n_vocab);
        out_generated.push_back(next);
    }
    const double t_gen_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_gen).count();
    if (out_gen_ms) *out_gen_ms = t_gen_ms;
    std::fprintf(stderr,
                 "gemma4 baseline-decode: generated %d tokens in %.2f s (%.3f s/tok)\n",
                 n_gen, t_gen_ms / 1000.0, t_gen_ms / 1000.0 / std::max(1, n_gen));

    llama_free(ctx);
    return true;
}
