#include "phi3_runtime.h"
#include "phi3_kernels.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

struct Phi3TurnMetrics {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int decode_calls = 0;
    int prefill_decode_calls = 0;
    int gen_decode_calls = 0;
    int max_ctx_used = 0;
    double tokenize_ms = 0.0;
    double decode_ms = 0.0;
    double prefill_decode_ms = 0.0;
    double gen_decode_ms = 0.0;
    double sample_ms = 0.0;
    double turn_total_ms = 0.0;
    double template_ms = 0.0;
    double ttft_ms = 0.0;
    double avg_gen_step_decode_ms = 0.0;
    double p95_gen_step_decode_ms = 0.0;
    double max_gen_step_decode_ms = 0.0;
    double avg_gen_step_sample_ms = 0.0;
    double p95_gen_step_sample_ms = 0.0;
    double max_gen_step_sample_ms = 0.0;
    double other_ms = 0.0;
    int fusion_eligible_gen_steps = 0;
    int generic_gen_steps = 0;
    int prefill_threads = 0;
    int gen_threads = 0;
    const char * gen_kernel_tag = "llama-generic";
    llama_perf_context_data perf_ctx = {};
    llama_perf_sampler_data perf_sampler = {};
};

static double phi3_percentile_95(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t idx = (size_t) ((values.size() - 1) * 95 / 100);
    return values[idx];
}

static void phi3_set_threads_if_needed(Phi3Runtime & runtime, int n_threads) {
    if (runtime.active_threads == n_threads) {
        return;
    }
    llama_set_n_threads(runtime.ctx, n_threads, n_threads);
    runtime.active_threads = n_threads;
}

static bool phi3_decode_prefill(
    Phi3Runtime & runtime,
    llama_batch & batch,
    Phi3TurnMetrics & metrics,
    std::string & error) {
    phi3_set_threads_if_needed(runtime, runtime.n_threads_prefill);
    metrics.prefill_threads = runtime.n_threads_prefill;
    double decode_dt_ms = 0.0;
    if (!phi3_decode_prefill_step(runtime.ctx, batch, decode_dt_ms, error)) {
        return false;
    }

    metrics.decode_ms += decode_dt_ms;
    metrics.decode_calls++;
    metrics.prefill_decode_ms += decode_dt_ms;
    metrics.prefill_decode_calls++;
    return true;
}

static bool phi3_decode_generate_step(
    Phi3Runtime & runtime,
    llama_token token,
    Phi3TurnMetrics & metrics,
    std::string & error) {
    phi3_set_threads_if_needed(runtime, runtime.n_threads_gen);
    metrics.gen_threads = runtime.n_threads_gen;
    double decode_dt_ms = 0.0;
    bool fusion_eligible = false;
    const char * kernel_tag = "llama-generic";
    if (!phi3_decode_generate_step(runtime.ctx, runtime.plan, runtime.gen_kernel_state, token, decode_dt_ms, fusion_eligible, kernel_tag, error)) {
        return false;
    }
    metrics.gen_kernel_tag = kernel_tag;

    metrics.decode_ms += decode_dt_ms;
    metrics.decode_calls++;
    metrics.gen_decode_ms += decode_dt_ms;
    metrics.gen_decode_calls++;
    metrics.generic_gen_steps++;
    if (fusion_eligible) {
        metrics.fusion_eligible_gen_steps++;
    }
    return true;
}

static bool phi3_generate(
    Phi3Runtime & runtime,
    const std::string & prompt,
    std::string & response,
    Phi3TurnMetrics & metrics,
    std::string & error) {
    response.clear();
    metrics = {};

    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(runtime.ctx), 0) == -1;
    const auto tok_start = std::chrono::steady_clock::now();
    const int n_prompt_tokens = -llama_tokenize(runtime.vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true);
    if (n_prompt_tokens <= 0) {
        error = "failed to tokenize prompt";
        return false;
    }

    std::vector<llama_token> prompt_tokens((size_t) n_prompt_tokens);
    if (llama_tokenize(runtime.vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        error = "failed to tokenize prompt";
        return false;
    }
    metrics.tokenize_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tok_start).count();
    metrics.prompt_tokens = n_prompt_tokens;

    llama_perf_context_reset(runtime.ctx);
    llama_perf_sampler_reset(runtime.smpl);

    llama_batch prefill_batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token gen_input_token = 0;
    bool prefill_done = false;
    const auto gen_start = std::chrono::steady_clock::now();
    std::vector<double> gen_step_decode_ms;
    std::vector<double> gen_step_sample_ms;
    gen_step_decode_ms.reserve(256);
    gen_step_sample_ms.reserve(256);
    while (true) {
        if (prefill_done && runtime.n_predict > 0 && metrics.generated_tokens >= runtime.n_predict) {
            break;
        }

        const int ctx_size = llama_n_ctx(runtime.ctx);
        const int ctx_used = llama_memory_seq_pos_max(llama_get_memory(runtime.ctx), 0) + 1;
        if (ctx_used > metrics.max_ctx_used) {
            metrics.max_ctx_used = ctx_used;
        }
        const int batch_tokens = prefill_done ? 1 : prefill_batch.n_tokens;
        if (ctx_used + batch_tokens > ctx_size) {
            error = "context size exceeded";
            return false;
        }

        if (!prefill_done) {
            if (!phi3_decode_prefill(runtime, prefill_batch, metrics, error)) {
                return false;
            }
            prefill_done = true;
        } else {
            const double prev_gen_decode_ms = metrics.gen_decode_ms;
            if (!phi3_decode_generate_step(runtime, gen_input_token, metrics, error)) {
                return false;
            }
            gen_step_decode_ms.push_back(metrics.gen_decode_ms - prev_gen_decode_ms);
        }

        const auto sample_start = std::chrono::steady_clock::now();
        llama_token tok = llama_sampler_sample(runtime.smpl, runtime.ctx, -1);
        const double sample_dt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sample_start).count();
        metrics.sample_ms += sample_dt_ms;
        if (prefill_done) {
            gen_step_sample_ms.push_back(sample_dt_ms);
        }
        if (llama_vocab_is_eog(runtime.vocab, tok)) {
            break;
        }

        char buf[256];
        const int n = llama_token_to_piece(runtime.vocab, tok, buf, sizeof(buf), 0, true);
        if (n < 0) {
            error = "failed to convert token to piece";
            return false;
        }

        std::string piece(buf, (size_t) n);
        printf("%s", piece.c_str());
        fflush(stdout);
        response += piece;
        metrics.generated_tokens++;
        if (metrics.generated_tokens == 1) {
            metrics.ttft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gen_start).count();
        }

        gen_input_token = tok;
    }

    if (!gen_step_decode_ms.empty()) {
        metrics.avg_gen_step_decode_ms = metrics.gen_decode_ms / gen_step_decode_ms.size();
        metrics.p95_gen_step_decode_ms = phi3_percentile_95(gen_step_decode_ms);
        metrics.max_gen_step_decode_ms = *std::max_element(gen_step_decode_ms.begin(), gen_step_decode_ms.end());
    }
    if (!gen_step_sample_ms.empty()) {
        metrics.avg_gen_step_sample_ms = metrics.sample_ms / gen_step_sample_ms.size();
        metrics.p95_gen_step_sample_ms = phi3_percentile_95(gen_step_sample_ms);
        metrics.max_gen_step_sample_ms = *std::max_element(gen_step_sample_ms.begin(), gen_step_sample_ms.end());
    }

    metrics.perf_ctx = llama_perf_context(runtime.ctx);
    metrics.perf_sampler = llama_perf_sampler(runtime.smpl);
    error.clear();
    return true;
}

static bool phi3_runtime_run_turn(Phi3Runtime & runtime, const std::string & user_text, std::string & error) {
    const auto turn_start = std::chrono::steady_clock::now();
    runtime.messages.push_back({"user", strdup(user_text.c_str())});

    const auto template_start = std::chrono::steady_clock::now();
    int new_len = llama_chat_apply_template(runtime.chat_template, runtime.messages.data(), runtime.messages.size(), true, runtime.formatted.data(), runtime.formatted.size());
    if (new_len > (int) runtime.formatted.size()) {
        runtime.formatted.resize((size_t) new_len);
        new_len = llama_chat_apply_template(runtime.chat_template, runtime.messages.data(), runtime.messages.size(), true, runtime.formatted.data(), runtime.formatted.size());
    }
    if (new_len < 0) {
        error = "failed to apply chat template";
        return false;
    }

    const std::string prompt(runtime.formatted.begin() + runtime.prev_len, runtime.formatted.begin() + new_len);
    std::string response;
    Phi3TurnMetrics metrics;
    metrics.template_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - template_start).count();
    printf("\033[33m");
    if (!phi3_generate(runtime, prompt, response, metrics, error)) {
        printf("\n\033[0m");
        return false;
    }
    printf("\n\033[0m");

    runtime.messages.push_back({"assistant", strdup(response.c_str())});
    runtime.prev_len = llama_chat_apply_template(runtime.chat_template, runtime.messages.data(), runtime.messages.size(), false, nullptr, 0);
    if (runtime.prev_len < 0) {
        error = "failed to apply chat template";
        return false;
    }

    runtime.turn_index++;
    runtime.active_threads = 0;
    metrics.turn_total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - turn_start).count();
    metrics.other_ms = std::max(0.0, metrics.turn_total_ms - (metrics.template_ms + metrics.tokenize_ms + metrics.decode_ms + metrics.sample_ms));
    if (runtime.enable_instrumentation) {
        const double gen_tps = metrics.gen_decode_ms > 0.0 ? (1000.0 * metrics.generated_tokens) / metrics.gen_decode_ms : 0.0;
        const double prefill_tps = metrics.prefill_decode_ms > 0.0 ? (1000.0 * metrics.prompt_tokens) / metrics.prefill_decode_ms : 0.0;
        const double prompt_tps = metrics.perf_ctx.t_p_eval_ms > 0.0 ? (1000.0 * metrics.perf_ctx.n_p_eval) / metrics.perf_ctx.t_p_eval_ms : 0.0;
        const double decode_ratio = metrics.turn_total_ms > 0.0 ? (100.0 * metrics.decode_ms) / metrics.turn_total_ms : 0.0;
        fprintf(stderr,
            "\nphi3 profile: turn=%d prompt_tok=%d gen_tok=%d decode_calls=%d prefill_calls=%d gen_calls=%d tokenize_ms=%.2f template_ms=%.2f "
            "decode_ms=%.2f prefill_decode_ms=%.2f gen_decode_ms=%.2f sample_ms=%.2f other_ms=%.2f decode_pct=%.2f "
            "ttft_ms=%.2f gen_step_avg_ms=%.2f gen_step_p95_ms=%.2f gen_step_max_ms=%.2f "
            "sample_step_avg_ms=%.3f sample_step_p95_ms=%.3f sample_step_max_ms=%.3f "
            "llama_prompt_ms=%.2f llama_gen_ms=%.2f prefill_tps=%.2f prompt_tps=%.2f gen_tps=%.2f prefill_threads=%d gen_threads=%d "
            "gen_kernel=%s fusion_eligible_gen_steps=%d generic_gen_steps=%d ctx_peak=%d/%d graph_reused=%d total_ms=%.2f\n",
            runtime.turn_index,
            metrics.prompt_tokens,
            metrics.generated_tokens,
            metrics.decode_calls,
            metrics.prefill_decode_calls,
            metrics.gen_decode_calls,
            metrics.tokenize_ms,
            metrics.template_ms,
            metrics.decode_ms,
            metrics.prefill_decode_ms,
            metrics.gen_decode_ms,
            metrics.sample_ms,
            metrics.other_ms,
            decode_ratio,
            metrics.ttft_ms,
            metrics.avg_gen_step_decode_ms,
            metrics.p95_gen_step_decode_ms,
            metrics.max_gen_step_decode_ms,
            metrics.avg_gen_step_sample_ms,
            metrics.p95_gen_step_sample_ms,
            metrics.max_gen_step_sample_ms,
            metrics.perf_ctx.t_p_eval_ms,
            metrics.perf_ctx.t_eval_ms,
            prefill_tps,
            prompt_tps,
            gen_tps,
            metrics.prefill_threads,
            metrics.gen_threads,
            metrics.gen_kernel_tag,
            metrics.fusion_eligible_gen_steps,
            metrics.generic_gen_steps,
            metrics.max_ctx_used,
            llama_n_ctx(runtime.ctx),
            metrics.perf_ctx.n_reused,
            metrics.turn_total_ms);
    }

    error.clear();
    return true;
}

bool phi3_runtime_init(
    const Phi3RawModel & raw,
    const Phi3ExecutionPlan & plan,
    const Phi3RuntimeParams & params,
    Phi3Runtime & out,
    std::string & error) {
    phi3_runtime_free(out);

    if (!raw.model || !raw.vocab) {
        error = "runtime init requires a loaded raw model";
        return false;
    }
    if (plan.n_layer <= 0 || plan.n_embd <= 0) {
        error = "runtime init requires a valid execution plan";
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_ctx;
    ctx_params.n_batch = params.n_ctx;

    out.ctx = llama_init_from_model(raw.model, ctx_params);
    if (!out.ctx) {
        error = "failed to create context";
        phi3_runtime_free(out);
        return false;
    }

    out.smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!out.smpl) {
        error = "failed to create sampler chain";
        phi3_runtime_free(out);
        return false;
    }
    llama_sampler_chain_add(out.smpl, llama_sampler_init_min_p(params.min_p, 1));
    llama_sampler_chain_add(out.smpl, llama_sampler_init_temp(params.temp));
    llama_sampler_chain_add(out.smpl, llama_sampler_init_dist(params.seed));

    out.vocab = raw.vocab;
    out.plan = plan;
    out.formatted.resize((size_t) params.n_ctx);
    out.chat_template = llama_model_chat_template(raw.model, nullptr);
    out.enable_instrumentation = params.enable_instrumentation;
    out.n_predict = params.n_predict;
    out.seed = params.seed;
    if (!out.chat_template) {
        error = "model is missing chat template";
        phi3_runtime_free(out);
        return false;
    }

    out.prev_len = 0;
    const unsigned hw_threads = std::thread::hardware_concurrency();
    const int detected_threads = (int) (hw_threads > 0 ? hw_threads : 4);
    out.n_threads_prefill = params.n_threads_prefill > 0 ? params.n_threads_prefill : std::max(1, std::min(32, detected_threads));
    out.n_threads_gen = params.n_threads_gen > 0 ? params.n_threads_gen : std::max(1, std::min(8, detected_threads / 2));
    if (out.n_threads_gen > out.n_threads_prefill) {
        out.n_threads_gen = out.n_threads_prefill;
    }
    out.active_threads = 0;
    phi3_gen_kernel_state_init(out.gen_kernel_state);
    error.clear();
    return true;
}

bool phi3_runtime_run_single_prompt(Phi3Runtime & runtime, const std::string & prompt, std::string & error) {
    if (prompt.empty()) {
        error = "prompt cannot be empty";
        return false;
    }
    return phi3_runtime_run_turn(runtime, prompt, error);
}

bool phi3_runtime_run_interactive(Phi3Runtime & runtime, std::string & error) {
    while (true) {
        printf("\033[32m> \033[0m");
        std::string user;
        std::getline(std::cin, user);
        if (user.empty()) {
            break;
        }
        if (!phi3_runtime_run_turn(runtime, user, error)) {
            return false;
        }
    }

    error.clear();
    return true;
}

void phi3_runtime_free(Phi3Runtime & runtime) {
    for (auto & msg : runtime.messages) {
        free(const_cast<char *>(msg.content));
    }
    runtime.messages.clear();
    runtime.formatted.clear();
    runtime.prev_len = 0;
    runtime.turn_index = 0;
    runtime.chat_template = nullptr;
    runtime.vocab = nullptr;
    runtime.plan = {};
    runtime.enable_instrumentation = true;
    runtime.n_predict = 256;
    runtime.seed = LLAMA_DEFAULT_SEED;
    runtime.n_threads_prefill = 0;
    runtime.n_threads_gen = 0;
    runtime.active_threads = 0;
    runtime.gen_kernel_state = {};

    if (runtime.smpl) {
        llama_sampler_free(runtime.smpl);
        runtime.smpl = nullptr;
    }
    if (runtime.ctx) {
        llama_free(runtime.ctx);
        runtime.ctx = nullptr;
    }
}
