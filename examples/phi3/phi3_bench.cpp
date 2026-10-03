// phi3_bench.cpp -- llama-bench-style pp{N}/tg{N} table for Phi-3.
// See phi3_bench.h for the protocol summary.
#include "phi3_bench.h"

#include "phi3_fused_graph.h"   // phi3_run_qquant_decode

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace phi3_bench {

namespace {

using clk = std::chrono::steady_clock;

static double elapsed_ms(clk::time_point t0, clk::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static std::string fmt_size_gib(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f GiB", (double) bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

static std::string fmt_params(uint64_t n_params) {
    char buf[32];
    if (n_params >= 1000ULL * 1000ULL * 1000ULL) {
        std::snprintf(buf, sizeof(buf), "%.2f B", (double) n_params / 1.0e9);
    } else {
        std::snprintf(buf, sizeof(buf), "%.2f M", (double) n_params / 1.0e6);
    }
    return buf;
}

static void print_header() {
    std::printf(
        "| %-30s | %10s | %10s | %-10s | %7s | %15s | %20s |\n",
        "model", "size", "params", "backend", "threads", "test", "t/s");
    std::printf(
        "| %-30s | %10s | %10s | %-10s | %7s | %15s | %20s |\n",
        "------------------------------", "---------:", "---------:",
        "----------", "------:", "--------------:", "-------------------:");
}

static void print_row(const std::string & model_name,
                      const std::string & size_str,
                      const std::string & params_str,
                      const std::string & backend,
                      int n_threads,
                      const std::string & test_name,
                      double mean_tps,
                      double stddev_tps) {
    char val[40];
    if (mean_tps <= 0.0) {
        std::snprintf(val, sizeof(val), "FAIL");
    } else {
        std::snprintf(val, sizeof(val), "%9.2f \xC2\xB1 %4.2f", mean_tps, stddev_tps);
    }
    std::printf(
        "| %-30s | %10s | %10s | %-10s | %7d | %15s | %20s |\n",
        model_name.c_str(), size_str.c_str(), params_str.c_str(),
        backend.c_str(), n_threads, test_name.c_str(), val);
    std::fflush(stdout);
}

static void mean_stddev(const std::vector<double> & v, double & mean, double & stddev) {
    if (v.empty()) { mean = 0.0; stddev = 0.0; return; }
    double sum = 0.0;
    for (double x : v) sum += x;
    mean = sum / (double) v.size();
    double sqsum = 0.0;
    for (double x : v) sqsum += (x - mean) * (x - mean);
    stddev = std::sqrt(sqsum / (double) v.size());
}

// -------------------------------------------------------------------------
// qquant runner: call phi3_run_qquant_decode once per rep and read out
// out_prefill_ms / out_gen_ms. This is the same path as
// --phi3-fused-qquant-debug. Tokens are random in-vocab; phi3_run_qquant_decode
// has no EOG short-circuit so the rep always completes n_gen steps.
//
// For pp test: prompt=N random tokens, n_gen=1; use prefill_ms.
// For tg test: prompt=1 random token,   n_gen=N; use gen_ms.

struct QQuantSamples {
    std::vector<double> pp_tps;
    std::vector<double> tg_tps;
};

static bool run_qquant_bench(const llama_model * model, const Params & p,
                             QQuantSamples & out, std::string & error) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::mt19937 rng(p.seed);
    std::uniform_int_distribution<int> dist(0, n_vocab - 1);

    auto synth = [&](std::vector<llama_token> & toks, int n) {
        toks.resize(n);
        for (int i = 0; i < n; ++i) toks[i] = dist(rng);
    };

    std::vector<llama_token> prompt;
    std::vector<llama_token> out_tokens;

    // ---- pp test (warmup + reps measured) ----
    for (int rep = -1; rep < p.reps; ++rep) {
        synth(prompt, p.pp_n);
        out_tokens.clear();

        double prefill_ms = 0.0, gen_ms = 0.0;
        const bool ok = phi3_run_qquant_decode(
                model, prompt, /*n_gen=*/1, p.n_threads,
                p.fuse_rmsnorm_quant, /*profile_per_op=*/false,
                p.attn_parallel, out_tokens, error,
                &prefill_ms, &gen_ms);
        if (!ok) return false;

        if (rep >= 0) {
            out.pp_tps.push_back(prefill_ms > 0.0
                ? (double) p.pp_n * 1000.0 / prefill_ms : 0.0);
        }
    }

    // ---- tg test ----
    for (int rep = -1; rep < p.reps; ++rep) {
        synth(prompt, 1);
        out_tokens.clear();

        double prefill_ms = 0.0, gen_ms = 0.0;
        const bool ok = phi3_run_qquant_decode(
                model, prompt, /*n_gen=*/p.tg_n, p.n_threads,
                p.fuse_rmsnorm_quant, /*profile_per_op=*/false,
                p.attn_parallel, out_tokens, error,
                &prefill_ms, &gen_ms);
        if (!ok) return false;

        if (rep >= 0) {
            out.tg_tps.push_back(gen_ms > 0.0
                ? (double) p.tg_n * 1000.0 / gen_ms : 0.0);
        }
    }

    return true;
}

// -------------------------------------------------------------------------
// upstream runner -- llama_context reused across reps; KV cleared between
// tests / reps. Matches tools/llama-bench/llama-bench.cpp's test_prompt
// and test_gen helpers exactly (random in-vocab tokens, llama_decode for
// every step). flash_attn forced OFF to match phi3_run_qquant_decode's
// llama_context configuration.

struct UpstreamSamples {
    std::vector<double> pp_tps;
    std::vector<double> tg_tps;
    bool ok = false;
    std::string err;
};

static UpstreamSamples run_upstream_bench(const llama_model * model,
                                          const Params & p) {
    UpstreamSamples r;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx   = p.pp_n + p.tg_n + 32;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_ctx;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf         = true;

    llama_context * ctx = llama_init_from_model(const_cast<llama_model *>(model), cp);
    if (!ctx) { r.err = "llama_init_from_model failed"; return r; }
    llama_set_n_threads(ctx, p.n_threads, p.n_threads);

    auto reset_kv = [&]() { llama_memory_clear(llama_get_memory(ctx), true); };

    std::vector<llama_token> toks;
    std::mt19937 rng(p.seed ^ 0xA5A5u);
    std::uniform_int_distribution<int> dist(0, n_vocab - 1);

    // ---- pp test ----
    for (int rep = -1; rep < p.reps; ++rep) {
        reset_kv();
        toks.resize(p.pp_n);
        for (int i = 0; i < p.pp_n; ++i) toks[i] = dist(rng);

        const auto t0 = clk::now();
        llama_batch b = llama_batch_get_one(toks.data(), p.pp_n);
        const int rc = llama_decode(ctx, b);
        const auto t1 = clk::now();

        if (rc != 0) {
            if (rep >= 0) r.pp_tps.push_back(0.0);
            continue;
        }
        if (rep >= 0) {
            const double ms = elapsed_ms(t0, t1);
            r.pp_tps.push_back(ms > 0.0 ? (double) p.pp_n * 1000.0 / ms : 0.0);
        }
    }

    // ---- tg test ----
    for (int rep = -1; rep < p.reps; ++rep) {
        reset_kv();

        // 1-token warmup decode at pos 0
        llama_token t0_tok = dist(rng);
        {
            llama_batch b = llama_batch_get_one(&t0_tok, 1);
            if (llama_decode(ctx, b) != 0) {
                if (rep >= 0) r.tg_tps.push_back(0.0);
                continue;
            }
        }

        bool ok = true;
        const auto t0 = clk::now();
        for (int i = 0; i < p.tg_n; ++i) {
            llama_token tok = dist(rng);
            llama_batch b = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, b) != 0) { ok = false; break; }
        }
        const auto t1 = clk::now();

        if (rep >= 0) {
            if (!ok) { r.tg_tps.push_back(0.0); }
            else {
                const double ms = elapsed_ms(t0, t1);
                r.tg_tps.push_back(ms > 0.0 ? (double) p.tg_n * 1000.0 / ms : 0.0);
            }
        }
    }

    llama_free(ctx);
    r.ok = true;
    return r;
}

}  // namespace

bool run_bench(const llama_model * model, const Params & p, std::string & error) {
    error.clear();
    if (model == nullptr) { error = "phi3 bench: model is null"; return false; }
    if (p.pp_n <= 0 || p.tg_n <= 0 || p.reps <= 0) {
        error = "phi3 bench: pp_n/tg_n/reps must be > 0";
        return false;
    }
    if (!p.include_qquant && !p.include_upstream) {
        error = "phi3 bench: at least one of qquant/upstream must be enabled";
        return false;
    }

    char desc[256] = {0};
    if (llama_model_desc(model, desc, sizeof(desc)) < 0) {
        std::strncpy(desc, "phi3", sizeof(desc) - 1);
    }
    const std::string model_name  = desc;
    const std::string size_str    = fmt_size_gib(llama_model_size(model));
    const std::string params_str  = fmt_params(llama_model_n_params(model));

    std::printf("\n");
    print_header();

    if (p.include_qquant) {
        std::fprintf(stderr,
            "phi3 bench: running qquant pp%d/tg%d (reps=%d, fuse=%d attn_parallel=%d) ...\n",
            p.pp_n, p.tg_n, p.reps, (int) p.fuse_rmsnorm_quant, (int) p.attn_parallel);
        QQuantSamples qs;
        std::string qerr;
        const bool ok = run_qquant_bench(model, p, qs, qerr);
        if (!ok) {
            std::fprintf(stderr, "phi3 bench: qquant run failed: %s\n", qerr.c_str());
        }
        double pp_mean = 0.0, pp_std = 0.0, tg_mean = 0.0, tg_std = 0.0;
        mean_stddev(qs.pp_tps, pp_mean, pp_std);
        mean_stddev(qs.tg_tps, tg_mean, tg_std);
        print_row(model_name, size_str, params_str, "qquant", p.n_threads,
                  "pp" + std::to_string(p.pp_n), pp_mean, pp_std);
        print_row(model_name, size_str, params_str, "qquant", p.n_threads,
                  "tg" + std::to_string(p.tg_n), tg_mean, tg_std);
    }

    if (p.include_upstream) {
        std::fprintf(stderr,
            "phi3 bench: running upstream pp%d/tg%d (reps=%d) ...\n",
            p.pp_n, p.tg_n, p.reps);
        UpstreamSamples us = run_upstream_bench(model, p);
        if (!us.ok) {
            std::fprintf(stderr, "phi3 bench: upstream run failed: %s\n", us.err.c_str());
            print_row(model_name, size_str, params_str, "upstream", p.n_threads,
                      "pp" + std::to_string(p.pp_n), 0.0, 0.0);
            print_row(model_name, size_str, params_str, "upstream", p.n_threads,
                      "tg" + std::to_string(p.tg_n), 0.0, 0.0);
        } else {
            double pp_mean = 0.0, pp_std = 0.0, tg_mean = 0.0, tg_std = 0.0;
            mean_stddev(us.pp_tps, pp_mean, pp_std);
            mean_stddev(us.tg_tps, tg_mean, tg_std);
            print_row(model_name, size_str, params_str, "upstream", p.n_threads,
                      "pp" + std::to_string(p.pp_n), pp_mean, pp_std);
            print_row(model_name, size_str, params_str, "upstream", p.n_threads,
                      "tg" + std::to_string(p.tg_n), tg_mean, tg_std);
        }
    }

    std::printf("\n");
    std::fflush(stdout);
    return true;
}

}  // namespace phi3_bench
