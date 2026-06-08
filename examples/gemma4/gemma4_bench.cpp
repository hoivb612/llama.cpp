// gemma4_bench.cpp -- llama-bench-style pp{N}/tg{N} throughput table
// for the Gemma-4 hand-built qquant path and (optionally) the upstream
// llama_decode forward, side-by-side.
//
// See gemma4_bench.h for the protocol summary.
#include "gemma4_bench.h"

#include "gemma4_forward.h"
#include "gemma4_weights.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gemma4 {

namespace {

using clk = std::chrono::steady_clock;

static double elapsed_ms(clk::time_point t0, clk::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// llama-bench-style format helpers ----------------------------------------

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

// Mirrors llama-bench's column widths so the side-by-side comparison
// against `llama-bench.exe` output is visually clean.
static void print_header() {
    std::printf(
        "| %-30s | %10s | %10s | %-10s | %7s | %15s | %20s |\n",
        "model", "size", "params", "backend", "threads", "test", "t/s");
    // Note: column widths must match the format string above. The colons
    // make the numeric columns right-aligned in markdown.
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

// Mean and population stddev over a sample vector.
static void mean_stddev(const std::vector<double> & v, double & mean, double & stddev) {
    if (v.empty()) { mean = 0.0; stddev = 0.0; return; }
    double sum = 0.0;
    for (double x : v) sum += x;
    mean = sum / (double) v.size();
    double sqsum = 0.0;
    for (double x : v) sqsum += (x - mean) * (x - mean);
    stddev = std::sqrt(sqsum / (double) v.size());
}

// Random in-vocab token ids, matching tools/llama-bench/llama-bench.cpp.
// We do NOT use BOS at position 0 -- the bench is content-agnostic.
static void synth_random_tokens(std::vector<int32_t> & out, int n,
                                std::mt19937 & rng, int n_vocab) {
    out.resize(n);
    std::uniform_int_distribution<int> dist(0, n_vocab - 1);
    for (int i = 0; i < n; ++i) out[i] = dist(rng);
}

// -------------------------------------------------------------------------
// qquant runner
// -------------------------------------------------------------------------
//
// pp test: one network_step(N) call with last_token_only=true.
//          t/s = N / prefill_ms * 1000.
// tg test: one network_step(1) warmup call (KV slot 0), then N
//          network_step(1) measured calls; t/s = N / decode_ms * 1000.

struct QQuantSamples {
    std::vector<double> pp_tps;  // one entry per measured rep
    std::vector<double> tg_tps;
};

static bool run_qquant_bench(const ModelF32 & m, const BenchParams & p,
                             QQuantSamples & out, std::string & error) {
    const int cap = p.pp_n + p.tg_n + 32;

    NetworkState st;
    if (!network_state_reserve(st, m, cap, error)) return false;

    std::vector<float>   logits(m.n_vocab);
    std::vector<int32_t> toks;
    std::mt19937 rng(p.seed);

    // ---- pp test ----
    for (int rep = -1; rep < p.reps; ++rep) {            // rep=-1 is warmup
        st.n_past = 0;
        synth_random_tokens(toks, p.pp_n, rng, m.n_vocab);

        const auto t0 = clk::now();
        if (!network_step(st, m, p.pp_n, toks.data(),
                          /*last_token_only=*/true, logits.data(), error)) {
            return false;
        }
        const auto t1 = clk::now();

        if (rep >= 0) {
            const double ms = elapsed_ms(t0, t1);
            out.pp_tps.push_back(ms > 0.0 ? (double) p.pp_n * 1000.0 / ms : 0.0);
        }
    }

    // ---- tg test ----
    for (int rep = -1; rep < p.reps; ++rep) {
        st.n_past = 0;

        // 1-token warmup decode (matches llama-bench's test_gen warmup
        // pattern -- KV slot at pos 0 has to exist before we start the
        // measured single-token loop).
        synth_random_tokens(toks, 1, rng, m.n_vocab);
        if (!network_step(st, m, 1, toks.data(), true, logits.data(), error)) {
            return false;
        }

        const auto t0 = clk::now();
        for (int i = 0; i < p.tg_n; ++i) {
            synth_random_tokens(toks, 1, rng, m.n_vocab);
            if (!network_step(st, m, 1, toks.data(), true, logits.data(), error)) {
                return false;
            }
        }
        const auto t1 = clk::now();

        if (rep >= 0) {
            const double ms = elapsed_ms(t0, t1);
            out.tg_tps.push_back(ms > 0.0 ? (double) p.tg_n * 1000.0 / ms : 0.0);
        }
    }

    return true;
}

// -------------------------------------------------------------------------
// upstream runner -- single llama_context reused across reps; KV cleared
// between tests / reps to keep each measurement independent.
// -------------------------------------------------------------------------

struct UpstreamSamples {
    std::vector<double> pp_tps;
    std::vector<double> tg_tps;
    bool ok = false;
    std::string err;
};

static UpstreamSamples run_upstream_bench(const llama_model * model,
                                          const BenchParams & p) {
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

    std::vector<int32_t> toks;
    std::mt19937 rng(p.seed ^ 0xA5A5u);  // offset so qquant/upstream don't share sequences

    // ---- pp test ----
    for (int rep = -1; rep < p.reps; ++rep) {
        reset_kv();
        synth_random_tokens(toks, p.pp_n, rng, n_vocab);

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
        synth_random_tokens(toks, 1, rng, n_vocab);
        {
            llama_batch b = llama_batch_get_one(toks.data(), 1);
            if (llama_decode(ctx, b) != 0) {
                if (rep >= 0) r.tg_tps.push_back(0.0);
                continue;
            }
        }

        bool ok = true;
        const auto t0 = clk::now();
        for (int i = 0; i < p.tg_n; ++i) {
            int32_t tok;
            std::uniform_int_distribution<int> dist(0, n_vocab - 1);
            tok = dist(rng);
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

// -------------------------------------------------------------------------

bool run_bench(const llama_model * model, const Weights & w,
               const BenchParams & p, std::string & error) {
    error.clear();
    if (model == nullptr) { error = "gemma4 bench: model is null"; return false; }
    if (p.pp_n <= 0 || p.tg_n <= 0 || p.reps <= 0) {
        error = "gemma4 bench: pp_n/tg_n/reps must be > 0";
        return false;
    }
    if (!p.include_qquant && !p.include_upstream) {
        error = "gemma4 bench: at least one of qquant/upstream must be enabled";
        return false;
    }

    // Model meta for the table.
    char desc[256] = {0};
    if (llama_model_desc(model, desc, sizeof(desc)) < 0) {
        std::strncpy(desc, "gemma4", sizeof(desc) - 1);
    }
    const std::string model_name  = desc;
    const std::string size_str    = fmt_size_gib(llama_model_size(model));
    const std::string params_str  = fmt_params(llama_model_n_params(model));

    // Header.
    std::printf("\n");
    print_header();

    // ---- qquant ----
    if (p.include_qquant) {
        std::fprintf(stderr, "gemma4 bench: dequantizing model for qquant path...\n");
        ModelF32 m;
        if (!dequant_model(model, w, m, error, p.n_threads)) {
            std::fprintf(stderr, "gemma4 bench: dequant_model failed: %s\n", error.c_str());
            print_row(model_name, size_str, params_str, "qquant", p.n_threads,
                      "pp" + std::to_string(p.pp_n), 0.0, 0.0);
            print_row(model_name, size_str, params_str, "qquant", p.n_threads,
                      "tg" + std::to_string(p.tg_n), 0.0, 0.0);
        } else {
            std::fprintf(stderr,
                "gemma4 bench: running qquant pp%d (reps=%d) ...\n", p.pp_n, p.reps);
            QQuantSamples qs;
            std::string qerr;
            const bool ok = run_qquant_bench(m, p, qs, qerr);
            if (!ok) {
                std::fprintf(stderr, "gemma4 bench: qquant run failed: %s\n", qerr.c_str());
            }
            double pp_mean = 0.0, pp_std = 0.0, tg_mean = 0.0, tg_std = 0.0;
            mean_stddev(qs.pp_tps, pp_mean, pp_std);
            mean_stddev(qs.tg_tps, tg_mean, tg_std);
            print_row(model_name, size_str, params_str, "qquant", p.n_threads,
                      "pp" + std::to_string(p.pp_n), pp_mean, pp_std);
            print_row(model_name, size_str, params_str, "qquant", p.n_threads,
                      "tg" + std::to_string(p.tg_n), tg_mean, tg_std);
        }
    }

    // ---- upstream ----
    if (p.include_upstream) {
        std::fprintf(stderr,
            "gemma4 bench: running upstream pp%d/tg%d (reps=%d) ...\n",
            p.pp_n, p.tg_n, p.reps);
        UpstreamSamples us = run_upstream_bench(model, p);
        if (!us.ok) {
            std::fprintf(stderr, "gemma4 bench: upstream run failed: %s\n", us.err.c_str());
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

}  // namespace gemma4
