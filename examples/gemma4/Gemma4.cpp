// G1 — Gemma-4 dense baseline CLI.
//
// Usage:
//   Gemma4 -m MODEL.gguf [-p "prompt"] [-n N] [-c N] [-ngl N]
//          [--threads-prefill N] [--threads-gen N]
//
// Loads the model, applies the model's built-in chat template to the
// prompt, runs gemma4_run_baseline_decode (batched prefill + greedy
// gen via llama logits + argmax), and prints the generated tokens.
//
// This is the oracle for future custom-forward work in this directory
// (mirrors the structure of examples/phi3/Phi3.cpp). Deliberately
// minimal: no qquant, no fused ops, no flags beyond what's needed to
// drive baseline decode on E2B and E4B.

#include "llama.h"
#include "gemma4_baseline.h"
#include "gemma4_bench.h"
#include "gemma4_chat.h"
#include "gemma4_forward.h"
#include "gemma4_kernels.h"
#include "gemma4_loader.h"
#include "gemma4_weights.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static void print_usage(int /*argc*/, char ** argv) {
    std::printf(
        "\n  %s -m gemma-4-E2B-it-Q4_K_M.gguf [-p \"why is the sky blue?\"] "
        "[-n 64] [-c 0] [-ngl 99] [--threads-prefill N] [--threads-gen N]\n"
        "\n"
        "  Hand-coded F32 self-tests (custom forward path):\n"
        "    --gemma4-dump-weights              resolve+print tensor schema\n"
        "    --gemma4-kernel-test               run kernel unit-tests (no model needed)\n"
        "    --gemma4-layer-test [IL]           hand vs ggml oracle for layer IL (default 0)\n"
        "    --gemma4-layer-test-ntok N         tokens for layer-test (default 8)\n"
        "    --gemma4-network-test [PROMPT]     hand vs upstream last-token logits\n"
        "    --gemma4-network-gen [PROMPT] [N]  greedy decode hand vs upstream (N tokens)\n"
        "    --gemma4-network-profile [PROMPT] [N_DECODE]\n"
        "                                       per-stage timing for prefill + N_DECODE decode steps\n"
        "    --gemma4-save-kv PATH [PROMPT] [N] prefill, save KV state to PATH, continue greedy gen N tokens\n"
        "    --gemma4-load-kv PATH [PROMPT] [N] skip prefill, load KV from PATH, greedy gen N tokens\n"
        "    --gemma4-load-kv-strict            promote weight_hash mismatch on load to fatal error\n"
            "    --gemma4-chat                      run hand-path chat (interactive if no -p, else single turn)\n"
            "    --gemma4-chat-test                 scripted multi-turn determinism test (greedy)\n"
            "    --gemma4-chat-ctx N                NetworkState capacity for chat (default 4096)\n"
            "    --temp F                           sampling temperature (default 0.0 = greedy)\n"
            "    --min-p F                          min-p sampler cutoff (default 0.05; ignored when --temp 0)\n"
            "    --seed N                           sampler seed (default LLAMA_DEFAULT_SEED)\n"
            "    --gemma4-tokenize-probe N S1 ... SN  tokenize N strings (parse_special=true) and exit\n"
            "    --gemma4-bench                     llama-bench-style pp{N}/tg{N} t/s table (qquant + upstream)\n"
            "    --bench-pp N                       bench prefill size (default 64)\n"
            "    --bench-tg N                       bench gen size (default 64)\n"
            "    --bench-reps N                     measured repeats per test (default 3; warmup is implicit)\n"
            "    --bench-threads N                  thread count for both backends (default: --threads-gen)\n"
            "    --bench-backend qquant|upstream|both\n"
            "                                       which backends to bench (default both)\n"
            "    --gemma4-attn-parallel 0|1         dispatch per-head attention across the matmul threadpool (default 1)\n"
            "\n",
        argv[0]);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string single_prompt = "Why is the sky blue?";
    int  n_predict          = 64;
    int  n_ctx              = 0;   // 0 = auto-size based on n_prompt + n_predict
    int  ngl                = 99;
    int  n_threads_prefill  = 0;   // 0 = let llama pick
    int  n_threads_gen      = 0;
    bool dump_weights       = false;  // G3.1: resolve + print schema, skip decode
    bool kernel_test        = false;  // G3.2: run kernel self-tests, skip everything else
    bool layer_test         = false;  // G3.3: hand-coded layer vs ggml oracle
    int  layer_test_il      = 0;
    int  layer_test_ntok    = 8;
    bool network_test       = false;  // G3.4a: full-network hand vs upstream
    std::string network_test_prompt = "The capital of France is";
    bool network_gen_test   = false;  // G3.4b: greedy decode with KV cache vs upstream
    std::string network_gen_prompt  = "The capital of France is";
    int  network_gen_n      = 32;
    bool network_profile    = false;  // profile prefill + N decode steps
    std::string profile_prompt = "The capital of France is";
    int  profile_n_decode   = 4;
    // G4.3: cached prefill
    std::string save_kv_path;          // --gemma4-save-kv PATH
    std::string load_kv_path;          // --gemma4-load-kv PATH
    bool        load_kv_strict = false;// --gemma4-load-kv-strict
    std::string cached_prompt = "The capital of France is";
    int         cached_n_gen  = 32;
    // G4.4: chat loop with sampling
    bool        chat_mode    = false;  // --gemma4-chat
    bool        chat_test    = false;  // --gemma4-chat-test
    int         chat_ctx     = 4096;   // --gemma4-chat-ctx N
    float       cli_temp     = 0.0f;   // --temp F  (0 = greedy)
    float       cli_min_p    = 0.05f;  // --min-p F
    uint32_t    cli_seed     = LLAMA_DEFAULT_SEED;  // --seed N

    // Debug helper: tokenize one or more strings and print id + piece.
    // Useful for documenting / reproducing BPE behaviour around the
    // chat-template special tokens. Pass --gemma4-tokenize-probe N
    // followed by N strings.
    int                       probe_count = 0;        // 0 = disabled
    std::vector<std::string>  probe_strings;

    // --gemma4-bench (llama-bench-style throughput table).
    bool bench_mode             = false;
    int  bench_pp_n             = 64;
    int  bench_tg_n             = 64;
    int  bench_reps             = 3;
    int  bench_threads          = 0;       // 0 = inherit from --threads-gen/--threads-prefill
    std::string bench_backend   = "both";  // qquant | upstream | both

    // G5.1 - parallel per-head attention. Default ON for the qquant path.
    // Pass --gemma4-attn-parallel 0 to fall back to the original serial
    // per-head loop (used for A/B comparison).
    int gemma4_attn_parallel    = 1;

    for (int i = 1; i < argc; ++i) {
        try {
            if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
                model_path = argv[++i];
            } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
                single_prompt = argv[++i];
            } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_predict = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
                n_ctx = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
                ngl = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--threads-prefill") == 0 && i + 1 < argc) {
                n_threads_prefill = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--threads-gen") == 0 && i + 1 < argc) {
                n_threads_gen = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--gemma4-dump-weights") == 0) {
                dump_weights = true;
            } else if (std::strcmp(argv[i], "--gemma4-kernel-test") == 0) {
                kernel_test = true;
            } else if (std::strcmp(argv[i], "--gemma4-layer-test") == 0) {
                layer_test = true;
                // Optional next-arg: layer index. Default 0.
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    layer_test_il = std::stoi(argv[++i]);
                }
            } else if (std::strcmp(argv[i], "--gemma4-layer-test-ntok") == 0 && i + 1 < argc) {
                layer_test_ntok = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--gemma4-network-test") == 0) {
                network_test = true;
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    network_test_prompt = argv[++i];
                }
            } else if (std::strcmp(argv[i], "--gemma4-network-gen") == 0) {
                network_gen_test = true;
                // Optional next-arg: prompt (if not starting with '-').
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    network_gen_prompt = argv[++i];
                }
                // Optional next-arg: N (if numeric).
                if (i + 1 < argc && argv[i+1][0] != '-' &&
                    (argv[i+1][0] >= '0' && argv[i+1][0] <= '9')) {
                    network_gen_n = std::stoi(argv[++i]);
                }
            } else if (std::strcmp(argv[i], "--gemma4-network-profile") == 0) {
                network_profile = true;
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    profile_prompt = argv[++i];
                }
                if (i + 1 < argc && argv[i+1][0] != '-' &&
                    (argv[i+1][0] >= '0' && argv[i+1][0] <= '9')) {
                    profile_n_decode = std::stoi(argv[++i]);
                }
            } else if (std::strcmp(argv[i], "--gemma4-save-kv") == 0 && i + 1 < argc) {
                save_kv_path = argv[++i];
                // Optional next-arg: prompt (if not starting with '-').
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    cached_prompt = argv[++i];
                }
                // Optional next-arg: N (if numeric).
                if (i + 1 < argc && argv[i+1][0] != '-' &&
                    (argv[i+1][0] >= '0' && argv[i+1][0] <= '9')) {
                    cached_n_gen = std::stoi(argv[++i]);
                }
            } else if (std::strcmp(argv[i], "--gemma4-load-kv") == 0 && i + 1 < argc) {
                load_kv_path = argv[++i];
                // Optional next-arg: prompt (advisory hash check only).
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    cached_prompt = argv[++i];
                }
                // Optional next-arg: N (if numeric).
                if (i + 1 < argc && argv[i+1][0] != '-' &&
                    (argv[i+1][0] >= '0' && argv[i+1][0] <= '9')) {
                    cached_n_gen = std::stoi(argv[++i]);
                }
            } else if (std::strcmp(argv[i], "--gemma4-load-kv-strict") == 0) {
                load_kv_strict = true;
            } else if (std::strcmp(argv[i], "--gemma4-chat") == 0) {
                chat_mode = true;
            } else if (std::strcmp(argv[i], "--gemma4-chat-test") == 0) {
                chat_test = true;
            } else if (std::strcmp(argv[i], "--gemma4-chat-ctx") == 0 && i + 1 < argc) {
                chat_ctx = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
                cli_temp = std::stof(argv[++i]);
            } else if (std::strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
                cli_min_p = std::stof(argv[++i]);
            } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
                cli_seed = (uint32_t) std::stoul(argv[++i]);
            } else if (std::strcmp(argv[i], "--gemma4-tokenize-probe") == 0 && i + 1 < argc) {
                probe_count = std::stoi(argv[++i]);
                for (int k = 0; k < probe_count && i + 1 < argc; ++k) {
                    probe_strings.emplace_back(argv[++i]);
                }
                if ((int) probe_strings.size() != probe_count) {
                    std::fprintf(stderr,
                        "error: --gemma4-tokenize-probe expects N strings after the count\n");
                    return 1;
                }
            } else if (std::strcmp(argv[i], "--gemma4-bench") == 0) {
                bench_mode = true;
            } else if (std::strcmp(argv[i], "--bench-pp") == 0 && i + 1 < argc) {
                bench_pp_n = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--bench-tg") == 0 && i + 1 < argc) {
                bench_tg_n = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--bench-reps") == 0 && i + 1 < argc) {
                bench_reps = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--bench-threads") == 0 && i + 1 < argc) {
                bench_threads = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--bench-backend") == 0 && i + 1 < argc) {
                bench_backend = argv[++i];
                if (bench_backend != "qquant" && bench_backend != "upstream" && bench_backend != "both") {
                    std::fprintf(stderr,
                        "error: --bench-backend must be one of qquant|upstream|both\n");
                    return 1;
                }
            } else if (std::strcmp(argv[i], "--gemma4-attn-parallel") == 0 && i + 1 < argc) {
                gemma4_attn_parallel = std::stoi(argv[++i]);
                if (gemma4_attn_parallel != 0 && gemma4_attn_parallel != 1) {
                    std::fprintf(stderr,
                        "error: --gemma4-attn-parallel expects 0 or 1\n");
                    return 1;
                }
            } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
                print_usage(argc, argv);
                return 0;
            } else {
                std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
                print_usage(argc, argv);
                return 1;
            }
        } catch (const std::exception & e) {
            std::fprintf(stderr, "error parsing %s: %s\n", argv[i], e.what());
            return 1;
        }
    }
    if (!kernel_test && model_path.empty()) { print_usage(argc, argv); return 1; }
    if (!save_kv_path.empty() && !load_kv_path.empty()) {
        std::fprintf(stderr,
            "error: --gemma4-save-kv and --gemma4-load-kv are mutually exclusive\n");
        return 1;
    }
    if (chat_mode && chat_test) {
        std::fprintf(stderr,
            "error: --gemma4-chat and --gemma4-chat-test are mutually exclusive\n");
        return 1;
    }

    // G5.1 - publish the attn-parallel toggle before any decode path runs.
    // Read by gemma4::get_attn_parallel() inside layer_forward_f32_cached.
    gemma4::set_attn_parallel(gemma4_attn_parallel != 0);

    // Quiet llama log noise: surface errors only.
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) std::fprintf(stderr, "%s", text);
    }, nullptr);

    ggml_backend_load_all();

    // ---------- G3.2: --gemma4-kernel-test ----------
    // Self-tests are model-independent; run before loading anything.
    if (kernel_test) {
        std::string kerr;
        if (!gemma4::kernel_self_test(kerr)) {
            std::fprintf(stderr, "gemma4 kernel self-test: FAIL: %s\n", kerr.c_str());
            return 1;
        }
        return 0;
    }

    // ---------- Load model ----------
    Gemma4LoadParams lp;
    lp.model_path   = model_path;
    lp.n_gpu_layers = ngl;

    Gemma4RawModel raw;
    std::string err;
    if (!gemma4_load_raw_model(lp, raw, err)) {
        std::fprintf(stderr, "gemma4: %s\n", err.c_str());
        return 1;
    }
    gemma4_log_summary(raw);
    if (raw.is_moe) {
        std::fprintf(stderr,
                     "gemma4: WARNING this model has MoE tensors "
                     "(ffn_gate_inp.* found); G1 baseline runs through the "
                     "upstream llama graph and will still work, but our "
                     "custom-forward work in this directory targets the "
                     "dense variants (E2B/E4B) first.\n");
    }

    // ---------- Tokenize-probe (G4.4 doc helper) ----------
    // Prints token ids + readable pieces for one or more strings using
    // parse_special=true and add_special=true. Designed to surface BPE
    // boundary effects around chat-template special tokens; e.g.
    //   "Paris."        vs.
    //   "Paris.<end_of_turn>"
    // typically tokenize "Paris" + "." identically, but a multi-piece
    // assistant response decoded one token at a time and then
    // re-tokenized in the middle of a longer string can split
    // differently. Useful as the live oracle for
    // examples/gemma4/__bpe_chat_pitfall__.md.
    if (probe_count > 0) {
        const llama_vocab * vocab = llama_model_get_vocab(raw.model);
        for (int pi = 0; pi < probe_count; ++pi) {
            const std::string & s = probe_strings[pi];
            std::vector<llama_token> ids(s.size() + 16);
            int n = llama_tokenize(vocab, s.data(), (int) s.size(),
                                   ids.data(), (int) ids.size(),
                                   /*add_special=*/true,
                                   /*parse_special=*/true);
            if (n < 0) {
                ids.resize((size_t) (-n + 16));
                n = llama_tokenize(vocab, s.data(), (int) s.size(),
                                   ids.data(), (int) ids.size(),
                                   true, true);
            }
            if (n < 0) {
                std::fprintf(stderr, "tokenize failed for #%d\n", pi);
                continue;
            }
            ids.resize((size_t) n);
            std::printf("probe[%d] in  = %s\n", pi, s.c_str());
            std::printf("probe[%d] out = %d tokens: [", pi, n);
            for (int k = 0; k < n; ++k) {
                if (k) std::printf(", ");
                std::printf("%d", (int) ids[k]);
            }
            std::printf("]\n");
            for (int k = 0; k < n; ++k) {
                char buf[256] = {0};
                int m = llama_token_to_piece(vocab, ids[k], buf,
                                             (int) sizeof(buf) - 1, 0, true);
                std::printf("  [%2d] %6d -> \"%s\"\n",
                            k, (int) ids[k], m > 0 ? buf : "");
            }
        }
        llama_model_free(raw.model);
        return 0;
    }

    // ---------- G3.1: --gemma4-dump-weights ----------
    // Resolve every tensor we need for the custom forward, validate
    // shapes/types, and print the schema. Exits before running decode.
    if (dump_weights) {
        gemma4::Weights w;
        std::string werr;
        if (!gemma4::resolve(raw.model, w, werr)) {
            std::fprintf(stderr, "gemma4: weights resolve FAIL: %s\n", werr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        gemma4::dump(w);
        gemma4_unload_raw_model(raw);
        return 0;
    }

    // ---------- G3.3: --gemma4-layer-test [IL] [--gemma4-layer-test-ntok N] ----
    // Run hand-coded single-layer F32 forward vs ggml-graph oracle on the
    // specified layer; both consume dequantized F32 weights so the only
    // numeric drift is op order. Skip baseline decode on success.
    if (layer_test) {
        gemma4::Weights w;
        std::string werr;
        if (!gemma4::resolve(raw.model, w, werr)) {
            std::fprintf(stderr, "gemma4: weights resolve FAIL: %s\n", werr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::string terr;
        const bool ok = gemma4::layer_self_test(raw.model, w, layer_test_il,
                                                layer_test_ntok, terr);
        if (!ok) {
            std::fprintf(stderr, "gemma4 layer_self_test: FAIL: %s\n", terr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::fprintf(stderr, "gemma4 layer_self_test: PASS (il=%d n_tokens=%d)\n",
                     layer_test_il, layer_test_ntok);
        gemma4_unload_raw_model(raw);
        return 0;
    }

    // ---------- G3.4a: --gemma4-network-test [PROMPT] -----------------
    // Run hand-coded full-network F32 forward over the prompt; compare
    // last-token logits against upstream llama_decode. Multi-metric.
    if (network_test) {
        gemma4::Weights w;
        std::string werr;
        if (!gemma4::resolve(raw.model, w, werr)) {
            std::fprintf(stderr, "gemma4: weights resolve FAIL: %s\n", werr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::string terr;
        const int n_threads = n_threads_prefill > 0 ? n_threads_prefill : 4;
        const bool ok = gemma4::network_self_test(raw.model, w, network_test_prompt,
                                                  n_threads, terr);
        if (!ok) {
            std::fprintf(stderr, "gemma4 network_self_test: FAIL: %s\n", terr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::fprintf(stderr, "gemma4 network_self_test: PASS\n");
        gemma4_unload_raw_model(raw);
        return 0;
    }

    // ---------- G3.4b: --gemma4-network-gen [PROMPT] [N] ---------------
    // Greedy decode with persistent KV cache; compare hand-path token
    // sequence against upstream llama_decode greedy.
    if (network_gen_test) {
        gemma4::Weights w;
        std::string werr;
        if (!gemma4::resolve(raw.model, w, werr)) {
            std::fprintf(stderr, "gemma4: weights resolve FAIL: %s\n", werr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::string terr;
        const int n_threads = n_threads_prefill > 0 ? n_threads_prefill : 4;
        const bool ok = gemma4::network_gen_self_test(raw.model, w,
                                                      network_gen_prompt,
                                                      network_gen_n,
                                                      n_threads, terr);
        if (!ok) {
            std::fprintf(stderr, "gemma4 network_gen_self_test: FAIL: %s\n", terr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::fprintf(stderr, "gemma4 network_gen_self_test: PASS\n");
        gemma4_unload_raw_model(raw);
        return 0;
    }

    // ---------- Profiling: --gemma4-network-profile [PROMPT] [N_DECODE] ----
    // Run prefill + N_DECODE decode steps with per-stage timing turned on.
    // Prints two breakdowns (prefill, decode) for hot-spot identification.
    if (network_profile) {
        gemma4::Weights w;
        std::string werr;
        if (!gemma4::resolve(raw.model, w, werr)) {
            std::fprintf(stderr, "gemma4: weights resolve FAIL: %s\n", werr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::string terr;
        const int n_threads = n_threads_prefill > 0 ? n_threads_prefill : 4;
        const bool ok = gemma4::network_profile(raw.model, w, profile_prompt,
                                                profile_n_decode, n_threads, terr);
        if (!ok) {
            std::fprintf(stderr, "gemma4 network_profile: FAIL: %s\n", terr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::fprintf(stderr, "gemma4 network_profile: DONE\n");
        gemma4_unload_raw_model(raw);
        return 0;
    }

    // ---------- --gemma4-bench (llama-bench-style table) ----------------
    // Runs pp{N}/tg{N} tests through the qquant hand path and/or the
    // upstream llama_decode path with R measured reps each (warmup is
    // implicit). Prints a markdown table matching tools/llama-bench's
    // output format so qquant numbers can be compared side-by-side
    // against `llama-bench.exe` runs on the same model.
    if (bench_mode) {
        gemma4::Weights w;
        std::string werr;
        if (!gemma4::resolve(raw.model, w, werr)) {
            std::fprintf(stderr, "gemma4 bench: weights resolve FAIL: %s\n", werr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        gemma4::BenchParams bp;
        bp.pp_n             = bench_pp_n;
        bp.tg_n             = bench_tg_n;
        bp.reps             = bench_reps;
        bp.n_threads        = bench_threads > 0 ? bench_threads
                              : (n_threads_gen > 0 ? n_threads_gen
                              : (n_threads_prefill > 0 ? n_threads_prefill : 4));
        bp.include_qquant   = (bench_backend == "qquant"   || bench_backend == "both");
        bp.include_upstream = (bench_backend == "upstream" || bench_backend == "both");
        bp.seed             = cli_seed != LLAMA_DEFAULT_SEED ? cli_seed : 1234u;

        std::string berr;
        const bool ok = gemma4::run_bench(raw.model, w, bp, berr);
        if (!ok) {
            std::fprintf(stderr, "gemma4 bench: FAIL: %s\n", berr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        gemma4_unload_raw_model(raw);
        return 0;
    }

    // ---------- G4.3: --gemma4-save-kv PATH [PROMPT] [N] ----------------
    // Run prefill on PROMPT, save NetworkState+first_gen_token to PATH,
    // then continue greedy gen for N total tokens (hand-only; no
    // upstream comparison). The on-disk cache is exactly what a fresh
    // --gemma4-load-kv run would consume.
    if (!save_kv_path.empty()) {
        gemma4::Weights w;
        std::string werr;
        if (!gemma4::resolve(raw.model, w, werr)) {
            std::fprintf(stderr, "gemma4: weights resolve FAIL: %s\n", werr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::string terr;
        const int n_threads = n_threads_gen > 0 ? n_threads_gen
                              : (n_threads_prefill > 0 ? n_threads_prefill : 4);
        const bool ok = gemma4::network_gen_save_kv(raw.model, w, cached_prompt,
                                                    cached_n_gen, n_threads,
                                                    save_kv_path, terr);
        if (!ok) {
            std::fprintf(stderr, "gemma4 save-kv: FAIL: %s\n", terr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::fprintf(stderr, "gemma4 save-kv: PASS\n");
        gemma4_unload_raw_model(raw);
        return 0;
    }

    // ---------- G4.3: --gemma4-load-kv PATH [PROMPT] [N] ----------------
    // Skip prefill entirely; load NetworkState from PATH and continue
    // greedy gen for N total tokens (first token comes from the cache).
    // PROMPT is optional and only used for an advisory prompt-hash
    // mismatch warning.
    if (!load_kv_path.empty()) {
        gemma4::Weights w;
        std::string werr;
        if (!gemma4::resolve(raw.model, w, werr)) {
            std::fprintf(stderr, "gemma4: weights resolve FAIL: %s\n", werr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::string terr;
        const int n_threads = n_threads_gen > 0 ? n_threads_gen
                              : (n_threads_prefill > 0 ? n_threads_prefill : 4);
        const bool ok = gemma4::network_gen_load_kv(raw.model, w, cached_prompt,
                                                    cached_n_gen, n_threads,
                                                    load_kv_path, load_kv_strict, terr);
        if (!ok) {
            std::fprintf(stderr, "gemma4 load-kv: FAIL: %s\n", terr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::fprintf(stderr, "gemma4 load-kv: PASS\n");
        gemma4_unload_raw_model(raw);
        return 0;
    }

    // ---------- G4.4: --gemma4-chat-test ----------------
    // Scripted 2-turn greedy determinism test: incremental delta+decode
    // vs fresh full-prefill of the same conversation. PASS iff
    // turn-2 token sequences match exactly.
    if (chat_test) {
        gemma4::Weights w;
        std::string werr;
        if (!gemma4::resolve(raw.model, w, werr)) {
            std::fprintf(stderr, "gemma4: weights resolve FAIL: %s\n", werr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::string terr;
        const int n_threads = n_threads_gen > 0 ? n_threads_gen
                              : (n_threads_prefill > 0 ? n_threads_prefill : 4);
        const bool ok = gemma4::chat_self_test(raw.model, w, n_threads, terr);
        if (!ok) {
            std::fprintf(stderr, "gemma4 chat_self_test: FAIL: %s\n", terr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        std::fprintf(stderr, "gemma4 chat_self_test: PASS\n");
        gemma4_unload_raw_model(raw);
        return 0;
    }

    // ---------- G4.4: --gemma4-chat ----------------------
    // Hand-path chat runtime. If -p was provided, runs one turn with
    // that prompt and exits. Otherwise reads user messages from stdin
    // until EOF or empty line. Supports greedy (--temp 0, default) and
    // sampled (--temp >0 with --min-p) decoding via llama_sampler_chain.
    if (chat_mode) {
        gemma4::Weights w;
        std::string werr;
        if (!gemma4::resolve(raw.model, w, werr)) {
            std::fprintf(stderr, "gemma4: weights resolve FAIL: %s\n", werr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        gemma4::ChatParams cp;
        cp.temp      = cli_temp;
        cp.min_p     = cli_min_p;
        cp.seed      = cli_seed;
        cp.n_predict = n_predict;
        cp.chat_ctx  = chat_ctx;
        cp.n_threads = n_threads_gen > 0 ? n_threads_gen
                       : (n_threads_prefill > 0 ? n_threads_prefill : 4);
        cp.stream    = true;
        // Single-turn from -p iff -p was actually supplied (we can't
        // distinguish "user passed -p" from default — but the default
        // is non-empty so we honor it). To get interactive mode, pass
        // -p "".
        std::string terr;
        const bool ok = gemma4::run_chat_loop(raw.model, w, single_prompt, cp, terr);
        if (!ok) {
            std::fprintf(stderr, "gemma4 chat: FAIL: %s\n", terr.c_str());
            gemma4_unload_raw_model(raw);
            return 1;
        }
        gemma4_unload_raw_model(raw);
        return 0;
    }

    // ---------- Apply chat template ----------
    const char * chat_template = llama_model_chat_template(raw.model, nullptr);
    std::string fmt_prompt;
    if (chat_template) {
        std::vector<llama_chat_message> msgs;
        char * user_cstr = strdup(single_prompt.c_str());
        msgs.push_back({"user", user_cstr});
        std::vector<char> formatted(single_prompt.size() + 1024);
        int n_fmt = llama_chat_apply_template(chat_template, msgs.data(), msgs.size(),
                                              /*add_assistant=*/true,
                                              formatted.data(), (int32_t) formatted.size());
        if (n_fmt > (int) formatted.size()) {
            formatted.resize((size_t) n_fmt);
            n_fmt = llama_chat_apply_template(chat_template, msgs.data(), msgs.size(),
                                              true, formatted.data(), (int32_t) formatted.size());
        }
        std::free(user_cstr);
        if (n_fmt < 0) {
            std::fprintf(stderr, "gemma4: chat_apply_template failed; using raw prompt\n");
            fmt_prompt = single_prompt;
        } else {
            fmt_prompt.assign(formatted.begin(), formatted.begin() + n_fmt);
        }
    } else {
        std::fprintf(stderr, "gemma4: model has no chat template; using raw prompt\n");
        fmt_prompt = single_prompt;
    }

    // ---------- Tokenize ----------
    const int n_neg = -llama_tokenize(raw.vocab, fmt_prompt.c_str(), (int) fmt_prompt.size(),
                                      nullptr, 0,
                                      /*add_special=*/true,
                                      /*parse_special=*/true);
    if (n_neg <= 0) {
        std::fprintf(stderr, "gemma4: tokenize sizing failed (n_neg=%d)\n", n_neg);
        gemma4_unload_raw_model(raw);
        return 1;
    }
    std::vector<llama_token> prompt_tokens((size_t) n_neg);
    const int n_tok = llama_tokenize(raw.vocab, fmt_prompt.c_str(), (int) fmt_prompt.size(),
                                     prompt_tokens.data(), (int) prompt_tokens.size(),
                                     true, true);
    if (n_tok < 0) {
        std::fprintf(stderr, "gemma4: llama_tokenize failed\n");
        gemma4_unload_raw_model(raw);
        return 1;
    }
    prompt_tokens.resize((size_t) n_tok);

    std::fprintf(stderr, "gemma4: prompt_tokens=%d  predict=%d\n",
                 (int) prompt_tokens.size(), n_predict);

    // ---------- Baseline decode ----------
    (void) n_ctx;  // baseline auto-sizes; flag is kept for future symmetry
    std::vector<llama_token> gen_tokens;
    std::string derr;
    double t_pre_ms = 0.0, t_gen_ms = 0.0;
    if (!gemma4_run_baseline_decode(raw.model, prompt_tokens, n_predict,
                                    n_threads_prefill, n_threads_gen,
                                    gen_tokens, derr,
                                    &t_pre_ms, &t_gen_ms)) {
        std::fprintf(stderr, "gemma4: baseline-decode FAIL: %s\n", derr.c_str());
        gemma4_unload_raw_model(raw);
        return 1;
    }

    // ---------- Print generated text ----------
    std::fprintf(stdout, "\ngemma4: generated %d tokens for prompt \"%s\":\n",
                 (int) gen_tokens.size(), single_prompt.c_str());
    for (llama_token tok : gen_tokens) {
        char piece[256];
        const int n = llama_token_to_piece(raw.vocab, tok, piece, sizeof(piece), 0, true);
        if (n > 0) std::fwrite(piece, 1, (size_t) n, stdout);
    }
    std::fputc('\n', stdout);
    std::fflush(stdout);

    // ---------- Summary ----------
    const double prefill_tps = t_pre_ms > 0
                                   ? 1000.0 * (double) prompt_tokens.size() / t_pre_ms
                                   : 0.0;
    const double gen_tps     = t_gen_ms > 0
                                   ? 1000.0 * (double) gen_tokens.size() / t_gen_ms
                                   : 0.0;
    std::fprintf(stderr,
                 "\ngemma4: summary  prefill=%d tok in %.2f ms (%.1f tps)  "
                 "gen=%d tok in %.2f ms (%.1f tps)\n",
                 (int) prompt_tokens.size(), t_pre_ms, prefill_tps,
                 (int) gen_tokens.size(),   t_gen_ms, gen_tps);

    gemma4_unload_raw_model(raw);
    return 0;
}
