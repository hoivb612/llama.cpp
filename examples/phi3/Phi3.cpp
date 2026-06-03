#include "llama.h"
#include "llama_b612.h"
#include "phi3_fused_graph.h"
#include "phi3_loader.h"
#include "phi3_runtime.h"
#include "phi3_transform.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m Phi-3-mini-4k-instruct.gguf [-p \"where is Paris\"] [-c 4096] [-ngl 99] [-n 256] [-s 1234] [--temp 0.0] [--min-p 0.05] [--threads-prefill 32] [--threads-gen 8] [--threads-gen-auto] [--phi3-fused-lmhead] [--phi3-fused-decode] [--phi3-dump-weights] [--phi3-test-kv] [--phi3-matmul-test] [--phi3-kernel-test] [--phi3-validate-fused] [--phi3-layer-test] [--phi3-full-test] [--phi3-qmatmul-test] [--phi3-fused-f32-debug] [--phi3-fused-f32-n-gen N] [--phi3-fused-qquant-debug] [--phi3-fused-qquant-n-gen N] [--phi3-fused-qquant-threads N] [--repack-ggml|--repack-xbox|--repack-xbcg]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string single_prompt;
    int ngl = 99;
    int n_ctx = 4096;
    int n_predict = 256;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    float temp = 0.8f;
    float min_p = 0.05f;
    int n_threads_prefill = 0;
    int n_threads_gen = 0;
    bool enable_gen_autotune = false;
    bool enable_fused_lmhead = false;
    bool enable_fused_decode = false;
    bool dump_weights = false;
    bool test_kv = false;
    bool test_matmul = false;
    bool test_kernels = false;
    bool validate_fused = false;
    bool test_layer = false;
    bool test_full = false;
    bool test_qmatmul = false;
    bool fused_f32_debug = false;
    int  fused_f32_n_gen = 4;
    bool fused_qquant_debug = false;
    int  fused_qquant_n_gen = 4;
    int  fused_qquant_threads = 0;   // 0 => use n_threads_gen
    ggml_tensor_repack_mode_t tensor_repack_mode = GGML_TENSOR_REPACK_MODE_NONE;

    for (int i = 1; i < argc; i++) {
        try {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-p") == 0) {
                if (i + 1 < argc) {
                    single_prompt = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-c") == 0) {
                if (i + 1 < argc) {
                    n_ctx = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    ngl = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    n_predict = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-s") == 0) {
                if (i + 1 < argc) {
                    seed = (uint32_t) std::stoul(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--temp") == 0) {
                if (i + 1 < argc) {
                    temp = std::stof(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--min-p") == 0) {
                if (i + 1 < argc) {
                    min_p = std::stof(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--threads-prefill") == 0) {
                if (i + 1 < argc) {
                    n_threads_prefill = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--threads-gen") == 0) {
                if (i + 1 < argc) {
                    n_threads_gen = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--threads-gen-auto") == 0) {
                enable_gen_autotune = true;
            } else if (strcmp(argv[i], "--phi3-fused-lmhead") == 0) {
                enable_fused_lmhead = true;
            } else if (strcmp(argv[i], "--phi3-fused-decode") == 0) {
                enable_fused_decode = true;
            } else if (strcmp(argv[i], "--phi3-dump-weights") == 0) {
                dump_weights = true;
            } else if (strcmp(argv[i], "--phi3-test-kv") == 0) {
                test_kv = true;
            } else if (strcmp(argv[i], "--phi3-matmul-test") == 0) {
                test_matmul = true;
            } else if (strcmp(argv[i], "--phi3-kernel-test") == 0) {
                test_kernels = true;
            } else if (strcmp(argv[i], "--phi3-validate-fused") == 0) {
                validate_fused = true;
            } else if (strcmp(argv[i], "--phi3-layer-test") == 0) {
                test_layer = true;
            } else if (strcmp(argv[i], "--phi3-full-test") == 0) {
                test_full = true;
            } else if (strcmp(argv[i], "--phi3-qmatmul-test") == 0) {
                test_qmatmul = true;
            } else if (strcmp(argv[i], "--phi3-fused-f32-debug") == 0) {
                fused_f32_debug = true;
            } else if (strcmp(argv[i], "--phi3-fused-f32-n-gen") == 0) {
                if (i + 1 < argc) {
                    fused_f32_n_gen = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--phi3-fused-qquant-debug") == 0) {
                fused_qquant_debug = true;
            } else if (strcmp(argv[i], "--phi3-fused-qquant-n-gen") == 0) {
                if (i + 1 < argc) {
                    fused_qquant_n_gen = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--phi3-fused-qquant-threads") == 0) {
                if (i + 1 < argc) {
                    fused_qquant_threads = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--repack-ggml") == 0) {
                tensor_repack_mode = GGML_TENSOR_REPACK_MODE_GGML;
            } else if (strcmp(argv[i], "--repack-xbox") == 0) {
                tensor_repack_mode = GGML_TENSOR_REPACK_MODE_XBOX;
            } else if (strcmp(argv[i], "--repack-xbcg") == 0) {
                tensor_repack_mode = GGML_TENSOR_REPACK_MODE_XBCG;
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            print_usage(argc, argv);
            return 1;
        }
    }

    if (model_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    ggml_backend_load_all();
    llama_set_tensor_repack_mode(tensor_repack_mode);

    Phi3LoadParams load_params;
    load_params.model_path = model_path;
    load_params.n_gpu_layers = ngl;

    Phi3RawModel raw_model;
    std::string error;
    if (!phi3_load_raw_model(load_params, raw_model, error)) {
        fprintf(stderr, "%s: error: %s\n", __func__, error.c_str());
        return 1;
    }
    Phi3ExecutionPlan plan;
    if (!phi3_build_execution_plan(raw_model, plan, error)) {
        fprintf(stderr, "%s: error: %s\n", __func__, error.c_str());
        phi3_unload_raw_model(raw_model);
        return 1;
    }
    fprintf(stderr, "phi3 plan: layers=%d embd=%d qkv_fuse=%d qkv_v2_fuse=%d mlp_fuse=%d\n",
        plan.n_layer, plan.n_embd, (int) plan.fuse_qkv, (int) plan.fuse_qkv_v2, (int) plan.fuse_mlp);
    fprintf(stderr, "phi3 transform: %s\n", plan.diagnostics.summary.c_str());

    if (dump_weights) {
        Phi3Weights w;
        std::string werr;
        if (!phi3_weights_resolve(raw_model.model, w, werr)) {
            fprintf(stderr, "phi3 weights: resolve failed: %s\n", werr.c_str());
            phi3_unload_raw_model(raw_model);
            return 1;
        }
        phi3_weights_dump(w);
    }

    if (test_kv) {
        Phi3Weights w;
        std::string werr;
        if (!phi3_weights_resolve(raw_model.model, w, werr)) {
            fprintf(stderr, "phi3 weights: resolve failed: %s\n", werr.c_str());
            phi3_unload_raw_model(raw_model);
            return 1;
        }
        std::string kerr;
        if (!phi3_kv_self_test(w, kerr)) {
            fprintf(stderr, "phi3 kv self-test: FAIL: %s\n", kerr.c_str());
            phi3_unload_raw_model(raw_model);
            return 1;
        }
    }

    if (test_matmul) {
        std::string merr;
        const int test_threads = n_threads_gen > 0 ? n_threads_gen : 4;
        if (!phi3_matmul_pool_self_test(test_threads, merr)) {
            fprintf(stderr, "phi3 matmul self-test: FAIL: %s\n", merr.c_str());
            phi3_unload_raw_model(raw_model);
            return 1;
        }
    }

    if (test_kernels) {
        std::string kerr;
        if (!phi3_kernel_self_test(kerr)) {
            fprintf(stderr, "phi3 kernel self-test: FAIL: %s\n", kerr.c_str());
            phi3_unload_raw_model(raw_model);
            return 1;
        }
    }

    if (test_layer) {
        std::string lerr;
        if (!phi3_layer_self_test(raw_model.model, lerr)) {
            fprintf(stderr, "phi3 layer self-test: FAIL: %s\n", lerr.c_str());
            phi3_unload_raw_model(raw_model);
            return 1;
        }
        phi3_unload_raw_model(raw_model);
        return 0;
    }

    if (test_full) {
        std::string ferr;
        if (!phi3_full_self_test(raw_model.model, ferr)) {
            fprintf(stderr, "phi3 full self-test: FAIL: %s\n", ferr.c_str());
            phi3_unload_raw_model(raw_model);
            return 1;
        }
        phi3_unload_raw_model(raw_model);
        return 0;
    }

    if (test_qmatmul) {
        std::string ferr;
        if (!phi3_qmatmul_self_test(raw_model.model, ferr)) {
            fprintf(stderr, "phi3 qmatmul self-test: FAIL: %s\n", ferr.c_str());
            phi3_unload_raw_model(raw_model);
            return 1;
        }
        phi3_unload_raw_model(raw_model);
        return 0;
    }

    Phi3RuntimeParams runtime_params;
    runtime_params.n_ctx = n_ctx;
    runtime_params.n_predict = n_predict;
    runtime_params.seed = seed;
    runtime_params.temp = temp;
    runtime_params.min_p = min_p;
    runtime_params.n_threads_prefill = n_threads_prefill;
    runtime_params.n_threads_gen = n_threads_gen;
    runtime_params.enable_gen_autotune = enable_gen_autotune;
    runtime_params.enable_fused_lmhead = enable_fused_lmhead;
    runtime_params.enable_fused_decode = enable_fused_decode;
    // The Phase A custom forward path requires flash_attn=false (the validator
    // enforces this). When --phi3-validate-fused is set, force-disable flash
    // so the validator can actually demonstrate the ACCEPT path.
    runtime_params.disable_flash_attn  = validate_fused;
    if (enable_fused_lmhead && (temp > 0.0f || min_p > 0.0f)) {
        fprintf(stderr, "%s: --phi3-fused-lmhead requires --temp 0 --min-p 0 (greedy decoding)\n", __func__);
        phi3_unload_raw_model(raw_model);
        return 1;
    }
    Phi3Runtime runtime;
    if (!phi3_runtime_init(raw_model, plan, runtime_params, runtime, error)) {
        fprintf(stderr, "%s: error: %s\n", __func__, error.c_str());
        phi3_unload_raw_model(raw_model);
        return 1;
    }
    fprintf(stderr, "phi3 runtime: prefill_threads=%d gen_threads=%d n_predict=%d seed=%u temp=%.3f min_p=%.3f fused_greedy_gen=%d fused_lmhead=%d fused_decode=%d gen_autotune=%d\n",
        runtime.n_threads_prefill, runtime.n_threads_gen, runtime.n_predict, runtime.seed, runtime_params.temp, runtime_params.min_p, (int) runtime.enable_fused_greedy_gen, (int) runtime.enable_fused_lmhead, (int) runtime.enable_fused_decode, (int) runtime.enable_gen_autotune);

    {
        // A2.1b smoke check: the Phase A custom forward fast-paths require no
        // active LoRA and no active control vectors. Probe early so a misuse
        // is reported once, here, instead of buried inside the decode loop.
        const bool has_lora = llama_b612_has_active_lora(runtime.ctx);
        const bool has_cvec = llama_b612_has_active_cvec(runtime.ctx);
        fprintf(stderr, "phi3 adapters: active_lora=%d active_cvec=%d\n",
            (int) has_lora, (int) has_cvec);
    }

    if (validate_fused) {
        Phi3Weights w;
        std::string werr;
        if (!phi3_weights_resolve(raw_model.model, w, werr)) {
            fprintf(stderr, "phi3 validate-fused: weights resolve failed: %s\n", werr.c_str());
            phi3_runtime_free(runtime);
            phi3_unload_raw_model(raw_model);
            return 1;
        }
        std::string verr;
        const bool vok = phi3_fused_validate_supported(raw_model.model, w, runtime.ctx, verr);
        if (!vok) {
            fprintf(stderr, "phi3 validate-fused: REJECT: %s\n", verr.c_str());
            phi3_runtime_free(runtime);
            phi3_unload_raw_model(raw_model);
            return 1;
        }
        fprintf(stderr, "phi3 validate-fused: ACCEPT (all 24 feature checks passed)\n");

        // Also exercise ctx_init/free with f32_debug=false to confirm the
        // alloc/free path works on the live model.
        Phi3FusedCtx cx;
        std::string cerr;
        const bool cok = phi3_fused_ctx_init(cx, w, /*matmul_pool=*/nullptr,
            raw_model.model, runtime.ctx, n_ctx, /*f32_debug=*/false, cerr);
        if (!cok) {
            fprintf(stderr, "phi3 validate-fused: ctx_init FAIL: %s\n", cerr.c_str());
            phi3_runtime_free(runtime);
            phi3_unload_raw_model(raw_model);
            return 1;
        }
        phi3_fused_ctx_dump(cx);
        phi3_fused_ctx_free(cx);
        fprintf(stderr, "phi3 validate-fused: ctx_init/free OK\n");

        phi3_runtime_free(runtime);
        phi3_unload_raw_model(raw_model);
        return 0;
    }

    bool ok = true;
    if (!single_prompt.empty()) {
        ok = phi3_runtime_run_single_prompt(runtime, single_prompt, error);
    } else {
        ok = phi3_runtime_run_interactive(runtime, error);
    }
    if (!ok && !error.empty()) {
        fprintf(stderr, "%s: error: %s\n", __func__, error.c_str());
    }

    // ----- A2.4b: F32-everywhere decode spot-check (optional) -----
    // Runs AFTER the baseline so the user sees baseline output first, then a
    // separate "phi3 f32-decode" section with the hand-path tokens for
    // visual comparison. Greedy-only path; baseline must also be greedy
    // (temp=0) for the side-by-side to be apples-to-apples.
    if (ok && fused_f32_debug && !single_prompt.empty()) {
        if (temp != 0.0f) {
            fprintf(stderr,
                "\nphi3 f32-debug: WARNING --temp=%g is non-zero. Baseline is sampling,\n"
                "F32 hand path is GREEDY. Side-by-side comparison is informational only.\n", temp);
        }
        fprintf(stderr, "\nphi3 f32-debug: extremely slow path (~6-7 s/tok). NOT for CI.\n");

        // Apply the SAME Phi-3 chat template that the baseline runtime used.
        // We re-derive it stateless (fresh messages vector) to avoid touching
        // runtime state which has already been advanced by the turn above.
        const char * chat_template = llama_model_chat_template(raw_model.model, nullptr);
        if (chat_template == nullptr) {
            fprintf(stderr, "phi3 f32-debug: model has no chat template; aborting f32 debug path\n");
        } else {
            std::vector<llama_chat_message> msgs;
            char * user_cstr = strdup(single_prompt.c_str());
            msgs.push_back({"user", user_cstr});
            std::vector<char> formatted(8192);
            int n_fmt = llama_chat_apply_template(chat_template, msgs.data(), msgs.size(),
                                                  /*add_ass=*/true, formatted.data(), formatted.size());
            if (n_fmt > (int) formatted.size()) {
                formatted.resize((size_t) n_fmt);
                n_fmt = llama_chat_apply_template(chat_template, msgs.data(), msgs.size(),
                                                  true, formatted.data(), formatted.size());
            }
            if (n_fmt < 0) {
                fprintf(stderr, "phi3 f32-debug: chat_apply_template failed; aborting\n");
            } else {
                const std::string fmt_prompt(formatted.begin(), formatted.begin() + n_fmt);

                const llama_vocab * vocab = llama_model_get_vocab(raw_model.model);
                const int n_neg = -llama_tokenize(vocab, fmt_prompt.c_str(), (int) fmt_prompt.size(),
                                                  nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
                if (n_neg <= 0) {
                    fprintf(stderr, "phi3 f32-debug: tokenize sizing failed (n_neg=%d)\n", n_neg);
                } else {
                    std::vector<llama_token> prompt_tokens((size_t) n_neg);
                    const int n_tok = llama_tokenize(vocab, fmt_prompt.c_str(), (int) fmt_prompt.size(),
                                                     prompt_tokens.data(), (int) prompt_tokens.size(),
                                                     /*add_special=*/true, /*parse_special=*/true);
                    if (n_tok < 0) {
                        fprintf(stderr, "phi3 f32-debug: llama_tokenize failed\n");
                    } else {
                        prompt_tokens.resize((size_t) n_tok);

                        std::vector<llama_token> gen_tokens;
                        std::string ferr;
                        const bool fok = phi3_run_f32_decode(raw_model.model, prompt_tokens,
                                                             fused_f32_n_gen, gen_tokens, ferr);
                        if (!fok) {
                            fprintf(stderr, "phi3 f32-debug: FAIL: %s\n", ferr.c_str());
                        } else {
                            // Detokenize and print side-by-side info.
                            fprintf(stderr, "\nphi3 f32-decode: n_prompt_tokens=%d  n_gen=%d\n",
                                    (int) prompt_tokens.size(), (int) gen_tokens.size());
                            fprintf(stderr, "phi3 f32-decode: prompt tokens (last 8): ");
                            const int show_n = std::min((int) prompt_tokens.size(), 8);
                            for (int i = (int) prompt_tokens.size() - show_n; i < (int) prompt_tokens.size(); ++i) {
                                char buf[64];
                                const int n = llama_token_to_piece(vocab, prompt_tokens[i], buf, sizeof(buf), 0, true);
                                fprintf(stderr, "[%d:%.*s] ", prompt_tokens[i], n > 0 ? n : 0, buf);
                            }
                            fprintf(stderr, "\n");

                            std::string gen_text;
                            fprintf(stderr, "phi3 f32-decode: generated tokens: ");
                            for (auto t : gen_tokens) {
                                char buf[64];
                                const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
                                fprintf(stderr, "[%d:%.*s] ", t, n > 0 ? n : 0, buf);
                                if (n > 0) gen_text.append(buf, (size_t) n);
                            }
                            fprintf(stderr, "\n");
                            fprintf(stderr, "phi3 f32-decode: generated text: \"%s\"\n", gen_text.c_str());
                        }
                    }
                }
            }
            free(user_cstr);
        }
    }

    // ----- A2.5a.1: Q-quant decode spot-check (optional) -----
    // Same structure as the F32 debug block: re-apply chat template fresh,
    // tokenize, call phi3_run_qquant_decode, print side-by-side. Uses the
    // ORIGINAL quantized weights via q_matmul_row + the existing fused
    // q-quant lm_head argmax. Much faster than the F32 debug path.
    if (ok && fused_qquant_debug && !single_prompt.empty()) {
        if (temp != 0.0f) {
            fprintf(stderr,
                "\nphi3 qquant-debug: WARNING --temp=%g is non-zero. Baseline is sampling,\n"
                "qquant hand path is GREEDY. Side-by-side comparison is informational only.\n", temp);
        }

        const char * chat_template = llama_model_chat_template(raw_model.model, nullptr);
        if (chat_template == nullptr) {
            fprintf(stderr, "phi3 qquant-debug: model has no chat template; aborting\n");
        } else {
            std::vector<llama_chat_message> msgs;
            char * user_cstr = strdup(single_prompt.c_str());
            msgs.push_back({"user", user_cstr});
            std::vector<char> formatted(8192);
            int n_fmt = llama_chat_apply_template(chat_template, msgs.data(), msgs.size(),
                                                  /*add_ass=*/true, formatted.data(), formatted.size());
            if (n_fmt > (int) formatted.size()) {
                formatted.resize((size_t) n_fmt);
                n_fmt = llama_chat_apply_template(chat_template, msgs.data(), msgs.size(),
                                                  true, formatted.data(), formatted.size());
            }
            if (n_fmt < 0) {
                fprintf(stderr, "phi3 qquant-debug: chat_apply_template failed; aborting\n");
            } else {
                const std::string fmt_prompt(formatted.begin(), formatted.begin() + n_fmt);
                const llama_vocab * vocab = llama_model_get_vocab(raw_model.model);
                const int n_neg = -llama_tokenize(vocab, fmt_prompt.c_str(), (int) fmt_prompt.size(),
                                                  nullptr, 0, true, true);
                if (n_neg <= 0) {
                    fprintf(stderr, "phi3 qquant-debug: tokenize sizing failed (n_neg=%d)\n", n_neg);
                } else {
                    std::vector<llama_token> prompt_tokens((size_t) n_neg);
                    const int n_tok = llama_tokenize(vocab, fmt_prompt.c_str(), (int) fmt_prompt.size(),
                                                     prompt_tokens.data(), (int) prompt_tokens.size(),
                                                     true, true);
                    if (n_tok < 0) {
                        fprintf(stderr, "phi3 qquant-debug: llama_tokenize failed\n");
                    } else {
                        prompt_tokens.resize((size_t) n_tok);
                        std::vector<llama_token> gen_tokens;
                        std::string qerr;
                        int qq_threads = fused_qquant_threads > 0 ? fused_qquant_threads : n_threads_gen;
                        if (qq_threads <= 0) qq_threads = 1;
                        const bool qok = phi3_run_qquant_decode(raw_model.model, prompt_tokens,
                                                                fused_qquant_n_gen, qq_threads,
                                                                gen_tokens, qerr);
                        if (!qok) {
                            fprintf(stderr, "phi3 qquant-debug: FAIL: %s\n", qerr.c_str());
                        } else {
                            fprintf(stderr, "\nphi3 qquant-decode: n_prompt_tokens=%d  n_gen=%d\n",
                                    (int) prompt_tokens.size(), (int) gen_tokens.size());
                            fprintf(stderr, "phi3 qquant-decode: prompt tokens (last 8): ");
                            const int show_n = std::min((int) prompt_tokens.size(), 8);
                            for (int i = (int) prompt_tokens.size() - show_n; i < (int) prompt_tokens.size(); ++i) {
                                char buf[64];
                                const int n = llama_token_to_piece(vocab, prompt_tokens[i], buf, sizeof(buf), 0, true);
                                fprintf(stderr, "[%d:%.*s] ", prompt_tokens[i], n > 0 ? n : 0, buf);
                            }
                            fprintf(stderr, "\n");

                            std::string gen_text;
                            fprintf(stderr, "phi3 qquant-decode: generated tokens: ");
                            for (auto t : gen_tokens) {
                                char buf[64];
                                const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
                                fprintf(stderr, "[%d:%.*s] ", t, n > 0 ? n : 0, buf);
                                if (n > 0) gen_text.append(buf, (size_t) n);
                            }
                            fprintf(stderr, "\n");
                            fprintf(stderr, "phi3 qquant-decode: generated text: \"%s\"\n", gen_text.c_str());
                        }
                    }
                }
            }
            free(user_cstr);
        }
    }

    phi3_runtime_free(runtime);
    phi3_unload_raw_model(raw_model);

    return ok ? 0 : 1;
}
