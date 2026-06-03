#include "llama.h"
#include "llama_b612.h"
#include "phi3_fused_graph.h"
#include "phi3_loader.h"
#include "phi3_runtime.h"
#include "phi3_transform.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m Phi-3-mini-4k-instruct.gguf [-p \"where is Paris\"] [-c 4096] [-ngl 99] [-n 256] [-s 1234] [--temp 0.0] [--min-p 0.05] [--threads-prefill 32] [--threads-gen 8] [--threads-gen-auto] [--phi3-fused-lmhead] [--phi3-fused-decode] [--phi3-dump-weights] [--phi3-test-kv] [--phi3-matmul-test] [--phi3-kernel-test] [--phi3-validate-fused] [--repack-ggml|--repack-xbox|--repack-xbcg]\n", argv[0]);
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

    phi3_runtime_free(runtime);
    phi3_unload_raw_model(raw_model);

    return ok ? 0 : 1;
}
