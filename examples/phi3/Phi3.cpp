#include "llama.h"
#include "phi3_loader.h"
#include "phi3_runtime.h"
#include "phi3_transform.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m Phi-3-mini-4k-instruct.gguf [-p \"where is Paris\"] [-c 4096] [-ngl 99] [-n 256] [-s 1234] [--temp 0.0] [--min-p 0.05] [--threads-prefill 32] [--threads-gen 8] [--threads-gen-auto] [--repack-ggml|--repack-xbox|--repack-xbcg]\n", argv[0]);
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
    fprintf(stderr, "phi3 plan: layers=%d embd=%d qkv_fuse=%d mlp_fuse=%d\n",
        plan.n_layer, plan.n_embd, (int) plan.fuse_qkv, (int) plan.fuse_mlp);
    fprintf(stderr, "phi3 transform: %s\n", plan.diagnostics.summary.c_str());

    Phi3RuntimeParams runtime_params;
    runtime_params.n_ctx = n_ctx;
    runtime_params.n_predict = n_predict;
    runtime_params.seed = seed;
    runtime_params.temp = temp;
    runtime_params.min_p = min_p;
    runtime_params.n_threads_prefill = n_threads_prefill;
    runtime_params.n_threads_gen = n_threads_gen;
    runtime_params.enable_gen_autotune = enable_gen_autotune;
    Phi3Runtime runtime;
    if (!phi3_runtime_init(raw_model, plan, runtime_params, runtime, error)) {
        fprintf(stderr, "%s: error: %s\n", __func__, error.c_str());
        phi3_unload_raw_model(raw_model);
        return 1;
    }
    fprintf(stderr, "phi3 runtime: prefill_threads=%d gen_threads=%d n_predict=%d seed=%u temp=%.3f min_p=%.3f fused_greedy_gen=%d gen_autotune=%d\n",
        runtime.n_threads_prefill, runtime.n_threads_gen, runtime.n_predict, runtime.seed, runtime_params.temp, runtime_params.min_p, (int) runtime.enable_fused_greedy_gen, (int) runtime.enable_gen_autotune);

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
