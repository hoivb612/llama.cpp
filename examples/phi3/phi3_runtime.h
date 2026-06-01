#pragma once

#include "phi3_loader.h"
#include "phi3_kernels.h"
#include "phi3_transform.h"

#include <string>
#include <vector>

struct Phi3RuntimeParams {
    int n_ctx = 4096;
    int n_predict = 256;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    float min_p = 0.05f;
    float temp = 0.8f;
    bool enable_instrumentation = true;
    int n_threads_prefill = 0;
    int n_threads_gen = 0;
    bool enable_gen_autotune = false;
};

struct Phi3Runtime {
    llama_context * ctx = nullptr;
    llama_sampler * smpl = nullptr;
    const llama_vocab * vocab = nullptr;
    Phi3ExecutionPlan plan = {};
    const char * chat_template = nullptr;
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted;
    int prev_len = 0;
    int turn_index = 0;
    bool enable_instrumentation = true;
    int n_predict = 256;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    int n_threads_prefill = 0;
    int n_threads_gen = 0;
    int active_threads = 0;
    bool enable_gen_autotune = false;
    bool gen_autotune_locked = false;
    int gen_autotune_steps_per_candidate = 12;
    int gen_autotune_eval_steps = 0;
    std::vector<int> gen_autotune_candidates;
    std::vector<double> gen_autotune_decode_ms_sum;
    std::vector<int> gen_autotune_decode_ms_count;
    bool enable_fused_greedy_gen = false;
    Phi3GenKernelState gen_kernel_state = {};
};

bool phi3_runtime_init(
    const Phi3RawModel & raw,
    const Phi3ExecutionPlan & plan,
    const Phi3RuntimeParams & params,
    Phi3Runtime & out,
    std::string & error);

bool phi3_runtime_run_single_prompt(Phi3Runtime & runtime, const std::string & prompt, std::string & error);
bool phi3_runtime_run_interactive(Phi3Runtime & runtime, std::string & error);
void phi3_runtime_free(Phi3Runtime & runtime);
