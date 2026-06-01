#pragma once

#include "llama.h"
#include "phi3_transform.h"

#include <string>

struct Phi3GenKernelState {
    llama_token token = 0;
    int32_t n_seq_id = 1;
    llama_seq_id seq0 = 0;
    llama_seq_id * seq_id_ptr = nullptr;
    int8_t logits = 1;
    llama_batch batch = {};
};

void phi3_gen_kernel_state_init(Phi3GenKernelState & state);

bool phi3_decode_prefill_step(
    llama_context * ctx,
    const llama_batch & batch,
    double & decode_dt_ms,
    std::string & error);

bool phi3_decode_generate_step(
    llama_context * ctx,
    const Phi3ExecutionPlan & plan,
    Phi3GenKernelState & state,
    llama_token token,
    double & decode_dt_ms,
    bool & fusion_eligible,
    const char *& kernel_tag,
    std::string & error);
