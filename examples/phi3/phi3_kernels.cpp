#include "phi3_kernels.h"

#include <chrono>

static bool phi3_decode_with_llama(
    llama_context * ctx,
    const llama_batch & batch,
    double & decode_dt_ms,
    std::string & error) {
    const auto decode_start = std::chrono::steady_clock::now();
    const int ret = llama_decode(ctx, batch);
    decode_dt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_start).count();
    if (ret != 0) {
        error = "failed to decode, ret = " + std::to_string(ret);
        return false;
    }

    error.clear();
    return true;
}

void phi3_gen_kernel_state_init(Phi3GenKernelState & state) {
    state.token = 0;
    state.n_seq_id = 1;
    state.seq0 = 0;
    state.seq_id_ptr = &state.seq0;
    state.logits = 1;
    state.batch = {
        /*n_tokens=*/1,
        /*token=*/&state.token,
        /*embd=*/nullptr,
        /*pos=*/nullptr,
        /*n_seq_id=*/&state.n_seq_id,
        /*seq_id=*/&state.seq_id_ptr,
        /*logits=*/&state.logits,
    };
}

bool phi3_decode_prefill_step(
    llama_context * ctx,
    const llama_batch & batch,
    double & decode_dt_ms,
    std::string & error) {
    return phi3_decode_with_llama(ctx, batch, decode_dt_ms, error);
}

bool phi3_decode_generate_step(
    llama_context * ctx,
    const Phi3ExecutionPlan & plan,
    Phi3GenKernelState & state,
    llama_token token,
    double & decode_dt_ms,
    bool & fusion_eligible,
    const char *& kernel_tag,
    std::string & error) {
    fusion_eligible = plan.diagnostics.decode_fusion_candidate && plan.fuse_qkv && plan.fuse_mlp;

    // Reuse pre-wired 1-token batch state across generation steps to keep
    // per-step overhead minimal and isolate this hot path for future fused kernels.
    state.token = token;

    kernel_tag = fusion_eligible ? "phi3-gen1tok-persist(fused-candidate)" : "phi3-gen1tok-persist";
    return phi3_decode_with_llama(ctx, state.batch, decode_dt_ms, error);
}
