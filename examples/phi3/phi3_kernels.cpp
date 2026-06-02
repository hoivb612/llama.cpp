#include "phi3_kernels.h"

#include <chrono>
#include <limits>

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
    const llama_vocab * vocab,
    bool enable_fused_greedy,
    llama_token & sampled_token,
    double & decode_dt_ms,
    double & sample_dt_ms,
    bool & fusion_eligible,
    bool & qkv_v2_eligible,
    bool & used_fused_sampling,
    const char *& kernel_tag,
    std::string & error) {
    sampled_token = LLAMA_TOKEN_NULL;
    sample_dt_ms = 0.0;
    used_fused_sampling = false;
    fusion_eligible = plan.diagnostics.decode_fusion_candidate && plan.fuse_qkv && plan.fuse_mlp;
    qkv_v2_eligible = fusion_eligible && plan.diagnostics.decode_qkv_v2_candidate && plan.fuse_qkv_v2;

    // Reuse pre-wired 1-token batch state across generation steps to keep
    // per-step overhead minimal and isolate this hot path for future fused kernels.
    state.token = token;

    const bool use_fused_greedy = fusion_eligible && enable_fused_greedy;
    kernel_tag = use_fused_greedy
        ? (qkv_v2_eligible ? "phi3-gen1tok-fusedv2-qkv-shape" : "phi3-gen1tok-fusedv1-persist")
        :
        (fusion_eligible ? "phi3-gen1tok-persist(fused-candidate)" : "phi3-gen1tok-persist");
    if (!phi3_decode_with_llama(ctx, state.batch, decode_dt_ms, error)) {
        return false;
    }
    if (!use_fused_greedy) {
        return true;
    }

    if (!vocab) {
        error = "missing vocab for fused generation sampling";
        return false;
    }
    float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        error = "missing logits for fused generation sampling";
        return false;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        error = "invalid vocabulary size for fused generation sampling";
        return false;
    }

    const auto sample_start = std::chrono::steady_clock::now();
    int32_t best_id = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best_id = i;
        }
    }

    sample_dt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sample_start).count();
    sampled_token = best_id;
    used_fused_sampling = true;
    error.clear();
    return true;
}
