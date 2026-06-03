#include "models.h"

#include "ggml-cpu.h"

#include <atomic>
#include <cstring>

namespace {

// =============================================================================
// Phi3 Option 1 — fused RMSNorm + quantize-to-Q8_K (in-graph)
// =============================================================================
// Eliminates the f32 intermediate buffer between RMSNorm and the downstream
// mul_mat, AND skips the mul_mat's internal `from_float` step (ggml-cpu's
// mul_mat fast-path uses `src1->data` directly when `src1->type ==
// vec_dot_type(src0->type)`). Output is a Q8_K tensor with the same
// [n_embd, n_tokens, 1, 1] shape that the standard `build_norm` would
// produce in f32, so the downstream `ggml_mul_mat(W_q4k, normed_q8K)` is
// semantically equivalent — it just skips the wdata round-trip.
//
// Parallelism (matters because ggml_custom_op_t has no barrier/work-buffer
// access — see ggml.h:2636): each thread independently computes the row's
// sum-of-squares (redundant ~3 us of cached FMA per thread) then partitions
// the Q8_K blocks across threads. This lets us parallelize the from_float
// step the same way the standard mul_mat path does, at the cost of a tiny
// amount of redundant reduction work.

struct phi3_fused_norm_q_params {
    float eps;
};

// Static arena for userdata. The ggml graph stores a raw void* into the op
// params (see ggml.c:6114), so the pointed-to bytes must outlive the cached
// graph. Sized to two fusion sites x 32 layers x several rebuilds before
// graph reuse kicks in. For long sessions with many distinct ubatches this
// could be exceeded — we'd then need per-context storage, but this is
// sufficient for the experimental measurement run.
constexpr int kPhi3FusedNormArenaSize = 512;
phi3_fused_norm_q_params      g_phi3_fused_norm_arena[kPhi3FusedNormArenaSize];
std::atomic<int>              g_phi3_fused_norm_arena_idx{0};

phi3_fused_norm_q_params * phi3_alloc_norm_q_params(float eps) {
    const int idx = g_phi3_fused_norm_arena_idx.fetch_add(1, std::memory_order_relaxed);
    GGML_ASSERT(idx < kPhi3FusedNormArenaSize && "phi3 fused-norm arena exhausted");
    auto * slot = &g_phi3_fused_norm_arena[idx];
    slot->eps = eps;
    return slot;
}

void phi3_fused_norm_q8K_kernel(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_K);
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[1] != nullptr && dst->src[1]->type == GGML_TYPE_F32);

    const struct ggml_tensor * src   = dst->src[0]; // [n_embd, n_tokens, ne2, ne3] f32
    const struct ggml_tensor * gamma = dst->src[1]; // [n_embd] f32 (norm weight)
    const auto * p = (const phi3_fused_norm_q_params *) userdata;
    const float eps = p ? p->eps : 1e-5f;

    const int64_t n_embd   = src->ne[0];
    const int64_t n_tokens = src->ne[1] * src->ne[2] * src->ne[3];
    const int64_t blck_q8K = (int64_t) ggml_blck_size(GGML_TYPE_Q8_K); // 256

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(n_embd % blck_q8K == 0 && "n_embd must be a multiple of QK_K (256) for Q8_K fusion");

    const int64_t n_blocks = n_embd / blck_q8K;

    const ggml_from_float_t from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    GGML_ASSERT(from_float != nullptr);

    const float * gamma_data = (const float *) gamma->data;

    // Block partition for this thread: blocks [blk_lo, blk_hi) across nth threads.
    // Use a balanced split so threads don't all collide on the same blocks.
    const int64_t blk_lo = (n_blocks * (int64_t) ith) / nth;
    const int64_t blk_hi = (n_blocks * (int64_t) (ith + 1)) / nth;
    const int64_t blk_n  = blk_hi - blk_lo;

    if (blk_n <= 0) {
        return;
    }

    // Thread-local staging buffer for the (normalized * gamma) f32 values
    // before quantization. Max size = full row n_embd * sizeof(float) ≈ 12 KB
    // for n_embd=3072. Fits comfortably in L1.
    constexpr int64_t kMaxStagingFloats = 16384; // 64 KB safety cap
    GGML_ASSERT(blk_n * blck_q8K <= kMaxStagingFloats);
    float staging[kMaxStagingFloats];

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * x = (const float *) ((const char *) src->data + t * src->nb[1]);
        char *        d = (char *) dst->data + t * dst->nb[1];

        // Redundant per-thread sum-of-squares across the full row. Matches the
        // standard ggml_compute_forward_rms_norm_f32 which accumulates in
        // double precision (ggml_float == double, see vec.h:15) — needed for
        // bit-exact output parity. ~3 us at n_embd=3072 in L1.
        double sum_sq = 0.0;
        for (int64_t i = 0; i < n_embd; ++i) {
            sum_sq += (double)(x[i] * x[i]);
        }
        const float mean  = (float) (sum_sq / (double) n_embd);
        const float scale = 1.0f / sqrtf(mean + eps);

        // Apply normalization * gamma to this thread's block range.
        for (int64_t b = 0; b < blk_n; ++b) {
            const int64_t base = (blk_lo + b) * blck_q8K;
            for (int64_t i = 0; i < blck_q8K; ++i) {
                staging[b * blck_q8K + i] = x[base + i] * scale * gamma_data[base + i];
            }
        }

        // Quantize this thread's block slice directly into the destination row.
        const size_t blk_bytes = ggml_row_size(GGML_TYPE_Q8_K, blck_q8K);
        from_float(staging, d + blk_lo * blk_bytes, blk_n * blck_q8K);
    }
}

// Builds the fused-norm-quantize node. Returns a Q8_K tensor that can be
// fed directly to `ggml_mul_mat(W_q4k, ...)` — the mul_mat fast-path
// recognizes the matching vec_dot_type and skips its internal from_float.
ggml_tensor * phi3_build_fused_norm_q8K(
        ggml_context * ctx0,
        ggml_tensor *  src,    // f32 [n_embd, n_tokens, ne2, ne3]
        ggml_tensor *  gamma,  // f32 [n_embd]
        float          eps) {
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(gamma->type == GGML_TYPE_F32);

    auto * params = phi3_alloc_norm_q_params(eps);

    ggml_tensor * args[2] = { src, gamma };
    return ggml_custom_4d(
        ctx0,
        GGML_TYPE_Q8_K,
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        args, /*n_args=*/2,
        phi3_fused_norm_q8K_kernel,
        GGML_N_TASKS_MAX,
        params);
}

// Returns true iff the weight tensor accepts Q8_K as its mul_mat src1 input
// (i.e. its on-CPU vec_dot_type is Q8_K). True for K-quants (Q2_K..Q6_K).
bool phi3_weight_accepts_q8K(const ggml_tensor * w) {
    if (w == nullptr) return false;
    const auto * traits = ggml_get_type_traits_cpu(w->type);
    return traits != nullptr && traits->vec_dot_type == GGML_TYPE_Q8_K;
}

} // namespace


void llama_model_phi3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_1B; break;
        case 32: type = LLM_TYPE_3B; break;
        case 40: type = LLM_TYPE_14B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);

    if (found_swa && hparams.n_swa > 0) {
        LLAMA_LOG_WARN("%s: Phi SWA is currently disabled - results might be suboptimal for some models (see %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13676");

        // TODO: fix conversion scripts to correctly populate `n_swa` and `n_swa_pattern`
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;

        hparams.n_swa         = 0;
        hparams.set_swa_pattern(1);
    }
}

void llama_model_phi3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, TENSOR_NOT_REQUIRED);
        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, 2 * n_ff }, 0);

        layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), { n_rot/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_rot/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
    }
}

std::unique_ptr<llm_graph_context> llama_model_phi3::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return std::make_unique<graph<true>> (*this, params);
    } else {
        return std::make_unique<graph<false>>(*this, params);
    }
}

template<bool iswa>
llama_model_phi3::graph<iswa>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>;
    inp_attn_type * inp_attn = nullptr;

    if constexpr (iswa) {
        inp_attn = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        auto * residual = inpL;

        // self-attention
        {
            // rope freq factors for 128k context
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            ggml_tensor * attn_norm_output;
            const bool fuse_attn_norm = cparams.fused_decode_phi3 &&
                model.layers[il].attn_norm_b == nullptr &&
                loras->empty() &&
                phi3_weight_accepts_q8K(model.layers[il].wqkv);
            if (fuse_attn_norm) {
                // Option 1: produce a Q8_K normed tensor directly. Downstream
                // ggml_mul_mat(wqkv, attn_norm_output) will skip from_float.
                attn_norm_output = phi3_build_fused_norm_q8K(
                    ctx0, inpL, model.layers[il].attn_norm, hparams.f_norm_rms_eps);
                cb(attn_norm_output, "attn_norm_q8K", il);
            } else {
                attn_norm_output = build_norm(inpL,
                        model.layers[il].attn_norm,
                        model.layers[il].attn_norm_b,
                        LLM_NORM_RMS, il);
                cb(attn_norm_output, "attn_norm", il);
            }

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], attn_norm_output,
                    n_embd_head, n_head, n_head_kv, il);
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
            cb(Qcur, "Qcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }
        cur = ggml_add(ctx0, cur, residual);
        residual = cur;

        const bool fuse_ffn_norm = cparams.fused_decode_phi3 &&
            model.layers[il].ffn_norm_b == nullptr &&
            model.layers[il].ffn_gate_inp == nullptr &&  // non-MoE only
            loras->empty() &&
            phi3_weight_accepts_q8K(model.layers[il].ffn_up);
        if (fuse_ffn_norm) {
            cur = phi3_build_fused_norm_q8K(
                ctx0, cur, model.layers[il].ffn_norm, hparams.f_norm_rms_eps);
            cb(cur, "ffn_norm_q8K", il);
        } else {
            cur = build_norm(cur,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
        }

        // feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, residual, cur);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = build_norm(inpL,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (cparams.fused_lmhead) {
        // Phase C: skip lm_head; pin t_embd into the forward graph so the final
        // RMSNorm survives pruning, and clear t_logits so the lm_head matmul is
        // unreachable (pruned by ggml_build_forward_expand). The runtime computes
        // argmax directly from the post-norm hidden state.
        ggml_build_forward_expand(gf, cur);
        res->t_logits = nullptr;
        return;
    }

    cur = build_lora_mm(model.output, cur, model.output_s);

    if (model.output_b != nullptr) {
        cb(cur, "result_output_no_bias", -1);
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Explicit template instantiations
template struct llama_model_phi3::graph<false>;
template struct llama_model_phi3::graph<true>;
