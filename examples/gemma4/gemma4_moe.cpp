// Gemma-4 MoE support -- per-expert view alignment self-test.
//
// See gemma4_moe.h. This gates matmul_expert_qf32 (the per-expert 2D view
// primitive) by proving its output is bit-identical to a contiguous copy of
// the same expert block, for both K-quant (gate_up, Q4_K) and legacy-quant
// (down, Q8_0/Q5_0) expert tensors.

#include "gemma4_moe.h"
#include "gemma4_weights.h"
#include "gemma4_matmul.h"
#include "gemma4_kernels.h"
#include "gemma4_expert_store.h"
#include "gemma4_route_stats.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

namespace gemma4 {

namespace {

// Softmax over v[0..n) in place (numerically stable).
void softmax_inplace(float * v, int n) {
    float m = v[0];
    for (int i = 1; i < n; ++i) m = std::max(m, v[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { v[i] = std::exp(v[i] - m); sum += v[i]; }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

// Indices of the top-k largest values of v[0..n), written to idx[0..k).
// Matches ggml_argsort_top_k selection (descending by value); order within
// the k is irrelevant here since the caller only sums the selected experts.
void top_k_indices(const float * v, int n, int k, int * idx) {
    for (int j = 0; j < k; ++j) idx[j] = j;
    // partial selection sort of the first k by value
    // (k is small -- 8 -- so O(n*k) is fine)
    std::vector<char> used(n, 0);
    for (int j = 0; j < k; ++j) {
        int best = -1;
        float bestv = -INFINITY;
        for (int i = 0; i < n; ++i) {
            if (used[i]) continue;
            if (v[i] > bestv) { bestv = v[i]; best = i; }
        }
        idx[j] = best;
        used[best] = 1;
    }
}

} // namespace

bool moe_ffn(MatmulCtx & mm, const MoeInputs & in,
             const float * attn_out2, float * out,
             int n_new, float eps, std::string & error) {
    const int n_embd        = in.n_embd;
    const int n_ff          = in.n_ff;          // shared/dense FFN width (2112)
    const int n_ff_exp      = in.n_ff_exp;      // per-expert FFN width (704)
    const int n_expert      = in.n_expert;      // 128
    const int n_expert_used = in.n_expert_used; // 8

    const float * ffn_norm        = in.ffn_norm;
    const float * post_ffw_norm_1 = static_cast<const float *>(in.ffn_post_norm_1->data);
    const float * ffn_pre_norm_2  = static_cast<const float *>(in.ffn_pre_norm_2->data);
    const float * post_ffw_norm_2 = static_cast<const float *>(in.ffn_post_norm_2->data);
    const float * gate_inp_s      = static_cast<const float *>(in.ffn_gate_inp_s->data);
    const float * down_exps_s     = in.ffn_down_exps_s
                                    ? static_cast<const float *>(in.ffn_down_exps_s->data) : nullptr;

    // ===================== shared MLP path =====================
    // ff_in = rms_norm(attn_out2) * ffn_norm
    std::vector<float> ff_in((size_t) n_embd * n_new);
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_mul_f32(ff_in.data() + (size_t) t * n_embd,
                        attn_out2 + (size_t) t * n_embd, ffn_norm, n_embd, eps);
    }

    std::vector<float> gate((size_t) n_ff * n_new);
    std::vector<float> up  ((size_t) n_ff * n_new);
    if (!matmul_qf32(mm, in.ffn_gate, ff_in.data(), gate.data(),
                     n_embd, n_ff, n_new, error)) return false;
    if (!matmul_qf32(mm, in.ffn_up, ff_in.data(), up.data(),
                     n_embd, n_ff, n_new, error)) return false;
    gelu_f32(gate.data(), gate.data(), n_ff * n_new);
    for (size_t i = 0; i < (size_t) n_ff * n_new; ++i) gate[i] *= up[i];

    std::vector<float> mlp_down((size_t) n_embd * n_new);
    if (!matmul_qf32(mm, in.ffn_down, gate.data(), mlp_down.data(),
                     n_ff, n_embd, n_new, error)) return false;

    // cur_mlp = rms_norm(mlp_down) * post_ffw_norm_1  (write into out)
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_mul_f32(out + (size_t) t * n_embd,
                        mlp_down.data() + (size_t) t * n_embd,
                        post_ffw_norm_1, n_embd, eps);
    }

    // ===================== router =====================
    // tmp = rms_norm(attn_out2) * (1/sqrt(n_embd)) * gate_inp_s
    std::vector<float> tmp((size_t) n_embd * n_new);
    const float inv_sqrt = 1.0f / std::sqrt((float) n_embd);
    for (int t = 0; t < n_new; ++t) {
        float * dst = tmp.data() + (size_t) t * n_embd;
        rmsnorm_mul_f32(dst, attn_out2 + (size_t) t * n_embd, nullptr, n_embd, eps);
        for (int i = 0; i < n_embd; ++i) dst[i] *= inv_sqrt * gate_inp_s[i];
    }
    std::vector<float> logits((size_t) n_expert * n_new);
    if (!matmul_qf32(mm, in.ffn_gate_inp, tmp.data(), logits.data(),
                     n_embd, n_expert, n_new, error)) return false;

    // ===================== expert path =====================
    // moe_in = rms_norm(attn_out2) * ffn_pre_norm_2
    std::vector<float> moe_in((size_t) n_embd * n_new);
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_mul_f32(moe_in.data() + (size_t) t * n_embd,
                        attn_out2 + (size_t) t * n_embd, ffn_pre_norm_2, n_embd, eps);
    }

    std::vector<float> moe_out((size_t) n_embd * n_new, 0.0f);

    // P1: when an ExpertStore is installed, stream each expert block from
    // disk (hard-capped LRU) instead of viewing the resident mmap tensor.
    ExpertStore * store = get_expert_store();
    const bool streaming = (store && store->ready());

    // G6.1: resident fast path -- fuse all experts + tokens through
    // ggml_mul_mat_id (mirrors upstream llm_graph_context::build_moe_ffn),
    // one threadpool dispatch per projection instead of the per-expert,
    // per-token GEMV loop below. The streaming ExpertStore path keeps the
    // per-block loop (Phase 2). Per-expert gate/up scales are unsupported by
    // the fused path; Gemma-4 Q4_K has none, but guard just in case.
    if (get_moe_fused() && !streaming) {
        std::vector<int32_t> ids((size_t) n_expert_used * n_new);
        std::vector<float>   wts((size_t) n_expert_used * n_new);
        std::vector<float>   rprobs(n_expert);
        std::vector<int>     rsel(n_expert_used);
        for (int t = 0; t < n_new; ++t) {
            std::memcpy(rprobs.data(), logits.data() + (size_t) t * n_expert,
                        n_expert * sizeof(float));
            softmax_inplace(rprobs.data(), n_expert);
            top_k_indices(rprobs.data(), n_expert, n_expert_used, rsel.data());
            route_stats_record(in.il, n_expert, rsel.data(), n_expert_used, n_new == 1);
            float wsum = 0.0f;
            for (int j = 0; j < n_expert_used; ++j) wsum += rprobs[rsel[j]];
            wsum = std::max(wsum, 6.103515625e-5f); // ggml clamp
            for (int j = 0; j < n_expert_used; ++j) {
                float wv = rprobs[rsel[j]] / wsum;
                if (down_exps_s) wv *= down_exps_s[rsel[j]];
                ids[(size_t) t * n_expert_used + j] = rsel[j];
                wts[(size_t) t * n_expert_used + j] = wv;
            }
        }
        if (!matmul_moe_id_qf32(mm, in.ffn_gate_up_exps, in.ffn_gate_exps,
                                in.ffn_up_exps, in.ffn_down_exps,
                                moe_in.data(), ids.data(), wts.data(),
                                moe_out.data(), n_embd, n_ff_exp,
                                n_expert_used, n_new, error)) {
            return false;
        }
    } else {

    std::vector<float> probs(n_expert);
    std::vector<int>   sel(n_expert_used);
    std::vector<float> gu((size_t) n_ff_exp * 2);  // per-expert merged gate_up
    std::vector<float> h(n_ff_exp);                // geglu activation
    std::vector<float> d(n_embd);                  // per-expert down output

    const bool merged = (in.ffn_gate_up_exps != nullptr);
    std::vector<float> gsep(merged ? 0 : (size_t) n_ff_exp); // separate gate buffer

    auto expert_mm = [&](const ggml_tensor * W3d, int e, const float * x,
                         float * y, int n_in, int n_out) -> bool {
        if (streaming && store->rec(W3d)) {
            const int w_type = store->rec(W3d)->type;
            const void * blk = store->fetch(W3d, e, error);
            if (!blk) return false;
            return matmul_qblock_qf32(mm, w_type, blk, x, y, n_in, n_out, 1, error);
        }
        return matmul_expert_qf32(mm, W3d, e, x, y, n_in, n_out, 1, error);
    };

    // P2 within-layer prefetch: when the background worker is enabled, warm a
    // token's full expert working set (in usage order) as soon as the router
    // selects it, overlapping the preads with the gate_up/geglu/down compute.
    // Only engage when the whole working set fits the budget; otherwise the
    // prefetched blocks would be evicted before use (the I/O-bound regime),
    // so we leave the plain synchronous path.
    const bool prefetching = streaming && store->prefetch_enabled();
    bool prefetch_fits = false;
    if (prefetching) {
        size_t per_expert = 0;
        if (merged) {
            const ExpertTensorRec * ru = store->rec(in.ffn_gate_up_exps);
            const ExpertTensorRec * rd = store->rec(in.ffn_down_exps);
            if (ru && rd) per_expert = (size_t) ru->block_bytes + (size_t) rd->block_bytes;
        } else {
            const ExpertTensorRec * rg = store->rec(in.ffn_gate_exps);
            const ExpertTensorRec * ru = store->rec(in.ffn_up_exps);
            const ExpertTensorRec * rd = store->rec(in.ffn_down_exps);
            if (rg && ru && rd) {
                per_expert = (size_t) rg->block_bytes + (size_t) ru->block_bytes
                           + (size_t) rd->block_bytes;
            }
        }
        prefetch_fits = per_expert > 0 &&
                        per_expert * (size_t) n_expert_used <= store->budget_bytes();
    }
    std::vector<ExpertStore::KeyPair> pf_keys;
    if (prefetch_fits) pf_keys.reserve((size_t) n_expert_used * (merged ? 2 : 3));

    for (int t = 0; t < n_new; ++t) {
        // softmax over all experts, then top-k select + weight renormalize
        std::memcpy(probs.data(), logits.data() + (size_t) t * n_expert,
                    n_expert * sizeof(float));
        softmax_inplace(probs.data(), n_expert);
        top_k_indices(probs.data(), n_expert, n_expert_used, sel.data());
        route_stats_record(in.il, n_expert, sel.data(), n_expert_used, n_new == 1);

        float wsum = 0.0f;
        std::vector<float> wsel(n_expert_used);
        for (int j = 0; j < n_expert_used; ++j) { wsel[j] = probs[sel[j]]; wsum += wsel[j]; }
        wsum = std::max(wsum, 6.103515625e-5f); // ggml clamp
        for (int j = 0; j < n_expert_used; ++j) wsel[j] /= wsum;

        // Enqueue this token's expert blocks in usage order for the worker.
        if (prefetch_fits) {
            pf_keys.clear();
            for (int j = 0; j < n_expert_used; ++j) {
                const int e = sel[j];
                if (merged) {
                    pf_keys.emplace_back(in.ffn_gate_up_exps, e);
                } else {
                    pf_keys.emplace_back(in.ffn_gate_exps, e);
                    pf_keys.emplace_back(in.ffn_up_exps, e);
                }
                pf_keys.emplace_back(in.ffn_down_exps, e);
            }
            store->prefetch(pf_keys);
        }

        const float * x = moe_in.data() + (size_t) t * n_embd;
        for (int j = 0; j < n_expert_used; ++j) {
            const int e = sel[j];

            // gate_up (merged) or separate gate/up -> geglu activation h
            if (merged) {
                if (!expert_mm(in.ffn_gate_up_exps, e, x, gu.data(),
                               n_embd, n_ff_exp * 2)) return false;
                // gate = gu[0:n_ff_exp], up = gu[n_ff_exp:2*n_ff_exp]
                gelu_f32(h.data(), gu.data(), n_ff_exp);
                for (int i = 0; i < n_ff_exp; ++i) h[i] *= gu[(size_t) n_ff_exp + i];
            } else {
                if (!expert_mm(in.ffn_gate_exps, e, x, gsep.data(),
                               n_embd, n_ff_exp)) return false;
                if (!expert_mm(in.ffn_up_exps, e, x, gu.data(),
                               n_embd, n_ff_exp)) return false;
                gelu_f32(h.data(), gsep.data(), n_ff_exp);
                for (int i = 0; i < n_ff_exp; ++i) h[i] *= gu[i];
            }

            // down projection
            if (!expert_mm(in.ffn_down_exps, e, h.data(), d.data(),
                           n_ff_exp, n_embd)) return false;

            float scale = wsel[j];
            if (down_exps_s) scale *= down_exps_s[e];

            float * acc = moe_out.data() + (size_t) t * n_embd;
            for (int i = 0; i < n_embd; ++i) acc[i] += scale * d[i];
        }
    }
    } // end per-expert (non-fused / streaming) path

    // cur_moe = rms_norm(moe_out) * post_ffw_norm_2, then out += cur_moe
    std::vector<float> cur_moe((size_t) n_embd * n_new);
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_mul_f32(cur_moe.data() + (size_t) t * n_embd,
                        moe_out.data() + (size_t) t * n_embd,
                        post_ffw_norm_2, n_embd, eps);
    }
    for (size_t i = 0; i < (size_t) n_embd * n_new; ++i) out[i] += cur_moe[i];

    return true;
}

namespace {

// Deterministic pseudo-random activation fill (no RNG dependency).
void fill_x(std::vector<float> & x, uint32_t seed) {
    uint32_t s = seed * 2654435761u + 1013904223u;
    for (float & v : x) {
        s = s * 1664525u + 1013904223u;
        // map to roughly [-1, 1)
        v = (float) ((int32_t) (s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    }
}

// Compare W3d[:,:,e] @ x computed via the streaming view vs a contiguous copy
// of the expert block. Returns false and fills `error` on any mismatch.
bool check_expert_tensor(MatmulCtx & mm, const ggml_tensor * W3d,
                         const char * name, int n_cols,
                         const std::vector<int> & experts, std::string & error) {
    const int64_t n_in  = W3d->ne[0];
    const int64_t n_out = W3d->ne[1];

    std::vector<float> x((std::size_t) n_in * n_cols);
    std::vector<float> y_view((std::size_t) n_out * n_cols);
    std::vector<float> y_ref((std::size_t) n_out * n_cols);

    for (int e : experts) {
        fill_x(x, (uint32_t) (e + 1));

        if (!matmul_expert_qf32(mm, W3d, e, x.data(), y_view.data(),
                                (int) n_in, (int) n_out, n_cols, error)) {
            return false;
        }

        // Contiguous copy of expert e's sub-block into a private context.
        const std::size_t block_off   = (std::size_t) e * W3d->nb[2];
        ggml_init_params ip{ (std::size_t) 32 * 1024 * 1024, nullptr, /*no_alloc=*/false };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) { error = "moe selftest: ggml_init failed"; return false; }
        ggml_tensor * Wcopy = ggml_new_tensor_2d(ctx, W3d->type, n_in, n_out);
        if (!Wcopy) { error = "moe selftest: alloc Wcopy failed"; ggml_free(ctx); return false; }
        std::memcpy(Wcopy->data, (const char *) W3d->data + block_off, ggml_nbytes(Wcopy));

        const bool ok = matmul_qf32(mm, Wcopy, x.data(), y_ref.data(),
                                    (int) n_in, (int) n_out, n_cols, error);
        ggml_free(ctx);
        if (!ok) return false;

        double max_abs = 0.0;
        for (std::size_t i = 0; i < y_view.size(); ++i) {
            max_abs = std::max(max_abs, (double) std::fabs(y_view[i] - y_ref[i]));
        }
        std::fprintf(stderr,
            "  %-16s expert %3d: max_abs(view-copy) = %.3e  [%s, %lldx%lld]\n",
            name, e, max_abs, ggml_type_name(W3d->type),
            (long long) n_in, (long long) n_out);
        if (max_abs != 0.0) {
            std::ostringstream ss;
            ss << "moe selftest: " << name << " expert " << e
               << " view != copy (max_abs=" << max_abs << ") -- sub-view misaligned";
            error = ss.str();
            return false;
        }
    }
    return true;
}

} // namespace

bool moe_expert_view_selftest(const llama_model * model, const Weights & w,
                              int il, int n_cols, int n_threads,
                              std::string & error) {
    (void) model;
    if (!w.is_moe) { error = "moe selftest: model is not MoE"; return false; }
    if (il < 0 || il >= w.n_layer) {
        std::ostringstream ss;
        ss << "moe selftest: layer " << il << " out of range [0," << w.n_layer << ")";
        error = ss.str();
        return false;
    }
    const LayerWeights & L = w.layers[il];
    if (!L.is_moe_layer) {
        std::ostringstream ss;
        ss << "moe selftest: layer " << il << " is not a MoE layer";
        error = ss.str();
        return false;
    }
    if (n_cols <= 0) n_cols = 4;
    if (n_threads <= 0) n_threads = 4;

    MatmulCtx mm;
    if (!matmul_ctx_init(mm, (std::size_t) 64 * 1024 * 1024, n_threads, error)) {
        return false;
    }

    // A spread of expert indices (endpoints + interior) so any offset error
    // that only shows up mid-tensor is caught.
    std::vector<int> experts = { 0, 1, w.n_expert / 2, w.n_expert - 1 };

    std::fprintf(stderr,
        "gemma4 moe_expert_view_selftest: il=%d n_expert=%d n_ff_exp=%d n_cols=%d n_threads=%d\n",
        il, w.n_expert, w.n_ff_exp, n_cols, n_threads);

    if (L.ffn_gate_up_exps) {
        if (!check_expert_tensor(mm, L.ffn_gate_up_exps, "ffn_gate_up_exps",
                                 n_cols, experts, error)) return false;
    } else {
        if (!check_expert_tensor(mm, L.ffn_gate_exps, "ffn_gate_exps",
                                 n_cols, experts, error)) return false;
        if (!check_expert_tensor(mm, L.ffn_up_exps, "ffn_up_exps",
                                 n_cols, experts, error)) return false;
    }
    if (!check_expert_tensor(mm, L.ffn_down_exps, "ffn_down_exps",
                             n_cols, experts, error)) return false;

    return true;
}

} // namespace gemma4
