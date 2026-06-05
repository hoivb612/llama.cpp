#include "gemma4_forward.h"
#include "gemma4_kernels.h"
#include "gemma4_weights.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

namespace gemma4 {

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

namespace {

// Copy an F32 tensor's contents into a std::vector<float>.
bool copy_f32(const ggml_tensor * t, std::vector<float> & out, std::string & err) {
    if (!t) { err = "copy_f32: tensor is null"; return false; }
    if (t->type != GGML_TYPE_F32) {
        std::ostringstream ss;
        ss << "copy_f32: tensor '" << (t->name[0] ? t->name : "?") << "' has type "
           << ggml_type_name(t->type) << ", expected F32";
        err = ss.str();
        return false;
    }
    const int64_t n = ggml_nelements(t);
    out.assign((size_t) n, 0.0f);
    std::memcpy(out.data(), t->data, (size_t) n * sizeof(float));
    return true;
}

// Dequantize a (possibly K-quant) tensor into a contiguous F32 buffer of
// ggml_nelements(t) elements.
bool dequant_f32(const ggml_tensor * t, std::vector<float> & out, std::string & err) {
    if (!t) { err = "dequant_f32: tensor is null"; return false; }
    const int64_t n = ggml_nelements(t);
    out.assign((size_t) n, 0.0f);

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t) n * sizeof(float));
        return true;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) {
        std::ostringstream ss;
        ss << "dequant_f32: no to_float for type " << ggml_type_name(t->type);
        err = ss.str();
        return false;
    }

    // Dequant row-by-row using row stride from ggml.
    const int64_t ne0 = t->ne[0];
    const int64_t n_rows = n / ne0;
    const size_t  row_bytes_src = ggml_row_size(t->type, ne0);
    const uint8_t * src = (const uint8_t *) t->data;
    float * dst = out.data();
    for (int64_t r = 0; r < n_rows; ++r) {
        traits->to_float(src + (size_t) r * row_bytes_src,
                         dst + (size_t) r * ne0,
                         (int) ne0);
    }
    return true;
}

// Hand-coded matmul matching ggml_mul_mat semantics.
//   W: [K, M]   stored row-major in std::vector with W[m*K + k]
//   x: [K, N]   x[n*K + k]
//   out: [M, N] out[n*M + m] = sum_k W[m*K+k] * x[n*K+k]
void matmul_f32(const float * W, const float * x, float * out,
                int K, int M, int N) {
    for (int n = 0; n < N; ++n) {
        const float * xn = x + (size_t) n * K;
        float * outn = out + (size_t) n * M;
        for (int m = 0; m < M; ++m) {
            const float * wm = W + (size_t) m * K;
            double s = 0.0;  // accumulate in double for numeric stability
            for (int k = 0; k < K; ++k) {
                s += (double) wm[k] * (double) xn[k];
            }
            outn[m] = (float) s;
        }
    }
}

// Convenience to create+init an F32 ggml tensor inside an oracle context.
ggml_tensor * new_f32_2d(ggml_context * gctx, int64_t d0, int64_t d1, const float * src) {
    ggml_tensor * t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d0, d1);
    if (src) std::memcpy(t->data, src, (size_t) d0 * d1 * sizeof(float));
    return t;
}
ggml_tensor * new_f32_1d(ggml_context * gctx, int64_t d0, const float * src) {
    ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, d0);
    if (src) std::memcpy(t->data, src, (size_t) d0 * sizeof(float));
    return t;
}

// First-mismatch diagnostic.
bool compare_f32(const char * tag, const float * a, const float * b, size_t n,
                 float atol, float rtol, std::string & err) {
    float max_abs = 0.0f;
    size_t worst_i = 0;
    for (size_t i = 0; i < n; ++i) {
        const float ai = a[i], bi = b[i];
        const float d = std::fabs(ai - bi);
        if (d > max_abs) { max_abs = d; worst_i = i; }
        const float tol = atol + rtol * std::fabs(bi);
        if (!(d <= tol)) {
            std::ostringstream ss;
            ss << tag << ": mismatch @i=" << i
               << " hand=" << ai << " oracle=" << bi
               << " |delta|=" << d << " tol=" << tol;
            err = ss.str();
            return false;
        }
    }
    std::fprintf(stderr, "  %-24s OK  max_abs=%.3e (i=%zu)\n", tag, max_abs, worst_i);
    return true;
}

} // anonymous namespace

// ---------------------------------------------------------------------
// dequant_layer
// ---------------------------------------------------------------------

bool dequant_layer(const llama_model * model, const Weights & w_global,
                   int il, LayerF32 & out, std::string & error) {
    (void) model;
    if (il < 0 || il >= (int) w_global.layers.size()) {
        error = "dequant_layer: il out of range";
        return false;
    }
    const LayerWeights & L = w_global.layers[il];
    // Gemma-4 shared-KV layers (kv_reuse_il >= 0) still have wk/wv tensors in
    // the GGUF; we just won't dequant or use them. The earlier check that
    // required has_kv has been removed -- shared-KV layers are now legal.

    out = LayerF32{};
    out.il        = il;
    out.n_embd    = w_global.n_embd;
    out.n_head    = w_global.n_head;
    out.n_head_kv = w_global.n_head_kv;
    out.head_dim  = L.head_dim;
    out.n_ff      = L.n_ff;
    out.n_embd_per_layer = w_global.n_embd_per_layer;
    out.is_swa    = L.is_swa;
    out.rms_eps   = w_global.rms_eps;
    out.rope_base = L.is_swa ? w_global.rope_freq_base_swa : w_global.rope_freq_base;
    out.rope_dim  = L.head_dim;
    out.kv_reuse_il = L.kv_reuse_il;

    // freq_factors: gemma4 uses rope_freqs only for non-SWA full-attn layers.
    if (!L.is_swa && w_global.rope_freqs && w_global.rope_freqs->type == GGML_TYPE_F32) {
        out.freq_factors = (const float *) w_global.rope_freqs->data;
    } else {
        out.freq_factors = nullptr;
    }

    if (!copy_f32(L.attn_norm,      out.attn_norm,      error)) return false;
    if (!copy_f32(L.attn_q_norm,    out.attn_q_norm,    error)) return false;
    if (!copy_f32(L.attn_k_norm,    out.attn_k_norm,    error)) return false;
    if (!copy_f32(L.post_attn_norm, out.post_attn_norm, error)) return false;
    if (!copy_f32(L.ffn_norm,       out.ffn_norm,       error)) return false;
    if (!copy_f32(L.post_ffw_norm,  out.post_ffw_norm,  error)) return false;
    if (!copy_f32(L.post_norm,      out.post_norm,      error)) return false;

    if (!dequant_f32(L.wq,       out.wq,       error)) return false;
    if (out.kv_reuse_il < 0) {
        // Own KV: dequant K and (optional) V projection.
        if (!dequant_f32(L.wk, out.wk, error)) return false;
        if (L.has_v_proj) {
            if (!dequant_f32(L.wv, out.wv, error)) return false;
        }
    }
    // else: shared-KV layer -- skip wk/wv dequant (memory saver; they exist
    // in the GGUF but upstream graph never reads them for these layers).
    if (!dequant_f32(L.wo,       out.wo,       error)) return false;
    if (!dequant_f32(L.ffn_gate, out.ffn_gate, error)) return false;
    if (!dequant_f32(L.ffn_up,   out.ffn_up,   error)) return false;
    if (!dequant_f32(L.ffn_down, out.ffn_down, error)) return false;
    if (!copy_f32(L.inp_gate,    out.inp_gate, error)) return false;
    if (!copy_f32(L.proj,        out.proj,     error)) return false;

    if (L.layer_output_scale) {
        std::vector<float> s;
        if (!copy_f32(L.layer_output_scale, s, error)) return false;
        if (s.size() == 1) {
            out.has_layer_output_scale = true;
            out.layer_output_scale     = s[0];
        }
    }
    return true;
}

// ---------------------------------------------------------------------
// Hand-coded F32 layer forward (cached version is the source of truth)
// ---------------------------------------------------------------------
//
// Conventions for layer_forward_f32_cached:
//   n_new          = number of new tokens this call (n_prompt for prefill,
//                    1 for a decode step).
//   n_past         = number of tokens already in K_cache (0 for prefill).
//   n_total        = n_past + n_new
//   pos_all[i]     = position of cached token i (i in [0..n_total))
//   K_cache,V_cache: size [n_kv * n_total]. For an owning layer, this call
//                    writes the new K (post-norm + post-RoPE) and V
//                    (post-norm) at offset n_past * n_kv. For a reuse layer
//                    (reuse_kv==true), the buffer is assumed already
//                    populated by the earlier owning layer.
//   n_swa          = SWA window. Only consulted for L.is_swa layers.
//                    Pass INT32_MAX to disable (no SWA mask).
bool layer_forward_f32_cached(const LayerF32 & L,
                              int n_new, int n_past, int n_swa,
                              const float * hidden_in,
                              const int32_t * pos_all,
                              const float * per_layer_input,
                              float * hidden_out,
                              float * K_cache, float * V_cache,
                              bool reuse_kv,
                              std::string & error) {
    if (n_new <= 0) { error = "layer_forward_f32_cached: n_new<=0"; return false; }
    if (n_past < 0) { error = "layer_forward_f32_cached: n_past<0"; return false; }
    if (!K_cache || !V_cache) { error = "layer_forward_f32_cached: null K/V cache"; return false; }
    if (reuse_kv && L.kv_reuse_il < 0) {
        error = "layer_forward_f32_cached: reuse_kv=true for an own-KV layer";
        return false;
    }
    if (!reuse_kv && L.kv_reuse_il >= 0) {
        error = "layer_forward_f32_cached: shared-KV layer requires reuse_kv=true";
        return false;
    }

    const int n_embd    = L.n_embd;
    const int n_head    = L.n_head;
    const int n_head_kv = L.n_head_kv;
    const int head_dim  = L.head_dim;
    const int n_ff      = L.n_ff;
    const int n_epl     = L.n_embd_per_layer;
    const int n_q       = n_head * head_dim;
    const int n_kv      = n_head_kv * head_dim;
    const int n_total   = n_past + n_new;
    const float eps     = L.rms_eps;

    // -------- attn_norm: norm1[t,:] = rmsnorm(hidden_in[t,:]) * attn_norm --
    std::vector<float> norm1((size_t) n_embd * n_new, 0.0f);
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_mul_f32(norm1.data() + (size_t) t * n_embd,
                        hidden_in + (size_t) t * n_embd,
                        L.attn_norm.data(), n_embd, eps);
    }

    // -------- Q = wq @ norm1, then q_norm + RoPE (always own) --------
    std::vector<float> Q((size_t) n_q * n_new, 0.0f);
    matmul_f32(L.wq.data(), norm1.data(), Q.data(), n_embd, n_q, n_new);
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_per_head_f32(Q.data() + (size_t) t * n_q,
                             Q.data() + (size_t) t * n_q,
                             L.attn_q_norm.data(), head_dim, n_head, eps);
    }
    for (int t = 0; t < n_new; ++t) {
        const int p = pos_all[n_past + t];
        for (int h = 0; h < n_head; ++h) {
            float * q_th = Q.data() + (size_t) t * n_q + (size_t) h * head_dim;
            rope_neox_f32(q_th, q_th, L.freq_factors,
                          L.rope_dim, head_dim, p, L.rope_base);
        }
    }

    // -------- K, V for new tokens (skip entirely on reuse layers) --------
    if (!reuse_kv) {
        float * K_new = K_cache + (size_t) n_past * n_kv;
        float * V_new = V_cache + (size_t) n_past * n_kv;

        matmul_f32(L.wk.data(), norm1.data(), K_new, n_embd, n_kv, n_new);

        std::vector<float> V_scratch;
        if (!L.wv.empty()) {
            matmul_f32(L.wv.data(), norm1.data(), V_new, n_embd, n_kv, n_new);
        } else {
            // V = K when wv missing; copy K_new -> V_new.
            std::memcpy(V_new, K_new, (size_t) n_kv * n_new * sizeof(float));
        }

        // K norm (per kv-head)
        for (int t = 0; t < n_new; ++t) {
            rmsnorm_per_head_f32(K_new + (size_t) t * n_kv,
                                 K_new + (size_t) t * n_kv,
                                 L.attn_k_norm.data(), head_dim, n_head_kv, eps);
        }
        // V norm (no weight, gemma4 quirk)
        for (int t = 0; t < n_new; ++t) {
            rmsnorm_per_head_f32(V_new + (size_t) t * n_kv,
                                 V_new + (size_t) t * n_kv,
                                 /*w=*/nullptr, head_dim, n_head_kv, eps);
        }
        // RoPE on K (new positions only)
        for (int t = 0; t < n_new; ++t) {
            const int p = pos_all[n_past + t];
            for (int h = 0; h < n_head_kv; ++h) {
                float * k_th = K_new + (size_t) t * n_kv + (size_t) h * head_dim;
                rope_neox_f32(k_th, k_th, L.freq_factors,
                              L.rope_dim, head_dim, p, L.rope_base);
            }
        }
    }

    // -------- Self-attention over n_total cached positions --------
    // scale = 1.0 (gemma4 hparams.f_attention_scale = 1.0).
    std::vector<float> attn_ctx((size_t) n_q * n_new, 0.0f);
    std::vector<float> scores((size_t) n_total, 0.0f);
    const bool apply_swa = L.is_swa;
    for (int t = 0; t < n_new; ++t) {
        const int p_t = pos_all[n_past + t];
        for (int h = 0; h < n_head; ++h) {
            const int h_kv = h * n_head_kv / n_head;
            const float * q_th = Q.data() + (size_t) t * n_q + (size_t) h * head_dim;
            float max_s = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < n_total; ++k) {
                const int p_k = pos_all[k];
                const bool masked = (p_k > p_t) ||
                                    (apply_swa && (p_t - p_k >= n_swa));
                if (masked) {
                    scores[k] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float * k_th = K_cache + (size_t) k * n_kv + (size_t) h_kv * head_dim;
                double s = 0.0;
                for (int d = 0; d < head_dim; ++d) s += (double) q_th[d] * (double) k_th[d];
                const float sf = (float) s;
                scores[k] = sf;
                if (sf > max_s) max_s = sf;
            }
            double sum = 0.0;
            for (int k = 0; k < n_total; ++k) {
                if (scores[k] == -std::numeric_limits<float>::infinity()) {
                    scores[k] = 0.0f;
                } else {
                    scores[k] = std::exp(scores[k] - max_s);
                    sum += (double) scores[k];
                }
            }
            const float inv_sum = sum > 0.0 ? (float) (1.0 / sum) : 0.0f;
            for (int k = 0; k < n_total; ++k) scores[k] *= inv_sum;
            float * out_th = attn_ctx.data() + (size_t) t * n_q + (size_t) h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (int k = 0; k < n_total; ++k) {
                    const float * v_th = V_cache + (size_t) k * n_kv + (size_t) h_kv * head_dim;
                    acc += (double) scores[k] * (double) v_th[d];
                }
                out_th[d] = (float) acc;
            }
        }
    }

    // -------- wo: attn_out = wo @ attn_ctx --------
    std::vector<float> attn_out((size_t) n_embd * n_new, 0.0f);
    matmul_f32(L.wo.data(), attn_ctx.data(), attn_out.data(), n_q, n_embd, n_new);

    // -------- post_attn_norm + residual1 --------
    std::vector<float> attn_out2((size_t) n_embd * n_new, 0.0f);
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_mul_f32(attn_out2.data() + (size_t) t * n_embd,
                        attn_out.data() + (size_t) t * n_embd,
                        L.post_attn_norm.data(), n_embd, eps);
    }
    for (size_t i = 0; i < (size_t) n_embd * n_new; ++i) {
        attn_out2[i] += hidden_in[i];
    }

    // -------- ffn_norm --------
    std::vector<float> ff_in((size_t) n_embd * n_new, 0.0f);
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_mul_f32(ff_in.data() + (size_t) t * n_embd,
                        attn_out2.data() + (size_t) t * n_embd,
                        L.ffn_norm.data(), n_embd, eps);
    }

    // -------- gate, up, gelu(gate)*up --------
    std::vector<float> gate((size_t) n_ff * n_new, 0.0f);
    std::vector<float> up  ((size_t) n_ff * n_new, 0.0f);
    matmul_f32(L.ffn_gate.data(), ff_in.data(), gate.data(), n_embd, n_ff, n_new);
    matmul_f32(L.ffn_up.data(),   ff_in.data(), up.data(),   n_embd, n_ff, n_new);
    gelu_f32(gate.data(), gate.data(), n_ff * n_new);
    for (size_t i = 0; i < (size_t) n_ff * n_new; ++i) gate[i] *= up[i];

    // -------- ffn_down --------
    std::vector<float> ff_out((size_t) n_embd * n_new, 0.0f);
    matmul_f32(L.ffn_down.data(), gate.data(), ff_out.data(), n_ff, n_embd, n_new);

    // -------- post_ffw_norm + residual2 --------
    std::vector<float> pe_in((size_t) n_embd * n_new, 0.0f);
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_mul_f32(pe_in.data() + (size_t) t * n_embd,
                        ff_out.data() + (size_t) t * n_embd,
                        L.post_ffw_norm.data(), n_embd, eps);
    }
    for (size_t i = 0; i < (size_t) n_embd * n_new; ++i) {
        pe_in[i] += attn_out2[i];
    }

    // -------- PLE: cur = pe_in + post_norm(proj(gelu(inp_gate @ pe_in) * slice)) --
    std::vector<float> ple_a((size_t) n_epl * n_new, 0.0f);
    matmul_f32(L.inp_gate.data(), pe_in.data(), ple_a.data(), n_embd, n_epl, n_new);
    gelu_f32(ple_a.data(), ple_a.data(), n_epl * n_new);
    for (size_t i = 0; i < (size_t) n_epl * n_new; ++i) {
        ple_a[i] *= per_layer_input[i];
    }
    std::vector<float> ple_b((size_t) n_embd * n_new, 0.0f);
    matmul_f32(L.proj.data(), ple_a.data(), ple_b.data(), n_epl, n_embd, n_new);
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_mul_f32(ple_b.data() + (size_t) t * n_embd,
                        ple_b.data() + (size_t) t * n_embd,
                        L.post_norm.data(), n_embd, eps);
    }
    for (size_t i = 0; i < (size_t) n_embd * n_new; ++i) {
        pe_in[i] += ple_b[i];
    }

    // -------- layer_output_scale --------
    if (L.has_layer_output_scale) {
        const float s = L.layer_output_scale;
        for (size_t i = 0; i < (size_t) n_embd * n_new; ++i) {
            pe_in[i] *= s;
        }
    }

    std::memcpy(hidden_out, pe_in.data(), (size_t) n_embd * n_new * sizeof(float));
    return true;
}

// Backwards-compatible G3.3 / G3.4a API: prefill-only (no past KV), no
// SWA mask. Delegates to layer_forward_f32_cached with n_past=0 and
// n_swa=INT32_MAX. The four optional KV pointers select between
// "compute K/V into caller-supplied buffer" and "reuse caller-supplied K/V".
bool layer_forward_f32(const LayerF32 & L,
                       int n_tokens,
                       const float * hidden_in,
                       const int32_t * pos,
                       const float * per_layer_input,
                       float * hidden_out,
                       std::string & error,
                       float * kv_K_self_out,
                       float * kv_V_self_out,
                       const float * kv_K_reuse,
                       const float * kv_V_reuse) {
    const int n_kv = L.n_head_kv * L.head_dim;
    const bool reuse = (kv_K_reuse != nullptr) && (kv_V_reuse != nullptr);

    float * K_buf = nullptr;
    float * V_buf = nullptr;
    std::vector<float> K_local, V_local;

    if (reuse) {
        // Reuse path: caller already populated the buffer.
        K_buf = const_cast<float *>(kv_K_reuse);
        V_buf = const_cast<float *>(kv_V_reuse);
    } else if (kv_K_self_out) {
        K_buf = kv_K_self_out;
        V_buf = kv_V_self_out;
    } else {
        K_local.assign((size_t) n_kv * n_tokens, 0.0f);
        V_local.assign((size_t) n_kv * n_tokens, 0.0f);
        K_buf = K_local.data();
        V_buf = V_local.data();
    }

    return layer_forward_f32_cached(L, n_tokens, /*n_past=*/0,
                                    /*n_swa=*/std::numeric_limits<int>::max(),
                                    hidden_in, pos, per_layer_input,
                                    hidden_out, K_buf, V_buf, reuse, error);
}

// ---------------------------------------------------------------------
// ggml-graph oracle
// ---------------------------------------------------------------------

bool oracle_layer_forward_f32(const LayerF32 & L,
                              int n_tokens,
                              const float * hidden_in,
                              const int32_t * pos,
                              const float * per_layer_input,
                              float * hidden_out,
                              std::string & error) {
    if (n_tokens <= 0) { error = "oracle: n_tokens<=0"; return false; }
    const int n_embd    = L.n_embd;
    const int n_head    = L.n_head;
    const int n_head_kv = L.n_head_kv;
    const int head_dim  = L.head_dim;
    const int n_ff      = L.n_ff;
    const int n_epl     = L.n_embd_per_layer;
    const int n_q       = n_head * head_dim;
    const int n_kv      = n_head_kv * head_dim;
    const float eps     = L.rms_eps;

    // Generous arena: per-test, allocate ~1 GiB on heap.
    std::vector<uint8_t> arena((size_t) 1ULL << 30);
    ggml_init_params ip{ arena.size(), arena.data(), false };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { error = "oracle: ggml_init failed"; return false; }

    // Inputs.
    ggml_tensor * t_x  = new_f32_2d(gctx, n_embd, n_tokens, hidden_in);
    ggml_tensor * t_pl = new_f32_2d(gctx, n_epl,  n_tokens, per_layer_input);
    ggml_tensor * t_pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    std::memcpy(t_pos->data, pos, (size_t) n_tokens * sizeof(int32_t));

    // Weights.
    ggml_tensor * t_attn_norm      = new_f32_1d(gctx, n_embd,  L.attn_norm.data());
    ggml_tensor * t_attn_q_norm    = new_f32_1d(gctx, head_dim, L.attn_q_norm.data());
    ggml_tensor * t_attn_k_norm    = new_f32_1d(gctx, head_dim, L.attn_k_norm.data());
    ggml_tensor * t_post_attn_norm = new_f32_1d(gctx, n_embd,  L.post_attn_norm.data());
    ggml_tensor * t_ffn_norm       = new_f32_1d(gctx, n_embd,  L.ffn_norm.data());
    ggml_tensor * t_post_ffw_norm  = new_f32_1d(gctx, n_embd,  L.post_ffw_norm.data());
    ggml_tensor * t_post_norm      = new_f32_1d(gctx, n_embd,  L.post_norm.data());

    ggml_tensor * t_wq    = new_f32_2d(gctx, n_embd, n_q, L.wq.data());
    ggml_tensor * t_wk    = new_f32_2d(gctx, n_embd, n_kv, L.wk.data());
    ggml_tensor * t_wv    = nullptr;
    if (!L.wv.empty()) t_wv = new_f32_2d(gctx, n_embd, n_kv, L.wv.data());
    ggml_tensor * t_wo    = new_f32_2d(gctx, n_q,    n_embd, L.wo.data());
    ggml_tensor * t_ffn_gate = new_f32_2d(gctx, n_embd, n_ff, L.ffn_gate.data());
    ggml_tensor * t_ffn_up   = new_f32_2d(gctx, n_embd, n_ff, L.ffn_up.data());
    ggml_tensor * t_ffn_down = new_f32_2d(gctx, n_ff,   n_embd, L.ffn_down.data());
    ggml_tensor * t_inp_gate = new_f32_2d(gctx, n_embd, n_epl, L.inp_gate.data());
    ggml_tensor * t_proj     = new_f32_2d(gctx, n_epl,  n_embd, L.proj.data());

    ggml_tensor * t_freq_factors = nullptr;
    if (L.freq_factors) {
        t_freq_factors = new_f32_1d(gctx, L.rope_dim / 2, L.freq_factors);
    }

    // Causal mask: F32 [n_tokens, n_tokens]. mask[k, t] = 0 if pos[k]<=pos[t] else -INF.
    ggml_tensor * t_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_tokens, n_tokens);
    {
        float * m = (float *) t_mask->data;
        for (int t = 0; t < n_tokens; ++t) {
            for (int k = 0; k < n_tokens; ++k) {
                m[(size_t) t * n_tokens + k] = (pos[k] <= pos[t])
                    ? 0.0f
                    : -std::numeric_limits<float>::infinity();
            }
        }
    }

    // ---------------- Build graph mirroring src/models/gemma4.cpp::graph ----
    // norm1 = rms_norm(x) * attn_norm
    ggml_tensor * norm1 = ggml_rms_norm(gctx, t_x, eps);
    norm1 = ggml_mul(gctx, norm1, t_attn_norm);

    // Q = wq @ norm1 -> reshape [head_dim, n_head, n_tokens]
    ggml_tensor * Q = ggml_mul_mat(gctx, t_wq, norm1);
    Q = ggml_reshape_3d(gctx, Q, head_dim, n_head, n_tokens);
    Q = ggml_rms_norm(gctx, Q, eps);
    Q = ggml_mul(gctx, Q, t_attn_q_norm);
    Q = ggml_rope_ext(gctx, Q, t_pos, t_freq_factors, L.rope_dim,
                      GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0,
                      L.rope_base, /*freq_scale=*/1.0f,
                      /*ext=*/0.0f, /*att=*/1.0f, /*bf=*/0.0f, /*bs=*/0.0f);

    // K = wk @ norm1 -> reshape [head_dim, n_head_kv, n_tokens]
    ggml_tensor * Kc = ggml_mul_mat(gctx, t_wk, norm1);
    Kc = ggml_reshape_3d(gctx, Kc, head_dim, n_head_kv, n_tokens);

    // V = wv @ norm1 (or K)
    ggml_tensor * Vc;
    if (t_wv) {
        Vc = ggml_mul_mat(gctx, t_wv, norm1);
        Vc = ggml_reshape_3d(gctx, Vc, head_dim, n_head_kv, n_tokens);
    } else {
        Vc = Kc;
    }

    Kc = ggml_rms_norm(gctx, Kc, eps);
    Kc = ggml_mul(gctx, Kc, t_attn_k_norm);
    Vc = ggml_rms_norm(gctx, Vc, eps);   // gemma4 V: rms_norm WITHOUT weight

    Kc = ggml_rope_ext(gctx, Kc, t_pos, t_freq_factors, L.rope_dim,
                       GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0,
                       L.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Attention: scale=1.0 (gemma4), causal mask provided as F32.
    // permute to [head_dim, n_tokens, n_head*] for K^T @ Q -> [n_tokens, n_tokens, n_head]
    ggml_tensor * Qp = ggml_cont(gctx, ggml_permute(gctx, Q,  0, 2, 1, 3));
    ggml_tensor * Kp = ggml_cont(gctx, ggml_permute(gctx, Kc, 0, 2, 1, 3));
    ggml_tensor * Vp = ggml_cont(gctx, ggml_permute(gctx, Vc, 0, 2, 1, 3));

    // GQA broadcast: if n_head != n_head_kv, ggml_mul_mat broadcasts dim-2 of Kp/Vp
    // automatically when n_head is a multiple of n_head_kv (matches upstream).
    ggml_tensor * kq = ggml_mul_mat(gctx, Kp, Qp);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(gctx, kq, t_mask, /*scale=*/1.0f, /*max_bias=*/0.0f);

    ggml_tensor * Vt  = ggml_cont(gctx, ggml_transpose(gctx, Vp));
    ggml_tensor * kqv = ggml_mul_mat(gctx, Vt, kq);
    kqv = ggml_cont(gctx, ggml_permute(gctx, kqv, 0, 2, 1, 3));
    // attn_ctx shape: [n_head * head_dim, n_tokens]. For gemma4 SWA
    // layers this is NOT n_embd (e.g. E2B SWA: 8*256=2048 vs n_embd=1536).
    ggml_tensor * attn_ctx_t = ggml_reshape_2d(gctx, kqv, n_q, n_tokens);

    ggml_tensor * attn_out = ggml_mul_mat(gctx, t_wo, attn_ctx_t);

    ggml_tensor * post_attn = ggml_rms_norm(gctx, attn_out, eps);
    post_attn = ggml_mul(gctx, post_attn, t_post_attn_norm);
    ggml_tensor * attn_out2 = ggml_add(gctx, post_attn, t_x);

    // FFN: gelu(gate(x)) * up(x) -> down
    ggml_tensor * ff_in = ggml_rms_norm(gctx, attn_out2, eps);
    ff_in = ggml_mul(gctx, ff_in, t_ffn_norm);

    ggml_tensor * gate = ggml_mul_mat(gctx, t_ffn_gate, ff_in);
    ggml_tensor * up   = ggml_mul_mat(gctx, t_ffn_up,   ff_in);
    gate = ggml_gelu(gctx, gate);
    ggml_tensor * mid  = ggml_mul(gctx, gate, up);
    ggml_tensor * ff_out = ggml_mul_mat(gctx, t_ffn_down, mid);

    ggml_tensor * post_ffw = ggml_rms_norm(gctx, ff_out, eps);
    post_ffw = ggml_mul(gctx, post_ffw, t_post_ffw_norm);
    ggml_tensor * pe_in = ggml_add(gctx, post_ffw, attn_out2);

    // PLE: cur = pe_in + post_norm(proj(gelu(inp_gate @ pe_in) * slice))
    ggml_tensor * ple = ggml_mul_mat(gctx, t_inp_gate, pe_in);
    ple = ggml_gelu(gctx, ple);
    ple = ggml_mul(gctx, ple, t_pl);
    ple = ggml_mul_mat(gctx, t_proj, ple);
    ple = ggml_rms_norm(gctx, ple, eps);
    ple = ggml_mul(gctx, ple, t_post_norm);
    ggml_tensor * cur = ggml_add(gctx, pe_in, ple);

    if (L.has_layer_output_scale) {
        cur = ggml_scale(gctx, cur, L.layer_output_scale);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 1024, false);
    ggml_build_forward_expand(gf, cur);

    const ggml_status status = ggml_graph_compute_with_ctx(gctx, gf, /*n_threads=*/1);
    if (status != GGML_STATUS_SUCCESS) {
        error = "oracle: ggml_graph_compute_with_ctx failed";
        ggml_free(gctx);
        return false;
    }

    std::memcpy(hidden_out, cur->data, (size_t) n_embd * n_tokens * sizeof(float));
    ggml_free(gctx);
    return true;
}

// ---------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------

bool layer_self_test(const llama_model * model, const Weights & w,
                     int il, int n_tokens, std::string & error) {
    LayerF32 L;
    if (!dequant_layer(model, w, il, L, error)) return false;

    std::fprintf(stderr,
        "gemma4 layer_self_test: il=%d n_tokens=%d n_embd=%d n_head=%d "
        "n_head_kv=%d head_dim=%d n_ff=%d n_epl=%d is_swa=%d rope_base=%.0f "
        "freq_factors=%s rms_eps=%g out_scale=%s%.6g\n",
        il, n_tokens, L.n_embd, L.n_head, L.n_head_kv, L.head_dim, L.n_ff,
        L.n_embd_per_layer, (int) L.is_swa, (double) L.rope_base,
        L.freq_factors ? "yes" : "no", (double) L.rms_eps,
        L.has_layer_output_scale ? "" : "none ",
        L.has_layer_output_scale ? (double) L.layer_output_scale : 1.0);

    // Random inputs. Seed deterministically for repro.
    std::mt19937 rng(0xC4F4u + (uint32_t) il * 7919u + (uint32_t) n_tokens);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float>   hidden_in((size_t) L.n_embd * n_tokens);
    std::vector<float>   per_layer_input((size_t) L.n_embd_per_layer * n_tokens);
    std::vector<int32_t> pos(n_tokens);
    for (auto & v : hidden_in)       v = nd(rng) * 0.1f;
    for (auto & v : per_layer_input) v = nd(rng) * 0.1f;
    for (int t = 0; t < n_tokens; ++t) pos[t] = t;

    std::vector<float> out_hand  ((size_t) L.n_embd * n_tokens, 0.0f);
    std::vector<float> out_oracle((size_t) L.n_embd * n_tokens, 0.0f);

    if (!layer_forward_f32(L, n_tokens, hidden_in.data(), pos.data(),
                           per_layer_input.data(), out_hand.data(), error)) {
        return false;
    }
    if (!oracle_layer_forward_f32(L, n_tokens, hidden_in.data(), pos.data(),
                                  per_layer_input.data(), out_oracle.data(), error)) {
        return false;
    }

    // Stage-end tolerance: hand path accumulates in double, oracle in
    // ggml's vec_dot (SIMD F32 reductions). For one full layer this is
    // typically ~1e-3 absolute. Start a touch looser; tighten in G3.4.
    return compare_f32("layer_forward",
                       out_hand.data(), out_oracle.data(),
                       out_hand.size(), /*atol=*/2e-2f, /*rtol=*/2e-2f, error);
}

// =====================================================================
// G3.4 -- whole-network F32 forward.
// =====================================================================

bool dequant_model(const llama_model * model, const Weights & w,
                   ModelF32 & out, std::string & error) {
    out = ModelF32{};
    out.n_layer            = w.n_layer;
    out.n_embd             = w.n_embd;
    out.n_head             = w.n_head;
    out.n_head_kv          = w.n_head_kv;
    out.n_vocab            = w.n_vocab;
    out.n_embd_per_layer   = w.n_embd_per_layer;
    out.n_swa              = w.n_swa;
    out.rms_eps            = w.rms_eps;
    out.final_logit_softcap = w.final_logit_softcap;
    out.output_tied_to_embd = w.output_tied_to_embd;

    if (!w.tok_embd) { error = "dequant_model: tok_embd missing"; return false; }
    if (!w.output_norm) { error = "dequant_model: output_norm missing"; return false; }
    if (!copy_f32(w.output_norm, out.output_norm, error)) return false;

    if (w.per_layer_model_proj) {
        if (!dequant_f32(w.per_layer_model_proj, out.per_layer_model_proj, error)) return false;
    }
    if (w.per_layer_proj_norm) {
        if (!copy_f32(w.per_layer_proj_norm, out.per_layer_proj_norm, error)) return false;
    }
    if (w.rope_freqs && w.rope_freqs->type == GGML_TYPE_F32) {
        const int64_t n = ggml_nelements(w.rope_freqs);
        out.freq_factors_data.assign((size_t) n, 0.0f);
        std::memcpy(out.freq_factors_data.data(), w.rope_freqs->data, (size_t) n * sizeof(float));
    }

    out.tok_embd_quant           = w.tok_embd;
    out.per_layer_tok_embd_quant = w.per_layer_tok_embd;

    // Dequant every layer. layer_forward_f32 needs LayerF32.freq_factors
    // pointing at OUR freq_factors_data (so the original model could in
    // principle be unloaded). We rebuild this pointer after dequant.
    out.layers.resize(w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        if (!dequant_layer(model, w, il, out.layers[il], error)) {
            std::ostringstream ss;
            ss << "dequant_model: layer " << il << ": " << error;
            error = ss.str();
            return false;
        }
        // Repoint freq_factors into our self-owned buffer if non-SWA.
        if (!out.layers[il].is_swa && !out.freq_factors_data.empty()) {
            out.layers[il].freq_factors = out.freq_factors_data.data();
        } else {
            out.layers[il].freq_factors = nullptr;
        }
    }

    std::fprintf(stderr,
        "gemma4 dequant_model: n_layer=%d n_embd=%d n_vocab=%d n_embd_per_layer=%d "
        "n_swa=%d softcap=%.1f tok_embd.type=%s per_layer_tok_embd.type=%s\n",
        out.n_layer, out.n_embd, out.n_vocab, out.n_embd_per_layer, out.n_swa,
        (double) out.final_logit_softcap,
        ggml_type_name(out.tok_embd_quant->type),
        out.per_layer_tok_embd_quant ? ggml_type_name(out.per_layer_tok_embd_quant->type) : "(none)");
    return true;
}

// Dequant one row of a 2D weight (shape [n_inner, n_outer]) into dst.
// Used for tok_embd lookups (per-token) and per-vocab-row lm_head matmul.
static void dequant_row(const ggml_tensor * t, int row_idx, float * dst) {
    if (t->type == GGML_TYPE_F32) {
        const float * base = (const float *) t->data;
        std::memcpy(dst, base + (size_t) row_idx * t->ne[0], (size_t) t->ne[0] * sizeof(float));
        return;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(t->type);
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    const uint8_t * base   = (const uint8_t *) t->data;
    traits->to_float(base + (size_t) row_idx * row_bytes, dst, (int) t->ne[0]);
}

// PLE preprocessing (project_per_layer_inputs in upstream).
// Outputs per_layer_final laid out as [n_embd_per_layer, n_tokens, n_layer]
// in contiguous memory, so slice for layer il is
//   per_layer_final.data() + (size_t) il * n_tokens * n_embd_per_layer
// and within that slice, token t starts at offset t * n_embd_per_layer.
static bool compute_per_layer_inputs(const ModelF32 & m,
                              int n_tokens,
                              const int32_t * token_ids,
                              const float * inpL,            // [n_embd, n_tokens]
                              std::vector<float> & per_layer_final,
                              std::string & error) {
    const int n_layer = m.n_layer;
    const int n_embd  = m.n_embd;
    const int n_epl   = m.n_embd_per_layer;
    const float eps   = m.rms_eps;
    if (m.per_layer_model_proj.empty()) {
        error = "compute_per_layer_inputs: per_layer_model_proj missing";
        return false;
    }
    if (m.per_layer_proj_norm.empty()) {
        error = "compute_per_layer_inputs: per_layer_proj_norm missing";
        return false;
    }
    if (!m.per_layer_tok_embd_quant) {
        error = "compute_per_layer_inputs: per_layer_tok_embd missing";
        return false;
    }

    // 1. per_layer_proj = (per_layer_model_proj @ inpL) / sqrt(n_embd)
    //    per_layer_model_proj: [n_embd, n_epl * n_layer]   (dequantized)
    //    inpL:                 [n_embd, n_tokens]
    //    out:                  [n_epl * n_layer, n_tokens]
    const int M = n_epl * n_layer;
    std::vector<float> proj_out((size_t) M * n_tokens, 0.0f);
    matmul_f32(m.per_layer_model_proj.data(), inpL, proj_out.data(),
               n_embd, M, n_tokens);
    const float inv_sqrt_nembd = 1.0f / std::sqrt((float) n_embd);
    for (auto & v : proj_out) v *= inv_sqrt_nembd;

    // 2. RMSnorm per (l, t) group along n_embd_per_layer with per_layer_proj_norm
    //    Logical shape is [n_epl, n_layer, n_tokens]; we operate on each
    //    [n_epl] slice in place.
    for (int t = 0; t < n_tokens; ++t) {
        float * base_t = proj_out.data() + (size_t) t * M;
        for (int l = 0; l < n_layer; ++l) {
            float * x = base_t + (size_t) l * n_epl;
            rmsnorm_mul_f32(x, x, m.per_layer_proj_norm.data(), n_epl, eps);
        }
    }

    // 3. Look up per_layer_tok_embd[token_ids] * sqrt(n_epl), shape [M, n_tokens].
    //    per_layer_tok_embd shape: [n_epl * n_layer, n_vocab] in ggml convention,
    //    row "tok" is the contiguous row of length M = n_epl * n_layer.
    //    On Q4_K_M E2B/E4B this tensor is Q5_K -- we dequant rows on-demand.
    const float scale_raw = std::sqrt((float) n_epl);
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    std::vector<float> ple_row(M);
    for (int t = 0; t < n_tokens; ++t) {
        const int tok = token_ids[t];
        if (tok < 0 || tok >= m.n_vocab) {
            std::ostringstream ss;
            ss << "compute_per_layer_inputs: token_ids[" << t << "]=" << tok
               << " out of range [0," << m.n_vocab << ")";
            error = ss.str();
            return false;
        }
        dequant_row(m.per_layer_tok_embd_quant, tok, ple_row.data());
        float * dst = proj_out.data() + (size_t) t * M;
        for (int i = 0; i < M; ++i) {
            dst[i] = (dst[i] + ple_row[i] * scale_raw) * inv_sqrt2;
        }
    }

    // 4. Permute [n_epl, n_layer, n_tokens] -> [n_epl, n_tokens, n_layer]
    //    so that per-layer slice extraction is just a single offset.
    //    src[t * (n_layer * n_epl) + l * n_epl + d]
    //    -> dst[l * (n_tokens * n_epl) + t * n_epl + d]
    per_layer_final.assign((size_t) n_layer * n_tokens * n_epl, 0.0f);
    for (int l = 0; l < n_layer; ++l) {
        for (int t = 0; t < n_tokens; ++t) {
            const float * src = proj_out.data()
                              + (size_t) t * M
                              + (size_t) l * n_epl;
            float * dst = per_layer_final.data()
                        + (size_t) l * n_tokens * n_epl
                        + (size_t) t * n_epl;
            std::memcpy(dst, src, (size_t) n_epl * sizeof(float));
        }
    }
    return true;
}

bool network_forward_f32(const ModelF32 & m,
                         int n_tokens,
                         const int32_t * token_ids,
                         const int32_t * pos,
                         bool last_token_only,
                         float * logits_out,
                         std::string & error) {
    if (n_tokens <= 0) { error = "network_forward_f32: n_tokens<=0"; return false; }
    if (n_tokens > m.n_swa) {
        std::ostringstream ss;
        ss << "network_forward_f32: n_tokens=" << n_tokens << " exceeds n_swa="
           << m.n_swa << "; SWA masking not implemented in G3.4a";
        error = ss.str();
        return false;
    }
    if (!m.tok_embd_quant) { error = "network_forward_f32: tok_embd not set"; return false; }
    if (m.output_norm.empty()) { error = "network_forward_f32: output_norm empty"; return false; }
    if (!m.output_tied_to_embd) {
        // gemma4 E2B/E4B always tie output to tok_embd; if a variant ever
        // ships untied, add a separate output matrix to ModelF32.
        error = "network_forward_f32: untied output not supported (no separate lm_head loaded)";
        return false;
    }
    const int n_embd  = m.n_embd;
    const int n_vocab = m.n_vocab;
    const float eps   = m.rms_eps;

    // ---- 1. Input embedding: inpL = tok_embd[token_ids] * sqrt(n_embd) ----
    std::vector<float> inpL((size_t) n_embd * n_tokens, 0.0f);
    const float emb_scale = std::sqrt((float) n_embd);
    for (int t = 0; t < n_tokens; ++t) {
        const int tok = token_ids[t];
        if (tok < 0 || tok >= n_vocab) {
            std::ostringstream ss;
            ss << "network_forward_f32: token_ids[" << t << "]=" << tok
               << " out of range";
            error = ss.str();
            return false;
        }
        dequant_row(m.tok_embd_quant, tok, inpL.data() + (size_t) t * n_embd);
        for (int e = 0; e < n_embd; ++e) {
            inpL[(size_t) t * n_embd + e] *= emb_scale;
        }
    }

    // ---- 2. PLE preprocessing ----
    std::vector<float> per_layer_final;
    if (!compute_per_layer_inputs(m, n_tokens, token_ids, inpL.data(),
                                  per_layer_final, error)) {
        return false;
    }
    const int n_epl = m.n_embd_per_layer;

    // ---- 3. Layer loop ----
    // Allocate per-layer K/V storage for layers that own their KV.
    // For E2B: n_layer_kv_from_start = 15, so K_storage[0..14] are used.
    // Shared-KV layers (15..34) point at K_storage[L.kv_reuse_il].
    std::vector<std::vector<float>> K_storage(m.n_layer);
    std::vector<std::vector<float>> V_storage(m.n_layer);

    std::vector<float> hidden_out((size_t) n_embd * n_tokens, 0.0f);
    for (int il = 0; il < m.n_layer; ++il) {
        const LayerF32 & L = m.layers[il];
        const float * slice = per_layer_final.data()
                            + (size_t) il * n_tokens * n_epl;
        const int n_kv = L.n_head_kv * L.head_dim;

        float * K_self_out = nullptr;
        float * V_self_out = nullptr;
        const float * K_reuse_in = nullptr;
        const float * V_reuse_in = nullptr;
        if (L.kv_reuse_il < 0) {
            // Own KV: allocate storage and let layer_forward fill it.
            K_storage[il].assign((size_t) n_kv * n_tokens, 0.0f);
            V_storage[il].assign((size_t) n_kv * n_tokens, 0.0f);
            K_self_out = K_storage[il].data();
            V_self_out = V_storage[il].data();
        } else {
            // Shared-KV: reuse the source layer's stored K, V (must already
            // be populated since kv_reuse_il < il by Gemma-4's design).
            const int src = L.kv_reuse_il;
            if (K_storage[src].empty() || V_storage[src].empty()) {
                std::ostringstream ss;
                ss << "network_forward_f32: layer " << il
                   << " reuses KV from layer " << src
                   << " but that source has empty storage";
                error = ss.str();
                return false;
            }
            K_reuse_in = K_storage[src].data();
            V_reuse_in = V_storage[src].data();
        }

        if (!layer_forward_f32(L, n_tokens, inpL.data(), pos, slice,
                               hidden_out.data(), error,
                               K_self_out, V_self_out,
                               K_reuse_in, V_reuse_in)) {
            std::ostringstream ss;
            ss << "network_forward_f32: layer " << il << ": " << error;
            error = ss.str();
            return false;
        }
        std::swap(inpL, hidden_out);  // inpL <- this layer's output
    }

    // ---- 4. Final output norm ----
    for (int t = 0; t < n_tokens; ++t) {
        rmsnorm_mul_f32(hidden_out.data() + (size_t) t * n_embd,
                        inpL.data() + (size_t) t * n_embd,
                        m.output_norm.data(), n_embd, eps);
    }
    std::swap(inpL, hidden_out);

    // ---- 5. lm_head: logits = tok_embd @ inpL (tied output) ----
    //   tok_embd : [n_embd, n_vocab]  (potentially quantized; dequant per row)
    //   inpL     : [n_embd, n_tokens]
    //   logits   : [n_vocab, n_tokens]  or [n_vocab] if last_token_only
    const int t_start = last_token_only ? n_tokens - 1 : 0;
    const int t_count = last_token_only ? 1 : n_tokens;
    std::vector<float> row(n_embd);
    for (int v = 0; v < n_vocab; ++v) {
        dequant_row(m.tok_embd_quant, v, row.data());
        for (int tc = 0; tc < t_count; ++tc) {
            const int t = t_start + tc;
            const float * xt = inpL.data() + (size_t) t * n_embd;
            double s = 0.0;
            for (int e = 0; e < n_embd; ++e) s += (double) row[e] * (double) xt[e];
            logits_out[(size_t) tc * n_vocab + v] = (float) s;
        }
    }

    // ---- 6. Final logit softcap (gemma4: cap = 30.0) ----
    if (m.final_logit_softcap > 0.0f) {
        const float cap = m.final_logit_softcap;
        const float inv = 1.0f / cap;
        const size_t total = (size_t) n_vocab * t_count;
        for (size_t i = 0; i < total; ++i) {
            logits_out[i] = cap * std::tanh(logits_out[i] * inv);
        }
    }
    return true;
}

// ---------------------------------------------------------------------
// Network self-test
// ---------------------------------------------------------------------

namespace {

// Helper: run upstream llama_decode on prompt_tokens and return the
// last-position logits as a std::vector.
bool upstream_last_token_logits(const llama_model * model,
                                const std::vector<int32_t> & prompt_tokens,
                                int n_threads,
                                std::vector<float> & logits_out,
                                std::string & error) {
    const int n_prompt = (int) prompt_tokens.size();
    const int n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_ctx    = n_prompt + 64;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_ctx;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf         = true;

    llama_context * ctx = llama_init_from_model(const_cast<llama_model *>(model), cp);
    if (!ctx) { error = "upstream_last_token_logits: llama_init_from_model failed"; return false; }
    llama_set_n_threads(ctx, n_threads, n_threads);

    llama_batch batch = llama_batch_get_one(
        const_cast<int32_t *>(prompt_tokens.data()), n_prompt);

    if (llama_decode(ctx, batch) != 0) {
        error = "upstream_last_token_logits: llama_decode failed";
        llama_free(ctx);
        return false;
    }

    const float * src = llama_get_logits_ith(ctx, n_prompt - 1);
    if (!src) { error = "upstream_last_token_logits: llama_get_logits_ith null"; llama_free(ctx); return false; }
    logits_out.assign(src, src + n_vocab);
    llama_free(ctx);
    return true;
}

// Top-k selection (returns sorted descending indices by value).
std::vector<int> top_k_indices(const std::vector<float> & logits, int k) {
    std::vector<int> idx(logits.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b){ return logits[a] > logits[b]; });
    idx.resize(k);
    return idx;
}

} // anonymous namespace

bool network_self_test(const llama_model * model, const Weights & w,
                       const std::string & prompt, int n_threads,
                       std::string & error) {
    if (n_threads <= 0) n_threads = 1;

    // -------- Tokenize the prompt --------
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<int32_t> tokens;
    tokens.resize(prompt.size() + 8);
    const int n_tok = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                                     tokens.data(), (int) tokens.size(),
                                     /*add_special=*/true, /*parse_special=*/true);
    if (n_tok < 0) { error = "network_self_test: tokenize failed"; return false; }
    tokens.resize(n_tok);
    if (n_tok > w.n_swa) {
        std::ostringstream ss;
        ss << "network_self_test: prompt tokenizes to " << n_tok
           << " tokens, exceeds n_swa=" << w.n_swa
           << "; pick a shorter prompt or wait for SWA mask in G3.4b";
        error = ss.str();
        return false;
    }
    std::fprintf(stderr, "gemma4 network_self_test: prompt=\"%s\" -> %d tokens (n_vocab=%d)\n",
                 prompt.c_str(), n_tok, n_vocab);

    // -------- Upstream reference --------
    std::vector<float> upstream_logits;
    if (!upstream_last_token_logits(model, tokens, n_threads, upstream_logits, error)) {
        return false;
    }

    // -------- Dequant + hand path --------
    ModelF32 mf;
    if (!dequant_model(model, w, mf, error)) return false;

    std::vector<int32_t> pos(n_tok);
    for (int i = 0; i < n_tok; ++i) pos[i] = i;

    std::vector<float> hand_logits((size_t) n_vocab, 0.0f);
    const auto t0 = std::chrono::steady_clock::now();
    if (!network_forward_f32(mf, n_tok, tokens.data(), pos.data(),
                             /*last_token_only=*/true,
                             hand_logits.data(), error)) {
        return false;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gemma4 network_self_test: hand path took %.1f ms (single batch, last-token logits)\n", ms);

    // -------- Metrics --------
    const int k = 10;
    auto top_h = top_k_indices(hand_logits, k);
    auto top_u = top_k_indices(upstream_logits, k);
    const int top1_h = top_h[0];
    const int top1_u = top_u[0];

    // Rank of upstream top-1 in hand ranking.
    int rank_top1_u_in_hand = -1;
    {
        std::vector<int> idx_full = top_k_indices(hand_logits, n_vocab);
        for (int r = 0; r < (int) idx_full.size(); ++r) {
            if (idx_full[r] == top1_u) { rank_top1_u_in_hand = r; break; }
        }
    }
    // Top-5 / top-10 set overlap.
    auto overlap = [&](int kk){
        int o = 0;
        for (int i = 0; i < kk; ++i)
            for (int j = 0; j < kk; ++j)
                if (top_h[i] == top_u[j]) { ++o; break; }
        return o;
    };
    const int o5  = overlap(5);
    const int o10 = overlap(10);

    // Numeric error metrics (full vocab).
    double sum_sq = 0.0, sum_abs = 0.0, max_abs = 0.0;
    double dot = 0.0, hh = 0.0, uu = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        const double d = (double) hand_logits[v] - (double) upstream_logits[v];
        sum_sq  += d * d;
        sum_abs += std::fabs(d);
        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
        dot += (double) hand_logits[v] * (double) upstream_logits[v];
        hh  += (double) hand_logits[v]   * (double) hand_logits[v];
        uu  += (double) upstream_logits[v] * (double) upstream_logits[v];
    }
    const double rms      = std::sqrt(sum_sq / n_vocab);
    const double mean_abs = sum_abs / n_vocab;
    const double cos_sim  = dot / std::sqrt(hh * uu);

    // Decode top-1 tokens (best-effort; skip on failure).
    auto piece = [&](int t) -> std::string {
        char buf[64] = {0};
        int n = llama_token_to_piece(vocab, t, buf, (int) sizeof(buf) - 1, 0, true);
        if (n <= 0) return std::string("?");
        return std::string(buf, buf + n);
    };

    std::fprintf(stderr, "gemma4 network_self_test results:\n");
    std::fprintf(stderr, "  hand top-1     = %d (%s)\n", top1_h, piece(top1_h).c_str());
    std::fprintf(stderr, "  upstream top-1 = %d (%s)\n", top1_u, piece(top1_u).c_str());
    std::fprintf(stderr, "  match top-1    = %s\n", top1_h == top1_u ? "YES" : "NO");
    std::fprintf(stderr, "  upstream top-1 rank in hand = %d (of %d)\n",
                 rank_top1_u_in_hand, n_vocab);
    std::fprintf(stderr, "  top-5  overlap = %d/5\n", o5);
    std::fprintf(stderr, "  top-10 overlap = %d/10\n", o10);
    std::fprintf(stderr, "  max_abs  = %.4e\n", max_abs);
    std::fprintf(stderr, "  mean_abs = %.4e\n", mean_abs);
    std::fprintf(stderr, "  RMS      = %.4e\n", rms);
    std::fprintf(stderr, "  cos_sim  = %.6f\n", cos_sim);

    std::fprintf(stderr, "  upstream top-10:");
    for (int i = 0; i < 10; ++i)
        std::fprintf(stderr, " %d(%.2f)", top_u[i], (double) upstream_logits[top_u[i]]);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "  hand     top-10:");
    for (int i = 0; i < 10; ++i)
        std::fprintf(stderr, " %d(%.2f)", top_h[i], (double) hand_logits[top_h[i]]);
    std::fprintf(stderr, "\n");

    if (top1_h != top1_u) {
        error = "network_self_test: top-1 token mismatch";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------
// G3.4b -- greedy decode with persistent KV cache
// ---------------------------------------------------------------------

bool network_state_reserve(NetworkState & s, const ModelF32 & m,
                           int cap_seq, std::string & error) {
    if (cap_seq <= 0) { error = "network_state_reserve: cap_seq<=0"; return false; }
    s.K_cache.assign(m.n_layer, {});
    s.V_cache.assign(m.n_layer, {});
    s.pos_all.clear();
    s.pos_all.reserve((size_t) cap_seq);
    s.n_past  = 0;
    s.cap_seq = cap_seq;
    for (int il = 0; il < m.n_layer; ++il) {
        const LayerF32 & L = m.layers[il];
        if (L.kv_reuse_il >= 0) continue;     // shared-KV layer: no own storage
        const int n_kv = L.n_head_kv * L.head_dim;
        s.K_cache[il].assign((size_t) n_kv * cap_seq, 0.0f);
        s.V_cache[il].assign((size_t) n_kv * cap_seq, 0.0f);
    }
    return true;
}

bool network_step(NetworkState & s, const ModelF32 & m,
                  int n_new,
                  const int32_t * token_ids,
                  bool last_token_only,
                  float * logits_out,
                  std::string & error) {
    if (n_new <= 0) { error = "network_step: n_new<=0"; return false; }
    if (s.cap_seq <= 0) { error = "network_step: state not reserved"; return false; }
    if (s.n_past + n_new > s.cap_seq) {
        std::ostringstream ss;
        ss << "network_step: n_past+n_new=" << (s.n_past+n_new)
           << " exceeds cap_seq=" << s.cap_seq;
        error = ss.str();
        return false;
    }
    if (!m.tok_embd_quant) { error = "network_step: tok_embd not set"; return false; }
    if (m.output_norm.empty()) { error = "network_step: output_norm empty"; return false; }
    if (!m.output_tied_to_embd) {
        error = "network_step: untied output not supported";
        return false;
    }
    const int n_embd  = m.n_embd;
    const int n_vocab = m.n_vocab;
    const int n_epl   = m.n_embd_per_layer;
    const float eps   = m.rms_eps;

    // Append positions [n_past .. n_past+n_new) to pos_all.
    for (int i = 0; i < n_new; ++i) s.pos_all.push_back(s.n_past + i);
    const int n_total = s.n_past + n_new;

    // ---- 1. Input embedding * sqrt(n_embd) ----
    std::vector<float> inpL((size_t) n_embd * n_new, 0.0f);
    const float emb_scale = std::sqrt((float) n_embd);
    for (int t = 0; t < n_new; ++t) {
        const int tok = token_ids[t];
        if (tok < 0 || tok >= n_vocab) {
            std::ostringstream ss;
            ss << "network_step: token_ids[" << t << "]=" << tok << " out of range";
            error = ss.str();
            return false;
        }
        dequant_row(m.tok_embd_quant, tok, inpL.data() + (size_t) t * n_embd);
        for (int e = 0; e < n_embd; ++e) inpL[(size_t) t * n_embd + e] *= emb_scale;
    }

    // ---- 2. PLE preprocessing on the n_new new tokens ----
    std::vector<float> per_layer_final;
    if (!compute_per_layer_inputs(m, n_new, token_ids, inpL.data(),
                                  per_layer_final, error)) return false;

    // ---- 3. Layer loop with cached K/V ----
    std::vector<float> hidden_out((size_t) n_embd * n_new, 0.0f);
    for (int il = 0; il < m.n_layer; ++il) {
        const LayerF32 & L = m.layers[il];
        const float * slice = per_layer_final.data()
                            + (size_t) il * n_new * n_epl;

        float * K_buf = nullptr;
        float * V_buf = nullptr;
        bool reuse = false;
        if (L.kv_reuse_il < 0) {
            K_buf = s.K_cache[il].data();
            V_buf = s.V_cache[il].data();
        } else {
            const int src = L.kv_reuse_il;
            if (s.K_cache[src].empty() || s.V_cache[src].empty()) {
                std::ostringstream ss;
                ss << "network_step: layer " << il << " reuses KV from layer "
                   << src << " but source has empty storage";
                error = ss.str();
                return false;
            }
            K_buf = s.K_cache[src].data();
            V_buf = s.V_cache[src].data();
            reuse = true;
        }

        if (!layer_forward_f32_cached(L, n_new, s.n_past, m.n_swa,
                                      inpL.data(), s.pos_all.data(), slice,
                                      hidden_out.data(),
                                      K_buf, V_buf, reuse, error)) {
            std::ostringstream ss;
            ss << "network_step: layer " << il << ": " << error;
            error = ss.str();
            return false;
        }
        std::swap(inpL, hidden_out);
    }

    // ---- 4. Final output norm on n_new tokens ----
    for (int t = 0; t < n_new; ++t) {
        rmsnorm_mul_f32(hidden_out.data() + (size_t) t * n_embd,
                        inpL.data() + (size_t) t * n_embd,
                        m.output_norm.data(), n_embd, eps);
    }
    std::swap(inpL, hidden_out);

    // ---- 5. lm_head (tied to tok_embd) ----
    const int t_start = last_token_only ? n_new - 1 : 0;
    const int t_count = last_token_only ? 1 : n_new;
    std::vector<float> row(n_embd);
    for (int v = 0; v < n_vocab; ++v) {
        dequant_row(m.tok_embd_quant, v, row.data());
        for (int tc = 0; tc < t_count; ++tc) {
            const int t = t_start + tc;
            const float * xt = inpL.data() + (size_t) t * n_embd;
            double s = 0.0;
            for (int e = 0; e < n_embd; ++e) s += (double) row[e] * (double) xt[e];
            logits_out[(size_t) tc * n_vocab + v] = (float) s;
        }
    }

    // ---- 6. Final logit softcap ----
    if (m.final_logit_softcap > 0.0f) {
        const float cap = m.final_logit_softcap;
        const float inv = 1.0f / cap;
        const size_t total = (size_t) n_vocab * t_count;
        for (size_t i = 0; i < total; ++i) {
            logits_out[i] = cap * std::tanh(logits_out[i] * inv);
        }
    }

    s.n_past += n_new;
    return true;
}

bool network_gen_self_test(const llama_model * model, const Weights & w,
                           const std::string & prompt, int n_gen,
                           int n_threads, std::string & error) {
    if (n_threads <= 0) n_threads = 1;
    if (n_gen <= 0)     { error = "network_gen_self_test: n_gen<=0"; return false; }

    // -------- Tokenize prompt --------
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<int32_t> prompt_tokens;
    prompt_tokens.resize(prompt.size() + 8);
    const int n_prompt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                                        prompt_tokens.data(), (int) prompt_tokens.size(),
                                        /*add_special=*/true, /*parse_special=*/true);
    if (n_prompt < 0) { error = "network_gen_self_test: tokenize failed"; return false; }
    prompt_tokens.resize(n_prompt);

    std::fprintf(stderr, "gemma4 network_gen_self_test: prompt=\"%s\" -> %d tokens, n_gen=%d\n",
                 prompt.c_str(), n_prompt, n_gen);

    // -------- Dequant + reserve state --------
    ModelF32 mf;
    if (!dequant_model(model, w, mf, error)) return false;
    NetworkState st;
    const int cap = n_prompt + n_gen + 4;
    if (!network_state_reserve(st, mf, cap, error)) return false;

    // -------- Hand path: prefill + greedy decode loop --------
    std::vector<int32_t> hand_gen;
    hand_gen.reserve(n_gen);
    std::vector<float> logits((size_t) n_vocab, 0.0f);
    const auto t0 = std::chrono::steady_clock::now();
    if (!network_step(st, mf, n_prompt, prompt_tokens.data(),
                      /*last_token_only=*/true, logits.data(), error)) {
        return false;
    }
    int next = (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    hand_gen.push_back(next);
    for (int g = 1; g < n_gen; ++g) {
        int32_t tok = next;
        if (!network_step(st, mf, 1, &tok, /*last_token_only=*/true,
                          logits.data(), error)) return false;
        next = (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        hand_gen.push_back(next);
    }
    const double hand_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // -------- Upstream path: persistent llama_context, greedy decode --------
    const int n_ctx = n_prompt + n_gen + 32;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_ctx;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf         = true;
    llama_context * ctx = llama_init_from_model(const_cast<llama_model *>(model), cp);
    if (!ctx) { error = "network_gen_self_test: llama_init_from_model failed"; return false; }
    llama_set_n_threads(ctx, n_threads, n_threads);

    std::vector<int32_t> up_gen;
    up_gen.reserve(n_gen);
    const auto u0 = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
    if (llama_decode(ctx, batch) != 0) {
        error = "network_gen_self_test: prompt llama_decode failed";
        llama_free(ctx); return false;
    }
    {
        const float * src = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!src) { error = "network_gen_self_test: get_logits_ith null"; llama_free(ctx); return false; }
        int up_next = (int) (std::max_element(src, src + n_vocab) - src);
        up_gen.push_back(up_next);
        for (int g = 1; g < n_gen; ++g) {
            int32_t tok = up_next;
            llama_batch one = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, one) != 0) {
                error = "network_gen_self_test: gen llama_decode failed";
                llama_free(ctx); return false;
            }
            const float * s2 = llama_get_logits_ith(ctx, -1);
            if (!s2) { error = "network_gen_self_test: get_logits_ith(-1) null"; llama_free(ctx); return false; }
            up_next = (int) (std::max_element(s2, s2 + n_vocab) - s2);
            up_gen.push_back(up_next);
        }
    }
    const double up_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - u0).count();
    llama_free(ctx);

    // -------- Compare token-by-token --------
    auto piece = [&](int t) -> std::string {
        char buf[64] = {0};
        int n = llama_token_to_piece(vocab, t, buf, (int) sizeof(buf) - 1, 0, true);
        if (n <= 0) return std::string("?");
        return std::string(buf, buf + n);
    };

    std::string hand_text, up_text;
    int matched = 0;
    int first_diverge = -1;
    for (int g = 0; g < n_gen; ++g) {
        if (hand_gen[g] == up_gen[g]) ++matched;
        else if (first_diverge < 0) first_diverge = g;
        hand_text += piece(hand_gen[g]);
        up_text   += piece(up_gen[g]);
    }

    std::fprintf(stderr, "gemma4 network_gen_self_test results:\n");
    std::fprintf(stderr, "  hand path took     %.1f ms (prefill %d + %d gen tokens)\n",
                 hand_ms, n_prompt, n_gen);
    std::fprintf(stderr, "  upstream path took %.1f ms\n", up_ms);
    std::fprintf(stderr, "  match              %d/%d tokens\n", matched, n_gen);
    if (first_diverge >= 0) {
        std::fprintf(stderr, "  first divergence at gen step %d: hand=%d(%s) up=%d(%s)\n",
                     first_diverge,
                     hand_gen[first_diverge], piece(hand_gen[first_diverge]).c_str(),
                     up_gen[first_diverge],   piece(up_gen[first_diverge]).c_str());
    }
    std::fprintf(stderr, "  hand text     : \"%s\"\n", hand_text.c_str());
    std::fprintf(stderr, "  upstream text : \"%s\"\n", up_text.c_str());

    if (matched != n_gen) {
        std::ostringstream ss;
        ss << "network_gen_self_test: " << matched << "/" << n_gen << " tokens matched";
        error = ss.str();
        return false;
    }
    return true;
}

} // namespace gemma4
