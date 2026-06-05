#include "gemma4_forward.h"
#include "gemma4_kernels.h"
#include "gemma4_weights.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
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
    if (!L.has_kv) {
        error = "dequant_layer: layer has no own KV (shared-KV layer not supported by G3.3)";
        return false;
    }

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
    // gemma4 rotates all head_dim dims per layer. The global rope_dim
    // metadata field reflects the FULL-attn head_dim; SWA layers have a
    // smaller head_dim and would crash if we tried to rotate beyond it.
    out.rope_dim  = L.head_dim;

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
    if (!dequant_f32(L.wk,       out.wk,       error)) return false;
    if (L.has_v_proj) {
        if (!dequant_f32(L.wv,   out.wv,       error)) return false;
    }
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
// Hand-coded F32 layer forward
// ---------------------------------------------------------------------

bool layer_forward_f32(const LayerF32 & L,
                       int n_tokens,
                       const float * hidden_in,
                       const int32_t * pos,
                       const float * per_layer_input,
                       float * hidden_out,
                       std::string & error) {
    if (n_tokens <= 0) { error = "layer_forward_f32: n_tokens<=0"; return false; }
    const int n_embd    = L.n_embd;
    const int n_head    = L.n_head;
    const int n_head_kv = L.n_head_kv;
    const int head_dim  = L.head_dim;
    const int n_ff      = L.n_ff;
    const int n_epl     = L.n_embd_per_layer;
    const int n_q       = n_head * head_dim;
    const int n_kv      = n_head_kv * head_dim;
    const float eps     = L.rms_eps;

    // -------- attn_norm: norm1[t,:] = rmsnorm(hidden_in[t,:]) * attn_norm --
    std::vector<float> norm1((size_t) n_embd * n_tokens, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        rmsnorm_mul_f32(norm1.data() + (size_t) t * n_embd,
                        hidden_in + (size_t) t * n_embd,
                        L.attn_norm.data(), n_embd, eps);
    }

    // -------- Q = wq @ norm1 --------
    std::vector<float> Q((size_t) n_q * n_tokens, 0.0f);
    matmul_f32(L.wq.data(), norm1.data(), Q.data(), n_embd, n_q, n_tokens);

    // -------- K = wk @ norm1 --------
    std::vector<float> K((size_t) n_kv * n_tokens, 0.0f);
    matmul_f32(L.wk.data(), norm1.data(), K.data(), n_embd, n_kv, n_tokens);

    // -------- V = wv @ norm1 (or = K if wv missing) --------
    std::vector<float> V;
    if (!L.wv.empty()) {
        V.assign((size_t) n_kv * n_tokens, 0.0f);
        matmul_f32(L.wv.data(), norm1.data(), V.data(), n_embd, n_kv, n_tokens);
    } else {
        V = K;
    }

    // -------- Q norm (per head, weight = attn_q_norm[head_dim]) --------
    for (int t = 0; t < n_tokens; ++t) {
        rmsnorm_per_head_f32(Q.data() + (size_t) t * n_q,
                             Q.data() + (size_t) t * n_q,
                             L.attn_q_norm.data(), head_dim, n_head, eps);
    }
    // -------- K norm (per kv-head, weight = attn_k_norm[head_dim]) --------
    for (int t = 0; t < n_tokens; ++t) {
        rmsnorm_per_head_f32(K.data() + (size_t) t * n_kv,
                             K.data() + (size_t) t * n_kv,
                             L.attn_k_norm.data(), head_dim, n_head_kv, eps);
    }
    // -------- V norm: pure rms (no weight), per kv-head, gemma4 quirk -----
    for (int t = 0; t < n_tokens; ++t) {
        rmsnorm_per_head_f32(V.data() + (size_t) t * n_kv,
                             V.data() + (size_t) t * n_kv,
                             /*w=*/nullptr, head_dim, n_head_kv, eps);
    }

    // -------- RoPE on Q and K --------
    for (int t = 0; t < n_tokens; ++t) {
        const int p = pos[t];
        for (int h = 0; h < n_head; ++h) {
            float * q_th = Q.data() + (size_t) t * n_q + (size_t) h * head_dim;
            rope_neox_f32(q_th, q_th, L.freq_factors,
                          L.rope_dim, head_dim, p, L.rope_base);
        }
        for (int h = 0; h < n_head_kv; ++h) {
            float * k_th = K.data() + (size_t) t * n_kv + (size_t) h * head_dim;
            rope_neox_f32(k_th, k_th, L.freq_factors,
                          L.rope_dim, head_dim, p, L.rope_base);
        }
    }

    // -------- Self-attention (no past KV; n_tokens new tokens, causal). --
    // scale = 1.0 (gemma4 hparams.f_attention_scale = 1.0).
    // attn_ctx[t, h, d] = sum_k softmax_k( QK^T[t,k] over k<=t ) * V[k, h', d]
    // where h' = h * n_head_kv / n_head (GQA broadcast).
    std::vector<float> attn_ctx((size_t) n_q * n_tokens, 0.0f);
    std::vector<float> scores((size_t) n_tokens, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        const int p_t = pos[t];
        for (int h = 0; h < n_head; ++h) {
            const int h_kv = h * n_head_kv / n_head;
            const float * q_th = Q.data() + (size_t) t * n_q + (size_t) h * head_dim;
            // scores[k] = q . K[k, h_kv, :]
            float max_s = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < n_tokens; ++k) {
                // causal: token at pos[t] attends to pos[k] only if pos[k] <= pos[t]
                if (pos[k] > p_t) {
                    scores[k] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float * k_th = K.data() + (size_t) k * n_kv + (size_t) h_kv * head_dim;
                double s = 0.0;
                for (int d = 0; d < head_dim; ++d) s += (double) q_th[d] * (double) k_th[d];
                const float sf = (float) s;
                scores[k] = sf;
                if (sf > max_s) max_s = sf;
            }
            // softmax (no extra scale; gemma4 attention scale = 1.0)
            double sum = 0.0;
            for (int k = 0; k < n_tokens; ++k) {
                if (scores[k] == -std::numeric_limits<float>::infinity()) {
                    scores[k] = 0.0f;
                } else {
                    scores[k] = std::exp(scores[k] - max_s);
                    sum += (double) scores[k];
                }
            }
            const float inv_sum = sum > 0.0 ? (float) (1.0 / sum) : 0.0f;
            for (int k = 0; k < n_tokens; ++k) scores[k] *= inv_sum;
            // attn_ctx[t, h, :] = sum_k scores[k] * V[k, h_kv, :]
            float * out_th = attn_ctx.data() + (size_t) t * n_q + (size_t) h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (int k = 0; k < n_tokens; ++k) {
                    const float * v_th = V.data() + (size_t) k * n_kv + (size_t) h_kv * head_dim;
                    acc += (double) scores[k] * (double) v_th[d];
                }
                out_th[d] = (float) acc;
            }
        }
    }

    // -------- wo: attn_out = wo @ attn_ctx --------
    std::vector<float> attn_out((size_t) n_embd * n_tokens, 0.0f);
    matmul_f32(L.wo.data(), attn_ctx.data(), attn_out.data(), n_q, n_embd, n_tokens);

    // -------- post_attn_norm + residual1: attn_out2 = post_norm(attn_out) + hidden_in --
    std::vector<float> attn_out2((size_t) n_embd * n_tokens, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        rmsnorm_mul_f32(attn_out2.data() + (size_t) t * n_embd,
                        attn_out.data() + (size_t) t * n_embd,
                        L.post_attn_norm.data(), n_embd, eps);
    }
    for (size_t i = 0; i < (size_t) n_embd * n_tokens; ++i) {
        attn_out2[i] += hidden_in[i];
    }

    // -------- ffn_norm --------
    std::vector<float> ff_in((size_t) n_embd * n_tokens, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        rmsnorm_mul_f32(ff_in.data() + (size_t) t * n_embd,
                        attn_out2.data() + (size_t) t * n_embd,
                        L.ffn_norm.data(), n_embd, eps);
    }

    // -------- gate, up, gelu(gate)*up --------
    std::vector<float> gate((size_t) n_ff * n_tokens, 0.0f);
    std::vector<float> up  ((size_t) n_ff * n_tokens, 0.0f);
    matmul_f32(L.ffn_gate.data(), ff_in.data(), gate.data(), n_embd, n_ff, n_tokens);
    matmul_f32(L.ffn_up.data(),   ff_in.data(), up.data(),   n_embd, n_ff, n_tokens);
    gelu_f32(gate.data(), gate.data(), n_ff * n_tokens);
    for (size_t i = 0; i < (size_t) n_ff * n_tokens; ++i) gate[i] *= up[i];

    // -------- ffn_down --------
    std::vector<float> ff_out((size_t) n_embd * n_tokens, 0.0f);
    matmul_f32(L.ffn_down.data(), gate.data(), ff_out.data(), n_ff, n_embd, n_tokens);

    // -------- post_ffw_norm + residual2 --------
    std::vector<float> pe_in((size_t) n_embd * n_tokens, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        rmsnorm_mul_f32(pe_in.data() + (size_t) t * n_embd,
                        ff_out.data() + (size_t) t * n_embd,
                        L.post_ffw_norm.data(), n_embd, eps);
    }
    for (size_t i = 0; i < (size_t) n_embd * n_tokens; ++i) {
        pe_in[i] += attn_out2[i];
    }

    // -------- PLE: cur = pe_in + post_norm(proj(gelu(inp_gate @ pe_in) * slice)) --
    std::vector<float> ple_a((size_t) n_epl * n_tokens, 0.0f);
    matmul_f32(L.inp_gate.data(), pe_in.data(), ple_a.data(), n_embd, n_epl, n_tokens);
    gelu_f32(ple_a.data(), ple_a.data(), n_epl * n_tokens);
    for (size_t i = 0; i < (size_t) n_epl * n_tokens; ++i) {
        ple_a[i] *= per_layer_input[i];
    }
    std::vector<float> ple_b((size_t) n_embd * n_tokens, 0.0f);
    matmul_f32(L.proj.data(), ple_a.data(), ple_b.data(), n_epl, n_embd, n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        rmsnorm_mul_f32(ple_b.data() + (size_t) t * n_embd,
                        ple_b.data() + (size_t) t * n_embd,
                        L.post_norm.data(), n_embd, eps);
    }
    for (size_t i = 0; i < (size_t) n_embd * n_tokens; ++i) {
        pe_in[i] += ple_b[i];
    }

    // -------- layer_output_scale --------
    if (L.has_layer_output_scale) {
        const float s = L.layer_output_scale;
        for (size_t i = 0; i < (size_t) n_embd * n_tokens; ++i) {
            pe_in[i] *= s;
        }
    }

    std::memcpy(hidden_out, pe_in.data(), (size_t) n_embd * n_tokens * sizeof(float));
    return true;
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

} // namespace gemma4
