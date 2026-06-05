#include "gemma4_kernels.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <vector>

namespace gemma4 {

// ---------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------

void rmsnorm_mul_f32(float * dst, const float * src, const float * w, int n, float eps) {
    // sum_sq accumulates in double to match ggml_float in
    // ggml-cpu/ops.cpp (rms_norm reduction).
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (double) src[i] * (double) src[i];
    }
    const float mean  = (float) (sum / (double) n);
    const float scale = 1.0f / std::sqrt(mean + eps);
    if (w) {
        for (int i = 0; i < n; ++i) {
            dst[i] = src[i] * scale * w[i];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            dst[i] = src[i] * scale;
        }
    }
}

void rmsnorm_per_head_f32(float * dst, const float * src, const float * w,
                          int head_dim, int n_head, float eps) {
    for (int h = 0; h < n_head; ++h) {
        rmsnorm_mul_f32(dst + (size_t) h * head_dim,
                        src + (size_t) h * head_dim,
                        w, head_dim, eps);
    }
}

void gelu_f32(float * dst, const float * src, int n) {
    // ggml_gelu uses the tanh approximation. ggml-cpu uses a lookup
    // table built from this exact formula, so we match it directly:
    //   0.5 * x * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
    const float kSqrt2OverPi = 0.79788456080286535587989211986876f;
    const float kC = 0.044715f;
    for (int i = 0; i < n; ++i) {
        const float x = src[i];
        const float inner = kSqrt2OverPi * (x + kC * x * x * x);
        dst[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void rope_neox_f32(float * dst, const float * src,
                   const float * freq_factors,
                   int n_dims, int head_dim,
                   int pos, float freq_base) {
    const float theta_scale = std::pow(freq_base, -2.0f / (float) n_dims);
    float theta = (float) pos;
    const int half = n_dims / 2;
    for (int i0 = 0; i0 < n_dims; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        const float cos_t = std::cos(theta / ff);
        const float sin_t = std::sin(theta / ff);
        const int ic = i0 / 2;
        const float x0 = src[ic];
        const float x1 = src[ic + half];
        dst[ic]        = x0 * cos_t - x1 * sin_t;
        dst[ic + half] = x0 * sin_t + x1 * cos_t;
        theta *= theta_scale;
    }
    for (int i = n_dims; i < head_dim; ++i) {
        dst[i] = src[i];
    }
}

void dequant_row_to_f32(float * dst, const ggml_type_traits * traits,
                        const uint8_t * w_base, size_t row_bytes,
                        int row_idx, int n_embd) {
    const uint8_t * row = w_base + (size_t) row_idx * row_bytes;
    traits->to_float(row, dst, n_embd);
}

// ---------------------------------------------------------------------
// Self-tests.
// ---------------------------------------------------------------------

namespace {

bool kernel_close(const float * a, const float * b, int n,
                  float atol, float rtol,
                  std::string & error, const char * tag) {
    for (int i = 0; i < n; ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float ref  = std::fabs(b[i]);
        if (diff > atol && diff > rtol * ref) {
            std::ostringstream oss;
            oss << "gemma4 kernel self-test (" << tag << "): mismatch at i=" << i
                << " ours=" << a[i] << " ggml=" << b[i] << " diff=" << diff;
            error = oss.str();
            return false;
        }
    }
    return true;
}

bool run_ggml_graph(ggml_context * ctx, ggml_tensor * result, float * out, int n_elts) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        return false;
    }
    std::memcpy(out, result->data, (size_t) n_elts * sizeof(float));
    return true;
}

bool test_rmsnorm_with_weight(std::string & error) {
    // Use n_embd = 1536 (matches E2B).
    const int n = 1536;
    const float eps = 1e-6f;
    std::mt19937 rng(0xA11CEu);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> src(n), w(n);
    for (auto & v : src) v = uni(rng);
    for (auto & v : w)   v = uni(rng) * 0.5f + 1.0f;

    std::vector<float> ours(n);
    rmsnorm_mul_f32(ours.data(), src.data(), w.data(), n, eps);

    ggml_init_params ip{ 16ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "rmsnorm_with_weight: ggml_init failed"; return false; }
    ggml_tensor * tsrc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * tw   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    std::memcpy(tsrc->data, src.data(), n * sizeof(float));
    std::memcpy(tw->data,   w.data(),   n * sizeof(float));
    ggml_tensor * tout = ggml_mul(ctx, ggml_rms_norm(ctx, tsrc, eps), tw);
    std::vector<float> oracle(n);
    bool ok = run_ggml_graph(ctx, tout, oracle.data(), n);
    ggml_free(ctx);
    if (!ok) { error = "rmsnorm_with_weight: graph_compute failed"; return false; }
    return kernel_close(ours.data(), oracle.data(), n, 1e-4f, 1e-4f, error, "rmsnorm_with_weight");
}

bool test_rmsnorm_no_weight(std::string & error) {
    // gemma4 calls ggml_rms_norm() on Vcur without a multiplicative weight.
    // Use head_dim = 256 (matches gemma4 SWA layer).
    const int n = 256;
    const float eps = 1e-6f;
    std::mt19937 rng(0xBEEFu);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> src(n);
    for (auto & v : src) v = uni(rng);

    std::vector<float> ours(n);
    rmsnorm_mul_f32(ours.data(), src.data(), nullptr, n, eps);

    ggml_init_params ip{ 4ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "rmsnorm_no_weight: ggml_init failed"; return false; }
    ggml_tensor * tsrc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    std::memcpy(tsrc->data, src.data(), n * sizeof(float));
    ggml_tensor * tout = ggml_rms_norm(ctx, tsrc, eps);
    std::vector<float> oracle(n);
    bool ok = run_ggml_graph(ctx, tout, oracle.data(), n);
    ggml_free(ctx);
    if (!ok) { error = "rmsnorm_no_weight: graph_compute failed"; return false; }
    return kernel_close(ours.data(), oracle.data(), n, 1e-5f, 1e-5f, error, "rmsnorm_no_weight");
}

bool test_rmsnorm_per_head(std::string & error) {
    // gemma4 q_norm/k_norm applies the same [head_dim] weight to each head
    // of a [head_dim, n_head] (or [head_dim, n_head_kv]) buffer.
    const int head_dim = 256;
    const int n_head   = 8;
    const float eps = 1e-6f;
    std::mt19937 rng(0xCAFEu);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> src((size_t) head_dim * n_head), w(head_dim);
    for (auto & v : src) v = uni(rng);
    for (auto & v : w)   v = uni(rng) * 0.5f + 1.0f;

    // ours
    std::vector<float> ours((size_t) head_dim * n_head);
    rmsnorm_per_head_f32(ours.data(), src.data(), w.data(), head_dim, n_head, eps);

    // oracle: ggml_rms_norm on a 2D tensor [head_dim, n_head] -- ggml
    // normalizes along the innermost dim (head_dim) per head, then mul by
    // a [head_dim] weight broadcasts across the n_head dim. This matches
    // exactly what gemma4 does in its graph (Qcur reshaped to
    // [n_embd_head, n_head, n_tokens] then build_norm with q_norm).
    ggml_init_params ip{ 16ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "rmsnorm_per_head: ggml_init failed"; return false; }
    ggml_tensor * tsrc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, n_head);
    ggml_tensor * tw   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    std::memcpy(tsrc->data, src.data(), src.size() * sizeof(float));
    std::memcpy(tw->data,   w.data(),   w.size()   * sizeof(float));
    ggml_tensor * tout = ggml_mul(ctx, ggml_rms_norm(ctx, tsrc, eps), tw);
    std::vector<float> oracle((size_t) head_dim * n_head);
    bool ok = run_ggml_graph(ctx, tout, oracle.data(), head_dim * n_head);
    ggml_free(ctx);
    if (!ok) { error = "rmsnorm_per_head: graph_compute failed"; return false; }
    return kernel_close(ours.data(), oracle.data(), head_dim * n_head, 1e-4f, 1e-4f, error, "rmsnorm_per_head");
}

bool test_gelu(std::string & error) {
    // ggml-cpu's gelu uses a 64KB lookup table built from the same
    // tanh-approximation formula we use. The lookup is on F16-quantized
    // x so we allow a few ULPs of slop.
    const int n = 1024;
    std::mt19937 rng(0xF00Du);
    std::uniform_real_distribution<float> uni(-3.0f, 3.0f);  // gelu interesting range
    std::vector<float> src(n);
    for (auto & v : src) v = uni(rng);

    std::vector<float> ours(n);
    gelu_f32(ours.data(), src.data(), n);

    ggml_init_params ip{ 4ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "gelu: ggml_init failed"; return false; }
    ggml_tensor * tsrc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    std::memcpy(tsrc->data, src.data(), n * sizeof(float));
    ggml_tensor * tout = ggml_gelu(ctx, tsrc);
    std::vector<float> oracle(n);
    bool ok = run_ggml_graph(ctx, tout, oracle.data(), n);
    ggml_free(ctx);
    if (!ok) { error = "gelu: graph_compute failed"; return false; }
    // ggml_gelu uses a 16-bit table lookup; tolerance reflects that.
    return kernel_close(ours.data(), oracle.data(), n, 1e-3f, 1e-3f, error, "gelu");
}

bool test_rope_neox(const char * tag, bool with_factors, std::string & error) {
    // Two configurations mirroring gemma4:
    //   SWA layers: head_dim=256, freq_base=1e4
    //   Full layers: head_dim=512, freq_base=1e6 (typically with freq_factors)
    const int head_dim = with_factors ? 512 : 256;
    const int n_dims   = head_dim;  // gemma4 rotates the whole head_dim
    const int n_head   = 2;
    const int pos      = 5;
    const float freq_base = with_factors ? 1e6f : 1e4f;

    std::mt19937 rng(0xC0DEu + (with_factors ? 1u : 0u));
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> src((size_t) head_dim * n_head);
    for (auto & v : src) v = uni(rng);
    std::vector<float> factors(n_dims / 2);
    for (auto & v : factors) v = 0.5f + 0.5f * uni(rng);
    const float * ff = with_factors ? factors.data() : nullptr;

    std::vector<float> ours((size_t) head_dim * n_head);
    for (int h = 0; h < n_head; ++h) {
        rope_neox_f32(ours.data() + (size_t) h * head_dim,
                      src .data() + (size_t) h * head_dim,
                      ff, n_dims, head_dim, pos, freq_base);
    }

    ggml_init_params ip{ 8ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = std::string("rope ") + tag + ": ggml_init failed"; return false; }
    ggml_tensor * tsrc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head, 1);
    std::memcpy(tsrc->data, src.data(), src.size() * sizeof(float));
    ggml_tensor * tpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    *(int32_t *) tpos->data = pos;
    ggml_tensor * tff = nullptr;
    if (with_factors) {
        tff = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims / 2);
        std::memcpy(tff->data, factors.data(), factors.size() * sizeof(float));
    }
    ggml_tensor * tout = ggml_rope_ext(ctx, tsrc, tpos, tff,
        n_dims, GGML_ROPE_TYPE_NEOX, 0,
        freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    std::vector<float> oracle((size_t) head_dim * n_head);
    bool ok = run_ggml_graph(ctx, tout, oracle.data(), head_dim * n_head);
    ggml_free(ctx);
    if (!ok) { error = std::string("rope ") + tag + ": graph_compute failed"; return false; }
    return kernel_close(ours.data(), oracle.data(), head_dim * n_head, 1e-4f, 1e-4f, error, tag);
}

bool test_dequant_row(std::string & error) {
    // Q4_K row dequant: matches what we'll do for token_embd at runtime.
    // Use n_embd = 256 (multiple of QK_K so K-quant block boundaries align).
    const int n_embd  = 256;
    const int n_vocab = 8;
    const int probe[] = { 0, 3, 5, 7 };

    std::mt19937 rng(0xB033Du);
    std::uniform_real_distribution<float> uni(-0.1f, 0.1f);
    std::vector<float> ref_full((size_t) n_embd * n_vocab);
    for (auto & v : ref_full) v = uni(rng);

    const ggml_type type = GGML_TYPE_Q4_K;
    const auto * trc = ggml_get_type_traits_cpu(type);
    if (!trc || !trc->from_float) {
        error = "dequant_row: Q4_K from_float not registered";
        return false;
    }
    ggml_init_params ip{ 4ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "dequant_row: ggml_init failed"; return false; }
    ggml_tensor * tw = ggml_new_tensor_2d(ctx, type, n_embd, n_vocab);
    const size_t row_bytes = ggml_row_size(type, n_embd);
    for (int v = 0; v < n_vocab; ++v) {
        trc->from_float(ref_full.data() + (size_t) v * n_embd,
                        (uint8_t *) tw->data + (size_t) v * row_bytes,
                        n_embd);
    }

    const int n_probe = (int) (sizeof(probe) / sizeof(probe[0]));
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_probe);
    int32_t * idx_data = (int32_t *) idx->data;
    for (int i = 0; i < n_probe; ++i) idx_data[i] = probe[i];
    ggml_tensor * tout = ggml_get_rows(ctx, tw, idx);
    std::vector<float> oracle((size_t) n_embd * n_probe);
    bool ok = run_ggml_graph(ctx, tout, oracle.data(), n_embd * n_probe);
    if (!ok) { ggml_free(ctx); error = "dequant_row: graph_compute failed"; return false; }

    std::vector<float> ours((size_t) n_embd * n_probe);
    const auto * tr = ggml_get_type_traits(type);
    for (int i = 0; i < n_probe; ++i) {
        dequant_row_to_f32(ours.data() + (size_t) i * n_embd, tr,
                           (const uint8_t *) tw->data, row_bytes, probe[i], n_embd);
    }
    ggml_free(ctx);
    // K-quant dequant is deterministic; allow tiny slop for SIMD path differences.
    return kernel_close(ours.data(), oracle.data(), n_embd * n_probe, 1e-5f, 1e-5f, error, "dequant_row_q4_K");
}

} // namespace

bool kernel_self_test(std::string & error) {
    if (!test_rmsnorm_with_weight(error)) return false;
    std::fprintf(stderr, "gemma4 kernel self-test: rmsnorm + weight                OK\n");
    if (!test_rmsnorm_no_weight(error))   return false;
    std::fprintf(stderr, "gemma4 kernel self-test: rmsnorm (no weight, V path)     OK\n");
    if (!test_rmsnorm_per_head(error))    return false;
    std::fprintf(stderr, "gemma4 kernel self-test: rmsnorm per-head (q/k norm)     OK\n");
    if (!test_gelu(error))                return false;
    std::fprintf(stderr, "gemma4 kernel self-test: gelu (tanh approx)              OK\n");
    if (!test_rope_neox("rope_neox_swa", false, error)) return false;
    std::fprintf(stderr, "gemma4 kernel self-test: rope_neox SWA (head_dim=256)    OK\n");
    if (!test_rope_neox("rope_neox_full_with_factors", true, error)) return false;
    std::fprintf(stderr, "gemma4 kernel self-test: rope_neox full (factors, hd=512) OK\n");
    if (!test_dequant_row(error))         return false;
    std::fprintf(stderr, "gemma4 kernel self-test: dequant_row Q4_K                OK\n");
    std::fprintf(stderr, "gemma4 kernel self-test: PASS\n");
    error.clear();
    return true;
}

} // namespace gemma4
