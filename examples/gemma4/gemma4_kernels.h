#pragma once

// G3.2 — Gemma-4 F32 kernels for the custom forward pass.
//
// Tiny set of hand-coded primitives that mirror the ggml ops used by
// the upstream gemma4 graph. These are validated bit-close-to-ggml
// by gemma4_kernel_self_test() so a later "hand-coded layer" pass
// can compose them with confidence.
//
// Differences from examples/phi3/phi3_kernels.h:
//   * rmsnorm_mul_f32 accepts a NULL weight pointer -- gemma4's V
//     projection uses ggml_rms_norm() without a multiplicative weight
//     (just normalization with eps), while attn_norm / ffn_norm /
//     q_norm / k_norm / etc. all carry a weight.
//   * gelu_f32 (gemma4 SwiGLU uses GELU, not SiLU).
//   * dequant helpers carry the same shape as phi3's.
//
// No callable allocation here -- callers own all buffers.

#include <cstdint>
#include <string>

struct ggml_type_traits;

namespace gemma4 {

// RMSNorm followed by optional elementwise multiply by a learned weight.
// If `w` is null, applies pure rms-normalization with `eps` (matches
// ggml_rms_norm with no weight, as gemma4 does for Vcur).
// `dst` and `src` may alias.
void rmsnorm_mul_f32(float * dst, const float * src, const float * w, int n, float eps);

// Standard GELU activation (ggml_gelu — tanh-approximation).
// Elementwise: dst[i] = 0.5*x*(1 + tanh(sqrt(2/PI)*(x + 0.044715*x^3))).
// `dst` and `src` may alias.
void gelu_f32(float * dst, const float * src, int n);

// NeoX-style RoPE for one head, one position.
// n_dims is the count of dims to rotate (== head_dim for gemma4).
// pos is a single integer position. freq_factors may be NULL.
// dst and src may alias.
void rope_neox_f32(float * dst, const float * src,
                   const float * freq_factors,
                   int n_dims, int head_dim,
                   int pos, float freq_base);

// Dequantize one row of length `n_embd` from a 2D weight matrix
// stored as [n_embd, n_vocab] with rows of size `row_bytes` bytes.
// Reads `traits->to_float` for the underlying type.
void dequant_row_to_f32(float * dst, const ggml_type_traits * traits,
                        const uint8_t * w_base, size_t row_bytes,
                        int row_idx, int n_embd);

// Convenience: apply rmsnorm_mul_f32 to each of `n_head` heads of
// a buffer laid out as [head_dim, n_head] (column-major in head_dim).
// Each head independently: rmsnorm(head, w_per_head_dim).
// `w` has length head_dim and is shared across heads (gemma4 Q/K norms).
void rmsnorm_per_head_f32(float * dst, const float * src, const float * w,
                          int head_dim, int n_head, float eps);

// Run a small battery of unit-tests comparing each kernel against the
// equivalent ggml graph for the same input. On failure, returns false
// and writes the offending tag/diag to `error`. Prints one OK line per
// test to stderr on success.
bool kernel_self_test(std::string & error);

} // namespace gemma4
