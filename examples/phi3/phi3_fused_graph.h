#pragma once

// Phi3 custom-forward scaffolding (Phase A).
//
// This header declares the Phi3Weights resolver — the foundation for the
// custom Phi-3 forward pass that bypasses `llama_decode` on gen steps.
//
// A0 (this commit) only resolves and validates the model tensors; no forward
// pass is built yet. The actual `phi3_custom_decode` / `phi3_custom_prefill`
// functions land in A2.
//
// References (for future readers):
//  - src/models/phi3.cpp: standard Phi3 graph implementation (the oracle).
//  - include/llama_b612.h: llama_model_get_tensor_by_name (used for lookup).

#include "llama.h"

#include <cstddef>
#include <string>

struct ggml_tensor;

// Per-layer tensor pointers. Borrowed from the model — do NOT free.
struct Phi3LayerWeights {
    const ggml_tensor * attn_norm = nullptr; // [n_embd]              F32
    const ggml_tensor * wqkv      = nullptr; // [n_embd, n_qkv]       K-quant (Q4_K_M)
    const ggml_tensor * wo        = nullptr; // [n_embd, n_embd]      K-quant
    const ggml_tensor * ffn_norm  = nullptr; // [n_embd]              F32
    const ggml_tensor * ffn_up    = nullptr; // [n_embd, 2 * n_ff]    K-quant   gate|up fused
    const ggml_tensor * ffn_down  = nullptr; // [n_ff, n_embd]        K-quant
};

// Global tensors + per-layer slots. Borrowed from the model.
struct Phi3Weights {
    // Hyper-parameters resolved from the model.
    int n_layer       = 0;
    int n_embd        = 0;
    int n_head        = 0;
    int n_head_kv     = 0;
    int n_ff          = 0;       // half-width: ffn_up is [n_embd, 2*n_ff].
    int n_embd_head   = 0;       // = n_embd / n_head.
    int n_embd_gqa    = 0;       // = n_embd_head * n_head_kv.
    int n_qkv         = 0;       // = n_embd + 2 * n_embd_gqa.
    int n_vocab       = 0;

    // Global tensors.
    const ggml_tensor * tok_embd    = nullptr; // [n_embd, n_vocab]   K-quant
    const ggml_tensor * output_norm = nullptr; // [n_embd]            F32
    const ggml_tensor * output      = nullptr; // [n_embd, n_vocab]   K-quant (may alias tok_embd if tied)

    // Optional RoPE factor tables (shared across all layers — TENSOR_DUPLICATED).
    const ggml_tensor * rope_long   = nullptr; // [n_rot/2]           F32 (may be null)
    const ggml_tensor * rope_short  = nullptr; // [n_rot/2]           F32 (may be null)

    // Did the loader tie `output` to `tok_embd`?
    bool output_tied_to_embd = false;

    // Per-layer tensors.
    static constexpr int MAX_LAYERS = 64;
    Phi3LayerWeights layers[MAX_LAYERS];
};

// Resolve every Phi3 tensor by name and validate shapes/types.
// On failure, returns false and writes a human-readable reason to `error`.
//
// What is validated:
//  - All required tensors are present (or tied, for `output`).
//  - K-quant matmul weights have a vec_dot_type of GGML_TYPE_Q8_K
//    (matches the existing phi3_weight_accepts_q8K() semantics — accepts
//    Q4_K/Q5_K/Q6_K mixes used in K_M quants).
//  - Norm tensors are F32 and have shape [n_embd].
//  - Per-layer matmul shapes match (n_embd, n_qkv), (n_embd, n_embd),
//    (n_embd, 2*n_ff), (n_ff, n_embd).
//  - Hyper-parameters are self-consistent (n_embd_head * n_head == n_embd).
//
// Pre-condition: model must already be loaded.
bool phi3_weights_resolve(const llama_model * model, Phi3Weights & out, std::string & error);

// Print the resolved tensor table to stderr (shape, type, address). Useful for
// confirming A0 worked correctly before any forward code lands.
void phi3_weights_dump(const Phi3Weights & w);


// =============================================================================
// Phi3KV — custom contiguous KV cache for Phase A custom forward.
// =============================================================================
// Owns its own F16 K/V storage, independent of llama's internal kv cache.
//
// Layout (matches what manual scaled-dot-product attention prefers):
//   K[il]  : [head_dim, n_head_kv, ctx_max]    F16
//            (innermost dim = head_dim, contiguous; matches standard layout)
//   V_T[il]: [ctx_max,  n_head_kv, head_dim]   F16
//            (TRANSPOSED so attn[1..n_kv] . V_T[:, h, d] is a contiguous
//             reduction over the n_kv dimension)
//
// Single-sequence, contiguous positions. Multi-turn rewind is supported via
// kv_truncate (O(1)), kv_drop_range (memmove), and kv_keep_prefix.
//
// Indexing helpers (do NOT bounds-check; caller is responsible):
//   K[il] at position pos, head h, dim d:
//       k_ptr[il][pos*n_head_kv*head_dim + h*head_dim + d]
//   V_T[il] at position pos, head h, dim d:
//       v_ptr[il][d*n_head_kv*ctx_max + h*ctx_max + pos]
//
// Memory budget at ctx_max=4096 for Phi-3-mini-4k (n_layer=32, head_dim=96,
// n_head_kv=32) is ~1.5 GiB total (K and V combined). Logged at init.

#include "ggml.h"   // for ggml_fp16_t

struct Phi3KV {
    int n_layer       = 0;
    int n_head_kv     = 0;
    int head_dim      = 0;
    int ctx_max       = 0;
    int current_len   = 0;   // logical number of populated positions [0, current_len)

    // Per-layer base pointers into the K_buf / V_buf flat arenas.
    ggml_fp16_t * K[Phi3Weights::MAX_LAYERS] = {};
    ggml_fp16_t * V[Phi3Weights::MAX_LAYERS] = {};

private:
    // Single backing allocations to keep alloc/free simple.
    // Sized: n_layer * head_dim * n_head_kv * ctx_max F16s each.
    ggml_fp16_t * K_buf = nullptr;
    ggml_fp16_t * V_buf = nullptr;
    friend bool   phi3_kv_init  (Phi3KV & kv, int n_layer, int n_head_kv, int head_dim, int ctx_max, std::string & error);
    friend void   phi3_kv_free  (Phi3KV & kv);
    friend void   phi3_kv_drop_range(Phi3KV & kv, int p0, int p1);
};

// Allocate K_buf / V_buf and populate per-layer base pointers.
// On failure (e.g. OOM), returns false and writes a reason to `error`.
bool   phi3_kv_init      (Phi3KV & kv, int n_layer, int n_head_kv, int head_dim, int ctx_max, std::string & error);

// Release all KV memory. Safe to call on an uninitialized struct.
void   phi3_kv_free      (Phi3KV & kv);

// Total bytes allocated for K + V across all layers.
size_t phi3_kv_bytes     (const Phi3KV & kv);

// O(1) — drop tokens at positions [new_len, current_len). Used between turns.
// new_len must be in [0, current_len].
void   phi3_kv_truncate  (Phi3KV & kv, int new_len);

// Drop the range [p0, p1) and shift positions [p1, current_len) left by (p1-p0).
// Cost: O((current_len - p1) * n_layer * 2 * row_bytes). Used for middle drops
// (rare). Caller must ensure no RoPE-position-shifting concerns (tokens after
// p1 keep their original RoPE phase, which is correct for our usage).
void   phi3_kv_drop_range(Phi3KV & kv, int p0, int p1);

// Convenience: keep_prefix(N) == truncate(N).
inline void phi3_kv_keep_prefix(Phi3KV & kv, int n_keep) { phi3_kv_truncate(kv, n_keep); }

// One-shot self-test: allocate a tiny KV using the model's dimensions, write
// known sentinels, exercise truncate / drop_range / keep_prefix, verify, free.
// Prints PASS/FAIL to stderr. Returns true on PASS.
bool   phi3_kv_self_test (const Phi3Weights & w, std::string & error);

// ---------------------------------------------------------------------------
// Phi3MatmulPool self-test — Phase A.
// Builds a small F32 weight matrix + F32 src vector, dispatches many matmul
// jobs through the pool, compares against a serial reference, and verifies
// byte-equality. Stress-tests phase-1/2/3 backoff transitions with
// deliberate inter-job sleeps. Does NOT need a model.
// Prints PASS/FAIL to stderr. Returns true on PASS.
// ---------------------------------------------------------------------------
bool   phi3_matmul_pool_self_test(int n_threads, std::string & error);

// ---------------------------------------------------------------------------
// Per-kernel self-test — Phase A.
// Validates the three small F32 kernels needed by the upcoming custom forward
// pass against the corresponding ggml ops (RMSNorm, token-embedding dequant,
// NeoX RoPE), executed on the standard ggml-cpu backend.
//
// Coverage:
//   - RMSNorm: n_embd=3072, eps=1e-5, fused with F32 weight (matches
//     llm_build_norm(LLM_NORM_RMS)).
//   - Token embedding dequant: ggml_get_rows on a synthetic F16 token-embd
//     tensor for several tokens.
//   - NeoX RoPE: head_dim=96, two heads, three positions; tested with both
//     freq_factors == NULL (Phi-3-mini-4k case) and with synthetic factors
//     (Phi-3-medium long/short rope case).
// Does NOT need a model. Prints PASS/FAIL to stderr. Returns true on PASS.
// ---------------------------------------------------------------------------
bool   phi3_kernel_self_test(std::string & error);
