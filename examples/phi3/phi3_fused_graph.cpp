#include "phi3_fused_graph.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "llama_b612.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

namespace {

// Mirror of the static helper in src/models/phi3.cpp.
//
// Accept any weight whose CPU vec-dot input type is Q8_K. That includes
// Q4_K/Q5_K/Q6_K and other K-quant family members used in Q4_K_M (per the
// duck's blocking finding #2 — exact type==Q4_K would reject valid K_M GGUFs).
bool accepts_q8K(const ggml_tensor * w) {
    if (w == nullptr) return false;
    const auto * traits = ggml_get_type_traits_cpu(w->type);
    return traits != nullptr && traits->vec_dot_type == GGML_TYPE_Q8_K;
}

// Helpers for building tensor names.
std::string layer_tensor_name(const char * leaf, int il) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s.weight", il, leaf);
    return buf;
}

const char * type_name(ggml_type t) {
    return ggml_type_name(t);
}

// Resolve a required tensor by name. On miss, sets `error` and returns nullptr.
const ggml_tensor * require_tensor(const llama_model * model, const std::string & name, std::string & error) {
    const ggml_tensor * t = llama_model_get_tensor_by_name(model, name.c_str());
    if (t == nullptr) {
        error = "phi3_weights_resolve: missing required tensor '" + name + "'";
    }
    return t;
}

// Optional tensor — null is OK.
const ggml_tensor * try_tensor(const llama_model * model, const char * name) {
    return llama_model_get_tensor_by_name(model, name);
}

// Verify a tensor is 1-D with the given size.
bool check_vec1d(const ggml_tensor * t, int64_t n, const char * what, std::string & error) {
    if (t->ne[0] != n || t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) {
        std::ostringstream oss;
        oss << "phi3_weights_resolve: " << what << " has shape ["
            << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3]
            << "], expected [" << n << ",1,1,1]";
        error = oss.str();
        return false;
    }
    return true;
}

// Verify a tensor is 2-D with the given shape.
bool check_mat2d(const ggml_tensor * t, int64_t d0, int64_t d1, const char * what, std::string & error) {
    if (t->ne[0] != d0 || t->ne[1] != d1 || t->ne[2] != 1 || t->ne[3] != 1) {
        std::ostringstream oss;
        oss << "phi3_weights_resolve: " << what << " has shape ["
            << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3]
            << "], expected [" << d0 << "," << d1 << ",1,1]";
        error = oss.str();
        return false;
    }
    return true;
}

bool check_f32_vec(const ggml_tensor * t, int64_t n, const char * what, std::string & error) {
    if (t->type != GGML_TYPE_F32) {
        std::ostringstream oss;
        oss << "phi3_weights_resolve: " << what << " has type " << type_name(t->type)
            << ", expected F32";
        error = oss.str();
        return false;
    }
    return check_vec1d(t, n, what, error);
}

bool check_q8K_mat(const ggml_tensor * t, int64_t d0, int64_t d1, const char * what, std::string & error) {
    if (!accepts_q8K(t)) {
        std::ostringstream oss;
        oss << "phi3_weights_resolve: " << what << " has type " << type_name(t->type)
            << " whose vec_dot_type is not Q8_K (cannot be used by phi3 fast path)";
        error = oss.str();
        return false;
    }
    return check_mat2d(t, d0, d1, what, error);
}

} // namespace


bool phi3_weights_resolve(const llama_model * model, Phi3Weights & out, std::string & error) {
    error.clear();
    out = Phi3Weights{};

    if (model == nullptr) {
        error = "phi3_weights_resolve: model is null";
        return false;
    }

    // Hyper-parameters.
    out.n_layer   = llama_model_n_layer(model);
    out.n_embd    = llama_model_n_embd(model);
    out.n_head    = llama_model_n_head(model);
    out.n_head_kv = llama_model_n_head_kv(model);

    if (out.n_layer <= 0 || out.n_layer > Phi3Weights::MAX_LAYERS) {
        std::ostringstream oss;
        oss << "phi3_weights_resolve: unsupported n_layer=" << out.n_layer
            << " (max " << Phi3Weights::MAX_LAYERS << ")";
        error = oss.str();
        return false;
    }
    if (out.n_head <= 0 || out.n_embd % out.n_head != 0) {
        std::ostringstream oss;
        oss << "phi3_weights_resolve: n_embd=" << out.n_embd
            << " not divisible by n_head=" << out.n_head;
        error = oss.str();
        return false;
    }
    out.n_embd_head = out.n_embd / out.n_head;
    out.n_embd_gqa  = out.n_embd_head * out.n_head_kv;
    out.n_qkv       = out.n_embd + 2 * out.n_embd_gqa;

    // n_ff derived from ffn_up later (after we resolve it).

    // Token embedding.
    out.tok_embd = require_tensor(model, "token_embd.weight", error);
    if (!out.tok_embd) return false;
    if (out.tok_embd->ne[0] != out.n_embd) {
        std::ostringstream oss;
        oss << "phi3_weights_resolve: token_embd.weight ne[0]=" << out.tok_embd->ne[0]
            << " != n_embd=" << out.n_embd;
        error = oss.str();
        return false;
    }
    out.n_vocab = (int) out.tok_embd->ne[1];

    // Output norm.
    out.output_norm = require_tensor(model, "output_norm.weight", error);
    if (!out.output_norm) return false;
    if (!check_f32_vec(out.output_norm, out.n_embd, "output_norm.weight", error)) return false;

    // Output (lm_head). May be tied to token_embd.
    out.output = try_tensor(model, "output.weight");
    if (out.output == nullptr) {
        out.output              = out.tok_embd;
        out.output_tied_to_embd = true;
    }
    if (!check_q8K_mat(out.output, out.n_embd, out.n_vocab, "output.weight", error)) return false;

    // Optional RoPE factor tables (Phi3 long-context variants only).
    out.rope_long  = try_tensor(model, "rope_factors_long.weight");
    out.rope_short = try_tensor(model, "rope_factors_short.weight");

    // Per-layer tensors. Resolve layer 0 first so we can derive n_ff from ffn_up.
    for (int il = 0; il < out.n_layer; ++il) {
        Phi3LayerWeights & L = out.layers[il];

        const std::string s_attn_norm = layer_tensor_name("attn_norm",   il);
        const std::string s_wqkv      = layer_tensor_name("attn_qkv",    il);
        const std::string s_wo        = layer_tensor_name("attn_output", il);
        const std::string s_ffn_norm  = layer_tensor_name("ffn_norm",    il);
        const std::string s_ffn_up    = layer_tensor_name("ffn_up",      il);
        const std::string s_ffn_down  = layer_tensor_name("ffn_down",    il);

        L.attn_norm = require_tensor(model, s_attn_norm, error); if (!L.attn_norm) return false;
        L.wqkv      = require_tensor(model, s_wqkv,      error); if (!L.wqkv)      return false;
        L.wo        = require_tensor(model, s_wo,        error); if (!L.wo)        return false;
        L.ffn_norm  = require_tensor(model, s_ffn_norm,  error); if (!L.ffn_norm)  return false;
        L.ffn_up    = require_tensor(model, s_ffn_up,    error); if (!L.ffn_up)    return false;
        L.ffn_down  = require_tensor(model, s_ffn_down,  error); if (!L.ffn_down)  return false;

        // On the first layer, derive n_ff from ffn_up.
        if (il == 0) {
            if (L.ffn_up->ne[0] != out.n_embd || (L.ffn_up->ne[1] & 1) != 0) {
                std::ostringstream oss;
                oss << "phi3_weights_resolve: ffn_up shape ["
                    << L.ffn_up->ne[0] << "," << L.ffn_up->ne[1]
                    << "] incompatible with fused gate|up layout";
                error = oss.str();
                return false;
            }
            out.n_ff = (int) (L.ffn_up->ne[1] / 2);
        }

        if (!check_f32_vec(L.attn_norm, out.n_embd,              s_attn_norm.c_str(), error)) return false;
        if (!check_f32_vec(L.ffn_norm,  out.n_embd,              s_ffn_norm.c_str(),  error)) return false;
        if (!check_q8K_mat(L.wqkv,      out.n_embd, out.n_qkv,   s_wqkv.c_str(),      error)) return false;
        if (!check_q8K_mat(L.wo,        out.n_embd, out.n_embd,  s_wo.c_str(),        error)) return false;
        if (!check_q8K_mat(L.ffn_up,    out.n_embd, 2*out.n_ff,  s_ffn_up.c_str(),    error)) return false;
        if (!check_q8K_mat(L.ffn_down,  out.n_ff,   out.n_embd,  s_ffn_down.c_str(),  error)) return false;
    }

    return true;
}

void phi3_weights_dump(const Phi3Weights & w) {
    fprintf(stderr, "phi3 weights: n_layer=%d n_embd=%d n_head=%d n_head_kv=%d n_embd_head=%d n_embd_gqa=%d n_qkv=%d n_ff=%d n_vocab=%d\n",
        w.n_layer, w.n_embd, w.n_head, w.n_head_kv, w.n_embd_head, w.n_embd_gqa, w.n_qkv, w.n_ff, w.n_vocab);

    auto dump = [](const char * tag, const ggml_tensor * t) {
        if (t == nullptr) {
            fprintf(stderr, "  %-32s <absent>\n", tag);
            return;
        }
        fprintf(stderr, "  %-32s shape=[%5lld,%5lld,%5lld,%5lld] type=%-6s data=%p\n",
            tag,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            ggml_type_name(t->type), (const void *) t->data);
    };

    dump("token_embd.weight",       w.tok_embd);
    dump("output_norm.weight",      w.output_norm);
    fprintf(stderr, "  %-32s (%s)\n", "output.weight",
        w.output_tied_to_embd ? "TIED to token_embd" : "distinct");
    dump("output.weight",           w.output);
    dump("rope_factors_long.weight",  w.rope_long);
    dump("rope_factors_short.weight", w.rope_short);

    fprintf(stderr, "phi3 weights: per-layer (showing first/last only)\n");
    for (int il : {0, w.n_layer - 1}) {
        if (il < 0) continue;
        char tag[64];
        const Phi3LayerWeights & L = w.layers[il];
        std::snprintf(tag, sizeof(tag), "blk.%d.attn_norm.weight",   il); dump(tag, L.attn_norm);
        std::snprintf(tag, sizeof(tag), "blk.%d.attn_qkv.weight",    il); dump(tag, L.wqkv);
        std::snprintf(tag, sizeof(tag), "blk.%d.attn_output.weight", il); dump(tag, L.wo);
        std::snprintf(tag, sizeof(tag), "blk.%d.ffn_norm.weight",    il); dump(tag, L.ffn_norm);
        std::snprintf(tag, sizeof(tag), "blk.%d.ffn_up.weight",      il); dump(tag, L.ffn_up);
        std::snprintf(tag, sizeof(tag), "blk.%d.ffn_down.weight",    il); dump(tag, L.ffn_down);
    }
}


// =============================================================================
// Phi3KV — implementation
// =============================================================================

namespace {

// Layer stride in F16 elements: head_dim * n_head_kv * ctx_max
// (identical for K layout [head_dim, n_head_kv, ctx_max] and
//  V_T layout [ctx_max, n_head_kv, head_dim] — same total cells, different order).
size_t phi3_kv_layer_stride(const Phi3KV & kv) {
    return (size_t) kv.head_dim * (size_t) kv.n_head_kv * (size_t) kv.ctx_max;
}

} // namespace

bool phi3_kv_init(Phi3KV & kv, int n_layer, int n_head_kv, int head_dim, int ctx_max, std::string & error) {
    error.clear();
    phi3_kv_free(kv); // safe on uninitialized struct

    if (n_layer <= 0 || n_layer > Phi3Weights::MAX_LAYERS ||
        n_head_kv <= 0 || head_dim <= 0 || ctx_max <= 0) {
        std::ostringstream oss;
        oss << "phi3_kv_init: invalid dims n_layer=" << n_layer
            << " n_head_kv=" << n_head_kv << " head_dim=" << head_dim
            << " ctx_max=" << ctx_max;
        error = oss.str();
        return false;
    }

    kv.n_layer     = n_layer;
    kv.n_head_kv   = n_head_kv;
    kv.head_dim    = head_dim;
    kv.ctx_max     = ctx_max;
    kv.current_len = 0;

    const size_t stride_elts = phi3_kv_layer_stride(kv);
    const size_t total_elts  = (size_t) n_layer * stride_elts;
    const size_t total_bytes = total_elts * sizeof(ggml_fp16_t);

    // Allocate K_buf and V_buf separately so each is cache-line aligned and we
    // can free them independently. Using new[]() zero-initializes (helpful for
    // any uninitialized-read corner cases during early development).
    try {
        kv.K_buf = new ggml_fp16_t[total_elts]();
        kv.V_buf = new ggml_fp16_t[total_elts]();
    } catch (const std::bad_alloc &) {
        phi3_kv_free(kv);
        std::ostringstream oss;
        oss << "phi3_kv_init: OOM allocating " << (2 * total_bytes / (1024 * 1024)) << " MiB";
        error = oss.str();
        return false;
    }

    for (int il = 0; il < n_layer; ++il) {
        kv.K[il] = kv.K_buf + (size_t) il * stride_elts;
        kv.V[il] = kv.V_buf + (size_t) il * stride_elts;
    }

    fprintf(stderr, "phi3 kv: allocated K=%.1f MiB + V=%.1f MiB (total %.2f GiB), n_layer=%d head_dim=%d n_head_kv=%d ctx_max=%d\n",
        total_bytes / (1024.0 * 1024.0),
        total_bytes / (1024.0 * 1024.0),
        (2.0 * total_bytes) / (1024.0 * 1024.0 * 1024.0),
        n_layer, head_dim, n_head_kv, ctx_max);

    return true;
}

void phi3_kv_free(Phi3KV & kv) {
    delete[] kv.K_buf; kv.K_buf = nullptr;
    delete[] kv.V_buf; kv.V_buf = nullptr;
    for (int il = 0; il < Phi3Weights::MAX_LAYERS; ++il) {
        kv.K[il] = nullptr;
        kv.V[il] = nullptr;
    }
    kv.n_layer = kv.n_head_kv = kv.head_dim = kv.ctx_max = kv.current_len = 0;
}

size_t phi3_kv_bytes(const Phi3KV & kv) {
    if (kv.n_layer <= 0) return 0;
    return 2 * (size_t) kv.n_layer * phi3_kv_layer_stride(kv) * sizeof(ggml_fp16_t);
}

void phi3_kv_truncate(Phi3KV & kv, int new_len) {
    if (new_len < 0)               new_len = 0;
    if (new_len > kv.current_len)  new_len = kv.current_len;
    kv.current_len = new_len;
    // Stale cells beyond current_len are intentionally NOT zeroed — they will
    // be overwritten on the next write at those positions.
}

void phi3_kv_drop_range(Phi3KV & kv, int p0, int p1) {
    if (p0 < 0) p0 = 0;
    if (p1 > kv.current_len) p1 = kv.current_len;
    if (p1 <= p0) return;

    const int n_drop = p1 - p0;
    const int n_tail = kv.current_len - p1;
    if (n_tail <= 0) {
        kv.current_len = p0;
        return;
    }

    const int head_dim   = kv.head_dim;
    const int n_head_kv  = kv.n_head_kv;
    const int ctx_max    = kv.ctx_max;

    for (int il = 0; il < kv.n_layer; ++il) {
        // K layout: [head_dim, n_head_kv, ctx_max]. A "row" at position pos
        // is (n_head_kv * head_dim) F16s starting at K[pos * n_head_kv * head_dim].
        // We can shift the entire tail in one memmove because positions are the
        // outer dimension in this layout.
        const size_t k_row_elts = (size_t) n_head_kv * (size_t) head_dim;
        ggml_fp16_t * k_dst = kv.K[il] + (size_t) p0 * k_row_elts;
        ggml_fp16_t * k_src = kv.K[il] + (size_t) p1 * k_row_elts;
        std::memmove(k_dst, k_src, (size_t) n_tail * k_row_elts * sizeof(ggml_fp16_t));

        // V_T layout: [ctx_max, n_head_kv, head_dim]. Position pos is the
        // INNERMOST dim, so dropping positions requires shifting each
        // (h, d) column independently with a strided memmove.
        for (int d = 0; d < head_dim; ++d) {
            for (int h = 0; h < n_head_kv; ++h) {
                ggml_fp16_t * v_col = kv.V[il] +
                    (size_t) d * (size_t) n_head_kv * (size_t) ctx_max +
                    (size_t) h * (size_t) ctx_max;
                std::memmove(v_col + p0, v_col + p1, (size_t) n_tail * sizeof(ggml_fp16_t));
            }
        }
    }

    kv.current_len -= n_drop;
}


// Self-test: write sentinel = (uint16_t)(0x1000 + il*0x100 + h*0x10 + (pos & 0xF))
// to K and V at each (il, h, d=0..head_dim-1, pos), then exercise the rewind
// primitives and check invariants.
namespace {

uint16_t kv_sentinel(int il, int h, int d, int pos) {
    return (uint16_t) (((il & 0xF) << 12) | ((h & 0xF) << 8) | ((d & 0xF) << 4) | (pos & 0xF));
}

void kv_write_sentinels(Phi3KV & kv, int p0, int p1) {
    const int head_dim   = kv.head_dim;
    const int n_head_kv  = kv.n_head_kv;
    const int ctx_max    = kv.ctx_max;
    const size_t k_row_elts = (size_t) n_head_kv * (size_t) head_dim;

    for (int il = 0; il < kv.n_layer; ++il) {
        for (int pos = p0; pos < p1; ++pos) {
            for (int h = 0; h < n_head_kv; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    kv.K[il][(size_t) pos * k_row_elts + (size_t) h * head_dim + d] =
                        (ggml_fp16_t) kv_sentinel(il, h, d, pos);
                    kv.V[il][(size_t) d * n_head_kv * ctx_max + (size_t) h * ctx_max + pos] =
                        (ggml_fp16_t) kv_sentinel(il, h, d, pos);
                }
            }
        }
    }
}

bool kv_check_sentinel(const Phi3KV & kv, int pos, int expect_il, int expect_h, int expect_d, int expect_pos, const char * tag, std::string & error) {
    const size_t k_row_elts = (size_t) kv.n_head_kv * (size_t) kv.head_dim;
    const uint16_t want = kv_sentinel(expect_il, expect_h, expect_d, expect_pos);
    const ggml_fp16_t got_k = kv.K[expect_il][(size_t) pos * k_row_elts + (size_t) expect_h * kv.head_dim + expect_d];
    const ggml_fp16_t got_v = kv.V[expect_il][(size_t) expect_d * kv.n_head_kv * kv.ctx_max + (size_t) expect_h * kv.ctx_max + pos];
    if ((uint16_t) got_k != want || (uint16_t) got_v != want) {
        std::ostringstream oss;
        oss << "kv self-test (" << tag << "): pos=" << pos << " expected_orig_pos=" << expect_pos
            << " want=0x" << std::hex << want
            << " got_K=0x" << (uint16_t) got_k
            << " got_V=0x" << (uint16_t) got_v;
        error = oss.str();
        return false;
    }
    return true;
}

} // namespace

bool phi3_kv_self_test(const Phi3Weights & w, std::string & error) {
    error.clear();

    Phi3KV kv;
    // Tiny ctx for the test — keep alloc fast (and total under a few MB).
    constexpr int kTestCtx = 16;
    if (!phi3_kv_init(kv, w.n_layer, w.n_head_kv, w.n_embd_head, kTestCtx, error)) {
        return false;
    }

    // 1. Write 10 positions of sentinels.
    kv.current_len = 10;
    kv_write_sentinels(kv, 0, 10);

    // 2. Spot check: pos 7, layer 3, head 5, dim 2 — must match sentinel(3,5,2,7).
    if (!kv_check_sentinel(kv, 7, 3, 5, 2, 7, "initial write", error)) { phi3_kv_free(kv); return false; }

    // 3. truncate(5).
    phi3_kv_truncate(kv, 5);
    if (kv.current_len != 5) {
        error = "phi3_kv_truncate did not update current_len";
        phi3_kv_free(kv);
        return false;
    }

    // 4. Re-populate positions 5..7 with NEW sentinels (different il pattern), check.
    kv.current_len = 8;
    kv_write_sentinels(kv, 5, 8);
    if (!kv_check_sentinel(kv, 6, 1, 0, 4, 6, "post-truncate rewrite", error)) { phi3_kv_free(kv); return false; }

    // 5. drop_range(1, 3) — drops positions 1,2 and shifts {3,4,5,6,7} -> {1,2,3,4,5}.
    phi3_kv_drop_range(kv, 1, 3);
    if (kv.current_len != 6) {
        std::ostringstream oss;
        oss << "phi3_kv_drop_range: expected current_len=6, got " << kv.current_len;
        error = oss.str();
        phi3_kv_free(kv);
        return false;
    }
    // After drop: NEW pos 1 holds what was at OLD pos 3, NEW pos 5 holds OLD pos 7.
    if (!kv_check_sentinel(kv, 1, 2, 1, 3, 3, "post-drop pos1<-3", error)) { phi3_kv_free(kv); return false; }
    if (!kv_check_sentinel(kv, 5, 0, 2, 1, 7, "post-drop pos5<-7", error)) { phi3_kv_free(kv); return false; }
    // Surviving prefix [0] should be untouched.
    if (!kv_check_sentinel(kv, 0, 7, 7, 7, 0, "post-drop pos0 prefix", error)) { phi3_kv_free(kv); return false; }

    // 6. keep_prefix(2).
    phi3_kv_keep_prefix(kv, 2);
    if (kv.current_len != 2) {
        error = "phi3_kv_keep_prefix did not update current_len";
        phi3_kv_free(kv);
        return false;
    }

    fprintf(stderr, "phi3 kv self-test: PASS (alloc=%.2f MiB, all rewind ops correct)\n",
        phi3_kv_bytes(kv) / (1024.0 * 1024.0));

    phi3_kv_free(kv);
    return true;
}


// ---------------------------------------------------------------------------
// Phi3MatmulPool self-test — Phase A.
// ---------------------------------------------------------------------------

#include "phi3_fused_ops.h"
#include <chrono>
#include <cmath>
#include <random>
#include <thread>

namespace {

void matmul_serial_ref(const float * w, const float * x, float * dst, int K, int N) {
    for (int v = 0; v < N; ++v) {
        const float * row = w + (size_t) v * K;
        float s = 0.0f;
        for (int k = 0; k < K; ++k) {
            s += row[k] * x[k];
        }
        dst[v] = s;
    }
}

bool matmul_compare(const float * a, const float * b, int N, std::string & error, const char * tag) {
    for (int v = 0; v < N; ++v) {
        // F32×F32 vec_dot may use SIMD with different summation order than the
        // strict left-to-right reference. Allow a tiny absolute+relative slop.
        const float diff = std::fabs(a[v] - b[v]);
        const float ref  = std::fabs(b[v]);
        if (diff > 1e-3f && diff > 1e-4f * ref) {
            std::ostringstream oss;
            oss << "phi3 matmul self-test (" << tag << "): mismatch at row " << v
                << " pool=" << a[v] << " ref=" << b[v]
                << " diff=" << diff;
            error = oss.str();
            return false;
        }
    }
    return true;
}

} // namespace

bool phi3_matmul_pool_self_test(int n_threads, std::string & error) {
    if (n_threads < 2) {
        n_threads = 4;
    }

    Phi3MatmulPool pool;
    if (!phi3_matmul_pool_init(pool, n_threads, error)) {
        return false;
    }

    const auto * traits = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_F32) {
        error = "phi3 matmul self-test: F32 type traits unavailable";
        phi3_matmul_pool_free(pool);
        return false;
    }

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

    struct Case { int K; int N; const char * tag; };
    const Case cases[] = {
        {  64,  512, "tiny"  },
        { 256, 1024, "small" },
        {3072, 9216, "phi3-wqkv" }, // exercises the actual phi3-mini wqkv shape
    };

    for (const auto & c : cases) {
        std::vector<float> w((size_t) c.K * c.N);
        std::vector<float> x((size_t) c.K);
        std::vector<float> dst_pool(c.N, 0.0f);
        std::vector<float> dst_ref (c.N, 0.0f);
        for (auto & v : w) { v = uniform(rng); }
        for (auto & v : x) { v = uniform(rng); }

        Phi3MatmulJob job{};
        job.w_traits    = traits;
        job.w_base      = (const uint8_t *) w.data();
        job.w_row_bytes = (size_t) c.K * sizeof(float);
        job.src_q       = x.data();
        job.dst         = dst_pool.data();
        job.K           = c.K;
        job.N_total     = c.N;

        // Run several iterations to exercise the phase-1 spin path
        // (back-to-back jobs).
        for (int i = 0; i < 8; ++i) {
            std::fill(dst_pool.begin(), dst_pool.end(), 0.0f);
            phi3_matmul_pool_run(&pool, job);
            matmul_serial_ref(w.data(), x.data(), dst_ref.data(), c.K, c.N);
            if (!matmul_compare(dst_pool.data(), dst_ref.data(), c.N, error, c.tag)) {
                phi3_matmul_pool_free(pool);
                return false;
            }
        }

        // Insert a sleep > phase2_us to drive workers into phase-3 (cv-park),
        // then dispatch again — exercises the wake path.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::fill(dst_pool.begin(), dst_pool.end(), 0.0f);
        phi3_matmul_pool_run(&pool, job);
        if (!matmul_compare(dst_pool.data(), dst_ref.data(), c.N, error, c.tag)) {
            phi3_matmul_pool_free(pool);
            return false;
        }
    }

    // Stress: kick 200 back-to-back jobs and time them. Just verifies no
    // crashes / hangs; output already validated above.
    {
        const Case c = cases[1]; // small
        std::vector<float> w((size_t) c.K * c.N);
        std::vector<float> x((size_t) c.K);
        std::vector<float> dst(c.N);
        for (auto & v : w) { v = uniform(rng); }
        for (auto & v : x) { v = uniform(rng); }
        Phi3MatmulJob job{traits, (const uint8_t *) w.data(),
                          (size_t) c.K * sizeof(float),
                          x.data(), dst.data(), c.K, c.N};
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 200; ++i) {
            phi3_matmul_pool_run(&pool, job);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        fprintf(stderr, "phi3 matmul self-test: stress 200x %s @ %d workers in %.1f us (%.2f us/dispatch)\n",
            c.tag, n_threads, us, us / 200.0);
    }

    phi3_matmul_pool_free(pool);
    fprintf(stderr, "phi3 matmul self-test: PASS (%d workers, 3 shapes, phase-1/3 transitions exercised)\n", n_threads);
    error.clear();
    return true;
}


// ---------------------------------------------------------------------------
// Per-kernel self-test — Phase A.
//
// Three small F32 kernel helpers compared against the standard ggml ops on
// the ggml-cpu backend. The helpers below mirror the math of:
//   - ggml_compute_forward_rms_norm_f32 fused with ggml_mul (LLM_NORM_RMS)
//   - ggml_compute_forward_get_rows for quantized tok_embd
//   - ggml_compute_forward_rope_f32 for GGML_ROPE_TYPE_NEOX
// They are intentionally simple, single-thread reference implementations.
// In A2.4 the inner loops will be promoted into the Phi3MatmulPool / hot
// path; the math contract proven here will not change.
// ---------------------------------------------------------------------------

#include <cmath>
#include <random>

namespace {

// ---- Helper 1: fused RMSNorm * weight (F32) -------------------------------
void phi3_kernel_rmsnorm_mul_f32(float * dst, const float * src, const float * w, int n, float eps) {
    // sum_sq must accumulate in double to match ggml_float in
    // ggml-cpu/ops.cpp:3757.
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (double) src[i] * (double) src[i];
    }
    const float mean  = (float) (sum / (double) n);
    const float scale = 1.0f / std::sqrt(mean + eps);
    for (int i = 0; i < n; ++i) {
        dst[i] = src[i] * scale * w[i];
    }
}

// ---- Helper 2: token embedding row dequant -------------------------------
// Reads one row of length n_embd from a quantized weight matrix of shape
// {n_embd, n_vocab} and writes n_embd F32 floats. tok_embd is stored
// row-contiguous: row stride = ggml_row_size(type, n_embd).
void phi3_kernel_tok_embd_row(float * dst, const ggml_type_traits * traits,
                              const uint8_t * w_base, size_t row_bytes,
                              int token, int n_embd) {
    const uint8_t * row = w_base + (size_t) token * row_bytes;
    traits->to_float(row, dst, n_embd);
}

// ---- Helper 3: NeoX RoPE for one head, one position ----------------------
// In-place style: dst and src may alias. n_dims is the number of dims to
// rotate (== head_dim for full rotation). pos is a single position. The
// remaining (head_dim - n_dims) dims are copied through unchanged
// (matches ggml's "fill the remain channels" branch).
void phi3_kernel_rope_neox_f32(float * dst, const float * src,
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
    // copy through any trailing dims (n_dims < head_dim)
    for (int i = n_dims; i < head_dim; ++i) {
        dst[i] = src[i];
    }
}

bool kernel_close(const float * a, const float * b, int n, float atol, float rtol,
                  std::string & error, const char * tag) {
    for (int i = 0; i < n; ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float ref  = std::fabs(b[i]);
        if (diff > atol && diff > rtol * ref) {
            std::ostringstream oss;
            oss << "phi3 kernel self-test (" << tag << "): mismatch at i=" << i
                << " ours=" << a[i] << " ggml=" << b[i] << " diff=" << diff;
            error = oss.str();
            return false;
        }
    }
    return true;
}

// Run a single-op ggml graph and copy the result into `out`.
bool run_ggml_graph(ggml_context * ctx, ggml_tensor * result, float * out, int n_elts) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        return false;
    }
    std::memcpy(out, result->data, (size_t) n_elts * sizeof(float));
    return true;
}

bool test_rmsnorm(std::string & error) {
    const int n_embd = 3072;
    const float eps = 1e-5f;

    std::mt19937 rng(0xA11CEu);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> src(n_embd), w(n_embd);
    for (auto & v : src) v = uni(rng);
    for (auto & v : w)   v = uni(rng) * 0.5f + 1.0f;

    // ours
    std::vector<float> ours(n_embd);
    phi3_kernel_rmsnorm_mul_f32(ours.data(), src.data(), w.data(), n_embd, eps);

    // ggml oracle: rms_norm(eps) * w
    ggml_init_params ip{ 16ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "rmsnorm: ggml_init failed"; return false; }

    ggml_tensor * tsrc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * tw   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    std::memcpy(tsrc->data, src.data(), n_embd * sizeof(float));
    std::memcpy(tw->data,   w.data(),   n_embd * sizeof(float));
    ggml_tensor * tout = ggml_mul(ctx, ggml_rms_norm(ctx, tsrc, eps), tw);

    std::vector<float> oracle(n_embd);
    bool ok = run_ggml_graph(ctx, tout, oracle.data(), n_embd);
    ggml_free(ctx);
    if (!ok) { error = "rmsnorm: graph_compute failed"; return false; }

    // Reduction order differs vs ggml SIMD path -> small relative tolerance.
    return kernel_close(ours.data(), oracle.data(), n_embd, 1e-4f, 1e-4f, error, "rmsnorm");
}

bool test_tok_embd_dequant(std::string & error) {
    const int n_embd  = 64;
    const int n_vocab = 32;
    const int probe_tokens[] = { 0, 7, 17, 31 };

    std::mt19937 rng(0xB033Du);
    std::uniform_real_distribution<float> uni(-0.1f, 0.1f);
    std::vector<float> ref_full((size_t) n_embd * n_vocab);
    for (auto & v : ref_full) v = uni(rng);

    // Test for F16 (the practical token embedding type for the custom forward
    // and for "phi3-mini-4k tied weights" case). Quantized types are stored
    // row-aligned to QK boundary which n_embd=64 satisfies for all K-quants,
    // but only F16 is required for the immediate A2 plan.
    const ggml_type type = GGML_TYPE_F16;

    ggml_init_params ip{ 4ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = "tok_embd: ggml_init failed"; return false; }

    ggml_tensor * tok_embd = ggml_new_tensor_2d(ctx, type, n_embd, n_vocab);
    // quantize the reference into tok_embd row-by-row using the type's from_float
    const auto * trc = ggml_get_type_traits_cpu(type);
    if (!trc || !trc->from_float) {
        ggml_free(ctx);
        error = "tok_embd: type has no from_float";
        return false;
    }
    const size_t row_bytes = ggml_row_size(type, n_embd);
    for (int v = 0; v < n_vocab; ++v) {
        trc->from_float(ref_full.data() + (size_t) v * n_embd,
                        (uint8_t *) tok_embd->data + (size_t) v * row_bytes,
                        n_embd);
    }

    // build get_rows graph for all probe tokens at once
    const int n_probe = (int) (sizeof(probe_tokens) / sizeof(probe_tokens[0]));
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_probe);
    int32_t * idx_data = (int32_t *) idx->data;
    for (int i = 0; i < n_probe; ++i) idx_data[i] = probe_tokens[i];
    ggml_tensor * tout = ggml_get_rows(ctx, tok_embd, idx);

    std::vector<float> oracle((size_t) n_embd * n_probe);
    bool ok = run_ggml_graph(ctx, tout, oracle.data(), n_embd * n_probe);
    if (!ok) { ggml_free(ctx); error = "tok_embd: graph_compute failed"; return false; }

    // ours: helper per-row, into a flat buffer
    std::vector<float> ours((size_t) n_embd * n_probe);
    const auto * tr = ggml_get_type_traits(type);
    for (int i = 0; i < n_probe; ++i) {
        phi3_kernel_tok_embd_row(ours.data() + (size_t) i * n_embd, tr,
                                 (const uint8_t *) tok_embd->data, row_bytes,
                                 probe_tokens[i], n_embd);
    }
    ggml_free(ctx);
    // F16 dequant is bit-exact in both paths but allow tiny slop just in case
    return kernel_close(ours.data(), oracle.data(), n_embd * n_probe, 1e-6f, 1e-6f, error, "tok_embd");
}

bool test_rope_neox(const char * tag, bool with_factors, std::string & error) {
    const int head_dim = 96;
    const int n_dims   = 96;  // full rotation
    const int n_head   = 2;
    const int pos      = 5;
    const float freq_base = 10000.0f;

    std::mt19937 rng(0xC0DEu + (with_factors ? 1u : 0u));
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> src((size_t) head_dim * n_head);
    for (auto & v : src) v = uni(rng);
    std::vector<float> factors(n_dims / 2);
    for (auto & v : factors) v = 0.5f + 0.5f * uni(rng);  // > 0 to avoid div-by-zero
    const float * ff = with_factors ? factors.data() : nullptr;

    // ours
    std::vector<float> ours((size_t) head_dim * n_head);
    for (int h = 0; h < n_head; ++h) {
        phi3_kernel_rope_neox_f32(ours.data() + (size_t) h * head_dim,
                                  src .data() + (size_t) h * head_dim,
                                  ff, n_dims, head_dim, pos, freq_base);
    }

    // ggml oracle: ggml_rope_ext(src, pos_tensor, factors_or_null,
    //   n_dims, GGML_ROPE_TYPE_NEOX, n_ctx_orig=0, freq_base, 1.0, 0.0, 1.0, 32, 1)
    ggml_init_params ip{ 4ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { error = std::string("rope ") + tag + ": ggml_init failed"; return false; }

    // ggml rope expects shape [head_dim, n_head, n_pos, n_batch]. We have
    // one position so n_pos=1, n_batch=1.
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

    return kernel_close(ours.data(), oracle.data(), head_dim * n_head, 1e-5f, 1e-5f, error, tag);
}

} // namespace

bool phi3_kernel_self_test(std::string & error) {
    if (!test_rmsnorm(error))                                 return false;
    fprintf(stderr, "phi3 kernel self-test: rmsnorm                       OK\n");
    if (!test_tok_embd_dequant(error))                        return false;
    fprintf(stderr, "phi3 kernel self-test: tok_embd_dequant (F16)        OK\n");
    if (!test_rope_neox("rope_neox_no_factors", false, error)) return false;
    fprintf(stderr, "phi3 kernel self-test: rope_neox (factors=NULL)      OK\n");
    if (!test_rope_neox("rope_neox_with_factors", true, error)) return false;
    fprintf(stderr, "phi3 kernel self-test: rope_neox (factors=synth)     OK\n");
    fprintf(stderr, "phi3 kernel self-test: PASS (rmsnorm + tok_embd + 2x rope_neox)\n");
    error.clear();
    return true;
}
