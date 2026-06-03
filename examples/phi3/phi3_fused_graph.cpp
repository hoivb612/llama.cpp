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

// ===========================================================================
// A2.2 — Validator + ctx alloc/free.
//
// Implementation of the Phase A custom-forward init path:
//   - phi3_fused_validate_supported : exhaustive feature-rejection check.
//   - phi3_fused_ctx_init / _free   : KV + scratch + resolved RoPE config.
//
// The validator returns a specific error naming the offending feature; the
// runtime treats a `false` return as fatal when --phi3-fused-forward was
// explicitly requested (no silent fallback to the standard graph). See
// __phase_A2_spec.md §1.5 for the full list.
// ===========================================================================

#include <cstdio>

namespace {

// Helper: look up a tensor by formatted name. Returns nullptr if not present.
const ggml_tensor * tensor_or_null(const llama_model * model, const char * fmt, int il) {
    char name[64];
    std::snprintf(name, sizeof(name), fmt, il);
    return llama_model_get_tensor_by_name(model, name);
}

const ggml_tensor * tensor_or_null(const llama_model * model, const char * name) {
    return llama_model_get_tensor_by_name(model, name);
}

// Helper: reject if `t` is present. Builds a descriptive error message.
bool reject_if_present(const ggml_tensor * t, const char * label, std::string & error) {
    if (t != nullptr) {
        error = std::string("phi3_fused_validate: feature not supported by Phase A path: ") + label;
        return false;
    }
    return true;
}

// Helper: assert tensor is canonical-layout host data (no view, contiguous row).
bool check_host_canonical(const ggml_tensor * w, const char * label, std::string & error) {
    if (w == nullptr) {
        return true;  // optional weights pass through
    }
    if (w->data == nullptr) {
        error = std::string("phi3_fused_validate: ") + label + " has w->data == NULL (likely a view)";
        return false;
    }
    const size_t expected_row = ggml_row_size(w->type, w->ne[0]);
    if ((size_t) w->nb[1] != expected_row) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "phi3_fused_validate: %s non-canonical layout: nb[1]=%zu expected %zu (type=%s, ne[0]=%lld)",
            label, (size_t) w->nb[1], expected_row, ggml_type_name(w->type), (long long) w->ne[0]);
        error = buf;
        return false;
    }
    return true;
}

// Helper: ensure a vec_dot exists for the weight type (CPU path will use it).
bool check_vec_dot_available(const ggml_tensor * w, const char * label, std::string & error) {
    if (w == nullptr) return true;
    const auto * tr = ggml_get_type_traits_cpu(w->type);
    if (tr == nullptr || tr->vec_dot == nullptr || tr->vec_dot_type == GGML_TYPE_COUNT) {
        error = std::string("phi3_fused_validate: ") + label + " type " + ggml_type_name(w->type)
              + " has no CPU vec_dot kernel";
        return false;
    }
    return true;
}

} // namespace

bool phi3_fused_validate_supported(
        const llama_model    * model,
        const Phi3Weights    & w,
        const llama_context  * lctx,
        std::string          & error) {
    error.clear();

    if (model == nullptr) {
        error = "phi3_fused_validate: model is null";
        return false;
    }

    // ----- context-side (cparams) checks -----
    // Skipped only if lctx is null (caller validating model-only).
    if (lctx != nullptr) {
        llama_b612_phi3_features f{};
        llama_b612_get_phi3_features(model, lctx, &f);

        if (f.cp_flash_attn) {
            error = "phi3_fused_validate: cparams.flash_attn == true (Phase A path requires manual SD attention)";
            return false;
        }
        if (f.cp_embeddings) {
            error = "phi3_fused_validate: cparams.embeddings == true (generation context only)";
            return false;
        }
        if (!f.cp_causal_attn) {
            error = "phi3_fused_validate: cparams.causal_attn == false (Phase A path is causal-only)";
            return false;
        }
        if (f.cp_n_seq_max != 1) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "phi3_fused_validate: cparams.n_seq_max=%u (Phase A path supports single-sequence only)",
                (unsigned) f.cp_n_seq_max);
            error = buf;
            return false;
        }

        // YARN long-context guard: if RoPE factor tables exist (Phi-3-medium-128k style)
        // AND we'd actually need them (ctx > orig), reject — Phase A doesn't implement
        // YARN-style ext_factor blending yet. Allow either rope_factors-absent OR
        // ctx <= orig (the factor tables are present but unused at short ctx).
        const bool has_rope_factors = (w.rope_long != nullptr) || (w.rope_short != nullptr);
        if (has_rope_factors && f.hp_n_ctx_orig_yarn > 0 && f.cp_n_ctx_seq > (uint32_t) f.hp_n_ctx_orig_yarn) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "phi3_fused_validate: YARN long-context engaged (n_ctx_seq=%u > n_ctx_orig_yarn=%d) "
                "with rope_factors present — Phase A path does not implement YARN blending yet",
                (unsigned) f.cp_n_ctx_seq, (int) f.hp_n_ctx_orig_yarn);
            error = buf;
            return false;
        }

        // Hparams flags that Phase A cannot honor.
        if (f.hp_f_clamp_kqv != 0.0f) {
            error = "phi3_fused_validate: hparams.f_clamp_kqv != 0 (KQV clamp not in Phase A path)";
            return false;
        }
        if (f.hp_attn_soft_cap) {
            error = "phi3_fused_validate: hparams.attn_soft_cap is enabled (tanh cap not in Phase A path)";
            return false;
        }
        if (f.hp_use_alibi || f.hp_f_max_alibi_bias != 0.0f) {
            error = "phi3_fused_validate: ALiBi attention not supported by Phase A path";
            return false;
        }
        if (f.hp_swa_type != 0) {  // LLAMA_SWA_TYPE_NONE == 0
            char buf[120];
            std::snprintf(buf, sizeof(buf),
                "phi3_fused_validate: hparams.swa_type=%d (Phase A path requires LLAMA_SWA_TYPE_NONE)",
                (int) f.hp_swa_type);
            error = buf;
            return false;
        }

        // LoRA / control-vectors must be inactive.
        if (llama_b612_has_active_lora(lctx)) {
            error = "phi3_fused_validate: active LoRA adapter (Phase A path bypasses build_lora_mm)";
            return false;
        }
        if (llama_b612_has_active_cvec(lctx)) {
            error = "phi3_fused_validate: active control vector (Phase A path bypasses build_cvec)";
            return false;
        }
    }

    // ----- GQA guard -----
    if (w.n_head != w.n_head_kv) {
        char buf[120];
        std::snprintf(buf, sizeof(buf),
            "phi3_fused_validate: GQA detected (n_head=%d, n_head_kv=%d) — Phase A path is dense-attention only",
            w.n_head, w.n_head_kv);
        error = buf;
        return false;
    }

    // ----- Per-layer absent-tensor checks -----
    for (int il = 0; il < w.n_layer; ++il) {
        // MoE: must be absent (Phi-3-MoE/Phi-4-MoE not yet supported).
        if (!reject_if_present(tensor_or_null(model, "blk.%d.ffn_gate_inp.weight",  il), "MoE ffn_gate_inp",  error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.ffn_up_exps.weight",   il), "MoE ffn_up_exps",   error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.ffn_gate_exps.weight", il), "MoE ffn_gate_exps", error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.ffn_down_exps.weight", il), "MoE ffn_down_exps", error)) return false;

        // Biases: must be absent.
        if (!reject_if_present(tensor_or_null(model, "blk.%d.attn_qkv.bias",    il), "attn_qkv.bias",    error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.attn_output.bias", il), "attn_output.bias", error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.attn_norm.bias",   il), "attn_norm.bias",   error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.ffn_norm.bias",    il), "ffn_norm.bias",    error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.ffn_up.bias",      il), "ffn_up.bias",      error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.ffn_down.bias",    il), "ffn_down.bias",    error)) return false;

        // Per-tensor scales: must be absent (NVFP4-style activation scales not honored).
        if (!reject_if_present(tensor_or_null(model, "blk.%d.attn_qkv.scale",    il), "attn_qkv.scale",    error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.attn_output.scale", il), "attn_output.scale", error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.ffn_up.scale",      il), "ffn_up.scale",      error)) return false;
        if (!reject_if_present(tensor_or_null(model, "blk.%d.ffn_down.scale",    il), "ffn_down.scale",    error)) return false;

        // Canonical layout + vec_dot for all per-layer weights we'll touch.
        const Phi3LayerWeights & L = w.layers[il];
        if (!check_host_canonical(L.attn_norm, "attn_norm", error)) return false;
        if (!check_host_canonical(L.wqkv,      "wqkv",      error)) return false;
        if (!check_host_canonical(L.wo,        "wo",        error)) return false;
        if (!check_host_canonical(L.ffn_norm,  "ffn_norm",  error)) return false;
        if (!check_host_canonical(L.ffn_up,    "ffn_up",    error)) return false;
        if (!check_host_canonical(L.ffn_down,  "ffn_down",  error)) return false;
        if (!check_vec_dot_available(L.wqkv,      "wqkv",      error)) return false;
        if (!check_vec_dot_available(L.wo,        "wo",        error)) return false;
        if (!check_vec_dot_available(L.ffn_up,    "ffn_up",    error)) return false;
        if (!check_vec_dot_available(L.ffn_down,  "ffn_down",  error)) return false;
    }

    // ----- Global absent-tensor checks -----
    if (!reject_if_present(tensor_or_null(model, "output.bias"),        "output.bias",        error)) return false;
    if (!reject_if_present(tensor_or_null(model, "output.scale"),       "output.scale",       error)) return false;
    if (!reject_if_present(tensor_or_null(model, "output.input_scale"), "output.input_scale", error)) return false;
    if (!reject_if_present(tensor_or_null(model, "output_norm.bias"),   "output_norm.bias",   error)) return false;

    // tok_embd dequant availability.
    if (w.tok_embd == nullptr) {
        error = "phi3_fused_validate: tok_embd is null";
        return false;
    }
    {
        const auto * tr = ggml_get_type_traits(w.tok_embd->type);
        if (tr == nullptr || tr->to_float == nullptr) {
            error = std::string("phi3_fused_validate: tok_embd type ") + ggml_type_name(w.tok_embd->type)
                  + " has no to_float dequantize kernel";
            return false;
        }
    }
    if (!check_host_canonical(w.tok_embd,    "tok_embd",    error)) return false;
    if (!check_host_canonical(w.output_norm, "output_norm", error)) return false;
    if (!check_host_canonical(w.output,      "output",      error)) return false;
    if (!check_vec_dot_available(w.output,   "output (lm_head)", error)) return false;

    return true;
}


bool phi3_fused_ctx_init(
        Phi3FusedCtx         & out,
        const Phi3Weights    & w,
        Phi3MatmulPool       * matmul_pool,
        const llama_model    * model,
        const llama_context  * lctx,
        int                    ctx_max,
        bool                   f32_debug,
        std::string          & error) {
    error.clear();
    out = Phi3FusedCtx{};

    if (!phi3_fused_validate_supported(model, w, lctx, error)) {
        return false;
    }
    if (ctx_max <= 0) {
        error = "phi3_fused_ctx_init: ctx_max must be > 0";
        return false;
    }

    out.w           = &w;
    out.matmul_pool = matmul_pool;

    // Resolve RoPE config from cparams + hparams (via accessor).
    llama_b612_phi3_features f{};
    if (lctx != nullptr) {
        llama_b612_get_phi3_features(model, lctx, &f);
        out.eps               = f.hp_f_norm_rms_eps;
        out.rope.n_rot        = f.hp_n_rot;
        out.rope.n_ctx_orig   = f.hp_n_ctx_orig_yarn;
        out.rope.freq_base    = f.cp_rope_freq_base;
        out.rope.freq_scale   = f.cp_rope_freq_scale;
        out.rope.ext_factor   = f.cp_yarn_ext_factor;
        out.rope.attn_factor  = f.cp_yarn_attn_factor;
        out.rope.beta_fast    = f.cp_yarn_beta_fast;
        out.rope.beta_slow    = f.cp_yarn_beta_slow;
    } else {
        // No lctx: fill from model only (cparams stay at training defaults).
        llama_b612_get_phi3_features(model, nullptr, &f);
        out.eps               = f.hp_f_norm_rms_eps;
        out.rope.n_rot        = f.hp_n_rot;
        out.rope.n_ctx_orig   = f.hp_n_ctx_orig_yarn;
        out.rope.freq_base    = f.hp_rope_freq_base_train;
        out.rope.freq_scale   = 1.0f;
        out.rope.attn_factor  = 1.0f;
    }
    // For short ctx we treat rope_factors as null even if they're loaded —
    // matches the standard graph (model.get_rope_factors returns short or
    // long based on pos at runtime; at short pos both reduce to identity).
    // Phase A only handles factors==null until A2.4+ adds long-ctx support.
    out.rope.rope_factors = nullptr;

    // Allocate KV cache.
    {
        const int head_dim = w.n_embd_head;
        if (!phi3_kv_init(out.kv, w.n_layer, w.n_head_kv, head_dim, ctx_max, error)) {
            return false;
        }
    }

    // Resolve quant types from weight traits.
    out.scratch.q_type_attn = ggml_get_type_traits_cpu(w.layers[0].wqkv->type)->vec_dot_type;
    out.scratch.q_type_ffn  = ggml_get_type_traits_cpu(w.layers[0].ffn_down->type)->vec_dot_type;

    // Scratch buffers (per-step).
    out.scratch.x_buf     .assign(w.n_embd,              0.0f);
    out.scratch.h_buf     .assign(w.n_embd,              0.0f);
    out.scratch.hq_buf    .assign(ggml_row_size(out.scratch.q_type_attn, w.n_embd), 0);
    out.scratch.ffq_buf   .assign(ggml_row_size(out.scratch.q_type_ffn,  w.n_ff),   0);
    out.scratch.qkv_buf   .assign(w.n_qkv,               0.0f);
    out.scratch.ctx_buf   .assign(w.n_embd,              0.0f);
    out.scratch.scores_buf.assign((size_t) ctx_max,      0.0f);
    out.scratch.upgate_buf.assign((size_t) 2 * w.n_ff,   0.0f);
    out.scratch.ff_buf    .assign(w.n_ff,                0.0f);

    if (f32_debug) {
        out.scratch.f32_debug = true;
        // Per-layer transient F32 mirrors. Allocated once; overwritten per layer.
        out.scratch.w_f32_attn_norm  .assign((size_t) w.n_embd,                   0.0f);
        out.scratch.w_f32_wqkv       .assign((size_t) w.n_embd * (size_t) w.n_qkv, 0.0f);
        out.scratch.w_f32_wo         .assign((size_t) w.n_embd * (size_t) w.n_embd, 0.0f);
        out.scratch.w_f32_ffn_norm   .assign((size_t) w.n_embd,                   0.0f);
        out.scratch.w_f32_ffn_up     .assign((size_t) w.n_embd * (size_t) 2 * (size_t) w.n_ff, 0.0f);
        out.scratch.w_f32_ffn_down   .assign((size_t) w.n_ff   * (size_t) w.n_embd, 0.0f);
        out.scratch.w_f32_vocab_chunk.assign((size_t) out.scratch.f32_vocab_chunk * (size_t) w.n_embd, 0.0f);
        out.scratch.f32_cached_layer = -1;
    }

    out.cur_pos = 0;
    return true;
}

void phi3_fused_ctx_free(Phi3FusedCtx & cx) {
    phi3_kv_free(cx.kv);
    cx.scratch.x_buf.clear();           cx.scratch.x_buf.shrink_to_fit();
    cx.scratch.h_buf.clear();           cx.scratch.h_buf.shrink_to_fit();
    cx.scratch.hq_buf.clear();          cx.scratch.hq_buf.shrink_to_fit();
    cx.scratch.ffq_buf.clear();         cx.scratch.ffq_buf.shrink_to_fit();
    cx.scratch.qkv_buf.clear();         cx.scratch.qkv_buf.shrink_to_fit();
    cx.scratch.ctx_buf.clear();         cx.scratch.ctx_buf.shrink_to_fit();
    cx.scratch.scores_buf.clear();      cx.scratch.scores_buf.shrink_to_fit();
    cx.scratch.upgate_buf.clear();      cx.scratch.upgate_buf.shrink_to_fit();
    cx.scratch.ff_buf.clear();          cx.scratch.ff_buf.shrink_to_fit();
    cx.scratch.w_f32_attn_norm.clear(); cx.scratch.w_f32_attn_norm.shrink_to_fit();
    cx.scratch.w_f32_wqkv.clear();      cx.scratch.w_f32_wqkv.shrink_to_fit();
    cx.scratch.w_f32_wo.clear();        cx.scratch.w_f32_wo.shrink_to_fit();
    cx.scratch.w_f32_ffn_norm.clear();  cx.scratch.w_f32_ffn_norm.shrink_to_fit();
    cx.scratch.w_f32_ffn_up.clear();    cx.scratch.w_f32_ffn_up.shrink_to_fit();
    cx.scratch.w_f32_ffn_down.clear();  cx.scratch.w_f32_ffn_down.shrink_to_fit();
    cx.scratch.w_f32_vocab_chunk.clear();
    cx.scratch.w_f32_vocab_chunk.shrink_to_fit();
    cx.w           = nullptr;
    cx.matmul_pool = nullptr;
    cx.cur_pos     = 0;
}

void phi3_fused_ctx_dump(const Phi3FusedCtx & cx) {
    fprintf(stderr, "phi3_fused_ctx_dump:\n");
    fprintf(stderr, "  eps                = %g\n", cx.eps);
    fprintf(stderr, "  rope.n_rot         = %d\n", cx.rope.n_rot);
    fprintf(stderr, "  rope.n_ctx_orig    = %d\n", cx.rope.n_ctx_orig);
    fprintf(stderr, "  rope.freq_base     = %g\n", cx.rope.freq_base);
    fprintf(stderr, "  rope.freq_scale    = %g\n", cx.rope.freq_scale);
    fprintf(stderr, "  rope.ext_factor    = %g\n", cx.rope.ext_factor);
    fprintf(stderr, "  rope.attn_factor   = %g\n", cx.rope.attn_factor);
    fprintf(stderr, "  rope.rope_factors  = %p (null=identity)\n", (const void *) cx.rope.rope_factors);
    fprintf(stderr, "  kv.ctx_max         = %d\n", cx.kv.ctx_max);
    fprintf(stderr, "  scratch.q_type_attn= %s\n", cx.scratch.q_type_attn != GGML_TYPE_COUNT ? ggml_type_name(cx.scratch.q_type_attn) : "(unset)");
    fprintf(stderr, "  scratch.q_type_ffn = %s\n", cx.scratch.q_type_ffn  != GGML_TYPE_COUNT ? ggml_type_name(cx.scratch.q_type_ffn)  : "(unset)");
    fprintf(stderr, "  scratch.f32_debug  = %s\n", cx.scratch.f32_debug ? "yes" : "no");
    size_t total = 0;
    total += cx.scratch.x_buf.size()      * sizeof(float);
    total += cx.scratch.h_buf.size()      * sizeof(float);
    total += cx.scratch.hq_buf.size();
    total += cx.scratch.ffq_buf.size();
    total += cx.scratch.qkv_buf.size()    * sizeof(float);
    total += cx.scratch.ctx_buf.size()    * sizeof(float);
    total += cx.scratch.scores_buf.size() * sizeof(float);
    total += cx.scratch.upgate_buf.size() * sizeof(float);
    total += cx.scratch.ff_buf.size()     * sizeof(float);
    fprintf(stderr, "  scratch.hot_bytes  = %zu (~%.1f KiB)\n", total, total / 1024.0);
    if (cx.scratch.f32_debug) {
        size_t dbg = 0;
        dbg += cx.scratch.w_f32_attn_norm.size()    * sizeof(float);
        dbg += cx.scratch.w_f32_wqkv.size()         * sizeof(float);
        dbg += cx.scratch.w_f32_wo.size()           * sizeof(float);
        dbg += cx.scratch.w_f32_ffn_norm.size()     * sizeof(float);
        dbg += cx.scratch.w_f32_ffn_up.size()       * sizeof(float);
        dbg += cx.scratch.w_f32_ffn_down.size()     * sizeof(float);
        dbg += cx.scratch.w_f32_vocab_chunk.size()  * sizeof(float);
        fprintf(stderr, "  scratch.f32_dbg_MiB= %.1f (per-layer mirrors + vocab chunk)\n", dbg / (1024.0 * 1024.0));
    }
}

// ===========================================================================
// A2.3 — Per-layer F32 warmup, F32 forward, and single-layer self-test.
// ===========================================================================

namespace {

// Dequantize a 2-D weight tensor into a flat F32 buffer. `dst_capacity` is
// the size of `dst` in floats; we need at least ne[0] * ne[1] floats.
bool dequant_2d_to_f32(const ggml_tensor * src, float * dst, size_t dst_capacity, std::string & error) {
    if (src == nullptr) {
        error = "dequant_2d_to_f32: src is null";
        return false;
    }
    const int64_t n0 = src->ne[0];
    const int64_t n1 = src->ne[1];
    if ((size_t) n0 * (size_t) n1 > dst_capacity) {
        std::ostringstream oss;
        oss << "dequant_2d_to_f32: capacity too small (need " << (size_t)n0*(size_t)n1
            << " got " << dst_capacity << ")";
        error = oss.str();
        return false;
    }
    const auto * traits = ggml_get_type_traits(src->type);
    if (src->type == GGML_TYPE_F32) {
        for (int64_t r = 0; r < n1; ++r) {
            const float * row = (const float *) ((const uint8_t *) src->data + r * src->nb[1]);
            std::memcpy(dst + r * n0, row, (size_t) n0 * sizeof(float));
        }
    } else if (src->type == GGML_TYPE_F16) {
        for (int64_t r = 0; r < n1; ++r) {
            const ggml_fp16_t * row = (const ggml_fp16_t *) ((const uint8_t *) src->data + r * src->nb[1]);
            ggml_fp16_to_fp32_row(row, dst + r * n0, n0);
        }
    } else {
        if (traits == nullptr || traits->to_float == nullptr) {
            error = std::string("dequant_2d_to_f32: no to_float for type ") + ggml_type_name(src->type);
            return false;
        }
        for (int64_t r = 0; r < n1; ++r) {
            const uint8_t * row = (const uint8_t *) src->data + r * src->nb[1];
            traits->to_float(row, dst + r * n0, n0);
        }
    }
    return true;
}

// Naive F32 matmul: y[j] = sum_k w[j*K + k] * x[k] for j in [0..M).
// w stored row-major: rows are length K, M rows total.
void f32_matmul_row(const float * w, const float * x, float * y, int M, int K) {
    for (int j = 0; j < M; ++j) {
        double acc = 0.0;
        const float * row = w + (size_t) j * K;
        for (int k = 0; k < K; ++k) {
            acc += (double) row[k] * (double) x[k];
        }
        y[j] = (float) acc;
    }
}

} // namespace


bool phi3_layer_warmup_f32_mirrors(Phi3FusedCtx & cx, int il, std::string & error) {
    error.clear();
    if (cx.w == nullptr) {
        error = "phi3_layer_warmup_f32_mirrors: cx.w is null";
        return false;
    }
    if (!cx.scratch.f32_debug) {
        error = "phi3_layer_warmup_f32_mirrors: cx.scratch.f32_debug is false (alloc the mirrors first)";
        return false;
    }
    if (il < 0 || il >= cx.w->n_layer) {
        std::ostringstream oss; oss << "phi3_layer_warmup_f32_mirrors: il=" << il << " out of range";
        error = oss.str();
        return false;
    }
    if (cx.scratch.f32_cached_layer == il) {
        return true;  // already warm
    }
    const Phi3LayerWeights & L = cx.w->layers[il];
    if (!dequant_2d_to_f32(L.attn_norm, cx.scratch.w_f32_attn_norm.data(), cx.scratch.w_f32_attn_norm.size(), error)) return false;
    if (!dequant_2d_to_f32(L.wqkv,      cx.scratch.w_f32_wqkv.data(),      cx.scratch.w_f32_wqkv.size(),      error)) return false;
    if (!dequant_2d_to_f32(L.wo,        cx.scratch.w_f32_wo.data(),        cx.scratch.w_f32_wo.size(),        error)) return false;
    if (!dequant_2d_to_f32(L.ffn_norm,  cx.scratch.w_f32_ffn_norm.data(),  cx.scratch.w_f32_ffn_norm.size(),  error)) return false;
    if (!dequant_2d_to_f32(L.ffn_up,    cx.scratch.w_f32_ffn_up.data(),    cx.scratch.w_f32_ffn_up.size(),    error)) return false;
    if (!dequant_2d_to_f32(L.ffn_down,  cx.scratch.w_f32_ffn_down.data(),  cx.scratch.w_f32_ffn_down.size(),  error)) return false;
    cx.scratch.f32_cached_layer = il;
    return true;
}


bool phi3_layer_forward_f32(
        Phi3FusedCtx       & cx,
        int                  il,
        int                  pos,
        float              * x_inout,
        Phi3LayerCapture   * capture,
        std::string        & error) {
    error.clear();
    if (cx.w == nullptr) { error = "phi3_layer_forward_f32: cx.w is null"; return false; }
    if (!cx.scratch.f32_debug) { error = "phi3_layer_forward_f32: requires f32_debug"; return false; }
    if (cx.scratch.f32_cached_layer != il) {
        if (!phi3_layer_warmup_f32_mirrors(cx, il, error)) return false;
    }

    const Phi3Weights & W = *cx.w;
    const int n_embd     = W.n_embd;
    const int n_head     = W.n_head;
    const int n_head_kv  = W.n_head_kv;
    const int head_dim   = W.n_embd_head;
    const int n_embd_kv  = head_dim * n_head_kv;
    const int n_embd_q   = head_dim * n_head;
    const int n_qkv      = W.n_qkv;
    const int n_ff       = W.n_ff;

    if (pos < 0 || pos >= cx.kv.ctx_max) {
        std::ostringstream oss; oss << "phi3_layer_forward_f32: pos=" << pos << " out of range";
        error = oss.str();
        return false;
    }

    // Local scratch (we cannot reuse cx.scratch.x_buf etc. since the capture
    // pointers may alias them; safer to use local std::vector here).
    std::vector<float> norm1(n_embd);
    std::vector<float> qkv(n_qkv);
    std::vector<float> attn_ctx(n_embd, 0.0f);
    std::vector<float> attn_out_buf(n_embd);
    std::vector<float> x_after_res1(n_embd);
    std::vector<float> norm2(n_embd);
    std::vector<float> upgate(2 * n_ff);
    std::vector<float> ff(n_ff);
    std::vector<float> ffn_out_buf(n_embd);

    // --- 1. Pre-attn RMSNorm ---
    phi3_kernel_rmsnorm_mul_f32(norm1.data(), x_inout,
                                cx.scratch.w_f32_attn_norm.data(),
                                n_embd, cx.eps);
    if (capture && capture->norm1) std::memcpy(capture->norm1, norm1.data(), n_embd * sizeof(float));

    // --- 2. wqkv ---
    f32_matmul_row(cx.scratch.w_f32_wqkv.data(), norm1.data(), qkv.data(), n_qkv, n_embd);
    if (capture && capture->qkv) std::memcpy(capture->qkv, qkv.data(), n_qkv * sizeof(float));

    float * q = qkv.data();
    float * k = qkv.data() + n_embd_q;
    float * v = qkv.data() + n_embd_q + n_embd_kv;

    // --- 3. RoPE on Q, K (per head, NeoX) ---
    for (int h = 0; h < n_head; ++h) {
        phi3_kernel_rope_neox_f32(q + h * head_dim, q + h * head_dim, /*factors=*/nullptr,
                                  cx.rope.n_rot, head_dim, pos, cx.rope.freq_base);
    }
    for (int h = 0; h < n_head_kv; ++h) {
        phi3_kernel_rope_neox_f32(k + h * head_dim, k + h * head_dim, /*factors=*/nullptr,
                                  cx.rope.n_rot, head_dim, pos, cx.rope.freq_base);
    }
    if (capture && capture->Q_post_rope) std::memcpy(capture->Q_post_rope, q, n_embd_q * sizeof(float));
    if (capture && capture->K_post_rope) std::memcpy(capture->K_post_rope, k, n_embd_kv * sizeof(float));
    if (capture && capture->V_cur)       std::memcpy(capture->V_cur, v, n_embd_kv * sizeof(float));

    // --- 4. Write K, V to cache at position `pos` (F16) ---
    //     Layout: cache[(pos * n_head_kv + h) * head_dim + d]
    {
        ggml_fp16_t * K_pos = cx.kv.K[il] + (size_t) pos * (size_t) n_embd_kv;
        ggml_fp16_t * V_pos = cx.kv.V[il] + (size_t) pos * (size_t) n_embd_kv;
        ggml_fp32_to_fp16_row(k, K_pos, n_embd_kv);
        ggml_fp32_to_fp16_row(v, V_pos, n_embd_kv);
    }
    const int new_len = pos + 1;
    if (cx.kv.current_len < new_len) cx.kv.current_len = new_len;

    // --- 5. Attention (per head) ---
    const float scale_q = 1.0f / std::sqrt((float) head_dim);
    std::vector<float> scores((size_t) new_len);
    for (int h = 0; h < n_head; ++h) {
        const float * q_h = q + h * head_dim;
        // K dots
        float max_s = -INFINITY;
        for (int p = 0; p < new_len; ++p) {
            const ggml_fp16_t * k_p = cx.kv.K[il] + ((size_t) p * n_head_kv + h) * head_dim;
            double acc = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                acc += (double) q_h[d] * (double) ggml_fp16_to_fp32(k_p[d]);
            }
            const float s = (float) (acc * (double) scale_q);
            scores[p] = s;
            if (s > max_s) max_s = s;
        }
        // softmax
        double sum_exp = 0.0;
        for (int p = 0; p < new_len; ++p) {
            scores[p] = std::exp(scores[p] - max_s);
            sum_exp += (double) scores[p];
        }
        const float inv_sum = (float) (1.0 / sum_exp);
        // dot V
        float * ctx_h = attn_ctx.data() + h * head_dim;
        for (int d = 0; d < head_dim; ++d) ctx_h[d] = 0.0f;
        for (int p = 0; p < new_len; ++p) {
            const float w_p = scores[p] * inv_sum;
            const ggml_fp16_t * v_p = cx.kv.V[il] + ((size_t) p * n_head_kv + h) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                ctx_h[d] += w_p * ggml_fp16_to_fp32(v_p[d]);
            }
        }
    }
    if (capture && capture->attn_ctx) std::memcpy(capture->attn_ctx, attn_ctx.data(), n_embd * sizeof(float));

    // --- 6. wo ---
    f32_matmul_row(cx.scratch.w_f32_wo.data(), attn_ctx.data(), attn_out_buf.data(), n_embd, n_embd);
    if (capture && capture->attn_out) std::memcpy(capture->attn_out, attn_out_buf.data(), n_embd * sizeof(float));

    // --- 7. Residual 1 ---
    for (int i = 0; i < n_embd; ++i) x_after_res1[i] = x_inout[i] + attn_out_buf[i];
    if (capture && capture->x_after_res1) std::memcpy(capture->x_after_res1, x_after_res1.data(), n_embd * sizeof(float));

    // --- 8. FFN RMSNorm ---
    phi3_kernel_rmsnorm_mul_f32(norm2.data(), x_after_res1.data(),
                                cx.scratch.w_f32_ffn_norm.data(),
                                n_embd, cx.eps);
    if (capture && capture->norm2) std::memcpy(capture->norm2, norm2.data(), n_embd * sizeof(float));

    // --- 9. ffn_up (gate||up fused) ---
    f32_matmul_row(cx.scratch.w_f32_ffn_up.data(), norm2.data(), upgate.data(), 2 * n_ff, n_embd);
    if (capture && capture->upgate) std::memcpy(capture->upgate, upgate.data(), 2 * n_ff * sizeof(float));

    // --- 10. SwiGLU: ff[i] = silu(gate[i]) * up[i]
    //         gate = upgate[0..n_ff), up = upgate[n_ff..2*n_ff)
    for (int i = 0; i < n_ff; ++i) {
        const float g = upgate[i];
        const float u = upgate[n_ff + i];
        const float silu = g / (1.0f + std::exp(-g));
        ff[i] = silu * u;
    }
    if (capture && capture->ff) std::memcpy(capture->ff, ff.data(), n_ff * sizeof(float));

    // --- 11. ffn_down ---
    f32_matmul_row(cx.scratch.w_f32_ffn_down.data(), ff.data(), ffn_out_buf.data(), n_embd, n_ff);
    if (capture && capture->ffn_out) std::memcpy(capture->ffn_out, ffn_out_buf.data(), n_embd * sizeof(float));

    // --- 12. Residual 2 ---
    for (int i = 0; i < n_embd; ++i) x_inout[i] = x_after_res1[i] + ffn_out_buf[i];

    return true;
}


// ---------------------------------------------------------------------------
// Oracle: build a ggml single-layer F32 graph that mirrors our naive impl.
//
// All weights are F32 (dequantized) — we are testing layout, not Q-quant
// precision. The prior K/V are passed in already-F16-rounded-to-F32 to
// match our F16 KV cache rounding. The new K/V (computed by oracle's
// wqkv+rope) is also rounded through F16 via ggml_cast to match the hand
// path's "write F16 then read F16" semantics.
// ---------------------------------------------------------------------------
namespace {

// Round an F32 array through F16 in-place (host-side).
void round_through_f16(float * x, int n) {
    std::vector<ggml_fp16_t> tmp((size_t) n);
    ggml_fp32_to_fp16_row(x, tmp.data(), n);
    ggml_fp16_to_fp32_row(tmp.data(), x, n);
}

bool buffers_close(const char * stage, const float * a, const float * b, int n,
                   float atol, float rtol, std::string & error) {
    int worst_i = -1;
    float worst_diff = 0.0f;
    float worst_a = 0.0f, worst_b = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float ref  = std::fabs(b[i]);
        if (diff > atol && diff > rtol * ref) {
            if (diff > worst_diff) {
                worst_diff = diff; worst_i = i; worst_a = a[i]; worst_b = b[i];
            }
        }
    }
    if (worst_i >= 0) {
        std::ostringstream oss;
        oss << "phi3 layer self-test: stage '" << stage << "' diverges at i=" << worst_i
            << " ours=" << worst_a << " oracle=" << worst_b << " diff=" << worst_diff
            << " (atol=" << atol << " rtol=" << rtol << " n=" << n << ")";
        error = oss.str();
        return false;
    }
    fprintf(stderr, "phi3 layer self-test: stage %-16s OK (n=%d, atol=%g, rtol=%g)\n",
            stage, n, atol, rtol);
    return true;
}

} // namespace


bool phi3_layer_self_test(const llama_model * model, std::string & error) {
    error.clear();
    if (model == nullptr) { error = "phi3_layer_self_test: model is null"; return false; }

    // ----- Resolve weights -----
    Phi3Weights W;
    if (!phi3_weights_resolve(model, W, error)) return false;
    const int n_embd     = W.n_embd;
    const int n_head     = W.n_head;
    const int n_head_kv  = W.n_head_kv;
    const int head_dim   = W.n_embd_head;
    const int n_embd_kv  = head_dim * n_head_kv;
    const int n_embd_q   = head_dim * n_head;
    const int n_qkv      = W.n_qkv;
    const int n_ff       = W.n_ff;

    // ----- Build a fused-ctx (no lctx — we set sensible defaults) -----
    Phi3FusedCtx cx;
    {
        const int test_ctx_max = 8;
        if (!phi3_fused_ctx_init(cx, W, /*pool=*/nullptr, model, /*lctx=*/nullptr,
                                 test_ctx_max, /*f32_debug=*/true, error)) {
            return false;
        }
    }
    if (!phi3_layer_warmup_f32_mirrors(cx, 0, error)) { phi3_fused_ctx_free(cx); return false; }

    // ----- Inputs -----
    const int n_prior = 4;
    const int cur_pos = n_prior;  // 4
    const int total   = n_prior + 1;
    std::mt19937 rng(2026u);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> x_in(n_embd);
    for (int i = 0; i < n_embd; ++i) x_in[i] = nd(rng);

    // Prior K/V — random, then rounded through F16 so both paths see the
    // same F16-quantized prior cache.
    std::vector<float> prior_K_f32((size_t) n_prior * n_embd_kv);
    std::vector<float> prior_V_f32((size_t) n_prior * n_embd_kv);
    for (auto & x : prior_K_f32) x = nd(rng);
    for (auto & x : prior_V_f32) x = nd(rng);
    round_through_f16(prior_K_f32.data(), (int) prior_K_f32.size());
    round_through_f16(prior_V_f32.data(), (int) prior_V_f32.size());

    // ----- Pre-load Phi3KV with the F16-rounded prior data -----
    for (int p = 0; p < n_prior; ++p) {
        ggml_fp16_t * K_p = cx.kv.K[0] + (size_t) p * n_embd_kv;
        ggml_fp16_t * V_p = cx.kv.V[0] + (size_t) p * n_embd_kv;
        ggml_fp32_to_fp16_row(prior_K_f32.data() + p * n_embd_kv, K_p, n_embd_kv);
        ggml_fp32_to_fp16_row(prior_V_f32.data() + p * n_embd_kv, V_p, n_embd_kv);
    }
    cx.kv.current_len = n_prior;

    // ----- Our path with capture -----
    std::vector<float> cap_norm1(n_embd), cap_qkv(n_qkv), cap_Q(n_embd_q), cap_K(n_embd_kv), cap_V(n_embd_kv);
    std::vector<float> cap_attn_ctx(n_embd), cap_attn_out(n_embd), cap_res1(n_embd);
    std::vector<float> cap_norm2(n_embd), cap_upgate(2 * n_ff), cap_ff(n_ff), cap_ffn_out(n_embd);
    Phi3LayerCapture cap_ours{
        cap_norm1.data(), cap_qkv.data(), cap_Q.data(), cap_K.data(), cap_V.data(),
        cap_attn_ctx.data(), cap_attn_out.data(), cap_res1.data(),
        cap_norm2.data(), cap_upgate.data(), cap_ff.data(), cap_ffn_out.data()
    };
    std::vector<float> x_out_ours(n_embd);
    std::memcpy(x_out_ours.data(), x_in.data(), n_embd * sizeof(float));
    if (!phi3_layer_forward_f32(cx, /*il=*/0, cur_pos, x_out_ours.data(), &cap_ours, error)) {
        phi3_fused_ctx_free(cx);
        return false;
    }

    // =====================================================================
    // ----- Oracle: ggml-cpu graph for one token at pos=cur_pos -----
    // =====================================================================
    // Strategy: build an F32 graph that does
    //   norm1 → wqkv(F32) → split Q|K|V → rope(Q), rope(K)
    //   → round(K_new, V) through F16 (via ggml_cast F32→F16→F32)
    //   → concat(prior_K, K_new) on dim 2 → same for V
    //   → permute(Q,K,V), kq = mul_mat(K,Q), softmax(kq*scale_q),
    //     V_T = cont(transpose(V)), kqv = mul_mat(V_T, kq)
    //   → wo(F32), residual1
    //   → norm2, ffn_up, swiglu, ffn_down, residual2.
    //
    // We capture the SAME intermediates as cap_ours via separate output
    // tensors (each added to the graph as build_forward_expand sources).
    //
    // 1 GiB ggml arena (F32 weight mirrors alone are ~450 MiB: wqkv 110 +
    // wo 38 + ffn_up 200 + ffn_down 100; plus intermediates and work data).
    const size_t graph_mem = (size_t) 1024 * 1024 * 1024;
    std::vector<uint8_t> arena(graph_mem);
    ggml_init_params ip{ graph_mem, arena.data(), false };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { error = "phi3_layer_self_test: ggml_init failed"; phi3_fused_ctx_free(cx); return false; }

    // Helper to create + populate an F32 tensor.
    auto new_f32 = [&](int64_t a, int64_t b, int64_t c, const float * src, int64_t nelt) -> ggml_tensor * {
        ggml_tensor * t;
        if (c > 1)       t = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, a, b, c);
        else if (b > 1)  t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, a, b);
        else             t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, a);
        if (src) std::memcpy(t->data, src, (size_t) nelt * sizeof(float));
        return t;
    };

    // Inputs.
    ggml_tensor * t_x       = new_f32(n_embd, 1, 1, x_in.data(), n_embd);
    ggml_tensor * t_priorK  = new_f32(head_dim, n_head_kv, n_prior, prior_K_f32.data(), (int64_t) n_prior * n_embd_kv);
    ggml_tensor * t_priorV  = new_f32(head_dim, n_head_kv, n_prior, prior_V_f32.data(), (int64_t) n_prior * n_embd_kv);

    // F32 weight mirrors (just borrow the buffers — no_alloc=false means ggml
    // allocates the tensor data; we memcpy).
    ggml_tensor * t_attn_norm = new_f32(n_embd,        1, 1, cx.scratch.w_f32_attn_norm.data(), n_embd);
    ggml_tensor * t_wqkv      = new_f32(n_embd,    n_qkv, 1, cx.scratch.w_f32_wqkv.data(),      (int64_t) n_embd * n_qkv);
    ggml_tensor * t_wo        = new_f32(n_embd,   n_embd, 1, cx.scratch.w_f32_wo.data(),        (int64_t) n_embd * n_embd);
    ggml_tensor * t_ffn_norm  = new_f32(n_embd,        1, 1, cx.scratch.w_f32_ffn_norm.data(),  n_embd);
    ggml_tensor * t_ffn_up    = new_f32(n_embd, 2 * n_ff, 1, cx.scratch.w_f32_ffn_up.data(),    (int64_t) n_embd * 2 * n_ff);
    ggml_tensor * t_ffn_down  = new_f32(n_ff,    n_embd,  1, cx.scratch.w_f32_ffn_down.data(),  (int64_t) n_ff * n_embd);

    // Position tensor for ggml_rope_ext: [n_tokens=1] int32 with value [cur_pos].
    ggml_tensor * t_pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ((int32_t *) t_pos->data)[0] = cur_pos;

    // ---- Graph: norm1 = rms_norm(x) * attn_norm ----
    ggml_tensor * norm1 = ggml_rms_norm(gctx, t_x, cx.eps);
    norm1 = ggml_mul(gctx, norm1, t_attn_norm);

    // ---- wqkv: qkv = mul_mat(W, norm1) ----
    ggml_tensor * qkv = ggml_mul_mat(gctx, t_wqkv, norm1);   // [n_qkv, 1]

    // ---- Split Q/K/V (3-D views) ----
    // Q view: offset 0, shape [head_dim, n_head,    1]
    // K view: offset n_embd_q*sizeof(f32), shape [head_dim, n_head_kv, 1]
    // V view: offset (n_embd_q + n_embd_kv)*sizeof(f32), shape [head_dim, n_head_kv, 1]
    ggml_tensor * Q = ggml_view_3d(gctx, qkv, head_dim, n_head,    1,
                                   head_dim * sizeof(float), qkv->nb[1], 0);
    ggml_tensor * K = ggml_view_3d(gctx, qkv, head_dim, n_head_kv, 1,
                                   head_dim * sizeof(float), qkv->nb[1],
                                   (size_t) n_embd_q * sizeof(float));
    ggml_tensor * V = ggml_view_3d(gctx, qkv, head_dim, n_head_kv, 1,
                                   head_dim * sizeof(float), qkv->nb[1],
                                   (size_t) (n_embd_q + n_embd_kv) * sizeof(float));

    // ---- RoPE on Q and K (NEOX) ----
    Q = ggml_rope_ext(gctx, Q, t_pos, /*factors=*/nullptr,
                      cx.rope.n_rot, GGML_ROPE_TYPE_NEOX, cx.rope.n_ctx_orig,
                      cx.rope.freq_base, cx.rope.freq_scale,
                      cx.rope.ext_factor, cx.rope.attn_factor,
                      cx.rope.beta_fast, cx.rope.beta_slow);
    K = ggml_rope_ext(gctx, K, t_pos, /*factors=*/nullptr,
                      cx.rope.n_rot, GGML_ROPE_TYPE_NEOX, cx.rope.n_ctx_orig,
                      cx.rope.freq_base, cx.rope.freq_scale,
                      cx.rope.ext_factor, cx.rope.attn_factor,
                      cx.rope.beta_fast, cx.rope.beta_slow);

    // ---- Round new K/V through F16 to match the hand path ----
    K = ggml_cast(gctx, K, GGML_TYPE_F16);
    K = ggml_cast(gctx, K, GGML_TYPE_F32);
    V = ggml_cast(gctx, V, GGML_TYPE_F16);
    V = ggml_cast(gctx, V, GGML_TYPE_F32);

    // ---- Concat prior + new on dim 2 (positions) ----
    ggml_tensor * K_all = ggml_concat(gctx, t_priorK, K, 2);   // [head_dim, n_head_kv, total]
    ggml_tensor * V_all = ggml_concat(gctx, t_priorV, V, 2);

    // ---- Attention: kq = mul_mat(K_all_perm, Q_perm) then softmax ----
    // build_attn_mha pattern: permute (0,2,1,3) so head becomes outer.
    ggml_tensor * Qp = ggml_permute(gctx, Q,     0, 2, 1, 3);  // [head_dim, n_tokens_q=1, n_head, 1]
    ggml_tensor * Kp = ggml_permute(gctx, K_all, 0, 2, 1, 3);  // [head_dim, n_tokens_kv=total, n_head_kv, 1]
    ggml_tensor * Vp = ggml_permute(gctx, V_all, 0, 2, 1, 3);  // [head_dim, n_tokens_kv=total, n_head_kv, 1]
    Qp = ggml_cont(gctx, Qp);
    Kp = ggml_cont(gctx, Kp);
    Vp = ggml_cont(gctx, Vp);

    ggml_tensor * kq = ggml_mul_mat(gctx, Kp, Qp);  // [n_tokens_kv, n_tokens_q=1, n_head, 1]
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    // soft_max_ext applies scale; we pass scale=1/sqrt(head_dim) so Q is effectively scaled.
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);
    kq = ggml_soft_max_ext(gctx, kq, /*mask=*/nullptr, kq_scale, /*max_bias=*/0.0f);

    // V_T: transpose then contiguous so positions are dim 0 (contraction axis).
    ggml_tensor * Vt = ggml_cont(gctx, ggml_transpose(gctx, Vp));  // [n_tokens_kv, head_dim, n_head_kv, 1] permuted

    ggml_tensor * kqv = ggml_mul_mat(gctx, Vt, kq);  // [head_dim, n_tokens_q=1, n_head, 1]
    // Permute back so heads are middle dim, then flatten to [n_embd, 1].
    kqv = ggml_permute(gctx, kqv, 0, 2, 1, 3);       // [head_dim, n_head, n_tokens_q=1, 1]
    kqv = ggml_cont(gctx, kqv);
    ggml_tensor * attn_ctx_t = ggml_reshape_2d(gctx, kqv, n_embd, 1);

    // ---- wo ----
    ggml_tensor * attn_out_t = ggml_mul_mat(gctx, t_wo, attn_ctx_t);

    // ---- Residual 1 ----
    ggml_tensor * x_after_res1_t = ggml_add(gctx, t_x, attn_out_t);

    // ---- ffn norm ----
    ggml_tensor * norm2 = ggml_rms_norm(gctx, x_after_res1_t, cx.eps);
    norm2 = ggml_mul(gctx, norm2, t_ffn_norm);

    // ---- ffn_up + swiglu ----
    ggml_tensor * upgate_t = ggml_mul_mat(gctx, t_ffn_up, norm2);  // [2*n_ff, 1]
    ggml_tensor * ff_t     = ggml_swiglu(gctx, upgate_t);          // [n_ff, 1]

    // ---- ffn_down + residual 2 ----
    ggml_tensor * ffn_out_t = ggml_mul_mat(gctx, t_ffn_down, ff_t);
    ggml_tensor * x_final_t = ggml_add(gctx, x_after_res1_t, ffn_out_t);

    // Build graph. Mark every comparison stage as a forward leaf so it
    // gets computed and we can inspect its data after compute.
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 256, false);
    ggml_build_forward_expand(gf, norm1);
    ggml_build_forward_expand(gf, qkv);
    ggml_build_forward_expand(gf, Q);
    ggml_build_forward_expand(gf, K);
    ggml_build_forward_expand(gf, V);
    ggml_build_forward_expand(gf, attn_ctx_t);
    ggml_build_forward_expand(gf, attn_out_t);
    ggml_build_forward_expand(gf, x_after_res1_t);
    ggml_build_forward_expand(gf, norm2);
    ggml_build_forward_expand(gf, upgate_t);
    ggml_build_forward_expand(gf, ff_t);
    ggml_build_forward_expand(gf, ffn_out_t);
    ggml_build_forward_expand(gf, x_final_t);

    const ggml_status status = ggml_graph_compute_with_ctx(gctx, gf, /*n_threads=*/1);
    if (status != GGML_STATUS_SUCCESS) {
        error = "phi3_layer_self_test: ggml_graph_compute_with_ctx failed";
        ggml_free(gctx);
        phi3_fused_ctx_free(cx);
        return false;
    }

    // Read oracle data into host buffers.
    auto read_f32 = [](ggml_tensor * t, std::vector<float> & dst) {
        const size_t n = (size_t) ggml_nelements(t);
        dst.assign(n, 0.0f);
        std::memcpy(dst.data(), t->data, n * sizeof(float));
    };
    std::vector<float> o_norm1, o_qkv, o_Q, o_K, o_V, o_attn_ctx, o_attn_out, o_res1;
    std::vector<float> o_norm2, o_upgate, o_ff, o_ffn_out, o_final;
    read_f32(norm1,             o_norm1);
    read_f32(qkv,               o_qkv);
    read_f32(Q,                 o_Q);
    read_f32(K,                 o_K);
    read_f32(V,                 o_V);
    read_f32(attn_ctx_t,        o_attn_ctx);
    read_f32(attn_out_t,        o_attn_out);
    read_f32(x_after_res1_t,    o_res1);
    read_f32(norm2,             o_norm2);
    read_f32(upgate_t,          o_upgate);
    read_f32(ff_t,              o_ff);
    read_f32(ffn_out_t,         o_ffn_out);
    read_f32(x_final_t,         o_final);

    // ----- Compare staged outputs -----
    // Tolerances: most stages should be tight (rel 5e-3). Final stage cumulates
    // ~12 ops worth of F32 reduction reorder noise; allow looser.
    bool ok = true;
    auto chk = [&](const char * name, const float * a, const float * b, int n,
                   float atol, float rtol) {
        if (!ok) return;
        if (!buffers_close(name, a, b, n, atol, rtol, error)) ok = false;
    };
    chk("norm1",          cap_norm1.data(),    o_norm1.data(),    n_embd,         1e-3f, 5e-3f);
    chk("qkv",            cap_qkv.data(),      o_qkv.data(),      n_qkv,          5e-3f, 5e-3f);
    chk("Q_post_rope",    cap_Q.data(),        o_Q.data(),        n_embd_q,       5e-3f, 5e-3f);
    chk("K_post_rope",    cap_K.data(),        o_K.data(),        n_embd_kv,      5e-3f, 5e-3f);
    chk("V_cur",          cap_V.data(),        o_V.data(),        n_embd_kv,      5e-3f, 5e-3f);
    chk("attn_ctx",       cap_attn_ctx.data(), o_attn_ctx.data(), n_embd,         1e-2f, 1e-2f);
    chk("attn_out",       cap_attn_out.data(), o_attn_out.data(), n_embd,         1e-2f, 1e-2f);
    chk("x_after_res1",   cap_res1.data(),     o_res1.data(),     n_embd,         1e-2f, 1e-2f);
    chk("norm2",          cap_norm2.data(),    o_norm2.data(),    n_embd,         1e-2f, 1e-2f);
    chk("upgate",         cap_upgate.data(),   o_upgate.data(),   2 * n_ff,       1e-2f, 1e-2f);
    chk("ff_swiglu",      cap_ff.data(),       o_ff.data(),       n_ff,           1e-2f, 1e-2f);
    chk("ffn_out",        cap_ffn_out.data(),  o_ffn_out.data(),  n_embd,         2e-2f, 2e-2f);
    chk("x_final",        x_out_ours.data(),   o_final.data(),    n_embd,         2e-2f, 2e-2f);

    ggml_free(gctx);
    phi3_fused_ctx_free(cx);

    if (ok) {
        fprintf(stderr, "phi3 layer self-test: PASS (13 stages, 1 token at pos=%d, %d prior K/V positions)\n",
                cur_pos, n_prior);
    }
    return ok;
}

// ===========================================================================
// A2.4a ? Full-network F32 forward + cross-layer self-test.
// ===========================================================================

namespace {

// Per-row dequant of a 2-D weight tensor (one outer-dim row).
bool dequant_row_to_f32(const ggml_tensor * src, int row, float * dst, std::string & error) {
    if (src == nullptr) { error = "dequant_row_to_f32: src is null"; return false; }
    const int64_t n0 = src->ne[0];
    const int64_t n1 = src->ne[1];
    if (row < 0 || row >= n1) {
        std::ostringstream oss; oss << "dequant_row_to_f32: row=" << row << " out of [0," << n1 << ")";
        error = oss.str(); return false;
    }
    const uint8_t * row_p = (const uint8_t *) src->data + (size_t) row * src->nb[1];
    if (src->type == GGML_TYPE_F32) {
        std::memcpy(dst, row_p, (size_t) n0 * sizeof(float));
        return true;
    }
    if (src->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) row_p, dst, n0);
        return true;
    }
    const auto * traits = ggml_get_type_traits(src->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        error = std::string("dequant_row_to_f32: no to_float for type ") + ggml_type_name(src->type);
        return false;
    }
    traits->to_float(row_p, dst, n0);
    return true;
}

struct CompareStats {
    int    n        = 0;
    float  max_abs  = 0.0f;
    float  mean_abs = 0.0f;
    float  rmse     = 0.0f;
    float  max_rel  = 0.0f;
    int    worst_i  = -1;
    float  worst_a  = 0.0f;
    float  worst_b  = 0.0f;
};

CompareStats compute_stats(const float * a, const float * b, int n) {
    CompareStats s; s.n = n;
    double sum_abs = 0.0;
    double sum_sq  = 0.0;
    for (int i = 0; i < n; ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float ref  = std::fabs(b[i]);
        const float rel  = (ref > 1e-30f) ? (diff / ref) : 0.0f;
        sum_abs += diff;
        sum_sq  += (double) diff * (double) diff;
        if (diff > s.max_abs) {
            s.max_abs = diff;
            s.worst_i = i;
            s.worst_a = a[i];
            s.worst_b = b[i];
        }
        if (rel > s.max_rel) s.max_rel = rel;
    }
    s.mean_abs = (float) (sum_abs / (double) n);
    s.rmse     = (float) std::sqrt(sum_sq / (double) n);
    return s;
}

void print_stats(const char * stage, const CompareStats & s, float tol_max, float tol_rmse, bool ok) {
    fprintf(stderr,
            "phi3 full self-test: %-22s %s  n=%-6d  max_abs=%-10.4g rmse=%-10.4g mean=%-10.4g max_rel=%-10.4g"
            "  (tol_max=%.2g tol_rmse=%.2g) worst_i=%d ours=%.6g oracle=%.6g\n",
            stage, ok ? "OK  " : "FAIL", s.n, s.max_abs, s.rmse, s.mean_abs, s.max_rel,
            tol_max, tol_rmse, s.worst_i, s.worst_a, s.worst_b);
}

bool check_close_stats(const char * stage, const float * a, const float * b, int n,
                       float tol_max, float tol_rmse, std::string & error) {
    CompareStats s = compute_stats(a, b, n);
    const bool ok = (s.max_abs <= tol_max) && (s.rmse <= tol_rmse);
    print_stats(stage, s, tol_max, tol_rmse, ok);
    if (!ok) {
        std::ostringstream oss;
        oss << "phi3 full self-test: stage '" << stage << "' FAIL"
            << " max_abs=" << s.max_abs << " (tol=" << tol_max << ")"
            << " rmse=" << s.rmse << " (tol=" << tol_rmse << ")"
            << " worst_i=" << s.worst_i << " ours=" << s.worst_a << " oracle=" << s.worst_b;
        error = oss.str();
    }
    return ok;
}

int argmax_f32(const float * x, int n) {
    int idx = 0;
    float best = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > best) { best = x[i]; idx = i; }
    }
    return idx;
}

// Returns the top-K indices of x in descending order. K must be <= n.
std::vector<int> top_k_indices(const float * x, int n, int k) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [x](int a, int b){ return x[a] > x[b]; });
    idx.resize(k);
    return idx;
}

// Runs a single decoder layer through a ggml-cpu oracle graph using F32
// weights from cx.scratch.w_f32_* (caller must warm them first via
// phi3_layer_warmup_f32_mirrors). prior_K_f32/prior_V_f32 are pre-F16-
// rounded host F32 buffers of shape [n_prior * n_embd_kv] (NULL allowed
// when n_prior == 0).
// Outputs:
//   x_out_f32[n_embd]              ? residual after this layer (F32)
//   K_new_f16_round_f32[n_embd_kv] ? new K_pos after F16 round-trip (F32)
//   V_new_f16_round_f32[n_embd_kv] ? new V_pos after F16 round-trip (F32)
// The arena buffer is owned by the caller and reused across layers.
bool oracle_run_one_layer_f32(
        Phi3FusedCtx       & cx,
        int                  pos,
        int                  n_prior,
        const float        * x_in_f32,
        const float        * prior_K_f32,
        const float        * prior_V_f32,
        std::vector<uint8_t> & arena,
        float              * x_out_f32,
        float              * K_new_f16_round_f32,
        float              * V_new_f16_round_f32,
        std::string        & error) {
    const Phi3Weights & W = *cx.w;
    const int n_embd     = W.n_embd;
    const int n_head     = W.n_head;
    const int n_head_kv  = W.n_head_kv;
    const int head_dim   = W.n_embd_head;
    const int n_embd_kv  = head_dim * n_head_kv;
    const int n_embd_q   = head_dim * n_head;
    const int n_qkv      = W.n_qkv;
    const int n_ff       = W.n_ff;
    const int total      = n_prior + 1;

    ggml_init_params ip{ arena.size(), arena.data(), false };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { error = "oracle_run_one_layer_f32: ggml_init failed"; return false; }

    auto new_f32 = [&](int64_t a, int64_t b, int64_t c, const float * src, int64_t nelt) -> ggml_tensor * {
        ggml_tensor * t;
        if (c > 1)       t = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, a, b, c);
        else if (b > 1)  t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, a, b);
        else             t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, a);
        if (src) std::memcpy(t->data, src, (size_t) nelt * sizeof(float));
        return t;
    };

    ggml_tensor * t_x         = new_f32(n_embd, 1, 1, x_in_f32, n_embd);
    ggml_tensor * t_attn_norm = new_f32(n_embd,        1, 1, cx.scratch.w_f32_attn_norm.data(), n_embd);
    ggml_tensor * t_wqkv      = new_f32(n_embd,    n_qkv, 1, cx.scratch.w_f32_wqkv.data(),      (int64_t) n_embd * n_qkv);
    ggml_tensor * t_wo        = new_f32(n_embd,   n_embd, 1, cx.scratch.w_f32_wo.data(),        (int64_t) n_embd * n_embd);
    ggml_tensor * t_ffn_norm  = new_f32(n_embd,        1, 1, cx.scratch.w_f32_ffn_norm.data(),  n_embd);
    ggml_tensor * t_ffn_up    = new_f32(n_embd, 2 * n_ff, 1, cx.scratch.w_f32_ffn_up.data(),    (int64_t) n_embd * 2 * n_ff);
    ggml_tensor * t_ffn_down  = new_f32(n_ff,    n_embd,  1, cx.scratch.w_f32_ffn_down.data(),  (int64_t) n_ff * n_embd);

    ggml_tensor * t_pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ((int32_t *) t_pos->data)[0] = pos;

    // ---- norm1 = rms_norm(x) * attn_norm ----
    ggml_tensor * norm1 = ggml_rms_norm(gctx, t_x, cx.eps);
    norm1 = ggml_mul(gctx, norm1, t_attn_norm);

    // ---- wqkv ----
    ggml_tensor * qkv = ggml_mul_mat(gctx, t_wqkv, norm1);

    // ---- Split Q/K/V ----
    ggml_tensor * Q = ggml_view_3d(gctx, qkv, head_dim, n_head,    1,
                                   head_dim * sizeof(float), qkv->nb[1], 0);
    ggml_tensor * K = ggml_view_3d(gctx, qkv, head_dim, n_head_kv, 1,
                                   head_dim * sizeof(float), qkv->nb[1],
                                   (size_t) n_embd_q * sizeof(float));
    ggml_tensor * V = ggml_view_3d(gctx, qkv, head_dim, n_head_kv, 1,
                                   head_dim * sizeof(float), qkv->nb[1],
                                   (size_t) (n_embd_q + n_embd_kv) * sizeof(float));

    // ---- RoPE on Q and K (NEOX) ----
    Q = ggml_rope_ext(gctx, Q, t_pos, /*factors=*/nullptr,
                      cx.rope.n_rot, GGML_ROPE_TYPE_NEOX, cx.rope.n_ctx_orig,
                      cx.rope.freq_base, cx.rope.freq_scale,
                      cx.rope.ext_factor, cx.rope.attn_factor,
                      cx.rope.beta_fast, cx.rope.beta_slow);
    K = ggml_rope_ext(gctx, K, t_pos, /*factors=*/nullptr,
                      cx.rope.n_rot, GGML_ROPE_TYPE_NEOX, cx.rope.n_ctx_orig,
                      cx.rope.freq_base, cx.rope.freq_scale,
                      cx.rope.ext_factor, cx.rope.attn_factor,
                      cx.rope.beta_fast, cx.rope.beta_slow);

    // ---- Round new K/V through F16 to match the hand path ----
    K = ggml_cast(gctx, K, GGML_TYPE_F16);
    K = ggml_cast(gctx, K, GGML_TYPE_F32);
    V = ggml_cast(gctx, V, GGML_TYPE_F16);
    V = ggml_cast(gctx, V, GGML_TYPE_F32);

    // ---- Concat prior + new on dim 2 (positions) ----
    ggml_tensor * K_all;
    ggml_tensor * V_all;
    if (n_prior > 0) {
        ggml_tensor * t_priorK = new_f32(head_dim, n_head_kv, n_prior, prior_K_f32, (int64_t) n_prior * n_embd_kv);
        ggml_tensor * t_priorV = new_f32(head_dim, n_head_kv, n_prior, prior_V_f32, (int64_t) n_prior * n_embd_kv);
        K_all = ggml_concat(gctx, t_priorK, K, 2);
        V_all = ggml_concat(gctx, t_priorV, V, 2);
    } else {
        K_all = K;
        V_all = V;
    }

    // ---- Attention ----
    ggml_tensor * Qp = ggml_cont(gctx, ggml_permute(gctx, Q,     0, 2, 1, 3));
    ggml_tensor * Kp = ggml_cont(gctx, ggml_permute(gctx, K_all, 0, 2, 1, 3));
    ggml_tensor * Vp = ggml_cont(gctx, ggml_permute(gctx, V_all, 0, 2, 1, 3));

    ggml_tensor * kq = ggml_mul_mat(gctx, Kp, Qp);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);
    kq = ggml_soft_max_ext(gctx, kq, /*mask=*/nullptr, kq_scale, /*max_bias=*/0.0f);

    ggml_tensor * Vt  = ggml_cont(gctx, ggml_transpose(gctx, Vp));
    ggml_tensor * kqv = ggml_mul_mat(gctx, Vt, kq);
    kqv = ggml_cont(gctx, ggml_permute(gctx, kqv, 0, 2, 1, 3));
    ggml_tensor * attn_ctx_t = ggml_reshape_2d(gctx, kqv, n_embd, 1);

    // ---- wo + residual 1 ----
    ggml_tensor * attn_out_t    = ggml_mul_mat(gctx, t_wo, attn_ctx_t);
    ggml_tensor * x_after_res1  = ggml_add(gctx, t_x, attn_out_t);

    // ---- ffn norm + ffn_up + swiglu + ffn_down + residual 2 ----
    ggml_tensor * norm2 = ggml_rms_norm(gctx, x_after_res1, cx.eps);
    norm2 = ggml_mul(gctx, norm2, t_ffn_norm);
    ggml_tensor * upgate_t = ggml_mul_mat(gctx, t_ffn_up, norm2);
    ggml_tensor * ff_t     = ggml_swiglu(gctx, upgate_t);
    ggml_tensor * ffn_out  = ggml_mul_mat(gctx, t_ffn_down, ff_t);
    ggml_tensor * x_final  = ggml_add(gctx, x_after_res1, ffn_out);

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 256, false);
    ggml_build_forward_expand(gf, x_final);
    ggml_build_forward_expand(gf, K);  // F16-rounded new K
    ggml_build_forward_expand(gf, V);  // F16-rounded new V

    const ggml_status status = ggml_graph_compute_with_ctx(gctx, gf, /*n_threads=*/1);
    if (status != GGML_STATUS_SUCCESS) {
        error = "oracle_run_one_layer_f32: ggml_graph_compute_with_ctx failed";
        ggml_free(gctx);
        return false;
    }

    std::memcpy(x_out_f32, x_final->data, (size_t) n_embd * sizeof(float));
    std::memcpy(K_new_f16_round_f32, K->data, (size_t) n_embd_kv * sizeof(float));
    std::memcpy(V_new_f16_round_f32, V->data, (size_t) n_embd_kv * sizeof(float));

    ggml_free(gctx);
    return true;
}

} // namespace


bool phi3_full_forward_f32(
        Phi3FusedCtx       & cx,
        int                  token_id,
        int                  pos,
        float              * out_logits,
        float              * tok_embd_capture,
        float * const      * capture_x_per_layer,
        std::string        & error) {
    error.clear();
    if (cx.w == nullptr)          { error = "phi3_full_forward_f32: cx.w is null"; return false; }
    if (!cx.scratch.f32_debug)    { error = "phi3_full_forward_f32: requires f32_debug"; return false; }
    if (out_logits == nullptr)    { error = "phi3_full_forward_f32: out_logits is null"; return false; }

    const Phi3Weights & W = *cx.w;
    const int n_embd  = W.n_embd;
    const int n_layer = W.n_layer;
    const int n_vocab = W.n_vocab;

    if (token_id < 0 || token_id >= n_vocab) {
        std::ostringstream oss; oss << "phi3_full_forward_f32: token_id=" << token_id << " out of [0," << n_vocab << ")";
        error = oss.str(); return false;
    }

    // ---- 1. tok_embd lookup -> F32 ----
    std::vector<float> x(n_embd);
    if (!dequant_row_to_f32(W.tok_embd, token_id, x.data(), error)) return false;
    if (tok_embd_capture) std::memcpy(tok_embd_capture, x.data(), n_embd * sizeof(float));

    // ---- 2. Iterate 32 layers ----
    for (int il = 0; il < n_layer; ++il) {
        if (!phi3_layer_forward_f32(cx, il, pos, x.data(), /*capture=*/nullptr, error)) {
            return false;
        }
        if (capture_x_per_layer && capture_x_per_layer[il]) {
            std::memcpy(capture_x_per_layer[il], x.data(), n_embd * sizeof(float));
        }
    }

    // ---- 3. Final RMSNorm with cx.eps and output_norm weights ----
    {
        std::vector<float> w_final(n_embd);
        if (!dequant_row_to_f32(W.output_norm, 0, w_final.data(), error)) {
            // output_norm is 1-D; try as a flat copy.
            const int64_t n = ggml_nelements(W.output_norm);
            if (n != n_embd) { return false; }
            if (W.output_norm->type == GGML_TYPE_F32) {
                std::memcpy(w_final.data(), W.output_norm->data, (size_t) n_embd * sizeof(float));
                error.clear();
            } else {
                return false;
            }
        }
        std::vector<float> tmp(n_embd);
        phi3_kernel_rmsnorm_mul_f32(tmp.data(), x.data(), w_final.data(), n_embd, cx.eps);
        x.swap(tmp);
    }
    if (capture_x_per_layer && capture_x_per_layer[n_layer]) {
        std::memcpy(capture_x_per_layer[n_layer], x.data(), n_embd * sizeof(float));
    }

    // ---- 4. lm_head: per-row dequant + dot ----
    const ggml_tensor * lm_head = W.output;
    std::vector<float> row(n_embd);
    for (int v = 0; v < n_vocab; ++v) {
        if (!dequant_row_to_f32(lm_head, v, row.data(), error)) return false;
        double acc = 0.0;
        for (int d = 0; d < n_embd; ++d) acc += (double) x[d] * (double) row[d];
        out_logits[v] = (float) acc;
    }
    return true;
}


bool phi3_full_self_test(const llama_model * model, std::string & error) {
    error.clear();
    if (model == nullptr) { error = "phi3_full_self_test: model is null"; return false; }

    Phi3Weights W;
    if (!phi3_weights_resolve(model, W, error)) return false;
    const int n_embd     = W.n_embd;
    const int n_layer    = W.n_layer;
    const int n_head_kv  = W.n_head_kv;
    const int head_dim   = W.n_embd_head;
    const int n_embd_kv  = head_dim * n_head_kv;
    const int n_vocab    = W.n_vocab;

    fprintf(stderr, "phi3 full self-test: model n_layer=%d n_embd=%d n_vocab=%d output_tied_to_embd=%d\n",
            n_layer, n_embd, n_vocab, (int) W.output_tied_to_embd);

    // Token used for both sub-tests: BOS or token id 1.
    const int token_id = 1;

    // Reusable 1 GiB arena for per-layer oracle sub-graphs.
    std::vector<uint8_t> arena_layer((size_t) 1024 * 1024 * 1024);

    auto run_sub_test = [&](const char * label, int pos, int n_prior, std::string & sub_err) -> bool {
        fprintf(stderr, "\n=== phi3 full self-test: %s  (pos=%d  n_prior=%d) ===\n",
                label, pos, n_prior);

        const int test_ctx_max = std::max(pos + 1, n_prior + 1) + 1;

        Phi3FusedCtx cx;
        if (!phi3_fused_ctx_init(cx, W, /*pool=*/nullptr, model, /*lctx=*/nullptr,
                                 test_ctx_max, /*f32_debug=*/true, sub_err)) {
            return false;
        }

        // Hand path state == oracle path state at this point (both have empty KV).

        // Seed prior K/V per layer if needed. Use deterministic RNG keyed by
        // sub-test label so the two sub-tests don't accidentally share state.
        const uint32_t seed = (pos == 0) ? 2027u : 2028u;
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);

        // prior_K_f32_per_layer[il] / prior_V_f32_per_layer[il] are the
        // F16-rounded prior K/V for layer il (shape [n_prior * n_embd_kv]).
        // We allocate them whether n_prior>0 or not; empty vectors when 0.
        std::vector<std::vector<float>> prior_K_per_layer(n_layer);
        std::vector<std::vector<float>> prior_V_per_layer(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            if (n_prior > 0) {
                prior_K_per_layer[il].resize((size_t) n_prior * n_embd_kv);
                prior_V_per_layer[il].resize((size_t) n_prior * n_embd_kv);
                for (auto & v : prior_K_per_layer[il]) v = nd(rng);
                for (auto & v : prior_V_per_layer[il]) v = nd(rng);
                round_through_f16(prior_K_per_layer[il].data(), (int) prior_K_per_layer[il].size());
                round_through_f16(prior_V_per_layer[il].data(), (int) prior_V_per_layer[il].size());
                // Load prior into hand path's cx.kv at positions [0..n_prior).
                for (int p = 0; p < n_prior; ++p) {
                    ggml_fp16_t * Kp = cx.kv.K[il] + (size_t) p * n_embd_kv;
                    ggml_fp16_t * Vp = cx.kv.V[il] + (size_t) p * n_embd_kv;
                    ggml_fp32_to_fp16_row(prior_K_per_layer[il].data() + (size_t) p * n_embd_kv, Kp, n_embd_kv);
                    ggml_fp32_to_fp16_row(prior_V_per_layer[il].data() + (size_t) p * n_embd_kv, Vp, n_embd_kv);
                }
            }
        }
        cx.kv.current_len = n_prior;

        // ---- Allocate per-layer captures for hand path ----
        // 32 layers + 1 final-norm slot.
        std::vector<std::vector<float>> hand_x_per_layer(n_layer + 1, std::vector<float>(n_embd));
        std::vector<float *> hand_cap_ptrs(n_layer + 1);
        for (int i = 0; i <= n_layer; ++i) hand_cap_ptrs[i] = hand_x_per_layer[i].data();

        std::vector<float> hand_tok_embd(n_embd);
        std::vector<float> hand_logits(n_vocab);

        if (!phi3_full_forward_f32(cx, token_id, pos, hand_logits.data(),
                                   hand_tok_embd.data(), hand_cap_ptrs.data(), sub_err)) {
            phi3_fused_ctx_free(cx);
            return false;
        }

        // ---- Oracle path: re-run independently using oracle_run_one_layer_f32 ----
        // We dequant + cache F32 mirrors once per layer (warmup), then run the
        // oracle sub-graph for that layer, then move on. The KV state for the
        // oracle is held in oracle_K_per_layer / oracle_V_per_layer: each is
        // a flat F32 buffer of length total*n_embd_kv with positions
        // [0..n_prior) = prior + [n_prior] = new K/V from this layer (post
        // F16 round).
        std::vector<float> x_oracle(n_embd);
        std::memcpy(x_oracle.data(), hand_tok_embd.data(), n_embd * sizeof(float));

        // We need the F32-dequantized tok_embd row to be identical to the
        // hand path's tok_embd row, since both went through dequant_row_to_f32.
        // Sanity check: bit-equal expected.
        {
            std::vector<float> tmp(n_embd);
            if (!dequant_row_to_f32(W.tok_embd, token_id, tmp.data(), sub_err)) {
                phi3_fused_ctx_free(cx);
                return false;
            }
            if (std::memcmp(tmp.data(), hand_tok_embd.data(), (size_t) n_embd * sizeof(float)) != 0) {
                sub_err = "phi3 full self-test: tok_embd dequant disagrees between hand path and oracle";
                phi3_fused_ctx_free(cx);
                return false;
            }
            fprintf(stderr, "phi3 full self-test: tok_embd dequant matches (bit-equal)\n");
        }

        // Per-layer compare buffers for oracle.
        std::vector<std::vector<float>> oracle_x_per_layer(n_layer + 1, std::vector<float>(n_embd));

        // Tolerance schedule.
        auto tol_max_for_depth  = [](int d){ return 2e-2f + d * 6e-3f; };   // 0.02 -> 0.21 at il=31
        auto tol_rmse_for_depth = [](int d){ return 5e-3f + d * 1.5e-3f; }; // 0.005 -> 0.05 at il=31

        bool ok = true;
        for (int il = 0; il < n_layer; ++il) {
            if (!phi3_layer_warmup_f32_mirrors(cx, il, sub_err)) { ok = false; break; }

            std::vector<float> K_new(n_embd_kv);
            std::vector<float> V_new(n_embd_kv);
            std::vector<float> x_next(n_embd);

            const float * pK = (n_prior > 0) ? prior_K_per_layer[il].data() : nullptr;
            const float * pV = (n_prior > 0) ? prior_V_per_layer[il].data() : nullptr;

            if (!oracle_run_one_layer_f32(cx, pos, n_prior,
                                          x_oracle.data(), pK, pV,
                                          arena_layer,
                                          x_next.data(), K_new.data(), V_new.data(),
                                          sub_err)) { ok = false; break; }
            x_oracle.swap(x_next);
            std::memcpy(oracle_x_per_layer[il].data(), x_oracle.data(), n_embd * sizeof(float));

            char stage[64]; std::snprintf(stage, sizeof(stage), "x_after_layer_%02d", il);
            if (!check_close_stats(stage,
                                   hand_x_per_layer[il].data(),
                                   oracle_x_per_layer[il].data(),
                                   n_embd,
                                   tol_max_for_depth(il),
                                   tol_rmse_for_depth(il),
                                   sub_err)) {
                ok = false; break;
            }
        }

        if (!ok) { phi3_fused_ctx_free(cx); return false; }

        // ---- Final RMSNorm (oracle) ----
        {
            std::vector<float> w_final(n_embd);
            const ggml_tensor * onm = W.output_norm;
            if (onm->type == GGML_TYPE_F32) {
                std::memcpy(w_final.data(), onm->data, (size_t) n_embd * sizeof(float));
            } else if (onm->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) onm->data, w_final.data(), n_embd);
            } else if (!dequant_row_to_f32(onm, 0, w_final.data(), sub_err)) {
                phi3_fused_ctx_free(cx);
                return false;
            }

            // Build a tiny ggml graph for the final norm (independent path).
            const size_t small_mem = 4 * 1024 * 1024;
            std::vector<uint8_t> arena_norm(small_mem);
            ggml_init_params ipn{ small_mem, arena_norm.data(), false };
            ggml_context * gctx = ggml_init(ipn);
            ggml_tensor * tx = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, n_embd);
            std::memcpy(tx->data, x_oracle.data(), n_embd * sizeof(float));
            ggml_tensor * tw = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, n_embd);
            std::memcpy(tw->data, w_final.data(), n_embd * sizeof(float));
            ggml_tensor * n = ggml_rms_norm(gctx, tx, cx.eps);
            n = ggml_mul(gctx, n, tw);
            ggml_cgraph * gf = ggml_new_graph(gctx);
            ggml_build_forward_expand(gf, n);
            ggml_graph_compute_with_ctx(gctx, gf, 1);
            std::memcpy(oracle_x_per_layer[n_layer].data(), n->data, n_embd * sizeof(float));
            x_oracle.assign((float *) n->data, (float *) n->data + n_embd);
            ggml_free(gctx);
        }

        if (!check_close_stats("post_final_norm",
                               hand_x_per_layer[n_layer].data(),
                               oracle_x_per_layer[n_layer].data(),
                               n_embd,
                               5e-1f, 6e-2f,
                               sub_err)) {
            phi3_fused_ctx_free(cx);
            return false;
        }

        // ---- Oracle lm_head via ggml_mul_mat over fully-dequantized F32 matrix ----
        std::vector<float> oracle_logits(n_vocab);
        {
            const ggml_tensor * lm_head = W.output;
            const size_t lm_bytes_f32 = (size_t) n_vocab * n_embd * sizeof(float);
            const size_t arena_lm = lm_bytes_f32 + (size_t) 32 * 1024 * 1024; // headroom for inputs+output
            std::vector<uint8_t> arena_lm_buf(arena_lm);
            ggml_init_params iplm{ arena_lm, arena_lm_buf.data(), false };
            ggml_context * gctx = ggml_init(iplm);
            ggml_tensor * t_lm = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, n_vocab);
            std::string deq_err;
            if (!dequant_2d_to_f32(lm_head, (float *) t_lm->data, (size_t) n_vocab * n_embd, deq_err)) {
                sub_err = "phi3 full self-test: lm_head dequant failed: " + deq_err;
                ggml_free(gctx);
                phi3_fused_ctx_free(cx);
                return false;
            }
            ggml_tensor * t_x  = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, n_embd);
            std::memcpy(t_x->data, x_oracle.data(), n_embd * sizeof(float));
            ggml_tensor * t_lg = ggml_mul_mat(gctx, t_lm, t_x);
            ggml_mul_mat_set_prec(t_lg, GGML_PREC_F32);
            ggml_cgraph * gf = ggml_new_graph(gctx);
            ggml_build_forward_expand(gf, t_lg);
            ggml_graph_compute_with_ctx(gctx, gf, 1);
            std::memcpy(oracle_logits.data(), t_lg->data, n_vocab * sizeof(float));
            ggml_free(gctx);
        }

        if (!check_close_stats("logits",
                               hand_logits.data(),
                               oracle_logits.data(),
                               n_vocab,
                               2.0f, 3e-1f,
                               sub_err)) {
            phi3_fused_ctx_free(cx);
            return false;
        }

        const int am_h = argmax_f32(hand_logits.data(),   n_vocab);
        const int am_o = argmax_f32(oracle_logits.data(), n_vocab);
        auto top5_h = top_k_indices(hand_logits.data(),   n_vocab, 5);
        auto top5_o = top_k_indices(oracle_logits.data(), n_vocab, 5);
        int overlap = 0;
        for (int i : top5_h) for (int j : top5_o) if (i == j) { ++overlap; break; }
        fprintf(stderr,
                "phi3 full self-test: argmax hand=%d oracle=%d  match=%d  top5_overlap=%d/5\n",
                am_h, am_o, (int)(am_h == am_o), overlap);

        phi3_fused_ctx_free(cx);
        return true;
    };

    std::string sub_err;
    if (!run_sub_test("sub-test 1 (pos=0, no prior KV)", /*pos=*/0, /*n_prior=*/0, sub_err)) {
        error = sub_err;
        return false;
    }
    if (!run_sub_test("sub-test 2 (pos=4, 4 prior K/V positions)", /*pos=*/4, /*n_prior=*/4, sub_err)) {
        error = sub_err;
        return false;
    }

    fprintf(stderr, "\nphi3 full self-test: PASS (2 sub-tests; %d layers + final_norm + logits each)\n", n_layer);
    return true;
}

// ===========================================================================
// A2.4b ? F32-everywhere decode (debug / spot-check).
// ===========================================================================

bool phi3_run_f32_decode(
        const llama_model            * model,
        const std::vector<llama_token> & prompt_tokens,
        int                            n_gen,
        std::vector<llama_token>     & out_generated,
        std::string                  & error) {
    error.clear();
    if (model == nullptr)         { error = "phi3_run_f32_decode: model is null"; return false; }
    if (prompt_tokens.empty())    { error = "phi3_run_f32_decode: prompt_tokens is empty"; return false; }
    if (n_gen <= 0)               { error = "phi3_run_f32_decode: n_gen must be > 0"; return false; }

    Phi3Weights W;
    if (!phi3_weights_resolve(model, W, error)) return false;

    if (W.n_head != W.n_head_kv) {
        std::ostringstream oss;
        oss << "phi3_run_f32_decode: GQA models not yet supported (n_head="
            << W.n_head << " n_head_kv=" << W.n_head_kv << "). "
            << "F32 hand path was built and tested for n_head==n_head_kv only.";
        error = oss.str();
        return false;
    }

    const int n_vocab  = W.n_vocab;
    const int n_prompt = (int) prompt_tokens.size();
    const int ctx_max  = n_prompt + n_gen + 2;

    // Validate prompt tokens in range.
    for (int p = 0; p < n_prompt; ++p) {
        if (prompt_tokens[p] < 0 || prompt_tokens[p] >= n_vocab) {
            std::ostringstream oss;
            oss << "phi3_run_f32_decode: prompt token " << p << "=" << prompt_tokens[p]
                << " out of [0," << n_vocab << ")";
            error = oss.str();
            return false;
        }
    }

    Phi3FusedCtx cx;
    if (!phi3_fused_ctx_init(cx, W, /*pool=*/nullptr, model, /*lctx=*/nullptr,
                             ctx_max, /*f32_debug=*/true, error)) {
        return false;
    }

    fprintf(stderr,
            "phi3 f32-decode: starting (n_prompt=%d, n_gen=%d, ctx_max=%d, est ~%.0fs total)\n",
            n_prompt, n_gen, ctx_max, (double)(n_prompt + n_gen) * 6.5);

    std::vector<float> logits((size_t) n_vocab);

    auto check_finite_and_argmax = [&](int * out_idx) -> bool {
        int best_idx = 0;
        float best   = logits[0];
        for (int v = 0; v < n_vocab; ++v) {
            const float x = logits[v];
            if (!std::isfinite(x)) {
                std::ostringstream oss;
                oss << "phi3_run_f32_decode: non-finite logit at v=" << v << " value=" << x;
                error = oss.str();
                return false;
            }
            if (x > best) { best = x; best_idx = v; }
        }
        *out_idx = best_idx;
        return true;
    };

    // ---- Prefill positions [0..n_prompt-1] ----
    const auto t_prefill_start = std::chrono::steady_clock::now();
    for (int p = 0; p < n_prompt; ++p) {
        if (!phi3_full_forward_f32(cx, prompt_tokens[p], p, logits.data(),
                                   nullptr, nullptr, error)) {
            phi3_fused_ctx_free(cx);
            return false;
        }
        // Sanity check on every prefill forward (cheap insurance against NaN).
        for (int v = 0; v < 8; ++v) {
            if (!std::isfinite(logits[v])) {
                std::ostringstream oss;
                oss << "phi3_run_f32_decode: non-finite logit during prefill pos=" << p
                    << " v=" << v << " value=" << logits[v];
                error = oss.str();
                phi3_fused_ctx_free(cx);
                return false;
            }
        }
    }
    const double prefill_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_prefill_start).count();
    fprintf(stderr, "phi3 f32-decode: prefill %d tokens in %.2f s (%.2f s/tok)\n",
            n_prompt, prefill_ms / 1000.0, prefill_ms / 1000.0 / std::max(1, n_prompt));

    // ---- Generation ----
    out_generated.reserve(out_generated.size() + n_gen);
    const auto t_gen_start = std::chrono::steady_clock::now();

    int next_tok = 0;
    if (!check_finite_and_argmax(&next_tok)) {
        phi3_fused_ctx_free(cx);
        return false;
    }
    out_generated.push_back(next_tok);

    for (int i = 1; i < n_gen; ++i) {
        const int pos = n_prompt + i - 1;  // position of the token we just generated
        if (!phi3_full_forward_f32(cx, next_tok, pos, logits.data(),
                                   nullptr, nullptr, error)) {
            phi3_fused_ctx_free(cx);
            return false;
        }
        if (!check_finite_and_argmax(&next_tok)) {
            phi3_fused_ctx_free(cx);
            return false;
        }
        out_generated.push_back(next_tok);
    }
    const double gen_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_gen_start).count();
    fprintf(stderr, "phi3 f32-decode: generated %d tokens in %.2f s (%.2f s/tok)\n",
            n_gen, gen_ms / 1000.0, gen_ms / 1000.0 / std::max(1, n_gen));

    phi3_fused_ctx_free(cx);
    return true;
}

// ===========================================================================
// A2.5a.0 - Q-quant matmul primitive + micro-oracle self-test.
// ===========================================================================

// Hand q-quant matmul: y[r] = vec_dot(W[r,:], x) for r in [0..M),
//                     where W is a quantized 2D tensor of shape {K, M}
//                     (K elements per row, M rows) and x is a F32 vector
//                     of length K. The src is q-quantized ONCE into
//                     q_scratch using w_traits->vec_dot_type's from_float.
//
// Aggressive validation: w not null, w->data not null, w->ne[0]==K, ne[1]==M,
// nb[0]==type_size, nb[1]==row_size; w_traits/q_traits non-null with
// vec_dot/from_float; K compatible with vec_dot_type's block size.
//
// q_scratch is resized inside; caller may reuse across calls.
//
// Returns false with a populated error on any failure.
namespace {

bool q_matmul_row(
        const ggml_tensor   * w,
        const float         * x_f32,
        float               * y_out,
        int                   M,
        int                   K,
        std::vector<uint8_t> & q_scratch,
        std::string         & error) {
    if (w == nullptr)           { error = "q_matmul_row: w is null"; return false; }
    if (w->data == nullptr)     { error = "q_matmul_row: w->data is null (weight not resident)"; return false; }
    if (x_f32 == nullptr)       { error = "q_matmul_row: x_f32 is null"; return false; }
    if (y_out == nullptr)       { error = "q_matmul_row: y_out is null"; return false; }
    if (M <= 0 || K <= 0)       { error = "q_matmul_row: M or K <= 0"; return false; }

    if ((int) w->ne[0] != K || (int) w->ne[1] != M) {
        std::ostringstream oss;
        oss << "q_matmul_row: shape mismatch: w->ne=(" << w->ne[0] << "," << w->ne[1]
            << ") expected (K=" << K << ", M=" << M << ")";
        error = oss.str();
        return false;
    }
    if ((size_t) w->nb[0] != ggml_type_size(w->type)) {
        std::ostringstream oss;
        oss << "q_matmul_row: w->nb[0]=" << w->nb[0]
            << " != ggml_type_size(" << ggml_type_name(w->type) << ")=" << ggml_type_size(w->type);
        error = oss.str();
        return false;
    }
    const size_t expected_row_bytes = ggml_row_size(w->type, K);
    if ((size_t) w->nb[1] != expected_row_bytes) {
        std::ostringstream oss;
        oss << "q_matmul_row: w->nb[1]=" << w->nb[1]
            << " != ggml_row_size(" << ggml_type_name(w->type) << ", K=" << K << ")="
            << expected_row_bytes;
        error = oss.str();
        return false;
    }

    const auto * w_traits = ggml_get_type_traits_cpu(w->type);
    if (!w_traits || !w_traits->vec_dot) {
        std::ostringstream oss;
        oss << "q_matmul_row: no CPU vec_dot for type " << ggml_type_name(w->type);
        error = oss.str();
        return false;
    }
    const ggml_type q_type   = w_traits->vec_dot_type;
    const auto *    q_traits = ggml_get_type_traits_cpu(q_type);
    if (!q_traits || !q_traits->from_float) {
        std::ostringstream oss;
        oss << "q_matmul_row: vec_dot_type=" << ggml_type_name(q_type)
            << " has no from_float kernel";
        error = oss.str();
        return false;
    }
    // K must be a multiple of vec_dot_type's block size (typical: 256 for Q8_K).
    const int64_t q_block = (int64_t) ggml_blck_size(q_type);
    if (q_block > 1 && (K % q_block) != 0) {
        std::ostringstream oss;
        oss << "q_matmul_row: K=" << K << " not a multiple of block size "
            << q_block << " for vec_dot_type=" << ggml_type_name(q_type);
        error = oss.str();
        return false;
    }

    const size_t q_bytes = ggml_row_size(q_type, K);
    if (q_scratch.size() < q_bytes) q_scratch.assign(q_bytes, 0);

    q_traits->from_float(x_f32, q_scratch.data(), K);

    const uint8_t * w_base   = (const uint8_t *) w->data;
    const size_t    row_b    = expected_row_bytes;
    const void *    q_src    = q_scratch.data();
    for (int r = 0; r < M; ++r) {
        float s = 0.0f;
        w_traits->vec_dot(K, &s, 0, w_base + (size_t) r * row_b, 0, q_src, 0, 1);
        y_out[r] = s;
    }
    return true;
}

// Build a single mul_mat oracle: y_oracle = ggml_mul_mat(w, x_f32) computed
// via ggml-cpu in the same process. Comparison vs q_matmul_row should be
// byte-identical because both call the same vec_dot kernel after the same
// from_float quantization.
bool qmatmul_oracle_one(
        const ggml_tensor * w,
        const float       * x_f32,
        int                 M,
        int                 K,
        std::vector<float> & y_oracle,
        std::string       & error) {
    y_oracle.assign((size_t) M, 0.0f);

    const size_t weight_bytes = ggml_nbytes(w);
    const size_t arena_bytes  = weight_bytes
                              + (size_t) K * sizeof(float)
                              + (size_t) M * sizeof(float)
                              + 8ull * 1024 * 1024;  // graph + overhead
    std::vector<uint8_t> arena(arena_bytes);
    ggml_init_params ip{ arena_bytes, arena.data(), false };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { error = "qmatmul_oracle_one: ggml_init failed"; return false; }

    ggml_tensor * tw = ggml_new_tensor_2d(gctx, w->type, K, M);
    std::memcpy(tw->data, w->data, weight_bytes);
    ggml_tensor * tx = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, K);
    std::memcpy(tx->data, x_f32, (size_t) K * sizeof(float));
    ggml_tensor * ty = ggml_mul_mat(gctx, tw, tx);
    ggml_mul_mat_set_prec(ty, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, ty);
    ggml_graph_compute_with_ctx(gctx, gf, 1);
    std::memcpy(y_oracle.data(), ty->data, (size_t) M * sizeof(float));
    ggml_free(gctx);
    return true;
}

// Pool-dispatched q-quant matmul. Same shape as q_matmul_row but routes the
// per-row vec_dot inner loop through the (optional) Phi3MatmulPool. If pool
// is null/uninitialized/single-threaded, falls back to the same serial path
// as q_matmul_row (so behavior is byte-identical regardless of threading).
//
// The src is quantized ONCE on the calling thread before the dispatch.
bool q_matmul_pool(
        const ggml_tensor    * w,
        const float          * x_f32,
        float                * y_out,
        int                    M,
        int                    K,
        std::vector<uint8_t> & q_scratch,
        Phi3MatmulPool       * pool,
        std::string          & error) {
    if (w == nullptr)        { error = "q_matmul_pool: w is null"; return false; }
    if (w->data == nullptr)  { error = "q_matmul_pool: w->data is null"; return false; }
    if (x_f32 == nullptr)    { error = "q_matmul_pool: x_f32 is null"; return false; }
    if (y_out == nullptr)    { error = "q_matmul_pool: y_out is null"; return false; }
    if (M <= 0 || K <= 0)    { error = "q_matmul_pool: M or K <= 0"; return false; }

    if ((int) w->ne[0] != K || (int) w->ne[1] != M) {
        std::ostringstream oss;
        oss << "q_matmul_pool: shape mismatch: w->ne=(" << w->ne[0] << "," << w->ne[1]
            << ") expected (K=" << K << ", M=" << M << ")";
        error = oss.str();
        return false;
    }
    const size_t row_bytes = ggml_row_size(w->type, K);
    if ((size_t) w->nb[1] != row_bytes) {
        std::ostringstream oss;
        oss << "q_matmul_pool: w->nb[1]=" << w->nb[1] << " != ggml_row_size=" << row_bytes;
        error = oss.str();
        return false;
    }

    const auto * w_traits = ggml_get_type_traits_cpu(w->type);
    if (!w_traits || !w_traits->vec_dot) {
        std::ostringstream oss;
        oss << "q_matmul_pool: no CPU vec_dot for type " << ggml_type_name(w->type);
        error = oss.str();
        return false;
    }
    const ggml_type q_type   = w_traits->vec_dot_type;
    const auto *    q_traits = ggml_get_type_traits_cpu(q_type);
    if (!q_traits || !q_traits->from_float) {
        std::ostringstream oss;
        oss << "q_matmul_pool: vec_dot_type=" << ggml_type_name(q_type) << " has no from_float";
        error = oss.str();
        return false;
    }
    const int64_t q_block = (int64_t) ggml_blck_size(q_type);
    if (q_block > 1 && (K % q_block) != 0) {
        std::ostringstream oss;
        oss << "q_matmul_pool: K=" << K << " not multiple of block size " << q_block;
        error = oss.str();
        return false;
    }

    const size_t q_bytes = ggml_row_size(q_type, K);
    if (q_scratch.size() < q_bytes) q_scratch.assign(q_bytes, 0);

    q_traits->from_float(x_f32, q_scratch.data(), K);

    Phi3MatmulJob job;
    job.w_traits    = w_traits;
    job.w_base      = (const uint8_t *) w->data;
    job.w_row_bytes = row_bytes;
    job.src_q       = q_scratch.data();
    job.dst         = y_out;
    job.K           = K;
    job.N_total     = M;

    phi3_matmul_pool_run(pool, job);
    return true;
}

} // namespace


bool phi3_qmatmul_self_test(const llama_model * model, std::string & error) {
    error.clear();
    if (model == nullptr) { error = "phi3_qmatmul_self_test: model is null"; return false; }

    Phi3Weights W;
    if (!phi3_weights_resolve(model, W, error)) return false;
    if (W.n_layer < 1) { error = "phi3_qmatmul_self_test: no layers"; return false; }
    const Phi3LayerWeights & L = W.layers[0];

    struct WtCase {
        const char        * name;
        const ggml_tensor * w;
        int                 M;
        int                 K;
    };
    const WtCase cases[] = {
        { "wqkv",     L.wqkv,     W.n_qkv,     W.n_embd },
        { "wo",       L.wo,       W.n_embd,    W.n_embd },
        { "ffn_up",   L.ffn_up,   2 * W.n_ff,  W.n_embd },
        { "ffn_down", L.ffn_down, W.n_embd,    W.n_ff   },
    };

    // Deterministic seeded input vectors (one per case width).
    auto make_x = [&](int K, uint32_t seed) {
        std::vector<float> x((size_t) K);
        uint32_t s = seed;
        for (int k = 0; k < K; ++k) {
            s = s * 1664525u + 1013904223u;
            const float v = ((float)(s >> 8) / (float)(1u << 24)) - 0.5f; // ~U[-0.5, 0.5)
            x[k] = v * 0.5f;
        }
        return x;
    };

    std::vector<uint8_t> q_scratch;
    std::vector<float>   hand;
    std::vector<float>   oracle;
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); ++c) {
        const WtCase & cs = cases[c];
        if (cs.w == nullptr) {
            std::ostringstream oss;
            oss << "phi3_qmatmul_self_test: weight '" << cs.name << "' is null";
            error = oss.str();
            return false;
        }

        const std::vector<float> x = make_x(cs.K, /*seed=*/0x9E3779B9u + (uint32_t) c);

        hand.assign((size_t) cs.M, 0.0f);
        if (!q_matmul_row(cs.w, x.data(), hand.data(), cs.M, cs.K, q_scratch, error)) {
            std::string sub = error;
            std::ostringstream oss;
            oss << "phi3_qmatmul_self_test: q_matmul_row '" << cs.name << "' failed: " << sub;
            error = oss.str();
            return false;
        }

        std::string sub_err;
        if (!qmatmul_oracle_one(cs.w, x.data(), cs.M, cs.K, oracle, sub_err)) {
            std::ostringstream oss;
            oss << "phi3_qmatmul_self_test: oracle '" << cs.name << "' failed: " << sub_err;
            error = oss.str();
            return false;
        }

        // Compare. Identical kernels => exact match expected. Allow ULP wiggle
        // (1 ulp ~= 1e-7 * |val|) in case ggml's mul_mat takes a different
        // internal accumulator path (e.g. AVX-512 vs scalar) but compute the
        // numerically same dot.
        int worst_i = -1;
        float worst_diff = 0.0f, worst_a = 0.0f, worst_b = 0.0f;
        double sumsq = 0.0;
        int n_exact = 0;
        for (int j = 0; j < cs.M; ++j) {
            const float a = hand[j], b = oracle[j];
            const float d = std::fabs(a - b);
            if (d == 0.0f) ++n_exact;
            sumsq += (double) d * (double) d;
            if (d > worst_diff) { worst_diff = d; worst_i = j; worst_a = a; worst_b = b; }
        }
        const float rmse = (float) std::sqrt(sumsq / (double) cs.M);
        const float tol_abs = 1e-3f;  // generous for K-quant arithmetic
        const float tol_rmse = 1e-4f;
        const bool pass = worst_diff <= tol_abs && rmse <= tol_rmse;
        fprintf(stderr,
                "phi3 qmatmul: %-9s  M=%-6d K=%-5d  type=%s  vec_dot_type=%s"
                "  max_abs=%.4g rmse=%.4g  exact=%d/%d  %s\n",
                cs.name, cs.M, cs.K, ggml_type_name(cs.w->type),
                ggml_type_name(ggml_get_type_traits_cpu(cs.w->type)->vec_dot_type),
                worst_diff, rmse, n_exact, cs.M,
                pass ? "PASS" : "FAIL");
        if (!pass) {
            std::ostringstream oss;
            oss << "phi3_qmatmul_self_test: '" << cs.name
                << "' diverged at j=" << worst_i
                << " hand=" << worst_a << " oracle=" << worst_b
                << " |diff|=" << worst_diff
                << "  tol_abs=" << tol_abs << " tol_rmse=" << tol_rmse;
            error = oss.str();
            return false;
        }
    }
    fprintf(stderr, "phi3 qmatmul self-test: PASS (4 weight classes on layer 0)\n");
    return true;
}


// ===========================================================================
// A2.5a.1 - Q-quant decode (sibling of phi3_run_f32_decode).
// ===========================================================================

// Sibling of phi3_layer_forward_f32 using the ORIGINAL quantized weights via
// q_matmul_row. Does NOT require cx.scratch.f32_debug; reuses cx.scratch.hq_buf
// and ffq_buf (already provisioned for q-scratch).
//
// Attention path remains F32 (Q*K^T in F32 against F16-rounded K) to match
// A2.4b. This is a known minor numerical divergence from baseline's
// ggml_vec_dot_f16, but A2.4 showed argmax still matches.
bool phi3_layer_forward_qquant(
        Phi3FusedCtx       & cx,
        int                  il,
        int                  pos,
        float              * x_inout,
        std::string        & error) {
    error.clear();
    if (cx.w == nullptr) { error = "phi3_layer_forward_qquant: cx.w is null"; return false; }
    if (x_inout == nullptr) { error = "phi3_layer_forward_qquant: x_inout is null"; return false; }

    const Phi3Weights & W = *cx.w;
    const Phi3LayerWeights & L = W.layers[il];
    const int n_embd     = W.n_embd;
    const int n_head     = W.n_head;
    const int n_head_kv  = W.n_head_kv;
    const int head_dim   = W.n_embd_head;
    const int n_embd_kv  = head_dim * n_head_kv;
    const int n_embd_q   = head_dim * n_head;
    const int n_qkv      = W.n_qkv;
    const int n_ff       = W.n_ff;

    if (pos < 0 || pos >= cx.kv.ctx_max) {
        std::ostringstream oss; oss << "phi3_layer_forward_qquant: pos=" << pos << " out of range";
        error = oss.str();
        return false;
    }

    // Dequant attn_norm and ffn_norm into small per-call vectors (n_embd each).
    // These are 1-D F16 or F32 tensors and cheap to handle.
    std::vector<float> attn_norm_w(n_embd);
    std::vector<float> ffn_norm_w(n_embd);
    {
        const ggml_tensor * an = L.attn_norm;
        if (an->type == GGML_TYPE_F32) {
            std::memcpy(attn_norm_w.data(), an->data, n_embd * sizeof(float));
        } else if (an->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) an->data, attn_norm_w.data(), n_embd);
        } else {
            if (!dequant_row_to_f32(an, 0, attn_norm_w.data(), error)) return false;
        }
        const ggml_tensor * fn = L.ffn_norm;
        if (fn->type == GGML_TYPE_F32) {
            std::memcpy(ffn_norm_w.data(), fn->data, n_embd * sizeof(float));
        } else if (fn->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) fn->data, ffn_norm_w.data(), n_embd);
        } else {
            if (!dequant_row_to_f32(fn, 0, ffn_norm_w.data(), error)) return false;
        }
    }

    // Local scratch (avoid aliasing with cx.scratch buffers used by q_matmul_row).
    std::vector<float> norm1(n_embd);
    std::vector<float> qkv(n_qkv);
    std::vector<float> attn_ctx(n_embd, 0.0f);
    std::vector<float> attn_out_buf(n_embd);
    std::vector<float> x_after_res1(n_embd);
    std::vector<float> norm2(n_embd);
    std::vector<float> upgate(2 * n_ff);
    std::vector<float> ff(n_ff);
    std::vector<float> ffn_out_buf(n_embd);

    // --- 1. Pre-attn RMSNorm ---
    phi3_kernel_rmsnorm_mul_f32(norm1.data(), x_inout, attn_norm_w.data(), n_embd, cx.eps);

    // --- 2. wqkv (q-quant, pool-dispatched) ---
    if (!q_matmul_pool(L.wqkv, norm1.data(), qkv.data(), n_qkv, n_embd,
                       cx.scratch.hq_buf, cx.matmul_pool, error)) {
        std::string s = error; error = "phi3_layer_forward_qquant: wqkv: " + s; return false;
    }

    float * q = qkv.data();
    float * k = qkv.data() + n_embd_q;
    float * v = qkv.data() + n_embd_q + n_embd_kv;

    // --- 3. RoPE on Q, K ---
    for (int h = 0; h < n_head; ++h) {
        phi3_kernel_rope_neox_f32(q + h * head_dim, q + h * head_dim, /*factors=*/nullptr,
                                  cx.rope.n_rot, head_dim, pos, cx.rope.freq_base);
    }
    for (int h = 0; h < n_head_kv; ++h) {
        phi3_kernel_rope_neox_f32(k + h * head_dim, k + h * head_dim, /*factors=*/nullptr,
                                  cx.rope.n_rot, head_dim, pos, cx.rope.freq_base);
    }

    // --- 4. Write K, V to cache at position pos (F16) ---
    {
        ggml_fp16_t * K_pos = cx.kv.K[il] + (size_t) pos * (size_t) n_embd_kv;
        ggml_fp16_t * V_pos = cx.kv.V[il] + (size_t) pos * (size_t) n_embd_kv;
        ggml_fp32_to_fp16_row(k, K_pos, n_embd_kv);
        ggml_fp32_to_fp16_row(v, V_pos, n_embd_kv);
    }
    const int new_len = pos + 1;
    if (cx.kv.current_len < new_len) cx.kv.current_len = new_len;

    // --- 5. Attention (F32 dot against F16 KV, matches A2.4b) ---
    const float scale_q = 1.0f / std::sqrt((float) head_dim);
    std::vector<float> scores((size_t) new_len);
    for (int h = 0; h < n_head; ++h) {
        const float * q_h = q + h * head_dim;
        float max_s = -INFINITY;
        for (int p = 0; p < new_len; ++p) {
            const ggml_fp16_t * k_p = cx.kv.K[il] + ((size_t) p * n_head_kv + h) * head_dim;
            double acc = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                acc += (double) q_h[d] * (double) ggml_fp16_to_fp32(k_p[d]);
            }
            const float s = (float) (acc * (double) scale_q);
            scores[p] = s;
            if (s > max_s) max_s = s;
        }
        double sum_exp = 0.0;
        for (int p = 0; p < new_len; ++p) {
            scores[p] = std::exp(scores[p] - max_s);
            sum_exp += (double) scores[p];
        }
        const float inv_sum = (float) (1.0 / sum_exp);
        float * ctx_h = attn_ctx.data() + h * head_dim;
        for (int d = 0; d < head_dim; ++d) ctx_h[d] = 0.0f;
        for (int p = 0; p < new_len; ++p) {
            const float w_p = scores[p] * inv_sum;
            const ggml_fp16_t * v_p = cx.kv.V[il] + ((size_t) p * n_head_kv + h) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                ctx_h[d] += w_p * ggml_fp16_to_fp32(v_p[d]);
            }
        }
    }

    // --- 6. wo (q-quant, pool-dispatched) ---
    if (!q_matmul_pool(L.wo, attn_ctx.data(), attn_out_buf.data(), n_embd, n_embd,
                       cx.scratch.hq_buf, cx.matmul_pool, error)) {
        std::string s = error; error = "phi3_layer_forward_qquant: wo: " + s; return false;
    }

    // --- 7. Residual 1 ---
    for (int i = 0; i < n_embd; ++i) x_after_res1[i] = x_inout[i] + attn_out_buf[i];

    // --- 8. FFN RMSNorm ---
    phi3_kernel_rmsnorm_mul_f32(norm2.data(), x_after_res1.data(), ffn_norm_w.data(), n_embd, cx.eps);

    // --- 9. ffn_up (q-quant, gate||up fused, pool-dispatched) ---
    if (!q_matmul_pool(L.ffn_up, norm2.data(), upgate.data(), 2 * n_ff, n_embd,
                       cx.scratch.hq_buf, cx.matmul_pool, error)) {
        std::string s = error; error = "phi3_layer_forward_qquant: ffn_up: " + s; return false;
    }

    // --- 10. SwiGLU ---
    for (int i = 0; i < n_ff; ++i) {
        const float g = upgate[i];
        const float u = upgate[n_ff + i];
        const float silu = g / (1.0f + std::exp(-g));
        ff[i] = silu * u;
    }

    // --- 11. ffn_down (q-quant, pool-dispatched) ---
    if (!q_matmul_pool(L.ffn_down, ff.data(), ffn_out_buf.data(), n_embd, n_ff,
                       cx.scratch.ffq_buf, cx.matmul_pool, error)) {
        std::string s = error; error = "phi3_layer_forward_qquant: ffn_down: " + s; return false;
    }

    // --- 12. Residual 2 ---
    for (int i = 0; i < n_embd; ++i) x_inout[i] = x_after_res1[i] + ffn_out_buf[i];

    return true;
}


// Full per-token forward in q-quant mode. Produces the post-final-norm hidden
// state into out_hidden (n_embd). lm_head argmax is performed by the caller
// using phi3_fused_lmhead_argmax against this hidden state.
bool phi3_full_forward_qquant(
        Phi3FusedCtx       & cx,
        int                  token_id,
        int                  pos,
        float              * out_hidden,
        std::string        & error) {
    error.clear();
    if (cx.w == nullptr)    { error = "phi3_full_forward_qquant: cx.w is null"; return false; }
    if (out_hidden == nullptr) { error = "phi3_full_forward_qquant: out_hidden is null"; return false; }

    const Phi3Weights & W = *cx.w;
    const int n_embd  = W.n_embd;
    const int n_layer = W.n_layer;
    const int n_vocab = W.n_vocab;

    if (token_id < 0 || token_id >= n_vocab) {
        std::ostringstream oss;
        oss << "phi3_full_forward_qquant: token_id=" << token_id << " out of [0," << n_vocab << ")";
        error = oss.str();
        return false;
    }

    // 1. tok_embd lookup -> F32
    std::vector<float> x(n_embd);
    if (!dequant_row_to_f32(W.tok_embd, token_id, x.data(), error)) return false;

    // 2. 32 layers in q-quant
    for (int il = 0; il < n_layer; ++il) {
        if (!phi3_layer_forward_qquant(cx, il, pos, x.data(), error)) return false;
    }

    // 3. Final RMSNorm
    {
        std::vector<float> w_final(n_embd);
        const ggml_tensor * onm = W.output_norm;
        if (onm->type == GGML_TYPE_F32) {
            std::memcpy(w_final.data(), onm->data, n_embd * sizeof(float));
        } else if (onm->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) onm->data, w_final.data(), n_embd);
        } else {
            if (!dequant_row_to_f32(onm, 0, w_final.data(), error)) return false;
        }
        std::vector<float> tmp(n_embd);
        phi3_kernel_rmsnorm_mul_f32(tmp.data(), x.data(), w_final.data(), n_embd, cx.eps);
        std::memcpy(out_hidden, tmp.data(), n_embd * sizeof(float));
    }
    return true;
}


bool phi3_run_qquant_decode(
        const llama_model            * model,
        const std::vector<llama_token> & prompt_tokens,
        int                            n_gen,
        int                            n_threads,
        std::vector<llama_token>     & out_generated,
        std::string                  & error) {
    error.clear();
    if (model == nullptr)        { error = "phi3_run_qquant_decode: model is null"; return false; }
    if (prompt_tokens.empty())   { error = "phi3_run_qquant_decode: prompt_tokens is empty"; return false; }
    if (n_gen <= 0)              { error = "phi3_run_qquant_decode: n_gen must be > 0"; return false; }
    if (n_threads <= 0)          { n_threads = 1; }

    Phi3Weights W;
    if (!phi3_weights_resolve(model, W, error)) return false;

    if (W.n_head != W.n_head_kv) {
        std::ostringstream oss;
        oss << "phi3_run_qquant_decode: GQA not supported (n_head=" << W.n_head
            << " n_head_kv=" << W.n_head_kv << ")";
        error = oss.str();
        return false;
    }

    const int n_embd   = W.n_embd;
    const int n_vocab  = W.n_vocab;
    const int n_prompt = (int) prompt_tokens.size();
    const int ctx_max  = n_prompt + n_gen + 2;

    for (int p = 0; p < n_prompt; ++p) {
        if (prompt_tokens[p] < 0 || prompt_tokens[p] >= n_vocab) {
            std::ostringstream oss;
            oss << "phi3_run_qquant_decode: prompt token " << p << "=" << prompt_tokens[p]
                << " out of [0," << n_vocab << ")";
            error = oss.str();
            return false;
        }
    }

    // Allocate matmul pool (n_threads > 1) and wire it through the ctx.
    // n_threads <= 1 leaves the pool uninitialized; phi3_matmul_pool_run
    // falls back to serial on the calling thread (byte-identical).
    Phi3MatmulPool pool;
    if (n_threads > 1) {
        std::string perr;
        if (!phi3_matmul_pool_init(pool, n_threads, perr)) {
            // Non-fatal: fall back to serial.
            fprintf(stderr, "phi3 qquant-decode: matmul_pool_init failed (%s); falling back to serial\n",
                    perr.c_str());
        }
    }

    // f32_debug=false  =>  no ~150MB F32 weight mirrors allocated
    Phi3FusedCtx cx;
    if (!phi3_fused_ctx_init(cx, W, pool.initialized ? &pool : nullptr,
                             model, /*lctx=*/nullptr,
                             ctx_max, /*f32_debug=*/false, error)) {
        phi3_matmul_pool_free(pool);
        return false;
    }

    // lm_head: use the existing fused q-quant argmax helper.
    Phi3FusedLmHead lm;
    if (!phi3_fused_lmhead_init(model, lm, error)) {
        phi3_fused_ctx_free(cx);
        phi3_matmul_pool_free(pool);
        return false;
    }

    fprintf(stderr,
            "phi3 qquant-decode: starting (n_prompt=%d, n_gen=%d, ctx_max=%d, n_threads=%d, pool=%s)\n",
            n_prompt, n_gen, ctx_max, n_threads, pool.initialized ? "on" : "off");

    std::vector<float> hidden((size_t) n_embd);

    auto step = [&](int tok, int pos, llama_token * out_tok_or_null) -> bool {
        if (!phi3_full_forward_qquant(cx, tok, pos, hidden.data(), error)) return false;
        // Finite check on hidden state.
        for (int i = 0; i < n_embd; ++i) {
            if (!std::isfinite(hidden[i])) {
                std::ostringstream oss;
                oss << "phi3 qquant-decode: non-finite hidden at pos=" << pos << " dim=" << i << " val=" << hidden[i];
                error = oss.str();
                return false;
            }
        }
        if (out_tok_or_null) {
            llama_token best = 0;
            if (!phi3_fused_lmhead_argmax(lm, /*pool=*/nullptr, hidden.data(),
                                          /*n_threads=*/1, best, error)) return false;
            *out_tok_or_null = best;
        }
        return true;
    };

    // ---- Prefill ----
    const auto t_prefill_start = std::chrono::steady_clock::now();
    llama_token next_tok = 0;
    for (int p = 0; p < n_prompt; ++p) {
        const bool is_last = (p == n_prompt - 1);
        if (!step(prompt_tokens[p], p, is_last ? &next_tok : nullptr)) {
            phi3_fused_ctx_free(cx);
            phi3_matmul_pool_free(pool);
            return false;
        }
    }
    const double prefill_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_prefill_start).count();
    fprintf(stderr, "phi3 qquant-decode: prefill %d tokens in %.2f s (%.3f s/tok)\n",
            n_prompt, prefill_ms / 1000.0, prefill_ms / 1000.0 / std::max(1, n_prompt));

    // ---- Generation ----
    out_generated.reserve(out_generated.size() + n_gen);
    const auto t_gen_start = std::chrono::steady_clock::now();

    out_generated.push_back(next_tok);
    for (int i = 1; i < n_gen; ++i) {
        const int pos = n_prompt + i - 1;
        if (!step(next_tok, pos, &next_tok)) {
            phi3_fused_ctx_free(cx);
            phi3_matmul_pool_free(pool);
            return false;
        }
        out_generated.push_back(next_tok);
    }
    const double gen_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_gen_start).count();
    fprintf(stderr, "phi3 qquant-decode: generated %d tokens in %.2f s (%.3f s/tok)\n",
            n_gen, gen_ms / 1000.0, gen_ms / 1000.0 / std::max(1, n_gen));

    phi3_fused_ctx_free(cx);
    phi3_matmul_pool_free(pool);
    return true;
}

