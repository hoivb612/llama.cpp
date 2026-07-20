#include "gemma4_weights.h"

#include "ggml.h"
#include "llama.h"
#include "llama_b612.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>

namespace gemma4 {

namespace {

// ---------------------------------------------------------------------
// GGUF metadata helpers (string-typed values from the public C API).
// ---------------------------------------------------------------------

std::string get_meta_str(const llama_model * model, const std::string & key) {
    std::array<char, 512> small = {};
    int n = llama_model_meta_val_str(model, key.c_str(), small.data(), small.size());
    if (n < 0) return "";
    if (n < (int) small.size()) {
        return std::string(small.data(), (size_t) n);
    }
    std::string big((size_t) n + 1, '\0');
    n = llama_model_meta_val_str(model, key.c_str(), big.data(), big.size());
    if (n < 0) return "";
    return std::string(big.data(), (size_t) n);
}

int32_t get_meta_i32(const llama_model * model, const std::string & key, int32_t fallback) {
    const std::string val = get_meta_str(model, key);
    if (val.empty()) return fallback;
    try { return (int32_t) std::stol(val); } catch (...) { return fallback; }
}

float get_meta_f32(const llama_model * model, const std::string & key, float fallback) {
    const std::string val = get_meta_str(model, key);
    if (val.empty()) return fallback;
    try { return std::stof(val); } catch (...) { return fallback; }
}

// ---------------------------------------------------------------------
// Tensor resolution helpers.
// ---------------------------------------------------------------------

const ggml_tensor * require_tensor(const llama_model * model, const std::string & name, std::string & error) {
    const ggml_tensor * t = llama_model_get_tensor_by_name(model, name.c_str());
    if (t == nullptr) {
        error = "gemma4::resolve: missing required tensor '" + name + "'";
    }
    return t;
}

const ggml_tensor * try_tensor(const llama_model * model, const char * name) {
    return llama_model_get_tensor_by_name(model, name);
}

std::string blk_name(int il, const char * leaf) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s.weight", il, leaf);
    return buf;
}

std::string blk_scale_name(int il, const char * leaf) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s.scale", il, leaf);
    return buf;
}

bool check_f32_1d(const ggml_tensor * t, int64_t n, const char * what, std::string & error) {
    if (t->type != GGML_TYPE_F32) {
        std::ostringstream oss;
        oss << "gemma4::resolve: " << what << " has type " << ggml_type_name(t->type)
            << ", expected F32";
        error = oss.str();
        return false;
    }
    if (t->ne[0] != n || t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) {
        std::ostringstream oss;
        oss << "gemma4::resolve: " << what << " has shape [" << t->ne[0] << ","
            << t->ne[1] << "," << t->ne[2] << "," << t->ne[3]
            << "], expected [" << n << ",1,1,1]";
        error = oss.str();
        return false;
    }
    return true;
}

bool check_mat2d(const ggml_tensor * t, int64_t d0, int64_t d1, const char * what, std::string & error) {
    if (t->ne[0] != d0 || t->ne[1] != d1 || t->ne[2] != 1 || t->ne[3] != 1) {
        std::ostringstream oss;
        oss << "gemma4::resolve: " << what << " has shape [" << t->ne[0] << ","
            << t->ne[1] << "," << t->ne[2] << "," << t->ne[3]
            << "], expected [" << d0 << "," << d1 << ",1,1]";
        error = oss.str();
        return false;
    }
    return true;
}

bool check_mat3d(const ggml_tensor * t, int64_t d0, int64_t d1, int64_t d2, const char * what, std::string & error) {
    if (t->ne[0] != d0 || t->ne[1] != d1 || t->ne[2] != d2 || t->ne[3] != 1) {
        std::ostringstream oss;
        oss << "gemma4::resolve: " << what << " has shape [" << t->ne[0] << ","
            << t->ne[1] << "," << t->ne[2] << "," << t->ne[3]
            << "], expected [" << d0 << "," << d1 << "," << d2 << ",1]";
        error = oss.str();
        return false;
    }
    return true;
}

bool detect_moe(const llama_model * model, int n_layer) {
    for (int il = 0; il < n_layer; ++il) {
        if (try_tensor(model, blk_name(il, "ffn_gate_inp").c_str()) != nullptr) {
            return true;
        }
    }
    return false;
}

} // namespace

// ---------------------------------------------------------------------
// Resolver.
// ---------------------------------------------------------------------

bool resolve(const llama_model * model, Weights & out, std::string & error) {
    error.clear();
    out = Weights{};

    if (!model) {
        error = "gemma4::resolve: model is null";
        return false;
    }

    const std::string arch = get_meta_str(model, "general.architecture");
    if (arch != "gemma4") {
        std::ostringstream oss;
        oss << "gemma4::resolve: expected general.architecture==\"gemma4\", got \""
            << arch << "\"";
        error = oss.str();
        return false;
    }

    out.n_layer    = llama_model_n_layer(model);
    out.n_embd     = llama_model_n_embd(model);
    out.n_head     = llama_model_n_head(model);
    out.n_head_kv  = llama_model_n_head_kv(model);

    if (out.n_layer <= 0 || out.n_head <= 0 || out.n_head_kv <= 0 || out.n_embd <= 0) {
        std::ostringstream oss;
        oss << "gemma4::resolve: invalid hparams (n_layer=" << out.n_layer
            << " n_embd=" << out.n_embd << " n_head=" << out.n_head
            << " n_head_kv=" << out.n_head_kv << ")";
        error = oss.str();
        return false;
    }

    out.is_moe = detect_moe(model, out.n_layer);
    if (out.is_moe) {
        out.n_expert      = get_meta_i32(model, "gemma4.expert_count",               0);
        out.n_expert_used = get_meta_i32(model, "gemma4.expert_used_count",          0);
        out.n_ff_exp      = get_meta_i32(model, "gemma4.expert_feed_forward_length", 0);
        if (out.n_expert <= 0 || out.n_expert_used <= 0 || out.n_expert_used > out.n_expert) {
            std::ostringstream oss;
            oss << "gemma4::resolve: invalid MoE hparams (n_expert=" << out.n_expert
                << " n_expert_used=" << out.n_expert_used << ")";
            error = oss.str();
            return false;
        }
    }

    out.n_swa             = get_meta_i32(model, "gemma4.attention.sliding_window", 0);
    {
        const int32_t shared = get_meta_i32(model, "gemma4.attention.shared_kv_layers", 0);
        if (shared > 0 && shared < out.n_layer) {
            out.n_layer_kv_from_start = out.n_layer - shared;
        } else {
            out.n_layer_kv_from_start = -1;
        }
    }
    out.rope_dim          = get_meta_i32(model, "gemma4.rope.dimension_count", 0);
    out.rope_freq_base    = get_meta_f32(model, "gemma4.rope.freq_base", 0.0f);
    out.rope_freq_base_swa = get_meta_f32(model, "gemma4.rope.freq_base_swa", out.rope_freq_base);
    out.rms_eps           = get_meta_f32(model, "gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
    out.final_logit_softcap = get_meta_f32(model, "gemma4.final_logit_softcapping", 0.0f);

    const int head_dim_full = get_meta_i32(model, "gemma4.attention.key_length",     0);
    const int head_dim_swa  = get_meta_i32(model, "gemma4.attention.key_length_swa", head_dim_full);
    if (head_dim_full <= 0) {
        error = "gemma4::resolve: missing gemma4.attention.key_length (head_dim full)";
        return false;
    }

    // Globals.
    out.tok_embd = require_tensor(model, "token_embd.weight", error);
    if (!out.tok_embd) return false;
    if (out.tok_embd->ne[0] != out.n_embd) {
        std::ostringstream oss;
        oss << "gemma4::resolve: token_embd.weight ne[0]=" << out.tok_embd->ne[0]
            << " != n_embd=" << out.n_embd;
        error = oss.str();
        return false;
    }
    out.n_vocab = (int) out.tok_embd->ne[1];

    out.output = try_tensor(model, "output.weight");
    if (!out.output) {
        out.output = out.tok_embd;
        out.output_tied_to_embd = true;
    } else if (out.output->ne[0] != out.n_embd || out.output->ne[1] != out.n_vocab) {
        std::ostringstream oss;
        oss << "gemma4::resolve: output.weight shape [" << out.output->ne[0]
            << "," << out.output->ne[1] << "] != expected [" << out.n_embd
            << "," << out.n_vocab << "]";
        error = oss.str();
        return false;
    }

    out.output_norm = require_tensor(model, "output_norm.weight", error);
    if (!out.output_norm) return false;
    if (!check_f32_1d(out.output_norm, out.n_embd, "output_norm.weight", error)) return false;

    // Per-layer embedding globals (Gemma PLE). Present on E2B/E4B, absent on the
    // 26B-A4B MoE variant (embedding_length_per_layer_input == 0). When
    // per_layer_token_embd is absent, n_embd_per_layer stays 0 and the whole PLE
    // path (globals + per-layer inp_gate/proj) is skipped.
    out.per_layer_tok_embd = try_tensor(model, "per_layer_token_embd.weight");
    if (out.per_layer_tok_embd) {
        out.per_layer_model_proj = require_tensor(model, "per_layer_model_proj.weight",  error);
        if (!out.per_layer_model_proj) return false;
        out.per_layer_proj_norm  = require_tensor(model, "per_layer_proj_norm.weight",   error);
        if (!out.per_layer_proj_norm) return false;

        // n_embd_per_layer is recoverable from per_layer_token_embd shape:
        // per_layer_token_embd.weight has shape [n_embd_per_layer * n_layer, n_vocab].
        if (out.per_layer_tok_embd->ne[1] != out.n_vocab) {
            std::ostringstream oss;
            oss << "gemma4::resolve: per_layer_token_embd.weight ne[1]="
                << out.per_layer_tok_embd->ne[1] << " != n_vocab=" << out.n_vocab;
            error = oss.str();
            return false;
        }
        if (out.per_layer_tok_embd->ne[0] % out.n_layer != 0) {
            std::ostringstream oss;
            oss << "gemma4::resolve: per_layer_token_embd.weight ne[0]="
                << out.per_layer_tok_embd->ne[0] << " not divisible by n_layer="
                << out.n_layer;
            error = oss.str();
            return false;
        }
        out.n_embd_per_layer = (int) (out.per_layer_tok_embd->ne[0] / out.n_layer);

        // Validate the other per-layer-embedding globals against the derived dim.
        if (out.per_layer_model_proj->ne[0] != out.n_embd ||
            out.per_layer_model_proj->ne[1] != (int64_t) out.n_embd_per_layer * out.n_layer) {
            std::ostringstream oss;
            oss << "gemma4::resolve: per_layer_model_proj shape ["
                << out.per_layer_model_proj->ne[0] << ","
                << out.per_layer_model_proj->ne[1] << "] != expected ["
                << out.n_embd << "," << out.n_embd_per_layer * out.n_layer << "]";
            error = oss.str();
            return false;
        }
        if (!check_f32_1d(out.per_layer_proj_norm, out.n_embd_per_layer,
                          "per_layer_proj_norm.weight", error)) return false;
    } else {
        out.n_embd_per_layer     = 0;
        out.per_layer_model_proj = nullptr;
        out.per_layer_proj_norm  = nullptr;
    }

    // rope_freqs is a single global in our dense Gemma-4 GGUFs (stored once).
    out.rope_freqs = try_tensor(model, "rope_freqs.weight");

    // ---- Per-layer tensors. ----
    out.layers.assign((size_t) out.n_layer, LayerWeights{});

    for (int il = 0; il < out.n_layer; ++il) {
        LayerWeights & L = out.layers[il];

        const std::string s_attn_norm        = blk_name(il, "attn_norm");
        const std::string s_attn_q           = blk_name(il, "attn_q");
        const std::string s_attn_k           = blk_name(il, "attn_k");
        const std::string s_attn_v           = blk_name(il, "attn_v");
        const std::string s_attn_q_norm      = blk_name(il, "attn_q_norm");
        const std::string s_attn_k_norm      = blk_name(il, "attn_k_norm");
        const std::string s_attn_output      = blk_name(il, "attn_output");
        const std::string s_post_attn_norm   = blk_name(il, "post_attention_norm");
        const std::string s_post_norm        = blk_name(il, "post_norm");
        const std::string s_ffn_norm         = blk_name(il, "ffn_norm");
        const std::string s_ffn_gate         = blk_name(il, "ffn_gate");
        const std::string s_ffn_up           = blk_name(il, "ffn_up");
        const std::string s_ffn_down         = blk_name(il, "ffn_down");
        const std::string s_post_ffw_norm    = blk_name(il, "post_ffw_norm");
        const std::string s_inp_gate         = blk_name(il, "inp_gate");
        const std::string s_proj             = blk_name(il, "proj");
        const std::string s_layer_out_scale  = blk_name(il, "layer_output_scale");

        L.attn_norm        = require_tensor(model, s_attn_norm,        error); if (!L.attn_norm)        return false;
        L.wq               = require_tensor(model, s_attn_q,           error); if (!L.wq)               return false;
        L.wk               = require_tensor(model, s_attn_k,           error); if (!L.wk)               return false;
        L.wv               = try_tensor    (model, s_attn_v.c_str());           // optional
        L.attn_q_norm      = require_tensor(model, s_attn_q_norm,      error); if (!L.attn_q_norm)      return false;
        L.attn_k_norm      = require_tensor(model, s_attn_k_norm,      error); if (!L.attn_k_norm)      return false;
        L.wo               = require_tensor(model, s_attn_output,      error); if (!L.wo)               return false;
        L.post_attn_norm   = require_tensor(model, s_post_attn_norm,   error); if (!L.post_attn_norm)   return false;
        L.post_norm        = try_tensor    (model, s_post_norm.c_str());        // optional (dense E2B/E4B only; absent on MoE)
        L.ffn_norm         = require_tensor(model, s_ffn_norm,         error); if (!L.ffn_norm)         return false;
        L.ffn_gate         = require_tensor(model, s_ffn_gate,         error); if (!L.ffn_gate)         return false;
        L.ffn_up           = require_tensor(model, s_ffn_up,           error); if (!L.ffn_up)           return false;
        L.ffn_down         = require_tensor(model, s_ffn_down,         error); if (!L.ffn_down)         return false;
        L.post_ffw_norm    = require_tensor(model, s_post_ffw_norm,    error); if (!L.post_ffw_norm)    return false;
        L.inp_gate         = (out.n_embd_per_layer > 0) ? require_tensor(model, s_inp_gate, error) : nullptr;
        if (out.n_embd_per_layer > 0 && !L.inp_gate) return false;
        L.proj             = (out.n_embd_per_layer > 0) ? require_tensor(model, s_proj, error) : nullptr;
        if (out.n_embd_per_layer > 0 && !L.proj) return false;
        L.layer_output_scale = try_tensor(model, s_layer_out_scale.c_str());   // optional

        L.has_kv     = (L.wk != nullptr);
        L.has_v_proj = (L.wv != nullptr);

        // Derive per-layer head_dim from attn_q shape.
        if (L.wq->ne[0] != out.n_embd) {
            std::ostringstream oss;
            oss << "gemma4::resolve: " << s_attn_q << " ne[0]=" << L.wq->ne[0]
                << " != n_embd=" << out.n_embd;
            error = oss.str();
            return false;
        }
        if (L.wq->ne[1] % out.n_head != 0) {
            std::ostringstream oss;
            oss << "gemma4::resolve: " << s_attn_q << " ne[1]=" << L.wq->ne[1]
                << " not divisible by n_head=" << out.n_head;
            error = oss.str();
            return false;
        }
        L.head_dim = (int) (L.wq->ne[1] / out.n_head);

        // Classify SWA vs full from observed head_dim.
        if (L.head_dim == head_dim_swa) {
            L.is_swa = true;
        } else if (L.head_dim == head_dim_full) {
            L.is_swa = false;
        } else {
            std::ostringstream oss;
            oss << "gemma4::resolve: layer " << il << " head_dim=" << L.head_dim
                << " matches neither key_length=" << head_dim_full
                << " nor key_length_swa=" << head_dim_swa;
            error = oss.str();
            return false;
        }

        // attn_k / attn_v / norms must match the derived head_dim.
        // n_head_kv is derived per-layer from attn_k (Gemma-4 26B-A4B varies it:
        // SWA layers use 8 KV heads, full-attention layers use 2).
        if (L.wk->ne[0] != out.n_embd || L.wk->ne[1] % L.head_dim != 0) {
            std::ostringstream oss;
            oss << "gemma4::resolve: " << s_attn_k << " shape [" << L.wk->ne[0] << ","
                << L.wk->ne[1] << "] incompatible with n_embd=" << out.n_embd
                << " head_dim=" << L.head_dim;
            error = oss.str();
            return false;
        }
        L.n_head_kv = (int) (L.wk->ne[1] / L.head_dim);
        if (L.wv && !check_mat2d(L.wv, out.n_embd, (int64_t) L.n_head_kv * L.head_dim,
                                 s_attn_v.c_str(), error)) return false;
        if (!check_f32_1d(L.attn_q_norm, L.head_dim, s_attn_q_norm.c_str(), error)) return false;
        if (!check_f32_1d(L.attn_k_norm, L.head_dim, s_attn_k_norm.c_str(), error)) return false;
        if (!check_mat2d(L.wo, (int64_t) out.n_head * L.head_dim, out.n_embd,
                         s_attn_output.c_str(), error)) return false;

        if (!check_f32_1d(L.attn_norm,      out.n_embd, s_attn_norm.c_str(),      error)) return false;
        if (!check_f32_1d(L.post_attn_norm, out.n_embd, s_post_attn_norm.c_str(), error)) return false;
        if (L.post_norm && !check_f32_1d(L.post_norm, out.n_embd, s_post_norm.c_str(), error)) return false;
        if (!check_f32_1d(L.ffn_norm,       out.n_embd, s_ffn_norm.c_str(),       error)) return false;
        if (!check_f32_1d(L.post_ffw_norm,  out.n_embd, s_post_ffw_norm.c_str(),  error)) return false;

        // Derive per-layer n_ff from ffn_gate.
        if (L.ffn_gate->ne[0] != out.n_embd) {
            std::ostringstream oss;
            oss << "gemma4::resolve: " << s_ffn_gate << " ne[0]=" << L.ffn_gate->ne[0]
                << " != n_embd=" << out.n_embd;
            error = oss.str();
            return false;
        }
        L.n_ff = (int) L.ffn_gate->ne[1];

        if (!check_mat2d(L.ffn_up,   out.n_embd, L.n_ff,    s_ffn_up.c_str(),   error)) return false;
        if (!check_mat2d(L.ffn_down, L.n_ff,    out.n_embd, s_ffn_down.c_str(), error)) return false;

        if (out.n_embd_per_layer > 0) {
            if (!check_mat2d(L.inp_gate, out.n_embd, out.n_embd_per_layer, s_inp_gate.c_str(), error)) return false;
            if (!check_mat2d(L.proj,     out.n_embd_per_layer, out.n_embd, s_proj.c_str(),     error)) return false;
        }

        if (L.layer_output_scale) {
            if (L.layer_output_scale->type != GGML_TYPE_F32 ||
                ggml_nelements(L.layer_output_scale) != 1) {
                std::ostringstream oss;
                oss << "gemma4::resolve: " << s_layer_out_scale << " expected scalar F32, got type "
                    << ggml_type_name(L.layer_output_scale->type) << " nelements="
                    << ggml_nelements(L.layer_output_scale);
                error = oss.str();
                return false;
            }
        }

        // ---- MoE tensors (26B-A4B). A layer is MoE iff ffn_gate_inp is present.
        // The dense ffn_gate/ffn_up/ffn_down resolved above act as the shared
        // expert on these layers. ----
        L.ffn_gate_inp = try_tensor(model, blk_name(il, "ffn_gate_inp").c_str());
        L.is_moe_layer = (L.ffn_gate_inp != nullptr);
        if (L.is_moe_layer) {
            const std::string s_gate_inp_s  = blk_scale_name(il, "ffn_gate_inp");
            const std::string s_pre_norm_2  = blk_name(il, "pre_ffw_norm_2");
            const std::string s_post_norm_1 = blk_name(il, "post_ffw_norm_1");
            const std::string s_post_norm_2 = blk_name(il, "post_ffw_norm_2");
            const std::string s_down_exps   = blk_name(il, "ffn_down_exps");

            L.ffn_gate_inp_s  = require_tensor(model, s_gate_inp_s,  error); if (!L.ffn_gate_inp_s)  return false;
            L.ffn_pre_norm_2  = require_tensor(model, s_pre_norm_2,  error); if (!L.ffn_pre_norm_2)  return false;
            L.ffn_post_norm_1 = require_tensor(model, s_post_norm_1, error); if (!L.ffn_post_norm_1) return false;
            L.ffn_post_norm_2 = require_tensor(model, s_post_norm_2, error); if (!L.ffn_post_norm_2) return false;

            if (!check_mat2d(L.ffn_gate_inp, out.n_embd, out.n_expert,
                             blk_name(il, "ffn_gate_inp").c_str(), error)) return false;
            if (!check_f32_1d(L.ffn_gate_inp_s,  out.n_embd, s_gate_inp_s.c_str(),  error)) return false;
            if (!check_f32_1d(L.ffn_pre_norm_2,  out.n_embd, s_pre_norm_2.c_str(),  error)) return false;
            if (!check_f32_1d(L.ffn_post_norm_1, out.n_embd, s_post_norm_1.c_str(), error)) return false;
            if (!check_f32_1d(L.ffn_post_norm_2, out.n_embd, s_post_norm_2.c_str(), error)) return false;

            // Experts: merged gate_up (as in the shipped GGUF) or separate gate/up.
            L.ffn_gate_up_exps = try_tensor(model, blk_name(il, "ffn_gate_up_exps").c_str());
            if (L.ffn_gate_up_exps) {
                if (!check_mat3d(L.ffn_gate_up_exps, out.n_embd, (int64_t) out.n_ff_exp * 2, out.n_expert,
                                 blk_name(il, "ffn_gate_up_exps").c_str(), error)) return false;
                L.ffn_gate_up_exps_s = try_tensor(model, blk_scale_name(il, "ffn_gate_up_exps").c_str()); // optional
                if (L.ffn_gate_up_exps_s &&
                    !check_f32_1d(L.ffn_gate_up_exps_s, out.n_expert,
                                  blk_scale_name(il, "ffn_gate_up_exps").c_str(), error)) return false;
            } else {
                L.ffn_gate_exps = require_tensor(model, blk_name(il, "ffn_gate_exps"), error); if (!L.ffn_gate_exps) return false;
                L.ffn_up_exps   = require_tensor(model, blk_name(il, "ffn_up_exps"),   error); if (!L.ffn_up_exps)   return false;
                if (!check_mat3d(L.ffn_gate_exps, out.n_embd, out.n_ff_exp, out.n_expert,
                                 blk_name(il, "ffn_gate_exps").c_str(), error)) return false;
                if (!check_mat3d(L.ffn_up_exps,   out.n_embd, out.n_ff_exp, out.n_expert,
                                 blk_name(il, "ffn_up_exps").c_str(), error)) return false;
            }

            L.ffn_down_exps = require_tensor(model, s_down_exps, error); if (!L.ffn_down_exps) return false;
            if (!check_mat3d(L.ffn_down_exps, out.n_ff_exp, out.n_embd, out.n_expert,
                             s_down_exps.c_str(), error)) return false;
            L.ffn_down_exps_s = try_tensor(model, blk_scale_name(il, "ffn_down_exps").c_str()); // optional
            if (L.ffn_down_exps_s &&
                !check_f32_1d(L.ffn_down_exps_s, out.n_expert,
                              blk_scale_name(il, "ffn_down_exps").c_str(), error)) return false;
        }
    }

    // Populate per-layer kv_reuse_il using the shared-KV rule from
    // llama-model.cpp (LLM_ARCH_GEMMA4 case):
    //   if il >= n_layer_kv_from_start:
    //       reuse = n_layer_kv_from_start - (is_swa(il) ? 2 : 1)
    if (out.n_layer_kv_from_start > 0 && out.n_layer_kv_from_start < out.n_layer) {
        for (int il = 0; il < out.n_layer; ++il) {
            LayerWeights & L = out.layers[il];
            if (il < out.n_layer_kv_from_start) {
                L.kv_reuse_il = -1;
            } else {
                L.kv_reuse_il = out.n_layer_kv_from_start - (L.is_swa ? 2 : 1);
                if (L.kv_reuse_il < 0 || L.kv_reuse_il >= out.n_layer_kv_from_start) {
                    std::ostringstream oss;
                    oss << "gemma4::resolve: layer " << il << " computed kv_reuse_il="
                        << L.kv_reuse_il << " out of valid range [0,"
                        << out.n_layer_kv_from_start << ")";
                    error = oss.str();
                    return false;
                }
                // The source layer must be the same SWA type as us (so head_dim and RoPE match).
                if (out.layers[L.kv_reuse_il].is_swa != L.is_swa) {
                    std::ostringstream oss;
                    oss << "gemma4::resolve: layer " << il << " (is_swa=" << L.is_swa
                        << ") reuses KV from layer " << L.kv_reuse_il
                        << " (is_swa=" << out.layers[L.kv_reuse_il].is_swa << ") -- mismatch";
                    error = oss.str();
                    return false;
                }
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------
// Dumper.
// ---------------------------------------------------------------------

void dump(const Weights & w) {
    auto tn = [](const ggml_tensor * t) { return t ? ggml_type_name(t->type) : "absent"; };

    std::fprintf(stderr,
        "gemma4 weights: n_layer=%d n_embd=%d n_head=%d n_head_kv=%d "
        "n_vocab=%d n_embd_per_layer=%d swa=%d rope_dim=%d rope_base=%.0f rope_base_swa=%.0f "
        "rms_eps=%g softcap=%.1f tied_output=%d rope_freqs=%s\n",
        w.n_layer, w.n_embd, w.n_head, w.n_head_kv, w.n_vocab, w.n_embd_per_layer,
        w.n_swa, w.rope_dim, (double) w.rope_freq_base, (double) w.rope_freq_base_swa,
        (double) w.rms_eps,
        (double) w.final_logit_softcap, (int) w.output_tied_to_embd,
        w.rope_freqs ? "present" : "absent");

    if (w.is_moe) {
        std::fprintf(stderr,
            "  MoE: n_expert=%d n_expert_used=%d n_ff_exp=%d\n",
            w.n_expert, w.n_expert_used, w.n_ff_exp);
    }

    // Per-layer summary -- group consecutive identical rows for brevity.
    int run_start = 0;
    auto layer_sig = [&](int il) {
        const auto & L = w.layers[il];
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "swa=%d moe=%d head_dim=%d n_head_kv=%d n_ff=%d has_v=%d has_scale=%d wq=%s wk=%s wo=%s ffn_gate=%s ffn_down=%s gate_up_exps=%s down_exps=%s",
            (int) L.is_swa, (int) L.is_moe_layer, L.head_dim, L.n_head_kv, L.n_ff,
            (int) L.has_v_proj, (int) (L.layer_output_scale != nullptr),
            ggml_type_name(L.wq->type), ggml_type_name(L.wk->type),
            ggml_type_name(L.wo->type),
            ggml_type_name(L.ffn_gate->type), ggml_type_name(L.ffn_down->type),
            tn(L.ffn_gate_up_exps), tn(L.ffn_down_exps));
        return std::string(buf);
    };

    std::string cur = w.layers.empty() ? std::string() : layer_sig(0);
    for (int il = 1; il <= w.n_layer; ++il) {
        std::string sig = (il < w.n_layer) ? layer_sig(il) : std::string("<sentinel>");
        if (sig != cur) {
            std::fprintf(stderr, "  layers %2d..%2d  %s\n", run_start, il - 1, cur.c_str());
            run_start = il;
            cur = sig;
        }
    }

    std::fprintf(stderr,
        "  globals: tok_embd=%s output=%s output_norm=%s per_layer_tok_embd=%s "
        "per_layer_model_proj=%s per_layer_proj_norm=%s\n",
        tn(w.tok_embd), tn(w.output), tn(w.output_norm),
        tn(w.per_layer_tok_embd), tn(w.per_layer_model_proj), tn(w.per_layer_proj_norm));
}

} // namespace gemma4
