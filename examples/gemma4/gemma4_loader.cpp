#include "gemma4_loader.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>

// -----------------------------------------------------------------------
// Small helpers for reading GGUF string metadata via the public C API.
// These mirror the phi3 loader so the two examples behave consistently.
// -----------------------------------------------------------------------

static std::string get_meta_str(const llama_model * model, const std::string & key) {
    // First sizing call: pass a small buffer; if it overflows, retry with
    // the exact size the API tells us we need.
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

static int32_t get_meta_i32(const llama_model * model, const std::string & key, int32_t fallback) {
    const std::string val = get_meta_str(model, key);
    if (val.empty()) return fallback;
    try {
        return (int32_t) std::stol(val);
    } catch (const std::exception &) {
        return fallback;
    }
}

static float get_meta_f32(const llama_model * model, const std::string & key, float fallback) {
    const std::string val = get_meta_str(model, key);
    if (val.empty()) return fallback;
    try {
        return std::stof(val);
    } catch (const std::exception &) {
        return fallback;
    }
}

// Detect MoE by checking whether the model has any tensor matching the
// expert-bank naming convention. We don't need the data, just the
// presence -- llama_model_get_tensor_by_name returns non-null when
// the tensor exists in the loaded model.
static bool detect_moe(const llama_model * model, int n_layer) {
    for (int il = 0; il < n_layer; ++il) {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", il);
        if (llama_model_get_tensor_by_name(model, nm) != nullptr) {
            return true;
        }
    }
    return false;
}

// -----------------------------------------------------------------------
// Public entry points.
// -----------------------------------------------------------------------

bool gemma4_load_raw_model(const Gemma4LoadParams & params,
                           Gemma4RawModel         & out,
                           std::string            & error) {
    gemma4_unload_raw_model(out);
    error.clear();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = params.n_gpu_layers;

    out.model = llama_model_load_from_file(params.model_path.c_str(), mp);
    if (!out.model) {
        error = "gemma4_load_raw_model: unable to load model from '" + params.model_path + "'";
        return false;
    }
    out.vocab = llama_model_get_vocab(out.model);
    if (!out.vocab) {
        error = "gemma4_load_raw_model: model has no vocab";
        gemma4_unload_raw_model(out);
        return false;
    }

    out.architecture = get_meta_str(out.model, "general.architecture");
    out.model_name   = get_meta_str(out.model, "general.name");

    if (out.architecture != "gemma4") {
        // Soft-fail with a clear message rather than crashing later in
        // the baseline path with shape errors.
        std::ostringstream oss;
        oss << "gemma4_load_raw_model: expected general.architecture == "
            << "\"gemma4\", got \"" << out.architecture << "\"";
        error = oss.str();
        gemma4_unload_raw_model(out);
        return false;
    }

    // Single-value hparams. We use the model APIs as the source of
    // truth where possible and fall back to GGUF metadata for keys
    // not exposed by llama_model_*.
    out.n_layer        = llama_model_n_layer(out.model);
    out.n_embd         = llama_model_n_embd(out.model);
    out.n_head         = llama_model_n_head(out.model);
    out.n_head_kv      = llama_model_n_head_kv(out.model);
    out.context_length = get_meta_i32(out.model, "gemma4.context_length", 0);
    out.head_dim_k     = get_meta_i32(out.model, "gemma4.attention.key_length", 0);
    out.head_dim_v     = get_meta_i32(out.model, "gemma4.attention.value_length", 0);
    out.sliding_window = get_meta_i32(out.model, "gemma4.attention.sliding_window", 0);
    out.rope_dim       = get_meta_i32(out.model, "gemma4.rope.dimension_count", 0);
    out.rope_freq_base = get_meta_f32(out.model, "gemma4.rope.freq_base", 0.0f);

    // Per-layer arrays. The metadata key may either be a scalar (one
    // value applied to every layer) or an array (one value per layer).
    // We populate the vector to length n_layer regardless so callers
    // can index by layer without bounds-checking.
    out.n_ff_per_layer.assign((size_t) out.n_layer, 0);
    {
        // Scalar-or-array: try array length via the val_str API which
        // returns a comma-ish serialised form; if that's not how the
        // GGUF reader exposes it, the value will either be a single
        // integer or empty.
        //
        // Practical approach: query the scalar first; if empty, leave
        // 0s. Downstream code that cares about per-layer ffn (i.e.
        // dense G1 baseline) only needs total layer count alignment;
        // for MoE we'll surface this differently in a later phase.
        const int32_t n_ff_scalar = get_meta_i32(out.model, "gemma4.feed_forward_length", 0);
        if (n_ff_scalar > 0) {
            for (auto & v : out.n_ff_per_layer) v = n_ff_scalar;
        }
    }
    out.sliding_window_pattern.assign((size_t) out.n_layer, 0);

    out.is_moe = detect_moe(out.model, out.n_layer);

    return true;
}

void gemma4_unload_raw_model(Gemma4RawModel & model) {
    if (model.model) {
        llama_model_free(model.model);
    }
    model = {};
}

void gemma4_log_summary(const Gemma4RawModel & m) {
    std::fprintf(stderr,
                 "gemma4 model: arch=%s name=\"%s\" n_layer=%d n_embd=%d "
                 "n_head=%d n_head_kv=%d head_dim_k=%d head_dim_v=%d "
                 "ctx_len=%d swa=%d rope_dim=%d rope_base=%.0f moe=%d\n",
                 m.architecture.c_str(),
                 m.model_name.c_str(),
                 (int) m.n_layer, (int) m.n_embd,
                 (int) m.n_head, (int) m.n_head_kv,
                 (int) m.head_dim_k, (int) m.head_dim_v,
                 (int) m.context_length, (int) m.sliding_window,
                 (int) m.rope_dim, (double) m.rope_freq_base,
                 (int) m.is_moe);
}
