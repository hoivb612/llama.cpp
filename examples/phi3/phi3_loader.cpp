#include "phi3_loader.h"

#include <array>
#include <exception>

static std::string get_meta_str(const llama_model * model, const std::string & key) {
    std::array<char, 256> buf = {};
    const int n = llama_model_meta_val_str(model, key.c_str(), buf.data(), buf.size());
    if (n < 0) {
        return "";
    }
    return std::string(buf.data(), (size_t) n);
}

static int32_t get_meta_i32(const llama_model * model, const std::string & key, int32_t fallback) {
    const std::string val = get_meta_str(model, key);
    if (val.empty()) {
        return fallback;
    }
    try {
        return std::stoi(val);
    } catch (const std::exception &) {
        return fallback;
    }
}

bool phi3_load_raw_model(const Phi3LoadParams & params, Phi3RawModel & out, std::string & error) {
    phi3_unload_raw_model(out);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    out.model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!out.model) {
        error = "unable to load model";
        return false;
    }

    out.vocab = llama_model_get_vocab(out.model);
    if (!out.vocab) {
        error = "unable to read vocab from model";
        phi3_unload_raw_model(out);
        return false;
    }

    out.architecture = get_meta_str(out.model, "general.architecture");
    out.model_name = get_meta_str(out.model, "general.name");
    if (out.architecture.empty()) {
        out.architecture = "phi3";
    }

    out.n_layer = get_meta_i32(out.model, out.architecture + ".block_count", llama_model_n_layer(out.model));
    out.n_embd  = get_meta_i32(out.model, out.architecture + ".embedding_length", llama_model_n_embd(out.model));
    out.n_head = get_meta_i32(out.model, out.architecture + ".attention.head_count", llama_model_n_head(out.model));
    out.n_head_kv = get_meta_i32(out.model, out.architecture + ".attention.head_count_kv", llama_model_n_head_kv(out.model));
    out.n_ff = get_meta_i32(out.model, out.architecture + ".feed_forward_length", 0);

    error.clear();
    return true;
}

void phi3_unload_raw_model(Phi3RawModel & model) {
    if (model.model) {
        llama_model_free(model.model);
    }
    model = {};
}
