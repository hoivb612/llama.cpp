#pragma once

#include "llama.h"

#include <cstdint>
#include <string>

struct Phi3LoadParams {
    std::string model_path;
    int n_gpu_layers = 99;
};

struct Phi3RawModel {
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    std::string architecture;
    std::string model_name;
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t n_ff = 0;
};

bool phi3_load_raw_model(const Phi3LoadParams & params, Phi3RawModel & out, std::string & error);
void phi3_unload_raw_model(Phi3RawModel & model);
