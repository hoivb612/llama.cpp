#pragma once

#include "phi3_loader.h"

#include <cstdint>
#include <string>

struct Phi3FusionDiagnostics {
    int32_t head_dim = 0;
    int32_t gqa_ratio = 0;
    bool qkv_shape_ok = false;
    bool qkv_v2_shape_ok = false;
    bool mlp_shape_ok = false;
    bool has_ffn_meta = false;
    bool decode_fusion_candidate = false;
    bool decode_qkv_v2_candidate = false;
    std::string summary;
};

struct Phi3ExecutionPlan {
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    bool fuse_qkv = false;
    bool fuse_qkv_v2 = false;
    bool fuse_mlp = false;
    std::string notes;
    Phi3FusionDiagnostics diagnostics;
};

bool phi3_build_execution_plan(const Phi3RawModel & raw, Phi3ExecutionPlan & out, std::string & error);
