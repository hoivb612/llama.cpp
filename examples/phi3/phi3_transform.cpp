#include "phi3_transform.h"

#include <sstream>

bool phi3_build_execution_plan(const Phi3RawModel & raw, Phi3ExecutionPlan & out, std::string & error) {
    if (!raw.model) {
        error = "raw model is not loaded";
        return false;
    }

    if (raw.architecture.rfind("phi", 0) != 0) {
        error = "expected a Phi-family GGUF, got architecture '" + raw.architecture + "'";
        return false;
    }

    out = {};
    out.n_layer = raw.n_layer;
    out.n_embd = raw.n_embd;

    out.diagnostics.head_dim = raw.n_head > 0 ? (raw.n_embd / raw.n_head) : 0;
    out.diagnostics.gqa_ratio = (raw.n_head_kv > 0) ? (raw.n_head / raw.n_head_kv) : 0;
    out.diagnostics.qkv_shape_ok = raw.n_head > 0
        && raw.n_head_kv > 0
        && raw.n_embd > 0
        && (raw.n_embd % raw.n_head == 0)
        && (raw.n_head % raw.n_head_kv == 0);
    out.diagnostics.qkv_v2_shape_ok = out.diagnostics.qkv_shape_ok
        && out.diagnostics.head_dim > 0
        && (out.diagnostics.head_dim % 32 == 0)
        && (raw.n_embd % 128 == 0);
    out.diagnostics.has_ffn_meta = raw.n_ff > 0;
    out.diagnostics.mlp_shape_ok = raw.n_ff > 0 && (raw.n_ff % 256 == 0);
    out.diagnostics.decode_fusion_candidate = out.diagnostics.qkv_shape_ok && out.diagnostics.mlp_shape_ok;
    out.diagnostics.decode_qkv_v2_candidate = out.diagnostics.qkv_v2_shape_ok && out.diagnostics.mlp_shape_ok;

    std::ostringstream oss;
    oss << "heads=" << raw.n_head
        << " kv_heads=" << raw.n_head_kv
        << " head_dim=" << out.diagnostics.head_dim
        << " gqa=" << out.diagnostics.gqa_ratio
        << " ffn=" << raw.n_ff
        << " qkv_shape_ok=" << (out.diagnostics.qkv_shape_ok ? 1 : 0)
        << " qkv_v2_shape_ok=" << (out.diagnostics.qkv_v2_shape_ok ? 1 : 0)
        << " mlp_shape_ok=" << (out.diagnostics.mlp_shape_ok ? 1 : 0)
        << " decode_fusion_candidate=" << (out.diagnostics.decode_fusion_candidate ? 1 : 0)
        << " decode_qkv_v2_candidate=" << (out.diagnostics.decode_qkv_v2_candidate ? 1 : 0);
    out.diagnostics.summary = oss.str();

    // Keep booleans as explicit transform gates driven by diagnostics.
    out.fuse_qkv = out.diagnostics.qkv_shape_ok;
    out.fuse_qkv_v2 = out.diagnostics.decode_qkv_v2_candidate;
    out.fuse_mlp = out.diagnostics.mlp_shape_ok;
    out.notes = "baseline phi3 plan (fusion readiness analyzed, qkv-v2 shape gate prepared)";

    error.clear();
    return true;
}
