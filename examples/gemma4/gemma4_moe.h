// Gemma-4 MoE (26B-A4B) support for the standalone hand-coded forward.
//
// P0 goal: run the MoE FFN with every expert resident, matching the upstream
// llama build_moe_ffn numerics, as the correctness baseline before per-expert
// streaming (P1) caps the resident-expert memory.
//
// This header currently exposes the per-expert view alignment self-test used
// to gate the expert-addressing primitive (matmul_expert_qf32). The MoE
// forward (moe_ffn) is added in a follow-up step.

#pragma once

#include <string>

struct llama_model;
struct ggml_tensor;

namespace gemma4 {

struct Weights;
struct LayerWeights;
struct MatmulCtx;

// Everything moe_ffn needs for one MoE layer, decoupled from the resolver
// structs so it can be populated either from LayerWeights (dump/self-test
// path) or from LayerF32 (the network forward path). Tensor pointers are
// read directly (require -ngl 0); ffn_norm is a plain F32 pointer since the
// forward path already keeps that norm as a std::vector<float>.
struct MoeInputs {
    int il            = -1;  // layer index (for routing telemetry; -1 = unknown)
    int n_embd        = 0;
    int n_ff          = 0;   // shared/dense FFN width
    int n_ff_exp      = 0;   // per-expert FFN width
    int n_expert      = 0;
    int n_expert_used = 0;

    const float       * ffn_norm         = nullptr;  // [n_embd] F32 (shared MLP pre-norm)
    const ggml_tensor * ffn_gate         = nullptr;  // shared MLP (quant)
    const ggml_tensor * ffn_up           = nullptr;
    const ggml_tensor * ffn_down         = nullptr;

    const ggml_tensor * ffn_post_norm_1  = nullptr;  // [n_embd] F32
    const ggml_tensor * ffn_pre_norm_2   = nullptr;  // [n_embd] F32
    const ggml_tensor * ffn_post_norm_2  = nullptr;  // [n_embd] F32
    const ggml_tensor * ffn_gate_inp     = nullptr;  // router
    const ggml_tensor * ffn_gate_inp_s   = nullptr;  // [n_embd] F32

    const ggml_tensor * ffn_gate_up_exps = nullptr;  // merged (one of merged / separate)
    const ggml_tensor * ffn_gate_exps    = nullptr;
    const ggml_tensor * ffn_up_exps      = nullptr;
    const ggml_tensor * ffn_down_exps    = nullptr;
    const ggml_tensor * ffn_down_exps_s  = nullptr;  // [n_expert] F32 (nullable)
};

// Compute the Gemma-4 MoE FFN block for one layer, matching the upstream
// src/models/gemma4.cpp graph (build_ffn shared MLP + build_moe_ffn expert
// path), all experts resident. Given the post-attention residual
//   attn_out2 = post_attn_norm(attn @ ...) + hidden_in     [n_embd, n_new]
// this writes
//   out = cur_mlp + cur_moe                                 [n_embd, n_new]
// where
//   cur_mlp = post_ffw_norm_1( build_ffn(GELU,PAR, ffn_norm(attn_out2)) )
//   cur_moe = post_ffw_norm_2( moe( pre_ffw_norm_2(attn_out2), router(attn_out2) ) )
// The caller then applies the shared post_ffw_norm (FFN_POST_NORM) and the
// residual (+ attn_out2), exactly as the dense path does with its ff_out.
//
// Experts stay quantized: shared FFN and router go through matmul_qf32, the
// per-expert gate_up/down go through matmul_expert_qf32 (2D expert views).
// Requires -ngl 0 so weight data (including the F32 norm/scale tensors read
// directly here) is host accessible, and an initialized MatmulCtx.
bool moe_ffn(MatmulCtx & mm, const MoeInputs & in,
             const float * attn_out2, float * out,
             int n_new, float eps, std::string & error);

// Validate the per-expert 2D view produced by matmul_expert_qf32 against a
// contiguous single-expert copy of the same block. For a MoE layer il, for a
// handful of expert indices and both stacked expert tensors present
// (ffn_gate_up_exps / ffn_gate_exps+ffn_up_exps and ffn_down_exps), this
// computes W[:,:,e] @ x two ways:
//   1. ggml_view_2d over the expert's sub-block (the streaming primitive)
//   2. a freshly-allocated contiguous copy of that block
// and asserts the outputs are bit-identical. A mismatch means the view offset
// or row stride does not land on a K-quant block boundary.
//
// Requires the model to be loaded on CPU (-ngl 0) so expert->data is host
// accessible. n_cols is the number of activation columns to test with.
bool moe_expert_view_selftest(const llama_model * model, const Weights & w,
                              int il, int n_cols, int n_threads,
                              std::string & error);

} // namespace gemma4
