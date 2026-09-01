#pragma once

#include <cstddef>

// MoE expert-level streaming (Vulkan UMA weight-budget path).
//
// Phase A (milestone 2): expert-level streaming with a residency split. The big
// per-expert weight matrices are NEVER made resident (see the layer-window's
// GGML_LW_MOE_STREAM path); this custom op serves them per ubatch by gathering
// each needed expert's quantized slab directly from the model's mmap (or file)
// into a compact tensor, then running mul_mat_id over that compact tensor.
//
// Decode (n_tokens == 1): gather only the n_expert_used routed experts into a
// compact [ne0, ne1, n_expert_used] tensor in *selection order* (slot i = expert
// selected_experts[i]) and index with iota ids — the existing per-expert scale
// path (get_rows(scale, selected_experts), also in selection order) lines up
// unchanged. Prefill (n_tokens > 1): gather ALL n_expert experts into a compact
// [ne0, ne1, n_expert] tensor (static shape) and index with the real routed ids,
// giving a result bit-identical to the full expert path.

struct ggml_context;
struct ggml_tensor;

// True when GGML_LW_MOE_STREAM is set (env-gated, checked once).
bool llama_moe_stream_enabled();

// Aggregate committed memory of all Phase-A2 expert pools. Each pool is a
// VirtualAlloc'd (MEM_COMMIT) host block imported as a host-coherent Vulkan
// buffer, so on UMA this memory counts BOTH toward the process private/committed
// pages AND toward what the GPU can read. Returns total committed bytes across
// all pools; optional out-params report the pool count, the summed slot count,
// and the cumulative decode hit-rate (0..100). Safe to call any time (returns 0
// before any pool is created).
size_t llama_moe_stream_pool_bytes(int * n_pools, long long * total_slots, double * hit_rate);

// Build a mul_mat_id over the routed experts by gathering their slabs from the
// model's mmap/file into a compact per-call tensor, so the full expert weight
// matrix never needs to be resident.
//
//   Decode (n_tokens == 1): gather only the n_expert_used routed experts (in
//   selection order) and run mul_mat_id with iota ids.
//   Prefill (n_tokens > 1): gather ALL n_expert experts (static shape) and run
//   mul_mat_id with the real selected_experts ids — bit-identical to the full
//   expert path, at the cost of a full expert-tensor copy per call.
//
//   full_exps        full [ne0, ne1, n_expert] expert weight tensor (used only
//                    for shape/type/file-location lookup; its data is NOT read).
//   cur              the mul_mat_id "b" operand.
//   selected_experts routed expert ids [n_expert_used, n_tokens].
//   n_expert_used    routed experts per token.
//   n_expert         total experts (full-copy count for prefill).
//   n_tokens         tokens in this ubatch (1 => decode gather, >1 => prefill).
//   il               layer index (keys the persistent userdata + file lookup).
//   role             0=gate_up, 1=up, 2=gate, 3=down (keys persistent userdata).
//
// Returns nullptr if the expert tensor's file location is unknown (caller must
// fall back to the normal path).
ggml_tensor * llama_moe_streamed_mul_mat_id(
        ggml_context * ctx0,
        ggml_tensor  * full_exps,
        ggml_tensor  * cur,
        ggml_tensor  * selected_experts,
        long long      n_expert_used,
        long long      n_expert,
        long long      n_tokens,
        int            il,
        int            role);
