#pragma once

// G4.3 — Cached prefill for the Gemma-4 hand forward.
//
// Mirrors examples/phi3/phi3_fused_graph (A5.5) for Gemma-4's NetworkState
// instead of llama_context. The key adaptations:
//
//   * Per-layer K/V dims vary in Gemma-4 (head_dim 256 for SWA layers,
//     512 for full-attn layers) so each layer slab encodes its own
//     (n_head_kv, head_dim).
//
//   * Shared-KV layers (LayerF32.kv_reuse_il >= 0) have empty K_cache /
//     V_cache slots. We do NOT serialise them; on load they remain empty
//     and network_step naturally reads through their kv_reuse_il source.
//
//   * NetworkState stores F32 (not F16), so file size is 4x what an
//     equivalent Phi-3 cache would be. Acceptable for offline cache;
//     no compression for v1.
//
//   * Greedy-only seed: the cache stores first_gen_token = argmax of the
//     final logits (already post-softcap inside network_step). The load
//     path uses this to bootstrap gen without re-running prefill.
//
// File format ("G4KV0001"): 64-byte fixed header + pos_all + per-owning
// layer slabs (see gemma4_kvcache.cpp for the exact byte layout). Header
// embeds two model fingerprints:
//
//   topology_hash  : FNV-1a over packed hparams + per-layer dim/rope
//                    fields. Tight; a different fine-tune with the same
//                    topology will agree.
//   weight_hash    : FNV-1a over output_norm.weight raw bytes. Cheap
//                    distinguishing signature across fine-tunes.
//
// Loader cross-checks topology_hash strictly (mismatch == hard fail) and
// weight_hash softly by default (warn-only; promotable to hard fail via
// strict_model_match). prompt_hash mismatch is warn-only — a cache may
// legally be reused across different continuations of the same context.

#include "gemma4_forward.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;
typedef int32_t llama_token;

namespace gemma4 {

// FNV-1a 64-bit over the raw bytes of the prompt token ids.
// Returns 0 for an empty vector so callers can treat 0 as "unknown".
uint64_t kv_compute_prompt_hash(const std::vector<llama_token> & prompt_tokens);

// FNV-1a 64-bit over output_norm.weight raw bytes. Used as a cheap
// per-model fingerprint that distinguishes fine-tunes with matching
// topology. Returns 0 on failure (tensor missing / data null); callers
// should treat 0 as "unknown" and not enforce strict equality.
uint64_t kv_compute_model_weight_hash(const llama_model * model);

// FNV-1a 64-bit over a packed blob of hparams + per-layer (head_dim,
// is_swa, kv_reuse_il, rope_dim, rope_base). Tight model topology
// fingerprint that catches "wrong model variant" mismatches before any
// per-layer slab dim check.
uint64_t kv_compute_topology_hash(const ModelF32 & m);

// Serialise the post-prefill NetworkState + first_gen_token + fingerprints
// to `path`. Atomic via "<path>.tmp" + std::filesystem::rename so a partial
// write never leaves a half-baked cache on disk.
//
// Requires:
//   * st.n_past > 0 (prefill must have run)
//   * first_gen_token >= 0  (greedy-only contract)
//   * st.K_cache[il] / st.V_cache[il] non-empty for every owning layer
//     (m.layers[il].kv_reuse_il == -1)
//   * st.pos_all.size() == st.n_past with contiguous positions 0..n_past-1
//
// On success returns true. On any failure returns false and writes a
// descriptive message to `error`; no partial file is left behind.
bool save_kv_to_disk(const NetworkState & st,
                     const ModelF32   & m,
                     int                first_gen_token,
                     uint64_t           prompt_hash,
                     uint64_t           weight_hash,
                     const std::string& path,
                     std::string      & error);

// Read a cache file produced by save_kv_to_disk, validate the header
// + per-slab dims against `m`, and populate `st`. Internally calls
// network_state_reserve with cap = n_tokens + continuation_capacity so
// the caller does NOT need to pre-reserve.
//
// Validations:
//   strict (hard fail with clear error):
//     * magic, hdr_size, version, kv_type=F32
//     * topology_hash matches kv_compute_topology_hash(m)
//     * n_layer / per-slab (il, n_head_kv, head_dim) match m
//     * exactly one slab per owning layer (kv_reuse_il == -1)
//     * n_tokens in (0, cap_seq], first_gen_token in [0, n_vocab)
//     * pos_all is contiguous 0..n_tokens-1
//     * file size matches expected payload size
//   soft (warn unless strict_model_match):
//     * weight_hash mismatch when both file and current model report
//       non-zero values
//   warn only (never fatal):
//     * prompt_hash mismatch
bool load_kv_from_disk(NetworkState     & st,
                       const ModelF32   & m,
                       const std::string& path,
                       int                continuation_capacity,
                       uint64_t           expected_prompt_hash,        // 0 to skip
                       uint64_t           expected_weight_hash,        // 0 to skip
                       bool               strict_model_match,
                       int              & out_n_tokens,
                       int              & out_first_gen_token,
                       std::string      & error);

} // namespace gemma4
