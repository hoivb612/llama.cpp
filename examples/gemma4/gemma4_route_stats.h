#pragma once

// Gemma-4 MoE routing telemetry (read-only).
//
// Characterizes the router's expert-selection behavior so we can decide
// whether a smarter ExpertStore retention policy (frequency-aware pinning,
// per-layer partitioning) will pay off at sub-working-set budgets, or whether
// streaming there is fundamentally bandwidth-bound.
//
// Two questions, two metrics, tracked per layer and split by phase
// (prefill = multi-token step, decode = single-token step):
//
//   1. GLOBAL SKEW -- per-expert selection histogram. If a hot minority of
//      experts dominates, pinning the top-K per layer converts a small budget
//      into a high hit rate. If usage is flat, only budget/bandwidth helps.
//
//   2. TEMPORAL LOCALITY -- overlap between consecutive tokens' selected
//      expert sets at the same layer. High overlap means a sweep-aware
//      retention policy can keep the reused experts resident and break the
//      LRU cyclic-sweep pathology; low overlap means each token genuinely
//      needs fresh experts.
//
// This is pure instrumentation: record() only counts, never changes routing
// or streaming behavior. It is a no-op unless enabled via the CLI flag
// (--gemma4-moe-route-stats). record() is called from moe_ffn on the single
// consumer thread (the MatmulCtx invariant), so no locking is needed.

#include <cstdint>
#include <vector>

namespace gemma4 {

class RouteStats {
public:
    // Record one token's top-k expert selection at layer `il`.
    // `n_expert` is the router width (experts available); `sel[0..k)` are the
    // selected expert indices for this token. `is_decode` buckets the sample
    // into the single-token (decode) vs multi-token (prefill) phase so the
    // decode-locality signal is not polluted by prefill's many columns.
    void record(int il, int n_expert, const int * sel, int k, bool is_decode);

    // Emit a human-readable per-layer + aggregate telemetry block to stderr.
    void log(const char * tag) const;

    // True once at least one sample has been recorded.
    bool has_data() const { return n_layer_ > 0; }

private:
    // Per (phase, layer) accumulators. Phase index: 0 = prefill, 1 = decode.
    struct Phase {
        std::vector<std::vector<uint64_t>> hist;        // [layer][expert] select count
        std::vector<uint64_t>              total_sel;   // [layer] sum of selections
        std::vector<uint64_t>              overlap_sum; // [layer] sum |cur int prev|
        std::vector<uint64_t>              overlap_cnt; // [layer] # consecutive pairs
        std::vector<std::vector<uint8_t>>  prev_mask;   // [layer][expert] previous set
        std::vector<uint8_t>               have_prev;   // [layer] prev_mask valid
    };

    void ensure(int il, int n_expert);
    void log_phase(const char * tag, const char * phase_name, const Phase & p) const;

    Phase phase_[2];
    int   n_layer_        = 0;
    int   n_expert_       = 0;
    int   n_expert_used_  = 0;
};

// Global accessor + enable toggle (matches the set_expert_store / set_matmul_*
// convention used elsewhere in this example).
void        set_route_stats_enabled(bool on);
bool        route_stats_enabled();
RouteStats & route_stats();

// Record helper used by moe_ffn; no-op when disabled.
void route_stats_record(int il, int n_expert, const int * sel, int k, bool is_decode);

// Emit the telemetry block if enabled and any samples were recorded.
void route_stats_log(const char * tag);

} // namespace gemma4
