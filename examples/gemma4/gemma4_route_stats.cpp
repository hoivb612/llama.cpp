// Gemma-4 MoE routing telemetry (read-only). See gemma4_route_stats.h.

#include "gemma4_route_stats.h"

#include <algorithm>
#include <cstdio>

namespace gemma4 {

void RouteStats::ensure(int il, int n_expert) {
    if (n_expert > n_expert_) n_expert_ = n_expert;
    if (il + 1 > n_layer_) {
        const int nl = il + 1;
        for (Phase & p : phase_) {
            p.hist.resize(nl);
            p.total_sel.resize(nl, 0);
            p.overlap_sum.resize(nl, 0);
            p.overlap_cnt.resize(nl, 0);
            p.prev_mask.resize(nl);
            p.have_prev.resize(nl, 0);
        }
        n_layer_ = nl;
    }
    for (Phase & p : phase_) {
        if ((int) p.hist[il].size() < n_expert_)   p.hist[il].resize(n_expert_, 0);
        if ((int) p.prev_mask[il].size() < n_expert_) p.prev_mask[il].resize(n_expert_, 0);
    }
}

void RouteStats::record(int il, int n_expert, const int * sel, int k, bool is_decode) {
    if (il < 0 || k <= 0) return;
    ensure(il, n_expert);
    if (k > n_expert_used_) n_expert_used_ = k;

    Phase & p = phase_[is_decode ? 1 : 0];
    std::vector<uint64_t> & H  = p.hist[il];
    std::vector<uint8_t>  & pm = p.prev_mask[il];

    // histogram + overlap vs the previous token's selection at this layer
    uint64_t inter = 0;
    for (int j = 0; j < k; ++j) {
        const int e = sel[j];
        if (e < 0 || e >= n_expert_) continue;
        H[e]++;
        if (p.have_prev[il] && pm[e]) inter++;
    }
    p.total_sel[il] += (uint64_t) k;
    if (p.have_prev[il]) {
        p.overlap_sum[il] += inter;
        p.overlap_cnt[il] += 1;
    }

    // install this token's set as the new "previous"
    std::fill(pm.begin(), pm.end(), (uint8_t) 0);
    for (int j = 0; j < k; ++j) {
        const int e = sel[j];
        if (e >= 0 && e < n_expert_) pm[e] = 1;
    }
    p.have_prev[il] = 1;
}

void RouteStats::log_phase(const char * tag, const char * phase_name,
                           const Phase & p) const {
    // Skip a phase with no samples.
    uint64_t any = 0;
    for (int il = 0; il < n_layer_; ++il) any += p.total_sel[il];
    if (any == 0) return;

    const int k = n_expert_used_ > 0 ? n_expert_used_ : 1;

    std::fprintf(stderr,
        "gemma4 RouteStats[%s/%s]: n_layer=%d n_expert=%d n_expert_used=%d\n",
        tag ? tag : "", phase_name, n_layer_, n_expert_, k);
    std::fprintf(stderr,
        "  layer   sel  distinct  topK_conc%%  overlap/%d  overlap%%\n", k);

    // Aggregates weighted across all layers.
    double   agg_conc_num   = 0.0;  // sum over layers of topK-selection count
    uint64_t agg_sel        = 0;
    double   agg_overlap    = 0.0;  // sum of mean overlaps (per pair)
    uint64_t agg_overlap_cnt = 0;
    uint64_t agg_overlap_sum = 0;
    long     agg_distinct   = 0;

    std::vector<uint64_t> counts;
    for (int il = 0; il < n_layer_; ++il) {
        const uint64_t sel = p.total_sel[il];
        if (sel == 0) continue;

        counts.assign(p.hist[il].begin(), p.hist[il].end());
        int distinct = 0;
        for (uint64_t c : counts) if (c) distinct++;

        // top-k concentration: fraction of all selections captured by the k
        // most-used experts (k = n_expert_used). 100% => a fixed hot set.
        std::partial_sort(counts.begin(),
                          counts.begin() + std::min((size_t) k, counts.size()),
                          counts.end(), std::greater<uint64_t>());
        uint64_t topk = 0;
        for (int j = 0; j < k && j < (int) counts.size(); ++j) topk += counts[j];
        const double conc = 100.0 * (double) topk / (double) sel;

        const double ov_abs = p.overlap_cnt[il]
            ? (double) p.overlap_sum[il] / (double) p.overlap_cnt[il] : 0.0;
        const double ov_pct = 100.0 * ov_abs / (double) k;

        std::fprintf(stderr,
            "  %5d %6llu  %7d   %8.1f   %8.2f   %7.1f\n",
            il, (unsigned long long) sel, distinct, conc, ov_abs, ov_pct);

        agg_conc_num    += (double) topk;
        agg_sel         += sel;
        agg_overlap     += ov_abs;
        agg_overlap_cnt += 1;
        agg_overlap_sum += p.overlap_sum[il];
        agg_distinct    += distinct;
    }

    const double agg_conc = agg_sel ? 100.0 * agg_conc_num / (double) agg_sel : 0.0;
    const double agg_ov   = agg_overlap_cnt ? agg_overlap / (double) agg_overlap_cnt : 0.0;
    const double agg_ovp  = 100.0 * agg_ov / (double) k;
    const double avg_dist = agg_overlap_cnt ? (double) agg_distinct / (double) agg_overlap_cnt : 0.0;

    std::fprintf(stderr,
        "  ALL   %6llu  %7.1f   %8.1f   %8.2f   %7.1f\n",
        (unsigned long long) agg_sel, avg_dist, agg_conc, agg_ov, agg_ovp);
}

void RouteStats::log(const char * tag) const {
    if (n_layer_ == 0) return;
    log_phase(tag, "prefill", phase_[0]);
    log_phase(tag, "decode",  phase_[1]);
}

// ---- global singleton + toggle --------------------------------------------

namespace {
bool        g_route_stats_on = false;
RouteStats  g_route_stats;
}

void set_route_stats_enabled(bool on) { g_route_stats_on = on; }
bool route_stats_enabled()            { return g_route_stats_on; }
RouteStats & route_stats()            { return g_route_stats; }

void route_stats_record(int il, int n_expert, const int * sel, int k, bool is_decode) {
    if (!g_route_stats_on) return;
    g_route_stats.record(il, n_expert, sel, k, is_decode);
}

void route_stats_log(const char * tag) {
    if (!g_route_stats_on) return;
    if (!g_route_stats.has_data()) return;
    g_route_stats.log(tag);
}

} // namespace gemma4
