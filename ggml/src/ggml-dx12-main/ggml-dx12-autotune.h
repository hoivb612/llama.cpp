#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace ggml_dx12_autotune {

static constexpr int CACHE_VERSION = 12;
static constexpr uint32_t NEVER = std::numeric_limits<uint32_t>::max();

struct cache_identity {
    uint64_t driver = 0;
    uint32_t stamp = 0;
    uint32_t wave = 0;
    int32_t luid_high = 0;
    uint32_t luid_low = 0;

    bool operator==(const cache_identity & other) const {
        return driver == other.driver &&
               stamp == other.stamp &&
               wave == other.wave &&
               luid_high == other.luid_high &&
               luid_low == other.luid_low;
    }
};

struct cache_values {
    bool q4k_32 = false;
    bool q5k_32 = false;
    uint32_t f16_k_256 = NEVER;
    uint32_t bf16_k_256 = NEVER;
    uint32_t f32_k_256 = NEVER;
};

inline uint64_t median(std::vector<uint64_t> values) {
    if (values.empty()) {
        return std::numeric_limits<uint64_t>::max();
    }
    std::sort(values.begin(), values.end());
    const size_t hi = values.size() / 2;
    if (values.size() & 1) {
        return values[hi];
    }
    return values[hi - 1] + (values[hi] - values[hi - 1]) / 2;
}

inline uint32_t interpolate_crossover(
        uint32_t xa, uint64_t a256, uint64_t a32,
        uint32_t xb, uint64_t b256, uint64_t b32) {
    const double gap_a = (double) a32 - (double) a256;
    const double gap_b = (double) b32 - (double) b256;
    const double denom = gap_a - gap_b;
    if (denom == 0.0) {
        return xa + (xb - xa) / 2;
    }
    double t = gap_a / denom;
    t = std::max(0.0, std::min(1.0, t));
    return (uint32_t) ((double) xa + t * (double) (xb - xa) + 0.5);
}

struct threshold_sample {
    uint32_t x;
    uint64_t t256;
    uint64_t t32;
};

inline uint32_t select_256_threshold(const std::vector<threshold_sample> & samples) {
    if (samples.empty()) {
        return NEVER;
    }

    bool all_256 = true;
    bool all_32 = true;
    for (const auto & sample : samples) {
        all_256 = all_256 && sample.t256 < sample.t32;
        all_32 = all_32 && sample.t32 <= sample.t256;
    }
    if (all_256) {
        return 0;
    }
    if (all_32) {
        return NEVER;
    }

    for (size_t i = 1; i < samples.size(); ++i) {
        const bool prev_32 = samples[i - 1].t32 <= samples[i - 1].t256;
        const bool curr_256 = samples[i].t256 < samples[i].t32;
        if (prev_32 && curr_256) {
            for (size_t j = 0; j + 1 < i; ++j) {
                if (samples[j].t256 < samples[j].t32) {
                    return NEVER;
                }
            }
            for (size_t j = i + 1; j < samples.size(); ++j) {
                if (samples[j].t32 <= samples[j].t256) {
                    return NEVER;
                }
            }
            return interpolate_crossover(
                samples[i - 1].x, samples[i - 1].t256, samples[i - 1].t32,
                samples[i].x, samples[i].t256, samples[i].t32);
        }
    }
    return NEVER;
}

inline std::string serialize_cache(
        const cache_identity & identity,
        const cache_values & values) {
    char line[512];
    std::snprintf(
        line, sizeof(line),
        "v=%d driver=%llu stamp=%u wave=%u luid_hi=%d luid_lo=%u "
        "q4k_dp4a_32=%d q5k_dp4a_32=%d "
        "f16_mr_k_thresh=%u bf16_mr_k_thresh=%u f32_mr_k_thresh=%u\n",
        CACHE_VERSION,
        (unsigned long long) identity.driver,
        identity.stamp,
        identity.wave,
        identity.luid_high,
        identity.luid_low,
        values.q4k_32 ? 1 : 0,
        values.q5k_32 ? 1 : 0,
        values.f16_k_256,
        values.bf16_k_256,
        values.f32_k_256);
    return line;
}

inline bool parse_cache_line(
        const char * line,
        cache_identity & identity,
        cache_values & values) {
    int version = 0;
    unsigned long long driver = 0;
    unsigned stamp = 0;
    unsigned wave = 0;
    int luid_high = 0;
    unsigned luid_low = 0;
    int q4k = 0;
    int q5k = 0;
    unsigned f16 = NEVER;
    unsigned bf16 = NEVER;
    unsigned f32 = NEVER;
    const int parsed = std::sscanf(
        line,
        "v=%d driver=%llu stamp=%u wave=%u luid_hi=%d luid_lo=%u "
        "q4k_dp4a_32=%d q5k_dp4a_32=%d "
        "f16_mr_k_thresh=%u bf16_mr_k_thresh=%u f32_mr_k_thresh=%u",
        &version, &driver, &stamp, &wave, &luid_high, &luid_low,
        &q4k, &q5k, &f16, &bf16, &f32);
    identity = {
        (uint64_t) driver, stamp, wave, luid_high, luid_low
    };
    if (parsed != 11 || version != CACHE_VERSION) {
        return false;
    }
    values.q4k_32 = q4k != 0;
    values.q5k_32 = q5k != 0;
    values.f16_k_256 = f16;
    values.bf16_k_256 = bf16;
    values.f32_k_256 = f32;
    return true;
}

inline bool parse_cache(
        const char * line,
        const cache_identity & expected,
        cache_values & values) {
    cache_identity actual;
    return parse_cache_line(line, actual, values) && actual == expected;
}

} // namespace ggml_dx12_autotune
