#include "ggml_common.hlsli"

// One 16-lane block per token, with lane = destination + 4 * source.
// op0: epsilon; op1: iterations; op2: base stride. src2 includes its tensor offset.
float hc_sum(float value, uint first, uint stride, float initial) {
    float sum = initial;
    [unroll] for (uint i = 0; i < 4; ++i) {
        sum += WaveReadLaneAt(value, first + i * stride);
    }
    return sum;
}

WAVE_SIZE_ATTR
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint idx = flat_idx_2d_256(tid);
    const uint token = idx / 16u;
    const uint idst = idx & 3u;
    const uint isrc = (idx & 15u) / 4u;
    const uint block = WaveGetLaneIndex() & ~15u;
    const uint row = block + 4u * isrc;
    const uint col = block + idst;
    const bool valid = token < ne2;
    const float eps = asfloat(op0);

    float value = 0.0f;
    if (valid) {
        const uint mix = 8u + (idx & 15u);
        value = load_f32(src0, src0_offset + mix * nb00 + token * nb01) *
                load_f32(src1, src1_offset + 2u * nb10) + load_f32(src2, mix * op2);
    }
    float vmax = WaveReadLaneAt(value, row);
    [unroll] for (uint h = 1; h < 4; ++h) {
        vmax = max(vmax, WaveReadLaneAt(value, row + h));
    }
    value = exp(value - vmax);
    value = value * (1.0f / hc_sum(value, row, 1u, 0.0f)) + eps;
    value *= 1.0f / hc_sum(value, col, 4u, eps);
    for (uint i = 1; i < op1; ++i) {
        value *= 1.0f / hc_sum(value, row, 1u, eps);
        value *= 1.0f / hc_sum(value, col, 4u, eps);
    }
    if (valid) {
        store_f32(dst, dst_offset + idst * nb0 + isrc * nb1 + token * nb2, value);
    }
}
