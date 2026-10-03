#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint idx = flat_idx_2d_256(tid);
    if (idx >= ne0 * ne1) return;

    const uint i0 = idx % ne0;
    const uint token = idx / ne0;
    const bool gated = op1 != 0u;
    float sum = 0.0f;
    for (uint h = 0; h < ne01; ++h) {
        const float x = load_f32(src0, src0_offset + i0 * nb00 + h * nb01 + token * nb02);
        const uint offset = gated ? i0 * nb10 + h * nb11 + token * nb12 : h * nb10 + token * nb11;
        float weight = load_f32(src1, src1_offset + offset);
        if (gated) weight = 1.0f / (1.0f + exp(-weight));
        sum += x * weight;
    }
    store_f32(dst, dst_offset + i0 * nb0 + token * nb1, asfloat(op0) * sum);
}
