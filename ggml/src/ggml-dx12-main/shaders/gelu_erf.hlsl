// gelu_erf.hlsl - GELU activation using error function: dst = 0.5 * x * (1 + erf(x / sqrt(2)))
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint off0 = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    float x = load_auto(src0, off0, src0_esize);
    // Abramowitz & Stegun erf approximation
    float a = abs(x * 0.7071067811865f);
    float p = 1.0f / (1.0f + 0.3275911f * a);
    float e = p * (0.254829592f + p * (-0.284496736f + p * (1.421413741f + p * (-1.453152027f + p * 1.061405429f))));
    float erf_val = sign(x) * (1.0f - e * exp(-a * a));
    store_auto(dst, off_d, 0.5f * x * (1.0f + erf_val), dst_esize);
}
