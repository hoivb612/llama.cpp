// gelu.hlsl - GELU activation: dst = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#include "ggml_common.hlsli"

static const float SQRT_2_OVER_PI = 0.7978845608028654f;
static const float GELU_COEFF = 0.044715f;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint off0  = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    float x = load_auto(src0, off0, src0_esize);
    float arg = SQRT_2_OVER_PI * x * (1.0f + GELU_COEFF * x * x);
    arg = clamp(arg, -20.0f, 20.0f); // prevent tanh overflow → NaN
    float val = 0.5f * x * (1.0f + tanh(arg));
    store_auto(dst, off_d, val, dst_esize);
}
