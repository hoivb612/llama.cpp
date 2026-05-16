// scale.hlsl - Scale tensor: dst = src0 * scale + bias
// op_params[0] = scale (float), op_params[1] = bias (float)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint off0  = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    float scale = op_param_f32(0);
    float bias  = op_param_f32(1);
    float val = load_auto(src0, off0, src0_esize);
    store_auto(dst, off_d, val * scale + bias, dst_esize);
}
