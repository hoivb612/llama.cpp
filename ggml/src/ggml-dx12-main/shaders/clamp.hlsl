// clamp.hlsl - Clamp: dst = clamp(src0, min_val, max_val)
// min_val in op_param_uint(0), max_val in op_param_uint(1)
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

    float min_val = op_param_f32(0);
    float max_val = op_param_f32(1);
    float val = load_auto(src0, off0, src0_esize);
    store_auto(dst, off_d, clamp(val, min_val, max_val), dst_esize);
}
