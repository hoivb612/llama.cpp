// diag.hlsl - dst[i0,i1,i2,i3] = (i0 == i1) ? src0[i1,0,i2,i3] : 0
// src0 has ne01 = 1; dst has ne0 = ne1 = ne00.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    float v = 0.0f;
    if (i0 == i1) {
        uint off0 = offset_4d(i1, 0, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        v = load_auto(src0, off0, src0_esize);
    }
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, v, dst_esize);
}
