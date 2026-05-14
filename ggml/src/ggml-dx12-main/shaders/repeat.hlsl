// repeat.hlsl - Repeat tensor: dst = repeat(src0) to match dst shape
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint s0 = i0 % ne00;
    uint s1 = i1 % ne01;
    uint s2 = i2 % ne02;
    uint s3 = i3 % ne03;

    uint off0  = offset_4d(s0, s1, s2, s3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    store_auto(dst, off_d, load_auto(src0, off0, src0_esize), dst_esize);
}
