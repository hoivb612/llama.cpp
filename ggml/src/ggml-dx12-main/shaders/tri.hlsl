// tri.hlsl - triangular mask: dst[i0,i1,...] = bipred(i0,i1) ? src0[...] : 0
// ggml_tri_type: UPPER_DIAG=0, UPPER=1, LOWER_DIAG=2, LOWER=3
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint ttype = op_param_uint(0);
    bool keep;
    switch (ttype) {
        case 0: keep = (i0 >= i1); break;  // UPPER_DIAG
        case 1: keep = (i0 >  i1); break;  // UPPER
        case 2: keep = (i0 <= i1); break;  // LOWER_DIAG
        default: keep = (i0 <  i1); break; // LOWER
    }

    uint off0 = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    float v = keep ? load_auto(src0, off0, src0_esize) : 0.0f;
    store_auto(dst, off_d, v, dst_esize);
}
