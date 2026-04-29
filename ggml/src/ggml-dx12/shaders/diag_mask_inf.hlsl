// diag_mask_inf.hlsl - Diagonal mask with -inf
// Masks elements above the diagonal + n_past with -infinity
// n_past is stored in op_param_uint(0)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    int n_past = asint(op_param_uint(0));

    uint off0  = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    float val = load_auto(src0, off0, src0_esize);

    // Mask: if column > row + n_past, set to -inf
    if ((int)i0 > (int)i1 + n_past) {
        val = asfloat(0xFF800000); // -infinity
    }

    store_auto(dst, off_d, val, dst_esize);
}
