// cpy.hlsl - Copy tensor data: dst = src0
// Handles different strides/layouts, reshape, and F32↔F16 conversion.
// For CONT (make contiguous), src0 may have different ne than dst.
// We iterate over the flat element index, decompose into src0 coordinates
// using src0 dimensions, and into dst coordinates using dst dimensions.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    // Decompose flat index into DESTINATION coordinates for dst offset
    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    // Decompose SAME flat index into SOURCE coordinates for src0 offset
    // Source may have different shape (e.g., reshape/view)
    uint j0, j1, j2, j3;
    flat_to_4d(idx, ne00, ne01, ne02, j0, j1, j2, j3);
    uint off0 = offset_4d(j0, j1, j2, j3, nb00, nb01, nb02, nb03, src0_offset);

    float val = load_auto(src0, off0, src0_esize);
    store_auto(dst, off_d, val, dst_esize);
}
