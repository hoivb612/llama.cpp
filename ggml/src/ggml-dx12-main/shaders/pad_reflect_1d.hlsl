// pad_reflect_1d.hlsl - Reflect-pad along ne0 by p0 (left) and p1 (right).
// dst index i0 maps to source j = i0 - p0, reflected at both edges:
//   j < 0        -> src[-j]
//   j >= ne00    -> src[2*(ne00-1) - j]
// op0 = p0, op1 = p1
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = flat_idx_2d_256(tid);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    int p0 = asint(op0);
    int j  = (int)i0 - p0;
    int n  = (int)ne00;
    if (j < 0)      j = -j;
    if (j >= n)     j = 2 * (n - 1) - j;

    uint off0  = offset_4d((uint)j, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, load_auto(src0, off0, src0_esize), dst_esize);
}
