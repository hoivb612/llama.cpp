// roll.hlsl - Cyclic shift along each dimension (numpy.roll equivalent)
//
// op_params[0..3] = int32 shifts (s0, s1, s2, s3) for ne0..ne3
// dst[i0,i1,i2,i3] = src0[wrap(i0-s0,ne00), wrap(i1-s1,ne01),
//                         wrap(i2-s2,ne02), wrap(i3-s3,ne03)]
//
// dst shape == src0 shape. F32 only.
#include "ggml_common.hlsli"

uint wrap_idx(int i, uint n) {
    int ni = (int)n;
    if (i < 0)   return (uint)(i + ni);
    if (i >= ni) return (uint)(i - ni);
    return (uint)i;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    int s0 = asint(op0);
    int s1 = asint(op1);
    int s2 = asint(op2);
    int s3 = asint(op3);

    uint i00 = wrap_idx((int)i0 - s0, ne00);
    uint i01 = wrap_idx((int)i1 - s1, ne01);
    uint i02 = wrap_idx((int)i2 - s2, ne02);
    uint i03 = wrap_idx((int)i3 - s3, ne03);

    uint off_s = offset_4d(i00, i01, i02, i03, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0,  i1,  i2,  i3,  nb0,  nb1,  nb2,  nb3,  dst_offset);

    dst.Store(off_d, src0.Load(off_s));
}
