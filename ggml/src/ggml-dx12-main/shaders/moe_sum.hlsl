#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint idx = flat_idx_2d(group_id, local_id);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0;
    uint rem = idx / ne0;
    uint i1 = rem % ne1;
    rem /= ne1;
    uint i2 = rem % ne2;
    uint i3 = rem / ne2;

    precise float sum = 0.0f;
    for (uint slot = 0; slot < op0; ++slot) {
        uint off = src0_offset + i0 * nb00 + slot * op1 +
                   i1 * op2 + i2 * op3 + i3 * op4;
        sum += asfloat(src0.Load(off));
    }

    uint dst_off = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    dst.Store(dst_off, asuint(sum));
}
