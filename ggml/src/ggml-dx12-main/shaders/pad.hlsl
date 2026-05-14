// pad.hlsl - Pad tensor with zeros (non-circular)
// op_params: [0]=lp0 [1]=rp0 [2]=lp1 [3]=rp1 [4]=lp2 [5]=rp2 [6]=lp3 [7]=rp3
// dst is contiguous F32
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    int lp0 = asint(op0); int rp0 = asint(op1);
    int lp1 = asint(op2); int rp1 = asint(op3);
    int lp2 = asint(op4); int rp2 = asint(op5);
    int lp3 = asint(op6); int rp3 = asint(op7);

    // dst is contiguous — use flat index
    uint off_d = dst_offset + idx * 4;

    if ((int)i0 >= lp0 && (int)i0 < (int)ne0 - rp0 &&
        (int)i1 >= lp1 && (int)i1 < (int)ne1 - rp1 &&
        (int)i2 >= lp2 && (int)i2 < (int)ne2 - rp2 &&
        (int)i3 >= lp3 && (int)i3 < (int)ne3 - rp3) {
        uint si0 = (uint)((int)i0 - lp0);
        uint si1 = (uint)((int)i1 - lp1);
        uint si2 = (uint)((int)i2 - lp2);
        uint si3 = (uint)((int)i3 - lp3);
        uint off0 = offset_4d(si0, si1, si2, si3, nb00, nb01, nb02, nb03, src0_offset);
        float val = load_auto(src0, off0, src0_esize);
        dst.Store(off_d, asuint(val));
    } else {
        dst.Store(off_d, asuint(0.0f));
    }
}
