// im2col.hlsl - Image to column transform for convolution
// src0: kernel [KW, KH, IC, OC] (only shape used, not data)
// src1: input  [IW, IH, IC, N]
// dst:  result [IC*KH*KW, OW, OH, N]
// op_params: [0]=s0 [1]=s1 [2]=p0 [3]=p1 [4]=d0 [5]=d1 [6]=is_2D
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    // dst layout: [IC*KH*KW, OW, OH, N]
    uint i0 = idx % ne0; uint rem = idx / ne0;  // i0 = ic*KH*KW + ikh*KW + ikw
    uint i1 = rem % ne1; rem = rem / ne1;        // i1 = OW index
    uint i2 = rem % ne2; uint i3 = rem / ne2;    // i2 = OH index, i3 = batch

    int s0 = asint(op0);  // stride width
    int s1 = asint(op1);  // stride height
    int p0 = asint(op2);  // pad width
    int p1 = asint(op3);  // pad height
    int d0 = asint(op4);  // dilation width
    int d1 = asint(op5);  // dilation height
    int is_2D = asint(op6);

    uint KW = ne00;
    uint KH = (is_2D != 0) ? ne01 : 1;
    uint IC = (is_2D != 0) ? ne02 : ne01;

    uint IW = ne10;
    uint IH = (is_2D != 0) ? ne11 : 1;

    // Decompose i0 into (ic, ikh, ikw)
    uint ikw = i0 % KW;
    uint ikh = (i0 / KW) % KH;
    uint iic = i0 / (KW * KH);

    uint iow = i1;  // output width index
    uint ioh = i2;  // output height index
    uint in_ = i3;  // batch index

    int iiw = (int)iow * s0 + (int)ikw * d0 - p0;
    int iih = (int)ioh * s1 + (int)ikh * d1 - p1;

    uint off_d = dst_offset + idx * nb0;  // dst stride: nb0 = 2 for F16, 4 for F32

    if (iih < 0 || iih >= (int)IH || iiw < 0 || iiw >= (int)IW) {
        store_auto(dst, off_d, 0.0f, dst_esize);
    } else {
        // src1 access: src1[iiw, iih, iic, in_]
        uint src_off;
        if (is_2D != 0) {
            src_off = src1_offset + (uint)iiw * nb10 + (uint)iih * nb11 + iic * nb12 + in_ * nb13;
        } else {
            src_off = src1_offset + (uint)iiw * nb10 + iic * nb11 + in_ * nb12;
        }
        float val = load_auto(src1, src_off, src1_esize);
        store_auto(dst, off_d, val, dst_esize);
    }
}
