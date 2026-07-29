// conv_transpose_2d.hlsl - 2D transposed convolution (one thread per dst element).
//
// src0: kernel [KW, KH, Cout, Cin]  (F16 or F32)
// src1: input  [IW, IH, Cin,  N]    (F32)
// dst:          [OW, OH, Cout, N]   (F32)
//   OW = (IW - 1) * stride + KW
//   OH = (IH - 1) * stride + KH
// op_params: [0] = stride (single value, used for both axes; padding = 0).
//
// For each output element (ow, oh, c_out, n) we gather over all (c_in, kh, kw)
// satisfying (oh - kh) % s == 0, (ow - kw) % s == 0, and 0 <= ih < IH,
// 0 <= iw < IW.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;     // ow
    uint i1 = rem % ne1; rem = rem / ne1;          // oh
    uint i2 = rem % ne2; uint i3 = rem / ne2;      // c_out, n

    int stride = asint(op0);
    if (stride <= 0) stride = 1;

    uint KW  = ne00;
    uint KH  = ne01;
    uint Cin = ne03;
    uint IW  = ne10;
    uint IH  = ne11;

    precise float acc = 0.0f;

    for (uint kh = 0; kh < KH; kh++) {
        int ih_num = (int)i1 - (int)kh;
        if (ih_num < 0) continue;
        if ((ih_num % stride) != 0) continue;
        int ih = ih_num / stride;
        if (ih >= (int)IH) continue;

        for (uint kw = 0; kw < KW; kw++) {
            int iw_num = (int)i0 - (int)kw;
            if (iw_num < 0) continue;
            if ((iw_num % stride) != 0) continue;
            int iw = iw_num / stride;
            if (iw >= (int)IW) continue;

            for (uint c_in = 0; c_in < Cin; c_in++) {
                uint k_off = src0_offset + kw * nb00 + kh * nb01 + i2 * nb02 + c_in * nb03;
                float w = load_auto(src0, k_off, src0_esize);

                uint i_off = src1_offset + (uint)iw * nb10 + (uint)ih * nb11 + c_in * nb12 + i3 * nb13;
                float v = load_auto(src1, i_off, src1_esize);

                acc += w * v;
            }
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
