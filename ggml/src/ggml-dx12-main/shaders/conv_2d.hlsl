// conv_2d.hlsl - 2D convolution
// src0: kernel [KW, KH, IC, OC]
// src1: input  [IW, IH, IC, batch]
// dst:  output [OW, OH, OC, batch]
// op_params: [0]=stride_x [1]=stride_y [2]=pad_x [3]=pad_y [4]=dilation_x [5]=dilation_y
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    // dst indices: i0=OW, i1=OH, i2=OC, i3=batch
    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    int stride_x   = asint(op0);
    int stride_y   = asint(op1);
    int pad_x      = asint(op2);
    int pad_y      = asint(op3);
    int dilation_x = asint(op4);
    int dilation_y = asint(op5);

    // Kernel dimensions from src0
    uint KW = ne00;  // kernel width
    uint KH = ne01;  // kernel height
    uint IC = ne02;  // input channels

    // Input dimensions from src1
    uint IW = ne10;  // input width
    uint IH = ne11;  // input height

    precise float acc = 0.0f;

    for (uint ic = 0; ic < IC; ic++) {
        for (uint ky = 0; ky < KH; ky++) {
            for (uint kx = 0; kx < KW; kx++) {
                int sy = (int)i1 * stride_y + (int)ky * dilation_y - pad_y;
                int sx = (int)i0 * stride_x + (int)kx * dilation_x - pad_x;

                if (sx >= 0 && sx < (int)IW && sy >= 0 && sy < (int)IH) {
                    // Kernel weight: src0[kx, ky, ic, oc]
                    uint k_off = src0_offset + kx * nb00 + ky * nb01 + ic * nb02 + i2 * nb03;
                    float w = load_auto(src0, k_off, src0_esize);

                    // Input value: src1[sx, sy, ic, batch]
                    uint i_off = src1_offset + (uint)sx * nb10 + (uint)sy * nb11 + ic * nb12 + i3 * nb13;
                    float v = load_auto(src1, i_off, src1_esize);

                    acc += w * v;
                }
            }
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
