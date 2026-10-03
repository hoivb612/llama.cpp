// CONV_2D_DW: depthwise 2D convolution, per-output-element gather (F32-only).
// src0 (kernel): ne = [KW, KH, 1, channels]
// src1 (input):  ne = [W,  H,  channels, batch]
// dst:           ne = [dst_w, dst_h, channels, batch]
// op_params[0..5] = stride_x, stride_y, pad_x, pad_y, dilation_x, dilation_y
//
// Layout-agnostic: src/dst nb's encode WHCN or CWHN; this shader addresses
// by nb so both layouts work without specialization.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint flat = flat_idx_2d(gid, gtid.x);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (flat >= total) return;

    uint dst_x = flat % ne0; uint rem = flat / ne0;
    uint dst_y = rem  % ne1; rem = rem / ne1;
    uint c     = rem  % ne2;
    uint n     = rem  / ne2;

    int stride_x   = asint(op0);
    int stride_y   = asint(op1);
    int pad_x      = asint(op2);
    int pad_y      = asint(op3);
    int dilation_x = asint(op4);
    int dilation_y = asint(op5);

    uint KW = ne00;
    uint KH = ne01;
    uint W  = ne10;
    uint H  = ne11;

    float sum = 0.0f;
    for (uint ky = 0; ky < KH; ++ky) {
        int sy = (int)dst_y * stride_y + (int)ky * dilation_y - pad_y;
        if (sy < 0 || sy >= (int)H) continue;
        for (uint kx = 0; kx < KW; ++kx) {
            int sx = (int)dst_x * stride_x + (int)kx * dilation_x - pad_x;
            if (sx < 0 || sx >= (int)W) continue;
            uint k_off = src0_offset + kx * nb00 + ky * nb01 + c * nb03;
            uint s_off = src1_offset + (uint)sx * nb10 + (uint)sy * nb11 + c * nb12 + n * nb13;
            float k = load_auto(src0, k_off, src0_esize);
            float v = load_auto(src1, s_off, src1_esize);
            sum += k * v;
        }
    }

    uint d_off = dst_offset + dst_x * nb0 + dst_y * nb1 + c * nb2 + n * nb3;
    store_auto(dst, d_off, sum, dst_esize);
}
