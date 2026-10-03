// pool_2d.hlsl - 2D max/average pooling
// op_params: [0]=pool_type (0=max, 1=avg) [1]=k0 (kernel_w) [2]=k1 (kernel_h) [3]=s0 (stride_w) [4]=s1 (stride_h) [5]=p0 (pad_w) [6]=p1 (pad_h)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint pool_type = op0;
    uint kw = op1; uint kh = op2;
    uint sw = op3; uint sh = op4;
    uint pw = op5; uint ph = op6;

    uint iw = ne00; uint ih = ne01;

    float result;
    if (pool_type == 0) {
        // Max pooling
        result = -3.402823466e+38f;
        for (uint ky = 0; ky < kh; ky++) {
            for (uint kx = 0; kx < kw; kx++) {
                int iy = (int)(i1 * sh + ky) - (int)ph;
                int ix = (int)(i0 * sw + kx) - (int)pw;
                if (ix >= 0 && ix < (int)iw && iy >= 0 && iy < (int)ih) {
                    uint off0 = src0_offset + (uint)ix * nb00 + (uint)iy * nb01 + i2 * nb02 + i3 * nb03;
                    result = max(result, load_auto(src0, off0, src0_esize));
                }
            }
        }
    } else {
        // Average pooling.
        // NOTE: divide by full kernel area (kw*kh), not by count of in-bounds
        // pixels — this matches the CPU reference (out-of-bounds cells
        // contribute 0 to the sum but still count in the denominator).
        result = 0.0f;
        for (uint ky = 0; ky < kh; ky++) {
            for (uint kx = 0; kx < kw; kx++) {
                int iy = (int)(i1 * sh + ky) - (int)ph;
                int ix = (int)(i0 * sw + kx) - (int)pw;
                if (ix >= 0 && ix < (int)iw && iy >= 0 && iy < (int)ih) {
                    uint off0 = src0_offset + (uint)ix * nb00 + (uint)iy * nb01 + i2 * nb02 + i3 * nb03;
                    result += load_auto(src0, off0, src0_esize);
                }
            }
        }
        uint ka = kw * kh;
        if (ka > 0) result /= (float)ka;
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, result, dst_esize);
}
