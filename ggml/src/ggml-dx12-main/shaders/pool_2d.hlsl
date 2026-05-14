// pool_2d.hlsl - 2D max/average pooling
// op_params: [0]=kernel_w [1]=kernel_h [2]=stride_w [3]=stride_h [4]=pad_w [5]=pad_h [6]=pool_type (0=max, 1=avg)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint kw = op0; uint kh = op1;
    uint sw = op2; uint sh = op3;
    uint pw = op4; uint ph = op5;
    uint pool_type = op6;

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
        // Average pooling
        result = 0.0f;
        uint count = 0;
        for (uint ky = 0; ky < kh; ky++) {
            for (uint kx = 0; kx < kw; kx++) {
                int iy = (int)(i1 * sh + ky) - (int)ph;
                int ix = (int)(i0 * sw + kx) - (int)pw;
                if (ix >= 0 && ix < (int)iw && iy >= 0 && iy < (int)ih) {
                    uint off0 = src0_offset + (uint)ix * nb00 + (uint)iy * nb01 + i2 * nb02 + i3 * nb03;
                    result += load_auto(src0, off0, src0_esize);
                    count++;
                }
            }
        }
        if (count > 0) result /= (float)count;
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, result, dst_esize);
}
