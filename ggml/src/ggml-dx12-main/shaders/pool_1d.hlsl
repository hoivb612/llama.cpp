// pool_1d.hlsl - 1D pooling (max or average)
// op_params: [0]=pool_type (0=max, 1=avg), [1]=kernel_size, [2]=stride, [3]=padding
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
    uint k = op1;
    uint s = op2;
    uint p = op3;

    uint iw = ne00;  // input width

    float result;
    if (pool_type == 0) {
        // Max pooling
        result = -3.402823466e+38f;
        for (uint ki = 0; ki < k; ki++) {
            int ix = (int)(i0 * s + ki) - (int)p;
            if (ix >= 0 && ix < (int)iw) {
                uint off0 = offset_4d((uint)ix, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
                result = max(result, load_auto(src0, off0, src0_esize));
            }
        }
    } else {
        // Average pooling
        result = 0.0f;
        uint count = 0;
        for (uint ki = 0; ki < k; ki++) {
            int ix = (int)(i0 * s + ki) - (int)p;
            if (ix >= 0 && ix < (int)iw) {
                uint off0 = offset_4d((uint)ix, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
                result += load_auto(src0, off0, src0_esize);
                count++;
            }
        }
        if (count > 0) result /= (float)count;
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, result, dst_esize);
}
