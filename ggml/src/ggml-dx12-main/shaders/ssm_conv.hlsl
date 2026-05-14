// ssm_conv.hlsl - State-space-model 1D depthwise convolution (Mamba/Gated Delta Net)
//
// src0 = conv_x : shape [ncs, nr, n_s]    (sliding window input; ncs = d_conv-1+n_t)
// src1 = conv_w : shape [nc,  nr]          (per-channel kernel; nc = d_conv)
// dst         : shape [nr, n_t, n_s]
//
// dst[i1, i2, i3] = sum_{i0=0..nc-1} src0[i2+i0, i1, i3] * src1[i0, i1]
//
// Where:
//   i1 = output channel  (row, 0..nr-1)        nr  = ne0
//   i2 = token index     (0..n_t-1)            n_t = ne1
//   i3 = sequence index  (0..n_s-1)            n_s = ne2
//   nc = kernel width                          nc  = ne10
//
// F32 only; src0->nb[0] = src1->nb[0] = 4.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2;
    if (idx >= total) return;

    uint i1 = idx % ne0;
    uint rem = idx / ne0;
    uint i2 = rem % ne1;
    uint i3 = rem / ne1;

    uint nc = ne10;

    uint s_base = src0_offset + i3 * nb02 + i1 * nb01 + i2 * nb00;
    uint c_base = src1_offset + i1 * nb11;

    float sum = 0.0f;
    for (uint i0 = 0; i0 < nc; ++i0) {
        float s_val = asfloat(src0.Load(s_base + i0 * nb00));
        float c_val = asfloat(src1.Load(c_base + i0 * nb10));
        sum += s_val * c_val;
    }

    uint d_off = dst_offset + i3 * nb2 + i2 * nb1 + i1 * nb0;
    dst.Store(d_off, asuint(sum));
}
