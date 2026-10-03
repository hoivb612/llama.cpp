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
//
// Plain SSM_CONV body is implemented in ssm_conv_impl.hlsli and shared with
// the fused variants (ssm_conv_silu, ssm_conv_bias_silu).
#include "ssm_conv_impl.hlsli"
