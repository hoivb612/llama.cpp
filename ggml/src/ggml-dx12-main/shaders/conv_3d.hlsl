// conv_3d.hlsl - direct 3D convolution.
//
// src0: kernel  [KW=ne00, KH=ne01, KD=ne02, c*oc=ne03]
//                element stride src0_esize (2=F16, 4=F32)
// src1: input   [IW=ne10, IH=ne11, ID=ne12, c*n=ne13]  (F32)
// dst:  output  [OW=ne0, OH=ne1, OD=ne2, oc*n=ne3]     (F32)
//
// op_params: [0]=s0 [1]=s1 [2]=s2 [3]=p0 [4]=p1 [5]=p2
//            [6]=d0 [7]=d1 [8]=d2 [9]=c  [10]=n [11]=oc
//
// Per dst element [dst_x, dst_y, dst_z, ocn]:
//   batch_idx = ocn / oc, ioc = ocn % oc
//   sum over (ic, kz, ky, kx) of
//     input[sx, sy, sz, batch_idx*c + ic] * kernel[kx, ky, kz, ioc*c + ic]
//   where sx = dst_x*s0 + kx*d0 - p0  (similarly sy, sz);
//   src_val = 0 outside [0, IW)/[0, IH)/[0, ID).
//
// One thread per dst element. Dispatch is split across X/Y to clear the
// 65535 per-dim group limit on perf cases.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint lt : SV_GroupIndex) {
    uint group_lin = gid.y * 65535u + gid.x;
    uint idx = group_lin * 256u + lt;

    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    int s0 = asint(op0);
    int s1 = asint(op1);
    int s2 = asint(op2);
    int p0 = asint(op3);
    int p1 = asint(op4);
    int p2 = asint(op5);
    int d0 = asint(op6);
    int d1 = asint(op7);
    int d2 = asint(op8);
    uint c  = op9;
    uint oc = op11;

    uint KW = ne00;
    uint KH = ne01;
    uint KD = ne02;
    uint IW = ne10;
    uint IH = ne11;
    uint ID = ne12;

    uint OW = ne0;
    uint OH = ne1;
    uint OD = ne2;

    // Decompose linear idx into (dst_x, dst_y, dst_z, ocn) per dst layout.
    uint dst_x = idx % OW;
    uint rem   = idx / OW;
    uint dst_y = rem % OH;
    rem /= OH;
    uint dst_z = rem % OD;
    uint ocn   = rem / OD;

    uint batch_idx = ocn / oc;
    uint ioc       = ocn - batch_idx * oc;

    uint KH_KW    = KH * KW;
    uint KD_KH_KW = KD * KH_KW;

    float acc = 0.0f;

    for (uint ic = 0; ic < c; ++ic) {
        uint cn_idx_in  = batch_idx * c + ic;
        uint cn_idx_knl = ioc       * c + ic;

        uint knl_chan_off = src0_offset + cn_idx_knl * nb03;
        uint src_chan_off = src1_offset + cn_idx_in  * nb13;

        for (uint kz = 0; kz < KD; ++kz) {
            int sz = (int)dst_z * s2 + (int)kz * d2 - p2;
            bool z_in = (sz >= 0) && (sz < (int)ID);
            uint knl_z_off = knl_chan_off + kz * nb02;
            uint src_z_off = src_chan_off + (uint)sz * nb12;

            for (uint ky = 0; ky < KH; ++ky) {
                int sy = (int)dst_y * s1 + (int)ky * d1 - p1;
                bool yz_in = z_in && (sy >= 0) && (sy < (int)IH);
                uint knl_y_off = knl_z_off + ky * nb01;
                uint src_y_off = src_z_off + (uint)sy * nb11;

                for (uint kx = 0; kx < KW; ++kx) {
                    int sx = (int)dst_x * s0 + (int)kx * d0 - p0;
                    float src_val = 0.0f;
                    if (yz_in && sx >= 0 && sx < (int)IW) {
                        uint src_off = src_y_off + (uint)sx * nb10;
                        src_val = asfloat(src1.Load(src_off));
                    }
                    uint knl_off = knl_y_off + kx * nb00;
                    float knl_val = load_auto(src0, knl_off, src0_esize);
                    acc += src_val * knl_val;
                }
            }
        }
    }

    uint dst_off = dst_offset + dst_x * nb0 + dst_y * nb1 + dst_z * nb2 + ocn * nb3;
    dst.Store(dst_off, asuint(acc));
}
