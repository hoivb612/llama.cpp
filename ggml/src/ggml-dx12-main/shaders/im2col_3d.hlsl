// im2col_3d.hlsl - 3D image-to-column transform for 3D convolution
// src0: kernel  (only shape used: ne00=KW, ne01=KH, ne02=KD)
// src1: input   [IW=ne10, IH=ne11, ID=ne12, IC*N=ne13]
// dst:  result  [IC*KD*KH*KW=ne0, OW=ne1, OH=ne2, OD*N=ne3]
//
// op_params: [0]=s0 [1]=s1 [2]=s2 [3]=p0 [4]=p1 [5]=p2 [6]=d0 [7]=d1 [8]=d2 [9]=IC
//
// One thread per dst element. Dispatch is split across X/Y to clear the
// 65535 per-dim group limit (perf cases produce ~4B elements).
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
    uint IC = op9;

    uint KW = ne00;
    uint KH = ne01;
    uint KD = ne02;
    uint KH_KW = KH * KW;
    uint KD_KH_KW = KD * KH_KW;
    uint IC_KD_KH_KW = IC * KD_KH_KW;

    uint IW = ne10;
    uint IH = ne11;
    uint ID = ne12;

    uint OW = ne1;
    uint OH = ne2;
    // ne3 = OD * N; we only need N for stepping through input batches and IC fold
    uint N  = (uint)ne13 / IC;
    uint OD = (uint)ne3 / N;

    // Decompose linear idx into (in_, iod, ioh, iow, iic, ikd, ikh, ikw)
    // matching dst layout: ne0=IC_KD_KH_KW, ne1=OW, ne2=OH, ne3=OD*N
    uint inner = idx % IC_KD_KH_KW;  // iic*KD_KH_KW + ikd*KH_KW + ikh*KW + ikw
    uint rem   = idx / IC_KD_KH_KW;
    uint iow   = rem % OW; rem /= OW;
    uint ioh   = rem % OH; rem /= OH;
    uint iod_in = rem;                // iod + in_*OD
    uint iod   = iod_in % OD;
    uint in_   = iod_in / OD;

    uint iic   = inner / KD_KH_KW;
    uint rk    = inner - iic * KD_KH_KW;
    uint ikd   = rk / KH_KW;
    uint rk2   = rk - ikd * KH_KW;
    uint ikh   = rk2 / KW;
    uint ikw   = rk2 - ikh * KW;

    int iiw = (int)iow * s0 + (int)ikw * d0 - p0;
    int iih = (int)ioh * s1 + (int)ikh * d1 - p1;
    int iid = (int)iod * s2 + (int)ikd * d2 - p2;

    uint off_d = dst_offset + idx * nb0;

    if (iiw < 0 || iiw >= (int)IW || iih < 0 || iih >= (int)IH || iid < 0 || iid >= (int)ID) {
        store_auto(dst, off_d, 0.0f, dst_esize);
    } else {
        // src1 ne13 is the IC*N axis; per the ggml_im2col_3d ref, batch+channel
        // stride is nb13 with index (in_*IC + iic). nb12=depth, nb11=height,
        // nb10=width (all in bytes).
        uint src_off = src1_offset
            + (in_ * IC + iic) * nb13
            + (uint)iid * nb12
            + (uint)iih * nb11
            + (uint)iiw * nb10;
        float val = load_auto(src1, src_off, src1_esize);
        store_auto(dst, off_d, val, dst_esize);
    }
}
