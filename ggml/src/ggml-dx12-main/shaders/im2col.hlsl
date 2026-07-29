// im2col.hlsl - Image to column transform for convolution
// src0: kernel [KW, KH, IC, OC] (only shape used, not data)
// src1: input  [IW, IH, IC, N]
// dst:  result [IC*KH*KW, OW, OH, N]
// op_params: [0]=s0 [1]=s1 [2]=p0 [3]=p1 [4]=d0 [5]=d1 [6]=is_2D
//
// One thread per dst element.  Dispatch is split X/Y so it scales past
// D3D12's 65535 per-dim group limit (matches IM2COL_3D).
//
// Layout/precompute follows the Vulkan optimization (PR #22685):
//   - hoist KHKW = KH*KW once
//   - precompute base_iiw / base_iih (per-thread, but reused across the
//     spatial test below)
//   - signed bounds check via uint() bitcast trick: a single
//     `uint(iiw) < IW` covers both `iiw >= 0` and `iiw < IW`
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    // 2D group dispatch: groups_y * 65535 + groups_x. groups_y > 0 only
    // when the host had to split (total_groups > 65535).
    uint flat_group = gid.x + gid.y * 65535u;
    uint idx = flat_group * 256u + gtid.x;

    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    int s0 = asint(op0);  // stride width
    int s1 = asint(op1);  // stride height
    int p0 = asint(op2);  // pad width
    int p1 = asint(op3);  // pad height
    int d0 = asint(op4);  // dilation width
    int d1 = asint(op5);  // dilation height
    int is_2D = asint(op6);

    uint KW   = ne00;
    uint KH   = (is_2D != 0) ? ne01 : 1;
    uint KHKW = KH * KW;

    uint IW = ne10;
    uint IH = (is_2D != 0) ? ne11 : 1;

    // dst layout: [IC*KH*KW, OW, OH, N]
    uint i0 = idx % ne0;       uint rem = idx / ne0;  // i0 = ic*KHKW + ikh*KW + ikw
    uint i1 = rem % ne1;       rem /= ne1;            // i1 = OW index
    uint i2 = rem % ne2;       uint i3 = rem / ne2;   // i2 = OH or N; i3 = N or 0

    // Decompose i0 into (iic, ikh, ikw) reusing KHKW.
    uint iic    = i0 / KHKW;
    uint i0_rem = i0 - iic * KHKW;
    uint ikh    = i0_rem / KW;
    uint ikw    = i0_rem - ikh * KW;

    // dst layout differs by is_2D (per ggml_im2col):
    //   is_2D=1: dst.ne = [IC*KH*KW, OW, OH, N]  -> i2 = OH, i3 = N
    //   is_2D=0: dst.ne = [IC*KW,    OW, N,  1]  -> i2 = N,  i3 = 0
    uint iow = i1;
    uint ioh = (is_2D != 0) ? i2 : 0;
    uint in_ = (is_2D != 0) ? i3 : i2;

    // Precompute spatial base; per-(ikw, ikh) coords are an additive delta.
    int base_iiw = (int)iow * s0 - p0;
    int base_iih = (int)ioh * s1 - p1;
    int iiw      = base_iiw + (int)ikw * d0;
    int iih      = base_iih + (int)ikh * d1;

    uint off_d = dst_offset + idx * nb0;  // dst stride: nb0 = 2 for F16, 4 for F32

    // Signed bounds via uint bitcast: negative -> large unsigned -> fails check.
    if (uint(iih) >= IH || uint(iiw) >= IW) {
        store_auto(dst, off_d, 0.0f, dst_esize);
    } else {
        // src1 access: src1[iiw, iih, iic, in_]
        uint src_off;
        if (is_2D != 0) {
            src_off = src1_offset + (uint)iiw * nb10 + (uint)iih * nb11 + iic * nb12 + in_ * nb13;
        } else {
            src_off = src1_offset + (uint)iiw * nb10 + iic * nb11 + in_ * nb12;
        }
        float val = load_auto(src1, src_off, src1_esize);
        store_auto(dst, off_d, val, dst_esize);
    }
}
