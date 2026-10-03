// acc.hlsl - ACC: dst[i] = src0[i] + (in_patch_region ? src1[mapped_idx] : 0)
//
// CPU reference: ggml_compute_forward_acc_f32 (ggml/src/ggml-cpu/ops.cpp)
// op_params: [0]=nb1 [1]=nb2 [2]=nb3 (BYTE strides for dst-view)
//            [3]=byte offset into dst, [4]=inplace flag (ignored on GPU;
//            issuing the src0->dst copy unconditionally is correct
//            because src0 and dst alias when inplace=true).
//
// src0 and dst are guaranteed SAME SHAPE and BOTH CONTIGUOUS F32.
// src1 is a smaller F32 patch tensor with shape (ne10, ne11, ne12, ne13)
// and may have arbitrary strides nb10..nb13.
//
// One thread per dst element. Outside the patch region, copy src0;
// inside, write src0 + src1.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint off_d = dst_offset  + idx * 4u;
    uint off0  = src0_offset + idx * 4u;
    float v0 = asfloat(src0.Load(off0));

    uint stride1_e = op0 / 4u;
    uint stride2_e = op1 / 4u;
    uint stride3_e = op2 / 4u;
    uint offset_e  = op3 / 4u;

    uint pidx = idx - offset_e;
    uint i3p  = pidx / stride3_e;
    uint r2   = pidx - i3p * stride3_e;
    uint i2p  = r2 / stride2_e;
    uint r1   = r2 - i2p * stride2_e;
    uint i1p  = r1 / stride1_e;
    uint i0p  = r1 - i1p * stride1_e;

    bool in_patch = (idx >= offset_e) &&
                    (i0p < ne10) && (i1p < ne11) &&
                    (i2p < ne12) && (i3p < ne13);

    if (in_patch) {
        uint off1 = offset_4d(i0p, i1p, i2p, i3p, nb10, nb11, nb12, nb13, src1_offset);
        float v1 = asfloat(src1.Load(off1));
        dst.Store(off_d, asuint(v0 + v1));
    } else {
        dst.Store(off_d, asuint(v0));
    }
}
