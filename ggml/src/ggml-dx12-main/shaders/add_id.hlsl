// add_id.hlsl - dst[i0, i1, i2] = src0[i0, i1, i2] + src1[ids[i1, i2], i0]
//
// src0 and dst share shape [n_embd, n_experts_used, n_token] (F32, contiguous).
// src1 is the expert weights [n_embd, n_experts] (F32, contiguous along dim0).
// src2 is the I32 ids tensor [n_experts_used, n_token]. It may be a view of a
// larger [n_experts, n_token] tensor; the per-tensor byte offset for src2 is
// already baked into the src2 GPU VA (gdn_or_ssm-style binding), so the
// shader reads from byte 0. nb20 and nb21 are passed via op_params.
//
// op_params layout (filled by ggml-dx12.cpp):
//   op0 = nb20 (src2 stride along dim 0, bytes — usually 4)
//   op1 = nb21 (src2 stride along dim 1, bytes)
//
// One thread group per dst row (over the i1, i2, i3 product); 256 threads
// per group cooperate on the n_embd add.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint nrows = ne1 * ne2 * ne3;
    if (row >= nrows) return;

    uint i3 = row / (ne2 * ne1);
    uint i2 = (row - i3 * ne2 * ne1) / ne1;
    uint i1 = row - i3 * ne2 * ne1 - i2 * ne1;

    uint id_off = i1 * op0 + i2 * op1;
    uint i11 = src2.Load(id_off);

    uint base_src0 = src0_offset + i3 * nb03 + i2 * nb02 + i1 * nb01;
    uint base_src1 = src1_offset + i11 * nb11;
    uint base_dst  = dst_offset  + i3 * nb3  + i2 * nb2  + i1 * nb1;

    for (uint i0 = gtid.x; i0 < ne0; i0 += 256) {
        float a = asfloat(src0.Load(base_src0 + i0 * nb00));
        float b = asfloat(src1.Load(base_src1 + i0 * nb10));
        dst.Store(base_dst + i0 * nb0, asuint(a + b));
    }
}
