// CONV_TRANSPOSE_1D: per-output-element gather (F32-only).
// src0 (kernel): ne = [K, Cout, Cin, 1]
// src1 (input):  ne = [L, Cin, 1, 1]
// dst:           ne = [KL, Cout, 1, 1]
//   KL = (L - 1) * s0 + K
// op_params[0] = s0 (stride)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint flat = flat_idx_2d(gid, gtid.x);
    uint KL  = ne0;
    uint Cout = ne1;
    uint total = KL * Cout;
    if (flat >= total) return;

    uint KL_idx   = flat % KL;
    uint Cout_idx = flat / KL;

    int s0  = asint(op0);
    uint K   = ne00;
    uint Cin = ne02;
    uint L   = ne10;

    float sum = 0.0f;
    for (uint K_idx = 0; K_idx < K; ++K_idx) {
        int diff = int(KL_idx) - int(K_idx);
        if (diff < 0) continue;
        if (s0 <= 0) continue;
        if (diff % s0 != 0) continue;
        int L_idx = diff / s0;
        if (L_idx >= (int)L) continue;
        for (uint Cin_idx = 0; Cin_idx < Cin; ++Cin_idx) {
            uint k_off = src0_offset + K_idx * nb00 + Cout_idx * nb01 + Cin_idx * nb02;
            uint i_off = src1_offset + (uint)L_idx * nb10 + Cin_idx * nb11;
            float w = load_auto(src0, k_off, src0_esize);
            float v = load_auto(src1, i_off, src1_esize);
            sum += w * v;
        }
    }

    uint dst_off = dst_offset + KL_idx * nb0 + Cout_idx * nb1;
    store_auto(dst, dst_off, sum, dst_esize);
}
