// Shared per-element quantized MUL_MAT implementation (batch matmul for
// IQ types that don't have a tiled WMMA path). Wrapper shaders define
// exactly one MMID_<TYPE> macro before including this file. One thread
// per dst output element; dequant happens on the fly per k.
//
// dst[i0, i1, i2, i3] = sum_k(src0[i0, k, i2_src0, i3_src0]
//                              * src1[i1, k, i2, i3])
// (with broadcast on src0's batch dims when ne02 != ne2 / ne03 != ne3.)
#include "quant_dequant.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint idx = flat_idx_2d(group_id, local_id);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_row = src1_offset + i1 * nb11 + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        float w = mmid_dequant(src0, src0_row, k);
        float x = load_auto(src1, src1_row + k * nb10, src1_esize);
        acc += w * x;
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
