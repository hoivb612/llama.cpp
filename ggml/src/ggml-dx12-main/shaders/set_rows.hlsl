// set_rows.hlsl - Set rows in destination tensor using indices from src1
// src0: [ne00, ne01, ne02, ne03] source data (F32)
// src1: [ne10] indices (I32 or I64)
// dst:  target tensor where rows are written (F32 or F16)
// dst[src1[i1], :] = src0[i1, :]
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Paired F16 mode for KV cache writes: 2 elements per thread
    bool paired = (dst_esize == 2 && nb0 == 2 && (ne00 & 1) == 0 &&
                   (dst_offset & 3) == 0 && (nb1 & 3) == 0);
    uint idx = tid.x * (paired ? 2u : 1u);
    uint total = ne00 * ne01 * ne02 * ne03;
    if (idx >= total) return;

    uint i0 = idx % ne00;
    uint rem = idx / ne00;
    uint i1 = rem % ne01;
    rem = rem / ne01;
    uint i2 = rem % ne02;
    uint i3 = rem / ne02;

    // Get destination row index from src1
    // src1 (indices) is 3D [ne10, ne11, ne12] mapping to dst (i1, i2, i3).
    // Match CPU semantics: broadcast dim2/3 with MODULO (i02 % ne11, i03 % ne12),
    // not integer division. ne02 % ne11 == 0 and ne03 % ne12 == 0 are asserted.
    uint i2_idx = ne11 > 0 ? (i2 % ne11) : 0;
    uint i3_idx = ne12 > 0 ? (i3 % ne12) : 0;
    uint idx_off = src1_offset + i1 * nb10 + i2_idx * nb11 + i3_idx * nb12;
    int row_idx = asint(src1.Load(idx_off));

    uint off0 = src0_offset + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    float v0 = asfloat(src0.Load(off0));

    if (paired) {
        float v1 = asfloat(src0.Load(off0 + nb00));
        uint off_d = dst_offset + i0 * 2u + (uint)row_idx * nb1 + i2 * nb2 + i3 * nb3;
        store_f16_pair(dst, off_d, v0, v1);
    } else {
        uint off_d = dst_offset + i0 * nb0 + (uint)row_idx * nb1 + i2 * nb2 + i3 * nb3;
        store_auto(dst, off_d, v0, dst_esize);
    }
}
