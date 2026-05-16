// get_rows.hlsl - Gather rows from src0 using indices in src1
// src0: [ne01, ne00] data (F32 or F16, detected via nb00)
// src1: [ne10] indices (int32)
// dst:  [ne10, ne00] gathered rows (F32)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    // i1 indexes into src1 to get the row index
    uint row_idx_off = src1_offset + i1 * nb10 + i2 * nb12 + i3 * nb13;
    int row_idx = asint(src1.Load(row_idx_off));

    // Broadcast src0 batch dims if needed (src0 may have fewer batch dims)
    uint i2_src0 = ne02 > 0 ? (i2 * ne02 / ne2) : 0;
    uint i3_src0 = ne03 > 0 ? (i3 * ne03 / ne3) : 0;

    uint off0  = offset_4d(i0, (uint)row_idx, i2_src0, i3_src0, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    float val = load_auto(src0, off0, src0_esize);
    store_auto(dst, off_d, val, dst_esize);
}
