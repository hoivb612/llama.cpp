// set_rows.hlsl - Set rows in destination tensor using indices from src1
// src0: [ne00, ne01, ne02, ne03] source data (F32)
// src1: [ne10] indices (I32 or I64)
// dst:  target tensor where rows are written (F32 or F16)
// dst[src1[i1], :] = src0[i1, :]
//
// Uses 2D dispatch (groups_y > 1) when total groups exceed D3D12's 65535 limit.
// group_flat = gid.x + gid.y * 65535, idx = group_flat * 256 + local_tid.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint local_tid : SV_GroupIndex) {
    uint group_flat = gid.x + gid.y * 65535u;
    uint idx = group_flat * 256u + local_tid;
    uint total = ne00 * ne01 * ne02 * ne03;
    if (idx >= total) return;

    // Optimized index decomposition avoiding expensive flat_to_4d divisions
    // For KV cache writes: ne03=1, ne02=n_heads, ne01=1, ne00=head_size
    // Most common: i3=0, i2=idx/(ne00*ne01), i1=(idx/ne00)%ne01, i0=idx%ne00
    uint i0 = idx % ne00;
    uint rem = idx / ne00;
    uint i1 = rem % ne01;
    rem = rem / ne01;
    uint i2 = rem % ne02;
    uint i3 = rem / ne02;

    // Get destination row index from src1
    uint idx_off = src1_offset + i1 * nb10 + i2 * nb12 + i3 * nb13;
    int row_idx = asint(src1.Load(idx_off));

    // Read from src0 — direct F32 load (src0 is always F32 for SET_ROWS)
    uint off0 = src0_offset + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    float val = asfloat(src0.Load(off0));

    // Write to dst at the indexed row
    uint off_d = dst_offset + i0 * nb0 + (uint)row_idx * nb1 + i2 * nb2 + i3 * nb3;
    store_auto(dst, off_d, val, dst_esize);
}
