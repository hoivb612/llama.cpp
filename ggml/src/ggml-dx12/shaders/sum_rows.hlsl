// sum_rows.hlsl - Sum rows: dst[i1,i2,i3] = sum_i0(src0[i0,i1,i2,i3])
// Output has ne0=1
#include "ggml_common.hlsli"

groupshared float shared_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint total_rows = ne1 * ne2 * ne3;
    if (row >= total_rows) return;

    uint i3 = row / (ne1 * ne2);
    uint rem = row % (ne1 * ne2);
    uint i2 = rem / ne1;
    uint i1 = rem % ne1;
    uint local_id = gtid.x;

    precise float local_sum = 0.0f;
    for (uint i0 = local_id; i0 < ne00; i0 += 256) {
        uint off = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        local_sum += load_auto(src0, off, src0_esize);
    }

    shared_sum[local_id] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            precise float a = shared_sum[local_id];
            precise float b = shared_sum[local_id + s];
            shared_sum[local_id] = a + b;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        uint off_dst = offset_4d(0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_dst, shared_sum[0], dst_esize);
    }
}
