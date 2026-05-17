// l2_norm.hlsl - L2 normalization (per-row): dst = src0 / max(sqrt(sum(src0^2)), eps)
// eps is stored in op_param_uint(0) as float.
// One thread group per row over (ne1, ne2, ne3); ne0 elements per row.
#include "ggml_common.hlsli"

groupshared float wave_sums[16];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint total_rows = ne1 * ne2 * ne3;
    if (row >= total_rows) return;

    uint i3 = row / (ne1 * ne2);
    uint rem = row % (ne1 * ne2);
    uint i2 = rem / ne1;
    uint i1 = rem % ne1;

    float eps = op_param_f32(0);
    uint local_id = gtid.x;
    uint wave_count = 256 / WARP_SIZE;
    uint wave_id = local_id / WARP_SIZE;

    precise float local_sum = 0.0f;
    for (uint i0 = local_id; i0 < ne00; i0 += 256) {
        uint off = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        float val = load_auto(src0, off, src0_esize);
        local_sum += val * val;
    }

    float wave_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        wave_sums[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    float total = 0.0f;
    if (local_id < wave_count) {
        total = wave_sums[local_id];
    }
    total = WaveActiveSum(total);
    if (local_id == 0) {
        wave_sums[0] = total;
    }
    GroupMemoryBarrierWithGroupSync();
    total = wave_sums[0];

    float norm = sqrt(total);
    float scale_val = 1.0f / max(norm, eps);

    for (uint i0 = local_id; i0 < ne0; i0 += 256) {
        uint off_src = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint off_dst = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        float val = load_auto(src0, off_src, src0_esize);
        store_auto(dst, off_dst, val * scale_val, dst_esize);
    }
}
