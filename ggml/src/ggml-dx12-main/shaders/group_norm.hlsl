// group_norm.hlsl - Group normalization
// op_params[0] = num_groups, op_params[1] = eps (float)
// Groups channels (ne[2]) into num_groups groups
#include "ggml_common.hlsli"

groupshared float wave_sums[16];

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint group = gid.x;  // which (batch, group) pair
    uint local_id = gtid.x;

    uint num_groups = op0;
    float eps = asfloat(op1);

    uint n_channels = ne02;
    uint n_channels_per_group = (n_channels + num_groups - 1) / num_groups;  // ceiling division

    uint batch = group / num_groups;
    uint g = group % num_groups;

    uint c_start = g * n_channels_per_group;
    uint c_end = min(c_start + n_channels_per_group, n_channels);
    uint step = c_end - c_start;

    uint spatial = ne00 * ne01;
    uint group_size = spatial * step;

    uint wave_count = 256 / WARP_SIZE;
    uint wave_id = local_id / WARP_SIZE;

    // Pass 1: compute mean
    precise float local_sum = 0.0f;
    for (uint i = local_id; i < group_size; i += 256) {
        uint c = c_start + i / spatial;
        uint s = i % spatial;
        uint i0 = s % ne00;
        uint i1 = s / ne00;
        if (c < c_end) {
            uint off = offset_4d(i0, i1, c, batch, nb00, nb01, nb02, nb03, src0_offset);
            local_sum += load_auto(src0, off, src0_esize);
        }
    }
    float ws = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) wave_sums[wave_id] = ws;
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0f;
    if (local_id < wave_count) total = wave_sums[local_id];
    total = WaveActiveSum(total);
    if (local_id == 0) wave_sums[0] = total;
    GroupMemoryBarrierWithGroupSync();
    float mean = wave_sums[0] / (float)group_size;

    // Pass 2: compute variance and write (val - mean) to dst
    precise float local_var = 0.0f;
    for (uint i = local_id; i < group_size; i += 256) {
        uint c = c_start + i / spatial;
        uint s = i % spatial;
        uint i0 = s % ne00;
        uint i1 = s / ne00;
        if (c < c_end) {
            uint off = offset_4d(i0, i1, c, batch, nb00, nb01, nb02, nb03, src0_offset);
            uint off_d = offset_4d(i0, i1, c, batch, nb0, nb1, nb2, nb3, dst_offset);
            float v = load_auto(src0, off, src0_esize) - mean;
            store_auto(dst, off_d, v, dst_esize);
            local_var += v * v;
        }
    }
    ws = WaveActiveSum(local_var);
    if (WaveIsFirstLane()) wave_sums[wave_id] = ws;
    GroupMemoryBarrierWithGroupSync();
    total = 0.0f;
    if (local_id < wave_count) total = wave_sums[local_id];
    total = WaveActiveSum(total);
    if (local_id == 0) wave_sums[0] = total;
    GroupMemoryBarrierWithGroupSync();
    float inv_std = rsqrt(wave_sums[0] / (float)group_size + eps);

    // Pass 3: scale by 1/std
    for (uint i = local_id; i < group_size; i += 256) {
        uint c = c_start + i / spatial;
        uint s = i % spatial;
        uint i0 = s % ne00;
        uint i1 = s / ne00;
        if (c < c_end) {
            uint off_d = offset_4d(i0, i1, c, batch, nb0, nb1, nb2, nb3, dst_offset);
            float v = load_auto_rw(dst, off_d, dst_esize);
            store_auto(dst, off_d, v * inv_std, dst_esize);
        }
    }
}
