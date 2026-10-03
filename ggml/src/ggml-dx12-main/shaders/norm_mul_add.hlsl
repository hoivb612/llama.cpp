// LayerNorm with affine scale and bias, optionally preceded by a residual add.
#include "ggml_common.hlsli"

#define MAX_CACHED 4

groupshared float wave_vals[32];

WAVE_SIZE_ATTR
[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint row = gid.x;
    const uint total_rows = ne1 * ne2 * ne3;
    if (row >= total_rows) {
        return;
    }

    const uint i3 = row / (ne1 * ne2);
    const uint rem = row % (ne1 * ne2);
    const uint i2 = rem / ne1;
    const uint i1 = rem % ne1;
    const uint local_id = gtid.x;
    const uint lane_count = WaveGetLaneCount();
    const uint wave_count = (256 + lane_count - 1) / lane_count;
    const uint wave_id = local_id / lane_count;

    const bool has_add = op0 != 0u;
    const uint residual_offset = op1;
    const float eps = asfloat(op2);
    const uint residual_esize = op3;
    const uint weight_offset = op4;
    const uint bias_offset = op5;

    float cached[MAX_CACHED];
    uint n_cached = 0;
    precise float local_sum = 0.0f;

    for (uint i0 = local_id; i0 < ne00; i0 += 256) {
        const uint off0 = offset_4d(i0, i1, i2, i3,
                                    nb00, nb01, nb02, nb03, src0_offset);
        float value = load_auto(src0, off0, src0_esize);
        if (has_add) {
            const uint b0 = i0 % ne10;
            const uint b1 = i1 % ne11;
            const uint b2 = i2 % ne12;
            const uint b3 = i3 % ne13;
            const uint off1 = offset_4d(b0, b1, b2, b3,
                                        nb10, nb11, nb12, nb13, src1_offset);
            value += load_auto(src1, off1, src1_esize);
            const uint residual_off = offset_4d(i0, i1, i2, i3,
                                                nb00, nb01, nb02, nb03,
                                                residual_offset);
            store_auto(dst, residual_off, value, residual_esize);
        }
        if (n_cached < MAX_CACHED) {
            cached[n_cached++] = value;
        }
        local_sum += value;
    }

    const float wave_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        wave_vals[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (local_id == 0) {
        float sum = wave_vals[0];
        for (uint w = 1; w < wave_count; ++w) {
            sum += wave_vals[w];
        }
        wave_vals[0] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    const float mean = wave_vals[0] / (float)ne00;

    precise float local_var = 0.0f;
    uint ci = 0;
    for (uint i0 = local_id; i0 < ne00; i0 += 256) {
        float value;
        if (ci < n_cached) {
            value = cached[ci++];
        } else if (has_add) {
            const uint residual_off = offset_4d(i0, i1, i2, i3,
                                                nb00, nb01, nb02, nb03,
                                                residual_offset);
            value = load_auto_rw(dst, residual_off, residual_esize);
        } else {
            const uint off0 = offset_4d(i0, i1, i2, i3,
                                        nb00, nb01, nb02, nb03, src0_offset);
            value = load_auto(src0, off0, src0_esize);
        }
        const float diff = value - mean;
        local_var += diff * diff;
    }

    const float wave_var = WaveActiveSum(local_var);
    if (WaveIsFirstLane()) {
        wave_vals[wave_id] = wave_var;
    }
    GroupMemoryBarrierWithGroupSync();

    if (local_id == 0) {
        float sum = wave_vals[0];
        for (uint w = 1; w < wave_count; ++w) {
            sum += wave_vals[w];
        }
        wave_vals[0] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    const float inv_std = 1.0f / sqrt(wave_vals[0] / (float)ne00 + eps);

    ci = 0;
    for (uint i0 = local_id; i0 < ne0; i0 += 256) {
        float value;
        if (ci < n_cached) {
            value = cached[ci++];
        } else if (has_add) {
            const uint residual_off = offset_4d(i0, i1, i2, i3,
                                                nb00, nb01, nb02, nb03,
                                                residual_offset);
            value = load_auto_rw(dst, residual_off, residual_esize);
        } else {
            const uint off0 = offset_4d(i0, i1, i2, i3,
                                        nb00, nb01, nb02, nb03, src0_offset);
            value = load_auto(src0, off0, src0_esize);
        }
        const float weight = asfloat(src2.Load(weight_offset + i0 * 4u));
        const float bias = asfloat(src3.Load(bias_offset + i0 * 4u));
        const uint off_dst = offset_4d(i0, i1, i2, i3,
                                       nb0, nb1, nb2, nb3, dst_offset);
        precise float normalized = (value - mean) * inv_std;
        precise float scaled = normalized * weight;
        precise float result = scaled + bias;
        store_auto(dst, off_dst, result, dst_esize);
    }
}
