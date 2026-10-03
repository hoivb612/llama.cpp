#include "quant_dequant.hlsli"

#define GROUP_SIZE 256
#define NUM_ROWS 2

groupshared float shared_acc[64];

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * NUM_ROWS;
    if (row0 >= ne0) return;

    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;
    bool has_row1 = row0 + 1 < ne0;
    for (uint k = tid; k < ne00; k += GROUP_SIZE) {
        float x = asfloat(src1.Load(src1_base + k * nb10));
        acc0 = mad(mmid_dequant(src0, src0_row0, k), x, acc0);
        if (has_row1) {
            acc1 = mad(mmid_dequant(src0, src0_row1, k), x, acc1);
        }
    }

    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint lane_count = WaveGetLaneCount();
    uint wave_id = tid / lane_count;
    uint num_waves = (GROUP_SIZE + lane_count - 1) / lane_count;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float result0 = shared_acc[0];
        float result1 = shared_acc[32];
        for (uint w = 1; w < num_waves; ++w) {
            result0 += shared_acc[w];
            result1 += shared_acc[32 + w];
        }

        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (has_row1) {
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
