// Shared cooperative quantized MUL_MAT_ID implementation.
// Wrapper shaders define one MMID_* format macro before including this file.
#include "quant_dequant.hlsli"

#define GROUP_SIZE 32
#define NUM_ROWS   2

groupshared float shared_acc[NUM_ROWS * 8];

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_id.x * NUM_ROWS;
    if (row0 >= ne0) return;

    uint expert_slot = group_id.y;
    uint flat_batch = group_id.z;
    uint token = flat_batch % ne2;
    uint batch = flat_batch / ne2;

    uint ids_off = op0 + expert_slot * op1 + token * op2;
    uint expert_id = (uint)asint(src2.Load(ids_off));

    uint i3_src0 = batch * ne03 / ne3;
    uint src0_base = src0_offset + expert_id * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;

    uint expert_src1 = expert_slot % ne11;
    uint src1_row = src1_offset + expert_src1 * nb11 + token * nb12 + batch * nb13;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;
    for (uint k = tid; k < ne00; k += GROUP_SIZE) {
        float x = load_auto(src1, src1_row + k * nb10, src1_esize);
        acc0 += mmid_dequant(src0, src0_row0, k) * x;
        if (row0 + 1 < ne0) {
            acc1 += mmid_dequant(src0, src0_row1, k) * x;
        }
    }

    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = tid / WaveGetLaneCount();
    uint num_waves = (GROUP_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    if (num_waves == 0) num_waves = 1;
    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[8 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float result0 = shared_acc[0];
        float result1 = shared_acc[8];
        for (uint w = 1; w < num_waves; ++w) {
            result0 += shared_acc[w];
            result1 += shared_acc[8 + w];
        }
        uint off0 = offset_4d(row0, expert_slot, token, batch, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off0, result0, dst_esize);
        if (row0 + 1 < ne0) {
            uint off1 = offset_4d(row0 + 1, expert_slot, token, batch, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off1, result1, dst_esize);
        }
    }
}
