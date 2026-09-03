// mul_mat_id_coop.hlsl - cooperative dense expert matvec.
//
// A 32-thread group splits K and computes NUM_ROWS output rows for one
// selected expert slot/token. Supports F32/F16/BF16 weights with F32
// activations.
#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif
#ifndef NUM_ROWS
#define NUM_ROWS   4
#endif
#define MAX_WAVES  8

groupshared float shared_acc[NUM_ROWS * MAX_WAVES];

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

    uint expert_src1 = expert_slot % ne11;
    uint src1_row = src1_offset + expert_src1 * nb11 + token * nb12 + batch * nb13;

    // Rows past ne0 alias row0 and are dropped at the store, which keeps the
    // inner loop free of per-row bounds tests.
    uint src0_row[NUM_ROWS];
    precise float acc[NUM_ROWS];
    [unroll] for (uint r = 0; r < NUM_ROWS; ++r) {
        uint row = row0 + r;
        src0_row[r] = src0_base + (row < ne0 ? row : row0) * nb01;
        acc[r] = 0.0f;
    }

    for (uint k = tid; k < ne00; k += GROUP_SIZE) {
        float x = load_auto(src1, src1_row + k * nb10, src1_esize);
        [unroll] for (uint r = 0; r < NUM_ROWS; ++r) {
            acc[r] += load_auto(src0, src0_row[r] + k * nb00, src0_esize) * x;
        }
    }

    uint wave_id = tid / WaveGetLaneCount();
    uint num_waves = (GROUP_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    if (num_waves == 0) num_waves = 1;
    [unroll] for (uint r = 0; r < NUM_ROWS; ++r) {
        float wave_sum = WaveActiveSum(acc[r]);
        if (WaveIsFirstLane()) {
            shared_acc[r * MAX_WAVES + wave_id] = wave_sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float weight = op15 != 0
            ? asfloat(src3.Load(expert_slot * op3 + token * op4 + batch * op5))
            : 1.0f;
        [unroll] for (uint r = 0; r < NUM_ROWS; ++r) {
            uint row = row0 + r;
            if (row < ne0) {
                float result = shared_acc[r * MAX_WAVES];
                for (uint w = 1; w < num_waves; ++w) {
                    result += shared_acc[r * MAX_WAVES + w];
                }
                uint off = offset_4d(row, expert_slot, token, batch,
                                     nb0, nb1, nb2, nb3, dst_offset);
                store_auto(dst, off, result * weight, dst_esize);
            }
        }
    }
}
