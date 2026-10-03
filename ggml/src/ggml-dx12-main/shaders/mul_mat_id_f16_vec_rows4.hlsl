// F16 expert matvec with four K values per lane and four output rows per group.
#include "ggml_common.hlsli"

#define GROUP_SIZE 64
#define NUM_ROWS   4
#define MAX_WAVES  4

groupshared float shared_acc[NUM_ROWS * MAX_WAVES];

float4 load_f16x4(ByteAddressBuffer buf, uint byte_off) {
    uint2 packed = buf.Load2(byte_off);
    return float4(
        f16_to_f32(packed.x & 0xFFFFu),
        f16_to_f32(packed.x >> 16),
        f16_to_f32(packed.y & 0xFFFFu),
        f16_to_f32(packed.y >> 16));
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_id.x * NUM_ROWS;
    if (row0 >= ne0) {
        return;
    }

    uint expert_slot = group_id.y;
    uint flat_batch = group_id.z;
    uint token = flat_batch % ne2;
    uint batch = flat_batch / ne2;

    uint ids_off = op0 + expert_slot * op1 + token * op2;
    uint expert_id = (uint)asint(src2.Load(ids_off));
    uint src0_base = src0_offset + expert_id * nb02 + (batch * ne03 / ne3) * nb03;

    uint expert_src1 = expert_slot % ne11;
    uint src1_row = src1_offset + expert_src1 * nb11 + token * nb12 + batch * nb13;

    uint src0_row[NUM_ROWS];
    precise float acc[NUM_ROWS];
    [unroll] for (uint r = 0u; r < NUM_ROWS; ++r) {
        uint row = row0 + r;
        src0_row[r] = src0_base + (row < ne0 ? row : row0) * nb01;
        acc[r] = 0.0f;
    }

    for (uint k = tid * 4u; k < ne00; k += GROUP_SIZE * 4u) {
        uint4 x_raw = src1.Load4(src1_row + k * nb10);
        float4 x = asfloat(x_raw);

        [unroll] for (uint r2 = 0u; r2 < NUM_ROWS; ++r2) {
            float4 w = load_f16x4(src0, src0_row[r2] + k * nb00);
            float sum = mad(w.x, x.x, w.y * x.y);
            sum = mad(w.z, x.z, sum);
            acc[r2] = mad(w.w, x.w, acc[r2] + sum);
        }
    }

    uint wave_lanes = WaveGetLaneCount();
    uint wave_id = tid / wave_lanes;
    uint num_waves = (GROUP_SIZE + wave_lanes - 1u) / wave_lanes;
    [unroll] for (uint r3 = 0u; r3 < NUM_ROWS; ++r3) {
        float wave_sum = WaveActiveSum(acc[r3]);
        if (WaveIsFirstLane()) {
            shared_acc[r3 * MAX_WAVES + wave_id] = wave_sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < NUM_ROWS) {
        uint row = row0 + tid;
        if (row < ne0) {
            float result = shared_acc[tid * MAX_WAVES];
            for (uint w = 1u; w < num_waves; ++w) {
                result += shared_acc[tid * MAX_WAVES + w];
            }
            if (op15 != 0u) {
                result *= asfloat(src3.Load(expert_slot * op3 + token * op4 + batch * op5));
            }
            uint off = offset_4d(row, expert_slot, token, batch, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off, result, dst_esize);
        }
    }
}
