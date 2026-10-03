#include "ggml_common.hlsli"

#ifndef NUM_ROWS
#define NUM_ROWS 4
#endif

float4 unpack_f16x4(uint2 packed) {
    return float4(f16_to_f32(packed.x & 0xFFFFu), f16_to_f32(packed.x >> 16),
                  f16_to_f32(packed.y & 0xFFFFu), f16_to_f32(packed.y >> 16));
}

// Same expert and weighted-output ABI as mul_mat_id_f16_vec_rows4.
[WaveSize(64)]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_id.x * NUM_ROWS;
    if (row0 >= ne0) {
        return;
    }

    uint expert_slot = group_id.y;
    uint token = group_id.z % ne2;
    uint batch = group_id.z / ne2;
    uint expert_id = (uint)asint(src2.Load(op0 + expert_slot * op1 + token * op2));
    uint weight_base = src0_offset + expert_id * nb02 + (batch * ne03 / ne3) * nb03;
    uint input_base = src1_offset + (expert_slot % ne11) * nb11 + token * nb12 + batch * nb13;

    uint weight_row[NUM_ROWS];
    precise float acc_lo[NUM_ROWS];
    precise float acc_hi[NUM_ROWS];
    [unroll] for (uint r = 0u; r < NUM_ROWS; ++r) {
        weight_row[r] = weight_base + min(row0 + r, ne0 - 1u) * nb01;
        acc_lo[r] = 0.0f;
        acc_hi[r] = 0.0f;
    }

    // Eight adjacent K values per lane give one Load4 per weight row.
    for (uint k = tid * 8u; k < ne00; k += 512u) {
        float4 x_lo = asfloat(src1.Load4(input_base + k * nb10));
        float4 x_hi = asfloat(src1.Load4(input_base + (k + 4u) * nb10));
        [unroll] for (uint r = 0u; r < NUM_ROWS; ++r) {
            uint4 packed = src0.Load4(weight_row[r] + k * nb00);
            float4 w_lo = unpack_f16x4(packed.xy);
            float4 w_hi = unpack_f16x4(packed.zw);
            float sum_lo = mad(w_lo.x, x_lo.x, w_lo.y * x_lo.y);
            sum_lo = mad(w_lo.z, x_lo.z, sum_lo);
            acc_lo[r] = mad(w_lo.w, x_lo.w, acc_lo[r] + sum_lo);
            float sum_hi = mad(w_hi.x, x_hi.x, w_hi.y * x_hi.y);
            sum_hi = mad(w_hi.z, x_hi.z, sum_hi);
            acc_hi[r] = mad(w_hi.w, x_hi.w, acc_hi[r] + sum_hi);
        }
    }

    [unroll] for (uint r = 0u; r < NUM_ROWS; ++r) {
        float result = WaveActiveSum(acc_lo[r] + acc_hi[r]);
        if (tid == 0u && row0 + r < ne0) {
            if (op15 != 0u) {
                result *= asfloat(src3.Load(expert_slot * op3 + token * op4 + batch * op5));
            }
            uint off = offset_4d(row0 + r, expert_slot, token, batch, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off, result, dst_esize);
        }
    }
}
