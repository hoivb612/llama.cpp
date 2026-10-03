#include "ggml_common.hlsli"

#define GROUP_SIZE       64
#define NUM_ROWS         4
#define QK2_0            64
#define Q2_0_BSIZE       18
#define Q8_1_BSIZE       36
#define BLOCKS_PER_ITER  4

uint read_u8_q20(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float read_f16_q20(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint unpack_q2(uint q) {
    return (q & 3u) |
           (((q >> 2u) & 3u) << 8u) |
           (((q >> 4u) & 3u) << 16u) |
           (((q >> 6u) & 3u) << 24u);
}

WAVE_SIZE_ATTR
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * NUM_ROWS;
    if (row0 >= ne0) return;

    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint num_q2_blocks = ne00 / QK2_0;
    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_rows[NUM_ROWS];
    [unroll] for (uint r = 0; r < NUM_ROWS; ++r) {
        src0_rows[r] = src0_base + min(row0 + r, ne0 - 1u) * nb01;
    }

    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_vec_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_q2_blocks * 2u * Q8_1_BSIZE;

    uint block_slot = tid >> 4u;
    uint q_lane = tid & 15u;
    uint q8_half = q_lane >> 3u;
    uint q8_lane = q_lane & 7u;

    precise float acc[NUM_ROWS];
    [unroll] for (uint r = 0; r < NUM_ROWS; ++r) {
        acc[r] = 0.0f;
    }

    for (uint block_iter = 0; block_iter < num_q2_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block_idx = block_iter + block_slot;
        bool block_valid = block_idx < num_q2_blocks;
        uint safe_block_idx = min(block_idx, num_q2_blocks - 1u);
        uint q8_off = q8_vec_base + (safe_block_idx * 2u + q8_half) * Q8_1_BSIZE;
        float a_d_lane = q8_lane == 0u ? f16_to_f32(src1.Load(q8_off) & 0xFFFFu) : 0.0f;
        float a_d = WaveReadLaneAt(a_d_lane, block_slot * 16u + q8_half * 8u);
        uint a_packed = src1.Load(q8_off + 4u + q8_lane * 4u);

        int a_sum = 0;
        a_sum = dot4add_i8packed(0x01010101u, a_packed, a_sum);

        [unroll] for (uint r = 0; r < NUM_ROWS; ++r) {
            uint w_off = src0_rows[r] + safe_block_idx * Q2_0_BSIZE;
            float w_d_lane = q_lane == 0u ? read_f16_q20(src0, w_off) : 0.0f;
            float w_d = WaveReadLaneAt(w_d_lane, block_slot * 16u);
            uint q_packed = unpack_q2(read_u8_q20(src0, w_off + 2u + q_lane));

            int isum = 0;
            isum = dot4add_i8packed(q_packed, a_packed, isum);
            if (block_valid) {
                acc[r] = mad(w_d * a_d, float(isum - a_sum), acc[r]);
            }
        }
    }

    [unroll] for (uint r = 0; r < NUM_ROWS; ++r) {
        float result = WaveActiveSum(acc[r]);
        if (tid == 0u && row0 + r < ne0) {
            result += load_fused_bias(row0 + r, i2, i3);
            uint off_d = offset_4d(row0 + r, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d, result, dst_esize);
        }
    }
}
