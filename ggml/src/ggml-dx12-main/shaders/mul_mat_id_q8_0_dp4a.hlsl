// mul_mat_id_q8_0_dp4a.hlsl - Q8_0 expert matvec with Q8_1 activations.
//
// One workgroup computes two output rows for one selected expert slot/token.
// The F32 activation is quantized once by the existing Q8_1 pre-pass and reused
// across both rows and consecutive expert projections.
#include "ggml_common.hlsli"

#define GROUP_SIZE  32
#define QK8_0       32
#define Q8_0_BSIZE  34
#define Q8_1_BSIZE  36
#define NUM_ROWS    2
#define BLOCKS_PER_ITER (GROUP_SIZE / 8)

groupshared float shared_acc[NUM_ROWS * 8];

uint read_u32_q80(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_q80(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

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

    uint K = ne00;
    uint num_blocks = K / QK8_0;

    uint expert_src1 = expert_slot % ne11;
    uint q8_vec_idx = (batch * ne12 + token) * ne11 + expert_src1;
    uint q8_vec_base = src1_offset + q8_vec_idx * num_blocks * Q8_1_BSIZE;

    uint sub = tid / 8;
    uint lane = tid % 8;
    uint l0 = lane * 4;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    for (uint block_iter = 0; block_iter < num_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block_idx = block_iter + sub;
        if (block_idx < num_blocks) {
            uint q8_off = q8_vec_base + block_idx * Q8_1_BSIZE;
            uint ds = src1.Load(q8_off);
            float a_d = f16_to_f32(ds & 0xFFFFu);
            uint a_packed = src1.Load(q8_off + 4 + l0);

            uint w_off0 = src0_row0 + block_idx * Q8_0_BSIZE;
            float w_d0 = read_f16_q80(src0, w_off0);
            uint w_packed0 = read_u32_q80(src0, w_off0 + 2 + l0);
            int isum0 = 0;
            isum0 = dot4add_i8packed(w_packed0, a_packed, isum0);
            acc0 += w_d0 * a_d * float(isum0);

            if (row0 + 1 < ne0) {
                uint w_off1 = src0_row1 + block_idx * Q8_0_BSIZE;
                float w_d1 = read_f16_q80(src0, w_off1);
                uint w_packed1 = read_u32_q80(src0, w_off1 + 2 + l0);
                int isum1 = 0;
                isum1 = dot4add_i8packed(w_packed1, a_packed, isum1);
                acc1 += w_d1 * a_d * float(isum1);
            }
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
