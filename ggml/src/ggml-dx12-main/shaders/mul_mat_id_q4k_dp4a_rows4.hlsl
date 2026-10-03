// Q4_K expert matvec with Q8_1 activations and four output rows per group.
// A lane keeps one activation slice and applies it to four weight rows.
#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define NUM_ROWS   4
#define QK_K       256
#define Q4K_BSIZE  144
#define Q8_1_BSIZE 36

groupshared float shared_acc[NUM_ROWS * GROUP_SIZE];

void decode_sc_mb(uint s0, uint s4, uint s8, uint j, out float sc, out float mb) {
    if (j < 4u) {
        uint shift = 8u * j;
        sc = float((s0 >> shift) & 0x3Fu);
        mb = float((s4 >> shift) & 0x3Fu);
    } else {
        uint shift = 8u * (j - 4u);
        sc = float(((s8 >> shift) & 0x0Fu) | (((s0 >> (shift + 6u)) & 0x03u) << 4u));
        mb = float(((s8 >> (shift + 4u)) & 0x0Fu) | (((s4 >> (shift + 6u)) & 0x03u) << 4u));
    }
}

float dot_q4_row(uint block_off, uint il,
                 float a_d_lo, float a_s_lo, float a_d_hi, float a_s_hi,
                 uint4 a_lo0, uint4 a_lo1, uint4 a_hi0, uint4 a_hi1) {
    uint dm_raw = src0.Load(block_off);
    float dall = f16_to_f32(dm_raw & 0xFFFFu);
    float dmin = f16_to_f32(dm_raw >> 16);
    uint s0 = src0.Load(block_off + 4u);
    uint s4 = src0.Load(block_off + 8u);
    uint s8 = src0.Load(block_off + 12u);

    uint j_lo = il * 2u;
    uint j_hi = j_lo + 1u;
    float sc_lo, mb_lo, sc_hi, mb_hi;
    decode_sc_mb(s0, s4, s8, j_lo, sc_lo, mb_lo);
    decode_sc_mb(s0, s4, s8, j_hi, sc_hi, mb_hi);

    uint qs_grp = 16u + il * 32u;
    uint4 qs0 = src0.Load4(block_off + qs_grp);
    uint4 qs1 = src0.Load4(block_off + qs_grp + 16u);

    int sum_lo = 0;
    sum_lo = dot4add_i8packed(qs0.x & 0x0F0F0F0Fu, a_lo0.x, sum_lo);
    sum_lo = dot4add_i8packed(qs0.y & 0x0F0F0F0Fu, a_lo0.y, sum_lo);
    sum_lo = dot4add_i8packed(qs0.z & 0x0F0F0F0Fu, a_lo0.z, sum_lo);
    sum_lo = dot4add_i8packed(qs0.w & 0x0F0F0F0Fu, a_lo0.w, sum_lo);
    sum_lo = dot4add_i8packed(qs1.x & 0x0F0F0F0Fu, a_lo1.x, sum_lo);
    sum_lo = dot4add_i8packed(qs1.y & 0x0F0F0F0Fu, a_lo1.y, sum_lo);
    sum_lo = dot4add_i8packed(qs1.z & 0x0F0F0F0Fu, a_lo1.z, sum_lo);
    sum_lo = dot4add_i8packed(qs1.w & 0x0F0F0F0Fu, a_lo1.w, sum_lo);

    int sum_hi = 0;
    sum_hi = dot4add_i8packed((qs0.x >> 4u) & 0x0F0F0F0Fu, a_hi0.x, sum_hi);
    sum_hi = dot4add_i8packed((qs0.y >> 4u) & 0x0F0F0F0Fu, a_hi0.y, sum_hi);
    sum_hi = dot4add_i8packed((qs0.z >> 4u) & 0x0F0F0F0Fu, a_hi0.z, sum_hi);
    sum_hi = dot4add_i8packed((qs0.w >> 4u) & 0x0F0F0F0Fu, a_hi0.w, sum_hi);
    sum_hi = dot4add_i8packed((qs1.x >> 4u) & 0x0F0F0F0Fu, a_hi1.x, sum_hi);
    sum_hi = dot4add_i8packed((qs1.y >> 4u) & 0x0F0F0F0Fu, a_hi1.y, sum_hi);
    sum_hi = dot4add_i8packed((qs1.z >> 4u) & 0x0F0F0F0Fu, a_hi1.z, sum_hi);
    sum_hi = dot4add_i8packed((qs1.w >> 4u) & 0x0F0F0F0Fu, a_hi1.w, sum_hi);

    float dot_term = mad(sc_lo * a_d_lo, float(sum_lo), sc_hi * a_d_hi * float(sum_hi));
    float min_term = mad(mb_lo, a_s_lo, mb_hi * a_s_hi);
    return dall * dot_term - dmin * min_term;
}

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

    uint num_blocks = ne00 / QK_K;
    uint num_q8 = ne00 / 32u;
    uint expert_src1 = expert_slot % ne11;
    uint q8_vec_idx = (batch * ne12 + token) * ne11 + expert_src1;
    uint q8_vec_base = src1_offset + q8_vec_idx * num_q8 * Q8_1_BSIZE;

    uint il = tid & 3u;
    uint block_slot = tid >> 2u;
    uint j_lo = il * 2u;
    uint j_hi = j_lo + 1u;
    precise float acc[NUM_ROWS];
    [unroll] for (uint r = 0u; r < NUM_ROWS; ++r) {
        acc[r] = 0.0f;
    }

    for (uint block_idx = block_slot; block_idx < num_blocks; block_idx += GROUP_SIZE / 4u) {
        uint q8_super = q8_vec_base + block_idx * 8u * Q8_1_BSIZE;
        uint q8_off_lo = q8_super + j_lo * Q8_1_BSIZE;
        uint q8_off_hi = q8_super + j_hi * Q8_1_BSIZE;
        uint ds_lo = src1.Load(q8_off_lo);
        uint ds_hi = src1.Load(q8_off_hi);
        float a_d_lo = f16_to_f32(ds_lo & 0xFFFFu);
        float a_s_lo = f16_to_f32(ds_lo >> 16);
        float a_d_hi = f16_to_f32(ds_hi & 0xFFFFu);
        float a_s_hi = f16_to_f32(ds_hi >> 16);
        uint4 a_lo0 = src1.Load4(q8_off_lo + 4u);
        uint4 a_lo1 = src1.Load4(q8_off_lo + 20u);
        uint4 a_hi0 = src1.Load4(q8_off_hi + 4u);
        uint4 a_hi1 = src1.Load4(q8_off_hi + 20u);

        [unroll] for (uint r2 = 0u; r2 < NUM_ROWS; ++r2) {
            uint row = row0 + r2;
            if (row < ne0) {
                uint block_off = src0_base + row * nb01 + block_idx * Q4K_BSIZE;
                acc[r2] += dot_q4_row(block_off, il,
                                      a_d_lo, a_s_lo, a_d_hi, a_s_hi,
                                      a_lo0, a_lo1, a_hi0, a_hi1);
            }
        }
    }

    [unroll] for (uint r3 = 0u; r3 < NUM_ROWS; ++r3) {
        shared_acc[r3 * GROUP_SIZE + tid] = acc[r3];
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < NUM_ROWS) {
        uint row = row0 + tid;
        if (row < ne0) {
            float result = 0.0f;
            for (uint t = 0u; t < GROUP_SIZE; ++t) {
                result += shared_acc[tid * GROUP_SIZE + t];
            }
            if (op15 != 0u) {
                result *= asfloat(src3.Load(expert_slot * op3 + token * op4 + batch * op5));
            }
            uint off = offset_4d(row, expert_slot, token, batch, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off, result, dst_esize);
        }
    }
}
