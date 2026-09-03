// mul_mat_id_q4k_dp4a.hlsl - Q4_K expert matvec with Q8_1 activations.
//
// The MoE counterpart of mul_mat_vec_q4k_dp4a.hlsl. Reuses the Q8_1 quantize
// pre-pass and scratch driven by use_dp4a_matvec, so the kernel is the only
// new piece.
//
// Q4_K superblock (144 bytes): d/dmin (2 x f16), 12 bytes of packed 6-bit
// scales and mins, then 128 bytes of nibbles. Sub-block j (32 elements) covers
// k in [j*32, j*32+32); its nibbles live in the 32-byte group il = j/2, low
// half for even j and high half for odd j. QK8_1 is also 32, so sub-block j
// maps 1:1 onto Q8_1 block j and one qs word feeds two dp4a lanes.
//
// One thread owns a whole 32-byte group, i.e. two complete sub-blocks. That
// amortises the superblock scale decode over 16 dp4a instead of 2, lets both
// qs and activations come in as Load4, and - because the thread spans the full
// sub-block - lets the min term use the Q8_1 's' field (d * sum(q)) directly
// instead of reconstructing the activation sum with extra dp4a.
//
// Lane layout: il = tid%4, row_sel = (tid/4)%NUM_ROWS, block slot = tid/8.
// Splitting rows across lanes (rather than having every thread do both rows)
// keeps the group busy when K is small: granite-a400m gate/up is K=1024, i.e.
// only 4 superblocks, which would otherwise idle most of the group.
#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  32
#endif
#define QK_K        256
#define Q4K_BSIZE   144
#define Q8_1_BSIZE  36
#define NUM_ROWS    2
#define BLOCKS_PER_ITER (GROUP_SIZE / (4 * NUM_ROWS))

groupshared float shared_acc[GROUP_SIZE];

void decode_sc_mb(uint s0, uint s4, uint s8, uint j, out float sc, out float mb) {
    if (j < 4) {
        uint sh = 8u * j;
        sc = float((s0 >> sh) & 0x3Fu);
        mb = float((s4 >> sh) & 0x3Fu);
    } else {
        uint sh = 8u * (j - 4u);
        sc = float(((s8 >> sh) & 0x0Fu) | (((s0 >> (sh + 6u)) & 0x03u) << 4u));
        mb = float(((s8 >> (sh + 4u)) & 0x0Fu) | (((s4 >> (sh + 6u)) & 0x03u) << 4u));
    }
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

    uint K = ne00;
    uint num_blocks = K / QK_K;
    uint num_q8 = K / 32;

    uint expert_src1 = expert_slot % ne11;
    uint q8_vec_idx = (batch * ne12 + token) * ne11 + expert_src1;
    uint q8_vec_base = src1_offset + q8_vec_idx * num_q8 * Q8_1_BSIZE;

    uint il      = tid % 4u;
    uint row_sel = (tid / 4u) % NUM_ROWS;
    uint slot    = tid / (4u * NUM_ROWS);

    uint row  = row0 + row_sel;
    bool live = row < ne0;
    uint row_base = src0_base + row * nb01;

    uint j_lo = il * 2u;
    uint j_hi = j_lo + 1u;
    uint qs_grp = 16u + il * 32u;

    precise float acc = 0.0f;

    for (uint block_iter = 0; block_iter < num_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block_idx = block_iter + slot;
        if (live && block_idx < num_blocks) {
            uint q8_super  = q8_vec_base + block_idx * 8u * Q8_1_BSIZE;
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

            uint block_off = row_base + block_idx * Q4K_BSIZE;
            uint dm_raw = src0.Load(block_off);
            float dall = f16_to_f32(dm_raw & 0xFFFFu);
            float dmin = f16_to_f32(dm_raw >> 16);

            uint s0 = src0.Load(block_off + 4);
            uint s4 = src0.Load(block_off + 8);
            uint s8 = src0.Load(block_off + 12);

            float sc_lo, mb_lo, sc_hi, mb_hi;
            decode_sc_mb(s0, s4, s8, j_lo, sc_lo, mb_lo);
            decode_sc_mb(s0, s4, s8, j_hi, sc_hi, mb_hi);

            uint4 qs0 = src0.Load4(block_off + qs_grp);
            uint4 qs1 = src0.Load4(block_off + qs_grp + 16u);

            int isum_lo = 0;
            isum_lo = dot4add_i8packed(qs0.x & 0x0F0F0F0Fu, a_lo0.x, isum_lo);
            isum_lo = dot4add_i8packed(qs0.y & 0x0F0F0F0Fu, a_lo0.y, isum_lo);
            isum_lo = dot4add_i8packed(qs0.z & 0x0F0F0F0Fu, a_lo0.z, isum_lo);
            isum_lo = dot4add_i8packed(qs0.w & 0x0F0F0F0Fu, a_lo0.w, isum_lo);
            isum_lo = dot4add_i8packed(qs1.x & 0x0F0F0F0Fu, a_lo1.x, isum_lo);
            isum_lo = dot4add_i8packed(qs1.y & 0x0F0F0F0Fu, a_lo1.y, isum_lo);
            isum_lo = dot4add_i8packed(qs1.z & 0x0F0F0F0Fu, a_lo1.z, isum_lo);
            isum_lo = dot4add_i8packed(qs1.w & 0x0F0F0F0Fu, a_lo1.w, isum_lo);

            int isum_hi = 0;
            isum_hi = dot4add_i8packed((qs0.x >> 4) & 0x0F0F0F0Fu, a_hi0.x, isum_hi);
            isum_hi = dot4add_i8packed((qs0.y >> 4) & 0x0F0F0F0Fu, a_hi0.y, isum_hi);
            isum_hi = dot4add_i8packed((qs0.z >> 4) & 0x0F0F0F0Fu, a_hi0.z, isum_hi);
            isum_hi = dot4add_i8packed((qs0.w >> 4) & 0x0F0F0F0Fu, a_hi0.w, isum_hi);
            isum_hi = dot4add_i8packed((qs1.x >> 4) & 0x0F0F0F0Fu, a_hi1.x, isum_hi);
            isum_hi = dot4add_i8packed((qs1.y >> 4) & 0x0F0F0F0Fu, a_hi1.y, isum_hi);
            isum_hi = dot4add_i8packed((qs1.z >> 4) & 0x0F0F0F0Fu, a_hi1.z, isum_hi);
            isum_hi = dot4add_i8packed((qs1.w >> 4) & 0x0F0F0F0Fu, a_hi1.w, isum_hi);

            float dot_term = mad(sc_lo * a_d_lo, float(isum_lo), sc_hi * a_d_hi * float(isum_hi));
            float min_term = mad(mb_lo, a_s_lo, mb_hi * a_s_hi);
            acc += dall * dot_term - dmin * min_term;
        }
    }

    shared_acc[tid] = acc;
    GroupMemoryBarrierWithGroupSync();

    if (tid < NUM_ROWS) {
        uint out_row = row0 + tid;
        if (out_row < ne0) {
            float result = 0.0f;
            for (uint t = 0; t < GROUP_SIZE; ++t) {
                if ((t / 4u) % NUM_ROWS == tid) result += shared_acc[t];
            }
            uint off = offset_4d(out_row, expert_slot, token, batch, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off, result, dst_esize);
        }
    }
}
