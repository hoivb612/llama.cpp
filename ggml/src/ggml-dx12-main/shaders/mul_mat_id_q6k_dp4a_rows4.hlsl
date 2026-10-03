// Q6_K expert matvec using packed Q6_K loads and four output rows per group.
#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define NUM_ROWS   4
#define QK_K       256
#define Q6K_BSIZE  210
#define Q8_1_BSIZE 36

groupshared float shared_acc[NUM_ROWS * GROUP_SIZE];

uint load_u32_unaligned(ByteAddressBuffer buf, uint byte_off) {
    uint base = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(base);
    if (shift == 0u) {
        return lo;
    }
    uint hi = buf.Load(base + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

int load_s8(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    uint value = (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
    return value < 128u ? (int)value : (int)value - 256;
}

float load_f16_unaligned(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

void load_q8_chunk(uint q8_super, uint block_idx, uint byte_idx,
                   out float d, out uint qs, out int sum) {
    uint off = q8_super + block_idx * Q8_1_BSIZE;
    d = f16_to_f32(src1.Load(off) & 0xFFFFu);
    qs = src1.Load(off + 4u + byte_idx);
    sum = 0;
    sum = dot4add_i8packed(0x01010101u, qs, sum);
}

float dot_q6_row(uint block_off, uint v_im, uint v_in,
                 float d0, float d1, float d2, float d3,
                 uint a0, uint a1, uint a2, uint a3,
                 int asum0, int asum1, int asum2, int asum3) {
    uint l0 = 4u * v_in;
    uint ql0 = load_u32_unaligned(src0, block_off + 64u * v_im + l0);
    uint ql1 = load_u32_unaligned(src0, block_off + 64u * v_im + l0 + 32u);
    uint qh = load_u32_unaligned(src0, block_off + 128u + 32u * v_im + l0);

    uint q0 = (ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4u);
    uint q1 = (ql1 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2u);
    uint q2 = ((ql0 >> 4u) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
    uint q3 = ((ql1 >> 4u) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2u);

    int dot0 = 0; dot0 = dot4add_i8packed(q0, a0, dot0);
    int dot1 = 0; dot1 = dot4add_i8packed(q1, a1, dot1);
    int dot2 = 0; dot2 = dot4add_i8packed(q2, a2, dot2);
    int dot3 = 0; dot3 = dot4add_i8packed(q3, a3, dot3);

    uint scale_base = block_off + 192u + 8u * v_im + v_in / 4u;
    float s0 = float(load_s8(src0, scale_base));
    float s1 = float(load_s8(src0, scale_base + 2u));
    float s2 = float(load_s8(src0, scale_base + 4u));
    float s3 = float(load_s8(src0, scale_base + 6u));
    float d = load_f16_unaligned(src0, block_off + 208u);

    float sum = s0 * d0 * float(dot0 - 32 * asum0);
    sum = mad(s1 * d1, float(dot1 - 32 * asum1), sum);
    sum = mad(s2 * d2, float(dot2 - 32 * asum2), sum);
    sum = mad(s3 * d3, float(dot3 - 32 * asum3), sum);
    return d * sum;
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

    uint itid = tid & 15u;
    uint block_slot = tid >> 4u;
    uint block_stride = GROUP_SIZE / 16u;
    uint v_im = itid >> 3u;
    uint v_in = itid & 7u;
    uint byte_idx = 4u * v_in;
    uint q8_block0 = 4u * v_im;

    precise float acc[NUM_ROWS];
    [unroll] for (uint r = 0u; r < NUM_ROWS; ++r) {
        acc[r] = 0.0f;
    }

    for (uint block_idx = block_slot; block_idx < num_blocks; block_idx += block_stride) {
        uint q8_super = q8_vec_base + block_idx * 8u * Q8_1_BSIZE;
        float d0, d1, d2, d3;
        uint a0, a1, a2, a3;
        int asum0, asum1, asum2, asum3;
        load_q8_chunk(q8_super, q8_block0 + 0u, byte_idx, d0, a0, asum0);
        load_q8_chunk(q8_super, q8_block0 + 1u, byte_idx, d1, a1, asum1);
        load_q8_chunk(q8_super, q8_block0 + 2u, byte_idx, d2, a2, asum2);
        load_q8_chunk(q8_super, q8_block0 + 3u, byte_idx, d3, a3, asum3);

        [unroll] for (uint r2 = 0u; r2 < NUM_ROWS; ++r2) {
            uint row = row0 + r2;
            if (row < ne0) {
                uint block_off = src0_base + row * nb01 + block_idx * Q6K_BSIZE;
                acc[r2] += dot_q6_row(block_off, v_im, v_in,
                                      d0, d1, d2, d3, a0, a1, a2, a3,
                                      asum0, asum1, asum2, asum3);
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
