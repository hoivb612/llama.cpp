// mul_mat_vec_q6k.hlsl - Cooperative matrix-vector multiply for Q6_K weights (M=1)
//
// 16 threads cooperate per Q6_K superblock. Each thread processes one
// 16-element scale group using vectorized uint32 loads (4× fewer loads
// than per-element scalar byte access).
//
// Q6_K block layout (210 bytes per 256 elements):
//   ql[128]: low 4 bits (2 nibbles/byte)
//   qh[64]:  high 2 bits (4 values/byte)
//   scales[16]: int8 per group of 16 elements
//   d (f16): superblock scale
//
// Dispatch: groups_x = N, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE        256
#define QK_K              256
#define Q6K_BSIZE         210
#define THREADS_PER_BLOCK 16
#define BLOCKS_PER_GROUP  (GROUP_SIZE / THREADS_PER_BLOCK)

groupshared float shared_acc[GROUP_SIZE];

// Misalignment-safe uint32 load (Q6_K block size 210 is not 4-byte aligned)
uint load_u32(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i0 = group_id.y * 65535u + group_id.x;  // linearized 2D for large N (>65535)
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    uint lane = tid % THREADS_PER_BLOCK;
    uint block_group = tid / THREADS_PER_BLOCK;
    uint num_blocks = K / QK_K;

    // Each lane handles one of 16 scale groups (16 elements each)
    uint ip = lane / 8;              // half-index: 0 (elements 0-127) or 1 (128-255)
    uint il_base = (lane % 8) * 16;  // offset within 128-element half

    // Pre-compute per-lane offsets within block (constant across iterations)
    uint ql_rel = 64 * ip + (il_base % 64);
    uint ql_use_high = (il_base >= 64) ? 1u : 0u;
    uint qh_rel = 128 + 32 * ip + (il_base % 32);
    uint qh_shift = (il_base / 32) * 2;
    uint sc_rel = 192 + ip * 8 + (lane % 8);
    uint elem_start = ip * 128 + il_base;

    precise float acc = 0.0f;

    for (uint bg = block_group; bg < num_blocks; bg += BLOCKS_PER_GROUP) {
        uint block_off = src0_row + bg * Q6K_BSIZE;

        // Load superblock scale d (f16 at byte 208)
        uint d_off = block_off + 208;
        uint d_word = src0.Load(d_off & ~3u);
        float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

        // Load group scale (int8)
        uint sc_off = block_off + sc_rel;
        uint sc_word = src0.Load(sc_off & ~3u);
        uint sc_byte = (sc_word >> ((sc_off & 3u) * 8u)) & 0xFFu;
        int scale = (sc_byte < 128) ? (int)sc_byte : (int)sc_byte - 256;
        float ds = d * (float)scale;

        // Load 16 ql bytes as 4 × uint32 (misalignment-safe)
        uint ql_off = block_off + ql_rel;
        uint ql0 = load_u32(src0, ql_off);
        uint ql1 = load_u32(src0, ql_off + 4);
        uint ql2 = load_u32(src0, ql_off + 8);
        uint ql3 = load_u32(src0, ql_off + 12);

        // Load 16 qh bytes as 4 × uint32 (misalignment-safe)
        uint qh_off = block_off + qh_rel;
        uint qh0 = load_u32(src0, qh_off);
        uint qh1 = load_u32(src0, qh_off + 4);
        uint qh2 = load_u32(src0, qh_off + 8);
        uint qh3 = load_u32(src0, qh_off + 12);

        // Load 16 activations as 4 × Load4 (F32, always aligned)
        uint y_off = src1_base + (bg * QK_K + elem_start) * 4;
        uint4 a0 = src1.Load4(y_off);
        uint4 a1 = src1.Load4(y_off + 16);
        uint4 a2 = src1.Load4(y_off + 32);
        uint4 a3 = src1.Load4(y_off + 48);

        // Dequantize 16 elements and dot with activations
        float sum16 = 0.0f;

        [unroll] for (uint e = 0; e < 4; e++) {
            uint qlb = (ql0 >> (e * 8)) & 0xFFu;
            uint qhb = (qh0 >> (e * 8)) & 0xFFu;
            uint bits = ql_use_high ? (qlb >> 4) : (qlb & 0x0Fu);
            int q = (int)(bits | (((qhb >> qh_shift) & 3u) << 4)) - 32;
            sum16 = mad((float)q, asfloat(a0[e]), sum16);
        }
        [unroll] for (uint e = 0; e < 4; e++) {
            uint qlb = (ql1 >> (e * 8)) & 0xFFu;
            uint qhb = (qh1 >> (e * 8)) & 0xFFu;
            uint bits = ql_use_high ? (qlb >> 4) : (qlb & 0x0Fu);
            int q = (int)(bits | (((qhb >> qh_shift) & 3u) << 4)) - 32;
            sum16 = mad((float)q, asfloat(a1[e]), sum16);
        }
        [unroll] for (uint e = 0; e < 4; e++) {
            uint qlb = (ql2 >> (e * 8)) & 0xFFu;
            uint qhb = (qh2 >> (e * 8)) & 0xFFu;
            uint bits = ql_use_high ? (qlb >> 4) : (qlb & 0x0Fu);
            int q = (int)(bits | (((qhb >> qh_shift) & 3u) << 4)) - 32;
            sum16 = mad((float)q, asfloat(a2[e]), sum16);
        }
        [unroll] for (uint e = 0; e < 4; e++) {
            uint qlb = (ql3 >> (e * 8)) & 0xFFu;
            uint qhb = (qh3 >> (e * 8)) & 0xFFu;
            uint bits = ql_use_high ? (qlb >> 4) : (qlb & 0x0Fu);
            int q = (int)(bits | (((qhb >> qh_shift) & 3u) << 4)) - 32;
            sum16 = mad((float)q, asfloat(a3[e]), sum16);
        }

        acc += ds * sum16;
    }

    // Wave-intrinsic reduction then cross-wave via shared memory
    float wave_sum = WaveActiveSum(acc);
    uint wave_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    uint num_waves = GROUP_SIZE / WaveGetLaneCount();
    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float result = shared_acc[0];
        if (op0 == 1u) result += asfloat(src2.Load(op1 + i0 * 4));
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}