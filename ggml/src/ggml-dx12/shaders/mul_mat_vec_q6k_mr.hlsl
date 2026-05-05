// mul_mat_vec_q6k_mr.hlsl - Multi-row Q6_K matvec (2 rows per group)
//
// Processes 2 output rows per thread group, sharing activation loads.
// Uses the simple per-element approach (like q6k_32) but with 256 threads
// and multi-row to amortize activation bandwidth.
//
// Dispatch: groups_x = ceil(N/2), groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q6K_BSIZE   210
#define N_ROWS      2

groupshared float shared_acc[GROUP_SIZE];

uint read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row_base = (group_id.y * 65535u + group_id.x) * N_ROWS;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (row_base >= ne0) return;
    bool valid_row1 = (row_base + 1 < ne0);

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint src0_row0 = src0_offset + row_base * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row1 = src0_row0 + nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Each thread processes elements k = tid, tid+256, tid+512, ...
    for (uint k = tid; k < K; k += GROUP_SIZE) {
        // Load activation once (shared between rows)
        float x = asfloat(src1.Load(src1_base + k * 4));

        uint block_idx = k / QK_K;
        uint elem = k % QK_K;

        // Row 0
        {
            uint block_off = src0_row0 + block_idx * Q6K_BSIZE;
            uint d_off = block_off + 208;
            uint d_word = src0.Load(d_off & ~3u);
            float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

            uint ip = elem / 128;
            uint il = elem % 128;

            uint sc_byte = read_byte(src0, block_off + 192 + 8 * ip + il / 16);
            int scale = (sc_byte < 128) ? (int)sc_byte : (int)sc_byte - 256;

            uint ql_val = read_byte(src0, block_off + 64 * ip + (il % 64));
            uint qh_val = read_byte(src0, block_off + 128 + 32 * ip + (il % 32));

            uint sub32 = il / 32;
            uint ql_bits = (il < 64) ? (ql_val & 0x0Fu) : (ql_val >> 4);
            int q = (int)(ql_bits | (((qh_val >> (sub32 * 2)) & 3u) << 4)) - 32;

            acc0 += d * (float)scale * (float)q * x;
        }

        // Row 1 (reuse activation x)
        if (valid_row1) {
            uint block_off = src0_row1 + block_idx * Q6K_BSIZE;
            uint d_off = block_off + 208;
            uint d_word = src0.Load(d_off & ~3u);
            float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

            uint ip = elem / 128;
            uint il = elem % 128;

            uint sc_byte = read_byte(src0, block_off + 192 + 8 * ip + il / 16);
            int scale = (sc_byte < 128) ? (int)sc_byte : (int)sc_byte - 256;

            uint ql_val = read_byte(src0, block_off + 64 * ip + (il % 64));
            uint qh_val = read_byte(src0, block_off + 128 + 32 * ip + (il % 32));

            uint sub32 = il / 32;
            uint ql_bits = (il < 64) ? (ql_val & 0x0Fu) : (ql_val >> 4);
            int q = (int)(ql_bits | (((qh_val >> (sub32 * 2)) & 3u) << 4)) - 32;

            acc1 += d * (float)scale * (float)q * x;
        }
    }

    // Reduction for row 0
    float wave_sum = WaveActiveSum(acc0);
    uint wave_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    uint num_waves = GROUP_SIZE / WaveGetLaneCount();
    if (tid < num_waves) {
        float v = shared_acc[tid];
        v = WaveActiveSum(v);
        if (tid == 0) shared_acc[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float result = shared_acc[0];
        if (op0 == 1u) result += asfloat(src2.Load(op1 + row_base * 4));
        uint off_d = offset_4d(row_base, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }

    // Reduction for row 1
    if (valid_row1) {
        wave_sum = WaveActiveSum(acc1);
        if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
        GroupMemoryBarrierWithGroupSync();

        if (tid < num_waves) {
            float v = shared_acc[tid];
            v = WaveActiveSum(v);
            if (tid == 0) shared_acc[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid == 0) {
            float result = shared_acc[0];
            if (op0 == 1u) result += asfloat(src2.Load(op1 + (row_base + 1) * 4));
            uint off_d = offset_4d(row_base + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d, result, dst_esize);
        }
    }
}
