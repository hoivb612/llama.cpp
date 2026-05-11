// mul_mat_vec_q6k_mr.hlsl - Multi-row Q6_K matvec (M=1, 2 rows/group)
//
// Q6_K block layout (210 bytes): ql[128] + qh[64] + scales[16] + d(f16)
// 256 threads, per-element dequant, shares activation loads across 2 rows.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q6K_BSIZE   210

groupshared float shared_acc[64];

uint read_byte_q6(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float dequant_q6k_element(uint src0_row, uint k) {
    uint block_idx = k / QK_K;
    uint elem = k % QK_K;
    uint block_off = src0_row + block_idx * Q6K_BSIZE;

    uint d_off = block_off + 208;
    uint d_word = src0.Load(d_off & ~3u);
    float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

    uint ip = elem / 128;
    uint il = elem % 128;

    uint sc_byte = read_byte_q6(src0, block_off + 192 + 8 * ip + il / 16);
    int scale = (sc_byte < 128) ? (int)sc_byte : (int)sc_byte - 256;

    uint ql_val = read_byte_q6(src0, block_off + 64 * ip + (il % 64));
    uint qh_val = read_byte_q6(src0, block_off + 128 + 32 * ip + (il % 32));

    uint sub32 = il / 32;
    uint ql_bits = (il < 64) ? (ql_val & 0x0Fu) : (ql_val >> 4);
    int q = (int)(ql_bits | (((qh_val >> (sub32 * 2)) & 3u) << 4)) - 32;

    return d * float(scale) * float(q);
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    for (uint k = tid; k < K; k += GROUP_SIZE) {
        float x = asfloat(src1.Load(src1_base + k * 4));
        acc0 += dequant_q6k_element(src0_row0, k) * x;
        acc1 += dequant_q6k_element(src0_row1, k) * x;
    }

    // Tree reduction
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
            shared_acc[32 + tid] += shared_acc[32 + tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float result0 = shared_acc[0];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[32];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
