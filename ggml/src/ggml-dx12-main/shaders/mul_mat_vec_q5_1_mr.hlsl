// mul_mat_vec_q5_1_mr.hlsl - Multi-row Q5_1 matvec (M=1, 2 rows/group)
//
// Q5_1 block: d(f16) + m(f16) + qh[4] + qs[16] = 24 bytes per 32 elements.
// Dequant: x = (unsigned 5-bit value) * d + m
//
// 32 threads (1 wave), shares activation loads across 2 rows.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch * ne2 * ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK5_1 32
#define Q5_1_BSIZE 24

groupshared float shared_acc[64];

uint q51mr_read_u32_fast(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float q51mr_read_f16(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

float q51mr_dequant(ByteAddressBuffer buf, uint block_off, uint elem) {
    float d = q51mr_read_f16(buf, block_off);
    float m = q51mr_read_f16(buf, block_off + 2);
    uint qh = q51mr_read_u32_fast(buf, block_off + 4);

    uint qs_idx = (elem < 16) ? elem : (elem - 16);
    uint qs_word = q51mr_read_u32_fast(buf, block_off + 8 + (qs_idx & ~3u));
    uint qs_byte = (qs_word >> ((qs_idx & 3u) * 8u)) & 0xFFu;

    uint val_u;
    if (elem < 16) {
        uint xh = ((qh >> elem) << 4) & 0x10u;
        val_u = (qs_byte & 0x0Fu) | xh;
    } else {
        uint jj = elem - 16;
        uint xh = ((qh >> (jj + 12)) & 0x10u);
        val_u = (qs_byte >> 4) | xh;
    }
    return float(val_u) * d + m;
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK5_1;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    uint elem = local_id;

    for (uint block = 0; block < num_blocks; block++) {
        uint k = block * QK5_1 + elem;
        float x = asfloat(src1.Load(src1_base + k * 4));

        float w0 = q51mr_dequant(src0, src0_row0 + block * Q5_1_BSIZE, elem);
        acc0 += w0 * x;

        if (row0 + 1 < ne0) {
            float w1 = q51mr_dequant(src0, src0_row1 + block * Q5_1_BSIZE, elem);
            acc1 += w1 * x;
        }
    }

    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = local_id / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s /= 2) {
        if (local_id < s) {
            shared_acc[local_id] += shared_acc[local_id + s];
            shared_acc[32 + local_id] += shared_acc[32 + local_id + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
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
