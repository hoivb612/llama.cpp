// mul_mat_vec_q2k_mr.hlsl - Multi-row Q2_K matvec (M=1, 2 rows/group)
//
// Q2_K block (84 bytes): scales[16] + qs[64] + d(f16) + dmin(f16)
// 256 threads, per-element dequant, shares activation loads across 2 rows.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q2K_BSIZE   84

groupshared float shared_acc[64];

uint read_byte_q2k_mr(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float dequant_q2k_element(uint src0_row, uint k) {
    uint blk  = k / QK_K;
    uint elem = k % QK_K;
    uint block_off = src0_row + blk * Q2K_BSIZE;

    // d, dmin packed into the same 4-byte word at offset 80
    uint dm = src0.Load(block_off + 80);
    float d    = f16_to_f32(dm & 0xFFFFu);
    float dmin = f16_to_f32(dm >> 16);

    uint n      = elem >> 7;        // 0 or 1
    uint y_in_n = elem & 127u;
    uint j      = y_in_n >> 5;      // 0..3
    uint pos    = y_in_n & 31u;
    uint l      = pos & 15u;
    uint half   = pos >> 4;
    uint shift  = j << 1;
    uint is     = (n << 3) + (j << 1) + half;
    uint qs_off = (n << 5) + (half << 4) + l;

    uint sc_byte = read_byte_q2k_mr(src0, block_off + is);
    float scale  = d    * float(sc_byte & 0x0Fu);
    float ml     = dmin * float(sc_byte >> 4);

    uint qb = read_byte_q2k_mr(src0, block_off + 16u + qs_off);
    int q   = int((qb >> shift) & 3u);

    return scale * float(q) - ml;
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
        acc0 += dequant_q2k_element(src0_row0, k) * x;
        acc1 += dequant_q2k_element(src0_row1, k) * x;
    }

    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id]      = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid]      += shared_acc[tid + s];
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
