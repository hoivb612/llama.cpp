// mul_mat_vec_mxfp4.hlsl - Specialized matrix-vector multiply for MXFP4 weights (M=1)
//
// MXFP4 block: e(E8M0, 1 byte) + qs[16] = 17 bytes per 32 elements.
// Each qs byte holds two 4-bit indices into the E2M1 codebook; kvalues_fp4
// stores 2x the real values, so the scale folds in a 0.5.
// 32 threads cooperate via WaveActiveSum.
//
// Dispatch: groups_x = N (output rows), groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK_MXFP4 32
#define MXFP4_BSIZE 17

groupshared float shared_acc[GROUP_SIZE];

// { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 } as signed bytes
int kvalues_fp4(uint idx) {
    static const uint packed[4] = {
        0x03020100u,  // 0, 1, 2, 3
        0x0C080604u,  // 4, 6, 8, 12
        0xFDFEFF00u,  // 0, -1, -2, -3
        0xF4F8FAFCu   // -4, -6, -8, -12
    };
    uint w = packed[idx >> 2];
    uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    return (int)(b << 24) >> 24;
}

// 0.5 * 2^(e-127), matching GGML_E8M0_TO_FP32_HALF.
float e8m0_half(uint e) {
    return asfloat((e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23));
}

uint read_u32_fast(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    uint hi = buf.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (hi << (32u - shift));
}

uint read_byte_v(ByteAddressBuffer buf, uint byte_off) {
    return (buf.Load(byte_off & ~3u) >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint i0 = group_x_2d(group_id);
    if (i0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_MXFP4;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;

    uint elem = local_id;  // 0..31

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * MXFP4_BSIZE;

        float d = e8m0_half(read_byte_v(src0, block_off));

        uint qs_idx  = (elem < 16) ? elem : (elem - 16);
        uint qs_word = read_u32_fast(src0, block_off + 1 + (qs_idx & ~3u));
        uint qs_byte = (qs_word >> ((qs_idx & 3u) * 8u)) & 0xFFu;

        uint nib = (elem < 16) ? (qs_byte & 0x0Fu) : ((qs_byte >> 4) & 0x0Fu);

        float w = d * (float)kvalues_fp4(nib);
        uint k = block * QK_MXFP4 + elem;
        float x = asfloat(src1.Load(src1_base + k * 4));
        acc += w * x;
    }

    float wave_sum = WaveActiveSum(acc);
    uint wave_id = local_id / WARP_SIZE;
    if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    uint num_waves = GROUP_SIZE / WARP_SIZE;
    if (num_waves <= WARP_SIZE) {
        if (local_id < num_waves) {
            float v = shared_acc[local_id];
            v = WaveActiveSum(v);
            if (local_id == 0) shared_acc[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
    } else {
        for (uint s = num_waves / 2; s > 0; s /= 2) {
            if (local_id < s) shared_acc[local_id] += shared_acc[local_id + s];
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (local_id == 0) {
        float result = shared_acc[0];
        result += load_fused_bias(i0, i2, i3);
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}
