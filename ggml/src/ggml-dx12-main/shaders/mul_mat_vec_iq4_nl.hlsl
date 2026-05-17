// mul_mat_vec_iq4_nl.hlsl - Specialized matrix-vector multiply for IQ4_NL weights (M=1)
//
// IQ4_NL block: d(f16) + qs[16] = 18 bytes per 32 elements
// Each qs byte holds two 4-bit indices; index -> non-linear codebook value.
// 32 threads cooperate via WaveActiveSum
//
// Dispatch: groups_x = N (output rows), groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK4_NL 32
#define IQ4_NL_BSIZE 18

groupshared float shared_acc[GROUP_SIZE];

// Non-linear 4-bit codebook for IQ4_NL (matches kvalues_iq4nl in ggml-common.h)
// Packed as 16 signed bytes into 4 uint32. Values fit in int8: -127..113.
//   { -127, -104, -83, -65, -49, -35, -22, -10,
//        1,   13,  25,  38,  53,  69,  89, 113 }
// As uint8: { 0x81, 0x98, 0xAD, 0xBF, 0xCF, 0xDD, 0xEA, 0xF6,
//             0x01, 0x0D, 0x19, 0x26, 0x35, 0x45, 0x59, 0x71 }
int kvalues_iq4nl(uint idx) {
    static const uint packed[4] = {
        0xBFAD9881u,  // bytes 0..3:  -127, -104, -83, -65
        0xF6EADDCFu,  // bytes 4..7:   -49,  -35, -22, -10
        0x26190D01u,  // bytes 8..11:    1,   13,  25,  38
        0x71594535u   // bytes 12..15:  53,   69,  89, 113
    };
    uint w = packed[idx >> 2];
    uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    // sign-extend 8-bit -> 32-bit
    return (int)(b << 24) >> 24;
}

uint read_u32_fast(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_v(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint i0 = group_x_2d(group_id);
    if (i0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK4_NL;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;

    uint elem = local_id;  // 0..31

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * IQ4_NL_BSIZE;

        float d = read_f16_v(src0, block_off);

        // qs starts at block_off + 2; 16 bytes total. Each thread reads one byte.
        // Threads 0..15 use low nibble (k = block*32 + j), threads 16..31 use high nibble (k = block*32 + 16 + j)
        uint qs_idx = (elem < 16) ? elem : (elem - 16);
        uint qs_word = read_u32_fast(src0, block_off + 2 + (qs_idx & ~3u));
        uint qs_byte = (qs_word >> ((qs_idx & 3u) * 8u)) & 0xFFu;

        int val;
        if (elem < 16) {
            val = kvalues_iq4nl(qs_byte & 0x0Fu);
        } else {
            val = kvalues_iq4nl((qs_byte >> 4) & 0x0Fu);
        }

        float w = d * (float)val;
        uint k = block * QK4_NL + elem;
        float x = asfloat(src1.Load(src1_base + k * 4));
        acc += w * x;
    }

    // Two-level wave + shared memory reduction (correct for any wave size)
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
