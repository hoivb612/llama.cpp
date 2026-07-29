// mul_mat_vec_iq4_nl_mr.hlsl - Multi-row IQ4_NL matvec (M=1, 2 rows/group)
//
// IQ4_NL block: d(f16) + qs[16] = 18 bytes per 32 elements
// Each qs byte holds two 4-bit indices into the IQ4_NL non-linear codebook.
//
// 32 threads (1 wave on most HW), 2 rows per group, share the activation
// load across both rows. Halves dispatch count and src1 bandwidth vs the
// single-row mul_mat_vec_iq4_nl variant.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK4_NL 32
#define IQ4_NL_BSIZE 18

groupshared float shared_acc[64];

// Non-linear 4-bit codebook for IQ4_NL (matches kvalues_iq4nl in ggml-common.h)
//   { -127, -104, -83, -65, -49, -35, -22, -10,
//        1,   13,  25,  38,  53,  69,  89, 113 }
int kvalues_iq4nl(uint idx) {
    static const uint packed[4] = {
        0xBFAD9881u,
        0xF6EADDCFu,
        0x26190D01u,
        0x71594535u
    };
    uint w = packed[idx >> 2];
    uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
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

// Per-element decode for IQ4_NL given a pre-loaded qs byte.
int dequant_iq4nl(uint qs_byte, uint elem) {
    if (elem < 16) {
        return kvalues_iq4nl(qs_byte & 0x0Fu);
    }
    return kvalues_iq4nl((qs_byte >> 4) & 0x0Fu);
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
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
    uint num_blocks = K / QK4_NL;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    uint elem = local_id;
    uint qs_idx = (elem < 16) ? elem : (elem - 16);
    uint qs_word_off = qs_idx & ~3u;
    uint qs_byte_shift = (qs_idx & 3u) * 8u;

    for (uint block = 0; block < num_blocks; block++) {
        // Shared activation load (1 per element per group, reused for both rows)
        uint k = block * QK4_NL + elem;
        float x = asfloat(src1.Load(src1_base + k * 4));

        uint blk_off0 = src0_row0 + block * IQ4_NL_BSIZE;
        uint blk_off1 = src0_row1 + block * IQ4_NL_BSIZE;

        // Row 0
        float d0 = read_f16_v(src0, blk_off0);
        uint qs_word0 = read_u32_fast(src0, blk_off0 + 2 + qs_word_off);
        uint qs_byte0 = (qs_word0 >> qs_byte_shift) & 0xFFu;
        int val0 = dequant_iq4nl(qs_byte0, elem);
        acc0 += d0 * float(val0) * x;

        // Row 1
        float d1 = read_f16_v(src0, blk_off1);
        uint qs_word1 = read_u32_fast(src0, blk_off1 + 2 + qs_word_off);
        uint qs_byte1 = (qs_word1 >> qs_byte_shift) & 0xFFu;
        int val1 = dequant_iq4nl(qs_byte1, elem);
        acc1 += d1 * float(val1) * x;
    }

    // Stage 1: per-wave reduction
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = local_id / WaveGetLaneCount();
    uint num_waves = GROUP_SIZE / WaveGetLaneCount();

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    if (local_id == 0) {
        // Stage 2: linear sum across waves on tid==0 (wave-portable).
        // num_waves: 1 (wave32/wave64), 2 (wave16), 4 (wave8). Tiny loop.
        float result0 = 0.0f;
        float result1 = 0.0f;
        for (uint w = 0; w < num_waves; ++w) {
            result0 += shared_acc[w];
            result1 += shared_acc[32 + w];
        }
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
