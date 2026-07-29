// mul_mat_vec_iq4_nl_mr256.hlsl - 256-thread multi-row IQ4_NL matvec (M=1, 2 rows/group)
//
// Mirrors the 32-thread mul_mat_vec_iq4_nl_mr shader structure, but bumps
// GROUP_SIZE to 256 so that wide GPUs (wave>=32) see >1 wave per group,
// matching the Q8_0 mr256 fix for the same "launch-bound on 32-thread
// shaders at small K" pattern (SmolLM2-135M Q3_K_M routes IQ4_NL FFN
// gate/up matvec K=576 N=1536 here ~60x per token).
//
// IQ4_NL block: d(f16) + qs[16] = 18 bytes per 32 elements. Each qs byte
// holds two 4-bit indices into the IQ4_NL non-linear codebook.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 256
#define QK4_NL     32
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

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
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
        // Shared: load activation once, reused for both rows
        float x = asfloat(src1.Load(src1_base + k * 4));

        uint block = k / QK4_NL;
        uint elem  = k % QK4_NL;

        // qs layout: low nibble holds elem<16, high nibble holds elem>=16,
        // packed into qs[elem & 15].
        uint qs_idx       = elem & 15u;
        uint qs_word_off  = qs_idx & ~3u;
        uint qs_byte_shift = (qs_idx & 3u) * 8u;
        bool hi_nibble    = (elem >= 16u);

        // Row 0
        uint blk_off0 = src0_row0 + block * IQ4_NL_BSIZE;
        float d0 = read_f16_v(src0, blk_off0);
        uint qs_word0 = read_u32_fast(src0, blk_off0 + 2 + qs_word_off);
        uint qs_byte0 = (qs_word0 >> qs_byte_shift) & 0xFFu;
        uint nib0 = hi_nibble ? ((qs_byte0 >> 4) & 0x0Fu) : (qs_byte0 & 0x0Fu);
        int  val0 = kvalues_iq4nl(nib0);
        acc0 += d0 * float(val0) * x;

        // Row 1
        uint blk_off1 = src0_row1 + block * IQ4_NL_BSIZE;
        float d1 = read_f16_v(src0, blk_off1);
        uint qs_word1 = read_u32_fast(src0, blk_off1 + 2 + qs_word_off);
        uint qs_byte1 = (qs_word1 >> qs_byte_shift) & 0xFFu;
        uint nib1 = hi_nibble ? ((qs_byte1 >> 4) & 0x0Fu) : (qs_byte1 & 0x0Fu);
        int  val1 = kvalues_iq4nl(nib1);
        acc1 += d1 * float(val1) * x;
    }

    // Two-level wave + LDS reduction (mirrors mul_mat_vec_q8_0_mr256).
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id   = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;
    uint row1_off  = (num_waves > 0) ? num_waves : 1;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[row1_off + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
            shared_acc[row1_off + tid] += shared_acc[row1_off + tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float result0 = shared_acc[0];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[row1_off];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
