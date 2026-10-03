// mul_mat_vec_iq4_xs.hlsl - Matrix-vector multiply for IQ4_XS weights (M=1)
//
// IQ4_XS block (136 bytes, 4-aligned, QK_K=256 elements):
//   offset 0..1   : d (fp16)
//   offset 2..3   : scales_h (uint16) — high 2 bits of 6-bit scale per sub-block
//   offset 4..7   : scales_l[4] (uint8) — low 4 bits of scale, 2 sub-blocks per byte
//   offset 8..135 : qs[128] (uint8) — 16 bytes per sub-block, 2 4-bit indices per byte
//
// 8 sub-blocks of 32 elements (ib32 = 0..7). Per sub-block:
//   scale6 = (scales_l[ib32/2] >> (4*(ib32%2))) & 0xF
//          | (((scales_h >> (2*ib32)) & 3) << 4)
//   dl = d * (scale6 - 32)
//   For j in 0..15:
//     elem[j]    = dl * kvalues_iq4nl[ qs[ib32*16 + j] & 0x0F ]
//     elem[j+16] = dl * kvalues_iq4nl[ qs[ib32*16 + j] >>   4 ]
//
// 32 threads/group, 1 row/group. Each thread handles (ib32 = tid/4, l = tid%4),
// covering 4 low-nibble + 4 high-nibble elements = 8 elements per superblock.
// Loop K/256 superblocks per row.

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK_K        256
#define IQ4XS_BSIZE 136

groupshared float shared_acc[GROUP_SIZE];

// Non-linear 4-bit codebook (matches kvalues_iq4nl in ggml-common.h).
// Identical to the table used by mul_mat_vec_iq4_nl.hlsl.
int kvalues_iq4xs(uint idx) {
    static const uint packed[4] = {
        0xBFAD9881u,  // -127, -104, -83, -65
        0xF6EADDCFu,  //  -49,  -35, -22, -10
        0x26190D01u,  //    1,   13,  25,  38
        0x71594535u   //   53,   69,  89, 113
    };
    uint w = packed[idx >> 2];
    uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    return (int)(b << 24) >> 24;
}

float read_f16_iq4xs(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i0 = group_x_2d(group_id);
    if (i0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint src0_row  = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    uint ib32 = tid >> 2;       // 0..7
    uint l    = tid & 3u;       // 0..3
    uint sub_off  = 32u * ib32 + 4u * l;   // first sub-element index for this thread

    precise float acc = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * IQ4XS_BSIZE;

        float d        = read_f16_iq4xs(src0, block_off);
        uint scales_h  = src0.Load(block_off) >> 16;          // bytes 2..3 of the d/scales_h word
        // scales_l[4] sit in the next 4 bytes (offset 4..7) — single Load.
        uint scales_l  = src0.Load(block_off + 4u);

        // 6-bit unsigned scale for this sub-block, biased -32.
        uint slo = (scales_l >> ((ib32 / 2u) * 8u + (ib32 & 1u) * 4u)) & 0xFu;
        uint shi = (scales_h >> (2u * ib32)) & 0x3u;
        int  s6  = (int)(slo | (shi << 4));
        float dl = d * (float)(s6 - 32);

        // qs for this sub-block live at block_off + 8 + 16*ib32. We want the
        // 4 bytes starting at byte index 4*l within the sub-block (one Load).
        uint qs_word = src0.Load(block_off + 8u + 16u * ib32 + 4u * l);

        float sblock = 0.0f;
        [unroll] for (uint j = 0; j < 4u; ++j) {
            uint qbyte = (qs_word >> (j * 8u)) & 0xFFu;
            int vlo = kvalues_iq4xs(qbyte & 0xFu);
            int vhi = kvalues_iq4xs(qbyte >>  4);

            uint k_lo = block * QK_K + 32u * ib32 + (4u * l + j);          // 0..15 within sub-block
            uint k_hi = block * QK_K + 32u * ib32 + (4u * l + j) + 16u;    // 16..31 within sub-block

            float xlo = asfloat(src1.Load(src1_base + k_lo * 4u));
            float xhi = asfloat(src1.Load(src1_base + k_hi * 4u));

            sblock += (float)vlo * xlo + (float)vhi * xhi;
        }
        acc += dl * sblock;
    }

    // Two-level wave + shared memory reduction (correct for any wave size)
    float wave_sum = WaveActiveSum(acc);
    uint wave_id = tid / WARP_SIZE;
    if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    uint num_waves = GROUP_SIZE / WARP_SIZE;
    if (num_waves <= WARP_SIZE) {
        if (tid < num_waves) {
            float v = shared_acc[tid];
            v = WaveActiveSum(v);
            if (tid == 0) shared_acc[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
    } else {
        for (uint s = num_waves / 2; s > 0; s /= 2) {
            if (tid < s) shared_acc[tid] += shared_acc[tid + s];
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (tid == 0) {
        float result = shared_acc[0];
        result += load_fused_bias(i0, i2, i3);
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}
