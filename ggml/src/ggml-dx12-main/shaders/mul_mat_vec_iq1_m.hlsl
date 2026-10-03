// mul_mat_vec_iq1_m.hlsl - Matrix-vector multiply for IQ1_M weights (M=1)
//
// IQ1_M block (56 bytes, QK_K=256 elements; NO explicit d field):
//   offset 0..31  : qs[32] uint8   (low 8 bits of grid index)
//   offset 32..47 : qh[16] uint8   (2 bytes per sub-block; high 3 bits + ml sign per grid)
//   offset 48..55 : scales[8] uint8 = 4 uint16 (each 16-bit holds packed scales; high nibble is d_f16 piece)
//
// d (fp16) reconstruction:
//   sc[i] are 4 uint16s at offset 48..55.
//   d_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
//   d     = f16_to_f32(d_u16)
//
// Per sub-block ib32 (0..7), pair l (0..3):
//   il          = l >> 1            (0 or 1; "half" within sub-block)
//   pair_in_il  = l & 1
//   qs_byte = qs[4*ib32 + l]
//   qh_byte = qh[2*ib32 + il]
//   shift   = (pair_in_il == 0) ? 8 : 4       // bits 0..2 or bits 4..6 of qh
//   grid_idx = qs_byte | ((qh_byte << shift) & 0x700)
//   dl_bits  = (sc[ib32/2] >> (6*(ib32%2) + 3*il)) & 0x7   // 3-bit scale
//   dl       = d * (2*dl_bits + 1)
//   ml       = dl * ((qh_byte & (pair_in_il==0 ? 0x08 : 0x80)) ? -1-IQ1M_DELTA : -1+IQ1M_DELTA)
//   8 elements as in IQ1_S layout.

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK_K        256
#define IQ1M_BSIZE   56
#define IQ1M_DELTA  0.125f

groupshared float shared_acc[GROUP_SIZE];

#include "iq1_grid.hlsli"

uint load_u8_iq(ByteAddressBuffer buf, uint addr) {
    uint a = addr & ~3u;
    uint shift = (addr & 3u) * 8u;
    return (buf.Load(a) >> shift) & 0xFFu;
}

uint load_u16_iq(ByteAddressBuffer buf, uint addr) {
    uint word = buf.Load(addr & ~3u);
    return (word >> ((addr & 2u) * 8u)) & 0xFFFFu;
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

    uint ib32 = tid >> 2;
    uint l    = tid & 3u;
    uint elem_off_in_block = 32u * ib32 + 8u * l;

    precise float acc = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * IQ1M_BSIZE;

        // Reconstruct d
        uint sc0 = load_u16_iq(src0, block_off + 48u);
        uint sc1 = load_u16_iq(src0, block_off + 50u);
        uint sc2 = load_u16_iq(src0, block_off + 52u);
        uint sc3 = load_u16_iq(src0, block_off + 54u);
        uint d_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0u) | ((sc2 >> 4) & 0x0f00u) | (sc3 & 0xf000u);
        float d = f16_to_f32(d_u16);

        uint il = l >> 1;
        uint pair_in_il = l & 1u;

        uint qs_byte = load_u8_iq(src0, block_off + 4u * ib32 + l);
        uint qh_byte = load_u8_iq(src0, block_off + 32u + 2u * ib32 + il);

        uint sh = (pair_in_il == 0u) ? 8u : 4u;
        uint hi3 = (qh_byte << sh) & 0x700u;
        uint grid_idx = qs_byte | hi3;

        // scale bits
        uint sc_word = (ib32 < 2u) ? sc0 : ((ib32 < 4u) ? sc1 : ((ib32 < 6u) ? sc2 : sc3));
        uint dl_bits = (sc_word >> (6u * (ib32 & 1u) + 3u * il)) & 0x7u;
        float dl = d * (2.0f * (float)dl_bits + 1.0f);

        uint sign_mask = (pair_in_il == 0u) ? 0x08u : 0x80u;
        float ml = dl * (((qh_byte & sign_mask) != 0u) ? (-1.0f - IQ1M_DELTA) : (-1.0f + IQ1M_DELTA));

        uint grid = iq1s_grid_gpu[grid_idx];

        float sblock = 0.0f;
        [unroll] for (uint j = 0; j < 8u; ++j) {
            uint b = j & 3u;
            uint nib_high = j >> 2;
            uint gbyte = (grid >> (b * 8u)) & 0xFFu;
            uint nib   = (nib_high == 0u) ? (gbyte & 0xFu) : ((gbyte >> 4) & 0xFu);
            float val  = dl * (float)nib + ml;
            float xval = asfloat(src1.Load(src1_base + (block * QK_K + elem_off_in_block + j) * 4u));
            sblock += val * xval;
        }

        acc += sblock;
    }

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