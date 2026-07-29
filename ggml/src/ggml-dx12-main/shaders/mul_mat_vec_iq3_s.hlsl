// mul_mat_vec_iq3_s.hlsl - Matrix-vector multiply for IQ3_S weights (M=1)
//
// IQ3_S block (110 bytes, QK_K=256 elements):
//   offset 0..1     : d (fp16)
//   offset 2..65    : qs[64] uint8   (8 grid indices per sub-block; low 8 bits)
//   offset 66..73   : qh[8] uint8    (one byte per sub-block; 8 high bits)
//   offset 74..105  : signs[32] uint8 (4 sign bytes per sub-block)
//   offset 106..109 : scales[4] uint8 (4 bits per sub-block; ib32%2 selects nibble)
//
// Per sub-block ib32 (0..7), pair l (0..3) covers 8 elements:
//   qs_a = qs[8*ib32 + 2*l],     qs_b = qs[8*ib32 + 2*l + 1]
//   bit_a = (qh[ib32] >> (2*l))   & 1
//   bit_b = (qh[ib32] >> (2*l+1)) & 1
//   grid_a = iq3s_grid[qs_a | (bit_a << 8)]
//   grid_b = iq3s_grid[qs_b | (bit_b << 8)]
//   signs  = signs[4*ib32 + l]   (8 sign bits: 0..3 for grid_a, 4..7 for grid_b)
//   dl     = d * (1 + 2 * ((scales[ib32/2] >> 4*(ib32%2)) & 0xF))

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK_K        256
#define IQ3S_BSIZE  110

groupshared float shared_acc[GROUP_SIZE];

#include "iq3s_grid.hlsli"

uint load_u8_iq(ByteAddressBuffer buf, uint addr) {
    uint a = addr & ~3u;
    uint shift = (addr & 3u) * 8u;
    return (buf.Load(a) >> shift) & 0xFFu;
}

float read_f16_iq(ByteAddressBuffer buf, uint byte_off) {
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

    uint ib32 = tid >> 2;
    uint l    = tid & 3u;
    uint elem_off_in_block = 32u * ib32 + 8u * l;

    precise float acc = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * IQ3S_BSIZE;

        float d = read_f16_iq(src0, block_off);

        uint qs_a = load_u8_iq(src0, block_off + 2u + 8u * ib32 + 2u * l);
        uint qs_b = load_u8_iq(src0, block_off + 2u + 8u * ib32 + 2u * l + 1u);
        uint qh   = load_u8_iq(src0, block_off + 66u + ib32);
        uint bit_a = (qh >> (2u * l))      & 1u;
        uint bit_b = (qh >> (2u * l + 1u)) & 1u;
        uint idx_a = qs_a | (bit_a << 8u);
        uint idx_b = qs_b | (bit_b << 8u);

        uint signs = load_u8_iq(src0, block_off + 74u + 4u * ib32 + l);

        uint scale_byte = load_u8_iq(src0, block_off + 106u + (ib32 >> 1));
        uint scale_nib = (scale_byte >> (4u * (ib32 & 1u))) & 0xFu;
        float dl = d * (1.0f + 2.0f * (float)scale_nib);

        uint ga = iq3s_grid[idx_a];
        uint gb = iq3s_grid[idx_b];

        float sblock = 0.0f;
        [unroll] for (uint j = 0; j < 8u; ++j) {
            uint gword = (j < 4u) ? ga : gb;
            uint gbyte = (gword >> ((j & 3u) * 8u)) & 0xFFu;
            float gval = (float)gbyte;
            uint sbit = (signs >> j) & 1u;
            float sgn = (sbit != 0u) ? -1.0f : 1.0f;
            float xval = asfloat(src1.Load(src1_base + (block * QK_K + elem_off_in_block + j) * 4u));
            sblock += gval * sgn * xval;
        }

        acc += dl * sblock;
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