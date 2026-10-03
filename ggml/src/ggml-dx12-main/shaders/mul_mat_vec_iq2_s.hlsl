// mul_mat_vec_iq2_s.hlsl - Matrix-vector multiply for IQ2_S weights (M=1)
//
// IQ2_S block (82 bytes, QK_K=256 elements):
//   offset 0..1   : d (fp16)
//   offset 2..33  : qs[32] uint8  (grid index low 8 bits, 4 entries per sub-block)
//   offset 34..65 : qs cont -> signs[32] uint8  (4 sign bytes per sub-block)
//   offset 66..73 : qh[8] uint8   (one per sub-block; 2 high bits per entry)
//   offset 74..81 : scales[8] uint8 (one per sub-block, two 4-bit fields)
//
// Per sub-block ib32, pair l (0..3):
//   q_lo   = qs[4*ib32 + l]
//   q_hi2  = (qh[ib32] >> (2*l)) & 0x3
//   grid   = iq2s_grid[q_lo | (q_hi2 << 8)]    (10-bit index)
//   sign_b = qs[32 + 4*ib32 + l]               (8-bit sign mask, 1=neg)
//   il     = l >> 1                            (0 or 1)
//   dl     = d * (0.5 + ((scales[ib32] >> 4*il) & 0xF)) * 0.25
//   8 elements at sub-block offset 8*l.

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK_K        256
#define IQ2S_BSIZE   82

groupshared float shared_acc[GROUP_SIZE];

#include "iq2s_grid.hlsli"

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
        uint block_off = src0_row + block * IQ2S_BSIZE;

        float d = read_f16_iq(src0, block_off);

        uint q_lo  = load_u8_iq(src0, block_off + 2u + 4u * ib32 + l);
        uint qh    = load_u8_iq(src0, block_off + 66u + ib32);
        uint q_hi2 = (qh >> (2u * l)) & 0x3u;
        uint grid_idx = q_lo | (q_hi2 << 8u);

        uint signs = load_u8_iq(src0, block_off + 2u + 32u + 4u * ib32 + l);

        uint il = l >> 1;
        uint scale_byte = load_u8_iq(src0, block_off + 74u + ib32);
        float dl = d * (0.5f + (float)((scale_byte >> (4u * il)) & 0xFu)) * 0.25f;

        uint2 grid8 = iq2s_grid[grid_idx];
        uint glo = grid8.x;
        uint ghi = grid8.y;

        float sblock = 0.0f;
        [unroll] for (uint j = 0; j < 8u; ++j) {
            uint gword = (j < 4u) ? glo : ghi;
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