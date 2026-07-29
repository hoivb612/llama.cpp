// mul_mat_vec_iq3_xxs.hlsl - Matrix-vector multiply for IQ3_XXS weights (M=1)
//
// IQ3_XXS block (98 bytes, QK_K=256 elements):
//   offset 0..1   : d (fp16)
//   offset 2..65  : q3[64] uint8   (8 bytes per sub-block x 8 sub-blocks)
//   offset 66..97 : gas[16] uint16 -> packed as 4 bytes per sub-block (aux32)
//
// Per sub-block ib32 (0..7) of 32 elements:
//   aux32   = (gas[2*ib32] | (gas[2*ib32+1] << 16))   from bytes 66+4*ib32 .. 69+4*ib32
//   scale   = d * 0.5 * (0.5 + (aux32 >> 28))
//   signs_l = (aux32 >> (7*l)) & 0x7F   for l in 0..3   (8th sign bit = popcount&1)
//   For pair l in 0..3, the 8 elements [ib32*32 + l*8 + 0..7] come from:
//     elem[0..3] = signs_l-applied iq3xxs_grid[q3[8*ib32 + 2*l    ]]  (4 packed bytes)
//     elem[4..7] = signs_l-applied iq3xxs_grid[q3[8*ib32 + 2*l + 1]]
//
// 32 threads/group, 1 row per group. Each thread covers (ib32=tid/4, l=tid%4)
// = 8 elements per superblock. Loop K/256 superblocks per row.

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK_K        256
#define IQ3XXS_BSIZE 98

groupshared float shared_acc[GROUP_SIZE];

#include "iq3xxs_grid.hlsli"

// Aligned 32-bit load from a 2-byte aligned ByteAddressBuffer offset.
uint load_u32_iq3(ByteAddressBuffer buf, uint addr) {
    uint a = addr & ~3u;
    uint shift = (addr & 3u) * 8u;
    uint w0 = buf.Load(a);
    if (shift == 0u) return w0;
    uint w1 = buf.Load(a + 4u);
    return (w0 >> shift) | (w1 << (32u - shift));
}

float read_f16_iq3(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

// Load a single byte from a 2-aligned offset.
uint load_u8_iq3(ByteAddressBuffer buf, uint addr) {
    uint a = addr & ~3u;
    uint shift = (addr & 3u) * 8u;
    return (buf.Load(a) >> shift) & 0xFFu;
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
    uint l    = tid & 3u;       // pair 0..3
    uint elem_off_in_block = 32u * ib32 + 8u * l;

    precise float acc = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * IQ3XXS_BSIZE;

        float d = read_f16_iq3(src0, block_off);

        // q3 pair lives at q3[8*ib32 + 2*l + 0..1] -> bytes 2 + 8*ib32 + 2*l
        uint q3_off = block_off + 2u + 8u * ib32 + 2u * l;
        uint q3a = load_u8_iq3(src0, q3_off);
        uint q3b = load_u8_iq3(src0, q3_off + 1u);

        // aux32 at bytes 66 + 4*ib32
        uint aux32 = load_u32_iq3(src0, block_off + 66u + 4u * ib32);
        uint signs = (aux32 >> (l * 7u)) & 0x7Fu;
        uint parity = countbits(signs) & 1u;

        float scale = d * 0.5f * (0.5f + (float)(aux32 >> 28));

        uint ga = iq3xxs_grid[q3a];
        uint gb = iq3xxs_grid[q3b];

        float sblock = 0.0f;
        [unroll] for (uint j = 0; j < 8u; ++j) {
            uint gword = (j < 4u) ? ga : gb;
            uint gbyte = (gword >> ((j & 3u) * 8u)) & 0xFFu;
            float gval = (float)gbyte;
            uint sbit = (j < 7u) ? ((signs >> j) & 1u) : parity;
            float sgn = (sbit != 0u) ? -1.0f : 1.0f;
            float xval = asfloat(src1.Load(src1_base + (block * QK_K + elem_off_in_block + j) * 4u));
            sblock += gval * sgn * xval;
        }

        acc += scale * sblock;
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