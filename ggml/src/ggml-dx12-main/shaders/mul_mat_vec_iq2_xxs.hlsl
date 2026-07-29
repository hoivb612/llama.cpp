// mul_mat_vec_iq2_xxs.hlsl - Matrix-vector multiply for IQ2_XXS weights (M=1)
//
// IQ2_XXS block (66 bytes, QK_K=256 elements):
//   offset 0..1   : d (fp16)
//   offset 2..65  : qs[32] uint16  (32 * 2 = 64 bytes)
//
// Each 32-element sub-block (ib32 = 0..7) uses 4 uint16 = 8 bytes laid out as:
//   bytes 0..3 (qs[4*ib32+0..1]): aux32_g — 4 grid indices, 8 bits each
//   bytes 4..7 (qs[4*ib32+2..3]): aux32_s — packed signs (4*7 bits) + scale (4 bits)
//
// scale  = d * 0.25 * (0.5 + (aux32_s >> 28))
// signs  = (aux32_s >> (7*l)) & 0x7F   for l in 0..3   (8th sign bit = popcount(signs)&1)
// grid   = iq2xxs_grid[(aux32_g >> (8*l)) & 0xFF]    (8 packed signed bytes)
//
// 32 threads/group, 1 row per group. Each thread covers (ib32=tid/4, l=tid%4) =
// 8 elements per superblock. Loop K/256 superblocks per row.

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK_K        256
#define IQ2XXS_BSIZE 66

groupshared float shared_acc[GROUP_SIZE];

// Packed grid table: 256 entries x 8 bytes (uint2). Lower uint = bytes 0..3,
// upper uint = bytes 4..7 (matches iq2xxs_grid uint64 little-endian byte view).
static const uint2 iq2xxs_grid[256] = {
    uint2(0x08080808u,0x08080808u), uint2(0x0808082bu,0x08080808u), uint2(0x08081919u,0x08080808u), uint2(0x08082b08u,0x08080808u),
    uint2(0x08082b2bu,0x08080808u), uint2(0x08190819u,0x08080808u), uint2(0x08191908u,0x08080808u), uint2(0x082b0808u,0x08080808u),
    uint2(0x082b082bu,0x08080808u), uint2(0x082b2b08u,0x08080808u), uint2(0x082b2b2bu,0x08080808u), uint2(0x19080819u,0x08080808u),
    uint2(0x19081908u,0x08080808u), uint2(0x19190808u,0x08080808u), uint2(0x19192b08u,0x08080808u), uint2(0x192b0819u,0x08080808u),
    uint2(0x192b1908u,0x08080808u), uint2(0x2b080808u,0x08080808u), uint2(0x2b08082bu,0x08080808u), uint2(0x2b082b2bu,0x08080808u),
    uint2(0x2b2b082bu,0x08080808u), uint2(0x08080819u,0x08080819u), uint2(0x08081908u,0x08080819u), uint2(0x08190808u,0x08080819u),
    uint2(0x08191919u,0x08080819u), uint2(0x19080808u,0x08080819u), uint2(0x2b081908u,0x08080819u), uint2(0x2b192b08u,0x08080819u),
    uint2(0x08080808u,0x0808082bu), uint2(0x0808082bu,0x0808082bu), uint2(0x082b082bu,0x0808082bu), uint2(0x2b08082bu,0x0808082bu),
    uint2(0x08080819u,0x08081908u), uint2(0x08081908u,0x08081908u), uint2(0x08190808u,0x08081908u), uint2(0x082b0819u,0x08081908u),
    uint2(0x082b1908u,0x08081908u), uint2(0x19080808u,0x08081908u), uint2(0x1908082bu,0x08081908u), uint2(0x19082b08u,0x08081908u),
    uint2(0x192b0808u,0x08081908u), uint2(0x2b080819u,0x08081908u), uint2(0x2b081908u,0x08081908u), uint2(0x2b190808u,0x08081908u),
    uint2(0x2b2b1908u,0x08081908u), uint2(0x08080808u,0x08081919u), uint2(0x0808082bu,0x08081919u), uint2(0x08082b08u,0x08081919u),
    uint2(0x082b0808u,0x08081919u), uint2(0x1908192bu,0x08081919u), uint2(0x192b2b19u,0x08081919u), uint2(0x2b080808u,0x08081919u),
    uint2(0x2b190819u,0x08081919u), uint2(0x08082b19u,0x0808192bu), uint2(0x08190808u,0x0808192bu), uint2(0x19080808u,0x0808192bu),
    uint2(0x2b081908u,0x0808192bu), uint2(0x2b2b1908u,0x0808192bu), uint2(0x08080808u,0x08082b08u), uint2(0x08081919u,0x08082b08u),
    uint2(0x08082b08u,0x08082b08u), uint2(0x08191908u,0x08082b08u), uint2(0x082b2b08u,0x08082b08u), uint2(0x19080819u,0x08082b08u),
    uint2(0x19081908u,0x08082b08u), uint2(0x19190808u,0x08082b08u), uint2(0x1919082bu,0x08082b08u), uint2(0x2b082b08u,0x08082b08u),
    uint2(0x08081908u,0x08082b19u), uint2(0x19080808u,0x08082b19u), uint2(0x0808082bu,0x08082b2bu), uint2(0x08191908u,0x08082b2bu),
    uint2(0x08080819u,0x08190808u), uint2(0x08081908u,0x08190808u), uint2(0x08190808u,0x08190808u), uint2(0x082b0819u,0x08190808u),
    uint2(0x19080808u,0x08190808u), uint2(0x192b0808u,0x08190808u), uint2(0x2b081908u,0x08190808u), uint2(0x2b190808u,0x08190808u),
    uint2(0x2b191919u,0x08190808u), uint2(0x08080808u,0x08190819u), uint2(0x08082b08u,0x08190819u), uint2(0x082b0808u,0x08190819u),
    uint2(0x19190808u,0x08190819u), uint2(0x19192b2bu,0x08190819u), uint2(0x2b080808u,0x08190819u), uint2(0x082b1908u,0x0819082bu),
    uint2(0x19081919u,0x0819082bu), uint2(0x08080808u,0x08191908u), uint2(0x08082b08u,0x08191908u), uint2(0x082b0808u,0x08191908u),
    uint2(0x082b1919u,0x08191908u), uint2(0x19082b19u,0x08191908u), uint2(0x2b080808u,0x08191908u), uint2(0x08192b08u,0x08191919u),
    uint2(0x192b082bu,0x08191919u), uint2(0x08080808u,0x0819192bu), uint2(0x0819192bu,0x0819192bu), uint2(0x08080819u,0x08192b08u),
    uint2(0x08081908u,0x08192b08u), uint2(0x08190808u,0x08192b08u), uint2(0x19080808u,0x08192b08u), uint2(0x2b080819u,0x08192b08u),
    uint2(0x08080808u,0x08192b19u), uint2(0x08081919u,0x08192b19u), uint2(0x2b2b0808u,0x08192b19u), uint2(0x19190819u,0x08192b2bu),
    uint2(0x08080808u,0x082b0808u), uint2(0x0808082bu,0x082b0808u), uint2(0x08082b2bu,0x082b0808u), uint2(0x19081908u,0x082b0808u),
    uint2(0x192b0819u,0x082b0808u), uint2(0x2b080808u,0x082b0808u), uint2(0x2b08082bu,0x082b0808u), uint2(0x082b2b19u,0x082b0819u),
    uint2(0x19082b08u,0x082b0819u), uint2(0x08080808u,0x082b082bu), uint2(0x0808082bu,0x082b082bu), uint2(0x08080819u,0x082b1908u),
    uint2(0x08081908u,0x082b1908u), uint2(0x08190808u,0x082b1908u), uint2(0x19080808u,0x082b1908u), uint2(0x1919192bu,0x082b1908u),
    uint2(0x08080808u,0x082b1919u), uint2(0x19080819u,0x082b1919u), uint2(0x192b1908u,0x082b1919u), uint2(0x2b190808u,0x082b192bu),
    uint2(0x08082b08u,0x082b2b08u), uint2(0x082b0808u,0x082b2b08u), uint2(0x2b191908u,0x082b2b08u), uint2(0x19081908u,0x082b2b2bu),
    uint2(0x08080819u,0x19080808u), uint2(0x08081908u,0x19080808u), uint2(0x08190808u,0x19080808u), uint2(0x08192b08u,0x19080808u),
    uint2(0x082b0819u,0x19080808u), uint2(0x082b1908u,0x19080808u), uint2(0x19080808u,0x19080808u), uint2(0x19082b08u,0x19080808u),
    uint2(0x1919192bu,0x19080808u), uint2(0x192b0808u,0x19080808u), uint2(0x2b080819u,0x19080808u), uint2(0x2b081908u,0x19080808u),
    uint2(0x2b190808u,0x19080808u), uint2(0x08080808u,0x19080819u), uint2(0x082b0808u,0x19080819u), uint2(0x192b0819u,0x19080819u),
    uint2(0x2b080808u,0x19080819u), uint2(0x2b081919u,0x19080819u), uint2(0x08080819u,0x1908082bu), uint2(0x08190808u,0x1908082bu),
    uint2(0x19082b08u,0x1908082bu), uint2(0x1919192bu,0x1908082bu), uint2(0x192b2b08u,0x1908082bu), uint2(0x08080808u,0x19081908u),
    uint2(0x08082b08u,0x19081908u), uint2(0x082b0808u,0x19081908u), uint2(0x2b080808u,0x19081908u), uint2(0x2b192b19u,0x19081908u),
    uint2(0x0819082bu,0x19081919u), uint2(0x082b1908u,0x19081919u), uint2(0x08080808u,0x1908192bu), uint2(0x08080819u,0x19082b08u),
    uint2(0x08081908u,0x19082b08u), uint2(0x08190808u,0x19082b08u), uint2(0x19080808u,0x19082b08u), uint2(0x19081919u,0x19082b08u),
    uint2(0x08080808u,0x19082b19u), uint2(0x19192b08u,0x19082b19u), uint2(0x192b0819u,0x19082b19u), uint2(0x2b08082bu,0x19082b19u),
    uint2(0x19081919u,0x19082b2bu), uint2(0x2b190808u,0x19082b2bu), uint2(0x08080808u,0x19190808u), uint2(0x08082b08u,0x19190808u),
    uint2(0x08190819u,0x19190808u), uint2(0x08192b19u,0x19190808u), uint2(0x082b0808u,0x19190808u), uint2(0x2b080808u,0x19190808u),
    uint2(0x2b082b08u,0x19190808u), uint2(0x08081908u,0x19190819u), uint2(0x1908082bu,0x19190819u), uint2(0x2b2b1908u,0x19190819u),
    uint2(0x2b190819u,0x1919082bu), uint2(0x2b190808u,0x19191908u), uint2(0x2b19082bu,0x19191908u), uint2(0x08082b2bu,0x19191919u),
    uint2(0x08080819u,0x1919192bu), uint2(0x19191908u,0x1919192bu), uint2(0x08080808u,0x19192b08u), uint2(0x08190819u,0x19192b08u),
    uint2(0x08192b19u,0x19192b08u), uint2(0x192b1908u,0x19192b08u), uint2(0x19080808u,0x19192b19u), uint2(0x08082b08u,0x19192b2bu),
    uint2(0x08081908u,0x192b0808u), uint2(0x08190808u,0x192b0808u), uint2(0x19080808u,0x192b0808u), uint2(0x192b2b08u,0x192b0808u),
    uint2(0x08080808u,0x192b0819u), uint2(0x19191919u,0x192b0819u), uint2(0x08192b08u,0x192b082bu), uint2(0x192b0808u,0x192b082bu),
    uint2(0x08080808u,0x192b1908u), uint2(0x08081919u,0x192b1908u), uint2(0x08190808u,0x192b1919u), uint2(0x0819082bu,0x192b1919u),
    uint2(0x2b081908u,0x192b1919u), uint2(0x1908082bu,0x192b2b08u), uint2(0x08080808u,0x2b080808u), uint2(0x0808082bu,0x2b080808u),
    uint2(0x08082b2bu,0x2b080808u), uint2(0x19080819u,0x2b080808u), uint2(0x2b08082bu,0x2b080808u), uint2(0x08081908u,0x2b080819u),
    uint2(0x08192b08u,0x2b080819u), uint2(0x19080808u,0x2b080819u), uint2(0x08190819u,0x2b08082bu), uint2(0x08080819u,0x2b081908u),
    uint2(0x08081908u,0x2b081908u), uint2(0x08190808u,0x2b081908u), uint2(0x08191919u,0x2b081908u), uint2(0x19080808u,0x2b081908u),
    uint2(0x192b0808u,0x2b081908u), uint2(0x08080808u,0x2b081919u), uint2(0x1908192bu,0x2b081919u), uint2(0x2b191908u,0x2b081919u),
    uint2(0x08082b19u,0x2b08192bu), uint2(0x19080808u,0x2b08192bu), uint2(0x192b0808u,0x2b08192bu), uint2(0x0808082bu,0x2b082b08u),
    uint2(0x08081908u,0x2b082b19u), uint2(0x08190819u,0x2b082b2bu), uint2(0x08081908u,0x2b190808u), uint2(0x08190808u,0x2b190808u),
    uint2(0x082b1908u,0x2b190808u), uint2(0x19080808u,0x2b190808u), uint2(0x2b2b0819u,0x2b190808u), uint2(0x0819192bu,0x2b190819u),
    uint2(0x2b080808u,0x2b190819u), uint2(0x19081919u,0x2b19082bu), uint2(0x08080808u,0x2b191908u), uint2(0x082b082bu,0x2b191908u),
    uint2(0x19081908u,0x2b191908u), uint2(0x19190819u,0x2b191919u), uint2(0x2b080819u,0x2b192b08u), uint2(0x082b0808u,0x2b192b19u),
    uint2(0x0808082bu,0x2b2b0808u), uint2(0x19190808u,0x2b2b0808u), uint2(0x2b081919u,0x2b2b0808u), uint2(0x08082b19u,0x2b2b0819u),
    uint2(0x08080808u,0x2b2b082bu), uint2(0x08192b08u,0x2b2b1908u), uint2(0x19190808u,0x2b2b2b08u), uint2(0x08081908u,0x2b2b2b19u)
};

// Aligned 32-bit load from a 2-byte aligned ByteAddressBuffer offset.
// IQ2_XXS block size 66 is 2-aligned (not 4-aligned); rely on AMD/Intel
// Load4-on-2-aligned tolerance verified for Q3_K and Q6_K.
uint load_u32_iq2(ByteAddressBuffer buf, uint addr) {
    uint a = addr & ~3u;
    uint shift = (addr & 3u) * 8u;
    uint w0 = buf.Load(a);
    if (shift == 0u) return w0;
    uint w1 = buf.Load(a + 4u);
    return (w0 >> shift) | (w1 << (32u - shift));
}

float read_f16_iq2(ByteAddressBuffer buf, uint byte_off) {
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
    uint elem_off_in_block = 32u * ib32 + 8u * l;

    precise float acc = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * IQ2XXS_BSIZE;

        float d = read_f16_iq2(src0, block_off);

        // qs[4*ib32 .. 4*ib32+3] live at byte offset 2 + 8*ib32.
        uint sub_off  = block_off + 2u + 8u * ib32;
        uint aux32_g  = load_u32_iq2(src0, sub_off);          // 4 grid indices
        uint aux32_s  = load_u32_iq2(src0, sub_off + 4u);     // signs + scale

        uint grid_idx = (aux32_g >> (l * 8u)) & 0xFFu;
        uint signs    = (aux32_s >> (l * 7u)) & 0x7Fu;
        uint parity   = countbits(signs) & 1u;

        float scale = d * 0.25f * (0.5f + (float)(aux32_s >> 28));

        uint2 grid8 = iq2xxs_grid[grid_idx];
        uint glo = grid8.x;
        uint ghi = grid8.y;

        // Element 0..3 from grid8.x, element 4..7 from grid8.y; each is u8.
        // Sign for j: (j<7) ? bit j of signs : parity.
        float sblock = 0.0f;
        [unroll] for (uint j = 0; j < 8u; ++j) {
            uint gword = (j < 4u) ? glo : ghi;
            uint gbyte = (gword >> ((j & 3u) * 8u)) & 0xFFu;
            float gval = (float)gbyte;
            uint sbit  = (j < 7u) ? ((signs >> j) & 1u) : parity;
            float sgn  = (sbit != 0u) ? -1.0f : 1.0f;
            float xval = asfloat(src1.Load(src1_base + (block * QK_K + elem_off_in_block + j) * 4u));
            sblock += gval * sgn * xval;
        }

        acc += scale * sblock;
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
