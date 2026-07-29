// mul_mat_vec_iq2_xxs_mr.hlsl - Multi-row IQ2_XXS matvec (M=1, 2 rows/group)
//
// Same per-block decode as mul_mat_vec_iq2_xxs.hlsl, but processes two output
// rows per workgroup. The 8 src1 activations covered by each (ib32, l) are
// loaded once and re-used for both rows, halving the activation traffic and
// the dispatch count vs the single-row variant.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK_K        256
#define IQ2XXS_BSIZE 66

groupshared float shared_acc[64];

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
    uint row0 = group_x_2d(group_id) * 2u;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1u) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    uint ib32 = tid >> 2;       // 0..7
    uint l    = tid & 3u;       // 0..3
    uint elem_off_in_block = 32u * ib32 + 8u * l;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        // Load the 8 activations for this thread's element strip once.
        float xv[8];
        uint x_base = src1_base + (block * QK_K + elem_off_in_block) * 4u;
        [unroll] for (uint jj = 0; jj < 8u; ++jj) {
            xv[jj] = asfloat(src1.Load(x_base + jj * 4u));
        }

        uint blk_off0 = src0_row0 + block * IQ2XXS_BSIZE;
        uint blk_off1 = src0_row1 + block * IQ2XXS_BSIZE;
        uint sub_off0 = blk_off0 + 2u + 8u * ib32;
        uint sub_off1 = blk_off1 + 2u + 8u * ib32;

        // ---- Row 0 ----
        {
            float d         = read_f16_iq2(src0, blk_off0);
            uint  aux32_g   = load_u32_iq2(src0, sub_off0);
            uint  aux32_s   = load_u32_iq2(src0, sub_off0 + 4u);
            uint  grid_idx  = (aux32_g >> (l * 8u)) & 0xFFu;
            uint  signs     = (aux32_s >> (l * 7u)) & 0x7Fu;
            uint  parity    = countbits(signs) & 1u;
            float scale     = d * 0.25f * (0.5f + (float)(aux32_s >> 28));
            uint2 grid8     = iq2xxs_grid[grid_idx];
            float sblock    = 0.0f;
            [unroll] for (uint j = 0; j < 8u; ++j) {
                uint  gword = (j < 4u) ? grid8.x : grid8.y;
                uint  gbyte = (gword >> ((j & 3u) * 8u)) & 0xFFu;
                uint  sbit  = (j < 7u) ? ((signs >> j) & 1u) : parity;
                float sgn   = (sbit != 0u) ? -1.0f : 1.0f;
                sblock     += (float)gbyte * sgn * xv[j];
            }
            acc0 += scale * sblock;
        }

        // ---- Row 1 ----
        {
            float d         = read_f16_iq2(src0, blk_off1);
            uint  aux32_g   = load_u32_iq2(src0, sub_off1);
            uint  aux32_s   = load_u32_iq2(src0, sub_off1 + 4u);
            uint  grid_idx  = (aux32_g >> (l * 8u)) & 0xFFu;
            uint  signs     = (aux32_s >> (l * 7u)) & 0x7Fu;
            uint  parity    = countbits(signs) & 1u;
            float scale     = d * 0.25f * (0.5f + (float)(aux32_s >> 28));
            uint2 grid8     = iq2xxs_grid[grid_idx];
            float sblock    = 0.0f;
            [unroll] for (uint j = 0; j < 8u; ++j) {
                uint  gword = (j < 4u) ? grid8.x : grid8.y;
                uint  gbyte = (gword >> ((j & 3u) * 8u)) & 0xFFu;
                uint  sbit  = (j < 7u) ? ((signs >> j) & 1u) : parity;
                float sgn   = (sbit != 0u) ? -1.0f : 1.0f;
                sblock     += (float)gbyte * sgn * xv[j];
            }
            acc1 += scale * sblock;
        }
    }

    // Two-level wave + shared memory reduction
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint  wave_id   = tid / WaveGetLaneCount();
    uint  num_waves = GROUP_SIZE / WaveGetLaneCount();

    if (WaveIsFirstLane()) {
        shared_acc[wave_id]       = wave_sum0;
        shared_acc[32 + wave_id]  = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float r0 = 0.0f;
        float r1 = 0.0f;
        for (uint w = 0; w < num_waves; ++w) {
            r0 += shared_acc[w];
            r1 += shared_acc[32 + w];
        }
        r0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, r0, dst_esize);

        if (row0 + 1u < ne0) {
            r1 += load_fused_bias(row0 + 1u, i2, i3);
            uint off_d1 = offset_4d(row0 + 1u, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, r1, dst_esize);
        }
    }
}
