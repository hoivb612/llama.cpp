// mul_mat_vec_iq1_s.hlsl - Matrix-vector multiply for IQ1_S weights (M=1)
//
// IQ1_S block (50 bytes, QK_K=256 elements):
//   offset 0..1   : d (fp16)
//   offset 2..33  : qs[32] uint8   (low 8 bits of grid index)
//   offset 34..49 : qh[8] uint16   (per sub-block: 4x3 bit high index + 3 bit scale + sign)
//
// Per sub-block ib32, pair l (0..3) covers 8 elements:
//   qs_byte = qs[4*ib32 + l]
//   qh_full = qh[ib32] uint16
//   hi3     = (qh_full >> (3*l)) & 0x7    (bits l*3..l*3+2 of qh)
//   grid_idx = qs_byte | (hi3 << 8)        (11-bit index)
//   grid    = iq1s_grid_gpu[grid_idx]      (uint32 = 4 packed bytes; 2 nibbles each)
//   dl     = d * (2*((qh_full >> 12) & 7) + 1)
//   ml     = dl * ((qh_full & 0x8000) ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)
//   elements: 8 vals; for byte b=0..3 -> val_lo = grid_byte_b & 0xF, val_hi = grid_byte_b >> 4
//             reg[0][b] (low nibble of byte b) for b=0..3 = elems 0..3
//             reg[1][b] (high nibble of byte b)            = elems 4..7

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK_K        256
#define IQ1S_BSIZE  50
#define IQ1S_DELTA  0.125f

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
        uint block_off = src0_row + block * IQ1S_BSIZE;

        float d = read_f16_iq(src0, block_off);

        uint qs_byte = load_u8_iq(src0, block_off + 2u + 4u * ib32 + l);
        uint qh_full = load_u16_iq(src0, block_off + 34u + 2u * ib32);

        uint hi3 = (qh_full >> (3u * l)) & 0x7u;
        uint grid_idx = qs_byte | (hi3 << 8u);

        float dl = d * (2.0f * (float)((qh_full >> 12) & 0x7u) + 1.0f);
        float ml = dl * (((qh_full & 0x8000u) != 0u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA));

        uint grid = iq1s_grid_gpu[grid_idx];

        float sblock = 0.0f;
        [unroll] for (uint j = 0; j < 8u; ++j) {
            uint b = j & 3u;                // byte index 0..3
            uint nib_high = j >> 2;         // 0=low nibble, 1=high nibble
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