// mul_mat_vec_q5_0_mr_lds.hlsl - Multi-row Q5_0 matvec (M=1, 2 rows/group)
//                                with LDS pre-decode of scales (wave64-tuned).
//
// Mirrors mul_mat_vec_glu_q5_0.hlsl's optimization shape, scoped down from
// 4 accumulators (gate0/gate1/up0/up1) to 2 accumulators (row0/row1) since
// this is the standalone (non-fused) attention/output projection matvec.
//
// GROUP_SIZE = 64 fully populates one AMD wave64; on wave32 devices the
// group runs as 2 waves and a tree reduction combines them; on wave16 as 4.
// Each iteration processes BLOCKS_PER_ITER blocks in parallel — the wave
// is split into BLOCKS_PER_ITER × 32-lane sub-groups indexed by sub_block.
//
// LDS pre-decode of scales: the 2 × num_blocks (d, qh) tuples for the two
// output rows are cooperatively loaded into shared memory once before the
// K loop.  The K loop then reads scales from LDS instead of issuing 4
// uniform memory loads per iteration.  Capped at MAX_BLOCKS = 32 (handles
// K up to 1024).

#include "ggml_common.hlsli"

#define GROUP_SIZE       64
#define BLOCKS_PER_ITER  2
#define QK5_0            32
#define Q5_0_BSIZE       22
#define MAX_BLOCKS       32

groupshared float  scales_d[2 * MAX_BLOCKS];   // d0, d1 per block
groupshared uint   scales_qh[2 * MAX_BLOCKS];  // qh0, qh1 per block
groupshared float  shared_acc[128];

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

int dequant_q5_0_qs(ByteAddressBuffer buf, uint block_off, uint qh, uint elem) {
    uint qs_idx = (elem < 16) ? elem : (elem - 16);
    uint qs_word = read_u32_fast(buf, block_off + 6 + (qs_idx & ~3u));
    uint qs_byte = (qs_word >> ((qs_idx & 3u) * 8u)) & 0xFFu;

    if (elem < 16) {
        uint xh = ((qh >> elem) << 4) & 0x10u;
        return (int)((qs_byte & 0x0Fu) | xh) - 16;
    } else {
        uint jj = elem - 16;
        uint xh = ((qh >> (jj + 12)) & 0x10u);
        return (int)((qs_byte >> 4) | xh) - 16;
    }
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2;
    if (row0 >= ne0) return;
    uint row1 = min(row0 + 1, ne0 - 1);

    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K          = ne00;
    uint num_blocks = K / QK5_0;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + row1 * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    // Phase A: cooperatively pre-decode all (d, qh) tuples into LDS.
    // Each lane handles one (row, block) entry; with 64 lanes and
    // 2 * num_blocks entries (e.g. 2*18 = 36 for K=576), we issue
    // ceil(2*num_blocks / 64) iterations.
    uint total_entries = 2u * num_blocks;
    for (uint e = local_id; e < total_entries; e += GROUP_SIZE) {
        uint row_idx = e & 1u;       // 0 or 1
        uint b       = e >> 1;       // block index
        uint base    = (row_idx == 0u) ? src0_row0 : src0_row1;
        uint blk_off = base + b * Q5_0_BSIZE;
        scales_d[e]  = read_f16_v(src0, blk_off);
        scales_qh[e] = read_u32_fast(src0, blk_off + 2);
    }
    GroupMemoryBarrierWithGroupSync();

    uint elem      = local_id & (QK5_0 - 1);   // 0..31
    uint sub_block = local_id >> 5;            // 0..BLOCKS_PER_ITER-1

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    for (uint block = 0; block < num_blocks; block += BLOCKS_PER_ITER) {
        uint b = block + sub_block;
        if (b >= num_blocks) {
            continue;
        }

        uint k = b * QK5_0 + elem;
        float x = asfloat(src1.Load(src1_base + k * 4));

        uint blk_off0 = src0_row0 + b * Q5_0_BSIZE;
        uint blk_off1 = src0_row1 + b * Q5_0_BSIZE;

        uint sb = b * 2u;
        float d0  = scales_d[sb + 0u];
        float d1  = scales_d[sb + 1u];
        uint  qh0 = scales_qh[sb + 0u];
        uint  qh1 = scales_qh[sb + 1u];

        int v0 = dequant_q5_0_qs(src0, blk_off0, qh0, elem);
        int v1 = dequant_q5_0_qs(src0, blk_off1, qh1, elem);

        acc0 += d0 * float(v0) * x;
        acc1 += d1 * float(v1) * x;
    }

    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);

    uint wave_id   = local_id / WARP_SIZE;
    uint num_waves = (GROUP_SIZE + WARP_SIZE - 1) / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id]              = wave_sum0;
        shared_acc[num_waves + wave_id]  = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_acc[local_id]              += shared_acc[local_id + s];
            shared_acc[num_waves + local_id]  += shared_acc[num_waves + local_id + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        float result0 = shared_acc[0];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[num_waves];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
