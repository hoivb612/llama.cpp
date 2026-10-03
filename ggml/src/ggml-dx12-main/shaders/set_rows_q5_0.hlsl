// set_rows_q5_0.hlsl - SET_ROWS with F32 src0 -> Q5_0 dst.
//
// Vulkan parity: copy_to_quant.comp DATA_A_Q5_0 quantize.
// One thread per Q5_0 block (32 F32 elements). 32 threads per workgroup.
// Block layout: { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } = 22 bytes.
#include "ggml_common.hlsli"

#define QK5_0 32

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint local_id : SV_GroupThreadID) {
    uint global_thread = (gid.z * 262144u + gid.y * 512u + gid.x) * 32u + local_id;
    uint block_idx = global_thread;

    uint nb_per_row   = ne00 / QK5_0;
    uint total_blocks = nb_per_row * ne01 * ne02 * ne03;
    if (block_idx >= total_blocks) return;

    uint bi0 = block_idx % nb_per_row;
    uint rem = block_idx / nb_per_row;
    uint i1  = rem % ne01;
    rem      = rem / ne01;
    uint i2  = rem % ne02;
    uint i3  = rem / ne02;

    uint i2_idx  = ne11 > 0 ? (i2 % ne11) : 0;
    uint i3_idx  = ne12 > 0 ? (i3 % ne12) : 0;
    uint idx_off = src1_offset + i1 * nb10 + i2_idx * nb11 + i3_idx * nb12;
    int  row_idx = asint(src1.Load(idx_off));

    uint base_i0 = bi0 * QK5_0;
    uint off0    = src0_offset + base_i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;

    float amax = 0.0f;
    float vmax = 0.0f;
    [unroll] for (uint j = 0; j < QK5_0; ++j) {
        float v = asfloat(src0.Load(off0 + j * nb00));
        if (abs(v) > amax) { amax = abs(v); vmax = v; }
    }
    float d  = vmax / -16.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    uint dst_block_off = dst_offset + bi0 * 22u + (uint)row_idx * nb1 + i2 * nb2 + i3 * nb3;
    dst.Store<uint16_t>(dst_block_off, asuint16((float16_t)d));

    uint qh = 0u;
    // Compute all 16 packed qs bytes + qh bits.  Two consecutive bytes are
    // packed into one uint16_t store, so we iterate 8 stores.
    [unroll] for (uint k = 0; k < 8; ++k) {
        uint  b0 = 2u * k;
        uint  b1 = b0 + 1u;
        float x0_lo = asfloat(src0.Load(off0 + (b0)              * nb00)) * id;
        float x0_hi = asfloat(src0.Load(off0 + (b0 + (QK5_0/2u)) * nb00)) * id;
        float x1_lo = asfloat(src0.Load(off0 + (b1)              * nb00)) * id;
        float x1_hi = asfloat(src0.Load(off0 + (b1 + (QK5_0/2u)) * nb00)) * id;
        uint  xi0_lo = (uint)min(31, (int)(x0_lo + 16.5f));
        uint  xi0_hi = (uint)min(31, (int)(x0_hi + 16.5f));
        uint  xi1_lo = (uint)min(31, (int)(x1_lo + 16.5f));
        uint  xi1_hi = (uint)min(31, (int)(x1_hi + 16.5f));
        uint  byte0  = (xi0_lo & 0xFu) | ((xi0_hi & 0xFu) << 4);
        uint  byte1  = (xi1_lo & 0xFu) | ((xi1_hi & 0xFu) << 4);
        uint16_t packed = (uint16_t)(byte0 | (byte1 << 8));
        // qs starts at offset 2 (d) + 4 (qh) = 6.
        dst.Store<uint16_t>(dst_block_off + 6u + k * 2u, packed);

        qh |= ((xi0_lo & 0x10u) >> 4) << (b0 + 0u);
        qh |= ((xi0_hi & 0x10u) >> 4) << (b0 + (QK5_0/2u));
        qh |= ((xi1_lo & 0x10u) >> 4) << (b1 + 0u);
        qh |= ((xi1_hi & 0x10u) >> 4) << (b1 + (QK5_0/2u));
    }
    dst.Store<uint16_t>(dst_block_off + 2u,      (uint16_t)(qh & 0xFFFFu));
    dst.Store<uint16_t>(dst_block_off + 2u + 2u, (uint16_t)((qh >> 16) & 0xFFFFu));
}
