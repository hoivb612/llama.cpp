// mul_mat_vec_q6k_dp4a_nc2.hlsl - Q6_K dp4a matvec for M==2 batch.
//
// NUM_ROWS=2 x NUM_COLS=2 outputs per workgroup. Decodes each Q6_K super-block
// once, reuses decoded weights across both activation columns.

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  256
#endif
#define QK_K        256
#define Q6K_BSIZE   210
#define Q8_1_BSIZE  36
#define NUM_ROWS    2
#define NUM_COLS    2

groupshared float shared_acc00[64];
groupshared float shared_acc01[64];
groupshared float shared_acc10[64];
groupshared float shared_acc11[64];

uint load_u32_u(ByteAddressBuffer buf, uint byte_off) {
    uint align_off = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint w0 = buf.Load(align_off);
    if (shift == 0) return w0;
    uint w1 = buf.Load(align_off + 4);
    return (w0 >> shift) | (w1 << (32u - shift));
}

uint read_byte_q6(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

void decode_q6k_row(uint block_off, uint t,
                    out float d_super, out int scale_int8,
                    out uint uq0, out uint uq1, out uint uq2, out uint uq3) {
    uint d_off = block_off + 208;
    uint d_word = src0.Load(d_off & ~3u);
    d_super = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

    scale_int8 = (int)read_byte_q6(src0, block_off + 192 + t);
    if (scale_int8 >= 128) scale_int8 -= 256;

    uint ip  = t >> 3;
    uint sub = t & 7u;
    uint ql_base_in_block = 64u * ip + 16u * (sub & 3u);
    uint qh_base_in_block = 128u + 32u * ip + 16u * (sub & 1u);
    uint qh_shift = (sub & ~1u);

    bool high_nib = (sub >= 4u);

    uint ql_w0 = load_u32_u(src0, block_off + ql_base_in_block + 0);
    uint ql_w1 = load_u32_u(src0, block_off + ql_base_in_block + 4);
    uint ql_w2 = load_u32_u(src0, block_off + ql_base_in_block + 8);
    uint ql_w3 = load_u32_u(src0, block_off + ql_base_in_block + 12);

    uint qh_w0 = load_u32_u(src0, block_off + qh_base_in_block + 0);
    uint qh_w1 = load_u32_u(src0, block_off + qh_base_in_block + 4);
    uint qh_w2 = load_u32_u(src0, block_off + qh_base_in_block + 8);
    uint qh_w3 = load_u32_u(src0, block_off + qh_base_in_block + 12);

    if (high_nib) {
        ql_w0 = (ql_w0 >> 4) & 0x0F0F0F0Fu;
        ql_w1 = (ql_w1 >> 4) & 0x0F0F0F0Fu;
        ql_w2 = (ql_w2 >> 4) & 0x0F0F0F0Fu;
        ql_w3 = (ql_w3 >> 4) & 0x0F0F0F0Fu;
    } else {
        ql_w0 = ql_w0 & 0x0F0F0F0Fu;
        ql_w1 = ql_w1 & 0x0F0F0F0Fu;
        ql_w2 = ql_w2 & 0x0F0F0F0Fu;
        ql_w3 = ql_w3 & 0x0F0F0F0Fu;
    }

    qh_w0 = (qh_w0 >> qh_shift) & 0x03030303u;
    qh_w1 = (qh_w1 >> qh_shift) & 0x03030303u;
    qh_w2 = (qh_w2 >> qh_shift) & 0x03030303u;
    qh_w3 = (qh_w3 >> qh_shift) & 0x03030303u;

    uq0 = ql_w0 | (qh_w0 << 4);
    uq1 = ql_w1 | (qh_w1 << 4);
    uq2 = ql_w2 | (qh_w2 << 4);
    uq3 = ql_w3 | (qh_w3 << 4);
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * NUM_ROWS;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;
    uint num_q8_per_vec = K / 32;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;

    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_batch_base = src1_offset +
        ((i3_q8 * ne12 + i2_q8) * ne11) * num_q8_per_vec * Q8_1_BSIZE;
    uint q8_col0_base = q8_batch_base + 0u * num_q8_per_vec * Q8_1_BSIZE;
    uint q8_col1_base = q8_batch_base + 1u * num_q8_per_vec * Q8_1_BSIZE;

    uint it_size = GROUP_SIZE / 16;
    uint itid = tid % 16;
    uint ix = tid / 16;

    uint q8_blk = itid / 2;
    uint q8_byte_off = 16u * (itid & 1u);

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        // Decode rows ONCE per super-block.
        float d_super0; int scale_int80;
        uint uq0_0, uq1_0, uq2_0, uq3_0;
        decode_q6k_row(src0_row0 + block_idx * Q6K_BSIZE, itid,
                       d_super0, scale_int80, uq0_0, uq1_0, uq2_0, uq3_0);

        float d_super1; int scale_int81;
        uint uq0_1, uq1_1, uq2_1, uq3_1;
        decode_q6k_row(src0_row1 + block_idx * Q6K_BSIZE, itid,
                       d_super1, scale_int81, uq0_1, uq1_1, uq2_1, uq3_1);

        // --- Col 0 ---
        {
            uint q8_super_base = q8_col0_base + block_idx * 8u * Q8_1_BSIZE;
            uint q8_off = q8_super_base + q8_blk * Q8_1_BSIZE;

            uint ds = src1.Load(q8_off);
            float q8d = f16_to_f32(ds & 0xFFFFu);

            uint q8_qs0 = src1.Load(q8_off + 4 + q8_byte_off + 0);
            uint q8_qs1 = src1.Load(q8_off + 4 + q8_byte_off + 4);
            uint q8_qs2 = src1.Load(q8_off + 4 + q8_byte_off + 8);
            uint q8_qs3 = src1.Load(q8_off + 4 + q8_byte_off + 12);

            // separate zero-init accumulators (chained-constant form is
            // miscompiled on some drivers -- see mul_mat_vec_q4k_dp4a.hlsl)
            int p0 = 0; p0 = dot4add_i8packed(0x01010101u, q8_qs0, p0);
            int p1 = 0; p1 = dot4add_i8packed(0x01010101u, q8_qs1, p1);
            int p2 = 0; p2 = dot4add_i8packed(0x01010101u, q8_qs2, p2);
            int p3 = 0; p3 = dot4add_i8packed(0x01010101u, q8_qs3, p3);
            int q8_psum = p0 + p1 + p2 + p3;

            int isx0 = 0;
            isx0 = dot4add_i8packed(uq0_0, q8_qs0, isx0);
            isx0 = dot4add_i8packed(uq1_0, q8_qs1, isx0);
            isx0 = dot4add_i8packed(uq2_0, q8_qs2, isx0);
            isx0 = dot4add_i8packed(uq3_0, q8_qs3, isx0);
            float scale_f0 = d_super0 * float(scale_int80) * q8d;
            acc00 = mad(scale_f0, float(isx0 - 32 * q8_psum), acc00);

            int isx1 = 0;
            isx1 = dot4add_i8packed(uq0_1, q8_qs0, isx1);
            isx1 = dot4add_i8packed(uq1_1, q8_qs1, isx1);
            isx1 = dot4add_i8packed(uq2_1, q8_qs2, isx1);
            isx1 = dot4add_i8packed(uq3_1, q8_qs3, isx1);
            float scale_f1 = d_super1 * float(scale_int81) * q8d;
            acc10 = mad(scale_f1, float(isx1 - 32 * q8_psum), acc10);
        }

        // --- Col 1 ---
        {
            uint q8_super_base = q8_col1_base + block_idx * 8u * Q8_1_BSIZE;
            uint q8_off = q8_super_base + q8_blk * Q8_1_BSIZE;

            uint ds = src1.Load(q8_off);
            float q8d = f16_to_f32(ds & 0xFFFFu);

            uint q8_qs0 = src1.Load(q8_off + 4 + q8_byte_off + 0);
            uint q8_qs1 = src1.Load(q8_off + 4 + q8_byte_off + 4);
            uint q8_qs2 = src1.Load(q8_off + 4 + q8_byte_off + 8);
            uint q8_qs3 = src1.Load(q8_off + 4 + q8_byte_off + 12);

            int p0 = 0; p0 = dot4add_i8packed(0x01010101u, q8_qs0, p0);
            int p1 = 0; p1 = dot4add_i8packed(0x01010101u, q8_qs1, p1);
            int p2 = 0; p2 = dot4add_i8packed(0x01010101u, q8_qs2, p2);
            int p3 = 0; p3 = dot4add_i8packed(0x01010101u, q8_qs3, p3);
            int q8_psum = p0 + p1 + p2 + p3;

            int isx0 = 0;
            isx0 = dot4add_i8packed(uq0_0, q8_qs0, isx0);
            isx0 = dot4add_i8packed(uq1_0, q8_qs1, isx0);
            isx0 = dot4add_i8packed(uq2_0, q8_qs2, isx0);
            isx0 = dot4add_i8packed(uq3_0, q8_qs3, isx0);
            float scale_f0 = d_super0 * float(scale_int80) * q8d;
            acc01 = mad(scale_f0, float(isx0 - 32 * q8_psum), acc01);

            int isx1 = 0;
            isx1 = dot4add_i8packed(uq0_1, q8_qs0, isx1);
            isx1 = dot4add_i8packed(uq1_1, q8_qs1, isx1);
            isx1 = dot4add_i8packed(uq2_1, q8_qs2, isx1);
            isx1 = dot4add_i8packed(uq3_1, q8_qs3, isx1);
            float scale_f1 = d_super1 * float(scale_int81) * q8d;
            acc11 = mad(scale_f1, float(isx1 - 32 * q8_psum), acc11);
        }
    }

    float wave_sum00 = WaveActiveSum(acc00);
    float wave_sum01 = WaveActiveSum(acc01);
    float wave_sum10 = WaveActiveSum(acc10);
    float wave_sum11 = WaveActiveSum(acc11);
    uint wave_lanes = WaveGetLaneCount();
    uint wave_id = tid / wave_lanes;
    uint num_waves = (GROUP_SIZE + wave_lanes - 1) / wave_lanes;
    if (num_waves == 0) num_waves = 1;

    if (WaveIsFirstLane()) {
        shared_acc00[wave_id] = wave_sum00;
        shared_acc01[wave_id] = wave_sum01;
        shared_acc10[wave_id] = wave_sum10;
        shared_acc11[wave_id] = wave_sum11;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float r00 = shared_acc00[0];
        float r01 = shared_acc01[0];
        float r10 = shared_acc10[0];
        float r11 = shared_acc11[0];
        for (uint w = 1; w < num_waves; w++) {
            r00 += shared_acc00[w];
            r01 += shared_acc01[w];
            r10 += shared_acc10[w];
            r11 += shared_acc11[w];
        }
        uint off00 = offset_4d(row0,     0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        uint off01 = offset_4d(row0,     1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off00, r00, dst_esize);
        store_auto(dst, off01, r01, dst_esize);

        if (row0 + 1 < ne0) {
            uint off10 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            uint off11 = offset_4d(row0 + 1, 1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off10, r10, dst_esize);
            store_auto(dst, off11, r11, dst_esize);
        }
    }
}
