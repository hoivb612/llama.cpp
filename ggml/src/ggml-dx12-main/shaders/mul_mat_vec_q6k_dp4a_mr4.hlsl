// mul_mat_vec_q6k_dp4a_mr4.hlsl - dp4a Q6_K matvec, 4 rows/group
//
// 4-row variant of mul_mat_vec_q6k_dp4a.hlsl. The 2-row shader re-reads the
// whole Q8_1 activation vector once per group; at N=17408 that is 8704 groups
// each pulling K/32*36 bytes, so activations account for roughly a third of
// the traffic the matvec generates. Decoding 4 rows per group halves both the
// dispatch count and the activation re-reads for the same weight bytes.
//
// Reference: mul_mat_vec_q4k_dp4a_mr4.hlsl does the same for Q4_K, following
// Vulkan's `rm_kq=4`.
//
// Dispatch: groups_x = (N+3)/4, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  256
#endif
#define QK_K        256
#define Q6K_BSIZE   210
#define Q8_1_BSIZE  36
#define NUM_ROWS    4

// Worst-case LDS: 4 rows x 64 waves (GROUP_SIZE=256 / HW wave=4).
groupshared float shared_acc_mr4[NUM_ROWS][64];

// 16-byte fetch at a possibly-misaligned offset. Four separate load_u32_u
// calls cost 8 loads on the misaligned path and re-read each boundary word
// twice; this is 1 Load4 when aligned and 5 word loads when not. Same helper
// as mul_mat_vec_q6k_mr_blocked.hlsl.
//
// Load4 requires 4-byte alignment: desktop AMD GCN/RDNA tolerates 2-byte
// alignment, but some AMD console HW (Xbox GDKX) silently masks the low bits
// and returns the wrong 16 bytes, hence the explicit reconstruction.
uint4 load4_u_q6k(uint byte_off) {
    uint shift = (byte_off & 3u) * 8u;
    if (shift == 0u) {
        return src0.Load4(byte_off);
    }
    uint base = byte_off & ~3u;
    uint w0 = src0.Load(base);
    uint w1 = src0.Load(base + 4);
    uint w2 = src0.Load(base + 8);
    uint w3 = src0.Load(base + 12);
    // Addressed defensively: this load sits after an early return, but the
    // compiler may still speculate it, and src0 is bound as a root SRV, which
    // D3D12 does not bounds check. base+16 for an aligned offset would read
    // past the end of the last tensor in the allocation.
    uint w4 = src0.Load(base + (shift == 0u ? 12u : 16u));
    uint isr = 32u - shift;
    uint4 r;
    r.x = (w0 >> shift) | (w1 << isr);
    r.y = (w1 >> shift) | (w2 << isr);
    r.z = (w2 >> shift) | (w3 << isr);
    r.w = (w3 >> shift) | (w4 << isr);
    return r;
}

uint read_byte_q6(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

// Decode Q6_K data for one row's superblock for thread `t` (subblock index 0..15).
// Outputs: d_super, scale_int8, and 4 packed unsigned int8 vectors (q in [0,63]).
void decode_q6k_row_mr4(uint block_off, uint t,
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

    uint4 ql4 = load4_u_q6k(block_off + ql_base_in_block);
    uint ql_w0 = ql4.x;
    uint ql_w1 = ql4.y;
    uint ql_w2 = ql4.z;
    uint ql_w3 = ql4.w;

    uint4 qh4 = load4_u_q6k(block_off + qh_base_in_block);
    uint qh_w0 = qh4.x;
    uint qh_w1 = qh4.y;
    uint qh_w2 = qh4.z;
    uint qh_w3 = qh4.w;

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

    // Rows past the end clamp onto row0: they still decode, but their results
    // are discarded at store time. Keeps the inner loop branch-free.
    uint row_valid[NUM_ROWS];
    uint src0_row[NUM_ROWS];
    [unroll] for (uint r = 0; r < NUM_ROWS; r++) {
        uint row = row0 + r;
        row_valid[r] = (row < ne0) ? 1u : 0u;
        uint row_c = (row < ne0) ? row : row0;
        src0_row[r] = src0_base + row_c * nb01;
    }

    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_vec_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_q8_per_vec * Q8_1_BSIZE;

    uint it_size = GROUP_SIZE / 16;
    uint itid = tid % 16;
    uint ix = tid / 16;

    uint q8_blk = itid / 2;
    uint q8_byte_off = 16u * (itid & 1u);

    float acc[NUM_ROWS];
    [unroll] for (uint ra = 0; ra < NUM_ROWS; ra++) acc[ra] = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        // --- Load Q8_1 data (shared across all NUM_ROWS rows) ---
        uint q8_super_base = q8_vec_base + block_idx * 8u * Q8_1_BSIZE;
        uint q8_off = q8_super_base + q8_blk * Q8_1_BSIZE;

        uint ds = src1.Load(q8_off);
        float q8d = f16_to_f32(ds & 0xFFFFu);

        uint q8_qs0 = src1.Load(q8_off + 4 + q8_byte_off + 0);
        uint q8_qs1 = src1.Load(q8_off + 4 + q8_byte_off + 4);
        uint q8_qs2 = src1.Load(q8_off + 4 + q8_byte_off + 8);
        uint q8_qs3 = src1.Load(q8_off + 4 + q8_byte_off + 12);

        // Separate zero-initialized accumulators: the chained-constant form is
        // miscompiled on some drivers. See mul_mat_vec_q6k_dp4a.hlsl.
        int p0 = 0; p0 = dot4add_i8packed(0x01010101u, q8_qs0, p0);
        int p1 = 0; p1 = dot4add_i8packed(0x01010101u, q8_qs1, p1);
        int p2 = 0; p2 = dot4add_i8packed(0x01010101u, q8_qs2, p2);
        int p3 = 0; p3 = dot4add_i8packed(0x01010101u, q8_qs3, p3);
        int q8_psum = p0 + p1 + p2 + p3;

        [unroll] for (uint r = 0; r < NUM_ROWS; r++) {
            float d_super; int scale_int8;
            uint uq0, uq1, uq2, uq3;
            decode_q6k_row_mr4(src0_row[r] + block_idx * Q6K_BSIZE, itid,
                               d_super, scale_int8, uq0, uq1, uq2, uq3);

            int isx = 0;
            isx = dot4add_i8packed(uq0, q8_qs0, isx);
            isx = dot4add_i8packed(uq1, q8_qs1, isx);
            isx = dot4add_i8packed(uq2, q8_qs2, isx);
            isx = dot4add_i8packed(uq3, q8_qs3, isx);

            float scale_f = d_super * float(scale_int8) * q8d;
            acc[r] = mad(scale_f, float(isx - 32 * q8_psum), acc[r]);
        }
    }

    // Wave-portable reduction. See mul_mat_vec_q4k_dp4a.hlsl for rationale.
    uint wave_lanes = WaveGetLaneCount();
    uint wave_id = tid / wave_lanes;
    uint num_waves = (GROUP_SIZE + wave_lanes - 1) / wave_lanes;
    if (num_waves == 0) num_waves = 1;

    [unroll] for (uint rw = 0; rw < NUM_ROWS; rw++) {
        float wave_sum = WaveActiveSum(acc[rw]);
        if (WaveIsFirstLane()) {
            shared_acc_mr4[rw][wave_id] = wave_sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        [unroll] for (uint rs = 0; rs < NUM_ROWS; rs++) {
            if (row_valid[rs] == 0u) continue;
            uint row = row0 + rs;

            float result = shared_acc_mr4[rs][0];
            for (uint w = 1; w < num_waves; w++) result += shared_acc_mr4[rs][w];
            result += load_fused_bias(row, i2, i3);

            if (mmv_scatter_active()) {
                mmv_store_scatter(row, 0u, result);
            } else {
                uint off_d = offset_4d(row, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                store_auto(dst, off_d, result, dst_esize);
            }
        }
    }
}
