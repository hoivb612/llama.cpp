// mul_mat_vec_q3k.hlsl - Cooperative matrix-vector multiply for Q3_K weights (M=1)
//
// 16 threads cooperate per Q3_K superblock (256 elements).
// Each thread processes 16 elements across 8 scale groups.
//
// Q3_K block layout (110 bytes per 256 elements):
//   hmask[32]: high bit (1 bit per element, bit-packed)
//   qs[64]:    low 2 bits (4 elements per byte)
//   scales[12]: 6-bit scales, packed
//   d (f16):   super-block scale (NO dmin)
//
// NOTE: Block size 110 is NOT 4-byte aligned; all loads use load_u32().
//
// Dispatch: groups_x = N, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE        256
#define QK_K              256
#define Q3K_BSIZE         110   // 32 + 64 + 12 + 2
#define THREADS_PER_BLOCK 16
#define BLOCKS_PER_GROUP  (GROUP_SIZE / THREADS_PER_BLOCK)

groupshared float shared_acc[GROUP_SIZE];

// Misalignment-safe uint32 load (Q3_K block size 110 is not 4-byte aligned)
uint load_u32(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i0 = group_id.y * 65535u + group_id.x;  // linearized 2D for large N (>65535)
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    uint lane = tid % THREADS_PER_BLOCK;
    uint block_group = tid / THREADS_PER_BLOCK;
    uint num_blocks = K / QK_K;

    // Thread mapping (matches Vulkan mul_mat_vec_q3_k.comp)
    uint v_im = lane / 8;           // 0 or 1: which 128-element half
    uint v_in = lane % 8;           // 0..7: position within half
    uint v_im4 = v_im * 4;
    uint l0 = 2 * v_in;             // 0,2,...,14
    uint q_offset = 32 * v_im + l0; // byte offset within qs[64]
    uint y_offset = 128 * v_im + l0; // element offset within block

    // Pre-compute hmask bit masks for 4 shift groups
    uint m_base = 0x01010101u << v_im4;
    uint hm_m0 = m_base;        // j=0
    uint hm_m1 = m_base << 1;   // j=1
    uint hm_m2 = m_base << 2;   // j=2
    uint hm_m3 = m_base << 3;   // j=3

    float acc = 0.0f;

    for (uint bg = block_group; bg < num_blocks; bg += BLOCKS_PER_GROUP) {
        uint block_off = src0_row + bg * Q3K_BSIZE;

        // Load d (f16 at byte offset 108, potentially misaligned)
        uint d_off = block_off + 108;
        uint d_word = load_u32(src0, d_off);
        float d_all = f16_to_f32(d_word & 0xFFFFu);

        // Unpack 6-bit scales from 12 packed bytes at offset 96
        uint sc_base = block_off + 96;
        uint sc0_raw = load_u32(src0, sc_base);      // scales[0..3]
        uint sc1_raw = load_u32(src0, sc_base + 4);   // scales[4..7]
        uint sc2_raw = load_u32(src0, sc_base + 8);   // scales[8..11]

        // Decode 16 6-bit scales into 4 × uint32 (4 scales per word, one per byte)
        uint aux0 = (sc0_raw & 0x0F0F0F0Fu) | (((sc2_raw >> 0) & 0x03030303u) << 4);
        uint aux1 = (sc1_raw & 0x0F0F0F0Fu) | (((sc2_raw >> 2) & 0x03030303u) << 4);
        uint aux2 = ((sc0_raw >> 4) & 0x0F0F0F0Fu) | (((sc2_raw >> 4) & 0x03030303u) << 4);
        uint aux3 = ((sc1_raw >> 4) & 0x0F0F0F0Fu) | (((sc2_raw >> 6) & 0x03030303u) << 4);

        // Select scales for this half
        uint sc_lo = (v_im == 0) ? aux0 : aux2;
        uint sc_hi = (v_im == 0) ? aux1 : aux3;

        // Extract 8 individual signed scales (6-bit, bias -32)
        float s0 = d_all * (float(int( sc_lo        & 0x3Fu)) - 32.0f);
        float s1 = d_all * (float(int((sc_lo >>  8) & 0x3Fu)) - 32.0f);
        float s2 = d_all * (float(int((sc_lo >> 16) & 0x3Fu)) - 32.0f);
        float s3 = d_all * (float(int((sc_lo >> 24) & 0x3Fu)) - 32.0f);
        float s4 = d_all * (float(int( sc_hi        & 0x3Fu)) - 32.0f);
        float s5 = d_all * (float(int((sc_hi >>  8) & 0x3Fu)) - 32.0f);
        float s6 = d_all * (float(int((sc_hi >> 16) & 0x3Fu)) - 32.0f);
        float s7 = d_all * (float(int((sc_hi >> 24) & 0x3Fu)) - 32.0f);

        // Load 4 qs bytes (2-bit low values): qs at offset 32
        uint qs_abs = block_off + 32;
        uint addr_a = qs_abs + q_offset;
        uint pair_a = load_u32(src0, addr_a) & 0xFFFFu;
        uint addr_b = qs_abs + q_offset + 16;
        uint pair_b = load_u32(src0, addr_b) & 0xFFFFu;
        uint qs_u32 = pair_a | (pair_b << 16);

        // Extract 16 2-bit low values via 4 bit-shifts
        float ql0  = float( qs_u32        & 3u);
        float ql1  = float((qs_u32 >>  8) & 3u);
        float ql2  = float((qs_u32 >> 16) & 3u);
        float ql3  = float((qs_u32 >> 24) & 3u);
        float ql4  = float((qs_u32 >>  2) & 3u);
        float ql5  = float((qs_u32 >> 10) & 3u);
        float ql6  = float((qs_u32 >> 18) & 3u);
        float ql7  = float((qs_u32 >> 26) & 3u);
        float ql8  = float((qs_u32 >>  4) & 3u);
        float ql9  = float((qs_u32 >> 12) & 3u);
        float ql10 = float((qs_u32 >> 20) & 3u);
        float ql11 = float((qs_u32 >> 28) & 3u);
        float ql12 = float((qs_u32 >>  6) & 3u);
        float ql13 = float((qs_u32 >> 14) & 3u);
        float ql14 = float((qs_u32 >> 22) & 3u);
        float ql15 = float((qs_u32 >> 30) & 3u);

        // Load 4 hmask bytes and compute high-bit corrections
        // hmask at offset 0: need bytes [l0, l0+1, l0+16, l0+17]
        uint hm_pair_a = load_u32(src0, block_off + l0) & 0xFFFFu;
        uint hm_pair_b = load_u32(src0, block_off + l0 + 16) & 0xFFFFu;
        uint hmk = ~(hm_pair_a | (hm_pair_b << 16));

        // For each shift group j, extract correction (0 or 4) for each of 4 bytes
        // When hmask bit is CLEAR → subtract 4; when SET → subtract 0
        uint hc0_u = ((hmk & hm_m0) >> (v_im4 + 0)) << 2;
        uint hc1_u = ((hmk & hm_m1) >> (v_im4 + 1)) << 2;
        uint hc2_u = ((hmk & hm_m2) >> (v_im4 + 2)) << 2;
        uint hc3_u = ((hmk & hm_m3) >> (v_im4 + 3)) << 2;

        // Unpack corrections per byte
        float h00 = float(hc0_u & 0xFFu); float h01 = float((hc0_u >> 8) & 0xFFu);
        float h02 = float((hc0_u >> 16) & 0xFFu); float h03 = float((hc0_u >> 24) & 0xFFu);
        float h10 = float(hc1_u & 0xFFu); float h11 = float((hc1_u >> 8) & 0xFFu);
        float h12 = float((hc1_u >> 16) & 0xFFu); float h13 = float((hc1_u >> 24) & 0xFFu);
        float h20 = float(hc2_u & 0xFFu); float h21 = float((hc2_u >> 8) & 0xFFu);
        float h22 = float((hc2_u >> 16) & 0xFFu); float h23 = float((hc2_u >> 24) & 0xFFu);
        float h30 = float(hc3_u & 0xFFu); float h31 = float((hc3_u >> 8) & 0xFFu);
        float h32 = float((hc3_u >> 16) & 0xFFu); float h33 = float((hc3_u >> 24) & 0xFFu);

        // Load 16 activations as 8 × Load2 (stride 16 elements = 64 bytes)
        uint y_off = src1_base + (bg * QK_K + y_offset) * 4;
        uint2 a0 = src1.Load2(y_off);
        uint2 a1 = src1.Load2(y_off + 64);
        uint2 a2 = src1.Load2(y_off + 128);
        uint2 a3 = src1.Load2(y_off + 192);
        uint2 a4 = src1.Load2(y_off + 256);
        uint2 a5 = src1.Load2(y_off + 320);
        uint2 a6 = src1.Load2(y_off + 384);
        uint2 a7 = src1.Load2(y_off + 448);

        float by0  = asfloat(a0.x); float by1  = asfloat(a0.y);
        float by2  = asfloat(a1.x); float by3  = asfloat(a1.y);
        float by4  = asfloat(a2.x); float by5  = asfloat(a2.y);
        float by6  = asfloat(a3.x); float by7  = asfloat(a3.y);
        float by8  = asfloat(a4.x); float by9  = asfloat(a4.y);
        float by10 = asfloat(a5.x); float by11 = asfloat(a5.y);
        float by12 = asfloat(a6.x); float by13 = asfloat(a6.y);
        float by14 = asfloat(a7.x); float by15 = asfloat(a7.y);

        // Dot product: d * scale * sum((qs_2bit - hmask_correction) * activation)
        acc += s0 * mad(ql0  - h00, by0,  (ql1  - h01) * by1)
            +  s1 * mad(ql2  - h02, by2,  (ql3  - h03) * by3)
            +  s2 * mad(ql4  - h10, by4,  (ql5  - h11) * by5)
            +  s3 * mad(ql6  - h12, by6,  (ql7  - h13) * by7)
            +  s4 * mad(ql8  - h20, by8,  (ql9  - h21) * by9)
            +  s5 * mad(ql10 - h22, by10, (ql11 - h23) * by11)
            +  s6 * mad(ql12 - h30, by12, (ql13 - h31) * by13)
            +  s7 * mad(ql14 - h32, by14, (ql15 - h33) * by15);
    }

    // Wave-intrinsic reduction then cross-wave via shared memory
    float wave_sum = WaveActiveSum(acc);
    uint wave_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    uint num_waves = GROUP_SIZE / WaveGetLaneCount();
    if (tid < num_waves) {
        float v = shared_acc[tid];
        v = WaveActiveSum(v);
        if (tid == 0) shared_acc[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float result = shared_acc[0];
        if (op0 == 1u) result += asfloat(src2.Load(op1 + i0 * 4));
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}
