// mul_mat_q5_0_wmma.hlsl - Register-blocked tiled MUL_MAT with Q5_0 weights
// Dequantizes Q5_0 into groupshared memory, then uses 2x2 register blocking.
// BK=32 matches QK5_0 exactly (one Q5_0 block per K-tile).
//
// Q5_0 block layout (22 bytes per 32 elements):
//   offset 0: d (float16, 2 bytes)
//   offset 2: qh (uint32, 4 bytes) -- high bits for all 32 elements
//   offset 6: qs[16] (uint8) -- low 4 bits, packed pairs
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3
//
// Build variants: see mul_mat_q4k_wmma.hlsl for the USE_F16_TILE story.
#include "ggml_common.hlsli"

#ifdef USE_F16_TILE
typedef float16_t tile_t;
#else
typedef float     tile_t;
#endif

#define QK5_0 32
#define Q5_0_BSIZE 22

#define BM 32
#define BN 32
#define BK 32

uint read_byte_q50(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float read_f16_q50(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    uint shift = (byte_off & 2u) * 8u;
    return f16_to_f32((word >> shift) & 0xFFFFu);
}

// Dequantize one element from a Q5_0 block
// Uses the exact same logic as mul_mat_q5_0.hlsl (flat shader)
float dequant_q5_0(ByteAddressBuffer buf, uint block_off, uint elem_in_block) {
    // d: float16 at block_off
    uint d_word = buf.Load(block_off & ~3u);
    uint d_shift = (block_off & 2u) * 8u;
    float d = f16_to_f32((d_word >> d_shift) & 0xFFFFu);

    // qh: uint32 at block_off + 2 (may not be dword-aligned)
    // Reconstruct from individual byte reads to handle arbitrary alignment
    uint qh_off = block_off + 2;
    uint w0 = buf.Load(qh_off & ~3u);
    uint w1 = buf.Load((qh_off + 3) & ~3u); // might be in next dword
    // Extract 4 bytes starting at qh_off
    uint shift0 = (qh_off & 3u) * 8u;
    uint qh;
    if ((qh_off & 3u) == 0u) {
        qh = w0;
    } else {
        // Straddles two dwords
        qh = (w0 >> shift0) | (w1 << (32u - shift0));
    }

    uint qs_off = block_off + 6;

    if (elem_in_block < 16) {
        uint j = elem_in_block;
        uint qs_byte_off = qs_off + j;
        uint qs_w = buf.Load(qs_byte_off & ~3u);
        uint qs_byte = (qs_w >> ((qs_byte_off & 3u) * 8u)) & 0xFFu;
        uint xh = ((qh >> j) << 4) & 0x10u;
        int x = (int)((qs_byte & 0x0Fu) | xh) - 16;
        return d * (float)x;
    } else {
        uint j = elem_in_block - 16;
        uint qs_byte_off = qs_off + j;
        uint qs_w = buf.Load(qs_byte_off & ~3u);
        uint qs_byte = (qs_w >> ((qs_byte_off & 3u) * 8u)) & 0xFFu;
        uint xh = (qh >> (j + 12)) & 0x10u;
        int x = (int)((qs_byte >> 4) | xh) - 16;
        return d * (float)x;
    }
}

groupshared tile_t tile_a[BM][BK]; // src1: batch x K
groupshared tile_t tile_b[BK][BN]; // src0: K x output features (dequantized)

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;
    uint flat_id = ty * 16 + tx;

    uint col_block = gid.x;
    uint row_block = gid.y;
    uint batch = gid.z;

    uint i2 = batch % ne2;
    uint i3 = batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_k_tiles = (K + BK - 1) / BK;

    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // Load tile_a: BM x BK from src1 (1024 elements, 256 threads, 4 each)
        {
            uint base = flat_id * 4;
            [unroll] for (uint e = 0; e < 4; e++) {
                uint idx = base + e;
                uint m_local = idx / BK;
                uint k_local = idx % BK;
                uint global_m = row_block * BM + m_local;
                uint global_k = k_start + k_local;
                float val = 0.0f;
                if (global_m < ne11 && global_k < K) {
                    uint off = offset_4d(global_k, global_m, i2, i3,
                                         nb10, nb11, nb12, nb13, src1_offset);
                    val = load_auto(src1, off, src1_esize);
                }
                tile_a[m_local][k_local] = (tile_t)val;
            }
        }

        // Load tile_b: BK x BN from src0 -- dequantize Q5_0
        // BK=32 = QK5_0, so each column needs exactly one Q5_0 block
        {
            uint base = flat_id * 4;
            [unroll] for (uint e = 0; e < 4; e++) {
                uint idx = base + e;
                uint k_local = idx / BN;
                uint n_local = idx % BN;
                uint global_k = k_start + k_local;
                uint global_n = col_block * BN + n_local;
                float val = 0.0f;
                if (global_k < K && global_n < ne01) {
                    uint block_idx = global_k / QK5_0;
                    uint elem_in_block = global_k % QK5_0;
                    uint row_off = src0_offset + global_n * nb01
                                 + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q5_0_BSIZE;
                    val = dequant_q5_0(src0, block_off, elem_in_block);
                }
                tile_b[k_local][n_local] = (tile_t)val;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // 2x2 register-blocked accumulation from shared memory
        [unroll]
        for (uint k = 0; k < BK; k++) {
            tile_t a0 = tile_a[ty * 2    ][k];
            tile_t a1 = tile_a[ty * 2 + 1][k];
            tile_t b0 = tile_b[k][tx * 2    ];
            tile_t b1 = tile_b[k][tx * 2 + 1];
            // Multiply in tile_t precision; accumulate in F32 across the K reduction.
            acc00 += (float)(a0 * b0);
            acc01 += (float)(a0 * b1);
            acc10 += (float)(a1 * b0);
            acc11 += (float)(a1 * b1);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write 2x2 output tile
    uint m0 = row_block * BM + ty * 2;
    uint m1 = m0 + 1;
    uint n0 = col_block * BN + tx * 2;
    uint n1 = n0 + 1;

    if (m0 < ne1 && n0 < ne0) {
        uint off = offset_4d(n0, m0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc00, dst_esize);
    }
    if (m0 < ne1 && n1 < ne0) {
        uint off = offset_4d(n1, m0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc01, dst_esize);
    }
    if (m1 < ne1 && n0 < ne0) {
        uint off = offset_4d(n0, m1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc10, dst_esize);
    }
    if (m1 < ne1 && n1 < ne0) {
        uint off = offset_4d(n1, m1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc11, dst_esize);
    }
}
