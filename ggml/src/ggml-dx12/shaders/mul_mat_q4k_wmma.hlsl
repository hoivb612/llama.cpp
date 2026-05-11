// mul_mat_q4k_wmma.hlsl - Register-blocked tiled MUL_MAT with Q4_K weights
//
// Cooperative dequant: threads cooperatively load raw Q4_K block bytes into
// shared memory, decode scales once per column, then dequant nibbles from LDS.
// This reduces global memory traffic by ~4x vs per-element scalar dequant.
//
// BK=32 aligns with Q4_K sub-block halves (each 64-element sub-block has
// 32 low-nibble + 32 high-nibble elements sharing one scale/min pair).
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3
//
// Build variants:
//   default       : tile_a/tile_b stored as F32 (works on any SM 6.0 device)
//   USE_F16_TILE=1: tile_a/tile_b stored as float16_t. Halves LDS bandwidth
//                   for the tile arrays and lets the compiler emit packed
//                   FP16 multiplies (e.g. v_pk_mul_f16 on RDNA2). Requires
//                   -enable-16bit-types and SM >= 6.2. Accumulators stay F32.
#include "ggml_common.hlsli"

#ifdef USE_F16_TILE
typedef float16_t tile_t;
#else
typedef float     tile_t;
#endif

#define QK_K 256
#define Q4K_BLOCK_SIZE 144

#define BM 32
#define BN 32
#define BK 32

// Shared memory for cooperative Q4_K dequant:
//   raw_qs[BN][8]: 32 nibble bytes per column, stored as 8 uint32 (4 bytes each)
//   col_dm[BN][2]: (dall, dmin) per column
//   col_sc[BN][2]: (scale, min) per column (for current sub-block pair)
groupshared uint  raw_qs[BN][8];   // 32x8 = 256 uints = 1 KB
groupshared float col_dm[BN][2];   // 32x2 = 64 floats = 256 B
groupshared float col_sc[BN][2];   // 32x2 = 64 floats = 256 B

// tile_a/tile_b dominate LDS use. F32: 4 KB each. F16: 2 KB each.
groupshared tile_t tile_a[BM][BK];
groupshared tile_t tile_b[BK][BN];

uint read_byte_q4k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

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

        // Load tile_b: BK × BN from src0 -- cooperative Q4_K dequant
        //
        // Phase 1: Load raw block data into shared memory.
        //   - Header (d, dmin): 1 Load per column (256 threads / 32 cols = 8 threads idle)
        //   - Scales: decode for current sub-block pair
        //   - qs nibble bytes: 8 uint32 loads per column = 32 bytes = 32 nibbles
        //
        // Phase 2: Dequant from shared memory (no global loads).
        {
            // Determine which Q4_K block and sub-position this K-tile maps to
            uint block_idx    = k_start / QK_K;
            uint elem_in_blk  = k_start % QK_K;   // 0, 32, 64, ..., 224
            uint sub_chunk    = elem_in_blk / 64;  // 0..3 (which 64-element chunk)
            bool is_high_half = (elem_in_blk % 64) >= 32;  // low or high nibbles

            // Phase 1a: First 32 threads load headers and scales (one per column)
            if (flat_id < BN) {
                uint n = flat_id;
                uint global_n = col_block * BN + n;
                if (global_n < ne01) {
                    uint row_off = src0_offset + global_n * nb01
                                 + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q4K_BLOCK_SIZE;

                    // Load d, dmin (packed F16 pair at block start)
                    uint dm_raw = src0.Load(block_off);
                    col_dm[n][0] = f16_to_f32(dm_raw & 0xFFFFu);  // dall
                    col_dm[n][1] = f16_to_f32(dm_raw >> 16);       // dmin

                    // Decode scale and min for this sub-block
                    uint scales_off = block_off + 4;
                    uint is = sub_chunk * 2 + (is_high_half ? 1u : 0u);
                    uint sc_raw, mb_raw;
                    if (is < 4) {
                        sc_raw = read_byte_q4k(src0, scales_off + is) & 0x3Fu;
                        mb_raw = read_byte_q4k(src0, scales_off + is + 4) & 0x3Fu;
                    } else {
                        uint lo_sc = read_byte_q4k(src0, scales_off + is + 4) & 0x0Fu;
                        uint hi_sc = (read_byte_q4k(src0, scales_off + is - 4) >> 6) & 0x03u;
                        sc_raw = lo_sc | (hi_sc << 4);
                        uint lo_mb = (read_byte_q4k(src0, scales_off + is + 4) >> 4) & 0x0Fu;
                        uint hi_mb = (read_byte_q4k(src0, scales_off + is) >> 6) & 0x03u;
                        mb_raw = lo_mb | (hi_mb << 4);
                    }
                    col_sc[n][0] = col_dm[n][0] * (float)sc_raw;  // d * scale
                    col_sc[n][1] = col_dm[n][1] * (float)mb_raw;  // dmin * min
                } else {
                    col_dm[n][0] = 0.0f; col_dm[n][1] = 0.0f;
                    col_sc[n][0] = 0.0f; col_sc[n][1] = 0.0f;
                }
            }

            // Phase 1b: All 256 threads cooperatively load qs bytes
            // 32 columns × 8 uint32 = 256 loads, perfectly 1:1 with threads
            {
                uint col = flat_id / 8;   // which column (0..31)
                uint qw  = flat_id % 8;   // which uint32 within the 32 bytes (0..7)

                uint global_n = col_block * BN + col;
                if (global_n < ne01 && k_start < K) {
                    uint row_off = src0_offset + global_n * nb01
                                 + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q4K_BLOCK_SIZE;
                    // qs starts at block_off + 16, each sub-chunk is 32 bytes
                    uint qs_base = block_off + 16 + sub_chunk * 32;
                    raw_qs[col][qw] = src0.Load(qs_base + qw * 4);
                } else {
                    raw_qs[col][qw] = 0;
                }
            }

            GroupMemoryBarrierWithGroupSync();

            // Phase 2: Dequant from shared memory into tile_b
            // Each thread handles 4 elements (same as original)
            {
                uint base = flat_id * 4;
                [unroll] for (uint e = 0; e < 4; e++) {
                    uint idx = base + e;
                    uint k_local = idx / BN;   // 0..31
                    uint n_local = idx % BN;   // 0..31

                    float val = 0.0f;
                    if ((k_start + k_local) < K && (col_block * BN + n_local) < ne01) {
                        // Extract nibble from raw_qs[n_local][...]
                        // k_local is the byte index within the 32-byte qs chunk
                        uint word_idx = k_local / 4;  // which uint32 (0..7)
                        uint byte_in_word = k_local % 4;
                        uint qs_byte = (raw_qs[n_local][word_idx] >> (byte_in_word * 8)) & 0xFFu;

                        float d = col_sc[n_local][0];
                        float m = col_sc[n_local][1];
                        float q;
                        if (!is_high_half) {
                            q = (float)(qs_byte & 0x0Fu);
                        } else {
                            q = (float)(qs_byte >> 4);
                        }
                        val = d * q - m;
                    }
                    tile_b[k_local][n_local] = (tile_t)val;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < BK; k++) {
            tile_t a0 = tile_a[ty * 2    ][k];
            tile_t a1 = tile_a[ty * 2 + 1][k];
            tile_t b0 = tile_b[k][tx * 2    ];
            tile_t b1 = tile_b[k][tx * 2 + 1];
            // Multiply in tile_t precision (FP16 packed when USE_F16_TILE),
            // accumulate in F32 to keep numerical headroom across the K reduction.
            acc00 += (float)(a0 * b0);
            acc01 += (float)(a0 * b1);
            acc10 += (float)(a1 * b0);
            acc11 += (float)(a1 * b1);
        }

        GroupMemoryBarrierWithGroupSync();
    }

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
