// mul_mat_q6k_wmma.hlsl - Register-blocked tiled MUL_MAT with Q6_K weights
//
// Cooperative dequant: threads cooperatively load Q6_K block metadata (d, scale)
// once per column, then load ql/qh bytes into shared memory using alignment-safe
// reads. Dequant from LDS with zero redundant global loads.
//
// CRITICAL: Q6_K block_size=210 is NOT 4-byte aligned. All global loads MUST
// use read_byte_q6k() (which does Load(addr & ~3u) + shift) to avoid
// ByteAddressBuffer.Load() misalignment.
//
// Q6_K block layout: ql[128] + qh[64] + scales[16] + d(f16) = 210 bytes
// BK=16 aligns with Q6_K scale granularity (16 elements per scale).
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3
#include "ggml_common.hlsli"

#define QK_K 256
#define Q6K_BLOCK_SIZE 210

#define BM 32
#define BN 32
#define BK 16

// Shared memory for cooperative Q6_K dequant
groupshared uint  raw_ql[BN][16];  // 16 ql bytes per column (stored as uint for LDS efficiency)
groupshared uint  raw_qh[BN][16];  // 16 qh bytes per column
groupshared float col_ds[BN];     // d * scale per column

groupshared float tile_a[BM][BK];
groupshared float tile_b[BK][BN];

uint read_byte_q6k(ByteAddressBuffer buf, uint byte_off) {
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

        // Load tile_a from src1 (512 elements, 256 threads, 2 each)
        {
            uint base = flat_id * 2;
            [unroll] for (uint e = 0; e < 2; e++) {
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
                tile_a[m_local][k_local] = val;
            }
        }

        // Load tile_b: cooperative Q6_K dequant
        //
        // For BK=16, each K-tile maps to 16 consecutive elements within a Q6_K
        // block. All 16 share the same scale and same d. The 16 ql bytes and
        // 16 qh bytes are at consecutive (but potentially misaligned) addresses.
        //
        // Element mapping (il = elem_in_block % 128, ip = elem_in_block / 128):
        //   il 0..31:   ql at 64*ip+il,     low nibble, qh at 32*ip+(il%32), shift 0
        //   il 32..63:  ql at 64*ip+il%64,   low nibble, qh at 32*ip+(il%32), shift 2
        //   il 64..95:  ql at 64*ip+il%64,   HIGH nibble, qh at 32*ip+(il%32), shift 4
        //   il 96..127: ql at 64*ip+il%64,   HIGH nibble, qh at 32*ip+(il%32), shift 6
        {
            uint block_idx   = k_start / QK_K;
            uint elem_in_blk = k_start % QK_K;
            uint ip       = elem_in_blk / 128;
            uint il_start = elem_in_blk % 128;

            // Precompute uniform parameters for this tile
            bool use_ql_high = (il_start >= 64);
            uint qh_shift;
            if      (il_start < 32)  qh_shift = 0;
            else if (il_start < 64)  qh_shift = 2;
            else if (il_start < 96)  qh_shift = 4;
            else                     qh_shift = 6;

            // Phase 1a: First 32 threads load d and scale (one per column)
            if (flat_id < BN) {
                uint n = flat_id;
                uint global_n = col_block * BN + n;
                if (global_n < ne01) {
                    uint row_off = src0_offset + global_n * nb01
                                 + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q6K_BLOCK_SIZE;

                    // d at offset 208 (misaligned -- use safe read)
                    uint d_off = block_off + 208;
                    uint d_word = src0.Load(d_off & ~3u);
                    float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

                    // scale is signed byte at offset 192 + scale_idx
                    uint scale_idx = 8 * ip + il_start / 16;
                    uint sc_off = block_off + 192 + scale_idx;
                    uint sc_byte = read_byte_q6k(src0, sc_off);
                    int scale = (sc_byte < 128) ? (int)sc_byte : (int)sc_byte - 256;

                    col_ds[n] = d * (float)scale;
                } else {
                    col_ds[n] = 0.0f;
                }
            }

            // Phase 1b: All 256 threads cooperatively load ql and qh bytes
            // 32 cols × 16 bytes = 512 ql bytes + 512 qh bytes = 1024 total
            // 256 threads × 4 bytes each = 1024. Load 2 ql + 2 qh per thread.
            {
                // First pass: 256 threads load 512 ql bytes (2 per thread)
                // Thread flat_id loads ql bytes for col (flat_id/16), byte (flat_id%16)
                // But 256/16 = 16 cols per pass, need 2 passes for 32 cols
                uint pass_col = flat_id / 16;        // 0..15
                uint pass_byte = flat_id % 16;       // 0..15

                // Pass 1: columns 0..15
                {
                    uint col = pass_col;
                    uint global_n = col_block * BN + col;
                    if (global_n < ne01 && k_start < K) {
                        uint row_off = src0_offset + global_n * nb01
                                     + i2_src0 * nb02 + i3_src0 * nb03;
                        uint block_off = row_off + block_idx * Q6K_BLOCK_SIZE;
                        uint ql_addr = block_off + 64 * ip + (il_start % 64) + pass_byte;
                        uint qh_addr = block_off + 128 + 32 * ip + (il_start % 32) + pass_byte;
                        raw_ql[col][pass_byte] = read_byte_q6k(src0, ql_addr);
                        raw_qh[col][pass_byte] = read_byte_q6k(src0, qh_addr);
                    } else {
                        raw_ql[col][pass_byte] = 0;
                        raw_qh[col][pass_byte] = 0;
                    }
                }

                // Pass 2: columns 16..31
                {
                    uint col = pass_col + 16;
                    uint global_n = col_block * BN + col;
                    if (global_n < ne01 && k_start < K) {
                        uint row_off = src0_offset + global_n * nb01
                                     + i2_src0 * nb02 + i3_src0 * nb03;
                        uint block_off = row_off + block_idx * Q6K_BLOCK_SIZE;
                        uint ql_addr = block_off + 64 * ip + (il_start % 64) + pass_byte;
                        uint qh_addr = block_off + 128 + 32 * ip + (il_start % 32) + pass_byte;
                        raw_ql[col][pass_byte] = read_byte_q6k(src0, ql_addr);
                        raw_qh[col][pass_byte] = read_byte_q6k(src0, qh_addr);
                    } else {
                        raw_ql[col][pass_byte] = 0;
                        raw_qh[col][pass_byte] = 0;
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();

            // Phase 2: Dequant from shared memory
            {
                uint base = flat_id * 2;
                [unroll] for (uint e = 0; e < 2; e++) {
                    uint idx = base + e;
                    uint k_local = idx / BN;
                    uint n_local = idx % BN;

                    float val = 0.0f;
                    if ((k_start + k_local) < K && (col_block * BN + n_local) < ne01) {
                        uint ql_val = raw_ql[n_local][k_local];
                        uint qh_val = raw_qh[n_local][k_local];

                        uint ql_nibble = use_ql_high ? (ql_val >> 4) : (ql_val & 0x0Fu);
                        uint qh_bits = (qh_val >> qh_shift) & 3u;
                        int q = (int)(ql_nibble | (qh_bits << 4)) - 32;

                        val = col_ds[n_local] * (float)q;
                    }
                    tile_b[k_local][n_local] = val;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < BK; k++) {
            float a0 = tile_a[ty * 2    ][k];
            float a1 = tile_a[ty * 2 + 1][k];
            float b0 = tile_b[k][tx * 2    ];
            float b1 = tile_b[k][tx * 2 + 1];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
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
