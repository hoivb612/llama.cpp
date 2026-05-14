// mul_mat_q4k_wmma_lds.hlsl - Tiled MUL_MAT with cooperative Q4_K dequant.
//
// Same I/O as mul_mat_q4k_wmma.hlsl but pre-decodes Q4_K scale/min metadata
// into groupshared memory once per K-tile per N-column, instead of having
// every thread re-decode from global memory per dequanted element.
//
// Per K-tile (BK=32) within a single Q4_K block (256 elems → 8 tiles):
//   tile_idx_in_block = (k_start % 256) / 32   ∈ [0..7]
//   il               = tile_idx_in_block / 2
//   is_high          = (tile_idx_in_block & 1)
// Each (n_local, kt) pair has exactly one (sc_idx, mb_val) governing all 32
// dequanted values. Original shader recomputed it 32× per tile per column.
//
// Cooperative loader (256 threads, BN=32 columns):
//   Phase A (32 threads): decode (d_eff = dall*sc, m_eff = dmin*mb) per column.
//   Phase B (256 threads, 1 dword each): load 32 qs bytes per column into LDS.
//   Phase C (256 threads, 4 elems each): write tile_b[k_local][n_local].
//
// Q4_K block = 144 bytes: [d:f16][dmin:f16][scales:12B][qs:128B]
// All offsets used here are 4-byte aligned (block_off mod 4 == 0, 16+il*32+dw*4
// preserves alignment), so ByteAddressBuffer.Load is safe.

#include "ggml_common.hlsli"

#define QK_K           256
#define Q4K_BLOCK_SIZE 144

#define BM 32
#define BN 32
#define BK 32

groupshared float tile_a[BM][BK];
groupshared float tile_b[BK][BN];

groupshared float d_eff_lds[BN];        // per-column dall * sc_idx
groupshared float m_eff_lds[BN];        // per-column dmin * mb_val
groupshared uint  qs_dw_lds[BN][BK / 4]; // 32 cols × 8 dwords (= 32 qs bytes per col)

uint read_byte_q4k_lds(ByteAddressBuffer buf, uint byte_off) {
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
    uint batch     = gid.z;

    uint i2 = batch % ne2;
    uint i3 = batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_k_tiles = (K + BK - 1) / BK;

    precise float acc00 = 0.0f, acc01 = 0.0f;
    precise float acc10 = 0.0f, acc11 = 0.0f;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // Load tile_a (activations) - same pattern as base shader.
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
                tile_a[m_local][k_local] = val;
            }
        }

        // For Q4_K, all BK=32 K-elements lie within one block and share scale.
        uint block_idx        = k_start / QK_K;
        uint tile_idx_in_blk  = (k_start % QK_K) / BK;     // 0..7
        uint il               = tile_idx_in_blk / 2u;       // 0..3
        bool is_high          = (tile_idx_in_blk & 1u) != 0u;
        uint is_eff           = tile_idx_in_blk;            // == 2*il + is_high
        bool is_lt4           = (il < 2u);                  // matches original (is < 4)

        // --- Phase A: decode (d_eff, m_eff) once per N column ---
        if (flat_id < BN) {
            uint n_local  = flat_id;
            uint global_n = col_block * BN + n_local;
            float d_eff = 0.0f;
            float m_eff = 0.0f;
            if (global_n < ne01) {
                uint row_off   = src0_offset + global_n * nb01
                               + i2_src0 * nb02 + i3_src0 * nb03;
                uint block_off = row_off + block_idx * Q4K_BLOCK_SIZE;

                uint dm_raw = src0.Load(block_off);
                float dall    = f16_to_f32(dm_raw & 0xFFFFu);
                float dmin_v  = f16_to_f32(dm_raw >> 16);

                uint scales_off = block_off + 4;

                uint scidx0 = is_lt4 ? is_eff : (is_eff + 4u);
                uint scidx1 = is_lt4 ? is_eff : (is_eff - 4u);
                uint scmask1 = is_lt4 ? 0x30u : 0xC0u;
                uint scshift1 = is_lt4 ? 0u : 2u;
                uint mbidx0 = is_eff + 4u;
                uint mbidx1 = is_lt4 ? (is_eff + 4u) : is_eff;
                uint mbmask0 = is_lt4 ? 0x0Fu : 0xF0u;
                uint mbshift0 = is_lt4 ? 0u : 4u;
                uint mbmask1 = is_lt4 ? 0x30u : 0xC0u;
                uint mbshift1 = is_lt4 ? 0u : 2u;

                uint sc_idx = (read_byte_q4k_lds(src0, scales_off + scidx0) & 0x0Fu)
                            | ((read_byte_q4k_lds(src0, scales_off + scidx1) & scmask1) >> scshift1);
                uint mb_val = ((read_byte_q4k_lds(src0, scales_off + mbidx0) & mbmask0) >> mbshift0)
                            | ((read_byte_q4k_lds(src0, scales_off + mbidx1) & mbmask1) >> mbshift1);

                d_eff = dall   * float(sc_idx);
                m_eff = dmin_v * float(mb_val);
            }
            d_eff_lds[n_local] = d_eff;
            m_eff_lds[n_local] = m_eff;
        }

        // --- Phase B: cooperatively load qs bytes (32 cols × 8 dwords = 256) ---
        {
            uint n_local   = flat_id / 8u;
            uint dw_in_col = flat_id % 8u;
            uint global_n  = col_block * BN + n_local;
            uint qs_dw = 0u;
            if (global_n < ne01) {
                uint row_off   = src0_offset + global_n * nb01
                               + i2_src0 * nb02 + i3_src0 * nb03;
                uint block_off = row_off + block_idx * Q4K_BLOCK_SIZE;
                uint qs_off    = block_off + 16u + il * 32u + dw_in_col * 4u;
                qs_dw = src0.Load(qs_off);
            }
            qs_dw_lds[n_local][dw_in_col] = qs_dw;
        }

        GroupMemoryBarrierWithGroupSync();

        // --- Phase C: dequant tile_b[k][n] from LDS metadata + qs ---
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
                    uint qs_dw = qs_dw_lds[n_local][k_local / 4u];
                    uint qs_byte = (qs_dw >> ((k_local & 3u) * 8u)) & 0xFFu;
                    uint q = is_high ? (qs_byte >> 4) : (qs_byte & 0x0Fu);
                    val = d_eff_lds[n_local] * float(q) - m_eff_lds[n_local];
                }
                tile_b[k_local][n_local] = val;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Inner k loop: 2x2 register-blocked accumulation (unchanged).
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
