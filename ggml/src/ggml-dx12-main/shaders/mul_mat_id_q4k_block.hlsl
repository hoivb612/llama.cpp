// Q4_K per-element MMID with block-level scale/min decode hoisted out of
// the per-element loop. The shared mul_mat_id_quant.hlsli per-element
// template recomputes 8 scale/min values + 5 ByteAddressBuffer.Load calls
// per K iteration; on Intel UHD wave=8 this produces wrong results for
// later thread groups (cumulative register pressure / scratch state).
// This variant decodes a Q4_K block's 8 scales + 8 mins ONCE per 256-
// element segment then runs a tight inner loop with a single qs byte
// read and an FMA per element.

#include "ggml_common.hlsli"

#define QK_K 256
#define Q4K_BSIZE 144

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint idx = flat_idx_2d(group_id, local_id);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint ids_off = op0 + i1 * op1 + i2 * op2;
    int expert_id = asint(src2.Load(ids_off));

    uint K = ne00;
    uint i3_src0 = i3 * ne03 / ne3;
    uint src0_row = src0_offset + i0 * nb01 + (uint)expert_id * nb02 + i3_src0 * nb03;
    uint i1_src1 = i1 % ne11;
    uint src1_row = src1_offset + i1_src1 * nb11 + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;
    uint num_blocks = K / QK_K;
    for (uint b_idx = 0; b_idx < num_blocks; b_idx++) {
        uint block_off = src0_row + b_idx * Q4K_BSIZE;

        uint dm_raw = src0.Load(block_off);
        float dall = f16_to_f32(dm_raw & 0xFFFFu);
        float dmin = f16_to_f32(dm_raw >> 16);

        uint scales_off = block_off + 4;
        uint s0 = src0.Load(scales_off);
        uint s4 = src0.Load(scales_off + 4);
        uint s8 = src0.Load(scales_off + 8);

        float sc[8], mb[8];
        sc[0] = (float)( s0        & 0x3Fu);
        sc[1] = (float)((s0 >>  8) & 0x3Fu);
        sc[2] = (float)((s0 >> 16) & 0x3Fu);
        sc[3] = (float)((s0 >> 24) & 0x3Fu);
        mb[0] = (float)( s4        & 0x3Fu);
        mb[1] = (float)((s4 >>  8) & 0x3Fu);
        mb[2] = (float)((s4 >> 16) & 0x3Fu);
        mb[3] = (float)((s4 >> 24) & 0x3Fu);
        sc[4] = (float)((( s8        & 0x0Fu)      ) | (((s0 >> 6) & 0x03u) << 4));
        sc[5] = (float)((((s8 >>  8) & 0x0Fu)      ) | (((s0 >> 14) & 0x03u) << 4));
        sc[6] = (float)((((s8 >> 16) & 0x0Fu)      ) | (((s0 >> 22) & 0x03u) << 4));
        sc[7] = (float)((((s8 >> 24) & 0x0Fu)      ) | (((s0 >> 30) & 0x03u) << 4));
        mb[4] = (float)((( s8        >> 4) & 0x0Fu) | (((s4 >> 6) & 0x03u) << 4));
        mb[5] = (float)((((s8 >>  8) >> 4) & 0x0Fu) | (((s4 >> 14) & 0x03u) << 4));
        mb[6] = (float)((((s8 >> 16) >> 4) & 0x0Fu) | (((s4 >> 22) & 0x03u) << 4));
        mb[7] = (float)((((s8 >> 24) >> 4) & 0x0Fu) | (((s4 >> 30) & 0x03u) << 4));

        uint qs_off = block_off + 16;
        uint k0 = b_idx * QK_K;
        for (uint il = 0; il < 4; il++) {
            uint il_qs_off = qs_off + il * 32;
            uint is_lo = 2 * il;
            uint is_hi = 2 * il + 1;
            float dsc_lo = dall * sc[is_lo];
            float dmb_lo = dmin * mb[is_lo];
            float dsc_hi = dall * sc[is_hi];
            float dmb_hi = dmin * mb[is_hi];
            // 32 qs bytes per il = 8 uint32 words. Load aligned, unpack
            // 4 bytes manually, then process 4 elements per word.
            for (uint w = 0; w < 8; w++) {
                uint qs_word = src0.Load(il_qs_off + w * 4);
                [unroll] for (uint sub = 0; sub < 4; sub++) {
                    uint qs_byte = (qs_word >> (sub * 8u)) & 0xFFu;
                    float w_lo = dsc_lo * (float)(qs_byte & 0x0Fu) - dmb_lo;
                    float w_hi = dsc_hi * (float)(qs_byte >> 4)    - dmb_hi;
                    uint elem_in_half = w * 4 + sub;
                    uint k_lo = k0 + il * 64 + elem_in_half;
                    uint k_hi = k_lo + 32;
                    float x_lo = load_auto(src1, src1_row + k_lo * nb10, src1_esize);
                    float x_hi = load_auto(src1, src1_row + k_hi * nb10, src1_esize);
                    acc += w_lo * x_lo + w_hi * x_hi;
                }
            }
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
