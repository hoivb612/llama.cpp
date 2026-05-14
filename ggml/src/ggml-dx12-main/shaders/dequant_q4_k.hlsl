// dequant_q4_k.hlsl - Dequantize Q4_K format to F32
// Q4_K: 256 elements per super-block, 144 bytes each
// Layout: d(f16) + dmin(f16) + scales[12] + qs[128]
//
// Each thread handles 8 elements (4 + 4 from two sub-blocks)
// 32 threads per group = 256 elements = 1 super-block
#include "ggml_common.hlsli"

// Q4_K block layout offsets
#define QK_K 256
#define K_SCALE_SIZE 12
#define Q4_K_BLOCK_SIZE 144  // 2+2+12+128

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    // Process multiple blocks per work group for throughput
    for (uint wgy = 0; wgy < 256; wgy++) {
        uint ib = gid.x * 256 + wgy; // block index
        uint total_blocks = (ne00 * ne01 * ne02 * ne03 + QK_K - 1) / QK_K;
        if (ib >= total_blocks) return;

        uint local_id = gtid.x;
        uint il = local_id / 8;   // 0..3 (which 64-element chunk)
        uint ir = local_id % 8;   // 0..7 (which 4-element group within chunk)
        uint is = 2 * il;         // scale index base
        uint n = 4;

        // Block base offset in src0
        uint block_off = src0_offset + ib * Q4_K_BLOCK_SIZE;

        // Load d and dmin (f16 values)
        uint dm_raw = src0.Load(block_off);
        float dall = f16_to_f32(dm_raw & 0xFFFF);
        float dmin = f16_to_f32(dm_raw >> 16);

        // scales start at offset 4
        uint scales_off = block_off + 4;
        // qs start at offset 4 + 12 = 16
        uint qs_off = block_off + 16;

        // Output index
        uint y_idx = ib * QK_K + 64 * il + n * ir;
        uint qs_idx = 32 * il + n * ir;

        // Load scale bytes (we need indices computed from is)
        // First sub-block scale/min extraction
        uint scidx0 = (is < 4) ? is : (is + 4);
        uint scidx1 = (is < 4) ? is : (is - 4);
        uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
        uint scidxshift1 = (is < 4) ? 0 : 2;
        uint mbidx0 = is + 4;
        uint mbidx1 = (is < 4) ? is + 4 : is;
        uint mbidxmask0 = (is < 4) ? 0x0F : 0xF0;
        uint mbidxshift0 = (is < 4) ? 0 : 4;
        uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
        uint mbidxshift1 = (is < 4) ? 0 : 2;

        // Load scale bytes
        uint sc_byte0 = (src0.Load(scales_off + (scidx0 & ~3)) >> ((scidx0 & 3) * 8)) & 0xFF;
        uint sc_byte1 = (src0.Load(scales_off + (scidx1 & ~3)) >> ((scidx1 & 3) * 8)) & 0xFF;
        uint mb_byte0 = (src0.Load(scales_off + (mbidx0 & ~3)) >> ((mbidx0 & 3) * 8)) & 0xFF;
        uint mb_byte1 = (src0.Load(scales_off + (mbidx1 & ~3)) >> ((mbidx1 & 3) * 8)) & 0xFF;

        uint sc = (sc_byte0 & 0x0F) | ((sc_byte1 & scidxmask1) >> scidxshift1);
        uint mbyte = ((mb_byte0 & mbidxmask0) >> mbidxshift0) | ((mb_byte1 & mbidxmask1) >> mbidxshift1);

        float d1 = dall * (float)sc;
        float m1 = dmin * (float)mbyte;

        // Second sub-block
        scidx0 = (is < 4) ? is + 1 : (is + 5);
        scidx1 = (is < 4) ? is + 1 : (is - 3);
        mbidx0 = is + 5;
        mbidx1 = (is < 4) ? is + 5 : is + 1;

        sc_byte0 = (src0.Load(scales_off + (scidx0 & ~3)) >> ((scidx0 & 3) * 8)) & 0xFF;
        sc_byte1 = (src0.Load(scales_off + (scidx1 & ~3)) >> ((scidx1 & 3) * 8)) & 0xFF;
        mb_byte0 = (src0.Load(scales_off + (mbidx0 & ~3)) >> ((mbidx0 & 3) * 8)) & 0xFF;
        mb_byte1 = (src0.Load(scales_off + (mbidx1 & ~3)) >> ((mbidx1 & 3) * 8)) & 0xFF;

        sc = (sc_byte0 & 0x0F) | ((sc_byte1 & scidxmask1) >> scidxshift1);
        mbyte = ((mb_byte0 & mbidxmask0) >> mbidxshift0) | ((mb_byte1 & mbidxmask1) >> mbidxshift1);

        float d2 = dall * (float)sc;
        float m2 = dmin * (float)mbyte;

        // Dequantize 4 + 4 elements
        for (uint l = 0; l < n; l++) {
            uint qs_byte_off = qs_off + ((qs_idx + l) & ~3);
            uint qs_byte_shift = ((qs_idx + l) & 3) * 8;
            uint qs_val = (src0.Load(qs_byte_off) >> qs_byte_shift) & 0xFF;

            float val_lo = d1 * (float)(qs_val & 0x0F) - m1;
            float val_hi = d2 * (float)(qs_val >> 4) - m2;

            store_auto(dst, dst_offset + (y_idx + l) * 4, val_lo, dst_esize);
            store_auto(dst, dst_offset + (y_idx + l + 32) * 4, val_hi, dst_esize);
        }
    }
}
