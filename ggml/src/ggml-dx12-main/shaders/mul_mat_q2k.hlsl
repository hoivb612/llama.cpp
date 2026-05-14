// mul_mat_q2k.hlsl - Generic MUL_MAT (any M) for Q2_K weights.
// 1 thread per output element (slow but correct). Used for batch (M>1)
// prompt processing; matvec path uses mul_mat_vec_q2k.hlsl.
//
// Q2_K block (84 bytes):
//   offset  0..15 : scales[16] (low 4 bits = scale, high 4 bits = min)
//   offset 16..79 : qs[64]     (2 bits per element)
//   offset 80..81 : d (fp16)
//   offset 82..83 : dmin (fp16)
//
// dequantize_row_q2_K element ordering (canonical):
//   is = 0
//   for n in [0,128) step 128:
//     for j in [0,4):
//       shift = 2*j
//       sc = scales[is++]; dl = d*(sc&0xF); ml = dmin*(sc>>4)
//       for l in [0,16): y[n + j*32 + l + 0]  = dl * ((q[l   ] >> shift) & 3) - ml
//       sc = scales[is++]; dl = d*(sc&0xF); ml = dmin*(sc>>4)
//       for l in [0,16): y[n + j*32 + l + 16] = dl * ((q[l+16] >> shift) & 3) - ml
//     q += 32
//
// Q2_K block size 84 is 4-byte aligned, so plain Load() at block_off+offset is safe.

#include "ggml_common.hlsli"

#define QK_K 256
#define Q2K_BSIZE 84

uint read_byte_q2k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint idx = flat_idx_2d(group_id, local_id);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_row = src1_offset + i1 * nb11 + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;

    for (uint blk = 0; blk < num_blocks; blk++) {
        uint block_off = src0_row + blk * Q2K_BSIZE;

        // Load d, dmin (offsets 80,82 -> one 4-byte aligned word)
        uint dm = src0.Load(block_off + 80);
        float d    = f16_to_f32(dm & 0xFFFFu);
        float dmin = f16_to_f32(dm >> 16);

        uint y_block = blk * QK_K;
        uint is = 0;
        uint q_base = 0;  // qs offset within the block's qs[] array (0,32 across n)

        [unroll] for (uint n_iter = 0; n_iter < 2; n_iter++) {
            [unroll] for (uint j = 0; j < 4; j++) {
                uint shift = j * 2;
                uint y_pos_base = n_iter * 128u + j * 32u;

                // Half 1: l in [0,16), q[l + 0]
                {
                    uint sc_byte = read_byte_q2k(src0, block_off + is);
                    float scale = d    * float(sc_byte & 0x0Fu);
                    float ml    = dmin * float(sc_byte >> 4);
                    is++;
                    [unroll] for (uint l = 0; l < 16; l++) {
                        uint qb = read_byte_q2k(src0, block_off + 16u + q_base + l);
                        int q = int((qb >> shift) & 3u);
                        float w = scale * float(q) - ml;
                        uint k = y_block + y_pos_base + l;
                        float x = load_auto(src1, src1_row + k * nb10, src1_esize);
                        acc += w * x;
                    }
                }
                // Half 2: l in [0,16), q[l + 16]
                {
                    uint sc_byte = read_byte_q2k(src0, block_off + is);
                    float scale = d    * float(sc_byte & 0x0Fu);
                    float ml    = dmin * float(sc_byte >> 4);
                    is++;
                    [unroll] for (uint l = 0; l < 16; l++) {
                        uint qb = read_byte_q2k(src0, block_off + 16u + q_base + 16u + l);
                        int q = int((qb >> shift) & 3u);
                        float w = scale * float(q) - ml;
                        uint k = y_block + y_pos_base + 16u + l;
                        float x = load_auto(src1, src1_row + k * nb10, src1_esize);
                        acc += w * x;
                    }
                }
            }
            q_base += 32u;
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
