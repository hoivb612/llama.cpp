// top_k_large.hlsl - partial-selection TOP_K for ncols > 1024.
//
// argsort_large.hlsl fully sorts the row with a global bitonic sweep, which
// costs log2(N)*(log2(N)+1)/2 dispatches (171 for a 151936-wide vocab row).
// TOP_K only needs the K largest, so this shader instead reduces the row in
// a few passes: every block sorts up to BLOCK_SIZE candidates in LDS and
// emits only its top KL.  A global top-K element is always inside its own
// block's top-KL as long as KL >= K, so the reduction is exact.
//
// Passes for ncols=151936, KL=64:  151936 -> 9536 -> 640 -> dst.
//
// One physical shader, three logical phases selected by op_params[3]:
//   kind=0 (FIRST)  - read f32 values from src0, emit KL pairs per block.
//   kind=1 (REDUCE) - read pairs from scratch region op6, emit KL per block.
//   kind=2 (FINAL)  - read pairs from scratch region op6 (single block),
//                     write the K largest source indices to dst.
//
// Scratch layout (bound at root slot u1, same buffer as argsort_large):
//   row r, region p:  r * (2 * cap * 8) + p * cap * 8 bytes
//   slot i: uint2(col_idx, value_bits)
//
// op_params:
//   op1 = ncols        (source row length)
//   op2 = KL           (per-block emit count, power of 2, >= K)
//   op3 = kind         (0..2)
//   op4 = K            (final output count)
//   op5 = n_in         (input element count for this pass)
//   op6 = in_region    (0/1, unused for kind=0)
//   op7 = out_region   (0/1, unused for kind=2)
//   op8 = cap          (region capacity in pairs)

#include "ggml_common.hlsli"

#define BLOCK_SIZE 1024
#define NEG_INF_BITS 0xFF800000u
#define IDX_NONE     0xFFFFFFFFu

groupshared uint2 buf[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint blk   = gid.x;
    uint i_row = gid.y;
    uint nrows = ne01 * ne02 * ne03;
    if (i_row >= nrows) return;

    uint ncols = op1;
    uint KL    = op2;
    uint kind  = op3;
    uint K     = op4;
    uint n_in  = op5;
    uint cap   = op8;

    uint i1  = i_row % ne01;
    uint i23 = i_row / ne01;
    uint i2  = i23  % ne02;
    uint i3  = i23  / ne02;

    uint row_base = i_row * cap * 2u * 8u;

    uint t    = gtid.x;
    uint slot = blk * BLOCK_SIZE + t;

    // Load this thread's candidate, or a -inf sentinel past the end.
    uint2 cand = uint2(IDX_NONE, NEG_INF_BITS);
    if (kind == 0u) {
        if (slot < ncols) {
            uint src_off = src0_offset + i3 * nb03 + i2 * nb02 + i1 * nb01 + slot * nb00;
            cand = uint2(slot, src0.Load(src_off));
        }
    } else {
        if (slot < n_in) {
            uint in_off = row_base + op6 * cap * 8u + slot * 8u;
            cand = temp.Load2(in_off);
        }
    }
    buf[t] = cand;
    GroupMemoryBarrierWithGroupSync();

    // Bitonic sort, descending. Sentinels carry -inf so they sink to the tail.
    [unroll] for (uint kk = 2u; kk <= BLOCK_SIZE; kk <<= 1u) {
        [unroll] for (uint j = kk >> 1u; j > 0u; j >>= 1u) {
            uint ixj = t ^ j;
            if (ixj > t) {
                uint2 a = buf[t];
                uint2 b = buf[ixj];
                float va = asfloat(a.y);
                float vb = asfloat(b.y);
                bool descending = (t & kk) == 0u;
                bool swap = descending ? (va < vb) : (va > vb);
                if (swap) {
                    buf[t]   = b;
                    buf[ixj] = a;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (kind == 2u) {
        if (t < K) {
            uint dst_row_off = dst_offset + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dst.Store(dst_row_off + t * nb0, buf[t].x);
        }
    } else {
        if (t < KL) {
            uint out_off = row_base + op7 * cap * 8u + (blk * KL + t) * 8u;
            temp.Store2(out_off, buf[t]);
        }
    }
}
