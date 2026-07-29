// argsort_large.hlsl - multi-pass bitonic sort with global scratch.
//
// Handles ARGSORT and TOP_K for ncols > 1024 (single-WG bitonic in
// argsort.hlsl/top_k.hlsl is capped at BLOCK_SIZE=1024 because LDS limits
// the sort buffer to that size).  This shader stages every sort slot
// in the global RWByteAddressBuffer `temp` (root slot u1, bound by host
// to a per-device scratch buffer of nrows * ncols_padded * 8 bytes).
//
// One physical shader, four logical phases selected by op_params[3]:
//   kind=0 (INIT)             — load src0 values into scratch, OOB slots
//                               get col_idx >= ncols (sorts to tail).
//   kind=1 (SWAP)             — one bitonic compare-and-swap step with
//                               outer step k = op5, inner step j = op6.
//                               Each thread owns one (icol, icol|j) pair.
//   kind=2 (WRITEOUT_ARGSORT) — emit dst[col] = scratch[ASC ? col :
//                               ncols-1-col].col_idx for col in [0, ncols).
//   kind=3 (WRITEOUT_TOP_K)   — emit dst[col] = scratch[ncols-1-col].col_idx
//                               for col in [0, K) — the K largest indices,
//                               written in descending-value order.
//
// Host coordinates: init → log2(ncols_padded) * (log2(ncols_padded)+1)/2
// swap dispatches → final writeout, each separated by UAV barriers.
//
// Scratch layout (per row, contiguous, row stride = ncols_padded * 8 bytes):
//   slot i: uint2(col_idx, value_bits)
//
// op_params:
//   op0  = ggml_sort_order (ASC=0, DESC=1)     (ARGSORT only)
//   op1  = ncols                                (= ne00)
//   op2  = ncols_padded                         (power of 2, >= ncols)
//   op3  = kind                                 (0..3)
//   op4  = K                                    (TOP_K only)
//   op5  = k                                    (SWAP only, outer step)
//   op6  = j                                    (SWAP only, inner step)

#include "ggml_common.hlsli"

#define BLOCK_SIZE 256

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint i_row = gid.y;
    uint nrows = ne01 * ne02 * ne03;
    if (i_row >= nrows) return;

    uint ncols        = op1;
    uint ncols_padded = op2;
    uint kind         = op3;

    // Decompose flat row index into (i1, i2, i3) for src/dst stride math.
    uint i1  = i_row % ne01;
    uint i23 = i_row / ne01;
    uint i2  = i23  % ne02;
    uint i3  = i23  / ne02;

    uint row_off_scratch = i_row * ncols_padded * 8u; // bytes

    if (kind == 0u) {
        // INIT: each thread loads one src element (or sets OOB sentinel).
        uint slot = gid.x * BLOCK_SIZE + gtid.x;
        if (slot >= ncols_padded) return;

        uint val_bits = 0;
        uint col_idx  = slot;
        if (slot < ncols) {
            uint src_off = src0_offset + i3 * nb03 + i2 * nb02 + i1 * nb01 + slot * nb00;
            val_bits = src0.Load(src_off);
        } else {
            // OOB: keep col_idx = slot (>= ncols) as the OOB marker; value
            // is replaced with +inf at compare time so OOB sorts to the tail.
            val_bits = 0;
        }
        temp.Store2(row_off_scratch + slot * 8u, uint2(col_idx, val_bits));

    } else if (kind == 1u) {
        // SWAP: one compare-and-swap step. Each thread = one pair.
        uint k_step = op5;
        uint j_step = op6;

        uint pair_id = gid.x * BLOCK_SIZE + gtid.x;
        uint half_n  = ncols_padded >> 1u;
        if (pair_id >= half_n) return;

        // Map pair_id -> icol by inserting a zero bit at position log2(j).
        uint low_mask = j_step - 1u;
        uint icol = ((pair_id & ~low_mask) << 1u) | (pair_id & low_mask);
        uint ixj  = icol | j_step;

        uint off_icol = row_off_scratch + icol * 8u;
        uint off_ixj  = row_off_scratch + ixj  * 8u;

        uint2 sh_icol = temp.Load2(off_icol);
        uint2 sh_ixj  = temp.Load2(off_ixj);

        bool oob_icol = sh_icol.x >= ncols;
        bool oob_ixj  = sh_ixj.x  >= ncols;
        float v_icol  = oob_icol ? asfloat(0x7F800000u) : asfloat(sh_icol.y);
        float v_ixj   = oob_ixj  ? asfloat(0x7F800000u) : asfloat(sh_ixj.y);

        // Bitonic direction: (icol & k_step) == 0 => ascending subsequence.
        bool ascending = (icol & k_step) == 0u;
        bool do_swap   = ascending ? (v_icol > v_ixj) : (v_icol < v_ixj);

        if (do_swap) {
            temp.Store2(off_icol, sh_ixj);
            temp.Store2(off_ixj,  sh_icol);
        }

    } else if (kind == 2u) {
        // WRITEOUT_ARGSORT: dst[col] = scratch[(order==ASC) ? col : ncols-1-col].col_idx
        uint col = gid.x * BLOCK_SIZE + gtid.x;
        if (col >= ncols) return;

        uint order    = op0;
        uint src_slot = (order == 0u) ? col : (ncols - 1u - col);
        uint2 sh = temp.Load2(row_off_scratch + src_slot * 8u);

        uint dst_row_off = dst_offset + i3 * nb3 + i2 * nb2 + i1 * nb1;
        dst.Store(dst_row_off + col * nb0, sh.x);

    } else if (kind == 3u) {
        // WRITEOUT_TOP_K: dst[col] = scratch[ncols-1-col].col_idx for col in [0, K).
        uint K   = op4;
        uint col = gid.x * BLOCK_SIZE + gtid.x;
        if (col >= K) return;
        if (col >= ncols) return; // defensive: K > ncols would be degenerate

        uint src_slot = ncols - 1u - col;
        uint2 sh = temp.Load2(row_off_scratch + src_slot * 8u);

        uint dst_row_off = dst_offset + i3 * nb3 + i2 * nb2 + i1 * nb1;
        dst.Store(dst_row_off + col * nb0, sh.x);
    }
}
