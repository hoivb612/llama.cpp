// cpy_quant_f32.hlsli - CPY/DUP from a quantized src0 to F32/F16/BF16 dst.
// Wrappers define exactly one MMID_<TYPE> macro and include this file.
// Source rows must be contiguous along dim 0 (block layout); dims 1..3 use
// the tensor's byte strides so permuted sources work.
#pragma once
#include "ggml_common.hlsli"
#include "quant_dequant.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = flat_idx_2d_256(tid);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint j0, j1, j2, j3;
    flat_to_4d(idx, ne00, ne01, ne02, j0, j1, j2, j3);

    uint row_off = src0_offset + j1 * nb01 + j2 * nb02 + j3 * nb03;

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto_bf16(dst, off_d, mmid_dequant(src0, row_off, j0), dst_esize);
}
