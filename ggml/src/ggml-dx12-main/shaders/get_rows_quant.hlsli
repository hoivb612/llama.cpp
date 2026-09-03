// get_rows_quant.hlsli - generic gather/dequantize driven by quant_dequant.hlsli.
// Wrappers define exactly one MMID_<TYPE> macro and include this file.
#pragma once
#include "ggml_common.hlsli"
#include "quant_dequant.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint row_idx_off = src1_offset + i1 * nb10 + i2 * nb11 + i3 * nb12;
    int  row_idx     = asint(src1.Load(row_idx_off));

    uint row_off = src0_offset + (uint)row_idx * nb01 + i2 * nb02 + i3 * nb03;

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, mmid_dequant(src0, row_off, i0), dst_esize);
}
