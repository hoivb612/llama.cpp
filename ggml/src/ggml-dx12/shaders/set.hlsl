// set.hlsl - SET operation: overlay src1 data into dst at specified offset/strides
// Elements within src1's mapped region: dst = src1
// Elements outside: dst = src0 (copy through)
//
// op_params layout (from ggml):
//   op0 = nb1 (overlay stride dim1, bytes)
//   op1 = nb2 (overlay stride dim2, bytes)
//   op2 = nb3 (overlay stride dim3, bytes)
//   op3 = offset (byte offset into dst where overlay begins)
//   op4 = inplace flag (unused in shader ΓÇö handled by scheduler)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne00 * ne01 * ne02 * ne03;
    if (idx >= total) return;

    // Overlay strides and offset from op_params (all in bytes)
    uint op_nb1    = op0;
    uint op_nb2    = op1;
    uint op_nb3    = op2;
    uint op_offset = op3;

    // Convert byte values to element counts (F32=4 bytes, I32=4 bytes)
    uint esize = dst_esize;
    uint stride1 = op_nb1 / esize;
    uint stride2 = op_nb2 / esize;
    uint stride3 = op_nb3 / esize;
    uint offset_elem = op_offset / esize;

    // Check if this flat index falls within the overlay region
    if (idx >= offset_elem && stride3 > 0) {
        uint src1_i = idx - offset_elem;
        uint i3 = src1_i / stride3;
        uint rem2 = src1_i - i3 * stride3;
        uint i2 = rem2 / stride2;
        uint rem1 = rem2 - i2 * stride2;
        uint i1 = rem1 / stride1;
        uint i0 = rem1 % stride1;

        if (i0 < ne10 && i1 < ne11 && i2 < ne12 && i3 < ne13) {
            // Inside overlay region ΓÇö read from src1
            uint off1 = src1_offset + i0 * nb10 + i1 * nb11 + i2 * nb12 + i3 * nb13;
            float val = load_auto(src1, off1, src1_esize);
            store_auto(dst, dst_offset + idx * esize, val, dst_esize);
            return;
        }
    }

    // Outside overlay ΓÇö copy from src0
    float val = load_auto(src0, src0_offset + idx * esize, src0_esize);
    store_auto(dst, dst_offset + idx * esize, val, dst_esize);
}
