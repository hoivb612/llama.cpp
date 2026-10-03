#include "ggml_common.hlsli"

// op0..op1: post strides; op2..op4: comb strides; op5: has comb.
// src2/src3 tensor offsets are part of the bound SRV addresses.
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint idx = flat_idx_2d_256(tid);
    if (idx >= ne0 * ne1 * ne2) return;

    const uint i0 = idx % ne0;
    const uint h = (idx / ne0) % ne1;
    const uint token = idx / (ne0 * ne1);
    const float x = load_f32(src0, src0_offset + i0 * nb00 + token * nb01);
    const float post = load_f32(src2, h * op0 + token * op1);
    const uint residual = src1_offset + i0 * nb10 + token * nb12;
    float sum = x * post;
    if (op5 != 0u) {
        for (uint s = 0; s < ne1; ++s) {
            sum += load_f32(src1, residual + s * nb11) * load_f32(src3, h * op2 + s * op3 + token * op4);
        }
    } else {
        sum += load_f32(src1, residual + h * nb11);
    }
    store_f32(dst, dst_offset + i0 * nb0 + h * nb1 + token * nb2, sum);
}
