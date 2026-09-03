// Fused strided gate gather, sigmoid and attention multiply.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint gate_i0 = idx % op0;
    uint gate_rem = idx / op0;
    uint gate_i1 = gate_rem % op1;
    gate_rem /= op1;
    uint gate_i2 = gate_rem % op2;
    uint gate_i3 = gate_rem / op2;

    uint gate_off = offset_4d(gate_i0, gate_i1, gate_i2, gate_i3,
                              nb00, nb01, nb02, nb03, src0_offset);
    uint attn_off = offset_4d(i0, i1, i2, i3, nb10, nb11, nb12, nb13, src1_offset);
    uint dst_off  = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    precise float gate = load_auto(src0, gate_off, src0_esize);
    precise float sigmoid_gate = 1.0f / (1.0f + exp(-gate));
    precise float value = load_auto(src1, attn_off, src1_esize) * sigmoid_gate;
    store_auto(dst, dst_off, value, dst_esize);
}
