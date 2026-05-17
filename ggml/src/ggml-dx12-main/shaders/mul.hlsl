// mul.hlsl - Element-wise multiplication: dst = src0 * src1
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    bool paired = can_pair_f16();
    uint idx = tid.x * (paired ? 2u : 1u);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint off0 = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off1 = offset_4d(i0 % ne10, i1 % ne11, i2 % ne12, i3 % ne13,
                          nb10, nb11, nb12, nb13, src1_offset);
    float v0 = load_auto(src0, off0, src0_esize) * load_auto(src1, off1, src1_esize);

    if (paired) {
        float v1 = load_auto(src0, off0 + nb00, src0_esize) *
                   load_auto(src1, offset_4d((i0+1) % ne10, i1 % ne11, i2 % ne12, i3 % ne13,
                             nb10, nb11, nb12, nb13, src1_offset), src1_esize);
        uint off_d = dst_offset + i0 * 2u + i1 * nb1 + i2 * nb2 + i3 * nb3;
        store_f16_pair(dst, off_d, v0, v1);
    } else {
        uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, v0, dst_esize);
    }
}