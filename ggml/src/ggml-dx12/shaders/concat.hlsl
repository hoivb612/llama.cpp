// concat.hlsl - Concatenate tensors along dimension
// op_param_uint(0) = concat dimension (0, 1, 2, or 3)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    int dim = asint(op_param_uint(0));

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    // Determine which source and the source index
    bool from_src0;
    uint s0 = i0, s1 = i1, s2 = i2, s3 = i3;

    if (dim == 0) {
        from_src0 = (i0 < ne00);
        if (!from_src0) s0 = i0 - ne00;
    } else if (dim == 1) {
        from_src0 = (i1 < ne01);
        if (!from_src0) s1 = i1 - ne01;
    } else if (dim == 2) {
        from_src0 = (i2 < ne02);
        if (!from_src0) s2 = i2 - ne02;
    } else {
        from_src0 = (i3 < ne03);
        if (!from_src0) s3 = i3 - ne03;
    }

    float val;
    if (from_src0) {
        uint off = offset_4d(s0, s1, s2, s3, nb00, nb01, nb02, nb03, src0_offset);
        val = load_auto(src0, off, src0_esize);
    } else {
        uint off = offset_4d(s0, s1, s2, s3, nb10, nb11, nb12, nb13, src1_offset);
        val = load_auto(src1, off, src1_esize);
    }

    store_auto(dst, off_d, val, dst_esize);
}
