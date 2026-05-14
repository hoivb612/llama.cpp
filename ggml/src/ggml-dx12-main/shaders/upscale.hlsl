// upscale.hlsl - Upscale/interpolate: nearest-neighbor
// Scale factors derived from output/input dimension ratios
// op_params[0] = mode_flags (mode & 0xFF: 0=nearest)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    // Scale factors: output_dim / input_dim
    float sf0 = (float)ne0 / (float)ne00;
    float sf1 = (float)ne1 / (float)ne01;

    // Nearest-neighbor: map output coord to input coord
    uint si0 = min((uint)((float)i0 / sf0), ne00 - 1);
    uint si1 = min((uint)((float)i1 / sf1), ne01 - 1);

    uint off0 = offset_4d(si0, si1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, load_auto(src0, off0, src0_esize), dst_esize);
}
