// xielu.hlsl - XIELU parametric unary
// CPU reference (unary-ops.cpp:op_xielu):
//   if (x > 0)  -> alpha_p * x*x + beta * x
//   else        -> (expm1(min(x, eps)) - x) * alpha_n + beta * x
// op_params layout (matches ggml_compute_forward_xielu):
//   op_params[1]: alpha_n
//   op_params[2]: alpha_p
//   op_params[3]: beta
//   op_params[4]: eps
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint off0  = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

    float alpha_n = op_param_f32(1);
    float alpha_p = op_param_f32(2);
    float beta    = op_param_f32(3);
    float eps     = op_param_f32(4);

    float x = load_auto(src0, off0, src0_esize);
    float r;
    if (x > 0.0f) {
        r = alpha_p * x * x + beta * x;
    } else {
        float min_x_eps = min(x, eps);
        r = (exp(min_x_eps) - 1.0f - x) * alpha_n + beta * x;
    }
    store_auto(dst, off_d, r, dst_esize);
}
