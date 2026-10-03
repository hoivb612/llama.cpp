// timestep_embedding.hlsl - sinusoidal timestep embedding
// dst shape: (dim, n_timesteps)
//   for j in [0, half):       dst[i, j]       = cos(timestep[i] * exp(-log(max_period) * j / half))
//   for j in [half, 2*half):  dst[i, j]       = sin(timestep[i] * exp(-log(max_period) * (j-half) / half))
//   if dim is odd:            dst[i, 2*half]  = 0
// op_params: [0]=dim (i32), [1]=max_period (i32)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1;
    if (idx >= total) return;

    uint j = idx % ne0;
    uint i = idx / ne0;

    uint dim_u        = op_param_uint(0);
    uint max_period_u = op_param_uint(1);
    uint half_u       = dim_u / 2;
    float half_f      = (float)half_u;
    float log_max_p   = log((float)max_period_u);

    uint src0_off = offset_4d(i, 0, 0, 0, nb00, nb01, nb02, nb03, src0_offset);
    float ts = load_auto(src0, src0_off, src0_esize);

    float val;
    if (j < half_u) {
        float freq = exp(-log_max_p * (float)j / half_f);
        val = cos(ts * freq);
    } else if (j < 2u * half_u) {
        uint jh = j - half_u;
        float freq = exp(-log_max_p * (float)jh / half_f);
        val = sin(ts * freq);
    } else {
        val = 0.0f;
    }

    uint off_d = offset_4d(j, i, 0, 0, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, val, dst_esize);
}
