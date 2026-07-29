// arange.hlsl - dst[i] = start + step * i
// op_params: [0]=start (f32), [1]=stop (f32), [2]=step (f32)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= ne0) return;

    float start = op_param_f32(0);
    float step  = op_param_f32(2);
    float value = start + step * (float)idx;

    uint off_d = dst_offset + idx * nb0;
    store_auto(dst, off_d, value, dst_esize);
}
