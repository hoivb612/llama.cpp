// col2im_1d.hlsl - Scatter-add columns [K*OC, T_in] -> signal [T_out, OC].
// Gather form: each output element sums the inputs that map onto it.
// op0 = s0 (stride), op1 = OC, op2 = p0 (padding)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint flat_group = gid.x + gid.y * 65535u;
    uint idx = flat_group * 256u + gtid.x;
    uint total = ne0 * ne1;
    if (idx >= total) return;

    int T_out = (int)ne0;
    int t_out = (int)(idx % ne0);
    int oc    = (int)(idx / ne0);

    int s0   = asint(op0);
    int OC   = asint(op1);
    int p0   = asint(op2);
    int K_OC = (int)ne00;
    int T_in = (int)ne01;
    int K    = K_OC / OC;

    int t_abs = t_out + p0;

    int t_in_min = (t_abs - K + 1 + s0 - 1) / s0;
    if (t_in_min < 0) t_in_min = 0;
    int t_in_max = t_abs / s0;
    if (t_in_max >= T_in) t_in_max = T_in - 1;

    // esize 3 is the BF16 sentinel; its physical stride is 2 bytes
    uint s_stride = (src0_esize == 3) ? 2u : src0_esize;
    uint d_stride = (dst_esize == 3) ? 2u : dst_esize;

    float sum = 0.0f;
    for (int t_in = t_in_min; t_in <= t_in_max; t_in++) {
        int k = t_abs - t_in * s0;
        if (k >= 0 && k < K) {
            uint off = src0_offset + (uint)((oc * K + k) + t_in * K_OC) * s_stride;
            sum += load_auto(src0, off, src0_esize);
        }
    }

    uint off_d = dst_offset + (uint)(t_out + oc * T_out) * d_stride;
    store_auto(dst, off_d, sum, dst_esize);
}
