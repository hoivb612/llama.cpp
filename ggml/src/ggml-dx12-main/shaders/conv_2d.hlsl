// conv_2d.hlsl - 2D convolution
// src0: kernel [KW, KH, IC, OC]
// src1: input  [IW, IH, IC, batch]
// dst:  output [OW, OH, OC, batch]
// op_params: [0]=stride_x [1]=stride_y [2]=pad_x [3]=pad_y [4]=dilation_x [5]=dilation_y
#include "ggml_common.hlsli"

// Every thread in a group re-reads the same kernel weights, so stage them once
// when the group maps to a single output channel. Capped at 8 KB: the array is
// allocated for every dispatch of this shader, so a larger cap would cost
// occupancy on the shapes that fall back to loading weights directly.
#define CONV2D_LDS_W 2048
groupshared float s_w[CONV2D_LDS_W];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;

    int stride_x   = asint(op0);
    int stride_y   = asint(op1);
    int pad_x      = asint(op2);
    int pad_y      = asint(op3);
    int dilation_x = asint(op4);
    int dilation_y = asint(op5);

    // Kernel dimensions from src0
    uint KW = ne00;  // kernel width
    uint KH = ne01;  // kernel height
    uint IC = ne02;  // input channels

    // Input dimensions from src1
    uint IW = ne10;  // input width
    uint IH = ne11;  // input height

    uint wcount = KW * KH * IC;
    uint plane  = ne0 * ne1;

    // A group spans 256 consecutive dst indices; it is output-channel uniform
    // when its first and last index land on the same (i2, i3) plane.
    uint g_first = gid.x * 256u;
    uint g_last  = min(g_first + 255u, total - 1u);
    // Only worth a barrier when the tile is large enough to amortise it: tiny
    // kernels (2x2x1) reload so few weights that staging them costs more.
    bool use_lds = total > 0u && g_first < total &&
                   (g_first / plane) == (g_last / plane) &&
                   wcount >= 32u && wcount <= CONV2D_LDS_W;

    if (use_lds) {
        uint oc = (g_first / plane) % ne2;
        for (uint w = gtid.x; w < wcount; w += 256u) {
            uint wkx = w % KW;
            uint wr  = w / KW;
            uint wky = wr % KH;
            uint wic = wr / KH;
            s_w[w] = load_auto(src0,
                src0_offset + wkx * nb00 + wky * nb01 + wic * nb02 + oc * nb03,
                src0_esize);
        }
        // use_lds is group-uniform, so this barrier is in uniform control flow.
        GroupMemoryBarrierWithGroupSync();
    }

    if (idx >= total) return;

    // dst indices: i0=OW, i1=OH, i2=OC, i3=batch
    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    precise float acc = 0.0f;

    // Channel loop stays outermost: the kernel window then walks within one
    // input plane, which keeps the input reads cache local. Reversing it makes
    // the innermost step jump a whole plane (measured -18% on 1536x1536).
    for (uint ic = 0; ic < IC; ic++) {
        for (uint ky = 0; ky < KH; ky++) {
            int sy = (int)i1 * stride_y + (int)ky * dilation_y - pad_y;
            if (sy < 0 || sy >= (int)IH) {
                continue;
            }
            for (uint kx = 0; kx < KW; kx++) {
                int sx = (int)i0 * stride_x + (int)kx * dilation_x - pad_x;
                if (sx < 0 || sx >= (int)IW) {
                    continue;
                }

                float w;
                if (use_lds) {
                    w = s_w[(ic * KH + ky) * KW + kx];
                } else {
                    uint k_off = src0_offset + kx * nb00 + ky * nb01 + ic * nb02 + i2 * nb03;
                    w = load_auto(src0, k_off, src0_esize);
                }

                // Input value: src1[sx, sy, ic, batch]
                uint i_off = src1_offset + (uint)sx * nb10 + (uint)sy * nb11 + ic * nb12 + i3 * nb13;
                float v = load_auto(src1, i_off, src1_esize);

                acc += w * v;
            }
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
