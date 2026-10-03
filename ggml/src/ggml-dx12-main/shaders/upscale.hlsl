// upscale.hlsl - Interpolate to a target shape.
// Supports: NEAREST, BILINEAR (+ ALIGN_CORNERS, + ANTIALIAS), BICUBIC (+ ALIGN_CORNERS).
// Scales dim0/dim1 (with optional align-corners coord mapping) and dim2/dim3
// via integer-floor (matches CPU reference in ggml-cpu/ops.cpp:
// ggml_compute_forward_upscale_f32).
//
// op_params[0] = mode_flags (low 8 bits = ggml_scale_mode kind, high bits = ggml_scale_flag mask)
//
// IMPLEMENTATION NOTE:
// Earlier iterations computed sf2/sf3 + i02/i03 once at function scope (mirroring CPU's
// outer-loop hoisting).  On DX12 this produced wrong results for the [512,512,3,2]
// scale=2 case even when sf2/sf3 should be exactly 1.0.  The compiled DXIL appeared to
// alias the i02/i03 registers and feed garbage into the offset calculation.  Computing
// everything inside the mode branches (and using pure-integer nearest mapping
// i02 = i2 * ne02 / ne2 where possible) avoids the issue and matches CPU exactly.
#include "ggml_common.hlsli"

#define GG_SCALE_MODE_NEAREST  0u
#define GG_SCALE_MODE_BILINEAR 1u
#define GG_SCALE_MODE_BICUBIC  2u
#define GG_SCALE_FLAG_ALIGN_CORNERS (1u << 8)
#define GG_SCALE_FLAG_ANTIALIAS     (1u << 9)

float load_src(uint i0, uint i1, uint i2, uint i3) {
    uint off = src0_offset + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    return load_auto(src0, off, src0_esize);
}

float bc_weight1(float x) {
    const float a = -0.75f;
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
}
float bc_weight2(float x) {
    const float a = -0.75f;
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}
float bicubic_row(float p0, float p1, float p2, float p3, float t) {
    return p0 * bc_weight2(t + 1.0f)
         + p1 * bc_weight1(t)
         + p2 * bc_weight1(1.0f - t)
         + p3 * bc_weight2(2.0f - t);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    const uint mode_flags    = op0;
    const uint mode_kind     = mode_flags & 0xFFu;
    const bool align_corners = (mode_flags & GG_SCALE_FLAG_ALIGN_CORNERS) != 0u;
    const bool antialias     = (mode_flags & GG_SCALE_FLAG_ANTIALIAS) != 0u;

    // dim2/dim3 are always nearest-neighbour; use exact integer mapping so we
    // never read out of bounds due to float round-trip on sf=1.0.
    uint i02 = i2 * ne02 / ne2;
    uint i03 = i3 * ne03 / ne3;

    float val = 0.0f;

    if (mode_kind == GG_SCALE_MODE_NEAREST) {
        uint i00 = i0 * ne00 / ne0;
        uint i01 = i1 * ne01 / ne1;
        val = load_src(i00, i01, i02, i03);
    } else {
        // Bilinear / bicubic share the dim0/dim1 scale-factor + pixel_offset setup.
        float sf0 = (float)ne0 / (float)ne00;
        float sf1 = (float)ne1 / (float)ne01;
        float pixel_offset = 0.5f;
        if (align_corners) {
            pixel_offset = 0.0f;
            if (ne0 > 1u && ne00 > 1u) sf0 = (float)(ne0 - 1u) / (float)(ne00 - 1u);
            if (ne1 > 1u && ne01 > 1u) sf1 = (float)(ne1 - 1u) / (float)(ne01 - 1u);
        }

        if (mode_kind == GG_SCALE_MODE_BILINEAR && antialias) {
            // PIL/PyTorch bilinear with triangle anti-alias filter.
            const float support0  = max(1.0f, 1.0f / sf0);
            const float support1  = max(1.0f, 1.0f / sf1);
            const float invscale0 = 1.0f / support0;
            const float invscale1 = 1.0f / support1;

            const float x = ((float)i0 + pixel_offset) / sf0;
            const float y = ((float)i1 + pixel_offset) / sf1;

            int x_min = max(0, (int)floor(x - support0 + pixel_offset));
            int x_max = min((int)ne00, (int)floor(x + support0 + pixel_offset));
            int y_min = max(0, (int)floor(y - support1 + pixel_offset));
            int y_max = min((int)ne01, (int)floor(y + support1 + pixel_offset));

            float acc = 0.0f;
            float wsum = 0.0f;
            for (int sy = y_min; sy < y_max; sy++) {
                float wy = max(1.0f - abs(((float)sy - y + pixel_offset) * invscale1), 0.0f);
                for (int sx = x_min; sx < x_max; sx++) {
                    float wx = max(1.0f - abs(((float)sx - x + pixel_offset) * invscale0), 0.0f);
                    float w  = wx * wy;
                    if (w <= 0.0f) continue;
                    acc  += w * load_src((uint)sx, (uint)sy, i02, i03);
                    wsum += w;
                }
            }
            val = (wsum > 0.0f) ? (acc / wsum) : 0.0f;
        } else if (mode_kind == GG_SCALE_MODE_BILINEAR) {
            const float xf = ((float)i0 + pixel_offset) / sf0 - pixel_offset;
            const float yf = ((float)i1 + pixel_offset) / sf1 - pixel_offset;
            int x0 = (int)floor(xf);
            int y0 = (int)floor(yf);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            float dx = clamp(xf - (float)x0, 0.0f, 1.0f);
            float dy = clamp(yf - (float)y0, 0.0f, 1.0f);
            x0 = clamp(x0, 0, (int)ne00 - 1);
            x1 = clamp(x1, 0, (int)ne00 - 1);
            y0 = clamp(y0, 0, (int)ne01 - 1);
            y1 = clamp(y1, 0, (int)ne01 - 1);
            float a = load_src((uint)x0, (uint)y0, i02, i03);
            float b = load_src((uint)x1, (uint)y0, i02, i03);
            float c = load_src((uint)x0, (uint)y1, i02, i03);
            float d = load_src((uint)x1, (uint)y1, i02, i03);
            val = a * (1.0f - dx) * (1.0f - dy)
                + b *         dx  * (1.0f - dy)
                + c * (1.0f - dx) *         dy
                + d *         dx  *         dy;
        } else if (mode_kind == GG_SCALE_MODE_BICUBIC) {
            const float xf = ((float)i0 + pixel_offset) / sf0 - pixel_offset;
            const float yf = ((float)i1 + pixel_offset) / sf1 - pixel_offset;
            int x0 = (int)floor(xf);
            int y0 = (int)floor(yf);
            float dx = xf - (float)x0;
            float dy = yf - (float)y0;

            float rows[4];
            [unroll] for (int j = 0; j < 4; j++) {
                int yj = clamp(y0 + (j - 1), 0, (int)ne01 - 1);
                float p[4];
                [unroll] for (int k = 0; k < 4; k++) {
                    int xk = clamp(x0 + (k - 1), 0, (int)ne00 - 1);
                    p[k] = load_src((uint)xk, (uint)yj, i02, i03);
                }
                rows[j] = bicubic_row(p[0], p[1], p[2], p[3], dx);
            }
            val = bicubic_row(rows[0], rows[1], rows[2], rows[3], dy);
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, val, dst_esize);
}
