// One wave reduces a 4-token by 8-output tile without shared memory.
#include "ggml_common.hlsli"

#define BM 4
#define BN 8
#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

[WaveSize(WAVE_SIZE)]
[numthreads(WAVE_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint m_base = gid.y * BM;
    uint n_base = gid.x * BN;
    float acc[BM][BN];
    [unroll] for (uint m = 0; m < BM; ++m) {
        [unroll] for (uint n = 0; n < BN; ++n) {
            acc[m][n] = 0.0f;
        }
    }

    for (uint k = tid * 4; k < ne00; k += WAVE_SIZE * 4) {
        float4 a[BM];
        [unroll] for (uint m = 0; m < BM; ++m) {
            a[m] = m_base + m < ne1 ?
                asfloat(src1.Load4(src1_offset + (m_base + m) * nb11 + k * 4)) : 0.0f;
        }
        [unroll] for (uint n = 0; n < BN; ++n) {
            float4 b = asfloat(src0.Load4(src0_offset + (n_base + n) * nb01 + k * 4));
            [unroll] for (uint m = 0; m < BM; ++m) {
                acc[m][n] = mad(a[m].x, b.x, acc[m][n]);
                acc[m][n] = mad(a[m].y, b.y, acc[m][n]);
                acc[m][n] = mad(a[m].z, b.z, acc[m][n]);
                acc[m][n] = mad(a[m].w, b.w, acc[m][n]);
            }
        }
    }

    [unroll] for (uint m = 0; m < BM; ++m) {
        [unroll] for (uint n = 0; n < BN; ++n) {
            float sum = WaveActiveSum(acc[m][n]);
            if (tid == 0 && m_base + m < ne1) {
                store_f32(dst, dst_offset + (m_base + m) * nb1 + (n_base + n) * nb0, sum);
            }
        }
    }
}
