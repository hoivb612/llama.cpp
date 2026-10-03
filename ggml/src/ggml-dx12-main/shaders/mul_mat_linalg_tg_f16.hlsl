// F16 x F32 MUL_MAT using a threadgroup-scope SM 6.10 matrix.
//
// The activation tile is converted to F16 in groupshared memory. F16 weights
// load directly from the tensor buffer, and the complete output tile stays in
// one threadgroup matrix until its aligned store.
#include "ggml_common.hlsli"
#include <dx/linalg.h>

using namespace dx::linalg;

#ifndef TG_BM
#define TG_BM 128
#endif
#ifndef TG_BN
#define TG_BN 256
#endif

#define BM TG_BM
#define BN TG_BN
#define BK 16
#ifndef THREADS
#define THREADS 256
#endif

typedef Matrix<ComponentType::F16, BM, BK, MatrixUse::A, MatrixScope::ThreadGroup> MatA;
typedef Matrix<ComponentType::F16, BK, BN, MatrixUse::B, MatrixScope::ThreadGroup> MatB;
typedef Matrix<ComponentType::F32, BM, BN, MatrixUse::Accumulator, MatrixScope::ThreadGroup> MatAcc;

groupshared float16_t tile_a[BM * BK];

[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint row0  = gid.x * BM;
    const uint col0  = gid.y * BN;
    const uint batch = gid.z;
    const uint i2 = batch % ne2;
    const uint i3 = batch / ne2;
    const uint i2_src0 = i2 * ne02 / ne2;
    const uint i3_src0 = i3 * ne03 / ne3;

    MatAcc acc = MatAcc::Splat(0.0f);

    for (uint k0 = 0; k0 < ne00; k0 += BK) {
        [unroll] for (uint e = 0; e < (BM * BK) / THREADS; ++e) {
            const uint idx = tid + e * THREADS;
            const uint r = idx / BK;
            const uint k = idx % BK;
            const uint off = offset_4d(k0 + k, row0 + r, i2, i3,
                                       nb10, nb11, nb12, nb13, src1_offset);
            tile_a[idx] = (float16_t)asfloat(src1.Load(off));
        }
        GroupMemoryBarrierWithGroupSync();

        const uint b_off = offset_4d(k0, col0, i2_src0, i3_src0,
                                     nb00, nb01, nb02, nb03, src0_offset);
        MatA a = MatA::Load(tile_a, 0, BK, MatrixLayout::RowMajor);
        MatB b = MatB::Load(src0, b_off, nb01, MatrixLayout::ColMajor, 32u);
        acc.MultiplyAccumulate(a, b);
    }

    const uint d_off = offset_4d(col0, row0, i2, i3,
                                 nb0, nb1, nb2, nb3, dst_offset);
    acc.Store(dst, d_off, nb1, MatrixLayout::RowMajor, 128u);
}
