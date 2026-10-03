// out_prod_linalg.hlsl - OUT_PROD via SM 6.10 wave matrices.
//
// ggml stores dst as [M, N]. Compute its transpose as:
//   B[N, K] x A^T[K, M] = dst^T[N, M]
// so both the output stores and B's logical rows are contiguous.

#include "ggml_common.hlsli"
#include <dx/linalg.h>

using namespace dx::linalg;

#ifndef WAVE_SIZE
#error "out_prod_linalg requires -D WAVE_SIZE"
#endif

#define TILE    16
#define BM      32
#define BN      32
#define BK      32
#ifdef LA_INTEL_WAVE
#if WAVE_SIZE != 16
#error "LA_INTEL_WAVE requires WAVE_SIZE=16"
#endif
#define NWAVE   (BM / 8)
#define ACC_E   (8 * TILE / WAVE_SIZE)
#else
#define NWAVE   2
#endif
#define THREADS (NWAVE * WAVE_SIZE)

groupshared float16_t tile_a[BM * BK];
groupshared float16_t tile_b[BK * BN];
#ifdef LA_INTEL_WAVE
typedef Matrix<ComponentType::F16, 8, TILE, MatrixUse::A, MatrixScope::Wave>              MatA;
typedef Matrix<ComponentType::F16, TILE, TILE, MatrixUse::B, MatrixScope::Wave>           MatB;
typedef Matrix<ComponentType::F16, 8, TILE, MatrixUse::Accumulator, MatrixScope::Wave>    MatAcc;
#else
groupshared float     tile_c[NWAVE * TILE * TILE];

typedef Matrix<ComponentType::F16, TILE, TILE, MatrixUse::A, MatrixScope::Wave>           MatA;
typedef Matrix<ComponentType::F16, TILE, TILE, MatrixUse::B, MatrixScope::Wave>           MatB;
typedef Matrix<ComponentType::F32, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;
#endif

WAVE_SIZE_ATTR
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint rows = ne1;
    const uint cols = ne0;
    const uint reduction = ne01;
    const uint col_groups = (cols + BN - 1u) / BN;
    const uint flat = gid.y + gid.z * 65535u;
    const uint batch = flat / col_groups;
    const uint col_block = flat - batch * col_groups;
    const uint batch_count = ne2 * ne3;
    if (batch >= batch_count) {
        return;
    }

    const uint row_start = gid.x * BM;
    const uint col_start = col_block * BN;
    if (row_start >= rows || col_start >= cols) {
        return;
    }

    const uint i2 = batch % ne2;
    const uint i3 = batch / ne2;
    const uint dps2 = ne2 / ne02;
    const uint dps3 = ne3 / ne03;
    const uint i02 = i2 / dps2;
    const uint i03 = i3 / dps3;

    const uint wave = tid / WAVE_SIZE;
    const uint lane = tid % WAVE_SIZE;
#ifdef LA_INTEL_WAVE
    float acc0[ACC_E];
    float acc1[ACC_E];
    [unroll] for (uint e = 0; e < ACC_E; e++) {
        acc0[e] = 0.0f;
        acc1[e] = 0.0f;
    }
#else
    MatAcc acc0 = MatAcc::Splat(0.0f);
    MatAcc acc1 = MatAcc::Splat(0.0f);
#endif

    for (uint k0 = 0; k0 < reduction; k0 += BK) {
        for (uint i = tid; i < BM * BK; i += THREADS) {
            const uint r = i / BK;
            const uint k = i % BK;
            const uint gr = row_start + r;
            const uint gk = k0 + k;
            float value = 0.0f;
            if (gr < rows && gk < reduction) {
                value = load_auto(src1, src1_offset + gr * nb10 + gk * nb11
                                                   + i2 * nb12 + i3 * nb13, src1_esize);
            }
            tile_a[i] = (float16_t)value;
        }
        for (uint i = tid; i < BK * BN; i += THREADS) {
            const uint k = i / BN;
            const uint c = i % BN;
            const uint gk = k0 + k;
            const uint gc = col_start + c;
            float value = 0.0f;
            if (gk < reduction && gc < cols) {
                value = load_auto(src0, src0_offset + gc * nb00 + gk * nb01
                                                   + i02 * nb02 + i03 * nb03, src0_esize);
            }
            tile_b[i] = (float16_t)value;
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint kk = 0; kk < BK; kk += TILE) {
#ifdef LA_INTEL_WAVE
            MatA a = MatA::Load(tile_a, wave * 8 * BK + kk, BK, MatrixLayout::RowMajor);
            MatB b0 = MatB::Load(tile_b, kk * BN, BN, MatrixLayout::RowMajor);
            MatB b1 = MatB::Load(tile_b, kk * BN + TILE, BN, MatrixLayout::RowMajor);
            MatAcc partial0 = MatAcc::Splat((float16_t)0);
            MatAcc partial1 = MatAcc::Splat((float16_t)0);
            partial0.MultiplyAccumulate(a, b0);
            partial1.MultiplyAccumulate(a, b1);
            // Drain each K=16 partial before the next F16 accumulation.
            [unroll] for (uint e = 0; e < ACC_E; e++) {
                acc0[e] += (float)partial0.Get(e);
                acc1[e] += (float)partial1.Get(e);
            }
#else
            MatA a = MatA::Load(tile_a, wave * TILE * BK + kk, BK, MatrixLayout::RowMajor);
            MatB b0 = MatB::Load(tile_b, kk * BN, BN, MatrixLayout::RowMajor);
            MatB b1 = MatB::Load(tile_b, kk * BN + TILE, BN, MatrixLayout::RowMajor);
            acc0.MultiplyAccumulate(a, b0);
            acc1.MultiplyAccumulate(a, b1);
#endif
        }

        GroupMemoryBarrierWithGroupSync();
    }

#ifdef LA_INTEL_WAVE
    MatAcc coords = MatAcc::Splat((float16_t)0);
    [unroll] for (uint e = 0; e < ACC_E; e++) {
        const uint2 rc = coords.GetCoordinate(e);
        const uint row = row_start + wave * 8 + rc.x;
        const uint col = col_start + rc.y;
        if (row < rows && col < cols) {
            const uint off = dst_offset + col * nb0 + row * nb1 + i2 * nb2 + i3 * nb3;
            dst.Store(off, asuint(acc0[e]));
        }
        if (row < rows && col + TILE < cols) {
            const uint off = dst_offset + (col + TILE) * nb0 + row * nb1 + i2 * nb2 + i3 * nb3;
            dst.Store(off, asuint(acc1[e]));
        }
    }
#else
    const uint out_row = row_start + wave * TILE;
    const uint slot = wave * TILE * TILE;

    acc0.Store(tile_c, slot, TILE, MatrixLayout::RowMajor);
    GroupMemoryBarrier();
    for (uint e = lane; e < TILE * TILE; e += WAVE_SIZE) {
        const uint r = e / TILE;
        const uint c = e % TILE;
        if (out_row + r < rows && col_start + c < cols) {
            const uint off = dst_offset + (col_start + c) * nb0 + (out_row + r) * nb1
                                        + i2 * nb2 + i3 * nb3;
            dst.Store(off, asuint(tile_c[slot + e]));
        }
    }

    acc1.Store(tile_c, slot, TILE, MatrixLayout::RowMajor);
    GroupMemoryBarrier();
    for (uint e = lane; e < TILE * TILE; e += WAVE_SIZE) {
        const uint r = e / TILE;
        const uint c = e % TILE;
        if (out_row + r < rows && col_start + TILE + c < cols) {
            const uint off = dst_offset + (col_start + TILE + c) * nb0 + (out_row + r) * nb1
                                        + i2 * nb2 + i3 * nb3;
            dst.Store(off, asuint(tile_c[slot + e]));
        }
    }
#endif
}
