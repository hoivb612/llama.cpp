// conv_linalg.hlsl - direct convolution via SM 6.10 wave matrices.
//
// The convolution is evaluated as:
//   [output_channels, reduction] x [reduction, output_positions]
//
// Input tiles are gathered directly into LDS, avoiding a materialized im2col
// tensor. CONV_KIND selects the address mapping:
//   2 = CONV_2D
//   3 = CONV_TRANSPOSE_2D
//   4 = CONV_3D

#include "ggml_common.hlsli"
#include <dx/linalg.h>

using namespace dx::linalg;

#ifndef WAVE_SIZE
#error "conv_linalg requires -D WAVE_SIZE"
#endif
#ifndef CONV_KIND
#error "conv_linalg requires -D CONV_KIND"
#endif

#define TILE    16
#ifndef CONV_BM
#define CONV_BM 32
#endif
#define BM      CONV_BM
#define BN      32
#define BK      32
#ifdef LA_INTEL_WAVE
#if WAVE_SIZE != 16
#error "LA_INTEL_WAVE requires WAVE_SIZE=16"
#endif
#if CONV_BM % 8 != 0
#error "LA_INTEL_WAVE requires CONV_BM to be a multiple of 8"
#endif
#define NWAVE   (BM / 8)
#define ACC_E   (8 * TILE / WAVE_SIZE)
#else
#define NWAVE   (BM / TILE)
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

uint fastdiv(uint n, uint mp, uint l) {
    const uint hi = (uint)(((uint64_t)n * (uint64_t)mp) >> 32);
    return (hi + n) >> l;
}

float load_kernel(uint oc, uint rk) {
#if CONV_KIND == 2
    const uint packed_l = op10;
    const uint cin = fastdiv(rk, op9, packed_l >> 24);
    const uint rem = rk - cin * ne00 * ne01;
    const uint kh  = fastdiv(rem, op8, (packed_l >> 16) & 0xFFu);
    const uint kw  = rem - kh * ne00;
    return load_auto(src0, src0_offset + kw * nb00 + kh * nb01 + cin * nb02 + oc * nb03,
                     src0_esize);
#elif CONV_KIND == 3
    const uint packed_l = op10;
    const uint cin = fastdiv(rk, op9, packed_l >> 24);
    const uint rem = rk - cin * ne00 * ne01;
    const uint kh  = fastdiv(rem, op8, (packed_l >> 16) & 0xFFu);
    const uint kw  = rem - kh * ne00;
    return load_auto(src0, src0_offset + kw * nb00 + kh * nb01 + oc * nb02 + cin * nb03,
                     src0_esize);
#else
    const uint kw = rk % ne00;
    uint rem = rk / ne00;
    const uint kh = rem % ne01;
    rem /= ne01;
    const uint kd  = rem % ne02;
    const uint cin = rem / ne02;
    const uint ic = op9;
    return load_auto(src0, src0_offset + kw * nb00 + kh * nb01 + kd * nb02
                                      + (oc * ic + cin) * nb03, src0_esize);
#endif
}

float load_input(uint pos, uint rk) {
#if CONV_KIND == 2 || CONV_KIND == 3
    const uint packed_l = op10;
    const uint batch = fastdiv(pos, op7, (packed_l >> 8) & 0xFFu);
    const uint remp = pos - batch * ne0 * ne1;
    const uint oh = fastdiv(remp, op6, packed_l & 0xFFu);
    const uint ow = remp - oh * ne0;

    const uint cin = fastdiv(rk, op9, packed_l >> 24);
    const uint remk = rk - cin * ne00 * ne01;
    const uint kh  = fastdiv(remk, op8, (packed_l >> 16) & 0xFFu);
    const uint kw  = remk - kh * ne00;

#if CONV_KIND == 2
    const int ix = (int)ow * asint(op0) + (int)kw * asint(op4) - asint(op2);
    const int iy = (int)oh * asint(op1) + (int)kh * asint(op5) - asint(op3);
#else
    const int stride = asint(op0);
    const int dx = (int)ow - (int)kw;
    const int dy = (int)oh - (int)kh;
    if (stride <= 0 || dx < 0 || dy < 0 || (dx % stride) != 0 || (dy % stride) != 0) {
        return 0.0f;
    }
    const int ix = dx / stride;
    const int iy = dy / stride;
#endif
    if (ix < 0 || iy < 0 || ix >= (int)ne10 || iy >= (int)ne11) {
        return 0.0f;
    }
    return load_auto(src1, src1_offset + (uint)ix * nb10 + (uint)iy * nb11
                                      + cin * nb12 + batch * nb13, src1_esize);
#else
    const uint ox = pos % ne0;
    uint remp = pos / ne0;
    const uint oy = remp % ne1;
    remp /= ne1;
    const uint oz = remp % ne2;
    const uint batch = remp / ne2;

    const uint kw = rk % ne00;
    uint remk = rk / ne00;
    const uint kh = remk % ne01;
    remk /= ne01;
    const uint kd  = remk % ne02;
    const uint cin = remk / ne02;

    const int ix = (int)ox * asint(op0) + (int)kw * asint(op6) - asint(op3);
    const int iy = (int)oy * asint(op1) + (int)kh * asint(op7) - asint(op4);
    const int iz = (int)oz * asint(op2) + (int)kd * asint(op8) - asint(op5);
    if (ix < 0 || iy < 0 || iz < 0 ||
        ix >= (int)ne10 || iy >= (int)ne11 || iz >= (int)ne12) {
        return 0.0f;
    }
    const uint ic = op9;
    return load_auto(src1, src1_offset + (uint)ix * nb10 + (uint)iy * nb11
                                      + (uint)iz * nb12 + (batch * ic + cin) * nb13,
                     src1_esize);
#endif
}

void store_output(uint oc, uint pos, float value) {
#if CONV_KIND == 2 || CONV_KIND == 3
    const uint packed_l = op10;
    const uint batch = fastdiv(pos, op7, (packed_l >> 8) & 0xFFu);
    const uint rem = pos - batch * ne0 * ne1;
    const uint oh = fastdiv(rem, op6, packed_l & 0xFFu);
    const uint ow = rem - oh * ne0;
    store_auto(dst, dst_offset + ow * nb0 + oh * nb1 + oc * nb2 + batch * nb3,
               value, dst_esize);
#else
    const uint ox = pos % ne0;
    uint rem = pos / ne0;
    const uint oy = rem % ne1;
    rem /= ne1;
    const uint oz = rem % ne2;
    const uint batch = rem / ne2;
    const uint out_channels = op11;
    store_auto(dst, dst_offset + ox * nb0 + oy * nb1 + oz * nb2
                           + (batch * out_channels + oc) * nb3,
               value, dst_esize);
#endif
}

WAVE_SIZE_ATTR
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint rows;
    uint cols;
    uint reduction;

#if CONV_KIND == 2
    rows = ne2;
    cols = ne0 * ne1 * ne3;
    reduction = ne00 * ne01 * ne02;
#elif CONV_KIND == 3
    rows = ne2;
    cols = ne0 * ne1 * ne3;
    reduction = ne00 * ne01 * ne03;
#else
    rows = op11;
    const uint batches = rows == 0 ? 0 : ne3 / rows;
    cols = ne0 * ne1 * ne2 * batches;
    reduction = ne00 * ne01 * ne02 * op9;
#endif

    const uint row_block = gid.x;
    const uint col_block = gid.y + gid.z * 65535u;
    const uint row_start = row_block * BM;
    const uint col_start = col_block * BN;
    if (row_start >= rows || col_start >= cols) {
        return;
    }

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
            tile_a[i] = (float16_t)((gr < rows && gk < reduction) ? load_kernel(gr, gk) : 0.0f);
        }
        for (uint i = tid; i < BK * BN; i += THREADS) {
            const uint k = i / BN;
            const uint c = i % BN;
            const uint gk = k0 + k;
            const uint gc = col_start + c;
            tile_b[i] = (float16_t)((gk < reduction && gc < cols) ? load_input(gc, gk) : 0.0f);
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
            store_output(row, col, acc0[e]);
        }
        if (row < rows && col + TILE < cols) {
            store_output(row, col + TILE, acc1[e]);
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
            store_output(out_row + r, col_start + c, tile_c[slot + e]);
        }
    }

    acc1.Store(tile_c, slot, TILE, MatrixLayout::RowMajor);
    GroupMemoryBarrier();
    for (uint e = lane; e < TILE * TILE; e += WAVE_SIZE) {
        const uint r = e / TILE;
        const uint c = e % TILE;
        if (out_row + r < rows && col_start + TILE + c < cols) {
            store_output(out_row + r, col_start + TILE + c, tile_c[slot + e]);
        }
    }
#endif
}
