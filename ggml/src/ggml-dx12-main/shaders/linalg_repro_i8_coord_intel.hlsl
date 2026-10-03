// linalg_repro_i8_coord_intel.hlsl - does GetCoordinate() work on the one wave
// shape Intel Xe3 actually implements?
//
// The existing GetCoordinate repro targets 16x16x16 F16/I8 on wave 64, which
// this part does not support at all. Intel reports exactly:
//
//   wave f16xf16 -> f16   shapes: 8x16x16
//   wave s8xs8   -> s32   shapes: 8x32x16
//
// so the question has to be re-asked at M=8 K=32 N=16 on wave 16. The answer
// decides the epilogue of the integer GEMM: if GetCoordinate is right, a lane
// can keep its f32 accumulators in registers, and only if it is wrong does
// every Q8_0 block have to round-trip an int32 tile through LDS.
//
// The test data makes each output cell name itself. A is the identity in its
// first 8 columns and B[k][c] = k*16+c for k < 8, so
//
//   C[r][c] = sum_k A[r][k] * B[k][c] = B[r][c] = r*16 + c
//
// Every value fits in int8 (max 7*16+15 = 127), and the product is exact, so
// any deviation is a real defect rather than rounding.
//
// Results:
//   OutBuff[0]  cells where the Store path and the GetCoordinate path disagree
//   OutBuff[1]  cells where Get(e) != reported_row*16 + reported_col
//   OutBuff[2]  cells where the Store path itself is wrong (checks the MMA)
//   OutBuff[3]  accumulator elements per lane
//   OutBuff[513 + lane*ACC_E + e]  reported coordinate, (row << 16) | col
//   OutBuff[700 + cell]            value placed by Store
//   OutBuff[900 + cell]            value placed by GetCoordinate
//
// All three counters must be zero.
//
// Build:
//   dxc -T cs_6_10 -E main -I <dxc>/inc/hlsl -Fo repro_i8.cso \
//       linalg_repro_i8_coord_intel.hlsl

#include <dx/linalg.h>
using namespace dx::linalg;

RWByteAddressBuffer OutBuff : register(u0);

#define LA_M 8
#define LA_K 32
#define LA_N 16
#define WAVE 16

#define ACC_CELLS (LA_M * LA_N)
#define ACC_E     (ACC_CELLS / WAVE)

typedef Matrix<ComponentType::I8,  LA_M, LA_K, MatrixUse::A,           MatrixScope::Wave> MatA8;
typedef Matrix<ComponentType::I8,  LA_K, LA_N, MatrixUse::B,           MatrixScope::Wave> MatB8;
typedef Matrix<ComponentType::I32, LA_M, LA_N, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;

// I8 packs 4 elements per uint.
groupshared int a_tile[LA_M * LA_K / 4];
groupshared int b_tile[LA_K * LA_N / 4];
groupshared int c_store[ACC_CELLS];
groupshared int c_coord[ACC_CELLS];

[numthreads(WAVE, 1, 1)]
[shader("compute")]
void main(uint tid : SV_GroupIndex) {
    // A[r][k] = (k == r), row-major, stride LA_K.
    for (uint wa = tid; wa < LA_M * LA_K / 4; wa += WAVE) {
        int packed = 0;
        [unroll] for (uint ba = 0; ba < 4; ba++) {
            const uint idx = wa * 4 + ba;
            const uint r   = idx / LA_K;
            const uint k   = idx % LA_K;
            const int  v   = (k == r) ? 1 : 0;
            packed |= (v & 0xFF) << (ba * 8);
        }
        a_tile[wa] = packed;
    }
    // B[k][c] = k < 8 ? k*16 + c : 0, column-major, stride LA_K.
    for (uint wb = tid; wb < LA_K * LA_N / 4; wb += WAVE) {
        int packed = 0;
        [unroll] for (uint bb = 0; bb < 4; bb++) {
            const uint idx = wb * 4 + bb;
            const uint c   = idx / LA_K;
            const uint k   = idx % LA_K;
            const int  v   = (k < LA_M) ? (int)(k * LA_N + c) : 0;
            packed |= (v & 0xFF) << (bb * 8);
        }
        b_tile[wb] = packed;
    }
    for (uint z = tid; z < ACC_CELLS; z += WAVE) {
        c_store[z] = 0;
        c_coord[z] = -1;
    }
    GroupMemoryBarrierWithGroupSync();

    // Stride is in array indices, not matrix elements: I8 packs 4 per uint, so
    // a LA_K-element row/column spans LA_K/4 array slots.
    MatA8  a   = MatA8::Load(a_tile, 0, LA_K / 4, MatrixLayout::RowMajor);
    MatB8  b   = MatB8::Load(b_tile, 0, LA_K / 4, MatrixLayout::ColMajor);
    MatAcc acc = MatAcc::Splat(0);
    acc.MultiplyAccumulate(a, b);

    // Reference placement.
    acc.Store(c_store, 0, LA_N, MatrixLayout::RowMajor);

    // Placement by the coordinates the API reports.
    uint bad_self = 0;
    for (uint e = 0; e < ACC_E; e++) {
        const uint2 rc = acc.GetCoordinate(e);
        const int   v  = asint(acc.Get(e));
        OutBuff.Store((513u + tid * ACC_E + e) * 4u, (rc.x << 16) | rc.y);
        if (rc.x < LA_M && rc.y < LA_N) {
            c_coord[rc.x * LA_N + rc.y] = v;
            if (v != (int)(rc.x * LA_N + rc.y)) {
                bad_self++;
            }
        } else {
            bad_self++;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint bad_cmp = 0;
    uint bad_mma = 0;
    for (uint j = tid; j < ACC_CELLS; j += WAVE) {
        if (c_store[j] != c_coord[j]) {
            bad_cmp++;
        }
        if (c_store[j] != (int)j) {
            bad_mma++;
        }
        OutBuff.Store((700u + j) * 4u, asuint(c_store[j]));
        OutBuff.Store((900u + j) * 4u, asuint(c_coord[j]));
    }
    const uint tot_cmp  = WaveActiveSum(bad_cmp);
    const uint tot_self = WaveActiveSum(bad_self);
    const uint tot_mma  = WaveActiveSum(bad_mma);
    if (WaveIsFirstLane()) {
        OutBuff.Store(0, tot_cmp);
        OutBuff.Store(4, tot_self);
        OutBuff.Store(8, tot_mma);
        OutBuff.Store(12, ACC_E);
    }
}
