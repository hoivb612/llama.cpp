// Minimal repro: Matrix::Load from groupshared ignores the array contents.
//
// Build:
//   dxc -T cs_6_10 -E main -I <dxc>/inc/hlsl -Fo repro.cso linalg_repro_groupshared_load.hlsl
//
// Device:  Intel Arc B390 (Xe3), wave 16, driver 32.0.101.8992
// Runtime: Agility SDK 1.721.3-preview, DXC 1.10.2605.24, SM 6.10
//
// Shape used is the one this part advertises:
//   wave s8 x s8 -> s32, 8x32x16   (M=8, K=32, N=16)
//
// The test is a multiply of two constant matrices, so the expected answer is
// exact and trivial:
//
//   A[r][k] = FILL_A  for all r,k        (8 x 32, RowMajor)
//   B[k][c] = FILL_B  for all k,c        (32 x 16, ColMajor)
//   C[r][c] = sum over 32 k of A*B = 32 * FILL_A * FILL_B
//
// With FILL_A = 3 and FILL_B = 1 that is 96 in every cell.
//
// OBSERVED on this device: every cell is 516128 = 32 * 127 * 127, i.e. both
// operands were read as 0x7F. 127 is never written by this shader - the
// largest byte it stores is FILL_A - so the value cannot come from the input.
//
// The result does not change when FILL_A / FILL_B change, when Stride changes,
// or when the probed operand switches from A to B.
//
// Section 2 of the output is the control: the same groupshared arrays read
// back with ordinary loads. It shows the arrays do hold the expected bytes at
// the time of the matrix load, so the data is present and the barrier is
// correct - only Matrix::Load fails to see it.
//
// Note the accumulator's groupshared Store works correctly in the same
// dispatch, so the defect is specific to loading operands.

#include <dx/linalg.h>
using namespace dx::linalg;

RWByteAddressBuffer OutBuff : register(u0);

#define LA_M 8
#define LA_K 32
#define LA_N 16
#define WAVE 16

#define FILL_A 3
#define FILL_B 1

#define A_SLOTS (LA_M * LA_K / 4)   // 64  ints, 4 packed i8 each
#define B_SLOTS (LA_K * LA_N / 4)   // 128 ints
#define C_CELLS (LA_M * LA_N)       // 128 i32

typedef Matrix<ComponentType::I8,  LA_M, LA_K, MatrixUse::A,           MatrixScope::Wave> MatA8;
typedef Matrix<ComponentType::I8,  LA_K, LA_N, MatrixUse::B,           MatrixScope::Wave> MatB8;
typedef Matrix<ComponentType::I32, LA_M, LA_N, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;

groupshared int a_arr[A_SLOTS];
groupshared int b_arr[B_SLOTS];
groupshared int c_arr[C_CELLS];

[numthreads(WAVE, 1, 1)]
[shader("compute")]
void main(uint tid : SV_GroupIndex) {
    const int a_packed = (FILL_A) | (FILL_A << 8) | (FILL_A << 16) | (FILL_A << 24);
    const int b_packed = (FILL_B) | (FILL_B << 8) | (FILL_B << 16) | (FILL_B << 24);

    for (uint i = tid; i < A_SLOTS; i += WAVE) {
        a_arr[i] = a_packed;
    }
    for (uint j = tid; j < B_SLOTS; j += WAVE) {
        b_arr[j] = b_packed;
    }
    GroupMemoryBarrierWithGroupSync();

    // Stride is in packed slots here (LA_K / 4). Passing LA_K instead, or any
    // other value, gives the same output - which is the point.
    MatAcc acc = MatAcc::Splat(0);
    acc.MultiplyAccumulate(
        MatA8::Load(a_arr, 0, LA_K / 4, MatrixLayout::RowMajor),
        MatB8::Load(b_arr, 0, LA_K / 4, MatrixLayout::ColMajor));
    acc.Store(c_arr, 0, LA_N, MatrixLayout::RowMajor);

    GroupMemoryBarrierWithGroupSync();

    // 1. matrix multiply result: expect 32*FILL_A*FILL_B = 96 in all 128 cells
    for (uint z = tid; z < C_CELLS; z += WAVE) {
        OutBuff.Store((100u + z) * 4u, asuint(c_arr[z]));
    }

    // 2. control - the same groupshared arrays via ordinary loads.
    //    Expect a_packed in [300..363] and b_packed in [400..527].
    for (uint ai = tid; ai < A_SLOTS; ai += WAVE) {
        OutBuff.Store((300u + ai) * 4u, asuint(a_arr[ai]));
    }
    for (uint bi = tid; bi < B_SLOTS; bi += WAVE) {
        OutBuff.Store((400u + bi) * 4u, asuint(b_arr[bi]));
    }

    if (WaveIsFirstLane()) {
        OutBuff.Store(0, 0);
    }
}
