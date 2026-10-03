// linalg_probe_i8_rows.hlsl - read back the groupshared I8 row mapping directly.
//
// Follow-up to linalg_probe_i8_layout.hlsl. That probe showed rows 2..7 coming
// back exactly zero when Stride was 32, which is what happens if Stride counts
// array slots rather than matrix elements: row r then starts at slot r*32 and
// walks off the populated region of the oversized array. The magnitudes were
// still wrong, but that probe passed the accumulator by value into a helper and
// reused one Matrix variable - these are opaque handles and nothing else in the
// tree does that, so it is not a safe way to ask the question.
//
// Everything here is inline, and each measurement gets its own accumulator.
//
// The probe operand carries value (element_index / LA_K) + 1, so every element
// of row-major row r holds r+1. The other operand is all ones. If the hardware
// agrees with row-major-at-this-stride then
//
//   C[r][c] = sum_k A[r][k] = 32 * (r+1)  ->  32 64 96 128 160 192 224 256
//
// and any other mapping shows up as a mixture, which is still readable: divide
// by 32 to see which source rows the hardware pulled together.
//
// Output, four 8x16 tiles:
//   OutBuff[100 + cell]  A probe, Stride = LA_K/4 slots   expect 32*(r+1)
//   OutBuff[300 + cell]  A probe, Stride = LA_K           expect 32*(r+1)
//   OutBuff[500 + cell]  B probe, Stride = LA_K/4 slots   expect 32*(c+1)
//   OutBuff[700 + cell]  B probe, Stride = LA_K           expect 32*(c+1)

#include <dx/linalg.h>
using namespace dx::linalg;

RWByteAddressBuffer OutBuff : register(u0);

#define LA_M 8
#define LA_K 32
#define LA_N 16
#define WAVE 16

#define ACC_CELLS (LA_M * LA_N)
#define A_SLOTS   (LA_M * LA_K / 4)
#define B_SLOTS   (LA_K * LA_N / 4)

typedef Matrix<ComponentType::I8,  LA_M, LA_K, MatrixUse::A,           MatrixScope::Wave> MatA8;
typedef Matrix<ComponentType::I8,  LA_K, LA_N, MatrixUse::B,           MatrixScope::Wave> MatB8;
typedef Matrix<ComponentType::I32, LA_M, LA_N, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;

groupshared int a_probe[A_SLOTS];
groupshared int b_probe[B_SLOTS];
groupshared int a_ones [A_SLOTS];
groupshared int b_ones [B_SLOTS];
groupshared int c0[ACC_CELLS];
groupshared int c1[ACC_CELLS];
groupshared int c2[ACC_CELLS];
groupshared int c3[ACC_CELLS];

[numthreads(WAVE, 1, 1)]
[shader("compute")]
void main(uint tid : SV_GroupIndex) {
    for (uint i = tid; i < A_SLOTS; i += WAVE) {
        int packed = 0;
        [unroll] for (uint ba = 0; ba < 4; ba++) {
            const uint e = i * 4 + ba;
            packed |= (int)(((e / LA_K) + 1) & 0xFF) << (ba * 8);
        }
        a_probe[i] = packed;
        a_ones[i]  = 0x01010101;
    }
    for (uint j = tid; j < B_SLOTS; j += WAVE) {
        int packed = 0;
        [unroll] for (uint bb = 0; bb < 4; bb++) {
            const uint e = j * 4 + bb;
            packed |= (int)(((e / LA_K) + 1) & 0xFF) << (bb * 8);
        }
        b_probe[j] = packed;
        b_ones[j]  = 0x01010101;
    }
    GroupMemoryBarrierWithGroupSync();

    MatAcc acc0 = MatAcc::Splat(0);
    acc0.MultiplyAccumulate(MatA8::Load(a_probe, 0, LA_K / 4, MatrixLayout::RowMajor),
                            MatB8::Load(b_ones,  0, LA_K / 4, MatrixLayout::ColMajor));
    acc0.Store(c0, 0, LA_N, MatrixLayout::RowMajor);

    MatAcc acc1 = MatAcc::Splat(0);
    acc1.MultiplyAccumulate(MatA8::Load(a_probe, 0, LA_K, MatrixLayout::RowMajor),
                            MatB8::Load(b_ones,  0, LA_K / 4, MatrixLayout::ColMajor));
    acc1.Store(c1, 0, LA_N, MatrixLayout::RowMajor);

    MatAcc acc2 = MatAcc::Splat(0);
    acc2.MultiplyAccumulate(MatA8::Load(a_ones,  0, LA_K / 4, MatrixLayout::RowMajor),
                            MatB8::Load(b_probe, 0, LA_K / 4, MatrixLayout::ColMajor));
    acc2.Store(c2, 0, LA_N, MatrixLayout::RowMajor);

    MatAcc acc3 = MatAcc::Splat(0);
    acc3.MultiplyAccumulate(MatA8::Load(a_ones,  0, LA_K / 4, MatrixLayout::RowMajor),
                            MatB8::Load(b_probe, 0, LA_K, MatrixLayout::ColMajor));
    acc3.Store(c3, 0, LA_N, MatrixLayout::RowMajor);

    GroupMemoryBarrierWithGroupSync();

    for (uint z = tid; z < ACC_CELLS; z += WAVE) {
        OutBuff.Store((100u + z) * 4u, asuint(c0[z]));
        OutBuff.Store((300u + z) * 4u, asuint(c1[z]));
        OutBuff.Store((500u + z) * 4u, asuint(c2[z]));
        OutBuff.Store((700u + z) * 4u, asuint(c3[z]));
    }
    if (WaveIsFirstLane()) {
        OutBuff.Store(0, 0);
    }
}
