// linalg_probe_i8_desc.hlsl - same question as linalg_probe_i8_rows.hlsl, but
// loading the operands from a descriptor instead of from groupshared.
//
// The groupshared probe returned 127*127 per accumulated term in every
// configuration, identical whether the probe operand was A or B and whether
// Stride was 8 or 32. A result that does not move when the input moves means
// the operand data was never read: that path hands back 0x7F-filled matrices.
// The accumulator side was fine in the same dispatch, so the defect is
// specific to loading operands out of groupshared.
//
// This probe asks whether the descriptor path is usable instead. Descriptor
// Load takes StartOffset and Stride in bytes, which is unambiguous, so there
// is no units question left to answer here - only whether the data arrives.
//
// Same isolation trick: one operand all ones, the other carrying
// (element_index / LA_K) + 1, so a correct result is
//
//   A probe: C[r][c] = 32 * (r+1)   constant across each row
//   B probe: C[r][c] = 32 * (c+1)   constant down each column
//
// Operand staging lives high in the same UAV, clear of the output tiles, at
// 128-byte aligned offsets to satisfy the default Align.
//
//   OutBuff[100 + cell]  A probe   expect 32*(r+1)
//   OutBuff[300 + cell]  B probe   expect 32*(c+1)
//   OutBuff[500 + cell]  A*B both carrying probe data, self-naming check

#include <dx/linalg.h>
using namespace dx::linalg;

RWByteAddressBuffer OutBuff : register(u0);

#define LA_M 8
#define LA_K 32
#define LA_N 16
#define WAVE 16

#define ACC_CELLS (LA_M * LA_N)

#define OFF_A_PROBE 8192u
#define OFF_A_ONES  8448u
#define OFF_B_PROBE 8704u
#define OFF_B_ONES  9216u

typedef Matrix<ComponentType::I8,  LA_M, LA_K, MatrixUse::A,           MatrixScope::Wave> MatA8;
typedef Matrix<ComponentType::I8,  LA_K, LA_N, MatrixUse::B,           MatrixScope::Wave> MatB8;
typedef Matrix<ComponentType::I32, LA_M, LA_N, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;

groupshared int c0[ACC_CELLS];
groupshared int c1[ACC_CELLS];
groupshared int c2[ACC_CELLS];

[numthreads(WAVE, 1, 1)]
[shader("compute")]
void main(uint tid : SV_GroupIndex) {
    // A is 8x32 row-major: element (r,k) at byte r*32 + k.
    for (uint i = tid; i < LA_M * LA_K / 4; i += WAVE) {
        uint packed = 0;
        [unroll] for (uint ba = 0; ba < 4; ba++) {
            const uint e = i * 4 + ba;
            packed |= (((e / LA_K) + 1) & 0xFFu) << (ba * 8);
        }
        OutBuff.Store(OFF_A_PROBE + i * 4u, packed);
        OutBuff.Store(OFF_A_ONES  + i * 4u, 0x01010101u);
    }
    // B is 32x16 column-major: element (k,c) at byte c*32 + k.
    for (uint j = tid; j < LA_K * LA_N / 4; j += WAVE) {
        uint packed = 0;
        [unroll] for (uint bb = 0; bb < 4; bb++) {
            const uint e = j * 4 + bb;
            packed |= (((e / LA_K) + 1) & 0xFFu) << (bb * 8);
        }
        OutBuff.Store(OFF_B_PROBE + j * 4u, packed);
        OutBuff.Store(OFF_B_ONES  + j * 4u, 0x01010101u);
    }
    DeviceMemoryBarrierWithGroupSync();

    MatAcc acc0 = MatAcc::Splat(0);
    acc0.MultiplyAccumulate(MatA8::Load(OutBuff, OFF_A_PROBE, LA_K, MatrixLayout::RowMajor),
                            MatB8::Load(OutBuff, OFF_B_ONES,  LA_K, MatrixLayout::ColMajor));
    acc0.Store(c0, 0, LA_N, MatrixLayout::RowMajor);

    MatAcc acc1 = MatAcc::Splat(0);
    acc1.MultiplyAccumulate(MatA8::Load(OutBuff, OFF_A_ONES,  LA_K, MatrixLayout::RowMajor),
                            MatB8::Load(OutBuff, OFF_B_PROBE, LA_K, MatrixLayout::ColMajor));
    acc1.Store(c1, 0, LA_N, MatrixLayout::RowMajor);

    MatAcc acc2 = MatAcc::Splat(0);
    acc2.MultiplyAccumulate(MatA8::Load(OutBuff, OFF_A_PROBE, LA_K, MatrixLayout::RowMajor),
                            MatB8::Load(OutBuff, OFF_B_PROBE, LA_K, MatrixLayout::ColMajor));
    acc2.Store(c2, 0, LA_N, MatrixLayout::RowMajor);

    GroupMemoryBarrierWithGroupSync();

    for (uint z = tid; z < ACC_CELLS; z += WAVE) {
        OutBuff.Store((100u + z) * 4u, asuint(c0[z]));
        OutBuff.Store((300u + z) * 4u, asuint(c1[z]));
        OutBuff.Store((500u + z) * 4u, asuint(c2[z]));
    }
    if (WaveIsFirstLane()) {
        OutBuff.Store(0, 0);
    }
}
