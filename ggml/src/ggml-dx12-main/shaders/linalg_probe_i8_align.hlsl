// linalg_probe_i8_align.hlsl - how much alignment does the descriptor operand
// load actually require?
//
// The descriptor path is the only working way to feed operands on this driver
// (groupshared operand loads return 0x7F fill), so the shape of any Q8_0 GEMM
// now depends on what offsets and strides Load will accept. Q8_0 is awkward
// here: a block is 34 bytes, and its 32 quants start 2 bytes in, so consecutive
// blocks along K sit at a 34-byte pitch and nothing is 4-byte aligned, let
// alone the 128 the Load default asks for.
//
// Each case repeats the known-good "A probe x B ones" measurement, which must
// come back as 32*(r+1) down the rows, but stages A at a deliberately awkward
// offset and passes a matching Align. A case that still reads 32,64,...,256 is
// safe to build on; anything else marks that alignment as unusable.
//
//   OutBuff[100 + cell]  offset 8192 (128-aligned), Align 128   control
//   OutBuff[300 + cell]  offset 8196 (4-aligned),   Align 4
//   OutBuff[500 + cell]  offset 8194 (2-aligned),   Align 2
//   OutBuff[700 + cell]  offset 8193 (unaligned),   Align 1
//   OutBuff[900 + cell]  offset 8192, stride 34 (Q8_0 pitch), Align 4

#include <dx/linalg.h>
using namespace dx::linalg;

RWByteAddressBuffer OutBuff : register(u0);

#define LA_M 8
#define LA_K 32
#define LA_N 16
#define WAVE 16

#define ACC_CELLS (LA_M * LA_N)

#define OFF_B_ONES 12288u

typedef Matrix<ComponentType::I8,  LA_M, LA_K, MatrixUse::A,           MatrixScope::Wave> MatA8;
typedef Matrix<ComponentType::I8,  LA_K, LA_N, MatrixUse::B,           MatrixScope::Wave> MatB8;
typedef Matrix<ComponentType::I32, LA_M, LA_N, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;

groupshared int cs[ACC_CELLS];

// Write A rows carrying value r+1 at byte pitch `pitch`, starting at `off`.
// Zeroing has to clear whole dwords over the whole staging window first: at an
// unaligned offset or pitch, rows share dwords, so a byte-addressed Store would
// blank the neighbouring row it overlaps.
void stage_a(uint tid, uint off, uint pitch) {
    for (uint w = tid; w < 256; w += WAVE) {
        OutBuff.Store(8192u + w * 4u, 0u);
    }
    DeviceMemoryBarrierWithGroupSync();
    for (uint e = tid; e < LA_M * LA_K; e += WAVE) {
        const uint r = e / LA_K;
        const uint k = e % LA_K;
        const uint a = off + r * pitch + k;
        uint old;
        OutBuff.InterlockedOr(a & ~3u, ((r + 1) & 0xFFu) << ((a & 3u) * 8u), old);
    }
    DeviceMemoryBarrierWithGroupSync();
}

[numthreads(WAVE, 1, 1)]
[shader("compute")]
void main(uint tid : SV_GroupIndex) {
    for (uint j = tid; j < LA_K * LA_N / 4; j += WAVE) {
        OutBuff.Store(OFF_B_ONES + j * 4u, 0x01010101u);
    }
    DeviceMemoryBarrierWithGroupSync();

    stage_a(tid, 8192u, LA_K);
    MatAcc a0 = MatAcc::Splat(0);
    a0.MultiplyAccumulate(MatA8::Load(OutBuff, 8192u, LA_K, MatrixLayout::RowMajor, 128),
                          MatB8::Load(OutBuff, OFF_B_ONES, LA_K, MatrixLayout::ColMajor, 128));
    a0.Store(cs, 0, LA_N, MatrixLayout::RowMajor);
    GroupMemoryBarrierWithGroupSync();
    for (uint z0 = tid; z0 < ACC_CELLS; z0 += WAVE) { OutBuff.Store((100u + z0) * 4u, asuint(cs[z0])); }

    stage_a(tid, 8196u, LA_K);
    MatAcc a1 = MatAcc::Splat(0);
    a1.MultiplyAccumulate(MatA8::Load(OutBuff, 8196u, LA_K, MatrixLayout::RowMajor, 4),
                          MatB8::Load(OutBuff, OFF_B_ONES, LA_K, MatrixLayout::ColMajor, 128));
    a1.Store(cs, 0, LA_N, MatrixLayout::RowMajor);
    GroupMemoryBarrierWithGroupSync();
    for (uint z1 = tid; z1 < ACC_CELLS; z1 += WAVE) { OutBuff.Store((300u + z1) * 4u, asuint(cs[z1])); }

    stage_a(tid, 8194u, LA_K);
    MatAcc a2 = MatAcc::Splat(0);
    a2.MultiplyAccumulate(MatA8::Load(OutBuff, 8194u, LA_K, MatrixLayout::RowMajor, 2),
                          MatB8::Load(OutBuff, OFF_B_ONES, LA_K, MatrixLayout::ColMajor, 128));
    a2.Store(cs, 0, LA_N, MatrixLayout::RowMajor);
    GroupMemoryBarrierWithGroupSync();
    for (uint z2 = tid; z2 < ACC_CELLS; z2 += WAVE) { OutBuff.Store((500u + z2) * 4u, asuint(cs[z2])); }

    stage_a(tid, 8193u, LA_K);
    MatAcc a3 = MatAcc::Splat(0);
    a3.MultiplyAccumulate(MatA8::Load(OutBuff, 8193u, LA_K, MatrixLayout::RowMajor, 1),
                          MatB8::Load(OutBuff, OFF_B_ONES, LA_K, MatrixLayout::ColMajor, 128));
    a3.Store(cs, 0, LA_N, MatrixLayout::RowMajor);
    GroupMemoryBarrierWithGroupSync();
    for (uint z3 = tid; z3 < ACC_CELLS; z3 += WAVE) { OutBuff.Store((700u + z3) * 4u, asuint(cs[z3])); }

    stage_a(tid, 8192u, 34u);
    MatAcc a4 = MatAcc::Splat(0);
    a4.MultiplyAccumulate(MatA8::Load(OutBuff, 8192u, 34u, MatrixLayout::RowMajor, 4),
                          MatB8::Load(OutBuff, OFF_B_ONES, LA_K, MatrixLayout::ColMajor, 128));
    a4.Store(cs, 0, LA_N, MatrixLayout::RowMajor);
    GroupMemoryBarrierWithGroupSync();
    for (uint z4 = tid; z4 < ACC_CELLS; z4 += WAVE) { OutBuff.Store((900u + z4) * 4u, asuint(cs[z4])); }

    if (WaveIsFirstLane()) {
        OutBuff.Store(0, 0);
    }
}
