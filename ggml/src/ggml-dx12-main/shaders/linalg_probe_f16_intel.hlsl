// linalg_probe_f16_intel.hlsl - validate the f16 x f16 -> f16 8x16x16 wave shape
// before building a GEMM on it.
//
// The I8 probes established that operands must come from a descriptor and that
// offsets and strides get truncated to 4 bytes. F16 suits that far better than
// Q8_0: at even K an f16 row is 4-byte aligned and contiguous along K, so both
// operands can be read straight out of the model buffers with no staging.
//
// Two things still need checking, because this is a different operation to the
// I8 one and nothing about it is implied by the I8 result:
//   - is the multiply correct, given the accumulator is f16 rather than s32
//   - does GetCoordinate report the same lane=column, element=row mapping
//
// Same all-ones isolation as before. Every expected value here is an integer
// under 2048, which f16 represents exactly, so a mismatch is a real defect and
// not rounding.
//
//   OutBuff[100 + cell]  A probe x B ones   expect 16*(r+1)
//   OutBuff[300 + cell]  A ones  x B probe  expect 16*(c+1)
//   OutBuff[500 + cell]  A probe x B probe  expect 16*(r+1)*(c+1)
//   OutBuff[700 + cell]  same, placed by GetCoordinate instead of Store
//   OutBuff[900 + lane*ACC_E + e]  reported coordinate, (row << 16) | col
//
// Values are scaled by 16 on the way out so the integer readback keeps one
// fractional digit of whatever the hardware actually produced.

#include <dx/linalg.h>
using namespace dx::linalg;

RWByteAddressBuffer OutBuff : register(u0);

#define LA_M 8
#define LA_K 16
#define LA_N 16
#define WAVE 16

#define ACC_CELLS (LA_M * LA_N)
#define ACC_E     (ACC_CELLS / WAVE)

#define OFF_A_PROBE 8192u
#define OFF_A_ONES  8448u
#define OFF_B_PROBE 8704u
#define OFF_B_ONES  9216u

typedef Matrix<ComponentType::F16, LA_M, LA_K, MatrixUse::A,           MatrixScope::Wave> MatAf;
typedef Matrix<ComponentType::F16, LA_K, LA_N, MatrixUse::B,           MatrixScope::Wave> MatBf;
typedef Matrix<ComponentType::F16, LA_M, LA_N, MatrixUse::Accumulator, MatrixScope::Wave> MatAccf;

groupshared float16_t c0[ACC_CELLS];
groupshared float16_t c1[ACC_CELLS];
groupshared float16_t c2[ACC_CELLS];

[numthreads(WAVE, 1, 1)]
[shader("compute")]
void main(uint tid : SV_GroupIndex) {
    // A is 8x16 row-major, 2 bytes per element: element (r,k) at byte r*32 + k*2.
    // Row r carries the value r+1 in every k.
    for (uint i = tid; i < LA_M * LA_K / 2; i += WAVE) {
        const uint e = i * 2;
        const float16_t v0 = (float16_t)((e       / LA_K) + 1);
        const float16_t v1 = (float16_t)(((e + 1) / LA_K) + 1);
        OutBuff.Store(OFF_A_PROBE + i * 4u,
                      (uint)asuint16(v0) | ((uint)asuint16(v1) << 16));
        OutBuff.Store(OFF_A_ONES + i * 4u,
                      (uint)asuint16((float16_t)1.0) | ((uint)asuint16((float16_t)1.0) << 16));
    }
    // B is 16x16 column-major: element (k,c) at byte c*32 + k*2.
    // Column c carries the value c+1 in every k.
    for (uint j = tid; j < LA_K * LA_N / 2; j += WAVE) {
        const uint e = j * 2;
        const float16_t v0 = (float16_t)((e       / LA_K) + 1);
        const float16_t v1 = (float16_t)(((e + 1) / LA_K) + 1);
        OutBuff.Store(OFF_B_PROBE + j * 4u,
                      (uint)asuint16(v0) | ((uint)asuint16(v1) << 16));
        OutBuff.Store(OFF_B_ONES + j * 4u,
                      (uint)asuint16((float16_t)1.0) | ((uint)asuint16((float16_t)1.0) << 16));
    }
    DeviceMemoryBarrierWithGroupSync();

    const uint stride_b = LA_K * 2;

    MatAccf acc0 = MatAccf::Splat((float16_t)0.0);
    acc0.MultiplyAccumulate(MatAf::Load(OutBuff, OFF_A_PROBE, stride_b, MatrixLayout::RowMajor),
                            MatBf::Load(OutBuff, OFF_B_ONES,  stride_b, MatrixLayout::ColMajor));
    acc0.Store(c0, 0, LA_N, MatrixLayout::RowMajor);

    MatAccf acc1 = MatAccf::Splat((float16_t)0.0);
    acc1.MultiplyAccumulate(MatAf::Load(OutBuff, OFF_A_ONES,  stride_b, MatrixLayout::RowMajor),
                            MatBf::Load(OutBuff, OFF_B_PROBE, stride_b, MatrixLayout::ColMajor));
    acc1.Store(c1, 0, LA_N, MatrixLayout::RowMajor);

    MatAccf acc2 = MatAccf::Splat((float16_t)0.0);
    acc2.MultiplyAccumulate(MatAf::Load(OutBuff, OFF_A_PROBE, stride_b, MatrixLayout::RowMajor),
                            MatBf::Load(OutBuff, OFF_B_PROBE, stride_b, MatrixLayout::ColMajor));
    acc2.Store(c2, 0, LA_N, MatrixLayout::RowMajor);

    // Same accumulator, drained through the coordinates the API reports.
    for (uint e2 = 0; e2 < ACC_E; e2++) {
        const uint2 rc = acc2.GetCoordinate(e2);
        OutBuff.Store((900u + tid * ACC_E + e2) * 4u, (rc.x << 16) | rc.y);
        if (rc.x < LA_M && rc.y < LA_N) {
            OutBuff.Store((700u + rc.x * LA_N + rc.y) * 4u,
                          (uint)(int)(acc2.Get(e2) * (float16_t)16.0));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint z = tid; z < ACC_CELLS; z += WAVE) {
        OutBuff.Store((100u + z) * 4u, (uint)(int)(c0[z] * (float16_t)16.0));
        OutBuff.Store((300u + z) * 4u, (uint)(int)(c1[z] * (float16_t)16.0));
        OutBuff.Store((500u + z) * 4u, (uint)(int)(c2[z] * (float16_t)16.0));
    }
    if (WaveIsFirstLane()) {
        OutBuff.Store(0, 0);
    }
}
