// Probe: does groupshared Matrix::Load work when the array element type
// matches the matrix component type?
//
// Background. linalg_repro_groupshared_load.hlsl reported groupshared loads as
// broken: an I8 matrix read from a groupshared int[] returned 0x7F for every
// element. Intel replied that the groupshared load/store semantics changed in
// late July and shipping drivers still implement the OLD rule: the value is
// CONVERTED when the array element type differs from the matrix component
// type. Under that rule the old repro was miscoded, not the driver broken -
// each int slot held four packed bytes (0x03030303 = 50529027), and converting
// that to i8 saturates to 127. That explains 32*127*127 = 516128 exactly, and
// explains why the result did not change with FILL_A.
//
// It also explains the one case that worked: the accumulator Store used
// ComponentType::I32 into a groupshared int[], where the types already match.
//
// So the arrays here hold ONE logical element per slot with a matching type.
//
//   dxc -T cs_6_10 -E main -enable-16bit-types -I <dxc>/inc/hlsl \
//       -Fo probe.cso linalg_probe_gs_typed.hlsl
//
// Output bases (read with raw-count 1100):
//   100 : test 1, f16 matching  - expect 32 in all 128 cells
//   300 : test 1 control, a_arr via ordinary load - expect 1
//   500 : test 2, i8 one-elem-per-int - expect 96 in all 128 cells
//   700 : test 2 control, a8 via ordinary load - expect 3

#include <dx/linalg.h>
using namespace dx::linalg;

RWByteAddressBuffer OutBuff : register(u0);

#define WAVE 16

// ---- test 1: f16, the shape this part actually advertises (8x16x16) -------
#define F_M 8
#define F_K 16
#define F_N 16

typedef Matrix<ComponentType::F16, F_M, F_K, MatrixUse::A,           MatrixScope::Wave> MatAf;
typedef Matrix<ComponentType::F16, F_K, F_N, MatrixUse::B,           MatrixScope::Wave> MatBf;
typedef Matrix<ComponentType::F16, F_M, F_N, MatrixUse::Accumulator, MatrixScope::Wave> MatAccf;

groupshared float16_t af_arr[F_M * F_K];   // 128, one element per slot
groupshared float16_t bf_arr[F_K * F_N];   // 256
groupshared float16_t cf_arr[F_M * F_N];   // 128

// ---- test 2: i8 conversion from int[], one logical element per slot -------
#define I_M 8
#define I_K 32
#define I_N 16

#define FILL_A 3
#define FILL_B 1

typedef Matrix<ComponentType::I8,  I_M, I_K, MatrixUse::A,           MatrixScope::Wave> MatA8;
typedef Matrix<ComponentType::I8,  I_K, I_N, MatrixUse::B,           MatrixScope::Wave> MatB8;
typedef Matrix<ComponentType::I32, I_M, I_N, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc8;

groupshared int a8_arr[I_M * I_K];   // 256, NOT packed - one i8 value per int
groupshared int b8_arr[I_K * I_N];   // 512
groupshared int c8_arr[I_M * I_N];   // 128

[numthreads(WAVE, 1, 1)]
[shader("compute")]
void main(uint tid : SV_GroupIndex) {
    // ---------------- test 1: f16 matching types ----------------
    for (uint i = tid; i < F_M * F_K; i += WAVE) {
        af_arr[i] = (float16_t)1.0;
    }
    for (uint j = tid; j < F_K * F_N; j += WAVE) {
        bf_arr[j] = (float16_t)2.0;
    }
    GroupMemoryBarrierWithGroupSync();

    MatAccf accf = MatAccf::Splat((float16_t)0.0);
    accf.MultiplyAccumulate(
        MatAf::Load(af_arr, 0, F_K, MatrixLayout::RowMajor),
        MatBf::Load(bf_arr, 0, F_K, MatrixLayout::ColMajor));
    accf.Store(cf_arr, 0, F_N, MatrixLayout::RowMajor);

    GroupMemoryBarrierWithGroupSync();

    // expect K * 1.0 * 2.0 = 32 in every cell
    for (uint z = tid; z < F_M * F_N; z += WAVE) {
        OutBuff.Store((100u + z) * 4u, asuint((int)cf_arr[z]));
    }
    for (uint ac = tid; ac < F_M * F_K; ac += WAVE) {
        OutBuff.Store((300u + ac) * 4u, asuint((int)af_arr[ac]));
    }

    // ---------------- test 2: i8 from int[], one elem per slot ----------------
    for (uint p = tid; p < I_M * I_K; p += WAVE) {
        a8_arr[p] = FILL_A;
    }
    for (uint q = tid; q < I_K * I_N; q += WAVE) {
        b8_arr[q] = FILL_B;
    }
    GroupMemoryBarrierWithGroupSync();

    MatAcc8 acc8 = MatAcc8::Splat(0);
    acc8.MultiplyAccumulate(
        MatA8::Load(a8_arr, 0, I_K, MatrixLayout::RowMajor),
        MatB8::Load(b8_arr, 0, I_K, MatrixLayout::ColMajor));
    acc8.Store(c8_arr, 0, I_N, MatrixLayout::RowMajor);

    GroupMemoryBarrierWithGroupSync();

    // expect K * FILL_A * FILL_B = 32*3*1 = 96 in every cell
    for (uint z2 = tid; z2 < I_M * I_N; z2 += WAVE) {
        OutBuff.Store((500u + z2) * 4u, asuint(c8_arr[z2]));
    }
    for (uint a2 = tid; a2 < I_M * I_K; a2 += WAVE) {
        OutBuff.Store((700u + a2) * 4u, asuint(a8_arr[a2]));
    }

    if (WaveIsFirstLane()) {
        OutBuff.Store(0, 0);
    }
}
