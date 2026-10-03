// linalg_probe_i8.hlsl - toolchain probe for integer wave matrices.
//
// The F16 wave-matrix GEMM tops out well below the dp4a kernel on RDNA4, so
// the question is whether the SM 6.10 I8 component type is usable: an
// I8 x I8 -> I32 MultiplyAccumulate would skip the dequant-to-F16 staging
// entirely and has several times the arithmetic peak. This probe only
// establishes that the compiler and driver accept the type.

#include <dx/linalg.h>

using namespace dx::linalg;

RWByteAddressBuffer OutBuff : register(u0);

#define TILE 16

typedef Matrix<ComponentType::I8,  TILE, TILE, MatrixUse::A,           MatrixScope::Wave> MatA8;
typedef Matrix<ComponentType::I8,  TILE, TILE, MatrixUse::B,           MatrixScope::Wave> MatB8;
typedef Matrix<ComponentType::I32, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave> MatAccI;

groupshared int tile_a[TILE * TILE];
groupshared int tile_b[TILE * TILE];
groupshared int tile_c[TILE * TILE];

[numthreads(64, 1, 1)]
[shader("compute")]
void main(uint tid : SV_GroupIndex) {
    for (uint i = tid; i < TILE * TILE; i += 64) {
        tile_a[i] = (int)i;
        tile_b[i] = (int)i;
    }
    GroupMemoryBarrierWithGroupSync();

    MatA8   a   = MatA8::Load(tile_a, 0, TILE, MatrixLayout::RowMajor);
    MatB8   b   = MatB8::Load(tile_b, 0, TILE, MatrixLayout::ColMajor);
    MatAccI acc = MatAccI::Splat(0);
    acc.MultiplyAccumulate(a, b);
    acc.Store(tile_c, 0, TILE, MatrixLayout::RowMajor);

    GroupMemoryBarrierWithGroupSync();
    OutBuff.Store(tid * 4, asuint(tile_c[tid]));
}
