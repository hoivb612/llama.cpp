#include <dx/linalg.h>

using namespace dx::linalg;

ByteAddressBuffer ABuffer : register(t0);
ByteAddressBuffer BBuffer : register(t1);
RWByteAddressBuffer CBuffer : register(u0);

#ifndef BENCH_SCOPE
#define BENCH_SCOPE 1
#endif
#ifndef BENCH_M
#define BENCH_M 16
#endif
#ifndef BENCH_N
#define BENCH_N 16
#endif
#ifndef BENCH_K
#define BENCH_K 16
#endif
#ifndef K_STEPS
#define K_STEPS 16
#endif
#ifndef INNER_REPEATS
#define INNER_REPEATS 1
#endif
#ifndef WAVE_SIZE
#define WAVE_SIZE 32
#endif
#ifndef NUM_WAVES
#define NUM_WAVES 1
#endif
#ifndef GROUP_THREADS
#define GROUP_THREADS (WAVE_SIZE * NUM_WAVES)
#endif
#ifndef A_LOAD
#define A_LOAD 0
#endif
#ifndef B_LOAD
#define B_LOAD 0
#endif
#ifndef A_LAYOUT
#define A_LAYOUT 0
#endif
#ifndef B_LAYOUT
#define B_LAYOUT 0
#endif
#ifndef C_LAYOUT
#define C_LAYOUT 0
#endif
#ifndef A_SOURCE_TYPE
#define A_SOURCE_TYPE 0
#endif
#ifndef B_SOURCE_TYPE
#define B_SOURCE_TYPE 0
#endif
#ifndef A_SOURCE_LAYOUT
#define A_SOURCE_LAYOUT A_LAYOUT
#endif
#ifndef B_SOURCE_LAYOUT
#define B_SOURCE_LAYOUT B_LAYOUT
#endif
#ifndef DESCRIPTOR_ALIGN
#define DESCRIPTOR_ALIGN 32
#endif
#ifndef A_BASE_OFFSET
#define A_BASE_OFFSET 0
#endif
#ifndef B_BASE_OFFSET
#define B_BASE_OFFSET 0
#endif
#ifndef C_BASE_OFFSET
#define C_BASE_OFFSET 0
#endif
#ifndef EPILOGUE
#define EPILOGUE 0
#endif
#ifndef A_SOURCE_PAD
#define A_SOURCE_PAD 0
#endif
#ifndef B_SOURCE_PAD
#define B_SOURCE_PAD 0
#endif
#ifndef A_LDS_PAD
#define A_LDS_PAD 0
#endif
#ifndef B_LDS_PAD
#define B_LDS_PAD 0
#endif
#ifndef LOAD_VECTOR_WIDTH
#define LOAD_VECTOR_WIDTH 4
#endif
#ifndef ACC_TILES
#define ACC_TILES 1
#endif
#ifndef TILE_ORDER
#define TILE_ORDER 0
#endif

#if LOAD_VECTOR_WIDTH != 2 && LOAD_VECTOR_WIDTH != 4 && LOAD_VECTOR_WIDTH != 8
#error "LOAD_VECTOR_WIDTH must be 2, 4, or 8"
#endif

#define LOAD_DIRECT 0
#define LOAD_LDS_SCALAR 1
#define LOAD_LDS_VECTOR 2
#define LOAD_LDS_PREFETCH 3
#define LOAD_DIRECT_TRANSPOSE_CAST 4

#define TYPE_F16 0
#define TYPE_F32 1
#define TYPE_BF16 2

#if A_LAYOUT == 0
#define A_LAYOUT_ENUM MatrixLayout::RowMajor
#define A_STRIDE_BYTES ((BENCH_K + A_SOURCE_PAD) * 2)
#define A_LDS_STRIDE (BENCH_K + A_LDS_PAD)
#define A_LDS_MATRIX_ELEMS (BENCH_M * A_LDS_STRIDE)
#else
#define A_LAYOUT_ENUM MatrixLayout::ColMajor
#define A_STRIDE_BYTES ((BENCH_M + A_SOURCE_PAD) * 2)
#define A_LDS_STRIDE (BENCH_M + A_LDS_PAD)
#define A_LDS_MATRIX_ELEMS (BENCH_K * A_LDS_STRIDE)
#endif

#if B_LAYOUT == 0
#define B_LAYOUT_ENUM MatrixLayout::RowMajor
#define B_STRIDE_BYTES ((BENCH_N + B_SOURCE_PAD) * 2)
#define B_LDS_STRIDE (BENCH_N + B_LDS_PAD)
#define B_LDS_MATRIX_ELEMS (BENCH_K * B_LDS_STRIDE)
#else
#define B_LAYOUT_ENUM MatrixLayout::ColMajor
#define B_STRIDE_BYTES ((BENCH_K + B_SOURCE_PAD) * 2)
#define B_LDS_STRIDE (BENCH_K + B_LDS_PAD)
#define B_LDS_MATRIX_ELEMS (BENCH_N * B_LDS_STRIDE)
#endif

#if C_LAYOUT == 0
#define C_LAYOUT_ENUM MatrixLayout::RowMajor
#define C_STRIDE_BYTES (BENCH_N * 4)
#else
#define C_LAYOUT_ENUM MatrixLayout::ColMajor
#define C_STRIDE_BYTES (BENCH_M * 4)
#endif

#if A_SOURCE_TYPE == TYPE_F32
#define A_SOURCE_BYTES 4
#else
#define A_SOURCE_BYTES 2
#endif
#if B_SOURCE_TYPE == TYPE_F32
#define B_SOURCE_BYTES 4
#else
#define B_SOURCE_BYTES 2
#endif

#define A_MATRIX_ELEMS (BENCH_M * BENCH_K)
#define B_MATRIX_ELEMS (BENCH_K * BENCH_N)
#define C_MATRIX_ELEMS (BENCH_M * BENCH_N)

#if A_SOURCE_LAYOUT == 0
#define A_SOURCE_STRIDE (BENCH_K + A_SOURCE_PAD)
#define A_SOURCE_MATRIX_ELEMS (BENCH_M * A_SOURCE_STRIDE)
#else
#define A_SOURCE_STRIDE (BENCH_M + A_SOURCE_PAD)
#define A_SOURCE_MATRIX_ELEMS (BENCH_K * A_SOURCE_STRIDE)
#endif

#if B_SOURCE_LAYOUT == 0
#define B_SOURCE_STRIDE (BENCH_N + B_SOURCE_PAD)
#define B_SOURCE_MATRIX_ELEMS (BENCH_K * B_SOURCE_STRIDE)
#else
#define B_SOURCE_STRIDE (BENCH_K + B_SOURCE_PAD)
#define B_SOURCE_MATRIX_ELEMS (BENCH_N * B_SOURCE_STRIDE)
#endif

#define ALIGN_128(value) (((value) + 127u) & ~127u)
#define A_SOURCE_MATRIX_BYTES (A_SOURCE_MATRIX_ELEMS * A_SOURCE_BYTES)
#define B_SOURCE_MATRIX_BYTES (B_SOURCE_MATRIX_ELEMS * B_SOURCE_BYTES)
#define A_SOURCE_PITCH_BYTES ALIGN_128(A_SOURCE_MATRIX_BYTES)
#define B_SOURCE_PITCH_BYTES ALIGN_128(B_SOURCE_MATRIX_BYTES)

uint a_source_index(uint idx) {
#if A_SOURCE_LAYOUT == 0
    return (idx / BENCH_K) * A_SOURCE_STRIDE + idx % BENCH_K;
#else
    return (idx / BENCH_M) * A_SOURCE_STRIDE + idx % BENCH_M;
#endif
}

uint b_source_index(uint idx) {
#if B_SOURCE_LAYOUT == 0
    return (idx / BENCH_N) * B_SOURCE_STRIDE + idx % BENCH_N;
#else
    return (idx / BENCH_K) * B_SOURCE_STRIDE + idx % BENCH_K;
#endif
}

float16_t load_a_scalar(uint byte_off) {
#if A_SOURCE_TYPE == TYPE_F16
    const uint word = ABuffer.Load(byte_off & ~3u);
    return asfloat16((uint16_t)((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu));
#elif A_SOURCE_TYPE == TYPE_BF16
    const uint word = ABuffer.Load(byte_off & ~3u);
    const uint bits = ((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu) << 16;
    return (float16_t)asfloat(bits);
#else
    return (float16_t)asfloat(ABuffer.Load(byte_off));
#endif
}

float16_t load_b_scalar(uint byte_off) {
#if B_SOURCE_TYPE == TYPE_F16
    const uint word = BBuffer.Load(byte_off & ~3u);
    return asfloat16((uint16_t)((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu));
#elif B_SOURCE_TYPE == TYPE_BF16
    const uint word = BBuffer.Load(byte_off & ~3u);
    const uint bits = ((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu) << 16;
    return (float16_t)asfloat(bits);
#else
    return (float16_t)asfloat(BBuffer.Load(byte_off));
#endif
}

void load_a4(uint byte_off, out float16_t v0, out float16_t v1,
             out float16_t v2, out float16_t v3) {
#if A_SOURCE_TYPE == TYPE_F16
    const uint2 w = ABuffer.Load2(byte_off);
    v0 = asfloat16((uint16_t)(w.x & 0xFFFFu));
    v1 = asfloat16((uint16_t)(w.x >> 16));
    v2 = asfloat16((uint16_t)(w.y & 0xFFFFu));
    v3 = asfloat16((uint16_t)(w.y >> 16));
#elif A_SOURCE_TYPE == TYPE_BF16
    const uint2 w = ABuffer.Load2(byte_off);
    v0 = (float16_t)asfloat((w.x & 0xFFFFu) << 16);
    v1 = (float16_t)asfloat(w.x & 0xFFFF0000u);
    v2 = (float16_t)asfloat((w.y & 0xFFFFu) << 16);
    v3 = (float16_t)asfloat(w.y & 0xFFFF0000u);
#else
    const uint4 w = ABuffer.Load4(byte_off);
    v0 = (float16_t)asfloat(w.x);
    v1 = (float16_t)asfloat(w.y);
    v2 = (float16_t)asfloat(w.z);
    v3 = (float16_t)asfloat(w.w);
#endif
}

void load_b4(uint byte_off, out float16_t v0, out float16_t v1,
             out float16_t v2, out float16_t v3) {
#if B_SOURCE_TYPE == TYPE_F16
    const uint2 w = BBuffer.Load2(byte_off);
    v0 = asfloat16((uint16_t)(w.x & 0xFFFFu));
    v1 = asfloat16((uint16_t)(w.x >> 16));
    v2 = asfloat16((uint16_t)(w.y & 0xFFFFu));
    v3 = asfloat16((uint16_t)(w.y >> 16));
#elif B_SOURCE_TYPE == TYPE_BF16
    const uint2 w = BBuffer.Load2(byte_off);
    v0 = (float16_t)asfloat((w.x & 0xFFFFu) << 16);
    v1 = (float16_t)asfloat(w.x & 0xFFFF0000u);
    v2 = (float16_t)asfloat((w.y & 0xFFFFu) << 16);
    v3 = (float16_t)asfloat(w.y & 0xFFFF0000u);
#else
    const uint4 w = BBuffer.Load4(byte_off);
    v0 = (float16_t)asfloat(w.x);
    v1 = (float16_t)asfloat(w.y);
    v2 = (float16_t)asfloat(w.z);
    v3 = (float16_t)asfloat(w.w);
#endif
}

void load_a_vector(uint byte_off, out float16_t values[LOAD_VECTOR_WIDTH]) {
#if LOAD_VECTOR_WIDTH == 2
#if A_SOURCE_TYPE == TYPE_F16
    const uint w = ABuffer.Load(byte_off);
    values[0] = asfloat16((uint16_t)(w & 0xFFFFu));
    values[1] = asfloat16((uint16_t)(w >> 16));
#elif A_SOURCE_TYPE == TYPE_BF16
    const uint w = ABuffer.Load(byte_off);
    values[0] = (float16_t)asfloat((w & 0xFFFFu) << 16);
    values[1] = (float16_t)asfloat(w & 0xFFFF0000u);
#else
    const uint2 w = ABuffer.Load2(byte_off);
    values[0] = (float16_t)asfloat(w.x);
    values[1] = (float16_t)asfloat(w.y);
#endif
#elif LOAD_VECTOR_WIDTH == 4
    load_a4(byte_off, values[0], values[1], values[2], values[3]);
#else
    load_a4(byte_off, values[0], values[1], values[2], values[3]);
    load_a4(byte_off + 4u * A_SOURCE_BYTES,
            values[4], values[5], values[6], values[7]);
#endif
}

void load_b_vector(uint byte_off, out float16_t values[LOAD_VECTOR_WIDTH]) {
#if LOAD_VECTOR_WIDTH == 2
#if B_SOURCE_TYPE == TYPE_F16
    const uint w = BBuffer.Load(byte_off);
    values[0] = asfloat16((uint16_t)(w & 0xFFFFu));
    values[1] = asfloat16((uint16_t)(w >> 16));
#elif B_SOURCE_TYPE == TYPE_BF16
    const uint w = BBuffer.Load(byte_off);
    values[0] = (float16_t)asfloat((w & 0xFFFFu) << 16);
    values[1] = (float16_t)asfloat(w & 0xFFFF0000u);
#else
    const uint2 w = BBuffer.Load2(byte_off);
    values[0] = (float16_t)asfloat(w.x);
    values[1] = (float16_t)asfloat(w.y);
#endif
#elif LOAD_VECTOR_WIDTH == 4
    load_b4(byte_off, values[0], values[1], values[2], values[3]);
#else
    load_b4(byte_off, values[0], values[1], values[2], values[3]);
    load_b4(byte_off + 4u * B_SOURCE_BYTES,
            values[4], values[5], values[6], values[7]);
#endif
}

#if BENCH_SCOPE == 0

typedef Matrix<ComponentType::F16, BENCH_M, BENCH_K,
               MatrixUse::A, MatrixScope::Thread> ThreadMatA;

[WaveSize(WAVE_SIZE)]
[numthreads(GROUP_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint matrix_id = gid.x * GROUP_THREADS + tid;
    vector<float, BENCH_M> sum = (vector<float, BENCH_M>)0.0f;

    [loop] for (uint rep = 0; rep < INNER_REPEATS; ++rep) {
        [loop] for (uint ks = 0; ks < K_STEPS; ++ks) {
            const uint a_off = A_BASE_OFFSET +
                (matrix_id * K_STEPS + ks) * A_MATRIX_ELEMS * 2u;
            const uint b_off = B_BASE_OFFSET +
                (matrix_id * K_STEPS + ks) * BENCH_K * 2u;
#if A_LAYOUT == 0
            ThreadMatA a = ThreadMatA::Load<MatrixLayout::RowMajor>(
                ABuffer, a_off, BENCH_K * 2u, DESCRIPTOR_ALIGN);
#else
            ThreadMatA a = ThreadMatA::Load<MatrixLayout::ColMajor>(
                ABuffer, a_off, BENCH_M * 2u, DESCRIPTOR_ALIGN);
#endif
            vector<float16_t, BENCH_K> b;
            [unroll] for (uint k = 0; k < BENCH_K; ++k) {
                const uint word = BBuffer.Load((b_off + k * 2u) & ~3u);
                b[k] = asfloat16((uint16_t)(
                    (word >> (((b_off + k * 2u) & 2u) * 8u)) & 0xFFFFu));
            }
            sum += Multiply<float, float16_t, BENCH_M, BENCH_K,
                            ComponentType::F16>(a, b);
        }
    }

    const uint c_off = C_BASE_OFFSET + matrix_id * BENCH_M * 4u;
    [unroll] for (uint r = 0; r < BENCH_M; ++r) {
        CBuffer.Store(c_off + r * 4u, asuint(sum[r]));
    }
}

#else

#if BENCH_SCOPE == 1
#define MATRIX_SCOPE MatrixScope::Wave
#define MATRIX_COUNT_PER_GROUP NUM_WAVES
#define LOCAL_COUNT WAVE_SIZE
#else
#define MATRIX_SCOPE MatrixScope::ThreadGroup
#define MATRIX_COUNT_PER_GROUP 1
#define LOCAL_COUNT GROUP_THREADS
#endif

typedef Matrix<ComponentType::F16, BENCH_M, BENCH_K,
               MatrixUse::A, MATRIX_SCOPE> MatA;
typedef Matrix<ComponentType::F16, BENCH_K, BENCH_N,
               MatrixUse::B, MATRIX_SCOPE> MatB;
typedef Matrix<ComponentType::F32, BENCH_M, BENCH_N,
               MatrixUse::Accumulator, MATRIX_SCOPE> MatAcc;
#if B_LOAD == LOAD_DIRECT_TRANSPOSE_CAST
typedef Matrix<ComponentType::F16, BENCH_N, BENCH_K,
               MatrixUse::A, MATRIX_SCOPE> MatBSource;
#endif

#if ACC_TILES > 1

#if EPILOGUE == 1
#error "Multiple accumulator tiles do not support the LDS epilogue"
#endif
#if BENCH_SCOPE == 1
#if A_LOAD != LOAD_DIRECT
#error "Wave multi-accumulator cases require direct A loads"
#endif
#if B_LOAD != LOAD_DIRECT && B_LOAD != LOAD_DIRECT_TRANSPOSE_CAST
#error "Wave multi-accumulator cases require direct B loads"
#endif
#else
#if A_LOAD != LOAD_LDS_VECTOR || B_LOAD != LOAD_LDS_VECTOR
#error "Threadgroup multi-accumulator cases require vector LDS staging"
#endif
groupshared float16_t multi_tile_a[A_LDS_MATRIX_ELEMS];
groupshared float16_t multi_tile_b[B_LDS_MATRIX_ELEMS];
#endif

[WaveSize(WAVE_SIZE)]
[numthreads(GROUP_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
#if BENCH_SCOPE == 1
    const uint owner = tid / WAVE_SIZE;
#else
    const uint owner = 0;
#endif
    MatAcc acc[ACC_TILES];
    [unroll] for (uint tile = 0; tile < ACC_TILES; ++tile) {
        acc[tile] = MatAcc::Splat(0.0f);
    }

    [loop] for (uint rep = 0; rep < INNER_REPEATS; ++rep) {
        [loop] for (uint ks = 0; ks < K_STEPS; ++ks) {
            [unroll] for (uint tile = 0; tile < ACC_TILES; ++tile) {
#if TILE_ORDER == 0
                const uint matrix_id =
                    (gid.x * MATRIX_COUNT_PER_GROUP + owner) * ACC_TILES + tile;
#else
                const uint matrix_id =
                    gid.x * MATRIX_COUNT_PER_GROUP * ACC_TILES +
                    tile * MATRIX_COUNT_PER_GROUP + owner;
#endif
                const uint a_src = A_BASE_OFFSET +
                    (matrix_id * K_STEPS + ks) *
                    A_SOURCE_PITCH_BYTES;
                const uint b_src = B_BASE_OFFSET +
                    (matrix_id * K_STEPS + ks) *
#if B_LOAD == LOAD_DIRECT_TRANSPOSE_CAST
                    ALIGN_128(BENCH_N * (BENCH_K + B_SOURCE_PAD) *
                              B_SOURCE_BYTES);
#else
                    B_SOURCE_PITCH_BYTES;
#endif
#if BENCH_SCOPE == 2
                for (uint idx = tid * LOAD_VECTOR_WIDTH;
                     idx < A_MATRIX_ELEMS;
                     idx += GROUP_THREADS * LOAD_VECTOR_WIDTH) {
                    float16_t values[LOAD_VECTOR_WIDTH];
                    load_a_vector(
                        a_src + a_source_index(idx) * A_SOURCE_BYTES, values);
                    [unroll] for (uint j = 0; j < LOAD_VECTOR_WIDTH; ++j) {
                        const uint src_idx = idx + j;
                        const uint r = A_SOURCE_LAYOUT == 0
                            ? src_idx / BENCH_K : src_idx % BENCH_M;
                        const uint k = A_SOURCE_LAYOUT == 0
                            ? src_idx % BENCH_K : src_idx / BENCH_M;
                        const uint dst = A_LAYOUT == 0
                            ? r * A_LDS_STRIDE + k : k * A_LDS_STRIDE + r;
                        multi_tile_a[dst] = values[j];
                    }
                }
                for (uint idx = tid * LOAD_VECTOR_WIDTH;
                     idx < B_MATRIX_ELEMS;
                     idx += GROUP_THREADS * LOAD_VECTOR_WIDTH) {
                    float16_t values[LOAD_VECTOR_WIDTH];
                    load_b_vector(
                        b_src + b_source_index(idx) * B_SOURCE_BYTES, values);
                    [unroll] for (uint j = 0; j < LOAD_VECTOR_WIDTH; ++j) {
                        const uint src_idx = idx + j;
                        const uint k = B_SOURCE_LAYOUT == 0
                            ? src_idx / BENCH_N : src_idx % BENCH_K;
                        const uint c = B_SOURCE_LAYOUT == 0
                            ? src_idx % BENCH_N : src_idx / BENCH_K;
                        const uint dst = B_LAYOUT == 0
                            ? k * B_LDS_STRIDE + c : c * B_LDS_STRIDE + k;
                        multi_tile_b[dst] = values[j];
                    }
                }
                GroupMemoryBarrierWithGroupSync();
                MatA a = MatA::Load(
                    multi_tile_a, 0, A_LDS_STRIDE, A_LAYOUT_ENUM);
                MatB b = MatB::Load(
                    multi_tile_b, 0, B_LDS_STRIDE, B_LAYOUT_ENUM);
                acc[tile].MultiplyAccumulate(a, b);
                GroupMemoryBarrierWithGroupSync();
#else
                MatA a = MatA::Load(
                    ABuffer, a_src, A_STRIDE_BYTES,
                    A_LAYOUT_ENUM, DESCRIPTOR_ALIGN);
                MatB b;
#if B_LOAD == LOAD_DIRECT
                b = MatB::Load(
                    BBuffer, b_src, B_STRIDE_BYTES,
                    B_LAYOUT_ENUM, DESCRIPTOR_ALIGN);
#else
                MatBSource bt = MatBSource::Load(
                    BBuffer, b_src, (BENCH_K + B_SOURCE_PAD) * 2u,
                    MatrixLayout::RowMajor, DESCRIPTOR_ALIGN);
                b = bt.Cast<ComponentType::F16, MatrixUse::B, true>();
#endif
                acc[tile].MultiplyAccumulate(a, b);
#endif
            }
        }
    }

    [unroll] for (uint tile = 0; tile < ACC_TILES; ++tile) {
#if TILE_ORDER == 0
        const uint matrix_id =
            (gid.x * MATRIX_COUNT_PER_GROUP + owner) * ACC_TILES + tile;
#else
        const uint matrix_id =
            gid.x * MATRIX_COUNT_PER_GROUP * ACC_TILES +
            tile * MATRIX_COUNT_PER_GROUP + owner;
#endif
        const uint c_off = C_BASE_OFFSET + matrix_id * C_MATRIX_ELEMS * 4u;
#if EPILOGUE == 0
        acc[tile].Store(
            CBuffer, c_off, C_STRIDE_BYTES,
            C_LAYOUT_ENUM, DESCRIPTOR_ALIGN);
#else
        [loop] for (uint e = 0; e < acc[tile].Length(); ++e) {
            const uint2 rc = acc[tile].GetCoordinate(e);
            const uint dst = C_LAYOUT == 0
                ? rc.x * BENCH_N + rc.y : rc.y * BENCH_M + rc.x;
            CBuffer.Store(c_off + dst * 4u, asuint(acc[tile].Get(e)));
        }
#endif
    }
}

#else

#if A_LOAD != LOAD_DIRECT
#if A_LOAD == LOAD_LDS_PREFETCH
#define A_LDS_SLOTS 2
#else
#define A_LDS_SLOTS 1
#endif
groupshared float16_t tile_a[MATRIX_COUNT_PER_GROUP * A_LDS_SLOTS * A_LDS_MATRIX_ELEMS];
#endif
#if B_LOAD != LOAD_DIRECT && B_LOAD != LOAD_DIRECT_TRANSPOSE_CAST
#if B_LOAD == LOAD_LDS_PREFETCH
#define B_LDS_SLOTS 2
#else
#define B_LDS_SLOTS 1
#endif
groupshared float16_t tile_b[MATRIX_COUNT_PER_GROUP * B_LDS_SLOTS * B_LDS_MATRIX_ELEMS];
#endif
#if EPILOGUE == 1
groupshared float tile_c[MATRIX_COUNT_PER_GROUP * C_MATRIX_ELEMS];
#endif

#if A_LOAD == LOAD_LDS_PREFETCH
#if (A_MATRIX_ELEMS % LOCAL_COUNT) != 0
#error "A prefetch requires matrix elements divisible by participants"
#endif
#define A_EPT (A_MATRIX_ELEMS / LOCAL_COUNT)
#if (A_EPT % LOAD_VECTOR_WIDTH) != 0
#error "A prefetch elements per participant must be divisible by vector width"
#endif
#endif
#if B_LOAD == LOAD_LDS_PREFETCH
#if (B_MATRIX_ELEMS % LOCAL_COUNT) != 0
#error "B prefetch requires matrix elements divisible by participants"
#endif
#define B_EPT (B_MATRIX_ELEMS / LOCAL_COUNT)
#if (B_EPT % LOAD_VECTOR_WIDTH) != 0
#error "B prefetch elements per participant must be divisible by vector width"
#endif
#endif

[WaveSize(WAVE_SIZE)]
[numthreads(GROUP_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
#if BENCH_SCOPE == 1
    const uint owner = tid / WAVE_SIZE;
    const uint local = tid % WAVE_SIZE;
#else
    const uint owner = 0;
    const uint local = tid;
#endif
    const uint matrix_id = gid.x * MATRIX_COUNT_PER_GROUP + owner;
    const uint a_matrix_bytes = A_SOURCE_PITCH_BYTES;
    const uint b_matrix_bytes =
#if B_LOAD == LOAD_DIRECT_TRANSPOSE_CAST
        ALIGN_128(BENCH_N * (BENCH_K + B_SOURCE_PAD) * B_SOURCE_BYTES);
#else
        B_SOURCE_PITCH_BYTES;
#endif

    MatAcc acc = MatAcc::Splat(0.0f);

#if A_LOAD == LOAD_LDS_PREFETCH
    float16_t ra[A_EPT];
#endif
#if B_LOAD == LOAD_LDS_PREFETCH
    float16_t rb[B_EPT];
#endif

    [loop] for (uint rep = 0; rep < INNER_REPEATS; ++rep) {
#if A_LOAD == LOAD_LDS_PREFETCH
        {
            const uint src = A_BASE_OFFSET +
                (matrix_id * K_STEPS) * a_matrix_bytes;
            [unroll] for (uint e = 0; e < A_EPT; e += LOAD_VECTOR_WIDTH) {
                float16_t values[LOAD_VECTOR_WIDTH];
                const uint idx = local * A_EPT + e;
                load_a_vector(src + a_source_index(idx) * A_SOURCE_BYTES,
                              values);
                [unroll] for (uint j = 0; j < LOAD_VECTOR_WIDTH; ++j) {
                    ra[e + j] = values[j];
                }
            }
        }
#endif
#if B_LOAD == LOAD_LDS_PREFETCH
        {
            const uint src = B_BASE_OFFSET +
                (matrix_id * K_STEPS) * b_matrix_bytes;
            [unroll] for (uint e = 0; e < B_EPT; e += LOAD_VECTOR_WIDTH) {
                float16_t values[LOAD_VECTOR_WIDTH];
                const uint idx = local * B_EPT + e;
                load_b_vector(src + b_source_index(idx) * B_SOURCE_BYTES,
                              values);
                [unroll] for (uint j = 0; j < LOAD_VECTOR_WIDTH; ++j) {
                    rb[e + j] = values[j];
                }
            }
        }
#endif

        [loop] for (uint ks = 0; ks < K_STEPS; ++ks) {
            const uint a_src = A_BASE_OFFSET +
                (matrix_id * K_STEPS + ks) * a_matrix_bytes;
            const uint b_src = B_BASE_OFFSET +
                (matrix_id * K_STEPS + ks) * b_matrix_bytes;
            const uint a_slot =
#if A_LOAD == LOAD_LDS_PREFETCH
                ks & 1u;
#else
                0u;
#endif
            const uint b_slot =
#if B_LOAD == LOAD_LDS_PREFETCH
                ks & 1u;
#else
                0u;
#endif

#if A_LOAD == LOAD_LDS_SCALAR
            for (uint idx = local; idx < A_MATRIX_ELEMS; idx += LOCAL_COUNT) {
                const uint r = A_SOURCE_LAYOUT == 0 ? idx / BENCH_K : idx % BENCH_M;
                const uint k = A_SOURCE_LAYOUT == 0 ? idx % BENCH_K : idx / BENCH_M;
                const uint dst = A_LAYOUT == 0
                    ? r * A_LDS_STRIDE + k : k * A_LDS_STRIDE + r;
                tile_a[owner * A_LDS_SLOTS * A_LDS_MATRIX_ELEMS + dst] =
                    load_a_scalar(a_src + a_source_index(idx) * A_SOURCE_BYTES);
            }
#elif A_LOAD == LOAD_LDS_VECTOR
            for (uint idx = local * LOAD_VECTOR_WIDTH; idx < A_MATRIX_ELEMS;
                 idx += LOCAL_COUNT * LOAD_VECTOR_WIDTH) {
                float16_t values[LOAD_VECTOR_WIDTH];
                load_a_vector(
                    a_src + a_source_index(idx) * A_SOURCE_BYTES, values);
                [unroll] for (uint j = 0; j < LOAD_VECTOR_WIDTH; ++j) {
                    const uint src_idx = idx + j;
                    const uint r = A_SOURCE_LAYOUT == 0
                        ? src_idx / BENCH_K : src_idx % BENCH_M;
                    const uint k = A_SOURCE_LAYOUT == 0
                        ? src_idx % BENCH_K : src_idx / BENCH_M;
                    const uint dst = A_LAYOUT == 0
                        ? r * A_LDS_STRIDE + k : k * A_LDS_STRIDE + r;
                    tile_a[owner * A_LDS_SLOTS * A_LDS_MATRIX_ELEMS + dst] =
                        values[j];
                }
            }
#elif A_LOAD == LOAD_LDS_PREFETCH
            {
                const uint base =
                    (owner * A_LDS_SLOTS + a_slot) * A_LDS_MATRIX_ELEMS;
                [unroll] for (uint e = 0; e < A_EPT; ++e) {
                    const uint idx = local * A_EPT + e;
                    const uint r = A_SOURCE_LAYOUT == 0
                        ? idx / BENCH_K : idx % BENCH_M;
                    const uint k = A_SOURCE_LAYOUT == 0
                        ? idx % BENCH_K : idx / BENCH_M;
                    const uint dst = A_LAYOUT == 0
                        ? r * A_LDS_STRIDE + k : k * A_LDS_STRIDE + r;
                    tile_a[base + dst] = ra[e];
                }
                if (ks + 1u < K_STEPS) {
                    const uint next = a_src + a_matrix_bytes;
                    [unroll] for (uint e = 0; e < A_EPT;
                              e += LOAD_VECTOR_WIDTH) {
                        float16_t values[LOAD_VECTOR_WIDTH];
                        const uint idx = local * A_EPT + e;
                        load_a_vector(
                            next + a_source_index(idx) * A_SOURCE_BYTES,
                            values);
                        [unroll] for (uint j = 0; j < LOAD_VECTOR_WIDTH; ++j) {
                            ra[e + j] = values[j];
                        }
                    }
                }
            }
#endif

#if B_LOAD == LOAD_LDS_SCALAR
            for (uint idx = local; idx < B_MATRIX_ELEMS; idx += LOCAL_COUNT) {
                const uint k = B_SOURCE_LAYOUT == 0 ? idx / BENCH_N : idx % BENCH_K;
                const uint c = B_SOURCE_LAYOUT == 0 ? idx % BENCH_N : idx / BENCH_K;
                const uint dst = B_LAYOUT == 0
                    ? k * B_LDS_STRIDE + c : c * B_LDS_STRIDE + k;
                tile_b[owner * B_LDS_SLOTS * B_LDS_MATRIX_ELEMS + dst] =
                    load_b_scalar(b_src + b_source_index(idx) * B_SOURCE_BYTES);
            }
#elif B_LOAD == LOAD_LDS_VECTOR
            for (uint idx = local * LOAD_VECTOR_WIDTH; idx < B_MATRIX_ELEMS;
                 idx += LOCAL_COUNT * LOAD_VECTOR_WIDTH) {
                float16_t values[LOAD_VECTOR_WIDTH];
                load_b_vector(
                    b_src + b_source_index(idx) * B_SOURCE_BYTES, values);
                [unroll] for (uint j = 0; j < LOAD_VECTOR_WIDTH; ++j) {
                    const uint src_idx = idx + j;
                    const uint k = B_SOURCE_LAYOUT == 0
                        ? src_idx / BENCH_N : src_idx % BENCH_K;
                    const uint c = B_SOURCE_LAYOUT == 0
                        ? src_idx % BENCH_N : src_idx / BENCH_K;
                    const uint dst = B_LAYOUT == 0
                        ? k * B_LDS_STRIDE + c : c * B_LDS_STRIDE + k;
                    tile_b[owner * B_LDS_SLOTS * B_LDS_MATRIX_ELEMS + dst] =
                        values[j];
                }
            }
#elif B_LOAD == LOAD_LDS_PREFETCH
            {
                const uint base =
                    (owner * B_LDS_SLOTS + b_slot) * B_LDS_MATRIX_ELEMS;
                [unroll] for (uint e = 0; e < B_EPT; ++e) {
                    const uint idx = local * B_EPT + e;
                    const uint k = B_SOURCE_LAYOUT == 0
                        ? idx / BENCH_N : idx % BENCH_K;
                    const uint c = B_SOURCE_LAYOUT == 0
                        ? idx % BENCH_N : idx / BENCH_K;
                    const uint dst = B_LAYOUT == 0
                        ? k * B_LDS_STRIDE + c : c * B_LDS_STRIDE + k;
                    tile_b[base + dst] = rb[e];
                }
                if (ks + 1u < K_STEPS) {
                    const uint next = b_src + b_matrix_bytes;
                    [unroll] for (uint e = 0; e < B_EPT;
                              e += LOAD_VECTOR_WIDTH) {
                        float16_t values[LOAD_VECTOR_WIDTH];
                        const uint idx = local * B_EPT + e;
                        load_b_vector(
                            next + b_source_index(idx) * B_SOURCE_BYTES,
                            values);
                        [unroll] for (uint j = 0; j < LOAD_VECTOR_WIDTH; ++j) {
                            rb[e + j] = values[j];
                        }
                    }
                }
            }
#endif

#if A_LOAD != LOAD_DIRECT || (B_LOAD != LOAD_DIRECT && B_LOAD != LOAD_DIRECT_TRANSPOSE_CAST)
            GroupMemoryBarrierWithGroupSync();
#endif

            MatA a;
#if A_LOAD == LOAD_DIRECT
            a = MatA::Load(ABuffer, a_src, A_STRIDE_BYTES,
                           A_LAYOUT_ENUM, DESCRIPTOR_ALIGN);
#else
            a = MatA::Load(tile_a,
                (owner * A_LDS_SLOTS + a_slot) * A_LDS_MATRIX_ELEMS,
                A_LDS_STRIDE, A_LAYOUT_ENUM);
#endif

            MatB b;
#if B_LOAD == LOAD_DIRECT
            b = MatB::Load(BBuffer, b_src, B_STRIDE_BYTES,
                           B_LAYOUT_ENUM, DESCRIPTOR_ALIGN);
#elif B_LOAD == LOAD_DIRECT_TRANSPOSE_CAST
            MatBSource bt = MatBSource::Load(
                                             BBuffer, b_src,
                                             (BENCH_K + B_SOURCE_PAD) * 2u,
                                             MatrixLayout::RowMajor,
                                             DESCRIPTOR_ALIGN);
            b = bt.Cast<ComponentType::F16, MatrixUse::B, true>();
#else
            b = MatB::Load(tile_b,
                (owner * B_LDS_SLOTS + b_slot) * B_LDS_MATRIX_ELEMS,
                B_LDS_STRIDE, B_LAYOUT_ENUM);
#endif
            acc.MultiplyAccumulate(a, b);
        }
    }

    const uint c_off = C_BASE_OFFSET + matrix_id * C_MATRIX_ELEMS * 4u;
#if EPILOGUE == 0
    acc.Store(CBuffer, c_off, C_STRIDE_BYTES, C_LAYOUT_ENUM, DESCRIPTOR_ALIGN);
#elif EPILOGUE == 1
    const uint c_base = owner * C_MATRIX_ELEMS;
    acc.Store(tile_c, c_base,
              C_LAYOUT == 0 ? BENCH_N : BENCH_M, C_LAYOUT_ENUM);
    GroupMemoryBarrierWithGroupSync();
    for (uint idx = local; idx < C_MATRIX_ELEMS; idx += LOCAL_COUNT) {
        CBuffer.Store(c_off + idx * 4u, asuint(tile_c[c_base + idx]));
    }
#else
    [loop] for (uint e = 0; e < acc.Length(); ++e) {
        const uint2 rc = acc.GetCoordinate(e);
        const uint dst = C_LAYOUT == 0 ? rc.x * BENCH_N + rc.y
                                       : rc.y * BENCH_M + rc.x;
        CBuffer.Store(c_off + dst * 4u, asuint(acc.Get(e)));
    }
#endif
}

#endif
#endif
