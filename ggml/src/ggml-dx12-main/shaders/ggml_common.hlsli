// ggml_common.hlsli - Common definitions for all ggml DX12 shaders
//
// Root signature layout:
//   b0: Root constants (dx12_shader_params)
//   t0: src0 ByteAddressBuffer
//   t1: src1 ByteAddressBuffer
//   u0: dst  RWByteAddressBuffer
//   t2: src2 ByteAddressBuffer (optional)

// Shader parameter block - must match dx12_shader_params in ggml-dx12.cpp
cbuffer ShaderParams : register(b0) {
    uint ne00, ne01, ne02, ne03;
    uint nb00, nb01, nb02, nb03;
    uint ne10, ne11, ne12, ne13;
    uint nb10, nb11, nb12, nb13;
    uint ne0, ne1, ne2, ne3;
    uint nb0, nb1, nb2, nb3;
    uint src0_offset;
    uint src1_offset;
    uint dst_offset;
    // Element sizes in bytes (2=F16, 4=F32) — NOT inferred from stride
    uint src0_esize;
    uint src1_esize;
    uint dst_esize;
    // op_params as individual scalars to avoid HLSL array packing (16-byte per element)
    uint op0, op1, op2, op3, op4, op5, op6, op7;
    uint op8, op9, op10, op11, op12, op13, op14, op15;
};

ByteAddressBuffer   src0 : register(t0);
ByteAddressBuffer   src1 : register(t1);
RWByteAddressBuffer dst  : register(u0);
ByteAddressBuffer   src2 : register(t2);
ByteAddressBuffer   src3 : register(t3);
RWByteAddressBuffer temp : register(u1);  // auxiliary temp buffer (split-KV flash attention)
ByteAddressBuffer   src4 : register(t4);  // optional, GATED_DELTA_NET / SSM_SCAN
ByteAddressBuffer   src5 : register(t5);  // optional, GATED_DELTA_NET / SSM_SCAN
ByteAddressBuffer   src6 : register(t6);  // optional, SSM_SCAN ids

// Wave size: compile-time constant when defined via -D WAVE_SIZE=N,
// otherwise falls back to runtime WaveGetLaneCount().
// Compile-time constant enables: division→shift, branch elimination,
// loop unrolling, and dead code removal in wave reduction patterns.
#ifdef WAVE_SIZE
#define WARP_SIZE WAVE_SIZE
#else
#define WARP_SIZE WaveGetLaneCount()
#endif

// Pin the wave/subgroup size at PSO creation when supported (cs_6_6+).
// The DX12 runtime selects the matching shader blob (w16/w32/w64) for
// the device's preferred wave size, so WAVE_SIZE always equals a wave
// size the GPU actually supports. Without this attribute, the driver
// is free to pick any supported wave size for the dispatch — Intel Xe
// drivers have been observed to pick wave=32 for shaders we tuned for
// wave=16, regressing perf. Each annotated shader emits this guarded
// by `WAVE_SIZE <= GROUP_SIZE` because [WaveSize(N)] requires the
// threadgroup to contain at least one full wave.
//
// Vulkan parity: matches VK_EXT_subgroup_size_control with
// requiredSubgroupSize set per-pipeline (Intel-targeted in
// ggml-vulkan.cpp).
#ifdef WAVE_SIZE
#define WAVE_SIZE_ATTR [WaveSize(WAVE_SIZE)]
#else
#define WAVE_SIZE_ATTR
#endif

// Helper: load float from ByteAddressBuffer at byte offset
float load_f32(ByteAddressBuffer buf, uint byte_offset) {
    return asfloat(buf.Load(byte_offset));
}

// Helper: store float to RWByteAddressBuffer at byte offset
void store_f32(RWByteAddressBuffer buf, uint byte_offset, float val) {
    buf.Store(byte_offset, asuint(val));
}

// Helper: load float from RWByteAddressBuffer at byte offset
float load_f32_rw(RWByteAddressBuffer buf, uint byte_offset) {
    return asfloat(buf.Load(byte_offset));
}

// Helper: compute flat thread index for 2D dispatch overflow.
// When total thread groups > 65535, dispatch splits across group_id.y.
// This reconstructs the flat index from 2D group coords + local thread ID.
uint flat_idx_2d(uint3 group_id, uint local_id) {
    return (group_id.y * 65535u + group_id.x) * 256u + local_id;
}

// Matvec row-group index. Supports 2D dispatches that exceed the D3D12
// per-dimension group limit; chunked matvecs keep op15 at zero and advance
// buffer offsets so shaders still see a local row range.
uint group_x_2d(uint3 group_id) {
    return op15 + group_id.y * 65535u + group_id.x;
}

// Helper: convert flat index to 4D indices
void flat_to_4d(uint flat_idx, uint d0, uint d1, uint d2,
                out uint i0, out uint i1, out uint i2, out uint i3) {
    i3 = flat_idx / (d0 * d1 * d2);
    uint rem = flat_idx % (d0 * d1 * d2);
    i2 = rem / (d0 * d1);
    rem = rem % (d0 * d1);
    i1 = rem / d0;
    i0 = rem % d0;
}

// Helper: compute byte offset for a 4D tensor element
uint offset_4d(uint i0, uint i1, uint i2, uint i3,
               uint nb0_, uint nb1_, uint nb2_, uint nb3_, uint base) {
    return base + i0 * nb0_ + i1 * nb1_ + i2 * nb2_ + i3 * nb3_;
}

// Helper: convert float16 bits to float32
float f16_to_f32(uint h) {
    return f16tof32(h);
}

// Helper: convert float32 to float16 bits, matching ggml_compute_fp32_to_fp16.
uint f32_to_f16(float f) {
    float base = (abs(f) * asfloat(0x77800000u)) * asfloat(0x08800000u);

    const uint w = asuint(f);
    const uint shl1_w = w + w;
    const uint sign = w & 0x80000000u;
    uint bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) {
        bias = 0x71000000u;
    }

    base = asfloat((bias >> 1) + 0x07800000u) + base;
    const uint bits = asuint(base);
    const uint exp_bits = (bits >> 13) & 0x00007C00u;
    const uint mantissa_bits = bits & 0x00000FFFu;
    const uint nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign);
}

uint f32_to_bf16(float f) {
    uint bits = asuint(f);
    uint nan = (bits & 0x7FFFFFFFu) > 0x7F800000u;
    uint round = 0x7FFFu + ((bits >> 16) & 1u);
    return nan ? ((bits >> 16) | 64u) : ((bits + round) >> 16);
}

// Helper: load a float value, auto-detecting format from element stride
// elem_stride: 2=F16, 3=BF16 (sentinel; physical stride is 2), 4=F32
float load_auto(ByteAddressBuffer buf, uint byte_offset, uint elem_stride) {
    if (elem_stride == 2) {
        // F16
        uint word = buf.Load(byte_offset & ~3u);
        uint shift = (byte_offset & 2u) * 8u;
        return f16_to_f32((word >> shift) & 0xFFFFu);
    }
    if (elem_stride == 3) {
        // BF16: upper 16 bits of F32
        uint word = buf.Load(byte_offset & ~3u);
        uint shift = (byte_offset & 2u) * 8u;
        uint bf16 = (word >> shift) & 0xFFFFu;
        return asfloat(bf16 << 16);
    }
    return asfloat(buf.Load(byte_offset));
}

// BF16 variant — only used by BF16-specific dispatch paths
float load_auto_bf16(ByteAddressBuffer buf, uint byte_offset, uint elem_stride) {
    if (elem_stride == 3) {
        // BF16: upper 16 bits of F32
        uint word = buf.Load(byte_offset & ~3u);
        uint shift = (byte_offset & 2u) * 8u;
        uint bf16 = (word >> shift) & 0xFFFFu;
        return asfloat(bf16 << 16);
    }
    return load_auto(buf, byte_offset, elem_stride);
}

// Same for RWByteAddressBuffer
float load_auto_rw(RWByteAddressBuffer buf, uint byte_offset, uint elem_stride) {
    if (elem_stride == 2) {
        uint word = buf.Load(byte_offset & ~3u);
        uint shift = (byte_offset & 2u) * 8u;
        return f16_to_f32((word >> shift) & 0xFFFFu);
    }
    if (elem_stride == 3) {
        // BF16: upper 16 bits of F32
        uint word = buf.Load(byte_offset & ~3u);
        uint shift = (byte_offset & 2u) * 8u;
        uint bf16 = (word >> shift) & 0xFFFFu;
        return asfloat(bf16 << 16);
    }
    return asfloat(buf.Load(byte_offset));
}

float load_auto_rw_bf16(RWByteAddressBuffer buf, uint byte_offset, uint elem_stride) {
    if (elem_stride == 3) {
        uint word = buf.Load(byte_offset & ~3u);
        uint shift = (byte_offset & 2u) * 8u;
        uint bf16 = (word >> shift) & 0xFFFFu;
        return asfloat(bf16 << 16);
    }
    return load_auto_rw(buf, byte_offset, elem_stride);
}

// Helper: store a float value, auto-detecting F16/BF16/F32 destination
void store_auto(RWByteAddressBuffer buf, uint byte_offset, float val, uint elem_stride) {
    if (elem_stride == 2 || elem_stride == 3) {
        // F16 (esize=2) or BF16 (esize=3 sentinel; physical stride 2)
        uint half_val = (elem_stride == 3) ? f32_to_bf16(val) : f32_to_f16(val);
        uint word_addr = byte_offset & ~3u;
        uint shift = (byte_offset & 2u) * 8u;
        uint mask = (uint)0xFFFFu << shift;
        uint new_bits = (half_val & (uint)0xFFFFu) << shift;

        uint expected, original;
        [allow_uav_condition] do {
            buf.InterlockedOr(word_addr, 0, expected);
            uint desired = (expected & ~mask) | new_bits;
            buf.InterlockedCompareExchange(word_addr, expected, desired, original);
        } while (original != expected);
    } else {
        buf.Store(byte_offset, asuint(val));
    }
}

// Paired F16 store: write two F16 values as a single 32-bit word (no atomics needed)
// byte_offset must be 4-byte aligned
void store_f16_pair(RWByteAddressBuffer buf, uint byte_offset, float val0, float val1) {
    uint lo = f32_to_f16(val0) & 0xFFFFu;
    uint hi = f32_to_f16(val1) & 0xFFFFu;
    buf.Store(byte_offset, lo | (hi << 16u));
}

// Paired BF16 store: write two BF16 values as a single 32-bit word
void store_bf16_pair(RWByteAddressBuffer buf, uint byte_offset, float val0, float val1) {
    uint lo = f32_to_bf16(val0) & 0xFFFFu;
    uint hi = f32_to_bf16(val1) & 0xFFFFu;
    buf.Store(byte_offset, lo | (hi << 16u));
}

// Check if paired F16 output mode is available (contiguous, even, aligned)
bool can_pair_f16() {
    return dst_esize == 2 && nb0 == 2 && (ne0 & 1) == 0 &&
           (dst_offset & 3) == 0 && (nb1 & 3) == 0;
}

float load_fused_bias(uint row, uint i2, uint i3) {
    if (op0 != 1u) {
        return 0.0f;
    }
    uint b2 = (op5 > 1u) ? i2 : 0u;
    uint b3 = (op6 > 1u) ? i3 : 0u;
    return asfloat(src2.Load(op1 + row * op2 + b2 * op3 + b3 * op4));
}

// BF16 store variant — only used by BF16-specific dispatch paths
void store_auto_bf16(RWByteAddressBuffer buf, uint byte_offset, float val, uint elem_stride) {
    if (elem_stride == 3) {
        uint bf16_val = f32_to_bf16(val);
        uint word_addr = byte_offset & ~3u;
        uint shift = (byte_offset & 2u) * 8u;
        uint mask = (uint)0xFFFFu << shift;
        uint new_bits = (bf16_val & (uint)0xFFFFu) << shift;

        uint expected, original;
        [allow_uav_condition] do {
            buf.InterlockedOr(word_addr, 0, expected);
            uint desired = (expected & ~mask) | new_bits;
            buf.InterlockedCompareExchange(word_addr, expected, desired, original);
        } while (original != expected);
    } else {
        store_auto(buf, byte_offset, val, elem_stride);
    }
}

// op_params accessors — individual scalars to match cbuffer packing
uint op_param_uint(uint idx) {
    switch(idx) {
        case 0: return op0; case 1: return op1;
        case 2: return op2; case 3: return op3;
        case 4: return op4; case 5: return op5;
        case 6: return op6; case 7: return op7;
        case 8: return op8; case 9: return op9;
        case 10: return op10; case 11: return op11;
        case 12: return op12; case 13: return op13;
        case 14: return op14; case 15: return op15;
        default: return 0;
    }
}

float op_param_f32(uint idx) {
    return asfloat(op_param_uint(idx));
}
