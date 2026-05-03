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

// Helper: convert float32 to float16 bits
uint f32_to_f16(float f) {
    return f32tof16(f);
}

// Helper: load a float value, auto-detecting F16 (stride=2) vs F32 (stride=4)
// Uses the element stride (nb0) to distinguish format
float load_auto(ByteAddressBuffer buf, uint byte_offset, uint elem_stride) {
    if (elem_stride == 2) {
        // F16: load containing 32-bit word, extract the right 16-bit half
        uint word = buf.Load(byte_offset & ~3u);
        uint shift = (byte_offset & 2u) * 8u;
        return f16_to_f32((word >> shift) & 0xFFFFu);
    } else {
        return asfloat(buf.Load(byte_offset));
    }
}

// Same for RWByteAddressBuffer
float load_auto_rw(RWByteAddressBuffer buf, uint byte_offset, uint elem_stride) {
    if (elem_stride == 2) {
        uint word = buf.Load(byte_offset & ~3u);
        uint shift = (byte_offset & 2u) * 8u;
        return f16_to_f32((word >> shift) & 0xFFFFu);
    } else {
        return asfloat(buf.Load(byte_offset));
    }
}

// Helper: store a float value, auto-detecting F16 vs F32 destination
void store_auto(RWByteAddressBuffer buf, uint byte_offset, float val, uint elem_stride) {
    if (elem_stride == 2) {
        // F16 destination: use atomic CAS to avoid races on shared 32-bit words
        uint f16_val = f32_to_f16(val);
        uint word_addr = byte_offset & ~3u;
        uint shift = (byte_offset & 2u) * 8u;
        uint mask = (uint)0xFFFFu << shift;
        uint new_bits = (f16_val & (uint)0xFFFFu) << shift;

        uint expected, original;
        [allow_uav_condition] do {
            buf.InterlockedOr(word_addr, 0, expected); // atomic read
            uint desired = (expected & ~mask) | new_bits;
            buf.InterlockedCompareExchange(word_addr, expected, desired, original);
        } while (original != expected);
    } else {
        buf.Store(byte_offset, asuint(val));
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
