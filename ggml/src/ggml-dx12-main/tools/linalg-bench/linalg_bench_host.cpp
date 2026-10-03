#define NOMINMAX
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

struct config {
    const char * cso;
    const char * agility;
    uint32_t sdk;
    uint32_t adapter;
    uint32_t scope;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t k_steps;
    uint32_t inner;
    uint32_t wave;
    uint32_t waves;
    uint32_t threads;
    uint32_t groups;
    uint32_t timed_dispatches;
    uint32_t warmup_dispatches;
    uint32_t a_layout;
    uint32_t b_layout;
    uint32_t c_layout;
    uint32_t a_source_layout;
    uint32_t b_source_layout;
    uint32_t b_cast;
    uint32_t a_type;
    uint32_t b_type;
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t c_offset;
    uint32_t a_source_pad;
    uint32_t b_source_pad;
    uint32_t a_lds_pad;
    uint32_t b_lds_pad;
    uint32_t vector_width;
    uint32_t acc_tiles;
    uint32_t tile_order;
};

static void json_error(const char * status, HRESULT hr, const char * what) {
    std::printf("{\"status\":\"%s\",\"hr\":\"0x%08X\",\"error\":\"%s\"}\n",
                status, (unsigned)hr, what);
}

#define CHECK_JSON(expr, status, what)                                         \
    do {                                                                       \
        HRESULT _hr = (expr);                                                  \
        if (FAILED(_hr)) {                                                     \
            json_error((status), _hr, (what));                                 \
            return 2;                                                          \
        }                                                                      \
    } while (0)

static std::vector<uint8_t> read_file(const char * path) {
    std::vector<uint8_t> data;
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        return data;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size > 0) {
        data.resize((size_t)size);
        if (std::fread(data.data(), 1, data.size(), f) != data.size()) {
            data.clear();
        }
    }
    std::fclose(f);
    return data;
}

static uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) |
                      ((mant + 0x1000u) >> 13));
}

static float half_to_float(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exp = (value >> 10) & 0x1fu;
    uint32_t mant = value & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static size_t element_size(uint32_t type) {
    return type == 1 ? 4u : 2u;
}

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static void store_source(std::vector<uint8_t> & dst, size_t index,
                         uint32_t type, float value) {
    const size_t off = index * element_size(type);
    if (type == 1) {
        std::memcpy(dst.data() + off, &value, sizeof(value));
    } else {
        const uint16_t bits = type == 2 ? float_to_bf16(value)
                                        : float_to_half(value);
        std::memcpy(dst.data() + off, &bits, sizeof(bits));
    }
}

static float a_value(uint32_t r, uint32_t k, uint32_t step) {
    return (float)((int)((r * 3u + k * 5u + step) % 7u) - 3) * 0.125f;
}

static float b_value(uint32_t k, uint32_t c, uint32_t step) {
    return (float)((int)((k * 2u + c * 3u + step) % 5u) - 2) * 0.125f;
}

static size_t layout_index(uint32_t row, uint32_t col,
                           uint32_t rows, uint32_t cols, uint32_t layout,
                           uint32_t pad) {
    return layout == 0 ? (size_t)row * (cols + pad) + col
                       : (size_t)col * (rows + pad) + row;
}

static size_t layout_storage_elems(uint32_t rows, uint32_t cols,
                                   uint32_t layout, uint32_t pad) {
    return layout == 0 ? (size_t)rows * (cols + pad)
                       : (size_t)cols * (rows + pad);
}

static D3D12_RESOURCE_DESC buffer_desc(uint64_t bytes,
                                       D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = (bytes + 3u) & ~3ull;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

static bool parse_config(int argc, char ** argv, config & c) {
    if (argc != 36) {
        std::fprintf(stderr,
            "usage: %s cso agility sdk adapter scope m n k ksteps inner "
            "wave waves threads groups timed warmup alayout blayout clayout "
            "asrc_layout bsrc_layout bcast atype btype "
            "a_offset b_offset c_offset asrc_pad bsrc_pad "
            "a_lds_pad b_lds_pad vector_width "
            "acc_tiles tile_order reserved\n",
            argv[0]);
        return false;
    }
    c.cso = argv[1];
    c.agility = argv[2];
    uint32_t * values[] = {
        &c.sdk, &c.adapter, &c.scope, &c.m, &c.n, &c.k, &c.k_steps,
        &c.inner, &c.wave, &c.waves, &c.threads, &c.groups,
        &c.timed_dispatches, &c.warmup_dispatches, &c.a_layout,
        &c.b_layout, &c.c_layout, &c.a_source_layout,
        &c.b_source_layout, &c.b_cast, &c.a_type, &c.b_type,
        &c.a_offset, &c.b_offset, &c.c_offset,
        &c.a_source_pad, &c.b_source_pad, &c.a_lds_pad, &c.b_lds_pad,
        &c.vector_width, &c.acc_tiles, &c.tile_order
    };
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        *values[i] = (uint32_t)std::strtoul(argv[i + 3], nullptr, 10);
    }
    return true;
}

int main(int argc, char ** argv) {
    config c = {};
    if (!parse_config(argc, argv, c)) {
        return 1;
    }

    const std::vector<uint8_t> dxil = read_file(c.cso);
    if (dxil.empty()) {
        json_error("host_error", E_FAIL, "could not read shader");
        return 2;
    }

    ComPtr<ID3D12SDKConfiguration1> sdk_config;
    ComPtr<ID3D12DeviceFactory> device_factory;
    CHECK_JSON(D3D12GetInterface(CLSID_D3D12SDKConfiguration,
                                 IID_PPV_ARGS(&sdk_config)),
               "host_error", "D3D12GetInterface");
    std::string agility = c.agility;
    if (!agility.empty() && agility.back() != '\\' && agility.back() != '/') {
        agility += '\\';
    }
    CHECK_JSON(sdk_config->CreateDeviceFactory(
                   c.sdk, agility.c_str(), IID_PPV_ARGS(&device_factory)),
               "host_error", "CreateDeviceFactory");

    UUID features[] = { D3D12ExperimentalShaderModels };
    CHECK_JSON(device_factory->EnableExperimentalFeatures(
                   1, features, nullptr, nullptr),
               "feature_error", "EnableExperimentalFeatures");

    ComPtr<IDXGIFactory6> dxgi;
    CHECK_JSON(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi)),
               "host_error", "CreateDXGIFactory2");
    ComPtr<IDXGIAdapter1> adapter;
    CHECK_JSON(dxgi->EnumAdapterByGpuPreference(
                   c.adapter, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                   IID_PPV_ARGS(&adapter)),
               "host_error", "EnumAdapterByGpuPreference");
    DXGI_ADAPTER_DESC1 adapter_desc = {};
    adapter->GetDesc1(&adapter_desc);

    ComPtr<ID3D12Device> device;
    CHECK_JSON(device_factory->CreateDevice(
                   adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                   IID_PPV_ARGS(&device)),
               "host_error", "CreateDevice");
    D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT linalg_support = {};
    const HRESULT linalg_hr = device->CheckFeatureSupport(
        (D3D12_FEATURE)77, &linalg_support, sizeof(linalg_support));
    LARGE_INTEGER driver_version = {};
    adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver_version);

    UINT tg_support_flags = 0;
    UINT tg_min_threads = 0;
    UINT tg_max_threads = 0;
    UINT tg_preferred_threads = 0;
    UINT wave_support_flags = 0;
    UINT wave_native_shapes = 0;
    bool tg_group_size_valid = false;
    bool wave_shape_supported = false;
    if (c.scope == 1) {
        D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT query = {};
        query.OperationType =
            D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_WAVE_MATRIX_MULTIPLY;
        auto & wave = query.WaveMatrixMultiply;
        wave.Inputs.WaveSize = c.wave;
        wave.Inputs.MatrixAComponentType =
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        wave.Inputs.MatrixBComponentType =
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        wave.Inputs.AccumulatorComponentType =
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32;

        HRESULT query_hr = device->CheckFeatureSupport(
            (D3D12_FEATURE)78, &query, sizeof(query));
        if (FAILED(query_hr)) {
            std::printf(
                "{\"status\":\"capability_query_failed\","
                "\"hr\":\"0x%08X\",\"error\":\"WaveMatrixMultiply\","
                "\"vendor_id\":%u,\"device_id\":%u,\"linalg_tier\":%d,"
                "\"driver_version\":\"%u.%u.%u.%u\"}\n",
                (unsigned)query_hr, adapter_desc.VendorId, adapter_desc.DeviceId,
                SUCCEEDED(linalg_hr) ? (int)linalg_support.LinearAlgebraTier : -1,
                HIWORD(driver_version.HighPart), LOWORD(driver_version.HighPart),
                HIWORD(driver_version.LowPart), LOWORD(driver_version.LowPart));
            return 0;
        }

        wave_support_flags = (UINT)wave.SupportFlags;
        wave_native_shapes = wave.NumShapes;
        std::vector<D3D12_LINEAR_ALGEBRA_MATRIX_MULTIPLY_SHAPE> shapes(
            wave_native_shapes);
        if (wave_native_shapes != 0) {
            wave.Shapes = shapes.data();
            query_hr = device->CheckFeatureSupport(
                (D3D12_FEATURE)78, &query, sizeof(query));
            if (FAILED(query_hr)) {
                std::printf(
                    "{\"status\":\"capability_query_failed\","
                    "\"hr\":\"0x%08X\",\"error\":\"WaveMatrixMultiplyShapes\","
                    "\"vendor_id\":%u,\"device_id\":%u,\"linalg_tier\":%d,"
                    "\"driver_version\":\"%u.%u.%u.%u\"}\n",
                    (unsigned)query_hr, adapter_desc.VendorId,
                    adapter_desc.DeviceId,
                    SUCCEEDED(linalg_hr)
                        ? (int)linalg_support.LinearAlgebraTier
                        : -1,
                    HIWORD(driver_version.HighPart),
                    LOWORD(driver_version.HighPart),
                    HIWORD(driver_version.LowPart),
                    LOWORD(driver_version.LowPart));
                return 0;
            }
        }

        for (const auto & shape : shapes) {
            if (shape.M != 0 && shape.K != 0 && shape.N != 0 &&
                c.m % shape.M == 0 && c.k % shape.K == 0 &&
                c.n % shape.N == 0) {
                wave_shape_supported = true;
                break;
            }
        }
        const bool supported =
            (wave_support_flags &
             D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED) != 0;
        if (!supported || !wave_shape_supported) {
            std::printf(
                "{\"status\":\"unsupported\",\"vendor_id\":%u,"
                "\"device_id\":%u,\"linalg_tier\":%d,"
                "\"driver_version\":\"%u.%u.%u.%u\","
                "\"wave_support_flags\":%u,\"wave_native_shapes\":%u,"
                "\"wave_shape_supported\":%u,\"adapter\":\"",
                adapter_desc.VendorId, adapter_desc.DeviceId,
                SUCCEEDED(linalg_hr) ? (int)linalg_support.LinearAlgebraTier : -1,
                HIWORD(driver_version.HighPart), LOWORD(driver_version.HighPart),
                HIWORD(driver_version.LowPart), LOWORD(driver_version.LowPart),
                wave_support_flags, wave_native_shapes,
                wave_shape_supported ? 1u : 0u);
            for (const wchar_t ch : std::wstring(adapter_desc.Description)) {
                std::putchar(ch >= 32 && ch < 127 ? (char)ch : '?');
            }
            std::printf("\"}\n");
            return 0;
        }
    }
    if (c.scope == 2) {
        D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT query = {};
        query.OperationType =
            D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREADGROUP_MATRIX_MULTIPLY;
        auto & tg = query.ThreadGroupMatrixMultiply;
        tg.WaveInputs.WaveSize = c.wave;
        tg.WaveInputs.MatrixAComponentType =
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        tg.WaveInputs.MatrixBComponentType =
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        tg.WaveInputs.AccumulatorComponentType =
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32;
        tg.Shape = { c.m, c.k, c.n };

        const HRESULT query_hr = device->CheckFeatureSupport(
            (D3D12_FEATURE)78, &query, sizeof(query));
        if (FAILED(query_hr)) {
            std::printf(
                "{\"status\":\"capability_query_failed\","
                "\"hr\":\"0x%08X\",\"error\":\"ThreadGroupMatrixMultiply\","
                "\"vendor_id\":%u,\"device_id\":%u,\"linalg_tier\":%d,"
                "\"driver_version\":\"%u.%u.%u.%u\"}\n",
                (unsigned)query_hr, adapter_desc.VendorId, adapter_desc.DeviceId,
                SUCCEEDED(linalg_hr) ? (int)linalg_support.LinearAlgebraTier : -1,
                HIWORD(driver_version.HighPart), LOWORD(driver_version.HighPart),
                HIWORD(driver_version.LowPart), LOWORD(driver_version.LowPart));
            return 0;
        }

        tg_support_flags = (UINT)tg.SupportFlags;
        tg_min_threads = tg.MinThreadGroupSize;
        tg_max_threads = tg.MaxThreadGroupSize;
        tg_preferred_threads = tg.PreferredThreadGroupSize;
        const bool supported =
            (tg_support_flags &
             D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED) != 0;
        tg_group_size_valid =
            supported && tg_min_threads != 0 &&
            c.threads >= tg_min_threads && c.threads <= tg_max_threads &&
            (c.threads % tg_min_threads) == 0;
        if (!supported || !tg_group_size_valid) {
            std::printf(
                "{\"status\":\"%s\",\"vendor_id\":%u,\"device_id\":%u,"
                "\"linalg_tier\":%d,\"driver_version\":\"%u.%u.%u.%u\","
                "\"tg_support_flags\":%u,\"tg_min_threads\":%u,"
                "\"tg_max_threads\":%u,\"tg_preferred_threads\":%u,"
                "\"tg_group_size_valid\":%u,\"adapter\":\"",
                supported ? "unsupported_group_size" : "unsupported",
                adapter_desc.VendorId, adapter_desc.DeviceId,
                SUCCEEDED(linalg_hr) ? (int)linalg_support.LinearAlgebraTier : -1,
                HIWORD(driver_version.HighPart), LOWORD(driver_version.HighPart),
                HIWORD(driver_version.LowPart), LOWORD(driver_version.LowPart),
                tg_support_flags, tg_min_threads, tg_max_threads,
                tg_preferred_threads, tg_group_size_valid ? 1u : 0u);
            for (const wchar_t ch : std::wstring(adapter_desc.Description)) {
                std::putchar(ch >= 32 && ch < 127 ? (char)ch : '?');
            }
            std::printf("\"}\n");
            return 0;
        }
    }

    const uint32_t owner_count = c.scope == 0
        ? c.groups * c.threads
        : c.groups * (c.scope == 1 ? c.waves : 1u);
    const uint32_t matrix_count = owner_count * c.acc_tiles;
    const uint32_t out_n = c.scope == 0 ? 1u : c.n;
    const size_t a_matrix_elems = layout_storage_elems(
        c.m, c.k, c.a_source_layout, c.a_source_pad);
    const size_t b_matrix_elems = c.scope == 0
        ? c.k + c.b_source_pad
        : (c.b_cast
            ? layout_storage_elems(c.n, c.k, 0, c.b_source_pad)
            : layout_storage_elems(
                c.k, c.n, c.b_source_layout, c.b_source_pad));
    const size_t a_matrix_pitch_elems =
        align_up(a_matrix_elems * element_size(c.a_type), 128u) /
        element_size(c.a_type);
    const size_t b_matrix_pitch_elems =
        align_up(b_matrix_elems * element_size(c.b_type), 128u) /
        element_size(c.b_type);
    const uint64_t a_elems =
        (uint64_t)matrix_count * c.k_steps * a_matrix_pitch_elems;
    const uint64_t b_elems =
        (uint64_t)matrix_count * c.k_steps * b_matrix_pitch_elems;
    const uint64_t c_elems = (uint64_t)matrix_count * c.m * out_n;
    const uint64_t a_payload_bytes = a_elems * element_size(c.a_type);
    const uint64_t b_payload_bytes = b_elems * element_size(c.b_type);
    const uint64_t c_payload_bytes = c_elems * sizeof(float);
    const uint64_t a_bytes = c.a_offset + a_payload_bytes;
    const uint64_t b_bytes = c.b_offset + b_payload_bytes;
    const uint64_t c_bytes = c.c_offset + c_payload_bytes;

    std::vector<uint8_t> a_data((size_t)a_payload_bytes);
    std::vector<uint8_t> b_data((size_t)b_payload_bytes);
    std::vector<float> expected((size_t)c_elems, 0.0f);

    for (uint32_t matrix = 0; matrix < matrix_count; ++matrix) {
        for (uint32_t step = 0; step < c.k_steps; ++step) {
            const size_t a_base =
                ((size_t)matrix * c.k_steps + step) *
                a_matrix_pitch_elems;
            for (uint32_t r = 0; r < c.m; ++r) {
                for (uint32_t k = 0; k < c.k; ++k) {
                    const size_t idx = a_base +
                        layout_index(r, k, c.m, c.k, c.a_source_layout,
                                     c.a_source_pad);
                    store_source(a_data, idx, c.a_type, a_value(r, k, step));
                }
            }

            const size_t b_base = ((size_t)matrix * c.k_steps + step) *
                                  b_matrix_pitch_elems;
            if (c.scope == 0) {
                for (uint32_t k = 0; k < c.k; ++k) {
                    store_source(b_data, b_base + k, c.b_type,
                                 b_value(k, 0, step));
                }
            } else if (c.b_cast) {
                for (uint32_t col = 0; col < c.n; ++col) {
                    for (uint32_t k = 0; k < c.k; ++k) {
                        store_source(
                            b_data,
                            b_base + (size_t)col *
                                (c.k + c.b_source_pad) + k,
                                     c.b_type, b_value(k, col, step));
                    }
                }
            } else {
                for (uint32_t k = 0; k < c.k; ++k) {
                    for (uint32_t col = 0; col < c.n; ++col) {
                        const size_t idx = b_base +
                            layout_index(k, col, c.k, c.n,
                                         c.b_source_layout, c.b_source_pad);
                        store_source(b_data, idx, c.b_type,
                                     b_value(k, col, step));
                    }
                }
            }
        }

        for (uint32_t r = 0; r < c.m; ++r) {
            for (uint32_t col = 0; col < out_n; ++col) {
                float sum = 0.0f;
                for (uint32_t rep = 0; rep < c.inner; ++rep) {
                    for (uint32_t step = 0; step < c.k_steps; ++step) {
                        for (uint32_t k = 0; k < c.k; ++k) {
                            const float av = half_to_float(
                                float_to_half(a_value(r, k, step)));
                            const float bv = half_to_float(
                                float_to_half(b_value(k, col, step)));
                            sum += av * bv;
                        }
                    }
                }
                const size_t idx = c.scope == 0
                    ? (size_t)matrix * c.m + r
                    : (size_t)matrix * c.m * c.n +
                      layout_index(r, col, c.m, c.n, c.c_layout, 0);
                expected[idx] = sum;
            }
        }
    }

    D3D12_ROOT_PARAMETER1 params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[0].Descriptor.ShaderRegister = 0;
    params[0].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 1;
    params[1].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs_desc = {};
    rs_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rs_desc.Desc_1_1.NumParameters = 3;
    rs_desc.Desc_1_1.pParameters = params;
    ComPtr<ID3DBlob> rs_blob;
    ComPtr<ID3DBlob> rs_error;
    CHECK_JSON(D3D12SerializeVersionedRootSignature(
                   &rs_desc, &rs_blob, &rs_error),
               "host_error", "D3D12SerializeVersionedRootSignature");
    ComPtr<ID3D12RootSignature> root_signature;
    CHECK_JSON(device->CreateRootSignature(
                   0, rs_blob->GetBufferPointer(), rs_blob->GetBufferSize(),
                   IID_PPV_ARGS(&root_signature)),
               "host_error", "CreateRootSignature");

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
    pso_desc.pRootSignature = root_signature.Get();
    pso_desc.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    const HRESULT pso_hr = device->CreateComputePipelineState(
        &pso_desc, IID_PPV_ARGS(&pso));
    if (FAILED(pso_hr)) {
        std::printf(
            "{\"status\":\"pso_failed\",\"hr\":\"0x%08X\","
            "\"error\":\"CreateComputePipelineState\","
            "\"vendor_id\":%u,\"device_id\":%u,\"linalg_tier\":%d,"
            "\"driver_version\":\"%u.%u.%u.%u\"}\n",
            (unsigned)pso_hr, adapter_desc.VendorId, adapter_desc.DeviceId,
            SUCCEEDED(linalg_hr) ? (int)linalg_support.LinearAlgebraTier : -1,
            HIWORD(driver_version.HighPart), LOWORD(driver_version.HighPart),
            HIWORD(driver_version.LowPart), LOWORD(driver_version.LowPart));
        return 3;
    }

    D3D12_HEAP_PROPERTIES default_heap = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES upload_heap = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_HEAP_PROPERTIES readback_heap = { D3D12_HEAP_TYPE_READBACK };
    ComPtr<ID3D12Resource> a_gpu, b_gpu, c_gpu, upload, readback, query_readback;
    const uint64_t upload_bytes = ((a_bytes + 255u) & ~255ull) + b_bytes;
    auto a_desc = buffer_desc(a_bytes, D3D12_RESOURCE_FLAG_NONE);
    auto b_desc = buffer_desc(b_bytes, D3D12_RESOURCE_FLAG_NONE);
    auto c_desc = buffer_desc(c_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto upload_desc = buffer_desc(upload_bytes, D3D12_RESOURCE_FLAG_NONE);
    auto readback_desc = buffer_desc(c_bytes, D3D12_RESOURCE_FLAG_NONE);
    auto query_desc = buffer_desc(16, D3D12_RESOURCE_FLAG_NONE);
    CHECK_JSON(device->CreateCommittedResource(
                   &default_heap, D3D12_HEAP_FLAG_NONE, &a_desc,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   IID_PPV_ARGS(&a_gpu)),
               "host_error", "CreateResource(A)");
    CHECK_JSON(device->CreateCommittedResource(
                   &default_heap, D3D12_HEAP_FLAG_NONE, &b_desc,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   IID_PPV_ARGS(&b_gpu)),
               "host_error", "CreateResource(B)");
    CHECK_JSON(device->CreateCommittedResource(
                   &default_heap, D3D12_HEAP_FLAG_NONE, &c_desc,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                   IID_PPV_ARGS(&c_gpu)),
               "host_error", "CreateResource(C)");
    CHECK_JSON(device->CreateCommittedResource(
                   &upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                   IID_PPV_ARGS(&upload)),
               "host_error", "CreateResource(upload)");
    CHECK_JSON(device->CreateCommittedResource(
                   &readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   IID_PPV_ARGS(&readback)),
               "host_error", "CreateResource(readback)");
    CHECK_JSON(device->CreateCommittedResource(
                   &readback_heap, D3D12_HEAP_FLAG_NONE, &query_desc,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   IID_PPV_ARGS(&query_readback)),
               "host_error", "CreateResource(query readback)");

    const uint64_t b_upload_offset = (a_bytes + 255u) & ~255ull;
    uint8_t * mapped = nullptr;
    D3D12_RANGE no_read = { 0, 0 };
    CHECK_JSON(upload->Map(0, &no_read, (void **)&mapped),
               "host_error", "Map(upload)");
    std::memset(mapped, 0xcd, (size_t)upload_bytes);
    std::memcpy(mapped + c.a_offset, a_data.data(), a_data.size());
    std::memcpy(mapped + b_upload_offset + c.b_offset,
                b_data.data(), b_data.size());
    upload->Unmap(0, nullptr);

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ComPtr<ID3D12CommandQueue> queue;
    CHECK_JSON(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)),
               "host_error", "CreateCommandQueue");
    UINT64 timestamp_frequency = 0;
    CHECK_JSON(queue->GetTimestampFrequency(&timestamp_frequency),
               "host_error", "GetTimestampFrequency");
    ComPtr<ID3D12CommandAllocator> allocator;
    CHECK_JSON(device->CreateCommandAllocator(
                   D3D12_COMMAND_LIST_TYPE_COMPUTE,
                   IID_PPV_ARGS(&allocator)),
               "host_error", "CreateCommandAllocator");
    ComPtr<ID3D12GraphicsCommandList> command_list;
    CHECK_JSON(device->CreateCommandList(
                   0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.Get(),
                   nullptr, IID_PPV_ARGS(&command_list)),
               "host_error", "CreateCommandList");

    D3D12_QUERY_HEAP_DESC qh_desc = {};
    qh_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qh_desc.Count = 2;
    ComPtr<ID3D12QueryHeap> query_heap;
    CHECK_JSON(device->CreateQueryHeap(&qh_desc, IID_PPV_ARGS(&query_heap)),
               "host_error", "CreateQueryHeap");

    command_list->CopyBufferRegion(a_gpu.Get(), 0, upload.Get(), 0, a_bytes);
    command_list->CopyBufferRegion(b_gpu.Get(), 0, upload.Get(),
                                   b_upload_offset, b_bytes);
    D3D12_RESOURCE_BARRIER input_barriers[2] = {};
    ID3D12Resource * input_resources[] = { a_gpu.Get(), b_gpu.Get() };
    for (int i = 0; i < 2; ++i) {
        input_barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        input_barriers[i].Transition.pResource = input_resources[i];
        input_barriers[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        input_barriers[i].Transition.StateAfter =
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        input_barriers[i].Transition.Subresource =
            D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }
    command_list->ResourceBarrier(2, input_barriers);
    command_list->SetComputeRootSignature(root_signature.Get());
    command_list->SetPipelineState(pso.Get());
    command_list->SetComputeRootShaderResourceView(
        0, a_gpu->GetGPUVirtualAddress());
    command_list->SetComputeRootShaderResourceView(
        1, b_gpu->GetGPUVirtualAddress());
    command_list->SetComputeRootUnorderedAccessView(
        2, c_gpu->GetGPUVirtualAddress());
    for (uint32_t i = 0; i < c.warmup_dispatches; ++i) {
        command_list->Dispatch(c.groups, 1, 1);
    }
    command_list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    for (uint32_t i = 0; i < c.timed_dispatches; ++i) {
        command_list->Dispatch(c.groups, 1, 1);
    }
    command_list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    command_list->ResolveQueryData(
        query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
        query_readback.Get(), 0);

    D3D12_RESOURCE_BARRIER output_barrier = {};
    output_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    output_barrier.Transition.pResource = c_gpu.Get();
    output_barrier.Transition.StateBefore =
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    output_barrier.Transition.StateAfter =
        D3D12_RESOURCE_STATE_COPY_SOURCE;
    output_barrier.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    command_list->ResourceBarrier(1, &output_barrier);
    command_list->CopyBufferRegion(readback.Get(), 0, c_gpu.Get(), 0, c_bytes);
    CHECK_JSON(command_list->Close(), "host_error", "Close(command list)");

    ID3D12CommandList * lists[] = { command_list.Get() };
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    CHECK_JSON(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                   IID_PPV_ARGS(&fence)),
               "host_error", "CreateFence");
    HANDLE event_handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK_JSON(queue->Signal(fence.Get(), 1), "host_error", "Signal");
    CHECK_JSON(fence->SetEventOnCompletion(1, event_handle),
               "host_error", "SetEventOnCompletion");
    WaitForSingleObject(event_handle, INFINITE);
    CloseHandle(event_handle);

    uint64_t * timestamps = nullptr;
    D3D12_RANGE query_range = { 0, 16 };
    CHECK_JSON(query_readback->Map(
                   0, &query_range, (void **)&timestamps),
               "host_error", "Map(query)");
    const uint64_t ticks = timestamps[1] - timestamps[0];
    query_readback->Unmap(0, nullptr);
    const double total_us = (double)ticks * 1.0e6 /
                            (double)timestamp_frequency;
    const double us_per_dispatch = total_us / c.timed_dispatches;
    const double matrices_per_dispatch =
        (double)c.groups * (c.scope == 0 ? c.threads :
                            (c.scope == 1 ? c.waves : 1u)) * c.acc_tiles;
    const double flops_per_dispatch =
        2.0 * c.m * out_n * c.k * c.k_steps * c.inner *
        matrices_per_dispatch;
    const double tflops = us_per_dispatch > 0.0
        ? flops_per_dispatch / (us_per_dispatch * 1.0e6)
        : 0.0;

    uint8_t * output_bytes = nullptr;
    D3D12_RANGE output_range = { 0, (SIZE_T)c_bytes };
    CHECK_JSON(readback->Map(0, &output_range, (void **)&output_bytes),
               "host_error", "Map(output)");
    float * output = reinterpret_cast<float *>(output_bytes + c.c_offset);
    uint64_t mismatches = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t first_mismatch = 0;
    double first_actual = 0.0;
    double first_expected = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double actual = output[i];
        const double ref = expected[i];
        const double abs_err = std::abs(actual - ref);
        const double rel_err = abs_err / std::max(std::abs(ref), 1.0e-6);
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
        if (!std::isfinite(actual) ||
            (abs_err > 0.02 && rel_err > 0.005)) {
            if (mismatches == 0) {
                first_mismatch = i;
                first_actual = actual;
                first_expected = ref;
            }
            ++mismatches;
        }
    }
    readback->Unmap(0, nullptr);

    std::printf(
        "{\"status\":\"%s\",\"gpu_us\":%.6f,\"tflops\":%.6f,"
        "\"mismatches\":%llu,\"max_abs\":%.9g,\"max_rel\":%.9g,"
        "\"first_mismatch\":%llu,\"first_actual\":%.9g,"
        "\"first_expected\":%.9g,"
        "\"matrices\":%u,\"vendor_id\":%u,\"device_id\":%u,"
        "\"linalg_tier\":%d,\"driver_version\":\"%u.%u.%u.%u\","
        "\"timestamp_frequency\":%llu,\"tg_support_flags\":%u,"
        "\"tg_min_threads\":%u,\"tg_max_threads\":%u,"
        "\"tg_preferred_threads\":%u,\"tg_group_size_valid\":%u,"
        "\"wave_support_flags\":%u,\"wave_native_shapes\":%u,"
        "\"wave_shape_supported\":%u,"
        "\"adapter\":\"",
        mismatches == 0 ? "ok" : "mismatch",
        us_per_dispatch, tflops,
        (unsigned long long)mismatches, max_abs, max_rel,
        (unsigned long long)first_mismatch, first_actual, first_expected,
        matrix_count,
        adapter_desc.VendorId, adapter_desc.DeviceId,
        SUCCEEDED(linalg_hr) ? (int)linalg_support.LinearAlgebraTier : -1,
        HIWORD(driver_version.HighPart), LOWORD(driver_version.HighPart),
        HIWORD(driver_version.LowPart), LOWORD(driver_version.LowPart),
        (unsigned long long)timestamp_frequency,
        tg_support_flags, tg_min_threads, tg_max_threads,
        tg_preferred_threads, tg_group_size_valid ? 1u : 0u,
        wave_support_flags, wave_native_shapes,
        wave_shape_supported ? 1u : 0u);
    for (const wchar_t ch : std::wstring(adapter_desc.Description)) {
        std::putchar(ch >= 32 && ch < 127 ? (char)ch : '?');
    }
    std::printf("\"}\n");
    return mismatches == 0 ? 0 : 4;
}
