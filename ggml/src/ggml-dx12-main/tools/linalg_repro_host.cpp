// linalg_repro_host.cpp - standalone runner for the LinAlg probe shaders.
//
// Dispatches one probe shader on a chosen adapter and dumps its output buffer.
// Each probe is self-checking and documents its own expected values in its
// header. See tools/LINALG-PROBES.md for the survey procedure.
//
// Build (from a VS x64 developer prompt):
//   dxc -T cs_6_10 -E main -enable-16bit-types -I <dxc>/inc/hlsl -Fo probe.cso \
//       ggml/src/ggml-dx12/shaders/<probe>.hlsl
//   cl /nologo /std:c++17 /EHsc /O2 linalg_repro_host.cpp \
//       /I <agility>/build/native/include /link d3d12.lib dxgi.lib dxguid.lib
//
// Run:
//   linalg_repro_host.exe probe.cso <agility-bin-x64-dir> <sdk-version> [adapter] [raw-count]
//
// Most probes write outside the 16x16 window the detailed dump assumes, so
// pass raw-count to see their cells - 1100 covers every probe shipped here.
//
// Requires Windows Developer Mode (experimental shader models).

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

#define CHECK(hr, what)                                                        \
    do {                                                                       \
        HRESULT _hr = (hr);                                                    \
        if (FAILED(_hr)) {                                                     \
            printf("FAILED: %s (0x%08X)\n", (what), (unsigned)_hr);            \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static std::vector<uint8_t> read_file(const char * path) {
    std::vector<uint8_t> data;
    FILE * f = fopen(path, "rb");
    if (!f) return data;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    data.resize((size_t)n);
    if (fread(data.data(), 1, (size_t)n, f) != (size_t)n) data.clear();
    fclose(f);
    return data;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        printf("usage: %s <probe.cso> <agility-bin-x64-dir> <sdk-version> [adapter-index] [raw-count]\n", argv[0]);
        return 1;
    }
    const char * cso_path    = argv[1];
    const char * agility_dir = argv[2];
    const UINT   sdk_version = (UINT)atoi(argv[3]);
    const UINT   adapter_idx = argc > 4 ? (UINT)atoi(argv[4]) : 0;

    std::vector<uint8_t> dxil = read_file(cso_path);
    if (dxil.empty()) {
        printf("FAILED: could not read %s\n", cso_path);
        return 1;
    }

    // The Agility loader only honours D3D12SDKVersion exports on the host EXE,
    // so route device creation through an explicit device factory instead.
    ComPtr<ID3D12SDKConfiguration1> sdk_cfg;
    ComPtr<ID3D12DeviceFactory>     factory;
    CHECK(D3D12GetInterface(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(&sdk_cfg)),
          "D3D12GetInterface(CLSID_D3D12SDKConfiguration)");

    std::string dir = agility_dir;
    if (!dir.empty() && dir.back() != '\\' && dir.back() != '/') dir += '\\';
    CHECK(sdk_cfg->CreateDeviceFactory(sdk_version, dir.c_str(), IID_PPV_ARGS(&factory)),
          "CreateDeviceFactory");
    printf("Agility SDK v%u loaded from %s\n", sdk_version, dir.c_str());

    UUID features[] = { D3D12ExperimentalShaderModels };
    CHECK(factory->EnableExperimentalFeatures(1, features, nullptr, nullptr),
          "EnableExperimentalFeatures(D3D12ExperimentalShaderModels) - is Developer Mode on?");
    printf("Experimental shader models: enabled\n");

    ComPtr<IDXGIFactory6> dxgi;
    CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi)), "CreateDXGIFactory2");
    ComPtr<IDXGIAdapter1> adapter;
    CHECK(dxgi->EnumAdapterByGpuPreference(adapter_idx,
                                           DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                           IID_PPV_ARGS(&adapter)),
          "EnumAdapterByGpuPreference");
    DXGI_ADAPTER_DESC1 adesc = {};
    adapter->GetDesc1(&adesc);
    printf("Adapter %u: %ls\n", adapter_idx, adesc.Description);

    ComPtr<ID3D12Device> device;
    CHECK(factory->CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)),
          "CreateDevice");

    D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT la = {};
    if (SUCCEEDED(device->CheckFeatureSupport((D3D12_FEATURE)77, &la, sizeof(la)))) {
        printf("LinAlg tier: %d\n", (int)la.LinearAlgebraTier);
    } else {
        printf("LinAlg tier: query failed\n");
    }

    const UINT num_uints = 4096;
    const UINT buf_bytes = num_uints * sizeof(uint32_t);

    D3D12_HEAP_PROPERTIES hp_default  = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES hp_readback = { D3D12_HEAP_TYPE_READBACK };

    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width            = buf_bytes;
    bd.Height           = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels        = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ComPtr<ID3D12Resource> uav_buf;
    CHECK(device->CreateCommittedResource(&hp_default, D3D12_HEAP_FLAG_NONE, &bd,
                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                          nullptr, IID_PPV_ARGS(&uav_buf)),
          "CreateCommittedResource(uav)");

    bd.Flags = D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> rb_buf;
    CHECK(device->CreateCommittedResource(&hp_readback, D3D12_HEAP_FLAG_NONE, &bd,
                                          D3D12_RESOURCE_STATE_COPY_DEST,
                                          nullptr, IID_PPV_ARGS(&rb_buf)),
          "CreateCommittedResource(readback)");

    // A single root UAV matches the shader's `RWStructuredBuffer OutBuff : u0`.
    D3D12_ROOT_PARAMETER rp = {};
    rp.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp.Descriptor.ShaderRegister = 0;

    D3D12_ROOT_SIGNATURE_DESC rs_desc = {};
    rs_desc.NumParameters = 1;
    rs_desc.pParameters   = &rp;

    ComPtr<ID3DBlob> rs_blob, rs_err;
    CHECK(D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                      &rs_blob, &rs_err),
          "D3D12SerializeRootSignature");
    ComPtr<ID3D12RootSignature> root_sig;
    CHECK(device->CreateRootSignature(0, rs_blob->GetBufferPointer(),
                                      rs_blob->GetBufferSize(), IID_PPV_ARGS(&root_sig)),
          "CreateRootSignature");

    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature     = root_sig.Get();
    pd.CS.pShaderBytecode = dxil.data();
    pd.CS.BytecodeLength  = dxil.size();
    ComPtr<ID3D12PipelineState> pso;
    CHECK(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)),
          "CreateComputePipelineState - was the shader built as cs_6_10?");

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ComPtr<ID3D12CommandQueue> queue;
    CHECK(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");

    ComPtr<ID3D12CommandAllocator> alloc;
    CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc)),
          "CreateCommandAllocator");
    ComPtr<ID3D12GraphicsCommandList> cl;
    CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(),
                                    pso.Get(), IID_PPV_ARGS(&cl)),
          "CreateCommandList");

    cl->SetComputeRootSignature(root_sig.Get());
    cl->SetPipelineState(pso.Get());
    cl->SetComputeRootUnorderedAccessView(0, uav_buf->GetGPUVirtualAddress());
    cl->Dispatch(1, 1, 1);

    D3D12_RESOURCE_BARRIER rb = {};
    rb.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    rb.Transition.pResource   = uav_buf.Get();
    rb.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    rb.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
    rb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cl->ResourceBarrier(1, &rb);
    cl->CopyResource(rb_buf.Get(), uav_buf.Get());
    CHECK(cl->Close(), "Close");

    ID3D12CommandList * lists[] = { cl.Get() };
    queue->ExecuteCommandLists(1, lists);

    ComPtr<ID3D12Fence> fence;
    CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK(queue->Signal(fence.Get(), 1), "Signal");
    CHECK(fence->SetEventOnCompletion(1, ev), "SetEventOnCompletion");
    WaitForSingleObject(ev, INFINITE);
    CloseHandle(ev);

    uint32_t *  data  = nullptr;
    D3D12_RANGE range = { 0, buf_bytes };
    CHECK(rb_buf->Map(0, &range, (void **)&data), "Map");

    const uint32_t mismatches = data[0];
    printf("\nmismatching cells: %u (expected 0)\n", mismatches);

    // Optional raw window, for repro shaders whose buffer layout differs from
    // the 16x16 one the detailed dump below assumes.
    if (argc > 5) {
        const uint32_t raw_n = (uint32_t)atoi(argv[5]);
        printf("raw OutBuff[0..%u]:\n", raw_n ? raw_n - 1 : 0);
        for (uint32_t i = 0; i < raw_n && i < num_uints; ++i) {
            printf("  [%4u] = %d\n", i, (int32_t)data[i]);
        }
        rb_buf->Unmap(0, nullptr);
        return mismatches == 0 ? 0 : 2;
    }

    printf("hardcoded-mapping mismatches: %u (0 means the layout is knowable)\n",
           data[769]);

    printf("\nderived true mapping (lane: e -> (row,col)), first 20 lanes:\n");
    for (int L = 0; L < 64; ++L) {
        printf("  lane %2d:", L);
        for (int e = 0; e < 8; ++e) {
            const uint32_t v = data[770 + L * 8 + e];
            if (v == 0 || v > 256) {
                printf("  e%d=( ?, ?)", e);
            } else {
                printf("  e%d=(%2u,%2u)", e, (v - 1) / 16, (v - 1) % 16);
            }
        }
        printf("\n");
    }

    if (mismatches != 0) {
        printf("\nfirst 8 disagreeing tile cells (row,col): store vs getcoordinate\n");
        int shown = 0;
        for (int i = 0; i < 256 && shown < 8; ++i) {
            const uint32_t a = data[1 + i];
            const uint32_t b = data[257 + i];
            if (a != b) {
                printf("  (%2d,%2d): store=%-11d getcoord=%-11d\n",
                       i / 16, i % 16, (int)a, (int)b);
                ++shown;
            }
        }

        // Block 513.. is indexed by (lane * ACC_E + element), ACC_E = 4.
        printf("\ncoordinates reported by GetCoordinate, all 64 lanes:\n");
        for (int lane = 0; lane < 64; ++lane) {
            printf("  lane %2d:", lane);
            for (int e = 0; e < 4; ++e) {
                const uint32_t packed = data[513 + lane * 4 + e];
                printf("  e%d=(%2u,%2u)", e, packed >> 16, packed & 0xFFFF);
            }
            printf("\n");
        }

        // Every one of the 256 (lane,element) slots should map to a distinct
        // cell of the 16x16 tile. Count how many cells are hit.
        int hits[256] = {};
        int oob = 0;
        for (int i = 0; i < 256; ++i) {
            const uint32_t packed = data[513 + i];
            const uint32_t r = packed >> 16, c = packed & 0xFFFF;
            if (r < 16 && c < 16) hits[r * 16 + c]++; else ++oob;
        }
        int covered = 0, duped = 0;
        for (int i = 0; i < 256; ++i) {
            if (hits[i] > 0) ++covered;
            if (hits[i] > 1) ++duped;
        }
        printf("\ncoverage: %d/256 tile cells addressed, %d cells addressed more than once, "
               "%d (lane,element) slots reported out of range\n", covered, duped, oob);
    }

    printf("\n%s\n", mismatches == 0 ? "PASS - GetCoordinate agrees with Store"
                                     : "FAIL - GetCoordinate disagrees with Store");
    rb_buf->Unmap(0, nullptr);
    return mismatches == 0 ? 0 : 2;
}
