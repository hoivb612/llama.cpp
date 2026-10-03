// Reports the wave lane range D3D12 allows, and the wave sizes a compute PSO
// will actually accept via [WaveSize(N)]. The LinAlg GEMM is compiled for
// WaveLaneCountMin (16 on this part) while Vulkan drives the same matrix
// hardware from a 32-lane subgroup - see TUNING.md section 39.
//
// cl /std:c++17 /EHsc wave_caps_probe.cpp /link d3d12.lib dxgi.lib dxguid.lib

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdio>

using Microsoft::WRL::ComPtr;

int main() {
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
        printf("CreateDXGIFactory2 failed\n");
        return 1;
    }

    for (UINT i = 0;; i++) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapterByGpuPreference(
                i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&adapter)) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }

        ComPtr<ID3D12Device> device;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&device)))) {
            continue;
        }

        printf("Adapter %u: %ls\n", i, desc.Description);

        D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1 = {};
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1,
                                                  &o1, sizeof(o1)))) {
            printf("  WaveLaneCountMin      : %u\n", o1.WaveLaneCountMin);
            printf("  WaveLaneCountMax      : %u\n", o1.WaveLaneCountMax);
            printf("  TotalLaneCount        : %u\n", o1.TotalLaneCount);
            printf("  WaveOps               : %s\n", o1.WaveOps ? "yes" : "no");
        }

        // WaveSize(N) needs SM 6.6; the range form needs 6.8.
        D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL(0x69) };
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,
                                                  &sm, sizeof(sm)))) {
            printf("  Max shader model      : 0x%02x\n", (unsigned)sm.HighestShaderModel);
        }
        printf("\n");
    }
    return 0;
}
