// Dump VK_KHR_cooperative_matrix shapes, to compare against what the D3D12
// Linear Algebra (Cooperative Vector) path advertises on the same part.
//
// build:
//   cl /nologo /O2 /EHsc /I "%VULKAN_SDK%\Include" vk_coopmat_probe.cpp \
//      /link "%VULKAN_SDK%\Lib\vulkan-1.lib"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <cstdio>
#include <vector>
#include <windows.h>

static const char * comp_name(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
        case VK_COMPONENT_TYPE_SINT8_KHR:   return "s8";
        case VK_COMPONENT_TYPE_SINT16_KHR:  return "s16";
        case VK_COMPONENT_TYPE_SINT32_KHR:  return "s32";
        case VK_COMPONENT_TYPE_SINT64_KHR:  return "s64";
        case VK_COMPONENT_TYPE_UINT8_KHR:   return "u8";
        case VK_COMPONENT_TYPE_UINT16_KHR:  return "u16";
        case VK_COMPONENT_TYPE_UINT32_KHR:  return "u32";
        case VK_COMPONENT_TYPE_UINT64_KHR:  return "u64";
        default: return "?";
    }
}

static const char * scope_name(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR:      return "Device";
        case VK_SCOPE_WORKGROUP_KHR:   return "Workgroup";
        case VK_SCOPE_SUBGROUP_KHR:    return "Subgroup";
        case VK_SCOPE_QUEUE_FAMILY_KHR:return "QueueFamily";
        default: return "?";
    }
}

int main() {
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) { printf("no vulkan-1.dll\n"); return 1; }

    auto gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(lib, "vkGetInstanceProcAddr");
    auto vkCreateInstance = (PFN_vkCreateInstance)gipa(nullptr, "vkCreateInstance");

    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;

    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        printf("vkCreateInstance failed\n"); return 1;
    }

    auto vkEnumeratePhysicalDevices =
        (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
    auto vkGetPhysicalDeviceProperties =
        (PFN_vkGetPhysicalDeviceProperties)gipa(inst, "vkGetPhysicalDeviceProperties");
    auto vkGetPhysicalDeviceProperties2 =
        (PFN_vkGetPhysicalDeviceProperties2)gipa(inst, "vkGetPhysicalDeviceProperties2");
    auto vkGetCoopMat = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        gipa(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

    if (!vkGetCoopMat) {
        printf("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR not available\n");
        return 1;
    }

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());

    for (uint32_t d = 0; d < ndev; d++) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devs[d], &props);

        VkPhysicalDeviceSubgroupProperties sg{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        VkPhysicalDeviceProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(devs[d], &p2);

        printf("\n=== %s  (subgroup size %u) ===\n", props.deviceName, sg.subgroupSize);

        uint32_t n = 0;
        vkGetCoopMat(devs[d], &n, nullptr);
        if (n == 0) { printf("  no cooperative matrix shapes\n"); continue; }

        std::vector<VkCooperativeMatrixPropertiesKHR> cm(n);
        for (auto & c : cm) c.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        vkGetCoopMat(devs[d], &n, cm.data());

        printf("  %u shapes\n", n);
        printf("  %-4s %-4s %-4s  %-5s %-5s %-5s %-7s  %-10s %s\n",
               "M", "N", "K", "A", "B", "C", "Result", "scope", "sat");
        for (uint32_t i = 0; i < n; i++) {
            const auto & c = cm[i];
            printf("  %-4u %-4u %-4u  %-5s %-5s %-5s %-7s  %-10s %s\n",
                   c.MSize, c.NSize, c.KSize,
                   comp_name(c.AType), comp_name(c.BType),
                   comp_name(c.CType), comp_name(c.ResultType),
                   scope_name(c.scope),
                   c.saturatingAccumulation ? "yes" : "no");
        }
    }
    return 0;
}
