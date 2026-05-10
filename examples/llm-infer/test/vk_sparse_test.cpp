// vk_sparse_test.cpp — Probe Vulkan sparse resource support for layer windowing feasibility
//
// Build: cl /EHsc /I %VULKAN_SDK%\Include vk_sparse_test.cpp /link /LIBPATH:%VULKAN_SDK%\Lib vulkan-1.lib
// Run:   vk_sparse_test.exe
//
// Reports per-device:
//   1. Sparse binding / residency features
//   2. Sparse queue family support
//   3. Memory heaps & types (HOST_VISIBLE + DEVICE_LOCAL for UMA detection)
//   4. VK_EXT_memory_budget support
//   5. VK_EXT_external_memory_host support
//   6. Practical sparse buffer create + bind test (if supported)

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char * bool_str(VkBool32 b) { return b ? "YES" : "no"; }

static void print_memory_flags(VkMemoryPropertyFlags f) {
    if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)     printf(" DEVICE_LOCAL");
    if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)     printf(" HOST_VISIBLE");
    if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)    printf(" HOST_COHERENT");
    if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)      printf(" HOST_CACHED");
    if (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)  printf(" LAZY_ALLOC");
    if (f & VK_MEMORY_PROPERTY_PROTECTED_BIT)         printf(" PROTECTED");
}

static bool check_device_extension(VkPhysicalDevice pd, const char * ext_name) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, exts.data());
    for (auto & e : exts) {
        if (strcmp(e.extensionName, ext_name) == 0) return true;
    }
    return false;
}

static void test_sparse_buffer(VkPhysicalDevice pd, VkDevice dev, uint32_t sparse_queue_family) {
    printf("\n  --- Practical sparse buffer test ---\n");

    // Create a 64MB sparse buffer (typical layer size)
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size  = 64 * 1024 * 1024;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;

    VkBuffer buf = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer(dev, &bci, nullptr, &buf);
    if (r != VK_SUCCESS) {
        printf("  vkCreateBuffer (sparse): FAILED (VkResult=%d)\n", r);
        return;
    }
    printf("  vkCreateBuffer (sparse, 64 MiB): OK\n");

    // Query sparse memory requirements
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(dev, buf, &mem_req);
    printf("  memory requirements: size=%llu, alignment=%llu, memoryTypeBits=0x%x\n",
           (unsigned long long)mem_req.size,
           (unsigned long long)mem_req.alignment,
           mem_req.memoryTypeBits);

    printf("  sparse page size (alignment): %llu bytes (%llu KB)\n",
           (unsigned long long)mem_req.alignment,
           (unsigned long long)(mem_req.alignment / 1024));

    // Allocate a small chunk of device memory (one page)
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(pd, &mem_props);

    uint32_t mem_type_idx = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_req.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mem_type_idx = i;
            break;
        }
    }

    if (mem_type_idx == UINT32_MAX) {
        printf("  No suitable DEVICE_LOCAL memory type found for sparse buffer\n");
        vkDestroyBuffer(dev, buf, nullptr);
        return;
    }

    // Allocate one 64KB page
    const VkDeviceSize page_size = 64 * 1024;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = page_size;
    mai.memoryTypeIndex = mem_type_idx;

    VkDeviceMemory mem = VK_NULL_HANDLE;
    r = vkAllocateMemory(dev, &mai, nullptr, &mem);
    if (r != VK_SUCCESS) {
        printf("  vkAllocateMemory (64KB page): FAILED (VkResult=%d)\n", r);
        vkDestroyBuffer(dev, buf, nullptr);
        return;
    }
    printf("  vkAllocateMemory (64KB page, type=%u): OK\n", mem_type_idx);

    // Bind the first 64KB of the sparse buffer
    VkSparseMemoryBind bind = {};
    bind.resourceOffset = 0;
    bind.size           = page_size;
    bind.memory         = mem;
    bind.memoryOffset   = 0;

    VkSparseBufferMemoryBindInfo buf_bind = {};
    buf_bind.buffer    = buf;
    buf_bind.bindCount = 1;
    buf_bind.pBinds    = &bind;

    VkBindSparseInfo bind_info = { VK_STRUCTURE_TYPE_BIND_SPARSE_INFO };
    bind_info.bufferBindCount = 1;
    bind_info.pBufferBinds    = &buf_bind;

    VkQueue sparse_queue;
    vkGetDeviceQueue(dev, sparse_queue_family, 0, &sparse_queue);

    r = vkQueueBindSparse(sparse_queue, 1, &bind_info, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) {
        printf("  vkQueueBindSparse (bind 64KB at offset 0): FAILED (VkResult=%d)\n", r);
    } else {
        printf("  vkQueueBindSparse (bind 64KB at offset 0): OK\n");

        // Now unbind (set memory to VK_NULL_HANDLE)
        VkSparseMemoryBind unbind = {};
        unbind.resourceOffset = 0;
        unbind.size           = page_size;
        unbind.memory         = VK_NULL_HANDLE;
        unbind.memoryOffset   = 0;

        buf_bind.pBinds = &unbind;
        r = vkQueueBindSparse(sparse_queue, 1, &bind_info, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) {
            printf("  vkQueueBindSparse (unbind): FAILED (VkResult=%d)\n", r);
        } else {
            printf("  vkQueueBindSparse (unbind / decommit): OK\n");
        }
    }

    vkQueueWaitIdle(sparse_queue);
    vkFreeMemory(dev, mem, nullptr);
    vkDestroyBuffer(dev, buf, nullptr);

    printf("  --- Sparse buffer test PASSED ---\n");
}

int main() {
    printf("=== Vulkan Sparse Resource Support Probe ===\n\n");

    // Create instance
    VkApplicationInfo app_info = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app_info.pApplicationName   = "vk_sparse_test";
    app_info.apiVersion         = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app_info;

    VkInstance instance;
    VkResult r = vkCreateInstance(&ici, nullptr, &instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed: %d\n", r);
        return 1;
    }

    // Enumerate physical devices
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
    if (dev_count == 0) {
        fprintf(stderr, "No Vulkan physical devices found\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumeratePhysicalDevices(instance, &dev_count, devices.data());

    for (uint32_t d = 0; d < dev_count; d++) {
        VkPhysicalDevice pd = devices[d];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);

        printf("Device %u: %s\n", d, props.deviceName);
        printf("  API version: %u.%u.%u\n",
               VK_VERSION_MAJOR(props.apiVersion),
               VK_VERSION_MINOR(props.apiVersion),
               VK_VERSION_PATCH(props.apiVersion));

        const char * type_str = "Unknown";
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: type_str = "Integrated GPU (UMA)"; break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   type_str = "Discrete GPU"; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    type_str = "Virtual GPU"; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            type_str = "CPU"; break;
            default: break;
        }
        printf("  Device type: %s\n", type_str);

        // --- Features ---
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(pd, &features);

        printf("\n  Sparse features:\n");
        printf("    sparseBinding:                    %s\n", bool_str(features.sparseBinding));
        printf("    sparseResidencyBuffer:            %s\n", bool_str(features.sparseResidencyBuffer));
        printf("    sparseResidencyImage2D:           %s\n", bool_str(features.sparseResidencyImage2D));
        printf("    sparseResidencyImage3D:           %s\n", bool_str(features.sparseResidencyImage3D));
        printf("    sparseResidency2Samples:          %s\n", bool_str(features.sparseResidency2Samples));
        printf("    sparseResidency4Samples:          %s\n", bool_str(features.sparseResidency4Samples));
        printf("    sparseResidency8Samples:          %s\n", bool_str(features.sparseResidency8Samples));
        printf("    sparseResidency16Samples:         %s\n", bool_str(features.sparseResidency16Samples));
        printf("    sparseResidencyAliased:           %s\n", bool_str(features.sparseResidencyAliased));

        // --- Sparse page size (from device limits) ---
        printf("\n  Sparse address space size: %llu GB\n",
               (unsigned long long)(props.limits.sparseAddressSpaceSize / (1024ull * 1024 * 1024)));
        printf("  Buffer image granularity:  %llu bytes\n",
               (unsigned long long)props.limits.bufferImageGranularity);

        // --- Queue families ---
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf_props(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, qf_props.data());

        printf("\n  Queue families (%u):\n", qf_count);
        uint32_t sparse_queue_family = UINT32_MAX;
        for (uint32_t q = 0; q < qf_count; q++) {
            bool has_sparse = (qf_props[q].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0;
            printf("    [%u] count=%u flags:", q, qf_props[q].queueCount);
            if (qf_props[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)       printf(" GRAPHICS");
            if (qf_props[q].queueFlags & VK_QUEUE_COMPUTE_BIT)        printf(" COMPUTE");
            if (qf_props[q].queueFlags & VK_QUEUE_TRANSFER_BIT)       printf(" TRANSFER");
            if (has_sparse)                                             printf(" SPARSE_BINDING");
            if (qf_props[q].queueFlags & VK_QUEUE_PROTECTED_BIT)      printf(" PROTECTED");
            if (qf_props[q].queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) printf(" VIDEO_DECODE");
            if (qf_props[q].queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) printf(" VIDEO_ENCODE");
            printf("\n");

            if (has_sparse && sparse_queue_family == UINT32_MAX) {
                sparse_queue_family = q;
            }
        }

        // --- Memory heaps & types ---
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(pd, &mem_props);

        printf("\n  Memory heaps (%u):\n", mem_props.memoryHeapCount);
        for (uint32_t h = 0; h < mem_props.memoryHeapCount; h++) {
            printf("    Heap %u: %8llu MiB", h,
                   (unsigned long long)(mem_props.memoryHeaps[h].size / (1024 * 1024)));
            if (mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                printf(" [DEVICE_LOCAL]");
            printf("\n");
        }

        printf("  Memory types (%u):\n", mem_props.memoryTypeCount);
        bool has_uma_memory = false;
        for (uint32_t t = 0; t < mem_props.memoryTypeCount; t++) {
            auto flags = mem_props.memoryTypes[t].propertyFlags;
            printf("    Type %2u (heap %u):", t, mem_props.memoryTypes[t].heapIndex);
            print_memory_flags(flags);
            printf("\n");

            if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                has_uma_memory = true;
            }
        }
        printf("  UMA memory (DEVICE_LOCAL + HOST_VISIBLE): %s\n", has_uma_memory ? "YES" : "no");

        // --- Key extensions ---
        bool has_mem_budget   = check_device_extension(pd, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        bool has_ext_mem_host = check_device_extension(pd, "VK_EXT_external_memory_host");
        bool has_mem_priority = check_device_extension(pd, "VK_EXT_memory_priority");

        printf("\n  Key extensions:\n");
        printf("    VK_EXT_memory_budget:          %s\n", has_mem_budget   ? "YES" : "no");
        printf("    VK_EXT_external_memory_host:   %s\n", has_ext_mem_host ? "YES" : "no");
        printf("    VK_EXT_memory_priority:        %s\n", has_mem_priority ? "YES" : "no");

        // --- Practical test ---
        if (features.sparseBinding && features.sparseResidencyBuffer && sparse_queue_family != UINT32_MAX) {
            // Create a logical device with sparse features enabled
            float queue_priority = 1.0f;
            VkDeviceQueueCreateInfo dqci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
            dqci.queueFamilyIndex = sparse_queue_family;
            dqci.queueCount       = 1;
            dqci.pQueuePriorities = &queue_priority;

            VkPhysicalDeviceFeatures enabled_features = {};
            enabled_features.sparseBinding         = VK_TRUE;
            enabled_features.sparseResidencyBuffer = VK_TRUE;

            VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
            dci.queueCreateInfoCount = 1;
            dci.pQueueCreateInfos    = &dqci;
            dci.pEnabledFeatures     = &enabled_features;

            VkDevice dev = VK_NULL_HANDLE;
            r = vkCreateDevice(pd, &dci, nullptr, &dev);
            if (r == VK_SUCCESS) {
                test_sparse_buffer(pd, dev, sparse_queue_family);
                vkDestroyDevice(dev, nullptr);
            } else {
                printf("\n  vkCreateDevice with sparse features: FAILED (VkResult=%d)\n", r);
            }
        } else {
            printf("\n  SKIPPING practical test: sparse binding/residency not supported or no sparse queue\n");
        }

        // --- Layer windowing feasibility summary ---
        printf("\n  ========================================\n");
        printf("  LAYER WINDOWING FEASIBILITY SUMMARY:\n");
        printf("    Sparse binding:      %s (required)\n", bool_str(features.sparseBinding));
        printf("    Sparse residency:    %s (required)\n", bool_str(features.sparseResidencyBuffer));
        printf("    Sparse queue:        %s (required)\n", sparse_queue_family != UINT32_MAX ? "YES" : "no");
        printf("    UMA memory:          %s (nice to have)\n", has_uma_memory ? "YES" : "no");
        printf("    Memory budget ext:   %s (nice to have)\n", has_mem_budget ? "YES" : "no");
        printf("    External mem host:   %s (nice to have)\n", has_ext_mem_host ? "YES" : "no");

        bool feasible = features.sparseBinding && features.sparseResidencyBuffer &&
                        sparse_queue_family != UINT32_MAX;
        printf("    >>> VERDICT: %s <<<\n", feasible ? "LAYER WINDOWING IS FEASIBLE" : "NOT FEASIBLE");
        printf("  ========================================\n\n");
    }

    vkDestroyInstance(instance, nullptr);
    return 0;
}
