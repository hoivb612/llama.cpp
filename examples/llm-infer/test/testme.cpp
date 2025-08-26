#pragma warning (disable:4267) //  conversion from 'size_t' to 'int' ...

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>

#if !defined WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#ifndef AFFINITY_MASK
#define AFFINITY_MASK(n) ((ULONG_PTR)1 << (n))
#endif

static bool __BitScanForward64(
   unsigned long * Index,
   UINT64 Mask) {
    UINT64 bit = 1;
    long i;
    for (i = 0; i < 64; i++)
    {
        if (Mask & bit)
        {
            *Index = i;
            return 1;
        }
        bit <<= 1;
    }
    return 0;
}

static bool __BitScanReverse64(
   unsigned long * Index,
   UINT64 Mask) {
    UINT64 bit = 1;
    long i;
    bit <<= 63;
    for (i = 63; i >= 0; i--)
    {
        if (Mask & bit)
        {
            *Index = i;
            return 1;
        }
        bit >>= 1;
    }
    return 0;
}

void setThreadStart(int vpIndex) {
    ULONG cpuIndex;
    KAFFINITY preferredCpuMask;
    KAFFINITY systemGroupAffinityMask;
    int testNumberProcessors = 24;
    GROUP_AFFINITY groupAffinity = { 0 };
    PROCESSOR_NUMBER idealProcessor = { 0 };

    UCHAR vpCount = 7;
    systemGroupAffinityMask = 0x00ffffff;
    preferredCpuMask = systemGroupAffinityMask;

    printf("%s: vpCount[%2d]: starting Mask - 0x%08x\n",
        __func__,
        vpCount,
        (ULONG) preferredCpuMask);
        
    while ((vpCount > 0) && (testNumberProcessors > vpCount)) {
        BitScanForward64(&cpuIndex, preferredCpuMask);
        preferredCpuMask ^= AFFINITY_MASK(cpuIndex);
        printf("%s: -- vpCount[%2d]:cpuIndex %d - 0x%X - 0x%08x\n",
             __FUNCTION__,
             vpCount,
             cpuIndex,
             cpuIndex,
            (ULONG) preferredCpuMask);
        vpCount--;
    }

    printf("%s: >> vpCount[%2d]:cpuIndex %d - 0x%X - 0x%08x\n",
        __FUNCTION__,
        vpCount,
        cpuIndex,
        cpuIndex,
        (ULONG) preferredCpuMask);

    // Push everything back 1 position to avoid the beginning core
    preferredCpuMask = ((preferredCpuMask << 1) | 0x1) & systemGroupAffinityMask;

    printf("%s: ++ vpCount[%2d]: single affinity starting Mask - 0x%08x\n",
        __FUNCTION__,
        vpCount,
        (ULONG) preferredCpuMask);

    if (vpCount == 0) {
        groupAffinity.Mask = systemGroupAffinityMask & ~preferredCpuMask;
    }

    //
    // This can happen if the VM VP count == number of system processors.
    //NT_ASSERT(groupAffinity.Mask != 0);
    //

    if (groupAffinity.Mask == 0) {
        groupAffinity.Mask = systemGroupAffinityMask;
    }

    {
        preferredCpuMask = ~preferredCpuMask & systemGroupAffinityMask;
        vpCount = vpIndex + 1;

        if (preferredCpuMask == 0) {
            preferredCpuMask = systemGroupAffinityMask;
        }

        printf("%s: :: vpCount[%2d]: single affinity starting Mask - 0x%08x\n",
            __FUNCTION__,
            vpCount,
            (ULONG) preferredCpuMask);

        //
        // N.B. This BitScanReverse64 is just there to make the compiler happy.
        //

        BitScanReverse64(&cpuIndex, preferredCpuMask);
        while (vpCount > 0) {
            BitScanReverse64(&cpuIndex, preferredCpuMask);
            preferredCpuMask ^= AFFINITY_MASK(cpuIndex);
            vpCount--;
        }
        groupAffinity.Mask = AFFINITY_MASK(cpuIndex);
        printf("%s: ** groupAffinity Mask (cpuIndex:%d) - 0x%08x\n",
            __FUNCTION__,
            cpuIndex,
            (ULONG) groupAffinity.Mask);
    }
}

int main(int, char **) {

    for (int i = 0; i < 7; i++) {
        setThreadStart(i);
        printf("\n-------\n");
    }
    return 0;
}
