#include <windows.h>
#include <stdio.h>
#include <intrin.h>
#include <string.h>

#define MAX_CORES 256
#define CORES_PER_CCD 8
#define MAX_CCDS (MAX_CORES / CORES_PER_CCD)

typedef struct {
    DWORD cpuId;
    int coreId;
    int ccdId;
    int threadIndex; // Inferred thread ID within the core
} CoreInfo;

int main() {
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    DWORD numCpus = sysInfo.dwNumberOfProcessors;
    int threadsPerCore = 0;

    CoreInfo cores[MAX_CORES];
    int coreId;
    int coresPerCcd[MAX_CORES / CORES_PER_CCD] = {0};
    int coreThreadMap[MAX_CORES] = {0};

    //
    // Compute the base information for each processor.
    //

    for (DWORD i = 0; i < numCpus; ++i) {
        DWORD_PTR affinityMask = (1ULL << i);
        HANDLE thread = GetCurrentThread();
        DWORD_PTR prevAffinity = SetThreadAffinityMask(thread, affinityMask);

        int cpuInfo[4] = {0};
        __cpuidex(cpuInfo, 0x8000001E, 0); // AMD topology leaf

        coreId = cpuInfo[1] & 0xFF;        // EBX[7:0]: Core ID

        //
        // Make sure the threads per core is the same for all processors.
        //

        if (!threadsPerCore) {
            threadsPerCore = ((cpuInfo[1] >> 8) & 0xFF) + 1; // EBX[15:8] + 1

        } else {
            if (threadsPerCore != ((cpuInfo[1] >> 8) & 0xFF) + 1) {
                printf("****** error - threads per core mismatch\n");
            }
        }

        CoreInfo info;
        info.cpuId = i;
        info.coreId = coreId;
        info.threadIndex = coreThreadMap[coreId]++;
        cores[i] = info;

        SetThreadAffinityMask(thread, prevAffinity);
    }

    //
    // Display the base information.
    //

    BOOLEAN classic = TRUE;
    int lastId = cores[0].coreId;

    printf("Zen 5 Core Type Base Information\n\n");
    for (DWORD i = 0; i < numCpus; i += 1) {
        coreId = cores[i].coreId;
        if ((coreId != lastId) && (coreId != (lastId + threadsPerCore - 1))) {
            classic = FALSE;
        }

        printf("Cpu Id: %2lu, Core Id: %2d, SMT Thread: %1d, Inferred type %s\n",
               cores[i].cpuId,
               coreId,
               cores[i].threadIndex,
               classic ? "classic" : "dense");

        lastId = coreId;
    }

    printf("\n");

    //
    // Compute the number of cores per CCD.
    //

    int baseId = cores[0].coreId;
    int ccdId = 0;

    for (DWORD i = 0; i < numCpus; ++i) {
        ccdId = (cores[i].coreId - baseId) / CORES_PER_CCD;
        cores[i].ccdId = ccdId;
        coresPerCcd[ccdId] += 1;
    }

    //
    // Display CCD, cores/threads, information.
    //

    printf("Zen 5 Cores/SMT Threads Per CCD Information\n\n");
    for (int i = 0; i < MAX_CCDS; i += 1) {
        if (!coresPerCcd[i]) {
            break;
        }

        ccdId = ((i * CORES_PER_CCD) + baseId) / CORES_PER_CCD;
        printf("CCD Id: %2d, Cores: %2d, SMT Threads: %2d\n",
                ccdId,
                coresPerCcd[i] / threadsPerCore,
                coresPerCcd[i]);
    }

    printf("\n");

    return 0;
}
