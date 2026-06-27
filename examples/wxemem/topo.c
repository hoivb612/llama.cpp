#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static void print_mask(const char *label, KAFFINITY mask, WORD group) {
    printf("    %s group=%u mask=0x%016llx LPs=[", label, (unsigned)group, (unsigned long long)mask);
    int first = 1;
    for (int b = 0; b < 64; b++) {
        if (mask & (1ull << b)) {
            printf("%s%d", first ? "" : ",", b);
            first = 0;
        }
    }
    printf("]\n");
}

static BYTE *get_info(LOGICAL_PROCESSOR_RELATIONSHIP rel, DWORD *outLen) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(rel, NULL, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) return NULL;
    BYTE *buf = (BYTE *)malloc(len);
    if (!GetLogicalProcessorInformationEx(rel, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len)) {
        free(buf);
        return NULL;
    }
    *outLen = len;
    return buf;
}

static void walk(LOGICAL_PROCESSOR_RELATIONSHIP rel, const char *title) {
    DWORD len = 0;
    BYTE *buf = get_info(rel, &len);
    if (!buf) { printf("%s: query failed (err=%lu)\n", title, GetLastError()); return; }
    printf("=== %s ===\n", title);
    BYTE *p = buf;
    int idx = 0;
    while (p < buf + len) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)p;
        if (info->Relationship == RelationProcessorCore) {
            printf("  Core %d (flags=0x%x, EfficiencyClass=%u, groups=%u):\n",
                   idx, info->Processor.Flags, info->Processor.EfficiencyClass,
                   info->Processor.GroupCount);
            for (int g = 0; g < info->Processor.GroupCount; g++)
                print_mask("core", info->Processor.GroupMask[g].Mask, info->Processor.GroupMask[g].Group);
            idx++;
        } else if (info->Relationship == RelationCache) {
            CACHE_RELATIONSHIP *c = &info->Cache;
            const char *t = c->Type == CacheUnified ? "U" : c->Type == CacheInstruction ? "I" :
                            c->Type == CacheData ? "D" : "T";
            printf("  L%u %s cache, %u KB:\n", c->Level, t, c->CacheSize / 1024);
            // GroupMask is a single GROUP_AFFINITY in older headers; GroupCount in newer
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0A00)
            for (int g = 0; g < (c->GroupCount ? c->GroupCount : 1); g++)
                print_mask("cache", c->GroupMasks ? c->GroupMasks[g].Mask : c->GroupMask.Mask,
                           c->GroupMasks ? c->GroupMasks[g].Group : c->GroupMask.Group);
#else
            print_mask("cache", c->GroupMask.Mask, c->GroupMask.Group);
#endif
        } else if (info->Relationship == RelationNumaNode) {
            printf("  NUMA node %lu:\n", info->NumaNode.NodeNumber);
            print_mask("numa", info->NumaNode.GroupMask.Mask, info->NumaNode.GroupMask.Group);
        } else if (info->Relationship == RelationGroup) {
            printf("  Groups: ActiveGroupCount=%u\n", info->Group.ActiveGroupCount);
            for (int g = 0; g < info->Group.ActiveGroupCount; g++)
                printf("    group %d: active=%u max=%u mask=0x%016llx\n", g,
                       info->Group.GroupInfo[g].ActiveProcessorCount,
                       info->Group.GroupInfo[g].MaximumProcessorCount,
                       (unsigned long long)info->Group.GroupInfo[g].ActiveProcessorMask);
        }
        p += info->Size;
    }
    printf("\n");
    free(buf);
}

static int popcount64(KAFFINITY m) {
    int n = 0;
    for (int b = 0; b < 64; b++) if (m & (1ull << b)) n++;
    return n;
}

static void print_summary(void) {
    DWORD len = 0;
    BYTE *p;

    // L3 cache domains == CCXs; record LPs (threads) sharing each L3.
    int ccx_threads[256];
    int nccx = 0;
    BYTE *buf = get_info(RelationCache, &len);
    if (buf) {
        p = buf;
        while (p < buf + len) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)p;
            if (info->Relationship == RelationCache && info->Cache.Level == 3 && nccx < 256) {
                ccx_threads[nccx++] = popcount64(info->Cache.GroupMask.Mask);
            }
            p += info->Size;
        }
        free(buf);
    }

    // physical cores (one RelationProcessorCore entry each)
    int ncores = 0;
    buf = get_info(RelationProcessorCore, &len);
    if (buf) {
        p = buf;
        while (p < buf + len) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)p;
            if (info->Relationship == RelationProcessorCore) ncores++;
            p += info->Size;
        }
        free(buf);
    }

    // NUMA nodes
    int nnuma = 0;
    buf = get_info(RelationNumaNode, &len);
    if (buf) {
        p = buf;
        while (p < buf + len) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)p;
            if (info->Relationship == RelationNumaNode) nnuma++;
            p += info->Size;
        }
        free(buf);
    }

    int ngroups = (int)GetActiveProcessorGroupCount();
    int nlp     = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);

    printf("=== SUMMARY ===\n");
    if (nccx > 0) {
        int uniform = 1;
        for (int i = 1; i < nccx; i++) if (ccx_threads[i] != ccx_threads[0]) { uniform = 0; break; }
        if (uniform) {
            int t = ccx_threads[0];
            printf("  %d CCX(s) x %d cores (%d threads) each  [cores assume SMT pairs]\n",
                   nccx, t / 2, t);
        } else {
            printf("  %d CCX(s), mixed sizes:\n", nccx);
            for (int i = 0; i < nccx; i++)
                printf("    CCX %2d: %d cores (%d threads)\n", i, ccx_threads[i] / 2, ccx_threads[i]);
        }
    } else {
        printf("  no L3/CCX topology found\n");
    }
    printf("  %d physical cores, %d logical processors\n", ncores, nlp);
    printf("  %d processor group(s), %d NUMA node(s)\n", ngroups, nnuma);
    if (nccx > 0)
        printf("  one-thread-per-CCX spread: ccxrun -c 1 <cmd>  (= -n %d on this box)\n", nccx);
    printf("\n");
}

int main(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    printf("ActiveProcessorGroups=%u, total logical=%lu\n\n",
           GetActiveProcessorGroupCount(), GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));

    print_summary();

    walk(RelationGroup, "PROCESSOR GROUPS");
    walk(RelationProcessorCore, "PHYSICAL CORES (each = 1 core, mask shows its SMT LPs)");
    walk(RelationNumaNode, "NUMA NODES");
    walk(RelationCache, "CACHES (L3 mask shows the CCX/LP sharing the cache)");
    return 0;
}
