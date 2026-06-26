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

int main(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    printf("ActiveProcessorGroups=%u, total logical=%lu\n\n",
           GetActiveProcessorGroupCount(), GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));

    walk(RelationGroup, "PROCESSOR GROUPS");
    walk(RelationProcessorCore, "PHYSICAL CORES (each = 1 core, mask shows its SMT LPs)");
    walk(RelationNumaNode, "NUMA NODES");
    walk(RelationCache, "CACHES (L3 mask shows the CCX/LP sharing the cache)");
    return 0;
}
