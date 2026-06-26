// bwtest.c - memory read-bandwidth vs thread placement on a chiplet CPU.
// Mirrors llama.cpp decode: N threads each stream a DISTINCT slice of a big
// (>>cache) buffer; aggregate GB/s = bytes / wall-time. We compare placement:
//   0 floating       - no affinity (OS default, all groups)
//   1 single-CCX     - all threads pinned into one CCX (one fabric link)
//   2 group0-spread  - one thread per CCX, but only CCXs in processor group 0
//   3 all-CCX spread - one thread per CCX across ALL groups (group-aware)
//
// Build: cl /O2 /D_WIN32_WINNT=0x0601 bwtest.c
// Run:   bwtest.exe [n_threads] [buf_GiB] [iters]

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// ---- CCX map (from L3 cache topology) ----
typedef struct { WORD group; KAFFINITY mask; int base_lp; int n_cores; } ccx_t;
static ccx_t g_ccx[64];
static int   g_nccx = 0;

static void build_ccx_map(void) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, NULL, &len);
    BYTE *buf = (BYTE *)malloc(len);
    GetLogicalProcessorInformationEx(RelationCache, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len);
    BYTE *p = buf;
    while (p < buf + len) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)p;
        if (info->Relationship == RelationCache && info->Cache.Level == 3) {
            GROUP_AFFINITY ga = info->Cache.GroupMask;
            ccx_t *c = &g_ccx[g_nccx++];
            c->group = ga.Group;
            c->mask  = ga.Mask;
            int base = -1, cores = 0;
            for (int b = 0; b < 64; b++) if (ga.Mask & (1ull << b)) { if (base < 0) base = b; cores++; }
            c->base_lp = base;
            c->n_cores = cores / 2; // SMT pairs
        }
        p += info->Size;
    }
    free(buf);
}

// ---- worker ----
typedef struct {
    const uint64_t *base;   // slice start
    size_t          words;  // slice length in uint64
    int             iters;
    int             mode;
    int             ith;
    int             nthr;
    HANDLE          start_ev;
    volatile uint64_t sink;
} work_t;

static void pin(int mode, int ith, int nthr) {
    if (mode == 0) return; // floating
    GROUP_AFFINITY ga; ZeroMemory(&ga, sizeof(ga));
    if (mode == 1) {                       // single CCX: distinct cores in ccx0
        ccx_t *c = &g_ccx[0];
        int lp = c->base_lp + 2 * (ith % c->n_cores);
        ga.Group = c->group; ga.Mask = 1ull << lp;
    } else {                               // spread: one thread per CCX
        int nsel = g_nccx, sel[64], ns = 0;
        if (mode == 2) { for (int i = 0; i < g_nccx; i++) if (g_ccx[i].group == 0) sel[ns++] = i; }
        else           { for (int i = 0; i < g_nccx; i++) sel[ns++] = i; }
        nsel = ns;
        int ci   = sel[ith % nsel];
        int core = (ith / nsel);           // stack onto next core if nthr > nsel
        ccx_t *c = &g_ccx[ci];
        int lp = c->base_lp + 2 * (core % c->n_cores);
        ga.Group = c->group; ga.Mask = 1ull << lp;
    }
    if (!SetThreadGroupAffinity(GetCurrentThread(), &ga, NULL))
        fprintf(stderr, "  [ith %d] SetThreadGroupAffinity g=%u mask=0x%llx FAILED err=%lu\n",
                ith, ga.Group, (unsigned long long)ga.Mask, GetLastError());
}

static DWORD WINAPI worker(LPVOID arg) {
    work_t *w = (work_t *)arg;
    pin(w->mode, w->ith, w->nthr);
    WaitForSingleObject(w->start_ev, INFINITE);
    uint64_t a0=0,a1=0,a2=0,a3=0;
    const uint64_t *b = w->base;
    size_t n = w->words;
    for (int it = 0; it < w->iters; it++) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) { a0+=b[i]; a1+=b[i+1]; a2+=b[i+2]; a3+=b[i+3]; }
        for (; i < n; i++) a0 += b[i];
    }
    w->sink = a0+a1+a2+a3;
    return 0;
}

static double run_mode(int mode, const char *name, uint64_t *buf, size_t words,
                       int nthr, int iters) {
    work_t   w[256]; HANDLE th[256];
    HANDLE start_ev = CreateEvent(NULL, TRUE, FALSE, NULL);
    size_t per = words / nthr;
    for (int i = 0; i < nthr; i++) {
        w[i].base = buf + (size_t)i * per;
        w[i].words = (i == nthr - 1) ? (words - (size_t)i * per) : per;
        w[i].iters = iters; w[i].mode = mode; w[i].ith = i; w[i].nthr = nthr;
        w[i].start_ev = start_ev; w[i].sink = 0;
        th[i] = CreateThread(NULL, 0, worker, &w[i], 0, NULL);
    }
    Sleep(50);
    LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    SetEvent(start_ev);
    WaitForMultipleObjects(nthr, th, TRUE, INFINITE);
    QueryPerformanceCounter(&t1);
    for (int i = 0; i < nthr; i++) CloseHandle(th[i]);
    CloseHandle(start_ev);
    double secs  = (double)(t1.QuadPart - t0.QuadPart) / (double)f.QuadPart;
    double bytes = (double)words * 8.0 * (double)iters;
    double gbps  = bytes / secs / 1e9;
    double ms_pass = secs / iters * 1000.0;
    printf("  %-14s  %7.1f GB/s   %8.2f ms/pass   (%.2fs)\n", name, gbps, ms_pass, secs);
    return gbps;
}

int main(int argc, char **argv) {
    int    nthr  = argc > 1 ? atoi(argv[1]) : 8;
    double gib   = argc > 2 ? atof(argv[2]) : 2.0;
    int    iters = argc > 3 ? atoi(argv[3]) : 6;

    if (nthr < 1) nthr = 1;
    // WaitForMultipleObjects caps at MAXIMUM_WAIT_OBJECTS (64) handles.
    if (nthr > MAXIMUM_WAIT_OBJECTS) {
        printf("note: clamping nthr %d -> %d (WaitForMultipleObjects limit)\n", nthr, MAXIMUM_WAIT_OBJECTS);
        nthr = MAXIMUM_WAIT_OBJECTS;
    }

    build_ccx_map();
    printf("CCX map: %d CCXs across groups; nthr=%d  buf=%.1f GiB  iters=%d\n",
           g_nccx, nthr, gib, iters);
    for (int i = 0; i < g_nccx; i++)
        printf("  CCX %2d: group=%u base_lp=%d cores=%d mask=0x%016llx\n",
               i, g_ccx[i].group, g_ccx[i].base_lp, g_ccx[i].n_cores,
               (unsigned long long)g_ccx[i].mask);

    size_t bytes = (size_t)(gib * (1024.0*1024.0*1024.0));
    size_t words = bytes / 8;
    uint64_t *buf = (uint64_t *)VirtualAlloc(NULL, words*8, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    if (!buf) { printf("alloc failed\n"); return 1; }
    for (size_t i = 0; i < words; i++) buf[i] = i * 2654435761u; // first-touch + fill

    printf("\nplacement         aggregate     per-pass\n");
    // run twice each to expose variance; floating first to warm
    for (int rep = 0; rep < 2; rep++) {
        printf("-- pass %d --\n", rep+1);
        run_mode(0, "floating",      buf, words, nthr, iters);
        run_mode(1, "single-CCX",    buf, words, nthr, iters);
        run_mode(2, "group0-spread", buf, words, nthr, iters);
        run_mode(3, "allCCX-spread", buf, words, nthr, iters);
    }
    VirtualFree(buf, 0, MEM_RELEASE);
    return 0;
}
