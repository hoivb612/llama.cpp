/* usampler.c - admin-free user-mode sampling profiler
 *
 * Launches a child process, periodically suspends each of its threads, 
 * reads the instruction pointer (RIP), resumes, records the sample,
 * then symbolizes against the child's modules/PDBs and prints the hottest
 * modules and functions. No elevation required (it only touches a process it
 * launched itself).
 *
 * Usage: usampler.exe [options] <child.exe> [child args...]
 *   --out FILE       write the PROFILE section to FILE instead of stdout so
 *                    child stdout remains clean.
 *   --sleep-us N     target inter-sweep interval in microseconds (default 1000).
 *                    Values < 1000 spin on QueryPerformanceCounter for a higher,
 *                    steadier sample rate; >= 1000 uses Sleep(N/1000).
 *   --skip-ms N      discard samples during the first N ms of the child's life
 *                    (default 0). Use to drop model-load + first-prefill so the
 *                    profile is biased toward steady-state DECODE.
 *   --               end of usampler options; the rest is the child command.
 */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <tlhelp32.h>
#include <dbghelp.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#pragma comment(lib, "dbghelp.lib")
#endif

/* ---- types --------------------------------------------------------------- */
typedef struct { DWORD64 base; DWORD64 size; char name[64]; char path[MAX_PATH]; } ModInfo;
typedef struct { DWORD64 addr; unsigned int count; } Hit;         /* count==0 => empty */
typedef struct { char name[256]; ULONG64 count; } NameCount;

/* ---- globals ------------------------------------------------------------- */
static FILE*    g_out    = NULL;
static int      g_suppressStdout = 0;
static DWORD    g_pid    = 0;
static ModInfo* g_mods   = NULL;  
static size_t g_mn = 0, g_mcap = 0;
/* Cache of the target's thread handles (see refreshThreads). */
typedef struct { DWORD tid; HANDLE h; int seen; } TgtThread;
static TgtThread* g_thr = NULL;  
static size_t g_tn = 0, g_tcap = 0;
static Hit*     g_hits   = NULL;  
static size_t g_hcap = 0, g_hcount = 0;
static long     g_sleepUs = 1000;
static double   g_qpcToUs = 0.0;

typedef UINT(WINAPI* TimePeriodFn)(UINT period);

static TimePeriodFn g_timeBeginPeriod = NULL;
static TimePeriodFn g_timeEndPeriod = NULL;
static int g_timePeriodInitialized = 0;

static void initTimePeriodFunctions(void) {
    HMODULE module;
    if (g_timePeriodInitialized) return;
    g_timePeriodInitialized = 1;

    module = LoadLibraryA("winmm.dll");
    if (!module) return;

    g_timeBeginPeriod = (TimePeriodFn)GetProcAddress(module, "timeBeginPeriod");
    g_timeEndPeriod = (TimePeriodFn)GetProcAddress(module, "timeEndPeriod");
}

static int beginTimerPeriod(UINT period) {
    initTimePeriodFunctions();
    return g_timeBeginPeriod && g_timeEndPeriod && g_timeBeginPeriod(period) == 0;
}

static void endTimerPeriod(UINT period) {
    if (g_timeEndPeriod) g_timeEndPeriod(period);
}

/* Substrings of module!symbol names that represent IDLE / scheduling wait,
 * excluded from the "compute-only" ranking so decode kernels stand out. */
static const char* kIdleNeedles[] = {
    "ZwWaitForWorkViaWorkerFactory", "NtWaitForWorkViaWorkerFactory",
    "ZwDelayExecution", "NtDelayExecution", "RtlDelayExecution",
    "ZwWaitForSingleObject", "NtWaitForSingleObject",
    "ZwWaitForAlertByThreadId", "NtWaitForAlertByThreadId",
    "RtlWaitOnAddress", "WaitForSingleObject", "SwitchToThread",
    "ggml_wait_for_done_xbox",  /* b612 threadpool spin-wait */
    "ggml_barrier",             /* threadpool barrier spin   */
    "vcomp_barrier"             /* OpenMP barrier            */
};

/* ---- profile output: file when --out is active, otherwise stdout --------- */
static void emit(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (g_out) {
        vfprintf(g_out, fmt, ap);
    } else if (g_suppressStdout) {
        vfprintf(stderr, fmt, ap);
    } else {
        vprintf(fmt, ap);
    }
    va_end(ap);
}

static void status(const char* fmt, ...) {
    FILE* stream = g_suppressStdout ? stderr : stdout;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stream, fmt, ap);
    va_end(ap);
}

static int isIdle(const char* fn) {
    size_t i, n = sizeof(kIdleNeedles) / sizeof(kIdleNeedles[0]);
    for (i = 0; i < n; i++)
        if (strstr(fn, kIdleNeedles[i]) != NULL) return 1;
    return 0;
}

/* ---- RIP hit hash table (open addressing) -------------------------------- */
static void hits_grow(void) {
    size_t old = g_hcap, i, j, mask;
    Hit* oldh = g_hits;
    g_hcap = old ? old * 2 : 4096;
    g_hits = (Hit*)calloc(g_hcap, sizeof(Hit));
    mask = g_hcap - 1;
    for (i = 0; i < old; i++) {
        if (oldh[i].count) {
            DWORD64 a = oldh[i].addr;
            j = (size_t)((a * (DWORD64)1099511628211ULL) & mask);
            for (;;) {
                if (g_hits[j].count == 0) { g_hits[j] = oldh[i]; break; }
                j = (j + 1) & mask;
            }
        }
    }
    free(oldh);
}

static void hits_add(DWORD64 a) {
    size_t i, mask;
    if (g_hcap == 0) hits_grow();
    if ((g_hcount + 1) * 10 >= g_hcap * 7) hits_grow();
    mask = g_hcap - 1;
    i = (size_t)((a * (DWORD64)1099511628211ULL) & mask);
    for (;;) {
        if (g_hits[i].count == 0) { g_hits[i].addr = a; g_hits[i].count = 1; g_hcount++; return; }
        if (g_hits[i].addr == a)  { g_hits[i].count++; return; }
        i = (i + 1) & mask;
    }
}

/* ---- name -> count aggregation ------------------------------------------- */
static void nc_add(NameCount** arr, size_t* n, size_t* cap, const char* name, ULONG64 add) {
    size_t i;
    for (i = 0; i < *n; i++) {
        if (strcmp((*arr)[i].name, name) == 0) { (*arr)[i].count += add; return; }
    }
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 64;
        *arr = (NameCount*)realloc(*arr, *cap * sizeof(NameCount));
    }
    strncpy((*arr)[*n].name, name, 255);
    (*arr)[*n].name[255] = 0;
    (*arr)[*n].count = add;
    (*n)++;
}

static int nc_cmp(const void* a, const void* b) {
    const NameCount* x = (const NameCount*)a;
    const NameCount* y = (const NameCount*)b;
    if (x->count < y->count) return 1;
    if (x->count > y->count) return -1;
    return 0;
}

/* ---- module tracking ----------------------------------------------------- */
static void refreshMods(void) {
    HANDLE snap;
    MODULEENTRY32 me;
    int attempt;
    /* CreateToolhelp32Snapshot(TH32CS_SNAPMODULE) can transiently fail with
     * ERROR_PARTIAL_COPY (299) / ERROR_BAD_LENGTH (24) while the target is still
     * loading DLLs or its loader list is in flux, so retry briefly. */
    for (attempt = 0; attempt < 10; attempt++) {
        snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, g_pid);
        if (snap == INVALID_HANDLE_VALUE) {
            DWORD e = GetLastError();
            if (e == ERROR_BAD_LENGTH || e == ERROR_PARTIAL_COPY) { 
                Sleep(1); 
                continue; 
            }
            return;
        }
        me.dwSize = sizeof(me);
        if (Module32First(snap, &me)) {
            do {
                DWORD64 base = (DWORD64)(ULONG_PTR)me.modBaseAddr;
                size_t k; int found = 0;
                for (k = 0; k < g_mn; k++) {
                    if (g_mods[k].base == base) { 
                        found = 1; break; 
                    }
                }
                if (!found) {
                    if (g_mn == g_mcap) {
                        g_mcap = g_mcap ? g_mcap * 2 : 64;
                        g_mods = (ModInfo*)realloc(g_mods, g_mcap * sizeof(ModInfo));
                    }
                    g_mods[g_mn].base = base;
                    g_mods[g_mn].size = me.modBaseSize;
                    strncpy(g_mods[g_mn].name, me.szModule, sizeof(g_mods[g_mn].name) - 1);
                    g_mods[g_mn].name[sizeof(g_mods[g_mn].name) - 1] = 0;
                    strncpy(g_mods[g_mn].path, me.szExePath, sizeof(g_mods[g_mn].path) - 1);
                    g_mods[g_mn].path[sizeof(g_mods[g_mn].path) - 1] = 0;
                    g_mn++;
                }
            } while (Module32Next(snap, &me));
        }
        CloseHandle(snap);
        return;
    }
}

/* Rebuild the target's thread handle cache. TH32CS_SNAPTHREAD is system-wide
 * (no per-pid filter), so snapshotting + walking every thread on the box each
 * sweep is O(all-system-threads) and collapses on big machines (e.g. a 96-core
 * host with 20k+ threads across 400+ processes): the sample loop crawls and
 * module capture starves. So we rebuild this cache on a coarse cadence and reuse
 * the handles for the tight per-sweep suspend/sample. */
static void refreshThreads(void) {
    HANDLE tsnap;
    THREADENTRY32 te;
    size_t k, w;
    for (k = 0; k < g_tn; k++) {
        g_thr[k].seen = 0;
    }

    tsnap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (tsnap == INVALID_HANDLE_VALUE) {
        return;
    }

    te.dwSize = sizeof(te);
    if (Thread32First(tsnap, &te)) {
        do {
            int have = 0;
            if (te.th32OwnerProcessID != g_pid) {
                continue;
            }

            for (k = 0; k < g_tn; k++) {
                if (g_thr[k].tid == te.th32ThreadID) { 
                    g_thr[k].seen = 1; have = 1; break; 
                }
            }
            if (!have) {
                HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT,
                                      FALSE, te.th32ThreadID);
                if (h) {
                    if (g_tn == g_tcap) {
                        g_tcap = g_tcap ? g_tcap * 2 : 64;
                        g_thr = (TgtThread*)realloc(g_thr, g_tcap * sizeof(TgtThread));
                    }
                    g_thr[g_tn].tid = te.th32ThreadID;
                    g_thr[g_tn].h = h;
                    g_thr[g_tn].seen = 1;
                    g_tn++;
                }
            }
        } while (Thread32Next(tsnap, &te));
    }
    CloseHandle(tsnap);
    /* prune handles for threads that have exited */
    w = 0;
    for (k = 0; k < g_tn; k++) {
        if (g_thr[k].seen) { 
            if (w != k) {
                g_thr[w] = g_thr[k]; w++; 
            }

        } else {
            CloseHandle(g_thr[k].h);
        }
    }
    g_tn = w;
}

static int modFor(DWORD64 a) {
    size_t k;
    for (k = 0; k < g_mn; k++)
        if (a >= g_mods[k].base && a < g_mods[k].base + g_mods[k].size) {
            return (int)k;
        }
    return -1;
}

/* ---- high-resolution inter-sweep wait ------------------------------------ */
static void waitInterval(void) {
    LARGE_INTEGER a, b;
    if (g_sleepUs >= 1000) { 
        Sleep((DWORD)(g_sleepUs / 1000)); return; 
    }
    QueryPerformanceCounter(&a);
    for (;;) {
        QueryPerformanceCounter(&b);
        if ((double)(b.QuadPart - a.QuadPart) * g_qpcToUs >= (double)g_sleepUs) {
            break;
        }
        YieldProcessor();
    }
}

/* ---- ranking dumps ------------------------------------------------------- */
static void dumpNC(NameCount* arr, size_t n, double denom, int skipIdle,
                   const char* title, size_t topN) {
    size_t i, shown = 0;
    qsort(arr, n, sizeof(NameCount), nc_cmp);
    emit("\n%s\n", title);

    double cum_percent = 0.0;
    for (i = 0; i < n && shown < topN; i++) {
        if (skipIdle && isIdle(arr[i].name)) {
            continue;
        }

        double percent = 100.0 * (double)arr[i].count / denom;
        cum_percent += percent;

        //
        // Look for module name suffix of either ".dll" or ".exe".
        //

        char * type = strstr(arr[i].name, ".dll!");
        if (!type) {
            type = strstr(arr[i].name, ".exe!");
        }

        //
        // If no file type, then use full name. Otherwise, trim out the type to
        // save line space.
        //

        if (!type) {
            emit("%6.2f%% %6.2f%% %8llu %s\n", percent, cum_percent,
                 (ULONG64)arr[i].count, arr[i].name);

        } else {
            *type = '\0';
            emit("%6.2f%% %6.2f%% %8llu %s%s\n", percent, cum_percent,
                 (ULONG64)arr[i].count, arr[i].name, type + 4);

            *type = '.';
        }

        shown++;
    }
}

int main(int argc, char** argv) {
    const char* outPath = NULL;
    long skipMs = 0;
    int  ci = 1;
    char* cmd = NULL; size_t cmdlen = 0, cmdcap = 0;
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    LARGE_INTEGER qpf, tStart;
    HANDLE hCur;
    char symbuf[sizeof(SYMBOL_INFO) + 1024];
    NameCount* byFunc = NULL; size_t nFunc = 0, capFunc = 0;
    NameCount* byMod  = NULL; size_t nMod  = 0, capMod  = 0;
    ULONG64 total = 0, skipped = 0, idleSamples = 0, computeSamples = 0;
    int warmupDone;
    double lastModMs, lastThrMs;
    int timerPeriodActive = 0;
    size_t i;

    /* ---- parse usampler options up to the child exe -------------------- */
    for (; ci < argc; ci++) {
        const char* a = argv[ci];
        if (strcmp(a, "--") == 0) { 
            ci++; 
            break; 
        }

        else if (strcmp(a, "--out") == 0 && ci + 1 < argc) { 
            outPath = argv[++ci]; 

        } else if (strcmp(a, "--sleep-us") == 0 && ci + 1 < argc) { 
            g_sleepUs = atol(argv[++ci]); 

        } else if (strcmp(a, "--skip-ms") == 0 && ci + 1 < argc)  { 
            skipMs = atol(argv[++ci]); 
        } else if (a[0] == '-' && a[1] == '-' && a[2] != 0) {
            status("[usampler] unknown option: %s\n", a); return 1;
        }
        else break;
    }
    if (ci >= argc) {
        status("usage: usampler [--out FILE] [--sleep-us N] [--skip-ms N] <child.exe> [args...]\n");
        return 1;
    }

    if (g_sleepUs < 1) {
        g_sleepUs = 1;
    }

    /* build a quoted command line from the child args */
    for (i = (size_t)ci; i < (size_t)argc; i++) {
        const char* a = argv[i];
        int quote = (strchr(a, ' ') != NULL && a[0] != '"');
        size_t need = cmdlen + strlen(a) + 4;

        if (need > cmdcap) { 
            cmdcap = need * 2; 
            cmd = (char*)realloc(cmd, cmdcap); 
        }

        if (cmdlen) {
            cmd[cmdlen++] = ' ';
        }
    
        if (quote) {
            cmd[cmdlen++] = '"';
        }
        memcpy(cmd + cmdlen, a, strlen(a)); cmdlen += strlen(a);
        if (quote) cmd[cmdlen++] = '"';
        cmd[cmdlen] = 0;
    }

    if (outPath) {
        g_suppressStdout = 1;
        g_out = fopen(outPath, "w");
        if (!g_out) status("[usampler] WARNING: cannot open --out %s\n", outPath);
    }

    timerPeriodActive = beginTimerPeriod(1);
    QueryPerformanceFrequency(&qpf);
    g_qpcToUs = 1e6 / (double)qpf.QuadPart;

    status("[usampler] launching: %s\n", cmd);
    status("[usampler] sleep-us=%ld  skip-ms=%ld  out=%s\n",
           g_sleepUs, skipMs, outPath ? outPath : "(none)");

    ZeroMemory(&si, sizeof(si)); si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));
    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE,
                        CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
        status("[usampler] CreateProcess failed: %lu\n", GetLastError());
        if (timerPeriodActive) {
            endTimerPeriod(1);
        }
        return 1;
    }
    g_pid = pi.dwProcessId;
    QueryPerformanceCounter(&tStart);
    ResumeThread(pi.hThread);

    warmupDone = (skipMs <= 0);
    lastModMs = -1e9;
    lastThrMs = -1e9;

    while (WaitForSingleObject(pi.hProcess, 0) == WAIT_TIMEOUT) {
        LARGE_INTEGER now;
        double ms;
        size_t k;
        QueryPerformanceCounter(&now);
        ms = (double)(now.QuadPart - tStart.QuadPart) * g_qpcToUs / 1000.0;

        /* Time-based cadences instead of per-iteration system-wide snapshots.
         * Module list catches delay-loaded DLLs; thread cache stays small. */
        if (ms - lastModMs >= 250.0) { 
            refreshMods();    lastModMs = ms; 
        }
        if (ms - lastThrMs >= 500.0) { 
            refreshThreads(); lastThrMs = ms; 
        }

        if (!warmupDone && ms >= (double)skipMs) {
            warmupDone = 1;
        }

        for (k = 0; k < g_tn; k++) {
            CONTEXT ctx;
            if (SuspendThread(g_thr[k].h) == (DWORD)-1) {
                continue;
            }
            ZeroMemory(&ctx, sizeof(ctx));
            ctx.ContextFlags = CONTEXT_CONTROL;
            if (GetThreadContext(g_thr[k].h, &ctx)) {
                if (warmupDone) { 
                    hits_add(ctx.Rip); total++; 
                } else { 
                    skipped++; 
                }
            }
            ResumeThread(g_thr[k].h);
        }
        waitInterval();
    }
    refreshMods();
    {
        size_t k;
        for (k = 0; k < g_tn; k++) {
            CloseHandle(g_thr[k].h);
        }
        free(g_thr); 
        g_thr = NULL; 
        g_tcap = 0;
        g_tn = 0;
    }
    if (timerPeriodActive) endTimerPeriod(1);

    emit("[usampler] child exited. total samples = %llu (warmup-skipped = %llu) across %zu modules.\n",
         (ULONG64)total, (ULONG64)skipped, g_mn);
    if (total == 0) { 
        emit("[usampler] no samples collected.\n"); if (g_out) fclose(g_out); return 0; 
    }

    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    hCur = GetCurrentProcess();
    if (!SymInitialize(hCur, NULL, FALSE)) {
        status("[usampler] SymInitialize failed %lu\n", GetLastError());
    }
    for (i = 0; i < g_mn; i++)
        SymLoadModuleEx(hCur, NULL, g_mods[i].path, g_mods[i].name,
                        g_mods[i].base, (DWORD)g_mods[i].size, NULL, 0);

    for (i = 0; i < g_hcap; i++) {
        DWORD64 a; unsigned int c; 
        int mi; 
        const char* modName;
        char fn[512];
        SYMBOL_INFO* sym;
        DWORD64 disp = 0;
        if (!g_hits[i].count) {
            continue;
        }
        a = g_hits[i].addr; 
        c = g_hits[i].count;
        mi = modFor(a);
        modName = (mi >= 0) ? g_mods[mi].name : "[kernel/unknown]";
        nc_add(&byMod, &nMod, &capMod, modName, c);

        sym = (SYMBOL_INFO*)symbuf;
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 1024;
        if (mi >= 0 && SymFromAddr(hCur, a, &disp, sym)) {
            int flen = _snprintf(fn, sizeof(fn), "%s!%s", modName, sym->Name);
            if (flen < 0 || flen >= (int)sizeof(fn)) {
                flen = (int)sizeof(fn) - 1;
            }
            /* Append the function's definition file:line so same-named symbols
             * from different translation units (e.g. vec.c vs b612/vec-b612.c,
             * or per-TU static helpers) can be told apart. Resolve at the SYMBOL
             * START address (sym->Address), not the sampled address, so every
             * sample inside one function shares the same tag and the histogram
             * stays aggregated per-function instead of per-line. */
            {
                IMAGEHLP_LINE64 li; DWORD lineDisp = 0;
                memset(&li, 0, sizeof(li));
                li.SizeOfStruct = sizeof(li);
                if (SymGetLineFromAddr64(hCur, sym->Address, &lineDisp, &li) && li.FileName) {
                    const char* bs = strrchr(li.FileName, '\\');
                    const char* fs = strrchr(li.FileName, '/');
                    const char* base = (fs > bs ? fs : bs);
                    base = base ? base + 1 : li.FileName;
                    _snprintf(fn + flen, sizeof(fn) - flen, "  (%s:%lu)",
                              base, (unsigned long)li.LineNumber);
                }
            }
        } else {
            _snprintf(fn, sizeof(fn), "%s+0x%llx", modName,
                      (ULONG64)(mi >= 0 ? (a - g_mods[mi].base) : a));
        }
        fn[sizeof(fn) - 1] = 0;
        nc_add(&byFunc, &nFunc, &capFunc, fn, c);
    }

    for (i = 0; i < nFunc; i++) {
        if (isIdle(byFunc[i].name)) {
            idleSamples += byFunc[i].count;
        }
    }

    computeSamples = total - idleSamples;

    emit("\n================ PROFILE (total samples = %llu) ================\n",
         (ULONG64)total);
    emit("  idle/scheduler = %6.2f%% (%llu)   compute = %6.2f%% (%llu)\n",
         100.0 * (double)idleSamples / (double)total,    (ULONG64)idleSamples,
         100.0 * (double)computeSamples / (double)total, (ULONG64)computeSamples);

    dumpNC(byMod, nMod, (double)total, 0, "-- top modules by self-samples (all) --", 15);
    dumpNC(byFunc, nFunc, (double)total, 0, "-- top 45 functions by self-samples (all, incl. idle) --", 45);
    dumpNC(byFunc, nFunc, computeSamples ? (double)computeSamples : 1.0, 1,
           "-- top 45 COMPUTE functions (idle excluded, % of compute + cumulative %) --", 45);

    SymCleanup(hCur);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    if (g_out) fclose(g_out);
    free(cmd); 
    free(g_hits); 
    free(g_mods); 
    free(g_thr); 
    free(byFunc); 
    free(byMod);

    return 0;
}
