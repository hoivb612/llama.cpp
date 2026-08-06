// usampler.cpp - admin-free user-mode sampling profiler (Very-Sleepy style).
//
// Launches a child process, periodically suspends each of its threads, reads the
// instruction pointer (RIP), resumes, and records the sample. After the child
// exits, it symbolizes the sampled addresses against the child's modules/PDBs and
// prints top self-time modules and functions.
//
// No elevation required: it only touches a process it launched itself.
//
// Build (x64): cl /O2 /EHsc usampler.cpp /link dbghelp.lib winmm.lib
//
// Usage: usampler.exe [options] <child.exe> [child args...]
//   --out FILE       also write the PROFILE section (only) to FILE (clean, no
//                    child stdout) so captures are easy to share between boxes.
//   --sleep-us N     target inter-sweep interval in microseconds (default 1000).
//                    Values < 1000 spin on QueryPerformanceCounter for a higher,
//                    steadier sample rate; >= 1000 uses Sleep(N/1000).
//   --skip-ms N      discard all samples during the first N ms of the child's
//                    life (default 0). Use to drop model-load + first-prefill so
//                    the profile is biased toward steady-state DECODE.
//   --                end of usampler options; everything after is the child cmd.
//
// The child's own stdout is untouched (it shares this console). Only dbghelp.dll
// is required (present on all Windows); winmm is used for a 1 ms timer.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <dbghelp.h>
#include <mmsystem.h>
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "winmm.lib")

struct ModInfo { uint64_t base; uint64_t size; std::string name; std::string path; };

// ---- dual-sink output: stdout + optional clean profile file --------------
static FILE* g_out = nullptr;
static void emit(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    if (g_out) { va_start(ap, fmt); vfprintf(g_out, fmt, ap); va_end(ap); }
}

// Substrings of module!symbol names that represent IDLE / scheduling wait, not
// real compute. Excluded from the "compute-only" ranking so decode kernels
// stand out. Covers both the b612 xbox threadpool spin and the OpenMP/worker-
// factory waits seen on the upstream path.
static const char* kIdleNeedles[] = {
    "ZwWaitForWorkViaWorkerFactory", "NtWaitForWorkViaWorkerFactory",
    "ZwDelayExecution", "NtDelayExecution", "RtlDelayExecution",
    "ZwWaitForSingleObject", "NtWaitForSingleObject",
    "ZwWaitForAlertByThreadId", "NtWaitForAlertByThreadId",
    "RtlWaitOnAddress", "WaitForSingleObject", "SwitchToThread",
    "ggml_wait_for_done_xbox",   // b612 threadpool spin-wait
    "ggml_barrier",              // threadpool barrier spin
    "vcomp_barrier",             // OpenMP barrier
};
static bool isIdle(const std::string& fn) {
    for (const char* n : kIdleNeedles)
        if (fn.find(n) != std::string::npos) return true;
    return false;
}

int main(int argc, char** argv) {
    // ---- parse usampler options up to the child exe --------------------
    std::string outPath;
    long sleepUs = 1000;
    long skipMs  = 0;
    int  ci      = 1;               // index of child exe in argv
    for (; ci < argc; ci++) {
        std::string a = argv[ci];
        if (a == "--") { ci++; break; }
        else if (a == "--out"      && ci + 1 < argc) { outPath = argv[++ci]; }
        else if (a == "--sleep-us" && ci + 1 < argc) { sleepUs = atol(argv[++ci]); }
        else if (a == "--skip-ms"  && ci + 1 < argc) { skipMs  = atol(argv[++ci]); }
        else if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
            printf("[usampler] unknown option: %s\n", a.c_str()); return 1;
        }
        else break;                 // first non-option token = child exe
    }
    if (ci >= argc) {
        printf("usage: usampler [--out FILE] [--sleep-us N] [--skip-ms N] <child.exe> [args...]\n");
        return 1;
    }
    if (sleepUs < 1) sleepUs = 1;

    // Build a quoted command line from the child args.
    std::string cmd;
    for (int i = ci; i < argc; i++) {
        std::string a = argv[i];
        if (a.find(' ') != std::string::npos && a.front() != '"') a = "\"" + a + "\"";
        if (!cmd.empty()) cmd += " ";
        cmd += a;
    }

    if (!outPath.empty()) {
        g_out = fopen(outPath.c_str(), "w");
        if (!g_out) printf("[usampler] WARNING: cannot open --out %s\n", outPath.c_str());
    }

    timeBeginPeriod(1);             // steadier Sleep()/sampling cadence

    LARGE_INTEGER qpf; QueryPerformanceFrequency(&qpf);
    const double qpcToUs = 1e6 / (double)qpf.QuadPart;

    printf("[usampler] launching: %s\n", cmd.c_str());
    printf("[usampler] sleep-us=%ld  skip-ms=%ld  out=%s\n",
           sleepUs, skipMs, outPath.empty() ? "(none)" : outPath.c_str());

    STARTUPINFOA si; ZeroMemory(&si, sizeof(si)); si.cb = sizeof(si);
    PROCESS_INFORMATION pi; ZeroMemory(&pi, sizeof(pi));
    std::vector<char> cmdbuf(cmd.begin(), cmd.end()); cmdbuf.push_back(0);
    if (!CreateProcessA(NULL, cmdbuf.data(), NULL, NULL, TRUE,
                        CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
        printf("[usampler] CreateProcess failed: %lu\n", GetLastError());
        timeEndPeriod(1);
        return 1;
    }
    const DWORD pid = pi.dwProcessId;

    LARGE_INTEGER tStart; QueryPerformanceCounter(&tStart);
    ResumeThread(pi.hThread);

    std::vector<ModInfo> mods;
    const bool dbg = getenv("USAMPLER_DBG") != NULL;
    // Module capture. CreateToolhelp32Snapshot(TH32CS_SNAPMODULE) can transiently
    // fail with ERROR_PARTIAL_COPY (299) / ERROR_BAD_LENGTH (24) while the target
    // is still loading DLLs or its loader list is changing, so retry briefly.
    // Called on a wall-clock cadence (not iteration count) so it reliably fires
    // mid-run once the process is loaded, and keeps catching delay-loaded DLLs.
    auto refreshMods = [&]() {
        for (int attempt = 0; attempt < 10; ++attempt) {
            HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, pid);
            if (snap == INVALID_HANDLE_VALUE) {
                DWORD e = GetLastError();
                if (e == ERROR_BAD_LENGTH || e == ERROR_PARTIAL_COPY) { Sleep(1); continue; }
                if (dbg) fprintf(stderr, "[dbg] SNAPMODULE INVALID err=%lu mods=%zu\n", e, mods.size());
                return;
            }
            MODULEENTRY32 me; me.dwSize = sizeof(me);
            if (Module32First(snap, &me)) {
                do {
                    uint64_t base = (uint64_t)me.modBaseAddr;
                    bool found = false;
                    for (auto& m : mods) if (m.base == base) { found = true; break; }
                    if (!found) {
                        ModInfo mi; mi.base = base; mi.size = me.modBaseSize;
                        mi.name = me.szModule; mi.path = me.szExePath;
                        mods.push_back(mi);
                    }
                } while (Module32Next(snap, &me));
            }
            CloseHandle(snap);
            if (dbg) fprintf(stderr, "[dbg] SNAPMODULE ok mods=%zu\n", mods.size());
            return;
        }
        if (dbg) fprintf(stderr, "[dbg] SNAPMODULE gave up (partial-copy) mods=%zu\n", mods.size());
    };

    // Cache of the target's thread handles. TH32CS_SNAPTHREAD is system-wide (no
    // per-pid filter), so snapshotting + walking every thread on the box each
    // sweep is O(all-system-threads) and collapses on big machines (e.g. a 96-core
    // host with 20k+ threads across 400+ processes): the sample loop crawls and
    // module capture starves. Instead we rebuild the handle cache on a coarse
    // cadence and reuse the handles for the tight per-sweep suspend/sample.
    struct TgtThread { DWORD tid; HANDLE h; };
    std::vector<TgtThread> threads;
    auto refreshThreads = [&]() {
        HANDLE tsnap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (tsnap == INVALID_HANDLE_VALUE) return;
        std::unordered_map<DWORD, bool> live;
        THREADENTRY32 te; te.dwSize = sizeof(te);
        if (Thread32First(tsnap, &te)) {
            do {
                if (te.th32OwnerProcessID != pid) continue;
                live[te.th32ThreadID] = true;
                bool have = false;
                for (auto& t : threads) if (t.tid == te.th32ThreadID) { have = true; break; }
                if (!have) {
                    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT,
                                          FALSE, te.th32ThreadID);
                    if (h) threads.push_back({ te.th32ThreadID, h });
                }
            } while (Thread32Next(tsnap, &te));
        }
        CloseHandle(tsnap);
        // release handles for threads that have exited
        for (size_t i = 0; i < threads.size(); ) {
            if (!live.count(threads[i].tid)) {
                CloseHandle(threads[i].h);
                threads[i] = threads.back();
                threads.pop_back();
            } else ++i;
        }
        if (dbg) fprintf(stderr, "[dbg] threads cached=%zu\n", threads.size());
    };

    // High-resolution inter-sweep wait.
    auto waitInterval = [&]() {
        if (sleepUs >= 1000) { Sleep((DWORD)(sleepUs / 1000)); return; }
        LARGE_INTEGER a; QueryPerformanceCounter(&a);
        for (;;) {
            LARGE_INTEGER b; QueryPerformanceCounter(&b);
            if ((b.QuadPart - a.QuadPart) * qpcToUs >= (double)sleepUs) break;
            YieldProcessor();
        }
    };

    auto elapsedMs = [&]() -> double {
        LARGE_INTEGER now; QueryPerformanceCounter(&now);
        return (now.QuadPart - tStart.QuadPart) * qpcToUs / 1000.0;
    };

    std::unordered_map<uint64_t, uint32_t> hits;
    uint64_t total = 0, skipped = 0;
    bool warmupDone = (skipMs <= 0);
    // Worker threads are created once at startup and are stable thereafter, and
    // the module set settles quickly, so both refreshes are coarse. This keeps
    // the expensive system-wide snapshots to a few per second regardless of the
    // sample rate.
    double lastModMs = -1e9, lastThrMs = -1e9;

    while (WaitForSingleObject(pi.hProcess, 0) == WAIT_TIMEOUT) {
        const double ms = elapsedMs();
        if (ms - lastModMs >= 250.0) { refreshMods();    lastModMs = ms; }
        if (ms - lastThrMs >= 500.0) { refreshThreads(); lastThrMs = ms; }

        if (!warmupDone && ms >= (double)skipMs) warmupDone = true;

        for (auto& t : threads) {
            if (SuspendThread(t.h) == (DWORD)-1) continue;
            CONTEXT ctx; ZeroMemory(&ctx, sizeof(ctx));
            ctx.ContextFlags = CONTEXT_CONTROL;
            if (GetThreadContext(t.h, &ctx)) {
                if (warmupDone) { hits[ctx.Rip]++; total++; }
                else            { skipped++; }
            }
            ResumeThread(t.h);
        }
        waitInterval();
    }
    refreshMods();
    for (auto& t : threads) CloseHandle(t.h);
    timeEndPeriod(1);

    emit("[usampler] child exited. total samples = %llu (warmup-skipped = %llu) across %zu modules.\n",
         (unsigned long long)total, (unsigned long long)skipped, mods.size());
    if (total == 0) { emit("[usampler] no samples collected.\n"); if (g_out) fclose(g_out); return 0; }

    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    HANDLE hCur = GetCurrentProcess();
    if (!SymInitialize(hCur, NULL, FALSE))
        printf("[usampler] SymInitialize failed %lu\n", GetLastError());
    for (auto& m : mods)
        SymLoadModuleEx(hCur, NULL, m.path.c_str(), m.name.c_str(), m.base, (DWORD)m.size, NULL, 0);

    auto modFor = [&](uint64_t a) -> ModInfo* {
        for (auto& m : mods) if (a >= m.base && a < m.base + m.size) return &m;
        return nullptr;
    };

    std::unordered_map<std::string, uint64_t> byFunc, byMod;
    char symbuf[sizeof(SYMBOL_INFO) + 1024];
    for (auto& kv : hits) {
        uint64_t a = kv.first; uint32_t c = kv.second;
        ModInfo* m = modFor(a);
        std::string modName = m ? m->name : "[kernel/unknown]";
        byMod[modName] += c;
        std::string fn;
        SYMBOL_INFO* sym = (SYMBOL_INFO*)symbuf;
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 1024;
        DWORD64 disp = 0;
        if (m && SymFromAddr(hCur, a, &disp, sym)) {
            fn = modName + "!" + sym->Name;
            // Append the function's definition file:line so same-named symbols
            // from different translation units (e.g. vec.cpp vs b612/vec-b612.cpp,
            // template instantiations, or per-TU static helpers) can be told apart.
            // Resolve at the SYMBOL START address (sym->Address), not the sampled
            // address, so every sample inside one function shares the same tag and
            // the histogram stays aggregated per-function instead of per-line.
            IMAGEHLP_LINE64 li; ZeroMemory(&li, sizeof(li));
            li.SizeOfStruct = sizeof(li);
            DWORD lineDisp = 0;
            if (SymGetLineFromAddr64(hCur, sym->Address, &lineDisp, &li) && li.FileName) {
                const char* bs = strrchr(li.FileName, '\\');
                const char* fs = strrchr(li.FileName, '/');
                const char* base = (fs > bs ? fs : bs);
                base = base ? base + 1 : li.FileName;
                char tag[300];
                snprintf(tag, sizeof(tag), "  (%s:%lu)", base, (unsigned long)li.LineNumber);
                fn += tag;
            }
        } else {
            char buf[80];
            sprintf(buf, "%s+0x%llx", modName.c_str(),
                    (unsigned long long)(m ? (a - m->base) : a));
            fn = buf;
        }
        byFunc[fn] += c;
    }

    // idle vs compute split
    uint64_t idleSamples = 0;
    for (auto& kv : byFunc) if (isIdle(kv.first)) idleSamples += kv.second;
    const uint64_t computeSamples = total - idleSamples;

    auto dumpAll = [&](std::unordered_map<std::string, uint64_t>& map,
                       const char* title, size_t topN) {
        std::vector<std::pair<std::string, uint64_t>> v(map.begin(), map.end());
        std::sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.second > y.second; });
        emit("\n%s\n", title);
        for (size_t i = 0; i < v.size() && i < topN; i++)
            emit("  %6.2f%%  %8llu  %s\n", 100.0 * v[i].second / total,
                 (unsigned long long)v[i].second, v[i].first.c_str());
    };

    // compute-only: drop idle symbols, renormalize to computeSamples
    auto dumpCompute = [&](const char* title, size_t topN) {
        std::vector<std::pair<std::string, uint64_t>> v;
        for (auto& kv : byFunc) if (!isIdle(kv.first)) v.push_back(kv);
        std::sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.second > y.second; });
        emit("\n%s\n", title);
        double denom = computeSamples ? (double)computeSamples : 1.0;
        for (size_t i = 0; i < v.size() && i < topN; i++)
            emit("  %6.2f%%  %8llu  %s\n", 100.0 * v[i].second / denom,
                 (unsigned long long)v[i].second, v[i].first.c_str());
    };

    emit("\n================ PROFILE (total samples = %llu) ================\n",
         (unsigned long long)total);
    emit("  idle/scheduler = %6.2f%% (%llu)   compute = %6.2f%% (%llu)\n",
         100.0 * idleSamples / total,    (unsigned long long)idleSamples,
         100.0 * computeSamples / total, (unsigned long long)computeSamples);
    dumpAll(byMod,  "-- top modules by self-samples (all) --", 15);
    dumpAll(byFunc, "-- top 45 functions by self-samples (all, incl. idle) --", 45);
    dumpCompute("-- top 45 COMPUTE functions (idle excluded, % of compute) --", 45);

    SymCleanup(hCur);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    if (g_out) fclose(g_out);
    return 0;
}
