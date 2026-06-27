// ccxrun.c - launch ANY command with CCX-spread CPU placement.
//
// Many tools are not CCX/group aware: on a chiplet CPU their worker threads end
// up clustered on one CCX (one memory/fabric link), or confined to processor
// group 0 (<=64 LPs) because classic affinity masks cannot cross groups. This
// launcher spreads the child across CCXs - one primary core per CCX, interleaving
// processor groups - exactly like the in-process GGML_B612_CCX_SPREAD hook, but
// for arbitrary binaries (e.g. llama-bench).
//
// It works by creating the child SUSPENDED and calling SetProcessDefaultCpuSetMasks
// (Windows 11 / Server 2022+), which takes an array of GROUP_AFFINITY and is the
// only documented API that can place a process's threads across processor groups.
// New threads inherit the default CPU-set selection, so they schedule across the
// chosen LPs in every group. Falls back to SetProcessAffinityMask when the masks
// API is unavailable AND the selection fits in a single group.
//
// Build: cl /O2 /D_WIN32_WINNT=0x0601 ccxrun.c shell32.lib
// Run:   ccxrun [-n K] [-c C] [-v] [--] <command> [args...]
//   -n K : total worker LPs to engage, spread round-robin across CCXs
//          (default: one primary core per CCX = K = number of CCXs)
//   -c C : use C primary cores in EVERY CCX (overrides -n)
//   -v   : verbose - print the CCX map and the selected LP set
//   --   : end of options; everything after is the command line
//
// Why not "start /affinity 0x...": classic affinity is a single 64-bit mask
// scoped to ONE processor group, so it can never express a >64-LP, multi-group
// spread (e.g. one bit per CCX across 12 CCXs / 3 groups). Hence this tool.

#include <windows.h>
#include <shellapi.h>
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
    if (len == 0) { return; }
    BYTE *buf = (BYTE *)malloc(len);
    if (!buf) { return; }
    if (!GetLogicalProcessorInformationEx(RelationCache, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len)) {
        free(buf);
        return;
    }
    BYTE *p = buf;
    while (p < buf + len) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)p;
        if (info->Relationship == RelationCache && info->Cache.Level == 3 && g_nccx < 64) {
            GROUP_AFFINITY ga = info->Cache.GroupMask;
            ccx_t *c = &g_ccx[g_nccx++];
            int base = -1, cores = 0;
            c->group = ga.Group;
            c->mask  = ga.Mask;
            for (int b = 0; b < 64; b++) if (ga.Mask & (1ull << b)) { if (base < 0) base = b; cores++; }
            c->base_lp = (base < 0) ? 0 : base;
            c->n_cores = (cores / 2 > 0) ? (cores / 2) : 1; // SMT pairs
        }
        p += info->Size;
    }
    free(buf);
}

// Skip n whitespace-separated tokens of a raw command line, honoring "quoted"
// spans, and return a pointer to the remainder. Used to extract the child's
// command line verbatim (preserving its original quoting) from GetCommandLineW.
static const wchar_t *skip_tokens(const wchar_t *s, int n) {
    while (n-- > 0) {
        while (*s == L' ' || *s == L'\t') s++;
        if (*s == 0) break;
        int inq = 0;
        while (*s) {
            if (*s == L'"') { inq = !inq; s++; continue; }
            if (!inq && (*s == L' ' || *s == L'\t')) break;
            s++;
        }
    }
    while (*s == L' ' || *s == L'\t') s++;
    return s;
}

typedef BOOL (WINAPI *PFN_SetProcDefCpuSetMasks)(HANDLE, const GROUP_AFFINITY *, USHORT);

// Minimal command-line tokenizer used in place of shell32's CommandLineToArgvW
// so ccxrun stays within the GameCore layer (shell32 lives in DesktopEditions).
// Splits on unquoted whitespace; double quotes group a token and are stripped,
// matching skip_tokens() semantics. Sufficient for ccxrun's own simple flags;
// the child command tail is reconstructed verbatim via skip_tokens().
// Returns a single heap block (free with one free()); argv[argc] == NULL.
static wchar_t **split_args(const wchar_t *cmd, int *out_argc) {
    int argc = 0;
    const wchar_t *s = cmd;
    while (*s) {
        while (*s == L' ' || *s == L'\t') s++;
        if (!*s) break;
        argc++;
        int inq = 0;
        while (*s) {
            if (*s == L'"') { inq = !inq; s++; continue; }
            if (!inq && (*s == L' ' || *s == L'\t')) break;
            s++;
        }
    }
    size_t len      = wcslen(cmd);
    size_t ptrbytes = (size_t)(argc + 1) * sizeof(wchar_t *);
    char  *block    = (char *)malloc(ptrbytes + (len + 1) * sizeof(wchar_t));
    if (!block) return NULL;
    wchar_t **argv = (wchar_t **)block;
    wchar_t  *buf  = (wchar_t *)(block + ptrbytes);
    int    ai = 0;
    size_t bi = 0;
    s = cmd;
    while (*s) {
        while (*s == L' ' || *s == L'\t') s++;
        if (!*s) break;
        argv[ai++] = &buf[bi];
        int inq = 0;
        while (*s) {
            if (*s == L'"') { inq = !inq; s++; continue; }
            if (!inq && (*s == L' ' || *s == L'\t')) break;
            buf[bi++] = *s++;
        }
        buf[bi++] = 0;
    }
    argv[ai]  = NULL;
    *out_argc = ai;
    return argv;
}

int main(void) {
    int    want_n = 0;      // -n K   (0 => default: one per CCX)
    int    want_c = 0;      // -c C   (0 => unused)
    int    verbose = 0;
    int    child_index = 1; // index into wargv where the command starts

    int      wargc = 0;
    wchar_t **wargv = split_args(GetCommandLineW(), &wargc);
    if (!wargv) { fprintf(stderr, "ccxrun: failed to parse command line\n"); return 2; }

    int i = 1;
    for (; i < wargc; i++) {
        if (wargv[i][0] != L'-') break;
        if (!wcscmp(wargv[i], L"--")) { i++; break; }
        else if (!wcscmp(wargv[i], L"-v")) { verbose = 1; }
        else if (!wcscmp(wargv[i], L"-n") && i + 1 < wargc) { want_n = _wtoi(wargv[++i]); }
        else if (!wcscmp(wargv[i], L"-c") && i + 1 < wargc) { want_c = _wtoi(wargv[++i]); }
        else { fprintf(stderr, "ccxrun: unknown option '%ls'\n", wargv[i]); free(wargv); return 2; }
    }
    child_index = i;

    if (child_index >= wargc) {
        fprintf(stderr,
            "usage: ccxrun [-n K] [-c C] [-v] [--] <command> [args...]\n"
            "  -n K : total worker LPs spread round-robin across CCXs (default: one per CCX)\n"
            "  -c C : use C primary cores in every CCX (overrides -n)\n"
            "  -v   : verbose (print CCX map and selected LPs)\n");
        free(wargv);
        return 2;
    }

    build_ccx_map();
    if (g_nccx <= 0) {
        fprintf(stderr, "ccxrun: no L3/CCX topology found; launching without placement\n");
    }

    if (verbose) {
        printf("ccxrun: %d CCX domain(s)\n", g_nccx);
        for (int k = 0; k < g_nccx; k++) {
            printf("  CCX %2d: group=%u base_lp=%d cores=%d mask=0x%016llx\n",
                   k, g_ccx[k].group, g_ccx[k].base_lp, g_ccx[k].n_cores,
                   (unsigned long long)g_ccx[k].mask);
        }
    }

    // ---- select the LP set (primary SMT thread of each chosen physical core) ----
    typedef struct { WORD group; int lp; } sel_t;
    sel_t sel[256];
    int   nsel = 0;

    if (g_nccx > 0) {
        if (want_c > 0) {
            for (int c = 0; c < g_nccx && nsel < 256; c++) {
                for (int k = 0; k < want_c && nsel < 256; k++) {
                    int lp = g_ccx[c].base_lp + 2 * (k % g_ccx[c].n_cores);
                    sel[nsel].group = g_ccx[c].group;
                    sel[nsel].lp    = lp;
                    nsel++;
                }
            }
        } else {
            int K = (want_n > 0) ? want_n : g_nccx;   // default: one per CCX
            for (int k = 0; k < K && nsel < 256; k++) {
                int ci   = k % g_nccx;
                int core = k / g_nccx;
                ccx_t *c = &g_ccx[ci];
                int lp   = c->base_lp + 2 * (core % c->n_cores);
                sel[nsel].group = c->group;
                sel[nsel].lp    = lp;
                nsel++;
            }
        }
    }

    // ---- coalesce selected LPs into one GROUP_AFFINITY per processor group ----
    GROUP_AFFINITY masks[64];
    USHORT         nmask = 0;
    for (int s = 0; s < nsel; s++) {
        int found = -1;
        for (int m = 0; m < nmask; m++) if (masks[m].Group == sel[s].group) { found = m; break; }
        if (found < 0 && nmask < 64) {
            ZeroMemory(&masks[nmask], sizeof(masks[nmask]));
            masks[nmask].Group = sel[s].group;
            found = nmask++;
        }
        if (found >= 0) masks[found].Mask |= (1ull << sel[s].lp);
    }

    if (verbose && nmask > 0) {
        printf("ccxrun: selected %d LP(s) across %u group(s):\n", nsel, nmask);
        for (int m = 0; m < nmask; m++)
            printf("  group=%u mask=0x%016llx\n", masks[m].Group, (unsigned long long)masks[m].Mask);
    }

    // ---- reconstruct the child's command line verbatim ----
    const wchar_t *tail = skip_tokens(GetCommandLineW(), child_index);
    wchar_t *cmdline = _wcsdup(tail);   // CreateProcessW may modify the buffer
    free(wargv);
    if (!cmdline) { fprintf(stderr, "ccxrun: out of memory\n"); return 2; }

    // ---- create the child suspended, apply placement, resume ----
    STARTUPINFOW        si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    if (!CreateProcessW(NULL, cmdline, NULL, NULL, TRUE, CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
        fprintf(stderr, "ccxrun: CreateProcess failed (err=%lu) for: %ls\n",
                (unsigned long)GetLastError(), cmdline);
        free(cmdline);
        return 2;
    }

    int placed = 0;
    if (nmask > 0) {
        PFN_SetProcDefCpuSetMasks pset = (PFN_SetProcDefCpuSetMasks)GetProcAddress(
            GetModuleHandleW(L"kernel32.dll"), "SetProcessDefaultCpuSetMasks");
        if (pset && pset(pi.hProcess, masks, nmask)) {
            placed = 1;
            printf("ccxrun: placed child across %d CCX(s), %d LP(s), %u group(s) via CpuSetMasks\n",
                   g_nccx, nsel, nmask);
        } else if (nmask == 1 && masks[0].Group == 0) {
            if (SetProcessAffinityMask(pi.hProcess, masks[0].Mask)) {
                placed = 1;
                printf("ccxrun: placed child via affinity mask 0x%016llx (group 0)\n",
                       (unsigned long long)masks[0].Mask);
            }
        }
        if (!placed) {
            fprintf(stderr,
                "ccxrun: WARNING could not apply cross-group placement "
                "(needs Windows 11 / Server 2022+); running without it\n");
        }
    }

    ResumeThread(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    free(cmdline);
    return (int)code;
}
