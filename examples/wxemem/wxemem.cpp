// wxemem.cpp — WXEmem (Windows Xbox Edition memory inspector)
//
// A standalone command-line tool that reports current system memory usage
// so an operator can decide whether there's enough headroom to run an SLM
// or other large workload.
//
// Reports:
//   - Physical RAM totals + memory load + commit charge
//   - Kernel memory breakdown (paged/non-paged pool, kernel stacks, etc.)
//   - Standby cache / modified pages (reclaimable memory)
//   - Top-N processes by working set / private bytes
//   - Service inventory mapped to host processes
//   - Driver code-section sizes (admin)
//   - JSON output for scripting (--json)
//
// Windows-only. Runs without admin (best-effort) and lights up additional
// breakdowns when elevated.

#include <windows.h>
#include <psapi.h>
#include <winternl.h>
#include <winsvc.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// --- NtQuerySystemInformation glue ---------------------------------------
// These are declared in winternl.h but the relevant SYSTEM_INFORMATION_CLASS
// values and structs are not. They are stable Microsoft-internal APIs used by
// Task Manager / Process Explorer / Sysinternals tools.

#ifndef STATUS_SUCCESS
#define STATUS_SUCCESS ((NTSTATUS)0)
#endif

extern "C" {
typedef LONG KPRIORITY;

// SYSTEM_INFORMATION_CLASS values we use (others exist in winternl.h).
enum {
    SystemPerformanceInformation_Class    = 2,
    SystemModuleInformation_Class         = 11,
    SystemMemoryListInformation_Class     = 80,
};

// SYSTEM_PERFORMANCE_INFORMATION (subset; only the fields we use).
// Layout matches the public Windows documentation; trailing fields exist
// but we don't read them.
struct SYSTEM_PERFORMANCE_INFORMATION_LITE {
    LARGE_INTEGER IdleProcessTime;
    LARGE_INTEGER IoReadTransferCount;
    LARGE_INTEGER IoWriteTransferCount;
    LARGE_INTEGER IoOtherTransferCount;
    ULONG         IoReadOperationCount;
    ULONG         IoWriteOperationCount;
    ULONG         IoOtherOperationCount;
    ULONG         AvailablePages;
    ULONG         CommittedPages;
    ULONG         CommitLimit;
    ULONG         PeakCommitment;
    ULONG         PageFaultCount;
    ULONG         CopyOnWriteCount;
    ULONG         TransitionCount;
    ULONG         CacheTransitionCount;
    ULONG         DemandZeroCount;
    ULONG         PageReadCount;
    ULONG         PageReadIoCount;
    ULONG         CacheReadCount;
    ULONG         CacheIoCount;
    ULONG         DirtyPagesWriteCount;
    ULONG         DirtyWriteIoCount;
    ULONG         MappedPagesWriteCount;
    ULONG         MappedWriteIoCount;
    ULONG         PagedPoolPages;
    ULONG         NonPagedPoolPages;
    ULONG         PagedPoolAllocs;
    ULONG         PagedPoolFrees;
    ULONG         NonPagedPoolAllocs;
    ULONG         NonPagedPoolFrees;
    ULONG         FreeSystemPtes;
    ULONG         ResidentSystemCodePage;
    ULONG         TotalSystemDriverPages;
    ULONG         TotalSystemCodePages;
    ULONG         NonPagedPoolLookasideHits;
    ULONG         PagedPoolLookasideHits;
    ULONG         AvailablePagedPoolPages;
    ULONG         ResidentSystemCachePage;
    ULONG         ResidentPagedPoolPage;
    ULONG         ResidentSystemDriverPage;
};

// SYSTEM_MEMORY_LIST_INFORMATION — page counts on each memory list.
struct SYSTEM_MEMORY_LIST_INFORMATION {
    SIZE_T ZeroPageCount;
    SIZE_T FreePageCount;
    SIZE_T ModifiedPageCount;
    SIZE_T ModifiedNoWritePageCount;
    SIZE_T BadPageCount;
    SIZE_T PageCountByPriority[8];
    SIZE_T RepurposedPagesByPriority[8];
    SIZE_T ModifiedPageCountPageFile;
};

// RTL_PROCESS_MODULE_INFORMATION — driver/module list.
struct RTL_PROCESS_MODULE_INFORMATION {
    HANDLE Section;
    PVOID  MappedBase;
    PVOID  ImageBase;
    ULONG  ImageSize;
    ULONG  Flags;
    USHORT LoadOrderIndex;
    USHORT InitOrderIndex;
    USHORT LoadCount;
    USHORT OffsetToFileName;
    UCHAR  FullPathName[256];
};

struct RTL_PROCESS_MODULES {
    ULONG NumberOfModules;
    RTL_PROCESS_MODULE_INFORMATION Modules[1];
};
}  // extern "C"

// --- helpers --------------------------------------------------------------

static std::string g_program = "wxemem";

static std::string format_bytes(uint64_t bytes, int width = 0) {
    static const char * units[] = {"B", "KB", "MB", "GB", "TB"};
    double v = (double)bytes;
    int    u = 0;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        ++u;
    }
    char buf[64];
    if (u == 0) {
        snprintf(buf, sizeof(buf), "%*llu %s", width, (unsigned long long)bytes, units[u]);
    } else {
        snprintf(buf, sizeof(buf), "%*.*f %s", width, (v >= 100 ? 1 : 2), v, units[u]);
    }
    return buf;
}

static std::string format_mib(uint64_t bytes) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.1f MB", (double)bytes / (1024.0 * 1024.0));
    return buf;
}

static bool is_running_as_admin() {
    BOOL  is_admin = FALSE;
    PSID  admins_sid = nullptr;
    SID_IDENTIFIER_AUTHORITY nt = SECURITY_NT_AUTHORITY;
    if (AllocateAndInitializeSid(&nt, 2,
                                  SECURITY_BUILTIN_DOMAIN_RID,
                                  DOMAIN_ALIAS_RID_ADMINS,
                                  0, 0, 0, 0, 0, 0,
                                  &admins_sid)) {
        CheckTokenMembership(nullptr, admins_sid, &is_admin);
        FreeSid(admins_sid);
    }
    return is_admin != FALSE;
}

static std::string ws_to_utf8(const wchar_t * w) {
    if (!w || !*w) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return {};
    std::string s(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8]; snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// --- data structures ------------------------------------------------------

struct PhysicalMem {
    uint64_t total          = 0;
    uint64_t available      = 0;
    uint64_t used            = 0;
    uint32_t memory_load_pct = 0;
    uint64_t commit_total    = 0;  // process commit charge
    uint64_t commit_limit    = 0;
    uint64_t total_virtual   = 0;
    uint64_t avail_virtual   = 0;
    uint64_t total_pagefile  = 0;  // physical + page file backing
    uint64_t avail_pagefile  = 0;
};

struct KernelMem {
    uint64_t paged_pool         = 0;
    uint64_t non_paged_pool     = 0;
    uint64_t paged_pool_avail   = 0;  // free portion of paged pool
    uint64_t driver_code        = 0;  // resident driver code pages
    uint64_t system_code        = 0;  // resident system (kernel) code pages
    uint64_t system_cache       = 0;  // resident system cache
    uint64_t resident_paged_pool = 0;
    uint64_t commit_total       = 0;  // pages
    uint64_t commit_limit       = 0;  // pages
    bool     available          = false;
};

struct MemList {
    uint64_t zero_bytes     = 0;
    uint64_t free_bytes     = 0;
    uint64_t modified_bytes = 0;
    uint64_t standby_bytes  = 0;
    bool     available      = false;
};

struct ProcInfo {
    DWORD       pid          = 0;
    std::string name;
    uint64_t    ws           = 0;  // working set (resident)
    uint64_t    peak_ws      = 0;
    uint64_t    private_bytes = 0; // PrivateUsage on PROCESS_MEMORY_COUNTERS_EX
    uint64_t    pagefile     = 0;
};

enum class ServiceStartKind : uint8_t {
    Unknown      = 0,
    Auto         = 1,   // SERVICE_AUTO_START
    AutoDelayed  = 2,   // SERVICE_AUTO_START + delayed flag
    Demand       = 3,   // SERVICE_DEMAND_START
    Boot         = 4,   // SERVICE_BOOT_START (drivers only)
    System       = 5,   // SERVICE_SYSTEM_START (drivers only)
    Disabled     = 6,   // SERVICE_DISABLED
};

static const char * start_kind_short(ServiceStartKind k) {
    switch (k) {
        case ServiceStartKind::Auto:        return "auto";
        case ServiceStartKind::AutoDelayed: return "auto-d";
        case ServiceStartKind::Demand:      return "demand";
        case ServiceStartKind::Boot:        return "boot";
        case ServiceStartKind::System:      return "system";
        case ServiceStartKind::Disabled:    return "disabled";
        default:                            return "?";
    }
}

struct ServiceInfo {
    std::string name;
    std::string display_name;
    DWORD       pid          = 0;
    DWORD       service_type = 0;
    DWORD       current_state = 0;
    ServiceStartKind start_kind = ServiceStartKind::Unknown;
};

struct DriverInfo {
    std::string name;
    uint64_t    image_size = 0;
    uintptr_t   image_base = 0;
};

struct PageFile {
    std::string path;
    uint64_t    total_bytes = 0;  // current allocated size on disk
    uint64_t    used_bytes  = 0;  // currently used commit charge backed by this file
    uint64_t    peak_bytes  = 0;  // peak commit charge backed by this file
};

// If `driver` is a kernel hotpatch image (Windows 11/Server 2025 hot-patching),
// returns the name of the original kernel module it patches. Returns an empty
// string for non-hotpatch images. See `print_human` for usage / display.
static std::string hotpatch_original(const std::string & driver) {
    auto ends_with = [](const std::string & s, const std::string & suf) {
        return s.size() >= suf.size() &&
               _stricmp(s.c_str() + s.size() - suf.size(), suf.c_str()) == 0;
    };
    if (driver.size() < strlen("_hotpatch") + 4) return {};
    if (!ends_with(driver, "_hotpatch.sys") &&
        !ends_with(driver, "_hotpatch.exe") &&
        !ends_with(driver, "_hotpatch.dll")) return {};

    // Strip "_hotpatch" before the extension.
    size_t dot = driver.find_last_of('.');
    std::string base = driver.substr(0, dot - strlen("_hotpatch"));
    std::string ext  = driver.substr(dot);  // includes the leading dot

    // Special-case the NT kernel: hotpatch is named "ntkrnlmp_hotpatch.exe",
    // but the running image is "ntoskrnl.exe".
    if (_stricmp(base.c_str(), "ntkrnlmp") == 0) {
        return "ntoskrnl.exe";
    }
    return base + ext;
}

// --- collectors -----------------------------------------------------------

static PhysicalMem collect_physical() {
    PhysicalMem p{};
    MEMORYSTATUSEX ms{}; ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        p.total           = ms.ullTotalPhys;
        p.available       = ms.ullAvailPhys;
        p.used            = ms.ullTotalPhys > ms.ullAvailPhys
                                ? ms.ullTotalPhys - ms.ullAvailPhys : 0;
        p.memory_load_pct = ms.dwMemoryLoad;
        p.commit_total    = ms.ullTotalPageFile > ms.ullAvailPageFile
                                ? ms.ullTotalPageFile - ms.ullAvailPageFile : 0;
        p.commit_limit    = ms.ullTotalPageFile;
        p.total_virtual   = ms.ullTotalVirtual;
        p.avail_virtual   = ms.ullAvailVirtual;
        p.total_pagefile  = ms.ullTotalPageFile;
        p.avail_pagefile  = ms.ullAvailPageFile;
    }
    return p;
}

static KernelMem collect_kernel() {
    KernelMem k{};
    // SYSTEM_PERFORMANCE_INFORMATION has grown over Windows revisions; rather
    // than mirroring the exact layout, query the actual size first and then
    // pass a buffer at least that large. We read fields by their offsets via
    // a casted SYSTEM_PERFORMANCE_INFORMATION_LITE view.
    ULONG needed = 0;
    NtQuerySystemInformation((SYSTEM_INFORMATION_CLASS)SystemPerformanceInformation_Class,
                              nullptr, 0, &needed);
    if (needed < sizeof(SYSTEM_PERFORMANCE_INFORMATION_LITE)) {
        needed = sizeof(SYSTEM_PERFORMANCE_INFORMATION_LITE);
    }
    // Add some slack in case the kernel writes more bytes than our LITE view
    // describes; we only read the prefix fields.
    std::vector<uint8_t> buf(std::max<ULONG>(needed, 8192));
    NTSTATUS s = NtQuerySystemInformation((SYSTEM_INFORMATION_CLASS)SystemPerformanceInformation_Class,
                                          buf.data(), (ULONG)buf.size(), &needed);
    if (s != STATUS_SUCCESS) {
        return k;
    }
    const SYSTEM_PERFORMANCE_INFORMATION_LITE & info =
        *reinterpret_cast<SYSTEM_PERFORMANCE_INFORMATION_LITE *>(buf.data());
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const uint64_t page = si.dwPageSize;

    k.paged_pool          = (uint64_t)info.PagedPoolPages * page;
    k.non_paged_pool      = (uint64_t)info.NonPagedPoolPages * page;
    k.paged_pool_avail    = (uint64_t)info.AvailablePagedPoolPages * page;
    k.driver_code         = (uint64_t)info.TotalSystemDriverPages * page;
    k.system_code         = (uint64_t)info.TotalSystemCodePages * page;
    k.system_cache        = (uint64_t)info.ResidentSystemCachePage * page;
    k.resident_paged_pool = (uint64_t)info.ResidentPagedPoolPage * page;
    k.commit_total        = (uint64_t)info.CommittedPages * page;
    k.commit_limit        = (uint64_t)info.CommitLimit * page;
    k.available           = true;
    return k;
}

static MemList collect_memlist() {
    MemList m{};
    SYSTEM_MEMORY_LIST_INFORMATION info{};
    ULONG ret = 0;
    NTSTATUS s = NtQuerySystemInformation((SYSTEM_INFORMATION_CLASS)SystemMemoryListInformation_Class,
                                          &info, sizeof(info), &ret);
    if (s != STATUS_SUCCESS) {
        return m;
    }
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const uint64_t page = si.dwPageSize;

    m.zero_bytes     = (uint64_t)info.ZeroPageCount * page;
    m.free_bytes     = (uint64_t)info.FreePageCount * page;
    m.modified_bytes = (uint64_t)info.ModifiedPageCount * page;
    // Standby = sum of PageCountByPriority across the 8 priority levels.
    uint64_t standby = 0;
    for (int i = 0; i < 8; ++i) {
        standby += (uint64_t)info.PageCountByPriority[i] * page;
    }
    m.standby_bytes = standby;
    m.available     = true;
    return m;
}

static std::vector<ProcInfo> collect_processes() {
    std::vector<ProcInfo> out;
    std::vector<DWORD> pids(2048);
    DWORD got_bytes = 0;
    for (;;) {
        if (!K32EnumProcesses(pids.data(), (DWORD)(pids.size() * sizeof(DWORD)), &got_bytes)) {
            return out;
        }
        if (got_bytes < pids.size() * sizeof(DWORD)) break;
        pids.resize(pids.size() * 2);
    }
    DWORD n_pids = got_bytes / sizeof(DWORD);
    out.reserve(n_pids);

    for (DWORD i = 0; i < n_pids; ++i) {
        DWORD pid = pids[i];
        if (pid == 0) continue;  // System Idle Process

        HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
                               FALSE, pid);
        if (!h) {
            // Fall back: PROCESS_QUERY_LIMITED_INFORMATION alone (no VM_READ)
            h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
        }
        if (!h) {
            ProcInfo pi{};
            pi.pid  = pid;
            pi.name = (pid == 4) ? "System" : "<inaccessible>";
            out.push_back(pi);
            continue;
        }

        ProcInfo pi{};
        pi.pid = pid;

        // Name (image base name).
        wchar_t name[MAX_PATH] = L"";
        if (K32GetModuleBaseNameW(h, nullptr, name, MAX_PATH) > 0) {
            pi.name = ws_to_utf8(name);
        } else {
            // Try GetProcessImageFileName as a fallback for protected processes.
            wchar_t path[MAX_PATH] = L"";
            DWORD path_len = MAX_PATH;
            if (QueryFullProcessImageNameW(h, 0, path, &path_len)) {
                const wchar_t * base = wcsrchr(path, L'\\');
                pi.name = ws_to_utf8(base ? base + 1 : path);
            } else {
                pi.name = "<unknown>";
            }
        }

        PROCESS_MEMORY_COUNTERS_EX pmc{}; pmc.cb = sizeof(pmc);
        if (K32GetProcessMemoryInfo(h, (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
            pi.ws            = pmc.WorkingSetSize;
            pi.peak_ws       = pmc.PeakWorkingSetSize;
            pi.private_bytes = pmc.PrivateUsage;
            pi.pagefile      = pmc.PagefileUsage;
        }

        CloseHandle(h);
        out.push_back(pi);
    }
    return out;
}

// Enumerate one filter (SERVICE_WIN32 or SERVICE_DRIVER) and append results
// to `out`. Reuses cfg_buf for QueryServiceConfigW scratch.
static void enum_one_filter(SC_HANDLE scm, DWORD type_filter,
                            std::vector<ServiceInfo> & out,
                            std::vector<uint8_t> & cfg_buf) {
    DWORD bytes_needed = 0, services_returned = 0, resume = 0;
    EnumServicesStatusExW(scm, SC_ENUM_PROCESS_INFO,
                          type_filter, SERVICE_ACTIVE,
                          nullptr, 0, &bytes_needed, &services_returned, &resume, nullptr);
    if (bytes_needed == 0) return;

    std::vector<uint8_t> buf(bytes_needed);
    if (!EnumServicesStatusExW(scm, SC_ENUM_PROCESS_INFO,
                               type_filter, SERVICE_ACTIVE,
                               buf.data(), (DWORD)buf.size(),
                               &bytes_needed, &services_returned, &resume, nullptr)) {
        return;
    }

    auto * status = (ENUM_SERVICE_STATUS_PROCESSW *)buf.data();
    out.reserve(out.size() + services_returned);
    for (DWORD i = 0; i < services_returned; ++i) {
        ServiceInfo si{};
        si.name          = ws_to_utf8(status[i].lpServiceName);
        si.display_name  = ws_to_utf8(status[i].lpDisplayName);
        si.pid           = status[i].ServiceStatusProcess.dwProcessId;
        si.service_type  = status[i].ServiceStatusProcess.dwServiceType;
        si.current_state = status[i].ServiceStatusProcess.dwCurrentState;

        // Per-service start type via QueryServiceConfig.  Open with the
        // minimum-needed rights so we don't fail when running non-elevated.
        SC_HANDLE svc = OpenServiceW(scm, status[i].lpServiceName, SERVICE_QUERY_CONFIG);
        if (svc) {
            DWORD cfg_needed = 0;
            QueryServiceConfigW(svc, nullptr, 0, &cfg_needed);
            if (cfg_needed > 0) {
                if (cfg_buf.size() < cfg_needed) cfg_buf.resize(cfg_needed);
                auto * cfg = (QUERY_SERVICE_CONFIGW *)cfg_buf.data();
                if (QueryServiceConfigW(svc, cfg, (DWORD)cfg_buf.size(), &cfg_needed)) {
                    switch (cfg->dwStartType) {
                        case SERVICE_BOOT_START:   si.start_kind = ServiceStartKind::Boot;     break;
                        case SERVICE_SYSTEM_START: si.start_kind = ServiceStartKind::System;   break;
                        case SERVICE_AUTO_START:   si.start_kind = ServiceStartKind::Auto;     break;
                        case SERVICE_DEMAND_START: si.start_kind = ServiceStartKind::Demand;   break;
                        case SERVICE_DISABLED:     si.start_kind = ServiceStartKind::Disabled; break;
                        default:                   si.start_kind = ServiceStartKind::Unknown;  break;
                    }
                    if (si.start_kind == ServiceStartKind::Auto) {
                        SERVICE_DELAYED_AUTO_START_INFO dasi{};
                        DWORD dasi_needed = 0;
                        if (QueryServiceConfig2W(svc, SERVICE_CONFIG_DELAYED_AUTO_START_INFO,
                                                 (LPBYTE)&dasi, sizeof(dasi), &dasi_needed) &&
                            dasi.fDelayedAutostart) {
                            si.start_kind = ServiceStartKind::AutoDelayed;
                        }
                    }
                }
            }
            CloseServiceHandle(svc);
        }

        out.push_back(si);
    }
}

static std::vector<ServiceInfo> collect_services() {
    std::vector<ServiceInfo> out;
    SC_HANDLE scm = OpenSCManager(nullptr, nullptr, SC_MANAGER_ENUMERATE_SERVICE);
    if (!scm) return out;

    std::vector<uint8_t> cfg_buf;
    // Win32 user-mode services (svchost-hosted + dedicated .exe).
    enum_one_filter(scm, SERVICE_WIN32, out, cfg_buf);
    // Kernel drivers (boot-start, system-start, and a few auto-start drivers).
    // These don't have a PID/host process; they live in kernel address space.
    // We include them so the start-kind summary reflects boot-time charge.
    enum_one_filter(scm, SERVICE_DRIVER, out, cfg_buf);

    CloseServiceHandle(scm);
    return out;
}

static std::vector<DriverInfo> collect_drivers() {
    std::vector<DriverInfo> out;
    ULONG ret = 0;
    NTSTATUS s = NtQuerySystemInformation((SYSTEM_INFORMATION_CLASS)SystemModuleInformation_Class,
                                          nullptr, 0, &ret);
    // Expected to fail with STATUS_INFO_LENGTH_MISMATCH (0xC0000004) and
    // return the required size in 'ret'.
    if (ret == 0) {
        return out;
    }
    std::vector<uint8_t> buf(ret);
    s = NtQuerySystemInformation((SYSTEM_INFORMATION_CLASS)SystemModuleInformation_Class,
                                 buf.data(), (ULONG)buf.size(), &ret);
    if (s != STATUS_SUCCESS) {
        return out;
    }
    auto * mods = (RTL_PROCESS_MODULES *)buf.data();
    out.reserve(mods->NumberOfModules);
    for (ULONG i = 0; i < mods->NumberOfModules; ++i) {
        const auto & m = mods->Modules[i];
        DriverInfo d{};
        const char * full = (const char *)m.FullPathName;
        const char * base = full + m.OffsetToFileName;
        d.name       = base;
        d.image_size = m.ImageSize;
        d.image_base = (uintptr_t)m.ImageBase;
        out.push_back(d);
    }
    return out;
}

static std::vector<PageFile> collect_pagefiles() {
    std::vector<PageFile> out;
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const uint64_t page = si.dwPageSize;

    auto cb = [](LPVOID context, PENUM_PAGE_FILE_INFORMATION info,
                 LPCWSTR filename) -> BOOL {
        auto * vec  = (std::vector<PageFile> *)context;
        // Reconstruct page_size locally; the callback can't capture state.
        SYSTEM_INFO si2{};
        GetSystemInfo(&si2);
        const uint64_t page_size = si2.dwPageSize;
        PageFile pf;
        pf.path        = ws_to_utf8(filename);
        pf.total_bytes = (uint64_t)info->TotalSize * page_size;
        pf.used_bytes  = (uint64_t)info->TotalInUse * page_size;
        pf.peak_bytes  = (uint64_t)info->PeakUsage * page_size;
        vec->push_back(std::move(pf));
        return TRUE;
    };
    EnumPageFilesW(cb, &out);
    (void)page;
    return out;
}

// --- formatting -----------------------------------------------------------

struct Options {
    bool        json            = false;
    bool        processes_only  = false;
    bool        no_processes    = false;
    bool        no_services     = false;
    bool        show_all_procs  = false;
    int         top_n           = 30;
    bool        help            = false;
};

static void print_human(const PhysicalMem & p, const KernelMem & k,
                        const MemList & ml,
                        const std::vector<PageFile> & pagefiles,
                        const std::vector<ProcInfo> & procs_sorted,
                        const std::vector<ServiceInfo> & all_services,
                        const std::map<DWORD, std::vector<ServiceInfo>> & services_by_pid,
                        const std::vector<DriverInfo> & drivers_top,
                        const Options & opts, bool admin) {
    auto line = []{ printf("--------------------------------------------------------------------------------\n"); };

    if (!opts.processes_only) {
        printf("WXEmem -- Windows memory snapshot\n");
        line();
        printf(" Physical RAM\n");
        printf("   %-22s %12s\n",                 "Total:",     format_bytes(p.total,     12).c_str());
        printf("   %-22s %12s   (%u%% load)\n",  "Used:",      format_bytes(p.used,      12).c_str(), p.memory_load_pct);
        printf("   %-22s %12s\n",                 "Available:", format_bytes(p.available, 12).c_str());
        line();
        printf(" Commit charge\n");
        printf("   %-22s %12s\n",                 "Used:",     format_bytes(p.commit_total, 12).c_str());
        printf("   %-22s %12s   (physical RAM + page file backing)\n",
                                                  "Limit:",    format_bytes(p.commit_limit, 12).c_str());
        printf("   %-22s %12s\n",                 "Headroom:",
            format_bytes(p.commit_limit > p.commit_total ? p.commit_limit - p.commit_total : 0, 12).c_str());
        line();

        if (k.available) {
            printf(" Kernel-side memory\n");
            printf("   %-22s %12s   (avail: %s)\n",
                   "Paged pool:",          format_bytes(k.paged_pool,        12).c_str(),
                                           format_bytes(k.paged_pool_avail).c_str());
            printf("   %-22s %12s\n",
                   "Resident paged pool:", format_bytes(k.resident_paged_pool, 12).c_str());
            printf("   %-22s %12s\n",
                   "Non-paged pool:",      format_bytes(k.non_paged_pool,    12).c_str());
            printf("   %-22s %12s\n",
                   "Driver code (res):",   format_bytes(k.driver_code,       12).c_str());
            printf("   %-22s %12s\n",
                   "Kernel code (res):",   format_bytes(k.system_code,       12).c_str());
            printf("   %-22s %12s\n",
                   "System cache (res):",  format_bytes(k.system_cache,      12).c_str());
            line();
        } else {
            printf(" Kernel-side memory: <NtQuerySystemInformation failed>\n");
            line();
        }

        if (ml.available) {
            printf(" Memory-list breakdown (reclaimable)\n");
            printf("   %-22s %12s\n",
                   "Free:",     format_bytes(ml.free_bytes,     12).c_str());
            printf("   %-22s %12s\n",
                   "Zero:",     format_bytes(ml.zero_bytes,     12).c_str());
            printf("   %-22s %12s   (will be written to page file before reuse)\n",
                   "Modified:", format_bytes(ml.modified_bytes, 12).c_str());
            printf("   %-22s %12s   (reusable cache; counts as 'available')\n",
                   "Standby:",  format_bytes(ml.standby_bytes,  12).c_str());
            line();
        }

        if (!pagefiles.empty()) {
            printf(" Page files (%zu)\n", pagefiles.size());
            uint64_t total_size = 0, total_used = 0, total_peak = 0;
            for (const auto & pf : pagefiles) {
                printf("   %-22s %12s   used %12s   peak %12s\n",
                       pf.path.c_str(),
                       format_bytes(pf.total_bytes, 12).c_str(),
                       format_bytes(pf.used_bytes,  12).c_str(),
                       format_bytes(pf.peak_bytes,  12).c_str());
                total_size += pf.total_bytes;
                total_used += pf.used_bytes;
                total_peak += pf.peak_bytes;
            }
            if (pagefiles.size() > 1) {
                printf("   %-22s %12s   used %12s   peak %12s\n",
                       "(total)",
                       format_bytes(total_size, 12).c_str(),
                       format_bytes(total_used, 12).c_str(),
                       format_bytes(total_peak, 12).c_str());
            }
            line();
        } else {
            printf(" Page files: none configured (commit backed by physical RAM only)\n");
            line();
        }

        if (!drivers_top.empty()) {
            printf(" Top drivers / kernel modules by image size (%zu loaded)\n", drivers_top.size());
            int shown = (int)std::min<size_t>(drivers_top.size(), 15);
            for (int i = 0; i < shown; ++i) {
                std::string orig = hotpatch_original(drivers_top[i].name);
                if (!orig.empty()) {
                    printf("   %10s   %s   (hotpatch for %s)\n",
                           format_bytes(drivers_top[i].image_size, 10).c_str(),
                           drivers_top[i].name.c_str(),
                           orig.c_str());
                } else {
                    printf("   %10s   %s\n",
                           format_bytes(drivers_top[i].image_size, 10).c_str(),
                           drivers_top[i].name.c_str());
                }
            }
            line();
        }
    }

    if (!opts.no_processes) {
        size_t to_show = opts.show_all_procs ? procs_sorted.size()
                                              : std::min<size_t>(procs_sorted.size(), (size_t)opts.top_n);
        printf(" Top %zu processes by working set (of %zu total)\n", to_show, procs_sorted.size());
        printf("    %6s  %15s  %15s  %15s   %s\n",
               "PID", "WorkingSet", "Private", "PageFile", "Image / [services]");
        for (size_t i = 0; i < to_show; ++i) {
            const ProcInfo & pi = procs_sorted[i];
            printf("    %6lu  %15s  %15s  %15s   %s",
                   (unsigned long)pi.pid,
                   format_bytes(pi.ws,            12).c_str(),
                   format_bytes(pi.private_bytes, 12).c_str(),
                   format_bytes(pi.pagefile,      12).c_str(),
                   pi.name.c_str());
            auto it = services_by_pid.find(pi.pid);
            if (it != services_by_pid.end() && !it->second.empty()) {
                printf("  [");
                for (size_t s = 0; s < it->second.size(); ++s) {
                    const ServiceInfo & svc = it->second[s];
                    // Suppress tag for Auto (the common case) and for
                    // Disabled (extremely rare here: a Running service whose
                    // config was flipped to Disabled after start; will not
                    // auto-start next boot but is irrelevant to this snapshot).
                    const bool tagged = (svc.start_kind != ServiceStartKind::Auto     &&
                                         svc.start_kind != ServiceStartKind::Disabled &&
                                         svc.start_kind != ServiceStartKind::Unknown);
                    if (tagged) {
                        printf("%s%s:%s", (s ? "," : ""),
                               svc.name.c_str(),
                               start_kind_short(svc.start_kind));
                    } else {
                        printf("%s%s", (s ? "," : ""), svc.name.c_str());
                    }
                    if (s >= 6 && it->second.size() > 7) {
                        printf(",...+%zu", it->second.size() - s - 1);
                        break;
                    }
                }
                printf("]");
            }
            printf("\n");
        }
        line();
    }

    if (!opts.no_processes) {
        printf(" Service start-kind tag legend: unmarked = auto, :demand, :auto-d\n");
        line();
    }

    // Running services section: always show services regardless of where their
    // host process landed in the WS-sorted process table. Useful because
    // typical svchost.exe hosts have small WS (~15-50 MB) and rarely make
    // the default --top 30 cutoff.
    if (!opts.no_services && !all_services.empty()) {
        // Tally ALL active services (user-mode + kernel drivers) by start
        // kind for the summary. Drivers won't appear in the per-host table
        // below (they have no PID), but they DO contribute to boot charge,
        // so the counts include them.
        size_t n_auto = 0, n_auto_d = 0, n_demand = 0;
        size_t n_boot = 0, n_system = 0, n_other = 0;
        size_t n_drivers = 0, n_usermode = 0;
        for (const auto & svc : all_services) {
            const bool is_driver = (svc.service_type &
                                    (SERVICE_KERNEL_DRIVER | SERVICE_FILE_SYSTEM_DRIVER)) != 0;
            if (is_driver) ++n_drivers; else ++n_usermode;
            switch (svc.start_kind) {
                case ServiceStartKind::Auto:        ++n_auto;   break;
                case ServiceStartKind::AutoDelayed: ++n_auto_d; break;
                case ServiceStartKind::Demand:      ++n_demand; break;
                case ServiceStartKind::Boot:        ++n_boot;   break;
                case ServiceStartKind::System:      ++n_system; break;
                default:                            ++n_other;  break;
            }
        }

        printf(" Running services (%zu total: %zu user-mode + %zu kernel drivers, across %zu host processes)\n",
               all_services.size(), n_usermode, n_drivers, services_by_pid.size());
        printf("   By start kind:");
        bool first = true;
        auto tally = [&](const char * label, size_t n) {
            if (!n) return;
            printf("%s %zu %s", first ? "" : " |", n, label);
            first = false;
        };
        tally("auto",          n_auto);
        tally("auto-delayed",  n_auto_d);
        tally("demand",        n_demand);
        tally("system",        n_system);
        tally("boot",          n_boot);
        tally("other",         n_other);
        printf("\n");
        line();

        // Look up host process WS for sort + display.
        std::map<DWORD, const ProcInfo *> proc_by_pid;
        for (const auto & pi : procs_sorted) {
            proc_by_pid[pi.pid] = &pi;
        }

        // Build a sortable list of user-mode service hosts (kernel drivers
        // have no PID/host process and are not in services_by_pid).
        struct HostRow {
            DWORD pid;
            uint64_t ws;
            std::string name;
            const std::vector<ServiceInfo> * services;
        };
        std::vector<HostRow> hosts;
        hosts.reserve(services_by_pid.size());
        for (const auto & kv : services_by_pid) {
            HostRow h{};
            h.pid = kv.first;
            h.services = &kv.second;
            auto pit = proc_by_pid.find(kv.first);
            if (pit != proc_by_pid.end()) {
                h.ws   = pit->second->ws;
                h.name = pit->second->name;
            } else {
                h.ws = 0;
                h.name = "<unknown>";
            }
            hosts.push_back(std::move(h));
        }
        std::sort(hosts.begin(), hosts.end(),
                  [](const HostRow & a, const HostRow & b) { return a.ws > b.ws; });

        printf("    %6s  %12s   %-22s   %s\n",
               "PID", "HostWS", "Host", "Services");
        for (const auto & h : hosts) {
            printf("    %6lu  %12s   %-22.22s   ",
                   (unsigned long)h.pid,
                   format_bytes(h.ws, 12).c_str(),
                   h.name.c_str());
            for (size_t s = 0; s < h.services->size(); ++s) {
                const ServiceInfo & svc = (*h.services)[s];
                const bool tagged = (svc.start_kind != ServiceStartKind::Auto     &&
                                     svc.start_kind != ServiceStartKind::Disabled &&
                                     svc.start_kind != ServiceStartKind::Unknown);
                if (tagged) {
                    printf("%s%s:%s", (s ? ", " : ""),
                           svc.name.c_str(),
                           start_kind_short(svc.start_kind));
                } else {
                    printf("%s%s", (s ? ", " : ""), svc.name.c_str());
                }
            }
            printf("\n");
        }
        line();
    }

    if (!admin) {
        printf(" Tip: re-run elevated (admin) for full kernel pool + driver image breakdown\n");
    }
}

static void print_json(const PhysicalMem & p, const KernelMem & k,
                       const MemList & ml,
                       const std::vector<PageFile> & pagefiles,
                       const std::vector<ProcInfo> & procs_sorted,
                       const std::vector<ServiceInfo> & all_services,
                       const std::map<DWORD, std::vector<ServiceInfo>> & services_by_pid,
                       const std::vector<DriverInfo> & drivers_top,
                       const Options & opts, bool admin) {
    printf("{\n");
    printf("  \"admin\": %s,\n", admin ? "true" : "false");
    printf("  \"physical\": {\n");
    printf("    \"total_bytes\": %llu,\n",     (unsigned long long)p.total);
    printf("    \"used_bytes\": %llu,\n",      (unsigned long long)p.used);
    printf("    \"available_bytes\": %llu,\n", (unsigned long long)p.available);
    printf("    \"memory_load_pct\": %u,\n",   (unsigned)p.memory_load_pct);
    printf("    \"commit_used_bytes\": %llu,\n",  (unsigned long long)p.commit_total);
    printf("    \"commit_limit_bytes\": %llu,\n", (unsigned long long)p.commit_limit);
    printf("    \"total_virtual_bytes\": %llu,\n",(unsigned long long)p.total_virtual);
    printf("    \"avail_virtual_bytes\": %llu\n", (unsigned long long)p.avail_virtual);
    printf("  },\n");

    if (k.available) {
        printf("  \"kernel\": {\n");
        printf("    \"paged_pool_bytes\": %llu,\n",          (unsigned long long)k.paged_pool);
        printf("    \"paged_pool_avail_bytes\": %llu,\n",    (unsigned long long)k.paged_pool_avail);
        printf("    \"resident_paged_pool_bytes\": %llu,\n", (unsigned long long)k.resident_paged_pool);
        printf("    \"non_paged_pool_bytes\": %llu,\n",      (unsigned long long)k.non_paged_pool);
        printf("    \"driver_code_bytes\": %llu,\n",         (unsigned long long)k.driver_code);
        printf("    \"system_code_bytes\": %llu,\n",         (unsigned long long)k.system_code);
        printf("    \"system_cache_bytes\": %llu\n",         (unsigned long long)k.system_cache);
        printf("  },\n");
    } else {
        printf("  \"kernel\": null,\n");
    }

    if (ml.available) {
        printf("  \"memory_list\": {\n");
        printf("    \"free_bytes\": %llu,\n",     (unsigned long long)ml.free_bytes);
        printf("    \"zero_bytes\": %llu,\n",     (unsigned long long)ml.zero_bytes);
        printf("    \"modified_bytes\": %llu,\n", (unsigned long long)ml.modified_bytes);
        printf("    \"standby_bytes\": %llu\n",   (unsigned long long)ml.standby_bytes);
        printf("  },\n");
    } else {
        printf("  \"memory_list\": null,\n");
    }

    printf("  \"pagefiles\": [\n");
    for (size_t i = 0; i < pagefiles.size(); ++i) {
        const auto & pf = pagefiles[i];
        printf("    {\"path\": \"%s\", \"total_bytes\": %llu, \"used_bytes\": %llu, \"peak_bytes\": %llu}%s\n",
               json_escape(pf.path).c_str(),
               (unsigned long long)pf.total_bytes,
               (unsigned long long)pf.used_bytes,
               (unsigned long long)pf.peak_bytes,
               (i + 1 < pagefiles.size() ? "," : ""));
    }
    printf("  ],\n");

    printf("  \"drivers_top\": [\n");
    for (size_t i = 0; i < drivers_top.size(); ++i) {
        const auto & d = drivers_top[i];
        std::string orig = hotpatch_original(d.name);
        printf("    {\"name\": \"%s\", \"image_size_bytes\": %llu, \"image_base\": \"0x%llx\"",
               json_escape(d.name).c_str(),
               (unsigned long long)d.image_size,
               (unsigned long long)d.image_base);
        if (!orig.empty()) {
            printf(", \"hotpatch_for\": \"%s\"", json_escape(orig).c_str());
        }
        printf("}%s\n", (i + 1 < drivers_top.size() ? "," : ""));
    }
    printf("  ],\n");

    // Full services list (user-mode + kernel drivers) with start kind for
    // boot-charge analysis.
    {
        size_t n_auto = 0, n_auto_d = 0, n_demand = 0;
        size_t n_boot = 0, n_system = 0, n_other = 0;
        size_t n_drivers = 0, n_usermode = 0;
        for (const auto & svc : all_services) {
            const bool is_driver = (svc.service_type &
                                    (SERVICE_KERNEL_DRIVER | SERVICE_FILE_SYSTEM_DRIVER)) != 0;
            if (is_driver) ++n_drivers; else ++n_usermode;
            switch (svc.start_kind) {
                case ServiceStartKind::Auto:        ++n_auto;   break;
                case ServiceStartKind::AutoDelayed: ++n_auto_d; break;
                case ServiceStartKind::Demand:      ++n_demand; break;
                case ServiceStartKind::Boot:        ++n_boot;   break;
                case ServiceStartKind::System:      ++n_system; break;
                default:                            ++n_other;  break;
            }
        }
        printf("  \"services_summary\": {\n");
        printf("    \"total\": %zu, \"user_mode\": %zu, \"kernel_driver\": %zu,\n",
               all_services.size(), n_usermode, n_drivers);
        printf("    \"by_start_kind\": {\"auto\": %zu, \"auto_delayed\": %zu, \"demand\": %zu, "
               "\"system\": %zu, \"boot\": %zu, \"other\": %zu}\n",
               n_auto, n_auto_d, n_demand, n_system, n_boot, n_other);
        printf("  },\n");

        printf("  \"services\": [\n");
        for (size_t i = 0; i < all_services.size(); ++i) {
            const auto & svc = all_services[i];
            const bool is_driver = (svc.service_type &
                                    (SERVICE_KERNEL_DRIVER | SERVICE_FILE_SYSTEM_DRIVER)) != 0;
            printf("    {\"name\":\"%s\",\"display\":\"%s\",\"start\":\"%s\",\"kind\":\"%s\"",
                   json_escape(svc.name).c_str(),
                   json_escape(svc.display_name).c_str(),
                   start_kind_short(svc.start_kind),
                   is_driver ? "driver" : "user");
            if (svc.pid != 0) printf(",\"pid\":%lu", (unsigned long)svc.pid);
            printf("}%s\n", (i + 1 < all_services.size() ? "," : ""));
        }
        printf("  ],\n");
    }

    size_t to_show = opts.no_processes ? 0 :
                     (opts.show_all_procs ? procs_sorted.size()
                                          : std::min<size_t>(procs_sorted.size(), (size_t)opts.top_n));

    printf("  \"processes\": [\n");
    for (size_t i = 0; i < to_show; ++i) {
        const auto & pi = procs_sorted[i];
        printf("    {\"pid\": %lu, \"name\": \"%s\", \"ws_bytes\": %llu, \"private_bytes\": %llu, \"pagefile_bytes\": %llu",
               (unsigned long)pi.pid,
               json_escape(pi.name).c_str(),
               (unsigned long long)pi.ws,
               (unsigned long long)pi.private_bytes,
               (unsigned long long)pi.pagefile);
        auto it = services_by_pid.find(pi.pid);
        if (it != services_by_pid.end() && !it->second.empty()) {
            printf(", \"services\": [");
            for (size_t s = 0; s < it->second.size(); ++s) {
                const ServiceInfo & svc = it->second[s];
                printf("%s{\"name\":\"%s\",\"start\":\"%s\"}",
                       (s ? "," : ""),
                       json_escape(svc.name).c_str(),
                       start_kind_short(svc.start_kind));
            }
            printf("]");
        }
        printf("}%s\n", (i + 1 < to_show ? "," : ""));
    }
    printf("  ]\n");
    printf("}\n");
}

// --- CLI ------------------------------------------------------------------

static void print_usage() {
    fprintf(stderr,
        "WXEmem — Windows memory inspector\n"
        "Usage: wxemem [options]\n"
        "\n"
        "Options:\n"
        "  --json              emit JSON instead of human-readable text\n"
        "  --top N             show top N processes by working set (default 30)\n"
        "  --all               show all processes (overrides --top)\n"
        "  --processes-only    only print the process table\n"
        "  --no-processes      omit the process table\n"
        "  -h, --help          show this help and exit\n"
        "\n"
        "Some breakdowns (kernel pools, driver image list, protected process names)\n"
        "require elevation. Run from an admin shell for the full report.\n");
}

static bool parse_args(int argc, char ** argv, Options & opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") { opts.help = true; }
        else if (a == "--json")         { opts.json = true; }
        else if (a == "--processes-only"){ opts.processes_only = true; }
        else if (a == "--no-processes") { opts.no_processes = true; }
        else if (a == "--no-services")  { opts.no_services = true; }
        else if (a == "--all")          { opts.show_all_procs = true; }
        else if (a == "--top") {
            if (++i >= argc) { fprintf(stderr, "wxemem: --top requires N\n"); return false; }
            opts.top_n = atoi(argv[i]);
            if (opts.top_n <= 0) { fprintf(stderr, "wxemem: --top must be positive\n"); return false; }
        }
        else {
            fprintf(stderr, "wxemem: unknown option '%s'\n", a.c_str());
            return false;
        }
    }
    if (opts.processes_only && opts.no_processes) {
        fprintf(stderr, "wxemem: --processes-only and --no-processes are mutually exclusive\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) { print_usage(); return 2; }
    if (opts.help) { print_usage(); return 0; }

    const bool admin = is_running_as_admin();

    PhysicalMem p   = collect_physical();
    KernelMem   k   = collect_kernel();
    MemList     ml  = collect_memlist();
    std::vector<PageFile> pagefiles = collect_pagefiles();

    std::vector<ProcInfo> procs = collect_processes();
    std::sort(procs.begin(), procs.end(),
              [](const ProcInfo & a, const ProcInfo & b) { return a.ws > b.ws; });

    std::vector<ServiceInfo> svcs = collect_services();
    std::map<DWORD, std::vector<ServiceInfo>> svc_by_pid;
    for (auto & s : svcs) {
        if (s.pid != 0) svc_by_pid[s.pid].push_back(s);
    }

    std::vector<DriverInfo> drivers = collect_drivers();
    std::sort(drivers.begin(), drivers.end(),
              [](const DriverInfo & a, const DriverInfo & b) { return a.image_size > b.image_size; });
    // Cap the human display to 15; JSON receives all of them.
    std::vector<DriverInfo> drivers_top = drivers;
    if (drivers_top.size() > 64) drivers_top.resize(64);

    if (opts.json) {
        print_json(p, k, ml, pagefiles, procs, svcs, svc_by_pid, drivers_top, opts, admin);
    } else {
        print_human(p, k, ml, pagefiles, procs, svcs, svc_by_pid, drivers_top, opts, admin);
    }
    return 0;
}
