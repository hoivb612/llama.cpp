// wxememdiff -- diff two WXEmem JSON snapshots and attribute memory growth.
//
// Given two snapshots produced by `wxemem --json` (snapshot A = baseline,
// snapshot B = later / newer build or stage), this tool reports:
//
//   (a) overall "Used" physical RAM growth between A and B, and
//   (b) an attribution of that growth to likely components:
//         - user-mode processes / services (by private working set)
//         - kernel data (non-paged pool, resident paged pool, driver code,
//           system cache)
//         - new / removed drivers
//         - per-pool-tag non-paged pool growth (which ExAllocatePoolWithTag
//           tag is responsible), when both snapshots include `pool_tags`
//
// The goal is to zero in on *what* changed from stage to stage and build to
// build. Output is plain, aligned text so it stays diff-able and consistent
// over time.
//
// Usage:
//   wxememdiff <jsonA> <jsonB> [--top N] [--out FILE]
//
//     <jsonA>    baseline snapshot (the "before")
//     <jsonB>    comparison snapshot (the "after")
//     --top N    number of rows to show in top-N tables (default 15)
//     --out FILE also write the report to FILE (in addition to stdout)

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using i64  = long long;

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

// Human-readable byte size, e.g. "12.34 MB". Sign is preserved so it works for
// deltas ("-4.00 KB").
static std::string humanBytes(i64 bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    double v = static_cast<double>(bytes);
    bool neg = v < 0;
    if (neg) v = -v;

    int u = 0;
    while (v >= 1024.0 && u < 5) {
        v /= 1024.0;
        ++u;
    }

    char buf[64];
    if (u == 0)
        std::snprintf(buf, sizeof(buf), "%s%.0f %s", neg ? "-" : "", v, units[u]);
    else
        std::snprintf(buf, sizeof(buf), "%s%.2f %s", neg ? "-" : "", v, units[u]);
    return buf;
}

// Signed byte delta with an explicit leading '+' for growth.
static std::string signedBytes(i64 bytes) {
    if (bytes >= 0) return "+" + humanBytes(bytes);
    return humanBytes(bytes);  // humanBytes already prints the '-'
}

// Percentage change from a->b, guarding against divide-by-zero.
static std::string pct(i64 a, i64 b) {
    if (a == 0) {
        if (b == 0) return "  0.0%";
        return "   new";
    }
    double p = 100.0 * (static_cast<double>(b) - static_cast<double>(a)) /
               static_cast<double>(a);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%+.1f%%", p);
    return buf;
}

// Right-pad a string to width w.
static std::string padR(const std::string& s, size_t w) {
    if (s.size() >= w) return s;
    return s + std::string(w - s.size(), ' ');
}

// Left-pad a string to width w.
static std::string padL(const std::string& s, size_t w) {
    if (s.size() >= w) return s;
    return std::string(w - s.size(), ' ') + s;
}

static std::string rule() { return std::string(78, '-'); }

// ---------------------------------------------------------------------------
// Snapshot model
// ---------------------------------------------------------------------------

struct Physical {
    i64 total = 0, used = 0, avail = 0, commit_used = 0, commit_limit = 0;
    i64 load_pct = 0;
};

struct Kernel {
    i64 paged_pool = 0, resident_paged_pool = 0, non_paged_pool = 0;
    i64 driver_code = 0, system_code = 0, system_cache = 0;
};

struct MemList {
    i64 free_ = 0, zero_ = 0, modified_ = 0, standby_ = 0;
};

struct Proc {
    i64 pid = 0;
    std::string name;
    i64 ws = 0, pws = 0, priv = 0, pagefile = 0;
    bool pws_fallback = false;  // true if pws was substituted from ws_bytes
    std::vector<std::string> services;
};

// Per-process-name aggregate (handles many svchost.exe / etc.).
struct ProcAgg {
    int count = 0;
    i64 ws = 0, pws = 0, priv = 0;
};

struct Svc {
    std::string name, display, start, kind;
    i64 pid = 0;
};

struct Drv {
    std::string name;
    i64 size = 0;
};

struct PoolTagInfo {
    std::string tag;         // 4-char tag string
    i64 nonpaged = 0;        // nonpaged_bytes
    i64 paged = 0;           // paged_bytes
    i64 np_allocs = 0;       // nonpaged_allocs (outstanding)
};

struct Snapshot {
    std::string label;
    bool admin = false;
    Physical phys;
    Kernel   kern;
    MemList  mem;

    std::vector<Proc> procs;
    std::map<std::string, ProcAgg> procByName;  // aggregated

    std::map<std::string, Svc> services;        // by service name
    json services_summary;                      // kept raw for count deltas

    std::map<std::string, i64> drivers;         // name -> image_size_bytes

    bool has_pool_tags = false;
    std::map<std::string, PoolTagInfo> pool_tags;  // keyed by tag_hex

    // Private-WS availability tracking. Older wxemem captures don't emit
    // "private_ws_bytes"; when absent we fall back to ws_bytes so the diff
    // still conveys the shape of the change (see pws_fallback below).
    int proc_count = 0;          // number of process records parsed
    int pws_missing_count = 0;   // records lacking private_ws_bytes (fell back to ws)
};

static i64 jget(const json& j, const char* key, i64 def = 0) {
    if (j.contains(key) && j[key].is_number()) return j[key].get<i64>();
    return def;
}

static std::string jstr(const json& j, const char* key, const char* def = "") {
    if (j.contains(key) && j[key].is_string()) return j[key].get<std::string>();
    return def;
}

static bool loadSnapshot(const std::string& path, Snapshot& s, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "cannot open file: " + path;
        return false;
    }
    json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        err = std::string("JSON parse error in ") + path + ": " + e.what();
        return false;
    }

    s.label = path;
    s.admin = j.value("admin", false);

    if (j.contains("physical")) {
        const json& p = j["physical"];
        s.phys.total        = jget(p, "total_bytes");
        s.phys.used         = jget(p, "used_bytes");
        s.phys.avail        = jget(p, "available_bytes");
        s.phys.load_pct     = jget(p, "memory_load_pct");
        s.phys.commit_used  = jget(p, "commit_used_bytes");
        s.phys.commit_limit = jget(p, "commit_limit_bytes");
    }
    if (j.contains("kernel")) {
        const json& k = j["kernel"];
        s.kern.paged_pool          = jget(k, "paged_pool_bytes");
        s.kern.resident_paged_pool = jget(k, "resident_paged_pool_bytes");
        s.kern.non_paged_pool      = jget(k, "non_paged_pool_bytes");
        s.kern.driver_code         = jget(k, "driver_code_bytes");
        s.kern.system_code         = jget(k, "system_code_bytes");
        s.kern.system_cache        = jget(k, "system_cache_bytes");
    }
    if (j.contains("memory_list")) {
        const json& m = j["memory_list"];
        s.mem.free_     = jget(m, "free_bytes");
        s.mem.zero_     = jget(m, "zero_bytes");
        s.mem.modified_ = jget(m, "modified_bytes");
        s.mem.standby_  = jget(m, "standby_bytes");
    }

    if (j.contains("processes") && j["processes"].is_array()) {
        for (const json& pj : j["processes"]) {
            Proc p;
            p.pid      = jget(pj, "pid");
            p.name     = jstr(pj, "name");
            p.ws       = jget(pj, "ws_bytes");
            p.priv     = jget(pj, "private_bytes");
            p.pagefile = jget(pj, "pagefile_bytes");
            // PrivateWS is the resident-private metric this tool attributes on.
            // Older captures omit it entirely; fall back to WorkingSet so the
            // comparison still reflects change direction/magnitude (less
            // accurate: WS includes shared resident pages).
            if (pj.contains("private_ws_bytes") && pj["private_ws_bytes"].is_number()) {
                p.pws = jget(pj, "private_ws_bytes");
            } else {
                p.pws = p.ws;
                p.pws_fallback = true;
                s.pws_missing_count++;
            }
            s.proc_count++;
            if (pj.contains("services") && pj["services"].is_array()) {
                for (const json& sv : pj["services"])
                    p.services.push_back(jstr(sv, "name"));
            }
            ProcAgg& a = s.procByName[p.name];
            a.count++;
            a.ws   += p.ws;
            a.pws  += p.pws;
            a.priv += p.priv;
            s.procs.push_back(std::move(p));
        }
    }

    s.services_summary = j.value("services_summary", json::object());
    if (j.contains("services") && j["services"].is_array()) {
        for (const json& sj : j["services"]) {
            Svc sv;
            sv.name    = jstr(sj, "name");
            sv.display = jstr(sj, "display");
            sv.start   = jstr(sj, "start");
            sv.kind    = jstr(sj, "kind");
            sv.pid     = jget(sj, "pid");
            if (!sv.name.empty()) s.services[sv.name] = sv;
        }
    }

    if (j.contains("drivers_top") && j["drivers_top"].is_array()) {
        for (const json& dj : j["drivers_top"]) {
            std::string name = jstr(dj, "name");
            if (!name.empty()) s.drivers[name] = jget(dj, "image_size_bytes");
        }
    }

    if (j.contains("pool_tags") && j["pool_tags"].is_array()) {
        s.has_pool_tags = true;
        for (const json& tj : j["pool_tags"]) {
            PoolTagInfo pt;
            pt.tag       = jstr(tj, "tag");
            pt.nonpaged  = jget(tj, "nonpaged_bytes");
            pt.paged     = jget(tj, "paged_bytes");
            pt.np_allocs = jget(tj, "nonpaged_allocs");
            // Key by tag_hex so identical tag strings from different raw values
            // stay distinct; fall back to the tag string if hex is missing.
            std::string key = jstr(tj, "tag_hex");
            if (key.empty()) key = pt.tag;
            s.pool_tags[key] = pt;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

struct NamedDelta {
    std::string name;
    i64 a = 0, b = 0, delta = 0;
    int countA = 0, countB = 0;
};

static void writeReport(std::ostream& o, const Snapshot& A, const Snapshot& B,
                        int topN) {
    o << "WXEmem diff -- memory growth attribution\n";
    o << rule() << "\n";
    o << " A (baseline): " << A.label << "\n";
    o << " B (compare):  " << B.label << "\n";
    if (!A.admin || !B.admin)
        o << " NOTE: at least one snapshot was captured without admin rights; "
             "kernel/driver figures may be incomplete.\n";
    o << "\n";

    // -- Physical memory summary ------------------------------------------
    o << " Physical RAM\n";
    o << "   " << padR("metric", 16) << padL("A", 14) << padL("B", 14)
      << padL("delta", 16) << padL("chg", 10) << "\n";

    auto physRow = [&](const char* name, i64 a, i64 b) {
        o << "   " << padR(name, 16) << padL(humanBytes(a), 14)
          << padL(humanBytes(b), 14) << padL(signedBytes(b - a), 16)
          << padL(pct(a, b), 10) << "\n";
    };
    physRow("Total",        A.phys.total,        B.phys.total);
    physRow("Used",         A.phys.used,         B.phys.used);
    physRow("Available",    A.phys.avail,        B.phys.avail);
    physRow("Commit used",  A.phys.commit_used,  B.phys.commit_used);
    physRow("Commit limit", A.phys.commit_limit, B.phys.commit_limit);
    o << "   " << padR("Load", 16) << padL(std::to_string(A.phys.load_pct) + "%", 14)
      << padL(std::to_string(B.phys.load_pct) + "%", 14)
      << padL(std::to_string(B.phys.load_pct - A.phys.load_pct) + " pt", 16) << "\n";
    o << "\n";

    // -- Headline ----------------------------------------------------------
    i64 usedDelta = B.phys.used - A.phys.used;
    o << rule() << "\n";
    o << " HEADLINE: Used physical RAM " << (usedDelta >= 0 ? "grew" : "shrank")
      << " by " << signedBytes(usedDelta) << "  (" << pct(A.phys.used, B.phys.used)
      << ")\n";
    o << rule() << "\n\n";

    // -- Metric selection (symmetric) --------------------------------------
    // Default attribution metric is PrivateWS (resident & private). If EITHER
    // capture is missing private_ws_bytes for any process, both sides fall
    // back to WorkingSet so the comparison stays apples-to-apples (both then
    // include shared resident pages). We never mix metrics across sides.
    const bool useWS = (A.pws_missing_count > 0 || B.pws_missing_count > 0);
    auto M = [useWS](const ProcAgg& a) -> i64 { return useWS ? a.ws : a.pws; };
    const char* metricName = useWS ? "WorkingSet" : "private WS";

    // -- Attribution: user vs kernel vs other ------------------------------
    i64 userDelta = 0;
    {
        // Sum of per-process working-set metric (PrivateWS, or WS on fallback).
        i64 aPWS = 0, bPWS = 0;
        for (const auto& kv : A.procByName) aPWS += M(kv.second);
        for (const auto& kv : B.procByName) bPWS += M(kv.second);
        userDelta = bPWS - aPWS;
    }
    i64 kNonPaged = B.kern.non_paged_pool      - A.kern.non_paged_pool;
    i64 kPagedRes = B.kern.resident_paged_pool - A.kern.resident_paged_pool;
    i64 kDrvCode  = B.kern.driver_code         - A.kern.driver_code;
    i64 kSysCache = B.kern.system_cache        - A.kern.system_cache;
    i64 kernDelta = kNonPaged + kPagedRes + kDrvCode + kSysCache;
    i64 otherDelta = usedDelta - userDelta - kernDelta;

    o << " Growth attribution (approximate; components may overlap)\n";
    auto attrRow = [&](const char* name, i64 v, const char* note) {
        o << "   " << padR(name, 30) << padL(signedBytes(v), 16) << "   " << note
          << "\n";
    };
    attrRow("User-mode processes", userDelta,
            useWS ? "sum of process working set (PrivateWS unavailable)"
                  : "sum of process private working set");
    attrRow("Kernel (subtotal)", kernDelta, "non-paged + paged(res) + drv + cache");
    attrRow("  non-paged pool", kNonPaged, "fully resident kernel pool");
    attrRow("  paged pool (resident)", kPagedRes, "resident portion of paged pool");
    attrRow("  driver code (resident)", kDrvCode, "loaded driver images");
    attrRow("  system cache (resident)", kSysCache, "file cache resident pages");
    attrRow("Other / unaccounted", otherDelta,
            "VBS / Hyper-V / GPU / driver allocs / etc.");
    o << "\n";
    o << "   (User + Kernel + Other == Used delta by construction. 'Other' is the\n";
    o << "    residual not captured by process WS or the kernel counters above.)\n";
    o << "\n";

    // -- User-mode process attribution -------------------------------------
    o << rule() << "\n";
    o << " User-mode processes (aggregated by image name, metric = " << metricName
      << ")\n";
    o << rule() << "\n";

    // When either capture lacks private_ws_bytes we fall back to WorkingSet
    // for BOTH sides (symmetric) so deltas stay apples-to-apples. WS includes
    // shared resident pages, so magnitudes are larger than PrivateWS would be.
    auto pwsNote = [&](const char* tag, const Snapshot& s) {
        if (s.pws_missing_count == 0) {
            o << "   [ ] " << tag << ": has private_ws_bytes "
                 "(using WorkingSet anyway for symmetry)\n";
            return;
        }
        o << "   [!] " << tag << " (" << s.label << "):\n";
        if (s.pws_missing_count == s.proc_count) {
            o << "       no 'private_ws_bytes' in capture (old wxemem build)\n";
        } else {
            o << "       " << s.pws_missing_count << " of " << s.proc_count
              << " processes lack 'private_ws_bytes'\n";
        }
    };
    if (useWS) {
        o << "   WARNING: PrivateWS unavailable in at least one capture; BOTH "
             "sides fell\n";
        o << "   back to WorkingSet (less accurate: WS includes shared resident "
             "pages;\n";
        o << "   figures below are indicative only).\n";
        pwsNote("A", A);
        pwsNote("B", B);
        o << "\n";
    }

    // Build union of names.
    std::vector<NamedDelta> procDeltas;
    {
        std::map<std::string, NamedDelta> m;
        for (const auto& kv : A.procByName) {
            NamedDelta& d = m[kv.first];
            d.name = kv.first;
            d.a = M(kv.second);
            d.countA = kv.second.count;
        }
        for (const auto& kv : B.procByName) {
            NamedDelta& d = m[kv.first];
            d.name = kv.first;
            d.b = M(kv.second);
            d.countB = kv.second.count;
        }
        for (auto& kv : m) {
            kv.second.delta = kv.second.b - kv.second.a;
            procDeltas.push_back(kv.second);
        }
    }

    // New processes (present only in B).
    {
        std::vector<NamedDelta> news;
        for (const auto& d : procDeltas)
            if (d.countA == 0 && d.countB > 0) news.push_back(d);
        std::sort(news.begin(), news.end(),
                  [](const NamedDelta& x, const NamedDelta& y) { return x.b > y.b; });
        i64 totBytes = 0, totProcs = 0;
        for (const auto& d : news) { totBytes += d.b; totProcs += d.countB; }
        o << "\n NEW processes (in B, not in A): " << news.size() << " images, "
          << totProcs << " procs, +" << humanBytes(totBytes) << " total\n";
        if (news.empty()) o << "   (none)\n";
        for (const auto& d : news)
            o << "   " << padL("+" + humanBytes(d.b), 14) << "   "
              << padR(d.name, 34) << " (x" << d.countB << ")\n";
    }

    // Gone processes (present only in A).
    {
        std::vector<NamedDelta> gone;
        for (const auto& d : procDeltas)
            if (d.countB == 0 && d.countA > 0) gone.push_back(d);
        std::sort(gone.begin(), gone.end(),
                  [](const NamedDelta& x, const NamedDelta& y) { return x.a > y.a; });
        i64 totBytes = 0, totProcs = 0;
        for (const auto& d : gone) { totBytes += d.a; totProcs += d.countA; }
        o << "\n GONE processes (in A, not in B): " << gone.size() << " images, "
          << totProcs << " procs, -" << humanBytes(totBytes) << " total\n";
        if (gone.empty()) o << "   (none)\n";
        for (const auto& d : gone)
            o << "   " << padL("-" + humanBytes(d.a), 14) << "   "
              << padR(d.name, 34) << " (x" << d.countA << ")\n";
    }

    // Top growers among processes present in both.
    {
        std::vector<NamedDelta> both;
        for (const auto& d : procDeltas)
            if (d.countA > 0 && d.countB > 0) both.push_back(d);
        std::sort(both.begin(), both.end(),
                  [](const NamedDelta& x, const NamedDelta& y) {
                      return x.delta > y.delta;
                  });
        i64 totGrow = 0, totShrink = 0;
        int nGrew = 0, nShrank = 0;
        for (const auto& d : both) {
            if (d.delta > 0) { totGrow += d.delta; ++nGrew; }
            else if (d.delta < 0) { totShrink += d.delta; ++nShrank; }
        }
        o << "\n TOP GROWERS (present in both, by " << metricName << " delta): "
          << nGrew << " grew, +" << humanBytes(totGrow) << " total\n";
        o << "   " << padL("delta", 14) << "   " << padL("A", 12) << padL("B", 12)
          << "   name\n";
        int shown = 0;
        for (const auto& d : both) {
            if (d.delta <= 0) break;
            if (shown++ >= topN) break;
            o << "   " << padL(signedBytes(d.delta), 14) << "   "
              << padL(humanBytes(d.a), 12) << padL(humanBytes(d.b), 12) << "   "
              << d.name << "\n";
        }
        if (shown == 0) o << "   (no processes grew)\n";

        o << "\n TOP SHRINKERS (present in both, by " << metricName << " delta): "
          << nShrank << " shrank, -" << humanBytes(-totShrink) << " total\n";
        shown = 0;
        for (auto it = both.rbegin(); it != both.rend(); ++it) {
            if (it->delta >= 0) break;
            if (shown++ >= topN) break;
            o << "   " << padL(signedBytes(it->delta), 14) << "   "
              << padL(humanBytes(it->a), 12) << padL(humanBytes(it->b), 12) << "   "
              << it->name << "\n";
        }
        if (shown == 0) o << "   (no processes shrank)\n";
    }
    o << "\n";

    // -- Services ----------------------------------------------------------
    o << rule() << "\n";
    o << " Services\n";
    o << rule() << "\n";
    {
        auto sv = [&](const json& j, const char* k) -> i64 {
            return (j.contains(k) && j[k].is_number()) ? j[k].get<i64>() : 0;
        };
        const json& sa = A.services_summary;
        const json& sb = B.services_summary;
        auto svRow = [&](const char* name, i64 a, i64 b) {
            o << "   " << padR(name, 18) << padL(std::to_string(a), 8)
              << padL(std::to_string(b), 8) << padL(std::to_string(b - a), 10)
              << "\n";
        };
        o << "   " << padR("summary", 18) << padL("A", 8) << padL("B", 8)
          << padL("delta", 10) << "\n";
        svRow("total", sv(sa, "total"), sv(sb, "total"));
        svRow("user_mode", sv(sa, "user_mode"), sv(sb, "user_mode"));
        svRow("kernel_driver", sv(sa, "kernel_driver"), sv(sb, "kernel_driver"));
    }

    {
        std::vector<const Svc*> v;
        for (const auto& kv : B.services)
            if (A.services.find(kv.first) == A.services.end())
                v.push_back(&kv.second);
        o << "\n NEW services (in B, not in A): " << v.size() << " total\n";
        if (v.empty()) o << "   (none)\n";
        for (const Svc* s : v)
            o << "   " << padR(s->name, 28) << " [" << padR(s->kind, 6)
              << " " << padR(s->start, 12) << "] " << s->display << "\n";
    }
    {
        std::vector<const Svc*> v;
        for (const auto& kv : A.services)
            if (B.services.find(kv.first) == B.services.end())
                v.push_back(&kv.second);
        o << "\n REMOVED services (in A, not in B): " << v.size() << " total\n";
        if (v.empty()) o << "   (none)\n";
        for (const Svc* s : v)
            o << "   " << padR(s->name, 28) << " [" << padR(s->kind, 6)
              << " " << padR(s->start, 12) << "] " << s->display << "\n";
    }
    o << "\n";

    // -- Drivers -----------------------------------------------------------
    o << rule() << "\n";
    o << " Drivers / kernel modules (metric = mapped image size, i.e. PE\n";
    o << " SizeOfImage = virtual-address footprint; UPPER BOUND on resident RAM,\n";
    o << " not the on-disk file size. For actual resident driver code see the\n";
    o << " 'driver code (resident)' line in the attribution section above.)\n";
    o << rule() << "\n";

    {
        std::vector<Drv> v;
        for (const auto& kv : B.drivers)
            if (A.drivers.find(kv.first) == A.drivers.end())
                v.push_back({kv.first, kv.second});
        std::sort(v.begin(), v.end(),
                  [](const Drv& x, const Drv& y) { return x.size > y.size; });
        i64 tot = 0;
        for (const auto& d : v) tot += d.size;
        o << "\n NEW drivers (loaded in B, absent in A) [+ = mapped image size]: "
          << v.size() << " total, +" << humanBytes(tot) << "\n";
        if (v.empty()) o << "   (none)\n";
        for (const auto& d : v)
            o << "   " << padL("+" + humanBytes(d.size), 14) << "   " << d.name
              << "\n";
    }
    {
        std::vector<Drv> v;
        for (const auto& kv : A.drivers)
            if (B.drivers.find(kv.first) == B.drivers.end())
                v.push_back({kv.first, kv.second});
        std::sort(v.begin(), v.end(),
                  [](const Drv& x, const Drv& y) { return x.size > y.size; });
        i64 tot = 0;
        for (const auto& d : v) tot += d.size;
        o << "\n REMOVED drivers (loaded in A, absent in B) [- = mapped image size]: "
          << v.size() << " total, -" << humanBytes(tot) << "\n";
        if (v.empty()) o << "   (none)\n";
        for (const auto& d : v)
            o << "   " << padL("-" + humanBytes(d.size), 14) << "   " << d.name
              << "\n";
    }
    {
        std::vector<NamedDelta> v;
        for (const auto& kv : B.drivers) {
            auto it = A.drivers.find(kv.first);
            if (it != A.drivers.end() && it->second != kv.second) {
                NamedDelta d;
                d.name = kv.first;
                d.a = it->second;
                d.b = kv.second;
                d.delta = d.b - d.a;
                v.push_back(d);
            }
        }
        std::sort(v.begin(), v.end(),
                  [](const NamedDelta& x, const NamedDelta& y) {
                      return std::llabs(x.delta) > std::llabs(y.delta);
                  });
        i64 net = 0;
        for (const auto& d : v) net += d.delta;
        o << "\n RESIZED drivers (loaded in both, mapped image size changed): "
          << v.size() << " changed, net " << signedBytes(net) << "\n";
        if (v.empty()) o << "   (none)\n";
        for (const auto& d : v)
            o << "   " << padL(signedBytes(d.delta), 14) << "   " << padR(d.name, 30)
              << " (" << humanBytes(d.a) << " -> " << humanBytes(d.b) << ")\n";
    }
    o << "\n";

    // -- Non-paged pool by tag --------------------------------------------
    o << rule() << "\n";
    o << " Non-paged pool by tag (metric = current in-use non-paged bytes per\n";
    o << " pool tag; this is true ExAllocatePoolWithTag(NonPaged) usage and\n";
    o << " directly attributes non-paged pool growth to the responsible tag.)\n";
    o << rule() << "\n";

    if (!A.has_pool_tags || !B.has_pool_tags) {
        o << "\n   (no pool_tags in "
          << (!A.has_pool_tags && !B.has_pool_tags ? "either snapshot"
              : !A.has_pool_tags ? "snapshot A" : "snapshot B")
          << "; re-capture with a wxemem build that emits pool_tags to enable\n"
             "    per-tag non-paged pool attribution.)\n";
    } else {
        struct TagDelta {
            std::string key, tag;
            i64 a = 0, b = 0, delta = 0;
        };
        std::map<std::string, TagDelta> m;
        for (const auto& kv : A.pool_tags) {
            TagDelta& d = m[kv.first];
            d.key = kv.first;
            d.tag = kv.second.tag;
            d.a   = kv.second.nonpaged;
        }
        for (const auto& kv : B.pool_tags) {
            TagDelta& d = m[kv.first];
            d.key = kv.first;
            if (d.tag.empty()) d.tag = kv.second.tag;
            d.b = kv.second.nonpaged;
        }
        std::vector<TagDelta> v;
        i64 tagTotalDelta = 0;
        for (auto& kv : m) {
            kv.second.delta = kv.second.b - kv.second.a;
            tagTotalDelta += kv.second.delta;
            v.push_back(kv.second);
        }

        o << "\n   Sum of per-tag non-paged delta: " << signedBytes(tagTotalDelta)
          << "   (kernel 'non-paged pool' counter delta: "
          << signedBytes(B.kern.non_paged_pool - A.kern.non_paged_pool) << ")\n";

        // New tags (present only in B).
        {
            std::vector<TagDelta> nw;
            for (const auto& d : v)
                if (d.a == 0 && d.b > 0) nw.push_back(d);
            std::sort(nw.begin(), nw.end(),
                      [](const TagDelta& x, const TagDelta& y) { return x.b > y.b; });
            o << "\n NEW tags (allocating in B, none in A):\n";
            if (nw.empty()) o << "   (none)\n";
            int shown = 0;
            for (const auto& d : nw) {
                if (shown++ >= topN) break;
                o << "   " << padL("+" + humanBytes(d.b), 14) << "   " << padR(d.tag, 6)
                  << "  " << d.key << "\n";
            }
        }
        // Gone tags (present only in A).
        {
            std::vector<TagDelta> gone;
            for (const auto& d : v)
                if (d.b == 0 && d.a > 0) gone.push_back(d);
            std::sort(gone.begin(), gone.end(),
                      [](const TagDelta& x, const TagDelta& y) { return x.a > y.a; });
            o << "\n GONE tags (allocating in A, none in B):\n";
            if (gone.empty()) o << "   (none)\n";
            int shown = 0;
            for (const auto& d : gone) {
                if (shown++ >= topN) break;
                o << "   " << padL("-" + humanBytes(d.a), 14) << "   " << padR(d.tag, 6)
                  << "  " << d.key << "\n";
            }
        }
        // Top growers (present in both, by non-paged delta).
        {
            std::vector<TagDelta> both;
            for (const auto& d : v)
                if (d.a > 0 && d.b > 0) both.push_back(d);
            std::sort(both.begin(), both.end(),
                      [](const TagDelta& x, const TagDelta& y) {
                          return x.delta > y.delta;
                      });
            o << "\n TOP GROWERS (in both, by non-paged delta):\n";
            o << "   " << padL("delta", 14) << "   " << padL("A", 12)
              << padL("B", 12) << "   tag\n";
            int shown = 0;
            for (const auto& d : both) {
                if (d.delta <= 0) break;
                if (shown++ >= topN) break;
                o << "   " << padL(signedBytes(d.delta), 14) << "   "
                  << padL(humanBytes(d.a), 12) << padL(humanBytes(d.b), 12) << "   "
                  << padR(d.tag, 6) << "  " << d.key << "\n";
            }
            if (shown == 0) o << "   (no tags grew)\n";

            o << "\n TOP SHRINKERS (in both, by non-paged delta):\n";
            shown = 0;
            for (auto it = both.rbegin(); it != both.rend(); ++it) {
                if (it->delta >= 0) break;
                if (shown++ >= topN) break;
                o << "   " << padL(signedBytes(it->delta), 14) << "   "
                  << padL(humanBytes(it->a), 12) << padL(humanBytes(it->b), 12)
                  << "   " << padR(it->tag, 6) << "  " << it->key << "\n";
            }
            if (shown == 0) o << "   (no tags shrank)\n";
        }
    }
    o << "\n";
    o << rule() << "\n";
    o << " End of report.\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <jsonA> <jsonB> [--top N] [--out FILE]\n"
                 "  <jsonA>   baseline snapshot (before)\n"
                 "  <jsonB>   comparison snapshot (after)\n"
                 "  --top N   rows in top-N tables (default 15)\n"
                 "  --out F   also write the report to file F\n";
}

int main(int argc, char** argv) {
    std::string pathA, pathB, outPath;
    int topN = 15;

    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--top" && i + 1 < argc) {
            topN = std::atoi(argv[++i]);
            if (topN <= 0) topN = 15;
        } else if (a == "--out" && i + 1 < argc) {
            outPath = argv[++i];
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            pos.push_back(a);
        }
    }

    if (pos.size() != 2) {
        usage(argv[0]);
        return 2;
    }
    pathA = pos[0];
    pathB = pos[1];

    Snapshot A, B;
    std::string err;
    if (!loadSnapshot(pathA, A, err)) {
        std::cerr << "error: " << err << "\n";
        return 1;
    }
    if (!loadSnapshot(pathB, B, err)) {
        std::cerr << "error: " << err << "\n";
        return 1;
    }

    std::ostringstream report;
    writeReport(report, A, B, topN);

    std::cout << report.str();

    if (!outPath.empty()) {
        std::ofstream out(outPath, std::ios::binary);
        if (!out) {
            std::cerr << "warning: cannot write --out file: " << outPath << "\n";
        } else {
            out << report.str();
        }
    }

    return 0;
}
