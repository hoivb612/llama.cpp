// membw.cpp -- multi-threaded DRAM READ-bandwidth probe.
//
// Weight streaming during decode is a pure sequential READ, so this measures the
// relevant roofline: achievable read GB/s vs. thread count. Compare the peak here
// against the engine's ~56 GB/s decode stream to see how much headroom exists.
//
// Build (MSVC, from a vcvars64 shell):
//   cl /nologo /O2 /std:c++17 /arch:AVX512 membw.cpp
// Build (g++/clang):
//   g++ -O3 -std=c++17 -march=native -pthread membw.cpp -o membw
//
// Run:  membw [bufGB=4] [iters=20] [threadlist=1,2,4,8,12,16]
//   e.g.  membw            (defaults)
//         membw 4 30 1,2,4,8,16
//
// Notes: buffer is faulted in up front; each thread streams its own contiguous
// slice (mirrors how the engine's workers each read distinct weight tiles). Use a
// buffer several x larger than total L3 (Strix Halo: pass >=4) so the read hits DRAM.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <string>

static uint64_t sum_range(const uint64_t* p, size_t n) {
    // 8 independent accumulators -> saturate load ports; /O2 vectorizes to wide loads.
    uint64_t a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 += p[i+0]; a1 += p[i+1]; a2 += p[i+2]; a3 += p[i+3];
        a4 += p[i+4]; a5 += p[i+5]; a6 += p[i+6]; a7 += p[i+7];
    }
    for (; i < n; ++i) a0 += p[i];
    return a0+a1+a2+a3+a4+a5+a6+a7;
}

int main(int argc, char** argv) {
    double bufGB = argc > 1 ? atof(argv[1]) : 4.0;
    int    iters = argc > 2 ? atoi(argv[2]) : 20;
    std::vector<int> tl;
    if (argc > 3) {
        std::string s = argv[3], cur;
        for (char c : s) { if (c == ',') { if(!cur.empty()){tl.push_back(atoi(cur.c_str()));cur.clear();} } else cur += c; }
        if (!cur.empty()) tl.push_back(atoi(cur.c_str()));
    } else {
        tl = {1, 2, 4, 8, 12, 16};
    }

    size_t bytes = (size_t)(bufGB * 1e9) & ~(size_t)63;
    size_t n = bytes / sizeof(uint64_t);
    uint64_t* buf = (uint64_t*)malloc(bytes);
    if (!buf) { printf("alloc %.2f GB failed\n", bytes / 1e9); return 1; }
    memset(buf, 1, bytes);  // fault in all pages (resident before timing)

    volatile uint64_t sink = 0;
    printf("buffer=%.2f GB  iters=%d  (sequential read)\n", bytes / 1e9, iters);
    for (int T : tl) {
        if (T < 1) continue;
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            std::vector<std::thread> th; th.reserve(T);
            std::vector<uint64_t> part(T, 0);
            size_t chunk = n / (size_t)T;
            for (int t = 0; t < T; ++t) {
                size_t b = (size_t)t * chunk;
                size_t cnt = (t == T - 1) ? n - b : chunk;
                th.emplace_back([&, t, b, cnt] { part[t] = sum_range(buf + b, cnt); });
            }
            for (auto& x : th) x.join();
            uint64_t s = 0; for (auto v : part) s += v; sink += s;
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double gb = (double)bytes * iters / 1e9;
        printf("  threads=%2d  %7.1f GB/s   (%.3fs, %.0f GB read)\n", T, gb / sec, sec, gb);
    }
    (void)sink;
    free(buf);
    return 0;
}
