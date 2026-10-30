// fusion_cpu_ep.cpp
//
// M1/M2a/M2b/M3 implementation of the FusionCPU OrtEp:
//   * GetCapability claims:
//       - Mul (default domain, fp32, two identical-shape inputs) [M1],
//       - Sigmoid (default domain, fp16, unary) [M2a] -- the Phi-3 SiLU gate,
//       - the SwiGLU activation triple {Sigmoid, Mul, Mul_1} as one fused
//         3-node group [M2b] -- y = u * g * sigmoid(g), and
//       - MatMulNBits (com.microsoft, int4 block-wise, fp16 A) [M3] -- gated
//         behind ORT_FUSIONCPU_MATMULNBITS=1 (naive scalar reference kernel),
//         and the {Mul, MatMulNBits} AWQ-scale pair as one fused 2-node group
//         [M3+] -- the *_weight_only_mul per-channel activation scale folds
//         into the int4 kernel, eliminating the Mul.
//   * Compile builds one CPU kernel per fused node, keyed by fused-node name.
//   * Compute reads inputs as plain CPU pointers and writes CPU outputs.
//   * Sync is a no-op.
//
// M2a/M2b execute *real* Phi-3 ops (32 fp16 Sigmoid / 32 SwiGLU triples) on the
// plugin EP with top-1 parity vs the built-in CPU EP, proving fp16 kernel
// execution + multi-node graph partitioning + fallback end-to-end (not just
// registration plumbing).

#include "fusion_cpu_ep.h"

#include "version.h"
#include "fp16_utils.h"
#include "matmulnbits_avx512.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#endif // _WIN32

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "fusion_cpu_extra_ops.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fusion_cpu_ep {
namespace {

// --- AVX-512F runtime dispatch (M3++) -------------------------------------
//
// Defined in this BASELINE translation unit (NOT the /arch:AVX512 one) so it is
// always safe to call: it uses only cpuid/xgetbv intrinsics, never an AVX-512
// instruction. The /arch:AVX512 int4 kernel (MatMulNBitsRowDotAvx512) is only
// ever called after this returns true.
bool DetectAvx512f() {
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;            // max leaf < 7
    __cpuidex(info, 1, 0);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx     = (info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;
    const unsigned long long xcr0 = _xgetbv(0);
    // XMM(0x2) | YMM(0x4) | opmask(0x20) | ZMM_Hi256(0x40) | Hi16_ZMM(0x80)
    if ((xcr0 & 0xE6ull) != 0xE6ull) return false;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 16)) != 0;        // EBX bit 16 = AVX512F
#else
    return false;
#endif
}

bool HasAvx512f() {
    static const bool kHas = DetectAvx512f();
    return kHas;
}

// Detects AVX-512-VNNI (CPUID leaf 7, subleaf 0, ECX bit 11). VNNI also requires
// the AVX-512F OS-enablement checks, so this is gated on DetectAvx512f() first.
// Used by the opt-in int8 dot path (Path B); only ever called from the baseline
// TU (cpuid only, never an AVX-512 instruction).
bool DetectAvx512Vnni() {
#if defined(_MSC_VER)
    if (!DetectAvx512f()) return false;
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[2] & (1 << 11)) != 0;            // ECX bit 11 = AVX512_VNNI
#else
    return false;
#endif
}

bool HasAvx512Vnni() {
    static const bool kHas = DetectAvx512Vnni();
    return kHas;
}

// Env override: force the scalar int4 path (for scalar-vs-AVX512 parity tests
// and A/B benchmarking) even when AVX-512F is available.
bool MatMulNBitsForceScalar() {
    const char* v = std::getenv("ORT_FUSIONCPU_MATMULNBITS_SCALAR");
    return v != nullptr && v[0] == '1';
}

// Diagnostic gate: when ORT_FUSIONCPU_MATMULNBITS_VERBOSE=1, the kernel prints a
// one-shot line to stderr reporting which int4 dispatch path is live. Lets us
// confirm on a target machine whether the AVX-512F kernel is actually engaged
// (vs. silently falling back to scalar because the CPU lacks AVX-512F or a stale
// DLL is loaded).
bool MatMulNBitsVerbose() {
    const char* v = std::getenv("ORT_FUSIONCPU_MATMULNBITS_VERBOSE");
    return v != nullptr && v[0] == '1';
}

// Opt-in gate for the AVX-512-VNNI int8 dot path (Path B). When set AND the CPU
// supports AVX-512-VNNI AND block_size % 64 == 0, the kernel quantizes the
// (AWQ-folded) activations to int8 per block and accumulates the int4 x int8
// products with _mm512_dpbusd_epi32 instead of dequantizing to fp32. This trades
// a small, bounded accuracy loss (per-block int8 activation quant) for ~4x denser
// arithmetic and a 4x smaller weight working set. OFF by default so the bit-exact
// fp32 AVX-512 path stays the default and all existing parity tests are unchanged.
bool MatMulNBitsUseVnni() {
    const char* v = std::getenv("ORT_FUSIONCPU_MATMULNBITS_VNNI");
    return v == nullptr || v[0] != '0'; // Default ON
}

// Opt-out gate for caching the register-blocked int8 weight prepack. Weights
// are constant initializers, but without this the VNNI path re-unpacks int4->int8
// and recomputes the per-block weight sum on every Compute -- i.e. ~224 times per
// decode token. With it set (AND VNNI enabled), each weight tensor is prepacked
// once on first Compute into the tile-interleaved int8 layout and reused across
// all subsequent Runs, so the steady-state decode loop pays no repack cost and can
// use the reduction-free tile kernel even at M=1. Cost: the int8 prepack roughly
// doubles resident weight memory.
bool MatMulNBitsPrepack() {
    const char* v = std::getenv("ORT_FUSIONCPU_MATMULNBITS_PREPACK");
    return v == nullptr || v[0] != '0'; // Default ON
}

// Gate for the COMPACT int4 prepacked tile (decode bandwidth fix). The original
// M5 prepack stores weights as int8 (one byte each), which doubles the resident
// weight bytes vs the model's native int4 -- and decode is cold-cache BANDWIDTH
// bound (every token re-reads the full weight set from RAM), so reading 2x bytes
// makes us ~2x slower than MLAS even though our per-op MATH is at parity. This
// path keeps the prepacked tile nibble-packed (half the bytes, == MLAS's int4
// footprint) and expands int4->int8 in-register inside the dpbusd microkernel.
// ON by default; set ORT_FUSIONCPU_MATMULNBITS_INT4TILE=0 to A/B against the int8
// tile. Only used when the weights have no zero_points (default_zp=8 => (q-zp) in
// [-8,7] fits signed int4); zero-point models keep the int8 tile.
bool MatMulNBitsInt4Tile() {
    const char* v = std::getenv("ORT_FUSIONCPU_MATMULNBITS_INT4TILE");
    return v == nullptr || v[0] != '0';
}

// Gate for COMPACT side data on the int4 tile path (decode bandwidth fix, stage 2).
// The int4 tile already halves the WEIGHT bytes, but the two per-(block,col) side
// arrays -- the fp32 scale and the int32 weight-sum -- are each as large as ~25% of
// the nibble weights and are re-read every decode token. This packs them to half
// width: fp16 scale (matches MLAS's native fp16 scale; lossless for fp16-source
// models, ~5e-4 relative for fp32-source) + int16 weight-sum (LOSSLESS for the
// no-zp range, block_size <= 4096). ON by default; set
// ORT_FUSIONCPU_MATMULNBITS_INT4COMPACT=0 to A/B against the fp32-scale int4 tile.
bool MatMulNBitsInt4Compact() {
    const char* v = std::getenv("ORT_FUSIONCPU_MATMULNBITS_INT4COMPACT");
    return v == nullptr || v[0] != '0';
}

// Gate for the COMPACT int4 tile WITH per-block zero_points (decode bandwidth fix
// for zp-carrying models, e.g. Gemma int4). Models whose MatMulNBits carry
// zero_points otherwise fall onto the int8 tile (2x the model's native int4 weight
// bytes); this keeps the weights nibble-packed and applies the zp correction in the
// dot via a per-(row,block) activation sum -- ~1.8x fewer resident weight bytes on
// the bandwidth-bound decode path, BIT-IDENTICAL integer math to the int8 path.
// ON by default; set ORT_FUSIONCPU_MATMULNBITS_INT4COMPACTZP=0 to A/B against int8.
bool MatMulNBitsInt4CompactZp() {
    const char* v = std::getenv("ORT_FUSIONCPU_MATMULNBITS_INT4COMPACTZP");
    return v == nullptr || v[0] != '0';
}

// Opt-out gate for the M-blocked compact tile kernel (default ON). When the
// cached compact int4 prepack is active, the tile dot is normally evaluated
// kCompactMBMax activation rows at a time so each weight strip is read +
// nibble-expanded once per block of rows instead of once per row -- the prefill
// steady-state GEMM win. Set ORT_FUSIONCPU_MATMULNBITS_MBLOCK=0 to force the
// per-row single-row kernel (A/B benchmarking / fallback). Decode (M=1) is
// unaffected either way (a 1-row block IS the single-row kernel).
bool MatMulNBitsMBlock() {
    const char* v = std::getenv("ORT_FUSIONCPU_MATMULNBITS_MBLOCK");
    return v == nullptr || v[0] != '0';
}

// Opt-in gate to dispatch the N-loop onto ORT's intra-op thread pool (via
// KernelContext::ParallelFor) instead of our private FusionThreadPool. The
// private pool runs a second set of worker threads ALONGSIDE ORT's intra-op
// pool, which oversubscribes the core budget in a whole-model decode (every
// CPU-EP op -- GQA, layernorm, rotary -- then contends with our parked workers,
// roughly doubling their cost) and pays a per-node condition-variable wake
// latency across the ~225 fused nodes. Routing through ORT's pool shares the one
// spin-tuned thread set with MLAS and all CPU ops, eliminating the
// oversubscription. Threading is governed by ORT_FUSIONCPU_MATMULNBITS_THREADS.
bool MatMulNBitsUseOrtPool() {
    const char* v = std::getenv("ORT_FUSIONCPU_MATMULNBITS_ORTPOOL");
    return v != nullptr && v[0] == '1';
}

// Maximum worker-thread count (cap) for the int4 N-loop. Default = hardware
// concurrency, capped at 64. ORT_FUSIONCPU_MATMULNBITS_THREADS overrides:
//   * "1"  -> serial (no worker threads) -- A/B benchmarking and parity.
//   * "N"  -> cap at N worker threads (clamped to [1, 256]).
//   * unset/invalid -> auto (hardware_concurrency, min 1, capped 64).
// This is only a CAP: the actual thread count is scaled down by the work size
// (see kMacsPerThread in Compute) so small ops -- notably M=1 decode -- run
// serially and never pay thread-spawn overhead.
// Cached on first call (env is read once per process, like the other gates).
int MatMulNBitsThreadCount() {
    static const int n = [] {
        const char* v = std::getenv("ORT_FUSIONCPU_MATMULNBITS_THREADS");
        if (v != nullptr && v[0] != '\0') {
            const long parsed = std::strtol(v, nullptr, 10);
            if (parsed >= 1) return static_cast<int>(std::min<long>(parsed, 256));
        }
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 1;
        return static_cast<int>(std::min<unsigned>(hc, 64u));
    }();
    return n;
}

#if defined(_WIN32)

bool set_thread_priority (
    uint32_t id
    )
{

    //
    // Set the priority of the current thread to THREAD_PRIORITY_TIME_CRITICAL.
    //

    if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)) {
        return true;

    } else {
        printf("failed to set thread %d priority\n", id);
        return false;
    }
}

//
// Establishes the process affinity of the current process dependent on the number
// of ccx's and threads requested. The process affinity is computed such that the
// threads are as sparsely spread as possible across the ccx's in the system. This
// has a marked effect on performance.
//

void PinWorkers(uint32_t nthreads) {

    typedef struct {
        WORD group;
        ULONG base_lp;
        uint32_t cores;
        KAFFINITY mask;
    } ccx_t;

    char *buf = NULL;
    ccx_t ccx[64];

    struct {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
    } cpu_info;

    uint32_t i;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info;
    DWORD len = 0;
    uint32_t n_ccx = 0;
    char *p = NULL;

    //
    // Check if the process group affinity has already been set.
    //

    USHORT group_count = 0;

    GetProcessDefaultCpuSetMasks(GetCurrentProcess(), NULL, 0, &group_count);
    if (group_count != 0) {
        return;
    }

    //
    // Check for AMD processor.
    //

    // ****** add cpuid code

    //
    // Check if SMT is enabled.
    //

    __cpuid((int *)&cpu_info, 0x8000001e);
    if ((((cpu_info.ebx & 0xf00) >> 8) + 1) != 2) {
        printf("SMT is not enabled\n");
        goto skip;
    }

    //
    // Query the processor relationship information.
    //

    GetLogicalProcessorInformationEx(RelationCache, NULL, &len);
    if (!len) {
        printf("failed to get processor information length\n");
        goto skip;
    }

    buf = (char *)malloc(len);
    if (!buf) {
        printf("failed tp allocate processor information buffer\n");
        goto skip;
    }

    info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf;
    if (!GetLogicalProcessorInformationEx(RelationCache, info, &len)) {
        printf("failed to get processor information\n");
        free(buf);
        goto skip;
    }

    //
    // Parse the processor relationship information.
    //

    p = buf;
    while (p < (buf + len)) {
        info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)p;

        if ((info->Relationship == RelationCache) &&
            (info->Cache.Level == 3) &&
            (n_ccx < ARRAYSIZE(ccx))) {

            ccx[n_ccx].group = info->Cache.GroupMask.Group;
            ccx[n_ccx].mask = info->Cache.GroupMask.Mask;
            ccx[n_ccx].base_lp = 0;
            ccx[n_ccx].cores = (uint32_t)PopulationCount64(ccx[n_ccx].mask);

            BitScanForward64(&ccx[n_ccx].base_lp, info->Cache.GroupMask.Mask);
            n_ccx += 1;
        }

        p += info->Size;
    }

    free(buf);

    //
    // Check if any ccx's were discovered.
    //

    if (n_ccx) {
        GROUP_AFFINITY ga[ARRAYSIZE(ccx) / 16] = {0};

        if (MatMulNBitsVerbose()) {
            printf("Populated CCX table\n");

            for (uint32_t i = 0; i < n_ccx; i += 1) {
                printf("  CCX %2d: group=%u base_lp=%2d cores=%2d mask=0x%016llx\n",
                       i,
                       ccx[i].group,
                       ccx[i].base_lp,
                       ccx[i].cores,
                       ccx[i].mask);
            }
        }

        //
        // Compute the number of SMT pairs.
        //

        uint32_t smt_pairs = 0;

        for (i = 0; i < n_ccx; i += 1) {
            ccx[i].cores /= 2;
            smt_pairs += ccx[i].cores;
        }

        if (smt_pairs < nthreads) {
            printf("insufficient SMT pairs %u to allocate by ccx\n", smt_pairs);
            goto skip;
        }

        //
        // Depopulate the SMT paired logical processor ccx masks.
        //
        // N.B. It is possible that a ccx does not have the full complement of cores.
        //

        for (i = 0; i < n_ccx; i += 1) {
            for (uint32_t j = 0; j < 8; j += 1) {
                ccx[i].mask &= ~(1ull << ((j * 2) + ccx[i].base_lp + 1));
            }
        }

        if (MatMulNBitsVerbose()) {
            printf("Depopulated CCX table\n");

            for (i = 0; i < n_ccx; i += 1) {
                printf("  CCX %2d: group=%u base_lp=%2d cores=%2d mask=0x%016llx\n",
                       i,
                       ccx[i].group,
                       ccx[i].base_lp,
                       ccx[i].cores,
                       ccx[i].mask);
            }
        }

        //
        // Construct a set of group affinity structures which includes the specified
        // number of threads. These group affinity structures are used to set the
        // process affinity.
        //

        std::srand(static_cast<unsigned>(std::time(nullptr)));

        //
        // Start from a random ccx.
        //

        uint32_t k = std::rand() % n_ccx;

        for (i = 0; i < nthreads; i += 1) {

            //
            // N.B. The ccx mask may not have any bits set if it does not have the
            //      full complement of cores.
            //

            uint32_t count = (uint32_t)PopulationCount64(ccx[k].mask);

            if (count) {
                WORD group = ccx[k].group;
                uint32_t select = ccx[k].base_lp;
                uint32_t position = std::rand() % count;
    
                //
                // Assign the affinity randomly within the remaining bits.
                //

#if 0
                printf("count %u, select %u, position %u, mask 0x%016llx\n",
                        count,
                        select,
                        position,
                        ccx[k].mask);
#endif // #if 0

    
                do {
                    if (ccx[k].mask & (1ull << select)) {
                        if (!position) {
                            break;
                        }
    
                        position -= 1;
                    }
    
                    select += 1;
                } while (true);
    
                ccx[k].mask ^= 1ull << select;
    
                ga[group].Group = group;
                ga[group].Mask |= 1ull << select;
            }

            //
            // Round robin the ccx's.
            //

            k += 1;
            if (k >= n_ccx) {
                k = 0;
            }
        }

        //
        // Compact the group affinity array.
        //

        uint32_t n_groups = 0;

        for (i = 0; i < ARRAYSIZE(ga); i += 1) {
            if (ga[i].Mask) {
                ga[n_groups] = ga[i];
                n_groups += 1;
            }
        }

        if (MatMulNBitsVerbose()) {
            printf("Allocated CCX table\n");

            for (i = 0; i < n_ccx; i += 1) {
                printf("  CCX %2d: group=%u base_lp=%2d cores=%2d mask=0x%016llx\n",
                       i,
                       ccx[i].group,
                       ccx[i].base_lp,
                       ccx[i].cores,
                       ccx[i].mask);
            }

            printf("Process group affinity\n");

            for (i = 0; i < n_groups; i += 1) {
                printf("  group= %2u mask=0x%016llx\n", ga[i].Group, ga[i].Mask);
            }
        }

        //
        // Set process affinity.
        //

        if (!SetProcessDefaultCpuSetMasks(GetCurrentProcess(), ga, (WORD)n_groups)) {
            printf("Set process affinity group set failed\n");

        } else {
            printf("set process affinity group set succeed\n");
        }

        return;
    }

skip:

    //
    // The number of threads must be greater than zero and less than or equal to
    // one affinity group (64).
    //

    if (!nthreads || (nthreads > 64)) {
        printf("number of threads must be non-zero and less than or equal to 64\n");
        return;
    }

    //
    // The hardware concurrency must not be odd and greater than 1 - assume 2-way SMT.
    //

    const uint32_t logical = std::max(1u, std::thread::hardware_concurrency());

    if ((logical <= 1) || (logical & 1)) {
        printf("hardware concurency is odd or zero\n");
        return;
    }

    //
    // The number of physical cores must be greater than or equal to the number of
    // threads.
    //

    uint32_t phys = logical / 2u;

    if (nthreads > phys) {
        printf("number of threads is greater than the number of physical cores\n");
        return;
    }

    //
    // Attempt to spread the workers across multiple ccx's.
    //

    if (!(nthreads % 8)) {
        phys -= nthreads / 2;
        printf("phys = %d\n", phys);
    }

    //
    // Evenly spread workers across the middle/high physical cores.
    //

    uint32_t core;
    GROUP_AFFINITY ga = {0};
    uint32_t lp;

    for (uint32_t i = 0; i < nthreads; i += 1) {
        core = phys - nthreads + i;
        lp = (core * 2u) % logical;

        printf("core %d, lp %d\n", core, lp);

        ga.Group = (WORD)(lp / 64u);
        ga.Mask |= static_cast<KAFFINITY>(1ull << (lp % 64u));
    }

    //
    // Set the process group affinity.
    //

    if (SetProcessAffinityMask(GetCurrentProcess(), ga.Mask)) {
        printf("process set group affinity group %u, mask 0x%08I64x\n", ga.Group, ga.Mask);

    } else {
        printf("process set group affinity failed\n");
    }
}

#else

void PinWorkers(uint32_t) {}

#endif // _WIN32

// --- persistent worker pool (M4) ------------------------------------------
//
// The int4 kernel is invoked once per MatMulNBits node -- ~224 nodes per token
// during decode. Spawning std::threads per Compute is far too costly at that
// rate (it also regressed decode and crashed at teardown), so the work-scaling
// gate previously forced every M=1 decode op to run serially. A persistent pool
// fixes that: workers are created once and park on a condition variable when
// idle (zero CPU while you game), so each op -- including decode -- can fan out
// for only the cost of a wakeup + a barrier.
//
// Design choices that keep it correct and crash-free inside a dynamically
// (un)loaded plugin DLL:
//   * Fixed-index task assignment (worker i always runs task i, the submitting
//     thread runs task 0) -- no atomic work-stealing, so a worker can never pick
//     up a task from the next generation. This avoids cross-generation races on
//     the single shared body/nways slot.
//   * A run-mutex serializes submitters, so concurrent Compute calls (parallel
//     executor) share the pool safely instead of corrupting its single slot.
//   * The singleton is intentionally leaked (never joined): joining worker
//     threads during static destruction / FreeLibrary can deadlock under the
//     Windows loader lock. Idle workers cost nothing; the OS reclaims them at
//     process exit.
class FusionThreadPool {
 public:
    explicit FusionThreadPool(int num_ways) : nthreads_(std::max(1, num_ways)) {

        //
        // If the number of threads is greater than one, then initialize and create
        // worker threads.
        //

        if (nthreads_ != 1) {

            //
            // Pin worker threads including the master thread.
            //
    
            PinWorkers(nthreads_);

            //
            // Set master thread priority.
            //

            set_thread_priority(0);

            //
            // Start worker threads. 
            //
    
            for (int id = 1; id < nthreads_; ++id) {
                workers_.emplace_back([this, id] { WorkerLoop(id); });
            }
        }
    }

    int size() const { return nthreads_; }

    //
    // Runs body(task, nways) for task in (0, nways) across the calling thread
    // (task 0) plus the worker threads, blocking until all tasks complete.
    // nways is clamped to the thread pool size. body must be safe to call
    // concurrently.
    //

    void ParallelFor(int nways, const std::function<void(int, int)>& body) {

        if (nways <= 1 || nthreads_ == 1) {
            body(0, 1);
            return;
        }

        if (nways > nthreads_) {
            nways = nthreads_;
        }

        //
        // Set the execution function and the number of ways the body is to be
        // executed in parallel and the predicate value gen_ under lock.
        //

        AcquireSRWLockExclusive(&mu_srw);
        body_ = &body;
        nways_ = nways;
        remaining_.store(nways, std::memory_order_release);
        gen_ += 1;
        ReleaseSRWLockExclusive(&mu_srw);

        //
        // Notify all thread pool threads there is work to be performed.
        //

        WakeAllConditionVariable(&cv_mu);

        //
        // Execute the function body using the master thread.
        //

        body(0, nways);

        //
        // Interlocked exchange decrement the remaining count. If the previous value
        // was not one, then this is not the last thread to finish the body execution
        // and the master thread needs to wait for all workers to threads complete.
        //

        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
            do {
            } while (remaining_.load(std::memory_order_acquire) != 0);
        }
    }

 private:
    void WorkerLoop(int id) {

        uint64_t my_gen = 0;

        //
        // Set worker thread priority.
        //

        set_thread_priority(id);

        for (;;) {

            const std::function<void(int, int)>* body;
            int nways;

            //
            // Wait for work to be signalled.
            //

            AcquireSRWLockExclusive(&mu_srw);
            do {
                if (stop_ || (gen_ != my_gen)) {
                    body = body_;
                    nways = nways_;
                    my_gen = gen_;
                    break;
                }

                SleepConditionVariableSRW(&cv_mu, &mu_srw, INFINITE, 0);

            } while (true);

            ReleaseSRWLockExclusive(&mu_srw);
            if (stop_) {
                return;
            }

            //
            // If this thread is within the execution group, then execute the function
            // body.
            //

            if (id < nways) {
                (*body)(id, nways);

                //
                // Interlocked exchange decrement the remaining count and loop back
                // to wait for more work.
                //

                remaining_.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }

    const int nthreads_;
    std::vector<std::thread> workers_;
    SRWLOCK mu_srw = {0};
    CONDITION_VARIABLE cv_mu = {0};

    const std::function<void(int, int)>* body_ = nullptr;
    int nways_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
    std::atomic<int> remaining_{0};
};

// Leaked singleton, sized to the thread cap and created lazily on first use
// (i.e. the first MatMulNBits Compute), so the pool only exists when the int4
// kernel actually runs.
FusionThreadPool& MatMulNBitsThreadPool() {
    static FusionThreadPool* pool = new FusionThreadPool(MatMulNBitsThreadCount());
    return *pool;
}

// --- fp16 <-> fp32 conversion ---------------------------------------------
// (HalfToFloat / FloatToHalf now live in fp16_utils.h, shared with the
//  /arch:AVX512 translation unit so both round identically.)

// --- small type/shape helpers (mirrors the Vulkan EP) ---------------------

ONNXTensorElementDataType TensorElementTypeOf(Ort::ConstValueInfo vi) {
    Ort::ConstTypeInfo t = vi.TypeInfo();
    if (t.GetONNXType() != ONNX_TYPE_TENSOR) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    return t.GetTensorTypeAndShapeInfo().GetElementType();
}

// Returns the value-info's shape with -1 placeholders for symbolic dims.
// std::nullopt only if there is no shape info at all (rank unknown).
std::optional<std::vector<int64_t>> GetShapeAllowSymbolic(Ort::ConstValueInfo vi) {
    Ort::ConstTypeInfo t = vi.TypeInfo();
    if (t.GetONNXType() != ONNX_TYPE_TENSOR) return std::nullopt;
    return t.GetTensorTypeAndShapeInfo().GetShape();
}

// Reads an INT graph attribute (int64) by name. Returns false if absent.
bool ReadInt64Attr(const Ort::ConstNode& node, const char* name, int64_t& out) {
    Ort::ConstOpAttr attr;
    Ort::Status st = node.GetAttributeByName(name, attr);
    if (!st.IsOK()) return false;
    return attr.GetValue<int64_t>(out).IsOK();
}

// Reads a FLOAT graph attribute by name. Returns false if absent.
bool ReadFloatAttr(const Ort::ConstNode& node, const char* name, float& out) {
    Ort::ConstOpAttr attr;
    Ort::Status st = node.GetAttributeByName(name, attr);
    if (!st.IsOK()) return false;
    return attr.GetValue<float>(out).IsOK();
}

// Selects the GQA decode kernel structure: default fused cache-blocked pass;
// ORT_FUSIONCPU_GQA_FUSED=0 forces the legacy two-phase path (A/B diagnostic).
bool GqaUseFusedPath() {
    const char* v = std::getenv("ORT_FUSIONCPU_GQA_FUSED");
    const bool fused = (v == nullptr) || (v[0] != '0');  // default ON
    static const bool logged = [&]() {
        if (!fused) {
            std::fprintf(stderr,
                "[FusionCpuEP] GQA decode path = two-phase (legacy) (ORT_FUSIONCPU_GQA_FUSED=%s)\n",
                v ? v : "unset");
        }
        return true;
    }();
    (void)logged;
    return fused;
}

// fp16 KV cache is the DEFAULT decode path (opt-out ORT_FUSIONCPU_GQA_FP16KV=0).
// Each GQA kernel keeps a private, persistent fp16 K/V cache and skips the
// per-step fp32 past->present copy-forward, cutting GQA's ~1.3 GB/step DRAM
// traffic to ~fp16 attend-reads only. Requires AVX-512 (F16C widen in the hot
// dot); the caller additionally falls back to the fp32 path when !HasAvx512f().
bool GqaFp16KvEnabled() {
    const char* v = std::getenv("ORT_FUSIONCPU_GQA_FP16KV");
    const bool on = (v == nullptr) || (v[0] != '0');  // default ON
    static const bool logged = [&]() {
        if (!on) {
            std::fprintf(stderr,
                "[FusionCpuEP] GQA KV cache = fp32 (legacy) (ORT_FUSIONCPU_GQA_FP16KV=%s)\n",
                v ? v : "unset");
        }
        return true;
    }();
    (void)logged;
    return on;
}

// B1 own-KV footprint mode (default OFF; opt-in ORT_FUSIONCPU_GQA_OWNKV=1).
// The fp16 private cache becomes the SOLE KV copy: the kernel derives the KV
// length/position from the node's seqlens_k input (base = seqlens_k[0]+1 - S)
// instead of the incoming past tensor, so the harness can feed an empty past
// and discard present (no fp32 grow-and-swap). seqlens_k[0] also drives a
// unified clear/rewind control: base==0 clears, 0<base<cache_len_ rewinds,
// base==cache_len_ appends. Requires the fp16 path (AVX-512).
bool GqaOwnKvEnabled() {
    const char* v = std::getenv("ORT_FUSIONCPU_GQA_OWNKV");
    const bool on = (v != nullptr) && (v[0] != '0');  // default OFF
    static const bool logged = [&]() {
        if (on) {
            std::fprintf(stderr, "[FusionCpuEP] GQA own-KV (footprint) = %s (ORT_FUSIONCPU_GQA_OWNKV=%s)\n",
                        "ENABLED - EP owns fp16 KV, empty past/discard present",
                        v ? v : "set");
        }
        return true;
    }();
    (void)logged;
    return on;
}

}  // namespace

// Shared-pool wrapper exposed to fusion_cpu_extra_ops.cpp (declared in
// fusion_cpu_extra_ops.h). Routes the extra ops' parallel regions through the
// SAME FusionThreadPool that MatMulNBits/GQA use, so no second pool is created.
// Chunks [0, count) into per-worker [begin, end) sub-ranges, mirroring the GQA
// kernel's run_parallel helper.
void FusionParallelFor(int count, const std::function<void(int, int)>& fn) {
    MatMulNBitsThreadPool().ParallelFor(count, [&](int task, int nways) {
        const int begin = static_cast<int>(static_cast<int64_t>(task) * count / nways);
        const int end   = static_cast<int>(static_cast<int64_t>(task + 1) * count / nways);
        fn(begin, end);
    });
}

// ===== Mul kernel ==========================================================

struct FusionCpuMulKernel {
    explicit FusionCpuMulKernel(const OrtApi* ort_api) : ort_api_(ort_api) {}

    // Elementwise fp32 multiply. Requires identical input shapes (M1 scope).
    OrtStatus* Compute(OrtKernelContext* kctx) {
        Ort::KernelContext ctx{kctx};
        if (ctx.GetInputCount() != 2 || ctx.GetOutputCount() != 1) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuMulKernel expects 2 inputs + 1 output");
        }

        Ort::ConstValue in0 = ctx.GetInput(0);
        Ort::ConstValue in1 = ctx.GetInput(1);
        auto t0 = in0.GetTensorTypeAndShapeInfo();
        auto t1 = in1.GetTensorTypeAndShapeInfo();
        if (t0.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            t1.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuMulKernel requires float32 inputs");
        }
        std::vector<int64_t> shape0 = t0.GetShape();
        std::vector<int64_t> shape1 = t1.GetShape();
        if (shape0 != shape1) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuMulKernel requires identical input shapes (M1: no broadcasting)");
        }
        int64_t n = 1;
        for (int64_t d : shape0) n *= d;
        if (n < 0) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuMulKernel: negative element count");
        }

        const float* a = in0.GetTensorData<float>();
        const float* b = in1.GetTensorData<float>();
        Ort::UnownedValue out = ctx.GetOutput(0, shape0);
        float* y = out.GetTensorMutableData<float>();
        for (int64_t i = 0; i < n; ++i) {
            y[i] = a[i] * b[i];
        }
        return nullptr;
    }

private:
    const OrtApi* ort_api_ = nullptr;
};

// ===== Sigmoid kernel (fp16) ===============================================

struct FusionCpuSigmoidKernel {
    explicit FusionCpuSigmoidKernel(const OrtApi* ort_api) : ort_api_(ort_api) {}

    // Elementwise logistic sigmoid: y = 1 / (1 + exp(-x)), computed in fp32.
    // Supports float16 (rounded to nearest-even half) and float32. Unary,
    // shape-preserving.
    OrtStatus* Compute(OrtKernelContext* kctx) {
        Ort::KernelContext ctx{kctx};
        if (ctx.GetInputCount() != 1 || ctx.GetOutputCount() != 1) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuSigmoidKernel expects 1 input + 1 output");
        }
        Ort::ConstValue in0 = ctx.GetInput(0);
        auto t0 = in0.GetTensorTypeAndShapeInfo();
        const auto dt = t0.GetElementType();
        if (dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
            dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuSigmoidKernel requires float16 or float32 input");
        }
        std::vector<int64_t> shape0 = t0.GetShape();
        int64_t n = 1;
        for (int64_t d : shape0) n *= d;
        if (n < 0) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuSigmoidKernel: negative element count");
        }

        Ort::UnownedValue out = ctx.GetOutput(0, shape0);
        if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const uint16_t* x = in0.GetTensorData<uint16_t>();
            uint16_t* y = out.GetTensorMutableData<uint16_t>();
            for (int64_t i = 0; i < n; ++i) {
                const float xf = HalfToFloat(x[i]);
                y[i] = FloatToHalf(1.0f / (1.0f + std::exp(-xf)));
            }
        } else {
            const float* x = in0.GetTensorData<float>();
            float* y = out.GetTensorMutableData<float>();
            for (int64_t i = 0; i < n; ++i)
                y[i] = 1.0f / (1.0f + std::exp(-x[i]));
        }
        return nullptr;
    }

private:
    const OrtApi* ort_api_ = nullptr;
};

// ===== SwiGLU activation kernel (fp16) =====================================
//
// Fuses the Phi-3 MLP activation triple {Sigmoid, Mul, Mul_1}:
//     s = Sigmoid(g);  p = Mul(u, g);  y = Mul_1(p, s)   =>   y = u * g * sigmoid(g)
// into a single pass, eliminating the s and p intermediate tensors from ORT's
// allocator. idx_g / idx_u select which fused-node input is the gate (g, the
// Sigmoid operand) vs the up projection (u); they are resolved by tensor name
// at Compile time (fused-node input order is not assumed).
//
// To stay numerically equivalent to the unfused graph (so parity is exact, not
// approximate) the fp16 intermediate products are rounded to fp16 exactly where
// the original ops would round: p = half(u*g), s = half(sigmoid(g)),
// y = half(p*s). The fp32 path accumulates in fp32 (no intermediate rounding).
// Both associations of the two Muls collapse to y = u * g * sigmoid(g).
struct FusionCpuSwiGluKernel {
    FusionCpuSwiGluKernel(const OrtApi* ort_api, size_t idx_g, size_t idx_u)
        : ort_api_(ort_api), idx_g_(idx_g), idx_u_(idx_u) {}

    OrtStatus* Compute(OrtKernelContext* kctx) {
        Ort::KernelContext ctx{kctx};
        if (ctx.GetInputCount() != 2 || ctx.GetOutputCount() != 1) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuSwiGluKernel expects 2 inputs + 1 output");
        }
        Ort::ConstValue in_g = ctx.GetInput(idx_g_);
        Ort::ConstValue in_u = ctx.GetInput(idx_u_);
        auto tg = in_g.GetTensorTypeAndShapeInfo();
        auto tu = in_u.GetTensorTypeAndShapeInfo();
        const auto dt = tg.GetElementType();
        if (dt != tu.GetElementType() ||
            (dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
             dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuSwiGluKernel requires matching float16 or float32 inputs");
        }
        std::vector<int64_t> shape_g = tg.GetShape();
        std::vector<int64_t> shape_u = tu.GetShape();
        if (shape_g != shape_u) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuSwiGluKernel requires identical gate/up shapes (no broadcasting)");
        }
        int64_t n = 1;
        for (int64_t d : shape_g) n *= d;
        if (n < 0) {
            return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT,
                "FusionCpuSwiGluKernel: negative element count");
        }

        Ort::UnownedValue out = ctx.GetOutput(0, shape_g);
        if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const uint16_t* g = in_g.GetTensorData<uint16_t>();
            const uint16_t* u = in_u.GetTensorData<uint16_t>();
            uint16_t* y = out.GetTensorMutableData<uint16_t>();
            for (int64_t i = 0; i < n; ++i) {
                const float gf = HalfToFloat(g[i]);
                const float uf = HalfToFloat(u[i]);
                const float p  = HalfToFloat(FloatToHalf(uf * gf));            // Mul(u, g)
                const float s  = HalfToFloat(FloatToHalf(1.0f / (1.0f + std::exp(-gf))));  // Sigmoid(g)
                y[i] = FloatToHalf(p * s);                                     // Mul_1(p, s)
            }
        } else {
            const float* g = in_g.GetTensorData<float>();
            const float* u = in_u.GetTensorData<float>();
            float* y = out.GetTensorMutableData<float>();
            for (int64_t i = 0; i < n; ++i) {
                const float gf = g[i];
                const float s  = 1.0f / (1.0f + std::exp(-gf));  // Sigmoid(g)
                y[i] = gf * s * u[i];                            // u * g * sigmoid(g)
            }
        }
        return nullptr;
    }

private:
    const OrtApi* ort_api_ = nullptr;
    size_t idx_g_ = 0;
    size_t idx_u_ = 1;
};

// ===== MatMulNBits kernel (int4, block-wise, optional AWQ-scale fold) =======
//
// Reference dequant-GEMM for the com.microsoft MatMulNBits op (M3 scope:
// bits=4, fp16 A, uint8 packed B, fp16 scales, optional packed uint8
// zero_points, accuracy_level=0, no g_idx/bias). Computes
//     Y[m,n] = sum_k A'[m,k] * (q[n,k] - zp[n,blk]) * scale[n,blk]
// where blk = k / block_size, q is the 4-bit weight (low nibble = even k),
// accumulated in fp32 and rounded once to fp16. This matches the op's
// mathematical definition; it is NOT bit-identical to MLAS (different blocked
// kernels / accumulation order) but is within fp16 tolerance / top-1.
//
// M3+ AWQ fold: when this kernel stands in for a fused {Mul, MatMulNBits} pair
// (the *_weight_only_mul AWQ activation scale, a per-channel fp16 const [K]),
// A' = round_fp16(A * awq_scale[k]); otherwise A' = A. The fp16 round mirrors
// the unfused ONNX Mul (fp16 output) so the fold stays within tolerance.
//
// Inputs are addressed through an index map (idx_*), because a fused node's
// input order is assigned by ORT and differs between the standalone form and
// the {Mul, MatMulNBits} fused form. idx_scale_/idx_zp_ are -1 when absent.
//
// Deliberately naive scalar code: M3 proves correctness; a SIMD/prepacked
// variant is a later optimization milestone.
struct FusionCpuMatMulNBitsKernel {
    FusionCpuMatMulNBitsKernel(const OrtApi* ort_api, int bits, int block_size,
                               int idx_a, int idx_scale, int idx_b, int idx_scales, int idx_zp)
        : ort_api_(ort_api), bits_(bits), block_size_(block_size),
          idx_a_(idx_a), idx_scale_(idx_scale), idx_b_(idx_b),
          idx_scales_(idx_scales), idx_zp_(idx_zp) {}

    OrtStatus* Fail(const char* msg) const {
        return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT, msg);
    }

    OrtStatus* Compute(OrtKernelContext* kctx) {
        Ort::KernelContext ctx{kctx};
        if (ctx.GetOutputCount() != 1) return Fail("MatMulNBits: expected 1 output");
        const int n_in = static_cast<int>(ctx.GetInputCount());
        const bool has_scale = idx_scale_ >= 0;
        const bool has_zp = idx_zp_ >= 0;

        // Bounds-check every used index against the actual fused-node input
        // count (do not assume role-count == physical-input-count).
        int max_idx = std::max({idx_a_, idx_b_, idx_scales_});
        if (has_scale) max_idx = std::max(max_idx, idx_scale_);
        if (has_zp) max_idx = std::max(max_idx, idx_zp_);
        if (idx_a_ < 0 || idx_b_ < 0 || idx_scales_ < 0 || max_idx >= n_in) {
            return Fail("MatMulNBits: input index out of range");
        }

        Ort::ConstValue A = ctx.GetInput(idx_a_);
        Ort::ConstValue B = ctx.GetInput(idx_b_);
        Ort::ConstValue S = ctx.GetInput(idx_scales_);
        auto ta = A.GetTensorTypeAndShapeInfo();
        auto tb = B.GetTensorTypeAndShapeInfo();
        auto ts = S.GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType a_type = ta.GetElementType();
        // T1 (A / scales / Y) is fp16 (AWQ/DirectML flavor) or fp32 (the genai
        // int4-RTN accuracy_level=4 flavor). The int8 VNNI compute path is the
        // accuracy_level=4 algorithm and is dtype-agnostic: A is normalized to
        // fp32 internally and scales are consumed as fp32 (zero-copy when the
        // model already stores fp32 scales).
        const bool io_fp16 = (a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        if (a_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
            a_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return Fail("MatMulNBits: A must be float16 or float");
        if (tb.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) return Fail("MatMulNBits: B must be uint8");
        if (ts.GetElementType() != a_type) return Fail("MatMulNBits: scales dtype must match A (T1)");

        std::vector<int64_t> shape_a = ta.GetShape();
        if (shape_a.empty()) return Fail("MatMulNBits: A must have rank >= 1");
        const int64_t K = shape_a.back();
        int64_t M = 1;
        for (size_t i = 0; i + 1 < shape_a.size(); ++i) M *= shape_a[i];
        if (K <= 0 || M < 0) return Fail("MatMulNBits: bad A shape");

        std::vector<int64_t> shape_b = tb.GetShape();
        if (shape_b.size() != 3) return Fail("MatMulNBits: B must be rank-3 [N, k_blocks, blob]");
        const int64_t N = shape_b[0];
        const int64_t k_blocks = shape_b[1];
        const int64_t blob = shape_b[2];
        const int64_t bs = block_size_;
        if (bs <= 0 || blob != bs * bits_ / 8) return Fail("MatMulNBits: blob_size != block_size*bits/8");
        if (k_blocks * bs != K) return Fail("MatMulNBits: K must equal k_blocks*block_size (no partial block)");

        int64_t s_numel = 1;
        for (int64_t d : ts.GetShape()) s_numel *= d;
        if (s_numel != N * k_blocks) return Fail("MatMulNBits: scales numel != N*k_blocks");

        // AWQ activation scale (fused M3+): per-channel fp16 vector of length K.
        const uint16_t* awq = nullptr;
        if (has_scale) {
            Ort::ConstValue SC = ctx.GetInput(idx_scale_);
            auto tsc = SC.GetTensorTypeAndShapeInfo();
            if (tsc.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) return Fail("MatMulNBits: AWQ scale must be float16");
            int64_t a_numel = 1;
            for (int64_t d : tsc.GetShape()) a_numel *= d;
            if (a_numel != K) return Fail("MatMulNBits: AWQ scale numel != K");
            awq = SC.GetTensorData<uint16_t>();
        }

        const int64_t zp_cols = (k_blocks * bits_ + 7) / 8;
        const uint8_t* zp = nullptr;
        if (has_zp) {
            Ort::ConstValue Z = ctx.GetInput(idx_zp_);
            auto tz = Z.GetTensorTypeAndShapeInfo();
            if (tz.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) return Fail("MatMulNBits: zero_points must be uint8 (packed)");
            int64_t z_numel = 1;
            for (int64_t d : tz.GetShape()) z_numel *= d;
            if (z_numel != N * zp_cols) return Fail("MatMulNBits: zero_points numel != N*ceil(k_blocks*bits/8)");
            zp = Z.GetTensorData<uint8_t>();
        }

        const uint16_t* a16 = io_fp16 ? A.GetTensorData<uint16_t>() : nullptr;
        const float*    a32 = io_fp16 ? nullptr : A.GetTensorData<float>();
        const uint8_t*  bq = B.GetTensorData<uint8_t>();

        // Normalize scales to fp32 for the dtype-agnostic AVX-512 kernels.
        // fp32 scales (genai acc-4 flavor) are passed through zero-copy; fp16
        // scales (AWQ flavor) are converted in small per-column/per-tile chunks
        // at each call site (see scales_fp32). This avoids a full-tensor scale
        // conversion every Compute -- crucial for fp16 decode, where the cached
        // prepack path never re-reads scales at all.
        const uint16_t* sc16     = io_fp16 ? S.GetTensorData<uint16_t>() : nullptr;
        const float*    sc32base = io_fp16 ? nullptr : S.GetTensorData<float>();
        // Returns an fp32 pointer to `cols` columns of scales starting at column
        // n0 (each k_blocks contiguous, matching the [N, k_blocks] layout). fp32:
        // zero-copy raw pointer. fp16: converts into `buf` (grown as needed).
        auto scales_fp32 = [&](int64_t n0, int64_t cols, std::vector<float>& buf) -> const float* {
            if (!io_fp16) return sc32base + static_cast<size_t>(n0) * static_cast<size_t>(k_blocks);
            const size_t cnt = static_cast<size_t>(cols) * static_cast<size_t>(k_blocks);
            if (buf.size() < cnt) buf.resize(cnt);
            const uint16_t* src = sc16 + static_cast<size_t>(n0) * static_cast<size_t>(k_blocks);
            for (size_t i = 0; i < cnt; ++i) buf[i] = HalfToFloat(src[i]);
            return buf.data();
        };

        std::vector<int64_t> shape_y = shape_a;
        shape_y.back() = N;
        Ort::UnownedValue out = ctx.GetOutput(0, shape_y);
        uint16_t* y16 = io_fp16 ? out.GetTensorMutableData<uint16_t>() : nullptr;
        float*    y32 = io_fp16 ? nullptr : out.GetTensorMutableData<float>();
        // Store an fp32 result to output element [off] in the model's T1 dtype.
        auto store_y = [&](size_t off, float v) {
            if (io_fp16) y16[off] = FloatToHalf(v); else y32[off] = v;
        };

        const int default_zp = 1 << (bits_ - 1);  // 8 for 4-bit
        // M3++: dispatch the per-(m,n) dot to the AVX-512F kernel when the CPU
        // supports it and the block tiles 16-wide (always true: block_size is a
        // power of two >= 16). Falls back to the scalar reference otherwise, or
        // when ORT_FUSIONCPU_MATMULNBITS_SCALAR=1 forces it (parity testing).
        const bool use_avx512 = HasAvx512f() && (bs % 16 == 0) && !MatMulNBitsForceScalar();
        // Path B (opt-in): AVX-512-VNNI int8 dot. Requires VNNI support, an
        // opt-in env flag, and block_size % 16 == 0 (the register-blocked tile
        // kernel processes weights in 16-wide unpacks and 4-wide dpbusd strips,
        // so any block_size that is a multiple of 16 works). Otherwise we use the
        // bit-exact fp32 AVX-512 path.
        const bool use_vnni = use_avx512 && MatMulNBitsUseVnni() &&
                              HasAvx512Vnni() && (bs % 16 == 0);
        if (MatMulNBitsVerbose()) {
            static std::atomic<bool> logged{false};
            if (!logged.exchange(true)) {
                std::fprintf(stderr,
                    "[FusionCpuEP] MatMulNBits int4 dispatch: path=%s "
                    "(HasAvx512f=%d, HasAvx512Vnni=%d, block_size=%lld, force_scalar=%d, prepack=%d, threads=%d)\n",
                    use_vnni ? "AVX512-VNNI-int8" : (use_avx512 ? "AVX512F-fp32" : "scalar"),
                    static_cast<int>(HasAvx512f()),
                    static_cast<int>(HasAvx512Vnni()),
                    static_cast<long long>(bs),
                    static_cast<int>(MatMulNBitsForceScalar()),
                    static_cast<int>(use_vnni && MatMulNBitsPrepack()),
                    MatMulNBitsThreadCount());
                std::fflush(stderr);
            }
        }
        // M3+++: hoist the int4 weight dequant out of the per-output-row loop.
        // For each M-tile we dequantize each weight column once into a dense
        // fp32 buffer (Wdq, kept hot in cache) and reuse it across the tile's
        // rows, turning the hot path into a pure fp32 dot. This cuts redundant B
        // dequant work by the tile size for prefill (M>1); decode (M=1) is one
        // tile, so B is dequantized exactly once -- same as before. (The VNNI
        // path B does its own 16-column tiling and ignores mtile.)
        //
        // Tile size targets an ~1 MB Af tile so it stays resident across the N
        // sweep; clamped to [1, 64] rows.
        int64_t mtile = 64;
        {
            const int64_t per_row = K * 4;  // fp32 bytes per activation row
            const int64_t fit = (1 << 20) / std::max<int64_t>(per_row, 1);
            mtile = std::max<int64_t>(1, std::min<int64_t>(64, fit));
        }

        // Full activation matrix in fp32 (AWQ-scaled once), laid out [M, K].
        std::vector<float> Af(static_cast<size_t>(M) * static_cast<size_t>(K));
        for (int64_t m = 0; m < M; ++m) {
            const uint16_t* arow16 = io_fp16 ? a16 + m * K : nullptr;
            const float*    arow32 = io_fp16 ? nullptr : a32 + m * K;
            float* af = Af.data() + static_cast<size_t>(m) * static_cast<size_t>(K);
            for (int64_t k = 0; k < K; ++k) {
                float v = io_fp16 ? HalfToFloat(arow16[k]) : arow32[k];
                if (has_scale) v = HalfToFloat(FloatToHalf(v * HalfToFloat(awq[k])));
                af[static_cast<size_t>(k)] = v;
            }
        }

        // Path B: per-block int8 quantization of the activations (done once,
        // reused across the whole N sweep). For each (row m, block b):
        //   ascale[m,b] = max_k |Af[m,k]| / 127     (symmetric, per block)
        //   Au8[m,k]    = clamp(round(Af[m,k]/ascale) , -127, 127) + 128  in [1,255]
        // The +128 bias makes the value an unsigned uint8 for dpbusd; the caller
        // (DotInt8VnniColumn) undoes it via the per-block weight sum. A zero block
        // (ascale==0) maps to 128 (== quantized 0), contributing nothing.
        std::vector<uint8_t> Au8;
        std::vector<float> Ascale;
        if (use_vnni) {
            Au8.resize(static_cast<size_t>(M) * static_cast<size_t>(K));
            Ascale.resize(static_cast<size_t>(M) * static_cast<size_t>(k_blocks));
            for (int64_t m = 0; m < M; ++m) {
                const float* af = Af.data() + static_cast<size_t>(m) * static_cast<size_t>(K);
                uint8_t* au = Au8.data() + static_cast<size_t>(m) * static_cast<size_t>(K);
                float* asc = Ascale.data() + static_cast<size_t>(m) * static_cast<size_t>(k_blocks);
                for (int64_t b = 0; b < k_blocks; ++b) {
                    const int64_t base = b * bs;
                    float maxabs = 0.0f;
                    for (int64_t t = 0; t < bs; ++t) {
                        const float av = std::fabs(af[base + t]);
                        if (av > maxabs) maxabs = av;
                    }
                    const float scale = maxabs / 127.0f;
                    asc[b] = scale;
                    const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
                    for (int64_t t = 0; t < bs; ++t) {
                        int q = static_cast<int>(std::lrintf(af[base + t] * inv));
                        if (q < -127) q = -127;
                        else if (q > 127) q = 127;
                        au[base + t] = static_cast<uint8_t>(q + 128);
                    }
                }
            }
        }

        // M5: cached register-blocked int8 weight prepack. The weights are
        // constant initializers, so prepack each 16-column tile exactly once (on
        // the first Compute) into the tile-interleaved int8 layout and reuse it for
        // every subsequent Run. This removes the per-step int4->int8 unpack + weight
        // -sum recompute that otherwise dominates decode, letting M=1 also use the
        // reduction-free tile kernel. Guarded on a stable B pointer (true for
        // initializers); falls back to the per-call repack path if it ever differs.
        bool use_prepack = use_vnni && MatMulNBitsPrepack();
        // Compact int4 tile: only without zero_points (default_zp=8 => (q-zp) in
        // [-8,7] fits a signed nibble). Halves the resident weight bytes, which is
        // the decode bottleneck (cold-cache bandwidth bound).
        const bool want_int4 = !has_zp && MatMulNBitsInt4Tile();
        // Compact side data (fp16 scale + int16 wsum) is valid only on the int4
        // (no-zp) path and only when a block sum fits int16 (block_size <= 4096).
        const bool want_compact = want_int4 && MatMulNBitsInt4Compact() && (bs <= 4096);
        if (use_prepack) {
            std::call_once(prepack_once_, [&] {
                const int64_t n_tiles = (N + 15) / 16;
                pp_n_tiles_ = n_tiles;
                pp_int4_ = want_int4;
                pp_compact_ = want_compact;
                // Allocate the resident arrays up front; the per-tile prepack below
                // writes DISJOINT slices, so the tile loop is embarrassingly parallel.
                if (want_int4) {
                    pp_wq4_.resize(static_cast<size_t>(n_tiles) * k_blocks * bs * 8);
                    if (want_compact) {
                        pp_wscale_h_.resize(static_cast<size_t>(n_tiles) * k_blocks * 16);
                        pp_wsum16_.resize(static_cast<size_t>(n_tiles) * k_blocks * 16);
                    } else {
                        pp_wscale_.resize(static_cast<size_t>(n_tiles) * k_blocks * 16);
                        pp_wsum_.resize(static_cast<size_t>(n_tiles) * k_blocks * 16);
                    }
                } else {
                    pp_wscale_.resize(static_cast<size_t>(n_tiles) * k_blocks * 16);
                    pp_wsum_.resize(static_cast<size_t>(n_tiles) * k_blocks * 16);
                    pp_wq8_.resize(static_cast<size_t>(n_tiles) * k_blocks * bs * 16);
                }

                // Prepack a single 16-column tile. Uses ONLY thread-local scratch so
                // it is safe to run concurrently across tiles.
                auto prepack_tile = [&](int64_t t) {
                    std::vector<float> sct_l;  // per-tile fp16->fp32 scratch (fp16 only)
                    const int64_t n0 = t * 16;
                    const int cnt = static_cast<int>(std::min<int64_t>(16, N - n0));
                    if (want_int4) {
                        uint8_t* wq4 = pp_wq4_.data() + static_cast<size_t>(t) * k_blocks * bs * 8;
                        if (want_compact) {
                            // Repack into local fp32/int32 scratch, then NARROW into the
                            // resident fp16 scale + int16 wsum arrays (the wide copies
                            // never stay resident).
                            std::vector<float>   sc_t(static_cast<size_t>(k_blocks) * 16);
                            std::vector<int32_t> su_t(static_cast<size_t>(k_blocks) * 16);
                            RepackWeightTile16Int4Avx512(
                                wq4, sc_t.data(), su_t.data(),
                                bq + static_cast<size_t>(n0) * k_blocks * blob, k_blocks * blob,
                                scales_fp32(n0, cnt, sct_l), k_blocks,
                                cnt, k_blocks, bs, default_zp);
                            uint16_t* sh = pp_wscale_h_.data() + static_cast<size_t>(t) * k_blocks * 16;
                            int16_t*  s16 = pp_wsum16_.data() + static_cast<size_t>(t) * k_blocks * 16;
                            const size_t cnt_sd = static_cast<size_t>(k_blocks) * 16;
                            for (size_t i = 0; i < cnt_sd; ++i) {
                                sh[i] = FloatToHalf(sc_t[i]);
                                s16[i] = static_cast<int16_t>(su_t[i]);
                            }
                        } else {
                            RepackWeightTile16Int4Avx512(
                                wq4,
                                pp_wscale_.data() + static_cast<size_t>(t) * k_blocks * 16,
                                pp_wsum_.data() + static_cast<size_t>(t) * k_blocks * 16,
                                bq + static_cast<size_t>(n0) * k_blocks * blob, k_blocks * blob,
                                scales_fp32(n0, cnt, sct_l), k_blocks,
                                cnt, k_blocks, bs, default_zp);
                        }
                    } else {
                        RepackWeightTile16Int8Avx512(
                            pp_wq8_.data() + static_cast<size_t>(t) * k_blocks * bs * 16,
                            pp_wscale_.data() + static_cast<size_t>(t) * k_blocks * 16,
                            pp_wsum_.data() + static_cast<size_t>(t) * k_blocks * 16,
                            bq + static_cast<size_t>(n0) * k_blocks * blob, k_blocks * blob,
                            scales_fp32(n0, cnt, sct_l), k_blocks,
                            has_zp ? zp + static_cast<size_t>(n0) * zp_cols : nullptr, zp_cols,
                            has_zp, cnt, k_blocks, bs, default_zp);
                    }
                };

                // The prepack is model-sized one-time work that lands on the FIRST
                // Run (= prefill). Done serially it is single-core bound, which on a
                // many-slower-core box dominates first-token latency (and makes
                // --threads look like a no-op for prefill). Fan it across the same
                // pool the N-loop uses so first-token latency scales with cores.
                auto prepack_tile_range = [&](int64_t b, int64_t e) {
                    for (int64_t t = b; t < e; ++t) prepack_tile(t);
                };
                if (MatMulNBitsUseOrtPool()) {
                    constexpr int64_t kOrtPoolMaxWays = 64;
                    const int ways = static_cast<int>(std::min<int64_t>(n_tiles, kOrtPoolMaxWays));
                    if (ways <= 1) {
                        prepack_tile_range(0, n_tiles);
                    } else {
                        std::function<void(int64_t, int64_t)> rng_fn = prepack_tile_range;
                        struct PPCtx {
                            std::function<void(int64_t, int64_t)>* fn;
                            int64_t nt;
                            int ways;
                        } pp{&rng_fn, n_tiles, ways};
                        ctx.ParallelFor(
                            [](void* p, size_t task) {
                                auto* c = static_cast<PPCtx*>(p);
                                const int64_t chunk = (c->nt + c->ways - 1) / c->ways;
                                const int64_t b = std::min<int64_t>(static_cast<int64_t>(task) * chunk, c->nt);
                                const int64_t e = std::min<int64_t>(b + chunk, c->nt);
                                if (b < e) (*c->fn)(b, e);
                            },
                            static_cast<size_t>(ways), 0, &pp);
                    }
                } else {
                    FusionThreadPool& tp = MatMulNBitsThreadPool();
                    const int ways = static_cast<int>(std::min<int64_t>(n_tiles, static_cast<int64_t>(tp.size())));
                    if (ways <= 1) {
                        prepack_tile_range(0, n_tiles);
                    } else {
                        tp.ParallelFor(ways, [&](int task, int w) {
                            const int64_t chunk = (n_tiles + w - 1) / w;
                            const int64_t b = std::min<int64_t>(static_cast<int64_t>(task) * chunk, n_tiles);
                            const int64_t e = std::min<int64_t>(b + chunk, n_tiles);
                            if (b < e) prepack_tile_range(b, e);
                        });
                    }
                }
                prepack_b_ = bq;
                prepack_ready_ = true;
            });
            // Identity guard: only trust the cache if it was built from this exact
            // weight pointer and matches the current tiling.
            if (!prepack_ready_ || prepack_b_ != bq || pp_n_tiles_ != (N + 15) / 16) {
                use_prepack = false;
            }
        }

        // Dequantized weight column (fp32), reused across an M-tile's rows.
        // M4: the N-loop is parallelized across worker threads -- each output
        // column n is independent (writes a distinct y[*, n], reads shared Af
        // and its own B/scales/zp slice), so the work is embarrassingly parallel
        // and bit-identical to the serial result. Each worker owns a *private*
        // Wdq scratch buffer (the dequantized weight column is per-column state,
        // so it must not be shared across threads).
        // M-block width for the compact tile dot: kCompactMBMax when enabled, else
        // 1 (forces the single-row kernel). Read once per Compute.
        const int mb_step = MatMulNBitsMBlock() ? kCompactMBMax : 1;
        auto compute_n_range = [&](int64_t n_begin, int64_t n_end) {
            // Path B: AVX-512-VNNI int8 dot. Two kernels with an M-based dispatch:
            //   * Small M (decode): the per-column kernel streams each int8 weight
            //     column directly -- no prepack/interleave, which is pure overhead
            //     when there is only ~1 row to amortize it over.
            //   * Large M (prefill): the register-blocked tile kernel computes 16
            //     output columns at once (the dpbusd int32 lanes ARE the columns,
            //     so no per-block horizontal reduction) and streams the activation
            //     row once per 16-column tile. Its weight prepack/interleave is
            //     amortized across the M rows, so it wins once M is large enough.
            // The per-column kernel needs block_size % 64 == 0 (64-lane dpbusd
            // chunks); for smaller blocks the tile kernel (block_size % 16) is used
            // regardless of M.
            if (use_vnni) {
                // M5: cached prepack -- use the reduction-free tile kernel for ALL
                // M (including M=1 decode), reading the prepacked int8 tiles from
                // the per-kernel cache. Column ranges need not be tile-aligned: we
                // walk every tile overlapping [n_begin, n_end) and mask the writes
                // to the requested columns (boundary tiles may be recomputed by two
                // threads, but they write disjoint y columns, so the result is the
                // same as the serial computation).
                if (use_prepack) {
                    float out16[16];
                    float outMB[kCompactMBMax * 16];
                    const int64_t t0 = n_begin / 16;
                    const int64_t t1 = (n_end + 15) / 16;
                    for (int64_t t = t0; t < t1; ++t) {
                        const int64_t n0 = t * 16;
                        const float*   ws = pp_compact_ ? nullptr
                            : pp_wscale_.data() + static_cast<size_t>(t) * k_blocks * 16;
                        const int32_t* su = pp_compact_ ? nullptr
                            : pp_wsum_.data() + static_cast<size_t>(t) * k_blocks * 16;
                        const uint16_t* wsh = pp_compact_
                            ? pp_wscale_h_.data() + static_cast<size_t>(t) * k_blocks * 16 : nullptr;
                        const int16_t*  su16 = pp_compact_
                            ? pp_wsum16_.data() + static_cast<size_t>(t) * k_blocks * 16 : nullptr;
                        const int8_t*  wq = pp_int4_ ? nullptr
                            : pp_wq8_.data() + static_cast<size_t>(t) * k_blocks * bs * 16;
                        const uint8_t* wq4 = pp_int4_
                            ? pp_wq4_.data() + static_cast<size_t>(t) * k_blocks * bs * 8 : nullptr;
                        const int c0 = static_cast<int>(std::max<int64_t>(0, n_begin - n0));
                        const int c1 = static_cast<int>(std::min<int64_t>(16, n_end - n0));
                        if (pp_compact_) {
                            // M-blocked: expand each weight strip once per kCompactMBMax
                            // rows instead of once per row -- the prefill GEMM win. The
                            // single-row kernel (decode, M=1) is reached via mrows==1.
                            for (int64_t m0 = 0; m0 < M; m0 += mb_step) {
                                const int mr = static_cast<int>(std::min<int64_t>(mb_step, M - m0));
                                const uint8_t* au = Au8.data() + static_cast<size_t>(m0) * static_cast<size_t>(K);
                                const float* asc = Ascale.data() + static_cast<size_t>(m0) * static_cast<size_t>(k_blocks);
                                DotInt8VnniTile16Int4CompactMBAvx512(
                                    outMB, au, static_cast<int64_t>(K), asc, static_cast<int64_t>(k_blocks),
                                    wq4, wsh, su16, mr, k_blocks, bs);
                                for (int r = 0; r < mr; ++r) {
                                    const size_t yoff = static_cast<size_t>(m0 + r) * N + n0;
                                    for (int c = c0; c < c1; ++c) store_y(yoff + c, outMB[r * 16 + c]);
                                }
                            }
                        } else {
                            for (int64_t m = 0; m < M; ++m) {
                                const uint8_t* au = Au8.data() + static_cast<size_t>(m) * static_cast<size_t>(K);
                                const float* asc = Ascale.data() + static_cast<size_t>(m) * static_cast<size_t>(k_blocks);
                                if (pp_int4_) {
                                    DotInt8VnniTile16Int4Avx512(out16, au, asc, wq4, ws, su, k_blocks, bs);
                                } else {
                                    DotInt8VnniTile16Avx512(out16, au, asc, wq, ws, su, k_blocks, bs);
                                }
                                const size_t yoff = static_cast<size_t>(m) * N + n0;
                                for (int c = c0; c < c1; ++c) store_y(yoff + c, out16[c]);
                            }
                        }
                    }
                    return;
                }
                constexpr int64_t kVnniTileMinM = 16;  // M at/above which the tile kernel wins
                const bool use_percol = (M < kVnniTileMinM) && (bs % 64 == 0);

                if (use_percol) {
                    std::vector<int8_t>  Wq8(static_cast<size_t>(K));
                    std::vector<float>   Wscale(static_cast<size_t>(k_blocks));
                    std::vector<int32_t> Wsum(static_cast<size_t>(k_blocks));
                    std::vector<float>   sct1;  // per-column fp16->fp32 scratch (fp16 only)
                    for (int64_t n = n_begin; n < n_end; ++n) {
                        RepackWeightColumnInt8Avx512(
                            Wq8.data(), Wscale.data(), Wsum.data(),
                            bq + static_cast<size_t>(n) * k_blocks * blob,
                            scales_fp32(n, 1, sct1),
                            has_zp ? zp + static_cast<size_t>(n) * zp_cols : nullptr,
                            has_zp, k_blocks, bs, default_zp);
                        for (int64_t m = 0; m < M; ++m) {
                            const uint8_t* au = Au8.data() + static_cast<size_t>(m) * static_cast<size_t>(K);
                            const float* asc = Ascale.data() + static_cast<size_t>(m) * static_cast<size_t>(k_blocks);
                            const float acc = DotInt8VnniColumn(au, asc, Wq8.data(),
                                                                Wscale.data(), Wsum.data(),
                                                                k_blocks, bs);
                            store_y(static_cast<size_t>(m) * N + n, acc);
                        }
                    }
                    return;
                }

                constexpr int kTile = 16;
                std::vector<int8_t>  Wq8p(static_cast<size_t>(kTile) * static_cast<size_t>(K));
                std::vector<float>   Wscalep(static_cast<size_t>(kTile) * static_cast<size_t>(k_blocks));
                std::vector<int32_t> Wsump(static_cast<size_t>(kTile) * static_cast<size_t>(k_blocks));
                std::vector<float>   sctT;  // per-tile fp16->fp32 scratch (fp16 only)
                float out16[kTile];
                for (int64_t n0 = n_begin; n0 < n_end; n0 += kTile) {
                    const int cnt = static_cast<int>(std::min<int64_t>(kTile, n_end - n0));
                    RepackWeightTile16Int8Avx512(
                        Wq8p.data(), Wscalep.data(), Wsump.data(),
                        bq + static_cast<size_t>(n0) * k_blocks * blob, k_blocks * blob,
                        scales_fp32(n0, cnt, sctT), k_blocks,
                        has_zp ? zp + static_cast<size_t>(n0) * zp_cols : nullptr, zp_cols,
                        has_zp, cnt, k_blocks, bs, default_zp);
                    for (int64_t m = 0; m < M; ++m) {
                        const uint8_t* au = Au8.data() + static_cast<size_t>(m) * static_cast<size_t>(K);
                        const float* asc = Ascale.data() + static_cast<size_t>(m) * static_cast<size_t>(k_blocks);
                        DotInt8VnniTile16Avx512(out16, au, asc, Wq8p.data(),
                                                Wscalep.data(), Wsump.data(), k_blocks, bs);
                        const size_t yoff = static_cast<size_t>(m) * N + n0;
                        for (int c = 0; c < cnt; ++c) store_y(yoff + c, out16[c]);
                    }
                }
                return;
            }

            std::vector<float> Wdq(static_cast<size_t>(K));
            std::vector<float> sct1;  // per-column fp16->fp32 scratch (fp16 only)
            for (int64_t m0 = 0; m0 < M; m0 += mtile) {
                const int64_t m1 = std::min(m0 + mtile, M);
                for (int64_t n = n_begin; n < n_end; ++n) {
                    const uint8_t*  brow = bq + static_cast<size_t>(n) * k_blocks * blob;
                    const float*    srow = scales_fp32(n, 1, sct1);
                    const uint8_t*  zrow = has_zp ? zp + static_cast<size_t>(n) * zp_cols : nullptr;

                    if (use_avx512) {
                        DequantColumnAvx512(Wdq.data(), brow, srow, zrow,
                                            has_zp, k_blocks, bs, default_zp);
                    } else {
                        for (int64_t b = 0; b < k_blocks; ++b) {
                            const float scale = srow[b];
                            const int zpv = has_zp
                                ? ((b & 1) ? (zrow[b >> 1] >> 4) : (zrow[b >> 1] & 0x0F))
                                : default_zp;
                            const uint8_t* blk = brow + static_cast<size_t>(b) * blob;
                            const int64_t base = b * bs;
                            for (int64_t t = 0; t < bs; ++t) {
                                const uint8_t byte = blk[t >> 1];
                                const int q = (t & 1) ? (byte >> 4) : (byte & 0x0F);
                                Wdq[static_cast<size_t>(base + t)] =
                                    (static_cast<float>(q) - static_cast<float>(zpv)) * scale;
                            }
                        }
                    }

                    for (int64_t m = m0; m < m1; ++m) {
                        const float* af = Af.data() + static_cast<size_t>(m) * static_cast<size_t>(K);
                        float acc;
                        if (use_avx512) {
                            acc = DotFp32Avx512(af, Wdq.data(), K);
                        } else {
                            acc = 0.0f;
                            for (int64_t k = 0; k < K; ++k) {
                                acc += af[k] * Wdq[static_cast<size_t>(k)];
                            }
                        }
                        store_y(static_cast<size_t>(m) * N + n, acc);
                    }
                }
            }
        };

        // Decide how many ways to split the N-loop. With the persistent pool the
        // per-op synchronization cost is just a wakeup + barrier (microseconds),
        // not a thread spawn, so we parallelize far more aggressively than before
        // -- crucially, M=1 decode ops (tens of millions of MACs) now fan out
        // instead of running serially. kMacsPerWay is the minimum work a task
        // should own to clear the barrier overhead; the thread cap and the column
        // count N are upper bounds.
        constexpr int64_t kMacsPerWay = 1 << 20;  // ~1M MACs/task
        const int64_t total_macs = M * N * K;
        int64_t ways64 = std::max<int64_t>(1, total_macs / kMacsPerWay);
        if (ways64 > N) ways64 = std::max<int64_t>(1, N);

        if (MatMulNBitsUseOrtPool()) {
            // Dispatch onto ORT's intra-op pool (shared with MLAS and every CPU-EP
            // op) to avoid oversubscribing a second private pool. Cap the task count
            // so small decode ops don't schedule more tasks than the pool can use.
            constexpr int64_t kOrtPoolMaxWays = 64;
            const int nways = static_cast<int>(std::min<int64_t>(ways64, kOrtPoolMaxWays));
            if (nways <= 1) {
                compute_n_range(0, N);
            } else {
                std::function<void(int64_t, int64_t)> cn_fn = compute_n_range;
                struct PFCtx {
                    std::function<void(int64_t, int64_t)>* fn;
                    int64_t N;
                    int ways;
                } pf{&cn_fn, N, nways};
                // Captureless thunk -> convertible to the C function pointer the
                // ParallelFor API takes. pf lives on this stack frame and
                // ParallelFor is synchronous, so the pointer stays valid.
                ctx.ParallelFor(
                    [](void* p, size_t task) {
                        auto* c = static_cast<PFCtx*>(p);
                        const int64_t chunk = (c->N + c->ways - 1) / c->ways;
                        const int64_t nb = std::min<int64_t>(static_cast<int64_t>(task) * chunk, c->N);
                        const int64_t ne = std::min<int64_t>(nb + chunk, c->N);
                        if (nb < ne) (*c->fn)(nb, ne);
                    },
                    static_cast<size_t>(nways), 0, &pf);
            }
            return nullptr;
        }

        FusionThreadPool& tp = MatMulNBitsThreadPool();
        int nways = static_cast<int>(std::min<int64_t>(ways64, tp.size()));

        if (nways <= 1) {
            compute_n_range(0, N);
        } else {
            // Contiguous, balanced column ranges, one per task. Tasks are
            // independent (disjoint y columns, shared read-only Af/Au8), so the
            // result is identical regardless of the split.
            tp.ParallelFor(nways, [&compute_n_range, N](int task, int ways) {
                const int64_t chunk = (N + ways - 1) / ways;
                const int64_t nb = std::min<int64_t>(static_cast<int64_t>(task) * chunk, N);
                const int64_t ne = std::min<int64_t>(nb + chunk, N);
                if (nb < ne) compute_n_range(nb, ne);
            });
        }
        return nullptr;
    }

private:
    const OrtApi* ort_api_ = nullptr;
    int bits_ = 4;
    int block_size_ = 0;
    int idx_a_ = 0;
    int idx_scale_ = -1;
    int idx_b_ = 1;
    int idx_scales_ = 2;
    int idx_zp_ = -1;

    // M5 cached register-blocked int8 weight prepack (opt-in via
    // ORT_FUSIONCPU_MATMULNBITS_PREPACK). Built once on the first Compute and
    // reused across Runs; tile-interleaved int8 layout consumed by the
    // reduction-free DotInt8VnniTile16Avx512 kernel at all M.
    std::once_flag prepack_once_;
    bool prepack_ready_ = false;
    const uint8_t* prepack_b_ = nullptr;
    int64_t pp_n_tiles_ = 0;
    bool pp_int4_ = false;            // true: compact nibble tile (pp_wq4_); false: int8 (pp_wq8_)
    bool pp_compact_ = false;         // true: fp16 scale (pp_wscale_h_) + int16 wsum (pp_wsum16_)
    std::vector<int8_t>  pp_wq8_;     // n_tiles * k_blocks * block_size * 16 (int8 tile)
    std::vector<uint8_t> pp_wq4_;     // n_tiles * k_blocks * block_size * 8  (compact int4 tile)
    std::vector<float>   pp_wscale_;  // n_tiles * k_blocks * 16
    std::vector<int32_t> pp_wsum_;    // n_tiles * k_blocks * 16
    std::vector<uint16_t> pp_wscale_h_;  // n_tiles * k_blocks * 16 (compact: fp16 scale)
    std::vector<int16_t>  pp_wsum16_;    // n_tiles * k_blocks * 16 (compact: int16 wsum)
};

// ===== GroupQueryAttention kernel (fused decode/prefill, fp32, packed QKV) ===
//
// Claims the Phi-3 GroupQueryAttention (com.microsoft) node and runs a fused
// per-head attention that eliminates the stock kernel's per-head MlasGemm call
// overhead + intermediate score-buffer round trips -- the source of its
// ~46 ms/step decode cost. (The KV read is only ~14 MB/token, so the stock op
// is overhead-bound, not bandwidth-bound: a tight single-pass kernel has large
// headroom.)
//
// Supported configuration (GetCapability only claims this exact shape; anything
// else falls back to the stock CPUExecutionProvider GQA kernel): packed QKV
// (key/value inputs absent), fp32 Q/KV, grow-and-swap KV cache (SEPARATE
// past/present tensors -- NOT shared-buffer; the harness std::move()s present
// into next step's past), no attention_bias / position_ids / head_sink /
// KV-quant / local-window / softcap / smooth-softmax / output_qk, batch_size 1.
// do_rotary in {0,1}; group = num_heads / kv_num_heads >= 1.
//
// Unified math for prefill (past_seqlen P = 0) and decode (P > 0): with S new
// tokens, total = P + S; new token s has absolute position pos = P + s and
// causal length L = P + s + 1 (attends to present keys [0, L)).
struct __declspec(code_seg(".hotkern")) FusionCpuGqaKernel {
    FusionCpuGqaKernel(const OrtApi* ort_api, int num_heads, int kv_num_heads,
                       float scale, bool do_rotary, bool rotary_interleaved,
                       int idx_qkv, int idx_past_key, int idx_past_value,
                       int idx_cos, int idx_sin,
                       int idx_out, int idx_present_key, int idx_present_value,
                       int idx_seqlens_k,
                       int idx_key = -1, int idx_value = -1, int64_t local_window = -1)
        : ort_api_(ort_api), num_heads_(num_heads), kv_num_heads_(kv_num_heads),
          scale_(scale), do_rotary_(do_rotary), rotary_interleaved_(rotary_interleaved),
          idx_qkv_(idx_qkv), idx_past_key_(idx_past_key), idx_past_value_(idx_past_value),
          idx_cos_(idx_cos), idx_sin_(idx_sin),
          idx_out_(idx_out), idx_present_key_(idx_present_key),
          idx_present_value_(idx_present_value), idx_seqlens_k_(idx_seqlens_k),
          idx_key_(idx_key), idx_value_(idx_value), local_window_(local_window) {}

    OrtStatus* Fail(const char* msg) const {
        return ort_api_->CreateStatus(ORT_INVALID_ARGUMENT, msg);
    }

    // RoPE for one head row. Bit-identical to MlasRotaryEmbedOneRow_FallBack
    // (contrib_ops/.../mlas/lib/rotary_embedding.cpp): cos/sin caches store
    // rotary_dim/2 columns; `interleaved` selects GPT (pairwise) vs NeoX (split-
    // half) layout. Dimensions past rotary_dim are copied through unchanged.
    static void RotateRow(const float* in, const float* cosr, const float* sinr,
                          int rotary_dim, int head_size, bool interleaved, float* out) {
        const int half = rotary_dim / 2;
        for (int i = 0; i < rotary_dim; ++i) {
            int cache_idx, j;
            bool sign;
            if (interleaved) {
                cache_idx = (i / 2) % half;
                sign = (i & 1) != 0;
                j = sign ? i - 1 : i + 1;
            } else {
                cache_idx = i % half;
                sign = (i >= half);
                j = (i + half) % rotary_dim;
            }
            float v = in[i] * cosr[cache_idx];
            v += (sign ? in[j] : -in[j]) * sinr[cache_idx];
            out[i] = v;
        }
        for (int i = rotary_dim; i < head_size; ++i) out[i] = in[i];
    }

    OrtStatus* Compute(OrtKernelContext* kctx) {
        Ort::KernelContext ctx{kctx};
        if (ctx.GetOutputCount() != 3) return Fail("GQA: expected 3 outputs");

        Ort::ConstValue qkv = ctx.GetInput(idx_qkv_);
        Ort::ConstValue past_k = ctx.GetInput(idx_past_key_);
        Ort::ConstValue past_v = ctx.GetInput(idx_past_value_);

        const bool unpacked = (idx_key_ >= 0 && idx_value_ >= 0);

        auto t_qkv = qkv.GetTensorTypeAndShapeInfo();
        if (t_qkv.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            return Fail("GQA: query/qkv must be float32");
        std::vector<int64_t> sh_qkv = t_qkv.GetShape();  // packed [B,S,qkv_hidden] or unpacked query [B,S,q_hidden]
        if (sh_qkv.size() != 3) return Fail("GQA: query/qkv must be rank 3 [B,S,H]");
        const int64_t B = sh_qkv[0];
        const int64_t S = sh_qkv[1];
        if (B != 1) return Fail("GQA: only batch_size==1 is supported (set ORT_FUSIONCPU_GQA=0)");

        const int nh = num_heads_;
        const int kvh = kv_num_heads_;
        if (kvh <= 0 || nh % kvh != 0) return Fail("GQA: num_heads not a multiple of kv_num_heads");
        const int group = nh / kvh;

        // Head size + the per-row base/stride for each of Q, K, V.
        //  * packed:   one [B,S,(nh+2kvh)*hs] tensor sliced Q|K|V, all stride qkv_hidden.
        //  * unpacked: three tensors -- query [B,S,nh*hs], key/value [B,S,kvh*hs].
        int64_t hs, q_hidden, kv_hidden;
        const float* q_base;
        const float* k_base;
        const float* v_base;
        int64_t q_stride, k_stride, v_stride;
        const float* qkv_data = qkv.GetTensorData<float>();
        // key/value ConstValues must outlive the compute; declared here so their
        // backing tensors stay alive for the unpacked branch.
        Ort::ConstValue key_val, value_val;
        if (!unpacked) {
            const int64_t qkv_hidden = sh_qkv[2];
            if (qkv_hidden % (nh + 2 * kvh) != 0)
                return Fail("GQA: qkv_hidden not divisible by (num_heads + 2*kv_num_heads)");
            hs = qkv_hidden / (nh + 2 * kvh);
            q_hidden = static_cast<int64_t>(nh) * hs;
            kv_hidden = static_cast<int64_t>(kvh) * hs;
            q_base = qkv_data;
            k_base = qkv_data + q_hidden;
            v_base = qkv_data + q_hidden + kv_hidden;
            q_stride = k_stride = v_stride = qkv_hidden;
        } else {
            q_hidden = sh_qkv[2];
            if (q_hidden % nh != 0) return Fail("GQA: query hidden not divisible by num_heads");
            hs = q_hidden / nh;
            kv_hidden = static_cast<int64_t>(kvh) * hs;
            key_val = ctx.GetInput(idx_key_);
            value_val = ctx.GetInput(idx_value_);
            auto t_k = key_val.GetTensorTypeAndShapeInfo();
            auto t_v = value_val.GetTensorTypeAndShapeInfo();
            if (t_k.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                t_v.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                return Fail("GQA: key/value must be float32");
            std::vector<int64_t> sh_k = t_k.GetShape();
            std::vector<int64_t> sh_v = t_v.GetShape();
            if (sh_k.size() != 3 || sh_v.size() != 3) return Fail("GQA: key/value must be rank 3 [B,S,H]");
            if (sh_k[0] != B || sh_k[1] != S || sh_v[0] != B || sh_v[1] != S)
                return Fail("GQA: key/value batch/seq must match query");
            if (sh_k[2] != kv_hidden || sh_v[2] != kv_hidden)
                return Fail("GQA: key/value hidden must equal kv_num_heads*head_size");
            q_base = qkv_data;
            k_base = key_val.GetTensorData<float>();
            v_base = value_val.GetTensorData<float>();
            q_stride = q_hidden;
            k_stride = v_stride = kv_hidden;
        }

        std::vector<int64_t> sh_pk = past_k.GetTensorTypeAndShapeInfo().GetShape();  // [B, kvh, P, hs]
        if (sh_pk.size() != 4) return Fail("GQA: past_key must be rank 4 [B,kvh,P,hs]");
        const int64_t P = sh_pk[2];
        const int64_t total = P + S;

        // B1 own-KV (footprint) mode: derive the KV base length from the node's
        // seqlens_k input (= true total_kv - 1, driven by the growing attention
        // mask) rather than the incoming past tensor. This lets the harness feed
        // an EMPTY past (P==0) and discard present while the private fp16 cache
        // holds the real history -- the sole resident KV copy. seqlens_k[0] also
        // gives a unified clear/rewind control point (see below). Requires the
        // fp16 path (AVX-512); if requested without it, fail loudly rather than
        // silently attend over an empty past.
        const bool own_kv = GqaOwnKvEnabled();
        int64_t own_base = P;  // effective past length; == P unless own-KV overrides
        if (own_kv) {
            if (!(GqaFp16KvEnabled() && HasAvx512f()))
                return Fail("GQA own-KV requires the fp16 KV path (AVX-512); "
                            "unset ORT_FUSIONCPU_GQA_OWNKV or run on an AVX-512 host");
            if (idx_seqlens_k_ < 0) return Fail("GQA own-KV: seqlens_k input unresolved");
            Ort::ConstValue seqv = ctx.GetInput(idx_seqlens_k_);
            auto seqti = seqv.GetTensorTypeAndShapeInfo();
            const auto seq_et = seqti.GetElementType();
            int64_t total_kv;
            if (seq_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
                total_kv = static_cast<int64_t>(seqv.GetTensorData<int32_t>()[0]) + 1;
            else if (seq_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                total_kv = seqv.GetTensorData<int64_t>()[0] + 1;
            else
                return Fail("GQA own-KV: seqlens_k must be int32 or int64");
            own_base = total_kv - S;  // absolute position of the first new token
            if (own_base < 0) return Fail("GQA own-KV: seqlens_k+1 < S (bad length)");
        }

        // Sliding-window attention (local_window_): a query at absolute position
        // pos attends only to the most recent `local_window_` keys, i.e. the causal
        // range [max(0, L - local_window_), L) with L = pos + 1. The per-query start
        // offset is applied in both attend paths below (fp16 and fp32). KV rows
        // before the window are still stored but never scored, so long contexts
        // (KV length > window) are now correct rather than capped.

        const float scale = (scale_ != 0.0f) ? scale_ : (1.0f / std::sqrt(static_cast<float>(hs)));

        const float* cos_data = nullptr;
        const float* sin_data = nullptr;
        int rotary_dim = 0;
        if (do_rotary_) {
            Ort::ConstValue cosv = ctx.GetInput(idx_cos_);
            Ort::ConstValue sinv = ctx.GetInput(idx_sin_);
            std::vector<int64_t> sh_cos = cosv.GetTensorTypeAndShapeInfo().GetShape();  // [max_seq, rot/2]
            if (sh_cos.size() != 2) return Fail("GQA: cos_cache must be rank 2");
            rotary_dim = static_cast<int>(sh_cos[1] * 2);
            if (rotary_dim > hs) return Fail("GQA: rotary_dim > head_size");
            // Guard the cos/sin cache row access. We rotate only the NEW tokens,
            // at absolute positions [base, base + S), so the highest row read is
            // base + S - 1. If that exceeds the cache's row count we would read
            // out of bounds (a silent segfault / access violation). Fail loudly
            // instead -- mirrors the CPU GQA op's "cos_cache dimension 0 shall
            // not be less than total_sequence_length" check.
            const int64_t cos_rows = sh_cos[0];
            const int64_t rope_pos_end = (own_kv ? own_base : P) + S;  // one past the max row
            if (rope_pos_end > cos_rows)
                return Fail("GQA: total_sequence_length exceeds cos_cache rows "
                            "(rotary position out of range); the sequence is longer "
                            "than the model's rotary cache supports");
            cos_data = cosv.GetTensorData<float>();
            sin_data = sinv.GetTensorData<float>();
        }
        const int rot_half = rotary_dim / 2;

        const float* past_k_data = past_k.GetTensorData<float>();
        const float* past_v_data = past_v.GetTensorData<float>();

        std::vector<int64_t> out_shape = {B, S, q_hidden};
        std::vector<int64_t> pres_shape = {B, static_cast<int64_t>(kvh), total, hs};
        float* out_data = ctx.GetOutput(idx_out_, out_shape).GetTensorMutableData<float>();
        float* pk_data = ctx.GetOutput(idx_present_key_, pres_shape).GetTensorMutableData<float>();
        float* pv_data = ctx.GetOutput(idx_present_value_, pres_shape).GetTensorMutableData<float>();

        // Per-kv-head: build present K/V slice (past copied forward, new K
        // RoPE-rotated + appended, new V copied verbatim).
        auto build_one = [&](int kh) {
            const int64_t pres_off = static_cast<int64_t>(kh) * total * hs;
            const int64_t past_off = static_cast<int64_t>(kh) * P * hs;
            float* pk = pk_data + pres_off;
            float* pv = pv_data + pres_off;
            if (P > 0) {
                std::memcpy(pk, past_k_data + past_off, static_cast<size_t>(P) * hs * sizeof(float));
                std::memcpy(pv, past_v_data + past_off, static_cast<size_t>(P) * hs * sizeof(float));
            }
            for (int64_t s = 0; s < S; ++s) {
                const float* srcK = k_base + s * k_stride + static_cast<int64_t>(kh) * hs;
                const float* srcV = v_base + s * v_stride + static_cast<int64_t>(kh) * hs;
                // [*Fp16-KV_Append_New*]
                float* dstK = pk + (P + s) * hs;
                float* dstV = pv + (P + s) * hs;
                if (do_rotary_) {
                    const int64_t pos = P + s;
                    RotateRow(srcK, cos_data + pos * rot_half, sin_data + pos * rot_half,
                              rotary_dim, static_cast<int>(hs), rotary_interleaved_, dstK);
                } else {
                    std::memcpy(dstK, srcK, static_cast<size_t>(hs) * sizeof(float));
                }
                std::memcpy(dstV, srcV, static_cast<size_t>(hs) * sizeof(float));
            }
        };

        // Per-query-head: attention over the (already-built) present K/V of its
        // kv head. Reusable scratch passed in so callers control allocation.
        auto attend_one = [&](int h, std::vector<float>& scores, std::vector<float>& qrot) {
            const int kh = h / group;
            const float* pk = pk_data + static_cast<int64_t>(kh) * total * hs;
            const float* pv = pv_data + static_cast<int64_t>(kh) * total * hs;
            for (int64_t s = 0; s < S; ++s) {
                const int64_t pos = P + s;
                const int64_t L = pos + 1;  // causal length: attend to keys [0, L)
                // Sliding window: attend only the most recent local_window_ keys.
                const int64_t t0 = (local_window_ > 0 && L > local_window_) ? (L - local_window_) : 0;
                const float* srcQ = q_base + s * q_stride + static_cast<int64_t>(h) * hs;
                const float* q;
                if (do_rotary_) {
                    RotateRow(srcQ, cos_data + pos * rot_half, sin_data + pos * rot_half,
                              rotary_dim, static_cast<int>(hs), rotary_interleaved_, qrot.data());
                    q = qrot.data();
                } else {
                    q = srcQ;
                }
                // scores[t] = scale * (q . K[t]); track max for stable softmax.
                float maxv = -std::numeric_limits<float>::infinity();
                for (int64_t t = t0; t < L; ++t) {
                    const float* kt = pk + t * hs;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < hs; ++d) dot += q[d] * kt[d];
                    dot *= scale;
                    scores[static_cast<size_t>(t)] = dot;
                    if (dot > maxv) maxv = dot;
                }
                float sum = 0.0f;
                for (int64_t t = t0; t < L; ++t) {
                    const float e = std::exp(scores[static_cast<size_t>(t)] - maxv);
                    scores[static_cast<size_t>(t)] = e;
                    sum += e;
                }
                const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                float* o = out_data + s * q_hidden + static_cast<int64_t>(h) * hs;
                for (int64_t d = 0; d < hs; ++d) o[d] = 0.0f;
                for (int64_t t = t0; t < L; ++t) {
                    const float w = scores[static_cast<size_t>(t)] * inv;
                    const float* vt = pv + t * hs;
                    for (int64_t d = 0; d < hs; ++d) o[d] += w * vt[d];
                }
            }
        };

        FusionThreadPool& pool = MatMulNBitsThreadPool();
        auto run_parallel = [&pool](int count, const std::function<void(int)>& fn) {
            pool.ParallelFor(count, [&](int task, int nways) {
                const int begin = static_cast<int>(static_cast<int64_t>(task) * count / nways);
                const int end = static_cast<int>(static_cast<int64_t>(task + 1) * count / nways);
                for (int idx = begin; idx < end; ++idx) fn(idx);
            });
        };

        // fp16 private-cache fast path (DEFAULT). Requires AVX-512 for the F16C
        // in-register widen; without it we fall through to the fp32 paths below.
        // Keeps a persistent per-node fp16 K/V cache: appends only the S new rows
        // each step and attends by reading fp16 (widened to fp32), eliminating the
        // fp32 past->present copy-forward and halving the KV DRAM bytes.
        if (GqaFp16KvEnabled() && HasAvx512f()) {
            const int64_t hs64 = hs;

            // Effective past length (base) and cache reconciliation.
            //  - B2 (grow-and-swap): base = P; a new sequence (P==0) or any desync
            //    (P != cache_len_) rebuilds the fp16 cache from the incoming fp32 past.
            //  - own-KV (footprint): base = own_base (from seqlens_k). The past is
            //    empty and never read; seqlens_k[0] is the sole length/position AND
            //    a unified clear/rewind control: base==0 clears, 0<base<cache_len_
            //    rewinds (keep rows [0,base), discard the tail via cache_len_),
            //    base==cache_len_ appends. base>cache_len_ is an illegal gap.
            const int64_t base = own_kv ? own_base : P;
            const int64_t eff_total = base + S;
            bool reseed;
            if (own_kv) {
                if (base > cache_len_)
                    return Fail("GQA own-KV: position gap (seqlens_k base > cached length)");
                reseed = false;         // never read the (empty) past
                cache_len_ = base;      // clear/rewind: truncate to the resume point
            } else {
                reseed = (cache_len_ != P);
            }
            auto ensure_cap = [&](int64_t need_rows) {
                if (need_rows <= cache_cap_) return;
                int64_t new_cap = cache_cap_ > 0 ? cache_cap_ : 256;
                while (new_cap < need_rows) new_cap *= 2;
                const int64_t keep = reseed ? 0 : cache_len_;  // rows to preserve
                std::vector<uint16_t> nk(static_cast<size_t>(kvh) * new_cap * hs);
                std::vector<uint16_t> nv(static_cast<size_t>(kvh) * new_cap * hs);
                if (keep > 0 && cache_cap_ > 0) {
                    for (int kh = 0; kh < kvh; ++kh) {
                        std::memcpy(nk.data() + static_cast<int64_t>(kh) * new_cap * hs,
                                    kcache_.data() + static_cast<int64_t>(kh) * cache_cap_ * hs,
                                    static_cast<size_t>(keep) * hs * sizeof(uint16_t));
                        std::memcpy(nv.data() + static_cast<int64_t>(kh) * new_cap * hs,
                                    vcache_.data() + static_cast<int64_t>(kh) * cache_cap_ * hs,
                                    static_cast<size_t>(keep) * hs * sizeof(uint16_t));
                    }
                }
                kcache_.swap(nk);
                vcache_.swap(nv);
                cache_cap_ = new_cap;
            };
            ensure_cap(eff_total);
            const int64_t cap = cache_cap_;

            // On desync, rebuild rows [0,P) from the fp32 past. Past K is already
            // RoPE-rotated in the buffer, so store verbatim as fp16.
            if (reseed && P > 0) {
                run_parallel(kvh, [&](int kh) {
                    uint16_t* kc = kcache_.data() + static_cast<int64_t>(kh) * cap * hs;
                    uint16_t* vc = vcache_.data() + static_cast<int64_t>(kh) * cap * hs;
                    const float* pk = past_k_data + static_cast<int64_t>(kh) * P * hs;
                    const float* pv = past_v_data + static_cast<int64_t>(kh) * P * hs;
                    for (int64_t i = 0; i < P * hs; ++i) kc[i] = FloatToHalf(pk[i]);
                    for (int64_t i = 0; i < P * hs; ++i) vc[i] = FloatToHalf(pv[i]);
                });
            }

            // Append the S new rows (RoPE-rotate K, raw V) as fp16 -- the only
            // cache write on the decode fast path (S rows, not the whole history).
            run_parallel(kvh, [&](int kh) { //  [*Fp16-Append_Cache*]
                uint16_t* kc = kcache_.data() + static_cast<int64_t>(kh) * cap * hs;
                uint16_t* vc = vcache_.data() + static_cast<int64_t>(kh) * cap * hs;
                std::vector<float> krot(static_cast<size_t>(hs));
                for (int64_t s = 0; s < S; ++s) {
                    const float* srcK = k_base + s * k_stride + static_cast<int64_t>(kh) * hs;
                    const float* srcV = v_base + s * v_stride + static_cast<int64_t>(kh) * hs;
                    const int64_t pos = base + s;
                    uint16_t* dstK = kc + pos * hs;
                    uint16_t* dstV = vc + pos * hs;
                    if (do_rotary_) {
                        RotateRow(srcK, cos_data + pos * rot_half, sin_data + pos * rot_half,
                                  rotary_dim, static_cast<int>(hs), rotary_interleaved_, krot.data());
                        for (int64_t d = 0; d < hs; ++d) dstK[d] = FloatToHalf(krot[d]);
                    } else {
                        for (int64_t d = 0; d < hs; ++d) dstK[d] = FloatToHalf(srcK[d]);
                    }
                    for (int64_t d = 0; d < hs; ++d) dstV[d] = FloatToHalf(srcV[d]);
                }
            });
            cache_len_ = eff_total;

            // Attend from the fp16 cache (widen in-register), writing fp32 out.
            // present_key/value were allocated via GetOutput above but are left
            // UNWRITTEN: the harness moves the present handle into next-step past
            // without reading its contents, and this kernel is the sole KV
            // consumer (private cache is the source of truth), so skipping the
            // ~fp32 history copy is safe and is the bulk of the saved bandwidth.
            auto attend_f16 = [&](int h, std::vector<float>& scores, std::vector<float>& qrot) {
                const int kh = h / group;
                const uint16_t* kc = kcache_.data() + static_cast<int64_t>(kh) * cap * hs;
                const uint16_t* vc = vcache_.data() + static_cast<int64_t>(kh) * cap * hs;
                for (int64_t s = 0; s < S; ++s) {
                    const int64_t pos = base + s;
                    const int64_t L = pos + 1;  // causal length: attend keys [0, L)
                    // Sliding window: attend only the most recent local_window_ keys.
                    const int64_t t0 = (local_window_ > 0 && L > local_window_) ? (L - local_window_) : 0;
                    const float* srcQ = q_base + s * q_stride + static_cast<int64_t>(h) * hs;
                    const float* q;
                    if (do_rotary_) {
                        RotateRow(srcQ, cos_data + pos * rot_half, sin_data + pos * rot_half,
                                  rotary_dim, static_cast<int>(hs), rotary_interleaved_, qrot.data());
                        q = qrot.data();
                    } else {
                        q = srcQ;
                    }
                    float maxv = -std::numeric_limits<float>::infinity();
                    for (int64_t t = t0; t < L; ++t) {
                        const float dot = GqaDotF16Avx512(q, kc + t * hs, hs64) * scale;
                        scores[static_cast<size_t>(t)] = dot;
                        if (dot > maxv) maxv = dot;
                    }
                    float sum = 0.0f;
                    for (int64_t t = t0; t < L; ++t) {
                        const float e = std::exp(scores[static_cast<size_t>(t)] - maxv);
                        scores[static_cast<size_t>(t)] = e;
                        sum += e;
                    }
                    const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                    // Attends from the private cache (attend_f16), writing only out_data (via  o)
                    float* o = out_data + s * q_hidden + static_cast<int64_t>(h) * hs;
                    for (int64_t d = 0; d < hs; ++d) o[d] = 0.0f;
                    for (int64_t t = t0; t < L; ++t) {
                        const float w = scores[static_cast<size_t>(t)] * inv;
                        GqaAccumVF16Avx512(o, w, vc + t * hs, hs64);
                    }
                }
            };
            run_parallel(kvh, [&](int kh) {
                std::vector<float> scores(static_cast<size_t>(eff_total));
                std::vector<float> qrot(static_cast<size_t>(hs));
                const int h_begin = kh * group;
                for (int h = h_begin; h < h_begin + group; ++h) attend_f16(h, scores, qrot);
            });
            return nullptr; // Fp16 exits before the Fp32 path
        }

        // Default: FUSED cache-blocked pass (parallel over kv heads) -- build a kv
        // head's present K/V then immediately attend its query-head group while the
        // data is hot in L2/L1, avoiding a separate phase's ~438 MB/token DRAM
        // re-read of present. Phi-3 (group==1) keeps full num_heads-way parallelism.
        // Set ORT_FUSIONCPU_GQA_FUSED=0 for the legacy two-phase path (build-all then
        // attend-all, parallel over kv heads then query heads) -- A/B diagnostic only.
        if (GqaUseFusedPath()) {
            run_parallel(kvh, [&](int kh) {
                build_one(kh);
                std::vector<float> scores(static_cast<size_t>(total));
                std::vector<float> qrot(static_cast<size_t>(hs));
                const int h_begin = kh * group;
                for (int h = h_begin; h < h_begin + group; ++h) attend_one(h, scores, qrot);
            });
        } else {
            run_parallel(kvh, build_one);  // barrier: all present K/V ready
            run_parallel(nh, [&](int h) {
                std::vector<float> scores(static_cast<size_t>(total));
                std::vector<float> qrot(static_cast<size_t>(hs));
                attend_one(h, scores, qrot);
            });
        }
        return nullptr;
    }

    const OrtApi* ort_api_;
    int num_heads_;
    int kv_num_heads_;
    float scale_;
    bool do_rotary_;
    bool rotary_interleaved_;
    int idx_qkv_, idx_past_key_, idx_past_value_, idx_cos_, idx_sin_;
    int idx_out_, idx_present_key_, idx_present_value_;
    int idx_seqlens_k_ = -1;  // node input 5 (own-KV: KV length source)
    // Unpacked QKV: separate key/value node inputs (indices 1,2). When both are
    // >= 0 the kernel reads Q/K/V from three tensors instead of slicing one
    // packed [B,S,(nh+2kvh)*hs] tensor. -1/-1 = packed (single-tensor) mode.
    int idx_key_ = -1, idx_value_ = -1;
    // Sliding-window attention size (GQA local_window_size attr; -1 = global).
    // Each query at absolute position pos attends only the most recent
    // local_window_ keys, i.e. the causal range [max(0, pos+1-local_window_),
    // pos+1). Applied in both attend paths (fp16 default + fp32 fallback). KV rows
    // before the window are still stored (no eviction) but never scored, so long
    // contexts beyond the window are correct; ring-buffer memory bounding is a
    // possible future optimization.
    int64_t local_window_ = -1;

    // Private persistent fp16 KV cache (default path; see GqaFp16KvEnabled).
    // Layout per kv-head: [kvh][cache_cap_][hs], post-RoPE K and raw V stored as
    // fp16. Survives across Run() calls (the kernel object is per-node), so
    // decode appends one row/step and never re-copies history. cache_len_ tracks
    // valid rows; it must equal the incoming past_len P on the fast path (a
    // new sequence resets via P==0; any P!=cache_len_ desync rebuilds from the
    // fp32 past). Serialized by the EP's per-node run mutex (Compute is single-
    // threaded per node per Run; the internal parallelism is over kv-heads).
    std::vector<uint16_t> kcache_;
    std::vector<uint16_t> vcache_;
    int64_t cache_len_ = 0;
    int64_t cache_cap_ = 0;
};

namespace {

// Single concrete NodeComputeInfo that dispatches through per-op function
// pointers set at construction time (the ORT struct has no virtual dtor, so we
// must not delete two distinct derived types via the base pointer).
struct FusionCpuNodeComputeInfo : OrtNodeComputeInfo {
    using StateLookupFn = OrtStatus* (*)(FusionCpuEp& ep, const std::string& node_name, void** out_state);
    using ComputeFn     = OrtStatus* (*)(void* state, OrtKernelContext* kctx);

    FusionCpuNodeComputeInfo(FusionCpuEp& ep_in, StateLookupFn lookup, ComputeFn compute)
        : ep(ep_in), lookup_fn(lookup), compute_fn_(compute) {
        ort_version_supported = ORT_API_VERSION;
        CreateState  = &CreateStateImpl;
        Compute      = &ComputeImpl;
        ReleaseState = &ReleaseStateImpl;
    }

    static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                   OrtNodeComputeContext* ctx,
                                                   void** state) {
        auto* self = static_cast<FusionCpuNodeComputeInfo*>(this_ptr);
        std::string node_name = self->ep.ep_api()->NodeComputeContext_NodeName(ctx);
        return self->lookup_fn(self->ep, node_name, state);
    }
    static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* state,
                                               OrtKernelContext* kctx) {
        auto* self = static_cast<FusionCpuNodeComputeInfo*>(this_ptr);
        return self->compute_fn_(state, kctx);
    }
    static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/, void* /*state*/) {
        // Kernels live in the FusionCpuEp's per-op maps; nothing per-call.
    }

    FusionCpuEp&  ep;
    StateLookupFn lookup_fn;
    ComputeFn     compute_fn_;
};

OrtStatus* LookupMulKernel(FusionCpuEp& ep, const std::string& node_name, void** out_state) {
    auto it = ep.mul_kernels().find(node_name);
    if (it == ep.mul_kernels().end()) {
        std::string msg = "FusionCpuEp: no Mul kernel registered for fused node '" + node_name + "'";
        return ep.ort_api()->CreateStatus(ORT_EP_FAIL, msg.c_str());
    }
    *out_state = it->second.get();
    return nullptr;
}
OrtStatus* ComputeMulKernel(void* state, OrtKernelContext* kctx) {
    return reinterpret_cast<FusionCpuMulKernel*>(state)->Compute(kctx);
}

OrtStatus* LookupSigmoidKernel(FusionCpuEp& ep, const std::string& node_name, void** out_state) {
    auto it = ep.sigmoid_kernels().find(node_name);
    if (it == ep.sigmoid_kernels().end()) {
        std::string msg = "FusionCpuEp: no Sigmoid kernel registered for fused node '" + node_name + "'";
        return ep.ort_api()->CreateStatus(ORT_EP_FAIL, msg.c_str());
    }
    *out_state = it->second.get();
    return nullptr;
}
OrtStatus* ComputeSigmoidKernel(void* state, OrtKernelContext* kctx) {
    return reinterpret_cast<FusionCpuSigmoidKernel*>(state)->Compute(kctx);
}

OrtStatus* LookupSwiGluKernel(FusionCpuEp& ep, const std::string& node_name, void** out_state) {
    auto it = ep.swiglu_kernels().find(node_name);
    if (it == ep.swiglu_kernels().end()) {
        std::string msg = "FusionCpuEp: no SwiGLU kernel registered for fused node '" + node_name + "'";
        return ep.ort_api()->CreateStatus(ORT_EP_FAIL, msg.c_str());
    }
    *out_state = it->second.get();
    return nullptr;
}
OrtStatus* ComputeSwiGluKernel(void* state, OrtKernelContext* kctx) {
    return reinterpret_cast<FusionCpuSwiGluKernel*>(state)->Compute(kctx);
}

OrtStatus* LookupMatMulNBitsKernel(FusionCpuEp& ep, const std::string& node_name, void** out_state) {
    auto it = ep.matmulnbits_kernels().find(node_name);
    if (it == ep.matmulnbits_kernels().end()) {
        std::string msg = "FusionCpuEp: no MatMulNBits kernel registered for fused node '" + node_name + "'";
        return ep.ort_api()->CreateStatus(ORT_EP_FAIL, msg.c_str());
    }
    *out_state = it->second.get();
    return nullptr;
}
OrtStatus* ComputeMatMulNBitsKernel(void* state, OrtKernelContext* kctx) {
    return reinterpret_cast<FusionCpuMatMulNBitsKernel*>(state)->Compute(kctx);
}

OrtStatus* LookupGqaKernel(FusionCpuEp& ep, const std::string& node_name, void** out_state) {
    auto it = ep.gqa_kernels().find(node_name);
    if (it == ep.gqa_kernels().end()) {
        std::string msg = "FusionCpuEp: no GroupQueryAttention kernel registered for fused node '" + node_name + "'";
        return ep.ort_api()->CreateStatus(ORT_EP_FAIL, msg.c_str());
    }
    *out_state = it->second.get();
    return nullptr;
}
OrtStatus* ComputeGqaKernel(void* state, OrtKernelContext* kctx) {
    return reinterpret_cast<FusionCpuGqaKernel*>(state)->Compute(kctx);
}

}  // namespace

// ===== FusionCpuEp =========================================================

FusionCpuEp::FusionCpuEp(const OrtApi* ort_api, const OrtEpApi* ep_api,
                         const OrtLogger* logger, std::string name)
    : OrtEp{}, ort_api_(ort_api), ep_api_(ep_api), logger_(logger), name_(std::move(name)) {
    ort_version_supported   = ORT_API_VERSION;
    GetName                 = &GetNameImpl;
    GetCapability           = &GetCapabilityImpl;
    Compile                 = &CompileImpl;
    ReleaseNodeComputeInfos = &ReleaseNodeComputeInfosImpl;
    Sync                    = &SyncImpl;
    // OnRunEnd / IsConcurrentRunSupported / GetKernelRegistry /
    // GetCompiledModelCompatibilityInfo / CreateAllocator /
    // CreateSyncStreamForDevice are intentionally left null. ORT falls back to
    // its base behaviour for each (verified against the 1.27 plugin-EP driver).
}

FusionCpuEp::~FusionCpuEp() {
    mul_kernels_.clear();
    sigmoid_kernels_.clear();
    swiglu_kernels_.clear();
    matmulnbits_kernels_.clear();
    gqa_kernels_.clear();
}

/*static*/ const char* ORT_API_CALL FusionCpuEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
    return static_cast<const FusionCpuEp*>(this_ptr)->name_.c_str();
}

/*static*/ OrtStatus* ORT_API_CALL FusionCpuEp::GetCapabilityImpl(
        OrtEp* this_ptr, const OrtGraph* ort_graph, OrtEpGraphSupportInfo* info) noexcept {
    auto* self = static_cast<FusionCpuEp*>(this_ptr);
    try {
        Ort::ConstGraph graph{ort_graph};
        std::vector<Ort::ConstNode> nodes = graph.GetNodes();

        // Producer/consumer indexes + graph-output set, used by the SwiGLU
        // pattern matcher to prove the s/p intermediates are internal-only
        // (single consumer, not a graph output) before eliminating them.
        std::unordered_map<std::string, size_t> producer;             // tensor name -> producing node idx
        std::unordered_map<std::string, std::vector<size_t>> consumers;  // tensor name -> consuming node idxs
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (const auto& o : nodes[i].GetOutputs()) {
                if (static_cast<const OrtValueInfo*>(o) == nullptr) continue;  // absent optional output
                const std::string nm = o.GetName();
                if (!nm.empty()) producer[nm] = i;
            }
            for (const auto& in : nodes[i].GetInputs()) {
                if (static_cast<const OrtValueInfo*>(in) == nullptr) continue;  // absent optional input
                const std::string nm = in.GetName();
                if (!nm.empty()) consumers[nm].push_back(i);
            }
        }
        std::unordered_set<std::string> graph_outputs;
        for (const auto& go : graph.GetOutputs()) {
            const std::string nm = go.GetName();
            if (!nm.empty()) graph_outputs.insert(nm);
        }

        // ---- SwiGLU triple pre-pass ----
        // Detect and claim {Sigmoid, Mul, Mul} activation triples
        // (y = u * g * sigmoid(g)) BEFORE the standalone Mul/Sigmoid branches, so
        // the two Muls of each triple are not double-claimed by the standalone
        // fp32-Mul branch (which would otherwise conflict with the triple claim).
        // Two graph associations occur in the wild (both collapse to the same y):
        //   A (combine-first): p = Mul(u, g);  y = Mul(p, s)
        //   B (SiLU-first):    p = Mul(g, s);  y = Mul(p, u)
        std::unordered_set<size_t> swiglu_claimed;  // node indices consumed by a triple
        for (size_t si = 0; si < nodes.size(); ++si) {
            const Ort::ConstNode& snode = nodes[si];
            if (snode.GetOperatorType() != "Sigmoid" || !snode.GetDomain().empty()) continue;
            auto ins  = snode.GetInputs();
            auto outs = snode.GetOutputs();
            if (ins.size() != 1 || outs.size() != 1) continue;
            const auto act_dt = TensorElementTypeOf(ins[0]);
            const auto y_dt   = TensorElementTypeOf(outs[0]);
            if (act_dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
                act_dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
            if (y_dt != act_dt) continue;

            const std::string g_name = ins[0].GetName();
            const std::string s_name = outs[0].GetName();
            if (g_name.empty() || s_name.empty()) continue;

            // A Mul is claimable only if all boundary tensors match act_dt.
            auto mul_dt_ok = [&](const auto& mi, const auto& mo) {
                for (const auto& v : mi) if (TensorElementTypeOf(v) != act_dt) return false;
                for (const auto& v : mo) if (TensorElementTypeOf(v) != act_dt) return false;
                return true;
            };

            // s must be internal: exactly one consumer and not a graph output.
            if (graph_outputs.count(s_name)) continue;
            auto cs = consumers.find(s_name);
            if (cs == consumers.end() || cs->second.size() != 1) continue;
            const size_t mul_s_idx = cs->second[0];
            const Ort::ConstNode& mul_s = nodes[mul_s_idx];  // the Mul that consumes s
            if (mul_s.GetOperatorType() != "Mul" || !mul_s.GetDomain().empty()) continue;
            auto ms_ins  = mul_s.GetInputs();
            auto ms_outs = mul_s.GetOutputs();
            if (ms_ins.size() != 2 || ms_outs.size() != 1) continue;
            if (!mul_dt_ok(ms_ins, ms_outs)) continue;

            // other_s = the mul_s operand that is not s.
            const std::string msa = ms_ins[0].GetName(), msb = ms_ins[1].GetName();
            std::string other_s;
            if (msa == s_name && msb != s_name)      other_s = msb;
            else if (msb == s_name && msa != s_name) other_s = msa;
            else continue;

            size_t mul_a_idx = SIZE_MAX, mul_b_idx = SIZE_MAX;
            Ort::ConstValueInfo u_vi = ins[0];
            std::string u_name;

            if (other_s == g_name) {
                // Topology B: mul_s = Mul(g, s) -> p (SiLU); p feeds a second
                // Mul with u.  y = Mul(p, u).
                const std::string p_name = ms_outs[0].GetName();
                if (p_name.empty() || graph_outputs.count(p_name)) continue;
                auto cp = consumers.find(p_name);
                if (cp == consumers.end() || cp->second.size() != 1) continue;
                const size_t mul_f_idx = cp->second[0];
                const Ort::ConstNode& mul_f = nodes[mul_f_idx];
                if (mul_f.GetOperatorType() != "Mul" || !mul_f.GetDomain().empty()) continue;
                auto mf_ins  = mul_f.GetInputs();
                auto mf_outs = mul_f.GetOutputs();
                if (mf_ins.size() != 2 || mf_outs.size() != 1) continue;
                if (!mul_dt_ok(mf_ins, mf_outs)) continue;
                const std::string mfa = mf_ins[0].GetName(), mfb = mf_ins[1].GetName();
                if (mfa == p_name && mfb != p_name)      { u_name = mfb; u_vi = mf_ins[1]; }
                else if (mfb == p_name && mfa != p_name) { u_name = mfa; u_vi = mf_ins[0]; }
                else continue;
                mul_a_idx = mul_s_idx; mul_b_idx = mul_f_idx;
            } else {
                // Topology A: other_s = p, produced by a Mul(u, g); mul_s =
                // Mul(p, s) -> y.
                const std::string p_name = other_s;
                if (graph_outputs.count(p_name)) continue;
                auto cp = consumers.find(p_name);
                if (cp == consumers.end() || cp->second.size() != 1) continue;
                auto pp = producer.find(p_name);
                if (pp == producer.end()) continue;
                const size_t mul_o_idx = pp->second;
                const Ort::ConstNode& mul_o = nodes[mul_o_idx];
                if (mul_o.GetOperatorType() != "Mul" || !mul_o.GetDomain().empty()) continue;
                auto mo_ins  = mul_o.GetInputs();
                auto mo_outs = mul_o.GetOutputs();
                if (mo_ins.size() != 2 || mo_outs.size() != 1) continue;
                if (!mul_dt_ok(mo_ins, mo_outs)) continue;
                const std::string moa = mo_ins[0].GetName(), mob = mo_ins[1].GetName();
                if (moa == g_name && mob != g_name)      { u_name = mob; u_vi = mo_ins[1]; }
                else if (mob == g_name && moa != g_name) { u_name = moa; u_vi = mo_ins[0]; }
                else continue;  // g absent, or g == u boundary collapse
                mul_a_idx = mul_o_idx; mul_b_idx = mul_s_idx;
            }
            if (u_name.empty() || u_name == g_name) continue;

            // g and u must be elementwise-identical shape (no broadcasting).
            auto sg = GetShapeAllowSymbolic(ins[0]);
            auto su = GetShapeAllowSymbolic(u_vi);
            if (!sg || !su || sg->size() != su->size()) continue;
            bool dim_conflict = false;
            for (size_t k = 0; k < sg->size(); ++k) {
                int64_t a = (*sg)[k], b = (*su)[k];
                if (a >= 0 && b >= 0 && a != b) { dim_conflict = true; break; }
            }
            if (dim_conflict) continue;

            OrtNodeFusionOptions fopts{};
            fopts.ort_version_supported      = ORT_API_VERSION;
            fopts.drop_constant_initializers = false;
            const OrtNode* group[3] = {
                static_cast<const OrtNode*>(snode),
                static_cast<const OrtNode*>(nodes[mul_a_idx]),
                static_cast<const OrtNode*>(nodes[mul_b_idx]),
            };
            if (OrtStatus* st = self->ep_api_->EpGraphSupportInfo_AddNodesToFuse(
                    info, group, 3, &fopts)) return st;
            swiglu_claimed.insert(si);
            swiglu_claimed.insert(mul_a_idx);
            swiglu_claimed.insert(mul_b_idx);
        }

        for (size_t ni = 0; ni < nodes.size(); ++ni) {
            if (swiglu_claimed.count(ni)) continue;  // already fused into a SwiGLU triple
            const Ort::ConstNode& node = nodes[ni];
            const std::string op  = node.GetOperatorType();
            const std::string dom = node.GetDomain();

            // ---- Mul (default ONNX domain) — fp32, identical-shape ----
            if (op == "Mul" && dom.empty()) {
                auto ins  = node.GetInputs();
                auto outs = node.GetOutputs();
                if (ins.size() != 2 || outs.size() != 1) continue;
                const auto a_dt = TensorElementTypeOf(ins[0]);
                const auto b_dt = TensorElementTypeOf(ins[1]);
                const auto y_dt = TensorElementTypeOf(outs[0]);
                if (a_dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
                if (b_dt != a_dt || y_dt != a_dt) continue;
                auto s0 = GetShapeAllowSymbolic(ins[0]);
                auto s1 = GetShapeAllowSymbolic(ins[1]);
                if (!s0 || !s1) continue;
                if (s0->size() != s1->size()) continue;  // ranks must match
                // Reject only if both dims are concrete and unequal; the
                // Compute() shape guard catches symbolic mismatches.
                bool dim_conflict = false;
                for (size_t i = 0; i < s0->size(); ++i) {
                    int64_t a = (*s0)[i], b = (*s1)[i];
                    if (a >= 0 && b >= 0 && a != b) { dim_conflict = true; break; }
                }
                if (dim_conflict) continue;

                OrtNodeFusionOptions fopts{};
                fopts.ort_version_supported      = ORT_API_VERSION;
                fopts.drop_constant_initializers = false;
                const OrtNode* one_node = static_cast<const OrtNode*>(node);
                if (OrtStatus* st = self->ep_api_->EpGraphSupportInfo_AddNodesToFuse(
                        info, &one_node, 1, &fopts)) return st;
                continue;
            }

            // ---- Sigmoid (default ONNX domain) — fp16 or fp32, unary ----
            // SwiGLU triples were already claimed in the pre-pass above; any
            // Sigmoid reaching here is standalone (M2a).
            if (op == "Sigmoid" && dom.empty()) {
                auto ins  = node.GetInputs();
                auto outs = node.GetOutputs();
                if (ins.size() != 1 || outs.size() != 1) continue;
                const auto act_dt = TensorElementTypeOf(ins[0]);
                const auto y_dt   = TensorElementTypeOf(outs[0]);
                if (act_dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
                    act_dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
                if (y_dt != act_dt) continue;

                OrtNodeFusionOptions fopts{};
                fopts.ort_version_supported      = ORT_API_VERSION;
                fopts.drop_constant_initializers = false;
                const OrtNode* one_node = static_cast<const OrtNode*>(node);
                if (OrtStatus* st = self->ep_api_->EpGraphSupportInfo_AddNodesToFuse(
                        info, &one_node, 1, &fopts)) return st;
                continue;
            }

            // ---- MatMulNBits (com.microsoft) — int4 block-wise, fp16 A ----
            // Claimed by default (VNNI/compact-int4 kernel); ORT_FUSIONCPU_MATMULNBITS=0
            // opts out to cpuEP/MLAS. Support bits=4, accuracy_level=0, fp16 A, uint8 
            // packed 8, fp16 scales, optional packed uint8 zero_points, no g_idx/bias. 
            // Attrs are authoritative and must agree with the constant initializer shapes.

            if (op == "MatMulNBits" && dom == "com.microsoft") {
                auto ins  = node.GetInputs();
                auto outs = node.GetOutputs();
                if (outs.size() != 1) continue;
                const size_t nin = ins.size();
                if (nin != 3 && nin != 4) continue;  // reject g_idx/bias forms

                int64_t bits = 4, block_size = 0, K = 0, N = 0, acc_level = 0;
                if (!ReadInt64Attr(node, "bits", bits)) bits = 4;
                if (bits != 4) continue;
                if (!ReadInt64Attr(node, "block_size", block_size) || block_size <= 0) continue;
                if (!ReadInt64Attr(node, "K", K) || !ReadInt64Attr(node, "N", N)) continue;
                if (K <= 0 || N <= 0 || K % block_size != 0) continue;
                if (!ReadInt64Attr(node, "accuracy_level", acc_level)) acc_level = 0;
                // accuracy_level 0 (unset/fp compute, AWQ/DirectML flavor) and 4
                // (int8 dynamic-activation compute, genai int4-RTN flavor) are both
                // handled by the int8 VNNI kernel. Reject 1/2/3 (explicit fp32/
                // fp16/bf16 compute) — not our path.
                if (acc_level != 0 && acc_level != 4) continue;

                // dtypes: A/scales share T1 (fp16 AWQ flavor or fp32 genai acc-4
                // flavor), B uint8; B/scales must be const.
                const ONNXTensorElementDataType a_dt = TensorElementTypeOf(ins[0]);
                if (a_dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
                    a_dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
                if (TensorElementTypeOf(ins[1]) != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) continue;
                if (TensorElementTypeOf(ins[2]) != a_dt) continue;  // scales dtype == A (T1)
                if (!ins[1].IsConstantInitializer() || !ins[2].IsConstantInitializer()) continue;
                const bool has_zp = (nin == 4);
                if (has_zp) {
                    if (TensorElementTypeOf(ins[3]) != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) continue;  // packed only
                    if (!ins[3].IsConstantInitializer()) continue;
                }

                const int64_t k_blocks = K / block_size;
                const int64_t blob = block_size * bits / 8;

                // A's last dim must be concrete and equal K.
                auto sa = GetShapeAllowSymbolic(ins[0]);
                if (!sa || sa->empty() || sa->back() != K) continue;
                // B == [N, k_blocks, blob].
                auto sb = GetShapeAllowSymbolic(ins[1]);
                if (!sb || sb->size() != 3) continue;
                if ((*sb)[0] != N || (*sb)[1] != k_blocks || (*sb)[2] != blob) continue;
                // scales numel == N*k_blocks.
                auto ss = GetShapeAllowSymbolic(ins[2]);
                if (!ss) continue;
                int64_t s_numel = 1;
                for (int64_t d : *ss) s_numel *= d;
                if (s_numel != N * k_blocks) continue;
                // zero_points numel == N*ceil(k_blocks*bits/8).
                if (has_zp) {
                    auto sz = GetShapeAllowSymbolic(ins[3]);
                    if (!sz) continue;
                    int64_t z_numel = 1;
                    for (int64_t d : *sz) z_numel *= d;
                    if (z_numel != N * ((k_blocks * bits + 7) / 8)) continue;
                }

                OrtNodeFusionOptions fopts{};
                fopts.ort_version_supported      = ORT_API_VERSION;
                fopts.drop_constant_initializers = false;  // need B/scales/zp (+AWQ scale) as Compute inputs
                const OrtNode* mm_node = static_cast<const OrtNode*>(node);

                // M3+ AWQ fold: if A is produced by a *_weight_only_mul (Mul with
                // a per-channel fp16 const [K] operand, single internal
                // consumer), claim {Mul, MatMulNBits} as one 2-node group so the
                // scale folds into the kernel. Otherwise claim MatMulNBits alone.
                const OrtNode* foldable_mul = nullptr;
                {
                    auto pit = producer.find(ins[0].GetName());
                    if (pit != producer.end()) {
                        const Ort::ConstNode& mul = nodes[pit->second];
                        if (mul.GetOperatorType() == "Mul" && mul.GetDomain().empty()) {
                            auto m_in  = mul.GetInputs();
                            auto m_out = mul.GetOutputs();
                            if (m_in.size() == 2 && m_out.size() == 1) {
                                const std::string mo = m_out[0].GetName();
                                auto cit = consumers.find(mo);
                                const bool internal = cit != consumers.end() &&
                                                      cit->second.size() == 1 &&
                                                      graph_outputs.find(mo) == graph_outputs.end();
                                // exactly one const-init fp16 rank-1 [K] operand,
                                // the other dynamic.
                                int n_const = 0, n_dyn = 0;
                                for (const auto& mi : m_in) {
                                    if (mi.IsConstantInitializer()) {
                                        ++n_const;
                                        if (TensorElementTypeOf(mi) != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) { n_const = -100; break; }
                                        auto msh = GetShapeAllowSymbolic(mi);
                                        int64_t mnumel = 1;
                                        if (msh) for (int64_t d : *msh) mnumel *= d;
                                        // Per-channel along K: last dim == K and
                                        // numel == K (=> all leading dims are 1).
                                        // Accepts [K], [1,K]; rejects [K,1], [K+1].
                                        if (!msh || msh->empty() || mnumel != K || msh->back() != K) { n_const = -100; break; }
                                    } else {
                                        ++n_dyn;
                                    }
                                }
                                if (internal && n_const == 1 && n_dyn == 1) {
                                    foldable_mul = static_cast<const OrtNode*>(mul);
                                }
                            }
                        }
                    }
                }

                if (foldable_mul) {
                    const OrtNode* pair[2] = {foldable_mul, mm_node};
                    if (OrtStatus* st = self->ep_api_->EpGraphSupportInfo_AddNodesToFuse(
                            info, pair, 2, &fopts)) return st;
                } else {
                    if (OrtStatus* st = self->ep_api_->EpGraphSupportInfo_AddNodesToFuse(
                            info, &mm_node, 1, &fopts)) return st;
                }
                continue;
            }
            // ---- GroupQueryAttention (com.microsoft) — fused decode kernel ----
            // OPT-IN (ORT_FUSIONCPU_GQA=1). Claims only the exact Phi-3 packed-QKV
            // grow-and-swap configuration the fused kernel implements; anything
            // else falls back to the stock CPUExecutionProvider GQA op.
            if (op == "GroupQueryAttention" && dom == "com.microsoft") {
                auto ins  = node.GetInputs();
                auto outs = node.GetOutputs();
                // Require all 3 of {output, present_key, present_value}; reject
                // output_qk (4th output).
                if (outs.size() != 3) continue;
                bool out_absent = false;
                for (const auto& o : outs)
                    if (static_cast<const OrtValueInfo*>(o) == nullptr) { out_absent = true; break; }
                if (out_absent) continue;

                // Need indices 0..6 declared (query, [key], [value], past_key,
                // past_value, seqlens_k, total_sequence_length); cos/sin (7,8)
                // are required only when do_rotary.
                if (ins.size() < 7) continue;
                auto present = [&](size_t idx) {
                    return idx < ins.size() && static_cast<const OrtValueInfo*>(ins[idx]) != nullptr;
                };
                // QKV inputs: accept EITHER packed (key & value both absent, one
                // fused query tensor) OR unpacked (key & value both present, three
                // separate q/k/v tensors). A mixed state (exactly one present) is
                // not a valid GQA node -- skip it.
                const bool has_key = present(1), has_val = present(2);
                if (has_key != has_val) continue;
                // key/value fp32 when present (unpacked).
                if (has_key) {
                    if (TensorElementTypeOf(ins[1]) != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
                    if (TensorElementTypeOf(ins[2]) != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
                }
                // past_key / past_value / seqlens_k / total_sequence_length required.
                if (!present(3) || !present(4) || !present(5) || !present(6)) continue;
                // Reject any custom inputs 9..13 (position_ids, attention_bias,
                // head_sink, k_scale, v_scale).
                bool extra = false;
                for (size_t k = 9; k < ins.size(); ++k)
                    if (present(k)) { extra = true; break; }
                if (extra) continue;

                int64_t num_heads = 0, kv_num_heads = 0;
                if (!ReadInt64Attr(node, "num_heads", num_heads) || num_heads <= 0) continue;
                if (!ReadInt64Attr(node, "kv_num_heads", kv_num_heads) || kv_num_heads <= 0) continue;
                if (num_heads % kv_num_heads != 0) continue;

                int64_t local_window = -1, smooth = -1, qk_out = 0, kv_bits = 0;
                int64_t do_rotary = 0, rotary_interleaved = 0;
                float softcap = 0.0f;
                // local_window_size is ACCEPTED (not rejected): the kernel applies
                // real sliding-window masking in both attend paths (each query scores
                // only the most recent local_window_size keys), so sliding-window
                // models (e.g. Phi-3 window=2047) run correctly at any context length.
                ReadInt64Attr(node, "local_window_size", local_window);
                if (ReadInt64Attr(node, "smooth_softmax", smooth) && smooth == 1) continue;
                if (ReadInt64Attr(node, "qk_output", qk_out) && qk_out != 0) continue;
                if (ReadInt64Attr(node, "kv_cache_bit_width", kv_bits) && kv_bits != 0) continue;  // no KV quant
                if (ReadFloatAttr(node, "softcap", softcap) && softcap != 0.0f) continue;
                ReadInt64Attr(node, "do_rotary", do_rotary);
                ReadInt64Attr(node, "rotary_interleaved", rotary_interleaved);

                // fp32 Q and fp32 KV cache only.
                if (TensorElementTypeOf(ins[0]) != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
                if (TensorElementTypeOf(ins[3]) != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;

                // RoPE needs constant fp32 cos/sin caches.
                if (do_rotary) {
                    if (!present(7) || !present(8)) continue;
                    if (TensorElementTypeOf(ins[7]) != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
                    if (TensorElementTypeOf(ins[8]) != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
                    if (!ins[7].IsConstantInitializer() || !ins[8].IsConstantInitializer()) continue;
                }

                OrtNodeFusionOptions fopts{};
                fopts.ort_version_supported      = ORT_API_VERSION;
                fopts.drop_constant_initializers = false;  // keep cos/sin as Compute inputs
                const OrtNode* gqa_node = static_cast<const OrtNode*>(node);
                if (OrtStatus* st = self->ep_api_->EpGraphSupportInfo_AddNodesToFuse(
                        info, &gqa_node, 1, &fopts)) return st;
                continue;
            }

            // ---- extra ops (fusion_cpu_extra_ops.cpp) — norm / gather / mask ----
            // Any remaining claimable op (SkipSimplifiedLayerNormalization,
            // SimplifiedLayerNormalization, Gather, Shape, ReduceSum, Sub, Cast,
            // Constant). Kept in a separate TU so fusion_cpu_ep.cpp and the
            // MatMulNBits kernel stay intact; all run on the shared pool.
            ExtraTryClaim(*self, static_cast<const OrtNode*>(node), info);
        }
    } catch (const Ort::Exception& ex) {
        return Ort::Status(ex).release();
    } catch (const std::exception& ex) {
        return Ort::Status(ex.what(), ORT_EP_FAIL).release();
    }
    return nullptr;
}

/*static*/ OrtStatus* ORT_API_CALL FusionCpuEp::CompileImpl(
        OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** fused_nodes,
        size_t count, OrtNodeComputeInfo** node_compute_infos,
        OrtNode** /*ep_context_nodes*/) noexcept {
    auto* self = static_cast<FusionCpuEp*>(this_ptr);
    try {
        for (size_t i = 0; i < count; ++i) {
            Ort::ConstNode fused{fused_nodes[i]};
            std::string fname = fused.GetName();

            Ort::ConstGraph sub{graphs[i]};
            std::vector<Ort::ConstNode> sub_nodes = sub.GetNodes();

            // ---- single-node fused subgraphs (M1 Mul / M2a Sigmoid) ----
            if (sub_nodes.size() == 1) {
                const auto& inner = sub_nodes[0];
                const std::string op  = inner.GetOperatorType();
                const std::string dom = inner.GetDomain();

//                printf("CompileImpl: 1 node op - %s, dom - %s\n", op.c_str(), dom.c_str());

                if (op == "Mul" && dom.empty()) {
                    self->mul_kernels_[fname] = std::make_unique<FusionCpuMulKernel>(self->ort_api_);
                    auto nci = std::make_unique<FusionCpuNodeComputeInfo>(
                        *self, &LookupMulKernel, &ComputeMulKernel);
                    node_compute_infos[i] = nci.release();
                    continue;
                }
                if (op == "Sigmoid" && dom.empty()) {
                    self->sigmoid_kernels_[fname] = std::make_unique<FusionCpuSigmoidKernel>(self->ort_api_);
                    auto nci = std::make_unique<FusionCpuNodeComputeInfo>(
                        *self, &LookupSigmoidKernel, &ComputeSigmoidKernel);
                    node_compute_infos[i] = nci.release();
                    continue;
                }

                if (op == "MatMulNBits" && dom == "com.microsoft") {
                    int64_t bits = 4, block_size = 0;
                    if (!ReadInt64Attr(inner, "bits", bits)) bits = 4;
                    if (!ReadInt64Attr(inner, "block_size", block_size) || block_size <= 0) {
                        std::string msg = "FusionCpuEp: MatMulNBits '" + fname + "' missing block_size attribute";
                        return self->ort_api_->CreateStatus(ORT_EP_FAIL, msg.c_str());
                    }
                    // Standalone (no AWQ fold): fixed input order A,B,scales[,zp].
                    auto mm_in = inner.GetInputs();
                    const int idx_zp = (mm_in.size() >= 4 && !mm_in[3].GetName().empty()) ? 3 : -1;
                    self->matmulnbits_kernels_[fname] = std::make_unique<FusionCpuMatMulNBitsKernel>(
                        self->ort_api_, static_cast<int>(bits), static_cast<int>(block_size),
                        /*idx_a=*/0, /*idx_scale=*/-1, /*idx_b=*/1, /*idx_scales=*/2, idx_zp);
                    auto nci = std::make_unique<FusionCpuNodeComputeInfo>(
                        *self, &LookupMatMulNBitsKernel, &ComputeMatMulNBitsKernel);
                    node_compute_infos[i] = nci.release();
                    continue;
                }

                if (op == "GroupQueryAttention" && dom == "com.microsoft"){
                    int64_t num_heads = 0, kv_num_heads = 0, do_rotary = 0, rotary_interleaved = 0;
                    float scale = 0.0f;
                    if (!ReadInt64Attr(inner, "num_heads", num_heads) ||
                        !ReadInt64Attr(inner, "kv_num_heads", kv_num_heads)) {
                        std::string msg = "FusionCpuEp: GroupQueryAttention '" + fname +
                                          "' missing num_heads/kv_num_heads attribute";
                        return self->ort_api_->CreateStatus(ORT_EP_FAIL, msg.c_str());
                    }
                    ReadFloatAttr(inner, "scale", scale);
                    ReadInt64Attr(inner, "do_rotary", do_rotary);
                    ReadInt64Attr(inner, "rotary_interleaved", rotary_interleaved);
                    int64_t local_window = -1;
                    ReadInt64Attr(inner, "local_window_size", local_window);

                    // Resolve each role's tensor name from the inner node, then
                    // map to the fused-node input index by name (the fused
                    // boundary drops absent optional inputs, so indices shift).
                    auto gin = inner.GetInputs();
                    auto name_at = [&](size_t idx) -> std::string {
                        if (idx >= gin.size() || static_cast<const OrtValueInfo*>(gin[idx]) == nullptr)
                            return std::string();
                        return gin[idx].GetName();
                    };
                    const std::string qkv_name = name_at(0);
                    const std::string key_name = name_at(1);  // unpacked only
                    const std::string val_name = name_at(2);  // unpacked only
                    const std::string pk_name  = name_at(3);
                    const std::string pv_name  = name_at(4);
                    const std::string seq_name = name_at(5);  // seqlens_k (own-KV)
                    const std::string cos_name = do_rotary ? name_at(7) : std::string();
                    const std::string sin_name = do_rotary ? name_at(8) : std::string();

                    auto fin = fused.GetInputs();
                    auto find_idx = [&](const std::string& nm) -> int {
                        if (nm.empty()) return -1;
                        for (size_t k = 0; k < fin.size(); ++k)
                            if (fin[k].GetName() == nm) return static_cast<int>(k);
                        return -1;
                    };
                    const int idx_qkv = find_idx(qkv_name);
                    const int idx_key = find_idx(key_name);  // -1 when packed
                    const int idx_value = find_idx(val_name);  // -1 when packed
                    const int idx_pk  = find_idx(pk_name);
                    const int idx_pv  = find_idx(pv_name);
                    const int idx_seq = find_idx(seq_name);  // seqlens_k (own-KV)
                    const int idx_cos = find_idx(cos_name);
                    const int idx_sin = find_idx(sin_name);
                    // Unpacked requires BOTH key and value resolved; packed requires neither.
                    const bool unpacked = !key_name.empty();
                    const bool ok = idx_qkv >= 0 && idx_pk >= 0 && idx_pv >= 0 &&
                                    (!unpacked || (idx_key >= 0 && idx_value >= 0)) &&
                                    (!do_rotary || (idx_cos >= 0 && idx_sin >= 0));
                    if (!ok) {
                        std::string msg = "FusionCpuEp: GroupQueryAttention '" + fname +
                                          "' could not resolve required input indices";
                        return self->ort_api_->CreateStatus(ORT_EP_FAIL, msg.c_str());
                    }

                    // Map the 3 output roles by name too: the fused-node output
                    // order need not match the inner GQA op order (output 0,
                    // present_key 1, present_value 2), so a positional GetOutput
                    // would write the wrong shape/data to each tensor.
                    auto gout = inner.GetOutputs();
                    auto out_name_at = [&](size_t idx) -> std::string {
                        if (idx >= gout.size() || static_cast<const OrtValueInfo*>(gout[idx]) == nullptr)
                            return std::string();
                        return gout[idx].GetName();
                    };
                    auto fout = fused.GetOutputs();
                    auto find_out_idx = [&](const std::string& nm) -> int {
                        if (nm.empty()) return -1;
                        for (size_t k = 0; k < fout.size(); ++k)
                            if (fout[k].GetName() == nm) return static_cast<int>(k);
                        return -1;
                    };
                    const int idx_out = find_out_idx(out_name_at(0));
                    const int idx_pk_out = find_out_idx(out_name_at(1));
                    const int idx_pv_out = find_out_idx(out_name_at(2));
                    if (idx_out < 0 || idx_pk_out < 0 || idx_pv_out < 0) {
                        std::string msg = "FusionCpuEp: GroupQueryAttention '" + fname +
                                          "' could not resolve required output indices";
                        return self->ort_api_->CreateStatus(ORT_EP_FAIL, msg.c_str());
                    }
                    self->gqa_kernels_[fname] = std::make_unique<FusionCpuGqaKernel>(
                        self->ort_api_, static_cast<int>(num_heads), static_cast<int>(kv_num_heads),
                        scale, do_rotary != 0, rotary_interleaved != 0,
                        idx_qkv, idx_pk, idx_pv, idx_cos, idx_sin,
                        idx_out, idx_pk_out, idx_pv_out, idx_seq,
                        idx_key, idx_value, local_window);
                    auto nci = std::make_unique<FusionCpuNodeComputeInfo>(
                        *self, &LookupGqaKernel, &ComputeGqaKernel);
                    node_compute_infos[i] = nci.release();
                    continue;
                }

                // ---- extra ops (fusion_cpu_extra_ops.cpp): single-node norm /
                // gather / mask kernels. The extra TU builds the kernel and hands
                // back the shared dispatch pair; we wrap it in the same
                // FusionCpuNodeComputeInfo trampoline (and free it identically via
                // ReleaseNodeComputeInfosImpl).
                {
                    ExtraLookupFn xlookup = nullptr;
                    ExtraComputeFn xcompute = nullptr;
                    if (ExtraTryCompile(*self, static_cast<const OrtNode*>(fused),
                                        static_cast<const OrtNode*>(inner),
                                        &xlookup, &xcompute)) {
                        auto nci = std::make_unique<FusionCpuNodeComputeInfo>(
                            *self, xlookup, xcompute);
                        node_compute_infos[i] = nci.release();
                        continue;
                    }
                }

//                printf("FusionCpuEp: unsupported op - %s\n", op.c_str());

                std::string msg = "FusionCpuEp: unsupported op '" + op + "' (domain '" + dom +
                                  "') in fused node '" + fname + "'";
                return self->ort_api_->CreateStatus(ORT_EP_FAIL, msg.c_str());
            }

            // ---- 2-node AWQ fold {Mul, MatMulNBits} (M3+) ----
            // Fold the *_weight_only_mul per-channel activation scale into the
            // int4 kernel. Identify the Mul/MatMulNBits by op, verify the Mul
            // feeds MatMulNBits' A, classify the Mul's const operand as the AWQ
            // scale, and resolve every role to a fused-node input index by name.
            if (sub_nodes.size() == 2) {
                const Ort::ConstNode* mul = nullptr;
                const Ort::ConstNode* mm = nullptr;
                for (const auto& n : sub_nodes) {
                    const std::string o = n.GetOperatorType();
                    const std::string d = n.GetDomain();

//                    printf("CompileImpl: 2-node op - %s, dom - %s\n", o.c_str(), d.c_str());

                    if (o == "Mul" && d.empty()) mul = &n;
                    else if (o == "MatMulNBits" && d == "com.microsoft") mm = &n;
                }
                if (mul && mm) {
                    auto mm_in   = mm->GetInputs();
                    auto mul_out = mul->GetOutputs();
                    auto mul_in  = mul->GetInputs();
                    int64_t bits = 4, block_size = 0;
                    if (!ReadInt64Attr(*mm, "bits", bits)) bits = 4;
                    const bool attrs_ok = ReadInt64Attr(*mm, "block_size", block_size) && block_size > 0;
                    if (attrs_ok && mm_in.size() >= 3 && mul_out.size() == 1 && mul_in.size() == 2 &&
                        mm_in[0].GetName() == mul_out[0].GetName()) {
                        // Classify the Mul operands: const initializer = AWQ
                        // scale, dynamic = activation A.
                        std::string a_name, scale_name;
                        for (const auto& mi : mul_in) {
                            if (mi.IsConstantInitializer()) scale_name = mi.GetName();
                            else a_name = mi.GetName();
                        }
                        const std::string b_name = mm_in[1].GetName();
                        const std::string scales_name = mm_in[2].GetName();
                        const bool has_zp = (mm_in.size() >= 4 && !mm_in[3].GetName().empty());
                        const std::string zp_name = has_zp ? mm_in[3].GetName() : std::string();

                        auto fin = fused.GetInputs();
                        auto find_idx = [&](const std::string& nm) -> int {
                            if (nm.empty()) return -1;
                            for (size_t k = 0; k < fin.size(); ++k)
                                if (fin[k].GetName() == nm) return static_cast<int>(k);
                            return -1;
                        };
                        const int idx_a = find_idx(a_name);
                        const int idx_scale = find_idx(scale_name);
                        const int idx_b = find_idx(b_name);
                        const int idx_scales = find_idx(scales_name);
                        const int idx_zp = has_zp ? find_idx(zp_name) : -1;

                        const bool ok = idx_a >= 0 && idx_scale >= 0 && idx_b >= 0 &&
                                        idx_scales >= 0 && (!has_zp || idx_zp >= 0);
                        if (ok) {
                            self->matmulnbits_kernels_[fname] = std::make_unique<FusionCpuMatMulNBitsKernel>(
                                self->ort_api_, static_cast<int>(bits), static_cast<int>(block_size),
                                idx_a, idx_scale, idx_b, idx_scales, idx_zp);
                            auto nci = std::make_unique<FusionCpuNodeComputeInfo>(
                                *self, &LookupMatMulNBitsKernel, &ComputeMatMulNBitsKernel);
                            node_compute_infos[i] = nci.release();
                            continue;
                        }
                    }
                }
                std::string msg = "FusionCpuEp: 2-node fused subgraph '" + fname +
                                  "' did not match the {Mul, MatMulNBits} AWQ-fold topology";
                return self->ort_api_->CreateStatus(ORT_EP_FAIL, msg.c_str());
            }

            // ---- 3-node SwiGLU activation triple (M2b) ----
            // The claim guarantees the group is {Sigmoid, Mul, Mul} with the
            // fused node exposing exactly two inputs: g (the Sigmoid operand) and
            // u (the up projection). Resolve idx_g/idx_u by name from the Sigmoid
            // input; this is association-agnostic (works for both SwiGLU graph
            // topologies A and B).
            if (sub_nodes.size() == 3) {
                const Ort::ConstNode* sig = nullptr;
                size_t n_mul = 0;
                for (const auto& n : sub_nodes) {
                    const std::string o = n.GetOperatorType();
                    const std::string d = n.GetDomain();
                    if (!d.empty()) { sig = nullptr; n_mul = 3; break; }  // force the error below
                    if (o == "Sigmoid") sig = &n;
                    else if (o == "Mul" && n_mul < 2) ++n_mul;
                    else { sig = nullptr; n_mul = 3; break; }
                }
                if (sig && n_mul == 2) {
                    const std::string g_name = sig->GetInputs()[0].GetName();
                    auto fin = fused.GetInputs();
                    size_t idx_g = SIZE_MAX, idx_u = SIZE_MAX;
                    if (fin.size() == 2 && !g_name.empty()) {
                        for (size_t k = 0; k < fin.size(); ++k) {
                            const std::string nm = fin[k].GetName();
                            if (nm == g_name) idx_g = k;
                            else               idx_u = k;
                        }
                    }
                    if (idx_g != SIZE_MAX && idx_u != SIZE_MAX && idx_g != idx_u) {
                        self->swiglu_kernels_[fname] =
                            std::make_unique<FusionCpuSwiGluKernel>(self->ort_api_, idx_g, idx_u);
                        auto nci = std::make_unique<FusionCpuNodeComputeInfo>(
                            *self, &LookupSwiGluKernel, &ComputeSwiGluKernel);
                        node_compute_infos[i] = nci.release();
                        continue;
                    }
                }
                std::string msg = "FusionCpuEp: 3-node fused subgraph '" + fname +
                                  "' did not match the expected SwiGLU {Sigmoid, Mul, Mul} topology";
                return self->ort_api_->CreateStatus(ORT_EP_FAIL, msg.c_str());
            }

            std::string msg = "FusionCpuEp: unexpected fused subgraph size "
                              + std::to_string(sub_nodes.size()) + " in '" + fname + "'";
            return self->ort_api_->CreateStatus(ORT_EP_FAIL, msg.c_str());
        }
    } catch (const Ort::Exception& ex) {
        return Ort::Status(ex).release();
    } catch (const std::exception& ex) {
        return Ort::Status(ex.what(), ORT_EP_FAIL).release();
    }
    return nullptr;
}

/*static*/ void ORT_API_CALL FusionCpuEp::ReleaseNodeComputeInfosImpl(
        OrtEp* /*this_ptr*/, OrtNodeComputeInfo** infos, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        delete static_cast<FusionCpuNodeComputeInfo*>(infos[i]);
    }
}

/*static*/ OrtStatus* ORT_API_CALL FusionCpuEp::SyncImpl(OrtEp* /*this_ptr*/) noexcept {
    // CPU kernels run synchronously inside Compute — nothing to drain.
    return nullptr;
}

}  // namespace fusion_cpu_ep
