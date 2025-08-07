namespace ggml_b612 {

#if defined(_WIN32)

//
// This file contains common affinity, core parking, cache enumeration code. 
//

#include <powrprof.h> // UNDONE: powerprof.lib

void xb_disable_core_parking(void)
{
#if WIN32_POWERPROF // UNDONE: requires powerprof.lib

    GUID* activeScheme = NULL;
    DWORD acValue = 100; // Set to 100% to disable core parking
#if 0
    DWORD dcValue = 100; // Set to 100% to disable core parking
#endif // #if 0

    //
    // Retrieve the active power scheme.
    //

    if (PowerGetActiveScheme(NULL, &activeScheme) != ERROR_SUCCESS) {
        printf("Failed to get the active power scheme.\n");
        return;
    }

    //
    // Processor Performance Core Parking Minimum Cores (AC).
    //

    GUID SUB_PROCESSOR = GUID_PROCESSOR_SETTINGS_SUBGROUP;
    GUID CORE_PARK_MIN_CORES = GUID_PROCESSOR_CORE_PARKING_MIN_CORES;

    if (PowerWriteACValueIndex(NULL, activeScheme, &SUB_PROCESSOR, &CORE_PARK_MIN_CORES, acValue) != ERROR_SUCCESS) {
        printf("Failed to set core parking minimum cores (AC).\n");
        return;
    }

#if 0
    //
    // Processor Performance Core Parking Minimum Cores (DC).
    //

    if (PowerWriteDCValueIndex(NULL, activeScheme, &SUB_PROCESSOR, &CORE_PARK_MIN_CORES, dcValue) != ERROR_SUCCESS) {
        printf("Failed to set core parking minimum cores (DC).\n");
        return;
    }
#endif // #if 0

    //
    // Apply the updated settings.
    //

    if (PowerSetActiveScheme(NULL, activeScheme) != ERROR_SUCCESS) {
        printf("Failed to apply the power scheme.\n");
        return;
    }

    //
    // Clean up.
    //

    LocalFree(activeScheme);

#endif // #if WIN32_POWERPROF

    printf("Core parking disabled successfully.\n");
}

class CPUInfo {
	class CPUID {
		uint32_t regs[4] = { 0 };

	public:
		inline explicit CPUID(uint32_t funcId, uint32_t subFuncId) {
			::__cpuidex((int*)regs, (int)funcId, (int)subFuncId);
		}

		inline const uint32_t& EBX() const { return regs[1]; }
		inline const uint32_t& EAX() const { return regs[0]; }
		inline const uint32_t& ECX() const { return regs[2]; }
		inline const uint32_t& EDX() const { return regs[3]; }
	};

public:
	inline CPUInfo();
	inline std::string vendor()        const { return mVendorId; }
	inline std::string model()         const { return mModelName; }
	inline int     cores()             const { return mNumCores; }
	// WARNING! CPUID reports hardware CAPABILITIES. For Intel CPUs you will still get HT=on and logicalCpus() > cores() even if HT is OFF in the BIOS.
	// Query the OS for actual correct runtime info.
	inline int     logicalCpus()       const { return mNumLogCpus; }
	inline bool    isHyperThreaded()   const { return mIsHTT; }
	inline bool    haveSSE()           const { return mIsSSE; }
	inline bool    haveSSE2()          const { return mIsSSE2; }
	inline bool    haveSSE3()          const { return mIsSSE3; }
	inline bool    haveSSE41()         const { return mIsSSE41; }
	inline bool    haveSSE42()         const { return mIsSSE42; }
	inline bool    haveAVX()           const { return mIsAVX; }
	inline bool    haveAVX2()          const { return mIsAVX2; }
	inline bool    haveAVX512F()       const { return mIsAVX512F; }

private:
	// Bit positions for data extractions
	static constexpr uint32_t SSE_POS = 0x02000000;
	static constexpr uint32_t SSE2_POS = 0x04000000;
	static constexpr uint32_t SSE3_POS = 0x00000001;
	static constexpr uint32_t SSE41_POS = 0x00080000;
	static constexpr uint32_t SSE42_POS = 0x00100000;
	static constexpr uint32_t AVX_POS = 0x10000000;
	static constexpr uint32_t AVX2_POS = 0x00000020;
	static constexpr uint32_t AVX512F_POS = 1u << 15; // Bit 16
	static constexpr uint32_t LVL_NUM = 0x000000FF;
	static constexpr uint32_t LVL_TYPE = 0x0000FF00;
	static constexpr uint32_t LVL_CORES = 0x0000FFFF;

	// Attributes
	std::string mVendorId;
	std::string mModelName;
	int    mNumSMT = 0;
	int    mNumCores = 0;
	int    mNumLogCpus = 0;
	bool   mIsHTT = 0;
	bool   mIsSSE = false;
	bool   mIsSSE2 = false;
	bool   mIsSSE3 = false;
	bool   mIsSSE41 = false;
	bool   mIsSSE42 = false;
	bool   mIsAVX = false;
	bool   mIsAVX2 = false;
	bool   mIsAVX512F = false;
};

CPUInfo::CPUInfo() {
	// Get vendor name EAX=0
	CPUID cpuID0(0, 0);
	const uint32_t HFS = cpuID0.EAX();
	// Reinterpret bytes as ASCII characters
	mVendorId += std::string((const char*)&cpuID0.EBX(), 4);
	mVendorId += std::string((const char*)&cpuID0.EDX(), 4);
	mVendorId += std::string((const char*)&cpuID0.ECX(), 4);
	// Get SSE instructions availability
	CPUID cpuID1(1, 0);
	mIsHTT = cpuID1.EDX() & AVX_POS;
	mIsSSE = cpuID1.EDX() & SSE_POS;
	mIsSSE2 = cpuID1.EDX() & SSE2_POS;
	mIsSSE3 = cpuID1.ECX() & SSE3_POS;
	mIsSSE41 = cpuID1.ECX() & SSE41_POS;
	mIsSSE42 = cpuID1.ECX() & SSE41_POS;
	mIsAVX = cpuID1.ECX() & AVX_POS;
	// Get AVX2 instructions availability
	CPUID cpuID7(7, 0);
	mIsAVX2 = cpuID7.EBX() & AVX2_POS;
	mIsAVX512F = cpuID7.EBX() & AVX512F_POS;

	std::string vendorIdUppercase = mVendorId;
	std::for_each(vendorIdUppercase.begin(), vendorIdUppercase.end(), [](char& character) { character = static_cast<char>(::toupper(character)); });
	// Get num of cores
	if (vendorIdUppercase.find("INTEL") != std::string::npos) {
		if (HFS >= 11) {
			static constexpr int MAX_INTEL_TOP_LVL = 4;
			for (int lvl = 0; lvl < MAX_INTEL_TOP_LVL; ++lvl) {
				CPUID cpuID4(0x0B, lvl);
				uint32_t currLevel = (LVL_TYPE & cpuID4.ECX()) >> 8;
				switch (currLevel) {
				    case 0x01: mNumSMT = LVL_CORES & cpuID4.EBX(); break; //  EAX=0xB, ECX=0 - EBX is the number of logical processors (threads) per core
				    case 0x02: mNumLogCpus = LVL_CORES & cpuID4.EBX(); break; // EAX=0xB, ECX=1 - EBX is the number of logical processors per processor package
				    default: break;
				}
			}
			mNumCores = mNumLogCpus / mNumSMT;
			mIsHTT = mNumSMT > 1;
		}
		else
		{
			if (HFS >= 1) {
				mNumLogCpus = (cpuID1.EBX() >> 16) & 0xFF;
				if (HFS >= 4) {
					mNumCores = 1 + (CPUID(4, 0).EAX() >> 26) & 0x3F;
				}
			}
			if (mIsHTT)	{
				if (!(mNumCores > 1)) {
					mNumCores = 1;
					mNumLogCpus = (mNumLogCpus >= 2 ? mNumLogCpus : 2);
				}
			} else {
				mNumCores = mNumLogCpus = 1;
			}
		}
	}
	else if (vendorIdUppercase.find("AMD") != std::string::npos) {
        CPUID cpuID1(1, 0);
        mNumLogCpus = (cpuID1.EBX() & 0xff0000) >> 16;
        CPUID cpuID8000001E(0x8000001E, 0);
		mNumSMT = ((cpuID8000001E.EBX() & 0x300) >> 8) + 1;
        // printf("%s: AMD mNumSMT %d\n", __func__, mNumSMT);
		if ((mNumLogCpus > 0) && (mNumSMT > 0)) {
			mNumCores = mNumLogCpus / mNumSMT;
		}
		else {
			if (HFS >= 1) {
				if (CPUID(0x80000000, 0).EAX() >= 8) {
					mNumCores = 1 + (CPUID(0x80000008, 0).ECX() & 0xFF);
				}
			}
			if (mIsHTT) {
				if (mNumCores < 1) {
					mNumCores = 1;
				}
			}
			else {
				mNumCores = 1;
			}
		}
	} else {
		throw std::runtime_error{"Unknown vendor! Reported vendor name is: " + mVendorId};
	}

	// Get processor brand string
	// This seems to be working for both Intel & AMD vendors
	for (int i = 0x80000002; i < 0x80000005; ++i) {
		CPUID cpuID(i, 0);
		mModelName += std::string((const char*)&cpuID.EAX(), 4);
		mModelName += std::string((const char*)&cpuID.EBX(), 4);
		mModelName += std::string((const char*)&cpuID.ECX(), 4);
		mModelName += std::string((const char*)&cpuID.EDX(), 4);
	}
}

uint64_t l1d_cache_size = 48ull * 1024ull;
uint64_t l1i_cache_size = 32ull * 1024ull;
uint64_t l2_cache_size = 1024ull * 1024ull;
uint64_t l3_cache_size = 1024ull * 1024ull;

typedef struct {
    uint64_t mask;
    uint16_t group;
    uint16_t reserved[3];
} group_affinity_t;

ULONG master_index = 0;
uint32_t maximum_logical = 0;

bool
xb_set_thread_affinity (
    uint32_t ith,
    uint64_t * affinity
    )

{

    //
    // Set the affinity of the current thread if the maximum number of logical
    // processors is less than or equal to 64, i.e., one affinity group.
    //

    if (maximum_logical <= 64) {
        uint32_t index = master_index + (2 * ith);

        if (SetThreadAffinityMask(GetCurrentThread(), 1ull << index)) {
            *affinity = SetThreadAffinityMask(GetCurrentThread(), 1ull << index);
            return true;

        } else {
            printf("failed to set thread affinity\n");
        }
    }

    return false;
}

char * ggml_cache_type[4] = {
    "null",
    "data",
    "instruction",
    "unified"
};

uint64_t
xb_set_process_affinity (
    uint32_t n_threads,
    uint64_t affinity_mask_requested,
    bool verbose = false
    )
{

#if 0 // Require powerprof.lib

    //
    // Disable core parking.
    //

    ggml_disable_core_parking();

#endif

    uint64_t affinity_mask = affinity_mask_requested;

    if (affinity_mask_requested != 0) {
        goto set_affinity;
    }

    //
    // Get the default rounding mode.
    //

    char * default_mode = "none";

    uint32_t mxcsr = _mm_getcsr();

    uint32_t round_mode = mxcsr & _MM_ROUND_MASK;

    switch (round_mode) {
    case _MM_ROUND_NEAREST:
        default_mode = "round nearest";
        break;

    case _MM_ROUND_DOWN:
        default_mode = "round_down";
        break;

    case _MM_ROUND_UP:
        default_mode = "round_up";
        break;

    case _MM_ROUND_TOWARD_ZERO:
        default_mode = "round_toward_zero";
        break;

    }

    if (verbose) {
        printf("mxcsr 0x%08x, default rounding mode - %s\n", mxcsr, default_mode);
    }

    //
    // Get cpuid information.
    //

    struct {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
    } cpu_info;

    if (verbose) {
        //
        // Get avx features for the current system.
        //

        printf("host system AVX capabilities:\n");
        __cpuid((int *)&cpu_info, 0x00000001);
        printf("  cpuid function 0x00000001\n");  
        if (cpu_info.ecx & (1 << 28)) {
            printf("    Avx\n");
        }

        __cpuidex((int *)&cpu_info, 0x00000007, 0);
        printf("  cpuidex function 0x00000007, subleaf 0\n");
        if (cpu_info.ebx & (1 << 5)) {
            printf("    Avx2\n");
        }

        if (cpu_info.ebx & (1 << 16)) {
            printf("    Avx512F\n");
        }

        if (cpu_info.ebx & (1 << 17)) {
            printf("    Avx512DQ\n");
        }

        if (cpu_info.ebx & (1 << 21)) {
            printf("    Avx512Ifma\n");
        }

        if (cpu_info.ebx & (1 << 27)) {
            printf("    Avx512CD\n");
        }

        if (cpu_info.ebx & (1 << 29)) {
            printf("    Avx512BW\n");
        }

        if (cpu_info.ebx & (1 << 30)) {
            printf("    Avx512VL\n");
        }

        if (cpu_info.ecx & (1 << 1)) {
            printf("    Avx512Vbmi\n");
        }

        if (cpu_info.ecx & (1 << 6)) {
            printf("    Avx512Vbmi2\n");
        }

        if (cpu_info.ecx & (1 << 11)) {
            printf("    Avx512Vnni\n");
        }

        if (cpu_info.ecx & (1 << 12)) {
            printf("    Avx512Bitalg\n");
        }

        if (cpu_info.ecx & (1 << 14)) {
            printf("    Avx512Vpopcntdq\n");
        }

        if (cpu_info.edx & (1 << 8)) {
            printf("    Avx512Vp2Intersect\n");
        }

        if (cpu_info.edx & (1 << 23)) {
            printf("    Avx512FP16\n");
        }

        __cpuidex((int *)&cpu_info, 0x00000007, 1);
        printf("  cpuidex function 0x00000007, subleaf 1\n");
        if (cpu_info.eax & (1 << 4)) {
            printf("    AvxVnni\n");
        }

        if (cpu_info.eax & (1 << 5)) {
            printf("    Avx512Bfloat16\n");
        }

        if (cpu_info.eax & (1 << 23)) {
            printf("    AvxIfma\n");
        }

        if (cpu_info.edx & (1 << 4)) {
            printf("    AvxVnniInt8\n");
        }

        if (cpu_info.edx & (1 << 5)) {
            printf("    AvxNeConvert\n");
        }

        if (cpu_info.edx & (1 << 9)) {
            printf("    AvxVnniInt16\n");
        }

        if (cpu_info.edx & (1 << 19)) {
            printf("    Avx10\n");
        }

        printf("\n");
    }

    //
    // Get L1 instruction and data cache attributes.
    //
    __cpuid((int *)&cpu_info, 0x80000005);
    if (verbose) {
        printf("l1 d-cache line size %d\n", cpu_info.ecx & 0xff);
        printf("l1 d-cache lines per tag %d\n", (cpu_info.ecx >> 8) & 0xff);
        printf("l1 d-cache associativity %d\n", (cpu_info.ecx >> 16) & 0xff);
    }
    l1d_cache_size = ((cpu_info.ecx >> 24) & 0xff) * 1024ull;
    if (verbose) {
        printf("l1 i-cache line size %d\n", cpu_info.edx & 0xff);
        printf("l1 i-cache lines per tag %d\n", (cpu_info.edx >> 8) & 0xff);
        printf("l1 i-cache associativity %d\n", (cpu_info.edx >> 16) & 0xff);
    }
    l1i_cache_size = ((cpu_info.edx >> 24) & 0xff) * 1024ull;
    __cpuidex((int *)&cpu_info, 0x8000001d, 0);
    const int32_t l1d_cache_type = cpu_info.eax & 0x3;
    const int32_t l1d_cache_sharing = ((cpu_info.eax >> 14) & 0xfff) + 1;
    __cpuidex((int *)&cpu_info, 0x8000001d, 1);
    const int32_t l1i_cache_type = cpu_info.eax & 0x3;
    const int32_t l1i_cache_sharing = ((cpu_info.eax >> 14) & 0xfff) + 1;
    if (verbose) {
        printf("l1 d-cache size %zdkb, type - %s, SMT sharing %d\n",
               l1d_cache_size / 1024,
               ggml_cache_type[l1d_cache_type],
               l1d_cache_sharing);
        printf("l1 i-cache size %zdkb, type - %s, SMT sharing %d\n",
               l1i_cache_size / 1024,
               ggml_cache_type[l1i_cache_type],
               l1i_cache_sharing);
    }
    //
    // Get l2 cache information.
    //
    __cpuid((int *)&cpu_info, 0x80000006);
    l2_cache_size = ((cpu_info.ecx >> 16) & 0xffff) * 1024ull;
    __cpuidex((int *)&cpu_info, 0x8000001d, 2);
    const int32_t l2_cache_type = cpu_info.eax & 0x3;
    const int32_t l2_cache_sharing = ((cpu_info.eax >> 14) & 0xfff) + 1;
    if (verbose) {
        printf("l2 cache size %zdkb, type - %s. SMT sharing %d\n",
               l2_cache_size / 1024,
               ggml_cache_type[l2_cache_type],
               l2_cache_sharing);
    }
    //
    // Get l3 cache information.
    //
    __cpuidex((int *)&cpu_info, 0x8000001d, 3);
    uint32_t line_size = (cpu_info.ebx & 0xfff) + 1;
    uint32_t partitions = ((cpu_info.ebx >> 12) & 0x3ff) + 1;
    uint32_t associativity = ((cpu_info.ebx >> 22) & 0x3ff) + 1;
    uint32_t sets = cpu_info.ecx + 1;
    if (verbose) {
        printf("l3 line size %d\n", line_size);
        printf("l3 partitions %d\n", partitions);
        printf("l3 associativity %d\n", associativity);
        printf("l3 sets %d\n", sets);
    }
    l3_cache_size = line_size * partitions * associativity * sets;
    const int32_t l3_cache_type = cpu_info.eax & 0x3;
    const int32_t l3_cache_sharing = ((cpu_info.eax >> 14) & 0xfff) + 1;
    if (verbose) {
        printf("l3 cache size %zdmb, type - %s, SMT sharing %d\n\n",
               l3_cache_size / (1024 * 1024),
               ggml_cache_type[l3_cache_type],
               l3_cache_sharing);
    }

    //
    // Get logical processors per core and maximum logical processsors.
    //

    __cpuid((int *)&cpu_info, 0x8000001e);
    const uint32_t logical_per_core = ((cpu_info.ebx & 0x300) >> 8) + 1;

    __cpuid((int *)&cpu_info, 0x00000001);
    maximum_logical = (cpu_info.ebx & 0xff0000) >> 16;

    //
    // Compute the total l1, l2, and l3 cache.
    //

    if (verbose) {
        float total_l1_cache = (float)(l1d_cache_size + l1i_cache_size);
        total_l1_cache *= (float)(maximum_logical / l1d_cache_sharing);
        total_l1_cache /= (1024.f * 1024.f);
        printf("total l1 cache %6.1fmb\n", total_l1_cache);

        float total_l2_cache = (float)(l2_cache_size);
        total_l2_cache *= (float)(maximum_logical / l2_cache_sharing);
        total_l2_cache /= (1024.f * 1024.f);
        printf("total l2 cache %6.1fmb\n", total_l2_cache);

        float total_l3_cache = (float)(l3_cache_size);
        total_l3_cache *= (float)(maximum_logical / l3_cache_sharing);
        total_l3_cache /= (1024.f * 1024.f);
        printf("total l3 cache %6.1fmb\n\n", total_l3_cache);

        printf("n_threads specified %d\n", n_threads);
        printf("logical processors per core %d\n", logical_per_core);
        printf("maximum logical processors %d\n", maximum_logical);
    }

    //
    // Check the logical processors per core.
    //

    if (logical_per_core == 1) {
        if (verbose) {
            printf("bypassing set process affinity - not SMT system\n");
        }
        return 0;
    }

    //
    // Check the specified number of threads against the maximum logical processor count.
    //

    const uint32_t maximum_smt_threads = maximum_logical / 2;
    if ((n_threads & 1) || (n_threads > maximum_smt_threads)) {
        if (verbose) {
            printf("bypassing set process affinity - number threads odd or gt maximum logical / 2\n");
        }
        return 0;
    }

    //
    // Get the current process group count.
    //

#if 0
    uint16_t group_array[4];
    uint16_t group_count = 4;

    if (GetProcessGroupAffinity(GetCurrentProcess(), &group_count, group_array)) {
        printf("GetProcessGroupAffinity succeeded with %d groups\n", group_count);
        if (group_count != 1) {
            printf("bypassing set affinity process because group count is greater than one\n");
            return 0;
        }

    } else {
        printf("GetProcessGroupAffinity failed\n");
        return 0;
    }
#endif // #if 0

    //
    // Set process affinity.
    //

    affinity_mask = ((1ull << (n_threads * 2)) - 1) & 0x55555555ull;

    //
    // It is known that the number of threads fits within the maximum smt set. If the
    // maximum smt set is less than or equal to 32, then the threads can be pushed
    // up to higher numbered threads which will remove them from contention issues
    // with clock and device interrupts.
    //

    if (maximum_smt_threads <= 32) {

        //
        // Compute the shift up such that the thread affinity straddles CCDs.
        //

        uint32_t half_shift = maximum_logical - (n_threads * 2);

        half_shift = ((half_shift / 2) + 1) & 0x1e;

        affinity_mask <<= half_shift;
    }

    set_affinity:
    if (SetProcessAffinityMask(GetCurrentProcess(), affinity_mask)) {
        if (verbose) {
           printf("process group affinity set to 0x%016llx\n", affinity_mask);
        }

        //
        // Compute the processor index of the master thread.
        //

        BitScanForward64(&master_index, affinity_mask);
        if (verbose) {
            printf("processor index of master thread %lu\n", master_index);
        }

#if 0
        //
        // Attempt to set the affinity of the master thread.
        //

        uint64_t master_affinity;

        if (xb_set_thread_affinity(0, &master_affinity)) {
            printf("master thread affinity set to 0x%016llx\n", master_affinity);
        }
#endif // #if 0


    } else {
        printf("failed to set process affinity mask\n");
    }

    return affinity_mask;
}

uint64_t
xb_set_optimal_process_affinity(uint32_t n_threads, bool verbose = false) {
    //
    // This routine selects the most optimal affinity mask based
    // on the processor it recognizes. Otherwise it just calls
    // xb_set_process_affinity() to figure out the default config.
    //

    bool AMD_Ryzen_HX_370 = false;
    bool AMD_Ryzen_PRO_395 = false;
    bool AMD_Ryzen_7_350 = false;
    bool AMD_Ryzen_9_9950X = false;
    ggml_b612::CPUInfo cpuInfo;

    if (verbose) {
        std::cout << "CPU vendor = " << cpuInfo.vendor() << std::endl;
        std::cout << "CPU Brand String = " << cpuInfo.model() << std::endl;
        std::cout << "# of cores = " << cpuInfo.cores() << std::endl;
        std::cout << "# of logical cores = " << cpuInfo.logicalCpus() << std::endl;
        std::cout << "Is CPU Hyper threaded = " << cpuInfo.isHyperThreaded() << std::endl;
    }

    if (cpuInfo.vendor().find("AMD") != std::string::npos) {
        if (verbose) {
            printf("%s: Detected [%s]\n", __func__, cpuInfo.model().c_str());
        }
        if (cpuInfo.model().find("AMD Ryzen AI 9 HX 370") != std::string::npos) {
            AMD_Ryzen_HX_370 = true;
        } else if (cpuInfo.model().find("AMD RYZEN AI MAX+ PRO 395") != std::string::npos) {
            AMD_Ryzen_PRO_395 = true;
        } else if (cpuInfo.model().find("AMD Ryzen AI 7 350") != std::string::npos) {
            AMD_Ryzen_7_350 = true;
        } else if (cpuInfo.model().find("AMD Ryzen 9 9950X") != std::string::npos) {
            AMD_Ryzen_9_9950X = true;
        }
    }

    // setup affinity mask for all threads
    int64_t affinity_mask = 0;        
    // For uneven systems leverage the Classic cores 
    // if possible. On systems with 16 cores (32 LP)
    // then use the cores crossing the two CCDs (yes!)
    switch (n_threads) {
        case 2:
            if (AMD_Ryzen_HX_370 ||
                AMD_Ryzen_7_350) {
                // use perf cores as available
                affinity_mask = 0x00000Aul;
            } else if (AMD_Ryzen_PRO_395 ||
                       AMD_Ryzen_9_9950X) {
                // use the middle cores spannning across the CPU
                affinity_mask = 0x00018000uL;
            }
            break;
        case 4: 
            if (AMD_Ryzen_HX_370 ||
                AMD_Ryzen_7_350) {
                // use perf cores as available
                affinity_mask = 0x0000AAul;
            } else if (AMD_Ryzen_PRO_395 ||
                       AMD_Ryzen_9_9950X) {
                // use the middle cores spannning across the CPU
                affinity_mask = 0x000AA000uL;
            }
            break;
        case 6: 
            if (AMD_Ryzen_HX_370 ||
                AMD_Ryzen_7_350) {
                // use perf cores as available
                affinity_mask = 0x000AAAul;
            } else if (AMD_Ryzen_PRO_395 ||
                       AMD_Ryzen_9_9950X) {
                // use the middle cores spannning across the CPU
                affinity_mask = 0x004AA400uL;
            }
            break;
        case 8: 
            if (AMD_Ryzen_HX_370 ||
                AMD_Ryzen_7_350) {
                // use perf cores as available
                affinity_mask = 0x00AAAAul;
            } else if (AMD_Ryzen_PRO_395 ||
                       AMD_Ryzen_9_9950X) {
                // use the middle cores spannning across the CPU
                affinity_mask = 0x00AAAA00uL;
            }
            break;
        case 10: 
            if (AMD_Ryzen_HX_370) {
                // use perf cores as available
                affinity_mask = 0x0AAAAAul;
            } else if (AMD_Ryzen_PRO_395 ||
                       AMD_Ryzen_9_9950X) {
                // use the middle cores spannning across the CPU
                affinity_mask = 0x02AAAA80uL;
            }
            break;
        case 12: 
            if (AMD_Ryzen_HX_370) {
                // use perf cores as available
                affinity_mask = 0xAAAAAAul;
            } else if (AMD_Ryzen_PRO_395 ||
                       AMD_Ryzen_9_9950X) {
                // use the middle cores spannning across the CPU
                affinity_mask = 0x0AAAAAA0uL;
            }
            break;
        case 16: 
            if (AMD_Ryzen_PRO_395 ||
                AMD_Ryzen_9_9950X) {
                affinity_mask = 0xAAAAAAAAuL;
            }
            break;
        default: 
            break;
    }

    affinity_mask = xb_set_process_affinity(n_threads, affinity_mask);
    return(affinity_mask);
}

#else

uint64_t
xb_set_optimal_process_affinity(uint32_t n_threads, bool verbose = false) {
    return 0;
}

#endif // _WIN32

} // namespace ggml_b612
