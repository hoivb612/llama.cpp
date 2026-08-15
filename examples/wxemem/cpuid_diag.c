// cpuid_diag.c - AVX-512 / AVX512-FP16 CPUID diagnostic.
// Build (MSVC):  cl /nologo cpuid_diag.c
// Run:           cpuid_diag.exe
#include <stdio.h>
#include <string.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

static void brand(char* out /*49 bytes*/) {
    int r[4]; unsigned i; char* p = out;
    for (i = 0x80000002u; i <= 0x80000004u; ++i) {
        __cpuidex(r, (int)i, 0);
        memcpy(p, r, 16); p += 16;
    }
    out[48] = 0;
}

#define BIT(x, n) (((x) >> (n)) & 1u)

int main(void) {
    int r[4];
    char b[49];
    unsigned int maxleaf, l1c, l1d, l7b, l7c, l7d, l7a1;
    unsigned __int64 xcr0 = 0;

    __cpuid(r, 0);
    maxleaf = (unsigned)r[0];
    brand(b);
    printf("CPU: %s\n", b);
    printf("max standard CPUID leaf: 0x%X\n\n", maxleaf);

    __cpuidex(r, 1, 0);
    l1c = (unsigned)r[2]; l1d = (unsigned)r[3];
    printf("leaf1  ECX=0x%08X EDX=0x%08X\n", l1c, l1d);
    printf("  OSXSAVE(ECX27)=%u  AVX(ECX28)=%u  XSAVE(ECX26)=%u  F16C(ECX29)=%u\n",
           BIT(l1c,27), BIT(l1c,28), BIT(l1c,26), BIT(l1c,29));

    if (BIT(l1c,27)) {
        xcr0 = _xgetbv(0);
        printf("XCR0=0x%llX  (SSE=%llu YMM=%llu OPMASK=%llu ZMM_Hi256=%llu Hi16_ZMM=%llu)\n",
               (unsigned long long)xcr0,
               (unsigned long long)BIT(xcr0,1), (unsigned long long)BIT(xcr0,2),
               (unsigned long long)BIT(xcr0,5), (unsigned long long)BIT(xcr0,6),
               (unsigned long long)BIT(xcr0,7));
        printf("  (XCR0 & 0xE6)==0xE6 -> AVX512 OS-enabled: %s\n",
               ((xcr0 & 0xE6u) == 0xE6u) ? "YES" : "NO");
    } else {
        printf("OSXSAVE=0 -> cannot read XCR0 (AVX512 will be reported unusable)\n");
    }
    printf("\n");

    l7b = l7c = l7d = l7a1 = 0;
    if (maxleaf >= 7) {
        __cpuidex(r, 7, 0);
        l7b = (unsigned)r[1]; l7c = (unsigned)r[2]; l7d = (unsigned)r[3];
        printf("leaf7.0  EBX=0x%08X ECX=0x%08X EDX=0x%08X\n", l7b, l7c, l7d);
        __cpuidex(r, 7, 1);
        l7a1 = (unsigned)r[0];
        printf("leaf7.1  EAX=0x%08X\n\n", l7a1);
    } else {
        printf("leaf7 not available (maxleaf<7)\n\n");
    }

    printf("=== AVX-512 feature bits ===\n");
    printf("  AVX512F     (7.0 EBX[16]) = %u\n", BIT(l7b,16));
    printf("  AVX512DQ    (7.0 EBX[17]) = %u\n", BIT(l7b,17));
    printf("  AVX512_IFMA (7.0 EBX[21]) = %u\n", BIT(l7b,21));
    printf("  AVX512CD    (7.0 EBX[28]) = %u\n", BIT(l7b,28));
    printf("  AVX512BW    (7.0 EBX[30]) = %u\n", BIT(l7b,30));
    printf("  AVX512VL    (7.0 EBX[31]) = %u\n", BIT(l7b,31));
    printf("  AVX512_VBMI (7.0 ECX[ 1]) = %u\n", BIT(l7c,1));
    printf("  AVX512_VNNI (7.0 ECX[11]) = %u\n", BIT(l7c,11));
    printf("  AVX512_BF16 (7.1 EAX[ 5]) = %u\n", BIT(l7a1,5));
    printf("  >>> AVX512_FP16 (7.0 EDX[23]) = %u <<<\n", BIT(l7d,23));

    return 0;
}
