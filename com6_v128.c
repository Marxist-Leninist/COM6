/*
 * COM6 v128 - Physical-Core Pinning + A-Pack Prefetch Fix
 * ========================================================
 * v128 core + two targeted fixes:
 *
 * 1. Physical-core-only threading: HT threads share FMA ports with the
 *    physical core, adding heat without throughput. Auto-detect HT and
 *    pin to physical cores. On i7-10510U: 4 threads instead of 8 ->
 *    higher sustained turbo frequency under 15W TDP.
 *
 * 2. A-packing prefetch fix: v128 only prefetched row 0 (a0+k+8).
 *    Rows 1-5 were unprefetched, each on a different cache line
 *    (stride = n*8 bytes apart). Now prefetch all 6 rows.
 *
 * 3. Reduced thermal pacing (fewer threads = less heat):
 *    50ms at 4096, 150ms at 8192 (was 100/250).
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp -flto \
 *       -o com6_v128 com6_v128.c -lm
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sched.h>
#endif

#define MR  6
#define NR  8
#define NC_DEFAULT  2048
#define NC_JC_DEFAULT 1024
#define NC_MAX 2048
#define ALIGN 64

#define KC_SMALL 256
#define MC_SMALL 120
#define KC_LARGE 320
#define MC_LARGE 96
#define KC_MAX 640
#define MC_MAX 192

#define MC_MT_SMALL 48

#define POOL_MAX_THREADS 64

static int g_is_chiplet = 0;
static int g_l2_kb = 256;
static int g_l3_kb = 6144;
static int g_phys_cores = 0;
static int g_detected = 0;
static int g_strassen_depth = 0;
static int g_max_strassen_depth = 1;
static int g_kc_auto = 0;
static int g_nc_1t = NC_DEFAULT;
static int g_nc_jc = NC_JC_DEFAULT;

static double* g_bufA[POOL_MAX_THREADS];
static double* g_bufB[POOL_MAX_THREADS];
static int g_pool_size = 0;

static inline double* aa(size_t c){return(double*)_mm_malloc(c*sizeof(double),ALIGN);}
static inline void af(double*p){_mm_free(p);}

static void ensure_pool(int nt) {
    if (nt <= g_pool_size) return;
    if (nt > POOL_MAX_THREADS) nt = POOL_MAX_THREADS;
    for (int t = g_pool_size; t < nt; t++) {
        g_bufA[t] = aa((size_t)MC_MAX * KC_MAX);
        g_bufB[t] = aa((size_t)KC_MAX * NC_MAX);
        memset(g_bufA[t], 0, (size_t)MC_MAX * KC_MAX * sizeof(double));
        memset(g_bufB[t], 0, (size_t)KC_MAX * NC_MAX * sizeof(double));
    }
    g_pool_size = nt;
}

/* ================================================================
 * PLATFORM DETECTION
 * ================================================================ */

static int get_num_cpus(void) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 4;
#endif
}

#ifdef _WIN32
static int detect_physical_cores_win(void) {
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (len == 0) return get_num_cpus();
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buf = malloc(len);
    if (!buf) return get_num_cpus();
    if (!GetLogicalProcessorInformation(buf, &len)) { free(buf); return get_num_cpus(); }
    int cores = 0;
    DWORD count = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    for (DWORD i = 0; i < count; i++) {
        if (buf[i].Relationship == RelationProcessorCore) cores++;
    }
    free(buf);
    return cores > 0 ? cores : get_num_cpus();
}

static int detect_l2_kb_win(void) {
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (len == 0) return 256;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buf = malloc(len);
    if (!buf) return 256;
    if (!GetLogicalProcessorInformation(buf, &len)) { free(buf); return 256; }
    int l2_kb = 256;
    DWORD count = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    for (DWORD i = 0; i < count; i++) {
        if (buf[i].Relationship == RelationCache && buf[i].Cache.Level == 2) {
            l2_kb = (int)(buf[i].Cache.Size / 1024);
            break;
        }
    }
    free(buf);
    return l2_kb;
}

static int detect_l3_kb_win(void) {
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (len == 0) return 6144;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buf = malloc(len);
    if (!buf) return 6144;
    if (!GetLogicalProcessorInformation(buf, &len)) { free(buf); return 6144; }
    int l3_kb = 0;
    DWORD count = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    for (DWORD i = 0; i < count; i++) {
        if (buf[i].Relationship == RelationCache && buf[i].Cache.Level == 3) {
            int this_kb = (int)(buf[i].Cache.Size / 1024);
            if (this_kb > l3_kb) l3_kb = this_kb;
        }
    }
    free(buf);
    return l3_kb > 0 ? l3_kb : 6144;
}
#endif

#ifndef _WIN32
static int detect_l3_groups(void) {
    int groups = 0;
    char prev[256] = "";
    for (int cpu = 0; cpu < 256; cpu++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", cpu);
        FILE* f = fopen(path, "r");
        if (!f) break;
        char buf[256] = "";
        if (fgets(buf, sizeof(buf), f)) {
            buf[strcspn(buf, "\n")] = 0;
            if (strcmp(buf, prev) != 0) { groups++; strncpy(prev, buf, 255); }
        }
        fclose(f);
    }
    return groups > 0 ? groups : 1;
}

static int detect_physical_cores(void) {
    int seen[256]; int nseen = 0;
    for (int cpu = 0; cpu < 256; cpu++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        FILE* f = fopen(path, "r");
        if (!f) break;
        int cid = -1; if (fscanf(f, "%d", &cid) != 1) cid = -1;
        fclose(f);
        int dup = 0;
        for (int i = 0; i < nseen; i++) if (seen[i] == cid) { dup = 1; break; }
        if (!dup && nseen < 256) seen[nseen++] = cid;
    }
    return nseen > 0 ? nseen : get_num_cpus();
}

static int detect_l2_kb(void) {
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
    if (!f) return 256;
    int kb = 256; char unit = 'K';
    if (fscanf(f, "%d%c", &kb, &unit) < 1) kb = 256;
    fclose(f);
    return kb;
}

static int detect_l3_kb(void) {
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
    if (!f) return 6144;
    int val = 6144; char unit = 'K';
    if (fscanf(f, "%d%c", &val, &unit) < 1) val = 6144;
    fclose(f);
    if (unit == 'M' || unit == 'm') val *= 1024;
    return val;
}
#endif

static int compute_nc(int kc, int l3_bytes, int max_nc) {
    int nc = (int)((double)l3_bytes * 0.75 / (kc * (int)sizeof(double)));
    nc = (nc / NR) * NR;
    if (nc < 256) nc = 256;
    if (nc > max_nc) nc = max_nc;
    return nc;
}

static void detect_platform(void) {
    if (g_detected) return;
    g_detected = 1;
#ifdef _WIN32
    g_is_chiplet = 0;
    g_phys_cores = detect_physical_cores_win();
    g_l2_kb = detect_l2_kb_win();
    g_l3_kb = detect_l3_kb_win();
#else
    int l3g = detect_l3_groups();
    g_is_chiplet = (l3g > 1);
    g_phys_cores = detect_physical_cores();
    g_l2_kb = detect_l2_kb();
    g_l3_kb = detect_l3_kb();
#endif

    int logical = get_num_cpus();
    if (g_phys_cores > 0 && g_phys_cores < logical) {
#ifdef _OPENMP
        omp_set_num_threads(g_phys_cores);
#endif
    }

    const char* tenv = getenv("COM6_THREADS");
    if (tenv && atoi(tenv) > 0) {
#ifdef _OPENMP
        omp_set_num_threads(atoi(tenv));
#endif
    }

    /* Auto-tune KC based on L2 size */
    if (g_l2_kb >= 512) {
        int budget = (int)(g_l2_kb * 1024 * 0.85);
        int kc = budget / (MC_SMALL * (int)sizeof(double));
        kc = (kc / 8) * 8;
        if (kc > 640) kc = 640;
        if (kc < KC_LARGE) kc = KC_LARGE;
        g_kc_auto = kc;
    }

    /* Auto-tune NC based on L3 size */
    int kc_for_nc = g_kc_auto ? g_kc_auto : KC_LARGE;
    int l3_bytes = g_l3_kb * 1024;
    g_nc_1t = compute_nc(kc_for_nc, l3_bytes, NC_DEFAULT);
    g_nc_jc = compute_nc(kc_for_nc, l3_bytes, NC_JC_DEFAULT);

    /* Strassen depth */
    g_max_strassen_depth = g_is_chiplet ? 2 : 1;
    const char* sdenv = getenv("COM6_STRASSEN_DEPTH");
    if (sdenv) {
        int d = atoi(sdenv);
        if (d >= 0 && d <= 3) g_max_strassen_depth = d;
    }

    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    #pragma omp parallel
    { volatile int x = omp_get_thread_num(); (void)x; }
#endif
    ensure_pool(nt);
}

/* ================================================================
 * MICRO-KERNEL: beta=1 (C += A*B) with C-prefetch
 * ================================================================ */
static void __attribute__((noinline))
micro_6x8_beta1(int kc, const double* pA, const double* pB, double* C, int ldc)
{
    long long kc_8 = kc >> 3;
    long long kc_rem = kc & 7;
    long long ldc_bytes = (long long)ldc * 8;

    __asm__ volatile(
        "movq %[C],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"   "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"   "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"   "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"   "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"   "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"   "prefetcht0 32(%%r15)\n\t"

        "vxorpd %%ymm0,%%ymm0,%%ymm0\n\t"
        "vxorpd %%ymm1,%%ymm1,%%ymm1\n\t"
        "vxorpd %%ymm2,%%ymm2,%%ymm2\n\t"
        "vxorpd %%ymm3,%%ymm3,%%ymm3\n\t"
        "vxorpd %%ymm4,%%ymm4,%%ymm4\n\t"
        "vxorpd %%ymm5,%%ymm5,%%ymm5\n\t"
        "vxorpd %%ymm6,%%ymm6,%%ymm6\n\t"
        "vxorpd %%ymm7,%%ymm7,%%ymm7\n\t"
        "vxorpd %%ymm8,%%ymm8,%%ymm8\n\t"
        "vxorpd %%ymm9,%%ymm9,%%ymm9\n\t"
        "vxorpd %%ymm10,%%ymm10,%%ymm10\n\t"
        "vxorpd %%ymm11,%%ymm11,%%ymm11\n\t"

        "testq %[kc8],%[kc8]\n\t"
        "jle 3f\n\t"
        ".p2align 5\n\t"
        "1:\n\t"

        "prefetcht0 768(%[pA])\n\t"
        "prefetcht0 1024(%[pB])\n\t"

#define RANK1(AO,BO) \
        "vmovapd " #BO "(%[pB]),%%ymm12\n\t" "vmovapd " #BO "+32(%[pB]),%%ymm13\n\t" \
        "vbroadcastsd " #AO "(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t" \
        "vbroadcastsd " #AO "+8(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t" \
        "vbroadcastsd " #AO "+16(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t" \
        "vbroadcastsd " #AO "+24(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t" \
        "vbroadcastsd " #AO "+32(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t" \
        "vbroadcastsd " #AO "+40(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        RANK1(0,0) RANK1(48,64) RANK1(96,128) RANK1(144,192)
        "prefetcht0 1152(%[pA])\n\t" "prefetcht0 1536(%[pB])\n\t"
        RANK1(192,256) RANK1(240,320) RANK1(288,384) RANK1(336,448)
#undef RANK1

        "addq $384,%[pA]\n\t" "addq $512,%[pB]\n\t"
        "decq %[kc8]\n\t" "jnz 1b\n\t"

        "3:\n\t" "testq %[kcr],%[kcr]\n\t" "jle 2f\n\t"
        ".p2align 4\n\t"
        "4:\n\t"
        "vmovapd (%[pB]),%%ymm12\n\t" "vmovapd 32(%[pB]),%%ymm13\n\t"
        "vbroadcastsd (%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 8(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 16(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 24(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 32(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 40(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"
        "addq $48,%[pA]\n\t" "addq $64,%[pB]\n\t"
        "decq %[kcr]\n\t" "jnz 4b\n\t"

        "2:\n\t"
        "vaddpd (%[C]),%%ymm0,%%ymm0\n\t"  "vmovupd %%ymm0,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm1,%%ymm1\n\t" "vmovupd %%ymm1,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm2,%%ymm2\n\t"  "vmovupd %%ymm2,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm3,%%ymm3\n\t" "vmovupd %%ymm3,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm4,%%ymm4\n\t"  "vmovupd %%ymm4,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm5,%%ymm5\n\t" "vmovupd %%ymm5,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm6,%%ymm6\n\t"  "vmovupd %%ymm6,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm7,%%ymm7\n\t" "vmovupd %%ymm7,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm8,%%ymm8\n\t"  "vmovupd %%ymm8,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm9,%%ymm9\n\t" "vmovupd %%ymm9,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm10,%%ymm10\n\t" "vmovupd %%ymm10,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm11,%%ymm11\n\t" "vmovupd %%ymm11,32(%[C])\n\t"

        : [pA]"+r"(pA),[pB]"+r"(pB),[kc8]"+r"(kc_8),[kcr]"+r"(kc_rem),[C]"+r"(C)
        : [ldc]"r"(ldc_bytes)
        : "r15","ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
          "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","memory"
    );
}

/* ================================================================
 * MICRO-KERNEL: beta=0 (C = A*B, just store)
 * ================================================================ */
static void __attribute__((noinline))
micro_6x8_beta0(int kc, const double* pA, const double* pB, double* C, int ldc)
{
    long long kc_8 = kc >> 3;
    long long kc_rem = kc & 7;
    long long ldc_bytes = (long long)ldc * 8;

    __asm__ volatile(
        "vxorpd %%ymm0,%%ymm0,%%ymm0\n\t"
        "vxorpd %%ymm1,%%ymm1,%%ymm1\n\t"
        "vxorpd %%ymm2,%%ymm2,%%ymm2\n\t"
        "vxorpd %%ymm3,%%ymm3,%%ymm3\n\t"
        "vxorpd %%ymm4,%%ymm4,%%ymm4\n\t"
        "vxorpd %%ymm5,%%ymm5,%%ymm5\n\t"
        "vxorpd %%ymm6,%%ymm6,%%ymm6\n\t"
        "vxorpd %%ymm7,%%ymm7,%%ymm7\n\t"
        "vxorpd %%ymm8,%%ymm8,%%ymm8\n\t"
        "vxorpd %%ymm9,%%ymm9,%%ymm9\n\t"
        "vxorpd %%ymm10,%%ymm10,%%ymm10\n\t"
        "vxorpd %%ymm11,%%ymm11,%%ymm11\n\t"

        "testq %[kc8],%[kc8]\n\t"
        "jle 3f\n\t"
        ".p2align 5\n\t"
        "1:\n\t"

        "prefetcht0 768(%[pA])\n\t"
        "prefetcht0 1024(%[pB])\n\t"

#define RANK1(AO,BO) \
        "vmovapd " #BO "(%[pB]),%%ymm12\n\t" "vmovapd " #BO "+32(%[pB]),%%ymm13\n\t" \
        "vbroadcastsd " #AO "(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t" \
        "vbroadcastsd " #AO "+8(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t" \
        "vbroadcastsd " #AO "+16(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t" \
        "vbroadcastsd " #AO "+24(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t" \
        "vbroadcastsd " #AO "+32(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t" \
        "vbroadcastsd " #AO "+40(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        RANK1(0,0) RANK1(48,64) RANK1(96,128) RANK1(144,192)
        "prefetcht0 1152(%[pA])\n\t" "prefetcht0 1536(%[pB])\n\t"
        RANK1(192,256) RANK1(240,320) RANK1(288,384) RANK1(336,448)
#undef RANK1

        "addq $384,%[pA]\n\t" "addq $512,%[pB]\n\t"
        "decq %[kc8]\n\t" "jnz 1b\n\t"

        "3:\n\t" "testq %[kcr],%[kcr]\n\t" "jle 2f\n\t"
        ".p2align 4\n\t"
        "4:\n\t"
        "vmovapd (%[pB]),%%ymm12\n\t" "vmovapd 32(%[pB]),%%ymm13\n\t"
        "vbroadcastsd (%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 8(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 16(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 24(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 32(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 40(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"
        "addq $48,%[pA]\n\t" "addq $64,%[pB]\n\t"
        "decq %[kcr]\n\t" "jnz 4b\n\t"

        "2:\n\t"
        "vmovupd %%ymm0,(%[C])\n\t" "vmovupd %%ymm1,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm2,(%[C])\n\t" "vmovupd %%ymm3,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm4,(%[C])\n\t" "vmovupd %%ymm5,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm6,(%[C])\n\t" "vmovupd %%ymm7,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm8,(%[C])\n\t" "vmovupd %%ymm9,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm10,(%[C])\n\t" "vmovupd %%ymm11,32(%[C])\n\t"

        : [pA]"+r"(pA),[pB]"+r"(pB),[kc8]"+r"(kc_8),[kcr]"+r"(kc_rem),[C]"+r"(C)
        : [ldc]"r"(ldc_bytes)
        : "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
          "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","memory"
    );
}

/* ================================================================
 * EDGE CASES
 * ================================================================ */
static void micro_edge(int mr, int nr, int kc, const double*pA, const double*pB,
                        double*C, int ldc, int beta){
    if(nr == NR && mr < MR){
        double tmp[MR*NR] __attribute__((aligned(64)));
        if(beta){
            for(int i=0;i<MR;i++){
                if(i<mr) memcpy(tmp+i*NR, C+i*(size_t)ldc, NR*sizeof(double));
                else memset(tmp+i*NR, 0, NR*sizeof(double));
            }
            micro_6x8_beta1(kc, pA, pB, tmp, NR);
        } else {
            micro_6x8_beta0(kc, pA, pB, tmp, NR);
        }
        for(int i=0;i<mr;i++)
            memcpy(C+i*(size_t)ldc, tmp+i*NR, NR*sizeof(double));
        return;
    }
    double tmp[MR*NR] __attribute__((aligned(64)));
    if(beta){for(int i=0;i<mr;i++)for(int j=0;j<nr;j++)tmp[i*NR+j]=C[i*ldc+j];}
    else{memset(tmp,0,sizeof(tmp));}
    for(int k=0;k<kc;k++){
        for(int i=0;i<mr;i++){
            double a=pA[k*MR+i];
            for(int j=0;j<nr;j++) tmp[i*NR+j]+=a*pB[k*NR+j];
        }
    }
    for(int i=0;i<mr;i++)for(int j=0;j<nr;j++)C[i*ldc+j]=tmp[i*NR+j];
}

/* ================================================================
 * PACKING — 4x k-unrolled B-pack, 4x k-unrolled A-pack
 * ================================================================ */
static void pack_B_chunk(const double*B, double*pb, int kc, int nc, int n,
                          int jc, int pc, int j_start, int j_end){
    for(int j=j_start;j<j_end;j+=NR){
        int nr=(j+NR<=nc)?NR:nc-j;
        double*dst=pb+(j/NR)*((size_t)NR*kc);
        if(nr==NR){
            int k=0;
            for(;k+3<kc;k+=4){
                const double*s0=B+(size_t)(pc+k)*n+(jc+j);
                const double*s1=B+(size_t)(pc+k+1)*n+(jc+j);
                const double*s2=B+(size_t)(pc+k+2)*n+(jc+j);
                const double*s3=B+(size_t)(pc+k+3)*n+(jc+j);
                if(k+7<kc) __builtin_prefetch(B+(size_t)(pc+k+4)*n+(jc+j), 0, 1);
                _mm256_storeu_pd(dst,    _mm256_loadu_pd(s0));
                _mm256_storeu_pd(dst+4,  _mm256_loadu_pd(s0+4));
                _mm256_storeu_pd(dst+8,  _mm256_loadu_pd(s1));
                _mm256_storeu_pd(dst+12, _mm256_loadu_pd(s1+4));
                _mm256_storeu_pd(dst+16, _mm256_loadu_pd(s2));
                _mm256_storeu_pd(dst+20, _mm256_loadu_pd(s2+4));
                _mm256_storeu_pd(dst+24, _mm256_loadu_pd(s3));
                _mm256_storeu_pd(dst+28, _mm256_loadu_pd(s3+4));
                dst+=32;
            }
            for(;k<kc;k++){
                const double*src=B+(size_t)(pc+k)*n+(jc+j);
                _mm256_storeu_pd(dst, _mm256_loadu_pd(src));
                _mm256_storeu_pd(dst+4, _mm256_loadu_pd(src+4));
                dst+=NR;
            }
        } else {
            for(int k=0;k<kc;k++){
                const double*src=B+(size_t)(pc+k)*n+(jc+j);
                int jj;for(jj=0;jj<nr;jj++)dst[jj]=src[jj];
                for(;jj<NR;jj++)dst[jj]=0.0;
                dst+=NR;
            }
        }
    }
}

static void pack_A(const double*A, double*pa, int mc, int kc, int n, int ic, int pc){
    const double*Ab=A+(size_t)pc;
    for(int i=0;i<mc;i+=MR){
        int mr=(i+MR<=mc)?MR:mc-i;
        if(mr==MR){
            const double*a0=Ab+(size_t)(ic+i)*n, *a1=a0+n, *a2=a1+n;
            const double*a3=a2+n, *a4=a3+n, *a5=a4+n;
            int k=0;
            for(;k+3<kc;k+=4){
                if(k+7<kc){
                    __builtin_prefetch(a0+k+8, 0, 1);
                    __builtin_prefetch(a1+k+8, 0, 1);
                    __builtin_prefetch(a2+k+8, 0, 1);
                    __builtin_prefetch(a3+k+8, 0, 1);
                    __builtin_prefetch(a4+k+8, 0, 1);
                    __builtin_prefetch(a5+k+8, 0, 1);
                }
                pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];
                pa[6]=a0[k+1];pa[7]=a1[k+1];pa[8]=a2[k+1];pa[9]=a3[k+1];pa[10]=a4[k+1];pa[11]=a5[k+1];
                pa[12]=a0[k+2];pa[13]=a1[k+2];pa[14]=a2[k+2];pa[15]=a3[k+2];pa[16]=a4[k+2];pa[17]=a5[k+2];
                pa[18]=a0[k+3];pa[19]=a1[k+3];pa[20]=a2[k+3];pa[21]=a3[k+3];pa[22]=a4[k+3];pa[23]=a5[k+3];
                pa+=24;
            }
            for(;k<kc;k++){
                pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];
                pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];pa+=MR;
            }
        }else{
            for(int k=0;k<kc;k++){
                int ii;for(ii=0;ii<mr;ii++)pa[ii]=(Ab+(ic+i+ii)*(size_t)n)[k];
                for(;ii<MR;ii++)pa[ii]=0.0;pa+=MR;
            }
        }
    }
}

static void macro_kernel(const double*pa,const double*pb,
                          double*C,int mc,int nc,int kc,int n,int ic,int jc,int beta){
    for(int jr=0;jr<nc;jr+=NR){
        int nr=(jr+NR<=nc)?NR:nc-jr;
        const double*pB=pb+(jr/NR)*((size_t)NR*kc);
        for(int ir=0;ir<mc;ir+=MR){
            int mr=(ir+MR<=mc)?MR:mc-ir;
            const double*pA=pa+(ir/MR)*((size_t)MR*kc);
            double*Cij=C+(size_t)(ic+ir)*n+(jc+jr);
            if(mr==MR&&nr==NR){
                if(beta) micro_6x8_beta1(kc,pA,pB,Cij,n);
                else     micro_6x8_beta0(kc,pA,pB,Cij,n);
            } else micro_edge(mr,nr,kc,pA,pB,Cij,n,beta);
        }
    }
}

/* ================================================================
 * BLOCKING
 * ================================================================ */
static int get_mc_for_l2(int kc) {
    int l2_bytes = g_l2_kb * 1024;
    int mc = (int)(l2_bytes * 0.88 / (kc * sizeof(double)));
    mc = (mc / MR) * MR;
    if (mc < MR) mc = MR;
    if (mc > MC_MAX) mc = MC_MAX;
    return mc;
}

static int get_kc_large(void) {
    return g_kc_auto ? g_kc_auto : KC_LARGE;
}

static void get_blocking(int n, int*pMC, int*pKC){
    if (g_is_chiplet || g_l2_kb >= 512) {
        *pKC = (n <= 1024) ? KC_SMALL : get_kc_large();
        *pMC = get_mc_for_l2(*pKC);
    } else {
        if(n <= 1024){*pMC=MC_SMALL;*pKC=KC_SMALL;}
        else{*pMC=MC_LARGE;*pKC=get_kc_large();}
    }
}

static void get_blocking_mt(int n, int*pMC, int*pKC){
    if (g_is_chiplet) {
        *pKC = (n <= 1024) ? KC_SMALL : get_kc_large();
        *pMC = get_mc_for_l2(*pKC);
    } else {
        if(n <= 1024){*pMC=MC_MT_SMALL;*pKC=KC_SMALL;}
        else if(n >= 4096){*pMC=MC_MT_SMALL;*pKC=get_kc_large();}
        else{*pMC=MC_LARGE;*pKC=get_kc_large();}
    }
}

/* ================================================================
 * SINGLE-THREADED MULTIPLY
 * ================================================================ */
static void com6_multiply_1t(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C,int n)
{
    int mc_blk, kc_blk;
    get_blocking(n, &mc_blk, &kc_blk);
    int nc_blk = g_nc_1t;
    ensure_pool(1);
    double *pa = g_bufA[0], *pb = g_bufB[0];
    for(int jc=0;jc<n;jc+=nc_blk){int nc=(jc+nc_blk<=n)?nc_blk:n-jc;
        for(int pc=0;pc<n;pc+=kc_blk){int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
            int beta=(pc>0)?1:0;
            pack_B_chunk(B,pb,kc,nc,n,jc,pc,0,nc);
            for(int ic=0;ic<n;ic+=mc_blk){int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                pack_A(A,pa,mc,kc,n,ic,pc);
                macro_kernel(pa,pb,C,mc,nc,kc,n,ic,jc,beta);
    }}}
}

/* ================================================================
 * JC-PARALLEL
 * ================================================================ */
static void com6_multiply_jc(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C, int n,
                              int mc_blk, int kc_blk)
{
#ifdef _OPENMP
    int nc_blk = g_nc_jc;
    ensure_pool(omp_get_max_threads());
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();

        int cols_per = ((n / nt + NR - 1) / NR) * NR;
        int j_start = tid * cols_per;
        int j_end = j_start + cols_per;
        if(j_end > n) j_end = n;

        if(j_start < n){
            double *pa = g_bufA[tid], *pb = g_bufB[tid];

            for(int jc=j_start; jc<j_end; jc+=nc_blk){
                int nc = (jc+nc_blk<=j_end) ? nc_blk : j_end-jc;
                for(int pc=0; pc<n; pc+=kc_blk){
                    int kc = (pc+kc_blk<=n) ? kc_blk : n-pc;
                    int beta = (pc>0) ? 1 : 0;
                    pack_B_chunk(B, pb, kc, nc, n, jc, pc, 0, nc);
                    for(int ic=0; ic<n; ic+=mc_blk){
                        int mc = (ic+mc_blk<=n) ? mc_blk : n-ic;
                        pack_A(A, pa, mc, kc, n, ic, pc);
                        macro_kernel(pa, pb, C, mc, nc, kc, n, ic, jc, beta);
                    }
                }
            }
        }
    }
#else
    com6_multiply_1t(A,B,C,n);
#endif
}

/* ================================================================
 * IC-PARALLEL
 * ================================================================ */
static void com6_multiply_ic(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C, int n)
{
    int mc_blk, kc_blk;
    get_blocking_mt(n, &mc_blk, &kc_blk);
    int nc_blk = g_nc_1t;

    int pace_ms = 0;
    if(g_strassen_depth == 0){
        int pace_min_n = 4096;
        const char *min_env = getenv("COM6_PACE_MIN_N");
        if(min_env){ int v = atoi(min_env); if(v > 0 && v <= 16384) pace_min_n = v; }
        if(n >= pace_min_n){
            pace_ms = (n >= 8192) ? 150 : 50;
            const char *env = getenv("COM6_PACE_MS");
            if(env){ int v = atoi(env); if(v >= 0 && v <= 10000) pace_ms = v; }
        }
    }

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    ensure_pool(nthreads);
    double* pb = g_bufB[0];

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        double* pa = g_bufA[tid];

        for(int jc=0;jc<n;jc+=nc_blk){
            int nc=(jc+nc_blk<=n)?nc_blk:n-jc;
            if(jc > 0 && pace_ms > 0){
                #pragma omp master
                {
#ifdef _WIN32
                    Sleep(pace_ms);
#else
                    usleep((unsigned)(pace_ms) * 1000u);
#endif
                }
                #pragma omp barrier
            }
            for(int pc=0;pc<n;pc+=kc_blk){
                int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
                int beta=(pc>0)?1:0;

                int npanels = (nc + NR - 1) / NR;
                int panels_per = (npanels + nt - 1) / nt;
                int p0 = tid * panels_per;
                int p1 = p0 + panels_per;
                if(p1 > npanels) p1 = npanels;
                int j_start = p0 * NR;
                int j_end = p1 * NR;
                if(j_end > nc) j_end = nc;
                if(j_start < nc)
                    pack_B_chunk(B,pb,kc,nc,n,jc,pc,j_start,j_end);

                #pragma omp barrier

                #pragma omp for schedule(dynamic,2)
                for(int ic=0;ic<n;ic+=mc_blk){
                    int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                    pack_A(A,pa,mc,kc,n,ic,pc);
                    macro_kernel(pa,pb,C,mc,nc,kc,n,ic,jc,beta);
                }
            }
        }
    }
#else
    com6_multiply_1t(A,B,C,n);
#endif
}

/* ================================================================
 * STRASSEN
 * ================================================================ */

static void com6_multiply(const double*__restrict__ A,
                           const double*__restrict__ B,
                           double*__restrict__ C, int n);

static void quad_copy(const double*S, int lds, int r, int c, int h, double*D){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<h;i++)
        memcpy(D+(size_t)i*h, S+(size_t)(r+i)*lds+c, (size_t)h*sizeof(double));
}

static void quad_add(const double*A, int lda, int ra, int ca,
                     const double*B, int ldb, int rb, int cb,
                     double*out, int h){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<h;i++){
        const double*ai=A+(size_t)(ra+i)*lda+ca;
        const double*bi=B+(size_t)(rb+i)*ldb+cb;
        double*oi=out+(size_t)i*h;
        int j=0;
        for(;j+3<h;j+=4){
            _mm256_storeu_pd(oi+j,_mm256_add_pd(_mm256_loadu_pd(ai+j),_mm256_loadu_pd(bi+j)));
        }
        for(;j<h;j++) oi[j]=ai[j]+bi[j];
    }
}

static void quad_sub(const double*A, int lda, int ra, int ca,
                     const double*B, int ldb, int rb, int cb,
                     double*out, int h){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<h;i++){
        const double*ai=A+(size_t)(ra+i)*lda+ca;
        const double*bi=B+(size_t)(rb+i)*ldb+cb;
        double*oi=out+(size_t)i*h;
        int j=0;
        for(;j+3<h;j+=4){
            _mm256_storeu_pd(oi+j,_mm256_sub_pd(_mm256_loadu_pd(ai+j),_mm256_loadu_pd(bi+j)));
        }
        for(;j<h;j++) oi[j]=ai[j]-bi[j];
    }
}

static void c_acc_add(double*C, int ldc, int r, int c, int h, const double*M){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<h;i++){
        double*ci=C+(size_t)(r+i)*ldc+c;
        const double*mi=M+(size_t)i*h;
        int j=0;
        for(;j+3<h;j+=4){
            _mm256_storeu_pd(ci+j,_mm256_add_pd(_mm256_loadu_pd(ci+j),_mm256_loadu_pd(mi+j)));
        }
        for(;j<h;j++) ci[j]+=mi[j];
    }
}

static void c_acc_sub(double*C, int ldc, int r, int c, int h, const double*M){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<h;i++){
        double*ci=C+(size_t)(r+i)*ldc+c;
        const double*mi=M+(size_t)i*h;
        int j=0;
        for(;j+3<h;j+=4){
            _mm256_storeu_pd(ci+j,_mm256_sub_pd(_mm256_loadu_pd(ci+j),_mm256_loadu_pd(mi+j)));
        }
        for(;j<h;j++) ci[j]-=mi[j];
    }
}

static void com6_strassen(const double*A, const double*B, double*C, int n){
    int h=n/2;
    size_t hh=(size_t)h*h;
    double*T1=aa(hh), *T2=aa(hh), *M=aa(hh);
    if(!T1||!T2||!M){ if(T1)af(T1); if(T2)af(T2); if(M)af(M);
        com6_multiply_ic(A,B,C,n); return; }

    g_strassen_depth++;
    memset(C, 0, (size_t)n*n*sizeof(double));

    quad_add(A,n,0,0, A,n,h,h, T1,h);
    quad_add(B,n,0,0, B,n,h,h, T2,h);
    com6_multiply(T1,T2,M,h);
    c_acc_add(C,n,0,0,h,M);
    c_acc_add(C,n,h,h,h,M);

    quad_add(A,n,h,0, A,n,h,h, T1,h);
    quad_copy(B,n,0,0,h,T2);
    com6_multiply(T1,T2,M,h);
    c_acc_add(C,n,h,0,h,M);
    c_acc_sub(C,n,h,h,h,M);

    quad_copy(A,n,0,0,h,T1);
    quad_sub(B,n,0,h, B,n,h,h, T2,h);
    com6_multiply(T1,T2,M,h);
    c_acc_add(C,n,0,h,h,M);
    c_acc_add(C,n,h,h,h,M);

    quad_copy(A,n,h,h,h,T1);
    quad_sub(B,n,h,0, B,n,0,0, T2,h);
    com6_multiply(T1,T2,M,h);
    c_acc_add(C,n,0,0,h,M);
    c_acc_add(C,n,h,0,h,M);

    quad_add(A,n,0,0, A,n,0,h, T1,h);
    quad_copy(B,n,h,h,h,T2);
    com6_multiply(T1,T2,M,h);
    c_acc_sub(C,n,0,0,h,M);
    c_acc_add(C,n,0,h,h,M);

    quad_sub(A,n,h,0, A,n,0,0, T1,h);
    quad_add(B,n,0,0, B,n,0,h, T2,h);
    com6_multiply(T1,T2,M,h);
    c_acc_add(C,n,h,h,h,M);

    quad_sub(A,n,0,h, A,n,h,h, T1,h);
    quad_add(B,n,h,0, B,n,h,h, T2,h);
    com6_multiply(T1,T2,M,h);
    c_acc_add(C,n,0,0,h,M);

    g_strassen_depth--;
    af(T1); af(T2); af(M);
}

/* ================================================================
 * DISPATCH
 * ================================================================ */
static void com6_multiply(const double*__restrict__ A,
                           const double*__restrict__ B,
                           double*__restrict__ C, int n)
{
    detect_platform();
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    if(n < 512 || nthreads <= 1){ com6_multiply_1t(A,B,C,n); return; }

    int use_strassen = 0;
    if ((n & 1) == 0 && g_strassen_depth < g_max_strassen_depth) {
        int threshold = (g_strassen_depth == 0) ? 8192 : 4096;
        if (n >= threshold) use_strassen = 1;
    }
    const char *no_strassen = getenv("COM6_NO_STRASSEN");
    if(no_strassen && atoi(no_strassen)==1) use_strassen = 0;
    const char *force_strassen = getenv("COM6_USE_STRASSEN");
    if(force_strassen && atoi(force_strassen)==1 && n >= 4096 && (n & 1) == 0) use_strassen = 1;
    if(use_strassen){ com6_strassen(A, B, C, n); return; }

    const char *fjc = getenv("COM6_FORCE_JC");
    const char *fic = getenv("COM6_FORCE_IC");
    if (fjc && atoi(fjc) == 1) {
        int mc_blk, kc_blk;
        get_blocking_mt(n, &mc_blk, &kc_blk);
        com6_multiply_jc(A, B, C, n, mc_blk, kc_blk);
        return;
    }
    if (fic && atoi(fic) == 1) {
        com6_multiply_ic(A, B, C, n);
        return;
    }

    if (g_is_chiplet) {
        int mc_blk, kc_blk;
        get_blocking_mt(n, &mc_blk, &kc_blk);
        com6_multiply_jc(A, B, C, n, mc_blk, kc_blk);
    } else if (g_l2_kb >= 512) {
        int mc_blk, kc_blk;
        get_blocking_mt(n, &mc_blk, &kc_blk);
        com6_multiply_jc(A, B, C, n, mc_blk, kc_blk);
    } else {
        int mc_blk, kc_blk;
        get_blocking_mt(n, &mc_blk, &kc_blk);
        int kc = (n <= 1024) ? KC_SMALL : get_kc_large();
        int bpanel_per_thread = (n / 8) * kc * (int)sizeof(double);
        int apanel = mc_blk * kc * (int)sizeof(double);
        int l2_bytes = g_l2_kb * 1024;
        if(n >= 4096){
            com6_multiply_ic(A, B, C, n);
        } else if(bpanel_per_thread + apanel <= (int)(l2_bytes * 0.95)){
            com6_multiply_jc(A, B, C, n, mc_blk, kc_blk);
        } else if(n >= 2048){
            com6_multiply_jc(A, B, C, n, mc_blk, kc_blk);
        } else {
            com6_multiply_ic(A, B, C, n);
        }
    }
}

/* ================================================================
 * BENCHMARKING
 * ================================================================ */
static void naive(const double*A,const double*B,double*C,int n){
    memset(C,0,(size_t)n*n*sizeof(double));
    for(int i=0;i<n;i++)for(int k=0;k<n;k++){double a=A[i*n+k];for(int j=0;j<n;j++)C[i*n+j]+=a*B[k*n+j];}
}
static double now(void){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec+t.tv_nsec*1e-9;}
static void randf(double*M,int n){for(int i=0;i<n*n;i++)M[i]=(double)rand()/RAND_MAX*2-1;}
static double maxerr(const double*A,const double*B,int n){
    double m=0;for(int i=0;i<n*n;i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;}return m;
}

int main(int argc, char**argv){
    detect_platform();
    int nth=1;
#ifdef _OPENMP
    nth=omp_get_max_threads();
#endif

    if(argc >= 2){
        int n=atoi(argv[1]);
        if(n<64||n>32768){printf("Size must be 64..32768\n");return 1;}
        int mode=0;
        if(argc>=3){
            if(strcmp(argv[2],"1t")==0) mode=1;
            else if(strcmp(argv[2],"mt")==0) mode=2;
        }
        size_t nn=(size_t)n*n;
        double*A=aa(nn),*B=aa(nn),*C=aa(nn);
        if(!A||!B||!C){printf("OOM for %dx%d (need %.1f GB)\n",n,n,3.0*nn*8/1e9);return 1;}
        srand(42);randf(A,n);randf(B,n);

        int runs=(n<=512)?7:(n<=1024)?5:(n<=2048)?3:2;
        if(n>=16384) runs=1;

        if(mode==0||mode==1){
            com6_multiply_1t(A,B,C,n);
            double best=1e30;
            for(int r=0;r<runs;r++){double t0=now();com6_multiply_1t(A,B,C,n);double t=now()-t0;if(t<best)best=t;}
            printf("%dx%d 1T: %.1f ms (%.1f GF)\n",n,n,best*1000,(2.0*n*n*(double)n)/(best*1e9));
        }
        if(mode==0||mode==2){
            com6_multiply(A,B,C,n);
            double best=1e30;
            for(int r=0;r<runs;r++){double t0=now();com6_multiply(A,B,C,n);double t=now()-t0;if(t<best)best=t;}
            printf("%dx%d MT(%dT): %.1f ms (%.1f GF)\n",n,n,nth,best*1000,(2.0*n*n*(double)n)/(best*1e9));
        }
        af(A);af(B);af(C);
        return 0;
    }

    int kc_l = get_kc_large();
    printf("COM6 v128 - Physical-Core Pinning + A-Pack Prefetch Fix\n");
    printf("Threads: %d (physical cores: %d, L2: %dKB, L3: %dKB, %s)\n",
           nth, g_phys_cores, g_l2_kb, g_l3_kb, g_is_chiplet ? "CHIPLET" : "MONOLITHIC");
    printf("KC: small=%d, large=%d%s\n", KC_SMALL, kc_l,
           g_kc_auto ? " (L2-auto)" : " (default)");
    printf("NC: 1T=%d, JC=%d (L3-auto)\n", g_nc_1t, g_nc_jc);
    printf("Strassen: max_depth=%d\n", g_max_strassen_depth);
    printf("Dispatch: %s\n\n", g_is_chiplet ?
           "JC-par all sizes (chiplet)" :
           g_l2_kb >= 512 ?
           "JC-par all sizes (server, L2-aware MC)" :
           "L2-fit JC | IC-dyn 1024 | JC 2048 | IC-dyn+pace 4096+ (laptop)");
    printf(" Size      |   1T (ms)  |   MT (ms)  |  GF(1T) |  GF(MT) | Verify\n");
    printf("-----------|------------|------------|---------|---------|-------\n");

    int sizes_laptop[]={8192,4096,2048,1024,512,256};
    int sizes_server[]={16384,8192,4096,2048,1024,512,256};
    int *sizes;
    int nsizes;
    if(g_l2_kb >= 512 || g_is_chiplet){
        sizes = sizes_server; nsizes = 7;
    } else {
        sizes = sizes_laptop; nsizes = 6;
    }

    for(int si=0;si<nsizes;si++){
        int n=sizes[si];
        size_t nn=(size_t)n*n;
        double*A=aa(nn),*B=aa(nn),*C1=NULL,*C2=aa(nn);
        if(!A||!B||!C2){printf("OOM at %d\n",n);break;}
        srand(42);randf(A,n);randf(B,n);

        int do_1t = (g_is_chiplet || g_l2_kb >= 512) ? (n<=4096) : (n<=2048);
        int runs=(n<=512)?7:(n<=1024)?5:(n<=2048)?3:2;
        if(n>=16384) runs=1;
        double best_1=1e30,gf1=0;
        if(do_1t){
            C1=aa(nn);com6_multiply_1t(A,B,C1,n);
            for(int r=0;r<runs;r++){double t0=now();com6_multiply_1t(A,B,C1,n);double t=now()-t0;if(t<best_1)best_1=t;}
            gf1=(2.0*n*n*(double)n)/(best_1*1e9);
        }

        com6_multiply(A,B,C2,n);
        double best_a=1e30;
        for(int r=0;r<runs;r++){double t0=now();com6_multiply(A,B,C2,n);double t=now()-t0;if(t<best_a)best_a=t;}
        double gfa=(2.0*n*n*(double)n)/(best_a*1e9);

        const char*v;
        if(n<=512){double*Cr=aa(nn);naive(A,B,Cr,n);double e=fmax(do_1t?maxerr(C1,Cr,n):0,maxerr(C2,Cr,n));v=e<1e-6?"OK":"FAIL";af(Cr);}
        else if(do_1t&&C1){v=maxerr(C1,C2,n)<1e-6?"OK":"FAIL";}
        else{double*C3=aa(nn);if(C3){com6_multiply_1t(A,B,C3,n);v=maxerr(C2,C3,n)<1e-6?"OK":"FAIL";af(C3);}else v="SKIP";}

        if(do_1t)
            printf("%5dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %6.1f   | %s\n",n,n,best_1*1000,best_a*1000,gf1,gfa,v);
        else
            printf("%5dx%-5d | %10s | %8.1f ms | %6s   | %6.1f   | %s\n",n,n,"--",best_a*1000,"--",gfa,v);

        af(A);af(B);if(C1)af(C1);af(C2);

        if(si < nsizes-1){
#ifdef _WIN32
            Sleep(4000);
#else
            struct timespec ts = {4, 0};
            nanosleep(&ts, NULL);
#endif
        }
    }
    printf("\nv128: phys-core pinning, 6-row A-pack prefetch. Strassen depth=%d, KC=%d, NC_1T=%d, NC_JC=%d.\n",
           g_max_strassen_depth, kc_l, g_nc_1t, g_nc_jc);
    printf("%s, %d physical cores, L2=%dKB, L3=%dKB.\n",
           g_is_chiplet ? "Multi-CCD" : "Monolithic", g_phys_cores, g_l2_kb, g_l3_kb);
    printf("COM6_THREADS=N | COM6_STRASSEN_DEPTH=N | COM6_NO_STRASSEN=1\n");
    printf("Run individual: ./com6_v128 <size> [mt|1t]\n");

    return 0;
}
