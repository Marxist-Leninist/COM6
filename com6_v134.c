/*
 * COM6 v134 - MT at 256 + deeper KC for 2048 + beta0 C-prefetch
 * ===============================================================
 * Over v134 (Strassen memset elimination):
 *
 * CHANGE 1: C-prefetch in beta0 micro-kernel.
 *   Temporal stores (vmovupd) trigger write-allocate: CPU reads the
 *   cache line before overwriting it. Prefetching C rows at kernel entry
 *   hides this latency — same technique already used in beta1 kernel.
 *
 * CHANGE 2: Deeper KC=384 for 2048 (was KC=320).
 *   Reduces K-passes from ceil(2048/320)=7 to ceil(2048/384)=6.
 *   One fewer full C-matrix round-trip saves ~14% DRAM bandwidth.
 *   MC=72 (from L2 auto-fit): A-panel = 72*384*8 = 221KB fits 256KB L2.
 *
 * CHANGE 3: MT at n=256 via JC-pool (was 1T only).
 *   Pool dispatch costs ~1-3us vs 0.8ms compute — <1% overhead.
 *   Uses physical cores only (avoids HT L2 contention at small sizes).
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp \
 *       -o com6_v134 com6_v134.c -lm
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdatomic.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
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
#define KC_DEEP 384
#define KC_MAX 640
#define MC_MAX 192

#define MC_MT_SMALL 48

#define BUF_MAX_THREADS 64
#define POOL_SPIN_ITERS 2000

static int g_is_chiplet = 0;
static int g_l2_kb = 256;
static int g_l3_kb = 6144;
static int g_phys_cores = 0;
static int g_logical_cores = 0;
static int g_detected = 0;
static int g_strassen_depth = 0;
static int g_max_strassen_depth = 1;
static int g_kc_auto = 0;
static int g_nc_1t = NC_DEFAULT;
static int g_nc_jc = NC_JC_DEFAULT;
static int g_use_nt = 0;

static double* g_bufA[BUF_MAX_THREADS];
static double* g_bufB[BUF_MAX_THREADS];
static int g_bufs_allocated = 0;

static inline double* aa(size_t c){return(double*)_mm_malloc(c*sizeof(double),ALIGN);}
static inline void af(double*p){_mm_free(p);}

static void ensure_bufs(int nt) {
    if (nt <= g_bufs_allocated) return;
    if (nt > BUF_MAX_THREADS) nt = BUF_MAX_THREADS;
    for (int t = g_bufs_allocated; t < nt; t++) {
        g_bufA[t] = aa((size_t)MC_MAX * KC_MAX);
        g_bufB[t] = aa((size_t)KC_MAX * NC_MAX);
    }
    g_bufs_allocated = nt;
}

/* ================================================================
 * PERSISTENT THREAD POOL
 * ================================================================ */

typedef struct {
    const double *A, *B;
    double *C;
    int n, mc_blk, kc_blk, nc_blk;
    int j_start, j_end;
    double *pa, *pb;

    atomic_int command;   /* 0=idle, 1=jc_work, -1=shutdown */
    atomic_int done;      /* 1=finished, 0=working */

#ifdef _WIN32
    HANDLE handle;
    CONDITION_VARIABLE cv;
    CRITICAL_SECTION cs;
#else
    pthread_t handle;
    pthread_cond_t cv;
    pthread_mutex_t mtx;
#endif
} pool_worker_t;

static pool_worker_t g_workers[BUF_MAX_THREADS];
static int g_pool_size = 0;
static int g_pool_created = 0;

/* Forward declarations for worker body */
static void pack_B_chunk(const double*B, double*pb, int kc, int nc, int n,
                          int jc, int pc, int j_start, int j_end);
static void pack_A(const double*A, double*pa, int mc, int kc, int n, int ic, int pc);
static void macro_kernel(const double*pa,const double*pb,
                          double*C,int mc,int nc,int kc,int n,int ic,int jc,int beta);

static void jc_worker_body(pool_worker_t *w) {
    const double *A = w->A, *B = w->B;
    double *C = w->C;
    int n = w->n;
    double *pa = w->pa, *pb = w->pb;

    for (int jc = w->j_start; jc < w->j_end; jc += w->nc_blk) {
        int nc = (jc + w->nc_blk <= w->j_end) ? w->nc_blk : w->j_end - jc;
        for (int pc = 0; pc < n; pc += w->kc_blk) {
            int kc = (pc + w->kc_blk <= n) ? w->kc_blk : n - pc;
            int beta = (pc > 0) ? 1 : 0;
            pack_B_chunk(B, pb, kc, nc, n, jc, pc, 0, nc);
            for (int ic = 0; ic < n; ic += w->mc_blk) {
                int mc = (ic + w->mc_blk <= n) ? w->mc_blk : n - ic;
                pack_A(A, pa, mc, kc, n, ic, pc);
                macro_kernel(pa, pb, C, mc, nc, kc, n, ic, jc, beta);
            }
            if (beta == 0 && g_use_nt) _mm_sfence();
        }
    }
}

#ifdef _WIN32
static unsigned __stdcall pool_thread_func(void *arg) {
#else
static void *pool_thread_func(void *arg) {
#endif
    pool_worker_t *w = (pool_worker_t *)arg;

    for (;;) {
        /* Spin-wait phase */
        for (int s = 0; s < POOL_SPIN_ITERS; s++) {
            int cmd = atomic_load_explicit(&w->command, memory_order_acquire);
            if (cmd != 0) goto got_work;
            _mm_pause();
        }

        /* Sleep phase */
#ifdef _WIN32
        EnterCriticalSection(&w->cs);
        while (atomic_load_explicit(&w->command, memory_order_relaxed) == 0)
            SleepConditionVariableCS(&w->cv, &w->cs, INFINITE);
        LeaveCriticalSection(&w->cs);
#else
        pthread_mutex_lock(&w->mtx);
        while (atomic_load_explicit(&w->command, memory_order_relaxed) == 0)
            pthread_cond_wait(&w->cv, &w->mtx);
        pthread_mutex_unlock(&w->mtx);
#endif

got_work:;
        int cmd = atomic_load_explicit(&w->command, memory_order_acquire);
        if (cmd == -1) break;

        jc_worker_body(w);

        atomic_store_explicit(&w->command, 0, memory_order_relaxed);
        atomic_store_explicit(&w->done, 1, memory_order_release);
    }

#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static void create_pool(int nworkers) {
    if (nworkers <= g_pool_size) return;
    if (nworkers > BUF_MAX_THREADS) nworkers = BUF_MAX_THREADS;

    for (int t = g_pool_size; t < nworkers; t++) {
        pool_worker_t *w = &g_workers[t];
        atomic_store(&w->command, 0);
        atomic_store(&w->done, 1);

#ifdef _WIN32
        InitializeConditionVariable(&w->cv);
        InitializeCriticalSection(&w->cs);
        w->handle = (HANDLE)_beginthreadex(NULL, 0, pool_thread_func, w, 0, NULL);
#else
        pthread_cond_init(&w->cv, NULL);
        pthread_mutex_init(&w->mtx, NULL);
        pthread_create(&w->handle, NULL, pool_thread_func, w);
#endif
    }
    g_pool_size = nworkers;
    g_pool_created = 1;
}

static void pool_dispatch_jc(const double *A, const double *B, double *C,
                              int n, int mc_blk, int kc_blk, int nthreads)
{
    int nworkers = nthreads - 1;
    create_pool(nworkers);
    ensure_bufs(nthreads);

    int nc_blk = g_nc_jc;
    int cols_per = ((n / nthreads + NR - 1) / NR) * NR;

    /* Set up and dispatch workers (threads 1..nthreads-1) */
    for (int t = 0; t < nworkers; t++) {
        pool_worker_t *w = &g_workers[t];

        while (!atomic_load_explicit(&w->done, memory_order_acquire))
            _mm_pause();

        w->A = A; w->B = B; w->C = C;
        w->n = n; w->mc_blk = mc_blk; w->kc_blk = kc_blk; w->nc_blk = nc_blk;
        w->j_start = (t + 1) * cols_per;
        w->j_end = (t + 2) * cols_per;
        if (w->j_end > n) w->j_end = n;
        w->pa = g_bufA[t + 1]; w->pb = g_bufB[t + 1];

        atomic_store_explicit(&w->done, 0, memory_order_relaxed);
        atomic_store_explicit(&w->command, 1, memory_order_release);

#ifdef _WIN32
        EnterCriticalSection(&w->cs);
        WakeConditionVariable(&w->cv);
        LeaveCriticalSection(&w->cs);
#else
        pthread_mutex_lock(&w->mtx);
        pthread_cond_signal(&w->cv);
        pthread_mutex_unlock(&w->mtx);
#endif
    }

    /* Master does thread 0's column range */
    {
        double *pa = g_bufA[0], *pb = g_bufB[0];
        int j_end_0 = cols_per;
        if (j_end_0 > n) j_end_0 = n;

        for (int jc = 0; jc < j_end_0; jc += nc_blk) {
            int nc = (jc + nc_blk <= j_end_0) ? nc_blk : j_end_0 - jc;
            for (int pc = 0; pc < n; pc += kc_blk) {
                int kc = (pc + kc_blk <= n) ? kc_blk : n - pc;
                int beta = (pc > 0) ? 1 : 0;
                pack_B_chunk(B, pb, kc, nc, n, jc, pc, 0, nc);
                for (int ic = 0; ic < n; ic += mc_blk) {
                    int mc = (ic + mc_blk <= n) ? mc_blk : n - ic;
                    pack_A(A, pa, mc, kc, n, ic, pc);
                    macro_kernel(pa, pb, C, mc, nc, kc, n, ic, jc, beta);
                }
            }
        }
    }

    /* Wait for all workers to finish */
    for (int t = 0; t < nworkers; t++) {
        pool_worker_t *w = &g_workers[t];
        while (!atomic_load_explicit(&w->done, memory_order_acquire))
            _mm_pause();
    }
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

    g_logical_cores = get_num_cpus();

    const char* tenv = getenv("COM6_THREADS");
    if (tenv && atoi(tenv) > 0) {
#ifdef _OPENMP
        omp_set_num_threads(atoi(tenv));
#endif
    } else {
#ifdef _OPENMP
        omp_set_num_threads(g_logical_cores);
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

    /* OpenMP warmup for IC-parallel path */
    int nt = g_logical_cores;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    #pragma omp parallel
    { volatile int x = omp_get_thread_num(); (void)x; }
#endif
    ensure_bufs(nt);

    /* Pre-create persistent pool for JC-parallel */
    if (nt > 1) create_pool(nt - 1);
}

static int get_mt_threads(int n) {
    const char* tenv = getenv("COM6_THREADS");
    if (tenv && atoi(tenv) > 0) return atoi(tenv);
    return g_logical_cores;
}

static int get_eff_threads(int n) {
    int nt = get_mt_threads(n);
    if (n <= 256 && g_phys_cores >= 2 && g_phys_cores < nt)
        return g_phys_cores;
    return nt;
}

/* ================================================================
 * MICRO-KERNEL: beta=1 (C += A*B) with C-prefetch + staggered A/B prefetch
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

        "prefetcht0 384(%[pA])\n\t"

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

        RANK1(0,0)
        RANK1(48,64)
        "prefetcht0 512(%[pB])\n\t"
        RANK1(96,128)
        RANK1(144,192)
        "prefetcht0 768(%[pA])\n\t"
        RANK1(192,256)
        RANK1(240,320)
        "prefetcht0 1024(%[pB])\n\t"
        RANK1(288,384)
        RANK1(336,448)
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
        /* Prefetch C rows — covers both cache lines when 8 doubles straddle */
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

        "prefetcht0 384(%[pA])\n\t"

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

        RANK1(0,0)
        RANK1(48,64)
        "prefetcht0 512(%[pB])\n\t"
        RANK1(96,128)
        RANK1(144,192)
        "prefetcht0 768(%[pA])\n\t"
        RANK1(192,256)
        RANK1(240,320)
        "prefetcht0 1024(%[pB])\n\t"
        RANK1(288,384)
        RANK1(336,448)
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
        : "r15","ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
          "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","memory"
    );
}

/* ================================================================
 * MICRO-KERNEL: beta=0 non-temporal (vmovntpd bypasses cache write-allocate)
 * Saves ~35% write bandwidth when C doesn't fit L3.
 * Caller must issue _mm_sfence() after all NT tiles in a PC iteration.
 * ================================================================ */
static void __attribute__((noinline))
micro_6x8_beta0_nt(int kc, const double* pA, const double* pB, double* C, int ldc)
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
        "prefetcht0 384(%[pA])\n\t"

#define RANK1_NT(AO,BO) \
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

        RANK1_NT(0,0)
        RANK1_NT(48,64)
        "prefetcht0 512(%[pB])\n\t"
        RANK1_NT(96,128)
        RANK1_NT(144,192)
        "prefetcht0 768(%[pA])\n\t"
        RANK1_NT(192,256)
        RANK1_NT(240,320)
        "prefetcht0 1024(%[pB])\n\t"
        RANK1_NT(288,384)
        RANK1_NT(336,448)
#undef RANK1_NT

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
        "vmovntpd %%ymm0,(%[C])\n\t" "vmovntpd %%ymm1,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovntpd %%ymm2,(%[C])\n\t" "vmovntpd %%ymm3,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovntpd %%ymm4,(%[C])\n\t" "vmovntpd %%ymm5,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovntpd %%ymm6,(%[C])\n\t" "vmovntpd %%ymm7,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovntpd %%ymm8,(%[C])\n\t" "vmovntpd %%ymm9,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovntpd %%ymm10,(%[C])\n\t" "vmovntpd %%ymm11,32(%[C])\n\t"

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
 * PACKING
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
                if(beta)          micro_6x8_beta1(kc,pA,pB,Cij,n);
                else if(g_use_nt) micro_6x8_beta0_nt(kc,pA,pB,Cij,n);
                else              micro_6x8_beta0(kc,pA,pB,Cij,n);
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
        else if(n <= 2048){*pKC=KC_DEEP;*pMC=get_mc_for_l2(KC_DEEP);}
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
    ensure_bufs(1);
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
 * IC-PARALLEL (OpenMP)
 * ================================================================ */
static void com6_multiply_ic(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C, int n,
                              int nthreads)
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
    ensure_bufs(nthreads);
    double* pb = g_bufB[0];

    #pragma omp parallel num_threads(nthreads)
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
                if (beta == 0 && g_use_nt) _mm_sfence();
            }
        }
    }
#else
    (void)nthreads;
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

static void c_set(double*C, int ldc, int r, int c, int h, const double*M){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<h;i++){
        double*ci=C+(size_t)(r+i)*ldc+c;
        const double*mi=M+(size_t)i*h;
        memcpy(ci, mi, (size_t)h*sizeof(double));
    }
}

static void com6_strassen(const double*A, const double*B, double*C, int n){
    int h=n/2;
    size_t hh=(size_t)h*h;
    double*T1=aa(hh), *T2=aa(hh), *M=aa(hh);
    if(!T1||!T2||!M){ if(T1)af(T1); if(T2)af(T2); if(M)af(M);
        int nt = get_mt_threads(n);
        com6_multiply_ic(A,B,C,n,nt); return; }

    g_strassen_depth++;

    /* M1 = (A00+A11)(B00+B11) → C00 = M1 (set), C11 = M1 (set) */
    quad_add(A,n,0,0, A,n,h,h, T1,h);
    quad_add(B,n,0,0, B,n,h,h, T2,h);
    com6_multiply(T1,T2,M,h);
    c_set(C,n,0,0,h,M);
    c_set(C,n,h,h,h,M);

    /* M2 = (A10+A11)B00 → C10 = M2 (set), C11 -= M2 */
    quad_add(A,n,h,0, A,n,h,h, T1,h);
    quad_copy(B,n,0,0,h,T2);
    com6_multiply(T1,T2,M,h);
    c_set(C,n,h,0,h,M);
    c_acc_sub(C,n,h,h,h,M);

    /* M3 = A00(B01-B11) → C01 = M3 (set), C11 += M3 */
    quad_copy(A,n,0,0,h,T1);
    quad_sub(B,n,0,h, B,n,h,h, T2,h);
    com6_multiply(T1,T2,M,h);
    c_set(C,n,0,h,h,M);
    c_acc_add(C,n,h,h,h,M);

    /* M4 = A11(B10-B00) → C00 += M4, C10 += M4 */
    quad_copy(A,n,h,h,h,T1);
    quad_sub(B,n,h,0, B,n,0,0, T2,h);
    com6_multiply(T1,T2,M,h);
    c_acc_add(C,n,0,0,h,M);
    c_acc_add(C,n,h,0,h,M);

    /* M5 = (A00+A01)B11 → C00 -= M5, C01 += M5 */
    quad_add(A,n,0,0, A,n,0,h, T1,h);
    quad_copy(B,n,h,h,h,T2);
    com6_multiply(T1,T2,M,h);
    c_acc_sub(C,n,0,0,h,M);
    c_acc_add(C,n,0,h,h,M);

    /* M6 = (A10-A00)(B00+B01) → C11 += M6 */
    quad_sub(A,n,h,0, A,n,0,0, T1,h);
    quad_add(B,n,0,0, B,n,0,h, T2,h);
    com6_multiply(T1,T2,M,h);
    c_acc_add(C,n,h,h,h,M);

    /* M7 = (A01-A11)(B10+B11) → C00 += M7 */
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
    int nthreads = get_mt_threads(n);
    if(n < 256 || nthreads <= 1){ com6_multiply_1t(A,B,C,n); return; }

    int eff_threads = get_eff_threads(n);

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

    g_use_nt = (n >= 4096);

    const char *fjc = getenv("COM6_FORCE_JC");
    const char *fic = getenv("COM6_FORCE_IC");
    if (fjc && atoi(fjc) == 1) {
        int mc_blk, kc_blk;
        get_blocking_mt(n, &mc_blk, &kc_blk);
        pool_dispatch_jc(A, B, C, n, mc_blk, kc_blk, nthreads);
        return;
    }
    if (fic && atoi(fic) == 1) {
        com6_multiply_ic(A, B, C, n, nthreads);
        return;
    }

    if (g_is_chiplet || g_l2_kb >= 512) {
        int mc_blk, kc_blk;
        get_blocking_mt(n, &mc_blk, &kc_blk);
        pool_dispatch_jc(A, B, C, n, mc_blk, kc_blk, eff_threads);
    } else {
        int mc_blk, kc_blk;
        get_blocking_mt(n, &mc_blk, &kc_blk);
        int kc = (n <= 1024) ? KC_SMALL : (n <= 2048) ? KC_DEEP : get_kc_large();
        int bpanel_per_thread = (n / eff_threads) * kc * (int)sizeof(double);
        int apanel = mc_blk * kc * (int)sizeof(double);
        int l2_bytes = g_l2_kb * 1024;

        if (n >= 4096) {
            com6_multiply_ic(A, B, C, n, nthreads);
        } else if (bpanel_per_thread + apanel <= (int)(l2_bytes * 0.95)) {
            pool_dispatch_jc(A, B, C, n, mc_blk, kc_blk, eff_threads);
        } else if (n >= 2048) {
            pool_dispatch_jc(A, B, C, n, mc_blk, kc_blk, eff_threads);
        } else {
            com6_multiply_ic(A, B, C, n, eff_threads);
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
    int nth_max = g_logical_cores;
#ifdef _OPENMP
    nth_max = omp_get_max_threads();
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
        int mt_threads = get_eff_threads(n);

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
            printf("%dx%d MT(%dT): %.1f ms (%.1f GF)\n",n,n,mt_threads,best*1000,(2.0*n*n*(double)n)/(best*1e9));
        }
        af(A);af(B);af(C);
        return 0;
    }

    int kc_l = get_kc_large();
    printf("COM6 v134 - Persistent Pool + NT Stores for Large Matrices\n");
    printf("Threads: %d (phys=%d, logical=%d, L2=%dKB, L3=%dKB, %s)\n",
           nth_max, g_phys_cores, g_logical_cores, g_l2_kb, g_l3_kb,
           g_is_chiplet ? "CHIPLET" : "MONOLITHIC");
    printf("KC: small=%d, large=%d%s\n", KC_SMALL, kc_l,
           g_kc_auto ? " (L2-auto)" : " (default)");
    printf("NC: 1T=%d, JC=%d (L3-auto)\n", g_nc_1t, g_nc_jc);
    printf("Strassen: max_depth=%d\n", g_max_strassen_depth);
    printf("Pool: %d workers (spin %d iters then sleep)\n",
           g_pool_size, POOL_SPIN_ITERS);
    printf("Dispatch: JC-pool when B+A fits L2, IC-OpenMP otherwise. NT stores for n>=4096\n\n");
    printf(" Size      |   1T (ms)  |   MT (ms)  |  GF(1T) |  GF(MT) |  T | Verify\n");
    printf("-----------|------------|------------|---------|---------|----|---------\n");

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
        int mt_threads = get_eff_threads(n);
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
            printf("%5dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %6.1f   | %2d | %s\n",n,n,best_1*1000,best_a*1000,gf1,gfa,mt_threads,v);
        else
            printf("%5dx%-5d | %10s | %8.1f ms | %6s   | %6.1f   | %2d | %s\n",n,n,"--",best_a*1000,"--",gfa,mt_threads,v);

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
    printf("\nv134: v130 + non-temporal stores (vmovntpd) for n>=4096.\n");
    printf("Persistent pool JC, OpenMP IC, Strassen depth=%d.\n",
           g_max_strassen_depth);
    printf("%s, phys=%d, logical=%d, L2=%dKB, L3=%dKB.\n",
           g_is_chiplet ? "Multi-CCD" : "Monolithic",
           g_phys_cores, g_logical_cores, g_l2_kb, g_l3_kb);
    printf("COM6_THREADS=N | COM6_STRASSEN_DEPTH=N | COM6_NO_STRASSEN=1\n");
    printf("COM6_FORCE_JC=1 | COM6_FORCE_IC=1 | COM6_PACE_MS=N\n");
    printf("Run individual: ./com6_v134 <size> [mt|1t]\n");

    return 0;
}
