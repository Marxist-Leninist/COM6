#define _GNU_SOURCE
/*
 * COM6 v118 - Chiplet-aware NUMA dispatch + physical-core threading
 * ==================================================================
 * v117 used IC-parallel at n>=4096 (shared B panel). On chiplet CPUs
 * (EPYC/Ryzen with multiple L3 groups), cross-CCD traffic killed perf:
 * 47 GF at 8192 on 32T EPYC 7282. v118 auto-detects chiplet topology
 * and uses JC-parallel (private B panels per thread) at ALL sizes.
 * Result: 209 GF at 8192 on 16T EPYC — 4.4x over v117.
 *
 * Also detects physical core count (skips HT for FMA-bound code),
 * auto-tunes MC blocking based on L2 size, and pins to physical cores.
 *
 * Dispatch (chiplet):    JC-parallel always (zero cross-CCD traffic)
 * Dispatch (monolithic): v117 rules (IC-par+pacing at 4096+)
 *
 * Env: COM6_THREADS=N | COM6_FORCE_JC=1 | COM6_FORCE_IC=1
 *      COM6_PACE_MS=N | COM6_PACE_MIN_N=N | COM6_USE_STRASSEN=1
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -lpthread \
 *       -o com6_v118 com6_v118.c -lm
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define MR  6
#define NR  8
#define NC  2048
#define NC_JC 1024
#define NC_MAX 2048
#define ALIGN 64

#ifndef _WIN32
#include <sched.h>
#endif

#define KC_SMALL 256
#define MC_SMALL 120
#define KC_LARGE 320
#define MC_LARGE 96
#define KC_MAX 320
#define MC_MAX 192
#define MC_MT_SMALL 48

#define MAX_THREADS 64
#define SPIN_ITERS 2000

/* v118: platform detection state (set once in pool_init) */
static int g_is_chiplet = 0;     /* 1 = multi-L3 (EPYC/Ryzen chiplet) */
static int g_l2_kb = 256;        /* L2 cache size per core in KB */
static int g_phys_cores = 0;     /* physical core count (no HT) */

static inline double* aa(size_t c){return(double*)_mm_malloc(c*sizeof(double),ALIGN);}
static inline void af(double*p){_mm_free(p);}

/* ================================================================
 * PERSISTENT THREAD POOL
 * ================================================================
 * Threads spin-wait (~1μs) then fall asleep on condvar.
 * Wake latency: ~1μs spin, ~10μs from condvar sleep.
 * vs OpenMP: ~50-100μs per fork-join.
 * ================================================================ */

typedef struct {
    /* Work descriptor — set by dispatcher before wake */
    void (*fn)(int tid, int nthreads, void* arg);
    void* arg;

    /* Synchronization */
    atomic_int generation;      /* bumped each dispatch */
    atomic_int threads_done;    /* count of threads that finished */
    int nthreads;

    /* Thread state */
    pthread_t threads[MAX_THREADS];
    int thread_count;
    int shutdown;

    /* Condvar for sleeping threads */
    pthread_mutex_t mutex;
    pthread_cond_t wake_cond;
    pthread_cond_t done_cond;
} thread_pool_t;

static thread_pool_t g_pool = {
    .generation = 0,
    .threads_done = 0,
    .thread_count = 0,
    .shutdown = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .wake_cond = PTHREAD_COND_INITIALIZER,
    .done_cond = PTHREAD_COND_INITIALIZER,
};

static void* pool_worker(void* arg) {
    thread_pool_t* pool = &g_pool;
    int tid = (int)(intptr_t)arg;
    int my_gen = 0;

    while (1) {
        /* Spin-wait for new work */
        int spins = 0;
        while (atomic_load_explicit(&pool->generation, memory_order_acquire) == my_gen) {
            if (pool->shutdown) return NULL;
            if (++spins < SPIN_ITERS) {
                _mm_pause(); _mm_pause(); _mm_pause(); _mm_pause();
            } else {
                /* Fall asleep on condvar */
                pthread_mutex_lock(&pool->mutex);
                while (atomic_load(&pool->generation) == my_gen && !pool->shutdown)
                    pthread_cond_wait(&pool->wake_cond, &pool->mutex);
                pthread_mutex_unlock(&pool->mutex);
                break;
            }
        }
        if (pool->shutdown) return NULL;
        my_gen = atomic_load(&pool->generation);

        /* Execute work */
        pool->fn(tid, pool->nthreads, pool->arg);

        /* Signal completion */
        int done = atomic_fetch_add(&pool->threads_done, 1) + 1;
        if (done == pool->nthreads) {
            pthread_mutex_lock(&pool->mutex);
            pthread_cond_signal(&pool->done_cond);
            pthread_mutex_unlock(&pool->mutex);
        }
    }
    return NULL;
}

static int get_num_cpus(void) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 4;
#endif
}

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

static int* get_physical_cpu_list(int* count) {
    static int cpus[256];
    int seen_cores[256]; int nseen = 0; *count = 0;
    for (int cpu = 0; cpu < 256 && *count < 256; cpu++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        FILE* f = fopen(path, "r");
        if (!f) break;
        int cid = -1; if (fscanf(f, "%d", &cid) != 1) cid = -1;
        fclose(f);
        int dup = 0;
        for (int i = 0; i < nseen; i++) if (seen_cores[i] == cid) { dup = 1; break; }
        if (!dup) { seen_cores[nseen++] = cid; cpus[(*count)++] = cpu; }
    }
    return cpus;
}
#endif

static void detect_platform(void) {
#ifndef _WIN32
    int l3g = detect_l3_groups();
    g_is_chiplet = (l3g > 1);
    g_phys_cores = detect_physical_cores();
    g_l2_kb = detect_l2_kb();
#else
    g_is_chiplet = 0;
    g_phys_cores = get_num_cpus();
    g_l2_kb = 256;
#endif
}

static void pool_init(void) {
    if (g_pool.thread_count > 0) return;
    detect_platform();

    int ncpus;
    const char* tenv = getenv("COM6_THREADS");
    if (tenv && atoi(tenv) > 0) {
        ncpus = atoi(tenv);
    } else if (g_is_chiplet) {
        ncpus = g_phys_cores;
    } else {
        ncpus = get_num_cpus();
    }
    if (ncpus > MAX_THREADS) ncpus = MAX_THREADS;
    g_pool.thread_count = ncpus;
    g_pool.nthreads = ncpus;

#ifndef _WIN32
    int phys_count = 0;
    int* phys_cpus = get_physical_cpu_list(&phys_count);
#endif

    for (int i = 0; i < ncpus; i++) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
#ifndef _WIN32
        if (g_is_chiplet && i < phys_count) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(phys_cpus[i], &cpuset);
            pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
        }
#endif
        pthread_create(&g_pool.threads[i], &attr, pool_worker, (void*)(intptr_t)i);
        pthread_attr_destroy(&attr);
    }
}

/* Dispatch work to all pool threads, wait for completion.
 * fn(tid, nthreads, arg) called on each thread. ~1μs dispatch. */
static void pool_dispatch(void (*fn)(int,int,void*), void* arg) {
    pool_init();
    thread_pool_t* pool = &g_pool;

    pool->fn = fn;
    pool->arg = arg;
    pool->nthreads = pool->thread_count;
    atomic_store(&pool->threads_done, 0);

    /* Wake all threads */
    atomic_fetch_add_explicit(&pool->generation, 1, memory_order_release);
    pthread_mutex_lock(&pool->mutex);
    pthread_cond_broadcast(&pool->wake_cond);
    pthread_mutex_unlock(&pool->mutex);

    /* Wait for all threads to finish */
    pthread_mutex_lock(&pool->mutex);
    while (atomic_load(&pool->threads_done) < pool->nthreads)
        pthread_cond_wait(&pool->done_cond, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

static void pool_shutdown(void) {
    if (g_pool.thread_count == 0) return;
    g_pool.shutdown = 1;
    atomic_fetch_add(&g_pool.generation, 1);
    pthread_mutex_lock(&g_pool.mutex);
    pthread_cond_broadcast(&g_pool.wake_cond);
    pthread_mutex_unlock(&g_pool.mutex);
    for (int i = 0; i < g_pool.thread_count; i++)
        pthread_join(g_pool.threads[i], NULL);
    g_pool.thread_count = 0;
    g_pool.shutdown = 0;
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
        /* Prefetch C rows into L1 */
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

        /* k+0 */
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

        /* k+1 */
        "vmovapd 64(%[pB]),%%ymm12\n\t" "vmovapd 96(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 48(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 56(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 64(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 72(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 80(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 88(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+2 */
        "vmovapd 128(%[pB]),%%ymm12\n\t" "vmovapd 160(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 96(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 104(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 112(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 120(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 128(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 136(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+3 */
        "vmovapd 192(%[pB]),%%ymm12\n\t" "vmovapd 224(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 144(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 152(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 160(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 168(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 176(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 184(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+4 — second prefetch wave (from v108) */
        "prefetcht0 1152(%[pA])\n\t" "prefetcht0 1536(%[pB])\n\t"
        "vmovapd 256(%[pB]),%%ymm12\n\t" "vmovapd 288(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 192(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 200(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 208(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 216(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 224(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 232(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+5 */
        "vmovapd 320(%[pB]),%%ymm12\n\t" "vmovapd 352(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 240(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 248(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 256(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 264(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 272(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 280(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+6 */
        "vmovapd 384(%[pB]),%%ymm12\n\t" "vmovapd 416(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 288(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 296(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 304(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 312(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 320(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 328(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+7 */
        "vmovapd 448(%[pB]),%%ymm12\n\t" "vmovapd 480(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 336(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 344(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 352(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 360(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 368(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 376(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        "addq $384,%[pA]\n\t" "addq $512,%[pB]\n\t"
        "decq %[kc8]\n\t" "jnz 1b\n\t"

        "3:\n\t"
        "testq %[kcr],%[kcr]\n\t"
        "jle 2f\n\t"
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
 * MICRO-KERNEL: beta=0 (C = A*B, just store, no load from C)
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

        /* k+0 through k+7 — same FMA body as beta1 */
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

        "vmovapd 64(%[pB]),%%ymm12\n\t" "vmovapd 96(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 48(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 56(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 64(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 72(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 80(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 88(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        "vmovapd 128(%[pB]),%%ymm12\n\t" "vmovapd 160(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 96(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 104(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 112(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 120(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 128(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 136(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        "vmovapd 192(%[pB]),%%ymm12\n\t" "vmovapd 224(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 144(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 152(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 160(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 168(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 176(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 184(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        "prefetcht0 1152(%[pA])\n\t" "prefetcht0 1536(%[pB])\n\t"
        "vmovapd 256(%[pB]),%%ymm12\n\t" "vmovapd 288(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 192(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 200(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 208(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 216(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 224(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 232(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        "vmovapd 320(%[pB]),%%ymm12\n\t" "vmovapd 352(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 240(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 248(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 256(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 264(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 272(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 280(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        "vmovapd 384(%[pB]),%%ymm12\n\t" "vmovapd 416(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 288(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 296(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 304(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 312(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 320(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 328(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        "vmovapd 448(%[pB]),%%ymm12\n\t" "vmovapd 480(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 336(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 344(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 352(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 360(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 368(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 376(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        "addq $384,%[pA]\n\t" "addq $512,%[pB]\n\t"
        "decq %[kc8]\n\t" "jnz 1b\n\t"

        "3:\n\t"
        "testq %[kcr],%[kcr]\n\t"
        "jle 2f\n\t"
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
        /* beta=0: just store, no load from C */
        "vmovupd %%ymm0,(%[C])\n\t"
        "vmovupd %%ymm1,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm2,(%[C])\n\t"
        "vmovupd %%ymm3,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm4,(%[C])\n\t"
        "vmovupd %%ymm5,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm6,(%[C])\n\t"
        "vmovupd %%ymm7,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm8,(%[C])\n\t"
        "vmovupd %%ymm9,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm10,(%[C])\n\t"
        "vmovupd %%ymm11,32(%[C])\n\t"

        : [pA]"+r"(pA),[pB]"+r"(pB),[kc8]"+r"(kc_8),[kcr]"+r"(kc_rem),[C]"+r"(C)
        : [ldc]"r"(ldc_bytes)
        : "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
          "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","memory"
    );
}

/* ================================================================
 * EDGE CASES: partial MR/NR tiles
 * ================================================================ */
static void micro_edge(int mr, int nr, int kc, const double*pA, const double*pB,
                        double*C, int ldc, int beta){
    double tmp[MR*NR] __attribute__((aligned(64)));
    if(beta){for(int i=0;i<mr;i++)for(int j=0;j<nr;j++)tmp[i*NR+j]=C[i*ldc+j];}
    else{memset(tmp,0,sizeof(tmp));}
    for(int k=0;k<kc;k++){
        for(int i=0;i<MR;i++){
            double a=pA[k*MR+i];
            for(int j=0;j<NR;j++) tmp[i*NR+j]+=a*pB[k*NR+j];
        }
    }
    for(int i=0;i<mr;i++)for(int j=0;j<nr;j++)C[i*ldc+j]=tmp[i*NR+j];
}

/* ================================================================
 * PACKING
 * ================================================================ */
static void pack_B(const double*B, double*pb, int kc, int nc, int n, int jc, int pc){
    for(int j=0;j<nc;j+=NR){
        int nr=(j+NR<=nc)?NR:nc-j;
        for(int k=0;k<kc;k++){
            const double*src=B+(size_t)(pc+k)*n+(jc+j);
            if(nr==NR){
                _mm256_storeu_pd(pb, _mm256_loadu_pd(src));
                _mm256_storeu_pd(pb+4, _mm256_loadu_pd(src+4));
            } else {
                int jj;for(jj=0;jj<nr;jj++)pb[jj]=src[jj];
                for(;jj<NR;jj++)pb[jj]=0.0;
            }
            pb+=NR;
        }
    }
}

/* Pack B-panel chunk [j_start..j_end) within a larger nc panel */
static void pack_B_chunk(const double*B, double*pb, int kc, int nc, int n,
                          int jc, int pc, int j_start, int j_end){
    for(int j=j_start;j<j_end;j+=NR){
        int nr=(j+NR<=nc)?NR:nc-j;
        double*dst=pb+(j/NR)*((size_t)NR*kc);
        for(int k=0;k<kc;k++){
            const double*src=B+(size_t)(pc+k)*n+(jc+j);
            if(nr==NR){
                _mm256_storeu_pd(dst, _mm256_loadu_pd(src));
                _mm256_storeu_pd(dst+4, _mm256_loadu_pd(src+4));
            } else {
                int jj;for(jj=0;jj<nr;jj++)dst[jj]=src[jj];
                for(;jj<NR;jj++)dst[jj]=0.0;
            }
            dst+=NR;
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
            /* 4x unrolled */
            for(;k+3<kc;k+=4){
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

static int get_mc_for_l2(int kc) {
    int l2_bytes = g_l2_kb * 1024;
    int mc = (int)(l2_bytes * 0.88 / (kc * sizeof(double)));
    mc = (mc / MR) * MR;
    if (mc < MR) mc = MR;
    if (mc > MC_MAX) mc = MC_MAX;
    return mc;
}

static void get_blocking(int n, int*pMC, int*pKC){
    if(n <= 1024){*pKC=KC_SMALL; *pMC = g_is_chiplet ? get_mc_for_l2(KC_SMALL) : MC_SMALL;}
    else{*pKC=KC_LARGE; *pMC = g_is_chiplet ? get_mc_for_l2(KC_LARGE) : MC_LARGE;}
}
static void get_blocking_mt(int n, int*pMC, int*pKC){
    if (g_is_chiplet) {
        *pKC = (n <= 1024) ? KC_SMALL : KC_LARGE;
        *pMC = get_mc_for_l2(*pKC);
    } else {
        if(n <= 1024){*pMC=MC_MT_SMALL;*pKC=KC_SMALL;}
        else if(n >= 4096){*pMC=MC_MT_SMALL;*pKC=KC_LARGE;}
        else{*pMC=MC_LARGE;*pKC=KC_LARGE;}
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
    double*pa=aa((size_t)MC_MAX*KC_MAX),*pb=aa((size_t)KC_MAX*NC);
    for(int jc=0;jc<n;jc+=NC){int nc=(jc+NC<=n)?NC:n-jc;
        for(int pc=0;pc<n;pc+=kc_blk){int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
            int beta=(pc>0)?1:0;
            pack_B(B,pb,kc,nc,n,jc,pc);
            for(int ic=0;ic<n;ic+=mc_blk){int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                pack_A(A,pa,mc,kc,n,ic,pc);
                macro_kernel(pa,pb,C,mc,nc,kc,n,ic,jc,beta);
    }}}
    af(pa);af(pb);
}

/* ================================================================
 * IC-PARALLEL (persistent pthreads pool, merged dispatch)
 * ================================================================
 * Single dispatch per pc-iteration:
 *   1. Parallel B-pack (each thread packs a chunk)
 *   2. Atomic spin-barrier (~10ns)
 *   3. Atomic work-stealing for ic-blocks
 * Total overhead: ~1μs per dispatch (vs ~100μs for OpenMP fork-join)
 * ================================================================ */

typedef struct {
    const double* A;
    const double* B;
    double* C;
    double* pb;         /* shared B-panel buffer */
    double** pa_bufs;   /* per-thread A-panel buffers */
    int n, mc_blk, kc_blk;
    int pace_ms;        /* v117: thermal pacing between NC blocks (0=disabled) */
    atomic_int barrier_count;   /* reusable spin barrier */
    atomic_int ic_next;         /* work-stealing counter */
} ic_work_t;

/* Reusable spin barrier: all threads call, all block until nthreads arrive.
 * Uses phase (even/odd) to allow reuse without reset. ~10ns on hot cache. */
static inline void spin_barrier(atomic_int* counter, int nthreads, int phase) {
    int target = (phase + 1) * nthreads;
    int val = atomic_fetch_add(counter, 1) + 1;
    if (val == target) return;  /* last thread arrives, done */
    while (atomic_load_explicit(counter, memory_order_acquire) < target)
        _mm_pause();
}

/* Single-dispatch IC-parallel worker: handles entire jc/pc/ic loop.
 * Threads stay alive across all pc-iterations — zero re-dispatch overhead.
 * B-pack is parallel, ic-loop uses atomic work-stealing. */
static void ic_worker(int tid, int nthreads, void* arg) {
    ic_work_t* w = (ic_work_t*)arg;
    double* pa = w->pa_bufs[tid];
    int n = w->n;
    int mc_blk = w->mc_blk;
    int kc_blk = w->kc_blk;
    int phase = 0;

    for (int jc = 0; jc < n; jc += NC) {
        int nc = (jc + NC <= n) ? NC : n - jc;
        int npanels = (nc + NR - 1) / NR;

        /* v117: thermal pacing — all threads sleep between NC blocks.
         * Better than v108's master-only Sleep (other threads spin-wait there).
         * Here everyone sleeps, reducing total thermal output during pause. */
        if (jc > 0 && w->pace_ms > 0) {
#ifdef _WIN32
            Sleep(w->pace_ms);
#else
            usleep((unsigned)(w->pace_ms) * 1000u);
#endif
            spin_barrier(&w->barrier_count, nthreads, phase++);
        }

        for (int pc = 0; pc < n; pc += kc_blk) {
            int kc = (pc + kc_blk <= n) ? kc_blk : n - pc;
            int beta = (pc > 0) ? 1 : 0;

            /* Phase 1: parallel B-pack */
            int panels_per = (npanels + nthreads - 1) / nthreads;
            int p0 = tid * panels_per;
            int p1 = p0 + panels_per;
            if (p1 > npanels) p1 = npanels;
            int j_start = p0 * NR;
            int j_end = p1 * NR;
            if (j_end > nc) j_end = nc;
            if (j_start < nc)
                pack_B_chunk(w->B, w->pb, kc, nc, n, jc, pc, j_start, j_end);

            /* Reset work counter before barrier (only thread 0) */
            if (tid == 0)
                atomic_store_explicit(&w->ic_next, 0, memory_order_relaxed);

            /* Spin barrier: wait for B-pack completion */
            spin_barrier(&w->barrier_count, nthreads, phase++);

            /* Phase 2: atomic work-stealing for ic-blocks */
            int ic;
            while ((ic = atomic_fetch_add(&w->ic_next, mc_blk)) < n) {
                int mc = (ic + mc_blk <= n) ? mc_blk : n - ic;
                pack_A(w->A, pa, mc, kc, n, ic, pc);
                macro_kernel(pa, w->pb, w->C, mc, nc, kc, n, ic, jc, beta);
            }

            /* Spin barrier: wait for all ic-work done before next pc iteration */
            spin_barrier(&w->barrier_count, nthreads, phase++);
        }
    }
}

static void com6_multiply_ic(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C, int n)
{
    pool_init();
    int nthreads = g_pool.thread_count;

    int mc_blk, kc_blk;
    get_blocking_mt(n, &mc_blk, &kc_blk);

    double** pa_bufs = (double**)malloc(nthreads * sizeof(double*));
    for (int t = 0; t < nthreads; t++) pa_bufs[t] = aa((size_t)MC_MAX * KC_MAX);
    double* pb = aa((size_t)KC_MAX * NC);

    /* v117: thermal pacing at n>=4096 (from v108).
     * Defaults: 4096 -> 150ms, 8192+ -> 400ms. */
    int pace_ms = 0;
    int pace_min_n = 4096;
    const char *min_env = getenv("COM6_PACE_MIN_N");
    if (min_env) { int v = atoi(min_env); if (v > 0 && v <= 16384) pace_min_n = v; }
    if (n >= pace_min_n) {
        pace_ms = (n >= 8192) ? 400 : 150;
        const char *env = getenv("COM6_PACE_MS");
        if (env) { int v = atoi(env); if (v >= 0 && v <= 10000) pace_ms = v; }
    }

    ic_work_t work;
    work.A = A; work.B = B; work.C = C;
    work.pb = pb; work.pa_bufs = pa_bufs;
    work.n = n; work.mc_blk = mc_blk; work.kc_blk = kc_blk;
    work.pace_ms = pace_ms;
    atomic_store(&work.barrier_count, 0);
    atomic_store(&work.ic_next, 0);

    pool_dispatch(ic_worker, &work);

    for (int t = 0; t < nthreads; t++) af(pa_bufs[t]);
    free(pa_bufs); af(pb);
}


/* ================================================================
 * JC-PARALLEL (persistent pthreads pool, private B-panels)
 * ================================================================ */
typedef struct {
    const double* A;
    const double* B;
    double* C;
    int n, mc_blk, kc_blk;
} jc_work_t;

static void jc_worker(int tid, int nthreads, void* arg) {
    jc_work_t* w = (jc_work_t*)arg;
    int n = w->n;

    int cols_per = ((n / nthreads + NR - 1) / NR) * NR;
    int j_start = tid * cols_per;
    int j_end = j_start + cols_per;
    if (j_end > n) j_end = n;
    if (j_start >= n) return;

    double* pa = aa((size_t)MC_MAX * KC_MAX);
    double* pb = aa((size_t)KC_MAX * NC_MAX);

    for (int jc = j_start; jc < j_end; jc += NC_JC) {
        int nc = (jc + NC_JC <= j_end) ? NC_JC : j_end - jc;
        for (int pc = 0; pc < n; pc += w->kc_blk) {
            int kc = (pc + w->kc_blk <= n) ? w->kc_blk : n - pc;
            int beta = (pc > 0) ? 1 : 0;
            pack_B_chunk(w->B, pb, kc, nc, n, jc, pc, 0, nc);
            for (int ic = 0; ic < n; ic += w->mc_blk) {
                int mc = (ic + w->mc_blk <= n) ? w->mc_blk : n - ic;
                pack_A(w->A, pa, mc, kc, n, ic, pc);
                macro_kernel(pa, pb, w->C, mc, nc, kc, n, ic, jc, beta);
            }
        }
    }
    af(pa); af(pb);
}

static void com6_multiply_jc(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C, int n,
                              int mc_blk, int kc_blk)
{
    jc_work_t work = { .A = A, .B = B, .C = C, .n = n,
                        .mc_blk = mc_blk, .kc_blk = kc_blk };
    pool_dispatch(jc_worker, &work);
}

/* ================================================================
 * STRASSEN: single-level for n>=8192 (Winograd variant)
 * ================================================================ */
static void sub_copy(const double*A, int lda, int r, int c, int h, double*out){
    for(int i=0;i<h;i++)
        memcpy(out+(size_t)i*h, A+(size_t)(r+i)*lda+c, (size_t)h*sizeof(double));
}
static void mat_add(const double*X, const double*Y, double*out, int h){
    size_t nn=(size_t)h*h; size_t i=0;
    for(;i+3<nn;i+=4){
        __m256d a=_mm256_loadu_pd(X+i), b=_mm256_loadu_pd(Y+i);
        _mm256_storeu_pd(out+i,_mm256_add_pd(a,b));
    }
    for(;i<nn;i++) out[i]=X[i]+Y[i];
}
static void mat_sub(const double*X, const double*Y, double*out, int h){
    size_t nn=(size_t)h*h; size_t i=0;
    for(;i+3<nn;i+=4){
        __m256d a=_mm256_loadu_pd(X+i), b=_mm256_loadu_pd(Y+i);
        _mm256_storeu_pd(out+i,_mm256_sub_pd(a,b));
    }
    for(;i<nn;i++) out[i]=X[i]-Y[i];
}
static void sub_acc(double*C, int ldc, int r, int c, int h, const double*M){
    for(int i=0;i<h;i++){
        double*row=C+(size_t)(r+i)*ldc+c;
        const double*mr=M+(size_t)i*h;
        size_t j=0;
        for(;j+3<(size_t)h;j+=4){
            __m256d a=_mm256_loadu_pd(row+j), b=_mm256_loadu_pd(mr+j);
            _mm256_storeu_pd(row+j,_mm256_add_pd(a,b));
        }
        for(;j<(size_t)h;j++) row[j]+=mr[j];
    }
}
static void sub_write(double*C, int ldc, int r, int c, int h, const double*M){
    for(int i=0;i<h;i++)
        memcpy(C+(size_t)(r+i)*ldc+c, M+(size_t)i*h, (size_t)h*sizeof(double));
}

/* Forward declaration */
static void com6_multiply(const double*__restrict__ A,
                           const double*__restrict__ B,
                           double*__restrict__ C, int n);

static void com6_strassen(const double*A, const double*B, double*C, int n){
    int h=n/2; size_t hh=(size_t)h*h;
    double *A11=aa(hh),*A12=aa(hh),*A21=aa(hh),*A22=aa(hh);
    double *B11=aa(hh),*B12=aa(hh),*B21=aa(hh),*B22=aa(hh);
    sub_copy(A,n,0,0,h,A11);sub_copy(A,n,0,h,h,A12);
    sub_copy(A,n,h,0,h,A21);sub_copy(A,n,h,h,h,A22);
    sub_copy(B,n,0,0,h,B11);sub_copy(B,n,0,h,h,B12);
    sub_copy(B,n,h,0,h,B21);sub_copy(B,n,h,h,h,B22);

    double *S1=aa(hh),*S2=aa(hh),*S3=aa(hh),*S4=aa(hh);
    double *T1=aa(hh),*T2=aa(hh),*T3=aa(hh),*T4=aa(hh);
    mat_add(A21,A22,S1,h); mat_sub(S1,A11,S2,h);
    mat_sub(A11,A21,S3,h); mat_sub(A12,S2,S4,h);
    mat_sub(B12,B11,T1,h); mat_sub(B22,T1,T2,h);
    mat_sub(B22,B12,T3,h); mat_sub(T2,B21,T4,h);

    double *M1=aa(hh),*M2=aa(hh),*M3=aa(hh),*M4=aa(hh);
    double *M5=aa(hh),*M6=aa(hh),*M7=aa(hh);
    com6_multiply(A11,B11,M1,h); com6_multiply(A12,B21,M2,h);
    com6_multiply(S4,B22,M3,h);  com6_multiply(A22,T4,M4,h);
    com6_multiply(S1,T1,M5,h);  com6_multiply(S2,T2,M6,h);
    com6_multiply(S3,T3,M7,h);

    af(A11);af(A12);af(A21);af(A22);
    af(B11);af(B12);af(B21);af(B22);
    af(S1);af(S2);af(S3);af(S4);
    af(T1);af(T2);af(T3);af(T4);

    double *U=aa(hh),*tmp=aa(hh);
    mat_add(M1,M6,U,h);
    mat_add(M1,M2,tmp,h);       sub_write(C,n,0,0,h,tmp);
    mat_add(U,M5,tmp,h); mat_add(tmp,M3,tmp,h); sub_write(C,n,0,h,h,tmp);
    mat_add(U,M7,tmp,h); mat_sub(tmp,M4,tmp,h); sub_write(C,n,h,0,h,tmp);
    mat_add(U,M7,tmp,h); mat_add(tmp,M5,tmp,h); sub_write(C,n,h,h,h,tmp);

    af(tmp);af(U);
    af(M1);af(M2);af(M3);af(M4);af(M5);af(M6);af(M7);
}

/* ================================================================
 * DISPATCH
 * ================================================================ */
static void com6_multiply(const double*__restrict__ A,
                           const double*__restrict__ B,
                           double*__restrict__ C, int n)
{
    pool_init();
    int nthreads = g_pool.thread_count;
    if (n < 512 || nthreads <= 1) { com6_multiply_1t(A,B,C,n); return; }

    const char *strassen_env = getenv("COM6_USE_STRASSEN");
    if (strassen_env && atoi(strassen_env) == 1 && n >= 8192 && (n & 1) == 0) {
        com6_strassen(A, B, C, n);
        return;
    }

    const char *fjc = getenv("COM6_FORCE_JC");
    const char *fic = getenv("COM6_FORCE_IC");
    int use_jc;

    if (fjc && atoi(fjc) == 1) use_jc = 1;
    else if (fic && atoi(fic) == 1 && n >= 4096) use_jc = 0;
    else if (g_is_chiplet) use_jc = 1;
    else use_jc = (n < 4096);

    if (!use_jc) {
        com6_multiply_ic(A, B, C, n);
    } else {
        int mc_blk, kc_blk;
        get_blocking_mt(n, &mc_blk, &kc_blk);
        com6_multiply_jc(A, B, C, n, mc_blk, kc_blk);
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
    pool_init();
    int nth = g_pool.thread_count;

    if(argc >= 2){
        int n=atoi(argv[1]);
        if(n<64||n>16384){printf("Size must be 64..16384\n");return 1;}
        int mode=0;
        if(argc>=3){
            if(strcmp(argv[2],"1t")==0) mode=1;
            else if(strcmp(argv[2],"mt")==0) mode=2;
        }
        size_t nn=(size_t)n*n;
        double*A=aa(nn),*B=aa(nn),*C=aa(nn);
        srand(42);randf(A,n);randf(B,n);

        int runs=(n<=512)?7:(n<=1024)?5:(n<=2048)?3:2;

        if(mode==0||mode==1){
            com6_multiply_1t(A,B,C,n); /* warmup */
            double best=1e30;
            for(int r=0;r<runs;r++){double t0=now();com6_multiply_1t(A,B,C,n);double t=now()-t0;if(t<best)best=t;}
            printf("%dx%d 1T: %.1f ms (%.1f GF)\n",n,n,best*1000,(2.0*n*n*(double)n)/(best*1e9));
        }
        if(mode==0||mode==2){
            com6_multiply(A,B,C,n); /* warmup */
            double best=1e30;
            for(int r=0;r<runs;r++){double t0=now();com6_multiply(A,B,C,n);double t=now()-t0;if(t<best)best=t;}
            printf("%dx%d MT(%dT): %.1f ms (%.1f GF)\n",n,n,nth,best*1000,(2.0*n*n*(double)n)/(best*1e9));
        }
        af(A);af(B);af(C);
        pool_shutdown();
        return 0;
    }

    printf("COM6 v118 - chiplet-aware NUMA dispatch + physical-core threading\n");
    printf("Threads: %d (physical cores: %d, L2: %dKB, %s)\n",
           nth, g_phys_cores, g_l2_kb, g_is_chiplet ? "CHIPLET" : "MONOLITHIC");
    printf("Dispatch: %s\n\n", g_is_chiplet ?
           "JC-par all sizes (chiplet detected)" :
           "4096+ IC-par+pacing | <4096 JC-par | Strassen via COM6_USE_STRASSEN=1");
    printf(" Size      |   1T (ms)  |   MT (ms)  |  GF(1T) |  GF(MT) | Verify\n");
    printf("-----------|------------|------------|---------|---------|-------\n");

    int sizes[]={8192,4096,2048,1024,512,256};
    int nsizes=6;

    for(int si=0;si<nsizes;si++){
        int n=sizes[si];
        size_t nn=(size_t)n*n;
        double*A=aa(nn),*B=aa(nn),*C1=NULL,*C2=aa(nn);
        srand(42);randf(A,n);randf(B,n);

        int do_1t = g_is_chiplet ? (n<=4096) : (n<=2048);
        int runs=(n<=512)?7:(n<=1024)?5:(n<=2048)?3:2;
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
            printf("%4dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %6.1f   | %s\n",n,n,best_1*1000,best_a*1000,gf1,gfa,v);
        else
            printf("%4dx%-5d | %10s | %8.1f ms | %6s   | %6.1f   | %s\n",n,n,"--",best_a*1000,"--",gfa,v);

        af(A);af(B);if(C1)af(C1);af(C2);

        /* 4s cooldown between sizes to combat thermal throttling */
        if(si < nsizes-1){
#ifdef _WIN32
            Sleep(4000);
#else
            struct timespec ts = {4, 0};
            nanosleep(&ts, NULL);
#endif
        }
    }
    printf("\nv118: chiplet-aware NUMA dispatch. %s, %d physical cores, L2=%dKB.\n",
           g_is_chiplet ? "Multi-CCD" : "Monolithic", g_phys_cores, g_l2_kb);
    printf("COM6_THREADS=N | COM6_FORCE_JC=1 | COM6_FORCE_IC=1\n");
    printf("Run individual: ./com6_v118 <size> [mt|1t]\n");

    pool_shutdown();
    return 0;
}
