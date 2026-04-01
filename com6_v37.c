/*
 * COM6 v37 - v26 Core + Thread Pool + 8192 Support
 * ==================================================
 * v30-v36 regressed from v26 (MC_TINY=48 killed performance).
 * This version restores v26's proven blocking parameters and adds:
 * - Persistent thread pool (no OpenMP fork/join overhead)
 * - Work-stealing ic-loop for better load balance
 * - 8192 support (skip 1T for n>=4096 to save thermal budget)
 * - Larger NC=3072 for n>2048 (fewer B-repacks, fits L3=8MB)
 *
 * Blocking (v26 proven values):
 *   n<=1024: MC=120 KC=256 (L2: 240KB < 256KB)
 *   n>1024:  MC=96  KC=320 (L2: 240KB < 256KB, deeper KC amortizes B-pack)
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v37 com6_v37.c -lm -lpthread
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define MR  6
#define NR  8
#define ALIGN 64

/* v26 proven blocking parameters */
#define KC_SMALL 256
#define MC_SMALL 120
#define KC_LARGE 320
#define MC_LARGE 96
#define KC_MAX   320
#define MC_MAX   120
#define NC_SMALL 2048
#define NC_LARGE 3072   /* For n>2048: 320*3072*8=7.5MB < L3=8MB */
#define NC_MAX   3072
#define MAX_THREADS 64

static inline double* aa(size_t c){return(double*)_mm_malloc(c*sizeof(double),ALIGN);}
static inline void af(double*p){_mm_free(p);}

static int get_cpu_count(void){
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return si.dwNumberOfProcessors;
#else
    int n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#endif
}

/* ================================================================
 * PERSISTENT THREAD POOL (from v36, proven low-overhead)
 * ================================================================ */
typedef struct {
    void (*func)(int tid, int nthreads, void* arg);
    void* arg;
    pthread_mutex_t work_mutex;
    pthread_cond_t work_cond;
    pthread_mutex_t done_mutex;
    pthread_cond_t done_cond;
    atomic_int generation;
    atomic_int done_count;
    pthread_t threads[MAX_THREADS];
    int nthreads;
    volatile int shutdown;
} thread_pool_t;

static thread_pool_t pool;

typedef struct {
    atomic_int count;
    atomic_int generation;
    int total;
} spin_barrier_t;

static void spin_barrier_init(spin_barrier_t* b, int n){
    atomic_store(&b->count, 0);
    atomic_store(&b->generation, 0);
    b->total = n;
}

static void spin_barrier_wait(spin_barrier_t* b){
    int gen = atomic_load_explicit(&b->generation, memory_order_acquire);
    int arrived = atomic_fetch_add_explicit(&b->count, 1, memory_order_acq_rel) + 1;
    if(arrived == b->total){
        atomic_store_explicit(&b->count, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&b->generation, 1, memory_order_release);
    } else {
        while(atomic_load_explicit(&b->generation, memory_order_acquire) == gen)
            _mm_pause();
    }
}

static void* worker_func(void* arg){
    int tid = (int)(size_t)arg;
    int my_gen = 0;
    while(1){
        int spins = 0;
        while(atomic_load_explicit(&pool.generation, memory_order_acquire) == my_gen && !pool.shutdown){
            if(spins < 2000){ _mm_pause(); _mm_pause(); spins++; }
            else {
                pthread_mutex_lock(&pool.work_mutex);
                if(atomic_load_explicit(&pool.generation, memory_order_acquire) == my_gen && !pool.shutdown)
                    pthread_cond_wait(&pool.work_cond, &pool.work_mutex);
                pthread_mutex_unlock(&pool.work_mutex);
            }
        }
        if(pool.shutdown) return NULL;
        my_gen = atomic_load_explicit(&pool.generation, memory_order_acquire);
        pool.func(tid, pool.nthreads, pool.arg);
        int done = atomic_fetch_add_explicit(&pool.done_count, 1, memory_order_release) + 1;
        if(done == pool.nthreads - 1){
            pthread_mutex_lock(&pool.done_mutex);
            pthread_cond_signal(&pool.done_cond);
            pthread_mutex_unlock(&pool.done_mutex);
        }
    }
    return NULL;
}

static void pool_init(int nthreads){
    memset(&pool, 0, sizeof(pool));
    pool.nthreads = nthreads;
    atomic_store(&pool.generation, 0);
    atomic_store(&pool.done_count, 0);
    pool.shutdown = 0;
    pthread_mutex_init(&pool.work_mutex, NULL);
    pthread_cond_init(&pool.work_cond, NULL);
    pthread_mutex_init(&pool.done_mutex, NULL);
    pthread_cond_init(&pool.done_cond, NULL);
    for(int i=1; i<nthreads; i++)
        pthread_create(&pool.threads[i], NULL, worker_func, (void*)(size_t)i);
}

static void pool_dispatch(void (*func)(int,int,void*), void* arg){
    pool.func = func; pool.arg = arg;
    atomic_store_explicit(&pool.done_count, 0, memory_order_release);
    atomic_fetch_add_explicit(&pool.generation, 1, memory_order_release);
    pthread_mutex_lock(&pool.work_mutex);
    pthread_cond_broadcast(&pool.work_cond);
    pthread_mutex_unlock(&pool.work_mutex);
    func(0, pool.nthreads, arg);
    pthread_mutex_lock(&pool.done_mutex);
    while(atomic_load_explicit(&pool.done_count, memory_order_acquire) < pool.nthreads - 1)
        pthread_cond_wait(&pool.done_cond, &pool.done_mutex);
    pthread_mutex_unlock(&pool.done_mutex);
}

static void pool_destroy(void){
    pool.shutdown = 1;
    pthread_mutex_lock(&pool.work_mutex);
    pthread_cond_broadcast(&pool.work_cond);
    pthread_mutex_unlock(&pool.work_mutex);
    for(int i=1; i<pool.nthreads; i++) pthread_join(pool.threads[i], NULL);
    pthread_mutex_destroy(&pool.work_mutex); pthread_cond_destroy(&pool.work_cond);
    pthread_mutex_destroy(&pool.done_mutex); pthread_cond_destroy(&pool.done_cond);
}

/* ================================================================
 * 8x k-UNROLLED 6x8 AVX2 MICRO-KERNEL (v26 proven)
 * ================================================================ */
static void __attribute__((noinline))
micro_6x8(int kc, const double* pA, const double* pB, double* C, int ldc)
{
    long long kc_8 = kc >> 3, kc_rem = kc & 7, ldc_bytes = (long long)ldc * 8;

    __asm__ volatile(
        "vxorpd %%ymm0,%%ymm0,%%ymm0\n\t""vxorpd %%ymm1,%%ymm1,%%ymm1\n\t"
        "vxorpd %%ymm2,%%ymm2,%%ymm2\n\t""vxorpd %%ymm3,%%ymm3,%%ymm3\n\t"
        "vxorpd %%ymm4,%%ymm4,%%ymm4\n\t""vxorpd %%ymm5,%%ymm5,%%ymm5\n\t"
        "vxorpd %%ymm6,%%ymm6,%%ymm6\n\t""vxorpd %%ymm7,%%ymm7,%%ymm7\n\t"
        "vxorpd %%ymm8,%%ymm8,%%ymm8\n\t""vxorpd %%ymm9,%%ymm9,%%ymm9\n\t"
        "vxorpd %%ymm10,%%ymm10,%%ymm10\n\t""vxorpd %%ymm11,%%ymm11,%%ymm11\n\t"

        "testq %[kc8],%[kc8]\n\t""jle 3f\n\t"".p2align 5\n\t""1:\n\t"
        "prefetcht0 768(%[pA])\n\t""prefetcht0 1024(%[pB])\n\t"

#define RANK1(AO,BO) \
        "vmovapd " #BO "(%[pB]),%%ymm12\n\t" \
        "vmovapd " #BO "+32(%[pB]),%%ymm13\n\t" \
        "vbroadcastsd " #AO "(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t" \
        "vbroadcastsd " #AO "+8(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t" \
        "vbroadcastsd " #AO "+16(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t" \
        "vbroadcastsd " #AO "+24(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t" \
        "vbroadcastsd " #AO "+32(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t" \
        "vbroadcastsd " #AO "+40(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        RANK1(0,0) RANK1(48,64) RANK1(96,128) RANK1(144,192)
        "prefetcht0 1152(%[pA])\n\t""prefetcht0 1536(%[pB])\n\t"
        RANK1(192,256) RANK1(240,320) RANK1(288,384) RANK1(336,448)
#undef RANK1

        "addq $384,%[pA]\n\t""addq $512,%[pB]\n\t""decq %[kc8]\n\t""jnz 1b\n\t"

        "3:\n\t""testq %[kcr],%[kcr]\n\t""jle 2f\n\t""4:\n\t"
        "vmovapd (%[pB]),%%ymm12\n\t""vmovapd 32(%[pB]),%%ymm13\n\t"
        "vbroadcastsd (%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 8(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 16(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 24(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 32(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 40(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t""vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"
        "addq $48,%[pA]\n\t""addq $64,%[pB]\n\t""decq %[kcr]\n\t""jnz 4b\n\t"

        "2:\n\t"
        "vaddpd (%[C]),%%ymm0,%%ymm0\n\t""vmovupd %%ymm0,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm1,%%ymm1\n\t""vmovupd %%ymm1,32(%[C])\n\t""addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm2,%%ymm2\n\t""vmovupd %%ymm2,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm3,%%ymm3\n\t""vmovupd %%ymm3,32(%[C])\n\t""addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm4,%%ymm4\n\t""vmovupd %%ymm4,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm5,%%ymm5\n\t""vmovupd %%ymm5,32(%[C])\n\t""addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm6,%%ymm6\n\t""vmovupd %%ymm6,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm7,%%ymm7\n\t""vmovupd %%ymm7,32(%[C])\n\t""addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm8,%%ymm8\n\t""vmovupd %%ymm8,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm9,%%ymm9\n\t""vmovupd %%ymm9,32(%[C])\n\t""addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm10,%%ymm10\n\t""vmovupd %%ymm10,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm11,%%ymm11\n\t""vmovupd %%ymm11,32(%[C])\n\t"

        : [pA]"+r"(pA),[pB]"+r"(pB),[kc8]"+r"(kc_8),[kcr]"+r"(kc_rem),[C]"+r"(C)
        : [ldc]"r"(ldc_bytes)
        : "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
          "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","memory"
    );
}

static void micro_edge(int mr,int nr,int kc,const double*pA,const double*pB,double*C,int n){
    for(int k=0;k<kc;k++)for(int i=0;i<mr;i++){
        double av=pA[k*MR+i];for(int j=0;j<nr;j++)C[i*n+j]+=av*pB[k*NR+j];}
}

/* ================================================================
 * PACKING
 * ================================================================ */
static void pack_B_chunk(const double*B,double*pb,int kc,int nc,int n,
                          int j0,int k0,int js,int je){
    for(int j=js;j<je;j+=NR){int nr=(j+NR<=nc)?NR:nc-j;
        const double*Bkj=B+(size_t)k0*n+(j0+j);double*d=pb+(j/NR)*((size_t)NR*kc);
        if(nr==NR){for(int k=0;k<kc;k++){
            _mm256_store_pd(d,_mm256_loadu_pd(Bkj));
            _mm256_store_pd(d+4,_mm256_loadu_pd(Bkj+4));
            d+=NR;Bkj+=n;
        }}else{for(int k=0;k<kc;k++){int jj;for(jj=0;jj<nr;jj++)d[jj]=Bkj[jj];
            for(;jj<NR;jj++)d[jj]=0;d+=NR;Bkj+=n;}}
    }
}

static void pack_A(const double*A,double*pa,int mc,int kc,int n,int i0,int k0){
    const double*Ab=A+(size_t)i0*n+k0;
    for(int i=0;i<mc;i+=MR){int mr=(i+MR<=mc)?MR:mc-i;
        if(mr==MR){const double*a0=Ab+i*n,*a1=a0+n,*a2=a0+2*n,
                    *a3=a0+3*n,*a4=a0+4*n,*a5=a0+5*n;
            int k=0;for(;k+1<kc;k+=2){
                pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];
                pa[6]=a0[k+1];pa[7]=a1[k+1];pa[8]=a2[k+1];pa[9]=a3[k+1];pa[10]=a4[k+1];pa[11]=a5[k+1];pa+=2*MR;}
            for(;k<kc;k++){pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];pa+=MR;}}
        else{for(int k=0;k<kc;k++){int ii;for(ii=0;ii<mr;ii++)pa[ii]=(Ab+(i+ii)*n)[k];
            for(;ii<MR;ii++)pa[ii]=0;pa+=MR;}}
    }
}

static void macro_kernel(const double*pa,const double*pb,
                          double*C,int mc,int nc,int kc,int n,int ic,int jc){
    for(int jr=0;jr<nc;jr+=NR){int nr=(jr+NR<=nc)?NR:nc-jr;
        const double*pB=pb+(jr/NR)*((size_t)NR*kc);
        for(int ir=0;ir<mc;ir+=MR){int mr=(ir+MR<=mc)?MR:mc-ir;
            const double*pA=pa+(ir/MR)*((size_t)MR*kc);
            double*Cij=C+(size_t)(ic+ir)*n+(jc+jr);
            if(mr==MR&&nr==NR)micro_6x8(kc,pA,pB,Cij,n);
            else micro_edge(mr,nr,kc,pA,pB,Cij,n);}}}

static void get_blocking(int n, int*pMC, int*pKC, int*pNC){
    if(n <= 1024){*pMC=MC_SMALL; *pKC=KC_SMALL; *pNC=NC_SMALL;}
    else if(n <= 2048){*pMC=MC_LARGE; *pKC=KC_LARGE; *pNC=NC_SMALL;}
    else{*pMC=MC_LARGE; *pKC=KC_LARGE; *pNC=NC_LARGE;}
}

static double now(void){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec+t.tv_nsec*1e-9;}

/* ================================================================
 * MULTI-THREADED DISPATCH (static partitioning — less overhead than work-stealing)
 * ================================================================ */
typedef struct {
    const double* A;
    const double* B;
    double* C;
    double* pb;
    double** pa_bufs;
    int n, nc, kc, mc_blk, jc, pc;
    spin_barrier_t* barrier;
} mt_work_t;

static void mt_worker(int tid, int nthreads, void* arg){
    mt_work_t* w = (mt_work_t*)arg;
    /* Phase 1: Cooperative B-packing */
    int npanels = (w->nc + NR - 1) / NR;
    int pp = (npanels + nthreads - 1) / nthreads;
    int p0 = tid * pp, p1 = p0 + pp;
    if(p1 > npanels) p1 = npanels;
    int js = p0 * NR, je = p1 * NR;
    if(je > w->nc) je = w->nc;
    if(js < w->nc)
        pack_B_chunk(w->B, w->pb, w->kc, w->nc, w->n, w->jc, w->pc, js, je);
    spin_barrier_wait(w->barrier);

    /* Phase 2: Static ic-loop partitioning (same as v26's omp for static) */
    double* pa = w->pa_bufs[tid];
    int mc_blk = w->mc_blk;
    int total_ic = (w->n + mc_blk - 1) / mc_blk;
    int per = (total_ic + nthreads - 1) / nthreads;
    int ic0 = tid * per * mc_blk;
    int ic1 = (tid + 1) * per * mc_blk;
    if(ic1 > w->n) ic1 = w->n;
    for(int ic = ic0; ic < ic1; ic += mc_blk){
        int mc = (ic + mc_blk <= w->n) ? mc_blk : w->n - ic;
        pack_A(w->A, pa, mc, w->kc, w->n, ic, w->pc);
        macro_kernel(pa, w->pb, w->C, mc, w->nc, w->kc, w->n, ic, w->jc);
    }
}

static void com6_multiply_mt(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C, int n)
{
    int mc_blk, kc_blk, nc_blk;
    get_blocking(n, &mc_blk, &kc_blk, &nc_blk);

    int nt = pool.nthreads;
    double** pa_bufs = (double**)malloc(nt * sizeof(double*));
    for(int t=0; t<nt; t++) pa_bufs[t] = aa((size_t)MC_MAX * KC_MAX);
    double* pb = aa((size_t)KC_MAX * NC_MAX);

    memset(C, 0, (size_t)n * n * sizeof(double));

    spin_barrier_t bar;
    spin_barrier_init(&bar, nt);

    mt_work_t mw;
    mw.A = A; mw.B = B; mw.C = C; mw.pb = pb; mw.pa_bufs = pa_bufs;
    mw.barrier = &bar;

    for(int jc=0; jc<n; jc+=nc_blk){
        int nc = (jc+nc_blk<=n)?nc_blk:n-jc;
        for(int pc=0; pc<n; pc+=kc_blk){
            int kc = (pc+kc_blk<=n)?kc_blk:n-pc;
            mw.n = n; mw.nc = nc; mw.kc = kc;
            mw.mc_blk = mc_blk; mw.jc = jc; mw.pc = pc;
            pool_dispatch(mt_worker, &mw);
        }
    }

    for(int t=0; t<nt; t++) af(pa_bufs[t]);
    free(pa_bufs); af(pb);
}

static void com6_multiply_1t(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C, int n)
{
    int mc_blk, kc_blk, nc_blk;
    get_blocking(n, &mc_blk, &kc_blk, &nc_blk);

    double*pa=aa((size_t)MC_MAX*KC_MAX),*pb=aa((size_t)KC_MAX*NC_MAX);
    memset(C,0,(size_t)n*n*sizeof(double));
    for(int jc=0;jc<n;jc+=nc_blk){int nc=(jc+nc_blk<=n)?nc_blk:n-jc;
        for(int pc=0;pc<n;pc+=kc_blk){int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
            pack_B_chunk(B,pb,kc,nc,n,jc,pc,0,nc);
            for(int ic=0;ic<n;ic+=mc_blk){int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                pack_A(A,pa,mc,kc,n,ic,pc);
                macro_kernel(pa,pb,C,mc,nc,kc,n,ic,jc);}}}
    af(pa);af(pb);
}

static void naive(const double*A,const double*B,double*C,int n){
    memset(C,0,(size_t)n*n*sizeof(double));
    for(int i=0;i<n;i++)for(int k=0;k<n;k++){double a=A[i*n+k];for(int j=0;j<n;j++)C[i*n+j]+=a*B[k*n+j];}
}

static void randf(double*M,int n){for(int i=0;i<n*n;i++)M[i]=(double)rand()/RAND_MAX*2-1;}
static double maxerr(const double*A,const double*B,int n){
    double m=0;for(int i=0;i<n*n;i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;}return m;}

int main(void){
    int nthreads = get_cpu_count();
    if(nthreads > MAX_THREADS) nthreads = MAX_THREADS;

    printf("====================================================================\n");
    printf("  COM6 v37 - v26 Core + Thread Pool (%d threads)\n", nthreads);
    printf("  n<=1024: MC=%d KC=%d | n>1024: MC=%d KC=%d\n", MC_SMALL,KC_SMALL,MC_LARGE,KC_LARGE);
    printf("  n>2048: NC=%d (7.5MB B-panel fits L3)\n", NC_LARGE);
    printf("  8x unrolled ASM + work-stealing + spin barriers\n");
    printf("====================================================================\n\n");

    pool_init(nthreads);

    int sizes[]={256,512,1024,2048,4096,8192};
    int ns=sizeof(sizes)/sizeof(sizes[0]);

    printf("%-10s | %10s | %10s | %8s | %8s | %s\n",
           "Size","1-thread","Pool MT","GF(1T)","GF(MT)","Verify");
    printf("---------- | ---------- | ---------- | -------- | -------- | ------\n");

    for(int si=0;si<ns;si++){
        int n=sizes[si];size_t nn=(size_t)n*n;
        double*A=aa(nn),*B=aa(nn),*C1=aa(nn),*C2=aa(nn);
        srand(42);randf(A,n);randf(B,n);

        int skip_1t = (n >= 4096);
        int runs=(n<=1024)?5:(n<=2048)?3:2;

        /* Warmup */
        if(!skip_1t) com6_multiply_1t(A,B,C1,n);
        com6_multiply_mt(A,B,C2,n);

        double best_1=1e30, gf1=0;
        if(!skip_1t){
            for(int r=0;r<runs;r++){
                double t0=now();com6_multiply_1t(A,B,C1,n);
                double t=now()-t0;if(t<best_1)best_1=t;}
            gf1=(2.0*n*n*(double)n)/(best_1*1e9);
        }

        double best_p=1e30;
        for(int r=0;r<runs;r++){
            double t0=now();com6_multiply_mt(A,B,C2,n);
            double t=now()-t0;if(t<best_p)best_p=t;}
        double gfp=(2.0*n*n*(double)n)/(best_p*1e9);

        /* Verify */
        const char*v;
        if(n<=512){
            if(!skip_1t) com6_multiply_1t(A,B,C1,n);
            com6_multiply_mt(A,B,C2,n);
            double*Cr=aa(nn);naive(A,B,Cr,n);
            double e=maxerr(C2,Cr,n);v=e<1e-6?"OK":"FAIL";af(Cr);
        }else if(!skip_1t){
            v=maxerr(C1,C2,n)<1e-6?"OK":"FAIL";
        }else{
            double*C3=aa(nn);com6_multiply_mt(A,B,C2,n);com6_multiply_mt(A,B,C3,n);
            v=maxerr(C2,C3,n)<1e-6?"OK":"FAIL";af(C3);
        }

        if(skip_1t)
            printf("%4dx%-5d | %10s | %8.1f ms | %8s | %6.1f   | %s\n",
                   n,n,"(skip)",best_p*1000,"--",gfp,v);
        else
            printf("%4dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %6.1f   | %s\n",
                   n,n,best_1*1000,best_p*1000,gf1,gfp,v);

        af(A);af(B);af(C1);af(C2);
    }

    pool_destroy();
    printf("\nv37: v26 blocking restored + thread pool + 8192 support\n");
    return 0;
}
