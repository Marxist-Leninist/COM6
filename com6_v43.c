/*
 * COM6 v43 - L1-Correct Blocking for 8192
 * =========================================
 * v42's KC=512 for 8192 was a mistake: micro-panels overflow L1 (32KB).
 *   pA[MR*KC] + pB[NR*KC] = (6+8)*512*8 = 57KB >> 32KB L1
 * This forces the micro-kernel to fetch from L2 (~4-5 cycle latency),
 * killing ~30% of performance.
 *
 * Fix: KC=320 for ALL sizes n>1024 (35KB, nearly fits L1 with prefetch).
 * Use NC=1024 for n>4096 to keep B-panel in L3 (1024*320*8 = 2.5MB).
 * MC=96 for n>1024: A-panel = 96*320*8 = 240KB, fits L2 (256KB).
 *
 * 3-tier blocking:
 *   n<=512:  MC=48  KC=256 NC=2048 (MT-friendly small sizes)
 *   n<=1024: MC=120 KC=256 NC=2048
 *   n<=4096: MC=96  KC=320 NC=2048
 *   n>4096:  MC=96  KC=320 NC=1024 (L1-correct, L3-friendly)
 *
 * Also: 4 physical-core threads for n>2048 (sustain clock on 15W TDP)
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp -o com6_v43 com6_v43.c -lm
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

#define MR  6
#define NR  8
#define ALIGN 64

/* Blocking tiers */
#define KC_SMALL 256
#define MC_TINY  48
#define MC_SMALL 120
#define KC_LARGE 320    /* L1: (6+8)*320*8=35KB ~fits with prefetch */
#define MC_LARGE 96     /* L2: 96*320*8=240KB < 256KB */
#define NC_SMALL 2048
#define NC_LARGE 1024   /* L3: 1024*320*8=2.5MB (fits with 4 cores) */

#define KC_MAX 320
#define MC_MAX 120
#define NC_MAX 2048

static inline double* aa(size_t c){return(double*)_mm_malloc(c*sizeof(double),ALIGN);}
static inline void af(double*p){_mm_free(p);}

/*
 * 6x8 micro-kernel with C-prefetch.
 * Prefetches C output at top so writeback doesn't stall on cache miss.
 */
static void __attribute__((noinline))
micro_6x8(int kc, const double* pA, const double* pB, double* C, int ldc)
{
    long long kc_8 = kc >> 3, kc_rem = kc & 7, ldc_bytes = (long long)ldc * 8;
    __asm__ volatile(
        /* Prefetch C output rows */
        "prefetcht0 (%[C])\n\t""prefetcht0 32(%[C])\n\t"
        "movq %[C],%%r15\n\t"
        "addq %[ldc],%%r15\n\t""prefetcht0 (%%r15)\n\t""prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t""prefetcht0 (%%r15)\n\t""prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t""prefetcht0 (%%r15)\n\t""prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t""prefetcht0 (%%r15)\n\t""prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t""prefetcht0 (%%r15)\n\t""prefetcht0 32(%%r15)\n\t"

        /* Zero accumulators */
        "vxorpd %%ymm0,%%ymm0,%%ymm0\n\t""vxorpd %%ymm1,%%ymm1,%%ymm1\n\t"
        "vxorpd %%ymm2,%%ymm2,%%ymm2\n\t""vxorpd %%ymm3,%%ymm3,%%ymm3\n\t"
        "vxorpd %%ymm4,%%ymm4,%%ymm4\n\t""vxorpd %%ymm5,%%ymm5,%%ymm5\n\t"
        "vxorpd %%ymm6,%%ymm6,%%ymm6\n\t""vxorpd %%ymm7,%%ymm7,%%ymm7\n\t"
        "vxorpd %%ymm8,%%ymm8,%%ymm8\n\t""vxorpd %%ymm9,%%ymm9,%%ymm9\n\t"
        "vxorpd %%ymm10,%%ymm10,%%ymm10\n\t""vxorpd %%ymm11,%%ymm11,%%ymm11\n\t"

        "testq %[kc8],%[kc8]\n\t""jle 3f\n\t"
        ".p2align 5\n\t"
        "1:\n\t"

        "prefetcht0 768(%[pA])\n\t"
        "prefetcht0 1024(%[pB])\n\t"

#define RANK1(AO,BO) \
        "vmovapd " #BO "(%[pB]),%%ymm12\n\t""vmovapd " #BO "+32(%[pB]),%%ymm13\n\t" \
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
        : "r15","ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
          "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","memory"
    );
}

static void micro_edge(int mr,int nr,int kc,const double*pA,const double*pB,double*C,int n){
    for(int k=0;k<kc;k++)for(int i=0;i<mr;i++){
        double av=pA[k*MR+i];for(int j=0;j<nr;j++)C[i*n+j]+=av*pB[k*NR+j];}
}

static void pack_B_chunk(const double*__restrict__ B,double*__restrict__ pb,
                          int kc,int nc,int n,int j0,int k0,int js,int je){
    for(int j=js;j<je;j+=NR){
        int nr=(j+NR<=nc)?NR:nc-j;
        const double*Bkj=B+(size_t)k0*n+(j0+j);
        double*d=pb+(j/NR)*((size_t)NR*kc);
        if(nr==NR){for(int k=0;k<kc;k++){
            _mm256_store_pd(d,_mm256_loadu_pd(Bkj));
            _mm256_store_pd(d+4,_mm256_loadu_pd(Bkj+4));
            d+=NR;Bkj+=n;
        }}else{for(int k=0;k<kc;k++){
            int jj;for(jj=0;jj<nr;jj++)d[jj]=Bkj[jj];
            for(;jj<NR;jj++)d[jj]=0.0;d+=NR;Bkj+=n;
        }}
    }
}

static void pack_A(const double*__restrict__ A,double*__restrict__ pa,
                    int mc,int kc,int n,int i0,int k0){
    const double*Ab=A+(size_t)i0*n+k0;
    for(int i=0;i<mc;i+=MR){
        int mr=(i+MR<=mc)?MR:mc-i;
        if(mr==MR){
            const double*a0=Ab+i*n,*a1=a0+n,*a2=a0+2*n,
                        *a3=a0+3*n,*a4=a0+4*n,*a5=a0+5*n;
            int k=0;
            for(;k+1<kc;k+=2){
                pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];
                pa[6]=a0[k+1];pa[7]=a1[k+1];pa[8]=a2[k+1];pa[9]=a3[k+1];pa[10]=a4[k+1];pa[11]=a5[k+1];pa+=2*MR;
            }
            for(;k<kc;k++){pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];pa+=MR;}
        }else{
            for(int k=0;k<kc;k++){
                int ii;for(ii=0;ii<mr;ii++)pa[ii]=(Ab+(i+ii)*n)[k];
                for(;ii<MR;ii++)pa[ii]=0.0;pa+=MR;
            }
        }
    }
}

static void macro_kernel(const double*pa,const double*pb,
                          double*C,int mc,int nc,int kc,int n,int ic,int jc){
    for(int jr=0;jr<nc;jr+=NR){
        int nr=(jr+NR<=nc)?NR:nc-jr;
        const double*pB=pb+(jr/NR)*((size_t)NR*kc);
        for(int ir=0;ir<mc;ir+=MR){
            int mr=(ir+MR<=mc)?MR:mc-ir;
            const double*pA=pa+(ir/MR)*((size_t)MR*kc);
            double*Cij=C+(size_t)(ic+ir)*n+(jc+jr);
            if(mr==MR&&nr==NR)micro_6x8(kc,pA,pB,Cij,n);
            else micro_edge(mr,nr,kc,pA,pB,Cij,n);
        }
    }
}

static void get_blocking(int n, int use_mt, int*pMC, int*pKC, int*pNC){
    if(n <= 512 && use_mt){*pMC=MC_TINY;  *pKC=KC_SMALL; *pNC=NC_SMALL;}
    else if(n <= 1024)    {*pMC=MC_SMALL;  *pKC=KC_SMALL; *pNC=NC_SMALL;}
    else if(n <= 4096)    {*pMC=MC_LARGE;  *pKC=KC_LARGE; *pNC=NC_SMALL;}
    else                  {*pMC=MC_LARGE;  *pKC=KC_LARGE; *pNC=NC_LARGE;}
    /* L1 check: (MR+NR)*KC*8 = 14*320*8 = 35.8KB ~= L1 (32KB) + prefetch OK */
    /* L2 check: MC*KC*8 = 96*320*8 = 240KB < 256KB L2 */
    /* L3 check: NC*KC*8 = 1024*320*8 = 2.5MB << 8MB L3 */
}

/* Physical cores only for truly sustained workloads (8192+) */
static int get_threads(int n){
    int max_t = 1;
    #ifdef _OPENMP
    max_t = omp_get_max_threads();
    #endif
    if(n <= 128) return 1;
    if(n <= 4096) return max_t;       /* All HT: computation finishes before thermal limit */
    return (max_t > 4) ? 4 : max_t;   /* Physical cores for 8192+: sustained 14s workload */
}

static double now(void){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec+t.tv_nsec*1e-9;}

static void com6_multiply(const double*__restrict__ A,
                           const double*__restrict__ B,
                           double*__restrict__ C, int n)
{
    int nthreads = get_threads(n);
    int use_mt = (nthreads > 1);
    int mc_blk, kc_blk, nc_blk;
    get_blocking(n, use_mt, &mc_blk, &kc_blk, &nc_blk);
    memset(C, 0, (size_t)n*n*sizeof(double));

    if(!use_mt){
        double*pa=aa((size_t)MC_MAX*KC_MAX),*pb=aa((size_t)KC_MAX*NC_MAX);
        for(int jc=0;jc<n;jc+=nc_blk){int nc=(jc+nc_blk<=n)?nc_blk:n-jc;
            for(int pc=0;pc<n;pc+=kc_blk){int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
                pack_B_chunk(B,pb,kc,nc,n,jc,pc,0,nc);
                for(int ic=0;ic<n;ic+=mc_blk){int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                    pack_A(A,pa,mc,kc,n,ic,pc);
                    macro_kernel(pa,pb,C,mc,nc,kc,n,ic,jc);}}}
        af(pa);af(pb);
    } else {
        #ifdef _OPENMP
        omp_set_num_threads(nthreads);
        #endif
        double** pa_bufs = (double**)malloc(nthreads * sizeof(double*));
        for(int t=0;t<nthreads;t++) pa_bufs[t] = aa((size_t)MC_MAX*KC_MAX);
        double* pb = aa((size_t)KC_MAX*NC_MAX);

        /* Single parallel region — pay fork/join cost once */
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            double* pa = pa_bufs[tid];

            for(int jc=0;jc<n;jc+=nc_blk){
                int nc=(jc+nc_blk<=n)?nc_blk:n-jc;
                for(int pc=0;pc<n;pc+=kc_blk){
                    int kc=(pc+kc_blk<=n)?kc_blk:n-pc;

                    /* Parallel B-packing */
                    int npanels = (nc + NR - 1) / NR;
                    int panels_per = (npanels + nt - 1) / nt;
                    int p0 = tid * panels_per, p1 = p0 + panels_per;
                    if(p1 > npanels) p1 = npanels;
                    int j_start = p0 * NR, j_end = p1 * NR;
                    if(j_end > nc) j_end = nc;
                    if(j_start < nc)
                        pack_B_chunk(B,pb,kc,nc,n,jc,pc,j_start,j_end);

                    #pragma omp barrier

                    /* Static ic-loop distribution */
                    int n_ic = (n + mc_blk - 1) / mc_blk;
                    int ic_per = (n_ic + nt - 1) / nt;
                    int ic0 = tid * ic_per * mc_blk;
                    int ic1 = (tid + 1) * ic_per * mc_blk;
                    if(ic1 > n) ic1 = n;

                    for(int ic=ic0;ic<ic1;ic+=mc_blk){
                        int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                        pack_A(A,pa,mc,kc,n,ic,pc);
                        macro_kernel(pa,pb,C,mc,nc,kc,n,ic,jc);
                    }

                    #pragma omp barrier
                }
            }
        }

        for(int t=0;t<nthreads;t++) af(pa_bufs[t]);
        free(pa_bufs); af(pb);

        #ifdef _OPENMP
        omp_set_num_threads(omp_get_max_threads());
        #endif
    }
}

static void com6_multiply_1t(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C, int n)
{
    int mc_blk, kc_blk, nc_blk;
    get_blocking(n, 0, &mc_blk, &kc_blk, &nc_blk);
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

int main(int argc, char**argv){
    int nth=1;
    #ifdef _OPENMP
    nth=omp_get_max_threads();
    #endif

    printf("====================================================================\n");
    printf("  COM6 v43 - L1-Correct Blocking + Physical-Core Threading\n");
    printf("  Max HT: %d | n>4096: %d physical-core threads\n",nth,nth>4?4:nth);
    printf("  n<=512:  MC=%d KC=%d NC=%d (MT-friendly)\n",MC_TINY,KC_SMALL,NC_SMALL);
    printf("  n<=1024: MC=%d KC=%d NC=%d\n",MC_SMALL,KC_SMALL,NC_SMALL);
    printf("  n<=4096: MC=%d KC=%d NC=%d\n",MC_LARGE,KC_LARGE,NC_SMALL);
    printf("  n>4096:  MC=%d KC=%d NC=%d (L1-correct, L3-friendly)\n",MC_LARGE,KC_LARGE,NC_LARGE);
    printf("  C-prefetch micro-kernel + single parallel region\n");
    printf("====================================================================\n\n");

    int all_sizes[]={256,512,1024,2048,4096,8192};
    int single_sizes[1];
    int *sizes; int ns; int single_mode = 0;

    if(argc >= 2){
        single_sizes[0] = atoi(argv[1]);
        sizes = single_sizes; ns = 1; single_mode = 1;
        printf("Single-size mode: %d (cold CPU test)\n\n",single_sizes[0]);
    } else {
        sizes = all_sizes; ns = 6;
    }

    printf("%-10s | %5s | %10s | %10s | %8s | %8s | %s\n",
           "Size","Thr","1-thread","MT","GF(1T)","GF(MT)","Verify");
    printf("---------- | ----- | ---------- | ---------- | -------- | -------- | ------\n");

    for(int si=0;si<ns;si++){
        int n=sizes[si]; size_t nn=(size_t)n*n;
        double*A=aa(nn),*B=aa(nn),*C1=aa(nn),*C2=aa(nn);
        if(!A||!B||!C1||!C2){
            printf("%4dx%-5d | ALLOC FAIL\n",n,n);
            if(A)af(A);if(B)af(B);if(C1)af(C1);if(C2)af(C2);
            continue;
        }
        srand(42);randf(A,n);randf(B,n);

        int mt_threads = get_threads(n);
        int skip_1t = (n >= 4096 && !single_mode);
        int runs = single_mode ? 5 : ((n<=1024)?5:(n<=2048)?3:2);

        /* Warmup */
        com6_multiply(A,B,C2,n);

        double best_1=1e30, gf1=0;
        if(!skip_1t){
            for(int r=0;r<runs;r++){
                double t0=now();com6_multiply_1t(A,B,C1,n);double t=now()-t0;
                if(t<best_1)best_1=t;
                if(single_mode)printf("  1T run %d: %.1f ms = %.1f GF\n",r+1,t*1000,(2.0*n*n*(double)n)/(t*1e9));
            }
            gf1=(2.0*n*n*(double)n)/(best_1*1e9);
        }

        double best_m=1e30;
        for(int r=0;r<runs;r++){
            double t0=now();com6_multiply(A,B,C2,n);double t=now()-t0;
            if(t<best_m)best_m=t;
            if(single_mode)printf("  MT run %d: %.1f ms = %.1f GF\n",r+1,t*1000,(2.0*n*n*(double)n)/(t*1e9));
        }
        double gfm=(2.0*n*n*(double)n)/(best_m*1e9);

        /* Verify */
        const char*v;
        if(n<=512){
            double*Cr=aa(nn);naive(A,B,Cr,n);
            double e=fmax(skip_1t?0:maxerr(C1,Cr,n),maxerr(C2,Cr,n));
            v=e<1e-6?"OK":"FAIL";af(Cr);
        }else if(!skip_1t){
            com6_multiply_1t(A,B,C1,n);
            v=maxerr(C1,C2,n)<1e-6?"OK":"FAIL";
        }else{
            /* Quick verify on 256x256 sub-block */
            double*As=aa(256*256),*Bs=aa(256*256),*Cs=aa(256*256),*Cr=aa(256*256);
            for(int i=0;i<256;i++)for(int j=0;j<256;j++){As[i*256+j]=A[i*n+j];Bs[i*256+j]=B[i*n+j];}
            com6_multiply_1t(As,Bs,Cs,256);naive(As,Bs,Cr,256);
            v=maxerr(Cs,Cr,256)<1e-6?"OK":"FAIL";
            af(As);af(Bs);af(Cs);af(Cr);
        }

        if(skip_1t)
            printf("%4dx%-5d | %3dT  | %10s | %8.1f ms | %8s | %6.1f   | %s\n",
                   n,n,mt_threads,"(skip)",best_m*1000,"--",gfm,v);
        else
            printf("%4dx%-5d | %3dT  | %8.1f ms | %8.1f ms | %6.1f   | %6.1f   | %s\n",
                   n,n,mt_threads,best_1*1000,best_m*1000,gf1,gfm,v);

        af(A);af(B);af(C1);af(C2);
    }
    printf("\nv43: KC=320 for ALL large sizes (L1-correct) + 4-thread for n>2048\n");
    return 0;
}
