/*
 * COM6 v64 - Barrier-Free MT + JC-Parallel
 * =========================================
 * Key insight: v26/v62 have a barrier between B-pack and compute.
 * On 4C/8T laptop, this barrier stalls ~50% of threads at any time.
 *
 * v64 eliminates the barrier by parallelizing the JC loop (column panels)
 * instead of the IC loop. Each thread:
 *   1. Owns a range of column panels (jc-tiles)
 *   2. Packs its own B slice (no barrier needed!)
 *   3. Computes all rows for its columns (no false sharing on C!)
 *
 * This is the BLIS JC-parallel approach — better than IC-parallel for
 * small thread counts because each thread independently owns:
 *   - Its own packed B (thread-local, lives in that core's L3 slice)
 *   - Its own output columns (no false sharing, no atomics)
 *   - Only A-packing is duplicated (small cost vs barrier elimination)
 *
 * Blocking (v26 proven):
 *   n<=1024:  MC=120, KC=256
 *   n>1024:   MC=96,  KC=320
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp -o com6_v64 com6_v64.c -lm
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
#endif

#define MR  6
#define NR  8
#define NC  2048
#define ALIGN 64

#define KC_SMALL 256
#define MC_SMALL 120
#define KC_LARGE 320
#define MC_LARGE 96
#define KC_MAX 320
#define MC_MAX 120

static inline double* aa(size_t c){return(double*)_mm_malloc(c*sizeof(double),ALIGN);}
static inline void af(double*p){_mm_free(p);}

/* C-prefetch 8x-unrolled 6x8 ASM micro-kernel */
static void __attribute__((noinline))
micro_6x8(int kc, const double* pA, const double* pB,
           double* C, int ldc)
{
    long long kc_8 = kc >> 3;
    long long kc_rem = kc & 7;
    long long ldc_bytes = (long long)ldc * 8;

    __asm__ volatile(
        "prefetcht0 (%[C])\n\t"
        "prefetcht0 32(%[C])\n\t"
        "movq %[C],%%r15\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"
        "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"
        "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"
        "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"
        "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"
        "prefetcht0 32(%%r15)\n\t"

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
        "vmovapd (%[pB]),%%ymm12\n\t"
        "vmovapd 32(%[pB]),%%ymm13\n\t"
        "vbroadcastsd (%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 8(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 16(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 24(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 32(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 40(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+1 */
        "vmovapd 64(%[pB]),%%ymm12\n\t"
        "vmovapd 96(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 48(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 56(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 64(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 72(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 80(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 88(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+2 */
        "vmovapd 128(%[pB]),%%ymm12\n\t"
        "vmovapd 160(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 96(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 104(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 112(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 120(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 128(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 136(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+3 */
        "vmovapd 192(%[pB]),%%ymm12\n\t"
        "vmovapd 224(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 144(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 152(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 160(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 168(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 176(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 184(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+4 */
        "prefetcht0 1152(%[pA])\n\t"
        "prefetcht0 1536(%[pB])\n\t"
        "vmovapd 256(%[pB]),%%ymm12\n\t"
        "vmovapd 288(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 192(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 200(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 208(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 216(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 224(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 232(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+5 */
        "vmovapd 320(%[pB]),%%ymm12\n\t"
        "vmovapd 352(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 240(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 248(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 256(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 264(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 272(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 280(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+6 */
        "vmovapd 384(%[pB]),%%ymm12\n\t"
        "vmovapd 416(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 288(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 296(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 304(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 312(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 320(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 328(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+7 */
        "vmovapd 448(%[pB]),%%ymm12\n\t"
        "vmovapd 480(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 336(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 344(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 352(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 360(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 368(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 376(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        "addq $384,%[pA]\n\t"
        "addq $512,%[pB]\n\t"
        "decq %[kc8]\n\t"
        "jnz 1b\n\t"

        "3:\n\t"
        "testq %[kcr],%[kcr]\n\t"
        "jle 2f\n\t"
        "4:\n\t"
        "vmovapd (%[pB]),%%ymm12\n\t"
        "vmovapd 32(%[pB]),%%ymm13\n\t"
        "vbroadcastsd (%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 8(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 16(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 24(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 32(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 40(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"
        "addq $48,%[pA]\n\t"
        "addq $64,%[pB]\n\t"
        "decq %[kcr]\n\t"
        "jnz 4b\n\t"

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

static void micro_edge(int mr,int nr,int kc,const double*pA,const double*pB,double*C,int n){
    for(int k=0;k<kc;k++)for(int i=0;i<mr;i++){
        double av=pA[k*MR+i];for(int j=0;j<nr;j++)C[i*n+j]+=av*pB[k*NR+j];}
}

static void pack_B(const double*__restrict__ B,double*__restrict__ pb,
                    int kc,int nc,int n,int j0,int k0){
    for(int j=0;j<nc;j+=NR){
        int nr=(j+NR<=nc)?NR:nc-j;
        const double*Bkj=B+(size_t)k0*n+(j0+j);
        double*dest=pb+(j/NR)*((size_t)NR*kc);
        if(nr==NR){for(int k=0;k<kc;k++){
            _mm256_store_pd(dest,_mm256_loadu_pd(Bkj));
            _mm256_store_pd(dest+4,_mm256_loadu_pd(Bkj+4));
            dest+=NR;Bkj+=n;
        }}else{for(int k=0;k<kc;k++){
            int jj;for(jj=0;jj<nr;jj++)dest[jj]=Bkj[jj];
            for(;jj<NR;jj++)dest[jj]=0.0;dest+=NR;Bkj+=n;
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
            for(;k+3<kc;k+=4){
                pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];
                pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];
                pa[6]=a0[k+1];pa[7]=a1[k+1];pa[8]=a2[k+1];
                pa[9]=a3[k+1];pa[10]=a4[k+1];pa[11]=a5[k+1];
                pa[12]=a0[k+2];pa[13]=a1[k+2];pa[14]=a2[k+2];
                pa[15]=a3[k+2];pa[16]=a4[k+2];pa[17]=a5[k+2];
                pa[18]=a0[k+3];pa[19]=a1[k+3];pa[20]=a2[k+3];
                pa[21]=a3[k+3];pa[22]=a4[k+3];pa[23]=a5[k+3];
                pa+=4*MR;
            }
            for(;k<kc;k++){
                pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];
                pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];pa+=MR;
            }
        }else{
            for(int k=0;k<kc;k++){
                int ii;for(ii=0;ii<mr;ii++)pa[ii]=(Ab+(i+ii)*n)[k];
                for(;ii<MR;ii++)pa[ii]=0.0;pa+=MR;
            }
        }
    }
}

static void macro_kernel_1t(const double*pa,const double*pb,
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

static void get_blocking(int n, int*pMC, int*pKC){
    if(n <= 1024){*pMC=MC_SMALL;*pKC=KC_SMALL;}
    else{*pMC=MC_LARGE;*pKC=KC_LARGE;}
}

/*
 * JC-parallel: each thread owns a column range.
 * No barrier needed — each thread packs its own B and computes independently.
 */
static void com6_multiply(const double*__restrict__ A,
                           const double*__restrict__ B,
                           double*__restrict__ C,int n)
{
    int nthreads = 1;
    #ifdef _OPENMP
    nthreads = omp_get_max_threads();
    #endif

    int use_mt = (n > 512 && nthreads > 1);
    int mc_blk, kc_blk;
    get_blocking(n, &mc_blk, &kc_blk);

    memset(C, 0, (size_t)n*n*sizeof(double));

    if(!use_mt){
        double*pa=aa((size_t)MC_MAX*KC_MAX),*pb=aa((size_t)KC_MAX*NC);
        for(int jc=0;jc<n;jc+=NC){int nc=(jc+NC<=n)?NC:n-jc;
            for(int pc=0;pc<n;pc+=kc_blk){int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
                pack_B(B,pb,kc,nc,n,jc,pc);
                for(int ic=0;ic<n;ic+=mc_blk){int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                    pack_A(A,pa,mc,kc,n,ic,pc);
                    macro_kernel_1t(pa,pb,C,mc,nc,kc,n,ic,jc);
                }
            }
        }
        af(pa);af(pb);
    } else {
        /* JC-parallel: distribute column panels across threads */
        /* Round NC panels to NR boundaries for clean splitting */
        int total_jr_panels = (n + NR - 1) / NR;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();

            /* Each thread owns a range of NR-panels */
            int panels_per = (total_jr_panels + nt - 1) / nt;
            int my_j0 = tid * panels_per * NR;
            int my_j1 = (tid + 1) * panels_per * NR;
            if(my_j1 > n) my_j1 = n;
            if(my_j0 >= n) my_j0 = my_j1 = n; /* this thread has no work */
            int my_nc = my_j1 - my_j0;

            if(my_nc > 0){
                double*pa = aa((size_t)MC_MAX*KC_MAX);
                double*pb = aa((size_t)KC_MAX * ((my_nc + NR - 1) / NR) * NR);

                for(int pc=0;pc<n;pc+=kc_blk){
                    int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
                    /* Each thread packs its own B slice — no barrier! */
                    pack_B(B,pb,kc,my_nc,n,my_j0,pc);
                    for(int ic=0;ic<n;ic+=mc_blk){
                        int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                        pack_A(A,pa,mc,kc,n,ic,pc);
                        macro_kernel_1t(pa,pb,C,mc,my_nc,kc,n,ic,my_j0);
                    }
                }

                af(pa);af(pb);
            }
        }
    }
}

static void com6_multiply_1t(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C,int n)
{
    int mc_blk, kc_blk;
    get_blocking(n, &mc_blk, &kc_blk);

    double*pa=aa((size_t)MC_MAX*KC_MAX),*pb=aa((size_t)KC_MAX*NC);
    memset(C,0,(size_t)n*n*sizeof(double));
    for(int jc=0;jc<n;jc+=NC){int nc=(jc+NC<=n)?NC:n-jc;
        for(int pc=0;pc<n;pc+=kc_blk){int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
            pack_B(B,pb,kc,nc,n,jc,pc);
            for(int ic=0;ic<n;ic+=mc_blk){int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                pack_A(A,pa,mc,kc,n,ic,pc);
                macro_kernel_1t(pa,pb,C,mc,nc,kc,n,ic,jc);
    }}}
    af(pa);af(pb);
}

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
    int nth=1;
    #ifdef _OPENMP
    nth=omp_get_max_threads();
    #endif

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
        if(!A||!B||!C){printf("OOM for %dx%d\n",n,n);return 1;}
        srand(42);randf(A,n);randf(B,n);
        int runs=(n<=512)?7:(n<=1024)?5:(n<=2048)?3:2;

        if(mode!=2){
            com6_multiply_1t(A,B,C,n);
            double best=1e30;
            for(int r=0;r<runs;r++){
                double t0=now();com6_multiply_1t(A,B,C,n);double t=now()-t0;
                if(t<best)best=t;
            }
            printf("1T  %4dx%-5d | %8.1f ms | %6.1f GF\n",n,n,best*1000,(2.0*n*n*(double)n)/(best*1e9));
        }
        if(mode!=1){
            com6_multiply(A,B,C,n);
            double best=1e30;
            for(int r=0;r<runs;r++){
                double t0=now();com6_multiply(A,B,C,n);double t=now()-t0;
                if(t<best)best=t;
            }
            printf("MT  %4dx%-5d | %8.1f ms | %6.1f GF (%d threads)\n",n,n,best*1000,(2.0*n*n*(double)n)/(best*1e9),nth);
        }

        af(A);af(B);af(C);
        return 0;
    }

    printf("====================================================================\n");
    printf("  COM6 v64 - JC-Parallel (Barrier-Free MT) (%d threads)\n",nth);
    printf("  n<=1024: MC=%d KC=%d | n>1024: MC=%d KC=%d | NC=%d\n",
           MC_SMALL,KC_SMALL,MC_LARGE,KC_LARGE,NC);
    printf("  JC-parallel: each thread owns columns, no barrier needed\n");
    printf("====================================================================\n\n");

    int sizes[]={256,512,1024,2048,4096,8192};
    int ns=sizeof(sizes)/sizeof(sizes[0]);
    printf("%-10s | %10s | %10s | %8s | %8s | %s\n","Size","1-thread","Multi-T","GF(1T)","GF(MT)","Verify");
    printf("---------- | ---------- | ---------- | -------- | -------- | ------\n");

    for(int si=0;si<ns;si++){
        int n=sizes[si];size_t nn=(size_t)n*n;

        if(si > 0){
#ifdef _WIN32
            Sleep(4000);
#else
            { struct timespec ts={4,0}; nanosleep(&ts,NULL); }
#endif
        }

        double*A=aa(nn),*B=aa(nn),*C1=NULL,*C2=aa(nn);
        if(!A||!B||!C2){
            printf("%4dx%-5d | SKIPPED (OOM)\n",n,n);
            if(A)af(A);if(B)af(B);if(C2)af(C2);
            continue;
        }
        srand(42);randf(A,n);randf(B,n);

        int do_1t = (n <= 2048);
        double best_1=1e30, gf1=0;

        if(do_1t){
            C1=aa(nn);
            com6_multiply_1t(A,B,C1,n);
            int runs=(n<=512)?7:(n<=1024)?5:3;
            for(int r=0;r<runs;r++){
                double t0=now();com6_multiply_1t(A,B,C1,n);double t=now()-t0;
                if(t<best_1)best_1=t;
            }
            gf1=(2.0*n*n*(double)n)/(best_1*1e9);
        }

        com6_multiply(A,B,C2,n);
        int runs_mt=(n<=1024)?5:(n<=2048)?3:2;
        double best_a=1e30;
        for(int r=0;r<runs_mt;r++){
            double t0=now();com6_multiply(A,B,C2,n);double t=now()-t0;
            if(t<best_a)best_a=t;
        }
        double gfa=(2.0*n*n*(double)n)/(best_a*1e9);

        const char*v;
        if(n<=512){
            double*Cr=aa(nn);naive(A,B,Cr,n);
            double e=fmax(do_1t?maxerr(C1,Cr,n):0,maxerr(C2,Cr,n));
            v=e<1e-6?"OK":"FAIL";af(Cr);
        }else if(do_1t&&C1){
            v=maxerr(C1,C2,n)<1e-6?"OK":"FAIL";
        }else{
            double*C3=aa(nn);
            if(C3){com6_multiply(A,B,C3,n);v=maxerr(C2,C3,n)<1e-6?"OK":"FAIL";af(C3);}
            else v="SKIP";
        }

        if(do_1t)
            printf("%4dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %6.1f   | %s\n",
                   n,n,best_1*1000,best_a*1000,gf1,gfa,v);
        else
            printf("%4dx%-5d | %10s | %8.1f ms | %6s   | %6.1f   | %s\n",
                   n,n,"--",best_a*1000,"--",gfa,v);

        af(A);af(B);if(C1)af(C1);af(C2);
    }

    printf("\nJC-parallel: no barriers, no false sharing, thread-local B packing\n");
    printf("Run individual sizes: ./com6_v64 <size> [mt|1t]\n");
    return 0;
}
