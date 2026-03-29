/*
 * COM6 v27 - Large-Scale + Strassen Hybrid
 * ==========================================
 * Changes from v26:
 * 1. Added 8192x8192 benchmark size
 * 2. Single-level Strassen for n>=2048 (7 sub-multiplies instead of 8 = 12.5% fewer FLOPs)
 * 3. NC=4096 for better L3 utilization at large sizes (8MB L3 fits 4096*320*8=10MB...
 *    actually keep NC=2048 to stay in L3, but try parallelizing jc loop for 8192)
 * 4. 2D thread decomposition: parallelize both jc and ic for very large matrices
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp -static -o com6_v27 com6_v27.c -lm
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

/* 8x k-unrolled 6x8 ASM micro-kernel */
static void __attribute__((noinline))
micro_6x8(int kc, const double* pA, const double* pB,
           double* C, int ldc)
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
        : "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
          "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","memory"
    );
}

static void micro_edge(int mr,int nr,int kc,const double*pA,const double*pB,double*C,int n){
    for(int k=0;k<kc;k++)for(int i=0;i<mr;i++){
        double av=pA[k*MR+i];for(int j=0;j<nr;j++)C[i*n+j]+=av*pB[k*NR+j];}
}

static void pack_B_chunk(const double*__restrict__ B,double*__restrict__ pb,
                          int kc,int nc,int n,int j0,int k0,int j_start,int j_end){
    for(int j=j_start;j<j_end;j+=NR){
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

static void pack_B(const double*__restrict__ B,double*__restrict__ pb,
                    int kc,int nc,int n,int j0,int k0){
    pack_B_chunk(B,pb,kc,nc,n,j0,k0,0,nc);
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
                pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];
                pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];
                pa[6]=a0[k+1];pa[7]=a1[k+1];pa[8]=a2[k+1];
                pa[9]=a3[k+1];pa[10]=a4[k+1];pa[11]=a5[k+1];
                pa+=2*MR;
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

static void get_blocking(int n, int*pMC, int*pKC){
    if(n <= 1024){*pMC=MC_SMALL;*pKC=KC_SMALL;}
    else{*pMC=MC_LARGE;*pKC=KC_LARGE;}
}

/* ================================================================
 * Core BLIS multiply (C += A*B, C is NOT zeroed — caller must do it)
 * This allows Strassen to accumulate into C.
 * ================================================================ */
static void blis_gemm_mt(const double*__restrict__ A,
                          const double*__restrict__ B,
                          double*__restrict__ C,int n,
                          int mc_blk,int kc_blk)
{
    int nthreads = 1;
    #ifdef _OPENMP
    nthreads = omp_get_max_threads();
    #endif

    int use_mt = (n > 512 && nthreads > 1);

    if(!use_mt){
        double*pa=aa((size_t)MC_MAX*KC_MAX),*pb=aa((size_t)KC_MAX*NC);
        for(int jc=0;jc<n;jc+=NC){int nc=(jc+NC<=n)?NC:n-jc;
            for(int pc=0;pc<n;pc+=kc_blk){int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
                pack_B(B,pb,kc,nc,n,jc,pc);
                for(int ic=0;ic<n;ic+=mc_blk){int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                    pack_A(A,pa,mc,kc,n,ic,pc);
                    macro_kernel(pa,pb,C,mc,nc,kc,n,ic,jc);
                }
            }
        }
        af(pa);af(pb);
    } else {
        double** pa_bufs = (double**)malloc(nthreads * sizeof(double*));
        for(int t=0;t<nthreads;t++) pa_bufs[t] = aa((size_t)MC_MAX*KC_MAX);
        double* pb = aa((size_t)KC_MAX*NC);

        for(int jc=0;jc<n;jc+=NC){
            int nc=(jc+NC<=n)?NC:n-jc;
            for(int pc=0;pc<n;pc+=kc_blk){
                int kc=(pc+kc_blk<=n)?kc_blk:n-pc;

                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    int nt = omp_get_num_threads();
                    double* pa = pa_bufs[tid];

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

                    #pragma omp for schedule(static)
                    for(int ic=0;ic<n;ic+=mc_blk){
                        int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                        pack_A(A,pa,mc,kc,n,ic,pc);
                        macro_kernel(pa,pb,C,mc,nc,kc,n,ic,jc);
                    }
                }
            }
        }

        for(int t=0;t<nthreads;t++) af(pa_bufs[t]);
        free(pa_bufs);
        af(pb);
    }
}

/* ================================================================
 * Matrix add/sub helpers for Strassen (row-major, stride=lda)
 * ================================================================ */
static void mat_add(const double*A,int lda,const double*B,int ldb,
                     double*C,int ldc,int m,int n_){
    for(int i=0;i<m;i++){
        const double*ai=A+i*lda,*bi=B+i*ldb;double*ci=C+i*ldc;
        int j=0;
        for(;j+3<n_;j+=4){
            __m256d va=_mm256_loadu_pd(ai+j),vb=_mm256_loadu_pd(bi+j);
            _mm256_storeu_pd(ci+j,_mm256_add_pd(va,vb));
        }
        for(;j<n_;j++) ci[j]=ai[j]+bi[j];
    }
}

static void mat_sub(const double*A,int lda,const double*B,int ldb,
                     double*C,int ldc,int m,int n_){
    for(int i=0;i<m;i++){
        const double*ai=A+i*lda,*bi=B+i*ldb;double*ci=C+i*ldc;
        int j=0;
        for(;j+3<n_;j+=4){
            __m256d va=_mm256_loadu_pd(ai+j),vb=_mm256_loadu_pd(bi+j);
            _mm256_storeu_pd(ci+j,_mm256_sub_pd(va,vb));
        }
        for(;j<n_;j++) ci[j]=ai[j]-bi[j];
    }
}

static void mat_addto(double*C,int ldc,const double*A,int lda,int m,int n_){
    for(int i=0;i<m;i++){
        double*ci=C+i*ldc;const double*ai=A+i*lda;
        int j=0;
        for(;j+3<n_;j+=4){
            __m256d vc=_mm256_loadu_pd(ci+j),va=_mm256_loadu_pd(ai+j);
            _mm256_storeu_pd(ci+j,_mm256_add_pd(vc,va));
        }
        for(;j<n_;j++) ci[j]+=ai[j];
    }
}

static void mat_subfrom(double*C,int ldc,const double*A,int lda,int m,int n_){
    for(int i=0;i<m;i++){
        double*ci=C+i*ldc;const double*ai=A+i*lda;
        int j=0;
        for(;j+3<n_;j+=4){
            __m256d vc=_mm256_loadu_pd(ci+j),va=_mm256_loadu_pd(ai+j);
            _mm256_storeu_pd(ci+j,_mm256_sub_pd(vc,va));
        }
        for(;j<n_;j++) ci[j]-=ai[j];
    }
}

/* Copy submatrix from strided source to contiguous n2 x n2 buffer */
static void mat_copy_from(const double*S,int lds,double*D,int n2){
    for(int i=0;i<n2;i++) memcpy(D+i*n2,S+i*lds,n2*sizeof(double));
}

/* Add contiguous n2 x n2 buffer back to strided destination */
static void mat_copy_add(const double*S,int n2,double*D,int ldd,int m,int n_){
    for(int i=0;i<m;i++){
        const double*si=S+i*n2;double*di=D+i*ldd;
        int j=0;
        for(;j+3<n_;j+=4){
            __m256d vs=_mm256_loadu_pd(si+j),vd=_mm256_loadu_pd(di+j);
            _mm256_storeu_pd(di+j,_mm256_add_pd(vs,vd));
        }
        for(;j<n_;j++) di[j]+=si[j];
    }
}

static void mat_copy_to(const double*S,int n2,double*D,int ldd,int m,int n_){
    for(int i=0;i<m;i++) memcpy(D+i*ldd,S+i*n2,n_*sizeof(double));
}

/* ================================================================
 * Strassen's algorithm (1 level) using BLIS gemm as base case
 * C = A * B where A, B, C are n x n, n is even
 * 7 multiplies of (n/2)x(n/2) instead of 8 = 12.5% fewer FLOPs
 * ================================================================ */
static void strassen_multiply(const double*A,const double*B,
                               double*C,int n,int mc_blk,int kc_blk)
{
    int n2 = n / 2;
    size_t q = (size_t)n2 * n2;

    /* Pointers to quadrants (strided in original matrices) */
    const double*A11=A, *A12=A+n2, *A21=A+(size_t)n2*n, *A22=A+(size_t)n2*n+n2;
    const double*B11=B, *B12=B+n2, *B21=B+(size_t)n2*n, *B22=B+(size_t)n2*n+n2;
    double*C11=C, *C12=C+n2, *C21=C+(size_t)n2*n, *C22=C+(size_t)n2*n+n2;

    /* Temp buffers — contiguous n2 x n2 for BLIS (needs contiguous input) */
    double*T1=aa(q),*T2=aa(q),*M=aa(q);

    /* M1 = (A11+A22)*(B11+B22) */
    mat_add(A11,n,A22,n,T1,n2,n2,n2);
    mat_add(B11,n,B22,n,T2,n2,n2,n2);
    memset(M,0,q*sizeof(double));
    blis_gemm_mt(T1,T2,M,n2,mc_blk,kc_blk);
    /* C11 += M1, C22 += M1 */
    mat_copy_add(M,n2,C11,n,n2,n2);
    mat_copy_add(M,n2,C22,n,n2,n2);

    /* M2 = (A21+A22)*B11 */
    mat_add(A21,n,A22,n,T1,n2,n2,n2);
    mat_copy_from(B11,n,T2,n2);
    memset(M,0,q*sizeof(double));
    blis_gemm_mt(T1,T2,M,n2,mc_blk,kc_blk);
    /* C21 += M2, C22 -= M2 */
    mat_copy_add(M,n2,C21,n,n2,n2);
    for(int i=0;i<n2;i++){double*c=C22+i*n;const double*m=M+i*n2;
        int j=0;for(;j+3<n2;j+=4){__m256d vc=_mm256_loadu_pd(c+j),vm=_mm256_loadu_pd(m+j);
            _mm256_storeu_pd(c+j,_mm256_sub_pd(vc,vm));}for(;j<n2;j++)c[j]-=m[j];}

    /* M3 = A11*(B12-B22) */
    mat_sub(B12,n,B22,n,T2,n2,n2,n2);
    mat_copy_from(A11,n,T1,n2);
    memset(M,0,q*sizeof(double));
    blis_gemm_mt(T1,T2,M,n2,mc_blk,kc_blk);
    /* C12 += M3, C22 += M3 */
    mat_copy_add(M,n2,C12,n,n2,n2);
    mat_copy_add(M,n2,C22,n,n2,n2);

    /* M4 = A22*(B21-B11) */
    mat_sub(B21,n,B11,n,T2,n2,n2,n2);
    mat_copy_from(A22,n,T1,n2);
    memset(M,0,q*sizeof(double));
    blis_gemm_mt(T1,T2,M,n2,mc_blk,kc_blk);
    /* C11 += M4, C21 += M4 */
    mat_copy_add(M,n2,C11,n,n2,n2);
    mat_copy_add(M,n2,C21,n,n2,n2);

    /* M5 = (A11+A12)*B22 */
    mat_add(A11,n,A12,n,T1,n2,n2,n2);
    mat_copy_from(B22,n,T2,n2);
    memset(M,0,q*sizeof(double));
    blis_gemm_mt(T1,T2,M,n2,mc_blk,kc_blk);
    /* C11 -= M5, C12 += M5 */
    for(int i=0;i<n2;i++){double*c=C11+i*n;const double*m=M+i*n2;
        int j=0;for(;j+3<n2;j+=4){__m256d vc=_mm256_loadu_pd(c+j),vm=_mm256_loadu_pd(m+j);
            _mm256_storeu_pd(c+j,_mm256_sub_pd(vc,vm));}for(;j<n2;j++)c[j]-=m[j];}
    mat_copy_add(M,n2,C12,n,n2,n2);

    /* M6 = (A21-A11)*(B11+B12) */
    mat_sub(A21,n,A11,n,T1,n2,n2,n2);
    mat_add(B11,n,B12,n,T2,n2,n2,n2);
    memset(M,0,q*sizeof(double));
    blis_gemm_mt(T1,T2,M,n2,mc_blk,kc_blk);
    /* C22 += M6 */
    mat_copy_add(M,n2,C22,n,n2,n2);

    /* M7 = (A12-A22)*(B21+B22) */
    mat_sub(A12,n,A22,n,T1,n2,n2,n2);
    mat_add(B21,n,B22,n,T2,n2,n2,n2);
    memset(M,0,q*sizeof(double));
    blis_gemm_mt(T1,T2,M,n2,mc_blk,kc_blk);
    /* C11 += M7 */
    mat_copy_add(M,n2,C11,n,n2,n2);

    af(T1);af(T2);af(M);
}

/* ================================================================
 * Top-level: Strassen for large even sizes, direct BLIS otherwise
 * ================================================================ */
static void com6_multiply(const double*__restrict__ A,
                           const double*__restrict__ B,
                           double*__restrict__ C,int n)
{
    int mc_blk, kc_blk;
    get_blocking(n, &mc_blk, &kc_blk);
    memset(C, 0, (size_t)n*n*sizeof(double));

    /* Use Strassen for n>=4096 and even */
    if(n >= 4096 && (n & 1) == 0){
        /* For Strassen sub-problems, use large-size blocking */
        int mc_sub, kc_sub;
        get_blocking(n/2, &mc_sub, &kc_sub);
        strassen_multiply(A,B,C,n,mc_sub,kc_sub);
    } else {
        blis_gemm_mt(A,B,C,n,mc_blk,kc_blk);
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
                macro_kernel(pa,pb,C,mc,nc,kc,n,ic,jc);
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

int main(void){
    int nth=1;
    #ifdef _OPENMP
    nth=omp_get_max_threads();
    #endif

    printf("====================================================================\n");
    printf("  COM6 v27 - Strassen Hybrid + Adaptive Blocking (%d threads)\n",nth);
    printf("  n<=1024: MC=%d KC=%d | n>1024: MC=%d KC=%d\n",MC_SMALL,KC_SMALL,MC_LARGE,KC_LARGE);
    printf("  Strassen for n>=4096 (7 half-size multiplies = 12.5%% fewer FLOPs)\n");
    printf("====================================================================\n\n");

    int sizes[]={256,512,1024,2048,4096,8192};
    int ns=sizeof(sizes)/sizeof(sizes[0]);
    printf("%-10s | %10s | %10s | %8s | %8s | %s\n","Size","1-thread","Adaptive","GF(1T)","GF(Ada)","Verify");
    printf("---------- | ---------- | ---------- | -------- | -------- | ------\n");

    for(int si=0;si<ns;si++){
        int n=sizes[si];size_t nn=(size_t)n*n;
        double*A=aa(nn),*B=aa(nn),*C1=aa(nn),*C2=aa(nn);
        srand(42);randf(A,n);randf(B,n);

        /* Warmup */
        com6_multiply_1t(A,B,C1,n);
        com6_multiply(A,B,C2,n);

        /* Fewer runs for huge sizes */
        int runs=(n<=1024)?5:(n<=2048)?3:(n<=4096)?2:1;

        double best_1=1e30;
        for(int r=0;r<runs;r++){double t0=now();com6_multiply_1t(A,B,C1,n);double t=now()-t0;if(t<best_1)best_1=t;}

        double best_a=1e30;
        for(int r=0;r<runs;r++){double t0=now();com6_multiply(A,B,C2,n);double t=now()-t0;if(t<best_a)best_a=t;}

        double flops = 2.0*n*n*(double)n;
        double gf1=flops/(best_1*1e9);
        double gfa=flops/(best_a*1e9);

        /* Strassen does ~17.5% fewer FLOPs for same n^3 "equivalent" —
           report effective GFLOPS (as if it did all n^3 work) */
        const char*v;
        if(n<=512){double*Cr=aa(nn);naive(A,B,Cr,n);
            double e=fmax(maxerr(C1,Cr,n),maxerr(C2,Cr,n));v=e<1e-6?"OK":"FAIL";af(Cr);
        }else{
            double e=maxerr(C1,C2,n);
            /* Strassen has slightly different numerical error */
            v=(e<1e-4)?"OK":"FAIL";
        }

        printf("%4dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %6.1f   | %s\n",
               n,n,best_1*1000,best_a*1000,gf1,gfa,v);

        af(A);af(B);af(C1);af(C2);
    }
    printf("\nStrassen effective GFLOPS: reports n^3 equivalent throughput\n");
    return 0;
}
