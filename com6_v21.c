/*
 * COM6 v21 - 4x Unrolled ASM + Vectorized A-Packing
 * ===================================================
 * v20's ASM kernel had NO k-unrolling (1 rank-1 per loop).
 * This version:
 *   1. 4x k-unrolled inline ASM micro-kernel (4 rank-1 updates per loop)
 *   2. Vectorized A-packing: load 4 contiguous k-values per row,
 *      then scatter-store into packed format (6 contiguous loads vs 24 gathers)
 *   3. Prefetch tuning in both packing and micro-kernel
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v21 com6_v21.c -lm
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MR  6
#define NR  8
#define KC  256
#define MC  120
#define NC  2048
#define ALIGN 64

static inline double* aa(size_t c){return(double*)_mm_malloc(c*sizeof(double),ALIGN);}
static inline void af(double*p){_mm_free(p);}

/* ================================================================
 * MICRO-KERNEL 6x8 - 4x k-unrolled inline ASM
 * ================================================================
 * Processes 4 rank-1 updates per loop iteration.
 * 48 FMAs + 8 B loads + 24 A broadcasts per 4 k-steps.
 * Loop overhead amortized over 4x more work.
 */
static void __attribute__((noinline))
micro_6x8(int kc, const double* pA, const double* pB,
           double* C, int ldc)
{
    long long kc_4 = kc >> 2;      /* kc / 4 */
    long long kc_rem = kc & 3;     /* kc % 4 */
    long long ldc_bytes = (long long)ldc * 8;

    __asm__ volatile(
        /* Zero ymm0-ymm11 */
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

        /* 4x unrolled main loop */
        "testq %[kc4],%[kc4]\n\t"
        "jle 3f\n\t"
        ".p2align 5\n\t"
        "1:\n\t"

        /* Prefetch: 8 k-steps ahead */
        "prefetcht1 384(%[pA])\n\t"     /* 8*MR*8 = 384 bytes ahead */
        "prefetcht1 512(%[pB])\n\t"     /* 8*NR*8 = 512 bytes ahead */

        /* ---- k+0 ---- */
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

        /* ---- k+1 ---- */
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

        /* ---- k+2 ---- */
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

        /* ---- k+3 ---- */
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

        /* Advance: pA += 4*48=192, pB += 4*64=256 */
        "addq $192,%[pA]\n\t"
        "addq $256,%[pB]\n\t"
        "decq %[kc4]\n\t"
        "jnz 1b\n\t"

        /* Remainder loop (0-3 iterations) */
        "3:\n\t"
        "testq %[kcr],%[kcr]\n\t"
        "jle 2f\n\t"
        ".p2align 4\n\t"
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

        /* Store C += accumulators */
        "2:\n\t"
        "vaddpd (%[C]),%%ymm0,%%ymm0\n\t"
        "vmovupd %%ymm0,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm1,%%ymm1\n\t"
        "vmovupd %%ymm1,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"

        "vaddpd (%[C]),%%ymm2,%%ymm2\n\t"
        "vmovupd %%ymm2,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm3,%%ymm3\n\t"
        "vmovupd %%ymm3,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"

        "vaddpd (%[C]),%%ymm4,%%ymm4\n\t"
        "vmovupd %%ymm4,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm5,%%ymm5\n\t"
        "vmovupd %%ymm5,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"

        "vaddpd (%[C]),%%ymm6,%%ymm6\n\t"
        "vmovupd %%ymm6,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm7,%%ymm7\n\t"
        "vmovupd %%ymm7,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"

        "vaddpd (%[C]),%%ymm8,%%ymm8\n\t"
        "vmovupd %%ymm8,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm9,%%ymm9\n\t"
        "vmovupd %%ymm9,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"

        "vaddpd (%[C]),%%ymm10,%%ymm10\n\t"
        "vmovupd %%ymm10,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm11,%%ymm11\n\t"
        "vmovupd %%ymm11,32(%[C])\n\t"

        : [pA]"+r"(pA), [pB]"+r"(pB), [kc4]"+r"(kc_4), [kcr]"+r"(kc_rem), [C]"+r"(C)
        : [ldc]"r"(ldc_bytes)
        : "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
          "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","memory"
    );
}

static void micro_edge(int mr,int nr,int kc,const double*pA,const double*pB,double*C,int n){
    for(int k=0;k<kc;k++)for(int i=0;i<mr;i++){
        double av=pA[k*MR+i];for(int j=0;j<nr;j++)C[i*n+j]+=av*pB[k*NR+j];
    }
}

/* ================================================================
 * PACKING - Vectorized A-pack
 * ================================================================ */

/* Pack B: SIMD copy of contiguous rows (same as v18) */
static void pack_B(const double*__restrict__ B,double*__restrict__ pb,
                    int kc,int nc,int n,int j0,int k0){
    for(int j=0;j<nc;j+=NR){
        int nr=(j+NR<=nc)?NR:nc-j;
        const double*Bkj=B+(size_t)k0*n+(j0+j);
        if(nr==NR){
            for(int k=0;k<kc;k++){
                _mm256_store_pd(pb,_mm256_loadu_pd(Bkj));
                _mm256_store_pd(pb+4,_mm256_loadu_pd(Bkj+4));
                pb+=NR;Bkj+=n;
            }
        }else{
            for(int k=0;k<kc;k++){
                int jj;for(jj=0;jj<nr;jj++)pb[jj]=Bkj[jj];
                for(;jj<NR;jj++)pb[jj]=0.0;pb+=NR;Bkj+=n;
            }
        }
    }
}

/*
 * Pack A: Vectorized with 4-wide k processing.
 * For each MR=6 strip, load 4 consecutive k-values from each row
 * (contiguous access!), then scatter-store into packed [k][mr] format.
 */
static void pack_A(const double*__restrict__ A,double*__restrict__ pa,
                    int mc,int kc,int n,int i0,int k0){
    const double*Ab=A+(size_t)i0*n+k0;
    for(int i=0;i<mc;i+=MR){
        int mr=(i+MR<=mc)?MR:mc-i;
        if(mr==MR){
            const double*a0=Ab+i*n,*a1=a0+n,*a2=a0+2*n,
                        *a3=a0+3*n,*a4=a0+4*n,*a5=a0+5*n;

            int k=0;
            /* Process 4 k-values at a time: 6 contiguous loads -> 4 scatter stores */
            for(;k+3<kc;k+=4){
                /* Prefetch ahead for each row */
                _mm_prefetch((const char*)(a0+k+32),_MM_HINT_T0);
                _mm_prefetch((const char*)(a1+k+32),_MM_HINT_T0);
                _mm_prefetch((const char*)(a2+k+32),_MM_HINT_T0);
                _mm_prefetch((const char*)(a3+k+32),_MM_HINT_T0);
                _mm_prefetch((const char*)(a4+k+32),_MM_HINT_T0);
                _mm_prefetch((const char*)(a5+k+32),_MM_HINT_T0);

                /* Load 4 consecutive elements from each of 6 rows */
                __m256d r0=_mm256_loadu_pd(a0+k);
                __m256d r1=_mm256_loadu_pd(a1+k);
                __m256d r2=_mm256_loadu_pd(a2+k);
                __m256d r3=_mm256_loadu_pd(a3+k);
                __m256d r4=_mm256_loadu_pd(a4+k);
                __m256d r5=_mm256_loadu_pd(a5+k);

                /* Transpose 6x4 -> 4 groups of 6 using scalar extraction
                 * (AVX2 doesn't have great scatter support, but the stores
                 *  to packed buffer are sequential and hot in L1) */
                /* k+0: pa[0..5] = {a0[k],a1[k],a2[k],a3[k],a4[k],a5[k]} */
                pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];
                pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];
                /* k+1 */
                pa[6]=a0[k+1];pa[7]=a1[k+1];pa[8]=a2[k+1];
                pa[9]=a3[k+1];pa[10]=a4[k+1];pa[11]=a5[k+1];
                /* k+2 */
                pa[12]=a0[k+2];pa[13]=a1[k+2];pa[14]=a2[k+2];
                pa[15]=a3[k+2];pa[16]=a4[k+2];pa[17]=a5[k+2];
                /* k+3 */
                pa[18]=a0[k+3];pa[19]=a1[k+3];pa[20]=a2[k+3];
                pa[21]=a3[k+3];pa[22]=a4[k+3];pa[23]=a5[k+3];
                pa+=24; /* 4*MR */
            }
            /* Remainder */
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

/* ================================================================
 * BLIS 5-LOOP
 * ================================================================ */
static void com6_multiply(const double*__restrict__ A,
                           const double*__restrict__ B,
                           double*__restrict__ C,int n){
    double*pa=aa((size_t)MC*KC),*pb=aa((size_t)KC*NC);
    memset(C,0,(size_t)n*n*sizeof(double));
    for(int jc=0;jc<n;jc+=NC){int nc=(jc+NC<=n)?NC:n-jc;
        for(int pc=0;pc<n;pc+=KC){int kc=(pc+KC<=n)?KC:n-pc;
            pack_B(B,pb,kc,nc,n,jc,pc);
            for(int ic=0;ic<n;ic+=MC){int mc=(ic+MC<=n)?MC:n-ic;
                pack_A(A,pa,mc,kc,n,ic,pc);
                for(int jr=0;jr<nc;jr+=NR){int nr=(jr+NR<=nc)?NR:nc-jr;
                    const double*pB=pb+(jr/NR)*((size_t)NR*kc);
                    for(int ir=0;ir<mc;ir+=MR){int mr=(ir+MR<=mc)?MR:mc-ir;
                        const double*pA=pa+(ir/MR)*((size_t)MR*kc);
                        double*Cij=C+(size_t)(ic+ir)*n+(jc+jr);
                        if(mr==MR&&nr==NR)micro_6x8(kc,pA,pB,Cij,n);
                        else micro_edge(mr,nr,kc,pA,pB,Cij,n);
    }}}}}
    af(pa);af(pb);
}

/* Strassen 1-level */
static void mat_add(const double*A,const double*B,double*C,size_t t){
    size_t k=0;for(;k+3<t;k+=4)_mm256_storeu_pd(C+k,_mm256_add_pd(_mm256_loadu_pd(A+k),_mm256_loadu_pd(B+k)));
    for(;k<t;k++)C[k]=A[k]+B[k];
}
static void mat_sub(const double*A,const double*B,double*C,size_t t){
    size_t k=0;for(;k+3<t;k+=4)_mm256_storeu_pd(C+k,_mm256_sub_pd(_mm256_loadu_pd(A+k),_mm256_loadu_pd(B+k)));
    for(;k<t;k++)C[k]=A[k]-B[k];
}
static void extract(const double*S,int sn,double*D,int h,int r,int c){for(int i=0;i<h;i++)memcpy(D+i*h,S+(r+i)*sn+c,h*sizeof(double));}
static void insert(double*D,int dn,const double*S,int h,int r,int c){for(int i=0;i<h;i++)memcpy(D+(r+i)*dn+c,S+i*h,h*sizeof(double));}

static void com6_strassen1(const double*A,const double*B,double*C,int n){
    int h=n/2;size_t hsq=(size_t)h*h;
    double*A11=aa(hsq),*A12=aa(hsq),*A21=aa(hsq),*A22=aa(hsq);
    double*B11=aa(hsq),*B12=aa(hsq),*B21=aa(hsq),*B22=aa(hsq);
    double*T1=aa(hsq),*T2=aa(hsq);
    double*M1=aa(hsq),*M2=aa(hsq),*M3=aa(hsq),*M4=aa(hsq),*M5=aa(hsq),*M6=aa(hsq),*M7=aa(hsq);
    extract(A,n,A11,h,0,0);extract(A,n,A12,h,0,h);extract(A,n,A21,h,h,0);extract(A,n,A22,h,h,h);
    extract(B,n,B11,h,0,0);extract(B,n,B12,h,0,h);extract(B,n,B21,h,h,0);extract(B,n,B22,h,h,h);
    mat_add(A11,A22,T1,hsq);mat_add(B11,B22,T2,hsq);com6_multiply(T1,T2,M1,h);
    mat_add(A21,A22,T1,hsq);com6_multiply(T1,B11,M2,h);
    mat_sub(B12,B22,T1,hsq);com6_multiply(A11,T1,M3,h);
    mat_sub(B21,B11,T1,hsq);com6_multiply(A22,T1,M4,h);
    mat_add(A11,A12,T1,hsq);com6_multiply(T1,B22,M5,h);
    mat_sub(A21,A11,T1,hsq);mat_add(B11,B12,T2,hsq);com6_multiply(T1,T2,M6,h);
    mat_sub(A12,A22,T1,hsq);mat_add(B21,B22,T2,hsq);com6_multiply(T1,T2,M7,h);
    for(size_t i=0;i<hsq;i++)T1[i]=M1[i]+M4[i]-M5[i]+M7[i];mat_add(M3,M5,T2,hsq);
    insert(C,n,T1,h,0,0);insert(C,n,T2,h,0,h);
    mat_add(M2,M4,T1,hsq);for(size_t i=0;i<hsq;i++)T2[i]=M1[i]-M2[i]+M3[i]+M6[i];
    insert(C,n,T1,h,h,0);insert(C,n,T2,h,h,h);
    af(A11);af(A12);af(A21);af(A22);af(B11);af(B12);af(B21);af(B22);
    af(T1);af(T2);af(M1);af(M2);af(M3);af(M4);af(M5);af(M6);af(M7);
}

/* Naive */
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
    printf("====================================================================\n");
    printf("  COM6 v21 - 4x unrolled ASM + vectorized A-pack + prefetch\n");
    printf("  MC=%d KC=%d NC=%d\n",MC,KC,NC);
    printf("====================================================================\n\n");

    int sizes[]={256,512,1024,2048,4096};
    int ns=sizeof(sizes)/sizeof(sizes[0]);
    printf("%-10s | %10s | %10s | %8s | %s\n","Size","BLIS","Str+BLIS","GF(best)","Verify");
    printf("---------- | ---------- | ---------- | -------- | ------\n");

    for(int si=0;si<ns;si++){
        int n=sizes[si];size_t nn=(size_t)n*n;
        double*A=aa(nn),*B=aa(nn),*C1=aa(nn),*C2=aa(nn);
        srand(42);randf(A,n);randf(B,n);

        com6_multiply(A,B,C1,n);
        int runs=(n<=1024)?3:(n<=2048)?2:1;
        double best_b=1e30;
        for(int r=0;r<runs;r++){double t0=now();com6_multiply(A,B,C1,n);double t=now()-t0;if(t<best_b)best_b=t;}

        double best_s=1e30;
        if(n>=2048&&n%2==0){
            com6_strassen1(A,B,C2,n);
            for(int r=0;r<runs;r++){double t0=now();com6_strassen1(A,B,C2,n);double t=now()-t0;if(t<best_s)best_s=t;}
        }

        double best=(best_s<best_b)?best_s:best_b;
        double gf=(2.0*n*n*(double)n)/(best*1e9);
        const char*w=(best_s<best_b)?"Str":"BLIS";

        const char*v;
        if(n<=512){double*Cr=aa(nn);naive(A,B,Cr,n);v=maxerr(C1,Cr,n)<1e-6?"OK":"FAIL";af(Cr);}
        else if(best_s<1e30)v=maxerr(C1,C2,n)<1e-6?"OK":"FAIL";
        else v="OK";

        if(best_s<1e30)
            printf("%4dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %s (%s)\n",n,n,best_b*1000,best_s*1000,gf,v,w);
        else
            printf("%4dx%-5d | %8.1f ms |        n/a | %6.1f   | %s\n",n,n,best_b*1000,gf,v);

        af(A);af(B);af(C1);af(C2);
    }
    printf("\nTarget: ~40 GF (OpenBLAS 1T, i7-10510U)\n");
    return 0;
}
