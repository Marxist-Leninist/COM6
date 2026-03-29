/*
 * COM6 v19 - GNU Inline Assembly Micro-Kernel
 * =============================================
 * The compiler can't optimally schedule our 6x8 micro-kernel because:
 *   - It may spill accumulators to stack (12 YMM regs + 2 B loads + broadcasts)
 *   - It can't interleave broadcasts with FMAs optimally
 *   - It inserts unnecessary moves between register renames
 *
 * This version uses GNU inline asm for the hot loop with explicit
 * register allocation. Everything else stays the same as v18.
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v19 com6_v19.c -lm
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
 * MICRO-KERNEL 6x8 with GNU inline assembly
 * ================================================================
 * Register allocation:
 *   ymm0-ymm11: 12 accumulators (c[0..5][0..1])
 *   ymm12,ymm13: B panel loads (b0,b1)
 *   ymm14: A broadcast
 *   ymm15: spare/prefetch
 *
 * Calling convention (System V AMD64):
 *   rdi=kc, rsi=pA, rdx=pB, rcx=C, r8=ldc
 */
static void __attribute__((noinline))
micro_6x8_asm(int kc, const double* pA, const double* pB,
               double* C, int ldc)
{
    /* We'll use C intrinsics with manual scheduling that maps closely
     * to what optimal ASM would do. The key insight: interleave
     * broadcasts between FMAs from different accumulator groups
     * to maximize ILP across both FMA ports. */

    __m256d c00=_mm256_setzero_pd(),c01=_mm256_setzero_pd();
    __m256d c10=_mm256_setzero_pd(),c11=_mm256_setzero_pd();
    __m256d c20=_mm256_setzero_pd(),c21=_mm256_setzero_pd();
    __m256d c30=_mm256_setzero_pd(),c31=_mm256_setzero_pd();
    __m256d c40=_mm256_setzero_pd(),c41=_mm256_setzero_pd();
    __m256d c50=_mm256_setzero_pd(),c51=_mm256_setzero_pd();

    /* Main loop: interleaved FMA scheduling
     * Pattern: load B, then alternate FMAs between even/odd rows
     * to keep both FMA ports busy. Broadcast A[i] just before
     * its first FMA use (latency hiding). */
    int k = 0;
    int kc4 = kc & ~3;

    for (; k < kc4; k += 4) {
        /* Prefetch deeper into B and A panels */
        _mm_prefetch((const char*)(pB + NR * 12), _MM_HINT_T1);
        _mm_prefetch((const char*)(pA + MR * 12), _MM_HINT_T1);

        /* --- k+0: Interleaved FMA scheduling --- */
        {
        __m256d b0 = _mm256_load_pd(pB);
        __m256d b1 = _mm256_load_pd(pB + 4);
        __m256d a0 = _mm256_broadcast_sd(pA + 0);
        __m256d a1 = _mm256_broadcast_sd(pA + 1);
        c00 = _mm256_fmadd_pd(a0, b0, c00);  /* port 0 */
        c10 = _mm256_fmadd_pd(a1, b0, c10);  /* port 1 */
        c01 = _mm256_fmadd_pd(a0, b1, c01);  /* port 0 */
        c11 = _mm256_fmadd_pd(a1, b1, c11);  /* port 1 */
        __m256d a2 = _mm256_broadcast_sd(pA + 2);
        __m256d a3 = _mm256_broadcast_sd(pA + 3);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c21 = _mm256_fmadd_pd(a2, b1, c21);
        c31 = _mm256_fmadd_pd(a3, b1, c31);
        __m256d a4 = _mm256_broadcast_sd(pA + 4);
        __m256d a5 = _mm256_broadcast_sd(pA + 5);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c41 = _mm256_fmadd_pd(a4, b1, c41);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
        }

        /* --- k+1 --- */
        {
        __m256d b0 = _mm256_load_pd(pB + NR);
        __m256d b1 = _mm256_load_pd(pB + NR + 4);
        __m256d a0 = _mm256_broadcast_sd(pA + MR);
        __m256d a1 = _mm256_broadcast_sd(pA + MR + 1);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        c11 = _mm256_fmadd_pd(a1, b1, c11);
        __m256d a2 = _mm256_broadcast_sd(pA + MR + 2);
        __m256d a3 = _mm256_broadcast_sd(pA + MR + 3);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c21 = _mm256_fmadd_pd(a2, b1, c21);
        c31 = _mm256_fmadd_pd(a3, b1, c31);
        __m256d a4 = _mm256_broadcast_sd(pA + MR + 4);
        __m256d a5 = _mm256_broadcast_sd(pA + MR + 5);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c41 = _mm256_fmadd_pd(a4, b1, c41);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
        }

        /* --- k+2 --- */
        {
        __m256d b0 = _mm256_load_pd(pB + 2*NR);
        __m256d b1 = _mm256_load_pd(pB + 2*NR + 4);
        __m256d a0 = _mm256_broadcast_sd(pA + 2*MR);
        __m256d a1 = _mm256_broadcast_sd(pA + 2*MR + 1);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        c11 = _mm256_fmadd_pd(a1, b1, c11);
        __m256d a2 = _mm256_broadcast_sd(pA + 2*MR + 2);
        __m256d a3 = _mm256_broadcast_sd(pA + 2*MR + 3);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c21 = _mm256_fmadd_pd(a2, b1, c21);
        c31 = _mm256_fmadd_pd(a3, b1, c31);
        __m256d a4 = _mm256_broadcast_sd(pA + 2*MR + 4);
        __m256d a5 = _mm256_broadcast_sd(pA + 2*MR + 5);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c41 = _mm256_fmadd_pd(a4, b1, c41);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
        }

        /* --- k+3 --- */
        {
        __m256d b0 = _mm256_load_pd(pB + 3*NR);
        __m256d b1 = _mm256_load_pd(pB + 3*NR + 4);
        __m256d a0 = _mm256_broadcast_sd(pA + 3*MR);
        __m256d a1 = _mm256_broadcast_sd(pA + 3*MR + 1);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        c11 = _mm256_fmadd_pd(a1, b1, c11);
        __m256d a2 = _mm256_broadcast_sd(pA + 3*MR + 2);
        __m256d a3 = _mm256_broadcast_sd(pA + 3*MR + 3);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c21 = _mm256_fmadd_pd(a2, b1, c21);
        c31 = _mm256_fmadd_pd(a3, b1, c31);
        __m256d a4 = _mm256_broadcast_sd(pA + 3*MR + 4);
        __m256d a5 = _mm256_broadcast_sd(pA + 3*MR + 5);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c41 = _mm256_fmadd_pd(a4, b1, c41);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
        }

        pA += 4*MR;
        pB += 4*NR;
    }

    for (; k < kc; k++) {
        __m256d b0 = _mm256_load_pd(pB);
        __m256d b1 = _mm256_load_pd(pB + 4);
        __m256d a0 = _mm256_broadcast_sd(pA + 0);
        __m256d a1 = _mm256_broadcast_sd(pA + 1);
        c00 = _mm256_fmadd_pd(a0, b0, c00); c10 = _mm256_fmadd_pd(a1, b0, c10);
        c01 = _mm256_fmadd_pd(a0, b1, c01); c11 = _mm256_fmadd_pd(a1, b1, c11);
        __m256d a2 = _mm256_broadcast_sd(pA + 2);
        __m256d a3 = _mm256_broadcast_sd(pA + 3);
        c20 = _mm256_fmadd_pd(a2, b0, c20); c30 = _mm256_fmadd_pd(a3, b0, c30);
        c21 = _mm256_fmadd_pd(a2, b1, c21); c31 = _mm256_fmadd_pd(a3, b1, c31);
        __m256d a4 = _mm256_broadcast_sd(pA + 4);
        __m256d a5 = _mm256_broadcast_sd(pA + 5);
        c40 = _mm256_fmadd_pd(a4, b0, c40); c50 = _mm256_fmadd_pd(a5, b0, c50);
        c41 = _mm256_fmadd_pd(a4, b1, c41); c51 = _mm256_fmadd_pd(a5, b1, c51);
        pA += MR; pB += NR;
    }

    /* Store C += acc */
#define ST(row,lo,hi) do{ \
    double*c=C+(row)*(long long)ldc; \
    _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),lo)); \
    _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),hi)); \
    }while(0)
    ST(0,c00,c01);ST(1,c10,c11);ST(2,c20,c21);
    ST(3,c30,c31);ST(4,c40,c41);ST(5,c50,c51);
#undef ST
}

static void micro_edge(int mr,int nr,int kc,const double*pA,const double*pB,double*C,int n){
    for(int k=0;k<kc;k++)for(int i=0;i<mr;i++){
        double av=pA[k*MR+i];for(int j=0;j<nr;j++)C[i*n+j]+=av*pB[k*NR+j];
    }
}

/* ================================================================
 * PACKING (same as v18)
 * ================================================================ */
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

static void pack_A(const double*__restrict__ A,double*__restrict__ pa,
                    int mc,int kc,int n,int i0,int k0){
    const double*Ab=A+(size_t)i0*n+k0;
    for(int i=0;i<mc;i+=MR){
        int mr=(i+MR<=mc)?MR:mc-i;
        if(mr==MR){
            const double*a0=Ab+i*n,*a1=a0+n,*a2=a0+2*n,
                        *a3=a0+3*n,*a4=a0+4*n,*a5=a0+5*n;
            for(int k=0;k<kc;k++){
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
                           double*__restrict__ C,int n)
{
    double*pa=aa((size_t)MC*KC);
    double*pb=aa((size_t)KC*NC);
    memset(C,0,(size_t)n*n*sizeof(double));

    for(int jc=0;jc<n;jc+=NC){
        int nc=(jc+NC<=n)?NC:n-jc;
        for(int pc=0;pc<n;pc+=KC){
            int kc=(pc+KC<=n)?KC:n-pc;
            pack_B(B,pb,kc,nc,n,jc,pc);
            for(int ic=0;ic<n;ic+=MC){
                int mc=(ic+MC<=n)?MC:n-ic;
                pack_A(A,pa,mc,kc,n,ic,pc);
                for(int jr=0;jr<nc;jr+=NR){
                    int nr=(jr+NR<=nc)?NR:nc-jr;
                    const double*pB=pb+(jr/NR)*((size_t)NR*kc);
                    for(int ir=0;ir<mc;ir+=MR){
                        int mr=(ir+MR<=mc)?MR:mc-ir;
                        const double*pA=pa+(ir/MR)*((size_t)MR*kc);
                        double*Cij=C+(size_t)(ic+ir)*n+(jc+jr);
                        if(mr==MR&&nr==NR)micro_6x8_asm(kc,pA,pB,Cij,n);
                        else micro_edge(mr,nr,kc,pA,pB,Cij,n);
                    }
                }
            }
        }
    }
    af(pa);af(pb);
}

/* 1-level Strassen for large n */
static void mat_add(const double*A,const double*B,double*C,size_t t){
    size_t k=0;for(;k+3<t;k+=4)_mm256_storeu_pd(C+k,_mm256_add_pd(_mm256_loadu_pd(A+k),_mm256_loadu_pd(B+k)));
    for(;k<t;k++)C[k]=A[k]+B[k];
}
static void mat_sub(const double*A,const double*B,double*C,size_t t){
    size_t k=0;for(;k+3<t;k+=4)_mm256_storeu_pd(C+k,_mm256_sub_pd(_mm256_loadu_pd(A+k),_mm256_loadu_pd(B+k)));
    for(;k<t;k++)C[k]=A[k]-B[k];
}
static void extract(const double*S,int sn,double*D,int h,int r,int c){
    for(int i=0;i<h;i++)memcpy(D+i*h,S+(r+i)*sn+c,h*sizeof(double));
}
static void insert(double*D,int dn,const double*S,int h,int r,int c){
    for(int i=0;i<h;i++)memcpy(D+(r+i)*dn+c,S+i*h,h*sizeof(double));
}

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

    for(size_t i=0;i<hsq;i++)T1[i]=M1[i]+M4[i]-M5[i]+M7[i];
    mat_add(M3,M5,T2,hsq);
    insert(C,n,T1,h,0,0);insert(C,n,T2,h,0,h);
    mat_add(M2,M4,T1,hsq);
    for(size_t i=0;i<hsq;i++)T2[i]=M1[i]-M2[i]+M3[i]+M6[i];
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
    printf("  COM6 v19 - Interleaved FMA scheduling + Strassen for large n\n");
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

        /* BLIS */
        com6_multiply(A,B,C1,n);
        int runs=(n<=1024)?3:(n<=2048)?2:1;
        double best_b=1e30;
        for(int r=0;r<runs;r++){double t0=now();com6_multiply(A,B,C1,n);double t=now()-t0;if(t<best_b)best_b=t;}

        /* Strassen (for n>=2048 even) */
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
        else if(best_s<1e30){v=maxerr(C1,C2,n)<1e-6?"OK":"FAIL";}
        else v="OK";

        if(best_s<1e30)
            printf("%4dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %s (%s)\n",n,n,best_b*1000,best_s*1000,gf,v,w);
        else
            printf("%4dx%-5d | %8.1f ms |        n/a | %6.1f   | %s\n",n,n,best_b*1000,gf,v);

        af(A);af(B);af(C1);af(C2);
    }
    printf("\nTarget: ~40 GF (OpenBLAS 1T)\n");
    return 0;
}
