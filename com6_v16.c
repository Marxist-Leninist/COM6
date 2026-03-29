/*
 * COM6 v16 - Direct B-packing (no transpose overhead)
 * ====================================================
 * KEY INSIGHT: In v13-v15 we transposed B->BT then packed BT into
 * interleaved format. But standard BLAS packs B DIRECTLY because
 * B[k][j..j+NR-1] is already contiguous — just memcpy NR doubles!
 * Our transpose was actually ADDING overhead.
 *
 * This version:
 *   1. Pack B directly from row-major B (contiguous NR loads per k)
 *   2. Pack A with gather (same as before — unavoidable)
 *   3. 6x8 micro-kernel with 4x k-unroll (proven best in v14)
 *   4. MC=96, KC=256 (v14's optimal params)
 *   5. SIMD-optimized B packing: _mm256_load 8 contiguous doubles per k
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v16 com6_v16.c -lm
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
#define MC  96
#define NC  4096
#define ALIGN 64

static inline double* aa(size_t c) { return (double*)_mm_malloc(c*sizeof(double),ALIGN); }
static inline void af(double* p) { _mm_free(p); }

/* ================================================================
 * MICRO-KERNEL 6x8, 4x k-unrolled (same as v14 — proven fastest)
 * ================================================================ */
static void __attribute__((noinline))
micro_6x8(int kc, const double* __restrict__ pA, const double* __restrict__ pB,
           double* __restrict__ C, int ldc)
{
    __m256d c00=_mm256_setzero_pd(), c01=_mm256_setzero_pd();
    __m256d c10=_mm256_setzero_pd(), c11=_mm256_setzero_pd();
    __m256d c20=_mm256_setzero_pd(), c21=_mm256_setzero_pd();
    __m256d c30=_mm256_setzero_pd(), c31=_mm256_setzero_pd();
    __m256d c40=_mm256_setzero_pd(), c41=_mm256_setzero_pd();
    __m256d c50=_mm256_setzero_pd(), c51=_mm256_setzero_pd();

    int k=0, kc4=kc&~3;
    for (; k<kc4; k+=4) {
        _mm_prefetch((const char*)(pA+MR*8), _MM_HINT_T0);
        _mm_prefetch((const char*)(pB+NR*4), _MM_HINT_T0);

        __m256d b0,b1,a;
        /* k+0 */
        b0=_mm256_load_pd(pB); b1=_mm256_load_pd(pB+4);
        a=_mm256_broadcast_sd(pA+0); c00=_mm256_fmadd_pd(a,b0,c00); c01=_mm256_fmadd_pd(a,b1,c01);
        a=_mm256_broadcast_sd(pA+1); c10=_mm256_fmadd_pd(a,b0,c10); c11=_mm256_fmadd_pd(a,b1,c11);
        a=_mm256_broadcast_sd(pA+2); c20=_mm256_fmadd_pd(a,b0,c20); c21=_mm256_fmadd_pd(a,b1,c21);
        a=_mm256_broadcast_sd(pA+3); c30=_mm256_fmadd_pd(a,b0,c30); c31=_mm256_fmadd_pd(a,b1,c31);
        a=_mm256_broadcast_sd(pA+4); c40=_mm256_fmadd_pd(a,b0,c40); c41=_mm256_fmadd_pd(a,b1,c41);
        a=_mm256_broadcast_sd(pA+5); c50=_mm256_fmadd_pd(a,b0,c50); c51=_mm256_fmadd_pd(a,b1,c51);
        /* k+1 */
        b0=_mm256_load_pd(pB+NR); b1=_mm256_load_pd(pB+NR+4);
        a=_mm256_broadcast_sd(pA+MR+0); c00=_mm256_fmadd_pd(a,b0,c00); c01=_mm256_fmadd_pd(a,b1,c01);
        a=_mm256_broadcast_sd(pA+MR+1); c10=_mm256_fmadd_pd(a,b0,c10); c11=_mm256_fmadd_pd(a,b1,c11);
        a=_mm256_broadcast_sd(pA+MR+2); c20=_mm256_fmadd_pd(a,b0,c20); c21=_mm256_fmadd_pd(a,b1,c21);
        a=_mm256_broadcast_sd(pA+MR+3); c30=_mm256_fmadd_pd(a,b0,c30); c31=_mm256_fmadd_pd(a,b1,c31);
        a=_mm256_broadcast_sd(pA+MR+4); c40=_mm256_fmadd_pd(a,b0,c40); c41=_mm256_fmadd_pd(a,b1,c41);
        a=_mm256_broadcast_sd(pA+MR+5); c50=_mm256_fmadd_pd(a,b0,c50); c51=_mm256_fmadd_pd(a,b1,c51);
        /* k+2 */
        b0=_mm256_load_pd(pB+2*NR); b1=_mm256_load_pd(pB+2*NR+4);
        a=_mm256_broadcast_sd(pA+2*MR+0); c00=_mm256_fmadd_pd(a,b0,c00); c01=_mm256_fmadd_pd(a,b1,c01);
        a=_mm256_broadcast_sd(pA+2*MR+1); c10=_mm256_fmadd_pd(a,b0,c10); c11=_mm256_fmadd_pd(a,b1,c11);
        a=_mm256_broadcast_sd(pA+2*MR+2); c20=_mm256_fmadd_pd(a,b0,c20); c21=_mm256_fmadd_pd(a,b1,c21);
        a=_mm256_broadcast_sd(pA+2*MR+3); c30=_mm256_fmadd_pd(a,b0,c30); c31=_mm256_fmadd_pd(a,b1,c31);
        a=_mm256_broadcast_sd(pA+2*MR+4); c40=_mm256_fmadd_pd(a,b0,c40); c41=_mm256_fmadd_pd(a,b1,c41);
        a=_mm256_broadcast_sd(pA+2*MR+5); c50=_mm256_fmadd_pd(a,b0,c50); c51=_mm256_fmadd_pd(a,b1,c51);
        /* k+3 */
        b0=_mm256_load_pd(pB+3*NR); b1=_mm256_load_pd(pB+3*NR+4);
        a=_mm256_broadcast_sd(pA+3*MR+0); c00=_mm256_fmadd_pd(a,b0,c00); c01=_mm256_fmadd_pd(a,b1,c01);
        a=_mm256_broadcast_sd(pA+3*MR+1); c10=_mm256_fmadd_pd(a,b0,c10); c11=_mm256_fmadd_pd(a,b1,c11);
        a=_mm256_broadcast_sd(pA+3*MR+2); c20=_mm256_fmadd_pd(a,b0,c20); c21=_mm256_fmadd_pd(a,b1,c21);
        a=_mm256_broadcast_sd(pA+3*MR+3); c30=_mm256_fmadd_pd(a,b0,c30); c31=_mm256_fmadd_pd(a,b1,c31);
        a=_mm256_broadcast_sd(pA+3*MR+4); c40=_mm256_fmadd_pd(a,b0,c40); c41=_mm256_fmadd_pd(a,b1,c41);
        a=_mm256_broadcast_sd(pA+3*MR+5); c50=_mm256_fmadd_pd(a,b0,c50); c51=_mm256_fmadd_pd(a,b1,c51);

        pA+=4*MR; pB+=4*NR;
    }
    for (; k<kc; k++) {
        __m256d b0=_mm256_load_pd(pB), b1=_mm256_load_pd(pB+4), a;
        a=_mm256_broadcast_sd(pA+0); c00=_mm256_fmadd_pd(a,b0,c00); c01=_mm256_fmadd_pd(a,b1,c01);
        a=_mm256_broadcast_sd(pA+1); c10=_mm256_fmadd_pd(a,b0,c10); c11=_mm256_fmadd_pd(a,b1,c11);
        a=_mm256_broadcast_sd(pA+2); c20=_mm256_fmadd_pd(a,b0,c20); c21=_mm256_fmadd_pd(a,b1,c21);
        a=_mm256_broadcast_sd(pA+3); c30=_mm256_fmadd_pd(a,b0,c30); c31=_mm256_fmadd_pd(a,b1,c31);
        a=_mm256_broadcast_sd(pA+4); c40=_mm256_fmadd_pd(a,b0,c40); c41=_mm256_fmadd_pd(a,b1,c41);
        a=_mm256_broadcast_sd(pA+5); c50=_mm256_fmadd_pd(a,b0,c50); c51=_mm256_fmadd_pd(a,b1,c51);
        pA+=MR; pB+=NR;
    }

    double* c;
    c=C;       _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c00));
               _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c01));
    c=C+ldc;   _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c10));
               _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c11));
    c=C+2*ldc; _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c20));
               _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c21));
    c=C+3*ldc; _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c30));
               _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c31));
    c=C+4*ldc; _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c40));
               _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c41));
    c=C+5*ldc; _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c50));
               _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c51));
}

static void micro_edge(int mr, int nr, int kc,
                        const double* pA, const double* pB,
                        double* C, int ldc)
{
    for (int k=0;k<kc;k++)
        for (int i=0;i<mr;i++) {
            double av=pA[k*MR+i];
            for (int j=0;j<nr;j++) C[i*ldc+j]+=av*pB[k*NR+j];
        }
}

/* ================================================================
 * PACKING — Direct from B (no transpose!)
 * ================================================================
 * B is row-major: B[k][j] at B + k*n + j
 * Pack NR columns of B for a K-slab: for each k, B[k][j..j+NR-1]
 * is CONTIGUOUS — can use SIMD loads!
 */
static void pack_B_direct(const double* __restrict__ B, double* __restrict__ pb,
                           int kc, int nc, int n, int j0, int k0)
{
    for (int j = 0; j < nc; j += NR) {
        int nr = (j+NR<=nc) ? NR : nc-j;
        if (nr == NR) {
            const double* Bkj = B + (size_t)k0 * n + (j0 + j);
            for (int k = 0; k < kc; k++) {
                /* B[k0+k][j0+j .. j0+j+7] is contiguous! SIMD copy! */
                _mm256_store_pd(pb,   _mm256_loadu_pd(Bkj));
                _mm256_store_pd(pb+4, _mm256_loadu_pd(Bkj+4));
                pb += NR;
                Bkj += n;  /* next row of B */
            }
        } else {
            const double* Bkj = B + (size_t)k0 * n + (j0 + j);
            for (int k = 0; k < kc; k++) {
                int jj;
                for (jj=0; jj<nr; jj++) pb[jj] = Bkj[jj];
                for (; jj<NR; jj++) pb[jj] = 0.0;
                pb += NR;
                Bkj += n;
            }
        }
    }
}

/* Pack A: gather MR rows per k */
static void pack_A(const double* __restrict__ A, double* __restrict__ pa,
                    int mc, int kc, int n, int i0, int k0)
{
    const double* Ab = A + (size_t)i0*n + k0;
    for (int i = 0; i < mc; i += MR) {
        int mr = (i+MR<=mc) ? MR : mc-i;
        if (mr == MR) {
            const double *a0=Ab+i*n, *a1=a0+n, *a2=a0+2*n,
                         *a3=a0+3*n, *a4=a0+4*n, *a5=a0+5*n;
            for (int k=0; k<kc; k++) {
                pa[0]=a0[k]; pa[1]=a1[k]; pa[2]=a2[k];
                pa[3]=a3[k]; pa[4]=a4[k]; pa[5]=a5[k];
                pa+=MR;
            }
        } else {
            for (int k=0; k<kc; k++) {
                int ii;
                for (ii=0;ii<mr;ii++) pa[ii]=(Ab+(i+ii)*n)[k];
                for (;ii<MR;ii++) pa[ii]=0.0;
                pa+=MR;
            }
        }
    }
}

/* ================================================================
 * BLIS 5-LOOP — Direct B packing, no transpose
 * ================================================================ */
static void com6_multiply(const double* __restrict__ A,
                           const double* __restrict__ B,
                           double* __restrict__ C, int n)
{
    double* pa = aa((size_t)MC*KC);
    double* pb = aa((size_t)KC*NC);
    memset(C, 0, (size_t)n*n*sizeof(double));

    for (int jc=0; jc<n; jc+=NC) {
        int nc = (jc+NC<=n)?NC:n-jc;
        for (int pc=0; pc<n; pc+=KC) {
            int kc = (pc+KC<=n)?KC:n-pc;

            /* Pack B: CONTIGUOUS access — the big win */
            pack_B_direct(B, pb, kc, nc, n, jc, pc);

            for (int ic=0; ic<n; ic+=MC) {
                int mc = (ic+MC<=n)?MC:n-ic;

                pack_A(A, pa, mc, kc, n, ic, pc);

                for (int jr=0; jr<nc; jr+=NR) {
                    int nr=(jr+NR<=nc)?NR:nc-jr;
                    const double* pB = pb + (jr/NR)*((size_t)NR*kc);

                    for (int ir=0; ir<mc; ir+=MR) {
                        int mr=(ir+MR<=mc)?MR:mc-ir;
                        const double* pA = pa + (ir/MR)*((size_t)MR*kc);
                        double* Cij = C + (size_t)(ic+ir)*n + (jc+jr);

                        if (mr==MR && nr==NR)
                            micro_6x8(kc, pA, pB, Cij, n);
                        else
                            micro_edge(mr, nr, kc, pA, pB, Cij, n);
                    }
                }
            }
        }
    }
    af(pa); af(pb);
}

/* Naive reference */
static void naive(const double* A, const double* B, double* C, int n) {
    memset(C,0,(size_t)n*n*sizeof(double));
    for (int i=0;i<n;i++) for (int k=0;k<n;k++) {
        double a=A[i*n+k]; for (int j=0;j<n;j++) C[i*n+j]+=a*B[k*n+j];
    }
}

static double now(void) { struct timespec t; timespec_get(&t,TIME_UTC); return t.tv_sec+t.tv_nsec*1e-9; }
static void randfill(double* M, int n) { for (int i=0;i<n*n;i++) M[i]=(double)rand()/RAND_MAX*2-1; }
static double maxerr(const double* A, const double* B, int n) {
    double m=0; for (int i=0;i<n*n;i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;} return m;
}

int main(void) {
    printf("====================================================================\n");
    printf("  COM6 v16 - Direct B-pack (no transpose), 6x8, 4x k-unroll\n");
    printf("  MC=%d KC=%d NC=%d\n", MC, KC, NC);
    printf("====================================================================\n\n");

    int sizes[]={256,512,1024,2048,4096};
    int ns=sizeof(sizes)/sizeof(sizes[0]);
    printf("%-10s | %10s | %8s | %s\n","Size","COM6-v16","GFLOPS","Verify");
    printf("---------- | ---------- | -------- | ------\n");

    for (int si=0;si<ns;si++) {
        int n=sizes[si]; size_t nn=(size_t)n*n;
        double *A=aa(nn),*B=aa(nn),*C1=aa(nn),*C2=aa(nn);
        srand(42); randfill(A,n); randfill(B,n);

        com6_multiply(A,B,C1,n); /* warmup */

        int runs=(n<=1024)?3:(n<=2048)?2:1;
        double best=1e30;
        for (int r=0;r<runs;r++) {
            double t0=now(); com6_multiply(A,B,C1,n);
            double t=now()-t0; if(t<best)best=t;
        }
        double gf=(2.0*n*n*(double)n)/(best*1e9);

        const char* v="skip";
        if (n<=512) { naive(A,B,C2,n); v=maxerr(C1,C2,n)<1e-6?"OK":"FAIL"; }

        printf("%4dx%-5d | %8.1f ms | %6.1f   | %s\n",n,n,best*1000,gf,v);
        af(A);af(B);af(C1);af(C2);
    }
    printf("\nTarget: ~40 GFLOPS (OpenBLAS 1T, i7-10510U)\n");
    return 0;
}
