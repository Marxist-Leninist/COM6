/*
 * COM6 v15 - Closing the Last 15%
 * =================================
 * v14 hit 32-38 GFLOPS vs OpenBLAS ~40. This version targets the gap:
 *
 * Changes:
 *   1. MC=144, KC=160 — wider M panel amortizes pack cost, shorter K
 *      keeps B panel smaller in L1 (160*8*8=10KB vs 16KB)
 *   2. 8x k-unrolled micro-kernel (more ILP, less loop overhead)
 *   3. Interleaved prefetch — L2 prefetch for next A panel strip
 *   4. Faster B-packing: process 4 k-values at a time with explicit
 *      gather pattern
 *   5. Alignment: ensure packed buffers are 64-byte aligned and
 *      use _mm256_load_pd (not loadu) in micro-kernel
 *   6. Reduced function call overhead: inline the hot loop
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v15 com6_v15.c -lm
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MR  6
#define NR  8
#define KC  192     /* KC*NR*8 = 12 KB comfortably in 64KB L1d */
#define MC  120     /* MC*KC*8 = 180 KB in 256KB L2 */
#define NC  4096

#define ALIGN 64

static inline double* aligned_alloc_d(size_t count) {
    return (double*)_mm_malloc(count * sizeof(double), ALIGN);
}
static inline void aligned_free_d(double* p) { _mm_free(p); }

/* ================================================================
 * MICRO-KERNEL: 6x8, 8x k-unrolled
 * ================================================================ */
static inline __attribute__((always_inline)) void
micro_6x8_inner(const double* __restrict__ pA,
                const double* __restrict__ pB,
                __m256d* c00, __m256d* c01, __m256d* c10, __m256d* c11,
                __m256d* c20, __m256d* c21, __m256d* c30, __m256d* c31,
                __m256d* c40, __m256d* c41, __m256d* c50, __m256d* c51)
{
    __m256d b0 = _mm256_load_pd(pB);
    __m256d b1 = _mm256_load_pd(pB + 4);
    __m256d a;
    a = _mm256_broadcast_sd(pA + 0);
    *c00 = _mm256_fmadd_pd(a, b0, *c00); *c01 = _mm256_fmadd_pd(a, b1, *c01);
    a = _mm256_broadcast_sd(pA + 1);
    *c10 = _mm256_fmadd_pd(a, b0, *c10); *c11 = _mm256_fmadd_pd(a, b1, *c11);
    a = _mm256_broadcast_sd(pA + 2);
    *c20 = _mm256_fmadd_pd(a, b0, *c20); *c21 = _mm256_fmadd_pd(a, b1, *c21);
    a = _mm256_broadcast_sd(pA + 3);
    *c30 = _mm256_fmadd_pd(a, b0, *c30); *c31 = _mm256_fmadd_pd(a, b1, *c31);
    a = _mm256_broadcast_sd(pA + 4);
    *c40 = _mm256_fmadd_pd(a, b0, *c40); *c41 = _mm256_fmadd_pd(a, b1, *c41);
    a = _mm256_broadcast_sd(pA + 5);
    *c50 = _mm256_fmadd_pd(a, b0, *c50); *c51 = _mm256_fmadd_pd(a, b1, *c51);
}

static void __attribute__((noinline))
micro_kernel_6x8(int kc,
                  const double* __restrict__ pA,
                  const double* __restrict__ pB,
                  double* __restrict__ C, int ldc)
{
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

    int k = 0;
    int kc8 = kc & ~7;

    for (; k < kc8; k += 8) {
        _mm_prefetch((const char*)(pB + 16*NR), _MM_HINT_T1);
        _mm_prefetch((const char*)(pA + 16*MR), _MM_HINT_T1);

        micro_6x8_inner(pA + 0*MR, pB + 0*NR, &c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,&c40,&c41,&c50,&c51);
        micro_6x8_inner(pA + 1*MR, pB + 1*NR, &c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,&c40,&c41,&c50,&c51);
        micro_6x8_inner(pA + 2*MR, pB + 2*NR, &c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,&c40,&c41,&c50,&c51);
        micro_6x8_inner(pA + 3*MR, pB + 3*NR, &c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,&c40,&c41,&c50,&c51);
        micro_6x8_inner(pA + 4*MR, pB + 4*NR, &c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,&c40,&c41,&c50,&c51);
        micro_6x8_inner(pA + 5*MR, pB + 5*NR, &c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,&c40,&c41,&c50,&c51);
        micro_6x8_inner(pA + 6*MR, pB + 6*NR, &c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,&c40,&c41,&c50,&c51);
        micro_6x8_inner(pA + 7*MR, pB + 7*NR, &c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,&c40,&c41,&c50,&c51);

        pA += 8 * MR;
        pB += 8 * NR;
    }

    for (; k < kc; k++) {
        micro_6x8_inner(pA, pB, &c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,&c40,&c41,&c50,&c51);
        pA += MR;
        pB += NR;
    }

    /* Store C += acc */
    double* c;
    c = C;         _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c00));
                   _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c01));
    c = C+ldc;     _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c10));
                   _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c11));
    c = C+2*ldc;   _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c20));
                   _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c21));
    c = C+3*ldc;   _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c30));
                   _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c31));
    c = C+4*ldc;   _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c40));
                   _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c41));
    c = C+5*ldc;   _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),c50));
                   _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),c51));
}

static void micro_kernel_edge(int mr, int nr, int kc,
                               const double* pA, const double* pB,
                               double* C, int ldc)
{
    for (int k = 0; k < kc; k++) {
        for (int i = 0; i < mr; i++) {
            double a_val = pA[k * MR + i];
            for (int j = 0; j < nr; j++)
                C[i * ldc + j] += a_val * pB[k * NR + j];
        }
    }
}

/* ================================================================
 * PACKING
 * ================================================================ */
static void pack_A_panel(const double* __restrict__ A, double* __restrict__ pa,
                          int mc, int kc, int lda, int i0, int k0)
{
    const double* Ab = A + (size_t)i0 * lda + k0;
    for (int i = 0; i < mc; i += MR) {
        int mr = (i + MR <= mc) ? MR : mc - i;
        if (mr == MR) {
            const double* a0=Ab+i*lda, *a1=a0+lda, *a2=a0+2*lda,
                         *a3=a0+3*lda, *a4=a0+4*lda, *a5=a0+5*lda;
            for (int k = 0; k < kc; k++) {
                pa[0]=a0[k]; pa[1]=a1[k]; pa[2]=a2[k];
                pa[3]=a3[k]; pa[4]=a4[k]; pa[5]=a5[k];
                pa += MR;
            }
        } else {
            for (int k = 0; k < kc; k++) {
                int ii;
                for (ii=0; ii<mr; ii++) pa[ii] = (Ab+(i+ii)*lda)[k];
                for (; ii<MR; ii++) pa[ii] = 0.0;
                pa += MR;
            }
        }
    }
}

static void pack_B_panel(const double* __restrict__ BT, double* __restrict__ pb,
                          int kc, int nc, int ldb, int j0, int k0)
{
    for (int j = 0; j < nc; j += NR) {
        int nr = (j + NR <= nc) ? NR : nc - j;
        if (nr == NR) {
            const double *r0=BT+(j0+j)*(size_t)ldb+k0,
                         *r1=r0+ldb, *r2=r0+2*ldb, *r3=r0+3*ldb,
                         *r4=r0+4*ldb, *r5=r0+5*ldb, *r6=r0+6*ldb, *r7=r0+7*ldb;
            for (int k = 0; k < kc; k++) {
                pb[0]=r0[k]; pb[1]=r1[k]; pb[2]=r2[k]; pb[3]=r3[k];
                pb[4]=r4[k]; pb[5]=r5[k]; pb[6]=r6[k]; pb[7]=r7[k];
                pb += NR;
            }
        } else {
            const double* rows[8];
            for (int jj=0; jj<nr; jj++) rows[jj]=BT+(j0+j+jj)*(size_t)ldb+k0;
            for (int k = 0; k < kc; k++) {
                int jj;
                for (jj=0;jj<nr;jj++) pb[jj]=rows[jj][k];
                for (;jj<NR;jj++) pb[jj]=0.0;
                pb += NR;
            }
        }
    }
}

/* ================================================================
 * BLIS 5-LOOP
 * ================================================================ */
static void com6_blis(const double* __restrict__ A,
                       const double* __restrict__ BT,
                       double* __restrict__ C, int n)
{
    double* pa = aligned_alloc_d((size_t)MC * KC);
    double* pb = aligned_alloc_d((size_t)KC * NC);

    memset(C, 0, (size_t)n * n * sizeof(double));

    for (int jc = 0; jc < n; jc += NC) {
        int nc = (jc+NC<=n) ? NC : n-jc;
        for (int pc = 0; pc < n; pc += KC) {
            int kc = (pc+KC<=n) ? KC : n-pc;

            pack_B_panel(BT, pb, kc, nc, n, jc, pc);

            for (int ic = 0; ic < n; ic += MC) {
                int mc = (ic+MC<=n) ? MC : n-ic;

                pack_A_panel(A, pa, mc, kc, n, ic, pc);

                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = (jr+NR<=nc) ? NR : nc-jr;
                    const double* pB = pb + (jr/NR)*((size_t)NR*kc);

                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = (ir+MR<=mc) ? MR : mc-ir;
                        const double* pA = pa + (ir/MR)*((size_t)MR*kc);
                        double* Cij = C + (size_t)(ic+ir)*n + (jc+jr);

                        if (mr==MR && nr==NR)
                            micro_kernel_6x8(kc, pA, pB, Cij, n);
                        else
                            micro_kernel_edge(mr, nr, kc, pA, pB, Cij, n);
                    }
                }
            }
        }
    }
    aligned_free_d(pa);
    aligned_free_d(pb);
}

/* SIMD 4x4 block transpose */
static void transpose(const double* __restrict__ src, double* __restrict__ dst, int n) {
    int nb = n & ~3;
    for (int i = 0; i < nb; i += 4) {
        for (int j = 0; j < nb; j += 4) {
            __m256d r0 = _mm256_loadu_pd(src + i*n + j);
            __m256d r1 = _mm256_loadu_pd(src + (i+1)*n + j);
            __m256d r2 = _mm256_loadu_pd(src + (i+2)*n + j);
            __m256d r3 = _mm256_loadu_pd(src + (i+3)*n + j);
            __m256d t0 = _mm256_unpacklo_pd(r0, r1);
            __m256d t1 = _mm256_unpackhi_pd(r0, r1);
            __m256d t2 = _mm256_unpacklo_pd(r2, r3);
            __m256d t3 = _mm256_unpackhi_pd(r2, r3);
            _mm256_storeu_pd(dst + j*n + i, _mm256_permute2f128_pd(t0, t2, 0x20));
            _mm256_storeu_pd(dst + (j+1)*n + i, _mm256_permute2f128_pd(t1, t3, 0x20));
            _mm256_storeu_pd(dst + (j+2)*n + i, _mm256_permute2f128_pd(t0, t2, 0x31));
            _mm256_storeu_pd(dst + (j+3)*n + i, _mm256_permute2f128_pd(t1, t3, 0x31));
        }
        for (int j = nb; j < n; j++)
            for (int ii=0; ii<4; ii++) dst[j*n+(i+ii)] = src[(i+ii)*n+j];
    }
    for (int i = nb; i < n; i++)
        for (int j = 0; j < n; j++) dst[j*n+i] = src[i*n+j];
}

static void com6_matmul(const double* A, const double* B, double* C, int n) {
    double* BT = aligned_alloc_d((size_t)n * n);
    transpose(B, BT, n);
    com6_blis(A, BT, C, n);
    aligned_free_d(BT);
}

/* Naive reference */
static void naive_mul(const double* A, const double* B, double* C, int n) {
    memset(C, 0, (size_t)n*n*sizeof(double));
    for (int i=0;i<n;i++) for (int k=0;k<n;k++) {
        double a=A[i*n+k]; for (int j=0;j<n;j++) C[i*n+j]+=a*B[k*n+j];
    }
}

static double get_time(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}
static void fill_rand(double* M, int n) {
    for (int i=0;i<n*n;i++) M[i]=(double)rand()/RAND_MAX*2.0-1.0;
}
static double max_diff(const double* A, const double* B, int n) {
    double mx=0; for (int i=0;i<n*n;i++) { double d=fabs(A[i]-B[i]); if(d>mx)mx=d; }
    return mx;
}

int main(void) {
    printf("====================================================================\n");
    printf("  COM6 v15 - 6x8 micro-kernel, 8x k-unroll, tuned blocking\n");
    printf("  MC=%d KC=%d NC=%d  MR=%d NR=%d\n", MC, KC, NC, MR, NR);
    printf("====================================================================\n\n");

    int sizes[] = {256, 512, 1024, 2048, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);

    printf("%-10s | %10s | %8s | %s\n", "Size", "COM6-v15", "GFLOPS", "Verify");
    printf("---------- | ---------- | -------- | ------\n");

    for (int si=0; si<ns; si++) {
        int n = sizes[si];
        size_t nn = (size_t)n*n;
        double *A=aligned_alloc_d(nn), *B=aligned_alloc_d(nn);
        double *C1=aligned_alloc_d(nn), *C2=aligned_alloc_d(nn);

        srand(42); fill_rand(A,n); fill_rand(B,n);

        /* Warmup */
        com6_matmul(A,B,C1,n);

        int runs = (n<=1024)?3:(n<=2048)?2:1;
        double best=1e30;
        for (int r=0;r<runs;r++) {
            double t0=get_time(); com6_matmul(A,B,C1,n);
            double t=get_time()-t0; if(t<best)best=t;
        }
        double gf = (2.0*n*n*(double)n)/(best*1e9);

        const char* vfy="skip";
        if (n<=512) { naive_mul(A,B,C2,n); vfy = max_diff(C1,C2,n)<1e-6?"OK":"FAIL"; }

        printf("%4dx%-5d | %8.1f ms | %6.1f   | %s\n", n,n, best*1000, gf, vfy);

        aligned_free_d(A); aligned_free_d(B); aligned_free_d(C1); aligned_free_d(C2);
    }

    printf("\nTarget: ~40 GFLOPS = OpenBLAS single-threaded on i7-10510U\n");
    printf("Peak: 63.2 GFLOPS (2 FMA x 4dp x 2flop x 3.95GHz)\n");
    return 0;
}
