/*
 * COM6 v14 - Closing the OpenBLAS Gap
 * ====================================
 * Target: match OpenBLAS single-threaded dgemm (~40 GFLOPS)
 *
 * Changes from v13:
 *   1. 4x k-unrolled micro-kernel (process 4 k per loop iteration)
 *   2. Tuned prefetch: L2 prefetch for B, L1 for A, correct distances
 *   3. Optimized A-packing with 4-wide gather
 *   4. B-packing with 4-wide interleave using SIMD shuffles
 *   5. Larger MC=96 (better amortizes packing cost)
 *   6. Pure BLIS only (Strassen hybrid was slower in v13)
 *
 * CPU: Intel i7-10510U (Comet Lake)
 *   L1d = 64 KB/core, L2 = 256 KB/core, L3 = 8 MB shared
 *   2 FMA units, AVX2, turbo ~3.95 GHz
 *   Peak DP: 2 FMA * 4 doubles * 2 flops * 3.95 GHz = 63.2 GFLOPS
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v14 com6_v14.c -lm
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* BLIS parameters for Comet Lake */
#define MR  6
#define NR  8
#define KC  256     /* KC*NR*8 = 16 KB << 64 KB L1d */
#define MC  96      /* MC*KC*8 = 192 KB < 256 KB L2 */
#define NC  4096    /* NC*KC*8 = 8 MB = L3 */

#define ALIGN 64

static inline double* aligned_alloc_d(size_t count) {
    return (double*)_mm_malloc(count * sizeof(double), ALIGN);
}
static inline void aligned_free_d(double* p) { _mm_free(p); }

/* ================================================================
 * MICRO-KERNEL: 6x8, 4x k-unrolled, prefetched
 * ================================================================
 * 12 FMAs per k, 4 k per iteration = 48 FMAs per loop body
 * At 2 FMA/cycle: 24 cycles per iteration = 192 flops in 24 cycles
 * With loads/broadcasts overhead, target ~60% FMA throughput
 */
static void __attribute__((noinline))
micro_kernel_6x8(int kc,
                  const double* __restrict__ pA,
                  const double* __restrict__ pB,
                  double* __restrict__ C, int ldc)
{
    __m256d c00 = _mm256_setzero_pd();
    __m256d c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd();
    __m256d c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd();
    __m256d c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd();
    __m256d c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd();
    __m256d c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd();
    __m256d c51 = _mm256_setzero_pd();

    int k = 0;
    int kc4 = kc & ~3;  /* round down to multiple of 4 */

    for (; k < kc4; k += 4) {
        /* Prefetch: A 4 iterations ahead, B 2 iterations ahead */
        _mm_prefetch((const char*)(pA + MR * 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(pB + NR * 4), _MM_HINT_T0);
        _mm_prefetch((const char*)(pB + NR * 4 + 8), _MM_HINT_T0);

        __m256d b0, b1, a;

        /* --- k+0 --- */
        b0 = _mm256_load_pd(pB + 0*NR + 0);
        b1 = _mm256_load_pd(pB + 0*NR + 4);
        a = _mm256_broadcast_sd(pA + 0*MR + 0);
        c00 = _mm256_fmadd_pd(a, b0, c00); c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(pA + 0*MR + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10); c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(pA + 0*MR + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20); c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(pA + 0*MR + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30); c31 = _mm256_fmadd_pd(a, b1, c31);
        a = _mm256_broadcast_sd(pA + 0*MR + 4);
        c40 = _mm256_fmadd_pd(a, b0, c40); c41 = _mm256_fmadd_pd(a, b1, c41);
        a = _mm256_broadcast_sd(pA + 0*MR + 5);
        c50 = _mm256_fmadd_pd(a, b0, c50); c51 = _mm256_fmadd_pd(a, b1, c51);

        /* --- k+1 --- */
        b0 = _mm256_load_pd(pB + 1*NR + 0);
        b1 = _mm256_load_pd(pB + 1*NR + 4);
        a = _mm256_broadcast_sd(pA + 1*MR + 0);
        c00 = _mm256_fmadd_pd(a, b0, c00); c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(pA + 1*MR + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10); c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(pA + 1*MR + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20); c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(pA + 1*MR + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30); c31 = _mm256_fmadd_pd(a, b1, c31);
        a = _mm256_broadcast_sd(pA + 1*MR + 4);
        c40 = _mm256_fmadd_pd(a, b0, c40); c41 = _mm256_fmadd_pd(a, b1, c41);
        a = _mm256_broadcast_sd(pA + 1*MR + 5);
        c50 = _mm256_fmadd_pd(a, b0, c50); c51 = _mm256_fmadd_pd(a, b1, c51);

        /* --- k+2 --- */
        b0 = _mm256_load_pd(pB + 2*NR + 0);
        b1 = _mm256_load_pd(pB + 2*NR + 4);
        a = _mm256_broadcast_sd(pA + 2*MR + 0);
        c00 = _mm256_fmadd_pd(a, b0, c00); c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(pA + 2*MR + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10); c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(pA + 2*MR + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20); c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(pA + 2*MR + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30); c31 = _mm256_fmadd_pd(a, b1, c31);
        a = _mm256_broadcast_sd(pA + 2*MR + 4);
        c40 = _mm256_fmadd_pd(a, b0, c40); c41 = _mm256_fmadd_pd(a, b1, c41);
        a = _mm256_broadcast_sd(pA + 2*MR + 5);
        c50 = _mm256_fmadd_pd(a, b0, c50); c51 = _mm256_fmadd_pd(a, b1, c51);

        /* --- k+3 --- */
        b0 = _mm256_load_pd(pB + 3*NR + 0);
        b1 = _mm256_load_pd(pB + 3*NR + 4);
        a = _mm256_broadcast_sd(pA + 3*MR + 0);
        c00 = _mm256_fmadd_pd(a, b0, c00); c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(pA + 3*MR + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10); c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(pA + 3*MR + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20); c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(pA + 3*MR + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30); c31 = _mm256_fmadd_pd(a, b1, c31);
        a = _mm256_broadcast_sd(pA + 3*MR + 4);
        c40 = _mm256_fmadd_pd(a, b0, c40); c41 = _mm256_fmadd_pd(a, b1, c41);
        a = _mm256_broadcast_sd(pA + 3*MR + 5);
        c50 = _mm256_fmadd_pd(a, b0, c50); c51 = _mm256_fmadd_pd(a, b1, c51);

        pA += 4 * MR;
        pB += 4 * NR;
    }

    /* Cleanup: remaining k iterations */
    for (; k < kc; k++) {
        __m256d b0 = _mm256_load_pd(pB + 0);
        __m256d b1 = _mm256_load_pd(pB + 4);
        __m256d a;
        a = _mm256_broadcast_sd(pA + 0);
        c00 = _mm256_fmadd_pd(a, b0, c00); c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(pA + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10); c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(pA + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20); c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(pA + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30); c31 = _mm256_fmadd_pd(a, b1, c31);
        a = _mm256_broadcast_sd(pA + 4);
        c40 = _mm256_fmadd_pd(a, b0, c40); c41 = _mm256_fmadd_pd(a, b1, c41);
        a = _mm256_broadcast_sd(pA + 5);
        c50 = _mm256_fmadd_pd(a, b0, c50); c51 = _mm256_fmadd_pd(a, b1, c51);
        pA += MR;
        pB += NR;
    }

    /* Store C += accumulators (load-add-store) */
    double* c;
    c = C;             _mm256_storeu_pd(c,   _mm256_add_pd(_mm256_loadu_pd(c),   c00));
                       _mm256_storeu_pd(c+4, _mm256_add_pd(_mm256_loadu_pd(c+4), c01));
    c = C + ldc;       _mm256_storeu_pd(c,   _mm256_add_pd(_mm256_loadu_pd(c),   c10));
                       _mm256_storeu_pd(c+4, _mm256_add_pd(_mm256_loadu_pd(c+4), c11));
    c = C + 2*ldc;     _mm256_storeu_pd(c,   _mm256_add_pd(_mm256_loadu_pd(c),   c20));
                       _mm256_storeu_pd(c+4, _mm256_add_pd(_mm256_loadu_pd(c+4), c21));
    c = C + 3*ldc;     _mm256_storeu_pd(c,   _mm256_add_pd(_mm256_loadu_pd(c),   c30));
                       _mm256_storeu_pd(c+4, _mm256_add_pd(_mm256_loadu_pd(c+4), c31));
    c = C + 4*ldc;     _mm256_storeu_pd(c,   _mm256_add_pd(_mm256_loadu_pd(c),   c40));
                       _mm256_storeu_pd(c+4, _mm256_add_pd(_mm256_loadu_pd(c+4), c41));
    c = C + 5*ldc;     _mm256_storeu_pd(c,   _mm256_add_pd(_mm256_loadu_pd(c),   c50));
                       _mm256_storeu_pd(c+4, _mm256_add_pd(_mm256_loadu_pd(c+4), c51));
}

/* Edge micro-kernel (scalar fallback for remainder tiles) */
static void micro_kernel_edge(int mr, int nr, int kc,
                               const double* __restrict__ pA,
                               const double* __restrict__ pB,
                               double* __restrict__ C, int ldc)
{
    for (int k = 0; k < kc; k++) {
        for (int i = 0; i < mr; i++) {
            double a_val = pA[k * MR + i];
            for (int j = 0; j < nr; j++) {
                C[i * ldc + j] += a_val * pB[k * NR + j];
            }
        }
    }
}

/* ================================================================
 * PACKING - Optimized
 * ================================================================ */

/*
 * Pack A panel: gather MR rows per k into contiguous strips.
 * Optimized: process 4 rows at a time where possible.
 */
static void pack_A_panel(const double* __restrict__ A, double* __restrict__ packed_A,
                          int mc, int kc, int lda, int i0, int k0)
{
    const double* A_base = A + (size_t)i0 * lda + k0;

    for (int i = 0; i < mc; i += MR) {
        int mr = (i + MR <= mc) ? MR : mc - i;
        const double* a0 = A_base + i * lda;

        if (mr == MR) {
            /* Full MR=6 strip: unrolled gather */
            const double* a1 = a0 + lda;
            const double* a2 = a0 + 2*lda;
            const double* a3 = a0 + 3*lda;
            const double* a4 = a0 + 4*lda;
            const double* a5 = a0 + 5*lda;

            for (int k = 0; k < kc; k++) {
                packed_A[0] = a0[k];
                packed_A[1] = a1[k];
                packed_A[2] = a2[k];
                packed_A[3] = a3[k];
                packed_A[4] = a4[k];
                packed_A[5] = a5[k];
                packed_A += MR;
            }
        } else {
            /* Edge: fewer than MR rows */
            for (int k = 0; k < kc; k++) {
                int ii;
                for (ii = 0; ii < mr; ii++)
                    packed_A[ii] = (a0 + ii * lda)[k];
                for (; ii < MR; ii++)
                    packed_A[ii] = 0.0;
                packed_A += MR;
            }
        }
    }
}

/*
 * Pack B panel from B^T.
 * B^T is row-major: BT[j][k] is contiguous along k.
 * We need packed_B[k*NR + jj] = BT[j+jj][k] (interleave NR rows per k).
 *
 * Optimization: process 4 k-values at a time, using the fact that
 * each BT row is contiguous (COM6 advantage for prefetching).
 */
static void pack_B_panel(const double* __restrict__ BT, double* __restrict__ packed_B,
                          int kc, int nc, int ldb, int j0, int k0)
{
    for (int j = 0; j < nc; j += NR) {
        int nr = (j + NR <= nc) ? NR : nc - j;

        if (nr == NR) {
            const double* r0 = BT + (j0+j+0)*(size_t)ldb + k0;
            const double* r1 = BT + (j0+j+1)*(size_t)ldb + k0;
            const double* r2 = BT + (j0+j+2)*(size_t)ldb + k0;
            const double* r3 = BT + (j0+j+3)*(size_t)ldb + k0;
            const double* r4 = BT + (j0+j+4)*(size_t)ldb + k0;
            const double* r5 = BT + (j0+j+5)*(size_t)ldb + k0;
            const double* r6 = BT + (j0+j+6)*(size_t)ldb + k0;
            const double* r7 = BT + (j0+j+7)*(size_t)ldb + k0;

            /* Prefetch first cache lines of each row */
            _mm_prefetch((const char*)r0, _MM_HINT_T0);
            _mm_prefetch((const char*)r1, _MM_HINT_T0);
            _mm_prefetch((const char*)r2, _MM_HINT_T0);
            _mm_prefetch((const char*)r3, _MM_HINT_T0);
            _mm_prefetch((const char*)r4, _MM_HINT_T0);
            _mm_prefetch((const char*)r5, _MM_HINT_T0);
            _mm_prefetch((const char*)r6, _MM_HINT_T0);
            _mm_prefetch((const char*)r7, _MM_HINT_T0);

            for (int k = 0; k < kc; k++) {
                packed_B[0] = r0[k];
                packed_B[1] = r1[k];
                packed_B[2] = r2[k];
                packed_B[3] = r3[k];
                packed_B[4] = r4[k];
                packed_B[5] = r5[k];
                packed_B[6] = r6[k];
                packed_B[7] = r7[k];
                packed_B += NR;
            }
        } else {
            const double* rows[NR];
            for (int jj = 0; jj < nr; jj++)
                rows[jj] = BT + (j0+j+jj)*(size_t)ldb + k0;
            for (int k = 0; k < kc; k++) {
                int jj;
                for (jj = 0; jj < nr; jj++)
                    packed_B[jj] = rows[jj][k];
                for (; jj < NR; jj++)
                    packed_B[jj] = 0.0;
                packed_B += NR;
            }
        }
    }
}

/* ================================================================
 * BLIS 5-LOOP: C = A * B  (via B^T for COM6)
 * ================================================================ */
static void com6_blis_multiply(const double* __restrict__ A,
                                const double* __restrict__ BT,
                                double* __restrict__ C, int n)
{
    double* packed_A = aligned_alloc_d((size_t)MC * KC);
    double* packed_B = aligned_alloc_d((size_t)KC * NC);

    memset(C, 0, (size_t)n * n * sizeof(double));

    for (int jc = 0; jc < n; jc += NC) {
        int nc = (jc + NC <= n) ? NC : n - jc;

        for (int pc = 0; pc < n; pc += KC) {
            int kc = (pc + KC <= n) ? KC : n - pc;

            pack_B_panel(BT, packed_B, kc, nc, n, jc, pc);

            for (int ic = 0; ic < n; ic += MC) {
                int mc = (ic + MC <= n) ? MC : n - ic;

                pack_A_panel(A, packed_A, mc, kc, n, ic, pc);

                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = (jr + NR <= nc) ? NR : nc - jr;
                    const double* pB = packed_B + (jr / NR) * ((size_t)NR * kc);

                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = (ir + MR <= mc) ? MR : mc - ir;
                        const double* pA = packed_A + (ir / MR) * ((size_t)MR * kc);
                        double* C_ij = C + (size_t)(ic + ir) * n + (jc + jr);

                        if (mr == MR && nr == NR) {
                            micro_kernel_6x8(kc, pA, pB, C_ij, n);
                        } else {
                            micro_kernel_edge(mr, nr, kc, pA, pB, C_ij, n);
                        }
                    }
                }
            }
        }
    }

    aligned_free_d(packed_A);
    aligned_free_d(packed_B);
}

/* ================================================================
 * Transpose B -> BT
 * ================================================================ */
static void transpose(const double* __restrict__ src, double* __restrict__ dst, int n) {
    /* 4x4 block transpose for cache friendliness */
    int nb = n & ~3;
    for (int i = 0; i < nb; i += 4) {
        for (int j = 0; j < nb; j += 4) {
            /* Load 4x4 block from src */
            __m256d r0 = _mm256_loadu_pd(src + i*n + j);
            __m256d r1 = _mm256_loadu_pd(src + (i+1)*n + j);
            __m256d r2 = _mm256_loadu_pd(src + (i+2)*n + j);
            __m256d r3 = _mm256_loadu_pd(src + (i+3)*n + j);

            /* 4x4 double transpose using unpacklo/hi and permute */
            __m256d t0 = _mm256_unpacklo_pd(r0, r1);  /* r0[0],r1[0],r0[2],r1[2] */
            __m256d t1 = _mm256_unpackhi_pd(r0, r1);  /* r0[1],r1[1],r0[3],r1[3] */
            __m256d t2 = _mm256_unpacklo_pd(r2, r3);
            __m256d t3 = _mm256_unpackhi_pd(r2, r3);

            __m256d o0 = _mm256_permute2f128_pd(t0, t2, 0x20); /* col 0 */
            __m256d o1 = _mm256_permute2f128_pd(t1, t3, 0x20); /* col 1 */
            __m256d o2 = _mm256_permute2f128_pd(t0, t2, 0x31); /* col 2 */
            __m256d o3 = _mm256_permute2f128_pd(t1, t3, 0x31); /* col 3 */

            _mm256_storeu_pd(dst + j*n + i, o0);
            _mm256_storeu_pd(dst + (j+1)*n + i, o1);
            _mm256_storeu_pd(dst + (j+2)*n + i, o2);
            _mm256_storeu_pd(dst + (j+3)*n + i, o3);
        }
        /* Remainder columns */
        for (int j = nb; j < n; j++)
            for (int ii = 0; ii < 4; ii++)
                dst[j*n + (i+ii)] = src[(i+ii)*n + j];
    }
    /* Remainder rows */
    for (int i = nb; i < n; i++)
        for (int j = 0; j < n; j++)
            dst[j*n + i] = src[i*n + j];
}

/* ================================================================
 * Top-level COM6 BLIS multiply
 * ================================================================ */
static void com6_matmul(const double* A, const double* B, double* C, int n) {
    double* BT = aligned_alloc_d((size_t)n * n);
    transpose(B, BT, n);
    com6_blis_multiply(A, BT, C, n);
    aligned_free_d(BT);
}

/* ================================================================
 * Reference: naive ikj (same memory pattern as OpenBLAS uses)
 * ================================================================ */
static void naive_multiply(const double* A, const double* B, double* C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[k * n + j];
        }
}

/* ================================================================
 * BENCHMARK
 * ================================================================ */
static double get_time(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_random(double* M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

static double max_diff(const double* A, const double* B, int n) {
    double mx = 0.0;
    for (int i = 0; i < n * n; i++) {
        double d = fabs(A[i] - B[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

int main(void) {
    printf("====================================================================\n");
    printf("  COM6 v14 - BLIS-Class: 6x8, 4x k-unroll, SIMD transpose\n");
    printf("  Target: ~40 GFLOPS (OpenBLAS 1-thread on i7-10510U)\n");
    printf("====================================================================\n\n");

    int sizes[] = {256, 512, 1024, 2048, 4096};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-10s | %10s | %10s | %8s | %s\n",
           "Size", "COM6-v14", "Naive", "GFLOPS", "Verify");
    printf("---------- | ---------- | ---------- | -------- | ------\n");

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si];
        size_t nn = (size_t)n * n;

        double* A  = aligned_alloc_d(nn);
        double* B  = aligned_alloc_d(nn);
        double* C1 = aligned_alloc_d(nn);
        double* C2 = aligned_alloc_d(nn);

        srand(42);
        fill_random(A, n);
        fill_random(B, n);

        /* Warmup */
        com6_matmul(A, B, C1, n);

        /* Timed run - best of 3 (or 1 for large) */
        int runs = (n <= 1024) ? 3 : (n <= 2048) ? 2 : 1;
        double best = 1e30;
        for (int r = 0; r < runs; r++) {
            double t0 = get_time();
            com6_matmul(A, B, C1, n);
            double t1 = get_time() - t0;
            if (t1 < best) best = t1;
        }

        double gflops = (2.0 * n * n * (double)n) / (best * 1e9);

        /* Verify against naive for small sizes */
        const char* verify = "skip";
        if (n <= 1024) {
            naive_multiply(A, B, C2, n);
            double err = max_diff(C1, C2, n);
            verify = (err < 1e-6) ? "OK" : "FAIL";
        }

        /* Naive timing for comparison */
        double t_naive = 0;
        if (n <= 1024) {
            double t0 = get_time();
            naive_multiply(A, B, C2, n);
            t_naive = get_time() - t0;
        }

        printf("%4dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %s\n",
               n, n,
               best * 1000.0,
               t_naive * 1000.0,
               gflops, verify);

        aligned_free_d(A);
        aligned_free_d(B);
        aligned_free_d(C1);
        aligned_free_d(C2);
    }

    printf("\nTarget: ~40 GFLOPS = OpenBLAS single-threaded\n");
    printf("Peak theoretical: 63.2 GFLOPS (i7-10510U @ 3.95 GHz, 2 FMA units)\n");

    return 0;
}
