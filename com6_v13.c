/*
 * COM6 v13 - BLIS-Class Implementation
 * =====================================
 * Target: match or beat OpenBLAS single-threaded dgemm
 *
 * Key changes from v12:
 *   1. 6x8 outer-product micro-kernel (12 YMM accumulators, broadcast A + load B)
 *   2. A-panel packing into MR-wide contiguous strips
 *   3. B-panel packing from B^T (COM6 advantage: row-contiguous source)
 *   4. BLIS 5-loop nest with MC/KC/NC tuned for Kaby Lake cache hierarchy
 *   5. Prefetch hints in micro-kernel
 *
 * Cache model (Intel Kaby Lake / Coffee Lake):
 *   L1d = 32 KB, L2 = 256 KB, L3 = 6-8 MB
 *   MC*KC*8 should fit in L2 (~144 KB)
 *   KC*NR*8 should fit in L1 (~16 KB)
 *   NC*KC*8 should fit in L3
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v13 com6_v13.c -lm
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- BLIS blocking parameters ---- */
#define MR  6
#define NR  8
#define KC  256     /* KC*NR*8 = 16 KB fits in 64KB L1d per core */
#define MC  72      /* MC*KC*8 = 144 KB fits in 256KB L2 per core */
#define NC  4096    /* NC*KC*8 = 8 MB fits in 8MB L3 */

#define ALIGN 64

static inline double* aligned_alloc_d(size_t count) {
    return (double*)_mm_malloc(count * sizeof(double), ALIGN);
}

static inline void aligned_free_d(double* p) {
    _mm_free(p);
}

/* ================================================================
 * MICRO-KERNEL: 6x8 outer-product, AVX2 FMA (doubles)
 * ================================================================
 * Computes C[6][8] += packed_A[6 x kc] * packed_B[kc x 8]
 *
 * Register map (16 YMM total):
 *   c00..c50 = 6 accumulators for cols 0-3   (6 regs)
 *   c01..c51 = 6 accumulators for cols 4-7   (6 regs)
 *   b0, b1   = B panel loads                  (2 regs)
 *   a_bc     = A broadcast                    (1 reg)
 *   spare    =                                (1 reg)
 *   Total = 16 YMM registers (fully utilized)
 */
static void micro_kernel_6x8(int kc,
                              const double* __restrict__ packed_A,
                              const double* __restrict__ packed_B,
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

    for (int k = 0; k < kc; k++) {
        /* Prefetch next iteration's A and B */
        _mm_prefetch((const char*)(packed_A + MR * 2), _MM_HINT_T0);
        _mm_prefetch((const char*)(packed_B + NR * 2), _MM_HINT_T0);

        /* Load B panel: 8 doubles = 2 YMM */
        __m256d b0 = _mm256_load_pd(packed_B);
        __m256d b1 = _mm256_load_pd(packed_B + 4);

        __m256d a;

        /* Row 0 */
        a = _mm256_broadcast_sd(packed_A + 0);
        c00 = _mm256_fmadd_pd(a, b0, c00);
        c01 = _mm256_fmadd_pd(a, b1, c01);

        /* Row 1 */
        a = _mm256_broadcast_sd(packed_A + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10);
        c11 = _mm256_fmadd_pd(a, b1, c11);

        /* Row 2 */
        a = _mm256_broadcast_sd(packed_A + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20);
        c21 = _mm256_fmadd_pd(a, b1, c21);

        /* Row 3 */
        a = _mm256_broadcast_sd(packed_A + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30);
        c31 = _mm256_fmadd_pd(a, b1, c31);

        /* Row 4 */
        a = _mm256_broadcast_sd(packed_A + 4);
        c40 = _mm256_fmadd_pd(a, b0, c40);
        c41 = _mm256_fmadd_pd(a, b1, c41);

        /* Row 5 */
        a = _mm256_broadcast_sd(packed_A + 5);
        c50 = _mm256_fmadd_pd(a, b0, c50);
        c51 = _mm256_fmadd_pd(a, b1, c51);

        packed_A += MR;
        packed_B += NR;
    }

    /* Store C += accumulators */
    _mm256_storeu_pd(C + 0*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 0*ldc + 0), c00));
    _mm256_storeu_pd(C + 0*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 0*ldc + 4), c01));
    _mm256_storeu_pd(C + 1*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 1*ldc + 0), c10));
    _mm256_storeu_pd(C + 1*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 1*ldc + 4), c11));
    _mm256_storeu_pd(C + 2*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 2*ldc + 0), c20));
    _mm256_storeu_pd(C + 2*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 2*ldc + 4), c21));
    _mm256_storeu_pd(C + 3*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 3*ldc + 0), c30));
    _mm256_storeu_pd(C + 3*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 3*ldc + 4), c31));
    _mm256_storeu_pd(C + 4*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 4*ldc + 0), c40));
    _mm256_storeu_pd(C + 4*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 4*ldc + 4), c41));
    _mm256_storeu_pd(C + 5*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 5*ldc + 0), c50));
    _mm256_storeu_pd(C + 5*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 5*ldc + 4), c51));
}

/* Edge micro-kernel for non-MR/NR-aligned remainders */
static void micro_kernel_edge(int mr, int nr, int kc,
                               const double* __restrict__ packed_A,
                               const double* __restrict__ packed_B,
                               double* __restrict__ C, int ldc)
{
    /* Simple scalar fallback for edge tiles */
    for (int k = 0; k < kc; k++) {
        for (int i = 0; i < mr; i++) {
            double a_val = packed_A[k * MR + i];
            for (int j = 0; j < nr; j++) {
                C[i * ldc + j] += a_val * packed_B[k * NR + j];
            }
        }
    }
}

/* ================================================================
 * PACKING ROUTINES
 * ================================================================ */

/*
 * Pack A panel: rows [i0, i0+mc) x cols [k0, k0+kc) into MR-wide strips.
 * Output: packed_A[strip][k][mr] stored contiguously.
 * For each k, MR consecutive elements from MR consecutive rows.
 */
static void pack_A_panel(const double* A, double* packed_A,
                          int mc, int kc, int lda, int i0, int k0)
{
    for (int i = 0; i < mc; i += MR) {
        int mr = (i + MR <= mc) ? MR : mc - i;
        for (int k = 0; k < kc; k++) {
            int ii;
            for (ii = 0; ii < mr; ii++) {
                packed_A[ii] = A[(i0 + i + ii) * lda + (k0 + k)];
            }
            for (; ii < MR; ii++) {
                packed_A[ii] = 0.0;
            }
            packed_A += MR;
        }
    }
}

/*
 * Pack B panel from B^T: rows [j0, j0+nc) of BT, cols [k0, k0+kc)
 * into NR-wide strips for outer-product micro-kernel.
 *
 * COM6 ADVANTAGE: BT is row-major, so BT[j][k] is contiguous along k.
 * We pack into: packed_B[strip][k][nr] format.
 * For each k, we gather NR elements from NR different rows of BT.
 *
 * The actual packing pattern:
 *   packed_B[k * NR + jj] = BT[(j0 + j + jj) * ldb + (k0 + k)]
 *
 * Since BT rows are contiguous, we can optimize with partial vectorization.
 */
static void pack_B_panel(const double* BT, double* packed_B,
                          int kc, int nc, int ldb, int j0, int k0)
{
    for (int j = 0; j < nc; j += NR) {
        int nr = (j + NR <= nc) ? NR : nc - j;
        /* Pointers to the NR rows of BT we need */
        const double* rows[NR];
        int jj;
        for (jj = 0; jj < nr; jj++) {
            rows[jj] = BT + (j0 + j + jj) * ldb + k0;
        }

        if (nr == NR) {
            /* Full NR=8 strip: interleave 8 rows */
            for (int k = 0; k < kc; k++) {
                packed_B[0] = rows[0][k];
                packed_B[1] = rows[1][k];
                packed_B[2] = rows[2][k];
                packed_B[3] = rows[3][k];
                packed_B[4] = rows[4][k];
                packed_B[5] = rows[5][k];
                packed_B[6] = rows[6][k];
                packed_B[7] = rows[7][k];
                packed_B += NR;
            }
        } else {
            /* Edge case: fewer than NR columns */
            for (int k = 0; k < kc; k++) {
                for (jj = 0; jj < nr; jj++) {
                    packed_B[jj] = rows[jj][k];
                }
                for (; jj < NR; jj++) {
                    packed_B[jj] = 0.0;
                }
                packed_B += NR;
            }
        }
    }
}

/* ================================================================
 * BLIS 5-LOOP MATRIX MULTIPLY: C = A * B  (via B^T)
 * ================================================================
 * Loop order: jc -> pc -> ic -> jr -> ir -> micro-kernel
 *
 *   jc: partition N into NC-wide panels (L3)
 *   pc: partition K into KC-deep slabs (L2/L1 boundary)
 *   ic: partition M into MC-tall panels (L2)
 *   jr: step NR across packed B panel
 *   ir: step MR down packed A panel
 */
static void com6_blis_multiply(const double* A, const double* BT,
                                double* C, int n)
{
    /* Allocate packing buffers */
    double* packed_A = aligned_alloc_d(MC * KC);
    double* packed_B = aligned_alloc_d(KC * NC);

    /* Zero C */
    memset(C, 0, (size_t)n * n * sizeof(double));

    /* jc loop: partition columns (N dimension) */
    for (int jc = 0; jc < n; jc += NC) {
        int nc = (jc + NC <= n) ? NC : n - jc;

        /* pc loop: partition depth (K dimension) */
        for (int pc = 0; pc < n; pc += KC) {
            int kc = (pc + KC <= n) ? KC : n - pc;

            /* Pack B panel: BT rows [jc, jc+nc), cols [pc, pc+kc) */
            pack_B_panel(BT, packed_B, kc, nc, n, jc, pc);

            /* ic loop: partition rows (M dimension) */
            for (int ic = 0; ic < n; ic += MC) {
                int mc = (ic + MC <= n) ? MC : n - ic;

                /* Pack A panel: A rows [ic, ic+mc), cols [pc, pc+kc) */
                pack_A_panel(A, packed_A, mc, kc, n, ic, pc);

                /* jr loop: step NR across packed B */
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = (jr + NR <= nc) ? NR : nc - jr;

                    /* ir loop: step MR down packed A */
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = (ir + MR <= mc) ? MR : mc - ir;

                        /* Micro-kernel: C[ic+ir : +mr][jc+jr : +nr] += A_panel * B_panel */
                        double* C_ij = C + (ic + ir) * n + (jc + jr);
                        const double* pA = packed_A + (ir / MR) * (MR * kc);
                        /* Wait, this isn't right. packed_A stores strips sequentially:
                           strip 0: MR*kc elements, strip 1: MR*kc elements, etc.
                           Strip index = ir / MR */
                        const double* pB = packed_B + (jr / NR) * (NR * kc);

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
 * STRASSEN + BLIS HYBRID
 * ================================================================
 * Use Strassen at the top level(s) for O(n^2.807) complexity,
 * then BLIS micro-kernel for base cases.
 */

/* Memory pool for Strassen temporaries */
typedef struct {
    double* base;
    size_t  used;
    size_t  capacity;
} Pool;

static Pool pool;

static void pool_init(size_t bytes) {
    pool.base = aligned_alloc_d(bytes / sizeof(double) + 1);
    pool.used = 0;
    pool.capacity = bytes;
}

static void pool_destroy(void) {
    aligned_free_d(pool.base);
}

static double* pool_alloc(size_t count) {
    /* Align to 64 bytes */
    size_t bytes = count * sizeof(double);
    size_t aligned_used = (pool.used + 63) & ~(size_t)63;
    if (aligned_used + bytes > pool.capacity) {
        fprintf(stderr, "Pool exhausted: need %zu, have %zu\n",
                aligned_used + bytes, pool.capacity);
        exit(1);
    }
    double* ptr = (double*)((char*)pool.base + aligned_used);
    pool.used = aligned_used + bytes;
    return ptr;
}

/* Extract quadrant: dst[h x h] from src[n x n] at (row_off, col_off) */
static void extract_quad(const double* src, int n, double* dst, int h,
                          int row_off, int col_off)
{
    for (int i = 0; i < h; i++)
        memcpy(dst + i * h, src + (row_off + i) * n + col_off, h * sizeof(double));
}

/* Insert quadrant: src[h x h] into dst[n x n] at (row_off, col_off) */
static void insert_quad(double* dst, int n, const double* src, int h,
                         int row_off, int col_off)
{
    for (int i = 0; i < h; i++)
        memcpy(dst + (row_off + i) * n + col_off, src + i * h, h * sizeof(double));
}

/* Matrix add: C = A + B (all h x h) */
static void mat_add(const double* A, const double* B, double* C, int h) {
    int total = h * h;
    int k = 0;
    for (; k + 3 < total; k += 4) {
        __m256d a = _mm256_loadu_pd(A + k);
        __m256d b = _mm256_loadu_pd(B + k);
        _mm256_storeu_pd(C + k, _mm256_add_pd(a, b));
    }
    for (; k < total; k++) C[k] = A[k] + B[k];
}

/* Matrix sub: C = A - B (all h x h) */
static void mat_sub(const double* A, const double* B, double* C, int h) {
    int total = h * h;
    int k = 0;
    for (; k + 3 < total; k += 4) {
        __m256d a = _mm256_loadu_pd(A + k);
        __m256d b = _mm256_loadu_pd(B + k);
        _mm256_storeu_pd(C + k, _mm256_sub_pd(a, b));
    }
    for (; k < total; k++) C[k] = A[k] - B[k];
}

/* Forward declaration */
static void strassen_blis(const double* A, const double* B,
                           double* C, int n);

/* Transpose n x n matrix */
static void transpose(const double* src, double* dst, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dst[j * n + i] = src[i * n + j];
}

/* BLIS base case: compute C = A * B using COM6 BLIS method */
static void blis_base(const double* A, const double* B, double* C, int n) {
    /* Transpose B for COM6 access pattern */
    double* BT = pool_alloc((size_t)n * n);
    transpose(B, BT, n);
    com6_blis_multiply(A, BT, C, n);
}

#define STRASSEN_THRESHOLD 256

static void strassen_blis(const double* A, const double* B,
                           double* C, int n)
{
    if (n <= STRASSEN_THRESHOLD) {
        blis_base(A, B, C, n);
        return;
    }

    int h = n / 2;
    size_t hsq = (size_t)h * h;

    /* Save pool state for rollback */
    size_t pool_save = pool.used;

    /* Extract quadrants */
    double* A11 = pool_alloc(hsq); double* A12 = pool_alloc(hsq);
    double* A21 = pool_alloc(hsq); double* A22 = pool_alloc(hsq);
    double* B11 = pool_alloc(hsq); double* B12 = pool_alloc(hsq);
    double* B21 = pool_alloc(hsq); double* B22 = pool_alloc(hsq);

    extract_quad(A, n, A11, h, 0, 0); extract_quad(A, n, A12, h, 0, h);
    extract_quad(A, n, A21, h, h, 0); extract_quad(A, n, A22, h, h, h);
    extract_quad(B, n, B11, h, 0, 0); extract_quad(B, n, B12, h, 0, h);
    extract_quad(B, n, B21, h, h, 0); extract_quad(B, n, B22, h, h, h);

    double* T1 = pool_alloc(hsq);
    double* T2 = pool_alloc(hsq);
    double* M1 = pool_alloc(hsq); double* M2 = pool_alloc(hsq);
    double* M3 = pool_alloc(hsq); double* M4 = pool_alloc(hsq);
    double* M5 = pool_alloc(hsq); double* M6 = pool_alloc(hsq);
    double* M7 = pool_alloc(hsq);

    /* M1 = (A11+A22)(B11+B22) */
    mat_add(A11, A22, T1, h); mat_add(B11, B22, T2, h);
    strassen_blis(T1, T2, M1, h);

    /* M2 = (A21+A22)*B11 */
    mat_add(A21, A22, T1, h);
    strassen_blis(T1, B11, M2, h);

    /* M3 = A11*(B12-B22) */
    mat_sub(B12, B22, T1, h);
    strassen_blis(A11, T1, M3, h);

    /* M4 = A22*(B21-B11) */
    mat_sub(B21, B11, T1, h);
    strassen_blis(A22, T1, M4, h);

    /* M5 = (A11+A12)*B22 */
    mat_add(A11, A12, T1, h);
    strassen_blis(T1, B22, M5, h);

    /* M6 = (A21-A11)*(B11+B12) */
    mat_sub(A21, A11, T1, h); mat_add(B11, B12, T2, h);
    strassen_blis(T1, T2, M6, h);

    /* M7 = (A12-A22)*(B21+B22) */
    mat_sub(A12, A22, T1, h); mat_add(B21, B22, T2, h);
    strassen_blis(T1, T2, M7, h);

    /* Combine: C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6 */
    double* C11 = T1;  /* reuse T1 */
    double* C12 = T2;  /* reuse T2 */
    double* C21 = pool_alloc(hsq);
    double* C22 = pool_alloc(hsq);

    /* C11 = M1+M4-M5+M7 */
    for (size_t i = 0; i < hsq; i++)
        C11[i] = M1[i] + M4[i] - M5[i] + M7[i];
    /* C12 = M3+M5 */
    mat_add(M3, M5, C12, h);
    /* C21 = M2+M4 */
    mat_add(M2, M4, C21, h);
    /* C22 = M1-M2+M3+M6 */
    for (size_t i = 0; i < hsq; i++)
        C22[i] = M1[i] - M2[i] + M3[i] + M6[i];

    insert_quad(C, n, C11, h, 0, 0);
    insert_quad(C, n, C12, h, 0, h);
    insert_quad(C, n, C21, h, h, 0);
    insert_quad(C, n, C22, h, h, h);

    /* Roll back pool */
    pool.used = pool_save;
}

/* ================================================================
 * PURE BLIS (no Strassen) — for fair comparison
 * ================================================================ */
static void com6_pure_blis(const double* A, const double* B, double* C, int n) {
    double* BT = aligned_alloc_d((size_t)n * n);
    transpose(B, BT, n);
    com6_blis_multiply(A, BT, C, n);
    aligned_free_d(BT);
}

/* ================================================================
 * REFERENCE: Strassen with simple base case (same as v11/v12)
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

static void strassen_ref(const double* A, const double* B, double* C, int n);

static void strassen_ref(const double* A, const double* B, double* C, int n) {
    if (n <= 64) {
        naive_multiply(A, B, C, n);
        return;
    }
    int h = n / 2;
    size_t hsq = (size_t)h * h;

    size_t pool_save = pool.used;

    double* A11 = pool_alloc(hsq); double* A12 = pool_alloc(hsq);
    double* A21 = pool_alloc(hsq); double* A22 = pool_alloc(hsq);
    double* B11 = pool_alloc(hsq); double* B12 = pool_alloc(hsq);
    double* B21 = pool_alloc(hsq); double* B22 = pool_alloc(hsq);

    extract_quad(A, n, A11, h, 0, 0); extract_quad(A, n, A12, h, 0, h);
    extract_quad(A, n, A21, h, h, 0); extract_quad(A, n, A22, h, h, h);
    extract_quad(B, n, B11, h, 0, 0); extract_quad(B, n, B12, h, 0, h);
    extract_quad(B, n, B21, h, h, 0); extract_quad(B, n, B22, h, h, h);

    double* T1 = pool_alloc(hsq); double* T2 = pool_alloc(hsq);
    double* M1 = pool_alloc(hsq); double* M2 = pool_alloc(hsq);
    double* M3 = pool_alloc(hsq); double* M4 = pool_alloc(hsq);
    double* M5 = pool_alloc(hsq); double* M6 = pool_alloc(hsq);
    double* M7 = pool_alloc(hsq);

    mat_add(A11, A22, T1, h); mat_add(B11, B22, T2, h);
    strassen_ref(T1, T2, M1, h);
    mat_add(A21, A22, T1, h); strassen_ref(T1, B11, M2, h);
    mat_sub(B12, B22, T1, h); strassen_ref(A11, T1, M3, h);
    mat_sub(B21, B11, T1, h); strassen_ref(A22, T1, M4, h);
    mat_add(A11, A12, T1, h); strassen_ref(T1, B22, M5, h);
    mat_sub(A21, A11, T1, h); mat_add(B11, B12, T2, h);
    strassen_ref(T1, T2, M6, h);
    mat_sub(A12, A22, T1, h); mat_add(B21, B22, T2, h);
    strassen_ref(T1, T2, M7, h);

    double* C11 = T1; double* C12 = T2;
    double* C21 = pool_alloc(hsq); double* C22 = pool_alloc(hsq);

    for (size_t i = 0; i < hsq; i++) C11[i] = M1[i]+M4[i]-M5[i]+M7[i];
    mat_add(M3, M5, C12, h);
    mat_add(M2, M4, C21, h);
    for (size_t i = 0; i < hsq; i++) C22[i] = M1[i]-M2[i]+M3[i]+M6[i];

    insert_quad(C, n, C11, h, 0, 0); insert_quad(C, n, C12, h, 0, h);
    insert_quad(C, n, C21, h, h, 0); insert_quad(C, n, C22, h, h, h);

    pool.used = pool_save;
}

/* ================================================================
 * BENCHMARK HARNESS
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
    printf("  COM6 v13 - BLIS-Class: 6x8 micro-kernel + packing\n");
    printf("  Target: match OpenBLAS single-threaded performance\n");
    printf("====================================================================\n\n");

    int sizes[] = {256, 512, 1024, 2048, 4096};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-10s | %10s | %10s | %10s | %8s | %8s | %s\n",
           "Size", "Strassen", "COM6-BLIS", "COM6+Str", "GFLOPS", "v13/Str", "Verify");
    printf("---------- | ---------- | ---------- | ---------- | -------- | -------- | ------\n");

    /* Pool: enough for Strassen recursion on largest size */
    size_t pool_size = (size_t)4096 * 4096 * sizeof(double) * 20;
    pool_init(pool_size);

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si];
        size_t nn = (size_t)n * n;

        double* A   = aligned_alloc_d(nn);
        double* B   = aligned_alloc_d(nn);
        double* C1  = aligned_alloc_d(nn);
        double* C2  = aligned_alloc_d(nn);
        double* C3  = aligned_alloc_d(nn);

        srand(42);
        fill_random(A, n);
        fill_random(B, n);

        /* --- Reference Strassen --- */
        pool.used = 0;
        double t0 = get_time();
        strassen_ref(A, B, C1, n);
        double t_str = get_time() - t0;

        /* --- COM6 BLIS (pure, no Strassen) --- */
        t0 = get_time();
        com6_pure_blis(A, B, C2, n);
        double t_blis = get_time() - t0;

        /* --- COM6 BLIS + Strassen hybrid --- */
        pool.used = 0;
        t0 = get_time();
        strassen_blis(A, B, C3, n);
        double t_hybrid = get_time() - t0;

        /* Verify */
        double err1 = max_diff(C1, C2, n);
        double err2 = max_diff(C1, C3, n);
        double err = fmax(err1, err2);
        const char* verify = (err < 1e-6) ? "OK" : "FAIL";

        /* Best COM6 time */
        double t_best = (t_blis < t_hybrid) ? t_blis : t_hybrid;
        const char* best_label = (t_blis < t_hybrid) ? "BLIS" : "Hyb";

        double gflops = (2.0 * n * n * (double)n) / (t_best * 1e9);
        double ratio = t_str / t_best;

        printf("%4dx%-5d | %8.1f ms | %8.1f ms | %8.1f ms | %6.1f   | %6.2fx  | %s\n",
               n, n,
               t_str * 1000.0,
               t_blis * 1000.0,
               t_hybrid * 1000.0,
               gflops, ratio, verify);

        aligned_free_d(A);
        aligned_free_d(B);
        aligned_free_d(C1);
        aligned_free_d(C2);
        aligned_free_d(C3);
    }

    pool_destroy();

    printf("\nGFLOPS target: ~40 (OpenBLAS single-threaded on this CPU)\n");
    printf("v13/Str > 1.0 = COM6 WINS vs Strassen\n");

    return 0;
}
