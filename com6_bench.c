/*
 * COM6 vs Strassen vs Standard Matrix Multiplication - C Benchmark
 * ================================================================
 * This is where COM6's cache optimization actually shows its teeth.
 * No BLAS, no cheating - raw C loops, same playing field.
 *
 * Compile: gcc -O2 -o com6_bench com6_bench.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 32

/* ============================================================
 * Standard naive matmul - O(n^3) triple loop
 * ============================================================ */
void standard_matmul(double *A, double *B, double *C, int n) {
    memset(C, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* ============================================================
 * Strassen's algorithm - O(n^2.807)
 * ============================================================ */
void matrix_add(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = A[i] + B[i];
}

void matrix_sub(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = A[i] - B[i];
}

/* Extract submatrix from n*2 x n*2 matrix */
void get_submatrix(double *src, double *dst, int n, int row, int col, int full_n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dst[i * n + j] = src[(row + i) * full_n + (col + j)];
}

void set_submatrix(double *dst, double *src, int n, int row, int col, int full_n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dst[(row + i) * full_n + (col + j)] = src[i * n + j];
}

void strassen(double *A, double *B, double *C, int n) {
    if (n <= 64) {
        /* Base case: standard matmul */
        standard_matmul(A, B, C, n);
        return;
    }

    int half = n / 2;
    int sz = half * half;

    double *A11 = malloc(sz * sizeof(double));
    double *A12 = malloc(sz * sizeof(double));
    double *A21 = malloc(sz * sizeof(double));
    double *A22 = malloc(sz * sizeof(double));
    double *B11 = malloc(sz * sizeof(double));
    double *B12 = malloc(sz * sizeof(double));
    double *B21 = malloc(sz * sizeof(double));
    double *B22 = malloc(sz * sizeof(double));
    double *M1 = malloc(sz * sizeof(double));
    double *M2 = malloc(sz * sizeof(double));
    double *M3 = malloc(sz * sizeof(double));
    double *M4 = malloc(sz * sizeof(double));
    double *M5 = malloc(sz * sizeof(double));
    double *M6 = malloc(sz * sizeof(double));
    double *M7 = malloc(sz * sizeof(double));
    double *T1 = malloc(sz * sizeof(double));
    double *T2 = malloc(sz * sizeof(double));

    get_submatrix(A, A11, half, 0, 0, n);
    get_submatrix(A, A12, half, 0, half, n);
    get_submatrix(A, A21, half, half, 0, n);
    get_submatrix(A, A22, half, half, half, n);
    get_submatrix(B, B11, half, 0, 0, n);
    get_submatrix(B, B12, half, 0, half, n);
    get_submatrix(B, B21, half, half, 0, n);
    get_submatrix(B, B22, half, half, half, n);

    /* M1 = (A11+A22)(B11+B22) */
    matrix_add(A11, A22, T1, half);
    matrix_add(B11, B22, T2, half);
    strassen(T1, T2, M1, half);

    /* M2 = (A21+A22)*B11 */
    matrix_add(A21, A22, T1, half);
    strassen(T1, B11, M2, half);

    /* M3 = A11*(B12-B22) */
    matrix_sub(B12, B22, T1, half);
    strassen(A11, T1, M3, half);

    /* M4 = A22*(B21-B11) */
    matrix_sub(B21, B11, T1, half);
    strassen(A22, T1, M4, half);

    /* M5 = (A11+A12)*B22 */
    matrix_add(A11, A12, T1, half);
    strassen(T1, B22, M5, half);

    /* M6 = (A21-A11)*(B11+B12) */
    matrix_sub(A21, A11, T1, half);
    matrix_add(B11, B12, T2, half);
    strassen(T1, T2, M6, half);

    /* M7 = (A12-A22)*(B21+B22) */
    matrix_sub(A12, A22, T1, half);
    matrix_add(B21, B22, T2, half);
    strassen(T1, T2, M7, half);

    /* C11 = M1+M4-M5+M7 */
    matrix_add(M1, M4, T1, half);
    matrix_sub(T1, M5, T2, half);
    matrix_add(T2, M7, T1, half);
    set_submatrix(C, T1, half, 0, 0, n);

    /* C12 = M3+M5 */
    matrix_add(M3, M5, T1, half);
    set_submatrix(C, T1, half, 0, half, n);

    /* C21 = M2+M4 */
    matrix_add(M2, M4, T1, half);
    set_submatrix(C, T1, half, half, 0, n);

    /* C22 = M1-M2+M3+M6 */
    matrix_sub(M1, M2, T1, half);
    matrix_add(T1, M3, T2, half);
    matrix_add(T2, M6, T1, half);
    set_submatrix(C, T1, half, half, half, n);

    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(M1); free(M2); free(M3); free(M4);
    free(M5); free(M6); free(M7);
    free(T1); free(T2);
}

/* ============================================================
 * COM6 Matrix Multiplication
 * Block-tiled with B transposition for cache-line optimization.
 * ============================================================ */
void com6_matmul(double *A, double *B, double *C, int n) {
    memset(C, 0, n * n * sizeof(double));

    /* COM6 core: transpose B for sequential memory access */
    double *BT = malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[j * n + i] = B[i * n + j];

    /* Block-tiled multiplication */
    for (int i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
        int i1 = i0 + BLOCK_SIZE < n ? i0 + BLOCK_SIZE : n;
        for (int j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            int j1 = j0 + BLOCK_SIZE < n ? j0 + BLOCK_SIZE : n;
            for (int k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
                int k1 = k0 + BLOCK_SIZE < n ? k0 + BLOCK_SIZE : n;
                /* Inner block: both A row and BT row are sequential in memory */
                for (int i = i0; i < i1; i++) {
                    for (int j = j0; j < j1; j++) {
                        double sum = 0.0;
                        /* KEY: A[i*n+k] and BT[j*n+k] are both sequential!
                         * Standard matmul accesses B[k*n+j] which jumps by n each k.
                         * COM6 accesses BT[j*n+k] which is stride-1. Cache line hit. */
                        for (int k = k0; k < k1; k++) {
                            sum += A[i * n + k] * BT[j * n + k];
                        }
                        C[i * n + j] += sum;
                    }
                }
            }
        }
    }
    free(BT);
}

/* ============================================================
 * COM6 with loop unrolling (aggressive optimization)
 * ============================================================ */
void com6_matmul_unrolled(double *A, double *B, double *C, int n) {
    memset(C, 0, n * n * sizeof(double));

    double *BT = malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[j * n + i] = B[i * n + j];

    for (int i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
        int i1 = i0 + BLOCK_SIZE < n ? i0 + BLOCK_SIZE : n;
        for (int j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            int j1 = j0 + BLOCK_SIZE < n ? j0 + BLOCK_SIZE : n;
            for (int k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
                int k1 = k0 + BLOCK_SIZE < n ? k0 + BLOCK_SIZE : n;
                int klen = k1 - k0;
                for (int i = i0; i < i1; i++) {
                    double *Ai = A + i * n + k0;
                    for (int j = j0; j < j1; j++) {
                        double *BTj = BT + j * n + k0;
                        double sum = 0.0;
                        int k = 0;
                        /* Unroll by 4 */
                        for (; k + 3 < klen; k += 4) {
                            sum += Ai[k] * BTj[k]
                                 + Ai[k+1] * BTj[k+1]
                                 + Ai[k+2] * BTj[k+2]
                                 + Ai[k+3] * BTj[k+3];
                        }
                        for (; k < klen; k++) {
                            sum += Ai[k] * BTj[k];
                        }
                        C[i * n + j] += sum;
                    }
                }
            }
        }
    }
    free(BT);
}

/* ============================================================
 * Timing and verification
 * ============================================================ */
double get_time_ms() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void fill_random(double *M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

double max_diff(double *A, double *B, int n) {
    double mx = 0.0;
    for (int i = 0; i < n * n; i++) {
        double d = fabs(A[i] - B[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

int main() {
    printf("========================================================================\n");
    printf("  COM6 vs Strassen vs Standard - C Benchmark (same playing field)\n");
    printf("  Block size: %d\n", BLOCK_SIZE);
    printf("========================================================================\n\n");

    int sizes[] = {128, 256, 512, 1024, 2048};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-8s | %12s | %12s | %12s | %12s | %9s | %9s\n",
           "Size", "Standard", "Strassen", "COM6", "COM6-Unroll", "C6/Std", "C6u/Std");
    printf("------------------------------------------------------------------------"
           "-------------\n");

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si];
        double *A = malloc(n * n * sizeof(double));
        double *B = malloc(n * n * sizeof(double));
        double *C_std = malloc(n * n * sizeof(double));
        double *C_str = malloc(n * n * sizeof(double));
        double *C_com = malloc(n * n * sizeof(double));
        double *C_unr = malloc(n * n * sizeof(double));

        srand(42);
        fill_random(A, n);
        fill_random(B, n);

        int runs = n <= 512 ? 3 : 1;

        /* Standard */
        double t0 = get_time_ms();
        for (int r = 0; r < runs; r++) standard_matmul(A, B, C_std, n);
        double t_std = (get_time_ms() - t0) / runs;

        /* Strassen (only for power-of-2 sizes) */
        double t_str = 0;
        if ((n & (n - 1)) == 0) {  /* power of 2 */
            t0 = get_time_ms();
            for (int r = 0; r < runs; r++) strassen(A, B, C_str, n);
            t_str = (get_time_ms() - t0) / runs;
        }

        /* COM6 */
        t0 = get_time_ms();
        for (int r = 0; r < runs; r++) com6_matmul(A, B, C_com, n);
        double t_com = (get_time_ms() - t0) / runs;

        /* COM6 unrolled */
        t0 = get_time_ms();
        for (int r = 0; r < runs; r++) com6_matmul_unrolled(A, B, C_unr, n);
        double t_unr = (get_time_ms() - t0) / runs;

        /* Verify */
        double d_com = max_diff(C_std, C_com, n);
        double d_unr = max_diff(C_std, C_unr, n);

        char str_buf[32];
        if ((n & (n - 1)) == 0) {
            double d_str = max_diff(C_std, C_str, n);
            snprintf(str_buf, sizeof(str_buf), "%9.2f ms", t_str);
        } else {
            snprintf(str_buf, sizeof(str_buf), "%12s", "N/A");
        }

        printf("%4dx%-4d| %9.2f ms | %s | %9.2f ms | %9.2f ms | %8.2fx | %8.2fx",
               n, n, t_std, str_buf, t_com, t_unr,
               t_std / t_com, t_std / t_unr);

        if (d_com > 1e-6 || d_unr > 1e-6)
            printf(" MISMATCH!");
        printf("\n");

        free(A); free(B); free(C_std); free(C_str); free(C_com); free(C_unr);
    }

    printf("\nNote: C6/Std = COM6 speedup over standard, C6u/Std = COM6-unrolled speedup\n");
    printf("Values > 1.0x mean COM6 is faster than standard triple-loop.\n");

    return 0;
}
