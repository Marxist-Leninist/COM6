#define _GNU_SOURCE
/*
 * COM6 v118 vs OpenBLAS head-to-head on EPYC
 * Links against OpenBLAS cblas_dgemm and compares with COM6 v118.
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -lpthread \
 *       -I/tmp/OpenBLAS -L/tmp/OpenBLAS -lopenblas \
 *       -o bench_vs_openblas bench_vs_openblas_epyc.c com6_v118.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* OpenBLAS cblas interface */
enum CBLAS_ORDER { CblasRowMajor=101, CblasColMajor=102 };
enum CBLAS_TRANSPOSE { CblasNoTrans=111, CblasTrans=112 };
extern void cblas_dgemm(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE,
                        int M, int N, int K,
                        double alpha, const double *A, int lda,
                        const double *B, int ldb,
                        double beta, double *C, int ldc);
extern void openblas_set_num_threads(int);

/* COM6 v118 external declaration */
extern void com6_multiply(const double* A, const double* B, double* C, int n);
extern void com6_multiply_1t(const double* A, const double* B, double* C, int n);

static double now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int nsizes = 6;

    printf("COM6 v118 vs OpenBLAS on EPYC 7282\n");
    printf("%-12s | %10s %8s | %10s %8s | %6s\n",
           "Size", "COM6 (ms)", "GF", "BLAS (ms)", "GF", "Ratio");
    printf("-------------|----------------------|----------------------|------\n");

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si];
        size_t nn = (size_t)n * n;
        double *A = (double*)malloc(nn * sizeof(double));
        double *B = (double*)malloc(nn * sizeof(double));
        double *C1 = (double*)malloc(nn * sizeof(double));
        double *C2 = (double*)malloc(nn * sizeof(double));

        srand(42);
        for (size_t i = 0; i < nn; i++) {
            A[i] = (double)rand() / RAND_MAX * 2 - 1;
            B[i] = (double)rand() / RAND_MAX * 2 - 1;
        }

        int runs = (n <= 512) ? 7 : (n <= 1024) ? 5 : (n <= 2048) ? 3 : 2;
        double flops = 2.0 * n * n * (double)n;

        /* COM6 MT */
        com6_multiply(A, B, C1, n);
        double best_com6 = 1e30;
        for (int r = 0; r < runs; r++) {
            double t0 = now();
            com6_multiply(A, B, C1, n);
            double t = now() - t0;
            if (t < best_com6) best_com6 = t;
        }

        /* OpenBLAS MT */
        openblas_set_num_threads(16);
        memset(C2, 0, nn * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0, A, n, B, n, 0.0, C2, n);
        double best_blas = 1e30;
        for (int r = 0; r < runs; r++) {
            double t0 = now();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        n, n, n, 1.0, A, n, B, n, 0.0, C2, n);
            double t = now() - t0;
            if (t < best_blas) best_blas = t;
        }

        double gf_com6 = flops / (best_com6 * 1e9);
        double gf_blas = flops / (best_blas * 1e9);
        double ratio = gf_com6 / gf_blas;

        printf("%4dx%-7d | %8.1f ms %6.1f GF | %8.1f ms %6.1f GF | %5.2fx\n",
               n, n, best_com6*1000, gf_com6, best_blas*1000, gf_blas, ratio);

        free(A); free(B); free(C1); free(C2);

        if (si < nsizes - 1) {
            struct timespec ts = {2, 0};
            nanosleep(&ts, NULL);
        }
    }
    return 0;
}
