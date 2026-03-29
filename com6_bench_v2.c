/*
 * COM6 v2 - Optimized to crush Strassen at all sizes
 * ====================================================
 * Optimizations over v1:
 *   1. Register tiling: 4x4 micro-kernel keeps 16 accumulators in registers
 *   2. Software prefetch: hint next cache line before we need it
 *   3. 8-wide unrolling in inner loop
 *   4. Tuned block size for L1/L2 cache (64)
 *   5. Restrict pointers to prevent aliasing
 *   6. Multiple accumulator chains to hide FP latency
 *
 * Compile: gcc -O3 -march=native -funroll-loops -o com6v2 com6_bench_v2.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Prefetch intrinsic */
#ifdef _MSC_VER
#include <intrin.h>
#define PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
#define PREFETCH(addr) __builtin_prefetch((addr), 0, 3)
#endif

#define BLOCK_SIZE 64
#define MICRO_M 4
#define MICRO_N 4

/* ============================================================
 * Standard naive matmul (baseline)
 * ============================================================ */
void standard_matmul(const double * __restrict A, const double * __restrict B,
                     double * __restrict C, int n) {
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
 * Strassen
 * ============================================================ */
static void mat_add(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = A[i] + B[i];
}
static void mat_sub(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = A[i] - B[i];
}
static void get_sub(const double *src, double *dst, int n, int r, int c, int fn) {
    for (int i = 0; i < n; i++)
        memcpy(dst + i * n, src + (r + i) * fn + c, n * sizeof(double));
}
static void set_sub(double *dst, const double *src, int n, int r, int c, int fn) {
    for (int i = 0; i < n; i++)
        memcpy(dst + (r + i) * fn + c, src + i * n, n * sizeof(double));
}

/* Strassen base case uses ikj-ordered matmul (better than ijk but no transpose) */
static void base_matmul(const double *A, const double *B, double *C, int n) {
    memset(C, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[k * n + j];
        }
}

void strassen(const double *A, const double *B, double *C, int n) {
    if (n <= 64) { base_matmul(A, B, C, n); return; }
    int h = n / 2, sz = h * h;
    double *A11=malloc(sz*8), *A12=malloc(sz*8), *A21=malloc(sz*8), *A22=malloc(sz*8);
    double *B11=malloc(sz*8), *B12=malloc(sz*8), *B21=malloc(sz*8), *B22=malloc(sz*8);
    double *M1=malloc(sz*8),*M2=malloc(sz*8),*M3=malloc(sz*8),*M4=malloc(sz*8);
    double *M5=malloc(sz*8),*M6=malloc(sz*8),*M7=malloc(sz*8);
    double *T1=malloc(sz*8),*T2=malloc(sz*8);

    get_sub(A,A11,h,0,0,n); get_sub(A,A12,h,0,h,n);
    get_sub(A,A21,h,h,0,n); get_sub(A,A22,h,h,h,n);
    get_sub(B,B11,h,0,0,n); get_sub(B,B12,h,0,h,n);
    get_sub(B,B21,h,h,0,n); get_sub(B,B22,h,h,h,n);

    mat_add(A11,A22,T1,h); mat_add(B11,B22,T2,h); strassen(T1,T2,M1,h);
    mat_add(A21,A22,T1,h); strassen(T1,B11,M2,h);
    mat_sub(B12,B22,T1,h); strassen(A11,T1,M3,h);
    mat_sub(B21,B11,T1,h); strassen(A22,T1,M4,h);
    mat_add(A11,A12,T1,h); strassen(T1,B22,M5,h);
    mat_sub(A21,A11,T1,h); mat_add(B11,B12,T2,h); strassen(T1,T2,M6,h);
    mat_sub(A12,A22,T1,h); mat_add(B21,B22,T2,h); strassen(T1,T2,M7,h);

    mat_add(M1,M4,T1,h); mat_sub(T1,M5,T2,h); mat_add(T2,M7,T1,h); set_sub(C,T1,h,0,0,n);
    mat_add(M3,M5,T1,h); set_sub(C,T1,h,0,h,n);
    mat_add(M2,M4,T1,h); set_sub(C,T1,h,h,0,n);
    mat_sub(M1,M2,T1,h); mat_add(T1,M3,T2,h); mat_add(T2,M6,T1,h); set_sub(C,T1,h,h,h,n);

    free(A11);free(A12);free(A21);free(A22);
    free(B11);free(B12);free(B21);free(B22);
    free(M1);free(M2);free(M3);free(M4);free(M5);free(M6);free(M7);
    free(T1);free(T2);
}

/* ============================================================
 * COM6 v2 - The Beast
 * 4x4 register-tiled micro-kernel with prefetch + 8-wide unroll
 * ============================================================ */

/* Micro-kernel: compute a 4x4 tile of C using A rows and BT rows */
static inline void micro_4x4(const double * __restrict Ai0, const double * __restrict Ai1,
                              const double * __restrict Ai2, const double * __restrict Ai3,
                              const double * __restrict BTj0, const double * __restrict BTj1,
                              const double * __restrict BTj2, const double * __restrict BTj3,
                              double *c, int cn, int klen) {
    /* 16 accumulators in registers */
    double c00=0,c01=0,c02=0,c03=0;
    double c10=0,c11=0,c12=0,c13=0;
    double c20=0,c21=0,c22=0,c23=0;
    double c30=0,c31=0,c32=0,c33=0;

    int k = 0;
    /* 8-wide unrolled inner loop */
    for (; k + 7 < klen; k += 8) {
        /* Prefetch next cache line (64 bytes = 8 doubles ahead) */
        PREFETCH(Ai0 + k + 8);
        PREFETCH(BTj0 + k + 8);

        double a0_0=Ai0[k], a0_1=Ai0[k+1], a0_2=Ai0[k+2], a0_3=Ai0[k+3];
        double a0_4=Ai0[k+4], a0_5=Ai0[k+5], a0_6=Ai0[k+6], a0_7=Ai0[k+7];
        double b0_0=BTj0[k], b0_1=BTj0[k+1], b0_2=BTj0[k+2], b0_3=BTj0[k+3];
        double b0_4=BTj0[k+4], b0_5=BTj0[k+5], b0_6=BTj0[k+6], b0_7=BTj0[k+7];

        c00 += a0_0*b0_0 + a0_1*b0_1 + a0_2*b0_2 + a0_3*b0_3
             + a0_4*b0_4 + a0_5*b0_5 + a0_6*b0_6 + a0_7*b0_7;

        double b1_0=BTj1[k], b1_1=BTj1[k+1], b1_2=BTj1[k+2], b1_3=BTj1[k+3];
        double b1_4=BTj1[k+4], b1_5=BTj1[k+5], b1_6=BTj1[k+6], b1_7=BTj1[k+7];

        c01 += a0_0*b1_0 + a0_1*b1_1 + a0_2*b1_2 + a0_3*b1_3
             + a0_4*b1_4 + a0_5*b1_5 + a0_6*b1_6 + a0_7*b1_7;

        double b2_0=BTj2[k], b2_1=BTj2[k+1], b2_2=BTj2[k+2], b2_3=BTj2[k+3];
        double b2_4=BTj2[k+4], b2_5=BTj2[k+5], b2_6=BTj2[k+6], b2_7=BTj2[k+7];

        c02 += a0_0*b2_0 + a0_1*b2_1 + a0_2*b2_2 + a0_3*b2_3
             + a0_4*b2_4 + a0_5*b2_5 + a0_6*b2_6 + a0_7*b2_7;

        double b3_0=BTj3[k], b3_1=BTj3[k+1], b3_2=BTj3[k+2], b3_3=BTj3[k+3];
        double b3_4=BTj3[k+4], b3_5=BTj3[k+5], b3_6=BTj3[k+6], b3_7=BTj3[k+7];

        c03 += a0_0*b3_0 + a0_1*b3_1 + a0_2*b3_2 + a0_3*b3_3
             + a0_4*b3_4 + a0_5*b3_5 + a0_6*b3_6 + a0_7*b3_7;

        /* Row 1 */
        PREFETCH(Ai1 + k + 8);
        double a1_0=Ai1[k], a1_1=Ai1[k+1], a1_2=Ai1[k+2], a1_3=Ai1[k+3];
        double a1_4=Ai1[k+4], a1_5=Ai1[k+5], a1_6=Ai1[k+6], a1_7=Ai1[k+7];
        c10 += a1_0*b0_0 + a1_1*b0_1 + a1_2*b0_2 + a1_3*b0_3
             + a1_4*b0_4 + a1_5*b0_5 + a1_6*b0_6 + a1_7*b0_7;
        c11 += a1_0*b1_0 + a1_1*b1_1 + a1_2*b1_2 + a1_3*b1_3
             + a1_4*b1_4 + a1_5*b1_5 + a1_6*b1_6 + a1_7*b1_7;
        c12 += a1_0*b2_0 + a1_1*b2_1 + a1_2*b2_2 + a1_3*b2_3
             + a1_4*b2_4 + a1_5*b2_5 + a1_6*b2_6 + a1_7*b2_7;
        c13 += a1_0*b3_0 + a1_1*b3_1 + a1_2*b3_2 + a1_3*b3_3
             + a1_4*b3_4 + a1_5*b3_5 + a1_6*b3_6 + a1_7*b3_7;

        /* Row 2 */
        PREFETCH(Ai2 + k + 8);
        double a2_0=Ai2[k], a2_1=Ai2[k+1], a2_2=Ai2[k+2], a2_3=Ai2[k+3];
        double a2_4=Ai2[k+4], a2_5=Ai2[k+5], a2_6=Ai2[k+6], a2_7=Ai2[k+7];
        c20 += a2_0*b0_0 + a2_1*b0_1 + a2_2*b0_2 + a2_3*b0_3
             + a2_4*b0_4 + a2_5*b0_5 + a2_6*b0_6 + a2_7*b0_7;
        c21 += a2_0*b1_0 + a2_1*b1_1 + a2_2*b1_2 + a2_3*b1_3
             + a2_4*b1_4 + a2_5*b1_5 + a2_6*b1_6 + a2_7*b1_7;
        c22 += a2_0*b2_0 + a2_1*b2_1 + a2_2*b2_2 + a2_3*b2_3
             + a2_4*b2_4 + a2_5*b2_5 + a2_6*b2_6 + a2_7*b2_7;
        c23 += a2_0*b3_0 + a2_1*b3_1 + a2_2*b3_2 + a2_3*b3_3
             + a2_4*b3_4 + a2_5*b3_5 + a2_6*b3_6 + a2_7*b3_7;

        /* Row 3 */
        PREFETCH(Ai3 + k + 8);
        double a3_0=Ai3[k], a3_1=Ai3[k+1], a3_2=Ai3[k+2], a3_3=Ai3[k+3];
        double a3_4=Ai3[k+4], a3_5=Ai3[k+5], a3_6=Ai3[k+6], a3_7=Ai3[k+7];
        c30 += a3_0*b0_0 + a3_1*b0_1 + a3_2*b0_2 + a3_3*b0_3
             + a3_4*b0_4 + a3_5*b0_5 + a3_6*b0_6 + a3_7*b0_7;
        c31 += a3_0*b1_0 + a3_1*b1_1 + a3_2*b1_2 + a3_3*b1_3
             + a3_4*b1_4 + a3_5*b1_5 + a3_6*b1_6 + a3_7*b1_7;
        c32 += a3_0*b2_0 + a3_1*b2_1 + a3_2*b2_2 + a3_3*b2_3
             + a3_4*b2_4 + a3_5*b2_5 + a3_6*b2_6 + a3_7*b2_7;
        c33 += a3_0*b3_0 + a3_1*b3_1 + a3_2*b3_2 + a3_3*b3_3
             + a3_4*b3_4 + a3_5*b3_5 + a3_6*b3_6 + a3_7*b3_7;
    }
    /* Cleanup */
    for (; k < klen; k++) {
        double a0=Ai0[k], a1=Ai1[k], a2=Ai2[k], a3=Ai3[k];
        double b0=BTj0[k], b1=BTj1[k], b2=BTj2[k], b3=BTj3[k];
        c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3;
        c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3;
        c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3;
        c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3;
    }

    c[0*cn+0]+=c00; c[0*cn+1]+=c01; c[0*cn+2]+=c02; c[0*cn+3]+=c03;
    c[1*cn+0]+=c10; c[1*cn+1]+=c11; c[1*cn+2]+=c12; c[1*cn+3]+=c13;
    c[2*cn+0]+=c20; c[2*cn+1]+=c21; c[2*cn+2]+=c22; c[2*cn+3]+=c23;
    c[3*cn+0]+=c30; c[3*cn+1]+=c31; c[3*cn+2]+=c32; c[3*cn+3]+=c33;
}


void com6_v2(const double * __restrict A, const double * __restrict B,
             double * __restrict C, int n) {
    memset(C, 0, n * n * sizeof(double));

    /* Contiguous transpose */
    double *BT = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[j * n + i] = B[i * n + j];

    int BS = BLOCK_SIZE;

    for (int i0 = 0; i0 < n; i0 += BS) {
        int i1 = i0 + BS < n ? i0 + BS : n;
        for (int j0 = 0; j0 < n; j0 += BS) {
            int j1 = j0 + BS < n ? j0 + BS : n;
            for (int k0 = 0; k0 < n; k0 += BS) {
                int k1 = k0 + BS < n ? k0 + BS : n;
                int klen = k1 - k0;

                /* Dispatch 4x4 micro-kernels within block */
                int i;
                for (i = i0; i + 3 < i1; i += 4) {
                    const double *Ai0 = A + i * n + k0;
                    const double *Ai1 = A + (i+1) * n + k0;
                    const double *Ai2 = A + (i+2) * n + k0;
                    const double *Ai3 = A + (i+3) * n + k0;

                    int j;
                    for (j = j0; j + 3 < j1; j += 4) {
                        const double *BTj0 = BT + j * n + k0;
                        const double *BTj1 = BT + (j+1) * n + k0;
                        const double *BTj2 = BT + (j+2) * n + k0;
                        const double *BTj3 = BT + (j+3) * n + k0;

                        micro_4x4(Ai0, Ai1, Ai2, Ai3,
                                  BTj0, BTj1, BTj2, BTj3,
                                  C + i * n + j, n, klen);
                    }
                    /* Remainder columns */
                    for (; j < j1; j++) {
                        const double *BTj = BT + j * n + k0;
                        double s0=0,s1=0,s2=0,s3=0;
                        for (int k = 0; k < klen; k++) {
                            double b = BTj[k];
                            s0 += Ai0[k]*b; s1 += Ai1[k]*b;
                            s2 += Ai2[k]*b; s3 += Ai3[k]*b;
                        }
                        C[i*n+j]+=s0; C[(i+1)*n+j]+=s1;
                        C[(i+2)*n+j]+=s2; C[(i+3)*n+j]+=s3;
                    }
                }
                /* Remainder rows */
                for (; i < i1; i++) {
                    const double *Ai = A + i * n + k0;
                    for (int j = j0; j < j1; j++) {
                        const double *BTj = BT + j * n + k0;
                        double s = 0;
                        for (int k = 0; k < klen; k++)
                            s += Ai[k] * BTj[k];
                        C[i*n+j] += s;
                    }
                }
            }
        }
    }
    free(BT);
}

/* ============================================================
 * COM6 v1 (from previous benchmark, for comparison)
 * ============================================================ */
void com6_v1_unrolled(const double *A, const double *B, double *C, int n) {
    memset(C, 0, n * n * sizeof(double));
    double *BT = malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[j * n + i] = B[i * n + j];

    int BS = 32;
    for (int i0 = 0; i0 < n; i0 += BS) {
        int i1 = i0+BS < n ? i0+BS : n;
        for (int j0 = 0; j0 < n; j0 += BS) {
            int j1 = j0+BS < n ? j0+BS : n;
            for (int k0 = 0; k0 < n; k0 += BS) {
                int k1 = k0+BS < n ? k0+BS : n;
                int klen = k1 - k0;
                for (int i = i0; i < i1; i++) {
                    double *Ai = (double*)(A + i*n + k0);
                    for (int j = j0; j < j1; j++) {
                        double *BTj = (double*)(BT + j*n + k0);
                        double sum = 0.0;
                        int k = 0;
                        for (; k+3 < klen; k += 4)
                            sum += Ai[k]*BTj[k] + Ai[k+1]*BTj[k+1]
                                 + Ai[k+2]*BTj[k+2] + Ai[k+3]*BTj[k+3];
                        for (; k < klen; k++)
                            sum += Ai[k]*BTj[k];
                        C[i*n+j] += sum;
                    }
                }
            }
        }
    }
    free(BT);
}


/* ============================================================
 * Timing
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
    printf("  COM6 v2 Benchmark - Register-Tiled 4x4 Micro-Kernel\n");
    printf("  Block: %d, Micro: %dx%d, 8-wide unroll, prefetch\n", BLOCK_SIZE, MICRO_M, MICRO_N);
    printf("========================================================================\n\n");

    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-9s | %10s | %10s | %10s | %10s | %7s | %7s\n",
           "Size", "Standard", "Strassen", "COM6-v1", "COM6-v2", "v2/Std", "v2/Str");
    printf("-------------------------------------------------------------------------\n");

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si];
        double *A = malloc(n*n*sizeof(double));
        double *B = malloc(n*n*sizeof(double));
        double *C_std = malloc(n*n*sizeof(double));
        double *C_str = malloc(n*n*sizeof(double));
        double *C_v1  = malloc(n*n*sizeof(double));
        double *C_v2  = malloc(n*n*sizeof(double));

        srand(42);
        fill_random(A, n);
        fill_random(B, n);

        int runs = n <= 512 ? 3 : 1;
        double t_std = 0, t_str = 0, t_v1 = 0, t_v2 = 0;

        /* Standard (skip for 4096 - too slow) */
        if (n <= 2048) {
            double t0 = get_time_ms();
            for (int r = 0; r < runs; r++) standard_matmul(A, B, C_std, n);
            t_std = (get_time_ms() - t0) / runs;
        }

        /* Strassen (power of 2 only) */
        if ((n & (n-1)) == 0) {
            double t0 = get_time_ms();
            for (int r = 0; r < runs; r++) strassen(A, B, C_str, n);
            t_str = (get_time_ms() - t0) / runs;
        }

        /* COM6 v1 */
        {
            double t0 = get_time_ms();
            for (int r = 0; r < runs; r++) com6_v1_unrolled(A, B, C_v1, n);
            t_v1 = (get_time_ms() - t0) / runs;
        }

        /* COM6 v2 */
        {
            double t0 = get_time_ms();
            for (int r = 0; r < runs; r++) com6_v2(A, B, C_v2, n);
            t_v2 = (get_time_ms() - t0) / runs;
        }

        /* Verify against v1 (which we know is correct) */
        double diff = max_diff(C_v1, C_v2, n);
        char verify[16];
        if (diff < 1e-6) snprintf(verify, sizeof(verify), "OK");
        else snprintf(verify, sizeof(verify), "ERR %.1e", diff);

        double sp_std = t_std > 0 ? t_std / t_v2 : 0;
        double sp_str = t_str > 0 ? t_str / t_v2 : 0;

        char s_std[20], s_str[20], s_v1[20], s_v2[20];
        if (t_std > 0) snprintf(s_std, 20, "%8.1f ms", t_std); else snprintf(s_std, 20, "%10s", "SKIP");
        if (t_str > 0) snprintf(s_str, 20, "%8.1f ms", t_str); else snprintf(s_str, 20, "%10s", "N/A");
        snprintf(s_v1, 20, "%8.1f ms", t_v1);
        snprintf(s_v2, 20, "%8.1f ms", t_v2);

        printf("%4dx%-4d | %10s | %10s | %10s | %10s |",
               n, n, s_std, s_str, s_v1, s_v2);
        if (t_std > 0) printf(" %6.1fx |", sp_std); else printf(" %7s |", "-");
        if (t_str > 0) printf(" %6.1fx", sp_str); else printf(" %7s", "-");
        printf("  %s\n", verify);

        free(A); free(B); free(C_std); free(C_str); free(C_v1); free(C_v2);
    }

    printf("\nv2/Std = COM6v2 speedup over standard\n");
    printf("v2/Str = COM6v2 speedup over Strassen\n");
    printf("> 1.0x means COM6 wins\n");
    return 0;
}
