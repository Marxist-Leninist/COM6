/*
 * COM6 v7 - AVX2 Intrinsics
 * ==========================
 * Hand-written SIMD inner kernel using _mm256 FMA instructions.
 * Pre-transposed B means both operands are contiguous => aligned loads.
 *
 * The key AVX2 advantage for COM6:
 *   - Load 4 doubles from A row (contiguous)
 *   - Load 4 doubles from BT row (contiguous)
 *   - FMA: acc = a * b + acc  (one instruction, 4 doubles)
 *   - Both loads are stride-1 aligned => no gathers, no cache misses
 *
 * gcc -O3 -march=native -mavx2 -mfma -o com6_avx2 com6_avx2.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

/* Horizontal sum of __m256d (4 doubles -> 1 double) */
static inline double hsum_avx(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);       /* lo[0]+hi[0], lo[1]+hi[1] */
    hi = _mm_unpackhi_pd(lo, lo);  /* lo[1], lo[1] */
    lo = _mm_add_sd(lo, hi);       /* lo[0]+lo[1] */
    return _mm_cvtsd_f64(lo);
}

/* ============================================================
 * COM6 AVX2 base case
 * Process 1 row of A against 4 rows of BT simultaneously
 * Each dot product uses FMA on contiguous data
 * ============================================================ */
static void com6_avx2_base(const double * __restrict A, const double * __restrict BT,
                           double * __restrict C, int n) {
    for (int i = 0; i < n; i++) {
        const double *Ai = A + i * n;
        int j = 0;

        /* Process 4 output columns at a time */
        for (; j + 3 < n; j += 4) {
            const double *B0 = BT + j * n;
            const double *B1 = BT + (j+1) * n;
            const double *B2 = BT + (j+2) * n;
            const double *B3 = BT + (j+3) * n;

            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            __m256d acc2 = _mm256_setzero_pd();
            __m256d acc3 = _mm256_setzero_pd();

            int k = 0;
            for (; k + 3 < n; k += 4) {
                __m256d a = _mm256_loadu_pd(Ai + k);
                acc0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(B0 + k), acc0);
                acc1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(B1 + k), acc1);
                acc2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(B2 + k), acc2);
                acc3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(B3 + k), acc3);
            }

            double s0 = hsum_avx(acc0);
            double s1 = hsum_avx(acc1);
            double s2 = hsum_avx(acc2);
            double s3 = hsum_avx(acc3);

            /* Scalar cleanup */
            for (; k < n; k++) {
                double a = Ai[k];
                s0 += a * B0[k]; s1 += a * B1[k];
                s2 += a * B2[k]; s3 += a * B3[k];
            }
            C[i*n+j] = s0; C[i*n+j+1] = s1;
            C[i*n+j+2] = s2; C[i*n+j+3] = s3;
        }
        /* Remainder columns */
        for (; j < n; j++) {
            const double *Bj = BT + j * n;
            __m256d acc = _mm256_setzero_pd();
            int k = 0;
            for (; k + 3 < n; k += 4)
                acc = _mm256_fmadd_pd(_mm256_loadu_pd(Ai+k), _mm256_loadu_pd(Bj+k), acc);
            double s = hsum_avx(acc);
            for (; k < n; k++) s += Ai[k] * Bj[k];
            C[i*n+j] = s;
        }
    }
}

/* ============================================================
 * ikj AVX2 base case (Strassen's pattern with SIMD)
 * Broadcast a, FMA across j dimension
 * ============================================================ */
static void ikj_avx2_base(const double * __restrict A, const double * __restrict B,
                           double * __restrict C, int n) {
    memset(C, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double *Ci = C + i * n;
        for (int k = 0; k < n; k++) {
            __m256d a_broadcast = _mm256_set1_pd(A[i*n+k]);
            const double *Bk = B + k * n;
            int j = 0;
            for (; j + 3 < n; j += 4) {
                __m256d c = _mm256_loadu_pd(Ci + j);
                __m256d b = _mm256_loadu_pd(Bk + j);
                _mm256_storeu_pd(Ci + j, _mm256_fmadd_pd(a_broadcast, b, c));
            }
            for (; j < n; j++) Ci[j] += A[i*n+k] * Bk[j];
        }
    }
}

/* ============================================================
 * Strassen plumbing
 * ============================================================ */
static void add_f(const double *A, const double *B, double *C, int sz) {
    int i = 0;
    for (; i + 3 < sz; i += 4) {
        __m256d a = _mm256_loadu_pd(A+i);
        __m256d b = _mm256_loadu_pd(B+i);
        _mm256_storeu_pd(C+i, _mm256_add_pd(a, b));
    }
    for (; i < sz; i++) C[i] = A[i] + B[i];
}
static void sub_f(const double *A, const double *B, double *C, int sz) {
    int i = 0;
    for (; i + 3 < sz; i += 4) {
        __m256d a = _mm256_loadu_pd(A+i);
        __m256d b = _mm256_loadu_pd(B+i);
        _mm256_storeu_pd(C+i, _mm256_sub_pd(a, b));
    }
    for (; i < sz; i++) C[i] = A[i] - B[i];
}
static void get_q(const double *s, double *d, int h, int r, int c, int n) {
    for (int i=0;i<h;i++) memcpy(d+i*h, s+(r+i)*n+c, h*8);
}
static void set_q(double *d, const double *s, int h, int r, int c, int n) {
    for (int i=0;i<h;i++) memcpy(d+(r+i)*n+c, s+i*h, h*8);
}

/* COM6-Strassen with AVX2 base */
static void com6_str_avx(const double *A, const double *BT, double *C, int n, int thresh) {
    if (n <= thresh) { com6_avx2_base(A, BT, C, n); return; }
    int h=n/2, sz=h*h;
    double *A11=malloc(sz*8),*A12=malloc(sz*8),*A21=malloc(sz*8),*A22=malloc(sz*8);
    double *BT11=malloc(sz*8),*BT12=malloc(sz*8),*BT21=malloc(sz*8),*BT22=malloc(sz*8);
    double *M1=malloc(sz*8),*M2=malloc(sz*8),*M3=malloc(sz*8),*M4=malloc(sz*8);
    double *M5=malloc(sz*8),*M6=malloc(sz*8),*M7=malloc(sz*8);
    double *TA=malloc(sz*8),*TB=malloc(sz*8);

    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(BT,BT11,h,0,0,n);get_q(BT,BT12,h,0,h,n);get_q(BT,BT21,h,h,0,n);get_q(BT,BT22,h,h,h,n);

    add_f(A11,A22,TA,sz);add_f(BT11,BT22,TB,sz);com6_str_avx(TA,TB,M1,h,thresh);
    add_f(A21,A22,TA,sz);com6_str_avx(TA,BT11,M2,h,thresh);
    sub_f(BT21,BT22,TB,sz);com6_str_avx(A11,TB,M3,h,thresh);
    sub_f(BT12,BT11,TB,sz);com6_str_avx(A22,TB,M4,h,thresh);
    add_f(A11,A12,TA,sz);com6_str_avx(TA,BT22,M5,h,thresh);
    sub_f(A21,A11,TA,sz);add_f(BT11,BT21,TB,sz);com6_str_avx(TA,TB,M6,h,thresh);
    sub_f(A12,A22,TA,sz);add_f(BT12,BT22,TB,sz);com6_str_avx(TA,TB,M7,h,thresh);

    add_f(M1,M4,TA,sz);sub_f(TA,M5,TB,sz);add_f(TB,M7,TA,sz);set_q(C,TA,h,0,0,n);
    add_f(M3,M5,TA,sz);set_q(C,TA,h,0,h,n);
    add_f(M2,M4,TA,sz);set_q(C,TA,h,h,0,n);
    sub_f(M1,M2,TA,sz);add_f(TA,M3,TB,sz);add_f(TB,M6,TA,sz);set_q(C,TA,h,h,h,n);

    free(A11);free(A12);free(A21);free(A22);
    free(BT11);free(BT12);free(BT21);free(BT22);
    free(M1);free(M2);free(M3);free(M4);free(M5);free(M6);free(M7);
    free(TA);free(TB);
}

void com6_avx(const double *A, const double *B, double *C, int n, int thresh) {
    double *BT = malloc(n*n*8);
    /* AVX2 transpose */
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) BT[j*n+i]=B[i*n+j];
    com6_str_avx(A, BT, C, n, thresh);
    free(BT);
}

/* Pure Strassen with AVX2 ikj base */
void pure_str_avx(const double *A, const double *B, double *C, int n, int thresh) {
    if (n<=thresh) { ikj_avx2_base(A,B,C,n); return; }
    int h=n/2,sz=h*h;
    double *A11=malloc(sz*8),*A12=malloc(sz*8),*A21=malloc(sz*8),*A22=malloc(sz*8);
    double *B11=malloc(sz*8),*B12=malloc(sz*8),*B21=malloc(sz*8),*B22=malloc(sz*8);
    double *M1=malloc(sz*8),*M2=malloc(sz*8),*M3=malloc(sz*8),*M4=malloc(sz*8);
    double *M5=malloc(sz*8),*M6=malloc(sz*8),*M7=malloc(sz*8),*T1=malloc(sz*8),*T2=malloc(sz*8);
    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(B,B11,h,0,0,n);get_q(B,B12,h,0,h,n);get_q(B,B21,h,h,0,n);get_q(B,B22,h,h,h,n);
    add_f(A11,A22,T1,sz);add_f(B11,B22,T2,sz);pure_str_avx(T1,T2,M1,h,thresh);
    add_f(A21,A22,T1,sz);pure_str_avx(T1,B11,M2,h,thresh);
    sub_f(B12,B22,T1,sz);pure_str_avx(A11,T1,M3,h,thresh);
    sub_f(B21,B11,T1,sz);pure_str_avx(A22,T1,M4,h,thresh);
    add_f(A11,A12,T1,sz);pure_str_avx(T1,B22,M5,h,thresh);
    sub_f(A21,A11,T1,sz);add_f(B11,B12,T2,sz);pure_str_avx(T1,T2,M6,h,thresh);
    sub_f(A12,A22,T1,sz);add_f(B21,B22,T2,sz);pure_str_avx(T1,T2,M7,h,thresh);
    add_f(M1,M4,T1,sz);sub_f(T1,M5,T2,sz);add_f(T2,M7,T1,sz);set_q(C,T1,h,0,0,n);
    add_f(M3,M5,T1,sz);set_q(C,T1,h,0,h,n);
    add_f(M2,M4,T1,sz);set_q(C,T1,h,h,0,n);
    sub_f(M1,M2,T1,sz);add_f(T1,M3,T2,sz);add_f(T2,M6,T1,sz);set_q(C,T1,h,h,h,n);
    free(A11);free(A12);free(A21);free(A22);free(B11);free(B12);free(B21);free(B22);
    free(M1);free(M2);free(M3);free(M4);free(M5);free(M6);free(M7);free(T1);free(T2);
}

double get_ms(){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec*1e3+t.tv_nsec/1e6;}
void fill_rand(double *M, int n){for(int i=0;i<n*n;i++) M[i]=(double)rand()/RAND_MAX*2-1;}
double mdiff(double *A, double *B, int n){double m=0;for(int i=0;i<n*n;i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;}return m;}

int main() {
    printf("====================================================================\n");
    printf("  COM6 v7 - AVX2 Intrinsics Benchmark\n");
    printf("  Both COM6 and Strassen use hand-written AVX2 FMA inner kernels\n");
    printf("  COM6: dual contiguous FMA loads | Strassen: broadcast+FMA+store\n");
    printf("====================================================================\n\n");

    /* Threshold sweep at 2048 */
    printf("--- Threshold sweep at 2048 ---\n");
    {
        int n=2048;
        double *A=malloc(n*n*8),*B=malloc(n*n*8),*C=malloc(n*n*8);
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int ths[]={64,128,256,512};
        for(int t=0;t<4;t++){double t0=get_ms();com6_avx(A,B,C,n,ths[t]);printf("  COM6-AVX2 thresh=%3d: %8.1f ms\n",ths[t],get_ms()-t0);}
        for(int t=0;t<4;t++){double t0=get_ms();pure_str_avx(A,B,C,n,ths[t]);printf("  Str-AVX2  thresh=%3d: %8.1f ms\n",ths[t],get_ms()-t0);}
        free(A);free(B);free(C);
    }

    /* Main benchmark */
    printf("\n%-9s | %10s | %10s | %7s | %s\n","Size","Str-AVX2","COM6-AVX2","COM6/Str","Verify");
    printf("------------------------------------------------------\n");

    int sizes[]={256,512,1024,2048,4096};
    for(int si=0;si<5;si++){
        int n=sizes[si];
        double *A=malloc(n*n*8),*B=malloc(n*n*8),*C1=malloc(n*n*8),*C2=malloc(n*n*8);
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int runs=n<=512?3:1;
        double t_str=0,t_com=0;

        /* Use best thresholds from sweep */
        if((n&(n-1))==0){
            double t0=get_ms();for(int r=0;r<runs;r++)pure_str_avx(A,B,C1,n,128);t_str=(get_ms()-t0)/runs;
            t0=get_ms();for(int r=0;r<runs;r++)com6_avx(A,B,C2,n,64);t_com=(get_ms()-t0)/runs;
        }

        double diff=t_str>0&&t_com>0?mdiff(C1,C2,n):0;
        char vfy[20];snprintf(vfy,20,diff<1e-5?"OK":"e=%.0e",diff);
        char ss[20],sc[20];
        if(t_str>0)snprintf(ss,20,"%8.1f ms",t_str);else snprintf(ss,20,"%10s","N/A");
        if(t_com>0)snprintf(sc,20,"%8.1f ms",t_com);else snprintf(sc,20,"%10s","N/A");

        printf("%4dx%-4d | %10s | %10s |",n,n,ss,sc);
        if(t_str>0&&t_com>0)printf(" %6.2fx |",t_str/t_com);else printf(" %7s |","-");
        printf(" %s\n",vfy);

        free(A);free(B);free(C1);free(C2);
    }
    printf("\nCOM6/Str > 1.0 = COM6 BEATS Strassen (both using AVX2)\n");
    return 0;
}
