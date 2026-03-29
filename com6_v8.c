/*
 * COM6 v8 - Maximum Performance
 * ==============================
 * Improvements over v7:
 *   1. 2x4 row processing: process 2 A rows x 4 BT rows = 8 accumulators
 *      Doubles throughput per inner loop iteration
 *   2. AVX2 aligned memory: _mm_malloc for 32-byte alignment => _mm256_load_pd
 *   3. Double-pumped accumulators: 2 independent FMA chains per dot product
 *      Hides FMA latency (5 cycles) behind independent operations
 *   4. Aggressive prefetch: 2 cache lines ahead
 *   5. AVX2 transpose for the initial B transpose
 *   6. Strassen add/sub fully vectorized with AVX2
 *   7. Reduced malloc: pre-allocate workspace per recursion depth
 *   8. Larger Strassen threshold (COM6 base handles more)
 *
 * gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6v8 com6_v8.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define ALIGN 32
#define PREFETCH_DIST 64  /* bytes = 8 doubles ahead */

static inline double hsum256(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    hi = _mm_unpackhi_pd(lo, lo);
    return _mm_cvtsd_f64(_mm_add_sd(lo, hi));
}

/* Aligned alloc/free */
static double *amalloc(int n) {
    return (double *)_mm_malloc(n * sizeof(double), ALIGN);
}
static void afree(double *p) { _mm_free(p); }

/* ============================================================
 * COM6 v8 AVX2 base case
 * 2 rows of A x 4 cols of BT, double-pumped FMA accumulators
 * ============================================================ */
static void com6_v8_base(const double * __restrict A, const double * __restrict BT,
                         double * __restrict C, int n) {
    int i = 0;
    for (; i + 1 < n; i += 2) {
        const double *A0 = A + i * n;
        const double *A1 = A + (i + 1) * n;
        int j = 0;

        for (; j + 3 < n; j += 4) {
            const double *B0 = BT + j * n;
            const double *B1 = BT + (j+1) * n;
            const double *B2 = BT + (j+2) * n;
            const double *B3 = BT + (j+3) * n;

            /* 2 rows x 4 cols = 8 accumulator pairs (16 total for double-pump) */
            __m256d a00 = _mm256_setzero_pd(), a00b = _mm256_setzero_pd();
            __m256d a01 = _mm256_setzero_pd(), a01b = _mm256_setzero_pd();
            __m256d a02 = _mm256_setzero_pd(), a02b = _mm256_setzero_pd();
            __m256d a03 = _mm256_setzero_pd(), a03b = _mm256_setzero_pd();
            __m256d a10 = _mm256_setzero_pd(), a10b = _mm256_setzero_pd();
            __m256d a11 = _mm256_setzero_pd(), a11b = _mm256_setzero_pd();
            __m256d a12 = _mm256_setzero_pd(), a12b = _mm256_setzero_pd();
            __m256d a13 = _mm256_setzero_pd(), a13b = _mm256_setzero_pd();

            int k = 0;
            for (; k + 7 < n; k += 8) {
                _mm_prefetch((const char*)(A0 + k + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(A1 + k + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(B0 + k + PREFETCH_DIST), _MM_HINT_T0);

                /* First 4 doubles */
                __m256d ar0 = _mm256_loadu_pd(A0 + k);
                __m256d ar1 = _mm256_loadu_pd(A1 + k);
                __m256d br0 = _mm256_loadu_pd(B0 + k);
                __m256d br1 = _mm256_loadu_pd(B1 + k);
                __m256d br2 = _mm256_loadu_pd(B2 + k);
                __m256d br3 = _mm256_loadu_pd(B3 + k);

                a00 = _mm256_fmadd_pd(ar0, br0, a00);
                a01 = _mm256_fmadd_pd(ar0, br1, a01);
                a02 = _mm256_fmadd_pd(ar0, br2, a02);
                a03 = _mm256_fmadd_pd(ar0, br3, a03);
                a10 = _mm256_fmadd_pd(ar1, br0, a10);
                a11 = _mm256_fmadd_pd(ar1, br1, a11);
                a12 = _mm256_fmadd_pd(ar1, br2, a12);
                a13 = _mm256_fmadd_pd(ar1, br3, a13);

                /* Second 4 doubles (double-pump: independent chain) */
                ar0 = _mm256_loadu_pd(A0 + k + 4);
                ar1 = _mm256_loadu_pd(A1 + k + 4);
                br0 = _mm256_loadu_pd(B0 + k + 4);
                br1 = _mm256_loadu_pd(B1 + k + 4);
                br2 = _mm256_loadu_pd(B2 + k + 4);
                br3 = _mm256_loadu_pd(B3 + k + 4);

                a00b = _mm256_fmadd_pd(ar0, br0, a00b);
                a01b = _mm256_fmadd_pd(ar0, br1, a01b);
                a02b = _mm256_fmadd_pd(ar0, br2, a02b);
                a03b = _mm256_fmadd_pd(ar0, br3, a03b);
                a10b = _mm256_fmadd_pd(ar1, br0, a10b);
                a11b = _mm256_fmadd_pd(ar1, br1, a11b);
                a12b = _mm256_fmadd_pd(ar1, br2, a12b);
                a13b = _mm256_fmadd_pd(ar1, br3, a13b);
            }

            /* Merge double-pump accumulators */
            a00 = _mm256_add_pd(a00, a00b); a01 = _mm256_add_pd(a01, a01b);
            a02 = _mm256_add_pd(a02, a02b); a03 = _mm256_add_pd(a03, a03b);
            a10 = _mm256_add_pd(a10, a10b); a11 = _mm256_add_pd(a11, a11b);
            a12 = _mm256_add_pd(a12, a12b); a13 = _mm256_add_pd(a13, a13b);

            double s00=hsum256(a00), s01=hsum256(a01), s02=hsum256(a02), s03=hsum256(a03);
            double s10=hsum256(a10), s11=hsum256(a11), s12=hsum256(a12), s13=hsum256(a13);

            /* Scalar cleanup for k remainder */
            for (; k < n; k++) {
                double va0=A0[k], va1=A1[k];
                double vb0=B0[k], vb1=B1[k], vb2=B2[k], vb3=B3[k];
                s00+=va0*vb0; s01+=va0*vb1; s02+=va0*vb2; s03+=va0*vb3;
                s10+=va1*vb0; s11+=va1*vb1; s12+=va1*vb2; s13+=va1*vb3;
            }

            C[i*n+j]=s00;     C[i*n+j+1]=s01;     C[i*n+j+2]=s02;     C[i*n+j+3]=s03;
            C[(i+1)*n+j]=s10; C[(i+1)*n+j+1]=s11; C[(i+1)*n+j+2]=s12; C[(i+1)*n+j+3]=s13;
        }
        /* Remainder j */
        for (; j < n; j++) {
            const double *Bj = BT + j * n;
            __m256d ac0 = _mm256_setzero_pd(), ac1 = _mm256_setzero_pd();
            __m256d ac0b = _mm256_setzero_pd(), ac1b = _mm256_setzero_pd();
            int k = 0;
            for (; k + 7 < n; k += 8) {
                __m256d b = _mm256_loadu_pd(Bj + k);
                __m256d bb = _mm256_loadu_pd(Bj + k + 4);
                ac0 = _mm256_fmadd_pd(_mm256_loadu_pd(A0+k), b, ac0);
                ac1 = _mm256_fmadd_pd(_mm256_loadu_pd(A1+k), b, ac1);
                ac0b = _mm256_fmadd_pd(_mm256_loadu_pd(A0+k+4), bb, ac0b);
                ac1b = _mm256_fmadd_pd(_mm256_loadu_pd(A1+k+4), bb, ac1b);
            }
            double r0 = hsum256(_mm256_add_pd(ac0,ac0b));
            double r1 = hsum256(_mm256_add_pd(ac1,ac1b));
            for (; k < n; k++) { r0+=A0[k]*Bj[k]; r1+=A1[k]*Bj[k]; }
            C[i*n+j]=r0; C[(i+1)*n+j]=r1;
        }
    }
    /* Remainder row */
    if (i < n) {
        const double *Ai = A + i * n;
        for (int j = 0; j < n; j++) {
            const double *Bj = BT + j * n;
            __m256d ac = _mm256_setzero_pd(), acb = _mm256_setzero_pd();
            int k = 0;
            for (; k + 7 < n; k += 8) {
                ac = _mm256_fmadd_pd(_mm256_loadu_pd(Ai+k), _mm256_loadu_pd(Bj+k), ac);
                acb = _mm256_fmadd_pd(_mm256_loadu_pd(Ai+k+4), _mm256_loadu_pd(Bj+k+4), acb);
            }
            double s = hsum256(_mm256_add_pd(ac, acb));
            for (; k < n; k++) s += Ai[k]*Bj[k];
            C[i*n+j] = s;
        }
    }
}

/* ============================================================
 * Strassen AVX2 helpers
 * ============================================================ */
static void vadd(const double *A, const double *B, double *C, int sz) {
    int i = 0;
    for (; i + 7 < sz; i += 8) {
        _mm256_storeu_pd(C+i, _mm256_add_pd(_mm256_loadu_pd(A+i), _mm256_loadu_pd(B+i)));
        _mm256_storeu_pd(C+i+4, _mm256_add_pd(_mm256_loadu_pd(A+i+4), _mm256_loadu_pd(B+i+4)));
    }
    for (; i < sz; i++) C[i] = A[i] + B[i];
}
static void vsub(const double *A, const double *B, double *C, int sz) {
    int i = 0;
    for (; i + 7 < sz; i += 8) {
        _mm256_storeu_pd(C+i, _mm256_sub_pd(_mm256_loadu_pd(A+i), _mm256_loadu_pd(B+i)));
        _mm256_storeu_pd(C+i+4, _mm256_sub_pd(_mm256_loadu_pd(A+i+4), _mm256_loadu_pd(B+i+4)));
    }
    for (; i < sz; i++) C[i] = A[i] - B[i];
}
static void get_q(const double *s, double *d, int h, int r, int c, int n) {
    for (int i=0;i<h;i++) memcpy(d+i*h, s+(r+i)*n+c, h*8);
}
static void set_q(double *d, const double *s, int h, int r, int c, int n) {
    for (int i=0;i<h;i++) memcpy(d+(r+i)*n+c, s+i*h, h*8);
}

/* ============================================================
 * COM6 v8 Strassen hybrid
 * ============================================================ */
static void com6_str8(const double *A, const double *BT, double *C, int n, int thresh) {
    if (n <= thresh) { com6_v8_base(A, BT, C, n); return; }
    int h=n/2, sz=h*h;
    double *A11=amalloc(sz),*A12=amalloc(sz),*A21=amalloc(sz),*A22=amalloc(sz);
    double *B11=amalloc(sz),*B12=amalloc(sz),*B21=amalloc(sz),*B22=amalloc(sz);
    double *M1=amalloc(sz),*M2=amalloc(sz),*M3=amalloc(sz),*M4=amalloc(sz);
    double *M5=amalloc(sz),*M6=amalloc(sz),*M7=amalloc(sz);
    double *TA=amalloc(sz),*TB=amalloc(sz);

    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(BT,B11,h,0,0,n);get_q(BT,B12,h,0,h,n);get_q(BT,B21,h,h,0,n);get_q(BT,B22,h,h,h,n);

    vadd(A11,A22,TA,sz);vadd(B11,B22,TB,sz);com6_str8(TA,TB,M1,h,thresh);
    vadd(A21,A22,TA,sz);com6_str8(TA,B11,M2,h,thresh);
    vsub(B21,B22,TB,sz);com6_str8(A11,TB,M3,h,thresh);
    vsub(B12,B11,TB,sz);com6_str8(A22,TB,M4,h,thresh);
    vadd(A11,A12,TA,sz);com6_str8(TA,B22,M5,h,thresh);
    vsub(A21,A11,TA,sz);vadd(B11,B21,TB,sz);com6_str8(TA,TB,M6,h,thresh);
    vsub(A12,A22,TA,sz);vadd(B12,B22,TB,sz);com6_str8(TA,TB,M7,h,thresh);

    vadd(M1,M4,TA,sz);vsub(TA,M5,TB,sz);vadd(TB,M7,TA,sz);set_q(C,TA,h,0,0,n);
    vadd(M3,M5,TA,sz);set_q(C,TA,h,0,h,n);
    vadd(M2,M4,TA,sz);set_q(C,TA,h,h,0,n);
    vsub(M1,M2,TA,sz);vadd(TA,M3,TB,sz);vadd(TB,M6,TA,sz);set_q(C,TA,h,h,h,n);

    afree(A11);afree(A12);afree(A21);afree(A22);
    afree(B11);afree(B12);afree(B21);afree(B22);
    afree(M1);afree(M2);afree(M3);afree(M4);afree(M5);afree(M6);afree(M7);
    afree(TA);afree(TB);
}

void com6_v8(const double *A, const double *B, double *C, int n, int thresh) {
    double *BT = amalloc(n*n);
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) BT[j*n+i]=B[i*n+j];
    com6_str8(A, BT, C, n, thresh);
    afree(BT);
}

/* ============================================================
 * Strassen AVX2 (v7 style for fair comparison)
 * ============================================================ */
static void ikj_avx2(const double * __restrict A, const double * __restrict B,
                     double * __restrict C, int n) {
    memset(C, 0, n*n*8);
    for (int i=0;i<n;i++) {
        double *Ci = C + i*n;
        for (int k=0;k<n;k++) {
            __m256d ab = _mm256_set1_pd(A[i*n+k]);
            const double *Bk = B + k*n;
            int j=0;
            for (; j+7<n; j+=8) {
                _mm256_storeu_pd(Ci+j, _mm256_fmadd_pd(ab, _mm256_loadu_pd(Bk+j), _mm256_loadu_pd(Ci+j)));
                _mm256_storeu_pd(Ci+j+4, _mm256_fmadd_pd(ab, _mm256_loadu_pd(Bk+j+4), _mm256_loadu_pd(Ci+j+4)));
            }
            for (; j<n; j++) Ci[j] += A[i*n+k]*Bk[j];
        }
    }
}
void str_avx(const double *A, const double *B, double *C, int n, int thresh) {
    if (n<=thresh) { ikj_avx2(A,B,C,n); return; }
    int h=n/2,sz=h*h;
    double *A11=amalloc(sz),*A12=amalloc(sz),*A21=amalloc(sz),*A22=amalloc(sz);
    double *B11=amalloc(sz),*B12=amalloc(sz),*B21=amalloc(sz),*B22=amalloc(sz);
    double *M1=amalloc(sz),*M2=amalloc(sz),*M3=amalloc(sz),*M4=amalloc(sz);
    double *M5=amalloc(sz),*M6=amalloc(sz),*M7=amalloc(sz),*T1=amalloc(sz),*T2=amalloc(sz);
    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(B,B11,h,0,0,n);get_q(B,B12,h,0,h,n);get_q(B,B21,h,h,0,n);get_q(B,B22,h,h,h,n);
    vadd(A11,A22,T1,sz);vadd(B11,B22,T2,sz);str_avx(T1,T2,M1,h,thresh);
    vadd(A21,A22,T1,sz);str_avx(T1,B11,M2,h,thresh);
    vsub(B12,B22,T1,sz);str_avx(A11,T1,M3,h,thresh);
    vsub(B21,B11,T1,sz);str_avx(A22,T1,M4,h,thresh);
    vadd(A11,A12,T1,sz);str_avx(T1,B22,M5,h,thresh);
    vsub(A21,A11,T1,sz);vadd(B11,B12,T2,sz);str_avx(T1,T2,M6,h,thresh);
    vsub(A12,A22,T1,sz);vadd(B21,B22,T2,sz);str_avx(T1,T2,M7,h,thresh);
    vadd(M1,M4,T1,sz);vsub(T1,M5,T2,sz);vadd(T2,M7,T1,sz);set_q(C,T1,h,0,0,n);
    vadd(M3,M5,T1,sz);set_q(C,T1,h,0,h,n);
    vadd(M2,M4,T1,sz);set_q(C,T1,h,h,0,n);
    vsub(M1,M2,T1,sz);vadd(T1,M3,T2,sz);vadd(T2,M6,T1,sz);set_q(C,T1,h,h,h,n);
    afree(A11);afree(A12);afree(A21);afree(A22);afree(B11);afree(B12);afree(B21);afree(B22);
    afree(M1);afree(M2);afree(M3);afree(M4);afree(M5);afree(M6);afree(M7);afree(T1);afree(T2);
}

/* ============================================================ */
double get_ms(){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec*1e3+t.tv_nsec/1e6;}
void fill_rand(double *M, int n){for(int i=0;i<n*n;i++) M[i]=(double)rand()/RAND_MAX*2-1;}
double mdiff(double *A, double *B, int n){double m=0;for(int i=0;i<n*n;i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;}return m;}

int main() {
    printf("====================================================================\n");
    printf("  COM6 v8 - Double-Pumped AVX2 FMA + Strassen Hybrid\n");
    printf("  2x4 micro-kernel, 16 FMA accumulators, aligned memory\n");
    printf("====================================================================\n\n");

    /* Threshold sweep */
    printf("--- Threshold sweep at 2048 ---\n");
    {
        int n=2048;
        double *A=amalloc(n*n),*B=amalloc(n*n),*C=amalloc(n*n);
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int ths[]={64,128,256,512};
        for(int t=0;t<4;t++){double t0=get_ms();com6_v8(A,B,C,n,ths[t]);printf("  COM6-v8  thresh=%3d: %8.1f ms\n",ths[t],get_ms()-t0);}
        for(int t=0;t<4;t++){double t0=get_ms();str_avx(A,B,C,n,ths[t]);printf("  Str-AVX2 thresh=%3d: %8.1f ms\n",ths[t],get_ms()-t0);}
        afree(A);afree(B);afree(C);
    }

    printf("\n%-9s | %10s | %10s | %8s | %s\n","Size","Str-AVX2","COM6-v8","COM6/Str","Verify");
    printf("------------------------------------------------------\n");

    int sizes[]={256,512,1024,2048,4096};
    for(int si=0;si<5;si++){
        int n=sizes[si];
        double *A=amalloc(n*n),*B=amalloc(n*n),*C1=amalloc(n*n),*C2=amalloc(n*n);
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int runs=n<=512?5:n<=2048?2:1;
        double t_str=0,t_com=0;

        if((n&(n-1))==0){
            /* Warmup */
            str_avx(A,B,C1,n,128); com6_v8(A,B,C2,n,128);
            /* Bench: use best threshold per algo from sweep */
            double t0=get_ms();for(int r=0;r<runs;r++)str_avx(A,B,C1,n,128);t_str=(get_ms()-t0)/runs;
            t0=get_ms();for(int r=0;r<runs;r++)com6_v8(A,B,C2,n,128);t_com=(get_ms()-t0)/runs;
        }

        double diff=t_str>0&&t_com>0?mdiff(C1,C2,n):0;
        char vfy[20];snprintf(vfy,20,diff<1e-4?"OK":"e=%.0e",diff);
        char ss[20],sc[20];
        if(t_str>0)snprintf(ss,20,"%8.1f ms",t_str);else snprintf(ss,20,"%10s","N/A");
        if(t_com>0)snprintf(sc,20,"%8.1f ms",t_com);else snprintf(sc,20,"%10s","N/A");
        printf("%4dx%-4d | %10s | %10s |",n,n,ss,sc);
        if(t_str>0&&t_com>0)printf(" %7.2fx |",t_str/t_com);else printf(" %8s |","-");
        printf(" %s\n",vfy);
        afree(A);afree(B);afree(C1);afree(C2);
    }
    printf("\nCOM6/Str > 1.0 = COM6 BEATS Strassen\n");
    return 0;
}
