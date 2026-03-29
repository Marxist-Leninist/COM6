/*
 * COM6 v6 - Compiler-Friendly Hybrid
 * ====================================
 * Key realization: GCC -O3 -march=native auto-vectorizes simple loops
 * into AVX2 packed operations (4 doubles/cycle). Manual scalar unrolling
 * BLOCKS this. So: write clean, simple loops and let the compiler win.
 *
 * COM6 advantage: transpose B once, then inner loop has stride-1 on both
 * arrays => compiler can generate aligned AVX2 loads for both operands.
 * Standard ikj has stride-1 on B but C accumulation needs scatter.
 *
 * gcc -O3 -march=native -ftree-vectorize -ffast-math -o com6v6 com6_bench_v6.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================
 * COM6 base case: clean loop for auto-vectorization
 * Both A row and BT row are contiguous => compiler generates
 * packed AVX2 vmulpd + vaddpd for the inner dot product
 * ============================================================ */
static void com6_dot_base(const double * __restrict A, const double * __restrict BT,
                          double * __restrict C, int n) {
    for (int i = 0; i < n; i++) {
        const double * __restrict Ai = A + i * n;
        for (int j = 0; j < n; j++) {
            const double * __restrict Bj = BT + j * n;
            double sum = 0.0;
            /* Simple loop — compiler auto-vectorizes this to AVX2
             * Both Ai[k] and Bj[k] are stride-1, contiguous in memory */
            for (int k = 0; k < n; k++) {
                sum += Ai[k] * Bj[k];
            }
            C[i * n + j] = sum;
        }
    }
}

/* ============================================================
 * ikj base case (Strassen's standard base)
 * Inner loop: C[i*n+j] += a * B[k*n+j] — compiler vectorizes on j
 * ============================================================ */
static void ikj_base(const double * __restrict A, const double * __restrict B,
                     double * __restrict C, int n) {
    memset(C, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a = A[i * n + k];
            double * __restrict Ci = C + i * n;
            const double * __restrict Bk = B + k * n;
            /* Compiler vectorizes this: Ci[j] += a * Bk[j] */
            for (int j = 0; j < n; j++) {
                Ci[j] += a * Bk[j];
            }
        }
    }
}

/* ============================================================
 * Strassen helpers
 * ============================================================ */
static void add_f(const double *A, const double *B, double *C, int sz) {
    for (int i = 0; i < sz; i++) C[i] = A[i] + B[i];
}
static void sub_f(const double *A, const double *B, double *C, int sz) {
    for (int i = 0; i < sz; i++) C[i] = A[i] - B[i];
}
static void get_q(const double *s, double *d, int h, int r, int c, int n) {
    for (int i = 0; i < h; i++) memcpy(d + i*h, s + (r+i)*n + c, h*8);
}
static void set_q(double *d, const double *s, int h, int r, int c, int n) {
    for (int i = 0; i < h; i++) memcpy(d + (r+i)*n + c, s + i*h, h*8);
}

/* ============================================================
 * COM6-Strassen v6: Strassen + auto-vectorized COM6 base
 * ============================================================ */
static void com6_str6(const double *A, const double *BT, double *C, int n, int thresh) {
    if (n <= thresh) { com6_dot_base(A, BT, C, n); return; }
    int h=n/2, sz=h*h;
    double *A11=malloc(sz*8),*A12=malloc(sz*8),*A21=malloc(sz*8),*A22=malloc(sz*8);
    double *BT11=malloc(sz*8),*BT12=malloc(sz*8),*BT21=malloc(sz*8),*BT22=malloc(sz*8);
    double *M1=malloc(sz*8),*M2=malloc(sz*8),*M3=malloc(sz*8),*M4=malloc(sz*8);
    double *M5=malloc(sz*8),*M6=malloc(sz*8),*M7=malloc(sz*8);
    double *TA=malloc(sz*8),*TB=malloc(sz*8);

    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(BT,BT11,h,0,0,n);get_q(BT,BT12,h,0,h,n);get_q(BT,BT21,h,h,0,n);get_q(BT,BT22,h,h,h,n);

    add_f(A11,A22,TA,sz);add_f(BT11,BT22,TB,sz);com6_str6(TA,TB,M1,h,thresh);
    add_f(A21,A22,TA,sz);com6_str6(TA,BT11,M2,h,thresh);
    sub_f(BT21,BT22,TB,sz);com6_str6(A11,TB,M3,h,thresh);
    sub_f(BT12,BT11,TB,sz);com6_str6(A22,TB,M4,h,thresh);
    add_f(A11,A12,TA,sz);com6_str6(TA,BT22,M5,h,thresh);
    sub_f(A21,A11,TA,sz);add_f(BT11,BT21,TB,sz);com6_str6(TA,TB,M6,h,thresh);
    sub_f(A12,A22,TA,sz);add_f(BT12,BT22,TB,sz);com6_str6(TA,TB,M7,h,thresh);

    add_f(M1,M4,TA,sz);sub_f(TA,M5,TB,sz);add_f(TB,M7,TA,sz);set_q(C,TA,h,0,0,n);
    add_f(M3,M5,TA,sz);set_q(C,TA,h,0,h,n);
    add_f(M2,M4,TA,sz);set_q(C,TA,h,h,0,n);
    sub_f(M1,M2,TA,sz);add_f(TA,M3,TB,sz);add_f(TB,M6,TA,sz);set_q(C,TA,h,h,h,n);

    free(A11);free(A12);free(A21);free(A22);
    free(BT11);free(BT12);free(BT21);free(BT22);
    free(M1);free(M2);free(M3);free(M4);free(M5);free(M6);free(M7);
    free(TA);free(TB);
}

void com6_v6(const double *A, const double *B, double *C, int n, int thresh) {
    double *BT = malloc(n*n*8);
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) BT[j*n+i]=B[i*n+j];
    com6_str6(A, BT, C, n, thresh);
    free(BT);
}

/* Pure Strassen */
void pure_str(const double *A, const double *B, double *C, int n, int thresh) {
    if (n<=thresh) { ikj_base(A,B,C,n); return; }
    int h=n/2,sz=h*h;
    double *A11=malloc(sz*8),*A12=malloc(sz*8),*A21=malloc(sz*8),*A22=malloc(sz*8);
    double *B11=malloc(sz*8),*B12=malloc(sz*8),*B21=malloc(sz*8),*B22=malloc(sz*8);
    double *M1=malloc(sz*8),*M2=malloc(sz*8),*M3=malloc(sz*8),*M4=malloc(sz*8);
    double *M5=malloc(sz*8),*M6=malloc(sz*8),*M7=malloc(sz*8),*T1=malloc(sz*8),*T2=malloc(sz*8);
    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(B,B11,h,0,0,n);get_q(B,B12,h,0,h,n);get_q(B,B21,h,h,0,n);get_q(B,B22,h,h,h,n);
    add_f(A11,A22,T1,sz);add_f(B11,B22,T2,sz);pure_str(T1,T2,M1,h,thresh);
    add_f(A21,A22,T1,sz);pure_str(T1,B11,M2,h,thresh);
    sub_f(B12,B22,T1,sz);pure_str(A11,T1,M3,h,thresh);
    sub_f(B21,B11,T1,sz);pure_str(A22,T1,M4,h,thresh);
    add_f(A11,A12,T1,sz);pure_str(T1,B22,M5,h,thresh);
    sub_f(A21,A11,T1,sz);add_f(B11,B12,T2,sz);pure_str(T1,T2,M6,h,thresh);
    sub_f(A12,A22,T1,sz);add_f(B21,B22,T2,sz);pure_str(T1,T2,M7,h,thresh);
    add_f(M1,M4,T1,sz);sub_f(T1,M5,T2,sz);add_f(T2,M7,T1,sz);set_q(C,T1,h,0,0,n);
    add_f(M3,M5,T1,sz);set_q(C,T1,h,0,h,n);
    add_f(M2,M4,T1,sz);set_q(C,T1,h,h,0,n);
    sub_f(M1,M2,T1,sz);add_f(T1,M3,T2,sz);add_f(T2,M6,T1,sz);set_q(C,T1,h,h,h,n);
    free(A11);free(A12);free(A21);free(A22);free(B11);free(B12);free(B21);free(B22);
    free(M1);free(M2);free(M3);free(M4);free(M5);free(M6);free(M7);free(T1);free(T2);
}

/* Standard */
void std_mm(const double *A, const double *B, double *C, int n) {
    memset(C,0,n*n*8);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) {
        double s=0; for(int k=0;k<n;k++) s+=A[i*n+k]*B[k*n+j]; C[i*n+j]=s;
    }
}

double get_ms(){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec*1e3+t.tv_nsec/1e6;}
void fill_rand(double *M, int n){for(int i=0;i<n*n;i++) M[i]=(double)rand()/RAND_MAX*2-1;}
double mdiff(double *A, double *B, int n){double m=0;for(int i=0;i<n*n;i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;}return m;}

int main() {
    printf("========================================================================\n");
    printf("  COM6 v6: Auto-Vectorized Hybrid\n");
    printf("  Compiler generates AVX2 SIMD for both COM6 and Strassen base cases\n");
    printf("  COM6 advantage: both operands stride-1 => dual aligned AVX2 loads\n");
    printf("========================================================================\n\n");

    /* Threshold sweep */
    printf("--- Threshold sweep at 2048x2048 ---\n");
    {
        int n=2048;
        double *A=malloc(n*n*8),*B=malloc(n*n*8),*C=malloc(n*n*8);
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int ths[]={64,128,256,512};
        for(int t=0;t<4;t++){double t0=get_ms();com6_v6(A,B,C,n,ths[t]);printf("  COM6-v6  thresh=%3d: %8.1f ms\n",ths[t],get_ms()-t0);}
        for(int t=0;t<4;t++){double t0=get_ms();pure_str(A,B,C,n,ths[t]);printf("  Strassen thresh=%3d: %8.1f ms\n",ths[t],get_ms()-t0);}
        free(A);free(B);free(C);
    }

    /* Main benchmark */
    printf("\n--- Main Benchmark (best threshold per algo) ---\n");
    printf("%-9s | %10s | %10s | %10s | %7s | %s\n","Size","Standard","Strassen","COM6-v6","v6/Str","Verify");
    printf("-------------------------------------------------------------------\n");

    int sizes[]={256,512,1024,2048,4096};
    for(int si=0;si<5;si++){
        int n=sizes[si];
        double *A=malloc(n*n*8),*B=malloc(n*n*8),*C1=malloc(n*n*8),*C2=malloc(n*n*8),*C3=malloc(n*n*8);
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int runs=n<=512?3:1;
        double t_std=0,t_str=0,t_v6=0;

        if(n<=1024){double t0=get_ms();for(int r=0;r<runs;r++)std_mm(A,B,C3,n);t_std=(get_ms()-t0)/runs;}
        /* Best Strassen threshold from sweep */
        if((n&(n-1))==0){double t0=get_ms();for(int r=0;r<runs;r++)pure_str(A,B,C1,n,128);t_str=(get_ms()-t0)/runs;}
        /* Best COM6 threshold from sweep */
        if((n&(n-1))==0){double t0=get_ms();for(int r=0;r<runs;r++)com6_v6(A,B,C2,n,256);t_v6=(get_ms()-t0)/runs;}

        double diff=t_str>0&&t_v6>0?mdiff(C1,C2,n):0;
        char vfy[20];snprintf(vfy,20,diff<1e-5?"OK":"e=%.0e",diff);
        char ss[20],sr[20],sv[20];
        if(t_std>0)snprintf(ss,20,"%8.1f ms",t_std);else snprintf(ss,20,"%10s","SKIP");
        if(t_str>0)snprintf(sr,20,"%8.1f ms",t_str);else snprintf(sr,20,"%10s","N/A");
        if(t_v6>0)snprintf(sv,20,"%8.1f ms",t_v6);else snprintf(sv,20,"%10s","N/A");

        printf("%4dx%-4d | %10s | %10s | %10s |",n,n,ss,sr,sv);
        if(t_str>0&&t_v6>0)printf(" %6.2fx |",t_str/t_v6);else printf(" %7s |","-");
        printf(" %s\n",vfy);

        free(A);free(B);free(C1);free(C2);free(C3);
    }
    printf("\nv6/Str > 1.0 = COM6 BEATS Strassen\n");
    return 0;
}
