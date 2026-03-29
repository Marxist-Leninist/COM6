/*
 * COM6 v5 - Lean Hybrid
 * ======================
 * Strassen recursion + ultra-lean COM6 base case:
 *   - Pre-transpose B once at top
 *   - Base case: direct 4x1 micro-kernel, NO block tiling overhead
 *   - 8-wide unrolled inner loop on stride-1 arrays
 *   - Tunable threshold
 *
 * gcc -O3 -march=native -funroll-loops -o com6v5 com6_bench_v5.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PREFETCH(addr) __builtin_prefetch((addr), 0, 3)

/* ============================================================
 * COM6 lean base case: no block tiling, just micro-kernel rows
 * A[n,n] normal, BT[n,n] pre-transposed, C[n,n] output
 * ============================================================ */
static void com6_lean(const double * __restrict A, const double * __restrict BT,
                      double * __restrict C, int n) {
    memset(C, 0, n * n * sizeof(double));
    int i = 0;

    /* Process 4 rows at a time */
    for (; i + 3 < n; i += 4) {
        const double *A0 = A + i*n;
        const double *A1 = A + (i+1)*n;
        const double *A2 = A + (i+2)*n;
        const double *A3 = A + (i+3)*n;

        for (int j = 0; j < n; j++) {
            const double *Bj = BT + j*n;
            double s0=0, s1=0, s2=0, s3=0;
            int k = 0;

            /* 8-wide unroll — both Ai and Bj are stride-1 */
            for (; k + 7 < n; k += 8) {
                PREFETCH(A0 + k + 16);
                PREFETCH(Bj + k + 16);
                double b0=Bj[k],b1=Bj[k+1],b2=Bj[k+2],b3=Bj[k+3];
                double b4=Bj[k+4],b5=Bj[k+5],b6=Bj[k+6],b7=Bj[k+7];
                s0 += A0[k]*b0+A0[k+1]*b1+A0[k+2]*b2+A0[k+3]*b3
                    + A0[k+4]*b4+A0[k+5]*b5+A0[k+6]*b6+A0[k+7]*b7;
                s1 += A1[k]*b0+A1[k+1]*b1+A1[k+2]*b2+A1[k+3]*b3
                    + A1[k+4]*b4+A1[k+5]*b5+A1[k+6]*b6+A1[k+7]*b7;
                s2 += A2[k]*b0+A2[k+1]*b1+A2[k+2]*b2+A2[k+3]*b3
                    + A2[k+4]*b4+A2[k+5]*b5+A2[k+6]*b6+A2[k+7]*b7;
                s3 += A3[k]*b0+A3[k+1]*b1+A3[k+2]*b2+A3[k+3]*b3
                    + A3[k+4]*b4+A3[k+5]*b5+A3[k+6]*b6+A3[k+7]*b7;
            }
            for (; k < n; k++) {
                double b = Bj[k];
                s0+=A0[k]*b; s1+=A1[k]*b; s2+=A2[k]*b; s3+=A3[k]*b;
            }
            C[i*n+j]=s0; C[(i+1)*n+j]=s1; C[(i+2)*n+j]=s2; C[(i+3)*n+j]=s3;
        }
    }
    /* Remainder rows */
    for (; i < n; i++) {
        const double *Ai = A + i*n;
        for (int j = 0; j < n; j++) {
            const double *Bj = BT + j*n;
            double s = 0;
            for (int k = 0; k < n; k++) s += Ai[k]*Bj[k];
            C[i*n+j] = s;
        }
    }
}

/* ============================================================
 * Strassen helpers
 * ============================================================ */
static void add_f(const double *A, const double *B, double *C, int sz) {
    for (int i=0;i<sz;i++) C[i]=A[i]+B[i];
}
static void sub_f(const double *A, const double *B, double *C, int sz) {
    for (int i=0;i<sz;i++) C[i]=A[i]-B[i];
}
static void get_q(const double *s, double *d, int h, int r, int c, int n) {
    for (int i=0;i<h;i++) memcpy(d+i*h, s+(r+i)*n+c, h*8);
}
static void set_q(double *d, const double *s, int h, int r, int c, int n) {
    for (int i=0;i<h;i++) memcpy(d+(r+i)*n+c, s+i*h, h*8);
}

/* ============================================================
 * COM6-Strassen v5: pre-transpose + lean base case
 * BT quadrant mapping (BT = B^T):
 *   BT11 = B11^T, BT12 = B21^T, BT21 = B12^T, BT22 = B22^T
 * ============================================================ */
static void com6_str_inner(const double *A, const double *BT, double *C, int n, int thresh) {
    if (n <= thresh) {
        com6_lean(A, BT, C, n);
        return;
    }
    int h=n/2, sz=h*h;
    double *A11=malloc(sz*8),*A12=malloc(sz*8),*A21=malloc(sz*8),*A22=malloc(sz*8);
    double *BT11=malloc(sz*8),*BT12=malloc(sz*8),*BT21=malloc(sz*8),*BT22=malloc(sz*8);
    double *M1=malloc(sz*8),*M2=malloc(sz*8),*M3=malloc(sz*8),*M4=malloc(sz*8);
    double *M5=malloc(sz*8),*M6=malloc(sz*8),*M7=malloc(sz*8);
    double *TA=malloc(sz*8),*TB=malloc(sz*8);

    get_q(A,A11,h,0,0,n); get_q(A,A12,h,0,h,n);
    get_q(A,A21,h,h,0,n); get_q(A,A22,h,h,h,n);
    get_q(BT,BT11,h,0,0,n); get_q(BT,BT12,h,0,h,n);
    get_q(BT,BT21,h,h,0,n); get_q(BT,BT22,h,h,h,n);

    add_f(A11,A22,TA,sz); add_f(BT11,BT22,TB,sz); com6_str_inner(TA,TB,M1,h,thresh);
    add_f(A21,A22,TA,sz); com6_str_inner(TA,BT11,M2,h,thresh);
    sub_f(BT21,BT22,TB,sz); com6_str_inner(A11,TB,M3,h,thresh);
    sub_f(BT12,BT11,TB,sz); com6_str_inner(A22,TB,M4,h,thresh);
    add_f(A11,A12,TA,sz); com6_str_inner(TA,BT22,M5,h,thresh);
    sub_f(A21,A11,TA,sz); add_f(BT11,BT21,TB,sz); com6_str_inner(TA,TB,M6,h,thresh);
    sub_f(A12,A22,TA,sz); add_f(BT12,BT22,TB,sz); com6_str_inner(TA,TB,M7,h,thresh);

    add_f(M1,M4,TA,sz); sub_f(TA,M5,TB,sz); add_f(TB,M7,TA,sz); set_q(C,TA,h,0,0,n);
    add_f(M3,M5,TA,sz); set_q(C,TA,h,0,h,n);
    add_f(M2,M4,TA,sz); set_q(C,TA,h,h,0,n);
    sub_f(M1,M2,TA,sz); add_f(TA,M3,TB,sz); add_f(TB,M6,TA,sz); set_q(C,TA,h,h,h,n);

    free(A11);free(A12);free(A21);free(A22);
    free(BT11);free(BT12);free(BT21);free(BT22);
    free(M1);free(M2);free(M3);free(M4);free(M5);free(M6);free(M7);
    free(TA);free(TB);
}

void com6_v5(const double *A, const double *B, double *C, int n, int thresh) {
    double *BT = malloc(n*n*8);
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) BT[j*n+i]=B[i*n+j];
    com6_str_inner(A, BT, C, n, thresh);
    free(BT);
}


/* Pure Strassen with ikj base */
static void ikj_mm(const double *A, const double *B, double *C, int n) {
    memset(C,0,n*n*8);
    for (int i=0;i<n;i++) for (int k=0;k<n;k++) {
        double a=A[i*n+k]; for (int j=0;j<n;j++) C[i*n+j]+=a*B[k*n+j];
    }
}
void pure_strassen(const double *A, const double *B, double *C, int n, int thresh) {
    if (n<=thresh) { ikj_mm(A,B,C,n); return; }
    int h=n/2,sz=h*h;
    double *A11=malloc(sz*8),*A12=malloc(sz*8),*A21=malloc(sz*8),*A22=malloc(sz*8);
    double *B11=malloc(sz*8),*B12=malloc(sz*8),*B21=malloc(sz*8),*B22=malloc(sz*8);
    double *M1=malloc(sz*8),*M2=malloc(sz*8),*M3=malloc(sz*8),*M4=malloc(sz*8);
    double *M5=malloc(sz*8),*M6=malloc(sz*8),*M7=malloc(sz*8),*T1=malloc(sz*8),*T2=malloc(sz*8);
    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(B,B11,h,0,0,n);get_q(B,B12,h,0,h,n);get_q(B,B21,h,h,0,n);get_q(B,B22,h,h,h,n);
    add_f(A11,A22,T1,sz);add_f(B11,B22,T2,sz);pure_strassen(T1,T2,M1,h,thresh);
    add_f(A21,A22,T1,sz);pure_strassen(T1,B11,M2,h,thresh);
    sub_f(B12,B22,T1,sz);pure_strassen(A11,T1,M3,h,thresh);
    sub_f(B21,B11,T1,sz);pure_strassen(A22,T1,M4,h,thresh);
    add_f(A11,A12,T1,sz);pure_strassen(T1,B22,M5,h,thresh);
    sub_f(A21,A11,T1,sz);add_f(B11,B12,T2,sz);pure_strassen(T1,T2,M6,h,thresh);
    sub_f(A12,A22,T1,sz);add_f(B21,B22,T2,sz);pure_strassen(T1,T2,M7,h,thresh);
    add_f(M1,M4,T1,sz);sub_f(T1,M5,T2,sz);add_f(T2,M7,T1,sz);set_q(C,T1,h,0,0,n);
    add_f(M3,M5,T1,sz);set_q(C,T1,h,0,h,n);
    add_f(M2,M4,T1,sz);set_q(C,T1,h,h,0,n);
    sub_f(M1,M2,T1,sz);add_f(T1,M3,T2,sz);add_f(T2,M6,T1,sz);set_q(C,T1,h,h,h,n);
    free(A11);free(A12);free(A21);free(A22);free(B11);free(B12);free(B21);free(B22);
    free(M1);free(M2);free(M3);free(M4);free(M5);free(M6);free(M7);free(T1);free(T2);
}

/* Standard ijk */
void std_mm(const double *A, const double *B, double *C, int n) {
    memset(C,0,n*n*8);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) {
        double s=0; for(int k=0;k<n;k++) s+=A[i*n+k]*B[k*n+j]; C[i*n+j]=s;
    }
}


/* ============================================================ */
double get_ms() { struct timespec t; timespec_get(&t,TIME_UTC); return t.tv_sec*1e3+t.tv_nsec/1e6; }
void fill_rand(double *M, int n) { for(int i=0;i<n*n;i++) M[i]=(double)rand()/RAND_MAX*2-1; }
double mdiff(double *A, double *B, int n) { double m=0; for(int i=0;i<n*n;i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;} return m; }

int main() {
    printf("========================================================================\n");
    printf("  COM6 v5: Lean Hybrid (Pre-Transpose + Strassen + Micro-Kernel)\n");
    printf("========================================================================\n\n");

    /* Threshold sweep at 2048 */
    printf("--- Threshold sweep at 2048x2048 ---\n");
    {
        int n=2048;
        double *A=malloc(n*n*8),*B=malloc(n*n*8),*C=malloc(n*n*8);
        srand(42); fill_rand(A,n); fill_rand(B,n);

        int threshs[] = {64, 128, 256, 512};
        for (int ti=0; ti<4; ti++) {
            int th = threshs[ti];
            double t0=get_ms();
            com6_v5(A,B,C,n,th);
            double t1=get_ms()-t0;
            printf("  COM6-v5 thresh=%3d: %8.1f ms\n", th, t1);
        }
        for (int ti=0; ti<4; ti++) {
            int th = threshs[ti];
            double t0=get_ms();
            pure_strassen(A,B,C,n,th);
            double t1=get_ms()-t0;
            printf("  Strassen thresh=%3d: %8.1f ms\n", th, t1);
        }
        free(A);free(B);free(C);
    }

    /* Main benchmark with best thresholds */
    printf("\n--- Main Benchmark ---\n");
    printf("%-9s | %10s | %10s | %10s | %7s | %s\n",
           "Size","Strassen","COM6-v5","COM6-lean","v5/Str","Verify");
    printf("-------------------------------------------------------------------\n");

    int sizes[] = {256, 512, 1024, 2048, 4096};
    for (int si=0; si<5; si++) {
        int n=sizes[si];
        double *A=malloc(n*n*8),*B=malloc(n*n*8);
        double *C1=malloc(n*n*8),*C2=malloc(n*n*8),*C3=malloc(n*n*8);
        srand(42); fill_rand(A,n); fill_rand(B,n);
        int runs = n<=512?3:1;

        /* Pure Strassen, best threshold */
        double t0,t_str=0,t_v5=0,t_lean=0;
        if ((n&(n-1))==0) {
            t0=get_ms(); for(int r=0;r<runs;r++) pure_strassen(A,B,C1,n,128); t_str=(get_ms()-t0)/runs;
            t0=get_ms(); for(int r=0;r<runs;r++) com6_v5(A,B,C2,n,128); t_v5=(get_ms()-t0)/runs;
        }

        /* COM6 lean standalone (no Strassen, just direct) */
        {
            double *BT=malloc(n*n*8);
            for(int i=0;i<n;i++) for(int j=0;j<n;j++) BT[j*n+i]=B[i*n+j];
            t0=get_ms(); for(int r=0;r<runs;r++) com6_lean(A,BT,C3,n); t_lean=(get_ms()-t0)/runs;
            free(BT);
        }

        double diff = t_str>0 ? mdiff(C1,C2,n) : 0;
        char vfy[20]; snprintf(vfy,20,diff<1e-5?"OK":"e=%.0e",diff);

        char ss[20],sv[20],sl[20];
        if(t_str>0) snprintf(ss,20,"%8.1f ms",t_str); else snprintf(ss,20,"%10s","N/A");
        if(t_v5>0) snprintf(sv,20,"%8.1f ms",t_v5); else snprintf(sv,20,"%10s","N/A");
        snprintf(sl,20,"%8.1f ms",t_lean);

        printf("%4dx%-4d | %10s | %10s | %10s |",n,n,ss,sv,sl);
        if(t_str>0&&t_v5>0) printf(" %6.2fx |",t_str/t_v5); else printf(" %7s |","-");
        printf(" %s\n",vfy);

        free(A);free(B);free(C1);free(C2);free(C3);
    }

    printf("\nv5/Str > 1.0 = COM6 BEATS Strassen\n");
    return 0;
}
