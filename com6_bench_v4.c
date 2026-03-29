/*
 * COM6 v4 - Zero-overhead hybrid
 * ===============================
 * Key insight: transpose B ONCE at top level, then Strassen recursion
 * extracts sub-matrices from BOTH A and BT. Base case micro-kernel
 * gets pre-transposed data — zero transpose cost.
 *
 * gcc -O3 -march=native -funroll-loops -o com6v4 com6_bench_v4.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _MSC_VER
#define PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
#define PREFETCH(addr) __builtin_prefetch((addr), 0, 3)
#endif

#define COM_BLOCK 64
#define STRASSEN_THRESH 128

/* ============================================================
 * COM6 micro-kernel 4x4 (same as v2/v3)
 * ============================================================ */
static inline void micro_4x4(const double *Ai0, const double *Ai1,
                              const double *Ai2, const double *Ai3,
                              const double *BTj0, const double *BTj1,
                              const double *BTj2, const double *BTj3,
                              double *c, int cn, int klen) {
    double c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;
    double c20=0,c21=0,c22=0,c23=0, c30=0,c31=0,c32=0,c33=0;
    int k=0;
    for (; k+7<klen; k+=8) {
        PREFETCH(Ai0+k+8); PREFETCH(BTj0+k+8);
        double a00=Ai0[k],a01=Ai0[k+1],a02=Ai0[k+2],a03=Ai0[k+3];
        double a04=Ai0[k+4],a05=Ai0[k+5],a06=Ai0[k+6],a07=Ai0[k+7];
        double b00=BTj0[k],b01=BTj0[k+1],b02=BTj0[k+2],b03=BTj0[k+3];
        double b04=BTj0[k+4],b05=BTj0[k+5],b06=BTj0[k+6],b07=BTj0[k+7];
        double b10=BTj1[k],b11=BTj1[k+1],b12=BTj1[k+2],b13=BTj1[k+3];
        double b14=BTj1[k+4],b15=BTj1[k+5],b16=BTj1[k+6],b17=BTj1[k+7];
        double b20=BTj2[k],b21=BTj2[k+1],b22=BTj2[k+2],b23=BTj2[k+3];
        double b24=BTj2[k+4],b25=BTj2[k+5],b26=BTj2[k+6],b27=BTj2[k+7];
        double b30=BTj3[k],b31=BTj3[k+1],b32=BTj3[k+2],b33=BTj3[k+3];
        double b34=BTj3[k+4],b35=BTj3[k+5],b36=BTj3[k+6],b37=BTj3[k+7];
        c00+=a00*b00+a01*b01+a02*b02+a03*b03+a04*b04+a05*b05+a06*b06+a07*b07;
        c01+=a00*b10+a01*b11+a02*b12+a03*b13+a04*b14+a05*b15+a06*b16+a07*b17;
        c02+=a00*b20+a01*b21+a02*b22+a03*b23+a04*b24+a05*b25+a06*b26+a07*b27;
        c03+=a00*b30+a01*b31+a02*b32+a03*b33+a04*b34+a05*b35+a06*b36+a07*b37;
        PREFETCH(Ai1+k+8);
        double a10=Ai1[k],a11=Ai1[k+1],a12=Ai1[k+2],a13=Ai1[k+3];
        double a14=Ai1[k+4],a15=Ai1[k+5],a16=Ai1[k+6],a17=Ai1[k+7];
        c10+=a10*b00+a11*b01+a12*b02+a13*b03+a14*b04+a15*b05+a16*b06+a17*b07;
        c11+=a10*b10+a11*b11+a12*b12+a13*b13+a14*b14+a15*b15+a16*b16+a17*b17;
        c12+=a10*b20+a11*b21+a12*b22+a13*b23+a14*b24+a15*b25+a16*b26+a17*b27;
        c13+=a10*b30+a11*b31+a12*b32+a13*b33+a14*b34+a15*b35+a16*b36+a17*b37;
        PREFETCH(Ai2+k+8);
        double a20=Ai2[k],a21=Ai2[k+1],a22=Ai2[k+2],a23=Ai2[k+3];
        double a24=Ai2[k+4],a25=Ai2[k+5],a26=Ai2[k+6],a27=Ai2[k+7];
        c20+=a20*b00+a21*b01+a22*b02+a23*b03+a24*b04+a25*b05+a26*b06+a27*b07;
        c21+=a20*b10+a21*b11+a22*b12+a23*b13+a24*b14+a25*b15+a26*b16+a27*b17;
        c22+=a20*b20+a21*b21+a22*b22+a23*b23+a24*b24+a25*b25+a26*b26+a27*b27;
        c23+=a20*b30+a21*b31+a22*b32+a23*b33+a24*b34+a25*b35+a26*b36+a27*b37;
        PREFETCH(Ai3+k+8);
        double a30=Ai3[k],a31=Ai3[k+1],a32=Ai3[k+2],a33=Ai3[k+3];
        double a34=Ai3[k+4],a35=Ai3[k+5],a36=Ai3[k+6],a37=Ai3[k+7];
        c30+=a30*b00+a31*b01+a32*b02+a33*b03+a34*b04+a35*b05+a36*b06+a37*b07;
        c31+=a30*b10+a31*b11+a32*b12+a33*b13+a34*b14+a35*b15+a36*b16+a37*b17;
        c32+=a30*b20+a31*b21+a32*b22+a33*b23+a34*b24+a35*b25+a36*b26+a37*b27;
        c33+=a30*b30+a31*b31+a32*b32+a33*b33+a34*b34+a35*b35+a36*b36+a37*b37;
    }
    for (; k<klen; k++) {
        double a0=Ai0[k],a1=Ai1[k],a2=Ai2[k],a3=Ai3[k];
        double b0=BTj0[k],b1=BTj1[k],b2=BTj2[k],b3=BTj3[k];
        c00+=a0*b0;c01+=a0*b1;c02+=a0*b2;c03+=a0*b3;
        c10+=a1*b0;c11+=a1*b1;c12+=a1*b2;c13+=a1*b3;
        c20+=a2*b0;c21+=a2*b1;c22+=a2*b2;c23+=a2*b3;
        c30+=a3*b0;c31+=a3*b1;c32+=a3*b2;c33+=a3*b3;
    }
    c[0*cn]+=c00;c[1]+=c01;c[2]+=c02;c[3]+=c03;
    c[cn]+=c10;c[cn+1]+=c11;c[cn+2]+=c12;c[cn+3]+=c13;
    c[2*cn]+=c20;c[2*cn+1]+=c21;c[2*cn+2]+=c22;c[2*cn+3]+=c23;
    c[3*cn]+=c30;c[3*cn+1]+=c31;c[3*cn+2]+=c32;c[3*cn+3]+=c33;
}

/* ============================================================
 * COM6 base case: A is normal layout, BT is already transposed
 * NO TRANSPOSE NEEDED — data arrives pre-transposed
 * ============================================================ */
static void com6_base_pretransposed(const double *A, const double *BT,
                                     double *C, int n, int lda, int ldbt, int ldc) {
    /* Zero output */
    for (int i = 0; i < n; i++)
        memset(C + i * ldc, 0, n * sizeof(double));

    for (int i0 = 0; i0 < n; i0 += COM_BLOCK) {
        int i1 = i0+COM_BLOCK < n ? i0+COM_BLOCK : n;
        for (int j0 = 0; j0 < n; j0 += COM_BLOCK) {
            int j1 = j0+COM_BLOCK < n ? j0+COM_BLOCK : n;
            for (int k0 = 0; k0 < n; k0 += COM_BLOCK) {
                int k1 = k0+COM_BLOCK < n ? k0+COM_BLOCK : n;
                int klen = k1 - k0;
                int i;
                for (i = i0; i+3 < i1; i += 4) {
                    int j;
                    for (j = j0; j+3 < j1; j += 4) {
                        micro_4x4(A+i*lda+k0, A+(i+1)*lda+k0,
                                  A+(i+2)*lda+k0, A+(i+3)*lda+k0,
                                  BT+j*ldbt+k0, BT+(j+1)*ldbt+k0,
                                  BT+(j+2)*ldbt+k0, BT+(j+3)*ldbt+k0,
                                  C+i*ldc+j, ldc, klen);
                    }
                    for (; j < j1; j++) {
                        double s0=0,s1=0,s2=0,s3=0;
                        for (int k=0;k<klen;k++) {
                            double b=BT[j*ldbt+k0+k];
                            s0+=A[i*lda+k0+k]*b; s1+=A[(i+1)*lda+k0+k]*b;
                            s2+=A[(i+2)*lda+k0+k]*b; s3+=A[(i+3)*lda+k0+k]*b;
                        }
                        C[i*ldc+j]+=s0; C[(i+1)*ldc+j]+=s1;
                        C[(i+2)*ldc+j]+=s2; C[(i+3)*ldc+j]+=s3;
                    }
                }
                for (; i < i1; i++)
                    for (int j=j0; j<j1; j++) {
                        double s=0;
                        for (int k=0;k<klen;k++) s+=A[i*lda+k0+k]*BT[j*ldbt+k0+k];
                        C[i*ldc+j]+=s;
                    }
            }
        }
    }
}

/* ============================================================
 * Strassen helpers — contiguous sub-matrices
 * ============================================================ */
static void add_flat(const double *A, const double *B, double *C, int sz) {
    for (int i=0;i<sz;i++) C[i]=A[i]+B[i];
}
static void sub_flat(const double *A, const double *B, double *C, int sz) {
    for (int i=0;i<sz;i++) C[i]=A[i]-B[i];
}
static void get_quad(const double *src, double *dst, int h, int r, int c, int n) {
    for (int i=0;i<h;i++) memcpy(dst+i*h, src+(r+i)*n+c, h*sizeof(double));
}
static void set_quad(double *dst, const double *src, int h, int r, int c, int n) {
    for (int i=0;i<h;i++) memcpy(dst+(r+i)*n+c, src+i*h, h*sizeof(double));
}

/* Transpose a contiguous hxh matrix */
static void transpose_flat(const double *src, double *dst, int h) {
    for (int i=0;i<h;i++)
        for (int j=0;j<h;j++)
            dst[j*h+i] = src[i*h+j];
}

/* ============================================================
 * COM6-Strassen v4: pre-transpose aware recursion
 *
 * We pass BOTH A (normal) and BT (transposed) through recursion.
 * At each level, Strassen computes sums/diffs of A quadrants normally,
 * and sums/diffs of BT quadrants (which correspond to B column blocks).
 *
 * BT quadrant mapping:
 *   B = [B11 B12; B21 B22]  =>  BT = [B11^T B21^T; B12^T B22^T]
 *   BT11 = B11^T, BT12 = B21^T, BT21 = B12^T, BT22 = B22^T
 *
 * So Strassen's M1 = (A11+A22) * (B11+B22)
 *   = (A11+A22) * (BT11^T + BT22^T)
 *   With T1 = A11+A22 (normal), T2 = BT11+BT22 (already transposed!)
 *   base_case(T1, T2, M1) where T2 is the transposed form
 * ============================================================ */
void com6_strassen_v4(const double *A, const double *BT, double *C, int n) {
    if (n <= STRASSEN_THRESH) {
        com6_base_pretransposed(A, BT, C, n, n, n, n);
        return;
    }

    int h = n/2, sz = h*h;

    /* Extract A quadrants (normal layout) */
    double *A11=malloc(sz*8),*A12=malloc(sz*8),*A21=malloc(sz*8),*A22=malloc(sz*8);
    get_quad(A,A11,h,0,0,n); get_quad(A,A12,h,0,h,n);
    get_quad(A,A21,h,h,0,n); get_quad(A,A22,h,h,h,n);

    /* Extract BT quadrants
     * BT is transposed B, so BT[j*n+k] = B[k*n+j]
     * BT top-left hxh block (rows 0..h-1) corresponds to B columns 0..h-1 = B11^T, B21^T
     * BT11 (BT rows 0..h-1, cols 0..h-1) = B[0..h-1, 0..h-1]^T = B11^T
     * BT12 (BT rows 0..h-1, cols h..n-1) = B[h..n-1, 0..h-1]^T = B21^T
     * BT21 (BT rows h..n-1, cols 0..h-1) = B[0..h-1, h..n-1]^T = B12^T
     * BT22 (BT rows h..n-1, cols h..n-1) = B[h..n-1, h..n-1]^T = B22^T
     */
    double *BT11=malloc(sz*8),*BT12=malloc(sz*8),*BT21=malloc(sz*8),*BT22=malloc(sz*8);
    get_quad(BT,BT11,h,0,0,n); get_quad(BT,BT12,h,0,h,n);
    get_quad(BT,BT21,h,h,0,n); get_quad(BT,BT22,h,h,h,n);

    /* BT11=B11^T, BT12=B21^T, BT21=B12^T, BT22=B22^T
     * Strassen needs:
     *   M1 = (A11+A22)*(B11+B22) => T_A=A11+A22, T_BT=BT11+BT22 (=B11^T+B22^T=(B11+B22)^T)
     *   M2 = (A21+A22)*B11       => T_A=A21+A22, T_BT=BT11 (=B11^T)
     *   M3 = A11*(B12-B22)       => T_A=A11, T_BT=BT21-BT22 (=B12^T-B22^T=(B12-B22)^T)
     *   M4 = A22*(B21-B11)       => T_A=A22, T_BT=BT12-BT11 (=B21^T-B11^T=(B21-B11)^T)
     *   M5 = (A11+A12)*B22       => T_A=A11+A12, T_BT=BT22 (=B22^T)
     *   M6 = (A21-A11)*(B11+B12) => T_A=A21-A11, T_BT=BT11+BT21 (=B11^T+B12^T=(B11+B12)^T)
     *   M7 = (A12-A22)*(B21+B22) => T_A=A12-A22, T_BT=BT12+BT22 (=B21^T+B22^T=(B21+B22)^T)
     */
    double *M1=malloc(sz*8),*M2=malloc(sz*8),*M3=malloc(sz*8),*M4=malloc(sz*8);
    double *M5=malloc(sz*8),*M6=malloc(sz*8),*M7=malloc(sz*8);
    double *TA=malloc(sz*8),*TBT=malloc(sz*8);

    /* M1 = (A11+A22)*(B11+B22) */
    add_flat(A11,A22,TA,sz); add_flat(BT11,BT22,TBT,sz);
    com6_strassen_v4(TA,TBT,M1,h);

    /* M2 = (A21+A22)*B11 */
    add_flat(A21,A22,TA,sz);
    com6_strassen_v4(TA,BT11,M2,h);

    /* M3 = A11*(B12-B22) => BT: BT21-BT22 */
    sub_flat(BT21,BT22,TBT,sz);
    com6_strassen_v4(A11,TBT,M3,h);

    /* M4 = A22*(B21-B11) => BT: BT12-BT11 */
    sub_flat(BT12,BT11,TBT,sz);
    com6_strassen_v4(A22,TBT,M4,h);

    /* M5 = (A11+A12)*B22 => BT: BT22 */
    add_flat(A11,A12,TA,sz);
    com6_strassen_v4(TA,BT22,M5,h);

    /* M6 = (A21-A11)*(B11+B12) => BT: BT11+BT21 */
    sub_flat(A21,A11,TA,sz); add_flat(BT11,BT21,TBT,sz);
    com6_strassen_v4(TA,TBT,M6,h);

    /* M7 = (A12-A22)*(B21+B22) => BT: BT12+BT22 */
    sub_flat(A12,A22,TA,sz); add_flat(BT12,BT22,TBT,sz);
    com6_strassen_v4(TA,TBT,M7,h);

    /* Assemble C */
    add_flat(M1,M4,TA,sz); sub_flat(TA,M5,TBT,sz); add_flat(TBT,M7,TA,sz); set_quad(C,TA,h,0,0,n);
    add_flat(M3,M5,TA,sz); set_quad(C,TA,h,0,h,n);
    add_flat(M2,M4,TA,sz); set_quad(C,TA,h,h,0,n);
    sub_flat(M1,M2,TA,sz); add_flat(TA,M3,TBT,sz); add_flat(TBT,M6,TA,sz); set_quad(C,TA,h,h,h,n);

    free(A11);free(A12);free(A21);free(A22);
    free(BT11);free(BT12);free(BT21);free(BT22);
    free(M1);free(M2);free(M3);free(M4);free(M5);free(M6);free(M7);
    free(TA);free(TBT);
}

/* Top-level: transpose B once, then recurse */
void com6_v4(const double *A, const double *B, double *C, int n) {
    double *BT = malloc(n*n*sizeof(double));
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
            BT[j*n+i] = B[i*n+j];
    com6_strassen_v4(A, BT, C, n);
    free(BT);
}


/* ============================================================
 * Pure Strassen (ikj base) for comparison
 * ============================================================ */
static void ikj_matmul(const double *A, const double *B, double *C, int n) {
    memset(C,0,n*n*sizeof(double));
    for (int i=0;i<n;i++) for (int k=0;k<n;k++) {
        double a=A[i*n+k];
        for (int j=0;j<n;j++) C[i*n+j]+=a*B[k*n+j];
    }
}
void pure_strassen(const double *A, const double *B, double *C, int n) {
    if (n<=STRASSEN_THRESH) { ikj_matmul(A,B,C,n); return; }
    int h=n/2, sz=h*h;
    double *A11=malloc(sz*8),*A12=malloc(sz*8),*A21=malloc(sz*8),*A22=malloc(sz*8);
    double *B11=malloc(sz*8),*B12=malloc(sz*8),*B21=malloc(sz*8),*B22=malloc(sz*8);
    double *M1=malloc(sz*8),*M2=malloc(sz*8),*M3=malloc(sz*8),*M4=malloc(sz*8);
    double *M5=malloc(sz*8),*M6=malloc(sz*8),*M7=malloc(sz*8),*T1=malloc(sz*8),*T2=malloc(sz*8);
    get_quad(A,A11,h,0,0,n);get_quad(A,A12,h,0,h,n);get_quad(A,A21,h,h,0,n);get_quad(A,A22,h,h,h,n);
    get_quad(B,B11,h,0,0,n);get_quad(B,B12,h,0,h,n);get_quad(B,B21,h,h,0,n);get_quad(B,B22,h,h,h,n);
    add_flat(A11,A22,T1,sz);add_flat(B11,B22,T2,sz);pure_strassen(T1,T2,M1,h);
    add_flat(A21,A22,T1,sz);pure_strassen(T1,B11,M2,h);
    sub_flat(B12,B22,T1,sz);pure_strassen(A11,T1,M3,h);
    sub_flat(B21,B11,T1,sz);pure_strassen(A22,T1,M4,h);
    add_flat(A11,A12,T1,sz);pure_strassen(T1,B22,M5,h);
    sub_flat(A21,A11,T1,sz);add_flat(B11,B12,T2,sz);pure_strassen(T1,T2,M6,h);
    sub_flat(A12,A22,T1,sz);add_flat(B21,B22,T2,sz);pure_strassen(T1,T2,M7,h);
    add_flat(M1,M4,T1,sz);sub_flat(T1,M5,T2,sz);add_flat(T2,M7,T1,sz);set_quad(C,T1,h,0,0,n);
    add_flat(M3,M5,T1,sz);set_quad(C,T1,h,0,h,n);
    add_flat(M2,M4,T1,sz);set_quad(C,T1,h,h,0,n);
    sub_flat(M1,M2,T1,sz);add_flat(T1,M3,T2,sz);add_flat(T2,M6,T1,sz);set_quad(C,T1,h,h,h,n);
    free(A11);free(A12);free(A21);free(A22);free(B11);free(B12);free(B21);free(B22);
    free(M1);free(M2);free(M3);free(M4);free(M5);free(M6);free(M7);free(T1);free(T2);
}

/* Standard naive */
void standard_matmul(const double *A, const double *B, double *C, int n) {
    memset(C,0,n*n*sizeof(double));
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) {
        double s=0; for (int k=0;k<n;k++) s+=A[i*n+k]*B[k*n+j]; C[i*n+j]=s;
    }
}

/* COM6 v2 standalone (for reference) */
void com6_v2_standalone(const double *A, const double *B, double *C, int n) {
    memset(C,0,n*n*sizeof(double));
    double *BT=malloc(n*n*sizeof(double));
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) BT[j*n+i]=B[i*n+j];
    for (int i0=0;i0<n;i0+=COM_BLOCK) {
        int i1=i0+COM_BLOCK<n?i0+COM_BLOCK:n;
        for (int j0=0;j0<n;j0+=COM_BLOCK) {
            int j1=j0+COM_BLOCK<n?j0+COM_BLOCK:n;
            for (int k0=0;k0<n;k0+=COM_BLOCK) {
                int k1=k0+COM_BLOCK<n?k0+COM_BLOCK:n;
                int klen=k1-k0, i;
                for (i=i0;i+3<i1;i+=4) {
                    int j;
                    for (j=j0;j+3<j1;j+=4)
                        micro_4x4(A+i*n+k0,A+(i+1)*n+k0,A+(i+2)*n+k0,A+(i+3)*n+k0,
                                  BT+j*n+k0,BT+(j+1)*n+k0,BT+(j+2)*n+k0,BT+(j+3)*n+k0,
                                  C+i*n+j,n,klen);
                    for (;j<j1;j++) {
                        double s0=0,s1=0,s2=0,s3=0;
                        for (int k=0;k<klen;k++) {
                            double b=BT[j*n+k0+k];
                            s0+=A[i*n+k0+k]*b;s1+=A[(i+1)*n+k0+k]*b;
                            s2+=A[(i+2)*n+k0+k]*b;s3+=A[(i+3)*n+k0+k]*b;
                        }
                        C[i*n+j]+=s0;C[(i+1)*n+j]+=s1;C[(i+2)*n+j]+=s2;C[(i+3)*n+j]+=s3;
                    }
                }
                for (;i<i1;i++) for (int j=j0;j<j1;j++) {
                    double s=0; for (int k=0;k<klen;k++) s+=A[i*n+k0+k]*BT[j*n+k0+k];
                    C[i*n+j]+=s;
                }
            }
        }
    }
    free(BT);
}


/* ============================================================ */
double get_time_ms() { struct timespec ts; timespec_get(&ts,TIME_UTC); return ts.tv_sec*1000.0+ts.tv_nsec/1e6; }
void fill_random(double *M, int n) { for (int i=0;i<n*n;i++) M[i]=(double)rand()/RAND_MAX*2.0-1.0; }
double max_diff(double *A, double *B, int n) { double mx=0; for(int i=0;i<n*n;i++){double d=fabs(A[i]-B[i]);if(d>mx)mx=d;} return mx; }

int main() {
    printf("========================================================================\n");
    printf("  COM6 v4: Pre-Transpose Strassen + COM6 Micro-Kernel\n");
    printf("  B transposed ONCE at top, zero transpose cost in recursion\n");
    printf("  Threshold: %d, Block: %d, Micro: 4x4, 8-wide\n", STRASSEN_THRESH, COM_BLOCK);
    printf("========================================================================\n\n");

    int sizes[] = {256, 512, 1024, 2048, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);

    printf("%-9s | %10s | %10s | %10s | %10s | %7s | %7s | %s\n",
           "Size","Standard","Strassen","COM6-solo","COM6-v4","v4/Std","v4/Str","Verify");
    printf("--------------------------------------------------------------------------\n");

    for (int si=0; si<ns; si++) {
        int n = sizes[si];
        double *A=malloc(n*n*8),*B=malloc(n*n*8);
        double *C_std=malloc(n*n*8),*C_str=malloc(n*n*8),*C_solo=malloc(n*n*8),*C_v4=malloc(n*n*8);
        srand(42); fill_random(A,n); fill_random(B,n);
        int runs = n<=512?3:1;
        double t_std=0,t_str=0,t_solo=0,t_v4=0;

        if (n<=2048) { double t0=get_time_ms(); for(int r=0;r<runs;r++) standard_matmul(A,B,C_std,n); t_std=(get_time_ms()-t0)/runs; }
        if ((n&(n-1))==0) { double t0=get_time_ms(); for(int r=0;r<runs;r++) pure_strassen(A,B,C_str,n); t_str=(get_time_ms()-t0)/runs; }
        { double t0=get_time_ms(); for(int r=0;r<runs;r++) com6_v2_standalone(A,B,C_solo,n); t_solo=(get_time_ms()-t0)/runs; }
        if ((n&(n-1))==0) { double t0=get_time_ms(); for(int r=0;r<runs;r++) com6_v4(A,B,C_v4,n); t_v4=(get_time_ms()-t0)/runs; }

        double diff = t_std>0&&t_v4>0 ? max_diff(C_std,C_v4,n) : (t_solo>0&&t_v4>0 ? max_diff(C_solo,C_v4,n) : 0);
        char vfy[20]; snprintf(vfy,20,diff<1e-5?"OK":"e=%.0e",diff);

        char ss[20],sr[20],sl[20],sv[20];
        if(t_std>0) snprintf(ss,20,"%8.1f ms",t_std); else snprintf(ss,20,"%10s","SKIP");
        if(t_str>0) snprintf(sr,20,"%8.1f ms",t_str); else snprintf(sr,20,"%10s","N/A");
        snprintf(sl,20,"%8.1f ms",t_solo);
        if(t_v4>0) snprintf(sv,20,"%8.1f ms",t_v4); else snprintf(sv,20,"%10s","N/A");

        printf("%4dx%-4d | %10s | %10s | %10s | %10s |",n,n,ss,sr,sl,sv);
        if(t_std>0&&t_v4>0) printf(" %6.1fx |",t_std/t_v4); else printf(" %7s |","-");
        if(t_str>0&&t_v4>0) printf(" %6.1fx |",t_str/t_v4); else printf(" %7s |","-");
        printf(" %s\n",vfy);

        free(A);free(B);free(C_std);free(C_str);free(C_solo);free(C_v4);
    }

    printf("\nv4/Str > 1.0 = COM6 BEATS Strassen\n");
    return 0;
}
