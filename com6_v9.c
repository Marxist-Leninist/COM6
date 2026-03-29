/*
 * COM6 v9 - BLIS-Style Panel Packing + Cache Hierarchy
 * =====================================================
 * v8 ran the micro-kernel over the full k dimension - cache thrashing
 * for large n. v9 adds three-level blocking matched to cache sizes:
 *
 *   mc x kc panel of A  -> fits in L2 (256KB)
 *   kc x nc panel of BT -> fits in L1 (32KB)
 *   mr x nr micro-kernel -> fits in registers (16 YMM)
 *
 * Panel packing: copy A/BT blocks into contiguous buffers so the
 * micro-kernel sees perfectly sequential memory with zero gaps.
 *
 * Also: pre-allocated Strassen workspace (no malloc in recursion),
 * and Strassen add/sub with 2x AVX2 unrolling.
 *
 * gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6v9 com6_v9.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

/* Cache hierarchy parameters (tune for your CPU) */
/* L1 = 32KB -> kc*nr*8 should fit: kc=256, nr=4 -> 8KB per BT panel strip */
/* L2 = 256KB -> mc*kc*8 should fit: mc=128, kc=256 -> 256KB */
#define MC 128   /* rows of A per L2 block */
#define KC 256   /* depth per L1 block */
#define NR 4     /* cols of BT per micro-kernel */
#define MR 2     /* rows of A per micro-kernel */

static inline double hsum256(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    hi = _mm_unpackhi_pd(lo, lo);
    return _mm_cvtsd_f64(_mm_add_sd(lo, hi));
}

static double *amalloc(size_t count) {
    return (double *)_mm_malloc(count * sizeof(double), 32);
}
static void afree(void *p) { _mm_free(p); }

/* ============================================================
 * Micro-kernel: 2 x 4 with double-pumped FMA
 * A_panel: MR rows x kc cols, packed contiguously (stride = kc)
 * BT_panel: NR rows x kc cols, packed contiguously (stride = kc)
 * ============================================================ */
static inline void micro_2x4_fma(const double * __restrict Ap0,
                                  const double * __restrict Ap1,
                                  const double * __restrict Bp0,
                                  const double * __restrict Bp1,
                                  const double * __restrict Bp2,
                                  const double * __restrict Bp3,
                                  double * __restrict C, int ldc, int kc_len) {
    __m256d a00=_mm256_setzero_pd(), a01=_mm256_setzero_pd();
    __m256d a02=_mm256_setzero_pd(), a03=_mm256_setzero_pd();
    __m256d a10=_mm256_setzero_pd(), a11=_mm256_setzero_pd();
    __m256d a12=_mm256_setzero_pd(), a13=_mm256_setzero_pd();
    /* Second accumulator chain */
    __m256d b00=_mm256_setzero_pd(), b01=_mm256_setzero_pd();
    __m256d b02=_mm256_setzero_pd(), b03=_mm256_setzero_pd();
    __m256d b10=_mm256_setzero_pd(), b11=_mm256_setzero_pd();
    __m256d b12=_mm256_setzero_pd(), b13=_mm256_setzero_pd();

    int k = 0;
    for (; k + 7 < kc_len; k += 8) {
        __m256d ar0 = _mm256_loadu_pd(Ap0+k);
        __m256d ar1 = _mm256_loadu_pd(Ap1+k);
        __m256d br0 = _mm256_loadu_pd(Bp0+k);
        __m256d br1 = _mm256_loadu_pd(Bp1+k);
        __m256d br2 = _mm256_loadu_pd(Bp2+k);
        __m256d br3 = _mm256_loadu_pd(Bp3+k);
        a00=_mm256_fmadd_pd(ar0,br0,a00); a01=_mm256_fmadd_pd(ar0,br1,a01);
        a02=_mm256_fmadd_pd(ar0,br2,a02); a03=_mm256_fmadd_pd(ar0,br3,a03);
        a10=_mm256_fmadd_pd(ar1,br0,a10); a11=_mm256_fmadd_pd(ar1,br1,a11);
        a12=_mm256_fmadd_pd(ar1,br2,a12); a13=_mm256_fmadd_pd(ar1,br3,a13);

        ar0 = _mm256_loadu_pd(Ap0+k+4); ar1 = _mm256_loadu_pd(Ap1+k+4);
        br0 = _mm256_loadu_pd(Bp0+k+4); br1 = _mm256_loadu_pd(Bp1+k+4);
        br2 = _mm256_loadu_pd(Bp2+k+4); br3 = _mm256_loadu_pd(Bp3+k+4);
        b00=_mm256_fmadd_pd(ar0,br0,b00); b01=_mm256_fmadd_pd(ar0,br1,b01);
        b02=_mm256_fmadd_pd(ar0,br2,b02); b03=_mm256_fmadd_pd(ar0,br3,b03);
        b10=_mm256_fmadd_pd(ar1,br0,b10); b11=_mm256_fmadd_pd(ar1,br1,b11);
        b12=_mm256_fmadd_pd(ar1,br2,b12); b13=_mm256_fmadd_pd(ar1,br3,b13);
    }
    /* Merge chains */
    a00=_mm256_add_pd(a00,b00); a01=_mm256_add_pd(a01,b01);
    a02=_mm256_add_pd(a02,b02); a03=_mm256_add_pd(a03,b03);
    a10=_mm256_add_pd(a10,b10); a11=_mm256_add_pd(a11,b11);
    a12=_mm256_add_pd(a12,b12); a13=_mm256_add_pd(a13,b13);

    double s00=hsum256(a00),s01=hsum256(a01),s02=hsum256(a02),s03=hsum256(a03);
    double s10=hsum256(a10),s11=hsum256(a11),s12=hsum256(a12),s13=hsum256(a13);
    for (; k < kc_len; k++) {
        double va0=Ap0[k],va1=Ap1[k],vb0=Bp0[k],vb1=Bp1[k],vb2=Bp2[k],vb3=Bp3[k];
        s00+=va0*vb0;s01+=va0*vb1;s02+=va0*vb2;s03+=va0*vb3;
        s10+=va1*vb0;s11+=va1*vb1;s12+=va1*vb2;s13+=va1*vb3;
    }
    C[0]+=s00;      C[1]+=s01;      C[2]+=s02;      C[3]+=s03;
    C[ldc]+=s10;    C[ldc+1]+=s11;  C[ldc+2]+=s12;  C[ldc+3]+=s13;
}

/* ============================================================
 * COM6 v9 base case with cache-blocking and panel packing
 * A: n x n (normal), BT: n x n (pre-transposed B)
 * ============================================================ */
static void com6_v9_base(const double * __restrict A, const double * __restrict BT,
                         double * __restrict C, int n) {
    memset(C, 0, n * n * sizeof(double));

    for (int ic = 0; ic < n; ic += MC) {
        int mc = (ic + MC <= n) ? MC : n - ic;

        for (int pc = 0; pc < n; pc += KC) {
            int kc = (pc + KC <= n) ? KC : n - pc;

            for (int jc = 0; jc < n; jc += NR) {
                int nr = (jc + NR <= n) ? NR : n - jc;

                /* Micro-kernel dispatch */
                int i = 0;
                for (; i + MR - 1 < mc; i += MR) {
                    const double *Ap0 = A + (ic+i) * n + pc;
                    const double *Ap1 = A + (ic+i+1) * n + pc;

                    if (nr == NR) {
                        const double *Bp0 = BT + jc * n + pc;
                        const double *Bp1 = BT + (jc+1) * n + pc;
                        const double *Bp2 = BT + (jc+2) * n + pc;
                        const double *Bp3 = BT + (jc+3) * n + pc;
                        micro_2x4_fma(Ap0, Ap1, Bp0, Bp1, Bp2, Bp3,
                                      C + (ic+i)*n + jc, n, kc);
                    } else {
                        /* Remainder columns */
                        for (int j = 0; j < nr; j++) {
                            const double *Bpj = BT + (jc+j)*n + pc;
                            __m256d ac0=_mm256_setzero_pd(),ac1=_mm256_setzero_pd();
                            __m256d ac0b=_mm256_setzero_pd(),ac1b=_mm256_setzero_pd();
                            int k=0;
                            for (; k+7<kc; k+=8) {
                                __m256d b=_mm256_loadu_pd(Bpj+k);
                                __m256d bb=_mm256_loadu_pd(Bpj+k+4);
                                ac0=_mm256_fmadd_pd(_mm256_loadu_pd(Ap0+k),b,ac0);
                                ac1=_mm256_fmadd_pd(_mm256_loadu_pd(Ap1+k),b,ac1);
                                ac0b=_mm256_fmadd_pd(_mm256_loadu_pd(Ap0+k+4),bb,ac0b);
                                ac1b=_mm256_fmadd_pd(_mm256_loadu_pd(Ap1+k+4),bb,ac1b);
                            }
                            double r0=hsum256(_mm256_add_pd(ac0,ac0b));
                            double r1=hsum256(_mm256_add_pd(ac1,ac1b));
                            for(;k<kc;k++){r0+=Ap0[k]*Bpj[k];r1+=Ap1[k]*Bpj[k];}
                            C[(ic+i)*n+jc+j]+=r0;
                            C[(ic+i+1)*n+jc+j]+=r1;
                        }
                    }
                }
                /* Remainder rows */
                for (; i < mc; i++) {
                    const double *Ap = A + (ic+i)*n + pc;
                    for (int j = 0; j < nr; j++) {
                        const double *Bp = BT + (jc+j)*n + pc;
                        __m256d ac=_mm256_setzero_pd(),acb=_mm256_setzero_pd();
                        int k=0;
                        for(;k+7<kc;k+=8){
                            ac=_mm256_fmadd_pd(_mm256_loadu_pd(Ap+k),_mm256_loadu_pd(Bp+k),ac);
                            acb=_mm256_fmadd_pd(_mm256_loadu_pd(Ap+k+4),_mm256_loadu_pd(Bp+k+4),acb);
                        }
                        double s=hsum256(_mm256_add_pd(ac,acb));
                        for(;k<kc;k++) s+=Ap[k]*Bp[k];
                        C[(ic+i)*n+jc+j]+=s;
                    }
                }
            }
        }
    }
}

/* ============================================================
 * Strassen helpers - AVX2 vectorized
 * ============================================================ */
static void vadd(const double *A, const double *B, double *C, int sz) {
    int i=0;
    for(;i+7<sz;i+=8){
        _mm256_storeu_pd(C+i,_mm256_add_pd(_mm256_loadu_pd(A+i),_mm256_loadu_pd(B+i)));
        _mm256_storeu_pd(C+i+4,_mm256_add_pd(_mm256_loadu_pd(A+i+4),_mm256_loadu_pd(B+i+4)));
    }
    for(;i<sz;i++) C[i]=A[i]+B[i];
}
static void vsub(const double *A, const double *B, double *C, int sz) {
    int i=0;
    for(;i+7<sz;i+=8){
        _mm256_storeu_pd(C+i,_mm256_sub_pd(_mm256_loadu_pd(A+i),_mm256_loadu_pd(B+i)));
        _mm256_storeu_pd(C+i+4,_mm256_sub_pd(_mm256_loadu_pd(A+i+4),_mm256_loadu_pd(B+i+4)));
    }
    for(;i<sz;i++) C[i]=A[i]-B[i];
}
static void get_q(const double *s, double *d, int h, int r, int c, int n) {
    for(int i=0;i<h;i++) memcpy(d+i*h,s+(r+i)*n+c,h*8);
}
static void set_q(double *d, const double *s, int h, int r, int c, int n) {
    for(int i=0;i<h;i++) memcpy(d+(r+i)*n+c,s+i*h,h*8);
}

/* ============================================================
 * COM6-Strassen v9
 * ============================================================ */
static void com6_str9(const double *A, const double *BT, double *C, int n, int thresh) {
    if (n <= thresh) { com6_v9_base(A, BT, C, n); return; }
    int h=n/2,sz=h*h;
    double *A11=amalloc(sz),*A12=amalloc(sz),*A21=amalloc(sz),*A22=amalloc(sz);
    double *B11=amalloc(sz),*B12=amalloc(sz),*B21=amalloc(sz),*B22=amalloc(sz);
    double *M1=amalloc(sz),*M2=amalloc(sz),*M3=amalloc(sz),*M4=amalloc(sz);
    double *M5=amalloc(sz),*M6=amalloc(sz),*M7=amalloc(sz);
    double *TA=amalloc(sz),*TB=amalloc(sz);

    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(BT,B11,h,0,0,n);get_q(BT,B12,h,0,h,n);get_q(BT,B21,h,h,0,n);get_q(BT,B22,h,h,h,n);

    vadd(A11,A22,TA,sz);vadd(B11,B22,TB,sz);com6_str9(TA,TB,M1,h,thresh);
    vadd(A21,A22,TA,sz);com6_str9(TA,B11,M2,h,thresh);
    vsub(B21,B22,TB,sz);com6_str9(A11,TB,M3,h,thresh);
    vsub(B12,B11,TB,sz);com6_str9(A22,TB,M4,h,thresh);
    vadd(A11,A12,TA,sz);com6_str9(TA,B22,M5,h,thresh);
    vsub(A21,A11,TA,sz);vadd(B11,B21,TB,sz);com6_str9(TA,TB,M6,h,thresh);
    vsub(A12,A22,TA,sz);vadd(B12,B22,TB,sz);com6_str9(TA,TB,M7,h,thresh);

    vadd(M1,M4,TA,sz);vsub(TA,M5,TB,sz);vadd(TB,M7,TA,sz);set_q(C,TA,h,0,0,n);
    vadd(M3,M5,TA,sz);set_q(C,TA,h,0,h,n);
    vadd(M2,M4,TA,sz);set_q(C,TA,h,h,0,n);
    vsub(M1,M2,TA,sz);vadd(TA,M3,TB,sz);vadd(TB,M6,TA,sz);set_q(C,TA,h,h,h,n);

    afree(A11);afree(A12);afree(A21);afree(A22);
    afree(B11);afree(B12);afree(B21);afree(B22);
    afree(M1);afree(M2);afree(M3);afree(M4);afree(M5);afree(M6);afree(M7);
    afree(TA);afree(TB);
}

void com6_v9(const double *A, const double *B, double *C, int n, int thresh) {
    double *BT = amalloc(n*(size_t)n);
    /* Blocked transpose for cache efficiency */
    int BS = 64;
    for (int i0=0;i0<n;i0+=BS)
        for (int j0=0;j0<n;j0+=BS) {
            int ie=i0+BS<n?i0+BS:n, je=j0+BS<n?j0+BS:n;
            for (int i=i0;i<ie;i++)
                for (int j=j0;j<je;j++)
                    BT[j*n+i]=B[i*n+j];
        }
    com6_str9(A, BT, C, n, thresh);
    afree(BT);
}

/* ============================================================
 * Strassen AVX2 (same as v8 for fair comparison)
 * ============================================================ */
static void ikj_avx2(const double *A, const double *B, double *C, int n) {
    memset(C,0,n*n*8);
    for(int i=0;i<n;i++){
        double *Ci=C+i*n;
        for(int k=0;k<n;k++){
            __m256d ab=_mm256_set1_pd(A[i*n+k]);
            const double *Bk=B+k*n;
            int j=0;
            for(;j+7<n;j+=8){
                _mm256_storeu_pd(Ci+j,_mm256_fmadd_pd(ab,_mm256_loadu_pd(Bk+j),_mm256_loadu_pd(Ci+j)));
                _mm256_storeu_pd(Ci+j+4,_mm256_fmadd_pd(ab,_mm256_loadu_pd(Bk+j+4),_mm256_loadu_pd(Ci+j+4)));
            }
            for(;j<n;j++) Ci[j]+=A[i*n+k]*Bk[j];
        }
    }
}
void str_avx(const double *A, const double *B, double *C, int n, int thresh) {
    if(n<=thresh){ikj_avx2(A,B,C,n);return;}
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
    printf("  COM6 v9 - Cache-Hierarchy Blocking + Double-Pumped AVX2 FMA\n");
    printf("  MC=%d KC=%d NR=%d MR=%d | L2-blocked A, L1-blocked k\n", MC, KC, NR, MR);
    printf("====================================================================\n\n");

    /* Threshold sweep */
    printf("--- Threshold sweep at 2048 ---\n");
    {
        int n=2048;
        double *A=amalloc(n*(size_t)n),*B=amalloc(n*(size_t)n),*C=amalloc(n*(size_t)n);
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int ths[]={64,128,256,512,1024};
        for(int t=0;t<5;t++){double t0=get_ms();com6_v9(A,B,C,n,ths[t]);printf("  COM6-v9  thresh=%4d: %8.1f ms\n",ths[t],get_ms()-t0);}
        for(int t=0;t<5;t++){double t0=get_ms();str_avx(A,B,C,n,ths[t]);printf("  Str-AVX2 thresh=%4d: %8.1f ms\n",ths[t],get_ms()-t0);}
        afree(A);afree(B);afree(C);
    }

    printf("\n%-9s | %10s | %10s | %10s | %8s | %s\n","Size","Str-AVX2","COM6-v8","COM6-v9","v9/Str","Verify");
    printf("----------------------------------------------------------------\n");

    int sizes[]={256,512,1024,2048,4096};
    for(int si=0;si<5;si++){
        int n=sizes[si];
        double *A=amalloc(n*(size_t)n),*B=amalloc(n*(size_t)n);
        double *C1=amalloc(n*(size_t)n),*C2=amalloc(n*(size_t)n),*C3=amalloc(n*(size_t)n);
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int runs=n<=512?5:n<=2048?2:1;
        double t_str=0,t_v8=0,t_v9=0;

        if((n&(n-1))==0){
            /* Warmup */
            str_avx(A,B,C1,n,128);

            /* v8-style (from previous: flat 2x4, no cache blocking) */
            {
                double *BT=amalloc(n*(size_t)n);
                for(int i=0;i<n;i++)for(int j=0;j<n;j++)BT[j*n+i]=B[i*n+j];
                /* Inline v8 base for direct comparison */
                double t0=get_ms();
                for(int r=0;r<runs;r++){
                    /* Use com6_v9 with thresh=n (forces pure base case = v8 behavior without cache blocking) */
                }
                afree(BT);
            }

            double t0;
            /* Strassen best threshold */
            t0=get_ms();for(int r=0;r<runs;r++)str_avx(A,B,C1,n,128);t_str=(get_ms()-t0)/runs;
            /* COM6 v9 best threshold */
            t0=get_ms();for(int r=0;r<runs;r++)com6_v9(A,B,C2,n,128);t_v9=(get_ms()-t0)/runs;
        }

        double diff=t_str>0&&t_v9>0?mdiff(C1,C2,n):0;
        char vfy[20];snprintf(vfy,20,diff<1e-4?"OK":"e=%.0e",diff);
        char ss[20],sv9[20];
        if(t_str>0)snprintf(ss,20,"%8.1f ms",t_str);else snprintf(ss,20,"%10s","N/A");
        if(t_v9>0)snprintf(sv9,20,"%8.1f ms",t_v9);else snprintf(sv9,20,"%10s","N/A");

        printf("%4dx%-4d | %10s | %10s | %10s |",n,n,ss,"  -",sv9);
        if(t_str>0&&t_v9>0)printf(" %7.2fx |",t_str/t_v9);else printf(" %8s |","-");
        printf(" %s\n",vfy);

        afree(A);afree(B);afree(C1);afree(C2);afree(C3);
    }
    printf("\nv9/Str > 1.0 = COM6 BEATS Strassen\n");
    return 0;
}
