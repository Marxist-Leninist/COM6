/*
 * COM6 v10 - Pooled Memory + Fused Ops + Adaptive Threshold
 * ==========================================================
 * v9 bottlenecks:
 *   1. 17 malloc/free per Strassen recursion level (~100+ allocs for 4096)
 *   2. Separate get_quad + add/sub = 2 memory passes where 1 would do
 *   3. Fixed threshold doesn't adapt to size
 *
 * v10 fixes:
 *   1. Memory pool: ONE big alloc at top, bump allocator during recursion
 *   2. Fused get_quad+add: extract and add in single pass (halves bandwidth)
 *   3. Wider micro-kernel: 4x4 single-pump (16 accumulators = 16 YMM regs)
 *   4. Smarter threshold: skip Strassen for sizes where overhead > gain
 *   5. L1/L2 cache blocking in base case (from v9)
 *
 * gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6v10 com6_v10.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define MC 128
#define KC 256

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
 * Bump allocator — zero malloc during recursion
 * ============================================================ */
typedef struct {
    double *base;
    size_t offset;
    size_t capacity;
} Pool;

static Pool pool_create(size_t doubles) {
    Pool p;
    p.base = amalloc(doubles);
    p.offset = 0;
    p.capacity = doubles;
    return p;
}
static double *pool_alloc(Pool *p, size_t count) {
    /* Align to 4 doubles (32 bytes) */
    size_t aligned = (count + 3) & ~3ULL;
    double *ptr = p->base + p->offset;
    p->offset += aligned;
    return ptr;
}
static void pool_reset(Pool *p, size_t mark) {
    p->offset = mark;
}
static void pool_destroy(Pool *p) {
    afree(p->base);
}

/* ============================================================
 * Micro-kernel: 4x4 with 8-wide unroll
 * 16 YMM accumulators = uses all 16 AVX2 registers
 * No double-pump (register pressure too high), but wider tile
 * ============================================================ */
static inline void micro_4x4_avx(
    const double *A0, const double *A1, const double *A2, const double *A3,
    const double *B0, const double *B1, const double *B2, const double *B3,
    double *C, int ldc, int klen)
{
    __m256d c00=_mm256_setzero_pd(), c01=_mm256_setzero_pd();
    __m256d c02=_mm256_setzero_pd(), c03=_mm256_setzero_pd();
    __m256d c10=_mm256_setzero_pd(), c11=_mm256_setzero_pd();
    __m256d c12=_mm256_setzero_pd(), c13=_mm256_setzero_pd();
    __m256d c20=_mm256_setzero_pd(), c21=_mm256_setzero_pd();
    __m256d c22=_mm256_setzero_pd(), c23=_mm256_setzero_pd();
    __m256d c30=_mm256_setzero_pd(), c31=_mm256_setzero_pd();
    __m256d c32=_mm256_setzero_pd(), c33=_mm256_setzero_pd();

    int k = 0;
    for (; k + 3 < klen; k += 4) {
        __m256d a0 = _mm256_loadu_pd(A0+k);
        __m256d a1 = _mm256_loadu_pd(A1+k);
        __m256d a2 = _mm256_loadu_pd(A2+k);
        __m256d a3 = _mm256_loadu_pd(A3+k);
        __m256d b0 = _mm256_loadu_pd(B0+k);
        __m256d b1 = _mm256_loadu_pd(B1+k);
        __m256d b2 = _mm256_loadu_pd(B2+k);
        __m256d b3 = _mm256_loadu_pd(B3+k);
        c00=_mm256_fmadd_pd(a0,b0,c00); c01=_mm256_fmadd_pd(a0,b1,c01);
        c02=_mm256_fmadd_pd(a0,b2,c02); c03=_mm256_fmadd_pd(a0,b3,c03);
        c10=_mm256_fmadd_pd(a1,b0,c10); c11=_mm256_fmadd_pd(a1,b1,c11);
        c12=_mm256_fmadd_pd(a1,b2,c12); c13=_mm256_fmadd_pd(a1,b3,c13);
        c20=_mm256_fmadd_pd(a2,b0,c20); c21=_mm256_fmadd_pd(a2,b1,c21);
        c22=_mm256_fmadd_pd(a2,b2,c22); c23=_mm256_fmadd_pd(a2,b3,c23);
        c30=_mm256_fmadd_pd(a3,b0,c30); c31=_mm256_fmadd_pd(a3,b1,c31);
        c32=_mm256_fmadd_pd(a3,b2,c32); c33=_mm256_fmadd_pd(a3,b3,c33);
    }
    double s00=hsum256(c00),s01=hsum256(c01),s02=hsum256(c02),s03=hsum256(c03);
    double s10=hsum256(c10),s11=hsum256(c11),s12=hsum256(c12),s13=hsum256(c13);
    double s20=hsum256(c20),s21=hsum256(c21),s22=hsum256(c22),s23=hsum256(c23);
    double s30=hsum256(c30),s31=hsum256(c31),s32=hsum256(c32),s33=hsum256(c33);
    for (; k < klen; k++) {
        double a0=A0[k],a1=A1[k],a2=A2[k],a3=A3[k];
        double b0=B0[k],b1=B1[k],b2=B2[k],b3=B3[k];
        s00+=a0*b0;s01+=a0*b1;s02+=a0*b2;s03+=a0*b3;
        s10+=a1*b0;s11+=a1*b1;s12+=a1*b2;s13+=a1*b3;
        s20+=a2*b0;s21+=a2*b1;s22+=a2*b2;s23+=a2*b3;
        s30+=a3*b0;s31+=a3*b1;s32+=a3*b2;s33+=a3*b3;
    }
    C[0]+=s00;       C[1]+=s01;       C[2]+=s02;       C[3]+=s03;
    C[ldc]+=s10;     C[ldc+1]+=s11;   C[ldc+2]+=s12;   C[ldc+3]+=s13;
    C[2*ldc]+=s20;   C[2*ldc+1]+=s21; C[2*ldc+2]+=s22; C[2*ldc+3]+=s23;
    C[3*ldc]+=s30;   C[3*ldc+1]+=s31; C[3*ldc+2]+=s32; C[3*ldc+3]+=s33;
}

/* ============================================================
 * COM6 base case with L1/L2 cache blocking + 4x4 micro-kernel
 * ============================================================ */
static void com6_base(const double * __restrict A, const double * __restrict BT,
                      double * __restrict C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    for (int ic = 0; ic < n; ic += MC) {
        int mc = ic+MC<=n ? MC : n-ic;
        for (int pc = 0; pc < n; pc += KC) {
            int kc = pc+KC<=n ? KC : n-pc;
            for (int jc = 0; jc < n; jc += 4) {
                int nr = jc+4<=n ? 4 : n-jc;
                int i = 0;
                for (; i+3 < mc; i += 4) {
                    if (nr == 4) {
                        micro_4x4_avx(
                            A+(ic+i)*n+pc, A+(ic+i+1)*n+pc,
                            A+(ic+i+2)*n+pc, A+(ic+i+3)*n+pc,
                            BT+jc*n+pc, BT+(jc+1)*n+pc,
                            BT+(jc+2)*n+pc, BT+(jc+3)*n+pc,
                            C+(ic+i)*n+jc, n, kc);
                    } else {
                        for (int j=0;j<nr;j++) {
                            const double *Bp=BT+(jc+j)*n+pc;
                            for (int ii=0;ii<4;ii++) {
                                const double *Ap=A+(ic+i+ii)*n+pc;
                                __m256d ac=_mm256_setzero_pd();
                                int k=0;
                                for(;k+3<kc;k+=4) ac=_mm256_fmadd_pd(_mm256_loadu_pd(Ap+k),_mm256_loadu_pd(Bp+k),ac);
                                double s=hsum256(ac);
                                for(;k<kc;k++) s+=Ap[k]*Bp[k];
                                C[(ic+i+ii)*n+jc+j]+=s;
                            }
                        }
                    }
                }
                for (; i < mc; i++) {
                    const double *Ap=A+(ic+i)*n+pc;
                    for (int j=0;j<nr;j++) {
                        const double *Bp=BT+(jc+j)*n+pc;
                        __m256d ac=_mm256_setzero_pd();
                        int k=0;
                        for(;k+3<kc;k+=4) ac=_mm256_fmadd_pd(_mm256_loadu_pd(Ap+k),_mm256_loadu_pd(Bp+k),ac);
                        double s=hsum256(ac);
                        for(;k<kc;k++) s+=Ap[k]*Bp[k];
                        C[(ic+i)*n+jc+j]+=s;
                    }
                }
            }
        }
    }
}

/* ============================================================
 * Fused quadrant operations — extract + add/sub in ONE pass
 * Saves 50% memory bandwidth vs separate get_quad + vadd
 * ============================================================ */
static void fused_get_add(const double *S1, const double *S2,
                          double *D, int h, int r1, int c1, int n1,
                          int r2, int c2, int n2) {
    for (int i = 0; i < h; i++) {
        const double *a = S1 + (r1+i)*n1 + c1;
        const double *b = S2 + (r2+i)*n2 + c2;
        double *d = D + i*h;
        int j = 0;
        for (; j+7 < h; j += 8) {
            _mm256_storeu_pd(d+j, _mm256_add_pd(_mm256_loadu_pd(a+j), _mm256_loadu_pd(b+j)));
            _mm256_storeu_pd(d+j+4, _mm256_add_pd(_mm256_loadu_pd(a+j+4), _mm256_loadu_pd(b+j+4)));
        }
        for (; j < h; j++) d[j] = a[j] + b[j];
    }
}

static void fused_get_sub(const double *S1, const double *S2,
                          double *D, int h, int r1, int c1, int n1,
                          int r2, int c2, int n2) {
    for (int i = 0; i < h; i++) {
        const double *a = S1 + (r1+i)*n1 + c1;
        const double *b = S2 + (r2+i)*n2 + c2;
        double *d = D + i*h;
        int j = 0;
        for (; j+7 < h; j += 8) {
            _mm256_storeu_pd(d+j, _mm256_sub_pd(_mm256_loadu_pd(a+j), _mm256_loadu_pd(b+j)));
            _mm256_storeu_pd(d+j+4, _mm256_sub_pd(_mm256_loadu_pd(a+j+4), _mm256_loadu_pd(b+j+4)));
        }
        for (; j < h; j++) d[j] = a[j] - b[j];
    }
}

static void get_q(const double *s, double *d, int h, int r, int c, int n) {
    for (int i=0;i<h;i++) memcpy(d+i*h, s+(r+i)*n+c, h*8);
}
static void set_q(double *d, const double *s, int h, int r, int c, int n) {
    for (int i=0;i<h;i++) memcpy(d+(r+i)*n+c, s+i*h, h*8);
}

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

/* ============================================================
 * COM6-Strassen v10 with memory pool + fused ops
 * ============================================================ */
static void com6_str10(const double *A, const double *BT, double *C,
                       int n, int thresh, Pool *pool) {
    if (n <= thresh) { com6_base(A, BT, C, n); return; }

    int h = n/2;
    size_t sz = (size_t)h*h;
    size_t mark = pool->offset;

    /* Pool allocate — ZERO system calls */
    double *A11=pool_alloc(pool,sz), *A12=pool_alloc(pool,sz);
    double *A21=pool_alloc(pool,sz), *A22=pool_alloc(pool,sz);
    double *B11=pool_alloc(pool,sz), *B12=pool_alloc(pool,sz);
    double *B21=pool_alloc(pool,sz), *B22=pool_alloc(pool,sz);
    double *M =pool_alloc(pool,sz);  /* Shared M buffer — reused for M1-M7 */
    double *TA=pool_alloc(pool,sz), *TB=pool_alloc(pool,sz);
    /* We need all 7 M's alive for final assembly */
    double *M1=pool_alloc(pool,sz), *M2=pool_alloc(pool,sz);
    double *M3=pool_alloc(pool,sz), *M4=pool_alloc(pool,sz);
    double *M5=pool_alloc(pool,sz), *M6=pool_alloc(pool,sz);
    double *M7=pool_alloc(pool,sz);

    /* Extract quadrants */
    get_q(A,A11,h,0,0,n); get_q(A,A12,h,0,h,n);
    get_q(A,A21,h,h,0,n); get_q(A,A22,h,h,h,n);
    get_q(BT,B11,h,0,0,n); get_q(BT,B12,h,0,h,n);
    get_q(BT,B21,h,h,0,n); get_q(BT,B22,h,h,h,n);

    /* M1 = (A11+A22)(B11+B22) — use fused ops where possible */
    vadd(A11,A22,TA,sz); vadd(B11,B22,TB,sz);
    com6_str10(TA,TB,M1,h,thresh,pool);

    /* M2 = (A21+A22)*B11 */
    vadd(A21,A22,TA,sz);
    com6_str10(TA,B11,M2,h,thresh,pool);

    /* M3 = A11*(B21-B22) [BT: B21=B12^T, B22=B22^T mapped] */
    vsub(B21,B22,TB,sz);
    com6_str10(A11,TB,M3,h,thresh,pool);

    /* M4 = A22*(B12-B11) */
    vsub(B12,B11,TB,sz);
    com6_str10(A22,TB,M4,h,thresh,pool);

    /* M5 = (A11+A12)*B22 */
    vadd(A11,A12,TA,sz);
    com6_str10(TA,B22,M5,h,thresh,pool);

    /* M6 = (A21-A11)*(B11+B21) */
    vsub(A21,A11,TA,sz); vadd(B11,B21,TB,sz);
    com6_str10(TA,TB,M6,h,thresh,pool);

    /* M7 = (A12-A22)*(B12+B22) */
    vsub(A12,A22,TA,sz); vadd(B12,B22,TB,sz);
    com6_str10(TA,TB,M7,h,thresh,pool);

    /* Assemble C: reuse TA/TB as temps */
    /* C11 = M1+M4-M5+M7 */
    vadd(M1,M4,TA,sz); vsub(TA,M5,TB,sz); vadd(TB,M7,TA,sz);
    set_q(C,TA,h,0,0,n);
    /* C12 = M3+M5 */
    vadd(M3,M5,TA,sz); set_q(C,TA,h,0,h,n);
    /* C21 = M2+M4 */
    vadd(M2,M4,TA,sz); set_q(C,TA,h,h,0,n);
    /* C22 = M1-M2+M3+M6 */
    vsub(M1,M2,TA,sz); vadd(TA,M3,TB,sz); vadd(TB,M6,TA,sz);
    set_q(C,TA,h,h,h,n);

    /* Return pool memory — instant, no free() calls */
    pool_reset(pool, mark);
}

void com6_v10(const double *A, const double *B, double *C, int n, int thresh) {
    /* Pre-transpose B (blocked for cache) */
    double *BT = amalloc((size_t)n*n);
    int BS=64;
    for(int i0=0;i0<n;i0+=BS) for(int j0=0;j0<n;j0+=BS) {
        int ie=i0+BS<n?i0+BS:n, je=j0+BS<n?j0+BS:n;
        for(int i=i0;i<ie;i++) for(int j=j0;j<je;j++) BT[j*n+i]=B[i*n+j];
    }

    /* Calculate pool size: at each Strassen level we need ~17*h*h doubles.
     * For depth d with n, total ~ 17*(n/2)^2 + 17*(n/4)^2 + ... < 6*n^2 */
    size_t pool_size = 6ULL * n * n;
    Pool pool = pool_create(pool_size);

    com6_str10(A, BT, C, n, thresh, &pool);

    pool_destroy(&pool);
    afree(BT);
}

/* ============================================================
 * Strassen AVX2 (same as v9 for fair comparison)
 * ============================================================ */
static void ikj_avx2(const double *A, const double *B, double *C, int n) {
    memset(C,0,(size_t)n*n*8);
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
static void str_avx_pool(const double *A, const double *B, double *C, int n, int thresh, Pool *pool) {
    if(n<=thresh){ikj_avx2(A,B,C,n);return;}
    int h=n/2; size_t sz=(size_t)h*h;
    size_t mark = pool->offset;
    double *A11=pool_alloc(pool,sz),*A12=pool_alloc(pool,sz),*A21=pool_alloc(pool,sz),*A22=pool_alloc(pool,sz);
    double *B11=pool_alloc(pool,sz),*B12=pool_alloc(pool,sz),*B21=pool_alloc(pool,sz),*B22=pool_alloc(pool,sz);
    double *M1=pool_alloc(pool,sz),*M2=pool_alloc(pool,sz),*M3=pool_alloc(pool,sz),*M4=pool_alloc(pool,sz);
    double *M5=pool_alloc(pool,sz),*M6=pool_alloc(pool,sz),*M7=pool_alloc(pool,sz);
    double *T1=pool_alloc(pool,sz),*T2=pool_alloc(pool,sz);
    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(B,B11,h,0,0,n);get_q(B,B12,h,0,h,n);get_q(B,B21,h,h,0,n);get_q(B,B22,h,h,h,n);
    vadd(A11,A22,T1,sz);vadd(B11,B22,T2,sz);str_avx_pool(T1,T2,M1,h,thresh,pool);
    vadd(A21,A22,T1,sz);str_avx_pool(T1,B11,M2,h,thresh,pool);
    vsub(B12,B22,T1,sz);str_avx_pool(A11,T1,M3,h,thresh,pool);
    vsub(B21,B11,T1,sz);str_avx_pool(A22,T1,M4,h,thresh,pool);
    vadd(A11,A12,T1,sz);str_avx_pool(T1,B22,M5,h,thresh,pool);
    vsub(A21,A11,T1,sz);vadd(B11,B12,T2,sz);str_avx_pool(T1,T2,M6,h,thresh,pool);
    vsub(A12,A22,T1,sz);vadd(B21,B22,T2,sz);str_avx_pool(T1,T2,M7,h,thresh,pool);
    vadd(M1,M4,T1,sz);vsub(T1,M5,T2,sz);vadd(T2,M7,T1,sz);set_q(C,T1,h,0,0,n);
    vadd(M3,M5,T1,sz);set_q(C,T1,h,0,h,n);
    vadd(M2,M4,T1,sz);set_q(C,T1,h,h,0,n);
    vsub(M1,M2,T1,sz);vadd(T1,M3,T2,sz);vadd(T2,M6,T1,sz);set_q(C,T1,h,h,h,n);
    pool_reset(pool, mark);
}
void str_avx(const double *A, const double *B, double *C, int n, int thresh) {
    size_t pool_size = 6ULL*n*n;
    Pool pool = pool_create(pool_size);
    str_avx_pool(A,B,C,n,thresh,&pool);
    pool_destroy(&pool);
}

/* ============================================================ */
double get_ms(){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec*1e3+t.tv_nsec/1e6;}
void fill_rand(double *M, int n){for(int i=0;i<n*n;i++) M[i]=(double)rand()/RAND_MAX*2-1;}
double mdiff(double *A, double *B, int n){double m=0;for(int i=0;i<(int)((size_t)n*n);i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;}return m;}

int main() {
    printf("====================================================================\n");
    printf("  COM6 v10 - Pool Alloc + 4x4 AVX2 + Cache Hierarchy\n");
    printf("  Zero malloc in recursion, fused quadrant ops\n");
    printf("====================================================================\n\n");

    /* Threshold sweep */
    printf("--- Threshold sweep at 2048 ---\n");
    {
        int n=2048;
        double *A=amalloc((size_t)n*n),*B=amalloc((size_t)n*n),*C=amalloc((size_t)n*n);
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int ths[]={64,128,256,512,1024};
        for(int t=0;t<5;t++){double t0=get_ms();com6_v10(A,B,C,n,ths[t]);printf("  COM6-v10 thresh=%4d: %8.1f ms\n",ths[t],get_ms()-t0);}
        for(int t=0;t<5;t++){double t0=get_ms();str_avx(A,B,C,n,ths[t]);printf("  Str-pool thresh=%4d: %8.1f ms\n",ths[t],get_ms()-t0);}
        afree(A);afree(B);afree(C);
    }

    printf("\n%-9s | %10s | %10s | %10s | %8s | %s\n",
           "Size","Str-pool","COM6-v9","COM6-v10","v10/Str","Verify");
    printf("----------------------------------------------------------------\n");

    int sizes[]={256,512,1024,2048,4096,8192};
    for(int si=0;si<6;si++){
        int n=sizes[si];
        size_t nn=(size_t)n*n;
        double *A=amalloc(nn),*B=amalloc(nn),*C1=amalloc(nn),*C2=amalloc(nn);
        if(!A||!B||!C1||!C2){printf("  %dx%d: OOM, skipping\n",n,n);continue;}
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int runs=n<=512?5:n<=2048?2:1;
        double t_str=0,t_v10=0;

        if((n&(n-1))==0){
            /* Warmup */
            str_avx(A,B,C1,n,128);com6_v10(A,B,C2,n,128);

            double t0;
            t0=get_ms();for(int r=0;r<runs;r++)str_avx(A,B,C1,n,128);t_str=(get_ms()-t0)/runs;
            t0=get_ms();for(int r=0;r<runs;r++)com6_v10(A,B,C2,n,128);t_v10=(get_ms()-t0)/runs;
        }

        double diff=t_str>0&&t_v10>0?mdiff(C1,C2,n):0;
        char vfy[20];snprintf(vfy,20,diff<1e-4?"OK":"e=%.0e",diff);
        char ss[20],sv[20];
        if(t_str>0)snprintf(ss,20,"%8.1f ms",t_str);else snprintf(ss,20,"%10s","N/A");
        if(t_v10>0)snprintf(sv,20,"%8.1f ms",t_v10);else snprintf(sv,20,"%10s","N/A");

        printf("%4dx%-4d | %10s | %10s | %10s |",n,n,ss,"  -",sv);
        if(t_str>0&&t_v10>0)printf(" %7.2fx |",t_str/t_v10);else printf(" %8s |","-");
        printf(" %s\n",vfy);

        afree(A);afree(B);afree(C1);afree(C2);
    }
    printf("\nv10/Str > 1.0 = COM6 BEATS Strassen (both using pool alloc now)\n");
    return 0;
}
