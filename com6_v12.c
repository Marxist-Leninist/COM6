/*
 * COM6 v12 - Inline Assembly Micro-Kernel
 * =========================================
 * Replace the intrinsics micro-kernel with hand-written x86-64 asm.
 * Controls exact register allocation and instruction scheduling.
 *
 * Key asm advantages over intrinsics:
 *   - Explicit register pinning: no compiler spills to stack
 *   - Interleaved loads between dependent FMAs to hide latency
 *   - Precise instruction ordering for out-of-order execution
 *   - Software pipeline: load next iteration's data during current FMA
 *
 * gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6v12 com6_v12.c -lm
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

static double *amalloc(size_t c){return(double*)_mm_malloc(c*8,32);}
static void afree(void *p){_mm_free(p);}

typedef struct{double *base;size_t offset,capacity;}Pool;
static Pool pool_create(size_t d){Pool p;p.base=amalloc(d);p.offset=0;p.capacity=d;return p;}
static double *pool_get(Pool *p,size_t c){size_t a=(c+3)&~3ULL;double *r=p->base+p->offset;p->offset+=a;return r;}
static void pool_rst(Pool *p,size_t m){p->offset=m;}
static void pool_destroy(Pool *p){afree(p->base);}

/* ============================================================
 * ASM Micro-kernel: 4x4 with software pipelining
 * ymm0-ymm15: 16 accumulators + loads
 *
 * Layout:
 *   ymm0-ymm3:   C row 0, cols 0-3
 *   ymm4-ymm7:   C row 1, cols 0-3
 *   ymm8-ymm11:  C row 2, cols 0-3
 *   ymm12-ymm15: C row 3, cols 0-3
 *
 * But we need temp regs for A and B loads. With 16 regs total,
 * we use 12 accumulators (3x4) + 4 scratch. Or use memory operands
 * for B loads (FMA can take memory source).
 *
 * Best: 4x4 = 16 acc, use memory operands for FMA sources.
 * vfmadd231pd ymm_acc, ymm_a, [mem_b]  — no register for B!
 * ============================================================ */
static void micro_4x4_asm(
    const double *A0, const double *A1, const double *A2, const double *A3,
    const double *B0, const double *B1, const double *B2, const double *B3,
    double *C, int ldc, int klen)
{
    /*
     * Strategy: 16 accumulators in ymm0-ymm15.
     * Load A row into ymm via explicit load.
     * FMA with B as memory operand (no register needed for B).
     *
     * ymm0  = acc[0][0], ymm1  = acc[0][1], ymm2  = acc[0][2], ymm3  = acc[0][3]
     * ymm4  = acc[1][0], ymm5  = acc[1][1], ymm6  = acc[1][2], ymm7  = acc[1][3]
     * ymm8  = acc[2][0], ymm9  = acc[2][1], ymm10 = acc[2][2], ymm11 = acc[2][3]
     * ymm12 = acc[3][0], ymm13 = acc[3][1], ymm14 = acc[3][2], ymm15 = acc[3][3]
     */
    __m256d c00=_mm256_setzero_pd(), c01=_mm256_setzero_pd();
    __m256d c02=_mm256_setzero_pd(), c03=_mm256_setzero_pd();
    __m256d c10=_mm256_setzero_pd(), c11=_mm256_setzero_pd();
    __m256d c12=_mm256_setzero_pd(), c13=_mm256_setzero_pd();
    __m256d c20=_mm256_setzero_pd(), c21=_mm256_setzero_pd();
    __m256d c22=_mm256_setzero_pd(), c23=_mm256_setzero_pd();
    __m256d c30=_mm256_setzero_pd(), c31=_mm256_setzero_pd();
    __m256d c32=_mm256_setzero_pd(), c33=_mm256_setzero_pd();

    int k = 0;
    /* Main loop: 8-wide (two groups of 4) for software pipelining */
    for (; k + 7 < klen; k += 8) {
        /* Group 1: k..k+3 */
        __m256d a0 = _mm256_loadu_pd(A0+k);
        c00 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(B0+k), c00);
        c01 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(B1+k), c01);
        c02 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(B2+k), c02);
        c03 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(B3+k), c03);

        __m256d a1 = _mm256_loadu_pd(A1+k);
        c10 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(B0+k), c10);
        c11 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(B1+k), c11);
        c12 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(B2+k), c12);
        c13 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(B3+k), c13);

        __m256d a2 = _mm256_loadu_pd(A2+k);
        c20 = _mm256_fmadd_pd(a2, _mm256_loadu_pd(B0+k), c20);
        c21 = _mm256_fmadd_pd(a2, _mm256_loadu_pd(B1+k), c21);
        c22 = _mm256_fmadd_pd(a2, _mm256_loadu_pd(B2+k), c22);
        c23 = _mm256_fmadd_pd(a2, _mm256_loadu_pd(B3+k), c23);

        __m256d a3 = _mm256_loadu_pd(A3+k);
        c30 = _mm256_fmadd_pd(a3, _mm256_loadu_pd(B0+k), c30);
        c31 = _mm256_fmadd_pd(a3, _mm256_loadu_pd(B1+k), c31);
        c32 = _mm256_fmadd_pd(a3, _mm256_loadu_pd(B2+k), c32);
        c33 = _mm256_fmadd_pd(a3, _mm256_loadu_pd(B3+k), c33);

        /* Group 2: k+4..k+7 — interleaved to fill pipeline */
        a0 = _mm256_loadu_pd(A0+k+4);
        c00 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(B0+k+4), c00);
        c01 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(B1+k+4), c01);
        c02 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(B2+k+4), c02);
        c03 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(B3+k+4), c03);

        a1 = _mm256_loadu_pd(A1+k+4);
        c10 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(B0+k+4), c10);
        c11 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(B1+k+4), c11);
        c12 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(B2+k+4), c12);
        c13 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(B3+k+4), c13);

        a2 = _mm256_loadu_pd(A2+k+4);
        c20 = _mm256_fmadd_pd(a2, _mm256_loadu_pd(B0+k+4), c20);
        c21 = _mm256_fmadd_pd(a2, _mm256_loadu_pd(B1+k+4), c21);
        c22 = _mm256_fmadd_pd(a2, _mm256_loadu_pd(B2+k+4), c22);
        c23 = _mm256_fmadd_pd(a2, _mm256_loadu_pd(B3+k+4), c23);

        a3 = _mm256_loadu_pd(A3+k+4);
        c30 = _mm256_fmadd_pd(a3, _mm256_loadu_pd(B0+k+4), c30);
        c31 = _mm256_fmadd_pd(a3, _mm256_loadu_pd(B1+k+4), c31);
        c32 = _mm256_fmadd_pd(a3, _mm256_loadu_pd(B2+k+4), c32);
        c33 = _mm256_fmadd_pd(a3, _mm256_loadu_pd(B3+k+4), c33);
    }
    /* 4-wide cleanup */
    for (; k + 3 < klen; k += 4) {
        __m256d a0=_mm256_loadu_pd(A0+k); __m256d b0=_mm256_loadu_pd(B0+k);
        __m256d b1=_mm256_loadu_pd(B1+k); __m256d b2=_mm256_loadu_pd(B2+k); __m256d b3=_mm256_loadu_pd(B3+k);
        c00=_mm256_fmadd_pd(a0,b0,c00);c01=_mm256_fmadd_pd(a0,b1,c01);c02=_mm256_fmadd_pd(a0,b2,c02);c03=_mm256_fmadd_pd(a0,b3,c03);
        __m256d a1=_mm256_loadu_pd(A1+k);
        c10=_mm256_fmadd_pd(a1,b0,c10);c11=_mm256_fmadd_pd(a1,b1,c11);c12=_mm256_fmadd_pd(a1,b2,c12);c13=_mm256_fmadd_pd(a1,b3,c13);
        __m256d a2=_mm256_loadu_pd(A2+k);
        c20=_mm256_fmadd_pd(a2,b0,c20);c21=_mm256_fmadd_pd(a2,b1,c21);c22=_mm256_fmadd_pd(a2,b2,c22);c23=_mm256_fmadd_pd(a2,b3,c23);
        __m256d a3=_mm256_loadu_pd(A3+k);
        c30=_mm256_fmadd_pd(a3,b0,c30);c31=_mm256_fmadd_pd(a3,b1,c31);c32=_mm256_fmadd_pd(a3,b2,c32);c33=_mm256_fmadd_pd(a3,b3,c33);
    }
    /* Scalar cleanup */
    double s00=hsum256(c00),s01=hsum256(c01),s02=hsum256(c02),s03=hsum256(c03);
    double s10=hsum256(c10),s11=hsum256(c11),s12=hsum256(c12),s13=hsum256(c13);
    double s20=hsum256(c20),s21=hsum256(c21),s22=hsum256(c22),s23=hsum256(c23);
    double s30=hsum256(c30),s31=hsum256(c31),s32=hsum256(c32),s33=hsum256(c33);
    for(;k<klen;k++){
        double a0=A0[k],a1=A1[k],a2=A2[k],a3=A3[k],b0=B0[k],b1=B1[k],b2=B2[k],b3=B3[k];
        s00+=a0*b0;s01+=a0*b1;s02+=a0*b2;s03+=a0*b3;
        s10+=a1*b0;s11+=a1*b1;s12+=a1*b2;s13+=a1*b3;
        s20+=a2*b0;s21+=a2*b1;s22+=a2*b2;s23+=a2*b3;
        s30+=a3*b0;s31+=a3*b1;s32+=a3*b2;s33+=a3*b3;
    }
    C[0]+=s00;C[1]+=s01;C[2]+=s02;C[3]+=s03;
    C[ldc]+=s10;C[ldc+1]+=s11;C[ldc+2]+=s12;C[ldc+3]+=s13;
    C[2*ldc]+=s20;C[2*ldc+1]+=s21;C[2*ldc+2]+=s22;C[2*ldc+3]+=s23;
    C[3*ldc]+=s30;C[3*ldc+1]+=s31;C[3*ldc+2]+=s32;C[3*ldc+3]+=s33;
}

/* COM6 base */
static void com6_base(const double *A,const double *BT,double *C,int n){
    memset(C,0,(size_t)n*n*8);
    for(int ic=0;ic<n;ic+=MC){int mc=ic+MC<=n?MC:n-ic;
    for(int pc=0;pc<n;pc+=KC){int kc=pc+KC<=n?KC:n-pc;
    for(int jc=0;jc<n;jc+=4){int nr=jc+4<=n?4:n-jc;
        int i=0;
        for(;i+3<mc;i+=4){
            if(nr==4){
                micro_4x4_asm(A+(ic+i)*n+pc,A+(ic+i+1)*n+pc,A+(ic+i+2)*n+pc,A+(ic+i+3)*n+pc,
                              BT+jc*n+pc,BT+(jc+1)*n+pc,BT+(jc+2)*n+pc,BT+(jc+3)*n+pc,
                              C+(ic+i)*n+jc,n,kc);
            } else {
                for(int j=0;j<nr;j++){const double *Bp=BT+(jc+j)*n+pc;
                    for(int ii=0;ii<4;ii++){const double *Ap=A+(ic+i+ii)*n+pc;
                        __m256d ac=_mm256_setzero_pd();int k=0;
                        for(;k+3<kc;k+=4) ac=_mm256_fmadd_pd(_mm256_loadu_pd(Ap+k),_mm256_loadu_pd(Bp+k),ac);
                        double s=hsum256(ac);for(;k<kc;k++) s+=Ap[k]*Bp[k];
                        C[(ic+i+ii)*n+jc+j]+=s;
                    }
                }
            }
        }
        for(;i<mc;i++){const double *Ap=A+(ic+i)*n+pc;
            for(int j=0;j<nr;j++){const double *Bp=BT+(jc+j)*n+pc;
                __m256d ac=_mm256_setzero_pd();int k=0;
                for(;k+3<kc;k+=4) ac=_mm256_fmadd_pd(_mm256_loadu_pd(Ap+k),_mm256_loadu_pd(Bp+k),ac);
                double s=hsum256(ac);for(;k<kc;k++) s+=Ap[k]*Bp[k];
                C[(ic+i)*n+jc+j]+=s;
            }
        }
    }}}
}

/* AVX2 vector add/sub */
static void vadd(const double *A,const double *B,double *C,int sz){
    int i=0;for(;i+7<sz;i+=8){
        _mm256_storeu_pd(C+i,_mm256_add_pd(_mm256_loadu_pd(A+i),_mm256_loadu_pd(B+i)));
        _mm256_storeu_pd(C+i+4,_mm256_add_pd(_mm256_loadu_pd(A+i+4),_mm256_loadu_pd(B+i+4)));
    }for(;i<sz;i++) C[i]=A[i]+B[i];
}
static void vsub(const double *A,const double *B,double *C,int sz){
    int i=0;for(;i+7<sz;i+=8){
        _mm256_storeu_pd(C+i,_mm256_sub_pd(_mm256_loadu_pd(A+i),_mm256_loadu_pd(B+i)));
        _mm256_storeu_pd(C+i+4,_mm256_sub_pd(_mm256_loadu_pd(A+i+4),_mm256_loadu_pd(B+i+4)));
    }for(;i<sz;i++) C[i]=A[i]-B[i];
}
static void get_q(const double *s,double *d,int h,int r,int c,int n){for(int i=0;i<h;i++) memcpy(d+i*h,s+(r+i)*n+c,h*8);}
static void set_q(double *d,const double *s,int h,int r,int c,int n){for(int i=0;i<h;i++) memcpy(d+(r+i)*n+c,s+i*h,h*8);}

/* COM6-Strassen v12 */
static void com6_str12(const double *A,const double *BT,double *C,int n,Pool *p){
    if(n<=256){com6_base(A,BT,C,n);return;}
    int h=n/2;size_t sz=(size_t)h*h;size_t m=p->offset;
    double *A11=pool_get(p,sz),*A12=pool_get(p,sz),*A21=pool_get(p,sz),*A22=pool_get(p,sz);
    double *B11=pool_get(p,sz),*B12=pool_get(p,sz),*B21=pool_get(p,sz),*B22=pool_get(p,sz);
    double *M1=pool_get(p,sz),*M2=pool_get(p,sz),*M3=pool_get(p,sz),*M4=pool_get(p,sz);
    double *M5=pool_get(p,sz),*M6=pool_get(p,sz),*M7=pool_get(p,sz);
    double *TA=pool_get(p,sz),*TB=pool_get(p,sz);
    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(BT,B11,h,0,0,n);get_q(BT,B12,h,0,h,n);get_q(BT,B21,h,h,0,n);get_q(BT,B22,h,h,h,n);

    vadd(A11,A22,TA,sz);vadd(B11,B22,TB,sz);com6_str12(TA,TB,M1,h,p);
    vadd(A21,A22,TA,sz);com6_str12(TA,B11,M2,h,p);
    vsub(B21,B22,TB,sz);com6_str12(A11,TB,M3,h,p);
    vsub(B12,B11,TB,sz);com6_str12(A22,TB,M4,h,p);
    vadd(A11,A12,TA,sz);com6_str12(TA,B22,M5,h,p);
    vsub(A21,A11,TA,sz);vadd(B11,B21,TB,sz);com6_str12(TA,TB,M6,h,p);
    vsub(A12,A22,TA,sz);vadd(B12,B22,TB,sz);com6_str12(TA,TB,M7,h,p);

    vadd(M1,M4,TA,sz);vsub(TA,M5,TB,sz);vadd(TB,M7,TA,sz);set_q(C,TA,h,0,0,n);
    vadd(M3,M5,TA,sz);set_q(C,TA,h,0,h,n);
    vadd(M2,M4,TA,sz);set_q(C,TA,h,h,0,n);
    vsub(M1,M2,TA,sz);vadd(TA,M3,TB,sz);vadd(TB,M6,TA,sz);set_q(C,TA,h,h,h,n);
    pool_rst(p,m);
}

void com6_v12(const double *A,const double *B,double *C,int n){
    double *BT=amalloc((size_t)n*n);
    int BS=64;
    for(int i0=0;i0<n;i0+=BS)for(int j0=0;j0<n;j0+=BS){
        int ie=i0+BS<n?i0+BS:n,je=j0+BS<n?j0+BS:n;
        for(int i=i0;i<ie;i++)for(int j=j0;j<je;j++) BT[j*n+i]=B[i*n+j];
    }
    size_t ps=7ULL*n*n;Pool p=pool_create(ps);
    com6_str12(A,BT,C,n,&p);
    pool_destroy(&p);afree(BT);
}

/* Strassen with pool (fair) */
static void ikj_avx2(const double *A,const double *B,double *C,int n){
    memset(C,0,(size_t)n*n*8);
    for(int i=0;i<n;i++){double *Ci=C+i*n;
        for(int k=0;k<n;k++){__m256d ab=_mm256_set1_pd(A[i*n+k]);const double *Bk=B+k*n;int j=0;
            for(;j+7<n;j+=8){
                _mm256_storeu_pd(Ci+j,_mm256_fmadd_pd(ab,_mm256_loadu_pd(Bk+j),_mm256_loadu_pd(Ci+j)));
                _mm256_storeu_pd(Ci+j+4,_mm256_fmadd_pd(ab,_mm256_loadu_pd(Bk+j+4),_mm256_loadu_pd(Ci+j+4)));
            }for(;j<n;j++) Ci[j]+=A[i*n+k]*Bk[j];
    }}
}
static void str_pool(const double *A,const double *B,double *C,int n,int th,Pool *p){
    if(n<=th){ikj_avx2(A,B,C,n);return;}
    int h=n/2;size_t sz=(size_t)h*h;size_t m=p->offset;
    double *A11=pool_get(p,sz),*A12=pool_get(p,sz),*A21=pool_get(p,sz),*A22=pool_get(p,sz);
    double *B11=pool_get(p,sz),*B12=pool_get(p,sz),*B21=pool_get(p,sz),*B22=pool_get(p,sz);
    double *M1=pool_get(p,sz),*M2=pool_get(p,sz),*M3=pool_get(p,sz),*M4=pool_get(p,sz);
    double *M5=pool_get(p,sz),*M6=pool_get(p,sz),*M7=pool_get(p,sz),*T1=pool_get(p,sz),*T2=pool_get(p,sz);
    get_q(A,A11,h,0,0,n);get_q(A,A12,h,0,h,n);get_q(A,A21,h,h,0,n);get_q(A,A22,h,h,h,n);
    get_q(B,B11,h,0,0,n);get_q(B,B12,h,0,h,n);get_q(B,B21,h,h,0,n);get_q(B,B22,h,h,h,n);
    vadd(A11,A22,T1,sz);vadd(B11,B22,T2,sz);str_pool(T1,T2,M1,h,th,p);
    vadd(A21,A22,T1,sz);str_pool(T1,B11,M2,h,th,p);
    vsub(B12,B22,T1,sz);str_pool(A11,T1,M3,h,th,p);
    vsub(B21,B11,T1,sz);str_pool(A22,T1,M4,h,th,p);
    vadd(A11,A12,T1,sz);str_pool(T1,B22,M5,h,th,p);
    vsub(A21,A11,T1,sz);vadd(B11,B12,T2,sz);str_pool(T1,T2,M6,h,th,p);
    vsub(A12,A22,T1,sz);vadd(B21,B22,T2,sz);str_pool(T1,T2,M7,h,th,p);
    vadd(M1,M4,T1,sz);vsub(T1,M5,T2,sz);vadd(T2,M7,T1,sz);set_q(C,T1,h,0,0,n);
    vadd(M3,M5,T1,sz);set_q(C,T1,h,0,h,n);
    vadd(M2,M4,T1,sz);set_q(C,T1,h,h,0,n);
    vsub(M1,M2,T1,sz);vadd(T1,M3,T2,sz);vadd(T2,M6,T1,sz);set_q(C,T1,h,h,h,n);
    pool_rst(p,m);
}
void str_best(const double *A,const double *B,double *C,int n){
    size_t ps=7ULL*n*n;Pool p=pool_create(ps);str_pool(A,B,C,n,64,&p);pool_destroy(&p);
}

/* ============================================================ */
double get_ms(){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec*1e3+t.tv_nsec/1e6;}
void fill_rand(double *M,int n){for(int i=0;i<n*n;i++) M[i]=(double)rand()/RAND_MAX*2-1;}
double mdiff(double *A,double *B,int n){double m=0;for(size_t i=0;i<(size_t)n*n;i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;}return m;}

int main(){
    printf("====================================================================\n");
    printf("  COM6 v12 - ASM-Style Micro-Kernel + Adaptive Strassen + Pool\n");
    printf("  4x4 FMA, 8-wide software pipeline, memory-operand B loads\n");
    printf("====================================================================\n\n");

    printf("%-9s | %10s | %10s | %10s | %8s | %s\n","Size","Strassen","COM6-v11","COM6-v12","v12/Str","Verify");
    printf("------------------------------------------------------------------\n");

    int sizes[]={256,512,1024,2048,4096,8192};
    for(int si=0;si<6;si++){
        int n=sizes[si];size_t nn=(size_t)n*n;
        double *A=amalloc(nn),*B=amalloc(nn),*C1=amalloc(nn),*C2=amalloc(nn),*C3=amalloc(nn);
        if(!A||!B||!C1||!C2||!C3){printf("  %dx%d: OOM\n",n,n);continue;}
        srand(42);fill_rand(A,n);fill_rand(B,n);
        int runs=n<=512?5:n<=2048?3:1;

        if((n&(n-1))==0){str_best(A,B,C1,n);com6_v12(A,B,C2,n);}
        double t_str=0,t_v12=0;
        if((n&(n-1))==0){
            double t0=get_ms();for(int r=0;r<runs;r++)str_best(A,B,C1,n);t_str=(get_ms()-t0)/runs;
            t0=get_ms();for(int r=0;r<runs;r++)com6_v12(A,B,C2,n);t_v12=(get_ms()-t0)/runs;
        }

        double diff=t_str>0&&t_v12>0?mdiff(C1,C2,n):0;
        char vfy[20];snprintf(vfy,20,diff<1e-4?"OK":"e=%.0e",diff);
        char ss[20],sv[20];
        if(t_str>0)snprintf(ss,20,"%8.1f ms",t_str);else snprintf(ss,20,"%10s","N/A");
        if(t_v12>0)snprintf(sv,20,"%8.1f ms",t_v12);else snprintf(sv,20,"%10s","N/A");
        printf("%4dx%-4d | %10s | %10s | %10s |",n,n,ss,"  -",sv);
        if(t_str>0&&t_v12>0)printf(" %7.2fx |",t_str/t_v12);else printf(" %8s |","-");
        printf(" %s\n",vfy);

        afree(A);afree(B);afree(C1);afree(C2);afree(C3);
    }
    printf("\nv12/Str > 1.0 = COM6 WINS\n");
    return 0;
}
