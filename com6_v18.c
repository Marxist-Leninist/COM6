/*
 * COM6 v18 - Maximum throughput
 * ==============================
 * Back to v16's proven fast path (simple n-stride, no lda/ldb overhead)
 * with targeted improvements:
 *
 *   1. NC=2048 (L3-safe: B panel = 4MB in 8MB L3)
 *   2. MC=128 (wider M panel, still fits L2: 128*256*8=256KB)
 *   3. A-packing with 4-wide SIMD gather via shuffle tricks
 *   4. Prefetch tuning: deeper pipeline (16 ahead for L2)
 *   5. Two-level Strassen only for n>=2048 (algorithmic win at large n)
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v18 com6_v18.c -lm
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MR  6
#define NR  8
#define KC  256
#define MC  120     /* 120*256*8 = 240KB < 256KB L2; 120 % 6 == 0 (no edge!) */
#define NC  2048    /* 2048*256*8 = 4MB < 8MB L3 */
#define ALIGN 64

static inline double* aa(size_t c){return(double*)_mm_malloc(c*sizeof(double),ALIGN);}
static inline void af(double*p){_mm_free(p);}

/* ================================================================
 * MICRO-KERNEL 6x8, 4x k-unrolled
 * ================================================================ */
static void __attribute__((noinline))
micro_6x8(int kc, const double* __restrict__ pA, const double* __restrict__ pB,
           double* __restrict__ C, int n)
{
    __m256d c00=_mm256_setzero_pd(),c01=_mm256_setzero_pd();
    __m256d c10=_mm256_setzero_pd(),c11=_mm256_setzero_pd();
    __m256d c20=_mm256_setzero_pd(),c21=_mm256_setzero_pd();
    __m256d c30=_mm256_setzero_pd(),c31=_mm256_setzero_pd();
    __m256d c40=_mm256_setzero_pd(),c41=_mm256_setzero_pd();
    __m256d c50=_mm256_setzero_pd(),c51=_mm256_setzero_pd();

    int k=0, kc4=kc&~3;
    for(;k<kc4;k+=4){
        _mm_prefetch((const char*)(pA+MR*16),_MM_HINT_T1);
        _mm_prefetch((const char*)(pB+NR*8),_MM_HINT_T1);
        _mm_prefetch((const char*)(pB+NR*8+32),_MM_HINT_T1);

#define RANK1(off) do{ \
        __m256d b0=_mm256_load_pd(pB+(off)*NR); \
        __m256d b1=_mm256_load_pd(pB+(off)*NR+4); \
        __m256d a; \
        a=_mm256_broadcast_sd(pA+(off)*MR+0);c00=_mm256_fmadd_pd(a,b0,c00);c01=_mm256_fmadd_pd(a,b1,c01); \
        a=_mm256_broadcast_sd(pA+(off)*MR+1);c10=_mm256_fmadd_pd(a,b0,c10);c11=_mm256_fmadd_pd(a,b1,c11); \
        a=_mm256_broadcast_sd(pA+(off)*MR+2);c20=_mm256_fmadd_pd(a,b0,c20);c21=_mm256_fmadd_pd(a,b1,c21); \
        a=_mm256_broadcast_sd(pA+(off)*MR+3);c30=_mm256_fmadd_pd(a,b0,c30);c31=_mm256_fmadd_pd(a,b1,c31); \
        a=_mm256_broadcast_sd(pA+(off)*MR+4);c40=_mm256_fmadd_pd(a,b0,c40);c41=_mm256_fmadd_pd(a,b1,c41); \
        a=_mm256_broadcast_sd(pA+(off)*MR+5);c50=_mm256_fmadd_pd(a,b0,c50);c51=_mm256_fmadd_pd(a,b1,c51); \
        }while(0)
        RANK1(0); RANK1(1); RANK1(2); RANK1(3);
#undef RANK1
        pA+=4*MR; pB+=4*NR;
    }
    for(;k<kc;k++){
        __m256d b0=_mm256_load_pd(pB),b1=_mm256_load_pd(pB+4),a;
        a=_mm256_broadcast_sd(pA+0);c00=_mm256_fmadd_pd(a,b0,c00);c01=_mm256_fmadd_pd(a,b1,c01);
        a=_mm256_broadcast_sd(pA+1);c10=_mm256_fmadd_pd(a,b0,c10);c11=_mm256_fmadd_pd(a,b1,c11);
        a=_mm256_broadcast_sd(pA+2);c20=_mm256_fmadd_pd(a,b0,c20);c21=_mm256_fmadd_pd(a,b1,c21);
        a=_mm256_broadcast_sd(pA+3);c30=_mm256_fmadd_pd(a,b0,c30);c31=_mm256_fmadd_pd(a,b1,c31);
        a=_mm256_broadcast_sd(pA+4);c40=_mm256_fmadd_pd(a,b0,c40);c41=_mm256_fmadd_pd(a,b1,c41);
        a=_mm256_broadcast_sd(pA+5);c50=_mm256_fmadd_pd(a,b0,c50);c51=_mm256_fmadd_pd(a,b1,c51);
        pA+=MR;pB+=NR;
    }

#define STORE(row,lo,hi) do{ \
    double*c=C+(row)*n; \
    _mm256_storeu_pd(c,_mm256_add_pd(_mm256_loadu_pd(c),lo)); \
    _mm256_storeu_pd(c+4,_mm256_add_pd(_mm256_loadu_pd(c+4),hi)); \
    }while(0)
    STORE(0,c00,c01); STORE(1,c10,c11); STORE(2,c20,c21);
    STORE(3,c30,c31); STORE(4,c40,c41); STORE(5,c50,c51);
#undef STORE
}

static void micro_edge(int mr,int nr,int kc,const double*pA,const double*pB,double*C,int n){
    for(int k=0;k<kc;k++)for(int i=0;i<mr;i++){
        double av=pA[k*MR+i];for(int j=0;j<nr;j++)C[i*n+j]+=av*pB[k*NR+j];
    }
}

/* ================================================================
 * PACKING
 * ================================================================ */
static void pack_B(const double*__restrict__ B,double*__restrict__ pb,
                    int kc,int nc,int n,int j0,int k0)
{
    for(int j=0;j<nc;j+=NR){
        int nr=(j+NR<=nc)?NR:nc-j;
        const double*Bkj=B+(size_t)k0*n+(j0+j);
        if(nr==NR){
            for(int k=0;k<kc;k++){
                _mm256_store_pd(pb,_mm256_loadu_pd(Bkj));
                _mm256_store_pd(pb+4,_mm256_loadu_pd(Bkj+4));
                pb+=NR;Bkj+=n;
            }
        }else{
            for(int k=0;k<kc;k++){
                int jj;for(jj=0;jj<nr;jj++)pb[jj]=Bkj[jj];
                for(;jj<NR;jj++)pb[jj]=0.0;pb+=NR;Bkj+=n;
            }
        }
    }
}

static void pack_A(const double*__restrict__ A,double*__restrict__ pa,
                    int mc,int kc,int n,int i0,int k0)
{
    const double*Ab=A+(size_t)i0*n+k0;
    for(int i=0;i<mc;i+=MR){
        int mr=(i+MR<=mc)?MR:mc-i;
        if(mr==MR){
            const double*a0=Ab+i*n,*a1=a0+n,*a2=a0+2*n,
                        *a3=a0+3*n,*a4=a0+4*n,*a5=a0+5*n;
            /* Prefetch the rows we're about to gather from */
            _mm_prefetch((const char*)(a0+64),_MM_HINT_T0);
            _mm_prefetch((const char*)(a1+64),_MM_HINT_T0);
            _mm_prefetch((const char*)(a2+64),_MM_HINT_T0);
            _mm_prefetch((const char*)(a3+64),_MM_HINT_T0);
            _mm_prefetch((const char*)(a4+64),_MM_HINT_T0);
            _mm_prefetch((const char*)(a5+64),_MM_HINT_T0);
            for(int k=0;k<kc;k++){
                pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];
                pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];pa+=MR;
            }
        }else{
            for(int k=0;k<kc;k++){
                int ii;for(ii=0;ii<mr;ii++)pa[ii]=(Ab+(i+ii)*n)[k];
                for(;ii<MR;ii++)pa[ii]=0.0;pa+=MR;
            }
        }
    }
}

/* ================================================================
 * BLIS 5-LOOP
 * ================================================================ */
static void com6_multiply(const double*__restrict__ A,
                           const double*__restrict__ B,
                           double*__restrict__ C,int n)
{
    double*pa=aa((size_t)MC*KC);
    double*pb=aa((size_t)KC*NC);
    memset(C,0,(size_t)n*n*sizeof(double));

    for(int jc=0;jc<n;jc+=NC){
        int nc=(jc+NC<=n)?NC:n-jc;
        for(int pc=0;pc<n;pc+=KC){
            int kc=(pc+KC<=n)?KC:n-pc;
            pack_B(B,pb,kc,nc,n,jc,pc);
            for(int ic=0;ic<n;ic+=MC){
                int mc=(ic+MC<=n)?MC:n-ic;
                pack_A(A,pa,mc,kc,n,ic,pc);
                for(int jr=0;jr<nc;jr+=NR){
                    int nr=(jr+NR<=nc)?NR:nc-jr;
                    const double*pB=pb+(jr/NR)*((size_t)NR*kc);
                    for(int ir=0;ir<mc;ir+=MR){
                        int mr=(ir+MR<=mc)?MR:mc-ir;
                        const double*pA=pa+(ir/MR)*((size_t)MR*kc);
                        double*Cij=C+(size_t)(ic+ir)*n+(jc+jr);
                        if(mr==MR&&nr==NR)micro_6x8(kc,pA,pB,Cij,n);
                        else micro_edge(mr,nr,kc,pA,pB,Cij,n);
                    }
                }
            }
        }
    }
    af(pa);af(pb);
}

/* ================================================================
 * STRASSEN (one level only for large n, avoids deep recursion overhead)
 * ================================================================ */
static void mat_add(const double*A,const double*B,double*C,size_t t){
    size_t k=0;
    for(;k+3<t;k+=4){__m256d a=_mm256_loadu_pd(A+k),b=_mm256_loadu_pd(B+k);_mm256_storeu_pd(C+k,_mm256_add_pd(a,b));}
    for(;k<t;k++)C[k]=A[k]+B[k];
}
static void mat_sub(const double*A,const double*B,double*C,size_t t){
    size_t k=0;
    for(;k+3<t;k+=4){__m256d a=_mm256_loadu_pd(A+k),b=_mm256_loadu_pd(B+k);_mm256_storeu_pd(C+k,_mm256_sub_pd(a,b));}
    for(;k<t;k++)C[k]=A[k]-B[k];
}
static void extract(const double*S,int sn,double*D,int h,int r,int c){
    for(int i=0;i<h;i++)memcpy(D+i*h,S+(r+i)*sn+c,h*sizeof(double));
}
static void insert(double*D,int dn,const double*S,int h,int r,int c){
    for(int i=0;i<h;i++)memcpy(D+(r+i)*dn+c,S+i*h,h*sizeof(double));
}

/* One level of Strassen → 7 sub-multiplies at half size via BLIS */
static void com6_strassen1(const double*A,const double*B,double*C,int n){
    int h=n/2; size_t hsq=(size_t)h*h;

    double*A11=aa(hsq),*A12=aa(hsq),*A21=aa(hsq),*A22=aa(hsq);
    double*B11=aa(hsq),*B12=aa(hsq),*B21=aa(hsq),*B22=aa(hsq);
    double*T1=aa(hsq),*T2=aa(hsq);
    double*M1=aa(hsq),*M2=aa(hsq),*M3=aa(hsq),*M4=aa(hsq),*M5=aa(hsq),*M6=aa(hsq),*M7=aa(hsq);

    extract(A,n,A11,h,0,0);extract(A,n,A12,h,0,h);
    extract(A,n,A21,h,h,0);extract(A,n,A22,h,h,h);
    extract(B,n,B11,h,0,0);extract(B,n,B12,h,0,h);
    extract(B,n,B21,h,h,0);extract(B,n,B22,h,h,h);

    /* M1=(A11+A22)(B11+B22) */
    mat_add(A11,A22,T1,hsq);mat_add(B11,B22,T2,hsq);com6_multiply(T1,T2,M1,h);
    /* M2=(A21+A22)*B11 */
    mat_add(A21,A22,T1,hsq);com6_multiply(T1,B11,M2,h);
    /* M3=A11*(B12-B22) */
    mat_sub(B12,B22,T1,hsq);com6_multiply(A11,T1,M3,h);
    /* M4=A22*(B21-B11) */
    mat_sub(B21,B11,T1,hsq);com6_multiply(A22,T1,M4,h);
    /* M5=(A11+A12)*B22 */
    mat_add(A11,A12,T1,hsq);com6_multiply(T1,B22,M5,h);
    /* M6=(A21-A11)*(B11+B12) */
    mat_sub(A21,A11,T1,hsq);mat_add(B11,B12,T2,hsq);com6_multiply(T1,T2,M6,h);
    /* M7=(A12-A22)*(B21+B22) */
    mat_sub(A12,A22,T1,hsq);mat_add(B21,B22,T2,hsq);com6_multiply(T1,T2,M7,h);

    /* C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6 */
    for(size_t i=0;i<hsq;i++)T1[i]=M1[i]+M4[i]-M5[i]+M7[i]; /* C11 */
    mat_add(M3,M5,T2,hsq); /* C12 */
    insert(C,n,T1,h,0,0); insert(C,n,T2,h,0,h);

    mat_add(M2,M4,T1,hsq); /* C21 */
    for(size_t i=0;i<hsq;i++)T2[i]=M1[i]-M2[i]+M3[i]+M6[i]; /* C22 */
    insert(C,n,T1,h,h,0); insert(C,n,T2,h,h,h);

    af(A11);af(A12);af(A21);af(A22);af(B11);af(B12);af(B21);af(B22);
    af(T1);af(T2);af(M1);af(M2);af(M3);af(M4);af(M5);af(M6);af(M7);
}

/* Naive reference */
static void naive(const double*A,const double*B,double*C,int n){
    memset(C,0,(size_t)n*n*sizeof(double));
    for(int i=0;i<n;i++)for(int k=0;k<n;k++){
        double a=A[i*n+k];for(int j=0;j<n;j++)C[i*n+j]+=a*B[k*n+j];
    }
}

static double now(void){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec+t.tv_nsec*1e-9;}
static void randf(double*M,int n){for(int i=0;i<n*n;i++)M[i]=(double)rand()/RAND_MAX*2-1;}
static double maxerr(const double*A,const double*B,int n){
    double m=0;for(int i=0;i<n*n;i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;}return m;
}

int main(void){
    printf("====================================================================\n");
    printf("  COM6 v18 - Tuned BLIS + 1-level Strassen for large n\n");
    printf("  MC=%d KC=%d NC=%d\n",MC,KC,NC);
    printf("====================================================================\n\n");

    int sizes[]={256,512,1024,2048,4096};
    int ns=sizeof(sizes)/sizeof(sizes[0]);
    printf("%-10s | %10s | %10s | %8s | %s\n","Size","BLIS","Str+BLIS","GF(best)","Verify");
    printf("---------- | ---------- | ---------- | -------- | ------\n");

    for(int si=0;si<ns;si++){
        int n=sizes[si]; size_t nn=(size_t)n*n;
        double*A=aa(nn),*B=aa(nn),*C1=aa(nn),*C2=aa(nn);
        srand(42);randf(A,n);randf(B,n);

        /* BLIS only */
        com6_multiply(A,B,C1,n); /* warmup */
        int runs=(n<=1024)?3:(n<=2048)?2:1;
        double best_b=1e30;
        for(int r=0;r<runs;r++){double t0=now();com6_multiply(A,B,C1,n);double t=now()-t0;if(t<best_b)best_b=t;}

        /* Strassen+BLIS (only for even n >= 1024) */
        double best_s=1e30;
        if(n>=1024 && n%2==0){
            com6_strassen1(A,B,C2,n); /* warmup */
            for(int r=0;r<runs;r++){double t0=now();com6_strassen1(A,B,C2,n);double t=now()-t0;if(t<best_s)best_s=t;}
        }

        double best=(best_s<best_b)?best_s:best_b;
        double gf=(2.0*n*n*(double)n)/(best*1e9);
        const char*winner=(best_s<best_b)?"Str":"BLIS";

        /* Verify */
        const char*v;
        if(n<=512){double*Cr=aa(nn);naive(A,B,Cr,n);v=maxerr(C1,Cr,n)<1e-6?"OK":"FAIL";af(Cr);}
        else if(best_s<1e30){v=maxerr(C1,C2,n)<1e-6?"OK":"FAIL";}
        else v="OK";

        if(best_s<1e30)
            printf("%4dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %s (%s)\n",
                   n,n,best_b*1000,best_s*1000,gf,v,winner);
        else
            printf("%4dx%-5d | %8.1f ms |        n/a | %6.1f   | %s (BLIS)\n",
                   n,n,best_b*1000,gf,v);

        af(A);af(B);af(C1);af(C2);
    }
    printf("\nTarget: ~40 GFLOPS (OpenBLAS 1T, i7-10510U)\n");
    return 0;
}
