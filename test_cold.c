/* Cold single-shot: just run one 8192 with no warmup */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define MR 6
#define NR 8
#define ALIGN 64
#define KC_MAX 384
#define MC_MAX 120
static inline double* aa(size_t c){return(double*)_mm_malloc(c*sizeof(double),ALIGN);}
static inline void af(double*p){_mm_free(p);}
static double now(void){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec+t.tv_nsec*1e-9;}

static void __attribute__((noinline))
micro_6x8(int kc, const double* pA, const double* pB, double* C, int ldc){
    long long kc_8=kc>>3,kc_rem=kc&7,ldc_bytes=(long long)ldc*8;
    __asm__ volatile(
        "vxorpd %%ymm0,%%ymm0,%%ymm0\n\t" "vxorpd %%ymm1,%%ymm1,%%ymm1\n\t"
        "vxorpd %%ymm2,%%ymm2,%%ymm2\n\t" "vxorpd %%ymm3,%%ymm3,%%ymm3\n\t"
        "vxorpd %%ymm4,%%ymm4,%%ymm4\n\t" "vxorpd %%ymm5,%%ymm5,%%ymm5\n\t"
        "vxorpd %%ymm6,%%ymm6,%%ymm6\n\t" "vxorpd %%ymm7,%%ymm7,%%ymm7\n\t"
        "vxorpd %%ymm8,%%ymm8,%%ymm8\n\t" "vxorpd %%ymm9,%%ymm9,%%ymm9\n\t"
        "vxorpd %%ymm10,%%ymm10,%%ymm10\n\t" "vxorpd %%ymm11,%%ymm11,%%ymm11\n\t"
        "testq %[kc8],%[kc8]\n\t" "jle 3f\n\t" ".p2align 5\n\t" "1:\n\t"
        "prefetcht0 768(%[pA])\n\t" "prefetcht0 1024(%[pB])\n\t"
#define R1(AO,BO) \
        "vmovapd " #BO "(%[pB]),%%ymm12\n\t" "vmovapd " #BO "+32(%[pB]),%%ymm13\n\t" \
        "vbroadcastsd " #AO "(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t" \
        "vbroadcastsd " #AO "+8(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t" \
        "vbroadcastsd " #AO "+16(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t" \
        "vbroadcastsd " #AO "+24(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t" \
        "vbroadcastsd " #AO "+32(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t" \
        "vbroadcastsd " #AO "+40(%[pA]),%%ymm14\n\t" \
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"
        R1(0,0) R1(48,64) R1(96,128) R1(144,192)
        "prefetcht0 1152(%[pA])\n\t" "prefetcht0 1536(%[pB])\n\t"
        R1(192,256) R1(240,320) R1(288,384) R1(336,448)
#undef R1
        "addq $384,%[pA]\n\t" "addq $512,%[pB]\n\t" "decq %[kc8]\n\t" "jnz 1b\n\t"
        "3:\n\t" "testq %[kcr],%[kcr]\n\t" "jle 2f\n\t" "4:\n\t"
        "vmovapd (%[pB]),%%ymm12\n\t" "vmovapd 32(%[pB]),%%ymm13\n\t"
        "vbroadcastsd (%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 8(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 16(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 24(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 32(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 40(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t" "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"
        "addq $48,%[pA]\n\t" "addq $64,%[pB]\n\t" "decq %[kcr]\n\t" "jnz 4b\n\t"
        "2:\n\t"
        "vaddpd (%[C]),%%ymm0,%%ymm0\n\t" "vmovupd %%ymm0,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm1,%%ymm1\n\t" "vmovupd %%ymm1,32(%[C])\n\t" "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm2,%%ymm2\n\t" "vmovupd %%ymm2,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm3,%%ymm3\n\t" "vmovupd %%ymm3,32(%[C])\n\t" "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm4,%%ymm4\n\t" "vmovupd %%ymm4,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm5,%%ymm5\n\t" "vmovupd %%ymm5,32(%[C])\n\t" "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm6,%%ymm6\n\t" "vmovupd %%ymm6,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm7,%%ymm7\n\t" "vmovupd %%ymm7,32(%[C])\n\t" "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm8,%%ymm8\n\t" "vmovupd %%ymm8,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm9,%%ymm9\n\t" "vmovupd %%ymm9,32(%[C])\n\t" "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm10,%%ymm10\n\t" "vmovupd %%ymm10,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm11,%%ymm11\n\t" "vmovupd %%ymm11,32(%[C])\n\t"
        :[pA]"+r"(pA),[pB]"+r"(pB),[kc8]"+r"(kc_8),[kcr]"+r"(kc_rem),[C]"+r"(C)
        :[ldc]"r"(ldc_bytes)
        :"ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
         "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","memory");
}
static void micro_edge(int mr,int nr,int kc,const double*pA,const double*pB,double*C,int n){
    for(int k=0;k<kc;k++)for(int i=0;i<mr;i++){double av=pA[k*MR+i];for(int j=0;j<nr;j++)C[i*n+j]+=av*pB[k*NR+j];}}
static void pack_B_chunk(const double*B,double*pb,int kc,int nc,int n,int j0,int k0,int js,int je){
    for(int j=js;j<je;j+=NR){int nr=(j+NR<=nc)?NR:nc-j;
        const double*Bkj=B+(size_t)k0*n+(j0+j);double*d=pb+(j/NR)*((size_t)NR*kc);
        if(nr==NR){for(int k=0;k<kc;k++){_mm256_store_pd(d,_mm256_loadu_pd(Bkj));
            _mm256_store_pd(d+4,_mm256_loadu_pd(Bkj+4));d+=NR;Bkj+=n;}}
        else{for(int k=0;k<kc;k++){int jj;for(jj=0;jj<nr;jj++)d[jj]=Bkj[jj];for(;jj<NR;jj++)d[jj]=0;d+=NR;Bkj+=n;}}}}
static void pack_A(const double*A,double*pa,int mc,int kc,int n,int i0,int k0){
    const double*Ab=A+(size_t)i0*n+k0;
    for(int i=0;i<mc;i+=MR){int mr=(i+MR<=mc)?MR:mc-i;
        if(mr==MR){const double*a0=Ab+i*n,*a1=a0+n,*a2=a0+2*n,*a3=a0+3*n,*a4=a0+4*n,*a5=a0+5*n;
            int k=0;for(;k+1<kc;k+=2){pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];
                pa[6]=a0[k+1];pa[7]=a1[k+1];pa[8]=a2[k+1];pa[9]=a3[k+1];pa[10]=a4[k+1];pa[11]=a5[k+1];pa+=2*MR;}
            for(;k<kc;k++){pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];pa+=MR;}}
        else{for(int k=0;k<kc;k++){int ii;for(ii=0;ii<mr;ii++)pa[ii]=(Ab+(i+ii)*n)[k];for(;ii<MR;ii++)pa[ii]=0;pa+=MR;}}}}
static void macro_kernel(const double*pa,const double*pb,double*C,int mc,int nc,int kc,int n,int ic,int jc){
    for(int jr=0;jr<nc;jr+=NR){int nr=(jr+NR<=nc)?NR:nc-jr;
        const double*pB=pb+(jr/NR)*((size_t)NR*kc);
        for(int ir=0;ir<mc;ir+=MR){int mr=(ir+MR<=mc)?MR:mc-ir;
            const double*pA=pa+(ir/MR)*((size_t)MR*kc);
            double*Cij=C+(size_t)(ic+ir)*n+(jc+jr);
            if(mr==MR&&nr==NR)micro_6x8(kc,pA,pB,Cij,n);else micro_edge(mr,nr,kc,pA,pB,Cij,n);}}}
static void randf(double*M,int n){for(int i=0;i<n*n;i++)M[i]=(double)rand()/RAND_MAX*2-1;}

int main(int argc, char**argv){
    int n = (argc>1) ? atoi(argv[1]) : 8192;
    int nth = omp_get_max_threads();
    int mc=72, kc=384, nc=1024;
    if(n<=512){mc=120;kc=256;nc=2048;}
    else if(n<=4096){mc=96;kc=320;nc=2048;}

    printf("%dx%d cold-start, %d threads, MC=%d KC=%d NC=%d\n",n,n,nth,mc,kc,nc);
    size_t nn=(size_t)n*n;
    double*A=aa(nn),*B=aa(nn),*C=aa(nn);
    srand(42);randf(A,n);randf(B,n);
    memset(C,0,nn*sizeof(double));

    double** pa_bufs=(double**)malloc(nth*sizeof(double*));
    for(int t=0;t<nth;t++) pa_bufs[t]=aa((size_t)MC_MAX*KC_MAX);
    double*pb=aa((size_t)KC_MAX*nc);

    double t0=now();
    for(int jc=0;jc<n;jc+=nc){int ncb=(jc+nc<=n)?nc:n-jc;
        for(int pc=0;pc<n;pc+=kc){int kcb=(pc+kc<=n)?kc:n-pc;
            #pragma omp parallel
            {int tid=omp_get_thread_num();int nt=omp_get_num_threads();double*pa=pa_bufs[tid];
                int np=(ncb+NR-1)/NR,pp=(np+nt-1)/nt,p0=tid*pp,p1=p0+pp;
                if(p1>np)p1=np;int js=p0*NR,je=p1*NR;if(je>ncb)je=ncb;
                if(js<ncb)pack_B_chunk(B,pb,kcb,ncb,n,jc,pc,js,je);
                #pragma omp barrier
                #pragma omp for schedule(dynamic,2)
                for(int ic=0;ic<n;ic+=mc){int mcb=(ic+mc<=n)?mc:n-ic;
                    pack_A(A,pa,mcb,kcb,n,ic,pc);
                    macro_kernel(pa,pb,C,mcb,ncb,kcb,n,ic,jc);}}}}
    double elapsed=now()-t0;
    double flops=2.0*n*n*(double)n;
    printf("Time: %.1f ms   %.1f GF/s\n",elapsed*1000,flops/(elapsed*1e9));

    for(int t=0;t<nth;t++) af(pa_bufs[t]);
    free(pa_bufs);af(pb);af(A);af(B);af(C);
    return 0;
}
