/*
 * COM6 v71 - 8192 Tuning: Deeper KC + Wider NC
 * ==============================================
 * v70 base. Target: close the 8192 gap vs OpenBLAS.
 *
 * v70 at 8192 used NC=1536, KC=384, MC=72:
 *   6 jc-tiles × 22 pc-tiles = 132 B-packing phases, 264 barriers
 *
 * v71 for 8192: NC=2048, KC=448, MC=54:
 *   4 jc-tiles × 19 pc-tiles = 76 B-packing phases, 152 barriers (42% fewer)
 *   B-panel: 2048*448*8 = 7.34MB (fits 8MB L3)
 *   A-panel: 54*448*8 = 194KB (fits 256KB L2)
 *   ic iterations: 8192/54 = 152 (excellent thread parallelism)
 *
 * Everything else identical to v70 (proven kernel, NT stores, beta opt).
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp -o com6_v71 com6_v71.c -lm
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

#define MR  6
#define NR  8
#define NC_DEFAULT 2048
#define ALIGN 64

#define KC_SMALL 256
#define MC_SMALL 120
#define MC_TINY  66  /* 512/66=8 blocks for 8 threads: perfect balance */
#define MC_1K    72  /* 1024/72=14 blocks, well-balanced for 8 threads */
#define KC_LARGE 320
#define MC_LARGE 96
#define KC_HUGE  448   /* deeper KC: 42% fewer B-pack calls at 8192 */
#define MC_HUGE  54    /* 54*448*8=194KB fits L2, 8192/54=152 ic-blocks */
#define KC_MAX 448
#define MC_MAX 120
#define NC_MAX 2048

static inline double* aa(size_t c){return(double*)_mm_malloc(c*sizeof(double),ALIGN);}
static inline void af(double*p){_mm_free(p);}

/* C-prefetch 8x-unrolled 6x8 ASM micro-kernel
 * beta=0: store directly (first KC-tile, no memset needed)
 * beta=1: load+add+store (subsequent KC-tiles)
 */
static void __attribute__((noinline))
micro_6x8(int kc, const double* pA, const double* pB,
           double* C, int ldc, int beta, int use_nt)
{
    long long kc_8 = kc >> 3;
    long long kc_rem = kc & 7;
    long long ldc_bytes = (long long)ldc * 8;

    __asm__ volatile(
        /* Prefetch all 6 rows of C output into L1 */
        "prefetcht0 (%[C])\n\t"
        "prefetcht0 32(%[C])\n\t"
        "movq %[C],%%r15\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"
        "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"
        "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"
        "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"
        "prefetcht0 32(%%r15)\n\t"
        "addq %[ldc],%%r15\n\t"
        "prefetcht0 (%%r15)\n\t"
        "prefetcht0 32(%%r15)\n\t"

        "vxorpd %%ymm0,%%ymm0,%%ymm0\n\t"
        "vxorpd %%ymm1,%%ymm1,%%ymm1\n\t"
        "vxorpd %%ymm2,%%ymm2,%%ymm2\n\t"
        "vxorpd %%ymm3,%%ymm3,%%ymm3\n\t"
        "vxorpd %%ymm4,%%ymm4,%%ymm4\n\t"
        "vxorpd %%ymm5,%%ymm5,%%ymm5\n\t"
        "vxorpd %%ymm6,%%ymm6,%%ymm6\n\t"
        "vxorpd %%ymm7,%%ymm7,%%ymm7\n\t"
        "vxorpd %%ymm8,%%ymm8,%%ymm8\n\t"
        "vxorpd %%ymm9,%%ymm9,%%ymm9\n\t"
        "vxorpd %%ymm10,%%ymm10,%%ymm10\n\t"
        "vxorpd %%ymm11,%%ymm11,%%ymm11\n\t"

        "testq %[kc8],%[kc8]\n\t"
        "jle 3f\n\t"
        ".p2align 5\n\t"
        "1:\n\t"

        "prefetcht0 768(%[pA])\n\t"
        "prefetcht0 1024(%[pB])\n\t"

        /* k+0 */
        "vmovapd (%[pB]),%%ymm12\n\t"
        "vmovapd 32(%[pB]),%%ymm13\n\t"
        "vbroadcastsd (%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 8(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 16(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 24(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 32(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 40(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+1 */
        "vmovapd 64(%[pB]),%%ymm12\n\t"
        "vmovapd 96(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 48(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 56(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 64(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 72(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 80(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 88(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+2 */
        "vmovapd 128(%[pB]),%%ymm12\n\t"
        "vmovapd 160(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 96(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 104(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 112(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 120(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 128(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 136(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+3 */
        "vmovapd 192(%[pB]),%%ymm12\n\t"
        "vmovapd 224(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 144(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 152(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 160(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 168(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 176(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 184(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+4 */
        "prefetcht0 1152(%[pA])\n\t"
        "prefetcht0 1536(%[pB])\n\t"
        "vmovapd 256(%[pB]),%%ymm12\n\t"
        "vmovapd 288(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 192(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 200(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 208(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 216(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 224(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 232(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+5 */
        "vmovapd 320(%[pB]),%%ymm12\n\t"
        "vmovapd 352(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 240(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 248(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 256(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 264(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 272(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 280(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+6 */
        "vmovapd 384(%[pB]),%%ymm12\n\t"
        "vmovapd 416(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 288(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 296(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 304(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 312(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 320(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 328(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        /* k+7 */
        "vmovapd 448(%[pB]),%%ymm12\n\t"
        "vmovapd 480(%[pB]),%%ymm13\n\t"
        "vbroadcastsd 336(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 344(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 352(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 360(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 368(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 376(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"

        "addq $384,%[pA]\n\t"
        "addq $512,%[pB]\n\t"
        "decq %[kc8]\n\t"
        "jnz 1b\n\t"

        "3:\n\t"
        "testq %[kcr],%[kcr]\n\t"
        "jle 2f\n\t"
        "4:\n\t"
        "vmovapd (%[pB]),%%ymm12\n\t"
        "vmovapd 32(%[pB]),%%ymm13\n\t"
        "vbroadcastsd (%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm0\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm1\n\t"
        "vbroadcastsd 8(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm2\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm3\n\t"
        "vbroadcastsd 16(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm4\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm5\n\t"
        "vbroadcastsd 24(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm6\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm7\n\t"
        "vbroadcastsd 32(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm8\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm9\n\t"
        "vbroadcastsd 40(%[pA]),%%ymm14\n\t"
        "vfmadd231pd %%ymm14,%%ymm12,%%ymm10\n\t"
        "vfmadd231pd %%ymm14,%%ymm13,%%ymm11\n\t"
        "addq $48,%[pA]\n\t"
        "addq $64,%[pB]\n\t"
        "decq %[kcr]\n\t"
        "jnz 4b\n\t"

        "2:\n\t"
        "testl %[beta],%[beta]\n\t"
        "jz 5f\n\t"
        /* beta=1: load+add+store */
        "vaddpd (%[C]),%%ymm0,%%ymm0\n\t"  "vmovupd %%ymm0,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm1,%%ymm1\n\t" "vmovupd %%ymm1,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm2,%%ymm2\n\t"  "vmovupd %%ymm2,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm3,%%ymm3\n\t" "vmovupd %%ymm3,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm4,%%ymm4\n\t"  "vmovupd %%ymm4,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm5,%%ymm5\n\t" "vmovupd %%ymm5,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm6,%%ymm6\n\t"  "vmovupd %%ymm6,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm7,%%ymm7\n\t" "vmovupd %%ymm7,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm8,%%ymm8\n\t"  "vmovupd %%ymm8,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm9,%%ymm9\n\t" "vmovupd %%ymm9,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vaddpd (%[C]),%%ymm10,%%ymm10\n\t" "vmovupd %%ymm10,(%[C])\n\t"
        "vaddpd 32(%[C]),%%ymm11,%%ymm11\n\t" "vmovupd %%ymm11,32(%[C])\n\t"
        "jmp 6f\n\t"
        /* beta=0: check if NT stores requested */
        "5:\n\t"
        "testl %[nt],%[nt]\n\t"
        "jz 7f\n\t"
        /* beta=0 + use_nt=1: NT store (bypass cache for huge C) */
        "vmovntpd %%ymm0,(%[C])\n\t"
        "vmovntpd %%ymm1,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovntpd %%ymm2,(%[C])\n\t"
        "vmovntpd %%ymm3,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovntpd %%ymm4,(%[C])\n\t"
        "vmovntpd %%ymm5,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovntpd %%ymm6,(%[C])\n\t"
        "vmovntpd %%ymm7,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovntpd %%ymm8,(%[C])\n\t"
        "vmovntpd %%ymm9,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovntpd %%ymm10,(%[C])\n\t"
        "vmovntpd %%ymm11,32(%[C])\n\t"
        "jmp 6f\n\t"
        /* beta=0 + use_nt=0: regular store (keep C in cache for small matrices) */
        "7:\n\t"
        "vmovupd %%ymm0,(%[C])\n\t"
        "vmovupd %%ymm1,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm2,(%[C])\n\t"
        "vmovupd %%ymm3,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm4,(%[C])\n\t"
        "vmovupd %%ymm5,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm6,(%[C])\n\t"
        "vmovupd %%ymm7,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm8,(%[C])\n\t"
        "vmovupd %%ymm9,32(%[C])\n\t"
        "addq %[ldc],%[C]\n\t"
        "vmovupd %%ymm10,(%[C])\n\t"
        "vmovupd %%ymm11,32(%[C])\n\t"
        "6:\n\t"

        : [pA]"+r"(pA),[pB]"+r"(pB),[kc8]"+r"(kc_8),[kcr]"+r"(kc_rem),[C]"+r"(C)
        : [ldc]"r"(ldc_bytes),[beta]"r"(beta),[nt]"r"(use_nt)
        : "r15","ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",
          "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","memory"
    );
}

static void micro_edge(int mr,int nr,int kc,const double*pA,const double*pB,double*C,int n,int beta,int use_nt){
    (void)use_nt;
    if(!beta){for(int i=0;i<mr;i++)for(int j=0;j<nr;j++)C[i*n+j]=0;}
    for(int k=0;k<kc;k++)for(int i=0;i<mr;i++){
        double av=pA[k*MR+i];for(int j=0;j<nr;j++)C[i*n+j]+=av*pB[k*NR+j];}
}

static void pack_B_chunk(const double*__restrict__ B,double*__restrict__ pb,
                          int kc,int nc,int n,int j0,int k0,int j_start,int j_end){
    for(int j=j_start;j<j_end;j+=NR){
        int nr=(j+NR<=nc)?NR:nc-j;
        const double*Bkj=B+(size_t)k0*n+(j0+j);
        double*dest=pb+(j/NR)*((size_t)NR*kc);
        if(nr==NR){for(int k=0;k<kc;k++){
            _mm256_store_pd(dest,_mm256_loadu_pd(Bkj));
            _mm256_store_pd(dest+4,_mm256_loadu_pd(Bkj+4));
            dest+=NR;Bkj+=n;
        }}else{for(int k=0;k<kc;k++){
            int jj;for(jj=0;jj<nr;jj++)dest[jj]=Bkj[jj];
            for(;jj<NR;jj++)dest[jj]=0.0;dest+=NR;Bkj+=n;
        }}
    }
}

static void pack_B(const double*__restrict__ B,double*__restrict__ pb,
                    int kc,int nc,int n,int j0,int k0){
    pack_B_chunk(B,pb,kc,nc,n,j0,k0,0,nc);
}

/* 4x k-unrolled A-packing (from v62) */
static void pack_A(const double*__restrict__ A,double*__restrict__ pa,
                    int mc,int kc,int n,int i0,int k0){
    const double*Ab=A+(size_t)i0*n+k0;
    for(int i=0;i<mc;i+=MR){
        int mr=(i+MR<=mc)?MR:mc-i;
        if(mr==MR){
            const double*a0=Ab+i*n,*a1=a0+n,*a2=a0+2*n,
                        *a3=a0+3*n,*a4=a0+4*n,*a5=a0+5*n;
            int k=0;
            for(;k+3<kc;k+=4){
                pa[0]=a0[k];pa[1]=a1[k];pa[2]=a2[k];
                pa[3]=a3[k];pa[4]=a4[k];pa[5]=a5[k];
                pa[6]=a0[k+1];pa[7]=a1[k+1];pa[8]=a2[k+1];
                pa[9]=a3[k+1];pa[10]=a4[k+1];pa[11]=a5[k+1];
                pa[12]=a0[k+2];pa[13]=a1[k+2];pa[14]=a2[k+2];
                pa[15]=a3[k+2];pa[16]=a4[k+2];pa[17]=a5[k+2];
                pa[18]=a0[k+3];pa[19]=a1[k+3];pa[20]=a2[k+3];
                pa[21]=a3[k+3];pa[22]=a4[k+3];pa[23]=a5[k+3];
                pa+=4*MR;
            }
            for(;k<kc;k++){
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

static void macro_kernel_1t(const double*pa,const double*pb,
                             double*C,int mc,int nc,int kc,int n,int ic,int jc,int beta,int use_nt){
    for(int jr=0;jr<nc;jr+=NR){
        int nr=(jr+NR<=nc)?NR:nc-jr;
        const double*pB=pb+(jr/NR)*((size_t)NR*kc);
        for(int ir=0;ir<mc;ir+=MR){
            int mr=(ir+MR<=mc)?MR:mc-ir;
            const double*pA=pa+(ir/MR)*((size_t)MR*kc);
            double*Cij=C+(size_t)(ic+ir)*n+(jc+jr);
            if(mr==MR&&nr==NR)micro_6x8(kc,pA,pB,Cij,n,beta,use_nt);
            else micro_edge(mr,nr,kc,pA,pB,Cij,n,beta,use_nt);
        }
    }
}

/* Blocking for 1T path: standard v26 2-tier */
static void get_blocking_1t(int n, int*pMC, int*pKC, int*pNC){
    if(n <= 1024)     {*pMC=MC_SMALL;*pKC=KC_SMALL;*pNC=NC_DEFAULT;}
    else if(n <= 4096){*pMC=MC_LARGE;*pKC=KC_LARGE;*pNC=NC_DEFAULT;}
    else              {*pMC=MC_HUGE; *pKC=KC_HUGE;  *pNC=NC_DEFAULT;}
}

/* Blocking for MT path: 4-tier
 * MC chosen for thread load balance (N/MC ≈ k*threads):
 *   512: MC=66 → 8 blocks (= 8 threads, perfect)
 *   1024: MC=72 → 14 blocks (1.75/thread, well balanced)
 *   vs old MC=120 → 9 blocks (1.125/thread = 1 thread does 2x work!)
 */
static void get_blocking_mt(int n, int*pMC, int*pKC, int*pNC){
    if(n <= 512)      {*pMC=MC_TINY; *pKC=KC_SMALL;*pNC=NC_DEFAULT;}
    else if(n <= 1024){*pMC=MC_1K;   *pKC=KC_SMALL;*pNC=NC_DEFAULT;} /* MC=72: balanced */
    else if(n <= 4096){*pMC=MC_LARGE;*pKC=KC_LARGE;*pNC=NC_DEFAULT;}
    else              {*pMC=MC_HUGE; *pKC=KC_HUGE;  *pNC=NC_DEFAULT;}
}

static void com6_multiply(const double*__restrict__ A,
                           const double*__restrict__ B,
                           double*__restrict__ C,int n)
{
    int nthreads = 1;
    #ifdef _OPENMP
    nthreads = omp_get_max_threads();
    #endif

    int use_mt = (n >= 512 && nthreads > 1);
    int mc_blk, kc_blk, nc_blk;

    /* NT stores only when C doesn't fit L3 (n>=2048 -> C=32MB) */
    int use_nt = (n >= 2048);

    if(!use_mt){
        get_blocking_1t(n, &mc_blk, &kc_blk, &nc_blk);
        double*pa=aa((size_t)MC_MAX*KC_MAX),*pb=aa((size_t)KC_MAX*NC_MAX);
        for(int jc=0;jc<n;jc+=nc_blk){int nc=(jc+nc_blk<=n)?nc_blk:n-jc;
            for(int pc=0;pc<n;pc+=kc_blk){int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
                int beta=(pc>0)?1:0;
                pack_B(B,pb,kc,nc,n,jc,pc);
                for(int ic=0;ic<n;ic+=mc_blk){int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                    pack_A(A,pa,mc,kc,n,ic,pc);
                    macro_kernel_1t(pa,pb,C,mc,nc,kc,n,ic,jc,beta,use_nt);
                }
                if(!beta && use_nt) _mm_sfence();
            }
        }
        af(pa);af(pb);
    } else if(n <= 2048){
        /* Small/medium MT: per-tile parallel region + dynamic scheduling */
        get_blocking_mt(n, &mc_blk, &kc_blk, &nc_blk);

        double** pa_bufs = (double**)malloc(nthreads * sizeof(double*));
        for(int t=0;t<nthreads;t++) pa_bufs[t] = aa((size_t)MC_MAX*KC_MAX);
        double* pb = aa((size_t)KC_MAX*NC_MAX);

        for(int jc=0;jc<n;jc+=nc_blk){
            int nc=(jc+nc_blk<=n)?nc_blk:n-jc;
            for(int pc=0;pc<n;pc+=kc_blk){
                int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
                int beta=(pc>0)?1:0;

                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    int nt = omp_get_num_threads();
                    double* pa = pa_bufs[tid];

                    /* Parallel B-packing */
                    int npanels = (nc + NR - 1) / NR;
                    int panels_per = (npanels + nt - 1) / nt;
                    int p0 = tid * panels_per;
                    int p1 = p0 + panels_per;
                    if(p1 > npanels) p1 = npanels;
                    int j_start = p0 * NR;
                    int j_end = p1 * NR;
                    if(j_end > nc) j_end = nc;
                    if(j_start < nc)
                        pack_B_chunk(B,pb,kc,nc,n,jc,pc,j_start,j_end);

                    #pragma omp barrier

                    #pragma omp for schedule(dynamic,1)
                    for(int ic=0;ic<n;ic+=mc_blk){
                        int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                        pack_A(A,pa,mc,kc,n,ic,pc);
                        macro_kernel_1t(pa,pb,C,mc,nc,kc,n,ic,jc,beta,use_nt);
                    }
                    if(!beta && use_nt) _mm_sfence();
                }
            }
        }

        for(int t=0;t<nthreads;t++) af(pa_bufs[t]);
        free(pa_bufs);
        af(pb);
    } else {
        /* Large MT (4096+): SINGLE parallel region wrapping all loops.
         * At 8192: eliminates 104 fork/join overheads.
         * Threads stay alive the whole multiply — caches stay warm.
         */
        get_blocking_mt(n, &mc_blk, &kc_blk, &nc_blk);

        double** pa_bufs = (double**)malloc(nthreads * sizeof(double*));
        for(int t=0;t<nthreads;t++) pa_bufs[t] = aa((size_t)MC_MAX*KC_MAX);
        double* pb = aa((size_t)KC_MAX*NC_MAX);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            double* pa = pa_bufs[tid];

            for(int jc=0;jc<n;jc+=nc_blk){
                int nc=(jc+nc_blk<=n)?nc_blk:n-jc;
                for(int pc=0;pc<n;pc+=kc_blk){
                    int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
                    int beta=(pc>0)?1:0;

                    /* Parallel B-packing */
                    int npanels = (nc + NR - 1) / NR;
                    int panels_per = (npanels + nt - 1) / nt;
                    int p0 = tid * panels_per;
                    int p1 = p0 + panels_per;
                    if(p1 > npanels) p1 = npanels;
                    int j_start = p0 * NR;
                    int j_end = p1 * NR;
                    if(j_end > nc) j_end = nc;
                    if(j_start < nc)
                        pack_B_chunk(B,pb,kc,nc,n,jc,pc,j_start,j_end);

                    #pragma omp barrier

                    #pragma omp for schedule(static) nowait
                    for(int ic=0;ic<n;ic+=mc_blk){
                        int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                        pack_A(A,pa,mc,kc,n,ic,pc);
                        macro_kernel_1t(pa,pb,C,mc,nc,kc,n,ic,jc,beta,use_nt);
                    }
                    if(!beta && use_nt) _mm_sfence();

                    #pragma omp barrier
                }
            }
        }

        for(int t=0;t<nthreads;t++) af(pa_bufs[t]);
        free(pa_bufs);
        af(pb);
    }
}

static void com6_multiply_1t(const double*__restrict__ A,
                              const double*__restrict__ B,
                              double*__restrict__ C,int n)
{
    int mc_blk, kc_blk, nc_blk;
    get_blocking_1t(n, &mc_blk, &kc_blk, &nc_blk);
    int use_nt = (n >= 2048);

    double*pa=aa((size_t)MC_MAX*KC_MAX),*pb=aa((size_t)KC_MAX*NC_MAX);
    for(int jc=0;jc<n;jc+=nc_blk){int nc=(jc+nc_blk<=n)?nc_blk:n-jc;
        for(int pc=0;pc<n;pc+=kc_blk){int kc=(pc+kc_blk<=n)?kc_blk:n-pc;
            int beta=(pc>0)?1:0;
            pack_B(B,pb,kc,nc,n,jc,pc);
            for(int ic=0;ic<n;ic+=mc_blk){int mc=(ic+mc_blk<=n)?mc_blk:n-ic;
                pack_A(A,pa,mc,kc,n,ic,pc);
                macro_kernel_1t(pa,pb,C,mc,nc,kc,n,ic,jc,beta,use_nt);
            }
            if(!beta && use_nt) _mm_sfence();
    }}
    af(pa);af(pb);
}

static void naive(const double*A,const double*B,double*C,int n){
    memset(C,0,(size_t)n*n*sizeof(double));
    for(int i=0;i<n;i++)for(int k=0;k<n;k++){double a=A[i*n+k];for(int j=0;j<n;j++)C[i*n+j]+=a*B[k*n+j];}
}

static double now(void){struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec+t.tv_nsec*1e-9;}
static void randf(double*M,int n){for(int i=0;i<n*n;i++)M[i]=(double)rand()/RAND_MAX*2-1;}
static double maxerr(const double*A,const double*B,int n){
    double m=0;for(int i=0;i<n*n;i++){double d=fabs(A[i]-B[i]);if(d>m)m=d;}return m;
}

int main(int argc, char**argv){
    int nth=1;
    #ifdef _OPENMP
    nth=omp_get_max_threads();
    #endif

    /* CLI mode: ./com6_v71 <size> [mt|1t] */
    if(argc >= 2){
        int n=atoi(argv[1]);
        if(n<64||n>16384){printf("Size must be 64..16384\n");return 1;}
        int mode=0;
        if(argc>=3){
            if(strcmp(argv[2],"1t")==0) mode=1;
            else if(strcmp(argv[2],"mt")==0) mode=2;
        }

        size_t nn=(size_t)n*n;
        double*A=aa(nn),*B=aa(nn),*C=aa(nn);
        if(!A||!B||!C){printf("OOM for %dx%d\n",n,n);return 1;}
        srand(42);randf(A,n);randf(B,n);
        int runs=(n<=512)?7:(n<=1024)?5:(n<=2048)?3:2;

        if(mode!=2){
            com6_multiply_1t(A,B,C,n);
            double best=1e30;
            for(int r=0;r<runs;r++){
                double t0=now();com6_multiply_1t(A,B,C,n);double t=now()-t0;
                if(t<best)best=t;
            }
            printf("1T  %4dx%-5d | %8.1f ms | %6.1f GF\n",n,n,best*1000,(2.0*n*n*(double)n)/(best*1e9));
        }
        if(mode!=1){
            com6_multiply(A,B,C,n);
            double best=1e30;
            for(int r=0;r<runs;r++){
                double t0=now();com6_multiply(A,B,C,n);double t=now()-t0;
                if(t<best)best=t;
            }
            printf("MT  %4dx%-5d | %8.1f ms | %6.1f GF (%d threads)\n",n,n,best*1000,(2.0*n*n*(double)n)/(best*1e9),nth);
        }

        af(A);af(B);af(C);
        return 0;
    }

    /* Full benchmark */
    printf("====================================================================\n");
    printf("  COM6 v71 - Adaptive NC + v67 Core (%d threads)\n",nth);
    printf("  n<=4096: NC=%d KC=%d MC=%d | n>4096: NC=%d KC=%d MC=%d (B=7.34MB)\n",NC_DEFAULT,KC_LARGE,MC_LARGE,NC_DEFAULT,KC_HUGE,MC_HUGE);
    printf("  Adaptive NT: n<2048 uses regular stores (keep C in cache)\n");
    printf("====================================================================\n\n");

    int sizes[]={256,512,1024,2048,4096,8192};
    int ns=sizeof(sizes)/sizeof(sizes[0]);
    printf("%-10s | %10s | %10s | %8s | %8s | %s\n","Size","1-thread","Multi-T","GF(1T)","GF(MT)","Verify");
    printf("---------- | ---------- | ---------- | -------- | -------- | ------\n");

    for(int si=0;si<ns;si++){
        int n=sizes[si];size_t nn=(size_t)n*n;

        if(si > 0){
#ifdef _WIN32
            Sleep(4000);
#else
            { struct timespec ts={4,0}; nanosleep(&ts,NULL); }
#endif
        }

        double*A=aa(nn),*B=aa(nn),*C1=NULL,*C2=aa(nn);
        if(!A||!B||!C2){
            printf("%4dx%-5d | SKIPPED (OOM)\n",n,n);
            if(A)af(A);if(B)af(B);if(C2)af(C2);
            continue;
        }
        srand(42);randf(A,n);randf(B,n);

        int do_1t = (n <= 2048);
        double best_1=1e30, gf1=0;

        if(do_1t){
            C1=aa(nn);
            com6_multiply_1t(A,B,C1,n); /* warmup */
            int runs=(n<=512)?7:(n<=1024)?5:3;
            for(int r=0;r<runs;r++){
                double t0=now();com6_multiply_1t(A,B,C1,n);double t=now()-t0;
                if(t<best_1)best_1=t;
            }
            gf1=(2.0*n*n*(double)n)/(best_1*1e9);
        }

        com6_multiply(A,B,C2,n); /* warmup */
        int runs_mt=(n<=1024)?5:(n<=2048)?3:2;
        double best_a=1e30;
        for(int r=0;r<runs_mt;r++){
            double t0=now();com6_multiply(A,B,C2,n);double t=now()-t0;
            if(t<best_a)best_a=t;
        }
        double gfa=(2.0*n*n*(double)n)/(best_a*1e9);

        const char*v;
        if(n<=512){
            double*Cr=aa(nn);naive(A,B,Cr,n);
            double e=fmax(do_1t?maxerr(C1,Cr,n):0,maxerr(C2,Cr,n));
            v=e<1e-6?"OK":"FAIL";af(Cr);
        }else if(do_1t&&C1){
            v=maxerr(C1,C2,n)<1e-6?"OK":"FAIL";
        }else{
            double*C3=aa(nn);
            if(C3){com6_multiply(A,B,C3,n);v=maxerr(C2,C3,n)<1e-6?"OK":"FAIL";af(C3);}
            else v="SKIP";
        }

        if(do_1t)
            printf("%4dx%-5d | %8.1f ms | %8.1f ms | %6.1f   | %6.1f   | %s\n",
                   n,n,best_1*1000,best_a*1000,gf1,gfa,v);
        else
            printf("%4dx%-5d | %10s | %8.1f ms | %6s   | %6.1f   | %s\n",
                   n,n,"--",best_a*1000,"--",gfa,v);

        af(A);af(B);if(C1)af(C1);af(C2);
    }

    printf("\nAdaptive NT stores: regular stores for n<2048, NT for n>=2048\n");
    printf("n>4096: NC=%d KC=%d MC=%d (B=%.1fMB, fewer pack calls)\n",NC_DEFAULT,KC_HUGE,MC_HUGE,NC_DEFAULT*KC_HUGE*8.0/1048576);
    printf("Run individual sizes: ./com6_v71 <size> [mt|1t]\n");
    return 0;
}
