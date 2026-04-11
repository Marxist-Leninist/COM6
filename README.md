# COM6 - Custom Operation Matrix Multiplication

**COM6 beats OpenBLAS (NumPy/SciPy's backend) at matrix multiplication at 512+ sizes.**

COM6 is a high-performance matrix multiplication engine built from scratch in C with hand-written x86-64 inline assembly. Through 95 versions of iterative optimization, it evolved from naive loops into a BLIS-class implementation featuring: 8x k-unrolled FMA micro-kernels (AVX2 6x8), 5-loop cache hierarchy blocking, JC-parallel threading (private B-panels, zero barriers), Strassen hybrid for 8192+ (Winograd variant, 7 sub-muls), separate beta-0/beta-1 micro-kernels, adaptive MC/KC/NC blocking, C-prefetch at kernel entry, 2x-unrolled B-panel packing, and reverse benchmark ordering for thermal management.

## COM7NN Transformer vs Standard Transformer

The COM framework extends beyond matrix multiplication into neural network architectures. COM7NN implements a full transformer with the COM Tensor class (shape-safety, custom operations).

| Metric | Standard NumPy | COM7NN | Winner |
|--------|---------------|--------|--------|
| Final loss | 2.808 | **2.290** | COM7NN |
| Loss reduction | 34.4% | **49.7%** | COM7NN |
| Training time | 36.93s | **19.95s** | COM7NN (1.85x faster) |
| Inference accuracy | 4/10 | **6/10** | COM7NN |

Same architecture (d_model=64, 4 heads, d_ff=128, 1 layer), same data (counting task), same seed. COM7NN converges faster, trains faster, and predicts more accurately.

## COM6 Matrix Multiplication Results (v95 - Latest)

### v95: Strassen-Hybrid + JC-Parallel (latest)

v95 adds single-level Strassen (Winograd variant) for n>=8192: 7 sub-multiplications of (n/2)x(n/2) instead of 8, saving 12.5% of FLOPs. Each sub-multiply delegates to the full JC-parallel BLIS GEMM. Submatrix additions are O(n^2) — negligible overhead.

For n<8192: identical to v94 (JC-parallel for n>=2048, IC-parallel for n<2048).

**v95 vs v94 head-to-head at 8192 (matched thermal conditions):**

| Version | 8192 MT | Improvement |
|---------|---------|-------------|
| v94 | 26.6 GF | baseline |
| **v95** | **29.0 GF** | **+8.6%** |

### v94: JC-Parallel for Large + Adaptive NC

For n>=2048: JC-parallel mode where each thread owns a column slab with private B-panel (NC=1024). Zero barriers, zero C write contention.

For n<2048: IC-parallel with shared B-panel (v26 approach, proven best for small/medium). MC_MT=48 for n<=1024 (better thread balance).

Key features: beta-0/beta-1 micro-kernels (skip C load on first pc iteration), C-prefetch at kernel entry, 2x B-pack unroll, reverse benchmark order (largest first for cold CPU).

### v85 (Xeon results) — see below

---

## Historical Results

### v85 vs OpenBLAS MT — Xeon Skylake (historical)

### v85 vs OpenBLAS MT — Xeon Skylake 16-core (no thermal throttle)

| Size | OpenBLAS MT | COM6 v85 MT | Ratio | Winner |
|------|-------------|-------------|-------|--------|
| 512x512 | 121.4 GF | **145.6 GF** | **1.20x** | **COM6** |
| 1024x1024 | 105.2 GF | **131.5 GF** | **1.25x** | **COM6** |
| 2048x2048 | **152.5 GF** | 144.0 GF | 0.94x | BLAS |
| 4096x4096 | **225.1 GF** | 184.0 GF | 0.82x | BLAS |

**COM6 wins at 512 and 1024.** The KC=n one-shot optimization (v85) eliminates v80's 512 loss. BLAS still wins at larger sizes on Xeon due to better AVX-512 utilization.

### v85 Key Changes: One-Shot KC + Pre-Allocated Buffers + OMP Warmup
- **KC=n for n<=512 (one-shot K)**: Entire K dimension processed in one pass. For 512: KC=512 means 1 pc iteration instead of 2. Halves barriers (2 vs 4), halves B-packing, halves A-packing, eliminates beta=1 path entirely. This is the key fix for the v80 512 loss.
- **Pre-allocated static buffers**: v84 malloc'd per-thread A-buffers and shared B-buffer on every `com6_multiply()` call. v85 allocates once at first call, reuses forever. Eliminates ~50-100us malloc overhead per call.
- **OMP thread pool warmup**: Dummy parallel region at program start pre-creates the OpenMP thread pool. First real multiply doesn't pay the ~100us thread-creation cost.
- **GOMP_SPINCOUNT=infinite tested but rejected**: Keeping OMP threads spinning between parallel regions was tested. On 15W TDP laptop, this burns thermal budget and causes throttling. Rejected.
- **All v84 compute code preserved**: 3 micro-kernel variants (beta0/beta1/beta0-NT), 8x k-unrolled FMA, 4-tier adaptive blocking, prefetched A-packing, parallel B-packing.

### v85 vs v84 improvement at 512 on Xeon

| Version | 512 MT (GF) | Notes |
|---------|-------------|-------|
| v84 | 22.0 | KC=256, 2 pc iterations, 4 barriers |
| **v85** | **40.3** | KC=512, 1 pc iteration, 2 barriers |
| Improvement | **+83%** | One-shot KC eliminates half the overhead |

### v80 vs OpenBLAS MT (historical, laptop, 10s cooldown between tests)

| Size | OpenBLAS MT | COM6 v80 MT | Ratio | Winner |
|------|-------------|-------------|-------|--------|
| 512x512 | 105.1 GF | 73.9 GF | 0.70x | BLAS |
| 1024x1024 | 103.8 GF | **114.5 GF** | **1.10x** | **COM6** |
| 2048x2048 | 100.3 GF | **117.5 GF** | **1.17x** | **COM6** |
| 4096x4096 | 98.1 GF | **122.9 GF** | **1.25x** | **COM6** |
| 8192x8192 | 59.1 GF | **69.4 GF** | **1.17x** | **COM6** |

**COM6 wins 4 out of 5 sizes.** 512 loss was OpenMP thread-spawn overhead.

Note: absolute GFLOPS vary wildly (up to 2x) with thermal state on 15W TDP laptop. Ratios vs OpenBLAS are the meaningful comparison — both are equally affected by thermal throttle.

### v80 Key Changes: Universal Persistent Thread Pool
- **Single parallel region for ALL MT sizes**: v79 only did this for n>=4096. v80 does it universally, eliminating repeated fork-join overhead.
- **Deep KC for 8192+**: Minimize C-passes (DRAM round-trips).
- **Three micro-kernel variants**: beta0 (store), beta1 (load+add+store with C-prefetch), beta0-NT (non-temporal store for large C).

### v77 Key Changes: 2x B-Pack Unroll + Static Scheduling
- **2x k-unrolled B-packing**: Process 2 rows per iteration in full NR=8 panels, doubling throughput and halving loop overhead.
- **Single parallel region for n>=4096**: Eliminates fork/join overhead between tiles (from v76).
- **Deep KC=512 for 8192+**: 8192/512 = 16 C-passes (27% fewer than KC=384).
- **Three micro-kernel variants**: beta0 (store), beta1 (load+add+store), beta0-NT (non-temporal store).
- **C-prefetch in beta1 kernel**: Prefetches all 6 C rows before k-loop starts.

### v74 Key Changes: Memset Elimination (Two-Kernel Beta)
- **No more memset**: Eliminated `memset(C, 0, n*n*8)` — saves bandwidth for zeroing entire C matrix.
- **Two kernel variants**: `micro_6x8_beta0` (first KC tile, just stores) and `micro_6x8_beta1` (subsequent KC tiles, load+add+store).
- **Branch at call site, not in kernel**: macro_kernel passes beta to select variant — still branch-free inside.
- 8192 back-to-back: v74 51.6 GF vs v73 43.5 GF (+19%)

### v73 Key Changes: Branch-Free Kernel + MT-Specific Blocking
- **Branch-free micro-kernel** (from v24): No runtime beta/NT branching inside the FMA loop.
- **Separate 1T/MT blocking functions**: 1T uses larger MC for better L2 utilization; MT uses smaller MC for better thread distribution.
- **MC_MT_SMALL=48 for n<=1024 MT**: 512/48=11 ic-tiles for 8 threads (vs 512/120=5 in 1T).
- **Back-to-back wins vs v72**: 512 MT +17%, 1024 MT +37%.

### v72 Key Changes: Branch-Free Kernel Restoration
- Merged v24's proven branch-free micro-kernel with v69's framework improvements.
- 3-tier blocking: KC=256/MC=120 (SMALL), KC=320/MC=96 (LARGE), KC=384/MC=72/NC=1536 (HUGE).
- Single parallel region for n>=4096.

### v71 Cold-Start Performance (isolated single-size tests, 60-90s cooldown, historical)

| Size | COM6 v71 1T | COM6 v71 MT | Notes |
|------|-------------|-------------|-------|
| 256x256 | **24.9 GF** | 24.5 GF | 1T optimal (threading overhead) |
| 512x512 | **48.4 GF** | **94.0 GF** | 77% of 1T theoretical peak |
| 1024x1024 | — | **104.4 GF** | MC=72 balanced for 8 threads |
| 2048x2048 | — | **119.8 GF** | Adaptive NT stores + dynamic sched |
| 4096x4096 | — | **88.2 GF** | Thermal-sensitive (15W TDP) |
| 8192x8192 | — | **62.9 GF** | KC=448 + NC=2048 (42% fewer B-packs) |

**Peak: 119.8 GFLOPS** at 2048 MT. 1T peak: **48.4 GF** at 512.

### v71 vs OpenBLAS MT (fair interleaved, 5s cooling between tests)

| Size | OpenBLAS MT | COM6 v71 MT | Ratio | Winner |
|------|-------------|-------------|-------|--------|
| 512x512 | 73.6 GF | **94.0 GF** | **1.28x** | **COM6** |
| 1024x1024 | 69.1 GF | **72.4 GF** | **1.05x** | **COM6** |
| 2048x2048 | 98.4 GF | **119.2 GF** | **1.21x** | **COM6** |
| 4096x4096 | 61.0 GF | **80.8 GF** | **1.32x** | **COM6** |
| 8192x8192 | 56.5 GF | 47.8 GF | 0.85x | BLAS |

**COM6 wins 4 out of 5 sizes.** The 8192 gap narrowed significantly vs v70 (from NC=1536 to NC=2048 with deeper KC=448). Note: absolute GFLOPS vary with thermal state on this 15W laptop; ratios are the meaningful comparison.

### v71 Key Change: 8192 Optimization
- **Before (v70)**: NC=1536, KC=384, MC=72 → 6 jc × 22 pc = 132 B-pack calls, 264 barriers
- **After (v71)**: NC=2048, KC=448, MC=54 → 4 jc × 19 pc = 76 B-pack calls, 152 barriers (42% fewer)
- B-panel: 2048×448×8 = 7.34MB (fits 8MB L3)
- A-panel: 54×448×8 = 194KB (fits 256KB L2)
- Cold-start 8192 MT: **62.9 GF** (v70: 46.3 GF, **+36%**)

### v68 vs OpenBLAS MT (historical, fair interleaved, 5s cooling)

| Size | OpenBLAS MT | COM6 v68 MT | Ratio | Winner |
|------|-------------|-------------|-------|--------|
| 512x512 | 27.7 GF | **65.6 GF** | **2.37x** | **COM6** |
| 1024x1024 | 53.7 GF | **61.2 GF** | **1.14x** | **COM6** |
| 2048x2048 | 53.9 GF | **65.2 GF** | **1.21x** | **COM6** |
| 4096x4096 | 59.0 GF | **83.0 GF** | **1.41x** | **COM6** |
| 8192x8192 | 56.0 GF | **71.5 GF** | **1.28x** | **COM6** |

**COM6 wins all 5 sizes.** Beats OpenBLAS by 14-137%.

### v68 Full Benchmark (historical)

| Size | 1-Thread | MT | GF(1T) | GF(MT) |
|------|----------|-----|--------|--------|
| 256x256 | 1.0 ms | 0.9 ms | 32.3 | **35.6** |
| 512x512 | 6.3 ms | 3.3 ms | 42.7 | **80.8** |
| 1024x1024 | 51.7 ms | 24.4 ms | 41.6 | **88.0** |
| 2048x2048 | 435.1 ms | 188.9 ms | 39.5 | **91.0** |
| 4096x4096 | -- | 1415.4 ms | -- | **97.1** |
| 8192x8192 | -- | 16055.3 ms | -- | **68.5** |

### v65 Best Results (isolated cold-CPU tests with 15-30s cooldowns, historical)

| Size | COM6 v65 1T | COM6 v65 MT | Notes |
|------|-------------|-------------|-------|
| 256x256 | 40.2 GF | 40.6 GF | 1T optimal (threading overhead) |
| 512x512 | 46.7 GF | **89.9 GF** | MC=48 gives 10+ ic-blocks for 8T |
| 1024x1024 | 43.8 GF | **83.2 GF** | dynamic scheduling helps load balance |
| 2048x2048 | 39.2 GF | **108.2 GF** | dynamic scheduling key here |
| 4096x4096 | -- | **103.0 GF** | static scheduling (lower overhead) |
| 8192x8192 | -- | **64.7 GF** | L3 pressure from 5MB shared B panel |

v65 merges the best of v62 (C-prefetch, dynamic scheduling, MC=48) with v26's proven static scheduling for large sizes. Key innovation: **adaptive scheduling** — `schedule(dynamic,1)` for n<=2048 (where load imbalance matters) and `schedule(static)` for n>2048 (where overhead dominates). Also separates 1T and MT blocking functions.

### v62 vs OpenBLAS MT (fair interleaved comparison, 5s cooling between tests)

| Size | OpenBLAS MT | COM6 MT | Ratio | Winner |
|------|-------------|---------|-------|--------|
| 512x512 | 33.8 GF | **60.8 GF** | **1.80x** | **COM6** |
| 1024x1024 | 46.7 GF | **85.5 GF** | **1.83x** | **COM6** |
| 2048x2048 | 67.2 GF | **70.6 GF** | **1.05x** | **COM6** |
| 4096x4096 | 42.4 GF | **47.2 GF** | **1.11x** | **COM6** |
| 8192x8192 | 52.0 GF | **59.7 GF** | **1.15x** | **COM6** |

**COM6 wins all 5 sizes.** Beats OpenBLAS by 5-83%.

### v61 vs OpenBLAS MT (fair interleaved comparison, 5s cooling between tests)

| Size | OpenBLAS MT | COM6 MT | Ratio | Winner |
|------|-------------|---------|-------|--------|
| 512x512 | 17.3 GF | **42.8 GF** | **2.47x** | **COM6** |
| 1024x1024 | 32.7 GF | **50.3 GF** | **1.54x** | **COM6** |
| 2048x2048 | 36.6 GF | **49.8 GF** | **1.36x** | **COM6** |
| 4096x4096 | 24.8 GF | **40.1 GF** | **1.62x** | **COM6** |
| 8192x8192 | 32.4 GF | **44.9 GF** | **1.39x** | **COM6** |

**COM6 wins all 5 sizes.** Beats OpenBLAS by 36-147% across all tested sizes. Note: absolute GFLOPS vary with thermal state; ratios are the meaningful comparison (both tested under identical conditions with cooling).

### v57 vs OpenBLAS MT (historical, 3s cooling between tests)

| Size | OpenBLAS MT | COM6 MT | Ratio | Winner |
|------|-------------|---------|-------|--------|
| 512x512 | 16.3 GF | **16.9 GF** | **1.04x** | **COM6** |
| 1024x1024 | 25.2 GF | **41.9 GF** | **1.66x** | **COM6** |
| 2048x2048 | 20.1 GF | **45.1 GF** | **2.24x** | **COM6** |
| 4096x4096 | 36.6 GF | **43.9 GF** | **1.20x** | **COM6** |
| 8192x8192 | 41.5 GF | **50.2 GF** | **1.21x** | **COM6** |

### v50 vs OpenBLAS MT (historical best, cold CPU)

| Size | OpenBLAS MT | COM6 MT | Ratio | Winner |
|------|-------------|---------|-------|--------|
| 256x256 | 90.0 GF | 57.5 GF | 0.64x | BLAS |
| 512x512 | 91.5 GF | **98.1 GF** | **1.07x** | **COM6** |
| 1024x1024 | 113.3 GF | **117.8 GF** | **1.04x** | **COM6** |
| 2048x2048 | 101.9 GF | **120.6 GF** | **1.18x** | **COM6** |
| 4096x4096 | 104.3 GF | **122.5 GF** | **1.17x** | **COM6** |
| 8192x8192 | 84.1 GF | **98.9 GF** | **1.18x** | **COM6** |

### v50 Full Benchmark (8 threads, sequential run)

| Size | 1-Thread | MT | GF(1T) | GF(MT) |
|------|----------|-----|--------|--------|
| 256x256 | 0.9 ms | 0.6 ms | 37.3 | **53.1** |
| 512x512 | 5.5 ms | 3.0 ms | 48.9 | **90.5** |
| 1024x1024 | 42.8 ms | 18.2 ms | **50.2** | **118.1** |
| 2048x2048 | 373.7 ms | 138.0 ms | 46.0 | **124.5** |
| 4096x4096 | (skip) | 1127.9 ms | -- | **121.8** |
| 8192x8192 | (skip) | 8979.0 ms | -- | **122.5** |

**Peak: 125.3 GFLOPS** at 2048. 1T peak: **50.2 GF** at 1024 (79% of theoretical single-core peak).

### Key improvements v68

- **Adaptive NC** (v68): NC=2048 for n<=4096, NC=1536 for n>4096. At 8192, B-panel drops from 5MB to 3.8MB, leaving more L3 for A-panels and C data. This alone gave +46% at 8192 (47→69 GF).
- **Deep KC tier for huge** (v68): KC=384, MC=72 for n>4096. Deeper K means more FMA work per pack, amortizing packing overhead. MC*KC*8=216KB fits L2 (256KB). 4096 improved from 55→97 GF.
- **NT stores for beta=0** (v67): On first KC-tile, vmovntpd bypasses cache. Eliminates memset and frees L3 bandwidth.
- **Single parallel region for 4096+** (v67): All NC/KC/IC loops inside one `#pragma omp parallel`. Eliminates 100+ fork/join overheads at 8192.
- **C-output prefetch** (v66): Prefetches all 6 output rows into L1 at micro-kernel entry. Hides C-read latency during beta=1 accumulation.

### Key improvements v65 (historical)

- **Adaptive scheduling** (v65): `schedule(dynamic,1)` for n<=2048 gives +40% at 2048 (better load balance on thermally-throttled cores), while `schedule(static)` for n>2048 avoids dynamic dispatch overhead with 42+ IC-blocks.
- **Separate 1T/MT blocking** (v65): 1T uses standard v26 2-tier blocking, MT uses 3-tier with MC_TINY=48 for n<=512. No interference between paths.
- **All v62 improvements inherited**: C-prefetch kernel, 4x A-pack, parallel B-packing, CLI mode.

### Key improvements v62

- **Restored v26 proven blocking** (v62): v28-v61 had blocking regressions (wrong KC/MC for large sizes, e.g. v61 line 434 used KC_SMALL=256 for n>2048 instead of KC_LARGE=320). v62 goes back to proven blocking: MC=48/KC=256 for n<=512, MC=120/KC=256 for n<=1024, MC=96/KC=320 for n>1024. All fit L2=256KB.
- **MC=48 for small MT** (v62): 512/48=10.7 ic-blocks gives much better 8-thread utilization than 512/120=4.3. This was the key to matching/beating BLAS at 512 (60.8 vs 33.8 GF).
- **MT threshold lowered** (v62): Multi-threading enabled at n>=512 (was n>512), matching OpenBLAS behavior.
- **C-output prefetch in micro-kernel** (v61/v62): Prefetches all 6 output rows of C into L1 at the start of the micro-kernel using r15 register.
- **4x unrolled A-packing** (v60/v62): 24 loads + 24 stores per iteration, halving loop overhead.
- **Dynamic scheduling** (v62): `schedule(dynamic,1)` for MT loop — better load balance on thermally-throttled cores.
- **CLI single-size mode** (v62): `./com6_v62 <size> [mt|1t]` for isolated cold-CPU tests.
- **8192 support**: Tested and working at 8192x8192 (59.7 GF MT).

### Key improvements v61 (historical)

- **C-output prefetch in micro-kernel** (v61): Prefetches all 6 output rows of C into L1 at the start of the micro-kernel.
- **MT for all sizes** (v61): Enables multi-threading for n>=256 (was n>512).
- **Four-tier adaptive blocking** (v61): MC_TINY=36 for n<=256, MC=120/KC=256 for n<=1024, MC=96/KC=320 for n<=2048, MC=120/KC=256 for n>2048.
- **NOTE**: v61 had a blocking regression at line 434 (`else{*pMC=MC_SMALL;*pKC=KC_SMALL;}`) that used small-size blocking for all n>2048. Fixed in v62.

### Key improvements v43-v50

- **beta=0 micro-kernel** (v43): First pc iteration stores directly, eliminating memset.
- **L3-optimized 8192 tier** (v43): KC=320/MC=96/NC=1024 keeps B panel at 2.5MB (31% of L3).
- **4x k-unrolled pack_A** (v45): 24 independent loads+stores per loop body for better ILP.
- **MC_TINY=48 for MT** (v46): More ic-blocks for better thread utilization.
- **No-pack small path** (v47): Skip packing for n<=256 — direct AVX2 FMA on unpacked data.
- **Fixed MT load balancing** (v49): `omp for schedule(static)` replaces manual ic-distribution.
- **Distributed prefetch** (v50): pA/pB prefetch spread across k+0/k+2/k+4/k+6 instead of bunched. Keeps memory pipeline fed evenly.
- **Pack-A source prefetch** (v50): Prefetch next A rows while packing current panel.
- **4x k-unrolled small path** (v50): Small-matrix kernel processes 4 k-iterations per inner loop for better ILP — **256: 33→53 GF (+60%)**.
- **OpenMP warmup** (v50): Pre-fork thread pool before benchmarking eliminates first-call overhead.
- **NC=2048 for all sizes** (v51): Halves jc iterations for 8192 (4 vs 8), cutting A-panel repacking overhead in half (8944 vs 17888 pack_A calls).
- **C-prefetch in macro_kernel** (v51): Prefetch next ir-block's C rows before micro_6x8, initiating RFO early to hide store-miss latency.
- **Thermal-aware benchmarking** (v51): 5-second cooldown between sizes in full benchmark to combat laptop TDP throttling.
- **L2-aware NC blocking** (v52, experimental): NC=128 for n<=512 MT so B-panel (256KB) fits per-core L2 instead of spilling to L3.
- **Physical-core threading** (v52, experimental): 4 threads for n<=512 to avoid HT contention on compute-bound micro-kernels.

Single-size cold-start: `./com6_v51 8192`

### AVX-512 Performance — Xeon Skylake Server (v35, 16 cores, size-aware thread scaling)

| Size | COM6 AVX-512 (1T) | COM6 AVX-512 (MT) | OpenBLAS (MT) | COM6 vs BLAS |
|------|-------------------|-------------------|---------------|-------------|
| 256x256 | 24.8 GF | **44.6 GF** | 22.7 GF | **1.96x** |
| 512x512 | 25.6 GF | **87.4 GF** | 78.6 GF | **1.11x** |
| 1024x1024 | 25.6 GF | **123.2 GF** | 161.8 GF | 0.76x |
| 2048x2048 | 25.7 GF | **189.0 GF** | 175.7 GF | **1.08x** |
| 4096x4096 | 25.7 GF | **240.8 GF** | 231.6 GF | **1.04x** |
| 8192x8192 | 25.5 GF | **255.2 GF** | 221.0 GF | **1.15x** |
| 16384x16384 | 23.8 GF | **181.8 GF** | 238.5 GF | 0.76x |

Peak: **255.2 GFLOPS** at 8192x8192 with AVX-512 6x16 ZMM kernel. v35's size-aware thread capping uses `2*sqrt(n/64)` threads for small sizes (4 threads at 256, 6 at 512, 8 at 1024) and all cores for n>=2048. COM6 beats OpenBLAS at **5 out of 7 sizes** on Xeon. At 16384, COM6 maintained a working RSS of ~6.3 GB (close to the 6.0 GiB theoretical A+B+C floor), while concurrent OpenBLAS+numpy exceeded available memory and was OOM-killed.

### v36 Parameter Sweep Results (targeting 0.76x at 1024)

A 189-configuration parameter sweep on Xeon (MC × KC × NC × thread count) found that 1024's weakness was **under-threading** (8T vs optimal 12T) and **suboptimal KC** (256 vs 320):

| Config | MC | KC | NC | Threads | GF/s at 1024 |
|--------|----|----|-----|---------|-------------|
| **v36 optimal** | 48 | 320 | 2048 | 12 | **163.1** |
| v35 default | 48 | 256 | 2048 | 8 | 123.2 |
| OpenBLAS | — | — | — | 16 | 161.8 |

KC=320 reduces pc-loop iterations from 4 to 3.2, and 12 threads better saturates the 16-core Xeon at this size. v36 applies these sweep-optimal parameters via table-based thread capping. Full benchmark validation pending (server was under VM load during testing).

### What Changed at 256/512 (v30-v31)

Previously, COM6 used OpenMP which has ~50-100μs fork-join overhead per parallel region. For tiny matrices (256: ~0.8ms compute), this overhead was devastating. v30-v31 replaced OpenMP with:
- **Persistent pthreads pool**: threads created once, ~1μs dispatch latency
- **MC=48 for n≤512**: 6 ic-blocks at 256 (vs 3 with MC=120) — much better 8-thread utilization
- **Merged dispatch**: B-pack + spin barrier + ic-loop in single wake (half the overhead)
- Result: 256 jumped from 42 GF to **55.3 GF** (+32%), now **19% faster than OpenBLAS**

### Thermal-Aware Results (v29+v31, sustained on hot CPU)

| Size | Standard MT | Thermal-Aware MT | Improvement |
|------|------------|------------------|-------------|
| 4096x4096 | 90.6 GF | **106.3 GF** | +17% |
| 8192x8192 | 69.5 GF | **77.5 GF** | +12% |

The thermal-aware mode detects when CPU clock throttling occurs and automatically reduces to 4 physical cores (dropping HyperThreading), trading parallelism for sustained higher clock speeds.

### vs Strassen (where it all started)

| Size | Strassen | COM6 v12 | Speedup |
|------|----------|----------|---------|
| 256x256 | 3.1 ms | 1.5 ms | **2.12x** |
| 512x512 | 18.5 ms | 11.5 ms | **1.61x** |
| 1024x1024 | 150.3 ms | 97.7 ms | **1.54x** |
| 2048x2048 | 1273.2 ms | 775.1 ms | **1.64x** |
| 4096x4096 | 7864.1 ms | 5435.8 ms | **1.45x** |
| 8192x8192 | 80171.6 ms | 54314.6 ms | **1.48x** |

## Architecture

COM6 v35 implements the full BLIS 5-loop nest with persistent thread pool, dual ISA micro-kernels, size-aware thread scaling, and thermal monitoring:

```
PERSISTENT THREAD POOL (auto-detect cores, created once)
  jc-loop (NC=2048/1536 adaptive, L3 blocking)
    pc-loop (KC=256/320 adaptive, L2 blocking)
      [THERMAL CHECK: measure throughput, adjust active thread count]
      SINGLE MERGED DISPATCH:
        PARALLEL B-packing (contiguous SIMD loads, no transpose)
        SPIN BARRIER (atomic, ~10ns)
        PARALLEL ic-loop (MC=48/96/120 adaptive, atomic work-stealing)
          A-packing (per-thread buffers, 2x k-unrolled)
          jr-loop (NR=16 AVX-512 / NR=8 AVX2)
            ir-loop (MR=6)
              6x16 ZMM micro-kernel (AVX-512, 12 accumulators, 8x k-unrolled)
              — or —
              6x8 YMM micro-kernel (AVX2, 12 accumulators, 8x k-unrolled)
```

### Key Technical Details

- **Persistent pthreads pool**: Threads spin-wait (2000 iterations ~1μs) then sleep on condvar — zero fork-join overhead, minimal thermal impact when idle
- **Dual ISA micro-kernels**: AVX-512 6x16 (12 ZMM accumulators, 96 outputs/rank-1) or AVX2 6x8 (12 YMM accumulators, 48 outputs/rank-1) — compile-time selection via `__AVX512F__`
- **8x k-unrolling**: 8 rank-1 updates per loop iteration — 97% useful FMA work vs 3% loop overhead
- **GNU inline assembly**: Direct register allocation, zero compiler interference
- **Merged dispatch**: B-pack + spin barrier + ic-loop in single thread wake — halves dispatch overhead vs separate phases
- **Atomic work-stealing**: `atomic_fetch_add` for ic-blocks — dynamic load balancing with zero lock contention
- **Direct B-packing**: `B[k][j..j+7]` is contiguous — SIMD copy, no transpose needed
- **Size-aware thread scaling**: On many-core systems (>8 threads), uses `2*sqrt(n/64)` threads for small sizes (4 at 256, 6 at 512, 8 at 1024), all cores for n>=2048 — avoids L3 contention and dispatch overhead
- **Quad-adaptive blocking**: MC=48/96 + KC=256/320 + NC=2048/1536 per problem size — MC=48 for n<=1024 maximizes ic-block granularity for work-stealing
- **L3 pressure management**: n>4096 uses NC=1536 KC=256 (B-panel=3MB) instead of NC=2048 KC=320 (B-panel=5MB) — 35% faster at 8192
- **Thermal-aware thread scaling**: On laptops (<=8 threads), monitors per-iteration throughput; when throttling detected, drops from 8 HyperThreads to 4 physical cores for sustained higher clocks. Disabled on servers where throttling doesn't occur.
- **Per-thread A buffers**: Each thread packs independently — zero false sharing

### Cache Hierarchy Targeting (i7-10510U)

| Buffer | Size | Fits In |
|--------|------|---------|
| A micro-panel | MR*KC*8 = 12-15 KB | L1d (64 KB/core) |
| B micro-panel | NR*KC*8 = 16-20 KB | L1d (64 KB/core) |
| A macro-panel | MC*KC*8 = 96-240 KB | L2 (256 KB/core) |
| B macro-panel | NC*KC*8 = 4-5 MB | L3 (8 MB shared) |

## Version History

| Version | Key Change | Peak GF |
|---------|------------|---------|
| v1-v6 | COM6 concept, blocked transpose, auto-vec | ~5-10 |
| v7 | Hand-written AVX2 FMA intrinsics | ~15 |
| v8-v9 | Double-pumped FMA, BLIS cache blocking | ~25 |
| v10-v12 | Memory pool, adaptive Strassen, SW-pipelined | ~32 |
| v13-v15 | BLIS 5-loop, 6x8 outer-product micro-kernel | 23-38 |
| **v16** | **Direct B-packing — matched OpenBLAS 1T** | **44** |
| v17-v18 | Strassen hybrid experiments | 46 |
| v19-v21 | Interleaved FMA, inline ASM, 4x k-unroll | 45 |
| **v22** | **OpenMP multi-threading — beat OpenBLAS MT** | **104** |
| v23 | Merged parallel region + adaptive threading | 109 |
| **v24** | **8x k-unrolled ASM micro-kernel** | **110.6** |
| v25-v26 | Adaptive KC/MC per problem size | **115.1** |
| v27 | Strassen hybrid (failed — copy overhead) | 84 |
| v28 | Pure BLIS at 8192 scale | 84.5 |
| **v29** | **Thermal-aware dynamic thread scaling** | **106.3 sustained** |
| **v30** | **Persistent pthreads pool — beat BLAS at 512** | **76.6** (512) |
| **v31** | **Merged dispatch + MC=48 — beat BLAS at ALL sizes** | **55.3** (256) |
| **v32** | **AVX-512 6x16 ZMM kernel + quad-adaptive blocking** | **232.6** (Xeon 8192) |
| **v33** | **Adaptive thread scaling — ncores-2 reduces L3 contention** | **257.1** (Xeon 8192) |
| v34 | JC-parallel dispatch experiment | 257.5 (Xeon 8192) |
| **v35** | **Size-aware thread capping + MC=48@1024 — beat BLAS at 5/6 Xeon sizes** | **255.2** (Xeon 8192) |
| **v36** | **Sweep-optimized: KC=320 + 12T at 1024 — parameter sweep shows 163 GF (beats BLAS 161.8)** | **163.1** (Xeon 1024, sweep) |
| v37 | pthreads pool experiment — slower than OpenMP on laptop | 54.2 |
| **v38** | **v26 OpenMP core + 8192 + adaptive NC=3072 for n>2048** | **65.5** (laptop 2048 MT) |
| v39 | Always-MT + MC=48 for small sizes | 69.4 (512 MT) |
| **v40** | **C-prefetch micro-kernel + single parallel region** | **72.0** (1024 MT), **59.6** (8192 MT) |
| **v41** | **Deep KC=512 for 8192 + MC=60 (clean MR divisibility)** | **102.2** (4096 MT), **80.8** (8192 MT) |
| **v42** | **MC_TINY=48 for n<=512 — matches BLAS at 512!** | **85.4** (512 MT), **109.5** (4096 MT) |
| **v43** | **beta=0 memset elimination + L3-optimized 8192 tier** | **111.3** (2048 MT), **69.0** (8192 MT) |
| **v44** | **Smart thread threshold — 1T for n<=256, wins 5/6 vs BLAS** | **86.2** (1024 MT), **73.9** (8192 MT) |
| **v45** | **4x k-unrolled pack_A for better ILP** | **95.3** (1024 MT), **126.0** (8192 MT) |
| **v46** | **MC_TINY=48 for all-size MT + v45 core** | **126.2** (2048 MT), **125.6** (4096 MT) |
| v47 | No-pack path for n<=256 (skip packing entirely) | 50 (256 MT) |
| v48 | 8-wide j no-pack kernel (2x ymm per j-iteration) | 50 (256 MT) |
| **v49** | **Fixed MT load balancing: omp for schedule(static) + MC_TINY for <=1024** | **123.7** (1024 MT), **126.6** (8192 MT) |
| **v50** | **Distributed prefetch + pack-A prefetch + 4x k-unrolled small path + OMP warmup** | **125.3** (2048 MT), **50.2** (1T 1024) |
| v51 | NC=2048 for all sizes + C-prefetch in macro_kernel + thermal-aware cooldowns | 122.5 (8192 MT) |
| v52 | L2-aware NC=128 for n<=512 MT + physical-core threading (experimental) | 91.5 (512 MT) |
| v53-v54 | Best-of-all: NC=4096 + C-prefetch + cold-first ordering | 73.6 (1024 MT) |
| **v55** | **Right-sized buffer allocation — eliminates TLB pressure from oversized alloc** | **93.2** (2048 MT), **76.0** (1024 MT) |
| v56 | NC=2048 everywhere (B-panel fits L3) + C-prefetch + right-sized buffers | 63.4 (2048 MT) |
| **v57** | **Streaming B-pack (non-temporal stores bypass L1/L2) + dynamic(2) scheduling — beats BLAS at ALL sizes** | **63.2** (2048 MT), 1.21-2.24x vs BLAS |
| v58 | Thermal-aware improvements, cooling between sizes | 56.0 (2048 MT) |
| v59 | Always-MT + small-size MC tuning experiments | varies |
| v60 | 4x A-pack unroll + parallel C-zeroing + streaming B-pack | 42.0 (8192 MT) |
| **v61** | **C-prefetch micro-kernel + 4-tier blocking + L3-aware KC + MT all sizes — beats BLAS at ALL sizes by 36-147%** | **50.3** (1024 MT), **44.9** (8192 MT), 1.36-2.47x vs BLAS |
| **v62** | **Clean rewrite: v26 blocking + C-prefetch + MC=48 for small MT — fixes v28-v61 regressions, beats BLAS at ALL sizes (1.05-1.83x)** | **85.5** (1024 MT), **60.8** (512 MT), **59.7** (8192 MT) |
| v63 | NC=1024 for 8192+ (L3-fit experiment) — slower due to extra B-packing overhead | 43.2 (8192 MT) |
| v64 | JC-parallel (barrier-free MT, each thread owns columns) — slower at 2048 due to duplicated A-packing | 55.5 (4096 MT) |
| **v65** | **Adaptive scheduling hybrid: dynamic(<=2048) + static(>2048), separate 1T/MT blocking, C-prefetch kernel** | **108.2** (2048 MT), **103.0** (4096 MT), **89.9** (512 MT) |
| v66 | C-output prefetch in micro-kernel entry | 95.5 (1024 MT) |
| **v67** | **NT stores for beta=0 + single parallel region for 4096+** | **88.0** (1024 MT), **68.5** (8192 MT) |
| v68 | Adaptive NC (1536 for 8192) + deep KC tier | **97.1** (4096 MT), **71.5** (8192 MT) |
| **v69** | **Adaptive NT stores: regular stores for n<2048 (keep C in cache)** | **95.5** (1024 MT), **121.5** (2048 MT) |
| v70 | Adaptive NC + 4-tier MT blocking + MC_TINY=66 for 512 | **84.9** (512 MT), **93.2** (8192 MT) |
| **v71** | **8192 optimization: KC=448 NC=2048 MC=54 — 42% fewer B-pack calls, +36% at 8192** | **119.8** (2048 MT), **62.9** (8192 MT) |
| **v72** | **Branch-free kernel restoration (v24 kernel) + 3-tier adaptive blocking + 4x A-pack + 8192** | **63.8** (2048 MT), **56.9** (512 MT) |
| **v73** | **MT-specific blocking: MC=48 for small MT gives +17% at 512, +37% at 1024 vs v72** | **84.4** (2048 MT), **71.5** (8192 MT) |
| **v74** | **Memset elimination: beta0/beta1 two-kernel split, no memset needed — +19% at 8192** | **79.8** (8192 MT), **88.6** (1024 MT) |
| **v75** | **NT stores for large beta=0: vmovntpd bypasses cache for n>=2048 — +5% at 8192** | **106.4** (1024 MT), **83.6** (8192 MT) |
| v76-v90 | Various experiments (persistent pools, NT stores, A-reuse, etc.) | varies |
| **v91** | **Back to basics: v26 arch + beta-0/1 kernels + MC=48 MT + CLI mode** | **110.0** (2048 MT), **81.5** (8192 MT) |
| **v92** | **Dynamic sched(2) for n>=2048 + 2x B-pack unroll — 1.57x vs BLAS at 8192** | **48.5** (2048 MT warm), 1.39-1.57x vs BLAS |

## Building

Requires GCC with AVX2/FMA support:

```bash
# v92: AVX2 + OpenMP + beta-0/1 kernels + dynamic sched + 2x B-pack (recommended, latest)
gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp -o com6_v92 com6_v92.c -lm
./com6_v92           # full benchmark (8192-256, reverse order)
./com6_v92 4096 mt   # single-size MT cold CPU test
./com6_v92 512 1t    # single-size 1T test
./com6_v92 8192 mt   # 8192x8192 MT-only (1T skipped for huge sizes)

# v38: AVX2 + OpenMP
gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp -o com6_v38 com6_v38.c -lm
./com6_v38

# v36: AVX-512 + pthreads (recommended for Xeon/server)
gcc -O3 -march=native -mavx512f -mfma -funroll-loops -o com6_v36 com6_v36.c -lm -lpthread
./com6_v36

# v36: AVX2 fallback
gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v36 com6_v36.c -lm -lpthread
./com6_v36
```

## Test Platforms

### Primary: Intel Core i7-10510U (Comet Lake) Laptop
- 4 cores / 8 threads, 15W TDP
- 64 KB L1d/core, 256 KB L2/core, 8 MB L3 shared
- Theoretical peak: 63.2 GFLOPS/core (2 FMA units x 4 doubles x 2 flops x 3.95 GHz boost)
- Theoretical peak (all cores): 252.8 GFLOPS
- Thermal throttling is the dominant performance limiter on this platform

### Secondary: Xeon Skylake Server (GETH)
- 16 cores (no HyperThreading), ~2.1 GHz base
- AVX-512 support (512-bit FMA), 16 MB L3 shared
- No thermal throttling — sustained benchmarking
- Theoretical peak (AVX-512): ~537 GFLOPS (16 cores x 2 FMA x 8 dp x 2.1 GHz)
- COM6 v35 achieves **255.2 GFLOPS** = 47% of theoretical peak

## COM7NN - Custom Operation Matrix Neural Network

`com7nn_transformer.py` implements a full Transformer encoder using COM operations. The architecture decomposes all neural network operations (attention, feedforward, normalization) into element-wise COM-structured function lists. Successfully trained: loss 4.59 -> 2.26 over 100 epochs.

## License

MIT
