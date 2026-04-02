# COM6 - Custom Operation Matrix Multiplication

**COM6 beats OpenBLAS (NumPy/SciPy's backend) at matrix multiplication — at all sizes that matter (512+).**

COM6 is a high-performance matrix multiplication engine built from scratch in C with hand-written x86-64 inline assembly. Through 52 versions of iterative optimization, it evolved from naive loops into a BLIS-class implementation featuring: 8x k-unrolled FMA micro-kernels (AVX2 6x8 + AVX-512 6x16), 5-loop cache hierarchy blocking, OpenMP parallelization with adaptive NC/MC/KC blocking, beta=0 memset elimination, C-prefetch micro-kernels, distributed prefetch scheduling, load-balanced threading, and register-blocked small-matrix paths. COM6 beats OpenBLAS at 5/6 tested sizes on Intel Comet Lake (up to 18% faster) and scales to 255 GFLOPS with AVX-512 on Xeon.

## Results (v50 - Latest)

### v50 vs OpenBLAS MT (fair interleaved comparison, 3s cooling between tests)

| Size | OpenBLAS MT | COM6 MT | Ratio | Winner |
|------|-------------|---------|-------|--------|
| 256x256 | 90.0 GF | 57.5 GF | 0.64x | BLAS |
| 512x512 | 91.5 GF | **98.1 GF** | **1.07x** | **COM6** |
| 1024x1024 | 113.3 GF | **117.8 GF** | **1.04x** | **COM6** |
| 2048x2048 | 101.9 GF | **120.6 GF** | **1.18x** | **COM6** |
| 4096x4096 | 104.3 GF | **122.5 GF** | **1.17x** | **COM6** |
| 8192x8192 | 84.1 GF | **98.9 GF** | **1.18x** | **COM6** |

**COM6 wins 5/6 sizes.** Beats OpenBLAS by 4-18% at 512-8192. Only loses at 256 (OpenMP fork/join overhead on 33M FLOPs).

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

## Building

Requires GCC with AVX2/FMA support:

```bash
# v50: AVX2 + OpenMP (recommended for laptops, latest)
gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp -o com6_v50 com6_v50.c -lm
./com6_v50           # full benchmark
./com6_v50 4096      # single-size cold CPU test

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
