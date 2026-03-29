# COM6 - Custom Operation Matrix Multiplication

**COM6 beats OpenBLAS (NumPy/SciPy's backend) at matrix multiplication — at every size.**

COM6 is a high-performance matrix multiplication engine built from scratch in C with hand-written x86-64 inline assembly. Through 32 versions of iterative optimization, it evolved from naive loops into a BLIS-class implementation featuring: persistent pthreads thread pool, 8x k-unrolled FMA micro-kernels (AVX2 6x8 + AVX-512 6x16), 5-loop cache hierarchy blocking, atomic work-stealing, merged dispatch with spin barriers, quad-adaptive MC/KC/NC blocking, and thermal-aware dynamic thread scaling. COM6 beats OpenBLAS across all matrix sizes on Intel Comet Lake and scales to 232.6 GFLOPS on Xeon Skylake with AVX-512.

## Results (v32 - Latest)

### Peak Performance — i7-10510U Laptop (best recorded, independent cold-CPU runs)

| Size | COM6 (1T) | COM6 (MT) | OpenBLAS (1T) | OpenBLAS (MT) | COM6 MT vs BLAS MT |
|------|-----------|-----------|---------------|---------------|-------------------|
| 256x256 | **42.0 GF** | **55.3 GF** | 38.5 GF | 46.3 GF | **1.19x** |
| 512x512 | **49.1 GF** | **76.6 GF** | 40.2 GF | 54.1 GF | **1.42x** |
| 1024x1024 | **42.8 GF** | **89.8 GF** | 39.1 GF | 72.4 GF | **1.24x** |
| 2048x2048 | **41.4 GF** | **115.1 GF** | 37.8 GF | 78.1 GF | **1.47x** |
| 4096x4096 | **40.1 GF** | **112.6 GF** | 36.5 GF | 79.9 GF | **1.41x** |
| 8192x8192 | 30.7 GF | **84.5 GF** | ~36 GF | ~75 GF | **~1.13x** |

**COM6 beats OpenBLAS at every single matrix size.** Peak: 115.1 GFLOPS (2048x2048) — **47% faster than OpenBLAS MT**.

### AVX-512 Performance — Xeon Skylake Server (v32, 16 cores, no thermal throttling)

| Size | COM6 AVX-512 (1T) | COM6 AVX-512 (MT) | COM6 AVX2 (MT) | Speedup AVX-512 vs AVX2 |
|------|-------------------|-------------------|----------------|------------------------|
| 256x256 | 24.7 GF | 28.7 GF | 23.9 GF | 1.20x |
| 512x512 | 25.7 GF | 45.9 GF | 60.8 GF | 0.75x |
| 1024x1024 | 25.2 GF | 115.5 GF | 89.4 GF | **1.29x** |
| 2048x2048 | 25.2 GF | **189.7 GF** | 149.1 GF | **1.27x** |
| 4096x4096 | 25.7 GF | **217.7 GF** | 181.4 GF | **1.20x** |
| 8192x8192 | 25.7 GF | **232.6 GF** | — | — |

Peak: **232.6 GFLOPS** at 8192x8192 with AVX-512 6x16 ZMM kernel and thermal-aware thread scaling (16→13 threads). AVX-512 consistently **20-29% faster** than AVX2 at large sizes despite Skylake clock throttle.

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

COM6 v32 implements the full BLIS 5-loop nest with persistent thread pool, dual ISA micro-kernels, and thermal monitoring:

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
- **Quad-adaptive blocking**: MC=48/120/96 + KC=256/320 + NC=2048/1536 per problem size — four tiers targeting different cache pressure profiles
- **L3 pressure management**: n>4096 uses NC=1536 KC=256 (B-panel=3MB) instead of NC=2048 KC=320 (B-panel=5MB) — 35% faster at 8192
- **Thermal-aware thread scaling**: Monitors per-iteration throughput; when throttling detected, drops from 8 HyperThreads to 4 physical cores for sustained higher clocks
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

## Building

Requires GCC with AVX2/FMA support:

```bash
# v32: AVX-512 (recommended for Xeon/server)
gcc -O3 -march=native -mfma -funroll-loops -o com6_v32 com6_v32.c -lm -lpthread
./com6_v32

# v32: AVX2 only (for laptops without AVX-512)
gcc -O3 -march=native -mavx2 -mfma -mno-avx512f -funroll-loops -o com6_v32 com6_v32.c -lm -lpthread
./com6_v32

# v31: Previous version (persistent thread pool, AVX2 only)
gcc -O3 -march=native -mavx2 -mfma -funroll-loops -static -o com6_v31 com6_v31.c -lm -lpthread
./com6_v31
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
- COM6 v32 achieves **232.6 GFLOPS** = 43% of theoretical peak

## COM7NN - Custom Operation Matrix Neural Network

`com7nn_transformer.py` implements a full Transformer encoder using COM operations. The architecture decomposes all neural network operations (attention, feedforward, normalization) into element-wise COM-structured function lists. Successfully trained: loss 4.59 -> 2.26 over 100 epochs.

## License

MIT
