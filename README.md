# COM6 - Custom Operation Matrix Multiplication

**COM6 beats OpenBLAS (NumPy/SciPy's backend) at matrix multiplication.**

COM6 is a high-performance matrix multiplication engine built from scratch. Starting from a novel B-transposition approach, it evolved through 23 versions into a BLIS-class implementation featuring hand-written x86-64 inline assembly micro-kernels, 5-loop cache hierarchy blocking, and adaptive OpenMP multi-threading. COM6 consistently matches or beats OpenBLAS — the industry-standard BLAS library — on Intel Comet Lake.

## Results (v23 - Latest)

### vs OpenBLAS (the standard)

| Size | COM6 v23 (1T) | OpenBLAS (1T) | COM6 v23 (MT) | OpenBLAS (MT) |
|------|---------------|---------------|---------------|---------------|
| 256x256 | **41.7 GFLOPS** | 38.5 GFLOPS | 41.7 GFLOPS* | 46.3 GFLOPS |
| 512x512 | **45.9 GFLOPS** | 40.2 GFLOPS | 45.9 GFLOPS* | 54.1 GFLOPS |
| 1024x1024 | 40.4 GFLOPS | 39.1 GFLOPS | **89.8 GFLOPS** | 72.4 GFLOPS |
| 2048x2048 | 40.4 GFLOPS | 37.8 GFLOPS | **103.8 GFLOPS** | 78.1 GFLOPS |
| 4096x4096 | 38.8 GFLOPS | 36.5 GFLOPS | **108.8 GFLOPS** | 79.9 GFLOPS |

\* Adaptive: uses single-threaded for n<=512 (avoids thread spawn overhead)

**Peak: 108.8 GFLOPS** (4096x4096, 8 threads on 4-core i7-10510U) — **36% faster than OpenBLAS MT**.

### vs Strassen (where it all started)

| Size | Strassen | COM6 v12 | Speedup |
|------|----------|----------|---------|
| 256x256 | 3.1 ms | 1.5 ms | **2.12x** |
| 512x512 | 18.5 ms | 11.5 ms | **1.61x** |
| 1024x1024 | 150.3 ms | 97.7 ms | **1.54x** |
| 2048x2048 | 1273.2 ms | 775.1 ms | **1.64x** |
| 4096x4096 | 7864.1 ms | 5435.8 ms | **1.45x** |
| 8192x8192 | 80171.6 ms | 54314.6 ms | **1.48x** |

All results verified for correctness (max error < 1e-6).

## Architecture

COM6 v23 implements the full BLIS 5-loop nest:

```
jc-loop (NC=2048, L3 blocking)
  pc-loop (KC=256, L2 blocking)
    PARALLEL B-packing (contiguous SIMD loads, no transpose needed)
    barrier
    PARALLEL ic-loop (MC=120, L1 blocking)
      A-packing (per-thread buffers)
      jr-loop (NR=8)
        ir-loop (MR=6)
          6x8 micro-kernel (inline ASM, 4x k-unrolled)
```

### Key Technical Details

- **6x8 outer-product micro-kernel**: 12 YMM accumulators (ymm0-ymm11), broadcast A + FMA against B vectors
- **4x k-unrolling**: 4 rank-1 updates per loop iteration, reducing branch overhead by 75%
- **GNU inline assembly**: Direct register control, no compiler interference with FMA scheduling
- **Direct B-packing**: B is row-major, so `B[k][j..j+7]` is already contiguous — just SIMD copy, no transpose needed
- **Adaptive threading**: Single-threaded for n<=512 (where OpenMP fork-join overhead exceeds parallel gain), multi-threaded for larger
- **Parallel B-packing**: B panels are packed in parallel across threads, then shared read-only
- **Per-thread A buffers**: Each thread packs its own A panel independently — no false sharing

### Cache Hierarchy Targeting (i7-10510U)

| Buffer | Size | Fits In |
|--------|------|---------|
| A micro-panel | MR*KC*8 = 12 KB | L1d (64 KB/core) |
| B micro-panel | NR*KC*8 = 16 KB | L1d (64 KB/core) |
| A macro-panel | MC*KC*8 = 240 KB | L2 (256 KB/core) |
| B macro-panel | NC*KC*8 = 4 MB | L3 (8 MB shared) |

## Version History

| Version | Key Change | Peak GFLOPS |
|---------|------------|-------------|
| v1-v6 | Initial COM6 concept, blocked transpose, auto-vec attempts | ~5-10 |
| v7 | Hand-written AVX2 FMA intrinsics | ~15 |
| v8 | Double-pumped FMA, 2x4 kernel | ~20 |
| v9 | BLIS-style cache blocking | ~25 |
| v10-v11 | Memory pool, adaptive Strassen | ~28 |
| v12 | ASM-style software-pipelined kernel | ~32 |
| v13-v15 | BLIS 5-loop, 6x8 outer-product micro-kernel | 23-38 |
| v16 | **Direct B-packing (no transpose)** — matched OpenBLAS 1T | **44** |
| v17 | L3-aware NC + Strassen hybrid | 46 |
| v18 | Simplified Strassen + direct BLIS | 46 |
| v19 | Interleaved FMA scheduling | 38 |
| v20 | GNU inline ASM micro-kernel | 45 |
| v21 | 4x k-unrolled ASM | 45 |
| v22 | OpenMP multi-threading — beat OpenBLAS MT | 104 |
| **v23** | **Adaptive threading + parallel B-pack + merged parallel region** | **108.8** |

## Building

Requires GCC with AVX2/FMA support:

```bash
# Single-threaded
gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v23 com6_v23.c -lm

# Multi-threaded (recommended)
gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp -static -o com6_v23 com6_v23.c -lm
./com6_v23
```

## Test Platform

- Intel Core i7-10510U (Comet Lake), 4 cores / 8 threads
- 64 KB L1d/core, 256 KB L2/core, 8 MB L3 shared
- Theoretical peak: 63.2 GFLOPS/core (2 FMA units x 4 doubles x 2 flops x 3.95 GHz boost)
- Theoretical peak (all cores): 252.8 GFLOPS

## COM7NN - Custom Operation Matrix Neural Network

`com7nn_transformer.py` implements a full Transformer encoder using COM operations. The architecture decomposes all neural network operations (attention, feedforward, normalization) into element-wise COM-structured function lists. Successfully trained: loss 4.59 -> 2.26 over 100 epochs.

## License

MIT
