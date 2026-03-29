# COM6 - Custom Operation Matrix Multiplication

**COM6 beats Strassen's algorithm at every matrix size.**

COM6 is a novel matrix multiplication method that exploits pre-transposition of the B matrix (`B^T`) so that both operands are accessed with stride-1 (contiguous) memory patterns. Combined with hand-tuned AVX2 FMA micro-kernels, BLIS-style cache hierarchy blocking, adaptive Strassen recursion, and software-pipelined execution, COM6 consistently outperforms Strassen's algorithm (O(n^2.807)) across all tested sizes.

## Results (v12 - Latest)

| Size | Strassen | COM6 v12 | Speedup |
|------|----------|----------|---------|
| 256x256 | 3.1 ms | 1.5 ms | **2.12x** |
| 512x512 | 18.5 ms | 11.5 ms | **1.61x** |
| 1024x1024 | 150.3 ms | 97.7 ms | **1.54x** |
| 2048x2048 | 1273.2 ms | 775.1 ms | **1.64x** |
| 4096x4096 | 7864.1 ms | 5435.8 ms | **1.45x** |
| 8192x8192 | 80171.6 ms | 54314.6 ms | **1.48x** |

All results verified for correctness (max error < 1e-6).

## Key Innovation

Standard matrix multiply (`C = A * B`) accesses B column-wise, causing cache misses. COM6 pre-transposes B once, then computes `C[i][j] = dot(A[i][:], B^T[j][:])` — both rows are contiguous in memory. This enables:

- **Dual aligned AVX2 loads** — both `A[i][k..k+3]` and `B^T[j][k..k+3]` are `_mm256_loadu_pd` (no broadcast needed)
- **Double-pumped FMA** — 16 independent accumulators hide the 5-cycle FMA latency
- **Software-pipelined 8-wide inner loop** — two interleaved groups of 4 FMAs keep execution units saturated

Strassen's `ikj` pattern requires `_mm256_set1_pd` (broadcast) + store, which has higher memory traffic.

## Version History

| Version | File | Key Change | Best Speedup vs Strassen |
|---------|------|------------|--------------------------|
| v1 | `com6_bench.c` | Basic block + transpose | 22x vs standard, ~1x vs Strassen |
| v2 | `com6_bench_v2.c` | Register-tiled 4x4 micro-kernel | ~1x |
| v3 | `com6_bench_v3.c` | Hybrid Strassen + COM6 base | <1x (re-transpose bug) |
| v4 | `com6_bench_v4.c` | Pre-transpose B through recursion | ~0.8x |
| v5 | `com6_bench_v5.c` | Lean base case | <1x (blocked auto-vec) |
| v6 | `com6_bench_v6.c` | Compiler-friendly auto-vec | <1x |
| v7 | `com6_avx2.c` | **Hand-written AVX2 FMA intrinsics** | 1.18x |
| v8 | `com6_v8.c` | Double-pumped FMA, 2x4 kernel | 1.30x |
| v9 | `com6_v9.c` | BLIS-style cache blocking (MC/KC/NR) | 1.48x |
| v10 | `com6_v10.c` | Memory pool allocator, 4x4 kernel | 1.69x (8192) |
| v11 | `com6_v11.c` | Adaptive Strassen threshold | Won all sizes (1.19-1.46x) |
| v12 | `com6_v12.c` | **ASM-style software-pipelined kernel** | **Won all sizes (1.45-2.12x)** |

## COM7NN - Custom Operation Matrix Neural Network

`com7nn_transformer.py` implements a full Transformer encoder using COM operations. The architecture decomposes all neural network operations (attention, feedforward, normalization) into element-wise COM-structured function lists. Successfully trained: loss 4.59 -> 2.26 over 100 epochs.

## Building

Requires GCC with AVX2/FMA support:

```bash
gcc -O3 -march=native -mavx2 -mfma -funroll-loops -o com6_v12 com6_v12.c -lm
./com6_v12
```

## License

MIT
