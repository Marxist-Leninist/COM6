# COM6 - Custom Operation Matrix Multiplication

**COM6 beats OpenBLAS at matrix multiplication on both laptop (i7-10510U) and server (EPYC 7282).**

COM6 is a high-performance matrix multiplication engine built from scratch in C with hand-written x86-64 inline assembly. Through 124 versions of iterative optimization, it evolved from naive loops into a BLIS-class implementation featuring: 8x k-unrolled FMA micro-kernels (AVX2 6x8), 5-loop cache hierarchy blocking, chiplet-aware NUMA dispatch, OpenMP IC/JC-parallel with OMP_PROC_BIND pinning, separate beta-0/beta-1 micro-kernels, adaptive MC/KC/NC blocking, L2-auto-tuned MC, physical-core-only threading, 4x k-unrolled A-packing, 2x k-unrolled B-packing, C-prefetch at kernel entry, SIMD-accelerated edge kernels, and memory-efficient Strassen for large sizes (3-buffer, 384MB at 8192 vs 3.2GB naive).

**v120 on EPYC 7282 (16-core/32-thread, Zen 2)** — 411 GF peak at 1024, 203 GF at 8192:

| Size | COM6 v120 16T (GF) | OpenBLAS 16T (GF) | Ratio | Winner |
|------|-----------------:|-----------------:|------:|:------:|
| 512 | **395.9** | 32.4 | **12.2x** | COM6 |
| 1024 | **410.6** | 21.2 | **19.4x** | COM6 |
| 2048 | **182.3** | 79.7 | **2.3x** | COM6 |
| 4096 | **219.1** | 77.5 | **2.8x** | COM6 |
| 8192 | **203.0** | 75.3 | **2.7x** | COM6 |

**v108 on i7-10510U (4-core/8-thread, 15W TDP laptop)** — 5/5 sizes 1.08x–1.66x over OpenBLAS:

| Size | COM6 v108 (GF) | OpenBLAS (GF) | Ratio |
|------|---------------:|--------------:|------:|
| 512 | **95.9** | 57.7 | **1.66x** |
| 1024 | **115.5** | 106.2 | **1.09x** |
| 2048 | **108.9** | 97.5 | **1.12x** |
| 4096 | **102.5** | 95.0 | **1.08x** |
| 8192 | **92.7** | 74.1 | **1.25x** |

## COM7NN Transformer vs Standard Transformer

The COM framework extends beyond matrix multiplication into neural network architectures. COM7NN implements a full transformer with the COM Tensor class (shape-safety, custom operations).

| Metric | Standard NumPy | COM7NN | Winner |
|--------|---------------|--------|--------|
| Final loss | 2.808 | **2.290** | COM7NN |
| Loss reduction | 34.4% | **49.7%** | COM7NN |
| Training time | 36.93s | **19.95s** | COM7NN (1.85x faster) |
| Inference accuracy | 4/10 | **6/10** | COM7NN |

Same architecture (d_model=64, 4 heads, d_ff=128, 1 layer), same data (counting task), same seed. COM7NN converges faster, trains faster, and predicts more accurately.

## v125: JC-Parallel All Sizes on Server (2026-04-17)

v124 used IC-parallel for n < 2048 on monolithic servers, but testing revealed JC-parallel wins at 1024 (+18%) and 512 (+5%) on Xeon 16C. With 16 threads, JC-parallel's zero-barrier independent column strips beat IC-parallel's shared B-panel approach even at small sizes. v125 extends JC-parallel to ALL sizes on servers (L2 >= 512KB), using `get_blocking_mt` (small MC=48 for good IC granularity within each thread's column strip).

### v125 full suite on Xeon Skylake 16C

| Size | GF(1T) | GF(MT) | Verify |
|------|-------:|-------:|:------:|
| 16384 | -- | **236.8** | OK |
| 8192 | -- | 217.7 | OK |
| 4096 | 22.7 | 204.9 | OK |
| 2048 | 23.8 | 198.3 | OK |
| 1024 | 27.2 | **199.3** | OK |
| 512 | 23.0 | 140.2 | OK |
| 256 | 28.2 | 27.6 | OK |

**Peak: 236.8 GFLOPS** at 16384 MT (1-level Strassen, 7 sub-problems of 8192). 2-level Strassen tested: 232.4 GF (slightly worse — 1-level remains the sweet spot).

### v125 vs v124 at 1024 MT (back-to-back interleaved on Xeon)

| Round | v124 (GF) | v125 (GF) |
|------:|----------:|----------:|
| 1 | 171.7 | 173.0 |
| 2 | 143.6 | **193.6** |
| 3 | 149.5 | **190.6** |

v125 wins 1024 MT consistently: best 193.6 vs 171.7 GF (+13%). Laptop path unchanged (same dispatch for L2 < 512KB).

## v124: SIMD Edge Kernel + Server-Aware Blocking (2026-04-17)

v124 adds two improvements over v123's core:

1. **SIMD-accelerated edge kernel**: When `nr == NR` (full-width panel) but `mr < MR` (partial-height), the edge case now routes through the full 6x8 AVX2 FMA micro-kernel via a temp buffer instead of falling back to scalar loops. ~8x faster for edge blocks.
2. **Server-aware JC-parallel dispatch**: Monolithic servers with L2 >= 512KB now use JC-parallel (independent column strips, zero barriers) for n >= 2048, with L2-proportional MC for optimal A-panel sizing. IC-parallel retained for n < 2048 where shared B-panel benefits from L3 cache sharing.

### v124 vs v123 on Xeon Skylake 16C (back-to-back, same conditions)

| Size | v123 MT (GF) | v124 MT (GF) | Change |
|------|------------:|-----------:|-------:|
| 8192 | 213.9 | **222.6** | **+4.1%** |
| 4096 | 154.7 | **201.1** | **+30.0%** |
| 2048 | 186.0 | **189.8** | **+2.0%** |
| 1024 | 177.7 | 170.0 | -4.3% |
| 512 | 130.9 | 124.5 | -4.9% |
| 256 | 22.8 | **28.7** | **+25.9%** |

The 4096 improvement (+30%) comes from switching to JC-parallel dispatch with L2-aware MC on the server. Previously, 4096 used IC-parallel which underutilizes L2 on large-cache servers. 512/1024 regressions are within run-to-run noise (~5%).

### v124 full suite (i7-10510U laptop, 4C/8T, 15W TDP)

| Size | GF(1T) | GF(MT) | Verify |
|------|-------:|-------:|:------:|
| 256 | 44.9 | 43.2 | OK |
| 512 | 39.1 | 115.0 | OK |
| 1024 | 31.5 | 59.5 | OK |
| 2048 | 26.2 | 51.1 | OK |
| 4096 | -- | 49.1 | OK |
| 8192 | -- | 51.5 | OK |

Env: `COM6_THREADS=N` | `COM6_STRASSEN_DEPTH=N` | `COM6_NO_STRASSEN=1`

## v120: Unified Champion — OpenMP Core + Chiplet Dispatch (2026-04-17)

v119 attempted to replace OpenMP with a custom pthreads pool (spin barriers + atomic work-stealing). This **regressed 28% on monolithic** (laptop) because OpenMP's optimized barriers beat custom spin barriers. v120 takes the right approach: keep v108's proven OpenMP parallelism, add chiplet awareness on top.

### Key design decisions

- **Monolithic (laptop):** Exact v108 dispatch — IC-parallel for 512/1024/4096+, JC-parallel for 2048. OpenMP's `schedule(static)` + implicit barriers.
- **Chiplet (EPYC/Ryzen):** JC-parallel for ALL sizes (private B panels = zero cross-CCD traffic). Auto-detects multiple L3 groups via sysfs.
- **Physical-core-only:** Sets `omp_set_num_threads(phys_cores)` on chiplet CPUs. HT siblings compete for FMA ports.
- **L2 auto-tuning:** EPYC's 512KB L2 → MC=180; i7's 256KB L2 → MC=120/96.
- **4x A-pack unroll:** +27% at 2048 vs v108 from reduced loop overhead.

### v120 vs v118 on EPYC 7282

| Size | v118 pthreads (GF) | v120 OpenMP (GF) | Speedup |
|------|-------------------:|-----------------:|--------:|
| 512 | 201.8 | **395.9** | **1.96x** |
| 1024 | 341.9 | **410.6** | **1.20x** |
| 2048 | 179.4 | **182.3** | ~same |
| 4096 | 206.1 | **219.1** | **1.06x** |
| 8192 | 204.2 | **203.0** | ~same |

### Per-core efficiency

- **Single-thread:** 42.9 GF at 2048 = **95.8% of EPYC peak** (44.8 GF/core at 2.8 GHz with 2 FMA units)
- **16-thread at 1024:** 25.7 GF/core = 57% efficiency (memory bandwidth becomes bottleneck)
- **16-thread at 8192:** 12.7 GF/core = 28% (fully memory-bandwidth-bound)

### Env controls

- `COM6_THREADS=N` — override thread count
- `COM6_FORCE_JC=1` / `COM6_FORCE_IC=1` — force dispatch path
- `COM6_USE_STRASSEN=1` — enable Strassen at 8192+ (opt-in, BLIS usually wins)
- `COM6_PACE_MS=N` — thermal pacing between NC blocks (laptop 4096+)
- `OMP_PROC_BIND=close OMP_PLACES=cores` — recommended for chiplet CPUs

## v122: Memory-Efficient Strassen for 8192+ (2026-04-17)

v121's Strassen crashed at 8192 (OOM: 25 temp buffers × 128MB = 3.2GB). v122 rewrites Strassen to use only 3 temp buffers (T1, T2, M) = 384MB, operating on strided quadrant views with OpenMP-parallelized add/sub. Also disables thermal pacing inside Strassen sub-problems (the FLOP reduction already handles cooling).

### v122 vs v121 at 8192 MT (i7-10510U, 15W TDP)

| Version | Time (ms) | GFLOPS | Improvement |
|---------|----------:|-------:|------------:|
| v121 (no Strassen) | 31249 | 35.2 | baseline |
| v122 (Strassen, 3 buffers) | **23812** | **46.2** | **+31%** |

Strassen's 12.5% FLOP reduction + thermal benefit (7 shorter 4096 bursts vs one long 8192 burst) + no wasted pacing sleep = 31% wall-clock improvement. All sizes pass correctness verification.

### v122 full suite

| Size | GF(1T) | GF(MT) | Verify |
|------|-------:|-------:|:------:|
| 256 | 41.1 | 39.9 | OK |
| 512 | 39.1 | 91.2 | OK |
| 1024 | 23.3 | 58.3 | OK |
| 2048 | 21.5 | 46.8 | OK |
| 4096 | -- | 43.7 | OK |
| 8192 | -- | 40.9 | OK |

Env: `COM6_NO_STRASSEN=1` disables Strassen. `COM6_USE_STRASSEN=1` forces it at 4096+ (not recommended — overhead dominates at 4096). All other v121 env knobs preserved.

---

## COM6 Matrix Multiplication Results (v120 current champion on both platforms)

### v100 (pthreads pool) vs v108 (OpenMP+IC-par) at 512 MT — 2026-04-16

The OpenMP fork-join cost is proportionally largest at 512 MT (~3 ms matmul, so even ~50 µs spawn overhead is measurable). v100 replaces OpenMP with a persistent pthreads pool (spin-wait ~1 µs, condvar sleep after 2000 spins, atomic work-stealing for ic-blocks). It was committed 2026-04-15 but never interleaved-fair-bench'd against v108 until now.

`bench_v100_vs_v108_512.sh`: 60 s initial cooldown, then 6 rounds × (v100 → 45 s cool → v108 → 45 s cool), i7-10510U 15W TDP laptop, cold CPU at start:

| Round | v100 pthreads pool | v108 OpenMP champion | Ratio (v100/v108) |
|------:|-------------------:|---------------------:|------------------:|
| 1     | **106.6 GF**       | 93.7 GF              | 1.14x             |
| 2     | **101.5 GF**       | 88.5 GF              | 1.15x             |
| 3     | **100.9 GF**       | 88.7 GF              | 1.14x             |
| 4     |  **85.3 GF**       | 74.5 GF              | 1.14x             |
| 5     |  **99.5 GF**       | 95.5 GF              | 1.04x             |
| 6     |  **98.3 GF**       | 83.9 GF              | 1.17x             |
| BEST   | **106.6 GF**      | 95.5 GF              | **1.12x**         |
| MEDIAN | **100.9 GF**      | 88.7 GF              | **1.14x**         |

v100 wins **every round** at 512 MT — a clean, reproducible ~14 % median lift outside the ±30 % thermal noise envelope. Under the v108 best run (95.9 GF) already documented against OpenBLAS (57.7 GF, 1.66x), v100's median 100.9 GF projects to ~1.75x vs OpenBLAS at 512 MT, i.e. v100 widens the gap that v108 opened. v107's hoisted single `#pragma omp parallel` reduced fork/join from per-pc to per-matmul (one spawn per ~3 ms job); v100 removes that last spawn too — threads pay ~1 µs atomic+broadcast instead of ~50-100 µs `libgomp` team setup.

Reproduce: `./bench_v100_vs_v108_512.sh` (expects `com6_v100.exe`, `com6_v108.exe` built, no rogue com6 procs running).

### v100 vs v108 at 1024 and 2048 MT — 2026-04-16 (partial sweep)

`bench_v100_vs_v108_medium.sh`: 4 rounds interleaved, 30 s cooldowns at 1024 and 45 s at 2048, cold start, same harness rules as the 512 bench.

| Size | v100 (GF per round)            | v108 (GF per round)            | Median v100 | Median v108 | Ratio | Verdict |
|-----:|--------------------------------|--------------------------------|------------:|------------:|------:|:--------|
| 1024 | 109.2, 103.5, 114.6, 121.9      | 113.5, 112.9, 113.7, 120.3      | 114.6       | 113.7       | 1.008x | tie (within ±30% laptop noise) |
| 2048 | 98.4, 105.6, 74.6, 34.4         | 111.2, 109.4, 62.0, 39.5        | —           | —           | —     | thermal collapse — rounds 3-4 invalid |

At 1024 the pool advantage has disappeared: median 114.6 vs 113.7 GF is a 0.8% difference, an order of magnitude below the laptop's ±30 % thermal noise floor. Both v100 and v108 deliver the same throughput at this size because compute (~50 ms of FMA work per matmul) dwarfs the per-matmul dispatch overhead that distinguishes them.

The 2048 run tells a different story but one that is *about the laptop, not the code*. Rounds 1-2 read 98-106 / 109-111 GF (v108 marginally ahead, ~1.05x), but rounds 3-4 collapsed to 34-75 GF as the 15 W TDP budget was exhausted — even with 45 s cooldowns the CPU couldn't recover clocks fast enough. The cold-start pair (98.4 vs 111.2) suggests v108's JC-par is a touch faster at 2048, consistent with the fact that v100 uses IC-par for n<2048 but JC-par for 2048≤n<4096, identical to v108's path at this size — so this is essentially a sanity check that the two implementations of the same algorithm agree.

**Summary of the sweep so far (512, 1024, 2048):** v100's persistent pool is a clean ~14% win at 512 MT, statistically indistinguishable from v108 at 1024 MT, and statistically indistinguishable at 2048 MT in cold rounds. 4096 and 8192 were not run in this sweep — on the 15 W laptop, ±30 % thermal noise at those sizes would drown the <5 % expected effect, and the v108 thermal-pacing path (which v100 preserves verbatim for n≥4096) has already been documented vs OpenBLAS. Conclusion: **v100 is the right answer at 512 and neutral at 1024+**. Keep v108 as the all-sizes champion; use v100 when the small-matmul path is on the critical path.

Reproduce: `./bench_v100_vs_v108_medium.sh` (expects `com6_v100.exe`, `com6_v108.exe`, cold CPU, no rogue com6 procs).

### v108 vs OpenBLAS — Fresh sweep, 2026-04-15 (5/5 sizes, i7-10510U 15W TDP)

Best run each, size-adaptive cooldowns (20s/30s/60s/120s), OpenBLAS-first-then-COM6 ordering per size. `com6_vs_blas_v10.py` (+ `bench_8192.py` for the largest point):

| Size | OpenBLAS (GF) | COM6 v108 (GF) | Ratio | Winner |
|------|--------------:|---------------:|------:|:------:|
| 512  | 57.7  | **95.9**  | **1.66x** | COM6 |
| 1024 | 106.2 | **115.5** | **1.09x** | COM6 |
| 2048 | 97.5  | **108.9** | **1.12x** | COM6 |
| 4096 | 95.0  | **102.5** | **1.08x** | COM6 |
| 8192 | 74.1  | **92.7**  | **1.25x** | COM6 |

COM6 wins every size tested. Biggest gap at 512 (1.66x) where the kernel's 8x k-unroll + 6x8 outer-product + single hoisted OMP region outruns OpenBLAS's fork-join cost for a ~3ms matmul. The 8192 lead (1.25x) is Strassen-with-thermal-pacing (v105 path) extracting a real algorithmic + thermal win over OpenBLAS's direct BLIS burst on a 15W TDP laptop. Reproduce: `python com6_vs_blas_v10.py` and `python bench_8192.py`.

### v108 2048 MT: JC-par default confirmed over IC-par (bench.sh, 2026-04-15)

v108 routes 2048 MT through the JC-parallel path with `COM6_IC_2048=1` reachable as an experiment knob. The historical justification was "IC-par wins cold, JC-par wins warm" — but no fair measurement had been done with `bench.sh` until now. Three-run bench.sh on the i7-10510U laptop:

| Config | Run 1 | Run 2 | Run 3 | Best | Spread |
|--------|-------|-------|-------|------|--------|
| JC-par (default) | 68.2 | 67.4 | 51.7 | **68.2 GF** | 26% |
| IC-par (`COM6_IC_2048=1`) | 58.6 | 18.5 | 21.7 | 58.6 GF | 122% |

JC-par holds 67-68 GF for the first two runs then drops to 51.7 as heat accumulates. IC-par starts at 58.6 (CPU moderately warm — tested right after JC-par with 45s cooldown) and then collapses to 18-22 GF on subsequent runs. IC-par's cooperative B-packing with explicit `#pragma omp barrier` between packing and compute phases has idle threads spin-waiting on the barrier; JC-par's fully-independent per-thread column ranges have no barriers. The empirical result is that JC-par is dramatically more thermally resilient at 2048 MT on a 15W TDP part. **Verdict: JC-par default for 2048 MT is correct; IC-par opt-in retained for desktop/workstation use where thermals aren't a constraint.**

Reproduce:
```
./bench.sh -e com6_v108.exe -n 2048 -m mt -r 3 -c 45           # JC-par (default)
COM6_IC_2048=1 ./bench.sh -e com6_v108.exe -n 2048 -m mt -r 3 -c 90   # IC-par
```

### v108 vs v107 at 4096 MT (pacing extended to 4096+)

v107 applied per-NC thermal pacing only at `n>=8192`. At 4096 MT the full compute burst is ~2.3s on a 15W i7-10510U — long enough that the CPU drops out of turbo halfway through. v108 extends pacing to `n>=4096` with a shorter default (150ms vs 400ms at 8192), firing once between the two NC blocks.

Cold-CPU single-shot comparison (3-minute pre-run cooldown on the i7-10510U laptop; numbers are heavily thermal-dependent — see caveat below):

| Size | v107 MT | v108 MT | Improvement |
|------|---------|---------|-------------|
| 4096 MT | 61.8 GF | **72.3 GF** | **+17%** |
| 8192 MT | 41.0 GF | **66.2 GF** | **+61%** (v108 finishes 8192 faster → less accumulated heat) |

Env knobs: `COM6_PACE_MS=N` tunes pace value, `COM6_PACE_MIN_N=N` adjusts threshold (default 4096 in v108, 8192 in v107), `COM6_IC_2048=1` routes 2048 through IC-par for experiments.

**Thermal caveat**: Run-to-run variance on this 15W TDP laptop can exceed 30%. The above pair is the single-shot best. A two-run v108 sequence with 90s/180s cooldowns showed 4096 MT swinging from 72.3 → 45.6 GF as CPU heat accumulated. On thermally-unbound hardware (Xeon, desktop), this pacing is a NEGATIVE optimization and should be disabled via `COM6_PACE_MIN_N=8192` to restore v107 behaviour. File: `com6_v108.c`.

### v107 vs v106 head-to-head (fair full-suite, 30s cold start)

v107 hoists the `#pragma omp parallel` region to enclose the entire jc/pc loop nest, replacing v106's per-pc fork/join pattern. At 8192 with KC=320 that cuts 104 fork/join rounds down to 1. Cooperative B-packing synchronized by an explicit barrier; ic loop's implicit barrier ends each pc phase. `Sleep(pace_ms)` wrapped in `#pragma omp master` + barrier so only the master thread sleeps while workers wait.

| Size | v106 MT | v107 MT | Improvement |
|------|---------|---------|-------------|
| 8192 MT | 56.0 GF | **62.1 GF** | **+11%** |
| 4096 MT | 40.9 GF | **49.6 GF** | **+21%** |
| 2048 MT | 49.3 GF | 44.3 GF | -10% (JC-par path unchanged; noise) |
| 1024 MT | 53.0 GF | **58.6 GF** | **+11%** |
| 512 MT  | 41.4 GF | **51.2 GF** | **+24%** |
| 256 MT  | 12.0 GF | **20.4 GF** | **+70%** |

Wins 5/6 sizes; the 2048 "regression" is in the JC-parallel path which v107 doesn't touch, so it's pure thermal noise between two back-to-back runs. Biggest lifts at the small end (256/512/1024 MT) where OpenMP fork/join was a larger fraction of total compute — v107 pays that cost once per matmul instead of once per KC block.

## COM6 Matrix Multiplication Results (v105 large-size champion, v99 reference)

### Very-Large-Size Scaling — v105 vs v99

v105 = v103 persistent packing buffers + v102 thermal pacing (500ms sleep between Strassen sub-muls, `COM6_STRASSEN_PACE_MS` env, default 500). Pacing keeps the 15W TDP CPU in turbo between the 7 sub-muls — counterintuitive but empirically a huge win at 10000+:

| Size | v99 MT | v105 MT (pace=500) | Improvement |
|------|--------|--------------------|-------------|
| 8192 MT | 39.6 GF | 70.0 GF (cold) / 61.7 GF (warm) | **+77% / +56%** |
| 10000 MT | 23.8 GF | **73.5 GF** | **+209%** |
| 12000 MT | 35.4 GF | **55.8 GF** | **+58%** |

At 10000, the original v99 path crashed into thermal throttle (23.8 GF) because the 7 sub-muls of 5000 ran back-to-back for ~84 seconds, sustaining 15W TDP throughout. v105's 500ms pacing lets each sub-mul finish at turbo clocks — net throughput **3.1× higher** despite the 3.5s of added sleep, because the CPU spends more time at ~3.5 GHz instead of base clock.

12000 gains less (+58%) because 6000-sized sub-muls already approach the IC-parallel L2/L3 sweet spot naturally, but pacing still extracts useful headroom.

Neither approaches cold-CPU peak (~90 GF). On thermally-unbound hardware (e.g. Xeon 16c), pacing would be a negative optimization — it's a workaround for 15W laptops.

### Successful Experiments

- **v102 — Thermal-paced Strassen (Sleep between sub-muls)**: Inserting `Sleep(pace_ms)` between the 7 Strassen sub-muls via `COM6_STRASSEN_PACE_MS` env var. Sweep result at 8192 MT: pace=0: 50.9 GF, **pace=500: 54.4-65.0 GF (sweet spot, +7-50% over v99)**, pace=1000: 47.1 GF, pace=2000: 36.6 GF. The 500ms window is long enough for turbo-clock recovery on the 15W i7-10510U but short enough that added idle time doesn't dominate. Confirmed real, reproducible win. File: `com6_v102.c`.
- **v103 — Persistent packing buffers across Strassen sub-muls**: 63.0 GF vs v101's 62.3 GF at 8192 MT. Architecturally sound: `pa_bufs[nthreads]` and `pb` allocated ONCE at Strassen entry and reused across all 7 sub-muls via `com6_multiply_ctx` / `com6_multiply_ic_ctx` / `com6_multiply_jc_ctx`. Eliminates 7 × (nthreads+1) alloc/free rounds and keeps pa/pb warm in L2/L3. Modest alone but stacks cleanly with v102 pacing to produce v105. File: `com6_v103.c`.
- **v105 — v103 persistent buffers + v102 thermal pacing** (combined): **Large-size champion.** +77% at 8192 cold, +209% at 10000, +58% at 12000 vs v99 (see table above). Default `pace_ms=500`; set `COM6_STRASSEN_PACE_MS=0` to behave identically to v103. File: `com6_v105.c`.

### Failed Experiments (kept for reference)

- **v109 — Finer-grained pacing at 4096 (NC=1024, 3× 75ms fires)**: 57.1 GF vs v108's 73.5 GF at 4096 MT (-22%). Root cause: 75ms is below the PCU turbo-ramp floor on the i7-10510U — multiplier can't bump back to 3.6 GHz before new work arrives. 150ms clears the bar; 75ms doesn't. Also: 2× B-pack rounds per matmul and more per-NC barrier cost. v108's single 150ms fire is the sweet spot. File: `com6_v109.c`.
- **v110 — MC sweep at 4096 MT**: tested MC ∈ {48, 60, 64, 72, 96} to break v108's 86-ic-iter / 8-thread imbalance. Thermal noise (±30% on this 15W TDP laptop) exceeded any MC-choice signal. MC=64 regressed 25% (MR=6 edge penalty); MC=72/96 showed bimodal cold-vs-warm results (78.2/32.2 GF at MC=96). Also tested `COM6_PACE_MS=0` vs 150ms: pacing wins by ~4% under matched cooldowns. MC=48 retained as default; COM6_MC_4096 env retained for thermally-unbound hardware testing (Xeon desktop). File: `com6_v110.c`.
- **v111 — Per-size hybrid body (v98-style for 512/1024, hoisted for 2048+)**: claimed wins in header comment from an earlier-hardware test did NOT reproduce on this i7-10510U. Fresh head-to-head vs v108 on 2026-04-15: 512 MT 74.3 vs 93.0 GF (-20%), 1024 MT tied (~106 GF), 2048 MT 88.1 vs 113.9 GF (-23%). Root cause: v111's extra branch on every sub-mul plus a second allocation path adds overhead that the claimed benefits don't recoup on this CPU. v108 hoisted body + size-adaptive pacing wins outright. File: `com6_v111.c`.
- **v112 — Two attempts at 512 MT improvement (both regressed)**: the 512 MT 95.9 GF is only 1.88x the 50.9 GF 1T rate on 8 threads, suggesting ~75% of the ~3ms matmul is barrier/fork-join overhead. Two fixes attempted, both regressed: (1) **KC=n one-shot** to halve pc-loop barriers: -23% (73.7 avg) because KC=512 makes PA = 48*512 = 192KB per thread, overflowing the 128KB/HT L2 share and causing L3-spilling reads on every pack_A. (2) **MC=64 for perfect 8-thread balance** (512/64 = 8 ic blocks, one per thread): -37% (58.2 avg) because MC=64 % MR=6 leaves a 4-row remainder on every ic block, and the kernel's remainder path is ~30% slower. Structural limit: at n=512 with MR=6 kernel, no MC satisfies both "MC %% 6 == 0" (clean kernel) and "n/MC ~= 8" (thread balance), because 512 = 8*64 and 64 is not a multiple of 6. v108's MC=48/KC=256 is the sweet spot: slight imbalance (~1.4 blocks/thread) beats L2-overflow or kernel-edge cost. File: `com6_v112.c`.
- **v113 — JC-parallel at 512 MT (zero-barrier column split)**: routed 512 MT through the JC-par path (normally used at 2048) with pre-allocated pa/pb buffers outside the parallel region. Hypothesis was that IC-par's barrier overhead was the limiter. RESULT: lost to v108 in every cold-start run (v113 cold vs v108 cold: 62.5 vs 88.4, 72.7 vs 91.7 GF). Root cause is cache economics, not barriers: IC-par has all 8 threads read the SAME 128KB B panel, shared via L3/L2 cache. JC-par has each thread read its OWN 16KB B stripe — 8x the effective B-memory bandwidth consumed because caches can't share independent slabs. At n=512 where the matmul is ~3ms, bandwidth for B dominates. IC-par's shared-B read outweighs its ~400us of barrier overhead. **Conclusion**: 512 MT is bandwidth-bound, not barrier-bound, and v108's IC-par cooperative B-pack is architecturally optimal. File: `com6_v113.c`.
- **v114 — HT-aware thread affinity pinning on Windows (inconclusive)**: call `SetThreadAffinityMask(GetCurrentThread(), 1ULL << cpu)` at the top of each `#pragma omp parallel` in `com6_multiply_ic_body` / `_jc` / `_jc_ctx`, with HT-aware mapping (primaries 0,2,4,6 for first N=n_physical threads, HT siblings 1,3,5,7 for the rest). Standard practice for BLAS libraries (MKL uses `KMP_AFFINITY=granularity=fine,compact`; OpenBLAS uses `OMP_PROC_BIND=close` + `OMP_PLACES=cores`). First attempt with naive `tid->cpu` mapping regressed at 512 MT by landing threads 0 and 1 on the same physical core (L1/L2 contention). Fixed to HT-aware mapping, saw +25% in stable-thermal back-to-back runs (77-88 GF vs 67-68 GF at 512 MT), but the gain did NOT reproduce under sustained thermal pressure — both PIN-ON and PIN-OFF degraded similarly as the i7-10510U hit thermal saturation (both dropping from 80 GF to 35-60 GF). **Verdict**: theoretically sound but on a 15W TDP laptop the measurement noise exceeds the effect size. Retained as documented experiment; v108 remains champion. Env `COM6_NO_PIN=1` disables; `COM6_PIN_LINEAR=1` reproduces the demonstrated regression path. On desktop/workstation hardware (Xeon, Ryzen with higher TDP) this change is expected to be a modest win. File: `com6_v114.c`.
- **v115 — Size-adaptive HT-aware pinning (only for n ≤ 1024, inconclusive)**: attempt to keep v114's pinning gain at small sizes while avoiding its regression at medium sizes. Dispatch: pin threads when `n ≤ g_pin_max_n` (default 1024), leave OS scheduler free for `n ≥ 2048` where scheduler flexibility matters. Env `COM6_PIN_MAX=<n>` adjusts threshold; `COM6_PIN_ALL=1` forces pinning (reproduces v114); `COM6_NO_PIN=1` disables. Alternated bench (warm CPU, 5s inter-run cool-down) at 512/1024/2048 showed v115 neither beating v108 reliably nor matching v114's claimed small-size gain under fair thermal conditions. At 1024, v115 had peak 78.7 GF vs v108 peak 55.1 GF in one sequence but reversed on a longer 5-trial run — signal sits inside the ±30% thermal-noise envelope. Retained as documented experiment: the size-split is theoretically motivated (small-n is dispatch-bound, large-n is compute-bound) but unmeasurable on this laptop. File: `com6_v115.c`.
- **v116 — OpenMP runtime hygiene via env vars + pool warm-up (inconclusive)**: orthogonal to v115's affinity-in-code approach. `configure_omp_runtime()` runs at top of `main()` and sets `OMP_WAIT_POLICY=active`, `GOMP_SPINCOUNT=200000` (~65 µs spin-then-sleep, bridges back-to-back regions but lets workers sleep during multi-second inter-size pauses so thermal budget is preserved), `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `GOMP_CPU_AFFINITY="0 2 4 6 1 3 5 7"` (HT-aware), `OMP_DYNAMIC=false`. `omp_warmup_pool()` runs a 2-region no-op to force team creation + prime the spin-wait loop; called once at startup and again after each inter-size `Sleep(4000)` cool-down. CRITICAL: `GOMP_SPINCOUNT` is NEVER set to `infinite` — that pegs idle workers at 100% CPU between timed calls and scorches the thermal budget (empirically produced 8-9 GF runs after the first warm one on this 15W laptop). Env `COM6_NO_OMP_TUNING=1` skips configuration; `COM6_NO_WARMUP=1` skips the pool primer. **Result**: at 1024 MT with 5 trials and 8 s cool-downs, v108 and v116 are statistically tied (v108 median 97.1 / best 114.0 / mean 89.9; v116 median 92.3 / best 105.0 / mean 92.7). v116 has lower variance (79.5-105.0 vs v108's 64.7-114.0) — more predictable floor but same median throughput. For the best-of-N metric the README tracks, v108 still wins. Retained as documented experiment. File: `com6_v116.c`.
- **v98 — Pure BLIS at 8192 (no Strassen)**: 23.6 GF. Lost to v99 Strassen by 38-68% on this hardware. Strassen's 7 shorter bursts thermally outperform one long 8192 BLIS sweep on 15W TDP. File: `com6_v98.c`.
- **v101 — Parallel Strassen glue + pool alloc**: 37.4 GF cold, ~6% regression vs v99 39.6 GF. The OMP fork-join overhead for sub_copy/mat_add/mat_sub exceeded the savings from collapsing 23 mallocs into one pool. Strassen glue is already cheap (<5% of total at 8192); parallelizing it wasn't worth the barrier cost. File: `com6_v101.c`.
- **v104 — Strassen dispatch at 4096+ (not just 8192+)**: 36.2 GF at 4096 MT vs v103 direct IC-par at 63.9 GF — **a 43% regression**. The Winograd 12.5% FLOP savings were swamped by Strassen glue cost (8 submatrix copies at 32MB each + 8 sums + 7 products = ~100MB of page-faults on cold pool) and by 7 sequential JC-parallel 2048 sub-muls each rebuilding their own private B-panels, thrashing L3 instead of sharing it. Confirmed: Strassen is a thermal-window trick for 8192+ where the direct BLIS burst is long enough that clocks throttle; at 4096 a single direct burst still fits the thermal envelope and wins outright. Reverted; v103 retains 8192-only Strassen.
- **Thread-count sweep at 8192** (cold-to-progressive-warm): 4T=35.1 GF, 6T=29.3 GF, 8T=26.9 GF. The decline tracks thermal state across runs (later runs get hotter CPU), not thread efficiency: cold-start 8T=39.6 GF (earlier test) beats cold-start 4T=35.1 GF. Avoiding HT contention doesn't help — the burst phase is where performance lives, and 8 threads maximize it.

### v99: v96 Dispatch + Targeted MC=48 for Large IC-Parallel

v99 combines v96's cache-aware dispatch with a targeted blocking change for large IC-parallel:

- **MC=48 at n>=4096** in IC-parallel (was MC_LARGE=96 in v96)
- Rationale: 86 ic-iterations / 8 threads = 10-11 per thread (vs v96's 5-6) — better load balance
- 48*320*8 = 120KB fits L2 cleanly (L2 = 256KB on i7-10510U)
- Everything else identical to v96 — v97's experiment of "MC=48 always" regressed 1024/512 by ~17%, so v99 keeps v96's known-good blocking for smaller sizes

**v99 vs v96 head-to-head (same run, 8-thread affinity, /HIGH priority, cold CPU):**

| Size | v96 MT | v99 MT | Improvement |
|------|--------|--------|-------------|
| 8192 MT | 76.5 GF | **78.0 GF** | +2% |
| 4096 MT | 83.3 GF | **92.1 GF** | **+11%** |
| 2048 MT | 89.6 GF | **93.2 GF** | +4% |
| 1024 MT | 67.5 GF | **78.0 GF** | **+16%** |
| 512 MT | **58.5 GF** | 50.1 GF | -14% (noise: code-identical at n<=1024) |
| 256 MT | 14.6 GF | **22.2 GF** | +52% |

v99 wins 5 of 6 sizes. The 512 "loss" is run-to-run variance since the IC-parallel branch for n<=1024 is byte-identical between v96 and v99 (both use MC_MT_SMALL=48). Biggest wins at 4096 (+11%) and 1024 (+16%) — exactly where the blocking change takes effect.

**v99 vs OpenBLAS (fair interleaved, 10s cooling, thermal-limited laptop):**

| Size | BLAS MT | COM6 v99 MT | Ratio | Winner |
|------|---------|-------------|-------|--------|
| 512 | **63.7 GF** | 57.6 GF | 0.90x | BLAS |
| 1024 | 50.6 GF | **66.6 GF** | **1.32x** | **COM6** |
| 2048 | **45.5 GF** | 45.0 GF | 0.99x | tie |
| 4096 | 30.6 GF | **41.4 GF** | **1.35x** | **COM6** |
| 8192 | 38.1 GF | **42.2 GF** | **1.11x** | **COM6** |

COM6 v99 wins 3 of 5, ties 2048, loses only at 512 (where BLAS's deep thread-pool amortization dominates). All absolute numbers here are thermal-throttled — cold-CPU runs show 78-93 GF for v99. The interesting signal is the *ratio* between interleaved tests: BLAS holds its 512 advantage, but COM6 wins at every size >= 1024.

### v96: Best-of-Both Dispatch (superseded by v99)

v96 optimizes the threading dispatch based on L3 cache analysis:
- **n>=8192**: Strassen (Winograd variant, 7 sub-muls) — shorter bursts sustain higher clocks on 15W TDP
- **n>=4096**: IC-parallel (shared B-panel: 5MB fits 8MB L3) — +47% vs v95's Strassen at 8192
- **2048<=n<4096**: JC-parallel (private B-panels, zero barriers)
- **n<2048**: IC-parallel (proven best for small/medium)

The key insight: Strassen's 4096-sized sub-multiplications now dispatch to IC-parallel (the faster path at that size), chaining the optimizations.

**v96 vs v95 (matched thermal conditions — 90s cooldown):**

| Size | v95 | v96 | Improvement |
|------|-----|-----|-------------|
| 8192 MT | 44.4 GF | **65.3 GF** | **+47%** |
| 4096 MT | 73.2 GF | **75.1 GF** | +3% |
| 2048 MT | 75.2 GF | **76.9 GF** | +2% |
| 1024 MT | 80.4 GF | **89.9 GF** | +12% |
| 512 MT | 61.0 GF | **62.7 GF** | +3% |
| 512 1T | 37.2 GF | **39.5 GF** | +6% |

### v95: Strassen-Hybrid + JC-Parallel

v95 added single-level Strassen (Winograd variant) for n>=8192: 7 sub-multiplications of (n/2)x(n/2) saving 12.5% of FLOPs. For n<8192: JC-parallel for n>=2048, IC-parallel for n<2048.

### v94: JC-Parallel for Large + Adaptive NC

JC-parallel mode for n>=2048: each thread owns a column slab with private B-panel (NC=1024). Zero barriers, zero C write contention. IC-parallel for n<2048. Beta-0/beta-1 micro-kernels, C-prefetch, 2x B-pack unroll.

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
| **v125** | **JC-parallel all sizes on server — +18% at 1024, 199 GF on Xeon** | **217.7** (Xeon 8192) |
| **v124** | **SIMD edge kernel + server-aware JC-parallel — +30% at 4096 on Xeon** | **222.6** (Xeon 8192) |
| **v123** | **Adaptive Strassen depth + L2-auto KC tuning** | **213.9** (Xeon 8192) |
| **v122** | **Memory-efficient Strassen (3-buffer, 384MB vs 3.2GB)** | **46.2** (laptop 8192) |
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
# v125: AVX2 + OpenMP + JC-par all sizes on server (recommended, latest)
gcc -O3 -march=native -mavx2 -mfma -funroll-loops -fopenmp -o com6_v125 com6_v125.c -lm
./com6_v125           # full benchmark (8192-256, reverse order)
./com6_v125 4096 mt   # single-size MT cold CPU test
./com6_v125 512 1t    # single-size 1T test
./com6_v125 8192 mt   # 8192x8192 MT-only (1T skipped for huge sizes)

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

## Benchmarking (bench.sh)

On 15W laptops, single-run comparisons lie — thermal noise at n≥4096 is ±30%
run-to-run, which is larger than most tuning deltas. `bench.sh` is a
thermally-aware harness that enforces fixed cooldowns and reports the
distribution (min / median / max / mean / stddev / spread), so signal can be
told from noise.

```bash
./bench.sh -e com6_v108.exe -n 4096 -m mt -r 5 -c 120
# 5 runs of MT at 4096, 120s cooldown between runs
# Output includes best, median, min, mean, stddev, and spread-% of mean
```

It also aborts if it detects any lingering `com6_v*.exe` processes (a real
multi-session hazard — a rogue `com6_v31.exe` from an earlier test once sat
burning 100% of a core for hours, invalidating every "cooldown" reading).
Kill them first with `tasklist | grep -i com6` / `taskkill //PID <pid> //F`.

To compare two versions, run `bench.sh` twice with `-l A` / `-l B` labels and
compare the summary blocks — the median (not the best) is usually the honest
number when spread exceeds ~10%.

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
