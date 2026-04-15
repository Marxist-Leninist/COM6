#!/usr/bin/env python3
"""Fair head-to-head: COM6 v100 (pthreads pool) vs OpenBLAS (numpy)
Interleaved runs with cooldown to equalize thermal state."""
import subprocess, time, numpy as np

SIZES = [512, 1024, 2048, 4096, 8192]
COOLDOWN = 5
RUNS = 3

def bench_blas(n, runs=3):
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    _ = A @ B  # warmup
    best = 1e30
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = A @ B
        t = time.perf_counter() - t0
        if t < best: best = t
    return 2.0*n*n*n / (best * 1e9)

def bench_com6(n, runs=3):
    r = subprocess.run(['./com6_v100', str(n), 'mt'], capture_output=True, text=True, timeout=120)
    for line in r.stdout.strip().split('\n'):
        if 'GF' in line:
            gf = float(line.split('(')[1].split(' GF')[0])
            return gf
    return 0.0

print(f"{'Size':>8} | {'OpenBLAS MT':>12} | {'COM6 v100 MT':>13} | {'Ratio':>6} | Winner")
print("-" * 65)

for n in SIZES:
    time.sleep(COOLDOWN)
    gf_blas = bench_blas(n, RUNS)
    time.sleep(COOLDOWN)
    gf_com6 = bench_com6(n, RUNS)
    ratio = gf_com6 / gf_blas if gf_blas > 0 else 0
    winner = "COM6" if ratio > 1.0 else "BLAS"
    print(f"{n:>8} | {gf_blas:>9.1f} GF | {gf_com6:>10.1f} GF | {ratio:>5.2f}x | {winner}")
