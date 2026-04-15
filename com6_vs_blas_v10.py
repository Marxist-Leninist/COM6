"""COM6 v108 vs OpenBLAS — Fair interleaved comparison (2026-04-15)

Thermal management: size-adaptive cooldowns. 4096+ gets 60s, 512-2048 get 20s.
Runs each size: OpenBLAS first (cold), then COM6 v108 after a cooldown,
then reports the ratio. Best-of-N internal to each binary.
"""
import numpy as np
import time
import subprocess
import os
import sys

EXE = "C:/Users/Scott/com6-matmul/com6_v108.exe"

def cooldown(n):
    if n >= 4096: return 60
    if n >= 2048: return 30
    return 20

def bench_openblas_single(n, threads):
    script = f'''
import numpy as np, time, os
os.environ["OMP_NUM_THREADS"]="{threads}"
os.environ["OPENBLAS_NUM_THREADS"]="{threads}"
n={n}
np.random.seed(42)
A=np.random.rand(n,n); B=np.random.rand(n,n)
_=np.dot(A,B)  # warmup
runs = 7 if n<=512 else (5 if n<=1024 else (3 if n<=2048 else 2))
times=[]
for _ in range(runs):
    t0=time.perf_counter(); C=np.dot(A,B); times.append(time.perf_counter()-t0)
best=min(times)
gf=(2.0*n*n*n)/(best*1e9)
print(f"{{best*1000:.1f}},{{gf:.1f}}")
'''
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=900)
    for line in r.stdout.strip().split('\n'):
        p = line.split(',')
        if len(p)==2:
            try: return float(p[1])
            except: pass
    return 0

def bench_com6_single(exe, n, mode="mt"):
    r = subprocess.run([exe, str(n), mode], capture_output=True, text=True, timeout=900)
    for line in r.stdout.split('\n'):
        line = line.strip()
        if 'GF' in line:
            parts = line.split('|')
            for p in parts:
                if 'GF' in p:
                    try: return float(p.strip().split()[0])
                    except: pass
    return 0

if __name__ == "__main__":
    sizes = [512, 1024, 2048, 4096]

    print("=" * 84)
    print("  COM6 v108 vs OpenBLAS (numpy) -- Fresh 2026-04-15")
    print("  Cooldowns: 20s (<=1024) / 30s (2048) / 60s (4096)")
    print("  Test order: OpenBLAS first (colder), then COM6 after cooldown")
    print("=" * 84)
    print()
    print(f"{'Size':<12} | {'BLAS MT':>10} | {'COM6 MT':>10} | {'Ratio':>8} | {'Winner':>10}")
    print(f"{'-'*12} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*10}")

    wins_com6, wins_blas = 0, 0
    for n in sizes:
        cd = cooldown(n)
        sys.stdout.write(f"  Testing {n} (cooldown {cd}s)...")
        sys.stdout.flush()
        time.sleep(cd)
        blas_gf = bench_openblas_single(n, 8)
        time.sleep(cd)
        com6_gf = bench_com6_single(EXE, n)
        ratio = com6_gf / blas_gf if blas_gf > 0 else 0
        if ratio > 1: winner, wins_com6 = "COM6 WIN", wins_com6+1
        else:         winner, wins_blas = "BLAS win", wins_blas+1
        sys.stdout.write(f"\r{n:4d}x{n:<5d}   | {blas_gf:7.1f} GF | {com6_gf:7.1f} GF | {ratio:6.2f}x  | {winner:>10}\n")
        sys.stdout.flush()

    print()
    print(f"Final tally: COM6 {wins_com6}/{wins_com6+wins_blas} | OpenBLAS {wins_blas}/{wins_com6+wins_blas}")
    print("ratio > 1.00 = COM6 is faster than the BLAS backing numpy.dot")
