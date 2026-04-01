"""COM6 v43 vs OpenBLAS — Fair interleaved comparison
Tests both at each size with cooling to equalize thermal conditions."""
import numpy as np
import time
import subprocess
import os
import sys

def bench_openblas_single(n, threads):
    """Benchmark OpenBLAS at a single size"""
    script = f'''
import numpy as np, time, os
os.environ["OMP_NUM_THREADS"]="{threads}"
os.environ["OPENBLAS_NUM_THREADS"]="{threads}"
n={n}
A=np.random.rand(n,n); B=np.random.rand(n,n)
_=np.dot(A,B)
runs = 5 if n<=1024 else (3 if n<=2048 else 2)
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
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600)
    for line in r.stdout.strip().split('\n'):
        p = line.split(',')
        if len(p)==2: return float(p[1])
    return 0

def bench_com6_single(exe, n):
    """Benchmark COM6 at a single size"""
    r = subprocess.run([exe, str(n)], capture_output=True, text=True, timeout=600)
    for line in r.stdout.split('\n'):
        line = line.strip()
        if 'x' in line and ('OK' in line or 'FAIL' in line):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                try:
                    # Try different column positions for GF(MT)
                    for idx in range(len(parts)-2, 2, -1):
                        s = parts[idx].strip()
                        if s not in ('--', '-', '') and 'OK' not in s and 'FAIL' not in s and 'skip' not in s:
                            try:
                                return float(s)
                            except ValueError:
                                continue
                except: pass
    return 0

if __name__ == "__main__":
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    exe = "C:/Users/Scott/com6_v45.exe"

    print("=" * 80)
    print("  COM6 v45 vs OpenBLAS MT — Fair Interleaved Comparison")
    print("  (each size tested independently with cooling)")
    print("=" * 80)
    print()
    print(f"{'Size':<10} | {'BLAS MT':>8} | {'COM6 MT':>8} | {'Ratio':>8} | {'Winner':>10}")
    print(f"{'-'*10} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10}")

    for n in sizes:
        print(f"  Testing {n}...", end='', flush=True)
        time.sleep(3)  # Cool
        blas_gf = bench_openblas_single(n, 8)
        time.sleep(3)  # Cool
        com6_gf = bench_com6_single(exe, n)
        ratio = com6_gf / blas_gf if blas_gf > 0 else 0
        winner = "COM6 WIN" if ratio > 1 else "BLAS win"
        print(f"\r{n:4d}x{n:<4d}  | {blas_gf:6.1f} GF | {com6_gf:6.1f} GF | {ratio:6.2f}x  | {winner:>10}")

    print()
    print("ratio > 1.00 = COM6 is faster")
