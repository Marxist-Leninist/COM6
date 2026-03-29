"""COM6 v14 vs OpenBLAS head-to-head"""
import numpy as np
import time
import subprocess
import os
import sys

def bench_numpy_st(sizes):
    """Single-threaded OpenBLAS via subprocess"""
    script = '''
import numpy as np, time, os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
for n in [''' + ','.join(str(s) for s in sizes) + ''']:
    A=np.random.rand(n,n); B=np.random.rand(n,n)
    _=np.dot(A,B)
    times=[]
    for _ in range(3 if n<=1024 else 1):
        t0=time.perf_counter(); C=np.dot(A,B); times.append(time.perf_counter()-t0)
    best=min(times)
    print(f"{n},{best*1000:.1f},{(2.0*n*n*n)/(best*1e9):.1f}")
'''
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600)
    results = {}
    for line in r.stdout.strip().split('\n'):
        p = line.split(',')
        if len(p)==3: results[int(p[0])] = {'ms': float(p[1]), 'gf': float(p[2])}
    return results

if __name__ == "__main__":
    sizes = [256, 512, 1024, 2048, 4096]

    print("Running OpenBLAS single-threaded...")
    blas = bench_numpy_st(sizes)

    print("Running COM6 v14...")
    r = subprocess.run(["./com6_v14.exe"], capture_output=True, text=True, timeout=600)
    com6 = {}
    for line in r.stdout.split('\n'):
        line = line.strip()
        if 'x' in line and 'ms' in line and ('OK' in line or 'skip' in line):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                try:
                    n = int(parts[0].split('x')[0].strip())
                    ms = float(parts[1].replace('ms','').strip())
                    gf = float(parts[3].strip())
                    com6[n] = {'ms': ms, 'gf': gf}
                except: pass

    print()
    print("=" * 72)
    print("  COM6 v14 vs OpenBLAS (single-threaded) - HEAD TO HEAD")
    print("=" * 72)
    print(f"  {'Size':<10} {'BLAS ms':>10} {'BLAS GF':>10} {'COM6 ms':>10} {'COM6 GF':>10} {'Ratio':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for n in sizes:
        b = blas.get(n, {})
        c = com6.get(n, {})
        bms = b.get('ms', 0)
        bgf = b.get('gf', 0)
        cms = c.get('ms', 0)
        cgf = c.get('gf', 0)
        if bms > 0 and cms > 0:
            ratio = bms / cms
            winner = "COM6 WIN" if ratio > 1 else "BLAS win"
            print(f"  {n:4d}x{n:<4d}  {bms:8.1f}ms  {bgf:6.1f} GF  {cms:8.1f}ms  {cgf:6.1f} GF  {ratio:.2f}x {winner}")

    print()
    print("  ratio > 1.0 = COM6 is faster")
