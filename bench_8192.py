"""Single-size 8192 benchmark: v108 vs OpenBLAS with proper cooldowns."""
import numpy as np, time, subprocess, os, sys

EXE = "C:/Users/Scott/com6-matmul/com6_v108.exe"
COOLDOWN = 120

def bench_blas(n, t=8):
    script = f'''
import numpy as np, time, os
os.environ["OMP_NUM_THREADS"]="{t}"
os.environ["OPENBLAS_NUM_THREADS"]="{t}"
n={n}; np.random.seed(42)
A=np.random.rand(n,n); B=np.random.rand(n,n)
_=np.dot(A,B)
times=[]
for _ in range(2):
    t0=time.perf_counter(); C=np.dot(A,B); times.append(time.perf_counter()-t0)
best=min(times); gf=(2.0*n*n*n)/(best*1e9)
print(f"{{gf:.1f}}")
'''
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(t); env["OPENBLAS_NUM_THREADS"] = str(t)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=1800)
    for line in r.stdout.strip().split('\n'):
        try: return float(line)
        except: pass
    return 0

def bench_com6(n):
    r = subprocess.run([EXE, str(n), "mt"], capture_output=True, text=True, timeout=1800)
    for line in r.stdout.split('\n'):
        if 'GF' in line:
            for p in line.split('|'):
                if 'GF' in p:
                    try: return float(p.strip().split()[0])
                    except: pass
    return 0

if __name__ == "__main__":
    n = 8192
    print(f"8192 MT: cooldown {COOLDOWN}s between tests")
    print(f"  Warmup cooldown...")
    time.sleep(COOLDOWN)
    print(f"  OpenBLAS 8192...", end='', flush=True)
    blas = bench_blas(n)
    print(f" {blas:.1f} GF")
    print(f"  Cooldown {COOLDOWN}s...")
    time.sleep(COOLDOWN)
    print(f"  COM6 v108 8192...", end='', flush=True)
    com6 = bench_com6(n)
    print(f" {com6:.1f} GF")
    print()
    print(f"  Ratio: {com6/blas:.2f}x  ({'COM6 WIN' if com6>blas else 'BLAS win'})")
