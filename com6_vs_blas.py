"""
COM6 vs OpenBLAS (numpy.dot) benchmark
Fair comparison: times numpy.dot (backed by OpenBLAS dgemm) against
COM6 v12 compiled C code via subprocess.

Also tests single-threaded OpenBLAS for fair comparison since COM6 is single-threaded.
"""
import numpy as np
import time
import subprocess
import os
import struct

os.environ["OMP_NUM_THREADS"] = "1"  # Will only affect if set before numpy import
# For single-thread test we re-run ourselves with env var

def bench_numpy(n, runs=3):
    """Benchmark numpy.dot (OpenBLAS dgemm) for n x n doubles"""
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # Warmup
    _ = np.dot(A, B)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        C = np.dot(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times)

def get_thread_info():
    """Get OpenBLAS thread count"""
    try:
        import numpy.linalg as la
        # Try to detect thread count
        threads = os.environ.get("OPENBLAS_NUM_THREADS",
                   os.environ.get("OMP_NUM_THREADS", "?"))
        return threads
    except:
        return "?"

def parse_com6_output(output):
    """Parse COM6 v12 benchmark output to get times"""
    results = {}
    for line in output.split('\n'):
        line = line.strip()
        if 'x' in line and 'ms' in line and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                try:
                    size = int(parts[0].split('x')[0].strip())
                    strassen_ms = float(parts[1].replace('ms', '').strip())
                    com6_ms = float(parts[3].replace('ms', '').strip())
                    results[size] = {'strassen': strassen_ms, 'com6': com6_ms}
                except:
                    pass
    return results

if __name__ == "__main__":
    import sys

    # Check if numpy is using multiple threads
    openblas_threads = os.environ.get("OPENBLAS_NUM_THREADS", "auto")
    omp_threads = os.environ.get("OMP_NUM_THREADS", "auto")

    print("=" * 78)
    print("  COM6 v12 vs OpenBLAS (numpy.dot) - The Real Test")
    print("=" * 78)
    print()

    # Detect CPU
    try:
        import platform
        print(f"  CPU: {platform.processor()}")
    except:
        pass
    print(f"  numpy version: {np.__version__}")
    np.show_config()
    print()

    sizes = [256, 512, 1024, 2048, 4096]

    # ---- MULTI-THREADED BLAS ----
    print("-" * 78)
    print(f"  OpenBLAS (MULTI-THREADED, OPENBLAS_NUM_THREADS={openblas_threads})")
    print("-" * 78)

    blas_mt = {}
    for n in sizes:
        t = bench_numpy(n)
        blas_mt[n] = t * 1000  # to ms
        gflops = (2.0 * n * n * n) / (t * 1e9)
        print(f"  {n:4d}x{n:<4d}  {t*1000:10.1f} ms  ({gflops:.1f} GFLOPS)")

    print()

    # ---- SINGLE-THREADED BLAS ----
    print("-" * 78)
    print("  OpenBLAS (SINGLE-THREADED, OMP_NUM_THREADS=1)")
    print("-" * 78)

    # Run a subprocess with OMP_NUM_THREADS=1 for fair single-thread comparison
    st_script = '''
import numpy as np
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sizes = [256, 512, 1024, 2048, 4096]
for n in sizes:
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    _ = np.dot(A, B)  # warmup
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        C = np.dot(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    best = min(times)
    gflops = (2.0 * n * n * n) / (best * 1e9)
    print(f"{n},{best*1000:.1f},{gflops:.1f}")
'''

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", st_script],
        capture_output=True, text=True, env=env, timeout=600
    )

    blas_st = {}
    for line in result.stdout.strip().split('\n'):
        parts = line.split(',')
        if len(parts) == 3:
            n = int(parts[0])
            ms = float(parts[1])
            gf = float(parts[2])
            blas_st[n] = ms
            print(f"  {n:4d}x{n:<4d}  {ms:10.1f} ms  ({gf:.1f} GFLOPS)")

    print()

    # ---- COM6 v12 ----
    print("-" * 78)
    print("  COM6 v12 (single-threaded, AVX2 FMA)")
    print("-" * 78)

    # Run com6_v12.exe and parse output
    com6_exe = os.path.join(os.path.dirname(__file__), "com6_v12.exe")
    if not os.path.exists(com6_exe):
        com6_exe = "com6_v12.exe"

    result = subprocess.run([com6_exe], capture_output=True, text=True, timeout=600)
    com6_data = {}

    for line in result.stdout.split('\n'):
        line = line.strip()
        # Parse lines like: " 256x256  |      3.1 ms |          - |      1.5 ms |    2.12x | OK"
        if 'x' in line and 'ms' in line and 'OK' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                try:
                    size_str = parts[0].split('x')[0].strip()
                    size = int(size_str)
                    com6_ms = float(parts[3].replace('ms', '').strip())
                    strassen_ms = float(parts[1].replace('ms', '').strip())
                    com6_data[size] = com6_ms
                    gflops = (2.0 * size * size * size) / (com6_ms / 1000 * 1e9)
                    print(f"  {size:4d}x{size:<4d}  {com6_ms:10.1f} ms  ({gflops:.1f} GFLOPS)")
                except:
                    pass

    print()

    # ---- COMPARISON TABLE ----
    print("=" * 78)
    print("  FINAL COMPARISON")
    print("=" * 78)
    print()
    print(f"  {'Size':<10} {'BLAS(MT)':<12} {'BLAS(1T)':<12} {'COM6v12':<12} {'COM6/BLAS(MT)':<14} {'COM6/BLAS(1T)':<14}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")

    for n in sizes:
        bmt = blas_mt.get(n, 0)
        bst = blas_st.get(n, 0)
        c6 = com6_data.get(n, 0)

        if bmt > 0 and c6 > 0:
            ratio_mt = bmt / c6  # >1 means COM6 wins
            ratio_st = bst / c6 if bst > 0 else 0

            mt_str = f"{'COM6 WINS' if ratio_mt > 1 else 'BLAS wins'} {ratio_mt:.2f}x"
            st_str = f"{'COM6 WINS' if ratio_st > 1 else 'BLAS wins'} {ratio_st:.2f}x" if ratio_st > 0 else "N/A"

            print(f"  {n:4d}x{n:<4d}  {bmt:8.1f} ms  {bst:8.1f} ms  {c6:8.1f} ms  {mt_str:<14} {st_str:<14}")

    print()
    print("  BLAS(MT) = OpenBLAS multi-threaded (unfair - uses all CPU cores)")
    print("  BLAS(1T) = OpenBLAS single-threaded (fair 1-to-1 comparison)")
    print("  COM6v12  = COM6 single-threaded AVX2 FMA")
    print("  ratio > 1 = COM6 is faster")
