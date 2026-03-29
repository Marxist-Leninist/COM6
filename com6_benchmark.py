"""
COM6 vs Strassen vs Standard Matrix Multiplication Benchmark
=============================================================
Three implementations, pure Python + numpy backing, timed across matrix sizes.

COM6: Block-based with transposition optimization (block size 32)
Strassen: Recursive divide-and-conquer (O(n^2.807))
Standard: Triple-loop / numpy matmul baseline
"""

import numpy as np
import time
import sys

# ============================================================
# Standard Matrix Multiplication (naive)
# ============================================================
def standard_matmul(A, B):
    """Standard O(n^3) triple-loop matmul using numpy dot"""
    return np.dot(A, B)


def standard_matmul_pure(A, B):
    """Pure Python triple loop - for small sizes only"""
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += A[i, p] * B[p, j]
            C[i, j] = s
    return C


# ============================================================
# Strassen's Algorithm
# ============================================================
def strassen(A, B, threshold=64):
    """
    Strassen's algorithm - O(n^2.807)
    Falls back to numpy dot below threshold for efficiency.
    """
    n = A.shape[0]
    if n <= threshold:
        return np.dot(A, B)

    # Pad to even size if needed
    if n % 2 != 0:
        A = np.pad(A, ((0, 1), (0, 1)))
        B = np.pad(B, ((0, 1), (0, 1)))
        C = _strassen_recurse(A, B, threshold)
        return C[:n, :n]

    return _strassen_recurse(A, B, threshold)


def _strassen_recurse(A, B, threshold):
    n = A.shape[0]
    if n <= threshold:
        return np.dot(A, B)

    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]

    # 7 multiplications instead of 8
    M1 = _strassen_recurse(A11 + A22, B11 + B22, threshold)
    M2 = _strassen_recurse(A21 + A22, B11, threshold)
    M3 = _strassen_recurse(A11, B12 - B22, threshold)
    M4 = _strassen_recurse(A22, B21 - B11, threshold)
    M5 = _strassen_recurse(A11 + A12, B22, threshold)
    M6 = _strassen_recurse(A21 - A11, B11 + B12, threshold)
    M7 = _strassen_recurse(A12 - A22, B21 + B22, threshold)

    C = np.zeros((n, n))
    C[:mid, :mid] = M1 + M4 - M5 + M7
    C[:mid, mid:] = M3 + M5
    C[mid:, :mid] = M2 + M4
    C[mid:, mid:] = M1 - M2 + M3 + M6
    return C


# ============================================================
# COM6 Matrix Multiplication
# ============================================================
def com6_matmul(A, B, block_size=32):
    """
    COM6 - Custom Operation Matrix multiplication.
    Block-based with B transposition for cache-friendly access.
    Each block operation is an element-wise COM operation on sub-matrices.

    Key insight: transpose B once, then all inner loops access
    sequential memory. Block tiling keeps data in L1/L2 cache.
    """
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    C = np.zeros((n, m))

    # COM operation: transpose B for row-major sequential access
    BT = B.T.copy()  # Contiguous transpose - the COM6 trick

    # Block-tiled multiplication with transposed B
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        for j0 in range(0, m, block_size):
            j1 = min(j0 + block_size, m)
            # Accumulate block C[i0:i1, j0:j1]
            for k0 in range(0, k, block_size):
                k1 = min(k0 + block_size, k)
                # COM element-wise block operation:
                # A_block @ BT_block.T = A_block @ B_block
                # But using BT means sequential memory access in inner loop
                A_block = A[i0:i1, k0:k1]
                BT_block = BT[j0:j1, k0:k1]  # This is B[k0:k1, j0:j1].T
                C[i0:i1, j0:j1] += A_block @ BT_block.T

    return C


def com6_matmul_pure(A, B, block_size=32):
    """
    COM6 pure Python implementation - no numpy dot in inner loop.
    Shows the actual COM6 algorithm without BLAS acceleration.
    """
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    C = np.zeros((n, m))

    # COM6 transpose trick
    BT = np.ascontiguousarray(B.T)

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        for j0 in range(0, m, block_size):
            j1 = min(j0 + block_size, m)
            for k0 in range(0, k, block_size):
                k1 = min(k0 + block_size, k)
                # Inner block: element-wise accumulation
                for i in range(i0, i1):
                    for j in range(j0, j1):
                        s = 0.0
                        # Sequential access on both A[i,:] and BT[j,:]
                        for p in range(k0, k1):
                            s += A[i, p] * BT[j, p]
                        C[i, j] += s
    return C


def com6_matmul_vectorized(A, B, block_size=32):
    """
    COM6 vectorized - uses numpy vectorization within blocks
    but applies the COM6 blocking + transpose strategy.
    This is the practical high-performance version.
    """
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    C = np.zeros((n, m))

    # COM6 contiguous transpose
    BT = np.ascontiguousarray(B.T)

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        for j0 in range(0, m, block_size):
            j1 = min(j0 + block_size, m)
            for k0 in range(0, k, block_size):
                k1 = min(k0 + block_size, k)
                # Vectorized block: A_block dot BT_block^T
                # Both blocks are contiguous in memory
                C[i0:i1, j0:j1] += np.dot(A[i0:i1, k0:k1], BT[j0:j1, k0:k1].T)

    return C


# ============================================================
# Benchmark runner
# ============================================================
def benchmark(func, A, B, warmup=1, runs=3, label=""):
    """Time a matmul function, return (avg_time, result)"""
    # Warmup
    for _ in range(warmup):
        result = func(A, B)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = func(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    best = min(times)
    return best, avg, result


def verify_correctness(results, names, tol=1e-6):
    """Check all results match within tolerance"""
    ref = results[0]
    for i in range(1, len(results)):
        diff = np.max(np.abs(ref - results[i]))
        status = "PASS" if diff < tol else f"FAIL (max diff: {diff:.2e})"
        print(f"  {names[0]} vs {names[i]}: {status}")


def format_time(t):
    if t < 0.001:
        return f"{t*1e6:.1f}us"
    elif t < 1.0:
        return f"{t*1e3:.2f}ms"
    else:
        return f"{t:.3f}s"


def run_benchmark_suite():
    print("=" * 72)
    print("  COM6 vs Strassen vs Standard Matrix Multiplication Benchmark")
    print("=" * 72)
    print()

    # --- Numpy-backed benchmarks (practical speed) ---
    sizes_numpy = [64, 128, 256, 512, 1024, 2048]

    print("ROUND 1: Numpy-backed implementations")
    print("  (COM6 vectorized blocks vs Strassen vs numpy.dot)")
    print("-" * 72)
    print(f"{'Size':>6} | {'Standard':>12} | {'Strassen':>12} | {'COM6':>12} | {'COM6 vs Std':>12} | {'COM6 vs Str':>12}")
    print("-" * 72)

    for n in sizes_numpy:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)

        runs = max(1, 5 if n <= 512 else 3 if n <= 1024 else 1)

        t_std_best, t_std_avg, r_std = benchmark(standard_matmul, A, B, runs=runs, label="standard")
        t_str_best, t_str_avg, r_str = benchmark(strassen, A, B, runs=runs, label="strassen")
        t_com_best, t_com_avg, r_com = benchmark(com6_matmul_vectorized, A, B, runs=runs, label="com6")

        speedup_vs_std = t_std_best / t_com_best if t_com_best > 0 else float('inf')
        speedup_vs_str = t_str_best / t_com_best if t_com_best > 0 else float('inf')

        print(f"{n:>5}x{n} | {format_time(t_std_best):>12} | {format_time(t_str_best):>12} | {format_time(t_com_best):>12} | {speedup_vs_std:>11.2f}x | {speedup_vs_str:>11.2f}x")

        # Verify correctness at small size
        if n <= 256:
            verify_correctness([r_std, r_str, r_com], ["Standard", "Strassen", "COM6"])

    # --- Pure Python benchmarks (algorithm comparison) ---
    sizes_pure = [32, 64, 128]

    print()
    print("ROUND 2: Pure Python implementations (no BLAS in inner loop)")
    print("  (COM6 pure blocks vs Standard triple-loop)")
    print("-" * 72)
    print(f"{'Size':>6} | {'Std Pure':>12} | {'COM6 Pure':>12} | {'Speedup':>12}")
    print("-" * 72)

    for n in sizes_pure:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)

        runs = 1  # Pure python is slow

        t_std_best, _, r_std = benchmark(standard_matmul_pure, A, B, warmup=0, runs=runs)
        t_com_best, _, r_com = benchmark(com6_matmul_pure, A, B, warmup=0, runs=runs)

        speedup = t_std_best / t_com_best if t_com_best > 0 else float('inf')
        print(f"{n:>5}x{n} | {format_time(t_std_best):>12} | {format_time(t_com_best):>12} | {speedup:>11.2f}x")

        verify_correctness([r_std, r_com], ["StdPure", "COM6Pure"], tol=1e-4)

    # --- COM6 block size sweep ---
    print()
    print("ROUND 3: COM6 Block Size Sweep (n=512)")
    print("-" * 72)
    n = 512
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)

    block_sizes = [8, 16, 32, 64, 128, 256]
    print(f"{'Block':>6} | {'Time':>12} | {'vs numpy.dot':>12}")
    print("-" * 40)

    t_ref, _, _ = benchmark(standard_matmul, A, B, runs=5)
    for bs in block_sizes:
        t_best, _, _ = benchmark(lambda a, b, bs=bs: com6_matmul_vectorized(a, b, block_size=bs), A, B, runs=3)
        ratio = t_ref / t_best if t_best > 0 else 0
        print(f"{bs:>6} | {format_time(t_best):>12} | {ratio:>11.2f}x")

    # --- COM6 with COM operations demo ---
    print()
    print("ROUND 4: COM6 with Custom Operations")
    print("-" * 72)
    print("Applying COM element-wise operations to block results...")
    n = 256
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)

    # Standard matmul
    t0 = time.perf_counter()
    C_std = np.dot(A, B)
    t_std = time.perf_counter() - t0

    # COM6 matmul with post-processing COM ops on each block
    t0 = time.perf_counter()
    C_com = com6_with_ops(A, B)
    t_com = time.perf_counter() - t0

    print(f"  Standard matmul:        {format_time(t_std)}")
    print(f"  COM6 + element-wise ops: {format_time(t_com)}")
    print(f"  COM6 output sample [0:3,0:3]:")
    print(f"    {C_com[0:3, 0:3]}")

    print()
    print("=" * 72)
    print("  Benchmark complete.")
    print("=" * 72)


def com6_with_ops(A, B, block_size=32):
    """COM6 matmul that also applies element-wise COM operations per block"""
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    C = np.zeros((n, m))
    BT = np.ascontiguousarray(B.T)

    # COM operation: apply activation per block result
    def com_block_op(block):
        # Leaky ReLU as COM element-wise operation
        return np.where(block > 0, block, 0.01 * block)

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        for j0 in range(0, m, block_size):
            j1 = min(j0 + block_size, m)
            block_acc = np.zeros((i1 - i0, j1 - j0))
            for k0 in range(0, k, block_size):
                k1 = min(k0 + block_size, k)
                block_acc += np.dot(A[i0:i1, k0:k1], BT[j0:j1, k0:k1].T)
            # Apply COM operation to completed block
            C[i0:i1, j0:j1] = com_block_op(block_acc)

    return C


if __name__ == '__main__':
    run_benchmark_suite()
