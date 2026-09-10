"""
NanoVector Benchmark Suite
Compares NanoVector AVX2/FMA against NumPy baseline across sizes N=100..50000 (dim=384)
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import time
import numpy as np
import nanovector

def benchmark():
    dim = 384  # Standard sentence embedding dimension (all-MiniLM-L6-v2)
    sizes = [500, 2000, 10000, 50000]

    print("=" * 70)
    print(f" [*] NanoVector v{nanovector.__version__} Performance Benchmark")
    print(f" Hardware Backend: {nanovector.__backend__}")
    print(f" Vector Dimension: {dim}D (MiniLM / Small Embeddings)")
    print("=" * 70)

    for N in sizes:
        print(f"\n--- Benchmark N = {N:,} vectors ---")
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((N, dim)).astype(np.float32)
        query = rng.standard_normal(dim).astype(np.float32)
        ids = [f"id_{i}" for i in range(N)]

        # 1. Benchmark NanoVector Insertion
        idx = nanovector.Index(dim=dim, metric="cosine")
        t0 = time.perf_counter()
        idx.add_batch(ids, matrix)
        t_add_nanovec = (time.perf_counter() - t0) * 1000.0  # ms
        print(f"  [NanoVector] Add {N:,} vectors: {t_add_nanovec:.2f} ms")

        # 2. Benchmark NanoVector Search (100 iterations)
        warmup = idx.search(query, top_k=10)
        iters = 100 if N <= 10000 else 20
        t0 = time.perf_counter()
        for _ in range(iters):
            res_nanovec = idx.search(query, top_k=10)
        t_search_nanovec = ((time.perf_counter() - t0) / iters) * 1000.0  # ms per query
        print(f"  [NanoVector] Search Top-10:      {t_search_nanovec:.4f} ms ({1000.0 / t_search_nanovec:.1f} QPS)")

        # 3. Benchmark NumPy Baseline Search
        # Precompute matrix norms
        norms = np.linalg.norm(matrix, axis=1)
        q_norm = np.linalg.norm(query)
        t0 = time.perf_counter()
        for _ in range(iters):
            dots = np.dot(matrix, query)
            sims = dots / (norms * q_norm)
            top_k_indices = np.argpartition(sims, -10)[-10:]
            top_k_sorted = top_k_indices[np.argsort(-sims[top_k_indices])]
        t_search_numpy = ((time.perf_counter() - t0) / iters) * 1000.0
        print(f"  [NumPy Base] Search Top-10:      {t_search_numpy:.4f} ms ({1000.0 / t_search_numpy:.1f} QPS)")

        speedup = t_search_numpy / t_search_nanovec if t_search_nanovec > 0 else 1.0
        print(f"  >>> NanoVector Speedup vs NumPy: {speedup:.2f}x faster")

        # 4. Benchmark Save / Load
        test_file = "temp_bench.nvec"
        t0 = time.perf_counter()
        idx.save(test_file)
        t_save = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        loaded = nanovector.load(test_file)
        t_load = (time.perf_counter() - t0) * 1000.0
        print(f"  [Persistence] Save: {t_save:.2f} ms | Load: {t_load:.2f} ms (File size: {len(matrix) * dim * 4 / 1024 / 1024:.2f} MB)")

        import os
        if os.path.exists(test_file):
            os.remove(test_file)

    print("\n" + "=" * 70)
    print(" Benchmark completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    benchmark()
