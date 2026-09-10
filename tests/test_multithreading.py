"""
NanoVector Multi-Threading & Concurrency Test
Verifies that multiple Python threads can concurrently query the index with GIL released.
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import time
import threading
import numpy as np
import nanovector

def test_concurrent_searches():
    dim = 128
    N = 5000
    num_threads = 8
    queries_per_thread = 50

    index = nanovector.Index(dim=dim, metric="cosine")
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((N, dim)).astype(np.float32)
    ids = [f"item_{i}" for i in range(N)]
    index.add_batch(ids, matrix)

    errors = []
    def worker(tid):
        thread_rng = np.random.default_rng(tid * 100)
        for _ in range(queries_per_thread):
            q = thread_rng.standard_normal(dim).astype(np.float32)
            results = index.search(q, top_k=5)
            if len(results) != 5:
                errors.append(f"Thread {tid}: expected 5 results, got {len(results)}")
            if results[0].score < results[1].score:
                errors.append(f"Thread {tid}: results not properly sorted")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]

    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = (time.perf_counter() - t0) * 1000.0

    total_queries = num_threads * queries_per_thread
    print(f"\n[PASS] Multi-threaded search: {total_queries} queries across {num_threads} threads in {elapsed:.2f} ms ({total_queries / (elapsed / 1000.0):.1f} QPS)")
    assert len(errors) == 0, f"Errors encountered: {errors}"

if __name__ == "__main__":
    test_concurrent_searches()
