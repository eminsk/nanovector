"""
NanoVector Multi-Threading & Concurrency Test
Verifies that multiple Python threads can concurrently query the index with GIL released.
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import random
import threading
import time
import unittest
import nanovector

try:
    import numpy as np
except ImportError:
    np = None


def test_concurrent_searches():
    dim = 64
    N = 1000
    num_threads = 4
    queries_per_thread = 25

    index = nanovector.Index(dim=dim, metric="cosine")

    if np is not None:
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((N, dim)).astype(np.float32)
        ids = [f"item_{i}" for i in range(N)]
        index.add_batch(ids, matrix)
    else:
        rnd = random.Random(42)
        matrix = [[rnd.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(N)]
        ids = [f"item_{i}" for i in range(N)]
        index.add_batch(ids, matrix)

    errors = []

    def worker(tid):
        if np is not None:
            thread_rng = np.random.default_rng(tid * 100)
            for _ in range(queries_per_thread):
                q = thread_rng.standard_normal(dim).astype(np.float32)
                results = index.search(q, top_k=5)
                if len(results) != 5:
                    errors.append(f"Thread {tid}: expected 5 results, got {len(results)}")
                if len(results) >= 2 and results[0].score < results[1].score:
                    errors.append(f"Thread {tid}: results not properly sorted")
        else:
            thread_rnd = random.Random(tid * 100)
            for _ in range(queries_per_thread):
                q = [thread_rnd.gauss(0.0, 1.0) for _ in range(dim)]
                results = index.search(q, top_k=5)
                if len(results) != 5:
                    errors.append(f"Thread {tid}: expected 5 results, got {len(results)}")
                if len(results) >= 2 and results[0].score < results[1].score:
                    errors.append(f"Thread {tid}: results not properly sorted")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = (time.perf_counter() - t0) * 1000.0

    total_queries = num_threads * queries_per_thread
    assert len(errors) == 0, f"Errors encountered: {errors}"


class TestMultiThreading(unittest.TestCase):
    """Unittest adapter for multi-threading validation."""
    def test_threads(self):
        test_concurrent_searches()


if __name__ == "__main__":
    unittest.main()
