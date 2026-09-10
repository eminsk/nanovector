"""
NanoVector Zero-Copy NumPy Batch Example
Demonstrates high-throughput Zero-Copy ingestion and querying directly from NumPy 2D arrays.
"""

import time
import nanovector
import numpy as np

def main():
    dim = 512
    n_vectors = 20000

    print("=========================================================")
    print(f" NanoVector Zero-Copy NumPy Batch Demo ({dim}D x {n_vectors:,} items)")
    print("=========================================================")

    # 1. Allocate large 2D matrix
    rng = np.random.default_rng(2026)
    matrix = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    ids = [f"vec_{i:05d}" for i in range(n_vectors)]

    # 2. Ingest via Buffer Protocol
    index = nanovector.Index(dim=dim, metric="cosine")

    t0 = time.perf_counter()
    index.add_batch(ids, matrix)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[*] Ingested {len(index):,} vectors in {elapsed_ms:.2f} ms ({n_vectors / (elapsed_ms / 1000.0):,.0f} vectors/sec)")

    # 3. Query
    query = matrix[777]  # Exact match for vec_00777
    t0 = time.perf_counter()
    matches = index.search(query, top_k=5)
    query_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[*] Searched across {len(index):,} items in {query_ms:.4f} ms!")
    print("\nTop Matches:")
    for m in matches:
        print(f"  ID: {m.id} | Score: {m.score:.6f}")

if __name__ == "__main__":
    main()
