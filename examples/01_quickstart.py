"""
NanoVector Quickstart Example
Shows basic creation, vector insertion, top-k search, and file persistence.
"""

import nanovector
import numpy as np

def main():
    print(f"NanoVector v{nanovector.__version__} ({nanovector.__backend__})")

    # 1. Initialize an index with 384 dimensions (standard for small embedding models)
    dim = 384
    index = nanovector.Index(dim=dim, metric="cosine")

    # 2. Add some synthetic embeddings
    rng = np.random.default_rng(42)
    doc_vectors = rng.standard_normal((10, dim)).astype(np.float32)

    for i in range(10):
        index.add(
            id=f"doc_{i}",
            vector=doc_vectors[i],
            metadata=f'{{"topic": "science", "index": {i}}}'
        )

    print(f"Indexed {len(index)} documents.")

    # 3. Query the index with the 3rd document
    query = doc_vectors[3]
    matches = index.search(query, top_k=3)

    print("\nTop-3 Nearest Matches:")
    for match in matches:
        print(f"  ID: {match.id:<10} Score: {match.score:.4f}  Metadata: {match.metadata}")

    # 4. Save index to disk
    filepath = "quickstart.nvec"
    index.save(filepath)
    print(f"\nSaved index to '{filepath}'.")

    # 5. Reload from disk
    loaded = nanovector.load(filepath)
    print(f"Reloaded {len(loaded)} vectors from '{filepath}'.")

    import os
    if os.path.exists(filepath):
        os.remove(filepath)

if __name__ == "__main__":
    main()
