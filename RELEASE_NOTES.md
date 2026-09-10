## ⚡ NanoVector v0.1.0 — Initial Official Release

The **SQLite of Vector Search & Episodic Memory for AI Agents** in ~120KB.

### Key Highlights
- **Zero External Dependencies:** Self-contained C99 extension, no PyTorch, no SciPy, no heavy C++ runtimes.
- **AVX2 + FMA SIMD Kernel:** Unrolled 4x across 32 floats per iteration on x86_64.
- **Pure FASM x64 Assembly Kernel:** Standalone hand-crafted assembly kernel adhering strictly to Microsoft x64 ABI.
- **ARM NEON Kernel:** 128-bit FMA vectorization for Apple Silicon (M1/M2/M3/M4) and AWS Graviton.
- **In-Place Top-K Heap:** O(N log K) min-heap / max-heap with branch-predicted pruning.
- **Single-File Binary Persistence (`.nvec`):** Instant serialization and deserialization with 64-byte aligned header.
- **Python Buffer Protocol:** Direct Zero-Copy ingestion and search from 1D/2D `numpy.ndarray`.
- **Concurrency:** GIL released during searches (`Py_BEGIN_ALLOW_THREADS`) achieving 14,300+ QPS across 8 threads.

### Installation
```bash
pip install nanovector
```
