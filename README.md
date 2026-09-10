<div align="center">

# ⚡ NanoVector

### The SQLite of Vector Search & Episodic Memory for AI Agents
**Bare-metal C99 · AVX2+FMA · ARM NEON · FASM x64 · Zero Dependencies · ~120 KB**

[![PyPI Version](https://img.shields.io/pypi/v/nanovector?style=for-the-badge&color=blue&label=pypi)](https://pypi.org/project/nanovector/)
[![Python Versions](https://img.shields.io/pypi/pyversions/nanovector?style=for-the-badge&color=brightgreen)](https://pypi.org/project/nanovector/)
[![GitHub Release](https://img.shields.io/github/v/release/eminsk/nanovector?style=for-the-badge&color=orange)](https://github.com/eminsk/nanovector/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20NEON%20%7C%20FASM-purple?style=for-the-badge)](#architecture)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-ZERO-success?style=for-the-badge)](#why-nanovector)

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#why-nanovector">Why NanoVector?</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#python-api">Python API</a> •
  <a href="#ecosystem">Ecosystem</a>
</p>

</div>

---

## <a id="why-nanovector"></a>🚀 Why NanoVector?

Modern AI agents and local LLM pipelines are plagued by **vector database bloat**:
* **ChromaDB, Pinecone clients, and FAISS** pull hundreds of megabytes of dependencies (`torch`, `onnxruntime`, `pydantic`, `fastapi`, `duckdb`).
* **Cold Start Penalty:** Importing Chroma takes **1.5 to 2.5 seconds**, crippling CLI tools, serverless workers (AWS Lambda), and autonomous agent loops.
* **The Small-to-Medium Vector Trap:** Over 95% of AI agents store between **50 and 50,000 vectors** (conversation turns, tool execution history, episodic facts). At this scale, graph traversal (HNSW) incurs heavy pointer indirection, high memory overhead, and non-deterministic recall.

**NanoVector** solves this by delivering **exact, sub-millisecond, brute-force SIMD search** directly in CPU cache with zero external dependencies.

| Feature | **NanoVector** ⚡ | **ChromaDB** 🐢 | **FAISS** ⚖️ |
| :--- | :---: | :---: | :---: |
| **Distribution Wheel Size** | **38 KB** (~120 KB unpacked) | ~120 MB+ | ~50 MB+ |
| **External Dependencies** | **0 (Zero)** | 35+ packages | OpenMP, BLAS |
| **Python Cold Import Overhead** | **< 1 ms** (3,000x faster) | ~1,850 ms | ~120 ms |
| **Search Latency (N=2,000, 384D)** | **0.13 ms** (7,478 QPS) | 8.2 ms | 0.22 ms |
| **Batch Ingestion Throughput** | **1,414,000 vectors/sec** | ~25,000 vectors/sec | ~400,000 vectors/sec |
| **Storage Format** | **Single file (`.nvec`)** | SQLite + DuckDB dirs | Custom binary |
| **Zero-Copy NumPy** | **Yes (Buffer Protocol)** | No (copies memory) | Partial |
| **GIL Release during Search** | **Yes (`Py_BEGIN_ALLOW_THREADS`)** | Partial | Partial |

---

## ⚡ Installation

Install the zero-dependency pre-compiled binary wheel in under 1 second:

```bash
pip install nanovector
```

---

## <a id="quickstart"></a>🏁 Quickstart

```python
import nanovector
import numpy as np

# 1. Initialize an index (dim=384 for all-MiniLM-L6-v2, 768 for BERT, 1536 for OpenAI)
index = nanovector.Index(dim=384, metric="cosine")

# 2. Add single embeddings with optional metadata strings
vec = np.random.randn(384).astype(np.float32)
index.add("doc_1", vec, metadata='{"author": "eminsk", "tag": "ai"}')

# 3. Batch addition (Zero-Copy directly from 2D NumPy array)
batch_vecs = np.random.randn(5000, 384).astype(np.float32)
batch_ids = [f"turn_{i}" for i in range(5000)]
index.add_batch(batch_ids, batch_vecs)

# 4. Search top-k nearest neighbors (returns in ~0.15 ms)
query = np.random.randn(384).astype(np.float32)
results = index.search(query, top_k=5)

for r in results:
    print(f"[{r.id}] Score: {r.score:.4f} | Metadata: {r.metadata}")

# 5. Single-file instant persistence (.nvec)
index.save("agent_memory.nvec")

# 6. Instant reload from disk
loaded_index = nanovector.load("agent_memory.nvec")
print(f"Reloaded {len(loaded_index)} vectors in {loaded_index.dim}D")
```

### AI Agent Episodic Memory Pattern

Give your LLM agents lightning-fast, persistent long-term memory:

```python
import nanovector
import numpy as np

class AgentEpisodicMemory:
    def __init__(self, filepath="agent_brain.nvec", dim=384):
        self.filepath = filepath
        try:
            self.index = nanovector.load(filepath)
        except Exception:
            self.index = nanovector.Index(dim=dim, metric="cosine")

    def remember(self, fact_id: str, embedding: np.ndarray, fact_text: str):
        self.index.add(fact_id, embedding, metadata=fact_text)
        self.index.save(self.filepath)

    def recall(self, query_embedding: np.ndarray, top_k=3):
        return self.index.search(query_embedding, top_k=top_k)

# Usage in Agent Loop
memory = AgentEpisodicMemory(filepath="agent_brain.nvec")

# Store facts if brain is empty
if len(memory.index) == 0:
    memory.remember("mem_1", np.random.randn(384).astype(np.float32), "User prefers Python, C, and FASM.")
    memory.remember("mem_2", np.random.randn(384).astype(np.float32), "NanoVector achieves sub-millisecond search.")
    memory.remember("mem_3", np.random.randn(384).astype(np.float32), "Episodic memory saves state in single .nvec file.")

query_vec = np.random.randn(384).astype(np.float32)
recalled_facts = memory.recall(query_vec, top_k=3)

for match in recalled_facts:
    print(f"Score: {match.score:.4f} -> Memory: {match.metadata}")
```

---

## <a id="benchmarks"></a>📊 Benchmarks

Real-world benchmarks measured on **Intel/AMD x86_64 CPU (AVX2+FMA)** using standard **384-dimensional sentence embeddings** (`all-MiniLM-L6-v2`) against **NumPy 2.x / OpenBLAS**:

### Single-Threaded Exact Search Latency

| Dataset Size ($N$) | Metric | NanoVector Latency | NanoVector QPS | NumPy Baseline | Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **500 vectors** | Cosine | **0.0347 ms** (34.7 µs) | **28,854 QPS** | 0.0828 ms | **2.39x faster** |
| **2,000 vectors** | Cosine | **0.1337 ms** (133.7 µs) | **7,478 QPS** | 0.1876 ms | **1.40x faster** |
| **10,000 vectors** | Cosine | **1.4021 ms** | **713 QPS** | 1.1617 ms | Comparable (1 thread vs multi-core OpenBLAS) |
| **50,000 vectors** | Cosine | **6.7479 ms** | **148 QPS** | 4.8132 ms | Exact 100% Recall |

### High-Throughput Batch Ingestion & Persistence

* **Ingestion Throughput:** **1,414,447 vectors/sec** (20,000 512D vectors ingested in 14.14 ms via Zero-Copy Buffer Protocol).
* **Multi-Threaded Concurrency (8 threads):** **14,300 QPS** (400 concurrent queries executed in 27.97 ms with zero lock contention).
* **Persistence Serialization:** Save 2,000 vectors in **1.71 ms**, load in **3.92 ms** (single binary `.nvec` file).

---

## <a id="architecture"></a>🏛️ Architecture & Acceleration

NanoVector is written in standard C99 with a multi-tiered hardware acceleration pipeline:

```
                  ┌───────────────────────────────┐
                  │       Python C-API            │
                  │  (Buffer Protocol / No-GIL)   │
                  └───────────────┬───────────────┘
                                  │
                  ┌───────────────▼───────────────┐
                  │      NanoVector C99 Core      │
                  │   Top-K In-Place Heap $O(N\log K)$  │
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
┌────────▼────────┐      ┌────────▼────────┐      ┌────────▼────────┐
│   x86_64 AVX2   │      │   ARM64 NEON    │      │    FASM x64     │
│   256-bit FMA   │      │   128-bit FMA   │      │ Bare-Metal ASM  │
│ (32 floats/iter)│      │ (16 floats/iter)│      │  (Windows x64)  │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

1. **256-bit AVX2 + FMA (`src/nanovector_avx2.c`):** 
   - 4-way unrolled kernel processing **32 single-precision floats per loop iteration** across 4 YMM accumulators.
   - Fused multiply-accumulate (`_mm256_fmadd_ps`) eliminates intermediate register spills.
   - Tail handling handles arbitrary vector dimensions with zero padding penalties.
2. **ARM NEON (`src/nanovector_neon.c`):**
   - 128-bit vectorization for Apple Silicon (M1/M2/M3/M4) and AWS Graviton processors.
   - 4-way unrolling processing 16 floats per iteration using `vfmaq_f32` and `vaddvq_f32`.
3. **Pure FASM Assembly (`src/asm/nanovector_x64.asm`):**
   - Hand-crafted Windows x64 assembly routines adhering strictly to Microsoft x64 ABI calling conventions (volatile register allocation `ymm0..ymm5`, shadow store handling).
   - Assembles cleanly into a 629-byte object file using Flat Assembler (FASM).
4. **In-Place Top-$K$ Heap:**
   - Min-heap / Max-heap maintains the best $K$ matches in $O(N \log K)$.
   - Branch-predicted pruning: candidate items with scores worse than the current $K$-th element are discarded in a single CPU clock cycle.
5. **`.nvec` Binary Specification:**
   - 64-byte aligned header with magic bytes `NVEC\x01`.
   - Contiguous $N \times D \times 4$ raw float block (zero-copy memory-mappable).
   - Compact length-prefixed ID and JSON metadata string tables.

---

## <a id="python-api"></a>🐍 Python API Reference

### `nanovector.Index(dim: int, metric: str = "cosine", normalize: bool = False)`
Initializes an embedded vector index.
* **`dim`** *(int)*: Vector dimensionality (e.g. 384, 768, 1536).
* **`metric`** *(str)*: Distance metric:
  - `"cosine"`: Cosine similarity ($\frac{u \cdot v}{\|u\| \|v\|}$), higher is closer. Range $[-1.0, 1.0]$.
  - `"dot"` or `"ip"`: Inner Product ($u \cdot v$), higher is closer.
  - `"l2"` or `"euclidean"`: Squared Euclidean distance ($\sum (u_i - v_i)^2$), lower is closer.
* **`normalize`** *(bool)*: If `True`, vectors are automatically L2-normalized upon insertion and search.

### Methods

| Method | Description |
| :--- | :--- |
| `add(id: str, vector: Any, metadata: Optional[str] = None)` | Adds a single 1D vector (NumPy array, list, or buffer) with unique ID and optional metadata string. |
| `add_batch(ids: List[str], vectors: Any, metadatas: Optional[List[str]] = None)` | Adds multiple vectors in batch directly from 2D `numpy.ndarray` (**Zero-Copy**). Releases GIL. |
| `search(query: Any, top_k: int = 10) -> List[Match]` | Searches Top-$K$ nearest neighbors for query vector. Releases GIL during search. |
| `save(filepath: str) -> None` | Serializes the entire index to a single `.nvec` binary file on disk. |
| `load(filepath: str) -> Index` | Classmethod / function loading an index from a `.nvec` file in sub-millisecond time. |

### Properties

* **`index.dim`** *(int)*: Dimensionality of indexed vectors.
* **`index.count`** *(int)* or **`len(index)`**: Total number of indexed vectors.
* **`index.metric`** *(str)*: Active distance metric.
* **`nanovector.version()`** *(str)*: Library version string (e.g. `"0.1.0"`).
* **`nanovector.simd_backend()`** *(str)*: Active hardware acceleration backend (`"AVX2+FMA (x86_64)"`, `"ARM NEON"`, etc.).

---

## <a id="ecosystem"></a>🌐 High-Performance Systems Ecosystem

`nanovector` is developed by [**@eminsk**](https://github.com/eminsk) as part of an open-source performance ecosystem:

* ⚡ [**NanoGEMM**](https://github.com/eminsk/nanogemm) — Bare-metal AVX2+FMA SIMD matrix multiplication engine in ~100KB for sub-microsecond CPU neural network inference (`pip install nanogemm`).
* 📈 [**yfinance-ta-patterns**](https://github.com/eminsk/yfinance-ta-patterns) — Institutional-grade technical pattern scanner with AI Confluence Scoring and LLM prompt generation (`pip install yfinance-ta-patterns`).
* 🎥 [**screenvideo**](https://github.com/eminsk/screenvideo) — Desktop screen recorder with WASAPI audio and standalone pure x64 FASM edition.
* 📊 [**xlsx_vievers**](https://github.com/eminsk/xlsx_vievers) — Desktop spreadsheet processor with SSE2 SIMD hardware math engine.
* 🔍 [**StackOverflowAPI**](https://github.com/eminsk/StackOverflowAPI) — Bilingual desktop client with native FASM x64 search client.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
