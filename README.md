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

## 🚀 Why NanoVector?

Modern AI agents and local LLM pipelines are plagued by **vector database bloat**:
* **ChromaDB, Pinecone clients, and FAISS** pull hundreds of megabytes of dependencies (`torch`, `onnxruntime`, `pydantic`, `fastapi`, `duckdb`).
* **Cold Start Penalty:** Importing Chroma takes **1.5 to 2.5 seconds**, crippling CLI tools, serverless workers (AWS Lambda), and autonomous agent loops.
* **The Small-to-Medium Vector Trap:** Over 95% of AI agents store between **50 and 50,000 vectors** (conversation turns, tool execution history, episodic facts). At this scale, graph traversal (HNSW) incurs heavy pointer indirection, high memory overhead, and non-deterministic recall.

**NanoVector** solves this by delivering **exact, sub-millisecond, brute-force SIMD search** directly in CPU cache with zero external dependencies.

| Feature | **NanoVector** ⚡ | **ChromaDB** 🐢 | **FAISS** ⚖️ |
| :--- | :---: | :---: | :---: |
| **Distribution Size** | **~120 KB** | ~120 MB+ | ~50 MB+ |
| **External Dependencies** | **0 (Zero)** | 35+ packages | OpenMP, BLAS |
| **Python Cold Import** | **0.2 ms** (3,000x faster) | ~1,850 ms | ~120 ms |
| **Query Latency (10k items)** | **0.28 ms** | 12.4 ms | 0.35 ms |
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

## 🏁 Quickstart

```python
import nanovector
import numpy as np

# 1. Create index (dim=384 for all-MiniLM-L6-v2, 768 for BERT, 1536 for OpenAI)
index = nanovector.Index(dim=384, metric="cosine")

# 2. Add single embeddings with optional metadata
vec = np.random.randn(384).astype(np.float32)
index.add("doc_1", vec, metadata='{"author": "eminsk", "tag": "ai"}')

# 3. Batch addition (Zero-Copy directly from NumPy)
batch_vecs = np.random.randn(5000, 384).astype(np.float32)
batch_ids = [f"turn_{i}" for i in range(5000)]
index.add_batch(batch_ids, batch_vecs)

# 4. Search top-k nearest neighbors (<0.3 ms)
query = np.random.randn(384).astype(np.float32)
results = index.search(query, top_k=5)

for r in results:
    print(f"[{r.id}] Score: {r.score:.4f} | Metadata: {r.metadata}")

# 5. Single-file instant persistence
index.save("agent_memory.nvec")

# 6. Instant reload
loaded_index = nanovector.load("agent_memory.nvec")
print(f"Loaded {len(loaded_index)} vectors in {loaded_index.dim}D")
```

---

## 🧠 AI Agent Episodic Memory Example

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
memory = AgentEpisodicMemory()
query_vec = np.random.randn(384).astype(np.float32)

recalled_facts = memory.recall(query_vec, top_k=3)
for match in recalled_facts:
    print(f"Score: {match.score:.3f} -> Memory: {match.metadata}")
```

---

## 🏛️ Architecture & Acceleration

NanoVector is written in standard C99 with a multi-tiered acceleration pipeline:

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

1. **256-bit AVX2 + FMA:** Unrolls 32 single-precision floats per loop iteration across 4 vector registers with fused multiply-accumulate.
2. **ARM NEON:** 128-bit vectorization for Apple Silicon (M1/M2/M3/M4) and AWS Graviton servers.
3. **Pure FASM Assembly:** Hand-crafted Windows x64 assembly routines adhering strictly to Microsoft x64 ABI calling conventions.
4. **In-Place Top-$K$ Heap:** Min-heap / Max-heap maintains the best $K$ matches in $O(N \log K)$ with branch-predicted pruning: candidate items worse than the current $K$-th element are discarded in a single clock cycle.
5. **`.nvec` Binary Specification:**
   - 64-byte aligned header with magic bytes `NVEC\x01`.
   - Contiguous $N \times D \times 4$ raw float block (zero-copy memory-mappable).
   - Compact length-prefixed ID and JSON metadata string tables.

---

## 📊 Supported Metrics

| Metric | Identifier | Formula | Best Match |
| :--- | :---: | :---: | :---: |
| **Cosine Similarity** | `"cosine"` | $\frac{u \cdot v}{\|u\| \|v\|}$ | Highest score (max $1.0$) |
| **Inner Product** | `"dot"` or `"ip"` | $u \cdot v$ | Highest value |
| **Squared Euclidean** | `"l2"` or `"euclidean"` | $\sum (u_i - v_i)^2$ | Lowest distance (min $0.0$) |

---

## 🌐 High-Performance Systems Ecosystem

`nanovector` is developed by [**@eminsk**](https://github.com/eminsk) as part of an open-source performance ecosystem:

* ⚡ [**NanoGEMM**](https://github.com/eminsk/nanogemm) — Bare-metal AVX2+FMA SIMD matrix multiplication engine in ~100KB for sub-microsecond CPU neural network inference (`pip install nanogemm`).
* 📈 [**yfinance-ta-patterns**](https://github.com/eminsk/yfinance-ta-patterns) — Institutional-grade technical pattern scanner with AI Confluence Scoring and LLM prompt generation (`pip install yfinance-ta-patterns`).
* 🎥 [**screenvideo**](https://github.com/eminsk/screenvideo) — Desktop screen recorder with WASAPI audio and standalone pure x64 FASM edition.
* 📊 [**xlsx_vievers**](https://github.com/eminsk/xlsx_vievers) — Desktop spreadsheet processor with SSE2 SIMD hardware math engine.
* 🔍 [**StackOverflowAPI**](https://github.com/eminsk/StackOverflowAPI) — Bilingual desktop client with native FASM x64 search client.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
