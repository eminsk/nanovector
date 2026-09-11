"""
NanoVector LangChain Integration Test Suite
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import os
import tempfile
import pytest
import numpy as np
from nanovector import NanoVectorStore
from nanovector.integrations.langchain import Document


VOCAB = [
    "bare", "metal", "c", "simd", "vector", "engine",
    "python", "ai", "agents", "episodic", "memory",
    "cooking", "pasta", "garlic", "olive", "oil",
    "fast", "slow", "search", "persistent", "ephemeral", "reboots"
]


class DummyEmbeddings:
    """Mock embeddings model for fast, deterministic unit testing."""
    def embed_documents(self, texts):
        results = []
        for t in texts:
            v = np.zeros(len(VOCAB), dtype=np.float32)
            words = t.lower().split()
            for w in words:
                clean_w = "".join(ch for ch in w if ch.isalnum())
                if clean_w in VOCAB:
                    v[VOCAB.index(clean_w)] += 1.0
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            else:
                v[0] = 1.0
            results.append(v.tolist())
        return results

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_top_level_export():
    from nanovector import NanoVectorStore as NVS
    assert NVS is NanoVectorStore


def test_from_texts_and_similarity_search():
    embeddings = DummyEmbeddings()
    texts = [
        "Bare metal C SIMD vector engine",
        "Python AI agents with episodic memory",
        "Cooking pasta with garlic and olive oil",
    ]
    metadatas = [
        {"cat": "tech", "lang": "c"},
        {"cat": "tech", "lang": "python"},
        {"cat": "food", "lang": "it"},
    ]
    ids = ["doc_c", "doc_py", "doc_pasta"]

    store = NanoVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        metric="cosine",
    )

    assert len(store.index) == 3
    assert store.dim == len(VOCAB)

    # Query most similar to "C SIMD vector"
    docs = store.similarity_search("C SIMD vector", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "Bare metal C SIMD vector engine"
    assert docs[0].metadata["cat"] == "tech"
    assert docs[0].metadata["lang"] == "c"
    assert "_page_content" not in docs[0].metadata
    assert docs[0].id == "doc_c"


def test_similarity_search_with_score():
    embeddings = DummyEmbeddings()
    store = NanoVectorStore.from_texts(
        texts=["Fast search", "Slow search"],
        embedding=embeddings,
    )

    results = store.similarity_search_with_score("Fast search", k=2)
    assert len(results) == 2
    doc, score = results[0]
    assert doc.page_content == "Fast search"
    assert isinstance(score, float)
    assert score > 0.0


def test_langchain_filtering():
    embeddings = DummyEmbeddings()
    texts = [
        "NanoVector is written in pure C",
        "Chroma is written in Python",
        "Faiss is written in C++",
    ]
    metadatas = [
        {"author": "eminsk", "year": 2026, "type": "embedded"},
        {"author": "chroma", "year": 2023, "type": "service"},
        {"author": "meta", "year": 2017, "type": "library"},
    ]

    store = NanoVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )

    # Filter by exact match
    docs = store.similarity_search("embedded search", k=3, filter={"author": "eminsk"})
    assert len(docs) == 1
    assert docs[0].page_content == "NanoVector is written in pure C"

    # Filter with operator ($gte)
    docs_modern = store.similarity_search("vector engine", k=3, filter={"year": {"$gte": 2024}})
    assert len(docs_modern) == 1
    assert docs_modern[0].page_content == "NanoVector is written in pure C"


def test_from_documents_and_as_retriever():
    embeddings = DummyEmbeddings()
    documents = [
        Document(page_content="Document one for LCEL", metadata={"source": "doc1"}),
        Document(page_content="Document two for LCEL", metadata={"source": "doc2"}),
    ]

    store = NanoVectorStore.from_documents(documents, embedding=embeddings)
    assert len(store.index) == 2

    retriever = store.as_retriever(search_kwargs={"k": 1})
    retrieved = retriever.invoke("LCEL query")
    assert len(retrieved) == 1
    assert "Document" in retrieved[0].page_content


def test_save_and_load_persistence():
    embeddings = DummyEmbeddings()
    texts = ["Persistent memory across reboots", "Ephemeral session"]
    metadatas = [{"persisted": True}, {"persisted": False}]

    store = NanoVectorStore.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "langchain_store.nvec")
        store.save(filepath)
        assert os.path.exists(filepath)

        loaded_store = NanoVectorStore.load(filepath, embedding=embeddings)
        assert len(loaded_store.index) == 2
        assert loaded_store.dim == len(VOCAB)

        docs = loaded_store.similarity_search("Persistent memory", k=1)
        assert len(docs) == 1
        assert docs[0].page_content == "Persistent memory across reboots"
        assert docs[0].metadata["persisted"] is True