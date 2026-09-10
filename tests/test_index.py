"""
NanoVector Python API Test Suite
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import os
import pytest
import numpy as np
from nanovector import Index, Match, load, version, simd_backend


def test_version_and_backend():
    ver = version()
    backend = simd_backend()
    assert isinstance(ver, str) and len(ver) > 0
    assert isinstance(backend, str) and len(backend) > 0
    print(f"\n[INFO] NanoVector v{ver} on {backend}")


def test_cosine_similarity():
    dim = 384
    index = Index(dim=dim, metric="cosine")
    assert len(index) == 0
    assert index.dim == dim
    assert index.metric == "cosine"

    # Insert test vectors
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((100, dim)).astype(np.float32)
    ids = [f"item_{i}" for i in range(100)]
    metas = [f'{{"idx": {i}}}' for i in range(100)]

    index.add_batch(ids, vectors, metas)
    assert len(index) == 100

    # Query with exact item_42
    query = vectors[42]
    matches = index.search(query, top_k=5)

    assert len(matches) == 5
    assert matches[0].id == "item_42"
    assert pytest.approx(matches[0].score, abs=1e-4) == 1.0
    assert matches[0].metadata == '{"idx": 42}'

    # Scores must be descending
    for i in range(1, len(matches)):
        assert matches[i-1].score >= matches[i].score


def test_dot_product_metric():
    dim = 64
    index = Index(dim=dim, metric="dot")
    assert index.metric == "dot"

    v1 = np.ones(dim, dtype=np.float32)
    v2 = np.full(dim, 2.0, dtype=np.float32)

    index.add("one", v1)
    index.add("two", v2)

    # dot(v1, v1) = 64 * 1 = 64
    # dot(v1, v2) = 64 * 2 = 128
    matches = index.search(v1, top_k=2)
    assert len(matches) == 2
    assert matches[0].id == "two"
    assert pytest.approx(matches[0].score, abs=1e-4) == 128.0
    assert matches[1].id == "one"
    assert pytest.approx(matches[1].score, abs=1e-4) == 64.0


def test_l2_euclidean_distance():
    dim = 32
    index = Index(dim=dim, metric="l2")
    assert index.metric == "l2"

    v_zero = np.zeros(dim, dtype=np.float32)
    v_one = np.ones(dim, dtype=np.float32)
    v_ten = np.full(dim, 10.0, dtype=np.float32)

    index.add("zero", v_zero)
    index.add("one", v_one)
    index.add("ten", v_ten)

    # Query with zero: zero should be closest (d=0), then one (d=32), then ten (d=3200)
    matches = index.search(v_zero, top_k=3)
    assert len(matches) == 3
    assert matches[0].id == "zero"
    assert pytest.approx(matches[0].score, abs=1e-4) == 0.0
    assert matches[1].id == "one"
    assert pytest.approx(matches[1].score, abs=1e-4) == 32.0
    assert matches[2].id == "ten"
    assert pytest.approx(matches[2].score, abs=1e-4) == 3200.0


def test_persistence_save_load(tmp_path):
    dim = 128
    index = Index(dim=dim, metric="cosine")

    rng = np.random.default_rng(123)
    vecs = rng.standard_normal((50, dim)).astype(np.float32)
    ids = [f"agent_turn_{i}" for i in range(50)]
    metas = [f'{{"turn": {i}, "agent": "coder"}}' for i in range(50)]

    index.add_batch(ids, vecs, metas)

    filepath = str(tmp_path / "memory.nvec")
    index.save(filepath)
    assert os.path.exists(filepath)

    # Load into new index
    loaded = load(filepath)
    assert len(loaded) == 50
    assert loaded.dim == dim
    assert loaded.metric == "cosine"

    # Search in loaded index
    matches = loaded.search(vecs[10], top_k=3)
    assert len(matches) == 3
    assert matches[0].id == "agent_turn_10"
    assert pytest.approx(matches[0].score, abs=1e-4) == 1.0
    assert "coder" in matches[0].metadata


def test_edge_cases_and_exceptions():
    index = Index(dim=16)

    # Empty search
    matches = index.search(np.zeros(16, dtype=np.float32), top_k=5)
    assert matches == []

    # Dimension mismatch
    with pytest.raises(ValueError):
        index.add("bad", np.zeros(32, dtype=np.float32))

    with pytest.raises(ValueError):
        index.search(np.zeros(8, dtype=np.float32))

    # Python list support
    index.add("from_list", [1.0] * 16, metadata="list_item")
    assert len(index) == 1
    m = index.search([1.0] * 16, top_k=1)
    assert m[0].id == "from_list"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
