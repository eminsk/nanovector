"""
NanoVector Python API Test Suite
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import math
import os
import random
import tempfile
import unittest
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

try:
    import numpy as np
except ImportError:
    np = None

from nanovector import Index, Match, load, version, simd_backend


def _make_rand_vectors(count: int, dim: int, seed: int = 42):
    if np is not None:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((count, dim)).astype(np.float32)
    rnd = random.Random(seed)
    return [[rnd.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(count)]


def _make_const_vector(dim: int, val: float):
    if np is not None:
        return np.full(dim, val, dtype=np.float32)
    return [float(val)] * dim


def _approx(val, expected, tol=1e-4):
    assert abs(val - expected) <= tol, f"Expected {expected} +/- {tol}, got {val}"


def test_version_and_backend():
    ver = version()
    backend = simd_backend()
    assert isinstance(ver, str) and len(ver) > 0
    assert isinstance(backend, str) and len(backend) > 0


def test_cosine_similarity():
    dim = 384
    index = Index(dim=dim, metric="cosine")
    assert len(index) == 0
    assert index.dim == dim
    assert index.metric == "cosine"

    vectors = _make_rand_vectors(100, dim, seed=42)
    ids = [f"item_{i}" for i in range(100)]
    metas = [f'{{"idx": {i}}}' for i in range(100)]

    index.add_batch(ids, vectors, metas)
    assert len(index) == 100

    query = vectors[42]
    matches = index.search(query, top_k=5)

    assert len(matches) == 5
    assert matches[0].id == "item_42"
    _approx(matches[0].score, 1.0, tol=1e-4)
    assert matches[0].metadata == '{"idx": 42}'

    for i in range(1, len(matches)):
        assert matches[i - 1].score >= matches[i].score


def test_dot_product_metric():
    dim = 64
    index = Index(dim=dim, metric="dot")
    assert index.metric == "dot"

    v1 = _make_const_vector(dim, 1.0)
    v2 = _make_const_vector(dim, 2.0)

    index.add("one", v1)
    index.add("two", v2)

    matches = index.search(v1, top_k=2)
    assert len(matches) == 2
    assert matches[0].id == "two"
    _approx(matches[0].score, 128.0, tol=1e-3)
    assert matches[1].id == "one"
    _approx(matches[1].score, 64.0, tol=1e-3)


def test_l2_euclidean_distance():
    dim = 32
    index = Index(dim=dim, metric="l2")
    assert index.metric == "l2"

    v_zero = _make_const_vector(dim, 0.0)
    v_one = _make_const_vector(dim, 1.0)
    v_ten = _make_const_vector(dim, 10.0)

    index.add("zero", v_zero)
    index.add("one", v_one)
    index.add("ten", v_ten)

    matches = index.search(v_zero, top_k=3)
    assert len(matches) == 3
    assert matches[0].id == "zero"
    _approx(matches[0].score, 0.0, tol=1e-4)
    assert matches[1].id == "one"
    _approx(matches[1].score, 32.0, tol=1e-3)
    assert matches[2].id == "ten"
    _approx(matches[2].score, 3200.0, tol=1e-2)


def test_persistence_save_load(tmp_path=None):
    dim = 128
    index = Index(dim=dim, metric="cosine")

    vecs = _make_rand_vectors(50, dim, seed=123)
    ids = [f"agent_turn_{i}" for i in range(50)]
    metas = [f'{{"turn": {i}, "agent": "coder"}}' for i in range(50)]

    index.add_batch(ids, vecs, metas)

    if tmp_path is not None:
        filepath = str(Path(tmp_path) / "memory.nvec")
    else:
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, "memory_test_index.nvec")

    try:
        index.save(filepath)
        assert os.path.exists(filepath)

        loaded = load(filepath)
        assert len(loaded) == 50
        assert loaded.dim == dim
        assert loaded.metric == "cosine"

        matches = loaded.search(vecs[10], top_k=3)
        assert len(matches) == 3
        assert matches[0].id == "agent_turn_10"
        _approx(matches[0].score, 1.0, tol=1e-4)
        assert "coder" in matches[0].metadata
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_edge_cases_and_exceptions():
    index = Index(dim=16)

    matches = index.search(_make_const_vector(16, 0.0), top_k=5)
    assert matches == []

    # Dimension mismatch
    try:
        index.add("bad", _make_const_vector(32, 0.0))
        assert False, "Should raise ValueError on dimension mismatch"
    except ValueError:
        pass

    try:
        index.search(_make_const_vector(8, 0.0))
        assert False, "Should raise ValueError on dimension mismatch"
    except ValueError:
        pass

    # Python list support
    index.add("from_list", [1.0] * 16, metadata="list_item")
    assert len(index) == 1
    m = index.search([1.0] * 16, top_k=1)
    assert m[0].id == "from_list"


class TestIndex(unittest.TestCase):
    """Unittest adapter for standard discovery."""
    def test_version_and_backend(self):
        test_version_and_backend()

    def test_cosine_similarity(self):
        test_cosine_similarity()

    def test_dot_product_metric(self):
        test_dot_product_metric()

    def test_l2_euclidean_distance(self):
        test_l2_euclidean_distance()

    def test_persistence_save_load(self):
        test_persistence_save_load()

    def test_edge_cases_and_exceptions(self):
        test_edge_cases_and_exceptions()


if __name__ == "__main__":
    unittest.main()
