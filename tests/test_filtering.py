"""
NanoVector Metadata Filtering Test Suite
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import unittest
from nanovector import Index, Match


def _approx(val, expected, tol=1e-4):
    assert abs(val - expected) <= tol, f"Expected {expected} +/- {tol}, got {val}"


def test_dict_metadata_addition_and_meta_property():
    dim = 8
    idx = Index(dim=dim, metric="cosine")

    v1 = [1.0] * dim
    meta_dict = {"title": "Zero to Hero", "author": "eminsk", "views": 1500, "published": True}
    idx.add("doc_1", v1, metadata=meta_dict)

    res = idx.search(v1, top_k=1)
    assert len(res) == 1
    match = res[0]
    assert match.id == "doc_1"
    assert isinstance(match.meta, dict)
    assert match.meta["author"] == "eminsk"
    assert match.meta["views"] == 1500
    assert match.meta["published"] is True


def test_batch_metadata_addition():
    dim = 4
    idx = Index(dim=dim, metric="cosine")

    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    metas = [
        {"cat": "tech", "lang": "c"},
        {"cat": "tech", "lang": "python"},
        {"cat": "science", "lang": "en"},
    ]
    ids = ["id_0", "id_1", "id_2"]

    idx.add_batch(ids, vectors, metadatas=metas)
    assert len(idx) == 3

    # Exact key-value filter
    res = idx.search(vectors[0], top_k=5, filter={"lang": "python"})
    assert len(res) == 1
    assert res[0].id == "id_1"
    assert res[0].meta["lang"] == "python"


def test_filtering_operators():
    dim = 4
    idx = Index(dim=dim, metric="cosine")

    vectors = [[1.0, 0.0, 0.0, 0.0] for _ in range(5)]
    ids = [f"item_{i}" for i in range(5)]
    metas = [
        {"score": 10, "status": "draft", "tag": "c"},
        {"score": 20, "status": "active", "tag": "cpp"},
        {"score": 30, "status": "active", "tag": "python"},
        {"score": 40, "status": "archived", "tag": "rust"},
        {"score": 50, "status": "active", "tag": "go"},
    ]
    idx.add_batch(ids, vectors, metadatas=metas)

    # Operator $gte
    res_gte = idx.search(vectors[0], top_k=10, filter={"score": {"$gte": 30}})
    assert len(res_gte) == 3
    assert {m.id for m in res_gte} == {"item_2", "item_3", "item_4"}

    # Operator $lt
    res_lt = idx.search(vectors[0], top_k=10, filter={"score": {"$lt": 30}})
    assert len(res_lt) == 2
    assert {m.id for m in res_lt} == {"item_0", "item_1"}

    # Operator $in
    res_in = idx.search(vectors[0], top_k=10, filter={"status": {"$in": ["draft", "archived"]}})
    assert len(res_in) == 2
    assert {m.id for m in res_in} == {"item_0", "item_3"}

    # Operator $nin
    res_nin = idx.search(vectors[0], top_k=10, filter={"status": {"$nin": ["active"]}})
    assert len(res_nin) == 2
    assert {m.id for m in res_nin} == {"item_0", "item_3"}

    # Operator $ne
    res_ne = idx.search(vectors[0], top_k=10, filter={"status": {"$ne": "active"}})
    assert len(res_ne) == 2
    assert {m.id for m in res_ne} == {"item_0", "item_3"}

    # List inclusion condition
    res_list = idx.search(vectors[0], top_k=10, filter={"tag": ["c", "rust"]})
    assert len(res_list) == 2
    assert {m.id for m in res_list} == {"item_0", "item_3"}


def test_callable_filtering():
    dim = 4
    idx = Index(dim=dim, metric="cosine")

    vectors = [[1.0, 0.0, 0.0, 0.0] for _ in range(4)]
    ids = ["a", "b", "c", "d"]
    metas = [
        {"x": 10, "y": 20},
        {"x": 25, "y": 5},
        {"x": 5, "y": 50},
        {"x": 30, "y": 30},
    ]
    idx.add_batch(ids, vectors, metadatas=metas)

    # Filter where x + y >= 50
    res = idx.search(vectors[0], top_k=10, filter=lambda m: (m.get("x", 0) + m.get("y", 0)) >= 50)
    assert len(res) == 2
    assert {m.id for m in res} == {"c", "d"}


class TestFiltering(unittest.TestCase):
    """Unittest adapter for standard discovery."""
    def test_dict_metadata(self):
        test_dict_metadata_addition_and_meta_property()

    def test_batch_metadata(self):
        test_batch_metadata_addition()

    def test_operators(self):
        test_filtering_operators()

    def test_callable(self):
        test_callable_filtering()


if __name__ == "__main__":
    unittest.main()