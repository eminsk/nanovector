"""
NanoVector: Minimalist Bare-Metal Vector Search & Episodic Memory Engine
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

from __future__ import annotations

import sys
import json
import math
import struct
from typing import List, Optional, Any, Dict, Union, Callable, Sequence, Tuple
from dataclasses import dataclass

_dataclass_kwargs = {"slots": True} if sys.version_info >= (3, 10) else {}

try:
    from nanovector._ext import Index as _NativeIndex, version, simd_backend
except ImportError:
    # Fallback when running before build or in documentation generation
    try:
        from _ext import Index as _NativeIndex, version, simd_backend
    except ImportError:
        _NativeIndex = None
        def version(): return "0.1.8"
        def simd_backend(): return "Pure-Python (Zero-Dependency Fallback Engine)"

__version__ = version()
__backend__ = simd_backend()


def _matches_filter(meta: Any, filter_spec: Union[Dict[str, Any], Callable[[Any], bool]]) -> bool:
    """Internal helper to test whether a match's parsed metadata satisfies the filter specification."""
    if callable(filter_spec):
        try:
            return bool(filter_spec(meta))
        except Exception:
            return False

    if not isinstance(filter_spec, dict):
        return True

    if not isinstance(meta, dict):
        return False

    for key, condition in filter_spec.items():
        val = meta.get(key)
        if isinstance(condition, dict):
            for op, target in condition.items():
                if op == "$eq" and not (val == target):
                    return False
                elif op == "$ne" and not (val != target):
                    return False
                elif op == "$in" and not (val in target):
                    return False
                elif op == "$nin" and not (val not in target):
                    return False
                elif op == "$gt" and (val is None or not (val > target)):
                    return False
                elif op == "$gte" and (val is None or not (val >= target)):
                    return False
                elif op == "$lt" and (val is None or not (val < target)):
                    return False
                elif op == "$lte" and (val is None or not (val <= target)):
                    return False
                elif op == "$exists":
                    exists = key in meta
                    if bool(target) != exists:
                        return False
        elif isinstance(condition, (list, tuple, set)):
            if val not in condition:
                return False
        else:
            if val != condition:
                return False
    return True


@dataclass(frozen=True, **_dataclass_kwargs)
class Match:
    """Represents a single nearest-neighbor search result."""
    id: str
    score: float
    metadata: Optional[str] = None

    @property
    def meta(self) -> Optional[Any]:
        """
        Parses JSON metadata string into a Python dict or primitive if valid JSON.
        Returns None if metadata is empty, or the raw string if not JSON.
        """
        if not self.metadata:
            return None
        try:
            return json.loads(self.metadata)
        except Exception:
            return self.metadata

    def __repr__(self) -> str:
        return f"Match(id='{self.id}', score={self.score:.4f}, metadata={self.metadata!r})"


class _PurePythonIndex:
    """Zero-dependency pure-Python vector index fallback with exact parity."""

    def __init__(self, dim: int, metric: str = "cosine", normalize: bool = False):
        if dim == 0:
            raise ValueError("Dimension 'dim' must be greater than 0")
        m = metric.lower()
        if m in ("cosine", "dot", "ip", "l2", "euclidean"):
            if m == "ip":
                m = "dot"
            elif m == "euclidean":
                m = "l2"
        else:
            raise ValueError(f"Invalid metric '{metric}'. Supported metrics: 'cosine', 'dot'/'ip', 'l2'/'euclidean'")
        self.dim: int = int(dim)
        self.metric: str = m
        self.normalize: bool = bool(normalize)
        self.count: int = 0
        self._ids: List[str] = []
        self._vectors: List[List[float]] = []
        self._norms: List[float] = []
        self._metadatas: List[Optional[str]] = []

    def _convert_vec(self, v: Any) -> Tuple[List[float], float]:
        if hasattr(v, "tolist"):
            raw = v.tolist()
        else:
            raw = list(v)
        if len(raw) != self.dim:
            raise ValueError(f"Vector dim {len(raw)} does not match index dim {self.dim}")
        flts = [float(x) for x in raw]
        norm = math.sqrt(sum(x * x for x in flts))
        if self.normalize:
            if norm > 1e-12:
                flts = [x / norm for x in flts]
                norm = 1.0
        return flts, norm

    def add(self, id: str, vector: Any, metadata: Optional[str] = None) -> None:
        flts, norm = self._convert_vec(vector)
        self._ids.append(str(id))
        self._vectors.append(flts)
        self._norms.append(norm)
        self._metadatas.append(metadata)
        self.count += 1

    def add_batch(self, ids: List[str], vectors: Any, metadatas: Optional[Sequence[Optional[str]]] = None) -> None:
        for i, id_val in enumerate(ids):
            m = metadatas[i] if metadatas and i < len(metadatas) else None
            self.add(id_val, vectors[i], m)

    def search(self, query: Any, top_k: int = 10) -> List[Dict[str, Any]]:
        q_vec, q_norm = self._convert_vec(query)
        if self.count == 0 or top_k <= 0:
            return []
        scores: List[tuple[float, int]] = []
        is_l2 = (self.metric == "l2")

        for i in range(self.count):
            vec = self._vectors[i]
            if is_l2:
                dist_sq = sum((a - b) * (a - b) for a, b in zip(q_vec, vec))
                scores.append((dist_sq, i))
            elif self.metric == "cosine":
                denom = q_norm * self._norms[i]
                dot = sum(a * b for a, b in zip(q_vec, vec))
                score = (dot / denom) if denom > 1e-12 else 0.0
                scores.append((score, i))
            else:  # dot
                dot = sum(a * b for a, b in zip(q_vec, vec))
                scores.append((dot, i))

        scores.sort(key=lambda x: x[0], reverse=(not is_l2))
        top = scores[:top_k]
        return [
            {"id": self._ids[i], "score": float(s), "metadata": self._metadatas[i]}
            for s, i in top
        ]

    def save(self, filepath: str) -> None:
        metric_code = 0 if self.metric == "cosine" else (1 if self.metric == "dot" else 2)
        with open(filepath, "wb") as f:
            hdr = struct.pack("<4sIIQII36s", b"NVEC", 1, self.dim, self.count, metric_code, int(self.normalize), b"\x00" * 36)
            f.write(hdr)
            for v in self._vectors:
                f.write(struct.pack(f"<{len(v)}f", *v))
            for i in range(self.count):
                id_b = self._ids[i].encode("utf-8")
                f.write(struct.pack("<H", len(id_b)))
                f.write(id_b)
                meta_b = self._metadatas[i].encode("utf-8") if self._metadatas[i] else b""
                f.write(struct.pack("<I", len(meta_b)))
                f.write(meta_b)

    @classmethod
    def load(cls, filepath: str) -> "_PurePythonIndex":
        with open(filepath, "rb") as f:
            hdr_bytes = f.read(64)
            if len(hdr_bytes) < 64:
                raise IOError(f"Invalid header in {filepath}")
            magic, ver, dim, count, metric_code, norm_code, _ = struct.unpack("<4sIIQII36s", hdr_bytes)
            if magic != b"NVEC" or ver != 1:
                raise IOError(f"Invalid magic or version in {filepath}")
            metric_str = "cosine" if metric_code == 0 else ("dot" if metric_code == 1 else "l2")
            idx = cls(dim=dim, metric=metric_str, normalize=bool(norm_code))
            for _ in range(count):
                v_bytes = f.read(dim * 4)
                flts = list(struct.unpack(f"<{dim}f", v_bytes))
                idx._vectors.append(flts)
                norm = math.sqrt(sum(x * x for x in flts))
                idx._norms.append(norm)
            for _ in range(count):
                id_len = struct.unpack("<H", f.read(2))[0]
                id_str = f.read(id_len).decode("utf-8") if id_len > 0 else ""
                idx._ids.append(id_str)
                meta_len = struct.unpack("<I", f.read(4))[0]
                meta_str = f.read(meta_len).decode("utf-8") if meta_len > 0 else None
                idx._metadatas.append(meta_str)
            idx.count = count
            return idx


class Index:
    """
    High-performance embedded vector index with SIMD AVX2/NEON/FASM acceleration
    and pure-Python fallback.

    Parameters
    ----------
    dim : int
        Vector dimensionality (e.g. 384 for all-MiniLM-L6-v2, 768 for BERT, 1536 for OpenAI text-embedding-3-small).
    metric : str, default 'cosine'
        Distance metric: 'cosine', 'dot' / 'ip', or 'l2' / 'euclidean'.
    normalize : bool, default False
        If True, vectors are automatically L2-normalized upon insertion and search.
    """

    def __init__(self, dim: int, metric: str = "cosine", normalize: bool = False):
        if _NativeIndex is not None:
            self._index = _NativeIndex(dim=dim, metric=metric, normalize=normalize)
        else:
            self._index = _PurePythonIndex(dim=dim, metric=metric, normalize=normalize)

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._index.dim

    @property
    def count(self) -> int:
        """Total number of indexed vectors."""
        return self._index.count

    @property
    def metric(self) -> str:
        """Distance or similarity metric used by this index."""
        return self._index.metric

    def __len__(self) -> int:
        return self._index.count

    def add(self, id: str, vector: Any, metadata: Optional[Union[str, Dict[str, Any]]] = None) -> None:
        """
        Add a single vector with an ID and optional metadata (dict or JSON string).

        Parameters
        ----------
        id : str
            Unique document or chunk identifier.
        vector : array-like
            1D numpy array, list, or buffer of float32 values of size `dim`.
        metadata : dict or str, optional
            Arbitrary metadata dictionary (automatically serialized to JSON) or raw text/JSON string.
        """
        if isinstance(metadata, dict):
            metadata_str = json.dumps(metadata, ensure_ascii=False)
        elif metadata is not None:
            metadata_str = str(metadata)
        else:
            metadata_str = None
        self._index.add(id=id, vector=vector, metadata=metadata_str)

    def add_batch(
        self,
        ids: List[str],
        vectors: Any,
        metadatas: Optional[Sequence[Optional[Union[str, Dict[str, Any]]]]] = None
    ) -> None:
        """
        Add multiple vectors in batch (Zero-Copy from 2D NumPy array).

        Parameters
        ----------
        ids : list of str
            List of string IDs.
        vectors : 2D numpy.ndarray or sequence
            2D array of shape (N, dim) with float32 data.
        metadatas : list of dict or str, optional
            List of optional metadata dictionaries or JSON strings.
        """
        if metadatas is not None:
            str_metadatas = []
            for m in metadatas:
                if isinstance(m, dict):
                    str_metadatas.append(json.dumps(m, ensure_ascii=False))
                elif m is not None:
                    str_metadatas.append(str(m))
                else:
                    str_metadatas.append(None)
            metadatas = str_metadatas
        self._index.add_batch(ids=ids, vectors=vectors, metadatas=metadatas)

    def search(
        self,
        query: Any,
        top_k: int = 10,
        filter: Optional[Union[Dict[str, Any], Callable[[Any], bool]]] = None,
    ) -> List[Match]:
        """
        Search Top-K nearest neighbors for a query vector with optional metadata filtering.

        Parameters
        ----------
        query : array-like
            1D numpy array or sequence of length `dim`.
        top_k : int, default 10
            Maximum number of nearest matches to return.
        filter : dict or callable, optional
            Optional metadata filter. Can be:
            - A dictionary of exact key-values: `{"tag": "ai", "user_id": 42}`
            - A dictionary with list inclusion: `{"category": ["books", "news"]}`
            - A dictionary with operators: `{"views": {"$gte": 100}, "archived": {"$ne": True}}`
            - A callable predicate function: `lambda m: m and m.get("score", 0) > 0.5`

        Returns
        -------
        List[Match]
            List of matches sorted by score (descending for cosine/dot, ascending for L2).
        """
        if filter is None:
            raw_results = self._index.search(query=query, top_k=top_k)
            return [Match(id=r["id"], score=r["score"], metadata=r["metadata"]) for r in raw_results]

        total_count = self.count
        if total_count == 0 or top_k <= 0:
            return []

        oversample_k = min(total_count, max(top_k * 10, 100))
        raw_results = self._index.search(query=query, top_k=oversample_k)

        filtered: List[Match] = []
        for r in raw_results:
            match = Match(id=r["id"], score=r["score"], metadata=r["metadata"])
            if _matches_filter(match.meta, filter):
                filtered.append(match)
                if len(filtered) == top_k:
                    return filtered

        # If candidates pool was exhausted before finding top_k, do full scan for 100% exact recall
        if len(filtered) < top_k and oversample_k < total_count:
            filtered = []
            raw_results = self._index.search(query=query, top_k=total_count)
            for r in raw_results:
                match = Match(id=r["id"], score=r["score"], metadata=r["metadata"])
                if _matches_filter(match.meta, filter):
                    filtered.append(match)
                    if len(filtered) == top_k:
                        break

        return filtered

    def save(self, filepath: str) -> None:
        """
        Save the entire index to a single .nvec binary file.

        Parameters
        ----------
        filepath : str
            Path to output binary file (e.g. 'memory.nvec').
        """
        self._index.save(filepath)

    @classmethod
    def load(cls, filepath: str) -> "Index":
        """
        Load an index from a .nvec file.

        Parameters
        ----------
        filepath : str
            Path to .nvec file.

        Returns
        -------
        Index
            Loaded Index instance.
        """
        wrapper = cls.__new__(cls)
        if _NativeIndex is not None:
            try:
                wrapper._index = _NativeIndex.load(filepath)
                return wrapper
            except Exception:
                pass
        wrapper._index = _PurePythonIndex.load(filepath)
        return wrapper


def load(filepath: str) -> Index:
    """Convenience function to load a NanoVector index from disk."""
    return Index.load(filepath)


def __getattr__(name: str) -> Any:
    if name == "NanoVectorStore":
        from nanovector.integrations.langchain import NanoVectorStore
        return NanoVectorStore
    if name in ("NanoVectorMCPServer", "embed_text"):
        from nanovector import mcp_server
        return getattr(mcp_server, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "Index",
    "Match",
    "load",
    "version",
    "simd_backend",
    "__version__",
    "__backend__",
    "NanoVectorStore",
    "NanoVectorMCPServer",
    "embed_text",
]
