"""
NanoVector: Minimalist Bare-Metal Vector Search & Episodic Memory Engine
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import sys
import json
from typing import List, Optional, Any, Dict, Union, Callable, Sequence
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
        def version(): return "0.1.3"
        def simd_backend(): return "Scalar (Pending Build)"

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


class Index:
    """
    High-performance embedded vector index with SIMD AVX2/NEON/FASM acceleration.

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
        if _NativeIndex is None:
            raise RuntimeError("NanoVector C extension is not compiled.")
        self._index = _NativeIndex(dim=dim, metric=metric, normalize=normalize)

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
        if _NativeIndex is None:
            raise RuntimeError("NanoVector C extension is not compiled.")
        native = _NativeIndex.load(filepath)
        wrapper = cls.__new__(cls)
        wrapper._index = native
        return wrapper


def load(filepath: str) -> Index:
    """Convenience function to load a NanoVector index from disk."""
    return Index.load(filepath)


def __getattr__(name: str) -> Any:
    if name == "NanoVectorStore":
        from nanovector.integrations.langchain import NanoVectorStore
        return NanoVectorStore
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
]
