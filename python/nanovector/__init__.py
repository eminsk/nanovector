"""
NanoVector: Minimalist Bare-Metal Vector Search & Episodic Memory Engine
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import sys
from typing import List, Optional, Any, Dict, Union
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
        def version(): return "0.1.2"
        def simd_backend(): return "Scalar (Pending Build)"

__version__ = version()
__backend__ = simd_backend()


@dataclass(frozen=True, **_dataclass_kwargs)
class Match:
    """Represents a single nearest-neighbor search result."""
    id: str
    score: float
    metadata: Optional[str] = None

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

    def add(self, id: str, vector: Any, metadata: Optional[str] = None) -> None:
        """
        Add a single vector with an ID and optional metadata string.

        Parameters
        ----------
        id : str
            Unique document or chunk identifier.
        vector : array-like
            1D numpy array, list, or buffer of float32 values of size `dim`.
        metadata : str, optional
            Arbitrary JSON string or text tag.
        """
        self._index.add(id=id, vector=vector, metadata=metadata)

    def add_batch(
        self,
        ids: List[str],
        vectors: Any,
        metadatas: Optional[List[Optional[str]]] = None
    ) -> None:
        """
        Add multiple vectors in batch (Zero-Copy from 2D NumPy array).

        Parameters
        ----------
        ids : list of str
            List of string IDs.
        vectors : 2D numpy.ndarray or sequence
            2D array of shape (N, dim) with float32 data.
        metadatas : list of str, optional
            List of optional metadata strings.
        """
        self._index.add_batch(ids=ids, vectors=vectors, metadatas=metadatas)

    def search(self, query: Any, top_k: int = 10) -> List[Match]:
        """
        Search Top-K nearest neighbors for a query vector.

        Parameters
        ----------
        query : array-like
            1D numpy array or sequence of length `dim`.
        top_k : int, default 10
            Maximum number of nearest matches to return.

        Returns
        -------
        List[Match]
            List of matches sorted by score (descending for cosine/dot, ascending for L2).
        """
        raw_results = self._index.search(query=query, top_k=top_k)
        return [Match(id=r["id"], score=r["score"], metadata=r["metadata"]) for r in raw_results]

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


__all__ = ["Index", "Match", "load", "version", "simd_backend", "__version__", "__backend__"]
