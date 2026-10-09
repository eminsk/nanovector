"""
NanoVector FASM Hardware Assembly Acceleration Backend
Direct bare-metal AVX2+FMA (x86-64) and SSE2 (x86 32-bit) SIMD acceleration.
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_FASM_LIB: Optional[ctypes.CDLL] = None
_FASM_ISA: str = "Unavailable"


def _bind_fasm_library(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Bind CTypes signatures to the loaded FASM shared library."""
    if hasattr(lib, "nanovec_fasm_dot"):
        lib.nanovec_fasm_dot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.nanovec_fasm_dot.restype = ctypes.c_float

    if hasattr(lib, "nanovec_fasm_norm_sq"):
        lib.nanovec_fasm_norm_sq.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.nanovec_fasm_norm_sq.restype = ctypes.c_float

    if hasattr(lib, "nanovec_fasm_l2_sq"):
        lib.nanovec_fasm_l2_sq.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.nanovec_fasm_l2_sq.restype = ctypes.c_float

    if hasattr(lib, "nanovec_fasm_normalize"):
        lib.nanovec_fasm_normalize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.nanovec_fasm_normalize.restype = ctypes.c_float

    if hasattr(lib, "nanovec_fasm_batch_dot"):
        lib.nanovec_fasm_batch_dot.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p
        ]
        lib.nanovec_fasm_batch_dot.restype = None

    if hasattr(lib, "nanovec_fasm_batch_l2_sq"):
        lib.nanovec_fasm_batch_l2_sq.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p
        ]
        lib.nanovec_fasm_batch_l2_sq.restype = None

    if hasattr(lib, "nanovec_fasm_batch_cosine"):
        lib.nanovec_fasm_batch_cosine.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p
        ]
        lib.nanovec_fasm_batch_cosine.restype = None

    for isa_attr in ("nanovec_fasm_simd_isa", "nanovec_simd_isa"):
        if hasattr(lib, isa_attr):
            fn = getattr(lib, isa_attr)
            fn.argtypes = []
            fn.restype = ctypes.c_char_p

    return lib


def _find_and_load_fasm_lib() -> Optional[ctypes.CDLL]:
    """Locates and loads the native FASM DLL (64-bit or 32-bit)."""
    global _FASM_ISA
    if not sys.platform.startswith("win"):
        return None

    is_64bit = sys.maxsize > 2**32
    dll_name = "nanovector64.dll" if is_64bit else "nanovector32.dll"

    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / dll_name,
        pkg_dir.parent / dll_name,
        pkg_dir.parent / "asm" / dll_name,
        pkg_dir.parent.parent / "asm" / dll_name,
        Path(r"C:\proekts\nanovector\asm") / dll_name,
        Path(r"C:\proekts\nanovector\python\nanovector") / dll_name,
    ]

    for cand in candidates:
        if cand.exists():
            try:
                lib = ctypes.CDLL(str(cand))
                bound_lib = _bind_fasm_library(lib)
                for isa_attr in ("nanovec_fasm_simd_isa", "nanovec_simd_isa"):
                    if hasattr(bound_lib, isa_attr):
                        fn = getattr(bound_lib, isa_attr)
                        _FASM_ISA = fn().decode("utf-8", errors="replace")
                        break
                return bound_lib
            except Exception:
                continue

    return None


_FASM_LIB = _find_and_load_fasm_lib()


def is_available() -> bool:
    """Return True if hardware FASM SIMD engine is loaded and operational."""
    return _FASM_LIB is not None


def get_isa() -> str:
    """Return the detected FASM SIMD Instruction Set Architecture."""
    return _FASM_ISA


def _to_c_float_array(obj: Any) -> Tuple[ctypes.Array, int]:
    """Helper to convert list, tuple, buffer or numpy array to ctypes c_float array."""
    # NumPy fast path
    if hasattr(obj, "ctypes") and hasattr(obj.ctypes, "data"):
        return obj.ctypes.data, len(obj)

    # Flat sequence
    if isinstance(obj, (list, tuple)):
        dim = len(obj)
        arr = (ctypes.c_float * dim)(*(float(x) for x in obj))
        return arr, dim

    # ctypes array directly
    if isinstance(obj, ctypes.Array):
        return obj, len(obj)

    seq = list(obj)
    dim = len(seq)
    arr = (ctypes.c_float * dim)(*(float(x) for x in seq))
    return arr, dim


def dot(a: Any, b: Any, dim: Optional[int] = None) -> float:
    """Compute dot product of two float32 vectors using FASM hardware SIMD."""
    if _FASM_LIB is None:
        raise RuntimeError("FASM engine is not loaded")
    c_a, len_a = _to_c_float_array(a)
    c_b, len_b = _to_c_float_array(b)
    n = dim if dim is not None else min(len_a, len_b)
    ptr_a = c_a if isinstance(c_a, int) else ctypes.byref(c_a)
    ptr_b = c_b if isinstance(c_b, int) else ctypes.byref(c_b)
    return float(_FASM_LIB.nanovec_fasm_dot(ptr_a, ptr_b, n))


def norm_sq(a: Any, dim: Optional[int] = None) -> float:
    """Compute squared L2 norm using FASM hardware SIMD."""
    if _FASM_LIB is None:
        raise RuntimeError("FASM engine is not loaded")
    c_a, len_a = _to_c_float_array(a)
    n = dim if dim is not None else len_a
    ptr_a = c_a if isinstance(c_a, int) else ctypes.byref(c_a)
    return float(_FASM_LIB.nanovec_fasm_norm_sq(ptr_a, n))


def l2_sq(a: Any, b: Any, dim: Optional[int] = None) -> float:
    """Compute squared L2 Euclidean distance using FASM hardware SIMD."""
    if _FASM_LIB is None:
        raise RuntimeError("FASM engine is not loaded")
    c_a, len_a = _to_c_float_array(a)
    c_b, len_b = _to_c_float_array(b)
    n = dim if dim is not None else min(len_a, len_b)
    ptr_a = c_a if isinstance(c_a, int) else ctypes.byref(c_a)
    ptr_b = c_b if isinstance(c_b, int) else ctypes.byref(c_b)
    return float(_FASM_LIB.nanovec_fasm_l2_sq(ptr_a, ptr_b, n))


def normalize_inplace(a: Any, dim: Optional[int] = None) -> float:
    """Normalize vector in-place using FASM hardware SIMD. Returns original norm."""
    if _FASM_LIB is None:
        raise RuntimeError("FASM engine is not loaded")
    c_a, len_a = _to_c_float_array(a)
    n = dim if dim is not None else len_a
    ptr_a = c_a if isinstance(c_a, int) else ctypes.byref(c_a)
    return float(_FASM_LIB.nanovec_fasm_normalize(ptr_a, n))


class FasmIndex:
    """
    High-performance vector index powered directly by the Bare-Metal FASM Engine.
    Delivers 15-20 MVec/sec bare-metal search speed with zero dependencies.
    """

    def __init__(self, dim: int, metric: str = "cosine", normalize: bool = False):
        if _FASM_LIB is None:
            raise RuntimeError("FASM native library is not available on this platform")
        if dim <= 0:
            raise ValueError("Dimension 'dim' must be greater than 0")

        m = metric.lower()
        if m in ("cosine", "dot", "ip", "l2", "euclidean"):
            if m == "ip":
                m = "dot"
            elif m == "euclidean":
                m = "l2"
        else:
            raise ValueError(f"Invalid metric '{metric}'. Supported metrics: 'cosine', 'dot', 'l2'")

        self.dim: int = int(dim)
        self.metric: str = m
        self.normalize: bool = bool(normalize)
        self.count: int = 0

        self._ids: List[str] = []
        self._metadatas: List[Optional[str]] = []
        self._norms: List[float] = []
        # Store flat contiguous array of floats for SIMD batch sweeps
        self._flat_vectors: List[float] = []

    def _prepare_vector(self, v: Any) -> Tuple[List[float], float]:
        if hasattr(v, "tolist"):
            raw = v.tolist()
        else:
            raw = list(v)
        if len(raw) != self.dim:
            raise ValueError(f"Vector dim {len(raw)} does not match index dim {self.dim}")

        flts = [float(x) for x in raw]
        c_arr = (ctypes.c_float * self.dim)(*flts)
        norm = 0.0

        if self.normalize:
            norm = _FASM_LIB.nanovec_fasm_normalize(ctypes.byref(c_arr), self.dim)
            flts = [c_arr[i] for i in range(self.dim)]
            norm = 1.0
        else:
            n_sq = _FASM_LIB.nanovec_fasm_norm_sq(ctypes.byref(c_arr), self.dim)
            norm = math.sqrt(n_sq)

        return flts, norm

    def add(self, id: str, vector: Any, metadata: Optional[str] = None) -> None:
        flts, norm = self._prepare_vector(vector)
        self._ids.append(str(id))
        self._flat_vectors.extend(flts)
        self._norms.append(norm)
        self._metadatas.append(metadata)
        self.count += 1

    def add_batch(self, ids: List[str], vectors: Any, metadatas: Optional[Sequence[Optional[str]]] = None) -> None:
        for i, id_val in enumerate(ids):
            m = metadatas[i] if metadatas and i < len(metadatas) else None
            self.add(id_val, vectors[i], m)

    def search(self, query: Any, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.count == 0 or top_k <= 0:
            return []

        q_flts, q_norm = self._prepare_vector(query)
        q_c = (ctypes.c_float * self.dim)(*q_flts)
        mat_c = (ctypes.c_float * len(self._flat_vectors))(*self._flat_vectors)
        norms_c = (ctypes.c_float * len(self._norms))(*self._norms)
        scores_c = (ctypes.c_float * self.count)()

        if self.metric == "cosine":
            _FASM_LIB.nanovec_fasm_batch_cosine(
                ctypes.byref(q_c), float(q_norm),
                ctypes.byref(mat_c), ctypes.byref(norms_c),
                self.count, self.dim,
                ctypes.byref(scores_c)
            )
        elif self.metric == "dot":
            _FASM_LIB.nanovec_fasm_batch_dot(
                ctypes.byref(q_c), ctypes.byref(mat_c),
                self.count, self.dim,
                ctypes.byref(scores_c)
            )
        else:  # l2
            _FASM_LIB.nanovec_fasm_batch_l2_sq(
                ctypes.byref(q_c), ctypes.byref(mat_c),
                self.count, self.dim,
                ctypes.byref(scores_c)
            )

        is_l2 = (self.metric == "l2")
        scores_list = [(scores_c[i], i) for i in range(self.count)]
        scores_list.sort(key=lambda x: x[0], reverse=(not is_l2))

        top = scores_list[:top_k]
        return [
            {"id": self._ids[i], "score": float(s), "metadata": self._metadatas[i]}
            for s, i in top
        ]

    def save(self, filepath: str) -> None:
        metric_code = 0 if self.metric == "cosine" else (1 if self.metric == "dot" else 2)
        with open(filepath, "wb") as f:
            hdr = struct.pack("<4sIIQII36s", b"NVEC", 1, self.dim, self.count, metric_code, int(self.normalize), b"\x00" * 36)
            f.write(hdr)
            f.write(struct.pack(f"<{len(self._flat_vectors)}f", *self._flat_vectors))
            for i in range(self.count):
                id_b = self._ids[i].encode("utf-8")
                f.write(struct.pack("<H", len(id_b)))
                f.write(id_b)
                meta_b = self._metadatas[i].encode("utf-8") if self._metadatas[i] else b""
                f.write(struct.pack("<I", len(meta_b)))
                f.write(meta_b)

    @classmethod
    def load(cls, filepath: str) -> "FasmIndex":
        with open(filepath, "rb") as f:
            hdr_bytes = f.read(64)
            if len(hdr_bytes) < 64:
                raise IOError(f"Invalid header in {filepath}")
            magic, ver, dim, count, metric_code, norm_code, _ = struct.unpack("<4sIIQII36s", hdr_bytes)
            if magic != b"NVEC" or ver != 1:
                raise IOError(f"Invalid magic or version in {filepath}")
            metric_str = "cosine" if metric_code == 0 else ("dot" if metric_code == 1 else "l2")
            idx = cls(dim=dim, metric=metric_str, normalize=bool(norm_code))
            total_floats = count * dim
            v_bytes = f.read(total_floats * 4)
            idx._flat_vectors = list(struct.unpack(f"<{total_floats}f", v_bytes))
            for i in range(count):
                if idx.normalize:
                    idx._norms.append(1.0)
                else:
                    row_flts = idx._flat_vectors[i * dim : (i + 1) * dim]
                    idx._norms.append(math.sqrt(sum(x * x for x in row_flts)))
            for _ in range(count):
                id_len = struct.unpack("<H", f.read(2))[0]
                id_str = f.read(id_len).decode("utf-8") if id_len > 0 else ""
                idx._ids.append(id_str)
                meta_len = struct.unpack("<I", f.read(4))[0]
                meta_str = f.read(meta_len).decode("utf-8") if meta_len > 0 else None
                idx._metadatas.append(meta_str)
            idx.count = count
            return idx
