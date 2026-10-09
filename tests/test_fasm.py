"""
Comprehensive Tests for NanoVector FASM 32-bit and 64-bit Microkernels.
Tests native standalone executables and x86-64/x86-32 DLLs via ctypes against mathematical reference.
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

if sys.platform != "win32":
    raise unittest.SkipTest("FASM PE binaries (.dll / .exe) are Windows-specific")


def _find_asm_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "asm",
        Path(__file__).resolve().parent.parent / "asm",
        Path(r"C:\proekts\nanovector\asm"),
    ]
    for p in candidates:
        resolved = p.resolve()
        if resolved.is_dir() and ((resolved / "nanovector64.dll").exists() or (resolved / "build.bat").exists()):
            return resolved
    return candidates[0].resolve()


ASM_DIR = _find_asm_dir()
DLL64_PATH = ASM_DIR / "nanovector64.dll"
EXE64_PATH = ASM_DIR / "test_nanovector64.exe"
DLL32_PATH = ASM_DIR / "nanovector32.dll"
EXE32_PATH = ASM_DIR / "test_nanovector32.exe"
BUILD_BAT = ASM_DIR / "build.bat"


def ensure_binaries_built():
    """Builds FASM binaries using build.bat if any target is missing."""
    needed = [DLL64_PATH, EXE64_PATH, DLL32_PATH, EXE32_PATH]
    if any(not p.exists() for p in needed):
        res = subprocess.run(["cmd.exe", "/c", str(BUILD_BAT)], cwd=str(ASM_DIR), capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"Assembly build failed:\n{res.stdout}\n{res.stderr}"


def test_fasm64_standalone_exe():
    """Runs the 64-bit native PE console test suite."""
    ensure_binaries_built()
    assert EXE64_PATH.exists(), f"Executable not found: {EXE64_PATH}"
    res = subprocess.run([str(EXE64_PATH)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"test_nanovector64.exe failed with code {res.returncode}:\n{res.stdout}"
    assert "ALL 64-BIT FASM NATIVE TESTS PASSED" in res.stdout


def test_fasm32_standalone_exe():
    """Runs the 32-bit native PE console test suite under WoW64."""
    ensure_binaries_built()
    assert EXE32_PATH.exists(), f"Executable not found: {EXE32_PATH}"
    res = subprocess.run([str(EXE32_PATH)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"test_nanovector32.exe failed with code {res.returncode}:\n{res.stdout}"
    assert "ALL 32-BIT FASM NATIVE TESTS PASSED" in res.stdout


def test_fasm_ctypes_vector_metrics():
    """Tests direct FASM microkernel calculations against IEEE-754 Python math."""
    ensure_binaries_built()
    from nanovector import fasm
    assert fasm.is_available(), "FASM backend should be available on Windows"
    assert "SIMD" in fasm.get_isa() or "AVX" in fasm.get_isa() or "SSE" in fasm.get_isa()

    test_dims = [1, 2, 3, 4, 7, 8, 15, 16, 24, 32, 64, 77, 128, 384, 768]
    for dim in test_dims:
        a = [(float(i % 10) * 0.1) for i in range(dim)]
        b = [(float((i + 3) % 10) * 0.1) for i in range(dim)]

        ref_dot = sum(x * y for x, y in zip(a, b))
        ref_l2 = sum((x - y) ** 2 for x, y in zip(a, b))
        ref_norm_sq = sum(x * x for x in a)

        fasm_dot = fasm.dot(a, b, dim)
        fasm_l2 = fasm.l2_sq(a, b, dim)
        fasm_nsq = fasm.norm_sq(a, dim)

        assert abs(ref_dot - fasm_dot) < 1e-4, f"Dot product mismatch at dim={dim}: {ref_dot} vs {fasm_dot}"
        assert abs(ref_l2 - fasm_l2) < 1e-4, f"L2 sq mismatch at dim={dim}: {ref_l2} vs {fasm_l2}"
        assert abs(ref_norm_sq - fasm_nsq) < 1e-4, f"Norm sq mismatch at dim={dim}: {ref_norm_sq} vs {fasm_nsq}"


def test_fasm_index_workflow():
    """Tests FasmIndex end-to-end (add, batch_search, metadata, save/load)."""
    ensure_binaries_built()
    import nanovector

    dim = 64
    idx = nanovector.Index(dim=dim, metric="cosine", backend="fasm")
    assert len(idx) == 0

    # Add items
    v1 = [1.0 if i == 0 else 0.0 for i in range(dim)]
    v2 = [1.0 if i == 1 else 0.0 for i in range(dim)]
    v3 = [0.70710678 if i in (0, 1) else 0.0 for i in range(dim)]

    idx.add("x_axis", v1, metadata={"axis": "x", "weight": 10})
    idx.add("y_axis", v2, metadata={"axis": "y", "weight": 20})
    idx.add("xy_diag", v3, metadata={"axis": "diag", "weight": 30})
    assert len(idx) == 3

    # Query with v1
    matches = idx.search(v1, top_k=3)
    assert len(matches) == 3
    assert matches[0].id == "x_axis"
    assert abs(matches[0].score - 1.0) < 1e-4
    assert matches[1].id == "xy_diag"
    assert abs(matches[1].score - 0.7071) < 1e-3
    assert matches[2].id == "y_axis"
    assert abs(matches[2].score - 0.0) < 1e-4

    # Filtered query
    filtered = idx.search(v1, top_k=2, filter={"weight": {"$gte": 20}})
    assert len(filtered) == 2
    ids = [m.id for m in filtered]
    assert "xy_diag" in ids
    assert "y_axis" in ids

    # Save and Load
    tmp_path = str(Path(ASM_DIR) / "test_temp_fasm.nvec")
    try:
        idx.save(tmp_path)
        loaded = nanovector.Index.load(tmp_path, backend="fasm")
        assert len(loaded) == 3
        l_matches = loaded.search(v1, top_k=1)
        assert l_matches[0].id == "x_axis"
        assert abs(l_matches[0].score - 1.0) < 1e-4
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class TestFasm(unittest.TestCase):
    """Standard unittest wrapper for non-pytest runners (e.g. CPython 3.16t No-GIL)."""

    def test_exe64(self):
        test_fasm64_standalone_exe()

    def test_exe32(self):
        test_fasm32_standalone_exe()

    def test_metrics(self):
        test_fasm_ctypes_vector_metrics()

    def test_index(self):
        test_fasm_index_workflow()


if __name__ == "__main__":
    unittest.main()
