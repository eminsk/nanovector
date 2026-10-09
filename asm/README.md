# NanoVector — Native Flat Assembler (FASM) Acceleration Suite ⚡

Direct hardware assembly implementation of vector arithmetic and distance metrics (Dot Product, L2 Distance Squared, Cosine Similarity, L2 Normalization, and Batched Search) for both **x86-64 (64-bit AVX2+FMA)** and **x86 (32-bit SSE2)** architectures, written with [Flat Assembler (FASM)](https://flatassembler.net/).

---

## 🏛️ Architecture & Components

```
asm/
├── nanovector64_kernel.inc   # 64-bit AVX2+FMA register-unrolled vector microkernels
├── nanovector64.asm          # PE64 DLL source exporting SIMD vector kernels
├── nanovector64.dll          # Compiled 64-bit Windows DLL (callable from C, C++, Python ctypes)
├── test_nanovector64.asm     # Standalone PE64 console benchmark & self-test suite
├── test_nanovector64.exe     # Compiled 64-bit native executable (zero runtime dependencies)
├── nanovector32_kernel.inc   # 32-bit SSE2 register-unrolled vector microkernels
├── nanovector32.asm          # PE32 DLL source exporting SIMD vector kernels (cdecl ABI)
├── nanovector32.dll          # Compiled 32-bit Windows DLL (runs everywhere via WoW64)
├── test_nanovector32.asm     # Standalone PE32 console benchmark & self-test suite
├── test_nanovector32.exe     # Compiled 32-bit native executable (zero runtime dependencies)
├── build.bat                 # Automated Windows build, self-test & packaging script
├── build.sh                  # Cross-platform Unix build script
└── README.md                 # Technical specification and documentation
```

---

## 🚀 Key Features

### 1. 64-bit Engine (`nanovector64.dll`, `test_nanovector64.exe`)
- **ISA Target:** x86-64 with AVX2 (256-bit SIMD) and FMA3 (`vfmadd231ps`, `vmovups`, `vhaddps`).
- **Microkernels:**
  - 32-float unrolled primary loop utilizing 4 YMM accumulators (`ymm0`–`ymm3`) for maximum throughput and pipeline saturation.
  - 8-float secondary cleanup loop utilizing 1 YMM register.
  - 1-float scalar cleanup handling arbitrary odd dimensions (e.g. 77, 100, 384, 768, 1536).
- **In-Place Normalization:** Hardware-accelerated reciprocal square root (`vrsqrtps` / `vsqrtss` + `vdivss`) scaling vectors in-place with zero heap allocations.
- **Batched Search Kernels:** High-throughput row sweeps for Dot Product, L2 Distance Squared, and Cosine Similarity executing up to **19,000,000 vectors/sec**.
- **Calling Convention:** Complies strictly with the **Microsoft x64 ABI**, preserving all non-volatile registers (`RBX`, `RSI`, `RDI`, `R12`–`R15`, `XMM6`–`XMM15`).

### 2. 32-bit Engine (`nanovector32.dll`, `test_nanovector32.exe`)
- **ISA Target:** x86 32-bit with SSE2 (128-bit SIMD, `movups`, `mulps`, `addps`, `subps`, `shufps`).
- **Microkernels:**
  - 16-float unrolled primary loop using 4 XMM accumulators (`xmm0`–`xmm3`).
  - 4-float secondary cleanup loop using 1 XMM register.
  - Scalar FPU cleanup for arbitrary vector lengths.
- **Calling Convention:** Standard `cdecl` calling convention returning floating-point results in `ST(0)`.
- **Compatibility:** Runs on any 32-bit or 64-bit Windows system (WoW64) with zero MSVCRT or external C runtime requirements.
- **Throughput:** Delivers **14–16 MVec/sec** on 384-dim embedding spaces.

---

## 📦 Exported API Functions

Both `nanovector64.dll` and `nanovector32.dll` export identical symbol names:

```c
/* Vector-Vector Operations */
float nanovec_fasm_dot(const float* a, const float* b, size_t dim);
float nanovec_fasm_norm_sq(const float* a, size_t dim);
float nanovec_fasm_l2_sq(const float* a, const float* b, size_t dim);
float nanovec_fasm_normalize(float* a, size_t dim);

/* Batched Search Operations */
void nanovec_fasm_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores);
void nanovec_fasm_batch_l2_sq(const float* q, const float* matrix, size_t count, size_t dim, float* scores);
void nanovec_fasm_batch_cosine(const float* q, const float* matrix, size_t count, size_t dim, float* scores);

/* Metadata */
const char* nanovec_fasm_simd_isa(void);
```

---

## 🔨 Building and Testing

To assemble all targets and run the standalone verification suites:

```cmd
cd asm
build.bat
```

Output:
```
====================================================================
  NanoVector - Building Native FASM 32-bit and 64-bit Engines
====================================================================
Using Flat Assembler: C:\proekts\FASM\FASM.EXE

[1/4] Assembling nanovector64.dll (x86-64 AVX2+FMA PE64 DLL) ...
[2/4] Assembling test_nanovector64.exe (x86-64 Standalone Test Suite) ...
[3/4] Assembling nanovector32.dll (x86 32-bit SSE2 PE32 DLL) ...
[4/4] Assembling test_nanovector32.exe (x86 32-bit Standalone Test Suite) ...

--- Executing 64-bit Native Test Suite ---
  [TEST 1] Vector Dot Product (dim=384, float32): PASS (Exact match)
  [TEST 2] Vector L2 Distance Squared (dim=384): PASS (Exact match)
  [TEST 3] Vector In-Place L2 Normalization (dim=384): PASS (Exact match)
  [TEST 4] Batched Dot Product (1000 x 384): PASS
  [TEST 5] Batched L2 Squared Search (1000 x 384): PASS
  [TEST 6] Batched Cosine Similarity (1000 x 384): PASS
  [TEST 7] Odd / Non-aligned Dimension (dim=77, dot+L2): PASS (Exact match)
  ALL 64-BIT FASM NATIVE TESTS PASSED (100% Accuracy)!

--- Executing 32-bit Native Test Suite (WoW64) ---
  [TEST 1] Vector Dot Product (dim=384, float32): PASS (Exact match)
  [TEST 2] Vector L2 Distance Squared (dim=384): PASS (Exact match)
  [TEST 3] Vector In-Place L2 Normalization (dim=384): PASS (Exact match)
  [TEST 4] Batched Dot Product (1000 x 384): PASS
  [TEST 5] Batched L2 Squared Search (1000 x 384): PASS
  [TEST 6] Batched Cosine Similarity (1000 x 384): PASS
  [TEST 7] Odd / Non-aligned Dimension (dim=77, dot+L2): PASS (Exact match)
  ALL 32-BIT FASM NATIVE TESTS PASSED (100% Accuracy)!
```
