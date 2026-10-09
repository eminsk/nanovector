#!/usr/bin/env bash
# =============================================================================
# NanoVector — Cross-Platform Native Library Builder for Linux & macOS
# Compiles hardware-accelerated SIMD vector search kernels (AVX2+FMA / ARM NEON)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"
OUT_DIR="${ROOT_DIR}/python/nanovector"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "====================================================================="
echo "  NanoVector - Native Library Builder"
echo "  Detected OS:   ${OS}"
echo "  Architecture:  ${ARCH}"
echo "====================================================================="

mkdir -p "${OUT_DIR}"

if [ "${OS}" = "Darwin" ]; then
    # macOS (Apple Silicon M1/M2/M3/M4 or Intel x86_64)
    OUT_LIB="${OUT_DIR}/libnanovector.dylib"
    echo "Compiling for macOS using Apple Clang..."
    if [ "${ARCH}" = "arm64" ]; then
        clang -O3 -shared -fPIC -ffast-math -DBUILDING_NANOVECTOR \
            -arch arm64 "${SRC_DIR}/nanovector.c" "${SRC_DIR}/nanovector_neon.c" \
            -I"${SRC_DIR}" -o "${OUT_LIB}"
    elif [ "${ARCH}" = "x86_64" ]; then
        clang -O3 -shared -fPIC -mavx2 -mfma -ffast-math -DBUILDING_NANOVECTOR \
            -arch x86_64 "${SRC_DIR}/nanovector.c" "${SRC_DIR}/nanovector_avx2.c" \
            -I"${SRC_DIR}" -o "${OUT_LIB}"
    else
        clang -O3 -shared -fPIC -ffast-math -DBUILDING_NANOVECTOR \
            "${SRC_DIR}/nanovector.c" -I"${SRC_DIR}" -o "${OUT_LIB}"
    fi
    echo "[OK] Built macOS dynamic library: ${OUT_LIB}"

elif [ "${OS}" = "Linux" ]; then
    # Linux (x86_64 or aarch64)
    OUT_LIB="${OUT_DIR}/libnanovector.so"
    echo "Compiling for Linux using GCC/Clang..."
    CC=${CC:-gcc}
    if [ "${ARCH}" = "x86_64" ]; then
        ${CC} -O3 -shared -fPIC -mavx2 -mfma -ffast-math -DBUILDING_NANOVECTOR \
            "${SRC_DIR}/nanovector.c" "${SRC_DIR}/nanovector_avx2.c" \
            -I"${SRC_DIR}" -o "${OUT_LIB}" -lm
    elif [ "${ARCH}" = "aarch64" ]; then
        ${CC} -O3 -shared -fPIC -ffast-math -DBUILDING_NANOVECTOR \
            "${SRC_DIR}/nanovector.c" "${SRC_DIR}/nanovector_neon.c" \
            -I"${SRC_DIR}" -o "${OUT_LIB}" -lm
    else
        ${CC} -O3 -shared -fPIC -ffast-math -DBUILDING_NANOVECTOR \
            "${SRC_DIR}/nanovector.c" -I"${SRC_DIR}" -o "${OUT_LIB}" -lm
    fi
    echo "[OK] Built Linux shared object: ${OUT_LIB}"

else
    echo "[ERROR] Unsupported operating system: ${OS}"
    exit 1
fi

echo "====================================================================="
echo "  Build Complete! NanoVector native acceleration is ready."
echo "====================================================================="
