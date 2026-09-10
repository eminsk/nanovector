/*
 * NanoVector AVX2 + FMA Implementation
 * Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
 * MIT License
 */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include "nanovector_simd.h"
#include <immintrin.h>
#include <math.h>

static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128  = _mm_add_ps(vlow, vhigh);
    __m128 shuf  = _mm_movehl_ps(v128, v128);
    __m128 sums  = _mm_add_ps(v128, shuf);
    shuf         = _mm_shuffle_ps(sums, sums, 1);
    sums         = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

float nanovec_avx2_dot(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    /* 32 floats per iteration (4 x 8) */
    for (; i + 31 < dim; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    /* 8 floats per iteration */
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }

    float sum = hsum256_ps(acc0);

    /* Scalar remainder */
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

float nanovec_avx2_norm_sq(const float* a, size_t dim) {
    size_t i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (; i + 31 < dim; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        acc0 = _mm256_fmadd_ps(va0, va0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        acc1 = _mm256_fmadd_ps(va1, va1, acc1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        acc2 = _mm256_fmadd_ps(va2, va2, acc2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        acc3 = _mm256_fmadd_ps(va3, va3, acc3);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        acc0 = _mm256_fmadd_ps(va, va, acc0);
    }

    float sum = hsum256_ps(acc0);

    for (; i < dim; ++i) {
        sum += a[i] * a[i];
    }

    return sum;
}

float nanovec_avx2_l2_sq(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (; i + 31 < dim; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 d0  = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 d1  = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 d2  = _mm256_sub_ps(va2, vb2);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        __m256 d3  = _mm256_sub_ps(va3, vb3);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }

    float sum = hsum256_ps(acc0);

    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

void nanovec_avx2_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores) {
    for (size_t row = 0; row < count; ++row) {
        scores[row] = nanovec_avx2_dot(q, matrix + row * dim, dim);
    }
}

void nanovec_avx2_batch_l2_sq(const float* q, const float* matrix, size_t count, size_t dim, float* scores) {
    for (size_t row = 0; row < count; ++row) {
        scores[row] = nanovec_avx2_l2_sq(q, matrix + row * dim, dim);
    }
}

void nanovec_avx2_batch_cosine(const float* q, float q_norm, const float* matrix, const float* norms, size_t count, size_t dim, float* scores) {
    if (q_norm <= 0.0f) {
        for (size_t row = 0; row < count; ++row) scores[row] = 0.0f;
        return;
    }
    float inv_q_norm = 1.0f / q_norm;
    for (size_t row = 0; row < count; ++row) {
        float dot = nanovec_avx2_dot(q, matrix + row * dim, dim);
        float row_norm = (norms != NULL) ? norms[row] : sqrtf(nanovec_avx2_norm_sq(matrix + row * dim, dim));
        if (row_norm > 0.0f) {
            scores[row] = (dot * inv_q_norm) / row_norm;
        } else {
            scores[row] = 0.0f;
        }
    }
}

#endif
