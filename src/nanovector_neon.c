/*
 * NanoVector ARM NEON Implementation
 * Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
 * MIT License
 */

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)

#include "nanovector_simd.h"
#include <arm_neon.h>
#include <math.h>

static inline float hsum128_ps(float32x4_t v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_f32(v);
#else
    float32x2_t low = vget_low_f32(v);
    float32x2_t high = vget_high_f32(v);
    float32x2_t sum = vadd_f32(low, high);
    return vget_lane_f32(vpadd_f32(sum, sum), 0);
#endif
}

float nanovec_neon_dot(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    /* 16 floats per iteration (4 x 4) */
    for (; i + 15 < dim; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        acc0 = vfmaq_f32(acc0, va0, vb0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        acc1 = vfmaq_f32(acc1, va1, vb1);

        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        acc2 = vfmaq_f32(acc2, va2, vb2);

        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        acc3 = vfmaq_f32(acc3, va3, vb3);
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    /* 4 floats per iteration */
    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc0 = vfmaq_f32(acc0, va, vb);
    }

    float sum = hsum128_ps(acc0);

    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

float nanovec_neon_norm_sq(const float* a, size_t dim) {
    size_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (; i + 15 < dim; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        acc0 = vfmaq_f32(acc0, va0, va0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        acc1 = vfmaq_f32(acc1, va1, va1);

        float32x4_t va2 = vld1q_f32(a + i + 8);
        acc2 = vfmaq_f32(acc2, va2, va2);

        float32x4_t va3 = vld1q_f32(a + i + 12);
        acc3 = vfmaq_f32(acc3, va3, va3);
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        acc0 = vfmaq_f32(acc0, va, va);
    }

    float sum = hsum128_ps(acc0);

    for (; i < dim; ++i) {
        sum += a[i] * a[i];
    }

    return sum;
}

float nanovec_neon_l2_sq(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (; i + 15 < dim; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t d0  = vsubq_f32(va0, vb0);
        acc0 = vfmaq_f32(acc0, d0, d0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t d1  = vsubq_f32(va1, vb1);
        acc1 = vfmaq_f32(acc1, d1, d1);

        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t d2  = vsubq_f32(va2, vb2);
        acc2 = vfmaq_f32(acc2, d2, d2);

        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        float32x4_t d3  = vsubq_f32(va3, vb3);
        acc3 = vfmaq_f32(acc3, d3, d3);
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t d  = vsubq_f32(va, vb);
        acc0 = vfmaq_f32(acc0, d, d);
    }

    float sum = hsum128_ps(acc0);

    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

void nanovec_neon_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores) {
    for (size_t row = 0; row < count; ++row) {
        scores[row] = nanovec_neon_dot(q, matrix + row * dim, dim);
    }
}

void nanovec_neon_batch_l2_sq(const float* q, const float* matrix, size_t count, size_t dim, float* scores) {
    for (size_t row = 0; row < count; ++row) {
        scores[row] = nanovec_neon_l2_sq(q, matrix + row * dim, dim);
    }
}

void nanovec_neon_batch_cosine(const float* q, float q_norm, const float* matrix, const float* norms, size_t count, size_t dim, float* scores) {
    if (q_norm <= 0.0f) {
        for (size_t row = 0; row < count; ++row) scores[row] = 0.0f;
        return;
    }
    float inv_q_norm = 1.0f / q_norm;
    for (size_t row = 0; row < count; ++row) {
        float dot = nanovec_neon_dot(q, matrix + row * dim, dim);
        float row_norm = (norms != NULL) ? norms[row] : sqrtf(nanovec_neon_norm_sq(matrix + row * dim, dim));
        if (row_norm > 0.0f) {
            scores[row] = (dot * inv_q_norm) / row_norm;
        } else {
            scores[row] = 0.0f;
        }
    }
}

#endif
