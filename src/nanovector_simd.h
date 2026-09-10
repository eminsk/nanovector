/*
 * NanoVector SIMD Interface Header
 * Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
 * MIT License
 */

#ifndef NANOVECTOR_SIMD_H
#define NANOVECTOR_SIMD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single vector operations */
float nanovec_simd_dot(const float* a, const float* b, size_t dim);
float nanovec_simd_norm_sq(const float* a, size_t dim);
float nanovec_simd_l2_sq(const float* a, const float* b, size_t dim);
void nanovec_simd_normalize(float* a, size_t dim);

/* Batch vector operations: scores array must have at least 'count' floats */
void nanovec_simd_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores);
void nanovec_simd_batch_l2_sq(const float* q, const float* matrix, size_t count, size_t dim, float* scores);
void nanovec_simd_batch_cosine(const float* q, float q_norm, const float* matrix, const float* norms, size_t count, size_t dim, float* scores);

/* Scalar fallbacks */
float nanovec_scalar_dot(const float* a, const float* b, size_t dim);
float nanovec_scalar_norm_sq(const float* a, size_t dim);
float nanovec_scalar_l2_sq(const float* a, const float* b, size_t dim);

#ifdef __cplusplus
}
#endif

#endif /* NANOVECTOR_SIMD_H */
