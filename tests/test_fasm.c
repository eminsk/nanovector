/*
 * NanoVector FASM Assembly Integration Test
 * Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
 * MIT License
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

extern float nanovec_fasm_dot(const float* a, const float* b, size_t dim);
extern float nanovec_fasm_l2_sq(const float* a, const float* b, size_t dim);
extern void nanovec_fasm_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores);

int main(void) {
    printf("[TEST] FASM x64 Kernel... ");
    size_t dim = 384;
    float* a = (float*)malloc(dim * sizeof(float));
    float* b = (float*)malloc(dim * sizeof(float));
    float ref_dot = 0.0f;
    float ref_l2 = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        a[i] = (float)(i % 10) * 0.1f;
        b[i] = (float)((i + 3) % 10) * 0.1f;
        ref_dot += a[i] * b[i];
        float diff = a[i] - b[i];
        ref_l2 += diff * diff;
    }

    float fasm_dot = nanovec_fasm_dot(a, b, dim);
    float fasm_l2 = nanovec_fasm_l2_sq(a, b, dim);
    printf("ref_dot = %f, fasm_dot = %f, ref_l2 = %f, fasm_l2 = %f\n", ref_dot, fasm_dot, ref_l2, fasm_l2);

    assert(fabsf(ref_dot - fasm_dot) < 1e-3f);
    assert(fabsf(ref_l2 - fasm_l2) < 1e-3f);

    /* Test batch */
    size_t count = 10;
    float* matrix = (float*)malloc(count * dim * sizeof(float));
    for (size_t r = 0; r < count; ++r) {
        for (size_t i = 0; i < dim; ++i) {
            matrix[r * dim + i] = (float)(r + 1) * b[i];
        }
    }

    float scores[10];
    nanovec_fasm_batch_dot(a, matrix, count, dim, scores);

    for (size_t r = 0; r < count; ++r) {
        float expected = (float)(r + 1) * ref_dot;
        assert(fabsf(scores[r] - expected) < 1e-2f);
    }

    free(a);
    free(b);
    free(matrix);

    printf("PASSED! (FASM x64 AVX2/FMA matches reference IEEE-754)\n");
    return 0;
}
