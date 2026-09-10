/*
 * NanoVector C Core Unit Tests
 * Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
 * MIT License
 */

#include "nanovector.h"
#include "nanovector_simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define EPSILON 1e-4f

static void test_simd_vs_scalar(void) {
    printf("[TEST] SIMD vs Scalar accuracy... ");
    size_t dims[] = {4, 7, 16, 32, 64, 128, 384, 768, 1536};
    for (size_t d = 0; d < sizeof(dims)/sizeof(dims[0]); ++d) {
        size_t dim = dims[d];
        float* a = (float*)malloc(dim * sizeof(float));
        float* b = (float*)malloc(dim * sizeof(float));
        for (size_t i = 0; i < dim; ++i) {
            a[i] = (float)rand() / (float)RAND_MAX;
            b[i] = (float)rand() / (float)RAND_MAX;
        }

        float dot_scalar = nanovec_scalar_dot(a, b, dim);
        float dot_simd = nanovec_simd_dot(a, b, dim);
        assert(fabsf(dot_scalar - dot_simd) < 1e-3f);

        float norm_scalar = nanovec_scalar_norm_sq(a, dim);
        float norm_simd = nanovec_simd_norm_sq(a, dim);
        assert(fabsf(norm_scalar - norm_simd) < 1e-3f);

        float l2_scalar = nanovec_scalar_l2_sq(a, b, dim);
        float l2_simd = nanovec_simd_l2_sq(a, b, dim);
        assert(fabsf(l2_scalar - l2_simd) < 1e-3f);

        free(a);
        free(b);
    }
    printf("PASSED (%s)\n", nanovector_simd_backend());
}

static void test_index_crud_and_search(void) {
    printf("[TEST] Index CRUD, Top-K Search and Metrics... ");
    uint32_t dim = 128;
    nanovector_index_t* idx = nanovector_create(dim, NANOVEC_METRIC_COSINE, 0);
    assert(idx != NULL);

    /* Insert 100 random vectors */
    for (int i = 0; i < 100; ++i) {
        float vec[128];
        for (int j = 0; j < 128; ++j) {
            vec[j] = (float)(i + 1) * 0.01f + (float)j * 0.001f;
        }
        char id_buf[32];
        char meta_buf[64];
        snprintf(id_buf, sizeof(id_buf), "doc_%03d", i);
        snprintf(meta_buf, sizeof(meta_buf), "{\"index\": %d}", i);
        int rc = nanovector_add(idx, id_buf, vec, meta_buf);
        assert(rc == 0);
    }
    assert(idx->count == 100);

    /* Search for doc_050 */
    float q[128];
    for (int j = 0; j < 128; ++j) {
        q[j] = (float)(50 + 1) * 0.01f + (float)j * 0.001f;
    }

    nanovector_match_t results[5];
    size_t found = nanovector_search(idx, q, 5, results);
    assert(found == 5);

    /* Top match must be doc_050 with cosine similarity ~ 1.0 */
    assert(strcmp(results[0].id, "doc_050") == 0);
    assert(fabsf(results[0].score - 1.0f) < 1e-4f);
    assert(results[0].metadata != NULL);

    /* Order must be descending */
    for (size_t i = 1; i < found; ++i) {
        assert(results[i-1].score >= results[i].score);
    }

    nanovector_free(idx);
    printf("PASSED\n");
}

static void test_persistence(void) {
    printf("[TEST] Binary Persistence (.nvec)... ");
    const char* filename = "test_memory.nvec";
    uint32_t dim = 64;
    nanovector_index_t* idx = nanovector_create(dim, NANOVEC_METRIC_L2, 0);

    for (int i = 0; i < 25; ++i) {
        float v[64];
        for (int j = 0; j < 64; ++j) v[j] = (float)(i * 10 + j);
        char id[16];
        snprintf(id, sizeof(id), "item_%d", i);
        nanovector_add(idx, id, v, "{\"tag\": \"persistent\"}");
    }

    int rc = nanovector_save(idx, filename);
    assert(rc == 0);
    nanovector_free(idx);

    /* Load from disk */
    nanovector_index_t* loaded = nanovector_load(filename);
    assert(loaded != NULL);
    assert(loaded->dim == 64);
    assert(loaded->count == 25);
    assert(loaded->metric == NANOVEC_METRIC_L2);

    /* Search in loaded index */
    float q[64];
    for (int j = 0; j < 64; ++j) q[j] = (float)(10 * 10 + j); /* exact item_10 */

    nanovector_match_t matches[3];
    size_t k = nanovector_search(loaded, q, 3, matches);
    assert(k == 3);
    assert(strcmp(matches[0].id, "item_10") == 0);
    assert(matches[0].score < 1e-4f); /* L2 squared distance == 0 */

    nanovector_free(loaded);
    remove(filename);
    printf("PASSED\n");
}

int main(void) {
    printf("==================================================\n");
    printf(" NanoVector C Core Test Suite v%s\n", nanovector_version());
    printf(" Backend: %s\n", nanovector_simd_backend());
    printf("==================================================\n");

    test_simd_vs_scalar();
    test_index_crud_and_search();
    test_persistence();

    printf("\n>>> ALL C CORE TESTS PASSED SUCCESSFULLY! <<<\n");
    return 0;
}
