/*
 * NanoVector: Minimalist Bare-Metal Vector Search & Episodic Memory Engine
 * Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
 * MIT License
 */

#ifndef NANOVECTOR_H
#define NANOVECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define NANOVECTOR_VERSION_MAJOR 0
#define NANOVECTOR_VERSION_MINOR 1
#define NANOVECTOR_VERSION_PATCH 3
#define NANOVECTOR_VERSION_STRING "0.1.3"

/* Supported distance/similarity metrics */
typedef enum {
    NANOVEC_METRIC_COSINE = 0, /* Cosine similarity (higher is more similar, range [-1, 1]) */
    NANOVEC_METRIC_DOT    = 1, /* Inner / Dot product (higher is more similar) */
    NANOVEC_METRIC_L2     = 2  /* Squared Euclidean distance (lower is closer) */
} nanovector_metric_t;

/* Single search match */
typedef struct {
    char* id;          /* Document/chunk identifier (borrowed pointer) */
    float score;       /* Similarity score or distance */
    char* metadata;    /* JSON metadata string or NULL */
} nanovector_match_t;

/* In-memory vector index */
typedef struct {
    uint32_t dim;               /* Vector dimensionality (e.g. 384, 768, 1536) */
    uint64_t count;             /* Number of indexed vectors */
    uint64_t capacity;          /* Allocated capacity */
    nanovector_metric_t metric; /* Metric used for scoring */
    int normalize;              /* Automatically L2-normalize vectors on addition/query */
    float* vectors;             /* Contiguous aligned vector buffer (count * dim floats) */
    float* norms;               /* Precomputed vector L2 norms (for fast cosine) */
    char** ids;                 /* Document / message IDs */
    char** metadatas;           /* JSON metadata strings */
} nanovector_index_t;

/*
 * Lifecycle Management
 */
nanovector_index_t* nanovector_create(uint32_t dim, nanovector_metric_t metric, int normalize);
void nanovector_free(nanovector_index_t* index);
int nanovector_reserve(nanovector_index_t* index, uint64_t new_capacity);

/*
 * Adding Vectors
 */
int nanovector_add(nanovector_index_t* index, const char* id, const float* vector, const char* metadata);
int nanovector_add_batch(nanovector_index_t* index, const char** ids, const float* vectors, size_t n, const char** metadatas);

/*
 * Querying / Search
 * Returns the number of matches found (up to top_k).
 * Caller provides pre-allocated results array of at least top_k elements.
 */
size_t nanovector_search(const nanovector_index_t* index, const float* query, size_t top_k, nanovector_match_t* results);

/*
 * Single-File Persistence (.nvec)
 */
int nanovector_save(const nanovector_index_t* index, const char* filepath);
nanovector_index_t* nanovector_load(const char* filepath);

/*
 * Diagnostics & Metadata
 */
const char* nanovector_version(void);
const char* nanovector_simd_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* NANOVECTOR_H */
