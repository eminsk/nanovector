/*
 * NanoVector: Minimalist Bare-Metal Vector Search & Episodic Memory Engine
 * Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
 * MIT License
 */

#include "nanovector.h"
#include "nanovector_simd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <malloc.h>
#define NANOVEC_ALIGNED_ALLOC(size, align) _aligned_malloc((size), (align))
#define NANOVEC_ALIGNED_FREE(ptr)          _aligned_free(ptr)
#elif defined(__posix__) || defined(__linux__) || defined(__APPLE__)
#include <stdlib.h>
static inline void* nanovec_posix_aligned_alloc(size_t size, size_t align) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) return NULL;
    return ptr;
}
#define NANOVEC_ALIGNED_ALLOC(size, align) nanovec_posix_aligned_alloc((size), (align))
#define NANOVEC_ALIGNED_FREE(ptr)          free(ptr)
#else
#define NANOVEC_ALIGNED_ALLOC(size, align) malloc(size)
#define NANOVEC_ALIGNED_FREE(ptr)          free(ptr)
#endif

/* -------------------------------------------------------------------------
 * CPU Dispatch / SIMD Backend Detection
 * ------------------------------------------------------------------------- */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
extern float nanovec_avx2_dot(const float* a, const float* b, size_t dim);
extern float nanovec_avx2_norm_sq(const float* a, size_t dim);
extern float nanovec_avx2_l2_sq(const float* a, const float* b, size_t dim);
extern void nanovec_avx2_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores);
extern void nanovec_avx2_batch_l2_sq(const float* q, const float* matrix, size_t count, size_t dim, float* scores);
extern void nanovec_avx2_batch_cosine(const float* q, float q_norm, const float* matrix, const float* norms, size_t count, size_t dim, float* scores);
#endif

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
extern float nanovec_neon_dot(const float* a, const float* b, size_t dim);
extern float nanovec_neon_norm_sq(const float* a, size_t dim);
extern float nanovec_neon_l2_sq(const float* a, const float* b, size_t dim);
extern void nanovec_neon_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores);
extern void nanovec_neon_batch_l2_sq(const float* q, const float* matrix, size_t count, size_t dim, float* scores);
extern void nanovec_neon_batch_cosine(const float* q, float q_norm, const float* matrix, const float* norms, size_t count, size_t dim, float* scores);
#endif

/* Scalar implementations */
float nanovec_scalar_dot(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float nanovec_scalar_norm_sq(const float* a, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * a[i];
    }
    return sum;
}

float nanovec_scalar_l2_sq(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

void nanovec_scalar_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores) {
    for (size_t r = 0; r < count; ++r) {
        scores[r] = nanovec_scalar_dot(q, matrix + r * dim, dim);
    }
}

void nanovec_scalar_batch_l2_sq(const float* q, const float* matrix, size_t count, size_t dim, float* scores) {
    for (size_t r = 0; r < count; ++r) {
        scores[r] = nanovec_scalar_l2_sq(q, matrix + r * dim, dim);
    }
}

void nanovec_scalar_batch_cosine(const float* q, float q_norm, const float* matrix, const float* norms, size_t count, size_t dim, float* scores) {
    if (q_norm <= 0.0f) {
        for (size_t r = 0; r < count; ++r) scores[r] = 0.0f;
        return;
    }
    float inv_q_norm = 1.0f / q_norm;
    for (size_t r = 0; r < count; ++r) {
        float dot = nanovec_scalar_dot(q, matrix + r * dim, dim);
        float r_norm = (norms != NULL) ? norms[r] : sqrtf(nanovec_scalar_norm_sq(matrix + r * dim, dim));
        if (r_norm > 0.0f) {
            scores[r] = (dot * inv_q_norm) / r_norm;
        } else {
            scores[r] = 0.0f;
        }
    }
}

const char* nanovector_version(void) {
    return NANOVECTOR_VERSION_STRING;
}

const char* nanovector_simd_backend(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return "AVX2+FMA (x86_64)";
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    return "ARM NEON (arm64)";
#else
    return "Scalar (Portable C)";
#endif
}

float nanovec_simd_dot(const float* a, const float* b, size_t dim) {
#if defined(__x86_64__) || defined(_M_X64)
    return nanovec_avx2_dot(a, b, dim);
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    return nanovec_neon_dot(a, b, dim);
#else
    return nanovec_scalar_dot(a, b, dim);
#endif
}

float nanovec_simd_norm_sq(const float* a, size_t dim) {
#if defined(__x86_64__) || defined(_M_X64)
    return nanovec_avx2_norm_sq(a, dim);
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    return nanovec_neon_norm_sq(a, dim);
#else
    return nanovec_scalar_norm_sq(a, dim);
#endif
}

float nanovec_simd_l2_sq(const float* a, const float* b, size_t dim) {
#if defined(__x86_64__) || defined(_M_X64)
    return nanovec_avx2_l2_sq(a, b, dim);
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    return nanovec_neon_l2_sq(a, b, dim);
#else
    return nanovec_scalar_l2_sq(a, b, dim);
#endif
}

void nanovec_simd_normalize(float* a, size_t dim) {
    float nsq = nanovec_simd_norm_sq(a, dim);
    if (nsq > 0.0f) {
        float inv_norm = 1.0f / sqrtf(nsq);
        for (size_t i = 0; i < dim; ++i) {
            a[i] *= inv_norm;
        }
    }
}

void nanovec_simd_batch_dot(const float* q, const float* matrix, size_t count, size_t dim, float* scores) {
#if defined(__x86_64__) || defined(_M_X64)
    nanovec_avx2_batch_dot(q, matrix, count, dim, scores);
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    nanovec_neon_batch_dot(q, matrix, count, dim, scores);
#else
    nanovec_scalar_batch_dot(q, matrix, count, dim, scores);
#endif
}

void nanovec_simd_batch_l2_sq(const float* q, const float* matrix, size_t count, size_t dim, float* scores) {
#if defined(__x86_64__) || defined(_M_X64)
    nanovec_avx2_batch_l2_sq(q, matrix, count, dim, scores);
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    nanovec_neon_batch_l2_sq(q, matrix, count, dim, scores);
#else
    nanovec_scalar_batch_l2_sq(q, matrix, count, dim, scores);
#endif
}

void nanovec_simd_batch_cosine(const float* q, float q_norm, const float* matrix, const float* norms, size_t count, size_t dim, float* scores) {
#if defined(__x86_64__) || defined(_M_X64)
    nanovec_avx2_batch_cosine(q, q_norm, matrix, norms, count, dim, scores);
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    nanovec_neon_batch_cosine(q, q_norm, matrix, norms, count, dim, scores);
#else
    nanovec_scalar_batch_cosine(q, q_norm, matrix, norms, count, dim, scores);
#endif
}


/* -------------------------------------------------------------------------
 * Index Lifecycle & Memory Management
 * ------------------------------------------------------------------------- */
nanovector_index_t* nanovector_create(uint32_t dim, nanovector_metric_t metric, int normalize) {
    if (dim == 0) return NULL;
    nanovector_index_t* idx = (nanovector_index_t*)calloc(1, sizeof(nanovector_index_t));
    if (!idx) return NULL;

    idx->dim = dim;
    idx->metric = metric;
    idx->normalize = normalize;
    idx->count = 0;
    idx->capacity = 0;
    idx->vectors = NULL;
    idx->norms = NULL;
    idx->ids = NULL;
    idx->metadatas = NULL;

    return idx;
}

int nanovector_reserve(nanovector_index_t* index, uint64_t new_capacity) {
    if (!index || new_capacity <= index->capacity) return 0;

    size_t vector_bytes = (size_t)new_capacity * index->dim * sizeof(float);
    float* new_vectors = (float*)NANOVEC_ALIGNED_ALLOC(vector_bytes, 32);
    if (!new_vectors) return -1;

    if (index->vectors && index->count > 0) {
        memcpy(new_vectors, index->vectors, (size_t)index->count * index->dim * sizeof(float));
        NANOVEC_ALIGNED_FREE(index->vectors);
    }
    index->vectors = new_vectors;

    float* new_norms = (float*)realloc(index->norms, (size_t)new_capacity * sizeof(float));
    if (!new_norms) return -1;
    index->norms = new_norms;

    char** new_ids = (char**)realloc(index->ids, (size_t)new_capacity * sizeof(char*));
    if (!new_ids) return -1;
    index->ids = new_ids;

    char** new_metas = (char**)realloc(index->metadatas, (size_t)new_capacity * sizeof(char*));
    if (!new_metas) return -1;
    index->metadatas = new_metas;

    index->capacity = new_capacity;
    return 0;
}

void nanovector_free(nanovector_index_t* index) {
    if (!index) return;
    if (index->vectors) {
        NANOVEC_ALIGNED_FREE(index->vectors);
    }
    if (index->norms) {
        free(index->norms);
    }
    if (index->ids) {
        for (uint64_t i = 0; i < index->count; ++i) {
            if (index->ids[i]) free(index->ids[i]);
        }
        free(index->ids);
    }
    if (index->metadatas) {
        for (uint64_t i = 0; i < index->count; ++i) {
            if (index->metadatas[i]) free(index->metadatas[i]);
        }
        free(index->metadatas);
    }
    free(index);
}

static char* nanovec_strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char* copy = (char*)malloc(len + 1);
    if (copy) {
        memcpy(copy, s, len + 1);
    }
    return copy;
}

int nanovector_add(nanovector_index_t* index, const char* id, const float* vector, const char* metadata) {
    if (!index || !vector) return -1;

    if (index->count >= index->capacity) {
        uint64_t new_cap = (index->capacity == 0) ? 64 : (index->capacity * 2);
        if (nanovector_reserve(index, new_cap) != 0) return -1;
    }

    uint64_t pos = index->count;
    float* dst = index->vectors + pos * index->dim;
    memcpy(dst, vector, index->dim * sizeof(float));

    if (index->normalize) {
        nanovec_simd_normalize(dst, index->dim);
        index->norms[pos] = 1.0f;
    } else {
        index->norms[pos] = sqrtf(nanovec_simd_norm_sq(dst, index->dim));
    }

    index->ids[pos] = id ? nanovec_strdup(id) : NULL;
    index->metadatas[pos] = metadata ? nanovec_strdup(metadata) : NULL;

    index->count++;
    return 0;
}

int nanovector_add_batch(nanovector_index_t* index, const char** ids, const float* vectors, size_t n, const char** metadatas) {
    if (!index || !vectors || n == 0) return -1;

    uint64_t needed = index->count + n;
    if (needed > index->capacity) {
        uint64_t new_cap = index->capacity ? index->capacity : 64;
        while (new_cap < needed) new_cap *= 2;
        if (nanovector_reserve(index, new_cap) != 0) return -1;
    }

    for (size_t i = 0; i < n; ++i) {
        uint64_t pos = index->count + i;
        const float* src = vectors + i * index->dim;
        float* dst = index->vectors + pos * index->dim;
        memcpy(dst, src, index->dim * sizeof(float));

        if (index->normalize) {
            nanovec_simd_normalize(dst, index->dim);
            index->norms[pos] = 1.0f;
        } else {
            index->norms[pos] = sqrtf(nanovec_simd_norm_sq(dst, index->dim));
        }

        index->ids[pos] = (ids && ids[i]) ? nanovec_strdup(ids[i]) : NULL;
        index->metadatas[pos] = (metadatas && metadatas[i]) ? nanovec_strdup(metadatas[i]) : NULL;
    }

    index->count += n;
    return 0;
}


/* -------------------------------------------------------------------------
 * Top-K Heap Selection
 * ------------------------------------------------------------------------- */
typedef struct {
    uint64_t idx;
    float score;
} heap_node_t;

/* Min-Heap (used when higher score is better: Cosine, Dot) */
static inline void min_heap_sift_down(heap_node_t* heap, size_t k, size_t i) {
    while (2 * i + 1 < k) {
        size_t left = 2 * i + 1;
        size_t right = left + 1;
        size_t smallest = i;

        if (heap[left].score < heap[smallest].score) smallest = left;
        if (right < k && heap[right].score < heap[smallest].score) smallest = right;
        if (smallest == i) break;

        heap_node_t tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        i = smallest;
    }
}

/* Max-Heap (used when lower distance is better: L2) */
static inline void max_heap_sift_down(heap_node_t* heap, size_t k, size_t i) {
    while (2 * i + 1 < k) {
        size_t left = 2 * i + 1;
        size_t right = left + 1;
        size_t largest = i;

        if (heap[left].score > heap[largest].score) largest = left;
        if (right < k && heap[right].score > heap[largest].score) largest = right;
        if (largest == i) break;

        heap_node_t tmp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = tmp;
        i = largest;
    }
}

size_t nanovector_search(const nanovector_index_t* index, const float* query, size_t top_k, nanovector_match_t* results) {
    if (!index || !query || !results || top_k == 0 || index->count == 0) return 0;

    size_t k = (top_k < (size_t)index->count) ? top_k : (size_t)index->count;

    /* Compute scores for all vectors in index */
    float* scores = (float*)malloc((size_t)index->count * sizeof(float));
    if (!scores) return 0;

    /* Stack or heap temporary query for normalization */
    const float* eff_query = query;
    float norm_q = 0.0f;
    float* temp_q = NULL;

    if (index->normalize) {
        temp_q = (float*)malloc(index->dim * sizeof(float));
        if (temp_q) {
            memcpy(temp_q, query, index->dim * sizeof(float));
            nanovec_simd_normalize(temp_q, index->dim);
            eff_query = temp_q;
            norm_q = 1.0f;
        }
    } else {
        norm_q = sqrtf(nanovec_simd_norm_sq(query, index->dim));
    }

    if (index->metric == NANOVEC_METRIC_COSINE) {
        nanovec_simd_batch_cosine(eff_query, norm_q, index->vectors, index->norms, (size_t)index->count, index->dim, scores);
    } else if (index->metric == NANOVEC_METRIC_DOT) {
        nanovec_simd_batch_dot(eff_query, index->vectors, (size_t)index->count, index->dim, scores);
    } else { /* NANOVEC_METRIC_L2 */
        nanovec_simd_batch_l2_sq(eff_query, index->vectors, (size_t)index->count, index->dim, scores);
    }

    if (temp_q) free(temp_q);

    /* Allocate heap of size k */
    heap_node_t* heap = (heap_node_t*)malloc(k * sizeof(heap_node_t));
    if (!heap) {
        free(scores);
        return 0;
    }

    int is_higher_better = (index->metric != NANOVEC_METRIC_L2);

    /* Populate initial heap of k items */
    for (size_t i = 0; i < k; ++i) {
        heap[i].idx = i;
        heap[i].score = scores[i];
    }

    if (is_higher_better) {
        /* Min-heap: build in O(k) */
        for (int64_t i = (int64_t)(k / 2) - 1; i >= 0; --i) {
            min_heap_sift_down(heap, k, (size_t)i);
        }
        /* Scan remaining N - k items */
        for (size_t i = k; i < (size_t)index->count; ++i) {
            float s = scores[i];
            if (s > heap[0].score) {
                heap[0].idx = i;
                heap[0].score = s;
                min_heap_sift_down(heap, k, 0);
            }
        }
        /* Extract from heap into results (descending order) */
        for (size_t i = 0; i < k; ++i) {
            size_t last = k - 1 - i;
            heap_node_t best = heap[0];
            heap[0] = heap[last];
            min_heap_sift_down(heap, last, 0);

            results[last].id = index->ids[best.idx];
            results[last].score = best.score;
            results[last].metadata = index->metadatas[best.idx];
        }
    } else {
        /* Max-heap: build in O(k) */
        for (int64_t i = (int64_t)(k / 2) - 1; i >= 0; --i) {
            max_heap_sift_down(heap, k, (size_t)i);
        }
        /* Scan remaining N - k items */
        for (size_t i = k; i < (size_t)index->count; ++i) {
            float s = scores[i];
            if (s < heap[0].score) {
                heap[0].idx = i;
                heap[0].score = s;
                max_heap_sift_down(heap, k, 0);
            }
        }
        /* Extract from heap into results (ascending order of distance) */
        for (size_t i = 0; i < k; ++i) {
            size_t last = k - 1 - i;
            heap_node_t best = heap[0];
            heap[0] = heap[last];
            max_heap_sift_down(heap, last, 0);

            results[last].id = index->ids[best.idx];
            results[last].score = best.score;
            results[last].metadata = index->metadatas[best.idx];
        }
    }

    free(heap);
    free(scores);
    return k;
}


/* -------------------------------------------------------------------------
 * Single-File Persistence (.nvec)
 * Format:
 *   [Header: 64 bytes]
 *     Magic 'NVEC' (4 bytes)
 *     Version uint32 (4 bytes)
 *     Dim uint32 (4 bytes)
 *     Count uint64 (8 bytes)
 *     Metric uint32 (4 bytes)
 *     Normalize uint32 (4 bytes)
 *     Reserved (36 bytes)
 *   [Vectors Data: count * dim * 4 bytes]
 *   [Metadata Section: count entries]
 *     For each:
 *       id_len (uint16_t) + id_bytes
 *       meta_len (uint32_t) + meta_bytes
 * ------------------------------------------------------------------------- */
#pragma pack(push, 1)
typedef struct {
    char magic[4];          /* "NVEC" */
    uint32_t version;       /* 1 */
    uint32_t dim;
    uint64_t count;
    uint32_t metric;
    uint32_t normalize;
    uint8_t reserved[36];
} nvec_header_t;
#pragma pack(pop)

int nanovector_save(const nanovector_index_t* index, const char* filepath) {
    if (!index || !filepath) return -1;

    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    nvec_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic[0] = 'N'; hdr.magic[1] = 'V'; hdr.magic[2] = 'E'; hdr.magic[3] = 'C';
    hdr.version = 1;
    hdr.dim = index->dim;
    hdr.count = index->count;
    hdr.metric = (uint32_t)index->metric;
    hdr.normalize = (uint32_t)index->normalize;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    if (index->count > 0 && index->vectors) {
        size_t total_floats = (size_t)index->count * index->dim;
        if (fwrite(index->vectors, sizeof(float), total_floats, f) != total_floats) {
            fclose(f);
            return -1;
        }
    }

    for (uint64_t i = 0; i < index->count; ++i) {
        const char* id = index->ids[i];
        uint16_t id_len = id ? (uint16_t)strlen(id) : 0;
        fwrite(&id_len, sizeof(id_len), 1, f);
        if (id_len > 0) {
            fwrite(id, 1, id_len, f);
        }

        const char* meta = index->metadatas[i];
        uint32_t meta_len = meta ? (uint32_t)strlen(meta) : 0;
        fwrite(&meta_len, sizeof(meta_len), 1, f);
        if (meta_len > 0) {
            fwrite(meta, 1, meta_len, f);
        }
    }

    fclose(f);
    return 0;
}

nanovector_index_t* nanovector_load(const char* filepath) {
    if (!filepath) return NULL;

    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    nvec_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    if (hdr.magic[0] != 'N' || hdr.magic[1] != 'V' || hdr.magic[2] != 'E' || hdr.magic[3] != 'C') {
        fclose(f);
        return NULL;
    }
    if (hdr.version != 1) {
        fclose(f);
        return NULL;
    }

    nanovector_index_t* idx = nanovector_create(hdr.dim, (nanovector_metric_t)hdr.metric, (int)hdr.normalize);
    if (!idx) {
        fclose(f);
        return NULL;
    }

    if (nanovector_reserve(idx, hdr.count) != 0) {
        nanovector_free(idx);
        fclose(f);
        return NULL;
    }

    if (hdr.count > 0) {
        size_t total_floats = (size_t)hdr.count * hdr.dim;
        if (fread(idx->vectors, sizeof(float), total_floats, f) != total_floats) {
            nanovector_free(idx);
            fclose(f);
            return NULL;
        }

        for (uint64_t i = 0; i < hdr.count; ++i) {
            uint16_t id_len = 0;
            if (fread(&id_len, sizeof(id_len), 1, f) != 1) break;
            if (id_len > 0) {
                idx->ids[i] = (char*)malloc(id_len + 1);
                fread(idx->ids[i], 1, id_len, f);
                idx->ids[i][id_len] = '\0';
            } else {
                idx->ids[i] = NULL;
            }

            uint32_t meta_len = 0;
            if (fread(&meta_len, sizeof(meta_len), 1, f) != 1) break;
            if (meta_len > 0) {
                idx->metadatas[i] = (char*)malloc(meta_len + 1);
                fread(idx->metadatas[i], 1, meta_len, f);
                idx->metadatas[i][meta_len] = '\0';
            } else {
                idx->metadatas[i] = NULL;
            }

            const float* row = idx->vectors + i * idx->dim;
            idx->norms[i] = sqrtf(nanovec_simd_norm_sq(row, idx->dim));
        }
        idx->count = hdr.count;
    }

    fclose(f);
    return idx;
}
