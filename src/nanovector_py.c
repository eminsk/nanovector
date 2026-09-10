/*
 * NanoVector: Python C-API Extension Module with Buffer Protocol & GIL Release
 * Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
 * MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "nanovector.h"

typedef struct {
    PyObject_HEAD
    nanovector_index_t* index;
} PyNanoVectorIndex;

static void Index_dealloc(PyNanoVectorIndex* self) {
    if (self->index) {
        nanovector_free(self->index);
        self->index = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int parse_metric_str(const char* metric_str, nanovector_metric_t* out_metric) {
    if (!metric_str || strcmp(metric_str, "cosine") == 0) {
        *out_metric = NANOVEC_METRIC_COSINE;
        return 0;
    } else if (strcmp(metric_str, "dot") == 0 || strcmp(metric_str, "ip") == 0) {
        *out_metric = NANOVEC_METRIC_DOT;
        return 0;
    } else if (strcmp(metric_str, "l2") == 0 || strcmp(metric_str, "euclidean") == 0) {
        *out_metric = NANOVEC_METRIC_L2;
        return 0;
    }
    return -1;
}

static int Index_init(PyNanoVectorIndex* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"dim", "metric", "normalize", NULL};
    unsigned int dim = 0;
    const char* metric_str = "cosine";
    int normalize = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "I|sp", kwlist, &dim, &metric_str, &normalize)) {
        return -1;
    }

    if (dim == 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension 'dim' must be greater than 0");
        return -1;
    }

    nanovector_metric_t metric;
    if (parse_metric_str(metric_str, &metric) != 0) {
        PyErr_Format(PyExc_ValueError, "Invalid metric '%s'. Supported metrics: 'cosine', 'dot'/'ip', 'l2'/'euclidean'", metric_str);
        return -1;
    }

    if (self->index) {
        nanovector_free(self->index);
        self->index = NULL;
    }

    self->index = nanovector_create(dim, metric, normalize);
    if (!self->index) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate NanoVector index");
        return -1;
    }

    return 0;
}

/* Helper to extract a contiguous float array from Python object (Buffer Protocol or List) */
static int extract_float_vector(PyObject* obj, uint32_t expected_dim, float** out_buf, int* need_free) {
    *need_free = 0;

    /* 1. Fast Path: Python Buffer Protocol (NumPy array, array('f'), etc.) */
    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_FULL_RO) == 0) {
        if ((uint32_t)(view.len / sizeof(float)) != expected_dim) {
            PyBuffer_Release(&view);
            PyErr_Format(PyExc_ValueError, "Vector dimension mismatch: expected %u floats, got %zd",
                         expected_dim, view.len / sizeof(float));
            return -1;
        }

        if (PyBuffer_IsContiguous(&view, 'C') && (view.itemsize == sizeof(float))) {
            /* Zero-Copy directly pointing to buffer */
            *out_buf = (float*)view.buf;
            *need_free = 0;
            PyBuffer_Release(&view);
            return 0;
        }

        /* Non-contiguous or format conversion needed */
        float* copy = (float*)malloc(expected_dim * sizeof(float));
        if (!copy) {
            PyBuffer_Release(&view);
            PyErr_NoMemory();
            return -1;
        }
        if (PyBuffer_ToContiguous(copy, &view, view.len, 'C') != 0) {
            free(copy);
            PyBuffer_Release(&view);
            PyErr_SetString(PyExc_BufferError, "Failed to copy buffer to contiguous memory");
            return -1;
        }
        PyBuffer_Release(&view);
        *out_buf = copy;
        *need_free = 1;
        return 0;
    }
    PyErr_Clear();

    /* 2. Fallback: Python List or Tuple */
    if (PyList_Check(obj) || PyTuple_Check(obj)) {
        Py_ssize_t len = PySequence_Fast_GET_SIZE(obj);
        if ((uint32_t)len != expected_dim) {
            PyErr_Format(PyExc_ValueError, "Vector dimension mismatch: expected %u, got %zd", expected_dim, len);
            return -1;
        }
        float* copy = (float*)malloc(expected_dim * sizeof(float));
        if (!copy) {
            PyErr_NoMemory();
            return -1;
        }
        PyObject** items = PySequence_Fast_ITEMS(obj);
        for (uint32_t i = 0; i < expected_dim; ++i) {
            copy[i] = (float)PyFloat_AsDouble(items[i]);
            if (PyErr_Occurred()) {
                free(copy);
                return -1;
            }
        }
        *out_buf = copy;
        *need_free = 1;
        return 0;
    }

    PyErr_SetString(PyExc_TypeError, "Expected numpy array, buffer, list, or tuple of floats");
    return -1;
}

static PyObject* Index_add(PyNanoVectorIndex* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"id", "vector", "metadata", NULL};
    const char* id = NULL;
    PyObject* vec_obj = NULL;
    const char* metadata = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|z", kwlist, &id, &vec_obj, &metadata)) {
        return NULL;
    }

    if (!self->index) {
        PyErr_SetString(PyExc_RuntimeError, "NanoVector index is not initialized");
        return NULL;
    }

    float* buf = NULL;
    int need_free = 0;
    if (extract_float_vector(vec_obj, self->index->dim, &buf, &need_free) != 0) {
        return NULL;
    }

    int rc = nanovector_add(self->index, id, buf, metadata);
    if (need_free) free(buf);

    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to add vector to index");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* Index_add_batch(PyNanoVectorIndex* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"ids", "vectors", "metadatas", NULL};
    PyObject* ids_obj = NULL;
    PyObject* vecs_obj = NULL;
    PyObject* metas_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", kwlist, &ids_obj, &vecs_obj, &metas_obj)) {
        return NULL;
    }

    if (!self->index) {
        PyErr_SetString(PyExc_RuntimeError, "NanoVector index is not initialized");
        return NULL;
    }

    /* Check 2D Buffer Protocol for vectors */
    Py_buffer view;
    size_t n = 0;
    const float* raw_vectors = NULL;
    float* alloc_vectors = NULL;

    if (PyObject_GetBuffer(vecs_obj, &view, PyBUF_FULL_RO) == 0) {
        if (view.ndim != 2) {
            PyBuffer_Release(&view);
            PyErr_SetString(PyExc_ValueError, "vectors buffer must be 2-dimensional (N, dim)");
            return NULL;
        }
        n = (size_t)view.shape[0];
        if ((uint32_t)view.shape[1] != self->index->dim) {
            uint32_t got_dim = (uint32_t)view.shape[1];
            PyBuffer_Release(&view);
            PyErr_Format(PyExc_ValueError, "vectors dim mismatch: expected %u, got %u", self->index->dim, got_dim);
            return NULL;
        }
        if (PyBuffer_IsContiguous(&view, 'C') && view.itemsize == sizeof(float)) {
            raw_vectors = (const float*)view.buf;
        } else {
            alloc_vectors = (float*)malloc(n * self->index->dim * sizeof(float));
            if (!alloc_vectors) {
                PyBuffer_Release(&view);
                return PyErr_NoMemory();
            }
            PyBuffer_ToContiguous(alloc_vectors, &view, view.len, 'C');
            raw_vectors = alloc_vectors;
        }
        PyBuffer_Release(&view);
    } else {
        PyErr_Clear();
        /* List of 1D vectors */
        if (!PySequence_Check(vecs_obj)) {
            PyErr_SetString(PyExc_TypeError, "vectors must be 2D numpy array or sequence of 1D vectors");
            return NULL;
        }
        n = PySequence_Length(vecs_obj);
        alloc_vectors = (float*)malloc(n * self->index->dim * sizeof(float));
        if (!alloc_vectors) return PyErr_NoMemory();

        for (size_t i = 0; i < n; ++i) {
            PyObject* item = PySequence_GetItem(vecs_obj, i);
            float* row = NULL;
            int nf = 0;
            if (extract_float_vector(item, self->index->dim, &row, &nf) != 0) {
                Py_XDECREF(item);
                free(alloc_vectors);
                return NULL;
            }
            memcpy(alloc_vectors + i * self->index->dim, row, self->index->dim * sizeof(float));
            if (nf) free(row);
            Py_DECREF(item);
        }
        raw_vectors = alloc_vectors;
    }

    /* Extract IDs */
    if (!PySequence_Check(ids_obj) || (size_t)PySequence_Length(ids_obj) != n) {
        if (alloc_vectors) free(alloc_vectors);
        PyErr_SetString(PyExc_ValueError, "ids length must match number of vectors");
        return NULL;
    }

    const char** c_ids = (const char**)malloc(n * sizeof(const char*));
    if (!c_ids) {
        if (alloc_vectors) free(alloc_vectors);
        return PyErr_NoMemory();
    }
    for (size_t i = 0; i < n; ++i) {
        PyObject* str_item = PySequence_GetItem(ids_obj, i);
        if (PyUnicode_Check(str_item)) {
            c_ids[i] = PyUnicode_AsUTF8(str_item);
        } else {
            c_ids[i] = NULL;
        }
        Py_XDECREF(str_item);
    }

    /* Extract Metadatas */
    const char** c_metas = NULL;
    if (metas_obj != Py_None && PySequence_Check(metas_obj)) {
        if ((size_t)PySequence_Length(metas_obj) == n) {
            c_metas = (const char**)malloc(n * sizeof(const char*));
            if (c_metas) {
                for (size_t i = 0; i < n; ++i) {
                    PyObject* meta_item = PySequence_GetItem(metas_obj, i);
                    if (meta_item && PyUnicode_Check(meta_item)) {
                        c_metas[i] = PyUnicode_AsUTF8(meta_item);
                    } else {
                        c_metas[i] = NULL;
                    }
                    Py_XDECREF(meta_item);
                }
            }
        }
    }

    /* Add batch with GIL released */
    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    rc = nanovector_add_batch(self->index, c_ids, raw_vectors, n, c_metas);
    Py_END_ALLOW_THREADS

    free(c_ids);
    if (c_metas) free(c_metas);
    if (alloc_vectors) free(alloc_vectors);

    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "nanovector_add_batch failed");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* Index_search(PyNanoVectorIndex* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"query", "top_k", NULL};
    PyObject* q_obj = NULL;
    unsigned int top_k = 10;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|I", kwlist, &q_obj, &top_k)) {
        return NULL;
    }

    if (!self->index) {
        PyErr_SetString(PyExc_RuntimeError, "NanoVector index is not initialized");
        return NULL;
    }

    float* q_buf = NULL;
    int need_free = 0;
    if (extract_float_vector(q_obj, self->index->dim, &q_buf, &need_free) != 0) {
        return NULL;
    }

    if (top_k == 0 || self->index->count == 0) {
        if (need_free) free(q_buf);
        return PyList_New(0);
    }

    nanovector_match_t* matches = (nanovector_match_t*)malloc(top_k * sizeof(nanovector_match_t));
    if (!matches) {
        if (need_free) free(q_buf);
        return PyErr_NoMemory();
    }

    size_t found = 0;
    Py_BEGIN_ALLOW_THREADS
    found = nanovector_search(self->index, q_buf, top_k, matches);
    Py_END_ALLOW_THREADS

    if (need_free) free(q_buf);

    PyObject* py_results = PyList_New(found);
    if (!py_results) {
        free(matches);
        return NULL;
    }

    for (size_t i = 0; i < found; ++i) {
        PyObject* match_dict = PyDict_New();
        PyDict_SetItemString(match_dict, "id", matches[i].id ? PyUnicode_FromString(matches[i].id) : Py_None);
        PyDict_SetItemString(match_dict, "score", PyFloat_FromDouble((double)matches[i].score));
        PyDict_SetItemString(match_dict, "metadata", matches[i].metadata ? PyUnicode_FromString(matches[i].metadata) : Py_None);
        PyList_SET_ITEM(py_results, i, match_dict);
    }

    free(matches);
    return py_results;
}

static PyObject* Index_save(PyNanoVectorIndex* self, PyObject* args) {
    const char* filepath = NULL;
    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        return NULL;
    }

    if (!self->index) {
        PyErr_SetString(PyExc_RuntimeError, "NanoVector index is not initialized");
        return NULL;
    }

    int rc = nanovector_save(self->index, filepath);
    if (rc != 0) {
        PyErr_Format(PyExc_IOError, "Failed to save NanoVector index to '%s'", filepath);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* Index_load(PyObject* cls, PyObject* args) {
    const char* filepath = NULL;
    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        return NULL;
    }

    nanovector_index_t* idx = nanovector_load(filepath);
    if (!idx) {
        PyErr_Format(PyExc_IOError, "Failed to load NanoVector index from '%s' (corrupted or missing)", filepath);
        return NULL;
    }

    PyTypeObject* type = (PyTypeObject*)cls;
    PyNanoVectorIndex* obj = (PyNanoVectorIndex*)type->tp_alloc(type, 0);
    if (!obj) {
        nanovector_free(idx);
        return NULL;
    }

    obj->index = idx;
    return (PyObject*)obj;
}

static Py_ssize_t Index_len(PyNanoVectorIndex* self) {
    return self->index ? (Py_ssize_t)self->index->count : 0;
}

static PyObject* Index_get_dim(PyNanoVectorIndex* self, void* closure) {
    return PyLong_FromUnsignedLong(self->index ? self->index->dim : 0);
}

static PyObject* Index_get_count(PyNanoVectorIndex* self, void* closure) {
    return PyLong_FromUnsignedLongLong(self->index ? self->index->count : 0);
}

static PyObject* Index_get_metric(PyNanoVectorIndex* self, void* closure) {
    if (!self->index) Py_RETURN_NONE;
    switch (self->index->metric) {
        case NANOVEC_METRIC_COSINE: return PyUnicode_FromString("cosine");
        case NANOVEC_METRIC_DOT:    return PyUnicode_FromString("dot");
        case NANOVEC_METRIC_L2:     return PyUnicode_FromString("l2");
        default:                    return PyUnicode_FromString("unknown");
    }
}

static PyGetSetDef Index_getset[] = {
    {"dim", (getter)Index_get_dim, NULL, "Vector dimensionality", NULL},
    {"count", (getter)Index_get_count, NULL, "Number of vectors in index", NULL},
    {"metric", (getter)Index_get_metric, NULL, "Distance/similarity metric", NULL},
    {NULL}
};

static PyMethodDef Index_methods[] = {
    {"add", (PyCFunction)Index_add, METH_VARARGS | METH_KEYWORDS, "Add single vector with ID and optional metadata"},
    {"add_batch", (PyCFunction)Index_add_batch, METH_VARARGS | METH_KEYWORDS, "Add multiple vectors in batch (Zero-Copy from NumPy)"},
    {"search", (PyCFunction)Index_search, METH_VARARGS | METH_KEYWORDS, "Search Top-K nearest neighbors"},
    {"save", (PyCFunction)Index_save, METH_VARARGS, "Save index to a single .nvec file"},
    {"load", (PyCFunction)Index_load, METH_VARARGS | METH_CLASS, "Load index from a .nvec file"},
    {NULL}
};

static PySequenceMethods Index_sequence = {
    .sq_length = (lenfunc)Index_len,
};

static PyTypeObject PyNanoVectorIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nanovector._ext.Index",
    .tp_doc = "NanoVector embedded vector index",
    .tp_basicsize = sizeof(PyNanoVectorIndex),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Index_init,
    .tp_dealloc = (destructor)Index_dealloc,
    .tp_methods = Index_methods,
    .tp_getset = Index_getset,
    .tp_as_sequence = &Index_sequence,
};

static PyObject* py_version(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(nanovector_version());
}

static PyObject* py_backend(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(nanovector_simd_backend());
}

static PyMethodDef ModuleMethods[] = {
    {"version", py_version, METH_NOARGS, "Return NanoVector version string"},
    {"simd_backend", py_backend, METH_NOARGS, "Return active SIMD acceleration backend"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nanovectormodule = {
    PyModuleDef_HEAD_INIT,
    "nanovector._ext",
    "NanoVector C Extension Module",
    -1,
    ModuleMethods
};

PyMODINIT_FUNC PyInit__ext(void) {
    PyObject* m;
    if (PyType_Ready(&PyNanoVectorIndexType) < 0) return NULL;

    m = PyModule_Create(&nanovectormodule);
    if (!m) return NULL;

    Py_INCREF(&PyNanoVectorIndexType);
    if (PyModule_AddObject(m, "Index", (PyObject*)&PyNanoVectorIndexType) < 0) {
        Py_DECREF(&PyNanoVectorIndexType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
