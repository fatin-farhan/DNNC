#include "tensor.h"

#include <stdlib.h>
#include <stdio.h>

/* internal helper: compute flat index (NCHW layout) */
static inline size_t tensor_index(
    const Tensor4D *t,
    int n, int c, int h, int w
) {
    return ((size_t)n * t->c * t->h * t->w) +
           ((size_t)c * t->h * t->w) +
           ((size_t)h * t->w) +
           (size_t)w;
}

Tensor4D tensor_alloc(int n, int c, int h, int w) {
    Tensor4D t;
    t.n = n;
    t.c = c;
    t.h = h;
    t.w = w;

    size_t numel = (size_t)n * c * h * w;
    t.data = (float *)malloc(numel * sizeof(float));

    if (!t.data) {
        fprintf(stderr, "tensor_alloc: malloc failed\n");
        exit(1);
    }

    return t;
}

Tensor4D tensor_from_data(int n, int c, int h, int w, float *data) {
    Tensor4D t;
    t.n = n;
    t.c = c;
    t.h = h;
    t.w = w;
    t.data = data;  /* no allocation, just reference */

    return t;
}

void tensor_free(Tensor4D *t) {
    if (!t || !t->data) return;

    free(t->data);
    t->data = NULL;
}

size_t tensor_numel(const Tensor4D *t) {
    return (size_t)t->n * t->c * t->h * t->w;
}

float tensor_get(const Tensor4D *t, int n, int c, int h, int w) {
    return t->data[tensor_index(t, n, c, h, w)];
}

void tensor_set(Tensor4D *t, int n, int c, int h, int w, float value) {
    t->data[tensor_index(t, n, c, h, w)] = value;
}