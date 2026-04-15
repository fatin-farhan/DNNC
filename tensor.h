#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
    int n;      /* batch */
    int c;      /* channels */
    int h;      /* height */
    int w;      /* width */
    float *data;
} Tensor4D;

/* Allocate tensor with uninitialized data */
Tensor4D tensor_alloc(int n, int c, int h, int w);

/* Create tensor view/copy from existing data (no allocation) */
Tensor4D tensor_from_data(int n, int c, int h, int w, float *data);

/* Free tensor memory (only if allocated) */
void tensor_free(Tensor4D *t);

/* Number of elements */
size_t tensor_numel(const Tensor4D *t);

/* Accessors */
float tensor_get(const Tensor4D *t, int n, int c, int h, int w);
void tensor_set(Tensor4D *t, int n, int c, int h, int w, float value);

#endif