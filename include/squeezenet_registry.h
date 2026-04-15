#ifndef SQUEEZENET_REGISTRY_H
#define SQUEEZENET_REGISTRY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SQUEEZENET_MAX_DIMS 8
#define SQUEEZENET_TENSOR_COUNT 52

typedef struct {
    const char* name;
    const float* data;
    size_t numel;
    int ndim;
    int shape[SQUEEZENET_MAX_DIMS];
} tensor_view_t;

extern const tensor_view_t squeezenet_tensors[SQUEEZENET_TENSOR_COUNT];

#ifdef __cplusplus
}
#endif

#endif /* SQUEEZENET_REGISTRY_H */
