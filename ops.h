#ifndef OPS_H
#define OPS_H

#include "tensor.h"

Tensor4D conv2d(
    const Tensor4D *x,
    const float *weight,
    const float *bias,
    int out_c,
    int in_c,
    int kH,
    int kW,
    int stride,
    int pad
);

void relu_inplace(Tensor4D *x);

Tensor4D maxpool2d(
    const Tensor4D *x,
    int kH,
    int kW,
    int stride,
    int pad
);

Tensor4D avgpool2d(
    const Tensor4D *x,
    int kH,
    int kW,
    int stride,
    int pad
);

Tensor4D concat_channels(const Tensor4D *a, const Tensor4D *b);

Tensor4D fire_module(
    const Tensor4D *x,
    const float *sq_w, const float *sq_b,
    int sq_out, int sq_in,
    const float *e1_w, const float *e1_b,
    int e1_out, int e1_in,
    const float *e3_w, const float *e3_b,
    int e3_out, int e3_in
);

#endif