#include "ops.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

static inline int out_dim(int in, int k, int stride, int pad) {
    return (in + 2 * pad - k) / stride + 1;
}

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
) {
    if (x->c != in_c) {
        fprintf(stderr, "conv2d: input channels mismatch: got %d expected %d\n", x->c, in_c);
        exit(1);
    }

    int out_h = out_dim(x->h, kH, stride, pad);
    int out_w = out_dim(x->w, kW, stride, pad);
    Tensor4D y = tensor_alloc(x->n, out_c, out_h, out_w);

    for (int n = 0; n < x->n; n++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = bias ? bias[oc] : 0.0f;

                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride - pad + kh;
                                int iw = ow * stride - pad + kw;

                                if (ih < 0 || ih >= x->h || iw < 0 || iw >= x->w) {
                                    continue;
                                }

                                float xv = tensor_get(x, n, ic, ih, iw);
                                size_t widx = ((size_t)oc * in_c * kH * kW) +
                                              ((size_t)ic * kH * kW) +
                                              ((size_t)kh * kW) + kw;
                                float wv = weight[widx];
                                sum += xv * wv;
                            }
                        }
                    }

                    tensor_set(&y, n, oc, oh, ow, sum);
                }
            }
        }
    }

    return y;
}

void relu_inplace(Tensor4D *x) {
    size_t n = tensor_numel(x);
    for (size_t i = 0; i < n; i++) {
        if (x->data[i] < 0.0f) x->data[i] = 0.0f;
    }
}

Tensor4D maxpool2d(
    const Tensor4D *x,
    int kH,
    int kW,
    int stride,
    int pad
) {
    int out_h = out_dim(x->h, kH, stride, pad);
    int out_w = out_dim(x->w, kW, stride, pad);
    Tensor4D y = tensor_alloc(x->n, x->c, out_h, out_w);

    for (int n = 0; n < x->n; n++) {
        for (int c = 0; c < x->c; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float m = -FLT_MAX;

                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * stride - pad + kh;
                            int iw = ow * stride - pad + kw;

                            if (ih < 0 || ih >= x->h || iw < 0 || iw >= x->w) {
                                continue;
                            }

                            float v = tensor_get(x, n, c, ih, iw);
                            if (v > m) m = v;
                        }
                    }

                    tensor_set(&y, n, c, oh, ow, m);
                }
            }
        }
    }

    return y;
}

Tensor4D avgpool2d(
    const Tensor4D *x,
    int kH,
    int kW,
    int stride,
    int pad
) {
    int out_h = out_dim(x->h, kH, stride, pad);
    int out_w = out_dim(x->w, kW, stride, pad);
    Tensor4D y = tensor_alloc(x->n, x->c, out_h, out_w);

    for (int n = 0; n < x->n; n++) {
        for (int c = 0; c < x->c; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = 0.0f;
                    int count = 0;

                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * stride - pad + kh;
                            int iw = ow * stride - pad + kw;

                            if (ih < 0 || ih >= x->h || iw < 0 || iw >= x->w) {
                                continue;
                            }

                            sum += tensor_get(x, n, c, ih, iw);
                            count++;
                        }
                    }

                    tensor_set(&y, n, c, oh, ow, count > 0 ? sum / count : 0.0f);
                }
            }
        }
    }

    return y;
}

Tensor4D concat_channels(const Tensor4D *a, const Tensor4D *b) {
    if (a->n != b->n || a->h != b->h || a->w != b->w) {
        fprintf(stderr, "concat_channels: shape mismatch\n");
        exit(1);
    }

    Tensor4D y = tensor_alloc(a->n, a->c + b->c, a->h, a->w);

    for (int n = 0; n < a->n; n++) {
        for (int c = 0; c < a->c; c++) {
            for (int h = 0; h < a->h; h++) {
                for (int w = 0; w < a->w; w++) {
                    tensor_set(&y, n, c, h, w, tensor_get(a, n, c, h, w));
                }
            }
        }

        for (int c = 0; c < b->c; c++) {
            for (int h = 0; h < b->h; h++) {
                for (int w = 0; w < b->w; w++) {
                    tensor_set(&y, n, a->c + c, h, w, tensor_get(b, n, c, h, w));
                }
            }
        }
    }

    return y;
}

Tensor4D fire_module(
    const Tensor4D *x,
    const float *sq_w, const float *sq_b,
    int sq_out, int sq_in,
    const float *e1_w, const float *e1_b,
    int e1_out, int e1_in,
    const float *e3_w, const float *e3_b,
    int e3_out, int e3_in
) {
    Tensor4D sq = conv2d(x, sq_w, sq_b, sq_out, sq_in, 1, 1, 1, 0);
    relu_inplace(&sq);

    Tensor4D e1 = conv2d(&sq, e1_w, e1_b, e1_out, e1_in, 1, 1, 1, 0);
    relu_inplace(&e1);

    Tensor4D e3 = conv2d(&sq, e3_w, e3_b, e3_out, e3_in, 3, 3, 1, 1);
    relu_inplace(&e3);

    tensor_free(&sq);

    Tensor4D y = concat_channels(&e1, &e3);
    tensor_free(&e1);
    tensor_free(&e3);

    return y;
}