#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/features_0_weight.h"
#include "include/features_0_bias.h"

static inline int out_dim(int in, int k, int stride, int pad) {
    return (in + 2 * pad - k) / stride + 1;
}

static inline size_t weight_idx(
    int oc, int ic, int kh, int kw,
    int in_c, int kH, int kW
) {
    return ((size_t)oc * in_c * kH * kW) +
           ((size_t)ic * kH * kW) +
           ((size_t)kh * kW) +
           (size_t)kw;
}

static size_t tensor_bytes(const Tensor4D *t) {
    return tensor_numel(t) * sizeof(float);
}

static size_t input_window_bytes(const Tensor4D *input, const InputWindow *win) {
    size_t rows = (size_t)(win->in_h_end - win->in_h_begin);
    return (size_t)input->n * input->c * rows * input->w * sizeof(float);
}

double timer_now_ms(void) {
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
}

InputWindow features0_input_window(int oh_begin, int oh_end) {
    const int kH = 3;
    const int stride = 2;
    const int pad = 0;
    const int input_h = 224;
    const int input_w = 224;
    const int output_h = out_dim(input_h, kH, stride, pad);

    if (oh_begin < 0 || oh_end > output_h || oh_begin >= oh_end) {
        fprintf(stderr, "features0_input_window: invalid output range [%d, %d)\n", oh_begin, oh_end);
        exit(1);
    }

    InputWindow win;
    win.in_h_begin = oh_begin * stride - pad;
    win.in_h_end   = (oh_end - 1) * stride - pad + kH; /* exclusive */

    win.in_w_begin = 0;
    win.in_w_end   = input_w;

    if (win.in_h_begin < 0) win.in_h_begin = 0;
    if (win.in_h_end > input_h) win.in_h_end = input_h;

    return win;
}

Tensor4D execute_features0_single_channel(
    const Tensor4D *input,
    const InputWindow *win,
    int oh_begin,
    int oh_end,
    int oc
) {
    const int out_c = 64;
    const int in_c = 3;
    const int kH = 3;
    const int kW = 3;
    const int stride = 2;
    const int pad = 0;
    const int out_w = out_dim(224, kW, stride, pad);

    if (oc < 0 || oc >= out_c) {
        fprintf(stderr, "execute_features0_single_channel: invalid oc=%d\n", oc);
        exit(1);
    }

    if (input->n != 1 || input->c != in_c || input->h != 224 || input->w != 224) {
        fprintf(stderr, "execute_features0_single_channel: unexpected input shape\n");
        exit(1);
    }

    int out_h = oh_end - oh_begin;
    Tensor4D y = tensor_alloc(1, 1, out_h, out_w);

    for (int oh = oh_begin; oh < oh_end; oh++) {
        int local_oh = oh - oh_begin;

        for (int ow = 0; ow < out_w; ow++) {
            float sum = features_0_bias[oc];

            for (int ic = 0; ic < in_c; ic++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int global_ih = oh * stride - pad + kh;
                        int global_iw = ow * stride - pad + kw;

                        if (global_ih < win->in_h_begin || global_ih >= win->in_h_end ||
                            global_iw < win->in_w_begin || global_iw >= win->in_w_end) {
                            continue;
                        }

                        if (global_ih < 0 || global_ih >= input->h ||
                            global_iw < 0 || global_iw >= input->w) {
                            continue;
                        }

                        float xv = tensor_get(input, 0, ic, global_ih, global_iw);
                        float wv = features_0_weight[
                            weight_idx(oc, ic, kh, kw, in_c, kH, kW)
                        ];

                        sum += xv * wv;
                    }
                }
            }

            if (sum < 0.0f) {
                sum = 0.0f;
            }

            tensor_set(&y, 0, 0, local_oh, ow, sum);
        }
    }

    return y;
}

void execute_features0_one_channel_at_a_time(
    const Tensor4D *input,
    const InputWindow *win,
    int oh_begin,
    int oh_end
) {
    for (int oc = 0; oc < 64; oc++) {
        double t0 = timer_now_ms();
        Tensor4D output_part = execute_features0_single_channel(
            input, win, oh_begin, oh_end, oc
        );
        double t1 = timer_now_ms();

        print_channel_execution_summary(oc, input, win, &output_part, t1 - t0);
        tensor_free(&output_part);
    }
}

void print_portion_info(
    const char *name,
    int oh_begin,
    int oh_end,
    const InputWindow *win,
    const Tensor4D *input
) {
    size_t bytes = input_window_bytes(input, win);

    printf("%s output rows [%d, %d)\n", name, oh_begin, oh_end);
    printf("required input rows [%d, %d)\n", win->in_h_begin, win->in_h_end);
    printf("referenced input shape = [%d, %d, %d, %d]\n",
           input->n, input->c, win->in_h_end - win->in_h_begin, input->w);
    printf("referenced input bytes = %zu bytes (%.2f KB, %.2f MB)\n",
           bytes,
           (double)bytes / 1024.0,
           (double)bytes / (1024.0 * 1024.0));
    printf("input copied to RAM = 0 bytes\n");
}

void print_channel_execution_summary(
    int oc,
    const Tensor4D *input,
    const InputWindow *win,
    const Tensor4D *output_part,
    double elapsed_ms
) {
    size_t ref_bytes = input_window_bytes(input, win);
    size_t out_bytes = tensor_bytes(output_part);

    printf("channel %d\n", oc);
    printf("output shape = [%d, %d, %d, %d]\n",
           output_part->n, output_part->c, output_part->h, output_part->w);
    printf("output memory = %zu bytes (%.2f KB, %.2f MB)\n",
           out_bytes,
           (double)out_bytes / 1024.0,
           (double)out_bytes / (1024.0 * 1024.0));
    printf("referenced flash input = %zu bytes (%.2f KB, %.2f MB)\n",
           ref_bytes,
           (double)ref_bytes / 1024.0,
           (double)ref_bytes / (1024.0 * 1024.0));
    printf("RAM working set ~= %zu bytes (%.2f KB, %.2f MB)\n",
           out_bytes,
           (double)out_bytes / 1024.0,
           (double)out_bytes / (1024.0 * 1024.0));
    printf("execution time = %.3f ms\n", elapsed_ms);
}