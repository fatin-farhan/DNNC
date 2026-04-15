#ifndef RUNNER_H
#define RUNNER_H

#include "tensor.h"

typedef struct {
    int in_h_begin;
    int in_h_end;   /* exclusive */
    int in_w_begin;
    int in_w_end;   /* exclusive */
} InputWindow;

InputWindow features0_input_window(int oh_begin, int oh_end);

Tensor4D execute_features0_single_channel(
    const Tensor4D *input,
    const InputWindow *win,
    int oh_begin,
    int oh_end,
    int oc
);

void execute_features0_one_channel_at_a_time(
    const Tensor4D *input,
    const InputWindow *win,
    int oh_begin,
    int oh_end
);

double timer_now_ms(void);

void print_portion_info(
    const char *name,
    int oh_begin,
    int oh_end,
    const InputWindow *win,
    const Tensor4D *input
);

void print_channel_execution_summary(
    int oc,
    const Tensor4D *input,
    const InputWindow *win,
    const Tensor4D *output_part,
    double elapsed_ms
);

#endif