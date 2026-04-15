#include <stdio.h>
#include "tensor.h"
#include "input_tensor.h"
#include "runner.h"

void app_main(void) {
    Tensor4D input = tensor_from_data(1, 3, 224, 224, (float *)input_tensor);

    int ranges[][2] = {
        {0, 111}
    };

    int num_ranges = (int)(sizeof(ranges) / sizeof(ranges[0]));

    for (int i = 0; i < num_ranges; i++) {
        int oh_begin = ranges[i][0];
        int oh_end   = ranges[i][1];

        InputWindow win = features0_input_window(oh_begin, oh_end);

        print_portion_info("features.0", oh_begin, oh_end, &win, &input);
        execute_features0_one_channel_at_a_time(&input, &win, oh_begin, oh_end);
        printf("\n");
    }
}