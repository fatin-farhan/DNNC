#include "squeezenet_registry.h"
#include "features_0_weight.h"
#include "features_0_bias.h"
#include "features_3_squeeze_weight.h"
#include "features_3_squeeze_bias.h"
#include "features_3_expand1x1_weight.h"
#include "features_3_expand1x1_bias.h"
#include "features_3_expand3x3_weight.h"
#include "features_3_expand3x3_bias.h"
#include "features_4_squeeze_weight.h"
#include "features_4_squeeze_bias.h"
#include "features_4_expand1x1_weight.h"
#include "features_4_expand1x1_bias.h"
#include "features_4_expand3x3_weight.h"
#include "features_4_expand3x3_bias.h"
#include "features_5_squeeze_weight.h"
#include "features_5_squeeze_bias.h"
#include "features_5_expand1x1_weight.h"
#include "features_5_expand1x1_bias.h"
#include "features_5_expand3x3_weight.h"
#include "features_5_expand3x3_bias.h"
#include "features_7_squeeze_weight.h"
#include "features_7_squeeze_bias.h"
#include "features_7_expand1x1_weight.h"
#include "features_7_expand1x1_bias.h"
#include "features_7_expand3x3_weight.h"
#include "features_7_expand3x3_bias.h"
#include "features_8_squeeze_weight.h"
#include "features_8_squeeze_bias.h"
#include "features_8_expand1x1_weight.h"
#include "features_8_expand1x1_bias.h"
#include "features_8_expand3x3_weight.h"
#include "features_8_expand3x3_bias.h"
#include "features_9_squeeze_weight.h"
#include "features_9_squeeze_bias.h"
#include "features_9_expand1x1_weight.h"
#include "features_9_expand1x1_bias.h"
#include "features_9_expand3x3_weight.h"
#include "features_9_expand3x3_bias.h"
#include "features_10_squeeze_weight.h"
#include "features_10_squeeze_bias.h"
#include "features_10_expand1x1_weight.h"
#include "features_10_expand1x1_bias.h"
#include "features_10_expand3x3_weight.h"
#include "features_10_expand3x3_bias.h"
#include "features_12_squeeze_weight.h"
#include "features_12_squeeze_bias.h"
#include "features_12_expand1x1_weight.h"
#include "features_12_expand1x1_bias.h"
#include "features_12_expand3x3_weight.h"
#include "features_12_expand3x3_bias.h"
#include "classifier_1_weight.h"
#include "classifier_1_bias.h"

const tensor_view_t squeezenet_tensors[SQUEEZENET_TENSOR_COUNT] = {
    {
        "features.0.weight",
        features_0_weight,
        14112,
        4,
        { 96, 3, 7, 7, 0, 0, 0, 0 }
    },
    {
        "features.0.bias",
        features_0_bias,
        96,
        1,
        { 96, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.3.squeeze.weight",
        features_3_squeeze_weight,
        1536,
        4,
        { 16, 96, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.3.squeeze.bias",
        features_3_squeeze_bias,
        16,
        1,
        { 16, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.3.expand1x1.weight",
        features_3_expand1x1_weight,
        1024,
        4,
        { 64, 16, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.3.expand1x1.bias",
        features_3_expand1x1_bias,
        64,
        1,
        { 64, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.3.expand3x3.weight",
        features_3_expand3x3_weight,
        9216,
        4,
        { 64, 16, 3, 3, 0, 0, 0, 0 }
    },
    {
        "features.3.expand3x3.bias",
        features_3_expand3x3_bias,
        64,
        1,
        { 64, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.4.squeeze.weight",
        features_4_squeeze_weight,
        2048,
        4,
        { 16, 128, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.4.squeeze.bias",
        features_4_squeeze_bias,
        16,
        1,
        { 16, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.4.expand1x1.weight",
        features_4_expand1x1_weight,
        1024,
        4,
        { 64, 16, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.4.expand1x1.bias",
        features_4_expand1x1_bias,
        64,
        1,
        { 64, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.4.expand3x3.weight",
        features_4_expand3x3_weight,
        9216,
        4,
        { 64, 16, 3, 3, 0, 0, 0, 0 }
    },
    {
        "features.4.expand3x3.bias",
        features_4_expand3x3_bias,
        64,
        1,
        { 64, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.5.squeeze.weight",
        features_5_squeeze_weight,
        4096,
        4,
        { 32, 128, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.5.squeeze.bias",
        features_5_squeeze_bias,
        32,
        1,
        { 32, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.5.expand1x1.weight",
        features_5_expand1x1_weight,
        4096,
        4,
        { 128, 32, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.5.expand1x1.bias",
        features_5_expand1x1_bias,
        128,
        1,
        { 128, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.5.expand3x3.weight",
        features_5_expand3x3_weight,
        36864,
        4,
        { 128, 32, 3, 3, 0, 0, 0, 0 }
    },
    {
        "features.5.expand3x3.bias",
        features_5_expand3x3_bias,
        128,
        1,
        { 128, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.7.squeeze.weight",
        features_7_squeeze_weight,
        8192,
        4,
        { 32, 256, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.7.squeeze.bias",
        features_7_squeeze_bias,
        32,
        1,
        { 32, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.7.expand1x1.weight",
        features_7_expand1x1_weight,
        4096,
        4,
        { 128, 32, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.7.expand1x1.bias",
        features_7_expand1x1_bias,
        128,
        1,
        { 128, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.7.expand3x3.weight",
        features_7_expand3x3_weight,
        36864,
        4,
        { 128, 32, 3, 3, 0, 0, 0, 0 }
    },
    {
        "features.7.expand3x3.bias",
        features_7_expand3x3_bias,
        128,
        1,
        { 128, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.8.squeeze.weight",
        features_8_squeeze_weight,
        12288,
        4,
        { 48, 256, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.8.squeeze.bias",
        features_8_squeeze_bias,
        48,
        1,
        { 48, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.8.expand1x1.weight",
        features_8_expand1x1_weight,
        9216,
        4,
        { 192, 48, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.8.expand1x1.bias",
        features_8_expand1x1_bias,
        192,
        1,
        { 192, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.8.expand3x3.weight",
        features_8_expand3x3_weight,
        82944,
        4,
        { 192, 48, 3, 3, 0, 0, 0, 0 }
    },
    {
        "features.8.expand3x3.bias",
        features_8_expand3x3_bias,
        192,
        1,
        { 192, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.9.squeeze.weight",
        features_9_squeeze_weight,
        18432,
        4,
        { 48, 384, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.9.squeeze.bias",
        features_9_squeeze_bias,
        48,
        1,
        { 48, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.9.expand1x1.weight",
        features_9_expand1x1_weight,
        9216,
        4,
        { 192, 48, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.9.expand1x1.bias",
        features_9_expand1x1_bias,
        192,
        1,
        { 192, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.9.expand3x3.weight",
        features_9_expand3x3_weight,
        82944,
        4,
        { 192, 48, 3, 3, 0, 0, 0, 0 }
    },
    {
        "features.9.expand3x3.bias",
        features_9_expand3x3_bias,
        192,
        1,
        { 192, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.10.squeeze.weight",
        features_10_squeeze_weight,
        24576,
        4,
        { 64, 384, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.10.squeeze.bias",
        features_10_squeeze_bias,
        64,
        1,
        { 64, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.10.expand1x1.weight",
        features_10_expand1x1_weight,
        16384,
        4,
        { 256, 64, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.10.expand1x1.bias",
        features_10_expand1x1_bias,
        256,
        1,
        { 256, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.10.expand3x3.weight",
        features_10_expand3x3_weight,
        147456,
        4,
        { 256, 64, 3, 3, 0, 0, 0, 0 }
    },
    {
        "features.10.expand3x3.bias",
        features_10_expand3x3_bias,
        256,
        1,
        { 256, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.12.squeeze.weight",
        features_12_squeeze_weight,
        32768,
        4,
        { 64, 512, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.12.squeeze.bias",
        features_12_squeeze_bias,
        64,
        1,
        { 64, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.12.expand1x1.weight",
        features_12_expand1x1_weight,
        16384,
        4,
        { 256, 64, 1, 1, 0, 0, 0, 0 }
    },
    {
        "features.12.expand1x1.bias",
        features_12_expand1x1_bias,
        256,
        1,
        { 256, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "features.12.expand3x3.weight",
        features_12_expand3x3_weight,
        147456,
        4,
        { 256, 64, 3, 3, 0, 0, 0, 0 }
    },
    {
        "features.12.expand3x3.bias",
        features_12_expand3x3_bias,
        256,
        1,
        { 256, 0, 0, 0, 0, 0, 0, 0 }
    },
    {
        "classifier.1.weight",
        classifier_1_weight,
        512000,
        4,
        { 1000, 512, 1, 1, 0, 0, 0, 0 }
    },
    {
        "classifier.1.bias",
        classifier_1_bias,
        1000,
        1,
        { 1000, 0, 0, 0, 0, 0, 0, 0 }
    },
};
