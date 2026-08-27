/* 055_Parameter_Count.c */
#include <stdio.h>

int main(void)
{
    /* CNN for 1-channel, 16-wide input, 3 output
       classes */
    int conv1_params = 4 * 1 * 3 + 4;
    /* 4 filters, 1 in_ch, kernel 3, + bias */
    int conv2_params = 2 * 4 * 3 + 2;
    /* 2 filters, 4 in_ch, kernel 3, + bias */
    int flat_size = 2 * 2;
    /* 2 channels x 2 spatial after pooling */
    int dense_params = 3 * flat_size + 3;
    /* 3 classes */
    int cnn_total = conv1_params + conv2_params
        + dense_params;

    /* MLP for same input -> same output */
    int mlp_hidden = 32;
    int mlp_l1 = 16 * mlp_hidden + mlp_hidden;
    int mlp_l2 = mlp_hidden * 3 + 3;
    int mlp_total = mlp_l1 + mlp_l2;

    printf("CNN parameter count:\n");
    printf("  Conv1: %d filters x %d in_ch x %d kernel "
           "+ %d bias = %d\n",
           4, 1, 3, 4, conv1_params);
    printf("  Conv2: %d filters x %d in_ch x %d kernel "
           "+ %d bias = %d\n",
           2, 4, 3, 2, conv2_params);
    printf("  Dense: %d classes x %d flat + %d bias = "
           "%d\n",
           3, flat_size, 3, dense_params);
    printf("  Total: %d\n\n", cnn_total);

    printf("Equivalent MLP parameter count:\n");
    printf("  Layer 1: %d inputs x %d hidden + %d bias "
           "= %d\n",
           16, mlp_hidden, mlp_hidden, mlp_l1);
    printf("  Layer 2: %d hidden x %d classes + %d "
           "bias = %d\n",
           mlp_hidden, 3, 3, mlp_l2);
    printf("  Total: %d\n\n", mlp_total);

    printf("CNN uses %.1fx fewer parameters.\n",
           (float)mlp_total / cnn_total);
    printf("And the CNN would work on inputs of ANY "
           "length,\n");
    printf("not just 16. The MLP is fixed to 16 "
           "inputs.\n");

    return 0;
}
