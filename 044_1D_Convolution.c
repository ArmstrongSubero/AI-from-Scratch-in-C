/* 044_1D_Convolution.c */
#include <stdio.h>

int main(void)
{
    /* A simple signal */
    float signal[] = {0, 0, 1, 2, 1, 0, 0, 0};
    int sig_len = 8;

    /* A spike-detecting kernel */
    float kernel[] = { -1, 2, -1};
    int ker_len = 3;

    /* Output length = sig_len - ker_len + 1 */
    int out_len = sig_len - ker_len + 1;
    float output[6];
    int i, k;

    /* convolution: slide kernel across signal */
    for (i = 0; i < out_len; i++)
    {
        float sum = 0.0f;
        for (k = 0; k < ker_len; k++)
            sum += signal[i + k] * kernel[k];

        output[i] = sum;
    }

    printf("Signal: ");
    for (i = 0; i < sig_len; i++)
        printf("%5.1f ", signal[i]);

    printf("\nKernel: ");
    for (k = 0; k < ker_len; k++)
        printf("%5.1f ",kernel[k]);

    printf("\nOutput: ");
    for (i = 0; i < out_len; i++)
        printf("%5.1f ", output[i]);
    printf("\n");


    return 0;
}
