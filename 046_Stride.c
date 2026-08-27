/* 046_Stride.c */
#include <stdio.h>

static int conv1d(const float *signal, int sig_len, 
                  const float *kernel, int ker_len, 
                  int stride, float *output)
{
    int out_len = (sig_len - ker_len) / stride + 1;
    int i, k;

    for (i = 0; i < out_len; i++) {
        float sum = 0.0f;
        int start = i * stride;
        for (k = 0; k < ker_len; k++)
            sum += signal[start + k] * kernel[k];
        output[i] = sum;
    }
    return out_len;
}

int main(void)
{
    float signal[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int sig_len = 10;
    float kernel[] = { 1, 1, 1 };
    int ker_len = 3;
    float output[10];
    int out_len, i;

    printf("Signal: ");
    for (i = 0; i < sig_len; i++)
        printf("%.0f ", signal[i]);
    printf("\nKernel: 1 1 1 (sum of 3)\n\n");

    out_len = conv1d(signal, sig_len, kernel, 
        ker_len, 1, output);
    printf("Stride 1 (%d outputs): ", out_len);
    for (i = 0; i < out_len; i++)
        printf("%.0f ", output[i]);

    out_len = conv1d(signal, sig_len, kernel, 
        ker_len, 2, output);
    printf("\nStride 2 (%d outputs): ", out_len);
    for (i = 0; i < out_len; i++)
        printf("%.0f ", output[i]);

    out_len = conv1d(signal, sig_len, kernel, 
        ker_len, 3, output);
    printf("\nStride 3 (%d outputs): ", out_len);
    for (i = 0; i < out_len; i++)
        printf("%.0f ", output[i]);

    printf("\n");
    return 0;
}
