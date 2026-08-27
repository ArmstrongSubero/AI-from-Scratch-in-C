/* 047_Padding.c */
#include <stdio.h>
#include <string.h>

static int conv1d_padded(const float *signal, 
    int sig_len, 
                          const float *kernel, 
                              int ker_len, 
                          int stride, int pad, 
                              float *output)
{
    int padded_len = sig_len + 2 * pad;
    int out_len = (padded_len - ker_len) / stride + 1;
    int i, k;

    /* We do not actually allocate a padded array.
       Instead, 
      we handle the boundaries in the
      inner loop. */
    for (i = 0; i < out_len; i++) {
        float sum = 0.0f;
        int start = i * stride - pad;
        for (k = 0; k < ker_len; k++) {
            int idx = start + k;
            if (idx >= 0 && idx < sig_len)
                sum += signal[idx] * kernel[k];
            /* else: padded zero, contributes nothing */
        }
        output[i] = sum;
    }
    return out_len;
}

int main(void)
{
    float signal[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    int sig_len = 8;
    float kernel[] = { 1, 1, 1 };
    int ker_len = 3;
    float output[10];
    int out_len, i;

    printf("Signal (len=%d): ", sig_len);
    for (i = 0; i < sig_len; i++)
        printf("%.0f ", signal[i]);

    printf("\n\nNo padding:     ");
    out_len = conv1d_padded(signal, sig_len, kernel, 
        ker_len, 1, 0, output);
    printf("(%d outputs) ", out_len);
    for (i = 0; i < out_len; i++)
        printf("%.0f ", output[i]);

    printf("\nPad=1 ('same'):  ");
    out_len = conv1d_padded(signal, sig_len, kernel, 
        ker_len, 1, 1, output);
    printf("(%d outputs) ", out_len);
    for (i = 0; i < out_len; i++)
        printf("%.0f ", output[i]);

    printf("\n\nWith padding, output length = input "
           "length.\n");
    printf("Edge values are smaller because some "
           "kernel positions\n");
    printf("overlap with the zero-padded region.\n");

    return 0;
}
