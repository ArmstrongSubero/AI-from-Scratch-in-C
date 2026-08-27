/* 050_Max_Pooling.c */
#include <stdio.h>

/* 1D max pooling: take the max of each window of
   size pool_size */
static int maxpool1d(const float *input, int in_len, 
                     int pool_size, int stride, 
                     float *output, int *indices)
{
    int out_len = (in_len - pool_size) / stride + 1;
    int i, k;

    for (i = 0; i < out_len; i++) {
        int start = i * stride;
        float max_val = input[start];
        int max_idx = start;

        for (k = 1; k < pool_size; k++) {
            if (input[start + k] > max_val) {
                max_val = input[start + k];
                max_idx = start + k;
            }
        }
        output[i] = max_val;
        indices[i] = max_idx;
        /* save for backward pass */
    }
    return out_len;
}

int main(void)
{
    float input[] = { 1, 3, 2, 4, 6, 1, 3, 5, 2, 0 };
    int in_len = 10;
    float output[5];
    int indices[5];
    int out_len, i;

    /* Pool size 2, stride 2: halves the length */
    out_len = maxpool1d(input, in_len, 2, 2, 
        output, indices);

    printf("Input (%d):  ", in_len);
    for (i = 0; i < in_len; i++)
        printf("%.0f ", input[i]);

    printf("\nPool 2x2 (%d): ", out_len);
    for (i = 0; i < out_len; i++)
        printf("%.0f ", output[i]);

    printf("\nMax index:    ");
    for (i = 0; i < out_len; i++)
        printf("%d ", indices[i]);

    printf("\n\n");

    /* Pool size 3, stride 3 */
    out_len = maxpool1d(input, in_len, 3, 3, 
        output, indices);
    printf("Pool 3x3 (%d): ", out_len);
    for (i = 0; i < out_len; i++)
        printf("%.0f ", output[i]);
    printf("\n");

    return 0;
}
