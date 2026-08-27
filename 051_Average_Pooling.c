/* 051_Average_Pooling.c */
#include <stdio.h>

static int avgpool1d(const float *input, int in_len, 
                     int pool_size, int stride, 
                     float *output)
{
    int out_len = (in_len - pool_size) / stride + 1;
    int i, k;

    for (i = 0; i < out_len; i++) {
        float sum = 0.0f;
        int start = i * stride;
        for (k = 0; k < pool_size; k++)
            sum += input[start + k];
        output[i] = sum / pool_size;
    }
    return out_len;
}

int main(void)
{
    float input[] = { 1, 3, 2, 4, 6, 1, 3, 5, 2, 0 };
    int in_len = 10;
    float max_out[5], avg_out[5];
    int i;

    /* Compare max vs average pooling */
    printf("Input:    ");
    for (i = 0; i < in_len; i++)
        printf("%.0f ", input[i]);

    /* Max pool */
    float tmp_out[5];
    int tmp_idx[5];
    int out_len = (in_len - 2) / 2 + 1;
    for (i = 0; i < out_len; i++) {
        int s = i * 2;
        max_out[i] = input[s] > input[s+1]
            ? input[s]
            : input[s+1];
    }
    printf("\nMax pool: ");
    for (i = 0; i < out_len; i++)
        printf("%.0f ", max_out[i]);

    /* Avg pool */
    avgpool1d(input, in_len, 2, 2, avg_out);
    printf("\nAvg pool: ");
    for (i = 0; i < out_len; i++)
        printf("%.1f ", avg_out[i]);

    printf("\n\nMax pool keeps the peaks. Avg pool "
           "smooths everything.\n");
    printf("Max pool is the default for "
           "classification.\n");
    printf("Avg pool is sometimes used as the final "
           "pooling before\n");
    printf("the classifier "
           "(global average pooling).\n");

    return 0;
}
