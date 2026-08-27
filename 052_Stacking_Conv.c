/* 052_Stacking_Conv.c */
#include <stdio.h>
#include <math.h>

static float relu(float z)
{
    return z > 0 ? z : 0;
}

/* Simplified conv1d: single input channel, single
   output channel */
static void conv1d_simple(const float *in, int in_len, 
                           const float *ker, 
                               int ker_len, 
                           float bias, float *out, 
                               int *out_len)
{
    *out_len = in_len - ker_len + 1;
    int i, k;
    for (i = 0; i < *out_len; i++) {
        float sum = bias;
        for (k = 0; k < ker_len; k++)
            sum += in[i + k] * ker[k];
        out[i] = relu(sum);
    }
}

static void maxpool(const float *in, int in_len, 
                    float *out, int *out_len)
{
    *out_len = in_len / 2;
    int i;
    for (i = 0; i < *out_len; i++) {
        float a = in[i * 2], b = in[i * 2 + 1];
        out[i] = a > b ? a : b;
    }
}

int main(void)
{
    /* Input signal: two spikes */
    float input[16] = { 0, 0, 1, 3, 1, 0, 
        0, 0, 0, 0, 0, 
        1, 3, 1, 0, 0 };

    /* Layer 1: spike detector */
    float ker1[] = { -1, 2, -1 };
    float l1_out[16];
    int l1_len;
    conv1d_simple(input, 16, ker1, 3, 0, 
        l1_out, &l1_len);

    /* Pool 1 */
    float p1_out[16];
    int p1_len;
    maxpool(l1_out, l1_len, p1_out, &p1_len);

    /* Layer 2: detects patterns in the pooled spike
       map */
    float ker2[] = { 1, -1, 1 };
    float l2_out[16];
    int l2_len;
    conv1d_simple(p1_out, p1_len, ker2, 3, 0, 
        l2_out, &l2_len);

    int i;
    printf("Input (%d):         ", 16);
    for (i = 0; i < 16; i++) printf("%.0f ", input[i]);

    printf("\nConv1+ReLU (%d):     ", l1_len);
    for (i = 0; i < l1_len; i++)
        printf("%.0f ", l1_out[i]);

    printf("\nMaxPool (%d):         ", p1_len);
    for (i = 0; i < p1_len; i++)
        printf("%.0f ", p1_out[i]);

    printf("\nConv2+ReLU (%d):      ", l2_len);
    for (i = 0; i < l2_len; i++)
        printf("%.0f ", l2_out[i]);

    printf("\n\nThe input had 16 values. After two "
           "conv+pool stages\n");
    printf("we are down to %d values. Each value "
           "summarizes a\n", l2_len);
    printf("larger region of the original input.\n");

    return 0;
}
