/* 057_Dense_One_Hot.c */
#include <stdio.h>

/* Dense layer: output[i] = sum(W[i][j] * input[j]) +
   bias[i] */
static void dense(const float *W, const float *bias, 
                  const float *input, float *output, 
                  int out_size, int in_size)
{
    int i, j;
    for (i = 0; i < out_size; i++) {
        float sum = bias[i];
        for (j = 0; j < in_size; j++)
            sum += W[i * in_size + j] * input[j];
        output[i] = sum;
    }
}

int main(void)
{
    /* 5 categories, embedding dimension 3 */
    /* Weight matrix: 3 rows x 5 columns */
    float W[3 * 5] = {
        0.1f, 0.5f, -0.3f, 0.8f, -0.2f,   /* row 0 */
        0.4f, -0.1f, 0.6f, 0.2f, 0.7f,    /* row 1 */
        -0.5f, 0.3f, 0.1f, -0.4f, 0.9f,   /* row 2 */
    };
    float bias[3] = { 0, 0, 0 };

    /* One-hot for category 1 */
    float input[5] = { 0, 1, 0, 0, 0 };
    float output[3];
    int i;

    dense(W, bias, input, output, 3, 5);

    printf("Weight matrix (3x5):\n");
    for (i = 0; i < 3; i++)
        printf("  [%.1f, %.1f, %.1f, %.1f, %.1f]\n",
               W[i*5], W[i*5+1], W[i*5+2], 
                   W[i*5+3], W[i*5+4]);

    printf("\nOne-hot input (category 1): [");
    for (i = 0; i < 5; i++)
        printf("%.0f%s", input[i], i<4?", ":"");
    printf("]\n");

    printf("\nDense output: [%.1f, %.1f, %.1f]\n",
           output[0], output[1], output[2]);

    printf("\nColumn 1 of W: [%.1f, %.1f, %.1f]\n",
           W[1], W[1+5], W[1+10]);

    return 0;
}
