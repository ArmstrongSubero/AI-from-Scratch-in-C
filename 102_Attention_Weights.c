/* 102_Attention_Weights.c */
#include <stdio.h>
#include <math.h>
#include <float.h>

#define DIM 3
#define N_KEYS 5

static float dot(const float a[DIM], const float b[DIM])
{
    float sum = 0;
    int i;
    for (i = 0; i < DIM; i++) sum += a[i] * b[i];
    return sum;
}

static void softmax(float *x, int n)
{
    float mx = -FLT_MAX, s = 0;
    int i;
    for (i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    for (i = 0; i < n; i++) x[i] /= s;
}

int main(void)
{
    float query[DIM] = { 1.0f, 0.0f, 0.5f };
    float keys[N_KEYS][DIM] = {
        { 1.0f, 0.0f, 0.5f }, 
        { 0.8f, 0.1f, 0.4f }, 
        { 0.0f, 1.0f, 0.0f }, 
        { -0.5f, 0.2f, -0.3f }, 
        { 0.6f, 0.0f, 0.8f }, 
    };

    /* Compute raw scores */
    float scores[N_KEYS];
    int i;
    for (i = 0; i < N_KEYS; i++)
        scores[i] = dot(query, keys[i]);

    printf("Raw dot-product scores:\n  [");
    for (i = 0; i < N_KEYS; i++)
        printf("%.2f%s", scores[i],
               i < N_KEYS-1 ? ", " : "");
    printf("]\n\n");

    /* Convert to weights via softmax */
    softmax(scores, N_KEYS);

    printf("After softmax (attention weights):\n  [");
    for (i = 0; i < N_KEYS; i++)
        printf("%.3f%s", scores[i],
               i < N_KEYS-1 ? ", " : "");
    printf("]\n\n");

    float sum = 0;
    for (i = 0; i < N_KEYS; i++) sum += scores[i];
    printf("Sum of weights: %.4f\n", sum);
    printf("\nThe most similar key gets the highest "
           "weight.\n");
    printf("All weights are positive and sum to 1.\n");

    return 0;
}
