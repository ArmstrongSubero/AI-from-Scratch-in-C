/* 075_Gradient_Clipping.c */
#include <stdio.h>
#include <math.h>
#include <string.h>

/* Clip gradient if its norm exceeds threshold */
static float clip_gradient(float *grad, int n, 
                           float threshold)
{
    float norm = 0.0f;
    int i;

    for (i = 0; i < n; i++)
        norm += grad[i] * grad[i];
    norm = sqrtf(norm);

    if (norm > threshold) {
        float scale = threshold / norm;
        for (i = 0; i < n; i++)
            grad[i] *= scale;
    }
    return norm;
}

int main(void)
{
    /* Simulate a gradient vector */
    float grad[] = { 5.0f, -3.0f, 8.0f, -2.0f, 7.0f };
    int n = 5;
    float threshold = 5.0f;
    int i;

    printf("Before clipping:\n  [");
    for (i = 0; i < n; i++)
        printf("%.1f%s", grad[i], i<n-1?", ":"");
    printf("]\n");

    float norm = 0;
    for (i = 0; i < n; i++) norm += grad[i] * grad[i];
    norm = sqrtf(norm);
    printf("  Norm: %.2f\n", norm);

    float new_norm = clip_gradient(grad, n, threshold);

    printf("\nAfter clipping (threshold=%.1f):\n  [",
           threshold);
    for (i = 0; i < n; i++)
        printf("%.2f%s", grad[i], i<n-1?", ":"");
    printf("]\n");

    norm = 0;
    for (i = 0; i < n; i++) norm += grad[i] * grad[i];
    norm = sqrtf(norm);
    printf("  Norm: %.2f\n", norm);

    printf("\nThe direction is preserved, only "
           "the magnitude is capped.\n");
    printf("KANN uses the same approach "
           "(kann_grad_clip).\n");

    return 0;
}
