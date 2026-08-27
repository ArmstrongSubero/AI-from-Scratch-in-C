/* 036_Dropout.c */
#include <stdio.h>
#include <stdlib.h>

static float randf(void)
{
    return (float)rand() / RAND_MAX;
}

/* Inverted dropout: zero out random neurons, scale
   survivors up.
   mask[] stores which neurons were kept (1)
   or dropped (0).
   During inference, skip this function entirely. */

static void dropout(float *h, int *mask, 
    int n, float rate)
{
    float scale = 1.0f / (1.0f - rate);
    int i;

    for (i = 0; i < n; i++) {
        if (randf() < rate) {
            h[i] = 0.0f;
            mask[i] = 0;
        }
        else {
            h[i] *= scale;
            mask[i] = 1;
        }
    }
}

int main(void)
{
    float h[10] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 
                    6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
    int mask[10];
    float rate = 0.3f;
    int i;

    srand(42);

    printf("Before dropout:\n  ");
    for (i = 0; i < 10; i++) printf("%.1f ", h[i]);

    dropout(h, mask, 10, rate);

    printf("\n\nAfter dropout (rate=%.1f):\n  ", rate);
    for (i = 0; i < 10; i++) printf("%.1f ", h[i]);

    printf("\n\nMask:\n  ");
    for (i = 0; i < 10; i++) printf("%d ", mask[i]);

    /* Verify: expected value is preserved */
    float sum_before = 55.0f;  /* sum of 1..10 */
    float sum_after = 0.0f;
    for (i = 0; i < 10; i++) sum_after += h[i];
    printf("\n\nSum before: %.1f", sum_before);
    printf("\nSum after:  %.1f (varies by run, but "
           "expected value ~%.1f)\n",
           sum_after, sum_before);

    return 0;
}
