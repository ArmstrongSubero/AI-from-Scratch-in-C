/* 023_Softmax.c */
#include <stdio.h>
#include <math.h>
#include <float.h>

static void softmax(const float *z, float *out, int n)
{
    float max_val, sum;
    int i;

    /* Find max for numerical stability */
    max_val = -FLT_MAX;
    for (i = 0; i < n; i++)
        if (z[i] > max_val) max_val = z[i];

    /* Compute exp(z_i - max) and sum */
    sum = 0.0f;
    for (i = 0; i < n; i++) {
        out[i] = expf(z[i] - max_val);
        sum += out[i];
    }

    /* Normalize */
    for (i = 0; i < n; i++)
        out[i] /= sum;
}

int main(void)
{
    /* Raw logits from a 4-class network */
    float logits[] = { 2.0f, 1.0f, 0.1f, -1.0f };
    float probs[4];
    float sum = 0.0f;
    int i;

    softmax(logits, probs, 4);

    printf("Logits -> Softmax probabilities:\n");
    for (i = 0; i < 4; i++) {
        printf("  class %d: logit=%5.1f  prob=%.4f\n",
               i, logits[i], probs[i]);
        sum += probs[i];
    }
    printf("  Sum of probabilities: %.6f\n", sum);

    /* Test with extreme values */
    printf("\nExtreme logits (stability test):\n");
    float extreme[] = { 500.0f, 499.0f, 498.0f, 0.0f };
    softmax(extreme, probs, 4);
    for (i = 0; i < 4; i++)
        printf("  class %d: logit=%5.0f  prob=%.4f\n",
               i, extreme[i], probs[i]);

    return 0;
}
