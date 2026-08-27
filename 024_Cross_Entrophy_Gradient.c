/* 024_Cross_Entrophy_Gradient.c */
#include <stdio.h>
#include <math.h>
#include <float.h>

static void softmax(const float *z, float *out, int n)
{
    float max_val = -FLT_MAX, sum = 0.0f;
    int i;
    for (i = 0; i < n; i++)
        if (z[i] > max_val) max_val = z[i];
    for (i = 0; i < n; i++) {
        out[i] = expf(z[i] - max_val);
        sum += out[i];
    }
    for (i = 0; i < n; i++)
        out[i] /= sum;
}

static void softmax_ce_gradient(const float *probs,
                                int target,
                                 float *grad, int n)
{
    int i;
    for (i = 0; i < n; i++)
        grad[i] = probs[i];
    grad[target] -= 1.0f;   /* p_i - t_i, one-hot */
}

int main(void)
{
    float logits[] = { 2.0f, 1.0f, 0.1f, -1.0f };
    float probs[4], grad[4];
    int target = 0;
    int i;

    softmax(logits, probs, 4);
    softmax_ce_gradient(probs, target, grad, 4);

    printf("Target: class %d\n\n", target);
    printf("  class  logit   prob    gradient\n");
    for (i = 0; i < 4; i++)
        printf("  %d      %5.1f   %.4f  %+.4f\n",
               i, logits[i], probs[i], grad[i]);

    printf("\nInterpretation:\n");
    printf("  Class 0 (correct): gradient is"
           " negative -> push logit UP\n");
    printf("  Classes 1-3 (wrong): gradient is"
           " positive -> push logits DOWN\n");

    return 0;
}
