/* 083_Cell_Update.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}
static float my_tanh(float z)
{
    if (z < -20) return -1;
    float e = expf(-2 * z);
    return (1 - e) / (1 + e);
}

#define N 3

int main(void)
{
    /* existing cell state */
    float c_old[N] = { 1.0f, -0.5f, 2.0f };

    /* Simulate gate outputs */
    /* forget, keep c[0], erase c[1], keep c[2] */
    float f[N] = { 0.9f, 0.1f, 1.0f };
    /* input, write to c[1], not to c[2] */
    float i_gate[N] = { 0.3f, 0.8f, 0.0f };
    /* candidate values */
    float g[N] = { 0.5f, -0.7f, 0.9f };

    float c_new[N];
    int k;

    for (k = 0; k < N; k++)
        c_new[k] = f[k] * c_old[k] + i_gate[k] * g[k];

    printf("Cell state update: "
           "c_new = f * c_old + i * g\n\n");
    printf("  cell  c_old   f     i     g      "
           "c_new   what happened\n");
    for (k = 0; k < N; k++) {
        printf("  %d     %+5.2f  %.1f   %.1f   "
               "%+.1f   %+5.2f   ",
               k, c_old[k], f[k], i_gate[k], 
               g[k], c_new[k]);
        if (f[k] > 0.8f && i_gate[k] < 0.2f)
            printf("kept old value");
        else if (f[k] < 0.2f && i_gate[k] > 0.5f)
            printf("replaced with new");
        else if (f[k] > 0.8f && i_gate[k] > 0.5f)
            printf("kept old + added new");
        else
            printf("partial update");
        printf("\n");
    }

    printf("\nThe '+' is the gradient highway. "
           "During backprop,\n");
    printf("d(c_new)/d(c_old) = f, which is close "
           "to 1 when\n");
    printf("the forget gate is open. The gradient "
           "flows through\n");
    printf("without vanishing.\n");

    return 0;
}
