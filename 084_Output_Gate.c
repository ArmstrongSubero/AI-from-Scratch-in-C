/* 084_Output_Gate.c */
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
    float c[N] = { 1.5f, -0.3f, 2.8f };
    /* cell state */

    /* Output gate,
       o = sigmoid(W_o * x + U_o * h_prev + b_o) */
    float o[N] = { 0.9f, 0.2f, 0.7f };

    float h[N];
    int k;

    for (k = 0; k < N; k++)
        h[k] = o[k] * my_tanh(c[k]);

    printf("Output gate: h = o * tanh(c)\n\n");
    printf("  cell  c       tanh(c)  o      h\n");
    for (k = 0; k < N; k++)
        printf("  %d     %+5.2f   %+5.3f   %.1f   "
               "%+6.3f\n",
               k, c[k], my_tanh(c[k]), o[k], h[k]);

    printf("\nCell 1 has useful information "
           "(c=-0.3) but the output\n");
    printf("gate is nearly closed (o=0.2), so "
           "h[1] is small.\n");
    printf("The LSTM is choosing not to expose "
           "this information yet.\n");
    printf("It might be needed later.\n");

    return 0;
}
