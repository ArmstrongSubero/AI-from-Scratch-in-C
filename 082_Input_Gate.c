/* 082_Input_Gate.c */
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
    /* Input gate: i = sigmoid(W_i * x + U_i * h +
       b_i) */
    /* Candidate:  g = tanh(W_g * x + U_g * h + b_g) */

    float x = 1.0f;
    float h[N] = { 0.5f, 0.3f, 0.8f };

    /* Simple weights for demonstration */
    float W_i[N] = { 1.0f, 0.5f, -0.5f };
    float b_i[N] = { 0, 0, 0 };
    float W_g[N] = { 0.8f, -0.3f, 0.6f };
    float b_g[N] = { 0, 0, 0 };

    float i_gate[N], g_cand[N], new_info[N];
    int k;

    for (k = 0; k < N; k++) {
        i_gate[k] = sigmoid(W_i[k] * x
                            + 0.1f * h[k] + b_i[k]);
        g_cand[k] = my_tanh(W_g[k] * x
                            + 0.1f * h[k] + b_g[k]);
        new_info[k] = i_gate[k] * g_cand[k];
    }

    printf("Input gate and candidate (x=%.1f):\n\n", x);
    printf("  cell  i_gate  g_cand  i*g (new info)\n");
    for (k = 0; k < N; k++)
        printf("  %d     %.3f   %+.3f   %+.3f\n",
               k, i_gate[k], g_cand[k], new_info[k]);

    printf("\nThe input gate (sigmoid) decides "
           "HOW MUCH to write.\n");
    printf("The candidate (tanh) decides WHAT to "
           "write.\n");
    printf("Their product is the new information "
           "added to the cell.\n");

    return 0;
}
