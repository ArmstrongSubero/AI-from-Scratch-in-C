/* 081_Forget_Gate.c */
#include <stdio.h>
#include <math.h>

static float sigmoid(float z)
{
    return 1.0f / (1.0f + expf(-z));
}

#define N 3  /* cell size */

/* f = sigmoid(W_f * x + U_f * h_prev + b_f) */
static void forget_gate(float x, 
                        const float W_f[N], 
                        const float U_f[N][N], 
                        const float b_f[N], 
                        const float h_prev[N], 
                        float f[N])
{
    int i, j;
    for (i = 0; i < N; i++) {
        float z = b_f[i] + W_f[i] * x;
        for (j = 0; j < N; j++)
            z += U_f[i][j] * h_prev[j];
        f[i] = sigmoid(z);
    }
}

/* A gate near 1 keeps, a gate near 0 erases, and
   anything between the two partly keeps */
static const char *label(float g)
{
    if (g > 0.8f) return "keep";
    if (g < 0.3f) return "erase";
    return "partial";
}

int main(void)
{
    /* Bias initialized to 1.0, as KANN does, so the
       gate starts open. The network begins by
       remembering everything and learns what to
       forget. */
    float W_f[N] = { -2.0f, 1.0f, 0.5f };
    float U_f[N][N] = {{ 0.1f, 0, 0 }, 
                       { 0, 0.1f, 0 }, 
                       { 0, 0, 0.1f }};
    /* initialized to 1 */
    float b_f[N] = { 1.0f, 1.0f, 1.0f };

    float h_prev[N] = { 0.5f, 0.3f, 0.8f };
    float inputs[] = { -2.0f, -1.0f, 0.0f, 
                        0.5f, 1.0f, 1.5f };
    int n_inputs = 6;
    float f[N];
    int t, i;

    printf("Forget gate output for different "
           "inputs:\n\n");
    printf("  input   f[0]    f[1]    f[2]  ");
    printf("  cell 0   cell 1   cell 2\n");

    for (t = 0; t < n_inputs; t++) {
        forget_gate(inputs[t], W_f, U_f, b_f, 
                    h_prev, f);
        printf("  %+4.1f    %.3f   %.3f   %.3f  ",
               inputs[t], f[0], f[1], f[2]);
        for (i = 0; i < N; i++)
            printf("  %-7s", label(f[i]));
        printf("\n");
    }

    printf("\nThe bias of 1.0 makes the gate start "
           "open (keep).\n");
    printf("This is important: KANN initializes the "
           "forget gate\n");
    printf("bias to 1.0 "
           "(see Jozefowicz et al, 2015).\n");
    printf("Without this, the LSTM forgets everything "
           "at startup\n");
    printf("and training is much harder.\n");

    return 0;
}
